"""Train matched internal-module ablations and evaluate full OOF SGLA tracks."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
for folder in (REPO, REPO / "tracking", REPO / "tools"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import numpy as np

import stage194_module_common as protocol
import stage194_module_model as modules
from experiment_progress import OverallProgress
from stage18_environment import executed_crop_roi
from stage27_uav_collect import writer_lock
from stage27_uav_common import failure_events, publish_identical, source_group, success_auc, tracking_iou
from stage4_common import file_hash, fingerprint, write_cache, write_csv

SCHEMA, DESIGN = protocol.SCHEMA, protocol.DESIGN
TABLES = ("policy_summary", "paired", "factor_summary", "seed_video_summary", "scope_summary",
          "smoke_summary", "fit_support", "fixed_layers", "sampling", "routing", "readout_summary")


def sealed(body):
    # Metadata is deliberately JSON-native; numpy scalars must be converted at
    # construction rather than silently stringified by a custom JSON encoder.
    return dict(body, fingerprint=fingerprint(body))


def read_sealed(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: v for k, v in value.items() if k != "fingerprint"}
    if value.get("fingerprint") != fingerprint(body):
        raise ValueError("Changed sealed artifact: " + str(path))
    return value


def cache(path, metadata, compute):
    path = Path(path)
    receipt = path.with_suffix(".receipt.json")
    if receipt.exists():
        r = read_sealed(receipt)
        if r["metadata"] != metadata or file_hash(path) != r["sha256"]:
            raise ValueError("Changed committed cache: " + str(path))
        with np.load(path, allow_pickle=False) as f:
            if json.loads(str(f["metadata_json"].item())) != metadata:
                raise ValueError("Changed NPZ metadata")
            return {k: f[k] for k in f.files if k != "metadata_json"}, True
    data = compute()
    if path.exists():
        # A crash between atomic data publication and receipt is recoverable
        # only if the orphan has the same sealed metadata.
        with np.load(path, allow_pickle=False) as f:
            if json.loads(str(f["metadata_json"].item())) != metadata:
                raise ValueError("Unidentified orphan cache: " + str(path))
    write_cache(path, data, metadata)
    publish_identical(receipt, sealed(dict(schema=SCHEMA, metadata=metadata, sha256=file_hash(path))))
    return data, False


def record(root, path, artifacts):
    for item in (path, path.with_suffix(".receipt.json")):
        artifacts[item.relative_to(root).as_posix()] = file_hash(item)


def reused_work(progress, count):
    progress.reused += count
    progress.advance(count=count)


def model_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_model(torch, path, payload, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = path.with_suffix(".receipt.json")
    if receipt.exists():
        raise ValueError("Never overwrite a committed training artifact")
    if path.exists():
        orphan = torch.load(path, map_location="cpu", weights_only=True)
        if orphan.get("metadata") != metadata:
            raise ValueError("Unidentified orphan model")
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
        tmp = Path(f.name)
    try:
        torch.save(dict(payload, metadata=metadata), tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    publish_identical(receipt, sealed(dict(schema=SCHEMA, metadata=metadata, sha256=file_hash(path))))


def load_model(torch, path, metadata):
    if not path.with_suffix(".receipt.json").exists():
        return None
    receipt = read_sealed(path.with_suffix(".receipt.json"))
    if receipt["metadata"] != metadata or file_hash(path) != receipt["sha256"]:
        raise ValueError("Changed committed model: " + str(path))
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["metadata"] != metadata:
        raise ValueError("Changed model identity")
    return payload


def verify_results(root):
    decision = read_sealed(root / "stage194_decision.json")
    plan = read_sealed(root / "manifest.json")
    if (decision["schema"] != SCHEMA or decision["plan_fingerprint"] != plan["fingerprint"]
            or decision["status"] != "REVIEW_MATCHED_MODULE_ABLATION_PILOT"
            or decision["confirmation_groups_opened"] != 0):
        raise ValueError("Incomplete Stage-194 decision")
    for name, digest in (decision["table_sha256"] | decision["artifact_sha256"]).items():
        if file_hash(root / name) != digest:
            raise ValueError("Changed Stage-194 result: " + name)
    if set(decision["table_sha256"]) != {f"stage194_{k}.csv" for k in TABLES}:
        raise ValueError("Incomplete result tables")
    expected = len(plan["entries"]) * (len(protocol.CONTROLS) + len(protocol.TRAINED) * len(DESIGN["seeds"]))
    if decision["full_trajectories"] != expected or decision["training_runs"] != 36:
        raise ValueError("Incomplete full OOF coverage")
    print("Stage-194 artifacts verified; REVIEW is a pilot outcome, not method adoption", flush=True)


def collect_features(root, plan, entries, dataset, native, progress, artifacts):
    rt, bank, rows = native.rt, {}, []
    for spec in plan["entries"]:
        seq, group = spec["sequence"], spec["group"]
        files, gt, trace, reader, _ = entries[seq]
        frames = np.asarray(spec["sample_frames"], int)
        path = root / "features" / (seq + ".npz")
        meta = dict(plan_fingerprint=plan["fingerprint"], sequence=seq, kind="native_parent_prefix")

        def compute():
            native.initialize(reader(dataset, files[0]), gt[0])
            stored = []
            for frame in frames:
                parent = np.asarray(trace["parent"][frame - 1], float)
                image = reader(dataset, files[frame])
                captured = []
                actual = []
                namespace = rt.tracker.track.__func__.__globals__
                sample = namespace["sample_target"]

                def inspect(*args, **kwargs):
                    value = sample(*args, **kwargs)
                    actual.append(float(value[1]))
                    return value

                rt.tracker.frame_id = int(frame) - 1
                with ExitStack() as stack:
                    stack.callback(rt.backbone.blocks[5].register_forward_hook(
                        lambda module, args, output: captured.append(output.detach().cpu().numpy()[0].copy())).remove)
                    stack.enter_context(patch.dict(namespace, sample_target=inspect))
                    result = native.step(image, parent)
                if len(captured) != 1 or len(actual) != 1 or rt.tracker.frame_id != frame:
                    raise ValueError("Expected one original prefix and one executed crop")
                factor = actual[0]
                target = executed_crop_roi(gt[frame], parent, factor, rt.tracker.params.search_size)
                stored.append(dict(tokens=captured[0], target=target, parent=parent, factor=factor,
                                   image_hw=image.shape[:2], truth=gt[frame], native_box=result["box"],
                                   native_layer=result["layer"]))
                progress.advance()
            return dict(frame_id=frames, **{k: np.asarray([r[k] for r in stored]) for k in stored[0]})

        data, reused = cache(path, meta, compute)
        if reused:
            reused_work(progress, len(frames))
        np.testing.assert_array_equal(data["frame_id"], frames)
        shape = (len(frames), rt.lens_z + rt.lens_x, rt.identity["embed_dim"])
        if data["tokens"].shape != shape or not all(np.isfinite(v).all() for v in data.values()):
            raise ValueError("Invalid native-parent feature cache")
        # Native forward and manual frozen prefix/head must agree before any fit.
        with rt.torch.no_grad():
            token = rt.torch.as_tensor(data["tokens"][:1], device=next(rt.net.parameters()).device)
            output = rt.candidate(token, int(data["native_layer"][0]))
            boxes = protocol.dense_image_boxes(output["size_map"][0].cpu().numpy(),
                output["offset_map"][0].cpu().numpy(), data["parent"][0], data["factor"][0],
                rt.tracker.params.search_size, data["image_hw"][0])
            cell = int((output["score_map"] * rt.tracker.output_window).flatten(1).argmax(1).item())
            error = float(np.abs(boxes[cell] - data["native_box"][0]).max())
            if error > .001:
                raise ValueError(f"Native dense decode parity failed at {seq}: {error}")
        bank[group] = data
        rows.append(dict(sequence=seq, group=group, views=len(frames),
                         off_crop_centers=int(((data["target"][:, :2] + data["target"][:, 2:] / 2 < 0)
                             | (data["target"][:, :2] + data["target"][:, 2:] / 2 >= 1)).any(1).sum()),
                         dense_decode_error=error, feature_sha256=file_hash(path)))
        record(root, path, artifacts)
    native.assert_frozen()
    return bank, rows


def fit(root, plan, native, bank, fold, seed, arm, progress, artifacts, smoke=False):
    rt, torch = native.rt, native.rt.torch
    device = next(rt.net.parameters()).device
    model = modules.build_modules(torch, rt.identity["embed_dim"], rt.lens_z, arm).to(device).eval()
    params = [p for p in model.parameters() if p.requires_grad]
    draws = protocol.draws_for(fold, seed, {g: len(d["frame_id"]) for g, d in bank.items()})
    updates = DESIGN["smoke_updates"] if smoke else DESIGN["updates"]
    mode = "smoke" if smoke else "training"
    path = root / mode / f"fold{fold['fold']}" / f"seed{seed}" / (arm + ".pt")
    metadata = dict(plan_fingerprint=plan["fingerprint"], fold=fold["fold"], seed=seed, arm=arm,
                    fit=fold["fit"], held=fold["held"], updates=updates, draws_sha256=fingerprint(draws[:updates]),
                    features={g: artifacts["features/" + next(e["sequence"] for e in plan["entries"] if e["group"] == g) + ".npz"]
                              for g in fold["fit"]})
    saved = load_model(torch, path, metadata)
    if saved is not None:
        model.load_state_dict(saved["state"], strict=True)
        reused_work(progress, updates)
        record(root, path, artifacts)
        return model.requires_grad_(False), saved["scope"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    optimizer = torch.optim.AdamW(params, lr=DESIGN["learning_rate"], weight_decay=DESIGN["weight_decay"])
    initial = model_digest(model)
    maxima = dict(adapter=0., router=0., quality=0.)
    history = []
    start = perf_counter()
    for step, batch in enumerate(draws[:updates], 1):
        if any(g not in fold["fit"] or g in fold["held"] for g, _ in batch):
            raise ValueError("Held-source training leakage")
        tensor = lambda k: torch.as_tensor(np.stack([bank[g][k][i] for g, i in batch]), device=device,
                        dtype=torch.float64 if k in ("parent", "factor", "image_hw", "truth") else torch.float32)
        tokens = tensor("tokens")
        if step == 1:
            with torch.no_grad():
                if not torch.equal(model.adapt(tokens), tokens):
                    raise ValueError("Adapter must initialize to the exact native function")
                raw = torch.linspace(.0001, .9999, rt.lens_x, device=device).reshape(1, 1, rt.search_side, rt.search_side)
                feature = tokens[:, -rt.lens_x:].transpose(1, 2).reshape(-1, tokens.shape[-1], rt.search_side, rt.search_side)
                if float((model.score(feature, raw) - raw).abs().max()) > 1e-6:
                    raise ValueError("Initial quality residual changes the native map")
        optimizer.zero_grad(set_to_none=True)
        loss, stats = modules.train_objective(rt, model, arm, tokens, tensor("target"), tensor("parent"),
                                              tensor("factor"), tensor("image_hw"), tensor("truth"))
        loss.backward()
        # Losses own disjoint trainable parameter sets. Clip each module
        # separately so activating another loss cannot rescale its gradient.
        norms = [float(torch.nn.utils.clip_grad_norm_(getattr(model, name).parameters(),
                    DESIGN["gradient_clip"], error_if_nonfinite=True))
                 for name, active in zip(("adapter", "router", "quality"), protocol.flags(arm)) if active]
        norm = max(norms)
        if not np.isfinite([float(loss.detach()), norm]).all() or any(p.grad is not None for p in rt.net.parameters()):
            raise ValueError("Invalid training loss or gradient reached original SGLA weights")
        for name in maxima:
            own = getattr(model, name)
            maxima[name] = max(maxima[name], max((float(p.grad.abs().max()) for p in own.parameters() if p.grad is not None), default=0.))
        optimizer.step()
        history.append(dict(step=step, total=float(loss.detach()), **stats))
        progress.advance()
    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    if any((maxima[k] > 0) != bool(v) for k, v in zip(maxima, protocol.flags(arm))):
        raise ValueError("Active/inactive module gradient scope failed: " + arm)
    if model_digest(model) == initial:
        raise ValueError("No learned module changed")
    native.assert_frozen()
    scope = dict(fold=fold["fold"], seed=seed, arm=arm, updates=updates,
                 examples=updates * DESIGN["batch"], candidate_block_calls=updates * 6,
                 candidate_examples=updates * DESIGN["batch"] * 6,
                 trainable_parameters=sum(p.numel() for p in params), training_seconds=elapsed,
                 fit_groups="|".join(fold["fit"]), held_groups="|".join(fold["held"]),
                 draws_sha256=metadata["draws_sha256"], frozen_native_unchanged=True,
                 maximum_adapter_gradient=maxima["adapter"], maximum_router_gradient=maxima["router"],
                 maximum_quality_gradient=maxima["quality"], first_loss=history[0]["total"], final_loss=history[-1]["total"],
                 state_sha256=model_digest(model))
    save_model(torch, path, dict(state={k: v.detach().cpu() for k, v in model.state_dict().items()}, scope=scope, history=history), metadata)
    record(root, path, artifacts)
    return model.requires_grad_(False), scope


def reference_quality(root, plan, native, bank, progress, artifacts):
    rt, labels = native.rt, {}
    for group, data in bank.items():
        path = root / "reference" / (group + ".npz")
        meta = dict(plan_fingerprint=plan["fingerprint"], kind="fixed_layer_fit_labels", group=group)

        def compute():
            values = []
            with rt.torch.no_grad():
                for i, array in enumerate(data["tokens"]):
                    token = rt.torch.as_tensor(array[None], device=next(rt.net.parameters()).device)
                    row = []
                    for layer in range(7, 13):
                        out = rt.candidate(token, layer)
                        grid = protocol.dense_image_boxes(out["size_map"][0].cpu().numpy(), out["offset_map"][0].cpu().numpy(),
                            data["parent"][i], data["factor"][i], rt.tracker.params.search_size, data["image_hw"][i])
                        cell = int((out["score_map"] * rt.tracker.output_window).flatten(1).argmax(1).item())
                        row.append(float(tracking_iou(grid[cell], data["truth"][i])))
                    values.append(row)
                    progress.advance()
            return dict(iou=np.asarray(values))

        value, reused = cache(path, meta, compute)
        if reused:
            reused_work(progress, len(data["frame_id"]))
        if value["iou"].shape != (len(data["frame_id"]), 6) or not np.isfinite(value["iou"]).all():
            raise ValueError("Invalid fixed-layer labels")
        labels[group] = value["iou"]
        record(root, path, artifacts)
    return labels


def full_trace(native, model, arm, layer, files, initial, reader, dataset, progress):
    rt, rows = native.rt, []
    native.initialize(reader(dataset, files[0]), initial)
    parent = np.asarray(initial, float)
    with modules.InstalledModules(native, model, arm, layer):
        for frame, file in enumerate(files[1:], 1):
            start = perf_counter()
            image = reader(dataset, file)
            namespace = rt.tracker.track.__func__.__globals__
            sample = namespace["sample_target"]
            crops = 0

            def counted(*args, **kwargs):
                nonlocal crops
                crops += 1
                return sample(*args, **kwargs)

            with patch.dict(namespace, sample_target=counted):
                out = native.step(image, parent)
            elapsed = 1000 * (perf_counter() - start)
            if crops != 1 or rt.tracker.frame_id != frame:
                raise ValueError("Full tracking must make one search per frame")
            rows.append(dict(frame_id=frame, parent=parent.copy(), boxes=out["box"], layer=out["layer"],
                             block_calls=out["block_calls"], head_calls=out["head_calls"],
                             selector_calls=out["selector_calls"], crop_calls=crops, pipeline_ms=elapsed))
            parent = out["box"]
            progress.advance()
    return {k: np.asarray([r[k] for r in rows]) for k in rows[0]}


def audit_trace(trace, initial, frames):
    np.testing.assert_array_equal(trace["frame_id"], np.arange(1, frames + 1))
    np.testing.assert_array_equal(trace["parent"], np.vstack((initial, trace["boxes"][:-1])))
    if trace["boxes"].shape != (frames, 4) or not np.isfinite(trace["boxes"]).all() or (trace["boxes"][:, 2:] <= 0).any():
        raise ValueError("Invalid full-video predictions")
    if not np.isin(trace["layer"], range(7, 13)).all():
        raise ValueError("Unexpected execution layer")
    calls = np.zeros((frames, 12), int)
    calls[:, :6] = 1
    calls[np.arange(frames), trace["layer"] - 1] = 1
    np.testing.assert_array_equal(trace["block_calls"], calls)
    for key in ("head_calls", "selector_calls", "crop_calls"):
        np.testing.assert_array_equal(trace[key], np.ones(frames, int))


def describe_trace(trace, baseline, gt, seq, arm, seed, fold):
    values, control = tracking_iou(trace["boxes"], gt[1:]), tracking_iou(baseline["boxes"], gt[1:])
    events, ref_events = failure_events(values), failure_events(control)
    lost = 0
    for event in ref_events:
        if event["recovery"] is None:
            continue
        recovery = event["recovery"]
        # Use the same three-frame recovery definition at the recorded time;
        # no newly tuned grace period is introduced for the ablation.
        lost += not bool((values[recovery:recovery + 3] >= .5).all())
    stable = control >= .5
    return dict(sequence=seq, group=source_group(seq), arm=arm, seed=seed, fold=fold,
                auc=success_auc(values), mean_iou=float(values.mean()), low_iou_fraction=float((values < .2).mean()),
                failure_events=len(events), longest_failure_fraction=max((e["end"] - e["start"] + 1 for e in events), default=0) / len(values),
                unrecovered_events=sum(e["recovery"] is None for e in events), missed_native_recoveries=int(lost),
                introduced_failure=bool(not ref_events and events),
                native_good_frame_loss=float(np.maximum(control[stable] - values[stable], 0).mean()) if stable.any() else 0.,
                median_pipeline_ms=float(np.median(trace["pipeline_ms"])))


def readout(root, plan, native, model, arm, seq, seed, fold, data, progress, artifacts):
    """Held-source labels are attached after prediction, never used to fit."""
    rt, torch = native.rt, native.rt.torch
    path = root / "readout" / f"{seq}__s{seed}__{arm}.npz"
    meta = dict(plan_fingerprint=plan["fingerprint"], sequence=seq, seed=seed, arm=arm, fold=fold)

    def compute():
        rows = []
        with torch.no_grad():
            for i, array in enumerate(data["tokens"]):
                token = torch.as_tensor(array[None], device=next(rt.net.parameters()).device)
                encoded = model.adapt(token)
                scores = model.route(encoded) if protocol.flags(arm)[1] else rt.selector(encoded)
                chosen = rt.selected_layer(scores) - 7
                choices, best = [], []
                for layer in range(7, 13):
                    feature = rt.backbone.norm(rt.backbone.blocks[layer - 1](encoded))
                    out = rt.net.forward_head(feature)
                    search = feature[:, -rt.lens_x:].transpose(1, 2).reshape(1, feature.shape[-1], rt.search_side, rt.search_side)
                    score = model.score(search, out["score_map"])
                    cell = int((score * rt.tracker.output_window).flatten(1).argmax(1).item())
                    grid = protocol.dense_image_boxes(out["size_map"][0].cpu().numpy(), out["offset_map"][0].cpu().numpy(),
                        data["parent"][i], data["factor"][i], rt.tracker.params.search_size, data["image_hw"][i])
                    iou = tracking_iou(grid, data["truth"][i])
                    choices.append(float(iou[cell]))
                    best.append(float(iou.max()))
                rows.append([choices[chosen], max(choices), best[chosen], max(best), chosen + 7])
                progress.advance()
        return dict(values=np.asarray(rows))

    saved, reused = cache(path, meta, compute)
    if reused:
        reused_work(progress, len(data["frame_id"]))
    value = saved["values"]
    if value.shape != (len(data["frame_id"]), 5) or not np.isfinite(value).all():
        raise ValueError("Invalid held-source readout")
    record(root, path, artifacts)
    return dict(sequence=seq, group=source_group(seq), fold=fold, seed=seed, arm=arm, views=len(value),
                selected_mean_iou=float(value[:, 0].mean()), best_branch_mean_iou=float(value[:, 1].mean()),
                chosen_grid_best_mean_iou=float(value[:, 2].mean()), all_grid_best_mean_iou=float(value[:, 3].mean()),
                branch_regret=float((value[:, 1] - value[:, 0]).mean()),
                chosen_grid_good=int((value[:, 2] >= .5).sum()), selected_good=int((value[:, 0] >= .5).sum()),
                good_grid_misses=int(((value[:, 2] >= .5) & (value[:, 0] < .5)).sum()))


def runtime_smoke(root, plan, entries, dataset, native, progress):
    rt, torch = native.rt, native.rt.torch
    seq = plan["entries"][0]["sequence"]
    files, gt, _, reader, _ = entries[seq]
    image = reader(dataset, files[1])
    native.initialize(reader(dataset, files[0]), gt[0])
    original = native.step(image, gt[0])
    progress.advance()
    model = modules.build_modules(torch, rt.identity["embed_dim"], rt.lens_z, "a1_r0_q1").to(next(rt.net.parameters()).device).eval()
    native.initialize(reader(dataset, files[0]), gt[0])
    with modules.InstalledModules(native, model, "a1_r0_q1"):
        adapted = native.step(image, gt[0])
    progress.advance()
    error = float(np.abs(original["box"] - adapted["box"]).max())
    if error > .001 or original["layer"] != adapted["layer"]:
        raise ValueError("Zero-residual public-path smoke failed")
    native.assert_frozen()
    publish_identical(root / "smoke_identity.json", sealed(dict(schema=SCHEMA,
        plan_fingerprint=plan["fingerprint"], zero_residual_box_error=error,
        unchanged_frozen_native=True, source_sequence=seq)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for stage in (76, 89, 93, 121, 170, 188, 191):
        parser.add_argument(f"--stage{stage}-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage194_module_ablation")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        verify_results(root)
        return
    if args.checkpoint is None or any(getattr(args, f"stage{k}_root") is None for k in (76, 89, 93, 121, 170, 188, 191)):
        parser.error("All source roots and the original checkpoint are required")
    import stage192_development_grid_rescue as prior

    paths = {k: getattr(args, f"stage{k}_root").expanduser().resolve() for k in prior.SOURCES}
    protocol.safe_output(root, [*paths.values(), args.checkpoint])
    with writer_lock(root):
        if (root / "stage194_decision.json").exists():
            verify_results(root)
            return
        entries, dataset, expected = prior.load_sources(paths)
        protocol.safe_output(root, [dataset])
        roster = [dict(sequence=seq, group=source_group(seq), cohort=entry[4],
                       tracked_frames=len(entry[0]) - 1, sample_frames=protocol.uniform_frames(len(entry[0]) - 1).tolist(),
                       truth_sha256=fingerprint(np.asarray(entry[1]).tolist()), frames_sha256=fingerprint(entry[0]))
                  for seq, entry in sorted(entries.items())]
        body = dict(schema=SCHEMA, design=DESIGN, entries=roster, folds=protocol.split_groups([e["group"] for e in roster]),
                    arms=list(protocol.ARMS), checkpoint_sha256=file_hash(args.checkpoint),
                    source_manifests={str(k): file_hash(p / "manifest.json") for k, p in paths.items()},
                    source_runtime=expected, confirmation_groups_opened=0,
                    scope="Historical development OOF pilot; no temporal-memory claim, no SOTA claim",
                    training_state_source="uniform native-parent replay; no on-policy data aggregation in this pilot",
                    routing_target="IoU of actual selected clipped box for each candidate branch",
                    geometry_target="original 2*GIoU+5*L1+focal at GT peak, averaged across six branches",
                    quality_target="stop-gradient IoU of each actual clipped dense box",
                    code_sha256={str(Path(p).resolve().relative_to(REPO).as_posix()): file_hash(p)
                        for p in (__file__, protocol.__file__, modules.__file__, REPO / "tools/experiment_progress.py",
                                  REPO / "tracking/stage18_tracking_loss.py", REPO / "tracking/stage18_environment.py",
                                  REPO / "tracking/stage46_motion_runtime.py", REPO / "tools/stage192_development_grid_rescue.py",
                                  REPO / "tracking/stage4_runtime.py", REPO / "tracking/stage1_validate.py",
                                  REPO / "tracking/validation_metrics.py", REPO / "lib/models/sglatrack/base_backbone.py",
                                  REPO / "lib/models/sglatrack/sglatrack.py", REPO / "lib/models/layers/head.py",
                                  REPO / "lib/test/tracker/sglatrack.py", REPO / "lib/train/data/processing_utils.py",
                                  REPO / "tools/stage61_runtime_replay.py")})
        # Prepare binds data and protocol but intentionally does not load torch,
        # images or a GPU. The actual run separately seals its current runtime.
        prepare = sealed(body)
        publish_identical(root / "protocol.json", prepare)
        if args.prepare_only:
            print(f"Stage-194 protocol fixed: 38 groups, 2 folds, 2 seeds, 36 fits; {root}", flush=True)
            return
        import torch

        torch.set_num_threads(1)
        runtime = prior.old.base.s103.s76.device_runtime(dict(runtime_identity=expected), torch, args.gpu)
        differences = prior.old.base.s109.check_runtime(expected, runtime)
        plan = sealed(dict(body, protocol_fingerprint=prepare["fingerprint"], runtime=runtime, runtime_differences=differences))
        publish_identical(root / "manifest.json", plan)
        args.parameter = runtime["tracker"]["parameter"]
        native, _ = prior.old.base.s103.s76.startup.checked_runtime(args,
            dict(fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"]), root)
        rt = native.rt
        if rt.backbone.cat_mode != "direct" or rt.lens_z != 64 or rt.lens_x != 256:
            raise ValueError("Need the audited direct template/search token layout")
        views = sum(len(e["sample_frames"]) for e in roster)
        frames = sum(e["tracked_frames"] for e in roster)
        total = 2 + views * 2 + len(protocol.TRAINED) * DESIGN["smoke_updates"]
        total += len(protocol.TRAINED) * len(DESIGN["seeds"]) * (DESIGN["folds"] * DESIGN["updates"] + views + frames)
        total += len(protocol.CONTROLS) * frames + 1
        progress = OverallProgress("Stage-194", total, unit="work")
        artifacts, tables = {}, {}
        progress.report("Public tracking identity smoke")
        runtime_smoke(root, plan, entries, dataset, native, progress)
        artifacts["smoke_identity.json"] = file_hash(root / "smoke_identity.json")
        progress.report("Collecting uniform native-prefix states")
        bank, tables["fit_support"] = collect_features(root, plan, entries, dataset, native, progress, artifacts)
        tables["sampling"] = [dict(sequence=e["sequence"], group=e["group"], frame=f) for e in roster for f in e["sample_frames"]]
        progress.report("Gradient and frozen-scope smoke")
        tables["smoke_summary"] = []
        for arm in protocol.TRAINED:
            _, scope = fit(root, plan, native, bank, plan["folds"][0], DESIGN["seeds"][0], arm, progress, artifacts, smoke=True)
            tables["smoke_summary"].append(scope)
        progress.report("Same-state fixed-layer reference labels")
        reference = reference_quality(root, plan, native, bank, progress, artifacts)
        fixed, tables["fixed_layers"] = {}, []
        for fold in plan["folds"]:
            mean = np.stack([reference[g].mean(0) for g in fold["fit"]]).mean(0)
            best = int(np.argmax(mean)) + 7
            fixed[fold["fold"]] = best
            tables["fixed_layers"].append(dict(fold=fold["fold"], layer=best, fit="|".join(fold["fit"]), held="|".join(fold["held"]),
                                               **{f"L{k}": float(v) for k, v in zip(range(7, 13), mean)}))
        models, tables["scope_summary"] = {}, []
        progress.report("Matched training of internal modules")
        for fold in plan["folds"]:
            for seed in DESIGN["seeds"]:
                for arm in protocol.TRAINED:
                    model, scope = fit(root, plan, native, bank, fold, seed, arm, progress, artifacts)
                    models[fold["fold"], seed, arm] = model
                    tables["scope_summary"].append(scope)
        progress.report("Full OOF videos and held-source readout")
        held = {g: f["fold"] for f in plan["folds"] for g in f["held"]}
        rows, routes, readouts = [], [], []
        for spec in roster:
            seq, group = spec["sequence"], spec["group"]
            files, gt, _, reader, _ = entries[seq]
            fold = held[group]
            jobs = [(a, 0, None, 9 if a == "fixed_L9" else fixed[fold] if a == "fit_best_fixed" else None)
                    for a in protocol.CONTROLS]
            jobs += [(a, s, models[fold, s, a], None) for s in DESIGN["seeds"] for a in protocol.TRAINED]
            baseline = None
            for arm, seed, model, layer in jobs:
                path = root / "evaluation" / f"{seq}__s{seed}__{arm}.npz"
                meta = dict(plan_fingerprint=plan["fingerprint"], sequence=seq, fold=fold, seed=seed, arm=arm,
                            model_sha256=model_digest(model) if model is not None else None, fixed_layer=layer)
                trace, reused = cache(path, meta, lambda: full_trace(native, model, arm, layer, files, gt[0], reader, dataset, progress))
                if reused:
                    reused_work(progress, len(files) - 1)
                audit_trace(trace, gt[0], len(files) - 1)
                if arm == "native":
                    baseline = trace
                rows.append(describe_trace(trace, baseline, gt, seq, arm, seed, fold))
                for k in range(7, 13):
                    routes.append(dict(sequence=seq, fold=fold, seed=seed, arm=arm, layer=k,
                                       frames=int((trace["layer"] == k).sum())))
                record(root, path, artifacts)
                if model is not None:
                    readouts.append(readout(root, plan, native, model, arm, seq, seed, fold, bank[group], progress, artifacts))
                    if model_digest(model) != meta["model_sha256"]:
                        raise ValueError("Inference changed learned module state")
            native.assert_frozen()
        tables["seed_video_summary"], tables["routing"], tables["readout_summary"] = rows, routes, readouts
        tables["policy_summary"], tables["paired"], tables["factor_summary"] = protocol.summarize(rows)
        hashes = {}
        for key in TABLES:
            path = root / f"stage194_{key}.csv"
            write_csv(path, tables[key])
            hashes[path.name] = file_hash(path)
        if progress.done != progress.total - 1:
            raise ValueError(f"Work accounting differs: {progress.done} != {progress.total - 1}")
        for row in tables["policy_summary"]:
            print(f"{row['arm']:24s} AUC={row['auc']:.4f} low-IoU={row['low_iou_fraction']:.2%}", flush=True)
        main_row = next(r for r in tables["paired"] if r["contrast"] == "a1_r1_q1_minus_current_a1_r1_q1" and r["metric"] == "auc")
        publish_identical(root / "stage194_decision.json", sealed(dict(schema=SCHEMA,
            status="REVIEW_MATCHED_MODULE_ABLATION_PILOT", plan_fingerprint=plan["fingerprint"],
            full_trajectories=len(rows), training_runs=len(tables["scope_summary"]), source_groups=38,
            confirmation_groups_opened=0, original_native_unchanged=True,
            training_budget="equal group/sample draws, updates and six-branch forward examples; unequal backward costs reported",
            primary_contrast=main_row, confidence_intervals="nominal paired source bootstrap; exploratory, no familywise adoption claim",
            limitations=["historically exposed development sources", "native-state replay training only",
                         "target-conditioned feature adapter, not temporal memory", "instrumented latency, not deployment benchmark"],
            next_step="Use OOF mechanism and full-track results to identify contributing modules before any temporal extension or confirmation opening",
            table_sha256=hashes, artifact_sha256=artifacts)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()
        verify_results(root)


if __name__ == "__main__":
    main()
