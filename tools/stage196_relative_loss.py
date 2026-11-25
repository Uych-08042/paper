"""Matched CPU training: absolute quality MSE versus centered layer-error MSE.

No image reads, SGLA forward calls, new feature collection or confirmation data.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter

REPO = Path(__file__).resolve().parents[1]
for directory in (REPO, REPO / "tools", REPO / "tracking"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import numpy as np

from experiment_progress import OverallProgress
from stage27_uav_collect import writer_lock
from stage27_uav_common import publish_identical
from stage4_common import file_hash, fingerprint, write_cache, write_csv
import stage194_module_common as original
import stage194_module_model as architecture
import stage195_router_objective_audit as audit
import stage195_routing_common as metrics

source = audit.source
SCHEMA = "stage196_matched_centered_routing_loss_v1"
SOURCE195 = "a282dbfb2ca856e5a2fe4009356a10e35a4784a011837fa8cf2381bd1622b764"
PLAN195 = "a4ebb55b99ec61c17489113af42dd8c3ef6c6e602b24a62161e6347653a8a2a5"
LOSSES = ("absolute", "relative")
TABLES = ("model_scope", "error_summary", "error_video", "policy_summary", "policy_video",
          "paired", "frame_predictions", "training_history", "source_bridge")


def components(torch, prediction, label):
    absolute = (prediction - label).square().mean()
    common = (prediction.mean(1) - label.mean(1)).square().mean()
    relative = ((prediction - prediction.mean(1, keepdim=True))
                - (label - label.mean(1, keepdim=True))).square().mean()
    return dict(absolute=absolute, common=common, relative=relative)


def pool(torch, tokens, lens_z):
    z, x = tokens[:, :lens_z], tokens[:, lens_z:]
    return torch.cat((z.mean(1), z.std(1, unbiased=False), x.mean(1), x.std(1, unbiased=False)), -1)


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_inputs(root194, root195):
    plan, decision, reader, tables = audit.load_source(root194)
    d195 = source.read_sealed(root195 / "stage195_decision.json")
    p195 = source.read_sealed(root195 / "manifest.json")
    if (d195["fingerprint"] != SOURCE195 or p195["fingerprint"] != PLAN195
            or d195["plan_fingerprint"] != p195["fingerprint"] or d195["source_stage194"] != audit.SOURCE194
            or d195["cpu_gpu_action_mismatches"] != 0 or d195["predicted_states"] != 7296):
        raise ValueError("Need the completed, reviewed Stage-195 source")
    for path in (audit.__file__, metrics.__file__, source.__file__, architecture.__file__):
        if p195["code_sha256"].get(Path(path).name) != file_hash(path):
            raise ValueError("Frozen Stage-195 analysis implementation changed: " + str(path))
    for name, digest in d195["table_sha256"].items():
        if file_hash(root195 / name) != digest:
            raise ValueError("Stage-195 table changed: " + name)
    for name, expected in d195["source_verified_artifacts"].items():
        if (decision["artifact_sha256"] | decision["table_sha256"]).get(name) != expected:
            raise ValueError("Stage-195 source binding changed")
    old_frames = read_csv(root195 / "stage195_frame_predictions.csv")
    return plan, decision, reader, tables, d195, old_frames


def cache(path, metadata, compute):
    receipt = path.with_suffix(".receipt.json")
    if receipt.exists():
        report = source.read_sealed(receipt)
        if report["schema"] != SCHEMA or report["metadata"] != metadata or file_hash(path) != report["sha256"]:
            raise ValueError("Committed Stage-196 cache changed")
        import json
        with np.load(path, allow_pickle=False) as f:
            if json.loads(str(f["metadata_json"].item())) != metadata:
                raise ValueError("Stage-196 NPZ metadata changed")
            return {k: f[k] for k in f.files if k != "metadata_json"}, True
    if path.exists():
        import json
        with np.load(path, allow_pickle=False) as f:
            if json.loads(str(f["metadata_json"].item())) != metadata:
                raise ValueError("Unidentified orphan prediction cache")
    value = compute()
    write_cache(path, value, metadata)
    publish_identical(receipt, source.sealed(dict(schema=SCHEMA, metadata=metadata, sha256=file_hash(path))))
    return value, False


def save_checkpoint(torch, path, value, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = path.with_suffix(".receipt.json")
    if receipt.exists():
        raise ValueError("Never overwrite a committed Stage-196 model")
    if path.exists() and torch.load(path, map_location="cpu", weights_only=True).get("metadata") != metadata:
        raise ValueError("Unidentified orphan model")
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
        temp = Path(f.name)
    try:
        torch.save(dict(value, metadata=metadata), temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    publish_identical(receipt, source.sealed(dict(schema=SCHEMA, metadata=metadata, sha256=file_hash(path))))


def fit(torch, root, plan, bank, fold, seed, loss_name, lens_z, progress, artifacts):
    dimension = next(iter(bank.values()))["tokens"].shape[-1]
    model = architecture.build_modules(torch, dimension, lens_z, metrics.ARM).eval()
    initial = source.model_digest(model)
    sizes = {g: len(v["frame_id"]) for g, v in bank.items()}
    draws = original.draws_for(fold, seed, sizes)
    metadata = dict(plan_fingerprint=plan["fingerprint"], fold=fold["fold"], seed=seed, loss=loss_name,
                    fit=fold["fit"], held=fold["held"], initial_sha256=initial, draws_sha256=fingerprint(draws),
                    updates=original.DESIGN["updates"], examples=original.DESIGN["updates"] * original.DESIGN["batch"])
    path = root / "training" / f"fold{fold['fold']}" / f"seed{seed}" / (loss_name + ".pt")
    receipt = path.with_suffix(".receipt.json")
    if receipt.exists():
        report = source.read_sealed(receipt)
        if report["schema"] != SCHEMA or report["metadata"] != metadata or report["sha256"] != file_hash(path):
            raise ValueError("Committed model metadata changed")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["metadata"] != metadata:
            raise ValueError("Model training identity changed")
        model.load_state_dict(saved["state"], strict=True)
        if source.model_digest(model) != saved["scope"]["final_sha256"]:
            raise ValueError("Model state changed")
        source.reused_work(progress, len(draws))
        source.record(root, path, artifacts)
        return model.requires_grad_(False), saved["scope"], saved["history"], draws
    parameters = list(model.router.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=original.DESIGN["learning_rate"], weight_decay=original.DESIGN["weight_decay"])
    history, maximum_gradient = [], 0.
    start = perf_counter()
    for step, batch in enumerate(draws, 1):
        if any(g not in fold["fit"] or g in fold["held"] for g, _ in batch):
            raise ValueError("Held source entered Stage-196 training")
        x = torch.as_tensor(np.stack([bank[g]["pooled"][i] for g, i in batch]), dtype=torch.float32)
        y = torch.as_tensor(np.stack([bank[g]["labels"][i] for g, i in batch]), dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        terms = components(torch, model.router(x).sigmoid(), y)
        loss = terms[loss_name]
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, original.DESIGN["gradient_clip"], error_if_nonfinite=True))
        maximum_gradient = max(maximum_gradient, norm)
        if not np.isfinite([norm, float(loss.detach())]).all():
            raise ValueError("Non-finite training update")
        if any(p.grad is not None for name in ("adapter", "quality") for p in getattr(model, name).parameters()):
            raise ValueError("Gradient reached disabled A/Q")
        optimizer.step()
        history.append(dict(fold=fold["fold"], seed=seed, loss_name=loss_name, step=step,
                            optimized_loss=float(loss.detach()), gradient_norm=norm,
                            **{k + "_mse": float(v.detach()) for k, v in terms.items()}))
        progress.advance()
    if maximum_gradient <= 0 or source.model_digest(model) == initial:
        raise ValueError("Routing parameters did not update")
    if any(torch.count_nonzero(p).item() for name in ("adapter", "quality") for p in getattr(model, name).parameters()):
        raise ValueError("Disabled A/Q changed")
    scope = dict(fold=fold["fold"], seed=seed, loss_name=loss_name, device="cpu", updates=len(draws),
                 examples=sum(len(b) for b in draws), trainable_parameters=sum(p.numel() for p in parameters),
                 fit_groups="|".join(fold["fit"]), held_groups="|".join(fold["held"]),
                 initial_sha256=initial, final_sha256=source.model_digest(model), draws_sha256=fingerprint(draws),
                 training_seconds=perf_counter() - start, maximum_gradient=maximum_gradient,
                 disabled_AQ_unchanged=True)
    save_checkpoint(torch, path, dict(state=model.state_dict(), optimizer=optimizer.state_dict(), scope=scope, history=history), metadata)
    source.record(root, path, artifacts)
    return model.requires_grad_(False), scope, history, draws


def predict(torch, model, tokens, pooled, progress):
    prediction, choices = [], []
    error = 0.
    with torch.no_grad():
        for token, value in zip(tokens, pooled, strict=True):
            score = model.route(torch.as_tensor(token[None], dtype=torch.float32))
            alternate = model.router(torch.as_tensor(value[None], dtype=torch.float32)).sigmoid()
            error = max(error, float((score - alternate).abs().max()))
            prediction.append(score.numpy()[0].copy())
            choices.append(int(torch.topk(score, 1, dim=1).indices[0, 0]))
            progress.advance()
    if error > 1e-7:
        raise ValueError("Prepooled features differ from the original routing function")
    return dict(prediction=np.asarray(prediction), choice=np.asarray(choices, int), maximum_error=np.asarray(error))


def pair_reports(policy_rows, error_rows):
    summary, paired = [], []
    for loss_name in LOSSES:
        rows = [{k: v for k, v in r.items() if k != "loss_name"} for r in policy_rows if r["loss_name"] == loss_name]
        own_summary, own_paired = audit.summarize_policies(rows)
        summary.extend(dict(loss_name=loss_name, **r) for r in own_summary)
        paired.extend(dict(comparison="within_" + loss_name, **r) for r in own_paired)
    for pool in ("fit", "held"):
        for category, metric, direction in (("policy", "mean_iou", "positive"), ("policy", "mean_regret", "negative"),
                                             ("prediction", "relative_mse", "negative"), ("prediction", "gap_weighted_pair_accuracy", "positive")):
            rows = policy_rows if category == "policy" else error_rows
            selected = [r for r in rows if r["pool"] == pool and r.get("policy", r.get("predictor")) == "trained"]
            groups = sorted({r["group"] for r in selected})
            if len(groups) != 38:
                raise ValueError("Incomplete paired source pool")
            delta = []
            for group in groups:
                values = {}
                for loss_name in LOSSES:
                    cells = [r[metric] for r in selected if r["group"] == group and r["loss_name"] == loss_name]
                    if len(cells) != 2:
                        raise ValueError("Each source needs two seeds per objective")
                    values[loss_name] = float(np.mean(cells)) if all(v is not None for v in cells) else None
                if all(v is not None for v in values.values()):
                    delta.append(values["relative"] - values["absolute"])
            if len(delta) >= 2:
                comparison = original.contrast(delta)
            else:
                comparison = dict(videos=len(delta), mean_delta=delta[0] if delta else None,
                                  median_delta=delta[0] if delta else None, ci_low=None, ci_high=None,
                                  positive=sum(v > 1e-9 for v in delta), negative=sum(v < -1e-9 for v in delta),
                                  worst_leave_one_video_out=None)
            paired.append(dict(comparison=category, pool=pool, contrast="relative_minus_absolute", metric=metric,
                               favorable_direction=direction, **comparison))
    return summary, paired


def verify_results(root):
    result = source.read_sealed(root / "stage196_decision.json")
    plan = source.read_sealed(root / "manifest.json")
    if (result["schema"] != SCHEMA or result["status"] != "REVIEW_MATCHED_RELATIVE_ROUTING_LOSS"
            or result["plan_fingerprint"] != plan["fingerprint"] or result["source_stage195"] != SOURCE195
            or result["training_runs"] != 8 or result["training_updates"] != 8 * original.DESIGN["updates"]
            or result["predicted_states"] != 8 * plan["states_per_model"]
            or result["new_image_reads"] != 0 or result["new_encoder_calls"] != 0
            or result["confirmation_groups_opened"] != 0):
        raise ValueError("Incomplete Stage-196 comparison")
    if set(result["table_sha256"]) != {f"stage196_{name}.csv" for name in TABLES}:
        raise ValueError("Incomplete Stage-196 tables")
    for name, sha in (result["table_sha256"] | result["artifact_sha256"]).items():
        if file_hash(root / name) != sha:
            raise ValueError("Stage-196 artifact changed: " + name)
    print("Stage-196 cache-only training comparison verified; REVIEW is not adoption", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage194-root", type=Path)
    parser.add_argument("--stage195-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-results", action="store_true")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        verify_results(root)
        return
    if args.stage194_root is None or args.stage195_root is None:
        parser.error("Both completed source directories are required")
    r194, r195 = args.stage194_root.expanduser().resolve(), args.stage195_root.expanduser().resolve()
    original.safe_output(root, [r194, r195])
    with writer_lock(root):
        if (root / "stage196_decision.json").exists():
            verify_results(root)
            return
        p194, d194, reader, old_tables, d195, old_frames = load_inputs(r194, r195)
        # This is a fresh CPU-only CLI run, including when torch is a CUDA build.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch

        torch.set_num_threads(1)
        plan = source.sealed(dict(schema=SCHEMA, source_stage194=d194["fingerprint"], source_stage195=SOURCE195,
            source_plan194=p194["fingerprint"], source_decision194_sha256=file_hash(r194 / "stage194_decision.json"),
            source_decision195_sha256=file_hash(r195 / "stage195_decision.json"), states_per_model=sum(len(e["sample_frames"]) for e in p194["entries"]),
            folds=p194["folds"], seeds=original.DESIGN["seeds"], training=original.DESIGN,
            losses=list(LOSSES), intervention="Remove only common-row error; retain original sigmoid and routing rule",
            device="cpu", torch=str(torch.__version__), evaluation="cached native-parent states; no new full trajectories",
            code_sha256={Path(p).name: file_hash(p) for p in (__file__, audit.__file__, metrics.__file__, architecture.__file__, source.__file__)}))
        publish_identical(root / "manifest.json", plan)
        views = plan["states_per_model"]
        total = 1 + len(p194["entries"]) + views + 8 * original.DESIGN["updates"] + 8 * views + 1
        progress = OverallProgress("Stage-196", total, unit="work")
        progress.report("Verifying frozen feature and label banks")
        progress.advance()
        bank, artifacts = {}, {}
        lens_z = int(p194["runtime"]["tracker"]["lens_z"])
        for spec in p194["entries"]:
            group, seq = spec["group"], spec["sequence"]
            data = reader.arrays(f"features/{seq}.npz", dict(plan_fingerprint=p194["fingerprint"], sequence=seq, kind="native_parent_prefix"))
            y = reader.arrays(f"reference/{group}.npz", dict(plan_fingerprint=p194["fingerprint"], kind="fixed_layer_fit_labels", group=group))["iou"]
            np.testing.assert_array_equal(data["frame_id"], spec["sample_frames"])
            metrics.validate_scores(y, y)
            if len(data["tokens"]) != len(y) or not np.isfinite(data["tokens"]).all():
                raise ValueError("Feature and label states disagree")
            progress.advance()
            pooled = []
            with torch.no_grad():
                for token in data["tokens"]:
                    pooled.append(pool(torch, torch.as_tensor(token[None], dtype=torch.float32), lens_z).numpy()[0].copy())
                    progress.advance()
            bank[group] = dict(data, pooled=np.asarray(pooled), labels=np.asarray(y, float), sequence=seq)
        rows = {name: [] for name in TABLES}
        old_index = {(int(r["fold"]), int(r["seed"]), r["group"], int(r["frame"])): r for r in old_frames}
        progress.report("Matched CPU routing training and held-source predictions")
        maximum_error = 0.
        for fold in p194["folds"]:
            for seed in original.DESIGN["seeds"]:
                original_scope = next(r for r in old_tables["scope_summary"] if int(r["fold"]) == fold["fold"]
                                      and int(r["seed"]) == seed and r["arm"] == metrics.ARM)
                scopes = []
                for loss_name in LOSSES:
                    model, scope, history, draws = fit(torch, root, plan, bank, fold, seed, loss_name, lens_z, progress, artifacts)
                    if scope["draws_sha256"] != original_scope["draws_sha256"]:
                        raise ValueError("Training schedule differs from the reviewed source")
                    scopes.append(scope)
                    before = source.model_digest(model)
                    counts = {g: np.zeros(len(d["frame_id"]), int) for g, d in bank.items()}
                    for batch in draws:
                        for group, index in batch:
                            counts[group][index] += 1
                    videos = {}
                    for group, data in bank.items():
                        meta = dict(plan_fingerprint=plan["fingerprint"], fold=fold["fold"], seed=seed,
                                    loss_name=loss_name, group=group, model_sha256=before)
                        path = root / "predictions" / f"f{fold['fold']}__s{seed}__{loss_name}__{group}.npz"
                        value, reused = cache(path, meta, lambda: predict(torch, model, data["tokens"], data["pooled"], progress))
                        if reused:
                            source.reused_work(progress, len(data["frame_id"]))
                        source.record(root, path, artifacts)
                        metrics.validate_scores(data["labels"], value["prediction"])
                        maximum_error = max(maximum_error, float(value["maximum_error"]))
                        videos[group] = dict(sequence=data["sequence"], frame_id=data["frame_id"], labels=data["labels"],
                                             prediction=value["prediction"], choice=value["choice"],
                                             archived_choice=None, draw_count=counts[group])
                        if loss_name == "absolute":
                            archive = [old_index[fold["fold"], seed, group, int(frame)] for frame in data["frame_id"]]
                            old_p = np.asarray([[float(r[f"prediction_L{k}"]) for k in range(7, 13)] for r in archive])
                            old_choice = np.asarray([int(r["cpu_layer"]) - 7 for r in archive])
                            rows["source_bridge"].append(dict(fold=fold["fold"], seed=seed, group=group, sequence=data["sequence"],
                                pool="fit" if group in fold["fit"] else "held", states=len(archive),
                                action_agreement=float((old_choice == value["choice"]).mean()),
                                maximum_probability_difference=float(np.abs(old_p - value["prediction"]).max()),
                                old_mean_iou=float(data["labels"][np.arange(len(archive)), old_choice].mean()),
                                new_absolute_mean_iou=float(data["labels"][np.arange(len(archive)), value["choice"]].mean())))
                    controls = metrics.fit_controls(videos, fold["fit"], draws)
                    error_rows, policy_rows, frame_rows = audit.make_model_reports(videos, controls, fold, seed)
                    rows["error_video"].extend(dict(loss_name=loss_name, **r) for r in error_rows)
                    rows["policy_video"].extend(dict(loss_name=loss_name, **r) for r in policy_rows)
                    rows["frame_predictions"].extend(dict(loss_name=loss_name, **r) for r in frame_rows)
                    rows["error_summary"].extend(dict(loss_name=loss_name, **r)
                        for r in audit.pool_error_summary(videos, controls, fold, seed, draws))
                    if source.model_digest(model) != before:
                        raise ValueError("Evaluation changed frozen routing parameters")
                    rows["model_scope"].append(scope)
                    rows["training_history"].extend(history)
                if (scopes[0]["initial_sha256"] != scopes[1]["initial_sha256"]
                        or scopes[0]["draws_sha256"] != scopes[1]["draws_sha256"]
                        or scopes[0]["trainable_parameters"] != scopes[1]["trainable_parameters"]):
                    raise ValueError("Loss comparison is not matched")
        rows["policy_summary"], rows["paired"] = pair_reports(rows["policy_video"], rows["error_video"])
        hashes = {}
        for name in TABLES:
            path = root / f"stage196_{name}.csv"
            write_csv(path, rows[name])
            hashes[path.name] = file_hash(path)
        if progress.done != progress.total - 1:
            raise ValueError("Incomplete Stage-196 work accounting")
        primary = next(r for r in rows["paired"] if r["comparison"] == "policy" and r["pool"] == "held" and r["metric"] == "mean_iou")
        result = source.sealed(dict(schema=SCHEMA, status="REVIEW_MATCHED_RELATIVE_ROUTING_LOSS",
            plan_fingerprint=plan["fingerprint"], source_stage194=d194["fingerprint"], source_stage195=SOURCE195,
            training_runs=len(rows["model_scope"]), training_updates=len(rows["training_history"]),
            predicted_states=len(rows["frame_predictions"]), source_groups=38, device="cpu",
            new_image_reads=0, new_encoder_calls=0, new_full_trajectories=0, confirmation_groups_opened=0,
            original_routing_function_maximum_error=maximum_error, primary_contrast=primary,
            interval_scope="nominal paired source bootstrap, historical development cache only",
            interpretation="Loss intervention only; a smaller optimized relative loss does not establish better selection",
            next_step="Check held relative error, ordering and gain over constant controls before full-video validation",
            source_verified_artifacts=reader.verified, table_sha256=hashes, artifact_sha256=artifacts))
        publish_identical(root / "stage196_decision.json", result)
        progress.phase = "Complete"
        progress.advance()
        progress.report()
        verify_results(root)


if __name__ == "__main__":
    main()
