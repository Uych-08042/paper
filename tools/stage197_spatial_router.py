"""CPU cached-state test of token order for task-quality routing, Stage-197."""
from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path
import sys
from time import perf_counter

REPO = Path(__file__).resolve().parents[1]
for folder in (REPO, REPO / "tools", REPO / "tracking"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import numpy as np
from experiment_progress import OverallProgress
from stage27_uav_collect import writer_lock
from stage27_uav_common import publish_identical
from stage4_common import file_hash, fingerprint, write_csv, write_cache
import stage196_relative_loss as prior
import stage197_spatial_router_model as architecture

source, original, audit, metrics = prior.source, prior.original, prior.audit, prior.metrics
SCHEMA = "stage197_matched_token_order_routing_v1"
SOURCE196 = "b2769871bf752da66efd6d5a6bb1f2f211affd3e18b0105113e27429eff9bc91"
PLAN196 = "6453511ab517f010f0bc79bee7222adc8f3e8e2be881bcb799956e6e37626121"
ARMS = architecture.ARMS
TABLES = ("model_scope", "input_audit", "error_summary", "error_video", "policy_summary", "policy_video",
          "paired", "frame_predictions", "training_history", "source_bridge")


def load_inputs(r194, r195, r196):
    inputs = prior.load_inputs(r194, r195)
    p196, d196 = source.read_sealed(r196 / "manifest.json"), source.read_sealed(r196 / "stage196_decision.json")
    if (p196["fingerprint"] != PLAN196 or d196["fingerprint"] != SOURCE196
            or d196["plan_fingerprint"] != PLAN196 or d196["source_stage194"] != audit.SOURCE194):
        raise ValueError("Need reviewed Stage-196 source")
    prior.verify_results(r196)
    for path in (prior.__file__, audit.__file__, metrics.__file__, prior.architecture.__file__, source.__file__):
        if p196["code_sha256"].get(Path(path).name) != file_hash(path):
            raise ValueError("Frozen source code changed: " + str(path))
    frames = prior.read_csv(r196 / "stage196_frame_predictions.csv")
    return (*inputs[:4], d196, frames)


def commit_model(torch, path, payload, metadata):
    # Atomic publication; only identified, uncommitted orphan data may be replaced.
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = path.with_suffix(".receipt.json")
    if receipt.exists():
        raise ValueError("Never overwrite a committed model")
    if path.exists() and torch.load(path, map_location="cpu", weights_only=True).get("metadata") != metadata:
        raise ValueError("Unidentified orphan model")
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
    try:
        torch.save(dict(payload, metadata=metadata), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    publish_identical(receipt, source.sealed(dict(schema=SCHEMA, metadata=metadata, sha256=file_hash(path))))


def cache(path, metadata, compute):
    import json
    receipt = path.with_suffix(".receipt.json")
    if receipt.exists():
        report = source.read_sealed(receipt)
        if report["schema"] != SCHEMA or report["metadata"] != metadata or report["sha256"] != file_hash(path):
            raise ValueError("Committed prediction cache changed")
    if path.exists():
        with np.load(path, allow_pickle=False) as f:
            if json.loads(str(f["metadata_json"].item())) != metadata:
                raise ValueError("Prediction metadata changed")
            if receipt.exists():
                return {k: f[k] for k in f.files if k != "metadata_json"}, True
    value = compute()
    write_cache(path, value, metadata)
    publish_identical(receipt, source.sealed(dict(schema=SCHEMA, metadata=metadata, sha256=file_hash(path))))
    return value, False


def fit(torch, root, plan, bank, fold, seed, arm, lens_z, progress, artifacts):
    shape = next(iter(bank.values()))["tokens"].shape
    model = architecture.build(torch, shape[-1], lens_z, shape[1], arm).eval()
    initial = source.model_digest(model)
    draws = original.draws_for(fold, seed, {g: len(v["frame_id"]) for g, v in bank.items()})
    metadata = dict(plan_fingerprint=plan["fingerprint"], fold=fold["fold"], seed=seed, arm=arm,
                    fit=fold["fit"], held=fold["held"], initial_sha256=initial, draws_sha256=fingerprint(draws),
                    updates=original.DESIGN["updates"], examples=len(draws) * original.DESIGN["batch"])
    path = root / "training" / f"fold{fold['fold']}" / f"seed{seed}" / (arm + ".pt")
    receipt = path.with_suffix(".receipt.json")
    if receipt.exists():
        report = source.read_sealed(receipt)
        if report["schema"] != SCHEMA or report["metadata"] != metadata or report["sha256"] != file_hash(path):
            raise ValueError("Committed model changed")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["metadata"] != metadata:
            raise ValueError("Model identity mismatch")
        model.load_state_dict(payload["state"], strict=True)
        if source.model_digest(model) != payload["scope"]["final_sha256"]:
            raise ValueError("Model parameter hash mismatch")
        source.reused_work(progress, len(draws))
        source.record(root, path, artifacts)
        return model.requires_grad_(False), payload["scope"], payload["history"], draws
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=original.DESIGN["learning_rate"], weight_decay=original.DESIGN["weight_decay"])
    history, maximum_gradient, maximum_projection_gradient = [], 0., 0.
    start = perf_counter()
    for step, batch in enumerate(draws, 1):
        if any(g not in fold["fit"] or g in fold["held"] for g, _ in batch):
            raise ValueError("Held source entered training")
        tokens = torch.as_tensor(np.stack([bank[g]["tokens"][i] for g, i in batch]), dtype=torch.float32)
        pooled = torch.as_tensor(np.stack([bank[g]["pooled"][i] for g, i in batch]), dtype=torch.float32)
        labels = torch.as_tensor(np.stack([bank[g]["labels"][i] for g, i in batch]), dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        terms = prior.components(torch, model(tokens, pooled), labels)
        terms["absolute"].backward()
        projection_gradient = float(model.projection.weight.grad.norm()) if arm != "global_stats" else 0.
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, original.DESIGN["gradient_clip"], error_if_nonfinite=True))
        maximum_gradient = max(maximum_gradient, norm)
        maximum_projection_gradient = max(maximum_projection_gradient, projection_gradient)
        if not np.isfinite([norm, projection_gradient, float(terms["absolute"].detach())]).all():
            raise ValueError("Nonfinite training")
        optimizer.step()
        history.append(dict(fold=fold["fold"], seed=seed, arm=arm, step=step, gradient_norm=norm,
                            projection_gradient_norm=projection_gradient,
                            **{k + "_mse": float(v.detach()) for k, v in terms.items()}))
        progress.advance()
    if maximum_gradient <= 0 or source.model_digest(model) == initial:
        raise ValueError("No parameter update")
    if arm != "global_stats" and maximum_projection_gradient <= 0:
        raise ValueError("Shared channel projection never received a gradient")
    scope = dict(fold=fold["fold"], seed=seed, arm=arm, device="cpu", updates=len(draws),
                 examples=sum(map(len, draws)), trainable_parameters=sum(p.numel() for p in parameters),
                 input_features=model.readout.in_features,
                 initial_sha256=initial, final_sha256=source.model_digest(model), draws_sha256=fingerprint(draws),
                 fit_groups="|".join(fold["fit"]), held_groups="|".join(fold["held"]),
                 maximum_gradient=maximum_gradient, maximum_projection_gradient=maximum_projection_gradient,
                 training_seconds=perf_counter() - start, loss_name="absolute")
    commit_model(torch, path, dict(state=model.state_dict(), optimizer=optimizer.state_dict(), scope=scope, history=history), metadata)
    source.record(root, path, artifacts)
    return model.requires_grad_(False), scope, history, draws


def predict(torch, model, tokens, pooled, progress):
    scores, choices, error = [], [], 0.
    with torch.no_grad():
        for token, value in zip(tokens, pooled, strict=True):
            token = torch.as_tensor(token[None], dtype=torch.float32)
            p = model(token, torch.as_tensor(value[None], dtype=torch.float32))
            error = max(error, float((p - model(token)).abs().max()))
            scores.append(p.numpy()[0].copy())
            choices.append(int(torch.topk(p, 1, dim=1).indices[0, 0]))
            progress.advance()
    if error > 1e-7:
        raise ValueError("Prepooled and online interface disagree")
    return dict(prediction=np.asarray(scores), choice=np.asarray(choices, int), maximum_error=np.asarray(error))


def summarize(policy_rows, error_rows):
    summary, paired = [], []
    for arm in ARMS:
        rows = [{k: v for k, v in r.items() if k != "arm"} for r in policy_rows if r["arm"] == arm]
        own, pairs = audit.summarize_policies(rows)
        summary.extend(dict(arm=arm, **r) for r in own)
        paired.extend(dict(comparison="within_" + arm, **r) for r in pairs)
    for a, b in itertools.combinations(ARMS, 2):
        for pool in ("fit", "held"):
            for category, metric, direction in (("policy", "mean_iou", "positive"), ("policy", "mean_regret", "negative"),
                                               ("prediction", "relative_mse", "negative"), ("prediction", "gap_weighted_pair_accuracy", "positive")):
                rows = policy_rows if category == "policy" else error_rows
                subset = [r for r in rows if r["pool"] == pool and r.get("policy", r.get("predictor")) == "trained"]
                groups = sorted({r["group"] for r in subset})
                if len(groups) != 38:
                    raise ValueError("Incomplete source pool")
                delta = []
                for group in groups:
                    values = {}
                    for arm in (a, b):
                        cells = [r[metric] for r in subset if r["arm"] == arm and r["group"] == group]
                        if len(cells) != 2:
                            raise ValueError("Need two seeds per source")
                        values[arm] = float(np.mean(cells)) if all(v is not None for v in cells) else None
                    if all(v is not None for v in values.values()):
                        delta.append(values[a] - values[b])
                comparison = original.contrast(delta) if len(delta) >= 2 else dict(videos=len(delta),
                    mean_delta=delta[0] if delta else None, median_delta=delta[0] if delta else None,
                    ci_low=None, ci_high=None, positive=sum(v > 1e-9 for v in delta), negative=sum(v < -1e-9 for v in delta),
                    worst_leave_one_video_out=None)
                paired.append(dict(comparison=category, pool=pool, contrast=a + "_minus_" + b, metric=metric,
                                   favorable_direction=direction, **comparison))
    return summary, paired


def verify_results(root):
    d, p = source.read_sealed(root / "stage197_decision.json"), source.read_sealed(root / "manifest.json")
    if (d["schema"] != SCHEMA or p["schema"] != SCHEMA or d["plan_fingerprint"] != p["fingerprint"]
            or d["source_stage196"] != SOURCE196 or d["training_runs"] != 12
            or d["training_updates"] != 12 * original.DESIGN["updates"]
            or d["predicted_states"] != 12 * p["states_per_model"] or d["source_groups"] != 38
            or any(d[k] != 0 for k in ("new_image_reads", "new_encoder_calls", "new_full_trajectories", "confirmation_groups_opened"))
            or d["status"] != "REVIEW_MATCHED_SPATIAL_ROUTING_INTERFACE"):
        raise ValueError("Incomplete Stage-197")
    if set(d["table_sha256"]) != {f"stage197_{n}.csv" for n in TABLES}:
        raise ValueError("Incomplete report tables")
    for name, sha in (d["table_sha256"] | d["artifact_sha256"]).items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != sha:
            raise ValueError("Changed Stage-197 artifact: " + name)
    print("Stage-197 verified; cached-state REVIEW is not full-video adoption", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for stage in (194, 195, 196):
        parser.add_argument(f"--stage{stage}-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--check-results", action="store_true")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        verify_results(root)
        return
    roots = [getattr(args, f"stage{n}_root") for n in (194, 195, 196)]
    if any(p is None for p in roots):
        parser.error("Completed Stage-194, 195 and 196 directories are required")
    r194, r195, r196 = [p.expanduser().resolve() for p in roots]
    original.safe_output(root, [r194, r195, r196])
    with writer_lock(root):
        if (root / "stage197_decision.json").exists():
            verify_results(root)
            return
        p194, d194, reader, old_tables, d196, old_frames = load_inputs(r194, r195, r196)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch
        torch.set_num_threads(1)
        plan = source.sealed(dict(schema=SCHEMA, source_stage194=d194["fingerprint"], source_stage196=SOURCE196,
            source_plan194=p194["fingerprint"], source_stage195=prior.SOURCE195, source_plan196=PLAN196,
            source_decision194_sha256=file_hash(r194 / "stage194_decision.json"),
            source_decision196_sha256=file_hash(r196 / "stage196_decision.json"),
            folds=p194["folds"], seeds=original.DESIGN["seeds"], training=original.DESIGN, arms=list(ARMS),
            loss="absolute", device="cpu", torch=str(torch.__version__), numpy=str(np.__version__), python=sys.version,
            states_per_model=sum(len(e["sample_frames"]) for e in p194["entries"]),
            primary="held ordered_minus_unordered mean_iou; same parameters, initialization, draws and budget",
            unordered="per-sample ascending sort within template/search of shared projected token scalars",
            projection="trainable D->1; initial channel-zero; ordered and unordered share initialization only",
            global_stats_control="original 4D->6 lower-capacity secondary baseline, freshly CPU trained",
            evaluation="fixed native-parent cached immediate IoU; not full-video AUC",
            code_sha256={Path(p).name: file_hash(p) for p in (__file__, architecture.__file__, prior.__file__,
                audit.__file__, metrics.__file__, original.__file__, source.__file__)}))
        publish_identical(root / "manifest.json", plan)
        views = plan["states_per_model"]
        progress = OverallProgress("Stage-197", 1 + len(p194["entries"]) + views + 12 * original.DESIGN["updates"] + 12 * views + 1, unit="work")
        progress.report("Verifying frozen inputs and spatial-order control")
        progress.advance()
        bank, artifacts = {}, {}
        rows = {n: [] for n in TABLES}
        lens_z, token_shape = int(p194["runtime"]["tracker"]["lens_z"]), None
        for spec in p194["entries"]:
            group, seq = spec["group"], spec["sequence"]
            data = reader.arrays(f"features/{seq}.npz", dict(plan_fingerprint=p194["fingerprint"], sequence=seq, kind="native_parent_prefix"))
            y = reader.arrays(f"reference/{group}.npz", dict(plan_fingerprint=p194["fingerprint"], kind="fixed_layer_fit_labels", group=group))["iou"]
            np.testing.assert_array_equal(data["frame_id"], spec["sample_frames"])
            metrics.validate_scores(y, y)
            tokens = np.asarray(data["tokens"], np.float32)
            if tokens.ndim != 3 or len(tokens) != len(y) or not np.isfinite(tokens).all():
                raise ValueError("Malformed cached tokens")
            if token_shape is not None and tokens.shape[1:] != token_shape:
                raise ValueError("Token layout differs across sources")
            token_shape = tokens.shape[1:]
            progress.advance()
            pooled = []
            with torch.no_grad():
                for token in tokens:
                    pooled.append(prior.pool(torch, torch.as_tensor(token[None]), lens_z).numpy()[0].copy())
                    progress.advance()
            pooled = np.asarray(pooled)
            proof = architecture.layout_audit(torch, torch.as_tensor(tokens), torch.as_tensor(pooled), lens_z)
            rows["input_audit"].append(dict(group=group, sequence=seq, **proof))
            bank[group] = dict(data, tokens=tokens, pooled=pooled, labels=np.asarray(y, float), sequence=seq)
        if len(bank) != 38 or sum(r["ordered_changed_states"] for r in rows["input_audit"]) == 0:
            raise ValueError("Spatial control has no usable token-order variation")
        old_index = {(int(r["fold"]), int(r["seed"]), r["group"], int(r["frame"])): r
                     for r in old_frames if r["loss_name"] == "absolute"}
        progress.report("Matched CPU training and held-source predictions")
        maximum_error = 0.
        for fold in p194["folds"]:
            for seed in original.DESIGN["seeds"]:
                scopes = []
                original_scope = next(r for r in old_tables["scope_summary"] if int(r["fold"]) == fold["fold"]
                                      and int(r["seed"]) == seed and r["arm"] == metrics.ARM)
                for arm in ARMS:
                    model, scope, history, draws = fit(torch, root, plan, bank, fold, seed, arm, lens_z, progress, artifacts)
                    if scope["draws_sha256"] != original_scope["draws_sha256"]:
                        raise ValueError("Training draws differ from source")
                    scopes.append(scope)
                    before = source.model_digest(model)
                    counts = {g: np.zeros(len(v["frame_id"]), int) for g, v in bank.items()}
                    for batch in draws:
                        for g, i in batch:
                            counts[g][i] += 1
                    videos = {}
                    for group, data in bank.items():
                        meta = dict(plan_fingerprint=plan["fingerprint"], fold=fold["fold"], seed=seed, arm=arm,
                                    group=group, model_sha256=before)
                        path = root / "predictions" / f"f{fold['fold']}__s{seed}__{arm}__{group}.npz"
                        value, reused = cache(path, meta, lambda: predict(torch, model, data["tokens"], data["pooled"], progress))
                        if reused:
                            source.reused_work(progress, len(data["frame_id"]))
                        source.record(root, path, artifacts)
                        metrics.validate_scores(data["labels"], value["prediction"])
                        if value["choice"].shape != (len(data["labels"]),):
                            raise ValueError("Prediction shape mismatch")
                        maximum_error = max(maximum_error, float(value["maximum_error"]))
                        videos[group] = dict(sequence=data["sequence"], frame_id=data["frame_id"], labels=data["labels"],
                            prediction=value["prediction"], choice=value["choice"], archived_choice=None, draw_count=counts[group])
                        if arm == "global_stats":
                            old = [old_index[fold["fold"], seed, group, int(f)] for f in data["frame_id"]]
                            p = np.asarray([[float(r[f"prediction_L{k}"]) for k in range(7, 13)] for r in old])
                            choice = np.asarray([int(r["cpu_layer"]) - 7 for r in old])
                            rows["source_bridge"].append(dict(fold=fold["fold"], seed=seed, group=group,
                                pool="fit" if group in fold["fit"] else "held", states=len(old),
                                action_agreement=float((choice == value["choice"]).mean()),
                                maximum_probability_difference=float(np.abs(p - value["prediction"]).max())))
                    controls = metrics.fit_controls(videos, fold["fit"], draws)
                    er, pr, fr = audit.make_model_reports(videos, controls, fold, seed)
                    for name, values in (("error_video", er), ("policy_video", pr), ("frame_predictions", fr),
                        ("error_summary", audit.pool_error_summary(videos, controls, fold, seed, draws))):
                        rows[name].extend(dict(arm=arm, **r) for r in values)
                    if source.model_digest(model) != before:
                        raise ValueError("Evaluation modified the trained model")
                    rows["model_scope"].append(scope)
                    rows["training_history"].extend(history)
                for field in ("initial_sha256", "draws_sha256", "trainable_parameters", "input_features"):
                    if scopes[0][field] != scopes[1][field]:
                        raise ValueError("Spatial-order primary comparison is not matched: " + field)
        rows["policy_summary"], rows["paired"] = summarize(rows["policy_video"], rows["error_video"])
        hashes = {}
        for name in TABLES:
            path = root / f"stage197_{name}.csv"
            write_csv(path, rows[name])
            hashes[path.name] = file_hash(path)
        if progress.done != progress.total - 1:
            raise ValueError("Incomplete work accounting")
        primary = next(r for r in rows["paired"] if r["comparison"] == "policy" and r["pool"] == "held"
                       and r["contrast"] == "ordered_minus_unordered" and r["metric"] == "mean_iou")
        result = source.sealed(dict(schema=SCHEMA, status="REVIEW_MATCHED_SPATIAL_ROUTING_INTERFACE",
            plan_fingerprint=plan["fingerprint"], source_stage194=d194["fingerprint"], source_stage196=d196["fingerprint"],
            source_groups=38, training_runs=len(rows["model_scope"]), training_updates=len(rows["training_history"]),
            predicted_states=len(rows["frame_predictions"]), new_image_reads=0, new_encoder_calls=0,
            new_full_trajectories=0, confirmation_groups_opened=0, device="cpu", primary_contrast=primary,
            prepooled_interface_maximum_error=maximum_error,
            interval_scope="nominal source bootstrap, historically exposed development cache; no adoption gate",
            next_step="Require held ranking and gain over unordered AND fit constant controls before trajectory validation",
            source_verified_artifacts=reader.verified, table_sha256=hashes, artifact_sha256=artifacts))
        publish_identical(root / "stage197_decision.json", result)
        progress.phase = "Complete"
        progress.advance()
        progress.report()
        verify_results(root)


if __name__ == "__main__":
    main()
