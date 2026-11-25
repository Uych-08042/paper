"""CPU matched layer-MSE versus localization-regret margin training; no image inference."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter

REPO = Path(__file__).resolve().parents[1]
for folder in (REPO / "tools", REPO / "tracking"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import numpy as np
from experiment_progress import OverallProgress
from stage27_uav_collect import writer_lock
from stage27_uav_common import publish_identical
from stage4_common import file_hash, fingerprint, write_csv, write_cache
import stage200_within_video as previous
import stage201_score_baseline as bundle_source
import stage202_regret_common as common

source, inputs, architecture, metrics, protocol = previous.source, previous.inputs, previous.architecture, previous.metrics, previous.protocol
ARMS = common.ARMS
SCHEMA = "stage202_matched_regret_margin_training_v1"
SOURCE200, PLAN200 = bundle_source.SOURCE, bundle_source.SOURCE_PLAN
DEFAULT_BUNDLE = bundle_source.DEFAULT_BUNDLE
PROTOCOL_FILE = REPO / "docs/experiments/06_sgla_uav_tracking/STAGE202_REGRET_MARGIN_ROUTING.md"
POLICIES, PREDICTORS = previous.previous.POLICIES, previous.previous.PREDICTORS
TABLES = (*previous.TABLES, "seed_summary")


def load_sources(root194, bundle):
    p194, d194, reader, tables194 = inputs.load_source(root194)
    d200, p200, tables200 = bundle_source.load_bundle(bundle)
    if d200["source_stage194"] != d194["fingerprint"] or p200["folds"] != p194["folds"]:
        raise ValueError("Stage-194 source or split differs from reviewed Stage-200")
    return p194, d194, reader, tables194, d200, tables200


def cache(path, metadata, compute):
    receipt = path.with_suffix(".receipt.json")
    if receipt.exists():
        r = source.read_sealed(receipt)
        if r["schema"] != SCHEMA or r["metadata"] != metadata or r["sha256"] != file_hash(path):
            raise ValueError("Committed Stage-202 cache changed")
    if path.exists():
        with np.load(path, allow_pickle=False) as f:
            if json.loads(str(f["metadata_json"].item())) != metadata:
                raise ValueError("Unknown prediction/statistic cache")
            if receipt.exists():
                return {k: f[k] for k in f.files if k != "metadata_json"}, True
    data = compute()
    write_cache(path, data, metadata)
    publish_identical(receipt, source.sealed(dict(schema=SCHEMA, metadata=metadata, sha256=file_hash(path))))
    return data, False


def save_model(torch, path, payload, metadata):
    receipt = path.with_suffix(".receipt.json")
    if receipt.exists():
        raise ValueError("Never overwrite committed model")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and torch.load(path, map_location="cpu", weights_only=True).get("metadata") != metadata:
        raise ValueError("Unknown orphan model")
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
    try:
        torch.save(dict(payload, metadata=metadata), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    publish_identical(receipt, source.sealed(dict(schema=SCHEMA, metadata=metadata, sha256=file_hash(path))))


def fit(torch, root, plan, bank, stats, fold, seed, arm, draws, lens_z, progress, artifacts):
    if arm not in ARMS:
        raise ValueError("Unknown MSE/regret arm")
    # BOTH arms use exactly the frozen Stage-199 signed-residual architecture.
    model = architecture.build(torch, stats, lens_z, "interaction_residual").eval()
    initial = source.model_digest(model)
    metadata = dict(plan_fingerprint=plan["fingerprint"], fold=fold["fold"], seed=seed, arm=arm,
                    fit=fold["fit"], held=fold["held"], initial_sha256=initial, pair_draws_sha256=fingerprint(draws),
                    updates=len(draws), examples=sum(2 * len(batch) for batch in draws))
    path = root / "training" / f"fold{fold['fold']}" / f"seed{seed}" / (arm + ".pt")
    receipt = path.with_suffix(".receipt.json")
    if receipt.exists():
        r = source.read_sealed(receipt)
        if r["schema"] != SCHEMA or r["metadata"] != metadata or r["sha256"] != file_hash(path):
            raise ValueError("Committed model changed")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["metadata"] != metadata:
            raise ValueError("Checkpoint metadata mismatch")
        model.load_state_dict(saved["state"], strict=True)
        if source.model_digest(model) != saved["scope"]["final_sha256"]:
            raise ValueError("Model state mismatch")
        source.reused_work(progress, len(draws)); source.record(root, path, artifacts)
        return model.requires_grad_(False), saved["scope"], saved["history"]
    frozen = {k: v.clone() for k, v in model.named_buffers()}
    prior_choice = int(metrics.choose(np.asarray(stats["prior"])[None])[0])
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=protocol.DESIGN["learning_rate"], weight_decay=protocol.DESIGN["weight_decay"])
    history, maximum_gradient, maximum_projection_gradient = [], 0., 0.
    started = perf_counter()
    for step, batch in enumerate(draws, 1):
        if len(batch) != 2 or any(g not in fold["fit"] or g in fold["held"] for g, _, _ in batch):
            raise ValueError("Invalid pair budget or held-source training")
        positions = common.flatten(batch)
        tokens = torch.as_tensor(np.stack([bank[g]["tokens"][i] for g, i in positions]), dtype=torch.float32)
        pooled = torch.as_tensor(np.stack([bank[g]["pooled"][i] for g, i in positions]), dtype=torch.float32)
        labels = torch.as_tensor(np.stack([bank[g]["labels"][i] for g, i in positions]), dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        terms = common.objectives(torch, model(tokens, pooled), labels)
        loss = terms[arm]
        loss.backward()
        pg = float(model.projection.weight.grad.norm())
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, protocol.DESIGN["gradient_clip"], error_if_nonfinite=True))
        if not np.isfinite([pg, norm, float(loss.detach())]).all():
            raise ValueError("Nonfinite update")
        maximum_gradient, maximum_projection_gradient = max(maximum_gradient, norm), max(maximum_projection_gradient, pg)
        optimizer.step()
        history.append(dict(fold=fold["fold"], seed=seed, arm=arm, step=step, optimized_loss=float(loss.detach()),
            gradient_norm=norm, projection_gradient_norm=pg, self_pairs=sum(a == b for _, a, b in batch),
            point_mse=float(terms["point"].detach()), regret_margin_loss=float(terms["regret_margin"].detach())))
        progress.advance()
    unchanged = all(torch.equal(v, frozen[k]) for k, v in model.named_buffers())
    if maximum_gradient <= 0 or maximum_projection_gradient <= 0 or source.model_digest(model) == initial or not unchanged:
        raise ValueError("Gradient or fixed-statistic check failed")
    scope = dict(fold=fold["fold"], seed=seed, arm=arm, device="cpu", updates=len(draws),
        pairs=sum(map(len, draws)), examples=sum(2 * len(b) for b in draws),
        trainable_parameters=sum(p.numel() for p in parameters), input_features=model.readout.in_features,
        initial_sha256=initial, final_sha256=source.model_digest(model), pair_draws_sha256=fingerprint(draws),
        fit_groups="|".join(fold["fit"]), held_groups="|".join(fold["held"]), initial_choice=prior_choice + 7,
        fixed_fit_buffers_unchanged=unchanged, maximum_gradient=maximum_gradient,
        maximum_projection_gradient=maximum_projection_gradient, training_seconds=perf_counter() - started)
    save_model(torch, path, dict(state=model.state_dict(), optimizer=optimizer.state_dict(), scope=scope, history=history), metadata)
    source.record(root, path, artifacts)
    return model.requires_grad_(False), scope, history


def reports(videos, stats, fold, seed, arm):
    errors, policies, frames, moment = previous.reports(videos, stats, fold, seed, arm)
    prior = np.asarray(stats["prior"], float)
    mean_score = np.stack([videos[g]["prediction"].mean(0) for g in fold["fit"]]).mean(0)
    for row in errors:
        video = videos[row["group"]]
        score = dict(trained=video["prediction"], prior_only=np.broadcast_to(prior, video["prediction"].shape),
                     fit_prediction_mean=np.broadcast_to(mean_score, video["prediction"].shape))[row["predictor"]]
        terms = common.decision_terms(video["labels"], score)
        row.update(regret_margin_loss=float(terms["bound"].mean()), mean_selection_regret=float(terms["regret"].mean()),
                   maximum_regret_bound_violation=float(terms["violation"].max()))
    for row in policies:
        row["mean_selected_layer"] = sum(k * row[f"L{k}_count"] for k in range(7,13)) / row["states"]
        row["actual_transformer_calls_measured"] = False
    for group, video in videos.items():
        terms = common.decision_terms(video["labels"], video["prediction"])
        selected = [r for r in frames if r["group"] == group]
        for i, row in enumerate(selected):
            row.update(regret_margin_bound=float(terms["bound"][i]), selection_regret=float(terms["regret"][i]),
                       regret_bound_violation=float(terms["violation"][i]))
    return errors, policies, frames, moment


def verify_results(root):
    d, p = source.read_sealed(root / "stage202_decision.json"), source.read_sealed(root / "manifest.json")
    if (d["schema"] != SCHEMA or d["plan_fingerprint"] != p["fingerprint"] or d["source_stage200"] != SOURCE200
            or d["training_runs"] != 8 or d["training_updates"] != 2048
            or d["training_state_references"] != 8192 or d["training_pair_uses"] != 4096 or d["pair_schedule_rows"] != 2048
            or d["predicted_states"] != 8 * p["states_per_model"]
            or d["status"] != "REVIEW_MATCHED_REGRET_MARGIN_ROUTING"
            or d["maximum_regret_bound_violation"] > 1e-12 or d["actual_transformer_calls_measured"]
            or any(d[k] != 0 for k in ("new_image_reads", "new_encoder_calls", "new_full_trajectories", "confirmation_groups_opened"))
            or set(d["table_sha256"]) != {f"stage202_{n}.csv" for n in TABLES}):
        raise ValueError("Incomplete Stage-202 comparison")
    for name, sha in (d["table_sha256"] | d["artifact_sha256"]).items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != sha:
            raise ValueError("Stage-202 artifact changed: " + name)
    for name, sha in p["code_sha256"].items():
        if file_hash(REPO / "tools" / name) != sha:
            raise ValueError("Stage-202 code changed: " + name)
    if file_hash(PROTOCOL_FILE) != p["protocol_sha256"]:
        raise ValueError("Stage-202 protocol changed")
    print("Stage-202 verified; REVIEW is not full-video adoption", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage194-root", type=Path)
    parser.add_argument("--stage200-bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-results", action="store_true")
    args = parser.parse_args(); root = args.output_dir.expanduser().resolve()
    if args.check_results:
        verify_results(root); return
    if args.stage194_root is None:
        parser.error("Stage-194 original feature directory required")
    r194, bundle = args.stage194_root.expanduser().resolve(), args.stage200_bundle.expanduser().resolve()
    protocol.safe_output(root, [r194, bundle.parent])
    with writer_lock(root):
        if (root / "stage202_decision.json").exists():
            verify_results(root); return
        p194, d194, reader, _, d200, old = load_sources(r194, bundle)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch
        torch.set_num_threads(1)
        plan = source.sealed(dict(schema=SCHEMA, source_stage194=d194["fingerprint"], source_stage200=SOURCE200,
            source_plan194=p194["fingerprint"], source_plan200=PLAN200, source200_bundle_sha256=file_hash(bundle),
            folds=p194["folds"], seeds=protocol.DESIGN["seeds"], arms=list(ARMS), training=protocol.DESIGN,
            states_per_model=sum(len(e["sample_frames"]) for e in p194["entries"]),
            pair_sampling="2 pairs/update; source uniform over fit; indices independently uniform with replacement; self pairs retained",
            objective="point mean(P(score-y)^2) versus mean(max_l(score_l-score_best+IoU_best-IoU_l)); exact true argmax",
            architecture="both frozen Stage-199 interaction_residual form; 6720 weights; same prior and fit-centered features",
            primary="held regret-minus-point, regret-minus-prior, and regret state-alignment gain",
            evaluation="cached native-parent immediate IoU, not AUC; historically exposed development sources",
            precision="float32 model training and exact target argmax at that precision; float64 decision-bound audit",
            compute_scope="router training/predictions only; layer ID is not executed Transformer block count or measured latency",
            device="cpu", torch=str(torch.__version__), numpy=str(np.__version__), python=sys.version,
            protocol_sha256=file_hash(PROTOCOL_FILE),
            code_sha256={Path(f).name: file_hash(f) for f in (__file__, common.__file__, common.pair_source.__file__, previous.__file__, bundle_source.__file__, architecture.__file__,
                inputs.__file__, metrics.__file__, protocol.__file__, previous.previous.__file__, previous.previous.diagnosis.__file__, architecture.pooling.__file__, source.__file__)}))
        publish_identical(root / "manifest.json", plan)
        n = plan["states_per_model"]
        progress = OverallProgress("Stage-202", 1 + len(p194["entries"]) + n + 8 * protocol.DESIGN["updates"] + 8 * n + 1, unit="work")
        progress.report("Verifying frozen feature, label and matched-pair sources"); progress.advance()
        rows, bank, artifacts = {k: [] for k in TABLES}, {}, {}
        old_labels = {}
        for r in old["frame_predictions"]:
            key = r["group"], int(r["frame"])
            y = np.asarray([float(r[f"label_L{k}"]) for k in range(7, 13)])
            if key in old_labels:
                np.testing.assert_array_equal(old_labels[key], y)
            old_labels[key] = y
        lens_z, shape = int(p194["runtime"]["tracker"]["lens_z"]), None
        for spec in p194["entries"]:
            group, seq = spec["group"], spec["sequence"]
            feature = reader.arrays(f"features/{seq}.npz", dict(plan_fingerprint=p194["fingerprint"], sequence=seq, kind="native_parent_prefix"))
            y = reader.arrays(f"reference/{group}.npz", dict(plan_fingerprint=p194["fingerprint"], kind="fixed_layer_fit_labels", group=group))["iou"]
            np.testing.assert_array_equal(feature["frame_id"], spec["sample_frames"]); metrics.validate_scores(y, y)
            tokens = np.asarray(feature["tokens"], np.float32)
            if tokens.ndim != 3 or len(tokens) != len(y) or not np.isfinite(tokens).all() or (shape is not None and tokens.shape[1:] != shape):
                raise ValueError("Feature layout changed")
            shape = tokens.shape[1:]
            np.testing.assert_array_equal(y, np.stack([old_labels[group, int(f)] for f in feature["frame_id"]]))
            rows["source_bridge"].append(dict(group=group, sequence=seq, states=len(y), maximum_label_error=0.))
            progress.advance(); pooled = []
            with torch.no_grad():
                for token in tokens:
                    pooled.append(architecture.pooling.pool(torch, torch.as_tensor(token[None]), lens_z).numpy()[0].copy()); progress.advance()
            bank[group] = dict(tokens=tokens, pooled=np.asarray(pooled), labels=np.asarray(y, float), frame_id=feature["frame_id"], sequence=seq)
        if len(bank) != 38 or sum(len(v["frame_id"]) for v in bank.values()) != len(old_labels):
            raise ValueError("Source state inventories disagree")
        progress.report("Matched MSE/regret CPU training and held-source predictions")
        for fold in p194["folds"]:
            stats = architecture.fit_statistics(bank, fold["fit"])
            old_fit = next(r for r in old["fit_summary"] if int(r["fold"]) == fold["fold"])
            np.testing.assert_allclose(stats["prior"], [float(old_fit[f"prior_L{k}"]) for k in range(7, 13)], atol=1e-12, rtol=0)
            path = root / "fit_statistics" / f"fold{fold['fold']}.npz"
            cache(path, dict(plan_fingerprint=plan["fingerprint"], fit=fold["fit"], fold=fold["fold"]), lambda: stats)
            source.record(root, path, artifacts)
            rows["fit_summary"].append(dict(fold=fold["fold"], fit_groups="|".join(fold["fit"]), held_groups="|".join(fold["held"]),
                fit_states=sum(len(bank[g]["frame_id"]) for g in fold["fit"]),
                **{f"prior_L{k}": float(stats["prior"][k - 7]) for k in range(7, 13)}, statistics_sha256=file_hash(path)))
            for seed in protocol.DESIGN["seeds"]:
                draws = common.pair_draws(fold, seed, {g: len(v["frame_id"]) for g, v in bank.items()})
                counts = {g: np.zeros(len(v["frame_id"]), int) for g, v in bank.items()}
                seen, self_counts, pair_counts = {g: set() for g in fold["fit"]}, dict.fromkeys(fold["fit"], 0), dict.fromkeys(fold["fit"], 0)
                for step, batch in enumerate(draws, 1):
                    for slot, (g, a, b) in enumerate(batch):
                        counts[g][a] += 1; counts[g][b] += 1
                        seen[g].update((a, b)); self_counts[g] += int(a == b); pair_counts[g] += 1
                        rows["pair_schedule"].append(dict(fold=fold["fold"], seed=seed, step=step, pair_slot=slot, group=g,
                            index_a=a, index_b=b, frame_a=int(bank[g]["frame_id"][a]), frame_b=int(bank[g]["frame_id"][b]), self_pair=a == b))
                for g in fold["fit"]:
                    rows["pair_support"].append(dict(fold=fold["fold"], seed=seed, group=g, pairs=pair_counts[g], self_pairs=self_counts[g],
                        state_references=int(counts[g].sum()), unique_states=len(seen[g])))
                scopes = []
                for arm in ARMS:
                    model, scope, history = fit(torch, root, plan, bank, stats, fold, seed, arm, draws, lens_z, progress, artifacts)
                    before = source.model_digest(model); videos = {}; maximum_error = 0.
                    for g, data in bank.items():
                        path = root / "predictions" / f"f{fold['fold']}__s{seed}__{arm}__{g}.npz"
                        meta = dict(plan_fingerprint=plan["fingerprint"], fold=fold["fold"], seed=seed, arm=arm, group=g, model_sha256=before)
                        value, reused = cache(path, meta, lambda: previous.previous.predict(torch, model, data, progress))
                        if reused: source.reused_work(progress, len(data["frame_id"]))
                        source.record(root, path, artifacts); metrics.validate_scores(data["labels"], value["prediction"])
                        np.testing.assert_array_equal(value["choice"], metrics.choose(value["prediction"]))
                        maximum_error = max(maximum_error, float(value["maximum_error"]))
                        videos[g] = dict(data, **value, counts=counts[g])
                    er, pr, fr, moment = reports(videos, stats, fold, seed, arm)
                    rows["error_video"].extend(er); rows["policy_video"].extend(pr); rows["frame_predictions"].extend(fr)
                    rows["model_scope"].append(dict(scope, **moment, prepooled_interface_maximum_error=maximum_error))
                    rows["training_history"].extend(history); scopes.append(scope)
                    if source.model_digest(model) != before:
                        raise ValueError("Evaluation changed frozen model")
                for field in ("initial_sha256", "initial_choice", "pair_draws_sha256", "examples", "pairs", "updates", "trainable_parameters", "input_features"):
                    if scopes[0][field] != scopes[1][field]: raise ValueError("Unmatched comparison: " + field)
        rows["policy_summary"] = common.aggregate(rows["policy_video"], "policy", POLICIES,
            ("mean_iou", "mean_regret", "modal_fraction", "good_fraction", "state_alignment_gain", "independent_mean_iou", "mean_selected_layer"))
        rows["error_summary"] = common.aggregate(rows["error_video"], "predictor", PREDICTORS,
            ("interaction_mse", "interaction_skill_vs_zero", "gap_weighted_pair_accuracy", "between_video_mse", "within_video_mse", "within_video_energy", "within_video_skill_vs_zero", "regret_margin_loss"))
        rows["paired"] = common.paired_reports(rows["policy_video"], rows["error_video"])
        for arm in ARMS:
            for pool in ("fit", "held"):
                for seed in protocol.DESIGN["seeds"]:
                    for policy in POLICIES:
                        cell = [r for r in rows["policy_video"] if r["arm"] == arm and r["pool"] == pool and r["seed"] == seed and r["policy"] == policy]
                        if len(cell) != 38: raise ValueError("Incomplete seed/source summary")
                        fields = ("mean_iou", "mean_regret", "state_alignment_gain", "mean_selected_layer")
                        rows["seed_summary"].append(dict(arm=arm,pool=pool,seed=seed,policy=policy,videos=38,
                            **{k: float(np.mean([r[k] for r in cell])) for k in fields}))
        hashes = {}
        for name in TABLES:
            path = root / f"stage202_{name}.csv"; write_csv(path, rows[name]); hashes[path.name] = file_hash(path)
        if progress.done != progress.total - 1: raise ValueError("Incomplete work accounting")
        primaries = [r for r in rows["paired"] if r["pool"] == "held" and r["arm"] == ARMS[1] and r["metric"] == "mean_iou"
                     and r["contrast"] in ("regret_minus_point", "trained_minus_prior_only", "trained_minus_independent_expectation")]
        decision = source.sealed(dict(schema=SCHEMA, status="REVIEW_MATCHED_REGRET_MARGIN_ROUTING",
            plan_fingerprint=plan["fingerprint"], source_stage194=d194["fingerprint"], source_stage200=d200["fingerprint"],
            source_groups=38, training_runs=len(rows["model_scope"]), training_updates=len(rows["training_history"]), predicted_states=len(rows["frame_predictions"]),
            training_state_references=sum(r["examples"] for r in rows["model_scope"]), training_pair_uses=sum(r["pairs"] for r in rows["model_scope"]),
            pair_schedule_rows=len(rows["pair_schedule"]), new_image_reads=0, new_encoder_calls=0, new_full_trajectories=0, confirmation_groups_opened=0,
            primary_contrasts=primaries, source_verified_artifacts=reader.verified, table_sha256=hashes, artifact_sha256=artifacts,
            maximum_regret_bound_violation=max(r["maximum_regret_bound_violation"] for r in rows["error_video"]),
            actual_transformer_calls_measured=False,
            next_step="Require held point/prior and alignment gains, source/seed consistency, then trajectory validation; no automatic adoption"))
        publish_identical(root / "stage202_decision.json", decision)
        progress.phase = "Complete"; progress.advance(); progress.report(); verify_results(root)


if __name__ == "__main__":
    main()
