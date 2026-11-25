"""CPU matched same-video point versus difference supervision; no image inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
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
import stage199_residual_routing as previous
import stage200_pair_common as common

source, inputs, architecture, metrics, protocol = previous.source, previous.inputs, previous.architecture, previous.metrics, previous.protocol
ARMS = common.ARMS
SCHEMA = "stage200_matched_within_video_difference_v1"
SOURCE199 = "6e98e01801e9f63f34d1a6d97ffb769a34d45566e698e608817805dfa41f10a5"
PLAN199 = "0b73e5c12d53685cae7228f94baea461231cf915503f9f3416fda1d5afbd1e06"
BUNDLE199 = "12fe41d3e3643e937bbabe922f0253f53788185fb89a82095e9182ecf0ade8df"
DEFAULT_BUNDLE = REPO / "docs/experiments/06_sgla_uav_tracking/results/stage199_results_20261125T113420Z_1204543.txt"
PROTOCOL_FILE = REPO / "docs/experiments/06_sgla_uav_tracking/STAGE200_WITHIN_VIDEO_DIFFERENCE.md"
TABLES = ("model_scope", "fit_summary", "source_bridge", "pair_schedule", "pair_support", "error_summary", "error_video",
          "policy_summary", "policy_video", "paired", "frame_predictions", "training_history")


def load199(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != BUNDLE199:
        raise ValueError("Need the exact reviewed Stage-199 bundle")
    ms = list(re.finditer(rb"(?:^|\r?\n)===== ([\w.]+) =====\r?\n", raw))
    parts = {}
    for i, m in enumerate(ms):
        name = m[1].decode()
        if name in parts:
            raise ValueError("Duplicate source section")
        parts[name] = raw[m.end():ms[i + 1].start() if i + 1 < len(ms) else len(raw)]
    d, p = json.loads(parts["stage199_decision.json"]), json.loads(parts["manifest.json"])
    for value in (d, p):
        if value["fingerprint"] != fingerprint({k: v for k, v in value.items() if k != "fingerprint"}):
            raise ValueError("Source fingerprint changed")
    if (d["fingerprint"] != SOURCE199 or p["fingerprint"] != PLAN199 or d["plan_fingerprint"] != PLAN199
            or d["training_runs"] != 12 or d["training_updates"] != 3072 or d["predicted_states"] != 21888
            or d["confirmation_groups_opened"] != 0):
        raise ValueError("Unexpected Stage-199 source")
    if set(parts) != {"stage199_decision.json", "manifest.json", *d["table_sha256"]}:
        raise ValueError("Source inventory changed")
    tables = {}
    for name, sha in d["table_sha256"].items():
        b = parts[name]
        candidates = [b, b[:-1] if b.endswith(b"\n") else b, b[:-2] if b.endswith(b"\r\n") else b]
        value = next((v for v in candidates if hashlib.sha256(v).hexdigest() == sha), None)
        if value is None:
            raise ValueError("Stage-199 table changed: " + name)
        tables[name.removeprefix("stage199_").removesuffix(".csv")] = list(csv.DictReader(io.StringIO(value.decode())))
    for name, sha in p["code_sha256"].items():
        if file_hash(REPO / "tools" / name) != sha:
            raise ValueError("Frozen Stage-199 code changed: " + name)
    if p["protocol_sha256"] != file_hash(previous.PROTOCOL_FILE):
        raise ValueError("Frozen Stage-199 protocol changed")
    return d, tables


def load_sources(root194, bundle):
    p194, d194, reader, tables194 = inputs.load_source(root194)
    d199, tables199 = load199(bundle)
    if d199["source_stage194"] != d194["fingerprint"]:
        raise ValueError("Stage-194 source differs")
    return p194, d194, reader, tables194, d199, tables199


def cache(path, metadata, compute):
    receipt = path.with_suffix(".receipt.json")
    if receipt.exists():
        r = source.read_sealed(receipt)
        if r["schema"] != SCHEMA or r["metadata"] != metadata or r["sha256"] != file_hash(path):
            raise ValueError("Committed Stage-200 cache changed")
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
        raise ValueError("Unknown point/difference arm")
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
            point_mse=float(terms["point"].detach()), half_difference_mse=float(terms["within_video_difference"].detach())))
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
    errors, policies, frames, moment = previous.model_reports(videos, stats, fold, seed, arm)
    mean_score = np.stack([videos[g]["prediction"].mean(0) for g in fold["fit"]]).mean(0)
    prior = np.asarray(stats["prior"], float)
    for row in errors:
        video = videos[row["group"]]
        p = dict(trained=video["prediction"], prior_only=np.broadcast_to(prior, video["prediction"].shape),
                 fit_prediction_mean=np.broadcast_to(mean_score, video["prediction"].shape))[row["predictor"]]
        row.update(common.decomposition(video["labels"], p, prior))
    return errors, policies, frames, moment


def verify_results(root):
    d, p = source.read_sealed(root / "stage200_decision.json"), source.read_sealed(root / "manifest.json")
    if (d["schema"] != SCHEMA or d["plan_fingerprint"] != p["fingerprint"] or d["source_stage199"] != SOURCE199
            or d["training_runs"] != 8 or d["training_updates"] != 2048
            or d["training_state_references"] != 8192 or d["training_pair_uses"] != 4096 or d["pair_schedule_rows"] != 2048
            or d["predicted_states"] != 8 * p["states_per_model"]
            or d["status"] != "REVIEW_MATCHED_WITHIN_VIDEO_SUPERVISION"
            or any(d[k] != 0 for k in ("new_image_reads", "new_encoder_calls", "new_full_trajectories", "confirmation_groups_opened"))
            or set(d["table_sha256"]) != {f"stage200_{n}.csv" for n in TABLES}):
        raise ValueError("Incomplete Stage-200 comparison")
    for name, sha in (d["table_sha256"] | d["artifact_sha256"]).items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != sha:
            raise ValueError("Stage-200 artifact changed: " + name)
    print("Stage-200 verified; REVIEW is not full-video adoption", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage194-root", type=Path)
    parser.add_argument("--stage199-bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-results", action="store_true")
    args = parser.parse_args(); root = args.output_dir.expanduser().resolve()
    if args.check_results:
        verify_results(root); return
    if args.stage194_root is None:
        parser.error("Stage-194 original feature directory required")
    r194, bundle = args.stage194_root.expanduser().resolve(), args.stage199_bundle.expanduser().resolve()
    protocol.safe_output(root, [r194, bundle.parent])
    with writer_lock(root):
        if (root / "stage200_decision.json").exists():
            verify_results(root); return
        p194, d194, reader, _, d199, old = load_sources(r194, bundle)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch
        torch.set_num_threads(1)
        plan = source.sealed(dict(schema=SCHEMA, source_stage194=d194["fingerprint"], source_stage199=SOURCE199,
            source_plan194=p194["fingerprint"], source_plan199=PLAN199, source199_bundle_sha256=file_hash(bundle),
            folds=p194["folds"], seeds=protocol.DESIGN["seeds"], arms=list(ARMS), training=protocol.DESIGN,
            states_per_model=sum(len(e["sample_frames"]) for e in p194["entries"]),
            pair_sampling="2 pairs/update; source uniform over fit; indices independently uniform with replacement; self pairs retained",
            objective="point mean(e^2) versus half mean((e_t-e_s)^2); e=P(score-y)",
            architecture="both frozen Stage-199 interaction_residual form; 6720 weights; same prior and fit-centered features",
            primary="held difference-minus-point, difference-minus-prior, and difference alignment gain",
            evaluation="cached native-parent immediate IoU, not AUC; historically exposed development sources",
            device="cpu", torch=str(torch.__version__), numpy=str(np.__version__), python=sys.version,
            protocol_sha256=file_hash(PROTOCOL_FILE),
            code_sha256={Path(f).name: file_hash(f) for f in (__file__, common.__file__, previous.__file__, architecture.__file__,
                inputs.__file__, metrics.__file__, protocol.__file__, previous.diagnosis.__file__, architecture.pooling.__file__, source.__file__)}))
        publish_identical(root / "manifest.json", plan)
        n = plan["states_per_model"]
        progress = OverallProgress("Stage-200", 1 + len(p194["entries"]) + n + 8 * protocol.DESIGN["updates"] + 8 * n + 1, unit="work")
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
        progress.report("Matched point/difference CPU training and held-source predictions")
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
                        value, reused = cache(path, meta, lambda: previous.predict(torch, model, data, progress))
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
        rows["policy_summary"] = common.aggregate(rows["policy_video"], "policy", previous.POLICIES,
            ("mean_iou", "mean_regret", "modal_fraction", "good_fraction", "state_alignment_gain", "independent_mean_iou"))
        rows["error_summary"] = common.aggregate(rows["error_video"], "predictor", previous.PREDICTORS,
            ("interaction_mse", "interaction_skill_vs_zero", "gap_weighted_pair_accuracy", "between_video_mse", "within_video_mse", "within_video_energy", "within_video_skill_vs_zero"))
        rows["paired"] = common.paired_reports(rows["policy_video"], rows["error_video"])
        hashes = {}
        for name in TABLES:
            path = root / f"stage200_{name}.csv"; write_csv(path, rows[name]); hashes[path.name] = file_hash(path)
        if progress.done != progress.total - 1: raise ValueError("Incomplete work accounting")
        primaries = [r for r in rows["paired"] if r["pool"] == "held" and r["arm"] == ARMS[1] and r["metric"] == "mean_iou"
                     and r["contrast"] in ("difference_minus_point", "trained_minus_prior_only", "trained_minus_independent_expectation")]
        decision = source.sealed(dict(schema=SCHEMA, status="REVIEW_MATCHED_WITHIN_VIDEO_SUPERVISION",
            plan_fingerprint=plan["fingerprint"], source_stage194=d194["fingerprint"], source_stage199=d199["fingerprint"],
            source_groups=38, training_runs=len(rows["model_scope"]), training_updates=len(rows["training_history"]), predicted_states=len(rows["frame_predictions"]),
            training_state_references=sum(r["examples"] for r in rows["model_scope"]), training_pair_uses=sum(r["pairs"] for r in rows["model_scope"]),
            pair_schedule_rows=len(rows["pair_schedule"]), new_image_reads=0, new_encoder_calls=0, new_full_trajectories=0, confirmation_groups_opened=0,
            primary_contrasts=primaries, source_verified_artifacts=reader.verified, table_sha256=hashes, artifact_sha256=artifacts,
            next_step="Require held point/prior and alignment gains, source/seed consistency, then trajectory validation; no automatic adoption"))
        publish_identical(root / "stage200_decision.json", decision)
        progress.phase = "Complete"; progress.advance(); progress.report(); verify_results(root)


if __name__ == "__main__":
    main()
