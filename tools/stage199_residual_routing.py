"""Matched CPU training of fit-centered layer-interaction routing, Stage-199."""
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
import stage195_router_objective_audit as inputs
import stage195_routing_common as metrics
import stage194_module_common as protocol
import stage198_routing_calibration as diagnosis
import stage199_residual_model as architecture

source = inputs.source
ARMS = architecture.ARMS
SCHEMA = "stage199_fit_centered_interaction_training_v1"
SOURCE198 = "447f9394ced978f2c3586b28b52ca89327d1c2cb05533b7990147f4732d93e34"
PLAN198 = "11d884a49bc372bdb81da4463e13be7849807c482fe9a7070273f37cae17b0c3"
BUNDLE198 = "0f3bcebd12370081e9e8a2319bb12458a537a1cdffe095491c422f2a2024ae77"
DEFAULT_BUNDLE = REPO / "docs/experiments/06_sgla_uav_tracking/results/stage198_results_447f9394ced9.txt"
POLICIES = ("trained", "prior_only", "fit_prediction_mean", "native_L9", "oracle")
PREDICTORS = ("trained", "prior_only", "fit_prediction_mean")
TABLES = ("model_scope", "fit_summary", "source_bridge", "error_summary", "error_video", "policy_summary",
          "policy_video", "paired", "frame_predictions", "training_history")
PROTOCOL_FILE = REPO / "docs/experiments/06_sgla_uav_tracking/STAGE199_FIT_CENTERED_INTERACTION.md"


def load198(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != BUNDLE198:
        raise ValueError("Need the exact reviewed Stage-198 result bundle")
    matches = list(re.finditer(rb"(?:^|\r?\n)===== ([\w.]+) =====\r?\n", raw))
    parts = {}
    for i, m in enumerate(matches):
        name = m[1].decode()
        if name in parts:
            raise ValueError("Duplicate source section")
        parts[name] = raw[m.end():matches[i + 1].start() if i + 1 < len(matches) else len(raw)]
    d, p = json.loads(parts["stage198_decision.json"]), json.loads(parts["manifest.json"])
    for value in (d, p):
        if value["fingerprint"] != fingerprint({k: v for k, v in value.items() if k != "fingerprint"}):
            raise ValueError("Source fingerprint mismatch")
    if (d["fingerprint"] != SOURCE198 or p["fingerprint"] != PLAN198 or d["plan_fingerprint"] != PLAN198
            or d["evaluated_states"] != 21888 or d["source_groups"] != 38 or d["models_audited"] != 12
            or d["confirmation_groups_opened"] != 0):
        raise ValueError("Unexpected reviewed Stage-198 run")
    if set(parts) != {"stage198_decision.json", "manifest.json", *d["table_sha256"]}:
        raise ValueError("Incomplete source bundle")
    tables = {}
    for name, sha in d["table_sha256"].items():
        data = parts[name]
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError("Stage-198 source table changed: " + name)
        tables[name.removeprefix("stage198_").removesuffix(".csv")] = list(csv.DictReader(io.StringIO(data.decode())))
    for name in ("stage198_routing_calibration.py", "stage194_module_common.py", "stage195_routing_common.py"):
        if p["code_sha256"].get(name) != file_hash(REPO / "tools" / name):
            raise ValueError("Frozen analysis code changed: " + name)
    return d, tables


def load_sources(root194, bundle):
    p194, d194, reader, tables194 = inputs.load_source(root194)
    d198, tables198 = load198(bundle)
    return p194, d194, reader, tables194, d198, tables198


def cache(path, metadata, compute):
    receipt = path.with_suffix(".receipt.json")
    if receipt.exists():
        r = source.read_sealed(receipt)
        if r["schema"] != SCHEMA or r["metadata"] != metadata or r["sha256"] != file_hash(path):
            raise ValueError("Committed Stage-199 cache changed")
    if path.exists():
        with np.load(path, allow_pickle=False) as f:
            if json.loads(str(f["metadata_json"].item())) != metadata:
                raise ValueError("Unidentified Stage-199 cache")
            if receipt.exists():
                return {k: f[k] for k in f.files if k != "metadata_json"}, True
    data = compute()
    write_cache(path, data, metadata)
    publish_identical(receipt, source.sealed(dict(schema=SCHEMA, metadata=metadata, sha256=file_hash(path))))
    return data, False


def save_model(torch, path, payload, metadata):
    receipt = path.with_suffix(".receipt.json")
    if receipt.exists():
        raise ValueError("Never overwrite a committed model")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and torch.load(path, map_location="cpu", weights_only=True).get("metadata") != metadata:
        raise ValueError("Unidentified orphan model")
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload, metadata=metadata), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    publish_identical(receipt, source.sealed(dict(schema=SCHEMA, metadata=metadata, sha256=file_hash(path))))


def fit(torch, root, plan, bank, stats, fold, seed, arm, lens_z, progress, artifacts):
    model = architecture.build(torch, stats, lens_z, arm).eval()
    initial = source.model_digest(model)
    draws = protocol.draws_for(fold, seed, {g: len(v["frame_id"]) for g, v in bank.items()})
    metadata = dict(plan_fingerprint=plan["fingerprint"], fold=fold["fold"], seed=seed, arm=arm,
                    fit=fold["fit"], held=fold["held"], initial_sha256=initial, draws_sha256=fingerprint(draws),
                    updates=protocol.DESIGN["updates"], examples=len(draws) * protocol.DESIGN["batch"])
    path = root / "training" / f"fold{fold['fold']}" / f"seed{seed}" / (arm + ".pt")
    receipt = path.with_suffix(".receipt.json")
    if receipt.exists():
        r = source.read_sealed(receipt)
        if r["schema"] != SCHEMA or r["metadata"] != metadata or r["sha256"] != file_hash(path):
            raise ValueError("Committed model identity changed")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["metadata"] != metadata:
            raise ValueError("Model metadata mismatch")
        model.load_state_dict(payload["state"], strict=True)
        if source.model_digest(model) != payload["scope"]["final_sha256"]:
            raise ValueError("Model state mismatch")
        source.reused_work(progress, len(draws))
        source.record(root, path, artifacts)
        return model.requires_grad_(False), payload["scope"], payload["history"]
    fixed_buffers = {k: v.clone() for k, v in model.named_buffers()}
    with torch.no_grad():
        sample = bank[fold["fit"][0]]
        initial_score = model(torch.as_tensor(sample["tokens"][:1], dtype=torch.float32),
                              torch.as_tensor(sample["pooled"][:1], dtype=torch.float32)).numpy()
    prior_choice = int(metrics.choose(np.asarray(stats["prior"])[None])[0])
    if int(metrics.choose(initial_score)[0]) != prior_choice:
        raise ValueError("Initialization does not retain the common fit prior")
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=protocol.DESIGN["learning_rate"], weight_decay=protocol.DESIGN["weight_decay"])
    history, maximum_gradient, projection_gradient = [], 0., 0.
    started = perf_counter()
    for step, batch in enumerate(draws, 1):
        if any(g not in fold["fit"] or g in fold["held"] for g, _ in batch):
            raise ValueError("Held source entered training")
        token = torch.as_tensor(np.stack([bank[g]["tokens"][i] for g, i in batch]), dtype=torch.float32)
        pooled = torch.as_tensor(np.stack([bank[g]["pooled"][i] for g, i in batch]), dtype=torch.float32)
        labels = torch.as_tensor(np.stack([bank[g]["labels"][i] for g, i in batch]), dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        loss, terms = model.objective(token, pooled, labels)
        loss.backward()
        pg = float(model.projection.weight.grad.norm())
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, protocol.DESIGN["gradient_clip"], error_if_nonfinite=True))
        if not np.isfinite([pg, norm, float(loss.detach())]).all():
            raise ValueError("Nonfinite update")
        maximum_gradient, projection_gradient = max(maximum_gradient, norm), max(projection_gradient, pg)
        optimizer.step()
        history.append(dict(fold=fold["fold"], seed=seed, arm=arm, step=step, optimized_loss=float(loss.detach()),
                            gradient_norm=norm, projection_gradient_norm=pg,
                            **{k: float(v.detach()) for k, v in terms.items()}))
        progress.advance()
    unchanged = all(torch.equal(v, fixed_buffers[k]) for k, v in model.named_buffers())
    if maximum_gradient <= 0 or projection_gradient <= 0 or source.model_digest(model) == initial or not unchanged:
        raise ValueError("Model failed gradient or fixed-prior checks")
    scope = dict(fold=fold["fold"], seed=seed, arm=arm, device="cpu", updates=len(draws),
        examples=sum(map(len, draws)), trainable_parameters=sum(p.numel() for p in parameters),
        input_features=model.readout.in_features, initial_sha256=initial, final_sha256=source.model_digest(model),
        draws_sha256=fingerprint(draws), fit_groups="|".join(fold["fit"]), held_groups="|".join(fold["held"]),
        fixed_fit_buffers_unchanged=unchanged, initial_choice=prior_choice + 7,
        initial_score_prior_max_error=float(np.abs(initial_score[0] - stats["prior"]).max()),
        maximum_gradient=maximum_gradient, maximum_projection_gradient=projection_gradient,
        training_seconds=perf_counter() - started)
    save_model(torch, path, dict(state=model.state_dict(), optimizer=optimizer.state_dict(), scope=scope, history=history), metadata)
    source.record(root, path, artifacts)
    return model.requires_grad_(False), scope, history


def predict(torch, model, data, progress):
    prediction, raw_values, error = [], [], 0.
    with torch.no_grad():
        for token, pooled in zip(data["tokens"], data["pooled"], strict=True):
            x = torch.as_tensor(token[None], dtype=torch.float32)
            g = torch.as_tensor(pooled[None], dtype=torch.float32)
            score = model(x, g)
            error = max(error, float((score - model(x)).abs().max()))
            prediction.append(score.numpy()[0].copy())
            raw_values.append(model.raw(x, g).numpy()[0].copy())
            progress.advance()
    if error > 1e-7:
        raise ValueError("Prepooled and direct interfaces disagree")
    p = np.asarray(prediction)
    return dict(prediction=p, choice=metrics.choose(p), raw=np.asarray(raw_values), maximum_error=np.asarray(error))


def model_reports(videos, stats, fold, seed, arm):
    er, pr, fr = [], [], []
    mean_score = np.stack([videos[g]["prediction"].mean(0) for g in fold["fit"]]).mean(0)
    prior = np.asarray(stats["prior"], float)
    for group, video in videos.items():
        y, p, a = video["labels"], video["prediction"], video["choice"]
        base = dict(fold=fold["fold"], seed=seed, arm=arm, group=group, sequence=video["sequence"],
                    pool="fit" if group in fold["fit"] else "held")
        scores = dict(trained=p, prior_only=np.broadcast_to(prior, p.shape), fit_prediction_mean=np.broadcast_to(mean_score, p.shape))
        for predictor, value in scores.items():
            residual = value - prior
            target = y - prior
            z, estimate = target - target.mean(1, keepdims=True), residual - residual.mean(1, keepdims=True)
            loss, energy = float(np.mean((estimate - z) ** 2)), float(np.mean(z ** 2))
            er.append(dict(base, predictor=predictor, interaction_mse=loss, interaction_energy=energy,
                interaction_skill_vs_zero=1 - loss / energy if energy else None, **metrics.errors(y, value)))
        actions = dict(trained=a, prior_only=metrics.choose(scores["prior_only"]),
                       fit_prediction_mean=metrics.choose(scores["fit_prediction_mean"]), native_L9=np.full(len(y), 2), oracle=metrics.choose(y))
        for policy in POLICIES:
            pr.append(dict(base, policy=policy, states=len(y), **diagnosis.correspondence(y, actions[policy])))
        for i, frame in enumerate(video["frame_id"]):
            fr.append(dict(base, frame=int(frame), selected_layer=int(a[i]) + 7, prior_layer=int(actions["prior_only"][i]) + 7,
                selected_iou=float(y[i, a[i]]), prior_iou=float(y[i, actions["prior_only"][i]]),
                training_draw_count=int(video["counts"][i]) if group in fold["fit"] else None,
                **{f"prediction_L{k}": float(p[i, k - 7]) for k in range(7, 13)},
                **{f"raw_L{k}": float(video["raw"][i, k - 7]) for k in range(7, 13)},
                **{f"label_L{k}": float(y[i, k - 7]) for k in range(7, 13)}))
    mean_raw = np.stack([videos[g]["raw"].astype(float).mean(0) for g in fold["fit"]]).mean(0)
    moment = dict(fit_raw_mean_max_error=float(np.abs(mean_raw).max()),
                  fit_score_prior_mean_max_error=float(np.abs(mean_score - prior).max()))
    if moment["fit_raw_mean_max_error"] > 1e-5:
        raise ValueError("Fit-centered readout has a nonzero column mean")
    if arm != "absolute_sigmoid" and moment["fit_score_prior_mean_max_error"] > 1e-5:
        raise ValueError("Signed score changed the fixed fit prior mean")
    return er, pr, fr, moment


def aggregate(rows, category, names, fields):
    output = []
    for arm in ARMS:
        for pool in ("fit", "held"):
            for name in names:
                selected = [r for r in rows if r["arm"] == arm and r["pool"] == pool and r[category] == name]
                groups = sorted({r["group"] for r in selected})
                if len(selected) != 76 or len(groups) != 38:
                    raise ValueError("Incomplete source summary")
                row = dict(arm=arm, pool=pool, **{category: name}, videos=38)
                for field in fields:
                    values = []
                    for group in groups:
                        cells = [r[field] for r in selected if r["group"] == group]
                        if all(v is not None for v in cells):
                            values.append(float(np.mean(cells)))
                    row[field] = float(np.mean(values)) if values else None
                    row[field + "_supported_videos"] = len(values)
                output.append(row)
    return output


def paired(rows, errors):
    output = []
    for pool in ("fit", "held"):
        def compare(arm_a, policy_a, arm_b, policy_b, metric, table, kind, contrast, direction="positive"):
            groups = sorted({r["group"] for r in table})
            delta = []
            for group in groups:
                get = lambda arm, name: [r[metric] for r in table if r["arm"] == arm and r["pool"] == pool
                    and r["group"] == group and r.get("policy", r.get("predictor")) == name]
                a, b = get(arm_a, policy_a), get(arm_b, policy_b)
                if len(a) != 2 or len(b) != 2:
                    raise ValueError("Incomplete paired source/seed grid")
                if all(v is not None for v in a + b):
                    delta.append(float(np.mean(a) - np.mean(b)))
            stats = protocol.contrast(delta) if len(delta) > 1 else dict(videos=len(delta), mean_delta=None,
                median_delta=None, ci_low=None, ci_high=None, positive=0, negative=0, worst_leave_one_video_out=None)
            output.append(dict(arm=arm_a, pool=pool, kind=kind, contrast=contrast, metric=metric,
                               favorable_direction=direction, **stats))
        for arm in ARMS:
            for control in ("prior_only", "fit_prediction_mean", "native_L9"):
                for metric in ("mean_iou", "state_alignment_gain"):
                    compare(arm, "trained", arm, control, metric, rows, "within", "trained_minus_" + control)
            selected = [r for r in rows if r["arm"] == arm and r["pool"] == pool and r["policy"] == "trained"]
            delta = [float(np.mean([r["state_alignment_gain"] for r in selected if r["group"] == group]))
                     for group in sorted({r["group"] for r in selected})]
            output.append(dict(arm=arm, pool=pool, kind="alignment", contrast="trained_minus_independent_expectation",
                metric="mean_iou", favorable_direction="positive", **protocol.contrast(delta)))
        for a, b in (("interaction_residual", "absolute_additive"), ("interaction_residual", "absolute_sigmoid"),
                     ("absolute_additive", "absolute_sigmoid")):
            for metric in ("mean_iou", "state_alignment_gain"):
                compare(a, "trained", b, "trained", metric, rows, "between", a + "_minus_" + b)
            for metric, direction in (("interaction_mse", "negative"), ("gap_weighted_pair_accuracy", "positive")):
                compare(a, "trained", b, "trained", metric, errors, "prediction", a + "_minus_" + b, direction)
    return output


def verify_results(root):
    d, p = source.read_sealed(root / "stage199_decision.json"), source.read_sealed(root / "manifest.json")
    if (d["schema"] != SCHEMA or d["plan_fingerprint"] != p["fingerprint"] or d["source_stage198"] != SOURCE198
            or d["status"] != "REVIEW_FIT_CENTERED_INTERACTION_TRAINING" or d["training_runs"] != 12
            or d["training_updates"] != 3072 or d["predicted_states"] != 12 * p["states_per_model"]
            or any(d[k] != 0 for k in ("new_image_reads", "new_encoder_calls", "new_full_trajectories", "confirmation_groups_opened"))
            or set(d["table_sha256"]) != {f"stage199_{n}.csv" for n in TABLES}):
        raise ValueError("Incomplete Stage-199")
    for name, sha in (d["table_sha256"] | d["artifact_sha256"]).items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != sha:
            raise ValueError("Stage-199 artifact changed: " + name)
    print("Stage-199 verified; REVIEW is not full-video adoption", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage194-root", type=Path)
    parser.add_argument("--stage198-bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-results", action="store_true")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        verify_results(root)
        return
    if args.stage194_root is None:
        parser.error("Stage-194 frozen feature directory required")
    r194, bundle = args.stage194_root.expanduser().resolve(), args.stage198_bundle.expanduser().resolve()
    protocol.safe_output(root, [r194, bundle.parent])
    with writer_lock(root):
        if (root / "stage199_decision.json").exists():
            verify_results(root)
            return
        p194, d194, reader, tables194, d198, tables198 = load_sources(r194, bundle)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch
        torch.set_num_threads(1)
        plan = source.sealed(dict(schema=SCHEMA, source_stage194=d194["fingerprint"], source_stage198=SOURCE198,
            source_plan194=p194["fingerprint"], source_plan198=PLAN198, source198_bundle_sha256=file_hash(bundle),
            folds=p194["folds"], arms=list(ARMS), seeds=protocol.DESIGN["seeds"], training=protocol.DESIGN,
            states_per_model=sum(len(e["sample_frames"]) for e in p194["entries"]), device="cpu",
            torch=str(torch.__version__), numpy=str(np.__version__), python=sys.version,
            feature_centering="fit E[token] and E[pooled stats] separately, equal source weights; zero biases",
            score="sigmoid(logit(mu)+r), mu+r, mu+P(r); r is the same centered ordered-token readout",
            objectives="absolute MSE for controls; centered layer error MSE for interaction_residual",
            primary="held residual-minus-prior, residual alignment, and residual-minus-absolute-additive",
            interval_scope="nominal historical development-source bootstrap; no adoption gate",
            protocol_sha256=file_hash(PROTOCOL_FILE),
            code_sha256={Path(f).name: file_hash(f) for f in (__file__, architecture.__file__, inputs.__file__,
                metrics.__file__, protocol.__file__, diagnosis.__file__, architecture.pooling.__file__, source.__file__)}))
        publish_identical(root / "manifest.json", plan)
        views = plan["states_per_model"]
        progress = OverallProgress("Stage-199", 1 + len(p194["entries"]) + views + 12 * protocol.DESIGN["updates"] + 12 * views + 1, unit="work")
        progress.report("Verifying frozen features, labels and Stage-198 correspondence source")
        progress.advance()
        bank, artifacts, rows = {}, {}, {n: [] for n in TABLES}
        old_labels = {}
        for r in tables198["frame_predictions"]:
            key = r["group"], int(r["frame"])
            y = np.asarray([float(r[f"label_L{k}"]) for k in range(7, 13)])
            if key in old_labels:
                np.testing.assert_array_equal(old_labels[key], y)
            old_labels[key] = y
        lens_z, token_shape = int(p194["runtime"]["tracker"]["lens_z"]), None
        for spec in p194["entries"]:
            seq, group = spec["sequence"], spec["group"]
            feature = reader.arrays(f"features/{seq}.npz", dict(plan_fingerprint=p194["fingerprint"], sequence=seq, kind="native_parent_prefix"))
            y = reader.arrays(f"reference/{group}.npz", dict(plan_fingerprint=p194["fingerprint"], kind="fixed_layer_fit_labels", group=group))["iou"]
            np.testing.assert_array_equal(feature["frame_id"], spec["sample_frames"])
            metrics.validate_scores(y, y)
            tokens = np.asarray(feature["tokens"], np.float32)
            if tokens.ndim != 3 or len(tokens) != len(y) or not np.isfinite(tokens).all():
                raise ValueError("Invalid prefix feature bank")
            if token_shape is not None and tokens.shape[1:] != token_shape:
                raise ValueError("Cross-source token layouts differ")
            token_shape = tokens.shape[1:]
            old_y = np.stack([old_labels[group, int(f)] for f in feature["frame_id"]])
            np.testing.assert_array_equal(y, old_y)
            rows["source_bridge"].append(dict(group=group, sequence=seq, states=len(y), maximum_label_error=float(np.abs(y - old_y).max())))
            progress.advance()
            pooled = []
            with torch.no_grad():
                for token in tokens:
                    pooled.append(architecture.pooling.pool(torch, torch.as_tensor(token[None]), lens_z).numpy()[0].copy())
                    progress.advance()
            bank[group] = dict(tokens=tokens, pooled=np.asarray(pooled), labels=np.asarray(y, float),
                               frame_id=feature["frame_id"], sequence=seq)
        if len(bank) != 38 or sum(len(v["frame_id"]) for v in bank.values()) != len(old_labels):
            raise ValueError("Source state inventories disagree")
        progress.report("Matched CPU routing training and held-source evaluation")
        for fold in p194["folds"]:
            stats = architecture.fit_statistics(bank, fold["fit"])
            old_priors = [r for r in tables198["calibration"] if int(r["fold"]) == fold["fold"]]
            for r in old_priors:
                if abs(float(r["fit_label_mean"]) - stats["prior"][int(r["layer"]) - 7]) > 1e-12:
                    raise ValueError("Fit prior differs from Stage-198")
            statpath = root / "fit_statistics" / f"fold{fold['fold']}.npz"
            _, _ = cache(statpath, dict(plan_fingerprint=plan["fingerprint"], fit=fold["fit"], fold=fold["fold"]), lambda: stats)
            source.record(root, statpath, artifacts)
            rows["fit_summary"].append(dict(fold=fold["fold"], fit_groups="|".join(fold["fit"]), held_groups="|".join(fold["held"]),
                fit_states=sum(len(bank[g]["frame_id"]) for g in fold["fit"]),
                **{f"prior_L{k}": float(stats["prior"][k - 7]) for k in range(7, 13)}, statistics_sha256=file_hash(statpath)))
            for seed in protocol.DESIGN["seeds"]:
                draw = protocol.draws_for(fold, seed, {g: len(v["frame_id"]) for g, v in bank.items()})
                old_scope = next(r for r in tables194["scope_summary"] if int(r["fold"]) == fold["fold"]
                                 and int(r["seed"]) == seed and r["arm"] == metrics.ARM)
                if fingerprint(draw) != old_scope["draws_sha256"]:
                    raise ValueError("Draw budget differs from reviewed Stage-194")
                scopes = []
                for arm in ARMS:
                    model, scope, history = fit(torch, root, plan, bank, stats, fold, seed, arm, lens_z, progress, artifacts)
                    before = source.model_digest(model)
                    counts = {g: np.zeros(len(v["frame_id"]), int) for g, v in bank.items()}
                    for batch in draw:
                        for group, i in batch:
                            counts[group][i] += 1
                    videos = {}
                    maximum_error = 0.
                    for group, data in bank.items():
                        path = root / "predictions" / f"f{fold['fold']}__s{seed}__{arm}__{group}.npz"
                        meta = dict(plan_fingerprint=plan["fingerprint"], fold=fold["fold"], seed=seed, arm=arm, group=group, model_sha256=before)
                        value, reused = cache(path, meta, lambda: predict(torch, model, data, progress))
                        if reused:
                            source.reused_work(progress, len(data["frame_id"]))
                        source.record(root, path, artifacts)
                        metrics.validate_scores(data["labels"], value["prediction"])
                        np.testing.assert_array_equal(value["choice"], metrics.choose(value["prediction"]))
                        maximum_error = max(maximum_error, float(value["maximum_error"]))
                        videos[group] = dict(data, **value, counts=counts[group])
                    er, pr, fr, moment = model_reports(videos, stats, fold, seed, arm)
                    rows["error_video"].extend(er); rows["policy_video"].extend(pr); rows["frame_predictions"].extend(fr)
                    rows["model_scope"].append(dict(scope, **moment, prepooled_interface_maximum_error=maximum_error))
                    rows["training_history"].extend(history)
                    scopes.append(scope)
                    if source.model_digest(model) != before:
                        raise ValueError("Evaluation changed trained weights or fit buffers")
                for field in ("initial_sha256", "initial_choice", "draws_sha256", "trainable_parameters", "input_features"):
                    if len({s[field] for s in scopes}) != 1:
                        raise ValueError("Unmatched comparison: " + field)
        rows["policy_summary"] = aggregate(rows["policy_video"], "policy", POLICIES,
            ("mean_iou", "mean_regret", "modal_fraction", "good_fraction", "state_alignment_gain", "independent_mean_iou"))
        rows["error_summary"] = aggregate(rows["error_video"], "predictor", PREDICTORS,
            ("mse", "relative_mse", "interaction_mse", "interaction_skill_vs_zero", "gap_weighted_pair_accuracy"))
        rows["paired"] = paired(rows["policy_video"], rows["error_video"])
        hashes = {}
        for name in TABLES:
            path = root / f"stage199_{name}.csv"
            write_csv(path, rows[name]); hashes[path.name] = file_hash(path)
        if progress.done != progress.total - 1:
            raise ValueError("Incomplete work accounting")
        primaries = [r for r in rows["paired"] if r["pool"] == "held" and r["arm"] == "interaction_residual"
            and r["metric"] == "mean_iou" and r["contrast"] in ("trained_minus_prior_only", "trained_minus_independent_expectation", "interaction_residual_minus_absolute_additive")]
        decision = source.sealed(dict(schema=SCHEMA, status="REVIEW_FIT_CENTERED_INTERACTION_TRAINING",
            plan_fingerprint=plan["fingerprint"], source_stage194=d194["fingerprint"], source_stage198=d198["fingerprint"],
            training_runs=len(rows["model_scope"]), training_updates=len(rows["training_history"]), predicted_states=len(rows["frame_predictions"]),
            source_groups=38, device="cpu", new_image_reads=0, new_encoder_calls=0, new_full_trajectories=0, confirmation_groups_opened=0,
            primary_contrasts=primaries, source_verified_artifacts=reader.verified, table_sha256=hashes, artifact_sha256=artifacts,
            next_step="Require held prior/control and state-alignment gains, seed/source consistency, then normal-trajectory validation; no automatic adoption"))
        publish_identical(root / "stage199_decision.json", decision)
        progress.phase = "Complete"; progress.advance(); progress.report()
        verify_results(root)


if __name__ == "__main__":
    main()
