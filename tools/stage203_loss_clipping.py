"""Matched loss-by-gradient-clipping factorial training on frozen L6 caches."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
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
import stage202_regret_routing as previous
import stage199_residual_routing as predictor
import stage203_clipping_common as common

source, inputs, architecture, metrics, protocol = previous.source, previous.inputs, previous.architecture, previous.metrics, previous.protocol
ARMS = common.ARMS
SCHEMA = "stage203_loss_by_gradient_clipping_v1"
SOURCE202 = "5048f7fe1c427f22b2f3e8c61533abe8b5cfeb70d362536a6e96acaa976602a4"
PLAN202 = "715da640896f6884dc930515174d62815d839d0520d7c9c9b48ee188a7e47616"
BUNDLE202 = "8f0b46df691d9c9a1027dc96af56bf699431f470b58027451742f536371f3562"
DEFAULT_BUNDLE = REPO / "docs/experiments/06_sgla_uav_tracking/results/stage202_results_20261125T154651Z_1417349.txt"
PROTOCOL_FILE = REPO / "docs/experiments/06_sgla_uav_tracking/STAGE203_LOSS_CLIPPING_FACTORIAL.md"
POLICIES, PREDICTORS = previous.POLICIES, previous.PREDICTORS
TABLES = (*previous.TABLES, "control_replay")


def load202(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != BUNDLE202:
        raise ValueError("Need the exact reviewed Stage-202 result bundle")
    matches = list(re.finditer(rb"(?:^|\r?\n)===== ([\w.]+) =====\r?\n", raw))
    parts = {}
    for i, m in enumerate(matches):
        name = m[1].decode()
        if name in parts: raise ValueError("Duplicate source section")
        parts[name] = raw[m.end():matches[i+1].start() if i+1 < len(matches) else len(raw)]
    expected = {"stage202_decision.json", "manifest.json", *(f"stage202_{n}.csv" for n in previous.TABLES)}
    if set(parts) != expected: raise ValueError("Source inventory changed")
    d, p = json.loads(parts["stage202_decision.json"]), json.loads(parts["manifest.json"])
    for value in (d,p):
        if value.get("fingerprint") != fingerprint({k:v for k,v in value.items() if k != "fingerprint"}):
            raise ValueError("Source fingerprint changed")
    if (d["fingerprint"] != SOURCE202 or p["fingerprint"] != PLAN202 or d["plan_fingerprint"] != PLAN202
            or d["training_runs"] != 8 or d["predicted_states"] != 14592
            or d["actual_transformer_calls_measured"] or d["maximum_regret_bound_violation"] > 1e-12
            or any(d[k] != 0 for k in ("new_image_reads","new_encoder_calls","new_full_trajectories","confirmation_groups_opened"))
            or set(d["table_sha256"]) != expected - {"stage202_decision.json","manifest.json"}):
        raise ValueError("Unexpected Stage-202 identity or scope")
    tables = {}
    for name, sha in d["table_sha256"].items():
        b = parts[name]
        candidates = [b, b[:-1] if b.endswith(b"\n") else b, b[:-2] if b.endswith(b"\r\n") else b]
        data = next((v for v in candidates if hashlib.sha256(v).hexdigest() == sha), None)
        if data is None: raise ValueError("Source table checksum mismatch: " + name)
        tables[name.removeprefix("stage202_").removesuffix(".csv")] = list(csv.DictReader(io.StringIO(data.decode())))
    for name, sha in p["code_sha256"].items():
        if file_hash(REPO / "tools" / name) != sha: raise ValueError("Frozen source code changed: " + name)
    if file_hash(previous.PROTOCOL_FILE) != p["protocol_sha256"]: raise ValueError("Frozen source protocol changed")
    return d, p, tables


def load_sources(root194, bundle):
    p194, d194, reader, tables194 = inputs.load_source(root194)
    d202, p202, tables202 = load202(bundle)
    if d202["source_stage194"] != d194["fingerprint"] or p202["folds"] != p194["folds"]:
        raise ValueError("Stage-194 source or split differs from reviewed Stage-202")
    return p194, d194, reader, tables194, d202, p202, tables202


def fail_nonfinite(root, metadata, step, field):
    path = root / "diagnostics" / "nonfinite_failure.json"
    publish_identical(path, source.sealed(dict(schema=SCHEMA,status="FAILED_NONFINITE_TRAINING",metadata=metadata,step=int(step),field=field)))
    raise FloatingPointError("Stage-203 stopped after nonfinite " + field + "; diagnostic: " + str(path))


def cache(path, metadata, compute):
    receipt = path.with_suffix(".receipt.json")
    if receipt.exists():
        r = source.read_sealed(receipt)
        if r["schema"] != SCHEMA or r["metadata"] != metadata or r["sha256"] != file_hash(path):
            raise ValueError("Committed Stage-203 cache changed")
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
    objective, clipping = common.flags(arm)
    model = architecture.build(torch, stats, lens_z, "interaction_residual").eval()
    initial = source.model_digest(model)
    metadata = dict(plan_fingerprint=plan["fingerprint"],fold=fold["fold"],seed=seed,arm=arm,objective=objective,clipping=clipping,
        fit=fold["fit"],held=fold["held"],initial_sha256=initial,pair_draws_sha256=fingerprint(draws),updates=len(draws),examples=sum(2*len(b) for b in draws))
    if (root / "diagnostics/nonfinite_failure.json").exists():
        raise ValueError("A nonfinite failure is recorded; do not continue this output path")
    path = root / "training" / f"fold{fold['fold']}" / f"seed{seed}" / (arm + ".pt")
    receipt = path.with_suffix(".receipt.json")
    if receipt.exists():
        r = source.read_sealed(receipt)
        if r["schema"] != SCHEMA or r["metadata"] != metadata or r["sha256"] != file_hash(path): raise ValueError("Committed model changed")
        saved = torch.load(path,map_location="cpu",weights_only=True)
        if saved["metadata"] != metadata: raise ValueError("Checkpoint metadata mismatch")
        model.load_state_dict(saved["state"],strict=True)
        if source.model_digest(model) != saved["scope"]["final_sha256"]: raise ValueError("Model state mismatch")
        source.reused_work(progress,len(draws));source.record(root,path,artifacts)
        return model.requires_grad_(False),saved["scope"],saved["history"]
    frozen = {k:v.clone() for k,v in model.named_buffers()}
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters,lr=protocol.DESIGN["learning_rate"],weight_decay=protocol.DESIGN["weight_decay"])
    history, maximum_gradient, maximum_projection_gradient = [],0.,0.
    started = perf_counter()
    for step,batch in enumerate(draws,1):
        if len(batch) != 2 or any(g not in fold["fit"] or g in fold["held"] for g,_,_ in batch): raise ValueError("Invalid pair budget or held training")
        positions = common.flatten(batch)
        tokens = torch.as_tensor(np.stack([bank[g]["tokens"][i] for g,i in positions]),dtype=torch.float32)
        pooled = torch.as_tensor(np.stack([bank[g]["pooled"][i] for g,i in positions]),dtype=torch.float32)
        labels = torch.as_tensor(np.stack([bank[g]["labels"][i] for g,i in positions]),dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        terms = common.objectives(torch,model(tokens,pooled),labels)
        loss = terms[objective]
        if not all(bool(torch.isfinite(v)) for v in terms.values()): fail_nonfinite(root,metadata,step,"loss")
        loss.backward()
        pg = float(model.projection.weight.grad.norm())
        if not np.isfinite(pg): fail_nonfinite(root,metadata,step,"projection_gradient")
        try: gradient = common.gradient_rule(torch,parameters,clipping)
        except FloatingPointError: fail_nonfinite(root,metadata,step,"gradient")
        maximum_gradient = max(maximum_gradient,gradient["gradient_norm"])
        maximum_projection_gradient = max(maximum_projection_gradient,pg)
        before = [p.detach().clone() for p in parameters]
        before_norm = float(common.tensor_norm(torch,before))
        if not np.isfinite(before_norm): fail_nonfinite(root,metadata,step,"parameter_norm")
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in parameters): fail_nonfinite(root,metadata,step,"parameters")
        if any(not bool(torch.isfinite(v).all()) for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)):
            fail_nonfinite(root,metadata,step,"optimizer_state")
        update_norm = float(common.tensor_norm(torch,[p.detach()-b for p,b in zip(parameters,before)]))
        if not np.isfinite(update_norm): fail_nonfinite(root,metadata,step,"parameter_update")
        history.append(dict(fold=fold["fold"],seed=seed,arm=arm,objective=objective,clipping=clipping,step=step,
            optimized_loss=float(loss.detach()),projection_gradient_norm=pg,**gradient,parameter_update_norm=update_norm,
            relative_parameter_update_norm=update_norm/before_norm if before_norm else None,self_pairs=sum(a==b for _,a,b in batch),
            point_mse=float(terms["point"].detach()),regret_margin_loss=float(terms["regret_margin"].detach())))
        progress.advance()
    unchanged = all(torch.equal(v,frozen[k]) for k,v in model.named_buffers())
    if maximum_gradient<=0 or maximum_projection_gradient<=0 or source.model_digest(model)==initial or not unchanged:
        raise ValueError("Gradient or fixed-statistic check failed")
    scope = dict(fold=fold["fold"],seed=seed,arm=arm,objective=objective,clipping=clipping,
        gradient_clip=protocol.DESIGN["gradient_clip"] if clipping else None,device="cpu",updates=len(draws),
        pairs=sum(map(len,draws)),examples=sum(2*len(b) for b in draws),trainable_parameters=sum(p.numel() for p in parameters),input_features=model.readout.in_features,
        initial_sha256=initial,final_sha256=source.model_digest(model),pair_draws_sha256=fingerprint(draws),
        fit_groups="|".join(fold["fit"]),held_groups="|".join(fold["held"]),initial_choice=int(metrics.choose(np.asarray(stats["prior"])[None])[0])+7,
        fixed_fit_buffers_unchanged=unchanged,maximum_gradient=maximum_gradient,maximum_projection_gradient=maximum_projection_gradient,
        clipped_updates=sum(r["gradient_was_clipped"] for r in history),threshold_exceeded_updates=sum(r["inherited_threshold_exceeded"] for r in history),
        mean_gradient_norm=float(np.mean([r["gradient_norm"] for r in history])),mean_post_rule_gradient_norm=float(np.mean([r["gradient_norm_after"] for r in history])),
        mean_parameter_update_norm=float(np.mean([r["parameter_update_norm"] for r in history])),maximum_parameter_update_norm=max(r["parameter_update_norm"] for r in history),
        training_seconds=perf_counter()-started)
    save_model(torch,path,dict(state=model.state_dict(),optimizer=optimizer.state_dict(),scope=scope,history=history),metadata)
    source.record(root,path,artifacts)
    return model.requires_grad_(False),scope,history


def reports(videos,stats,fold,seed,arm):
    return previous.reports(videos,stats,fold,seed,arm)


def verify_results(root):
    if (root / "diagnostics/nonfinite_failure.json").exists(): raise ValueError("Recorded nonfinite training failure")
    d, p = source.read_sealed(root / "stage203_decision.json"), source.read_sealed(root / "manifest.json")
    if (d["schema"] != SCHEMA or d["plan_fingerprint"] != p["fingerprint"] or d["source_stage202"] != SOURCE202
            or d["training_runs"] != 16 or d["training_updates"] != 4096
            or d["training_state_references"] != 16384 or d["training_pair_uses"] != 8192 or d["pair_schedule_rows"] != 2048
            or d["predicted_states"] != 16 * p["states_per_model"]
            or d["status"] != "REVIEW_LOSS_BY_GRADIENT_CLIPPING"
            or d["maximum_regret_bound_violation"] > 1e-12 or d["actual_transformer_calls_measured"]
            or any(d[k] != 0 for k in ("new_image_reads", "new_encoder_calls", "new_full_trajectories", "confirmation_groups_opened"))
            or set(d["table_sha256"]) != {f"stage203_{n}.csv" for n in TABLES}):
        raise ValueError("Incomplete Stage-203 comparison")
    for name, sha in (d["table_sha256"] | d["artifact_sha256"]).items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != sha:
            raise ValueError("Stage-203 artifact changed: " + name)
    for name, sha in p["code_sha256"].items():
        if file_hash(REPO / "tools" / name) != sha:
            raise ValueError("Stage-203 code changed: " + name)
    if file_hash(PROTOCOL_FILE) != p["protocol_sha256"]:
        raise ValueError("Stage-203 protocol changed")
    print("Stage-203 verified; REVIEW is not full-video adoption", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage194-root", type=Path)
    parser.add_argument("--stage202-bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-results", action="store_true")
    args = parser.parse_args(); root = args.output_dir.expanduser().resolve()
    if args.check_results:
        verify_results(root); return
    if args.stage194_root is None:
        parser.error("Stage-194 original feature directory required")
    r194, bundle = args.stage194_root.expanduser().resolve(), args.stage202_bundle.expanduser().resolve()
    protocol.safe_output(root, [r194, bundle.parent])
    with writer_lock(root):
        if (root / "diagnostics/nonfinite_failure.json").exists(): raise ValueError("Recorded nonfinite failure; use a new path only after review")
        if (root / "stage203_decision.json").exists():
            verify_results(root); return
        p194, d194, reader, _, d202, p202, old = load_sources(r194, bundle)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch
        torch.set_num_threads(1)
        plan = source.sealed(dict(schema=SCHEMA, source_stage194=d194["fingerprint"], source_stage202=SOURCE202,
            source_plan194=p194["fingerprint"], source_plan202=PLAN202, source202_bundle_sha256=file_hash(bundle),
            folds=p194["folds"], seeds=protocol.DESIGN["seeds"], arms=list(ARMS), training=protocol.DESIGN,
            states_per_model=sum(len(e["sample_frames"]) for e in p194["entries"]),
            pair_sampling="2 pairs/update; source uniform over fit; indices independently uniform with replacement; self pairs retained",
            objective="frozen Stage-202 point and regret_margin; factorial clip=1 versus untouched gradients",
            architecture="all four frozen Stage-199 interaction_residual form; 6720 weights; same prior and fit-centered features",
            primary="held regret no-clip effects versus clipped regret, no-clip point and prior; alignment and loss-by-clipping interaction",
            evaluation="cached native-parent immediate IoU, not AUC; historically exposed development sources",
            precision="float32 model training and exact target argmax at that precision; float64 decision-bound audit",
            compute_scope="router training/predictions only; layer ID is not executed Transformer block count or measured latency",
            device="cpu", torch=str(torch.__version__), numpy=str(np.__version__), python=sys.version,
            protocol_sha256=file_hash(PROTOCOL_FILE),
            code_sha256={**p202["code_sha256"],Path(__file__).name:file_hash(__file__),Path(common.__file__).name:file_hash(common.__file__)}))
        publish_identical(root / "manifest.json", plan)
        n = plan["states_per_model"]
        progress = OverallProgress("Stage-203", 1 + len(p194["entries"]) + n + 16 * protocol.DESIGN["updates"] + 16 * n + 1, unit="work")
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
        progress.report("Matched loss-by-clipping CPU training and held-source predictions")
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
                        value, reused = cache(path, meta, lambda: predictor.predict(torch, model, data, progress))
                        if reused: source.reused_work(progress, len(data["frame_id"]))
                        source.record(root, path, artifacts); metrics.validate_scores(data["labels"], value["prediction"])
                        np.testing.assert_array_equal(value["choice"], metrics.choose(value["prediction"]))
                        maximum_error = max(maximum_error, float(value["maximum_error"]))
                        videos[g] = dict(data, **value, counts=counts[g])
                    er, pr, fr, moment = reports(videos, stats, fold, seed, arm)
                    rows["error_video"].extend(er); rows["policy_video"].extend(pr); rows["frame_predictions"].extend(fr)
                    rows["model_scope"].append(dict(scope, **moment, prepooled_interface_maximum_error=maximum_error))
                    rows["training_history"].extend(history); scopes.append(scope)
                    objective, clipping = common.flags(arm)
                    if clipping:
                        old_scope = next(r for r in old["model_scope"] if int(r["fold"])==fold["fold"] and int(r["seed"])==seed and r["arm"]==objective)
                        old_predictions = {(r["group"],int(r["frame"])):r for r in old["frame_predictions"] if int(r["fold"])==fold["fold"] and int(r["seed"])==seed and r["arm"]==objective}
                        score_error, action_changes = 0.,0
                        for group, video in videos.items():
                            for i, frame in enumerate(video["frame_id"]):
                                ref = old_predictions[group,int(frame)]
                                score_error = max(score_error,float(np.abs(video["prediction"][i]-np.asarray([float(ref[f"prediction_L{k}"]) for k in range(7,13)])).max()))
                                action_changes += int(video["choice"][i]+7 != int(ref["selected_layer"]))
                        rows["control_replay"].append(dict(fold=fold["fold"],seed=seed,arm=arm,source_arm=objective,
                            source_final_sha256=old_scope["final_sha256"],final_sha256=scope["final_sha256"],
                            final_state_sha_matches=scope["final_sha256"]==old_scope["final_sha256"],maximum_score_error=score_error,action_changed_states=action_changes))
                    if source.model_digest(model) != before:
                        raise ValueError("Evaluation changed frozen model")
                for field in ("initial_sha256", "initial_choice", "pair_draws_sha256", "examples", "pairs", "updates", "trainable_parameters", "input_features"):
                    if any(scopes[0][field] != item[field] for item in scopes[1:]): raise ValueError("Unmatched comparison: " + field)
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
            path = root / f"stage203_{name}.csv"; write_csv(path, rows[name]); hashes[path.name] = file_hash(path)
        if progress.done != progress.total - 1: raise ValueError("Incomplete work accounting")
        primaries = [r for r in rows["paired"] if r["pool"]=="held" and r["metric"]=="mean_iou" and
            ((r["arm"]=="regret_no_clip" and r["contrast"] in ("regret_no_clip_minus_clip","regret_minus_point_no_clip","trained_minus_prior_only","trained_minus_independent_expectation"))
             or (r["arm"]=="factorial" and r["contrast"]=="loss_by_clipping_interaction"))]
        decision = source.sealed(dict(schema=SCHEMA, status="REVIEW_LOSS_BY_GRADIENT_CLIPPING",
            plan_fingerprint=plan["fingerprint"], source_stage194=d194["fingerprint"], source_stage202=d202["fingerprint"],
            source_groups=38, training_runs=len(rows["model_scope"]), training_updates=len(rows["training_history"]), predicted_states=len(rows["frame_predictions"]),
            training_state_references=sum(r["examples"] for r in rows["model_scope"]), training_pair_uses=sum(r["pairs"] for r in rows["model_scope"]),
            pair_schedule_rows=len(rows["pair_schedule"]), new_image_reads=0, new_encoder_calls=0, new_full_trajectories=0, confirmation_groups_opened=0,
            primary_contrasts=primaries, source_verified_artifacts=reader.verified, table_sha256=hashes, artifact_sha256=artifacts,
            maximum_regret_bound_violation=max(r["maximum_regret_bound_violation"] for r in rows["error_video"]),
            actual_transformer_calls_measured=False,
            next_step="Review loss-by-clipping interaction and no-clip practical controls; no automatic adoption or threshold search"))
        publish_identical(root / "stage203_decision.json", decision)
        progress.phase = "Complete"; progress.advance(); progress.report(); verify_results(root)


if __name__ == "__main__":
    main()
