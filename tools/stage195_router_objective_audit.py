"""CPU-only audit of completed Stage-194 R-only models and cached six-layer labels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
for folder in (REPO, REPO / "tools", REPO / "tracking"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import numpy as np

from experiment_progress import OverallProgress
from stage27_uav_collect import writer_lock
from stage27_uav_common import publish_identical
from stage4_common import file_hash, fingerprint, write_csv
import stage194_module_ablation as source
import stage194_module_common as previous
import stage194_module_model as model_code
import stage195_routing_common as audit

SCHEMA = audit.SCHEMA
SOURCE194 = "aecfc2112422a0cf5f1af3daa1ddc61bcce9f54fc4c3f2ae94f762429a9dd547"
SOURCE_PLAN = "69ce5be3cf224c940d203698d07049760a4018e9928dee31270babdd9f8a27e0"
TABLES = ("model_scope", "error_summary", "error_video", "policy_summary", "policy_video",
          "paired", "frame_predictions", "fixed_path_audit", "training_history")


class VerifiedSource:
    def __init__(self, root, decision):
        self.root, self.decision, self.verified = Path(root), decision, {}

    def file(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Source artifact escapes root")
        declared = self.decision["artifact_sha256"] | self.decision["table_sha256"]
        if relative not in declared:
            raise ValueError("Artifact not bound by Stage-194: " + relative)
        if relative not in self.verified:
            actual = file_hash(path)
            if actual != declared[relative]:
                raise ValueError("Source artifact hash changed: " + relative)
            self.verified[relative] = actual
        return path

    def arrays(self, relative, metadata):
        path = self.file(relative)
        receipt = source.read_sealed(self.file(Path(relative).with_suffix(".receipt.json").as_posix()))
        if receipt["metadata"] != metadata or receipt["sha256"] != self.verified[relative]:
            raise ValueError("Unexpected source receipt: " + relative)
        with np.load(path, allow_pickle=False) as f:
            if json.loads(str(f["metadata_json"].item())) != metadata:
                raise ValueError("Unexpected source NPZ identity: " + relative)
            return {k: f[k] for k in f.files if k != "metadata_json"}

    def table(self, name):
        with self.file(f"stage194_{name}.csv").open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))


def load_source(root):
    decision = source.read_sealed(root / "stage194_decision.json")
    plan = source.read_sealed(root / "manifest.json")
    if (decision["fingerprint"] != SOURCE194 or plan["fingerprint"] != SOURCE_PLAN
            or decision["plan_fingerprint"] != plan["fingerprint"] or decision["schema"] != previous.SCHEMA
            or decision["full_trajectories"] != 798 or decision["training_runs"] != 36
            or decision["confirmation_groups_opened"] != 0 or len(plan["entries"]) != 38):
        raise ValueError("Need the completed, reviewed Stage-194 source")
    for name in ("tools/stage194_module_model.py", "tools/stage194_module_common.py", "tools/stage194_module_ablation.py"):
        if plan["code_sha256"].get(name) != file_hash(REPO / name):
            raise ValueError("Original Stage-194 implementation changed: " + name)
    reader = VerifiedSource(root, decision)
    tables = {name: reader.table(name) for name in source.TABLES}
    if len(tables["seed_video_summary"]) != 798 or len(tables["scope_summary"]) != 36:
        raise ValueError("Incomplete Stage-194 source tables")
    return plan, decision, reader, tables


def load_router(torch, reader, plan, fold, seed, dimension, lens_z, scope):
    relative = f"training/fold{fold['fold']}/seed{seed}/{audit.ARM}.pt"
    path = reader.file(relative)
    receipt = source.read_sealed(reader.file(Path(relative).with_suffix(".receipt.json").as_posix()))
    meta = receipt["metadata"]
    if (meta["plan_fingerprint"] != plan["fingerprint"] or meta["arm"] != audit.ARM
            or meta["fold"] != fold["fold"] or meta["seed"] != seed
            or meta["fit"] != fold["fit"] or meta["held"] != fold["held"] or meta["updates"] != 256
            or set(meta["features"]) != set(fold["fit"]) or receipt["sha256"] != reader.verified[relative]):
        raise ValueError("Model fit scope changed")
    for group, digest in meta["features"].items():
        spec = next(e for e in plan["entries"] if e["group"] == group)
        key = f"features/{spec['sequence']}.npz"
        if reader.verified.get(key) != digest:
            raise ValueError("Model is not bound to the verified fit feature pool")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["metadata"] != meta:
        raise ValueError("Model metadata disagrees")
    compare_scope(payload["scope"], scope)
    model = model_code.build_modules(torch, dimension, lens_z, audit.ARM).eval()
    model.load_state_dict(payload["state"], strict=True)
    model.requires_grad_(False)
    if source.model_digest(model) != scope["state_sha256"]:
        raise ValueError("Model parameter digest changed")
    # R-only is what makes all true branch labels available without an encoder.
    if any(torch.count_nonzero(p).item() for name in ("adapter", "quality") for p in getattr(model, name).parameters()):
        raise ValueError("R-only checkpoint contains a nonzero A/Q intervention")
    return model, payload, relative


def compare_scope(payload_scope, table_scope):
    """CSV strings are compared with JSON-native payload values explicitly."""
    if set(payload_scope) != set(table_scope):
        raise ValueError("Source model scope fields changed")
    for key, expected in payload_scope.items():
        actual = table_scope[key]
        if isinstance(expected, bool):
            valid = actual == str(expected)
        elif isinstance(expected, (int, float)):
            valid = float(actual) == expected
        else:
            valid = actual == expected
        if not valid:
            raise ValueError("Source scope disagrees: " + key)


def predict_video(torch, model, tokens, progress):
    prediction, choice = [], []
    with torch.no_grad():
        # Keep the original batch-one invocation. CPU/GPU reduction or linear
        # roundoff can still affect near ties; archived action parity is reported.
        for token in tokens:
            score = model.route(torch.as_tensor(token[None], dtype=torch.float32))
            prediction.append(score.numpy()[0].copy())
            choice.append(int(torch.topk(score, 1, dim=1).indices[0, 0]))
            progress.advance()
    return np.asarray(prediction), np.asarray(choice, int)


def make_model_reports(videos, controls, fold, seed):
    error_rows, policy_rows, frame_rows = [], [], []
    for group, value in videos.items():
        y, p, choice = value["labels"], value["prediction"], value["choice"]
        pool = "fit" if group in fold["fit"] else "held"
        base = dict(fold=fold["fold"], seed=seed, sequence=value["sequence"], group=group, pool=pool)
        predictors = dict(trained=p, initial_half=np.full_like(p, .5),
                          **{k: np.broadcast_to(v, p.shape) for k, v in controls.items()})
        for name, scores in predictors.items():
            error_rows.append(dict(base, predictor=name, **audit.errors(y, scores)))
        choices = dict(trained=choice, native_L9=np.full(len(y), audit.NATIVE),
                       fit_label_mean=audit.choose(predictors["fit_label_mean"]),
                       fit_prediction_mean=audit.choose(predictors["fit_prediction_mean"]), oracle=audit.choose(y),
                       uniform_expectation=None)
        if "fit_draw_label_mean" in predictors:
            choices["fit_draw_label_mean"] = audit.choose(predictors["fit_draw_label_mean"])
        for name, selected in choices.items():
            policy_rows.append(dict(base, policy=name, states=len(y), **audit.policy_metrics(y, selected)))
        sorted_score = np.sort(p, axis=1)
        for i, frame in enumerate(value["frame_id"]):
            gpu = value["archived_choice"][i] if value["archived_choice"] is not None else None
            frame_rows.append(dict(base, frame=int(frame), cpu_layer=int(choice[i]) + 7,
                archived_gpu_layer=int(gpu) + 7 if gpu is not None else None,
                cpu_gpu_action_equal=bool(choice[i] == gpu) if gpu is not None else None,
                training_draw_count=int(value["draw_count"][i]) if pool == "fit" else None,
                predicted_margin=float(sorted_score[i, -1] - sorted_score[i, -2]),
                true_best_second_gap=float(np.sort(y[i])[-1] - np.sort(y[i])[-2]),
                selected_iou=float(y[i, choice[i]]), native_iou=float(y[i, audit.NATIVE]),
                **{f"prediction_L{k}": float(p[i, k - 7]) for k in range(7, 13)},
                **{f"label_L{k}": float(y[i, k - 7]) for k in range(7, 13)}))
    return error_rows, policy_rows, frame_rows


def average_rows(rows, selector, field):
    selected = [r for r in rows if selector(r)]
    values = [r[field] for r in selected if r[field] is not None]
    return float(np.mean(values)) if values else None


def pool_error_summary(videos, controls, fold, seed, draws=None):
    """Use the full pool for variance decomposition, including between-video difficulty."""
    result = []
    pools = [("fit", fold["fit"], None), ("held", fold["held"], None)]
    if draws is not None:
        pools.append(("fit_draws", fold["fit"], [(g, i) for batch in draws for g, i in batch]))
        for pool, seen in (("fit_seen_unique", True), ("fit_unseen", False)):
            positions = [(g, i) for g in fold["fit"] for i, count in enumerate(videos[g]["draw_count"]) if bool(count) == seen]
            if positions:
                pools.append((pool, sorted({g for g, _ in positions}), positions))
    for pool, groups, positions in pools:
        if positions is None:
            y = np.concatenate([videos[g]["labels"] for g in groups])
            p = np.concatenate([videos[g]["prediction"] for g in groups])
        else:
            y = np.stack([videos[g]["labels"][i] for g, i in positions])
            p = np.stack([videos[g]["prediction"][i] for g, i in positions])
        predictors = dict(trained=p, initial_half=np.full_like(p, .5),
                          **{k: np.broadcast_to(v, p.shape) for k, v in controls.items()})
        for name, scores in predictors.items():
            result.append(dict(fold=fold["fold"], seed=seed, pool=pool, predictor=name,
                               videos=len(groups), **audit.errors(y, scores)))
    return result


def summarize_policies(rows):
    result, paired = [], []
    for pool in ("fit", "held"):
        for name in audit.POLICIES:
            selected = [r for r in rows if r["pool"] == pool and r["policy"] == name]
            if len(selected) != 76:
                raise ValueError("Incomplete two-seed policy pool")
            result.append(dict(pool=pool, policy=name, videos=38,
                               **{k: average_rows(selected, lambda r: True, k)
                                  for k in ("mean_iou", "mean_regret", "gain_vs_L9", "good_fraction", "modal_fraction", "entropy_bits")}))
        for control in ("native_L9", "fit_label_mean", "fit_prediction_mean", "fit_draw_label_mean"):
            for metric in ("mean_iou", "mean_regret"):
                delta = []
                for group in sorted({r["group"] for r in rows}):
                    get = lambda name: average_rows(rows, lambda r: r["pool"] == pool and r["group"] == group and r["policy"] == name, metric)
                    delta.append(get("trained") - get(control))
                paired.append(dict(pool=pool, contrast="trained_minus_" + control, metric=metric,
                                   favorable_direction="negative" if metric == "mean_regret" else "positive",
                                   **previous.contrast(delta)))
    return result, paired


def full_path_audit(reader, plan, tables, fold, seed, group, model_sha):
    spec = next(e for e in plan["entries"] if e["group"] == group)
    seq = spec["sequence"]
    fixed_layer = int(next(r["layer"] for r in tables["fixed_layers"] if int(r["fold"]) == fold["fold"]))

    def read(arm, own_seed, digest=None, forced=None):
        relative = f"evaluation/{seq}__s{own_seed}__{arm}.npz"
        meta = dict(plan_fingerprint=plan["fingerprint"], sequence=seq, fold=fold["fold"], seed=own_seed,
                    arm=arm, model_sha256=digest, fixed_layer=forced)
        return reader.arrays(relative, meta)

    learned = read(audit.ARM, seed, model_sha)
    native, fixed = read("native", 0), read("fit_best_fixed", 0, forced=fixed_layer)
    source.audit_trace(learned, learned["parent"][0], spec["tracked_frames"])
    checks = audit.path_comparison(learned, {9: native, fixed_layer: fixed})
    matches = [r for r in tables["seed_video_summary"] if r["sequence"] == seq and r["arm"] == audit.ARM and int(r["seed"]) == seed]
    control = next(r for r in tables["seed_video_summary"] if r["sequence"] == seq and r["arm"] == "native")
    if len(matches) != 1:
        raise ValueError("Incomplete full-video source cell")
    return dict(fold=fold["fold"], seed=seed, sequence=seq, group=group, frames=spec["tracked_frames"],
                auc=float(matches[0]["auc"]), native_auc=float(control["auc"]),
                auc_delta=float(matches[0]["auc"]) - float(control["auc"]), **checks)


def verify_results(root):
    result = source.read_sealed(root / "stage195_decision.json")
    plan = source.read_sealed(root / "manifest.json")
    if (result["schema"] != SCHEMA or result["plan_fingerprint"] != plan["fingerprint"]
            or result["status"] != "COMPLETE_FROZEN_ROUTER_OBJECTIVE_AUDIT"
            or result["source_stage194"] != SOURCE194 or result["training_updates"] != 0
            or result["new_image_reads"] != 0 or result["new_encoder_calls"] != 0
            or result["confirmation_groups_opened"] != 0 or result["audited_models"] != 4
            or result["source_groups"] != 38 or result["full_path_audits"] != 76
            or result["predicted_states"] != 4 * plan["states_per_model"]):
        raise ValueError("Incomplete Stage-195 audit")
    if set(result["table_sha256"]) != {f"stage195_{k}.csv" for k in TABLES}:
        raise ValueError("Incomplete Stage-195 tables")
    for name, sha in result["table_sha256"].items():
        if file_hash(root / name) != sha:
            raise ValueError("Changed audit result: " + name)
    print("Stage-195 cache-only audit artifacts verified", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-results", action="store_true")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        verify_results(root)
        return
    if args.input is None:
        parser.error("--input must identify the completed Stage-194 directory")
    old = args.input.expanduser().resolve()
    previous.safe_output(root, [old])
    with writer_lock(root):
        if (root / "stage195_decision.json").exists():
            verify_results(root)
            return
        p194, d194, reader, tables194 = load_source(old)
        import torch

        torch.set_num_threads(1)
        plan = source.sealed(dict(schema=SCHEMA, source_stage194=SOURCE194, source_plan=SOURCE_PLAN,
            source_decision_sha256=file_hash(old / "stage194_decision.json"), source_manifest_sha256=file_hash(old / "manifest.json"),
            arm=audit.ARM, folds=p194["folds"], seeds=previous.DESIGN["seeds"], device="cpu", torch=str(torch.__version__),
            states_per_model=sum(len(e["sample_frames"]) for e in p194["entries"]),
            decomposition="MSE = common-row-mean error + centered-layer error",
            constant_controls="fit-source group macro only; held labels never determine a baseline",
            numerical_tie_epsilon=audit.EPS, source_selection="all 38 Stage-194 development groups; no new windows",
            code_sha256={Path(p).name: file_hash(p) for p in (__file__, audit.__file__, source.__file__, model_code.__file__)}))
        publish_identical(root / "manifest.json", plan)
        views = sum(len(e["sample_frames"]) for e in p194["entries"])
        count_models = len(p194["folds"]) * len(previous.DESIGN["seeds"])
        total = 1 + len(p194["entries"]) + count_models + count_models * views + 76 + 1
        progress = OverallProgress("Stage-195", total, unit="audit work")
        progress.report("Verified source tables; loading frozen CPU feature pools")
        progress.advance()
        bank = {}
        for entry in p194["entries"]:
            seq, group = entry["sequence"], entry["group"]
            feature = reader.arrays(f"features/{seq}.npz", dict(plan_fingerprint=SOURCE_PLAN, sequence=seq, kind="native_parent_prefix"))
            reference = reader.arrays(f"reference/{group}.npz", dict(plan_fingerprint=SOURCE_PLAN, kind="fixed_layer_fit_labels", group=group))
            np.testing.assert_array_equal(feature["frame_id"], entry["sample_frames"])
            y = np.asarray(reference["iou"], float)
            audit.validate_scores(y, y)
            if len(feature["tokens"]) != len(y) or not np.isfinite(feature["tokens"]).all():
                raise ValueError("Feature/label state identities changed")
            bank[group] = dict(feature, labels=y, sequence=seq)
            progress.advance()
        dimension = next(iter(bank.values()))["tokens"].shape[-1]
        lens_z = int(p194["runtime"]["tracker"]["lens_z"])
        rows = {k: [] for k in TABLES}
        all_mismatches, max_source_error = 0, 0.
        progress.report("Frozen fit/held routing predictions and full-pool errors")
        for fold in p194["folds"]:
            for seed in previous.DESIGN["seeds"]:
                scope = next(r for r in tables194["scope_summary"] if int(r["fold"]) == fold["fold"] and int(r["seed"]) == seed and r["arm"] == audit.ARM)
                # Verify artifact hash and receipt before tensor deserialization.
                model, payload, relative = load_router(torch, reader, p194, fold, seed, dimension, lens_z, scope)
                draws = previous.draws_for(fold, seed, {g: len(d["frame_id"]) for g, d in bank.items()})
                if fingerprint(draws) != payload["metadata"]["draws_sha256"]:
                    raise ValueError("Reconstructed training draw schedule changed")
                counts = {g: np.zeros(len(d["frame_id"]), int) for g, d in bank.items()}
                for batch in draws:
                    for group, index in batch:
                        if group not in fold["fit"]:
                            raise ValueError("Held source in actual training draws")
                        counts[group][index] += 1
                before = source.model_digest(model)
                progress.advance()
                videos, mismatches, max_error = {}, 0, 0.
                for group, data in bank.items():
                    p, choice = predict_video(torch, model, data["tokens"], progress)
                    archived_choice = None
                    if group in fold["held"]:
                        seq = data["sequence"]
                        cached = reader.arrays(f"readout/{seq}__s{seed}__{audit.ARM}.npz",
                            dict(plan_fingerprint=SOURCE_PLAN, sequence=seq, seed=seed, arm=audit.ARM, fold=fold["fold"]))["values"]
                        archived_choice = cached[:, 4].astype(int) - 7
                        if cached.shape != (len(p), 5) or not np.isin(archived_choice, np.arange(6)).all():
                            raise ValueError("Invalid archived GPU action trace")
                        y = data["labels"]
                        # These equalities verify that the new audit uses the
                        # exact reference labels from the original Q-off policy.
                        np.testing.assert_allclose(y[np.arange(len(y)), archived_choice], cached[:, 0], atol=1e-12, rtol=0)
                        np.testing.assert_allclose(y.max(1), cached[:, 1], atol=1e-12, rtol=0)
                        mismatches += int((choice != archived_choice).sum())
                        max_error = max(max_error, float(np.abs(y[np.arange(len(y)), choice] - cached[:, 0]).max()))
                    videos[group] = dict(sequence=data["sequence"], frame_id=data["frame_id"], labels=data["labels"],
                                          prediction=p, choice=choice, archived_choice=archived_choice, draw_count=counts[group])
                controls = audit.fit_controls(videos, fold["fit"], draws)
                fixed_record = next(r for r in tables194["fixed_layers"] if int(r["fold"]) == fold["fold"])
                np.testing.assert_allclose(controls["fit_label_mean"],
                    [float(fixed_record[f"L{k}"]) for k in range(7, 13)], atol=1e-12, rtol=0)
                errors, policies, frames = make_model_reports(videos, controls, fold, seed)
                rows["error_summary"].extend(pool_error_summary(videos, controls, fold, seed, draws))
                rows["error_video"].extend(errors)
                rows["policy_video"].extend(policies)
                rows["frame_predictions"].extend(frames)
                rows["model_scope"].append(dict(fold=fold["fold"], seed=seed, arm=audit.ARM,
                    source_checkpoint=relative, source_model_sha256=reader.verified[relative],
                    fit_groups="|".join(fold["fit"]), held_groups="|".join(fold["held"]),
                    fit_states=sum(len(videos[g]["labels"]) for g in fold["fit"]),
                    held_states=sum(len(videos[g]["labels"]) for g in fold["held"]),
                    training_draws_verified=True, training_draw_examples=sum(int(v.sum()) for v in counts.values()),
                    fit_seen_states=sum(int((counts[g] > 0).sum()) for g in fold["fit"]),
                    fit_unseen_states=sum(int((counts[g] == 0).sum()) for g in fold["fit"]),
                    frozen_model_unchanged=source.model_digest(model) == before,
                    cpu_gpu_action_mismatches=mismatches, maximum_selected_iou_replay_error=max_error,
                    fit_label_mean_layer=int(audit.choose(controls["fit_label_mean"][None])[0]) + 7,
                    fit_draw_label_mean_layer=int(audit.choose(controls["fit_draw_label_mean"][None])[0]) + 7,
                    fit_prediction_mean_layer=int(audit.choose(controls["fit_prediction_mean"][None])[0]) + 7))
                if source.model_digest(model) != before:
                    raise ValueError("Frozen audit changed model parameters")
                if [r["step"] for r in payload["history"]] != list(range(1, 257)):
                    raise ValueError("Incomplete original training history")
                for entry in payload["history"]:
                    if entry["total"] != entry["router"]:
                        raise ValueError("R-only history contains another active objective")
                    rows["training_history"].append(dict(fold=fold["fold"], seed=seed, **entry))
                for group in fold["held"]:
                    rows["fixed_path_audit"].append(full_path_audit(reader, p194, tables194, fold, seed, group, before))
                    progress.advance()
                all_mismatches += mismatches
                max_source_error = max(max_source_error, max_error)
        rows["policy_summary"], rows["paired"] = summarize_policies(rows["policy_video"])
        maximum_decomposition_error = max(r["decomposition_error"] for r in rows["error_summary"])
        if maximum_decomposition_error > 1e-10:
            raise ValueError("MSE orthogonal decomposition failed")
        for row in rows["model_scope"]:
            for pool in ("fit", "held", "fit_draws"):
                records = {r["predictor"]: r for r in rows["error_summary"]
                           if r["fold"] == row["fold"] and r["seed"] == row["seed"] and r["pool"] == pool}
                row.update({f"{pool}_{name}_mse": record["mse"] for name, record in records.items()})
                row[f"{pool}_relative_skill_vs_zero"] = records["trained"]["relative_skill_vs_zero"]
                row[f"{pool}_gap_weighted_pair_accuracy"] = records["trained"]["gap_weighted_pair_accuracy"]
        hashes = {}
        for name in TABLES:
            target = root / f"stage195_{name}.csv"
            write_csv(target, rows[name])
            hashes[target.name] = file_hash(target)
        if progress.done != total - 1:
            raise ValueError("Stage-195 work count changed")
        result = source.sealed(dict(schema=SCHEMA, status="COMPLETE_FROZEN_ROUTER_OBJECTIVE_AUDIT",
            source_stage194=SOURCE194, plan_fingerprint=plan["fingerprint"], audited_models=count_models,
            source_groups=38, predicted_states=count_models * views, full_path_audits=len(rows["fixed_path_audit"]),
            new_image_reads=0, new_encoder_calls=0, training_updates=0, confirmation_groups_opened=0,
            cpu_gpu_action_mismatches=all_mismatches, maximum_selected_iou_replay_error=max_source_error,
            maximum_decomposition_error=maximum_decomposition_error,
            inference_parity="exact archived actions" if not all_mismatches else "CPU/GPU action differences require caution",
            paired_intervals="nominal source-group bootstrap; cached immediate IoU, not full-video AUC",
            unmeasured="quality predictions on policy-own image states; full fixed L8/L10/L11/L12 controls",
            evidence_scope="R-only; mathematical decomposition does not prove the training objective caused collapse",
            source_verified_artifacts=reader.verified, table_sha256=hashes))
        publish_identical(root / "stage195_decision.json", result)
        progress.phase = "Complete"
        progress.advance()
        progress.report()
        verify_results(root)


if __name__ == "__main__":
    main()
