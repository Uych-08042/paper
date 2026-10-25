"""Fixed-budget, fit-only leave-one-video-out pooled-descriptor feasibility probe."""

import argparse
import json
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_cache, write_json
from stage27_uav_collect import writer_lock
from stage27_uav_common import nonempty_path, publish_identical, seal
from stage27_uav_summary import interval, write_table
from stage31_identity_common import load_contract
from stage32_identity_attribution import load_sources, verify_unchanged
from stage33_descriptor_common import (ARMS, DESIGN, evaluate_scores, initial_parameters,
    predict, prepare_video, train_model, training_pairs)


SCHEMA = "stage33_pooled_descriptor_oof_v1"


def dependency_hashes():
    names = ("stage33_descriptor_probe.py", "stage33_descriptor_common.py", "stage32_identity_attribution.py",
             "stage31_precision_compat.py", "stage4_common.py", "stage27_uav_common.py", "stage27_uav_summary.py", "validation_metrics.py")
    return {name: file_hash(Path(__file__).with_name(name)) for name in names}


def support_rows(videos):
    rows = []
    for video in videos:
        native = np.broadcast_to(-np.arange(video["valid"].shape[1], dtype=float), video["valid"].shape)
        metrics, _ = evaluate_scores(video, native)
        by_name = {m["cohort"]: m for m in metrics}
        rows.append(dict(sequence=video["sequence"], source_group=video["source_group"],
            tracked_frames=len(video["frame_id"]), easy_pair_frames=by_name["native_matched"]["pair_frames"],
            hard_pair_frames=by_name["hard"]["pair_frames"], low_pair_frames=by_name["low"]["pair_frames"],
            recoverable_frames=by_name["recoverable"]["frames"]))
    return rows


def sufficient_support(rows):
    return (sum(r["hard_pair_frames"] > 0 for r in rows) >= DESIGN["minimum_hard_videos"]
            and sum(r["hard_pair_frames"] for r in rows) >= DESIGN["minimum_hard_pair_frames"])


def load_arrays(path, expected_metadata):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata != expected_metadata:
            raise ValueError("Checkpoint/prediction belongs to a different frozen probe: " + str(path))
        return {key: archive[key] for key in archive.files if key != "metadata_json"}


def save_prediction(path, arrays, metadata):
    if path.exists():
        previous = load_arrays(path, metadata)
        if set(previous) != set(arrays):
            raise ValueError("Prediction fields changed")
        for key in arrays:
            np.testing.assert_array_equal(previous[key], arrays[key])
    else:
        write_cache(path, arrays, metadata)


def fitted_model(path, pairs, arm, seed, updates, metadata):
    expected = initial_parameters(arm, pairs["anchor"].shape[1], seed, DESIGN["projection_dim"])
    receipt_path = path.with_suffix(".json")
    if receipt_path.exists() and not path.exists():
        raise ValueError("Completed checkpoint missing: " + str(path))
    if path.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected_receipt = seal(dict(schema=SCHEMA, role="completed_fixed_budget_checkpoint",
            metadata=metadata, checkpoint_sha256=file_hash(path)))
        if receipt != expected_receipt:
            raise ValueError("Completed checkpoint receipt/hash mismatch: " + str(path))
        arrays = load_arrays(path, metadata)
        if set(arrays) != set(expected) | {"scale", "history", "maximum_data_projection_gradient"}:
            raise ValueError("Invalid checkpoint arrays")
        parameters = {key: arrays[key] for key in expected}
        if any(parameters[key].shape != expected[key].shape or not np.isfinite(parameters[key]).all() for key in expected):
            raise ValueError("Invalid checkpoint parameters")
        np.testing.assert_array_equal(arrays["scale"], pairs["scale"])
        history = arrays["history"]
        if history.ndim != 2 or history.shape[1] != 3 or history[-1, 0] != updates or not np.isfinite(history).all():
            raise ValueError("Checkpoint did not complete the fixed update budget")
        changed = {key: not np.array_equal(expected[key], parameters[key]) for key in expected}
        maximum_gradient = float(arrays["maximum_data_projection_gradient"])
        if not np.isfinite(maximum_gradient) or maximum_gradient < 0:
            raise ValueError("Invalid projection gradient audit")
        audit = dict(changed=changed, maximum_data_projection_gradient=maximum_gradient)
        print("[reuse, audited] " + path.name, flush=True)
    else:
        if path.exists():
            load_arrays(path, metadata)  # Owned orphan: retrain, never trust unsaved completion.
        parameters, history, audit = train_model(pairs, arm, seed, updates, DESIGN["projection_dim"])
        write_cache(path, dict(parameters, scale=pairs["scale"], history=history,
            maximum_data_projection_gradient=np.asarray(audit["maximum_data_projection_gradient"])), metadata)
        write_json(receipt_path, seal(dict(schema=SCHEMA, role="completed_fixed_budget_checkpoint",
            metadata=metadata, checkpoint_sha256=file_hash(path))))
    if arm == "descriptor_fixed" and audit["changed"]["projection"]:
        raise ValueError("Frozen projection changed")
    return parameters, history, audit


def aggregate_metrics(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["sequence"], row["arm"], row["condition"], row["cohort"])].append(row)
    videos = []
    fields = ("pair_accuracy", "selected_match_ratio", "native_match_ratio", "mean_immediate_iou_gain",
              "changed_frames", "corrected_low_frames", "regressed_matched_frames")
    for (sequence, arm, condition, cohort), values in sorted(grouped.items()):
        row = dict(sequence=sequence, arm=arm, condition=condition, cohort=cohort,
                   seeds=len(values), frames=values[0]["frames"], pair_frames=values[0]["pair_frames"])
        for key in fields:
            available = [v[key] for v in values if v[key] is not None]
            row[key] = float(np.mean(available)) if available else None
        videos.append(row)
    groups = defaultdict(list)
    for row in videos:
        groups[(row["arm"], row["condition"], row["cohort"])].append(row)
    summary = []
    rng = np.random.default_rng(DESIGN["bootstrap_seed"])
    for (arm, condition, cohort), values in sorted(groups.items()):
        for metric in ("pair_accuracy", "selected_match_ratio", "mean_immediate_iou_gain"):
            available = [v[metric] for v in values if v[metric] is not None]
            low, high = interval(available, rng, DESIGN["bootstrap_samples"])
            summary.append(dict(arm=arm, condition=condition, cohort=cohort, metric=metric,
                supported_videos=len(available), frames=sum(v["frames"] for v in values),
                pair_frames=sum(v["pair_frames"] for v in values),
                macro_mean=float(np.mean(available)) if available else None,
                descriptive_ci_low=low, descriptive_ci_high=high))
    return videos, summary


def paired_metrics(videos):
    index = {(r["sequence"], r["arm"], r["condition"], r["cohort"]): r for r in videos}
    rows = []
    rng = np.random.default_rng(DESIGN["bootstrap_seed"] + 1)
    for right in ("scalar_balanced", "descriptor_fixed", "native_rank"):
        for condition in ("nominal", "history_swap"):
            for cohort, metric in (("hard", "pair_accuracy"), ("low", "pair_accuracy"),
                    ("recoverable", "selected_match_ratio"), ("native_matched", "selected_match_ratio")):
                values = []
                for name in sorted({r["sequence"] for r in videos}):
                    left = index.get((name, "descriptor_learned", condition, cohort), {})
                    control = index.get((name, right, condition, cohort), {})
                    if left.get(metric) is not None and control.get(metric) is not None:
                        values.append(left[metric] - control[metric])
                low, high = interval(values, rng, DESIGN["bootstrap_samples"])
                rows.append(dict(left="descriptor_learned", right=right, condition=condition, cohort=cohort,
                    metric=metric, videos=len(values), mean_delta=float(np.mean(values)) if values else None,
                    descriptive_ci_low=low, descriptive_ci_high=high,
                    positive_videos=sum(v > 0 for v in values), tied_videos=sum(v == 0 for v in values),
                    negative_videos=sum(v < 0 for v in values)))
    return rows


def decide(support, summary, paired, raw_rows, mode, integrity):
    if mode == "audit":
        return dict(status="READY_DESCRIPTOR_PROBE" if sufficient_support(support) else "INSUFFICIENT_HARD_SUPPORT",
                    method_gain_claim=False)
    if mode == "smoke":
        passed = (all(r["head_changed"] for r in integrity)
            and all(not r["projection_changed"] for r in integrity if r["arm"] == "descriptor_fixed")
            and all(r["projection_changed"] and r["maximum_data_projection_gradient"] > 0 for r in integrity if r["arm"] == "descriptor_learned"))
        return dict(status="PASS_DESCRIPTOR_SMOKE" if passed else "FAIL_DESCRIPTOR_SMOKE", method_gain_claim=False)
    def metric(cohort, field):
        return next(r["macro_mean"] for r in summary if r["arm"] == "descriptor_learned"
                    and r["condition"] == "nominal" and r["cohort"] == cohort and r["metric"] == field)
    hard = metric("hard", "pair_accuracy")
    recovery = metric("recoverable", "selected_match_ratio")
    easy = metric("native_matched", "selected_match_ratio")
    comparison = next(r for r in paired if r["right"] == "scalar_balanced" and r["condition"] == "nominal" and r["cohort"] == "hard")
    random_comparison = next(r for r in paired if r["right"] == "descriptor_fixed" and r["condition"] == "nominal" and r["cohort"] == "hard")
    seed_deltas = []
    for seed in DESIGN["seeds"]:
        by_video = defaultdict(dict)
        for r in raw_rows:
            if r["seed"] == seed and r["condition"] == "nominal" and r["cohort"] == "hard" and r["pair_accuracy"] is not None:
                by_video[r["sequence"]][r["arm"]] = r["pair_accuracy"]
        differences = [v["descriptor_learned"] - v["scalar_balanced"] for v in by_video.values()]
        seed_deltas.append(float(np.mean(differences)) if differences else None)
    evidence = bool(sufficient_support(support) and hard is not None and hard >= DESIGN["minimum_hard_accuracy"]
        and comparison["mean_delta"] is not None and comparison["mean_delta"] >= DESIGN["minimum_hard_accuracy_lift"]
        and comparison["descriptive_ci_low"] is not None and comparison["descriptive_ci_low"] > 0
        and all(v is not None and v > 0 for v in seed_deltas)
        and recovery is not None and recovery >= DESIGN["minimum_recovery_recall"])
    safe = easy is not None and easy >= DESIGN["minimum_easy_retention"]
    status = ("INSUFFICIENT_HARD_SUPPORT" if not sufficient_support(support) else
              "PASS_POOLED_DESCRIPTOR_SCREEN" if evidence and safe else
              "SIGNAL_WITH_SAFETY_LOSS" if evidence else "NO_DESCRIPTOR_GAIN_ESTABLISHED")
    return dict(status=status, descriptor_signal_pass=evidence, same_state_easy_guard_pass=bool(safe),
        hard_pair_accuracy=hard, hard_minus_balanced_scalar=comparison,
        learned_projection_ablation=random_comparison,
        learned_projection_effect_supported=bool(random_comparison["descriptive_ci_low"] is not None and random_comparison["descriptive_ci_low"] > 0),
        seed_hard_accuracy_lifts=seed_deltas, recoverable_top1_recall=recovery, native_matched_retention=easy,
        method_gain_claim=False)


def run_probe(directory, output, mode):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output == directory or directory in output.parents or output in directory.parents:
        raise ValueError("Use a separate Stage-33 output directory outside the frozen source tree and its parents")
    contract, ignored_frozen_model, records, bindings, replay = load_sources(directory, "fit")
    del ignored_frozen_model  # Its all-fit weights are never OOF controls or scorer inputs.
    videos = sorted([prepare_video(entry, data) for entry, _, data, _ in records], key=lambda v: v["sequence"])
    if len(videos) != 8 or len({v["source_group"] for v in videos}) != 8:
        raise ValueError("Need exactly the eight frozen fit source groups")
    support = support_rows(videos)
    folds = list(range(8)) if mode == "full" else [0]
    seeds = DESIGN["seeds"] if mode == "full" else DESIGN["seeds"][:1]
    updates = DESIGN["updates"] if mode == "full" else DESIGN["smoke_updates"]
    plan = seal(dict(schema=SCHEMA, role="consumed_development_fit_only_fixed_budget_lovo", mode=mode,
        source_contract_fingerprint=contract["fingerprint"], source_file_sha256=bindings, code_sha256=dependency_hashes(),
        numpy_version=np.__version__, design=DESIGN, arms=list(ARMS), folds=[dict(index=i, held_out=videos[i]["sequence"],
            train=[v["sequence"] for j, v in enumerate(videos) if j != i]) for i in folds], seeds=seeds, updates=updates,
        input_scope="Normalized ROI-pooled descriptors and immutable first-frame anchor; no full token grid",
        descriptor_input="tanh shared projection; elementwise product/absolute difference to initial anchor; current two response features",
        scalar_control="All nine scalar features; same balanced pairs, steps, batches, head L2 and optimizer",
        training_balance="Equal easy/hard cohorts, equal supported videos per cohort, equal frames per video/cohort, equal pairs per frame",
        parameter_selection="No inner search, validation, early stopping or best-epoch selection",
        checking_labels_loaded=False))
    output.mkdir(parents=True, exist_ok=True)
    raw_rows, integrity, histories, artifacts = [], [], [], {}
    with writer_lock(output):
        publish_identical(output / "plan.json", plan)
        write_table(output / "stage33_support.csv", support)
        train = mode != "audit" and (mode == "smoke" or sufficient_support(support))
        if train:
            for fold in folds:
                test = videos[fold]
                pairs = training_pairs([v for j, v in enumerate(videos) if j != fold])
                print("[fold {}/8] held out {}; train support {}".format(fold + 1, test["sequence"], pairs["support"]), flush=True)
                for seed in seeds:
                    for arm in ARMS:
                        name = "fold{:02d}_seed{}_{}".format(fold, seed, arm)
                        metadata = dict(schema=SCHEMA, plan_fingerprint=plan["fingerprint"], fold=fold,
                            held_out=test["sequence"], train_sequences=[v["sequence"] for j, v in enumerate(videos) if j != fold],
                            seed=seed, initialization_seed=seed + 100 * fold, arm=arm, updates=updates,
                            train_support=pairs["support"], normalization="Training videos only")
                        path = output / "checkpoints" / (name + ".npz")
                        parameters, history, audit = fitted_model(path, pairs, arm, seed + 100 * fold, updates, metadata)
                        nominal, stress = predict(test, parameters, arm, pairs["scale"]), predict(test, parameters, arm, pairs["scale"], True)
                        invariant = float(np.max(np.abs(nominal[test["valid"]] - stress[test["valid"]])))
                        if arm != "scalar_balanced" and invariant != 0:
                            raise ValueError("Immutable-anchor scorer unexpectedly reads modified history features")
                        integrity.append(dict(fold=fold, seed=seed, arm=arm, head_changed=audit["changed"]["head"],
                            projection_changed=audit["changed"].get("projection", False),
                            maximum_data_projection_gradient=audit["maximum_data_projection_gradient"],
                            history_score_max_abs_delta=invariant))
                        for step, loss, norm in history:
                            histories.append(dict(fold=fold, seed=seed, arm=arm, update=int(step), loss=float(loss), gradient_norm=float(norm)))
                        predictions = dict(frame_id=test["frame_id"], valid=test["valid"],
                            nominal=np.where(test["valid"], nominal, 0.), history_swap=np.where(test["valid"], stress, 0.))
                        for condition, scores in (("nominal", nominal), ("history_swap", stress)):
                            metrics, _ = evaluate_scores(test, scores)
                            raw_rows.extend(dict(row, sequence=test["sequence"], source_group=test["source_group"], seed=seed,
                                arm=arm, condition=condition, fold=fold) for row in metrics)
                        prediction_path = output / "oof_predictions" / (name + ".npz")
                        save_prediction(prediction_path, predictions, metadata)
                        artifacts[str(path.relative_to(output))] = file_hash(path)
                        artifacts[str(path.with_suffix(".json").relative_to(output))] = file_hash(path.with_suffix(".json"))
                        artifacts[str(prediction_path.relative_to(output))] = file_hash(prediction_path)
                        print("[done] {} {} fixed updates".format(name, updates), flush=True)
                    for arm, scores in (("native_rank", np.where(test["valid"], -np.arange(test["valid"].shape[1]), -np.inf)),
                                        ("initial_cos", np.where(test["valid"], test["scalar"][..., 0], -np.inf))):
                        metrics, _ = evaluate_scores(test, scores)
                        for condition in ("nominal", "history_swap"):
                            raw_rows.extend(dict(row, sequence=test["sequence"], source_group=test["source_group"], seed=seed,
                                arm=arm, condition=condition, fold=fold) for row in metrics)
        video_rows, summary = aggregate_metrics(raw_rows)
        paired = paired_metrics(video_rows) if raw_rows else []
        decision_mode = mode if train or mode == "audit" else "audit"
        outcome = decide(support, summary, paired, raw_rows, decision_mode, integrity)
        verify_unchanged(bindings)
        load_contract(directory / "manifest.json")
        if dependency_hashes() != plan["code_sha256"]:
            raise ValueError("Probe code changed during execution")
        for filename, rows in (("stage33_seed_video_metrics", raw_rows), ("stage33_video_metrics", video_rows),
                ("stage33_metric_summary", summary), ("stage33_paired", paired), ("stage33_integrity", integrity),
                ("stage33_training_history", histories)):
            if rows:
                write_table(output / (filename + ".csv"), rows)
        report = seal(dict(schema=SCHEMA, role="development_only_pooled_descriptor_feasibility", **outcome,
            mode=mode, plan_fingerprint=plan["fingerprint"], fit_videos=8,
            excluded_checking_videos=6, checking_labels_loaded=False, backbone_inference_performed=False,
            probe_models_used=bool(train),
            backbone_updated=False, new_rollout_performed=False, source_artifacts_unchanged=True,
            source_replay_audit=replay, hard_support=dict(videos=sum(v["hard_pair_frames"] > 0 for v in support),
                pair_frames=sum(v["hard_pair_frames"] for v in support)), artifacts_sha256=artifacts,
            warnings=["All eight fit groups were already consumed in method development; LOVO isolates fitted probe weights, not method-design selection.",
                "Fixed-backbone ROI-pooled descriptors are tested, not all token information or every possible representation.",
                "Each held-out video is excluded from projection, head, normalization and sampling; no checking cache or label is read.",
                "Seeds are paired repetitions and are averaged within video before descriptive bootstrap.",
                "History rank-2 swap uses only previous recorded candidates; GT labels define subgroups after predictions.",
                "Descriptor scorers ignore mutable history by construction; score invariance is an engineering property, not evidence of robustness to crop drift.",
                "No on-policy or closed-loop AUC, novel method, compressed backbone, speedup or external generalization is established."],
            next_step="Preserve this fixed screen; only an established signal justifies designing a fresh-group closed-loop experiment"))
        write_json(output / "stage33_decision.json", report)
    print("Stage-33: " + report["status"], flush=True)
    print("Reports: " + str(output))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True)
    parser.add_argument("--output_dir", type=nonempty_path, required=True)
    parser.add_argument("--mode", choices=("audit", "smoke", "full"), default="audit")
    args = parser.parse_args()
    try:
        from threadpoolctl import threadpool_limits
        limit = threadpool_limits(limits=1)
    except ImportError:
        limit = nullcontext()
    with limit:
        run_probe(Path(args.input).expanduser(), Path(args.output_dir).expanduser(), args.mode)


if __name__ == "__main__":
    main()
