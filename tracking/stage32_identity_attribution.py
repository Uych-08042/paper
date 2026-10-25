"""Read-only attribution of frozen Stage-31 candidates, features and decisions.

No image decoding, network inference, fitting, parameter selection or new rollout.
Checking is opt-in and descriptive; candidate zero on an association trajectory
is not a replay of the separate original SGLA trajectory.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_json
from stage27_uav_collect import writer_lock
from stage27_uav_common import failure_events, nonempty_path, publish_identical, seal, success_auc, tracking_iou
from stage27_uav_summary import interval, write_table
from stage31_identity_common import FEATURES, METHODS, load_contract, load_sealed
from stage31_identity_fit import load_model
from stage31_identity_engine import load_video
from stage31_precision_compat import SCHEMA as COMPAT_SCHEMA, precision_replay_auditor


SCHEMA = "stage32_frozen_identity_attribution_v1"
DESIGN = dict(gt_match_iou=.5, low_iou=.2, pair_negative_iou=.1,
              bootstrap_samples=5000, bootstrap_seed=2098)
# These orientations state the meaning of each cue, not signs fitted to outcomes.
ORIENTATIONS = np.array([1., 1., 1., -1., 1., -1., -1., 1., 1.])
COHORTS = ("all_pair_frames", "native_gt_matched", "native_not_gt_matched", "native_low_iou")
SAME_STATE = "Frozen scores on recorded candidates; immediate proxy only, not an alternative rollout"


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def score_candidates(features, model):
    n, k, _ = features.shape
    return dict(native_rank=np.broadcast_to(-np.arange(k, dtype=float), (n, k)),
        motion=-features[..., 5] - .25 * features[..., 6], appearance=features[..., 0],
        static_ranker=(features / np.asarray(model["static_feature_scale"])) @ np.asarray(model["static_weights"]),
        identity=(features / np.asarray(model["feature_scale"])) @ np.asarray(model["weights"]))


def pair_diagnostics(entry, features, valid, overlap, scores, model):
    """Video/frame/pair balance matches the original training diagnostic.

    All positives versus all negatives per frame; ambiguous .1 <= IoU < .5
    boxes are excluded. Pair-score ties contribute .5, explicitly reported.
    """
    effective = np.asarray(model["weights"]) / np.asarray(model["feature_scale"])
    signals = {name: features[..., j] * ORIENTATIONS[j] for j, name in enumerate(FEATURES)}
    signals.update({"score_" + name: value for name, value in scores.items()})
    accum = defaultdict(list)
    for i, mask in enumerate(valid):
        pos = np.flatnonzero(mask & (overlap[i] >= DESIGN["gt_match_iou"]))
        neg = np.flatnonzero(mask & (overlap[i] < DESIGN["pair_negative_iou"]))
        if not len(pos) or not len(neg):
            continue
        cohorts = ["all_pair_frames", "native_gt_matched" if overlap[i, 0] >= .5 else "native_not_gt_matched"]
        if overlap[i, 0] < .2:
            cohorts.append("native_low_iou")
        for name, values in signals.items():
            delta = (values[i, pos, None] - values[i, neg][None]).ravel()
            accuracy = float(np.mean((delta > 0).astype(float) + .5 * (delta == 0)))
            strict_accuracy = float(np.mean(delta > 0))
            contribution = None
            if name in FEATURES:
                j = FEATURES.index(name)
                raw = (features[i, pos, j, None] - features[i, neg, j][None]).ravel()
                contribution = float(np.mean(raw * effective[j]))
            value = dict(pairs=len(delta), accuracy=accuracy, strict_accuracy=strict_accuracy,
                         tie_ratio=float(np.mean(delta == 0)), oriented_gap=float(np.mean(delta)), contribution=contribution)
            for cohort in cohorts:
                accum[(cohort, name)].append(value)
    rows = []
    for (cohort, signal), values in sorted(accum.items()):
        rows.append(dict(sequence=entry["sequence"], split=entry["split"], cohort=cohort, signal=signal,
            pair_frames=len(values), pairs=sum(v["pairs"] for v in values),
            pair_accuracy=float(np.mean([v["accuracy"] for v in values])),
            strict_pair_accuracy=float(np.mean([v["strict_accuracy"] for v in values])),
            tie_ratio=float(np.mean([v["tie_ratio"] for v in values])),
            oriented_gap=float(np.mean([v["oriented_gap"] for v in values])),
            identity_score_contribution=(float(np.mean([v["contribution"] for v in values]))
                                         if signal in FEATURES else None)))
    return rows


def analyze_video(entry, data, model, method, fit_collect=False):
    n = len(data["frame_id"])
    base = dict(split=entry["split"], trajectory=method, sequence=entry["sequence"],
                source_group_proxy=entry["source_group_proxy"])
    available = fit_collect or method != "sgla"
    coverage = dict(base, tracked_frames=n, candidate_sets_recorded=available,
        actual_auc=success_auc(data["iou"]), actual_low_iou_fraction=float(np.mean(data["iou"] < .2)),
        retained_gt_frames=None, native_matched_frames=None, rank_recoverable_frames=None,
        no_gt_matching_candidate_frames=None, native_low_iou_frames=None, low_iou_recoverable_frames=None,
        low_iou_candidate_missing_frames=None, retained_gt_ratio=None,
        retention_within_native_low_ratio=None, identity_recall_on_recoverable=None)
    # Baseline SGLA checking stored one output only. Its padded slots are not a
    # measured top-5 set and must never become an apparent zero-retention result.
    if not available:
        return coverage, [], [], []
    valid, features = data["valid"], data["features"]
    overlap = tracking_iou(data["candidate_boxes"], data["gt"][:, None, :])
    overlap = np.where(valid, overlap, -1.)
    native, best = overlap[:, 0], overlap.max(1)
    native_match, retained = native >= .5, best >= .5
    recoverable = ~native_match & retained
    low = native < .2
    scores = score_candidates(features, model)
    choices = {key: np.argmax(np.where(valid, value, -np.inf), axis=1) for key, value in scores.items()}
    choices["executed"] = data["chosen"]
    identity_overlap = overlap[np.arange(n), choices["identity"]]
    coverage.update(retained_gt_frames=int(retained.sum()), native_matched_frames=int(native_match.sum()),
        rank_recoverable_frames=int(recoverable.sum()), no_gt_matching_candidate_frames=int((~retained).sum()),
        native_low_iou_frames=int(low.sum()), low_iou_recoverable_frames=int((low & retained).sum()),
        low_iou_candidate_missing_frames=int((low & ~retained).sum()), retained_gt_ratio=float(retained.mean()),
        retention_within_native_low_ratio=ratio((low & retained).sum(), low.sum()),
        identity_recall_on_recoverable=ratio((recoverable & (identity_overlap >= .5)).sum(), recoverable.sum()))
    decisions = []
    for name, choice in choices.items():
        selected = overlap[np.arange(n), choice]
        decisions.append(dict(base, scorer=name, tracked_frames=n, changed_frames=int((choice != 0).sum()),
            selected_gt_matched_frames=int((selected >= .5).sum()),
            corrected_native_low_frames=int((low & (selected >= .5)).sum()),
            regressed_native_matched_frames=int((native_match & (selected < .2)).sum()),
            missed_recoverable_frames=int((recoverable & (selected < .5)).sum()),
            retained_but_selected_low_frames=int((retained & (selected < .2)).sum()),
            helpful_change_frames=int(((choice != 0) & (selected > native + 1e-12)).sum()),
            harmful_change_frames=int(((choice != 0) & (selected < native - 1e-12)).sum()),
            mean_selected_iou=float(selected.mean()), mean_native_candidate_iou=float(native.mean()),
            same_state_mean_iou_gain=float(np.mean(selected - native)),
            hindsight_best_candidate_mean_iou=float(best.mean()), role=SAME_STATE))
    pairs = [dict(row, trajectory=method) for row in pair_diagnostics(entry, features, valid, overlap, scores, model)]
    effective = np.asarray(model["weights"]) / np.asarray(model["feature_scale"])
    frames = []
    for i in range(n):
        oracle = int(np.argmax(overlap[i]))  # Fixed maximum current IoU; no future outcomes.
        choice = int(choices["identity"][i])
        outcome = "keep_native"
        if choice:
            if low[i] and identity_overlap[i] >= .5:
                outcome = "correct_native_low"
            elif native_match[i] and identity_overlap[i] < .2:
                outcome = "regress_native_matched"
            else:
                difference = identity_overlap[i] - native[i]
                outcome = "other_gain" if difference > 1e-12 else "other_loss" if difference < -1e-12 else "equal_iou_change"
        row = dict(base, frame_id=int(data["frame_id"][i]), candidates=int(valid[i].sum()),
            native_candidate_iou=float(native[i]), best_candidate_iou=float(best[i]),
            best_candidate_rank=oracle + 1, gt_retained=bool(retained[i]), rank_recoverable=bool(recoverable[i]),
            low_iou_recoverable=bool(low[i] and retained[i]), executed_rank=int(data["chosen"][i]) + 1,
            executed_iou=float(data["iou"][i]), frozen_identity_rank=choice + 1,
            frozen_identity_iou=float(identity_overlap[i]), identity_outcome=outcome,
            identity_missed_recoverable=bool(recoverable[i] and identity_overlap[i] < .5),
            identity_gt_best_minus_native_score=(float(scores["identity"][i, oracle] - scores["identity"][i, 0]) if retained[i] else None))
        for j, name in enumerate(FEATURES):
            gap = float(features[i, oracle, j] - features[i, 0, j]) if retained[i] else None
            row["gt_best_minus_native_" + name] = gap
            row["identity_contribution_" + name] = gap * float(effective[j]) if gap is not None else None
        frames.append(row)
    return coverage, decisions, pairs, frames


def summarize_pairs(rows, sources=None):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["trajectory"], row["cohort"], row["signal"])].append(row)
    rng = np.random.default_rng(DESIGN["bootstrap_seed"])
    result = []
    # Include explicit unsupported cohorts rather than silently omitting them.
    sources = sorted(sources if sources is not None else {(r["split"], r["trajectory"]) for r in rows})
    signals = list(FEATURES) + ["score_" + x for x in ("native_rank", "motion", "appearance", "static_ranker", "identity")]
    for split, trajectory in sources:
        for cohort in COHORTS:
            for signal in signals:
                values = grouped.get((split, trajectory, cohort, signal), [])
                accuracy = [r["pair_accuracy"] for r in values]
                lo, hi = interval(accuracy, rng, DESIGN["bootstrap_samples"])
                result.append(dict(split=split, trajectory=trajectory, cohort=cohort, signal=signal,
                    supported_videos=len(values), pair_frames=sum(r["pair_frames"] for r in values),
                    pairs=sum(r["pairs"] for r in values),
                    macro_pair_accuracy=float(np.mean(accuracy)) if values else None,
                    descriptive_ci_low=lo, descriptive_ci_high=hi,
                    macro_strict_pair_accuracy=float(np.mean([r["strict_pair_accuracy"] for r in values])) if values else None,
                    macro_tie_ratio=float(np.mean([r["tie_ratio"] for r in values])) if values else None,
                    macro_oriented_gap=float(np.mean([r["oriented_gap"] for r in values])) if values else None,
                    macro_identity_score_contribution=(float(np.mean([r["identity_score_contribution"] for r in values]))
                        if values and signal in FEATURES else None)))
    return result


def summarize_choices(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["trajectory"], row["scorer"])].append(row)
    result = []
    counts = ("tracked_frames", "changed_frames", "selected_gt_matched_frames", "corrected_native_low_frames",
              "regressed_native_matched_frames", "missed_recoverable_frames", "retained_but_selected_low_frames",
              "helpful_change_frames", "harmful_change_frames")
    for (split, trajectory, scorer), values in sorted(grouped.items()):
        row = dict(split=split, trajectory=trajectory, scorer=scorer, videos=len(values))
        row.update({key: sum(v[key] for v in values) for key in counts})
        row["macro_same_state_mean_iou_gain"] = float(np.mean([v["same_state_mean_iou_gain"] for v in values]))
        row["role"] = SAME_STATE
        result.append(row)
    return result


def summarize_candidates(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["trajectory"])].append(row)
    result = []
    counts = ("retained_gt_frames", "native_matched_frames", "rank_recoverable_frames", "no_gt_matching_candidate_frames",
              "native_low_iou_frames", "low_iou_recoverable_frames", "low_iou_candidate_missing_frames")
    for (split, trajectory), values in sorted(grouped.items()):
        available = all(v["candidate_sets_recorded"] for v in values)
        row = dict(split=split, trajectory=trajectory, videos=len(values), candidate_sets_recorded=available,
                   tracked_frames=sum(v["tracked_frames"] for v in values),
                   macro_actual_auc=float(np.mean([v["actual_auc"] for v in values])))
        row.update({key: sum(v[key] for v in values) if available else None for key in counts})
        row["micro_retention_within_native_low_ratio"] = ratio(row["low_iou_recoverable_frames"], row["native_low_iou_frames"]) if available else None
        result.append(row)
    return result


def closed_loop_row(entry, baseline, identity):
    """Compare actual independent whole-video runs, separate from proxy scores."""
    np.testing.assert_array_equal(baseline["frame_id"], identity["frame_id"])
    np.testing.assert_array_equal(baseline["gt"], identity["gt"])
    alternative = np.flatnonzero(identity["chosen"] > 0)
    first = int(alternative[0]) if len(alternative) else len(identity["frame_id"])
    difference = np.max(np.abs(identity["boxes"] - baseline["boxes"]), axis=1)
    divergence = np.flatnonzero(difference > 1e-9)
    baseline_events, identity_events = len(failure_events(baseline["iou"])), len(failure_events(identity["iou"]))
    return dict(sequence=entry["sequence"], source_group_proxy=entry["source_group_proxy"],
        baseline_auc=success_auc(baseline["iou"]), identity_auc=success_auc(identity["iou"]),
        auc_delta=success_auc(identity["iou"]) - success_auc(baseline["iou"]),
        low_iou_fraction_delta=float(np.mean(identity["iou"] < .2) - np.mean(baseline["iou"] < .2)),
        baseline_failure_events=baseline_events, identity_failure_events=identity_events,
        introduced_failure_on_original_stable=bool(baseline_events == 0 and identity_events > 0),
        alternative_frames=len(alternative),
        first_alternative_frame=int(identity["frame_id"][first]) if len(alternative) else None,
        first_bbox_divergence_frame=int(identity["frame_id"][divergence[0]]) if len(divergence) else None,
        frames_before_first_alternative=first,
        max_box_difference_before_first_alternative=float(difference[:first].max()) if first else 0.,
        prefix_difference_above_1e_minus9=bool(np.any(difference[:first] > 1e-9)))


def read_bound(path, bindings):
    path = Path(path)
    bindings[str(path)] = file_hash(path)
    return path


def load_sources(directory, scope):
    """Validate sources before analysis; fit scope never opens checking artifacts."""
    bindings = {}
    contract = load_contract(read_bound(directory / "manifest.json", bindings))
    read_bound(directory / "model.json", bindings)
    fit_path = read_bound(directory / "fit_run.json", bindings)
    model = load_model(directory, contract)
    fit_run = load_sealed(fit_path)
    if fit_run.get("role") != "native_fit_candidate_features":
        raise ValueError("Wrong Stage-31 fit run")
    compat_path = read_bound(directory / "stage31_precision_compat.json", bindings)
    compat = json.loads(compat_path.read_text(encoding="utf-8"))
    if (compat != seal({k: v for k, v in compat.items() if k != "fingerprint"})
            or compat.get("schema") != COMPAT_SCHEMA or compat.get("contract_fingerprint") != contract["fingerprint"]
            or compat.get("original_code_sha256") != contract["code_sha256"]
            or compat.get("compatibility_code_sha256") != file_hash(Path(__file__).with_name("stage31_precision_compat.py"))
            or compat.get("original_feature_atol") != 1e-9 or compat.get("original_feature_rtol") != 0):
        raise ValueError("Stage-31 precision compatibility provenance mismatch")
    records = []
    with precision_replay_auditor() as stats:
        for entry in contract["sequences"]:
            if entry["split"] == "fit":
                path = read_bound(directory / "fit_cache" / (entry["sequence"] + ".npz"), bindings)
                if file_hash(path) != model["fit_cache_sha256"][path.name]:
                    raise ValueError("Fit cache changed during attribution")
                data = load_video(path, contract, entry, "sgla", fit_run, fit_collect=True)
                records.append((entry, "sgla", data, True))
                print("[audit fit] {}: {} recorded frames".format(entry["sequence"], len(data["frame_id"])), flush=True)
        if scope == "both":
            checking = load_sealed(read_bound(directory / "checking_run.json", bindings))
            source_decision = load_sealed(read_bound(directory / "analysis" / "stage31_decision.json", bindings))
            expected = {e["sequence"] + "__" + m + ".npz" for e in contract["sequences"] if e["split"] == "checking" for m in METHODS}
            if (checking.get("role") != "frozen_full_video_identity_checking" or checking.get("methods") != list(METHODS)
                    or checking.get("model_fingerprint") != model["fingerprint"]
                    or checking.get("contract_fingerprint") != contract["fingerprint"]
                    or source_decision.get("role") != "fresh_group_development_only_identity_feasibility"
                    or source_decision.get("status") not in ("PASS_GT_FREE_IDENTITY_PILOT", "FAIL_GT_FREE_IDENTITY_PILOT")
                    or source_decision.get("missing_files") != []
                    or source_decision.get("checking_videos") != 6
                    or source_decision.get("analysis_code_sha256") != file_hash(Path(__file__).with_name("stage31_identity_summary.py"))
                    or source_decision.get("contract_fingerprint") != contract["fingerprint"]
                    or source_decision.get("model_fingerprint") != model["fingerprint"]
                    or source_decision.get("checking_run_fingerprint") != checking["fingerprint"]
                    or set(source_decision.get("cache_sha256", {})) != expected
                    or {p.name for p in (directory / "checking").glob("*.npz")} != expected):
                raise ValueError("Need all 30 matched checking files and the frozen complete Stage-31 decision")
            for entry in contract["sequences"]:
                if entry["split"] != "checking":
                    continue
                for method in METHODS:
                    path = read_bound(directory / "checking" / (entry["sequence"] + "__" + method + ".npz"), bindings)
                    if bindings[str(path)] != source_decision["cache_sha256"][path.name]:
                        raise ValueError("Checking cache differs from frozen Stage-31 decision: " + path.name)
                    data = load_video(path, contract, entry, method, checking, model)
                    records.append((entry, method, data, False))
                    print("[audit checking] {} {}: {} recorded frames".format(entry["sequence"], method, len(data["frame_id"])), flush=True)
        replay_stats = dict(stats, audited_sequences=sorted(stats["audited_sequences"]))
    return contract, model, records, bindings, replay_stats


def verify_unchanged(bindings):
    for name, digest in bindings.items():
        if file_hash(name) != digest:
            raise ValueError("Frozen source changed during analysis: " + name)


def run_attribution(directory, output, scope="fit"):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output == directory or directory in output.parents or output in directory.parents:
        raise ValueError("Use a separate Stage-32 output directory outside the Stage-31 source tree and its parents")
    contract, model, records, bindings, replay_stats = load_sources(directory, scope)
    candidates, decisions, pairs, frames, closed = [], [], [], [], []
    by_video = defaultdict(dict)
    for entry, method, data, fit_collect in records:
        candidate, choice, pair, frame = analyze_video(entry, data, model, method, fit_collect)
        candidates.append(candidate)
        decisions.extend(choice)
        pairs.extend(pair)
        frames.extend(frame)
        if entry["split"] == "checking":
            by_video[entry["sequence"]][method] = data
        print("[attribute] {} {} {}".format(entry["split"], entry["sequence"], method), flush=True)
    for entry in contract["sequences"]:
        if entry["sequence"] in by_video:
            data = by_video[entry["sequence"]]
            closed.append(closed_loop_row(entry, data["sgla"], data["identity"]))
    candidate_summary = summarize_candidates(candidates)
    feature_summary = summarize_pairs(pairs, {(r["split"], r["trajectory"]) for r in candidates if r["candidate_sets_recorded"]})
    choice_summary = summarize_choices(decisions)
    coefficients = [dict(feature=name, standardized_weight=model["weights"][j], feature_scale=model["feature_scale"][j],
        effective_coefficient=model["weights"][j] / model["feature_scale"][j],
        static_effective_coefficient=model["static_weights"][j] / model["static_feature_scale"][j],
        cue_orientation=ORIENTATIONS[j]) for j, name in enumerate(FEATURES)]
    verify_unchanged(bindings)
    # The source contract is rechecked as well; analysis never reseals it.
    load_contract(directory / "manifest.json")
    specification = seal(dict(schema=SCHEMA, role="post_failure_read_only_cache_attribution", scope=scope,
        source=str(directory), source_contract_fingerprint=contract["fingerprint"], source_model_fingerprint=model["fingerprint"],
        source_file_sha256=bindings, analysis_code_sha256=file_hash(Path(__file__)),
        compatibility_code_sha256=file_hash(Path(__file__).with_name("stage31_precision_compat.py")),
        analysis_dependency_sha256={name: file_hash(Path(__file__).with_name(name)) for name in
            ("stage4_common.py", "stage27_uav_common.py", "stage27_uav_summary.py", "validation_metrics.py")},
        numpy_version=np.__version__, design=DESIGN,
        analysis_definitions=dict(rank_recoverable="native candidate IoU<.5 and some recorded candidate IoU>=.5",
            low_recoverable="native candidate IoU<.2 and some recorded candidate IoU>=.5",
            pair_balance="Average positive/negative pairs within frame, frames within video, then videos equally",
            pair_ties=".5 credit; strict >0 accuracy also reported to match Stage-31 fit accuracy",
            missing_candidates="Recorded top-5 lacks IoU>=.5; this does not distinguish response, localization, visibility or crop causes",
            unsupported_baseline="Checking SGLA top-5 was not stored; retention is unknown, never zero")))
    tables = dict(stage32_video_candidates=candidates, stage32_candidate_summary=candidate_summary,
                  stage32_same_state_choices=decisions, stage32_choice_summary=choice_summary, stage32_video_feature_pairs=pairs,
                  stage32_feature_summary=feature_summary, stage32_frame_errors=frames,
                  stage32_model_coefficients=coefficients)
    if closed:
        tables["stage32_closed_loop_attribution"] = closed
    with writer_lock(output):
        publish_identical(output / "run.json", specification)
        for name, rows in tables.items():
            if rows:
                write_table(output / (name + ".csv"), rows)
        report = seal(dict(schema=SCHEMA, role="descriptive_error_attribution_no_method_screen",
            status="COMPLETE_READ_ONLY_ATTRIBUTION", run_fingerprint=specification["fingerprint"],
            source_contract_fingerprint=contract["fingerprint"], source_model_fingerprint=model["fingerprint"],
            fit_videos=8, checking_videos=6 if scope == "both" else 0,
            checking_labels_loaded=scope == "both", checking_role="Consumed outcome explanation only" if scope == "both" else "Not opened",
            training_performed=False, parameter_search_performed=False, image_decoding_performed=False,
            gpu_inference_performed=False, new_rollout_performed=False, source_artifacts_unchanged=True,
            replay_audit=replay_stats, candidate_summary=candidate_summary,
            fit_rank_accuracy_context=[r for r in feature_summary if r["split"] == "fit"
                and r["signal"] in ("score_native_rank", "score_identity", "score_static_ranker")],
            report_sha256={name + ".csv": file_hash(output / (name + ".csv")) for name, rows in tables.items() if rows},
            limitations=["Fit scores use the already fitted model; no OOF prediction or held-out fit accuracy is claimed.",
                "Each association method has its own recorded state distribution; candidate-zero proxies are not separate SGLA rollouts.",
                "GT-conditioned cohorts and hindsight candidate bounds are descriptive, not deployable policies or closed-loop gains.",
                "Feature-wise cue ordering cannot rule out information in descriptor interactions or a different learned representation.",
                "Video bootstrap intervals are descriptive for these small consumed groups; frames and methods are not independent videos.",
                "No automatic next-method pass/fail, threshold selection, speed or compression claim is made."] ))
        write_json(output / "stage32_attribution.json", report)
    print("Read-only Stage-32 attribution: 8 fit / {} checking videos; no training or GPU inference.".format(6 if scope == "both" else 0))
    print("split     trajectory       native_low  retained_at_low  recoverable  no_gt_candidate")
    for row in candidate_summary:
        if row["candidate_sets_recorded"]:
            print("{:<9s} {:<16s} {:>10d} {:>16d} {:>12d} {:>16d}".format(row["split"], row["trajectory"],
                row["native_low_iou_frames"], row["low_iou_recoverable_frames"], row["rank_recoverable_frames"], row["no_gt_matching_candidate_frames"]))
        else:
            print(row["split"] + " sgla: Top-5 not recorded; candidate retention UNKNOWN")
    print("Status: COMPLETE_READ_ONLY_ATTRIBUTION; no method pass/fail inferred.")
    print("Reports: " + str(output))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True, help="Frozen Stage-31 root")
    parser.add_argument("--output_dir", type=nonempty_path, required=True, help="Separate Stage-32 directory")
    parser.add_argument("--scope", choices=("fit", "both"), default="fit", help="Checking is opt-in explanation only")
    args = parser.parse_args()
    run_attribution(Path(args.input).expanduser(), Path(args.output_dir).expanduser(), args.scope)


if __name__ == "__main__":
    main()
