"""Diagnose where frozen Stage-7 temporal-policy gains and failures occur.

This is a post-hoc, CPU-only analysis of existing true closed-loop CSV files.
It preserves the sequence-macro AUC used by the earlier stages, audits the
per-clip cost match between Uniform-P5 and Late-P10-exact, and reports tails,
temporal bins, failure runs, and frame-level policy disagreement.
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from stage3_closed_loop_summary import load_closed_loop
from stage6_budget_oracle import write_csv
from stage6_temporal_stability import bootstrap_mean, segment_auc


POLICIES = ("fast", "uniform_p5", "late_p10_exact", "sgla")
COMPARISONS = (
    ("late_minus_fast", "late_p10_exact", "fast"),
    ("uniform_minus_fast", "uniform_p5", "fast"),
    ("late_minus_uniform", "late_p10_exact", "uniform_p5"),
    ("sgla_minus_fast", "sgla", "fast"),
)
FAILURE_THRESHOLDS = (0.1, 0.2)


def _group_sequences(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["sequence"])].append(row)
    return {
        key: sorted(sequence_rows, key=lambda row: row["frame_id"])
        for key, sequence_rows in grouped.items()
    }


def _load_policy(path, expected_policy):
    input_path, files, rows = load_closed_loop(path)
    policies = {row["policy"] for row in rows}
    if policies != {expected_policy}:
        raise ValueError(
            "Expected only policy {!r} under {}, got {}".format(
                expected_policy, input_path, sorted(policies)
            )
        )
    return {
        "path": input_path,
        "files": files,
        "rows": rows,
        "grouped": _group_sequences(rows),
    }


def _validate_aligned_sequence(name, key, policy_rows):
    reference_rows = policy_rows[POLICIES[0]]
    reference_ids = [row["frame_id"] for row in reference_rows]
    if not reference_ids or reference_ids[0] != 0:
        raise ValueError(
            "{}/{} in cohort {} must begin at frame 0".format(
                key[0], key[1], name
            )
        )
    if len(reference_ids) < 2:
        raise ValueError(
            "{}/{} in cohort {} has no tracked frames".format(
                key[0], key[1], name
            )
        )
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError(
            "Duplicate frame IDs for {}/{} in cohort {}".format(
                key[0], key[1], name
            )
        )
    if reference_ids != list(range(len(reference_ids))):
        raise ValueError(
            "Frame IDs are not contiguous for {}/{} in cohort {}".format(
                key[0], key[1], name
            )
        )

    for policy in POLICIES[1:]:
        rows = policy_rows[policy]
        frame_ids = [row["frame_id"] for row in rows]
        if frame_ids != reference_ids:
            raise ValueError(
                "Frame IDs differ for {}/{} in cohort {} ({})".format(
                    key[0], key[1], name, policy
                )
            )
    for frame_index in range(len(reference_rows)):
        validity = {
            policy: policy_rows[policy][frame_index]["gt_valid"]
            for policy in POLICIES
        }
        if len(set(validity.values())) != 1:
            raise ValueError(
                "GT validity differs for {}/{} frame {} in cohort {}: {}".format(
                    key[0], key[1], reference_ids[frame_index], name, validity
                )
            )


def build_cohort(name, policy_runs):
    """Validate four policy runs and build sequence-level records."""
    if not name or name == "pooled":
        raise ValueError("Cohort names must be non-empty and cannot be 'pooled'")
    if set(policy_runs) != set(POLICIES):
        raise ValueError("A cohort requires exactly: {}".format(", ".join(POLICIES)))

    sequence_sets = {
        policy: set(policy_runs[policy]["grouped"])
        for policy in POLICIES
    }
    reference_set = sequence_sets[POLICIES[0]]
    if not reference_set:
        raise ValueError("Cohort {} contains no sequences".format(name))
    for policy in POLICIES[1:]:
        if sequence_sets[policy] != reference_set:
            missing = reference_set.difference(sequence_sets[policy])
            extra = sequence_sets[policy].difference(reference_set)
            raise ValueError(
                "Cohort {} sequence set differs for {}: missing={}, extra={}".format(
                    name, policy, len(missing), len(extra)
                )
            )

    records = []
    for key in sorted(reference_set):
        policy_rows = {
            policy: policy_runs[policy]["grouped"][key]
            for policy in POLICIES
        }
        _validate_aligned_sequence(name, key, policy_rows)
        tracked_frames = len(policy_rows["fast"]) - 1
        costs = {
            policy: int(
                sum(
                    row["candidate_blocks"]
                    for row in policy_rows[policy]
                    if row["frame_id"] > 0
                )
            )
            for policy in POLICIES
        }
        if costs["fast"] != 0:
            raise ValueError(
                "Fast spent {} candidate blocks for {}/{} in cohort {}".format(
                    costs["fast"], key[0], key[1], name
                )
            )
        if costs["sgla"] != tracked_frames:
            raise ValueError(
                "SGLA spent {} blocks over {} tracked frames for {}/{} in "
                "cohort {}".format(
                    costs["sgla"], tracked_frames, key[0], key[1], name
                )
            )
        if costs["uniform_p5"] != costs["late_p10_exact"]:
            raise ValueError(
                "Equal-cost audit failed for {}/{} in cohort {}: "
                "Uniform-P5={} versus Late-P10-exact={}".format(
                    key[0],
                    key[1],
                    name,
                    costs["uniform_p5"],
                    costs["late_p10_exact"],
                )
            )
        valid_frames = sum(
            row["gt_valid"] == 1 and row["frame_id"] > 0
            for row in policy_rows["fast"]
        )
        records.append(
            {
                "cohort": name,
                "dataset": key[0],
                "sequence": key[1],
                "tracked_frames": tracked_frames,
                "valid_frames": int(valid_frames),
                "rows": policy_rows,
                "costs": costs,
                "aucs": {
                    policy: segment_auc(
                        [row["iou"] for row in policy_rows[policy]]
                    )
                    for policy in POLICIES
                },
            }
        )
    return records


def load_cohort(specification):
    name, fast, uniform, late, sgla = specification
    paths = dict(zip(POLICIES, (fast, uniform, late, sgla)))
    runs = {
        policy: _load_policy(paths[policy], policy) for policy in POLICIES
    }
    return build_cohort(name, runs), runs


def _trimmed_mean(values, fraction):
    values = np.sort(np.asarray(values, dtype=np.float64))
    trim = int(math.floor(len(values) * fraction + 1e-12))
    if trim == 0:
        return float(values.mean())
    if 2 * trim >= len(values):
        raise ValueError("Trim fraction removes every value")
    return float(values[trim:-trim].mean())


def _bootstrap(values, samples, rng):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        value = float(values[0])
        return {
            "ci_low": value,
            "ci_high": value,
            "probability_positive": float(value > 0.0),
        }
    return bootstrap_mean(values, samples, rng)


def _scope_records(cohorts):
    output = [(name, records) for name, records in cohorts]
    if len(cohorts) > 1:
        output.append(
            ("pooled", [record for _, records in cohorts for record in records])
        )
    return output


def policy_summary(scopes):
    output = []
    for scope, records in scopes:
        for policy in POLICIES:
            tracked_frames = sum(record["tracked_frames"] for record in records)
            valid_ious = [
                row["iou"]
                for record in records
                for row in record["rows"][policy]
                if row["frame_id"] > 0 and row["gt_valid"] == 1
            ]
            blocks = sum(record["costs"][policy] for record in records)
            output.append(
                {
                    "cohort": scope,
                    "policy": policy,
                    "num_sequences": len(records),
                    "tracked_frames": tracked_frames,
                    "valid_frames": len(valid_ious),
                    "total_candidate_blocks": blocks,
                    "candidate_blocks_per_frame": blocks / tracked_frames,
                    "auc": float(
                        np.mean([record["aucs"][policy] for record in records])
                    ),
                    "mean_iou_valid": float(np.mean(valid_ious)),
                }
            )
    return output


def sequence_delta_rows(cohorts):
    output = []
    for cohort, records in cohorts:
        for record in records:
            row = {
                "cohort": cohort,
                "dataset": record["dataset"],
                "sequence": record["sequence"],
                "tracked_frames": record["tracked_frames"],
                "valid_frames": record["valid_frames"],
                "uniform_candidate_blocks": record["costs"]["uniform_p5"],
                "late_candidate_blocks": record["costs"]["late_p10_exact"],
            }
            for policy in POLICIES:
                row["{}_auc".format(policy)] = record["aucs"][policy]
            for comparison, left, right in COMPARISONS:
                row[comparison] = record["aucs"][left] - record["aucs"][right]
            output.append(row)
    return output


def delta_summary(scopes, bootstrap_samples, rng):
    output = []
    for scope, records in scopes:
        for comparison, left, right in COMPARISONS:
            delta = np.asarray(
                [
                    record["aucs"][left] - record["aucs"][right]
                    for record in records
                ],
                dtype=np.float64,
            )
            bootstrap = _bootstrap(delta, bootstrap_samples, rng)
            percentiles = np.percentile(
                delta, [5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0]
            )
            output.append(
                {
                    "cohort": scope,
                    "comparison": comparison,
                    "left_policy": left,
                    "right_policy": right,
                    "num_sequences": len(records),
                    "left_auc": float(
                        np.mean([record["aucs"][left] for record in records])
                    ),
                    "right_auc": float(
                        np.mean([record["aucs"][right] for record in records])
                    ),
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(np.median(delta)),
                    "trimmed_mean_05": _trimmed_mean(delta, 0.05),
                    "trimmed_mean_10": _trimmed_mean(delta, 0.10),
                    "p05": float(percentiles[0]),
                    "p10": float(percentiles[1]),
                    "p25": float(percentiles[2]),
                    "p50": float(percentiles[3]),
                    "p75": float(percentiles[4]),
                    "p90": float(percentiles[5]),
                    "p95": float(percentiles[6]),
                    "wins": int(np.sum(delta > 1e-12)),
                    "ties": int(np.sum(np.abs(delta) <= 1e-12)),
                    "losses": int(np.sum(delta < -1e-12)),
                    "ci_low": bootstrap["ci_low"],
                    "ci_high": bootstrap["ci_high"],
                    "probability_positive": bootstrap[
                        "probability_positive"
                    ],
                }
            )
    return output


def tail_sequence_rows(scopes, tail_count):
    output = []
    for scope, records in scopes:
        for comparison, left, right in COMPARISONS:
            ranked = sorted(
                records,
                key=lambda record: (
                    record["aucs"][left] - record["aucs"][right],
                    record["dataset"],
                    record["sequence"],
                ),
            )
            count = min(tail_count, len(ranked))
            for tail, selected in (
                ("worst", ranked[:count]),
                ("best", list(reversed(ranked[-count:]))),
            ):
                for rank, record in enumerate(selected, 1):
                    output.append(
                        {
                            "cohort": scope,
                            "comparison": comparison,
                            "tail": tail,
                            "rank": rank,
                            "dataset": record["dataset"],
                            "sequence": record["sequence"],
                            "tracked_frames": record["tracked_frames"],
                            "valid_frames": record["valid_frames"],
                            "left_policy": left,
                            "right_policy": right,
                            "left_auc": record["aucs"][left],
                            "right_auc": record["aucs"][right],
                            "delta": record["aucs"][left]
                            - record["aucs"][right],
                        }
                    )
    return output


def _temporal_bin_values(record, policy, bins):
    tracked = [
        row for row in record["rows"][policy] if row["frame_id"] > 0
    ]
    output = [[] for _ in range(bins)]
    for index, row in enumerate(tracked):
        bin_index = min(index * bins // len(tracked), bins - 1)
        output[bin_index].append(row)
    return output


def temporal_bin_summary(scopes, bins, bootstrap_samples, rng):
    output = []
    for scope, records in scopes:
        for comparison, left, right in COMPARISONS:
            per_record = []
            for record in records:
                left_bins = _temporal_bin_values(record, left, bins)
                right_bins = _temporal_bin_values(record, right, bins)
                sequence_bins = []
                for bin_index in range(bins):
                    left_rows = left_bins[bin_index]
                    right_rows = right_bins[bin_index]
                    if not left_rows or not right_rows:
                        continue
                    left_auc = segment_auc([row["iou"] for row in left_rows])
                    right_auc = segment_auc([row["iou"] for row in right_rows])
                    valid_pairs = [
                        (left_row["iou"], right_row["iou"])
                        for left_row, right_row in zip(left_rows, right_rows)
                        if left_row["gt_valid"] == 1
                        and right_row["gt_valid"] == 1
                    ]
                    sequence_bins.append(
                        {
                            "bin": bin_index,
                            "frames": len(left_rows),
                            "valid_frames": len(valid_pairs),
                            "left_auc": left_auc,
                            "right_auc": right_auc,
                            "delta_auc": left_auc - right_auc,
                            "mean_valid_delta": (
                                float(
                                    np.mean(
                                        [
                                            left_iou - right_iou
                                            for left_iou, right_iou in valid_pairs
                                        ]
                                    )
                                )
                                if valid_pairs
                                else float("nan")
                            ),
                        }
                    )
                per_record.append(sequence_bins)

            for bin_index in range(bins):
                values = [
                    sequence_bins[bin_index]
                    for sequence_bins in per_record
                    if len(sequence_bins) == bins
                ]
                delta = np.asarray(
                    [value["delta_auc"] for value in values],
                    dtype=np.float64,
                )
                bootstrap = _bootstrap(delta, bootstrap_samples, rng)
                valid_delta = np.asarray(
                    [
                        value["mean_valid_delta"]
                        for value in values
                        if np.isfinite(value["mean_valid_delta"])
                    ],
                    dtype=np.float64,
                )
                output.append(
                    {
                        "cohort": scope,
                        "comparison": comparison,
                        "left_policy": left,
                        "right_policy": right,
                        "bin": bin_index + 1,
                        "normalized_start": bin_index / bins,
                        "normalized_end": (bin_index + 1) / bins,
                        "num_sequences": len(values),
                        "frames": sum(value["frames"] for value in values),
                        "valid_frames": sum(
                            value["valid_frames"] for value in values
                        ),
                        "left_auc_macro": float(
                            np.mean([value["left_auc"] for value in values])
                        ),
                        "right_auc_macro": float(
                            np.mean([value["right_auc"] for value in values])
                        ),
                        "delta_auc_macro": float(delta.mean()),
                        "delta_auc_ci_low": bootstrap["ci_low"],
                        "delta_auc_ci_high": bootstrap["ci_high"],
                        "delta_auc_probability_positive": bootstrap[
                            "probability_positive"
                        ],
                        "mean_valid_iou_delta_macro": (
                            float(valid_delta.mean())
                            if valid_delta.size
                            else float("nan")
                        ),
                    }
                )
    return output


def _failure_metrics(rows, threshold):
    valid_frames = 0
    failure_frames = 0
    longest_run = 0
    current_run = 0
    episodes = 0
    recovered_episodes = 0
    in_failure = False
    for row in rows:
        if row["frame_id"] == 0 or row["gt_valid"] != 1:
            current_run = 0
            in_failure = False
            continue
        valid_frames += 1
        failed = row["iou"] < threshold
        if failed:
            failure_frames += 1
            current_run += 1
            longest_run = max(longest_run, current_run)
            if not in_failure:
                episodes += 1
            in_failure = True
        else:
            if in_failure:
                recovered_episodes += 1
            current_run = 0
            in_failure = False
    return {
        "valid_frames": valid_frames,
        "failure_frames": failure_frames,
        "failure_ratio": failure_frames / valid_frames if valid_frames else float("nan"),
        "longest_failure_run": longest_run,
        "failure_episodes": episodes,
        "recovered_episodes": recovered_episodes,
        "terminal_failure": int(in_failure),
    }


def failure_rows_and_summary(scopes):
    sequence_rows = []
    summary_rows = []
    for scope, records in scopes:
        for policy in POLICIES:
            for threshold in FAILURE_THRESHOLDS:
                metrics = []
                for record in records:
                    values = _failure_metrics(record["rows"][policy], threshold)
                    metrics.append(values)
                    sequence_rows.append(
                        {
                            "cohort": scope,
                            "dataset": record["dataset"],
                            "sequence": record["sequence"],
                            "policy": policy,
                            "threshold": threshold,
                            **values,
                        }
                    )
                ratios = np.asarray(
                    [value["failure_ratio"] for value in metrics],
                    dtype=np.float64,
                )
                longest = np.asarray(
                    [value["longest_failure_run"] for value in metrics],
                    dtype=np.float64,
                )
                total_valid = sum(value["valid_frames"] for value in metrics)
                total_failed = sum(value["failure_frames"] for value in metrics)
                summary_rows.append(
                    {
                        "cohort": scope,
                        "policy": policy,
                        "threshold": threshold,
                        "num_sequences": len(records),
                        "valid_frames": total_valid,
                        "failure_frames": total_failed,
                        "macro_failure_ratio": float(ratios.mean()),
                        "pooled_failure_ratio": total_failed / total_valid,
                        "mean_longest_failure_run": float(longest.mean()),
                        "median_longest_failure_run": float(np.median(longest)),
                        "p90_longest_failure_run": float(
                            np.percentile(longest, 90.0)
                        ),
                        "p95_longest_failure_run": float(
                            np.percentile(longest, 95.0)
                        ),
                        "max_longest_failure_run": int(longest.max()),
                        "mean_failure_episodes": float(
                            np.mean(
                                [value["failure_episodes"] for value in metrics]
                            )
                        ),
                        "terminal_failure_ratio": float(
                            np.mean(
                                [value["terminal_failure"] for value in metrics]
                            )
                        ),
                    }
                )
    return sequence_rows, summary_rows


def _disagreement_metrics(left_rows, right_rows, threshold):
    tracked_pairs = [
        (left, right)
        for left, right in zip(left_rows, right_rows)
        if left["frame_id"] > 0
    ]
    differences = []
    positions = []
    signed_gaps = []
    for position, (left, right) in enumerate(tracked_pairs, 1):
        if left["gt_valid"] == 1 and right["gt_valid"] == 1:
            difference = left["iou"] - right["iou"]
            differences.append(difference)
            positions.append(position)
            signed_gaps.append(difference)
        else:
            signed_gaps.append(None)
    differences = np.asarray(differences, dtype=np.float64)
    if differences.size == 0:
        raise ValueError("A sequence has no valid paired tracked frames")

    disagreement = np.abs(differences) >= threshold
    first = np.flatnonzero(disagreement)
    first_fraction = (
        positions[int(first[0])] / len(tracked_pairs) if first.size else float("nan")
    )

    sustained_start = None
    run_sign = 0
    run_length = 0
    run_start = 0
    for position, difference in enumerate(signed_gaps, 1):
        if difference is None:
            run_sign = 0
            run_length = 0
            continue
        sign = (
            1
            if difference >= threshold
            else -1 if difference <= -threshold else 0
        )
        if sign != 0 and sign == run_sign:
            run_length += 1
        elif sign != 0:
            run_sign = sign
            run_length = 1
            run_start = position
        else:
            run_sign = 0
            run_length = 0
        if run_length >= 3:
            sustained_start = run_start
            break
    sustained_fraction = (
        sustained_start / len(tracked_pairs)
        if sustained_start is not None
        else float("nan")
    )
    return {
        "valid_paired_frames": int(differences.size),
        "disagreement_frames": int(disagreement.sum()),
        "disagreement_ratio": float(disagreement.mean()),
        "left_better_ratio": float(np.mean(differences > threshold)),
        "right_better_ratio": float(np.mean(differences < -threshold)),
        "first_disagreement_fraction": first_fraction,
        "first_sustained3_fraction": sustained_fraction,
        "max_positive_iou_gap": float(differences.max()),
        "max_negative_iou_gap": float(differences.min()),
    }


def disagreement_rows_and_summary(scopes, threshold):
    sequence_rows = []
    summary_rows = []
    for scope, records in scopes:
        for comparison, left, right in COMPARISONS:
            metrics = []
            for record in records:
                values = _disagreement_metrics(
                    record["rows"][left], record["rows"][right], threshold
                )
                metrics.append(values)
                sequence_rows.append(
                    {
                        "cohort": scope,
                        "dataset": record["dataset"],
                        "sequence": record["sequence"],
                        "comparison": comparison,
                        "left_policy": left,
                        "right_policy": right,
                        "threshold": threshold,
                        **values,
                    }
                )
            first = np.asarray(
                [
                    value["first_disagreement_fraction"]
                    for value in metrics
                    if np.isfinite(value["first_disagreement_fraction"])
                ],
                dtype=np.float64,
            )
            sustained = np.asarray(
                [
                    value["first_sustained3_fraction"]
                    for value in metrics
                    if np.isfinite(value["first_sustained3_fraction"])
                ],
                dtype=np.float64,
            )
            total_valid = sum(value["valid_paired_frames"] for value in metrics)
            total_disagreement = sum(
                value["disagreement_frames"] for value in metrics
            )
            summary_rows.append(
                {
                    "cohort": scope,
                    "comparison": comparison,
                    "left_policy": left,
                    "right_policy": right,
                    "threshold": threshold,
                    "num_sequences": len(records),
                    "valid_paired_frames": total_valid,
                    "macro_disagreement_ratio": float(
                        np.mean([value["disagreement_ratio"] for value in metrics])
                    ),
                    "pooled_disagreement_ratio": total_disagreement / total_valid,
                    "macro_left_better_ratio": float(
                        np.mean([value["left_better_ratio"] for value in metrics])
                    ),
                    "macro_right_better_ratio": float(
                        np.mean([value["right_better_ratio"] for value in metrics])
                    ),
                    "sequences_with_disagreement": int(first.size),
                    "median_first_disagreement_fraction": (
                        float(np.median(first)) if first.size else float("nan")
                    ),
                    "sequences_with_sustained3": int(sustained.size),
                    "median_first_sustained3_fraction": (
                        float(np.median(sustained))
                        if sustained.size
                        else float("nan")
                    ),
                    "mean_max_positive_iou_gap": float(
                        np.mean([value["max_positive_iou_gap"] for value in metrics])
                    ),
                    "mean_max_negative_iou_gap": float(
                        np.mean([value["max_negative_iou_gap"] for value in metrics])
                    ),
                }
            )
    return sequence_rows, summary_rows


def analyze(cohorts, bins, tail_count, bootstrap_samples, seed, disagreement_threshold):
    names = [name for name, _ in cohorts]
    if len(names) != len(set(names)):
        raise ValueError("Cohort names must be unique")
    seen = {}
    for name, records in cohorts:
        for record in records:
            key = (record["dataset"], record["sequence"])
            if key in seen:
                raise ValueError(
                    "Cohorts must be disjoint; {}/{} occurs in {} and {}".format(
                        key[0], key[1], seen[key], name
                    )
                )
            seen[key] = name

    shortest = min(
        record["tracked_frames"]
        for _, records in cohorts
        for record in records
    )
    if bins > shortest:
        raise ValueError(
            "--bins={} exceeds the shortest sequence ({} tracked frames)".format(
                bins, shortest
            )
        )

    scopes = _scope_records(cohorts)
    rng = np.random.default_rng(seed)
    failures, failure_summary_rows = failure_rows_and_summary(scopes)
    disagreements, disagreement_summary_rows = disagreement_rows_and_summary(
        scopes, disagreement_threshold
    )
    return {
        "policy_summary": policy_summary(scopes),
        "delta_summary": delta_summary(scopes, bootstrap_samples, rng),
        "sequence_deltas": sequence_delta_rows(cohorts),
        "tail_sequences": tail_sequence_rows(scopes, tail_count),
        "temporal_bins": temporal_bin_summary(
            scopes, bins, bootstrap_samples, rng
        ),
        "failure_sequences": failures,
        "failure_summary": failure_summary_rows,
        "disagreement_sequences": disagreements,
        "disagreement_summary": disagreement_summary_rows,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort",
        action="append",
        nargs=5,
        required=True,
        metavar=("NAME", "FAST_DIR", "UNIFORM_DIR", "LATE_DIR", "SGLA_DIR"),
        help="Repeat once per disjoint Stage-7 cohort.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--tail_count", type=int, default=10)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2029)
    parser.add_argument("--disagreement_threshold", type=float, default=0.10)
    return parser


def main():
    args = build_parser().parse_args()
    if args.bins < 2:
        raise ValueError("--bins must be at least 2")
    if args.tail_count < 1:
        raise ValueError("--tail_count must be positive")
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap_samples must be positive")
    if args.disagreement_threshold <= 0.0:
        raise ValueError("--disagreement_threshold must be positive")

    cohorts = []
    file_counts = {}
    for specification in args.cohort:
        records, runs = load_cohort(specification)
        name = specification[0]
        cohorts.append((name, records))
        file_counts[name] = {
            policy: len(runs[policy]["files"]) for policy in POLICIES
        }

    reports = analyze(
        cohorts,
        args.bins,
        args.tail_count,
        args.bootstrap_samples,
        args.seed,
        args.disagreement_threshold,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    paths = {}
    for name, rows in reports.items():
        path = output_dir / "stage8_{}.csv".format(name)
        write_csv(path, rows)
        paths[name] = path

    print("Stage-8A tail and temporal diagnostic")
    for name, records in cohorts:
        costs = sum(record["costs"]["uniform_p5"] for record in records)
        tracked = sum(record["tracked_frames"] for record in records)
        print(
            "Cohort {}: {} sequences, {} tracked frames, equal-cost blocks "
            "{} ({:.2%}); files {}".format(
                name, len(records), tracked, costs, costs / tracked, file_counts[name]
            )
        )

    print("\nSequence-level paired AUC deltas:")
    print("cohort       comparison              mean    median   trim10  paired_95%        W/T/L")
    for row in reports["delta_summary"]:
        interval = "[{:.4f},{:.4f}]".format(row["ci_low"], row["ci_high"])
        result = "{}/{}/{}".format(row["wins"], row["ties"], row["losses"])
        print(
            "{cohort:<12s} {comparison:<23s} {mean_delta:>7.4f}  "
            "{median_delta:>7.4f}  {trimmed_mean_10:>7.4f}  "
            "{interval:<18s} {result}".format(
                interval=interval, result=result, **row
            )
        )

    print("\nLate-minus-Uniform by normalized time bin:")
    print("cohort       bin       range  delta_auc  paired_95%")
    for row in reports["temporal_bins"]:
        if row["comparison"] != "late_minus_uniform":
            continue
        interval = "[{:.4f},{:.4f}]".format(
            row["delta_auc_ci_low"], row["delta_auc_ci_high"]
        )
        normalized_range = "{:.0%}-{:.0%}".format(
            row["normalized_start"], row["normalized_end"]
        )
        print(
            "{cohort:<12s} {bin:>3d}  {normalized_range:>9s}  "
            "{delta_auc_macro:>9.4f}  {interval}".format(
                normalized_range=normalized_range, interval=interval, **row
            )
        )

    print("\nReports:")
    for name, path in paths.items():
        print("  {:<24s} {}".format(name + ":", path))
    if len(cohorts) > 1:
        print(
            "Note: pooled rows are descriptive; keep each predeclared cohort "
            "separate for confirmatory claims."
        )


if __name__ == "__main__":
    main()
