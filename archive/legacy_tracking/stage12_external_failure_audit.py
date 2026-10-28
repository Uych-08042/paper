"""Explain the frozen Stage-11 OTB transfer failure without tuning on OTB.

This is a post-failure descriptive audit. It accepts only the complete frozen
Stage-11 run, verifies its FAIL_TRANSFER decision, and decomposes the result by
sequence, physical-video cluster, normalized time, and low-IoU failure events.
It does not evaluate or recommend a new OTB policy.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from stage3_closed_loop_summary import load_closed_loop
from stage4_common import write_csv, write_json
from stage6_temporal_stability import segment_auc
from stage11_external_summary import audit_rows, load_contract
from stage11_external_validate import DATASET_NAME, FROZEN_POLICIES


SCHEMA = "stage12_external_failure_audit_v1"
PRIMARY_POLICY = "periodic_l9_30"
FAILURE_IOU_THRESHOLD = 0.2
NORMALIZED_TIME_BINS = 5


def _mean_ci(values, names, sequence_clusters, bootstrap_samples, seed):
    values = np.asarray(values, dtype=np.float64)
    if len(values) != len(names) or not len(values):
        raise ValueError("Values and names must have the same non-zero length")
    cluster_order = list(dict.fromkeys(sequence_clusters[name] for name in names))
    cluster_sums = np.asarray(
        [
            values[
                [
                    index
                    for index, name in enumerate(names)
                    if sequence_clusters[name] == cluster
                ]
            ].sum()
            for cluster in cluster_order
        ],
        dtype=np.float64,
    )
    cluster_counts = np.asarray(
        [
            sum(sequence_clusters[name] == cluster for name in names)
            for cluster in cluster_order
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(cluster_order),
        size=(bootstrap_samples, len(cluster_order)),
    )
    samples = cluster_sums[indices].sum(axis=1) / cluster_counts[indices].sum(
        axis=1
    )
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
        "probability_positive": float(np.mean(samples > 0.0)),
        "clusters": len(cluster_order),
    }


def _average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _correlation(left, right, rank=False):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if rank:
        left = _average_ranks(left)
        right = _average_ranks(right)
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _correlation_ci(
    left,
    right,
    names,
    sequence_clusters,
    bootstrap_samples,
    seed,
    rank=False,
):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    cluster_order = list(dict.fromkeys(sequence_clusters[name] for name in names))
    members = {
        cluster: np.asarray(
            [
                index
                for index, name in enumerate(names)
                if sequence_clusters[name] == cluster
            ],
            dtype=np.int64,
        )
        for cluster in cluster_order
    }
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(bootstrap_samples):
        chosen = rng.integers(0, len(cluster_order), size=len(cluster_order))
        indices = np.concatenate([members[cluster_order[index]] for index in chosen])
        value = _correlation(left[indices], right[indices], rank=rank)
        if value is not None:
            samples.append(value)
    point = _correlation(left, right, rank=rank)
    if point is None or not samples:
        return {"value": None, "ci_low": None, "ci_high": None}
    samples = np.asarray(samples, dtype=np.float64)
    return {
        "value": point,
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
    }


def failure_statistics(policy_rows, threshold=FAILURE_IOU_THRESHOLD):
    overlaps = np.asarray(
        [
            row["iou"]
            for row in sorted(policy_rows, key=lambda item: item["frame_id"])
            if row["frame_id"] > 0 and row["gt_valid"] == 1
        ],
        dtype=np.float64,
    )
    if not len(overlaps):
        raise ValueError("A sequence must contain at least one valid tracked frame")
    failed = overlaps < threshold
    entries = int(failed[0]) + int(np.sum((~failed[:-1]) & failed[1:]))
    recoveries = int(np.sum(failed[:-1] & (~failed[1:])))
    longest = 0
    current = 0
    for value in failed:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return {
        "failure_frame_ratio": float(failed.mean()),
        "failure_entries": entries,
        "failure_recoveries": recoveries,
        "longest_failure_run": longest,
    }


def normalized_time_aucs(policy_rows, bins=NORMALIZED_TIME_BINS):
    tracked = [
        row
        for row in sorted(policy_rows, key=lambda item: item["frame_id"])
        if row["frame_id"] > 0 and row["gt_valid"] == 1
    ]
    if len(tracked) < bins:
        raise ValueError("A sequence has fewer tracked frames than time bins")
    return [
        segment_auc([tracked[index]["iou"] for index in indices])
        for indices in np.array_split(np.arange(len(tracked)), bins)
    ]


def build_sequence_metrics(grouped, run):
    rows = []
    temporal = {}
    for name in run["expected_sequences"]:
        policy_auc = {}
        policy_failure = {}
        temporal[name] = {}
        for policy in FROZEN_POLICIES:
            policy_rows = grouped[(DATASET_NAME, name, policy)]
            policy_auc[policy] = segment_auc([row["iou"] for row in policy_rows])
            policy_failure[policy] = failure_statistics(policy_rows)
            temporal[name][policy] = normalized_time_aucs(policy_rows)

        sgla_gain = policy_auc["sgla"] - policy_auc["fast"]
        p30_gain = policy_auc[PRIMARY_POLICY] - policy_auc["fast"]
        p20_gain = policy_auc["periodic_l9_20"] - policy_auc["fast"]
        row = {
            "sequence": name,
            "physical_video": run["sequence_clusters"][name],
            "tracked_frames": int(run["expected_sequences"][name]) - 1,
            "fast_auc": policy_auc["fast"],
            "p20_auc": policy_auc["periodic_l9_20"],
            "p30_auc": policy_auc[PRIMARY_POLICY],
            "sgla_auc": policy_auc["sgla"],
            "sgla_gain_vs_fast": sgla_gain,
            "p20_gain_vs_fast": p20_gain,
            "p30_gain_vs_fast": p30_gain,
            "p30_gap_to_sgla": policy_auc[PRIMARY_POLICY] - policy_auc["sgla"],
            "p30_recovery_of_positive_sequence_headroom": (
                p30_gain / sgla_gain if sgla_gain > 1e-12 else ""
            ),
            "sgla_beneficial": int(sgla_gain > 1e-12),
            "p30_beneficial": int(p30_gain > 1e-12),
            "p30_missed_positive_headroom": int(
                sgla_gain > 1e-12 and p30_gain <= 1e-12
            ),
        }
        for policy in FROZEN_POLICIES:
            prefix = {
                "fast": "fast",
                "periodic_l9_20": "p20",
                "periodic_l9_30": "p30",
                "sgla": "sgla",
            }[policy]
            for key, value in policy_failure[policy].items():
                row["{}_{}".format(prefix, key)] = value
        rows.append(row)
    return rows, temporal


def _subgroup_definitions(sequence_rows):
    names = [row["sequence"] for row in sequence_rows]
    return [
        ("all", "all", names),
        (
            "fast_difficulty",
            "hard_lt50",
            [row["sequence"] for row in sequence_rows if row["fast_auc"] < 50.0],
        ),
        (
            "fast_difficulty",
            "medium_50_to75",
            [
                row["sequence"]
                for row in sequence_rows
                if 50.0 <= row["fast_auc"] < 75.0
            ],
        ),
        (
            "fast_difficulty",
            "easy_ge75",
            [row["sequence"] for row in sequence_rows if row["fast_auc"] >= 75.0],
        ),
        (
            "length",
            "short_lt300",
            [
                row["sequence"]
                for row in sequence_rows
                if row["tracked_frames"] < 300
            ],
        ),
        (
            "length",
            "medium_300_to999",
            [
                row["sequence"]
                for row in sequence_rows
                if 300 <= row["tracked_frames"] < 1000
            ],
        ),
        (
            "length",
            "long_ge1000",
            [
                row["sequence"]
                for row in sequence_rows
                if row["tracked_frames"] >= 1000
            ],
        ),
        (
            "observed_sgla_headroom",
            "positive",
            [row["sequence"] for row in sequence_rows if row["sgla_beneficial"]],
        ),
        (
            "observed_sgla_headroom",
            "nonpositive",
            [
                row["sequence"]
                for row in sequence_rows
                if not row["sgla_beneficial"]
            ],
        ),
    ]


def build_subgroup_rows(
    sequence_rows,
    sequence_clusters,
    bootstrap_samples,
    seed,
):
    by_name = {row["sequence"]: row for row in sequence_rows}
    output = []
    for group_index, (group_type, subgroup, names) in enumerate(
        _subgroup_definitions(sequence_rows)
    ):
        if not names:
            continue
        sgla_gain = [by_name[name]["sgla_gain_vs_fast"] for name in names]
        p30_gain = [by_name[name]["p30_gain_vs_fast"] for name in names]
        p30_gap = [by_name[name]["p30_gap_to_sgla"] for name in names]
        sgla_summary = _mean_ci(
            sgla_gain,
            names,
            sequence_clusters,
            bootstrap_samples,
            seed + group_index * 10,
        )
        p30_summary = _mean_ci(
            p30_gain,
            names,
            sequence_clusters,
            bootstrap_samples,
            seed + group_index * 10 + 1,
        )
        gap_summary = _mean_ci(
            p30_gap,
            names,
            sequence_clusters,
            bootstrap_samples,
            seed + group_index * 10 + 2,
        )
        positive_headroom = [
            name for name in names if by_name[name]["sgla_beneficial"]
        ]
        missed = sum(
            by_name[name]["p30_missed_positive_headroom"]
            for name in positive_headroom
        )
        sgla_mean = sgla_summary["mean"]
        output.append(
            {
                "group_type": group_type,
                "subgroup": subgroup,
                "num_sequences": len(names),
                "num_physical_video_clusters": sgla_summary["clusters"],
                "sgla_gain_vs_fast": sgla_mean,
                "sgla_gain_ci_low": sgla_summary["ci_low"],
                "sgla_gain_ci_high": sgla_summary["ci_high"],
                "p30_gain_vs_fast": p30_summary["mean"],
                "p30_gain_ci_low": p30_summary["ci_low"],
                "p30_gain_ci_high": p30_summary["ci_high"],
                "p30_gap_to_sgla": gap_summary["mean"],
                "p30_gap_ci_low": gap_summary["ci_low"],
                "p30_gap_ci_high": gap_summary["ci_high"],
                "aggregate_recovery_of_sgla_gain": (
                    p30_summary["mean"] / sgla_mean
                    if abs(sgla_mean) > 1e-12
                    else ""
                ),
                "p30_beneficial_sequence_ratio": float(
                    np.mean([by_name[name]["p30_beneficial"] for name in names])
                ),
                "positive_sgla_headroom_sequences": len(positive_headroom),
                "p30_missed_positive_headroom_sequences": missed,
                "p30_miss_ratio_within_positive_headroom": (
                    missed / len(positive_headroom) if positive_headroom else ""
                ),
                "sgla_failure_frame_reduction_vs_fast": float(
                    np.mean(
                        [
                            by_name[name]["fast_failure_frame_ratio"]
                            - by_name[name]["sgla_failure_frame_ratio"]
                            for name in names
                        ]
                    )
                ),
                "p30_failure_frame_reduction_vs_fast": float(
                    np.mean(
                        [
                            by_name[name]["fast_failure_frame_ratio"]
                            - by_name[name]["p30_failure_frame_ratio"]
                            for name in names
                        ]
                    )
                ),
            }
        )
    return output


def build_temporal_rows(
    temporal,
    names,
    sequence_clusters,
    bootstrap_samples,
    seed,
):
    specifications = (
        ("sgla_minus_fast", "sgla", "fast"),
        ("p30_minus_fast", PRIMARY_POLICY, "fast"),
        ("p30_minus_sgla", PRIMARY_POLICY, "sgla"),
        ("p20_minus_fast", "periodic_l9_20", "fast"),
    )
    output = []
    for bin_index in range(NORMALIZED_TIME_BINS):
        for contrast_index, (contrast, left, right) in enumerate(specifications):
            left_auc = [temporal[name][left][bin_index] for name in names]
            right_auc = [temporal[name][right][bin_index] for name in names]
            delta = np.asarray(left_auc) - np.asarray(right_auc)
            summary = _mean_ci(
                delta,
                names,
                sequence_clusters,
                bootstrap_samples,
                seed + bin_index * 20 + contrast_index,
            )
            output.append(
                {
                    "time_bin": bin_index + 1,
                    "start_fraction": bin_index / NORMALIZED_TIME_BINS,
                    "end_fraction": (bin_index + 1) / NORMALIZED_TIME_BINS,
                    "contrast": contrast,
                    "left_policy": left,
                    "right_policy": right,
                    "num_sequences": len(names),
                    "num_physical_video_clusters": summary["clusters"],
                    "left_auc": float(np.mean(left_auc)),
                    "right_auc": float(np.mean(right_auc)),
                    "mean_delta": summary["mean"],
                    "ci_low": summary["ci_low"],
                    "ci_high": summary["ci_high"],
                    "probability_positive": summary["probability_positive"],
                }
            )
    return output


def build_failure_rows(
    sequence_rows,
    sequence_clusters,
    bootstrap_samples,
    seed,
):
    names = [row["sequence"] for row in sequence_rows]
    by_name = {row["sequence"]: row for row in sequence_rows}
    prefixes = {
        "fast": "fast",
        "periodic_l9_20": "p20",
        "periodic_l9_30": "p30",
        "sgla": "sgla",
    }
    fast_ratios = np.asarray(
        [by_name[name]["fast_failure_frame_ratio"] for name in names]
    )
    output = []
    for index, policy in enumerate(FROZEN_POLICIES):
        prefix = prefixes[policy]
        ratios = np.asarray(
            [by_name[name]["{}_failure_frame_ratio".format(prefix)] for name in names]
        )
        reductions = fast_ratios - ratios
        ratio_summary = _mean_ci(
            ratios,
            names,
            sequence_clusters,
            bootstrap_samples,
            seed + index * 2,
        )
        reduction_summary = _mean_ci(
            reductions,
            names,
            sequence_clusters,
            bootstrap_samples,
            seed + index * 2 + 1,
        )
        output.append(
            {
                "policy": policy,
                "failure_iou_threshold": FAILURE_IOU_THRESHOLD,
                "num_sequences": len(names),
                "num_physical_video_clusters": ratio_summary["clusters"],
                "macro_failure_frame_ratio": ratio_summary["mean"],
                "failure_ratio_ci_low": ratio_summary["ci_low"],
                "failure_ratio_ci_high": ratio_summary["ci_high"],
                "failure_frame_reduction_vs_fast": reduction_summary["mean"],
                "reduction_ci_low": reduction_summary["ci_low"],
                "reduction_ci_high": reduction_summary["ci_high"],
                "mean_failure_entries": float(
                    np.mean(
                        [by_name[name]["{}_failure_entries".format(prefix)] for name in names]
                    )
                ),
                "mean_failure_recoveries": float(
                    np.mean(
                        [
                            by_name[name]["{}_failure_recoveries".format(prefix)]
                            for name in names
                        ]
                    )
                ),
                "mean_longest_failure_run": float(
                    np.mean(
                        [
                            by_name[name]["{}_longest_failure_run".format(prefix)]
                            for name in names
                        ]
                    )
                ),
            }
        )
    return output


def load_failed_decision(input_dir, run):
    path = input_dir / "analysis" / "stage11_external_decision.json"
    if not path.is_file():
        raise FileNotFoundError(
            "Run Stage-11 summary before the failure audit: {}".format(path)
        )
    decision = json.loads(path.read_text(encoding="utf-8"))
    if decision.get("run_fingerprint") != run["fingerprint"]:
        raise ValueError("Stage-11 decision does not match the frozen run")
    if decision.get("status") != "FAIL_TRANSFER":
        raise ValueError(
            "Stage-12 failure audit requires FAIL_TRANSFER, got {}".format(
                decision.get("status")
            )
        )
    return path, decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2042)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap_samples must be positive")

    input_dir, run = load_contract(args.input)
    decision_path, stage11_decision = load_failed_decision(input_dir, run)
    _, paths, raw_rows = load_closed_loop(input_dir)
    grouped = audit_rows(paths, raw_rows, run)
    sequence_rows, temporal = build_sequence_metrics(grouped, run)
    names = [row["sequence"] for row in sequence_rows]
    sequence_clusters = run["sequence_clusters"]
    subgroup_rows = build_subgroup_rows(
        sequence_rows,
        sequence_clusters,
        args.bootstrap_samples,
        args.seed,
    )
    temporal_rows = build_temporal_rows(
        temporal,
        names,
        sequence_clusters,
        args.bootstrap_samples,
        args.seed + 1000,
    )
    failure_rows = build_failure_rows(
        sequence_rows,
        sequence_clusters,
        args.bootstrap_samples,
        args.seed + 2000,
    )

    sgla_gain = np.asarray(
        [row["sgla_gain_vs_fast"] for row in sequence_rows], dtype=np.float64
    )
    p30_gain = np.asarray(
        [row["p30_gain_vs_fast"] for row in sequence_rows], dtype=np.float64
    )
    pearson = _correlation_ci(
        sgla_gain,
        p30_gain,
        names,
        sequence_clusters,
        args.bootstrap_samples,
        args.seed + 3000,
        rank=False,
    )
    spearman = _correlation_ci(
        sgla_gain,
        p30_gain,
        names,
        sequence_clusters,
        args.bootstrap_samples,
        args.seed + 4000,
        rank=True,
    )
    positive_headroom = [row for row in sequence_rows if row["sgla_beneficial"]]
    missed_headroom = [
        row for row in positive_headroom if row["p30_missed_positive_headroom"]
    ]
    all_subgroup = next(
        row
        for row in subgroup_rows
        if row["group_type"] == "all" and row["subgroup"] == "all"
    )
    failure_by_policy = {row["policy"]: row for row in failure_rows}
    temporal_sgla = [
        row for row in temporal_rows if row["contrast"] == "sgla_minus_fast"
    ]
    temporal_p30 = [
        row for row in temporal_rows if row["contrast"] == "p30_minus_fast"
    ]
    mechanism = {
        "schema": SCHEMA,
        "analysis_role": "post_failure_descriptive_mechanism_audit",
        "stage11_status": stage11_decision["status"],
        "stage11_decision_path": str(decision_path),
        "run_fingerprint": run["fingerprint"],
        "dataset_fingerprint": run["dataset_fingerprint"],
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "num_sequence_objects": len(names),
        "num_physical_video_clusters": len(set(sequence_clusters.values())),
        "positive_sgla_headroom_sequences": len(positive_headroom),
        "p30_missed_positive_headroom_sequences": len(missed_headroom),
        "p30_miss_ratio_within_positive_headroom": (
            len(missed_headroom) / len(positive_headroom)
            if positive_headroom
            else None
        ),
        "aggregate_p30_recovery_of_sgla_gain": all_subgroup[
            "aggregate_recovery_of_sgla_gain"
        ],
        "sgla_p30_gain_pearson": pearson,
        "sgla_p30_gain_spearman": spearman,
        "fast_failure_frame_ratio": failure_by_policy["fast"][
            "macro_failure_frame_ratio"
        ],
        "sgla_failure_frame_reduction_vs_fast": failure_by_policy["sgla"][
            "failure_frame_reduction_vs_fast"
        ],
        "p30_failure_frame_reduction_vs_fast": failure_by_policy[PRIMARY_POLICY][
            "failure_frame_reduction_vs_fast"
        ],
        "largest_sgla_headroom_time_bin": int(
            max(temporal_sgla, key=lambda row: row["mean_delta"])["time_bin"]
        ),
        "largest_p30_gain_time_bin": int(
            max(temporal_p30, key=lambda row: row["mean_delta"])["time_bin"]
        ),
        "policy_search_permitted": False,
        "interpretation": (
            "This audit explains the frozen OTB failure only. It cannot select "
            "an OTB rate, phase, layer, threshold, or controller. Any new "
            "adaptive method must be developed away from OTB and evaluated on "
            "a new untouched external benchmark."
        ),
    }

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_dir / "analysis" / "stage12_failure_audit"
    )
    sequence_path = output_dir / "stage12_sequence_mechanism.csv"
    subgroup_path = output_dir / "stage12_subgroup_summary.csv"
    temporal_path = output_dir / "stage12_temporal_summary.csv"
    failure_path = output_dir / "stage12_failure_summary.csv"
    mechanism_path = output_dir / "stage12_mechanism.json"
    write_csv(sequence_path, sequence_rows)
    write_csv(subgroup_path, subgroup_rows)
    write_csv(temporal_path, temporal_rows)
    write_csv(failure_path, failure_rows)
    write_json(mechanism_path, mechanism)

    print(
        "Stage-12 external failure audit: {} sequence objects / {} physical "
        "videos; Stage-11 status={}.".format(
            len(names),
            len(set(sequence_clusters.values())),
            stage11_decision["status"],
        )
    )
    print(
        "Positive SGLA headroom: {} sequences; P30 missed {} ({:.2%}).".format(
            len(positive_headroom),
            len(missed_headroom),
            mechanism["p30_miss_ratio_within_positive_headroom"],
        )
    )
    print(
        "Gain alignment: Pearson={:.4f} [{:.4f},{:.4f}], Spearman={:.4f} "
        "[{:.4f},{:.4f}]".format(
            pearson["value"],
            pearson["ci_low"],
            pearson["ci_high"],
            spearman["value"],
            spearman["ci_low"],
            spearman["ci_high"],
        )
    )
    print(
        "IoU<{:.1f} failure-frame reduction vs Fast: P30={:.4%}, "
        "SGLA={:.4%}.".format(
            FAILURE_IOU_THRESHOLD,
            mechanism["p30_failure_frame_reduction_vs_fast"],
            mechanism["sgla_failure_frame_reduction_vs_fast"],
        )
    )
    print("Normalized temporal contrasts:")
    print("bin  sgla-fast  p30-fast  p30-sgla")
    for bin_index in range(1, NORMALIZED_TIME_BINS + 1):
        values = {
            row["contrast"]: row["mean_delta"]
            for row in temporal_rows
            if row["time_bin"] == bin_index
        }
        print(
            "{:>3d}  {:>9.4f}  {:>8.4f}  {:>9.4f}".format(
                bin_index,
                values["sgla_minus_fast"],
                values["p30_minus_fast"],
                values["p30_minus_sgla"],
            )
        )
    print("Sequence CSV: {}".format(sequence_path))
    print("Subgroup CSV: {}".format(subgroup_path))
    print("Temporal CSV: {}".format(temporal_path))
    print("Failure CSV: {}".format(failure_path))
    print("Mechanism JSON: {}".format(mechanism_path))
    print("No OTB policy search is permitted after this descriptive audit.")


if __name__ == "__main__":
    main()
