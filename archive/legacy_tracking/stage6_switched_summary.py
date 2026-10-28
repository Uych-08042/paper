"""Summarize true switched closed-loop runs against Stage-5 baselines."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from stage3_closed_loop_summary import load_closed_loop
from stage6_budget_oracle import write_csv
from stage6_temporal_stability import bootstrap_mean, segment_auc


DEFAULT_BASELINES = (
    "fast",
    "periodic_l9_10",
    "periodic_l9_20",
    "periodic_l9_30",
    "periodic_l9_50",
    "sgla",
)


def _group_sequences(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["sequence"])].append(row)
    return {
        key: sorted(values, key=lambda row: row["frame_id"])
        for key, values in grouped.items()
    }


def _sequence_auc(rows):
    return segment_auc([row["iou"] for row in rows])


def _bbox(row):
    return np.asarray(
        [float(row["bbox_{}".format(name)]) for name in "xywh"],
        dtype=np.float64,
    )


def _verify_fast_prefix(switched_grouped, fast_grouped, tolerance):
    max_bbox_difference = 0.0
    max_iou_difference = 0.0
    compared_frames = 0
    for key in sorted(switched_grouped):
        switched_rows = switched_grouped[key]
        fast_by_frame = {row["frame_id"]: row for row in fast_grouped[key]}
        early_frames = int(switched_rows[0]["early_frames"])
        for switched_row in switched_rows:
            frame_id = switched_row["frame_id"]
            if frame_id > early_frames:
                continue
            if frame_id not in fast_by_frame:
                raise ValueError(
                    "Fast baseline is missing {}/{} frame {}".format(
                        key[0], key[1], frame_id
                    )
                )
            if frame_id > 0 and switched_row["candidate_blocks"] != 0:
                raise ValueError(
                    "Fast prefix spent a candidate block at {}/{} frame {}".format(
                        key[0], key[1], frame_id
                    )
                )
            fast_row = fast_by_frame[frame_id]
            bbox_difference = float(
                np.max(np.abs(_bbox(switched_row) - _bbox(fast_row)))
            )
            iou_difference = abs(switched_row["iou"] - fast_row["iou"])
            max_bbox_difference = max(max_bbox_difference, bbox_difference)
            max_iou_difference = max(max_iou_difference, iou_difference)
            compared_frames += 1
    if max_bbox_difference > tolerance or max_iou_difference > tolerance:
        raise RuntimeError(
            "Switched Fast prefix differs from the Stage-5 Fast baseline: "
            "bbox max {:.8g}, IoU max {:.8g}, tolerance {:.8g}".format(
                max_bbox_difference, max_iou_difference, tolerance
            )
        )
    return compared_frames, max_bbox_difference, max_iou_difference


def _method_summary(method, grouped, switched=False):
    sequence_keys = sorted(grouped)
    sequence_aucs = np.asarray(
        [_sequence_auc(grouped[key]) for key in sequence_keys], dtype=np.float64
    )
    tracked = [
        row
        for key in sequence_keys
        for row in grouped[key]
        if row["frame_id"] > 0
    ]
    total_blocks = int(sum(row["candidate_blocks"] for row in tracked))
    latencies = np.asarray(
        [
            row["model_decode_latency_ms"]
            for row in tracked
            if row["model_decode_latency_ms"] is not None
        ],
        dtype=np.float64,
    )
    result = {
        "method": method,
        "is_switched": int(switched),
        "num_sequences": len(grouped),
        "tracked_frames": len(tracked),
        "total_candidate_blocks": total_blocks,
        "candidate_blocks_per_frame": total_blocks / len(tracked),
        "auc": float(sequence_aucs.mean()),
        "mean_model_decode_latency_ms": float(latencies.mean()),
        "fps_from_mean_latency": 1000.0 / float(latencies.mean()),
        "late_frames": "",
        "late_candidate_blocks": "",
        "late_candidate_blocks_per_frame": "",
        "assignment_distribution": "",
        "fast_prefix_compared_frames": "",
        "fast_prefix_max_bbox_abs_diff": "",
        "fast_prefix_max_iou_abs_diff": "",
    }
    if switched:
        late = [
            row
            for row in tracked
            if row["frame_id"] > int(row["early_frames"])
        ]
        late_blocks = int(sum(row["candidate_blocks"] for row in late))
        assignments = Counter(
            grouped[key][0]["assigned_policy"] for key in sequence_keys
        )
        result.update(
            {
                "late_frames": len(late),
                "late_candidate_blocks": late_blocks,
                "late_candidate_blocks_per_frame": late_blocks / len(late),
                "assignment_distribution": "|".join(
                    "{}:{}".format(policy, assignments[policy])
                    for policy in sorted(assignments)
                ),
            }
        )
    return result, dict(zip(sequence_keys, sequence_aucs.tolist()))


def analyze(
    switched_rows,
    baseline_rows,
    baselines,
    bootstrap_samples,
    seed,
    prefix_tolerance=1e-4,
):
    switched_policies = {row["policy"] for row in switched_rows}
    if len(switched_policies) != 1:
        raise ValueError(
            "Switched input must contain exactly one output policy, got {}".format(
                sorted(switched_policies)
            )
        )
    switched_name = next(iter(switched_policies))
    switched_grouped = _group_sequences(switched_rows)
    sequence_keys = set(switched_grouped)

    baseline_by_policy = defaultdict(list)
    for row in baseline_rows:
        if row["policy"] in baselines and (row["dataset"], row["sequence"]) in sequence_keys:
            baseline_by_policy[row["policy"]].append(row)
    missing_policies = set(baselines).difference(baseline_by_policy)
    if missing_policies:
        raise ValueError(
            "Baseline input is missing policies: {}".format(
                ", ".join(sorted(missing_policies))
            )
        )

    fast_grouped = _group_sequences(baseline_by_policy["fast"])
    if set(fast_grouped) != sequence_keys:
        raise ValueError("Fast baseline has a different sequence set")
    prefix_frames, prefix_bbox_diff, prefix_iou_diff = _verify_fast_prefix(
        switched_grouped, fast_grouped, prefix_tolerance
    )

    method_rows = []
    sequence_auc = {}
    switched_summary, switched_values = _method_summary(
        switched_name, switched_grouped, switched=True
    )
    switched_summary.update(
        {
            "fast_prefix_compared_frames": prefix_frames,
            "fast_prefix_max_bbox_abs_diff": prefix_bbox_diff,
            "fast_prefix_max_iou_abs_diff": prefix_iou_diff,
        }
    )
    method_rows.append(switched_summary)
    sequence_auc[switched_name] = switched_values

    for policy in baselines:
        grouped = _group_sequences(baseline_by_policy[policy])
        if set(grouped) != sequence_keys:
            raise ValueError("Baseline {} has a different sequence set".format(policy))
        summary, values = _method_summary(policy, grouped)
        method_rows.append(summary)
        sequence_auc[policy] = values

    fast_auc = next(row["auc"] for row in method_rows if row["method"] == "fast")
    sgla_auc = next(row["auc"] for row in method_rows if row["method"] == "sgla")
    for row in method_rows:
        row["gain_vs_fast"] = row["auc"] - fast_auc
        row["gain_vs_sgla"] = row["auc"] - sgla_auc

    rng = np.random.default_rng(seed)
    paired_rows = []
    comparison_rows = []
    switched_array = np.asarray(
        [sequence_auc[switched_name][key] for key in sorted(sequence_keys)]
    )
    for policy in baselines:
        baseline_array = np.asarray(
            [sequence_auc[policy][key] for key in sorted(sequence_keys)]
        )
        delta = switched_array - baseline_array
        bootstrap = bootstrap_mean(delta, bootstrap_samples, rng)
        paired_rows.append(
            {
                "switched_method": switched_name,
                "baseline": policy,
                "num_sequences": len(sequence_keys),
                "mean_delta": float(delta.mean()),
                "median_delta": float(np.median(delta)),
                "wins": int(np.sum(delta > 1e-12)),
                "ties": int(np.sum(np.abs(delta) <= 1e-12)),
                "losses": int(np.sum(delta < -1e-12)),
                "ci_low": bootstrap["ci_low"],
                "ci_high": bootstrap["ci_high"],
                "probability_positive": bootstrap["probability_positive"],
            }
        )
        for key, switched_value, baseline_value in zip(
            sorted(sequence_keys), switched_array, baseline_array
        ):
            comparison_rows.append(
                {
                    "dataset": key[0],
                    "sequence": key[1],
                    "switched_method": switched_name,
                    "baseline": policy,
                    "switched_auc": switched_value,
                    "baseline_auc": baseline_value,
                    "delta": switched_value - baseline_value,
                }
            )
    return method_rows, paired_rows, comparison_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--baseline_input", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--baselines", nargs="+", default=list(DEFAULT_BASELINES)
    )
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--prefix_tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap_samples must be positive")
    if args.prefix_tolerance < 0.0:
        raise ValueError("--prefix_tolerance must be non-negative")
    baselines = tuple(dict.fromkeys(args.baselines))
    if "fast" not in baselines or "sgla" not in baselines:
        raise ValueError("--baselines must include fast and sgla")

    input_path, input_files, switched_rows = load_closed_loop(args.input)
    baseline_path, baseline_files, baseline_rows = load_closed_loop(
        args.baseline_input
    )
    method_rows, paired_rows, comparison_rows = analyze(
        switched_rows,
        baseline_rows,
        baselines,
        args.bootstrap_samples,
        args.seed,
        args.prefix_tolerance,
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (input_path if input_path.is_dir() else input_path.parent)
    )
    summary_path = output_dir / "stage6_switched_summary.csv"
    paired_path = output_dir / "stage6_switched_paired.csv"
    sequence_path = output_dir / "stage6_switched_sequence_comparison.csv"
    write_csv(summary_path, method_rows)
    write_csv(paired_path, paired_rows)
    write_csv(sequence_path, comparison_rows)

    print(
        "Loaded {} switched files and {} baseline files.".format(
            len(input_files), len(baseline_files)
        )
    )
    print("Switched input: {}".format(input_path))
    print("Baseline input: {}".format(baseline_path))
    switched_summary = next(row for row in method_rows if row["is_switched"] == 1)
    print(
        "Fast-prefix check: {} frames, bbox max diff {:.3g}, IoU max diff "
        "{:.3g}".format(
            switched_summary["fast_prefix_compared_frames"],
            switched_summary["fast_prefix_max_bbox_abs_diff"],
            switched_summary["fast_prefix_max_iou_abs_diff"],
        )
    )
    print("method               blocks/f       auc  gain_fast  gain_sgla  latency_ms")
    for row in method_rows:
        print(
            "{method:<20s} {candidate_blocks_per_frame:>8.4f}  "
            "{auc:>8.4f}  {gain_vs_fast:>9.4f}  {gain_vs_sgla:>9.4f}  "
            "{mean_model_decode_latency_ms:>10.3f}".format(**row)
        )
    print("\nPaired sequence comparison for switched method:")
    print("baseline             mean_delta  paired_95%           W/T/L")
    for row in paired_rows:
        interval = "[{:.4f},{:.4f}]".format(row["ci_low"], row["ci_high"])
        win_tie_loss = "{}/{}/{}".format(
            row["wins"], row["ties"], row["losses"]
        )
        print(
            "{baseline:<20s} {mean_delta:>10.4f}  {interval:<19s} "
            "{win_tie_loss}".format(
                interval=interval,
                win_tie_loss=win_tie_loss,
                **row
            )
        )
    print("Summary CSV: {}".format(summary_path))
    print("Paired CSV: {}".format(paired_path))
    print("Sequence comparison CSV: {}".format(sequence_path))


if __name__ == "__main__":
    main()
