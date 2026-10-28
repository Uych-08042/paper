"""Compare true switched runs with paired sequence-level statistics."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from stage3_closed_loop_summary import load_closed_loop
from stage6_budget_oracle import write_csv
from stage6_temporal_stability import bootstrap_mean, segment_auc


def _group(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["sequence"])].append(row)
    return {
        key: sorted(sequence_rows, key=lambda row: row["frame_id"])
        for key, sequence_rows in grouped.items()
    }


def _load_run(path):
    input_path, files, rows = load_closed_loop(path)
    methods = {row["policy"] for row in rows}
    if len(methods) != 1:
        raise ValueError(
            "Each switched input must contain one policy, got {} under {}".format(
                sorted(methods), input_path
            )
        )
    return {
        "path": input_path,
        "files": files,
        "method": next(iter(methods)),
        "grouped": _group(rows),
    }


def _bbox(row):
    return np.asarray(
        [float(row["bbox_{}".format(name)]) for name in "xywh"],
        dtype=np.float64,
    )


def _check_alignment(reference, candidate, tolerance, check_prefix=True):
    if set(reference) != set(candidate):
        raise ValueError("Switched runs have different sequence sets")
    max_bbox_difference = 0.0
    max_iou_difference = 0.0
    prefix_frames = 0
    for key in sorted(reference):
        reference_rows = reference[key]
        candidate_rows = candidate[key]
        reference_ids = [row["frame_id"] for row in reference_rows]
        candidate_ids = [row["frame_id"] for row in candidate_rows]
        if reference_ids != candidate_ids:
            raise ValueError("Frame IDs differ for {}/{}".format(*key))
        if not check_prefix:
            continue
        reference_early = int(reference_rows[0]["early_frames"])
        candidate_early = int(candidate_rows[0]["early_frames"])
        if reference_early != candidate_early:
            raise ValueError("Switch boundaries differ for {}/{}".format(*key))
        for reference_row, candidate_row in zip(reference_rows, candidate_rows):
            if reference_row["frame_id"] > reference_early:
                continue
            max_bbox_difference = max(
                max_bbox_difference,
                float(np.max(np.abs(_bbox(reference_row) - _bbox(candidate_row)))),
            )
            max_iou_difference = max(
                max_iou_difference,
                abs(reference_row["iou"] - candidate_row["iou"]),
            )
            prefix_frames += 1
    if not check_prefix:
        return "", "", ""
    if max_bbox_difference > tolerance or max_iou_difference > tolerance:
        raise RuntimeError(
            "Fast prefixes differ: bbox max {:.8g}, IoU max {:.8g}, "
            "tolerance {:.8g}".format(
                max_bbox_difference, max_iou_difference, tolerance
            )
        )
    return prefix_frames, max_bbox_difference, max_iou_difference


def _run_summary(run):
    grouped = run["grouped"]
    sequence_keys = sorted(grouped)
    tracked = [
        row
        for key in sequence_keys
        for row in grouped[key]
        if row["frame_id"] > 0
    ]
    late = [
        row
        for row in tracked
        if row["frame_id"] > int(row["early_frames"])
    ]
    early = [
        row
        for row in tracked
        if row["frame_id"] <= int(row["early_frames"])
    ]
    sequence_auc = {
        key: segment_auc([row["iou"] for row in grouped[key]])
        for key in sequence_keys
    }
    blocks = int(sum(row["candidate_blocks"] for row in tracked))
    early_blocks = int(sum(row["candidate_blocks"] for row in early))
    late_blocks = int(sum(row["candidate_blocks"] for row in late))
    assignments = Counter(grouped[key][0]["assigned_policy"] for key in sequence_keys)
    return {
        "method": run["method"],
        "num_files": len(run["files"]),
        "num_sequences": len(sequence_keys),
        "tracked_frames": len(tracked),
        "candidate_blocks": blocks,
        "candidate_blocks_per_full_frame": blocks / len(tracked),
        "early_frames": len(early),
        "early_candidate_blocks": early_blocks,
        "candidate_blocks_per_early_frame": early_blocks / len(early),
        "late_frames": len(late),
        "late_candidate_blocks": late_blocks,
        "candidate_blocks_per_late_frame": late_blocks / len(late),
        "auc": float(np.mean(list(sequence_auc.values()))),
        "assignment_distribution": "|".join(
            "{}:{}".format(policy, assignments[policy])
            for policy in sorted(assignments)
        ),
        "sequence_auc": sequence_auc,
    }


def compare(
    reference_run,
    control_runs,
    bootstrap_samples,
    seed,
    tolerance,
    check_prefix=True,
):
    reference_summary = _run_summary(reference_run)
    output_summaries = [reference_summary]
    paired_rows = []
    rng = np.random.default_rng(seed)
    for control_run in control_runs:
        prefix_frames, bbox_difference, iou_difference = _check_alignment(
            reference_run["grouped"],
            control_run["grouped"],
            tolerance,
            check_prefix=check_prefix,
        )
        control_summary = _run_summary(control_run)
        control_summary["prefix_frames_compared"] = prefix_frames
        control_summary["prefix_max_bbox_abs_diff"] = bbox_difference
        control_summary["prefix_max_iou_abs_diff"] = iou_difference
        output_summaries.append(control_summary)

        keys = sorted(reference_summary["sequence_auc"])
        reference_values = np.asarray(
            [reference_summary["sequence_auc"][key] for key in keys]
        )
        control_values = np.asarray(
            [control_summary["sequence_auc"][key] for key in keys]
        )
        delta = reference_values - control_values
        bootstrap = bootstrap_mean(delta, bootstrap_samples, rng)
        paired_rows.append(
            {
                "reference": reference_summary["method"],
                "control": control_summary["method"],
                "num_sequences": len(keys),
                "reference_auc": reference_summary["auc"],
                "control_auc": control_summary["auc"],
                "reference_blocks_per_frame": reference_summary[
                    "candidate_blocks_per_full_frame"
                ],
                "control_blocks_per_frame": control_summary[
                    "candidate_blocks_per_full_frame"
                ],
                "mean_delta": float(delta.mean()),
                "median_delta": float(np.median(delta)),
                "wins": int(np.sum(delta > 1e-12)),
                "ties": int(np.sum(np.abs(delta) <= 1e-12)),
                "losses": int(np.sum(delta < -1e-12)),
                "ci_low": bootstrap["ci_low"],
                "ci_high": bootstrap["ci_high"],
                "probability_positive": bootstrap["probability_positive"],
                "prefix_max_bbox_abs_diff": bbox_difference,
                "prefix_max_iou_abs_diff": iou_difference,
            }
        )

    for row in output_summaries:
        row.pop("sequence_auc")
        row.setdefault("prefix_frames_compared", "")
        row.setdefault("prefix_max_bbox_abs_diff", "")
        row.setdefault("prefix_max_iou_abs_diff", "")
    return output_summaries, paired_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--controls", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--prefix_tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--skip_prefix_check",
        action="store_true",
        help="Compare temporal layouts whose early trajectories intentionally differ.",
    )
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap_samples must be positive")
    if args.prefix_tolerance < 0.0:
        raise ValueError("--prefix_tolerance must be non-negative")

    reference_run = _load_run(args.reference)
    control_runs = [_load_run(path) for path in args.controls]
    methods = [reference_run["method"]] + [run["method"] for run in control_runs]
    if len(set(methods)) != len(methods):
        raise ValueError("Switched run policy names must be unique: {}".format(methods))
    summaries, paired = compare(
        reference_run,
        control_runs,
        args.bootstrap_samples,
        args.seed,
        args.prefix_tolerance,
        check_prefix=not args.skip_prefix_check,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    summary_path = output_dir / "stage6_matched_run_summary.csv"
    paired_path = output_dir / "stage6_matched_run_paired.csv"
    write_csv(summary_path, summaries)
    write_csv(paired_path, paired)

    if args.skip_prefix_check:
        print("Fast-prefix equality check: skipped (temporal-layout ablation)")
    print(
        "method                     blocks/f   early%    late%       auc  "
        "distribution"
    )
    for row in summaries:
        print(
            "{method:<26s} {candidate_blocks_per_full_frame:>8.2%}  "
            "{candidate_blocks_per_early_frame:>7.2%}  "
            "{candidate_blocks_per_late_frame:>8.2%}  {auc:>8.4f}  "
            "{assignment_distribution}".format(**row)
        )
    print("\nPaired reference minus control:")
    print("control                    mean_delta  paired_95%           W/T/L")
    for row in paired:
        interval = "[{:.4f},{:.4f}]".format(row["ci_low"], row["ci_high"])
        result = "{}/{}/{}".format(row["wins"], row["ties"], row["losses"])
        print(
            "{control:<26s} {mean_delta:>10.4f}  {interval:<19s} "
            "{result}".format(interval=interval, result=result, **row)
        )
    print("Summary CSV: {}".format(summary_path))
    print("Paired CSV: {}".format(paired_path))


if __name__ == "__main__":
    main()
