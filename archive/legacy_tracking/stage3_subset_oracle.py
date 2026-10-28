"""Analyze compute-quality upper bounds for subsets of Stage-2 candidates.

The script consumes the existing Stage-2 per-frame CSV files and enumerates
all non-empty subsets of L7-L12. It is a diagnostic Oracle: ground-truth IoU
is used only to measure the best achievable result for each fixed subset.
"""

import argparse
import csv
import itertools
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import stage2_candidate_oracle as stage2


CANDIDATE_LAYERS = stage2.CANDIDATE_LAYERS
TOLERANCE = 1e-12


def _success_auc_from_values(rows, values):
    """Match the repository's macro-sequence success AUC definition."""
    if len(rows) != len(values):
        raise ValueError("Row and value counts differ")

    by_sequence = defaultdict(list)
    for row, value in zip(rows, values):
        by_sequence[(row["dataset"], row["sequence"])].append(float(value))
    if not by_sequence:
        raise ValueError("Cannot calculate AUC from zero rows")

    sequence_aucs = []
    for overlaps in by_sequence.values():
        overlaps = np.asarray(overlaps, dtype=np.float64)
        curve = [
            float(np.mean(overlaps > threshold))
            for threshold in stage2.OVERLAP_THRESHOLDS
        ]
        sequence_aucs.append(float(np.mean(curve)))
    return float(np.mean(sequence_aucs)) * 100.0


def _format_layers(layers):
    return "|".join(str(layer) for layer in layers)


def _format_value(value):
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return ""
        return "{:.8f}".format(float(value))
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: _format_value(row.get(field, ""))
                    for field in fieldnames
                }
            )
    os.replace(str(temporary), str(path))


def analyze_subsets(rows):
    fast = np.asarray([row["iou_fast"] for row in rows], dtype=np.float64)
    candidates = np.asarray(
        [
            [row["iou_layer{}".format(layer)] for layer in CANDIDATE_LAYERS]
            for row in rows
        ],
        dtype=np.float64,
    )
    stored_best = np.asarray(
        [row["iou_best_candidate"] for row in rows], dtype=np.float64
    )
    calculated_best = candidates.max(axis=1)
    maximum_error = float(np.max(np.abs(stored_best - calculated_best)))
    if maximum_error > 1e-8:
        raise ValueError(
            "Stored Best-of-6 differs from candidate maximum by {:.3g}".format(
                maximum_error
            )
        )

    valid = np.asarray(
        [row["frame_id"] > 0 and row["gt_valid"] == 1 for row in rows],
        dtype=np.bool_,
    )
    if not valid.any():
        raise ValueError("No valid tracked frames were found")

    fast_auc = _success_auc_from_values(rows, fast)
    all_forced_auc = _success_auc_from_values(rows, calculated_best)
    all_action = np.maximum(fast, calculated_best)
    all_action_auc = _success_auc_from_values(rows, all_action)
    forced_denominator = all_forced_auc - fast_auc
    action_denominator = all_action_auc - fast_auc

    subset_rows = []
    for num_blocks in range(1, len(CANDIDATE_LAYERS) + 1):
        for offsets in itertools.combinations(
            range(len(CANDIDATE_LAYERS)), num_blocks
        ):
            layers = tuple(CANDIDATE_LAYERS[offset] for offset in offsets)
            subset_best = candidates[:, offsets].max(axis=1)
            action_best = np.maximum(fast, subset_best)
            valid_delta = subset_best[valid] - fast[valid]
            forced_auc = _success_auc_from_values(rows, subset_best)
            action_auc = _success_auc_from_values(rows, action_best)
            subset_rows.append(
                {
                    "num_blocks": num_blocks,
                    "layers": _format_layers(layers),
                    "forced_candidate_auc": forced_auc,
                    "forced_gain_vs_fast": forced_auc - fast_auc,
                    "forced_recovery_vs_all6": (
                        (forced_auc - fast_auc) / forced_denominator
                        if forced_denominator > TOLERANCE
                        else float("nan")
                    ),
                    "action_oracle_auc": action_auc,
                    "action_gain_vs_fast": action_auc - fast_auc,
                    "action_recovery_vs_all6": (
                        (action_auc - fast_auc) / action_denominator
                        if action_denominator > TOLERANCE
                        else float("nan")
                    ),
                    "action_deep_ratio": float(
                        np.mean(valid_delta > TOLERANCE)
                    ),
                    "mean_best_delta_valid": float(valid_delta.mean()),
                    "positive_delta_ratio_valid": float(
                        np.mean(valid_delta > TOLERANCE)
                    ),
                }
            )

    best_by_k = []
    for num_blocks in range(1, len(CANDIDATE_LAYERS) + 1):
        candidates_at_k = [
            row for row in subset_rows if row["num_blocks"] == num_blocks
        ]
        best_forced = max(
            candidates_at_k,
            key=lambda row: (
                row["forced_candidate_auc"],
                row["action_oracle_auc"],
                row["layers"],
            ),
        )
        best_action = max(
            candidates_at_k,
            key=lambda row: (
                row["action_oracle_auc"],
                row["forced_candidate_auc"],
                row["layers"],
            ),
        )
        best_by_k.append(
            {
                "num_blocks": num_blocks,
                "num_subsets": len(candidates_at_k),
                "best_forced_layers": best_forced["layers"],
                "best_forced_auc": best_forced["forced_candidate_auc"],
                "best_forced_gain_vs_fast": best_forced[
                    "forced_gain_vs_fast"
                ],
                "best_forced_recovery_vs_all6": best_forced[
                    "forced_recovery_vs_all6"
                ],
                "best_action_layers": best_action["layers"],
                "best_action_auc": best_action["action_oracle_auc"],
                "best_action_gain_vs_fast": best_action[
                    "action_gain_vs_fast"
                ],
                "best_action_recovery_vs_all6": best_action[
                    "action_recovery_vs_all6"
                ],
                "best_action_deep_ratio": best_action[
                    "action_deep_ratio"
                ],
            }
        )

    leave_one_out = []
    for omitted_offset, omitted_layer in enumerate(CANDIDATE_LAYERS):
        kept_offsets = tuple(
            offset
            for offset in range(len(CANDIDATE_LAYERS))
            if offset != omitted_offset
        )
        kept_best = candidates[:, kept_offsets].max(axis=1)
        kept_action = np.maximum(fast, kept_best)
        omitted_values = candidates[:, omitted_offset]
        unique_action = (
            (omitted_values > fast + TOLERANCE)
            & (omitted_values > kept_best + TOLERANCE)
            & valid
        )
        leave_one_out.append(
            {
                "omitted_layer": omitted_layer,
                "kept_layers": _format_layers(
                    tuple(
                        layer
                        for layer in CANDIDATE_LAYERS
                        if layer != omitted_layer
                    )
                ),
                "forced_auc_without_layer": _success_auc_from_values(
                    rows, kept_best
                ),
                "forced_auc_drop": all_forced_auc
                - _success_auc_from_values(rows, kept_best),
                "action_auc_without_layer": _success_auc_from_values(
                    rows, kept_action
                ),
                "action_auc_drop": all_action_auc
                - _success_auc_from_values(rows, kept_action),
                "unique_best_action_frames": int(unique_action.sum()),
                "unique_best_action_ratio_valid": float(
                    unique_action.sum() / valid.sum()
                ),
            }
        )

    summary = {
        "num_frames": len(rows),
        "num_valid_decision_frames": int(valid.sum()),
        "num_sequences": len(
            {(row["dataset"], row["sequence"]) for row in rows}
        ),
        "fast_auc": fast_auc,
        "all6_forced_candidate_auc": all_forced_auc,
        "all6_forced_gain_vs_fast": all_forced_auc - fast_auc,
        "all6_action_oracle_auc": all_action_auc,
        "all6_action_gain_vs_fast": all_action_auc - fast_auc,
        "all6_action_deep_ratio": float(
            np.mean((calculated_best[valid] - fast[valid]) > TOLERANCE)
        ),
        "stored_best_max_abs_error": maximum_error,
    }
    return summary, subset_rows, best_by_k, leave_one_out


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate Stage-2 candidate subsets and their GT Oracle bounds."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output_prefix", default=None)
    return parser


def main():
    args = _build_parser().parse_args()
    input_path, _, rows, csv_files = stage2.load_rows(args.input)
    summary, subset_rows, best_by_k, leave_one_out = analyze_subsets(rows)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (input_path if input_path.is_dir() else input_path.parent)
    )
    datasets = sorted({row["dataset"] for row in rows})
    dataset_name = datasets[0] if len(datasets) == 1 else "multi_dataset"
    dataset_name = dataset_name.replace("/", "_").replace("\\", "_")
    prefix = args.output_prefix or "{}_stage3_subset".format(dataset_name)

    summary_path = output_dir / "{}_summary.csv".format(prefix)
    all_subsets_path = output_dir / "{}_all.csv".format(prefix)
    best_by_k_path = output_dir / "{}_best_by_k.csv".format(prefix)
    leave_one_out_path = output_dir / "{}_leave_one_out.csv".format(prefix)

    _write_csv(summary_path, list(summary.keys()), [summary])
    _write_csv(
        all_subsets_path, list(subset_rows[0].keys()), subset_rows
    )
    _write_csv(best_by_k_path, list(best_by_k[0].keys()), best_by_k)
    _write_csv(
        leave_one_out_path,
        list(leave_one_out[0].keys()),
        leave_one_out,
    )

    print(
        "Loaded {} frames, {} sequences from {} CSV file(s).".format(
            summary["num_frames"], summary["num_sequences"], len(csv_files)
        )
    )
    print(
        "Fast={:.4f}, Best-of-6={:.4f}, Best-of-7={:.4f}".format(
            summary["fast_auc"],
            summary["all6_forced_candidate_auc"],
            summary["all6_action_oracle_auc"],
        )
    )
    print(
        "blocks  best forced subset   forced_auc  recovery   "
        "best action subset   action_auc  recovery  deep_ratio"
    )
    for row in best_by_k:
        print(
            "{:>6d}  {:<19s}  {:>10.4f}  {:>8.2%}   "
            "{:<18s}  {:>10.4f}  {:>8.2%}  {:>10.2%}".format(
                row["num_blocks"],
                row["best_forced_layers"],
                row["best_forced_auc"],
                row["best_forced_recovery_vs_all6"],
                row["best_action_layers"],
                row["best_action_auc"],
                row["best_action_recovery_vs_all6"],
                row["best_action_deep_ratio"],
            )
        )
    print("Summary CSV: {}".format(summary_path))
    print("All subsets CSV: {}".format(all_subsets_path))
    print("Best-by-K CSV: {}".format(best_by_k_path))
    print("Leave-one-out CSV: {}".format(leave_one_out_path))


if __name__ == "__main__":
    main()
