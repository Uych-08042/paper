"""Summarize Stage-3 closed-loop median-fusion validation CSV files."""

import argparse
import csv
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


REQUIRED_FIELDS = {
    "dataset",
    "sequence",
    "policy",
    "frame_id",
    "gt_valid",
    "iou",
    "candidate_blocks",
    "model_decode_latency_ms",
}


def _success_auc(rows):
    by_sequence = defaultdict(list)
    for row in rows:
        by_sequence[(row["dataset"], row["sequence"])].append(row["iou"])
    sequence_aucs = []
    for overlaps in by_sequence.values():
        curve = [
            sum(overlap > threshold for overlap in overlaps) / len(overlaps)
            for threshold in stage2.OVERLAP_THRESHOLDS
        ]
        sequence_aucs.append(sum(curve) / len(curve))
    return 100.0 * sum(sequence_aucs) / len(sequence_aucs)


def load_closed_loop(input_path):
    input_path = Path(input_path).expanduser().resolve()
    if input_path.is_file():
        paths = [input_path]
    elif input_path.is_dir():
        paths = sorted(input_path.rglob("*_closed_loop.csv"))
    else:
        raise FileNotFoundError("Input does not exist: {}".format(input_path))
    if not paths:
        raise FileNotFoundError(
            "No *_closed_loop.csv files found under {}".format(input_path)
        )

    rows = []
    seen = set()
    for path in paths:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_FIELDS.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    "{} is missing fields: {}".format(
                        path, ", ".join(sorted(missing))
                    )
                )
            for row in reader:
                row["frame_id"] = int(row["frame_id"])
                row["gt_valid"] = int(row["gt_valid"])
                row["iou"] = float(row["iou"])
                row["candidate_blocks"] = int(row["candidate_blocks"])
                latency = row["model_decode_latency_ms"].strip()
                row["model_decode_latency_ms"] = (
                    None if latency == "" else float(latency)
                )
                key = (
                    row["dataset"],
                    row["sequence"],
                    row["policy"],
                    row["frame_id"],
                )
                if key in seen:
                    raise ValueError("Duplicate closed-loop row: {}".format(key))
                seen.add(key)
                rows.append(row)
    return input_path, paths, rows


def _baseline_rows(stage2_input, sequence_keys):
    _, _, rows, _ = stage2.load_rows(stage2_input)
    filtered = [
        row
        for row in rows
        if (row["dataset"], row["sequence"]) in sequence_keys
    ]
    found = {(row["dataset"], row["sequence"]) for row in filtered}
    missing = sequence_keys.difference(found)
    if missing:
        raise ValueError(
            "Stage-2 input is missing {} closed-loop sequences".format(
                len(missing)
            )
        )
    return filtered


def summarize(rows, stage2_input=None):
    by_policy = defaultdict(list)
    for row in rows:
        by_policy[row["policy"]].append(row)
    sequence_sets = {
        policy: {(row["dataset"], row["sequence"]) for row in policy_rows}
        for policy, policy_rows in by_policy.items()
    }
    reference_set = next(iter(sequence_sets.values()))
    differing = {
        policy: len(keys)
        for policy, keys in sequence_sets.items()
        if keys != reference_set
    }
    if differing:
        raise ValueError(
            "Policies do not cover identical sequence sets: {}".format(differing)
        )

    summaries = []
    baseline_auc = {}
    if stage2_input:
        baselines = _baseline_rows(stage2_input, reference_set)
        fast_value = stage2.success_auc(baselines, "iou_fast") * 100.0
        sgla_value = stage2.success_auc(baselines, "iou_sgla") * 100.0
        for name, field in (("fast", "iou_fast"), ("sgla", "iou_sgla")):
            value = stage2.success_auc(baselines, field) * 100.0
            baseline_auc[name] = value
            summaries.append(
                {
                    "policy": "stage2_{}".format(name),
                    "candidate_blocks": 0 if name == "fast" else 1,
                    "num_sequences": len(reference_set),
                    "num_frames": len(baselines),
                    "auc": value,
                    "gain_vs_fast": value - fast_value,
                    "gain_vs_sgla": value - sgla_value,
                    "mean_iou_valid": float(
                        np.mean(
                            [
                                row[field]
                                for row in baselines
                                if row["frame_id"] > 0 and row["gt_valid"] == 1
                            ]
                        )
                    ),
                    "mean_model_decode_latency_ms": "",
                    "median_model_decode_latency_ms": "",
                    "p90_model_decode_latency_ms": "",
                    "model_decode_fps_from_mean": "",
                }
            )

    for policy in sorted(by_policy):
        policy_rows = sorted(
            by_policy[policy],
            key=lambda row: (
                row["dataset"], row["sequence"], row["frame_id"]
            ),
        )
        latencies = np.asarray(
            [
                row["model_decode_latency_ms"]
                for row in policy_rows
                if row["model_decode_latency_ms"] is not None
            ],
            dtype=np.float64,
        )
        auc = _success_auc(policy_rows)
        fast_auc = baseline_auc.get("fast", float("nan"))
        sgla_auc = baseline_auc.get("sgla", float("nan"))
        valid_ious = [
            row["iou"]
            for row in policy_rows
            if row["frame_id"] > 0 and row["gt_valid"] == 1
        ]
        mean_latency = float(latencies.mean())
        summaries.append(
            {
                "policy": policy,
                "candidate_blocks": policy_rows[0]["candidate_blocks"],
                "num_sequences": len(reference_set),
                "num_frames": len(policy_rows),
                "auc": auc,
                "gain_vs_fast": auc - fast_auc,
                "gain_vs_sgla": auc - sgla_auc,
                "mean_iou_valid": float(np.mean(valid_ious)),
                "mean_model_decode_latency_ms": mean_latency,
                "median_model_decode_latency_ms": float(
                    np.median(latencies)
                ),
                "p90_model_decode_latency_ms": float(
                    np.percentile(latencies, 90.0)
                ),
                "model_decode_fps_from_mean": 1000.0 / mean_latency,
            }
        )
    return summaries


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
                {field: _format_value(row.get(field, "")) for field in fieldnames}
            )
    os.replace(str(temporary), str(path))


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Summarize Stage-3 closed-loop policy CSV files."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--stage2_input", default=None)
    parser.add_argument("--output", default=None)
    return parser


def main():
    args = _build_parser().parse_args()
    input_path, paths, rows = load_closed_loop(args.input)
    summaries = summarize(rows, args.stage2_input)
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (input_path if input_path.is_dir() else input_path.parent)
        / "stage3_closed_loop_summary.csv"
    )
    _write_csv(output_path, list(summaries[0].keys()), summaries)

    print(
        "Loaded {} rows from {} closed-loop CSV file(s).".format(
            len(rows), len(paths)
        )
    )
    print("policy       blocks  sequences       auc  gain_fast  gain_sgla  latency_ms")
    for row in summaries:
        latency = row["mean_model_decode_latency_ms"]
        latency_text = "n/a" if latency == "" else "{:.3f}".format(latency)
        gain_fast = row["gain_vs_fast"]
        gain_sgla = row["gain_vs_sgla"]
        print(
            "{:<12s} {:>6d}  {:>9d}  {:>8.4f}  {:>9}  {:>9}  {:>10s}".format(
                row["policy"],
                row["candidate_blocks"],
                row["num_sequences"],
                row["auc"],
                "n/a" if isinstance(gain_fast, float) and math.isnan(gain_fast) else "{:.4f}".format(gain_fast),
                "n/a" if isinstance(gain_sgla, float) and math.isnan(gain_sgla) else "{:.4f}".format(gain_sgla),
                latency_text,
            )
        )
    print("Summary CSV: {}".format(output_path))


if __name__ == "__main__":
    main()
