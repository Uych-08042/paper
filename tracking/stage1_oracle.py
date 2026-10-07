"""Compute fixed-budget Oracle results from stage1_validate.py CSV files."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


REQUIRED_FIELDS = {
    "dataset",
    "sequence",
    "frame_id",
    "gt_valid",
    "iou_fast",
    "iou_sgla",
    "iou_full",
    "delta_full_fast",
}
OVERLAP_THRESHOLDS = [index * 0.05 for index in range(21)]


def _read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = REQUIRED_FIELDS.difference(fieldnames)
        if missing:
            raise ValueError(
                "{} is missing fields: {}".format(
                    path, ", ".join(sorted(missing))
                )
            )
        rows = list(reader)
    return fieldnames, rows


def load_rows(input_path):
    input_path = Path(input_path).expanduser().resolve()
    if input_path.is_file():
        csv_files = [input_path]
    elif input_path.is_dir():
        csv_files = sorted(input_path.rglob("*_stage1.csv"))
    else:
        raise FileNotFoundError("Input does not exist: {}".format(input_path))

    if not csv_files:
        raise FileNotFoundError(
            "No *_stage1.csv files found under {}".format(input_path)
        )

    all_rows = []
    base_fields = None
    seen_frames = set()
    for csv_file in csv_files:
        fieldnames, rows = _read_csv(csv_file)
        if base_fields is None:
            base_fields = fieldnames
        elif fieldnames != base_fields:
            raise ValueError(
                "CSV header differs from the first file: {}".format(csv_file)
            )

        for row in rows:
            row["frame_id"] = int(row["frame_id"])
            row["gt_valid"] = int(row["gt_valid"])
            for field in (
                "iou_fast",
                "iou_sgla",
                "iou_full",
                "delta_full_fast",
            ):
                row[field] = float(row[field])

            frame_key = (row["dataset"], row["sequence"], row["frame_id"])
            if frame_key in seen_frames:
                raise ValueError(
                    "Duplicate dataset/sequence/frame row: {}".format(frame_key)
                )
            seen_frames.add(frame_key)
            all_rows.append(row)

    all_rows.sort(
        key=lambda row: (row["dataset"], row["sequence"], row["frame_id"])
    )
    return input_path, base_fields, all_rows, csv_files


def success_auc(rows, value_field):
    """Match this repository's macro-sequence success AUC definition."""
    by_sequence = defaultdict(list)
    for row in rows:
        by_sequence[(row["dataset"], row["sequence"])].append(
            float(row[value_field])
        )

    if not by_sequence:
        raise ValueError("Cannot calculate AUC from zero rows")

    sequence_aucs = []
    for overlaps in by_sequence.values():
        denominator = float(len(overlaps))
        curve = [
            sum(overlap > threshold for overlap in overlaps) / denominator
            for threshold in OVERLAP_THRESHOLDS
        ]
        sequence_aucs.append(sum(curve) / len(curve))
    return sum(sequence_aucs) / len(sequence_aucs)


def _mean(values):
    return sum(values) / len(values) if values else float("nan")


def _percentile(values, percentile):
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _budget_label(budget):
    if float(budget).is_integer():
        return str(int(budget))
    return str(budget).replace(".", "p")


def _format_number(value):
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return "{:.8f}".format(value)
    return value


def run_oracle(rows, budgets):
    fast_auc = success_auc(rows, "iou_fast") * 100.0
    sgla_auc = success_auc(rows, "iou_sgla") * 100.0
    full_auc = success_auc(rows, "iou_full") * 100.0

    decision_indices = [
        index for index, row in enumerate(rows) if row["frame_id"] > 0
    ]
    ranked_indices = sorted(
        decision_indices,
        key=lambda index: (
            rows[index]["delta_full_fast"],
            rows[index]["dataset"],
            rows[index]["sequence"],
            -rows[index]["frame_id"],
        ),
        reverse=True,
    )

    valid_deltas = [
        row["delta_full_fast"]
        for row in rows
        if row["frame_id"] > 0 and row["gt_valid"] == 1
    ]
    positive_delta_ratio = (
        sum(delta > 0.0 for delta in valid_deltas) / len(valid_deltas)
        if valid_deltas
        else float("nan")
    )

    summaries = []
    for budget in budgets:
        if budget == 0.0:
            num_full = 0
        elif budget == 100.0:
            num_full = len(decision_indices)
        else:
            num_full = int(
                math.ceil(len(decision_indices) * budget / 100.0)
            )

        selected = set(ranked_indices[:num_full])
        label = _budget_label(budget)
        iou_field = "iou_oracle_{}".format(label)
        path_field = "oracle_path_{}".format(label)

        for index, row in enumerate(rows):
            if row["frame_id"] == 0:
                row[path_field] = "init"
                row[iou_field] = row["iou_fast"]
            elif index in selected:
                row[path_field] = "full"
                row[iou_field] = row["iou_full"]
            else:
                row[path_field] = "fast"
                row[iou_field] = row["iou_fast"]

        oracle_auc = success_auc(rows, iou_field) * 100.0
        valid_oracle_ious = [
            row[iou_field] for row in rows if row["gt_valid"] == 1
        ]
        selected_deltas = [
            rows[index]["delta_full_fast"] for index in selected
        ]
        full_gap = full_auc - fast_auc
        recovery = (
            (oracle_auc - fast_auc) / full_gap
            if full_gap > 1e-12
            else float("nan")
        )

        summaries.append(
            {
                "budget_percent": budget,
                "num_frames": len(rows),
                "num_decision_frames": len(decision_indices),
                "num_full": num_full,
                "actual_full_ratio": (
                    num_full / len(decision_indices)
                    if decision_indices
                    else 0.0
                ),
                "oracle_auc": oracle_auc,
                "oracle_gain_vs_fast": oracle_auc - fast_auc,
                "recovery_ratio": recovery,
                "oracle_mean_iou_valid": _mean(valid_oracle_ious),
                "fast_auc": fast_auc,
                "sgla_auc": sgla_auc,
                "full_auc": full_auc,
                "positive_delta_ratio": positive_delta_ratio,
                "delta_mean_valid": _mean(valid_deltas),
                "delta_p50_valid": _percentile(valid_deltas, 0.50),
                "delta_p90_valid": _percentile(valid_deltas, 0.90),
                "delta_p95_valid": _percentile(valid_deltas, 0.95),
                "selected_delta_mean": _mean(selected_deltas),
                "selected_delta_min": min(selected_deltas)
                if selected_deltas
                else float("nan"),
            }
        )

    zero_summary = next(
        (item for item in summaries if item["budget_percent"] == 0.0), None
    )
    full_summary = next(
        (item for item in summaries if item["budget_percent"] == 100.0), None
    )
    if zero_summary and abs(zero_summary["oracle_auc"] - fast_auc) > 1e-9:
        raise AssertionError("0% Oracle AUC must equal Fast AUC")
    if full_summary and abs(full_summary["oracle_auc"] - full_auc) > 1e-9:
        raise AssertionError("100% Oracle AUC must equal Full AUC")

    return summaries


def _write_merged(path, base_fields, rows, budgets):
    extra_fields = []
    for budget in budgets:
        label = _budget_label(budget)
        extra_fields.extend(
            ["oracle_path_{}".format(label), "iou_oracle_{}".format(label)]
        )
    fields = list(base_fields) + [
        field for field in extra_fields if field not in base_fields
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: _format_number(row.get(field, "")) for field in fields}
            )


def _write_summary(path, summaries):
    fields = list(summaries[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {key: _format_number(value) for key, value in summary.items()}
            )


def _parse_budgets(values):
    budgets = sorted(set(float(value) for value in values))
    if any(value < 0.0 or value > 100.0 for value in budgets):
        raise ValueError("Every budget must be in [0, 100]")
    return budgets


def _default_output_paths(input_path, rows):
    output_dir = input_path if input_path.is_dir() else input_path.parent
    dataset_name = rows[0]["dataset"] or "dataset"
    safe_name = dataset_name.replace("/", "_").replace("\\", "_")
    return (
        output_dir / "{}_stage1_all.csv".format(safe_name),
        output_dir / "{}_oracle_budget.csv".format(safe_name),
    )


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Calculate fixed-budget Oracle AUC from stage-1 CSV files."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--summary_csv", default=None)
    parser.add_argument(
        "--budgets",
        nargs="+",
        default=["0", "10", "20", "30", "50", "100"],
    )
    return parser


def main():
    args = _build_parser().parse_args()
    budgets = _parse_budgets(args.budgets)
    input_path, base_fields, rows, csv_files = load_rows(args.input)
    default_output, default_summary = _default_output_paths(input_path, rows)
    output_path = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv
        else default_output
    )
    summary_path = (
        Path(args.summary_csv).expanduser().resolve()
        if args.summary_csv
        else default_summary
    )

    summaries = run_oracle(rows, budgets)
    _write_merged(output_path, base_fields, rows, budgets)
    _write_summary(summary_path, summaries)

    print(
        "Loaded {} frames from {} sequence CSV files.".format(
            len(rows), len(csv_files)
        )
    )
    print(
        "budget  full_frames  actual_ratio  oracle_auc  gain_vs_fast  recovery"
    )
    for item in summaries:
        recovery = item["recovery_ratio"]
        recovery_text = (
            "n/a" if math.isnan(recovery) else "{:.2%}".format(recovery)
        )
        print(
            "{:>6.1f}%  {:>11d}  {:>11.2%}  {:>10.4f}  {:>12.4f}  {:>8}".format(
                item["budget_percent"],
                item["num_full"],
                item["actual_full_ratio"],
                item["oracle_auc"],
                item["oracle_gain_vs_fast"],
                recovery_text,
            )
        )
    print("Merged frame CSV: {}".format(output_path))
    print("Budget summary CSV: {}".format(summary_path))


if __name__ == "__main__":
    main()
