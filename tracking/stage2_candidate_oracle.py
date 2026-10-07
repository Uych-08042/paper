"""Analyze independent SGLA candidate blocks and frame-budget Oracles."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


CANDIDATE_LAYERS = tuple(range(7, 13))
OVERLAP_THRESHOLDS = [index * 0.05 for index in range(21)]

METHODS = (
    ("fast", "iou_fast", 0, False),
    ("sgla_selector", "iou_sgla", 1, False),
) + tuple(
    ("fixed_layer{}".format(layer), "iou_layer{}".format(layer), 1, False)
    for layer in CANDIDATE_LAYERS
) + (
    ("confidence_select", "iou_confidence_select", 6, False),
    ("ensemble_mean", "iou_ensemble_mean", 6, False),
    ("best_candidate_oracle", "iou_best_candidate", 6, True),
)

BUDGET_POLICIES = (
    ("sgla_selector", "iou_sgla", 1),
    ("confidence_select", "iou_confidence_select", 6),
    ("ensemble_mean", "iou_ensemble_mean", 6),
    ("best_candidate_oracle", "iou_best_candidate", 6),
)

REQUIRED_FIELDS = {
    "dataset",
    "sequence",
    "frame_id",
    "gt_valid",
    "iou_fast",
    "iou_sgla",
    "iou_confidence_select",
    "iou_ensemble_mean",
    "iou_best_candidate",
    "sgla_selected_layer",
    "confidence_selected_layer",
    "best_candidate_layer",
    "best_candidate_tie_count",
    "selector_hit",
    "confidence_hit",
    "selector_regret",
    "confidence_regret",
}.union("iou_layer{}".format(layer) for layer in CANDIDATE_LAYERS)

NUMERIC_IOU_FIELDS = tuple(dict.fromkeys(field for _, field, _, _ in METHODS))
OPTIONAL_INT_FIELDS = (
    "sgla_selected_layer",
    "confidence_selected_layer",
    "best_candidate_layer",
    "best_candidate_tie_count",
    "selector_hit",
    "confidence_hit",
)
OPTIONAL_FLOAT_FIELDS = (
    "selector_regret",
    "confidence_regret",
    "selector_max",
    "selector_margin",
    "selector_entropy",
    "selector_selected_probability",
    "confidence_selected_peak",
)


def _optional_number(value, converter):
    if value is None or str(value).strip() == "":
        return None
    return converter(value)


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
        csv_files = sorted(input_path.rglob("*_stage2_candidates.csv"))
    else:
        raise FileNotFoundError("Input does not exist: {}".format(input_path))

    if not csv_files:
        raise FileNotFoundError(
            "No *_stage2_candidates.csv files found under {}".format(
                input_path
            )
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
            for field in NUMERIC_IOU_FIELDS:
                row[field] = float(row[field])
            for field in OPTIONAL_INT_FIELDS:
                row[field] = _optional_number(row.get(field), int)
            for field in OPTIONAL_FLOAT_FIELDS:
                row[field] = _optional_number(row.get(field), float)

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


def _valid_decision_rows(rows):
    return [
        row
        for row in rows
        if row["frame_id"] > 0 and row["gt_valid"] == 1
    ]


def _regret_metrics(rows, prefix):
    regret_field = "{}_regret".format(prefix)
    hit_field = "{}_hit".format(prefix)
    regrets = [
        row[regret_field]
        for row in rows
        if row.get(regret_field) is not None
    ]
    hits = [
        row[hit_field] for row in rows if row.get(hit_field) is not None
    ]
    return {
        "exact_best_hit_rate": _mean(hits),
        "near_best_001_rate": _mean([value <= 0.01 for value in regrets]),
        "near_best_005_rate": _mean([value <= 0.05 for value in regrets]),
        "regret_mean": _mean(regrets),
        "regret_p50": _percentile(regrets, 0.50),
        "regret_p90": _percentile(regrets, 0.90),
        "regret_p95": _percentile(regrets, 0.95),
    }


def method_summaries(rows):
    valid_rows = _valid_decision_rows(rows)
    fast_auc = success_auc(rows, "iou_fast") * 100.0
    selector_metrics = _regret_metrics(valid_rows, "selector")
    confidence_metrics = _regret_metrics(valid_rows, "confidence")
    summaries = []

    for method, iou_field, candidate_blocks, uses_ground_truth in METHODS:
        auc = success_auc(rows, iou_field) * 100.0
        valid_ious = [row[iou_field] for row in valid_rows]
        deltas = [row[iou_field] - row["iou_fast"] for row in valid_rows]
        summary = {
            "method": method,
            "iou_field": iou_field,
            "candidate_blocks_per_deep_frame": candidate_blocks,
            "uses_ground_truth": int(uses_ground_truth),
            "num_frames": len(rows),
            "num_valid_decision_frames": len(valid_rows),
            "auc": auc,
            "gain_vs_fast": auc - fast_auc,
            "mean_iou_valid": _mean(valid_ious),
            "mean_delta_vs_fast_valid": _mean(deltas),
            "positive_delta_ratio": _mean([delta > 0.0 for delta in deltas]),
            "exact_best_hit_rate": float("nan"),
            "near_best_001_rate": float("nan"),
            "near_best_005_rate": float("nan"),
            "regret_mean": float("nan"),
            "regret_p50": float("nan"),
            "regret_p90": float("nan"),
            "regret_p95": float("nan"),
        }
        if method == "sgla_selector":
            summary.update(selector_metrics)
        elif method == "confidence_select":
            summary.update(confidence_metrics)
        elif method == "best_candidate_oracle":
            summary.update(
                {
                    "exact_best_hit_rate": 1.0,
                    "near_best_001_rate": 1.0,
                    "near_best_005_rate": 1.0,
                    "regret_mean": 0.0,
                    "regret_p50": 0.0,
                    "regret_p90": 0.0,
                    "regret_p95": 0.0,
                }
            )
        summaries.append(summary)
    return summaries


def layer_summaries(rows):
    decision_rows = [row for row in rows if row["frame_id"] > 0]
    valid_rows = _valid_decision_rows(rows)
    fast_auc = success_auc(rows, "iou_fast") * 100.0
    summaries = []
    for layer in CANDIDATE_LAYERS:
        selected_rows = [
            row for row in decision_rows if row["sgla_selected_layer"] == layer
        ]
        confidence_rows = [
            row
            for row in decision_rows
            if row["confidence_selected_layer"] == layer
        ]
        best_rows = [
            row for row in valid_rows if row["best_candidate_layer"] == layer
        ]
        iou_field = "iou_layer{}".format(layer)
        best_or_tied_rows = [
            row
            for row in valid_rows
            if row["iou_best_candidate"] - row[iou_field] <= 1e-12
        ]
        fixed_auc = success_auc(rows, iou_field) * 100.0
        summaries.append(
            {
                "layer": layer,
                "fixed_auc": fixed_auc,
                "fixed_gain_vs_fast": fixed_auc - fast_auc,
                "selector_count": len(selected_rows),
                "selector_ratio": (
                    len(selected_rows) / len(decision_rows)
                    if decision_rows
                    else float("nan")
                ),
                "confidence_count": len(confidence_rows),
                "confidence_ratio": (
                    len(confidence_rows) / len(decision_rows)
                    if decision_rows
                    else float("nan")
                ),
                "best_count_valid": len(best_rows),
                "best_ratio_valid": (
                    len(best_rows) / len(valid_rows)
                    if valid_rows
                    else float("nan")
                ),
                "best_or_tied_count_valid": len(best_or_tied_rows),
                "best_or_tied_ratio_valid": (
                    len(best_or_tied_rows) / len(valid_rows)
                    if valid_rows
                    else float("nan")
                ),
                "selector_hit_count": sum(
                    row["selector_hit"] == 1
                    for row in selected_rows
                    if row["gt_valid"] == 1
                ),
                "mean_iou_when_selector_chose_layer": _mean(
                    [row[iou_field] for row in selected_rows if row["gt_valid"]]
                ),
            }
        )
    return summaries


def _budget_label(budget):
    if float(budget).is_integer():
        return str(int(budget))
    return str(budget).replace(".", "p")


def run_budget_oracles(rows, budgets):
    fast_auc = success_auc(rows, "iou_fast") * 100.0
    decision_indices = [
        index for index, row in enumerate(rows) if row["frame_id"] > 0
    ]
    summaries = []
    generated_fields = []

    for policy, policy_field, candidate_blocks in BUDGET_POLICIES:
        policy_auc = success_auc(rows, policy_field) * 100.0
        ranked_indices = sorted(
            decision_indices,
            key=lambda index: (
                rows[index][policy_field] - rows[index]["iou_fast"],
                rows[index]["dataset"],
                rows[index]["sequence"],
                -rows[index]["frame_id"],
            ),
            reverse=True,
        )
        valid_deltas = [
            row[policy_field] - row["iou_fast"]
            for row in rows
            if row["frame_id"] > 0 and row["gt_valid"] == 1
        ]

        for budget in budgets:
            if budget == 0.0:
                num_deep = 0
            elif budget == 100.0:
                num_deep = len(decision_indices)
            else:
                num_deep = int(
                    math.ceil(len(decision_indices) * budget / 100.0)
                )
            selected = set(ranked_indices[:num_deep])
            label = _budget_label(budget)
            path_field = "oracle_path_{}_{}".format(policy, label)
            iou_field = "iou_oracle_{}_{}".format(policy, label)
            generated_fields.extend((path_field, iou_field))

            for index, row in enumerate(rows):
                if row["frame_id"] == 0:
                    row[path_field] = "init"
                    row[iou_field] = row["iou_fast"]
                elif index in selected:
                    row[path_field] = policy
                    row[iou_field] = row[policy_field]
                else:
                    row[path_field] = "fast"
                    row[iou_field] = row["iou_fast"]

            oracle_auc = success_auc(rows, iou_field) * 100.0
            selected_deltas = [
                rows[index][policy_field] - rows[index]["iou_fast"]
                for index in selected
            ]
            actual_ratio = (
                num_deep / len(decision_indices) if decision_indices else 0.0
            )
            policy_gap = policy_auc - fast_auc
            recovery = (
                (oracle_auc - fast_auc) / policy_gap
                if policy_gap > 1e-12
                else float("nan")
            )
            summaries.append(
                {
                    "policy": policy,
                    "policy_iou_field": policy_field,
                    "budget_percent": budget,
                    "candidate_blocks_per_deep_frame": candidate_blocks,
                    "equivalent_candidate_blocks_per_frame": (
                        actual_ratio * candidate_blocks
                    ),
                    "num_frames": len(rows),
                    "num_decision_frames": len(decision_indices),
                    "num_deep": num_deep,
                    "actual_deep_ratio": actual_ratio,
                    "fast_auc": fast_auc,
                    "policy_100_auc": policy_auc,
                    "oracle_auc": oracle_auc,
                    "oracle_gain_vs_fast": oracle_auc - fast_auc,
                    "recovery_ratio": recovery,
                    "positive_delta_ratio": _mean(
                        [delta > 0.0 for delta in valid_deltas]
                    ),
                    "selected_delta_mean": _mean(selected_deltas),
                    "selected_delta_min": (
                        min(selected_deltas)
                        if selected_deltas
                        else float("nan")
                    ),
                }
            )

            if budget == 0.0 and abs(oracle_auc - fast_auc) > 1e-9:
                raise AssertionError("0% Oracle AUC must equal Fast AUC")
            if budget == 100.0 and abs(oracle_auc - policy_auc) > 1e-9:
                raise AssertionError(
                    "100% Oracle AUC must equal the policy AUC for {}".format(
                        policy
                    )
                )

    return summaries, generated_fields


def _format_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return "{:.8f}".format(value)
    return value


def _write_rows(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: _format_value(row.get(field, "")) for field in fields}
            )


def _write_summary(path, summaries):
    if not summaries:
        raise ValueError("Cannot write an empty summary")
    _write_rows(path, list(summaries[0].keys()), summaries)


def _default_output_paths(input_path, rows):
    output_dir = input_path if input_path.is_dir() else input_path.parent
    dataset_name = rows[0]["dataset"] or "dataset"
    safe_name = dataset_name.replace("/", "_").replace("\\", "_")
    return (
        output_dir / "{}_stage2_candidates_all.csv".format(safe_name),
        output_dir / "{}_stage2_method_summary.csv".format(safe_name),
        output_dir / "{}_stage2_budget_summary.csv".format(safe_name),
        output_dir / "{}_stage2_layer_distribution.csv".format(safe_name),
    )


def _parse_budgets(values):
    budgets = sorted(set(float(value) for value in values))
    if any(value < 0.0 or value > 100.0 for value in budgets):
        raise ValueError("Every budget must be in [0, 100]")
    return budgets


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize independent candidate blocks and calculate exact "
            "frame-budget Oracles."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--method_summary_csv", default=None)
    parser.add_argument("--budget_summary_csv", default=None)
    parser.add_argument("--layer_summary_csv", default=None)
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
    defaults = _default_output_paths(input_path, rows)
    output_path = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv
        else defaults[0]
    )
    method_path = (
        Path(args.method_summary_csv).expanduser().resolve()
        if args.method_summary_csv
        else defaults[1]
    )
    budget_path = (
        Path(args.budget_summary_csv).expanduser().resolve()
        if args.budget_summary_csv
        else defaults[2]
    )
    layer_path = (
        Path(args.layer_summary_csv).expanduser().resolve()
        if args.layer_summary_csv
        else defaults[3]
    )

    methods = method_summaries(rows)
    layers = layer_summaries(rows)
    budgets_summary, generated_fields = run_budget_oracles(rows, budgets)
    merged_fields = list(base_fields) + [
        field for field in generated_fields if field not in base_fields
    ]

    _write_rows(output_path, merged_fields, rows)
    _write_summary(method_path, methods)
    _write_summary(budget_path, budgets_summary)
    _write_summary(layer_path, layers)

    print(
        "Loaded {} frames from {} sequence CSV files.".format(
            len(rows), len(csv_files)
        )
    )
    print("method                       blocks       auc  gain_vs_fast")
    for item in methods:
        print(
            "{:<28}  {:>6d}  {:>8.4f}  {:>12.4f}".format(
                item["method"],
                item["candidate_blocks_per_deep_frame"],
                item["auc"],
                item["gain_vs_fast"],
            )
        )

    selector = next(item for item in methods if item["method"] == "sgla_selector")
    confidence = next(
        item for item in methods if item["method"] == "confidence_select"
    )
    print(
        "Selector: hit={:.2%}, near-0.01={:.2%}, mean-regret={:.6f}".format(
            selector["exact_best_hit_rate"],
            selector["near_best_001_rate"],
            selector["regret_mean"],
        )
    )
    print(
        "Confidence: hit={:.2%}, near-0.01={:.2%}, mean-regret={:.6f}".format(
            confidence["exact_best_hit_rate"],
            confidence["near_best_001_rate"],
            confidence["regret_mean"],
        )
    )

    print("policy                     budget  deep_frames  oracle_auc  gain")
    for item in budgets_summary:
        print(
            "{:<26}  {:>6.1f}%  {:>11d}  {:>10.4f}  {:>7.4f}".format(
                item["policy"],
                item["budget_percent"],
                item["num_deep"],
                item["oracle_auc"],
                item["oracle_gain_vs_fast"],
            )
        )

    print("Merged frame CSV: {}".format(output_path))
    print("Method summary CSV: {}".format(method_path))
    print("Budget summary CSV: {}".format(budget_path))
    print("Layer distribution CSV: {}".format(layer_path))


if __name__ == "__main__":
    main()
