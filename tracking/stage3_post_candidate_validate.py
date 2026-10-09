"""Evaluate observable post-candidate selection and fusion policies.

This Stage-3C diagnostic consumes Stage-2 CSV files only. It tests whether
response peaks and agreement between candidate boxes can select a useful
candidate after a fixed subset of independent blocks has been evaluated.
"""

import argparse
import csv
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import stage2_candidate_oracle as stage2
import stage3_router_validate as stage3


CANDIDATE_LAYERS = stage2.CANDIDATE_LAYERS
TOLERANCE = 1e-12
DEFAULT_CANDIDATE_SETS = (
    "7,11",
    "7,9,12",
    "7,9,11,12",
    "7,8,9,11,12",
    "7,8,9,10,11,12",
)


class MacroSuccessAUC:
    """Fast macro-sequence implementation of the repository AUC metric."""

    def __init__(self, rows):
        self.groups = np.asarray(
            ["{}\x1f{}".format(row["dataset"], row["sequence"]) for row in rows],
            dtype=object,
        )
        counts = Counter(self.groups.tolist())
        self.row_weights = np.asarray(
            [1.0 / counts[group] for group in self.groups], dtype=np.float64
        )
        self.thresholds = np.asarray(
            stage2.OVERLAP_THRESHOLDS, dtype=np.float64
        )

    def __call__(self, values, row_mask=None):
        values = np.asarray(values, dtype=np.float64)
        if len(values) != len(self.groups):
            raise ValueError("AUC value count differs from row count")
        if row_mask is None:
            row_mask = np.ones(len(values), dtype=np.bool_)
        else:
            row_mask = np.asarray(row_mask, dtype=np.bool_)
        selected_groups = set(self.groups[row_mask].tolist())
        if not selected_groups:
            raise ValueError("Cannot calculate AUC from zero sequences")

        threshold_hits = np.searchsorted(
            self.thresholds, values, side="left"
        ).astype(np.float64)
        contributions = threshold_hits / len(self.thresholds)
        return float(
            100.0
            * np.sum(contributions[row_mask] * self.row_weights[row_mask])
            / len(selected_groups)
        )


def _parse_candidate_sets(values):
    parsed = []
    seen = set()
    valid_layers = set(CANDIDATE_LAYERS)
    for value in values:
        try:
            layers = tuple(sorted(set(int(item) for item in value.split(","))))
        except ValueError as error:
            raise ValueError(
                "Invalid candidate set {!r}; use a value such as 7,9,11".format(
                    value
                )
            ) from error
        if not layers or not set(layers).issubset(valid_layers):
            raise ValueError(
                "Candidate set {!r} must contain only L7-L12".format(value)
            )
        if layers not in seen:
            parsed.append(layers)
            seen.add(layers)
    return parsed


def _required_post_fields():
    fields = {"gt_x", "gt_y", "gt_w", "gt_h"}
    for layer in CANDIDATE_LAYERS:
        fields.update(
            {
                "selector_prob_layer{}".format(layer),
                "response_peak_layer{}".format(layer),
            }
        )
        fields.update(
            "bbox_layer{}_{}".format(layer, coordinate)
            for coordinate in ("x", "y", "w", "h")
        )
    return fields


def _as_float(row, field):
    value = row.get(field)
    if value is None or str(value).strip() == "":
        raise ValueError(
            "Missing {} at {}/{} frame {}".format(
                field,
                row.get("dataset", ""),
                row.get("sequence", ""),
                row.get("frame_id", ""),
            )
        )
    return float(value)


def _load_arrays(rows, fieldnames):
    missing = _required_post_fields().difference(fieldnames)
    if missing:
        raise ValueError(
            "Stage-2 CSV is missing post-candidate fields: {}".format(
                ", ".join(sorted(missing))
            )
        )

    num_rows = len(rows)
    num_layers = len(CANDIDATE_LAYERS)
    selector = np.zeros((num_rows, num_layers), dtype=np.float64)
    peaks = np.zeros((num_rows, num_layers), dtype=np.float64)
    boxes = np.zeros((num_rows, num_layers, 4), dtype=np.float64)
    ground_truth = np.zeros((num_rows, 4), dtype=np.float64)
    decision = np.asarray(
        [row["frame_id"] > 0 for row in rows], dtype=np.bool_
    )
    valid = np.asarray(
        [row["frame_id"] > 0 and row["gt_valid"] == 1 for row in rows],
        dtype=np.bool_,
    )

    for row_index, row in enumerate(rows):
        ground_truth[row_index] = [
            _as_float(row, "gt_{}".format(coordinate))
            for coordinate in ("x", "y", "w", "h")
        ]
        for offset, layer in enumerate(CANDIDATE_LAYERS):
            boxes[row_index, offset] = [
                _as_float(
                    row, "bbox_layer{}_{}".format(layer, coordinate)
                )
                for coordinate in ("x", "y", "w", "h")
            ]
            if decision[row_index]:
                selector[row_index, offset] = _as_float(
                    row, "selector_prob_layer{}".format(layer)
                )
                peaks[row_index, offset] = _as_float(
                    row, "response_peak_layer{}".format(layer)
                )

    return selector, peaks, boxes, ground_truth, decision, valid


def _box_iou(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first_w = np.maximum(first[..., 2], 0.0)
    first_h = np.maximum(first[..., 3], 0.0)
    second_w = np.maximum(second[..., 2], 0.0)
    second_h = np.maximum(second[..., 3], 0.0)
    left = np.maximum(first[..., 0], second[..., 0])
    top = np.maximum(first[..., 1], second[..., 1])
    right = np.minimum(first[..., 0] + first_w, second[..., 0] + second_w)
    bottom = np.minimum(first[..., 1] + first_h, second[..., 1] + second_h)
    intersection = np.maximum(right - left, 0.0) * np.maximum(
        bottom - top, 0.0
    )
    union = first_w * first_h + second_w * second_h - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 1e-12,
    )


def _candidate_consensus(boxes):
    num_candidates = boxes.shape[1]
    if num_candidates == 1:
        return np.ones((len(boxes), 1), dtype=np.float64)
    consensus = np.zeros((len(boxes), num_candidates), dtype=np.float64)
    for first in range(num_candidates):
        for second in range(first + 1, num_candidates):
            overlap = _box_iou(boxes[:, first], boxes[:, second])
            consensus[:, first] += overlap
            consensus[:, second] += overlap
    return consensus / (num_candidates - 1)


def _row_minmax(values):
    minimum = values.min(axis=1, keepdims=True)
    span = values.max(axis=1, keepdims=True) - minimum
    return np.divide(
        values - minimum,
        span,
        out=np.zeros_like(values, dtype=np.float64),
        where=span > 1e-12,
    )


def _selected_ious(candidate_ious, choices):
    return candidate_ious[np.arange(len(candidate_ious)), choices]


def _fused_iou(fused_boxes, ground_truth, fast_iou, valid):
    values = fast_iou.copy()
    values[valid] = _box_iou(fused_boxes[valid], ground_truth[valid])
    return values


def _weighted_boxes(boxes, scores):
    weights = scores + 1e-6
    weights = weights / weights.sum(axis=1, keepdims=True)
    return np.sum(boxes * weights[:, :, None], axis=1)


def _layer_distribution(choices, layers, decision):
    counts = Counter(int(layers[index]) for index in choices[decision])
    total = max(int(decision.sum()), 1)
    return "|".join(
        "L{}:{:.2%}".format(layer, counts.get(layer, 0) / total)
        for layer in layers
    )


def _weight_grid(step):
    steps = int(round(1.0 / step))
    if steps < 1 or abs(steps * step - 1.0) > 1e-9:
        raise ValueError("--weight_step must divide 1.0 exactly")
    weights = []
    for selector_units in range(steps + 1):
        for peak_units in range(steps - selector_units + 1):
            consensus_units = steps - selector_units - peak_units
            weights.append(
                np.asarray(
                    [selector_units, peak_units, consensus_units],
                    dtype=np.float64,
                )
                / steps
            )
    return weights


def _oof_weighted_selection(
    rows,
    candidate_ious,
    normalized_scores,
    decision,
    fold_ids,
    auc,
    weight_grid,
):
    oof_values = np.asarray([row["iou_fast"] for row in rows], dtype=np.float64)
    oof_choices = np.zeros(len(rows), dtype=np.int64)
    fold_rows = []
    row_folds = np.full(len(rows), -1, dtype=np.int64)
    decision_indices = np.flatnonzero(decision)
    row_folds[decision_indices] = fold_ids
    group_to_fold = {}
    for row_index in decision_indices:
        row = rows[row_index]
        group_to_fold[(row["dataset"], row["sequence"])] = int(
            row_folds[row_index]
        )
    for row_index, row in enumerate(rows):
        if row_folds[row_index] < 0:
            row_folds[row_index] = group_to_fold[
                (row["dataset"], row["sequence"])
            ]

    num_folds = int(fold_ids.max()) + 1
    best_by_fold = [None] * num_folds
    for weights in weight_grid:
        scores = np.tensordot(normalized_scores, weights, axes=([2], [0]))
        choices = np.argmax(scores, axis=1)
        values = _selected_ious(candidate_ious, choices)
        for fold in range(num_folds):
            train_mask = row_folds != fold
            train_auc = auc(values, train_mask)
            key = (
                train_auc,
                float(weights[2]),
                float(weights[1]),
                float(weights[0]),
            )
            if best_by_fold[fold] is None or key > best_by_fold[fold][0]:
                test_mask = (row_folds == fold) & decision
                best_by_fold[fold] = (key, weights.copy(), train_auc)
                oof_choices[test_mask] = choices[test_mask]
                oof_values[test_mask] = values[test_mask]

    for fold, best in enumerate(best_by_fold):
        _, weights, train_auc = best
        train_mask = row_folds != fold
        train_groups = set(auc.groups[train_mask].tolist())
        test_groups = set(auc.groups[row_folds == fold].tolist())
        fold_rows.append(
            {
                "fold": fold,
                "train_sequences": len(train_groups),
                "test_sequences": len(test_groups),
                "selector_weight": weights[0],
                "peak_weight": weights[1],
                "consensus_weight": weights[2],
                "train_auc": train_auc,
            }
        )
    return oof_values, oof_choices, fold_rows


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


def analyze(
    rows,
    fieldnames,
    candidate_sets,
    folds,
    seed,
    weight_step,
):
    selector, peaks, boxes, ground_truth, decision, valid = _load_arrays(
        rows, fieldnames
    )
    auc = MacroSuccessAUC(rows)
    fast_iou = np.asarray([row["iou_fast"] for row in rows], dtype=np.float64)
    sgla_iou = np.asarray([row["iou_sgla"] for row in rows], dtype=np.float64)
    all_candidate_ious = np.asarray(
        [
            [row["iou_layer{}".format(layer)] for layer in CANDIDATE_LAYERS]
            for row in rows
        ],
        dtype=np.float64,
    )
    fast_auc = auc(fast_iou)
    sgla_auc = auc(sgla_iou)
    reference_fast_auc = stage2.success_auc(rows, "iou_fast") * 100.0
    if abs(fast_auc - reference_fast_auc) > 1e-10:
        raise AssertionError("Fast AUC implementation does not match Stage-2")

    groups = np.asarray(
        [
            "{}\x1f{}".format(rows[index]["dataset"], rows[index]["sequence"])
            for index in np.flatnonzero(decision)
        ],
        dtype=object,
    )
    fold_ids = stage3.assign_group_folds(groups, folds, seed)
    weight_grid = _weight_grid(weight_step)
    layer_to_offset = {
        layer: offset for offset, layer in enumerate(CANDIDATE_LAYERS)
    }

    summaries = []
    weight_rows = []
    for layers in candidate_sets:
        offsets = tuple(layer_to_offset[layer] for layer in layers)
        candidate_ious = all_candidate_ious[:, offsets]
        subset_selector = selector[:, offsets]
        subset_peaks = peaks[:, offsets]
        subset_boxes = boxes[:, offsets]
        consensus = _candidate_consensus(subset_boxes)

        selector_choices = np.argmax(subset_selector, axis=1)
        peak_choices = np.argmax(subset_peaks, axis=1)
        medoid_choices = np.argmax(consensus, axis=1)
        equal_choices = np.argmax(
            _row_minmax(subset_selector)
            + _row_minmax(subset_peaks)
            + _row_minmax(consensus),
            axis=1,
        )

        normalized_scores = np.stack(
            [
                _row_minmax(subset_selector),
                _row_minmax(subset_peaks),
                _row_minmax(consensus),
            ],
            axis=2,
        )
        oof_values, oof_choices, current_weight_rows = (
            _oof_weighted_selection(
                rows,
                candidate_ious,
                normalized_scores,
                decision,
                fold_ids,
                auc,
                weight_grid,
            )
        )
        layer_text = "|".join(str(layer) for layer in layers)
        for item in current_weight_rows:
            item["layers"] = layer_text
            weight_rows.append(item)

        forced_oracle = candidate_ious.max(axis=1)
        action_oracle = np.maximum(fast_iou, forced_oracle)
        forced_oracle_auc = auc(forced_oracle)
        action_oracle_auc = auc(action_oracle)
        forced_gain = forced_oracle_auc - fast_auc

        mean_boxes = subset_boxes.mean(axis=1)
        median_boxes = np.median(subset_boxes, axis=1)
        peak_weighted_boxes = _weighted_boxes(
            subset_boxes, _row_minmax(subset_peaks)
        )
        consensus_weighted_boxes = _weighted_boxes(
            subset_boxes, consensus
        )
        policies = (
            (
                "selector_prob",
                _selected_ious(candidate_ious, selector_choices),
                selector_choices,
                0,
                0,
            ),
            (
                "response_peak",
                _selected_ious(candidate_ious, peak_choices),
                peak_choices,
                0,
                0,
            ),
            (
                "box_medoid",
                _selected_ious(candidate_ious, medoid_choices),
                medoid_choices,
                0,
                0,
            ),
            (
                "equal_score_select",
                _selected_ious(candidate_ious, equal_choices),
                equal_choices,
                0,
                0,
            ),
            ("oof_weighted_select", oof_values, oof_choices, 0, 1),
            (
                "bbox_mean",
                _fused_iou(mean_boxes, ground_truth, fast_iou, valid),
                None,
                0,
                0,
            ),
            (
                "bbox_median",
                _fused_iou(median_boxes, ground_truth, fast_iou, valid),
                None,
                0,
                0,
            ),
            (
                "peak_weighted_bbox",
                _fused_iou(
                    peak_weighted_boxes, ground_truth, fast_iou, valid
                ),
                None,
                0,
                0,
            ),
            (
                "consensus_weighted_bbox",
                _fused_iou(
                    consensus_weighted_boxes,
                    ground_truth,
                    fast_iou,
                    valid,
                ),
                None,
                0,
                0,
            ),
            ("forced_candidate_oracle", forced_oracle, None, 1, 0),
            ("fast_plus_candidate_oracle", action_oracle, None, 1, 0),
        )
        for method, values, choices, uses_gt, trained_oof in policies:
            method_auc = auc(values)
            summaries.append(
                {
                    "layers": layer_text,
                    "num_blocks": len(layers),
                    "method": method,
                    "uses_ground_truth": uses_gt,
                    "trained_oof": trained_oof,
                    "candidate_blocks_per_frame": len(layers),
                    "auc": method_auc,
                    "gain_vs_fast": method_auc - fast_auc,
                    "gain_vs_sgla": method_auc - sgla_auc,
                    "recovery_of_forced_oracle_gain": (
                        (method_auc - fast_auc) / forced_gain
                        if forced_gain > TOLERANCE
                        else float("nan")
                    ),
                    "gap_to_forced_oracle": forced_oracle_auc - method_auc,
                    "gap_to_action_oracle": action_oracle_auc - method_auc,
                    "mean_iou_valid": float(np.mean(values[valid])),
                    "selected_layer_distribution": (
                        _layer_distribution(choices, layers, decision)
                        if choices is not None
                        else ""
                    ),
                }
            )

        if len(layers) == len(CANDIDATE_LAYERS):
            selector_auc = auc(
                _selected_ious(candidate_ious, selector_choices)
            )
            confidence_auc = auc(
                _selected_ious(candidate_ious, peak_choices)
            )
            expected_confidence_auc = (
                stage2.success_auc(rows, "iou_confidence_select") * 100.0
            )
            if abs(selector_auc - sgla_auc) > 1e-8:
                raise AssertionError(
                    "All-candidate selector does not reproduce SGLA AUC"
                )
            if abs(confidence_auc - expected_confidence_auc) > 1e-8:
                raise AssertionError(
                    "All-candidate peak policy does not reproduce confidence AUC"
                )

    return summaries, weight_rows, fast_auc, sgla_auc


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate candidate confidence, agreement, and bbox fusion from "
            "Stage-2 CSV files."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output_prefix", default=None)
    parser.add_argument(
        "--candidate_sets", nargs="+", default=list(DEFAULT_CANDIDATE_SETS)
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--weight_step", type=float, default=0.1)
    return parser


def main():
    args = _build_parser().parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if not 0.0 < args.weight_step <= 1.0:
        raise ValueError("--weight_step must be in (0, 1]")
    candidate_sets = _parse_candidate_sets(args.candidate_sets)
    input_path, fieldnames, rows, csv_files = stage2.load_rows(args.input)
    summaries, weight_rows, fast_auc, sgla_auc = analyze(
        rows,
        fieldnames,
        candidate_sets,
        args.folds,
        args.seed,
        args.weight_step,
    )

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (input_path if input_path.is_dir() else input_path.parent)
    )
    datasets = sorted({row["dataset"] for row in rows})
    dataset_name = datasets[0] if len(datasets) == 1 else "multi_dataset"
    dataset_name = dataset_name.replace("/", "_").replace("\\", "_")
    prefix = args.output_prefix or "{}_stage3_post_candidate".format(
        dataset_name
    )
    summary_path = output_dir / "{}_summary.csv".format(prefix)
    weights_path = output_dir / "{}_oof_weights.csv".format(prefix)
    _write_csv(summary_path, list(summaries[0].keys()), summaries)
    _write_csv(weights_path, list(weight_rows[0].keys()), weight_rows)

    print(
        "Loaded {} frames from {} CSV file(s); Fast={:.4f}, SGLA={:.4f}.".format(
            len(rows), len(csv_files), fast_auc, sgla_auc
        )
    )
    print(
        "Post-candidate policies use K evaluated blocks on every tracked frame."
    )
    print("layers             method                       auc  gain_fast  gain_sgla")
    shown_methods = {
        "selector_prob",
        "response_peak",
        "box_medoid",
        "oof_weighted_select",
        "bbox_median",
        "forced_candidate_oracle",
        "fast_plus_candidate_oracle",
    }
    for row in summaries:
        if row["method"] not in shown_methods:
            continue
        print(
            "{:<18s} {:<27s} {:>8.4f}  {:>9.4f}  {:>9.4f}".format(
                row["layers"],
                row["method"],
                row["auc"],
                row["gain_vs_fast"],
                row["gain_vs_sgla"],
            )
        )
    print("Summary CSV: {}".format(summary_path))
    print("OOF weight CSV: {}".format(weights_path))


if __name__ == "__main__":
    main()
