"""Stage-3 sequence-level OOF utility routing validation for SGLATrack.

This script consumes Stage-2 CSV files only. It never reads ground truth into
the predictor features. Ground-truth IoUs are used solely as training targets
and for held-out evaluation.
"""

import argparse
import csv
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import stage2_candidate_oracle as stage2


CANDIDATE_LAYERS = stage2.CANDIDATE_LAYERS
TOLERANCE = 1e-12

PROBABILITY_FIELDS = tuple(
    "selector_prob_layer{}".format(layer) for layer in CANDIDATE_LAYERS
)
FEATURE_NAMES = (
    tuple("prob_layer{}".format(layer) for layer in CANDIDATE_LAYERS)
    + ("prob_mean", "prob_std", "prob_max", "prob_margin", "prob_entropy")
    + tuple("delta_prob_layer{}".format(layer) for layer in CANDIDATE_LAYERS)
    + ("delta_prob_l1", "delta_prob_max_abs")
    + (
        "fast_vs_state_dx",
        "fast_vs_state_dy",
        "fast_vs_state_log_w",
        "fast_vs_state_log_h",
        "fast_vs_previous_fast_dx",
        "fast_vs_previous_fast_dy",
        "fast_vs_previous_fast_log_w",
        "fast_vs_previous_fast_log_h",
        "fast_log_aspect",
    )
)

REQUIRED_FEATURE_FIELDS = set(PROBABILITY_FIELDS)
for prefix in ("fast", "sgla"):
    REQUIRED_FEATURE_FIELDS.update(
        "bbox_{}_{}".format(prefix, coordinate)
        for coordinate in ("x", "y", "w", "h")
    )


def _as_float(row, field):
    value = row.get(field)
    if value is None or str(value).strip() == "":
        raise ValueError(
            "Missing numeric field {} at {}/{} frame {}".format(
                field,
                row.get("dataset", ""),
                row.get("sequence", ""),
                row.get("frame_id", ""),
            )
        )
    return float(value)


def _box(row, prefix):
    return np.asarray(
        [
            _as_float(row, "bbox_{}_{}".format(prefix, coordinate))
            for coordinate in ("x", "y", "w", "h")
        ],
        dtype=np.float64,
    )


def _relative_box(current, reference):
    epsilon = 1e-6
    current_width = max(float(current[2]), epsilon)
    current_height = max(float(current[3]), epsilon)
    reference_width = max(float(reference[2]), epsilon)
    reference_height = max(float(reference[3]), epsilon)
    current_center_x = float(current[0]) + 0.5 * current_width
    current_center_y = float(current[1]) + 0.5 * current_height
    reference_center_x = float(reference[0]) + 0.5 * reference_width
    reference_center_y = float(reference[1]) + 0.5 * reference_height
    return (
        (current_center_x - reference_center_x) / reference_width,
        (current_center_y - reference_center_y) / reference_height,
        math.log(current_width / reference_width),
        math.log(current_height / reference_height),
    )


def _probability_statistics(probabilities):
    ordered = np.sort(probabilities)[::-1]
    total = max(float(probabilities.sum()), 1e-12)
    normalized = probabilities / total
    entropy = -float(
        np.sum(normalized * np.log(np.maximum(normalized, 1e-12)))
    ) / math.log(len(probabilities))
    return (
        float(probabilities.mean()),
        float(probabilities.std()),
        float(ordered[0]),
        float(ordered[0] - ordered[1]),
        entropy,
    )


def extract_examples(rows, fieldnames):
    missing = REQUIRED_FEATURE_FIELDS.difference(fieldnames)
    if missing:
        raise ValueError(
            "Stage-2 CSV is missing Stage-3 feature fields: {}".format(
                ", ".join(sorted(missing))
            )
        )

    by_sequence = defaultdict(list)
    for row_index, row in enumerate(rows):
        by_sequence[(row["dataset"], row["sequence"])].append(
            (row_index, row)
        )

    features = []
    utilities = []
    groups = []
    row_indices = []
    for sequence_key in sorted(by_sequence):
        sequence_rows = sorted(
            by_sequence[sequence_key], key=lambda item: item[1]["frame_id"]
        )
        for position, (row_index, row) in enumerate(sequence_rows):
            if row["frame_id"] == 0 or row["gt_valid"] != 1:
                continue
            if position == 0:
                raise ValueError(
                    "Tracked frame has no preceding state: {}".format(
                        sequence_key
                    )
                )

            previous_row = sequence_rows[position - 1][1]
            probabilities = np.asarray(
                [_as_float(row, field) for field in PROBABILITY_FIELDS],
                dtype=np.float64,
            )
            if previous_row["frame_id"] == 0:
                previous_probabilities = probabilities.copy()
            else:
                previous_probabilities = np.asarray(
                    [
                        _as_float(previous_row, field)
                        for field in PROBABILITY_FIELDS
                    ],
                    dtype=np.float64,
                )
            probability_delta = probabilities - previous_probabilities

            fast_box = _box(row, "fast")
            previous_state = _box(previous_row, "sgla")
            previous_fast_box = _box(previous_row, "fast")
            fast_vs_state = _relative_box(fast_box, previous_state)
            fast_vs_previous_fast = _relative_box(
                fast_box, previous_fast_box
            )
            fast_log_aspect = math.log(
                max(float(fast_box[2]), 1e-6)
                / max(float(fast_box[3]), 1e-6)
            )

            feature = np.asarray(
                list(probabilities)
                + list(_probability_statistics(probabilities))
                + list(probability_delta)
                + [
                    float(np.abs(probability_delta).sum()),
                    float(np.abs(probability_delta).max()),
                ]
                + list(fast_vs_state)
                + list(fast_vs_previous_fast)
                + [fast_log_aspect],
                dtype=np.float64,
            )
            feature = np.nan_to_num(
                feature, nan=0.0, posinf=8.0, neginf=-8.0
            )
            feature = np.clip(feature, -8.0, 8.0)
            if len(feature) != len(FEATURE_NAMES):
                raise AssertionError("Feature-name and feature-value mismatch")

            fast_iou = float(row["iou_fast"])
            target = np.asarray(
                [
                    float(row["iou_layer{}".format(layer)]) - fast_iou
                    for layer in CANDIDATE_LAYERS
                ],
                dtype=np.float64,
            )
            features.append(feature)
            utilities.append(target)
            groups.append("{}\x1f{}".format(*sequence_key))
            row_indices.append(row_index)

    if not features:
        raise ValueError("No valid tracked frames were found")
    return (
        np.stack(features),
        np.stack(utilities),
        np.asarray(groups, dtype=object),
        np.asarray(row_indices, dtype=np.int64),
    )


def assign_group_folds(groups, num_folds, seed):
    counts = Counter(groups.tolist())
    if num_folds < 2:
        raise ValueError("--folds must be at least 2")
    if num_folds > len(counts):
        raise ValueError(
            "--folds={} exceeds the {} available sequences".format(
                num_folds, len(counts)
            )
        )

    items = list(counts.items())
    random.Random(seed).shuffle(items)
    items.sort(key=lambda item: item[1], reverse=True)
    fold_sizes = [0] * num_folds
    group_to_fold = {}
    for group, count in items:
        fold = min(range(num_folds), key=lambda index: fold_sizes[index])
        group_to_fold[group] = fold
        fold_sizes[fold] += count
    return np.asarray([group_to_fold[group] for group in groups], dtype=np.int64)


def _standardizer(features):
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def ridge_predict(
    train_features,
    train_targets,
    test_features,
    alpha,
):
    mean, scale = _standardizer(train_features)
    train = (train_features - mean) / scale
    test = (test_features - mean) / scale
    train = np.concatenate(
        [train, np.ones((len(train), 1), dtype=np.float64)], axis=1
    )
    test = np.concatenate(
        [test, np.ones((len(test), 1), dtype=np.float64)], axis=1
    )

    gram = train.T.dot(train) / len(train)
    cross = train.T.dot(train_targets) / len(train)
    regularizer = np.eye(gram.shape[0], dtype=np.float64) * alpha
    regularizer[-1, -1] = 0.0
    weights = np.linalg.solve(gram + regularizer, cross)
    return test.dot(weights), {
        "best_epoch": "",
        "validation_loss": "",
        "device": "cpu",
    }


def _validation_indices(train_indices, groups, fraction, seed):
    group_counts = Counter(groups[train_indices].tolist())
    group_items = list(group_counts.items())
    random.Random(seed).shuffle(group_items)
    target_frames = max(1, int(round(len(train_indices) * fraction)))
    validation_groups = set()
    validation_frames = 0
    for group, count in group_items:
        if len(validation_groups) >= len(group_items) - 1:
            break
        validation_groups.add(group)
        validation_frames += count
        if validation_frames >= target_frames:
            break
    validation_mask = np.asarray(
        [groups[index] in validation_groups for index in train_indices]
    )
    return train_indices[~validation_mask], train_indices[validation_mask]


def mlp_predict(
    features,
    targets,
    groups,
    train_indices,
    test_indices,
    args,
    fold,
):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:
        raise RuntimeError(
            "--model mlp requires PyTorch in the active environment"
        ) from error

    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)

    fit_indices, validation_indices = _validation_indices(
        train_indices,
        groups,
        args.validation_fraction,
        args.seed + fold,
    )
    mean, scale = _standardizer(features[fit_indices])

    def normalized_tensor(indices):
        values = ((features[indices] - mean) / scale).astype(np.float32)
        labels = targets[indices].astype(np.float32)
        return torch.from_numpy(values), torch.from_numpy(labels)

    fit_x, fit_y = normalized_tensor(fit_indices)
    validation_x, validation_y = normalized_tensor(validation_indices)
    test_x, _ = normalized_tensor(test_indices)

    torch.manual_seed(args.seed + fold)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + fold)

    hidden_dims = tuple(args.hidden_dims)

    class UtilityMLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            input_dim = features.shape[1]
            for hidden_dim in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(args.dropout),
                    ]
                )
                input_dim = hidden_dim
            layers.append(nn.Linear(input_dim, len(CANDIDATE_LAYERS)))
            self.layers = nn.Sequential(*layers)

        def forward(self, values):
            return self.layers(values)

    model = UtilityMLP().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    regression_loss = nn.SmoothL1Loss(beta=args.huber_beta)
    action_loss = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(args.seed + fold)
    loader = DataLoader(
        TensorDataset(fit_x, fit_y),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )

    def combined_loss(prediction, target):
        regression = regression_loss(prediction, target)
        zero = torch.zeros(
            (len(prediction), 1), device=prediction.device, dtype=prediction.dtype
        )
        target_zero = torch.zeros(
            (len(target), 1), device=target.device, dtype=target.dtype
        )
        action_target = torch.argmax(
            torch.cat([target_zero, target], dim=1), dim=1
        )
        logits = torch.cat([zero, prediction], dim=1) / args.utility_temperature
        return regression + args.action_loss_weight * action_loss(
            logits, action_target
        )

    best_state = None
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    validation_x = validation_x.to(device)
    validation_y = validation_y.to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = combined_loss(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = float(
                combined_loss(model(validation_x), validation_y).item()
            )
        if validation_loss < best_validation_loss - 1e-7:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("MLP training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(test_x), args.batch_size * 2):
            batch = test_x[start : start + args.batch_size * 2].to(device)
            predictions.append(model(batch).cpu().numpy())
    return np.concatenate(predictions, axis=0).astype(np.float64), {
        "best_epoch": best_epoch,
        "validation_loss": best_validation_loss,
        "device": str(device),
        "fit_frames": len(fit_indices),
        "validation_frames": len(validation_indices),
    }


def _binary_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = float(ranks[labels].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _pearson(first, second):
    if np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def _auc_from_deltas(rows, row_indices, fast_ious, deltas, field):
    for row in rows:
        row[field] = float(row["iou_fast"])
    for row_index, fast_iou, delta in zip(row_indices, fast_ious, deltas):
        rows[int(row_index)][field] = float(fast_iou + delta)
    return stage2.success_auc(rows, field) * 100.0


def _percentile(values, percentile):
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(values, percentile * 100.0))


def evaluate_oof(rows, row_indices, targets, predictions, budgets):
    fast_ious = np.asarray(
        [float(rows[index]["iou_fast"]) for index in row_indices],
        dtype=np.float64,
    )
    chosen_offsets = np.argmax(predictions, axis=1)
    predicted_utility = predictions[
        np.arange(len(predictions)), chosen_offsets
    ]
    chosen_actual_delta = targets[
        np.arange(len(targets)), chosen_offsets
    ]
    best_candidate_delta = targets.max(axis=1)
    best_action_delta = np.maximum(best_candidate_delta, 0.0)
    layer_hit = chosen_actual_delta >= best_candidate_delta - TOLERANCE

    fast_auc = stage2.success_auc(rows, "iou_fast") * 100.0
    sgla_auc = stage2.success_auc(rows, "iou_sgla") * 100.0
    best_candidate_auc = stage2.success_auc(
        rows, "iou_best_candidate"
    ) * 100.0
    best_action_auc = _auc_from_deltas(
        rows,
        row_indices,
        fast_ious,
        best_action_delta,
        "_stage3_best_action",
    )
    always_deep_auc = _auc_from_deltas(
        rows,
        row_indices,
        fast_ious,
        chosen_actual_delta,
        "_stage3_always_deep",
    )
    positive_mask = predicted_utility > 0.0
    positive_delta = np.where(positive_mask, chosen_actual_delta, 0.0)
    positive_gate_auc = _auc_from_deltas(
        rows,
        row_indices,
        fast_ious,
        positive_delta,
        "_stage3_positive_gate",
    )
    positive_regret = np.maximum(best_action_delta - positive_delta, 0.0)

    overall = {
        "num_frames": len(rows),
        "num_valid_decision_frames": len(row_indices),
        "num_sequences": len(
            {(row["dataset"], row["sequence"]) for row in rows}
        ),
        "num_features": len(FEATURE_NAMES),
        "fast_auc": fast_auc,
        "sgla_auc": sgla_auc,
        "best_candidate_auc": best_candidate_auc,
        "best_action_auc": best_action_auc,
        "best_action_gain_vs_fast": best_action_auc - fast_auc,
        "best_action_deep_ratio": float(np.mean(best_candidate_delta > 0.0)),
        "oof_layer_hit_rate": float(np.mean(layer_hit)),
        "oof_predicted_utility_pearson": _pearson(
            predicted_utility, best_candidate_delta
        ),
        "oof_benefit_auroc": _binary_auc(
            best_candidate_delta > 0.0, predicted_utility
        ),
        "oof_always_deep_auc": always_deep_auc,
        "oof_always_deep_gain_vs_fast": always_deep_auc - fast_auc,
        "oof_positive_gate_auc": positive_gate_auc,
        "oof_positive_gate_gain_vs_fast": positive_gate_auc - fast_auc,
        "oof_positive_gate_deep_ratio": float(np.mean(positive_mask)),
        "oof_positive_gate_action_hit_rate": float(
            np.mean(positive_regret <= TOLERANCE)
        ),
        "oof_positive_gate_regret_mean": float(positive_regret.mean()),
        "oof_positive_gate_regret_p90": _percentile(positive_regret, 0.90),
    }

    predicted_order = np.argsort(-predicted_utility, kind="mergesort")
    oracle_layer_order = np.argsort(
        -best_candidate_delta, kind="mergesort"
    )
    oracle_gate_predicted_layer_order = np.argsort(
        -chosen_actual_delta, kind="mergesort"
    )
    budget_summaries = []
    for budget in budgets:
        if budget == 0.0:
            num_deep = 0
        elif budget == 100.0:
            num_deep = len(row_indices)
        else:
            num_deep = int(math.ceil(len(row_indices) * budget / 100.0))

        selected = np.zeros(len(row_indices), dtype=np.bool_)
        selected[predicted_order[:num_deep]] = True
        router_delta = np.where(selected, chosen_actual_delta, 0.0)
        router_auc = _auc_from_deltas(
            rows,
            row_indices,
            fast_ious,
            router_delta,
            "_stage3_router_budget",
        )

        predicted_gate_oracle_layer_delta = np.where(
            selected, best_candidate_delta, 0.0
        )
        predicted_gate_oracle_layer_auc = _auc_from_deltas(
            rows,
            row_indices,
            fast_ious,
            predicted_gate_oracle_layer_delta,
            "_stage3_predicted_gate_oracle_layer",
        )

        oracle_gate_predicted_layer_selected = np.zeros(
            len(row_indices), dtype=np.bool_
        )
        oracle_gate_predicted_layer_selected[
            oracle_gate_predicted_layer_order[:num_deep]
        ] = True
        oracle_gate_predicted_layer_delta = np.where(
            oracle_gate_predicted_layer_selected,
            chosen_actual_delta,
            0.0,
        )
        oracle_gate_predicted_layer_auc = _auc_from_deltas(
            rows,
            row_indices,
            fast_ious,
            oracle_gate_predicted_layer_delta,
            "_stage3_oracle_gate_predicted_layer",
        )

        oracle_selected = np.zeros(len(row_indices), dtype=np.bool_)
        oracle_selected[oracle_layer_order[:num_deep]] = True
        oracle_delta = np.where(oracle_selected, best_candidate_delta, 0.0)
        oracle_budget_auc = _auc_from_deltas(
            rows,
            row_indices,
            fast_ious,
            oracle_delta,
            "_stage3_oracle_budget",
        )

        regret = np.maximum(best_action_delta - router_delta, 0.0)
        action_hit = regret <= TOLERANCE
        oracle_gain = oracle_budget_auc - fast_auc
        recovery = (
            (router_auc - fast_auc) / oracle_gain
            if oracle_gain > 1e-12
            else float("nan")
        )
        selected_deltas = chosen_actual_delta[selected]
        budget_summaries.append(
            {
                "budget_percent": budget,
                "num_deep": num_deep,
                "actual_deep_ratio": num_deep / len(row_indices),
                "candidate_blocks_per_frame": num_deep / len(row_indices),
                "router_auc": router_auc,
                "router_gain_vs_fast": router_auc - fast_auc,
                "router_gain_vs_sgla": router_auc - sgla_auc,
                "oracle_budget_auc": oracle_budget_auc,
                "oracle_budget_gain_vs_fast": oracle_gain,
                "oracle_recovery_ratio": recovery,
                "predicted_gate_oracle_layer_auc": (
                    predicted_gate_oracle_layer_auc
                ),
                "oracle_gate_predicted_layer_auc": (
                    oracle_gate_predicted_layer_auc
                ),
                "selected_layer_hit_rate": (
                    float(np.mean(layer_hit[selected]))
                    if num_deep
                    else float("nan")
                ),
                "selected_actual_delta_mean": (
                    float(selected_deltas.mean())
                    if num_deep
                    else float("nan")
                ),
                "selected_beneficial_ratio": (
                    float(np.mean(selected_deltas > 0.0))
                    if num_deep
                    else float("nan")
                ),
                "action_hit_rate": float(np.mean(action_hit)),
                "regret_mean": float(regret.mean()),
                "regret_p90": _percentile(regret, 0.90),
            }
        )
    return overall, budget_summaries, {
        "chosen_offsets": chosen_offsets,
        "predicted_utility": predicted_utility,
        "chosen_actual_delta": chosen_actual_delta,
        "best_candidate_delta": best_candidate_delta,
        "best_action_delta": best_action_delta,
        "layer_hit": layer_hit,
    }


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


def _prediction_rows(
    rows,
    row_indices,
    fold_ids,
    targets,
    predictions,
    details,
):
    output = []
    for index, row_index in enumerate(row_indices):
        row = rows[int(row_index)]
        chosen_offset = int(details["chosen_offsets"][index])
        chosen_layer = CANDIDATE_LAYERS[chosen_offset]
        best_delta = float(details["best_candidate_delta"][index])
        best_action = (
            "fast"
            if best_delta <= 0.0
            else "layer{}".format(
                CANDIDATE_LAYERS[int(np.argmax(targets[index]))]
            )
        )
        item = {
            "dataset": row["dataset"],
            "sequence": row["sequence"],
            "frame_id": row["frame_id"],
            "fold": int(fold_ids[index]),
            "iou_fast": row["iou_fast"],
            "iou_sgla": row["iou_sgla"],
            "iou_best_candidate": row["iou_best_candidate"],
            "iou_best_action": float(
                row["iou_fast"] + details["best_action_delta"][index]
            ),
            "best_action": best_action,
            "predicted_layer": chosen_layer,
            "predicted_utility": details["predicted_utility"][index],
            "predicted_positive_action": (
                "layer{}".format(chosen_layer)
                if details["predicted_utility"][index] > 0.0
                else "fast"
            ),
            "actual_delta_predicted_layer": details[
                "chosen_actual_delta"
            ][index],
            "actual_best_candidate_delta": best_delta,
            "layer_hit": int(details["layer_hit"][index]),
        }
        for offset, layer in enumerate(CANDIDATE_LAYERS):
            item["predicted_delta_layer{}".format(layer)] = predictions[
                index, offset
            ]
            item["actual_delta_layer{}".format(layer)] = targets[index, offset]
        output.append(item)
    return output


def _parse_budgets(values):
    budgets = sorted(set(float(value) for value in values))
    if any(value < 0.0 or value > 100.0 for value in budgets):
        raise ValueError("Every budget must be in [0, 100]")
    return budgets


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run sequence-level out-of-fold utility routing on Stage-2 CSVs."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output_prefix", default=None)
    parser.add_argument("--model", choices=("ridge", "mlp"), default="ridge")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--budgets",
        nargs="+",
        default=["0", "10", "20", "30", "50", "100"],
    )
    parser.add_argument("--ridge_alpha", type=float, default=0.001)

    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[64, 32])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--huber_beta", type=float, default=0.02)
    parser.add_argument("--action_loss_weight", type=float, default=0.01)
    parser.add_argument("--utility_temperature", type=float, default=0.05)
    return parser


def main():
    args = _build_parser().parse_args()
    if args.ridge_alpha < 0.0:
        raise ValueError("--ridge_alpha must be non-negative")
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation_fraction must be in (0, 0.5)")
    if any(value <= 0 for value in args.hidden_dims):
        raise ValueError("Every --hidden_dims value must be positive")
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("--epochs, --patience, and --batch_size must be positive")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError(
            "--learning_rate must be positive and --weight_decay non-negative"
        )
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if args.huber_beta <= 0.0 or args.utility_temperature <= 0.0:
        raise ValueError(
            "--huber_beta and --utility_temperature must be positive"
        )
    if args.action_loss_weight < 0.0:
        raise ValueError("--action_loss_weight must be non-negative")
    budgets = _parse_budgets(args.budgets)

    input_path, fieldnames, rows, csv_files = stage2.load_rows(args.input)
    features, targets, groups, row_indices = extract_examples(rows, fieldnames)
    fold_ids = assign_group_folds(groups, args.folds, args.seed)
    predictions = np.full(targets.shape, np.nan, dtype=np.float64)
    fold_summaries = []

    print(
        "Loaded {} frames, {} valid decisions, {} sequences from {} CSV file(s).".format(
            len(rows),
            len(row_indices),
            len(set(groups.tolist())),
            len(csv_files),
        )
    )
    print(
        "Features: {} causal pre-candidate values; model={}; folds={}".format(
            len(FEATURE_NAMES), args.model, args.folds
        )
    )

    for fold in range(args.folds):
        test_indices = np.flatnonzero(fold_ids == fold)
        train_indices = np.flatnonzero(fold_ids != fold)
        test_groups = set(groups[test_indices].tolist())
        train_groups = set(groups[train_indices].tolist())
        if test_groups.intersection(train_groups):
            raise AssertionError("Sequence leakage detected between folds")

        print(
            "[fold {}/{}] train={} frames/{} sequences, test={} frames/{} sequences".format(
                fold + 1,
                args.folds,
                len(train_indices),
                len(train_groups),
                len(test_indices),
                len(test_groups),
            )
        )
        if args.model == "ridge":
            fold_prediction, metadata = ridge_predict(
                features[train_indices],
                targets[train_indices],
                features[test_indices],
                args.ridge_alpha,
            )
        else:
            fold_prediction, metadata = mlp_predict(
                features,
                targets,
                groups,
                train_indices,
                test_indices,
                args,
                fold,
            )
        predictions[test_indices] = fold_prediction
        fold_summaries.append(
            {
                "fold": fold,
                "train_sequences": len(train_groups),
                "test_sequences": len(test_groups),
                "train_frames": len(train_indices),
                "test_frames": len(test_indices),
                "fit_frames": metadata.get("fit_frames", len(train_indices)),
                "validation_frames": metadata.get("validation_frames", 0),
                "best_epoch": metadata.get("best_epoch", ""),
                "validation_loss": metadata.get("validation_loss", ""),
                "device": metadata.get("device", ""),
            }
        )

    if not np.isfinite(predictions).all():
        raise RuntimeError("OOF predictions contain missing or non-finite values")

    overall, budget_summaries, details = evaluate_oof(
        rows, row_indices, targets, predictions, budgets
    )
    overall["model"] = args.model
    overall["folds"] = args.folds
    overall["seed"] = args.seed
    overall["ridge_alpha"] = args.ridge_alpha if args.model == "ridge" else ""
    overall["feature_names"] = "|".join(FEATURE_NAMES)

    prediction_rows = _prediction_rows(
        rows,
        row_indices,
        fold_ids,
        targets,
        predictions,
        details,
    )

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (input_path if input_path.is_dir() else input_path.parent)
    )
    dataset_name = rows[0]["dataset"].replace("/", "_").replace("\\", "_")
    prefix = args.output_prefix or "{}_stage3_{}".format(
        dataset_name, args.model
    )
    overall_path = output_dir / "{}_overall.csv".format(prefix)
    budget_path = output_dir / "{}_budget.csv".format(prefix)
    folds_path = output_dir / "{}_folds.csv".format(prefix)
    predictions_path = output_dir / "{}_oof_predictions.csv".format(prefix)

    _write_csv(overall_path, list(overall.keys()), [overall])
    _write_csv(
        budget_path, list(budget_summaries[0].keys()), budget_summaries
    )
    _write_csv(folds_path, list(fold_summaries[0].keys()), fold_summaries)
    _write_csv(
        predictions_path,
        list(prediction_rows[0].keys()),
        prediction_rows,
    )

    print("metric                              auc/gain")
    print("Fast AUC                         {:>10.4f}".format(overall["fast_auc"]))
    print("SGLA AUC                         {:>10.4f}".format(overall["sgla_auc"]))
    print(
        "Best candidate AUC               {:>10.4f}".format(
            overall["best_candidate_auc"]
        )
    )
    print(
        "Best action (Fast+L7-L12) AUC    {:>10.4f}".format(
            overall["best_action_auc"]
        )
    )
    print(
        "OOF always-deep AUC              {:>10.4f}".format(
            overall["oof_always_deep_auc"]
        )
    )
    print(
        "OOF predicted-positive AUC       {:>10.4f}  deep={:.2%}".format(
            overall["oof_positive_gate_auc"],
            overall["oof_positive_gate_deep_ratio"],
        )
    )
    print(
        "OOF benefit AUROC                {:>10.4f}".format(
            overall["oof_benefit_auroc"]
        )
    )
    print(
        "OOF candidate-layer hit          {:>10.2%}".format(
            overall["oof_layer_hit_rate"]
        )
    )
    print("budget  deep_frames  router_auc  gain_fast  gate+oracle  oracle+layer")
    for item in budget_summaries:
        print(
            "{:>6.1f}%  {:>11d}  {:>10.4f}  {:>9.4f}  {:>11.4f}  {:>12.4f}".format(
                item["budget_percent"],
                item["num_deep"],
                item["router_auc"],
                item["router_gain_vs_fast"],
                item["predicted_gate_oracle_layer_auc"],
                item["oracle_gate_predicted_layer_auc"],
            )
        )
    print("Overall CSV: {}".format(overall_path))
    print("Budget CSV: {}".format(budget_path))
    print("Fold CSV: {}".format(folds_path))
    print("OOF prediction CSV: {}".format(predictions_path))


if __name__ == "__main__":
    main()
