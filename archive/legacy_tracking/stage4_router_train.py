"""Train and validate Stage-4 utility routers from audited frozen caches.

The script is deliberately NumPy-only.  It fits all preprocessing and model
parameters on manifest training videos, uses a video-disjoint calibration
subset for hyperparameters and budget thresholds, and evaluates the untouched
manifest validation videos once.  Reported IoUs are current-state diagnostics,
not closed-loop tracking benchmark AUCs.
"""

import argparse
import csv
import json
import math
import os
import random
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from stage4_check_cache import check_directory
from stage4_common import ACTIONS, cache_path, file_hash, fingerprint, write_csv, write_json


CANDIDATE_LAYERS = np.arange(7, 13, dtype=np.int64)
MODEL_SCHEMA = "stage4_utility_ridge_v1"
VARIANTS = ("token_only", "token_selector", "fast_aware")
TOLERANCE = 1e-12


def _weighted_mean(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(values) == 0 or float(weights.sum()) <= 0:
        return float("nan")
    return float(np.sum(values * weights) / np.sum(weights))


def _weighted_mse(targets, predictions, weights):
    per_row = np.mean((np.asarray(targets) - np.asarray(predictions)) ** 2, axis=1)
    return _weighted_mean(per_row, weights)


def _weighted_pearson(first, second, weights):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0:
        return float("nan")
    first_mean = float(np.sum(weights * first) / total)
    second_mean = float(np.sum(weights * second) / total)
    first_centered = first - first_mean
    second_centered = second - second_mean
    first_var = float(np.sum(weights * first_centered ** 2) / total)
    second_var = float(np.sum(weights * second_centered ** 2) / total)
    if first_var < 1e-16 or second_var < 1e-16:
        return float("nan")
    covariance = float(np.sum(weights * first_centered * second_centered) / total)
    return covariance / math.sqrt(first_var * second_var)


def _weighted_binary_auc(labels, scores, weights):
    """Weighted Mann-Whitney AUROC with half credit for score ties."""
    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    positive_total = float(weights[labels].sum())
    negative_total = float(weights[~labels].sum())
    if positive_total <= 0 or negative_total <= 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    scores = scores[order]
    labels = labels[order]
    weights = weights[order]
    concordant = 0.0
    negative_before = 0.0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[end] == scores[start]:
            end += 1
        tied_positive = float(weights[start:end][labels[start:end]].sum())
        tied_negative = float(weights[start:end][~labels[start:end]].sum())
        concordant += tied_positive * (negative_before + 0.5 * tied_negative)
        negative_before += tied_negative
        start = end
    return concordant / (positive_total * negative_total)


def balanced_group_weights(sequences, trajectories):
    groups = ["{}\x1f{}".format(sequence, trajectory)
              for sequence, trajectory in zip(sequences, trajectories)]
    counts = Counter(groups)
    weights = np.asarray([1.0 / counts[group] for group in groups], dtype=np.float64)
    weights *= len(weights) / weights.sum()
    return weights


def calibration_split(sequences, fraction, seed):
    names = sorted(set(str(value) for value in sequences))
    if len(names) < 2:
        raise ValueError("At least two training videos are required")
    shuffled = names[:]
    random.Random(seed).shuffle(shuffled)
    count = min(len(names) - 1, max(1, int(round(len(names) * fraction))))
    calibration = set(shuffled[:count])
    calibration_mask = np.asarray([str(value) in calibration for value in sequences])
    return ~calibration_mask, calibration_mask, sorted(calibration)


def load_cache_dataset(directory):
    """Audit and load only the declared causal inputs plus labels."""
    directory = Path(directory).expanduser().resolve()
    check_directory(directory)
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    required_features = {
        "token_features", "fast_score_map", "selector_prob", "previous_state",
        "previous_fast_box", "previous_action", "fast_box", "image_hw",
        "resize_factor",
    }
    if not required_features.issubset(set(run["feature_fields"])):
        raise ValueError("Cache does not contain all Stage-4C causal features")
    if not {"valid_gt", "iou", "delta"}.issubset(set(run["label_fields"])):
        raise ValueError("Cache does not contain the required utility labels")

    fields = sorted(required_features | {
        "valid_gt", "iou", "delta", "frame_id", "selected_layer",
    })
    chunks = {field: [] for field in fields}
    metadata = {"split": [], "sequence": [], "trajectory": []}
    for entry in run["manifest"]["sequences"]:
        for trajectory in run["trajectories"]:
            path = cache_path(directory, entry, trajectory)
            with np.load(str(path), allow_pickle=False) as archive:
                valid = archive["valid_gt"].astype(np.bool_, copy=False)
                for field in fields:
                    if field == "valid_gt":
                        chunks[field].append(archive[field][valid].copy())
                    else:
                        chunks[field].append(archive[field][valid].copy())
                count = int(valid.sum())
            metadata["split"].extend([entry["split"]] * count)
            metadata["sequence"].extend([entry["sequence"]] * count)
            metadata["trajectory"].extend([trajectory] * count)

    data = {field: np.concatenate(values, axis=0) for field, values in chunks.items()}
    for field, values in metadata.items():
        data[field] = np.asarray(values, dtype=str)
    if not np.isfinite(data["token_features"]).all() or not np.isfinite(data["delta"]).all():
        raise ValueError("Valid examples must contain finite features and labels")
    return data, run


def subset_data(data, mask):
    mask = np.asarray(mask, dtype=np.bool_)
    return {key: values[mask] for key, values in data.items()}


def _relative_boxes(current, reference):
    current = np.asarray(current, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    epsilon = 1e-6
    current_size = np.maximum(current[:, 2:], epsilon)
    reference_size = np.maximum(reference[:, 2:], epsilon)
    current_center = current[:, :2] + 0.5 * current_size
    reference_center = reference[:, :2] + 0.5 * reference_size
    return np.concatenate(
        ((current_center - reference_center) / reference_size,
         np.log(current_size / reference_size)), axis=1)


def _normalized_boxes(boxes, image_hw):
    boxes = np.asarray(boxes, dtype=np.float64)
    image_hw = np.asarray(image_hw, dtype=np.float64)
    scale = np.column_stack((image_hw[:, 1], image_hw[:, 0],
                             image_hw[:, 1], image_hw[:, 0]))
    return boxes / np.maximum(scale, 1.0)


def _selector_statistics(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    ordered = np.sort(probabilities, axis=1)[:, ::-1]
    normalized = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    entropy = -np.sum(normalized * np.log(np.maximum(normalized, 1e-12)), axis=1)
    entropy /= math.log(probabilities.shape[1])
    return np.column_stack((probabilities.mean(axis=1), probabilities.std(axis=1),
                            ordered[:, 0], ordered[:, 0] - ordered[:, 1], entropy))


def _score_statistics(score_maps):
    score_maps = np.asarray(score_maps, dtype=np.float64)
    flat = score_maps.reshape(len(score_maps), -1)
    side_y, side_x = score_maps.shape[-2:]
    ordered = np.sort(flat, axis=1)[:, ::-1]
    shifted = flat - flat.max(axis=1, keepdims=True)
    spatial = np.exp(np.clip(shifted, -60.0, 0.0))
    spatial /= np.maximum(spatial.sum(axis=1, keepdims=True), 1e-12)
    entropy = -np.sum(spatial * np.log(np.maximum(spatial, 1e-12)), axis=1)
    entropy /= math.log(flat.shape[1])
    peak = np.argmax(flat, axis=1)
    peak_y, peak_x = np.divmod(peak, side_x)
    ys, xs = np.meshgrid(np.linspace(0.0, 1.0, side_y),
                         np.linspace(0.0, 1.0, side_x), indexing="ij")
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)
    center_x = spatial.dot(xs)
    center_y = spatial.dot(ys)
    spread_x = np.sqrt(np.maximum(spatial.dot(xs ** 2) - center_x ** 2, 0.0))
    spread_y = np.sqrt(np.maximum(spatial.dot(ys ** 2) - center_y ** 2, 0.0))
    return np.column_stack((
        flat.mean(axis=1), flat.std(axis=1), flat.max(axis=1), flat.min(axis=1),
        ordered[:, 0] - ordered[:, 1], entropy,
        peak_x / max(side_x - 1, 1), peak_y / max(side_y - 1, 1),
        center_x, center_y, spread_x, spread_y,
    ))


def build_low_features(data, variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown feature variant: {}".format(variant))
    previous_state = np.asarray(data["previous_state"], dtype=np.float64)
    previous_fast = np.asarray(data["previous_fast_box"], dtype=np.float64)
    image_hw = np.asarray(data["image_hw"], dtype=np.float64)
    actions = np.asarray(data["previous_action"], dtype=np.int64)
    if np.any((actions < -1) | (actions > 6)):
        raise ValueError("previous_action must be in {-1, 0, ..., 6}")
    action_one_hot = np.eye(8, dtype=np.float64)[actions + 1]
    base = [
        _normalized_boxes(previous_state, image_hw),
        _normalized_boxes(previous_fast, image_hw),
        _relative_boxes(previous_fast, previous_state),
        action_one_hot,
        np.log(np.maximum(np.asarray(data["resize_factor"], dtype=np.float64), 1e-8))[:, None],
        np.log(np.maximum(image_hw[:, 1], 1.0) / np.maximum(image_hw[:, 0], 1.0))[:, None],
        np.log(np.maximum(previous_state[:, 2], 1e-6)
               / np.maximum(previous_state[:, 3], 1e-6))[:, None],
    ]
    names = (
        ["previous_state_{}".format(value) for value in ("x", "y", "w", "h")]
        + ["previous_fast_{}".format(value) for value in ("x", "y", "w", "h")]
        + ["previous_fast_vs_state_{}".format(value)
           for value in ("dx", "dy", "log_w", "log_h")]
        + ["previous_action_{}".format(value) for value in ("init", "fast", "l7", "l8", "l9", "l10", "l11", "l12")]
        + ["log_resize_factor", "image_log_aspect", "previous_state_log_aspect"]
    )
    if variant == "token_selector":
        selector = np.asarray(data["selector_prob"], dtype=np.float64)
        base.extend((selector, _selector_statistics(selector)))
        names += (["selector_l{}".format(layer) for layer in CANDIDATE_LAYERS]
                  + ["selector_mean", "selector_std", "selector_max",
                     "selector_margin", "selector_entropy"])
    elif variant == "fast_aware":
        fast = np.asarray(data["fast_box"], dtype=np.float64)
        base.extend((
            _score_statistics(data["fast_score_map"]),
            _normalized_boxes(fast, image_hw),
            _relative_boxes(fast, previous_state),
            _relative_boxes(fast, previous_fast),
            np.log(np.maximum(fast[:, 2], 1e-6) / np.maximum(fast[:, 3], 1e-6))[:, None],
        ))
        names += (
            ["fast_score_{}".format(value) for value in
             ("mean", "std", "max", "min", "margin", "entropy", "peak_x", "peak_y",
              "center_x", "center_y", "spread_x", "spread_y")]
            + ["fast_box_{}".format(value) for value in ("x", "y", "w", "h")]
            + ["fast_vs_state_{}".format(value) for value in ("dx", "dy", "log_w", "log_h")]
            + ["fast_vs_previous_fast_{}".format(value)
               for value in ("dx", "dy", "log_w", "log_h")]
            + ["fast_log_aspect"]
        )
    features = np.concatenate(base, axis=1)
    features = np.nan_to_num(features, nan=0.0, posinf=20.0, neginf=-20.0)
    features = np.clip(features, -20.0, 20.0).astype(np.float32)
    if features.shape[1] != len(names):
        raise AssertionError("Low-dimensional feature-name mismatch")
    return features, tuple(names)


def _standardizer(values, weights=None):
    values = np.asarray(values)
    if weights is None:
        mean = values.mean(axis=0, dtype=np.float64)
        scale = values.std(axis=0, dtype=np.float64)
    else:
        values = values.astype(np.float64, copy=False)
        weights = np.asarray(weights, dtype=np.float64)
        normalized = weights / weights.sum()
        mean = np.sum(values * normalized[:, None], axis=0)
        scale = np.sqrt(np.sum((values - mean) ** 2 * normalized[:, None], axis=0))
    scale[scale < 1e-8] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def fit_token_pca(tokens, dimensions, oversample, power_iterations, seed):
    tokens = np.asarray(tokens)
    if tokens.ndim != 2 or len(tokens) < 2:
        raise ValueError("Token features must be a two-dimensional non-empty matrix")
    dimensions = min(int(dimensions), len(tokens) - 1, tokens.shape[1])
    if dimensions < 1:
        raise ValueError("--pca_dims must leave at least one component")
    mean, scale = _standardizer(tokens)
    standardized = ((tokens.astype(np.float32) - mean) / scale).astype(np.float32)
    sketch_size = min(dimensions + int(oversample), min(standardized.shape))
    generator = np.random.RandomState(seed)
    omega = generator.normal(0.0, 1.0 / math.sqrt(sketch_size),
                             size=(standardized.shape[1], sketch_size)).astype(np.float32)
    basis, _ = np.linalg.qr(standardized.dot(omega), mode="reduced")
    for _ in range(int(power_iterations)):
        feature_basis, _ = np.linalg.qr(standardized.T.dot(basis), mode="reduced")
        basis, _ = np.linalg.qr(standardized.dot(feature_basis), mode="reduced")
    reduced = basis.T.dot(standardized)
    _, singular_values, right = np.linalg.svd(reduced, full_matrices=False)
    components = right[:dimensions].astype(np.float32)
    for row in range(len(components)):
        pivot = int(np.argmax(np.abs(components[row])))
        if components[row, pivot] < 0:
            components[row] *= -1
    total_variance = float(np.var(standardized, axis=0, ddof=1).sum())
    captured = float(np.sum(singular_values[:dimensions] ** 2) / max(len(tokens) - 1, 1))
    explained_ratio = captured / max(total_variance, 1e-12)
    return {"mean": mean, "scale": scale, "components": components,
            "explained_variance_ratio": explained_ratio}


def transform_token_pca(tokens, pca, batch_size=1024):
    tokens = np.asarray(tokens)
    output = np.empty((len(tokens), len(pca["components"])), dtype=np.float32)
    for start in range(0, len(tokens), batch_size):
        stop = min(start + batch_size, len(tokens))
        standardized = ((tokens[start:stop].astype(np.float32) - pca["mean"])
                        / pca["scale"])
        output[start:stop] = standardized.dot(pca["components"].T)
    return output


def fit_combined_preprocessor(token_projection, low_features, weights):
    combined = np.concatenate((token_projection, low_features), axis=1)
    mean, scale = _standardizer(combined, weights)
    return {"mean": mean, "scale": scale}


def transform_combined(token_projection, low_features, preprocessor):
    combined = np.concatenate((token_projection, low_features), axis=1)
    return ((combined - preprocessor["mean"]) / preprocessor["scale"]).astype(np.float64)


def fit_ridge(features, targets, weights, alpha):
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    design = np.concatenate((features, np.ones((len(features), 1))), axis=1)
    root = np.sqrt(weights / weights.sum())[:, None]
    weighted_design = design * root
    weighted_targets = targets * root
    gram = weighted_design.T.dot(weighted_design)
    cross = weighted_design.T.dot(weighted_targets)
    regularizer = np.eye(gram.shape[0], dtype=np.float64) * float(alpha)
    regularizer[-1, -1] = 0.0
    parameters = np.linalg.solve(gram + regularizer, cross)
    return {"coefficient": parameters[:-1], "intercept": parameters[-1]}


def predict_ridge(features, model):
    return np.asarray(features, dtype=np.float64).dot(model["coefficient"]) + model["intercept"]


def calibration_threshold(scores, weights, maximum_budget_percent):
    """Return a score threshold whose calibration cost is at most the budget."""
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    budget = float(maximum_budget_percent) / 100.0
    if budget <= 0:
        return float("inf")
    if budget >= 1:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    cumulative = np.cumsum(weights[order]) / weights.sum()
    allowed = np.flatnonzero((cumulative <= budget + 1e-12) & (scores[order] > 0.0))
    if len(allowed) == 0:
        return float("inf")
    last = int(allowed[-1])
    if last + 1 == len(order):
        return 0.0
    upper = float(scores[order[last]])
    lower = max(0.0, float(scores[order[last + 1]]))
    return 0.5 * (upper + lower) if upper > lower else upper


def route_predictions(predictions, threshold, layer_costs=None, cost_lambda=0.0):
    predictions = np.asarray(predictions, dtype=np.float64)
    adjusted = predictions.copy()
    if cost_lambda:
        if layer_costs is None:
            raise ValueError("--cost_lambda requires --profile_dirs")
        adjusted -= float(cost_lambda) * np.asarray(layer_costs, dtype=np.float64)[None]
    offsets = np.argmax(adjusted, axis=1)
    scores = adjusted[np.arange(len(adjusted)), offsets]
    # Strict comparison keeps tied scores at a budget boundary outside the
    # deep set, so a fixed threshold never exceeds its calibration cap.
    deep = (scores > 0.0) & (scores > threshold)
    actions = np.where(deep, offsets + 1, 0).astype(np.int64)
    return actions, offsets, scores


def load_profile_costs(profile_paths):
    if not profile_paths:
        return None
    grouped = {}
    for raw_path in profile_paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_dir():
            path = path / "benchmark_summary.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                operation = row["operation"]
                samples = int(row["samples"])
                grouped.setdefault(operation, [0.0, 0])
                grouped[operation][0] += float(row["mean_ms"]) * samples
                grouped[operation][1] += samples
    means = {key: total / count for key, (total, count) in grouped.items()}
    required = {"fast", "sgla", "fast_then_sgla", "selector",
                "pool_all_channels_grid4"}
    required.update("block_L{}".format(layer) for layer in CANDIDATE_LAYERS)
    missing = required.difference(means)
    if missing:
        raise ValueError("Benchmark summaries are missing: {}".format(", ".join(sorted(missing))))
    return means


def estimated_latency(variant, actions, costs):
    if costs is None:
        return float("nan")
    actions = np.asarray(actions, dtype=np.int64)
    pool = costs["pool_all_channels_grid4"]
    values = np.full(len(actions), costs["fast"] + pool, dtype=np.float64)
    if variant == "token_selector":
        values += costs["selector"]
    deep = actions > 0
    if np.any(deep):
        layers = actions[deep] + 6
        blocks = np.asarray([costs["block_L{}".format(layer)] for layer in layers])
        if variant == "fast_aware":
            repeated_head = (costs["fast_then_sgla"] - costs["fast"]
                             - costs["selector"] - costs["block_L9"])
            values[deep] += np.maximum(repeated_head, 0.0) + blocks
        else:
            values[deep] += blocks
    return float(values.mean())


def additional_layer_costs(variant, costs):
    if costs is None:
        return None
    blocks = np.asarray(
        [costs["block_L{}".format(layer)] for layer in CANDIDATE_LAYERS],
        dtype=np.float64,
    )
    if variant == "fast_aware":
        repeated_head = (costs["fast_then_sgla"] - costs["fast"]
                         - costs["selector"] - costs["block_L9"])
        blocks += max(repeated_head, 0.0)
    return blocks


def _finite_or_blank(value):
    return value if np.isfinite(value) else ""


def _atomic_savez(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=str(path.parent),
                                     suffix=".tmp", delete=False) as handle:
        temporary = handle.name
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, str(path))


def save_model(path, variant, run, pca, combined, ridge, low_names, configuration):
    metadata = {
        "schema": MODEL_SCHEMA,
        "variant": variant,
        "cache_fingerprint": run["fingerprint"],
        "manifest_fingerprint": run["manifest"]["fingerprint"],
        "actions": list(ACTIONS),
        "configuration": configuration,
    }
    metadata["fingerprint"] = fingerprint(metadata)
    _atomic_savez(
        path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
        token_mean=pca["mean"], token_scale=pca["scale"],
        pca_components=pca["components"], combined_mean=combined["mean"],
        combined_scale=combined["scale"], ridge_coefficient=ridge["coefficient"],
        ridge_intercept=ridge["intercept"],
        low_feature_names=np.asarray(low_names, dtype="U"),
        candidate_layers=CANDIDATE_LAYERS,
    )
    return metadata


def evaluate_variant(variant, data, predictions, weights, calibration_predictions,
                     calibration_weights, budgets, costs, cost_lambda):
    layer_costs = additional_layer_costs(variant, costs)
    _, predicted_offsets, predicted_scores = route_predictions(
        predictions, 0.0, layer_costs, cost_lambda)
    targets = np.asarray(data["delta"], dtype=np.float64)
    iou = np.asarray(data["iou"], dtype=np.float64)
    chosen_delta = targets[np.arange(len(targets)), predicted_offsets]
    best_candidate_delta = targets.max(axis=1)
    best_action_iou = iou.max(axis=1)
    fast_iou = iou[:, 0]
    sgla_actions = np.asarray(data["selected_layer"], dtype=np.int64) - 6
    sgla_iou = iou[np.arange(len(iou)), sgla_actions]
    layer_hit = chosen_delta >= best_candidate_delta - TOLERANCE

    _, calibration_offsets, calibration_scores = route_predictions(
        calibration_predictions, 0.0, layer_costs, cost_lambda)
    summary = {
        "variant": variant,
        "val_examples": len(targets),
        "low_feature_dims": build_low_features(data, variant)[0].shape[1],
        "weighted_delta_mse": _weighted_mse(targets, predictions, weights),
        "weighted_delta_mae": _weighted_mean(np.mean(np.abs(targets - predictions), axis=1), weights),
        "benefit_auroc": _weighted_binary_auc(best_candidate_delta > 0.0,
                                                predicted_scores, weights),
        "predicted_utility_pearson": _weighted_pearson(predicted_scores,
                                                        best_candidate_delta, weights),
        "candidate_layer_hit_rate": _weighted_mean(layer_hit, weights),
        "fast_current_state_mean_iou": _weighted_mean(fast_iou, weights),
        "sgla_selector_current_state_mean_iou": _weighted_mean(sgla_iou, weights),
        "best_action_current_state_mean_iou": _weighted_mean(best_action_iou, weights),
        "sample_fast_current_state_mean_iou": float(fast_iou.mean()),
        "sample_sgla_selector_current_state_mean_iou": float(sgla_iou.mean()),
        "sample_best_action_current_state_mean_iou": float(best_action_iou.mean()),
    }

    budget_rows = []
    best_offsets = np.argmax(targets, axis=1)
    positive_best = np.maximum(best_candidate_delta, 0.0)
    # This is the upper bound for the reported group-balanced IoU under a
    # physical frame-count budget.
    oracle_order = np.argsort(-(weights * positive_best), kind="mergesort")
    calibration_cost_weights = np.ones(len(calibration_scores), dtype=np.float64)
    for budget in budgets:
        threshold = calibration_threshold(calibration_scores, calibration_cost_weights, budget)
        calibration_actions, _, _ = route_predictions(
            calibration_predictions, threshold, layer_costs, cost_lambda)
        actions, offsets, scores = route_predictions(predictions, threshold, layer_costs, cost_lambda)
        selected_iou = iou[np.arange(len(iou)), actions]
        selected_delta = selected_iou - fast_iou
        regret = best_action_iou - selected_iou
        deep = actions > 0

        oracle_count = int(math.floor(len(targets) * float(budget) / 100.0 + 1e-12))
        oracle_deep = np.zeros(len(targets), dtype=np.bool_)
        positive_order = oracle_order[positive_best[oracle_order] > 0.0]
        oracle_deep[positive_order[:oracle_count]] = True
        oracle_actions = np.where(oracle_deep, best_offsets + 1, 0)
        oracle_iou = iou[np.arange(len(iou)), oracle_actions]
        latency = estimated_latency(variant, actions, costs)
        selected_weights = weights[deep]
        selected_actual = selected_delta[deep]
        budget_rows.append({
            "variant": variant,
            "max_budget_percent": float(budget),
            "calibrated_score_threshold": _finite_or_blank(threshold),
            "calibration_deep_ratio": float(np.mean(calibration_actions > 0)),
            "val_deep_ratio": float(np.mean(deep)),
            "val_deep_frames": int(deep.sum()),
            "current_state_mean_iou": _weighted_mean(selected_iou, weights),
            "gain_vs_fast": _weighted_mean(selected_delta, weights),
            "gain_vs_sgla_selector": _weighted_mean(selected_iou - sgla_iou, weights),
            "oracle_current_state_mean_iou": _weighted_mean(oracle_iou, weights),
            "oracle_gain_vs_fast": _weighted_mean(oracle_iou - fast_iou, weights),
            "selected_actual_delta_mean": (_weighted_mean(selected_actual, selected_weights)
                                           if np.any(deep) else ""),
            "selected_beneficial_ratio": (_weighted_mean(selected_actual > 0.0, selected_weights)
                                           if np.any(deep) else ""),
            "action_hit_rate": _weighted_mean(regret <= TOLERANCE, weights),
            "regret_mean": _weighted_mean(regret, weights),
            "estimated_latency_ms_excluding_router": _finite_or_blank(latency),
            "estimated_fps_excluding_router": (_finite_or_blank(1000.0 / latency)
                                                if np.isfinite(latency) and latency > 0 else ""),
            "predicted_score_mean": _weighted_mean(scores, weights),
        })
    return summary, budget_rows, predicted_offsets, predicted_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Complete Stage-4B cache directory")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--pca_dims", type=int, default=64)
    parser.add_argument("--pca_oversample", type=int, default=12)
    parser.add_argument("--pca_power_iterations", type=int, default=1)
    parser.add_argument("--ridge_alphas", nargs="+", type=float,
                        default=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--calibration_fraction", type=float, default=0.2)
    parser.add_argument("--budgets", nargs="+", type=float,
                        default=[0, 10, 20, 30, 50, 100])
    parser.add_argument("--profile_dirs", nargs="*", default=[])
    parser.add_argument("--cost_lambda", type=float, default=0.0,
                        help="Predicted IoU utility penalty per additional millisecond")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if (args.pca_dims < 1 or args.pca_oversample < 0 or args.pca_power_iterations < 0
            or not 0.0 < args.calibration_fraction < 1.0 or args.batch_size < 1):
        raise ValueError("Invalid PCA, calibration or batch-size argument")
    if any(alpha <= 0 for alpha in args.ridge_alphas):
        raise ValueError("All ridge alphas must be positive")
    if any(budget < 0 or budget > 100 for budget in args.budgets):
        raise ValueError("Budgets must lie in [0, 100]")
    if len(set(args.variants)) != len(args.variants):
        raise ValueError("Feature variants must be unique")
    if args.cost_lambda < 0:
        raise ValueError("--cost_lambda must be non-negative")

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    try:
        output_dir.relative_to(input_dir)
    except ValueError:
        pass
    else:
        raise ValueError("--output_dir must be outside the audited cache directory")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError("Use a new --output_dir or pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    costs = load_profile_costs(args.profile_dirs)
    if args.cost_lambda and costs is None:
        raise ValueError("--cost_lambda requires benchmark summaries")
    data, run = load_cache_dataset(input_dir)
    if costs is not None and int(run["pool_grid"]) != 4:
        raise ValueError("The supplied profile measures grid4 pooling but the cache uses grid{}"
                         .format(run["pool_grid"]))
    train_mask = data["split"] == "train"
    val_mask = data["split"] == "val"
    train = subset_data(data, train_mask)
    val = subset_data(data, val_mask)
    if len(train["delta"]) == 0 or len(val["delta"]) == 0:
        raise ValueError("Both train and val need valid examples")
    fit_mask, calibration_mask, calibration_sequences = calibration_split(
        train["sequence"], args.calibration_fraction, args.seed)
    fit = subset_data(train, fit_mask)
    calibration = subset_data(train, calibration_mask)
    fit_weights = balanced_group_weights(fit["sequence"], fit["trajectory"])
    calibration_weights = balanced_group_weights(calibration["sequence"], calibration["trajectory"])
    train_weights = balanced_group_weights(train["sequence"], train["trajectory"])
    val_weights = balanced_group_weights(val["sequence"], val["trajectory"])

    print("Loaded {} train and {} val valid examples from {} videos.".format(
        len(train["delta"]), len(val["delta"]),
        len(set(data["sequence"].tolist()))))
    print("Fit/calibration: {}/{} examples; calibration videos: {}".format(
        len(fit["delta"]), len(calibration["delta"]), len(calibration_sequences)))
    print("Fitting calibration PCA (labels remain outside preprocessing)...", flush=True)
    calibration_pca = fit_token_pca(
        fit["token_features"], args.pca_dims, args.pca_oversample,
        args.pca_power_iterations, args.seed)
    fit_token_projection = transform_token_pca(
        fit["token_features"], calibration_pca, args.batch_size)
    calibration_token_projection = transform_token_pca(
        calibration["token_features"], calibration_pca, args.batch_size)

    selected_alphas = {}
    calibration_predictions = {}
    alpha_rows = []
    for variant in args.variants:
        fit_low, _ = build_low_features(fit, variant)
        calibration_low, _ = build_low_features(calibration, variant)
        combined = fit_combined_preprocessor(fit_token_projection, fit_low, fit_weights)
        fit_features = transform_combined(fit_token_projection, fit_low, combined)
        calibration_features = transform_combined(
            calibration_token_projection, calibration_low, combined)
        best = None
        for alpha in args.ridge_alphas:
            ridge = fit_ridge(fit_features, fit["delta"], fit_weights, alpha)
            prediction = predict_ridge(calibration_features, ridge)
            loss = _weighted_mse(calibration["delta"], prediction, calibration_weights)
            alpha_rows.append({"variant": variant, "ridge_alpha": alpha,
                               "calibration_weighted_mse": loss, "selected": 0})
            if best is None or loss < best[0] - 1e-15:
                best = (loss, float(alpha), prediction)
        selected_alphas[variant] = best[1]
        calibration_predictions[variant] = best[2]
        for row in alpha_rows:
            if row["variant"] == variant and row["ridge_alpha"] == best[1]:
                row["selected"] = 1
        print("[calibration] {:<14} alpha={:g} weighted_mse={:.8f}".format(
            variant, best[1], best[0]))

    print("Refitting preprocessing and routers on all training videos...", flush=True)
    final_pca = fit_token_pca(
        train["token_features"], args.pca_dims, args.pca_oversample,
        args.pca_power_iterations, args.seed + 1)
    train_token_projection = transform_token_pca(
        train["token_features"], final_pca, args.batch_size)
    val_token_projection = transform_token_pca(
        val["token_features"], final_pca, args.batch_size)

    summary_rows = []
    budget_rows = []
    prediction_rows = []
    model_reports = []
    for variant in args.variants:
        train_low, low_names = build_low_features(train, variant)
        val_low, _ = build_low_features(val, variant)
        combined = fit_combined_preprocessor(train_token_projection, train_low, train_weights)
        train_features = transform_combined(train_token_projection, train_low, combined)
        val_features = transform_combined(val_token_projection, val_low, combined)
        ridge = fit_ridge(train_features, train["delta"], train_weights,
                          selected_alphas[variant])
        predictions = predict_ridge(val_features, ridge)
        summary, rows, offsets, scores = evaluate_variant(
            variant, val, predictions, val_weights,
            calibration_predictions[variant], calibration_weights,
            args.budgets, costs, args.cost_lambda)
        summary.update({
            "train_examples": len(train["delta"]),
            "fit_examples": len(fit["delta"]),
            "calibration_examples": len(calibration["delta"]),
            "train_videos": len(set(train["sequence"].tolist())),
            "val_videos": len(set(val["sequence"].tolist())),
            "pca_dims": len(final_pca["components"]),
            "pca_explained_variance_ratio": final_pca["explained_variance_ratio"],
            "ridge_alpha": selected_alphas[variant],
            "cost_lambda": args.cost_lambda,
        })
        summary_rows.append(summary)
        budget_rows.extend(rows)

        configuration = {
            "pca_dims": len(final_pca["components"]),
            "pca_oversample": args.pca_oversample,
            "pca_power_iterations": args.pca_power_iterations,
            "ridge_alpha": selected_alphas[variant],
            "seed": args.seed,
            "cost_lambda": args.cost_lambda,
            "calibration_sequences": calibration_sequences,
        }
        model_path = output_dir / "stage4_router_{}.npz".format(variant)
        metadata = save_model(model_path, variant, run, final_pca, combined,
                              ridge, low_names, configuration)
        model_reports.append({"variant": variant, "path": str(model_path),
                              "sha256": file_hash(model_path),
                              "fingerprint": metadata["fingerprint"]})

        best_action = np.argmax(val["iou"], axis=1)
        for index in range(len(val["delta"])):
            row = {
                "variant": variant,
                "sequence": val["sequence"][index],
                "trajectory": val["trajectory"][index],
                "frame_id": int(val["frame_id"][index]),
                "balanced_weight": val_weights[index],
                "fast_iou": val["iou"][index, 0],
                "best_action": ACTIONS[int(best_action[index])],
                "best_action_iou": val["iou"][index, best_action[index]],
                "sgla_action": "layer{}".format(int(val["selected_layer"][index])),
                "sgla_iou": val["iou"][index, int(val["selected_layer"][index]) - 6],
                "predicted_layer": int(CANDIDATE_LAYERS[offsets[index]]),
                "predicted_utility": scores[index],
            }
            for offset, layer in enumerate(CANDIDATE_LAYERS):
                row["pred_delta_l{}".format(layer)] = predictions[index, offset]
                row["actual_delta_l{}".format(layer)] = val["delta"][index, offset]
            prediction_rows.append(row)
        print("[validation] {:<14} AUROC={:.4f} layer_hit={:.2%} positive@100={:.2%}".format(
            variant, summary["benefit_auroc"], summary["candidate_layer_hit_rate"],
            rows[-1]["val_deep_ratio"]))

    write_csv(output_dir / "stage4_router_alpha_selection.csv", alpha_rows)
    write_csv(output_dir / "stage4_router_summary.csv", summary_rows)
    write_csv(output_dir / "stage4_router_budget.csv", budget_rows)
    write_csv(output_dir / "stage4_router_val_predictions.csv", prediction_rows)
    experiment = {
        "schema": MODEL_SCHEMA,
        "cache": str(input_dir),
        "cache_fingerprint": run["fingerprint"],
        "manifest_fingerprint": run["manifest"]["fingerprint"],
        "feature_policy": "Causal cache fields only; GT/IoU/delta are labels and evaluation data.",
        "split_policy": "Video-disjoint train/val; video-disjoint fit/calibration within train.",
        "metric_warning": "Current-state mean IoU on frozen trajectories; not closed-loop AUC.",
        "latency_warning": ("Estimated from Stage-4A components and excludes PCA/Ridge router latency."
                            if costs is not None else "No latency profiles supplied."),
        "configuration": {
            "variants": args.variants, "budgets": args.budgets,
            "pca_dims": args.pca_dims, "pca_oversample": args.pca_oversample,
            "pca_power_iterations": args.pca_power_iterations,
            "ridge_alphas": args.ridge_alphas,
            "calibration_fraction": args.calibration_fraction,
            "calibration_sequences": calibration_sequences,
            "seed": args.seed, "cost_lambda": args.cost_lambda,
            "profile_paths": [str(Path(path).expanduser().resolve()) for path in args.profile_dirs],
        },
        "profile_means_ms": costs,
        "models": model_reports,
    }
    experiment["fingerprint"] = fingerprint(experiment)
    write_json(output_dir / "stage4_router_experiment.json", experiment)

    print("\nvariant          alpha   AUROC  layer_hit  fast_iou  oracle_iou")
    for row in summary_rows:
        print("{variant:<16} {ridge_alpha:>6g} {benefit_auroc:>7.4f} "
              "{candidate_layer_hit_rate:>10.2%} {fast_current_state_mean_iou:>9.4f} "
              "{best_action_current_state_mean_iou:>10.4f}".format(**row))
    print("Reports and deployable Ridge models: {}".format(output_dir))
    print("Next decision: keep a variant only if held-out utility prediction beats the "
          "Stage-3 causal baseline; then benchmark its real router latency before closed-loop use.")


if __name__ == "__main__":
    main()
