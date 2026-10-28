"""Validate a Fast-trajectory sequence router with out-of-fold predictions.

The predictor sees only bounding boxes produced by the Fast policy during an
early observation segment. Ground-truth overlaps are used as training targets
and held-out evaluation labels, never as predictor inputs. One policy is then
assigned to each sequence under an exact shared candidate-block budget.

This is an offline causal-feature transfer diagnostic. Its late-policy values
come from pre-existing closed-loop trajectories, so it is not yet a deployable
mid-sequence switching result.
"""

import argparse
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from stage3_closed_loop_summary import load_closed_loop
from stage6_budget_oracle import POLICY_ORDER, solve_multiple_choice, write_csv
from stage6_temporal_stability import (
    bootstrap_mean,
    build_temporal_sequences,
    temporal_matrices,
)


FEATURE_NAMES = (
    "log_observed_frames",
    "speed_mean",
    "speed_std",
    "speed_p90",
    "speed_max",
    "speed_last",
    "abs_dx_mean",
    "abs_dx_std",
    "abs_dy_mean",
    "abs_dy_std",
    "scale_change_mean",
    "scale_change_std",
    "scale_change_p90",
    "scale_change_last",
    "consecutive_iou_mean",
    "consecutive_iou_std",
    "consecutive_iou_p10",
    "consecutive_iou_last",
    "acceleration_mean",
    "acceleration_std",
    "acceleration_p90",
    "displacement_last",
    "displacement_max",
    "path_efficiency",
    "area_log_last",
    "area_log_std",
    "area_log_range",
    "area_log_trend",
    "aspect_log_last",
    "aspect_log_std",
    "aspect_log_range",
    "x_reversal_ratio",
    "y_reversal_ratio",
)


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


def _box(row):
    box = np.asarray(
        [_as_float(row, "bbox_{}".format(name)) for name in "xywh"],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(box)) or box[2] <= 0.0 or box[3] <= 0.0:
        raise ValueError(
            "Invalid Fast box at {}/{} frame {}".format(
                row.get("dataset", ""),
                row.get("sequence", ""),
                row.get("frame_id", ""),
            )
        )
    return box


def _box_iou(first, second):
    first_x2 = first[0] + first[2]
    first_y2 = first[1] + first[3]
    second_x2 = second[0] + second[2]
    second_y2 = second[1] + second[3]
    width = max(0.0, min(first_x2, second_x2) - max(first[0], second[0]))
    height = max(0.0, min(first_y2, second_y2) - max(first[1], second[1]))
    intersection = width * height
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / max(union, 1e-12)


def _mean(values):
    return float(np.mean(values)) if len(values) else 0.0


def _std(values):
    return float(np.std(values)) if len(values) else 0.0


def _percentile(values, percentile):
    return float(np.percentile(values, percentile)) if len(values) else 0.0


def _last(values):
    return float(values[-1]) if len(values) else 0.0


def _trend(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return 0.0
    x = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    centered = x - x.mean()
    return float(centered.dot(values - values.mean()) / centered.dot(centered))


def _reversal_ratio(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return 0.0
    products = values[1:] * values[:-1]
    active = (np.abs(values[1:]) > 1e-8) & (np.abs(values[:-1]) > 1e-8)
    if not np.any(active):
        return 0.0
    return float(np.mean(products[active] < 0.0))


def trajectory_features(rows, early_frames):
    """Extract scale-normalized features from initialization plus Fast prefix."""
    ordered = sorted(rows, key=lambda row: row["frame_id"])
    initial = [row for row in ordered if row["frame_id"] == 0]
    tracked = [row for row in ordered if row["frame_id"] > 0]
    if len(initial) != 1:
        raise ValueError("Each Fast trajectory needs exactly one frame 0 row")
    if early_frames < 1 or len(tracked) < early_frames:
        raise ValueError("Fast trajectory is shorter than the early segment")

    boxes = np.stack([_box(initial[0])] + [_box(row) for row in tracked[:early_frames]])
    epsilon = 1e-8
    centers = boxes[:, :2] + 0.5 * boxes[:, 2:]
    widths = np.maximum(boxes[:, 2], epsilon)
    heights = np.maximum(boxes[:, 3], epsilon)

    dx = np.diff(centers[:, 0]) / widths[:-1]
    dy = np.diff(centers[:, 1]) / heights[:-1]
    velocity = np.stack([dx, dy], axis=1)
    speed = np.linalg.norm(velocity, axis=1)
    log_width_change = np.diff(np.log(widths))
    log_height_change = np.diff(np.log(heights))
    scale_change = np.sqrt(log_width_change ** 2 + log_height_change ** 2)
    consecutive_iou = np.asarray(
        [_box_iou(boxes[index - 1], boxes[index]) for index in range(1, len(boxes))],
        dtype=np.float64,
    )
    acceleration = (
        np.linalg.norm(np.diff(velocity, axis=0), axis=1)
        if len(velocity) > 1
        else np.empty(0, dtype=np.float64)
    )

    relative_x = (centers[:, 0] - centers[0, 0]) / widths[0]
    relative_y = (centers[:, 1] - centers[0, 1]) / heights[0]
    displacement = np.sqrt(relative_x ** 2 + relative_y ** 2)
    path_length = float(speed.sum())
    path_efficiency = float(displacement[-1] / max(path_length, epsilon))
    area_log = np.log(widths * heights / (widths[0] * heights[0]))
    aspect_log = np.log(widths / heights)

    feature = np.asarray(
        [
            math.log1p(early_frames),
            _mean(speed),
            _std(speed),
            _percentile(speed, 90.0),
            float(np.max(speed)),
            _last(speed),
            _mean(np.abs(dx)),
            _std(np.abs(dx)),
            _mean(np.abs(dy)),
            _std(np.abs(dy)),
            _mean(scale_change),
            _std(scale_change),
            _percentile(scale_change, 90.0),
            _last(scale_change),
            _mean(consecutive_iou),
            _std(consecutive_iou),
            _percentile(consecutive_iou, 10.0),
            _last(consecutive_iou),
            _mean(acceleration),
            _std(acceleration),
            _percentile(acceleration, 90.0),
            float(displacement[-1]),
            float(np.max(displacement)),
            path_efficiency,
            float(area_log[-1]),
            _std(area_log),
            float(np.ptp(area_log)),
            _trend(area_log),
            float(aspect_log[-1]),
            _std(aspect_log),
            float(np.ptp(aspect_log)),
            _reversal_ratio(dx),
            _reversal_ratio(dy),
        ],
        dtype=np.float64,
    )
    if len(feature) != len(FEATURE_NAMES):
        raise AssertionError("Feature-name and feature-value mismatch")
    return np.clip(np.nan_to_num(feature, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0)


def build_router_sequences(rows, policies, fraction):
    sequences = build_temporal_sequences(rows, policies, fraction)
    fast_rows = defaultdict(list)
    for row in rows:
        if row["policy"] == "fast":
            fast_rows[(row["dataset"], row["sequence"])].append(row)
    for sequence in sequences:
        key = (sequence["dataset"], sequence["sequence"])
        if key not in fast_rows:
            raise ValueError("Missing Fast trajectory for {}/{}".format(*key))
        sequence["features"] = trajectory_features(
            fast_rows[key], sequence["early_frames"]
        )
    return sequences


def assign_folds(num_examples, num_folds, seed):
    if num_folds < 2:
        raise ValueError("--folds must be at least 2")
    if num_folds > num_examples:
        raise ValueError("Number of folds exceeds number of sequences")
    indices = list(range(num_examples))
    random.Random(seed).shuffle(indices)
    folds = np.empty(num_examples, dtype=np.int64)
    for position, index in enumerate(indices):
        folds[index] = position % num_folds
    return folds


def _ridge_predict(train_features, train_targets, test_features, alpha):
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train = (train_features - mean) / scale
    test = (test_features - mean) / scale
    train = np.concatenate([train, np.ones((len(train), 1))], axis=1)
    test = np.concatenate([test, np.ones((len(test), 1))], axis=1)
    gram = train.T.dot(train) / len(train)
    cross = train.T.dot(train_targets) / len(train)
    regularizer = np.eye(gram.shape[0], dtype=np.float64) * float(alpha)
    regularizer[-1, -1] = 0.0
    try:
        weights = np.linalg.solve(gram + regularizer, cross)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(gram + regularizer, cross, rcond=None)[0]
    return test.dot(weights)


def _pearson(first, second):
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if len(first) < 2 or np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def _select_alpha(features, targets, train_indices, alphas, inner_folds, seed):
    inner_folds = min(int(inner_folds), len(train_indices))
    if inner_folds < 2:
        raise ValueError("Not enough training sequences for inner validation")
    fold_ids = assign_folds(len(train_indices), inner_folds, seed)
    rows = []
    best = None
    for alpha in alphas:
        prediction = np.empty_like(targets[train_indices])
        for fold in range(inner_folds):
            inner_test_local = np.flatnonzero(fold_ids == fold)
            inner_train_local = np.flatnonzero(fold_ids != fold)
            inner_train = train_indices[inner_train_local]
            inner_test = train_indices[inner_test_local]
            prediction[inner_test_local] = _ridge_predict(
                features[inner_train],
                targets[inner_train],
                features[inner_test],
                alpha,
            )
        mse = float(np.mean((prediction - targets[train_indices]) ** 2))
        rows.append({"ridge_alpha": float(alpha), "inner_oof_mse": mse})
        candidate = (mse, -float(alpha), float(alpha))
        if best is None or candidate < best:
            best = candidate
    selected_alpha = best[2]
    for row in rows:
        row["selected"] = int(row["ridge_alpha"] == selected_alpha)
    return selected_alpha, rows


def fit_oof_router(features, utilities, fast_index, folds, inner_folds, alphas, seed):
    nonfast = [index for index in range(utilities.shape[1]) if index != fast_index]
    targets = utilities[:, nonfast]
    fold_ids = assign_folds(len(features), folds, seed)
    predictions = np.zeros_like(utilities)
    prior_predictions = np.zeros_like(utilities)
    fold_rows = []
    alpha_rows = []

    for fold in range(folds):
        test_indices = np.flatnonzero(fold_ids == fold)
        train_indices = np.flatnonzero(fold_ids != fold)
        selected_alpha, candidates = _select_alpha(
            features,
            targets,
            train_indices,
            alphas,
            inner_folds,
            seed + 1000 + fold,
        )
        fold_prediction = _ridge_predict(
            features[train_indices],
            targets[train_indices],
            features[test_indices],
            selected_alpha,
        )
        predictions[np.ix_(test_indices, nonfast)] = fold_prediction
        prior_predictions[np.ix_(test_indices, nonfast)] = targets[
            train_indices
        ].mean(axis=0)
        error = fold_prediction - targets[test_indices]
        fold_rows.append(
            {
                "fold": fold,
                "train_sequences": len(train_indices),
                "test_sequences": len(test_indices),
                "ridge_alpha": selected_alpha,
                "test_mse": float(np.mean(error ** 2)),
                "test_mae": float(np.mean(np.abs(error))),
                "test_pearson": _pearson(
                    fold_prediction, targets[test_indices]
                ),
            }
        )
        for row in candidates:
            row.update({"fold": fold})
            alpha_rows.append(row)
    return predictions, prior_predictions, fold_ids, fold_rows, alpha_rows


def _distribution(indices, policies):
    counts = Counter(policies[index] for index in indices.tolist())
    return "|".join(
        "{}:{}".format(policy, counts.get(policy, 0)) for policy in policies
    )


def _selected_values(values, indices):
    return values[np.arange(len(values)), indices]


def analyze_fraction(
    sequences,
    policies,
    fraction,
    budgets,
    folds,
    inner_folds,
    alphas,
    bootstrap_samples,
    rng,
    seed,
):
    features = np.stack([sequence["features"] for sequence in sequences])
    early_values, late_values, late_costs = temporal_matrices(sequences, policies)
    fast_index = policies.index("fast")
    nonfast_indices = [
        index for index in range(len(policies)) if index != fast_index
    ]
    actual_utilities = late_values - late_values[:, [fast_index]]
    predictions, priors, fold_ids, fold_rows, alpha_rows = fit_oof_router(
        features,
        actual_utilities,
        fast_index,
        folds,
        inner_folds,
        alphas,
        seed,
    )
    for row in fold_rows:
        row["early_fraction"] = fraction
    for row in alpha_rows:
        row["early_fraction"] = fraction

    total_late_frames = sum(sequence["late_frames"] for sequence in sequences)
    capacities = [
        int(math.floor(total_late_frames * budget / 100.0 + 1e-12))
        for budget in budgets
    ]
    router_solutions = solve_multiple_choice(predictions, late_costs, capacities)
    prior_solutions = solve_multiple_choice(priors, late_costs, capacities)
    early_solutions = solve_multiple_choice(early_values, late_costs, capacities)
    oracle_solutions = solve_multiple_choice(late_values, late_costs, capacities)

    fixed = []
    for policy_index, policy in enumerate(policies):
        fixed.append(
            {
                "policy": policy,
                "policy_index": policy_index,
                "total_cost": int(late_costs[:, policy_index].sum()),
                "auc": float(late_values[:, policy_index].mean()),
            }
        )

    prediction_rows = []
    feature_rows = []
    for sequence_index, sequence in enumerate(sequences):
        feature_row = {
            "early_fraction": fraction,
            "dataset": sequence["dataset"],
            "sequence": sequence["sequence"],
            "fold": int(fold_ids[sequence_index]),
            "early_frames": sequence["early_frames"],
            "late_frames": sequence["late_frames"],
        }
        feature_row.update(
            {
                name: features[sequence_index, index]
                for index, name in enumerate(FEATURE_NAMES)
            }
        )
        feature_rows.append(feature_row)
        for policy_index, policy in enumerate(policies):
            prediction_rows.append(
                {
                    "early_fraction": fraction,
                    "dataset": sequence["dataset"],
                    "sequence": sequence["sequence"],
                    "fold": int(fold_ids[sequence_index]),
                    "policy": policy,
                    "early_frames": sequence["early_frames"],
                    "late_frames": sequence["late_frames"],
                    "late_candidate_blocks": int(late_costs[sequence_index, policy_index]),
                    "late_auc": late_values[sequence_index, policy_index],
                    "actual_gain_vs_fast": actual_utilities[sequence_index, policy_index],
                    "oof_predicted_gain_vs_fast": predictions[sequence_index, policy_index],
                    "train_prior_gain_vs_fast": priors[sequence_index, policy_index],
                }
            )

    summary_rows = []
    assignment_rows = []
    for budget, capacity in zip(budgets, capacities):
        router = router_solutions[capacity]
        prior = prior_solutions[capacity]
        early = early_solutions[capacity]
        oracle = oracle_solutions[capacity]
        router_indices = router["policy_indices"]
        prior_indices = prior["policy_indices"]
        early_indices = early["policy_indices"]
        oracle_indices = oracle["policy_indices"]

        feasible_fixed = [row for row in fixed if row["total_cost"] <= capacity]
        best_fixed = max(
            feasible_fixed,
            key=lambda row: (row["auc"], -row["total_cost"]),
        )
        fixed_values = late_values[:, best_fixed["policy_index"]]
        router_values = _selected_values(late_values, router_indices)
        prior_values = _selected_values(late_values, prior_indices)
        early_values_late = _selected_values(late_values, early_indices)
        oracle_values = _selected_values(late_values, oracle_indices)
        router_auc = float(router_values.mean())
        prior_auc = float(prior_values.mean())
        early_auc = float(early_values_late.mean())
        oracle_auc = float(oracle_values.mean())
        headroom = oracle_auc - best_fixed["auc"]
        bootstrap_fixed = bootstrap_mean(
            router_values - fixed_values, bootstrap_samples, rng
        )
        bootstrap_prior = bootstrap_mean(
            router_values - prior_values, bootstrap_samples, rng
        )
        summary_rows.append(
            {
                "early_fraction": fraction,
                "budget_percent": float(budget),
                "action_set": "|".join(policies),
                "num_sequences": len(sequences),
                "num_features": len(FEATURE_NAMES),
                "folds": folds,
                "late_frames": total_late_frames,
                "budget_candidate_blocks": capacity,
                "router_candidate_blocks": router["total_cost"],
                "router_block_ratio": router["total_cost"] / total_late_frames,
                "router_auc": router_auc,
                "best_fixed_policy": best_fixed["policy"],
                "best_fixed_auc": best_fixed["auc"],
                "gain_vs_best_fixed": router_auc - best_fixed["auc"],
                "gain_vs_fixed_ci_low": bootstrap_fixed["ci_low"],
                "gain_vs_fixed_ci_high": bootstrap_fixed["ci_high"],
                "gain_vs_fixed_probability_positive": bootstrap_fixed[
                    "probability_positive"
                ],
                "mean_prior_auc": prior_auc,
                "gain_vs_mean_prior": router_auc - prior_auc,
                "gain_vs_prior_ci_low": bootstrap_prior["ci_low"],
                "gain_vs_prior_ci_high": bootstrap_prior["ci_high"],
                "gain_vs_prior_probability_positive": bootstrap_prior[
                    "probability_positive"
                ],
                "early_gt_transfer_auc": early_auc,
                "late_gt_oracle_auc": oracle_auc,
                "recovery_of_late_oracle_headroom": (
                    (router_auc - best_fixed["auc"]) / headroom
                    if headroom > 1e-12
                    else float("nan")
                ),
                "router_oracle_assignment_match": float(
                    np.mean(router_indices == oracle_indices)
                ),
                "router_selected_beneficial_ratio": float(
                    np.mean(
                        _selected_values(actual_utilities, router_indices) > 0.0
                    )
                ),
                "oof_gain_pearson": _pearson(
                    predictions[:, nonfast_indices],
                    actual_utilities[:, nonfast_indices],
                ),
                "router_distribution": _distribution(router_indices, policies),
                "mean_prior_distribution": _distribution(prior_indices, policies),
                "early_gt_distribution": _distribution(early_indices, policies),
                "late_oracle_distribution": _distribution(oracle_indices, policies),
            }
        )

        for sequence_index, sequence in enumerate(sequences):
            selected_index = int(router_indices[sequence_index])
            prior_index = int(prior_indices[sequence_index])
            early_index = int(early_indices[sequence_index])
            oracle_index = int(oracle_indices[sequence_index])
            assignment_rows.append(
                {
                    "early_fraction": fraction,
                    "budget_percent": float(budget),
                    "dataset": sequence["dataset"],
                    "sequence": sequence["sequence"],
                    "fold": int(fold_ids[sequence_index]),
                    "early_frames": sequence["early_frames"],
                    "late_frames": sequence["late_frames"],
                    "router_policy": policies[selected_index],
                    "router_candidate_blocks": int(
                        late_costs[sequence_index, selected_index]
                    ),
                    "router_predicted_gain": predictions[sequence_index, selected_index],
                    "router_actual_late_auc": late_values[sequence_index, selected_index],
                    "router_actual_gain_vs_fast": actual_utilities[sequence_index, selected_index],
                    "best_fixed_policy": best_fixed["policy"],
                    "best_fixed_late_auc": fixed_values[sequence_index],
                    "gain_vs_best_fixed": router_values[sequence_index]
                    - fixed_values[sequence_index],
                    "mean_prior_policy": policies[prior_index],
                    "mean_prior_late_auc": prior_values[sequence_index],
                    "early_gt_policy": policies[early_index],
                    "early_gt_policy_late_auc": early_values_late[sequence_index],
                    "late_oracle_policy": policies[oracle_index],
                    "late_oracle_auc": oracle_values[sequence_index],
                    "gap_to_late_oracle": oracle_values[sequence_index]
                    - router_values[sequence_index],
                }
            )
    return (
        summary_rows,
        assignment_rows,
        prediction_rows,
        feature_rows,
        fold_rows,
        alpha_rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--policies", nargs="+", choices=POLICY_ORDER, default=list(POLICY_ORDER)
    )
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.25, 0.5])
    parser.add_argument("--budgets", nargs="+", type=float, default=[10])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner_folds", type=int, default=4)
    parser.add_argument(
        "--ridge_alphas",
        nargs="+",
        type=float,
        default=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    )
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    policies = tuple(dict.fromkeys(args.policies))
    if len(policies) < 2 or "fast" not in policies or "sgla" not in policies:
        raise ValueError("--policies must contain at least fast and sgla")
    fractions = sorted(set(args.fractions))
    budgets = sorted(set(args.budgets))
    alphas = sorted(set(args.ridge_alphas))
    if any(not 0.0 < value < 1.0 for value in fractions):
        raise ValueError("--fractions must lie strictly between 0 and 1")
    if not budgets or budgets[0] < 0.0 or budgets[-1] > 100.0:
        raise ValueError("--budgets must lie in [0, 100]")
    if not alphas or any(value <= 0.0 for value in alphas):
        raise ValueError("--ridge_alphas must be positive")
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap_samples must be positive")

    input_path, paths, rows = load_closed_loop(args.input)
    output = Path(args.output_dir).expanduser().resolve()
    rng = np.random.default_rng(args.seed)
    all_results = [[] for _ in range(6)]
    for fraction in fractions:
        sequences = build_router_sequences(rows, policies, fraction)
        results = analyze_fraction(
            sequences,
            policies,
            fraction,
            budgets,
            args.folds,
            args.inner_folds,
            alphas,
            args.bootstrap_samples,
            rng,
            args.seed + int(round(fraction * 1000)),
        )
        for destination, result in zip(all_results, results):
            destination.extend(result)

    names = (
        "stage6_causal_router_summary.csv",
        "stage6_causal_router_assignments.csv",
        "stage6_causal_router_predictions.csv",
        "stage6_causal_router_features.csv",
        "stage6_causal_router_folds.csv",
        "stage6_causal_router_alpha_selection.csv",
    )
    paths_out = []
    for name, result in zip(names, all_results):
        path = output / name
        write_csv(path, result)
        paths_out.append(path)

    print(
        "Loaded {} rows from {} policy CSV files under {}.".format(
            len(rows), len(paths), input_path
        )
    )
    print(
        "Features: {} Fast-only trajectory values; {}-fold OOF Ridge.".format(
            len(FEATURE_NAMES), args.folds
        )
    )
    print(
        "early  budget  router_auc  best_fixed  gain  paired_95%           "
        "prior_auc  early_GT  late_oracle  recovery"
    )
    for row in all_results[0]:
        interval = "[{:.4f},{:.4f}]".format(
            row["gain_vs_fixed_ci_low"], row["gain_vs_fixed_ci_high"]
        )
        print(
            "{early_fraction:>5.0%}  {budget_percent:>5.1f}%  "
            "{router_auc:>10.4f}  {best_fixed_policy:<18s} "
            "{gain_vs_best_fixed:>6.4f}  {interval:<19s} "
            "{mean_prior_auc:>9.4f}  {early_gt_transfer_auc:>8.4f}  "
            "{late_gt_oracle_auc:>11.4f}  "
            "{recovery_of_late_oracle_headroom:>8.2%}".format(
                interval=interval, **row
            )
        )
    for path in paths_out:
        print("CSV: {}".format(path))
    print(
        "Warning: features are causal and OOF, but late outcomes reuse "
        "pre-existing policy trajectories and known late costs. Confirm any "
        "successful router with a true switched closed-loop experiment."
    )


if __name__ == "__main__":
    main()
