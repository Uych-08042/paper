"""Diagnose Stage-4C routers without retraining or running the tracker.

This consumes an audited Stage-4B cache and an existing Stage-4C output
directory.  It reports target statistics, constant-predictor baselines,
prediction/action collapse, model coefficient norms, and an exact top-k budget
diagnostic.  The top-k table uses no validation labels for router selection,
but it is an offline fixed-horizon comparison rather than an online policy.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from stage4_common import ACTIONS, write_csv, write_json
from stage4_router_train import (
    CANDIDATE_LAYERS,
    TOLERANCE,
    _weighted_binary_auc,
    _weighted_mean,
    _weighted_mse,
    _weighted_pearson,
    balanced_group_weights,
    calibration_split,
    estimated_latency,
    load_cache_dataset,
    subset_data,
)


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _key(sequence, trajectory, frame_id):
    return str(sequence), str(trajectory), int(frame_id)


def load_predictions(router_dir, val):
    rows = _read_csv(Path(router_dir) / "stage4_router_val_predictions.csv")
    variants = sorted(set(row["variant"] for row in rows))
    expected_keys = [
        _key(sequence, trajectory, frame_id)
        for sequence, trajectory, frame_id in zip(
            val["sequence"], val["trajectory"], val["frame_id"])
    ]
    if len(set(expected_keys)) != len(expected_keys):
        raise ValueError("Validation example keys are not unique")
    output = {}
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        lookup = {
            _key(row["sequence"], row["trajectory"], row["frame_id"]): row
            for row in selected
        }
        if len(lookup) != len(selected) or set(lookup) != set(expected_keys):
            raise ValueError("Prediction rows do not match the audited validation cache")
        ordered = [lookup[key] for key in expected_keys]
        predictions = np.asarray([
            [float(row["pred_delta_l{}".format(layer)])
             for layer in CANDIDATE_LAYERS]
            for row in ordered
        ], dtype=np.float64)
        layers = np.asarray([int(row["predicted_layer"]) for row in ordered],
                            dtype=np.int64)
        scores = np.asarray([float(row["predicted_utility"]) for row in ordered],
                            dtype=np.float64)
        if np.any((layers < 7) | (layers > 12)) or not np.isfinite(predictions).all():
            raise ValueError("Invalid router predictions")
        output[variant] = {
            "predictions": predictions,
            "layers": layers,
            "offsets": layers - 7,
            "scores": scores,
        }
    return output


def target_statistics(data):
    rows = []
    for split in ("train", "val"):
        split_data = subset_data(data, data["split"] == split)
        for trajectory in ("all",) + tuple(sorted(set(split_data["trajectory"].tolist()))):
            current = (split_data if trajectory == "all" else
                       subset_data(split_data, split_data["trajectory"] == trajectory))
            weights = balanced_group_weights(current["sequence"], current["trajectory"])
            delta = np.asarray(current["delta"], dtype=np.float64)
            iou = np.asarray(current["iou"], dtype=np.float64)
            best_candidate = delta.max(axis=1)
            best_action = iou.max(axis=1)
            selector_actions = np.asarray(current["selected_layer"], dtype=np.int64) - 6
            selector_iou = iou[np.arange(len(iou)), selector_actions]
            rows.append({
                "split": split, "trajectory": trajectory, "action": "overall",
                "examples": len(delta),
                "mean_delta": _weighted_mean(best_candidate, weights),
                "positive_delta_ratio": _weighted_mean(best_candidate > 0.0, weights),
                "best_or_tied_ratio": "",
                "fast_mean_iou": _weighted_mean(iou[:, 0], weights),
                "sgla_selector_mean_iou": _weighted_mean(selector_iou, weights),
                "best_action_mean_iou": _weighted_mean(best_action, weights),
                "selector_ratio": "",
            })
            for offset, layer in enumerate(CANDIDATE_LAYERS):
                layer_delta = delta[:, offset]
                rows.append({
                    "split": split, "trajectory": trajectory,
                    "action": "layer{}".format(layer), "examples": len(delta),
                    "mean_delta": _weighted_mean(layer_delta, weights),
                    "positive_delta_ratio": _weighted_mean(layer_delta > 0.0, weights),
                    "best_or_tied_ratio": _weighted_mean(
                        layer_delta >= best_candidate - TOLERANCE, weights),
                    "fast_mean_iou": "", "sgla_selector_mean_iou": "",
                    "best_action_mean_iou": "",
                    "selector_ratio": _weighted_mean(selector_actions == offset + 1, weights),
                })
    return rows


def constant_baselines(train, val, experiment, alpha_rows):
    fit_mask, calibration_mask, _ = calibration_split(
        train["sequence"],
        float(experiment["configuration"]["calibration_fraction"]),
        int(experiment["configuration"]["seed"]),
    )
    fit = subset_data(train, fit_mask)
    calibration = subset_data(train, calibration_mask)
    fit_weights = balanced_group_weights(fit["sequence"], fit["trajectory"])
    calibration_weights = balanced_group_weights(
        calibration["sequence"], calibration["trajectory"])
    train_weights = balanced_group_weights(train["sequence"], train["trajectory"])
    val_weights = balanced_group_weights(val["sequence"], val["trajectory"])

    def weighted_target_mean(data, weights):
        return np.sum(data["delta"] * weights[:, None], axis=0) / weights.sum()

    fit_mean = weighted_target_mean(fit, fit_weights)
    train_mean = weighted_target_mean(train, train_weights)
    fit_prediction = np.tile(fit_mean, (len(calibration["delta"]), 1))
    val_prediction = np.tile(train_mean, (len(val["delta"]), 1))
    selected_offset = int(np.argmax(train_mean))
    selected_delta = val["delta"][:, selected_offset]
    best_candidate = val["delta"].max(axis=1)
    rows = [{
        "baseline": "zero_delta",
        "predicted_layer": "",
        "calibration_weighted_mse": _weighted_mse(
            calibration["delta"], np.zeros_like(calibration["delta"]),
            calibration_weights),
        "val_weighted_mse": _weighted_mse(
            val["delta"], np.zeros_like(val["delta"]), val_weights),
        "val_benefit_auroc": 0.5,
        "val_predicted_utility_pearson": "",
        "val_all_deep_gain_vs_fast": 0.0,
    }, {
        "baseline": "weighted_train_mean_delta",
        "predicted_layer": int(CANDIDATE_LAYERS[selected_offset]),
        "calibration_weighted_mse": _weighted_mse(
            calibration["delta"], fit_prediction, calibration_weights),
        "val_weighted_mse": _weighted_mse(val["delta"], val_prediction, val_weights),
        "val_benefit_auroc": _weighted_binary_auc(
            best_candidate > 0.0,
            np.full(len(best_candidate), float(train_mean[selected_offset])),
            val_weights),
        "val_predicted_utility_pearson": "",
        "val_all_deep_gain_vs_fast": _weighted_mean(selected_delta, val_weights),
    }]
    for row in alpha_rows:
        if int(row["selected"]):
            rows.append({
                "baseline": "router_{}".format(row["variant"]),
                "predicted_layer": "multi-output",
                "calibration_weighted_mse": float(row["calibration_weighted_mse"]),
                "val_weighted_mse": "", "val_benefit_auroc": "",
                "val_predicted_utility_pearson": "",
                "val_all_deep_gain_vs_fast": "",
            })
    return rows


def prediction_statistics(predictions, val):
    weights = balanced_group_weights(val["sequence"], val["trajectory"])
    targets = np.asarray(val["delta"], dtype=np.float64)
    best_candidate = targets.max(axis=1)
    rows = []
    for variant, values in predictions.items():
        offsets = values["offsets"]
        actual = targets[np.arange(len(targets)), offsets]
        layer_counts = [int(np.sum(values["layers"] == layer))
                        for layer in CANDIDATE_LAYERS]
        rows.append({
            "variant": variant,
            "predicted_utility_mean": _weighted_mean(values["scores"], weights),
            "predicted_utility_std": float(np.std(values["scores"])),
            "predicted_utility_p10": float(np.percentile(values["scores"], 10)),
            "predicted_utility_p50": float(np.percentile(values["scores"], 50)),
            "predicted_utility_p90": float(np.percentile(values["scores"], 90)),
            "benefit_auroc": _weighted_binary_auc(
                best_candidate > 0.0, values["scores"], weights),
            "utility_pearson": _weighted_pearson(
                values["scores"], best_candidate, weights),
            "chosen_actual_delta_mean": _weighted_mean(actual, weights),
            "chosen_beneficial_ratio": _weighted_mean(actual > 0.0, weights),
            "layer_hit_rate": _weighted_mean(
                actual >= best_candidate - TOLERANCE, weights),
            "predicted_layer_distribution": "|".join(
                "L{}:{:.2%}".format(layer, count / len(targets))
                for layer, count in zip(CANDIDATE_LAYERS, layer_counts)),
        })
    return rows


def prediction_comparisons(predictions):
    names = sorted(predictions)
    rows = []
    for first_index, first in enumerate(names):
        for second in names[first_index + 1:]:
            first_values = predictions[first]
            second_values = predictions[second]
            difference = np.abs(first_values["predictions"]
                                - second_values["predictions"])
            rows.append({
                "first": first, "second": second,
                "max_abs_prediction_difference": float(difference.max()),
                "mean_abs_prediction_difference": float(difference.mean()),
                "layer_agreement": float(np.mean(
                    first_values["layers"] == second_values["layers"])),
                "score_pearson": (float(np.corrcoef(
                    first_values["scores"], second_values["scores"])[0, 1])
                    if np.std(first_values["scores"]) > 1e-12
                    and np.std(second_values["scores"]) > 1e-12 else ""),
            })
    if not rows:
        rows.append({
            "first": names[0], "second": "",
            "max_abs_prediction_difference": "",
            "mean_abs_prediction_difference": "",
            "layer_agreement": "", "score_pearson": "",
        })
    return rows


def coefficient_statistics(router_dir, variants):
    rows = []
    for variant in variants:
        path = Path(router_dir) / "stage4_router_{}.npz".format(variant)
        with np.load(str(path), allow_pickle=False) as archive:
            coefficient = archive["ridge_coefficient"].astype(np.float64)
            pca_dims = archive["pca_components"].shape[0]
            low_names = archive["low_feature_names"].astype(str)
            low = coefficient[pca_dims:]
            rows.append({
                "variant": variant,
                "pca_dims": pca_dims,
                "low_dims": len(low_names),
                "pca_coefficient_l2": float(np.linalg.norm(coefficient[:pca_dims])),
                "low_coefficient_l2": float(np.linalg.norm(low)),
                "selector_coefficient_l2": float(np.linalg.norm(
                    low[[name.startswith("selector_") for name in low_names]]))
                    if any(name.startswith("selector_") for name in low_names) else "",
                "fast_current_coefficient_l2": float(np.linalg.norm(
                    low[[name.startswith("fast_") for name in low_names]]))
                    if any(name.startswith("fast_") for name in low_names) else "",
                "total_coefficient_l2": float(np.linalg.norm(coefficient)),
                "intercept_range": float(np.ptp(archive["ridge_intercept"])),
            })
    return rows


def exact_budget_statistics(predictions, val, budgets, costs):
    weights = balanced_group_weights(val["sequence"], val["trajectory"])
    iou = np.asarray(val["iou"], dtype=np.float64)
    targets = np.asarray(val["delta"], dtype=np.float64)
    fast_iou = iou[:, 0]
    selector_actions = np.asarray(val["selected_layer"], dtype=np.int64) - 6
    selector_iou = iou[np.arange(len(iou)), selector_actions]
    best_action_iou = iou.max(axis=1)
    best_offsets = np.argmax(targets, axis=1)
    positive_best = np.maximum(targets.max(axis=1), 0.0)
    oracle_order = np.argsort(-(weights * positive_best), kind="mergesort")
    rows = []
    for variant, values in predictions.items():
        predicted_order = np.argsort(-values["scores"], kind="mergesort")
        eligible = predicted_order[values["scores"][predicted_order] > 0.0]
        for budget in budgets:
            maximum = int(math.floor(len(iou) * float(budget) / 100.0 + 1e-12))
            deep_indices = eligible[:maximum]
            actions = np.zeros(len(iou), dtype=np.int64)
            actions[deep_indices] = values["offsets"][deep_indices] + 1
            selected_iou = iou[np.arange(len(iou)), actions]
            regret = best_action_iou - selected_iou

            oracle_deep = oracle_order[positive_best[oracle_order] > 0.0][:maximum]
            oracle_actions = np.zeros(len(iou), dtype=np.int64)
            oracle_actions[oracle_deep] = best_offsets[oracle_deep] + 1
            oracle_iou = iou[np.arange(len(iou)), oracle_actions]
            latency = estimated_latency(variant, actions, costs)
            rows.append({
                "variant": variant,
                "budget_policy": "global_score_topk_offline_diagnostic",
                "budget_percent": float(budget),
                "deep_frames": int(np.sum(actions > 0)),
                "actual_deep_ratio": float(np.mean(actions > 0)),
                "current_state_mean_iou": _weighted_mean(selected_iou, weights),
                "gain_vs_fast": _weighted_mean(selected_iou - fast_iou, weights),
                "gain_vs_sgla_selector": _weighted_mean(
                    selected_iou - selector_iou, weights),
                "oracle_current_state_mean_iou": _weighted_mean(oracle_iou, weights),
                "oracle_gain_vs_fast": _weighted_mean(oracle_iou - fast_iou, weights),
                "action_hit_rate": _weighted_mean(regret <= TOLERANCE, weights),
                "regret_mean": _weighted_mean(regret, weights),
                "estimated_latency_ms_excluding_router": (
                    latency if np.isfinite(latency) else ""),
                "estimated_fps_excluding_router": (
                    1000.0 / latency if np.isfinite(latency) and latency > 0 else ""),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Audited Stage-4B cache")
    parser.add_argument("--router_dir", required=True, help="Existing Stage-4C output")
    parser.add_argument("--budgets", nargs="+", type=float,
                        default=[0, 10, 20, 30, 50, 100])
    args = parser.parse_args()
    if any(value < 0 or value > 100 for value in args.budgets):
        raise ValueError("Budgets must lie in [0, 100]")

    router_dir = Path(args.router_dir).expanduser().resolve()
    experiment = json.loads(
        (router_dir / "stage4_router_experiment.json").read_text(encoding="utf-8"))
    data, run = load_cache_dataset(args.input)
    if experiment["cache_fingerprint"] != run["fingerprint"]:
        raise ValueError("Router output and cache fingerprints disagree")
    train = subset_data(data, data["split"] == "train")
    val = subset_data(data, data["split"] == "val")
    predictions = load_predictions(router_dir, val)
    alpha_rows = _read_csv(router_dir / "stage4_router_alpha_selection.csv")
    costs = experiment.get("profile_means_ms")

    target_rows = target_statistics(data)
    baseline_rows = constant_baselines(train, val, experiment, alpha_rows)
    prediction_rows = prediction_statistics(predictions, val)
    comparison_rows = prediction_comparisons(predictions)
    coefficient_rows = coefficient_statistics(router_dir, predictions)
    budget_rows = exact_budget_statistics(predictions, val, args.budgets, costs)

    paths = {
        "targets": router_dir / "stage4_router_diagnostic_targets.csv",
        "baselines": router_dir / "stage4_router_diagnostic_baselines.csv",
        "predictions": router_dir / "stage4_router_diagnostic_predictions.csv",
        "comparisons": router_dir / "stage4_router_diagnostic_comparisons.csv",
        "coefficients": router_dir / "stage4_router_diagnostic_coefficients.csv",
        "exact_budget": router_dir / "stage4_router_exact_budget.csv",
    }
    for path, rows in zip(paths.values(), (
            target_rows, baseline_rows, prediction_rows, comparison_rows,
            coefficient_rows, budget_rows)):
        write_csv(path, rows)
    metadata = {
        "schema": "stage4_router_diagnostic_v1",
        "cache_fingerprint": run["fingerprint"],
        "router_experiment_fingerprint": experiment["fingerprint"],
        "warning": ("Exact top-k selection uses the complete validation score horizon "
                    "but never validation labels; it is a controlled offline budget "
                    "comparison, not an online causal scheduler."),
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    write_json(router_dir / "stage4_router_diagnostic.json", metadata)

    print("Constant baseline and selected-router calibration MSE:")
    for row in baseline_rows:
        print("  {:<34} {}".format(
            row["baseline"], row["calibration_weighted_mse"]))
    print("\nPrediction behavior on untouched val videos:")
    for row in prediction_rows:
        print("  {variant:<14} AUROC={benefit_auroc:.4f} score_std={predicted_utility_std:.6f} "
              "layers={predicted_layer_distribution}".format(**row))
    print("\nExact offline budget diagnostic:")
    print("variant          budget   deep     gain_fast   gain_sgla   latency_ms")
    for row in budget_rows:
        print("{variant:<16} {budget_percent:>6.1f}% {actual_deep_ratio:>7.2%} "
              "{gain_vs_fast:>11.6f} {gain_vs_sgla_selector:>11.6f} "
              "{estimated_latency_ms_excluding_router!s:>12}".format(**row))
    print("Diagnostic reports: {}".format(router_dir))


if __name__ == "__main__":
    main()
