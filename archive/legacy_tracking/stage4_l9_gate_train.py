"""Train a utility-aligned Fast-versus-L9 gate from audited Stage-4 caches.

Stage-4C showed that the frozen SGLA selector chooses L9 almost everywhere.
This diagnostic therefore removes candidate-layer selection from the learning
problem and asks the smaller causal question: should the current frame use the
Fast output or spend one extra block on L9?  All reported IoUs are frozen-
trajectory current-state diagnostics, not closed-loop tracking AUCs.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint, write_csv, write_json
from stage4_router_train import (
    _atomic_savez,
    _finite_or_blank,
    _weighted_binary_auc,
    _weighted_mean,
    _weighted_mse,
    _weighted_pearson,
    balanced_group_weights,
    build_low_features,
    calibration_split,
    fit_combined_preprocessor,
    fit_ridge,
    fit_token_pca,
    load_cache_dataset,
    load_profile_costs,
    predict_ridge,
    subset_data,
    transform_combined,
    transform_token_pca,
)


VARIANTS = ("history_only", "token_only", "fast_only", "token_fast")
OBJECTIVES = ("signed", "positive", "benefit")
MODEL_SCHEMA = "stage4_fast_l9_ridge_v1"
L9_DELTA_OFFSET = 2
TOLERANCE = 1e-12


def uses_tokens(variant):
    return variant in ("token_only", "token_fast")


def uses_fast_output(variant):
    return variant in ("fast_only", "token_fast")


def build_variant_low_features(data, variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown L9-gate feature variant: {}".format(variant))
    source = "fast_aware" if uses_fast_output(variant) else "token_only"
    return build_low_features(data, source)


def empty_projection(count):
    return np.empty((int(count), 0), dtype=np.float32)


def objective_specs(objectives, benefit_epsilons):
    specs = []
    for objective in objectives:
        if objective == "benefit":
            specs.extend((objective, float(epsilon)) for epsilon in benefit_epsilons)
        else:
            specs.append((objective, 0.0))
    return specs


def objective_target(delta_l9, objective, epsilon=0.0):
    delta_l9 = np.asarray(delta_l9, dtype=np.float64)
    if objective == "signed":
        return delta_l9
    if objective == "positive":
        return np.maximum(delta_l9, 0.0)
    if objective == "benefit":
        return (delta_l9 > float(epsilon)).astype(np.float64)
    raise ValueError("Unknown objective: {}".format(objective))


def natural_threshold(objective):
    return 0.5 if objective == "benefit" else 0.0


def fit_scalar_ridge(features, targets, weights, alpha):
    """Use the shared multi-output solver without NumPy 1-D broadcasting."""
    raw = fit_ridge(
        features, np.asarray(targets, dtype=np.float64)[:, None], weights, alpha)
    return {
        "coefficient": np.asarray(raw["coefficient"], dtype=np.float64)[:, 0],
        "intercept": float(np.asarray(raw["intercept"], dtype=np.float64)[0]),
    }


def model_id(variant, objective, epsilon):
    if objective != "benefit":
        return "{}__{}".format(variant, objective)
    epsilon_name = "{:g}".format(float(epsilon)).replace("-", "m").replace(".", "p")
    return "{}__benefit_eps{}".format(variant, epsilon_name)


def weighted_average_precision(labels, scores, weights):
    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    positive_total = float(weights[labels].sum())
    if positive_total <= 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    scores = scores[order]
    labels = labels[order]
    weights = weights[order]
    true_positive = 0.0
    predicted_positive = 0.0
    average_precision = 0.0
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and scores[stop] == scores[start]:
            stop += 1
        group_weight = float(weights[start:stop].sum())
        group_positive = float(weights[start:stop][labels[start:stop]].sum())
        true_positive += group_positive
        predicted_positive += group_weight
        average_precision += (group_positive / positive_total) * (
            true_positive / max(predicted_positive, 1e-12))
        start = stop
    return average_precision


def l9_latency(variant, deep, costs):
    """Estimate policy latency while excluding the new gate's own CPU/GPU cost."""
    if costs is None:
        return float("nan")
    deep = np.asarray(deep, dtype=np.bool_)
    base = float(costs["fast"])
    if uses_tokens(variant):
        base += float(costs["pool_all_channels_grid4"])
    extra = float(costs["block_L9"])
    if uses_fast_output(variant):
        repeated_head = (float(costs["fast_then_sgla"]) - float(costs["fast"])
                         - float(costs["selector"]) - float(costs["block_L9"]))
        extra += max(repeated_head, 0.0)
    return base + float(np.mean(deep)) * extra


def exact_budget_rows(model_name, variant, objective, epsilon, scores, val,
                      weights, budgets, costs):
    scores = np.asarray(scores, dtype=np.float64)
    delta_l9 = np.asarray(val["delta"][:, L9_DELTA_OFFSET], dtype=np.float64)
    fast_iou = np.asarray(val["iou"][:, 0], dtype=np.float64)
    l9_gain = _weighted_mean(delta_l9, weights)
    score_order = np.argsort(-scores, kind="mergesort")
    exact_oracle_order = np.argsort(-(weights * delta_l9), kind="mergesort")
    cap_oracle_order = np.argsort(-(weights * np.maximum(delta_l9, 0.0)),
                                  kind="mergesort")
    positive_oracle_order = cap_oracle_order[delta_l9[cap_oracle_order] > 0.0]
    rows = []
    for budget in budgets:
        count = int(math.floor(len(scores) * float(budget) / 100.0 + 1e-12))
        deep = np.zeros(len(scores), dtype=np.bool_)
        deep[score_order[:count]] = True
        selected_delta = np.where(deep, delta_l9, 0.0)
        gain = _weighted_mean(selected_delta, weights)

        oracle_exact = np.zeros(len(scores), dtype=np.bool_)
        oracle_exact[exact_oracle_order[:count]] = True
        oracle_exact_gain = _weighted_mean(
            np.where(oracle_exact, delta_l9, 0.0), weights)
        oracle_cap = np.zeros(len(scores), dtype=np.bool_)
        oracle_cap[positive_oracle_order[:count]] = True
        oracle_cap_gain = _weighted_mean(np.where(oracle_cap, delta_l9, 0.0), weights)

        actual_ratio = float(np.mean(deep))
        random_gain = actual_ratio * l9_gain
        latency = l9_latency(variant, deep, costs)
        selected_weights = weights[deep]
        selected_values = delta_l9[deep]
        if count == 0:
            threshold = float("inf")
        elif count == len(scores):
            threshold = float("-inf")
        else:
            threshold = 0.5 * (float(scores[score_order[count - 1]])
                               + float(scores[score_order[count]]))
        rows.append({
            "model_id": model_name,
            "variant": variant,
            "objective": objective,
            "benefit_epsilon": float(epsilon) if objective == "benefit" else "",
            "budget_policy": "global_score_exact_topk_offline_diagnostic",
            "budget_percent": float(budget),
            "deep_frames": int(deep.sum()),
            "actual_deep_ratio": actual_ratio,
            "score_threshold": _finite_or_blank(threshold),
            "current_state_mean_iou": _weighted_mean(fast_iou + selected_delta, weights),
            "gain_vs_fast": gain,
            "gain_vs_fixed_l9": gain - l9_gain,
            "random_l9_expected_gain": random_gain,
            "lift_over_random_l9": gain - random_gain,
            "l9_oracle_exact_gain": oracle_exact_gain,
            "l9_oracle_budget_cap_gain": oracle_cap_gain,
            "oracle_cap_recovery": (gain / oracle_cap_gain
                                    if oracle_cap_gain > TOLERANCE else ""),
            "selected_actual_delta_mean": (_weighted_mean(selected_values, selected_weights)
                                           if np.any(deep) else ""),
            "selected_beneficial_ratio": (
                _weighted_mean(selected_values > 0.0, selected_weights)
                if np.any(deep) else ""),
            "estimated_latency_ms_excluding_gate": _finite_or_blank(latency),
            "estimated_fps_excluding_gate": (
                _finite_or_blank(1000.0 / latency)
                if np.isfinite(latency) and latency > 0.0 else ""),
        })
    return rows


def save_gate_model(path, name, variant, objective, epsilon, run, pca,
                    preprocessor, ridge, low_names, configuration, token_dims):
    metadata = {
        "schema": MODEL_SCHEMA,
        "model_id": name,
        "variant": variant,
        "objective": objective,
        "benefit_epsilon": float(epsilon),
        "natural_score_threshold": natural_threshold(objective),
        "cache_fingerprint": run["fingerprint"],
        "manifest_fingerprint": run["manifest"]["fingerprint"],
        "action": "layer9",
        "configuration": configuration,
    }
    metadata["fingerprint"] = fingerprint(metadata)
    if pca is None:
        token_mean = np.empty(0, dtype=np.float32)
        token_scale = np.empty(0, dtype=np.float32)
        components = np.empty((0, int(token_dims)), dtype=np.float32)
    else:
        token_mean = pca["mean"]
        token_scale = pca["scale"]
        components = pca["components"]
    _atomic_savez(
        path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
        token_mean=token_mean,
        token_scale=token_scale,
        pca_components=components,
        combined_mean=preprocessor["mean"],
        combined_scale=preprocessor["scale"],
        ridge_coefficient=np.asarray(ridge["coefficient"], dtype=np.float64),
        ridge_intercept=np.asarray(ridge["intercept"], dtype=np.float64),
        low_feature_names=np.asarray(low_names, dtype="U"),
        candidate_layer=np.asarray(9, dtype=np.int64),
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Complete audited Stage-4B cache")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--objectives", nargs="+", choices=OBJECTIVES,
                        default=list(OBJECTIVES))
    parser.add_argument("--benefit_epsilons", nargs="+", type=float,
                        default=[0.0, 0.005, 0.01])
    parser.add_argument("--pca_dims", type=int, default=64)
    parser.add_argument("--pca_oversample", type=int, default=12)
    parser.add_argument("--pca_power_iterations", type=int, default=1)
    parser.add_argument("--ridge_alphas", nargs="+", type=float,
                        default=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--calibration_fraction", type=float, default=0.2)
    parser.add_argument("--budgets", nargs="+", type=float,
                        default=[0, 10, 20, 30, 50, 100])
    parser.add_argument("--profile_dirs", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if (args.pca_dims < 1 or args.pca_oversample < 0
            or args.pca_power_iterations < 0 or args.batch_size < 1
            or not 0.0 < args.calibration_fraction < 1.0):
        raise ValueError("Invalid PCA, calibration or batch-size argument")
    if any(alpha <= 0.0 for alpha in args.ridge_alphas):
        raise ValueError("All ridge alphas must be positive")
    if any(value < 0.0 for value in args.benefit_epsilons):
        raise ValueError("Benefit epsilons must be non-negative")
    if any(value < 0.0 or value > 100.0 for value in args.budgets):
        raise ValueError("Budgets must lie in [0, 100]")
    for values, name in ((args.variants, "variants"), (args.objectives, "objectives"),
                         (args.benefit_epsilons, "benefit epsilons")):
        if len(set(values)) != len(values):
            raise ValueError("{} must be unique".format(name.capitalize()))

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

    data, run = load_cache_dataset(input_dir)
    costs = load_profile_costs(args.profile_dirs)
    if costs is not None and int(run["pool_grid"]) != 4:
        raise ValueError("Profiles measure grid4 pooling but cache uses grid{}".format(
            run["pool_grid"]))
    train = subset_data(data, data["split"] == "train")
    val = subset_data(data, data["split"] == "val")
    if len(train["delta"]) == 0 or len(val["delta"]) == 0:
        raise ValueError("Both train and val need valid examples")

    fit_mask, calibration_mask, calibration_sequences = calibration_split(
        train["sequence"], args.calibration_fraction, args.seed)
    fit = subset_data(train, fit_mask)
    calibration = subset_data(train, calibration_mask)
    fit_weights = balanced_group_weights(fit["sequence"], fit["trajectory"])
    calibration_weights = balanced_group_weights(
        calibration["sequence"], calibration["trajectory"])
    train_weights = balanced_group_weights(train["sequence"], train["trajectory"])
    val_weights = balanced_group_weights(val["sequence"], val["trajectory"])
    specs = objective_specs(args.objectives, args.benefit_epsilons)

    token_variants = [variant for variant in args.variants if uses_tokens(variant)]
    calibration_pca = None
    if token_variants:
        print("Fitting calibration PCA for token variants...", flush=True)
        calibration_pca = fit_token_pca(
            fit["token_features"], args.pca_dims, args.pca_oversample,
            args.pca_power_iterations, args.seed)
        fit_token = transform_token_pca(
            fit["token_features"], calibration_pca, args.batch_size)
        calibration_token = transform_token_pca(
            calibration["token_features"], calibration_pca, args.batch_size)
    else:
        fit_token = empty_projection(len(fit["delta"]))
        calibration_token = empty_projection(len(calibration["delta"]))

    print("Loaded {} train and {} val examples from {} videos; L9-only gate.".format(
        len(train["delta"]), len(val["delta"]), len(set(data["sequence"].tolist()))))
    print("Fit/calibration: {}/{} examples; calibration videos: {}".format(
        len(fit["delta"]), len(calibration["delta"]), len(calibration_sequences)))

    selected = {}
    alpha_rows = []
    for variant in args.variants:
        fit_low, _ = build_variant_low_features(fit, variant)
        calibration_low, _ = build_variant_low_features(calibration, variant)
        if uses_tokens(variant):
            fit_projection = fit_token
            calibration_projection = calibration_token
        else:
            fit_projection = empty_projection(len(fit_low))
            calibration_projection = empty_projection(len(calibration_low))
        preprocessor = fit_combined_preprocessor(
            fit_projection, fit_low, fit_weights)
        fit_features = transform_combined(fit_projection, fit_low, preprocessor)
        calibration_features = transform_combined(
            calibration_projection, calibration_low, preprocessor)
        fit_delta = fit["delta"][:, L9_DELTA_OFFSET]
        calibration_delta = calibration["delta"][:, L9_DELTA_OFFSET]
        for objective, epsilon in specs:
            name = model_id(variant, objective, epsilon)
            fit_target = objective_target(fit_delta, objective, epsilon)
            calibration_target = objective_target(
                calibration_delta, objective, epsilon)
            constant = _weighted_mean(fit_target, fit_weights)
            constant_prediction = np.full(len(calibration_target), constant)
            constant_loss = _weighted_mse(
                calibration_target[:, None], constant_prediction[:, None],
                calibration_weights)
            best = None
            for alpha in args.ridge_alphas:
                ridge = fit_scalar_ridge(
                    fit_features, fit_target, fit_weights, float(alpha))
                prediction = predict_ridge(calibration_features, ridge)
                loss = _weighted_mse(
                    calibration_target[:, None], prediction[:, None],
                    calibration_weights)
                row = {
                    "model_id": name,
                    "variant": variant,
                    "objective": objective,
                    "benefit_epsilon": (float(epsilon)
                                        if objective == "benefit" else ""),
                    "ridge_alpha": float(alpha),
                    "calibration_weighted_mse": loss,
                    "constant_calibration_weighted_mse": constant_loss,
                    "selected": 0,
                }
                alpha_rows.append(row)
                if best is None or loss < best[0] - 1e-15:
                    best = (loss, float(alpha))
            selected[name] = {
                "alpha": best[1], "calibration_loss": best[0],
                "constant_loss": constant_loss,
            }
            for row in alpha_rows:
                if row["model_id"] == name and row["ridge_alpha"] == best[1]:
                    row["selected"] = 1
            print("[calibration] {:<38} alpha={:<7g} mse={:.8f} constant={:.8f}".format(
                name, best[1], best[0], constant_loss))

    final_pca = None
    if token_variants:
        print("Refitting PCA and gates on all training videos...", flush=True)
        final_pca = fit_token_pca(
            train["token_features"], args.pca_dims, args.pca_oversample,
            args.pca_power_iterations, args.seed + 1)
        train_token = transform_token_pca(
            train["token_features"], final_pca, args.batch_size)
        val_token = transform_token_pca(
            val["token_features"], final_pca, args.batch_size)
    else:
        print("Refitting gates on all training videos...", flush=True)
        train_token = empty_projection(len(train["delta"]))
        val_token = empty_projection(len(val["delta"]))

    summary_rows = []
    budget_rows = []
    prediction_rows = []
    model_reports = []
    val_delta = val["delta"][:, L9_DELTA_OFFSET]
    val_fast_iou = val["iou"][:, 0]
    fixed_l9_gain = _weighted_mean(val_delta, val_weights)
    l9_oracle_gain = _weighted_mean(np.maximum(val_delta, 0.0), val_weights)
    configuration = {
        "pca_dims": args.pca_dims,
        "pca_oversample": args.pca_oversample,
        "pca_power_iterations": args.pca_power_iterations,
        "ridge_alphas": args.ridge_alphas,
        "calibration_fraction": args.calibration_fraction,
        "calibration_sequences": calibration_sequences,
        "seed": args.seed,
    }

    for variant in args.variants:
        train_low, low_names = build_variant_low_features(train, variant)
        val_low, _ = build_variant_low_features(val, variant)
        if uses_tokens(variant):
            train_projection = train_token
            val_projection = val_token
            variant_pca = final_pca
        else:
            train_projection = empty_projection(len(train_low))
            val_projection = empty_projection(len(val_low))
            variant_pca = None
        preprocessor = fit_combined_preprocessor(
            train_projection, train_low, train_weights)
        train_features = transform_combined(
            train_projection, train_low, preprocessor)
        val_features = transform_combined(val_projection, val_low, preprocessor)
        train_delta = train["delta"][:, L9_DELTA_OFFSET]

        for objective, epsilon in specs:
            name = model_id(variant, objective, epsilon)
            target = objective_target(train_delta, objective, epsilon)
            val_target = objective_target(val_delta, objective, epsilon)
            ridge = fit_scalar_ridge(
                train_features, target, train_weights, selected[name]["alpha"])
            scores = np.asarray(predict_ridge(val_features, ridge), dtype=np.float64)
            threshold = natural_threshold(objective)
            natural_deep = scores > threshold
            natural_delta = np.where(natural_deep, val_delta, 0.0)
            natural_gain = _weighted_mean(natural_delta, val_weights)
            label_epsilon = epsilon if objective == "benefit" else 0.0
            labels = val_delta > label_epsilon
            main_labels = val_delta > 0.0
            latency = l9_latency(variant, natural_deep, costs)
            summary_rows.append({
                "model_id": name,
                "variant": variant,
                "objective": objective,
                "benefit_epsilon": (float(epsilon)
                                    if objective == "benefit" else ""),
                "train_examples": len(train_delta),
                "val_examples": len(val_delta),
                "train_videos": len(set(train["sequence"].tolist())),
                "val_videos": len(set(val["sequence"].tolist())),
                "feature_dims": int(train_features.shape[1]),
                "low_feature_dims": int(train_low.shape[1]),
                "pca_dims": int(train_projection.shape[1]),
                "pca_explained_variance_ratio": (
                    variant_pca["explained_variance_ratio"]
                    if variant_pca is not None else ""),
                "ridge_alpha": selected[name]["alpha"],
                "calibration_weighted_mse": selected[name]["calibration_loss"],
                "constant_calibration_weighted_mse": selected[name]["constant_loss"],
                "calibration_mse_improvement_vs_constant": (
                    selected[name]["constant_loss"] - selected[name]["calibration_loss"]),
                "passes_constant_calibration_baseline": int(
                    selected[name]["calibration_loss"]
                    < selected[name]["constant_loss"] - 1e-15),
                "val_target_weighted_mse": _weighted_mse(
                    val_target[:, None], scores[:, None], val_weights),
                "l9_benefit_auroc": _weighted_binary_auc(
                    main_labels, scores, val_weights),
                "label_auroc": _weighted_binary_auc(labels, scores, val_weights),
                "label_average_precision": weighted_average_precision(
                    labels, scores, val_weights),
                "score_delta_pearson": _weighted_pearson(
                    scores, val_delta, val_weights),
                "score_mean": _weighted_mean(scores, val_weights),
                "score_std": math.sqrt(max(_weighted_mean(
                    (scores - _weighted_mean(scores, val_weights)) ** 2,
                    val_weights), 0.0)),
                "fast_current_state_mean_iou": _weighted_mean(
                    val_fast_iou, val_weights),
                "fixed_l9_current_state_mean_iou": _weighted_mean(
                    val_fast_iou + val_delta, val_weights),
                "fixed_l9_gain_vs_fast": fixed_l9_gain,
                "frozen_selector_l9_ratio": _weighted_mean(
                    np.asarray(val["selected_layer"]) == 9, val_weights),
                "fast_l9_oracle_current_state_mean_iou": _weighted_mean(
                    val_fast_iou + np.maximum(val_delta, 0.0), val_weights),
                "fast_l9_oracle_gain_vs_fast": l9_oracle_gain,
                "natural_gate_threshold": threshold,
                "natural_gate_deep_ratio": float(np.mean(natural_deep)),
                "natural_gate_gain_vs_fast": natural_gain,
                "natural_gate_gain_vs_fixed_l9": natural_gain - fixed_l9_gain,
                "natural_gate_selected_beneficial_ratio": (
                    _weighted_mean(val_delta[natural_deep] > 0.0,
                                   val_weights[natural_deep])
                    if np.any(natural_deep) else ""),
                "natural_gate_estimated_latency_ms_excluding_gate": (
                    _finite_or_blank(latency)),
            })
            budget_rows.extend(exact_budget_rows(
                name, variant, objective, epsilon, scores, val,
                val_weights, args.budgets, costs))
            for index in range(len(val_delta)):
                prediction_rows.append({
                    "model_id": name,
                    "variant": variant,
                    "objective": objective,
                    "benefit_epsilon": (float(epsilon)
                                        if objective == "benefit" else ""),
                    "split": "val",
                    "sequence": str(val["sequence"][index]),
                    "trajectory": str(val["trajectory"][index]),
                    "frame_id": int(val["frame_id"][index]),
                    "score": float(scores[index]),
                    "natural_deep": int(natural_deep[index]),
                    "delta_l9_fast": float(val_delta[index]),
                    "l9_beneficial": int(main_labels[index]),
                })
            model_path = output_dir / "stage4_l9_gate_{}.npz".format(name)
            model_configuration = dict(configuration)
            model_configuration.update({
                "ridge_alpha": selected[name]["alpha"],
                "calibration_weighted_mse": selected[name]["calibration_loss"],
                "constant_calibration_weighted_mse": selected[name]["constant_loss"],
            })
            report = save_gate_model(
                model_path, name, variant, objective, epsilon, run, variant_pca,
                preprocessor, ridge, low_names, model_configuration,
                train["token_features"].shape[1])
            report["path"] = str(model_path)
            model_reports.append(report)

    paths = {
        "alpha_selection": output_dir / "stage4_l9_gate_alpha_selection.csv",
        "summary": output_dir / "stage4_l9_gate_summary.csv",
        "budget": output_dir / "stage4_l9_gate_budget.csv",
        "predictions": output_dir / "stage4_l9_gate_val_predictions.csv",
    }
    for path, rows in ((paths["alpha_selection"], alpha_rows),
                       (paths["summary"], summary_rows),
                       (paths["budget"], budget_rows),
                       (paths["predictions"], prediction_rows)):
        write_csv(path, rows)
    experiment = {
        "schema": "stage4_fast_l9_gate_experiment_v1",
        "cache_fingerprint": run["fingerprint"],
        "manifest_fingerprint": run["manifest"]["fingerprint"],
        "configuration": vars(args),
        "script_sha256": file_hash(Path(__file__).resolve()),
        "calibration_sequences": calibration_sequences,
        "profile_means_ms": costs,
        "models": model_reports,
        "outputs": {key: str(value) for key, value in paths.items()},
        "warning": (
            "Metrics are frozen-trajectory current-state diagnostics. Exact top-k "
            "uses the complete validation score horizon but no validation labels, "
            "so it is a controlled ranking test rather than an online scheduler."),
    }
    experiment["fingerprint"] = fingerprint(experiment)
    write_json(output_dir / "stage4_l9_gate_experiment.json", experiment)

    print("\nmodel                                  alpha  pass  AUROC  pearson  gate%  gain")
    for row in summary_rows:
        print("{model_id:<38} {ridge_alpha:>6g}  {passes_constant_calibration_baseline:>4d}  "
              "{l9_benefit_auroc:>6.4f}  {score_delta_pearson:>7.4f}  "
              "{natural_gate_deep_ratio:>5.1%}  {natural_gate_gain_vs_fast:>7.5f}"
              .format(**row))
    primary_budgets = [
        row for row in budget_rows
        if row["objective"] == "signed"
        and row["budget_percent"] in (10.0, 20.0, 30.0, 50.0)
    ]
    if primary_budgets:
        print("\nSigned-utility exact-budget ranking diagnostic:")
        print("variant          budget  gain_fast   random_l9   lift_random  oracle_cap  latency_ms")
        for row in primary_budgets:
            print("{variant:<16} {budget_percent:>5.1f}%  {gain_vs_fast:>9.6f}  "
                  "{random_l9_expected_gain:>9.6f}  {lift_over_random_l9:>11.6f}  "
                  "{l9_oracle_budget_cap_gain:>10.6f}  "
                  "{estimated_latency_ms_excluding_gate!s:>10}".format(**row))
    print("\nFixed L9 gain vs Fast: {:.6f}; binary Fast/L9 oracle: {:.6f}".format(
        fixed_l9_gain, l9_oracle_gain))
    print("Reports and diagnostic Ridge models: {}".format(output_dir))
    print("Primary files: {} and {}".format(paths["summary"], paths["budget"]))


if __name__ == "__main__":
    main()
