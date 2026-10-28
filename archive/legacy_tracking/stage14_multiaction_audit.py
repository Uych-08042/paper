"""Audit long-horizon L7-L12 action values and grouped-OOF learnability.

The branches are one-step interventions followed by Fast tracking.  They
estimate Q^Fast_H(s,a), not a complete reinforcement-learning value function.
All model predictions are nested, video-grouped and out of fold.
"""

import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint, write_csv, write_json
from stage4_router_train import (
    _selector_statistics,
    _weighted_binary_auc,
    _weighted_mse,
    _weighted_pearson,
    build_low_features,
    fit_combined_preprocessor,
    fit_ridge,
    fit_token_pca,
    predict_ridge,
    transform_combined,
    transform_token_pca,
)
from stage6_temporal_stability import bootstrap_mean
from stage8_immediate_gate import _group_weights, empty_projection, grouped_folds
from stage8_motion_gate import MOTION_FEATURE_NAMES, causal_motion_features
from stage14_multiaction_common import (
    ACTION_NAMES,
    CANDIDATE_LAYERS,
    load_and_audit_cache,
    load_multiaction_run,
    multiaction_cache_path,
)


SCHEMA = "stage14_multiaction_audit_v1"
VARIANTS = (
    "history_only",
    "selector_fast_motion",
    "token_selector_fast_motion",
)
PRIMARY_VARIANT = "token_selector_fast_motion"
SCREEN_PEARSON = 0.10


def _json_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def uses_tokens(variant):
    return variant == "token_selector_fast_motion"


def load_dataset(directory, allow_incomplete=False):
    directory, run = load_multiaction_run(directory)
    fields = (
        "token_features",
        "fast_score_map",
        "selector_prob",
        "selected_layer",
        "previous_state",
        "previous_fast_box",
        "fast_box",
        "image_hw",
        "resize_factor",
        "frame_id",
        "anchor_offset",
        "anchor_fraction",
        "discounted_delta",
        "valid_fraction",
    )
    chunks = {name: [] for name in fields}
    chunks.update(
        {
            "split": [],
            "sequence": [],
            "motion_history": [],
            "immediate_targets": [],
        }
    )
    reports = []
    missing = []
    entries = run["manifest"]["sequences"]
    for entry in entries:
        path = multiaction_cache_path(directory, entry)
        if not path.is_file():
            missing.append(path)
            continue
        report, values, metadata = load_and_audit_cache(path, entry, run)
        reports.append(report)
        motion = causal_motion_features(
            values["baseline_boxes"], values["anchor_offset"], metadata["init_box"]
        )
        eligible = values["eligible"]
        if not eligible.any():
            continue
        for name in fields:
            chunks[name].append(values[name][eligible].copy())
        chunks["motion_history"].append(motion[eligible])
        chunks["immediate_targets"].append(
            values["delta_iou"][eligible, :, 0].astype(np.float64, copy=True)
        )
        count = int(eligible.sum())
        chunks["split"].append(np.full(count, entry["split"], dtype="U8"))
        chunks["sequence"].append(np.full(count, entry["sequence"], dtype="U128"))
    if missing and not allow_incomplete:
        raise ValueError(
            "Stage-14 cache is incomplete: {} of {} files missing".format(
                len(missing), len(entries)
            )
        )
    if not chunks["sequence"]:
        raise ValueError("No eligible Stage-14 anchors were loaded")
    data = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    data["previous_action"] = np.zeros(len(data["sequence"]), dtype=np.int64)
    data["action_targets"] = np.asarray(data["discounted_delta"], dtype=np.float64)
    if data["action_targets"].shape[1] != len(CANDIDATE_LAYERS):
        raise ValueError("Stage-14 action target width changed")
    for name in (
        "action_targets",
        "immediate_targets",
        "motion_history",
        "valid_fraction",
    ):
        if not np.isfinite(data[name]).all():
            raise ValueError("Non-finite Stage-14 dataset field: {}".format(name))
    return directory, run, data, reports, missing, len(entries)


def build_variant_low_features(data, variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown Stage-14 feature variant: {}".format(variant))
    if variant == "history_only":
        return build_low_features(data, "token_only")
    fast, fast_names = build_low_features(data, "fast_aware")
    selector = np.asarray(data["selector_prob"], dtype=np.float64)
    selector_features = np.concatenate((selector, _selector_statistics(selector)), axis=1)
    selector_names = tuple(
        ["selector_l{}".format(layer) for layer in CANDIDATE_LAYERS]
        + ["selector_mean", "selector_std", "selector_max", "selector_margin", "selector_entropy"]
    )
    features = np.concatenate((fast, selector_features, data["motion_history"]), axis=1)
    names = tuple(fast_names) + selector_names + tuple(MOTION_FEATURE_NAMES)
    if features.shape[1] != len(names):
        raise AssertionError("Stage-14 feature-name mismatch")
    return features.astype(np.float32, copy=False), names


def _weighted_multi_pearson(predictions, targets, weights):
    return _weighted_pearson(
        np.asarray(predictions, dtype=np.float64).reshape(-1),
        np.asarray(targets, dtype=np.float64).reshape(-1),
        np.repeat(np.asarray(weights, dtype=np.float64), targets.shape[1]),
    )


def _fold_predictions(
    data,
    low_features,
    variants,
    fit_index,
    test_index,
    alphas,
    pca_dimensions,
    pca_oversample,
    pca_power_iterations,
    seed,
    batch_size,
):
    fit_weights = _group_weights(data["sequence"][fit_index])
    if any(uses_tokens(variant) for variant in variants):
        pca = fit_token_pca(
            data["token_features"][fit_index],
            pca_dimensions,
            pca_oversample,
            pca_power_iterations,
            seed,
        )
        fit_token = transform_token_pca(data["token_features"][fit_index], pca, batch_size)
        test_token = transform_token_pca(data["token_features"][test_index], pca, batch_size)
    else:
        fit_token = empty_projection(len(fit_index))
        test_token = empty_projection(len(test_index))
    predictions = {}
    for variant in variants:
        if uses_tokens(variant):
            fit_projection, test_projection = fit_token, test_token
        else:
            fit_projection = empty_projection(len(fit_index))
            test_projection = empty_projection(len(test_index))
        preprocessor = fit_combined_preprocessor(
            fit_projection, low_features[variant][fit_index], fit_weights
        )
        fit_features = transform_combined(
            fit_projection, low_features[variant][fit_index], preprocessor
        )
        test_features = transform_combined(
            test_projection, low_features[variant][test_index], preprocessor
        )
        predictions[variant] = {}
        for alpha in alphas:
            model = fit_ridge(
                fit_features,
                data["action_targets"][fit_index],
                fit_weights,
                alpha,
            )
            predictions[variant][alpha] = np.asarray(
                predict_ridge(test_features, model), dtype=np.float64
            )
    return predictions


def nested_group_oof(
    data,
    variants,
    outer_folds,
    inner_folds,
    alphas,
    pca_dimensions,
    pca_oversample,
    pca_power_iterations,
    seed,
    batch_size,
):
    sequences = np.asarray(data["sequence"], dtype=str)
    targets = data["action_targets"]
    low_features = {
        variant: build_variant_low_features(data, variant)[0] for variant in variants
    }
    outer = grouped_folds(sequences, outer_folds, seed)
    oof = {
        variant: np.full(targets.shape, np.nan, dtype=np.float64) for variant in variants
    }
    outer_rows = []
    alpha_rows = []
    selected_alphas = {variant: [] for variant in variants}

    for outer_index, (outer_train, outer_test, outer_names) in enumerate(outer):
        inner = grouped_folds(
            sequences[outer_train], inner_folds, seed + 1000 + outer_index
        )
        inner_predictions = {
            variant: {
                alpha: np.full((len(outer_train), len(CANDIDATE_LAYERS)), np.nan)
                for alpha in alphas
            }
            for variant in variants
        }
        for inner_index, (inner_fit, inner_test, _) in enumerate(inner):
            global_fit = outer_train[inner_fit]
            global_test = outer_train[inner_test]
            predicted = _fold_predictions(
                data,
                low_features,
                variants,
                global_fit,
                global_test,
                alphas,
                pca_dimensions,
                pca_oversample,
                pca_power_iterations,
                seed + outer_index * 100 + inner_index,
                batch_size,
            )
            for variant in variants:
                for alpha in alphas:
                    inner_predictions[variant][alpha][inner_test] = predicted[variant][alpha]

        chosen = {}
        inner_weights = _group_weights(sequences[outer_train])
        for variant in variants:
            best = None
            for alpha in alphas:
                prediction = inner_predictions[variant][alpha]
                if not np.isfinite(prediction).all():
                    raise RuntimeError("Nested inner OOF predictions are incomplete")
                loss = _weighted_mse(targets[outer_train], prediction, inner_weights)
                row = {
                    "outer_fold": outer_index,
                    "variant": variant,
                    "ridge_alpha": alpha,
                    "inner_oof_weighted_mse": loss,
                    "inner_oof_value_pearson": _weighted_multi_pearson(
                        prediction, targets[outer_train], inner_weights
                    ),
                    "selected": 0,
                }
                alpha_rows.append(row)
                if best is None or loss < best[0] - 1e-15:
                    best = (loss, alpha, row)
            chosen[variant] = best[1]
            best[2]["selected"] = 1
            selected_alphas[variant].append(best[1])

        outer_predictions = _fold_predictions(
            data,
            low_features,
            variants,
            outer_train,
            outer_test,
            sorted(set(chosen.values())),
            pca_dimensions,
            pca_oversample,
            pca_power_iterations,
            seed + 10000 + outer_index,
            batch_size,
        )
        test_weights = _group_weights(sequences[outer_test])
        for variant in variants:
            prediction = outer_predictions[variant][chosen[variant]]
            oof[variant][outer_test] = prediction
            predicted_layer = np.argmax(prediction, axis=1)
            true_layer = np.argmax(targets[outer_test], axis=1)
            outer_rows.append(
                {
                    "outer_fold": outer_index,
                    "variant": variant,
                    "selected_ridge_alpha": chosen[variant],
                    "train_sequences": len(set(sequences[outer_train].tolist())),
                    "test_sequences": len(outer_names),
                    "train_examples": len(outer_train),
                    "test_examples": len(outer_test),
                    "test_weighted_mse": _weighted_mse(
                        targets[outer_test], prediction, test_weights
                    ),
                    "test_value_pearson": _weighted_multi_pearson(
                        prediction, targets[outer_test], test_weights
                    ),
                    "test_candidate_layer_hit_rate": float(
                        np.mean(predicted_layer == true_layer)
                    ),
                }
            )
        print(
            "[outer {}/{}] train={} videos, test={} videos; alphas={}".format(
                outer_index + 1,
                len(outer),
                len(set(sequences[outer_train].tolist())),
                len(outer_names),
                "|".join(
                    "{}:{:g}".format(variant, chosen[variant]) for variant in variants
                ),
            ),
            flush=True,
        )
    if any(not np.isfinite(values).all() for values in oof.values()):
        raise RuntimeError("Nested outer OOF predictions are incomplete")
    return {
        "predictions": oof,
        "outer_rows": outer_rows,
        "alpha_rows": alpha_rows,
        "selected_alphas": selected_alphas,
        "low_features": low_features,
        "weights": _group_weights(sequences),
    }


def _sequence_means(sequences, values):
    sequences = np.asarray(sequences, dtype=str)
    values = np.asarray(values, dtype=np.float64)
    names = sorted(set(sequences.tolist()))
    return names, np.asarray(
        [values[sequences == name].mean() for name in names], dtype=np.float64
    )


def _bootstrap(values, samples, rng):
    return bootstrap_mean(np.asarray(values, dtype=np.float64), samples, rng)


def _sign_counts(values, tolerance=1e-12):
    values = np.asarray(values, dtype=np.float64)
    return (
        int(np.sum(values > tolerance)),
        int(np.sum(np.abs(values) <= tolerance)),
        int(np.sum(values < -tolerance)),
    )


def mechanism_rows(data, bootstrap_samples, seed):
    rows = []
    selector_rows = []
    rng = np.random.default_rng(seed)
    for scope in ("train", "val", "all"):
        mask = np.ones(len(data["sequence"]), dtype=bool) if scope == "all" else data["split"] == scope
        if not mask.any():
            continue
        sequences = data["sequence"][mask]
        targets = data["action_targets"][mask]
        native_index = data["selected_layer"][mask] - 7
        native = targets[np.arange(len(targets)), native_index]
        oracle = targets.max(axis=1)
        oracle_with_fast = np.maximum(oracle, 0.0)
        immediate_targets = data["immediate_targets"][mask]
        immediate_oracle_index = np.argmax(immediate_targets, axis=1)
        long_oracle_index = np.argmax(targets, axis=1)
        methods = [("fast", "", np.zeros(len(targets)))]
        methods.extend(
            ("fixed_l{}".format(layer), layer, targets[:, index])
            for index, layer in enumerate(CANDIDATE_LAYERS)
        )
        methods.extend(
            (
                ("native_selector", "dynamic", native),
                ("oracle_candidate", "dynamic", oracle),
                ("oracle_with_fast", "dynamic", oracle_with_fast),
            )
        )
        fixed_macro = []
        for method, layer, values in methods:
            names, sequence_values = _sequence_means(sequences, values)
            stats = _bootstrap(sequence_values, bootstrap_samples, rng)
            wins, ties, losses = _sign_counts(sequence_values)
            rows.append(
                {
                    "scope": scope,
                    "method": method,
                    "candidate_layer": layer,
                    "num_sequences": len(names),
                    "num_anchors": len(values),
                    "mean_discounted_gain_vs_fast": float(sequence_values.mean()),
                    "median_sequence_gain": float(np.median(sequence_values)),
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "probability_positive": stats["probability_positive"],
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                }
            )
            if isinstance(layer, int):
                fixed_macro.append((float(sequence_values.mean()), layer))

        _, headroom = _sequence_means(sequences, oracle - native)
        headroom_stats = _bootstrap(headroom, bootstrap_samples, rng)
        _, oracle_with_fast_macro = _sequence_means(sequences, oracle_with_fast)
        oracle_with_fast_stats = _bootstrap(
            oracle_with_fast_macro, bootstrap_samples, rng
        )
        _, regret = _sequence_means(sequences, oracle - native)
        hit = np.isclose(native, oracle, rtol=0, atol=1e-12)
        _, hit_macro = _sequence_means(sequences, hit.astype(np.float64))
        immediate_hit = np.isclose(
            immediate_targets[np.arange(len(targets)), native_index],
            immediate_targets.max(axis=1),
            rtol=0,
            atol=1e-12,
        )
        _, immediate_hit_macro = _sequence_means(
            sequences, immediate_hit.astype(np.float64)
        )
        _, horizon_agreement_macro = _sequence_means(
            sequences, (immediate_oracle_index == long_oracle_index).astype(np.float64)
        )
        native_distribution = Counter(data["selected_layer"][mask].tolist())
        oracle_layers = np.asarray(CANDIDATE_LAYERS)[np.argmax(targets, axis=1)]
        oracle_distribution = Counter(oracle_layers.tolist())
        best_fixed = max(fixed_macro)
        selector_rows.append(
            {
                "scope": scope,
                "num_sequences": len(set(sequences.tolist())),
                "num_anchors": len(targets),
                "best_fixed_layer": best_fixed[1],
                "best_fixed_gain": best_fixed[0],
                "native_selector_gain": float(_sequence_means(sequences, native)[1].mean()),
                "oracle_candidate_gain": float(_sequence_means(sequences, oracle)[1].mean()),
                "oracle_with_fast_gain": float(_sequence_means(sequences, oracle_with_fast)[1].mean()),
                "oracle_with_fast_ci_low": oracle_with_fast_stats["ci_low"],
                "oracle_with_fast_ci_high": oracle_with_fast_stats["ci_high"],
                "oracle_with_fast_probability_positive": oracle_with_fast_stats[
                    "probability_positive"
                ],
                "oracle_minus_native": float(headroom.mean()),
                "oracle_minus_native_ci_low": headroom_stats["ci_low"],
                "oracle_minus_native_ci_high": headroom_stats["ci_high"],
                "oracle_minus_native_probability_positive": headroom_stats["probability_positive"],
                "native_candidate_layer_hit_rate": float(hit_macro.mean()),
                "native_immediate_layer_hit_rate": float(immediate_hit_macro.mean()),
                "immediate_long_oracle_agreement": float(
                    horizon_agreement_macro.mean()
                ),
                "native_candidate_regret": float(regret.mean()),
                "native_distribution": "|".join(
                    "l{}:{}".format(layer, native_distribution.get(layer, 0))
                    for layer in CANDIDATE_LAYERS
                ),
                "oracle_distribution": "|".join(
                    "l{}:{}".format(layer, oracle_distribution.get(layer, 0))
                    for layer in CANDIDATE_LAYERS
                ),
            }
        )
    return rows, selector_rows


def model_summary_rows(
    data,
    result,
    variants,
    primary_variant,
    pca_dimensions,
    bootstrap_samples,
    seed,
):
    rows = []
    rng = np.random.default_rng(seed)
    targets = data["action_targets"]
    sequences = data["sequence"]
    weights = result["weights"]
    native_index = data["selected_layer"] - 7
    native = targets[np.arange(len(targets)), native_index]
    true_index = np.argmax(targets, axis=1)
    for variant in variants:
        prediction = result["predictions"][variant]
        predicted_index = np.argmax(prediction, axis=1)
        learned = targets[np.arange(len(targets)), predicted_index]
        learned_with_fast = np.where(prediction.max(axis=1) > 0.0, learned, 0.0)
        _, difference = _sequence_means(sequences, learned - native)
        stats = _bootstrap(difference, bootstrap_samples, rng)
        _, learned_macro = _sequence_means(sequences, learned)
        _, learned_with_fast_macro = _sequence_means(sequences, learned_with_fast)
        _, native_macro = _sequence_means(sequences, native)
        _, regret_macro = _sequence_means(sequences, targets.max(axis=1) - learned)
        distribution = Counter((predicted_index + 7).tolist())
        rows.append(
            {
                "variant": variant,
                "is_primary": int(variant == primary_variant),
                "num_examples": len(targets),
                "num_sequences": len(set(sequences.tolist())),
                "feature_dims_without_pca": result["low_features"][variant].shape[1],
                "pca_dims": 0 if not uses_tokens(variant) else pca_dimensions,
                "selected_alpha_by_outer_fold": "|".join(
                    "{:g}".format(value) for value in result["selected_alphas"][variant]
                ),
                "nested_oof_weighted_mse": _weighted_mse(targets, prediction, weights),
                "nested_oof_value_pearson": _weighted_multi_pearson(prediction, targets, weights),
                "nested_oof_any_benefit_auroc": _weighted_binary_auc(
                    targets.max(axis=1) > 0.0, prediction.max(axis=1), weights
                ),
                "candidate_layer_hit_rate": float(np.mean(predicted_index == true_index)),
                "learned_gain": float(learned_macro.mean()),
                "learned_with_fast_gain": float(learned_with_fast_macro.mean()),
                "native_selector_gain": float(native_macro.mean()),
                "learned_minus_native": float(difference.mean()),
                "learned_minus_native_ci_low": stats["ci_low"],
                "learned_minus_native_ci_high": stats["ci_high"],
                "learned_minus_native_probability_positive": stats["probability_positive"],
                "learned_candidate_regret": float(regret_macro.mean()),
                "predicted_action_distribution": "|".join(
                    "l{}:{}".format(layer, distribution.get(layer, 0))
                    for layer in CANDIDATE_LAYERS
                ),
            }
        )
    return rows


def _topk_mask(scores, budget):
    count = 0 if budget <= 0.0 else min(len(scores), int(math.ceil(len(scores) * budget / 100.0)))
    selected = np.zeros(len(scores), dtype=bool)
    if count:
        selected[np.argsort(-np.asarray(scores), kind="mergesort")[:count]] = True
    return selected


def budget_rows(
    data, result, variants, primary_variant, budgets, bootstrap_samples, seed
):
    rows = []
    rng = np.random.default_rng(seed)
    targets = data["action_targets"]
    sequences = data["sequence"]
    native_index = data["selected_layer"] - 7
    native = targets[np.arange(len(targets)), native_index]
    oracle = targets.max(axis=1)
    for variant in variants:
        prediction = result["predictions"][variant]
        predicted_index = np.argmax(prediction, axis=1)
        predicted_score = prediction[np.arange(len(prediction)), predicted_index]
        selected_values = targets[np.arange(len(targets)), predicted_index]
        for budget in budgets:
            selected = _topk_mask(predicted_score, budget)
            gain = np.where(selected, selected_values, 0.0)
            native_same = np.where(selected, native, 0.0)
            random_layer_same = np.where(selected, targets.mean(axis=1), 0.0)
            oracle_same = np.where(selected, oracle, 0.0)
            oracle_global_mask = _topk_mask(oracle, budget)
            oracle_global = np.where(oracle_global_mask, oracle, 0.0)
            _, gain_macro = _sequence_means(sequences, gain)
            _, native_diff = _sequence_means(sequences, gain - native_same)
            _, random_lift = _sequence_means(sequences, gain - random_layer_same)
            _, oracle_same_macro = _sequence_means(sequences, oracle_same)
            _, oracle_global_macro = _sequence_means(sequences, oracle_global)
            gain_stats = _bootstrap(gain_macro, bootstrap_samples, rng)
            native_stats = _bootstrap(native_diff, bootstrap_samples, rng)
            random_stats = _bootstrap(random_lift, bootstrap_samples, rng)
            rows.append(
                {
                    "variant": variant,
                    "is_primary": int(variant == primary_variant),
                    "budget_percent": float(budget),
                    "num_sequences": len(gain_macro),
                    "num_anchors": len(targets),
                    "num_selected": int(selected.sum()),
                    "selected_ratio": float(selected.mean()),
                    "mean_discounted_gain": float(gain_macro.mean()),
                    "gain_ci_low": gain_stats["ci_low"],
                    "gain_ci_high": gain_stats["ci_high"],
                    "gain_probability_positive": gain_stats["probability_positive"],
                    "native_same_anchor_gain": float((gain_macro - native_diff).mean()),
                    "gain_vs_native_same_anchors": float(native_diff.mean()),
                    "gain_vs_native_ci_low": native_stats["ci_low"],
                    "gain_vs_native_ci_high": native_stats["ci_high"],
                    "gain_vs_native_probability_positive": native_stats["probability_positive"],
                    "random_layer_same_anchor_gain": float((gain_macro - random_lift).mean()),
                    "lift_vs_random_layer": float(random_lift.mean()),
                    "lift_vs_random_ci_low": random_stats["ci_low"],
                    "lift_vs_random_ci_high": random_stats["ci_high"],
                    "lift_vs_random_probability_positive": random_stats["probability_positive"],
                    "oracle_same_anchor_gain": float(oracle_same_macro.mean()),
                    "oracle_global_budget_gain": float(oracle_global_macro.mean()),
                    "oracle_global_recovery": (
                        float(gain_macro.mean() / oracle_global_macro.mean())
                        if abs(float(oracle_global_macro.mean())) > 1e-15
                        else float("nan")
                    ),
                }
            )
    return rows


def prediction_rows(data, result, variants):
    rows = []
    targets = data["action_targets"]
    for index in range(len(targets)):
        row = {
            "split": data["split"][index],
            "sequence": data["sequence"][index],
            "frame_id": int(data["frame_id"][index]),
            "anchor_offset": int(data["anchor_offset"][index]),
            "native_layer": int(data["selected_layer"][index]),
            "immediate_oracle_layer": int(
                np.argmax(data["immediate_targets"][index]) + 7
            ),
            "oracle_layer": int(np.argmax(targets[index]) + 7),
        }
        for layer_index, layer in enumerate(CANDIDATE_LAYERS):
            row["actual_l{}".format(layer)] = float(targets[index, layer_index])
        for variant in variants:
            prediction = result["predictions"][variant][index]
            row["{}_layer".format(variant)] = int(np.argmax(prediction) + 7)
            row["{}_score".format(variant)] = float(np.max(prediction))
            for layer_index, layer in enumerate(CANDIDATE_LAYERS):
                row["{}_pred_l{}".format(variant, layer)] = float(prediction[layer_index])
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--primary_variant", choices=VARIANTS, default=PRIMARY_VARIANT)
    parser.add_argument("--outer_folds", type=int, default=5)
    parser.add_argument("--inner_folds", type=int, default=4)
    parser.add_argument("--pca_dims", type=int, default=64)
    parser.add_argument("--pca_oversample", type=int, default=12)
    parser.add_argument("--pca_power_iterations", type=int, default=1)
    parser.add_argument(
        "--ridge_alphas", nargs="+", type=float,
        default=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    )
    parser.add_argument("--budgets", nargs="+", type=float, default=[0, 5, 10, 20, 30, 50, 100])
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2044)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--allow_incomplete", action="store_true")
    parser.add_argument("--audit_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if len(set(args.variants)) != len(args.variants):
        raise ValueError("Variants must be unique")
    if args.primary_variant not in args.variants:
        raise ValueError("--primary_variant must be included in --variants")
    if (
        args.outer_folds < 2
        or args.inner_folds < 2
        or args.pca_dims < 1
        or args.pca_oversample < 0
        or args.pca_power_iterations < 0
        or args.bootstrap_samples < 1
        or args.batch_size < 1
    ):
        raise ValueError("Invalid fold, PCA, bootstrap or batch argument")
    if not args.ridge_alphas or any(value <= 0.0 for value in args.ridge_alphas):
        raise ValueError("Ridge alphas must be positive")
    if not args.budgets or any(value < 0.0 or value > 100.0 for value in args.budgets):
        raise ValueError("Budgets must lie in [0,100]")

    input_dir, source_run, data, cache_reports, missing, expected = load_dataset(
        args.input, args.allow_incomplete
    )
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError("Use a new --output_dir or pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "stage14_cache_summary.csv", cache_reports)
    print(
        "Audited {} of {} Stage-14 clips ({} missing); {} eligible anchors.".format(
            len(cache_reports), expected, len(missing), len(data["sequence"])
        )
    )
    if args.audit_only:
        print("Audit-only mode complete; no OOF models were fit.")
        return

    sequences = sorted(set(data["sequence"].tolist()))
    if args.outer_folds > len(sequences):
        raise ValueError("Outer folds exceed the number of videos")
    smallest_outer_train = len(sequences) - int(math.ceil(len(sequences) / args.outer_folds))
    if args.inner_folds > smallest_outer_train:
        raise ValueError("Inner folds exceed an outer training split")
    alphas = tuple(sorted(set(float(value) for value in args.ridge_alphas)))
    budgets = tuple(sorted(set(float(value) for value in args.budgets)))
    print(
        "Target: six H{} discounted candidate-minus-Fast values; nested {}/{}-fold OOF.".format(
            source_run["horizon"], args.outer_folds, args.inner_folds
        )
    )
    result = nested_group_oof(
        data,
        tuple(args.variants),
        args.outer_folds,
        args.inner_folds,
        alphas,
        args.pca_dims,
        args.pca_oversample,
        args.pca_power_iterations,
        args.seed,
        args.batch_size,
    )

    action_rows, selector_rows = mechanism_rows(
        data, args.bootstrap_samples, args.seed + 100
    )
    models = model_summary_rows(
        data,
        result,
        tuple(args.variants),
        args.primary_variant,
        args.pca_dims,
        args.bootstrap_samples,
        args.seed + 200,
    )
    budgets_rows = budget_rows(
        data,
        result,
        tuple(args.variants),
        args.primary_variant,
        budgets,
        args.bootstrap_samples,
        args.seed + 300,
    )
    predictions = prediction_rows(data, result, tuple(args.variants))
    write_csv(output / "stage14_action_summary.csv", action_rows)
    write_csv(output / "stage14_selector_oracle_summary.csv", selector_rows)
    write_csv(output / "stage14_model_summary.csv", models)
    write_csv(output / "stage14_model_budget.csv", budgets_rows)
    write_csv(output / "stage14_oof_predictions.csv", predictions)
    write_csv(output / "stage14_outer_folds.csv", result["outer_rows"])
    write_csv(output / "stage14_alpha_selection.csv", result["alpha_rows"])

    mechanism = next(row for row in selector_rows if row["scope"] == "all")
    primary = next(row for row in models if row["variant"] == args.primary_variant)
    mechanism_pass = (
        mechanism["oracle_minus_native_ci_low"] > 0.0
        and mechanism["oracle_with_fast_ci_low"] > 0.0
    )
    prediction_pass = (
        primary["nested_oof_value_pearson"] >= SCREEN_PEARSON
        and primary["learned_minus_native_ci_low"] > 0.0
    )
    if not mechanism_pass:
        status = "NO_MULTIACTION_HEADROOM"
        next_step = "Do not train an L7-L12 policy: matched long-horizon oracle does not beat the native selector."
    elif not prediction_pass:
        status = "HEADROOM_NOT_PREDICTABLE"
        next_step = (
            "Do not start RL with the current state representation. Preserve the oracle analysis, "
            "then redesign state or collect on-policy transitions before another frozen screen."
        )
    else:
        status = "PASS_SCREEN"
        next_step = (
            "Proceed to a small offline contextual-bandit or fitted-Q pilot, then require a true "
            "closed-loop held-out comparison before claiming improvement."
        )
    decision = {
        "schema": SCHEMA,
        "status": status,
        "analysis_role": "development_only_nested_group_oof_screen",
        "source_run_fingerprint": source_run["fingerprint"],
        "num_sequences": len(sequences),
        "num_anchors": len(data["sequence"]),
        "primary_variant": args.primary_variant,
        "horizon": source_run["horizon"],
        "gamma": source_run["gamma"],
        "screen": {
            "oracle_minus_native": _json_number(mechanism["oracle_minus_native"]),
            "oracle_minus_native_ci_low": _json_number(
                mechanism["oracle_minus_native_ci_low"]
            ),
            "oracle_with_fast_gain": _json_number(
                mechanism["oracle_with_fast_gain"]
            ),
            "oracle_with_fast_ci_low": _json_number(
                mechanism["oracle_with_fast_ci_low"]
            ),
            "mechanism_pass": mechanism_pass,
            "minimum_value_pearson": SCREEN_PEARSON,
            "observed_value_pearson": _json_number(
                primary["nested_oof_value_pearson"]
            ),
            "learned_minus_native": _json_number(primary["learned_minus_native"]),
            "learned_minus_native_ci_low": _json_number(
                primary["learned_minus_native_ci_low"]
            ),
            "prediction_pass": prediction_pass,
        },
        "next_step": next_step,
        "warning": (
            "These one-step branches estimate Q^Fast_H(s,a). They are not on-policy RL returns, "
            "not a closed-loop policy result, and not external validation."
        ),
        "code_sha256": {
            "stage14_multiaction_audit.py": file_hash(Path(__file__)),
        },
    }
    decision["fingerprint"] = fingerprint(decision)
    write_json(output / "stage14_decision.json", decision)

    print("\nMulti-action mechanism:")
    print("scope  native_gain  oracle_gain  headroom  paired_95%        hit_rate")
    for row in selector_rows:
        print(
            "{:<5s} {:>11.6f} {:>12.6f} {:>9.6f} [{:>7.5f},{:>7.5f}] {:>8.2%}".format(
                row["scope"],
                row["native_selector_gain"],
                row["oracle_candidate_gain"],
                row["oracle_minus_native"],
                row["oracle_minus_native_ci_low"],
                row["oracle_minus_native_ci_high"],
                row["native_candidate_layer_hit_rate"],
            )
        )
    print("\nNested OOF action-value diagnostics:")
    print("variant                       pearson   hit_rate  learned-native  paired_95%")
    for row in models:
        print(
            "{:<29s} {:>7.4f} {:>10.2%} {:>15.6f} [{:>7.5f},{:>7.5f}]".format(
                row["variant"],
                row["nested_oof_value_pearson"],
                row["candidate_layer_hit_rate"],
                row["learned_minus_native"],
                row["learned_minus_native_ci_low"],
                row["learned_minus_native_ci_high"],
            )
        )
    print("\nPredeclared Stage-14 decision: {}".format(status))
    print(next_step)
    print("Decision JSON: {}".format(output / "stage14_decision.json"))
    print("Reports: {}".format(output))


if __name__ == "__main__":
    main()
