"""Train a causal gate for the immediate value of one fixed-L9 action.

The 120 development videos are used for sequence-grouped cross-validation and
the final fit.  The 40 manifest validation videos are never used for PCA,
standardization, alpha selection, or score thresholds.  Validation budgets are
exact offline top-k ranking diagnostics based only on frozen model scores.
"""

import argparse
import json
import math
import random
from collections import Counter
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
    build_low_features,
    fit_combined_preprocessor,
    fit_ridge,
    fit_token_pca,
    predict_ridge,
    transform_combined,
    transform_token_pca,
)
from stage6_temporal_stability import bootstrap_mean
from stage8_intervention_summary import load_records


MODEL_SCHEMA = "stage8_immediate_l9_ridge_v1"
VARIANTS = (
    "history_only",
    "selector_only",
    "token_only",
    "token_selector",
    "fast_only",
    "token_fast",
)
DEFAULT_BUDGETS = (0, 5, 10, 20, 30, 50, 100)
PRIMARY_VARIANT = "token_selector"
MATERIAL_EPSILON = 0.01
TOLERANCE = 1e-12


def uses_tokens(variant):
    return variant in ("token_only", "token_selector", "token_fast")


def uses_selector(variant):
    return variant in ("selector_only", "token_selector")


def uses_fast_output(variant):
    return variant in ("fast_only", "token_fast")


def empty_projection(count):
    return np.empty((int(count), 0), dtype=np.float32)


def _group_weights(sequences):
    sequences = np.asarray(sequences, dtype=str)
    counts = Counter(sequences.tolist())
    weights = np.asarray(
        [1.0 / counts[name] for name in sequences], dtype=np.float64
    )
    weights *= len(weights) / weights.sum()
    return weights


def grouped_folds(sequences, folds, seed):
    """Return deterministic video-disjoint train/test index pairs."""
    sequences = np.asarray(sequences, dtype=str)
    names = sorted(set(sequences.tolist()))
    if folds < 2 or folds > len(names):
        raise ValueError("--folds must be in [2, number of training videos]")
    random.Random(seed).shuffle(names)
    buckets = [[] for _ in range(folds)]
    bucket_sizes = [0] * folds
    counts = Counter(sequences.tolist())
    for name in sorted(names, key=lambda value: -counts[value]):
        target = min(range(folds), key=lambda index: (bucket_sizes[index], index))
        buckets[target].append(name)
        bucket_sizes[target] += counts[name]
    output = []
    for bucket in buckets:
        test_names = set(bucket)
        test = np.asarray([name in test_names for name in sequences])
        if not test.any() or test.all():
            raise RuntimeError("Grouped fold construction produced an empty split")
        output.append((np.flatnonzero(~test), np.flatnonzero(test), tuple(sorted(bucket))))
    return output


def _subset(data, indices):
    return {key: values[indices] for key, values in data.items()}


def load_immediate_dataset(directory, allow_incomplete=False):
    directory, run, records, missing, expected = load_records(
        directory, require_complete=not allow_incomplete
    )
    if 1 not in tuple(run["horizons"]):
        raise ValueError("Stage-8D requires the horizon-1 intervention label")
    horizon_index = tuple(run["horizons"]).index(1)
    feature_fields = (
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
    )
    chunks = {name: [] for name in feature_fields}
    chunks.update(
        {
            "split": [],
            "sequence": [],
            "delta_iou": [],
            "delta_auc": [],
            "control_iou": [],
            "treatment_iou": [],
        }
    )
    for record in records:
        entry = record["entry"]
        values = record["data"]
        valid = values["valid_gt"][:, 0]
        if not valid.any():
            continue
        for name in feature_fields:
            chunks[name].append(values[name][valid].copy())
        count = int(valid.sum())
        chunks["split"].append(np.full(count, entry["split"], dtype="U8"))
        chunks["sequence"].append(
            np.full(count, entry["sequence"], dtype="U128")
        )
        chunks["delta_iou"].append(values["delta_iou"][valid, 0].copy())
        chunks["delta_auc"].append(
            values["auc_delta"][valid, horizon_index].copy()
        )
        chunks["control_iou"].append(values["control_iou"][valid, 0].copy())
        chunks["treatment_iou"].append(
            values["treatment_iou"][valid, 0].copy()
        )
    if not chunks["sequence"]:
        raise ValueError("No valid immediate intervention anchors")
    data = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    data["previous_action"] = np.zeros(len(data["sequence"]), dtype=np.int64)
    for name in ("delta_iou", "delta_auc", "control_iou", "treatment_iou"):
        if not np.isfinite(data[name]).all():
            raise ValueError("Immediate label {} contains non-finite values".format(name))
    return directory, run, data, missing, expected


def build_variant_low_features(data, variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown immediate-gate feature variant: {}".format(variant))
    if uses_fast_output(variant):
        source = "fast_aware"
    elif uses_selector(variant):
        source = "token_selector"
    else:
        source = "token_only"
    return build_low_features(data, source)


def fit_scalar_ridge(features, targets, weights, alpha):
    raw = fit_ridge(
        features,
        np.asarray(targets, dtype=np.float64)[:, None],
        weights,
        alpha,
    )
    return {
        "coefficient": np.asarray(raw["coefficient"], dtype=np.float64)[:, 0],
        "intercept": float(np.asarray(raw["intercept"], dtype=np.float64)[0]),
    }


def _weighted_mae(targets, predictions, weights):
    return _weighted_mean(
        np.abs(np.asarray(targets) - np.asarray(predictions)), weights
    )


def _score_std(scores, weights):
    mean = _weighted_mean(scores, weights)
    return math.sqrt(
        max(_weighted_mean((np.asarray(scores) - mean) ** 2, weights), 0.0)
    )


def _sequence_effects(sequences, selected, values):
    sequences = np.asarray(sequences, dtype=str)
    selected = np.asarray(selected, dtype=bool)
    values = np.asarray(values, dtype=np.float64)
    output = {}
    for name in sorted(set(sequences.tolist())):
        mask = sequences == name
        output[name] = float(np.mean(np.where(selected[mask], values[mask], 0.0)))
    return output


def _bootstrap(values, samples, rng):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "probability_positive": float("nan"),
        }
    return bootstrap_mean(values, samples, rng)


def ranked_budget_rows(
    model_id,
    variant,
    scores,
    data,
    budgets,
    bootstrap_samples,
    seed,
    primary=False,
):
    """Evaluate frozen validation scores under exact physical frame budgets."""
    scores = np.asarray(scores, dtype=np.float64)
    delta = np.asarray(data["delta_iou"], dtype=np.float64)
    delta_auc = np.asarray(data["delta_auc"], dtype=np.float64)
    sequences = np.asarray(data["sequence"], dtype=str)
    order = np.argsort(-scores, kind="mergesort")
    oracle_order = np.argsort(-delta, kind="mergesort")
    positive_order = oracle_order[delta[oracle_order] > 0.0]
    all_selected = np.ones(len(delta), dtype=bool)
    all_effect = _sequence_effects(sequences, all_selected, delta)
    all_auc_effect = _sequence_effects(sequences, all_selected, delta_auc)
    positive = delta > 0.0
    positive_headroom = np.mean(
        list(_sequence_effects(sequences, positive, delta).values())
    )
    rng = np.random.default_rng(seed)
    rows = []
    for budget in budgets:
        maximum = (
            0
            if float(budget) <= 0.0
            else min(len(delta), int(math.ceil(len(delta) * float(budget) / 100.0)))
        )
        selected = np.zeros(len(delta), dtype=bool)
        selected[order[:maximum]] = True
        oracle = np.zeros(len(delta), dtype=bool)
        oracle[oracle_order[:maximum]] = True
        oracle_cap = np.zeros(len(delta), dtype=bool)
        oracle_cap[positive_order[:maximum]] = True
        names = sorted(set(sequences.tolist()))
        gain_by_sequence = _sequence_effects(sequences, selected, delta)
        auc_by_sequence = _sequence_effects(sequences, selected, delta_auc)
        oracle_by_sequence = _sequence_effects(sequences, oracle, delta)
        cap_by_sequence = _sequence_effects(sequences, oracle_cap, delta)
        ratio = float(selected.mean())
        gain_values = np.asarray([gain_by_sequence[name] for name in names])
        auc_values = np.asarray([auc_by_sequence[name] for name in names])
        random_values = np.asarray([ratio * all_effect[name] for name in names])
        random_auc_values = np.asarray(
            [ratio * all_auc_effect[name] for name in names]
        )
        oracle_values = np.asarray([oracle_by_sequence[name] for name in names])
        cap_values = np.asarray([cap_by_sequence[name] for name in names])
        lift_values = gain_values - random_values
        gain_bootstrap = _bootstrap(gain_values, bootstrap_samples, rng)
        lift_bootstrap = _bootstrap(lift_values, bootstrap_samples, rng)
        chosen = delta[selected]
        if maximum == 0:
            threshold = float("inf")
        elif maximum == len(scores):
            threshold = float("-inf")
        else:
            threshold = 0.5 * (
                float(scores[order[maximum - 1]]) + float(scores[order[maximum]])
            )
        oracle_cap_gain = float(cap_values.mean())
        rows.append(
            {
                "model_id": model_id,
                "variant": variant,
                "is_primary": int(primary),
                "budget_policy": "frozen_score_exact_topk_offline_diagnostic",
                "budget_percent": float(budget),
                "num_sequences": len(names),
                "num_anchors": len(delta),
                "max_selected": maximum,
                "num_selected": int(selected.sum()),
                "selected_ratio": ratio,
                "score_threshold": _finite_or_blank(threshold),
                "mean_iou_gain": float(gain_values.mean()),
                "gain_ci_low": gain_bootstrap["ci_low"],
                "gain_ci_high": gain_bootstrap["ci_high"],
                "gain_probability_positive": gain_bootstrap[
                    "probability_positive"
                ],
                "auc_gain": float(auc_values.mean()),
                "random_matched_mean_iou_gain": float(random_values.mean()),
                "random_matched_auc_gain": float(random_auc_values.mean()),
                "lift_over_random_mean_iou": float(lift_values.mean()),
                "lift_ci_low": lift_bootstrap["ci_low"],
                "lift_ci_high": lift_bootstrap["ci_high"],
                "lift_probability_positive": lift_bootstrap[
                    "probability_positive"
                ],
                "oracle_exact_mean_iou_gain": float(oracle_values.mean()),
                "oracle_positive_cap_mean_iou_gain": oracle_cap_gain,
                "oracle_cap_recovery": (
                    float(gain_values.mean() / oracle_cap_gain)
                    if oracle_cap_gain > TOLERANCE
                    else float("nan")
                ),
                "positive_headroom_mean_iou_gain": float(positive_headroom),
                "positive_headroom_recovery": (
                    float(gain_values.mean() / positive_headroom)
                    if positive_headroom > TOLERANCE
                    else float("nan")
                ),
                "selected_delta_mean": (
                    float(chosen.mean()) if chosen.size else float("nan")
                ),
                "selected_positive_ratio": (
                    float(np.mean(chosen > 0.0)) if chosen.size else float("nan")
                ),
                "selected_negative_ratio": (
                    float(np.mean(chosen < 0.0)) if chosen.size else float("nan")
                ),
                "selected_material_positive_ratio_001": (
                    float(np.mean(chosen > MATERIAL_EPSILON))
                    if chosen.size
                    else float("nan")
                ),
                "selected_material_harm_ratio_001": (
                    float(np.mean(chosen < -MATERIAL_EPSILON))
                    if chosen.size
                    else float("nan")
                ),
            }
        )
    return rows


def save_model(
    path,
    variant,
    run,
    pca,
    preprocessor,
    ridge,
    low_names,
    configuration,
    token_dimensions,
    primary,
):
    metadata = {
        "schema": MODEL_SCHEMA,
        "variant": variant,
        "is_primary": bool(primary),
        "cache_fingerprint": run["fingerprint"],
        "manifest_fingerprint": run["manifest"]["fingerprint"],
        "target": "single_l9_immediate_delta_iou_h1",
        "requires_token_pool": uses_tokens(variant),
        "requires_selector": uses_selector(variant),
        "requires_fast_head": uses_fast_output(variant),
        "configuration": configuration,
    }
    metadata["fingerprint"] = fingerprint(metadata)
    if pca is None:
        token_mean = np.empty(0, dtype=np.float32)
        token_scale = np.empty(0, dtype=np.float32)
        components = np.empty((0, int(token_dimensions)), dtype=np.float32)
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
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument(
        "--primary_variant", choices=VARIANTS, default=PRIMARY_VARIANT
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--pca_dims", type=int, default=64)
    parser.add_argument("--pca_oversample", type=int, default=12)
    parser.add_argument("--pca_power_iterations", type=int, default=1)
    parser.add_argument(
        "--ridge_alphas",
        nargs="+",
        type=float,
        default=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    )
    parser.add_argument(
        "--budgets", nargs="+", type=float, default=list(DEFAULT_BUDGETS)
    )
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2032)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--allow_incomplete", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if len(set(args.variants)) != len(args.variants):
        raise ValueError("Variants must be unique")
    if args.primary_variant not in args.variants:
        raise ValueError("--primary_variant must be included in --variants")
    if (
        args.folds < 2
        or args.pca_dims < 1
        or args.pca_oversample < 0
        or args.pca_power_iterations < 0
        or args.batch_size < 1
        or args.bootstrap_samples < 1
    ):
        raise ValueError("Invalid folds, PCA, batch-size or bootstrap argument")
    if not args.ridge_alphas or any(alpha <= 0.0 for alpha in args.ridge_alphas):
        raise ValueError("All ridge alphas must be positive")
    if not args.budgets or any(value < 0.0 or value > 100.0 for value in args.budgets):
        raise ValueError("Budgets must lie in [0,100]")

    input_dir, run, data, missing, expected = load_immediate_dataset(
        args.input, args.allow_incomplete
    )
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

    train = _subset(data, np.flatnonzero(data["split"] == "train"))
    val = _subset(data, np.flatnonzero(data["split"] == "val"))
    if not len(train["sequence"]) or not len(val["sequence"]):
        raise ValueError("Both manifest train and validation anchors are required")
    train_names = sorted(set(train["sequence"].tolist()))
    val_names = sorted(set(val["sequence"].tolist()))
    if set(train_names).intersection(val_names):
        raise ValueError("Train and validation videos overlap")
    folds = grouped_folds(train["sequence"], args.folds, args.seed)
    train_weights = _group_weights(train["sequence"])
    val_weights = _group_weights(val["sequence"])
    low_features = {}
    low_names = {}
    for variant in args.variants:
        low_features[variant], low_names[variant] = build_variant_low_features(
            train, variant
        )
    val_low_features = {
        variant: build_variant_low_features(val, variant)[0]
        for variant in args.variants
    }

    print(
        "Audited {} of {} clips ({} missing); loaded {} train / {} val anchors "
        "from {} / {} videos.".format(
            expected - missing,
            expected,
            missing,
            len(train["sequence"]),
            len(val["sequence"]),
            len(train_names),
            len(val_names),
        )
    )
    print(
        "Target: signed immediate IoU delta; grouped {}-fold CV; primary={}.".format(
            args.folds, args.primary_variant
        )
    )

    oof = {
        variant: {
            float(alpha): np.full(len(train["sequence"]), np.nan, dtype=np.float64)
            for alpha in args.ridge_alphas
        }
        for variant in args.variants
    }
    constant_oof = np.full(len(train["sequence"]), np.nan, dtype=np.float64)
    fold_rows = []
    target = np.asarray(train["delta_iou"], dtype=np.float64)

    for fold_index, (fit_index, test_index, test_names) in enumerate(folds):
        fit_weights = _group_weights(train["sequence"][fit_index])
        test_weights = _group_weights(train["sequence"][test_index])
        constant_oof[test_index] = _weighted_mean(target[fit_index], fit_weights)
        fold_pca = None
        if any(uses_tokens(variant) for variant in args.variants):
            fold_pca = fit_token_pca(
                train["token_features"][fit_index],
                args.pca_dims,
                args.pca_oversample,
                args.pca_power_iterations,
                args.seed + fold_index,
            )
            fit_token = transform_token_pca(
                train["token_features"][fit_index], fold_pca, args.batch_size
            )
            test_token = transform_token_pca(
                train["token_features"][test_index], fold_pca, args.batch_size
            )
        else:
            fit_token = empty_projection(len(fit_index))
            test_token = empty_projection(len(test_index))

        for variant in args.variants:
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
            for alpha in args.ridge_alphas:
                alpha = float(alpha)
                ridge = fit_scalar_ridge(
                    fit_features, target[fit_index], fit_weights, alpha
                )
                prediction = predict_ridge(test_features, ridge)
                oof[variant][alpha][test_index] = prediction
                fold_rows.append(
                    {
                        "fold": fold_index,
                        "variant": variant,
                        "ridge_alpha": alpha,
                        "train_sequences": len(
                            set(train["sequence"][fit_index].tolist())
                        ),
                        "test_sequences": len(test_names),
                        "train_examples": len(fit_index),
                        "test_examples": len(test_index),
                        "test_weighted_mse": _weighted_mse(
                            target[test_index, None], prediction[:, None], test_weights
                        ),
                        "test_weighted_mae": _weighted_mae(
                            target[test_index], prediction, test_weights
                        ),
                        "test_pearson": _weighted_pearson(
                            prediction, target[test_index], test_weights
                        ),
                    }
                )
        print(
            "[fold {}/{}] train={} videos, test={} videos".format(
                fold_index + 1,
                len(folds),
                len(set(train["sequence"][fit_index].tolist())),
                len(test_names),
            ),
            flush=True,
        )

    if not np.isfinite(constant_oof).all():
        raise RuntimeError("OOF constant predictions are incomplete")
    constant_mse = _weighted_mse(
        target[:, None], constant_oof[:, None], train_weights
    )
    selected_alpha = {}
    alpha_rows = []
    for variant in args.variants:
        best = None
        for alpha in args.ridge_alphas:
            alpha = float(alpha)
            prediction = oof[variant][alpha]
            if not np.isfinite(prediction).all():
                raise RuntimeError("OOF predictions are incomplete")
            loss = _weighted_mse(target[:, None], prediction[:, None], train_weights)
            row = {
                "variant": variant,
                "ridge_alpha": alpha,
                "oof_weighted_mse": loss,
                "oof_weighted_mae": _weighted_mae(target, prediction, train_weights),
                "oof_pearson": _weighted_pearson(prediction, target, train_weights),
                "constant_oof_weighted_mse": constant_mse,
                "selected": 0,
            }
            alpha_rows.append(row)
            if best is None or loss < best[0] - 1e-15:
                best = (loss, alpha)
        selected_alpha[variant] = best[1]
        for row in alpha_rows:
            if row["variant"] == variant and row["ridge_alpha"] == best[1]:
                row["selected"] = 1
        print(
            "[CV] {:<16s} alpha={:<7g} mse={:.8f} constant={:.8f}".format(
                variant, best[1], best[0], constant_mse
            )
        )

    final_pca = None
    if any(uses_tokens(variant) for variant in args.variants):
        print("Refitting PCA and frozen gates on all training videos...", flush=True)
        final_pca = fit_token_pca(
            train["token_features"],
            args.pca_dims,
            args.pca_oversample,
            args.pca_power_iterations,
            args.seed + args.folds,
        )
        train_token = transform_token_pca(
            train["token_features"], final_pca, args.batch_size
        )
        val_token = transform_token_pca(
            val["token_features"], final_pca, args.batch_size
        )
    else:
        train_token = empty_projection(len(train["sequence"]))
        val_token = empty_projection(len(val["sequence"]))

    summary_rows = []
    budget_rows = []
    prediction_rows = []
    model_reports = []
    val_target = np.asarray(val["delta_iou"], dtype=np.float64)
    configuration = {
        "target": "single_l9_immediate_delta_iou_h1",
        "primary_variant": args.primary_variant,
        "folds": args.folds,
        "pca_dims": args.pca_dims,
        "pca_oversample": args.pca_oversample,
        "pca_power_iterations": args.pca_power_iterations,
        "ridge_alphas": [float(value) for value in args.ridge_alphas],
        "budgets": [float(value) for value in args.budgets],
        "seed": args.seed,
        "anchor_fraction_is_feature": False,
    }
    for variant_index, variant in enumerate(args.variants):
        if uses_tokens(variant):
            train_projection, val_projection = train_token, val_token
            variant_pca = final_pca
        else:
            train_projection = empty_projection(len(train["sequence"]))
            val_projection = empty_projection(len(val["sequence"]))
            variant_pca = None
        preprocessor = fit_combined_preprocessor(
            train_projection, low_features[variant], train_weights
        )
        train_features = transform_combined(
            train_projection, low_features[variant], preprocessor
        )
        val_features = transform_combined(
            val_projection, val_low_features[variant], preprocessor
        )
        ridge = fit_scalar_ridge(
            train_features,
            target,
            train_weights,
            selected_alpha[variant],
        )
        scores = np.asarray(predict_ridge(val_features, ridge), dtype=np.float64)
        oof_scores = oof[variant][selected_alpha[variant]]
        primary = variant == args.primary_variant
        summary_rows.append(
            {
                "variant": variant,
                "is_primary": int(primary),
                "train_examples": len(target),
                "val_examples": len(val_target),
                "train_sequences": len(train_names),
                "val_sequences": len(val_names),
                "feature_dims": int(train_features.shape[1]),
                "low_feature_dims": int(low_features[variant].shape[1]),
                "pca_dims": int(train_projection.shape[1]),
                "pca_explained_variance_ratio": (
                    variant_pca["explained_variance_ratio"]
                    if variant_pca is not None
                    else ""
                ),
                "requires_token_pool": int(uses_tokens(variant)),
                "requires_selector": int(uses_selector(variant)),
                "requires_fast_head": int(uses_fast_output(variant)),
                "ridge_alpha": selected_alpha[variant],
                "constant_oof_weighted_mse": constant_mse,
                "oof_weighted_mse": _weighted_mse(
                    target[:, None], oof_scores[:, None], train_weights
                ),
                "oof_weighted_mae": _weighted_mae(
                    target, oof_scores, train_weights
                ),
                "oof_pearson": _weighted_pearson(
                    oof_scores, target, train_weights
                ),
                "oof_benefit_auroc": _weighted_binary_auc(
                    target > 0.0, oof_scores, train_weights
                ),
                "val_weighted_mse": _weighted_mse(
                    val_target[:, None], scores[:, None], val_weights
                ),
                "val_weighted_mae": _weighted_mae(
                    val_target, scores, val_weights
                ),
                "val_pearson": _weighted_pearson(
                    scores, val_target, val_weights
                ),
                "val_benefit_auroc": _weighted_binary_auc(
                    val_target > 0.0, scores, val_weights
                ),
                "val_material_benefit_auroc_001": _weighted_binary_auc(
                    val_target > MATERIAL_EPSILON, scores, val_weights
                ),
                "val_material_harm_auroc_001": _weighted_binary_auc(
                    val_target < -MATERIAL_EPSILON, -scores, val_weights
                ),
                "val_score_mean": _weighted_mean(scores, val_weights),
                "val_score_std": _score_std(scores, val_weights),
                "val_always_l9_mean_iou_gain": _weighted_mean(
                    val_target, val_weights
                ),
                "val_positive_headroom_mean_iou_gain": _weighted_mean(
                    np.maximum(val_target, 0.0), val_weights
                ),
            }
        )
        budget_rows.extend(
            ranked_budget_rows(
                "stage8_{}".format(variant),
                variant,
                scores,
                val,
                tuple(sorted(set(float(value) for value in args.budgets))),
                args.bootstrap_samples,
                args.seed + 100 + variant_index,
                primary,
            )
        )
        for index in range(len(val_target)):
            prediction_rows.append(
                {
                    "variant": variant,
                    "is_primary": int(primary),
                    "split": "val",
                    "sequence": str(val["sequence"][index]),
                    "frame_id": int(val["frame_id"][index]),
                    "anchor_offset": int(val["anchor_offset"][index]),
                    "score": float(scores[index]),
                    "delta_iou_h1": float(val_target[index]),
                    "delta_auc_h1": float(val["delta_auc"][index]),
                    "beneficial": int(val_target[index] > 0.0),
                    "material_beneficial_001": int(
                        val_target[index] > MATERIAL_EPSILON
                    ),
                    "material_harmful_001": int(
                        val_target[index] < -MATERIAL_EPSILON
                    ),
                }
            )
        model_configuration = dict(configuration)
        model_configuration["ridge_alpha"] = selected_alpha[variant]
        model_path = output_dir / "stage8_immediate_gate_{}.npz".format(variant)
        report = save_model(
            model_path,
            variant,
            run,
            variant_pca,
            preprocessor,
            ridge,
            low_names[variant],
            model_configuration,
            train["token_features"].shape[1],
            primary,
        )
        report["path"] = str(model_path)
        model_reports.append(report)

    paths = {
        "summary": output_dir / "stage8_immediate_gate_summary.csv",
        "budget": output_dir / "stage8_immediate_gate_budget.csv",
        "alpha_selection": output_dir / "stage8_immediate_gate_alpha_selection.csv",
        "folds": output_dir / "stage8_immediate_gate_folds.csv",
        "predictions": output_dir / "stage8_immediate_gate_val_predictions.csv",
    }
    for path, rows in (
        (paths["summary"], summary_rows),
        (paths["budget"], budget_rows),
        (paths["alpha_selection"], alpha_rows),
        (paths["folds"], fold_rows),
        (paths["predictions"], prediction_rows),
    ):
        write_csv(path, rows)
    experiment = {
        "schema": "stage8_immediate_gate_experiment_v1",
        "cache_fingerprint": run["fingerprint"],
        "manifest_fingerprint": run["manifest"]["fingerprint"],
        "configuration": configuration,
        "selected_alphas": selected_alpha,
        "train_sequences": train_names,
        "validation_sequences": val_names,
        "script_sha256": file_hash(Path(__file__).resolve()),
        "models": model_reports,
        "outputs": {key: str(value) for key, value in paths.items()},
        "warning": (
            "Validation labels are used only for frozen evaluation. Exact top-k "
            "uses the complete validation score horizon and is an offline budget "
            "ranking diagnostic, not yet an online or closed-loop policy."
        ),
    }
    experiment["fingerprint"] = fingerprint(experiment)
    write_json(output_dir / "stage8_immediate_gate_experiment.json", experiment)

    print("\nFrozen validation diagnostics:")
    print("variant          primary  alpha       AUROC   pearson  score_std")
    for row in summary_rows:
        print(
            "{variant:<16s} {is_primary:>7d}  {ridge_alpha:>7g}  "
            "{val_benefit_auroc:>8.4f} {val_pearson:>9.4f} "
            "{val_score_std:>10.6f}".format(**row)
        )
    print("\nPrimary frozen-score budget ranking:")
    print(
        "budget  selected  iou_gain  random  lift_95%              oracle  recovery  "
        "benefit  harm"
    )
    for row in budget_rows:
        if not row["is_primary"]:
            continue
        interval = "[{:.6f},{:.6f}]".format(row["lift_ci_low"], row["lift_ci_high"])
        recovery = row["oracle_cap_recovery"]
        recovery_text = "n/a" if not np.isfinite(recovery) else "{:.2%}".format(recovery)
        benefit = row["selected_positive_ratio"]
        harm = row["selected_negative_ratio"]
        print(
            "{budget_percent:>6.1f}% {selected_ratio:>8.2%} {mean_iou_gain:>9.6f} "
            "{random_matched_mean_iou_gain:>7.6f} {interval:<21s} "
            "{oracle_positive_cap_mean_iou_gain:>7.6f} {recovery_text:>9s} "
            "{benefit_text:>8s} {harm_text:>7s}".format(
                interval=interval,
                recovery_text=recovery_text,
                benefit_text=("n/a" if not np.isfinite(benefit) else "{:.2%}".format(benefit)),
                harm_text=("n/a" if not np.isfinite(harm) else "{:.2%}".format(harm)),
                **row
            )
        )
    print("Reports and deployable Ridge models: {}".format(output_dir))
    print("Primary files: {} and {}".format(paths["summary"], paths["budget"]))


if __name__ == "__main__":
    main()
