"""Screen causal trajectory-history features for immediate L9 routing.

This Stage-8E diagnostic reads only the manifest training split.  Every score
is produced by nested video-grouped cross-validation: inner folds choose the
Ridge alpha and outer folds estimate generalization to unseen videos.  The
existing intervention validation split is deliberately not loaded.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint, write_csv, write_json
from stage4_router_train import (
    _weighted_binary_auc,
    _weighted_mean,
    _weighted_mse,
    _weighted_pearson,
    build_low_features,
    fit_combined_preprocessor,
    fit_token_pca,
    predict_ridge,
    transform_combined,
    transform_token_pca,
)
from stage8_immediate_gate import (
    _group_weights,
    _score_std,
    _weighted_mae,
    empty_projection,
    fit_scalar_ridge,
    grouped_folds,
    ranked_budget_rows,
)
from stage8_intervention_common import (
    intervention_cache_path,
    load_and_audit_cache,
    load_intervention_run,
)


SCHEMA = "stage8_motion_gate_nested_oof_v1"
VARIANTS = (
    "history_only",
    "token_selector",
    "motion_only",
    "motion_selector",
    "token_motion",
    "token_selector_motion",
)
PRIMARY_VARIANT = "token_selector_motion"
MOTION_LAGS = (1, 2, 4, 8, 16)
MOTION_WINDOWS = (4, 8, 16)
MATERIAL_EPSILON = 0.01
SCREEN_AUROC = 0.55
SCREEN_PEARSON = 0.10
SCREEN_BUDGETS = (10.0, 20.0, 30.0)


def uses_tokens(variant):
    return variant in (
        "token_selector",
        "token_motion",
        "token_selector_motion",
    )


def uses_selector(variant):
    return variant in (
        "token_selector",
        "motion_selector",
        "token_selector_motion",
    )


def uses_motion(variant):
    return variant in (
        "motion_only",
        "motion_selector",
        "token_motion",
        "token_selector_motion",
    )


def motion_feature_names():
    names = []
    for lag in MOTION_LAGS:
        names.extend(
            "lag{}_{}".format(lag, value)
            for value in ("available", "dx", "dy", "log_w", "log_h")
        )
    statistics = (
        "coverage",
        "mean_dx",
        "mean_dy",
        "speed_mean",
        "speed_std",
        "speed_max",
        "acceleration_mean",
        "acceleration_max",
        "path_efficiency",
        "log_area_mean",
        "log_area_std",
        "log_area_max_abs",
        "log_aspect_mean",
        "log_aspect_std",
        "log_aspect_max_abs",
    )
    for window in MOTION_WINDOWS:
        names.extend(
            "window{}_{}".format(window, value) for value in statistics
        )
    return tuple(names)


MOTION_FEATURE_NAMES = motion_feature_names()


def _box_centers(boxes):
    boxes = np.asarray(boxes, dtype=np.float64)
    return boxes[..., :2] + 0.5 * boxes[..., 2:]


def causal_motion_features(baseline_boxes, anchor_offsets, initial_box):
    """Describe only predicted states available before each anchor decision."""
    baseline_boxes = np.asarray(baseline_boxes, dtype=np.float64)
    anchor_offsets = np.asarray(anchor_offsets, dtype=np.int64)
    initial_box = np.asarray(initial_box, dtype=np.float64)
    if (
        baseline_boxes.ndim != 2
        or baseline_boxes.shape[1] != 4
        or initial_box.shape != (4,)
        or np.any(baseline_boxes[:, 2:] <= 0.0)
        or np.any(initial_box[2:] <= 0.0)
    ):
        raise ValueError("Motion history requires valid xywh tracker states")
    if np.any(anchor_offsets < 1) or np.any(anchor_offsets > len(baseline_boxes)):
        raise ValueError("Anchor offsets leave the tracked Fast trajectory")

    states = np.concatenate((initial_box[None], baseline_boxes), axis=0)
    output = np.zeros(
        (len(anchor_offsets), len(MOTION_FEATURE_NAMES)), dtype=np.float64
    )
    epsilon = 1e-8
    for row, offset in enumerate(anchor_offsets):
        # Offset 1 predicts the first tracked frame from the initialization.
        current_index = int(offset) - 1
        current = states[current_index]
        current_center = _box_centers(current)
        values = []
        for lag in MOTION_LAGS:
            available = current_index >= lag
            if available:
                reference = states[current_index - lag]
                reference_center = _box_centers(reference)
                size = np.maximum(reference[2:], epsilon)
                displacement = (current_center - reference_center) / size
                log_size = np.log(np.maximum(current[2:], epsilon) / size)
                values.extend(
                    (1.0, displacement[0], displacement[1], log_size[0], log_size[1])
                )
            else:
                values.extend((0.0, 0.0, 0.0, 0.0, 0.0))

        for window in MOTION_WINDOWS:
            start = max(0, current_index - window)
            history = states[start : current_index + 1]
            steps = len(history) - 1
            if steps < 1:
                values.extend((0.0,) * 15)
                continue
            previous = history[:-1]
            following = history[1:]
            previous_size = np.maximum(previous[:, 2:], epsilon)
            displacement = (
                _box_centers(following) - _box_centers(previous)
            ) / previous_size
            speed = np.linalg.norm(displacement, axis=1)
            acceleration = (
                np.linalg.norm(np.diff(displacement, axis=0), axis=1)
                if steps > 1
                else np.zeros(1, dtype=np.float64)
            )
            log_size = np.log(
                np.maximum(following[:, 2:], epsilon) / previous_size
            )
            log_area = log_size[:, 0] + log_size[:, 1]
            log_aspect = log_size[:, 0] - log_size[:, 1]
            path = float(speed.sum())
            efficiency = float(
                np.linalg.norm(displacement.sum(axis=0)) / max(path, epsilon)
            )
            values.extend(
                (
                    steps / float(window),
                    float(displacement[:, 0].mean()),
                    float(displacement[:, 1].mean()),
                    float(speed.mean()),
                    float(speed.std()),
                    float(speed.max()),
                    float(acceleration.mean()),
                    float(acceleration.max()),
                    efficiency,
                    float(log_area.mean()),
                    float(log_area.std()),
                    float(np.abs(log_area).max()),
                    float(log_aspect.mean()),
                    float(log_aspect.std()),
                    float(np.abs(log_aspect).max()),
                )
            )
        output[row] = values
    return np.clip(
        np.nan_to_num(output, nan=0.0, posinf=20.0, neginf=-20.0),
        -20.0,
        20.0,
    ).astype(np.float32)


def load_training_dataset(directory, allow_incomplete=False):
    """Audit and load training caches without opening validation cache files."""
    directory, run = load_intervention_run(directory)
    entries = [
        entry for entry in run["manifest"]["sequences"] if entry["split"] == "train"
    ]
    if not entries:
        raise ValueError("Intervention manifest has no training videos")
    if 1 not in tuple(run["horizons"]):
        raise ValueError("Stage-8E requires horizon-1 intervention labels")
    horizon_index = tuple(run["horizons"]).index(1)
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
    )
    chunks = {name: [] for name in fields}
    chunks.update(
        {
            "sequence": [],
            "motion_history": [],
            "delta_iou": [],
            "delta_auc": [],
            "control_iou": [],
            "treatment_iou": [],
        }
    )
    missing = []
    for entry in entries:
        path = intervention_cache_path(directory, entry)
        if not path.is_file():
            missing.append(path)
            continue
        _, values, metadata = load_and_audit_cache(path, entry, run)
        motion = causal_motion_features(
            values["baseline_boxes"],
            values["anchor_offset"],
            metadata["init_box"],
        )
        expected_previous = np.concatenate(
            (
                np.asarray(metadata["init_box"], dtype=np.float64)[None],
                values["baseline_boxes"][:-1],
            ),
            axis=0,
        )[values["anchor_offset"] - 1]
        if not np.allclose(
            values["previous_state"], expected_previous, rtol=0, atol=1e-9
        ):
            raise ValueError("Motion history is misaligned in {}".format(path))
        valid = values["valid_gt"][:, 0]
        if not valid.any():
            continue
        for name in fields:
            chunks[name].append(values[name][valid].copy())
        chunks["motion_history"].append(motion[valid])
        count = int(valid.sum())
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
    if missing and not allow_incomplete:
        raise ValueError(
            "Training intervention cache is incomplete: {} of {} files missing".format(
                len(missing), len(entries)
            )
        )
    if not chunks["sequence"]:
        raise ValueError("No valid Stage-8E training anchors")
    data = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    data["previous_action"] = np.zeros(len(data["sequence"]), dtype=np.int64)
    if not np.isfinite(data["motion_history"]).all():
        raise ValueError("Motion-history features contain non-finite values")
    return directory, run, data, len(entries) - len(missing), len(entries)


def build_variant_low_features(data, variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown motion-gate feature variant: {}".format(variant))
    source = "token_selector" if uses_selector(variant) else "token_only"
    base, names = build_low_features(data, source)
    if uses_motion(variant):
        base = np.concatenate((base, data["motion_history"]), axis=1)
        names = tuple(names) + MOTION_FEATURE_NAMES
    return base.astype(np.float32, copy=False), tuple(names)


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
        fit_token = transform_token_pca(
            data["token_features"][fit_index], pca, batch_size
        )
        test_token = transform_token_pca(
            data["token_features"][test_index], pca, batch_size
        )
    else:
        fit_token = empty_projection(len(fit_index))
        test_token = empty_projection(len(test_index))
    target = data["delta_iou"]
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
            model = fit_scalar_ridge(
                fit_features,
                target[fit_index],
                fit_weights,
                float(alpha),
            )
            predictions[variant][float(alpha)] = np.asarray(
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
    sequences = data["sequence"]
    target = np.asarray(data["delta_iou"], dtype=np.float64)
    weights = _group_weights(sequences)
    low_features = {
        variant: build_variant_low_features(data, variant)[0]
        for variant in variants
    }
    outer = grouped_folds(sequences, outer_folds, seed)
    oof = {
        variant: np.full(len(target), np.nan, dtype=np.float64)
        for variant in variants
    }
    constant_oof = np.full(len(target), np.nan, dtype=np.float64)
    outer_rows = []
    alpha_rows = []
    selected_alphas = {variant: [] for variant in variants}

    for outer_index, (outer_train, outer_test, outer_names) in enumerate(outer):
        outer_train_weights = _group_weights(sequences[outer_train])
        constant_oof[outer_test] = _weighted_mean(
            target[outer_train], outer_train_weights
        )
        inner = grouped_folds(
            sequences[outer_train], inner_folds, seed + 1000 + outer_index
        )
        inner_predictions = {
            variant: {
                alpha: np.full(len(outer_train), np.nan, dtype=np.float64)
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
                    inner_predictions[variant][alpha][inner_test] = predicted[
                        variant
                    ][alpha]

        chosen = {}
        for variant in variants:
            best = None
            for alpha in alphas:
                prediction = inner_predictions[variant][alpha]
                if not np.isfinite(prediction).all():
                    raise RuntimeError("Nested inner OOF predictions are incomplete")
                loss = _weighted_mse(
                    target[outer_train, None],
                    prediction[:, None],
                    outer_train_weights,
                )
                alpha_rows.append(
                    {
                        "outer_fold": outer_index,
                        "variant": variant,
                        "ridge_alpha": alpha,
                        "inner_oof_weighted_mse": loss,
                        "inner_oof_weighted_mae": _weighted_mae(
                            target[outer_train], prediction, outer_train_weights
                        ),
                        "inner_oof_pearson": _weighted_pearson(
                            prediction, target[outer_train], outer_train_weights
                        ),
                        "selected": 0,
                    }
                )
                if best is None or loss < best[0] - 1e-15:
                    best = (loss, alpha)
            chosen[variant] = best[1]
            selected_alphas[variant].append(best[1])
            for row in alpha_rows:
                if (
                    row["outer_fold"] == outer_index
                    and row["variant"] == variant
                    and row["ridge_alpha"] == best[1]
                ):
                    row["selected"] = 1

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
            outer_rows.append(
                {
                    "outer_fold": outer_index,
                    "variant": variant,
                    "selected_ridge_alpha": chosen[variant],
                    "train_sequences": len(
                        set(sequences[outer_train].tolist())
                    ),
                    "test_sequences": len(outer_names),
                    "train_examples": len(outer_train),
                    "test_examples": len(outer_test),
                    "test_weighted_mse": _weighted_mse(
                        target[outer_test, None], prediction[:, None], test_weights
                    ),
                    "test_weighted_mae": _weighted_mae(
                        target[outer_test], prediction, test_weights
                    ),
                    "test_pearson": _weighted_pearson(
                        prediction, target[outer_test], test_weights
                    ),
                    "test_benefit_auroc": _weighted_binary_auc(
                        target[outer_test] > 0.0, prediction, test_weights
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
                    "{}:{:g}".format(variant, chosen[variant])
                    for variant in variants
                ),
            ),
            flush=True,
        )

    if not np.isfinite(constant_oof).all() or any(
        not np.isfinite(values).all() for values in oof.values()
    ):
        raise RuntimeError("Nested outer OOF predictions are incomplete")
    return {
        "predictions": oof,
        "constant_predictions": constant_oof,
        "outer_rows": outer_rows,
        "alpha_rows": alpha_rows,
        "selected_alphas": selected_alphas,
        "low_features": low_features,
        "weights": weights,
    }


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
    parser.add_argument("--outer_folds", type=int, default=5)
    parser.add_argument("--inner_folds", type=int, default=4)
    parser.add_argument("--pca_dims", type=int, default=64)
    parser.add_argument("--pca_oversample", type=int, default=12)
    parser.add_argument("--pca_power_iterations", type=int, default=1)
    parser.add_argument(
        "--ridge_alphas",
        nargs="+",
        type=float,
        default=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    )
    parser.add_argument(
        "--budgets", nargs="+", type=float, default=[0, 5, 10, 20, 30, 50, 100]
    )
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2033)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--allow_incomplete", action="store_true")
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
        or args.batch_size < 1
        or args.bootstrap_samples < 1
    ):
        raise ValueError("Invalid fold, PCA, batch-size or bootstrap argument")
    if not args.ridge_alphas or any(value <= 0.0 for value in args.ridge_alphas):
        raise ValueError("Ridge alphas must be positive")
    if len(set(args.ridge_alphas)) != len(args.ridge_alphas):
        raise ValueError("Ridge alphas must be unique")
    if not args.budgets or any(value < 0.0 or value > 100.0 for value in args.budgets):
        raise ValueError("Budgets must lie in [0,100]")

    input_dir, run, data, completed, expected = load_training_dataset(
        args.input, args.allow_incomplete
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    try:
        output_dir.relative_to(input_dir)
    except ValueError:
        pass
    else:
        raise ValueError("--output_dir must be outside the intervention cache")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError("Use a new --output_dir or pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = sorted(set(data["sequence"].tolist()))
    if args.outer_folds > len(sequences):
        raise ValueError("Outer folds exceed the number of training videos")
    smallest_outer_train = len(sequences) - int(math.ceil(len(sequences) / args.outer_folds))
    if args.inner_folds > smallest_outer_train:
        raise ValueError("Inner folds exceed an outer training split")
    alphas = tuple(sorted(float(value) for value in args.ridge_alphas))
    print(
        "Audited {} of {} training clips; loaded {} anchors from {} videos.".format(
            completed, expected, len(data["sequence"]), len(sequences)
        )
    )
    print(
        "Existing validation split is not loaded. Primary={}; nested {}/{}-fold CV.".format(
            args.primary_variant, args.outer_folds, args.inner_folds
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

    target = np.asarray(data["delta_iou"], dtype=np.float64)
    weights = result["weights"]
    constant_mse = _weighted_mse(
        target[:, None], result["constant_predictions"][:, None], weights
    )
    summary_rows = []
    budget_rows = []
    prediction_rows = []
    budgets = tuple(sorted(set(float(value) for value in args.budgets)))
    for variant_index, variant in enumerate(args.variants):
        scores = result["predictions"][variant]
        primary = variant == args.primary_variant
        rows = ranked_budget_rows(
            "stage8e_{}".format(variant),
            variant,
            scores,
            data,
            budgets,
            args.bootstrap_samples,
            args.seed + 100 + variant_index,
            primary,
        )
        for row in rows:
            row["budget_policy"] = "nested_oof_score_exact_topk_training_diagnostic"
        budget_rows.extend(rows)
        screening = [
            row
            for row in rows
            if row["budget_percent"] in SCREEN_BUDGETS
        ]
        positive_lift_budgets = sum(row["lift_ci_low"] > 0.0 for row in screening)
        benefit_auroc = _weighted_binary_auc(target > 0.0, scores, weights)
        pearson = _weighted_pearson(scores, target, weights)
        passes = (
            primary
            and benefit_auroc >= SCREEN_AUROC
            and pearson >= SCREEN_PEARSON
            and positive_lift_budgets >= 1
        )
        summary_rows.append(
            {
                "variant": variant,
                "is_primary": int(primary),
                "num_examples": len(target),
                "num_sequences": len(sequences),
                "feature_dims_without_pca": int(
                    result["low_features"][variant].shape[1]
                ),
                "pca_dims": args.pca_dims if uses_tokens(variant) else 0,
                "requires_token_pool": int(uses_tokens(variant)),
                "requires_selector": int(uses_selector(variant)),
                "uses_motion_history": int(uses_motion(variant)),
                "selected_alpha_by_outer_fold": "|".join(
                    "{:g}".format(value)
                    for value in result["selected_alphas"][variant]
                ),
                "constant_nested_oof_mse": constant_mse,
                "nested_oof_mse": _weighted_mse(
                    target[:, None], scores[:, None], weights
                ),
                "nested_oof_mae": _weighted_mae(target, scores, weights),
                "nested_oof_pearson": pearson,
                "nested_oof_benefit_auroc": benefit_auroc,
                "nested_oof_material_benefit_auroc_001": _weighted_binary_auc(
                    target > MATERIAL_EPSILON, scores, weights
                ),
                "nested_oof_material_harm_auroc_001": _weighted_binary_auc(
                    target < -MATERIAL_EPSILON, -scores, weights
                ),
                "score_mean": _weighted_mean(scores, weights),
                "score_std": _score_std(scores, weights),
                "positive_lift_budgets_10_20_30": positive_lift_budgets,
                "screen_auroc_threshold": SCREEN_AUROC,
                "screen_pearson_threshold": SCREEN_PEARSON,
                "passes_predeclared_primary_screen": int(passes),
            }
        )
        for index in range(len(target)):
            prediction_rows.append(
                {
                    "variant": variant,
                    "is_primary": int(primary),
                    "sequence": str(data["sequence"][index]),
                    "frame_id": int(data["frame_id"][index]),
                    "anchor_offset": int(data["anchor_offset"][index]),
                    "nested_oof_score": float(scores[index]),
                    "delta_iou_h1": float(target[index]),
                    "delta_auc_h1": float(data["delta_auc"][index]),
                    "beneficial": int(target[index] > 0.0),
                    "material_beneficial_001": int(
                        target[index] > MATERIAL_EPSILON
                    ),
                    "material_harmful_001": int(
                        target[index] < -MATERIAL_EPSILON
                    ),
                }
            )

    paths = {
        "summary": output_dir / "stage8_motion_gate_summary.csv",
        "budget": output_dir / "stage8_motion_gate_budget.csv",
        "outer_folds": output_dir / "stage8_motion_gate_outer_folds.csv",
        "inner_alpha": output_dir / "stage8_motion_gate_inner_alpha.csv",
        "predictions": output_dir / "stage8_motion_gate_oof_predictions.csv",
    }
    for path, rows in (
        (paths["summary"], summary_rows),
        (paths["budget"], budget_rows),
        (paths["outer_folds"], result["outer_rows"]),
        (paths["inner_alpha"], result["alpha_rows"]),
        (paths["predictions"], prediction_rows),
    ):
        write_csv(path, rows)
    experiment = {
        "schema": SCHEMA,
        "cache_fingerprint": run["fingerprint"],
        "manifest_fingerprint": run["manifest"]["fingerprint"],
        "configuration": vars(args),
        "validation_split_loaded": False,
        "motion_lags": list(MOTION_LAGS),
        "motion_windows": list(MOTION_WINDOWS),
        "motion_feature_names": list(MOTION_FEATURE_NAMES),
        "screen": {
            "primary_variant": args.primary_variant,
            "minimum_benefit_auroc": SCREEN_AUROC,
            "minimum_pearson": SCREEN_PEARSON,
            "budgets": list(SCREEN_BUDGETS),
            "minimum_budgets_with_positive_lift_ci": 1,
        },
        "script_sha256": file_hash(Path(__file__).resolve()),
        "outputs": {key: str(value) for key, value in paths.items()},
        "warning": (
            "This is a nested training-OOF feature screen, not validation or "
            "closed-loop evidence. A passing model requires a fresh video holdout."
        ),
    }
    experiment["fingerprint"] = fingerprint(experiment)
    write_json(output_dir / "stage8_motion_gate_experiment.json", experiment)

    print("\nNested training-OOF diagnostics:")
    print("variant                    primary    AUROC   pearson       mse  lift+  pass")
    for row in summary_rows:
        print(
            "{variant:<26s} {is_primary:>7d} {nested_oof_benefit_auroc:>8.4f} "
            "{nested_oof_pearson:>9.4f} {nested_oof_mse:>9.7f} "
            "{positive_lift_budgets_10_20_30:>6d} "
            "{passes_predeclared_primary_screen:>5d}".format(**row)
        )
    print("\nPrimary nested-OOF budget ranking:")
    print("budget  gain      random    lift_95%              oracle   recovery")
    for row in budget_rows:
        if not row["is_primary"]:
            continue
        interval = "[{:.6f},{:.6f}]".format(row["lift_ci_low"], row["lift_ci_high"])
        recovery = row["oracle_cap_recovery"]
        recovery_text = "n/a" if not np.isfinite(recovery) else "{:.2%}".format(recovery)
        print(
            "{budget_percent:>6.1f}% {mean_iou_gain:>9.6f} "
            "{random_matched_mean_iou_gain:>9.6f} {interval:<21s} "
            "{oracle_positive_cap_mean_iou_gain:>8.6f} {recovery_text:>9s}".format(
                interval=interval, recovery_text=recovery_text, **row
            )
        )
    primary_row = next(row for row in summary_rows if row["is_primary"])
    print(
        "\nPredeclared primary screen: {}".format(
            "PASS - collect a fresh holdout"
            if primary_row["passes_predeclared_primary_screen"]
            else "FAIL - do not evaluate on the existing validation split"
        )
    )
    print("Reports: {}".format(output_dir))


if __name__ == "__main__":
    main()
