"""Screen a causal long-horizon value critic for selective L9 computation.

This is a development-only feasibility experiment.  It reuses the audited
Stage-8 single-intervention branches and asks whether causal tracker state can
predict the future value of one L9 action followed by Fast tracking.  The
primary target is a normalized discounted IoU return over H frames.  An
immediate H1 critic is fit with the same features and folds as a direct control.

All scores are nested, video-grouped out-of-fold predictions.  Passing this
screen is necessary before implementing an online controller or reinforcement
learning policy, but it is not closed-loop or external-validation evidence.
"""

import argparse
import math
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint, write_csv, write_json
from stage4_router_train import (
    _weighted_binary_auc,
    _weighted_mean,
    _weighted_mse,
    _weighted_pearson,
)
from stage6_temporal_stability import bootstrap_mean
from stage8_immediate_gate import _group_weights, _score_std, _weighted_mae
from stage8_intervention_summary import load_records
from stage8_motion_gate import (
    VARIANTS,
    causal_motion_features,
    nested_group_oof,
    uses_motion,
    uses_selector,
    uses_tokens,
)


SCHEMA = "stage13_long_horizon_value_screen_v1"
DEFAULT_VARIANTS = (
    "history_only",
    "token_selector",
    "motion_only",
    "token_motion",
    "token_selector_motion",
)
PRIMARY_VARIANT = "token_selector_motion"
PRIMARY_OBJECTIVE = "discounted_h20"
BASELINE_OBJECTIVE = "immediate_h1"
SCREEN_AUROC = 0.55
SCREEN_PEARSON = 0.10
SCREEN_BUDGETS = (10.0, 20.0, 30.0)
MATERIAL_EPSILON = 0.01


def discounted_value_target(
    delta_iou,
    valid_gt,
    horizon=20,
    gamma=0.9,
    min_valid_fraction=0.8,
):
    """Return normalized discounted treatment-minus-control IoU values.

    Invalid annotation frames are omitted from both numerator and denominator.
    An anchor remains eligible only when its current frame is valid and enough
    of the requested future window is annotated.
    """
    delta_iou = np.asarray(delta_iou, dtype=np.float64)
    valid_gt = np.asarray(valid_gt, dtype=bool)
    horizon = int(horizon)
    gamma = float(gamma)
    min_valid_fraction = float(min_valid_fraction)
    if delta_iou.ndim != 2 or delta_iou.shape != valid_gt.shape:
        raise ValueError("delta_iou and valid_gt must be matching [anchor,time] arrays")
    if horizon < 2 or horizon > delta_iou.shape[1]:
        raise ValueError("Long-horizon target must fit the collected trajectory")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("--gamma must lie in (0,1]")
    if not 0.0 < min_valid_fraction <= 1.0:
        raise ValueError("--min_valid_fraction must lie in (0,1]")
    valid = valid_gt[:, :horizon]
    values = delta_iou[:, :horizon]
    if not np.isfinite(values[valid]).all():
        raise ValueError("Valid intervention IoU deltas contain non-finite values")
    coverage = valid.mean(axis=1)
    eligible = valid[:, 0] & (coverage + 1e-12 >= min_valid_fraction)
    discounts = gamma ** np.arange(horizon, dtype=np.float64)
    weighted_valid = valid * discounts[None, :]
    numerator = np.sum(np.where(valid, values, 0.0) * discounts[None, :], axis=1)
    denominator = weighted_valid.sum(axis=1)
    target = np.full(len(values), np.nan, dtype=np.float64)
    target[eligible] = numerator[eligible] / denominator[eligible]
    return target, coverage.astype(np.float64), eligible


def load_value_dataset(
    directory,
    horizon=20,
    gamma=0.9,
    min_valid_fraction=0.8,
    allow_incomplete=False,
):
    """Audit all Stage-8 branches and build a causal development dataset."""
    directory, run, records, missing, expected = load_records(
        directory, require_complete=not allow_incomplete
    )
    if int(run["max_horizon"]) < int(horizon):
        raise ValueError(
            "Stage-8 cache max horizon {} is shorter than H{}".format(
                run["max_horizon"], horizon
            )
        )
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
        "anchor_fraction",
    )
    chunks = {name: [] for name in feature_fields}
    chunks.update(
        {
            "split": [],
            "sequence": [],
            "motion_history": [],
            "immediate_value": [],
            "discounted_value": [],
            "valid_fraction": [],
            "control_iou_h1": [],
            "treatment_iou_h1": [],
        }
    )
    excluded = 0
    for record in records:
        entry = record["entry"]
        values = record["data"]
        metadata = record["metadata"]
        target, coverage, eligible = discounted_value_target(
            values["delta_iou"],
            values["valid_gt"],
            horizon=horizon,
            gamma=gamma,
            min_valid_fraction=min_valid_fraction,
        )
        motion = causal_motion_features(
            values["baseline_boxes"],
            values["anchor_offset"],
            metadata["init_box"],
        )
        excluded += int((~eligible).sum())
        if not eligible.any():
            continue
        for name in feature_fields:
            chunks[name].append(values[name][eligible].copy())
        chunks["motion_history"].append(motion[eligible])
        count = int(eligible.sum())
        chunks["split"].append(np.full(count, entry["split"], dtype="U8"))
        chunks["sequence"].append(
            np.full(count, entry["sequence"], dtype="U128")
        )
        chunks["immediate_value"].append(
            values["delta_iou"][eligible, 0].astype(np.float64, copy=True)
        )
        chunks["discounted_value"].append(target[eligible].copy())
        chunks["valid_fraction"].append(coverage[eligible].copy())
        chunks["control_iou_h1"].append(
            values["control_iou"][eligible, 0].astype(np.float64, copy=True)
        )
        chunks["treatment_iou_h1"].append(
            values["treatment_iou"][eligible, 0].astype(np.float64, copy=True)
        )
    if not chunks["sequence"]:
        raise ValueError("No eligible long-horizon intervention anchors")
    data = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    data["previous_action"] = np.zeros(len(data["sequence"]), dtype=np.int64)
    data["delta_iou"] = data["discounted_value"].copy()
    for name in (
        "immediate_value",
        "discounted_value",
        "valid_fraction",
        "control_iou_h1",
        "treatment_iou_h1",
        "motion_history",
    ):
        if not np.isfinite(data[name]).all():
            raise ValueError("Long-horizon field {} contains non-finite values".format(name))
    return directory, run, data, missing, expected, excluded


def _objective_data(data, target):
    output = dict(data)
    output["delta_iou"] = np.asarray(target, dtype=np.float64)
    return output


def fit_nested_objectives(
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
    """Fit immediate and long critics with identical grouped folds/features."""
    objectives = (
        (BASELINE_OBJECTIVE, data["immediate_value"]),
        (PRIMARY_OBJECTIVE, data["discounted_value"]),
    )
    results = {}
    for objective, target in objectives:
        print("\nFitting objective: {}".format(objective), flush=True)
        results[objective] = nested_group_oof(
            _objective_data(data, target),
            variants,
            outer_folds,
            inner_folds,
            alphas,
            pca_dimensions,
            pca_oversample,
            pca_power_iterations,
            seed,
            batch_size,
        )
    return results


def _sequence_effects(sequences, selected, values):
    sequences = np.asarray(sequences, dtype=str)
    selected = np.asarray(selected, dtype=bool)
    values = np.asarray(values, dtype=np.float64)
    output = {}
    for name in sorted(set(sequences.tolist())):
        mask = sequences == name
        output[name] = float(np.mean(np.where(selected[mask], values[mask], 0.0)))
    return output


def _topk_mask(scores, budget):
    scores = np.asarray(scores, dtype=np.float64)
    maximum = (
        0
        if float(budget) <= 0.0
        else min(len(scores), int(math.ceil(len(scores) * float(budget) / 100.0)))
    )
    selected = np.zeros(len(scores), dtype=bool)
    if maximum:
        selected[np.argsort(-scores, kind="mergesort")[:maximum]] = True
    return selected


def _bootstrap(values, samples, rng):
    values = np.asarray(values, dtype=np.float64)
    return bootstrap_mean(values, samples, rng)


def ranked_long_value_rows(
    objective,
    variant,
    scores,
    data,
    budgets,
    bootstrap_samples,
    seed,
    primary=False,
):
    """Evaluate an OOF ranking against actual discounted future value."""
    scores = np.asarray(scores, dtype=np.float64)
    target = np.asarray(data["discounted_value"], dtype=np.float64)
    sequences = np.asarray(data["sequence"], dtype=str)
    names = sorted(set(sequences.tolist()))
    always = _sequence_effects(sequences, np.ones(len(target), dtype=bool), target)
    positive = target > 0.0
    positive_effects = _sequence_effects(sequences, positive, target)
    positive_headroom = float(np.mean([positive_effects[name] for name in names]))
    rng = np.random.default_rng(seed)
    rows = []
    for budget in budgets:
        selected = _topk_mask(scores, budget)
        ratio = float(selected.mean())
        effects = _sequence_effects(sequences, selected, target)
        sequence_gain = np.asarray([effects[name] for name in names], dtype=np.float64)
        random_gain = np.asarray(
            [ratio * always[name] for name in names], dtype=np.float64
        )
        lift = sequence_gain - random_gain
        oracle = _topk_mask(target, budget)
        oracle_effects = _sequence_effects(sequences, oracle, target)
        oracle_gain = float(np.mean([oracle_effects[name] for name in names]))
        capped = np.zeros(len(target), dtype=bool)
        maximum = int(selected.sum())
        positive_order = np.argsort(-target, kind="mergesort")
        positive_order = positive_order[target[positive_order] > 0.0][:maximum]
        capped[positive_order] = True
        capped_effects = _sequence_effects(sequences, capped, target)
        capped_gain = float(np.mean([capped_effects[name] for name in names]))
        gain_stats = _bootstrap(sequence_gain, bootstrap_samples, rng)
        lift_stats = _bootstrap(lift, bootstrap_samples, rng)
        selected_count = int(selected.sum())
        rows.append(
            {
                "objective": objective,
                "variant": variant,
                "is_primary": int(primary),
                "budget_percent": float(budget),
                "num_sequences": len(names),
                "num_anchors": len(target),
                "num_selected": selected_count,
                "selected_ratio": ratio,
                "mean_discounted_value_gain": float(sequence_gain.mean()),
                "gain_ci_low": gain_stats["ci_low"],
                "gain_ci_high": gain_stats["ci_high"],
                "gain_probability_positive": gain_stats["probability_positive"],
                "random_matched_gain": float(random_gain.mean()),
                "lift_over_random": float(lift.mean()),
                "lift_ci_low": lift_stats["ci_low"],
                "lift_ci_high": lift_stats["ci_high"],
                "lift_probability_positive": lift_stats["probability_positive"],
                "oracle_exact_topk_gain": oracle_gain,
                "oracle_positive_cap_gain": capped_gain,
                "positive_headroom_gain": positive_headroom,
                "positive_headroom_recovery": (
                    float(sequence_gain.mean() / positive_headroom)
                    if positive_headroom > 1e-15
                    else float("nan")
                ),
                "selected_beneficial_ratio": (
                    float(np.mean(target[selected] > 0.0)) if selected_count else ""
                ),
                "selected_harmful_ratio": (
                    float(np.mean(target[selected] < 0.0)) if selected_count else ""
                ),
                "selected_material_benefit_ratio_001": (
                    float(np.mean(target[selected] > MATERIAL_EPSILON))
                    if selected_count
                    else ""
                ),
                "selected_material_harm_ratio_001": (
                    float(np.mean(target[selected] < -MATERIAL_EPSILON))
                    if selected_count
                    else ""
                ),
            }
        )
    return rows


def compare_objective_rankings(
    variant,
    immediate_scores,
    long_scores,
    data,
    budgets,
    bootstrap_samples,
    seed,
    primary=False,
):
    """Compare long-target and H1-target rankings on future value."""
    target = np.asarray(data["discounted_value"], dtype=np.float64)
    sequences = np.asarray(data["sequence"], dtype=str)
    names = sorted(set(sequences.tolist()))
    rng = np.random.default_rng(seed)
    rows = []
    for budget in budgets:
        immediate_selected = _topk_mask(immediate_scores, budget)
        long_selected = _topk_mask(long_scores, budget)
        immediate_effect = _sequence_effects(
            sequences, immediate_selected, target
        )
        long_effect = _sequence_effects(sequences, long_selected, target)
        immediate_values = np.asarray(
            [immediate_effect[name] for name in names], dtype=np.float64
        )
        long_values = np.asarray(
            [long_effect[name] for name in names], dtype=np.float64
        )
        difference = long_values - immediate_values
        stats = _bootstrap(difference, bootstrap_samples, rng)
        union = int(np.sum(immediate_selected | long_selected))
        overlap = int(np.sum(immediate_selected & long_selected))
        rows.append(
            {
                "variant": variant,
                "is_primary": int(primary),
                "budget_percent": float(budget),
                "num_sequences": len(names),
                "immediate_objective_gain": float(immediate_values.mean()),
                "long_objective_gain": float(long_values.mean()),
                "long_minus_immediate": float(difference.mean()),
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "probability_positive": stats["probability_positive"],
                "selection_overlap": overlap,
                "selection_union": union,
                "selection_jaccard": float(overlap / union) if union else 1.0,
            }
        )
    return rows


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(DEFAULT_VARIANTS)
    )
    parser.add_argument("--primary_variant", choices=VARIANTS, default=PRIMARY_VARIANT)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--min_valid_fraction", type=float, default=0.8)
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
    parser.add_argument("--seed", type=int, default=2043)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--allow_incomplete", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if len(set(args.variants)) != len(args.variants):
        raise ValueError("Variants must be unique")
    if args.primary_variant not in args.variants:
        raise ValueError("--primary_variant must be included in --variants")
    if args.horizon != 20:
        raise ValueError("The predeclared Stage-13 screen requires --horizon 20")
    if not math.isclose(args.gamma, 0.9, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("The predeclared Stage-13 screen requires --gamma 0.9")
    if not math.isclose(
        args.min_valid_fraction, 0.8, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "The predeclared Stage-13 screen requires --min_valid_fraction 0.8"
        )
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
    if len(set(args.ridge_alphas)) != len(args.ridge_alphas):
        raise ValueError("Ridge alphas must be unique")
    if not args.budgets or any(value < 0.0 or value > 100.0 for value in args.budgets):
        raise ValueError("Budgets must lie in [0,100]")

    input_dir, run, data, missing, expected, excluded = load_value_dataset(
        args.input,
        horizon=args.horizon,
        gamma=args.gamma,
        min_valid_fraction=args.min_valid_fraction,
        allow_incomplete=args.allow_incomplete,
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
        raise ValueError("Outer folds exceed the number of development videos")
    smallest_outer_train = len(sequences) - int(
        math.ceil(len(sequences) / args.outer_folds)
    )
    if args.inner_folds > smallest_outer_train:
        raise ValueError("Inner folds exceed an outer training split")
    variants = tuple(args.variants)
    alphas = tuple(sorted(float(value) for value in args.ridge_alphas))
    budgets = tuple(sorted(set(float(value) for value in args.budgets)))
    print(
        "Audited {} of {} Stage-8 clips; loaded {} eligible anchors from {} videos "
        "({} anchors excluded by H{} coverage).".format(
            expected - missing,
            expected,
            len(data["sequence"]),
            len(sequences),
            excluded,
            args.horizon,
        )
    )
    print(
        "Development-only nested {}/{}-fold screen; target=discounted H{} gamma={:g}.".format(
            args.outer_folds, args.inner_folds, args.horizon, args.gamma
        )
    )
    results = fit_nested_objectives(
        data,
        variants,
        args.outer_folds,
        args.inner_folds,
        alphas,
        args.pca_dims,
        args.pca_oversample,
        args.pca_power_iterations,
        args.seed,
        args.batch_size,
    )

    targets = {
        BASELINE_OBJECTIVE: np.asarray(data["immediate_value"], dtype=np.float64),
        PRIMARY_OBJECTIVE: np.asarray(data["discounted_value"], dtype=np.float64),
    }
    summary_rows = []
    budget_rows = []
    comparison_rows = []
    fold_rows = []
    alpha_rows = []
    prediction_rows = []
    for objective_index, objective in enumerate((BASELINE_OBJECTIVE, PRIMARY_OBJECTIVE)):
        result = results[objective]
        target = targets[objective]
        weights = result["weights"]
        constant_mse = _weighted_mse(
            target[:, None], result["constant_predictions"][:, None], weights
        )
        for row in result["outer_rows"]:
            fold_rows.append({"objective": objective, **row})
        for row in result["alpha_rows"]:
            alpha_rows.append({"objective": objective, **row})
        for variant_index, variant in enumerate(variants):
            scores = result["predictions"][variant]
            primary = objective == PRIMARY_OBJECTIVE and variant == args.primary_variant
            summary_rows.append(
                {
                    "objective": objective,
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
                    "nested_oof_pearson": _weighted_pearson(scores, target, weights),
                    "nested_oof_benefit_auroc": _weighted_binary_auc(
                        target > 0.0, scores, weights
                    ),
                    "nested_oof_material_benefit_auroc_001": _weighted_binary_auc(
                        target > MATERIAL_EPSILON, scores, weights
                    ),
                    "nested_oof_material_harm_auroc_001": _weighted_binary_auc(
                        target < -MATERIAL_EPSILON, -scores, weights
                    ),
                    "score_mean": _weighted_mean(scores, weights),
                    "score_std": _score_std(scores, weights),
                }
            )
            budget_rows.extend(
                ranked_long_value_rows(
                    objective,
                    variant,
                    scores,
                    data,
                    budgets,
                    args.bootstrap_samples,
                    args.seed + objective_index * 1000 + variant_index * 100,
                    primary=primary,
                )
            )
            for index in range(len(target)):
                prediction_rows.append(
                    {
                        "objective": objective,
                        "variant": variant,
                        "is_primary": int(primary),
                        "split": str(data["split"][index]),
                        "sequence": str(data["sequence"][index]),
                        "frame_id": int(data["frame_id"][index]),
                        "anchor_offset": int(data["anchor_offset"][index]),
                        "anchor_fraction": float(data["anchor_fraction"][index]),
                        "valid_fraction_horizon": float(data["valid_fraction"][index]),
                        "nested_oof_score": float(scores[index]),
                        "objective_target": float(target[index]),
                        "immediate_value": float(data["immediate_value"][index]),
                        "discounted_value": float(data["discounted_value"][index]),
                    }
                )

    for variant_index, variant in enumerate(variants):
        comparison_rows.extend(
            compare_objective_rankings(
                variant,
                results[BASELINE_OBJECTIVE]["predictions"][variant],
                results[PRIMARY_OBJECTIVE]["predictions"][variant],
                data,
                budgets,
                args.bootstrap_samples,
                args.seed + 10000 + variant_index * 100,
                primary=variant == args.primary_variant,
            )
        )

    primary_summary = next(
        row
        for row in summary_rows
        if row["objective"] == PRIMARY_OBJECTIVE
        and row["variant"] == args.primary_variant
    )
    primary_budgets = [
        row
        for row in budget_rows
        if row["objective"] == PRIMARY_OBJECTIVE
        and row["variant"] == args.primary_variant
        and row["budget_percent"] in SCREEN_BUDGETS
    ]
    primary_comparisons = [
        row
        for row in comparison_rows
        if row["variant"] == args.primary_variant
        and row["budget_percent"] in SCREEN_BUDGETS
    ]
    pearson_pass = primary_summary["nested_oof_pearson"] >= SCREEN_PEARSON
    auroc_pass = primary_summary["nested_oof_benefit_auroc"] >= SCREEN_AUROC
    random_lift_pass = any(row["lift_ci_low"] > 0.0 for row in primary_budgets)
    objective_pass = any(row["ci_low"] > 0.0 for row in primary_comparisons)
    passed = pearson_pass and auroc_pass and random_lift_pass and objective_pass

    paths = {
        "summary": output_dir / "stage13_value_summary.csv",
        "budget": output_dir / "stage13_value_budget.csv",
        "objective_comparison": output_dir / "stage13_value_objective_comparison.csv",
        "outer_folds": output_dir / "stage13_value_outer_folds.csv",
        "inner_alpha": output_dir / "stage13_value_inner_alpha.csv",
        "predictions": output_dir / "stage13_value_predictions.csv",
        "decision": output_dir / "stage13_value_decision.json",
        "experiment": output_dir / "stage13_value_experiment.json",
    }
    for path, rows in (
        (paths["summary"], summary_rows),
        (paths["budget"], budget_rows),
        (paths["objective_comparison"], comparison_rows),
        (paths["outer_folds"], fold_rows),
        (paths["inner_alpha"], alpha_rows),
        (paths["predictions"], prediction_rows),
    ):
        write_csv(path, rows)

    decision = {
        "schema": SCHEMA,
        "status": "PASS_SCREEN" if passed else "FAIL_SCREEN",
        "analysis_role": "development_only_nested_oof_feasibility_screen",
        "primary_variant": args.primary_variant,
        "primary_objective": PRIMARY_OBJECTIVE,
        "horizon": args.horizon,
        "gamma": args.gamma,
        "min_valid_fraction": args.min_valid_fraction,
        "num_sequences": len(sequences),
        "num_anchors": len(data["sequence"]),
        "screen": {
            "minimum_pearson": SCREEN_PEARSON,
            "observed_pearson": _finite_or_none(
                primary_summary["nested_oof_pearson"]
            ),
            "pearson_pass": bool(pearson_pass),
            "minimum_benefit_auroc": SCREEN_AUROC,
            "observed_benefit_auroc": _finite_or_none(
                primary_summary["nested_oof_benefit_auroc"]
            ),
            "benefit_auroc_pass": bool(auroc_pass),
            "screen_budgets": list(SCREEN_BUDGETS),
            "positive_random_lift_ci_pass": bool(random_lift_pass),
            "long_beats_immediate_ci_pass": bool(objective_pass),
        },
        "next_step": (
            "Freeze a causal online controller and evaluate it on a newly acquired "
            "video-disjoint development holdout before any external benchmark."
            if passed
            else "Stop controller/RL development for this state representation; the "
            "counterfactual long-horizon value signal is not predictably actionable."
        ),
        "warning": (
            "All Stage-8 train and validation videos are development data here. "
            "This result cannot support a closed-loop or external generalization claim."
        ),
    }
    write_json(paths["decision"], decision)
    experiment = {
        "schema": SCHEMA,
        "cache_fingerprint": run["fingerprint"],
        "manifest_fingerprint": run["manifest"]["fingerprint"],
        "configuration": vars(args),
        "development_splits_loaded": sorted(set(data["split"].tolist())),
        "script_sha256": file_hash(Path(__file__).resolve()),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    experiment["fingerprint"] = fingerprint(experiment)
    write_json(paths["experiment"], experiment)

    print("\nNested OOF value diagnostics:")
    print("objective        variant                    AUROC   pearson       mse")
    for row in summary_rows:
        print(
            "{objective:<16s} {variant:<26s} {nested_oof_benefit_auroc:>7.4f} "
            "{nested_oof_pearson:>9.4f} {nested_oof_mse:>9.7f}".format(**row)
        )
    print("\nPrimary long-value budget ranking:")
    print("budget  gain      random    lift_95%              oracle   recovery")
    for row in budget_rows:
        if not row["is_primary"]:
            continue
        recovery = row["positive_headroom_recovery"]
        recovery_text = "n/a" if not np.isfinite(recovery) else "{:.2%}".format(recovery)
        print(
            "{budget_percent:>6.1f}% {mean_discounted_value_gain:>9.6f} "
            "{random_matched_gain:>9.6f} "
            "[{lift_ci_low:>8.6f},{lift_ci_high:>8.6f}] "
            "{oracle_positive_cap_gain:>8.6f} {recovery:>9s}".format(
                recovery=recovery_text, **row
            )
        )
    print("\nPrimary long-target minus immediate-target ranking:")
    print("budget  delta     paired_95%           overlap")
    for row in comparison_rows:
        if not row["is_primary"]:
            continue
        print(
            "{budget_percent:>6.1f}% {long_minus_immediate:>9.6f} "
            "[{ci_low:>8.6f},{ci_high:>8.6f}] {selection_jaccard:>8.2%}".format(
                **row
            )
        )
    print("\nPredeclared long-value screen: {}".format(decision["status"]))
    print("Decision JSON: {}".format(paths["decision"]))
    print("Reports: {}".format(output_dir))


if __name__ == "__main__":
    main()
