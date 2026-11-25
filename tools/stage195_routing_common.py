"""Exact error decomposition and cache-only policy diagnostics for Stage-195."""
from __future__ import annotations

import numpy as np

SCHEMA = "stage195_frozen_router_objective_audit_v1"
ARM = "a0_r1_q0"
LAYERS = np.arange(7, 13)
NATIVE = 2
EPS = 1e-8  # numerical equality, not a performance threshold
PREDICTORS = ("trained", "initial_half", "fit_label_mean", "fit_prediction_mean", "fit_draw_label_mean")
POLICIES = ("trained", "native_L9", "fit_label_mean", "fit_prediction_mean", "fit_draw_label_mean", "oracle", "uniform_expectation")


def validate_scores(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if y.ndim != 2 or y.shape[1] != 6 or p.shape != y.shape or not len(y):
        raise ValueError("Need nonempty [states,6] labels and predictions")
    if not np.isfinite(y).all() or not np.isfinite(p).all() or ((y < 0) | (y > 1)).any():
        raise ValueError("Invalid quality values")
    return y, p


def choose(p):
    """Explicit diagnostic tie rule: retain L9, then first maximum.

    Trained actions are supplied separately from original torch.topk. This
    rule belongs only to statistical controls and the oracle, not source replay.
    """
    p = np.asarray(p, float)
    selected = p.argmax(1)
    selected[np.abs(p[:, NATIVE] - p.max(1)) <= EPS] = NATIVE
    return selected


def ranking(y, p):
    y, p = validate_scores(y, p)
    i, j = np.triu_indices(6, 1)
    delta, predicted = y[:, i] - y[:, j], p[:, i] - p[:, j]
    eligible = np.abs(delta) > EPS
    ties = np.abs(predicted) <= EPS
    correct = ((delta * predicted > 0) & ~ties).astype(float) + .5 * ties
    weight = np.where(eligible, np.abs(delta), 0.)
    return dict(pair_count=int(eligible.sum()), truth_tied_pairs=int((~eligible).sum()),
                prediction_tied_pairs=int((ties & eligible).sum()),
                pair_accuracy=float(correct[eligible].mean()) if eligible.any() else None,
                gap_weighted_pair_accuracy=float((correct * weight).sum() / weight.sum()) if weight.sum() else None,
                mean_informative_gap=float(np.abs(delta[eligible]).mean()) if eligible.any() else None)


def errors(y, p):
    y, p = validate_scores(y, p)
    ym, pm = y.mean(1, keepdims=True), p.mean(1, keepdims=True)
    yc, pc = y - ym, p - pm
    absolute = float(np.mean((p - y) ** 2))
    common = float(np.mean((pm - ym) ** 2))
    relative = float(np.mean((pc - yc) ** 2))
    contrast_energy = float(np.mean(yc ** 2))
    total_var = float(np.mean((y - y.mean()) ** 2))
    frame_var = float(np.mean((ym - y.mean()) ** 2))
    other = np.arange(6) != NATIVE
    advantage_error = ((p - p[:, NATIVE, None]) - (y - y[:, NATIVE, None]))[:, other]
    gap = np.sort(y, axis=1)[:, -1] - np.sort(y, axis=1)[:, -2]
    return dict(states=len(y), mse=absolute, common_mse=common, relative_mse=relative,
                decomposition_error=abs(absolute - common - relative),
                target_total_variance=total_var, target_frame_variance=frame_var,
                target_contrast_energy=contrast_energy,
                target_frame_variance_fraction=frame_var / total_var if total_var else None,
                relative_skill_vs_zero=1 - relative / contrast_energy if contrast_energy else None,
                native_advantage_mse=float(np.mean(advantage_error ** 2)),
                prediction_centered_temporal_variance=float(np.var(pc, axis=0).mean()),
                prediction_mean=float(p.mean()), target_mean=float(y.mean()),
                mean_best_second_gap=float(gap.mean()), **ranking(y, p))


def policy_metrics(y, choice=None):
    y = np.asarray(y, float)
    best = y.max(1)
    if choice is None:
        value = y.mean(1)
        counts = np.full(6, len(y) / 6)
    else:
        choice = np.asarray(choice, int)
        if choice.shape != (len(y),) or ((choice < 0) | (choice > 5)).any():
            raise ValueError("Invalid layer choices")
        value = y[np.arange(len(y)), choice]
        counts = np.bincount(choice, minlength=6)
    probability = counts / counts.sum()
    return dict(mean_iou=float(value.mean()), mean_regret=float((best - value).mean()),
                gain_vs_L9=float((value - y[:, NATIVE]).mean()),
                good_fraction=float((value >= .5).mean()) if choice is not None else float((y >= .5).mean()),
                modal_fraction=float(probability.max()), entropy_bits=float(-(probability[probability > 0] * np.log2(probability[probability > 0])).sum()),
                **{f"L{k}_count": float(counts[k - 7]) for k in range(7, 13)})


def fit_controls(videos, fit, draws=None):
    """Only fit source groups may determine constant baselines."""
    if not fit or any(g not in videos for g in fit):
        raise ValueError("Missing fit-source prediction pool")
    controls = dict(fit_label_mean=np.stack([videos[g]["labels"].mean(0) for g in fit]).mean(0),
                    fit_prediction_mean=np.stack([videos[g]["prediction"].mean(0) for g in fit]).mean(0))
    if draws is not None:
        if any(g not in fit for batch in draws for g, _ in batch):
            raise ValueError("Held source in reconstructed training draws")
        controls["fit_draw_label_mean"] = np.stack([videos[g]["labels"][i] for batch in draws for g, i in batch]).mean(0)
    return controls


def path_comparison(learned, controls):
    """Assert identity only where an actually available constant path matches."""
    unique = np.unique(learned["layer"])
    row = dict(constant_layer=int(unique[0]) if len(unique) == 1 else None,
               distinct_layers=len(unique), fixed_control_available=False,
               checked_equal=False, maximum_box_error=None)
    if len(unique) == 1 and int(unique[0]) in controls:
        other = controls[int(unique[0])]
        row["fixed_control_available"] = True
        for key in ("frame_id", "layer", "parent", "boxes"):
            np.testing.assert_array_equal(learned[key], other[key])
        row.update(checked_equal=True, maximum_box_error=0.)
    return row
