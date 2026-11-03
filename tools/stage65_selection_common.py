"""Same-state diagnostics for frozen Stage-64 selectors; no new controller or fit."""
from __future__ import annotations
import numpy as np
import stage64_reliability_common as c64
import stage64_reliability_experiment as e64
import stage62_candidate_common as c62

SCHEMA = "stage65_cached_same_state_selector_diagnostic_v1"
FILES = ("stage65_selection_common.py", "stage65_selection_audit.py", "test_stage65_selection.py", "run_stage65_selection.sh")
DESIGN = dict(before=30, after=30, fit_quantiles=[.01, .99], epsilon=1e-6)
RULES = ("q0", "hann", "quality", "reliability")
EXTRA = ("changed_vs_recorded_fraction", "mean_target_pair_abs_error")
RATES = c62.RATE_METRICS + EXTRA
SHIFT = ("coordinate_outside_fit_fraction", "pair_any_outside_fit_fraction", "mean_abs_standardized_feature",
         "p99_pair_max_abs_standardized_feature", "same_utility_different_iou_pair_fraction")


def case(name, sequence, origin, seed, policy, anchor):
    return dict(case_id=name, sequence=sequence, origin=origin, seed=seed, anchor_policy=policy, anchor=anchor)


CASES = (
    case("F01", "uav0000003_00000_s", "native", 0, "reliability", "first_failure"),
    case("F02", "uav0000226_05370_s", "native", 0, "reliability", "first_failure"),
    case("F03", "uav0000226_05370_s", "native", 0, "quality", "first_recovery"),
    case("F04", "uav0000307_04531_s", "native", 0, "quality", "first_failure"),
    case("F05", "uav0000307_04531_s", "native", 0, "q0", "first_recovery"),
    case("F06", "uav0000226_05370_s", "endpoint", 2132, "reliability", "first_failure"),
    case("F07", "uav0000226_05370_s", "endpoint", 2133, "reliability", "first_failure"),
    case("F08", "uav0000304_00253_s", "endpoint", 2132, "hann", "first_failure"),
    case("F09", "uav0000307_04531_s", "endpoint", 2132, "hann", "source_current_first_recovery"),
    case("F10", "uav0000307_04531_s", "endpoint", 2133, "hann", "source_current_first_recovery"),
)


def aggregate(rows, keys, support, metrics=RATES):
    """Average supported seeds inside videos, then supported videos equally."""
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for values, items in groups.items():
        row = dict(zip(keys, values)); row[support] = len(items)
        row["frame_states_total"] = sum(r.get("frame_states", r.get("frame_states_total", 0)) for r in items)
        for metric in metrics:
            valid = [r[metric] for r in items if r[metric] is not None]
            row[metric] = float(np.mean(valid)) if valid else None
            row[metric+"_supported_"+support] = len(valid)
        out.append(row)
    return out


def reference(features, weights, models):
    f = np.asarray(features, float); w = np.asarray(weights, float)
    scale = np.asarray(models["quality"]["scale"], float)
    np.testing.assert_array_equal(scale, models["reliability"]["scale"])
    if f.ndim != 3 or f.shape[1:] != (4, 64) or w.shape != (len(f),) or np.any(w <= 0):
        raise ValueError("Need native fit features and positive state weights")
    z = ((f-f[:, :1])/scale)[:, 1:].reshape(-1, 64)
    if not np.isfinite(z).all(): raise ValueError("Nonfinite fit reference")
    pair_w = np.repeat(w/w.sum(), 3)/3
    bounds = []
    for j in range(64):
        order = np.argsort(z[:, j], kind="stable"); mass = np.cumsum(pair_w[order]); mass /= mass[-1]
        bounds.append([float(z[order[min(int(np.searchsorted(mass, q)), len(order)-1)], j]) for q in DESIGN["fit_quantiles"]])
    return dict(scale=scale.tolist(), bounds=bounds, states=len(f),
                quantile_definition="weighted inverse empirical CDF; video/policy/frame/three pairs equal as Stage-64")


def shifted(features, overlaps, ref):
    f, y = np.asarray(features, float), np.asarray(overlaps, float)
    z = ((f-f[:, :1])/np.asarray(ref["scale"]))[:, 1:]
    bounds = np.asarray(ref["bounds"])
    if z.shape[1:] != (3, 64) or y.shape != f.shape[:2] or not np.isfinite(z).all():
        raise ValueError("Invalid feature-shift input")
    outside = (z < bounds[:, 0]) | (z > bounds[:, 1])
    utility = (y >= .5).astype(int)-(y < .2).astype(int)
    return dict(coordinate_outside_fit_fraction=float(outside.mean()),
        pair_any_outside_fit_fraction=float(outside.any(2).mean()), mean_abs_standardized_feature=float(np.abs(z).mean()),
        p99_pair_max_abs_standardized_feature=float(np.quantile(np.abs(z).max(2), .99)),
        same_utility_different_iou_pair_fraction=float(((utility[:, 1:]==utility[:, :1]) &
            (np.abs(y[:, 1:]-y[:, :1]) > DESIGN["epsilon"])).mean()))


def score(data, features, models, epsilon):
    """GT is deliberately absent from this interface; states never advance."""
    base, f = np.asarray(data["features"], float), np.asarray(features, float)
    if base.shape != (len(f), 4, 24) or f.shape[1:] != (4, 64): raise ValueError("Missing cached features")
    if not np.isfinite(base).all() or not np.isfinite(f).all(): raise ValueError("Nonfinite cached features")
    choices, scores = {}, {}
    for rule in RULES:
        result = [c64.choose(rule, b, x, models.get(rule), epsilon) for b, x in zip(base, f)]
        choices[rule] = np.array([q for q, _ in result], dtype=int)
        scores[rule] = np.array([v for _, v in result])
    return choices, scores


def metrics(data, entry, features, models, epsilon):
    _, detail = c62.frame_metrics(data, entry)
    choices, scores = score(data, features, models, epsilon)
    result = {}; y = detail["iou"]
    for rule in RULES:
        selected = choices[rule]; shadow = dict(data, selected=selected,
            boxes=data["candidate_boxes"][np.arange(len(selected)), selected])
        m, _ = c62.frame_metrics(shadow, entry)
        m["changed_vs_recorded"] = selected != data["selected"]
        labels = y if rule == "quality" else (y >= .5).astype(float)-(y < .2).astype(float)
        m["target_pair_abs_error"] = (np.abs(scores[rule][:, 1:]-(labels[:, 1:]-labels[:, :1])).mean(1)
                                          if rule in c64.TARGETS else None)
        result[rule] = m
    return result, detail, choices, scores


def summarize(m, mask=None):
    mask = np.ones(len(m["frame_id"]), bool) if mask is None else np.asarray(mask, bool)
    row = c62.summarize_metrics(m, mask)
    row["changed_vs_recorded_fraction"] = float(m["changed_vs_recorded"][mask].mean()) if mask.any() else None
    row["mean_target_pair_abs_error"] = (float(m["target_pair_abs_error"][mask].mean())
        if mask.any() and m["target_pair_abs_error"] is not None else None)
    return row


def anchor_frame(kind, iou, fixed=None):
    events = c62.failure_events(iou)
    if kind == "fixed": value = fixed
    elif kind == "first_failure": value = None if not events else events[0]["start"]+1
    elif kind in ("first_recovery", "source_current_first_recovery"):
        value = None if not events or events[0]["recovery"] is None else events[0]["recovery"]+1
    else: raise ValueError("Unknown anchor rule")
    if value is None or not 1 <= value <= len(iou): raise ValueError("Predeclared anchor has no support")
    return int(value)
