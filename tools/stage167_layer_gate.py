"""Fit the shared online handoff threshold on source-held training predictions."""

from collections import Counter

import numpy as np

import stage166_pairwise_layers as stage166


stage165, LAYERS = stage166.stage165, stage166.LAYERS
gain, veto, risk = stage165.gain, stage165.veto, stage165.risk
history, trained, prior = stage165.history, stage166.trained, stage166.prior
SHORT, LONG = history.BASELINE, "history16_trend"


def score(model, features, pairwise=False):
    centered = (features - model["mean"]) / model["scale"]
    if pairwise:
        return centered @ model["coef"][:, :-1].T + model["coef"][:, -1]
    return centered @ model["coef"] + model["bias"]


def choose_pairwise(pairwise_scores, native, proposal):
    return max((k for k in LAYERS if k != native),
               key=lambda k: (pairwise_scores[LAYERS.index(k)], k == proposal, -k))


def fit_gain(points, features, groups, fit):
    local = [p for p, use in zip(points, fit, strict=True) if use]
    outcomes = np.stack([p["tail32"] - p["tail32"][LAYERS.index(p["native_layer"])] for p in local])
    counts = Counter(groups[fit])
    weights = np.array([1. / counts[g] for g in groups[fit]])
    mean, scale, bias, coef = trained.ridge(features[fit], outcomes, weights)
    return dict(mean=mean, scale=scale, bias=bias, coef=coef)


def training_predictions(points, groups, fold, vectors, progress):
    outer_fit = np.isin(groups, fold["fit"])
    predictions = {}
    for g in fold["fit"]:
        val = groups == g
        fit = outer_fit & ~val
        cases = [p for p, use in zip(points, val, strict=True) if use]
        short = vectors[SHORT]
        gains = score(fit_gain(points, short, groups, fit), short[val])
        downside, _ = risk.downside_matrix(points, vectors[LONG], fit, val)
        pairwise, _, _ = stage166.fit_ranker(points, short, fit, val)
        candidates, margins = veto.predictions(cases, gains)
        for p, k, margin, row, scores in zip(cases, candidates, margins, downside, pairwise, strict=True):
            selected = choose_pairwise(scores, p["native_layer"], k) if k != p["native_layer"] else k
            value = float(margin - risk.PENALTY * row[LAYERS.index(k)])
            predictions[p["case_id"]] = (k, selected, value)
        progress.advance()
    local = [p for p, use in zip(points, outer_fit, strict=True) if use]
    if set(predictions) != {p["case_id"] for p in local}:
        raise ValueError("Incomplete inner source-held gate predictions")
    return local, predictions


def threshold_curve(points, predictions):
    fixed, pairwise, values = zip(*(predictions[p["case_id"]] for p in points), strict=True)
    values = np.asarray(values, float)
    eligible = np.array([p["native_layer"] != k for p, k in zip(points, fixed, strict=True)])
    effects = [[veto.effects(p, k) for p, k in zip(points, layers, strict=True)] for layers in (fixed, pairwise)]
    harmful = np.any([[e["lost_good"] or e["new_low"] for e in arm] for arm in effects], axis=0)
    deltas = np.array([[e["delta32"] for e in arm] for arm in effects])
    groups = [veto.source_group(p["sequence"]) for p in points]
    counts = Counter(groups)
    weights = np.array([1. / (counts[g] * len(counts)) for g in groups])
    rows = []
    for threshold in sorted({0., *(float(x) for x in values[eligible] if x > veto.EPSILON)}):
        accepted = eligible & (values > threshold + veto.EPSILON)
        fixed_delta, pairwise_delta = deltas @ (weights * accepted)
        rows.append(dict(threshold=threshold, switches=int(accepted.sum()), harmful_switches=int((accepted & harmful).sum()),
                         fixed_delta=float(fixed_delta), pairwise_delta=float(pairwise_delta),
                         joint_min_delta=float(min(fixed_delta, pairwise_delta))))
    selected = max((r for r in rows if not r["harmful_switches"]),
                   key=lambda r: (r["joint_min_delta"], -r["switches"], r["threshold"]))
    return selected, rows


def fitted_gate(points, folds, vectors, gain_models, rank_models, rank_scores, progress):
    groups = np.array([veto.source_group(p["sequence"]) for p in points])
    models, summary, curves = {}, [], []
    for fold in folds:
        number = fold["fold"]
        fit, held = np.isin(groups, fold["fit"]), np.isin(groups, fold["held"])
        if not fit.any() or not held.any() or not np.array_equal(fit, ~held):
            raise ValueError("Invalid shared-gate source partition")
        np.testing.assert_array_equal(held, [p["fold"] == number for p in points])
        train, inner = training_predictions(points, groups, fold, vectors, progress)
        picked, rows = threshold_curve(train, inner)
        summary.append(dict(fold=number, fit_sources=len(fold["fit"]), fit_cases=len(train), **picked))
        curves.extend(dict(fold=number, selected=row is picked, **row) for row in rows)
        new = {k: np.asarray(v, float) for k, v in gain_models[f"fold{number}_future_supervised"].items()}
        rank = {k: np.asarray(v, float) for k, v in rank_models[f"fold{number}_{stage166.LEARNED}"].items()}
        _, frozen_downside = stage165.predict_held(points, np.flatnonzero(held), fit, held, vectors, new)
        held_rank = score(rank, vectors[SHORT][held], pairwise=True)
        np.testing.assert_allclose(held_rank, rank_scores[held], atol=1e-10, rtol=0)
        long = vectors[LONG]
        local = [p for p, use in zip(points, fit, strict=True) if use]
        targets = np.stack([np.maximum(0., p["tail32"][LAYERS.index(p["native_layer"])] - p["tail32"]) for p in local])
        counts = Counter(groups[fit])
        weights = np.array([1. / counts[g] for g in groups[fit]])
        mean, scale, bias, coef = trained.ridge(long[fit], targets, weights)
        downside = dict(mean=mean, scale=scale, bias=bias, coef=coef)
        np.testing.assert_allclose(np.maximum(0., score(downside, long[held])), frozen_downside, atol=1e-12, rtol=0)
        models[number] = dict(gain=new, risk=downside, pairwise=rank, threshold=picked["threshold"])
        progress.advance()
    return models, summary, curves


def online_decision(models, native, short, long, pairwise):
    values = score(models["gain"], short)
    proposal = prior.best_layer(values, native)
    j = LAYERS.index(proposal)
    margin = values[j] - values[LAYERS.index(native)]
    downside = max(0., score(models["risk"], long)[j])
    value = float(margin - risk.PENALTY * downside)
    if not np.isfinite(value):
        raise ValueError("Invalid online gate score")
    switched = bool(proposal != native and value > models["threshold"] + veto.EPSILON)
    chosen = native
    if switched:
        chosen = choose_pairwise(score(models["pairwise"], short, pairwise=True), native, proposal) if pairwise else proposal
    return proposal, value, switched, chosen
