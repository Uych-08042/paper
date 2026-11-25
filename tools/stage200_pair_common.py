"""Same-video independent pairs and point/difference supervision diagnostics."""
from __future__ import annotations

import numpy as np
import stage194_module_common as protocol

ARMS = ("point", "within_video_difference")


def pair_draws(fold, seed, sizes):
    """Two iid states (with replacement) per source; two pairs per update."""
    if set(fold["fit"]) & set(fold["held"]):
        raise ValueError("Fit and held groups overlap")
    if protocol.DESIGN["batch"] % 2:
        raise ValueError("State-reference budget must contain complete pairs")
    groups = list(fold["fit"])
    if not groups or any(g not in sizes or sizes[g] < 1 for g in groups):
        raise ValueError("Missing fit-source states")
    rng = np.random.default_rng(seed + 100 * fold["fold"])
    draws = []
    for _ in range(protocol.DESIGN["updates"]):
        batch = []
        for _ in range(protocol.DESIGN["batch"] // 2):
            group = groups[int(rng.integers(len(groups)))]
            a, b = map(int, rng.integers(sizes[group], size=2))
            batch.append((group, a, b))
        draws.append(batch)
    return draws


def flatten(batch):
    return [(g, i) for g, a, b in batch for i in (a, b)]


def objectives(torch, score, label):
    if score.shape != label.shape or score.ndim != 2 or score.shape[1] != 6 or len(score) % 2:
        raise ValueError("Expected paired [2*pairs,6] scores and labels")
    error = score - label
    error = error - error.mean(1, keepdim=True)
    point = error.square().mean()
    paired = error.reshape(-1, 2, 6)
    difference = .5 * (paired[:, 0] - paired[:, 1]).square().mean()
    return dict(point=point, within_video_difference=difference)


def decomposition(labels, prediction, prior):
    z, h = np.asarray(labels, float) - prior, np.asarray(prediction, float) - prior
    z, h = z - z.mean(1, keepdims=True), h - h.mean(1, keepdims=True)
    zm, hm = z.mean(0), h.mean(0)
    between, within = float(np.mean((hm - zm) ** 2)), float(np.mean(((h - hm) - (z - zm)) ** 2))
    be, we = float(np.mean(zm ** 2)), float(np.mean((z - zm) ** 2))
    error = abs(between + within - float(np.mean((h - z) ** 2)))
    if error > 1e-12:
        raise ValueError("Within/between-video MSE identity failed")
    return dict(between_video_mse=between, within_video_mse=within, between_video_energy=be, within_video_energy=we,
                within_video_skill_vs_zero=1 - within / we if we else None, video_decomposition_error=error)


def aggregate(rows, category, names, fields):
    result = []
    for arm in ARMS:
        for pool in ("fit", "held"):
            for name in names:
                selected = [r for r in rows if r["arm"] == arm and r["pool"] == pool and r[category] == name]
                groups = sorted({r["group"] for r in selected})
                if len(groups) != 38 or len(selected) != 76:
                    raise ValueError("Incomplete two-seed source summary")
                row = dict(arm=arm, pool=pool, **{category: name}, videos=38)
                for field in fields:
                    values = []
                    for group in groups:
                        cells = [r[field] for r in selected if r["group"] == group]
                        if all(v is not None for v in cells):
                            values.append(float(np.mean(cells)))
                    row[field] = float(np.mean(values)) if values else None
                    row[field + "_supported_videos"] = len(values)
                result.append(row)
    return result


def paired_reports(policies, errors):
    output = []
    def contrast(values):
        if len(values) > 1:
            return protocol.contrast(values)
        return dict(videos=len(values), mean_delta=values[0] if values else None, median_delta=values[0] if values else None,
                    ci_low=None, ci_high=None, positive=sum(v > 1e-9 for v in values), negative=sum(v < -1e-9 for v in values),
                    worst_leave_one_video_out=None)
    for pool in ("fit", "held"):
        def compare(a, pa, b, pb, metric, table, kind, name, direction="positive"):
            values = []
            groups = sorted({r["group"] for r in table})
            for g in groups:
                get = lambda arm, policy: [r[metric] for r in table if r["arm"] == arm and r["pool"] == pool
                    and r["group"] == g and r.get("policy", r.get("predictor")) == policy]
                x, y = get(a, pa), get(b, pb)
                if len(x) != 2 or len(y) != 2:
                    raise ValueError("Incomplete source/seed cell")
                if all(v is not None for v in x + y):
                    values.append(float(np.mean(x) - np.mean(y)))
            output.append(dict(arm=a, pool=pool, kind=kind, contrast=name, metric=metric,
                               favorable_direction=direction, **contrast(values)))
        for arm in ARMS:
            for control in ("prior_only", "fit_prediction_mean", "native_L9"):
                for metric in ("mean_iou", "state_alignment_gain"):
                    compare(arm, "trained", arm, control, metric, policies, "within", "trained_minus_" + control)
            selected = [r for r in policies if r["arm"] == arm and r["pool"] == pool and r["policy"] == "trained"]
            values = [float(np.mean([r["state_alignment_gain"] for r in selected if r["group"] == g]))
                      for g in sorted({r["group"] for r in selected})]
            output.append(dict(arm=arm, pool=pool, kind="alignment", contrast="trained_minus_independent_expectation",
                metric="mean_iou", favorable_direction="positive", **contrast(values)))
        for metric in ("mean_iou", "state_alignment_gain"):
            compare(ARMS[1], "trained", ARMS[0], "trained", metric, policies, "between", "difference_minus_point")
        for metric, direction in (("interaction_mse", "negative"), ("within_video_mse", "negative"),
                                  ("gap_weighted_pair_accuracy", "positive")):
            compare(ARMS[1], "trained", ARMS[0], "trained", metric, errors, "prediction", "difference_minus_point", direction)
    return output
