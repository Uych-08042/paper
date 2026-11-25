"""Pure-CPU protocol, geometry and summaries for the Stage-194 factorial pilot."""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

SCHEMA = "stage194_matched_module_ablation_v1"
# Training opportunity inherited from Stage-48/51. These are pilot settings,
# not theoretically optimal constants. All arms use exactly the same draws.
DESIGN = dict(folds=2, fold_seed=23194, seeds=[2132, 2133], samples_per_video=48,
              updates=256, batch=4, learning_rate=1e-5, weight_decay=1e-4,
              gradient_clip=1., smoke_updates=8, bootstrap_seed=24194,
              bootstrap_samples=20000, reference_layers=list(range(7, 13)))
FACTORIAL = tuple(f"a{a}_r{r}_q{q}" for a, r, q in itertools.product((0, 1), repeat=3))
TRAINED = (*FACTORIAL[1:], "current_a1_r0_q0", "current_a1_r1_q1")
CONTROLS = ("native", "fixed_L9", "fit_best_fixed")
ARMS = (*CONTROLS, *TRAINED)
METRICS = ("auc", "mean_iou", "low_iou_fraction", "failure_events",
           "longest_failure_fraction", "unrecovered_events", "missed_native_recoveries",
           "native_good_frame_loss", "median_pipeline_ms")


def flags(arm):
    name = arm.removeprefix("current_")
    if name not in FACTORIAL:
        raise ValueError("Unknown factorial arm: " + arm)
    return tuple(int(x[1]) for x in name.split("_"))


def split_groups(groups):
    groups = sorted(set(groups))
    if len(groups) != 38:
        raise ValueError("Stage-194 requires the 38 historical development groups")
    shuffled = np.random.default_rng(DESIGN["fold_seed"]).permutation(groups)
    return [dict(fold=i, fit=sorted(set(groups) - set(held)), held=sorted(map(str, held)))
            for i, held in enumerate(np.array_split(shuffled, DESIGN["folds"]))]


def uniform_frames(total):
    if total < 1:
        raise ValueError("Need at least one tracked frame")
    return np.unique(np.linspace(1, total, min(total, DESIGN["samples_per_video"])).round().astype(int))


def draws_for(fold, seed, sizes):
    """Equal probability for fit source groups, then uniform within that video."""
    groups = list(fold["fit"])
    rng = np.random.default_rng(seed + 100 * fold["fold"])
    return [[(groups[j], int(rng.integers(sizes[groups[j]])))
             for j in rng.integers(len(groups), size=DESIGN["batch"])]
            for _ in range(DESIGN["updates"])]


def safe_output(root, inputs):
    root = Path(root).resolve()
    for source in inputs:
        source = Path(source).resolve()
        if root == source or source in root.parents or root in source.parents:
            raise ValueError("Use an independent output directory outside sources and dataset")


def dense_image_boxes(size, offset, parent, factor, search_size, image_hw):
    """Match CENTER grid decoding, public map_box_back and clip_box(margin=10)."""
    size, offset = np.asarray(size, np.float32), np.asarray(offset, np.float32)
    side = size.shape[-1]
    if size.shape != (2, side, side) or offset.shape != size.shape:
        raise ValueError("Expected CENTER size and offset [2,S,S]")
    yy, xx = np.indices((side, side), dtype=np.float32)
    center = np.stack((xx + offset[0], yy + offset[1]), -1) / np.float32(side)
    normalized = np.concatenate((center, size.transpose(1, 2, 0)), -1)
    physical = (normalized * np.float32(search_size) / np.float32(factor)).astype(np.float64).reshape(-1, 4)
    center, wh = physical[:, :2].copy(), physical[:, 2:]
    parent = np.asarray(parent, float)
    center += parent[:2] + parent[2:] / 2 - .5 * search_size / factor
    lo, hi = center - wh / 2, center + wh / 2
    bound = np.asarray(image_hw, float)[::-1]
    lo = np.minimum(np.maximum(lo, 0), bound - 10)
    hi = np.minimum(np.maximum(hi, 10), bound)
    return np.c_[lo, np.maximum(hi - lo, 10)]


def contrast(values):
    values = np.asarray(values, float)
    rng = np.random.default_rng(DESIGN["bootstrap_seed"])
    boot = values[rng.integers(len(values), size=(DESIGN["bootstrap_samples"], len(values)))].mean(1)
    return dict(videos=len(values), mean_delta=float(values.mean()), median_delta=float(np.median(values)),
                ci_low=float(np.quantile(boot, .025)), ci_high=float(np.quantile(boot, .975)),
                positive=int((values > 1e-9).sum()), negative=int((values < -1e-9).sum()),
                worst_leave_one_video_out=float((values.sum() - values.max()) / (len(values) - 1)))


def summarize(rows):
    """Average seeds within source first. Sources, rather than frames, are units."""
    seqs = sorted({r["sequence"] for r in rows})
    index = {}
    for arm in ARMS:
        for seq in seqs:
            source = [r for r in rows if r["arm"] == arm and r["sequence"] == seq]
            expected = 1 if arm in CONTROLS else len(DESIGN["seeds"])
            if len(source) != expected or len({r["seed"] for r in source}) != expected:
                raise ValueError(f"Incomplete seed/video cell {arm} {seq}")
            index[arm, seq] = {m: float(np.mean([r[m] for r in source])) for m in METRICS}
            index[arm, seq]["introduced_failure_any_seed"] = any(r["introduced_failure"] for r in source)
    policies = [dict(arm=arm, videos=len(seqs),
                     introduced_failure_videos=sum(index[arm, s]["introduced_failure_any_seed"] for s in seqs),
                     **{m: float(np.mean([index[arm, s][m] for s in seqs])) for m in METRICS}) for arm in ARMS]
    pairs = [(a, b) for a in TRAINED for b in ("native", "fit_best_fixed")]
    pairs += [("a1_r1_q1", a) for a in ("a1_r0_q0", "a0_r1_q0", "a0_r0_q1", "current_a1_r1_q1")]
    pairs += [("a1_r0_q0", "current_a1_r0_q0")]
    paired = [dict(contrast=a + "_minus_" + b, metric=m,
                   **contrast([index[a, s][m] - index[b, s][m] for s in seqs]))
              for a, b in pairs for m in ("auc", "low_iou_fraction", "longest_failure_fraction",
                                          "missed_native_recoveries", "native_good_frame_loss")]
    # Main-effect contrasts in the trained factorial are descriptive because
    # the all-off cell is a frozen anchor, not a trained placebo model.
    factors = []
    for axis, name in enumerate(("adapter", "router", "quality")):
        per_source = []
        for seq in seqs:
            delta = []
            for bits in itertools.product((0, 1), repeat=3):
                if bits[axis]:
                    continue
                other = list(bits)
                other[axis] = 1
                key = lambda b: "native" if not any(b) else f"a{b[0]}_r{b[1]}_q{b[2]}"
                delta.append(index[key(other), seq]["auc"] - index[key(bits), seq]["auc"])
            per_source.append(float(np.mean(delta)))
        factors.append(dict(factor=name, metric="auc", **contrast(per_source)))
    return policies, paired, factors
