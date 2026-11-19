"""Compare absolute IoU regression and same-frame ranking on frozen SGLA pools."""
from __future__ import annotations

import argparse
import csv
from itertools import zip_longest
from math import isclose
from pathlib import Path

import numpy as np

import stage143_fixed_pool_quality as fixed
from stage27_uav_collect import writer_lock
from stage46_shift_audit import publish_table


SCHEMA = "stage144_pairwise_candidate_ranking_v1"
STATUS = "COMPLETE_FIXED_POOL_PAIRWISE_RANKING_DIAGNOSTIC"
SOURCE143 = "5433964d4a3a1a90d5ee0d03c07c45b1fd6f8dbb4c7e4ceddc2de8b6866a7240"
METHODS = ("pairwise", "hard_pairwise")
STRATA = ("all", "rescue", "protect")
DESIGN = dict(fixed.DESIGN, bootstrap_seed=23144, hard_mix=[.5, .25, .25])
COMPARISONS = tuple((arm, ref) for arm in METHODS for ref in ("quality", "hann", "raw")) + (
    ("hard_pairwise", "pairwise"),)
TABLES = ("policy", "paired", "stable", "fit", "focus", "frames")


def frame_pairs(row):
    """Enumerate coexisting cell pairs once, signed from lower to higher IoU."""
    x, y = row["x"], row["y"]
    membership = np.zeros((len(fixed.POOLS), len(y)), dtype=bool)
    raw_winners = set()
    for index, pool in enumerate(fixed.POOLS):
        cells = row["pools"][pool]
        membership[index, cells] = True
        raw_winners.add(int(cells[x[cells, 0].argmax()]))
    left, right = np.triu_indices(len(y), k=1)
    valid = (membership[:, left] & membership[:, right]).any(axis=0)
    valid &= (y[left] >= 0) & (y[right] >= 0) & (y[left] != y[right])
    left, right = left[valid], right[valid]
    unordered = y[left] < y[right]
    better, worse = np.where(unordered, right, left), np.where(unordered, left, right)
    gaps = y[better] - y[worse]
    features = x[better] - x[worse]
    native = row["pools"]["peak"][0]
    good, low = DESIGN["good"], DESIGN["low"]
    rescue = (row["native_iou"] < low) & (y[better] >= good) & (y[worse] < low)
    rescue &= np.isin(worse, sorted(raw_winners | {int(native)}))
    protect = (better == native) & (y[better] >= good) & (y[worse] < low)
    protect &= np.isin(worse, sorted(raw_winners))
    return features, gaps, (np.ones(len(gaps), dtype=bool), rescue, protect)


def source_moments(rows):
    dims = len(fixed.FEATURES)
    stats = np.zeros((len(STRATA), dims+1, dims+1))
    frames = np.zeros(len(STRATA), dtype=int)
    pairs = np.zeros(len(STRATA), dtype=int)
    for row in rows:
        if row["native_iou"] < 0:
            continue
        features, gaps, subsets = frame_pairs(row)
        for i, mask in enumerate(subsets):
            if not mask.any():
                continue
            weights = gaps[mask] / gaps[mask].sum()
            data = np.column_stack((features[mask], np.ones(mask.sum())))
            stats[i] += (data.T * weights) @ data
            frames[i] += 1
            pairs[i] += mask.sum()
    return stats / np.maximum(frames, 1)[:, None, None], frames, pairs


def fit_pairwise(bank, held, regression):
    peers = sorted(set(bank) - {held})
    if regression["held_source"] != held or regression["fit_sources"] != peers:
        raise ValueError("Stage-143 source split changed")
    frame_counts = np.array([bank[g][1] for g in peers])
    supported = frame_counts > 0
    counts = supported.sum(axis=0)
    if not counts[0]:
        raise ValueError("No within-frame comparison pairs in fit sources")
    stats = np.sum([bank[g][0] for g in peers], axis=0) / np.maximum(counts, 1)[:, None, None]
    scale = np.asarray(regression["scale"], dtype=float)
    models = {}
    for arm, weights in zip(METHODS, ([1., 0., 0.], DESIGN["hard_mix"])):
        weights = np.array(weights) * (counts > 0)
        weights /= weights.sum()
        aggregate = np.einsum("i,ijk->jk", weights, stats)
        gram = aggregate[:-1, :-1] / np.outer(scale, scale) + DESIGN["ridge"] * np.eye(len(scale))
        coef = np.linalg.solve(gram, aggregate[:-1, -1] / scale)
        models[arm] = dict(held_source=held, fit_sources=peers, mean=regression["mean"], scale=regression["scale"],
                           coef=coef.tolist(), intercept=0., mixture=weights.tolist(),
                           strata_sources=counts.tolist(), strata_frames=frame_counts.sum(axis=0).tolist(),
                           strata_pairs=np.sum([bank[g][2] for g in peers], axis=0).tolist())
    return models


def evaluate(row, regression, models):
    fields = {k: row[k] for k in ("source", "sequence", "frame", "state", "case", "branch", "relapse", "onset",
                                  "stable_window", "native_iou", "grid_iou", "budget")}
    fields.update(fixed.decide(row, fixed.score(regression, row["x"])))
    for arm, model in models.items():
        scores = fixed.score(model, row["x"])
        for pool, idx in row["pools"].items():
            choice = idx[scores[idx].argmax()]
            fields[f"{pool}_{arm}_cell"] = int(row["cells"][choice])
            fields[f"{pool}_{arm}_iou"] = float(row["y"][choice])
    return fields


def audit_reference(row, reference):
    for key, expected in reference.items():
        actual = row[key]
        if isinstance(actual, float):
            if not isclose(float(expected), actual, abs_tol=1e-12, rel_tol=0.):
                raise ValueError(f"Stage-143 baseline changed at {row['sequence']}:{row['frame']} ({key})")
        elif str(actual) != expected:
            raise ValueError(f"Stage-143 candidate or baseline changed at {row['sequence']}:{row['frame']} ({key})")


def score_rows(native, replays, models, baselines, references):
    decisions = []
    for row, reference in zip_longest(native + replays, references):
        if row is None or reference is None:
            raise ValueError("Stage-143 decision roster is incomplete")
        group = row["source"]
        decision = evaluate(row, baselines[group], models[group])
        audit_reference(decision, reference)
        decisions.append(decision)
    return decisions


def check_results(root):
    report = fixed.c.c66.sealed(root / "stage144_decision.json", SCHEMA)
    files = {f"stage144_{name}.csv" for name in TABLES} | {"stage144_models.json"}
    if (report["status"] != STATUS or report["source_stage143"] != SOURCE143 or report["design"] != DESIGN
            or report["verified_reference_states"] != 29287 or report["held_out_sources"] != 28
            or report["stable_windows"] != 645 or set(report["artifact_sha256"]) != files
            or any(report[k] != val for k, val in fixed.WORK.items())):
        raise ValueError("Incomplete Stage-144 candidate ranking diagnostic")
    fixed.io.verify({str(root / k): val for k, val in report["artifact_sha256"].items()})


def run(root, r143, by_source, replays, baselines):
    native = [r for rows in by_source.values() for r in rows]
    if (len(by_source), len(native), sum(r["onset"] for r in native), len(replays)) != (28, 29047, 19, 240):
        raise ValueError("Frozen Stage-143 state roster changed")
    if set(by_source) != set(baselines):
        raise ValueError("Frozen Stage-143 held-out source roster changed")
    bank = {group: source_moments(rows) for group, rows in by_source.items()}
    models = {g: fit_pairwise(bank, g, baselines[g]) for g in sorted(bank)}
    with (r143 / "stage143_frames.csv").open(encoding="utf-8", newline="") as stream:
        decisions = score_rows(native, replays, models, baselines, csv.DictReader(stream))
    policy, paired = fixed.summarize(decisions, arms=(*fixed.ARMS, *METHODS), comparisons=COMPARISONS, design=DESIGN)
    stable = fixed.stable_report(decisions, arms=("raw", "quality", *METHODS))
    if len(stable) != 645:
        raise ValueError("Stage-143 stable window roster changed")
    focus = [r for r in decisions if r["onset"] or (r["state"] == "native" and r["native_iou"] < DESIGN["low"]
             and r["dense_oracle_iou"] >= DESIGN["good"] > r["peak_oracle_iou"]) or
             (r["state"] == "replay" and r["source"] == "uav0000089" and r["frame"] in (1165, 1166))]
    fit = []
    for held, candidates in models.items():
        for arm, model in candidates.items():
            row = dict(held_source=held, arm=arm, fit_sources="|".join(model["fit_sources"]))
            for key in ("mixture", "strata_sources", "strata_frames", "strata_pairs"):
                row.update({f"{name}_{key}":val for name, val in zip(STRATA, model[key])})
            fit.append(row)
    hashes = {}
    for name, rows in zip(TABLES, (policy, paired, stable, fit, focus, decisions)):
        path = root / f"stage144_{name}.csv"
        publish_table(path, rows)
        hashes[path.name] = fixed.c.file_hash(path)
    path = root / "stage144_models.json"
    fixed.c.publish_identical(path, fixed.io.seal(dict(schema=SCHEMA, design=DESIGN, models=models)))
    hashes[path.name] = fixed.c.file_hash(path)
    runs = {f"{p}_{a}":sum(r[f"{p}_{a}_low_run"] for r in stable) for p in fixed.POOLS for a in METHODS}
    scripts = [Path(__file__), Path(fixed.__file__), Path(fixed.prior.__file__)]
    fixed.c.publish_identical(root / "stage144_decision.json", fixed.io.seal(dict(
        schema=SCHEMA, status=STATUS, role="consumed_development_out_of_source_fixed_state_ranking",
        design=DESIGN, source_stage143=SOURCE143, source_stage142=fixed.SOURCE142,
        verified_reference_states=len(decisions), held_out_sources=len(models), stable_windows=len(stable),
        stable_windows_with_new_low_run=runs, code_sha256={p.name:fixed.c.file_hash(p) for p in scripts},
        artifact_sha256=hashes, **fixed.WORK)))
    check_results(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76", "stage89", "stage93", "stage121", "stage133", "stage139", "stage142", "stage143")
    for name in names:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-results", action="store_true")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if any(getattr(args, name) is None for name in names):
        parser.error("Supply all frozen source directories")
    sources = [getattr(args, name).expanduser().resolve() for name in names]
    if any(root == p or root in p.parents or p in root.parents for p in sources):
        parser.error("Use a separate Stage-144 directory")
    for name, path in zip(names, sources):
        if not path.is_dir():
            parser.error(f"Missing --{name} source directory: {path}")
    with writer_lock(root):
        if (root / "stage144_decision.json").exists():
            check_results(root)
        else:
            r143 = sources[-1]
            fixed.check_results(r143)
            report = fixed.c.c66.sealed(r143 / "stage143_decision.json", fixed.SCHEMA)
            if report["fingerprint"] != SOURCE143:
                raise ValueError("Expected the reviewed Stage-143 result")
            pack = fixed.c.c66.sealed(r143 / "stage143_models.json", fixed.SCHEMA)
            if pack["design"] != fixed.DESIGN:
                raise ValueError("Stage-143 regression design changed")
            native, replays = fixed.load_inputs(*sources[:-1])
            run(root, r143, native, replays, pack["models"])
    print(f"Stage-144: {STATUS}", flush=True)


if __name__ == "__main__":
    main()
