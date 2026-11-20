"""Audit appearance evidence on frozen SGLA candidate pairs and sampled states."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

import stage120_appearance as rgb
import stage144_pairwise_candidate_ranking as prior
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image
from stage46_shift_audit import publish_table
from stage91_template_appearance import visible_roi


fixed, c, io = prior.fixed, prior.fixed.c, prior.fixed.io
SCHEMA = "stage145_fixed_pool_appearance_v1"
STATUS = "COMPLETE_FIXED_POOL_APPEARANCE_DIAGNOSTIC"
SOURCE144 = "6f062af452c6663d9d7ef74ec8326281ad3a872bfbd5290d41ac07a7c26ae734"
CUES = ("template_hist", "template_ncc", "recent_hist", "recent_ncc")
DESIGN = dict(fixed.DESIGN, descriptor=rgb.DESIGN, recent_lookback=12, recent_spacing=3, recent_limit=3,
              trust_max_center_shift=.75, uniform_states_per_video=32, stable_windows_per_video=4,
              bootstrap_seed=23145)
TABLES = ("pair_summary", "pairs", "policy", "paired", "stable", "scope", "focus", "frames")
WORK = dict(new_search_calls=0, state_updates=0, sgla_weight_updates=0, fitted_models=0)


def context_key(row):
    return row["state"], row["sequence"], row["case"], row["branch"]


def evenly_spaced(values, count):
    if len(values) <= count:
        return list(values)
    return [values[i] for i in np.linspace(0, len(values)-1, count, dtype=int)]


def study_roster(rows):
    starts = sorted({r["stable_window"] for r in rows if r["stable_window"]})
    stable = set(evenly_spaced(starts, DESIGN["stable_windows_per_video"]))
    uniform = set(evenly_spaced([r["frame"] for r in rows], DESIGN["uniform_states_per_video"]))
    return [dict(r, study_window=r["stable_window"] if r["stable_window"] in stable else 0)
            for r in rows if r["frame"] in uniform or r["stable_window"] in stable or r["onset"] or (
                0. <= r["native_iou"] < DESIGN["low"] and r["grid_iou"] >= DESIGN["good"])]


def trusted(row):
    native = row["pools"]["peak"][0]
    for idx in row["pools"].values():
        if idx[row["x"][idx, 0].argmax()] != native:
            return False
    box = row["boxes"][native]
    parent = row["parent"]
    center_shift = np.linalg.norm(box[:2] + .5 * box[2:] - parent[:2] - .5 * parent[2:])
    return center_shift / np.sqrt(parent[2:].prod()) <= DESIGN["trust_max_center_shift"]


def recent_rows(frame, history):
    picked = []
    for past in range(frame - 1, max(1, frame - DESIGN["recent_lookback"]) - 1, -1):
        if past not in history or (picked and picked[-1]["frame"] - past < DESIGN["recent_spacing"]):
            continue
        if trusted(history[past]):
            picked.append(history[past])
            if len(picked) == DESIGN["recent_limit"]:
                break
    return picked


def candidate_descriptor(image, box, usable):
    roi, coverage = visible_roi(box, usable)
    if coverage < rgb.DESIGN["min_coverage"]:
        return {"hist": None, "ncc": None}
    left, top = np.floor(roi[:2]).astype(int)
    right, bottom = np.ceil(roi[:2] + roi[2:]).astype(int)
    # Clamp the interpolator to the visible region, including for tiny boxes.
    local = roi.copy()
    local[:2] -= (left, top)
    return rgb.describe(image[top:bottom, left:right], local)


def similarities(query, template, recent):
    scores = {}
    for name, refs in (("template", [template]), ("recent", recent)):
        for signal in ("hist", "ncc"):
            vec = query[signal]
            values = [float(np.clip(vec @ ref[signal], -1., 1.)) for ref in refs
                      if vec is not None and ref[signal] is not None]
            scores[f"{name}_{signal}"] = float(np.mean(values)) if values else np.nan
    return scores


def signed_result(margin):
    return .5 + .5 * (int(margin > 1e-12) - int(margin < -1e-12))


def compare_pairs(row, decision, appearance, regression_score):
    records = []
    good, low = DESIGN["good"], DESIGN["low"]
    for pool, idx in row["pools"].items():
        y = row["y"]
        bad = int(np.searchsorted(row["cells"], decision[f"{pool}_quality_cell"]))
        best = idx[y[idx].argmax()]
        if 0. <= row["native_iou"] < low and 0. <= y[bad] < good <= y[best]:
            category = "native_rescue" if row["state"] == "native" else "replay_rescue"
        else:
            bad = int(np.searchsorted(row["cells"], decision[f"{pool}_raw_cell"]))
            best = idx[0]
            if not decision["stable_window"] or not 0. <= y[bad] < good <= y[best]:
                continue
            category = "stable_raw_trap"
        row_out = {k: decision[k] for k in ("source", "sequence", "state", "frame", "case", "branch", "onset",
                                          "stable_window", "recent_frames", "dense_only_good")}
        row_out.update(category=category, pool=pool, good_cell=int(row["cells"][best]), bad_cell=int(row["cells"][bad]),
                       good_iou=float(y[best]), bad_iou=float(y[bad]), raw_margin=float(row["x"][best, 0]-row["x"][bad, 0]),
                       hann_margin=float(row["x"][best, 1]-row["x"][bad, 1]),
                       quality_margin=float(regression_score[best]-regression_score[bad]))
        for cue, scores in appearance.items():
            row_out[cue+"_margin"] = float(scores[best]-scores[bad]) if np.isfinite(scores[[best,bad]]).all() else None
        records.append(row_out)
    return records


def pair_summary(pairs):
    groups = defaultdict(list)
    for row in pairs:
        groups[row["category"], row["pool"]].append(row)
        if row["category"] == "native_rescue" and row["onset"]:
            groups["native_onset_rescue", row["pool"]].append(row)
        if row["category"] == "native_rescue" and row["pool"] == "dense" and row["dense_only_good"]:
            groups["dense_only_good", "dense"].append(row)
    summary = []
    for (category, pool), rows in sorted(groups.items()):
        for cue in CUES:
            signal = cue + "_margin"
            matched = [r for r in rows if r[signal] is not None]
            by_source = defaultdict(list)
            for row in matched:
                by_source[row["source"]].append(row)
            for baseline in ("raw", "hann", "quality"):
                base = baseline + "_margin"
                differences = [np.mean([signed_result(r[signal])-signed_result(r[base]) for r in part])
                               for _, part in sorted(by_source.items())]
                ci = fixed.prior.base.s103.contrast(differences, cue+"_minus_"+baseline, "pair_accuracy", DESIGN) if len(differences) > 1 else {}
                summary.append(dict(group=category, pool=pool, appearance=cue, baseline=baseline,
                                    pairs=len(rows), available=len(matched), sources=len(by_source),
                                    appearance_wins=sum(r[signal] > 1e-12 for r in matched),
                                    appearance_ties=sum(abs(r[signal]) <= 1e-12 for r in matched),
                                    baseline_wins=sum(r[base] > 1e-12 for r in matched),
                                    baseline_ties=sum(abs(r[base]) <= 1e-12 for r in matched),
                                    mean_delta=float(np.mean(differences)) if differences else None,
                                    ci_low=ci.get("ci_low"), ci_high=ci.get("ci_high")))
    return summary


def read_sequence(entry, image_root, rows, histories, references, regression, rankers, reader=checked_image):
    reads = 0

    @lru_cache(maxsize=16)
    def image(frame):
        nonlocal reads
        reads += 1
        return reader(image_root, entry["frames"][frame])

    first = image(0)
    height, width = first.shape[:2]
    template = candidate_descriptor(first, entry["initial_box"], [0., 0., width, height])

    @lru_cache(maxsize=128)
    def past_descriptor(key, frame):
        old = histories[key][frame]
        idx = old["pools"]["peak"][0]
        return candidate_descriptor(image(frame), old["boxes"][idx], old["usable"])

    pairs, decisions = [], []
    for row in rows:
        key, frame = context_key(row), row["frame"]
        ref = references[key, frame]
        decision = prior.evaluate(row, regression, rankers)
        prior.audit_reference(decision, ref)
        # Stable membership describes the sampled, complete windows only.
        decision["stable_window"] = row["study_window"]
        recent = recent_rows(frame, histories[key])
        past_frames = [r["frame"] for r in recent]
        past = [past_descriptor(key, f) for f in past_frames]
        now = image(frame)
        matches = [similarities(candidate_descriptor(now, b, row["usable"]), template, past) for b in row["boxes"]]
        scores = {cue: np.array([m[cue] for m in matches]) for cue in CUES}
        decision.update(recent_frames="|".join(map(str, past_frames)), dense_only_good=(
            row["state"]=="native" and row["native_iou"] < DESIGN["low"]
            and decision["dense_oracle_iou"] >= DESIGN["good"] > decision["peak_oracle_iou"]))
        for pool, idx in row["pools"].items():
            for cue, score in scores.items():
                available = bool(np.isfinite(score[idx]).all())
                cell = int(row["cells"][idx[score[idx].argmax()]]) if available else decision[f"{pool}_quality_cell"]
                decision[f"{pool}_{cue}_cell"] = cell
                decision[f"{pool}_{cue}_iou"] = float(row["y"][int(np.searchsorted(row["cells"], cell))])
                decision[f"{pool}_{cue}_available"] = available
        pairs.extend(compare_pairs(row, decision, scores, fixed.score(regression, row["x"])))
        decisions.append(decision)
    return decisions, pairs, reads


def reference_rows(path, selected):
    refs = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = context_key(row), int(row["frame"])
            if key in selected:
                if key in refs:
                    raise ValueError("Duplicate Stage-144 state")
                refs[key] = row
    if set(refs) != selected:
        raise ValueError("Missing selected Stage-144 states")
    return refs


def run(root, native, replays, entry_for, image_root, r144, baselines, rankers, reader=checked_image):
    by_sequence = defaultdict(list)
    histories = defaultdict(dict)
    for rows in native.values():
        for row in rows:
            by_sequence[row["sequence"]].append(row)
    selected = defaultdict(list)
    for seq, rows in by_sequence.items():
        for r in rows:
            histories[context_key(r)][r["frame"]] = r
        selected[seq].extend(study_roster(rows))
    for r in replays:
        histories[context_key(r)][r["frame"]] = r
        selected[r["sequence"]].append(dict(r, study_window=0))
    keys = {(context_key(r), r["frame"]) for rows in selected.values() for r in rows}
    refs = reference_rows(r144 / "stage144_frames.csv", keys)
    decisions, pairs, scope = [], [], []
    for seq, rows in sorted(selected.items()):
        group = rows[0]["source"]
        entry = entry_for(seq)
        if entry["sequence"] != seq:
            raise ValueError("Appearance image source differs from cached state")
        local, examples, reads = read_sequence(entry, image_root, rows, histories, refs,
                                               baselines[group], rankers[group], reader)
        decisions.extend(local)
        pairs.extend(examples)
        scope.append(dict(sequence=seq, source=group, states=len(local), pairs=len(examples), image_reads=reads,
                          recent_states=sum(bool(r["recent_frames"]) for r in local),
                          **{f"{p}_{cue}_available_states":sum(r[f"{p}_{cue}_available"] for r in local)
                             for p in fixed.POOLS for cue in CUES}))
        print(f"[Stage-145] {seq}: {len(local)} states, {len(examples)} pairs, {reads} image reads", flush=True)
    if len(decisions) != len(keys):
        raise ValueError("Duplicate study decisions")
    policy, paired = fixed.summarize(decisions, arms=(*fixed.ARMS, *prior.METHODS, *CUES),
                                     comparisons=tuple((cue, "quality") for cue in CUES), design=DESIGN)
    for row in policy + paired:
        if row["population"] == "native_all":
            row["population"] = "native_sample"
    stable = fixed.stable_report(decisions, arms=("raw", "quality", *CUES))
    focus = [r for r in decisions if r["onset"] or r["dense_only_good"] or (
        r["state"] == "replay" and r["source"] == "uav0000089" and r["frame"] in (1165,1166))]
    support = dict(native_onsets=sum(r["onset"] for r in decisions), replay_states=sum(r["state"]=="replay" for r in decisions),
                   low_grid_good=sum(r["state"]=="native" and 0.<=r["native_iou"]<DESIGN["low"] and
                                     r["grid_iou"]>=DESIGN["good"] for r in decisions),
                   dense_only_good=sum(r["dense_only_good"] for r in decisions))
    if support != dict(native_onsets=19, replay_states=240, low_grid_good=281, dense_only_good=18):
        raise ValueError("Reviewed Stage-144 diagnostic populations changed")
    hashes = {}
    for name, output in zip(TABLES, (pair_summary(pairs), pairs, policy, paired, stable, scope, focus, decisions)):
        path = root / f"stage145_{name}.csv"
        publish_table(path, output)
        hashes[path.name] = c.file_hash(path)
    scripts = (Path(__file__), Path(fixed.__file__), Path(prior.__file__), Path(rgb.__file__))
    c.publish_identical(root / "stage145_decision.json", io.seal(dict(
        schema=SCHEMA, status=STATUS, role="consumed_development_frozen_pool_appearance_evidence",
        source_stage144=SOURCE144, source_stage143=prior.SOURCE143, design=DESIGN,
        states=len(decisions), pairs=len(pairs), sampled_stable_windows=len(stable), sequences=len(scope),
        image_reads=sum(r["image_reads"] for r in scope), code_sha256={p.name:c.file_hash(p) for p in scripts},
        table_sha256=hashes, **support, **WORK)))
    check_results(root)


def check_results(root):
    report = c.c66.sealed(root / "stage145_decision.json", SCHEMA)
    if (report["status"] != STATUS or report["source_stage144"] != SOURCE144 or report["design"] != DESIGN
            or report["sequences"] != 28 or report["native_onsets"] != 19 or report["replay_states"] != 240
            or report["low_grid_good"] != 281 or report["dense_only_good"] != 18
            or set(report["table_sha256"]) != {f"stage145_{name}.csv" for name in TABLES}
            or any(report[k] != v for k, v in WORK.items())):
        raise ValueError("Incomplete Stage-145 appearance diagnostic")
    io.verify({str(root / k): v for k, v in report["table_sha256"].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76", "stage89", "stage93", "stage121", "stage133", "stage139", "stage142", "stage143", "stage144")
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
        parser.error("Use a separate Stage-145 directory")
    for name, path in zip(names, sources):
        if not path.is_dir():
            parser.error(f"Missing --{name} source directory: {path}")
    with writer_lock(root):
        if (root / "stage145_decision.json").exists():
            check_results(root)
        else:
            r143, r144 = sources[-2:]
            fixed.check_results(r143)
            prior.check_results(r144)
            for path, schema, fingerprint in ((r143/"stage143_decision.json",fixed.SCHEMA,prior.SOURCE143),
                                               (r144/"stage144_decision.json",prior.SCHEMA,SOURCE144)):
                if c.c66.sealed(path, schema)["fingerprint"] != fingerprint:
                    raise ValueError(f"Need the reviewed source run at {path}")
            regression = c.c66.sealed(r143 / "stage143_models.json", fixed.SCHEMA)["models"]
            ranks = c.c66.sealed(r144 / "stage144_models.json", prior.SCHEMA)["models"]
            native_context, replay_context = fixed.load_context(*sources[:-2])
            native = fixed.native_rows(*native_context, include_regions=True)
            replays = fixed.replay_rows(*replay_context, include_regions=True)
            entry_for, ap = native_context[3:5]
            run(root, native, replays, entry_for, ap["root"], r144, regression, ranks)
    print(f"Stage-145: {STATUS}", flush=True)


if __name__ == "__main__":
    main()
