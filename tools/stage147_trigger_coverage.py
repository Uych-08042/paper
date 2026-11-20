"""Measure which fixed trigger rules exclude viable alternatives to the default box."""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np

import stage146_conflict_appearance as previous
from stage27_uav_collect import writer_lock
from stage28_response_common import continuous_iou
from stage46_shift_audit import publish_table


appearance, fixed, prior, c, io = previous.appearance, previous.fixed, previous.prior, previous.c, previous.io
SCHEMA = "stage147_trigger_coverage_v1"
STATUS = "COMPLETE_FROZEN_TRIGGER_COVERAGE_AUDIT"
SOURCE146 = "4136d0ee282cd85f2fbbbd3b75f676fdafd727d03f939d8d78494a6334db82b3"
DESIGN = dict(good=fixed.DESIGN["good"], low=fixed.DESIGN["low"],
              original_gap=previous.DESIGN["max_quality_gap"], original_overlap=previous.DESIGN["max_box_overlap"],
              gap_limits=[.05, .15, .25, .40, .60, None], overlap_limits=[.10, .25, .40, None],
              appearance_requirements=["none", "recent_color", "template_gray", "both"])
TABLES = ("coverage", "exclusions", "focus", "states", "candidates", "scope")
WORK = dict(new_search_calls=0, state_updates=0, sgla_weight_updates=0, fitted_models=0, new_policy_evaluations=0)


def blocked_by(gap, overlap):
    gap_failed = gap > DESIGN["original_gap"]
    overlap_failed = overlap > DESIGN["original_overlap"]
    if gap_failed and overlap_failed:
        return "both"
    if gap_failed:
        return "gap_only"
    return "overlap_only" if overlap_failed else "neither"


def alternatives(state_id, row, baseline, scores, regressor):
    pool = row["pools"]["dense"]
    q = fixed.score(regressor, row["x"])
    anchor = int(pool[q[pool].argmax()])
    idx = pool[pool != anchor]
    boxes, labels, cells = row["boxes"], row["y"], row["cells"]
    if int(cells[anchor]) != baseline["dense_quality_cell"]:
        raise ValueError("Frozen default choice changed")
    gaps = q[anchor] - q[idx]
    overlaps = continuous_iou(boxes[anchor], boxes[idx])
    available = [np.isfinite(scores[k][anchor]) & np.isfinite(scores[k][idx]) for k in previous.CUES]
    good, low = DESIGN["good"], DESIGN["low"]
    state = {k:baseline[k] for k in ("source", "sequence", "frame", "state", "case", "branch", "onset", "relapse",
                                    "stable_window", "dense_only_good", "native_iou", "grid_iou", "dense_oracle_iou")}
    state.update(state_id=state_id, anchor_cell=int(cells[anchor]), anchor_iou=float(labels[anchor]),
                 alternatives=len(idx))
    records = []
    for index, gap, overlap, recent, template in zip(idx, gaps, overlaps, *available):
        iou = float(labels[index])
        stable_good = bool(state["stable_window"]) and state["anchor_iou"] >= good
        records.append(dict(state_id=state_id, source=state["source"], sequence=state["sequence"], frame=state["frame"],
                            state=state["state"], case=state["case"], branch=state["branch"],
                            anchor_cell=state["anchor_cell"], anchor_iou=state["anchor_iou"],
                            cell=int(cells[index]), candidate_iou=iou, quality_gap=float(gap), box_overlap=float(overlap),
                            original_exclusion=blocked_by(gap, overlap), recent_color=bool(recent),
                            template_gray=bool(template), new_good=bool(0. <= state["anchor_iou"] < good <= iou),
                            stable_bad=bool(stable_good and iou < good), stable_severe=bool(stable_good and iou < low)))
    return state, records


def audit_original(state, rows, reference):
    geometry = [r for r in rows if r["original_exclusion"] == "neither"]
    matched = [r for r in geometry if r["recent_color"] and r["template_gray"]]
    audit = dict(state, geometry_candidates=len(geometry), matched_candidates=len(matched),
                 matched_cells="|".join(str(r["cell"]) for r in matched),
                 geometry_oracle_iou=max([state["anchor_iou"], *(r["candidate_iou"] for r in geometry)]),
                 dense_trigger_oracle_iou=max([state["anchor_iou"], *(r["candidate_iou"] for r in matched)]))
    prior.audit_reference(audit, {k:reference[k] for k in (
        "anchor_cell", "anchor_iou", "geometry_candidates", "matched_candidates", "matched_cells",
        "geometry_oracle_iou", "dense_trigger_oracle_iou")})


def populations(states):
    native = np.array([r["state"] == "native" for r in states])
    low = np.array([0. <= r["native_iou"] < DESIGN["low"] for r in states])
    grid_good = np.array([r["grid_iou"] >= DESIGN["good"] for r in states])
    return dict(native_sample=native, native_low_grid_good=native & low & grid_good,
                native_onsets=np.array([r["onset"] for r in states]),
                dense_only_good=np.array([r["dense_only_good"] for r in states]),
                native_stable_windows=np.array([bool(r["stable_window"]) for r in states]),
                branch_replays=~native, relapse_replays=np.array([r["relapse"] for r in states]))


def threshold_masks(gaps, overlaps, recent, template):
    support = {"none":np.ones(len(gaps), dtype=bool), "recent_color":recent,
               "template_gray":template, "both":recent & template}
    for gap, overlap, cue in product(DESIGN["gap_limits"], DESIGN["overlap_limits"], DESIGN["appearance_requirements"]):
        keep = support[cue].copy()
        if gap is not None:
            keep &= gaps <= gap
        if overlap is not None:
            keep &= overlaps <= overlap
        yield gap, overlap, cue, keep


def counts(keep, state_ids, sources, event_masks):
    result = dict(allowed_candidates=int(keep.sum()), allowed_states=int(np.unique(state_ids[keep]).size))
    for event, universe in event_masks.items():
        mask = keep & universe
        result.update({f"{event}_candidates":int(mask.sum()), f"{event}_states":int(np.unique(state_ids[mask]).size),
                       f"{event}_sources":int(np.unique(sources[mask]).size)})
    return result


def summarize(states, candidates):
    if [r["state_id"] for r in states] != list(range(len(states))):
        raise ValueError("Candidate table does not index the frozen states")
    ids = np.array([r["state_id"] for r in candidates], dtype=int)
    sources = np.array([r["source"] for r in candidates], dtype=str)
    floats = [np.array([r[k] for r in candidates], dtype=float) for k in ("quality_gap", "box_overlap")]
    flags = [np.array([r[k] for r in candidates], dtype=bool) for k in ("recent_color", "template_gray")]
    events = {k:np.array([r[k] for r in candidates], dtype=bool) for k in ("new_good", "stable_bad", "stable_severe")}
    masks = list(threshold_masks(*floats, *flags))
    exclusions = np.array([r["original_exclusion"] for r in candidates], dtype=str)
    coverage, breakdown = [], []
    for group, members in populations(states).items():
        universe = members[ids]
        baseline = dict(population=group, states=int(members.sum()),
                        possible_new_good_states=int(np.unique(ids[universe & events["new_good"]]).size))
        for g, o, cue, admitted in masks:
            coverage.append(dict(baseline, gap_limit=g, overlap_limit=o, appearance=cue,
                                 **counts(universe & admitted, ids, sources, events)))
        for reason in ("gap_only", "overlap_only", "both", "neither"):
            breakdown.append(dict(baseline, original_exclusion=reason,
                                  **counts(universe & (exclusions == reason), ids, sources, events)))
    return coverage, breakdown


def run(root, native, replays, entry_for, image_root, roots, baselines, rankers, reader=appearance.checked_image):
    r144, r146 = roots
    observations = []

    def observe(row, decision, scores):
        observations.append((row, decision, {k:scores[k] for k in previous.CUES}))

    _, _, scope = appearance.collect_study(native, replays, entry_for, image_root, r144, baselines, rankers,
                                           reader, observe=observe)
    keys = {(appearance.context_key(r), r["frame"]) for r, _, _ in observations}
    reference = appearance.reference_rows(r146 / "stage146_frames.csv", keys)
    if len(keys) != len(observations) or len(keys) != 4754:
        raise ValueError("Reviewed Stage-146 state roster changed")
    states, candidates, focus = [], [], []
    for i, (row, baseline, scores) in enumerate(observations):
        ref = reference[appearance.context_key(row), row["frame"]]
        prior.audit_reference(baseline, {k:ref[k] for k in baseline})
        state, records = alternatives(i, row, baseline, scores, baselines[row["source"]])
        audit_original(state, records, ref)
        states.append(state)
        candidates.extend(records)
        if state["onset"] or state["dense_only_good"] or (
                state["state"] == "replay" and state["source"] == "uav0000089" and state["frame"] in (1165, 1166)):
            focus.extend(records)
    coverage, exclusions = summarize(states, candidates)
    stable_windows = {(r["sequence"], r["stable_window"]) for r in states if r["stable_window"]}
    if len(scope) != 28 or len(stable_windows) != 110:
        raise ValueError("Reviewed Stage-146 source or stability support changed")
    hashes = {}
    for name, values in zip(TABLES, (coverage, exclusions, focus, states, candidates, scope)):
        path = root / f"stage147_{name}.csv"
        publish_table(path, values)
        hashes[path.name] = c.file_hash(path)
    files = (Path(__file__), Path(previous.__file__), Path(appearance.__file__), Path(fixed.__file__), Path(prior.__file__))
    c.publish_identical(root / "stage147_decision.json", io.seal(dict(schema=SCHEMA, status=STATUS,
        role="consumed_development_candidate_admission_audit", source_stage146=SOURCE146, design=DESIGN,
        evaluated_states=len(states), evaluated_alternatives=len(candidates), evaluated_sources=len(scope),
        sampled_stable_windows=len(stable_windows), image_reads=sum(r["image_reads"] for r in scope),
        code_sha256={p.name:c.file_hash(p) for p in files}, table_sha256=hashes, **WORK)))
    check_results(root)


def check_results(root):
    report = c.c66.sealed(root / "stage147_decision.json", SCHEMA)
    if (report["status"] != STATUS or report["source_stage146"] != SOURCE146 or report["design"] != DESIGN
            or (report["evaluated_states"], report["evaluated_sources"], report["sampled_stable_windows"]) != (4754,28,110)
            or set(report["table_sha256"]) != {f"stage147_{k}.csv" for k in TABLES}
            or any(report[k] != v for k,v in WORK.items())):
        raise ValueError("Incomplete Stage-147 trigger coverage audit")
    io.verify({str(root / k):v for k,v in report["table_sha256"].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76", "stage89", "stage93", "stage121", "stage133", "stage139", "stage142", "stage143", "stage144", "stage146")
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
        parser.error("Use a separate Stage-147 directory")
    for name, path in zip(names, sources):
        if not path.is_dir():
            parser.error(f"Missing --{name} source directory: {path}")
    with writer_lock(root):
        if (root / "stage147_decision.json").exists():
            check_results(root)
        else:
            r143, r144, r146 = sources[-3:]
            fixed.check_results(r143)
            prior.check_results(r144)
            previous.check_results(r146)
            for path, schema, fingerprint in ((r143/"stage143_decision.json",fixed.SCHEMA,prior.SOURCE143),
                                               (r144/"stage144_decision.json",prior.SCHEMA,appearance.SOURCE144),
                                               (r146/"stage146_decision.json",previous.SCHEMA,SOURCE146)):
                if c.c66.sealed(path, schema)["fingerprint"] != fingerprint:
                    raise ValueError(f"Need the reviewed source run at {path}")
            baselines = c.c66.sealed(r143 / "stage143_models.json", fixed.SCHEMA)["models"]
            rankers = c.c66.sealed(r144 / "stage144_models.json", prior.SCHEMA)["models"]
            native_context, replay_context = fixed.load_context(*sources[:-3])
            native = fixed.native_rows(*native_context, include_regions=True)
            replays = fixed.replay_rows(*replay_context, include_regions=True)
            entry_for, ap = native_context[3:5]
            run(root, native, replays, entry_for, ap["root"], (r144,r146), baselines, rankers)
    print(f"Stage-147: {STATUS}", flush=True)


if __name__ == "__main__":
    main()
