"""Evaluate appearance-assisted choices after widening the fixed overlap limit."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import stage147_trigger_coverage as audit
from stage27_uav_collect import writer_lock
from stage46_shift_audit import publish_table


previous, appearance, fixed, prior, c, io = audit.previous, audit.appearance, audit.fixed, audit.prior, audit.c, audit.io
SCHEMA = "stage148_overlap_appearance_v1"
STATUS = "COMPLETE_WIDER_OVERLAP_APPEARANCE_DIAGNOSTIC"
SOURCE147 = "b7be991958d9aac3a98d3858bfcce8d5d551ec4fe8d23dd1eb1b511d05e569ba"
METHODS = previous.METHODS
DESIGN = dict(previous.DESIGN, max_box_overlap=.40)
MODES = ("frozen_thresholds", "recalibrated")
TABLES = ("policy", "paired", "actions", "funnel", "stable", "calibration", "focus", "scope", "frames")
WORK = dict(previous.WORK)


def read_table(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def state_key(row):
    return appearance.context_key(row), int(row["frame"])


def stage146_thresholds(rows, sources):
    selected = {}
    for row in rows:
        if row["chosen"] != "True":
            continue
        key = row["held_source"], row["arm"]
        if key in selected or set(row["calibration_sources"].split("|")) != sources - {key[0]}:
            raise ValueError("Stage-146 threshold provenance changed")
        selected[key] = float(row["threshold"]) if row["threshold"] else None
    if set(selected) != {(g, arm) for g in sources for arm in METHODS}:
        raise ValueError("Missing Stage-146 threshold selections")
    return {g:{a:selected[g, a] for a in METHODS} for g in sources}


def frozen_menus(r147):
    states = read_table(r147 / "stage147_states.csv")
    state_ids = {int(row["state_id"]):state_key(row) for row in states}
    menus = {key:([], []) for key in state_ids.values()}
    if len(menus) != len(state_ids) or len(states) != len(state_ids):
        raise ValueError("Repeated Stage-147 state")
    with (r147 / "stage147_candidates.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if float(row["quality_gap"]) > DESIGN["max_quality_gap"] or float(row["box_overlap"]) > DESIGN["max_box_overlap"]:
                continue
            geometry, matched = menus[state_ids[int(row["state_id"])]]
            cell = int(row["cell"])
            geometry.append(cell)
            if row["recent_color"] == row["template_gray"] == "True":
                matched.append(cell)
    return menus


def audit_archived(decision, reference):
    # CSV preserves missing margins as empty fields.
    missing = {key for key in reference if decision[key] is None}
    if any(reference[key] for key in missing):
        raise ValueError("Archived Stage-146 missing margins changed")
    prior.audit_reference(decision, {k:v for k,v in reference.items() if k not in missing})


def evaluate(case, model, thresholds, reference, expected):
    row, baseline, _ = case
    old = previous.evaluate(case, model, thresholds["frozen_thresholds"])
    audit_archived(old, reference)
    menu = previous.evidence(case, model, design=DESIGN)
    cells = row["cells"]
    if (cells[menu["geometry"]].tolist(), cells[menu["eligible"]].tolist()) != expected:
        raise ValueError("Stage-147 widened candidate menu disagrees")
    matched = previous.proposals(case, menu)
    return [dict(previous.evaluate(case, model, limits, design=DESIGN), threshold_mode=mode,
                 new_geometry_candidates=matched["geometry_candidates"] - old["geometry_candidates"],
                 new_matched_candidates=matched["matched_candidates"] - old["matched_candidates"],
                 previous_trigger_oracle_iou=old["dense_trigger_oracle_iou"],
                 **{f"{a}_threshold":limits[a] for a in METHODS}) for mode, limits in thresholds.items()]


def funnel(decisions):
    good, low = DESIGN["good"], DESIGN["low"]
    populations = defaultdict(list)
    for row in decisions:
        groups = ["native_sample" if row["state"]=="native" else "branch_replays"]
        if row["onset"]:
            groups.append("native_onsets")
        if row["dense_only_good"]:
            groups.append("dense_only_good")
        if row["stable_window"]:
            groups.append("native_stable_windows")
        if row["state"]=="native" and 0.<=row["native_iou"]<low and row["grid_iou"]>=good:
            groups.append("native_low_grid_good")
        for name in groups:
            populations[name].append(row)
    output = []
    for name, rows in sorted(populations.items()):
        for arm in METHODS:
            available = [r for r in rows if r["anchor_iou"] < good <= r["dense_trigger_oracle_iou"]]
            ranked = [r for r in available if r[f"{arm}_best_iou"] >= good]
            positive = [r for r in ranked if r[f"{arm}_margin"] > 0.]
            output.append(dict(population=name, arm=arm, states=len(rows),
                               newly_accessible=sum(r["previous_trigger_oracle_iou"] < good for r in available),
                               accessible=len(available), scored_first=len(ranked), positive_margin=len(positive),
                               accepted_good=sum(r[f"dense_{arm}_switch"] for r in positive),
                               retained_default_good=sum(r["anchor_iou"] >= good and r[f"dense_{arm}_iou"] >= good for r in rows)))
    return output


def run(root, native, replays, entry_for, image_root, roots, baselines, rankers, reader=appearance.checked_image):
    r144, r146, r147 = roots
    observed, scope = previous.collect_observations(native, replays, entry_for, image_root, r144, baselines, rankers, reader)
    by_source = defaultdict(list)
    keys = {state_key(row) for row, _, _ in observed}
    menus = frozen_menus(r147)
    reference = appearance.reference_rows(r146 / "stage146_frames.csv", keys)
    if len(keys)!=4754 or len(observed)!=len(keys) or set(menus)!=keys or len(scope)!=28:
        raise ValueError("Stage-147 study roster changed")
    for case in observed:
        by_source[case[0]["source"]].append(case)
    if set(by_source) != set(native):
        raise ValueError("Stage-147 source roster changed")
    old = stage146_thresholds(read_table(r146 / "stage146_calibration.csv"), set(by_source))
    decisions, calibration = [], []
    for held, adjusted, trials in previous.calibrate_sources(by_source, native, baselines, design=DESIGN):
        thresholds = dict(frozen_thresholds=old[held], recalibrated=adjusted)
        calibration.extend(dict(row, previous_threshold=old[held][row["arm"]]) for row in trials)
        for case in by_source[held]:
            key = state_key(case[0])
            decisions.extend(evaluate(case, baselines[held], thresholds, reference[key], menus[key]))
        print(f"[Stage-148] {held}: {len(by_source[held])} held states", flush=True)
    decisions.sort(key=lambda r:(r["threshold_mode"], *state_key(r)))
    output_policy, output_paired, actions, stages, windows = [], [], [], [], []
    comparisons = tuple((arm,"quality") for arm in METHODS) + tuple((arm,METHODS[0]) for arm in METHODS[1:])
    for mode in MODES:
        subset = [r for r in decisions if r["threshold_mode"]==mode]
        policy, paired = fixed.summarize(subset, arms=("quality","oracle","trigger_oracle",*METHODS),
                                        comparisons=comparisons, design=DESIGN, pools=("dense",))
        for row in policy + paired:
            if row["population"] == "native_all":
                row["population"] = "native_sample"
        output_policy.extend(dict(row, threshold_mode=mode) for row in policy)
        output_paired.extend(dict(row, threshold_mode=mode) for row in paired)
        actions.extend(dict(row, threshold_mode=mode) for row in previous.sample_actions(subset))
        stages.extend(dict(row, threshold_mode=mode) for row in funnel(subset))
        stability = fixed.stable_report(subset, arms=("quality",*METHODS), pools=("dense",))
        if len(stability) != 110:
            raise ValueError("Stage-147 stable window roster changed")
        windows.extend(dict(row, threshold_mode=mode) for row in stability)
    focus = [r for r in decisions if r["onset"] or r["dense_only_good"] or r["anchor_iou"] < DESIGN["good"] <= r["dense_trigger_oracle_iou"] or (
        r["state"]=="replay" and r["source"]=="uav0000089" and r["frame"] in (1165,1166))]
    data = (output_policy, output_paired, actions, stages, windows, calibration, focus, scope, decisions)
    hashes = {}
    for name, rows in zip(TABLES, data):
        path = root / f"stage148_{name}.csv"
        publish_table(path, rows)
        hashes[path.name] = c.file_hash(path)
    files = (Path(__file__), Path(audit.__file__), Path(previous.__file__), Path(appearance.__file__), Path(fixed.__file__))
    runs = {mode:{a:sum(r[f"dense_{a}_low_run"] for r in windows if r["threshold_mode"]==mode) for a in METHODS} for mode in MODES}
    c.publish_identical(root / "stage148_decision.json", io.seal(dict(schema=SCHEMA, status=STATUS,
        role="consumed_development_frozen_wider_overlap_contrast", source_stage146=audit.SOURCE146,
        source_stage147=SOURCE147, design=DESIGN, states=len(keys), threshold_modes=list(MODES),
        sources=len(by_source), windows_per_mode=len(windows)//len(MODES), new_stable_low_run_windows=runs,
        image_reads=sum(r["image_reads"] for r in scope), code_sha256={p.name:c.file_hash(p) for p in files},
        table_sha256=hashes, **WORK)))
    check_results(root)


def check_results(root):
    report = c.c66.sealed(root / "stage148_decision.json", SCHEMA)
    if (report["status"] != STATUS or report["source_stage147"] != SOURCE147 or report["source_stage146"] != audit.SOURCE146
            or report["design"] != DESIGN or report["threshold_modes"] != list(MODES)
            or (report["states"],report["sources"],report["windows_per_mode"]) != (4754,28,110)
            or set(report["table_sha256"]) != {f"stage148_{k}.csv" for k in TABLES}
            or any(report[k] != v for k,v in WORK.items())):
        raise ValueError("Incomplete Stage-148 wider overlap audit")
    io.verify({str(root / k):v for k,v in report["table_sha256"].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76","stage89","stage93","stage121","stage133","stage139","stage142","stage143","stage144","stage146","stage147")
    for name in names:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-results", action="store_true")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if any(getattr(args,name) is None for name in names):
        parser.error("Supply all frozen source directories")
    sources = [getattr(args,name).expanduser().resolve() for name in names]
    if any(root == p or p in root.parents or root in p.parents for p in sources):
        parser.error("Use a separate Stage-148 directory")
    for name, path in zip(names, sources):
        if not path.is_dir():
            parser.error(f"Missing --{name} source directory: {path}")
    with writer_lock(root):
        if (root / "stage148_decision.json").exists():
            check_results(root)
        else:
            r143, r144, r146, r147 = sources[-4:]
            fixed.check_results(r143)
            prior.check_results(r144)
            previous.check_results(r146)
            audit.check_results(r147)
            for path, schema, fingerprint in ((r143/"stage143_decision.json",fixed.SCHEMA,prior.SOURCE143),
                                               (r144/"stage144_decision.json",prior.SCHEMA,appearance.SOURCE144),
                                               (r146/"stage146_decision.json",previous.SCHEMA,audit.SOURCE146),
                                               (r147/"stage147_decision.json",audit.SCHEMA,SOURCE147)):
                if c.c66.sealed(path, schema)["fingerprint"] != fingerprint:
                    raise ValueError(f"Need the reviewed source run at {path}")
            baselines = c.c66.sealed(r143 / "stage143_models.json", fixed.SCHEMA)["models"]
            rankers = c.c66.sealed(r144 / "stage144_models.json", prior.SCHEMA)["models"]
            native_context, replay_context = fixed.load_context(*sources[:-4])
            native = fixed.native_rows(*native_context, include_regions=True)
            replays = fixed.replay_rows(*replay_context, include_regions=True)
            entry_for, ap = native_context[3:5]
            run(root, native, replays, entry_for, ap["root"], (r144,r146,r147), baselines, rankers)
    print(f"Stage-148: {STATUS}", flush=True)


if __name__ == "__main__":
    main()
