"""Ablate acceptance score components on the fixed Stage-148 menu."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import stage148_overlap_appearance as wider
from stage27_uav_collect import writer_lock
from stage46_shift_audit import publish_table


previous, appearance, fixed, prior, c, io = wider.previous, wider.appearance, wider.fixed, wider.prior, wider.c, wider.io
SCHEMA = "stage149_score_ablation_v1"
STATUS = "COMPLETE_FIXED_MENU_SCORE_ABLATION_DIAGNOSTIC"
SOURCE148 = "63d5b107507c69f1cee9377e7300e28cf4db834866108ddc797959784cb7db72"
ABLATIONS = ("without_regression", "without_raw", "only")
FAMILIES = tuple(zip(previous.METHODS[1:], previous.CUES))
METHODS = previous.METHODS + tuple(f"{cue}_{ablation}" for _,cue in FAMILIES for ablation in ABLATIONS)
MODES = ("calibrated", "zero_threshold")
DESIGN = dict(wider.DESIGN, ablations=list(ABLATIONS), bootstrap_seed=23149)
TABLES = ("actions", "funnel", "policy", "paired", "stable", "selected", "calibration", "focus", "scope", "frames")
WORK = dict(previous.WORK)
RAW_INDEX = fixed.FEATURES.index("raw_over_peak")


def evidence(case, model, *, design=DESIGN):
    row, _, scores = case
    quality = fixed.score(model, row["x"])
    anchor, geometry, eligible, margins = previous.candidate_menu(row, scores, quality, design=design)
    regression_delta = quality[eligible] - quality[anchor]
    raw = row["x"][:, RAW_INDEX]
    raw_delta = design["raw_weight"] * (raw[eligible] - raw[anchor])
    for _, cue in FAMILIES:
        visual_delta = design["appearance_weight"] * (scores[cue][eligible] - scores[cue][anchor])
        margins[f"{cue}_without_regression"] = raw_delta + visual_delta
        margins[f"{cue}_without_raw"] = regression_delta + visual_delta
        margins[f"{cue}_only"] = visual_delta
    return dict(anchor=anchor, geometry=geometry, eligible=eligible, margins=margins)


def reference_rows(path, selected):
    reference = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["threshold_mode"] != "recalibrated":
                continue
            key = wider.state_key(row)
            if key in reference:
                raise ValueError("Duplicated Stage-148 recalibrated state")
            reference[key] = row
    if set(reference) != selected:
        raise ValueError("Stage-148 recalibrated state roster changed")
    return reference


def evaluate(case, model, calibrated, reference):
    _, baseline, _ = case
    proposal = previous.proposals(case, evidence(case, model))
    updated = previous.apply_thresholds(baseline, proposal, calibrated)
    audited = {k:reference[k] for k in updated if k in reference}
    for arm in previous.METHODS:
        value = calibrated[arm]
        archived = reference[f"{arm}_threshold"]
        if (float(archived) if archived else None) != value:
            raise ValueError(f"Stage-148 recalibration changed for {baseline['source']} / {arm}")
    wider.audit_archived(updated, audited)
    zero = dict.fromkeys(METHODS, 0.)
    return [dict(result, threshold_mode=mode, previous_trigger_oracle_iou=proposal["dense_trigger_oracle_iou"],
                 **{f"{arm}_threshold":limits[arm] for arm in METHODS})
            for mode, limits, result in ((MODES[0],calibrated,updated),(MODES[1],zero,previous.apply_thresholds(baseline,proposal,zero)))]


def run(root, native, replays, entry_for, image_root, roots, baselines, rankers, reader=appearance.checked_image):
    r144, r148 = roots
    observations, scope = previous.collect_observations(native, replays, entry_for, image_root, r144, baselines, rankers, reader)
    keys = {wider.state_key(row) for row, _, _ in observations}
    reference = reference_rows(r148 / "stage148_frames.csv", keys)
    if len(keys) != 4754 or len(observations) != len(keys) or len(scope) != 28:
        raise ValueError("Stage-148 study roster changed")
    by_source = defaultdict(list)
    for case in observations:
        by_source[case[0]["source"]].append(case)
    if set(by_source) != set(native):
        raise ValueError("Stage-148 source roster changed")
    decisions, calibration = [], []
    for held, thresholds, trials in previous.calibrate_sources(by_source, native, baselines, design=DESIGN,
                                                                methods=METHODS, build_evidence=evidence):
        calibration.extend(trials)
        for case in by_source[held]:
            decisions.extend(evaluate(case, baselines[held], thresholds, reference[wider.state_key(case[0])]))
        print(f"[Stage-149] {held}: {len(by_source[held])} held states", flush=True)
    decisions.sort(key=lambda r:(r["threshold_mode"], *wider.state_key(r)))
    actions, stages, policies, paired, stable = [], [], [], [], []
    comparisons = tuple((arm,"quality") for arm in METHODS) + tuple(
        (f"{cue}_{ablation}", full) for full,cue in FAMILIES for ablation in ABLATIONS)
    for mode in MODES:
        subset = [r for r in decisions if r["threshold_mode"]==mode]
        policy, differences = fixed.summarize(subset, arms=("quality","oracle","trigger_oracle",*METHODS),
                                              comparisons=comparisons, design=DESIGN, pools=("dense",))
        for row in policy + differences:
            if row["population"] == "native_all":
                row["population"] = "native_sample"
        windows = fixed.stable_report(subset, arms=("quality",*METHODS), pools=("dense",))
        if len(windows) != 110:
            raise ValueError("Stage-148 stability roster changed")
        actions.extend(dict(row, threshold_mode=mode) for row in previous.sample_actions(subset, methods=METHODS))
        stages.extend(dict(row, threshold_mode=mode) for row in wider.funnel(subset, methods=METHODS))
        policies.extend(dict(row, threshold_mode=mode) for row in policy)
        paired.extend(dict(row, threshold_mode=mode) for row in differences)
        stable.extend(dict(row, threshold_mode=mode) for row in windows)
    focus = [r for r in decisions if r["onset"] or r["dense_only_good"] or r["anchor_iou"] < DESIGN["good"] <= r["dense_trigger_oracle_iou"] or (
        r["state"]=="replay" and r["source"]=="uav0000089" and r["frame"] in (1165,1166))]
    selected = [r for r in calibration if r["chosen"]]
    data = (actions, stages, policies, paired, stable, selected, calibration, focus, scope, decisions)
    hashes = {}
    for name, rows in zip(TABLES, data):
        path = root / f"stage149_{name}.csv"
        publish_table(path, rows)
        hashes[path.name] = c.file_hash(path)
    files = (Path(__file__), Path(wider.__file__), Path(previous.__file__), Path(appearance.__file__), Path(fixed.__file__))
    runs = {mode:{a:sum(r[f"dense_{a}_low_run"] for r in stable if r["threshold_mode"]==mode) for a in METHODS} for mode in MODES}
    c.publish_identical(root / "stage149_decision.json", io.seal(dict(schema=SCHEMA, status=STATUS,
        role="consumed_development_fixed_menu_acceptance_score_ablation", source_stage148=SOURCE148,
        design=DESIGN, methods=list(METHODS), threshold_modes=list(MODES), states=len(keys),
        sources=len(by_source), windows_per_mode=len(stable)//len(MODES), new_stable_low_run_windows=runs,
        image_reads=sum(r["image_reads"] for r in scope), code_sha256={p.name:c.file_hash(p) for p in files},
        table_sha256=hashes, **WORK)))
    check_results(root)


def check_results(root):
    report = c.c66.sealed(root / "stage149_decision.json", SCHEMA)
    if (report["status"] != STATUS or report["source_stage148"] != SOURCE148 or report["design"] != DESIGN
            or report["methods"] != list(METHODS) or report["threshold_modes"] != list(MODES)
            or (report["states"],report["sources"],report["windows_per_mode"]) != (4754,28,110)
            or set(report["table_sha256"]) != {f"stage149_{k}.csv" for k in TABLES}
            or any(report[k] != v for k,v in WORK.items())):
        raise ValueError("Incomplete Stage-149 score ablation audit")
    io.verify({str(root / k):v for k,v in report["table_sha256"].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76","stage89","stage93","stage121","stage133","stage139","stage142","stage143","stage144","stage148")
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
        parser.error("Use a separate Stage-149 directory")
    for name, path in zip(names, sources):
        if not path.is_dir():
            parser.error(f"Missing --{name} source directory: {path}")
    with writer_lock(root):
        if (root / "stage149_decision.json").exists():
            check_results(root)
        else:
            r143,r144,r148 = sources[-3:]
            fixed.check_results(r143)
            prior.check_results(r144)
            wider.check_results(r148)
            for path,schema,fingerprint in ((r143/"stage143_decision.json",fixed.SCHEMA,prior.SOURCE143),
                                             (r144/"stage144_decision.json",prior.SCHEMA,appearance.SOURCE144),
                                             (r148/"stage148_decision.json",wider.SCHEMA,SOURCE148)):
                if c.c66.sealed(path,schema)["fingerprint"] != fingerprint:
                    raise ValueError(f"Need the reviewed source run at {path}")
            baselines = c.c66.sealed(r143/"stage143_models.json",fixed.SCHEMA)["models"]
            rankers = c.c66.sealed(r144/"stage144_models.json",prior.SCHEMA)["models"]
            native_context,replay_context = fixed.load_context(*sources[:-3])
            native = fixed.native_rows(*native_context,include_regions=True)
            replays = fixed.replay_rows(*replay_context,include_regions=True)
            entry_for,ap = native_context[3:5]
            run(root,native,replays,entry_for,ap["root"],(r144,r148),baselines,rankers)
    print(f"Stage-149: {STATUS}", flush=True)


if __name__ == "__main__":
    main()
