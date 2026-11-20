"""Calibrate appearance-assisted switches on the same frozen conflict candidates."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import stage145_fixed_pool_appearance as appearance
from stage27_uav_collect import writer_lock
from stage28_response_common import continuous_iou
from stage46_shift_audit import publish_table


fixed, prior, c, io = appearance.fixed, appearance.prior, appearance.c, appearance.io
SCHEMA = "stage146_conflict_appearance_v1"
STATUS = "COMPLETE_MATCHED_CONFLICT_APPEARANCE_DIAGNOSTIC"
SOURCE145 = "5d3d4a306f75d63b957392f530a70c13b1f2f0cbb41f0fcd0f288f35db36ff28"
METHODS = ("response_geometry", "with_recent_color", "with_template_gray")
CUES = ("recent_hist", "template_ncc")
DESIGN = dict(fixed.DESIGN, max_quality_gap=.15, max_box_overlap=.25, raw_weight=.10, appearance_weight=.10,
              acceptance_grid=[None,.25,.15,.10,.05,.025,.01,0.], bootstrap_seed=23146)
TABLES = ("policy", "paired", "actions", "calibration", "stable", "scope", "focus", "frames")
WORK = dict(new_search_calls=0, state_updates=0, sgla_weight_updates=0)


def candidate_menu(row, scores, anchor_scores):
    """Return one common menu and three fixed score deltas. No labels enter this step."""
    pool = row["pools"]["dense"]
    anchor = int(pool[anchor_scores[pool].argmax()])
    gap = anchor_scores[anchor] - anchor_scores[pool]
    overlap = continuous_iou(row["boxes"][anchor], row["boxes"][pool])
    geometry = pool[(pool != anchor) & (gap <= DESIGN["max_quality_gap"]) & (overlap <= DESIGN["max_box_overlap"])]
    valid = np.logical_and.reduce([np.isfinite(scores[k]) for k in CUES])
    eligible = geometry[valid[geometry]] if valid[anchor] else geometry[:0]
    normalized_raw = row["x"][:, fixed.FEATURES.index("raw_over_peak")]
    base = anchor_scores[eligible] - anchor_scores[anchor] + DESIGN["raw_weight"] * (
        normalized_raw[eligible] - normalized_raw[anchor])
    gains = [base]
    gains.extend(base + DESIGN["appearance_weight"] * (scores[k][eligible] - scores[k][anchor]) for k in CUES)
    return anchor, geometry, eligible, dict(zip(METHODS, gains))


def evidence(case, model):
    row, _, scores = case
    anchor, geometry, eligible, margins = candidate_menu(row, scores, fixed.score(model, row["x"]))
    return dict(anchor=anchor, geometry=geometry, eligible=eligible, margins=margins)


def proposals(case, menu):
    row, decision, _ = case
    anchor = menu["anchor"]
    geometry = menu["geometry"]
    eligible = menu["eligible"]
    labels = row["y"]
    candidate_cells = row["cells"]
    values = dict(source=decision["source"], anchor_cell=int(candidate_cells[anchor]), anchor_iou=float(labels[anchor]),
                  native_iou=row["native_iou"], stable_window=decision["stable_window"], onset=decision["onset"],
                  geometry_candidates=len(geometry), matched_candidates=len(eligible),
                  matched_cells="|".join(map(str, candidate_cells[eligible])),
                  geometry_oracle_iou=float(labels[np.r_[anchor, geometry]].max()),
                  dense_trigger_oracle_iou=float(labels[np.r_[anchor, eligible]].max()))
    for arm, scores in menu["margins"].items():
        if len(scores):
            winner = int(np.argmax(scores))
            idx = eligible[winner]
            values[arm+"_best_cell"] = int(candidate_cells[idx])
            values[arm+"_best_iou"] = float(labels[idx])
            values[arm+"_margin"] = float(scores[winner])
        else:
            values[arm+"_best_cell"] = values["anchor_cell"]
            values[arm+"_best_iou"] = values["anchor_iou"]
            values[arm+"_margin"] = None
    return values


def calibrate(rows, arm):
    """Select only from training-source shadow outcomes, keeping on tied utility."""
    groups = defaultdict(list)
    for row in rows:
        if row["anchor_iou"] >= 0.:
            groups[row["source"]].append(row)
    if not groups:
        raise ValueError("No calibration states from peer sources")
    segments = []
    for local in groups.values():
        incumbent = np.array([r["anchor_iou"] for r in local])
        proposal = np.maximum(0., [r[arm+"_best_iou"] for r in local])
        margin = np.array([r[arm+"_margin"] if r[arm+"_margin"] is not None else -np.inf for r in local])
        stable = np.array([bool(r["stable_window"]) for r in local]) & (incumbent >= DESIGN["good"])
        rescued = np.array([r["onset"] for r in local], bool) & (incumbent >= DESIGN["good"])
        segments.append((incumbent, proposal, margin, stable, rescued))
    results = []
    for threshold in DESIGN["acceptance_grid"]:
        limit = np.inf if threshold is None else threshold
        improvements, switches, harms, losses = [], 0, 0, 0
        for incumbent, proposal, margin, stable, rescued in segments:
            accepted = margin > limit
            iou = np.where(accepted, proposal, incumbent)
            improvements.append(float((iou-incumbent).mean()))
            switches += int(accepted.sum())
            harms += int((stable & (iou < DESIGN["low"])).sum())
            losses += int((rescued & (iou < DESIGN["good"])).sum())
        results.append(dict(threshold=threshold, calibration_mean_iou_delta=float(np.mean(improvements)),
                            calibration_switches=switches, calibration_stable_severe_harms=harms,
                            calibration_onset_losses=losses))
    best = results[0]
    for result in results[1:]:
        if (result["calibration_stable_severe_harms"] == result["calibration_onset_losses"] == 0
                and result["calibration_mean_iou_delta"] > best["calibration_mean_iou_delta"] + 1e-12):
            best = result
    return best, results


def evaluate(case, model, thresholds):
    _, baseline, _ = case
    values = proposals(case, evidence(case, model))
    if values["anchor_cell"] != baseline["dense_quality_cell"]:
        raise ValueError("Stage-145 default selection changed")
    decision = dict(baseline, **{k:v for k,v in values.items() if k not in baseline})
    for arm, limit in thresholds.items():
        margin = values[arm+"_margin"]
        accepted = bool(limit is not None and margin is not None and margin > limit)
        decision[f"dense_{arm}_cell"] = values[arm+"_best_cell"] if accepted else values["anchor_cell"]
        decision[f"dense_{arm}_iou"] = values[arm+"_best_iou"] if accepted else values["anchor_iou"]
        decision[f"dense_{arm}_switch"] = accepted
    return decision


def sample_actions(decisions):
    good, low = DESIGN["good"], DESIGN["low"]
    native = [r for r in decisions if r["state"] == "native"]
    samples = dict(native_sample=native, native_conflicts=[r for r in native if r["matched_candidates"]],
                   native_low_grid_good=[r for r in native if 0. <= r["native_iou"] < low and r["grid_iou"] >= good],
                   native_onsets=[r for r in native if r["onset"]], dense_only_good=[r for r in native if r["dense_only_good"]],
                   native_stable_windows=[r for r in native if r["stable_window"]],
                   branch_replays=[r for r in decisions if r["state"] == "replay"])
    result = []
    for name, part in samples.items():
        for arm in METHODS:
            result.append(dict(population=name, arm=arm, states=len(part),
                               geometry_conflicts=sum(r["geometry_candidates"] > 0 for r in part),
                               matched_conflicts=sum(r["matched_candidates"] > 0 for r in part),
                               geometry_new_good=sum(r["anchor_iou"] < good <= r["geometry_oracle_iou"] for r in part),
                               accessible_new_good=sum(r["anchor_iou"] < good <= r["dense_trigger_oracle_iou"] for r in part),
                               switches=sum(r[f"dense_{arm}_switch"] for r in part),
                               gained_good=sum(r["anchor_iou"] < good <= r[f"dense_{arm}_iou"] for r in part),
                               lost_good=sum(r["anchor_iou"] >= good > r[f"dense_{arm}_iou"] for r in part),
                               new_severe=sum(r["anchor_iou"] >= good and r[f"dense_{arm}_iou"] < low for r in part)))
    return result


def run(root, native, replays, entry_for, image_root, roots, baselines, rankers, reader=appearance.checked_image):
    r144, r145 = roots
    observations = []

    def observe(row, decision, scores):
        observations.append((row, decision, {k:scores[k] for k in CUES}))

    _, _, scope = appearance.collect_study(native, replays, entry_for, image_root, r144, baselines, rankers,
                                           reader, observe=observe)
    keys = {(appearance.context_key(r), r["frame"]) for r, _, _ in observations}
    reference = appearance.reference_rows(r145 / "stage145_frames.csv", keys)
    if len(keys) != len(observations):
        raise ValueError("Duplicated appearance states")
    by_source = defaultdict(list)
    for row, found, cues in observations:
        old = reference[appearance.context_key(row), row["frame"]]
        prior.audit_reference(found, old)
        by_source[row["source"]].append((row, found, cues))
    if set(by_source) != set(native) or len(observations) != 4754:
        raise ValueError("Stage-145 study roster changed")
    moments = {group:fixed.moments(rows) for group, rows in native.items()}
    decisions, calibration = [], []
    for held in sorted(by_source):
        # Both the evaluation source and each calibration source are excluded
        # from the provider used to generate that calibration source's actions.
        outer = {g:m for g,m in moments.items() if g != held}
        if baselines[held]["fit_sources"] != sorted(outer):
            raise ValueError("Unexpected Stage-143 outer split")
        train = []
        for peer in sorted(outer):
            model = fixed.fit_peer_sources(outer, peer)
            train.extend(proposals(case, evidence(case, model)) for case in by_source[peer] if case[0]["state"] == "native")
        thresholds = {}
        for arm in METHODS:
            chosen, grid = calibrate(train, arm)
            thresholds[arm] = chosen["threshold"]
            calibration.extend(dict(held_source=held, arm=arm, calibration_sources="|".join(sorted(outer)),
                                    calibration_states=len(train), chosen=trial is chosen, **trial) for trial in grid)
        decisions.extend(evaluate(case, baselines[held], thresholds) for case in by_source[held])
        print(f"[Stage-146] {held}: {len(by_source[held])} held states", flush=True)
    decisions.sort(key=lambda r:(appearance.context_key(r), r["frame"]))
    comparisons = tuple((arm, "quality") for arm in METHODS) + tuple((arm, METHODS[0]) for arm in METHODS[1:])
    policy, paired = fixed.summarize(decisions, arms=("hann","raw","quality","oracle","trigger_oracle",*METHODS),
                                    comparisons=comparisons, design=DESIGN, pools=("dense",))
    for row in policy + paired:
        if row["population"] == "native_all":
            row["population"] = "native_sample"
    stable = fixed.stable_report(decisions, arms=("quality",*METHODS), pools=("dense",))
    if len(stable) != 110:
        raise ValueError("Stage-145 stable windows changed")
    focus = [r for r in decisions if r["onset"] or r["dense_only_good"] or (
        r["state"] == "replay" and r["source"] == "uav0000089" and r["frame"] in (1165,1166))]
    hashes = {}
    for name, rows in zip(TABLES, (policy, paired, sample_actions(decisions), calibration, stable, scope, focus, decisions)):
        path = root / f"stage146_{name}.csv"
        publish_table(path, rows)
        hashes[path.name] = c.file_hash(path)
    files = (Path(__file__), Path(appearance.__file__), Path(fixed.__file__), Path(prior.__file__))
    c.publish_identical(root / "stage146_decision.json", io.seal(dict(schema=SCHEMA, status=STATUS,
        role="consumed_development_matched_conflict_selection", source_stage145=SOURCE145, design=DESIGN,
        evaluated_states=len(decisions), held_out_sources=len(by_source), sampled_stable_windows=len(stable),
        new_stable_low_run_windows={a:sum(r[f"dense_{a}_low_run"] for r in stable) for a in METHODS},
        image_reads=sum(r["image_reads"] for r in scope), code_sha256={path.name:c.file_hash(path) for path in files},
        table_sha256=hashes, **WORK)))
    check_results(root)


def check_results(root):
    report = c.c66.sealed(root / "stage146_decision.json", SCHEMA)
    if (report["status"] != STATUS or report["source_stage145"] != SOURCE145 or report["design"] != DESIGN
            or (report["evaluated_states"],report["held_out_sources"],report["sampled_stable_windows"]) != (4754,28,110)
            or set(report["table_sha256"]) != {f"stage146_{name}.csv" for name in TABLES}
            or any(report[k] != v for k,v in WORK.items())):
        raise ValueError("Incomplete Stage-146 conflict appearance diagnostic")
    io.verify({str(root / k):v for k,v in report["table_sha256"].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76","stage89","stage93","stage121","stage133","stage139","stage142","stage143","stage144","stage145")
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
    if any(root == p or root in p.parents or p in root.parents for p in sources):
        parser.error("Use a separate Stage-146 directory")
    for name, path in zip(names, sources):
        if not path.is_dir():
            parser.error(f"Missing --{name} source directory: {path}")
    with writer_lock(root):
        if (root / "stage146_decision.json").exists():
            check_results(root)
        else:
            r143, r144, r145 = sources[-3:]
            fixed.check_results(r143)
            prior.check_results(r144)
            appearance.check_results(r145)
            for path, schema, fingerprint in ((r143/"stage143_decision.json",fixed.SCHEMA,prior.SOURCE143),
                                               (r144/"stage144_decision.json",prior.SCHEMA,appearance.SOURCE144),
                                               (r145/"stage145_decision.json",appearance.SCHEMA,SOURCE145)):
                if c.c66.sealed(path, schema)["fingerprint"] != fingerprint:
                    raise ValueError(f"Need the reviewed source run at {path}")
            baselines = c.c66.sealed(r143 / "stage143_models.json", fixed.SCHEMA)["models"]
            rankers = c.c66.sealed(r144 / "stage144_models.json", prior.SCHEMA)["models"]
            native_context, replay_context = fixed.load_context(*sources[:-3])
            native = fixed.native_rows(*native_context, include_regions=True)
            replays = fixed.replay_rows(*replay_context, include_regions=True)
            entry_for, ap = native_context[3:5]
            run(root, native, replays, entry_for, ap["root"], (r144,r145), baselines, rankers)
    print(f"Stage-146: {STATUS}", flush=True)


if __name__ == "__main__":
    main()
