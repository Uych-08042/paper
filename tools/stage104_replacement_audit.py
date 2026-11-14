"""Audit four-candidate replacements and state commits from sealed Stage-103 caches."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import stage103_candidate_verification as source103
from stage27_uav_common import crop_geometry, failure_events, tracking_iou


c, io, s93, e63 = source103.c, source103.io, source103.s93, source103.e63
SCHEMA = "stage104_cached_replacement_state_audit_v1"
STATUS = "COMPLETE_CACHED_REPLACEMENT_STATE_AUDIT"
SOURCE103 = "81222ffbb0e81639ab6db572089694f2382e5a0a77d16c9d45423cf808bd6c97"
DESIGN = dict(window_before=32, window_after=17)
TABLES = ("summary", "unit_summary", "cases", "recovery", "signal_summary", "reference_summary",
          "chain_excerpt", "state_frames", "reference_drops")
PACK = TABLES[:7]
RATES = ("slot_replacement", "box_replacement", "outside_pair", "good_lost", "good_rescued",
         "low_introduced", "low_removed", "no_good_supply", "final_miss_good", "low_output",
         "low_memory_write", "rejected_low_geometry_commit", "reads_low_appearance")
SIGNALS = ("gain_over_pair", "gap_to_runner_up", "anchor_gain_over_pair", "min_reference_gain_over_pair",
           "all_refs_prefer_selected", "next_crop_changed")


def number(value):
    return float(value) if np.isfinite(value) else None


def encoded(values):
    return "|".join(str(v) for v in values)


def replay_decision(data, index, memory, mode, design):
    """Neither this replay nor its memory/reference interventions receives labels."""
    row = {k: data[k][index].copy() for k in ("features", "candidate_boxes", "candidate_parents", "history_latest")}
    row["selected"] = int(data["ppo_selected"][index])
    decision = source103.decide(row, data["all_queries"][index], data["anchor"], list(memory), index+1, mode, design)
    return row, decision


def geometry_shadow(history, actual_box, pair_box, next_hw, ap):
    """Change only the committed box; compute parents/crops, never invented outputs."""
    build = e63.source_common.prior.history_parents
    actual = build((history+[actual_box])[-ap["design"]["history"]:], ap["design"])
    alternate = build((history+[pair_box])[-ap["design"]["history"]:], ap["design"])
    actual_crop = np.asarray([crop_geometry(p, next_hw, ap["search_factor"], ap["search_size"])[0] for p in actual])
    other_crop = np.asarray([crop_geometry(p, next_hw, ap["search_factor"], ap["search_size"])[0] for p in alternate])
    return actual, dict(next_parent_changed=bool(np.any(actual != alternate)),
        next_crop_changed=bool(np.any(actual_crop != other_crop)),
        next_crop_changed_mask=encoded(np.any(actual_crop != other_crop, axis=1).astype(int)),
        next_max_center_delta_px=float(np.linalg.norm(source103.s86.centers(actual)-source103.s86.centers(alternate), axis=1).max()),
        next_max_abs_log_size_delta=float(np.abs(np.log(actual[:, 2:]/alternate[:, 2:])).max()))


def analyze_trace(data, entry, spec, arm, old, design):
    """Reconstruct actual memory, then evaluate frozen same-state alternative outputs."""
    ap, unit = e63.c.actor_plan(old), spec["unit"]
    history, memory, rows, edges = [np.asarray(entry["initial_box"], float)], [], [], []
    n = len(data["frame_id"])
    truth = np.asarray(entry["ground_truth"], float)
    if n != len(truth)-1:
        raise ValueError("Incomplete full trajectory")
    ident = dict(cohort=spec["cohort"], sequence=unit["sequence"], seed=unit["seed"], arm=arm)
    for i in range(n):
        frame = i+1
        np.testing.assert_array_equal(data["history_latest"][i], history[-1])
        row, actual = replay_decision(data, i, memory, arm, design)
        _, pair = replay_decision(data, i, memory, "pair", design)
        for key, value in actual.items():
            np.testing.assert_allclose(data[key][i], value, rtol=0, atol=1e-12, equal_nan=True) if key in ("scores", "gains") else np.testing.assert_array_equal(data[key][i], value)
        selected, base = actual["selected"], pair["selected"]
        np.testing.assert_array_equal(base, data["pair_selected"][i])
        np.testing.assert_array_equal(data["boxes"][i], row["candidate_boxes"][selected])
        references = [(f, q) for f, q in s93.select_memory(memory, frame, design) if np.isfinite(q).all()] if actual["used_history"] else []
        for f, q in references:
            if f >= frame or not data["memory_write"][f-1]:
                raise ValueError("Not an accepted past reference")
            np.testing.assert_array_equal(q, data["all_queries"][f-1, int(data["selected"][f-1])])
        delta = data["all_queries"][i, selected]-data["all_queries"][i, base]
        ref_gains = [float(delta @ q) for _, q in references]
        eligible = [j for j in range(4) if actual["support_mask"][j] and j != selected and np.isfinite(actual["scores"][j])]
        geometry = dict(next_parent_changed=None, next_crop_changed=None, next_crop_changed_mask="",
                        next_max_center_delta_px=None, next_max_abs_log_size_delta=None)
        if frame < n:
            parents, geometry = geometry_shadow(history, row["candidate_boxes"][selected], row["candidate_boxes"][base], data["image_hw"][i+1], ap)
            np.testing.assert_array_equal(parents, data["candidate_parents"][i+1])
        signals = dict(gain_over_pair=number(delta @ (np.mean([q for _, q in references], axis=0) if references else data["anchor"])),
            gap_to_runner_up=number(actual["scores"][selected]-max(actual["scores"][j] for j in eligible)) if eligible else None,
            anchor_gain_over_pair=number(delta @ data["anchor"]),
            min_reference_gain_over_pair=min(ref_gains) if ref_gains else None,
            all_refs_prefer_selected=all(v > design["switch_margin"] for v in ref_gains) if ref_gains else None)
        # Labels attach only after actual and shadow decisions/parents have been fixed.
        values = source103.source102.quality(row["candidate_boxes"], truth[frame])
        valid = values is not None
        ai, bi, best = (float(values[selected]), float(values[base]), float(values.max())) if valid else (None, None, None)
        low_refs = []
        for f, _ in references:
            value = source103.source102.quality(data["boxes"][f-1], truth[f])
            low_refs.append(value is not None and value < design["low_iou"])
        changed = not np.array_equal(row["candidate_boxes"][selected], row["candidate_boxes"][base])
        good, low = design["good_iou"], design["low_iou"]
        outcome = ("invalid_gt" if not valid else "unchanged_box" if not changed else
                   "lose_good" if bi >= good and ai < good else "rescue_good" if bi < good and ai >= good else
                   "both_good" if ai >= good else "both_not_good")
        rows.append(dict(ident, frame=frame, gt_valid=valid, selected=selected, same_state_pair_selected=base,
            ppo_selected=actual["ppo_selected"], challenger=actual["legacy_challenger"],
            iou=ai, same_state_pair_iou=bi, best_iou=best, same_state_iou_delta=ai-bi if valid else None,
            candidate_iou=encoded(values) if valid else "", support_mask=encoded(actual["support_mask"].astype(int)),
            scores=encoded(actual["scores"]), outcome=outcome, used_history=actual["used_history"],
            reference_kind=actual["reference_kind"], appearance_refs=encoded([f for f, _ in references]),
            memory_write=actual["memory_write"], geometry_committed=True,
            slot_replacement=selected != base, box_replacement=changed,
            outside_pair=selected not in (actual["ppo_selected"], actual["legacy_challenger"]),
            good_lost=valid and bi >= good and ai < good, good_rescued=valid and bi < good and ai >= good,
            low_introduced=valid and bi >= low and ai < low, low_removed=valid and bi < low and ai >= low,
            no_good_supply=valid and best < good, final_miss_good=valid and best >= good and ai < good,
            low_output=valid and ai < low, low_memory_write=valid and ai < low and actual["memory_write"],
            rejected_low_geometry_commit=valid and ai < low and not actual["memory_write"],
            reads_low_appearance=any(low_refs), **signals, **geometry))
        if changed:
            for (f, _), ref_low in zip(references, low_refs):
                _, dropped = replay_decision(data, i, [(rf, q) for rf, q in memory if rf != f], arm, design)
                ref_value = source103.source102.quality(data["boxes"][f-1], truth[f])
                edges.append(dict(ident, frame=frame, reference_frame=f,
                    reference_iou=float(ref_value) if ref_value is not None else None,
                    reference_quality="invalid_gt" if ref_value is None else "low" if ref_low else "good" if ref_value >= good else "intermediate",
                    reference_was_replacement=rows[f-1]["box_replacement"],
                    actual_selected=selected, drop_selected=dropped["selected"],
                    actual_iou=ai, drop_iou=float(values[dropped["selected"]]) if valid else None,
                    drop_iou_delta=float(values[dropped["selected"]])-ai if valid else None,
                    drop_used_history=dropped["used_history"],
                    drop_box_changed=not np.array_equal(row["candidate_boxes"][selected], row["candidate_boxes"][dropped["selected"]])))
        if actual["memory_write"]:
            memory.append((frame, data["all_queries"][i, selected].copy()))
        memory[:] = [(f, q) for f, q in memory if frame-f < design["lookback"]]
        history = (history+[data["boxes"][i]])[-ap["design"]["history"]:]
    return rows, edges


def mean(rows, key):
    values = [r[key] for r in rows if r[key] is not None]
    return float(np.mean(values)) if values else None


def aggregate(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["cohort"], r["sequence"], r["seed"], r["arm"]].append(r)
    units = []
    for (cohort, seq, seed, arm), local in sorted(grouped.items()):
        valid = [r for r in local if r["gt_valid"]]
        replaced = [r for r in valid if r["box_replacement"]]
        units.append(dict(cohort=cohort, sequence=seq, seed=seed, arm=arm, frames=len(local), valid_frames=len(valid),
            **{k+"_frames": sum(bool(r[k]) for r in valid) for k in RATES},
            **{k+"_fraction": mean(valid, k) for k in RATES},
            same_state_iou_delta=mean(valid, "same_state_iou_delta"),
            replaced_state_iou_delta=mean(replaced, "same_state_iou_delta"), replaced_valid_frames=len(replaced)))
    summary = []
    for cohort in ("all", "historical", "expanded"):
        for arm in source103.MODES:
            local = [r for r in units if r["arm"] == arm and (cohort == "all" or r["cohort"] == cohort)]
            if not local:
                continue
            videos = defaultdict(list)
            for r in local:
                videos[r["sequence"]].append(r)
            keys = [k+"_fraction" for k in RATES]+["same_state_iou_delta", "replaced_state_iou_delta"]
            summary.append(dict(cohort=cohort, arm=arm, videos=len(videos), units=len(local), frames=sum(r["frames"] for r in local),
                **{k+"_frames": sum(r[k+"_frames"] for r in local) for k in RATES},
                **{k+"_video": mean([dict(value=mean(v, k)) for v in videos.values()], "value") for k in keys},
                replaced_state_supported_videos=sum(any(r["replaced_valid_frames"] for r in v) for v in videos.values())))
    return summary, units


def stratify(rows, label, metrics):
    """Averaging remains within unit, then within video; frame totals are descriptive."""
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["cohort"], r["arm"], r[label], r["sequence"], r["seed"]].append(r)
    unit_rows = [dict(cohort=k[0], arm=k[1], category=k[2], sequence=k[3], seed=k[4], rows=len(v),
                     **{m: mean(v, m) for m in metrics}) for k, v in sorted(grouped.items())]
    out = []
    for cohort in ("all", "historical", "expanded"):
        bins = defaultdict(list)
        for r in unit_rows:
            if cohort == "all" or r["cohort"] == cohort:
                bins[r["arm"], r["category"]].append(r)
        for (arm, category), local in sorted(bins.items()):
            videos = defaultdict(list)
            for r in local:
                videos[r["sequence"]].append(r)
            out.append(dict(cohort=cohort, arm=arm, category=category, videos=len(videos), units=len(local), rows=sum(r["rows"] for r in local),
                **{m+"_video": mean([dict(value=mean(v, m)) for v in videos.values()], "value") for m in metrics},
                **{m+"_supported_videos": sum(mean(v, m) is not None for v in videos.values()) for m in metrics}))
    return out


def case_rows(spec, arm, rows, data, pair_trace, recoveries):
    """All units use the same roles, including every newly lost reference recovery."""
    unit, roles = spec["unit"], []
    for field in ("box_replacement", "good_lost", "low_introduced"):
        first = next((r["frame"] for r in rows if r[field]), None)
        if first is not None:
            roles.append((first, "first_"+field, ""))
    diff = np.flatnonzero(np.any(data["boxes"] != pair_trace["boxes"], axis=1))
    if len(diff):
        roles.append((int(diff[0])+1, "first_output_difference_from_full_pair", ""))
    failures = failure_events(tracking_iou(data["boxes"], np.asarray(spec["truth"])[1:]))
    if failures:
        roles.append((int(failures[0]["start"])+1, "first_failure", ""))
        if not failure_events(tracking_iou(pair_trace["boxes"], np.asarray(spec["truth"])[1:])):
            roles.append((int(failures[0]["start"])+1, "new_failure_on_pair_stable", ""))
    for ref in ("ppo", "pair"):
        baseline = {(r["ppo_failure_frame"], r["ppo_recovery_frame"]): r for r in recoveries
                    if r["arm"] == "pair" and r["reference_arm"] == ref}
        for r in recoveries:
            key = r["ppo_failure_frame"], r["ppo_recovery_frame"]
            if (r["arm"] == arm and r["reference_arm"] == ref and r["matched_good_frame"] is None
                    and baseline[key]["matched_good_frame"] is not None):
                roles.append((key[1], "newly_lost_"+ref+"_recovery", f"{key[0]}:{key[1]}"))
    cases, excerpt_frames = [], defaultdict(set)
    divergence = {}
    for field in ("candidate_parents", "crop", "ppo_selected", "memory_write", "refs", "used_history"):
        changed = np.flatnonzero(np.any((data[field] != pair_trace[field]).reshape(len(rows), -1), axis=1))
        divergence["first_full_pair_"+field+"_difference"] = int(changed[0])+1 if len(changed) else None
    for frame, role, event in sorted(roles):
        start, stop = max(1, frame-DESIGN["window_before"]), min(len(rows), frame+DESIGN["window_after"])
        before, window = rows[:frame], rows[start-1:stop]
        last = lambda predicate: next((r["frame"] for r in reversed(before) if predicate(r)), None)
        case = dict(cohort=spec["cohort"], sequence=unit["sequence"], seed=unit["seed"], arm=arm,
            role=role, frame=frame, reference_event=event, window_start=start, window_end=stop,
            last_good=last(lambda r: r["gt_valid"] and r["iou"] >= source103.DESIGN["good_iou"]),
            last_replacement=last(lambda r: r["box_replacement"]), last_good_lost=last(lambda r: r["good_lost"]),
            last_low_introduced=last(lambda r: r["low_introduced"]),
            first_no_good_in_window=next((r["frame"] for r in window if r["no_good_supply"]), None),
            window_good_lost=sum(r["good_lost"] for r in window), window_good_rescued=sum(r["good_rescued"] for r in window),
            window_low_memory_writes=sum(r["low_memory_write"] for r in window),
            window_rejected_low_geometry_commits=sum(r["rejected_low_geometry_commit"] for r in window),
            **divergence,
            **{k: rows[frame-1][k] for k in ("iou", "same_state_pair_iou", "best_iou", "used_history", "memory_write", "next_crop_changed")} )
        cases.append(case)
        selected = set(range(max(1, frame-2), min(len(rows), frame+2)+1))
        selected.update(case[k] for k in ("last_good", "last_replacement", "last_good_lost", "last_low_introduced") if case[k] is not None)
        for f in list(selected):
            selected.update(int(v) for v in rows[f-1]["appearance_refs"].split("|") if v)
        for f in selected:
            excerpt_frames[f].add(f"{role}@{frame}:{event}")
    excerpt = [dict(rows[f-1], case_roles="|".join(sorted(labels))) for f, labels in sorted(excerpt_frames.items())]
    return cases, excerpt


def required_trace(root, plan, unit, arm):
    path = root/"evaluation"/f"{c.key(unit)}__{arm}.npz"
    data = c.read_cache(path, dict(plan_fingerprint=plan["fingerprint"], unit=unit, arm=arm))
    if data is None:
        raise ValueError("Missing committed source: "+str(path))
    return data


def source_metrics(spec, traces, native, design):
    unit, cohort = spec["unit"], spec["cohort"]
    entry = spec["entry"]
    gt = np.asarray(entry["ground_truth"], float)[1:]
    pair_iou = tracking_iou(traces["pair"]["boxes"], gt)
    seeds, recovery = [], []
    for arm, data in traces.items():
        owner = dict(unit, seed=0, policy="q0") if arm == "native" else unit
        row, events = s93.describe_arm(entry, cohort, owner, arm, data, native, traces["ppo"], design)
        overlap = tracking_iou(data["boxes"], gt)
        bank = tracking_iou(data["candidate_boxes"], gt[:, None])
        pr = s93.recovery_comparison(pair_iou, overlap, design) if arm != "native" else []
        row.update(failure_events=len(failure_events(overlap)), good_candidate_fraction=float((bank.max(1) >= design["good_iou"]).mean()),
            selected_good_fraction=float((overlap >= design["good_iou"]).mean()),
            missed_good_fraction=float(((bank.max(1) >= design["good_iou"]) & (overlap < design["good_iou"])).mean()),
            new_on_pair_stable=not failure_events(pair_iou) and bool(failure_events(overlap)) if arm != "native" else None,
            pair_recoveries_lost=sum(r["matched_good_frame"] is None for r in pr) if arm != "native" else None)
        seeds.append(row)
        recovery.extend(dict(r, reference_arm="ppo") for r in events)
        recovery.extend(dict(cohort=cohort, sequence=unit["sequence"], seed=owner["seed"], arm=arm, reference_arm="pair", **r) for r in pr)
    return seeds, recovery


def run(root103, plan103, load_unit, choose_for, old):
    states, edges, cases, excerpts, seeds, recoveries = [], [], [], [], [], []
    scope = []
    native_by_sequence = {}
    for spec in plan103["units"]:
        unit = spec["unit"]
        entry = load_unit(spec)
        native_owner = dict(unit, seed=0, policy="q0")
        native = native_by_sequence.get(unit["sequence"])
        first = native is None
        if first:
            native = required_trace(root103, plan103, native_owner, "native")
            native_by_sequence[unit["sequence"]] = native
        traces = {arm: required_trace(root103, plan103, unit, arm) for arm in source103.ARMS[1:]}
        choose = choose_for(unit)
        for arm in source103.MODES:
            source103.audit(traces[arm], entry, old, choose, arm, plan103["design"])
        for arm, data in (({"native": native} if first else {}) | traces).items():
            scope.append(dict(cohort=spec["cohort"], sequence=unit["sequence"], seed=0 if arm == "native" else unit["seed"],
                arm=arm, frames=len(data["frame_id"]),
                first_output_difference_from_pair=source103.first_commit_difference(data, traces["pair"]) if arm in source103.MODES[1:] else None,
                outside_pair_frames=int(np.count_nonzero((data["selected"] != data["ppo_selected"]) & (data["selected"] != data["legacy_challenger"]))) if arm in source103.MODES else 0))
        local_seeds, local_recovery = source_metrics(dict(spec, entry=entry), ({"native": native} if first else {}) | traces, native, plan103["design"])
        seeds.extend(local_seeds); recoveries.extend(local_recovery)
        for arm in source103.MODES:
            rows, drops = analyze_trace(traces[arm], entry, spec, arm, old, plan103["design"])
            states.extend(rows); edges.extend(drops)
            if arm != "pair":
                found, excerpt = case_rows(dict(spec, truth=entry["ground_truth"]), arm, rows, traces[arm], traces["pair"], local_recovery)
                cases.extend(found); excerpts.extend(excerpt)
        print(f"[Stage-104] {c.key(unit)}: {len(entry['frames'])-1} frames, three cached histories audited", flush=True)
    policies, paired, videos = source103.summarize(seeds, plan103["design"])
    for name, values in (("policy_summary", policies), ("paired", paired), ("video_summary", videos), ("seed_video", seeds), ("recovery", recoveries)):
        source103.source102.compare_recorded(source103.prior.read_csv(root103/f"analysis/stage103_{name}.csv"), values)
    recorded_scope = source103.prior.read_csv(root103/"analysis/stage103_scope.csv")
    source103.source102.compare_recorded([{k: r[k] for k in scope[0]} for r in recorded_scope], scope)
    summary, units = aggregate(states)
    signals = stratify([r for r in states if r["box_replacement"]], "outcome", ("same_state_iou_delta",)+SIGNALS)
    references = stratify(edges, "reference_quality", ("drop_box_changed", "drop_iou_delta", "reference_was_replacement"))
    return dict(zip(TABLES, (summary, units, cases, recoveries, signals, references, excerpts, states, edges)))


def check_results(root):
    plan = c.c66.sealed(root/"manifest.json", SCHEMA)
    report = c.c66.sealed(root/"analysis/stage104_decision.json", SCHEMA)
    if (report["status"] != STATUS or report["plan_fingerprint"] != plan["fingerprint"]
            or report["source_stage103_fingerprint"] != SOURCE103 or report["source_runtime"] != plan["source_runtime"]
            or set(report["table_sha256"]) != {f"stage104_{n}.csv" for n in TABLES}
            or any(report[k] != 0 for k in ("image_reads", "new_search_calls", "new_block_calls", "training_updates", "confirmation_inference_calls"))
            or not all(report[k] for k in ("source_summary_reproduction_pass", "same_state_pair_replay_pass", "actual_geometry_commit_replay_pass"))
            or report["counts"]["unit_summary"] != 3*plan["expected"]["units"]
            or report["counts"]["state_frames"] != 3*plan["expected"]["unit_frames"]
            or report["counts"]["recovery"] != plan["expected"]["recovery_rows"]):
        raise ValueError("Incomplete Stage-104 cached audit")
    io.verify(plan["source_sha256"]); io.verify(plan["code_sha256"])
    io.verify({str(root/"analysis"/name): digest for name, digest in report["table_sha256"].items()})


def publish(root, plan, tables):
    if set(tables) != set(TABLES):
        raise ValueError("Missing audit table")
    counts = {n: len(rows) for n, rows in tables.items()}
    if (counts["unit_summary"] != 3*plan["expected"]["units"] or counts["state_frames"] != 3*plan["expected"]["unit_frames"]
            or counts["recovery"] != plan["expected"]["recovery_rows"]):
        raise ValueError("Incomplete development support")
    io.verify(plan["source_sha256"]); io.verify(plan["code_sha256"])
    out = root/"analysis"; out.mkdir(exist_ok=True)
    hashes = {}
    for name in TABLES:
        path = out/f"stage104_{name}.csv"
        source103.s86.publish_table(path, tables[name]); hashes[path.name] = c.file_hash(path)
    c.publish_identical(out/"stage104_decision.json", io.seal(dict(schema=SCHEMA, status=STATUS,
        role="consumed_development_conditional_replacement_diagnostic", plan_fingerprint=plan["fingerprint"],
        source_stage103_fingerprint=SOURCE103, source_runtime=plan["source_runtime"], counts=counts,
        source_summary_reproduction_pass=True, same_state_pair_replay_pass=True, actual_geometry_commit_replay_pass=True,
        image_reads=0, new_search_calls=0, new_block_calls=0, training_updates=0, confirmation_inference_calls=0,
        table_sha256=hashes)))
    check_results(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage76", "stage89", "stage93", "stage103", "output_dir"):
        parser.add_argument("--"+name, type=Path, required=name == "output_dir")
    parser.add_argument("--check-results", action="store_true")
    args = parser.parse_args(); root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root); return
    names = ("stage76", "stage89", "stage93", "stage103")
    if not all(getattr(args, n) for n in names):
        parser.error("Supply Stage-76/89/93/103 source directories")
    root76, root89, root93, root103 = sources = [getattr(args, n).expanduser().resolve() for n in names]
    if any(root == p or root in p.parents or p in root.parents for p in sources):
        parser.error("Use a separate Stage-104 directory")
    with s93.writer_lock(root):
        if (root/"analysis/stage104_decision.json").exists():
            check_results(root); return
        if not (root/"manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-104 directory")
        source103.check_results(root103)
        plan103 = c.c66.sealed(root103/"manifest.json", source103.SCHEMA)
        report103 = c.c66.sealed(root103/"analysis/stage103_decision.json", source103.SCHEMA)
        if report103["fingerprint"] != SOURCE103:
            raise ValueError("Need the frozen Stage-103 result")
        plan93, _, plan89, report89, source = source103.prior.s95.s94.load_run(root93, root76, root89)
        if plan103["units"] != plan89["units"]:
            raise ValueError("Development units changed")
        actor = io.load_source(Path(source["plan"]["source_input"]))
        if actor["plan"] != source["original"] or actor["models"]["fingerprint"] != source["report"]["source_training_fingerprint"]:
            raise ValueError("Frozen PPO changed")
        import torch
        torch.set_num_threads(1)
        paths = [root103/"manifest.json", root103/"analysis/stage103_decision.json"]
        bindings = plan103["source_sha256"] | {str(p): c.file_hash(p) for p in paths}
        bindings.update({str(root103/p): h for p, h in report103["cache_sha256"].items()})
        bindings.update({str(root103/"analysis"/p): h for p, h in report103["table_sha256"].items()})
        scope = source103.prior.read_csv(root103/"analysis/stage103_scope.csv")
        expected = dict(units=len(plan103["units"]), unit_frames=sum(int(r["frames"]) for r in scope if r["arm"] == "pair"),
                        recovery_rows=len(source103.prior.read_csv(root103/"analysis/stage103_recovery.csv")))
        if expected != dict(units=40, unit_frames=42572, recovery_rows=168):
            raise ValueError("Frozen Stage-103 coverage changed")
        code = [Path(__file__).resolve(), Path(__file__).with_name("run_stage104_replacement_audit.sh").resolve()]
        plan = io.seal(dict(schema=SCHEMA, design=DESIGN, expected=expected, source_runtime=report103["runtime"],
            source_sha256=bindings, code_sha256=plan103["code_sha256"] | {str(p): c.file_hash(p) for p in code}))
        c.publish_identical(root/"manifest.json", plan)
        def load_unit(spec):
            return source103.prior.s95.load_unit(root93, root89, plan93, plan89, report89, source, spec)[0]
        def choose_for(unit):
            pack = source103.e74.learned_pack(actor["models"], unit, actor["prior"])
            return lambda row, history: source103.e74.choose(row, history, unit, source["original"], actor["prior"], pack)
        print("Stage-104: cached replacements/state commits; no image reads, GPU inference or training", flush=True)
        tables = run(root103, plan103, load_unit, choose_for, source["original"])
        publish(root, plan, tables)
        print(f"Stage-104: {STATUS}; {len(tables['state_frames'])} cached states; no method adoption gate", flush=True)


if __name__ == "__main__":
    main()
