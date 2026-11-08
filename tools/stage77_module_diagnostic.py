"""Locate crop, dense decode and candidate choice limits on frozen Stage-76 histories."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path

import numpy as np

import stage76_horizon_experiment as s76
from stage27_uav_collect import writer_lock
from stage27_uav_common import failure_events, nonempty_path, scene_geometry, tracking_iou
from stage28_response_common import bind_native_top1, dense_boxes
from stage47_search_summary import shape_metrics, unclipped_box
from stage68_advantage_summary import publish_table


SCHEMA = "stage77_sgla_search_decode_diagnostic_v1"
DESIGN = dict(good_iou=.5, low_iou=.2, adequate_coverage=.9, hann_top_k=5)
c, e63, io = s76.c, s76.e.e63, s76.source_io
MAIN = "uav0000304_00253_s"
INPUT_FIELDS = ("history_latest", "candidate_parents", "image_hw", "crop", "crop_sha256")
CATEGORIES = ("selected_good", "outer_choice_missed", "decode_grid_has_good",
              "crop_support_limited", "covered_grid_lacks_good")


def load_source(root):
    p76 = c.c66.sealed(root / "manifest.json", s76.SCHEMA)
    report = c.c66.sealed(root / "analysis/stage76_decision.json", s76.SCHEMA)
    parent = Path(p76["source_input"])
    p74 = c.sealed(parent / "manifest.json")
    d74 = c.sealed(parent / "analysis/stage74_decision.json")
    allowed = {"runtime.tracker.device", "runtime.gpu_uuid"}
    changes = s76.startup.differences(p74["runtime_identity"], p76["runtime"])
    if (report["status"] != "COMPLETE_MATCHED_HORIZON_SUPERVISION_DEVELOPMENT"
            or report["plan_fingerprint"] != p76["fingerprint"] or report["runtime"] != p76["runtime"]
            or report["trajectories"] != len(p76["units"]) or report["evaluation_search_calls"] != p76["eval_search_calls"]
            or d74["plan_fingerprint"] != p74["fingerprint"] or d74["runtime"] != p74["runtime_identity"]
            or any(row["field"] not in allowed for row in changes)
            or not (report["source_report_fingerprint"] == p76["source_report_fingerprint"] == d74["fingerprint"])
            or not (report["source_training_fingerprint"] == p76["source_training_fingerprint"] == d74["training_fingerprint"])):
        raise ValueError("Need the complete sealed Stage-76 evaluation and its Stage-74 source")
    evidence = {}
    for unit in p76["units"]:
        s76.binding(evidence, root / "evaluation" / (c.key(unit) + ".npz"))
    if c.fingerprint(evidence) != report["evaluation_evidence_fingerprint"]:
        raise ValueError("Stage-76 evaluation evidence changed")
    for path in (root / "manifest.json", root / "analysis/stage76_decision.json", parent / "manifest.json",
                 parent / "analysis/stage74_decision.json"):
        evidence[str(path.resolve())] = c.file_hash(path)
    return dict(root=root, plan=p76, original=p74, report=report, evidence=evidence)


def windows(source):
    lookup = {(u["sequence"], u["policy"], u["seed"]): u for u in source["plan"]["units"]}
    definitions = []
    for policy, seed in (("horizon_ce", 2171), ("horizon_ce", 2172), ("immediate_ce", 2171),
                         ("immediate_ce", 2172), ("q0", 0)):
        onset = {2171: 556, 2172: 554}[seed] if policy == "horizon_ce" else None
        definitions.append(("early_0304", MAIN, policy, seed, 551, 560, onset))
    for policy, seed in (("ppo", 2172), ("ppo", 2171), ("q0", 0)):
        definitions.append(("late_0304", MAIN, policy, seed, 1076, 1092, 1090 if seed == 2172 else None))
    definitions.extend(("choice_control", "uav0000003_00000_s", "ppo", seed, frame, frame, frame)
                       for seed, frame in ((2171, 138), (2172, 140)))
    definitions.extend(("recovery_control", seq, "ppo", 2172, frame, frame, None)
                       for seq, frame in (("uav0000176_00000_s", 321), ("uav0000204_00000_s", 664)))
    return [dict(group=g, unit=lookup[seq, policy, seed], frames=list(range(first, last+1)), expected_first_failure=onset)
            for g, seq, policy, seed, first, last, onset in definitions]


def make_plan(source, runtime):
    blocks = windows(source)
    states = sum(len(w["frames"]) for w in blocks)
    code = [Path(__file__), Path(__file__).with_name("run_stage77_module.sh"), Path(s76.__file__), Path(e63.__file__)]
    code += [Path(fn.__code__.co_filename) for fn in (shape_metrics, dense_boxes, scene_geometry, publish_table, s76.startup.checked_runtime)]
    return io.seal(dict(schema=SCHEMA, role="consumed_development_frozen_module_diagnostic",
        source_input=str(source["root"]),
        source_report_fingerprint=source["report"]["fingerprint"], source_evidence_fingerprint=c.fingerprint(source["evidence"]),
        runtime=runtime, declared_runtime_differences=s76.startup.differences(source["plan"]["runtime"], runtime),
        design=dict(DESIGN), windows=blocks, states=states, expected_search_calls=4*states,
        code_sha256={str(p.resolve()): c.file_hash(p) for p in code}))


def read_source_trace(source, window):
    unit = window["unit"]
    report = source["report"]
    controller = report["model_fingerprints"][c.job_key(unit)] if unit["policy"] in s76.SUPERVISED else report["source_training_fingerprint"]
    path = source["root"] / "evaluation" / (c.key(unit) + ".npz")
    data = c.read_cache(path, dict(plan_fingerprint=source["plan"]["fingerprint"], unit=unit, controller=controller))
    entry = c.entry_for(source["original"], unit["sequence"])
    count = len(entry["frames"]) - 1
    if data is None or set(data) != set(s76.e.FIELDS) or not 1 <= min(window["frames"]) <= max(window["frames"]) <= count:
        raise ValueError("Missing source trace or invalid replay frame")
    s76.branches.prior.audit_trace(data, count, 4)
    gt = np.asarray(entry["ground_truth"], float)
    if window["expected_first_failure"] is not None:
        events = failure_events(tracking_iou(data["boxes"], gt[1:]))
        if not events or events[0]["start"] + 1 != window["expected_first_failure"]:
            raise ValueError("Source first failure does not match the fixed case")
    return entry, data


def inspect_crops(shared, old, gt, selected, design):
    hw = shared["image_hw"][0]
    window = np.asarray(old["window"], np.float32)
    rows = []
    for q in range(4):
        parent, native = (shared[k][0, q] for k in ("candidate_parents", "candidate_boxes"))
        raw, size, offset = (shared[k][0, q] for k in ("raw", "size", "offset"))
        geometry = scene_geometry(parent, gt, hw, old["search_factor"], old["search_size"])
        factor = geometry["factor"]
        weighted = (raw * window).ravel()
        order = np.argsort(-weighted, kind="stable")
        cell = int(order[0])
        grid = dense_boxes(size, offset, parent, factor, old["search_size"], hw)
        grid = bind_native_top1(raw, window, grid, native)
        overlap = tracking_iou(grid, gt)
        good = overlap >= design["good_iou"]
        ranks = np.flatnonzero(good[order])
        shape = shape_metrics(native, unclipped_box(size, offset, cell, parent, factor, old["search_size"]), gt)
        rows.append(dict(candidate=q, source_selected=q == selected, gt_coverage=float(geometry["coverage"]),
                         gt_center_inside=geometry["center_in_usable"], input_gt_width=float(geometry["input_wh"][0]),
                         input_gt_height=float(geometry["input_wh"][1]), raw_cell=int(raw.argmax()), hann_cell=cell,
                         raw_peak_iou=float(overlap[raw.argmax()]), best_grid_iou=float(overlap.max()),
                         top_hann_iou=float(overlap[order[:design["hann_top_k"]]].max()),
                         first_good_hann_rank=int(ranks[0]+1) if len(ranks) else None, good_grid_cells=int(good.sum()),
                         layer=int(shared["layer"][0, q]), **shape))
    return rows


def classify(rows, selected, design):
    good = design["good_iou"]
    if rows[selected]["iou"] >= good:
        return "selected_good"
    if any(r["iou"] >= good for r in rows):
        return "outer_choice_missed"
    if any(r["best_grid_iou"] >= good for r in rows):
        return "decode_grid_has_good"
    if any(r["gt_coverage"] >= design["adequate_coverage"] and r["gt_center_inside"] for r in rows):
        return "covered_grid_lacks_good"
    return "crop_support_limited"


def replay_window(root, plan, source, window, observer):
    entry, trace = read_source_trace(source, window)
    old = source["original"]
    unit = window["unit"]
    prefix = root / "collection" / c.key(unit)
    states, candidates, evidence = [], [], {}
    calls = 0
    for frame in window["frames"]:
        pos = frame - 1
        history = e63.history_at(trace, entry["initial_box"], frame, old)
        path = prefix / f"frame_{frame:06d}.npz"
        meta = dict(plan_fingerprint=plan["fingerprint"], unit=unit, frame=frame)
        data = c.read_cache(path, meta)
        new = data is None
        if new:
            data = e63.collect_shared(observer, entry["frames"], entry["initial_box"], history, frame, "history", s76.e.e67.actor_plan(old))
        e63.audit_segment(data, history, frame, 1, "history", old)
        np.testing.assert_array_equal(data["initial_history"], history)
        for field in INPUT_FIELDS:
            np.testing.assert_array_equal(data[field], trace[field][pos:pos+1], err_msg=f"Source inputs changed: {field}")
        if new:
            c.save(path, data, meta)
        s76.binding(evidence, path)
        calls += int(data["head_calls"].sum())
        selected = int(trace["selected"][pos])
        gt = np.asarray(entry["ground_truth"][frame], float)
        rows = inspect_crops(data, old, gt, selected, plan["design"])
        source_ious = tracking_iou(trace["candidate_boxes"][pos], gt)
        ident = dict(group=window["group"], **unit, frame=frame)
        candidates.extend(dict(ident, source_public_iou=float(iou), **row) for row, iou in zip(rows, source_ious))
        good_iou = plan["design"]["good_iou"]
        states.append(dict(ident, source_action=selected, source_iou=float(tracking_iou(trace["boxes"][pos], gt)),
            replay_iou_at_source_action=rows[selected]["iou"], best_public_iou=max(r["iou"] for r in rows),
            source_best_public_iou=float(source_ious.max()),
            public_support_changed=any((r["iou"] >= good_iou) != (iou >= good_iou) for r, iou in zip(rows, source_ious)),
            max_gt_coverage=max(r["gt_coverage"] for r in rows), best_grid_iou=max(r["best_grid_iou"] for r in rows),
            best_raw_peak_iou=max(r["raw_peak_iou"] for r in rows), best_top5_hann_iou=max(r["top_hann_iou"] for r in rows),
            max_public_box_error_px=float(np.abs(data["candidate_boxes"][0]-trace["candidate_boxes"][pos]).max()),
            category=classify(rows, selected, plan["design"])))
    print(f"[module replay] {window['group']} {c.key(unit)} {len(states)} states", flush=True)
    return states, candidates, evidence, calls


def summarize_states(states):
    groups = defaultdict(list)
    for row in states:
        keys = ("group", "sequence", "policy", "seed")
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for (group, sequence, policy, seed), rows in groups.items():
        counts = Counter(r["category"] for r in rows)
        result.append(dict(group=group, sequence=sequence, policy=policy, seed=seed, states=len(rows),
                           **{key: counts[key] for key in CATEGORIES},
                           low_iou_frames=sum(r["replay_iou_at_source_action"] < DESIGN["low_iou"] for r in rows),
                           public_support_changed_states=sum(r["public_support_changed"] for r in rows),
                           maximum_public_box_error_px=max(r["max_public_box_error_px"] for r in rows)))
    return result


def run(root, plan, source, observer):
    states, candidates, evidence = [], [], {}
    calls = 0
    with e63.source_experiment.installed(observer, [], plan["windows"][0]["unit"]) as head:
        if head != source["report"]["native_head_sha256"]:
            raise ValueError("Native head differs from Stage-76")
        for item in plan["windows"]:
            a, b, hashes, cost = replay_window(root, plan, source, item, observer)
            states.extend(a)
            candidates.extend(b)
            evidence.update(hashes)
            calls += cost
    if len(states) != plan["states"] or calls != plan["expected_search_calls"] or len(candidates) != calls:
        raise ValueError("Stage-77 replay budget changed")
    io.verify(source["evidence"])
    io.verify(plan["code_sha256"])
    directory = root / "analysis"
    directory.mkdir(exist_ok=True)
    tables = dict(summary=summarize_states(states), states=states, candidates=candidates)
    hashes = {}
    for name, rows in tables.items():
        path = directory / f"stage77_{name}.csv"
        publish_table(path, rows)
        hashes[path.name] = c.file_hash(path)
    report = io.seal(dict(schema=SCHEMA, status="COMPLETE_SGLA_MODULE_DIAGNOSTIC", plan_fingerprint=plan["fingerprint"],
        source_report_fingerprint=plan["source_report_fingerprint"], runtime=plan["runtime"],
        declared_runtime_differences=plan["declared_runtime_differences"], design=plan["design"],
        states=len(states), search_calls=calls, block_calls=7*calls, training_updates=0, new_full_trajectories=0,
        public_support_changed_states=sum(r["public_support_changed"] for r in states),
        maximum_public_box_error_px=max(r["max_public_box_error_px"] for r in states),
        native_head_sha256=head, replay_evidence_fingerprint=c.fingerprint(evidence), table_sha256=hashes))
    c.publish_identical(directory / "stage77_diagnostic.json", report)
    print(f"Stage-77: {report['status']} ({len(states)} states, {calls} searches)", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=nonempty_path)
    parser.add_argument("--output_dir", required=True, type=nonempty_path)
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--parameter", default="deit_distilled")
    parser.set_defaults(mode="stage77")
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("Supply the original frozen SGLA checkpoint")
    root, parent = (Path(value).expanduser().resolve() for value in (args.output_dir, args.input))
    if root == parent or root in parent.parents or parent in root.parents:
        parser.error("Use an output directory separate from Stage-76")
    import torch

    torch.set_num_threads(1)
    with writer_lock(root):
        if not (root / "manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-77 directory")
        source = load_source(parent)
        runtime = s76.device_runtime(source["original"], torch, args.gpu)
        plan = make_plan(source, runtime)
        c.publish_identical(root / "manifest.json", plan)
        frozen = dict(source["original"], fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = s76.startup.checked_runtime(args, frozen, root)
        observer = e63.Capture(native)
        try:
            run(root, plan, source, observer)
        finally:
            observer.handle.remove()


if __name__ == "__main__":
    main()
