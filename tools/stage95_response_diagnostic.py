"""Replay both Stage-93 histories at fixed recovery anchors and inspect dense responses."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import os
from pathlib import Path

import numpy as np

import stage94_recovery_audit as s94
from stage27_uav_common import scene_geometry, tracking_iou
from stage28_response_common import dense_boxes, response_metrics, DESIGN as PEAK_DESIGN


s93, c, io, s86 = s94.s93, s94.c, s94.io, s94.s86
e63, s76, module = s93.e63, s93.s76, s86.previous
SCHEMA = "stage95_recovery_dense_response_v1"
STATUS = "COMPLETE_RECOVERY_DENSE_RESPONSE_DIAGNOSTIC"
SOURCE94 = "e57c25f2abbb560eb388a080c5d99d749a0bb62614eae1cc27b14efcb7d4b44c"
DESIGN = dict(good_iou=.5, adequate_coverage=.9)
TABLES = ("events", "states", "crops", "summary")


def read_events(root):
    report = c.c66.sealed(root / "stage94_decision.json", s94.SCHEMA)
    if report["fingerprint"] != SOURCE94 or report["stage93_fingerprint"] != s94.SOURCE:
        raise ValueError("Need the frozen Stage-94 recovery audit")
    io.verify({str(root / k): v for k, v in report["table_sha256"].items()})
    with (root / "stage94_events.csv").open(encoding="utf-8", newline="") as handle:
        events = list(csv.DictReader(handle))
    if len(events) != 25 or sum(r["outcome"] == "missed" for r in events) != 4:
        raise ValueError("Recovery cohort changed")
    return events


def points_for(events):
    points = {}
    for event in events:
        onset, recovery = int(event["ppo_failure_frame"]), int(event["ppo_recovery_frame"])
        anchors = dict(onset_previous=onset-1, onset=onset, recovery_previous=recovery-1,
                       recovery=recovery, recovery_next=recovery+1)
        if event["first_no_good_in_window"]:
            anchors["first_missing_good"] = int(event["first_no_good_in_window"])
        for role, frame in anchors.items():
            if frame < 1:
                continue
            key = event["sequence"], int(event["seed"]), frame
            point = points.setdefault(key, dict(sequence=key[0], seed=key[1], frame=frame,
                cohort=event["cohort"], outcome=event["outcome"], roles=[]))
            if point["outcome"] != event["outcome"]:
                raise ValueError("Anchor shared by different outcome strata")
            point["roles"].append(f"{onset}:{recovery}:{role}")
    return sorted(points.values(), key=lambda r: (r["sequence"], r["seed"], r["frame"]))


def load_unit(root93, root89, plan93, plan89, report89, source, spec):
    unit = spec["unit"]
    name = c.key(unit)
    if spec["cohort"] == "historical":
        entry = c.entry_for(source["original"], unit["sequence"])
    else:
        path = f"entries/{unit['sequence']}.json"
        io.verify({str(root89 / path): report89["evidence_sha256"][path]})
        bound = c.c66.sealed(root89 / path, s94.s89.SCHEMA)
        if bound["plan_fingerprint"] != plan89["fingerprint"] or bound["entry"]["sequence"] != unit["sequence"]:
            raise ValueError("Expanded entry changed")
        entry = bound["entry"]
    meta = dict(plan_fingerprint=plan93["fingerprint"], unit=unit)
    gated = s94.required_cache(root93 / f"evaluation/{name}__gated.npz", dict(meta, arm="gated"))
    if not plan93["reuse_ppo"]:
        ppo = s94.required_cache(root93 / f"evaluation/{name}__ppo.npz", dict(meta, arm="ppo"))
    elif spec["cohort"] == "historical" and plan89["reuse_historical"]:
        _, ppo = module.read_source_trace(source, dict(unit=unit, frames=[1], expected_first_failure=None))
    else:
        path = f"baseline/{name}.npz"
        io.verify({str(root89 / path): report89["evidence_sha256"][path]})
        ppo = s94.required_cache(root89 / path, dict(plan_fingerprint=plan89["fingerprint"], unit=unit))
    return entry, dict(ppo=ppo, gated=gated)


def classify(rows, selected):
    if not rows[0]["gt_valid"]:
        return "invalid_gt"
    if rows[selected]["iou"] >= DESIGN["good_iou"]:
        return "selected_good"
    for key, threshold, category in (("iou", DESIGN["good_iou"], "outer_choice_missed"),
                                     ("weighted_top5_gt_recall", 1, "weighted_peaks_have_good"),
                                     ("raw_top5_gt_recall", 1, "raw_peaks_have_good"),
                                     ("dense_best_gt_iou", DESIGN["good_iou"], "dense_good_outside_kept_peaks")):
        if any(r[key] >= threshold for r in rows):
            return category
    if any(r["gt_coverage"] >= DESIGN["adequate_coverage"] and r["gt_center_inside"] for r in rows):
        return "covered_dense_lacks_good"
    return "crop_support_limited"


def inspect(data, old, gt, selected):
    rows = []
    window = np.asarray(old["window"], np.float32)
    valid = bool(np.isfinite(gt).all() and (gt[2:] > 0).all())
    for q in range(4):
        base = dict(candidate=q, source_selected=q == selected, gt_valid=valid)
        if not valid:
            rows.append(base)
            continue
        parent, box = (data[k][0, q] for k in ("candidate_parents", "candidate_boxes"))
        raw, size, offset = (data[k][0, q] for k in ("raw", "size", "offset"))
        hw = data["image_hw"][0]
        geometry = scene_geometry(parent, gt, hw, old["search_factor"], old["search_size"])
        dense = dense_boxes(size, offset, parent, geometry["factor"], old["search_size"], hw)
        metrics, _, _ = response_metrics(raw, window, dense, gt, native_box=box)
        cell = int(np.argmax(raw * window))
        unclipped = module.unclipped_box(size, offset, cell, parent, geometry["factor"], old["search_size"])
        rows.append(dict(base, **metrics, **module.shape_metrics(box, unclipped, gt),
            gt_coverage=float(geometry["coverage"]), gt_center_inside=geometry["center_in_usable"],
            input_gt_width=float(geometry["input_wh"][0]), input_gt_height=float(geometry["input_wh"][1]),
            gt_image_fraction=float(geometry["gt_image_fraction"]), layer=int(data["layer"][0, q])))
    return rows


def replay(root, plan, point, unit, history_name, trace, entry, old, observer):
    frame, pos = point["frame"], point["frame"] - 1
    history = e63.history_at(trace, entry["initial_box"], frame, old)
    path = root / "collection" / f"{c.key(unit)}__{history_name}__f{frame:06d}.npz"
    meta = dict(plan_fingerprint=plan["fingerprint"], unit=unit, history=history_name, frame=frame)
    data = c.read_cache(path, meta)
    fresh = data is None
    if fresh:
        ap = e63.c.actor_plan(old)
        data = e63.collect_shared(observer, entry["frames"], entry["initial_box"], history, frame, "history", ap)
    e63.audit_segment(data, history, frame, 1, "history", old)
    np.testing.assert_array_equal(data["initial_history"], history)
    for field in module.INPUT_FIELDS:
        np.testing.assert_array_equal(data[field], trace[field][pos:pos+1], err_msg=f"Changed source input: {field}")
    if fresh:
        c.save(path, data, meta)
    selected = int(trace["selected"][pos])
    gt = np.asarray(entry["ground_truth"][frame], float)
    crops = inspect(data, old, gt, selected)
    source_ious = tracking_iou(trace["candidate_boxes"][pos], gt)
    replay_ious = tracking_iou(data["candidate_boxes"][0], gt)
    ident = dict(cohort=point["cohort"], sequence=point["sequence"], seed=point["seed"], frame=frame,
                 outcome=point["outcome"], history=history_name, roles="|".join(point["roles"]))
    state = dict(ident, category=classify(crops, selected), source_selected=selected,
        source_iou=float(source_ious[selected]), replay_iou=float(replay_ious[selected]),
        source_best_iou=float(source_ious.max()), replay_best_iou=float(replay_ious.max()),
        source_support_changed=bool(np.any((source_ious >= .5) != (replay_ious >= .5))),
        max_public_box_error_px=float(np.abs(data["candidate_boxes"][0] - trace["candidate_boxes"][pos]).max()),
        gt_valid=crops[0]["gt_valid"],
        best_dense_iou=max(r.get("dense_best_gt_iou", -1.) for r in crops),
        weighted_peak_support=any(r.get("weighted_top5_gt_recall", False) for r in crops),
        raw_peak_support=any(r.get("raw_top5_gt_recall", False) for r in crops),
        maximum_gt_coverage=max(r.get("gt_coverage", -1.) for r in crops),
        selected_gt_size_at_pred_center_iou=crops[selected].get("gt_size_at_pred_center_iou"),
        selected_pred_size_at_gt_center_iou=crops[selected].get("pred_size_at_gt_center_iou"))
    return state, [dict(ident, **r) for r in crops], path


def summarize(events, states):
    lookup = {(r["sequence"], r["seed"], r["frame"], r["history"]): r for r in states}
    focus = []
    for event in events:
        row = dict(event)
        for history in ("ppo", "gated"):
            state = lookup[event["sequence"], int(event["seed"]), int(event["ppo_recovery_frame"]), history]
            for key in ("category", "source_iou", "replay_iou", "replay_best_iou", "best_dense_iou",
                        "weighted_peak_support", "raw_peak_support", "maximum_gt_coverage",
                        "source_support_changed", "max_public_box_error_px",
                        "selected_gt_size_at_pred_center_iou", "selected_pred_size_at_gt_center_iou"):
                row[f"{history}_{key}"] = state[key]
        focus.append(row)
    groups = defaultdict(list)
    for state in states:
        groups[state["outcome"], state["history"]].append(state)
    summary = []
    for (outcome, history), rows in sorted(groups.items()):
        counts = Counter(r["category"] for r in rows)
        for category in ("selected_good", "outer_choice_missed", "weighted_peaks_have_good", "raw_peaks_have_good",
                         "dense_good_outside_kept_peaks", "covered_dense_lacks_good", "crop_support_limited", "invalid_gt"):
            summary.append(dict(outcome=outcome, history=history, category=category, states=counts[category],
                                all_anchor_states=len(rows)))
    return focus, summary


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    report = c.c66.sealed(root / "analysis/stage95_decision.json", SCHEMA)
    if (report["status"] != STATUS or report["plan_fingerprint"] != plan["fingerprint"]
            or report["states"] != 2 * len(plan["points"]) or report["search_calls"] != 4 * report["states"]
            or report["block_calls"] != 7 * report["search_calls"]
            or set(report["table_sha256"]) != {f"stage95_{n}.csv" for n in TABLES}
            or len(report["cache_sha256"]) != report["states"]):
        raise ValueError("Incomplete Stage-95 diagnostic")
    io.verify({str(root / name): sha for name, sha in report["cache_sha256"].items()})
    io.verify({str(root / "analysis" / name): sha for name, sha in report["table_sha256"].items()})
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage76", "stage89", "stage93", "stage94", "output_dir"):
        parser.add_argument(f"--{name}", type=Path, required=name == "output_dir")
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage95")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not all((args.stage76, args.stage89, args.stage93, args.stage94, args.checkpoint)):
        parser.error("Supply Stage-76, 89, 93, 94 and the original SGLA checkpoint")
    roots = [p.expanduser().resolve() for p in (args.stage93, args.stage76, args.stage89, args.stage94)]
    if any(root == p or root in p.parents or p in root.parents for p in roots):
        parser.error("Use a separate Stage-95 output directory")
    root93, root76, root89, root94 = roots
    with s93.writer_lock(root):
        if (root / "analysis/stage95_decision.json").exists():
            check_results(root)
            return
        if not (root / "manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-95 output directory")
        plan93, report93, plan89, report89, source = s94.load_run(root93, root76, root89)
        events = read_events(root94)
        points = points_for(events)
        if len(points) != 137:
            raise ValueError("Frozen recovery anchors changed")
        import torch

        torch.set_num_threads(1)
        old = source["original"]
        runtime = s76.device_runtime(old, torch, args.gpu)
        differences = s76.startup.differences(report93["runtime"], runtime)
        if any(r["field"] not in ("runtime.gpu_uuid", "runtime.tracker.device") for r in differences):
            raise ValueError("Stage-93 software or weights changed")
        paths = [root93 / "manifest.json", root93 / "analysis/stage93_decision.json",
                 root89 / "manifest.json", root89 / "analysis/stage89_decision.json",
                 root94 / "stage94_decision.json", root94 / "stage94_events.csv", s86.ROSTER]
        code = [Path(__file__), Path(__file__).with_name("run_stage95_response_diagnostic.sh"), Path(s94.__file__),
                Path(s93.__file__), Path(module.__file__), Path(e63.__file__),
                Path(response_metrics.__code__.co_filename), Path(scene_geometry.__code__.co_filename),
                Path(module.shape_metrics.__code__.co_filename), Path(s76.startup.__file__),
                Path(c.__file__), Path(io.__file__), Path(s76.__file__), Path(e63.actor.__file__),
                Path(e63.source_common.__file__), Path(e63.source_common.prior.__file__)]
        plan = io.seal(dict(schema=SCHEMA, role="consumed_development_offline_module_diagnostic",
            source_stage94_fingerprint=SOURCE94, runtime=runtime, runtime_differences=differences,
            points=points, events=events, design=DESIGN, peak_design=PEAK_DESIGN,
            source_sha256={str(p): c.file_hash(p) for p in paths} | source["evidence"],
            code_sha256={str(p.resolve()): c.file_hash(p) for p in code}))
        c.publish_identical(root / "manifest.json", plan)
        frozen = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = s76.startup.checked_runtime(args, frozen, root)
        observer = e63.Capture(native)
        states, crops, evidence = [], [], {}
        try:
            groups = defaultdict(list)
            for point in points:
                groups[point["sequence"], point["seed"]].append(point)
            specs = {(r["unit"]["sequence"], r["unit"]["seed"]): r for r in plan89["units"]}
            for key, anchors in groups.items():
                spec = specs[key]
                entry, traces = load_unit(root93, root89, plan93, plan89, report89, source, spec)
                with e63.source_experiment.installed(observer, [], spec["unit"]) as head:
                    if head != source["report"]["native_head_sha256"]:
                        raise ValueError("Original SGLA head changed")
                    for point in anchors:
                        for history in ("ppo", "gated"):
                            state, views, path = replay(root, plan, point, spec["unit"], history,
                                                        traces[history], entry, old, observer)
                            states.append(state)
                            crops.extend(views)
                            evidence[path.relative_to(root).as_posix()] = c.file_hash(path)
                print(f"[Stage-95] {key[0]} seed={key[1]} anchors={len(anchors)}", flush=True)
        finally:
            observer.handle.remove()
        io.verify(plan["source_sha256"])
        io.verify(plan["code_sha256"])
        focus, summary = summarize(events, states)
        out = root / "analysis"
        out.mkdir(exist_ok=True)
        hashes = {}
        for name, rows in zip(TABLES, (focus, states, crops, summary)):
            path = out / f"stage95_{name}.csv"
            s86.publish_table(path, rows)
            hashes[path.name] = c.file_hash(path)
        c.publish_identical(out / "stage95_decision.json", io.seal(dict(
            schema=SCHEMA, status=STATUS, plan_fingerprint=plan["fingerprint"], runtime=runtime,
            source_stage94_fingerprint=SOURCE94, runtime_differences=differences,
            states=len(states), search_calls=4*len(states), block_calls=28*len(states),
            source_support_changed_states=sum(r["source_support_changed"] for r in states),
            maximum_public_box_error_px=max(r["max_public_box_error_px"] for r in states),
            source_correspondence_pass=all(not r["source_support_changed"]
                and r["max_public_box_error_px"] <= old["actor_design"]["box_atol"] for r in states),
            training_updates=0, new_full_trajectories=0, confirmation_inference_calls=0,
            cache_sha256=evidence, table_sha256=hashes)))
        check_results(root)
        print(f"Stage-95: {STATUS} ({len(states)} states, {4*len(states)} searches)")


if __name__ == "__main__":
    main()
