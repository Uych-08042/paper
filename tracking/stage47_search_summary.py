"""Offline response, crop/mapping hybrids and shape diagnostics for Stage-47."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stage27_uav_collect import writer_lock
from stage27_uav_common import nonempty_path, publish_identical, scene_geometry, tracking_iou
from stage28_response_common import bind_native_top1, dense_boxes, response_metrics
from stage46_shift_audit import publish_table, verify_files
import stage47_search_common as c


def unclipped_box(size, offset, cell, parent, factor, search_size):
    side = size.shape[1]
    y, x = divmod(cell, side)
    xy = (np.array([x, y], np.float32) + offset[:, y, x]) / np.float32(side)
    normalized = np.concatenate([xy, size[:, y, x]])
    physical = (normalized * np.float32(search_size) / np.float32(factor)).astype(float)
    center = c.motion.center(parent) - .5 * search_size / factor + physical[:2]
    return np.r_[center - physical[2:] / 2, physical[2:]]


def shape_metrics(box, unclipped, gt):
    box, unclipped, gt = (np.asarray(v, float) for v in (box, unclipped, gt))
    bc, gc = c.motion.center(box), c.motion.center(gt)
    gt_side = np.sqrt(gt[2:].prod())
    # Offline geometry replacements only; these boxes never enter a replay.
    gt_size_at_pred_center = np.r_[bc - gt[2:] / 2, gt[2:]]
    pred_size_at_gt_center = np.r_[gc - box[2:] / 2, box[2:]]
    return dict(iou=float(tracking_iou(box, gt)), center_error_gt_units=float(np.linalg.norm(bc - gc) / gt_side),
        dx_gt_width=float((bc[0] - gc[0]) / gt[2]), dy_gt_height=float((bc[1] - gc[1]) / gt[3]),
        width_ratio=float(box[2] / gt[2]), height_ratio=float(box[3] / gt[3]),
        gt_width=float(gt[2]), gt_height=float(gt[3]), predicted_width=float(box[2]), predicted_height=float(box[3]),
        unclipped_width_ratio=float(unclipped[2] / gt[2]), unclipped_height_ratio=float(unclipped[3] / gt[3]),
        clip_max_abs_delta=float(np.max(np.abs(box - unclipped))),
        gt_size_at_pred_center_iou=float(tracking_iou(gt_size_at_pred_center, gt)),
        pred_size_at_gt_center_iou=float(tracking_iou(pred_size_at_gt_center, gt)))


def analyze_case(event, data, inputs, window):
    entry = inputs["traces"][event["sequence"]][0]
    source = inputs["source"]
    responses, focus, hybrids, onset_hybrids, shape_focus = [], [], [], [], []
    for f, frame in enumerate(event["frames"]):
        gt = np.asarray(entry["ground_truth"][frame], float)
        for h, history in enumerate(c.HISTORIES):
            no_parent = data["parents"][f, h, 0]
            no_box = data["boxes"][f, h, 0]
            no_raw, no_size, no_offset = (data[n][f, h, 0] for n in ("raw", "size", "offset"))
            no_cell = int(np.argmax(no_raw * window))
            for t, treatment in enumerate(c.TREATMENTS):
                ix = (f, h, t)
                raw, size, offset, box, parent = (data[n][ix] for n in ("raw", "size", "offset", "boxes", "parents"))
                factor, hw = data["factor"][ix], data["image_hw"][f]
                weighted_cell, raw_cell = int(np.argmax(raw * window)), int(np.argmax(raw))
                boxes = dense_boxes(size, offset, parent, factor, source["search_size"], hw)
                boxes = bind_native_top1(raw, window, boxes, box)
                metrics, _, _ = response_metrics(raw, window, boxes, gt)
                unclipped = unclipped_box(size, offset, weighted_cell, parent, factor, source["search_size"])
                geometry = scene_geometry(parent, gt, hw, source["search_factor"], source["search_size"])
                identity = dict(case_id=event["case_id"], sequence=event["sequence"], frame=frame,
                    offset=frame - event["frame"], history=history, treatment=treatment,
                    recorded_treatment=treatment == c.EXECUTED[history])
                pixel_equal = data["crop_sha256"][ix] == data["crop_sha256"][f, h, 0]
                row = dict(**identity, **shape_metrics(box, unclipped, gt), coverage=float(geometry["coverage"]),
                    parent_error_gt_units=float(np.linalg.norm(c.motion.center(parent) - c.motion.center(gt)) / np.sqrt(gt[2:].prod())),
                    shift_px=float(np.linalg.norm(c.motion.center(parent) - c.motion.center(no_parent))),
                    crop_rectangle_equal_no_shift=bool(np.array_equal(data["crop"][ix], data["crop"][f, h, 0])),
                    crop_pixels_equal_no_shift=bool(pixel_equal), mask_equal_no_shift=bool(data["mask_sha256"][ix] == data["mask_sha256"][f, h, 0]),
                    maximum_map_delta_no_shift=max(float(np.max(np.abs(data[n][ix] - data[n][f, h, 0]))) for n in ("raw", "size", "offset")),
                    layer=int(data["layer"][ix]), raw_cell=raw_cell, weighted_cell=weighted_cell,
                    raw_peak_amplitude=float(raw.max()), **metrics)
                responses.append(row)
                if frame == event["frame"] and history == event["owner"]:
                    keys = ("iou", "coverage", "parent_error_gt_units", "center_error_gt_units", "dx_gt_width", "dy_gt_height",
                        "width_ratio", "height_ratio", "shift_px", "layer", "weighted_cell", "weighted_top5_gt_recall",
                        "weighted_gt_rank_in_top5", "weighted_gt_score_margin", "dense_best_gt_iou", "crop_pixels_equal_no_shift")
                    focus.append(dict(**identity, **{k: row[k] for k in keys}))
                # A two-by-two offline decode: source maps versus shifted maps,
                # each mapped with the unshifted or shifted floating parent.
                mapping = dense_boxes(no_size, no_offset, parent, factor, source["search_size"], hw)[no_cell]
                sampling = dense_boxes(size, offset, no_parent, factor, source["search_size"], hw)[weighted_cell]
                if np.array_equal(parent, no_parent):
                    mapping, sampling = no_box.copy(), box.copy()
                baseline_iou = float(tracking_iou(no_box, gt))
                mapping_iou, sampling_iou, full_iou = (float(tracking_iou(b, gt)) for b in (mapping, sampling, box))
                hybrid = dict(**identity, no_shift_iou=baseline_iou, mapping_only_iou=mapping_iou,
                    changed_maps_at_no_shift_parent_iou=sampling_iou, full_iou=full_iou,
                    mapping_only_gain=mapping_iou - baseline_iou,
                    changed_maps_gain=sampling_iou - baseline_iou, full_gain=full_iou - baseline_iou,
                    interaction=full_iou - mapping_iou - sampling_iou + baseline_iou,
                    crop_pixels_equal_no_shift=bool(pixel_equal),
                    mapping_only_center_change_px=float(np.linalg.norm(c.motion.center(mapping) - c.motion.center(no_box))))
                hybrids.append(hybrid)
                if frame == event["frame"] and history == event["owner"]:
                    onset_hybrids.append(hybrid)
                if event["case_id"] == "E04" and history == event["owner"]:
                    shape_focus.append(dict(**identity, **shape_metrics(box, unclipped, gt)))
    return dict(responses=responses, onset_summary=focus, hybrids=hybrids, onset_hybrids=onset_hybrids, shape_focus=shape_focus)


def summarize(root):
    root = Path(root).expanduser().resolve()
    plan, inputs = c.load_plan(root)
    run = c.read_sealed(root / "run.json")
    if (run.get("role") != "independent_current_frame_search_replay" or
        run.get("plan_fingerprint") != plan["fingerprint"] or run.get("runtime") != inputs["old_run"]["runtime"] or
        run.get("histories") != list(c.HISTORIES) or run.get("treatments") != list(c.TREATMENTS)):
        raise ValueError("Mixed Stage-47 run/plan/runtime")
    window = np.asarray(run["window"], np.float32)
    side = int(inputs["source"]["runtime"]["lens_x"] ** .5)
    if window.shape != (side, side) or not np.isfinite(window).all() or (window < 0).any() or not (window > 0).any():
        raise ValueError("Invalid fixed native window")
    expected = {c.path_for(root, e).name for e in plan["events"]}
    if {p.name for p in (root / "responses").glob("*.npz")} - expected:
        raise ValueError("Unexpected response caches")
    tables = {name: [] for name in ("responses", "onset_summary", "hybrids", "onset_hybrids", "shape_focus")}
    missing, integrity, cache_hashes = [], [], {}
    for event in plan["events"]:
        path = c.path_for(root, event)
        data = c.read_cache(path, c.metadata(run, event))
        if data is None:
            missing.append(event["case_id"])
            continue
        integrity.append(dict(case_id=event["case_id"], **c.audit_response(data, event, inputs, window)))
        cache_hashes[str(path)] = c.file_hash(path)
        cache_hashes[str(path.with_suffix(".receipt.json"))] = c.file_hash(path.with_suffix(".receipt.json"))
        parts = analyze_case(event, data, inputs, window)
        for name, rows in parts.items():
            tables[name].extend(rows)
    if missing:
        print("Stage-47: PILOT_INCOMPLETE; missing " + ", ".join(missing))
        return None
    if sum(r["head_calls"] for r in integrity) != plan["expected_head_calls"]:
        raise ValueError("Replay call budget mismatch")
    output = root / "analysis"
    with writer_lock(output):
        for name, rows in tables.items():
            publish_table(output / ("stage47_" + name + ".csv"), rows)
        verify_files(plan["source_file_sha256"])
        verify_files(cache_hashes)
        report = c.seal(dict(schema=c.SCHEMA, status="COMPLETE_MATCHED_SEARCH_RESPONSE_DIAGNOSTIC",
            role="posthoc_current_frame_search_intervention_not_closed_loop_policy",
            plan_fingerprint=plan["fingerprint"], run_fingerprint=run["fingerprint"], integrity=integrity,
            replay_head_calls=sum(r["head_calls"] for r in integrity), replay_block_calls=sum(r["block_calls"] for r in integrity),
            cache_sha256=cache_hashes, table_sha256={"stage47_" + n + ".csv": c.file_hash(output / ("stage47_" + n + ".csv")) for n in tables},
            training_performed=False, new_closed_loop_policy_evaluated=False, adoption_gate=None,
            warnings=["Four onsets and windows were selected after consumed development outcomes; no independent generalization test.",
                "Treatments share one recorded history at a time. Different histories retain their own masks, matrices and velocities.",
                "No GT enters proposals or inference; GT candidate matching, geometry replacements and margins are offline diagnostics.",
                "No state rolls forward from a replay. Local gains do not establish persistent recovery or full-video AUC gains.",
                "All-cell GT matches are hindsight ceilings; geometric non-target boxes do not prove semantic distractor identity or visibility.",
                "Grid cell IDs refer to each crop's own coordinates. Equal indices across crops are not necessarily the same image location.",
                "Mapping/map hybrids are offline decoder probes, not evaluated policies or unique causal head attribution.",
                "Shape replacements keep either GT size or GT center offline; their IoUs do not add to a shape/center decomposition.",
                "Actual module calls are audited, but instrumentation and extra replay calls provide no deployment speed claim."],
            next_step="Compare recorded-treatment replay, target retention, changed-map versus mapping effects and E04 shape before specifying a new controller"))
        publish_identical(output / "stage47_diagnostic.json", report)
    print("Stage-47: COMPLETE_MATCHED_SEARCH_RESPONSE_DIAGNOSTIC; {} independent replays".format(report["replay_head_calls"]))
    print("case history          treatment          IoU   top5  GT-rank cell width-ratio height-ratio")
    for row in tables["onset_summary"]:
        print("{} {:<16} {:<16} {:.3f} {:>6} {:>7} {:>4} {:.3f} {:.3f}".format(row["case_id"], row["history"], row["treatment"],
            row["iou"], str(row["weighted_top5_gt_recall"]), str(row["weighted_gt_rank_in_top5"]), row["weighted_cell"], row["width_ratio"], row["height_ratio"]))
    print("Reports: " + str(output))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=nonempty_path)
    summarize(p.parse_args().input)
