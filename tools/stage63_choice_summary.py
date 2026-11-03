"""Describe same-state one-choice effects; no fitting, policy AUC or adoption test."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import stage63_choice_common as c
import stage63_choice_experiment as experiment
from stage27_uav_common import crop_geometry, scene_geometry, nonempty_path
from stage28_response_common import dense_boxes
from stage46_shift_audit import publish_table
from stage27_uav_collect import writer_lock
from validation_metrics import tracking_iou, IOU_PROTOCOL


def first_three_good(overlap, frame):
    good = np.asarray(overlap) >= .5
    starts = np.flatnonzero(np.convolve(good.astype(int), np.ones(3, int), mode="valid") == 3) if len(good) >= 3 else []
    return int(starts[0])+frame if len(starts) else None


def branch_rows(plan, case, entry, shared, tails):
    horizon = plan["design"]["horizon"]
    first = case["frame"]
    gt = np.asarray(entry["ground_truth"], float)[first:first+horizon]
    rows, timelines = [], []
    for q, tail in enumerate(tails):
        boxes = np.vstack([shared["candidate_boxes"][0, q], tail["boxes"]])
        parents = np.vstack([shared["candidate_parents"][0, q], tail["parents"]])
        image_hw = np.vstack([shared["image_hw"], tail["image_hw"]])
        overlap = tracking_iou(boxes, gt)
        geometry = [scene_geometry(p, g, hw, plan["search_factor"], plan["search_size"]) for p,g,hw in zip(parents, gt, image_hw)]
        base = {k:case[k] for k in ("case_id", "sequence", "origin", "seed", "policy", "purpose", "frame")}
        row = dict(base, candidate=q, recorded_choice=case["recorded_choice"], immediate_best=case["immediate_best"],
            is_recorded_choice=q == case["recorded_choice"], frames=horizon, start_iou=float(overlap[0]),
            mean_iou=float(overlap.mean()), following_mean_iou=float(overlap[1:].mean()), endpoint_iou=float(overlap[-1]),
            low_iou_frames=int((overlap < .2).sum()), good_frames=int((overlap >= .5).sum()),
            first_three_good_frame=first_three_good(overlap, first), ends_good=bool(overlap[-1] >= .5),
            mean_gt_coverage=float(np.mean([g["coverage"] for g in geometry])),
            tail_search_calls=int(tail["head_calls"].sum()), logical_policy_search_calls=4*horizon)
        rows.append(row)
        for i, (value, g) in enumerate(zip(overlap, geometry)):
            timelines.append(dict(base, candidate=q, branch_frame=first+i, offset=i, iou=float(value),
                gt_coverage=g["coverage"], gt_center_inside=g["center_in_usable"],
                x=float(boxes[i, 0]), y=float(boxes[i, 1]), w=float(boxes[i, 2]), h=float(boxes[i, 3])))
    return rows, timelines


def contrasts(case, rows):
    recorded = case["recorded_choice"]
    immediate = case["immediate_best"]
    future = int(np.argmax([r["mean_iou"] for r in rows]))
    result = []
    for name, choice, offline in (("q0_minus_recorded", 0, False), ("immediate_best_minus_recorded", immediate, True),
                                  ("future_best_minus_recorded", future, True)):
        a, b = rows[choice], rows[recorded]
        result.append(dict(case_id=case["case_id"], sequence=case["sequence"], origin=case["origin"], seed=case["seed"],
            policy=case["policy"], purpose=case["purpose"], frame=case["frame"], contrast=name, candidate=choice,
            recorded_choice=recorded, offline_GT_choice=offline, start_iou_gain=a["start_iou"]-b["start_iou"],
            mean_iou_gain=a["mean_iou"]-b["mean_iou"], following_mean_iou_gain=a["following_mean_iou"]-b["following_mean_iou"],
            endpoint_iou_gain=a["endpoint_iou"]-b["endpoint_iou"], low_iou_frames_delta=a["low_iou_frames"]-b["low_iou_frames"],
            good_frames_delta=a["good_frames"]-b["good_frames"]))
    return result


def dense_rows(plan, case, entry, shared):
    gt = np.asarray(entry["ground_truth"], float)[case["frame"]]
    rows = []
    for q in range(4):
        parent = shared["candidate_parents"][0, q]
        hw = shared["image_hw"][0]
        _, _, factor = crop_geometry(parent, hw, plan["search_factor"], plan["search_size"])
        boxes = dense_boxes(shared["size"][0, q], shared["offset"][0, q], parent, factor, plan["search_size"], hw)
        raw = shared["raw"][0, q]
        weighted = raw*np.asarray(plan["window"], np.float32)
        order = np.lexsort((np.arange(weighted.size), -weighted.ravel()))
        overlap = tracking_iou(boxes, gt)
        best = int(np.argmax(overlap))
        native = shared["candidate_boxes"][0, q]
        native_iou = float(tracking_iou(native, gt))
        geometry = scene_geometry(parent, gt, hw, plan["search_factor"], plan["search_size"])
        rows.append(dict(case_id=case["case_id"], sequence=case["sequence"], origin=case["origin"], seed=case["seed"],
            policy=case["policy"], frame=case["frame"], candidate=q, is_recorded=q == case["recorded_choice"],
            public_box_iou=native_iou, raw_cell=int(raw.argmax()), hann_cell=int(weighted.argmax()),
            dense_hann_cell_iou=float(overlap[int(weighted.argmax())]), best_dense_gt_iou=float(overlap[best]),
            best_dense_gt_cell=best, best_dense_gt_cell_hann_rank=int(np.flatnonzero(order == best)[0])+1,
            top5_grid_good=bool(np.max(overlap[order[:5]]) >= .5), dense_good_cells=int((overlap >= .5).sum()),
            gt_coverage=geometry["coverage"], gt_center_inside=geometry["center_in_usable"],
            input_gt_width=float(geometry["input_wh"][0]), input_gt_height=float(geometry["input_wh"][1]),
            parent_dx_gt_width=float((parent[0]+parent[2]/2-gt[0]-gt[2]/2)/gt[2]),
            parent_dy_gt_height=float((parent[1]+parent[3]/2-gt[1]-gt[3]/2)/gt[3]),
            output_dx_gt_width=float((native[0]+native[2]/2-gt[0]-gt[2]/2)/gt[2]),
            output_dy_gt_height=float((native[1]+native[3]/2-gt[1]-gt[3]/2)/gt[3]),
            output_width_ratio=float(native[2]/gt[2]), output_height_ratio=float(native[3]/gt[3]),
            hann_peak=float(weighted.max())))
    return rows


def summarize(root):
    root = Path(root).expanduser().resolve()
    plan, inputs = c.load_plan(root)
    run = experiment.completed(plan, root)
    branches, comparisons, scopes, dense, cases, timelines = [], [], [], [], [], []
    for case in plan["cases"]:
        trace, entry, _, _ = c.prior.read_trace(inputs["source"], case["unit"])
        path = experiment.paths(root, case)
        shared = c.read_cache(path, c.metadata(plan, case, "shared_four_probes"))
        if shared is None:
            raise ValueError("Missing shared candidate replay")
        experiment.audit_shared(shared, case, entry, trace, plan)
        tails = []
        for q in range(4):
            tail = c.read_cache(experiment.paths(root, case, q), c.metadata(plan, case, "one_forced_choice_then_original_policy", q))
            if tail is None:
                raise ValueError("Missing forced-choice branch")
            experiment.audit_tail(tail, shared, case, entry, trace, plan, q)
            tails.append(tail)
        rows, timeline = branch_rows(plan, case, entry, shared, tails)
        delta = contrasts(case, rows)
        details = dense_rows(plan, case, entry, shared)
        scopes.append(dict(case_id=case["case_id"], sequence=case["sequence"], origin=case["origin"], seed=case["seed"],
            policy=case["policy"], frame=case["frame"], head_sha256=case["head_sha256"], shared_search_calls=int(shared["head_calls"].sum()),
            tail_search_calls=sum(r["tail_search_calls"] for r in rows), branches=4, horizon=plan["design"]["horizon"],
            unique_initial_output_boxes=len(np.unique(shared["candidate_boxes"][0], axis=0)),
            maximum_source_box_error=0., maximum_decode_error=max(float(x["decode_error"].max()) for x in [shared]+tails),
            training_performed=False))
        cases.append(dict(case_id=case["case_id"], sequence=case["sequence"], origin=case["origin"], seed=case["seed"],
            policy=case["policy"], purpose=case["purpose"], frame=case["frame"], recorded_choice=case["recorded_choice"],
            immediate_best=case["immediate_best"], future_best=int(np.argmax([r["mean_iou"] for r in rows])),
            q0_iou=case["q0_iou"], recorded_iou=case["recorded_iou"], best_decoded_iou=case["best_decoded_iou"],
            recorded_mean_iou=rows[case["recorded_choice"]]["mean_iou"],
            q0_mean_gain=delta[0]["mean_iou_gain"], immediate_best_mean_gain=delta[1]["mean_iou_gain"],
            future_best_mean_gain=delta[2]["mean_iou_gain"], best_dense_gt_iou=max(r["best_dense_gt_iou"] for r in details),
            any_decoded_good=case["best_decoded_iou"] >= .5, any_dense_good=any(r["dense_good_cells"] > 0 for r in details)))
        branches.extend(rows)
        comparisons.extend(delta)
        timelines.extend(timeline)
        dense.extend(details)
    total = sum(r["shared_search_calls"]+r["tail_search_calls"] for r in scopes)
    if total != plan["formal_search_calls"] or len(branches) != plan["counterfactual_branches"]:
        raise ValueError("Summary budget differs from committed collection")
    tables = dict(case_summary=cases, branch_summary=branches, contrasts=comparisons, scope=scopes, dense_focus=dense, timelines=timelines)
    analysis = root/"analysis"
    analysis.mkdir(exist_ok=True)
    with writer_lock(analysis):
        c.prior.c.prior.verify_files(plan["source_file_sha256"])
        c.prior.c.prior.verify_files(plan["code_sha256"])
        for name, rows in tables.items():
            publish_table(analysis/("stage63_"+name+".csv"), rows)
        report = c.seal(dict(schema=c.SCHEMA, status="COMPLETE_MATCHED_ONE_CHOICE_CONTINUATION_DIAGNOSTIC", role=plan["role"],
            plan_fingerprint=plan["fingerprint"], run_fingerprint=run["fingerprint"], source_report_fingerprint=plan["source_report_fingerprint"],
            source_runtime=plan["source_runtime_identity"], runtime=plan["runtime_identity"],
            declared_runtime_differences=plan["declared_runtime_differences"], cases=len(cases),
            source_videos=len({r["sequence"] for r in cases}), branches=len(branches), horizon=plan["design"]["horizon"],
            shared_anchor_calls=plan["shared_anchor_calls"], continuation_tail_calls=plan["tail_calls"],
            formal_search_calls=total, formal_block_calls=7*total, smoke_calls_reused=plan["smoke_calls_reused"], additional_smoke_calls=0,
            maximum_source_box_error=0., maximum_decode_error=max(r["maximum_decode_error"] for r in scopes),
            new_full_trajectories=0, training_performed=False, new_models=0, sgla_weights_unchanged=True,
            offline_development_gt_loaded=True, original_checking_labels_loaded=False,
            full_anchor_response_grids_available=True, counterfactual_continuations_performed=True,
            online_method_established=False, novel_method_established=False, iou_protocol=IOU_PROTOCOL,
            evidence_sha256=run["cache_sha256"],
            table_sha256={"stage63_"+name+".csv":c.file_hash(analysis/("stage63_"+name+".csv")) for name in tables},
            warnings=["Cases/rules use consumed development outcomes; no full-video policy score or method adoption gate.",
                "Every branch shares one incoming predicted history; only the accepted first output changes, then the original policy resumes.",
                "All four anchor probes are shared and charged once; every continuation actually performs four searches per frame.",
                "q0 choice uses no GT; immediate-best and future-best are offline descriptions, never online actors.",
                "The future-best candidate is chosen after observing these same short branches; it is not a learned policy or an independent result.",
                "No-good decoded boxes do not imply no-good dense grid; full maps are newly captured for these replays.",
                "Top5 means five grid cells by float32 Hann score, without peak NMS; this differs from prior local-peak top5 definitions.",
                "Dense public-decode float error can change integer-IoU rounding at a boundary; public boxes and dense Hann boxes are reported separately.",
                "A first three-good-frame run is a local window description, not an uncensored full-video recovery estimate.",
                "Multiple anchors/windows/seeds from a video are dependent; no frame-level significance or independent holdout claim.",
                "A different physical L40 UUID is allowed only as a declared new binding; source-choice replay must still match all recorded arrays exactly.",
                "Recorded counts cover network searches; image decoding/hashing and resets are not deployment timing. No speed claim."],
            next_step="Review q0 and alternative-choice continuation gains with dense-grid support; predeclare any later GT-free controller test"))
        c.publish_identical(analysis/"stage63_diagnostic.json", report)
    print("Stage-63: "+report["status"]+"; no training or adoption test")
    for row in cases:
        print("{} f{} q0={:+.4f} immediate_best={:+.4f} future_best={:+.4f}".format(
            row["case_id"], row["frame"], row["q0_mean_gain"], row["immediate_best_mean_gain"], row["future_best_mean_gain"]))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=nonempty_path)
    summarize(parser.parse_args().input)
