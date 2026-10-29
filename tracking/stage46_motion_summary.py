"""Paired source-video metrics for the frozen native-SGLA motion-search pilot."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stage4_common import write_csv, write_json
from stage27_uav_common import failure_events, scene_geometry, success_auc, tracking_iou
import stage46_motion_common as c


def metrics(data, entry, native, search_factor, search_size):
    gt = np.asarray(entry["ground_truth"], float)[1:]
    iou = tracking_iou(data["boxes"], gt)
    baseline_iou = tracking_iou(native["boxes"], gt)
    reliable = baseline_iou >= .5
    events = failure_events(iou)
    side = np.sqrt(np.prod(data["previous"][:,2:],axis=1))
    valid = data["motion_valid"]
    normalized = np.linalg.norm(data["camera_displacement"],axis=1)/side
    gt_center, gt_side = gt[:,:2]+gt[:,2:]/2, np.sqrt(np.prod(gt[:,2:],axis=1))
    parent_center = data["parents"][:,:2]+data["parents"][:,2:]/2
    previous_center = data["previous"][:,:2]+data["previous"][:,2:]/2
    parent_error = np.linalg.norm(parent_center-gt_center,axis=1)/gt_side
    previous_error = np.linalg.norm(previous_center-gt_center,axis=1)/gt_side
    coverage = np.array([scene_geometry(p,g,hw,search_factor,search_size)["coverage"]
                         for p,g,hw in zip(data["parents"],gt,data["image_hw"])])
    times = data["pipeline_ms"][10:]
    net_times = data["network_ms"][10:]
    result = dict(auc=success_auc(iou), low_iou_fraction=float((iou < .2).mean()),
        failure_events=len(events), longest_failure_fraction=max([e["end"]-e["start"]+1 for e in events]+[0])/len(iou),
        reliable_auc=success_auc(iou[reliable]) if reliable.any() else None,
        native_stable=len(failure_events(baseline_iou)) == 0,
        introduced_failure=len(failure_events(baseline_iou)) == 0 and bool(events),
        valid_motion_fraction=float(valid.mean()),
        median_background_shift_target_sides=float(np.median(normalized[valid])) if valid.any() else 0.,
        changed_crop_fraction=float((np.linalg.norm(data["shift"],axis=1) > 1e-8).mean()),
        capped_fraction=float(data["cap_hit"].mean()), head_calls=int(data["head_calls"].sum()),
        block_calls=int(data["block_calls"].sum()),
        mean_parent_center_error_target_sides=float(parent_error.mean()),
        mean_center_error_reduction_from_own_previous=float((previous_error-parent_error).mean()),
        mean_crop_coverage=float(coverage.mean()),inside_crop_fraction=float((coverage >= .9).mean()),
        median_network_ms=float(np.median(net_times)) if len(net_times) else None,
        median_pipeline_ms=float(np.median(times)) if len(times) else None,
        p95_pipeline_ms=float(np.quantile(times,.95)) if len(times) else None)
    for reason,label in enumerate(("shape","corners","tracks","fit","quality","geometry"),1):
        result["rejected_"+label+"_frames"] = int((data["motion_reason"] == reason).sum())
    return result


def paired(values, name, metric, draws):
    values = np.asarray(values, float)
    means = values[draws].mean(1)
    sign = -1 if metric in ("low_iou_fraction", "longest_failure_fraction") else 1
    n = len(values)
    return dict(contrast=name, metric=metric, videos=n, mean_delta=float(values.mean()),
        median_delta=float(np.median(values)), ci_low=float(np.quantile(means,.025)),
        ci_high=float(np.quantile(means,.975)), wins=int((sign*values > 1e-12).sum()),
        ties=int((np.abs(values) <= 1e-12).sum()), losses=int((sign*values < -1e-12).sum()),
        minimum_leave_one_video_out=float(np.min((values.sum()-values)/(n-1))) if n > 1 else None,
        favorable_direction="negative" if sign < 0 else "positive")


def summarize(root):
    root = Path(root)
    plan, inputs = c.load_plan(root)
    run = c.load_sealed(root/"run.json")
    if run["plan_fingerprint"] != plan["fingerprint"]:
        raise ValueError("Run/plan mismatch")
    _, source, traces, _ = inputs
    videos, caches, hashes, missing = [], {}, {}, []
    for name in plan["sequences"]:
        entry, native = traces[name]
        for method in c.METHODS:
            path = c.cache_path(root,name,method)
            data = c.load_result(path,c.metadata(run,name,method,False))
            if data is None:
                missing.append(path.name)
                continue
            c.audit_video(data,entry["frames"],entry["initial_box"],method,native)
            hashes[path.name] = c.file_hash(path)
            caches[name,method] = data
            videos.append(dict(sequence=name,source_group_proxy=entry["source_group_proxy"],method=method,
                               tracked_frames=len(data["boxes"]),**metrics(data,entry,native,source["search_factor"],source["search_size"])))
    output = root/"analysis"
    output.mkdir(parents=True,exist_ok=True)
    if missing:
        write_json(output/"stage46_decision.json",c.seal(dict(schema=c.SCHEMA,status="PILOT_INCOMPLETE",
            plan_fingerprint=plan["fingerprint"],missing_files=missing)))
        print("Stage-46: PILOT_INCOMPLETE; {} files missing".format(len(missing)))
        return
    policies = []
    for method in c.METHODS:
        rows = [r for r in videos if r["method"] == method]
        row = dict(method=method,videos=len(rows),auc=float(np.mean([r["auc"] for r in rows])),
                   low_iou_fraction=float(np.mean([r["low_iou_fraction"] for r in rows])),
                   failure_events=float(np.mean([r["failure_events"] for r in rows])),
                   introduced_failure_videos=sum(r["introduced_failure"] for r in rows),
                   head_calls=sum(r["head_calls"] for r in rows),block_calls=sum(r["block_calls"] for r in rows))
        for metric in ("network_ms","pipeline_ms"):
            samples = np.concatenate([caches[name,method][metric][10:] for name in plan["sequences"]])
            row["median_"+metric] = float(np.median(samples)) if len(samples) else None
            row["p95_"+metric] = float(np.quantile(samples,.95)) if len(samples) else None
        policies.append(row)
        print("{:<17} AUC={:.4f} low-IoU={:.2%}".format(method,row["auc"],row["low_iou_fraction"]),flush=True)
    lookup = {(r["sequence"],r["method"]):r for r in videos}
    names = plan["sequences"]
    rng = np.random.default_rng(c.DESIGN["bootstrap_seed"])
    draws = rng.integers(0,len(names),size=(c.DESIGN["bootstrap_samples"],len(names)))
    pairs = []
    for left,right in (("camera_velocity","velocity"),("camera_velocity","sgla"),
                       ("camera_only","sgla"),("velocity","sgla")):
        for metric in ("auc","low_iou_fraction","longest_failure_fraction","reliable_auc"):
            if metric == "reliable_auc" and any(lookup[n,left][metric] is None or lookup[n,right][metric] is None for n in names):
                continue  # do not silently change the video cohort
            values = [lookup[n,left][metric]-lookup[n,right][metric] for n in names]
            pairs.append(paired(values,left+"_minus_"+right,metric,draws))
    interaction = [lookup[n,"camera_velocity"]["auc"]-lookup[n,"camera_only"]["auc"]-
                   lookup[n,"velocity"]["auc"]+lookup[n,"sgla"]["auc"] for n in names]
    pairs.append(paired(interaction,"camera_by_velocity_interaction","auc",draws))
    primary = next(r for r in pairs if r["contrast"] == "camera_velocity_minus_velocity" and r["metric"] == "auc")
    overall = next(r for r in pairs if r["contrast"] == "camera_velocity_minus_sgla" and r["metric"] == "auc")
    support = [r for r in videos if r["method"] == "sgla" and r["valid_motion_fraction"] >= .5 and r["median_background_shift_target_sides"] >= .05]
    final = next(r for r in policies if r["method"] == "camera_velocity")
    base = next(r for r in policies if r["method"] == "sgla")
    stable_count = sum(r["native_stable"] for r in videos if r["method"] == "sgla")
    guarded = stable_count > 0 and final["low_iou_fraction"] <= base["low_iou_fraction"] and final["introduced_failure_videos"] == 0
    promising = len(support) >= 4 and primary["mean_delta"] >= c.DESIGN["auc_practice_target"] and overall["mean_delta"] >= c.DESIGN["auc_practice_target"] and guarded
    state = "PROMISING_CAMERA_SEARCH_PILOT" if promising else "REVIEW_CAMERA_SEARCH_PILOT"
    if len(support) < 4:
        state = "LIMITED_BACKGROUND_MOTION_SUPPORT"
    influence = []
    for name in names:
        p = lookup[name,"camera_velocity"]["auc"]-lookup[name,"velocity"]["auc"]
        g = lookup[name,"camera_velocity"]["auc"]-lookup[name,"sgla"]["auc"]
        influence.append(dict(sequence=name,primary_auc_delta=p,overall_auc_delta=g,
            primary_leave_one_out=(primary["mean_delta"]*len(names)-p)/(len(names)-1),
            native_motion_valid_fraction=lookup[name,"sgla"]["valid_motion_fraction"],
            native_median_background_target_sides=lookup[name,"sgla"]["median_background_shift_target_sides"]))
    max_error = max(float(np.max(np.abs(caches[n,"sgla"]["boxes"]-traces[n][1]["boxes"]))) for n in names)
    decision = c.seal(dict(schema=c.SCHEMA,status=state,role="consumed_development_native_sgla_motion_prototype",
        primary=primary,overall=overall,retention_guard=guarded,native_stable_videos=stable_count,
        background_supported_videos=len(support),
        support_rule="native SGLA: valid background fits >=50% and median fitted shift >=0.05 predicted target sides; at least four videos",
        practice_target="camera_velocity AUC >=+0.5 vs velocity and native; low-IoU mean not worse, no new native-stable failure",
        videos=len(names),full_trajectories=len(names)*len(c.METHODS),max_native_box_error=max_error,
        exact_native_call_budget=True,weights_unchanged=True,training_performed=False,
        original_checking_labels_loaded=False,external_generalization_established=False,
        plan_fingerprint=plan["fingerprint"],run_fingerprint=run["fingerprint"],cache_sha256=hashes,
        next_step="Inspect paired video effects, fit reliability, crop shifts and cost; refine on development evidence or plan independent source groups before using new outcomes",
        warnings=["Twelve already-consumed Stage-27 development source proxies; no fresh holdout or camera-ground-truth validation.",
                  "The original SGLA template, network, selector, candidate structure and head remain frozen; only search centers change.",
                  "Every policy computes background motion with its own previous box mask; Stage-27 GT-masked proxies never enter the actor.",
                  "Partial-affine background motion is a proxy; moving background objects, parallax and rolling shutter can violate it.",
                  "Bootstrap resamples videos, not frames; this is a fixed untrained algorithm and intervals condition on this development cohort.",
                  "The factorial interaction compares whole closed-loop policies, not unique attribution of one crop or layer.",
                  "All policies include shadow background estimation; network and instrumented pipeline time are separate, excluding image reads/hashes and first ten frames. No deployment speedup claim.",
                  "Global motion compensation and motion prediction are established ideas. This prototype tests added value on SGLA, not absolute novelty or a compression gain."]))
    write_csv(output/"stage46_video_summary.csv",videos)
    write_csv(output/"stage46_policy_summary.csv",policies)
    write_csv(output/"stage46_paired.csv",pairs)
    write_csv(output/"stage46_video_influence.csv",influence)
    write_json(output/"stage46_decision.json",decision)
    print("Stage-46:",state)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    summarize(p.parse_args().input)
