"""Video-paired native-head OOF results and separate same-state probe evidence."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stage27_uav_common import failure_events, nonempty_path, publish_identical, success_auc, tracking_iou
from stage27_uav_collect import writer_lock
from stage28_response_common import dense_boxes
from stage46_shift_audit import publish_table, verify_files
from stage46_motion_summary import paired
from stage48_stability_experiment import audit_trace
import stage48_stability_common as c


def video_metrics(boxes,entry,native_boxes):
    gt=np.asarray(entry["ground_truth"],float)[1:]
    iou=tracking_iou(boxes,gt); original=tracking_iou(native_boxes,gt)
    reliable=original>=.5
    return dict(auc=success_auc(iou),low_iou_fraction=float((iou<.2).mean()),failure_events=len(failure_events(iou)),
        reliable_auc=success_auc(iou[reliable]) if reliable.any() else None,
        introduced_failure=not failure_events(original) and bool(failure_events(iou)),
        mean_abs_log_size_error=float(np.abs(np.log(np.asarray(boxes)[:,2:]/gt[:,2:])).mean()),
        oversize3_fraction=float((np.asarray(boxes)[:,2:]/gt[:,2:]>3).any(1).mean()))


def probe_metrics(data,pack,plan,entry):
    n=len(pack["frame_id"]); s=int(plan["runtime"]["lens_x"]**.5)
    shapes=dict(boxes=(n,2,4),raw=(n,2,s,s),size=(n,2,2,s,s),offset=(n,2,2,s,s))
    if set(data)!=set(shapes) or any(data[k].shape!=v for k,v in shapes.items()) or not all(np.isfinite(v).all() for v in data.values()):
        raise ValueError("Invalid held-out cached-feature probe")
    values=[]; retained=missed=0; sizes=[]
    for i,frame in enumerate(pack["frame_id"]):
        gt=np.asarray(entry["ground_truth"][int(frame)],float)
        pair=[]
        hw=[entry["frames"][int(frame)]["height"],entry["frames"][int(frame)]["width"]]
        for v in range(2):
            factor=plan["search_size"]/pack["crop"][i,v,2]
            dense=dense_boxes(data["size"][i,v],data["offset"][i,v],pack["parents"][i,v],factor,plan["search_size"],hw)
            cell=int(np.argmax(data["raw"][i,v]*np.asarray(plan["window"])))
            if np.max(np.abs(dense[cell]-data["boxes"][i,v]))>.001:
                raise ValueError("Cached-feature probe decode mismatch")
            dense[cell]=data["boxes"][i,v]
            overlap=tracking_iou(dense,gt); iou=float(tracking_iou(data["boxes"][i,v],gt))
            keep=bool((overlap>=.5).any())
            retained+=keep; missed+=keep and iou<.2
            pair.append(iou)
            if pack["size_valid"][i,v]:
                size=data["size"][i,v].reshape(2,-1)[:,pack["size_cell"][i,v]]*pack["crop"][i,v,2]
                sizes.append(float(np.abs(np.log(size/gt[2:])).mean()))
        values.append(pair)
    values=np.asarray(values)
    return dict(unshifted_mean_iou=float(values[:,0].mean()),perturbed_mean_iou=float(values[:,1].mean()),
        paired_perturbation_loss=float((values[:,0]-values[:,1]).mean()),retained_views=retained,
        retention_fraction=retained/(2*n),
        retained_but_low_views=missed,rank_miss_given_retained=None if not retained else missed/retained,
        size_supported_views=len(sizes),gt_cell_log_size_error=float(np.mean(sizes)) if sizes else None)


def summarize(root):
    root=Path(root).expanduser().resolve(); plan=c.load_plan(root)
    run=c.read_sealed(root/"evaluation/run.json")
    if run.get("plan_fingerprint")!=plan["fingerprint"] or run.get("role")!="native_head_own_prediction_oof_evaluation":
        raise ValueError("Mixed evaluation run")
    verify_files(run["model_sha256"])
    verify_files(run["feature_sha256"])
    rows,probes,bindings,missing=[],[],{},[]
    head_calls=block_calls=0
    for entry in plan["entries"]:
        native=c.source_native(plan,entry)["boxes"]
        rows.append(dict(sequence=entry["sequence"],arm="native",seed=0,**video_metrics(native,entry,native)))
        pack_path=c.cache_path(root,entry)
        pack=c.read_cache(pack_path,c.feature_meta(plan,entry))
        if pack is None: raise ValueError("Missing diagnostic feature pack")
        c.audit_features(pack,plan,entry,c.source_native(plan,entry))
        probes.append(dict(sequence=entry["sequence"],arm="native",seed=0,
            **probe_metrics({k:pack[k] for k in ("boxes","raw","size","offset")},pack,plan,entry)))
        for p in (pack_path,pack_path.with_suffix(".receipt.json")): bindings[str(p)]=c.file_hash(p)
        for seed in c.SEEDS:
            for arm in c.ARMS:
                method=arm+"_seed"+str(seed)
                meta=dict(run_fingerprint=run["fingerprint"],sequence=entry["sequence"],method=method)
                path=root/"evaluation/traces"/(entry["sequence"]+"__"+method+".npz")
                probe_path=root/"evaluation/probes"/path.name
                data=c.read_cache(path,meta); probe=c.read_cache(probe_path,dict(meta,role="same_state_cached_features"))
                if data is None or probe is None:
                    missing.append(path.name); continue
                audit_trace(data,entry)
                head_calls+=int(data["head_calls"].sum()); block_calls+=int(data["block_calls"].sum())
                rows.append(dict(sequence=entry["sequence"],arm=arm,seed=seed,**video_metrics(data["boxes"],entry,native)))
                probes.append(dict(sequence=entry["sequence"],arm=arm,seed=seed,**probe_metrics(probe,pack,plan,entry)))
                for p in (path,path.with_suffix(".receipt.json"),probe_path,probe_path.with_suffix(".receipt.json")):
                    bindings[str(p)]=c.file_hash(p)
    if missing:
        print("Stage-48: PILOT_INCOMPLETE; {} missing trajectories/probes".format(len(missing))); return None
    videos=[]; policies=[]
    names=[e["sequence"] for e in plan["entries"]]
    for arm in ("native",)+c.ARMS:
        for name in names:
            selected=[r for r in rows if r["sequence"]==name and r["arm"]==arm]
            if len(selected)!=(1 if arm=="native" else len(c.SEEDS)): raise ValueError("Missing seed/video")
            metrics={k:float(np.mean([r[k] for r in selected])) for k in ("auc","low_iou_fraction","failure_events","mean_abs_log_size_error","oversize3_fraction")}
            metrics["reliable_auc"]=float(np.mean([r["reliable_auc"] for r in selected])) if all(r["reliable_auc"] is not None for r in selected) else None
            videos.append(dict(sequence=name,arm=arm,**metrics,introduced_failure=any(r["introduced_failure"] for r in selected)))
        selected=[r for r in videos if r["arm"]==arm]
        policies.append(dict(arm=arm,videos=len(selected),**{k:float(np.mean([r[k] for r in selected])) for k in
            ("auc","low_iou_fraction","failure_events","mean_abs_log_size_error","oversize3_fraction")},
            introduced_failure_videos=sum(r["introduced_failure"] for r in selected)))
    lookup={(r["sequence"],r["arm"]):r for r in videos}
    draws=np.random.default_rng(c.DESIGN["bootstrap_seed"]).integers(0,len(names),size=(c.DESIGN["bootstrap_samples"],len(names)))
    comparisons=[]
    for left,right in (("rank_shape","supervised"),("rank_shape","native"),("rank","supervised"),("shape","supervised")):
        for metric in ("auc","low_iou_fraction","reliable_auc"):
            if any(lookup[n,left][metric] is None or lookup[n,right][metric] is None for n in names): continue
            comparisons.append(paired([lookup[n,left][metric]-lookup[n,right][metric] for n in names],left+"_minus_"+right,metric,draws))
    for label,formula in (("rank_order_average",lambda x:.5*(x["rank"]-x["supervised"]+x["rank_shape"]-x["shape"])),
        ("shape_order_average",lambda x:.5*(x["shape"]-x["supervised"]+x["rank_shape"]-x["rank"])),
        ("rank_by_shape_interaction",lambda x:x["rank_shape"]-x["rank"]-x["shape"]+x["supervised"])):
        comparisons.append(paired([formula({a:lookup[n,a]["auc"] for a in c.ARMS}) for n in names],label,"auc",draws))
    find=lambda label,metric:next(r for r in comparisons if r["contrast"]==label and r["metric"]==metric)
    primary=find("rank_shape_minus_supervised","auc"); overall=find("rank_shape_minus_native","auc")
    reliable=find("rank_shape_minus_native","reliable_auc") if any(r["contrast"]=="rank_shape_minus_native" and r["metric"]=="reliable_auc" for r in comparisons) else None
    selected=next(r for r in policies if r["arm"]=="rank_shape"); base=next(r for r in policies if r["arm"]=="native")
    probe_policies=[]
    for arm in ("native",)+c.ARMS:
        record=dict(arm=arm,videos=len(names))
        for metric in ("unshifted_mean_iou","perturbed_mean_iou","paired_perturbation_loss","retention_fraction","rank_miss_given_retained","gt_cell_log_size_error"):
            values=[]
            for name in names:
                seed_values=[r[metric] for r in probes if r["arm"]==arm and r["sequence"]==name]
                if all(v is not None for v in seed_values): values.append(float(np.mean(seed_values)))
            record[metric]=float(np.mean(values)) if values else None
            record[metric+"_supported_videos"]=len(values)
        probe_policies.append(record)
    primary_seed_gains=[float(np.mean([next(r["auc"] for r in rows if r["sequence"]==n and r["arm"]=="rank_shape" and r["seed"]==seed)-
        next(r["auc"] for r in rows if r["sequence"]==n and r["arm"]=="supervised" and r["seed"]==seed) for n in names])) for seed in c.SEEDS]
    guard=bool(reliable and reliable["mean_delta"]>=c.DESIGN["reliable_mean_floor"] and selected["introduced_failure_videos"]==0 and selected["low_iou_fraction"]<=base["low_iou_fraction"])
    state="PROMISING_NATIVE_HEAD_STABILITY_PILOT" if primary["mean_delta"]>=c.DESIGN["practical_auc_gain"] and overall["mean_delta"]>=0 and guard else "REVIEW_NATIVE_HEAD_STABILITY_PILOT"
    with writer_lock(root/"analysis"):
        tables=dict(policy_summary=policies,video_summary=videos,seed_video_summary=rows,paired=comparisons,probe_summary=probes,probe_policy_summary=probe_policies)
        for name,values in tables.items(): publish_table(root/"analysis"/("stage48_"+name+".csv"),values)
        verify_files(bindings); verify_files(run["model_sha256"]); verify_files(plan["source_file_sha256"])
        report=c.seal(dict(schema=c.SCHEMA,status=state,role="consumed_development_native_head_two_factor_oof",
            plan_fingerprint=plan["fingerprint"],run_fingerprint=run["fingerprint"],primary=primary,overall=overall,
            retention_guard=guard,videos=len(names),seeds=list(c.SEEDS),trained_models=16,full_oof_trajectories=96,
            primary_seed_gains=primary_seed_gains,reliable_guard=reliable,
            low_iou_fraction_delta=selected["low_iou_fraction"]-base["low_iou_fraction"],
            introduced_failure_videos=selected["introduced_failure_videos"],
            native_stable_videos=sum(r["arm"]=="native" and r["failure_events"]==0 for r in rows),
            inference_head_calls=head_calls,inference_block_calls=block_calls,exact_native_call_budget=True,
            source_file_sha256=bindings,model_sha256=run["model_sha256"],
            table_sha256={"stage48_"+n+".csv":c.file_hash(root/"analysis"/("stage48_"+n+".csv")) for n in tables},
            training_performed=True,external_generalization_established=False,original_checking_labels_loaded=False,
            warnings=["All twelve source groups and previous outcomes informed design; grouped OOF remains development evidence.",
                "Seeds average within videos before video bootstrap; fixed overlapping fits and two seeds do not establish training uncertainty.",
                "All arms use identical cached native features, initialization, two views, labels, optimizer and update count; only training losses differ.",
                "Ranking masks use frozen native GT-qualified boxes; they do not guarantee trained geometry stays matched. Probe retention is reported separately.",
                "The original head structure and full SGLA selector/candidates remain. No motion correction, extra inference branch or compression/speed claim.",
                "Same-state cached-feature probes are immediate IoU diagnostics, not alternative rollouts. Reduced perturbation loss alone is not improved robustness.",
                "Reliable cohorts, GT-center size errors and oversize ratios are offline diagnostics, not online gates."],
            next_step="Review full-video gains, stable-video guard and absolute probe values; develop promising components without treating this cohort as fresh validation"))
        publish_identical(root/"analysis/stage48_decision.json",report)
    for r in policies: print("{:<13} AUC={:.4f} low-IoU={:.2%}".format(r["arm"],r["auc"],r["low_iou_fraction"]))
    print("Stage-48: "+state)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--input",required=True,type=nonempty_path)
    summarize(p.parse_args().input)
