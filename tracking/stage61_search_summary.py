"""Paired video effects with deterministic native runs counted once per video."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from stage27_uav_common import tracking_iou, failure_events, nonempty_path
from stage27_uav_collect import writer_lock
from stage46_shift_audit import publish_table
from stage46_motion_summary import paired
from stage51_state_summary import describe
import stage61_search_common as c
import stage61_search_runtime as actor
import stage61_search_experiment as experiment

METRICS=("auc","low_iou_fraction","failure_events","longest_failure_fraction","reliable_auc",
    "mean_abs_log_size_error","oversize3_fraction","unrecovered_events","recovered_events","relapse_events","native_failure_auc")
USAGE=("selected_noncurrent_fraction","changed_prediction_fraction","changed_crop_fraction","mean_distinct_crop_pixels")

def contrast(values,label,metric,design,family=False):
    values=np.asarray(values,float)
    draws=np.random.default_rng(design["bootstrap_seed"]).integers(0,len(values),size=(design["bootstrap_samples"],len(values)))
    out=paired(values,label,metric,draws)
    sign=-1 if metric in ("low_iou_fraction","longest_failure_fraction") else 1
    # Unlike frozen Stage-60, the adverse video is chosen in the metric's actual favorable direction.
    out["worst_video_delta"]=float(values.min() if sign>0 else values.max())
    loo=(values.sum()-values)/(len(values)-1) if len(values)>1 else np.array([])
    out["worst_leave_one_video_out_delta"]=float(loo.min() if sign>0 else loo.max()) if len(loo) else None
    out["family_ci_low"],out["family_ci_high"]=(map(float,np.quantile(values[draws].mean(1),[.0125,.9875])) if family else (None,None))
    return out

def summarize(root):
    root=Path(root).expanduser().resolve(); plan,source=c.load_plan(root)
    gate=experiment.smoke_gate(plan,root); run=experiment.completed(plan,root)
    records=[(r,None) for r in source["records"]]
    traces=dict(source["reused"]); hashes={}; calls=0; scope=[]; maximum_decode=0.
    for unit in plan["units"]:
        path=root/"evaluation/traces"/(c.key(unit)+".npz")
        data=c.read_cache(path,dict(c.meta(plan,unit,"full_evaluation"),model_identity=experiment.model_identity(plan,records,unit)))
        entry=next(e for e in plan["entries"] if e["sequence"]==unit["sequence"])
        if data is None or len(data["frame_id"])!=len(entry["frames"])-1: raise ValueError("Missing full trajectory")
        actor.audit(data,entry,plan,unit["policy"])
        maximum_decode=max(maximum_decode,float(data["decode_error"].max()))
        count=int(data["head_calls"].sum()); calls+=count; experiment.bind(hashes,path)
        identity=experiment.model_identity(plan,records,unit)
        expected_head=identity.split(":",1)[1] if unit["origin"]=="endpoint" else gate["observed_head_sha256"]["native_0"]
        if run["observed_head_sha256"][c.key(unit)]!=expected_head: raise ValueError("Inference head identity changed")
        traces[unit["sequence"],unit["origin"],unit["policy"],unit["seed"]]=data
        scope.append(dict(**unit,head_sha256=expected_head,tracked_frames=len(data["frame_id"]),
            head_calls=count,block_calls=int(data["block_calls"].sum()),calls_per_frame=4,
            maximum_decode_error=float(data["decode_error"].max()),weights_unchanged=True,training_performed=False))
    if (calls!=plan["formal_search_calls"] or hashes!=run["cache_sha256"] or maximum_decode!=run["maximum_decode_error"]):
        raise ValueError("Completed call/receipt/decode budget mismatch")
    names=[e["sequence"] for e in plan["entries"]]; rows=[]; events=[]; proposals=[]; times={}
    for entry in plan["entries"]:
        name=entry["sequence"]; native=traces[name,"native","current",0]
        gt=np.asarray(entry["ground_truth"],float)[1:]; native_mask=tracking_iou(native["boxes"],gt)<.2
        for origin in c.ORIGINS:
            seeds=[0] if origin=="native" else plan["seeds"]
            for seed in seeds:
                current=traces[name,origin,"current",seed]
                baseline_stable=not failure_events(tracking_iou(current["boxes"],gt))
                for policy in c.POLICIES:
                    data=traces[name,origin,policy,seed]; arm=origin+"_"+policy
                    metrics,ev=describe(data,entry,native["boxes"],native_mask)
                    metrics["native_failure_auc"]=metrics.pop("source_supervised_failure_auc")
                    changed=False
                    usage=dict.fromkeys(USAGE,0.)
                    if policy!="current":
                        idx=np.arange(len(data["boxes"])); selected=np.asarray(data["selected"],int)
                        usage=dict(selected_noncurrent_fraction=float((selected!=0).mean()),
                            changed_prediction_fraction=float((np.max(np.abs(data["boxes"]-data["candidate_boxes"][:,0]),axis=1)>1e-6).mean()),
                            changed_crop_fraction=float(np.any(data["crop"][idx,selected]!=data["crop"][:,0],axis=1).mean()),
                            mean_distinct_crop_pixels=float(np.mean([len(set(a.tolist())) for a in data["crop_sha256"]])))
                        changed=bool(usage["changed_prediction_fraction"]>0)
                    else: usage["mean_distinct_crop_pixels"]=1.
                    new=(origin=="native" and policy!="current") or (origin=="endpoint" and policy=="recrop")
                    own_events=failure_events(tracking_iou(data["boxes"],gt))
                    row=dict(sequence=name,origin=origin,policy=policy,arm=arm,seed=seed,
                        provenance="new_inference" if new else "reused_frozen_source",
                        introduced_vs_own_current=bool(baseline_stable and own_events),changed=changed,**metrics,**usage)
                    rows.append(row); events.extend(dict(sequence=name,arm=arm,seed=seed,**v) for v in ev)
                    proposals.append(dict(sequence=name,arm=arm,seed=seed,**usage))
                    if new: times.setdefault(arm,[]).extend(data["pipeline_ms"][10:].tolist())
    arms=[o+"_"+p for o in c.ORIGINS for p in c.POLICIES]; videos=[]; policies=[]
    for arm in arms:
        for name in names:
            selected=[r for r in rows if (r["arm"],r["sequence"])==(arm,name)]
            count=1 if arm.startswith("native_") else len(plan["seeds"])
            if len(selected)!=count: raise ValueError("Native duplicated or endpoint seed missing")
            row=dict(sequence=name,arm=arm,introduced_failure=any(r["introduced_failure"] for r in selected),
                introduced_vs_own_current=any(r["introduced_vs_own_current"] for r in selected))
            for metric in METRICS+USAGE:
                values=[r[metric] for r in selected]
                row[metric]=float(np.mean(values)) if all(v is not None for v in values) else None
            videos.append(row)
        selected=[r for r in videos if r["arm"]==arm]; origin,policy=arm.split("_",1)
        logical_calls=sum(int(traces[n,origin,policy,seed]["head_calls"].sum()) for n in names for seed in ([0] if origin=="native" else plan["seeds"]))
        own_time=times.get(arm,[])
        row=dict(arm=arm,videos=len(names),runs_per_video=1 if origin=="native" else len(plan["seeds"]),
            calls_per_frame=1 if policy=="current" else 4,logical_head_calls=logical_calls,
            newly_executed_head_calls=logical_calls if arm in times else 0,
            introduced_failure_videos=sum(r["introduced_failure"] for r in selected),
            introduced_vs_own_current_videos=sum(r["introduced_vs_own_current"] for r in selected),
            median_new_pipeline_ms=float(np.median(own_time)) if own_time else None,
            p95_new_pipeline_ms=float(np.quantile(own_time,.95)) if own_time else None)
        for metric in METRICS+USAGE:
            values=[r[metric] for r in selected if r[metric] is not None]
            row[metric]=float(np.mean(values)) if values else None
            if metric in ("reliable_auc","native_failure_auc"): row[metric+"_supported_videos"]=len(values)
        policies.append(row)
    lookup={(r["sequence"],r["arm"]):r for r in videos}; paired_rows=[]
    comparisons=[(o+"_"+a,o+"_"+b) for o in c.ORIGINS for a,b in (("history","recrop"),("history","current"),("recrop","current"))]
    for left,right in comparisons:
        for metric in ("auc","low_iou_fraction","longest_failure_fraction","reliable_auc","native_failure_auc"):
            support=[n for n in names if lookup[n,left][metric] is not None and lookup[n,right][metric] is not None]
            if support:
                family=metric=="auc" and left=="native_history" and right in ("native_recrop","native_current")
                out=contrast([lookup[n,left][metric]-lookup[n,right][metric] for n in support],left+"_minus_"+right,metric,plan["design"],family)
                out.update(supported_videos="|".join(support),excluded_videos="|".join(n for n in names if n not in support))
                paired_rows.append(out)
    interaction=contrast([lookup[n,"endpoint_history"]["auc"]-lookup[n,"endpoint_recrop"]["auc"]-
                         lookup[n,"native_history"]["auc"]+lookup[n,"native_recrop"]["auc"] for n in names],
                        "endpoint_minus_native_history_vs_recrop_interaction","auc",plan["design"])
    paired_rows.append(interaction)
    get=lambda label,metric:next(r for r in paired_rows if (r["contrast"],r["metric"])==(label,metric))
    primary=get("native_history_minus_native_recrop","auc"); overall=get("native_history_minus_native_current","auc")
    reliable=get("native_history_minus_native_current","reliable_auc"); low=get("native_history_minus_native_current","low_iou_fraction")
    own=next(r for r in policies if r["arm"]=="native_history")
    changed=sum(lookup[n,"native_history"]["changed_prediction_fraction"]>0 for n in names)
    support=changed>=min(4,len(names)) and own["changed_prediction_fraction"]>1e-4
    guard=own["introduced_vs_own_current_videos"]==0 and low["mean_delta"]<=0 and reliable["ci_low"]>=plan["design"]["reliable_ci_floor"]
    promising=(min(primary["mean_delta"],overall["mean_delta"])>=plan["design"]["practical_auc_gain"] and
               min(primary["family_ci_low"],overall["family_ci_low"])>0 and guard and support)
    influence=[]
    for n in names:
        p=lookup[n,"native_history"]["auc"]-lookup[n,"native_recrop"]["auc"]
        o=lookup[n,"native_history"]["auc"]-lookup[n,"native_current"]["auc"]
        influence.append(dict(sequence=n,native_history_minus_recrop=p,native_history_minus_current=o,
            primary_leave_one_out=(primary["mean_delta"]*len(names)-p)/(len(names)-1) if len(names)>1 else None,
            endpoint_history_minus_recrop=lookup[n,"endpoint_history"]["auc"]-lookup[n,"endpoint_recrop"]["auc"],
            source_stage60_response_minus_current=lookup[n,"endpoint_history"]["auc"]-lookup[n,"endpoint_current"]["auc"]))
    tables=dict(policy_summary=policies,paired=paired_rows,video_summary=videos,seed_video_summary=rows,
                scope_summary=scope,proposal_summary=proposals,video_influence=influence,events=events)
    with writer_lock(root/"analysis"):
        for label,data in tables.items():
            keys=list(dict.fromkeys(k for row in data for k in row))
            publish_table(root/"analysis"/("stage61_"+label+".csv"),[{k:r.get(k) for k in keys} for r in data])
        c.verify_files(plan["source_file_sha256"]); c.verify_files(hashes)
        report=c.seal(dict(schema=c.SCHEMA,status="PROMISING_NATIVE_HISTORY_RECROP_PILOT" if promising else "REVIEW_NATIVE_HISTORY_RECROP_PILOT",
            role=plan["role"],plan_fingerprint=plan["fingerprint"],run_fingerprint=run["fingerprint"],
            pilot_pass=bool(promising),primary=primary,overall=overall,retention_guard=bool(guard),support_pass=bool(support),
            origin_by_proposal_interaction=interaction,native_deterministic_runs_per_video=1,endpoint_seeds=plan["seeds"],videos=len(names),
            new_full_trajectories=plan["new_full_trajectories"],reused_full_trajectories=plan["reused_full_trajectories"],
            formal_search_calls=calls,formal_block_calls=calls*7,additional_smoke_search_calls=gate["extra_smoke_search_calls"],
            maximum_smoke_source_box_error=gate["maximum_source_box_error"],maximum_decode_error=run["maximum_decode_error"],
            exact_four_pass_budget=True,new_models=0,training_performed=False,sgla_weights_unchanged=True,
            original_checking_labels_loaded=False,external_generalization_established=False,novel_method_established=False,
            evidence_sha256=hashes,table_sha256={"stage61_"+n+".csv":c.file_hash(root/"analysis"/("stage61_"+n+".csv")) for n in tables},
            warnings=["Same consumed twelve development source groups; no independent confirmation.",
                "Original V1 runs are deterministic and counted once; E seeds average within videos before conditional video bootstrap.",
                "Recrop uses the current image's first decoded box, not older history, for the second center/size. q0 is reused, so total calls stay four.",
                "This control tests a current-frame refinement alternative; it does not exhaust all ordinary multiscale or spatial searches.",
                "Whole policies follow their own accepted boxes. Origin interaction is conditional on existing E fits, not a pure training causal effect.",
                "Only new traces contribute new timing; historical E response timings are not remeasured. Image reads/hashes and first ten frames are excluded.",
                "Current controls and E history prefixes are live-replayed in smoke only; full completed source controls are reused with sealed provenance.",
                "No V2, template update, scorer learning, crop-amplitude search or deployment speedup is claimed."],
            next_step="Preserve the fixed comparison; inspect native transfer, equal-call recrop control and failure retention before independent-source validation"))
        c.publish_identical(root/"analysis/stage61_decision.json",report)
    for r in policies: print("{:<19} AUC={:.4f} low-IoU={:.2%}".format(r["arm"],r["auc"],r["low_iou_fraction"]))
    print("Stage-61: "+report["status"])
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--input",required=True,type=nonempty_path)
    summarize(parser.parse_args().input)
