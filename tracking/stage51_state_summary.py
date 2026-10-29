"""Video-paired state-source training results, retention and recovery descriptions."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from stage27_uav_common import failure_events,nonempty_path,tracking_iou,success_auc
from stage27_uav_collect import writer_lock
from stage46_shift_audit import publish_table
from stage46_motion_summary import paired
from stage48_stability_experiment import audit_trace
from stage48_stability_summary import video_metrics
from stage51_state_experiment import feature_gate,model_records,smoke_gate
import stage51_state_common as c

METRICS=("auc","low_iou_fraction","failure_events","longest_failure_fraction","reliable_auc",
    "mean_abs_log_size_error","oversize3_fraction","unrecovered_events","recovered_events","relapse_events",
    "source_supervised_failure_auc")


def describe(data,entry,native_boxes,source_failure_mask):
    gt=np.asarray(entry["ground_truth"],float)[1:]; iou=tracking_iou(data["boxes"],gt); events=failure_events(iou)
    mask=np.asarray(source_failure_mask,bool)
    result=video_metrics(data["boxes"],entry,native_boxes)
    result.update(longest_failure_fraction=max([e["end"]-e["start"]+1 for e in events]+[0])/len(iou),
        unrecovered_events=sum(e["censor"]=="video_end" for e in events),
        recovered_events=sum(e["recovery"] is not None for e in events),
        relapse_events=sum(events[j-1]["recovery"] is not None for j in range(1,len(events))),
        source_supervised_failure_auc=success_auc(iou[mask]) if mask.any() else None,
        first_failure_frame=events[0]["start"]+1 if events else None)
    rows=[dict(event=i+1,onset_frame=e["start"]+1,confirmation_frame=e["confirmation"]+1,end_frame=e["end"]+1,
        recovery_frame=None if e["recovery"] is None else e["recovery"]+1,censor=e["censor"],
        duration_frames=e["end"]-e["start"]+1,low_iou_frames=int((iou[e["start"]:e["end"]+1]<.2).sum())) for i,e in enumerate(events)]
    return result,rows


def summarize(root):
    root=Path(root).expanduser().resolve(); plan,source=c.load_plan(root)
    smoke_gate(plan,root); collection=feature_gate(plan,root); _,models=model_records(plan,root)
    run=c.read_sealed(root/"evaluation/run.json")
    if (run.get("role")!="native_state_source_oof_evaluation" or run.get("plan_fingerprint")!=plan["fingerprint"] or
        run.get("model_sha256")!=models or run.get("collection_fingerprint")!=collection["fingerprint"]):
        raise ValueError("Mixed Stage-51 evaluation")
    c.verify_files(models); bindings={}; rows=[]; events=[]; missing=[]; calls=blocks=0
    oldrun=c.prior.read_sealed(Path(plan["source_stage48"])/"evaluation/run.json")
    for entry in plan["entries"]:
        native=c.prior.source_native(source,entry)
        reference={}
        for seed in c.SEEDS:
            method="supervised_seed"+str(seed)
            path=Path(plan["source_stage48"])/"evaluation/traces"/(entry["sequence"]+"__"+method+".npz")
            value=c.prior.read_cache(path,dict(run_fingerprint=oldrun["fingerprint"],sequence=entry["sequence"],method=method))
            if value is None: raise ValueError("Missing frozen supervised OOF reference")
            audit_trace(value,entry); reference[seed]=value
        source_failure_mask=np.any([tracking_iou(v["boxes"],np.asarray(entry["ground_truth"],float)[1:])<.2 for v in reference.values()],axis=0)
        selected=[("native",0,native)]+[("source_supervised",s,v) for s,v in reference.items()]
        for seed in c.SEEDS:
            for arm in c.ARMS:
                method=arm+"_seed"+str(seed); path=root/"evaluation/traces"/(entry["sequence"]+"__"+method+".npz")
                data=c.read_cache(path,dict(run_fingerprint=run["fingerprint"],sequence=entry["sequence"],method=method))
                if data is None: missing.append(path.name); continue
                audit_trace(data,entry); calls+=int(data["head_calls"].sum()); blocks+=int(data["block_calls"].sum())
                selected.append((arm,seed,data))
                for p in (path,path.with_suffix(".receipt.json")): bindings[str(p)]=c.file_hash(p)
        for arm,seed,data in selected:
            m,ev=describe(data,entry,native["boxes"],source_failure_mask)
            rows.append(dict(sequence=entry["sequence"],arm=arm,seed=seed,**m))
            events.extend(dict(sequence=entry["sequence"],arm=arm,seed=seed,**r) for r in ev)
    if missing:
        print("Stage-51: PILOT_INCOMPLETE; {} missing trajectories".format(len(missing))); return None
    names=[e["sequence"] for e in plan["entries"]]; arms=("native","source_supervised")+c.ARMS
    videos=[]; policies=[]
    for arm in arms:
        for name in names:
            selected=[r for r in rows if r["sequence"]==name and r["arm"]==arm]
            if len(selected)!=(1 if arm=="native" else len(c.SEEDS)): raise ValueError("Missing method/seed/video")
            record=dict(sequence=name,arm=arm,introduced_failure=any(r["introduced_failure"] for r in selected))
            for metric in METRICS:
                vals=[r[metric] for r in selected]
                record[metric]=float(np.mean(vals)) if all(v is not None for v in vals) else None
            videos.append(record)
        selected=[r for r in videos if r["arm"]==arm]
        record=dict(arm=arm,videos=len(names),introduced_failure_videos=sum(r["introduced_failure"] for r in selected),
            provenance="reused_frozen_source" if arm in ("native","source_supervised") else "new_matched_training")
        for metric in METRICS:
            vals=[r[metric] for r in selected if r[metric] is not None]
            record[metric]=float(np.mean(vals)) if vals else None
            if metric=="source_supervised_failure_auc": record[metric+"_supported_videos"]=len(vals)
        policies.append(record)
    lookup={(r["sequence"],r["arm"]):r for r in videos}; comparisons=[]
    for left,right in (("state_mix","native_replay"),("state_mix","native"),("native_replay","native"),("state_mix","source_supervised")):
        for metric in ("auc","low_iou_fraction","longest_failure_fraction","reliable_auc","source_supervised_failure_auc"):
            supported=[n for n in names if lookup[n,left][metric] is not None and lookup[n,right][metric] is not None]
            if not supported: continue
            draws=np.random.default_rng(c.DESIGN["bootstrap_seed"]).integers(0,len(supported),size=(c.DESIGN["bootstrap_samples"],len(supported)))
            comparisons.append(paired([lookup[n,left][metric]-lookup[n,right][metric] for n in supported],left+"_minus_"+right,metric,draws))
    find=lambda label,metric:next((r for r in comparisons if r["contrast"]==label and r["metric"]==metric),None)
    primary=find("state_mix_minus_native_replay","auc"); overall=find("state_mix_minus_native","auc")
    reliable=find("state_mix_minus_native","reliable_auc")
    selected=next(r for r in policies if r["arm"]=="state_mix"); base=next(r for r in policies if r["arm"]=="native")
    gains=[float(np.mean([next(r["auc"] for r in rows if r["sequence"]==n and r["arm"]=="state_mix" and r["seed"]==s)-
        next(r["auc"] for r in rows if r["sequence"]==n and r["arm"]=="native_replay" and r["seed"]==s) for n in names])) for s in c.SEEDS]
    guard=bool(reliable and reliable["mean_delta"]>=c.DESIGN["reliable_mean_floor"] and
        selected["introduced_failure_videos"]==0 and selected["low_iou_fraction"]<=base["low_iou_fraction"])
    passed=primary["mean_delta"]>=c.DESIGN["practical_auc_gain"] and primary["ci_low"]>0 and overall["mean_delta"]>=0 and min(gains)>0 and guard
    influence=[dict(sequence=n,auc_gain_vs_control=lookup[n,"state_mix"]["auc"]-lookup[n,"native_replay"]["auc"],
        leave_one_video_out=(primary["mean_delta"]*len(names)-lookup[n,"state_mix"]["auc"]+lookup[n,"native_replay"]["auc"])/(len(names)-1)) for n in names]
    with writer_lock(root/"analysis"):
        tables=dict(policy_summary=policies,video_summary=videos,seed_video_summary=rows,paired=comparisons,events=events,video_influence=influence)
        for name,values in tables.items():
            if values: publish_table(root/"analysis"/("stage51_"+name+".csv"),values)
        c.verify_files(bindings); c.verify_files(models); c.verify_files(plan["source_file_sha256"])
        report=c.seal(dict(schema=c.SCHEMA,status="PROMISING_FIT_STATE_SOURCE_PILOT" if passed else "REVIEW_FIT_STATE_SOURCE_PILOT",
            role="consumed_development_fixed_donor_state_source",plan_fingerprint=plan["fingerprint"],run_fingerprint=run["fingerprint"],
            primary=primary,overall=overall,retention_guard=guard,reliable_guard=reliable,primary_seed_gains=gains,
            videos=len(names),seeds=list(c.SEEDS),new_models=8,new_oof_trajectories=48,
            donor_trajectories=collection["donor_trajectories"],donor_head_calls=collection["donor_head_calls"],
            extra_feature_forwards=collection["extra_feature_forwards"],inference_head_calls=calls,inference_block_calls=blocks,
            source_file_sha256=bindings,model_sha256=models,
            table_sha256={"stage51_"+n+".csv":c.file_hash(root/"analysis"/("stage51_"+n+".csv")) for n,values in tables.items() if values},
            training_performed=True,held_out_examples_used_for_fitting=False,original_checking_labels_loaded=False,
            external_generalization_established=False,
            warnings=["Consumed development groups; seeds average inside videos before conditional bootstrap with overlapping fits.",
                "One fixed supervised donor per outer fold generates states only on its fit groups; no complementary-fold OOF donor states enter fitting.",
                "This is a fixed offline donor-state mixture, not iterative on-policy learning of each final model.",
                "All uniform frame draws are kept. Off-crop targets get ordinary negative heatmaps and no disjoint-box regression; no GT repair is used online.",
                "Both arms have two supervised views, identical original initialization, optimizer and updates. Actual positive/regression counts may differ with the states.",
                "Recovery/relapse events use each policy's own history; different event sets are descriptive rather than matched-start causal effects.",
                "Reliable and source-supervised failure cohorts are offline outcome-conditioned diagnostics, not actor inputs; the latter is the common union of both frozen source seeds' low-IoU frames.",
                "No architecture, candidate-selection, inference branch, compression or deployment speed change is tested."],
            next_step="Preserve fixed result; inspect exposure and full-video retention/recovery before extending the state curriculum"))
        c.publish_identical(root/"analysis/stage51_decision.json",report)
    for r in policies: print("{:<18} AUC={:.4f} low-IoU={:.2%}".format(r["arm"],r["auc"],r["low_iou_fraction"]))
    print("Stage-51: "+report["status"])
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--input",required=True,type=nonempty_path)
    summarize(p.parse_args().input)
