"""Native full-trajectory results for fit-only absolute-q0 takeover protection."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import stage66_takeover_common as c
import stage66_takeover_experiment as e
from stage27_uav_common import tracking_iou, failure_events, nonempty_path
from stage51_state_summary import describe
from stage61_l40_summary import contrast, METRICS
from stage46_shift_audit import publish_table
from stage27_uav_collect import writer_lock

USAGE=("selected_nonq0_fraction","quality_nonq0_fraction","veto_fraction","qualified_fraction","mean_q0_absolute_error","qualified_q0_good_precision")


def summarize(root):
    root=Path(root).resolve();plan,source=c.load_plan(root);models=e.load_models(plan,root,source);run=e.completed(plan,root,source,models)
    if e.build_models(plan,source)!=models:raise ValueError("Critics or fit-only acceptance calibration do not reconstruct")
    rows=[];scope=[];focus=[];events=[];hashes={}
    for u in plan["units"]:
        entry=next(r for r in plan["entries"] if r["sequence"]==u["sequence"])
        data=c.read_cache(e.path_for(root,u),c.metadata(plan,u,"evaluation",models))
        if data is None or len(data["frame_id"])!=len(entry["frames"])-1:raise ValueError("Missing full trajectory")
        e.audit(data,entry,u,plan,models);e.check_source(data,source,u)
        c.c64.e63.source_experiment.bind(hashes,e.path_for(root,u))
        ref,_,_,head=c.c64.prior.prior.read_trace(source["s64"]["old"]["l40"],c.e64.source_unit(source["s64"]["old"],u,"current"))
        if head!=run["observed_heads"][c.key(u)]:raise ValueError("Frozen head identity differs")
        gt=np.asarray(entry["ground_truth"],float)[1:];native_iou=tracking_iou(ref["boxes"],gt)
        metrics,ev=describe(data,entry,ref["boxes"],native_iou<.2);metrics["native_failure_auc"]=metrics.pop("source_supervised_failure_auc")
        q0_iou=tracking_iou(data["candidate_boxes"][:,0],gt);qualified=data["q0_qualified"].astype(bool)
        rows.append(dict(**u,**metrics,selected_nonq0_fraction=float((data["selected"]!=0).mean()),
            quality_nonq0_fraction=float((data["quality_choice"]!=0).mean()),veto_fraction=float(data["veto"].mean()),
            qualified_fraction=float(qualified.mean()),mean_q0_absolute_error=float(np.abs(data["q0_pred_iou"]-q0_iou).mean()),
            qualified_q0_good_precision=float((q0_iou[qualified]>=.5).mean()) if qualified.any() else None))
        own=failure_events(tracking_iou(data["boxes"],gt))
        focus.append(dict(sequence=u["sequence"],policy=u["policy"],auc=metrics["auc"],
            first_failure=None if not own else own[0]["start"]+1,first_recovery=None if not own or own[0]["recovery"] is None else own[0]["recovery"]+1,
            terminal_unrecovered=metrics["unrecovered_events"],introduced_failure=metrics["introduced_failure"]))
        events.extend(dict(sequence=u["sequence"],policy=u["policy"],**v) for v in ev)
        scope.append(dict(**u,head_sha256=head,tracked_frames=len(data["frame_id"]),head_calls=int(data["head_calls"].sum()),
            block_calls=int(data["block_calls"].sum()),maximum_decode_error=float(data["decode_error"].max()),sgla_training_performed=False))
    if hashes!=run["cache_sha256"]:raise ValueError("Evaluation bindings differ")
    policies=[];metrics=METRICS+USAGE
    for policy in c.POLICIES:
        selected=[r for r in rows if r["policy"]==policy]
        if len(selected)!=len(plan["entries"]):raise ValueError("Missing policy/video")
        p=dict(policy=policy,videos=len(selected),introduced_failure_videos=sum(r["introduced_failure"] for r in selected))
        for k in metrics:
            valid=[r[k] for r in selected if r[k] is not None];p[k]=float(np.mean(valid)) if valid else None
            if k=="qualified_q0_good_precision":p[k+"_supported_videos"]=len(valid)
        policies.append(p)
    names=[r["sequence"] for r in plan["entries"]];lookup={(r["sequence"],r["policy"]):r for r in rows};paired=[]
    for right in ("quality","margin","hann","q0"):
        for metric in ("auc","low_iou_fraction","reliable_auc","longest_failure_fraction"):
            valid=[n for n in names if all(lookup[n,p][metric] is not None for p in ("protect",right))]
            if not valid:continue
            values=np.array([lookup[n,"protect"][metric]-lookup[n,right][metric] for n in valid])
            r=contrast(values,"protect_minus_"+right,metric,plan["design"])
            if metric=="auc":
                draws=np.random.default_rng(plan["design"]["bootstrap_seed"]).integers(0,len(values),size=(plan["design"]["bootstrap_samples"],len(values)))
                r["family_ci_low"],r["family_ci_high"]=map(float,np.quantile(values[draws].mean(1),[.05/8,1-.05/8]))
            r["supported_videos"]="|".join(valid);paired.append(r)
    get=lambda p,m:next(r for r in paired if r["contrast"]=="protect_minus_"+p and r["metric"]==m)
    protect=next(p for p in policies if p["policy"]=="protect")
    guard=protect["introduced_failure_videos"]==0 and get("q0","low_iou_fraction")["mean_delta"]<=0 and get("q0","reliable_auc")["ci_low"]>=plan["design"]["reliable_floor"]
    support=sum(lookup[n,"protect"]["veto_fraction"]>0 for n in names)>=min(4,len(names)) and sum(lookup[n,"protect"]["selected_nonq0_fraction"]>0 for n in names)>=min(4,len(names))
    passes=all(get(p,"auc")["family_ci_low"]>0 for p in ("quality","margin","hann","q0")) and min(get(p,"auc")["mean_delta"] for p in ("quality","hann"))>=plan["design"]["practical_gain"] and guard and support
    calibration=[dict(fold=int(fold),fit_videos=len(p["fit"]),states=p["critic"]["states"],**p["margin"]) for fold,p in models["models"].items()]
    influence=[dict(sequence=n,**{p+"_gain":lookup[n,"protect"]["auc"]-lookup[n,p]["auc"] for p in ("quality","margin","hann","q0")}) for n in names]
    tables=dict(policy_summary=policies,paired=paired,calibration=calibration,scope=scope,focus=focus,
                video_summary=rows,events=events,video_influence=influence,fit_support=models["support"])
    out=root/"analysis";out.mkdir(exist_ok=True)
    with writer_lock(out):
        c.VERIFY(plan["source_file_sha256"]);c.VERIFY(plan["code_sha256"])
        for name,values in tables.items():publish_table(out/("stage66_"+name+".csv"),values)
        report=c.seal(dict(schema=c.SCHEMA,status="PROMISING_NATIVE_Q0_TAKEOVER_PILOT" if passes else "REVIEW_NATIVE_Q0_TAKEOVER_PILOT",
            role=plan["role"],plan_fingerprint=plan["fingerprint"],run_fingerprint=run["fingerprint"],training_fingerprint=models["fingerprint"],
            primary=get("quality","auc"),rate_control=get("margin","auc"),against_hann=get("hann","auc"),against_q0=get("q0","auc"),
            retention_guard=guard,support_pass=support,pilot_pass=passes,full_trajectories=len(plan["units"]),formal_search_calls=plan["formal_search_calls"],
            formal_block_calls=7*plan["formal_search_calls"],smoke_calls_reused=plan["smoke_calls_reused"],additional_smoke_calls=0,
            new_critics=models["new_critics"],source_quality_unchanged=True,sgla_training_performed=False,critic_fitting_performed=True,
            fit_origins=["native"],runtime=plan["runtime_identity"],declared_runtime_differences=plan["declared_runtime_differences"],
            maximum_source_box_error=0.,original_checking_labels_loaded=False,external_generalization_established=False,novel_method_established=False,
            evidence_sha256=run["cache_sha256"],table_sha256={"stage66_"+k+".csv":c.file_hash(out/("stage66_"+k+".csv")) for k in tables},
            warnings=["Consumed twelve-group development cohort; conditional video bootstrap is not independent external or training validation.",
                "Native V1 only; E transfer and weak-candidate geometry repair are not evaluated in this stage.",
                "Absolute q0 IoU is a clipped ridge prediction, not a calibrated reliability probability or guarantee.",
                "Fixed good threshold .5 and protected gain .02; no outcome-driven threshold or checkpoint search.",
                "Margin-only control matches weighted fit acceptance as closely as tied gains allow; realized own-history evaluation rates can differ.",
                "All five policies actually run four searches per frame, including q0; no speedup claim.",
                "Hann, q0 and quality controls reproduce frozen source outputs; only physical L40 UUID may differ explicitly.",
                "GT enters fit/evaluation only; online choices use current four candidates and own accepted history."],
            next_step="Preserve this fixed result; compare protection to frozen quality, fit-rate margin control, Hann and q0 before transfer or new design"))
        c.publish_identical(out/"stage66_decision.json",report)
    for r in policies:print("{:<12} AUC={:.4f} low-IoU={:.2%}".format(r["policy"],r["auc"],r["low_iou_fraction"]))
    print("Stage-66: "+report["status"]);return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",required=True,type=nonempty_path);summarize(p.parse_args().input)
