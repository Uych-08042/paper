"""Matched native geometry preservation: accuracy, retention and fixed-source diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from stage27_uav_collect import writer_lock
from stage27_uav_common import nonempty_path,tracking_iou
from stage46_motion_summary import paired
from stage46_shift_audit import publish_table
from stage48_stability_experiment import audit_trace
from stage51_state_summary import describe
from stage53_geometry_experiment import model_records,smoke_gate
import stage53_geometry_common as c

METRICS=("auc","low_iou_fraction","failure_events","longest_failure_fraction","reliable_auc",
    "mean_abs_log_size_error","oversize3_fraction","unrecovered_events","recovered_events","relapse_events","control_failure_auc")


def summarize(root):
    root=Path(root).expanduser().resolve(); plan,inputs=c.load_plan(root); smoke_gate(plan,root)
    records,models=model_records(plan,root); run=c.read_sealed(root/"evaluation/run.json")
    if run.get("role")!="native_geometry_preservation_oof" or run.get("plan_fingerprint")!=plan["fingerprint"] or run.get("model_sha256")!=models:
        raise ValueError("Mixed branch evaluation")
    rows=[]; events=[]; focus=[]; bindings={}; missing=[]; calls=0
    for entry in plan["entries"]:
        native=c.native_trace(inputs,entry)
        joints={s:c.source_trace(inputs,entry,"joint",s) for s in c.SEEDS}
        controls={s:c.source_trace(inputs,entry,"geometry_only",s) for s in c.SEEDS}
        mask=np.any([tracking_iou(v["boxes"],np.asarray(entry["ground_truth"],float)[1:])<.2 for v in controls.values()],axis=0)
        values=[("native",0,native)]+[("joint",s,v) for s,v in joints.items()]+[("geometry_control",s,v) for s,v in controls.items()]
        for seed in c.SEEDS:
            for arm in c.ARMS:
                method=arm+"_seed"+str(seed); path=root/"evaluation/traces"/(entry["sequence"]+"__"+method+".npz")
                data=c.read_cache(path,dict(run_fingerprint=run["fingerprint"],sequence=entry["sequence"],method=method))
                if data is None: missing.append(path.name); continue
                audit_trace(data,entry); calls+=int(data["head_calls"].sum()); values.append((arm,seed,data))
                for p in (path,path.with_suffix(".receipt.json")): bindings[str(p)]=c.file_hash(p)
        for arm,seed,data in values:
            metrics,ev=describe(data,entry,native["boxes"],mask)
            metrics["control_failure_auc"]=metrics.pop("source_supervised_failure_auc")
            rows.append(dict(sequence=entry["sequence"],arm=arm,seed=seed,**metrics))
            events.extend(dict(sequence=entry["sequence"],arm=arm,seed=seed,**r) for r in ev)
        if entry["sequence"] in plan["design"]["focus"]:
            for arm in ("native","geometry_control","joint")+c.ARMS:
                chosen=sorted([r for r in rows if r["sequence"]==entry["sequence"] and r["arm"]==arm],key=lambda r:r["seed"])
                if not chosen: continue
                own_events=[r for r in events if r["sequence"]==entry["sequence"] and r["arm"]==arm]
                focus.append(dict(sequence=entry["sequence"],arm=arm,seeds="|".join(str(r["seed"]) for r in chosen),
                    auc_by_seed="|".join("{:.8f}".format(r["auc"]) for r in chosen),
                    first_failure_frames="|".join(str(r["first_failure_frame"]) for r in chosen),
                    introduced_failure_any_seed=any(r["introduced_failure"] for r in chosen),
                    terminal_unrecovered_by_seed="|".join(str(r["unrecovered_events"]) for r in chosen),
                    recoveries="|".join("{}:{}".format(r["seed"],r["recovery_frame"]) for r in own_events if r["recovery_frame"] is not None)))
    if missing:
        print("Stage-53: PILOT_INCOMPLETE; {} missing trajectories".format(len(missing))); return None
    names=[e["sequence"] for e in plan["entries"]]; arms=("native","geometry_control","joint")+c.ARMS
    videos=[]; policies=[]
    for arm in arms:
        for name in names:
            chosen=[r for r in rows if r["sequence"]==name and r["arm"]==arm]
            if len(chosen)!=(1 if arm=="native" else len(c.SEEDS)): raise ValueError("Missing branch seed/video")
            result=dict(sequence=name,arm=arm,introduced_failure=any(r["introduced_failure"] for r in chosen))
            for metric in METRICS:
                vals=[r[metric] for r in chosen]
                result[metric]=float(np.mean(vals)) if all(v is not None for v in vals) else None
            videos.append(result)
        chosen=[r for r in videos if r["arm"]==arm]
        result=dict(arm=arm,videos=len(names),provenance="new_branch_training" if arm in c.ARMS else "reused_frozen_source",
            introduced_failure_videos=sum(r["introduced_failure"] for r in chosen))
        for metric in METRICS:
            vals=[r[metric] for r in chosen if r[metric] is not None]
            result[metric]=float(np.mean(vals)) if vals else None
            if metric=="control_failure_auc": result[metric+"_supported_videos"]=len(vals)
        policies.append(result)
    lookup={(r["sequence"],r["arm"]):r for r in videos}; comparisons=[]
    for left,right in (("anchor_reliable","geometry_control"),("anchor_all","geometry_control"),("anchor_reliable","anchor_all"),("anchor_reliable","native"),("anchor_all","native")):
        for metric in ("auc","low_iou_fraction","longest_failure_fraction","reliable_auc","control_failure_auc"):
            supported=[n for n in names if lookup[n,left][metric] is not None and lookup[n,right][metric] is not None]
            if not supported: continue
            vals=np.array([lookup[n,left][metric]-lookup[n,right][metric] for n in supported])
            draws=np.random.default_rng(plan["design"]["bootstrap_seed"]).integers(0,len(supported),size=(plan["design"]["bootstrap_samples"],len(supported)))
            record=paired(vals,left+"_minus_"+right,metric,draws)
            if metric=="auc" and right=="native":
                record.update(family_ci_low=float(np.quantile(vals[draws].mean(1),.0125)),family_ci_high=float(np.quantile(vals[draws].mean(1),.9875)))
            else: record.update(family_ci_low=None,family_ci_high=None)
            comparisons.append(record)
    lookup={(r["sequence"],r["arm"]):r for r in videos}
    record=lambda left,right,metric:next(r for r in comparisons if r["contrast"]==left+"_minus_"+right and r["metric"]==metric)
    primary=record("anchor_reliable","geometry_control","auc")
    overall=record("anchor_reliable","native","auc")
    retention=record("anchor_reliable","native","reliable_auc")
    low=record("anchor_reliable","native","low_iou_fraction")
    simple=record("anchor_reliable","anchor_all","auc")
    seed_gains=[float(np.mean([next(r["auc"] for r in rows if (r["sequence"],r["arm"],r["seed"])==(n,"anchor_reliable",s))-
        next(r["auc"] for r in rows if (r["sequence"],r["arm"],r["seed"])==(n,"geometry_control",s)) for n in names])) for s in c.SEEDS]
    positive=[n for n in names if lookup[n,"geometry_control"]["auc"]>lookup[n,"native"]["auc"]]
    retained=[]
    for arm in arms:
        gains=[lookup[n,arm]["auc"]-lookup[n,"native"]["auc"] for n in positive]
        retained.append(dict(arm=arm,source_positive_videos=len(positive),positive_videos_remaining=sum(g>0 for g in gains),
            mean_gain_vs_native_on_source_positive=float(np.mean(gains)) if gains else None,
            mean_delta_vs_control_on_source_positive=float(np.mean([lookup[n,arm]["auc"]-lookup[n,"geometry_control"]["auc"] for n in positive])) if gains else None,
            worst_video_gain_vs_native=min(lookup[n,arm]["auc"]-lookup[n,"native"]["auc"] for n in names)))
    scope=[dict(fold=r["fold"],seed=r["seed"],arm=r["arm"],frozen_response_and_buffers=r["initial_frozen_sha256"]==r["final_frozen_sha256"],
        maximum_frozen_response_error=r["maximum_frozen_map_error"],maximum_anchor_gradient=r["maximum_anchor_gradient"],
        maximum_response_gradient=r["maximum_gradient"]["response"],trainable_parameters=r["trainable_parameters"],
        supervised_views=r["counts"]["supervised"],qualified_views_seen=r["anchor_counts"]["reliable"],
        mean_anchor_weight=r["anchor_counts"]["weight_mass"]/r["anchor_counts"]["views"],
        mean_anchor_penalty=r["anchor_counts"]["penalty_sum"]/r["anchor_counts"]["views"],
        matched_control_draws=r["draws_fingerprint"]==c.control_record(plan,r["fold"],r["seed"])["draws_fingerprint"]) for r,_ in records]
    introduced=next(r["introduced_failure_videos"] for r in policies if r["arm"]=="anchor_reliable")
    pilot_pass=(primary["mean_delta"]>=.5 and primary["ci_low"]>0 and min(seed_gains)>0 and overall["mean_delta"]>=0 and
        retention["mean_delta"]>=-.5 and low["mean_delta"]<=0 and introduced==0 and simple["mean_delta"]>=0)
    with writer_lock(root/"analysis"):
        tables=dict(policy_summary=policies,paired=comparisons,scope_summary=scope,retention_summary=retained,
            video_summary=videos,seed_video_summary=rows,focus=focus,events=events)
        for key,values in tables.items(): publish_table(root/"analysis"/("stage53_"+key+".csv"),values)
        c.verify_files(bindings); c.verify_files(models); c.verify_files(plan["source_file_sha256"])
        report=c.seal(dict(schema=c.SCHEMA,status="PROMISING_NATIVE_GEOMETRY_PRESERVATION_PILOT" if pilot_pass else "REVIEW_NATIVE_GEOMETRY_PRESERVATION_PILOT",
            role="consumed_development_native_geometry_preservation",plan_fingerprint=plan["fingerprint"],run_fingerprint=run["fingerprint"],
            videos=len(names),seeds=list(c.SEEDS),new_models=8,new_full_trajectories=48,reused_full_trajectories=60,
            new_state_generation=False,new_feature_collection=False,inference_head_calls=calls,inference_block_calls=calls*7,
            source_file_sha256=bindings,model_sha256=models,pilot_pass=bool(pilot_pass),primary=primary,overall=overall,reliable_guard=retention,
            low_iou_effect=low,selectivity_contrast=simple,selectivity_supported=simple["ci_low"]>0,primary_seed_gains=seed_gains,
            introduced_failure_videos=introduced,fit_support=plan["support"],
            table_sha256={"stage53_"+k+".csv":c.file_hash(root/"analysis"/("stage53_"+k+".csv")) for k in tables},
            training_performed=True,held_out_examples_used_for_fitting=False,original_checking_labels_loaded=False,novel_method_established=False,
            warnings=["Consumed groups informed design. Video bootstrap is conditional on overlapping fixed fits, not external confirmation or retraining uncertainty.",
                "Native cached Hann-cell geometry qualifies fit-only continuous IoU >=0.5. GT qualification is a training rule, never an online trigger.",
                "Reliable anchors are normalized by the entire fit-pool reliable fraction. Expected mass matches all-state anchoring; realized sampled mass and gradient norms can differ.",
                "Both new arms freeze response parameters and all BN buffers. The anchor has verified geometry gradients; no response-only constant loss is presented as a regularizer.",
                "Ordinary loss still regresses raw argmax boxes; the anchor preserves the frozen native Hann winner. One sampled-cell constraint cannot guarantee whole-trajectory retention.",
                "Original native heads initialize both arms; source states, draws, optimizer settings and update counts match the frozen geometry control.",
                "Source-positive videos, reliable frames and control-failure masks are offline diagnostics; neither video names nor future GT enter inference.",
                "Primary contrast is fixed to anchor_reliable minus geometry_control. Additional family intervals cover the two new-versus-native AUC contrasts only; other intervals are nominal.",
                "This is a native-head preservation hypothesis, not compression, deployment speedup or established method novelty."],
            next_step="Preserve the fixed result; inspect matched anchor exposure, majority-gain retention and the five cases before any next design"))
        c.publish_identical(root/"analysis/stage53_decision.json",report)
    for r in policies: print("{:<18} AUC={:.4f} low-IoU={:.2%}".format(r["arm"],r["auc"],r["low_iou_fraction"]))
    print("Stage-53: "+report["status"])
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--input",required=True,type=nonempty_path)
    summarize(parser.parse_args().input)
