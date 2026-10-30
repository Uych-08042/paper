"""Full-cohort endpoint gradient projection with a norm-matched control."""
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
from stage58_gradient_experiment import model_records,smoke_gate
import stage58_gradient_common as c

METRICS=("auc","low_iou_fraction","failure_events","longest_failure_fraction","reliable_auc",
    "mean_abs_log_size_error","oversize3_fraction","unrecovered_events","recovered_events","relapse_events","control_failure_auc")


def summarize(root):
    root=Path(root).expanduser().resolve(); plan,inputs=c.load_plan(root); smoke_gate(plan,root)
    records,models=model_records(plan,root); run=c.read_sealed(root/"evaluation/run.json")
    if run.get("role")!="native_short_rollout_oof" or run.get("plan_fingerprint")!=plan["fingerprint"] or run.get("model_sha256")!=models:
        raise ValueError("Mixed branch evaluation")
    rows=[]; events=[]; focus=[]; bindings={}; missing=[]; calls=0
    for entry in plan["entries"]:
        native=c.native_trace(inputs,entry)
        joints={s:c.source_trace(inputs,entry,"anchor_reliable",s) for s in c.SEEDS}
        controls={s:c.source_trace(inputs,entry,"geometry_only",s) for s in c.SEEDS}
        mask=np.any([tracking_iou(v["boxes"],np.asarray(entry["ground_truth"],float)[1:])<.2 for v in controls.values()],axis=0)
        values=[("native",0,native)]+[("static_anchor",s,v) for s,v in joints.items()]+[("geometry_control",s,v) for s,v in controls.items()]
        for seed in c.SEEDS:
            for arm in c.ARMS:
                method=arm+"_seed"+str(seed); path=root/"evaluation/traces"/(entry["sequence"]+"__"+method+".npz")
                data=c.read_cache(path,dict(run_fingerprint=run["fingerprint"],sequence=entry["sequence"],method=method))
                if data is None: missing.append(path.name); continue
                audit_trace(data,entry)
                if arm=="endpoint_control":
                    original=inputs["arrays"][entry["sequence"]][1]["endpoint_control_seed"+str(seed)]
                    for key in ("boxes","previous","peak","layer"):
                        np.testing.assert_array_equal(data[key],original[key],err_msg="Stage-54 control replay changed")
                calls+=int(data["head_calls"].sum()); values.append((arm,seed,data))
                for p in (path,path.with_suffix(".receipt.json")): bindings[str(p)]=c.file_hash(p)
        for arm,seed,data in values:
            metrics,ev=describe(data,entry,native["boxes"],mask)
            metrics["control_failure_auc"]=metrics.pop("source_supervised_failure_auc")
            rows.append(dict(sequence=entry["sequence"],arm=arm,seed=seed,**metrics))
            events.extend(dict(sequence=entry["sequence"],arm=arm,seed=seed,**r) for r in ev)
        if entry["sequence"] in plan["design"]["focus"]:
            for arm in ("native","geometry_control","static_anchor")+c.ARMS:
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
        print("Stage-58: PILOT_INCOMPLETE; {} missing trajectories".format(len(missing))); return None
    names=[e["sequence"] for e in plan["entries"]]; arms=("native","geometry_control","static_anchor")+c.ARMS
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
    for left,right in (("projected_anchor","endpoint_control"),("projected_anchor","norm_matched"),
                       ("norm_matched","endpoint_control"),("projected_anchor","geometry_control"),
                       ("projected_anchor","static_anchor"))+tuple((arm,"native") for arm in c.ARMS):
        for metric in ("auc","low_iou_fraction","longest_failure_fraction","reliable_auc","control_failure_auc"):
            supported=[n for n in names if lookup[n,left][metric] is not None and lookup[n,right][metric] is not None]
            if not supported: continue
            vals=np.array([lookup[n,left][metric]-lookup[n,right][metric] for n in supported])
            draws=np.random.default_rng(plan["design"]["bootstrap_seed"]).integers(0,len(supported),size=(plan["design"]["bootstrap_samples"],len(supported)))
            record=paired(vals,left+"_minus_"+right,metric,draws)
            if metric=="auc" and left=="projected_anchor" and right in ("endpoint_control","norm_matched"):
                record.update(family_ci_low=float(np.quantile(vals[draws].mean(1),.0125)),family_ci_high=float(np.quantile(vals[draws].mean(1),.9875)))
            else: record.update(family_ci_low=None,family_ci_high=None)
            comparisons.append(record)
    lookup={(r["sequence"],r["arm"]):r for r in videos}
    record=lambda left,right,metric:next(r for r in comparisons if r["contrast"]==left+"_minus_"+right and r["metric"]==metric)
    primary=record("projected_anchor","endpoint_control","auc")
    strength_control=record("projected_anchor","norm_matched","auc")
    overall=record("projected_anchor","native","auc")
    retention=record("projected_anchor","native","reliable_auc")
    low=record("projected_anchor","native","low_iou_fraction")
    simple=record("projected_anchor","static_anchor","auc")
    seed_gains=[float(np.mean([next(r["auc"] for r in rows if (r["sequence"],r["arm"],r["seed"])==(n,"projected_anchor",s))-
        next(r["auc"] for r in rows if (r["sequence"],r["arm"],r["seed"])==(n,"endpoint_control",s)) for n in names])) for s in c.SEEDS]
    positive=[n for n in names if lookup[n,"geometry_control"]["auc"]>lookup[n,"native"]["auc"]]
    retained=[]
    for arm in arms:
        gains=[lookup[n,arm]["auc"]-lookup[n,"native"]["auc"] for n in positive]
        retained.append(dict(arm=arm,source_positive_videos=len(positive),positive_videos_remaining=sum(g>0 for g in gains),
            mean_gain_vs_native_on_source_positive=float(np.mean(gains)) if gains else None,
            mean_delta_vs_control_on_source_positive=float(np.mean([lookup[n,arm]["auc"]-lookup[n,"geometry_control"]["auc"] for n in positive])) if gains else None,
            worst_video_gain_vs_native=min(lookup[n,arm]["auc"]-lookup[n,"native"]["auc"] for n in names)))
    scope=[dict(fold=r["fold"],seed=r["seed"],arm=r["arm"],frozen_response_and_buffers=r["initial_frozen_sha256"]==r["final_frozen_sha256"],
        trainable_parameters=r["trainable_parameters"],maximum_response_gradient=r["maximum_gradient"]["response"],
        maximum_response_recompute_error=r["maximum_frozen_map_error"],maximum_decode_error=r["maximum_decode_error"],
        maximum_endpoint_gradient=r["maximum_endpoint_gradient"],
        reproduces_stage57_control=r["reproduces_stage57_control"],supervised_views=r["counts"]["supervised"],
        **r["totals"],**r["gradient_totals"],**r["quality_totals"],**r["maximum_gradient_audit"]) for r,_ in records]
    support_pass=all(r["totals"]["changed_successor_crops"]>0 for r,_ in records)
    required=max(1,int(np.ceil(plan["training"]["updates"]*plan["design"]["minimum_conflict_fraction"])))
    gradient_support=all(r["gradient_totals"]["nontrivial_projection_batches"]>=required and
        r["gradient_totals"]["actual_modified_batches"]>0 for r,_ in records if r["arm"]!="endpoint_control")
    introduced=next(r["introduced_failure_videos"] for r in policies if r["arm"]=="projected_anchor")
    pilot_pass=(primary["mean_delta"]>=.5 and primary["family_ci_low"]>0 and strength_control["family_ci_low"]>0 and min(seed_gains)>0 and overall["mean_delta"]>=0 and
        retention["mean_delta"]>=-.5 and low["mean_delta"]<=0 and introduced==0 and support_pass and gradient_support)
    with writer_lock(root/"analysis"):
        tables=dict(policy_summary=policies,paired=comparisons,scope_summary=scope,retention_summary=retained,
            video_summary=videos,seed_video_summary=rows,focus=focus,events=events)
        for key,values in tables.items(): publish_table(root/"analysis"/("stage58_"+key+".csv"),values)
        c.verify_files(bindings); c.verify_files(models); c.verify_files(plan["source_file_sha256"])
        network_calls=sum(r["totals"]["reference_calls"]+r["totals"]["student_calls"] for r,_ in records)
        report=c.seal(dict(schema=c.SCHEMA,status="PROMISING_PROJECTED_ENDPOINT_GRADIENT_PILOT" if pilot_pass else "REVIEW_PROJECTED_ENDPOINT_GRADIENT_PILOT",
            role="consumed_development_projected_endpoint_gradient",plan_fingerprint=plan["fingerprint"],run_fingerprint=run["fingerprint"],
            videos=len(names),seeds=list(c.SEEDS),new_models=12,new_full_trajectories=72,reused_full_trajectories=60,
            new_training_rollouts=True,new_offline_feature_collection=False,inference_head_calls=calls,inference_block_calls=calls*7,
            training_network_calls=network_calls,training_block_calls=network_calls*7,
            training_head_only_calls=sum(r["totals"]["head_only_batches"] for r,_ in records),
            source_file_sha256=bindings,model_sha256=models,pilot_pass=bool(pilot_pass),primary=primary,overall=overall,reliable_guard=retention,
            low_iou_effect=low,versus_static_anchor=simple,versus_norm_matched=strength_control,primary_seed_gains=seed_gains,introduced_failure_videos=introduced,
            nontrivial_successor_crops=bool(support_pass),gradient_support=bool(gradient_support),
            table_sha256={"stage58_"+k+".csv":c.file_hash(root/"analysis"/("stage58_"+k+".csv")) for k in tables},
            training_performed=True,held_out_examples_used_for_fitting=False,original_checking_labels_loaded=False,novel_method_established=False,
            warnings=["All twelve groups are consumed development evidence; video bootstrap conditions on fixed overlapping fits, not external confirmation.",
                "Every original eligible anchor is retained; fit IoU comparisons are diagnostic only, never selection rules or online gates.",
                "gS is ordinary supervision and gP is the aggregate full endpoint penalty over the original geometry parameters. Only gP is projected.",
                "The norm control matches projected anchor length on ITS OWN batch while retaining original direction. Different own states do not imply globally matched gradients.",
                "All arms compute gS/gP/shadow projection and execute the full scalar backward, with identical forward/start/update budgets. Backward algebra and wall time are not equal FLOPs guarantees.",
                "Projection is Euclidean parameter-gradient space before clipping and AdamW. Momentum, adaptive preconditioning and weight decay can reverse the actual displacement alignment; this is audited, not guaranteed away.",
                "Zero gradients and parallel opposing vectors are valid. Conflict/projection exposure is reported even when pilot_support is false, without changing threshold or seed.",
                "Only original size/offset parameters update; response/BN, backbone, candidates, selector and inference remain frozen. No first loss or BPTT.",
                "Full endpoint control must exactly reproduce Stage-57 control weights and OOF boxes/parents/peaks/layers.",
                "Two planned AUC contrasts each have a 97.5% family interval; diagnostic failure cohorts never define online decisions.",
                "Gradient projection is established prior work. This tests its added value for SGLA own-state endpoint preservation, not absolute novelty or a proven causal explanation of older losses.",
                "No deployment speedup, compression gain or independent generalization is established."],
            next_step="Inspect projection versus full and norm-matched controls, gradient/AdamW alignment, retained anchors and fixed recovery cases; preserve without weight/seed/checkpoint search"))
        c.publish_identical(root/"analysis/stage58_decision.json",report)
    for r in policies: print("{:<18} AUC={:.4f} low-IoU={:.2%}".format(r["arm"],r["auc"],r["low_iou_fraction"]))
    print("Stage-58: "+report["status"])
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--input",required=True,type=nonempty_path)
    summarize(parser.parse_args().input)
