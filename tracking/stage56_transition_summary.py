"""First-transition preservation versus matched endpoint strength controls."""
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
from stage56_transition_experiment import model_records,smoke_gate
import stage56_transition_common as c

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
                    original=inputs["arrays"][entry["sequence"]][1]["rollout_preserve_seed"+str(seed)]
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
        print("Stage-56: PILOT_INCOMPLETE; {} missing trajectories".format(len(missing))); return None
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
    for left,right in (("transition_preserve","endpoint_control"),("transition_preserve","endpoint_half"),
                       ("endpoint_half","endpoint_control"),("transition_preserve","geometry_control"),
                       ("transition_preserve","static_anchor"))+tuple((arm,"native") for arm in c.ARMS):
        for metric in ("auc","low_iou_fraction","longest_failure_fraction","reliable_auc","control_failure_auc"):
            supported=[n for n in names if lookup[n,left][metric] is not None and lookup[n,right][metric] is not None]
            if not supported: continue
            vals=np.array([lookup[n,left][metric]-lookup[n,right][metric] for n in supported])
            draws=np.random.default_rng(plan["design"]["bootstrap_seed"]).integers(0,len(supported),size=(plan["design"]["bootstrap_samples"],len(supported)))
            record=paired(vals,left+"_minus_"+right,metric,draws)
            if metric=="auc" and left=="transition_preserve" and right in ("endpoint_control","endpoint_half"):
                record.update(family_ci_low=float(np.quantile(vals[draws].mean(1),.0125)),family_ci_high=float(np.quantile(vals[draws].mean(1),.9875)))
            else: record.update(family_ci_low=None,family_ci_high=None)
            comparisons.append(record)
    lookup={(r["sequence"],r["arm"]):r for r in videos}
    record=lambda left,right,metric:next(r for r in comparisons if r["contrast"]==left+"_minus_"+right and r["metric"]==metric)
    primary=record("transition_preserve","endpoint_control","auc")
    strength_control=record("transition_preserve","endpoint_half","auc")
    overall=record("transition_preserve","native","auc")
    retention=record("transition_preserve","native","reliable_auc")
    low=record("transition_preserve","native","low_iou_fraction")
    simple=record("transition_preserve","static_anchor","auc")
    seed_gains=[float(np.mean([next(r["auc"] for r in rows if (r["sequence"],r["arm"],r["seed"])==(n,"transition_preserve",s))-
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
        maximum_response_recompute_error=r["maximum_frozen_map_error"],maximum_decode_error=r["maximum_decode_error"],
        maximum_first_gradient=r["maximum_first_gradient"],maximum_endpoint_gradient=r["maximum_endpoint_gradient"],
        reproduces_stage54_control=r["reproduces_stage54_control"],supervised_views=r["counts"]["supervised"],
        **r["totals"]) for r,_ in records]
    support_pass=all(r["totals"]["changed_successor_crops"]>0 for r,_ in records)
    introduced=next(r["introduced_failure_videos"] for r in policies if r["arm"]=="transition_preserve")
    pilot_pass=(primary["mean_delta"]>=.5 and primary["family_ci_low"]>0 and strength_control["family_ci_low"]>0 and min(seed_gains)>0 and overall["mean_delta"]>=0 and
        retention["mean_delta"]>=-.5 and low["mean_delta"]<=0 and introduced==0 and support_pass)
    with writer_lock(root/"analysis"):
        tables=dict(policy_summary=policies,paired=comparisons,scope_summary=scope,retention_summary=retained,
            video_summary=videos,seed_video_summary=rows,focus=focus,events=events)
        for key,values in tables.items(): publish_table(root/"analysis"/("stage56_"+key+".csv"),values)
        c.verify_files(bindings); c.verify_files(models); c.verify_files(plan["source_file_sha256"])
        network_calls=sum(r["totals"]["reference_calls"]+r["totals"]["student_calls"] for r,_ in records)
        report=c.seal(dict(schema=c.SCHEMA,status="PROMISING_FIRST_TRANSITION_PRESERVATION_PILOT" if pilot_pass else "REVIEW_FIRST_TRANSITION_PRESERVATION_PILOT",
            role="consumed_development_first_transition_preservation",plan_fingerprint=plan["fingerprint"],run_fingerprint=run["fingerprint"],
            videos=len(names),seeds=list(c.SEEDS),new_models=12,new_full_trajectories=72,reused_full_trajectories=60,
            new_training_rollouts=True,new_offline_feature_collection=False,inference_head_calls=calls,inference_block_calls=calls*7,
            training_network_calls=network_calls,training_block_calls=network_calls*7,
            training_head_only_calls=sum(r["totals"]["head_only_batches"] for r,_ in records),
            source_file_sha256=bindings,model_sha256=models,pilot_pass=bool(pilot_pass),primary=primary,overall=overall,reliable_guard=retention,
            low_iou_effect=low,versus_static_anchor=simple,versus_half_endpoint=strength_control,primary_seed_gains=seed_gains,introduced_failure_videos=introduced,
            nontrivial_successor_crops=bool(support_pass),
            table_sha256={"stage56_"+k+".csv":c.file_hash(root/"analysis"/("stage56_"+k+".csv")) for k in tables},
            training_performed=True,held_out_examples_used_for_fitting=False,original_checking_labels_loaded=False,novel_method_established=False,
            warnings=["Consumed source groups informed design. Video bootstrap conditions on overlapping fixed fits; no external generalization or retraining certainty.",
                "Each optimizer step regenerates two-frame student branches using current geometry. Reference branches begin at the SAME recorded predicted parents.",
                "All three arms have identical frame/start draws, ordinary supervision counts, original initialization, updates and charged network/head calls; own normalized labels may differ with their crops.",
                "Native teacher quality across BOTH frames and endpoint center visibility qualify only fit loss. Actors see no future GT or hidden GT reset.",
                "First and endpoint losses directly update their own frame geometry; cross-frame parent/crop/argmax are detached, not full BPTT.",
                "endpoint_control must reproduce Stage-54 rollout_preserve weights and OOF boxes exactly; otherwise stop before interpreting gains.",
                "Fixed coefficients: endpoint_control (1,0), endpoint_half (0.5,0), transition_preserve (0.5,0.5). Half endpoint isolates weakening the old penalty.",
                "Primary and half-endpoint AUC comparisons each use a 97.5% family interval (Bonferroni two-contrast development screen).",
                "The anchor targets a different-history native endpoint; native geometry and teacher features are not assumed to share the student crop.",
                "Only original size/offset parameters update. Response/BN buffers stay frozen. Inference is the unchanged original tracker with one adapted head.",
                "Geometry reference and static anchor are context baselines with different training state schedules; the three NEW arms provide the matched comparisons.",
                "Source-positive cohorts and failure masks are offline diagnostics. No video-specific policy, compression or deployment speedup is established."],
            next_step="Inspect direct first-frame gradients, both matched controls, source-control parity and native capability retention; preserve this result without further horizon/weight/seed search"))
        c.publish_identical(root/"analysis/stage56_decision.json",report)
    for r in policies: print("{:<18} AUC={:.4f} low-IoU={:.2%}".format(r["arm"],r["auc"],r["low_iou_fraction"]))
    print("Stage-56: "+report["status"])
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--input",required=True,type=nonempty_path)
    summarize(parser.parse_args().input)
