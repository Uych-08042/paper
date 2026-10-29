"""Frozen-state branch training factorial: accuracy, retention and trajectory descriptions."""
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
from stage52_branch_experiment import model_records,smoke_gate
import stage52_branch_common as c

METRICS=("auc","low_iou_fraction","failure_events","longest_failure_fraction","reliable_auc",
    "mean_abs_log_size_error","oversize3_fraction","unrecovered_events","recovered_events","relapse_events","joint_failure_auc")


def summarize(root):
    root=Path(root).expanduser().resolve(); plan,inputs=c.load_plan(root); smoke_gate(plan,root)
    records,models=model_records(plan,root); run=c.read_sealed(root/"evaluation/run.json")
    if run.get("role")!="native_branch_adaptation_oof" or run.get("plan_fingerprint")!=plan["fingerprint"] or run.get("model_sha256")!=models:
        raise ValueError("Mixed branch evaluation")
    rows=[]; events=[]; focus=[]; bindings={}; missing=[]; calls=0
    for entry in plan["entries"]:
        native=c.prior.prior.source_native(inputs["source48"],entry)
        joints={s:c.source_trace(Path(plan["source_input"]),inputs["run"],entry,"state_mix",s)[0] for s in c.SEEDS}
        mask=np.any([tracking_iou(v["boxes"],np.asarray(entry["ground_truth"],float)[1:])<.2 for v in joints.values()],axis=0)
        values=[("native",0,native)]+[("joint",s,v) for s,v in joints.items()]
        for seed in c.SEEDS:
            for arm in c.ARMS:
                method=arm+"_seed"+str(seed); path=root/"evaluation/traces"/(entry["sequence"]+"__"+method+".npz")
                data=c.read_cache(path,dict(run_fingerprint=run["fingerprint"],sequence=entry["sequence"],method=method))
                if data is None: missing.append(path.name); continue
                audit_trace(data,entry); calls+=int(data["head_calls"].sum()); values.append((arm,seed,data))
                for p in (path,path.with_suffix(".receipt.json")): bindings[str(p)]=c.file_hash(p)
        for arm,seed,data in values:
            metrics,ev=describe(data,entry,native["boxes"],mask)
            metrics["joint_failure_auc"]=metrics.pop("source_supervised_failure_auc")
            rows.append(dict(sequence=entry["sequence"],arm=arm,seed=seed,**metrics))
            events.extend(dict(sequence=entry["sequence"],arm=arm,seed=seed,**r) for r in ev)
        if entry["sequence"] in plan["design"]["focus"]:
            for arm in ("native",)+c.ARMS+("joint",):
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
        print("Stage-52: PILOT_INCOMPLETE; {} missing trajectories".format(len(missing))); return None
    names=[e["sequence"] for e in plan["entries"]]; arms=("native",)+c.ARMS+("joint",)
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
            if metric=="joint_failure_auc": result[metric+"_supported_videos"]=len(vals)
        policies.append(result)
    lookup={(r["sequence"],r["arm"]):r for r in videos}; comparisons=[]
    for left,right in (("response_only","native"),("geometry_only","native"),("response_only","joint"),("geometry_only","joint"),("response_only","geometry_only")):
        for metric in ("auc","low_iou_fraction","longest_failure_fraction","reliable_auc","joint_failure_auc"):
            supported=[n for n in names if lookup[n,left][metric] is not None and lookup[n,right][metric] is not None]
            if not supported: continue
            vals=np.array([lookup[n,left][metric]-lookup[n,right][metric] for n in supported])
            draws=np.random.default_rng(plan["design"]["bootstrap_seed"]).integers(0,len(supported),size=(plan["design"]["bootstrap_samples"],len(supported)))
            record=paired(vals,left+"_minus_"+right,metric,draws)
            if metric=="auc" and right=="native":
                record.update(family_ci_low=float(np.quantile(vals[draws].mean(1),.0125)),family_ci_high=float(np.quantile(vals[draws].mean(1),.9875)))
            else: record.update(family_ci_low=None,family_ci_high=None)
            comparisons.append(record)
    draws=np.random.default_rng(plan["design"]["bootstrap_seed"]).integers(0,len(names),size=(plan["design"]["bootstrap_samples"],len(names)))
    formulas=(("response_order_average",lambda x:.5*(x["response_only"]-x["native"]+x["joint"]-x["geometry_only"])),
        ("geometry_order_average",lambda x:.5*(x["geometry_only"]-x["native"]+x["joint"]-x["response_only"])),
        ("response_by_geometry_training_interaction",lambda x:x["joint"]-x["response_only"]-x["geometry_only"]+x["native"]))
    effects=[paired([f({a:lookup[n,a]["auc"] for a in arms}) for n in names],label,"auc",draws) for label,f in formulas]
    scope=[dict(fold=r["fold"],seed=r["seed"],arm=r["arm"],trainable_parameters=r["trainable_parameters"],
        head_parameters=r["head_parameters"],frozen_state_unchanged=r["initial_frozen_sha256"]==r["final_frozen_sha256"],
        maximum_frozen_map_error=r["maximum_frozen_map_error"],maximum_response_gradient=r["maximum_gradient"]["response"],
        maximum_geometry_gradient=r["maximum_gradient"]["geometry"],supervised_views=r["counts"]["supervised"],
        matched_joint_draws=r["draws_fingerprint"]==c.joint_record(plan,r["fold"],r["seed"])["draws_fingerprint"]) for r,_ in records]
    with writer_lock(root/"analysis"):
        tables=dict(policy_summary=policies,paired=comparisons,factor_summary=effects,scope_summary=scope,
            video_summary=videos,seed_video_summary=rows,focus=focus,events=events)
        for key,values in tables.items(): publish_table(root/"analysis"/("stage52_"+key+".csv"),values)
        c.verify_files(bindings); c.verify_files(models); c.verify_files(plan["source_file_sha256"])
        report=c.seal(dict(schema=c.SCHEMA,status="COMPLETE_FIXED_STATE_BRANCH_DIAGNOSTIC",
            role="consumed_development_branch_update_factorial",plan_fingerprint=plan["fingerprint"],run_fingerprint=run["fingerprint"],
            videos=len(names),seeds=list(c.SEEDS),new_models=8,new_full_trajectories=48,reused_joint_trajectories=24,
            new_state_generation=False,new_feature_collection=False,inference_head_calls=calls,inference_block_calls=calls*7,
            source_file_sha256=bindings,model_sha256=models,
            table_sha256={"stage52_"+k+".csv":c.file_hash(root/"analysis"/("stage52_"+k+".csv")) for k in tables},
            training_performed=True,held_out_examples_used_for_fitting=False,original_checking_labels_loaded=False,adoption_gate=None,
            warnings=["Consumed source groups and prior outcomes informed design; nominal video bootstrap is conditional on overlapping fixed fits.",
                "Original heads initialize all trained arms. Frozen state, labels, draws, optimizer settings and update counts match source joint training.",
                "Trainable parameter counts and gradient clipping norms differ by intervention; equal steps are not equal training FLOPs.",
                "Hard raw argmax does not carry box-selection gradients. Response-only updates come from focal; geometry-only focal is constant.",
                "Frozen maps are tested at shared features. Own states, peak selection and box geometry can still differ along trajectories.",
                "Factorial effects describe whole trained policies, not a unique causal decomposition of a trajectory or a posthoc inference head swap.",
                "Two branch-versus-native AUC comparisons also report Bonferroni percentile family intervals; other intervals remain nominal exploratory summaries.",
                "Joint-failure frames and own recovery events are offline descriptions; no controller sees future GT, and no new method adoption or speed claim is made."],
            next_step="Inspect branch scope, full-video retention and the fixed five cases before choosing any next method"))
        c.publish_identical(root/"analysis/stage52_diagnostic.json",report)
    for r in policies: print("{:<16} AUC={:.4f} low-IoU={:.2%}".format(r["arm"],r["auc"],r["low_iou_fraction"]))
    print("Stage-52: COMPLETE_FIXED_STATE_BRANCH_DIAGNOSTIC")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--input",required=True,type=nonempty_path)
    summarize(parser.parse_args().input)
