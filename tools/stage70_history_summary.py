"""Matched training-state source ablation before a bounded search-policy RL pilot."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import numpy as np
import stage70_history_common as c
import stage70_history_experiment as e
import stage69_gate_common as c69
import stage69_gate_audit as a69
from stage68_advantage_summary import publish_table
from stage27_uav_common import tracking_iou, failure_events, nonempty_path
from stage27_uav_collect import writer_lock
from stage51_state_summary import describe
from stage61_l40_summary import contrast, METRICS
from stage4_common import file_hash, fingerprint

USAGE=("selected_nonq0_fraction","changed_box_fraction","quality_nonq0_fraction","veto_fraction")
PRIMARY=(("mixed_future","future"),("mixed_future","mixed_immediate"),("mixed_future","mixed_rate_margin"),("mixed_future","quality"))


def fit_diagnostics(plan,root,source,pp,models):
    stats=[];refs={};calibration=[]
    for split in plan["folds"]:
        fold=split["fold"]
        for arm in ("baseline","mixed"):
            if arm=="baseline":x,y,w,_=e.e68.examples(source["s68"]["plan"],source["s68"]["root"],source["s68"]["old"],fold)
            else:x,y,w,_=e.examples(plan,root,source,pp,fold)
            pack=models[arm][str(fold)];refs[arm,fold]=c69.reference(x,w,pack["future"])
            for j,target in enumerate(c.c68.TARGETS):
                pred=np.asarray([c.c68.predict(v,pack[target]) for v in x])
                stats.append(dict(fold=fold,training_source=arm,target=target,role="in_sample_fit_check",
                    **c69.prediction_stats(pred,y[:,j],w,plan["design"]["tie_epsilon"])))
            calibration.append(dict(fold=fold,training_source=arm,**pack["rate_margin"]))
    probes=[]
    for arm in ("baseline","mixed"):
        shadow=copy.copy(source["s68"]);shadow["models"]=dict(shadow["models"],models=models[arm])
        cases,_=a69.paired_transfer(shadow,{f["fold"]:refs[arm,f["fold"]] for f in plan["folds"]})
        probes.extend(dict(training_source=arm,**row) for row in cases)
    return stats,calibration,probes


def summarize(root):
    root=Path(root).resolve();plan,source=c.load_plan(root);models=e.load_models(plan,root,source)
    bridge=e.gate(plan,root,"bridge");pp=e.pair_plan(plan,root,source);pairrun=e.pairs_completed(plan,root,source,pp);historyrun=e.history_completed(plan,root,source);run=e.completed(plan,root,source,models)
    if models!=e.build_models(plan,root,source):raise ValueError("Mixed models, source labels or rate calibration do not reconstruct")
    stats,calibration,probes=fit_diagnostics(plan,root,source,pp,models)
    rows=[];focus=[];events=[];scope=[];hashes={};natives={}
    for unit in plan["units"]:
        entry=e.entry_for(plan,unit);seq=unit["sequence"]
        data=c.read_cache(c.path_for(root,unit,"evaluation"),c.metadata(plan,unit,"evaluation",models))
        if data is None or len(data["frame_id"])!=len(entry["frames"])-1:raise ValueError("Missing complete held-out policy")
        e.audit_eval(data,entry,unit,plan,models);e.check_source(data,source,unit)
        e.e63.source_experiment.bind(hashes,c.path_for(root,unit,"evaluation"))
        if seq not in natives:natives[seq]=c.c68.prior.read_unit(source["s68"]["old"]["s66"],seq,"q0")[0]
        native=natives[seq];gt=np.asarray(entry["ground_truth"])[1:]
        metrics,ev=describe(data,entry,native["boxes"],tracking_iou(native["boxes"],gt)<.2)
        metrics["native_failure_auc"]=metrics.pop("source_supervised_failure_auc")
        rows.append(dict(**unit,**metrics,selected_nonq0_fraction=float((data["selected"]!=0).mean()),
            changed_box_fraction=float(np.any(data["boxes"]!=data["candidate_boxes"][:,0],axis=1).mean()),
            quality_nonq0_fraction=float((data["quality_choice"]!=0).mean()),veto_fraction=float(data["veto"].mean())))
        own=failure_events(tracking_iou(data["boxes"],gt))
        focus.append(dict(sequence=seq,policy=unit["policy"],auc=metrics["auc"],first_failure=None if not own else own[0]["start"]+1,
            first_recovery=None if not own or own[0]["recovery"] is None else own[0]["recovery"]+1,
            terminal_unrecovered=metrics["unrecovered_events"],introduced_failure=metrics["introduced_failure"]))
        events.extend(dict(sequence=seq,policy=unit["policy"],**v) for v in ev)
        scope.append(dict(**unit,frames=len(data["frame_id"]),search_calls=int(data["head_calls"].sum()),
            block_calls=int(data["block_calls"].sum()),head_sha256=run["observed_heads"][c.c66.key(unit)],
            maximum_decode_error=float(data["decode_error"].max()),exact_source_control=not unit["policy"].startswith("mixed_")))
    if hashes!=run["cache_sha256"]:raise ValueError("Evaluation evidence differs")
    policies=[]
    for policy in c.POLICIES:
        group=[r for r in rows if r["policy"]==policy]
        if len(group)!=len(plan["entries"]):raise ValueError("Missing video/policy")
        out=dict(policy=policy,videos=len(group),introduced_failure_videos=sum(r["introduced_failure"] for r in group))
        for k in METRICS+USAGE:
            values=[r[k] for r in group if r[k] is not None];out[k]=float(np.mean(values)) if values else None
        policies.append(out)
    names=[e["sequence"] for e in plan["entries"]];lookup={(r["sequence"],r["policy"]):r for r in rows};paired=[]
    comparisons=PRIMARY+(("mixed_immediate","immediate"),("mixed_future","q0"),("mixed_future","hann"))
    for new,old in comparisons:
        for metric in ("auc","low_iou_fraction","reliable_auc","longest_failure_fraction"):
            valid=[n for n in names if all(lookup[n,p][metric] is not None for p in (new,old))]
            if not valid:continue
            values=np.asarray([lookup[n,new][metric]-lookup[n,old][metric] for n in valid])
            row=contrast(values,new+"_minus_"+old,metric,plan["design"])
            if metric=="auc" and (new,old) in PRIMARY:
                draws=np.random.default_rng(plan["design"]["bootstrap_seed"]).integers(0,len(values),size=(plan["design"]["bootstrap_samples"],len(values)))
                row["family_ci_low"],row["family_ci_high"]=map(float,np.quantile(values[draws].mean(1),[.05/8,1-.05/8]))
            row["supported_videos"]="|".join(valid);paired.append(row)
    get=lambda old,metric:next(r for r in paired if r["contrast"]=="mixed_future_minus_"+old and r["metric"]==metric)
    factor=[]
    diffs=np.asarray([[lookup[n,p]["auc"] for p in ("immediate","future","mixed_immediate","mixed_future")] for n in names])
    for label,value in (("history_source_order_average",.5*((diffs[:,2]-diffs[:,0])+(diffs[:,3]-diffs[:,1]))),
        ("future_order_average",.5*((diffs[:,1]-diffs[:,0])+(diffs[:,3]-diffs[:,2]))),
        ("future_by_history_source_interaction",(diffs[:,3]-diffs[:,2])-(diffs[:,1]-diffs[:,0]))):
        factor.append(contrast(value,label,"auc",plan["design"]))
    history_rows=[]
    for unit in plan["history_units"]:
        trace=e.read_history(plan,root,source,unit);entry=e.entry_for(plan,unit)
        hann,_,_=c.c68.prior.read_unit(source["s68"]["old"]["s66"],unit["sequence"],"hann")
        y=tracking_iou(trace["boxes"],np.asarray(entry["ground_truth"])[1:]);best=tracking_iou(trace["candidate_boxes"],np.asarray(entry["ground_truth"])[1:,None]).max(1)
        history_rows.append(dict(**unit,role="fit_only_offline_description",frames=len(y),mean_iou=float(y.mean()),
            low_iou_fraction=float((y<.2).mean()),failure_events=len(failure_events(y)),any_good_fraction=float((best>=.5).mean()),
            changed_boxes_vs_hann_fraction=float(np.any(trace["boxes"]!=hann["boxes"],axis=1).mean()),
            changed_candidate_crops_vs_hann_fraction=float(np.any(trace["crop"]!=hann["crop"],axis=(1,2)).mean()),
            search_calls=int(trace["head_calls"].sum()),block_calls=int(trace["block_calls"].sum())))
    if sum(r["search_calls"] for r in history_rows)!=plan["history_search_calls"]:raise ValueError("History scope differs")
    pair_scope=[]
    for case in pp["cases"]:
        shared,tails=e.read_pair(root,plan,source,pp,case)
        for action,data in (("shared",shared),*tails.items()):
            pair_scope.append(dict(case_id=case["case_id"],fold=case["fold"],sequence=case["sequence"],provider=case["provider"],
                reused=case["reused"],action=action,frames=len(data["frame_id"]),search_calls=int(data["head_calls"].sum()),block_calls=int(data["block_calls"].sum())))
    if sum(r["search_calls"] for r in pair_scope if not r["reused"])!=plan["pair_search_calls"]:raise ValueError("New pair scope differs")
    guard=(next(r for r in policies if r["policy"]=="mixed_future")["introduced_failure_videos"]==0 and
           get("q0","low_iou_fraction")["mean_delta"]<=0 and get("q0","reliable_auc")["ci_low"]>=plan["design"]["reliable_floor"])
    source_gain=get("future","auc")["family_ci_low"]>0
    passes=(guard and all(get(old,"auc")["family_ci_low"]>0 for _,old in PRIMARY) and
            min(get(old,"auc")["mean_delta"] for old in ("future","quality"))>=plan["design"]["practical_gain"])
    influence=[dict(sequence=n,mixed_future_minus_future=lookup[n,"mixed_future"]["auc"]-lookup[n,"future"]["auc"],
        mixed_immediate_minus_immediate=lookup[n,"mixed_immediate"]["auc"]-lookup[n,"immediate"]["auc"],
        mixed_future_minus_quality=lookup[n,"mixed_future"]["auc"]-lookup[n,"quality"]["auc"],
        mixed_future_minus_rate=lookup[n,"mixed_future"]["auc"]-lookup[n,"mixed_rate_margin"]["auc"]) for n in names]
    tables=dict(policy_summary=policies,paired=paired,factor_summary=factor,calibration=calibration,fit_summary=stats,
        pair_support=pp["support"],history_support=history_rows,paired_probe=probes,focus=focus,scope=scope,video_influence=influence,
        fit_pairs=models["support"],pair_scope=pair_scope,video_summary=rows,events=events)
    out=root/"analysis";out.mkdir(exist_ok=True)
    with writer_lock(out):
        c.VERIFY(plan["source_file_sha256"]);c.VERIFY(plan["code_sha256"])
        for name,values in tables.items():publish_table(out/("stage70_"+name+".csv"),values)
        report=c.seal(dict(schema=c.SCHEMA,status="PROMISING_FIT_HISTORY_SOURCE_PILOT" if passes else "REVIEW_FIT_HISTORY_SOURCE_PILOT",
            role=plan["role"],plan_fingerprint=plan["fingerprint"],training_fingerprint=models["fingerprint"],
            history_run_fingerprint=historyrun["fingerprint"],pair_plan_fingerprint=pp["fingerprint"],pair_run_fingerprint=pairrun["fingerprint"],run_fingerprint=run["fingerprint"],
            source_report_fingerprint=source["report"]["fingerprint"],source_training_fingerprint=source["s68"]["models"]["fingerprint"],
            source_bindings_fingerprint=fingerprint(plan["source_file_sha256"]),runtime=plan["runtime_identity"],declared_runtime_differences=plan["declared_runtime_differences"],
            primary=get("future","auc"),history_source_gain_supported=source_gain,retention_guard=guard,pilot_pass=passes,
            videos=len(names),fit_histories=len(plan["history_units"]),new_predictors=models["new_predictors"],
            mixed_fit_pairs=len(pp["cases"]),baseline_fit_pairs=len(source["s68"]["plan"]["pairs"]),
            new_paired_starts=plan["expected_new_pairs"],reused_paired_starts=plan["expected_reused_pairs"],
            history_search_calls=plan["history_search_calls"],pair_search_calls=plan["pair_search_calls"],eval_search_calls=plan["formal_eval_calls"],
            source_bridge_search_calls=plan["source_bridge_search_calls"],source_bridge_run_fingerprint=bridge["fingerprint"],
            total_new_search_calls=sum(budget for budget in (plan["history_search_calls"],plan["pair_search_calls"],plan["formal_eval_calls"],plan["source_bridge_search_calls"])),
            total_new_block_calls=7*(plan["history_search_calls"]+plan["pair_search_calls"]+plan["formal_eval_calls"]+plan["source_bridge_search_calls"]),
            history_smoke_calls_reused=plan["history_smoke_calls_reused"],pair_smoke_calls_reused=plan["pair_smoke_calls_reused"],eval_smoke_calls_reused=plan["eval_smoke_calls_reused"],additional_smoke_calls=plan["source_bridge_search_calls"],
            full_evaluation_trajectories=len(plan["units"]),new_pair_branches=2*plan["expected_new_pairs"],
            sgla_training_performed=False,reinforcement_learning_performed=False,source_quality_unchanged=True,maximum_source_control_error=0.,
            original_checking_labels_loaded=False,external_generalization_established=False,novel_method_established=False,
            table_sha256={"stage70_"+n+".csv":file_hash(out/("stage70_"+n+".csv")) for n in tables},
            warnings=["Repeated twelve-source grouped development OOF; not independent external confirmation.",
                "Only the paired training-history source changes: fixed per-video count, 128 features, ridge, target horizon and zero-gain rule.",
                "Fit Q/protect histories and Q tails use the current fold's controllers, never the stored opposite-fold auxiliary decisions.",
                "Half of mixed pairs are reused Hann starts; one shared source fit start per fold is reproduced on the current GPU before collection.",
                "History/provider quotas and uniform eligible-time selection are fixed without GT or event triggers.",
                "Mixed and baseline rate controls calibrate on the same all-fit Hann state population; deployed intervention rates can differ.",
                "All ten full evaluation policies are rerun on the bound L40; seven original policies must reproduce their source controls.",
                "Fit/support and seven consumed Stage67 probes are descriptive diagnostics; probe outcomes never select samples or parameters.",
                "The paired target follows frozen Q after one forced action; the deployed gate's later actions can differ.",
                "All evaluation policies pay four searches per frame. New fit-history generation costs are separately charged; no speed or energy claim.",
                "Pilot success is not a prerequisite for any RL attempt; this bounded ablation informs the next independently predeclared PPO comparison."],
            next_step="Preserve this single coverage ablation; assess action sensitivity and predeclare a bounded PPO comparison if warranted, without threshold or sample search."))
        c.publish_identical(out/"stage70_decision.json",report)
    for row in policies:print("{:<19} AUC={:.4f} low-IoU={:.2%}".format(row["policy"],row["auc"],row["low_iou_fraction"]))
    print("Stage-70: "+report["status"]);return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--input",required=True,type=nonempty_path);summarize(parser.parse_args().input)
