"""Full-trajectory four-candidate PPO against equal-capacity CE and frozen source controls."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import stage74_ppo_common as c
import stage74_ppo_experiment as e
import stage72_ppo_common as aggregation
from stage27_uav_common import tracking_iou,failure_events,nonempty_path
from stage27_uav_collect import writer_lock
from stage51_state_summary import describe
from stage61_l40_summary import METRICS,contrast
from stage68_advantage_summary import publish_table

PRIMARY=("myopic","quality","binary_ppo","future")
COMPARISONS=PRIMARY+("binary_myopic","q0","hann")
USAGE=("selected_nonq0_fraction","changed_box_fraction","outside_binary_fraction","distinct_candidates_mean")+tuple("action_q"+str(q)+"_fraction" for q in range(4))


def training_tables(plan,models):
    rows=[];scope=[];support=[]
    for job in plan["jobs"]:
        episodes=[x for x in models["episodes"] if x["fold"]==job["fold"] and x["seed"]==job["seed"]]
        for item in episodes:
            metrics=[k[4:] for k in item if k.startswith("ppo_")]
            for arm in ("ppo","myopic"):
                rows.append(dict(fold=job["fold"],seed=job["seed"],arm=arm,episode=item["episode"],sequence=item["sequence"],pass_index=item["pass_index"],
                    **{k:item[arm+"_"+k] for k in metrics}))
        h=plan["design"]["hidden"];actor=256*h+h+4*h+4;critic=256*h+h+h+1
        scope.append(dict(fold=job["fold"],seed=job["seed"],episodes=len(episodes),frames=sum(x["frames"] for x in episodes),search_calls=sum(x["search_calls"] for x in episodes),
            block_calls=7*sum(x["search_calls"] for x in episodes),actor_parameters=actor,critic_parameters=critic,parameters_per_arm=actor+critic,
            matched_initialization=True,matched_state_exposure=True,matched_minibatches=True,sgla_trainable_parameters=0,
            ppo_updates=sum(x["ppo_updates"] for x in episodes),myopic_updates=sum(x["myopic_updates"] for x in episodes),
            maximum_ppo_actor_gradient=max(x["ppo_maximum_actor_gradient"] for x in episodes),maximum_myopic_actor_gradient=max(x["myopic_maximum_actor_gradient"] for x in episodes)))
        for seq in models["controllers"][c.job_key(job)]["fit"]:
            group=[x for x in episodes if x["sequence"]==seq]
            counts={k:sum(x[k] for x in group) for k in ("frames","eligible","accepted")+tuple("action_q"+str(q) for q in range(4))+tuple("label_q"+str(q) for q in range(4))}
            support.append(dict(fold=job["fold"],seed=job["seed"],sequence=seq,episodes=len(group),fit_only=True,**counts))
    identity=("fold","seed","arm","pass_index");metrics=[k for k in rows[0] if k not in identity+("episode","sequence")]
    passes=aggregation.aggregate(rows,identity,metrics)
    optimization=aggregation.aggregate(rows,("fold","seed","arm"),metrics)
    return dict(training_episode=rows,training_summary=passes,optimization_summary=optimization,training_scope=scope,fit_support=support,training_log=models["episodes"])


def summarize(root):
    root=Path(root).resolve();plan,source=c.load_plan(root);models=e.load_models(root,plan,source);run=e.completed(root,plan,source,models);bridge=e.bridge_gate(root,plan)
    seeds=[];focus=[];scope=[];events=[];candidates=[];hashes={}
    candidate_metrics=("selected_mean_iou","any_public_good_fraction","any_binary_good_fraction","selected_good_fraction","missed_public_good_fraction","public_current_regret","outside_binary_good_fraction","q0_good_to_low_fraction")
    for unit in plan["units"]:
        entry=c.entry_for(plan,unit["sequence"]);data=c.read_cache(c.eval_path(root,unit),e.eval_meta(plan,unit,models))
        if data is None or len(data["frame_id"])!=len(entry["frames"])-1:raise ValueError("Missing full own-history trajectory")
        pack=None if unit["policy"] in c.CONTROLS else e.learned_pack(models,unit,source)
        e.audit(data,entry,unit,plan,source,pack);e.e63.source_experiment.bind(hashes,c.eval_path(root,unit))
        native=c.source_trace(source,dict(unit,policy="q0",seed=0));gt=np.asarray(entry["ground_truth"],float)[1:]
        metrics,ev=describe(data,entry,native["boxes"],tracking_iou(native["boxes"],gt)<.2)
        metrics["native_failure_auc"]=metrics.pop("source_supervised_failure_auc")
        n=len(data["frame_id"]);idx=np.arange(n);selected=data["selected"].astype(int);q=data["quality_choice"].astype(int)
        outside=np.any(data["boxes"]!=data["candidate_boxes"][:,0],axis=1)&np.any(data["boxes"]!=data["candidate_boxes"][idx,q],axis=1)
        usage=dict(selected_nonq0_fraction=float((selected!=0).mean()),changed_box_fraction=float(np.any(data["boxes"]!=data["candidate_boxes"][:,0],axis=1).mean()),
            outside_binary_fraction=float(outside.mean()),distinct_candidates_mean=float(np.mean([c.action_mask(b).sum() for b in data["candidate_boxes"]])),
            **{"action_q"+str(j)+"_fraction":float((selected==j).mean()) for j in range(4)})
        seeds.append(dict(**unit,**metrics,**usage))
        y=tracking_iou(data["candidate_boxes"],gt[:,None]);chosen=y[idx,selected];public=y.max(1)>=.5;binary=np.maximum(y[:,0],y[idx,q])>=.5
        candidates.append(dict(**unit,selected_mean_iou=float(chosen.mean()),any_public_good_fraction=float(public.mean()),any_binary_good_fraction=float(binary.mean()),
            selected_good_fraction=float((chosen>=.5).mean()),missed_public_good_fraction=float((public&(chosen<.5)).mean()),public_current_regret=float((y.max(1)-chosen).mean()),
            outside_binary_good_fraction=float((outside&(chosen>=.5)).mean()),q0_good_to_low_fraction=float(((y[:,0]>=.5)&(chosen<.2)).mean())))
        own=failure_events(chosen)
        focus.append(dict(sequence=unit["sequence"],policy=unit["policy"],seed=unit["seed"],auc=metrics["auc"],first_failure=None if not own else own[0]["start"]+1,
            first_recovery=None if not own or own[0]["recovery"] is None else own[0]["recovery"]+1,terminal_unrecovered=metrics["unrecovered_events"],introduced_failure=metrics["introduced_failure"]))
        events.extend(dict(sequence=unit["sequence"],policy=unit["policy"],seed=unit["seed"],**x) for x in ev)
        scope.append(dict(**unit,frames=n,search_calls=int(data["head_calls"].sum()),block_calls=int(data["block_calls"].sum()),head_sha256=run["observed_heads"][c.key(unit)],
            maximum_decode_error=float(data["decode_error"].max()),exact_old_common_fields=unit["policy"] in c.CONTROLS+c.REUSED,sgla_training_performed=False))
    if hashes!=run["cache_sha256"]:raise ValueError("Evaluation evidence changed")
    names=[x["sequence"] for x in plan["entries"]];policies=c.CONTROLS+c.REUSED+c.LEARNED;videos=[];candidate_video=[]
    for seq in names:
        for p in policies:
            group=[x for x in seeds if x["sequence"]==seq and x["policy"]==p]
            if len(group)!=(1 if p in c.CONTROLS else len(plan["design"]["seeds"])):raise ValueError("Incomplete policy/seed scope")
            row=dict(sequence=seq,policy=p,introduced_failure=any(x["introduced_failure"] for x in group))
            for k in METRICS+USAGE:row[k]=aggregation.mean([x[k] for x in group if x[k] is not None])
            videos.append(row)
            group=[x for x in candidates if x["sequence"]==seq and x["policy"]==p]
            candidate_video.append(dict(sequence=seq,policy=p,**{k:aggregation.mean([x[k] for x in group]) for k in candidate_metrics}))
    macros=[];candidate_summary=[]
    for p in policies:
        group=[x for x in videos if x["policy"]==p]
        macros.append(dict(policy=p,videos=len(group),seeds=1 if p in c.CONTROLS else len(plan["design"]["seeds"]),introduced_failure_videos=sum(x["introduced_failure"] for x in group),
            **{k:aggregation.mean([x[k] for x in group if x[k] is not None]) for k in METRICS+USAGE}))
        group=[x for x in candidate_video if x["policy"]==p]
        candidate_summary.append(dict(policy=p,videos=len(group),**{k:aggregation.mean([x[k] for x in group]) for k in candidate_metrics}))
    lookup={(x["sequence"],x["policy"]):x for x in videos};paired=[];effects=[]
    for other in COMPARISONS:
        for metric in ("auc","low_iou_fraction","reliable_auc","longest_failure_fraction"):
            valid=[n for n in names if lookup[n,"ppo"][metric] is not None and lookup[n,other][metric] is not None]
            if not valid:continue
            diff=np.array([lookup[n,"ppo"][metric]-lookup[n,other][metric] for n in valid]);row=contrast(diff,"ppo_minus_"+other,metric,plan["design"])
            if metric=="auc" and other in PRIMARY:
                draws=np.random.default_rng(plan["design"]["bootstrap_seed"]).integers(0,len(diff),(plan["design"]["bootstrap_samples"],len(diff)))
                row["family_ci_low"],row["family_ci_high"]=map(float,np.quantile(diff[draws].mean(1),[.05/(2*len(PRIMARY)),1-.05/(2*len(PRIMARY))]))
            row["supported_videos"]="|".join(valid);paired.append(row)
    for seed in plan["design"]["seeds"]:
        row=dict(seed=seed)
        for other in COMPARISONS:
            a=[next(x["auc"] for x in seeds if x["sequence"]==n and x["policy"]=="ppo" and x["seed"]==seed) for n in names]
            b=[next(x["auc"] for x in seeds if x["sequence"]==n and x["policy"]==other and x["seed"]==(0 if other in c.CONTROLS else seed)) for n in names]
            row["auc_gain_vs_"+other]=float(np.mean(np.asarray(a)-b))
        effects.append(row)
    get=lambda other,metric:next(x for x in paired if x["contrast"]=="ppo_minus_"+other and x["metric"]==metric)
    ppo=next(x for x in macros if x["policy"]=="ppo")
    guard=ppo["introduced_failure_videos"]==0 and get("q0","low_iou_fraction")["mean_delta"]<=0 and get("q0","reliable_auc")["ci_low"]>=plan["design"]["reliable_floor"]
    support=sum(lookup[n,"ppo"]["changed_box_fraction"]>0 for n in names)>=min(4,len(names)) and sum(lookup[n,"ppo"]["action_q0_fraction"]>0 for n in names)>=min(4,len(names))
    seed_pass=all(all(x["auc_gain_vs_"+p]>0 for p in ("myopic","quality","binary_ppo")) for x in effects)
    passed=bool(guard and support and seed_pass and all(get(p,"auc")["family_ci_low"]>0 for p in PRIMARY) and min(get(p,"auc")["mean_delta"] for p in ("myopic","quality"))>=plan["design"]["practical_gain"])
    influence=[];values=np.array([lookup[n,"ppo"]["auc"]-lookup[n,"myopic"]["auc"] for n in names])
    for n in names:
        row=dict(sequence=n,**{p+"_auc_gain":lookup[n,"ppo"]["auc"]-lookup[n,p]["auc"] for p in COMPARISONS})
        row["primary_leave_one_out"]=float((values.sum()-row["myopic_auc_gain"])/(len(names)-1)) if len(names)>1 else None;influence.append(row)
    training=training_tables(plan,models)
    tables=dict(policy_summary=macros,paired=paired,seed_effects=effects,focus=focus,scope=scope,video_influence=influence,
        candidate_summary=candidate_summary,candidate_video=candidate_video,seed_video_summary=seeds,video_summary=videos,events=events,**training)
    out=root/"analysis";out.mkdir(exist_ok=True)
    with writer_lock(out):
        c.VERIFY(plan["source_file_sha256"]);c.VERIFY(plan["code_sha256"])
        for n,rows in tables.items():publish_table(out/("stage74_"+n+".csv"),rows)
        total=plan["training_search_calls"]+plan["formal_eval_calls"]+plan["source_bridge_search_calls"]
        report=c.seal(dict(schema=c.SCHEMA,status="PROMISING_DIRECT_FOUR_CANDIDATE_PPO_PILOT" if passed else "REVIEW_DIRECT_FOUR_CANDIDATE_PPO_PILOT",role=plan["role"],
            plan_fingerprint=plan["fingerprint"],training_fingerprint=models["fingerprint"],run_fingerprint=run["fingerprint"],source_report_fingerprint=source["stage73"]["report"]["fingerprint"],
            source_training_fingerprint=source["stage73"]["models"]["fingerprint"],source_bindings_fingerprint=c.fingerprint(plan["source_file_sha256"]),runtime=plan["runtime_identity"],declared_runtime_differences=plan["declared_runtime_differences"],
            primary=get("myopic","auc"),against_quality=get("quality","auc"),against_binary_ppo=get("binary_ppo","auc"),against_future=get("future","auc"),
            retention_guard=bool(guard),support_pass=bool(support),seed_pass=bool(seed_pass),pilot_pass=passed,videos=len(names),seeds=plan["design"]["seeds"],
            actions=[0,1,2,3],recovery_branch_present=False,new_recovery_search_calls=0,new_controller_models=2*len(plan["jobs"]),reused_binary_controller_models=2*len(plan["jobs"]),
            fit_episodes=sum(len(j["episodes"]) for j in plan["jobs"]),fit_environment_steps=plan["training_environment_steps"],fit_search_calls=plan["training_search_calls"],optimizer_updates_per_arm=plan["optimizer_updates_per_arm"],
            full_evaluation_trajectories=len(plan["units"]),source_evaluation_trajectories_retested=sum(u["policy"] in c.CONTROLS+c.REUSED for u in plan["units"]),new_policy_evaluation_trajectories=sum(u["policy"] in c.LEARNED for u in plan["units"]),
            eval_search_calls=plan["formal_eval_calls"],source_bridge_search_calls=plan["source_bridge_search_calls"],source_bridge_fingerprint=bridge["fingerprint"],total_new_search_calls=total,total_new_block_calls=7*total,
            train_smoke_calls_reused=plan["train_smoke_calls_reused"],eval_smoke_calls_reused=plan["eval_smoke_calls_reused"],additional_smoke_calls=plan["source_bridge_search_calls"],
            sgla_training_performed=False,reinforcement_learning_performed=True,source_models_unchanged=True,maximum_source_common_field_error=0.,
            training_evidence_fingerprint=c.fingerprint(models["training_evidence_sha256"]),eval_evidence_fingerprint=c.fingerprint(run["cache_sha256"]),original_checking_labels_loaded=False,external_generalization_established=False,novel_method_established=False,
            table_sha256={"stage74_"+n+".csv":c.file_hash(out/("stage74_"+n+".csv")) for n in tables},
            warnings=["Twelve consumed development groups; average seeds within videos before conditional bootstrap. No independent confirmation.",
                "All policies use the same frozen SGLA V1 four-search generator. No new recovery, template update, future observation, GT controller input or failure reset.",
                "The PPO and CE arms share architecture, initialization, normalization, exact PPO behavior observations/labels/returns and minibatch updates; CE does not collect a separate fit history.",
                "The old binary comparison changes available observations, input/output dimensions and capacity as well as action scope; it is not an isolated action-count ablation.",
                "Only exactly equal public boxes are aliased to the first index; a q0 representative is always legal. Masks use no GT and cannot discard an unequal low-quality candidate.",
                "Uniform initial exploration is over distinct public boxes; with four distinct boxes q0 has probability 1/4, unlike the old binary initialization.",
                "The CE label is the highest current IoU among legal public boxes with an epsilon tie going to the lowest index. Labels enter only after actual action selection.",
                "Outside-binary use and oracle current candidate support are offline descriptions, not counterfactual long-term rewards.",
                "Training pre/post probabilities and value errors use each preceding behavior episode, not a new-policy return estimate or independent value validation.",
                "Only the chosen source-policy subset is re-evaluated. Exact reproduction refers to original common search/output fields, not new-format categorical placeholders.",
                "Fixed final checkpoint, two seeds and source update budget. No reward, threshold, seed or checkpoint search and no deployment speedup claim."],
            next_step="Preserve both seeds and the fixed budget; assess equal-capacity supervised and source baselines plus retention before changing the design."))
        c.publish_identical(out/"stage74_decision.json",report)
    for x in macros:print("{:<16} AUC={:.4f} low-IoU={:.2%}".format(x["policy"],x["auc"],x["low_iou_fraction"]))
    print("Stage-74: "+report["status"]);return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--input",required=True,type=nonempty_path);summarize(parser.parse_args().input)
