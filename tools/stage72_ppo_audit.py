"""Audit Stage-71 selection and optimization using committed arrays and JSON only."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import stage72_ppo_common as c
from stage27_uav_common import nonempty_path
from stage27_uav_collect import writer_lock
from stage68_advantage_summary import publish_table


def macro(rows,keys,metrics):
    result=c.aggregate(rows,keys,metrics)
    for row in result:row["videos"]=row.pop("records")
    return result


def first_difference(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape:raise ValueError("Unaligned stored trajectories")
    changed=np.any(a!=b,axis=tuple(range(1,a.ndim))) if a.ndim>1 else a!=b
    indices=np.flatnonzero(changed);return int(indices[0])+1 if len(indices) else None


def divergence(sequence,seed,a,b):
    n=len(a["frame_id"]);np.testing.assert_array_equal(a["frame_id"],b["frame_id"]);idx=np.arange(n)
    same_candidates=np.all(a["candidate_boxes"]==b["candidate_boxes"],axis=(1,2))
    different_choice=a["selected"]!=b["selected"]
    return dict(sequence=sequence,seed=seed,comparison="own_PPO_vs_own_myopic",first_box_difference=first_difference(a["boxes"],b["boxes"]),
        first_candidate_difference=first_difference(a["candidate_boxes"],b["candidate_boxes"]),
        first_parent_difference=first_difference(a["candidate_parents"],b["candidate_parents"]),
        first_feature_difference=first_difference(a["choice_features"],b["choice_features"]),
        first_selected_crop_difference=first_difference(a["crop"][idx,a["selected"]],b["crop"][idx,b["selected"]]),
        same_candidate_frames=int(same_candidates.sum()),same_candidates_choice_diff_frames=int((same_candidates & different_choice).sum()))


def focus_row(anchor,unit,seed,data,scored,detail,metrics):
    frame=anchor["frame"];first=max(1,frame-c.DESIGN["before"]);last=min(len(data["frame_id"]),frame+c.DESIGN["after"])
    mask=(data["frame_id"]>=first)&(data["frame_id"]<=last);i=frame-1
    current=detail["iou"][np.arange(len(mask)),data["selected"]]
    row=dict(case_id=anchor["case_id"],sequence=unit["sequence"],provider=unit["policy"],provider_seed=unit["seed"],scorer_seed=seed,
        frame=frame,first=first,last=last,frames=int(mask.sum()),own_mean_iou=c.mean(current,mask),own_anchor_iou=float(current[i]),
        public_good_frames=int(((detail["best"]>=c.DESIGN["good_iou"])&mask).sum()),
        binary_good_frames=int(((detail["binary_best"]>=c.DESIGN["good_iou"])&mask).sum()),
        all_public_low_frames=int(((detail["best"]<c.DESIGN["low_iou"])&mask).sum()),
        ppo_myopic_disagreement_frames=int(((scored["choices"]["ppo"]!=scored["choices"]["myopic"])&mask).sum()),
        ppo_minus_myopic_current_mean_iou=c.mean(metrics["ppo"]["selected_iou"]-metrics["myopic"]["selected_iou"],mask),
        feature_clipped_coordinate_fraction=c.mean(np.mean(scored["feature_clipped"],axis=1),mask))
    for rule in c.RULES:
        row.update({rule+"_mean_iou":c.mean(metrics[rule]["selected_iou"],mask),rule+"_anchor_choice":int(scored["choices"][rule][i]),
            rule+"_anchor_iou":float(metrics[rule]["selected_iou"][i]),
            rule+"_missed_binary_good_frames":int((metrics[rule]["missed_binary_good"]&mask).sum())})
    for rule in ("ppo","myopic"):
        row[rule+"_anchor_logit"]=float(scored[rule]["logit"][i]);row[rule+"_anchor_probability"]=float(scored[rule]["probability"][i])
        row[rule+"_q0_good_to_low_frames"]=int((metrics[rule]["q0_good_to_low"]&mask).sum())
    return row,mask


def evaluation(source,anchors):
    plan=source["plan"];rows=[];matched=[];candidates=[];focus=[];frames=[];divergences=[];scope=[]
    for entry in plan["entries"]:
        seq=entry["sequence"];traces={}
        for unit in (u for u in plan["units"] if u["sequence"]==seq):
            data,_=c.read_unit(source,unit);traces[unit["policy"],unit["seed"]]=data;n=len(data["frame_id"])
            overlaps=c.tracking_iou(data["candidate_boxes"],np.asarray(entry["ground_truth"])[1:,None]);idx=np.arange(n);q=data["quality_choice"].astype(int)
            candidates.append(dict(sequence=seq,provider=unit["policy"],provider_seed=unit["seed"],
                frames=n,any_public_good_fraction=c.mean(overlaps.max(1)>=c.DESIGN["good_iou"]),
                any_binary_good_fraction=c.mean(np.maximum(overlaps[:,0],overlaps[idx,q])>=c.DESIGN["good_iou"]),
                all_public_low_fraction=c.mean(overlaps.max(1)<c.DESIGN["low_iou"]),
                q0_good_fraction=c.mean(overlaps[:,0]>=c.DESIGN["good_iou"]),mean_best_public_iou=c.mean(overlaps.max(1))))
            score_seeds=c.score_seeds(unit,plan)
            for seed in score_seeds:
                pack=source["models"]["controllers"][c.c71.job_key(dict(fold=unit["fold"],seed=seed))]
                if seq not in pack["held_out"] or seq in pack["fit"]:raise ValueError("Held-out scorer crosses fit source group")
                scored=c.score(data,pack,plan["design"]);metrics,detail=c.label_states(data,entry,scored)
                if unit["policy"] in c.RULES:
                    np.testing.assert_array_equal(scored["choices"][unit["policy"]],data["selected"])
                    if unit["policy"] in ("ppo","myopic"):
                        for key in ("logit","value","probability"):np.testing.assert_allclose(scored[unit["policy"]][key],data[key],rtol=0,atol=1e-10)
                identity=dict(sequence=seq,fold=unit["fold"],provider=unit["policy"],provider_seed=unit["seed"],scorer_seed=seed)
                for rule,values in metrics.items():rows.append(dict(**identity,shadow_rule=rule,frames=n,**{k:c.mean(v) for k,v in values.items()}))
                for other in ("myopic","quality"):
                    qa,qb=scored["choices"]["ppo"],scored["choices"][other]
                    matched.append(dict(**identity,contrast="ppo_minus_"+other,current_mean_iou_delta=c.mean(metrics["ppo"]["selected_iou"]-metrics[other]["selected_iou"]),
                        choice_disagreement_fraction=c.mean(qa!=qb),changed_box_disagreement_fraction=c.mean(np.any(data["candidate_boxes"][idx,qa]!=data["candidate_boxes"][idx,qb],axis=1)),
                        current_binary_regret_delta=c.mean(metrics["ppo"]["binary_current_regret"]-metrics[other]["binary_current_regret"])))
                for anchor in (a for a in anchors if a["sequence"]==seq and a["frame"] is not None):
                    row,mask=focus_row(anchor,unit,seed,data,scored,detail,metrics);focus.append(row)
                    for i in np.flatnonzero(mask):
                        row=dict(case_id=anchor["case_id"],**identity,frame=int(data["frame_id"][i]),proposal=int(scored["proposal"][i]),eligible=bool(scored["eligible"][i]),
                            current_Q_gain=float(detail["current_Q_gain"][i]),best_public_iou=float(detail["best"][i]),best_binary_iou=float(detail["binary_best"][i]),
                            own_current_iou=float(overlaps[i,data["selected"][i]]))
                        for rule in c.RULES:row.update({rule+"_choice":int(scored["choices"][rule][i]),rule+"_current_iou":float(metrics[rule]["selected_iou"][i])})
                        for rule in ("ppo","myopic"):row.update({rule+"_logit":float(scored[rule]["logit"][i]),rule+"_probability":float(scored[rule]["probability"][i])})
                        frames.append(row)
            scope.append(dict(phase="evaluation",sequence=seq,fold=unit["fold"],provider=unit["policy"],seed=unit["seed"],source_frames=n,
                scorer_seed_contexts=len(score_seeds),scored_state_occurrences=n*len(score_seeds),source_search_calls=int(data["head_calls"].sum()),
                exact_recorded_choice=True,new_search_calls=0,new_image_reads=0))
        for seed in plan["design"]["seeds"]:divergences.append(divergence(seq,seed,traces["ppo",seed],traces["myopic",seed]))
        print("[cached PPO states audited] "+seq,flush=True)
    # First average scorer/provider seeds WITHIN a video; only then macro-average videos.
    state_metrics=[k for k in rows[0] if k not in ("sequence","fold","provider","provider_seed","scorer_seed","shadow_rule")]
    video=c.aggregate(rows,("sequence","provider","shadow_rule"),state_metrics)
    match_metrics=[k for k in matched[0] if k not in ("sequence","fold","provider","provider_seed","scorer_seed","contrast")]
    matched_video=c.aggregate(matched,("sequence","provider","contrast"),match_metrics)
    candidate_metrics=[k for k in candidates[0] if k not in ("sequence","provider","provider_seed")]
    candidate_video=c.aggregate(candidates,("sequence","provider"),candidate_metrics)
    return dict(same_state_video=rows,same_state_summary=macro(video,("provider","shadow_rule"),state_metrics),
        matched_video=matched,matched_summary=macro(matched_video,("provider","contrast"),match_metrics),
        candidate_video=candidates,candidate_summary=macro(candidate_video,("provider",),candidate_metrics),focus=focus,focus_frames=frames,divergences=divergences),scope


def parameter_delta(before,after,branch):
    names=[k for k in before if k.startswith(branch+".")]
    if set(names)!={k for k in after if k.startswith(branch+".")}:raise ValueError("Controller parameter keys changed")
    difference=np.concatenate([(np.asarray(after[k])-np.asarray(before[k])).ravel() for k in names])
    return float(np.linalg.norm(difference))


def episode_metrics(data,before,after,design,arm,logged):
    eligible=data["eligible"].astype(bool);x=data["x"];pre=c.controller(x,eligible,before,design);post=c.controller(x,eligible,after,design)
    a,target=c.c71.gae(data["reward"],data["value"],design["gamma"],design["gae_lambda"])
    mc=c.discounted_return(data["reward"],design["gamma"]);label=data["myopic_label"].astype(bool)
    low,high=c.DESIGN["probability_band"]
    out=dict(frames=len(x),eligible=int(eligible.sum()),sampled_accepts=int(data["action"].sum()),mean_behavior_reward=c.mean(data["reward"]),
        mean_discounted_behavior_return=c.mean(mc),mean_fixed_GAE_target=c.mean(target),eligible_advantage_std=float(np.std(a[eligible])) if eligible.any() else None,
        actor_parameter_step_l2=parameter_delta(before["state"],after["state"],"actor"),critic_parameter_step_l2=parameter_delta(before["state"],after["state"],"critic"))
    for name,v in (("before",pre),("after",post)):
        out.update({name+"_mean_probability":c.mean(v["probability"],eligible),name+"_mean_abs_logit":c.mean(np.abs(v["logit"]),eligible),
            name+"_probability_near_half_fraction":c.mean((v["probability"]>=low)&(v["probability"]<=high),eligible),
            name+"_deterministic_accept_fraction":c.mean(v["accept"],eligible),
            name+"_current_label_accuracy":c.mean(v["accept"]==label,eligible),
            name+"_value_fixed_GAE_mse":c.mean((v["value"]-target)**2),name+"_value_behavior_MC_mse":c.mean((v["value"]-mc)**2),
            name+"_value_behavior_MC_explained_variance":c.explained_variance(v["value"],mc)})
    out["decision_flip_fraction"]=c.mean(pre["accept"]!=post["accept"],eligible)
    if arm=="ppo":
        np.testing.assert_allclose(pre["value"],data["value"],rtol=0,atol=1e-10)
        lp=-np.logaddexp(0.,np.where(data["action"].astype(bool),-post["logit"],post["logit"]))
        diff=lp[eligible]-data["log_prob"][eligible]
        if not np.isfinite(np.exp(diff)).all():raise ValueError("Nonfinite post-update likelihood ratio")
        out["post_episode_behavior_ratio_clip_fraction"]=c.mean(np.abs(np.expm1(diff))>design["clip"])
        out["post_episode_behavior_approximate_kl"]=c.mean(np.expm1(diff)-diff)
    else:
        out["post_episode_behavior_ratio_clip_fraction"]=None;out["post_episode_behavior_approximate_kl"]=None
    # Logged metrics are fixed source records, not freshly recomputed optimizer gradients.
    for key,value in logged.items():out["logged_"+key]=value
    return out


def training(source):
    plan=source["plan"];rows=[];scope=[]
    for job in plan["jobs"]:
        before=c.c71.sealed(c.c71.step_path(source["root"],job,0))
        for index,episode in enumerate(job["episodes"]):
            seq=episode["sequence"];c.c71.check_fit(plan,job,seq)
            path=c.c71.trace_path(source["root"],job,index);meta=c.e71.train_meta(plan,job,index,before)
            data=c.c71.read_cache(path,meta);after=c.c71.sealed(c.c71.step_path(source["root"],job,index+1))
            if data is None:raise ValueError("Missing committed training episode")
            c.e71.check_step(after,before,meta,path,index,plan["design"])
            reference=next(x for x in source["models"]["episodes"] if x["fold"]==job["fold"] and x["seed"]==job["seed"] and x["episode"]==index)
            for arm in ("ppo","myopic"):
                for key,value in after["stats"][arm].items():
                    if reference[arm+"_"+key]!=value:raise ValueError("Logged training metrics changed")
                stats=episode_metrics(data,before["packs"][arm],after["packs"][arm],plan["design"],arm,after["stats"][arm])
                rows.append(dict(fold=job["fold"],seed=job["seed"],arm=arm,episode=index,sequence=seq,pass_index=episode["pass_index"],**stats))
            scope.append(dict(phase="fit",sequence=seq,fold=job["fold"],provider="sampled_ppo",seed=job["seed"],source_frames=len(data["frame_id"]),
                scorer_seed_contexts=1,scored_state_occurrences=len(data["frame_id"]),source_search_calls=int(data["head_calls"].sum()),
                exact_recorded_choice=True,new_search_calls=0,new_image_reads=0))
            before=after
        print("[cached PPO training audited] "+c.c71.job_key(job),flush=True)
    keys=("fold","seed","arm","pass_index");exclude=set(keys)|{"episode","sequence"};metrics=[k for k in rows[0] if k not in exclude]
    passes=c.aggregate(rows,keys,metrics)
    optimization=[]
    for job in plan["jobs"]:
        for arm in ("ppo","myopic"):
            group=[r for r in rows if r["fold"]==job["fold"] and r["seed"]==job["seed"] and r["arm"]==arm]
            mean_actor=c.mean([r["logged_actor_gradient"] for r in group]);mean_critic=c.mean([r["logged_critic_gradient"] for r in group])
            optimization.append(dict(fold=job["fold"],seed=job["seed"],arm=arm,episodes=len(group),
                mean_logged_actor_gradient=mean_actor,mean_logged_critic_gradient=mean_critic,
                ratio_of_mean_critic_actor_norms=mean_critic/mean_actor if mean_actor else None,
                mean_logged_joint_gradient=c.mean([r["logged_gradient_norm"] for r in group]),
                maximum_logged_actor_gradient=max(r["logged_maximum_actor_gradient"] for r in group),
                maximum_logged_critic_gradient=max(r["logged_maximum_critic_gradient"] for r in group),
                mean_logged_policy_loss=c.mean([r["logged_policy_loss"] for r in group]),mean_logged_value_loss=c.mean([r["logged_value_loss"] for r in group]),
                mean_logged_entropy=c.mean([r["logged_entropy"] for r in group]),mean_logged_approximate_kl=c.mean([r["logged_approximate_kl"] for r in group]),
                mean_logged_clip_fraction=c.mean([r["logged_clip_fraction"] for r in group]),
                mean_actor_step_l2=c.mean([r["actor_parameter_step_l2"] for r in group]),mean_critic_step_l2=c.mean([r["critic_parameter_step_l2"] for r in group])))
    return dict(training_episode=rows,training_summary=passes,optimization_summary=optimization),scope


def summarize(input_root,output_root):
    input_root,output_root=Path(input_root).expanduser().resolve(),Path(output_root).expanduser().resolve()
    if input_root==output_root or input_root in output_root.parents or output_root in input_root.parents:raise ValueError("Use separate new Stage-72 root")
    source=c.load_source(input_root);anchors=c.resolve_cases(source);p71=source["plan"]
    occurrences=sum((len(c.c71.entry_for(p71,u["sequence"])["frames"])-1)*len(c.score_seeds(u,p71)) for u in p71["units"])
    plan=c.seal(dict(schema=c.SCHEMA,role="consumed_development_cached_ppo_diagnostic",source_input=str(input_root),
        source_report_fingerprint=source["report"]["fingerprint"],source_plan_fingerprint=p71["fingerprint"],source_training_fingerprint=source["models"]["fingerprint"],
        source_bindings=source["bindings"],design=dict(c.DESIGN),rules=list(c.RULES),anchors=anchors,expected_scored_eval_state_occurrences=occurrences,
        code_sha256={str(Path(__file__).with_name(n)):c.file_hash(Path(__file__).with_name(n)) for n in c.FILES}))
    with writer_lock(output_root):
        if not (output_root/"manifest.json").exists() and any(p.name!="collect.lock" for p in output_root.iterdir()):raise ValueError("Need empty Stage-72 root")
        c.publish_identical(output_root/"manifest.json",plan)
        tables,eval_scope=evaluation(source,anchors);fit_tables,fit_scope=training(source);tables.update(fit_tables);tables["anchors"]=anchors
        long_scope=eval_scope+fit_scope;tables["scope_video"]=long_scope
        tables["scope"]=[dict(phase=phase,fold=fold,source_traces=len(group),source_frames=sum(r["source_frames"] for r in group),
            scored_state_occurrences=sum(r["scored_state_occurrences"] for r in group),cached_source_search_calls=sum(r["source_search_calls"] for r in group),
            new_search_calls=0,new_image_reads=0,exact_recorded_choices=True)
            for phase in ("evaluation","fit") for fold in [s["fold"] for s in p71["folds"]]
            for group in [[r for r in long_scope if r["phase"]==phase and r["fold"]==fold]]]
        if sum(r["scored_state_occurrences"] for r in eval_scope)!=occurrences:raise ValueError("Same-state scoring scope changed")
        out=output_root/"analysis";out.mkdir(exist_ok=True)
        for name,rows in tables.items():publish_table(out/("stage72_"+name+".csv"),rows)
        c.VERIFY(source["bindings"]);c.VERIFY(plan["code_sha256"])
        report=c.seal(dict(schema=c.SCHEMA,status="COMPLETE_CACHED_PPO_DECISION_TRAINING_DIAGNOSTIC",role=plan["role"],plan_fingerprint=plan["fingerprint"],
            source_report_fingerprint=source["report"]["fingerprint"],source_plan_fingerprint=p71["fingerprint"],source_training_fingerprint=source["models"]["fingerprint"],
            source_bindings_fingerprint=c.fingerprint(source["bindings"]),source_runtime=p71["runtime_identity"],videos=len(p71["entries"]),seeds=p71["design"]["seeds"],
            source_evaluation_traces=len(eval_scope),source_evaluation_frames=sum(r["source_frames"] for r in eval_scope),
            scored_eval_state_occurrences=occurrences,shadow_rule_decisions=occurrences*len(c.RULES),source_training_episodes=len(fit_scope),
            source_training_frames=sum(r["source_frames"] for r in fit_scope),diagnostic_anchor_requests=len(anchors),supported_anchors=sum(a["frame"] is not None for a in anchors),
            new_models=0,new_image_reads=0,new_search_calls=0,new_full_trajectories=0,new_counterfactual_branches=0,
            training_performed=False,reinforcement_learning_performed=False,cpu_controller_rescoring_performed=True,
            source_controls_unchanged=True,exact_recorded_choices=True,method_adoption_gate=False,
            table_sha256={"stage72_"+n+".csv":c.file_hash(out/("stage72_"+n+".csv")) for n in tables},
            warnings=["All twelve source groups and reported events are consumed development evidence; no independent confirmation or method adoption gate.",
                "Same-state scores use only the stored public four candidates and causal features; GT enters offline labels only.",
                "Old single-seed providers are scored by both held-out seed models; new providers use their own seed. Seeds average within video before macro averaging.",
                "Current binary/public regret and current good-to-low replacements are descriptive, not a counterfactual long-term policy return.",
                "Own-history candidate availability and divergence order do not uniquely attribute a future failure to one action.",
                "Training before/after scores are in-sample diagnostics on the preceding PPO behavior trajectory; myopic critic is also trained on that shared return.",
                "Discounted behavior MC and frozen GAE targets have different definitions; post-update MC errors are not off-policy value validation.",
                "Logged gradients/KL/entropy/clip are source records; ratios of mean norms do not reconstruct per-minibatch clipping or prove optimizer causality.",
                "Near-half probabilities do not by themselves prove no learning or justify added PPO budget; no threshold, reward or checkpoint is selected.",
                "No images or SGLA calls are made; small frozen controllers are rescored with NumPy. The original Stage-71 result remains unchanged."],
            next_step="Preserve the fixed diagnostic; distinguish decision errors, candidate support and optimization evidence before proposing a new bounded treatment."))
        c.publish_identical(out/"stage72_diagnostic.json",report)
    print("Stage-72: "+report["status"]+"; no training, image reads or SGLA inference",flush=True)
    print("Reports: "+str(out),flush=True);return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",required=True,type=nonempty_path);p.add_argument("--output_dir",required=True,type=nonempty_path)
    args=p.parse_args();summarize(args.input,args.output_dir)
