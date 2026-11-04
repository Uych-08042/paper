"""Video-paired development evidence for one fixed-budget live PPO search pilot."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import stage71_ppo_common as c
import stage71_ppo_experiment as e
from stage27_uav_common import tracking_iou, failure_events, nonempty_path
from stage27_uav_collect import writer_lock
from stage51_state_summary import describe
from stage61_l40_summary import contrast, METRICS
from stage68_advantage_summary import publish_table

USAGE = ("selected_nonq0_fraction", "changed_box_fraction", "veto_fraction", "eligible_fraction")
PRIMARY = ("myopic", "quality", "future", "ppo_rate")


def summarize(root):
    root = Path(root).resolve(); plan, source = c.load_plan(root)
    models = e.load_models(root, plan, source); run = e.completed(root, plan, source, models); bridge = e.bridge_gate(root, plan)
    seed_rows = []; focus = []; scope = []; events = []; hashes = {}
    for unit in plan["units"]:
        entry = c.entry_for(plan, unit["sequence"])
        data = c.read_cache(c.eval_path(root, unit), e.eval_meta(plan, unit, models))
        if data is None or len(data["frame_id"]) != len(entry["frames"])-1: raise ValueError("Missing full own-history policy")
        pack = None if unit["policy"] in c.CONTROLS else e.learned_pack(models, unit)
        e.audit(data, entry, unit, plan, source, pack); e.e63.source_experiment.bind(hashes, c.eval_path(root, unit))
        native_unit = dict(unit, policy="q0", seed=0)
        native = c.prior.read_cache(c.prior.path_for(source["root"], native_unit, "evaluation"), c.prior.metadata(source["plan"], native_unit, "evaluation", source["models"]))
        gt = np.asarray(entry["ground_truth"], float)[1:]
        metrics, ev = describe(data, entry, native["boxes"], tracking_iou(native["boxes"], gt)<.2)
        metrics["native_failure_auc"] = metrics.pop("source_supervised_failure_auc")
        usage = dict(selected_nonq0_fraction=float((data["selected"]!=0).mean()),
                     changed_box_fraction=float(np.any(data["boxes"]!=data["candidate_boxes"][:,0],axis=1).mean()),
                     veto_fraction=float(data["veto"].mean()), eligible_fraction=float(data["eligible"].mean()))
        seed_rows.append(dict(**unit, **metrics, **usage))
        own = failure_events(tracking_iou(data["boxes"], gt))
        focus.append(dict(sequence=unit["sequence"], policy=unit["policy"], seed=unit["seed"], auc=metrics["auc"],
            first_failure=None if not own else own[0]["start"]+1, first_recovery=None if not own or own[0]["recovery"] is None else own[0]["recovery"]+1,
            terminal_unrecovered=metrics["unrecovered_events"], introduced_failure=metrics["introduced_failure"]))
        events.extend(dict(sequence=unit["sequence"], policy=unit["policy"], seed=unit["seed"], **v) for v in ev)
        scope.append(dict(**unit, frames=len(data["frame_id"]), search_calls=int(data["head_calls"].sum()), block_calls=int(data["block_calls"].sum()),
            head_sha256=run["observed_heads"][c.key(unit)], maximum_decode_error=float(data["decode_error"].max()),
            exact_old_control=unit["policy"] in c.CONTROLS, sgla_training_performed=False))
    if hashes != run["cache_sha256"]: raise ValueError("Evaluation evidence changed")
    names = [entry["sequence"] for entry in plan["entries"]]; policies = c.CONTROLS+c.LEARNED; videos = []
    for name in names:
        for policy in policies:
            group = [r for r in seed_rows if r["sequence"]==name and r["policy"]==policy]
            if len(group) != (1 if policy in c.CONTROLS else len(plan["design"]["seeds"])): raise ValueError("Missing evaluation seed")
            row = dict(sequence=name, policy=policy, introduced_failure=any(r["introduced_failure"] for r in group))
            for k in METRICS+USAGE:
                values = [r[k] for r in group if r[k] is not None]; row[k] = float(np.mean(values)) if values else None
            videos.append(row)
    macros = []
    for policy in policies:
        group = [r for r in videos if r["policy"]==policy]
        row = dict(policy=policy, videos=len(group), seeds=1 if policy in c.CONTROLS else len(plan["design"]["seeds"]),
                   introduced_failure_videos=sum(r["introduced_failure"] for r in group))
        for k in METRICS+USAGE:
            values = [r[k] for r in group if r[k] is not None]; row[k] = float(np.mean(values)) if values else None
        macros.append(row)
    lookup = {(r["sequence"],r["policy"]):r for r in videos}; paired=[]
    comparisons = PRIMARY+("mixed_future", "q0", "hann")
    for other in comparisons:
        for metric in ("auc", "low_iou_fraction", "reliable_auc", "longest_failure_fraction"):
            valid = [n for n in names if lookup[n,"ppo"][metric] is not None and lookup[n,other][metric] is not None]
            if not valid: continue
            diff = np.asarray([lookup[n,"ppo"][metric]-lookup[n,other][metric] for n in valid])
            row = contrast(diff, "ppo_minus_"+other, metric, plan["design"])
            if metric=="auc" and other in PRIMARY:
                draws = np.random.default_rng(plan["design"]["bootstrap_seed"]).integers(0,len(diff),(plan["design"]["bootstrap_samples"],len(diff)))
                row["family_ci_low"],row["family_ci_high"] = map(float,np.quantile(diff[draws].mean(1),[.05/(2*len(PRIMARY)),1-.05/(2*len(PRIMARY))]))
            row["supported_videos"] = "|".join(valid); paired.append(row)
    get = lambda other,metric: next(r for r in paired if r["contrast"]=="ppo_minus_"+other and r["metric"]==metric)
    seeds = []
    for seed in plan["design"]["seeds"]:
        row = dict(seed=seed)
        for other in comparisons:
            diffs=[]
            for n in names:
                a=next(r for r in seed_rows if r["sequence"]==n and r["policy"]=="ppo" and r["seed"]==seed)
                b=next(r for r in seed_rows if r["sequence"]==n and r["policy"]==other and r["seed"]==(0 if other in c.CONTROLS else seed))
                diffs.append(a["auc"]-b["auc"])
            row["auc_gain_vs_"+other] = float(np.mean(diffs))
        seeds.append(row)
    ppo = next(r for r in macros if r["policy"]=="ppo")
    guard = (ppo["introduced_failure_videos"]==0 and get("q0","low_iou_fraction")["mean_delta"]<=0 and
             get("q0","reliable_auc")["ci_low"]>=plan["design"]["reliable_floor"])
    support = sum(lookup[n,"ppo"]["selected_nonq0_fraction"]>0 for n in names)>=min(4,len(names)) and sum(lookup[n,"ppo"]["veto_fraction"]>0 for n in names)>=min(4,len(names))
    seed_pass = all(r["auc_gain_vs_myopic"]>0 and r["auc_gain_vs_quality"]>0 for r in seeds)
    passes = bool(guard and support and seed_pass and all(get(other,"auc")["family_ci_low"]>0 for other in PRIMARY) and
                  min(get(other,"auc")["mean_delta"] for other in ("myopic","quality"))>=plan["design"]["practical_gain"])
    calibration=[]; training_scope=[]; fit_support=[]
    parameters=2*(128*plan["design"]["hidden"]+plan["design"]["hidden"]+plan["design"]["hidden"]+1)
    for job in plan["jobs"]:
        group=[r for r in models["episodes"] if r["fold"]==job["fold"] and r["seed"]==job["seed"]]
        pack=models["controllers"][c.job_key(job)]
        calibration.append(dict(fold=job["fold"],seed=job["seed"],**pack["ppo"]["rate_margin"]))
        training_scope.append(dict(fold=job["fold"],seed=job["seed"],episodes=len(group),frames=sum(r["frames"] for r in group),
            search_calls=sum(r["search_calls"] for r in group),block_calls=7*sum(r["search_calls"] for r in group),
            parameters_per_arm=parameters,actor_parameters=parameters//2,critic_parameters=parameters//2,
            matched_state_exposure=True,matched_initialization=True,matched_minibatches=True,sgla_trainable_parameters=0,
            ppo_updates=sum(r["ppo_updates"] for r in group),myopic_updates=sum(r["myopic_updates"] for r in group),
            maximum_ppo_actor_gradient=max(r["ppo_maximum_actor_gradient"] for r in group),maximum_myopic_actor_gradient=max(r["myopic_maximum_actor_gradient"] for r in group)))
        for seq in pack["fit"]:
            episodes=[r for r in group if r["sequence"]==seq]
            fit_support.append(dict(fold=job["fold"],seed=job["seed"],sequence=seq,episodes=len(episodes),
                frames=sum(r["frames"] for r in episodes),eligible=sum(r["eligible"] for r in episodes),accepted=sum(r["accepted"] for r in episodes),
                myopic_positive=sum(r["myopic_positive"] for r in episodes),fit_only=True))
    if sum(r["search_calls"] for r in training_scope)!=plan["training_search_calls"]: raise ValueError("Training scope budget changed")
    influence=[]
    for n in names:
        row=dict(sequence=n, **{other+"_auc_gain":lookup[n,"ppo"]["auc"]-lookup[n,other]["auc"] for other in comparisons})
        values=np.asarray([lookup[x,"ppo"]["auc"]-lookup[x,"myopic"]["auc"] for x in names])
        row["primary_leave_one_out"]=float((values.sum()-row["myopic_auc_gain"])/(len(names)-1)) if len(names)>1 else None
        influence.append(row)
    tables=dict(policy_summary=macros,paired=paired,calibration=calibration,training_summary=models["episodes"],training_scope=training_scope,
                fit_support=fit_support,focus=focus,scope=scope,video_influence=influence,seed_effects=seeds,
                seed_video_summary=seed_rows,video_summary=videos,events=events)
    out=root/"analysis";out.mkdir(exist_ok=True)
    with writer_lock(out):
        c.VERIFY(plan["source_file_sha256"]);c.VERIFY(plan["code_sha256"])
        for name,rows in tables.items(): publish_table(out/("stage71_"+name+".csv"),rows)
        total=plan["training_search_calls"]+plan["formal_eval_calls"]+plan["source_bridge_search_calls"]
        report=c.seal(dict(schema=c.SCHEMA,status="PROMISING_BOUNDED_SEARCH_PPO_PILOT" if passes else "REVIEW_BOUNDED_SEARCH_PPO_PILOT",
            role=plan["role"],plan_fingerprint=plan["fingerprint"],training_fingerprint=models["fingerprint"],run_fingerprint=run["fingerprint"],
            source_report_fingerprint=source["report"]["fingerprint"],source_training_fingerprint=source["models"]["fingerprint"],
            source_bindings_fingerprint=c.fingerprint(plan["source_file_sha256"]),runtime=plan["runtime_identity"],declared_runtime_differences=plan["declared_runtime_differences"],
            primary=get("myopic","auc"),against_quality=get("quality","auc"),against_future=get("future","auc"),rate_control=get("ppo_rate","auc"),
            retention_guard=bool(guard),support_pass=bool(support),seed_pass=bool(seed_pass),pilot_pass=passes,videos=len(names),seeds=plan["design"]["seeds"],
            new_controller_models=2*len(plan["jobs"]),fit_episodes=sum(len(j["episodes"]) for j in plan["jobs"]),fit_environment_steps=plan["training_environment_steps"],
            fit_search_calls=plan["training_search_calls"],optimizer_updates_per_arm=plan["optimizer_updates_per_arm"],
            full_evaluation_trajectories=len(plan["units"]),eval_search_calls=plan["formal_eval_calls"],
            source_bridge_search_calls=plan["source_bridge_search_calls"],source_bridge_fingerprint=bridge["fingerprint"],
            total_new_search_calls=total,total_new_block_calls=7*total,train_smoke_calls_reused=plan["train_smoke_calls_reused"],
            eval_smoke_calls_reused=plan["eval_smoke_calls_reused"],additional_smoke_calls=plan["source_bridge_search_calls"],
            sgla_training_performed=False,reinforcement_learning_performed=True,source_quality_unchanged=True,maximum_source_control_error=0.,
            training_evidence_fingerprint=c.fingerprint(models["training_evidence_sha256"]),eval_evidence_fingerprint=c.fingerprint(run["cache_sha256"]),
            original_checking_labels_loaded=False,external_generalization_established=False,novel_method_established=False,
            table_sha256={"stage71_"+n+".csv":c.file_hash(out/("stage71_"+n+".csv")) for n in tables},
            warnings=["Twelve consumed source groups with overlapping fit folds; seeds average within videos before conditional bootstrap.",
                "PPO acts in live four-search SGLA trajectories; no cached counterfactual environment or GT resets after initialization.",
                "The myopic control sees exactly PPO's sampled states, current candidate-IoU labels, architecture, initialization and minibatch updates; it does not collect its own fit histories.",
                "This compares cumulative PPO and current supervised objectives on shared exposure, not a pure delayed-target-only causal ablation.",
                "All current candidate overlaps enter fit supervision only after action selection. No GT, frame ID, source identity or future observation is a controller input.",
                "Each full video is a naturally terminal episode; PPO policy is fixed during its collection, updated only after the episode.",
                "Four passes, two seeds, four epochs and the final checkpoint are fixed. Full episodes vary in length and are not equal per-video gradient weights.",
                "The proposal remains frozen Q; missing good candidates and limited observation cannot be solved by RL alone.",
                "Fit-Hann rate matching need not match realized own-history acceptance or candidate availability.",
                "All evaluation policies run four SGLA searches per frame; no deployment speedup or energy gain is established.",
                "PPO and GAE are established algorithms. Any added value concerns SGLA search decisions and must survive matched full-trajectory comparisons."],
            next_step="Preserve the fixed final budget and both seeds; inspect matched controls and retention, without checkpoint, reward, seed or hyperparameter search on these outcomes."))
        c.publish_identical(out/"stage71_decision.json",report)
    for row in macros: print("{:<16} AUC={:.4f} low-IoU={:.2%}".format(row["policy"],row["auc"],row["low_iou_fraction"]))
    print("Stage-71: "+report["status"]);return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--input",required=True,type=nonempty_path);summarize(parser.parse_args().input)
