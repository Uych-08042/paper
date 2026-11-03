"""OOF full trajectories for matched immediate versus paired Q-continuation gains."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import stage68_advantage_common as c
import stage68_advantage_experiment as e
from stage27_uav_common import tracking_iou, failure_events, nonempty_path
from stage27_uav_collect import writer_lock
from stage51_state_summary import describe
from stage61_l40_summary import contrast, METRICS
from stage46_shift_audit import publish_table as original_publish


def publish_table(path,rows):
    # Some descriptive tables have optional fields. Keep every key, in first-seen
    # order; never infer the whole schema from the first row or drop extra keys.
    fields=list(dict.fromkeys(k for row in rows for k in row))
    original_publish(path,[{k:r.get(k) for k in fields} for r in rows])


def summarize(root):
    root=Path(root).resolve();plan,source=c.load_plan(root);models=e.load_models(plan,root,source)
    pairs=e.pair_completed(plan,root);run=e.completed(plan,root,source,models)
    if models!=e.build_models(plan,root,source):raise ValueError("Paired predictors or fit-rate threshold do not reconstruct")
    rows=[];scope=[];focus=[];events=[];hashes={}
    for u in plan["units"]:
        entry=next(r for r in plan["entries"] if r["sequence"]==u["sequence"])
        data=c.read_cache(c.eval_path(root,u),c.eval_meta(plan,u,models))
        if data is None or len(data["frame_id"])!=len(entry["frames"])-1:raise ValueError("Missing full own-history trajectory")
        e.audit_eval(data,entry,u,plan,models);e.check_control(data,source,u)
        e.e63.source_experiment.bind(hashes,c.eval_path(root,u))
        native,_,_=c.prior.read_unit(source["s66"],u["sequence"],"q0")
        gt=np.asarray(entry["ground_truth"],float)[1:]
        metrics,ev=describe(data,entry,native["boxes"],tracking_iou(native["boxes"],gt)<.2)
        metrics["native_failure_auc"]=metrics.pop("source_supervised_failure_auc")
        rows.append(dict(**u,**metrics,selected_nonq0_fraction=float((data["selected"]!=0).mean()),
            changed_box_fraction=float(np.any(data["boxes"]!=data["candidate_boxes"][:,0],axis=1).mean()),
            quality_nonq0_fraction=float((data["quality_choice"]!=0).mean()),veto_fraction=float(data["veto"].mean())))
        own=failure_events(tracking_iou(data["boxes"],gt))
        focus.append(dict(sequence=u["sequence"],policy=u["policy"],auc=metrics["auc"],
            first_failure=None if not own else own[0]["start"]+1,
            first_recovery=None if not own or own[0]["recovery"] is None else own[0]["recovery"]+1,
            terminal_unrecovered=metrics["unrecovered_events"],introduced_failure=metrics["introduced_failure"]))
        events.extend(dict(sequence=u["sequence"],policy=u["policy"],**v) for v in ev)
        scope.append(dict(**u,head_sha256=run["observed_heads"][e.c66.key(u)],frames=len(data["frame_id"]),
            search_calls=int(data["head_calls"].sum()),block_calls=int(data["block_calls"].sum()),
            maximum_decode_error=float(data["decode_error"].max()),source_control=u["policy"] in ("hann","q0","quality","protect")))
    if hashes!=run["cache_sha256"]:raise ValueError("Evaluation evidence differs")
    policies=[]
    for p in c.POLICIES:
        group=[r for r in rows if r["policy"]==p]
        if len(group)!=len(plan["entries"]):raise ValueError("Missing policy/video")
        aggregate=dict(policy=p,videos=len(group),introduced_failure_videos=sum(r["introduced_failure"] for r in group))
        for k in METRICS+("selected_nonq0_fraction","changed_box_fraction","quality_nonq0_fraction","veto_fraction"):
            values=[r[k] for r in group if r[k] is not None];aggregate[k]=float(np.mean(values)) if values else None
        policies.append(aggregate)
    names=[r["sequence"] for r in plan["entries"]];lookup={(r["sequence"],r["policy"]):r for r in rows};paired=[]
    comparisons=("immediate","rate_margin","quality","protect","hann","q0")
    for other in comparisons:
        for metric in ("auc","low_iou_fraction","reliable_auc","longest_failure_fraction"):
            valid=[n for n in names if all(lookup[n,p][metric] is not None for p in ("future",other))]
            if not valid:continue
            values=np.array([lookup[n,"future"][metric]-lookup[n,other][metric] for n in valid])
            row=contrast(values,"future_minus_"+other,metric,plan["design"])
            if metric=="auc":
                rng=np.random.default_rng(plan["design"]["bootstrap_seed"])
                draws=rng.integers(0,len(values),size=(plan["design"]["bootstrap_samples"],len(values)))
                row["family_ci_low"],row["family_ci_high"]=map(float,np.quantile(values[draws].mean(1),[.05/12,1-.05/12]))
            row["supported_videos"]="|".join(valid);paired.append(row)
    get=lambda p,m:next(r for r in paired if r["contrast"]=="future_minus_"+p and r["metric"]==m)
    future=next(r for r in policies if r["policy"]=="future")
    guard=(future["introduced_failure_videos"]==0 and get("q0","low_iou_fraction")["mean_delta"]<=0 and
           get("q0","reliable_auc")["ci_low"]>=plan["design"]["reliable_floor"])
    support=all(sum(r["fold"]==f["fold"] and r["selected_pairs"]>0 for r in plan["pair_support"])>=min(4,len(f["fit"])) for f in plan["folds"])
    support=support and sum(lookup[n,"future"]["selected_nonq0_fraction"]>0 for n in names)>=min(4,len(names))
    support=support and sum(lookup[n,"future"]["veto_fraction"]>0 for n in names)>=min(4,len(names))
    passes=(guard and support and all(get(p,"auc")["family_ci_low"]>0 for p in comparisons) and
            min(get(p,"auc")["mean_delta"] for p in ("immediate","quality"))>=plan["design"]["practical_gain"])
    calibration=[];fit_summary=[]
    for fold,pack in models["models"].items():
        subset=[r for r in models["support"] if str(r["fold"])==fold]
        calibration.append(dict(fold=int(fold),pairs=len(subset),fit_supported_videos=len({r["sequence"] for r in subset}),**pack["rate_margin"]))
        fit_summary.append(dict(fold=int(fold),pairs=len(subset),
            mean_immediate_gain=sum(r["weight"]*r["immediate"] for r in subset),
            mean_future_gain=sum(r["weight"]*r["future"] for r in subset),
            positive_future_pairs=sum(r["future"]>plan["design"]["tie_epsilon"] for r in subset),
            negative_future_pairs=sum(r["future"] < -plan["design"]["tie_epsilon"] for r in subset),
            sign_disagreement_pairs=sum((r["immediate"]>plan["design"]["tie_epsilon"])!=(r["future"]>plan["design"]["tie_epsilon"]) for r in subset)))
    pair_scope=[]
    for case in plan["pairs"]:
        for a in (None,"allow","veto"):
            data=c.read_cache(c.pair_path(root,case,a),c.pair_meta(plan,case,a))
            pair_scope.append(dict(case_id=case["case_id"],fold=case["fold"],sequence=case["sequence"],action=a or "shared",frames=len(data["frame_id"]),
                search_calls=int(data["head_calls"].sum()),block_calls=int(data["block_calls"].sum())))
    if sum(r["search_calls"] for r in pair_scope)!=plan["pair_search_calls"]:raise ValueError("Fit pair scope budget differs")
    influence=[dict(sequence=n,**{p+"_gain":lookup[n,"future"]["auc"]-lookup[n,p]["auc"] for p in comparisons}) for n in names]
    tables=dict(policy_summary=policies,paired=paired,calibration=calibration,fit_summary=fit_summary,pair_support=plan["pair_support"],
        fit_pairs=models["support"],pair_scope=pair_scope,scope=scope,focus=focus,video_summary=rows,events=events,video_influence=influence)
    out=root/"analysis";out.mkdir(exist_ok=True)
    with writer_lock(out):
        c.VERIFY(plan["source_file_sha256"]);c.VERIFY(plan["code_sha256"])
        for name,values in tables.items():publish_table(out/("stage68_"+name+".csv"),values)
        report=c.seal(dict(schema=c.SCHEMA,status="PROMISING_PAIRED_TAKEOVER_ADVANTAGE_PILOT" if passes else "REVIEW_PAIRED_TAKEOVER_ADVANTAGE_PILOT",
            role=plan["role"],plan_fingerprint=plan["fingerprint"],training_fingerprint=models["fingerprint"],
            pair_run_fingerprint=pairs["fingerprint"],run_fingerprint=run["fingerprint"],primary=get("immediate","auc"),
            rate_control=get("rate_margin","auc"),against_quality=get("quality","auc"),against_protect=get("protect","auc"),
            retention_guard=guard,support_pass=support,pilot_pass=passes,fit_pairs=len(plan["pairs"]),fit_search_calls=plan["pair_search_calls"],
            fit_block_calls=7*plan["pair_search_calls"],full_trajectories=len(plan["units"]),eval_search_calls=plan["formal_eval_calls"],
            eval_block_calls=7*plan["formal_eval_calls"],pair_smoke_calls_reused=plan["pair_smoke_calls_reused"],
            eval_smoke_calls_reused=plan["eval_smoke_calls_reused"],additional_smoke_calls=0,new_predictors=models["new_predictors"],
            sgla_training_performed=False,critic_fitting_performed=True,source_quality_unchanged=True,fit_state_policy="native_hann",
            pair_continuation="fit_local_frozen_quality",maximum_source_box_error=0.,runtime=plan["runtime_identity"],
            declared_runtime_differences=plan["declared_runtime_differences"],original_checking_labels_loaded=False,
            external_generalization_established=False,novel_method_established=False,fit_evidence_sha256=pairs["cache_sha256"],
            eval_evidence_sha256=run["cache_sha256"],table_sha256={"stage68_"+n+".csv":c.file_hash(out/("stage68_"+n+".csv")) for n in tables},
            warnings=["Repeated twelve-source development OOF, not independent confirmation. Bootstrap conditions on overlapping fixed fits.",
                "Uniform eligible fit times on frozen native Hann states; no event/GT-based anchor search or Stage-67 outcome labels enter fitting.",
                "Stored OOF Q/critic decisions on Hann caches are not fit inputs; proposals and tails use the current fold's fit-local frozen Q.",
                "The future target is a one-decision allow-minus-veto 32-frame return under Q, not an optimal value or the deployed gate's own future policy.",
                "Fit native Hann states and deployed own histories differ. Sparse overlapping windows may miss rare recovery opportunities.",
                "Immediate/future models share pairs, 128 causal features, video weights, ridge and zero-gain selection threshold.",
                "Rate margin matches predicted-future acceptance on ALL fit native-Hann frames with equal video mass; realized own-history evaluation rates may differ.",
                "Q, protect, Hann and q0 are new same-device controls and must reproduce Stage-66. Four searches per evaluation frame are charged.",
                "SGLA weights, original first-frame template, candidates and selector remain frozen. No inference speedup or energy claim."],
            next_step="Compare future to matched immediate and fit-rate controls, verify full-trajectory retention, and preserve this fixed result before any refinement"))
        c.publish_identical(out/"stage68_decision.json",report)
    for r in policies:print("{:<13} AUC={:.4f} low-IoU={:.2%}".format(r["policy"],r["auc"],r["low_iou_fraction"]))
    print("Stage-68: "+report["status"]);return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",required=True,type=nonempty_path);summarize(p.parse_args().input)
