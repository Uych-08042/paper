"""Video-level development effects for a fixed GT-free reliability selector."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import stage64_reliability_common as c
import stage64_reliability_experiment as e
from stage27_uav_common import tracking_iou, failure_events, nonempty_path
from stage27_uav_collect import writer_lock
from stage46_shift_audit import publish_table
from stage51_state_summary import describe
from stage61_l40_summary import contrast, METRICS


def summarize(root):
    root=Path(root).resolve();plan,source=c.load_plan(root);models=e.load_models(plan,root);run=e.completed(plan,root,models)
    # Independently reconstruct the fixed fits from fit-group native states, with no image/network calls.
    for split in plan["folds"]:
        f,y,w,_=e.examples(plan,source,split["fold"])
        for target in c.TARGETS:
            if c.fit_selector(f,y,w,target,plan["design"]["ridge"])!=models["models"][str(split["fold"])]["models"][target]:raise ValueError("Selector coefficients do not reproduce fit-only labels")
    rows=[];scope=[];events=[];focus=[];hashes={}
    for unit in plan["units"]:
        entry=next(x for x in plan["entries"] if x["sequence"]==unit["sequence"])
        data=c.read_cache(e.path_for(root,unit),c.meta(plan,unit,"evaluation",models))
        if data is None or len(data["frame_id"])!=len(entry["frames"])-1:raise ValueError("Incomplete full trajectory")
        e.audit(data,entry,unit,plan,models);e.check_source(data,source,unit)
        c.e63.source_experiment.bind(hashes,e.path_for(root,unit))
        ref,_,_,head=c.prior.prior.read_trace(source["l40"],e.source_unit(source,unit,"current"))
        nativeunit=dict(unit,origin="native",seed=0)
        native,_,_,_=c.prior.prior.read_trace(source["l40"],e.source_unit(source,nativeunit,"current"))
        gt=np.asarray(entry["ground_truth"],float)[1:];mask=tracking_iou(native["boxes"],gt)<.2
        m,ev=describe(data,entry,native["boxes"],mask);m["native_failure_auc"]=m.pop("source_supervised_failure_auc")
        own_events=failure_events(tracking_iou(data["boxes"],gt))
        arm=unit["origin"]+"_"+unit["policy"]
        row=dict(**unit,arm=arm,**m,introduced_vs_own_current=bool(not failure_events(tracking_iou(ref["boxes"],gt)) and own_events),
                 selected_noncurrent_fraction=float((data["selected"]!=0).mean()),changed_vs_hann_fraction=float(np.mean([
                    int(q)!=c.choose("hann",b,f,None,plan["design"]["tie_epsilon"])[0] for q,b,f in zip(data["selected"],data["features"],data["choice_features"])])))
        rows.append(row);events.extend(dict(sequence=unit["sequence"],arm=arm,seed=unit["seed"],**v) for v in ev)
        if head!=run["observed_heads"].get(c.key(unit)):raise ValueError("Frozen head identity differs")
        scope.append(dict(**unit,head_sha256=head,tracked_frames=len(data["frame_id"]),head_calls=int(data["head_calls"].sum()),
            block_calls=int(data["block_calls"].sum()),maximum_decode_error=float(data["decode_error"].max()),sgla_training_performed=False))
        focus.append(dict(sequence=unit["sequence"],arm=arm,seed=unit["seed"],auc=m["auc"],
            first_failure=None if not own_events else own_events[0]["start"]+1,
            first_recovery=None if not own_events or own_events[0]["recovery"] is None else own_events[0]["recovery"]+1,
            terminal_unrecovered=m["unrecovered_events"],introduced_vs_own_current=row["introduced_vs_own_current"]))
    if hashes!=run["cache_sha256"]:raise ValueError("Full cache set differs")
    metrics=METRICS+("selected_noncurrent_fraction","changed_vs_hann_fraction")
    names=[r["sequence"] for r in plan["entries"]];arms=list(dict.fromkeys(r["arm"] for r in rows));videos=[];policies=[]
    for arm in arms:
        for name in names:
            selected=[r for r in rows if (r["sequence"],r["arm"])==(name,arm)]
            if len(selected)!=(len(plan["seeds"]) if arm.startswith("endpoint_") else 1):raise ValueError("Missing seed or duplicated native")
            videos.append(dict(sequence=name,arm=arm,introduced_failure=any(r["introduced_failure"] for r in selected),
                introduced_vs_own_current=any(r["introduced_vs_own_current"] for r in selected),
                **{k:float(np.mean([r[k] for r in selected])) if all(r[k] is not None for r in selected) else None for k in metrics}))
        selected=[r for r in videos if r["arm"]==arm]
        policies.append(dict(arm=arm,videos=len(selected),introduced_failure_videos=sum(r["introduced_failure"] for r in selected),
            introduced_vs_own_current_videos=sum(r["introduced_vs_own_current"] for r in selected),
            **{k:float(np.mean([r[k] for r in selected if r[k] is not None])) if any(r[k] is not None for r in selected) else None for k in metrics}))
    lookup={(r["sequence"],r["arm"]):r for r in videos};paired=[]
    comparisons=[("native_reliability","native_"+p) for p in ("quality","hann","q0")]+[("endpoint_reliability","endpoint_hann")]
    for left,right in comparisons:
        for metric in ("auc","low_iou_fraction","reliable_auc","longest_failure_fraction"):
            supported=[n for n in names if all(lookup[n,a][metric] is not None for a in (left,right))]
            if not supported:continue
            delta=np.array([lookup[n,left][metric]-lookup[n,right][metric] for n in supported])
            r=contrast(delta,left+"_minus_"+right,metric,plan["design"])
            if left=="native_reliability" and metric=="auc":
                draws=np.random.default_rng(plan["design"]["bootstrap_seed"]).integers(0,len(delta),size=(plan["design"]["bootstrap_samples"],len(delta)))
                r["family_ci_low"],r["family_ci_high"]=map(float,np.quantile(delta[draws].mean(1),[.05/6,1-.05/6]))
            r.update(supported_videos="|".join(supported));paired.append(r)
    get=lambda right,metric:next(r for r in paired if r["contrast"]=="native_reliability_minus_native_"+right and r["metric"]==metric)
    primary=get("quality","auc");hann=get("hann","auc");q0=get("q0","auc")
    own=next(r for r in policies if r["arm"]=="native_reliability")
    guard=own["introduced_vs_own_current_videos"]==0 and get("q0","low_iou_fraction")["mean_delta"]<=0 and get("q0","reliable_auc")["ci_low"]>=plan["design"]["reliable_floor"]
    support=sum(lookup[n,"native_reliability"]["changed_vs_hann_fraction"]>0 for n in names)>=min(4,len(names))
    promising=min(r["family_ci_low"] for r in (primary,hann,q0))>0 and min(primary["mean_delta"],hann["mean_delta"])>=plan["design"]["practical_gain"] and guard and support
    influence=[dict(sequence=n,quality_gain=lookup[n,"native_reliability"]["auc"]-lookup[n,"native_quality"]["auc"],
                    hann_gain=lookup[n,"native_reliability"]["auc"]-lookup[n,"native_hann"]["auc"],q0_gain=lookup[n,"native_reliability"]["auc"]-lookup[n,"native_q0"]["auc"]) for n in names]
    tables=dict(policy_summary=policies,paired=paired,scope=scope,fit_support=models["support"],video_summary=videos,
                seed_video_summary=rows,focus=focus,events=events,video_influence=influence)
    out=root/"analysis";out.mkdir(exist_ok=True)
    with writer_lock(out):
        e.VERIFY(plan["source_file_sha256"]);e.VERIFY(plan["code_sha256"])
        for name,values in tables.items():
            publish_table(out/("stage64_"+name+".csv"),values)
        report=c.seal(dict(schema=c.SCHEMA,status="PROMISING_CANDIDATE_RELIABILITY_PILOT" if promising else "REVIEW_CANDIDATE_RELIABILITY_PILOT",
            role=plan["role"],plan_fingerprint=plan["fingerprint"],run_fingerprint=run["fingerprint"],training_fingerprint=models["fingerprint"],
            primary=primary,against_hann=hann,against_q0=q0,retention_guard=guard,support_pass=support,pilot_pass=promising,
            full_trajectories=len(plan["units"]),formal_search_calls=plan["formal_search_calls"],formal_block_calls=7*plan["formal_search_calls"],
            smoke_calls_reused=plan["smoke_calls_reused"],additional_smoke_calls=0,new_selectors=models["scorers"],
            sgla_training_performed=False,selector_fitting_performed=True,fit_origins=["native"],runtime=plan["runtime_identity"],
            declared_runtime_differences=plan["declared_runtime_differences"],maximum_source_box_error=0.,
            original_checking_labels_loaded=False,external_generalization_established=False,novel_method_established=False,
            evidence_sha256=run["cache_sha256"],table_sha256={"stage64_"+name+".csv":c.file_hash(out/("stage64_"+name+".csv")) for name in tables},
            warnings=["Consumed 12-group development evidence; overlapping fits and frames do not establish independent training uncertainty.",
                "Only native fit-side history/recrop states supply training labels; E trajectories/heads never enter selector fitting.",
                "Reliability labels are current good-minus-low indicators, not unobserved future counterfactual outcomes.",
                "Matched quality control shares states, features, weights, ridge and tie rules; only labels differ.",
                "Native has four policies; E transfer has Hann/reliability only and two frozen OOF seeds averaged within videos.",
                "All policies run four searches per frame, including q0; source current uses one and is only an exact-output reference.",
                "Hann controls and native q0 outputs reproduce source arrays; physical L40 UUID change is declared if present.",
                "No thresholds/hyperparameters/checkpoints are selected on outcomes; no deployment timing or dense-grid repair claim."],
            next_step="Preserve fixed result; assess matched quality contrast, q0 retention and E transfer before a new design"))
        c.publish_identical(out/"stage64_decision.json",report)
    for r in policies:print("{:<22} AUC={:.4f} low-IoU={:.2%}".format(r["arm"],r["auc"],r["low_iou_fraction"]))
    print("Stage-64: "+report["status"]);return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",required=True,type=nonempty_path);summarize(p.parse_args().input)
