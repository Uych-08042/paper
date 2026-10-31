"""Full-video OOF contrasts and explicit four-probe inference costs."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from stage27_uav_common import nonempty_path, tracking_iou, failure_events
from stage27_uav_collect import writer_lock
from stage46_shift_audit import publish_table
from stage46_motion_summary import paired
from stage51_state_summary import describe
import stage60_state_common as c
import stage60_state_experiment as experiment

METRICS=("auc","low_iou_fraction","longest_failure_fraction","reliable_auc","failure_events",
         "mean_abs_log_size_error","oversize3_fraction","unrecovered_events","recovered_events","relapse_events","control_failure_auc")

def summarize(root):
    root=Path(root).expanduser().resolve(); plan,source=c.load_plan(root)
    gate=experiment.smoke_gate(plan,root); collection=experiment.collection_gate(plan,root)
    records=[(r,None) for r in source["records"]]
    models,training=experiment.load_models(plan,root,records)
    run=c.read_sealed(root/"evaluation/run.json")
    if (run.get("status")!="COMPLETE_CAUSAL_OOF" or run.get("plan_fingerprint")!=plan["fingerprint"] or
        run.get("training_fingerprint")!=training["fingerprint"] or run.get("run_fingerprint")!=experiment.run_record(plan)["fingerprint"]):
        raise ValueError("Need complete unmixed Stage-60 OOF")
    c.verify_files(run["cache_sha256"])
    rows=[]; events=[]; hashes={}; calls=0; maximum=0.; maximum_decode=0.; times={p:[] for p in c.POLICIES}
    for entry in plan["entries"]:
        sequence=entry["sequence"]; fold=next(f["fold"] for f in plan["folds"] if sequence in f["held_out"])
        old=source["arrays"][sequence][1]; native=old["native"]
        current={s:old["endpoint_control_seed"+str(s)] for s in plan["seeds"]}
        mask=np.any([tracking_iou(t["boxes"],np.asarray(entry["ground_truth"])[1:])<.2 for t in current.values()],axis=0)
        metrics,ev=describe(native,entry,native["boxes"],mask)
        metrics["control_failure_auc"]=metrics.pop("source_supervised_failure_auc")
        rows.append(dict(sequence=sequence,arm="native",seed=0,introduced_vs_current=False,**metrics))
        events.extend(dict(sequence=sequence,arm="native",seed=0,**r) for r in ev)
        for seed in plan["seeds"]:
            model=models[fold,seed]
            for policy in c.POLICIES:
                path=root/"evaluation/traces"/(sequence+"__s{}__{}.npz".format(seed,policy))
                data=c.read_cache(path,c.meta(plan,"oof",sequence=sequence,fold=fold,seed=seed,policy=policy,model_fingerprint=model["fingerprint"]))
                if data is None: raise ValueError("Missing complete OOF trajectory: "+str(path))
                if len(data["frame_id"])!=len(entry["frames"])-1: raise ValueError("Truncated formal evaluation")
                maximum=max(maximum,experiment.audit_evaluation(data,entry,plan,policy,model["models"],current[seed]))
                maximum_decode=max(maximum_decode,float(np.max(data["decode_error"])))
                experiment.bind(hashes,path); calls+=int(data["head_calls"].sum())
                metrics,ev=describe(data,entry,native["boxes"],mask)
                metrics["control_failure_auc"]=metrics.pop("source_supervised_failure_auc")
                idx=np.arange(len(data["frame_id"])); selected=data["selected"].astype(int)
                baseline_stable=not failure_events(tracking_iou(current[seed]["boxes"],np.asarray(entry["ground_truth"])[1:]))
                own_events=failure_events(tracking_iou(data["boxes"],np.asarray(entry["ground_truth"])[1:]))
                metrics.update(head_calls=int(data["head_calls"].sum()),block_calls=int(data["block_calls"].sum()),
                    selected_noncurrent_fraction=float((selected!=0).mean()),
                    changed_prediction_fraction=float((np.max(np.abs(data["boxes"]-data["candidate_boxes"][:,0]),axis=1)>1e-6).mean()),
                    changed_crop_fraction=float(np.any(data["crop"][idx,selected]!=data["crop"][:,0],axis=1).mean()),
                    mean_distinct_crop_pixels=float(np.mean([len(set(r.tolist())) for r in data["crop_sha256"]])),
                    introduced_vs_current=bool(baseline_stable and own_events))
                rows.append(dict(sequence=sequence,arm=policy,seed=seed,**metrics))
                events.extend(dict(sequence=sequence,arm=policy,seed=seed,**r) for r in ev)
                times[policy].extend(data["pipeline_ms"][10:].tolist())
    if calls!=plan["expected_evaluation_calls"] or hashes!=run["cache_sha256"] or maximum!=run["maximum_control_box_error"]:
        raise ValueError("OOF receipt/budget no longer matches complete evidence")
    names=[e["sequence"] for e in plan["entries"]]; arms=("native",)+c.POLICIES
    usage=("selected_noncurrent_fraction","changed_prediction_fraction","changed_crop_fraction","mean_distinct_crop_pixels")
    videos=[]; policies=[]
    for arm in arms:
        for name in names:
            chosen=[r for r in rows if (r["sequence"],r["arm"])==(name,arm)]
            if len(chosen)!=(1 if arm=="native" else len(plan["seeds"])): raise ValueError("Missing video/seed")
            out=dict(sequence=name,arm=arm,introduced_failure=any(r["introduced_failure"] for r in chosen),
                     introduced_vs_current=any(r["introduced_vs_current"] for r in chosen))
            for metric in METRICS+(usage if arm!="native" else ()):
                values=[r[metric] for r in chosen]
                out[metric]=float(np.mean(values)) if all(v is not None for v in values) else None
            videos.append(out)
        chosen=[r for r in videos if r["arm"]==arm]; own=[r for r in rows if r["arm"]==arm]
        out=dict(arm=arm,videos=len(names),introduced_failure_videos=sum(r["introduced_failure"] for r in chosen),
                 introduced_vs_current_videos=sum(r["introduced_vs_current"] for r in chosen),
                 search_calls_per_frame=1 if arm in ("native","current") else 4,
                 head_calls=None if arm=="native" else sum(r["head_calls"] for r in own),
                 block_calls=None if arm=="native" else sum(r["block_calls"] for r in own))
        for metric in METRICS+(usage if arm!="native" else ()):
            values=[r[metric] for r in chosen if r[metric] is not None]; out[metric]=float(np.mean(values)) if values else None
            if metric in ("reliable_auc","control_failure_auc"): out[metric+"_supported_videos"]=len(values)
        out.update(median_pipeline_ms=float(np.median(times[arm])) if arm in times and times[arm] else None,
                   p95_pipeline_ms=float(np.quantile(times[arm],.95)) if arm in times and times[arm] else None)
        policies.append(out)
    lookup={(r["sequence"],r["arm"]):r for r in videos}; comparisons=[]
    for left,right in (("future_value","immediate_value"),("future_value","current"),("future_value","response"),("future_value","native"),
                       ("immediate_value","current"),("response","current")):
        for metric in ("auc","low_iou_fraction","longest_failure_fraction","reliable_auc","control_failure_auc"):
            supported=[n for n in names if lookup[n,left][metric] is not None and lookup[n,right][metric] is not None]
            if not supported: continue
            delta=np.array([lookup[n,left][metric]-lookup[n,right][metric] for n in supported])
            draws=np.random.default_rng(plan["design"]["bootstrap_seed"]).integers(0,len(delta),size=(plan["design"]["bootstrap_samples"],len(delta)))
            out=paired(delta,left+"_minus_"+right,metric,draws); out["worst_video_delta"]=float(delta.min())
            family=metric=="auc" and left=="future_value" and right in ("immediate_value","current","response")
            out.update(family_ci_low=float(np.quantile(delta[draws].mean(1),.025/3)) if family else None,
                       family_ci_high=float(np.quantile(delta[draws].mean(1),1-.025/3)) if family else None)
            comparisons.append(out)
    get=lambda contrast,metric:next(r for r in comparisons if (r["contrast"],r["metric"])==(contrast,metric))
    primary=get("future_value_minus_immediate_value","auc"); overall=get("future_value_minus_current","auc")
    simple=get("future_value_minus_response","auc")
    retained=get("future_value_minus_current","reliable_auc"); low=get("future_value_minus_current","low_iou_fraction")
    future=next(r for r in policies if r["arm"]=="future_value")
    positive=[n for n in names if lookup[n,"current"]["auc"]>lookup[n,"native"]["auc"]]
    retention_rows=[]
    for arm in arms:
        gains=[lookup[n,arm]["auc"]-lookup[n,"native"]["auc"] for n in positive]
        retention_rows.append(dict(arm=arm,source_positive_videos=len(positive),positive_videos_remaining=sum(g>0 for g in gains),
            mean_gain_vs_native_on_source_positive=float(np.mean(gains)) if gains else None,
            worst_video_gain_vs_native=min(lookup[n,arm]["auc"]-lookup[n,"native"]["auc"] for n in names)))
    benefits_retained=next(r for r in retention_rows if r["arm"]=="future_value")["positive_videos_remaining"]==len(positive)
    seed_gains=[float(np.mean([next(r["auc"] for r in rows if (r["sequence"],r["arm"],r["seed"])==(n,"future_value",s))-
        next(r["auc"] for r in rows if (r["sequence"],r["arm"],r["seed"])==(n,"immediate_value",s)) for n in names])) for s in plan["seeds"]]
    changed=sum(lookup[n,"future_value"]["changed_prediction_fraction"]>0 for n in names)
    support=changed>=min(4,len(names)) and future["changed_prediction_fraction"]>1e-4
    retention=future["introduced_vs_current_videos"]==0 and low["mean_delta"]<=0 and retained["ci_low"]>=-1. and benefits_retained
    promising=(primary["mean_delta"]>=.5 and min(primary["family_ci_low"],overall["family_ci_low"],simple["family_ci_low"])>0 and
               overall["mean_delta"]>=.5 and min(seed_gains)>0 and retention and support)
    scope=[]; fit_support=[]
    for (fold,seed),model in models.items():
        for target,value in model["models"].items():
            scope.append(dict(fold=fold,seed=seed,target=target,fit_videos=len(model["fit"]),held_out_videos=len(model["held_out"]),
                features=len(c.FEATURES),**{k:v for k,v in value.items() if k not in ("weights","scale")}))
        record,_=experiment.record_for(records,fold,seed)
        for unit in [u for u in plan["fit_units"] if (u["fold"],u["seed"])==(fold,seed)]:
            # Keep this table explicitly fit-only; it is not validation accuracy.
            e=next(e for e in plan["entries"] if e["sequence"]==unit["sequence"])
            values=[]
            for frame in c.anchors(e,plan["design"]):
                path=root/"fit/anchors"/(c.key(unit)+"__t{:06d}.npz".format(frame))
                with np.load(path,allow_pickle=False) as z: values.append({k:z[k] for k in ("future_returns","immediate_returns","crop_sha256")})
            returns=np.array([v["future_returns"] for v in values]); delta=returns-returns[:,:1]
            fit_support.append(dict(**unit,states=len(values),mean_best_future_gain=float(delta.max(1).mean()),
                states_with_positive_future_option=int((delta.max(1)>1e-6).sum()),
                mean_distinct_crop_pixels=float(np.mean([len(set(v["crop_sha256"].tolist())) for v in values]))))
    focus=[r for r in rows if r["sequence"] in ("uav0000176_00000_s","uav0000226_05370_s","uav0000304_00253_s","uav0000307_04531_s","uav0000204_00000_s")]
    tables=dict(policy_summary=policies,video_summary=videos,seed_video_summary=rows,paired=comparisons,scope_summary=scope,
                fit_support=fit_support,retention_summary=retention_rows,focus=focus,events=events)
    with writer_lock(root/"analysis"):
        for name,data in tables.items():
            keys=list(dict.fromkeys(k for row in data for k in row))
            normalized=[{k:row.get(k) for k in keys} for row in data]
            publish_table(root/"analysis"/("stage60_"+name+".csv"),normalized)
        c.verify_files(plan["source_file_sha256"]); c.verify_files(collection["cache_sha256"]); c.verify_files(hashes)
        report=c.seal(dict(schema=c.SCHEMA,status="PROMISING_CAUSAL_SEARCH_STATE_PILOT" if promising else "REVIEW_CAUSAL_SEARCH_STATE_PILOT",
            role="consumed_development_causal_search_state_value",plan_fingerprint=plan["fingerprint"],run_fingerprint=run["fingerprint"],
            pilot_pass=bool(promising),primary=primary,overall=overall,response_control=simple,retention_guard=bool(retention),support_pass=bool(support),
            primary_seed_gains=seed_gains,changed_videos=changed,videos=len(names),seeds=plan["seeds"],
            new_scorers=training["new_scorers"],new_sgla_models=0,sgla_weights_unchanged=True,training_performed=True,
            new_oof_trajectories=plan["expected_evaluation_trajectories"],fit_search_calls=collection["full_search_calls"],
            evaluation_search_calls=calls,formal_search_calls=collection["full_search_calls"]+calls,
            additional_smoke_search_calls=gate["extra_smoke_search_calls"],maximum_control_box_error=maximum,
            maximum_decode_error=maximum_decode,
            model_sha256=training["model_sha256"],evidence_sha256=hashes,
            table_sha256={"stage60_"+n+".csv":c.file_hash(root/"analysis"/("stage60_"+n+".csv")) for n in tables},
            original_checking_labels_loaded=False,external_generalization_established=False,novel_method_established=False,
            warnings=["Same consumed twelve source groups; OOF and conditional video bootstrap are development evidence, not independent confirmation.",
                "Ridge fitting uses only each fold's fit videos, including feature normalization. All SGLA weights stay fixed.",
                "An eight-frame action target switches only the initial parent then continues the ordinary tracker; it is not an optimal value for repeated online selection.",
                "Actors use only own predicted history and current responses. Future GT enters fit labels/evaluation only.",
                "Response/immediate/future policies pay four complete original SGLA search calls per frame; current pays one. No deployment speedup claim.",
                "Current is replayed over each complete video and must match source endpoint control. Native is a frozen reused reference.",
                "Lag-center and causal median-size proposals may not cover states that require earlier protection or wider search.",
                "Scorer fitting is supervised training even though SGLA heads are unchanged. No threshold, horizon, checkpoint or ridge search on these outcomes."],
            next_step="Preserve the fixed pilot; inspect matched value controls, full-video retention, proposal coverage and actual cost before further design"))
        c.publish_identical(root/"analysis/stage60_decision.json",report)
    for r in policies: print("{:<17} AUC={:.4f} low-IoU={:.2%}".format(r["arm"],r["auc"],r["low_iou_fraction"]))
    print("Stage-60: "+report["status"])
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--input",required=True,type=nonempty_path)
    summarize(p.parse_args().input)
