"""Diagnose frozen gates from committed caches; no images, inference or new fit."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import stage69_gate_common as c
from stage68_advantage_summary import publish_table
from stage27_uav_common import nonempty_path, tracking_iou
from stage27_uav_collect import writer_lock
from stage4_common import file_hash, fingerprint


def fit_diagnostics(source):
    plan=source["plan"];refs={};stats=[];groups=[];observed=[];points=[]
    for split in plan["folds"]:
        fold=split["fold"];pack=source["models"]["models"][str(fold)]
        x,y,w,rows=c.e68.examples(plan,source["root"],source["old"],fold)
        if any(r["sequence"] not in split["fit"] or r["sequence"] in split["held_out"] for r in rows):raise ValueError("Mixed fit labels")
        observed.extend(rows)
        for j,target in enumerate(c.c68.TARGETS):
            # Reconstruct to verify the stored coefficients, never save/refit a candidate.
            if c.c68.fit_regressor(x,y[:,j],w,plan["design"]["ridge"],target)!=pack[target]:raise ValueError("Stored predictor does not reconstruct")
            pred=np.asarray([c.c68.predict(v,pack[target]) for v in x])
            stats.append(dict(fold=fold,target=target,role="in_sample_fit_check",**c.prediction_stats(pred,y[:,j],w,c.DESIGN["epsilon"])))
            accept=pred>c.DESIGN["epsilon"]
            for positive in (False,True):
                mask=accept==positive;mass=float(w[mask].sum())
                groups.append(dict(fold=fold,target=target,predicted_positive=positive,examples=int(mask.sum()),weight=mass,
                    mean_prediction=float(w[mask]@pred[mask]/mass) if mass else None,
                    mean_observed=float(w[mask]@y[mask,j]/mass) if mass else None,
                    observed_positive_fraction=float(w[mask]@(y[mask,j]>c.DESIGN["epsilon"])/mass) if mass else None))
            points.extend(dict(case_id=r["case_id"],fold=fold,sequence=r["sequence"],frame=r["frame"],target=target,
                               weight=float(weight),prediction=float(p),observed=float(t)) for r,weight,p,t in zip(rows,w,pred,y[:,j]))
        if any(pack["immediate"][k]!=pack["future"][k] for k in ("center","scale")):raise ValueError("Unmatched predictor normalization")
        refs[fold]=c.reference(x,w,pack["future"])
        print("[fit cache audited] fold{} {} paired starts".format(fold,len(x)),flush=True)
    if observed!=source["models"]["support"]:raise ValueError("Stored fit labels or weights changed")
    return refs,stats,groups,points


def paired_transfer(source,refs):
    old=source["old"];plan67=old["plan67"];plan=source["plan"];c67=c.c68.prior;e67=c.e68.e67
    if plan67["design"]["horizon"]!=plan["design"]["horizon"]:raise ValueError("Paired target horizons differ")
    rows=[];policies=[]
    for case in plan67["cases"]:
        fold=case["unit"]["fold"];pack=source["models"]["models"][str(fold)]
        if case["sequence"] not in pack["held_out"] or case["sequence"] in pack["fit"]:raise ValueError("Paired transfer must use held-out source groups")
        if pack["source"]!=old["s66"]["models"]["models"][str(fold)]:raise ValueError("Paired Q continuation differs")
        shared=c67.read_cache(c67.path_for(old["root"],case),c67.metadata(plan67,case))
        trace,entry,_=c67.read_unit(old["s66"],case["sequence"],"protect")
        if shared is None:raise ValueError("Missing shared paired start")
        e67.audit_shared(shared,trace,entry,case,plan67,pack["source"])
        labels={}
        for controller in ("quality","protect"):
            tails={a:c67.read_cache(c67.path_for(old["root"],case,controller,a),c67.metadata(plan67,case,controller,a)) for a in ("allow","veto")}
            if any(v is None for v in tails.values()):raise ValueError("Missing actual paired continuation")
            for action,data in tails.items():e67.audit_tail(data,shared,trace,case,controller,action,plan67,pack["source"])
            labels[controller]=c.e68.label_pair(shared,tails,entry,case,plan67)
        choices,immediate,future,proposal=c.score(shared,pack,plan)
        if proposal[0]!=case["allow_choice"]:raise ValueError("Frozen Q proposal changed at paired point")
        x=c.c68.pair_feature(shared["choice_features"][0],int(proposal[0]))
        support={k:float(v[0]) if np.isfinite(v[0]) else None for k,v in c.support_features(x[None],refs[fold]).items()}
        q=labels["quality"];p=labels["protect"];eps=c.DESIGN["epsilon"]
        row=dict(case_id=case["case_id"],sequence=case["sequence"],frame=case["frame"],fold=fold,
            history="Stage67_protect",horizon=plan67["design"]["horizon"],allow_choice=int(proposal[0]),
            immediate_prediction=float(immediate[0]),immediate_observed=q["immediate"],
            future_prediction=float(future[0]),future_observed_Q=q["future"],future_observed_P=p["future"],
            immediate_error=float(immediate[0]-q["immediate"]),future_error_Q=float(future[0]-q["future"]),
            immediate_positive_correct=bool((immediate[0]>eps)==(q["immediate"]>eps)),
            future_positive_correct_Q=bool((future[0]>eps)==(q["future"]>eps)),
            allow_mean_iou_Q=q["allow_mean_iou"],veto_mean_iou_Q=q["veto_mean_iou"],**support)
        rows.append(row)
        for rule in c.PAIR_RULES:
            choice=int(choices[rule][0])
            if choice not in (0,int(proposal[0])):raise ValueError("Unobserved paired alternative")
            action="allow" if choice else "veto"
            policies.append(dict(case_id=case["case_id"],sequence=case["sequence"],frame=case["frame"],fold=fold,
                shadow_rule=rule,choice=choice,action=action,continuation="frozen_quality_after_forced_action",
                **c.paired_outcome_metrics(q["allow_mean_iou"],q["veto_mean_iou"],action)))
    return rows,policies


def feature_row(detail,mask,**identity):
    eligible=detail["eligible"] & mask;n=int(eligible.sum());current=detail["current_Q_advantage"][eligible]
    pred=detail["immediate"][eligible]
    return dict(**identity,frames=int(mask.sum()),eligible_Q_changes=n,
        immediate_mae_current=c.mean(np.abs(pred-current)),
        immediate_positive_accuracy_current=c.mean((pred>c.DESIGN["epsilon"])==(current>c.DESIGN["epsilon"])) if n else None,
        mean_immediate_prediction=c.mean(pred),mean_current_Q_advantage=c.mean(current),
        mean_future_prediction=c.mean(detail["future"],eligible),future_positive_fraction=c.mean(detail["future"]>c.DESIGN["epsilon"],eligible),
        mean_outside_all=c.mean(detail["outside_all"],eligible),mean_outside_active=c.mean(detail["outside_active"],eligible),
        mean_max_abs_z=c.mean(detail["max_abs_z"],eligible),mean_active_z_rms=c.mean(detail["active_z_rms"],eligible),
        p95_max_abs_z=float(np.quantile(detail["max_abs_z"][eligible],.95)) if n else None)


def first_difference(a,b):
    different=np.any(a!=b,axis=tuple(range(1,a.ndim))) if a.ndim>1 else a!=b
    times=np.flatnonzero(different)
    return int(times[0])+1 if len(times) else None


def divergence(sequence,other,a,b):
    if not np.array_equal(a["frame_id"],b["frame_id"]):raise ValueError("Unaligned full histories")
    idx=np.arange(len(a["frame_id"]))
    return dict(sequence=sequence,comparison="future_vs_"+other,
        first_index_difference=first_difference(a["selected"],b["selected"]),
        first_box_difference=first_difference(a["boxes"],b["boxes"]),
        first_selected_crop_difference=first_difference(a["crop"][idx,a["selected"]],b["crop"][idx,b["selected"]]),
        first_candidate_difference=first_difference(a["candidate_boxes"],b["candidate_boxes"]),
        first_feature_difference=first_difference(a["choice_features"],b["choice_features"]),
        index_different_box_equal_frames=int(((a["selected"]!=b["selected"]) & np.all(a["boxes"]==b["boxes"],axis=1)).sum()))


def evaluation_diagnostics(source,refs,anchors):
    plan=source["plan"];video=[];matched=[];features=[];candidates=[];focus=[];frames=[];divergences=[];scope=[]
    for entry in plan["entries"]:
        sequence=entry["sequence"];traces={}
        for unit in (u for u in plan["units"] if u["sequence"]==sequence):
            data,_=c.read_unit(source,unit);traces[unit["policy"]]=data
            fold=unit["fold"];pack=source["models"]["models"][str(fold)]
            metrics,detail=c.state_metrics(data,entry,pack,plan,refs[fold]);mask=np.ones(len(data["frame_id"]),bool)
            provider=unit["policy"]
            np.testing.assert_array_equal(detail["choices"][provider],data["selected"])
            np.testing.assert_array_equal(detail["immediate"],data["immediate_gain"])
            np.testing.assert_array_equal(detail["future"],data["future_gain"])
            for rule,values in metrics.items():video.append(dict(sequence=sequence,fold=fold,provider=provider,shadow_rule=rule,**c.summarize_state(values)))
            for rule in ("quality","immediate","rate_margin"):
                matched.append(dict(sequence=sequence,provider=provider,contrast="future_minus_"+rule,
                    current_mean_iou_delta=c.mean(metrics["future"]["selected_iou"]-metrics[rule]["selected_iou"]),
                    choice_disagreement_fraction=c.mean(detail["choices"]["future"]!=detail["choices"][rule]),
                    changed_box_disagreement_fraction=c.mean(np.any(data["candidate_boxes"][np.arange(len(mask)),detail["choices"]["future"]]!=
                        data["candidate_boxes"][np.arange(len(mask)),detail["choices"][rule]],axis=1))))
            features.append(feature_row(detail,mask,sequence=sequence,fold=fold,provider=provider))
            candidates.append(dict(sequence=sequence,provider=provider,
                any_good_fraction=c.mean(detail["best"]>=c.DESIGN["good_iou"]),
                all_low_fraction=c.mean(detail["best"]<c.DESIGN["low_iou"]),mean_best_current_iou=c.mean(detail["best"]),
                q0_good_fraction=c.mean(detail["iou"][:,0]>=c.DESIGN["good_iou"])))
            for anchor in (a for a in anchors if a["sequence"]==sequence and a["frame"] is not None):
                first=max(1,anchor["frame"]-c.DESIGN["before"]);last=min(len(mask),anchor["frame"]+c.DESIGN["after"])
                window=(data["frame_id"]>=first) & (data["frame_id"]<=last)
                row=dict(case_id=anchor["case_id"],sequence=sequence,frame=anchor["frame"],provider=provider,
                         first=first,last=last,**feature_row(detail,window))
                for rule in c.RULES:
                    row[rule+"_current_mean_iou"]=c.mean(metrics[rule]["selected_iou"],window)
                    row[rule+"_missed_good_frames"]=int(metrics[rule]["missed_available"][window].sum())
                    row[rule+"_anchor_choice"]=int(detail["choices"][rule][anchor["frame"]-1])
                row.update(any_good_frames=int((detail["best"][window]>=c.DESIGN["good_iou"]).sum()),
                           all_low_frames=int((detail["best"][window]<c.DESIGN["low_iou"]).sum()))
                focus.append(row)
                for i in np.flatnonzero(window):
                    v=dict(case_id=anchor["case_id"],sequence=sequence,frame=int(data["frame_id"][i]),provider=provider,
                        quality_proposal=int(detail["proposal"][i]),eligible=bool(detail["eligible"][i]),
                        immediate_prediction=float(detail["immediate"][i]),future_prediction=float(detail["future"][i]),
                        current_Q_advantage=float(detail["current_Q_advantage"][i]),
                        best_current_iou=float(detail["best"][i]))
                    for rule in c.RULES:v.update({rule+"_choice":int(detail["choices"][rule][i]),rule+"_current_iou":float(metrics[rule]["selected_iou"][i])})
                    for k in ("outside_all","outside_active","max_abs_z","active_z_rms"):v[k]=float(detail[k][i]) if np.isfinite(detail[k][i]) else None
                    frames.append(v)
            scope.append(dict(sequence=sequence,fold=fold,provider=provider,frames=len(mask),
                source_search_calls=int(data["head_calls"].sum()),source_block_calls=int(data["block_calls"].sum()),
                exact_recorded_choice=True,source_control=provider in ("hann","q0","quality","protect")))
        for rule in ("quality","immediate","rate_margin"):divergences.append(divergence(sequence,rule,traces["future"],traces[rule]))
        print("[same-state cache audited] "+sequence,flush=True)
    def macro(rows,keys,exclude=()):
        metrics=[k for k in rows[0] if k not in set(keys) | {"sequence","fold"} | set(exclude)]
        return c.aggregate(rows,keys,metrics)
    return dict(same_state_video=video,same_state_summary=macro(video,("provider","shadow_rule")),matched_video=matched,
        matched_summary=macro(matched,("provider","contrast")),candidate_summary=macro(candidates,("provider",)),
        feature_video=features,feature_shift=macro(features,("fold","provider")),focus=focus,focus_frames=frames,
        divergences=divergences,scope=scope)


def summarize(root68,root69):
    root68,root69=Path(root68).expanduser().resolve(),Path(root69).expanduser().resolve()
    if root68==root69 or root68 in root69.parents or root69 in root68.parents:raise ValueError("Use a separate Stage-69 directory")
    source=c.load_source(root68);anchors=c.cases(source)
    plan=c.seal(dict(schema=c.SCHEMA,role="consumed_development_cached_gate_diagnostic",source_input=str(root68),
        design=dict(c.DESIGN),anchors=anchors,source_bindings=source["bindings"],source_report_fingerprint=source["report"]["fingerprint"],
        source_training_fingerprint=source["models"]["fingerprint"],source_plan_fingerprint=source["plan"]["fingerprint"],
        code_sha256={str(Path(__file__).with_name(n)):file_hash(Path(__file__).with_name(n)) for n in c.FILES}))
    with writer_lock(root69):
        if not (root69/"manifest.json").exists() and any(p.name!="collect.lock" for p in root69.iterdir()):raise ValueError("Need empty new Stage-69 root")
        c.publish_identical(root69/"manifest.json",plan)
        refs,stats,groups,points=fit_diagnostics(source)
        paired,policies=paired_transfer(source,refs)
        tables=evaluation_diagnostics(source,refs,anchors)
        tables.update(anchors=anchors,fit_prediction=stats,fit_sign_groups=groups,fit_points=points,paired_case=paired,paired_policy=policies)
        # Keep detailed source-frame rows on the server; the user bundle has compact aggregates.
        long_scope=tables.pop("scope");tables["scope_video"]=long_scope
        tables["scope"]=[dict(fold=f["fold"],providers=len(c.RULES),videos=len(f["held_out"]),
            provider_frames=sum(r["frames"] for r in long_scope if r["fold"]==f["fold"]),
            cached_source_search_calls=sum(r["source_search_calls"] for r in long_scope if r["fold"]==f["fold"]),
            cached_source_block_calls=sum(r["source_block_calls"] for r in long_scope if r["fold"]==f["fold"]),
            new_search_calls=0,new_image_reads=0,exact_recorded_choices=True) for f in source["plan"]["folds"]]
        out=root69/"analysis";out.mkdir(parents=True,exist_ok=True)
        for name,rows in tables.items():publish_table(out/("stage69_"+name+".csv"),rows)
        c.publish_identical(out/"stage69_fit_reference.json",c.seal(dict(schema=c.SCHEMA,plan_fingerprint=plan["fingerprint"],references={str(k):v for k,v in refs.items()})))
        c.VERIFY(source["bindings"])
        c.VERIFY(plan["code_sha256"])
        report=c.seal(dict(schema=c.SCHEMA,status="COMPLETE_CACHED_PAIRED_GATE_DIAGNOSTIC",role=plan["role"],
            plan_fingerprint=plan["fingerprint"],source_report_fingerprint=source["report"]["fingerprint"],
            source_plan_fingerprint=source["plan"]["fingerprint"],source_training_fingerprint=source["models"]["fingerprint"],
            source_pair_run_fingerprint=source["pairs"]["fingerprint"],source_eval_run_fingerprint=source["run"]["fingerprint"],
            source_bindings_fingerprint=fingerprint(source["bindings"]),runtime=source["plan"]["runtime_identity"],
            videos=len(source["plan"]["entries"]),providers=len(c.RULES),source_fit_pairs=len(source["plan"]["pairs"]),
            transferred_paired_starts=len(paired),paired_shadow_rules=len(c.PAIR_RULES),
            focus_requests=len(anchors),supported_focus_requests=sum(a["frame"] is not None for a in anchors),
            source_frames=sum(r["frames"] for r in long_scope),
            new_models=0,training_performed=False,new_image_reads=0,new_search_calls=0,new_full_trajectories=0,
            new_counterfactual_branches=0,method_adoption_gate=False,external_generalization_established=False,
            exact_recorded_choices=True,source_controls_unchanged=True,source_predictors_reconstructed=True,
            fit_reference_sha256=file_hash(out/"stage69_fit_reference.json"),
            table_sha256={"stage69_"+n+".csv":file_hash(out/("stage69_"+n+".csv")) for n in tables},
            warnings=["Fit prediction errors are in-sample checks, not held-out prediction accuracy.",
                "Stage-67 paired starts have actual Q continuations and held-out source groups, but are consumed development cases.",
                "A paired shadow action is followed by frozen Q, not by that gate; these values are not its deployment performance.",
                "Same-state shadow IoUs are current candidate outcomes; no unobserved future or shadow trajectory AUC is inferred.",
                "Rows overlap within histories/videos/windows; no frame-independent significance or adoption gate is used.",
                "Feature bounds are descriptive fit 1/99 weighted quantiles, never new online thresholds or a calibrated OOD test.",
                "Fit Hann versus held-out Hann also changes source groups and proposal models; it is not pure history drift.",
                "Current/future positive signs use the frozen 1e-6 rule; nonpositive includes zero.",
                "All weights, templates, candidate/selector structure and prior results stay frozen; no GPU inference is performed."],
            next_step="Inspect fixed same-state errors/support and actual paired signs before predeclaring a new intervention or continuation."))
        c.publish_identical(out/"stage69_diagnostic.json",report)
    print("Stage-69: "+report["status"]+"; no images, GPU inference or training")
    print("Reports: "+str(out));return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True,type=nonempty_path)
    parser.add_argument("--output_dir",required=True,type=nonempty_path)
    args=parser.parse_args();summarize(args.input,args.output_dir)
