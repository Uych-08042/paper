"""Cached same-state gate diagnostics and true paired-target transfer checks."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import stage68_advantage_common as c68
import stage68_advantage_experiment as e68
from stage27_uav_common import failure_events, tracking_iou
from stage4_common import file_hash

SCHEMA="stage69_cached_paired_gate_diagnostic_v1"
FILES=("stage69_gate_common.py","stage69_gate_audit.py","test_stage69_gate.py","run_stage69_gate.sh")
DESIGN=dict(before=30,after=30,quantiles=[.01,.99],epsilon=1e-6,good_iou=.5,low_iou=.2)
RULES=c68.POLICIES
PAIR_RULES=("q0","quality","protect","rate_margin","immediate","future")
CONFIG=(
    dict(case_id="C01",sequence="uav0000304_00253_s",anchor_policy="immediate",selector="first_failure"),
    dict(case_id="C02",sequence="uav0000304_00253_s",anchor_policy="future",selector="first_failure"),
    dict(case_id="C03",sequence="uav0000307_04531_s",anchor_policy="future",selector="first_failure"),
    dict(case_id="C04",sequence="uav0000307_04531_s",anchor_policy="q0",selector="first_recovery"),
    dict(case_id="C05",sequence="uav0000176_00000_s",anchor_policy="future",selector="first_recovery"),
    dict(case_id="C06",sequence="uav0000147_00000_s",anchor_policy="future",selector="first_recovery"),
    dict(case_id="C07",sequence="uav0000237_00001_s",anchor_policy="future",selector="first_failure"),
    dict(case_id="C08",sequence="uav0000204_00000_s",anchor_policy="q0",selector="first_failure"),
    dict(case_id="C09",sequence="uav0000226_05370_s",anchor_policy="q0",selector="first_recovery"))
seal,VERIFY,publish_identical=c68.seal,c68.VERIFY,c68.publish_identical


def load_source(root):
    root=Path(root).expanduser().resolve();plan,old=c68.load_plan(root)
    models=e68.load_models(plan,root,old);pairs=e68.pair_completed(plan,root);run=e68.completed(plan,root,old,models)
    report=c68.sealed(root/"analysis/stage68_decision.json")
    tables={"stage68_"+n+".csv" for n in ("policy_summary","paired","calibration","fit_summary","pair_support","fit_pairs",
        "pair_scope","scope","focus","video_summary","events","video_influence")}
    if (report.get("status") not in ("REVIEW_PAIRED_TAKEOVER_ADVANTAGE_PILOT","PROMISING_PAIRED_TAKEOVER_ADVANTAGE_PILOT") or
        report.get("plan_fingerprint")!=plan["fingerprint"] or report.get("training_fingerprint")!=models["fingerprint"] or
        report.get("run_fingerprint")!=run["fingerprint"] or report.get("pair_run_fingerprint")!=pairs["fingerprint"] or
        report.get("fit_evidence_sha256")!=pairs["cache_sha256"] or report.get("eval_evidence_sha256")!=run["cache_sha256"] or
        report.get("runtime")!=plan["runtime_identity"] or report.get("fit_search_calls")!=plan["pair_search_calls"] or
        report.get("eval_search_calls")!=plan["formal_eval_calls"] or set(report.get("table_sha256",{}))!=tables or
        report.get("sgla_training_performed") is not False or report.get("source_quality_unchanged") is not True):
        raise ValueError("Need complete unmixed Stage-68 results")
    bindings=dict(plan["source_file_sha256"]);bindings.update(plan["code_sha256"])
    bindings.update(pairs["cache_sha256"]);bindings.update(run["cache_sha256"])
    bindings.update({str(root/"analysis"/n):h for n,h in report["table_sha256"].items()})
    for name in ("manifest.json","fit_runtime.json","models/training.json","collection/run.json","run.json",
                 "evaluation/run.json","smoke/stage68_pair_smoke.json","smoke/stage68_eval_smoke.json","analysis/stage68_decision.json"):
        bindings[str(root/name)]=file_hash(root/name)
    VERIFY(bindings)
    return dict(root=root,plan=plan,old=old,models=models,pairs=pairs,run=run,report=report,bindings=bindings)


def read_unit(source,unit):
    plan=source["plan"];entry=next(x for x in plan["entries"] if x["sequence"]==unit["sequence"])
    data=c68.read_cache(c68.eval_path(source["root"],unit),c68.eval_meta(plan,unit,source["models"]))
    if data is None or len(data["frame_id"])!=len(entry["frames"])-1:raise ValueError("Missing full cached state provider")
    e68.audit_eval(data,entry,unit,plan,source["models"]);e68.check_control(data,source["old"],unit)
    return data,entry


def cases(source):
    rows=[]
    for item in CONFIG:
        unit=next(u for u in source["plan"]["units"] if u["sequence"]==item["sequence"] and u["policy"]==item["anchor_policy"])
        data,entry=read_unit(source,unit);events=failure_events(tracking_iou(data["boxes"],np.asarray(entry["ground_truth"])[1:]))
        mode=item["selector"]
        if mode=="fixed":frame=item["frame"]
        elif mode=="first_failure":frame=events[0]["start"]+1 if events else None
        elif mode=="first_recovery":
            found=[x["recovery"]+1 for x in events if x["recovery"] is not None];frame=found[0] if found else None
        else:raise ValueError("Unknown fixed diagnostic anchor rule")
        if frame is not None and not 1<=frame<=len(data["frame_id"]):raise ValueError("Fixed anchor outside source")
        rows.append(dict(case_id=item["case_id"],sequence=item["sequence"],anchor_policy=item["anchor_policy"],selector=mode,
                         frame=frame,status="supported" if frame is not None else "no_declared_source_event"))
    if len({x["case_id"] for x in rows})!=len(rows):raise ValueError("Duplicate focus ID")
    return rows


def reference(x,w,model):
    x,w=np.asarray(x,float),np.asarray(w,float)
    if x.ndim!=2 or x.shape[1]!=128 or w.shape!=(len(x),) or not len(x) or np.any(w<=0) or not np.isfinite(x).all() or not np.isfinite(w).all():raise ValueError("Invalid paired fit reference")
    w=w/w.sum();bounds=[]
    for j in range(128):
        order=np.argsort(x[:,j],kind="stable");mass=np.cumsum(w[order]);mass/=mass[-1]
        bounds.append([float(x[order[min(int(np.searchsorted(mass,q)),len(x)-1)],j]) for q in DESIGN["quantiles"]])
    return dict(bounds=bounds,center=model["center"],scale=model["scale"],pairs=len(x),
                active=(np.asarray(model["scale"])>1e-6).tolist(),quantiles=DESIGN["quantiles"],definition="weighted inverse CDF of this fold's actual fit pairs")


def support_features(x,ref):
    x=np.asarray(x,float);bounds=np.asarray(ref["bounds"]);active=np.asarray(ref["active"],bool)
    if x.ndim!=2 or x.shape[1]!=128 or not np.isfinite(x).all():raise ValueError("Invalid causal gate features")
    outside=(x<bounds[:,0]) | (x>bounds[:,1]);z=(x-np.asarray(ref["center"]))/np.asarray(ref["scale"])
    return dict(outside_all=outside.mean(1),outside_active=outside[:,active].mean(1) if active.any() else np.full(len(x),np.nan),
                max_abs_z=np.abs(z).max(1),active_z_rms=np.sqrt((z[:,active]**2).mean(1)) if active.any() else np.full(len(x),np.nan))


def mean(values,mask=None):
    a=np.asarray(values,float);valid=np.isfinite(a)
    if mask is not None:valid &= np.asarray(mask,bool)
    return float(a[valid].mean()) if valid.any() else None


def prediction_stats(pred,y,w,epsilon):
    pred,y,w=np.asarray(pred,float),np.asarray(y,float),np.asarray(w,float)
    if pred.ndim!=1 or not len(pred) or not all(v.shape==pred.shape for v in (y,w)) or not np.isfinite(np.r_[pred,y,w]).all() or np.any(w<=0):raise ValueError("Invalid observed prediction target")
    w=w/w.sum()
    accept=pred>epsilon;positive=y>epsilon;mass=float(w[accept].sum())
    return dict(examples=len(y),mean_prediction=float(w@pred),mean_target=float(w@y),mae=float(w@np.abs(pred-y)),
        rmse=float(np.sqrt(w@((pred-y)**2))),predicted_positive_fraction=mass,observed_positive_fraction=float(w[positive].sum()),
        positive_sign_accuracy=float(w@(accept==positive)),positive_precision=float(w[accept & positive].sum()/mass) if mass else None)


def paired_outcome_metrics(allow_mean,veto_mean,chosen):
    if chosen not in ("allow","veto"):raise ValueError("No stored future for this alternative")
    value=float(allow_mean if chosen=="allow" else veto_mean)
    return dict(chosen_mean_iou=value,gain_vs_veto=value-float(veto_mean),oracle_binary_regret=max(float(allow_mean),float(veto_mean))-value)


def score(data,pack,plan):
    choices={p:[] for p in RULES};immediate=[];future=[];proposal=[]
    for base,f in zip(data["features"],data["choice_features"]):
        decisions={p:c68.decide(p,base,f,pack,plan["design"],plan["controller_design"]) for p in RULES}
        for p in RULES:choices[p].append(decisions[p]["selected"])
        immediate.append(decisions["quality"]["immediate_gain"]);future.append(decisions["quality"]["future_gain"])
        proposal.append(decisions["quality"]["quality_choice"])
    return {p:np.asarray(v,int) for p,v in choices.items()},np.asarray(immediate),np.asarray(future),np.asarray(proposal,int)


def state_metrics(data,entry,pack,plan,ref):
    choices,immediate,future,proposal=score(data,pack,plan);n=len(proposal);idx=np.arange(n)
    y=tracking_iou(data["candidate_boxes"],np.asarray(entry["ground_truth"])[1:,None]);best=y.max(1)
    proposed_boxes=data["candidate_boxes"][idx,proposal]
    eligible=(proposal!=0) & np.any(proposed_boxes!=data["candidate_boxes"][:,0],axis=1)
    support={k:np.full(n,np.nan) for k in ("outside_all","outside_active","max_abs_z","active_z_rms")}
    if eligible.any():
        x=np.asarray([c68.pair_feature(f,q) for f,q in zip(data["choice_features"][eligible],proposal[eligible])])
        for k,v in support_features(x,ref).items():support[k][eligible]=v
    out={}
    for p,q in choices.items():
        selected=y[idx,q]
        out[p]=dict(selected_iou=selected,good=selected>=DESIGN["good_iou"],low=selected<DESIGN["low_iou"],
            missed_available=(best>=DESIGN["good_iou"]) & (selected<DESIGN["good_iou"]),
            q0_good_to_low=(y[:,0]>=DESIGN["good_iou"]) & (selected<DESIGN["low_iou"]),
            q0_low_to_good=(y[:,0]<DESIGN["low_iou"]) & (selected>=DESIGN["good_iou"]),
            changed_box=np.any(data["candidate_boxes"][idx,q]!=data["candidate_boxes"][:,0],axis=1),
            nonq0=q!=0,current_oracle_regret=best-selected,disagrees_Q=q!=proposal)
    return out,dict(iou=y,best=best,choices=choices,immediate=immediate,future=future,proposal=proposal,
                    eligible=eligible,current_Q_advantage=y[idx,proposal]-y[:,0],**support)


def summarize_state(metrics,mask=None):
    if mask is None:mask=np.ones(len(metrics["selected_iou"]),bool)
    return {k:mean(v,mask) for k,v in metrics.items()}


def aggregate(rows,keys,metrics):
    groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    result=[]
    for values,items in groups.items():
        out=dict(zip(keys,values));out["videos"]=len(items)
        for k in metrics:
            values=[r[k] for r in items if r[k] is not None]
            out[k]=float(np.mean(values)) if values else None;out[k+"_supported_videos"]=len(values)
        result.append(out)
    return result
