"""Immutable cached PPO diagnostics; NumPy scorers only, no new SGLA calls."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import stage71_ppo_common as c71
import stage71_ppo_experiment as e71
from stage27_uav_common import failure_events, tracking_iou

SCHEMA="stage72_cached_ppo_decision_training_diagnostic_v1"
FILES=("stage72_ppo_common.py","stage72_ppo_audit.py","test_stage72_ppo.py","run_stage72_ppo_audit.sh")
DESIGN=dict(before=30,after=30,good_iou=.5,low_iou=.2,epsilon=1e-6,
            boundary_atol=1e-12,probability_band=[.45,.55],variance_floor=1e-12)
RULES=("q0","quality","ppo","myopic","ppo_rate")
CONFIG=(
    dict(case_id="C01",sequence="uav0000304_00253_s",policy="ppo",seed=2172,selector="first_failure"),
    dict(case_id="C02",sequence="uav0000304_00253_s",policy="ppo",seed=2171,selector="first_failure"),
    dict(case_id="C03",sequence="uav0000307_04531_s",policy="ppo",seed=2171,selector="first_failure"),
    dict(case_id="C04",sequence="uav0000307_04531_s",policy="myopic",seed=2171,selector="first_recovery"),
    dict(case_id="C05",sequence="uav0000307_04531_s",policy="ppo",seed=2172,selector="first_recovery"),
    dict(case_id="C06",sequence="uav0000176_00000_s",policy="ppo",seed=2172,selector="first_recovery"),
    dict(case_id="C07",sequence="uav0000204_00000_s",policy="q0",seed=0,selector="first_failure"),
    dict(case_id="C08",sequence="uav0000226_05370_s",policy="ppo_rate",seed=2172,selector="first_failure"))
seal,VERIFY,publish_identical=c71.seal,c71.VERIFY,c71.publish_identical
file_hash,fingerprint=c71.file_hash,c71.fingerprint


def load_source(root):
    root=Path(root).expanduser().resolve();plan,old=c71.load_plan(root)
    models=e71.load_models(root,plan,old);run=e71.completed(root,plan,old,models)
    report=c71.sealed(root/"analysis/stage71_decision.json")
    tables={"stage71_"+n+".csv" for n in ("policy_summary","paired","calibration","training_summary","training_scope","fit_support",
        "focus","scope","video_influence","seed_effects","seed_video_summary","video_summary","events")}
    if (report.get("status") not in ("REVIEW_BOUNDED_SEARCH_PPO_PILOT","PROMISING_BOUNDED_SEARCH_PPO_PILOT") or
        report.get("plan_fingerprint")!=plan["fingerprint"] or report.get("training_fingerprint")!=models["fingerprint"] or
        report.get("run_fingerprint")!=run["fingerprint"] or report.get("runtime")!=plan["runtime_identity"] or
        report.get("source_report_fingerprint")!=old["report"]["fingerprint"] or
        report.get("source_training_fingerprint")!=old["models"]["fingerprint"] or
        report.get("source_bindings_fingerprint")!=fingerprint(plan["source_file_sha256"]) or
        report.get("training_evidence_fingerprint")!=fingerprint(models["training_evidence_sha256"]) or
        report.get("eval_evidence_fingerprint")!=fingerprint(run["cache_sha256"]) or
        report.get("sgla_training_performed") is not False or report.get("reinforcement_learning_performed") is not True or
        report.get("source_quality_unchanged") is not True or report.get("maximum_source_control_error")!=0. or
        report.get("fit_search_calls")!=plan["training_search_calls"] or report.get("eval_search_calls")!=plan["formal_eval_calls"] or
        report.get("full_evaluation_trajectories")!=len(plan["units"]) or set(report.get("table_sha256",{}))!=tables):
        raise ValueError("Need complete unmixed Stage-71 PPO result and frozen evidence")
    bindings=dict(plan["source_file_sha256"]);bindings.update(plan["code_sha256"])
    bindings.update(models["training_evidence_sha256"]);bindings.update(run["cache_sha256"])
    bindings.update({str(root/"analysis"/n):h for n,h in report["table_sha256"].items()})
    for name in ("manifest.json","models/training.json","evaluation/run.json","analysis/stage71_decision.json","smoke/train.json","smoke/evaluate.json","smoke/bridge.json"):
        bindings[str(root/name)]=file_hash(root/name)
    bridge=e71.bridge_gate(root,plan);bindings.update(bridge["cache_sha256"])
    if report.get("source_bridge_fingerprint")!=bridge["fingerprint"]:raise ValueError("Source bridge differs")
    VERIFY(bindings)
    return dict(root=root,plan=plan,old=old,models=models,run=run,report=report,bindings=bindings)


def read_unit(source,unit):
    plan=source["plan"];entry=c71.entry_for(plan,unit["sequence"])
    data=c71.read_cache(c71.eval_path(source["root"],unit),e71.eval_meta(plan,unit,source["models"]))
    if data is None or len(data["frame_id"])!=len(entry["frames"])-1:raise ValueError("Missing full evaluation state provider")
    pack=None if unit["policy"] in c71.CONTROLS else e71.learned_pack(source["models"],unit)
    e71.audit(data,entry,unit,plan,source["old"],pack)
    return data,entry


def resolve_cases(source):
    rows=[];cached={}
    for item in CONFIG:
        units=[u for u in source["plan"]["units"] if all(u[k]==item[k] for k in ("sequence","policy","seed"))]
        if len(units)!=1:raise ValueError("Missing fixed diagnostic provider: "+item["case_id"])
        unit=units[0];key=c71.key(unit)
        if key not in cached:cached[key]=read_unit(source,unit)
        data,entry=cached[key];events=failure_events(tracking_iou(data["boxes"],np.asarray(entry["ground_truth"])[1:]))
        mode=item["selector"]
        if mode=="fixed":frame=item["frame"]
        elif mode=="first_failure":frame=events[0]["start"]+1 if events else None
        elif mode=="first_recovery":
            found=[x["recovery"]+1 for x in events if x["recovery"] is not None];frame=found[0] if found else None
        else:raise ValueError("Unknown fixed anchor rule")
        if frame is not None and not 1<=frame<=len(data["frame_id"]):raise ValueError("Focus outside source")
        rows.append(dict(item,frame=frame,status="supported" if frame is not None else "no_declared_event"))
    if len({r["case_id"] for r in rows})!=len(rows):raise ValueError("Duplicate diagnostic case")
    return rows


def score_seeds(unit,plan):return plan["design"]["seeds"] if unit["policy"] in c71.CONTROLS else [unit["seed"]]


def controller(x,eligible,pack,design):
    """Vectorized stored-observation forward, with exact scalar near-zero decisions."""
    x=np.asarray(x,float);eligible=np.asarray(eligible,bool)
    if x.ndim!=2 or x.shape[1]!=128 or eligible.shape!=(len(x),) or not np.isfinite(x).all():raise ValueError("Invalid scorer input")
    logit=np.asarray(c71.forward(x,pack["state"],"actor"));value=np.asarray(c71.forward(x,pack["state"],"critic"))
    close=np.flatnonzero(np.abs(logit)<=DESIGN["boundary_atol"])
    for i in close:logit[i]=c71.forward(x[i],pack["state"],"actor")
    if not np.isfinite(logit).all() or not np.isfinite(value).all():raise ValueError("Nonfinite frozen controller")
    probability=np.exp(-np.logaddexp(0.,-logit));probability=np.where(eligible,probability,0.)
    return dict(logit=logit,value=value,probability=probability,accept=eligible & (logit>0.))


def score(data,pack,design):
    """No GT, video identity, frame ID or future observation enters scoring."""
    f=np.asarray(data["choice_features"],float);boxes=np.asarray(data["candidate_boxes"],float);q=np.asarray(data["quality_choice"],int)
    n=len(q);idx=np.arange(n)
    if f.shape!=(n,4,64) or boxes.shape!=(n,4,4) or np.any((q<0)|(q>3)):raise ValueError("Invalid cached proposals")
    if pack["ppo"]["norm"]!=pack["myopic"]["norm"]:raise ValueError("Unmatched fitted feature normalization")
    raw=np.concatenate([f[idx,q]-f[:,0],f[:,0]],axis=1);norm=pack["ppo"]["norm"]
    z=(raw-np.asarray(norm["center"]))/np.asarray(norm["scale"])
    x=np.clip(z,-design["feature_clip"],design["feature_clip"])
    eligible=(q!=0)&np.any(boxes[idx,q]!=boxes[:,0],axis=1)
    ppo=controller(x,eligible,pack["ppo"],design);myopic=controller(x,eligible,pack["myopic"],design)
    choices=dict(q0=np.zeros(n,int),quality=q.copy(),ppo=np.where(ppo["accept"],q,0),myopic=np.where(myopic["accept"],q,0),
                 ppo_rate=np.where(eligible & (np.asarray(data["quality_gain"])>pack["ppo"]["rate_margin"]["threshold"]),q,0))
    return dict(choices=choices,proposal=q,eligible=eligible,x=x,unclipped_z=z,feature_clipped=np.abs(z)>design["feature_clip"],ppo=ppo,myopic=myopic)


def label_states(data,entry,scored):
    """Offline current-box labels only; these are not counterfactual futures."""
    y=tracking_iou(data["candidate_boxes"],np.asarray(entry["ground_truth"])[1:,None]);n=len(y);idx=np.arange(n)
    q=scored["proposal"];best=y.max(1);binary=np.maximum(y[:,0],y[idx,q]);gain=y[idx,q]-y[:,0]
    metrics={}
    for rule,choice in scored["choices"].items():
        v=y[idx,choice];accept=choice!=0
        metrics[rule]=dict(selected_iou=v,good=v>=DESIGN["good_iou"],low=v<DESIGN["low_iou"],
            missed_public_good=(best>=DESIGN["good_iou"])&(v<DESIGN["good_iou"]),
            missed_binary_good=(binary>=DESIGN["good_iou"])&(v<DESIGN["good_iou"]),
            q0_good_to_low=(y[:,0]>=DESIGN["good_iou"])&(v<DESIGN["low_iou"]),
            q0_low_to_good=(y[:,0]<DESIGN["low_iou"])&(v>=DESIGN["good_iou"]),
            binary_current_regret=binary-v,public_current_regret=best-v,nonq0=accept,
            harmful_current_accept=accept&(gain < -DESIGN["epsilon"]),
            missed_current_gain=(~accept)&scored["eligible"]&(gain>DESIGN["epsilon"]))
    return metrics,dict(iou=y,best=best,binary_best=binary,current_Q_gain=gain)


def mean(values,mask=None):
    v=np.asarray(values,float);valid=np.isfinite(v)
    if mask is not None:valid &= np.asarray(mask,bool)
    return float(v[valid].mean()) if valid.any() else None


def aggregate(rows,keys,metrics):
    groups={}
    for row in rows:groups.setdefault(tuple(row[k] for k in keys),[]).append(row)
    result=[]
    for values,group in groups.items():
        out=dict(zip(keys,values));out["records"]=len(group)
        for metric in metrics:
            available=[r[metric] for r in group if r[metric] is not None]
            out[metric]=float(np.mean(available)) if available else None
            out[metric+"_supported_records"]=len(available)
        result.append(out)
    return result


def explained_variance(prediction,target):
    p,y=np.asarray(prediction,float),np.asarray(target,float)
    if p.ndim!=1 or y.shape!=p.shape or not len(p) or not np.isfinite(np.r_[p,y]).all():raise ValueError("Invalid critic diagnostic")
    variance=float(np.var(y))
    return None if variance<=DESIGN["variance_floor"] else float(1-np.var(y-p)/variance)


def discounted_return(reward,gamma):
    reward=np.asarray(reward,float)
    if reward.ndim!=1 or not len(reward) or not np.isfinite(reward).all() or not 0<=gamma<=1:raise ValueError("Invalid terminal reward sequence")
    out=np.empty_like(reward);tail=0.
    for i in range(len(out)-1,-1,-1):tail=reward[i]+gamma*tail;out[i]=tail
    return out
