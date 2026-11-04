"""Generate fit-local histories, collect matched pairs, fit gates, evaluate ten policies."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import stage70_history_common as c
from stage27_uav_common import checked_image, nonempty_path
from stage27_uav_collect import writer_lock
from stage4_common import file_hash

e68,e67,e63,e66=c.e68,c.e67,c.e63,c.e68.e66
FIELDS=e68.FIELDS


def prepare(source,root,hardware):
    source,root=Path(source).resolve(),Path(root).resolve()
    if source==root or source in root.parents or root in source.parents:raise ValueError("Use separate Stage-70 directory")
    plan,_=c.make_plan(source,hardware)
    with writer_lock(root):
        if not (root/"manifest.json").exists() and any(p.name!="collect.lock" for p in root.iterdir()):raise ValueError("Need empty Stage-70 root")
        c.publish_identical(root/"manifest.json",plan)
    print("Stage-70 prepared: {} fit histories, {} new + {} reused mixed pairs, {} full evaluation trajectories".format(
        len(plan["history_units"]),plan["expected_new_pairs"],plan["expected_reused_pairs"],len(plan["units"])),flush=True)
    print("New search budget: history={} pair={} evaluation={}".format(plan["history_search_calls"],plan["pair_search_calls"],plan["formal_eval_calls"]),flush=True)


def run_record(plan,phase,models=None,pp=None):
    return c.seal(dict(schema=c.SCHEMA,phase=phase,plan_fingerprint=plan["fingerprint"],runtime=plan["runtime_identity"],
        training_fingerprint=None if models is None else models["fingerprint"],pair_plan_fingerprint=None if pp is None else pp["fingerprint"]))


def budget(plan,phase):return plan[{"history":"history_search_calls","pairs":"pair_search_calls","evaluation":"formal_eval_calls","bridge":"source_bridge_search_calls"}[phase]]
def smoke_budget(plan,phase):return plan[{"history":"history_smoke_calls_reused","pairs":"pair_smoke_calls_reused","evaluation":"eval_smoke_calls_reused","bridge":"source_bridge_search_calls"}[phase]]
def record_path(root,phase,smoke):return Path(root)/("smoke/stage70_"+phase+".json" if smoke else phase+"/run.json")


def gate(plan,root,phase,models=None,pp=None):
    record=c.sealed(record_path(root,phase,True))
    if (record.get("status")!="PASS_STAGE70_"+phase.upper()+"_SMOKE" or record.get("run_fingerprint")!=run_record(plan,phase,models,pp)["fingerprint"] or
        record.get("search_calls")!=smoke_budget(plan,phase) or record.get("block_calls")!=7*smoke_budget(plan,phase)):
        raise ValueError("Need exact-budget "+phase+" smoke")
    c.VERIFY(record["cache_sha256"]);return record


def publish_run(root,plan,phase,hashes,heads,calls,count,smoke,models=None,pp=None):
    expected=smoke_budget(plan,phase) if smoke else budget(plan,phase)
    if calls!=expected:raise ValueError(phase+" call budget differs")
    record=c.seal(dict(schema=c.SCHEMA,status="PASS_STAGE70_"+phase.upper()+"_SMOKE" if smoke else "COMPLETE_STAGE70_"+phase.upper(),
        plan_fingerprint=plan["fingerprint"],run_fingerprint=run_record(plan,phase,models,pp)["fingerprint"],
        search_calls=calls,block_calls=7*calls,units=count,cache_sha256=hashes,observed_heads=heads,sgla_training_performed=False))
    c.publish_identical(record_path(root,phase,smoke),record);return record


def verify_record(root,plan,phase,paths,heads,count,models=None,pp=None):
    gate(plan,root,phase,models,pp)
    record=c.sealed(record_path(root,phase,False))
    expected={str(Path(p).with_suffix(s).resolve()) for p in paths for s in (".npz",".receipt.json")}
    if (record.get("status")!="COMPLETE_STAGE70_"+phase.upper() or record.get("run_fingerprint")!=run_record(plan,phase,models,pp)["fingerprint"] or
        record.get("plan_fingerprint")!=plan["fingerprint"] or record.get("search_calls")!=budget(plan,phase) or record.get("block_calls")!=7*budget(plan,phase) or
        record.get("units")!=count or record.get("observed_heads")!=heads or set(record.get("cache_sha256",{}))!=expected or
        record.get("sgla_training_performed") is not False):raise ValueError("Need complete unmixed "+phase)
    c.VERIFY(record["cache_sha256"]);return record


def entry_for(plan,unit):return next(e for e in plan["entries"] if e["sequence"]==unit["sequence"])
def fit_models(source,unit):return dict(models={str(unit["fold"]):c.fit_pack(source,unit["fold"])})
def history_plan(plan):return dict(plan,design=plan["controller_design"])


def source_bridge(root,plan,source,observer,identity,image_reader=checked_image):
    if identity!=plan["runtime_identity"]:raise ValueError("Need bound L40 runtime")
    old=source["s68"];hashes={};heads={};calls=0
    for split in plan["folds"]:
        case=next(x for x in old["plan"]["pairs"] if x["fold"]==split["fold"])
        trace,entry,pack=e68.context(old["plan"],old["old"],case)
        ref=c.c68.read_cache(c.c68.pair_path(old["root"],case),c.c68.pair_meta(old["plan"],case))
        if ref is None:raise ValueError("Missing source shared Hann start")
        unit=case["unit"];path=c.path_for(root,unit,"bridge",True);meta=c.metadata(plan,unit,"bridge",smoke=True);data=c.read_cache(path,meta)
        with e63.source_experiment.installed(observer,[],unit) as head:
            if head!=case["head_sha256"]:raise ValueError("Bridge native head differs")
            if data is None:
                observer.initialize(image_reader(plan["root"],entry["frames"][0]),entry["initial_box"])
                history=e63.history_at(trace,entry["initial_box"],case["frame"],plan)
                row=e67.observe(observer,image_reader(plan["root"],entry["frames"][case["frame"]]),history,case["frame"],"quality",plan,pack)
                data=dict(e67.stack([row]),initial_history=history.copy())
                e68.audit_pair(data,{},trace,entry,case,plan,pack)
                for k in e67.FIELDS:
                    if k!="decode_error":np.testing.assert_array_equal(data[k],ref[k],err_msg="Source fit bridge differs: "+k)
                c.save(path,data,meta)
            e68.audit_pair(data,{},trace,entry,case,plan,pack)
            for k in e67.FIELDS:
                if k!="decode_error":np.testing.assert_array_equal(data[k],ref[k],err_msg="Source fit bridge differs: "+k)
        e63.source_experiment.bind(hashes,path);heads[c.c66.key(unit)]=head;calls+=int(data["head_calls"].sum())
    observer.assert_frozen();c.VERIFY(plan["source_file_sha256"]);c.VERIFY(plan["code_sha256"])
    return publish_run(root,plan,"bridge",hashes,heads,calls,len(plan["folds"]),True)


def histories(root,plan,source,observer,identity,smoke=False,image_reader=checked_image):
    if identity!=plan["runtime_identity"]:raise ValueError("Need bound L40 runtime")
    gate(plan,root,"bridge")
    if not smoke:gate(plan,root,"history")
    units=plan["history_units"][:2] if smoke else plan["history_units"];hashes={};heads={};calls=0
    for unit in units:
        c.check_fit(plan,unit);entry=entry_for(plan,unit);models=fit_models(source,unit);pack=models["models"][str(unit["fold"])]
        split=next(f for f in plan["folds"] if f["fold"]==unit["fold"])
        if pack["fit"]!=split["fit"] or pack["held_out"]!=split["held_out"]:raise ValueError("History controller crosses folds")
        limit=len(entry["frames"])-1
        if smoke:limit=min(limit,plan["design"]["smoke_frames"])
        path=c.path_for(root,unit,"history",smoke);meta=c.metadata(plan,unit,"history",smoke=smoke);data=c.read_cache(path,meta)
        with e63.source_experiment.installed(observer,[],unit) as head:
            if head!=c.head_for(source,unit):raise ValueError("Original native head differs")
            if data is None:
                prefix=None
                if not smoke and unit in plan["history_units"][:2]:prefix=c.read_cache(c.path_for(root,unit,"history",True),c.metadata(plan,unit,"history",smoke=True))
                data=e66.run_actor(observer,entry["frames"],entry["initial_box"],history_plan(plan),unit,pack,limit,prefix,image_reader)
                e66.audit(data,entry,unit,history_plan(plan),models);c.save(path,data,meta)
            e66.audit(data,entry,unit,history_plan(plan),models)
        e63.source_experiment.bind(hashes,path);heads[c.c66.key(unit)]=head;calls+=int(data["head_calls"].sum())
        print("[fit-local history audited] "+c.c66.key(unit),flush=True)
    observer.assert_frozen();c.VERIFY(plan["source_file_sha256"]);c.VERIFY(plan["code_sha256"])
    return publish_run(root,plan,"history",hashes,heads,calls,len(units),smoke)


def history_completed(plan,root,source):
    gate(plan,root,"bridge")
    units=plan["history_units"]
    return verify_record(root,plan,"history",[c.path_for(root,u,"history") for u in units],
        {c.c66.key(u):c.head_for(source,u) for u in units},len(units))


def read_history(plan,root,source,unit):
    c.check_fit(plan,unit);entry=entry_for(plan,unit)
    data=c.read_cache(c.path_for(root,unit,"history"),c.metadata(plan,unit,"history"))
    if data is None or len(data["frame_id"])!=len(entry["frames"])-1:raise ValueError("Missing full fit-local history")
    e66.audit(data,entry,unit,history_plan(plan),fit_models(source,unit));return data


def make_pair_plan(plan,root,source):
    run=history_completed(plan,root,source)
    traces={c.c66.key(u):read_history(plan,root,source,u) for u in plan["history_units"]}
    cases,support=c.resolve_pairs(plan,source,traces,root)
    if sum(not x["reused"] for x in cases)!=plan["expected_new_pairs"] or sum(x["reused"] for x in cases)!=plan["expected_reused_pairs"]:raise ValueError("Matched quotas differ")
    return c.seal(dict(schema=c.SCHEMA,plan_fingerprint=plan["fingerprint"],history_run_fingerprint=run["fingerprint"],
        history_cache_sha256=run["cache_sha256"],cases=cases,support=support))


def resolve(plan,root,source):
    pp=make_pair_plan(plan,root,source);c.publish_identical(Path(root)/"collection/pair_plan.json",pp)
    print("Stage-70 fixed pair plan: {} new, {} reused".format(plan["expected_new_pairs"],plan["expected_reused_pairs"]),flush=True);return pp


def pair_plan(plan,root,source):
    pp=c.sealed(Path(root)/"collection/pair_plan.json")
    if pp!=make_pair_plan(plan,root,source):raise ValueError("Frozen mixed pairs changed")
    return pp


def context(plan,root,source,case,cache=None):
    c.check_fit(plan,case["unit"]);key=(case["fold"],case["sequence"],case["provider"])
    if cache is not None and key in cache:return cache[key]
    if case["reused"]:
        trace,entry,pack=e68.context(source["s68"]["plan"],source["s68"]["old"],case["source_case"])
    else:
        trace=read_history(plan,root,source,case["history_unit"]);entry=entry_for(plan,case);pack=c.fit_pack(source,case["fold"])
    result=(trace,entry,pack)
    if cache is not None:cache[key]=result
    return result


def read_pair(root,plan,source,pp,case):
    if case["reused"]:
        old=source["s68"];base=case["source_case"]
        return (c.c68.read_cache(c.c68.pair_path(old["root"],base),c.c68.pair_meta(old["plan"],base)),
            {a:c.c68.read_cache(c.c68.pair_path(old["root"],base,a),c.c68.pair_meta(old["plan"],base,a)) for a in ("allow","veto")})
    return c.read_cache(c.pair_path(root,case),c.pair_meta(plan,pp,case)),{a:c.read_cache(c.pair_path(root,case,a),c.pair_meta(plan,pp,case,a)) for a in ("allow","veto")}


def collect(root,plan,source,observer,identity,smoke=False,image_reader=checked_image):
    if identity!=plan["runtime_identity"]:raise ValueError("Need bound L40 runtime")
    pp=pair_plan(plan,root,source)
    if not smoke:gate(plan,root,"pairs",pp=pp)
    cases=[x for x in pp["cases"] if not x["reused"]];cases=cases[:1] if smoke else cases
    hashes={};heads={};calls=0;contexts={}
    for case in cases:
        trace,entry,pack=context(plan,root,source,case,contexts)
        if file_hash(c.path_for(root,case["history_unit"],"history"))!=case["source_trace_sha256"]:raise ValueError("History parent changed")
        with e63.source_experiment.installed(observer,[],case["unit"]) as head:
            if head!=case["head_sha256"]:raise ValueError("Original native head differs")
            path=c.pair_path(root,case);meta=c.pair_meta(plan,pp,case);shared=c.read_cache(path,meta)
            if shared is None:
                observer.initialize(image_reader(plan["root"],entry["frames"][0]),entry["initial_box"])
                history=e63.history_at(trace,entry["initial_box"],case["frame"],plan)
                row=e67.observe(observer,image_reader(plan["root"],entry["frames"][case["frame"]]),history,case["frame"],"quality",plan,pack)
                shared=dict(e67.stack([row]),initial_history=history.copy())
                e68.audit_pair(shared,{},trace,entry,case,plan,pack);c.save(path,shared,meta)
            tails={}
            for action in ("veto","allow"):
                p=c.pair_path(root,case,action);m=c.pair_meta(plan,pp,case,action);data=c.read_cache(p,m)
                if data is None:
                    data=e67.collect_tail(observer,entry,shared,case,"quality",action,plan,pack,image_reader)
                    e68.audit_pair(shared,{action:data},trace,entry,case,plan,pack);c.save(p,data,m)
                tails[action]=data;e63.source_experiment.bind(hashes,p);calls+=int(data["head_calls"].sum())
            e68.audit_pair(shared,tails,trace,entry,case,plan,pack)
        e63.source_experiment.bind(hashes,path);calls+=int(shared["head_calls"].sum());heads[case["case_id"]]=head
        print("[mixed pair audited] "+case["case_id"],flush=True)
    observer.assert_frozen();c.VERIFY(plan["source_file_sha256"]);c.VERIFY(plan["code_sha256"])
    return publish_run(root,plan,"pairs",hashes,heads,calls,len(cases),smoke,pp=pp)


def pairs_completed(plan,root,source,pp):
    new=[x for x in pp["cases"] if not x["reused"]]
    return verify_record(root,plan,"pairs",[c.pair_path(root,x,a) for x in new for a in (None,"allow","veto")],
        {x["case_id"]:x["head_sha256"] for x in new},len(new),pp=pp)


def examples(plan,root,source,pp,fold):
    cases=[x for x in pp["cases"] if x["fold"]==fold];split=next(f for f in plan["folds"] if f["fold"]==fold)
    total=sum(plan["design"][p+"_pairs"] for p in c.PROVIDERS);contexts={};x=[];y=[];w=[];rows=[]
    for case in cases:
        trace,entry,pack=context(plan,root,source,case,contexts);shared,tails=read_pair(root,plan,source,pp,case)
        if shared is None or any(v is None for v in tails.values()):raise ValueError("Missing committed mixed fit pair")
        e68.audit_pair(shared,tails,trace,entry,case,plan,pack)
        label=e68.label_pair(shared,tails,entry,case,plan);weight=1/(len(split["fit"])*total)
        x.append(c.c68.pair_feature(shared["choice_features"][0],case["allow_choice"]));y.append([label[t] for t in c.c68.TARGETS]);w.append(weight)
        rows.append(dict(case_id=case["case_id"],fold=fold,sequence=case["sequence"],provider=case["provider"],frame=case["frame"],
            allow_choice=case["allow_choice"],weight=weight,reused=case["reused"],**label))
    if len(x)!=len(split["fit"])*total:raise ValueError("Mixed fit count differs from control")
    return np.asarray(x),np.asarray(y),np.asarray(w),rows


def build_models(plan,root,source):
    pp=pair_plan(plan,root,source);pairrun=pairs_completed(plan,root,source,pp)
    # Reconstruct the original control, including its all-fit-Hann rate threshold.
    if source["s68"]["models"]!=e68.build_models(source["s68"]["plan"],source["s68"]["root"],source["s68"]["old"]):raise ValueError("Hann-only source does not reconstruct")
    mixed={};support=[]
    for split in plan["folds"]:
        fold=split["fold"];x,y,w,rows=examples(plan,root,source,pp,fold)
        predictors={t:c.c68.fit_regressor(x,y[:,j],w,plan["design"]["ridge"],t) for j,t in enumerate(c.c68.TARGETS)}
        gains=[];eligible=[];accepted=[];weights=[];pack=c.fit_pack(source,fold)
        # Both rate controls are calibrated on the SAME all-fit Hann states.
        for seq in split["fit"]:
            trace,_,_=c.c68.prior.read_unit(source["s68"]["old"]["s66"],seq,"hann")
            for base,f in zip(trace["features"],trace["choice_features"]):
                decision=c.c66.decide("quality",base,f,pack,plan["controller_design"]);q=decision["selected"]
                gains.append(decision["quality_gain"]);eligible.append(q!=0)
                accepted.append(bool(q and c.c68.predict(c.c68.pair_feature(f,q),predictors["future"])>plan["design"]["tie_epsilon"]))
                weights.append(1/(len(split["fit"])*len(trace["frame_id"])))
        margin=c.c66.match_margin(gains,eligible,accepted,weights);margin["future_fit_acceptance"]=margin.pop("protected_fit_acceptance")
        margin.update(calibration_states=len(gains),calibration_origin="all_fit_native_hann_frames")
        mixed[str(fold)]=dict(fit=split["fit"],held_out=split["held_out"],
            source=source["s68"]["models"]["models"][str(fold)]["source"],rate_margin=margin,**predictors)
        support.extend(rows)
    return c.seal(dict(schema=c.SCHEMA,status="COMPLETE_MATCHED_HISTORY_MODELS",plan_fingerprint=plan["fingerprint"],
        pair_plan_fingerprint=pp["fingerprint"],pair_run_fingerprint=pairrun["fingerprint"],source_training_fingerprint=source["s68"]["models"]["fingerprint"],
        baseline=source["s68"]["models"]["models"],mixed=mixed,support=support,new_predictors=2*len(mixed),sgla_training_performed=False))


def fit(plan,root,source):
    models=build_models(plan,root,source);c.VERIFY(plan["source_file_sha256"]);c.VERIFY(plan["code_sha256"])
    with writer_lock(Path(root)/"models"):c.publish_identical(Path(root)/"models/training.json",models)
    print("Stage-70 CPU fit: {} mixed predictors; Hann-only controls reused unchanged".format(models["new_predictors"]),flush=True);return models


def load_models(plan,root,source):
    models=c.sealed(Path(root)/"models/training.json");pp=pair_plan(plan,root,source);run=pairs_completed(plan,root,source,pp)
    if (models.get("status")!="COMPLETE_MATCHED_HISTORY_MODELS" or models.get("plan_fingerprint")!=plan["fingerprint"] or
        models.get("pair_plan_fingerprint")!=pp["fingerprint"] or models.get("pair_run_fingerprint")!=run["fingerprint"] or
        models.get("source_training_fingerprint")!=source["s68"]["models"]["fingerprint"] or
        models.get("baseline")!=source["s68"]["models"]["models"] or set(models.get("mixed",{}))!={str(f["fold"]) for f in plan["folds"]}):raise ValueError("Need bound matched history models")
    for split in plan["folds"]:
        p=models["mixed"][str(split["fold"])]
        if p["fit"]!=split["fit"] or p["held_out"]!=split["held_out"] or p["source"]!=models["baseline"][str(split["fold"])]["source"]:raise ValueError("Mixed model/source fold changed")
        for target in c.c68.TARGETS:
            m=p[target]
            if m["target"]!=target or any(len(m[k])!=128 for k in ("center","scale","weights")) or not all(np.isfinite(m[k]).all() for k in ("center","scale","weights","bias")) or np.any(np.asarray(m["scale"])<=0):raise ValueError("Invalid mixed predictor")
        if not np.isfinite(p["rate_margin"]["threshold"]) or p["rate_margin"]["threshold"]<0:raise ValueError("Invalid mixed rate control")
    return models


def audit_eval(data,entry,unit,plan,models):
    canonical=dict(unit,policy=c.canonical(unit["policy"]));pack=c.model_pack(models,unit)
    e68.audit_eval(data,entry,canonical,plan,dict(models={str(unit["fold"]):pack}))


def check_source(data,source,unit):
    if unit["policy"].startswith("mixed_"):return
    old=source["s68"];u=next(u for u in old["plan"]["units"] if all(u[k]==unit[k] for k in ("sequence","fold","policy","origin","seed")))
    ref=c.c68.read_cache(c.c68.eval_path(old["root"],u),c.c68.eval_meta(old["plan"],u,old["models"]))
    if ref is None:raise ValueError("Missing original Hann-only policy")
    for k in FIELDS:
        if k!="decode_error":np.testing.assert_array_equal(data[k],ref[k][:len(data["frame_id"])],err_msg="Stage-70 source control differs: "+k)


def evaluate(root,plan,source,observer,identity,smoke=False,image_reader=checked_image):
    if identity!=plan["runtime_identity"]:raise ValueError("Need bound L40 runtime")
    models=load_models(plan,root,source)
    if not smoke:gate(plan,root,"evaluation",models)
    hashes={};heads={};calls=0
    units=[u for u in plan["units"] if not smoke or u["sequence"]==plan["entries"][0]["sequence"]]
    for unit in units:
        entry=entry_for(plan,unit);limit=len(entry["frames"])-1
        if smoke:limit=min(limit,plan["design"]["smoke_frames"])
        path=c.path_for(root,unit,"evaluation",smoke);meta=c.metadata(plan,unit,"evaluation",models,smoke);data=c.read_cache(path,meta)
        with e63.source_experiment.installed(observer,[],unit) as head:
            if head!=c.head_for(source,unit):raise ValueError("Original native head differs")
            if data is None:
                prefix=None
                if not smoke and unit["sequence"]==plan["entries"][0]["sequence"]:
                    prefix=c.read_cache(c.path_for(root,unit,"evaluation",True),c.metadata(plan,unit,"evaluation",models,True))
                    if prefix is None:raise ValueError("Missing reusable ten-policy smoke")
                data=e68.run_actor(observer,entry,dict(unit,policy=c.canonical(unit["policy"])),plan,c.model_pack(models,unit),limit,prefix,image_reader)
                audit_eval(data,entry,unit,plan,models);check_source(data,source,unit);c.save(path,data,meta)
            audit_eval(data,entry,unit,plan,models);check_source(data,source,unit)
        e63.source_experiment.bind(hashes,path);heads[c.c66.key(unit)]=head;calls+=int(data["head_calls"].sum())
        print("[history-source evaluated] "+c.c66.key(unit),flush=True)
    observer.assert_frozen();c.VERIFY(plan["source_file_sha256"]);c.VERIFY(plan["code_sha256"])
    return publish_run(root,plan,"evaluation",hashes,heads,calls,len(units),smoke,models)


def completed(plan,root,source,models):
    units=plan["units"]
    return verify_record(root,plan,"evaluation",[c.path_for(root,u,"evaluation") for u in units],
        {c.c66.key(u):c.head_for(source,u) for u in units},len(units),models)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",required=True,choices=("prepare","source_bridge","history_smoke","histories","resolve","pair_smoke","collect","fit","smoke","evaluate"))
    parser.add_argument("--input",required=True,type=nonempty_path);parser.add_argument("--output_dir",type=nonempty_path)
    parser.add_argument("--checkpoint",default=os.environ.get("CHECKPOINT"));parser.add_argument("--gpu",type=int,default=0);parser.add_argument("--parameter",default="deit_distilled")
    args=parser.parse_args()
    if args.mode=="prepare":
        if not args.output_dir:parser.error("prepare requires --output_dir")
        import torch
        source=c.load_source(args.input)
        hardware=c.c66.c64.prior.prior.c.prior.device_binding(source["s68"]["plan"],args.gpu,torch)
        prepare(args.input,args.output_dir,hardware);return
    root=Path(args.input).resolve();plan,source=c.load_plan(root)
    if args.mode in ("resolve","fit"):
        with writer_lock(root):
            if args.mode=="resolve":resolve(plan,root,source)
            else:fit(plan,root,source)
        return
    if not args.checkpoint:parser.error("Need original SGLA V1 checkpoint")
    with writer_lock(root):
        native,identity=e63.startup.checked_runtime(args,plan,root);observer=e63.Capture(native)
        try:
            if args.mode=="source_bridge":source_bridge(root,plan,source,observer,identity)
            elif args.mode in ("history_smoke","histories"):histories(root,plan,source,observer,identity,args.mode=="history_smoke")
            elif args.mode in ("pair_smoke","collect"):collect(root,plan,source,observer,identity,args.mode=="pair_smoke")
            else:evaluate(root,plan,source,observer,identity,args.mode=="smoke")
        finally:observer.handle.remove()


if __name__=="__main__":main()
