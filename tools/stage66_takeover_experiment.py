"""Fit q0 critics and evaluate five native SGLA policies on their own histories."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import stage66_takeover_common as c
from stage27_uav_common import checked_image, crop_geometry, nonempty_path
from stage27_uav_collect import writer_lock

FIELDS = c.e64.FIELDS+("q0_pred_iou", "q0_qualified", "quality_choice", "quality_gain", "veto")


def prepare(source, root, hardware):
    source,root=Path(source).resolve(),Path(root).resolve()
    if source==root or source in root.parents or root in source.parents: raise ValueError("Use a separate Stage-66 directory")
    p,_=c.make_plan(source,hardware)
    with writer_lock(root):
        if not (root/"manifest.json").exists() and any(x.name!="collect.lock" for x in root.iterdir()): raise ValueError("Use empty Stage-66 directory")
        c.publish_identical(root/"manifest.json",p)
    print("Stage-66 prepared: {} native full trajectories / {} searches".format(len(p["units"]),p["formal_search_calls"]))


def build_models(plan, source):
    old=source["s64"]; packs={};support=[]
    for split in plan["folds"]:
        f,y,w,rows=c.e64.examples(old["plan"],old["old"],split["fold"])
        quality=old["models"]["models"][str(split["fold"])]["models"]["quality"]
        if c.c64.fit_selector(f,y,w,"quality",old["plan"]["design"]["ridge"]) != quality: raise ValueError("Source quality fit differs")
        critic=c.fit_absolute(f,y,w,plan["design"]["ridge"])
        base=f[:,:,:24]; temporary=dict(quality=quality,critic=critic)
        q=[];g=[];p=[]
        for b,x in zip(base,f):
            decision=c.decide("protect",b,x,temporary,plan["design"])
            q.append(decision["quality_choice"]!=0);g.append(decision["quality_gain"]);p.append(decision["selected"]!=0)
        margin=c.match_margin(g,q,p,w)
        packs[str(split["fold"])]=dict(fit=split["fit"],held_out=split["held_out"],quality=quality,critic=critic,margin=margin)
        support.extend(rows)
    return c.seal(dict(schema=c.SCHEMA,plan_fingerprint=plan["fingerprint"],status="COMPLETE_NATIVE_Q0_CRITICS",
        source_training_fingerprint=old["models"]["fingerprint"],fit_origins=["native"],new_critics=len(packs),
        source_quality_unchanged=True,sgla_training_performed=False,models=packs,support=support))


def fit(plan,root,source):
    models=build_models(plan,source);c.VERIFY(plan["source_file_sha256"]);c.VERIFY(plan["code_sha256"])
    with writer_lock(Path(root)/"models"):c.publish_identical(Path(root)/"models/training.json",models)
    for fold,pack in models["models"].items():print("[fit] fold {} matched acceptance {}".format(fold,pack["margin"]),flush=True)
    return models


def load_models(plan,root,source):
    models=c.sealed(Path(root)/"models/training.json")
    if models.get("status")!="COMPLETE_NATIVE_Q0_CRITICS" or models.get("plan_fingerprint")!=plan["fingerprint"] or models.get("source_training_fingerprint")!=source["s64"]["models"]["fingerprint"] or models.get("fit_origins")!=["native"]:
        raise ValueError("Need bound native-only q0 critics")
    if set(models["models"])!={str(f["fold"]) for f in plan["folds"]}:raise ValueError("Missing fold critic")
    for split in plan["folds"]:
        p=models["models"][str(split["fold"])];critic=p["critic"]
        if p["fit"]!=split["fit"] or p["held_out"]!=split["held_out"] or p["quality"]!=source["s64"]["models"]["models"][str(split["fold"])]["models"]["quality"]:raise ValueError("Fit or quality binding changed")
        if critic["target"]!="current_q0_iou" or any(len(critic[k])!=64 for k in ("center","scale","weights")) or not all(np.isfinite(critic[k]).all() for k in ("center","scale","weights","bias")) or np.any(np.asarray(critic["scale"])<=0) or not np.isfinite(p["margin"]["threshold"]) or p["margin"]["threshold"]<0:raise ValueError("Invalid critic or margin")
    return models


def path_for(root,unit,smoke=False):return Path(root)/("smoke/traces" if smoke else "evaluation/traces")/(c.key(unit)+".npz")
def run_record(plan,models):return c.seal(dict(schema=c.SCHEMA,plan_fingerprint=plan["fingerprint"],training_fingerprint=models["fingerprint"],runtime=plan["runtime_identity"]))


def check_source(data,source,unit):
    if unit["policy"] in ("hann","q0"):
        c.e64.check_source(data,source["s64"]["old"],unit)
    elif unit["policy"]=="quality":
        old=source["s64"];u=next(u for u in old["plan"]["units"] if all(u[k]==unit[k] for k in ("sequence","origin","seed","policy")))
        ref=c.c64.read_cache(c.e64.path_for(old["root"],u),c.c64.meta(old["plan"],u,"evaluation",old["models"]))
        if ref is None:raise ValueError("Missing source quality control")
        c.c64.e63.exact_source(data,ref,1,"Stage-66 frozen quality control")


def audit(data,entry,unit,plan,models):
    n=len(data["frame_id"]);c.c64.e63.source_common.prior.audit_trace(data,n,4)
    if not 0<n<=len(entry["frames"])-1 or np.max(data["decode_error"])>plan["actor_design"]["box_atol"]:raise ValueError("Invalid decode or length")
    for name in FIELDS[len(c.e64.FIELDS):]:
        if data[name].shape!=(n,) or not np.isfinite(data[name]).all():raise ValueError("Invalid controller trace")
    history=[np.asarray(entry["initial_box"],float)];pack=models["models"][str(unit["fold"])]
    for i in range(n):
        parents=c.c64.e63.source_common.prior.history_parents(history,plan["actor_design"])
        np.testing.assert_array_equal(data["candidate_parents"][i],parents)
        np.testing.assert_array_equal(data["history_latest"][i],history[-1])
        features=c.c64.choice_features(data["features"][i],data["candidate_boxes"][i],history)
        np.testing.assert_array_equal(features,data["choice_features"][i])
        for k,v in c.decide(unit["policy"],data["features"][i],features,pack,plan["design"]).items():np.testing.assert_array_equal(data[k][i],v)
        for q,p in enumerate(parents):np.testing.assert_array_equal(data["crop"][i,q],crop_geometry(p,data["image_hw"][i],plan["search_factor"],plan["search_size"])[0])
        history.append(data["boxes"][i]);history=history[-plan["actor_design"]["history"]:]


def run_actor(observer,frames,initial_box,plan,unit,pack,limit,prefix=None,image_reader=checked_image):
    observer.initialize(image_reader(plan["root"],frames[0]),initial_box)
    history=[np.asarray(initial_box,float)];rows=[];start=1
    if prefix is not None:
        start=len(prefix["frame_id"])+1
        for b in prefix["boxes"]:history.append(b);history=history[-plan["actor_design"]["history"]:]
    ap={k:plan[k] for k in ("root","search_size","search_factor","window")};ap["design"]=plan["actor_design"]
    for frame in range(start,limit+1):
        row=c.c64.e63.probe(observer,image_reader(plan["root"],frames[frame]),history,frame,"history",ap)
        f=c.c64.choice_features(row["features"],row["candidate_boxes"],history)
        decision=c.decide(unit["policy"],row["features"],f,pack,plan["design"]);q=decision["selected"]
        row.update(decision,choice_features=f,boxes=row["candidate_boxes"][q].copy(),parents=row["candidate_parents"][q].copy())
        rows.append(row);history.append(row["boxes"]);history=history[-plan["actor_design"]["history"]:]
        observer.rt.tracker.state=row["boxes"].tolist()
    data={k:np.asarray([r[k] for r in rows]) for k in FIELDS} if rows else {k:prefix[k][:0] for k in FIELDS}
    if prefix is not None:data={k:np.concatenate([prefix[k],data[k]]) for k in FIELDS}
    return data


def smoke_gate(plan,root,models):
    r=c.sealed(Path(root)/"smoke/stage66_smoke.json")
    if r.get("status")!="PASS_NATIVE_TAKEOVER_SMOKE" or r.get("run_fingerprint")!=run_record(plan,models)["fingerprint"] or r.get("search_calls")!=plan["smoke_calls_reused"]:raise ValueError("Need passing five-policy smoke")
    c.VERIFY(r["cache_sha256"]);return r


def execute(root,plan,source,observer,identity,smoke=False,image_reader=checked_image):
    if identity!=plan["runtime_identity"]:raise ValueError("Need bound L40 runtime")
    models=load_models(plan,root,source)
    if not smoke:smoke_gate(plan,root,models)
    c.publish_identical(Path(root)/"run.json",run_record(plan,models));hashes={};heads={};calls=0
    native_head=source["s64"]["old"]["l40"]["gate"]["observed_head_sha256"]["native_0"]
    units=[u for u in plan["units"] if not smoke or u["sequence"]==plan["entries"][0]["sequence"]]
    for u in units:
        entry=next(e for e in plan["entries"] if e["sequence"]==u["sequence"])
        limit=min(len(entry["frames"])-1,plan["design"]["smoke_frames"]) if smoke else len(entry["frames"])-1
        path=path_for(root,u,smoke);meta=c.metadata(plan,u,"smoke" if smoke else "evaluation",models);data=c.read_cache(path,meta)
        with c.c64.e63.source_experiment.installed(observer,[],u) as head:
            if head!=native_head:raise ValueError("Original native head differs")
            if data is None:
                prefix=None
                if not smoke and u["sequence"]==plan["entries"][0]["sequence"]:
                    prefix=c.read_cache(path_for(root,u,True),c.metadata(plan,u,"smoke",models))
                    if prefix is None:raise ValueError("Missing reusable smoke prefix")
                    audit(prefix,entry,u,plan,models)
                data=run_actor(observer,entry["frames"],entry["initial_box"],plan,u,models["models"][str(u["fold"])],limit,prefix,image_reader)
                audit(data,entry,u,plan,models);check_source(data,source,u);c.save(path,data,meta)
            audit(data,entry,u,plan,models);check_source(data,source,u)
        c.c64.e63.source_experiment.bind(hashes,path);heads[c.key(u)]=head;calls+=int(data["head_calls"].sum())
        print("[takeover audited] "+c.key(u),flush=True)
    observer.assert_frozen();c.VERIFY(plan["source_file_sha256"]);c.VERIFY(plan["code_sha256"])
    if calls!=(plan["smoke_calls_reused"] if smoke else plan["formal_search_calls"]):raise ValueError("Four-search budget differs")
    report=c.seal(dict(schema=c.SCHEMA,status="PASS_NATIVE_TAKEOVER_SMOKE" if smoke else "COMPLETE_NATIVE_TAKEOVER_OOF",
        plan_fingerprint=plan["fingerprint"],run_fingerprint=run_record(plan,models)["fingerprint"],search_calls=calls,
        trajectories=len(units),cache_sha256=hashes,observed_heads=heads,maximum_source_box_error=0.,sgla_training_performed=False))
    c.publish_identical(Path(root)/("smoke/stage66_smoke.json" if smoke else "evaluation/run.json"),report);return report


def completed(plan,root,source,models):
    smoke_gate(plan,root,models)
    if c.sealed(Path(root)/"run.json")!=run_record(plan,models):raise ValueError("Runtime/model record changed")
    r=c.sealed(Path(root)/"evaluation/run.json")
    expected={str(path_for(root,u).with_suffix(s).resolve()) for u in plan["units"] for s in (".npz",".receipt.json")}
    head=source["s64"]["old"]["l40"]["gate"]["observed_head_sha256"]["native_0"]
    if r.get("status")!="COMPLETE_NATIVE_TAKEOVER_OOF" or r.get("run_fingerprint")!=run_record(plan,models)["fingerprint"] or r.get("plan_fingerprint")!=plan["fingerprint"] or r.get("search_calls")!=plan["formal_search_calls"] or r.get("trajectories")!=len(plan["units"]) or set(r.get("cache_sha256",{}))!=expected or r.get("observed_heads")!={c.key(u):head for u in plan["units"]}:raise ValueError("Need complete native takeover trajectories")
    c.VERIFY(r["cache_sha256"]);return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--mode",required=True,choices=("prepare","fit","smoke","evaluate"))
    p.add_argument("--input",required=True,type=nonempty_path);p.add_argument("--output_dir",type=nonempty_path)
    p.add_argument("--checkpoint",default=os.environ.get("CHECKPOINT"));p.add_argument("--gpu",type=int,default=0);p.add_argument("--parameter",default="deit_distilled")
    args=p.parse_args()
    if args.mode=="prepare":
        if not args.output_dir:p.error("prepare requires --output_dir")
        import torch
        source=c.load_source(args.input);hardware=c.c64.prior.prior.c.prior.device_binding(source["s64"]["plan"],args.gpu,torch)
        prepare(args.input,args.output_dir,hardware);return
    root=Path(args.input).resolve();plan,source=c.load_plan(root)
    if args.mode=="fit":fit(plan,root,source);return
    if not args.checkpoint:p.error("Need original V1 CHECKPOINT")
    with writer_lock(root):
        native,identity=c.c64.e63.startup.checked_runtime(args,plan,root);observer=c.c64.e63.Capture(native)
        try:execute(root,plan,source,observer,identity,args.mode=="smoke")
        finally:observer.handle.remove()


if __name__=="__main__":main()
