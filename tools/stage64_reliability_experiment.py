"""Fit native-only selectors and evaluate complete own-history native/E trajectories."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import stage64_reliability_common as c
from stage27_uav_common import checked_image, crop_geometry, nonempty_path
from stage27_uav_collect import writer_lock
from validation_metrics import tracking_iou

e63=c.e63
VERIFY=c.prior.prior.c.prior.verify_files
FIELDS=tuple(k for k in e63.FIELDS if k not in ("raw","size","offset"))+("choice_features",)


def prepare(source,root,hardware):
    source,root=Path(source).resolve(),Path(root).resolve()
    if source==root or source in root.parents or root in source.parents: raise ValueError("Use separate Stage-64 directory")
    plan,_=c.make_plan(source,hardware)
    with writer_lock(root):
        if not (root/"manifest.json").exists() and any(p.name!="collect.lock" for p in root.iterdir()): raise ValueError("Use empty Stage-64 directory")
        c.publish_identical(root/"manifest.json",plan)
    print("Stage-64 prepared: {} complete trajectories / {} searches; no SGLA training".format(len(plan["units"]),plan["formal_search_calls"]))


def source_unit(source,unit,policy):
    old=source["l40"]["plan"]
    return next(u for u in old["units"] if all(u[k]==unit[k] for k in ("sequence","origin","seed")) and u["policy"]==policy)


def examples(plan,source,fold):
    split=next(f for f in plan["folds"] if f["fold"]==fold)
    features,labels,weights,support=[],[],[],[]
    for name in split["fit"]:
        for policy in ("history","recrop"):
            unit=next(u for u in source["l40"]["plan"]["units"] if (u["sequence"],u["origin"],u["policy"])==(name,"native",policy))
            data,entry,_,_=c.prior.prior.read_trace(source["l40"],unit)
            n=len(data["frame_id"]); gt=np.asarray(entry["ground_truth"],float)[1:]
            history=[np.asarray(entry["initial_box"],float)]
            rows=[]
            for i in range(n):
                rows.append(c.choice_features(data["features"][i],data["candidate_boxes"][i],history))
                history.append(data["boxes"][i]);history=history[-plan["actor_design"]["history"]:]
            features.append(np.asarray(rows));labels.append(tracking_iou(data["candidate_boxes"],gt[:,None]))
            weights.append(np.full(n,1/(len(split["fit"])*2*n)))
            support.append(dict(fold=fold,sequence=name,source_policy=policy,origin="native",states=n,video_policy_weight=1/(len(split["fit"])*2)))
    return np.concatenate(features),np.concatenate(labels),np.concatenate(weights),support


def fit(plan,root,source):
    models={};support=[]
    for split in plan["folds"]:
        f,y,w,rows=examples(plan,source,split["fold"]);support.extend(rows)
        models[str(split["fold"])]=dict(fit=split["fit"],held_out=split["held_out"],models={
            target:c.fit_selector(f,y,w,target,plan["design"]["ridge"]) for target in c.TARGETS})
    report=c.seal(dict(schema=c.SCHEMA,status="COMPLETE_NATIVE_ONLY_SELECTOR_FITS",plan_fingerprint=plan["fingerprint"],
        fit_origins=["native"],models=models,support=support,scorers=len(models)*2,sgla_training_performed=False))
    VERIFY(plan["source_file_sha256"]);VERIFY(plan["code_sha256"])
    with writer_lock(Path(root)/"models"):c.publish_identical(Path(root)/"models/training.json",report)
    print("Stage-64 fit: {} deterministic selectors; quality/reliability share all examples/features/weights".format(report["scorers"]))
    return report


def load_models(plan,root):
    value=c.read_sealed(Path(root)/"models/training.json")
    if value.get("status")!="COMPLETE_NATIVE_ONLY_SELECTOR_FITS" or value.get("plan_fingerprint")!=plan["fingerprint"] or value.get("fit_origins")!=["native"]:
        raise ValueError("Need unmixed native-only selector fits")
    if set(value["models"])!={str(f["fold"]) for f in plan["folds"]}:raise ValueError("Missing fold")
    for split in plan["folds"]:
        pack=value["models"][str(split["fold"])]
        if pack["fit"]!=split["fit"] or pack["held_out"]!=split["held_out"] or set(pack["models"])!=set(c.TARGETS):raise ValueError("Fit isolation changed")
        for target,m in pack["models"].items():
            if m["target"]!=target or len(m["weights"])!=len(c.FEATURES) or len(m["scale"])!=len(c.FEATURES) or not np.isfinite(m["weights"]).all() or not np.all(np.asarray(m["scale"])>0):raise ValueError("Invalid selector")
    return value


def selected_model(models,unit):
    return models["models"][str(unit["fold"])]["models"].get(unit["policy"])


def audit(data,entry,unit,plan,models):
    n=len(data["frame_id"]);e63.source_common.prior.audit_trace(data,n,4)
    if not 0<n<=len(entry["frames"])-1:raise ValueError("Invalid trajectory length")
    if np.max(data["decode_error"])>plan["actor_design"]["box_atol"]:raise ValueError("Decode error")
    history=[np.asarray(entry["initial_box"],float)]
    for i in range(n):
        parents=e63.source_common.prior.history_parents(history,plan["actor_design"])
        np.testing.assert_array_equal(data["candidate_parents"][i],parents)
        np.testing.assert_array_equal(data["history_latest"][i],history[-1])
        f=c.choice_features(data["features"][i],data["candidate_boxes"][i],history)
        np.testing.assert_array_equal(data["choice_features"][i],f)
        q,values=c.choose(unit["policy"],data["features"][i],f,selected_model(models,unit),plan["design"]["tie_epsilon"])
        if int(data["selected"][i])!=q:raise ValueError("Online selection mismatch")
        np.testing.assert_array_equal(data["scores"][i],values)
        for j,p in enumerate(parents):
            np.testing.assert_array_equal(data["crop"][i,j],crop_geometry(p,data["image_hw"][i],plan["search_factor"],plan["search_size"])[0])
        history.append(data["boxes"][i]);history=history[-plan["actor_design"]["history"]:]


def run_actor(observer,frames,initial_box,plan,unit,model,limit,prefix=None,image_reader=checked_image):
    observer.initialize(image_reader(plan["root"],frames[0]),initial_box)
    history=[np.asarray(initial_box,float)];rows=[];start=1
    if prefix is not None:
        start=len(prefix["frame_id"])+1
        for b in prefix["boxes"]:history.append(b);history=history[-plan["actor_design"]["history"]:]
    ap={k:plan[k] for k in ("root","search_size","search_factor","window")};ap["design"]=plan["actor_design"]
    for frame in range(start,limit+1):
        row=e63.probe(observer,image_reader(plan["root"],frames[frame]),history,frame,"history",ap)
        f=c.choice_features(row["features"],row["candidate_boxes"],history)
        q,scores=c.choose(unit["policy"],row["features"],f,model,plan["design"]["tie_epsilon"])
        row.update(choice_features=f,selected=q,scores=scores,boxes=row["candidate_boxes"][q].copy(),parents=row["candidate_parents"][q].copy())
        rows.append(row);history.append(row["boxes"]);history=history[-plan["actor_design"]["history"]:]
        observer.rt.tracker.state=row["boxes"].tolist()
    if rows:data={k:np.asarray([r[k] for r in rows]) for k in FIELDS}
    else:data={k:prefix[k][:0] for k in FIELDS}
    if prefix is not None:data={k:np.concatenate([prefix[k],data[k]]) for k in FIELDS}
    return data


def run_record(plan,models):return c.seal(dict(schema=c.SCHEMA,plan_fingerprint=plan["fingerprint"],training_fingerprint=models["fingerprint"],runtime=plan["runtime_identity"]))


def path_for(root,unit,smoke=False):return Path(root)/("smoke/traces" if smoke else "evaluation/traces")/(c.key(unit)+".npz")


def smoke_gate(plan,root,models):
    r=c.read_sealed(Path(root)/"smoke/stage64_smoke.json")
    if r.get("status")!="PASS_RELIABILITY_SMOKE" or r.get("run_fingerprint")!=run_record(plan,models)["fingerprint"] or r.get("search_calls")!=plan["smoke_calls_reused"]:raise ValueError("Need passing matched-control smoke")
    VERIFY(r["cache_sha256"]);return r


def execute(root,plan,source,observer,records,identity,smoke=False,image_reader=checked_image):
    if identity!=plan["runtime_identity"]:raise ValueError("Need bound L40 runtime")
    models=load_models(plan,root)
    if not smoke:smoke_gate(plan,root,models)
    c.publish_identical(Path(root)/"run.json",run_record(plan,models))
    bindings={};heads={};calls=0
    units=[u for u in plan["units"] if not smoke or u["sequence"]==plan["entries"][0]["sequence"]]
    for unit in units:
        entry=next(e for e in plan["entries"] if e["sequence"]==unit["sequence"])
        n=min(len(entry["frames"])-1,plan["design"]["smoke_frames"]) if smoke else len(entry["frames"])-1
        path=path_for(root,unit,smoke);meta=c.meta(plan,unit,"smoke" if smoke else "evaluation",models)
        data=c.read_cache(path,meta)
        with e63.source_experiment.installed(observer,records,unit) as head:
            if data is None:
                prefix=None
                if not smoke and unit["sequence"]==plan["entries"][0]["sequence"]:
                    prefix=c.read_cache(path_for(root,unit,True),c.meta(plan,unit,"smoke",models))
                    if prefix is None:raise ValueError("Missing reusable smoke prefix")
                    audit(prefix,entry,unit,plan,models)
                data=run_actor(observer,entry["frames"],entry["initial_box"],plan,unit,selected_model(models,unit),n,prefix,image_reader)
                audit(data,entry,unit,plan,models)
                # Before committing controls, verify the original source output/causal path.
                check_source(data,source,unit)
                c.save(path,data,meta)
            audit(data,entry,unit,plan,models);check_source(data,source,unit)
        e63.source_experiment.bind(bindings,path);heads[c.key(unit)]=head;calls+=int(data["head_calls"].sum())
        print("[reliability audited] "+c.key(unit),flush=True)
    observer.assert_frozen();VERIFY(plan["source_file_sha256"]);VERIFY(plan["code_sha256"])
    expected=plan["smoke_calls_reused"] if smoke else plan["formal_search_calls"]
    if calls!=expected:raise ValueError("Exact four-pass budget mismatch")
    r=c.seal(dict(schema=c.SCHEMA,status="PASS_RELIABILITY_SMOKE" if smoke else "COMPLETE_RELIABILITY_OOF",
        run_fingerprint=run_record(plan,models)["fingerprint"],plan_fingerprint=plan["fingerprint"],search_calls=calls,
        trajectories=len(units),cache_sha256=bindings,observed_heads=heads,maximum_source_box_error=0.,sgla_training_performed=False))
    c.publish_identical(Path(root)/("smoke/stage64_smoke.json" if smoke else "evaluation/run.json"),r)
    return r


def check_source(data,source,unit):
    if unit["policy"] not in ("hann","q0"):return
    policy="history" if unit["policy"]=="hann" else "current"
    ref,_,_,_=c.prior.prior.read_trace(source["l40"],source_unit(source,unit,policy))
    if policy=="history":e63.exact_source(data,ref,1,"Stage-64 Hann control")
    else:np.testing.assert_array_equal(data["boxes"],ref["boxes"][:len(data["boxes"])])


def completed(plan,root,models):
    smoke_gate(plan,root,models)
    if c.read_sealed(Path(root)/"run.json")!=run_record(plan,models):raise ValueError("Runtime/model record changed")
    r=c.read_sealed(Path(root)/"evaluation/run.json")
    expected={str(path_for(root,u).with_suffix(s).resolve()) for u in plan["units"] for s in (".npz",".receipt.json")}
    if r.get("status")!="COMPLETE_RELIABILITY_OOF" or r.get("plan_fingerprint")!=plan["fingerprint"] or r.get("run_fingerprint")!=run_record(plan,models)["fingerprint"] or r.get("search_calls")!=plan["formal_search_calls"] or r.get("trajectories")!=len(plan["units"]) or set(r.get("cache_sha256",{}))!=expected:raise ValueError("Need all complete Stage-64 trajectories")
    VERIFY(r["cache_sha256"]);return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--mode",required=True,choices=("prepare","fit","smoke","evaluate"))
    p.add_argument("--input",required=True,type=nonempty_path);p.add_argument("--output_dir",type=nonempty_path)
    p.add_argument("--checkpoint",default=os.environ.get("CHECKPOINT"));p.add_argument("--gpu",type=int,default=0);p.add_argument("--parameter",default="deit_distilled")
    args=p.parse_args()
    if args.mode=="prepare":
        if not args.output_dir:p.error("prepare requires --output_dir")
        import torch
        source=c.load_source(args.input);hardware=c.prior.prior.c.prior.device_binding(source["l40"]["plan"],args.gpu,torch)
        prepare(args.input,args.output_dir,hardware);return
    root=Path(args.input).resolve();plan,source=c.load_plan(root)
    if args.mode=="fit":fit(plan,root,source);return
    if not args.checkpoint:p.error("Need original V1 CHECKPOINT")
    with writer_lock(root):
        native,identity=e63.startup.checked_runtime(args,plan,root)
        records=e63.load_records(source["inputs62"],native.rt.torch);observer=e63.Capture(native)
        try:execute(root,plan,source,observer,records,identity,args.mode=="smoke")
        finally:observer.handle.remove()


if __name__=="__main__":main()
