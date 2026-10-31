"""Prepare/smoke, fit-only branch collection, ridge fitting and causal OOF inference."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import numpy as np
from stage27_uav_common import checked_image, nonempty_path
from stage27_uav_collect import writer_lock
from stage47_search_collect import Capture
from stage48_stability_runtime import digest
from stage54_rollout_runtime import temporary_head
import stage60_state_common as c
import stage60_state_runtime as actor

def prepare(source, root):
    source,root=Path(source).expanduser().resolve(),Path(root).expanduser().resolve()
    if source==root or source in root.parents or root in source.parents: raise ValueError("Use separate Stage-60 output directory")
    plan,_=c.make_plan(source)
    with writer_lock(root):
        if not (root/"manifest.json").exists() and any(p.name!="collect.lock" for p in root.iterdir()): raise ValueError("Use empty Stage-60 output directory")
        c.publish_identical(root/"manifest.json",plan)
    print("Stage-60: {} fit states, {} branch search calls, {} OOF trajectories / {} evaluation search calls".format(
        plan["expected_fit_anchors"],plan["expected_branch_calls"],plan["expected_evaluation_trajectories"],plan["expected_evaluation_calls"]))

def run_record(plan):
    return c.seal(dict(schema=c.SCHEMA,role="causal_self_history_oof",plan_fingerprint=plan["fingerprint"],runtime=plan["runtime_identity"]))

def smoke_gate(plan, root):
    value=c.read_sealed(Path(root)/"smoke/stage60_smoke.json")
    if value.get("status")!="PASS_CAUSAL_STATE_SMOKE" or value.get("run_fingerprint")!=run_record(plan)["fingerprint"]:
        raise ValueError("Need passing Stage-60 smoke")
    c.verify_files(value["cache_sha256"])
    return value

def record_for(records, fold, seed):
    selected=[(r,s) for r,s in records if (r["fold"],r["seed"],r["arm"])==(fold,seed,"endpoint_control")]
    if len(selected)!=1: raise ValueError("Need the exact endpoint-control head")
    return selected[0]

@contextmanager
def installed(observer, record, state):
    with temporary_head(observer.native,state):
        if digest(observer.rt.net.box_head)!=record["final_head_sha256"]: raise ValueError("Wrong frozen SGLA head")
        yield
        if digest(observer.rt.net.box_head)!=record["final_head_sha256"]: raise ValueError("Inference changed SGLA head")
    observer.assert_frozen()

def bind(bindings, path):
    for p in (path,path.with_suffix(".receipt.json")): bindings[str(p.resolve())]=c.file_hash(p)

def fit_base(plan, root, observer, unit, record, image_reader=checked_image):
    entry=next(e for e in plan["entries"] if e["sequence"]==unit["sequence"])
    if unit["sequence"] not in plan["folds"][unit["fold"]]["fit"]: raise ValueError("Held-out fit trajectory requested")
    path=Path(root)/"fit/traces"/(c.key(unit)+".npz")
    metadata=c.meta(plan,"fit_baseline",**unit,head_sha256=record["final_head_sha256"])
    data=c.read_cache(path,metadata)
    if data is None:
        data=actor.run_actor(observer,entry["frames"],entry["initial_box"],plan["root"],"current",{},plan,image_reader=image_reader)
        c.save(path,data,metadata)
    c.audit_trace(data,len(entry["frames"])-1,1)
    audit_evaluation(data,entry,plan,"current",{},None)
    return path,data

def fit_anchor(plan, root, observer, unit, record, baseline, frame, image_reader=checked_image):
    entry=next(e for e in plan["entries"] if e["sequence"]==unit["sequence"])
    if unit["sequence"] not in plan["folds"][unit["fold"]]["fit"] or frame not in c.anchors(entry,plan["design"]):
        raise ValueError("Held-out or nonuniform fit anchor requested")
    path=Path(root)/"fit/anchors"/(c.key(unit)+"__t{:06d}.npz".format(frame))
    metadata=c.meta(plan,"fit_anchor",**unit,frame=frame,head_sha256=record["final_head_sha256"],
                    baseline_sha256=c.file_hash(Path(root)/"fit/traces"/(c.key(unit)+".npz")))
    data=c.read_cache(path,metadata)
    if data is None:
        data=actor.collect_anchor(observer,entry,baseline,frame,plan,image_reader=image_reader)
        c.save(path,data,metadata)
    actor.audit_anchor(data,entry,baseline,frame,plan)
    return path,data

def collect_fit(plan, root, observer, records, image_reader=checked_image):
    smoke_gate(plan,root); bindings={}; states=0; calls=0
    for unit in plan["fit_units"]:
        record,state=record_for(records,unit["fold"],unit["seed"])
        with installed(observer,record,state):
            path,baseline=fit_base(plan,root,observer,unit,record,image_reader); bind(bindings,path)
            calls+=int(baseline["head_calls"].sum())
            entry=next(e for e in plan["entries"] if e["sequence"]==unit["sequence"])
            for frame in c.anchors(entry,plan["design"]):
                path,data=fit_anchor(plan,root,observer,unit,record,baseline,frame,image_reader); bind(bindings,path)
                states+=1; calls+=int(data["head_calls"].sum())
        print("[fit states audited] "+c.key(unit),flush=True)
    if (states,calls)!=(plan["expected_fit_anchors"],plan["expected_fit_frames"]+plan["expected_branch_calls"]):
        raise ValueError("Fit collection budget changed")
    c.verify_files(plan["source_file_sha256"])
    value=c.seal(dict(schema=c.SCHEMA,status="COMPLETE_FIT_STATE_BRANCHES",plan_fingerprint=plan["fingerprint"],
        states=states,full_search_calls=calls,block_calls=calls*7,cache_sha256=bindings,held_out_used_for_fitting=False))
    c.publish_identical(Path(root)/"fit/collection.json",value)
    return value

def collection_gate(plan, root):
    value=c.read_sealed(Path(root)/"fit/collection.json")
    if (value.get("plan_fingerprint")!=plan["fingerprint"] or value.get("status")!="COMPLETE_FIT_STATE_BRANCHES" or
        value.get("states")!=plan["expected_fit_anchors"] or
        value.get("full_search_calls")!=plan["expected_fit_frames"]+plan["expected_branch_calls"] or value.get("held_out_used_for_fitting") is not False):
        raise ValueError("Need complete fit-only state collection")
    c.verify_files(value["cache_sha256"])
    return value

def examples(plan, root, fold, seed, record, limit_units=None, limit_anchors=None):
    values=[]; bindings={}
    units=[u for u in plan["fit_units"] if (u["fold"],u["seed"])==(fold,seed)]
    if limit_units is not None: units=units[:limit_units]
    for unit in units:
        entry=next(e for e in plan["entries"] if e["sequence"]==unit["sequence"])
        if unit["sequence"] in plan["folds"][fold]["held_out"]: raise ValueError("Held-out example entered fit")
        basepath=Path(root)/"fit/traces"/(c.key(unit)+".npz")
        baseline=c.read_cache(basepath,c.meta(plan,"fit_baseline",**unit,head_sha256=record["final_head_sha256"]))
        if baseline is None: raise ValueError("Missing fit baseline")
        c.audit_trace(baseline,len(entry["frames"])-1,1); bind(bindings,basepath)
        frames=c.anchors(entry,plan["design"])
        if limit_anchors is not None: frames=frames[:limit_anchors]
        for frame in frames:
            path=Path(root)/"fit/anchors"/(c.key(unit)+"__t{:06d}.npz".format(frame))
            data=c.read_cache(path,c.meta(plan,"fit_anchor",**unit,frame=frame,head_sha256=record["final_head_sha256"],baseline_sha256=c.file_hash(basepath)))
            if data is None: raise ValueError("Missing fit branch")
            actor.audit_anchor(data,entry,baseline,frame,plan); bind(bindings,path); values.append(data)
    if not values: raise ValueError("No fit-only examples")
    return values,bindings

def model_body(plan, fold, seed, record, values, bindings, smoke=False):
    models={target:c.fit_value(np.array([d["features"] for d in values]),
        np.array([d["immediate_returns" if target=="immediate_value" else "future_returns"] for d in values]),plan["design"]) for target in c.TARGETS}
    return c.seal(dict(schema=c.SCHEMA,role="smoke_only" if smoke else "fit_only_ridge_action_value",plan_fingerprint=plan["fingerprint"],
        fold=fold,seed=seed,head_sha256=record["final_head_sha256"],fit=plan["folds"][fold]["fit"],held_out=plan["folds"][fold]["held_out"],
        features=list(c.FEATURES),example_sha256=bindings,models=models))

def fit_models(plan, root, records):
    collection_gate(plan,root); paths={}
    for fold in plan["folds"]:
        for seed in plan["seeds"]:
            record,_=record_for(records,fold["fold"],seed)
            values,bindings=examples(plan,root,fold["fold"],seed,record)
            model=model_body(plan,fold["fold"],seed,record,values,bindings)
            path=Path(root)/"models"/("fold{}__seed{}.json".format(fold["fold"],seed))
            c.publish_identical(path,model); paths[str(path.resolve())]=c.file_hash(path)
    report=c.seal(dict(schema=c.SCHEMA,status="COMPLETE_FIT_ONLY_SCORERS",plan_fingerprint=plan["fingerprint"],model_sha256=paths,
                       new_scorers=len(paths)*2,sgla_weights_updated=False))
    c.publish_identical(Path(root)/"models/training.json",report)
    return report

def load_models(plan, root, records, recompute=True):
    collection_gate(plan,root)
    report=c.read_sealed(Path(root)/"models/training.json")
    if report.get("plan_fingerprint")!=plan["fingerprint"] or report.get("status")!="COMPLETE_FIT_ONLY_SCORERS":
        raise ValueError("Need fitted Stage-60 scorers")
    expected={str((Path(root)/"models"/("fold{}__seed{}.json".format(f["fold"],s))).resolve()) for f in plan["folds"] for s in plan["seeds"]}
    if set(report["model_sha256"])!=expected: raise ValueError("Mixed scorer set")
    c.verify_files(report["model_sha256"]); result={}
    for f in plan["folds"]:
        for s in plan["seeds"]:
            path=Path(root)/"models"/("fold{}__seed{}.json".format(f["fold"],s)); model=c.read_sealed(path)
            record,_=record_for(records,f["fold"],s)
            if recompute:
                values,bindings=examples(plan,root,f["fold"],s,record)
                if model!=model_body(plan,f["fold"],s,record,values,bindings): raise ValueError("Scorer no longer matches fit data")
            result[f["fold"],s]=model
    return result,report

def audit_evaluation(data, entry, plan, policy, models, reference):
    n=len(data["frame_id"]); c.audit_trace(data,n,1 if policy=="current" else 4)
    history=[np.asarray(entry["initial_box"],float)]
    for i in range(n):
        expected=c.history_parents(history,plan["design"])
        if policy=="current": expected=expected[:1]
        np.testing.assert_array_equal(data["candidate_parents"][i],expected)
        np.testing.assert_array_equal(data["history_latest"][i],history[-1])
        choice,scores=c.choose(policy,data["features"][i],models,plan["design"])
        if choice!=int(data["selected"][i]): raise ValueError("Selection no longer matches causal scorer")
        np.testing.assert_allclose(data["scores"][i],scores[:len(expected)],atol=1e-12,rtol=0)
        for j,p in enumerate(expected):
            crop=c.crop_geometry(p,data["image_hw"][i],plan["search_factor"],plan["search_size"])[0]
            np.testing.assert_array_equal(data["crop"][i,j],crop)
        history.append(data["boxes"][i]); history=history[-plan["design"]["history"]:]
    error=0.
    if policy=="current" and reference is not None:
        error=float(np.max(np.abs(data["boxes"]-reference["boxes"][:n])))
        if error>plan["design"]["source_box_atol"]: raise ValueError("Current policy did not exactly reproduce source endpoint control")
        np.testing.assert_array_equal(data["layer"][:,0],reference["layer"][:n])
        if np.max(np.abs(data["features"][:,0,c.FEATURES.index("raw_peak")]-reference["peak"][:n]))>1e-5:
            raise ValueError("Current policy response differs from source")
    if np.max(data["decode_error"])>plan["design"]["box_atol"]: raise ValueError("Decode audit failed")
    return error

def evaluate(plan, root, source, observer, records, image_reader=checked_image):
    smoke_gate(plan,root); models,training=load_models(plan,root,records); maximum=0.; calls=0; hashes={}; count=0
    for entry in plan["entries"]:
        fold=next(f["fold"] for f in plan["folds"] if entry["sequence"] in f["held_out"])
        for seed in plan["seeds"]:
            record,state=record_for(records,fold,seed); model=models[fold,seed]
            with installed(observer,record,state):
                for policy in c.POLICIES:
                    path=Path(root)/"evaluation/traces"/(entry["sequence"]+"__s{}__{}.npz".format(seed,policy))
                    metadata=c.meta(plan,"oof",sequence=entry["sequence"],fold=fold,seed=seed,policy=policy,model_fingerprint=model["fingerprint"])
                    data=c.read_cache(path,metadata)
                    if data is None:
                        data=actor.run_actor(observer,entry["frames"],entry["initial_box"],plan["root"],policy,model["models"],plan,image_reader=image_reader)
                        c.save(path,data,metadata)
                    if len(data["frame_id"])!=len(entry["frames"])-1: raise ValueError("Truncated formal OOF trace")
                    reference=source["arrays"][entry["sequence"]][1]["endpoint_control_seed"+str(seed)]
                    maximum=max(maximum,audit_evaluation(data,entry,plan,policy,model["models"],reference))
                    calls+=int(data["head_calls"].sum()); bind(hashes,path); count+=1
                    print("[OOF audited] {} seed{} {}".format(entry["sequence"],seed,policy),flush=True)
    if (calls,count)!=(plan["expected_evaluation_calls"],plan["expected_evaluation_trajectories"]): raise ValueError("OOF call budget changed")
    c.verify_files(plan["source_file_sha256"])
    value=c.seal(dict(schema=c.SCHEMA,status="COMPLETE_CAUSAL_OOF",plan_fingerprint=plan["fingerprint"],
        run_fingerprint=run_record(plan)["fingerprint"],training_fingerprint=training["fingerprint"],full_search_calls=calls,
        full_trajectories=count,maximum_control_box_error=maximum,cache_sha256=hashes))
    c.publish_identical(Path(root)/"evaluation/run.json",value)
    return value

def smoke(plan, root, source, observer, records, image_reader=checked_image):
    unit=plan["fit_units"][0]; record,state=record_for(records,unit["fold"],unit["seed"]); bindings={}
    with installed(observer,record,state):
        path,baseline=fit_base(plan,root,observer,unit,record,image_reader); bind(bindings,path)
        entry=next(e for e in plan["entries"] if e["sequence"]==unit["sequence"])
        for frame in c.anchors(entry,plan["design"])[:plan["design"]["smoke_anchors"]]:
            path,_=fit_anchor(plan,root,observer,unit,record,baseline,frame,image_reader); bind(bindings,path)
        values,example_bindings=examples(plan,root,unit["fold"],unit["seed"],record,1,plan["design"]["smoke_anchors"])
        model=model_body(plan,unit["fold"],unit["seed"],record,values,example_bindings,True)
        entry=next(e for e in plan["entries"] if e["sequence"]==plan["folds"][unit["fold"]]["held_out"][0])
        limit=min(plan["design"]["smoke_frames"],len(entry["frames"])-1); calls=0; maximum=0.
        for policy in c.POLICIES:
            path=Path(root)/"smoke"/(policy+".npz"); metadata=c.meta(plan,"smoke",policy=policy,model_fingerprint=model["fingerprint"])
            data=c.read_cache(path,metadata)
            if data is None:
                data=actor.run_actor(observer,entry["frames"],entry["initial_box"],plan["root"],policy,model["models"],plan,limit,image_reader)
                c.save(path,data,metadata)
            reference=source["arrays"][entry["sequence"]][1]["endpoint_control_seed"+str(unit["seed"])]
            maximum=max(maximum,audit_evaluation(data,entry,plan,policy,model["models"],reference))
            if len(data["frame_id"])!=limit: raise ValueError("Incomplete smoke")
            calls+=int(data["head_calls"].sum()); bind(bindings,path)
    if calls!=limit*13: raise ValueError("Smoke budget changed")
    c.publish_identical(Path(root)/"smoke/model.json",model)
    bindings[str((Path(root)/"smoke/model.json").resolve())]=c.file_hash(Path(root)/"smoke/model.json")
    c.verify_files(plan["source_file_sha256"])
    value=c.seal(dict(schema=c.SCHEMA,status="PASS_CAUSAL_STATE_SMOKE",run_fingerprint=run_record(plan)["fingerprint"],
        maximum_control_box_error=maximum,extra_smoke_search_calls=calls,cache_sha256=bindings))
    c.publish_identical(Path(root)/"smoke/stage60_smoke.json",value)
    print("Stage-60: PASS_CAUSAL_STATE_SMOKE; fit caches reused; smoke-only scorers never enter formal evaluation")

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--mode",required=True,choices=("prepare","smoke","collect","fit","evaluate"))
    p.add_argument("--input",required=True,type=nonempty_path); p.add_argument("--output_dir",type=nonempty_path)
    p.add_argument("--checkpoint",default=os.environ.get("CHECKPOINT")); p.add_argument("--gpu",type=int,default=0); p.add_argument("--parameter",default="deit_distilled")
    args=p.parse_args()
    if args.mode=="prepare":
        if not args.output_dir: p.error("prepare needs --output_dir")
        prepare(args.input,args.output_dir); return
    root=Path(args.input).expanduser().resolve(); plan,source=c.load_plan(root)
    if args.mode=="fit":
        with writer_lock(root): fit_models(plan,root,[(r,None) for r in source["records"]])
        return
    if not args.checkpoint: p.error("Set original SGLA CHECKPOINT or --checkpoint")
    from stage46_motion_experiment import make_runtime
    with writer_lock(root):
        native,identity=make_runtime(args,dict(runtime=plan["runtime"]))
        if identity!=plan["runtime_identity"]: raise ValueError("Original runtime changed")
        records,_=c.prior.previous.model_records(source["plan"],source["root"],native.rt.torch)
        observer=Capture(native)
        try:
            if args.mode=="smoke": smoke(plan,root,source,observer,records)
            elif args.mode=="collect": collect_fit(plan,root,observer,records)
            else: evaluate(plan,root,source,observer,records)
        finally: observer.handle.remove()

if __name__=="__main__": main()
