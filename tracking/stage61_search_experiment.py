"""Prepare, verify source replay, then evaluate a fixed original-V1 recrop control."""
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
import stage61_search_common as c
import stage61_search_runtime as actor

def prepare(source,root):
    source,root=Path(source).expanduser().resolve(),Path(root).expanduser().resolve()
    if source==root or source in root.parents or root in source.parents: raise ValueError("Use a separate Stage-61 output directory")
    plan,_=c.make_plan(source)
    with writer_lock(root):
        if not (root/"manifest.json").exists() and any(p.name!="collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-61 directory")
        c.publish_identical(root/"manifest.json",plan)
    print("Stage-61: {} new trajectories / {} searches; {} reused trajectories; no training".format(
        plan["new_full_trajectories"],plan["formal_search_calls"],plan["reused_full_trajectories"]))

def run_record(plan):
    return c.seal(dict(schema=c.SCHEMA,role="frozen_v1_history_recrop",plan_fingerprint=plan["fingerprint"],runtime=plan["runtime_identity"]))

def bind(hashes,path):
    for p in (path,path.with_suffix(".receipt.json")): hashes[str(p.resolve())]=c.file_hash(p)

def model_identity(plan,records,unit):
    if unit["origin"]=="native": return "original:"+plan["runtime_identity"]["model_state_sha256"]
    record,_=c.previous.record_for(records,unit["fold"],unit["seed"])
    return "endpoint:"+record["final_head_sha256"]

@contextmanager
def installed(observer,records,unit):
    if unit["origin"]=="endpoint":
        record,state=c.previous.record_for(records,unit["fold"],unit["seed"])
        with c.previous.installed(observer,record,state): yield record["final_head_sha256"]
    elif unit["origin"]=="native":
        observer.assert_frozen(); original=digest(observer.rt.net.box_head)
        yield original
        if digest(observer.rt.net.box_head)!=original: raise ValueError("Native inference changed the head")
        observer.assert_frozen()
    else: raise ValueError("Unknown base network origin")

def trace(plan,root,observer,records,unit,smoke=False,reference=None,image_reader=checked_image):
    entry=next(e for e in plan["entries"] if e["sequence"]==unit["sequence"])
    limit=min(len(entry["frames"])-1,plan["design"]["smoke_frames"]) if smoke else len(entry["frames"])-1
    path=Path(root)/("smoke/traces" if smoke else "evaluation/traces")/(c.key(unit)+".npz")
    metadata=dict(c.meta(plan,unit,"smoke" if smoke else "full_evaluation"),model_identity=model_identity(plan,records,unit))
    data=c.read_cache(path,metadata)
    if data is None:
        data=actor.run_actor(observer,entry["frames"],entry["initial_box"],plan["root"],unit["policy"],plan,limit,image_reader)
        actor.audit(data,entry,plan,unit["policy"],reference)
        c.save(path,data,metadata)
    if len(data["frame_id"])!=limit: raise ValueError("Incomplete trajectory")
    error=actor.audit(data,entry,plan,unit["policy"],reference)
    return path,data,error

def smoke(plan,root,source,observer,records,image_reader=checked_image):
    c.publish_identical(Path(root)/"run.json",run_record(plan))
    entry=plan["entries"][0]; name=entry["sequence"]
    fold=next(f["fold"] for f in plan["folds"] if name in f["held_out"])
    bases=[dict(sequence=name,origin="native",seed=0,fold=-1)]
    bases += [dict(sequence=name,origin="endpoint",seed=s,fold=fold) for s in plan["seeds"]]
    hashes={}; calls=0; maximum=0.; decode=0.; heads={}
    for base in bases:
        with installed(observer,records,base) as head:
            heads[base["origin"]+"_"+str(base["seed"])]=head
            for policy in c.POLICIES:
                unit=dict(base,policy=policy)
                reference=source["reused"].get((name,base["origin"],policy,base["seed"]))
                path,data,error=trace(plan,root,observer,records,unit,True,reference,image_reader)
                bind(hashes,path); maximum=max(maximum,error); decode=max(decode,float(data["decode_error"].max()))
                calls+=int(data["head_calls"].sum())
    if calls!=plan["smoke_search_calls"]: raise ValueError("Smoke call budget changed")
    c.verify_files(plan["source_file_sha256"])
    value=c.seal(dict(schema=c.SCHEMA,status="PASS_V1_MATCHED_RECROP_SMOKE",role="execution_only_not_method_gain",
        plan_fingerprint=plan["fingerprint"],run_fingerprint=run_record(plan)["fingerprint"],
        maximum_source_box_error=maximum,maximum_decode_error=decode,extra_smoke_search_calls=calls,
        observed_head_sha256=heads,cache_sha256=hashes,training_performed=False))
    c.publish_identical(Path(root)/"smoke/stage61_smoke.json",value)
    print("Stage-61: PASS_V1_MATCHED_RECROP_SMOKE; original controls and both E history prefixes reproduce sources")
    return value

def smoke_gate(plan,root):
    value=c.read_sealed(Path(root)/"smoke/stage61_smoke.json")
    if (value.get("status")!="PASS_V1_MATCHED_RECROP_SMOKE" or value.get("plan_fingerprint")!=plan["fingerprint"] or
        value.get("run_fingerprint")!=run_record(plan)["fingerprint"] or
        value.get("extra_smoke_search_calls")!=plan["smoke_search_calls"] or value.get("maximum_source_box_error")!=0 or
        not 0<=value.get("maximum_decode_error",float("inf"))<=plan["design"]["box_atol"]):
        raise ValueError("Need passing Stage-61 smoke for exact runtime and source")
    c.verify_files(value["cache_sha256"])
    return value

def evaluate(plan,root,source,observer,records,image_reader=checked_image):
    smoke_gate(plan,root); c.publish_identical(Path(root)/"run.json",run_record(plan))
    hashes={}; heads={}; calls=0; count=0; maximum_decode=0.
    for unit in plan["units"]:
        with installed(observer,records,unit) as head:
            path,data,_=trace(plan,root,observer,records,unit,image_reader=image_reader)
        bind(hashes,path); heads[c.key(unit)]=head; calls+=int(data["head_calls"].sum()); count+=1
        maximum_decode=max(maximum_decode,float(data["decode_error"].max()))
        print("[audited] "+c.key(unit),flush=True)
    if (calls,count)!=(plan["formal_search_calls"],plan["new_full_trajectories"]):
        raise ValueError("Formal trajectory/call budget changed")
    c.verify_files(plan["source_file_sha256"])
    value=c.seal(dict(schema=c.SCHEMA,status="COMPLETE_V1_MATCHED_RECROP_EVALUATION",plan_fingerprint=plan["fingerprint"],
        run_fingerprint=run_record(plan)["fingerprint"],new_full_trajectories=count,formal_search_calls=calls,
        maximum_decode_error=maximum_decode,observed_head_sha256=heads,cache_sha256=hashes,training_performed=False))
    c.publish_identical(Path(root)/"evaluation/run.json",value)
    return value

def completed(plan,root):
    smoke_gate(plan,root)
    if c.read_sealed(Path(root)/"run.json")!=run_record(plan): raise ValueError("Runtime record changed")
    run=c.read_sealed(Path(root)/"evaluation/run.json")
    expected={str((Path(root)/"evaluation/traces"/(c.key(u)+suffix)).resolve()) for u in plan["units"] for suffix in (".npz",".receipt.json")}
    if (run.get("status")!="COMPLETE_V1_MATCHED_RECROP_EVALUATION" or
        run.get("plan_fingerprint")!=plan["fingerprint"] or run.get("run_fingerprint")!=run_record(plan)["fingerprint"] or
        run.get("new_full_trajectories")!=plan["new_full_trajectories"] or run.get("formal_search_calls")!=plan["formal_search_calls"] or
        set(run.get("cache_sha256",{}))!=expected or set(run.get("observed_head_sha256",{}))!={c.key(u) for u in plan["units"]}):
        raise ValueError("Need complete unmixed Stage-61 evaluation")
    c.verify_files(run["cache_sha256"])
    return run

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",required=True,choices=("prepare","smoke","evaluate"))
    p.add_argument("--input",required=True,type=nonempty_path); p.add_argument("--output_dir",type=nonempty_path)
    p.add_argument("--checkpoint",default=os.environ.get("CHECKPOINT")); p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--parameter",default="deit_distilled"); args=p.parse_args()
    if args.mode=="prepare":
        if not args.output_dir: p.error("prepare requires --output_dir")
        prepare(args.input,args.output_dir); return
    if not args.checkpoint: p.error("Set original V1 CHECKPOINT or --checkpoint")
    root=Path(args.input).expanduser().resolve(); plan,source=c.load_plan(root)
    from stage46_motion_experiment import make_runtime
    with writer_lock(root):
        native,identity=make_runtime(args,dict(runtime=plan["runtime"]))
        if identity!=plan["runtime_identity"]: raise ValueError("Need exact original V1 runtime")
        records,_=c.prior.prior.previous.model_records(source["old"]["plan"],source["old"]["root"],native.rt.torch)
        observer=Capture(native)
        try:
            if args.mode=="smoke": smoke(plan,root,source,observer,records)
            else: evaluate(plan,root,source,observer,records)
        finally: observer.handle.remove()

if __name__=="__main__": main()
