"""Prepare L40 hardware binding, smoke public current controls, then run all six arms."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import stage61_l40_common as c
import stage61_runtime_replay as startup
import stage61_search_experiment as original
import stage61_search_runtime as actor
from stage27_uav_common import checked_image, nonempty_path
from stage27_uav_collect import writer_lock
from stage47_search_collect import Capture

installed=original.installed
model_identity=original.model_identity
bind=original.bind


def prepare(source,root,hardware):
    source,root=Path(source).expanduser().resolve(),Path(root).expanduser().resolve()
    if source==root or source in root.parents or root in source.parents: raise ValueError("Use a separate L40 directory")
    plan,_=c.make_plan(source,hardware)
    with writer_lock(root):
        if not (root/"manifest.json").exists() and any(p.name!="collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty L40 directory")
        c.publish_identical(root/"manifest.json",plan)
    print("Stage-61 L40: {} new full trajectories / {} formal searches + {} smoke searches; no training".format(
        plan["new_full_trajectories"],plan["formal_search_calls"],plan["smoke_search_calls"]))


def run_record(plan):
    return c.seal(dict(schema=c.SCHEMA,role="same_l40_all_controls_rerun",plan_fingerprint=plan["fingerprint"],
                       runtime=plan["runtime_identity"],source_runtime=plan["source_runtime_identity"]))


def trace(plan,root,observer,records,unit,smoke=False,image_reader=checked_image):
    entry=next(e for e in plan["entries"] if e["sequence"]==unit["sequence"])
    limit=min(len(entry["frames"])-1,plan["design"]["smoke_frames"]) if smoke else len(entry["frames"])-1
    path=Path(root)/("smoke/traces" if smoke else "evaluation/traces")/(c.key(unit)+".npz")
    metadata=dict(c.meta(plan,unit,"smoke" if smoke else "full_evaluation"),model_identity=model_identity(plan,records,unit))
    data=c.read_cache(path,metadata)
    if data is None:
        data=actor.run_actor(observer,entry["frames"],entry["initial_box"],plan["root"],unit["policy"],plan,limit,image_reader)
        actor.audit(data,entry,plan,unit["policy"])
        c.save(path,data,metadata)
    if len(data["frame_id"])!=limit: raise ValueError("Incomplete L40 trajectory")
    actor.audit(data,entry,plan,unit["policy"])
    return path,data


def public_prefix(plan,root,observer,records,unit,image_reader=checked_image):
    """Independent public tracker.step path; no candidate proposal/selection helper."""
    entry=next(e for e in plan["entries"] if e["sequence"]==unit["sequence"])
    limit=min(len(entry["frames"])-1,plan["design"]["smoke_frames"])
    path=Path(root)/"smoke/public"/(c.key(unit)+".npz")
    metadata=dict(c.meta(plan,unit,"independent_public_current"),model_identity=model_identity(plan,records,unit))
    data=c.read_cache(path,metadata)
    fresh=data is None
    if fresh:
        observer.native.initialize(image_reader(plan["root"],entry["frames"][0]),entry["initial_box"])
        parent=np.asarray(entry["initial_box"],float).copy(); rows=[]
        for frame in range(1,limit+1):
            result=observer.native.step(image_reader(plan["root"],entry["frames"][frame]),parent)
            rows.append(dict(frame_id=frame,boxes=result["box"],layer=result["layer"],
                head_calls=result["head_calls"],block_calls=result["block_calls"],selector_calls=result["selector_calls"]))
            parent=np.asarray(result["box"],float).copy()
        data={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    np.testing.assert_array_equal(data["frame_id"],np.arange(1,limit+1))
    np.testing.assert_array_equal(data["head_calls"],np.ones(limit,int))
    np.testing.assert_array_equal(data["selector_calls"],np.ones(limit,int))
    if not np.isfinite(data["boxes"]).all() or not np.all(data["block_calls"].sum(1)==7):
        raise ValueError("Public L40 control budget/boxes invalid")
    if fresh: c.save(path,data,metadata)
    return path,data


def smoke(plan,root,source,observer,records,image_reader=checked_image):
    c.publish_identical(Path(root)/"run.json",run_record(plan))
    name=plan["entries"][0]["sequence"]
    bases=[u for u in plan["units"] if u["sequence"]==name and u["policy"]=="current"]
    hashes={}; calls=0; maximum=0.; decode=0.; heads={}; source_drift=[]
    for base in bases:
        with installed(observer,records,base) as head:
            heads[base["origin"]+"_"+str(base["seed"])]=head
            public_path,public=public_prefix(plan,root,observer,records,base,image_reader)
            bind(hashes,public_path); calls+=int(public["head_calls"].sum())
            for policy in c.POLICIES:
                unit=dict(base,policy=policy)
                path,data=trace(plan,root,observer,records,unit,True,image_reader)
                bind(hashes,path); calls+=int(data["head_calls"].sum())
                decode=max(decode,float(data["decode_error"].max()))
                if policy=="current":
                    error=float(np.max(np.abs(data["boxes"]-public["boxes"])))
                    maximum=max(maximum,error)
                    if error!=0: raise ValueError("L40 current path differs from independent L40 public replay")
                    np.testing.assert_array_equal(data["layer"][:,0],public["layer"])
                old=source["reused"].get((name,base["origin"],policy,base["seed"]))
                if old is not None:
                    source_drift.append(dict(origin=base["origin"],seed=base["seed"],policy=policy,
                        maximum_source_prefix_box_error=float(np.max(np.abs(data["boxes"]-old["boxes"][:len(data["boxes"])])))))
    if calls!=plan["smoke_search_calls"]: raise ValueError("L40 smoke call budget changed")
    c.verify_files(plan["source_file_sha256"])
    gate=c.seal(dict(schema=c.SCHEMA,status="PASS_L40_MATCHED_RECROP_SMOKE",role="execution_only_not_method_gain",
        plan_fingerprint=plan["fingerprint"],run_fingerprint=run_record(plan)["fingerprint"],
        maximum_public_control_box_error=maximum,maximum_decode_error=decode,extra_smoke_search_calls=calls,
        source_prefix_drift_descriptive_only=source_drift,observed_head_sha256=heads,cache_sha256=hashes,training_performed=False))
    c.publish_identical(Path(root)/"smoke/stage61_l40_smoke.json",gate)
    print("Stage-61 L40: PASS_L40_MATCHED_RECROP_SMOKE; three current prefixes exactly match independent public L40 replay")
    return gate


def smoke_gate(plan,root):
    value=c.read_sealed(Path(root)/"smoke/stage61_l40_smoke.json")
    if (value.get("status")!="PASS_L40_MATCHED_RECROP_SMOKE" or value.get("plan_fingerprint")!=plan["fingerprint"] or
        value.get("run_fingerprint")!=run_record(plan)["fingerprint"] or value.get("extra_smoke_search_calls")!=plan["smoke_search_calls"] or
        value.get("maximum_public_control_box_error")!=0 or
        not 0<=value.get("maximum_decode_error",float("inf"))<=plan["design"]["box_atol"]):
        raise ValueError("Need a passing exact-device L40 smoke")
    c.verify_files(value["cache_sha256"])
    return value


def evaluate(plan,root,source,observer,records,image_reader=checked_image):
    smoke_gate(plan,root); c.publish_identical(Path(root)/"run.json",run_record(plan))
    hashes={}; heads={}; calls=count=0; maximum_decode=0.
    for unit in plan["units"]:
        with installed(observer,records,unit) as head:
            path,data=trace(plan,root,observer,records,unit,image_reader=image_reader)
        bind(hashes,path); heads[c.key(unit)]=head; calls+=int(data["head_calls"].sum()); count+=1
        maximum_decode=max(maximum_decode,float(data["decode_error"].max()))
        print("[L40 audited] "+c.key(unit),flush=True)
    if (calls,count)!=(plan["formal_search_calls"],plan["new_full_trajectories"]): raise ValueError("L40 formal budget changed")
    c.verify_files(plan["source_file_sha256"])
    run=c.seal(dict(schema=c.SCHEMA,status="COMPLETE_L40_ALL_SIX_ARMS",plan_fingerprint=plan["fingerprint"],
        run_fingerprint=run_record(plan)["fingerprint"],new_full_trajectories=count,formal_search_calls=calls,
        maximum_decode_error=maximum_decode,observed_head_sha256=heads,cache_sha256=hashes,training_performed=False))
    c.publish_identical(Path(root)/"evaluation/run.json",run)
    return run


def completed(plan,root):
    smoke_gate(plan,root)
    if c.read_sealed(Path(root)/"run.json")!=run_record(plan): raise ValueError("L40 runtime record changed")
    run=c.read_sealed(Path(root)/"evaluation/run.json")
    expected={str((Path(root)/"evaluation/traces"/(c.key(u)+suffix)).resolve()) for u in plan["units"] for suffix in (".npz",".receipt.json")}
    if (run.get("status")!="COMPLETE_L40_ALL_SIX_ARMS" or run.get("plan_fingerprint")!=plan["fingerprint"] or
        run.get("run_fingerprint")!=run_record(plan)["fingerprint"] or run.get("new_full_trajectories")!=plan["new_full_trajectories"] or
        run.get("formal_search_calls")!=plan["formal_search_calls"] or set(run.get("cache_sha256",{}))!=expected or
        set(run.get("observed_head_sha256",{}))!={c.key(u) for u in plan["units"]}):
        raise ValueError("Need complete unmixed L40 evaluation")
    c.verify_files(run["cache_sha256"])
    return run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",required=True,choices=("prepare","smoke","evaluate"))
    parser.add_argument("--input",required=True,type=nonempty_path); parser.add_argument("--output_dir",type=nonempty_path)
    parser.add_argument("--checkpoint",default=os.environ.get("CHECKPOINT")); parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--parameter",default="deit_distilled"); args=parser.parse_args()
    if args.mode=="prepare":
        if not args.output_dir: parser.error("prepare requires --output_dir")
        import torch
        old,_=c.original.load_plan(args.input)
        prepare(args.input,args.output_dir,c.device_binding(old,args.gpu,torch)); return
    if not args.checkpoint: parser.error("Need original V1 CHECKPOINT")
    root=Path(args.input).expanduser().resolve(); plan,source=c.load_plan(root)
    with writer_lock(root):
        native,identity=startup.checked_runtime(args,plan,root)
        if identity!=plan["runtime_identity"]: raise ValueError("L40 runtime binding changed")
        records,_=c.original.prior.prior.previous.model_records(source["old"]["plan"],source["old"]["root"],native.rt.torch)
        observer=Capture(native)
        try:
            if args.mode=="smoke": smoke(plan,root,source,observer,records)
            else: evaluate(plan,root,source,observer,records)
        finally: observer.handle.remove()


if __name__=="__main__": main()
