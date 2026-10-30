"""Predicted-parent center/size factorial with frozen Stage-58 heads."""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import numpy as np
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image,crop_geometry,nonempty_path
from stage28_response_common import cpu_array
from stage48_stability_runtime import make_head,head_output,digest
from stage50_crossed_collect import NativeCapture,decode_pair
import stage59_parent_common as c

class HeadBank:
    def __init__(self,native,records,fold):
        from stage58_gradient_runtime import configure,frozen_digest
        self.heads={}; self.hashes={}; self.counts={}; self.handles=[]
        for record,state in records:
            if record["fold"]!=fold or record["arm"] not in c.ARMS: continue
            head=make_head(native.rt); _,names=configure(head,record["arm"])
            if names!=record["trainable_names"] or frozen_digest(head,names)!=record["initial_frozen_sha256"]:
                raise ValueError("OOF head initial frozen scope changed")
            head.load_state_dict(state,strict=True)
            if digest(head)!=record["final_head_sha256"] or frozen_digest(head,names)!=record["final_frozen_sha256"]:
                raise ValueError("OOF head final state changed")
            name=record["arm"]+"_seed"+str(record["seed"]); self.heads[name]=head.requires_grad_(False)
            self.hashes[name]=digest(head); self.counts[name]=0
            self.handles.append(head.register_forward_hook(self.counter(name)))
        if set(self.heads)!={c.evaluator(d,s) for s in c.prior.SEEDS for d in c.DONORS}: raise ValueError("Need all four source OOF heads")
    def counter(self,name):
        def hook(module,args,output): self.counts[name]+=1
        return hook
    def assert_frozen(self):
        if any(digest(h)!=self.hashes[n] for n,h in self.heads.items()): raise ValueError("Frozen OOF head changed")

def collect_point(observer,bank,image,parent,frame,item):
    for n in bank.counts: bank.counts[n]=0
    rt=observer.rt; result=observer.capture(image,parent,frame)
    feature_hash=hashlib.sha256(np.ascontiguousarray(cpu_array(observer.feature)).tobytes()).hexdigest()
    crop,_,factor=crop_geometry(parent,image.shape[:2],rt.tracker.params.search_factor,rt.tracker.params.search_size)
    rows=[]
    for name in c.point_heads(item):
        with rt.torch.no_grad():
            output=observer.output if name=="native" else head_output(bank.heads[name],observer.feature)
            weighted,raw=decode_pair(rt,output,parent,factor,image.shape[:2])
        if name=="native": np.testing.assert_allclose(weighted,result["boxes"],atol=1e-9,rtol=0)
        rows.append(dict(raw=cpu_array(output["score_map"])[0,0],size=cpu_array(output["size_map"])[0],
            offset=cpu_array(output["offset_map"])[0],boxes=weighted,raw_boxes=raw,raw_normalized=cpu_array(output["pred_boxes"]).reshape(-1,4)[0]))
    if hashlib.sha256(np.ascontiguousarray(cpu_array(observer.feature)).tobytes()).hexdigest()!=feature_hash:
        raise ValueError("Comparison head changed shared feature tensor")
    return dict({k:np.asarray([r[k] for r in rows]) for k in rows[0]},frame_id=np.asarray(frame),parent=np.asarray(parent,float),
        crop=crop,image_hw=np.asarray(image.shape[:2]),factor=np.asarray(factor),layer=np.asarray(result["layer"]),
        full_block_calls=result["block_calls"],selector_calls=np.asarray(result["selector_calls"]),
        head_forward_calls=np.asarray([result["head_calls"]]+[bank.counts[n] for n in c.point_heads(item)[1:]]),
        crop_sha256=np.asarray(result["crop_sha256"]),mask_sha256=np.asarray(result["mask_sha256"]),feature_sha256=np.asarray(feature_hash))

from stage55_matched_collect import collect_branch


def prepare(source,root):
    source,root=Path(source).expanduser().resolve(),Path(root).expanduser().resolve()
    if source==root or source in root.parents or root in source.parents: raise ValueError("Use separate Stage-59 directory")
    plan,_=c.make_plan(source)
    with writer_lock(root):
        if not (root/"manifest.json").exists() and any(p.name!="collect.lock" for p in root.iterdir()): raise ValueError("Use empty Stage-59 directory")
        c.publish_identical(root/"manifest.json",plan)
        from stage46_shift_audit import publish_table
        (root/"preparation").mkdir(exist_ok=True)
        publish_table(root/"preparation/stage59_cases.csv",[dict(**r,horizon=plan["design"]["horizon"],parent_combinations=4,evaluators=2) for r in plan["cases"]])
    print("Stage-59: {} shared inputs, {} short branches; {} full / {} head-only forwards; no training".format(
        plan["expected_points"],plan["expected_branches"],plan["expected_full_forwards"],plan["expected_head_only_forwards"]))

def execute(root,plan,inputs,observer,banks,identity,smoke=False,image_reader=checked_image):
    from contextlib import nullcontext
    from stage54_rollout_runtime import temporary_head
    if identity!=inputs["plan"]["runtime_identity"]: raise ValueError("Original runtime changed")
    window=cpu_array(observer.rt.tracker.output_window).reshape(observer.rt.search_side,observer.rt.search_side)
    np.testing.assert_array_equal(window,np.asarray(inputs["plan"]["window"],np.float32))
    run=c.seal(dict(schema=c.SCHEMA,role="frozen_parent_center_size_factorial",plan_fingerprint=plan["fingerprint"],runtime=identity,arms=list(c.ARMS)))
    c.publish_identical(root/"run.json",run)
    if not smoke:
        gate=c.read_sealed(root/"smoke/stage59_smoke.json")
        if gate.get("status")!="PASS_PARENT_FACTORIAL_SMOKE" or gate.get("run_fingerprint")!=run["fingerprint"]: raise ValueError("Need passing Stage-59 smoke")
    checks=[]; branch_checks=[]
    for case in (plan["cases"][:1] if smoke else plan["cases"]):
        entry,arrays=inputs["arrays"][case["sequence"]]; source=inputs["plan"]
        fold=next(f["fold"] for f in source["folds"] if case["sequence"] in f["held_out"]); bank=banks[fold]
        frames=set([0]+list(range(case["frame"],case["frame"]+plan["design"]["horizon"])))
        images={f:image_reader(source["root"],entry["frames"][f]) for f in sorted(frames)}
        observer.initialize(images[0],entry["initial_box"]); starts={}
        for point in [p for p in c.points(plan,smoke) if p["case_id"]==case["case_id"]]:
            path=c.path_for(root,point); meta=c.metadata(run,point); data=c.read_cache(path,meta)
            if data is None:
                data=collect_point(observer,bank,images[point["frame"]],c.parent_for(inputs,point),point["frame"],point)
                c.audit_point(data,point,plan,inputs); c.save(path,data,meta)
            checks.append(c.audit_point(data,point,plan,inputs))
            starts[point["point_id"]]=data
        for item in [b for b in c.branches(plan,smoke) if b["case_id"]==case["case_id"]]:
            path=c.path_for(root,item); meta=c.metadata(run,item); data=c.read_cache(path,meta)
            if data is None:
                context=nullcontext() if item["evaluator"]=="native" else temporary_head(observer.native,bank.heads[item["evaluator"]].state_dict())
                with context:
                    data=collect_branch(observer,images,entry["initial_box"],starts[item["point_id"]]["parent"],
                        range(item["frame"],item["frame"]+plan["design"]["horizon"]))
                c.audit_branch(data,item,starts[item["point_id"]],plan,inputs); c.save(path,data,meta)
            branch_checks.append(c.audit_branch(data,item,starts[item["point_id"]],plan,inputs))
        observer.assert_frozen(); bank.assert_frozen()
        print("[matched case audited] "+case["case_id"],flush=True)
    c.verify_files(plan["source_file_sha256"])
    if smoke:
        c.publish_identical(root/"smoke/stage59_smoke.json",c.seal(dict(schema=c.SCHEMA,status="PASS_PARENT_FACTORIAL_SMOKE",
            run_fingerprint=run["fingerprint"],points=len(checks),branches=len(branch_checks),
            maximum_diagonal_error=max(r["maximum_diagonal_error"] for r in checks),
            maximum_branch_diagonal_error=max(r["maximum_branch_diagonal_error"] for r in branch_checks))))
        print("Stage-59: PASS_PARENT_FACTORIAL_SMOKE; committed caches reused by collect")

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--mode",required=True,choices=("prepare","smoke","collect"))
    p.add_argument("--input",required=True,type=nonempty_path); p.add_argument("--output_dir",type=nonempty_path)
    p.add_argument("--checkpoint",default=os.environ.get("CHECKPOINT")); p.add_argument("--gpu",type=int,default=0); p.add_argument("--parameter",default="deit_distilled")
    args=p.parse_args()
    if args.mode=="prepare":
        if not args.output_dir: p.error("prepare needs --output_dir")
        prepare(args.input,args.output_dir); return
    root=Path(args.input).expanduser().resolve(); plan,inputs=c.load_plan(root)
    if not args.checkpoint: p.error("Set original SGLA CHECKPOINT or --checkpoint")
    from stage46_motion_experiment import make_runtime
    with writer_lock(root):
        native,identity=make_runtime(args,dict(runtime=inputs["plan"]["runtime"]))
        records,_=c.previous.model_records(inputs["plan"],inputs["root"],native.rt.torch)
        banks={f["fold"]:HeadBank(native,records,f["fold"]) for f in inputs["plan"]["folds"]}
        execute(root,plan,inputs,NativeCapture(native),banks,identity,args.mode=="smoke")

if __name__=="__main__": main()
