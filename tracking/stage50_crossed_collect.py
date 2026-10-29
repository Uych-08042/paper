"""Collect original SGLA features once per predicted parent and compare frozen heads."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np

from stage27_uav_common import checked_image, crop_geometry, nonempty_path, publish_identical
from stage27_uav_collect import writer_lock
from stage28_response_common import cpu_array
from stage46_shift_audit import verify_files
from stage47_search_collect import Capture
from stage48_stability_experiment import evaluation_records
from stage48_stability_runtime import digest,head_output,make_head
import stage50_crossed_common as c


class NativeCapture(Capture):
    def __init__(self,native):
        super().__init__(native)
        self.feature=None; self.output=None
        self.pre_handle=self.rt.net.box_head.register_forward_pre_hook(self._input)

    def _input(self,module,args):
        # Retain the exact GPU tensor/layout; clones do not mutate their input.
        self.feature=args[0].detach()

    def _output(self,module,args,output):
        super()._output(module,args,output)
        self.output={k:v.detach() for k,v in output.items() if k in ("pred_boxes","score_map","size_map","offset_map")}

    def capture(self,image,parent,frame):
        self.feature=self.output=None
        result=super().capture(image,parent,frame)
        if self.feature is None or self.output is None: raise ValueError("Missing original head input/output")
        return result


class HeadBank:
    def __init__(self,native,records,fold):
        self.rt=native.rt; self.heads={}; self.counts={}; self.hashes={}; self.handles=[]
        for record,state in records:
            if record["fold"]!=fold: continue
            name=record["arm"]+"_seed"+str(record["seed"])
            head=make_head(self.rt).requires_grad_(False)
            head.load_state_dict(state,strict=True)
            if digest(head)!=record["final_head_sha256"]: raise ValueError("Loaded frozen head state differs")
            self.heads[name]=head; self.hashes[name]=digest(head); self.counts[name]=0
            self.handles.append(head.register_forward_hook(self.counter(name)))
        if set(self.heads)!=set(c.HEADS[1:]): raise ValueError("Need all eight OOF heads for this source fold")

    def counter(self,name):
        def hook(module,args,output): self.counts[name]+=1
        return hook

    def reset(self):
        for name in self.counts: self.counts[name]=0

    def assert_frozen(self):
        if any(digest(h)!=self.hashes[n] for n,h in self.heads.items()): raise ValueError("Frozen comparison head/BN changed")


def decode_pair(rt,output,parent,factor,hw):
    rt.tracker.state=np.asarray(parent,float).tolist()
    weighted=rt.s1._decode_box(rt.tracker,output,factor,*hw)
    box=(output["pred_boxes"].reshape(-1,4).mean(0)*rt.tracker.params.search_size/factor).tolist()
    raw=rt.s1.clip_box(rt.tracker.map_box_back(box,factor),*hw,margin=10)
    return np.asarray(weighted,float),np.asarray(raw,float)


def collect_point(observer,bank,image,parent,frame):
    """No GT is passed to this actor; all heads receive one identical original feature tensor."""
    rt=observer.rt; bank.reset(); result=observer.capture(image,parent,frame)
    feature_hash=hashlib.sha256(np.ascontiguousarray(cpu_array(observer.feature)).tobytes()).hexdigest()
    crop,_,factor=crop_geometry(parent,image.shape[:2],rt.tracker.params.search_factor,rt.tracker.params.search_size)
    rows=[]
    for name in c.HEADS:
        with rt.torch.no_grad():
            output=observer.output if name=="native" else head_output(bank.heads[name],observer.feature)
            weighted,raw=decode_pair(rt,output,parent,factor,image.shape[:2])
        if name=="native": np.testing.assert_allclose(weighted,result["boxes"],atol=1e-9,rtol=0)
        rows.append(dict(raw=cpu_array(output["score_map"])[0,0],size=cpu_array(output["size_map"])[0],
            offset=cpu_array(output["offset_map"])[0],boxes=weighted,raw_boxes=raw,
            raw_normalized=cpu_array(output["pred_boxes"]).reshape(-1,4)[0]))
    if hashlib.sha256(np.ascontiguousarray(cpu_array(observer.feature)).tobytes()).hexdigest()!=feature_hash:
        raise ValueError("A comparison head modified the shared original feature tensor")
    counts=[result["head_calls"]]+[bank.counts[n] for n in c.HEADS[1:]]
    return dict({k:np.asarray([r[k] for r in rows]) for k in rows[0]},
        frame_id=np.asarray(frame),parent=np.asarray(parent,float),crop=crop,image_hw=np.asarray(image.shape[:2]),
        factor=np.asarray(factor),layer=np.asarray(result["layer"]),full_block_calls=result["block_calls"],
        selector_calls=np.asarray(result["selector_calls"]),head_forward_calls=np.asarray(counts),
        crop_sha256=np.asarray(result["crop_sha256"]),mask_sha256=np.asarray(result["mask_sha256"]),
        feature_sha256=np.asarray(feature_hash))


def prepare(source,root):
    source,root=Path(source).expanduser().resolve(),Path(root).expanduser().resolve()
    if source==root or source in root.parents or root in source.parents: raise ValueError("Use a separate new Stage-50 directory")
    plan,_=c.make_plan(source)
    with writer_lock(root):
        if not (root/"manifest.json").exists() and any(p.name!="collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-50 directory")
        publish_identical(root/"manifest.json",plan)
    print("Stage-50: {} shared inputs / {} full SGLA forwards / {} head-only forwards".format(
        plan["expected_inputs"],plan["expected_full_forwards"],plan["expected_head_only_forwards"]))


def execute(root,plan,inputs,observer,banks,identity,smoke=False,image_reader=checked_image):
    if identity!=inputs["plan"]["runtime_identity"]: raise ValueError("Need unchanged original Stage-46 runtime")
    window=cpu_array(observer.rt.tracker.output_window).reshape(observer.rt.search_side,observer.rt.search_side)
    np.testing.assert_array_equal(window,np.asarray(inputs["plan"]["window"],np.float32))
    run=c.seal(dict(schema=c.SCHEMA,role="independent_shared_feature_head_replay",plan_fingerprint=plan["fingerprint"],
        runtime=identity,window=window.tolist(),heads=list(c.HEADS)))
    publish_identical(root/"run.json",run)
    if not smoke:
        gate=c.read_sealed(root/"smoke/stage50_smoke.json")
        if gate.get("run_fingerprint")!=run["fingerprint"] or gate.get("status")!="PASS_SHARED_PARENT_HEAD_SMOKE":
            raise ValueError("Need the unchanged Stage-50 smoke")
    checks=[]; active=None
    for point in c.points(plan,smoke):
        entry,arrays=inputs["focus"][point["sequence"]]
        fold=next(f["fold"] for f in inputs["plan"]["folds"] if point["sequence"] in f["held_out"])
        if active!=point["case_id"]:
            observer.initialize(image_reader(inputs["plan"]["root"],entry["frames"][0]),entry["initial_box"])
            active=point["case_id"]
        path=c.path_for(root,point); data=c.read_cache(path,c.metadata(run,point))
        if data is None:
            frame=point["frame"]; parent=arrays[point["history"]]["previous"][frame-1]
            image=image_reader(inputs["plan"]["root"],entry["frames"][frame])
            data=collect_point(observer,banks[fold],image,parent,frame)
            c.audit_point(data,point,inputs,plan)
            c.save(path,data,c.metadata(run,point))
        checks.append(c.audit_point(data,point,inputs,plan))
        print("[shared input audited] "+point["point_id"],flush=True)
    observer.assert_frozen()
    for bank in banks.values(): bank.assert_frozen()
    verify_files(plan["source_file_sha256"])
    if smoke:
        publish_identical(root/"smoke/stage50_smoke.json",c.seal(dict(schema=c.SCHEMA,status="PASS_SHARED_PARENT_HEAD_SMOKE",
            role="execution_and_own_diagonal_only",run_fingerprint=run["fingerprint"],inputs=len(checks),
            maximum_diagonal_box_error=max(r["maximum_diagonal_box_error"] for r in checks),
            maximum_diagonal_peak_error=max(r["maximum_diagonal_peak_error"] for r in checks),
            maximum_dense_decode_error=max(r["maximum_dense_decode_error"] for r in checks),
            training_performed=False,continuation_performed=False)))
        print("Stage-50: PASS_SHARED_PARENT_HEAD_SMOKE; collected points reused by full collection")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",required=True,choices=("prepare","smoke","collect"))
    parser.add_argument("--input",required=True,type=nonempty_path)
    parser.add_argument("--output_dir",type=nonempty_path)
    parser.add_argument("--checkpoint",default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--parameter",default="deit_distilled"); parser.add_argument("--gpu",type=int,default=0)
    args=parser.parse_args()
    if args.mode=="prepare":
        if not args.output_dir: parser.error("prepare needs --output_dir")
        prepare(args.input,args.output_dir); return
    root=Path(args.input).expanduser().resolve(); plan,inputs=c.load_plan(root)
    if not args.checkpoint: parser.error("Use original SGLA CHECKPOINT or --checkpoint")
    from stage46_motion_experiment import make_runtime
    with writer_lock(root):
        native,identity=make_runtime(args,dict(runtime=inputs["plan"]["runtime"]))
        records,_=evaluation_records(inputs["plan"],Path(plan["source_input"]),native.rt.torch)
        banks={fold:HeadBank(native,records,fold) for fold in range(2)}
        execute(root,plan,inputs,NativeCapture(native),banks,identity,args.mode=="smoke")


if __name__=="__main__": main()
