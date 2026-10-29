"""Native SGLA head stability: frozen features, matched training, OOF rollouts."""
from __future__ import annotations

import argparse
import copy
import os
import tempfile
from pathlib import Path

import numpy as np

from stage27_uav_collect import writer_lock
from stage27_uav_common import nonempty_path, publish_identical
from stage28_response_common import cpu_array
import stage48_stability_common as c


def prepare(source, root):
    source,root=Path(source).expanduser().resolve(),Path(root).expanduser().resolve()
    if source==root or source in root.parents or root in source.parents:
        raise ValueError("Use a separate new Stage-48 tree")
    plan=c.make_plan(source)
    with writer_lock(root):
        if not (root/"manifest.json").exists() and any(p.name!="collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-48 directory")
        publish_identical(root/"manifest.json",plan)
    print("Stage-48: 12 consumed videos, 2 folds x 2 seeds x 4 head-only arms = 16 models / 96 OOF traces")
    print("Plan: "+plan["fingerprint"])


def collect(native, plan, root):
    from stage48_stability_runtime import FeatureCapture, collect_features
    observer=FeatureCapture(native)
    bindings={}; support=[]
    with writer_lock(root/"features"):
        for entry in plan["entries"]:
            path=c.cache_path(root,entry)
            data=c.read_cache(path,c.feature_meta(plan,entry))
            reference=c.source_native(plan,entry)
            if data is None:
                data=collect_features(observer,plan,entry,reference)
                c.save_cache(path,data,c.feature_meta(plan,entry))
            c.audit_features(data,plan,entry,reference)
            eligible=data["positive"].any((2,3))&data["negative"].any((2,3))
            cell=(data["raw"]*np.asarray(plan["window"])).reshape(len(data["frame_id"]),2,-1).argmax(2)
            matched=np.take_along_axis(data["positive"].reshape(len(data["frame_id"]),2,-1),cell[...,None],2)[...,0]
            support.append(dict(sequence=entry["sequence"],pairs=len(data["frame_id"]),rank_supported_views=int(eligible.sum()),
                frozen_rank_error_views=int((eligible&~matched).sum()),size_supported_views=int(data["size_valid"].sum())))
            for p in (path,path.with_suffix(".receipt.json")):
                bindings[str(p)]=c.file_hash(p)
            print("[features audited]",entry["sequence"],len(data["frame_id"]),flush=True)
        observer.assert_frozen()
        publish_identical(root/"features/stage48_features.json",c.seal(dict(schema=c.SCHEMA,
            status="COMPLETE_FROZEN_NATIVE_FEATURES",plan_fingerprint=plan["fingerprint"],file_sha256=bindings,
            support=support,backbone_and_head_frozen=True)))


def feature_gate(plan,root):
    report=c.read_sealed(root/"features/stage48_features.json")
    if report.get("status")!="COMPLETE_FROZEN_NATIVE_FEATURES" or report.get("plan_fingerprint")!=plan["fingerprint"]:
        raise ValueError("Collect all frozen features first")
    c.prior.prior.verify_files(report["file_sha256"])
    return report


def load_torch(torch,path):
    try:
        return torch.load(path,map_location="cpu",weights_only=False)
    except TypeError:
        return torch.load(path,map_location="cpu")


def save_step(torch,directory,state):
    path=directory/("step{:06d}.pt".format(state["step"]))
    receipt=path.with_suffix(".receipt.json")
    if receipt.exists():
        raise ValueError("Never overwrite committed training state")
    if path.exists() and load_torch(torch,path).get("metadata")!=state["metadata"]:
        raise ValueError("Unidentified orphan training checkpoint")
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix=".tmp",delete=False) as handle:
        temporary=handle.name
    try:
        torch.save(state,temporary)
        os.replace(temporary,path)
    finally:
        if os.path.isfile(temporary):
            os.unlink(temporary)
    publish_identical(receipt,c.seal(dict(schema=c.SCHEMA,metadata=state["metadata"],step=state["step"],sha256=c.file_hash(path))))
    return path


def restore_latest(torch,directory,metadata):
    latest=None
    for receipt in sorted(directory.glob("step*.receipt.json")):
        record=c.read_sealed(receipt)
        path=directory/("step{:06d}.pt".format(record["step"]))
        if (record.get("metadata")!=metadata or receipt!=path.with_suffix(".receipt.json") or
            not 0<record["step"]<=metadata["updates"] or not path.is_file() or c.file_hash(path)!=record["sha256"]):
            raise ValueError("Committed training checkpoint changed")
        latest=load_torch(torch,path)
        if latest.get("metadata")!=metadata or latest.get("step")!=record["step"]:
            raise ValueError("Checkpoint budget/source mismatch")
    return latest


def train_arm(native,plan,root,pool,bindings,supported,fold,seed,arm,smoke):
    from stage48_stability_runtime import digest,gradient_norm,head_output,make_head,objective
    rt,torch=native.rt,native.rt.torch
    head=make_head(rt)
    initial,buffers=digest(head),digest(head,True)
    updates=c.DESIGN["smoke_updates"] if smoke else c.DESIGN["updates"]
    draws=np.random.default_rng(seed+100*fold).integers(0,len(pool),size=(updates,c.DESIGN["batch_pairs"]))
    draws[0,0]=supported[0]  # Fit-only common gradient-check pair, same in every arm.
    meta=dict(schema=c.SCHEMA,plan_fingerprint=plan["fingerprint"],fold=fold,seed=seed,arm=arm,
        mode="smoke" if smoke else "full",updates=updates,initial_head_sha256=initial,
        runtime=plan["runtime_identity"],fit=plan["folds"][fold]["fit"],held_out=plan["folds"][fold]["held_out"],
        feature_sha256=bindings,draws_fingerprint=c.fingerprint(draws.tolist()))
    directory=root/("smoke/training" if smoke else "training")/("seed{}".format(seed))/("fold{}".format(fold))/arm
    opt=torch.optim.AdamW(head.parameters(),lr=c.DESIGN["learning_rate"],weight_decay=c.DESIGN["weight_decay"])
    saved=restore_latest(torch,directory,meta)
    start,history,maxima=(0,[],dict(rank=0.,shape=0.)) if saved is None else (saved["step"],saved["history"],saved["maximum_gradient"])
    if saved:
        head.load_state_dict(saved["head"],strict=True)
        opt.load_state_dict(saved["optimizer"])
    # A batch-one head round-trip checks captured features against native maps.
    parity=0. if saved is None else saved["initial_map_error"]
    if not start:
        sample=pool[supported[0]]
        with torch.no_grad():
            for view in range(2):
                feature=torch.as_tensor(sample["features"][view:view+1],device=next(head.parameters()).device)
                out=head_output(head,feature)
                for key,label in (("score_map","raw"),("size_map","size"),("offset_map","offset")):
                    parity=max(parity,float(np.max(np.abs(cpu_array(out[key])[0]-sample[label][view]))))
        if parity>1e-6:
            raise ValueError("Cached features do not reproduce the unchanged original head")
    for step in range(start+1,updates+1):
        opt.zero_grad(set_to_none=True)
        values=objective(torch,head,[pool[int(i)] for i in draws[step-1]],plan["window"],arm)
        if step==1 or step==updates:
            for key,loss in (("rank",values[2]),("shape",values[3])):
                maxima[key]=max(maxima[key],gradient_norm(torch,loss,head.parameters()))
        values[0].backward()
        norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),c.DESIGN["gradient_clip"]))
        if not np.isfinite([norm,*[float(v.detach()) for v in values[:4]]]).all():
            raise ValueError("Non-finite head training loss/gradient")
        opt.step()
        if step==1 or step%c.DESIGN["save_every"]==0 or step==updates:
            history.append(dict(step=step,total=float(values[0].detach()),ordinary=float(values[1].detach()),
                applied_rank=float(values[2].detach()),applied_shape=float(values[3].detach()),gradient_norm=norm,**values[4]))
            print("[head fold{} seed{} {}] {}/{} loss={:.4f}".format(fold,seed,arm,step,updates,history[-1]["total"]),flush=True)
        if step%c.DESIGN["save_every"]==0 or step==updates:
            if digest(head,True)!=buffers:
                raise ValueError("Frozen head BN buffers changed")
            save_step(torch,directory,dict(metadata=meta,step=step,head={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
                optimizer=opt.state_dict(),history=history,maximum_gradient=maxima,initial_map_error=parity))
    if digest(head)==initial or digest(head,True)!=buffers:
        raise ValueError("Head did not change, or frozen buffers changed")
    if (maxima["rank"]>0)!=(arm in ("rank","rank_shape")) or (maxima["shape"]>0)!=(arm in ("shape","rank_shape")):
        raise ValueError("Loss intervention gradient audit failed")
    path=directory/("step{:06d}.pt".format(updates))
    return dict(fold=fold,seed=seed,arm=arm,checkpoint=str(path.resolve()),checkpoint_sha256=c.file_hash(path),
        receipt_sha256=c.file_hash(path.with_suffix(".receipt.json")),initial_head_sha256=initial,final_head_sha256=digest(head),
        buffers_unchanged=True,draws_fingerprint=meta["draws_fingerprint"],maximum_gradient=maxima,initial_map_error=parity,history=history)


def train(native,plan,root,seed,smoke=False):
    from stage48_stability_runtime import run_trace
    feature_gate(plan,root)
    if not smoke:
        smoke_gate(plan,root)
    records=[]
    directory=root/("smoke" if smoke else "training/seed{}".format(seed))
    with writer_lock(directory):
        for fold in ([0] if smoke else range(c.DESIGN["folds"])):
            pool,bindings,supported=c.fit_pool(root,plan,fold)
            for arm in c.ARMS:
                records.append(train_arm(native,plan,root,pool,bindings,supported,fold,seed,arm,smoke))
            if len({r["draws_fingerprint"] for r in records if r["fold"]==fold})!=1 or len({r["initial_head_sha256"] for r in records if r["fold"]==fold})!=1:
                raise ValueError("Matched arm schedules or initialization differ")
        native.assert_frozen()
        c.prior.prior.verify_files(plan["source_file_sha256"])
        report=c.seal(dict(schema=c.SCHEMA,status="PASS_HEAD_STABILITY_SMOKE" if smoke else "COMPLETE_FROZEN_HEAD_TRAINING",
            plan_fingerprint=plan["fingerprint"],seed=seed,records=records,held_out_labels_used_for_fitting=False))
        if smoke:
            entry=next(e for e in plan["entries"] if e["sequence"]==plan["folds"][0]["held_out"][0])
            data=run_trace(native,entry["frames"][:c.DESIGN["smoke_frames"]+1],entry["initial_box"],plan["root"])
            ref=c.source_native(plan,entry)
            np.testing.assert_allclose(data["boxes"],ref["boxes"][:len(data["boxes"])],atol=.001,rtol=0)
            np.testing.assert_array_equal(np.trunc(data["boxes"]),np.trunc(ref["boxes"][:len(data["boxes"])]))
            for record in records:
                payload=load_torch(native.rt.torch,Path(record["checkpoint"]))
                # Test each learned head through the actual original tracker.
                original=copy.deepcopy(native.rt.net.box_head.state_dict())
                from stage46_motion_runtime import state_digest
                try:
                    native.rt.net.box_head.load_state_dict(payload["head"],strict=True)
                    native.digest=state_digest(native.rt.net)
                    run_trace(native,entry["frames"][:c.DESIGN["smoke_frames"]+1],entry["initial_box"],plan["root"])
                finally:
                    native.rt.net.box_head.load_state_dict(original,strict=True)
                    native.digest=state_digest(native.rt.net)
            native.assert_frozen()
            report=c.seal(dict(**{k:v for k,v in report.items() if k!="fingerprint"},native_replay_pass=True,learned_native_rollout_pass=True))
        publish_identical(directory/("stage48_smoke.json" if smoke else "stage48_training.json"),report)
    print("Stage-48: "+report["status"])
    return report


def smoke_gate(plan,root):
    report=c.read_sealed(root/"smoke/stage48_smoke.json")
    if report.get("status")!="PASS_HEAD_STABILITY_SMOKE" or report.get("plan_fingerprint")!=plan["fingerprint"]:
        raise ValueError("Need a passing Stage-48 smoke")
    for r in report["records"]:
        c.prior.prior.verify_files({r["checkpoint"]:r["checkpoint_sha256"],str(Path(r["checkpoint"]).with_suffix(".receipt.json")):r["receipt_sha256"]})


def evaluation_records(plan,root,torch):
    records,bindings=[],{}
    for seed in c.SEEDS:
        path=root/"training"/("seed{}".format(seed))/"stage48_training.json"
        report=c.read_sealed(path)
        expected={(f,a) for f in range(c.DESIGN["folds"]) for a in c.ARMS}
        if (report.get("status")!="COMPLETE_FROZEN_HEAD_TRAINING" or report.get("plan_fingerprint")!=plan["fingerprint"] or
            report.get("seed")!=seed or len(report["records"])!=len(expected) or {(r["fold"],r["arm"]) for r in report["records"]}!=expected):
            raise ValueError("Freeze all 16 models before full OOF evaluation")
        bindings[str(path)]=c.file_hash(path)
        for r in report["records"]:
            checkpoint=Path(r["checkpoint"])
            c.prior.prior.verify_files({str(checkpoint):r["checkpoint_sha256"],str(checkpoint.with_suffix(".receipt.json")):r["receipt_sha256"]})
            value=load_torch(torch,checkpoint)
            m=value["metadata"]
            if (value["step"]!=c.DESIGN["updates"] or m["plan_fingerprint"]!=plan["fingerprint"] or m["mode"]!="full" or
                m["fold"]!=r["fold"] or m["seed"]!=seed or m["arm"]!=r["arm"] or
                m["fit"]!=plan["folds"][r["fold"]]["fit"] or m["held_out"]!=plan["folds"][r["fold"]]["held_out"]):
                raise ValueError("Wrong OOF checkpoint source/budget")
            c.prior.prior.verify_files(m["feature_sha256"])
            bindings[str(checkpoint)]=r["checkpoint_sha256"]
            bindings[str(checkpoint.with_suffix(".receipt.json"))]=r["receipt_sha256"]
            records.append((r,value["head"]))
    return records,bindings


def evaluate(native,plan,root):
    from stage46_motion_runtime import state_digest
    from stage48_stability_runtime import head_output,run_trace
    smoke_gate(plan,root)
    feature_report=feature_gate(plan,root)
    records,bindings=evaluation_records(plan,root,native.rt.torch)
    feature_bindings=dict(feature_report["file_sha256"])
    feature_bindings[str(root/"features/stage48_features.json")]=c.file_hash(root/"features/stage48_features.json")
    run=c.seal(dict(schema=c.SCHEMA,role="native_head_own_prediction_oof_evaluation",plan_fingerprint=plan["fingerprint"],model_sha256=bindings,feature_sha256=feature_bindings))
    original=copy.deepcopy(native.rt.net.box_head.state_dict())
    with writer_lock(root/"evaluation"):
        publish_identical(root/"evaluation/run.json",run)
        try:
            for entry in plan["entries"]:
                fold=next(f["fold"] for f in plan["folds"] if entry["sequence"] in f["held_out"])
                for record,state in records:
                    if record["fold"]!=fold:
                        continue
                    method=record["arm"]+"_seed"+str(record["seed"])
                    meta=dict(run_fingerprint=run["fingerprint"],sequence=entry["sequence"],method=method)
                    path=root/"evaluation/traces"/(entry["sequence"]+"__"+method+".npz")
                    native.rt.net.box_head.load_state_dict(state,strict=True)
                    native.digest=state_digest(native.rt.net)
                    data=c.read_cache(path,meta)
                    if data is None:
                        data=run_trace(native,entry["frames"],entry["initial_box"],plan["root"])
                        audit_trace(data,entry)
                        c.save_cache(path,data,meta)
                    audit_trace(data,entry)
                    pack_path=c.cache_path(root,entry)
                    pack=c.read_cache(pack_path,c.feature_meta(plan,entry))
                    if pack is None:
                        raise ValueError("Missing held-out diagnostic features")
                    stress_path=root/"evaluation/probes"/path.name
                    probe=c.read_cache(stress_path,dict(meta,role="same_state_cached_features"))
                    if probe is None:
                        rows=[]
                        for i,frame in enumerate(pack["frame_id"]):
                            boxes=[]; outputs=[]
                            for v in range(2):
                                # Match original capture's batch-one head execution.
                                with native.rt.torch.no_grad():
                                    output=head_output(native.rt.net.box_head,native.rt.torch.as_tensor(pack["features"][i,v:v+1],device=next(native.rt.net.parameters()).device))
                                outputs.append(output)
                                parent=pack["parents"][i,v]
                                hw=[entry["frames"][int(frame)]["height"],entry["frames"][int(frame)]["width"]]
                                factor=plan["search_size"]/pack["crop"][i,v,2]
                                native.rt.tracker.state=parent.tolist()
                                boxes.append(native.rt.s1._decode_box(native.rt.tracker,output,factor,*hw))
                            rows.append(dict(boxes=boxes,raw=np.stack([cpu_array(o["score_map"])[0,0] for o in outputs]),
                                size=np.stack([cpu_array(o["size_map"])[0] for o in outputs]),offset=np.stack([cpu_array(o["offset_map"])[0] for o in outputs])))
                        probe={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
                        c.save_cache(stress_path,probe,dict(meta,role="same_state_cached_features"))
                    native.assert_frozen()
                    print("[OOF audited]",path.name,flush=True)
        finally:
            native.rt.net.box_head.load_state_dict(original,strict=True)
            native.digest=state_digest(native.rt.net)
        c.prior.prior.verify_files(bindings)
        c.prior.prior.verify_files(plan["source_file_sha256"])


def audit_trace(data,entry):
    n=len(entry["frames"])-1
    shapes=dict(previous=(n,4),boxes=(n,4),layer=(n,),peak=(n,),head_calls=(n,),block_calls=(n,12),image_hw=(n,2))
    if set(data)!=set(shapes) or any(data[k].shape!=s for k,s in shapes.items()) or not all(np.isfinite(v).all() for v in data.values()):
        raise ValueError("Invalid complete native OOF trace")
    np.testing.assert_array_equal(data["previous"],np.vstack([entry["initial_box"],data["boxes"][:-1]]))
    np.testing.assert_array_equal(data["head_calls"],np.ones(n,int))
    expected=np.zeros((n,12),int); expected[:,:6]=1
    if not np.isin(data["layer"],range(7,13)).all() or (data["boxes"][:,2:]<=0).any():
        raise ValueError("Invalid native layer/box")
    expected[np.arange(n),data["layer"]-1]=1
    np.testing.assert_array_equal(data["block_calls"],expected)
    np.testing.assert_array_equal(data["image_hw"],[[f["height"],f["width"]] for f in entry["frames"][1:]])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",required=True,choices=("prepare","collect","smoke","train","evaluate"))
    p.add_argument("--input",required=True,type=nonempty_path)
    p.add_argument("--output_dir",type=nonempty_path)
    p.add_argument("--checkpoint",default=os.environ.get("CHECKPOINT"))
    p.add_argument("--parameter",default="deit_distilled")
    p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--seed",type=int,choices=c.SEEDS,default=c.SEEDS[0])
    args=p.parse_args()
    if args.mode=="prepare":
        if not args.output_dir: p.error("prepare needs --output_dir")
        prepare(args.input,args.output_dir); return
    root=Path(args.input).expanduser().resolve(); plan=c.load_plan(root)
    if not args.checkpoint: p.error("Use CHECKPOINT or --checkpoint with original SGLA weights")
    from stage46_motion_experiment import make_runtime
    native,identity=make_runtime(args,dict(runtime=plan["runtime"]))
    if identity!=plan["runtime_identity"]: raise ValueError("Original Stage-46 runtime/model changed")
    if args.mode=="collect": collect(native,plan,root)
    elif args.mode=="smoke": train(native,plan,root,c.SEEDS[0],True)
    elif args.mode=="train": train(native,plan,root,args.seed)
    else: evaluate(native,plan,root)


if __name__=="__main__":
    main()
