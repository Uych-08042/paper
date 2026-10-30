"""Own two-frame rollouts with direct first- and final-frame geometry gradients."""
from __future__ import annotations
from contextlib import contextmanager
import os
import tempfile
from stage48_stability_experiment import load_torch
from pathlib import Path
import numpy as np
from stage18_tracking_loss import tracking_loss
from stage27_uav_common import checked_image,crop_geometry,tracking_iou
from stage48_stability_runtime import FeatureCapture,head_output,make_head,digest
from stage52_branch_runtime import configure as branch_configure,frozen_digest
import stage56_transition_common as c

def configure(head,arm):
    if arm not in c.ARMS: raise ValueError("Unknown rollout arm")
    return branch_configure(head,"geometry_only")

@contextmanager
def temporary_head(native,state):
    # Preserve original hook-bearing modules. Whole-network hashes are audited
    # after restoration at each checkpoint, rather than copied per short branch.
    original={k:v.detach().clone() for k,v in native.rt.net.box_head.state_dict().items()}
    try:
        native.rt.net.box_head.load_state_dict(state,strict=True)
        yield
    finally: native.rt.net.box_head.load_state_dict(original,strict=True)

def branch(observer,images,initial_box,parent,frames):
    """Actor receives images, first-frame template box and predicted start only."""
    observer.initialize(images[0],initial_box); result=[]; parent=np.asarray(parent,float).copy()
    for frame in frames:
        row=observer.capture(images[frame],parent,frame)
        result.append(dict(parent=parent.copy(),frame=frame,**row))
        parent=np.asarray(row["boxes"],float).copy()  # actual public output, never GT
    return result

def rollout_batch(native,head,plan,entries,anchors,image_reader=checked_image):
    rows=[]; ends=[]; audit=dict(reference_calls=0,student_calls=0,head_only_batches=0,
        head_only_examples=0,successor_parents=0,changed_successor_parents=0,changed_successor_crops=0,qualified_endpoints=0,endpoints=0,qualified_first_transitions=0,first_transitions=0)
    observer=FeatureCapture(native); maximum_native_error=0.
    try:
        for anchor in anchors:
            entry=entries[anchor["sequence"]]; f=anchor["frame"]; frames=list(range(f,f+plan["design"]["horizon"]))
            images={i:image_reader(plan["root"],entry["frames"][i]) for i in [0]+frames}
            teachers=[branch(observer,images,entry["initial_box"],p,frames) for p in anchor["parents"]]
            maximum_native_error=max(maximum_native_error,float(np.max(np.abs(np.array([r["boxes"] for r in teachers[0]])-anchor["native_expected"]))))
            with temporary_head(native,head.state_dict()):
                students=[branch(observer,images,entry["initial_box"],p,frames) for p in anchor["parents"]]
            # Labels are attached only after both branches have advanced.
            for teacher,student in zip(teachers,students):
                overlap=tracking_iou(np.array([r["boxes"] for r in teacher]),np.array([entry["ground_truth"][r["frame"]] for r in teacher]))
                good=bool((overlap>=plan["design"]["reliable_iou"]).all())
                for j,(t,s) in enumerate(zip(teacher,student)):
                    crop,usable,_=crop_geometry(s["parent"],images[s["frame"]].shape[:2],plan["search_factor"],plan["search_size"])
                    gt=np.asarray(entry["ground_truth"][s["frame"]],float)
                    cell=int(np.argmax(s["raw"]*np.asarray(plan["window"],np.float32)))
                    rows.append(dict(features=s["features"],target=np.r_[(gt[:2]-crop[:2])/crop[2],gt[2:]/crop[2]],
                        parent=s["parent"],factor=s["factor"],image_hw=images[s["frame"]].shape[:2],cell=cell,boxes=s["boxes"],raw=s["raw"]))
                    audit["reference_calls"]+=int(t["head_calls"]); audit["student_calls"]+=int(s["head_calls"])
                    if j:
                        audit["successor_parents"]+=1
                        audit["changed_successor_parents"]+=int(np.max(np.abs(s["parent"]-t["parent"]))>1e-6)
                        tcrop=crop_geometry(t["parent"],images[t["frame"]].shape[:2],plan["search_factor"],plan["search_size"])[0]
                        audit["changed_successor_crops"]+=int(not np.array_equal(crop,tcrop))
                teacher_box=np.asarray(teacher[-1]["boxes"],float); center=teacher_box[:2]+teacher_box[2:]/2
                visible=bool(((center>=usable[:2])&(center<usable[:2]+usable[2:])).all())
                first_box=np.asarray(teacher[0]["boxes"],float); first_center=first_box[:2]+first_box[2:]/2
                first_usable=crop_geometry(student[0]["parent"],images[student[0]["frame"]].shape[:2],plan["search_factor"],plan["search_size"])[1]
                first_good=good and bool(((first_center>=first_usable[:2])&(first_center<first_usable[:2]+first_usable[2:])).all())
                ends.append(dict(index=len(rows)-1,target=teacher_box,qualified=good and visible,
                    first_index=len(rows)-len(student),first_target=first_box,first_qualified=first_good))
                audit["first_transitions"]+=1; audit["qualified_first_transitions"]+=int(first_good)
                audit["endpoints"]+=1; audit["qualified_endpoints"]+=int(good and visible)
        audit["head_only_examples"]=len(rows)
        audit["head_only_batches"]=len(rows)
    finally: observer.handle.remove(); observer.pre_handle.remove()
    if maximum_native_error>plan["design"]["decode_atol"]: raise ValueError("Original short branch differs from source native trajectory")
    return rows,ends,audit,maximum_native_error

def decode_image_boxes(torch,output,rows,search_size):
    """Recorded actor Hann cells, original float32 scaling then float64 mapping."""
    size=output["size_map"]; cells=torch.tensor([r["cell"] for r in rows],device=size.device,dtype=torch.long)
    take=lambda value:value.flatten(2).gather(2,cells[:,None,None].expand(-1,2,1)).squeeze(-1)
    side=size.shape[-1]; grid=torch.stack([cells%side,cells//side],1).to(size.dtype)
    normalized=torch.cat([(grid+take(output["offset_map"]))/side,take(size)],1)
    factors=normalized.new_tensor([r["factor"] for r in rows])[:,None]
    pixels=(normalized*search_size/factors).to(torch.float64)
    parents=pixels.new_tensor(np.stack([r["parent"] for r in rows])); factors64=pixels.new_tensor([r["factor"] for r in rows])[:,None]
    center=pixels[:,:2]+parents[:,:2]+parents[:,2:]/2-search_size/(2*factors64)
    xy=center-pixels[:,2:]/2; hw=pixels.new_tensor([r["image_hw"] for r in rows]); wh=hw[:,[1,0]]
    lo=torch.minimum(xy.clamp_min(0),wh-10); hi=torch.minimum((xy+pixels[:,2:]).clamp_min(10),wh)
    return torch.cat([lo,(hi-lo).clamp_min(10)],1)

def objective(torch,head,rows,ends,plan,arm):
    device=next(head.parameters()).device
    # Recompute batch-one heads, matching the native actor's numerical path.
    outputs=[head_output(head,torch.as_tensor(r["features"],device=device)[None]) for r in rows]
    out={k:torch.cat([o[k] for o in outputs],0) for k in outputs[0]}
    response_error=float(np.max(np.abs(out["score_map"].detach().cpu().numpy()[:,0]-np.stack([r["raw"] for r in rows]))))
    if response_error>plan["design"]["frozen_map_atol"]: raise ValueError("Frozen response replay changed")
    target=torch.as_tensor(np.stack([r["target"] for r in rows]),device=device,dtype=out["score_map"].dtype)
    ordinary,stats=tracking_loss(torch,out,target,True)
    decoded=decode_image_boxes(torch,out,rows,plan["search_size"])
    error=float(np.max(np.abs(decoded.detach().cpu().numpy()-np.array([r["boxes"] for r in rows]))))
    if error>plan["design"]["decode_atol"]: raise ValueError("Differentiable endpoint decode differs from the native actor")
    endpoint_penalty=node_penalty(torch,decoded,ends,plan,False)
    first_penalty=node_penalty(torch,decoded,ends,plan,True)
    endpoint_weight,first_weight=c.WEIGHTS[arm]
    # Preserve the original control's exact scalar expression; no first loss is
    # silently added to it. Both shadow gradients are audited in every arm.
    loss=ordinary+endpoint_weight*endpoint_penalty
    if first_weight:
        loss=loss+first_weight*first_penalty
    return loss,stats,endpoint_penalty,first_penalty,error,response_error


def node_penalty(torch,decoded,ends,plan,first=False):
    prefix="first_" if first else ""
    indices=torch.tensor([r[prefix+"index"] for r in ends],device=decoded.device,dtype=torch.long)
    pred=decoded[indices]; ref=pred.new_tensor(np.stack([r[prefix+"target"] for r in ends])).detach()
    delta=(pred[:,:2]+pred[:,2:]/2-ref[:,:2]-ref[:,2:]/2)/ref[:,2:]
    scale=(pred[:,2:]/ref[:,2:]).log(); eps=plan["design"]["epsilon"]
    per=(delta.abs()-eps).clamp_min(0).mean(1)+(scale.abs()-eps).clamp_min(0).mean(1)
    mask=pred.new_tensor([r[prefix+"qualified"] for r in ends])
    return (per*mask).mean()



def folder(root,fold,seed,arm,smoke=False):
    return Path(root)/("smoke/training" if smoke else "training")/("seed{}".format(seed))/("fold{}".format(fold))/arm


def save_step(torch,path,state):
    if path.with_suffix(".receipt.json").exists(): raise ValueError("Never overwrite committed branch checkpoint")
    if path.exists() and load_torch(torch,path).get("metadata")!=state["metadata"]: raise ValueError("Unidentified orphan branch state")
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix=".tmp",delete=False) as h: temporary=h.name
    try:
        torch.save(state,temporary); os.replace(temporary,path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    c.publish_identical(path.with_suffix(".receipt.json"),c.seal(dict(schema=c.SCHEMA,metadata=state["metadata"],
        step=state["step"],sha256=c.file_hash(path))))


def restore(torch,directory,metadata):
    latest=None
    for receipt in sorted(directory.glob("step*.receipt.json")):
        r=c.read_sealed(receipt); path=directory/("step{:06d}.pt".format(r["step"]))
        if (r.get("metadata")!=metadata or not 0<r["step"]<=metadata["updates"] or
            receipt!=path.with_suffix(".receipt.json") or not path.is_file() or c.file_hash(path)!=r["sha256"]):
            raise ValueError("Committed branch checkpoint changed")
        latest=load_torch(torch,path)
        if latest.get("metadata")!=metadata or latest.get("step")!=r["step"]: raise ValueError("Branch checkpoint metadata changed")
    return latest

def train_arm(native,plan,root,pool,bindings,fold,seed,arm,smoke=False):
    torch=native.rt.torch; head=make_head(native.rt); groups,names=configure(head,arm)
    parameters=[p for p in head.parameters() if p.requires_grad]; initial=digest(head); frozen=frozen_digest(head,names)
    if initial!=c.initial_record(plan,fold,seed)["initial_head_sha256"]: raise ValueError("Original SGLA head initialization changed")
    if dict(fold=fold,pairs=len(pool),fingerprint=c.fingerprint(pool))!=plan["anchor_pools"][fold]: raise ValueError("Fit anchor pool changed")
    training=plan["training"]; updates=training["smoke_updates"] if smoke else training["updates"]
    draws=np.random.default_rng(seed+100*fold).integers(0,len(pool),size=(updates,training["batch_pairs"]))
    metadata=dict(plan_fingerprint=plan["fingerprint"],fold=fold,seed=seed,arm=arm,mode="smoke" if smoke else "full",
        updates=updates,fit=plan["folds"][fold]["fit"],held_out=plan["folds"][fold]["held_out"],feature_sha256=bindings,
        initial_head_sha256=initial,initial_frozen_sha256=frozen,trainable_names=names,draws_fingerprint=c.fingerprint(draws.tolist()),
        anchor_pool=plan["anchor_pools"][fold])
    directory=folder(root,fold,seed,arm,smoke); opt=torch.optim.AdamW(parameters,lr=training["learning_rate"],weight_decay=training["weight_decay"])
    entries={e["sequence"]:e for e in plan["entries"] if e["sequence"] in metadata["fit"]}
    state=restore(torch,directory,metadata); start=0; history=[]; counts=dict(supervised=0,regression=0,center=0)
    totals=dict(reference_calls=0,student_calls=0,head_only_batches=0,head_only_examples=0,successor_parents=0,
        changed_successor_parents=0,changed_successor_crops=0,qualified_endpoints=0,endpoints=0,qualified_first_transitions=0,first_transitions=0)
    maximum_gradient=dict(response=0.,geometry=0.); maximum_endpoint_gradient=0.; maximum_first_gradient=0.; decode_error=0.; response_error=0.; native_error=0.
    if state:
        head.load_state_dict(state["head"],strict=True); opt.load_state_dict(state["optimizer"])
        start=state["step"]; counts=state["counts"]; totals=state["totals"]; history=state["history"]
        maximum_gradient=state["maximum_gradient"]; maximum_endpoint_gradient=state["maximum_endpoint_gradient"]; maximum_first_gradient=state["maximum_first_gradient"]
        decode_error=state["decode_error"]; response_error=state["response_error"]; native_error=state["native_error"]
        if frozen_digest(head,names)!=frozen: raise ValueError("Frozen branch changed on resume")
    for step in range(start+1,updates+1):
        opt.zero_grad(set_to_none=True)
        rows,ends,audit,ne=rollout_batch(native,head,plan,entries,[pool[int(i)] for i in draws[step-1]])
        loss,stats,penalty,first_penalty,de,re=objective(torch,head,rows,ends,plan,arm)
        # All arms pay BOTH node-gradient audits, regardless of loss coefficients.
        eg=torch.autograd.grad(penalty,parameters,retain_graph=True,allow_unused=True)
        en=float(sum(g.detach().square().sum() for g in eg if g is not None).sqrt())
        maximum_endpoint_gradient=max(maximum_endpoint_gradient,en)
        fg=torch.autograd.grad(first_penalty,parameters,retain_graph=True,allow_unused=True)
        fn=float(sum(g.detach().square().sum() for g in fg if g is not None).sqrt())
        maximum_first_gradient=max(maximum_first_gradient,fn); loss.backward()
        if any(p.grad is not None for n,p in head.named_parameters() if n not in names): raise ValueError("Gradient reached frozen response")
        gn=float(torch.nn.utils.clip_grad_norm_(parameters,training["gradient_clip"]))
        if not np.isfinite([float(loss.detach()),gn,en,fn,de,re,ne]).all(): raise ValueError("Non-finite short rollout")
        maximum_gradient["geometry"]=max(maximum_gradient["geometry"],gn)
        opt.step(); decode_error=max(decode_error,de); response_error=max(response_error,re); native_error=max(native_error,ne)
        for k in counts: counts[k]+=int(stats[k+"_count"])
        for k in totals: totals[k]+=audit[k]
        if step==1 or step%training["save_every"]==0 or step==updates:
            history.append(dict(step=step,loss=float(loss.detach()),endpoint_penalty=float(penalty.detach()),geometry_gradient=gn,
                first_penalty=float(first_penalty.detach()),first_gradient=fn,endpoint_gradient=en,**counts,**totals))
            print("[short rollout fold{} seed{} {}] {}/{} loss={:.4f}".format(fold,seed,arm,step,updates,float(loss.detach())),flush=True)
        if step%training["save_every"]==0 or step==updates:
            native.assert_frozen()
            if frozen_digest(head,names)!=frozen: raise ValueError("Response/BN changed")
            save_step(torch,directory/("step{:06d}.pt".format(step)),dict(metadata=metadata,step=step,
                head={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},optimizer=opt.state_dict(),counts=counts,totals=totals,
                history=history,maximum_gradient=maximum_gradient,maximum_endpoint_gradient=maximum_endpoint_gradient,maximum_first_gradient=maximum_first_gradient,
                decode_error=decode_error,response_error=response_error,native_error=native_error))
    expected=updates*training["batch_pairs"]*2*plan["design"]["horizon"]
    if any(totals[k]!=expected for k in ("reference_calls","student_calls","head_only_batches","head_only_examples")) or counts["supervised"]!=expected:
        raise ValueError("Short-branch supervision/call budget changed")
    if (digest(head)==initial or maximum_gradient["geometry"]<=0 or frozen_digest(head,names)!=frozen or
        totals["qualified_endpoints"]<=0 or (maximum_endpoint_gradient<=0 or totals["qualified_first_transitions"]<=0 or (arm=="transition_preserve" and maximum_first_gradient<=0))):
        raise ValueError("No valid short-rollout intervention")
    native.assert_frozen(); c.verify_files(bindings)
    path=directory/("step{:06d}.pt".format(updates))
    source_control=next(r for r in plan["endpoint_controls"] if (r["fold"],r["seed"])==(fold,seed))
    reproduced=None
    if arm=="endpoint_control" and not smoke:
        reproduced=(digest(head)==source_control["final_head_sha256"] and counts==source_control["counts"] and
                    metadata["draws_fingerprint"]==source_control["draws_fingerprint"])
        if not reproduced: raise ValueError("Stage-54 endpoint control did not reproduce exactly; investigate before interpreting this pilot")
    return dict(source_control_sha256=source_control["checkpoint_sha256"],reproduces_stage54_control=reproduced,
        fold=fold,seed=seed,arm=arm,checkpoint=str(path.resolve()),checkpoint_sha256=c.file_hash(path),
        receipt_sha256=c.file_hash(path.with_suffix(".receipt.json")),initial_head_sha256=initial,final_head_sha256=digest(head),
        initial_frozen_sha256=frozen,final_frozen_sha256=frozen_digest(head,names),trainable_names=names,
        trainable_parameters=sum(p.numel() for p in parameters),draws_fingerprint=metadata["draws_fingerprint"],counts=counts,totals=totals,
        maximum_gradient=maximum_gradient,maximum_endpoint_gradient=maximum_endpoint_gradient,maximum_first_gradient=maximum_first_gradient,
        maximum_frozen_map_error=response_error,maximum_decode_error=decode_error,maximum_native_branch_error=native_error,history=history)


