"""Fit-label reference-quality gating with a count-matched random control."""
from __future__ import annotations
import os
import tempfile
from stage48_stability_experiment import load_torch
from pathlib import Path
import numpy as np
from stage18_tracking_loss import tracking_loss
from stage27_uav_common import checked_image,crop_geometry,tracking_iou
from stage48_stability_runtime import FeatureCapture,head_output,make_head,digest
from stage52_branch_runtime import configure as branch_configure,frozen_digest
import stage57_quality_common as c

def configure(head,arm):
    if arm not in c.ARMS: raise ValueError("Unknown rollout arm")
    return branch_configure(head,"geometry_only")

# Reuse the frozen native actor and two-frame state protocol byte-for-byte.
from stage56_transition_runtime import temporary_head, branch, decode_image_boxes
from stage56_transition_runtime import rollout_batch as original_rollout_batch


def rollout_batch(native,head,plan,entries,anchors,image_reader=checked_image):
    rows,ends,audit,error=original_rollout_batch(native,head,plan,entries,anchors,image_reader)
    # Attach fit GT quality only AFTER both actors have advanced without labels.
    for i,end in enumerate(ends):
        anchor=anchors[i//2]; entry=entries[anchor["sequence"]]
        gt=np.asarray(entry["ground_truth"][anchor["frame"]+plan["design"]["horizon"]-1],float)
        end["reference_iou"]=float(tracking_iou(np.asarray(end["target"])[None],gt)[0])
        end["student_iou"]=float(tracking_iou(np.asarray(rows[end["index"]]["boxes"])[None],gt)[0])
    return rows,ends,audit,error


def objective(torch,head,rows,ends,plan,arm,priorities):
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
    selected,selection=c.selection_masks(ends,priorities,arm)
    full=node_penalty(torch,decoded,ends,plan)
    chosen=full if arm=="endpoint_control" else node_penalty(torch,decoded,ends,plan,selected)
    loss=ordinary+1.*chosen
    return loss,stats,full,chosen,error,response_error,selection


def node_penalty(torch,decoded,ends,plan,selected=None):
    indices=torch.tensor([r["index"] for r in ends],device=decoded.device,dtype=torch.long)
    pred=decoded[indices]; ref=pred.new_tensor(np.stack([r["target"] for r in ends])).detach()
    delta=(pred[:,:2]+pred[:,2:]/2-ref[:,:2]-ref[:,2:]/2)/ref[:,2:]
    scale=(pred[:,2:]/ref[:,2:]).log(); eps=plan["design"]["epsilon"]
    per=(delta.abs()-eps).clamp_min(0).mean(1)+(scale.abs()-eps).clamp_min(0).mean(1)
    mask=pred.new_tensor([r["qualified"] for r in ends] if selected is None else selected)
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
    priorities=np.random.default_rng(seed+100*fold+plan["design"]["gate_seed_offset"]).random((updates,training["batch_pairs"]*2))
    metadata=dict(priority_fingerprint=c.fingerprint(priorities.tolist()),plan_fingerprint=plan["fingerprint"],fold=fold,seed=seed,arm=arm,mode="smoke" if smoke else "full",
        updates=updates,fit=plan["folds"][fold]["fit"],held_out=plan["folds"][fold]["held_out"],feature_sha256=bindings,
        initial_head_sha256=initial,initial_frozen_sha256=frozen,trainable_names=names,draws_fingerprint=c.fingerprint(draws.tolist()),
        anchor_pool=plan["anchor_pools"][fold])
    directory=folder(root,fold,seed,arm,smoke); opt=torch.optim.AdamW(parameters,lr=training["learning_rate"],weight_decay=training["weight_decay"])
    entries={e["sequence"]:e for e in plan["entries"] if e["sequence"] in metadata["fit"]}
    state=restore(torch,directory,metadata); start=0; history=[]; counts=dict(supervised=0,regression=0,center=0)
    totals=dict(reference_calls=0,student_calls=0,head_only_batches=0,head_only_examples=0,successor_parents=0,
        changed_successor_parents=0,changed_successor_crops=0,qualified_endpoints=0,endpoints=0,qualified_first_transitions=0,first_transitions=0)
    selection_totals={k:0 for k in c.selection_masks([dict(qualified=True,reference_iou=1.,student_iou=1.)],[0.],arm)[1]}
    maximum_gradient=dict(response=0.,geometry=0.); maximum_endpoint_gradient=0.; maximum_selected_gradient=0.; decode_error=0.; response_error=0.; native_error=0.
    if state:
        selection_totals=state["selection_totals"]
        head.load_state_dict(state["head"],strict=True); opt.load_state_dict(state["optimizer"])
        start=state["step"]; counts=state["counts"]; totals=state["totals"]; history=state["history"]
        maximum_gradient=state["maximum_gradient"]; maximum_endpoint_gradient=state["maximum_endpoint_gradient"]; maximum_selected_gradient=state["maximum_selected_gradient"]
        decode_error=state["decode_error"]; response_error=state["response_error"]; native_error=state["native_error"]
        if frozen_digest(head,names)!=frozen: raise ValueError("Frozen branch changed on resume")
    for step in range(start+1,updates+1):
        opt.zero_grad(set_to_none=True)
        rows,ends,audit,ne=rollout_batch(native,head,plan,entries,[pool[int(i)] for i in draws[step-1]])
        loss,stats,penalty,selected_penalty,de,re,selection=objective(torch,head,rows,ends,plan,arm,priorities[step-1])
        # All arms pay full-anchor AND selected-anchor gradient audits.
        eg=torch.autograd.grad(penalty,parameters,retain_graph=True,allow_unused=True)
        en=float(sum(g.detach().square().sum() for g in eg if g is not None).sqrt())
        maximum_endpoint_gradient=max(maximum_endpoint_gradient,en)
        fg=torch.autograd.grad(selected_penalty,parameters,retain_graph=True,allow_unused=True)
        fn=float(sum(g.detach().square().sum() for g in fg if g is not None).sqrt())
        maximum_selected_gradient=max(maximum_selected_gradient,fn); loss.backward()
        if any(p.grad is not None for n,p in head.named_parameters() if n not in names): raise ValueError("Gradient reached frozen response")
        gn=float(torch.nn.utils.clip_grad_norm_(parameters,training["gradient_clip"]))
        if not np.isfinite([float(loss.detach()),gn,en,fn,de,re,ne]).all(): raise ValueError("Non-finite short rollout")
        maximum_gradient["geometry"]=max(maximum_gradient["geometry"],gn)
        opt.step(); decode_error=max(decode_error,de); response_error=max(response_error,re); native_error=max(native_error,ne)
        for k in counts: counts[k]+=int(stats[k+"_count"])
        for k in totals: totals[k]+=audit[k]
        for k in selection_totals: selection_totals[k]+=selection[k]
        if step==1 or step%training["save_every"]==0 or step==updates:
            history.append(dict(step=step,loss=float(loss.detach()),endpoint_penalty=float(penalty.detach()),geometry_gradient=gn,
                selected_penalty=float(selected_penalty.detach()),selected_gradient=fn,endpoint_gradient=en,**counts,**totals,**selection_totals))
            print("[short rollout fold{} seed{} {}] {}/{} loss={:.4f}".format(fold,seed,arm,step,updates,float(loss.detach())),flush=True)
        if step%training["save_every"]==0 or step==updates:
            native.assert_frozen()
            if frozen_digest(head,names)!=frozen: raise ValueError("Response/BN changed")
            save_step(torch,directory/("step{:06d}.pt".format(step)),dict(metadata=metadata,step=step,
                head={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},optimizer=opt.state_dict(),counts=counts,totals=totals,
                selection_totals=selection_totals,history=history,maximum_gradient=maximum_gradient,maximum_endpoint_gradient=maximum_endpoint_gradient,maximum_selected_gradient=maximum_selected_gradient,
                decode_error=decode_error,response_error=response_error,native_error=native_error))
    expected=updates*training["batch_pairs"]*2*plan["design"]["horizon"]
    if any(totals[k]!=expected for k in ("reference_calls","student_calls","head_only_batches","head_only_examples")) or counts["supervised"]!=expected:
        raise ValueError("Short-branch supervision/call budget changed")
    if (digest(head)==initial or maximum_gradient["geometry"]<=0 or frozen_digest(head,names)!=frozen or
        totals["qualified_endpoints"]<=0 or maximum_endpoint_gradient<=0):
        raise ValueError("No valid short-rollout intervention")
    native.assert_frozen(); c.verify_files(bindings)
    path=directory/("step{:06d}.pt".format(updates))
    source_control=next(r for r in plan["endpoint_controls"] if (r["fold"],r["seed"])==(fold,seed))
    reproduced=None
    if arm=="endpoint_control" and not smoke:
        reproduced=(digest(head)==source_control["final_head_sha256"] and counts==source_control["counts"] and
                    metadata["draws_fingerprint"]==source_control["draws_fingerprint"])
        if not reproduced: raise ValueError("Stage-56 endpoint control did not reproduce exactly; investigate before interpreting this pilot")
    return dict(source_control_sha256=source_control["checkpoint_sha256"],reproduces_stage56_control=reproduced,
        fold=fold,seed=seed,arm=arm,checkpoint=str(path.resolve()),checkpoint_sha256=c.file_hash(path),
        receipt_sha256=c.file_hash(path.with_suffix(".receipt.json")),initial_head_sha256=initial,final_head_sha256=digest(head),
        initial_frozen_sha256=frozen,final_frozen_sha256=frozen_digest(head,names),trainable_names=names,
        trainable_parameters=sum(p.numel() for p in parameters),draws_fingerprint=metadata["draws_fingerprint"],counts=counts,totals=totals,
        maximum_gradient=maximum_gradient,maximum_endpoint_gradient=maximum_endpoint_gradient,maximum_selected_gradient=maximum_selected_gradient,
        selection_totals=selection_totals,priority_fingerprint=metadata["priority_fingerprint"],maximum_frozen_map_error=response_error,maximum_decode_error=decode_error,maximum_native_branch_error=native_error,history=history)


