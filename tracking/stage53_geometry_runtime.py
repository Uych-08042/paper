"""Fit-qualified geometry anchoring with an unchanged native response tower."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import numpy as np

from stage28_response_common import cpu_array
from stage48_stability_experiment import load_torch
from stage48_stability_runtime import digest,head_output,make_head
from stage18_tracking_loss import tracking_loss
from stage52_branch_runtime import branches,frozen_digest
import stage53_geometry_common as c


def configure(head,arm):
    from stage52_branch_runtime import configure as configure_branch
    if arm not in c.ARMS: raise ValueError("Unknown preservation arm")
    return configure_branch(head,"geometry_only")


def objective(torch,head,samples,plan,arm,support):
    if arm not in c.ARMS or head.training: raise ValueError("Need native eval-mode geometry arm")
    selected=(0,2); device=next(head.parameters()).device
    features=np.stack([s["features"][list(selected)] for s in samples])
    target=np.stack([s["target"][list(selected)] for s in samples])
    output=head_output(head,torch.as_tensor(features,device=device).flatten(0,1))
    ordinary,stats=tracking_loss(torch,output,torch.as_tensor(target,device=device,dtype=output["score_map"].dtype).reshape(-1,4),True)
    refs=[c.teacher_state(s,v,plan["window"],plan["design"]) for s in samples for v in selected]
    cells=torch.tensor([r["cell"] for r in refs],device=device,dtype=torch.long)
    take=lambda value:value.flatten(2).gather(2,cells[:,None,None].expand(-1,2,1)).squeeze(-1)
    wh=take(output["size_map"]); off=take(output["offset_map"])
    teacher_wh=wh.new_tensor([r["size"] for r in refs]).detach()
    teacher_off=off.new_tensor([r["offset"] for r in refs]).detach()
    sides=off.new_tensor([r["side"] for r in refs])[:,None]
    eps=plan["design"]["epsilon"]
    log_change=(wh.clamp_min(eps)/teacher_wh.clamp_min(eps)).log()
    center_change=(off-teacher_off)/(sides*teacher_wh.clamp_min(eps))
    # Numerical deadband avoids a spurious L1 gradient from cached float noise.
    per_view=(log_change.abs()-eps).clamp_min(0).mean(1)+(center_change.abs()-eps).clamp_min(0).mean(1)
    gate=wh.new_tensor([r["reliable"] for r in refs])
    weights=gate*support["reliable_scale"] if arm=="anchor_reliable" else torch.ones_like(gate)
    penalty=(weights*per_view).mean()
    audit=dict(views=len(refs),reliable=int(gate.sum()),weight_mass=float(weights.sum()),penalty_sum=float((weights*per_view).detach().sum()))
    return ordinary+plan["design"]["anchor_weight"]*penalty,stats,penalty,audit


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
    torch=native.rt.torch; head=make_head(native.rt); groups,trainable=configure(head,arm)
    parameters=[p for p in head.parameters() if p.requires_grad]; initial=digest(head); frozen=frozen_digest(head,trainable)
    design=plan["training"]; control=c.control_record(plan,fold,seed)
    support,_=c.qualify_pool(pool,plan["window"],plan["design"])
    if dict(fold=fold,**support)!=plan["support"][fold]: raise ValueError("Fit-only anchor support changed")
    updates=design["smoke_updates"] if smoke else design["updates"]
    draws=np.random.default_rng(seed+100*fold).integers(0,len(pool),size=(updates,design["batch_pairs"]))
    draw_hash=c.fingerprint(draws.tolist())
    if initial!=control["initial_head_sha256"] or (not smoke and draw_hash!=control["draws_fingerprint"]):
        raise ValueError("Branch and reused control initialization/draws differ")
    original_receipt=c.prior.read_sealed(Path(control["checkpoint"]).with_suffix(".receipt.json"))
    if original_receipt["metadata"]["feature_sha256"]!=bindings: raise ValueError("Branch and control state pool differ")
    metadata=dict(plan_fingerprint=plan["fingerprint"],fold=fold,seed=seed,arm=arm,mode="smoke" if smoke else "full",
        updates=updates,fit=plan["folds"][fold]["fit"],held_out=plan["folds"][fold]["held_out"],
        feature_sha256=bindings,initial_head_sha256=initial,initial_frozen_sha256=frozen,trainable_names=trainable,
        draws_fingerprint=draw_hash,control_checkpoint_sha256=control["checkpoint_sha256"],anchor_support=support)
    directory=folder(root,fold,seed,arm,smoke)
    opt=torch.optim.AdamW(parameters,lr=design["learning_rate"],weight_decay=design["weight_decay"])
    # Same batch-one features before/after training check unchanged output towers.
    reference=[]; parity=0.
    with torch.no_grad():
        for v in (0,2):
            out=head_output(head,torch.as_tensor(pool[0]["features"][v:v+1],device=parameters[0].device))
            reference.append({k:cpu_array(out[k]).copy() for k in ("score_map","size_map","offset_map")})
            for key,label in (("score_map","raw"),("size_map","size"),("offset_map","offset")):
                parity=max(parity,float(np.max(np.abs(cpu_array(out[key])[0]-pool[0][label][v]))))
    if parity>1e-6: raise ValueError("Original feature/head parity failed")
    state=restore(torch,directory,metadata); start=0; history=[]; counts=dict(supervised=0,regression=0,center=0)
    maxima=dict(response=0.,geometry=0.); maximum_anchor_gradient=0.; anchor_counts=dict(views=0,reliable=0,weight_mass=0.,penalty_sum=0.)
    if state:
        head.load_state_dict(state["head"],strict=True); opt.load_state_dict(state["optimizer"])
        start=state["step"]; history=state["history"]; counts=state["counts"]; maxima=state["maximum_gradient"]
        anchor_counts=state["anchor_counts"]; maximum_anchor_gradient=state["maximum_anchor_gradient"]
        if frozen_digest(head,trainable)!=frozen: raise ValueError("Frozen branch changed in resumed state")
    for step in range(start+1,updates+1):
        opt.zero_grad(set_to_none=True)
        loss,stats,penalty,anchor=objective(torch,head,[pool[int(i)] for i in draws[step-1]],plan,arm,support)
        ag=torch.autograd.grad(penalty,parameters,retain_graph=True,allow_unused=True)
        anchor_norm=float(sum(g.detach().square().sum() for g in ag if g is not None).sqrt())
        maximum_anchor_gradient=max(maximum_anchor_gradient,anchor_norm)
        for key in anchor_counts: anchor_counts[key]+=anchor[key]
        loss.backward()
        norms={}
        for group,names in groups.items():
            gradients=[p.grad for name,p in head.named_parameters() if name in names and p.grad is not None]
            norms[group]=float(sum(g.detach().square().sum() for g in gradients).sqrt()) if gradients else 0.
            if any(name not in trainable and p.grad is not None for name,p in head.named_parameters()):
                raise ValueError("Gradient entered a frozen parameter")
            maxima[group]=max(maxima[group],norms[group])
        norm=float(torch.nn.utils.clip_grad_norm_(parameters,design["gradient_clip"]))
        if not np.isfinite([float(loss.detach()),norm,anchor_norm,*norms.values()]).all(): raise ValueError("Non-finite branch objective/gradient")
        opt.step()
        for key in counts: counts[key]+=int(stats[key+"_count"])
        if step==1 or step%design["save_every"]==0 or step==updates:
            history.append(dict(step=step,loss=float(loss.detach()),gradient_norm=norm,**counts,**{"gradient_"+k:v for k,v in norms.items()}))
            print("[branch fold{} seed{} {}] {}/{} loss={:.4f}".format(fold,seed,arm,step,updates,float(loss.detach())),flush=True)
        if step%design["save_every"]==0 or step==updates:
            if frozen_digest(head,trainable)!=frozen: raise ValueError("Frozen branch or BN buffer changed")
            save_step(torch,directory/("step{:06d}.pt".format(step)),dict(metadata=metadata,step=step,
                head={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},optimizer=opt.state_dict(),
                history=history,counts=counts,maximum_gradient=maxima,anchor_counts=anchor_counts,maximum_anchor_gradient=maximum_anchor_gradient))
    active="geometry"; inactive="response"
    if digest(head)==initial or maxima[active]<=0 or maxima[inactive]!=0 or maximum_anchor_gradient<=0 or frozen_digest(head,trainable)!=frozen:
        raise ValueError("Branch intervention/update audit failed")
    if counts["supervised"]!=updates*design["batch_pairs"]*2 or (not smoke and counts!=control["counts"]):
        raise ValueError("Branch and control supervision counts differ")
    map_error=0.; frozen_keys=("score_map",)
    with torch.no_grad():
        for index,v in enumerate((0,2)):
            out=head_output(head,torch.as_tensor(pool[0]["features"][v:v+1],device=parameters[0].device))
            for key in frozen_keys: map_error=max(map_error,float(np.max(np.abs(cpu_array(out[key])-reference[index][key]))))
    if map_error>plan["design"]["frozen_map_atol"]: raise ValueError("Frozen output tower changed")
    path=directory/("step{:06d}.pt".format(updates))
    return dict(fold=fold,seed=seed,arm=arm,checkpoint=str(path.resolve()),checkpoint_sha256=c.file_hash(path),
        receipt_sha256=c.file_hash(path.with_suffix(".receipt.json")),initial_head_sha256=initial,final_head_sha256=digest(head),
        initial_frozen_sha256=frozen,final_frozen_sha256=frozen_digest(head,trainable),trainable_names=trainable,
        trainable_parameters=sum(p.numel() for p in parameters),head_parameters=sum(p.numel() for p in head.parameters()),
        draws_fingerprint=draw_hash,maximum_gradient=maxima,maximum_frozen_map_error=map_error,
        initial_map_error=parity,counts=counts,history=history,anchor_counts=anchor_counts,maximum_anchor_gradient=maximum_anchor_gradient,anchor_support=support)
