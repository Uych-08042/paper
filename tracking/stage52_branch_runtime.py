"""Exact CENTER parameter scopes and matched branch-only ordinary training."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import numpy as np

from stage28_response_common import cpu_array
from stage48_stability_experiment import load_torch
from stage48_stability_runtime import digest,head_output,make_head
from stage51_state_runtime import objective
import stage52_branch_common as c


def branches(head):
    expected={"conv{}_{}".format(i,s) for i in range(1,6) for s in ("ctr","size","offset")}
    if set(dict(head.named_children()))!=expected: raise ValueError("Need the original three independent CENTER towers")
    result=dict(response=[],geometry=[])
    for name,_ in head.named_parameters():
        root=name.split(".")[0]
        if root not in expected: raise ValueError("Unknown shared/head parameter: "+name)
        result["response" if root.endswith("_ctr") else "geometry"].append(name)
    if not all(result.values()): raise ValueError("Empty CENTER parameter group")
    return {k:sorted(v) for k,v in result.items()}


def configure(head,arm):
    groups=branches(head)
    if arm not in c.ARMS: raise ValueError("Unknown branch arm")
    chosen=groups["response" if arm=="response_only" else "geometry"]
    head.eval().requires_grad_(False)
    for name,p in head.named_parameters(): p.requires_grad_(name in chosen)
    return groups,chosen


def frozen_digest(head,trainable):
    h=hashlib.sha256()
    for name,value in sorted(head.state_dict().items()):
        if name in trainable: continue
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


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
    design=plan["training"]; joint=c.joint_record(plan,fold,seed)
    updates=design["smoke_updates"] if smoke else design["updates"]
    draws=np.random.default_rng(seed+100*fold).integers(0,len(pool),size=(updates,design["batch_pairs"]))
    draw_hash=c.fingerprint(draws.tolist())
    if initial!=joint["initial_head_sha256"] or (not smoke and draw_hash!=joint["draws_fingerprint"]):
        raise ValueError("Branch and reused joint initialization/draws differ")
    original_receipt=c.prior.read_sealed(Path(joint["checkpoint"]).with_suffix(".receipt.json"))
    if original_receipt["metadata"]["feature_sha256"]!=bindings: raise ValueError("Branch and joint state pool differ")
    metadata=dict(plan_fingerprint=plan["fingerprint"],fold=fold,seed=seed,arm=arm,mode="smoke" if smoke else "full",
        updates=updates,fit=plan["folds"][fold]["fit"],held_out=plan["folds"][fold]["held_out"],
        feature_sha256=bindings,initial_head_sha256=initial,initial_frozen_sha256=frozen,trainable_names=trainable,
        draws_fingerprint=draw_hash,joint_checkpoint_sha256=joint["checkpoint_sha256"])
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
    maxima=dict(response=0.,geometry=0.)
    if state:
        head.load_state_dict(state["head"],strict=True); opt.load_state_dict(state["optimizer"])
        start=state["step"]; history=state["history"]; counts=state["counts"]; maxima=state["maximum_gradient"]
        if frozen_digest(head,trainable)!=frozen: raise ValueError("Frozen branch changed in resumed state")
    for step in range(start+1,updates+1):
        opt.zero_grad(set_to_none=True)
        loss,stats=objective(torch,head,[pool[int(i)] for i in draws[step-1]],"state_mix")
        loss.backward()
        norms={}
        for group,names in groups.items():
            gradients=[p.grad for name,p in head.named_parameters() if name in names and p.grad is not None]
            norms[group]=float(sum(g.detach().square().sum() for g in gradients).sqrt()) if gradients else 0.
            if any(name not in trainable and p.grad is not None for name,p in head.named_parameters()):
                raise ValueError("Gradient entered a frozen parameter")
            maxima[group]=max(maxima[group],norms[group])
        norm=float(torch.nn.utils.clip_grad_norm_(parameters,design["gradient_clip"]))
        if not np.isfinite([float(loss.detach()),norm,*norms.values()]).all(): raise ValueError("Non-finite branch objective/gradient")
        opt.step()
        for key in counts: counts[key]+=int(stats[key+"_count"])
        if step==1 or step%design["save_every"]==0 or step==updates:
            history.append(dict(step=step,loss=float(loss.detach()),gradient_norm=norm,**counts,**{"gradient_"+k:v for k,v in norms.items()}))
            print("[branch fold{} seed{} {}] {}/{} loss={:.4f}".format(fold,seed,arm,step,updates,float(loss.detach())),flush=True)
        if step%design["save_every"]==0 or step==updates:
            if frozen_digest(head,trainable)!=frozen: raise ValueError("Frozen branch or BN buffer changed")
            save_step(torch,directory/("step{:06d}.pt".format(step)),dict(metadata=metadata,step=step,
                head={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},optimizer=opt.state_dict(),
                history=history,counts=counts,maximum_gradient=maxima))
    active="response" if arm=="response_only" else "geometry"; inactive="geometry" if active=="response" else "response"
    if digest(head)==initial or maxima[active]<=0 or maxima[inactive]!=0 or frozen_digest(head,trainable)!=frozen:
        raise ValueError("Branch intervention/update audit failed")
    if counts["supervised"]!=updates*design["batch_pairs"]*2 or (not smoke and counts!=joint["counts"]):
        raise ValueError("Branch and joint supervision counts differ")
    map_error=0.; frozen_keys=("size_map","offset_map") if arm=="response_only" else ("score_map",)
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
        initial_map_error=parity,counts=counts,history=history)
