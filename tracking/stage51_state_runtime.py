"""Fit-only donor rollouts and ordinary two-view head training."""
from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path
import numpy as np

from stage18_tracking_loss import tracking_loss
from stage27_uav_common import checked_image, crop_geometry
from stage28_response_common import cpu_array
from stage46_motion_runtime import state_digest
from stage48_stability_experiment import load_torch
from stage48_stability_runtime import FeatureCapture, digest, head_output, make_head
import stage51_state_common as c


@contextmanager
def installed_head(native,state):
    """Change only head weights temporarily; audit the whole installed network."""
    original=copy.deepcopy(native.rt.net.box_head.state_dict()); before=native.digest
    try:
        native.rt.net.box_head.load_state_dict(state,strict=True)
        native.digest=state_digest(native.rt.net)
        yield
        native.assert_frozen()
    finally:
        native.rt.net.box_head.load_state_dict(original,strict=True)
        native.digest=before
        native.assert_frozen()


def load_donor(native,plan,fold):
    record=plan["donors"][fold]; path=Path(record["checkpoint"])
    c.verify_files({str(path):record["checkpoint_sha256"],str(path.with_suffix(".receipt.json")):record["receipt_sha256"]})
    payload=load_torch(native.rt.torch,path); m=payload["metadata"]
    if (m["fold"]!=fold or m["fit"]!=plan["folds"][fold]["fit"] or m["held_out"]!=plan["folds"][fold]["held_out"] or
        m["arm"]!=c.DESIGN["donor_arm"] or m["seed"]!=c.DESIGN["donor_seed"] or
        m["mode"]!="full" or payload["step"]!=c.prior.DESIGN["updates"] or
        m["plan_fingerprint"]!=plan["source_stage48_plan_fingerprint"]):
        raise ValueError("Donor checkpoint training scope changed")
    head=make_head(native.rt); head.load_state_dict(payload["head"],strict=True)
    if digest(head)!=record["final_head_sha256"]: raise ValueError("Donor head weights changed")
    return payload["head"]


def collect_features(native,plan,entry,trace,image_reader=checked_image):
    """Parents chosen by frozen fit-only rollout; GT attached only after capture."""
    observer=FeatureCapture(native); rows=[]
    try:
        observer.initialize(image_reader(plan["root"],entry["frames"][0]),entry["initial_box"])
        for frame in c.prior.sample_frames(entry):
            image=image_reader(plan["root"],entry["frames"][frame]); parent=trace["previous"][frame-1]
            result=observer.capture(image,parent,frame)
            crop,_,_=crop_geometry(parent,image.shape[:2],plan["search_factor"],plan["search_size"])
            gt=np.asarray(entry["ground_truth"][frame],float)
            rows.append(dict(frame_id=frame,features=result["features"],parents=parent,crop=crop,
                target=np.r_[(gt[:2]-crop[:2])/crop[2],gt[2:]/crop[2]],
                **{k:result[k] for k in ("raw","size","offset","boxes","layer")}))
        observer.assert_frozen()
    finally:
        observer.handle.remove(); observer.pre_handle.remove()
    return {k:np.asarray([r[k] for r in rows]) for k in rows[0]}


def objective(torch,head,samples,arm):
    if arm not in c.ARMS or head.training: raise ValueError("Need fixed two-view arm and frozen BN buffers")
    views=[0,0] if arm=="native_replay" else [0,2]
    features=np.stack([s["features"][views] for s in samples])
    target=np.stack([s["target"][views] for s in samples])
    device=next(head.parameters()).device
    output=head_output(head,torch.as_tensor(features,device=device).flatten(0,1))
    loss,stats=tracking_loss(torch,output,torch.as_tensor(target,device=device,dtype=output["score_map"].dtype).reshape(-1,4),True)
    return loss,stats


def directory(root,fold,seed,arm,smoke=False):
    return Path(root)/("smoke/training" if smoke else "training")/("seed{}".format(seed))/("fold{}".format(fold))/arm


def save_step(torch,path,state):
    import os
    import tempfile
    if path.with_suffix(".receipt.json").exists(): raise ValueError("Never overwrite committed Stage-51 training state")
    if path.exists() and load_torch(torch,path).get("metadata")!=state["metadata"]: raise ValueError("Unidentified orphan training state")
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix=".tmp",delete=False) as h: tmp=h.name
    try:
        torch.save(state,tmp); os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    c.publish_identical(path.with_suffix(".receipt.json"),c.seal(dict(schema=c.SCHEMA,metadata=state["metadata"],step=state["step"],sha256=c.file_hash(path))))


def restore(torch,folder,metadata):
    latest=None
    for receipt in sorted(folder.glob("step*.receipt.json")):
        r=c.read_sealed(receipt); path=folder/("step{:06d}.pt".format(r["step"]))
        if (r.get("metadata")!=metadata or not 0<r["step"]<=metadata["updates"] or receipt!=path.with_suffix(".receipt.json") or
            not path.is_file() or c.file_hash(path)!=r["sha256"]): raise ValueError("Committed training checkpoint changed")
        latest=load_torch(torch,path)
        if latest.get("metadata")!=metadata or latest.get("step")!=r["step"]: raise ValueError("Training source/budget changed")
    return latest


def train_arm(native,plan,root,pool,bindings,fold,seed,arm,smoke=False):
    torch=native.rt.torch; head=make_head(native.rt); initial=digest(head); buffers=digest(head,True)
    updates=c.DESIGN["smoke_updates"] if smoke else c.DESIGN["updates"]
    draws=np.random.default_rng(seed+100*fold).integers(0,len(pool),size=(updates,c.DESIGN["batch_pairs"]))
    metadata=dict(plan_fingerprint=plan["fingerprint"],fold=fold,seed=seed,arm=arm,mode="smoke" if smoke else "full",
        updates=updates,fit=plan["folds"][fold]["fit"],held_out=plan["folds"][fold]["held_out"],
        initial_head_sha256=initial,feature_sha256=bindings,draws_fingerprint=c.fingerprint(draws.tolist()))
    folder=directory(root,fold,seed,arm,smoke)
    opt=torch.optim.AdamW(head.parameters(),lr=c.DESIGN["learning_rate"],weight_decay=c.DESIGN["weight_decay"])
    saved=restore(torch,folder,metadata); history=[]; counts=dict(supervised=0,regression=0,center=0)
    parity=0.; start=0
    if saved:
        head.load_state_dict(saved["head"],strict=True); opt.load_state_dict(saved["optimizer"])
        start=saved["step"]; history=saved["history"]; counts=saved["counts"]; parity=saved["initial_map_error"]
    else:
        # Batch-one parity at all three views, including donor state features.
        sample=pool[0]
        with torch.no_grad():
            for v in range(3):
                out=head_output(head,torch.as_tensor(sample["features"][v:v+1],device=next(head.parameters()).device))
                for key,label in (("score_map","raw"),("size_map","size"),("offset_map","offset")):
                    parity=max(parity,float(np.max(np.abs(cpu_array(out[key])[0]-sample[label][v]))))
        if parity>1e-6: raise ValueError("Original head fails native/own-state feature parity")
    for step in range(start+1,updates+1):
        opt.zero_grad(set_to_none=True)
        loss,stats=objective(torch,head,[pool[int(i)] for i in draws[step-1]],arm)
        loss.backward(); norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),c.DESIGN["gradient_clip"]))
        if not np.isfinite([float(loss.detach()),norm]).all(): raise ValueError("Non-finite ordinary loss/gradient")
        opt.step()
        for key in counts: counts[key]+=int(stats[key+"_count"])
        if step==1 or step%c.DESIGN["save_every"]==0 or step==updates:
            history.append(dict(step=step,loss=float(loss.detach()),gradient_norm=norm,**counts))
            print("[state source fold{} seed{} {}] {}/{} loss={:.4f}".format(fold,seed,arm,step,updates,float(loss.detach())),flush=True)
        if step%c.DESIGN["save_every"]==0 or step==updates:
            if digest(head,True)!=buffers: raise ValueError("Frozen head BN buffers changed")
            save_step(torch,folder/("step{:06d}.pt".format(step)),dict(metadata=metadata,step=step,
                head={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},optimizer=opt.state_dict(),
                history=history,counts=counts,initial_map_error=parity))
    if digest(head)==initial or digest(head,True)!=buffers: raise ValueError("No head update or BN changed")
    if counts["supervised"]!=updates*c.DESIGN["batch_pairs"]*2: raise ValueError("Two-view supervision budget changed")
    path=folder/("step{:06d}.pt".format(updates))
    return dict(fold=fold,seed=seed,arm=arm,checkpoint=str(path.resolve()),checkpoint_sha256=c.file_hash(path),
        receipt_sha256=c.file_hash(path.with_suffix(".receipt.json")),initial_head_sha256=initial,final_head_sha256=digest(head),
        buffers_unchanged=True,draws_fingerprint=metadata["draws_fingerprint"],initial_map_error=parity,counts=counts,history=history)
