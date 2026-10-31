"""Actors accept images, initialization and their own prediction history only."""
from __future__ import annotations
import time
import numpy as np
from stage27_uav_common import checked_image, crop_geometry, tracking_iou
from stage28_response_common import dense_boxes
import stage60_state_common as c

def observe(observer, image, parent, frame, plan):
    row=observer.capture(image,parent,frame)
    crop,usable,factor=crop_geometry(parent,image.shape[:2],plan["search_factor"],plan["search_size"])
    cell=int(np.argmax(row["raw"]*np.asarray(plan["window"],np.float32)))
    dense=dense_boxes(row["size"],row["offset"],parent,factor,plan["search_size"],image.shape[:2])
    error=float(np.max(np.abs(dense[cell]-row["boxes"])))
    if error>plan["design"]["box_atol"]: raise ValueError("Native decode changed")
    c.prior.native_calls(row["block_calls"],row["layer"],row["head_calls"],row["selector_calls"])
    return dict(row,crop=crop,usable=usable,parent=np.asarray(parent,float).copy(),image_hw=np.array(image.shape[:2]),decode_error=error)

def feature_vector(row, history, plan):
    """Explicit GT-free whitelist; no video identity, frame index or source head id."""
    h=np.asarray(history,float)[-plan["design"]["history"]:]; latest=h[-1]
    parent=np.asarray(row["parent"],float); output=np.asarray(row["boxes"],float)
    center=lambda b:b[:2]+b[2:]/2
    unit=np.sqrt(latest[2:].prod()); raw32=np.asarray(row["raw"],np.float32)
    # Native ties/argmax use the float32 product, before float64 statistics.
    weighted=(raw32*np.asarray(plan["window"],np.float32)).astype(float)
    raw=raw32.astype(float); side=raw.shape[0]
    rsort=np.sort(raw.ravel()); wsort=np.sort(weighted.ravel())
    raw_cell=np.array(np.unravel_index(int(raw.argmax()),raw.shape),float)
    cell=np.array(np.unravel_index(int(weighted.argmax()),raw.shape),float)
    q=np.maximum(raw.ravel(),1e-8); q=q/q.sum()
    out_delta=(center(output)-center(latest))/unit; parent_delta=(center(parent)-center(latest))/unit
    out_scale=np.log(output[2:]/latest[2:]); parent_scale=np.log(parent[2:]/latest[2:])
    center_move=float(np.linalg.norm(parent_delta)); scale_move=float(np.abs(parent_scale).mean())
    speed=float(np.linalg.norm(center(h[-1])-center(h[-2]))/unit) if len(h)>1 else 0.
    values=np.r_[rsort[-1],wsort[-1],wsort[-1]-wsort[-2],rsort[-1]-rsort[-2],
        -(q*np.log(q)).sum()/np.log(len(q)),(wsort[-1]-weighted.mean())/(weighted.std()+1e-6),
        np.linalg.norm(raw_cell-cell)/side,np.linalg.norm(cell-(side-1)/2)/side,
        out_delta,out_scale,parent_delta,parent_scale,
        parent[2:]/np.asarray(row["image_hw"])[[1,0]],
        1.-row["usable"][2:].prod()/row["crop"][2:].prod(),speed*np.linalg.norm(out_delta),np.std(np.log(h[:,2:]),axis=0).mean()*scale_move,
        center_move*scale_move,np.linalg.norm(out_delta)*scale_move,wsort[-1]*np.abs(out_scale).mean()]
    if values.shape!=(len(c.FEATURES),) or not np.isfinite(values).all(): raise ValueError("Invalid causal features")
    return np.clip(values,-plan["design"]["feature_clip"],plan["design"]["feature_clip"])

def run_actor(observer, frames, initial_box, dataset_root, policy, models, plan,
              frame_limit=None, image_reader=checked_image):
    """Each selection sees four current probes; only its chosen box enters history."""
    observer.initialize(image_reader(dataset_root,frames[0]),initial_box)
    history=[np.asarray(initial_box,float).copy()]; rows=[]
    limit=min(len(frames)-1,frame_limit) if frame_limit is not None else len(frames)-1
    for frame in range(1,limit+1):
        image=image_reader(dataset_root,frames[frame]); observer.native.synchronize(); started=time.perf_counter()
        parents=c.history_parents(history,plan["design"])
        if policy=="current": parents=parents[:1]
        observations=[observe(observer,image,p,frame,plan) for p in parents]
        features=np.array([feature_vector(r,history,plan) for r in observations])
        choice,values=c.choose(policy,features,models,plan["design"])
        observer.native.synchronize(); elapsed=(time.perf_counter()-started)*1000
        chosen=observations[choice]; output=np.asarray(chosen["boxes"],float).copy()
        rows.append(dict(frame_id=frame,boxes=output,parents=parents[choice].copy(),history_latest=history[-1].copy(),
            candidate_parents=parents,candidate_boxes=np.array([r["boxes"] for r in observations]),features=features,
            scores=np.asarray(values)[:len(parents)],selected=choice,crop=np.array([r["crop"] for r in observations]),
            crop_sha256=np.array([r["crop_sha256"] for r in observations]),image_hw=image.shape[:2],
            layer=np.array([r["layer"] for r in observations]),block_calls=np.array([r["block_calls"] for r in observations]),
            head_calls=np.array([r["head_calls"] for r in observations]),selector_calls=np.array([r["selector_calls"] for r in observations]),
            decode_error=max(r["decode_error"] for r in observations),pipeline_ms=elapsed))
        history.append(output); history=history[-plan["design"]["history"]:]
        # Alternative probes never become accepted history, hidden state or template.
        observer.rt.tracker.state=output.tolist()
        if frame%500==0: print("[{}] {}/{}".format(policy,frame,limit),flush=True)
    data={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    c.audit_trace(data,limit,1 if policy=="current" else 4)
    return data

def anchor_history(trace, initial_box, frame, design):
    previous=np.vstack([np.asarray(initial_box,float),trace["boxes"][:frame-1]])
    return previous[-design["history"]:]

def collect_anchor(observer, entry, baseline, frame, plan, image_reader=checked_image):
    """Offline fit labels are attached after every candidate branch has run."""
    history=anchor_history(baseline,entry["initial_box"],frame,plan["design"])
    parents=c.history_parents(history,plan["design"]); horizon=plan["design"]["horizon"]
    images={f:image_reader(plan["root"],entry["frames"][f]) for f in [0]+list(range(frame,frame+horizon))}
    branches=[]; first=[]
    for parent in parents:
        observer.initialize(images[0],entry["initial_box"]); rows=[]; own=parent.copy()
        for f in range(frame,frame+horizon):
            result=observe(observer,images[f],own,f,plan); rows.append(result); own=np.asarray(result["boxes"],float).copy()
        first.append(rows[0]); branches.append(rows)
    boxes=np.array([[r["boxes"] for r in b] for b in branches])
    np.testing.assert_allclose(boxes[0],baseline["boxes"][frame-1:frame-1+horizon],atol=plan["design"]["box_atol"],rtol=0,
                               err_msg="Fit current-parent continuation did not reproduce its source")
    # No GT has been accessed above this point.
    gt=np.asarray(entry["ground_truth"],float)[frame:frame+horizon]
    overlaps=tracking_iou(boxes,gt[None])
    data=dict(features=np.array([feature_vector(r,history,plan) for r in first]),history=history,
        initial_parents=parents,boxes=boxes,parents=np.array([[r["parent"] for r in b] for b in branches]),
        immediate_returns=overlaps[:,0],future_returns=overlaps.mean(1),iou=overlaps,
        frame_id=np.arange(frame,frame+horizon),decode_error=np.array([[r["decode_error"] for r in b] for b in branches]),
        crop=np.array([r["crop"] for r in first]),crop_sha256=np.array([r["crop_sha256"] for r in first]))
    for k in ("layer","block_calls","head_calls","selector_calls"):
        data[k]=np.array([[r[k] for r in b] for b in branches])
    audit_anchor(data,entry,baseline,frame,plan)
    return data

def audit_anchor(data, entry, baseline, frame, plan):
    horizon=plan["design"]["horizon"]
    if data["boxes"].shape!=(4,horizon,4) or data["features"].shape!=(4,len(c.FEATURES)):
        raise ValueError("Incomplete fit anchor")
    c.audit_calls(data,(4,horizon))
    if not np.array_equal(data["frame_id"],np.arange(frame,frame+horizon)): raise ValueError("Changed fit frames")
    history=anchor_history(baseline,entry["initial_box"],frame,plan["design"])
    np.testing.assert_array_equal(data["history"],history)
    np.testing.assert_array_equal(data["initial_parents"],c.history_parents(history,plan["design"]))
    np.testing.assert_array_equal(data["parents"][:,0],data["initial_parents"])
    if horizon>1: np.testing.assert_array_equal(data["parents"][:,1:],data["boxes"][:,:-1])
    np.testing.assert_allclose(data["boxes"][0],baseline["boxes"][frame-1:frame-1+horizon],atol=plan["design"]["box_atol"],rtol=0)
    overlap=tracking_iou(data["boxes"],np.asarray(entry["ground_truth"],float)[None,frame:frame+horizon])
    for key,value in (("iou",overlap),("immediate_returns",overlap[:,0]),("future_returns",overlap.mean(1))):
        np.testing.assert_array_equal(data[key],value)
    if not np.isfinite(data["features"]).all() or np.max(data["decode_error"])>plan["design"]["box_atol"]:
        raise ValueError("Invalid fit branch")
