"""Four actual public SGLA calls; only the selected output is accepted."""
from __future__ import annotations
import time
import numpy as np
from stage27_uav_common import checked_image
import stage60_state_runtime as previous
import stage61_search_common as c

def run_actor(observer, frames, initial_box, dataset_root, policy, plan, frame_limit=None, image_reader=checked_image):
    if policy not in c.POLICIES: raise ValueError("Unknown Stage-61 policy")
    observer.initialize(image_reader(dataset_root,frames[0]),initial_box)
    history=[np.asarray(initial_box,float).copy()]; rows=[]
    limit = min(len(frames)-1,frame_limit) if frame_limit is not None else len(frames)-1
    for frame in range(1,limit+1):
        image=image_reader(dataset_root,frames[frame]); observer.native.synchronize(); start=time.perf_counter()
        # q0 is executed exactly once, and retained as both a candidate and the control's proposal source.
        first=previous.observe(observer,image,history[-1],frame,plan)
        parents=(c.recrop_parents(history[-1],first["boxes"]) if policy=="recrop" else
                 c.prior.history_parents(history,plan["design"]))
        if policy=="current": parents=parents[:1]
        observations=[first]+[previous.observe(observer,image,p,frame,plan) for p in parents[1:]]
        features=np.array([previous.feature_vector(r,history,plan) for r in observations])
        chosen,scores=c.prior.choose("current" if policy=="current" else "response",features,{},plan["design"])
        observer.native.synchronize(); elapsed=(time.perf_counter()-start)*1000
        output=np.asarray(observations[chosen]["boxes"],float).copy()
        rows.append(dict(frame_id=frame,boxes=output,parents=parents[chosen].copy(),history_latest=history[-1].copy(),
            candidate_parents=parents,candidate_boxes=np.array([r["boxes"] for r in observations]),features=features,
            scores=np.asarray(scores)[:len(parents)],selected=chosen,crop=np.array([r["crop"] for r in observations]),
            crop_sha256=np.array([r["crop_sha256"] for r in observations]),image_hw=image.shape[:2],
            layer=np.array([r["layer"] for r in observations]),block_calls=np.array([r["block_calls"] for r in observations]),
            head_calls=np.array([r["head_calls"] for r in observations]),selector_calls=np.array([r["selector_calls"] for r in observations]),
            decode_error=max(r["decode_error"] for r in observations),pipeline_ms=elapsed))
        history.append(output); history=history[-plan["design"]["history"]:]
        observer.rt.tracker.state=output.tolist()
        if frame%500==0: print("[{}] {}/{}".format(policy,frame,limit),flush=True)
    data={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    audit(data,dict(frames=frames,initial_box=initial_box),plan,policy)
    return data

def audit(data, entry, plan, policy, reference=None):
    count=len(data["frame_id"])
    c.prior.audit_trace(data,count,1 if policy=="current" else 4)
    if not 0<count<=len(entry["frames"])-1: raise ValueError("Bad trajectory length")
    history=[np.asarray(entry["initial_box"],float).copy()]
    for i in range(count):
        expected=(c.recrop_parents(history[-1],data["candidate_boxes"][i,0]) if policy=="recrop" else
                  c.prior.history_parents(history,plan["design"]))
        if policy=="current": expected=expected[:1]
        np.testing.assert_array_equal(data["candidate_parents"][i],expected)
        np.testing.assert_array_equal(data["history_latest"][i],history[-1])
        selected,scores=c.prior.choose("current" if policy=="current" else "response",data["features"][i],{},plan["design"])
        if selected!=int(data["selected"][i]): raise ValueError("Selection mismatch")
        np.testing.assert_array_equal(data["scores"][i],scores[:len(expected)])
        for j,p in enumerate(expected):
            crop=c.crop_geometry(p,data["image_hw"][i],plan["search_factor"],plan["search_size"])[0]
            np.testing.assert_array_equal(data["crop"][i,j],crop)
        history.append(data["boxes"][i]); history=history[-plan["design"]["history"]:]
    if not np.isfinite(data["decode_error"]).all() or np.max(data["decode_error"])>plan["design"]["box_atol"]:
        raise ValueError("Decode audit failed")
    error=0.
    if reference is not None:
        error=float(np.max(np.abs(data["boxes"]-reference["boxes"][:count])))
        if error>plan["design"]["source_box_atol"]: raise ValueError("Source control replay is not exact")
        layers=reference["layer"][:count]
        if layers.ndim==1: layers=layers[:,None]
        np.testing.assert_array_equal(data["layer"],layers)
        if "candidate_parents" in reference:
            np.testing.assert_array_equal(data["candidate_parents"],reference["candidate_parents"][:count])
            np.testing.assert_array_equal(data["candidate_boxes"],reference["candidate_boxes"][:count])
            np.testing.assert_array_equal(data["crop_sha256"],reference["crop_sha256"][:count])
            np.testing.assert_array_equal(data["features"],reference["features"][:count])
        elif "peak" in reference:
            if np.max(np.abs(data["features"][:,0,0]-reference["peak"][:count]))>1e-5:
                raise ValueError("Native response changed")
    return error
