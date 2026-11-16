"""Public RGB appearance cues for cached q0/policy action pairs; no network inference."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import stage118_nested_commit as gate
import stage115_q0_continuation as branch
import stage117_sampled_q0 as sampled
from stage27_uav_common import checked_image, inside

DESIGN=dict(bins=8,patch_side=16,min_coverage=.5,recent_lookback=30,minimum_norm=1e-12)
FEATURES=tuple(f"{ref}_{signal}_{field}" for ref in ("template","recent") for signal in ("hist","ncc")
    for field in ("q0","policy","advantage","q0_valid","policy_valid"))+("q0_coverage","policy_coverage","recent_exists","recent_age")


def describe(image,box):
    image=np.asarray(image);b=np.asarray(box,float)
    if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8 or b.shape!=(4,) or not np.isfinite(b).all() or (b[2:]<=0).any():raise ValueError("Invalid public RGB/ROI")
    h,w=image.shape[:2];left,top=np.maximum(b[:2],0.);right,bottom=np.minimum(b[:2]+b[2:],[w,h])
    coverage=float(np.maximum(0.,right-left)*np.maximum(0.,bottom-top)/b[2:].prod())
    if coverage<DESIGN["min_coverage"]:return dict(hist=None,ncc=None,coverage=coverage)
    x0,y0=np.floor([left,top]).astype(int);x1,y1=np.ceil([right,bottom]).astype(int)
    pixels=image[y0:y1,x0:x1].reshape(-1,3);bins=DESIGN["bins"]
    index=pixels.astype(int)*bins//256
    histogram=np.sqrt(np.concatenate([np.bincount(index[:,k],minlength=bins) for k in range(3)])/(3*len(pixels)))
    # Fixed bilinear resampling of RGB luminance; no Pillow/OpenCV resize dependency.
    side=DESIGN["patch_side"];xx=np.clip(left+(np.arange(side)+.5)*(right-left)/side-.5,0,w-1)
    yy=np.clip(top+(np.arange(side)+.5)*(bottom-top)/side-.5,0,h-1)
    ix=np.floor(xx).astype(int);iy=np.floor(yy).astype(int);jx=np.minimum(ix+1,w-1);jy=np.minimum(iy+1,h-1)
    wx=xx-ix;wy=yy-iy
    patch=((1-wy)[:,None,None]*((1-wx)[None,:,None]*image[iy[:,None],ix]+wx[None,:,None]*image[iy[:,None],jx])
        +wy[:,None,None]*((1-wx)[None,:,None]*image[jy[:,None],ix]+wx[None,:,None]*image[jy[:,None],jx]))
    gray=(patch@np.array([.299,.587,.114])).ravel()/255.;gray-=gray.mean();norm=float(np.linalg.norm(gray))
    return dict(hist=histogram,ncc=gray/norm if norm>DESIGN["minimum_norm"] else None,coverage=coverage)


def recent_frame(trace,frame):
    start=max(1,frame-DESIGN["recent_lookback"])
    for f in range(frame-1,start-1,-1):
        if bool(trace["memory_write"][f-1]):return f
    return -1


def make_task(uid,frame,point,trace,entry):
    """Whitelist: initial SOT box, frozen public boxes, past trust and image metadata."""
    frame=int(frame)
    if int(np.asarray(point["frame_id"])[0])!=frame:raise ValueError("Wrong public point frame")
    bank=np.asarray(point["candidate_boxes"])[0];selected=int(np.asarray(point["selected"])[0])
    if bank.shape!=(4,4) or selected not in range(1,4) or np.array_equal(bank[0],bank[selected]):raise ValueError("Need the frozen physical action pair")
    if not np.array_equal(bank,trace["candidate_boxes"][frame-1]) or selected!=int(trace["selected"][frame-1]):raise ValueError("Shared point drifted from source history")
    recent=recent_frame(trace,frame)
    return dict(uid=uid,frame=frame,boxes=[bank[0].tolist(),bank[selected].tolist()],initial_box=list(entry["initial_box"]),
        recent_frame=recent,recent_box=np.asarray(trace["boxes"][recent-1]).tolist() if recent>=1 else None,
        images={str(f):entry["frames"][f] for f in sorted({0,frame}|({recent} if recent>=1 else set()))})


def appearance_vector(template,current,recent,task):
    frame=int(task["frame"]);past_frame=int(task["recent_frame"])
    if frame<1 or not (past_frame==-1 or max(1,frame-DESIGN["recent_lookback"])<=past_frame<frame):raise ValueError("Current/future or stale appearance reference")
    reference=describe(template,task["initial_box"]);queries=[describe(current,b) for b in task["boxes"]]
    past=describe(recent,task["recent_box"]) if task["recent_frame"]>=1 else dict(hist=None,ncc=None)
    result=[]
    for ref in (reference,past):
        for signal in ("hist","ncc"):
            valid=[ref[signal] is not None and q[signal] is not None for q in queries]
            values=[float(np.clip(ref[signal]@q[signal],-1.,1.)) if ok else 0. for q,ok in zip(queries,valid)]
            result.extend((values[0],values[1],values[0]-values[1],float(valid[0]),float(valid[1])))
    result.extend((queries[0]["coverage"],queries[1]["coverage"],float(task["recent_frame"]>=1),
        (task["frame"]-task["recent_frame"])/DESIGN["recent_lookback"] if task["recent_frame"]>=1 else 0.))
    x=np.asarray(result,float)
    if x.shape!=(len(FEATURES),) or not np.isfinite(x).all():raise ValueError("Malformed appearance evidence")
    return x


def collect(root,tasks,reader=checked_image):
    frames={}
    for task in tasks:
        for f,item in task["images"].items():
            if f in frames and frames[f]!=item:raise ValueError("Image identity changed within context")
            frames[f]=item
    images={f:reader(root,item) for f,item in frames.items()}
    x=[appearance_vector(images['0'],images[str(t["frame"])],images.get(str(t["recent_frame"])),t) for t in tasks]
    return dict(x=np.asarray(x),image_reads=np.asarray(len(images)),recent_frame=np.array([t["recent_frame"] for t in tasks]))


def prepare(records,r113,p113,models,r115,p115,r117,p117,load_entry,image_root):
    s=gate.s;jobs={branch.key(j):j for j in p113["jobs"] if j["arm"]=="generated_shared"};contexts={};images={}
    anchors={"stage115":{a["anchor_id"]:a for a in p115["anchors"]},"stage117":{a["anchor_id"]:a for a in p117["anchors"]}}
    for context in sorted({r["context"] for r in records}):
        job=jobs[context];trace=branch.required_trace(r113,p113,job,models);entry=load_entry(job["spec"]);tasks=[]
        if entry["sequence"]!=job["spec"]["unit"]["sequence"]:raise ValueError("Wrong public image entry")
        for r in sorted([r for r in records if r["context"]==context],key=lambda r:r["uid"]):
            origin=r["origin"];source,p=(r115,p115) if origin=="stage115" else (r117,p117);a=anchors[origin][r["anchor_id"]]
            point=gate.c.read_cache(source/"points"/(a["anchor_id"]+".npz"),dict(plan_fingerprint=p["fingerprint"],anchor=a,kind="shared_point"))
            if point is None:raise ValueError("Missing original matched public point")
            task=make_task(r["uid"],r["frame"],point,trace,entry);tasks.append(task)
            for item in task["images"].values():images[str(inside(image_root,item["file"]))]=item["sha256"]
        contexts[context]=tasks
    return contexts,images
