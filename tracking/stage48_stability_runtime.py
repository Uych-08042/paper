"""Original-head features, fit-only objectives and unchanged native inference."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys

import numpy as np

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage18_tracking_loss import tracking_loss
from stage27_uav_common import checked_image, crop_geometry, tracking_iou
from stage28_response_common import dense_boxes, cpu_array
from stage47_search_collect import Capture
import stage48_stability_common as c


def digest(module, buffers_only=False):
    values = dict(module.named_buffers()) if buffers_only else module.state_dict()
    h = hashlib.sha256()
    for name, value in sorted(values.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class FeatureCapture(Capture):
    def __init__(self, native):
        super().__init__(native)
        self.feature = None
        self.pre_handle = self.rt.net.box_head.register_forward_pre_hook(self._input)

    def _input(self, module, args):
        self.feature = cpu_array(args[0])[0].astype(np.float32)

    def capture(self, image, parent, frame):
        self.feature = None
        result = super().capture(image, parent, frame)
        if self.feature is None:
            raise ValueError("Missing original CENTER head input")
        return dict(result, features=self.feature.copy())


def collect_features(observer, plan, entry, reference, image_reader=checked_image):
    observer.initialize(image_reader(plan["root"], entry["frames"][0]), entry["initial_box"])
    rows = []
    side = int(plan["runtime"]["lens_x"] ** .5)
    for frame in c.sample_frames(entry):
        image = image_reader(plan["root"], entry["frames"][frame])
        parents = c.pair_parents(reference["previous"][frame-1], entry["sequence"], frame)
        gt = np.asarray(entry["ground_truth"][frame], float)  # Labels only after fixed parent selection.
        views = []
        for parent in parents:
            result = observer.capture(image, parent, frame)
            crop, _, factor = crop_geometry(parent, image.shape[:2], plan["search_factor"], plan["search_size"])
            target = np.r_[(gt[:2]-crop[:2])/crop[2], gt[2:]/crop[2]]
            boxes = dense_boxes(result["size"], result["offset"], parent, factor, plan["search_size"], image.shape[:2])
            cell = int(np.argmax(result["raw"]*np.asarray(plan["window"])))
            boxes[cell] = result["boxes"]
            overlap = tracking_iou(boxes, gt).reshape(side,side)
            center = target[:2]+target[2:]/2
            xy = np.clip(np.round(center*side).astype(int),0,side-1)
            views.append(dict(features=result["features"], parents=parent, crop=crop, raw=result["raw"],
                size=result["size"], offset=result["offset"], boxes=result["boxes"], layer=result["layer"],
                target=target, gt_wh=gt[2:], positive=overlap>=c.DESIGN["positive_iou"],
                negative=overlap<c.DESIGN["negative_iou"], size_cell=int(xy[1]*side+xy[0]),
                size_valid=bool(((center>=0)&(center<1)).all())))
        if abs(float(views[0]["raw"].max())-reference["peak"][frame-1]) > 1e-5:
            raise ValueError("Native feature score replay failed")
        rows.append(views)
    data = {key:np.asarray([[view[key] for view in pair] for pair in rows]) for key in rows[0][0]}
    data["frame_id"] = np.asarray(c.sample_frames(entry),int)
    observer.assert_frozen()
    c.audit_features(data,plan,entry,reference)
    return data


def make_head(rt):
    original=rt.net.box_head
    # NativeObserver's bound hooks retain the entire runtime (including torch).
    # Clone weights/modules without copying instrumentation or its owner graph.
    memo={}
    for module in original.modules():
        for name in ("_forward_hooks","_forward_pre_hooks","_backward_hooks","_backward_pre_hooks",
                     "_forward_hooks_with_kwargs","_forward_pre_hooks_with_kwargs","_forward_hooks_always_called"):
            hooks=getattr(module,name,None)
            if hooks is not None: memo[id(hooks)]=type(hooks)()
    head = copy.deepcopy(original,memo).eval().requires_grad_(True)
    if digest(head) != digest(rt.net.box_head):
        raise ValueError("Head initialization changed")
    return head


def head_output(head, features):
    raw, box, size, offset = head(features)
    return dict(score_map=raw,pred_boxes=box[:,None],size_map=size,offset_map=offset)


def loss_terms(torch, output, batch, window):
    """Frozen GT-qualified geometry labels supervise scores; size loss uses GT cells."""
    score, size = output["score_map"],output["size_map"]
    device = score.device
    target = torch.as_tensor(batch["target"],device=device,dtype=score.dtype).reshape(-1,4)
    ordinary,_ = tracking_loss(torch,output,target,True)
    positive = torch.as_tensor(batch["positive"],device=device).reshape(len(score),-1)
    negative = torch.as_tensor(batch["negative"],device=device).reshape(len(score),-1)
    logq = (score*torch.as_tensor(window,device=device,dtype=score.dtype)).flatten(1).clamp_min(1e-8).log()
    valid = positive.any(1)&negative.any(1)
    rank = score.sum()*0
    if valid.any():
        pos = logq[valid].masked_fill(~positive[valid],-float("inf")).amax(1)
        neg = logq[valid].masked_fill(~negative[valid],-float("inf")).amax(1)
        rank = torch.nn.functional.softplus(c.DESIGN["rank_margin"]+neg-pos).mean()
    cells = torch.as_tensor(batch["size_cell"],device=device,dtype=torch.long).reshape(-1)
    supported = torch.as_tensor(batch["size_valid"],device=device).reshape(-1)
    assigned = size.flatten(2).gather(2,cells[:,None,None].expand(-1,2,1)).squeeze(-1)
    crop_side = torch.as_tensor(batch["crop"],device=device,dtype=size.dtype).reshape(-1,4)[:,2:3]
    gt_wh = torch.as_tensor(batch["gt_wh"],device=device,dtype=size.dtype).reshape(-1,2)
    log_physical = (assigned.clamp_min(1e-6)*crop_side).log()
    shape = size.sum()*0
    if supported.any():
        shape = torch.nn.functional.smooth_l1_loss(log_physical[supported],gt_wh[supported].log(),beta=c.DESIGN["log_size_beta"])
    pair_valid = supported.reshape(-1,2).all(1)
    if pair_valid.any():
        paired = log_physical.reshape(-1,2,2)[pair_valid]
        shape = shape+c.DESIGN["size_pair_weight"]*torch.nn.functional.smooth_l1_loss(paired[:,0],paired[:,1],beta=c.DESIGN["log_size_beta"])
    return ordinary,rank,shape,dict(rank_views=int(valid.sum()),size_views=int(supported.sum()))


def objective(torch, head, samples, window, arm):
    if arm not in c.ARMS or head.training:
        raise ValueError("Need a fixed arm and eval-mode head with frozen BN buffers")
    batch = {key:np.stack([s[key] for s in samples]) for key in samples[0] if key not in ("sequence","frame")}
    features = torch.as_tensor(batch["features"],device=next(head.parameters()).device).flatten(0,1)
    output = head_output(head,features)
    ordinary,rank,shape,stats = loss_terms(torch,output,batch,window)
    applied_rank = rank if arm in ("rank","rank_shape") else rank*0
    applied_shape = shape if arm in ("shape","rank_shape") else shape*0
    return ordinary+c.DESIGN["rank_weight"]*applied_rank+c.DESIGN["shape_weight"]*applied_shape, ordinary,applied_rank,applied_shape,stats


def gradient_norm(torch, loss, parameters):
    gradients = torch.autograd.grad(loss,list(parameters),retain_graph=True,allow_unused=True)
    return float(sum(g.detach().square().sum() for g in gradients if g is not None).sqrt()) if any(g is not None for g in gradients) else 0.


def run_trace(native, frames, initial_box, root, image_reader=checked_image):
    """Inference receives images and initialization only; no future labels or gates."""
    native.initialize(image_reader(root,frames[0]),initial_box)
    previous = np.asarray(initial_box,float)
    rows=[]
    for frame,record in enumerate(frames[1:],1):
        image=image_reader(root,record)
        result=native.step(image,previous)
        rows.append(dict(previous=previous.copy(),boxes=result["box"],layer=result["layer"],peak=result["peak"],
            head_calls=result["head_calls"],block_calls=result["block_calls"],image_hw=image.shape[:2]))
        previous=result["box"].copy()
        if frame%250==0:
            print("[native trace] {}/{}".format(frame,len(frames)-1),flush=True)
    native.assert_frozen()
    return {k:np.asarray([r[k] for r in rows]) for k in rows[0]}
