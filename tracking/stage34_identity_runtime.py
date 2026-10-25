"""One physical six-block path, template-conditioned identity map and losses."""

import copy
import math
import sys
from pathlib import Path

import numpy as np

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage18_tracking_loss import tracking_loss
from stage31_identity_common import roi_weights
from stage34_identity_common import ARMS, DESIGN


def identity_module(torch, dim):
    class SpatialIdentity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.query = torch.nn.Linear(dim, DESIGN["identity_dim"], bias=False)
            self.key = torch.nn.Linear(dim, DESIGN["identity_dim"], bias=False)
            self.key.load_state_dict(self.query.state_dict())
            self.gain = torch.nn.Parameter(torch.zeros(()))

        def forward(self, template, search, weights):
            query = (template * weights[None, :, None]).sum(1)
            q = torch.nn.functional.normalize(self.query(query), dim=-1)
            k = torch.nn.functional.normalize(self.key(search), dim=-1)
            return (k * q[:, None]).sum(-1) / DESIGN["temperature"]
    return SpatialIdentity()


def make_student(rt, arm):
    if arm not in ARMS:
        raise ValueError("Unknown matched arm")
    torch = rt.torch
    net = copy.deepcopy(rt.net).eval().requires_grad_(False)
    net.backbone.blocks = torch.nn.ModuleList(list(net.backbone.blocks)[:6])
    if hasattr(net.backbone, "MLP"):
        del net.backbone.MLP
    net.spatial_identity = identity_module(torch, rt.identity["embed_dim"]).to(next(net.parameters()).device).eval()
    if arm.startswith("representation_"):
        net.backbone.blocks[4].requires_grad_(True)
        net.backbone.blocks[5].requires_grad_(True)
    return net


def allowed_parameters(net, arm):
    result = {}
    for name, parameter in net.named_parameters():
        allowed = name.startswith("spatial_identity.") or (arm.startswith("representation_") and name.startswith(("backbone.blocks.4.", "backbone.blocks.5.")))
        if parameter.requires_grad != allowed:
            raise ValueError("Trainability escaped matched arm: " + name)
        if allowed:
            result[name] = parameter
    return result


def template_weights(rt, entry):
    from stage27_uav_common import crop_geometry
    crop, _, _ = crop_geometry(entry["initial_box"],
        [entry["frames"][0]["height"], entry["frames"][0]["width"]],
        rt.tracker.params.template_factor, rt.tracker.params.template_size)
    return roi_weights(np.asarray(entry["initial_box"])[None], crop, int(math.isqrt(rt.lens_z)))[0]


def forward_student(rt, net, search, weights):
    torch, bb = rt.torch, net.backbone
    if len(bb.blocks) != 6 or hasattr(bb, "MLP"):
        raise ValueError("Student must physically contain exactly six blocks and no selector")
    z = bb.patch_embed(rt.tracker.z_dict1.tensors) + bb.pos_embed_z
    x = bb.patch_embed(search) + bb.pos_embed_x
    tokens = bb.pos_drop(rt.s1.combine_tokens(z, x, mode=bb.cat_mode))
    for block in bb.blocks:
        tokens = block(tokens)
    tokens = bb.norm(rt.s1.recover_tokens(tokens, rt.lens_z, rt.lens_x, mode=bb.cat_mode))
    output = net.forward_head(tokens, None)  # One original localization-head call.
    weights = torch.as_tensor(weights, device=tokens.device, dtype=tokens.dtype)
    identity = net.spatial_identity(tokens[:, :rt.lens_z], tokens[:, rt.lens_z:], weights)
    identity = identity.reshape_as(output["score_map"])
    original = output["score_map"]
    # Logit residual with an exact floating-point identity at zero gain. The
    # subtraction keeps gain gradients, without limiting corrections to a tiny
    # probability-space neighborhood of an already incorrect confident peak.
    bounded = original.clamp(1e-6, 1 - 1e-6)
    base = bounded.log() - (-bounded).log1p()
    fused = (original + ((base + net.spatial_identity.gain.tanh() * identity).sigmoid() - base.sigmoid())).clamp(0, 1)
    result = dict(output, score_map=fused,
        pred_boxes=net.box_head.cal_bbox(fused, output["size_map"], output["offset_map"]).view(-1, 1, 4))
    return result, identity


def spatial_masks(torch, target, height, width):
    """GT supervision only; always include the original CENTER's rounded cell."""
    target = target.detach()
    if target.ndim != 2 or target.shape[1] != 4 or not torch.isfinite(target).all() or (target[:, 2:] <= 0).any():
        raise ValueError("Invalid spatial identity targets")
    xx = (torch.arange(width, device=target.device, dtype=target.dtype) + .5) / width
    yy = (torch.arange(height, device=target.device, dtype=target.dtype) + .5) / height
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")
    center = target[:, :2] + target[:, 2:] / 2
    if not (((center >= 0) & (center < 1)).all()):
        raise ValueError("Training targets must have an in-crop center")
    positive = ((xx[None] >= target[:, 0, None, None]) & (xx[None] < (target[:, 0] + target[:, 2])[:, None, None])
        & (yy[None] >= target[:, 1, None, None]) & (yy[None] < (target[:, 1] + target[:, 3])[:, None, None]))
    pixels = (center * target.new_tensor([width, height])).round().long()
    pixels[:, 0].clamp_(0, width - 1)
    pixels[:, 1].clamp_(0, height - 1)
    positive[torch.arange(len(target), device=target.device), pixels[:, 1], pixels[:, 0]] = True
    expanded = ((xx[None] - center[:, 0, None, None]).abs() <= .75 * target[:, 2, None, None]) & (
        (yy[None] - center[:, 1, None, None]).abs() <= .75 * target[:, 3, None, None])
    neighbors = torch.nn.functional.max_pool2d(positive[:, None].float(), 3, stride=1, padding=1)[:, 0].bool()
    negative = ~expanded & ~neighbors
    return positive, negative


def identity_contrast(torch, identity, response, target):
    positive, negative = spatial_masks(torch, target, *identity.shape[-2:])
    losses = []
    counts = []
    for row in range(identity.shape[0]):
        pos = identity[row, 0][positive[row]]
        neg_ids = negative[row].flatten().nonzero().flatten()
        if not len(neg_ids):
            raise ValueError("No spatial negative after fixed exclusion; do not change masks silently")
        # Mining is detached and confined to GT-negative spatial cells.
        order = response[row, 0].detach().flatten()[neg_ids].topk(min(DESIGN["hard_negatives"], len(neg_ids))).indices
        neg = identity[row, 0].flatten()[neg_ids[order]]
        negative_logmean = torch.logsumexp(neg, 0) - math.log(len(neg))
        losses.append(torch.nn.functional.softplus(DESIGN["contrast_margin"] + negative_logmean - pos.mean()))
        counts.append(len(neg))
    return torch.stack(losses).mean(), counts


def losses(rt, net, search, weights, target, arm):
    output, identity = forward_student(rt, net, search, weights)
    supervised, _ = tracking_loss(rt.torch, output, target, True)
    contrast, negative_counts = identity_contrast(rt.torch, identity, output["score_map"], target)
    total = supervised + (DESIGN["contrast_weight"] * contrast if arm.endswith("_contrast") else contrast * 0)
    return total, supervised, contrast, output, negative_counts


def gradient_norm(torch, loss, parameters):
    parameters = tuple(p for p in parameters if p.requires_grad)
    if not parameters:
        return 0.
    values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    terms = [v.detach().square().sum() for v in values if v is not None]
    return float(torch.stack(terms).sum().sqrt()) if terms else 0.


def trainable_state(net, arm):
    return {name: p.detach().cpu().clone() for name, p in allowed_parameters(net, arm).items()}


def restore_state(net, arm, state):
    expected = allowed_parameters(net, arm)
    if set(state) != set(expected):
        raise ValueError("Trainable checkpoint keys changed")
    for name, parameter in expected.items():
        value = state[name]
        if value.shape != parameter.shape or not value.isfinite().all():
            raise ValueError("Invalid trained parameter: " + name)
        parameter.data.copy_(value.to(parameter.device))
