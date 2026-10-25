"""Plain six-block student; distillation and correction exist only in training."""

import copy
import math
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage18_tracking_loss import tracking_loss
from stage34_identity_runtime import spatial_masks
from stage35_preserve_common import ARMS, DESIGN


def make_student(rt):
    net = copy.deepcopy(rt.net).eval().requires_grad_(False)
    net.backbone.blocks = rt.torch.nn.ModuleList(list(net.backbone.blocks)[:6])
    if hasattr(net.backbone, "MLP"):
        del net.backbone.MLP
    if hasattr(net, "spatial_identity"):
        raise ValueError("Initialize from original SGLA, never a trained identity student")
    net.backbone.blocks[4].requires_grad_(True)
    net.backbone.blocks[5].requires_grad_(True)
    return net


def parameters(net):
    result = {}
    for name, p in net.named_parameters():
        allowed = name.startswith(("backbone.blocks.4.", "backbone.blocks.5."))
        if p.requires_grad != allowed:
            raise ValueError("Student trainability changed: " + name)
        if allowed:
            result[name] = p
    return result


def trainable_state(net):
    return {n: p.detach().cpu().clone() for n, p in parameters(net).items()}


def restore_state(net, state):
    expected = parameters(net)
    if set(expected) != set(state):
        raise ValueError("Checkpoint parameter keys changed")
    for name, p in expected.items():
        value = state[name]
        if p.shape != value.shape or not value.isfinite().all():
            raise ValueError("Invalid trained parameter: " + name)
        p.data.copy_(value.to(p.device))


def forward_student(rt, net, search):
    bb = net.backbone
    if len(bb.blocks) != 6 or hasattr(bb, "MLP") or hasattr(net, "spatial_identity"):
        raise ValueError("Need plain six-block student without a selector/identity branch")
    z = bb.patch_embed(rt.tracker.z_dict1.tensors) + bb.pos_embed_z
    x = bb.patch_embed(search) + bb.pos_embed_x
    tokens = bb.pos_drop(rt.s1.combine_tokens(z, x, mode=bb.cat_mode))
    for block in bb.blocks:
        tokens = block(tokens)
    tokens = bb.norm(rt.s1.recover_tokens(tokens, rt.lens_z, rt.lens_x, mode=bb.cat_mode))
    return net.forward_head(tokens, None)


def preserve_loss(rt, net, student, teacher):
    """Match windowed spatial distribution and box maps at TEACHER's selected cell."""
    torch = rt.torch
    temperature = DESIGN["kd_temperature"]
    window = rt.tracker.output_window.to(student["score_map"])
    s = (student["score_map"] * window).flatten(1).clamp_min(1e-8)
    t = (teacher["score_map"].detach() * window).flatten(1).clamp_min(1e-8)
    log_s = (s.log() / temperature).log_softmax(-1)
    log_t = (t.log() / temperature).log_softmax(-1)
    spatial = (log_t.exp() * (log_t - log_s)).sum(-1).mean() * temperature ** 2
    # One common teacher cell avoids student argmax changes breaking box KD.
    location = torch.zeros_like(t).scatter_(1, t.argmax(1, keepdim=True), 1).reshape_as(student["score_map"])
    sb = net.box_head.cal_bbox(location, student["size_map"], student["offset_map"])
    tb = net.box_head.cal_bbox(location, teacher["size_map"].detach(), teacher["offset_map"].detach())
    return spatial + DESIGN["box_kd_weight"] * torch.nn.functional.l1_loss(sb, tb.detach())


def correction_loss(rt, output, target):
    torch = rt.torch
    score = output["score_map"].clamp(1e-6, 1 - 1e-6)
    logits = score.log() - (-score).log1p()
    positive, negative = spatial_masks(torch, target, *score.shape[-2:])
    result = []
    for row in range(len(score)):
        ids = negative[row].flatten().nonzero().flatten()
        if not len(ids):
            raise ValueError("No spatial negatives under fixed GT exclusion")
        order = score[row, 0].detach().flatten()[ids].topk(min(DESIGN["hard_negatives"], len(ids))).indices
        neg = logits[row, 0].flatten()[ids[order]]
        pos = logits[row, 0][positive[row]]
        result.append(torch.nn.functional.softplus(DESIGN["contrast_margin"] +
            torch.logsumexp(neg, 0) - math.log(len(neg)) - pos.mean()))
    return torch.stack(result).mean()


def losses(rt, net, search, target, arm, reliable):
    if arm not in ARMS or not isinstance(reliable, bool) or len(target) != 1:
        raise ValueError("Need a frozen arm and one offline training reliability label")
    # All matched arms query the same teacher, irrespective of loss activation.
    with rt.torch.no_grad():
        tokens = rt.prefix(search)
        teacher = rt.candidate(tokens, rt.selected_layer(rt.selector(tokens)))
    output = forward_student(rt, net, search)
    supervised, _ = tracking_loss(rt.torch, output, target, True)
    kd = preserve_loss(rt, net, output, teacher)
    correction = correction_loss(rt, output, target)
    use_kd = arm == "distill_all_correct" or (reliable and arm in ("preserve", "preserve_correct"))
    applied_kd = kd if use_kd else kd * 0
    applied_correct = correction if not reliable and arm in ("correct", "preserve_correct", "distill_all_correct") else correction * 0
    total = supervised + DESIGN["kd_weight"] * applied_kd + DESIGN["correction_weight"] * applied_correct
    return total, supervised, applied_kd, applied_correct
