"""Target-conditioned prefix residual, task router and CENTER quality residual.

Torch is imported only inside build_modules so CPU manifest checks need no torch.
The frozen native parameters and buffers are never installed with new weights.
"""
from __future__ import annotations

from contextlib import ExitStack

from stage194_module_common import flags


def build_modules(torch, dimension, template_tokens, arm):
    a, r, q = flags(arm)

    class Modules(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = torch.nn.Linear(dimension, 2 * dimension)
            self.router = torch.nn.Linear(4 * dimension, 6)
            self.quality = torch.nn.Conv2d(dimension, 1, 1)
            for module in (self.adapter, self.router, self.quality):
                torch.nn.init.zeros_(module.weight)
                torch.nn.init.zeros_(module.bias)
            for module, active in ((self.adapter, a), (self.router, r), (self.quality, q)):
                module.requires_grad_(bool(active))

        def adapt(self, tokens):
            if not a:
                return tokens
            # Parameter-identical control replaces target context with current
            # search context. No history length or arbitrary bottleneck width.
            context = (tokens[:, template_tokens:] if arm.startswith("current_")
                       else tokens[:, :template_tokens]).mean(1)
            scale, shift = self.adapter(context).chunk(2, -1)
            normalized = torch.nn.functional.layer_norm(tokens, (dimension,))
            return tokens + scale.tanh()[:, None] * normalized + shift[:, None]

        def route(self, tokens):
            z, x = tokens[:, :template_tokens], tokens[:, template_tokens:]
            pooled = torch.cat((z.mean(1), z.std(1, unbiased=False),
                                x.mean(1), x.std(1, unbiased=False)), -1)
            return self.router(pooled).sigmoid()

        def score(self, features, original):
            if not q:
                return original
            eps = torch.finfo(original.dtype).eps
            logits = torch.logit(original.clamp(eps, 1 - eps))
            # Subtract the numerical reconstruction error as well. A zero
            # residual then returns the exact original map, preserving ties.
            delta = (logits + self.quality(features)).sigmoid() - logits.sigmoid()
            return (original + delta).clamp(0, 1)

    return Modules()


def dense_quality(torch, output, parents, factors, search_size, image_hw, truth):
    """IoU labels for the exact clipped image-space boxes, detached from geometry."""
    with torch.no_grad():
        size, offset = output["size_map"].detach(), output["offset_map"].detach()
        batch, _, side, _ = size.shape
        yy, xx = torch.meshgrid(torch.arange(side, device=size.device),
                                torch.arange(side, device=size.device), indexing="ij")
        grid = torch.stack((xx, yy), 0).to(size)[None]
        normalized = torch.cat(((grid + offset) / side, size), 1)
        physical = (normalized * search_size / factors.to(size)[:, None, None, None]).double()
        parents, factors, image_hw, truth = (x.double() for x in (parents, factors, image_hw, truth))
        origin = parents[:, :2] + parents[:, 2:] / 2 - search_size / factors[:, None] / 2
        center = physical[:, :2] + origin[:, :, None, None]
        wh = physical[:, 2:]
        limit = image_hw.flip(-1)[:, :, None, None]
        lo = torch.minimum((center - wh / 2).clamp_min(0), limit - 10)
        hi = torch.minimum(torch.maximum(center + wh / 2, size.new_tensor(10)), limit)
        # Match validation_metrics.tracking_iou: clip in floating point,
        # truncate prediction xywh, retain floating GT coordinates.
        wh = (hi - lo).clamp_min(10)
        lo = torch.trunc(lo)
        hi = lo + torch.trunc(wh)
        gtlo = truth[:, :2, None, None]
        gthi = gtlo + truth[:, 2:, None, None]
        inter = (torch.minimum(hi, gthi) - torch.maximum(lo, gtlo)).clamp_min(0).prod(1)
        union = (hi - lo).prod(1) + truth[:, 2:].prod(1)[:, None, None] - inter
        labels = inter / union.clamp_min(torch.finfo(size.dtype).eps)
        return labels[:, None].to(size)


def train_objective(rt, model, arm, tokens, target, parents, factors, hw, truth):
    from stage18_tracking_loss import _center_heatmap, tracking_loss

    torch = rt.torch
    a, r, q = flags(arm)
    encoded = model.adapt(tokens)
    labels, local_losses, quality_losses = [], [], []
    heatmap, _ = _center_heatmap(torch, target, rt.search_side, rt.search_side)
    for layer in range(7, 13):
        features = rt.backbone.blocks[layer - 1](encoded)
        features = rt.s1.recover_tokens(features, rt.lens_z, rt.lens_x, mode=rt.backbone.cat_mode)
        features = rt.backbone.norm(features)
        output = rt.net.forward_head(features, heatmap[:, 0])
        local, _ = tracking_loss(torch, output, target)
        local_losses.append(local)
        search = features[:, -rt.lens_x:].transpose(1, 2).reshape(-1, features.shape[-1], rt.search_side, rt.search_side)
        # Only the ordinary localization objective updates the adapter. Router
        # and quality regressions get detached features; their targets also are
        # detached. This prevents improving a self-generated label by moving it.
        score = model.score(search.detach(), output["score_map"].detach())
        quality = dense_quality(torch, output, parents, factors, rt.tracker.params.search_size, hw, truth)
        quality_losses.append(torch.nn.functional.mse_loss(score, quality))
        cell = (score.detach() * rt.tracker.output_window).flatten(1).argmax(1)
        labels.append(quality.flatten(1).gather(1, cell[:, None])[:, 0])
    local_loss = torch.stack(local_losses).mean()
    quality_loss = torch.stack(quality_losses).mean()
    route_loss = torch.nn.functional.mse_loss(model.route(encoded.detach()), torch.stack(labels, 1))
    losses = {"adapter": local_loss, "router": route_loss, "quality": quality_loss}
    total = sum(losses[k] for k, active in zip(losses, (a, r, q)) if active)
    return total, {k: float(v.detach()) for k, v in losses.items()}


class InstalledModules:
    """Modify internal modules inside the unaltered public tracker.track path."""
    def __init__(self, native, model=None, arm="native", fixed_layer=None):
        self.native, self.rt, self.model, self.arm = native, native.rt, model, arm
        self.fixed_layer = fixed_layer
        self.bits = flags(arm) if model is not None else (0, 0, 0)

    def __enter__(self):
        self.stack = ExitStack()
        if self.model is not None:
            self.stack.callback(self.rt.backbone.blocks[5].register_forward_hook(self.prefix).remove)
        if self.fixed_layer is not None or self.bits[1]:
            self.stack.callback(self.rt.backbone.MLP.register_forward_hook(self.route).remove)
        if self.bits[2]:
            self.stack.callback(self.rt.net.box_head.register_forward_hook(self.head).remove)
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def prefix(self, module, args, output):
        self.tokens = self.model.adapt(output)
        return self.tokens

    def route(self, module, args, output):
        if self.fixed_layer is not None:
            forced = self.rt.torch.zeros_like(output)
            forced[:, self.fixed_layer - 7] = 1
            return forced
        return self.model.route(self.tokens)

    def head(self, module, args, output):
        score, bbox, size, offset = output
        score = self.model.score(args[0], score)
        return score, module.cal_bbox(score, size, offset), size, offset
