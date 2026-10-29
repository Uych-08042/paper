"""Read-only temporal residual adaptation; no modification of upstream sources.

Consistency is colocated encoder/history cosine, NOT measured optical-flow error.
Scale is the previous predicted box side, NOT current GT or an oracle crop.
Age is frames since explicit stream reset, NOT an object's true memory age.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ResidualGates(nn.Module):
    def __init__(self, hidden=16, levels=3):
        super().__init__()
        self.layers = nn.ModuleList([nn.Sequential(nn.Conv2d(6, hidden, 1), nn.SiLU(),
                                                    nn.Conv2d(hidden, 1, 1)) for _ in range(levels)])
        for layer in self.layers:
            nn.init.zeros_(layer[-1].weight)
            nn.init.zeros_(layer[-1].bias)

    def forward(self, index, features):
        return 2 * self.layers[index](features).sigmoid()


def scale_map(predictions, shape, image_size, device, dtype, floor=.25):
    """Previous output boxes are XYXY, score, category in network pixel coordinates."""
    h, w = shape
    result = torch.zeros((1, 1, h, w), device=device, dtype=dtype)
    for p in sorted(predictions, key=lambda b: (b[4], -((b[2]-b[0])*(b[3]-b[1])))):
        if p[4] < floor or int(p[5]) == 4:
            continue
        x0, y0 = max(0, int(p[0]*w/image_size)), max(0, int(p[1]*h/image_size))
        x1, y1 = min(w, math.ceil(p[2]*w/image_size)), min(h, math.ceil(p[3]*h/image_size))
        if x1 > x0 and y1 > y0:
            side = math.sqrt(max(1., (p[2]-p[0])*(p[3]-p[1])))
            result[:, :, y0:y1, x0:x1] = min(1., math.log1p(side)/math.log1p(image_size))
    return result


def feature_maps(current, residual, encoder, previous, boxes, age, image_size, age_cap, floor):
    with torch.no_grad():
        current, residual = current.detach(), residual.detach()
        rms = current.square().mean(1, keepdim=True).add(1e-8).sqrt()
        base = torch.cat([current.mean(1, keepdim=True).tanh(), rms.tanh(),
                          (residual.square().mean(1, keepdim=True).sqrt()/(rms+1e-6)).tanh(),
                          residual.mean(1, keepdim=True).tanh(),
                          current.abs().mean(1, keepdim=True).tanh(),
                          residual.abs().mean(1, keepdim=True).tanh()], 1)
        consistency = F.cosine_similarity(encoder.detach(), previous.detach(), dim=1,
                                          eps=1e-6).unsqueeze(1)
        side = scale_map(boxes, current.shape[-2:], image_size, current.device, current.dtype, floor)
        elapsed = torch.full_like(side, min(age, age_cap)/age_cap)
        signals = torch.cat([base[:, :3], side, consistency, elapsed], 1)
        if not torch.isfinite(base).all() or not torch.isfinite(signals).all():
            raise FloatingPointError("Non-finite temporal gate features")
        return base, signals


class GateHooks:
    def __init__(self, modules, gates, arm, design):
        if arm not in ("capacity", "signals") or len(modules) != len(gates.layers):
            raise ValueError("Invalid temporal adapter wiring")
        self.gates, self.arm, self.design = gates, arm, design
        self.handles = []
        self.reset()
        for index, module in enumerate(modules):
            self.handles.append(module.register_forward_pre_hook(self._before(index)))
            self.handles.append(module.block.gru.register_forward_pre_hook(self._encoder(index)))
            self.handles.append(module.register_forward_hook(self._after(index)))

    def reset(self):
        self.previous_boxes, self.age = [], 0
        self.inputs, self.encoded, self.stats = {}, {}, []

    def begin(self, boxes, age):
        self.previous_boxes, self.age = boxes, age
        self.inputs, self.encoded, self.stats = {}, {}, []

    def _before(self, index):
        def hook(module, args):
            x1, x2, previous = args[0]
            self.inputs[index] = (torch.cat([x1, x2], 1), previous is None)
        return hook

    def _encoder(self, index):
        def hook(module, args):
            self.encoded[index] = args
        return hook

    def _after(self, index):
        def hook(module, args, output):
            current, first = self.inputs[index]
            previous, encoder = self.encoded[index]
            residual = output[0] - current
            base, signals = feature_maps(current, residual, encoder, previous,
                self.previous_boxes, self.age, self.design["image_size"], self.design["age_cap"],
                self.design["history_box_confidence"])
            gain = self.gates(index, signals if self.arm == "signals" else base)
            if first:
                gain = gain * 0 + 1  # exact native first-frame handling; no false history
            value = output[0] + (gain - 1) * residual
            self.stats.append(dict(level=index, mean_gain=float(gain.detach().mean()),
                                   changed_fraction=float(((gain.detach()-1).abs()>.02).float().mean())))
            return (value, *output[1:])  # retain original memory update, including its gradient
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
