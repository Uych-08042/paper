"""Temporary physical FFN channel surgery, with exact module restoration."""

from contextlib import contextmanager

import numpy as np


def make_masks(widths, fraction=0.25, per_layer=2, seed=2057):
    if not 0 < fraction < 1 or per_layer < 1:
        raise ValueError("Require 0 < fraction < 1 and per_layer >= 1")
    specs = []
    for layer, width in enumerate(widths, start=1):
        count = int(round(width * fraction))
        if count < 1 or count >= width or count % 32 or count * per_layer > width:
            raise ValueError("Removed channels must be a positive multiple of 32; groups must be disjoint")
        order = np.random.default_rng(np.random.SeedSequence([int(seed), layer])).permutation(width)
        for group in range(per_layer):
            specs.append({"name": "l{}_ffn_g{}".format(layer, group), "layer": layer,
                          "width": int(width), "drop": sorted(order[group * count:(group + 1) * count].tolist())})
    if len({(item["width"], len(item["drop"])) for item in specs}) != 1:
        raise ValueError("This pilot requires equal-size FFN deletion groups")
    return specs


def linear_slices(fc1_weight, fc1_bias, fc2_weight, drop):
    """One shared slice plan for numpy tests and torch parameter surgery."""
    width = fc1_weight.shape[0]
    drop = np.asarray(drop)
    if drop.size == 0:
        drop = drop.astype(np.int64)
    if (drop.ndim != 1 or not np.issubdtype(drop.dtype, np.integer)
            or len(np.unique(drop)) != len(drop) or np.any(drop < 0) or np.any(drop >= width)):
        raise ValueError("Invalid FFN channel indices")
    keep = np.setdiff1d(np.arange(width), drop)
    if not len(keep) or fc2_weight.shape[1] != width:
        raise ValueError("Cannot remove every FFN channel or slice mismatched weights")
    return keep


@contextmanager
def compress_ffn(rt, spec, verify=True):
    """Replace both dense matrices. No masks/hooks remain in the compiled path."""
    torch = rt.torch
    mlp = rt.backbone.blocks[spec["layer"] - 1].mlp
    original1, original2 = mlp.fc1, mlp.fc2
    if original1.out_features != spec["width"]:
        raise ValueError("FFN width disagrees with the frozen mask")
    keep = linear_slices(original1.weight, original1.bias, original2.weight, spec["drop"])
    index = torch.as_tensor(keep, dtype=torch.long, device=original1.weight.device)
    probe = expected = None
    if verify:
        probe = torch.sin(torch.arange(3 * original1.in_features, device=original1.weight.device,
                                       dtype=original1.weight.dtype)).reshape(1, 3, -1)
        def zero_channels(module, args):
            value = args[0].clone()
            value[..., spec["drop"]] = 0
            return (value,)
        handle = original2.register_forward_pre_hook(zero_channels)
        try:
            with torch.no_grad():
                expected = mlp(probe).clone()
        finally:
            handle.remove()
    try:
        fc1 = torch.nn.Linear(original1.in_features, len(keep), bias=original1.bias is not None)
        fc2 = torch.nn.Linear(len(keep), original2.out_features, bias=original2.bias is not None)
        fc1 = fc1.to(device=original1.weight.device, dtype=original1.weight.dtype)
        fc2 = fc2.to(device=original2.weight.device, dtype=original2.weight.dtype)
        with torch.no_grad():
            fc1.weight.copy_(original1.weight.index_select(0, index))
            fc2.weight.copy_(original2.weight.index_select(1, index))
            if original1.bias is not None:
                fc1.bias.copy_(original1.bias.index_select(0, index))
            if original2.bias is not None:
                fc2.bias.copy_(original2.bias)
        fc1.requires_grad_(False).eval()
        fc2.requires_grad_(False).eval()
        mlp.fc1, mlp.fc2 = fc1, fc2
        error = 0.0
        if verify:
            with torch.no_grad():
                actual = mlp(probe)
            error = float((actual - expected).abs().max().item())
            if not torch.allclose(actual, expected, rtol=1e-4, atol=1e-5):
                raise RuntimeError("Physical pruning disagrees with the zeroed-channel reference")
        removed = (original1.weight.numel() + original2.weight.numel()
                   - fc1.weight.numel() - fc2.weight.numel())
        if original1.bias is not None:
            removed += original1.bias.numel() - fc1.bias.numel()
        yield {"removed_parameters": int(removed), "masked_equivalence_max_abs_diff": error}
    finally:
        mlp.fc1, mlp.fc2 = original1, original2
