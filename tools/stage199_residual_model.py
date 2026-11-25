"""Fit-anchored, centered ordered-token routers with matched trainable weights."""
from __future__ import annotations

import numpy as np
import stage196_relative_loss as pooling

ARMS = ("absolute_sigmoid", "absolute_additive", "interaction_residual")


def center(value):
    return value - value.mean(-1, keepdim=True)


def fit_statistics(bank, groups):
    if not groups or len(groups) != len(set(groups)) or any(g not in bank for g in groups):
        raise ValueError("Invalid fit source groups")
    # Equal weight per source, then per state. E[std(F)] is not std(E[F]).
    return {key: np.stack([np.asarray(bank[g][source], float).mean(0) for g in groups]).mean(0)
            for key, source in (("prior", "labels"), ("mean_tokens", "tokens"), ("mean_pooled", "pooled"))}


def build(torch, stats, lens_z, arm):
    if arm not in ARMS:
        raise ValueError("Unknown residual routing arm")
    total, dimension = stats["mean_tokens"].shape
    if not 0 < lens_z < total or np.shape(stats["mean_pooled"]) != (4 * dimension,) or np.shape(stats["prior"]) != (6,):
        raise ValueError("Fit statistic shape changed")
    if not all(np.isfinite(v).all() for v in stats.values()) or not ((stats["prior"] > 0) & (stats["prior"] < 1)).all():
        raise ValueError("Need finite statistics and an interior prior for the sigmoid control")

    class Router(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in ("prior", "mean_tokens", "mean_pooled"):
                self.register_buffer(name, torch.as_tensor(stats[name], dtype=torch.float32).clone())
            self.register_buffer("prior_logits", torch.logit(self.prior))
            # Centering cancels all affine biases. Remove those seven redundant
            # parameters in EVERY arm, rather than counting unused parameters.
            self.projection = torch.nn.Linear(dimension, 1, bias=False)
            self.readout = torch.nn.Linear(4 * dimension + total, 6, bias=False)
            torch.nn.init.zeros_(self.projection.weight)
            torch.nn.init.zeros_(self.readout.weight)
            with torch.no_grad():
                self.projection.weight[0, 0] = 1

        def raw(self, tokens, pooled=None):
            if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (total, dimension):
                raise ValueError("Token layout changed")
            value = pooling.pool(torch, tokens, lens_z) if pooled is None else pooled
            if tuple(value.shape) != (len(tokens), 4 * dimension):
                raise ValueError("Pooled input shape changed")
            token_values = self.projection(tokens - self.mean_tokens).squeeze(-1)
            inputs = torch.cat((value - self.mean_pooled, token_values), -1)
            return self.readout(inputs)

        def forward(self, tokens, pooled=None):
            value = self.raw(tokens, pooled)
            if arm == "absolute_sigmoid":
                return (self.prior_logits + value).sigmoid()
            return self.prior + (center(value) if arm == "interaction_residual" else value)

        def objective(self, tokens, pooled, labels):
            score = self(tokens, pooled)
            error = score - labels
            absolute = error.square().mean()
            common = error.mean(1).square().mean()
            interaction = center(error).square().mean()
            loss = interaction if arm == "interaction_residual" else absolute
            return loss, dict(absolute_mse=absolute, common_mse=common, interaction_mse=interaction)

    return Router()
