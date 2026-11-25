"""Matched ordered/value-sorted token readouts; no backbone execution."""
from __future__ import annotations

import stage196_relative_loss as prior

ARMS = ("ordered", "unordered", "global_stats")


def build(torch, dimension, template_tokens, total_tokens, arm):
    if arm not in ARMS or not 0 < template_tokens < total_tokens or dimension < 1:
        raise ValueError("Invalid routing interface")

    class Router(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.arm = arm
            self.dimension, self.template_tokens, self.total_tokens = dimension, template_tokens, total_tokens
            if arm != "global_stats":
                self.projection = torch.nn.Linear(dimension, 1)
                # Original SGLA exposes channel zero of every ordered token to
                # its MLP. Warm-start that interface without adding a hidden width.
                torch.nn.init.zeros_(self.projection.weight)
                torch.nn.init.zeros_(self.projection.bias)
                with torch.no_grad():
                    self.projection.weight[0, 0] = 1
            self.readout = torch.nn.Linear(4 * dimension + (total_tokens if arm != "global_stats" else 0), 6)
            torch.nn.init.zeros_(self.readout.weight)
            torch.nn.init.zeros_(self.readout.bias)

        def features(self, tokens, pooled=None):
            if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (total_tokens, dimension):
                raise ValueError("Token shape changed")
            global_value = prior.pool(torch, tokens, template_tokens) if pooled is None else pooled
            if tuple(global_value.shape) != (len(tokens), 4 * dimension):
                raise ValueError("Global statistics shape changed")
            if arm == "global_stats":
                return global_value
            value = self.projection(tokens).squeeze(-1)
            if arm == "unordered":
                # Sort separately per sample and stream, preserving every scalar
                # and stream membership while removing its original grid address.
                value = torch.cat((value[:, :template_tokens].sort(1).values,
                                   value[:, template_tokens:].sort(1).values), 1)
            return torch.cat((global_value, value), 1)

        def forward(self, tokens, pooled=None):
            return self.readout(self.features(tokens, pooled)).sigmoid()

    return Router()


def layout_audit(torch, tokens, pooled, template_tokens):
    """Label-free invariance checks with an exact, within-stream reversal."""
    n, dimension = tokens.shape[1:]
    permutation = torch.cat((torch.arange(template_tokens - 1, -1, -1),
                             torch.arange(n - 1, template_tokens - 1, -1)))
    ordered = build(torch, dimension, template_tokens, n, "ordered")
    unordered = build(torch, dimension, template_tokens, n, "unordered")
    with torch.no_grad():
        original = ordered.features(tokens, pooled)[:, 4 * dimension:]
        permuted = ordered.features(tokens[:, permutation], pooled)[:, 4 * dimension:]
        value1 = unordered.features(tokens, pooled)[:, 4 * dimension:]
        value2 = unordered.features(tokens[:, permutation], pooled)[:, 4 * dimension:]
    if not torch.equal(value1, value2):
        raise ValueError("Unordered readout retained token order")
    for start, end in ((0, template_tokens), (template_tokens, n)):
        if not torch.equal(original[:, start:end].sort(1).values, value1[:, start:end]):
            raise ValueError("Sorting changed the projected value multiset")
    return dict(states=len(tokens), tokens=n, dimension=dimension, template_tokens=template_tokens,
                unordered_permutation_error=float((value1 - value2).abs().max()),
                ordered_permutation_max_change=float((original - permuted).abs().max()),
                ordered_changed_states=int((original != permuted).any(1).sum()),
                projected_multiset_preserved=True)
