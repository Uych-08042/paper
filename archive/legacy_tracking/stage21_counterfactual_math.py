"""Expected-action PPO credit over fresh, same-state on-policy branches."""

import numpy as np


def probabilities(logits):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (7,) or not np.isfinite(logits[0]) or not (np.isfinite(logits) | np.isneginf(logits)).all():
        raise ValueError("Expected seven finite/masked logits, with legal Fast")
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


def inverse_cdf(probability, uniform):
    probability = np.asarray(probability, dtype=float)
    if (probability.shape != (7,) or not np.isfinite(probability).all()
            or np.any(probability < 0) or not np.isclose(probability.sum(), 1, atol=1e-7)
            or not np.isfinite(uniform) or not 0 <= uniform < 1):
        raise ValueError("Invalid categorical probabilities or uniform draw")
    return min(6, int(np.searchsorted(probability.cumsum(), uniform, side="right")))


def branch_return(rewards, gamma, bootstrap):
    rewards = np.asarray(rewards, dtype=float)
    if (rewards.ndim != 1 or not len(rewards) or not np.isfinite(rewards).all()
            or not np.isfinite(gamma) or not 0 <= gamma < 1 or not np.isfinite(bootstrap)):
        raise ValueError("Invalid finite-horizon return input")
    return float(np.dot(gamma ** np.arange(len(rewards)), rewards) + gamma ** len(rewards) * bootstrap)


def centered_credit(q_values, old_probabilities, valid):
    q, p, valid = np.asarray(q_values, float), np.asarray(old_probabilities, float), np.asarray(valid)
    if (q.ndim != 2 or q.shape[1] != 7 or p.shape != q.shape or valid.shape != (len(q),)
            or valid.dtype != np.bool_ or not np.isfinite(q).all() or not np.isfinite(p).all()
            or np.any(p < 0) or not np.allclose(p.sum(1), 1, rtol=0, atol=1e-6)):
        raise ValueError("Invalid action-value credit matrix")
    advantage = q - (p * q).sum(1, keepdims=True)
    advantage[p == 0] = 0
    advantage[~valid] = 0
    scale = float(np.sqrt(np.mean((p[valid] * advantage[valid] ** 2).sum(1)))) if valid.any() else 0.
    normalized = advantage / max(scale, 1e-8)
    return normalized, scale


def expected_clipped_loss(torch, logits, old_probabilities, advantage, valid, clip=.2):
    """All-action clipped surrogate, weighted by frozen old-policy probability.

    This is a finite-branch auxiliary objective, not a new PPO theorem. Branch
    values are bootstrapped old-policy estimates and may contain critic error.
    """
    if (logits.ndim != 2 or logits.shape[1] != 7 or old_probabilities.shape != logits.shape
            or advantage.shape != logits.shape or valid.shape != (len(logits),)
            or valid.dtype != torch.bool or not 0 < clip < 1):
        raise ValueError("Invalid expected-action surrogate shapes")
    p, advantage = old_probabilities.detach(), advantage.detach()
    if (not torch.isfinite(p).all() or (p < 0).any()
            or not torch.allclose(p.sum(1), torch.ones_like(p[:, 0]), atol=1e-6, rtol=0)
            or not torch.isfinite(advantage).all()):
        raise ValueError("Invalid old probabilities or action advantages")
    if not (torch.isfinite(logits) | torch.isneginf(logits)).all() or not torch.isfinite(logits[:, 0]).all():
        raise ValueError("Invalid masked policy logits")
    if ((p > 0) & ~torch.isfinite(logits)).any():
        raise ValueError("Previously legal actions became illegal on a cached observation")
    if not valid.any():
        return logits[:, 0].sum() * 0
    current = torch.softmax(logits[valid].float(), dim=1)
    old = p[valid].float()
    ratio = current / old.clamp_min(1e-12)
    a = advantage[valid].float()
    return -(old * torch.minimum(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a)).sum(1).mean()
