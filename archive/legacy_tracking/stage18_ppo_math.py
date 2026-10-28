"""NumPy rollout math and an injected-torch PPO loss for frozen Stage-18 memory.

Importing this module requires only NumPy and the standard library. Rollouts
must provide the next observation's value before any episode reset; this is
especially important for truncations, which still bootstrap that value.
"""

import math

import numpy as np


def _coefficient(name, value, upper=None, exclusive=False):
    value = float(value)
    if (not math.isfinite(value) or value < 0
            or (upper is not None and (value >= upper if exclusive else value > upper))):
        raise ValueError("Invalid " + name)
    return value


def _vector(name, value):
    array = np.asarray(value)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(name + " must be a nonempty vector")
    if array.dtype.kind not in "iuf":
        raise TypeError(name + " must contain real numbers")
    array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError(name + " must be finite")
    return array


def _mask(name, value, shape):
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(name + " must match the vector shape")
    if array.dtype != np.bool_:
        raise TypeError(name + " must be boolean")
    return array


def gae(rewards, values, next_values, terminated, truncated, gamma=0.97, gae_lambda=0.95):
    """Return float64 ``(advantages, returns)`` for chronological vectors [T].

    Delta is ``reward + gamma * next_value * (not terminated) - value``.
    The recursive trace stops on either terminated or truncated, and at the
    array's end (an implicit rollout truncation). Never use a reset episode's
    value as next_values for the preceding transition. Boolean boundary masks
    and finite, equally sized, nonempty real vectors are required.
    """
    gamma = _coefficient("gamma", gamma, upper=1)
    gae_lambda = _coefficient("gae_lambda", gae_lambda, upper=1)
    rewards = _vector("rewards", rewards)
    values = _vector("values", values)
    next_values = _vector("next_values", next_values)
    if rewards.shape != values.shape or rewards.shape != next_values.shape:
        raise ValueError("rewards, values and next_values must have the same shape")
    terminated = _mask("terminated", terminated, rewards.shape)
    truncated = _mask("truncated", truncated, rewards.shape)
    advantages = np.empty_like(rewards)
    trace = 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        for index in range(len(rewards) - 1, -1, -1):
            bootstrap = 0.0 if terminated[index] else gamma * next_values[index]
            delta = rewards[index] + bootstrap - values[index]
            if terminated[index] or truncated[index]:
                trace = delta
            else:
                trace = delta + gamma * gae_lambda * trace
            advantages[index] = trace
        returns = advantages + values
    if not np.isfinite(advantages).all() or not np.isfinite(returns).all():
        raise ValueError("GAE produced nonfinite advantages or returns")
    return advantages, returns


def normalize_advantages(advantages, valid):
    """Normalize valid rows with population std; return float64, invalid rows 0.

    Constant/singleton valid advantages become zero. All input advantages must
    be finite, including invalid rows; at least one boolean valid row is needed.
    Inputs are never modified.
    """
    advantages = _vector("advantages", advantages)
    valid = _mask("valid", valid, advantages.shape)
    if not valid.any():
        raise ValueError("valid must select at least one row")
    selected = advantages[valid]
    result = np.zeros_like(advantages)
    # Rescale first so finite large inputs cannot overflow the mean/variance.
    scale = np.abs(selected).max()
    if scale > 0:
        centered = selected / scale
        centered -= centered.mean()
        std = centered.std()
        if std > 0:
            result[valid] = centered / std
    return result


def ppo_objective(torch, logits, values, actions, old_log_prob, returns, advantages, valid,
                  clip_epsilon=0.2, value_coefficient=0.5, entropy_coefficient=0.01):
    """Return scalar PPO ``(loss, metrics)`` without importing torch here.

    logits is floating [B, A], all other inputs are tensors [B] on its device.
    actions must be integer, valid boolean with at least one true row, and the
    remaining vectors floating and finite on every row. Logits may contain
    -inf masks, but Fast (column 0) and every selected action must be finite
    on every row. Shape/legality checks also apply to rows excluded by valid.

    Only valid rows enter arithmetic/reductions. Rollout old_log_prob, returns
    and advantages are detached. Loss is clipped policy loss plus
    value_coefficient * raw value MSE minus entropy_coefficient * entropy.
    Metrics are detached scalar tensors: policy_loss, value_loss (raw MSE),
    entropy, approx_kl (mean(exp(log_ratio) - 1 - log_ratio), nonnegative),
    and clip_fraction. Half/bfloat16 arithmetic is promoted to float32.
    Malformed/nonfinite inputs or a nonfinite objective raise explicitly.
    """
    clip_epsilon = _coefficient("clip_epsilon", clip_epsilon, upper=1, exclusive=True)
    value_coefficient = _coefficient("value_coefficient", value_coefficient)
    entropy_coefficient = _coefficient("entropy_coefficient", entropy_coefficient)
    if not torch.is_tensor(logits) or logits.ndim != 2 or min(logits.shape) == 0:
        raise ValueError("logits must be a nonempty tensor [B, A]")
    if not logits.is_floating_point():
        raise TypeError("logits must be floating-point")
    batch, action_count = logits.shape
    vectors = dict(values=values, actions=actions, old_log_prob=old_log_prob,
                   returns=returns, advantages=advantages, valid=valid)
    for name, tensor in vectors.items():
        if not torch.is_tensor(tensor) or tensor.shape != (batch,):
            raise ValueError(name + " must be a tensor [B]")
        if tensor.device != logits.device:
            raise ValueError(name + " must share the logits device")
        if name not in ("actions", "valid"):
            if not tensor.is_floating_point():
                raise TypeError(name + " must be floating-point")
            if not torch.isfinite(tensor).all():
                raise ValueError(name + " must be finite")
    if valid.dtype != torch.bool:
        raise TypeError("valid must be boolean")
    if not valid.any():
        raise ValueError("valid must select at least one row")
    if actions.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("actions must be integer indices")
    if ((actions < 0) | (actions >= action_count)).any():
        raise ValueError("actions are out of range")
    finite = torch.isfinite(logits)
    if not (finite | torch.isneginf(logits)).all():
        raise ValueError("logits must be finite or masked with -inf")
    if not finite[:, 0].all():
        raise ValueError("Fast (action 0) must be finite and legal on every row")
    actions = actions.to(dtype=torch.int64)
    if not finite.gather(1, actions[:, None]).all():
        raise ValueError("Selected actions must be finite and legal")

    floating = (logits, values, old_log_prob, returns, advantages)
    dtype = torch.float64 if any(t.dtype == torch.float64 for t in floating) else torch.float32
    selected_logits = logits[valid].to(dtype=dtype)
    predicted_values = values[valid].to(dtype=dtype)
    old = old_log_prob[valid].detach().to(dtype=dtype)
    targets = returns[valid].detach().to(dtype=dtype)
    advantage = advantages[valid].detach().to(dtype=dtype)
    log_probabilities = torch.nn.functional.log_softmax(selected_logits, dim=-1)
    log_prob = log_probabilities.gather(1, actions[valid, None]).squeeze(1)
    log_ratio = log_prob - old
    ratio = log_ratio.exp()
    if not torch.isfinite(log_ratio).all() or not torch.isfinite(ratio).all():
        raise ValueError("PPO log ratios and ratios must be finite")
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
    policy_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
    value_loss = (predicted_values - targets).square().mean()
    # Replace masked log probabilities before multiplying: 0 * -inf is NaN.
    entropy_logs = log_probabilities.masked_fill(~finite[valid], 0)
    entropy = -(log_probabilities.exp() * entropy_logs).sum(dim=-1).mean()
    loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
    metrics = dict(
        policy_loss=policy_loss, value_loss=value_loss, entropy=entropy,
        approx_kl=(torch.expm1(log_ratio) - log_ratio).clamp_min(0).mean(),
        clip_fraction=((ratio - 1).abs() > clip_epsilon).to(dtype=dtype).mean())
    if not torch.isfinite(loss) or any(not torch.isfinite(t) for t in metrics.values()):
        raise ValueError("PPO objective and metrics must be finite")
    return loss, {name: tensor.detach() for name, tensor in metrics.items()}
