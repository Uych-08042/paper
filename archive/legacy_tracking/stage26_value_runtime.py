"""Fresh finite-horizon interventions and differentiable L5-L6 replay."""

import numpy as np

from stage18_environment import executed_crop_roi
from stage18_policy_runtime import compact_observation
from stage18_representation_train import CapturedRuntime
from stage18_tracking_loss import tracking_loss
from stage21_counterfactual_runtime import CoupledEnvironment, assert_snapshot_equal
from stage23_branch_ppo import model_digest
from stage25_relation_collect import simulate
from stage26_value_common import (ARMS, HORIZON, RANK_WEIGHT, VALUE_SCALE, branch_actions,
                                 pairwise_value_loss, post_values, validate_sample)


def build_value_head(torch, embed_dim):
    nn = torch.nn

    class ValueHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.spatial = nn.Conv2d(embed_dim, 16, 1)
            self.template = nn.Linear(embed_dim, 64)
            self.output = nn.Sequential(nn.Linear(1024 + 64 + 256 + 14, 128), nn.GELU(),
                                        nn.LayerNorm(128), nn.Linear(128, 6))

        def forward(self, observation):
            grid = self.spatial(observation["grid"]).flatten(1)
            template = self.template(observation["template"])
            value = self.output(torch.cat((grid, template, observation["memory"].flatten(1),
                                           observation["history"]), 1))
            return value - value.mean(1, keepdim=True)

    return ValueHead()


class ValueActor:
    """One causal Q-head pass, followed by exactly one selected tracking head."""
    def __init__(self, torch, modules, head):
        self.torch, self.modules, self.head = torch, modules, head
        self.uniform = self.forced = self.last = None

    def __getattr__(self, name):
        return getattr(self.modules, name)

    def observe(self, tokens, memory, history):
        torch = self.torch
        if self.uniform is not None:
            raise ValueError("This fixed-call experiment has no stochastic deployment")
        observation = compact_observation(torch, self.modules, tokens, memory, history)
        q = self.head(observation)
        if not torch.isfinite(q).all():
            raise RuntimeError("Nonfinite pre-action candidate utilities")
        available = bool(history[0, 5] >= .5)
        action = (int(self.forced) if self.forced is not None else
                  1 + int(q[0].argmax()) if available else 0)
        if action not in range(7) or (action and not available):
            raise ValueError("Candidate action exceeds causal P30 credit")
        logits = tokens.new_full((1, 7), -torch.inf)
        logits[0, action] = 0
        self.last = dict(action=action, observation={k: v.detach().clone() for k, v in observation.items()},
                         candidate_values=q.detach() * VALUE_SCALE)
        return logits, tokens.new_zeros(1)


def value_environment(runtime, modules, head):
    env = CoupledEnvironment(CapturedRuntime(runtime), modules)
    env.actor_view = ValueActor(runtime.torch, modules, head)
    env.modules = env.actor_view
    return env


def trainable_prefix(runtime):
    runtime.net.eval().requires_grad_(False)
    selected = runtime.torch.nn.ModuleList([runtime.backbone.blocks[4], runtime.backbone.blocks[5]])
    selected.requires_grad_(True)
    allowed = {id(p) for p in selected.parameters()}
    if {id(p) for p in runtime.net.parameters() if p.requires_grad} != allowed:
        raise RuntimeError("Trainable runtime parameters escaped shared L5-L6")
    return selected


def gradient_norm(torch, loss, parameters):
    gradients = torch.autograd.grad(loss, tuple(parameters), retain_graph=True, allow_unused=True)
    squared = [g.detach().square().sum() for g in gradients if g is not None]
    return float(torch.sqrt(torch.stack(squared).sum())) if squared else 0.


def collect_example(rt, modules, head, prefix, source, actions, anchor, seed, round_index):
    """Each optimizer attempt has its own fresh, predeclared training episode."""
    prefix_id, head_id = model_digest(prefix), model_digest(head)
    env = value_environment(rt, modules, head)
    env.reset(rt.read(source["paths"][0]), source["gt"][0])
    for step in range(anchor + 1):
        image = rt.read(source["paths"][step + 1])
        before = env.snapshot()
        actual = env.step(image, action=int(actions[step]))
    after = env.snapshot()
    future = slice(anchor + 1, anchor + 1 + HORIZON)
    images = [rt.read(path) for path in source["paths"][future]]
    choices = branch_actions(actions, anchor, seed, source["sequence"], round_index)
    rewards, costs, error, replay = simulate(env, before, after, actual, images,
                                            source["gt"][future], source["valid"][future], choices)
    live = [np.asarray(actual["box"]).copy()]
    try:
        for offset in range(1, HORIZON):
            live.append(env.step(images[offset], action=int(choices[0, offset]))["box"])
    finally:
        env.restore(after)
        assert_snapshot_equal(env.snapshot(), after)
    values, eligible = post_values(rewards, source["valid"][future])
    metadata = dict(sequence=source["sequence"], anchor=int(anchor), round=int(round_index),
                    eligible=eligible, teacher_prefix_sha256=prefix_id, teacher_head_sha256=head_id,
                    prefix_frames=int(anchor + 1), replay_frames=HORIZON - 1, branch_frames=240,
                    first_replay_error=error)
    arrays = dict(rewards=rewards, valid=source["valid"][future].copy(), costs=costs,
                  continuation=choices, replay_boxes=replay, live_boxes=np.asarray(live), target=values.mean(0))
    validate_sample(arrays, metadata)
    if model_digest(prefix) != prefix_id or model_digest(head) != head_id:
        raise RuntimeError("Model changed while collecting its intervention targets")
    context = dict(env=env, before=before, after=after, actual=actual, image=images[0],
                   gt=source["gt"][anchor + 1].copy(), valid=bool(source["valid"][anchor + 1]))
    return arrays, metadata, context


def current_losses(rt, modules, head, prefix, context, arrays, metadata, arm):
    """Unlike TemporalEnvironment.step, this prefix path MUST retain autograd."""
    if arm not in ARMS or model_digest(prefix) != metadata["teacher_prefix_sha256"] or model_digest(head) != metadata["teacher_head_sha256"]:
        raise ValueError("Do not regress stale returns after changing the tracking model")
    torch, env = rt.torch, context["env"]
    env.restore(context["before"])
    try:
        with torch.enable_grad():
            search, factor = rt.prepare(context["image"])
            tokens = rt.prefix(search)
            if not tokens.requires_grad:
                raise RuntimeError("Shared-prefix autograd was disabled")
            memory = context["before"]["memory"]
            memory = modules.initial_memory(tokens) if memory is None else memory.detach()
            history = context["actual"]["observation"]["history"]
            observation = compact_observation(torch, modules, tokens, memory, history)
            grid_error = float((observation["grid"].detach() - context["actual"]["observation"]["grid"]).abs().max())
            rank_observation = ({k: v.detach() for k, v in observation.items()} if arm == "stop_gradient" else observation)
            prediction = head(rank_observation)
            rank = pairwise_value_loss(torch, prediction, arrays["target"], metadata["eligible"])
            features = modules.read_memory(tokens, memory)
            action = context["actual"]["action"]
            if action:
                features = rt.backbone.blocks[action + 5](features)
            features = modules.adapt(features, action)
            output = rt.head(features)
            box = np.asarray(rt.decode(output, factor, context["image"]))
            box_error = float(np.abs(box - context["actual"]["box"]).max())
            if max(box_error, grid_error) > 1e-6:
                raise RuntimeError("Differentiable replay differs from actual executed crop/output")
            target = np.zeros(4, np.float32)
            if context["valid"]:
                target = executed_crop_roi(context["gt"], context["before"]["box"], factor, rt.tracker.params.search_size)
            localization, _ = tracking_loss(torch, output, tokens.new_tensor(target)[None], context["valid"])
            total = localization + RANK_WEIGHT * rank
            if not torch.isfinite(total):
                raise RuntimeError("Nonfinite matched representation loss")
            if metadata["eligible"]:
                rank_norm = gradient_norm(torch, rank, prefix.parameters())
                track_norm = gradient_norm(torch, localization, prefix.parameters())
            else:
                rank_norm = track_norm = 0.
    finally:
        env.restore(context["after"])
    if arm == "stop_gradient" and rank_norm != 0:
        raise RuntimeError("Stop-gradient control leaked value supervision into tracking features")
    return total, dict(rank_loss=float(rank.detach()), tracking_loss=float(localization.detach()),
                      rank_prefix_gradient=rank_norm, tracking_prefix_gradient=track_norm,
                      differentiable_replay_error=max(box_error, grid_error))
