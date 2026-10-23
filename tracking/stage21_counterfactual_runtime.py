"""Paired branch simulation with immutable live tracking history and sampling RNG."""

import numpy as np

from stage4_common import tracking_iou
from stage18_environment import TemporalEnvironment
from stage18_onpolicy_ppo import Collector
from stage18_policy_runtime import PolicyEnvironment, compact_observation, policy_forward
from stage21_counterfactual_math import branch_return, inverse_cdf, probabilities


class CoupledActor:
    def __init__(self, torch, modules):
        self.torch, self.modules = torch, modules
        self.uniform = self.forced = None
        self.last = None

    def __getattr__(self, name):
        return getattr(self.modules, name)

    def observe(self, tokens, memory, history):
        torch = self.torch
        observation = compact_observation(torch, self.modules, tokens, memory, history)
        logits, value = policy_forward(torch, self.modules, observation)
        p = probabilities(logits[0].detach().cpu().numpy())
        action = (self.forced if self.forced is not None else
                  int(p.argmax()) if self.uniform is None else inverse_cdf(p, self.uniform))
        if not np.isfinite(float(logits[0, action])):
            raise ValueError("Forced action violates causal compute credit")
        index = torch.tensor([action], dtype=torch.long, device=logits.device)
        log_probability = torch.nn.functional.log_softmax(logits, dim=-1).gather(1, index[:, None]).squeeze(1)
        self.last = dict(observation={k: v.detach() for k, v in observation.items()},
                         logits=logits, value=value, log_probability=log_probability,
                         old_probabilities=p, action=action)
        return torch.full_like(logits, -torch.inf).scatter(1, index[:, None], 0.), value


class CoupledEnvironment(PolicyEnvironment):
    def __init__(self, runtime, modules):
        super().__init__(runtime, modules)
        self.actor_view = CoupledActor(runtime.torch, modules)
        self.modules = self.actor_view

    def step(self, image, uniform=None, action=None):
        self.actor_view.uniform, self.actor_view.forced = uniform, action
        with self.rt.torch.no_grad():
            result = TemporalEnvironment.step(self, image, action)
        result.update(self.actor_view.last)
        return result


def assert_snapshot_equal(left, right, tolerance=1e-6):
    for field in ("box", "motion"):
        if not np.allclose(left[field], right[field], rtol=0, atol=tolerance):
            raise RuntimeError("Branch contaminated live " + field)
    for field in ("frame_id", "previous_action", "confidence"):
        if left[field] != right[field]:
            raise RuntimeError("Branch contaminated live " + field)
    if vars(left["budget"]) != vars(right["budget"]):
        raise RuntimeError("Branch contaminated live budget")
    a, b = left["memory"], right["memory"]
    if (a is None) != (b is None):
        raise RuntimeError("Branch contaminated live memory")
    if a is not None:
        a = a.detach().cpu().numpy()
        b = b.detach().cpu().numpy()
        if not np.allclose(a, b, rtol=0, atol=tolerance):
            raise RuntimeError("Branch contaminated live memory")


def simulate_branches(env, before, after, actual, images, gt, valid, next_image,
                      gamma, uniforms, minimum_valid_fraction=.8):
    """Force each first action, then continue pi_old under shared random numbers.

    Labels are queried only after execution, never passed to the environment.
    An action's spending changes its subsequent legal actions independently.
    No branch may borrow credit, force future decisions, or reset crops to GT.
    """
    if (len(images) < 1 or np.shape(gt) != (len(images), 4) or np.shape(valid) != (len(images),)
            or uniforms.ndim != 2 or uniforms.shape[1] != len(images) - 1
            or not len(uniforms) or not 0 < minimum_valid_fraction <= 1):
        raise ValueError("Invalid branch horizon/labels/common random numbers")
    results = np.zeros((len(uniforms), 7))
    maximum_error, forwards = 0., 0
    p = actual["old_probabilities"]
    if not np.all(p > 0):
        raise ValueError("Branch anchors require all seven legal actions with positive support")
    old_last = env.actor_view.last
    try:
        for repeat in range(len(uniforms)):
            for action in range(7):
                env.restore(before)
                reward = []
                for index, image in enumerate(images):
                    output = env.step(image, action=action) if index == 0 else env.step(
                        image, uniform=float(uniforms[repeat, index - 1]))
                    if index == 0 and action == actual["action"]:
                        error = float(np.abs(np.asarray(output["box"]) - actual["box"]).max())
                        maximum_error = max(maximum_error, error)
                        assert_snapshot_equal(env.snapshot(), after)
                        if error > 1e-6:
                            raise RuntimeError("Chosen-action branch does not replay the actual transition")
                    overlap = float(tracking_iou(output["box"], gt[index], bool(valid[index])))
                    if valid[index] and (not np.isfinite(overlap) or overlap < 0):
                        raise RuntimeError("Invalid branch prediction")
                    reward.append(overlap if valid[index] else 0.)
                    forwards += 1
                tail = 0. if next_image is None else env.peek_value(next_image)
                results[repeat, action] = branch_return(reward, gamma, tail)
    finally:
        env.restore(after)
        env.actor_view.last = old_last
    assert_snapshot_equal(env.snapshot(), after)
    return dict(q=results.mean(0), q_repeats=results, valid=bool(np.mean(valid) >= minimum_valid_fraction),
                simulation_transitions=forwards, chosen_replay_error=maximum_error)


class CounterfactualCollector(Collector):
    def __init__(self, runtime, modules, sources, seed, horizon=20, repeats=2, anchors=4):
        super().__init__(runtime, modules, sources, seed)
        self.env = CoupledEnvironment(runtime, modules)
        self.seed, self.horizon, self.repeats, self.anchors = seed, horizon, repeats, anchors
        self.total_steps = 0

    def state_dict(self):
        return dict(super().state_dict(), total_steps=self.total_steps)

    def restore(self, state):
        super().restore(state)
        self.total_steps = state["total_steps"]

    def collect(self, count):
        if count < 1:
            raise ValueError("Empty live rollout")
        torch = self.rt.torch
        observations, actions, logp, values, rewards, valid, terminal, truncated = [], [], [], [], [], [], [], []
        anchors, bootstraps = [], {}
        interval = max(1, count // (self.anchors * 2))
        next_anchor = interval
        for step in range(count):
            if self.active is None:
                self._reset()
            source = self.sources[self.active]
            index = self.index
            before = self.env.snapshot()
            result = self.env.step(self.rt.read(source["paths"][index]), uniform=float(self.rng.random()))
            after = self.env.snapshot()
            is_valid = bool(source["valid"][index])
            overlap = float(tracking_iou(result["box"], source["gt"][index], is_valid))
            if is_valid and (not np.isfinite(overlap) or overlap < 0):
                raise RuntimeError("Invalid actual tracking prediction")
            observations.append({k: v.cpu() for k, v in result["observation"].items()})
            actions.append(result["action"])
            logp.append(float(result["log_probability"].item()))
            values.append(float(result["value"].item()))
            rewards.append(overlap if is_valid else 0.)
            valid.append(is_valid)
            if (len(anchors) < self.anchors and step >= next_anchor
                    and index + self.horizon <= len(source["paths"])
                    and np.all(result["old_probabilities"] > 0)):
                # Keyed branch randomness never advances live rollout sampling/order RNG.
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.total_steps + step, 21]))
                uniforms = rng.random((self.repeats, self.horizon - 1))
                paths = source["paths"][index:index + self.horizon]
                next_path = (source["paths"][index + self.horizon]
                             if index + self.horizon < len(source["paths"]) else source["bootstrap_path"])
                branch = simulate_branches(
                    self.env, before, after, result, [self.rt.read(path) for path in paths],
                    source["gt"][index:index + self.horizon], source["valid"][index:index + self.horizon],
                    None if next_path is None else self.rt.read(next_path), self.gamma, uniforms)
                anchors.append(dict(index=step, old_probabilities=result["old_probabilities"], **branch))
                next_anchor = step + interval
            self.index += 1
            ended = self.index == len(source["paths"])
            terminal.append(ended and source["terminal"])
            truncated.append(ended and not source["terminal"])
            if ended:
                bootstraps[step] = (0. if source["terminal"] else
                                    self.env.peek_value(self.rt.read(source["bootstrap_path"])))
                self.active = None
        if len(anchors) != self.anchors:
            raise RuntimeError("Counterfactual anchor quota not met ({}/{}); do not compare unmatched branch budgets".format(
                len(anchors), self.anchors))
        next_values = np.r_[values[1:], 0.]
        if self.active is not None:
            source = self.sources[self.active]
            bootstraps[count - 1] = self.env.peek_value(self.rt.read(source["paths"][self.index]))
            truncated[-1] = True
        for step, bootstrap in bootstraps.items():
            next_values[step] = bootstrap
        self.total_steps += count
        return dict(observation={k: torch.cat([o[k] for o in observations]) for k in observations[0]},
                    actions=np.asarray(actions), old_log_prob=np.asarray(logp), values=np.asarray(values),
                    rewards=np.asarray(rewards), valid=np.asarray(valid), terminated=np.asarray(terminal),
                    truncated=np.asarray(truncated), next_values=next_values, counterfactual=anchors)
