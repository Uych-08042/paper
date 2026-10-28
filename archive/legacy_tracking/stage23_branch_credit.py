"""Fresh on-policy branch credit; no labels enter the deployed environment."""

import copy
import random

import numpy as np

from stage4_common import tracking_iou
from stage18_onpolicy_ppo import Collector
from stage21_counterfactual_math import centered_credit
from stage21_counterfactual_runtime import CoupledEnvironment, assert_snapshot_equal


ARMS = ("ppo_gae", "ppo_cf_h1", "ppo_cf_h20")
HORIZON = 20
REPEATS = 2
ANCHORS = 4
GAMMA = .97


def branch_credit(rewards, probabilities, valid, arm):
    """Return normalized all-action advantages from [anchor, repeat, action, H]."""
    rewards, p = np.asarray(rewards, float), np.asarray(probabilities, float)
    valid = np.asarray(valid)
    if (arm not in ARMS or rewards.ndim != 4 or rewards.shape[1:] != (REPEATS, 7, HORIZON)
            or not len(rewards) or p.shape != (len(rewards), 7)
            or valid.shape != (len(rewards),) or valid.dtype != np.bool_
            or not np.isfinite(rewards).all() or np.any((rewards < 0) | (rewards > 1))):
        raise ValueError("Invalid finite-horizon branch credit input")
    horizon = 1 if arm == "ppo_cf_h1" else HORIZON
    weights = GAMMA ** np.arange(horizon)
    q = (rewards[..., :horizon] * weights).sum(-1).mean(1)
    advantage, scale = centered_credit(q, p, valid)
    return dict(q=q, advantage=advantage, scale=scale,
                weight=0. if arm == "ppo_gae" else .5)


def simulate_branches(env, before, after, actual, images, gt, valid, uniforms):
    """Execute all first actions, then pi_old with paired continuation draws.

    Returns are finite and unbootstrapped. The critic is not used for branch
    targets. The actual after-state and all sampling state survive even errors.
    """
    valid, gt, uniforms = np.asarray(valid), np.asarray(gt), np.asarray(uniforms)
    p = np.asarray(actual["old_probabilities"])
    if (len(images) != HORIZON or gt.shape != (HORIZON, 4)
            or valid.shape != (HORIZON,) or valid.dtype != np.bool_
            or uniforms.shape != (REPEATS, HORIZON - 1)
            or not np.isfinite(uniforms).all() or np.any((uniforms < 0) | (uniforms >= 1))
            or p.shape != (7,) or not np.isfinite(p).all()
            or not np.all(p > 0) or not np.isclose(p.sum(), 1)):
        raise ValueError("Branches require H20, two shared draws and seven legal actions")
    torch = env.rt.torch
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    actor = env.actor_view
    transient = actor.uniform, actor.forced, actor.last
    rewards = np.zeros((REPEATS, 7, HORIZON), dtype=np.float64)
    replay_error, candidate_calls = 0., 0
    try:
        for repeat in range(REPEATS):
            for action in range(7):
                env.restore(before)
                for step, image in enumerate(images):
                    result = (env.step(image, action=action) if step == 0 else
                              env.step(image, uniform=float(uniforms[repeat, step - 1])))
                    candidate_calls += int(result["action"] > 0)
                    if step == 0 and action == actual["action"]:
                        replay_error = max(replay_error, float(np.max(np.abs(
                            np.asarray(result["box"]) - actual["box"]))))
                        assert_snapshot_equal(env.snapshot(), after)
                        if replay_error > 1e-6:
                            raise RuntimeError("Chosen branch does not replay the actual transition")
                    overlap = float(tracking_iou(result["box"], gt[step], bool(valid[step])))
                    if valid[step] and (not np.isfinite(overlap) or not 0 <= overlap <= 1):
                        raise RuntimeError("Invalid branch reward")
                    rewards[repeat, action, step] = overlap if valid[step] else 0.
    finally:
        env.restore(after)
        actor.uniform, actor.forced, actor.last = transient
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)
        assert_snapshot_equal(env.snapshot(), after)
    return dict(rewards=rewards, valid=bool(valid[0] and valid.mean() >= .8),
                simulation_transitions=REPEATS * 7 * HORIZON,
                candidate_calls=candidate_calls,
                chosen_replay_error=replay_error)


class BranchCollector(Collector):
    def __init__(self, runtime, modules, sources, seed):
        super().__init__(runtime, modules, sources, seed)
        self.env = CoupledEnvironment(runtime, modules)
        self.seed, self.total_steps = seed, 0

    def state_dict(self):
        return copy.deepcopy(dict(super().state_dict(), total_steps=self.total_steps))

    def restore(self, state):
        super().restore(state)
        self.total_steps = state["total_steps"]

    def collect(self, count):
        if count < HORIZON + ANCHORS:
            raise ValueError("Rollout too short for the fixed branch quota")
        observations, actions, logp, values, rewards, valid = [], [], [], [], [], []
        terminal, truncated, anchors, bootstraps = [], [], [], {}
        # Greedy causal spacing has no dependence on labels, scores or outcomes.
        interval, next_anchor = max(1, count // (2 * ANCHORS)), 0
        for step in range(count):
            if self.active is None:
                self._reset()
            source, index = self.sources[self.active], self.index
            before = self.env.snapshot()
            result = self.env.step(self.rt.read(source["paths"][index]), uniform=float(self.rng.random()))
            after = self.env.snapshot()
            is_valid = bool(source["valid"][index])
            overlap = float(tracking_iou(result["box"], source["gt"][index], is_valid))
            if is_valid and (not np.isfinite(overlap) or not 0 <= overlap <= 1):
                raise RuntimeError("Invalid live reward")
            observations.append({k: v.detach().cpu() for k, v in result["observation"].items()})
            actions.append(result["action"])
            logp.append(float(result["log_probability"].item()))
            values.append(float(result["value"].item()))
            rewards.append(overlap if is_valid else 0.)
            valid.append(is_valid)
            if (len(anchors) < ANCHORS and step >= next_anchor
                    and index + HORIZON <= len(source["paths"])
                    and np.all(result["old_probabilities"] > 0)):
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.total_steps + step, 23]))
                branch = simulate_branches(
                    self.env, before, after, result,
                    [self.rt.read(p) for p in source["paths"][index:index + HORIZON]],
                    source["gt"][index:index + HORIZON], source["valid"][index:index + HORIZON],
                    rng.random((REPEATS, HORIZON - 1)))
                anchors.append(dict(index=step, sequence=source["sequence"], frame=index,
                                    old_probabilities=result["old_probabilities"].copy(), **branch))
                next_anchor = step + interval
            self.index += 1
            ended = self.index == len(source["paths"])
            terminal.append(ended and source["terminal"])
            truncated.append(ended and not source["terminal"])
            if ended:
                bootstraps[step] = (0. if source["terminal"] else
                                    self.env.peek_value(self.rt.read(source["bootstrap_path"])))
                self.active = None
        if len(anchors) != ANCHORS:
            raise RuntimeError("Branch quota {}/{}; stop instead of comparing unmatched training compute".format(
                len(anchors), ANCHORS))
        next_values = np.r_[values[1:], 0.]
        if self.active is not None:
            source = self.sources[self.active]
            bootstraps[count - 1] = self.env.peek_value(self.rt.read(source["paths"][self.index]))
            truncated[-1] = True
        for step, value in bootstraps.items():
            next_values[step] = value
        self.total_steps += count
        return dict(observation={k: self.rt.torch.cat([o[k] for o in observations]) for k in observations[0]},
                    actions=np.asarray(actions), old_log_prob=np.asarray(logp), values=np.asarray(values),
                    rewards=np.asarray(rewards), valid=np.asarray(valid), terminated=np.asarray(terminal),
                    truncated=np.asarray(truncated), next_values=next_values, counterfactual=anchors,
                    bootstrap_prefix_calls=sum(not terminal[i] for i in bootstraps))
