"""CPU branch isolation/math tests, plus optional real PyTorch PPO tests."""

import copy
import importlib.util
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from stage18_environment import TokenCredit
from stage21_counterfactual_math import inverse_cdf
from stage21_counterfactual_runtime import assert_snapshot_equal
from stage23_branch_credit import (ARMS, ANCHORS, HORIZON, REPEATS, BranchCollector,
                                    branch_credit, simulate_branches)
from stage23_branch_ppo import check_runtime, publish_contract, single_writer, update_policy
from stage23_branch_status import read_status
from stage4_common import write_csv, write_json


class Array(np.ndarray):
    def clone(self):
        return self.copy()

    def cpu(self):
        return self

    def detach(self):
        return self


def array(value):
    return np.asarray(value).view(Array)


class FakeTorch:
    cuda = SimpleNamespace(is_available=lambda: False)
    cat = staticmethod(np.concatenate)

    def __init__(self):
        self.rng = array([17])

    def get_rng_state(self):
        return self.rng.copy()

    def set_rng_state(self, state):
        self.rng = state.copy()


class FakeEnvironment:
    def __init__(self, runtime, modules):
        self.rt = runtime
        self.actor_view = SimpleNamespace(uniform=None, forced=None, last=None)
        self.reset(0, [0, 0, 20, 20])
        self.fail_at = None

    def reset(self, image, box):
        self.box, self.frame_id = list(box), 0
        self.budget = TokenCredit()
        self.previous_action, self.confidence = -1, 0.
        self.motion = np.zeros(4)

    def snapshot(self):
        return copy.deepcopy(dict(box=self.box, frame_id=self.frame_id, budget=self.budget,
                                  previous_action=self.previous_action, confidence=self.confidence,
                                  motion=self.motion, memory=None))

    def restore(self, state):
        for k, v in copy.deepcopy(state).items():
            if k != "memory":
                setattr(self, k, v)

    def step(self, image, uniform=None, action=None):
        if self.fail_at is not None and image == self.fail_at:
            raise RuntimeError("branch failure")
        self.budget.advance()
        p = np.array([.7] + [.05] * 6) if self.budget.credit >= 10 else np.array([1.] + [0.] * 6)
        action = inverse_cdf(p, uniform if uniform is not None else .1) if action is None else action
        if p[action] <= 0:
            raise RuntimeError("illegal forced action")
        self.budget.spend(action)
        self.box[0] += .05 * action
        self.frame_id += 1
        self.previous_action, self.confidence = action, .8
        self.actor_view.uniform, self.actor_view.forced = uniform, action
        self.actor_view.last = dict(step=self.frame_id)
        # Deliberately mutate all globals; branch simulation must restore them.
        np.random.random()
        random.random()
        self.rt.torch.rng += 1
        return dict(box=list(self.box), action=action, old_probabilities=p,
                    observation=dict(x=array([[image]])), log_probability=array([np.log(p[action])]),
                    value=array([float(image)]))

    def peek_value(self, image):
        return float(image)


class MathTests(unittest.TestCase):
    def test_delayed_credit_differs_from_immediate_with_identical_rewards(self):
        cube = np.full((2, REPEATS, 7, HORIZON), .3)
        cube[:, :, 1, 0] = .5
        cube[:, :, 2, 5:] = .9
        p, valid = np.full((2, 7), 1 / 7), np.ones(2, bool)
        h1 = branch_credit(cube, p, valid, "ppo_cf_h1")
        long = branch_credit(cube, p, valid, "ppo_cf_h20")
        self.assertEqual(h1["q"][0].argmax(), 1)
        self.assertEqual(long["q"][0].argmax(), 2)
        for result in (h1, long):
            np.testing.assert_allclose((p * result["advantage"]).sum(1), 0, atol=1e-12)
            self.assertAlmostEqual(np.mean((p * result["advantage"] ** 2).sum(1)), 1.)
        baseline = branch_credit(cube, p, valid, "ppo_gae")
        self.assertEqual(baseline["weight"], 0.)
        np.testing.assert_array_equal(baseline["q"], long["q"])

    def test_invalid_and_constant_credit_is_zero(self):
        cube = np.full((2, REPEATS, 7, HORIZON), .5)
        p = np.full((2, 7), 1 / 7)
        result = branch_credit(cube, p, np.array([True, False]), ARMS[-1])
        np.testing.assert_allclose(result["advantage"], 0, atol=1e-7)
        for bad in (cube[..., :-1], np.full_like(cube, np.nan), -cube):
            with self.assertRaises(ValueError):
                branch_credit(bad, p, np.ones(2, bool), ARMS[-1])

    def test_masked_actions_have_no_credit(self):
        cube = np.zeros((1, REPEATS, 7, HORIZON))
        cube[:, :, 6] = 1
        result = branch_credit(cube, np.array([[1., 0, 0, 0, 0, 0, 0]]), np.array([True]), ARMS[-1])
        np.testing.assert_array_equal(result["advantage"], 0.)


class IsolationTests(unittest.TestCase):
    def make_env(self):
        rt = SimpleNamespace(torch=FakeTorch(), read=lambda x: x)
        env = FakeEnvironment(rt, None)
        for i in range(3):
            env.step(i, uniform=.1)
        before = env.snapshot()
        result = env.step(3, uniform=.1)
        return env, before, env.snapshot(), result

    def branch(self, env, before, after, actual):
        return simulate_branches(env, before, after, actual, list(range(HORIZON)),
                                 np.tile([0, 0, 20, 20], (HORIZON, 1)), np.ones(HORIZON, bool),
                                 np.full((REPEATS, HORIZON - 1), .8))

    def test_all_branches_replay_and_restore_live_state_and_rng(self):
        env, before, after, actual = self.make_env()
        transient = env.actor_view.uniform, env.actor_view.forced, env.actor_view.last
        numpy_state, python_state = np.random.get_state(), random.getstate()
        torch_state = env.rt.torch.get_rng_state()
        expected_numpy = np.random.random()
        expected_python = random.random()
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        result = self.branch(env, before, after, actual)
        self.assertEqual(result["simulation_transitions"], REPEATS * 7 * HORIZON)
        self.assertEqual(result["chosen_replay_error"], 0.)
        self.assertEqual(result["rewards"].shape, (REPEATS, 7, HORIZON))
        assert_snapshot_equal(env.snapshot(), after)
        self.assertEqual(transient, (env.actor_view.uniform, env.actor_view.forced, env.actor_view.last))
        np.testing.assert_array_equal(torch_state, env.rt.torch.get_rng_state())
        self.assertEqual(np.random.random(), expected_numpy)
        self.assertEqual(random.random(), expected_python)

    def test_exception_restores_live_state_and_transient(self):
        env, before, after, actual = self.make_env()
        previous = env.actor_view.last
        env.fail_at = 5
        with self.assertRaisesRegex(RuntimeError, "branch failure"):
            self.branch(env, before, after, actual)
        assert_snapshot_equal(env.snapshot(), after)
        self.assertIs(env.actor_view.last, previous)

    def test_branches_never_query_critic_bootstrap_and_reject_illegal_anchors(self):
        env, before, after, actual = self.make_env()
        with patch.object(env, "peek_value", side_effect=AssertionError("branch critic leak")):
            self.branch(env, before, after, actual)
        with self.assertRaisesRegex(ValueError, "seven legal actions"):
            self.branch(env, before, after, dict(actual, old_probabilities=np.array([1.] + [0.] * 6)))

    def collector(self):
        rt = SimpleNamespace(torch=FakeTorch(), read=lambda x: x)
        sources = [dict(sequence="terminal", paths=list(range(65)), terminal=True, bootstrap_path=None),
                   dict(sequence="truncated", paths=list(range(65)), terminal=False, bootstrap_path=65)]
        for s in sources:
            s.update(gt=np.tile([0, 0, 20, 20], (65, 1)), valid=np.ones(65, bool))
        with patch("stage23_branch_credit.CoupledEnvironment", FakeEnvironment):
            return BranchCollector(rt, None, sources, 2074)

    def test_collector_has_fixed_quota_boundaries_budget_and_exact_resume(self):
        c = self.collector()
        first = c.collect(256)
        self.assertEqual(len(first["counterfactual"]), ANCHORS)
        self.assertEqual(sum(a["simulation_transitions"] for a in first["counterfactual"]), 1120)
        self.assertEqual(int((first["terminated"] | first["truncated"]).sum()), 4)
        for begin in range(0, 256, 64):
            self.assertLessEqual(np.sum(first["actions"][begin:begin + 64] > 0), 3 * 64 // 10)
        saved = c.state_dict()
        expected = c.collect(256)
        c.restore(saved)
        actual = c.collect(256)
        for k in ("actions", "old_log_prob", "values", "next_values", "rewards", "terminated", "truncated"):
            np.testing.assert_array_equal(expected[k], actual[k])
        for a, b in zip(expected["counterfactual"], actual["counterfactual"]):
            np.testing.assert_array_equal(a["rewards"], b["rewards"])
        self.assertEqual(c.total_steps, 512)

    def test_active_episode_resume_and_unmatched_quota_fail_closed(self):
        c = self.collector()
        c.collect(128 + 24)
        self.assertIsNotNone(c.active)
        saved = c.state_dict()
        expected = c.collect(256)
        c.restore(saved)
        actual = c.collect(256)
        np.testing.assert_array_equal(actual["actions"], expected["actions"])
        c = self.collector()
        for source in c.sources:
            source.update(paths=source["paths"][:25], gt=source["gt"][:25], valid=source["valid"][:25])
        with self.assertRaisesRegex(RuntimeError, "quota"):
            c.collect(24)


class ContractTests(unittest.TestCase):
    def test_lock_and_contract_prevent_overwrite(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as temp:
            out = Path(temp)
            with single_writer(out):
                publish_contract(out, {"seed": 1})
                publish_contract(out, {"seed": 1})
                with self.assertRaisesRegex(RuntimeError, "lock exists"):
                    with single_writer(out):
                        pass
                with self.assertRaisesRegex(ValueError, "Contract changed"):
                    publish_contract(out, {"seed": 2})
            self.assertFalse((out / ".stage23_writer.lock").exists())
            self.assertEqual(json.loads((out / "stage23_contract.json").read_text()), {"seed": 1})

    def test_runtime_allows_new_device_but_not_new_tracker(self):
        original = dict(parameter="deit", checkpoint_sha256="hash", config={}, lens_z=64, lens_x=256, embed_dim=192)
        check_runtime(dict(original, device="L20"), dict(original, device="old GPU"))
        with self.assertRaises(ValueError):
            check_runtime(dict(original, checkpoint_sha256="other"), original)

    def test_partial_status_uses_reports_without_unpickling(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as temp:
            out = Path(temp)
            write_json(out / "stage23_contract.json", {"fingerprint": "same"})
            write_csv(out / ARMS[0] / "stage23_history.csv", [dict(
                steps=256, branch_transitions=1120, mean_valid_iou=.8,
                likelihood_replay_error=1e-6, probability_replay_error=2e-6, chosen_replay_error=0.)])
            rows, decision = read_status(out)
            self.assertIsNone(decision)
            self.assertEqual([r["reported_steps"] for r in rows], [256, 0, 0])
            self.assertEqual(rows[0]["branch_transitions"], 1120)
            write_json(out / "stage23_decision.json", {"contract_fingerprint": "other"})
            with self.assertRaisesRegex(ValueError, "different contract"):
                read_status(out)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
class TorchTests(unittest.TestCase):
    def test_expected_credit_updates_actor_in_the_delayed_direction(self):
        import torch
        from stage21_counterfactual_math import expected_clipped_loss
        logits = torch.zeros((1, 7), requires_grad=True)
        cube = np.full((1, REPEATS, 7, HORIZON), .3)
        cube[:, :, 1, 0] = .5
        cube[:, :, 2, 5:] = .9
        p, valid = np.full((1, 7), 1 / 7), np.ones(1, bool)
        for arm, best in (("ppo_cf_h1", 1), ("ppo_cf_h20", 2)):
            credit = branch_credit(cube, p, valid, arm)
            loss = expected_clipped_loss(torch, logits, torch.tensor(p), torch.tensor(credit["advantage"]),
                                         torch.tensor(valid))
            gradient, = torch.autograd.grad(loss, logits)
            self.assertEqual(int(gradient.argmin()), best)
            self.assertTrue(torch.isfinite(gradient).all())

    def test_real_environment_branch_restore_replay_and_ppo_update(self):
        from test_stage18_temporal import ModelEnvironmentTests
        from stage18_policy_runtime import configure_policy
        torch, modules, rt = ModelEnvironmentTests().make_system()
        rt.read = lambda x: x
        parameters = configure_policy(modules)
        modules.eval()
        source = dict(sequence="fit", paths=[np.zeros((32, 32, 3)) for _ in range(65)],
                      gt=np.tile([0, 0, 20, 20], (65, 1)), valid=np.ones(65, bool),
                      terminal=True, bootstrap_path=None)
        collector = BranchCollector(rt, modules, [source], 2074)
        before = {n: p.detach().clone() for n, p in modules.named_parameters()}
        data = collector.collect(128)
        self.assertEqual(rt.tracker.head_calls, 128 + 1120)
        metrics = update_policy(torch, modules, torch.optim.Adam(parameters, lr=3e-4), data, ARMS[-1], 2074, 1)
        self.assertLess(metrics["likelihood_replay_error"], 1e-4)
        self.assertEqual(metrics["branch_transitions"], 1120)
        self.assertGreater(metrics["maximum_gradient_norm"], 0)
        for name, p in modules.named_parameters():
            if not p.requires_grad:
                torch.testing.assert_close(before[name], p, rtol=0, atol=0)
                self.assertIsNone(p.grad)


if __name__ == "__main__":
    unittest.main()
