"""Rollout boundaries and evaluation isolation without tracker dependencies."""

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from stage18_onpolicy_ppo import (Collector, evaluate_preserving_collector,
                                  success_auc, update_policy, verify_selection)


class Array(np.ndarray):
    def cpu(self):
        return self

    def detach(self):
        return self


def array(value):
    return np.asarray(value).view(Array)


class FakeEnvironment:
    def __init__(self, runtime, modules):
        self.steps = 0

    def reset(self, image, box):
        self.steps = 0

    def step(self, image, stochastic=False):
        self.steps += 1
        return dict(box=[0, 0, 20, 20], observation=dict(x=array([[image]])),
                    action=0, log_probability=array([0.]), value=array([float(image)]))

    def peek_value(self, image):
        return float(image)

    def snapshot(self):
        return dict(steps=self.steps, memory=None)

    def restore(self, state):
        self.steps = state["steps"]


class RolloutTests(unittest.TestCase):
    def collector(self):
        runtime = SimpleNamespace(read=lambda image: image,
                                  torch=SimpleNamespace(cat=np.concatenate))
        sources = [dict(sequence="terminal", paths=[0, 1, 2],
                        terminal=True, bootstrap_path=None),
                   dict(sequence="truncated", paths=[10, 11, 12],
                        terminal=False, bootstrap_path=13)]
        for source in sources:
            source.update(gt=np.tile([0, 0, 20, 20], (3, 1)), valid=np.ones(3, bool))
        with patch("stage18_onpolicy_ppo.PolicyEnvironment", FakeEnvironment):
            collector = Collector(runtime, None, sources, 1)
        collector.queue = [0, 1, 0]
        return collector

    def test_bootstrap_distinguishes_video_clip_and_rollout_end(self):
        collector = self.collector()
        data = collector.collect(5)
        np.testing.assert_array_equal(data["values"], [1, 2, 11, 12, 1])
        np.testing.assert_array_equal(data["next_values"], [2, 0, 12, 13, 2])
        np.testing.assert_array_equal(data["terminated"], [False, True, False, False, False])
        np.testing.assert_array_equal(data["truncated"], [False, False, False, True, True])
        self.assertEqual((collector.active, collector.index, collector.env.steps), (0, 2, 1))
        following = collector.collect(1)
        np.testing.assert_array_equal(following["values"], [2])
        np.testing.assert_array_equal(following["next_values"], [0])
        self.assertTrue(following["terminated"][0])
        self.assertEqual(collector.env.steps, 2)

    def test_resume_restores_queue_rng_and_active_episode(self):
        collector = self.collector()
        collector.collect(1)
        saved = collector.state_dict()
        expected = collector.collect(4)
        collector.restore(saved)
        actual = collector.collect(4)
        for key in ("actions", "values", "rewards", "terminated", "truncated", "next_values"):
            np.testing.assert_array_equal(actual[key], expected[key])

    def test_tune_evaluation_restores_fit_state_even_after_exception(self):
        collector = self.collector()
        collector.collect(1)
        def evaluate(*args):
            collector.env.reset(None, None)
            collector.index = 999
            raise RuntimeError("interrupted tune")
        with patch("stage18_onpolicy_ppo.evaluate", evaluate):
            with self.assertRaisesRegex(RuntimeError, "interrupted tune"):
                evaluate_preserving_collector(collector, None, [])
        self.assertEqual((collector.index, collector.env.steps), (2, 1))

    def test_invalid_label_keeps_transition_and_masks_loss(self):
        collector = self.collector()
        collector.sources[0]["valid"][1] = False
        data = collector.collect(2)
        np.testing.assert_array_equal(data["valid"], [False, True])
        np.testing.assert_array_equal(data["rewards"], [0., 1.])
        self.assertEqual(collector.env.steps, 2)

    def test_empty_rollout_is_rejected(self):
        with self.assertRaises(ValueError):
            self.collector().collect(0)

    def test_auc_uses_strict_thresholds_and_initial_frame(self):
        self.assertAlmostEqual(success_auc([1.]), 100 * 20 / 21)
        self.assertAlmostEqual(success_auc([-1.]), 100 * 10 / 21)

    def test_selected_checkpoint_matches_atomic_selection_metadata_and_weights(self):
        torch = SimpleNamespace(equal=np.array_equal)
        weights = dict(memory=array([1.]), actor=array([2.]))
        selected = dict(contract_fingerprint="contract", steps=20000, tune_auc=84., model=weights)
        verify_selection(torch, selected, "contract", 20000, dict(memory=weights["memory"]), 84., weights)
        for change in (dict(contract_fingerprint="other"), dict(steps=40000), dict(tune_auc=85.),
                       dict(model=dict(memory=array([9.]), actor=array([2.]))),
                       dict(model=dict(memory=array([1.]), actor=array([9.])))):
            with self.subTest(change=change), self.assertRaises(ValueError):
                verify_selection(torch, dict(selected, **change), "contract", 20000,
                                 dict(memory=weights["memory"]), 84., weights)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
class UpdateTests(unittest.TestCase):
    def test_all_invalid_rollout_advances_without_optimization(self):
        from test_stage18_temporal import ModelEnvironmentTests
        from stage18_policy_runtime import configure_policy
        torch, modules, runtime = ModelEnvironmentTests().make_system()
        parameters = configure_policy(modules)
        runtime.read = lambda image: image
        source = dict(sequence="fit", paths=[np.zeros((32, 32, 3)) for _ in range(5)],
                      gt=np.tile([0, 0, 20, 20], (5, 1)), valid=np.zeros(5, bool),
                      terminal=True, bootstrap_path=None)
        collector = Collector(runtime, modules, [source], 1)
        before = {name: p.detach().clone() for name, p in modules.named_parameters()}
        optimizer = torch.optim.Adam(parameters, lr=3e-4)
        args = SimpleNamespace(gamma=.97, minibatch_size=4, seed=1)
        result = update_policy(torch, modules, optimizer, collector.collect(4), args, 1)
        self.assertTrue(result["skipped_update"])
        self.assertIsNone(result["mean_valid_iou"])
        self.assertEqual(collector.env.budget.steps, 4)
        self.assertEqual(result["maximum_gradient_norm"], 0.)
        self.assertFalse(optimizer.state)
        for name, p in modules.named_parameters():
            torch.testing.assert_close(before[name], p, rtol=0, atol=0)

    def test_live_rollout_updates_actor_but_not_representation(self):
        from test_stage18_temporal import ModelEnvironmentTests
        from stage18_policy_runtime import PolicyEnvironment, configure_policy
        torch, modules, runtime = ModelEnvironmentTests().make_system()
        parameters = configure_policy(modules)
        runtime.read = lambda image: image
        source = dict(sequence="fit", paths=[np.zeros((32, 32, 3)) for _ in range(33)],
                      gt=np.tile([0, 0, 20, 20], (33, 1)), valid=np.ones(33, bool),
                      terminal=True, bootstrap_path=None)
        collector = Collector(runtime, modules, [source], 1)
        self.assertIsInstance(collector.env, PolicyEnvironment)
        before = {name: p.detach().clone() for name, p in modules.named_parameters()}
        optimizer = torch.optim.Adam(parameters, lr=3e-4)
        args = SimpleNamespace(gamma=.97, minibatch_size=16, seed=1)
        result = update_policy(torch, modules, optimizer, collector.collect(32), args, 1)
        self.assertLess(result["likelihood_replay_error"], 1e-4)
        self.assertGreater(result["maximum_gradient_norm"], 0)
        self.assertFalse(torch.equal(before["actor.weight"], modules.actor.weight))
        for name, p in modules.named_parameters():
            if not p.requires_grad:
                torch.testing.assert_close(before[name], p, rtol=0, atol=0)
                self.assertIsNone(p.grad)


if __name__ == "__main__":
    unittest.main()
