"""Stage-18 GAE boundaries, valid-only normalization, and PPO math/gradients."""

import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking.stage18_ppo_math import gae, normalize_advantages, ppo_objective


class RolloutMathTests(unittest.TestCase):
    def test_import_without_torch_or_tracker(self):
        code = """
import builtins
import importlib.util
import sys
original = builtins.__import__
def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split('.')[0] in {'torch', 'torchvision', 'lib'}:
        raise AssertionError('Eager dependency: ' + name)
    return original(name, globals, locals, fromlist, level)
builtins.__import__ = guarded
spec = importlib.util.spec_from_file_location('ppo_math', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert all(callable(getattr(module, name)) for name in
           ('gae', 'normalize_advantages', 'ppo_objective'))
"""
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", code,
             str(ROOT / "tracking/stage18_ppo_math.py")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_terminal_vs_truncated_bootstrap_and_terminal_precedence(self):
        for terminal, truncated, expected in ((True, False, -1),
                                               (False, True, 3.5),
                                               (True, True, -1),
                                               (False, False, 3.5)):
            with self.subTest(terminal=terminal, truncated=truncated):
                advantage, returns = gae([1], [2], [5], [terminal], [truncated], gamma=.9)
                np.testing.assert_allclose(advantage, [expected])
                np.testing.assert_allclose(returns, [expected + 2])

    def test_two_episodes_do_not_leak_at_either_boundary(self):
        for terminal in (True, False):
            with self.subTest(terminal=terminal):
                rewards = np.array([1., 2., 100., 200.])
                values = np.array([10., 20., 30., 40.])
                next_values = np.array([20., 7., 40., 9.])
                terminated = np.array([False, terminal, False, False])
                truncated = np.array([False, not terminal, False, True])
                original = [x.copy() for x in (rewards, values, next_values, terminated, truncated)]
                actual = gae(*original, gamma=.5, gae_lambda=.5)
                expected_first = [-3.5, -18] if terminal else [-2.625, -14.5]
                np.testing.assert_allclose(actual[0], expected_first + [131.125, 164.5])
                np.testing.assert_allclose(actual[1], actual[0] + values)
                first = gae(*(x[:2] for x in original), gamma=.5, gae_lambda=.5)
                np.testing.assert_array_equal(actual[0][:2], first[0])
                np.testing.assert_array_equal(actual[1][:2], first[1])
                for before, after in zip(original, (rewards, values, next_values,
                                                    terminated, truncated)):
                    np.testing.assert_array_equal(before, after)

    def test_default_discount_lambda_and_implicit_rollout_tail(self):
        advantages, returns = gae([1, 2], [.5, .7], [.7, 10],
                                 [False, False], [False, False])
        tail = 2 + .97 * 10 - .7
        np.testing.assert_allclose(advantages, [1 + .97 * .7 - .5 + .97 * .95 * tail, tail])
        np.testing.assert_allclose(returns, advantages + [.5, .7])
        explicit = gae([1, 2], [.5, .7], [.7, 10],
                       [False, False], [False, True])
        np.testing.assert_array_equal(advantages, explicit[0])

    def test_zero_lambda_is_one_step_td_and_zero_discount_ignores_future(self):
        for gamma, lam, expected in ((.5, 0, [2., 3.5]), (0, .95, [-1., -1.])):
            with self.subTest(gamma=gamma, lam=lam):
                actual, _ = gae([1, 2], [2, 3], [6, 9],
                                [False, False], [False, False], gamma=gamma, gae_lambda=lam)
                np.testing.assert_allclose(actual, expected)

    def test_normalization_uses_only_valid_rows_and_does_not_mutate(self):
        advantages = np.array([1., 1e30, 3., -1e30])
        before = advantages.copy()
        actual = normalize_advantages(advantages, [True, False, True, False])
        np.testing.assert_allclose(actual, [-1, 0, 1, 0])
        np.testing.assert_array_equal(advantages, before)

    def test_constant_singleton_and_large_finite_normalization(self):
        for advantages, valid in (([7, 99, 7], [True, False, True]),
                                  ([7, 99], [True, False]),
                                  ([0, 0], [True, True])):
            with self.subTest(advantages=advantages):
                np.testing.assert_array_equal(normalize_advantages(advantages, valid),
                                              np.zeros(len(advantages)))
        np.testing.assert_allclose(normalize_advantages([-1e308, 1e308], [True, True]), [-1, 1])

    def test_normalization_rejects_no_valid_nonfinite_and_bad_vectors(self):
        for advantages, valid in (([1, 2], [False, False]),
                                  ([math.nan, 2], [False, True]),
                                  ([1, math.inf], [True, False]),
                                  ([1, -math.inf], [True, True]),
                                  ([[1, 2]], [True, True]),
                                  ([1, 2], [True]), ([], [])):
            with self.subTest(advantages=advantages, valid=valid):
                with self.assertRaises(ValueError):
                    normalize_advantages(advantages, valid)
        with self.assertRaises(TypeError):
            normalize_advantages([1, 2], [1, 0])

    def test_gae_rejects_malformed_nonfinite_and_overflow(self):
        base = dict(rewards=[1., 2.], values=[0., 0.], next_values=[3., 4.],
                    terminated=[False, True], truncated=[False, False])
        invalid = [dict(rewards=[]), dict(values=[0]), dict(next_values=[[1, 2]]),
                   dict(terminated=[True]), dict(truncated=[False]),
                   dict(rewards=[math.nan, 1]), dict(values=[0, math.inf]),
                   dict(next_values=[1, -math.inf]), dict(gamma=-.1), dict(gamma=1.1),
                   dict(gae_lambda=math.nan), dict(gae_lambda=1.1),
                   dict(rewards=[1e308, 1e308], values=[-1e308, 0])]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                gae(**dict(base, **change))
        for name in ("terminated", "truncated"):
            with self.subTest(name=name), self.assertRaises(TypeError):
                gae(**dict(base, **{name: [0, 1]}))


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
class PPOObjectiveTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch

    def batch(self, ratios=(1.5, .5, 1.1, .9, 1.5, .5), dtype=None):
        torch = self.torch
        dtype = dtype or torch.float64
        size = len(ratios)
        return dict(
            logits=torch.tensor([[math.log(.3), math.log(.7)]] * size,
                                dtype=dtype, requires_grad=True),
            values=torch.zeros(size, dtype=dtype, requires_grad=True),
            actions=torch.zeros(size, dtype=torch.int64),
            old_log_prob=torch.tensor([math.log(.3 / r) for r in ratios],
                                      dtype=dtype, requires_grad=True),
            returns=torch.zeros(size, dtype=dtype, requires_grad=True),
            advantages=torch.ones(size, dtype=dtype, requires_grad=True),
            valid=torch.ones(size, dtype=torch.bool))

    def assert_finite_backward(self, loss, batch):
        torch = self.torch
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name in ("logits", "values"):
            self.assertIsNotNone(batch[name].grad)
            self.assertTrue(torch.isfinite(batch[name].grad).all())
        for name in ("old_log_prob", "returns", "advantages"):
            self.assertIsNone(batch[name].grad, name)

    def test_known_ratios_signed_clipping_kl_and_detached_metrics(self):
        torch = self.torch
        batch = self.batch()
        batch["advantages"] = torch.tensor([2., -2., 2., -2., -2., 2.],
                                           dtype=torch.float64, requires_grad=True)
        loss, metrics = ppo_objective(torch, **batch, entropy_coefficient=0)
        expected_terms = np.array([2.4, -1.6, 2.2, -1.8, -3., 1.])
        ratios = np.array([1.5, .5, 1.1, .9, 1.5, .5])
        self.assertAlmostEqual(float(loss.detach()), -expected_terms.mean())
        self.assertAlmostEqual(float(metrics["policy_loss"]), -expected_terms.mean())
        self.assertEqual(float(metrics["value_loss"]), 0)
        self.assertAlmostEqual(float(metrics["entropy"]), -.3 * math.log(.3) - .7 * math.log(.7))
        self.assertAlmostEqual(float(metrics["approx_kl"]), np.mean(ratios - 1 - np.log(ratios)))
        self.assertAlmostEqual(float(metrics["clip_fraction"]), 4 / 6)
        self.assertEqual(set(metrics), {"policy_loss", "value_loss", "entropy",
                                        "approx_kl", "clip_fraction"})
        for tensor in metrics.values():
            self.assertTrue(torch.is_tensor(tensor))
            self.assertEqual(tensor.ndim, 0)
            self.assertFalse(tensor.requires_grad)
            self.assertIsNone(tensor.grad_fn)
        self.assert_finite_backward(loss, batch)
        expected_gradient = np.zeros((6, 2))
        active = np.array([0., 0., 2., -2., -2., 2.])
        expected_gradient[:, 0] = -active * ratios * .7 / 6
        expected_gradient[:, 1] = active * ratios * .7 / 6
        torch.testing.assert_close(batch["logits"].grad, torch.tensor(expected_gradient))

    def test_default_value_weight_is_half_raw_mse_without_another_half(self):
        torch = self.torch
        batch = self.batch(ratios=(1., 1.))
        batch["values"] = torch.tensor([1., 3.], dtype=torch.float64, requires_grad=True)
        batch["returns"] = torch.tensor([3., -1.], dtype=torch.float64, requires_grad=True)
        batch["advantages"] = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        loss, metrics = ppo_objective(torch, **batch, entropy_coefficient=0)
        self.assertEqual(float(metrics["value_loss"]), 10)
        self.assertEqual(float(loss.detach()), 5)
        self.assert_finite_backward(loss, batch)
        torch.testing.assert_close(batch["values"].grad, torch.tensor([-1., 2.], dtype=torch.float64))

    def test_default_entropy_weight_and_gradient(self):
        torch = self.torch
        batch = self.batch(ratios=(1.,))
        batch["advantages"] = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        loss, metrics = ppo_objective(torch, **batch)
        entropy = -.3 * math.log(.3) - .7 * math.log(.7)
        self.assertAlmostEqual(float(loss.detach()), -.01 * entropy)
        self.assert_finite_backward(loss, batch)
        expected = .01 * np.array([[.3, .7]]) * (np.log([[.3, .7]]) + entropy)
        torch.testing.assert_close(batch["logits"].grad, torch.tensor(expected))
        self.assertGreater(float(metrics["entropy"]), 0)

    def test_only_fast_legal_is_exact_deterministic_distribution(self):
        torch = self.torch
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                batch = self.batch(ratios=(1.,), dtype=dtype)
                batch["logits"] = torch.tensor([[0., -math.inf, -math.inf]], dtype=dtype,
                                                requires_grad=True)
                batch["old_log_prob"] = torch.zeros(1, dtype=dtype, requires_grad=True)
                loss, metrics = ppo_objective(torch, **batch)
                self.assertEqual(float(loss.detach()), -1)
                for name in ("entropy", "approx_kl", "clip_fraction"):
                    self.assertEqual(float(metrics[name]), 0)
                self.assert_finite_backward(loss, batch)
                self.assertTrue(torch.equal(batch["logits"].grad, torch.zeros_like(batch["logits"])))

    def test_valid_and_action_masks_match_exact_subset_with_zero_gradients(self):
        torch = self.torch
        batch = self.batch(ratios=(1., 1., 1.))
        batch["logits"] = torch.tensor([[math.log(.3), math.log(.7), -math.inf],
                                        [1., 2., -math.inf], [0., -math.inf, -math.inf]],
                                       dtype=torch.float64, requires_grad=True)
        batch["valid"] = torch.tensor([True, False, True])
        for name, data in dict(values=[1., 1e200, 2.], returns=[2., -1e200, 0.],
                               advantages=[1., 1e200, -2.],
                               old_log_prob=[math.log(.3), -1e200, 0.]).items():
            batch[name] = torch.tensor(data, dtype=torch.float64, requires_grad=True)
        subset = {name: tensor.detach()[batch["valid"]].clone() for name, tensor in batch.items()}
        for name in ("logits", "values"):
            subset[name].requires_grad_()
        loss, metrics = ppo_objective(torch, **batch)
        subset_loss, subset_metrics = ppo_objective(torch, **subset)
        torch.testing.assert_close(loss, subset_loss, rtol=0, atol=0)
        for name in metrics:
            torch.testing.assert_close(metrics[name], subset_metrics[name], rtol=0, atol=0)
        self.assert_finite_backward(loss, batch)
        subset_loss.backward()
        for name in ("logits", "values"):
            torch.testing.assert_close(batch[name].grad[batch["valid"]], subset[name].grad,
                                       rtol=0, atol=0)
            self.assertTrue((batch[name].grad[~batch["valid"]] == 0).all())
        self.assertTrue((batch["logits"].grad[:, 2] == 0).all())

    def test_kl_is_nonnegative_near_equal_policies(self):
        torch = self.torch
        for offset in (0., -1e-8, 1e-8):
            with self.subTest(offset=offset):
                batch = self.batch(ratios=(1.,))
                batch["old_log_prob"] = torch.tensor([math.log(.3) - offset], dtype=torch.float64)
                _, metrics = ppo_objective(torch, **batch)
                self.assertGreaterEqual(float(metrics["approx_kl"]), 0)
                self.assertAlmostEqual(float(metrics["approx_kl"]), math.expm1(offset) - offset,
                                       delta=1e-23)

    def test_rejects_bad_actions_logits_and_masks_even_on_excluded_rows(self):
        torch = self.torch
        cases = [dict(actions=torch.tensor([-1, 0])), dict(actions=torch.tensor([0, 2])),
                 dict(actions=torch.tensor([0., 0.])), dict(actions=torch.tensor([False, False])),
                 dict(actions=torch.tensor([[0], [0]])), dict(actions=[0, 0]),
                 dict(logits=torch.tensor([[0., 0.], [math.nan, 0.]])),
                 dict(logits=torch.tensor([[0., 0.], [0., math.inf]])),
                 dict(logits=torch.tensor([[0., 0.], [-math.inf, 0.]])),
                 dict(logits=torch.tensor([[0., 0.], [-math.inf, -math.inf]])),
                 dict(logits=torch.tensor([[0., 0.], [0., -math.inf]]), actions=torch.tensor([0, 1])),
                 dict(logits=torch.zeros(2, dtype=torch.float64)),
                 dict(logits=torch.zeros((0, 2))), dict(logits=torch.zeros((2, 0))),
                 dict(logits=torch.zeros((2, 2), dtype=torch.int64)),
                 dict(valid=torch.zeros(2, dtype=torch.bool)), dict(valid=torch.tensor([True])),
                 dict(valid=torch.tensor([1, 0]))]
        for change in cases:
            with self.subTest(change=change):
                batch = self.batch(ratios=(1., 1.))
                batch["valid"] = torch.tensor([True, False])
                batch.update(change)
                with self.assertRaises((ValueError, TypeError)):
                    ppo_objective(torch, **batch)

    def test_rejects_invalid_value_vectors_and_nonfinite_objectives(self):
        torch = self.torch
        for name in ("values", "old_log_prob", "returns", "advantages"):
            for tensor in (torch.tensor([0., math.nan]), torch.tensor([0., math.inf]),
                           torch.zeros(2, dtype=torch.int64), torch.zeros((2, 1)),
                           torch.zeros(2, device="meta")):
                with self.subTest(name=name, tensor=tensor):
                    batch = self.batch(ratios=(1., 1.))
                    batch["valid"] = torch.tensor([True, False])
                    batch[name] = tensor
                    with self.assertRaises((ValueError, TypeError)):
                        ppo_objective(torch, **batch)
        for name, data in (("old_log_prob", [-1000.]), ("values", [1e200]),
                           ("advantages", [-1e308])):
            with self.subTest(name=name):
                batch = self.batch(ratios=(1.,))
                batch[name] = torch.tensor(data, dtype=torch.float64)
                if name == "advantages":
                    batch["old_log_prob"] = torch.tensor([-2.], dtype=torch.float64)
                with self.assertRaisesRegex(ValueError, "finite"):
                    ppo_objective(torch, **batch)

    def test_rejects_invalid_coefficients(self):
        for name, bad in (("clip_epsilon", -1), ("clip_epsilon", 1),
                          ("clip_epsilon", math.nan), ("value_coefficient", -.5),
                          ("entropy_coefficient", math.inf)):
            with self.subTest(name=name, bad=bad), self.assertRaises(ValueError):
                ppo_objective(self.torch, **self.batch(), **{name: bad})


if __name__ == "__main__":
    unittest.main()
