"""Stage-18 budget, geometry, memory, and differentiable frozen-path tests."""

import importlib.util
import unittest
from types import SimpleNamespace
import numpy as np
from stage18_environment import TokenCredit, TemporalEnvironment, executed_crop_roi
from stage18_temporal_model import build_temporal_modules
from stage18_pilot import select_fit_pilot


class BudgetGeometryTests(unittest.TestCase):
    def test_pilot_excludes_tune_validation_and_short_videos(self):
        entries = [dict(sequence="s{:03d}".format(i), split="train", start=0, stop=100)
                   for i in range(120)]
        entries += [dict(sequence="validation", split="val", start=0, stop=100),
                    dict(sequence="short", split="train", start=0, stop=10)]
        pilot, fit, tune = select_fit_pilot(dict(sequences=entries), 5, 64, 2053)
        self.assertEqual((len(pilot), len(fit), len(tune)), (5, 96, 24))
        self.assertFalse({e["sequence"] for e in fit} & {e["sequence"] for e in tune})
        self.assertTrue(set(e["sequence"] for e in pilot) <= set(e["sequence"] for e in fit))
        self.assertNotIn("short", [e["sequence"] for e in fit + tune])

    def test_prefix_budget_and_no_forced_spending(self):
        for offset in range(8):
            budget = TokenCredit()
            for frame in range(300):
                budget.advance()
                action = 1 + frame % 6 if budget.credit >= 10 and frame % 8 != offset else 0
                budget.spend(action)
                self.assertLessEqual(budget.calls, (frame + 1) * 3 // 10)
                self.assertTrue(0 <= budget.credit <= 20)
        budget = TokenCredit()
        budget.advance()
        with self.assertRaises(ValueError):
            budget.spend(1)
        for _ in range(100):
            budget.advance()
            budget.spend(0)
        self.assertEqual(budget.calls, 0)

    def test_executed_roi_inverts_crop_mapping(self):
        previous = [20, 40, 20, 40]
        # Crop center=(30,60), side=100, origin=(-20,10).
        np.testing.assert_allclose(executed_crop_roi([0, 30, 40, 20], previous, 2.56, 256),
                                   [.2, .2, .4, .2])


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
class ModelEnvironmentTests(unittest.TestCase):
    def make_system(self):
        import torch
        import torch.nn as nn
        torch.manual_seed(9)
        modules = build_temporal_modules(torch, nn, 8, 4, 2)
        blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(12)])
        head = nn.Linear(8, 1)
        blocks.requires_grad_(False)
        head.requires_grad_(False)
        tracker = SimpleNamespace(state=[0., 0., 20., 20.], frame_id=0, head_calls=0,
                                  params=SimpleNamespace(search_size=32))
        def initialize(image, box):
            tracker.state, tracker.frame_id = list(box), 0
        def forward_head(features):
            tracker.head_calls += 1
            return {"score_map": head(features).sigmoid()}
        rt = SimpleNamespace(torch=torch, tracker=tracker, initialize=initialize,
                             backbone=SimpleNamespace(blocks=blocks),
                             prepare=lambda image: (image, 1.0),
                             prefix=lambda image: torch.arange(64).reshape(1, 8, 8).float() / 100
                                                 + float(image[0, 0, 0]),
                             head=forward_head,
                             decode=lambda output, factor, image: list(tracker.state))
        return torch, modules, rt

    def test_identity_and_predicted_roi_gradient(self):
        torch, modules, rt = self.make_system()
        tokens = torch.randn(1, 8, 8)
        memory = modules.initial_memory(tokens)
        torch.testing.assert_close(modules.read_memory(tokens, memory), tokens, rtol=0, atol=0)
        for action in range(7):
            torch.testing.assert_close(modules.adapt(tokens, action), tokens, rtol=0, atol=0)
        roi = torch.tensor([[.1, .1, .5, .5]])
        feature = torch.randn(1, 8, 8, requires_grad=True)
        modules.pool_target(feature, roi).sum().backward()
        self.assertGreater(float(feature.grad.abs().sum()), 0)

    def test_causal_clone_replay_and_live_memory(self):
        torch, modules, rt = self.make_system()
        with torch.no_grad():
            modules.read_out.weight.fill_(.01)
            env = TemporalEnvironment(rt, modules)
            image = np.zeros((32, 32, 3))
            env.reset(image, [0, 0, 20, 20])
            first = env.step(image, 0)
            self.assertEqual(rt.tracker.head_calls, 1)
            self.assertTrue(torch.isneginf(first["logits"][:, 1:]).all())
            snapshot = env.snapshot()
            before = snapshot["memory"].clone()
            env.step(image + 10, 0)
            torch.testing.assert_close(first["memory"], before, rtol=0, atol=0)
            env.restore(snapshot)
            original = env.step(image, 0)
            original_memory = env.memory.clone()
            env.restore(snapshot)
            replay = env.step(image, 0)
            torch.testing.assert_close(original["features"], replay["features"], rtol=0, atol=0)
            torch.testing.assert_close(original_memory, env.memory, rtol=0, atol=0)
            self.assertFalse(torch.equal(before, env.memory))
            env.reset(image, [0, 0, 20, 20])
            self.assertIsNone(env.memory)
            self.assertEqual(env.budget.credit, 0)

    def test_memory_writer_and_adapter_receive_gradients_through_frozen_block(self):
        for action in range(1, 7):
            with self.subTest(action=action):
                torch, modules, rt = self.make_system()
                with torch.no_grad():
                    modules.read_out.weight.fill_(.01)
                env = TemporalEnvironment(rt, modules)
                image = np.zeros((32, 32, 3))
                env.reset(image, [0, 0, 20, 20])
                for index in range(4):
                    result = env.step(image, action if index == 3 else 0)
                head_gradient = torch.autograd.grad(result["output"]["score_map"].sum(),
                                                    result["features"], retain_graph=True)[0]
                self.assertGreater(float(head_gradient.abs().sum()), 0)
                (result["features"].square().mean()
                 + result["output"]["score_map"].square().mean()).backward()
                for group in ("read_out", "update", "adapters.{}".format(action - 1)):
                    gradients = [p.grad for name, p in modules.named_parameters()
                                 if name.startswith(group) and p.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                    self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
                self.assertTrue(all(p.grad is None for p in rt.backbone.blocks.parameters()))


if __name__ == "__main__":
    unittest.main()
