"""Representation warm-up contracts and real-loss temporal backpropagation."""

import importlib.util
import contextlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from stage4_common import fingerprint
from stage18_representation_train import (RepresentationEnvironment, RepresentationCredit,
    behavior_actions, clip_loss, configure_representation, load_pilot_contract)
from stage18_representation_train import profile_references
from stage18_environment import TokenCredit


class WarmupContractTests(unittest.TestCase):
    def test_behavior_is_reproducible_balanced_and_has_no_deployment_cap(self):
        actions = behavior_actions(2053, "video", 1, 60000)
        np.testing.assert_array_equal(actions, behavior_actions(2053, "video", 1, 60000))
        self.assertFalse(np.array_equal(actions, behavior_actions(2053, "video", 2, 60000)))
        counts = np.bincount(actions, minlength=7) / len(actions)
        np.testing.assert_allclose(counts, [.5] + [1 / 12] * 6, atol=.01)
        credit = RepresentationCredit()
        for action in actions[:100]:
            credit.advance()
            credit.spend(int(action))
        self.assertGreater(credit.calls, 30)

    def test_reset_replaces_only_warmup_budget(self):
        rt = SimpleNamespace(initialize=lambda image, box: None)
        env = RepresentationEnvironment(rt, object())
        env.reset(None, [0, 0, 20, 20])
        self.assertIsInstance(env.budget, RepresentationCredit)
        self.assertEqual(env.budget.calls, 0)

    def test_pilot_decision_must_match_verified_contract(self):
        contract = dict(code_sha256={"module.py": "digest"})
        contract["fingerprint"] = fingerprint(contract)
        decision = dict(status="PASS_EXECUTION_PILOT", contract_fingerprint=contract["fingerprint"])
        with patch("pathlib.Path.read_text", side_effect=[json.dumps(contract), json.dumps(decision)]), \
                patch("stage18_representation_train.file_hash", return_value="digest"):
            self.assertEqual(load_pilot_contract("pilot"), contract)
        decision["contract_fingerprint"] = "wrong"
        with patch("pathlib.Path.read_text", side_effect=[json.dumps(contract), json.dumps(decision)]):
            with self.assertRaises(ValueError):
                load_pilot_contract("pilot")

    def test_resource_comparison_uses_independent_predicted_trajectories(self):
        seen = []
        tracker = SimpleNamespace(state=[0, 0, 20, 20], frame_id=0)
        def initialize(image, box):
            tracker.state, tracker.frame_id = list(box), 0
        def run_image(image, method):
            seen.append((method, tracker.state[0]))
            return [tracker.state[0] + 1, 0, 20, 20]
        rt = SimpleNamespace(torch=SimpleNamespace(no_grad=contextlib.nullcontext,
                             cuda=SimpleNamespace(synchronize=lambda: None)),
                             tracker=tracker, initialize=initialize,
                             read=lambda path: None, run_image=run_image)
        class FakeEnvironment:
            def __init__(self, runtime, modules):
                self.budget = TokenCredit()
            def reset(self, image, box):
                initialize(image, box)
            def step(self, image, action):
                self.budget.advance()
                self.budget.spend(action)
                tracker.state = run_image(image, "new")
        source = dict(sequence="s", paths=["frame"] * 65, gt=[[0, 0, 20, 20]])
        with patch("stage18_representation_train.TemporalEnvironment", FakeEnvironment):
            rows, resource = profile_references(rt, SimpleNamespace(eval=lambda: None), [source], repeats=2)
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["frames"] == 44 for row in rows))
        for method in ("fast", "sgla", "new"):
            self.assertEqual([x for name, x in seen if name == method], list(range(64)) * 2)
        self.assertGreater(resource["memory_to_sgla_ratio"], 0)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
class TemporalWarmupTests(unittest.TestCase):
    def test_real_loss_multiple_backward_chunks_keep_base_frozen(self):
        from test_stage18_temporal import ModelEnvironmentTests
        torch, modules, rt = ModelEnvironmentTests().make_system()
        configure_representation(modules)
        with torch.no_grad():
            modules.read_out.weight.fill_(.01)
        initializations = []
        original_initialize = rt.initialize
        def initialize(image, box):
            initializations.append(list(box))
            original_initialize(image, box)
        rt.initialize = initialize
        rt.read = lambda path: np.zeros((32, 32, 3))
        def head(features):
            search = features[:, 4:]
            return dict(pred_boxes=search.mean(1)[:, :4].sigmoid()[:, None],
                        score_map=search[:, :, 0].sigmoid().reshape(1, 1, 2, 2))
        rt.head = head
        gt = np.tile([0., 0., 20., 20.], (9, 1))
        gt[2] = np.nan
        gt[7] = [200, 200, 20, 20]
        valid = np.ones(9, dtype=bool)
        valid[2] = False
        source = dict(sequence="toy", paths=["frame"] * 9, gt=gt, valid=valid)
        stats = clip_loss(rt, modules, source, np.arange(8) % 7, 3, backward_scale=1.)
        self.assertTrue(np.isfinite(stats["mean_loss"]))
        self.assertEqual(stats["valid_frames"], 7)
        self.assertLess(stats["regression_frames"], 7)
        self.assertEqual(len(initializations), 1)
        self.assertTrue(all(p.grad is None for p in rt.backbone.blocks.parameters()))
        self.assertTrue(all(p.grad is None for p in modules.actor.parameters()))
        self.assertGreater(float(modules.update.weight_hh.grad.abs().sum()), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in modules.parameters() if p.grad is not None))


if __name__ == "__main__":
    unittest.main()
