"""Protocol/geometry tests run on CPU; torch scope tests run when available."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for folder in (REPO / "tools", REPO / "tracking"):
    sys.path.insert(0, str(folder))

import stage194_module_common as common
import stage194_module_ablation as run
import stage194_module_model as model_code
from stage28_response_common import dense_boxes


class ProtocolTests(unittest.TestCase):
    def test_group_separation_and_identical_draws(self):
        groups = [f"uav{i:07d}" for i in range(38)]
        folds = common.split_groups(groups)
        self.assertEqual(set(folds[0]["held"]), set(folds[1]["fit"]))
        self.assertFalse(set(folds[0]["fit"]) & set(folds[0]["held"]))
        sizes = {g: 48 for g in groups}
        first = common.draws_for(folds[0], 2132, sizes)
        self.assertEqual(first, common.draws_for(folds[0], 2132, sizes))
        self.assertNotEqual(first, common.draws_for(folds[0], 2133, sizes))
        self.assertTrue(all(g in folds[0]["fit"] and 0 <= i < 48 for row in first for g, i in row))
        self.assertEqual(len(first) * len(first[0]), common.DESIGN["updates"] * common.DESIGN["batch"])

    def test_sampling_is_uniform_without_truth_or_failure_inputs(self):
        np.testing.assert_array_equal(common.uniform_frames(3), [1, 2, 3])
        frames = common.uniform_frames(1000)
        self.assertEqual(len(frames), 48)
        self.assertEqual((frames[0], frames[-1]), (1, 1000))
        self.assertLessEqual(np.diff(frames).max() - np.diff(frames).min(), 1)

    def test_dense_decoder_matches_existing_image_decode(self):
        rng = np.random.default_rng(194)
        for parent, hw in (([20, 30, 10, 25], [80, 120]), ([0, 0, 90, 50], [80, 120]),
                           ([100, 70, 10, 10], [80, 120])):
            size = rng.uniform(.01, 1.5, (2, 16, 16)).astype(np.float32)
            offset = rng.uniform(-.4, 1.4, size.shape).astype(np.float32)
            observed = common.dense_image_boxes(size, offset, parent, 1.25, 256, hw)
            expected = dense_boxes(size, offset, parent, 1.25, 256, hw)
            np.testing.assert_allclose(observed, expected, atol=1e-4, rtol=0)

    def test_cached_identity_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "example.npz"
            x, reused = run.cache(p, {"fold": 0}, lambda: {"value": np.arange(3)})
            self.assertFalse(reused)
            _, reused = run.cache(p, {"fold": 0}, lambda: self.fail("Must reuse"))
            self.assertTrue(reused)
            with self.assertRaises(ValueError):
                run.cache(p, {"fold": 1}, lambda: {})
            p.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                run.cache(p, {"fold": 0}, lambda: {})

    def test_unsafe_output_is_rejected(self):
        base = REPO / "output" / "test_stage194_source"
        with self.assertRaises(ValueError):
            common.safe_output(base / "nested", [base])
        with self.assertRaises(ValueError):
            common.safe_output(base.parent, [base])

    def test_seed_aggregation_does_not_treat_seeds_as_videos(self):
        rows = []
        for seq in ("a", "b", "c"):
            for arm in common.ARMS:
                seeds = [0] if arm in common.CONTROLS else common.DESIGN["seeds"]
                for seed in seeds:
                    rows.append(dict(sequence=seq, arm=arm, seed=seed, introduced_failure=False,
                                     **{m: 1. + (seed == 2133) for m in common.METRICS}))
        policies, paired, factors = common.summarize(rows)
        self.assertEqual(len(policies), len(common.ARMS))
        row = next(r for r in paired if r["contrast"] == "a1_r1_q1_minus_native" and r["metric"] == "auc")
        self.assertEqual(row["videos"], 3)
        self.assertEqual(row["mean_delta"], .5)
        self.assertEqual(len(factors), 3)
        rows.pop()
        with self.assertRaises(ValueError):
            common.summarize(rows)

    def test_no_all_off_training_and_parameter_matched_flags(self):
        self.assertEqual(len(common.TRAINED), 9)
        self.assertNotIn("a0_r0_q0", common.TRAINED)
        self.assertEqual(common.flags("a1_r1_q1"), common.flags("current_a1_r1_q1"))
        with self.assertRaises(ValueError):
            common.flags("a1_r2_q0")


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable locally; server gradient smoke is mandatory")
class TorchTests(unittest.TestCase):
    def test_six_branch_objective_with_real_center_head(self):
        import torch
        from types import ModuleType, SimpleNamespace
        from unittest.mock import patch
        # Load the real head without importing the unrelated timm backbone.
        package = ModuleType("lib.models")
        package.__path__ = [str(REPO / "lib/models")]
        with patch.dict(sys.modules, {"lib.models": package}):
            from lib.models.layers.head import CenterPredictor

        torch.set_num_threads(1)
        torch.manual_seed(194)
        backbone = torch.nn.Module()
        backbone.blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(12)])
        backbone.norm = torch.nn.LayerNorm(8)
        backbone.cat_mode = "direct"

        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.box_head = CenterPredictor(8, channel=16, feat_sz=4)

            def forward_head(self, feature, gt_score_map=None):
                search = feature[:, -16:].transpose(1, 2).reshape(-1, 8, 4, 4)
                score, box, size, offset = self.box_head(search, gt_score_map)
                return dict(score_map=score, pred_boxes=box[:, None], size_map=size, offset_map=offset)

        net = Network().eval().requires_grad_(False)
        rt = SimpleNamespace(torch=torch, backbone=backbone, net=net, lens_z=4, lens_x=16, search_side=4,
                             s1=SimpleNamespace(recover_tokens=lambda x, *args, **kwargs: x),
                             tracker=SimpleNamespace(params=SimpleNamespace(search_size=256), output_window=torch.ones(4, 4)))
        tensors = [torch.randn(2, 20, 8), torch.tensor([[.375, .375, .25, .25]] * 2),
                   torch.tensor([[20., 20., 16., 16.]] * 2, dtype=torch.float64),
                   torch.full((2,), 4., dtype=torch.float64), torch.full((2, 2), 96., dtype=torch.float64),
                   torch.tensor([[20., 20., 16., 16.]] * 2, dtype=torch.float64)]
        original = run.model_digest(net)
        for arm in common.TRAINED:
            model = model_code.build_modules(torch, 8, 4, arm)
            loss, stats = model_code.train_objective(rt, model, arm, *tensors)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(p.grad is None for p in net.parameters()))
            for name, active in zip(("adapter", "router", "quality"), common.flags(arm)):
                has_gradient = any(p.grad is not None and p.grad.abs().max() > 0 for p in getattr(model, name).parameters())
                self.assertEqual(has_gradient, bool(active))
            self.assertEqual(original, run.model_digest(net))

    def test_adapter_identity_and_gradient_scope(self):
        import torch
        torch.manual_seed(194)
        tokens = torch.randn(2, 10, 8)
        for arm in common.TRAINED:
            model = model_code.build_modules(torch, 8, 4, arm)
            self.assertTrue(torch.equal(model.adapt(tokens), tokens))
            features = torch.randn(2, 8, 3, 2)
            raw = torch.rand(2, 1, 3, 2) * .9 + .05
            self.assertTrue(torch.equal(model.score(features, raw), raw))
            a, r, q = common.flags(arm)
            loss = 0
            if a:
                loss = loss + (model.adapt(tokens) - tokens - 1).square().mean()
            if r:
                loss = loss + model.route(tokens).square().mean()
            if q:
                loss = loss + model.score(features, raw).square().mean()
            loss.backward()
            for name, active in zip(("adapter", "router", "quality"), (a, r, q)):
                gradients = [p.grad for p in getattr(model, name).parameters()]
                self.assertEqual(any(g is not None and g.abs().max() > 0 for g in gradients), bool(active))

    def test_exact_clipped_iou_labels(self):
        import torch
        rng = np.random.default_rng(195)
        size = rng.uniform(.05, 1., (2, 2, 4, 4)).astype(np.float32)
        offset = rng.uniform(0, 1., size.shape).astype(np.float32)
        parent = np.array([[20, 30, 15, 25], [80, 70, 12, 10]], np.float32)
        factor = np.array([1.25, .75], np.float32)
        hw = np.array([[120, 160], [120, 160]], np.float32)
        truth = np.array([[15, 20, 20, 35], [80, 60, 20, 30]], np.float32)
        labels = model_code.dense_quality(torch, {"size_map": torch.tensor(size), "offset_map": torch.tensor(offset)},
            torch.tensor(parent), torch.tensor(factor), 256, torch.tensor(hw), torch.tensor(truth)).numpy()
        for i in range(2):
            grid = common.dense_image_boxes(size[i], offset[i], parent[i], factor[i], 256, hw[i])
            expected = run.tracking_iou(grid, truth[i]).reshape(4, 4)
            np.testing.assert_allclose(labels[i, 0], expected, atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
