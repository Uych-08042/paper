"""Pair-budget matching, conditional-variance identities and complete CPU test."""
import contextlib
import csv
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "tools"), str(REPO / "tracking")]
import stage200_within_video as run


class Progress:
    def __init__(self): self.done, self.reused = 0, 0
    def advance(self, count=1): self.done += count


class PureTests(unittest.TestCase):
    def test_pair_schedule_scope_budget_and_with_replacement(self):
        groups = [f"uav{i:07d}" for i in range(38)]
        fold = run.protocol.split_groups(groups)[0]
        sizes = {g: 2 for g in groups}
        a = run.common.pair_draws(fold, 2132, sizes)
        b = run.common.pair_draws(fold, 2132, sizes)
        self.assertEqual(a, b)
        self.assertNotEqual(a, run.common.pair_draws(fold, 2133, sizes))
        self.assertEqual(len(a), 256)
        self.assertEqual(sum(len(run.common.flatten(batch)) for batch in a), 1024)
        self.assertTrue(all(len(batch) == 2 and all(g in fold["fit"] and 0 <= x < 2 and 0 <= y < 2 for g, x, y in batch) for batch in a))
        self.assertGreater(sum(x == y for batch in a for _, x, y in batch), 0)
        with self.assertRaises(ValueError):
            run.common.pair_draws(dict(fold=0, fit=[groups[0]], held=[groups[0]]), 2132, sizes)

    def test_real_source199_and_committed_cache_integrity(self):
        d, tables = run.load199(run.DEFAULT_BUNDLE)
        self.assertEqual(d["fingerprint"], run.SOURCE199)
        self.assertEqual(len(tables["frame_predictions"]), 21888)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.npz"
            run.cache(path, dict(plan="test"), lambda: dict(value=np.ones(3)))
            _, reused = run.cache(path, dict(plan="test"), lambda: self.fail("Must reuse"))
            self.assertTrue(reused)
            with self.assertRaises(ValueError): run.cache(path, dict(plan="other"), lambda: {})
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError): run.cache(path, dict(plan="test"), lambda: {})
            invalid = Path(tmp) / "source.txt"; invalid.write_bytes(b"not the reviewed source")
            with self.assertRaises(ValueError): run.load199(invalid)

    def test_video_decomposition_preserves_absolute_error_field(self):
        rng = np.random.default_rng(200)
        y, p = rng.uniform(0, 1, (7, 6)), rng.uniform(0, 1, (7, 6))
        value = run.common.decomposition(y, p, np.full(6, .5))
        metrics = run.metrics.errors(y, p)
        self.assertAlmostEqual(value["between_video_mse"] + value["within_video_mse"], metrics["relative_mse"])
        self.assertLess(value["video_decomposition_error"], 1e-12)
        self.assertNotIn("decomposition_error", value)


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU Torch required")
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def make_bank(self):
        groups = [f"uav{i:07d}" for i in range(38)]
        rng = np.random.default_rng(200)
        bank = {}
        for g in groups:
            tokens = rng.normal(size=(2, 6, 4)).astype(np.float32)
            bank[g] = dict(tokens=tokens,
                pooled=run.architecture.pooling.pool(self.torch, self.torch.as_tensor(tokens), 2).numpy(),
                labels=rng.uniform(.1, .9, (2, 6)), frame_id=np.array([1, 2]), sequence=g + "_test")
        return groups, bank

    def test_exhaustive_pair_variance_identity_and_offset_invariance(self):
        torch = self.torch
        p = torch.arange(18, dtype=torch.float64).reshape(3, 6) / 30
        y = torch.tensor([[.1, .7, .2, .4, .5, .6], [.6, .1, .3, .7, .2, .8], [.2, .8, .5, .1, .7, .3]], dtype=torch.float64)
        indices = [i for a in range(3) for b in range(3) for i in (a, b)]
        result = run.common.objectives(torch, p[indices], y[indices])
        e = p - y; e -= e.mean(1, keepdim=True)
        torch.testing.assert_close(result["within_video_difference"], (e - e.mean(0)).square().mean(), atol=1e-14, rtol=0)
        p4 = p[[0, 1, 1, 2]].clone().requires_grad_(True)
        y4 = y[[0, 1, 1, 2]]
        offset = torch.tensor([[.1, -.1, .05, -.05, 0, 0]] * 2 + [[-.04, .02, .03, -.01, 0, 0]] * 2, dtype=torch.float64)
        a = run.common.objectives(torch, p4, y4)
        b = run.common.objectives(torch, p4, y4 + offset)
        torch.testing.assert_close(a["within_video_difference"], b["within_video_difference"], atol=1e-14, rtol=0)
        self.assertGreater(abs(float(a["point"].detach() - b["point"].detach())), 1e-5)
        ga = torch.autograd.grad(a["within_video_difference"], p4, retain_graph=True)[0]
        gb = torch.autograd.grad(b["within_video_difference"], p4)[0]
        torch.testing.assert_close(ga, gb, atol=1e-14, rtol=0)
        self_pair = run.common.objectives(torch, p[[0, 0]], y[[0, 0]])
        self.assertEqual(float(self_pair["within_video_difference"]), 0.)

    def test_matched_fits_same_initialization_schedule_and_resume(self):
        groups, bank = self.make_bank(); fold = run.protocol.split_groups(groups)[0]
        stats = run.architecture.fit_statistics(bank, fold["fit"])
        draws = run.common.pair_draws(fold, 2132, {g: 2 for g in groups})
        with tempfile.TemporaryDirectory() as tmp:
            progress, artifacts, scopes = Progress(), {}, []
            for arm in run.ARMS:
                model, scope, history = run.fit(self.torch, Path(tmp), dict(fingerprint="test"), bank, stats, fold, 2132, arm, draws, 2, progress, artifacts)
                self.assertEqual(len(history), 256)
                self.assertEqual(scope["pairs"], 512)
                self.assertEqual(scope["examples"], 1024)
                self.assertTrue(scope["fixed_fit_buffers_unchanged"])
                self.assertGreater(scope["maximum_projection_gradient"], 0)
                self.assertTrue(all(not p.requires_grad for p in model.parameters()))
                scopes.append(scope)
            for key in ["initial_sha256", "pair_draws_sha256", "trainable_parameters", "initial_choice"]:
                self.assertEqual(scopes[0][key], scopes[1][key])
            _, scope, _ = run.fit(self.torch, Path(tmp), dict(fingerprint="test"), bank, stats, fold, 2132, run.ARMS[1], draws, 2, progress, artifacts)
            self.assertEqual(scope["final_sha256"], scopes[1]["final_sha256"])
            self.assertEqual(progress.reused, 256)
            checkpoint = Path(tmp) / f"training/fold{fold['fold']}/seed2132/{run.ARMS[1]}.pt"
            checkpoint.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                run.fit(self.torch, Path(tmp), dict(fingerprint="test"), bank, stats, fold, 2132, run.ARMS[1], draws, 2, progress, artifacts)

    def test_real_parameter_count_and_initial_prior(self):
        torch = self.torch
        stats = dict(prior=np.array([.7, .6, .5, .4, .3, .2]), mean_tokens=np.zeros((320, 192)), mean_pooled=np.zeros(768))
        models = [run.architecture.build(torch, stats, 64, "interaction_residual") for _ in run.ARMS]
        self.assertEqual(len({run.source.model_digest(m) for m in models}), 1)
        for model in models:
            self.assertEqual(sum(p.numel() for p in model.parameters()), 6720)
            self.assertEqual(int(run.metrics.choose(model(torch.zeros(1, 320, 192)).detach().numpy())[0]), 0)

    def test_synthetic_full_pipeline_same_pair_counts_no_cuda_and_tamper(self):
        torch = self.torch
        groups, bank = self.make_bank(); folds = run.protocol.split_groups(groups)
        entries = [dict(group=g, sequence=bank[g]["sequence"], sample_frames=[1, 2]) for g in groups]
        p194 = dict(fingerprint=run.inputs.SOURCE_PLAN, entries=entries, folds=folds, runtime=dict(tracker=dict(lens_z=2)))
        with tempfile.TemporaryDirectory() as tmp:
            r194, out, bundle = Path(tmp) / "194", Path(tmp) / "output", Path(tmp) / "source/199.txt"
            r194.mkdir(); bundle.parent.mkdir(); bundle.write_bytes(b"fixture")
            artifacts = {}
            for entry in entries:
                g, seq = entry["group"], entry["sequence"]
                for relative, data, meta in (
                    (f"features/{seq}.npz", dict(tokens=bank[g]["tokens"], frame_id=bank[g]["frame_id"]),
                     dict(plan_fingerprint=p194["fingerprint"], sequence=seq, kind="native_parent_prefix")),
                    (f"reference/{g}.npz", dict(iou=bank[g]["labels"]),
                     dict(plan_fingerprint=p194["fingerprint"], kind="fixed_layer_fit_labels", group=g))):
                    path = r194 / relative
                    run.source.cache(path, meta, lambda data=data: data)
                    run.source.record(r194, path, artifacts)
            old_fit = [dict(fold=str(f["fold"]), **{f"prior_L{k}": str(run.architecture.fit_statistics(bank, f["fit"])["prior"][k - 7]) for k in range(7, 13)}) for f in folds]
            old_frames = [dict(group=g, frame=str(i + 1), **{f"label_L{k}": str(bank[g]["labels"][i, k - 7]) for k in range(7, 13)}) for g in groups for i in range(2)]
            d194 = dict(fingerprint=run.inputs.SOURCE194, table_sha256={}, artifact_sha256=artifacts)
            reader = run.inputs.VerifiedSource(r194, d194)
            fixtures = (p194, d194, reader, {}, dict(fingerprint=run.SOURCE199), dict(frame_predictions=old_frames, fit_summary=old_fit))
            stdout = io.StringIO()
            with patch.object(run, "load_sources", return_value=fixtures), \
                 patch.object(sys, "argv", ["stage200", "--stage194-root", str(r194), "--stage199-bundle", str(bundle), "--output-dir", str(out)]), \
                 patch.dict(os.environ, {}), \
                 patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")), \
                 contextlib.redirect_stdout(stdout):
                run.main()
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
                run.verify_results(out)
                with patch.object(run, "load_sources", side_effect=AssertionError("Must reuse")):
                    run.main()
            d = run.source.read_sealed(out / "stage200_decision.json")
            self.assertEqual(d["training_runs"], 8); self.assertEqual(d["training_updates"], 2048)
            self.assertEqual(d["predicted_states"], 608); self.assertEqual(d["training_state_references"], 8192)
            self.assertEqual(d["training_pair_uses"], 4096); self.assertEqual(d["pair_schedule_rows"], 2048)
            self.assertEqual(len(d["primary_contrasts"]), 3); self.assertEqual(d["new_encoder_calls"], 0)
            self.assertIn("100.00%", stdout.getvalue())
            def read(name):
                with (out / f"stage200_{name}.csv").open(encoding="utf-8", newline="") as handle:
                    return list(csv.DictReader(handle))
            counts = {}
            for row in read("pair_schedule"):
                for side in ("a", "b"):
                    key = row["fold"], row["seed"], row["group"], row["frame_" + side]
                    counts[key] = counts.get(key, 0) + 1
            for row in read("frame_predictions"):
                if row["pool"] == "fit":
                    self.assertEqual(int(row["training_draw_count"]), counts.get((row["fold"], row["seed"], row["group"], row["frame"]), 0))
                else: self.assertEqual(row["training_draw_count"], "")
            self.assertEqual(len(read("paired")), 38)
            self.assertEqual(len(read("pair_support")), 76)
            for name, sha in artifacts.items(): self.assertEqual(run.file_hash(r194 / name), sha)
            (out / "stage200_policy_summary.csv").write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError): run.verify_results(out)


if __name__ == "__main__":
    unittest.main()
