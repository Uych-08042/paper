"""Structural order-control, gradient, resume and synthetic end-to-end tests."""
import contextlib
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
import stage197_spatial_router as run


class Progress:
    def __init__(self):
        self.done, self.reused = 0, 0

    def advance(self, count=1):
        self.done += count


class CacheTests(unittest.TestCase):
    def test_identity_tamper_and_output_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pred.npz"
            meta = dict(arm="ordered", plan="test")
            run.cache(path, meta, lambda: dict(prediction=np.ones((2, 6))))
            _, reused = run.cache(path, meta, lambda: self.fail("Committed cache should be reused"))
            self.assertTrue(reused)
            with self.assertRaises(ValueError):
                run.cache(path, dict(meta, arm="unordered"), lambda: {})
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                run.cache(path, meta, lambda: {})
            with self.assertRaises(ValueError):
                run.original.safe_output(Path(tmp) / "child", [Path(tmp)])


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU PyTorch required")
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.torch = torch

    def test_matched_initialization_and_real_parameter_budget(self):
        torch = self.torch
        a = run.architecture.build(torch, 192, 64, 320, "ordered")
        b = run.architecture.build(torch, 192, 64, 320, "unordered")
        self.assertEqual(run.source.model_digest(a), run.source.model_digest(b))
        self.assertEqual(sum(p.numel() for p in a.parameters()), 6727)
        self.assertEqual(a.readout.in_features, 1088)
        self.assertEqual(sum(p.numel() for p in run.architecture.build(torch, 192, 64, 320, "global_stats").parameters()), 4614)
        tokens = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4)
        small = run.architecture.build(torch, 4, 2, 6, "ordered")
        torch.testing.assert_close(small.features(tokens)[:, 16:], tokens[:, :, 0])
        torch.testing.assert_close(small(tokens), torch.full((2, 6), .5))

    def test_order_removal_is_per_sample_and_stream_not_global_permutation(self):
        torch = self.torch
        tokens = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4) / 10
        perm = torch.tensor([1, 0, 4, 2, 5, 3])
        pooled = run.prior.pool(torch, tokens, 2)
        a = run.architecture.build(torch, 4, 2, 6, "ordered")
        b = run.architecture.build(torch, 4, 2, 6, "unordered")
        with torch.no_grad():
            a.projection.weight.copy_(torch.tensor([[.2, -.3, .7, .1]]))
            a.readout.weight.copy_(torch.arange(6 * 22).reshape(6, 22) / 1000)
            b.load_state_dict(a.state_dict())
        # Fix the shared global values to isolate layout independently of
        # floating-point reduction order in the secondary global-stat inputs.
        torch.testing.assert_close(b.features(tokens, pooled), b.features(tokens[:, perm], pooled), rtol=0, atol=0)
        self.assertFalse(torch.equal(a.features(tokens, pooled), a.features(tokens[:, perm], pooled)))
        self.assertFalse(torch.equal(a(tokens, pooled), a(tokens[:, perm], pooled)))
        proof = run.architecture.layout_audit(torch, tokens, pooled, 2)
        self.assertEqual(proof["unordered_permutation_error"], 0)
        self.assertEqual(proof["ordered_changed_states"], 2)
        # Equal scalar multisets with different addresses produce identical
        # unordered features even when each sample has a different permutation.
        other = tokens.clone()
        other[0] = tokens[0, perm]
        other[1] = tokens[1, torch.tensor([0, 1, 5, 4, 3, 2])]
        torch.testing.assert_close(b.features(other, pooled), b.features(tokens, pooled), rtol=0, atol=0)

    def test_global_control_equals_original_routing_function(self):
        torch = self.torch
        baseline = run.prior.architecture.build_modules(torch, 4, 2, run.metrics.ARM)
        model = run.architecture.build(torch, 4, 2, 6, "global_stats")
        with torch.no_grad():
            baseline.router.weight.copy_(torch.arange(96).reshape(6, 16) / 1000)
            model.readout.load_state_dict(baseline.router.state_dict())
        tokens = torch.arange(48, dtype=torch.float32).reshape(2, 6, 4) / 10
        torch.testing.assert_close(model(tokens), baseline.route(tokens), rtol=0, atol=0)

    def bank(self):
        groups = [f"uav{i:07d}" for i in range(38)]
        rng = np.random.default_rng(197)
        bank = {}
        for g in groups:
            tokens = rng.normal(size=(2, 6, 4)).astype(np.float32)
            bank[g] = dict(tokens=tokens,
                pooled=run.prior.pool(self.torch, self.torch.as_tensor(tokens), 2).numpy(),
                labels=np.tile([.7, .4, .6, .3, .5, .2], (2, 1)), frame_id=np.array([1, 2]))
        return groups, bank

    def test_fresh_global_training_reproduces_stage196_absolute(self):
        torch = self.torch
        groups, bank = self.bank()
        fold = dict(fold=0, fit=groups[:19], held=groups[19:])
        with tempfile.TemporaryDirectory() as tmp:
            old, _, _, _ = run.prior.fit(torch, Path(tmp) / "196", dict(fingerprint="fixture"), bank,
                fold, 2132, "absolute", 2, Progress(), {})
            new, _, _, _ = run.fit(torch, Path(tmp) / "197", dict(fingerprint="fixture"), bank,
                fold, 2132, "global_stats", 2, Progress(), {})
            torch.testing.assert_close(old.router.weight, new.readout.weight, rtol=0, atol=0)
            torch.testing.assert_close(old.router.bias, new.readout.bias, rtol=0, atol=0)
            tokens = torch.as_tensor(bank[groups[0]]["tokens"])
            torch.testing.assert_close(old.route(tokens), new(tokens), rtol=0, atol=0)

    def test_training_gradient_match_resume_and_checkpoint_tampering(self):
        groups, bank = self.bank()
        fold = dict(fold=0, fit=groups[:19], held=groups[19:])
        with tempfile.TemporaryDirectory() as tmp:
            progress, artifacts, scopes = Progress(), {}, []
            for arm in ("ordered", "unordered"):
                model, scope, history, _ = run.fit(self.torch, Path(tmp), dict(fingerprint="test"), bank,
                                                  fold, 2132, arm, 2, progress, artifacts)
                scopes.append(scope)
                self.assertEqual(len(history), 256)
                self.assertGreater(scope["maximum_projection_gradient"], 0)
                self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            for key in ("draws_sha256", "initial_sha256", "trainable_parameters"):
                self.assertEqual(scopes[0][key], scopes[1][key])
            _, scope, _, _ = run.fit(self.torch, Path(tmp), dict(fingerprint="test"), bank, fold, 2132,
                                     "unordered", 2, progress, artifacts)
            self.assertEqual(progress.reused, 256)
            self.assertEqual(scope["final_sha256"], scopes[1]["final_sha256"])
            path = Path(tmp) / "training/fold0/seed2132/unordered.pt"
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                run.fit(self.torch, Path(tmp), dict(fingerprint="test"), bank, fold, 2132, "unordered", 2, progress, artifacts)

    def test_complete_synthetic_pipeline_no_cuda_and_immutable_sources(self):
        torch = self.torch
        groups, bank = self.bank()
        folds = [dict(fold=0, fit=groups[:19], held=groups[19:]), dict(fold=1, fit=groups[19:], held=groups[:19])]
        entries = [dict(sequence=g + "_00000_s", group=g, sample_frames=[1, 2]) for g in groups]
        plan = dict(fingerprint=run.audit.SOURCE_PLAN, entries=entries, folds=folds, runtime=dict(tracker=dict(lens_z=2)))
        with tempfile.TemporaryDirectory() as tmp:
            r194, r195, r196, out = [Path(tmp) / n for n in ("194", "195", "196", "new")]
            for root in (r194, r195, r196):
                root.mkdir()
            run.publish_identical(r194 / "stage194_decision.json", dict(fixture=True))
            run.publish_identical(r196 / "stage196_decision.json", dict(fixture=True))
            artifacts, scopes, old_frames = {}, [], []
            for entry in entries:
                g, seq = entry["group"], entry["sequence"]
                for relative, data, metadata in (
                    (f"features/{seq}.npz", dict(frame_id=bank[g]["frame_id"], tokens=bank[g]["tokens"]),
                     dict(plan_fingerprint=plan["fingerprint"], sequence=seq, kind="native_parent_prefix")),
                    (f"reference/{g}.npz", dict(iou=bank[g]["labels"]),
                     dict(plan_fingerprint=plan["fingerprint"], kind="fixed_layer_fit_labels", group=g))):
                    path = r194 / relative
                    run.source.cache(path, metadata, lambda data=data: data)
                    run.source.record(r194, path, artifacts)
            for fold in folds:
                for seed in run.original.DESIGN["seeds"]:
                    scopes.append(dict(fold=str(fold["fold"]), seed=str(seed), arm=run.metrics.ARM,
                        draws_sha256=run.fingerprint(run.original.draws_for(fold, seed, {g: 2 for g in groups}))))
                    for g in groups:
                        for frame in (1, 2):
                            old_frames.append(dict(loss_name="absolute", fold=str(fold["fold"]), seed=str(seed),
                                group=g, frame=str(frame), cpu_layer="7", **{f"prediction_L{k}": ".5" for k in range(7, 13)}))
            decision = dict(fingerprint=run.audit.SOURCE194, artifact_sha256=artifacts, table_sha256={})
            reader = run.audit.VerifiedSource(r194, decision)
            inputs = (plan, decision, reader, dict(scope_summary=scopes), dict(fingerprint=run.SOURCE196), old_frames)
            with patch.object(run, "load_inputs", return_value=inputs), \
                 patch.object(sys, "argv", ["stage197", "--stage194-root", str(r194), "--stage195-root", str(r195),
                    "--stage196-root", str(r196), "--output-dir", str(out)]), \
                 patch.dict(os.environ, {}), \
                 patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")), \
                 contextlib.redirect_stdout(io.StringIO()):
                run.main()
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
                # Finished results are reused without another source read/train.
                with patch.object(run, "load_inputs", side_effect=AssertionError("Must reuse")):
                    run.main()
            d = run.source.read_sealed(out / "stage197_decision.json")
            self.assertEqual(d["training_runs"], 12)
            self.assertEqual(d["training_updates"], 3072)
            self.assertEqual(d["predicted_states"], 912)
            self.assertEqual(d["new_image_reads"], 0)
            self.assertEqual(d["new_full_trajectories"], 0)
            self.assertEqual(d["primary_contrast"]["contrast"], "ordered_minus_unordered")
            self.assertLessEqual(d["prepooled_interface_maximum_error"], 1e-7)
            self.assertEqual(len(run.prior.read_csv(out / "stage197_paired.csv")), 72)
            self.assertEqual(len(run.prior.read_csv(out / "stage197_input_audit.csv")), 38)
            for relative, sha in artifacts.items():
                self.assertEqual(run.file_hash(r194 / relative), sha)
            with contextlib.redirect_stdout(io.StringIO()):
                run.verify_results(out)
            (out / "stage197_policy_summary.csv").write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                run.verify_results(out)


if __name__ == "__main__":
    unittest.main()
