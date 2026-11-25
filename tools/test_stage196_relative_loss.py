"""Centered-loss identities and matched CPU experiment pipeline tests."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "tools"), str(REPO / "tracking")]
import stage196_relative_loss as run


class CacheTests(unittest.TestCase):
    def test_prediction_identity_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.npz"
            meta = dict(plan="fixed", loss="relative")
            value, reused = run.cache(path, meta, lambda: dict(prediction=np.ones((2, 6))))
            self.assertFalse(reused)
            _, reused = run.cache(path, meta, lambda: self.fail("Should reuse"))
            self.assertTrue(reused)
            with self.assertRaises(ValueError):
                run.cache(path, dict(meta, loss="absolute"), lambda: {})
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                run.cache(path, meta, lambda: {})


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU PyTorch required")
class TorchTests(unittest.TestCase):
    def test_projection_identity_and_gradient(self):
        import torch
        p = torch.tensor([[.2, .3, .4, .5, .6, .7], [.7, .6, .5, .4, .3, .2]], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([[.1, .2, .3, .8, .5, .6], [.2, .3, .4, .9, .6, .7]], dtype=torch.float64)
        terms = run.components(torch, p, y)
        torch.testing.assert_close(terms["absolute"], terms["common"] + terms["relative"])
        torch.testing.assert_close(terms["relative"], run.components(torch, p + .05, y)["relative"])
        gradient = torch.autograd.grad(terms["relative"], p)[0]
        torch.testing.assert_close(gradient.sum(1), torch.zeros(2, dtype=torch.float64), atol=1e-14, rtol=0)
        np.testing.assert_array_equal(p.detach().argmax(1), (p - p.mean(1, keepdim=True)).detach().argmax(1))

    def test_prepooled_function_and_inactive_modules(self):
        import torch
        torch.set_num_threads(1)
        model = run.architecture.build_modules(torch, 4, 2, run.metrics.ARM)
        with torch.no_grad():
            model.router.weight.fill_(.03)
        tokens = torch.randn(3, 6, 4)
        torch.testing.assert_close(model.route(tokens), model.router(run.pool(torch, tokens, 2)).sigmoid())
        self.assertTrue(all(not p.requires_grad for name in ("adapter", "quality") for p in getattr(model, name).parameters()))

    def test_matched_fit_and_checkpoint_resume(self):
        import torch
        torch.set_num_threads(1)
        groups = [f"uav{i:07d}" for i in range(38)]
        fold = dict(fold=0, fit=groups[:19], held=groups[19:])
        rng = np.random.default_rng(196)
        bank = {g: dict(tokens=rng.normal(size=(2, 6, 4)).astype(np.float32),
                        pooled=rng.normal(size=(2, 16)).astype(np.float32),
                        labels=np.tile([.7, .4, .6, .3, .5, .2], (2, 1)), frame_id=np.array([1, 2])) for g in groups}
        class Progress:
            done, reused = 0, 0
            def advance(self, count=1): self.done += count
        with tempfile.TemporaryDirectory() as tmp:
            progress, artifacts, scopes = Progress(), {}, []
            for loss in run.LOSSES:
                model, scope, history, _ = run.fit(torch, Path(tmp), dict(fingerprint="test"), bank, fold, 2132, loss, 2, progress, artifacts)
                self.assertEqual(len(history), 256)
                self.assertEqual(scope["examples"], 1024)
                self.assertTrue(scope["disabled_AQ_unchanged"])
                self.assertTrue(all(not p.requires_grad for p in model.parameters()))
                scopes.append(scope)
            self.assertEqual(scopes[0]["initial_sha256"], scopes[1]["initial_sha256"])
            self.assertEqual(scopes[0]["draws_sha256"], scopes[1]["draws_sha256"])
            self.assertEqual(scopes[0]["trainable_parameters"], scopes[1]["trainable_parameters"])
            _, scope, _, _ = run.fit(torch, Path(tmp), dict(fingerprint="test"), bank, fold, 2132, "relative", 2, progress, artifacts)
            self.assertEqual(progress.reused, 256)
            self.assertEqual(scope["final_sha256"], scopes[1]["final_sha256"])

    def test_complete_synthetic_pipeline_without_images_or_gpu(self):
        import contextlib
        import io
        import os
        import torch
        from unittest.mock import patch

        torch.set_num_threads(1)
        groups = [f"uav{i:07d}" for i in range(38)]
        folds = [dict(fold=0, fit=groups[:19], held=groups[19:]), dict(fold=1, fit=groups[19:], held=groups[:19])]
        entries = [dict(sequence=g + "_00000_s", group=g, sample_frames=[1, 2]) for g in groups]
        plan = dict(fingerprint=run.audit.SOURCE_PLAN, entries=entries, folds=folds, runtime=dict(tracker=dict(lens_z=2)))
        with tempfile.TemporaryDirectory() as tmp:
            r194, r195, out = (Path(tmp) / s for s in ("source194", "source195", "new"))
            r194.mkdir(); r195.mkdir()
            run.publish_identical(r194 / "stage194_decision.json", dict(fixture=True))
            run.publish_identical(r195 / "stage195_decision.json", dict(fixture=True))
            artifacts, old_frames, scopes = {}, [], []
            rng = np.random.default_rng(196)
            for entry in entries:
                seq, group = entry["sequence"], entry["group"]
                for relative, data, meta in (
                    (f"features/{seq}.npz", dict(frame_id=np.array([1, 2]), tokens=rng.normal(size=(2, 6, 4)).astype(np.float32)),
                     dict(plan_fingerprint=plan["fingerprint"], sequence=seq, kind="native_parent_prefix")),
                    (f"reference/{group}.npz", dict(iou=np.tile([.7, .4, .6, .3, .5, .2], (2, 1))),
                     dict(plan_fingerprint=plan["fingerprint"], kind="fixed_layer_fit_labels", group=group))):
                    path = r194 / relative
                    run.source.cache(path, meta, lambda data=data: data)
                    run.source.record(r194, path, artifacts)
            for fold in folds:
                for seed in run.original.DESIGN["seeds"]:
                    scopes.append(dict(fold=str(fold["fold"]), seed=str(seed), arm=run.metrics.ARM,
                        draws_sha256=run.fingerprint(run.original.draws_for(fold, seed, {g: 2 for g in groups}))))
                    for entry in entries:
                        for frame in (1, 2):
                            old_frames.append(dict(fold=str(fold["fold"]), seed=str(seed), group=entry["group"], frame=str(frame),
                                cpu_layer="7", **{f"prediction_L{k}": str(.6 if k == 7 else .5) for k in range(7, 13)}))
            decision = dict(fingerprint=run.audit.SOURCE194, artifact_sha256=artifacts, table_sha256={})
            reader = run.audit.VerifiedSource(r194, decision)
            inputs = (plan, decision, reader, dict(scope_summary=scopes), dict(fingerprint=run.SOURCE195), old_frames)
            with patch.object(run, "load_inputs", return_value=inputs), \
                 patch.object(sys, "argv", ["stage196", "--stage194-root", str(r194), "--stage195-root", str(r195), "--output-dir", str(out)]), \
                 patch.dict(os.environ, {}), \
                 patch.object(torch.cuda, "is_available", return_value=False), \
                 patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("No CUDA initialization allowed")), \
                 contextlib.redirect_stdout(io.StringIO()):
                run.main()
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
            result = run.source.read_sealed(out / "stage196_decision.json")
            self.assertEqual(result["training_runs"], 8)
            self.assertEqual(result["training_updates"], 2048)
            self.assertEqual(result["predicted_states"], 608)
            self.assertEqual(result["new_image_reads"], 0)
            self.assertEqual(result["new_encoder_calls"], 0)
            self.assertLessEqual(result["original_routing_function_maximum_error"], 1e-7)
            with contextlib.redirect_stdout(io.StringIO()):
                run.verify_results(out)


if __name__ == "__main__":
    unittest.main()
