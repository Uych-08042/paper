"""Matched initialization, fit moments, objective identities and complete CPU pilot."""
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
import stage199_residual_routing as run


class Progress:
    def __init__(self): self.done, self.reused = 0, 0
    def advance(self, count=1): self.done += count


class SourceTests(unittest.TestCase):
    def test_source198_snapshot_and_cache_tampering(self):
        d, tables = run.load198(run.DEFAULT_BUNDLE)
        self.assertEqual(d["fingerprint"], run.SOURCE198)
        self.assertEqual(len(tables["frame_predictions"]), 21888)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.npz"
            run.cache(path, dict(kind="test"), lambda: dict(value=np.ones(3)))
            _, reused = run.cache(path, dict(kind="test"), lambda: self.fail("Must reuse"))
            self.assertTrue(reused)
            with self.assertRaises(ValueError): run.cache(path, dict(kind="changed"), lambda: {})
            path.write_bytes(b"tampered")
            with self.assertRaises(ValueError): run.cache(path, dict(kind="test"), lambda: {})
            bad = Path(tmp) / "source.txt"; bad.write_bytes(b"not the reviewed source")
            with self.assertRaises(ValueError): run.load198(bad)


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU Torch required")
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def make_bank(self):
        groups = [f"uav{i:07d}" for i in range(38)]
        rng = np.random.default_rng(199)
        bank = {}
        for g in groups:
            tokens = rng.normal(size=(2, 6, 4)).astype(np.float32)
            bank[g] = dict(tokens=tokens,
                pooled=run.architecture.pooling.pool(self.torch, self.torch.as_tensor(tokens), 2).numpy(),
                labels=rng.uniform(.1, .9, size=(2, 6)), frame_id=np.array([1, 2]), sequence=g + "_test")
        return groups, bank

    def test_equal_initial_state_decision_and_parameter_budget(self):
        torch = self.torch
        stats = dict(prior=np.array([.69, .68, .67, .66, .65, .64]),
                     mean_tokens=np.zeros((320, 192)), mean_pooled=np.zeros(768))
        models = [run.architecture.build(torch, stats, 64, arm) for arm in run.ARMS]
        self.assertEqual(len({run.source.model_digest(m) for m in models}), 1)
        self.assertEqual({sum(p.numel() for p in m.parameters()) for m in models}, {6720})
        x = torch.ones((1, 320, 192))
        for model in models:
            scores = model(x).detach().numpy()
            self.assertLess(np.abs(scores[0] - stats["prior"]).max(), 1e-7)
            self.assertEqual(int(run.metrics.choose(scores)[0]), 0)
            self.assertTrue(all(not b.requires_grad for b in model.buffers()))

    def test_fit_centering_for_arbitrary_weights_and_no_held_statistics(self):
        torch = self.torch
        groups, bank = self.make_bank()
        fit = groups[:19]
        stats = run.architecture.fit_statistics(bank, fit)
        original = {k: v.copy() for k, v in stats.items()}
        bank[groups[-1]]["tokens"].fill(123)
        bank[groups[-1]]["labels"].fill(0)
        for k, value in run.architecture.fit_statistics(bank, fit).items():
            np.testing.assert_array_equal(value, original[k])
        model = run.architecture.build(torch, stats, 2, "interaction_residual")
        with torch.no_grad():
            model.projection.weight.copy_(torch.tensor([[.2, -.1, .4, .3]]))
            model.readout.weight.copy_(torch.arange(132).reshape(6, 22) / 100)
            means = [model.raw(torch.as_tensor(bank[g]["tokens"]), torch.as_tensor(bank[g]["pooled"])).mean(0) for g in fit]
        torch.testing.assert_close(torch.stack(means).mean(0), torch.zeros(6), atol=2e-6, rtol=0)
        bad_stats = run.architecture.pooling.pool(torch, torch.as_tensor(stats["mean_tokens"][None], dtype=torch.float32), 2).numpy()[0]
        self.assertGreater(float(np.abs(bad_stats - stats["mean_pooled"]).max()), .1)

    def test_loss_decomposition_projection_and_same_parameter_choices(self):
        torch = self.torch
        groups, bank = self.make_bank()
        stats = run.architecture.fit_statistics(bank, groups[:19])
        a = run.architecture.build(torch, stats, 2, "absolute_additive").double()
        b = run.architecture.build(torch, stats, 2, "interaction_residual").double()
        with torch.no_grad():
            a.readout.weight.copy_(torch.arange(132).reshape(6, 22) / 200)
            b.load_state_dict(a.state_dict())
        x = torch.as_tensor(bank[groups[0]]["tokens"], dtype=torch.float64)
        g = torch.as_tensor(bank[groups[0]]["pooled"], dtype=torch.float64)
        y = torch.as_tensor(bank[groups[0]]["labels"], dtype=torch.float64)
        loss, terms = b.objective(x, g, y)
        torch.testing.assert_close(terms["absolute_mse"], terms["common_mse"] + terms["interaction_mse"], atol=1e-12, rtol=0)
        target = run.architecture.center(y - b.prior)
        expected = (run.architecture.center(b.raw(x, g)) - target).square().mean()
        torch.testing.assert_close(loss, expected, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(run.metrics.choose(a(x, g).detach().numpy()), run.metrics.choose(b(x, g).detach().numpy()))
        loss.backward()
        torch.testing.assert_close(b.readout.weight.grad.sum(0), torch.zeros(22, dtype=torch.float64), atol=1e-12, rtol=0)

    def test_matched_training_buffers_and_resume(self):
        groups, bank = self.make_bank()
        fold = run.protocol.split_groups(groups)[0]
        stats = run.architecture.fit_statistics(bank, fold["fit"])
        with tempfile.TemporaryDirectory() as tmp:
            progress, artifacts, scopes = Progress(), {}, []
            for arm in run.ARMS:
                model, scope, history = run.fit(self.torch, Path(tmp), dict(fingerprint="test"), bank, stats,
                                                fold, 2132, arm, 2, progress, artifacts)
                self.assertEqual(len(history), 256)
                self.assertTrue(scope["fixed_fit_buffers_unchanged"])
                self.assertGreater(scope["maximum_projection_gradient"], 0)
                self.assertTrue(all(not p.requires_grad for p in model.parameters()))
                scopes.append(scope)
            for key in ["initial_sha256", "draws_sha256", "trainable_parameters", "initial_choice"]:
                self.assertEqual(len({s[key] for s in scopes}), 1)
            _, scope, _ = run.fit(self.torch, Path(tmp), dict(fingerprint="test"), bank, stats,
                                 fold, 2132, run.ARMS[-1], 2, progress, artifacts)
            self.assertEqual(progress.reused, 256)
            self.assertEqual(scope["final_sha256"], scopes[-1]["final_sha256"])
            path = Path(tmp) / f"training/fold{fold['fold']}/seed2132/{run.ARMS[-1]}.pt"
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                run.fit(self.torch, Path(tmp), dict(fingerprint="test"), bank, stats, fold, 2132,
                        run.ARMS[-1], 2, progress, artifacts)

    def test_full_synthetic_38_group_cpu_pipeline_and_result_reuse(self):
        torch = self.torch
        groups, bank = self.make_bank()
        folds = run.protocol.split_groups(groups)
        entries = [dict(group=g, sequence=bank[g]["sequence"], sample_frames=[1, 2]) for g in groups]
        p194 = dict(fingerprint=run.inputs.SOURCE_PLAN, entries=entries, folds=folds, runtime=dict(tracker=dict(lens_z=2)))
        with tempfile.TemporaryDirectory() as tmp:
            r194, out, bundle = Path(tmp) / "194", Path(tmp) / "output", Path(tmp) / "source/198.txt"
            r194.mkdir(); bundle.parent.mkdir(); bundle.write_bytes(b"fixture")
            artifacts, scopes = {}, []
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
            old_calibration = []
            for fold in folds:
                stats = run.architecture.fit_statistics(bank, fold["fit"])
                old_calibration.extend(dict(fold=str(fold["fold"]), layer=str(k + 7), fit_label_mean=str(stats["prior"][k])) for k in range(6))
                for seed in run.protocol.DESIGN["seeds"]:
                    scopes.append(dict(fold=str(fold["fold"]), seed=str(seed), arm=run.metrics.ARM,
                        draws_sha256=run.fingerprint(run.protocol.draws_for(fold, seed, {g: 2 for g in groups}))))
            old_frames = [dict(group=g, frame=str(i + 1), **{f"label_L{k}": str(bank[g]["labels"][i, k - 7]) for k in range(7, 13)})
                          for g in groups for i in range(2)]
            d194 = dict(fingerprint=run.inputs.SOURCE194, table_sha256={}, artifact_sha256=artifacts)
            reader = run.inputs.VerifiedSource(r194, d194)
            fixtures = (p194, d194, reader, dict(scope_summary=scopes), dict(fingerprint=run.SOURCE198),
                        dict(frame_predictions=old_frames, calibration=old_calibration))
            stdout = io.StringIO()
            with patch.object(run, "load_sources", return_value=fixtures), \
                 patch.object(sys, "argv", ["stage199", "--stage194-root", str(r194), "--stage198-bundle", str(bundle), "--output-dir", str(out)]), \
                 patch.dict(os.environ, {}), \
                 patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("No CUDA allowed")), \
                 contextlib.redirect_stdout(stdout):
                run.main()
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
                run.verify_results(out)
                with patch.object(run, "load_sources", side_effect=AssertionError("Must reuse")):
                    run.main()
            d = run.source.read_sealed(out / "stage199_decision.json")
            self.assertEqual(d["training_runs"], 12)
            self.assertEqual(d["training_updates"], 3072)
            self.assertEqual(d["predicted_states"], 912)
            self.assertEqual(len(d["primary_contrasts"]), 3)
            self.assertEqual(d["new_encoder_calls"], 0)
            self.assertIn("100.00%", stdout.getvalue())
            for name, sha in artifacts.items(): self.assertEqual(run.file_hash(r194 / name), sha)
            (out / "stage199_policy_summary.csv").write_text("tampered", encoding="utf-8")
            with self.assertRaises(ValueError): run.verify_results(out)


if __name__ == "__main__":
    unittest.main()
