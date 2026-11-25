"""Stage-195 mathematical identities, leakage controls and frozen model loading."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "tools"), str(REPO / "tracking")]
import stage195_router_objective_audit as run
import stage195_routing_common as c


class AuditTests(unittest.TestCase):
    def test_error_decomposition_including_between_video_variance(self):
        rng = np.random.default_rng(195)
        y, p = rng.random((48, 6)), rng.random((48, 6))
        value = c.errors(y, p)
        self.assertLess(value["decomposition_error"], 1e-14)
        self.assertAlmostEqual(value["target_total_variance"], value["target_frame_variance"] + value["target_contrast_energy"])
        # A predictor can fit common difficulty exactly while reversing ranking.
        mean = np.linspace(.1, .9, 48)[:, None]
        centered = np.linspace(-.01, .01, 6)[None]
        value = c.errors(mean + centered, mean - centered)
        self.assertAlmostEqual(value["common_mse"], 0.)
        self.assertAlmostEqual(value["pair_accuracy"], 0.)
        self.assertAlmostEqual(value["relative_skill_vs_zero"], -3.)

    def test_ties_and_gap_weighted_ranking(self):
        y = np.tile(np.arange(6) / 5, (3, 1))
        ties = c.ranking(y, np.full_like(y, .5))
        self.assertEqual(ties["pair_accuracy"], .5)
        self.assertEqual(ties["prediction_tied_pairs"], 45)
        self.assertAlmostEqual(c.ranking(y, y)["gap_weighted_pair_accuracy"], 1.)
        self.assertEqual(c.choose(np.ones((2, 6))).tolist(), [2, 2])
        no_difference = c.ranking(np.zeros((3, 6)), np.ones((3, 6)))
        self.assertIsNone(no_difference["pair_accuracy"])

    def test_fit_controls_do_not_read_held_labels(self):
        videos = {"fit": dict(labels=np.full((3, 6), .4), prediction=np.full((3, 6), .3)),
                  "held": dict(labels=np.zeros((3, 6)), prediction=np.zeros((3, 6)))}
        before = c.fit_controls(videos, ["fit"])
        videos["held"]["labels"][:] = 1
        videos["held"]["prediction"][:] = 1
        after = c.fit_controls(videos, ["fit"])
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
        draws = [[("fit", 0), ("fit", 0), ("fit", 1)]]
        videos["fit"]["labels"][0, 0] = 1
        weighted = c.fit_controls(videos, ["fit"], draws)
        self.assertNotEqual(weighted["fit_draw_label_mean"][0], weighted["fit_label_mean"][0])
        with self.assertRaises(ValueError):
            c.fit_controls(videos, ["fit"], [[("held", 0)]])

    def test_policy_regret_and_uniform_expectation(self):
        y = np.array([[.2, .3, .6, .5, .4, .1], [.1, .2, .4, .3, .8, .7]])
        oracle = c.policy_metrics(y, c.choose(y))
        self.assertEqual(oracle["mean_regret"], 0.)
        self.assertAlmostEqual(c.policy_metrics(y)["mean_iou"], y.mean())
        self.assertAlmostEqual(c.policy_metrics(y)["good_fraction"], (y >= .5).mean())
        self.assertAlmostEqual(c.policy_metrics(y, np.full(2, 2))["gain_vs_L9"], 0.)

    def test_full_path_identity_requires_available_control(self):
        data = dict(layer=np.full(3, 7), frame_id=np.arange(1, 4), parent=np.ones((3, 4)), boxes=np.ones((3, 4)))
        self.assertTrue(c.path_comparison(data, {7: data.copy()})["checked_equal"])
        missing = c.path_comparison(data, {9: data})
        self.assertFalse(missing["fixed_control_available"])
        self.assertFalse(missing["checked_equal"])
        changed = dict(data, boxes=np.zeros((3, 4)))
        with self.assertRaises(AssertionError):
            c.path_comparison(data, {7: changed})

    def test_source_cache_hash_and_metadata_are_both_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "data.npz"
            meta = dict(plan_fingerprint="test", group="fit")
            run.source.cache(path, meta, lambda: dict(labels=np.ones((2, 6))))
            decision = dict(table_sha256={}, artifact_sha256={p.name: run.file_hash(p) for p in (path, path.with_suffix(".receipt.json"))})
            reader = run.VerifiedSource(root, decision)
            self.assertEqual(reader.arrays("data.npz", meta)["labels"].shape, (2, 6))
            with self.assertRaises(ValueError):
                reader.arrays("data.npz", dict(meta, group="held"))
            with self.assertRaises(ValueError):
                reader.file("../outside.npz")
            path.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                run.VerifiedSource(root, decision).arrays("data.npz", meta)

    def test_pool_summary_keeps_between_source_variance(self):
        videos = {"a": dict(labels=np.full((2, 6), .1), prediction=np.full((2, 6), .1)),
                  "b": dict(labels=np.full((2, 6), .9), prediction=np.full((2, 6), .9))}
        fold = dict(fold=0, fit=["a", "b"], held=["a", "b"])
        controls = c.fit_controls(videos, fold["fit"])
        rows = run.pool_error_summary(videos, controls, fold, 2132)
        learned = next(r for r in rows if r["pool"] == "fit" and r["predictor"] == "trained")
        self.assertEqual(learned["states"], 4)
        self.assertAlmostEqual(learned["target_frame_variance"], .16)
        self.assertAlmostEqual(learned["mse"], 0.)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch CPU package required for model tests")
class TorchAuditTests(unittest.TestCase):
    def test_end_to_end_cpu_audit_with_synthetic_source(self):
        import contextlib
        import io
        import torch
        from unittest.mock import patch

        groups = [f"uav{i:07d}" for i in range(38)]
        folds = [dict(fold=0, fit=groups[:19], held=groups[19:]),
                 dict(fold=1, fit=groups[19:], held=groups[:19])]
        entries = [dict(sequence=g + "_00000_s", group=g, sample_frames=[1, 2], tracked_frames=2) for g in groups]
        plan = dict(fingerprint=run.SOURCE_PLAN, entries=entries, folds=folds, runtime=dict(tracker=dict(lens_z=2)))
        label = np.tile([.7, .5, .6, .5, .5, .5], (2, 1))
        tables = dict(scope_summary=[], seed_video_summary=[], fixed_layers=[dict(fold=i, layer=7,
            **{f"L{k}": str(float(label[:, k - 7].mean())) for k in range(7, 13)}) for i in (0, 1)])
        with tempfile.TemporaryDirectory() as tmp:
            old, out = Path(tmp) / "source", Path(tmp) / "audit"
            old.mkdir()
            run.publish_identical(old / "manifest.json", plan)
            run.publish_identical(old / "stage194_decision.json", dict(test_fixture=True))
            artifacts = {}
            feature_hash = {}

            def arrays(relative, data, metadata):
                path = old / relative
                run.source.cache(path, metadata, lambda: data)
                run.source.record(old, path, artifacts)

            for spec in entries:
                seq, group = spec["sequence"], spec["group"]
                key = f"features/{seq}.npz"
                arrays(key, dict(frame_id=np.array([1, 2]), tokens=np.zeros((2, 6, 4), np.float32)),
                       dict(plan_fingerprint=run.SOURCE_PLAN, sequence=seq, kind="native_parent_prefix"))
                feature_hash[group] = artifacts[key]
                arrays(f"reference/{group}.npz", dict(iou=label.copy()),
                       dict(plan_fingerprint=run.SOURCE_PLAN, kind="fixed_layer_fit_labels", group=group))
            for fold in folds:
                for seed in (2132, 2133):
                    model = run.model_code.build_modules(torch, 4, 2, c.ARM)
                    with torch.no_grad():
                        model.router.bias[0] = .4
                    digest = run.source.model_digest(model)
                    scope = dict(fold=fold["fold"], seed=seed, arm=c.ARM, state_sha256=digest,
                                 trainable_parameters=102, frozen_native_unchanged=True)
                    meta = dict(plan_fingerprint=run.SOURCE_PLAN, arm=c.ARM, fold=fold["fold"], seed=seed,
                                fit=fold["fit"], held=fold["held"], updates=256,
                                draws_sha256=run.fingerprint(run.previous.draws_for(fold, seed, {g: 2 for g in groups})),
                                features={g: feature_hash[g] for g in fold["fit"]})
                    path = old / f"training/fold{fold['fold']}/seed{seed}/{c.ARM}.pt"
                    history = [dict(step=i, total=.1, router=.1, adapter=0., quality=0.) for i in range(1, 257)]
                    run.source.save_model(torch, path, dict(state=model.state_dict(), scope=scope, history=history), meta)
                    run.source.record(old, path, artifacts)
                    tables["scope_summary"].append({k: str(v) for k, v in scope.items()})
                    for group in fold["held"]:
                        seq = group + "_00000_s"
                        readout = np.tile([.7, .7, .8, .9, 7], (2, 1))
                        arrays(f"readout/{seq}__s{seed}__{c.ARM}.npz", dict(values=readout),
                            dict(plan_fingerprint=run.SOURCE_PLAN, sequence=seq, seed=seed, arm=c.ARM, fold=fold["fold"]))
                        box = np.tile([10., 10., 10., 10.], (2, 1))
                        for arm, own_seed, sha, fixed in ((c.ARM, seed, digest, None), ("native", 0, None, None),
                                                        ("fit_best_fixed", 0, None, 7)):
                            key = f"evaluation/{seq}__s{own_seed}__{arm}.npz"
                            if key in artifacts:
                                continue
                            layer = 9 if arm == "native" else 7
                            calls = np.zeros((2, 12), int)
                            calls[:, :6] = 1
                            calls[:, layer - 1] = 1
                            trace = dict(frame_id=np.array([1, 2]), parent=box.copy(), boxes=box.copy(),
                                         layer=np.full(2, layer), block_calls=calls,
                                         head_calls=np.ones(2, int), selector_calls=np.ones(2, int), crop_calls=np.ones(2, int))
                            arrays(key, trace, dict(plan_fingerprint=run.SOURCE_PLAN, sequence=seq, fold=fold["fold"],
                                                   seed=own_seed, arm=arm, model_sha256=sha, fixed_layer=fixed))
                            tables["seed_video_summary"].append(dict(sequence=seq, arm=arm, seed=str(own_seed), auc="60.0"))
            decision = dict(artifact_sha256=artifacts, table_sha256={})
            reader = run.VerifiedSource(old, decision)
            with patch.object(run, "load_source", return_value=(plan, decision, reader, tables)), \
                 patch.object(sys, "argv", ["stage195", "--input", str(old), "--output-dir", str(out)]), \
                 patch.object(torch.cuda, "is_available", side_effect=AssertionError("No GPU access allowed")), \
                 contextlib.redirect_stdout(io.StringIO()):
                run.main()
            result = run.source.read_sealed(out / "stage195_decision.json")
            self.assertEqual(result["audited_models"], 4)
            self.assertEqual(result["predicted_states"], 304)
            self.assertEqual(result["full_path_audits"], 76)
            self.assertEqual(result["cpu_gpu_action_mismatches"], 0)
            self.assertEqual(result["training_updates"], 0)
            self.assertEqual(result["new_image_reads"], 0)
            self.assertEqual(result["new_encoder_calls"], 0)
            with contextlib.redirect_stdout(io.StringIO()):
                run.verify_results(out)

    def test_receipted_frozen_router_load_and_prediction(self):
        import torch
        groups = [f"uav{i:07d}" for i in range(38)]
        fold = dict(fold=0, fit=groups[:19], held=groups[19:])
        plan = dict(fingerprint="test-plan", entries=[dict(sequence=g + "_00000_s", group=g) for g in groups])
        model = run.model_code.build_modules(torch, 4, 2, c.ARM)
        with torch.no_grad():
            model.router.bias[0] = .4
        scope = dict(state_sha256=run.source.model_digest(model), trainable_parameters=102, frozen_native_unchanged=True)
        metadata = dict(plan_fingerprint=plan["fingerprint"], arm=c.ARM, fold=0, seed=2132, fit=fold["fit"],
                        held=fold["held"], updates=256, features={g: "fake-verified-feature" for g in fold["fit"]})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            relative = "training/fold0/seed2132/" + c.ARM + ".pt"
            path = root / relative
            run.source.save_model(torch, path, dict(state=model.state_dict(), scope=scope, history=[]), metadata)
            decision = dict(table_sha256={}, artifact_sha256={p.relative_to(root).as_posix(): run.file_hash(p)
                for p in (path, path.with_suffix(".receipt.json"))})
            reader = run.VerifiedSource(root, decision)
            reader.verified.update({"features/" + g + "_00000_s.npz": "fake-verified-feature" for g in fold["fit"]})
            loaded, _, _ = run.load_router(torch, reader, plan, fold, 2132, 4, 2, {k: str(v) for k, v in scope.items()})
            class Progress:
                done = 0
                def advance(self): self.done += 1
            progress = Progress()
            before = run.source.model_digest(loaded)
            p, choice = run.predict_video(torch, loaded, np.zeros((3, 6, 4), np.float32), progress)
            self.assertTrue((choice == 0).all())
            self.assertEqual(progress.done, 3)
            self.assertEqual(before, run.source.model_digest(loaded))
            self.assertEqual(p.shape, (3, 6))
            self.assertTrue(all(not x.requires_grad for x in loaded.parameters()))


if __name__ == "__main__":
    unittest.main()
