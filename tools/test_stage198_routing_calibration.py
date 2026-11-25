"""Fit/held leakage, exact product expectation and immutable bundle tests."""
import builtins
import contextlib
import copy
import csv
import hashlib
import io
from itertools import permutations
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "tools"), str(REPO / "tracking")]
import stage198_routing_calibration as run


def fixture(path):
    groups = [f"uav{i:07d}" for i in range(38)]
    folds = run.protocol.split_groups(groups)
    rng = np.random.default_rng(198)
    bank = {g: rng.uniform(.05, .95, (3, 6)) for g in groups}
    tables = {k: [dict(fixture="yes")] for k in run.SOURCE_TABLES}
    scopes, frames, policies = [], [], []
    for fold in folds:
        for seed in run.protocol.DESIGN["seeds"]:
            draws = run.protocol.draws_for(fold, seed, {g: 3 for g in groups})
            counts = {g: np.zeros(3, int) for g in groups}
            for batch in draws:
                for g, i in batch:
                    counts[g][i] += 1
            for arm in run.ARMS:
                pred = {g: (.3 + bank[g] * .3 + np.arange(6)[None] * .01).astype(np.float32) for g in groups}
                mu_p = np.stack([pred[g].mean(0) for g in fold["fit"]]).mean(0)
                scopes.append(dict(fold=fold["fold"], seed=seed, arm=arm,
                    fit_groups="|".join(fold["fit"]), held_groups="|".join(fold["held"]),
                    final_sha256=hashlib.sha256(f"{fold['fold']}/{seed}/{arm}".encode()).hexdigest(),
                    draws_sha256=run.fingerprint(draws)))
                for g in groups:
                    y, p = bank[g], pred[g]
                    action = run.metrics.choose(p)
                    base = dict(fold=fold["fold"], seed=seed, arm=arm, group=g, sequence=g + "_test",
                                pool="fit" if g in fold["fit"] else "held")
                    for policy, choice in (("trained", action), ("fit_prediction_mean", run.metrics.choose(np.broadcast_to(mu_p, p.shape)))):
                        policies.append(dict(base, policy=policy, **run.metrics.policy_metrics(y, choice)))
                    for i in range(3):
                        frames.append(dict(base, frame=i + 1, cpu_layer=int(action[i]) + 7,
                            training_draw_count=int(counts[g][i]) if g in fold["fit"] else None,
                            selected_iou=float(y[i, action[i]]),
                            **{f"prediction_L{k}": float(p[i, k - 7]) for k in range(7, 13)},
                            **{f"label_L{k}": float(y[i, k - 7]) for k in range(7, 13)}))
    tables.update(model_scope=scopes, frame_predictions=frames, policy_video=policies)
    data = {}
    for name, rows in tables.items():
        text = io.StringIO(newline="")
        writer = csv.DictWriter(text, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
        data[f"stage197_{name}.csv"] = text.getvalue().encode()
    d = run.sealed(dict(schema="stage197_matched_token_order_routing_v1", plan_fingerprint="fixture_plan",
        training_runs=12, training_updates=3072, source_groups=38, predicted_states=len(frames), new_image_reads=0,
        new_encoder_calls=0, new_full_trajectories=0, confirmation_groups_opened=0,
        table_sha256={name: hashlib.sha256(value).hexdigest() for name, value in data.items()}))
    body = ("\n===== stage197_decision.json =====\n" + json.dumps(d) + "\n").encode()
    for name, value in data.items():
        body += ("\n===== " + name + " =====\n").encode() + value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return dict(SOURCE197=d["fingerprint"], PLAN197="fixture_plan", BUNDLE_SHA256=hashlib.sha256(body).hexdigest())


class Tests(unittest.TestCase):
    def test_fit_means_are_group_weighted_and_held_labels_cannot_change_choices(self):
        videos = {
            "fitA": dict(labels=np.full((1, 6), .2), prediction=np.full((1, 6), .3, np.float32)),
            "fitB": dict(labels=np.full((3, 6), .8), prediction=np.full((3, 6), .7, np.float32)),
            "held": dict(labels=np.ones((2, 6)), prediction=np.tile(np.linspace(.1, .9, 6), (2, 1)).astype(np.float32)),
        }
        a, b = run.fit_means(videos, ["fitA", "fitB"])
        np.testing.assert_allclose(a, .5)
        np.testing.assert_allclose(b, .5)
        before = run.calibration_scores(videos["held"]["prediction"], a, b)
        videos["held"]["labels"].fill(0)
        after = run.calibration_scores(videos["held"]["prediction"], *run.fit_means(videos, ["fitA", "fitB"]))
        np.testing.assert_array_equal(before, after)
        with self.assertRaises(ValueError):
            run.fit_means(videos, ["fitA", "fitA"])

    def test_product_expectation_equals_exhaustive_permutations_and_covariance(self):
        y = np.zeros((3, 6)); y[np.arange(3), np.arange(3)] = 1
        a = np.arange(3)
        report = run.correspondence(y, a)
        all_scores = [y[np.arange(3), p].mean() for p in permutations(a)]
        self.assertAlmostEqual(report["independent_mean_iou"], np.mean(all_scores))
        self.assertAlmostEqual(report["state_alignment_gain"], 2 / 3)
        self.assertLess(report["covariance_identity_error"], 1e-12)
        self.assertAlmostEqual(run.correspondence(y, np.zeros(3, int))["state_alignment_gain"], 0.)
        wrong = np.array([1, 2, 0])
        self.assertLess(run.correspondence(y, wrong)["state_alignment_gain"], 0.)
        self.assertEqual(report["independent_mean_iou"], run.correspondence(y, wrong)["independent_mean_iou"])

    def test_scores_are_not_clipped_and_ties_have_explicit_control(self):
        scores = run.calibration_scores(np.array([[.99, .01, .5, .4, .3, .2]]),
                                        np.array([.9, .1, .5, .4, .3, .2]), np.array([.1, .9, .5, .4, .3, .2]))
        self.assertGreater(scores.max(), 1)
        self.assertLess(scores.min(), 0)
        self.assertEqual(int(run.metrics.choose(np.ones((1, 6)))[0]), 2)
        self.assertEqual(int(run.metrics.choose(np.array([[1, 1, 0, 0, 0, 0]]))[0]), 0)

    def test_bundle_fingerprint_checksum_and_fold_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source/bundle.txt"
            constants = fixture(path)
            with patch.multiple(run, **constants):
                d, t = run.load_bundle(path)
                run.source_models(d, t)
                modified = copy.deepcopy(t)
                modified["model_scope"][0]["fit_groups"] = modified["model_scope"][0]["held_groups"]
                with self.assertRaises(ValueError): run.source_models(d, modified)
                original = path.read_bytes()
                path.write_bytes(original + b"tampered")
                with self.assertRaises(ValueError): run.load_bundle(path)
                changed = original.replace(b"fixture\r\nyes\r\n", b"fixture\r\nno!\r\n", 1)
                self.assertNotEqual(changed, original)
                path.write_bytes(changed)
                with patch.object(run, "BUNDLE_SHA256", hashlib.sha256(changed).hexdigest()):
                    with self.assertRaisesRegex(ValueError, "checksum"):
                        run.load_bundle(path)

    def test_full_synthetic_diagnostic_reuses_and_verifies_without_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, out = Path(tmp) / "source/bundle.txt", Path(tmp) / "output"
            constants = fixture(path)
            original_import = builtins.__import__
            def forbid_torch(name, *a, **kw):
                if name == "torch" or name.startswith("torch."):
                    raise AssertionError("No Torch/model operations allowed")
                return original_import(name, *a, **kw)
            stdout = io.StringIO()
            with patch.multiple(run, **constants), \
                 patch.object(sys, "argv", ["stage198", "--source-bundle", str(path), "--output-dir", str(out)]), \
                 patch.object(builtins, "__import__", side_effect=forbid_torch), \
                 contextlib.redirect_stdout(stdout):
                before = run.file_hash(path)
                run.main()
                d = run.verify_results(out)
                self.assertEqual(d["models_audited"], 12)
                self.assertEqual(d["evaluated_states"], 1368)
                self.assertEqual(d["calibration_constants"], 72)
                self.assertEqual(len(d["primary_contrasts"]), 2)
                self.assertEqual(d["new_training_updates"], 0)
                self.assertLess(d["maximum_covariance_identity_error"], 1e-12)
                self.assertIn("100.00%", stdout.getvalue())
                self.assertEqual(run.file_hash(path), before)
                with patch.object(run, "load_bundle", side_effect=AssertionError("Finished output should be reused")):
                    run.main()
                self.assertEqual(len(list(out.glob("stage198_results_*.txt"))), 1)
                (out / "stage198_policy_summary.csv").write_text("changed", encoding="utf-8")
                with self.assertRaises(ValueError): run.verify_results(out)


if __name__ == "__main__":
    unittest.main()
