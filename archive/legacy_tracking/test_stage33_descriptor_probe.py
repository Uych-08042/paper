"""Gradient, weighting, causal-input, source-isolation and resume regressions."""

import contextlib
import copy
import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from stage4_common import file_hash, write_cache, write_json
from stage27_uav_common import seal
from stage31_identity_common import normalize
import stage31_identity_engine as engine
from stage31_precision_compat import compatibility_record, precision_replay_auditor
from stage33_descriptor_common import (ARMS, DESIGN, evaluate_scores, initial_parameters,
    pair_loss_gradient, predict, prepare_video, train_model, training_pairs)
from stage33_descriptor_probe import aggregate_metrics, fitted_model, run_probe
from test_stage31_precision_compat import Float32Observer, contract_fixture


HERE = Path(__file__).resolve().parent


def video_fixture(name, easy=4, hard=4):
    n, dim = easy + hard, 4
    labels = np.zeros((n, 2))
    labels[:easy, 0], labels[easy:, 1] = 1., 1.
    descriptors = np.zeros((n, 2, dim))
    descriptors[..., 0] = labels
    descriptors[..., 1] = 1 - labels
    scalar = np.zeros((n, 2, 9))
    scalar[..., 0] = labels
    scalar[..., 7] = [.9, .8]
    scalar[..., 8] = [.5, .4]
    return dict(sequence=name, source_group=name, frame_id=np.arange(1, n + 1),
        anchor=np.array([1., 0, 0, 0]), descriptors=descriptors, valid=np.ones((n, 2), bool),
        scalar=scalar, stress_scalar=scalar.copy(), overlap=labels,
        swapped=np.zeros(n, bool), known_negative_history=np.zeros(n, bool))


class MathTests(unittest.TestCase):
    def test_analytic_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(17)
        batch = dict(anchor=normalize(rng.normal(size=(7, 4))), positive=normalize(rng.normal(size=(7, 4))),
                     negative=normalize(rng.normal(size=(7, 4))), sp=rng.normal(size=(7, 9)), sn=rng.normal(size=(7, 9)))
        scale = np.arange(1, 10) / 3
        for arm in ARMS:
            parameters = initial_parameters(arm, 4, 19, 3)
            parameters["head"] = rng.normal(0, .2, parameters["head"].shape)
            loss, gradient = pair_loss_gradient(parameters, arm, batch, scale)
            self.assertTrue(np.isfinite(loss))
            for key, expected in gradient.items():
                numerical = np.zeros_like(expected)
                for index in np.ndindex(expected.shape):
                    original = parameters[key][index]
                    parameters[key][index] = original + 1e-6
                    plus, _ = pair_loss_gradient(parameters, arm, batch, scale)
                    parameters[key][index] = original - 1e-6
                    minus, _ = pair_loss_gradient(parameters, arm, batch, scale)
                    parameters[key][index] = original
                    numerical[index] = (plus - minus) / 2e-6
                np.testing.assert_allclose(expected, numerical, atol=2e-8, rtol=2e-5)

    def test_balancing_has_half_hard_mass_and_no_held_out_statistics(self):
        first, second, held = video_fixture("a", 100, 1), video_fixture("b", 1, 1), video_fixture("held", 1, 1)
        pairs = training_pairs([first, second])
        hard = pairs["sp"][:, 7] < pairs["sn"][:, 7]
        self.assertAlmostEqual(float(pairs["weight"][hard].sum()), .5)
        self.assertAlmostEqual(float(pairs["weight"][~hard].sum()), .5)
        held["scalar"] *= 1000
        held["descriptors"] *= 1000
        repeated = training_pairs([first, second])
        for key in ("weight", "scale", "positive", "negative"):
            np.testing.assert_array_equal(pairs[key], repeated[key])
        with self.assertRaisesRegex(ValueError, "Both easy and hard"):
            training_pairs([video_fixture("no_hard", 8, 0)])

    def test_frozen_projection_stays_frozen_and_training_is_reproducible(self):
        pairs = training_pairs([video_fixture("a"), video_fixture("b")])
        for arm in ARMS:
            first, history, audit = train_model(pairs, arm, 2100, DESIGN["updates"], 3)
            second, _, _ = train_model(pairs, arm, 2100, DESIGN["updates"], 3)
            for key in first:
                np.testing.assert_array_equal(first[key], second[key])
            self.assertTrue(audit["changed"]["head"])
            self.assertEqual(history[-1, 0], DESIGN["updates"])
            self.assertLess(history[-1, 1], history[0, 1])
            if arm == "descriptor_fixed":
                self.assertFalse(audit["changed"]["projection"])
            if arm == "descriptor_learned":
                self.assertTrue(audit["changed"]["projection"])
                self.assertGreater(audit["maximum_data_projection_gradient"], 0)

    def test_memory_swap_is_causal_label_free_and_not_a_descriptor_robustness_claim(self):
        n = 5
        data = dict(descriptors=np.tile(np.eye(2)[None], (n, 1, 1)), valid=np.ones((n, 2), bool),
            features=np.zeros((n, 2, 9)), anchor=np.array([1., 0.]), frame_id=np.arange(n),
            candidate_boxes=np.tile(np.array([[40., 20., 10, 10], [10., 20., 10, 10]])[None], (n, 1, 1)),
            gt=np.tile([40., 20., 10, 10], (n, 1)), labels=np.tile([1, 0], (n, 1)))
        entry = dict(sequence="a", source_group_proxy="a")
        video = prepare_video(entry, data)
        altered = copy.deepcopy(data)
        altered["descriptors"][-1] = np.eye(2)[::-1]
        future = prepare_video(entry, altered)
        np.testing.assert_array_equal(video["stress_scalar"][:-1], future["stress_scalar"][:-1])
        altered = copy.deepcopy(data)
        altered["gt"] += 500
        altered["labels"][:] = 1
        changed_gt = prepare_video(entry, altered)
        np.testing.assert_array_equal(video["stress_scalar"], changed_gt["stress_scalar"])
        self.assertTrue(video["known_negative_history"][1:].all())
        for arm in ARMS:
            parameters = initial_parameters(arm, 2, 2100, 3)
            parameters["head"][:] = .2
            a, b = predict(video, parameters, arm, np.ones(9)), predict(changed_gt, parameters, arm, np.ones(9))
            np.testing.assert_array_equal(a, b)
            if arm != "scalar_balanced":
                np.testing.assert_array_equal(a, predict(video, parameters, arm, np.ones(9), True))

    def test_seed_repetitions_do_not_inflate_independent_video_count(self):
        rows = []
        for seed in DESIGN["seeds"]:
            for name in ("a", "b"):
                metrics, _ = evaluate_scores(video_fixture(name), np.tile([1., 0.], (8, 1)))
                rows.extend(dict(row, sequence=name, seed=seed, arm="native_rank", condition="nominal") for row in metrics)
        videos, summary = aggregate_metrics(rows)
        hard = next(r for r in summary if r["cohort"] == "hard" and r["metric"] == "pair_accuracy")
        self.assertEqual(hard["supported_videos"], 2)
        self.assertEqual(hard["pair_frames"], 8)
        self.assertTrue(all(r["seeds"] == 2 for r in videos))


class MixedObserver(Float32Observer):
    def observe(self, image, features):
        packet = super().observe(image, features)
        if features and self.frame % 4 < 2:
            packet["boxes"] = packet["boxes"][::-1].copy()
            packet["descriptors"] = packet["descriptors"][::-1].copy()
            packet["native_box"] = packet["boxes"][0].copy()
        return packet


def frozen_fit_fixture(root):
    directory, contract, reader = contract_fixture(root)
    for entry in contract["sequences"]:
        if entry["split"] == "fit":
            entry["frames"] += [dict(entry["frames"][-1]) for _ in range(24)]
            annotation = Path(contract["root"]) / entry["annotation"]
            annotation.write_text("\n".join(["40,20,10,10"] * len(entry["frames"])))
            entry["annotation_sha256"] = file_hash(annotation)
    contract = seal({k: v for k, v in contract.items() if k != "fingerprint"})
    write_json(directory / "manifest.json", contract)
    run = seal(dict(schema=contract["schema"], role="native_fit_candidate_features", contract_fingerprint=contract["fingerprint"]))
    write_json(directory / "fit_run.json", run)
    write_json(directory / "stage31_precision_compat.json", compatibility_record(contract))
    with precision_replay_auditor():
        for entry in contract["sequences"]:
            if entry["split"] == "fit":
                data = engine.run_video(MixedObserver(contract["runtime"]), contract, entry, "sgla", fit_collect=True, image_reader=reader)
                engine.audit_video(data, contract, entry, "sgla", fit_collect=True)
                write_cache(directory / "fit_cache" / (entry["sequence"] + ".npz"), data, engine.cache_metadata(run, entry, "sgla"))
    result = subprocess.run([sys.executable, "-B", str(HERE / "stage31_precision_compat.py"), "train", "--input", str(directory)],
                            text=True, capture_output=True)
    if result.returncode:
        raise AssertionError(result.stderr)
    return directory, contract


class IntegrationTests(unittest.TestCase):
    def test_all_folds_source_isolation_resume_and_checkpoint_tamper(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            root = Path(temporary)
            source, contract = frozen_fit_fixture(root)
            before = {str(p): file_hash(p) for p in source.rglob("*") if p.is_file()}
            for entry in contract["sequences"]:
                for frame in entry["frames"]:
                    (Path(contract["root"]) / frame["file"]).unlink(missing_ok=True)
                if entry["split"] == "checking":
                    (Path(contract["root"]) / entry["annotation"]).unlink()
            original_gt = engine.ground_truth
            def no_checking(contract, entry):
                self.assertEqual(entry["split"], "fit")
                return original_gt(contract, entry)
            with mock.patch.object(engine, "ground_truth", side_effect=no_checking), contextlib.redirect_stdout(io.StringIO()):
                audit = run_probe(source, root / "audit", "audit")
                self.assertEqual(audit["status"], "READY_DESCRIPTOR_PROBE")
                smoke = run_probe(source, root / "smoke", "smoke")
                self.assertEqual(smoke["status"], "PASS_DESCRIPTOR_SMOKE")
                # A short internal fixture budget exercises all production folds;
                # users cannot override the actual fixed CLI budget.
                with mock.patch.dict(DESIGN, updates=3, projection_dim=3, batch_size=16, bootstrap_samples=64):
                    report = run_probe(source, root / "full", "full")
                    self.assertFalse(report["checking_labels_loaded"])
                    self.assertFalse(report["backbone_updated"])
                    self.assertFalse(report["backbone_inference_performed"])
                    self.assertTrue(report["probe_models_used"])
                    self.assertEqual(len(list((root / "full" / "checkpoints").glob("*.npz"))), 48)
                    for p in (root / "full" / "checkpoints").glob("*.npz"):
                        with np.load(p, allow_pickle=False) as archive:
                            meta = json.loads(str(archive["metadata_json"].item()))
                        self.assertNotIn(meta["held_out"], meta["train_sequences"])
                        self.assertEqual(len(meta["train_sequences"]), 7)
                    hashes = {str(p): file_hash(p) for p in (root / "full" / "checkpoints").rglob("*") if p.is_file()}
                    run_probe(source, root / "full", "full")
                    self.assertEqual(hashes, {str(p): file_hash(p) for p in (root / "full" / "checkpoints").rglob("*") if p.is_file()})
                    corrupt = next((root / "full" / "checkpoints").glob("*.npz"))
                    corrupt.write_bytes(corrupt.read_bytes() + b"changed bytes")
                    with self.assertRaisesRegex(ValueError, "receipt/hash mismatch"):
                        run_probe(source, root / "full", "full")
                self.assertEqual(before, {str(p): file_hash(p) for p in source.rglob("*") if p.is_file()})
            # Exercise the real default audit CLI without images/checking labels.
            cli = subprocess.run([sys.executable, "-B", str(HERE / "stage33_descriptor_probe.py"),
                "--input", str(source), "--output_dir", str(root / "cli_audit")], capture_output=True, text=True)
            self.assertEqual(cli.returncode, 0, cli.stderr)
            with self.assertRaisesRegex(ValueError, "separate Stage-33"):
                run_probe(source, source / "analysis", "audit")


if __name__ == "__main__":
    unittest.main()
