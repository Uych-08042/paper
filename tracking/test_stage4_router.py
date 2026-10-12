"""CPU-only tests for Stage-4C utility-router training helpers."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from stage4_collect import make_run, publish_run
from stage4_common import cache_path, fingerprint, tracking_iou, write_cache
from stage4_manifest import build_manifest
from stage4_router_train import (
    balanced_group_weights,
    build_low_features,
    calibration_split,
    calibration_threshold,
    estimated_latency,
    fit_combined_preprocessor,
    fit_ridge,
    fit_token_pca,
    main,
    predict_ridge,
    route_predictions,
    save_model,
    transform_combined,
    transform_token_pca,
)


def fake_features(count=12):
    rng = np.random.RandomState(17)
    previous_state = np.column_stack((
        rng.uniform(0, 200, (count, 2)), rng.uniform(10, 80, (count, 2))))
    previous_fast = previous_state + rng.normal(0, 2, (count, 4))
    previous_fast[:, 2:] = np.maximum(previous_fast[:, 2:], 2)
    fast = previous_state + rng.normal(0, 3, (count, 4))
    fast[:, 2:] = np.maximum(fast[:, 2:], 2)
    return {
        "previous_state": previous_state,
        "previous_fast_box": previous_fast,
        "previous_action": np.asarray([-1] + [index % 7 for index in range(count - 1)]),
        "fast_box": fast,
        "image_hw": np.tile([360, 640], (count, 1)),
        "resize_factor": rng.uniform(0.5, 2.0, count),
        "selector_prob": rng.uniform(0.05, 0.95, (count, 6)),
        "fast_score_map": rng.normal(size=(count, 1, 4, 4)),
    }


class FeatureBoundaryTests(unittest.TestCase):
    def test_variants_use_only_their_declared_current_frame_inputs(self):
        original = fake_features()
        changed_selector = {key: value.copy() for key, value in original.items()}
        changed_selector["selector_prob"] += 0.02
        changed_fast = {key: value.copy() for key, value in original.items()}
        changed_fast["fast_box"][:, 0] += 10
        changed_fast["fast_score_map"] *= 2

        token, token_names = build_low_features(original, "token_only")
        selector, selector_names = build_low_features(original, "token_selector")
        fast, fast_names = build_low_features(original, "fast_aware")
        np.testing.assert_array_equal(
            token, build_low_features(changed_selector, "token_only")[0])
        np.testing.assert_array_equal(
            token, build_low_features(changed_fast, "token_only")[0])
        self.assertFalse(np.array_equal(
            selector, build_low_features(changed_selector, "token_selector")[0]))
        np.testing.assert_array_equal(
            selector, build_low_features(changed_fast, "token_selector")[0])
        self.assertFalse(np.array_equal(
            fast, build_low_features(changed_fast, "fast_aware")[0]))
        np.testing.assert_array_equal(
            fast, build_low_features(changed_selector, "fast_aware")[0])
        self.assertEqual(token.shape[1], len(token_names))
        self.assertEqual(selector.shape[1], len(selector_names))
        self.assertEqual(fast.shape[1], len(fast_names))

    def test_video_calibration_split_keeps_both_trajectories_together(self):
        sequences = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
        fit, calibration, names = calibration_split(sequences, 0.25, 2026)
        self.assertFalse(set(sequences[fit]).intersection(set(sequences[calibration])))
        self.assertEqual(set(sequences[calibration]), set(names))
        self.assertTrue(fit.any())
        self.assertTrue(calibration.any())


class ModelTests(unittest.TestCase):
    def test_pca_ridge_is_deterministic_and_export_is_pickle_free(self):
        rng = np.random.RandomState(5)
        tokens = rng.normal(size=(120, 24)).astype(np.float32)
        low = rng.normal(size=(120, 5)).astype(np.float32)
        targets = np.column_stack([
            -0.1 * low[:, index % 5] for index in range(6)
        ])
        sequences = np.asarray(["s{}".format(index // 10) for index in range(120)])
        trajectories = np.asarray(["fast" if index % 2 else "sgla" for index in range(120)])
        weights = balanced_group_weights(sequences, trajectories)
        first = fit_token_pca(tokens, 10, 3, 1, 7)
        second = fit_token_pca(tokens, 10, 3, 1, 7)
        np.testing.assert_allclose(first["components"], second["components"])
        projection = transform_token_pca(tokens, first, 17)
        combined = fit_combined_preprocessor(projection, low, weights)
        features = transform_combined(projection, low, combined)
        ridge = fit_ridge(features, targets, weights, 0.001)
        predictions = predict_ridge(features, ridge)
        self.assertEqual(predictions.shape, targets.shape)
        self.assertLess(float(np.mean((predictions - targets) ** 2)), 0.0001)

        manifest = {"fingerprint": "manifest"}
        run = {"fingerprint": "cache", "manifest": manifest}
        configuration = {"ridge_alpha": 0.001, "calibration_sequences": ["s0"]}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.npz"
            metadata = save_model(path, "token_only", run, first, combined,
                                  ridge, tuple("f{}".format(i) for i in range(5)),
                                  configuration)
            self.assertEqual(metadata["fingerprint"], fingerprint({
                key: value for key, value in metadata.items() if key != "fingerprint"
            }))
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(archive["ridge_coefficient"].shape[1], 6)
                self.assertEqual(archive["pca_components"].shape, (10, 24))

    def test_budget_is_a_cap_and_never_forces_negative_scores(self):
        scores = np.asarray([1.0, 0.9, 0.8, 0.7, 0.6, -0.1, -0.2, -0.3, -0.4, -0.5])
        weights = np.ones(len(scores))
        predictions = np.tile(scores[:, None], (1, 6))
        threshold = calibration_threshold(scores, weights, 30)
        actions, _, _ = route_predictions(predictions, threshold)
        self.assertEqual(int((actions > 0).sum()), 3)
        threshold = calibration_threshold(scores, weights, 100)
        actions, _, _ = route_predictions(predictions, threshold)
        self.assertEqual(int((actions > 0).sum()), 5)

    def test_latency_model_distinguishes_pre_and_post_fast_routing(self):
        costs = {
            "fast": 4.0, "sgla": 4.4, "fast_then_sgla": 5.8,
            "selector": 0.13, "pool_all_channels_grid4": 0.12,
            "block_L7": 0.37, "block_L8": 0.31, "block_L9": 0.29,
            "block_L10": 0.29, "block_L11": 0.28, "block_L12": 0.28,
        }
        actions = np.asarray([0, 1, 3, 0])
        token = estimated_latency("token_only", actions, costs)
        selector = estimated_latency("token_selector", actions, costs)
        fast_aware = estimated_latency("fast_aware", actions, costs)
        self.assertGreater(selector, token)
        self.assertGreater(fast_aware, selector)


class PipelineTests(unittest.TestCase):
    def test_small_audited_cache_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            names = ["GOT-10k_Train_{:06d}".format(index) for index in range(1, 6)]
            for name in names:
                directory = dataset / name
                directory.mkdir(parents=True)
                (directory / "groundtruth.txt").write_text(
                    "10,10,20,20\n" * 4, encoding="utf-8")
                (directory / "absence.label").write_text("0\n" * 4, encoding="utf-8")
                (directory / "cover.label").write_text("8\n" * 4, encoding="utf-8")
                for frame in range(1, 5):
                    (directory / "{:08d}.jpg".format(frame)).write_bytes(b"image")
            manifest = build_manifest("got10k_train", dataset, names, 3, 2, 4, 19)
            runtime = {"lens_z": 2, "lens_x": 4, "embed_dim": 2}
            run = make_run(manifest, runtime, 2, False, ["fast", "sgla"], False)
            cache = root / "cache"
            publish_run(cache, run)

            for sequence_index, entry in enumerate(manifest["sequences"]):
                for trajectory in run["trajectories"]:
                    count = entry["stop"] - entry["start"] - 1
                    initial = np.asarray([10., 10., 20., 20.])
                    state = initial.copy()
                    previous_fast = initial.copy()
                    executed = 0 if trajectory == "fast" else 3
                    rows = []
                    rng = np.random.RandomState(sequence_index * 10 + executed)
                    for offset in range(count):
                        boxes = np.stack([
                            state + np.asarray([0.15 * (action + 1), 0., 0., 0.])
                            for action in range(7)
                        ])
                        gt = np.asarray([10. + 0.2 * offset, 10., 20., 20.])
                        iou = tracking_iou(boxes, gt, True)
                        probabilities = np.asarray([.1, .2, .9, .4, .3, .2])
                        rows.append({
                            "frame_id": np.int64(entry["start"] + offset + 1),
                            "token_features": rng.normal(size=12).astype(np.float16),
                            "fast_score_map": rng.normal(size=(1, 2, 2)).astype(np.float16),
                            "selector_prob": probabilities.astype(np.float32),
                            "previous_state": state.copy(),
                            "previous_fast_box": previous_fast.copy(),
                            "previous_action": np.int64(-1 if offset == 0 else executed),
                            "fast_box": boxes[0].copy(),
                            "selected_layer": np.int64(9),
                            "action_boxes": boxes,
                            "gt": gt,
                            "valid_gt": np.bool_(True),
                            "iou": iou,
                            "delta": iou[1:] - iou[0],
                            "executed_action": np.int64(executed),
                            "image_hw": np.asarray([64, 64], dtype=np.int64),
                            "resize_factor": np.float64(1.0),
                        })
                        previous_fast = boxes[0].copy()
                        state = boxes[executed].copy()
                    arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
                    write_cache(cache_path(cache, entry, trajectory), arrays, {
                        "run_fingerprint": run["fingerprint"], "entry": entry,
                        "trajectory": trajectory, "init_box": initial.tolist(),
                    })

            output = root / "router"
            arguments = [
                "stage4_router_train.py", "--input", str(cache),
                "--output_dir", str(output), "--variants", "token_only",
                "--pca_dims", "2", "--pca_oversample", "1",
                "--pca_power_iterations", "0", "--ridge_alphas", "0.01",
                "--calibration_fraction", "0.34", "--budgets", "0", "50", "100",
                "--seed", "3",
            ]
            with mock.patch.object(sys, "argv", arguments):
                main()
            with (output / "stage4_router_summary.csv").open(
                    newline="", encoding="utf-8") as handle:
                summary = list(csv.DictReader(handle))
            with (output / "stage4_router_budget.csv").open(
                    newline="", encoding="utf-8") as handle:
                budgets = list(csv.DictReader(handle))
            self.assertEqual(len(summary), 1)
            self.assertEqual(len(budgets), 3)
            self.assertTrue((output / "stage4_router_token_only.npz").is_file())
            self.assertIn("sgla_selector_current_state_mean_iou", summary[0])


if __name__ == "__main__":
    unittest.main()
