"""CPU-only tests for Stage-14 multi-action counterfactual collection."""

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from stage4_common import fingerprint, write_cache
from stage14_multiaction_collect import collect_multiaction
from stage14_multiaction_audit import (
    budget_rows,
    mechanism_rows,
    model_summary_rows,
    nested_group_oof,
    prediction_rows,
)
from stage14_multiaction_common import (
    ACTION_NAMES,
    discounted_policy_values,
    load_and_audit_cache,
    make_run,
    multiaction_cache_path,
)
from stage8_motion_gate import MOTION_FEATURE_NAMES


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeRuntime:
    def __init__(self):
        self.torch = SimpleNamespace(no_grad=nullcontext)
        self.tracker = SimpleNamespace(state=None, frame_id=0)
        self.identity = {"lens_z": 2, "lens_x": 4, "embed_dim": 2}
        self.candidate_calls = 0

    def read(self, path):
        return np.full((32, 32, 3), int(Path(path).stem), dtype=np.uint8)

    def initialize(self, image, box):
        self.tracker.state = list(box)
        self.tracker.frame_id = 0

    def prepare(self, image):
        return float(image[0, 0, 0]) + self.tracker.state[0], 1.0

    def prefix(self, search):
        return FakeTensor(np.full((1, 6, 2), search))

    def head(self, tokens):
        return {
            "score_map": FakeTensor(np.full((1, 1, 2, 2), 0.5)),
            "box": np.asarray(self.tracker.state) + [0.5, 0.0, 0.0, 0.0],
        }

    def selector(self, tokens):
        return FakeTensor([[0.1, 0.2, 0.9, 0.4, 0.5, 0.6]])

    def selected_layer(self, probabilities):
        return 9

    def candidate(self, tokens, layer):
        if layer not in range(7, 13):
            raise AssertionError("Unexpected candidate layer")
        self.candidate_calls += 1
        shift = 0.2 * (layer - 6)
        return {
            "score_map": FakeTensor(np.full((1, 1, 2, 2), 0.6)),
            "box": np.asarray(self.tracker.state) + [shift, 0.0, 0.0, 0.0],
        }

    def pool(self, tokens, grid):
        return FakeTensor(np.full((1, (2 + grid ** 2) * 2), tokens.value.mean()))

    def decode(self, output, factor, image):
        return output["box"].tolist()

    def run_image(self, image, mode):
        search, factor = self.prepare(image)
        tokens = self.prefix(search)
        if mode == "fast":
            output = self.head(tokens)
        elif mode == "sgla":
            layer = self.selected_layer(self.selector(tokens))
            output = self.candidate(tokens, layer)
        else:
            raise ValueError("Unexpected fake runtime mode: {}".format(mode))
        return self.decode(output, factor, image)


def create_dataset(root, names, length=16):
    for name in names:
        directory = root / name
        directory.mkdir(parents=True)
        boxes = ["{},{},10,10".format(10 + index * 0.5, 10) for index in range(length)]
        (directory / "groundtruth.txt").write_text("\n".join(boxes) + "\n", encoding="utf-8")
        (directory / "absence.label").write_text("0\n" * length, encoding="utf-8")
        (directory / "cover.label").write_text("8\n" * length, encoding="utf-8")
        for index in range(1, length + 1):
            (directory / "{:08d}.jpg".format(index)).write_bytes(b"image")


class MultiactionValueTests(unittest.TestCase):
    def test_discounted_values_respect_valid_mask_and_fast_baseline(self):
        policy_iou = np.zeros((2, 7, 4), dtype=np.float64)
        policy_iou[:, 0] = 0.5
        policy_iou[:, 1] = 0.7
        policy_iou[:, 2:] = 0.4
        valid = np.asarray([[True, True, True, True], [True, False, False, False]])
        values, deltas, coverage, eligible = discounted_policy_values(
            policy_iou, valid, horizon=4, gamma=0.9, min_valid_fraction=0.5
        )
        self.assertTrue(eligible[0])
        self.assertFalse(eligible[1])
        self.assertAlmostEqual(deltas[0, 0], 0.2)
        self.assertTrue(np.isnan(values[1]).all())
        np.testing.assert_allclose(coverage, [1.0, 0.25])

    def test_nested_group_oof_predicts_all_six_actions(self):
        generator = np.random.default_rng(14)
        sequences = np.repeat(
            ["video_{:02d}".format(index) for index in range(8)], 3
        )
        count = len(sequences)
        selector = generator.uniform(0.0, 1.0, size=(count, 6)).astype(np.float32)
        previous = np.tile([8.0, 9.0, 10.0, 11.0], (count, 1))
        data = {
            "sequence": sequences,
            "split": np.repeat(["train"] * 4 + ["val"] * 4, 3),
            "frame_id": np.arange(count, dtype=np.int64) + 1,
            "anchor_offset": np.tile([1, 2, 3], 8),
            "token_features": generator.normal(size=(count, 12)).astype(np.float32),
            "fast_score_map": generator.normal(size=(count, 1, 2, 2)).astype(np.float32),
            "selector_prob": selector,
            "selected_layer": np.argmax(selector, axis=1) + 7,
            "previous_state": previous.copy(),
            "previous_fast_box": previous.copy(),
            "fast_box": previous + [0.2, 0.0, 0.0, 0.0],
            "image_hw": np.tile([64, 64], (count, 1)),
            "resize_factor": np.ones(count),
            "previous_action": np.zeros(count, dtype=np.int64),
            "motion_history": generator.normal(
                scale=0.01, size=(count, len(MOTION_FEATURE_NAMES))
            ).astype(np.float32),
            "action_targets": selector.astype(np.float64) - 0.5,
            "immediate_targets": selector.astype(np.float64) - 0.45,
        }
        result = nested_group_oof(
            data,
            ("history_only", "selector_fast_motion", "token_selector_fast_motion"),
            outer_folds=4,
            inner_folds=2,
            alphas=(1.0, 10.0),
            pca_dimensions=4,
            pca_oversample=2,
            pca_power_iterations=0,
            seed=14,
            batch_size=32,
        )
        for prediction in result["predictions"].values():
            self.assertEqual(prediction.shape, (count, 6))
            self.assertTrue(np.isfinite(prediction).all())
        actions, selectors = mechanism_rows(data, 20, 14)
        models = model_summary_rows(
            data,
            result,
            ("history_only", "selector_fast_motion", "token_selector_fast_motion"),
            "token_selector_fast_motion",
            4,
            20,
            15,
        )
        budgets = budget_rows(
            data,
            result,
            ("token_selector_fast_motion",),
            "token_selector_fast_motion",
            (0.0, 50.0, 100.0),
            20,
            16,
        )
        predictions = prediction_rows(
            data, result, ("token_selector_fast_motion",)
        )
        self.assertTrue(actions)
        self.assertEqual({row["scope"] for row in selectors}, {"train", "val", "all"})
        self.assertEqual(len(models), 3)
        self.assertEqual(len(budgets), 3)
        self.assertEqual(len(predictions), count)


class MultiactionCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        create_dataset(
            self.root / "dataset",
            ["GOT-10k_Train_000001", "GOT-10k_Train_000002"],
        )
        self.entry = {
            "sequence": "GOT-10k_Train_000001",
            "split": "train",
            "start": 0,
            "stop": 16,
            "annotation_sha256": None,
            "visible_sha256": None,
        }
        self.runtime = FakeRuntime()
        directory = self.root / "dataset" / self.entry["sequence"]
        self.gt = np.loadtxt(directory / "groundtruth.txt", delimiter=",", ndmin=2)
        self.valid = np.ones(len(self.gt), dtype=bool)
        self.arrays, self.initial = collect_multiaction(
            self.runtime,
            directory,
            "got10k_train",
            self.entry,
            self.gt,
            self.valid,
            pool_grid=2,
            anchor_count=3,
            horizon=4,
            gamma=0.9,
            min_valid_fraction=0.75,
            verify_actions=True,
        )

    def _run(self):
        second = dict(self.entry, sequence="GOT-10k_Train_000002", split="val")
        manifest = {
            "schema": "stage4_ab_v1",
            "dataset": "got10k_train",
            "root": str((self.root / "dataset").resolve()),
            "seed": 1,
            "clip_frames": 16,
            "sequences": [self.entry, second],
        }
        manifest["fingerprint"] = fingerprint(manifest)
        return make_run(manifest, self.runtime.identity, 2, 3, 4, 0.9, 0.75, True)

    def test_all_actions_share_anchor_and_fast_branch_is_exact(self):
        self.assertEqual(self.arrays["policy_boxes"].shape, (3, len(ACTION_NAMES), 4, 4))
        np.testing.assert_allclose(
            self.arrays["action_boxes"], self.arrays["policy_boxes"][:, :, 0]
        )
        offsets = self.arrays["anchor_offset"]
        expected_fast = np.stack(
            [self.arrays["baseline_boxes"][offset - 1 : offset + 3] for offset in offsets]
        )
        np.testing.assert_allclose(self.arrays["policy_boxes"][:, 0], expected_fast)
        self.assertEqual(self.runtime.candidate_calls, 19)
        self.assertTrue(np.isfinite(self.arrays["discounted_delta"]).all())

    def test_cache_roundtrip_rejects_branch_tampering(self):
        run = self._run()
        path = multiaction_cache_path(self.root / "cache", self.entry)
        metadata = {
            "run_fingerprint": run["fingerprint"],
            "entry": self.entry,
            "init_box": self.initial,
        }
        write_cache(path, self.arrays, metadata)
        report, _, _ = load_and_audit_cache(path, self.entry, run)
        self.assertEqual(report["eligible_anchors"], 3)

        changed = {key: value.copy() for key, value in self.arrays.items()}
        changed["policy_boxes"][0, 2, 0, 0] += 1.0
        bad_path = self.root / "bad_multiaction.npz"
        write_cache(bad_path, changed, metadata)
        with self.assertRaisesRegex(ValueError, "Action boxes disagree"):
            load_and_audit_cache(bad_path, self.entry, run)


if __name__ == "__main__":
    unittest.main()
