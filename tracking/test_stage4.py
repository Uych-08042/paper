"""CPU-only regression tests; no model weights, images or CUDA required."""

import copy
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from stage3_post_candidate_validate import MacroSuccessAUC, _box_iou, _fused_iou
from stage4_benchmark import summarize
from stage4_check_cache import check_directory
from stage4_collect import collect_clip, make_run, publish_run
from stage4_common import (
    audit_cache, cache_path, fingerprint, load_annotations, load_manifest,
    write_cache, write_json,
)
from stage4_manifest import build_manifest, subset_names, training_names
from validation_metrics import tracking_iou


def legacy_iou(pred, gt):
    pred = np.asarray(pred).astype(np.int64).astype(np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    left = max(pred[0], gt[0])
    top = max(pred[1], gt[1])
    right = min(pred[0] + pred[2] - 1, gt[0] + gt[2] - 1)
    bottom = min(pred[1] + pred[3] - 1, gt[1] + gt[3] - 1)
    intersection = max(0, right - left + 1) * max(0, bottom - top + 1)
    return intersection / (pred[2] * pred[3] + gt[2] * gt[3] - intersection)


class MetricTests(unittest.TestCase):
    def test_matches_legacy_with_negative_and_fractional_coordinates(self):
        rng = np.random.RandomState(2026)
        pred = rng.uniform(-10, 100, (100, 4))
        gt = rng.uniform(-10, 100, (100, 4))
        pred[:, 2:] = rng.uniform(1, 100, (100, 2))
        gt[:, 2:] = rng.uniform(1, 100, (100, 2))
        expected = np.asarray([legacy_iou(p, g) for p, g in zip(pred, gt)])
        np.testing.assert_allclose(tracking_iou(pred, gt), expected, atol=1e-15)
        np.testing.assert_allclose(tracking_iou(pred[:, None], gt[:, None]), expected[:, None])

    def test_invalid_zero_sized_and_touching_boxes(self):
        gt = [0, 0, 10, 10]
        self.assertEqual(float(tracking_iou([10, 0, 10, 10], gt)), 0)
        self.assertEqual(float(tracking_iou([0, 0, 0, 0], gt)), 0)
        for pred, truth, valid in (([np.nan, 0, 10, 10], gt, True),
                                   ([0, 0, -1, 10], gt, True),
                                   (gt, [0, 0, 0, 10], True), (gt, gt, False)):
            self.assertEqual(float(tracking_iou(pred, truth, valid)), -1)

    def test_fusion_quantizes_only_evaluation_not_consensus(self):
        box = np.asarray([[0.9, 0.9, 10.9, 10.9]] * 3)
        gt = box.copy()
        valid = np.asarray([False, True, False])
        result = _fused_iou(box, gt, np.asarray([1., 0.5, -1.]), valid)
        np.testing.assert_allclose(result, [1, legacy_iou(box[1], gt[1]), -1])
        self.assertLess(result[1], 1.)
        np.testing.assert_allclose(_box_iou(box, gt), 1.)

    def test_macro_auc_uses_strict_thresholds_and_equal_sequence_weight(self):
        rows = [{"dataset": "x", "sequence": "short"}] + [{"dataset": "x", "sequence": "long"}] * 3
        metric = MacroSuccessAUC(rows)
        values = np.asarray([0., .5, 1., -1.])
        expected = np.mean([(values[:1, None] > metric.thresholds).mean(),
                            (values[1:, None] > metric.thresholds).mean()]) * 100
        self.assertAlmostEqual(metric(values), expected)


def create_dataset(root, dataset, names, length=8):
    for name in names:
        directory = root / name if dataset == "got10k_train" else root / name.rsplit("-", 1)[0] / name
        directory.mkdir(parents=True)
        (directory / "groundtruth.txt").write_text("10,10,20,20\n" * length)
        if dataset == "got10k_train":
            (directory / "absence.label").write_text("0\n" * length)
            (directory / "cover.label").write_text("8\n" * length)
            images = directory
        else:
            (directory / "full_occlusion.txt").write_text(",".join(["0"] * length))
            (directory / "out_of_view.txt").write_text(",".join(["0"] * length))
            images = directory / "img"
            images.mkdir()
        for index in range(1, length + 1):
            (images / "{:08d}.jpg".format(index)).write_bytes(b"mock image")


class ManifestTests(unittest.TestCase):
    def test_deterministic_video_disjoint_splits_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = ["GOT-10k_Train_{:06d}".format(index) for index in range(4)]
            create_dataset(root, "got10k_train", names)
            args = ("got10k_train", root, names, 2, 2, 5, 2026)
            manifest = build_manifest(*args)
            self.assertEqual(manifest, build_manifest(*args))
            self.assertEqual(len({entry["sequence"] for entry in manifest["sequences"]}), 4)
            self.assertEqual([entry["split"] for entry in manifest["sequences"]], ["train", "train", "val", "val"])
            path = root / "manifest.json"
            write_json(path, manifest)
            self.assertEqual(load_manifest(path), manifest)
            manifest["sequences"][0]["start"] += 1
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_manifest(manifest)

    def test_lasot_layout_and_missing_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_dataset(root, "lasot_train", ["airplane-1", "airplane-2"])
            manifest = build_manifest("lasot_train", root, ["airplane-1", "airplane-2"], 1, 1, 8, 0)
            self.assertEqual(manifest["sequences"][0]["stop"], 8)
            (root / "airplane/airplane-1/img/00000001.jpg").unlink()
            with self.assertRaises(FileNotFoundError):
                build_manifest("lasot_train", root, ["airplane-1", "airplane-2"], 1, 1, 8, 0)

    def test_reject_official_got_test_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "list.txt").write_text("GOT-10k_Test_000001\n")
            with self.assertRaisesRegex(ValueError, "train directory"):
                training_names("got10k_train", root)

    def test_explicit_subset_needs_no_full_list_and_enforces_training_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            indices = np.loadtxt(
                Path(__file__).resolve().parents[1]
                / "lib/train/data_specs/got10k_train_split.txt",
                dtype=int,
                ndmin=1,
            )
            allowed = "GOT-10k_Train_{:06d}".format(int(indices[0]) + 1)
            path = root / "subset.txt"
            path.write_text(allowed + "\n", encoding="utf-8")
            self.assertEqual(subset_names("got10k_train", path), [allowed])
            path.write_text("GOT-10k_Test_000001\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside the repository training split"):
                subset_names("got10k_train", path)

    def test_short_videos_and_visibility_lengths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = ["GOT-10k_Train_1", "GOT-10k_Train_2"]
            create_dataset(root, "got10k_train", names, length=3)
            manifest = build_manifest("got10k_train", root, names, 1, 1, 128, 0)
            self.assertEqual(manifest["sequences"][0]["stop"], 3)
            (root / names[0] / "cover.label").write_text("8\n")
            with self.assertRaisesRegex(ValueError, "length mismatch"):
                load_annotations(root, "got10k_train", names[0])


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
    """Exercise the actual rollout/serialization code with a cheap fake model."""
    def __init__(self):
        self.torch = SimpleNamespace(no_grad=nullcontext)
        self.tracker = SimpleNamespace(state=None, frame_id=0)
        self.identity = {"lens_z": 2, "lens_x": 4, "embed_dim": 2}
        self.verifications = 0

    def read(self, path):
        return np.full((64, 64, 3), int(Path(path).stem), dtype=np.uint8)

    def initialize(self, image, box):
        self.tracker.state, self.tracker.frame_id = list(box), 0

    def prepare(self, image):
        return float(image[0, 0, 0]) + self.tracker.state[0], 1.0

    def prefix(self, search):
        return FakeTensor(np.full((1, 6, 2), search))

    def head(self, tokens):
        return {"score_map": FakeTensor(np.full((1, 1, 2, 2), 0.5)),
                "box": np.asarray(self.tracker.state) + [0.5, 0, 0, 0]}

    def selector(self, tokens):
        return FakeTensor([[.1, .2, .9, .4, .5, .6]])

    def selected_layer(self, probabilities):
        return 9

    def pool(self, tokens, grid):
        return FakeTensor(np.full((1, (2 + grid ** 2) * 2), tokens.value.mean()))

    def decode(self, output, factor, image):
        return output["box"].tolist()

    def candidate(self, tokens, layer):
        result = self.head(tokens)
        result["box"] = np.asarray(self.tracker.state) + [(layer - 6) * .25, 0, 0, 0]
        return result

    def verify(self, search, scripted):
        self.verifications += 1


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        names = ["GOT-10k_Train_1", "GOT-10k_Train_2"]
        create_dataset(self.root / "dataset", "got10k_train", names, 4)
        self.manifest = build_manifest("got10k_train", self.root / "dataset", names, 1, 1, 4, 0)
        self.rt = FakeRuntime()
        self.run = make_run(self.manifest, self.rt.identity, 2, True, ["sgla", "fast"], True)
        self.output = self.root / "cache"

    def collect(self, entry, trajectory, gt_override=None):
        directory, gt, valid = load_annotations(self.manifest["root"], "got10k_train", entry["sequence"])
        if gt_override is not None:
            gt = gt_override
        data, initial = collect_clip(self.rt, directory, "got10k_train", entry, gt, valid, trajectory, 2, True, True)
        meta = {"run_fingerprint": self.run["fingerprint"], "entry": entry,
                "trajectory": trajectory, "init_box": initial}
        return data, meta

    def save_all(self):
        publish_run(self.output, self.run)
        for entry in self.manifest["sequences"]:
            for trajectory in self.run["trajectories"]:
                data, meta = self.collect(entry, trajectory)
                write_cache(cache_path(self.output, entry, trajectory), data, meta)

    def test_actual_rollout_complete_roundtrip_and_full_tokens(self):
        self.save_all()
        reports = check_directory(self.output)
        self.assertEqual(len(reports), 4)
        self.assertEqual(sum(row["decision_frames"] for row in reports), 12)
        self.assertEqual(self.rt.verifications, 4)
        self.assertFalse(list(self.output.rglob("*.tmp")))
        publish_run(self.output, self.run)
        changed = copy.deepcopy(self.run)
        changed["pool_grid"] = 3
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            publish_run(self.output, changed)
        self.assertFalse((self.output / "run.lock").exists())

    def test_current_gt_cannot_change_features_or_rollout(self):
        entry = self.manifest["sequences"][0]
        original, _ = self.collect(entry, "sgla")
        gt = np.tile([10., 10., 20., 20.], (4, 1))
        gt[1:, :2] = 40
        changed, _ = self.collect(entry, "sgla", gt)
        for key in self.run["feature_fields"] + ["executed_action", "action_boxes"]:
            np.testing.assert_array_equal(original[key], changed[key], err_msg=key)
        self.assertFalse(np.array_equal(original["iou"], changed["iou"]))

    def test_fast_and_sgla_follow_their_own_predictions(self):
        entry = self.manifest["sequences"][0]
        fast, _ = self.collect(entry, "fast")
        sgla, _ = self.collect(entry, "sgla")
        np.testing.assert_array_equal(fast["previous_state"][0], sgla["previous_state"][0])
        self.assertFalse(np.array_equal(fast["previous_state"][1:], sgla["previous_state"][1:]))

    def test_audit_rejects_corruption(self):
        entry = self.manifest["sequences"][0]
        data, meta = self.collect(entry, "sgla")
        path = cache_path(self.output, entry, "sgla")
        mutations = {
            "previous_state": (1, 0), "previous_fast_box": (1, 0), "previous_action": (1,),
            "iou": (0, 1), "delta": (0, 1), "executed_action": (0,), "frame_id": (0,),
        }
        for key, index in mutations.items():
            with self.subTest(key=key):
                bad = {name: values.copy() for name, values in data.items()}
                bad[key][index] += 1
                write_cache(path, bad, meta)
                with self.assertRaises(ValueError):
                    audit_cache(path, entry, "sgla", self.run)
        data["token_features"][0, 0] = np.nan
        write_cache(path, data, meta)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            audit_cache(path, entry, "sgla", self.run)

    def test_invalid_frames_are_masked_not_used_as_zero_gain(self):
        entry = self.manifest["sequences"][0]
        data, meta = self.collect(entry, "fast")
        data["valid_gt"][1] = False
        data["iou"][1] = -1
        data["delta"][1] = np.nan
        path = cache_path(self.output, entry, "fast")
        write_cache(path, data, meta)
        self.assertEqual(audit_cache(path, entry, "fast", self.run)["valid_frames"], 2)
        data["delta"][1] = 0
        write_cache(path, data, meta)
        with self.assertRaisesRegex(ValueError, "Delta"):
            audit_cache(path, entry, "fast", self.run)

    def test_checker_rejects_missing_and_mixed_files(self):
        self.save_all()
        entry = self.manifest["sequences"][0]
        cache_path(self.output, entry, "fast").unlink()
        with self.assertRaisesRegex(ValueError, "1 missing"):
            check_directory(self.output)
        self.save_all()
        (self.output / "unknown.npz").write_bytes(b"unknown")
        with self.assertRaisesRegex(ValueError, "1 unexpected"):
            check_directory(self.output)

    def test_checker_rejects_video_overlap_and_bad_action_order(self):
        self.save_all()
        bad = copy.deepcopy(self.manifest)
        bad["sequences"][1]["sequence"] = bad["sequences"][0]["sequence"]
        bad["fingerprint"] = fingerprint({key: value for key, value in bad.items() if key != "fingerprint"})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_manifest(bad)
        run = copy.deepcopy(self.run)
        run["actions"].reverse()
        run["fingerprint"] = fingerprint({key: value for key, value in run.items() if key != "fingerprint"})
        write_json(self.output / "run.json", run)
        with self.assertRaisesRegex(ValueError, "action order"):
            check_directory(self.output)


class BenchmarkTests(unittest.TestCase):
    def test_summary(self):
        rows = [{"scope": "integrated", "operation": "fast", "latency_ms": value} for value in (1., 2., 6.)]
        result = summarize(rows)[0]
        self.assertEqual(result["samples"], 3)
        self.assertEqual(result["mean_ms"], 3.)
        self.assertEqual(result["p50_ms"], 2.)
        self.assertAlmostEqual(result["p95_ms"], 5.6)


if __name__ == "__main__":
    unittest.main()
