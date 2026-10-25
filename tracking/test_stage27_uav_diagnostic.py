"""Integrity, crop geometry and causal rollout tests without CUDA dependencies."""

import copy
import csv
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from stage4_common import write_cache, write_json
from stage27_uav_common import (SCHEMA, audit_trace, checked_image, crop_geometry,
    estimate_motion, failure_events, load_manifest, load_trace, publish_identical,
    scan_dataset, scene_geometry, seal, select_entries, success_auc)
from stage27_uav_collect import collect_video, make_run, writer_lock
from stage27_uav_manifest import build_manifest
from stage27_uav_summary import EVENT_FIELDS, event_rows, summarize_scenarios, write_table


HERE = Path(__file__).resolve().parent
CV2 = importlib.util.find_spec("cv2") is not None
TORCH = importlib.util.find_spec("torch") is not None


def fake_image(path):
    number = int(Path(path).stem[3:])
    return np.full((64, 96, 3), number, np.uint8)


def fixture(root):
    names = ["uav0000001_00000_s", "uav0000001_00020_s", "uav0000002_00000_s", "uav0000003_00000_s"]
    (root / "annotations").mkdir(parents=True)
    for name in names:
        directory = root / "sequences" / name
        directory.mkdir(parents=True)
        for i in range(6):
            (directory / ("img{:07d}.jpg".format(i + 1))).write_bytes(bytes([i + 1]) * 12)
        (root / "annotations" / (name + ".txt")).write_text(
            "\n".join("{},{},10,10".format(5 + i, 5) for i in range(6)), encoding="utf-8")
    with redirect_stdout(io.StringIO()):
        return build_manifest(root, sequences=3, seed=2091, reader=fake_image,
                              expected_sequences=4, expected_frames=24)


class ScalarTensor:
    def detach(self):
        return self

    def amax(self):
        return self

    def item(self):
        return .8


class FakeRuntime:
    def __init__(self):
        self.torch = SimpleNamespace(no_grad=nullcontext)
        self.tracker = SimpleNamespace(params=SimpleNamespace(search_factor=4, search_size=256),
                                       state=None, frame_id=0)
        self.identity = dict(checkpoint_sha256="fixture")
        self.verifications = 0

    def initialize(self, image, box):
        self.tracker.state = list(box)
        self.tracker.frame_id = 0

    def prepare(self, image):
        _, _, factor = crop_geometry(self.tracker.state, image.shape[:2], 4, 256)
        return np.asarray(self.tracker.state).copy(), factor

    def prefix(self, search):
        return search

    def selector(self, tokens):
        return tokens

    def selected_layer(self, value):
        return 7 + int(value[0]) % 6

    def candidate(self, tokens, layer):
        return dict(box=tokens + np.array([1, 0, 0, 0]), score_map=ScalarTensor())

    def verify(self, search, output):
        self.verifications += 1

    def decode(self, output, factor, image):
        return output["box"].tolist()


def fake_rollout(manifest):
    entry = manifest["sequences"][0]
    rt = FakeRuntime()
    run = make_run(rt, manifest)
    data = collect_video(rt, entry, run,
        image_reader=lambda root, frame: checked_image(root, frame, reader=fake_image),
        motion_estimator=lambda *args: np.array([np.nan, np.nan, 0, 0, 0]),
        sharpness_estimator=lambda *args: np.nan)
    return entry, run, data, rt


class DatasetTests(unittest.TestCase):
    def test_selection_uses_distinct_source_groups_and_stable_order(self):
        entries = [dict(sequence="s{}".format(i), source_group_proxy="g{}".format(i // 2)) for i in range(12)]
        a = select_entries(entries, 4, 7)
        self.assertEqual(a, select_entries(list(reversed(entries)), 4, 7))
        self.assertEqual(len({e["source_group_proxy"] for e in a}), 4)
        with self.assertRaises(ValueError):
            select_entries(entries, 7)

    def test_manifest_decodes_selected_images_and_is_immutable(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            manifest = fixture(root)
            self.assertEqual(manifest["dataset_frames"], 24)
            self.assertEqual(len(manifest["sequences"]), 3)
            path = root / "manifest.json"
            self.assertTrue(publish_identical(path, manifest))
            stamp = path.stat().st_mtime_ns
            self.assertFalse(publish_identical(path, manifest))
            self.assertEqual(stamp, path.stat().st_mtime_ns)
            self.assertEqual(load_manifest(path), manifest)
            changed = seal({k: v for k, v in manifest.items() if k != "fingerprint"} | {"seed": 999})
            with self.assertRaises(ValueError):
                publish_identical(path, changed)

    def test_image_tampering_and_invalid_annotations_are_detected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            manifest = fixture(root)
            frame = manifest["sequences"][0]["frames"][0]
            (root / frame["file"]).write_bytes(b"changed")
            with self.assertRaises(ValueError):
                checked_image(root, frame, reader=fake_image)
            (root / manifest["sequences"][0]["annotation"]).write_text("0,0,0,0\n")
            with self.assertRaises(ValueError):
                scan_dataset(root, 4, 24)

    def test_duplicate_frame_number_and_orphan_annotation_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            manifest = fixture(root)
            image = root / manifest["sequences"][0]["frames"][1]["file"]
            image.rename(image.with_name("img0000001.png"))
            with self.assertRaises(ValueError):
                scan_dataset(root, 4, 24)
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            fixture(root)
            (root / "annotations" / "orphan.txt").write_text("1,1,1,1")
            with self.assertRaises(ValueError):
                scan_dataset(root, 4, 24)


class GeometryTests(unittest.TestCase):
    def test_parent_not_current_gt_determines_crop(self):
        a = scene_geometry([10, 10, 10, 10], [12, 12, 8, 8], [100, 100], 4, 256)
        b = scene_geometry([10, 10, 10, 10], [80, 80, 8, 8], [100, 100], 4, 256)
        np.testing.assert_array_equal(a["crop"], b["crop"])
        self.assertEqual(a["coverage"], 1)
        self.assertEqual(b["coverage"], 0)

    def test_legacy_right_bottom_padding_is_preserved(self):
        crop, usable, factor = crop_geometry([85, 85, 10, 10], [100, 100], 4, 256)
        np.testing.assert_array_equal(crop, [70, 70, 40, 40])
        np.testing.assert_array_equal(usable, [70, 70, 29, 29])
        self.assertEqual(factor, 6.4)
        value = scene_geometry([85, 85, 10, 10], [99, 99, 1, 1], [100, 100], 4, 256)
        self.assertEqual(value["coverage"], 0)
        self.assertAlmostEqual(value["padding_fraction"], 1 - 29 * 29 / 1600)

    def test_subpixel_rounding_and_projected_size(self):
        value = scene_geometry([30.5, 30.5, 9, 16], [32, 32, 2, 3], [100, 100], 4, 256)
        self.assertEqual(value["crop"][2], 48)
        np.testing.assert_allclose(value["input_wh"], np.array([2, 3]) * 256 / 48)
        with self.assertRaises(ValueError):
            crop_geometry([0, 0, 0, 2], [10, 10], 4, 256)

    @unittest.skipUnless(CV2 and TORCH, "Original crop comparison requires OpenCV and PyTorch")
    def test_geometry_matches_original_sample_target_actual_padding(self):
        spec = importlib.util.spec_from_file_location("stage27_crop_reference",
            HERE.parent / "lib" / "train" / "data" / "processing_utils.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        image = np.ones((100, 120, 3), np.uint8)
        for parent in ([0, 0, 10, 10], [105, 85, 10, 10], [43.5, 32.5, 9, 16], [30, 20, 80, 70]):
            crop, usable, _ = crop_geometry(parent, image.shape[:2], 4, 256)
            patch, mask, _ = module.sample_target(image, parent, 4, output_sz=None)
            self.assertEqual(patch.shape[:2], (crop[3], crop[2]))
            self.assertEqual(int((~mask).sum()), int(usable[2:].prod()))

    @unittest.skipUnless(CV2, "OpenCV unavailable")
    def test_background_translation_and_textureless_unknown(self):
        import cv2
        rng = np.random.default_rng(5)
        old = rng.integers(0, 256, (160, 240, 3), dtype=np.uint8)
        new = cv2.warpAffine(old, np.array([[1., 0., 4.], [0., 1., 3.]]), (240, 160))
        motion = estimate_motion(old, new, [[50, 50, 10, 10]], [[54, 53, 10, 10]])
        self.assertAlmostEqual(motion[0], 5 / np.hypot(240, 160), delta=.003)
        self.assertLess(motion[1], .15)
        blank = np.zeros_like(old)
        self.assertTrue(np.isnan(estimate_motion(blank, blank, [[1, 1, 4, 4]], [[1, 1, 4, 4]])[0]))


class RolloutTests(unittest.TestCase):
    def test_full_trace_native_selector_and_future_gt_cannot_drive_boxes(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            manifest = fixture(Path(temp))
            entry, run, data, rt = fake_rollout(manifest)
            self.assertEqual(rt.verifications, 1)
            self.assertEqual(len(data["boxes"]), 5)
            self.assertGreater(len(set(data["layer"])), 1)
            changed = copy.deepcopy(manifest)
            for box in changed["sequences"][0]["ground_truth"][1:]:
                box[0] += 40
            _, _, other, _ = fake_rollout(changed)
            np.testing.assert_array_equal(data["boxes"], other["boxes"])
            self.assertFalse(np.array_equal(data["coverage"], other["coverage"]))

    def test_trace_checks_causal_history_geometry_ground_truth_and_identity(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            manifest = fixture(Path(temp))
            entry, run, data, _ = fake_rollout(manifest)
            path = Path(temp) / "trace.npz"
            write_cache(path, data, dict(run_fingerprint=run["fingerprint"], sequence=entry["sequence"]))
            load_trace(path, entry, run)
            for key in ("previous", "coverage", "gt", "crop", "iou"):
                bad = {k: v.copy() for k, v in data.items()}
                bad[key].flat[0] += 1
                with self.assertRaises((ValueError, AssertionError)):
                    audit_trace(bad, entry, run)
            with self.assertRaises(ValueError):
                load_trace(path, entry, {**run, "fingerprint": "wrong"})

    def test_writer_lock_rejects_concurrency_and_cleans_after_error(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with writer_lock(root):
                    with self.assertRaisesRegex(RuntimeError, "Collector lock"):
                        with writer_lock(root):
                            pass
                    raise RuntimeError("intentional")
            self.assertFalse((root / "collect.lock").exists())


class AnalysisTests(unittest.TestCase):
    def test_events_require_persistence_and_three_frame_recovery(self):
        values = [.8, .1, .1, .8, .1, .1, .1, .3, .6, .4, .6, .6, .6, .8]
        events = failure_events(values)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0], dict(start=4, confirmation=6, end=9, recovery=10, censor="none"))
        self.assertEqual(failure_events([.1, .1]), [])
        self.assertEqual(failure_events([.1, .1, .1])[-1]["censor"], "video_end")
        self.assertEqual(failure_events([.1, .1, .1, -1, .9])[-1]["censor"], "invalid_gap")

    def test_unknown_motion_is_separate_and_bootstrap_unit_is_video(self):
        a = dict(iou=np.array([0., 0., 0., 0.]), input_wh=np.ones((4, 2)) * 10,
                 coverage=np.ones(4), motion=np.full((4, 5), np.nan))
        b = dict(iou=np.ones(1), input_wh=np.ones((1, 2)) * 10,
                 coverage=np.ones(1), motion=np.full((1, 5), np.nan))
        rows = summarize_scenarios([({}, a), ({}, b)], samples=100)
        all_row = next(r for r in rows if r["axis"] == "all")
        self.assertEqual(all_row["macro_low_iou_fraction"], .5)
        self.assertEqual(all_row["videos"], 2)
        self.assertAlmostEqual(all_row["macro_auc"], success_auc([1.]) / 2)
        self.assertTrue(any(r["stratum"] == "unknown" for r in rows))
        singleton = summarize_scenarios([({}, b)], samples=100)[0]
        self.assertIsNone(singleton["auc_ci_low"])

    def test_no_failure_csv_still_has_header(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            path = Path(temp) / "events.csv"
            write_table(path, [], EVENT_FIELDS)
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, list(EVENT_FIELDS))
                self.assertEqual(list(reader), [])

    def test_summary_cli_checks_completion_and_emits_full_reports(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            manifest = fixture(root)
            entry, run, data, _ = fake_rollout(manifest)
            output = root / "collected"
            write_json(output / "run.json", run)
            for selected in manifest["sequences"]:
                adjusted = dict(data)
                adjusted["gt"] = np.asarray(selected["ground_truth"])[1:]
                write_cache(output / "traces" / (selected["sequence"] + "__stage27_uav.npz"), adjusted,
                            dict(run_fingerprint=run["fingerprint"], sequence=selected["sequence"]))
            command = [sys.executable, "-B", str(HERE / "stage27_uav_summary.py"), "--input", str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((output / "analysis" / "stage27_diagnostic.json").read_text())
            self.assertEqual(report["status"], "COMPLETE_DIAGNOSTIC")
            self.assertEqual(report["tracked_frames"], 15)
            self.assertEqual(report["failure_events"], 0)
            path = output / "traces" / (entry["sequence"] + "__stage27_uav.npz")
            path.unlink()
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(subprocess.run(command + ["--allow_incomplete"], capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
