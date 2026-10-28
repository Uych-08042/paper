"""Read-only review tests, including rendering when Pillow is available."""

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_cache, write_json
from stage27_uav_common import checked_image, failure_events
from stage27_uav_collect import collect_video, make_run
from stage27_uav_review import (aggregate, box_errors, build_review, describe_event,
    manual_template, render_contact, sample_frames, search_patch)
from test_stage27_uav_diagnostic import FakeRuntime, fake_image, fixture


HERE = Path(__file__).resolve().parent
PILLOW = importlib.util.find_spec("PIL") is not None


class JumpRuntime(FakeRuntime):
    def decode(self, output, factor, image):
        box = super().decode(output, factor, image)
        if self.tracker.frame_id == 0:
            box[0] += 20
        return box


def reviewed_fixture(root, failure=True):
    manifest = fixture(root / "dataset")
    rt = JumpRuntime() if failure else FakeRuntime()
    run = make_run(rt, manifest)
    source = root / "cache"
    write_json(source / "run.json", run)
    pairs = []
    reader = lambda directory, frame: checked_image(directory, frame, reader=fake_image)
    for entry in manifest["sequences"]:
        data = collect_video(rt, entry, run, image_reader=reader,
            motion_estimator=lambda *args: np.array([np.nan, np.nan, 0., 0., 0.]),
            sharpness_estimator=lambda *args: np.nan)
        write_cache(source / "traces" / (entry["sequence"] + "__stage27_uav.npz"), data,
                    dict(run_fingerprint=run["fingerprint"], sequence=entry["sequence"]))
        pairs.append((entry, data))
    return source, run, pairs, reader


class ReviewGeometryTests(unittest.TestCase):
    def test_center_and_scale_are_measured_separately(self):
        errors = box_errors([10, 0, 10, 10], [0, 0, 10, 10])
        self.assertEqual(errors["center_error_gt_units"], 1)
        self.assertEqual(errors["area_ratio"], 1)
        errors = box_errors([-5, -5, 20, 20], [0, 0, 10, 10])
        self.assertEqual(errors["center_error_gt_units"], 0)
        self.assertEqual(errors["width_ratio"], 2)
        self.assertEqual(errors["area_ratio"], 4)
        with self.assertRaises(ValueError):
            box_errors([0, 0, 0, 1], [0, 0, 1, 1])

    def test_onset_samples_do_not_wrap_or_repeat_short_video_frames(self):
        self.assertEqual(sample_frames(1, 6), [0, 1, 2, 4])
        self.assertEqual(sample_frames(96, 553), [86, 93, 95, 96, 97, 99, 106, 126])
        with self.assertRaises(ValueError):
            sample_frames(0, 6)

    def test_crop_view_preserves_black_padding_and_original_pixels(self):
        image = np.arange(10 * 12 * 3, dtype=np.uint8).reshape(10, 12, 3)
        crop = search_patch(image, [-2, -3, 8, 8], [0, 0, 6, 5])
        np.testing.assert_array_equal(crop[3:, 2:], image[:5, :6])
        self.assertFalse(crop[:3].any())
        self.assertFalse(crop[:, :2].any())
        with self.assertRaises(ValueError):
            search_patch(image, [0, 0, 200, 200], [0, 0, 2, 2], max_pixels=20)

    def test_raw_target_size_is_not_confused_with_projected_size(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            _, _, traces, _ = reviewed_fixture(Path(temp))
            entry, data = traces[0]
            event = failure_events(data["iou"])[0]
            row = describe_event(entry, data, event, 1)
            self.assertEqual(row["raw_gt_side_px"], 10)
            self.assertEqual(row["projected_input_side_px"], 64)
            self.assertIsNone(row["previous_iou"])
            self.assertIsNone(row["previous_10_background_motion_max"])
            self.assertEqual(row["onset_source_file"], entry["frames"][1]["file"])
            self.assertEqual(row["manual_cause"], "unreviewed")
            report = aggregate(traces, [row])
            self.assertEqual(report["events_with_low_onset_motion"], 0)
            json.dumps(report, allow_nan=False)


class ReviewIntegrityTests(unittest.TestCase):
    def test_complete_review_is_read_only_and_preserves_manual_notes(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            source, _, _, reader = reviewed_fixture(root)
            paths = list(source.rglob("*"))
            hashes = {str(path): file_hash(path) for path in paths if path.is_file()}
            output = root / "review"
            summary = build_review(source, output, tables_only=True, reader=reader)
            self.assertEqual(summary["videos"], 3)
            self.assertEqual(summary["events"], 3)
            self.assertEqual(summary["low_iou_frames"], 15)
            self.assertEqual(hashes, {str(path): file_hash(path) for path in paths if path.is_file()})
            labels = output / "manual_review.csv"
            with labels.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["notes"] = "uncertain, keep for continuous review"
            from stage27_uav_summary import write_table
            write_table(labels, rows)
            before = labels.read_bytes()
            build_review(source, output, tables_only=True, reader=reader)
            self.assertEqual(before, labels.read_bytes())
            self.assertIn("Tables-only", (output / "index.html").read_text())

    def test_missing_trace_and_nonempty_destination_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            source, _, traces, reader = reviewed_fixture(root)
            output = root / "review"
            output.mkdir()
            (output / "existing.txt").write_text("preserve")
            with self.assertRaisesRegex(ValueError, "empty review"):
                build_review(source, output, tables_only=True, reader=reader)
            self.assertEqual((output / "existing.txt").read_text(), "preserve")
            (source / "traces" / (traces[0][0]["sequence"] + "__stage27_uav.npz")).unlink()
            with self.assertRaisesRegex(ValueError, "complete frozen traces"):
                build_review(source, root / "another", tables_only=True, reader=reader)

    def test_zero_events_emits_valid_empty_tables(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            source, _, _, reader = reviewed_fixture(root, failure=False)
            output = root / "review"
            summary = build_review(source, output, tables_only=True, reader=reader)
            self.assertEqual(summary["events"], 0)
            with (output / "manual_review.csv").open(newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])

    @unittest.skipUnless(PILLOW, "Rendering requires Pillow")
    def test_rendered_review_opens_and_keeps_viewports_in_bounds(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            source, _, _, reader = reviewed_fixture(root)
            output = root / "review"
            build_review(source, output, reader=reader)
            paths = list((output / "contacts").glob("*.png"))
            self.assertEqual(len(paths), 3)
            with Image.open(paths[0]) as image:
                self.assertEqual(image.size, (1840, 880))
                self.assertEqual(image.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
