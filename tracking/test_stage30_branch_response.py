"""Corrected-parent, override semantics and six-point workflow tests without CUDA."""

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from stage4_common import file_hash, write_cache, write_json
from stage27_uav_common import checked_image
from stage29_candidate_common import event_filename, event_metadata
from stage29_candidate_summary import make_decision, summarize_event
from stage30_branch_response_collect import collect_point, make_run
from stage30_branch_response_common import (audit_point, load_inputs, load_point, make_plan,
    point_filename, point_metadata, point_source, validate_plan)
from stage30_branch_response_summary import describe_point, render_point
import test_stage29_candidate as previous_tests
from test_stage28_response import ArrayTensor


HERE = Path(__file__).resolve().parent
PILLOW = importlib.util.find_spec("PIL") is not None


def frame_reader(path):
    path = Path(path)
    video = int(path.parent.name[3:10])
    frame = int(path.stem[3:])
    return np.full((64, 96, 3), 64 * video + frame, np.uint8)


class FailureRuntime(previous_tests.BranchRuntime):
    def initialize(self, image, box):
        super().initialize(image, box)
        self.video_mode = int(image[0, 0, 0]) // 64

    def candidate(self, tokens, layer):
        output = super().candidate(tokens, layer)
        repeat_frame = {1: 2, 3: 4}.get(self.video_mode)
        if self.tracker.frame_id == repeat_frame and abs(float(tokens[0]) - 40) < .001:
            raw = output["score_map"].numpy().copy()
            raw[0, 0, 2, 0], raw[0, 0, 2, 2] = .95, .8
            output["score_map"] = ArrayTensor(raw)
        return output


def input_fixture(root):
    with patch.object(previous_tests, "BranchRuntime", FailureRuntime), patch.object(previous_tests, "fake_image", frame_reader):
        upstream, plan, run, caches, _ = previous_tests.branch_fixture(root, all_events=True)
    source = root / "stage29"
    write_json(source / "plan.json", plan)
    write_json(source / "run.json", run)
    hashes, rows = {}, []
    for event in plan["events"]:
        path = source / "branches" / event_filename(event)
        data = caches[path.name]
        write_cache(path, data, event_metadata(run, event))
        hashes[path.name] = file_hash(path)
        trace = upstream["source_info"][2][event["sequence"]][1]
        rows.append(summarize_event(event, data, trace)[1])
    write_json(source / "analysis" / "stage29_decision.json", make_decision(plan, rows, [], run, hashes))
    reader = lambda directory, frame: checked_image(directory, frame, reader=frame_reader)
    return load_inputs(source), reader


def point_fixture(root):
    inputs, reader = input_fixture(root)
    plan = make_plan(inputs)
    rt = FailureRuntime()
    run = make_run(rt, inputs, plan)
    caches = {point_filename(p): collect_point(rt, p, inputs, run, image_reader=reader) for p in plan["points"]}
    return inputs, plan, run, caches, reader


class CorrectedParentTests(unittest.TestCase):
    def test_frozen_selection_first_failures_context_and_stable_controls(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            inputs, _ = input_fixture(Path(temp))
            plan = make_plan(inputs)
            self.assertEqual(plan["replay_frames"], 6)
            low = [p for p in plan["points"] if p["purpose"] == "first_low_iou"]
            self.assertEqual({p["branch_offset"] for p in low}, {1, 3})
            self.assertEqual(len([p for p in plan["points"] if p["purpose"] == "stable_control"]), 2)
            self.assertEqual(len([p for p in plan["points"] if p["purpose"] == "pre_failure_context"]), 2)
            changed = copy.deepcopy(plan)
            changed["points"][0]["branch_rank"] = 1
            with self.assertRaises(ValueError):
                validate_plan(changed, inputs)

    def test_replay_uses_corrected_parent_even_when_wrong_parent_produces_same_box(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            inputs, plan, run, caches, _ = point_fixture(Path(temp))
            point = next(p for p in plan["points"] if p["purpose"] == "first_low_iou" and p["branch_offset"] == 1)
            data = caches[point_filename(point)]
            _, trace, branch = point_source(inputs, point)
            self.assertFalse(np.array_equal(data["previous"], trace["previous"][point["frame_id"] - 1]))
            self.assertEqual(float(data["replay_box_error"]), 0)
            row, _, _, _ = describe_point(point, data, inputs["source"])
            self.assertEqual(row["model_top1_iou"], 0)
            self.assertTrue(row["weighted_top5_gt_recall"])
            self.assertEqual(row["observation"], "inside_gt_in_weighted_top5_not_top1")
            wrong = {k: v.copy() for k, v in data.items()}
            wrong["previous"] = trace["previous"][point["frame_id"] - 1].copy()
            with self.assertRaises(AssertionError):
                audit_point(wrong, point, branch, inputs["source"])

    def test_offset_zero_distinguishes_model_output_from_injected_output(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            inputs, plan, run, caches, _ = point_fixture(Path(temp))
            point = next(p for p in plan["points"] if p["branch_offset"] == 0)
            data = caches[point_filename(point)]
            row, _, _, _ = describe_point(point, data, inputs["source"])
            self.assertTrue(row["output_was_overridden"])
            self.assertEqual(row["recorded_branch_iou"], 1)
            self.assertEqual(row["model_top1_iou"], 0)
            self.assertEqual(float(data["replay_box_error"]), 0)
            self.assertFalse(np.array_equal(data["native_box"], data["recorded_branch_box"]))

    def test_tampered_maps_geometry_and_metadata_fail(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            inputs, plan, run, caches, _ = point_fixture(Path(temp))
            point = plan["points"][0]
            data = caches[point_filename(point)]
            _, _, branch = point_source(inputs, point)
            for key in ("previous", "gt", "native_box", "recorded_branch_box", "raw", "size", "layer", "crop"):
                wrong = {k: v.copy() for k, v in data.items()}
                if key == "size":
                    y, x = np.unravel_index(int(np.argmax(data["raw"] * data["window"])), data["raw"].shape)
                    wrong[key][0, y, x] += 1
                else:
                    wrong[key].flat[0] += 1
                with self.assertRaises((ValueError, AssertionError)):
                    audit_point(wrong, point, branch, inputs["source"])
            path = Path(temp) / "point.npz"
            write_cache(path, data, point_metadata(run, point))
            load_point(path, point, inputs, run)
            with self.assertRaises(ValueError):
                load_point(path, point, inputs, {**run, "fingerprint": "wrong"})


class WorkflowTests(unittest.TestCase):
    def test_cpu_plan_cli_and_idempotent_resume(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            inputs, _ = input_fixture(root)
            output = root / "stage30"
            cmd = [sys.executable, "-B", str(HERE / "stage30_branch_response_collect.py"),
                   "--input", str(inputs["directory"]), "--output_dir", str(output), "--plan_only"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("6 single-frame replays", result.stdout)
            before = (output / "plan.json").read_bytes()
            self.assertEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            self.assertEqual(before, (output / "plan.json").read_bytes())
            self.assertFalse((output / "run.json").exists())

    def test_full_summary_partial_rejection_and_upstream_immutability(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            inputs, plan, run, caches, _ = point_fixture(root)
            files = [p for d in (inputs["directory"], inputs["upstream"]["directory"],
                                inputs["upstream"]["source_info"][0]) for p in d.rglob("*.npz")]
            before = {str(p): file_hash(p) for p in files}
            output = root / "stage30"
            write_json(output / "plan.json", plan)
            write_json(output / "run.json", run)
            for point in plan["points"]:
                write_cache(output / "responses" / point_filename(point), caches[point_filename(point)], point_metadata(run, point))
            cmd = [sys.executable, "-B", str(HERE / "stage30_branch_response_summary.py"), "--input", str(output)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((output / "analysis" / "stage30_response_diagnostic.json").read_text())
            self.assertEqual(report["status"], "COMPLETE_CORRECTED_BRANCH_RESPONSE_AUDIT")
            self.assertEqual(report["audited_failure_points"], 2)
            self.assertFalse(report["new_policy_evaluated"])
            self.assertEqual(before, {str(p): file_hash(p) for p in files})
            (output / "responses" / point_filename(plan["points"][0])).unlink()
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            result = subprocess.run(cmd + ["--allow_incomplete"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((output / "analysis_pilot" / "stage30_response_diagnostic.json").read_text())
            self.assertEqual(report["status"], "PILOT_INCOMPLETE")

    def test_missing_stage29_cache_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            inputs, _ = input_fixture(Path(temp))
            next((inputs["directory"] / "branches").glob("*.npz")).unlink()
            with self.assertRaises(ValueError):
                load_inputs(inputs["directory"])

    @unittest.skipUnless(PILLOW, "Overlay rendering requires Pillow")
    def test_render_injected_context_and_corrected_failure(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            inputs, plan, run, caches, reader = point_fixture(root)
            for point in [p for p in plan["points"] if p["branch_offset"] in (0, 1)][:2]:
                path = render_point(point, caches[point_filename(point)], inputs, root / "render", reader)
                with Image.open(path) as image:
                    self.assertEqual(image.size, (1200, 850))


if __name__ == "__main__":
    unittest.main()
