"""Replay, spatial candidate and ranking tests; GPU inference is tested on server data."""

import ast
import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from stage4_common import file_hash, write_cache, write_json
from stage27_uav_collect import collect_video as collect_stage27, make_run as make_stage27_run
from stage27_uav_common import checked_image, tracking_iou
from stage28_response_collect import collect_video, make_run
from stage28_response_common import (DESIGN, audit_response, dense_boxes, load_response,
    load_source, make_plan, observation_label, peak_candidates, plan_video,
    response_metrics, validate_plan)
from stage28_response_summary import describe_frames, render_event, scale_temporal_rows
from test_stage27_uav_diagnostic import fake_image, fixture


HERE = Path(__file__).resolve().parent
TORCH = importlib.util.find_spec("torch") is not None
PILLOW = importlib.util.find_spec("PIL") is not None


class ArrayTensor:
    def __init__(self, array):
        self.array = np.asarray(array, np.float32)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def amax(self):
        return ArrayTensor(self.array.max())

    def item(self):
        return float(self.array)


class MapRuntime:
    """A deterministic CENTER tracker that jumps to a neighbour on frame 2."""
    def __init__(self):
        self.search_side = 4
        self.identity = dict(checkpoint_sha256="fixture", lens_x=16, parameter="deit_distilled")
        self.torch = SimpleNamespace(no_grad=nullcontext)
        self.tracker = SimpleNamespace(params=SimpleNamespace(search_factor=4., search_size=256),
            state=None, frame_id=0, output_window=np.ones((4, 4), np.float32))
        self.verifications = 0

    def initialize(self, image, box):
        self.tracker.state = list(box)
        self.tracker.frame_id = 0

    def prepare(self, image):
        import math
        state = np.asarray(self.tracker.state, float)
        factor = 256 / math.ceil(4 * np.sqrt(state[2:].prod()))
        return state.copy(), factor

    def prefix(self, search):
        return search

    def selector(self, tokens):
        return tokens

    def selected_layer(self, value):
        return 7 + int(value[0]) % 6

    def candidate(self, tokens, layer):
        raw = np.full((1, 1, 4, 4), .01, np.float32)
        raw[0, 0, 2, 2] = .9
        if self.tracker.frame_id == 1:
            raw[0, 0, 2, 3] = .95
        size = np.full((1, 2, 4, 4), .25, np.float32)
        offset = np.zeros((1, 2, 4, 4), np.float32)
        offset[:, 0] = .1
        return dict(score_map=ArrayTensor(raw), size_map=ArrayTensor(size), offset_map=ArrayTensor(offset))

    def decode(self, output, factor, image):
        boxes = dense_boxes(output["size_map"].numpy()[0], output["offset_map"].numpy()[0],
                            self.tracker.state, factor, 256, image.shape[:2])
        index = int(np.argmax(output["score_map"].numpy().reshape(4, 4) * self.tracker.output_window))
        return boxes[index].tolist()

    def verify(self, search, output):
        self.verifications += 1


def source_fixture(root):
    manifest = fixture(root / "dataset")
    runtime = MapRuntime()
    source = root / "source"
    run = make_stage27_run(runtime, manifest)
    write_json(source / "run.json", run)
    reader = lambda directory, frame: checked_image(directory, frame, reader=fake_image)
    for entry in manifest["sequences"]:
        data = collect_stage27(runtime, entry, run, image_reader=reader,
            motion_estimator=lambda *args: np.array([np.nan, np.nan, 0., 0., 0.]),
            sharpness_estimator=lambda *args: np.nan)
        write_cache(source / "traces" / (entry["sequence"] + "__stage27_uav.npz"), data,
                    dict(run_fingerprint=run["fingerprint"], sequence=entry["sequence"]))
    info = load_source(source)
    plan = make_plan(*info)
    return source, info, plan, reader


def response_fixture(root):
    source, info, plan, reader = source_fixture(root)
    _, old, traces, _ = info
    rt = MapRuntime()
    run = make_run(rt, old, plan)
    caches = {}
    for video in plan["videos"]:
        entry, trace = traces[video["sequence"]]
        caches[entry["sequence"]] = collect_video(rt, entry, trace, video, old, run, image_reader=reader)
    return info, plan, run, caches, reader


class DecoderTests(unittest.TestCase):
    def test_mapping_uses_floating_parent_center_and_native_minimum_box(self):
        size = np.full((2, 4, 4), .25, np.float32)
        offset = np.zeros_like(size)
        box = dense_boxes(size, offset, [30.3, 20.7, 10, 10], 6.4, 256, [100, 120])[10]
        np.testing.assert_allclose(box, [30.3, 20.7, 10, 10], atol=1e-6)
        self.assertNotAlmostEqual(box[0], 30., places=1)
        corner = dense_boxes(size * .01, offset, [0, 0, 10, 10], 6.4, 256, [100, 120])[0]
        np.testing.assert_array_equal(corner, [0, 0, 10, 10])

    @unittest.skipUnless(TORCH, "Original CENTER decoder comparison requires PyTorch")
    def test_numpy_dense_decode_agrees_with_original_cal_bbox_source(self):
        import torch
        tree = ast.parse((HERE.parent / "lib/models/layers/head.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CenterPredictor")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "cal_bbox")
        namespace = {"torch": torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "head.py", "exec"), namespace)
        rng = np.random.default_rng(4)
        for side in (4, 16):
            raw = rng.random((1, 1, side, side), dtype=np.float32)
            size = rng.uniform(.01, .8, (1, 2, side, side)).astype(np.float32)
            offset = rng.normal(0, .4, size.shape).astype(np.float32)
            norm = namespace["cal_bbox"](SimpleNamespace(feat_sz=side), torch.tensor(raw),
                                         torch.tensor(size), torch.tensor(offset)).view(-1, 4)
            factor, previous = 1.27, np.array([31.3, 20.7, 49.4, 70.1])
            native = (norm.mean(0) * 256 / factor).tolist()
            center = previous[:2] + previous[2:] / 2 - 128 / factor + native[:2]
            from stage28_response_common import clip_boxes
            expected = clip_boxes(np.r_[center - np.array(native[2:]) / 2, native[2:]], [480, 640])
            boxes = dense_boxes(size[0], offset[0], previous, factor, 256, [480, 640])
            np.testing.assert_allclose(boxes[int(raw.argmax())], expected, atol=.001, rtol=0)


class CandidateTests(unittest.TestCase):
    @staticmethod
    def boxes():
        return np.array([[i * 30., 0, 10, 10] for i in range(16)])

    def test_window_can_demote_correct_raw_winner(self):
        boxes = self.boxes()
        raw = np.full((4, 4), .01, np.float32)
        raw[0, 0], raw[3, 3] = .9, .8
        window = np.ones((4, 4), np.float32)
        window[0, 0] = .1
        metrics, details, _ = response_metrics(raw, window, boxes, boxes[0])
        self.assertEqual(metrics["raw_top1_iou"], 1)
        self.assertEqual(metrics["weighted_top1_iou"], 0)
        self.assertGreater(metrics["raw_gt_score_margin"], 0)
        self.assertLess(metrics["weighted_gt_score_margin"], 0)
        self.assertEqual(observation_label(metrics, 1), "inside_window_demotes_correct_raw_top1")

    def test_adjacent_gt_cell_can_be_dense_rank_two_but_not_a_local_peak(self):
        boxes = self.boxes()
        raw = np.full((4, 4), .01, np.float32)
        raw[1, 1], raw[1, 2] = .7, .9
        metrics, _, _ = response_metrics(raw, np.ones_like(raw), boxes, boxes[5])
        self.assertEqual(metrics["raw_best_gt_dense_score_rank"], 2)
        self.assertFalse(metrics["raw_gt_local_peak_present"])
        self.assertFalse(metrics["raw_top5_gt_recall"])
        self.assertEqual(metrics["dense_best_gt_iou"], 1)
        self.assertEqual(observation_label(metrics, 1), "inside_gt_box_exists_not_retained_by_fixed_top5")

    def test_nms_and_ties_are_deterministic_and_do_not_invent_five_candidates(self):
        boxes = np.tile([10., 10, 20, 20], (16, 1))
        raw = np.ones((4, 4), np.float32)
        self.assertEqual(peak_candidates(raw, boxes), [0])
        metrics, _, _ = response_metrics(raw, raw, boxes, np.array([200., 200, 20, 20]))
        self.assertEqual(metrics["raw_candidate_count"], 1)
        self.assertIsNone(metrics["raw_gt_score_margin"])
        self.assertEqual(observation_label(metrics, 1), "inside_no_gt_matching_dense_box")
        self.assertEqual(observation_label(metrics, .5), "low_iou_with_low_crop_coverage")


class PlanAndReplayTests(unittest.TestCase):
    def test_plan_covers_all_event_window_frames_and_unfiltered_quartile_controls(self):
        entry = dict(sequence="x", source_group_proxy="x")
        iou = np.r_[np.ones(20), np.zeros(5), np.ones(15)]
        video, events = plan_video(entry, dict(frame_id=np.arange(1, 41), iou=iou))
        self.assertEqual(events[0]["onset_frame"], 21)
        self.assertEqual(events[0]["first_frame"], 11)
        self.assertEqual(events[0]["stop_frame_exclusive"], 41)
        controls = [t["frame_id"] for t in video["frames"] if t["controls"]]
        self.assertEqual(controls, [10, 20, 30])
        self.assertEqual(len(video["frames"]), 31)
        self.assertEqual(sum(any(e["offset"] == 0 for e in t["events"]) for t in video["frames"]), 1)

    def test_replay_and_cache_bind_to_source_parent_and_ground_truth(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            info, plan, run, caches, _ = response_fixture(Path(temp))
            _, source, traces, _ = info
            self.assertEqual(len(plan["events"]), 3)
            self.assertEqual(plan["unique_replay_frames"], 15)
            video = plan["videos"][0]
            entry, trace = traces[video["sequence"]]
            data = caches[entry["sequence"]]
            self.assertEqual(data["replay_box_error"].max(), 0)
            audit_response(data, entry, trace, video, source)
            for key in ("previous", "native_boxes", "raw_boxes", "raw", "layer"):
                bad = {k: v.copy() for k, v in data.items()}
                bad[key].flat[0] += 1
                with self.assertRaises((ValueError, AssertionError)):
                    audit_response(bad, entry, trace, video, source)
            path = Path(temp) / "response.npz"
            write_cache(path, data, dict(run_fingerprint=run["fingerprint"], sequence=entry["sequence"]))
            load_response(path, entry, trace, video, source, run)
            with self.assertRaises(ValueError):
                load_response(path, entry, trace, video, source, {**run, "fingerprint": "bad"})

    def test_gt_label_changes_cannot_change_replayed_model_outputs(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            info, plan, run, caches, reader = response_fixture(Path(temp))
            _, source, traces, _ = info
            video = plan["videos"][0]
            entry, trace = traces[video["sequence"]]
            changed = {k: v.copy() for k, v in trace.items()}
            changed["gt"][:, 0] += 30
            changed["iou"] = tracking_iou(changed["boxes"], changed["gt"])
            other = collect_video(MapRuntime(), entry, changed, video, source, run, image_reader=reader)
            np.testing.assert_array_equal(other["native_boxes"], caches[entry["sequence"]]["native_boxes"])
            np.testing.assert_array_equal(other["raw"], caches[entry["sequence"]]["raw"])

    def test_plan_only_cli_has_no_torch_requirement_and_reuses_identical_plan(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            source, info, plan, _ = source_fixture(root)
            command = [sys.executable, "-B", str(HERE / "stage28_response_collect.py"),
                       "--input", str(source), "--output_dir", str(root / "replay"), "--plan_only"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("No GPU", result.stdout)
            path = root / "replay" / "plan.json"
            before = path.read_bytes()
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(before, path.read_bytes())
            bad = copy.deepcopy(plan)
            bad["design"]["topk"] = 9
            with self.assertRaises(ValueError):
                validate_plan(bad, info)


class SummaryTests(unittest.TestCase):
    def test_summary_cli_completion_source_immutability_and_csvs(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            info, plan, run, caches, _ = response_fixture(root)
            source_root, source, traces, _ = info
            before = {p.name: file_hash(p) for p in (source_root / "traces").glob("*.npz")}
            output = root / "responses"
            write_json(output / "plan.json", plan)
            write_json(output / "run.json", run)
            for name, data in caches.items():
                write_cache(output / "responses" / (name + "__stage28_response.npz"), data,
                            dict(run_fingerprint=run["fingerprint"], sequence=name))
            command = [sys.executable, "-B", str(HERE / "stage28_response_summary.py"), "--input", str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((output / "analysis" / "stage28_response_diagnostic.json").read_text())
            self.assertEqual(report["status"], "COMPLETE_RESPONSE_DIAGNOSTIC")
            self.assertEqual(report["audited_event_onsets"], 3)
            self.assertFalse(report["training_performed"])
            self.assertEqual(before, {p.name: file_hash(p) for p in (source_root / "traces").glob("*.npz")})
            self.assertTrue((output / "analysis" / "stage28_source_raw_size_summary.csv").is_file())
            first = next(iter(caches))
            (output / "responses" / (first + "__stage28_response.npz")).unlink()
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(subprocess.run(command + ["--allow_incomplete"], capture_output=True).returncode, 0)

    @unittest.skipUnless(PILLOW, "Heatmap rendering requires Pillow")
    def test_heatmap_rendering_uses_saved_responses(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            info, plan, run, caches, reader = response_fixture(root)
            _, source, traces, _ = info
            event = plan["events"][0]
            entry, trace = traces[event["sequence"]]
            path = render_event(source, entry, trace, caches[event["sequence"]], event, root / "figures", reader)
            with Image.open(path) as image:
                self.assertEqual(image.width, 960)
                self.assertEqual(image.height % 380, 0)


if __name__ == "__main__":
    unittest.main()
