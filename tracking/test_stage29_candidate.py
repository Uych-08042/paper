"""Synthetic causal-branch and frozen-gate tests; real SGLA runs on the server."""

import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_cache, write_json
from stage27_uav_collect import collect_video as collect_source, make_run as source_run
from stage27_uav_common import checked_image, seal, tracking_iou
from stage27_uav_manifest import build_manifest
from stage28_response_collect import collect_video as collect_responses, make_run as response_run
from stage28_response_common import make_plan as response_plan, load_source
from stage29_candidate_collect import collect_event, make_run
from stage29_candidate_common import (DESIGN, audit_event, event_filename, event_metadata,
    load_event, load_inputs, make_plan, validate_stage29_plan)
from stage29_candidate_summary import event_pass, make_decision, post_metrics, summarize_event
from test_stage27_uav_diagnostic import fake_image
from test_stage28_response import ArrayTensor, MapRuntime


HERE = Path(__file__).resolve().parent


class BranchRuntime(MapRuntime):
    """Only frame 2 switches to a separated peak; later frames retain their own box."""
    def candidate(self, tokens, layer):
        raw = np.full((1, 1, 4, 4), .001, np.float32)
        raw[0, 0, 2, 2] = .9
        if self.tracker.frame_id == 1:
            raw[0, 0, 2, 0], raw[0, 0, 2, 2] = .95, .8
        size = np.full((1, 2, 4, 4), .25, np.float32)
        offset = np.zeros_like(size)
        return dict(score_map=ArrayTensor(raw), size_map=ArrayTensor(size), offset_map=ArrayTensor(offset))


def input_fixture(root):
    dataset = root / "dataset"
    (dataset / "annotations").mkdir(parents=True)
    for video in range(1, 4):
        name = "uav{:07d}_00000_s".format(video)
        directory = dataset / "sequences" / name
        directory.mkdir(parents=True)
        for frame in range(48):
            (directory / ("img{:07d}.jpg".format(frame + 1))).write_bytes(bytes([frame + 1]) * 12)
        (dataset / "annotations" / (name + ".txt")).write_text("\n".join(["40,20,10,10"] * 48))
    with redirect_stdout(io.StringIO()):
        manifest = build_manifest(dataset, sequences=3, seed=2091, reader=fake_image,
                                  expected_sequences=3, expected_frames=144)
    rt, source = BranchRuntime(), root / "source"
    old_run = source_run(rt, manifest)
    write_json(source / "run.json", old_run)
    reader = lambda directory, frame: checked_image(directory, frame, reader=fake_image)
    for entry in manifest["sequences"]:
        data = collect_source(rt, entry, old_run, image_reader=reader,
            motion_estimator=lambda *args: np.array([np.nan, np.nan, 0., 0., 0.]),
            sharpness_estimator=lambda *args: np.nan)
        write_cache(source / "traces" / (entry["sequence"] + "__stage27_uav.npz"), data,
                    dict(run_fingerprint=old_run["fingerprint"], sequence=entry["sequence"]))
    info = load_source(source)
    old_plan = response_plan(*info)
    old_responses_run = response_run(rt, old_run, old_plan)
    replay = root / "responses"
    write_json(replay / "run.json", old_responses_run)
    write_json(replay / "plan.json", old_plan)
    hashes = {}
    for video in old_plan["videos"]:
        entry, trace = info[2][video["sequence"]]
        data = collect_responses(rt, entry, trace, video, old_run, old_responses_run, image_reader=reader)
        path = replay / "responses" / (entry["sequence"] + "__stage28_response.npz")
        write_cache(path, data, dict(run_fingerprint=old_responses_run["fingerprint"], sequence=entry["sequence"]))
        hashes[path.name] = file_hash(path)
    report = seal(dict(schema=old_plan["schema"], role="descriptive_response_competition_audit",
        status="COMPLETE_RESPONSE_DIAGNOSTIC", source_run_fingerprint=old_run["fingerprint"],
        plan_fingerprint=old_plan["fingerprint"], cache_sha256=hashes, missing_videos=[],
        audited_videos=3, replayed_unique_frames=old_plan["unique_replay_frames"],
        audited_event_onsets=3, planned_event_onsets=3))
    write_json(replay / "analysis" / "stage28_response_diagnostic.json", report)
    return load_inputs(replay), reader


def branch_fixture(root, all_events=False):
    inputs, reader = input_fixture(root)
    plan = make_plan(inputs)
    rt = BranchRuntime()
    run = make_run(rt, inputs, plan)
    _, source, traces, _ = inputs["source_info"]
    caches = {}
    with redirect_stdout(io.StringIO()):
        for event in plan["events"] if all_events else plan["events"][:1]:
            entry, trace = traces[event["sequence"]]
            data = collect_event(rt, event, entry, trace, inputs["caches"][entry["sequence"]],
                                 source, run, image_reader=reader)
            caches[event_filename(event)] = data
    return inputs, plan, run, caches, reader


class BranchTests(unittest.TestCase):
    def test_true_continuations_and_exact_native_replay(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            inputs, plan, run, caches, _ = branch_fixture(Path(temp))
            event = plan["events"][0]
            data = caches[event_filename(event)]
            self.assertTrue(event["primary_eligible"])
            self.assertEqual(event["onset_gt_best_rank"], 2)
            self.assertEqual(data["iou"].shape[1], 41)
            self.assertEqual(data["iou"][0].max(), 0)
            self.assertEqual(data["iou"][1].min(), 1)
            np.testing.assert_array_equal(data["previous"][:, 1:], data["boxes"][:, :-1])
            np.testing.assert_array_equal(data["model_boxes"][1, 1:], data["boxes"][1, 1:])
            self.assertEqual(data["output_override"][1].sum(), 1)
            _, source, traces, _ = inputs["source_info"]
            entry, trace = traces[event["sequence"]]
            audit_event(data, event, entry, trace, source)
            rows, row = summarize_event(event, data, trace)
            self.assertTrue(row["event_gate_pass"])
            self.assertEqual(row["selected_post40_gain_vs_native"], 1)
            self.assertEqual(row["post_low_iou_frame_reduction"], 40)
            self.assertEqual(row["selected_post_recovery_offset"], 1)
            self.assertIsNone(row["selected_post_relapse_offset"])
            self.assertEqual(rows[0]["post40_gain_vs_native"], 0)

    def test_tampered_continuity_override_metadata_and_native_replay_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            inputs, plan, run, caches, _ = branch_fixture(root)
            event = plan["events"][0]
            data = caches[event_filename(event)]
            _, source, traces, _ = inputs["source_info"]
            entry, trace = traces[event["sequence"]]
            for key in ("previous", "boxes", "model_boxes", "factor", "iou", "coverage", "layer", "candidate_cell"):
                bad = {k: v.copy() for k, v in data.items()}
                bad[key].flat[0] += 1
                with self.assertRaises((ValueError, AssertionError)):
                    audit_event(bad, event, entry, trace, source)
            bad = {k: v.copy() for k, v in data.items()}
            bad["output_override"][1, 1] = True
            with self.assertRaises(AssertionError):
                audit_event(bad, event, entry, trace, source)
            path = root / "event.npz"
            write_cache(path, data, event_metadata(run, event))
            load_event(path, event, entry, trace, source, run)
            with self.assertRaises(ValueError):
                load_event(path, event, entry, trace, source, {**run, "fingerprint": "wrong"})

    def test_future_gt_labels_cannot_change_boxes_or_candidate_selection(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            inputs, plan, run, caches, reader = branch_fixture(Path(temp))
            event = plan["events"][0]
            _, source, traces, _ = inputs["source_info"]
            entry, trace = traces[event["sequence"]]
            changed_entry = copy.deepcopy(entry)
            gt = np.array(entry["ground_truth"], float)
            gt[event["onset_frame"] + 1:, 0] += 3
            changed_entry["ground_truth"] = gt.tolist()
            changed_trace = {k: v.copy() for k, v in trace.items()}
            changed_trace["gt"] = gt[1:]
            changed_trace["iou"] = tracking_iou(changed_trace["boxes"], gt[1:])
            with redirect_stdout(io.StringIO()):
                other = collect_event(BranchRuntime(), event, changed_entry, changed_trace,
                    inputs["caches"][entry["sequence"]], source, run, image_reader=reader)
            original = caches[event_filename(event)]
            for key in ("boxes", "previous", "model_boxes", "layer", "factor"):
                np.testing.assert_array_equal(other[key], original[key])
            self.assertFalse(np.array_equal(other["iou"], original["iou"]))


class GateTests(unittest.TestCase):
    def test_gain_boundary_and_both_conditions_required(self):
        self.assertTrue(event_pass(True, .05, 1))
        self.assertFalse(event_pass(True, .049, 40))
        self.assertFalse(event_pass(True, .1, 0))
        self.assertFalse(event_pass(False, .9, 40))
        self.assertFalse(event_pass(True, None, None))

    def test_current_frame_excluded_and_relapse_offsets(self):
        values = np.r_[0., np.ones(5), np.zeros(35)]
        metrics = post_metrics(values, np.ones(41))
        values[0] = 1
        self.assertEqual(metrics, post_metrics(values, np.ones(41)))
        self.assertEqual(metrics["post_recovery_offset"], 1)
        self.assertEqual(metrics["post_relapse_offset"], 6)
        self.assertEqual(metrics["post_low_iou_frames"], 35)

    def test_fixed_two_of_three_gate_and_incomplete_never_passes(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            inputs, plan, run, caches, _ = branch_fixture(Path(temp), all_events=True)
            traces = inputs["source_info"][2]
            rows = [summarize_event(e, caches[event_filename(e)], traces[e["sequence"]][1])[1] for e in plan["events"]]
            decision = make_decision(plan, rows, [], run, {})
            self.assertEqual(decision["status"], "PASS_CANDIDATE_PERSISTENCE_PILOT")
            self.assertEqual(decision["primary_success_events"], 3)
            partial = make_decision(plan, rows[:2], [dict(sequence=rows[2]["sequence"], event=1)], run, {})
            self.assertEqual(partial["status"], "PILOT_INCOMPLETE")
            self.assertFalse(partial["persistence_gate_pass"])
            failed = copy.deepcopy(rows)
            for row in failed[1:]:
                row["selected_post40_gain_vs_native"] = .01
                row["event_gate_pass"] = False
            self.assertEqual(make_decision(plan, failed, [], run, {})["status"], "FAIL_CANDIDATE_PERSISTENCE_PILOT")


class WorkflowTests(unittest.TestCase):
    def test_complete_stage28_required_and_frozen_plan_rejects_changes(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            inputs, _ = input_fixture(Path(temp))
            plan = make_plan(inputs)
            self.assertEqual(sum(e["primary_eligible"] for e in plan["events"]), 3)
            self.assertEqual(plan["future_model_forwards"], plan["branches"] * 40)
            changed = copy.deepcopy(plan)
            changed["events"][0]["onset_gt_best_rank"] = 1
            with self.assertRaises(ValueError):
                validate_stage29_plan(changed, inputs)
            short = copy.deepcopy(inputs)
            first = plan["events"][0]["sequence"]
            short["source_info"][2][first][0]["frames"] = short["source_info"][2][first][0]["frames"][:40]
            with self.assertRaisesRegex(ValueError, "Full H40 unavailable"):
                make_plan(short)
            report_path = inputs["directory"] / "analysis" / "stage28_response_diagnostic.json"
            bad_report = {**inputs["report"], "status": "PILOT_INCOMPLETE"}
            bad_report.pop("fingerprint")
            write_json(report_path, seal(bad_report))
            with self.assertRaises(ValueError):
                load_inputs(inputs["directory"])

    def test_cpu_plan_cli_and_identical_resume(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            inputs, _ = input_fixture(root)
            output = root / "stage29"
            cmd = [sys.executable, "-B", str(HERE / "stage29_candidate_collect.py"),
                   "--input", str(inputs["directory"]), "--output_dir", str(output), "--plan_only"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("CPU plan only", result.stdout)
            self.assertIn("3 primary events", result.stdout)
            before = (output / "plan.json").read_bytes()
            self.assertEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            self.assertEqual(before, (output / "plan.json").read_bytes())
            self.assertFalse((output / "run.json").exists())

    def test_full_summary_cli_and_source_immutability(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            inputs, plan, run, caches, _ = branch_fixture(root, all_events=True)
            sources = [p for d in (inputs["directory"], inputs["source_info"][0]) for p in d.rglob("*.npz")]
            before = {str(p): file_hash(p) for p in sources}
            output = root / "stage29"
            write_json(output / "plan.json", plan)
            write_json(output / "run.json", run)
            for event in plan["events"]:
                write_cache(output / "branches" / event_filename(event), caches[event_filename(event)], event_metadata(run, event))
            cmd = [sys.executable, "-B", str(HERE / "stage29_candidate_summary.py"), "--input", str(output)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            decision = json.loads((output / "analysis" / "stage29_decision.json").read_text())
            self.assertEqual(decision["status"], "PASS_CANDIDATE_PERSISTENCE_PILOT")
            self.assertFalse(decision["learned_policy_evaluated"])
            self.assertTrue((output / "analysis" / "stage29_step_effects.csv").is_file())
            self.assertEqual(before, {str(p): file_hash(p) for p in sources})
            (output / "branches" / event_filename(plan["events"][0])).unlink()
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            partial = subprocess.run(cmd + ["--allow_incomplete"], capture_output=True, text=True)
            self.assertEqual(partial.returncode, 0, partial.stderr)
            partial_decision = json.loads((output / "analysis_pilot" / "stage29_decision.json").read_text())
            self.assertEqual(partial_decision["status"], "PILOT_INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
