"""CPU-only tests; synthetic timing fixtures are never hardware evidence."""

import contextlib
import io
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from stage4_common import file_hash, fingerprint, write_csv, write_json
from stage22_device_calibrate import parse_args, profile_clip
from stage22_device_summary import audit_files, calibration_decision, summarize_run
from stage22_gpu_hardware import METRICS, LoadProcess, TelemetrySampler, load_worker, metric_number, resolve_gpu
from stage22_resource_common import (
    CONDITIONS, OPERATIONS, SCHEMA, calibration_sources, check_execution_counts, interleaved_order, load_phase,
    predict_box, restore_call, run_operation, sealed, summarize_samples,
)


def synthetic_run():
    return sealed(dict(schema=SCHEMA, conditions=list(CONDITIONS),
        config=dict(mode="smoke", frames=8, videos=1, repeats=1, warmup=1, deadline_ms=1000 / 30,
                    seed=2070, sustained_duty=.6, switching_duty=.85, switch_hold_s=.1,
                    maximum_idle_kernel_ms=5),
        hardware=dict(uuid="GPU-test"), sources=[dict(sequence="fit_video", start=0)]))


def synthetic_records(run):
    rows = []
    for condition_index, condition in enumerate(run["conditions"]):
        origin = (condition_index + 1) * 10**10
        ordinal = 0
        for frame in range(1, 9):
            for index, op in enumerate(interleaved_order(2070, condition, "fit_video", frame, 0)):
                start = origin + 40_000_000 * ordinal + 10_000_000
                elapsed = (start - origin) / 1e9
                phase, duty = load_phase(condition, elapsed, run["config"])
                rows.append(dict(run_fingerprint=run["fingerprint"], condition=condition,
                    sequence="fit_video", frame_id=frame, repeat=0, order=index, operation=op,
                    crop_state_fingerprint=fingerprint(frame), start_ns=start, end_ns=start + 1_000_000,
                    latency_ms=1., condition_elapsed_s=elapsed, phase=phase, nominal_duty=duty,
                    timing_basis="cpu_wall" if op == "predict_only" else "cuda_synchronized_host_wall",
                    telemetry_sample_id=0, telemetry_end_ns=1, telemetry_age_ms=(start - 1) / 1e6,
                    telemetry_lookup_ms=.001))
                ordinal += 1
    return rows


def csv_rows(rows):
    return [{k: str(v) for k, v in row.items()} for row in rows]


def write_fixture(directory):
    run = synthetic_run()
    records = synthetic_records(run)
    write_json(directory / "run.json", run)
    write_csv(directory / "latency_raw.csv", records)
    sensor = dict(sample_id=0, start_ns=0, end_ns=1, gpu_uuid="GPU-test")
    sensor.update({key: 50. for key in METRICS})
    sensor["error"] = ""
    write_csv(directory / "telemetry.csv", [sensor])
    timelines = []
    files = ["latency_raw.csv", "telemetry.csv", "conditions.json"]
    for index, condition in enumerate(CONDITIONS):
        origin = (index + 1) * 10**10
        worker = None if condition == "idle" else dict(status="ready", gpu_uuid="GPU-test", idle_max_kernel_ms=1.)
        checks = {"predict_only": {}}
        prefix = {"l%d" % n: 1 for n in range(1, 7)}
        checks["fast"] = dict(prefix, head=1)
        checks["sgla"] = dict(prefix, head=1, l9=1)
        for layer in range(7, 13):
            checks["l%d" % layer] = dict(prefix, head=1, **{"l%d" % layer: 1})
        checks["native_scripted_bbox_max_abs_diff"] = 0.
        timelines.append(dict(condition=condition, origin_ns=origin, end_ns=origin + 4_000_000_000,
                              worker=worker, execution_checks={"fit_video": checks}))
        if worker:
            phase, duty = load_phase(condition, .11, run["config"])
            name = "load_%s.csv" % condition
            write_csv(directory / name, [dict(start_ns=origin + 110_000_000, end_ns=origin + 130_000_000,
                elapsed_s=.11, phase=phase, nominal_duty=duty, kernels=2, busy_ms=10., maximum_kernel_ms=1.)])
            files.append(name)
    write_json(directory / "conditions.json", timelines)
    completion = sealed(dict(schema=SCHEMA + "_completion", run_fingerprint=run["fingerprint"],
                             samples=len(records), files={name: file_hash(directory / name) for name in files}))
    write_json(directory / "completion.json", completion)
    return run, records


class MathTests(unittest.TestCase):
    def test_extra_head_or_sequential_candidates_rejected(self):
        counts = {"l%d" % n: 1 for n in range(1, 7)}
        counts.update(head=1, l12=1)
        check_execution_counts("l12", counts)
        counts["l7"] = 1
        with self.assertRaises(ValueError):
            check_execution_counts("l12", counts)
        counts.pop("l7")
        counts["head"] = 2
        with self.assertRaises(ValueError):
            check_execution_counts("l12", counts)

    def test_load_schedule_is_reproducible(self):
        config = synthetic_run()["config"]
        self.assertEqual(load_phase("idle", 100, config), (0, 0.))
        self.assertEqual(load_phase("sustained", 100, config), (0, .6))
        self.assertEqual(load_phase("switching", .11, config), (1, .85))
        self.assertEqual(load_phase("switching", .21, config), (2, 0.))
        with self.assertRaises(ValueError):
            load_phase("switching", -1, config)

    def test_interleaving_has_every_path_once(self):
        order = interleaved_order(2070, "idle", "x", 2, 1)
        self.assertCountEqual(order, OPERATIONS)
        self.assertEqual(order, interleaved_order(2070, "idle", "x", 2, 1))
        self.assertNotEqual(order, interleaved_order(2070, "idle", "x", 2, 2))

    def test_causal_motion_and_image_clipping(self):
        history = [(0., [0, 1, 10, 10]), (1., [2, 3, 10, 10])]
        self.assertEqual(predict_box(history, 2., (20, 20)), [4., 5., 10., 10.])
        self.assertEqual(predict_box(history, 100., (20, 20)), [10., 10., 10., 10.])
        self.assertEqual(history[-1][1], [2, 3, 10, 10])
        with self.assertRaises(ValueError):
            predict_box(history, .5, (20, 20))
        with self.assertRaises(ValueError):
            predict_box([(1., [0, 0, 10, 10])] * 2, 2., (20, 20))

    def test_missing_hardware_values_are_not_zero(self):
        self.assertEqual(metric_number("[N/A]"), "")
        self.assertEqual(metric_number("nan"), "")
        self.assertEqual(metric_number("0"), 0.)

    def test_predict_and_independent_candidate_execution(self):
        calls = []
        tracker = SimpleNamespace(state=[0, 0, 10, 10], frame_id=4)
        rt = SimpleNamespace(tracker=tracker,
            prepare=lambda image: (calls.append("prepare") or "search", 2),
            prefix=lambda search: calls.append("prefix") or "tokens",
            head=lambda tokens: calls.append("head") or "output",
            candidate=lambda tokens, layer: calls.append(layer) or "output",
            decode=lambda output, factor, image: [1, 2, 10, 10])
        image = np.zeros((20, 20, 3))
        run_operation(rt, image, "predict_only", [(0., tracker.state)], .1)
        self.assertEqual(calls, [])
        run_operation(rt, image, "l12", [], .1)
        self.assertEqual(calls, ["prepare", "prefix", 12])
        calls.clear()
        run_operation(rt, image, "fast", [], .1)
        self.assertEqual(calls, ["prepare", "prefix", "head"])

    def test_crop_restored_even_on_error(self):
        tracker = SimpleNamespace(state=[1, 2, 3, 4], frame_id=9)
        rt = SimpleNamespace(tracker=tracker)
        def fail():
            tracker.state[:] = [99] * 4
            tracker.frame_id = 10
            raise RuntimeError("injected error")
        with self.assertRaises(RuntimeError):
            restore_call(rt, fail)
        self.assertEqual(tracker.state, [1, 2, 3, 4])
        self.assertEqual(tracker.frame_id, 9)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.run = synthetic_run()
        self.rows = csv_rows(synthetic_records(self.run))

    def test_complete_grid(self):
        report = summarize_samples(self.run, self.rows)
        self.assertEqual(len(report), len(CONDITIONS) * len(OPERATIONS))
        self.assertTrue(all(r["samples"] == 8 and r["p95_ms"] == 1. for r in report))

    def test_duplicate_or_missing_sample_rejected(self):
        for rows in (self.rows[:-1], self.rows + self.rows[:1]):
            with self.assertRaises(ValueError):
                summarize_samples(self.run, rows)

    def test_future_telemetry_rejected(self):
        row = self.rows[0]
        row["telemetry_end_ns"] = str(int(row["start_ns"]) + 1)
        with self.assertRaisesRegex(ValueError, "Future telemetry"):
            summarize_samples(self.run, self.rows)

    def test_future_or_inconsistent_crop_rejected(self):
        self.rows[0]["crop_state_fingerprint"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "same reference crop"):
            summarize_samples(self.run, self.rows)

    def test_mixed_fingerprint_rejected(self):
        self.rows[0]["run_fingerprint"] = "other"
        with self.assertRaises(ValueError):
            summarize_samples(self.run, self.rows)

    def test_negative_or_fabricated_latency_rejected(self):
        self.rows[0]["latency_ms"] = "0.5"
        with self.assertRaisesRegex(ValueError, "monotonic latency"):
            summarize_samples(self.run, self.rows)

    def test_load_phase_mismatch_rejected(self):
        self.rows[0]["nominal_duty"] = "0.5"
        with self.assertRaisesRegex(ValueError, "phase mismatch"):
            summarize_samples(self.run, self.rows)

    def test_source_selection_excludes_tune(self):
        bundle = dict(source_manifest_fingerprint="x", pilot_videos=["tune"],
                      fit_videos=["fit"], tune_videos=["tune"])
        with self.assertRaisesRegex(ValueError, "fit-pilot"):
            calibration_sources(dict(fingerprint="x"), bundle, 8, 1)

    def test_empty_environment_paths_and_no_permission_fail_before_runtime(self):
        base = ["--bundle", "bundle.json", "--manifest", "manifest.json", "--checkpoint", "model.pt",
                "--output_dir", "new-output"]
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(base)
            with self.assertRaises(SystemExit):
                parse_args(base + ["--exclusive_gpu_confirmed", "--manifest", ""])
        args = parse_args(base + ["--exclusive_gpu_confirmed"])
        self.assertEqual(args.mode, "smoke")


class HardwareTests(unittest.TestCase):
    def test_logical_index_not_assumed_physical_index(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda gpu:
            SimpleNamespace(total_memory=123, uuid="GPU-correct")))
        def query(kind, fields, gpu_uuid=None):
            if kind == "compute-apps":
                return []
            self.assertEqual(gpu_uuid, "GPU-correct")
            return [dict(uuid="GPU-correct", name="test", driver_version="x",
                         **{"memory.total": "123", "mig.mode.current": "Disabled"})]
        with patch("stage22_gpu_hardware.smi_query", side_effect=query):
            result = resolve_gpu(torch, 0)
        self.assertEqual(result["uuid"], "GPU-correct")
        self.assertEqual(result["logical_cuda_device"], 0)

    def test_ambiguous_physical_gpu_rejected(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda gpu: SimpleNamespace()))
        with patch("stage22_gpu_hardware.smi_query", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "safely map"):
                resolve_gpu(torch, 0)

    def test_latest_telemetry_excludes_unfinished_and_error_samples(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            sampler = TelemetrySampler("GPU-test", Path(tmp) / "telemetry.csv")
            try:
                with patch("stage22_gpu_hardware.smi_query", return_value=[{k: "50" for k in METRICS}]):
                    sampler.sample()
                end = sampler.rows[0]["end_ns"]
                self.assertIsNone(sampler.latest(end - 1))
                self.assertEqual(sampler.latest(end)["sample_id"], 0)
                with patch("stage22_gpu_hardware.smi_query", side_effect=RuntimeError("unsupported")):
                    sampler.sample()
                self.assertEqual(sampler.latest(sampler.rows[-1]["end_ns"])["sample_id"], 0)
            finally:
                sampler.close()

    def test_dead_parent_worker_never_imports_or_allocates_cuda(self):
        connection = SimpleNamespace(close=unittest.mock.Mock())
        with patch("stage22_gpu_hardware.mp.parent_process", return_value=SimpleNamespace(is_alive=lambda: False)):
            load_worker(connection, None, {}, "sustained", 0, "GPU-test", "unused.csv")
        connection.close.assert_called_once()

    def test_worker_cleanup_joins_even_after_pipe_eof(self):
        worker = LoadProcess.__new__(LoadProcess)
        worker.stopped = False
        worker.stop_event = SimpleNamespace(set=unittest.mock.Mock())
        worker.parent = SimpleNamespace(poll=lambda timeout: True,
            recv=unittest.mock.Mock(side_effect=EOFError), close=unittest.mock.Mock())
        worker.child = SimpleNamespace(close=unittest.mock.Mock())
        worker.process = SimpleNamespace(pid=123, join=unittest.mock.Mock(), is_alive=lambda: False)
        with self.assertRaisesRegex(RuntimeError, "exit cleanly"):
            worker.close()
        worker.process.join.assert_called_once()
        worker.stop_event.set.assert_called_once()
        worker.parent.close.assert_called_once()


class ProfilerTests(unittest.TestCase):
    def test_real_loop_restores_common_crop_and_reads_telemetry_before_action(self):
        run = synthetic_run()
        tracker = SimpleNamespace(state=[1., 1., 10., 10.], frame_id=0)
        def track(image):
            tracker.state = [tracker.state[0] + .1, 1., 10., 10.]
            tracker.frame_id += 1
            return dict(target_bbox=list(tracker.state))
        tracker.track = track
        rt = SimpleNamespace(tracker=tracker,
            torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)),
            read=lambda path: np.zeros((24, 24, 3)), initialize=lambda image, box: None,
            prepare=lambda image: ("search", 1.), prefix=lambda search: "tokens",
            head=lambda tokens: "head", candidate=lambda tokens, layer: "head",
            decode=lambda output, factor, image: [1., 1., 10., 10.])
        source = dict(sequence="fit_video", start=0, init_box=list(tracker.state),
                      images=[dict(path="unused.jpg", sha256="a")] * 9)
        records, lookup_times = [], []
        def latest(before):
            lookup_times.append(before)
            return None
        sampler = SimpleNamespace(latest=latest)
        journal = SimpleNamespace(append=records.append, flush=lambda: None)
        with patch("stage22_device_calibrate.file_hash", return_value="a"), \
                patch("stage22_device_calibrate.verify_paths", return_value={}), \
                contextlib.redirect_stdout(io.StringIO()):
            profile_clip(rt, source, run["config"], "idle", time.monotonic_ns(),
                         sampler, None, journal, run["fingerprint"])
        self.assertEqual(len(records), 8 * len(OPERATIONS))
        self.assertEqual(tracker.frame_id, 8)
        self.assertAlmostEqual(tracker.state[0], 1.8)
        self.assertTrue(all(before <= row["start_ns"] for before, row in zip(lookup_times, records)))
        for frame in range(1, 9):
            self.assertEqual(len({r["crop_state_fingerprint"] for r in records if r["frame_id"] == frame}), 1)


class ReportTests(unittest.TestCase):
    def test_complete_report_and_tamper_detection(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            directory = Path(tmp)
            write_fixture(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                decision = summarize_run(directory)
            self.assertEqual(decision["status"], "PILOT_CALIBRATION_ONLY")
            self.assertFalse(decision["environment_ready_for_next_design_step"])
            self.assertTrue((directory / "latency_summary.csv").is_file())
            with (directory / "latency_raw.csv").open("a") as handle:
                handle.write("tampered\n")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                audit_files(directory)

    def test_interrupted_run_is_not_complete(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            directory = Path(tmp)
            write_fixture(directory)
            write_json(directory / "interrupted.json", dict(error="injected"))
            with self.assertRaisesRegex(ValueError, "Interrupted"):
                audit_files(directory)

    def test_fast_gpu_does_not_create_scheduling_motivation(self):
        run = synthetic_run()
        run["config"]["mode"] = "full"
        reports = [dict(condition=c, operation="sgla", deadline_miss_ratio=0.) for c in CONDITIONS]
        rows = []
        for op in OPERATIONS:
            for i in range(20):
                rows.append(dict(condition="switching", operation=op, nominal_duty=.85 if i % 2 else 0.,
                                 telemetry_sample_id=0, telemetry_age_ms=1))
        telemetry = [dict(sample_id=0, **{"utilization.gpu": 30., "memory.free": 1000.})]
        result = calibration_decision(run, dict(fingerprint="fixture"), rows, reports, telemetry, [])
        self.assertEqual(result["resource_pressure"], "NO_MEANINGFUL_DEADLINE_PRESSURE_OBSERVED")
        self.assertFalse(result["environment_ready_for_next_design_step"])
        reports[-1]["deadline_miss_ratio"] = .2
        result = calibration_decision(run, dict(fingerprint="fixture"), rows, reports, telemetry, [])
        self.assertEqual(result["resource_pressure"], "LOAD_DEPENDENT_DEADLINE_PRESSURE_OBSERVED")
        self.assertTrue(result["environment_ready_for_next_design_step"])


if __name__ == "__main__":
    unittest.main()
