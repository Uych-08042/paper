"""NumPy-only cached-audit tests, including native rounding and frozen-input checks."""
import ast
import contextlib
import io
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import stage46_motion_common as c
import stage46_motion_summary as summary
import stage46_shift_audit as a


def trace(method, boxes, frames, matrix=None):
    matrix = np.array([[1., 0., .2], [0., 1., .1]]) if matrix is None else matrix
    state, rows = c.MotionState([40, 40, 20, 20]), []
    for i, box in enumerate(boxes):
        proposal = state.propose(method, matrix, True, (160, 180))
        blocks = np.zeros(12, int)
        blocks[:6], blocks[8] = 1, 1
        rows.append(dict(frame_id=i + 1, previous=state.previous.copy(), parents=proposal["parent"],
            boxes=np.asarray(box, float), image_hw=[160, 180], layer=9, peak=.8,
            matrix=matrix.copy(), motion_valid=True, motion_reason=0,
            motion_stats=[100, 80, 70, .875, .2, 12],
            **{k: proposal[k] for k in ("raw_velocity", "relative_velocity", "proposal", "shift", "cap_hit", "camera_displacement")},
            motion_ms=2., network_ms=1., pipeline_ms=3., block_calls=blocks, head_calls=1, selector_calls=1))
        state.advance(box, matrix, True)
    data = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    for key in ("frame_id", "image_hw", "layer", "motion_reason", "block_calls", "head_calls", "selector_calls"):
        data[key] = data[key].astype(np.int64)
    c.audit_video(data, frames, [40, 40, 20, 20], method)
    return data


def entry(n=45):
    return dict(initial_box=[40, 40, 20, 20], ground_truth=[[40, 40, 20, 20]] * (n + 1),
        frames=[dict(file="unopened/img{:07d}.jpg".format(i + 1), height=160, width=180) for i in range(n + 1)])


class GeometryTests(unittest.TestCase):
    def test_actual_native_sampling_and_floating_mapback(self):
        root = Path(__file__).resolve().parents[1]
        sample_path = root / "lib/train/data/processing_utils.py"
        sample_tree = ast.parse(sample_path.read_text(encoding="utf-8"))
        func = next(x for x in sample_tree.body if isinstance(x, ast.FunctionDef) and x.name == "sample_target")
        # Execute the original sampler's crop/slice logic without importing Torch/OpenCV.
        cv = types.SimpleNamespace(BORDER_CONSTANT=0,
            copyMakeBorder=lambda image, top, bottom, left, right, mode:
                np.pad(image, ((top, bottom), (left, right), (0, 0))))
        ns = dict(np=np, math=math, cv=cv)
        exec(compile(ast.Module(body=[func], type_ignores=[]), str(sample_path), "exec"), ns)
        image = np.arange(160 * 180 * 3).reshape(160, 180, 3)
        native_path = root / "lib/test/tracker/sglatrack.py"
        tree = ast.parse(native_path.read_text(encoding="utf-8"))
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "sglatrack")
        func = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "map_box_back")
        namespace = {}
        exec(compile(ast.Module(body=[func], type_ignores=[]), str(native_path), "exec"), namespace)
        old, new = [40., 40., 20., 20.], [40.2, 40.1, 20., 20.]
        crop0 = a.crop_geometry(old, image.shape[:2], 4., 256)[0]
        crop1 = a.crop_geometry(new, image.shape[:2], 4., 256)[0]
        np.testing.assert_array_equal(crop0, crop1)
        p0, p1 = ns["sample_target"](image, old, 4.)[0], ns["sample_target"](image, new, 4.)[0]
        np.testing.assert_array_equal(p0, p1)
        maps = []
        for parent in (old, new):
            tracker = types.SimpleNamespace(state=parent, params=types.SimpleNamespace(search_size=256))
            maps.append(namespace["map_box_back"](tracker, [40., 40., 20., 20.], 256 / 80))
        np.testing.assert_allclose(np.asarray(maps[1]) - maps[0], [.2, .1, 0, 0], atol=1e-12)
        shifted = [40.6, 40., 20., 20.]
        self.assertEqual(a.crop_geometry(shifted, image.shape[:2], 4., 256)[0][0] - crop0[0], 1)
        self.assertFalse(np.array_equal(p0, ns["sample_target"](image, shifted, 4.)[0]))

    def test_transition_signed_accounting_nonzero_and_zero(self):
        e = entry(4)
        boxes = [[40, 40, 20, 20], [43, 41, 22, 18], [100, 60, 18, 24], [100, 60, 18, 24]]
        data = trace("camera_velocity", boxes, e["frames"])
        rows = a.frame_rows(e, data, data, 4., 256)
        result = a.transition(rows[1], rows[2])
        self.assertLess(result["accounting_residual_px"], 1e-10)
        self.assertAlmostEqual(sum(result[t + "_along_error_gt_units"] for t in a.TERMS),
                               result["error_vector_change_gt_units"])
        zero = a.transition(rows[2], rows[2])
        self.assertEqual(zero["error_vector_change_gt_units"], 0.)
        self.assertTrue(all(zero[t + "_along_error_gt_units"] is None for t in a.TERMS))
        constant = trace("camera_velocity", [[40, 40, 20, 20]] * 4, e["frames"])
        stationary = a.frame_rows(e, constant, constant, 4., 256)
        self.assertLess(stationary[-1]["camera_relative_cosine"], -.999999)
        self.assertLess(stationary[-1]["camera_plus_relative_target_sides"], stationary[-1]["camera_target_sides"])

    def test_failure_indices_window_clipping_and_cross_arm_context(self):
        e = entry(9)
        boxes = [[100, 40, 20, 20]] * 3 + [[40, 40, 20, 20]] * 6
        native = trace("sgla", [[40, 40, 20, 20]] * 9, e["frames"])
        arrays = {m: trace(m, boxes if m == "camera_only" else [[40, 40, 20, 20]] * 9, e["frames"])
                  for m in c.METHODS}
        arrays["sgla"] = native
        result = a.analyze_case("fixture", e, arrays, 4., 256)
        event = result["onsets"][0]
        self.assertEqual((event["onset_frame"], event["confirmed_frame"], event["end_frame"], event["recovery_frame"]), (1, 3, 3, 4))
        own = next(r for r in result["case_summary"] if r["method"] == "camera_only")
        self.assertEqual(own["first_output_divergence"], 1)
        self.assertIsNone(own["first_long_failure_onset"])
        anchor = next(r for r in result["anchors"] if r["owner"] == "camera_only" and r["frame_id"] == 1)
        phases = [r for r in result["phase_summary"] if r["anchor_id"] == anchor["anchor_id"]]
        self.assertEqual({r["evaluator"] for r in phases}, set(c.METHODS))
        self.assertNotIn("before", {r["phase"] for r in phases})
        self.assertTrue(all(1 <= r["frame_id"] <= 9 for r in result["timeline"]))
        self.assertEqual(len(result["timeline"]), len({(r["method"], r["frame_id"]) for r in result["timeline"]}))
        self.assertFalse(any("peak_cell" in r or "top5" in r for r in result["timeline"]))


class FrozenWorkflowTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "source"
        source.mkdir()
        c.write_json(source / "run.json", dict(fixture=True))
        names = list(a.CONFIG["cases"]) + ["uav{:07d}_00000_s".format(i) for i in range(100, 109)]
        traces, arrays, source_hashes = {}, {}, {}
        for name in names:
            e = entry()
            e.update(sequence=name, source_group_proxy=name.split("_")[0])
            for method in c.METHODS:
                boxes = [[40, 40, 20, 20]] * 45
                fail_at = None
                if name == names[0] and method in ("sgla", "camera_velocity"):
                    fail_at = 15
                if name == names[1] and method != "camera_only":
                    fail_at = 20
                if name == names[2] and method != "sgla":
                    fail_at = 15 if method != "camera_velocity" else 25
                if fail_at:
                    boxes[fail_at - 1:] = [[100, 40, 20, 20]] * (46 - fail_at)
                arrays[name, method] = trace(method, boxes, e["frames"])
            native = arrays[name, "sgla"]
            traces[name] = e, dict(boxes=native["boxes"], layer=native["layer"], raw_score_peak=native["peak"])
            path = source / "traces" / (name + "__stage27_uav.npz")
            c.write_cache(path, traces[name][1], dict(fixture=True))
            source_hashes[path.name] = c.file_hash(path)
        source_run = dict(fingerprint="fixture-source", search_factor=4., search_size=256)
        inputs = (source, source_run, traces, source_hashes)
        output = root / "stage46"
        output.mkdir()
        with patch.object(c, "load_source", return_value=inputs):
            plan, _ = c.make_plan(source)
            c.write_json(output / "manifest.json", plan)
            run = c.seal(dict(schema=c.SCHEMA, role="frozen_native_sgla_four_policy_motion_run",
                plan_fingerprint=plan["fingerprint"], runtime=dict(fixture=True), methods=list(c.METHODS)))
            c.write_json(output / "run.json", run)
            for (name, method), data in arrays.items():
                c.save_result(c.cache_path(output, name, method), data, c.metadata(run, name, method, False))
            with contextlib.redirect_stdout(io.StringIO()):
                summary.summarize(output)
        return output, inputs

    def test_complete_reuse_no_image_calls_and_table_tamper_rejection(self):
        with tempfile.TemporaryDirectory(prefix="stage46b-test-", dir=Path(__file__).parent) as folder:
            root = Path(folder)
            output, inputs = self.fixture(root)
            old_bytes = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            with patch.object(c, "load_source", return_value=inputs), \
                 patch.object(c, "checked_image", side_effect=AssertionError("Images must not be opened")), \
                 contextlib.redirect_stdout(io.StringIO()):
                report = a.audit(output)
                self.assertEqual(report["status"], "COMPLETE_CACHED_SEARCH_SHIFT_AUDIT")
                self.assertEqual(report["source_trajectories_audited"], 48)
                self.assertEqual(report["focus_trajectories"], 12)
                self.assertFalse(report["new_inference_performed"])
                self.assertLess(report["maximum_vector_accounting_residual_px"], 1e-8)
                self.assertEqual(report, a.audit(output))
                for name, payload in old_bytes.items():
                    self.assertEqual(Path(name).read_bytes(), payload)
                table = output / "analysis/search_shift_audit_v1/stage46b_onsets.csv"
                table.write_bytes(table.read_bytes() + b"tampered")
                with self.assertRaisesRegex(ValueError, "input/output changed"):
                    a.audit(output)

    def test_incomplete_decision_mixing_and_committed_trace_tamper(self):
        with tempfile.TemporaryDirectory(prefix="stage46b-test-", dir=Path(__file__).parent) as folder:
            output, inputs = self.fixture(Path(folder))
            with patch.object(c, "load_source", return_value=inputs), contextlib.redirect_stdout(io.StringIO()):
                decision_path = output / "analysis/stage46_decision.json"
                saved = c.load_sealed(decision_path)
                changed = {k: v for k, v in saved.items() if k != "fingerprint"}
                changed["run_fingerprint"] = "mixed"
                c.write_json(decision_path, c.seal(changed))
                with self.assertRaisesRegex(ValueError, "complete, unmixed"):
                    a.audit(output)
                c.write_json(decision_path, saved)
                path = c.cache_path(output, a.CONFIG["cases"][0], "camera_only")
                with path.open("ab") as handle:
                    handle.write(b"tampered")
                with self.assertRaisesRegex(ValueError, "trace changed"):
                    a.audit(output)

    def test_source_changes_and_unsafe_output_rejected(self):
        with tempfile.TemporaryDirectory(prefix="stage46b-test-", dir=Path(__file__).parent) as folder:
            output, inputs = self.fixture(Path(folder))
            with self.assertRaisesRegex(ValueError, "dedicated"):
                a.audit(output, output / "traces")
            with patch.object(c, "load_source", return_value=inputs), contextlib.redirect_stdout(io.StringIO()):
                source_path = inputs[0] / "run.json"
                source_bytes = source_path.read_bytes()
                c.write_json(source_path, dict(fixture=False))
                with self.assertRaisesRegex(ValueError, "source/code/design changed"):
                    a.audit(output)
                source_path.write_bytes(source_bytes)
                # A receipt can be internally valid yet inconsistent with the completed decision.
                run = c.load_sealed(output / "run.json")
                name, method = a.CONFIG["cases"][0], "camera_only"
                path = c.cache_path(output, name, method)
                data = c.load_result(path, c.metadata(run, name, method, False))
                data["peak"][0] += .01
                path.with_suffix(".receipt.json").unlink()
                c.save_result(path, data, c.metadata(run, name, method, False))
                with self.assertRaisesRegex(ValueError, "decision/cache mismatch"):
                    a.audit(output)
            self.assertFalse((output / "analysis/search_shift_audit_v1").exists())


if __name__ == "__main__":
    unittest.main()
