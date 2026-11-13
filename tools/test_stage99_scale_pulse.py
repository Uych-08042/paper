"""Verify matched causal state, a single intervention and resumable publication."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage99_scale_pulse as stage
import test_stage97_trusted_scale as fixture


class ScalePulse(unittest.TestCase):
    def setUp(self):
        self.old = dict(root="fixture", search_factor=5., search_size=100,
            window=np.ones((5, 5), np.float32), actor_design=dict(stage.e63.source_common.DESIGN))
        initial = np.array([80., 80., 20., 20.])
        self.entry = dict(initial_box=initial, frames=[{}]*11, ground_truth=np.tile(initial, (11, 1)))
        self.reader = lambda *_: np.zeros((200, 200, 3), np.uint8)
        self.observer = fixture.Observer()
        self.tokens = patch.object(stage.s93.s91, "cpu_array", side_effect=lambda x: x.copy())
        self.tokens.start()
        self.addCleanup(self.tokens.stop)
        empty = dict(boxes=np.empty((0, 4)), anchor=np.array([1., 0.]))
        self.trace = stage.s97.collect(self.observer, self.entry, empty, 1, 10, fixture.choose,
            "control", self.old, stage.s97.DESIGN, image_reader=self.reader)
        self.control = stage.s97.control_slice(self.trace, 3, 10)
        self.trusted = stage.s97.collect(self.observer, self.entry, self.trace, 3, 10, fixture.choose,
            "trusted_scale", self.old, stage.s97.DESIGN, image_reader=self.reader)
        self.first = 3
        self.snapshot = stage.matched_snapshot(self.trace, self.control, self.trusted, self.first)

    def collect(self, *args):
        return stage.collect(*args, image_reader=self.reader)

    def test_pulse_schedule_actual_geometry_and_future_label_isolation(self):
        branches = {}
        for mode in stage.ARMS:
            data = self.collect(self.observer, self.entry, self.snapshot, 3, 10, fixture.choose,
                                mode, self.old, stage.s97.DESIGN)
            stage.audit(data, self.snapshot, self.entry, 3, 10, fixture.choose, mode, self.old, stage.s97.DESIGN)
            branches[mode] = data
        for key in stage.s97.FIELDS:
            np.testing.assert_array_equal(branches["control"][key], self.control[key], err_msg=key)
            np.testing.assert_array_equal(branches["continuous"][key], self.trusted[key], err_msg=key)
            np.testing.assert_array_equal(branches["once"][key][0], branches["continuous"][key][0], err_msg=key)
        self.assertTrue(branches["once"]["scale_active"][0])
        self.assertTrue(branches["once"]["scale_changed"][0])
        self.assertFalse(branches["once"]["scale_active"][1:].any())
        self.assertFalse(branches["once"]["scale_changed"][1:].any())
        self.assertTrue((branches["once"]["scale_refs"][1:] == -1).all())
        self.assertTrue(branches["continuous"]["scale_active"][1:].any())
        changed = dict(self.entry, ground_truth=np.full((11, 4), 999.))
        repeat = self.collect(self.observer, changed, self.snapshot, 3, 10, fixture.choose,
                              "once", self.old, stage.s97.DESIGN)
        for key in stage.s97.FIELDS:
            np.testing.assert_array_equal(branches["once"][key], repeat[key], err_msg=key)
        self.assertIsNone(self.observer.embed.hook)
        with self.assertRaises(ValueError):
            stage.rule_at("once", 2, 3)

    def test_prefix_memory_mismatch_and_invalid_audit_are_rejected(self):
        # A common prefix can precede the intervention. Memory differences must be rejected
        # even if both histories still have identical output boxes.
        trusted = {k: v.copy() for k, v in self.control.items()}
        snapshot = stage.matched_snapshot(self.trace, self.control, trusted, 5)
        np.testing.assert_array_equal(snapshot["boxes"], self.trace["boxes"][:4])
        trusted["queries"][0, 0, 0] += .1
        with self.assertRaises(AssertionError):
            stage.matched_snapshot(self.trace, self.control, trusted, 5)
        with self.assertRaises(ValueError):
            stage.matched_snapshot(self.trace, self.control, self.trusted, 11)
        data = self.collect(self.observer, self.entry, self.snapshot, 3, 10, fixture.choose,
                            "once", self.old, stage.s97.DESIGN)
        data["scale_refs"][1, 0] = 4
        with self.assertRaises(AssertionError):
            stage.audit(data, self.snapshot, self.entry, 3, 10, fixture.choose, "once", self.old, stage.s97.DESIGN)

    def test_pulse_keeps_own_output_feedback_after_reverting_input_rule(self):
        def choose_history(row, history):
            return dict(row, selected=1, history_latest=history[-1], boxes=row["candidate_boxes"][1].copy())

        empty = dict(boxes=np.empty((0, 4)), anchor=np.array([1., 0.]))
        prefix = stage.s97.collect(self.observer, self.entry, empty, 1, 10, choose_history,
            "control", self.old, stage.s97.DESIGN, image_reader=self.reader)
        snapshot = {k: v.copy() for k, v in prefix.items()}
        branches = {mode: self.collect(self.observer, self.entry, snapshot, 3, 10, choose_history,
                    mode, self.old, stage.s97.DESIGN) for mode in ("control", "once")}
        for mode, data in branches.items():
            stage.audit(data, snapshot, self.entry, 3, 10, choose_history, mode, self.old, stage.s97.DESIGN)
        self.assertFalse(np.array_equal(branches["once"]["boxes"][0], branches["control"]["boxes"][0]))
        np.testing.assert_array_equal(branches["once"]["history_latest"][1], branches["once"]["boxes"][0])
        self.assertFalse(np.array_equal(branches["once"]["history_latest"][1], branches["control"]["history_latest"][1]))
        self.assertFalse(branches["once"]["scale_active"][1:].any())

    def test_complete_publication_both_gpu_modes_resume_and_tamper_rejection(self):
        event = dict(sequence="test", seed="2171", outcome="retained", first_audit_frame="3",
                     last_audit_frame="10", ppo_recovery_frame="5")
        unit = dict(sequence="test", seed=2171, origin="native", policy="ppo", fold=0)
        source = dict(original=self.old, report=dict(native_head_sha256="native"))
        actor = dict(models={}, prior={})
        original_collect = stage.collect
        collect = lambda *args: original_collect(*args, image_reader=self.reader)
        for reuse in (True, False):
            with self.subTest(reuse=reuse), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                root, root97 = base / "stage99", base / "stage97"
                root.mkdir()
                (root97 / "analysis").mkdir(parents=True)
                plan97 = stage.io.seal(dict(schema=stage.s97.SCHEMA))
                name = stage.s97.case_key(event)
                previous = dict(event)
                for arm, data in (("control", self.control), ("trusted_scale", self.trusted)):
                    meta = dict(plan_fingerprint=plan97["fingerprint"], unit=unit, case=name, arm=arm)
                    stage.c.save(root97 / "branches" / f"{name}__{arm}.npz", data, meta)
                    metrics, _ = stage.s97.describe(data, self.entry, event, stage.s97.DESIGN)
                    previous[arm+"_matched_good_frame"] = metrics["matched_good_frame"]
                stage.s86.publish_table(root97 / "analysis/stage97_windows.csv", [previous])
                plan = stage.io.seal(dict(schema=stage.SCHEMA, design=stage.s97.DESIGN,
                    cases=[dict(event=event, anchor=3, transition="retained")], reuse_sources=reuse,
                    source_sha256={}, code_sha256={}, runtime={}, runtime_differences=[]))
                stage.c.publish_identical(root / "manifest.json", plan)
                args = (root, root97, root, root, plan, plan97, {}, dict(units=[dict(unit=unit)]),
                        {}, source, actor, self.observer)
                with (patch.object(stage.s95, "load_unit", return_value=(self.entry, dict(gated=self.trace))),
                      patch.object(stage.e74, "learned_pack"),
                      patch.object(stage.e74, "choose", side_effect=lambda r,h,*_: fixture.choose(r,h)),
                      patch.object(stage.e63.source_experiment, "installed", side_effect=lambda *_: contextlib.nullcontext("native")),
                      patch.object(stage, "collect", side_effect=collect), contextlib.redirect_stdout(io.StringIO())):
                    stage.run(*args)
                    stage.check_results(root)
                report = stage.c.c66.sealed(root / "analysis/stage99_decision.json", stage.SCHEMA)
                self.assertEqual(report["branches"], 3)
                self.assertEqual(report["new_search_calls"], 32 if reuse else 96)
                self.assertEqual(report["maximum_source_tail_box_error_px"], 0.)
                with (patch.object(stage.s95, "load_unit", return_value=(self.entry, dict(gated=self.trace))),
                      patch.object(stage.e74, "learned_pack"),
                      patch.object(stage.e74, "choose", side_effect=lambda r,h,*_: fixture.choose(r,h)),
                      patch.object(stage.e63.source_experiment, "installed", side_effect=lambda *_: contextlib.nullcontext("native")),
                      patch.object(stage, "collect", side_effect=AssertionError("Repeated inference")),
                      contextlib.redirect_stdout(io.StringIO())):
                    stage.run(*args)
                cache = next((root / "branches").glob("*.npz"))
                cache.write_bytes(b"tampered")
                with self.assertRaises(ValueError):
                    stage.check_results(root)


if __name__ == "__main__":
    unittest.main()
