"""Verify matched past state, source-only anchors and budgeted replay caching."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage101_trigger_replay as stage
import test_stage100_online_scale as previous


def choose(row, history):
    return dict(row, selected=1, history_latest=history[-1], boxes=row["candidate_boxes"][1].copy())


class TriggerReplay(unittest.TestCase):
    setUp = previous.OnlineScale.setUp

    def sources(self):
        data = {mode: stage.s100.collect(self.observer, self.entry, self.ap, choose, mode,
                stage.DESIGN, image_reader=self.reader) for mode in ("gated", "pulse")}
        self.assertTrue(data["pulse"]["scale_active"].any())
        return data

    def test_source_only_anchor_selection_merges_roles_and_extends_lost_recovery(self):
        initial = np.array([80., 80., 20., 20.])
        entry = dict(self.entry, frames=[{}]*21, ground_truth=np.tile(initial, (21, 1)))
        pulse = dict(frame_id=np.arange(1, 21), boxes=np.tile(initial, (20, 1)),
                     scale_active=np.isin(np.arange(1, 21), [3, 7, 12]))
        gated = dict(boxes=pulse["boxes"].copy())
        pulse["boxes"][9:14, :2] += 200
        gated["boxes"][14, :2] += 200
        spec = dict(unit=dict(sequence="test", seed=2171))
        points, anchors = stage.select_anchors(spec, pulse, gated, entry, {("test", 2171, 17, 19)}, stage.DESIGN)
        self.assertEqual(points, [3, 7, 12])
        self.assertEqual([a["frame"] for a in anchors], [3, 7, 12])
        self.assertEqual(anchors[-1]["last"], 20)
        self.assertEqual({r["role"] for r in anchors[-1]["roles"]}, {"first_observed_gain", "newly_lost_ppo_recovery"})

    def test_matched_first_frame_own_feedback_future_labels_and_invalid_receipt(self):
        sources = self.sources()
        trace = sources["pulse"]
        frame = int(np.flatnonzero(trace["scale_active"])[0])+1
        last = min(12, frame+5)
        keep = stage.source_slice(trace, frame, last)
        skip = stage.collect(self.observer, trace, self.entry, frame, last, "skip", self.old, choose,
                             stage.DESIGN, image_reader=self.reader)
        for mode, data in (("keep", keep), ("skip", skip)):
            stage.audit(data, trace, self.entry, frame, last, mode, self.old, choose, stage.DESIGN)
        np.testing.assert_array_equal(keep["candidate_boxes"][0, [0, 2]], skip["candidate_boxes"][0, [0, 2]])
        self.assertTrue(keep["scale_active"][0])
        self.assertFalse(skip["scale_active"][0])
        np.testing.assert_array_equal(skip["history_latest"][1], skip["boxes"][0])
        shared = {k: v[:1].copy() for k, v in skip.items() if k != "anchor"} | dict(anchor=skip["anchor"])
        changed = dict(self.entry, ground_truth=np.full((13, 4), 999.))
        repeated = stage.collect(self.observer, trace, changed, frame, last, "skip", self.old, choose,
                                 stage.DESIGN, shared=shared, image_reader=self.reader)
        for key in stage.s100.FIELDS:
            np.testing.assert_array_equal(skip[key], repeated[key])
        self.assertIsNone(self.observer.embed.hook)
        skip["scale_refs"][0, 0] = frame
        with self.assertRaises(AssertionError):
            stage.audit(skip, trace, self.entry, frame, last, "skip", self.old, choose, stage.DESIGN)

    def test_full_matched_pipeline_both_cards_shared_first_budget_resume_and_tamper(self):
        sources = self.sources()
        unit = dict(sequence="test", seed=2171, origin="native", policy="ppo", fold=0)
        specs = [dict(unit=unit, cohort="historical"),
                 dict(unit=dict(unit, sequence="other", seed=2172), cohort="expanded")]
        source = dict(original=self.old, report=dict(native_head_sha256="native"))
        actor = dict(models={}, prior={})
        with (patch.object(stage, "load_unit", side_effect=lambda *a: (dict(self.entry, sequence=a[-1]["unit"]["sequence"]), sources)),
              patch.object(stage.e74, "learned_pack"),
              patch.object(stage.e74, "choose", side_effect=lambda r,h,*_: choose(r,h))):
            units = stage.prepare_units(Path('.'), dict(units=specs, design=stage.s100.DESIGN), Path('.'), Path('.'),
                                        {}, {}, {}, source, actor, set())
            for reuse in (True, False):
                with self.subTest(reuse=reuse), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    plan = stage.io.seal(dict(schema=stage.SCHEMA, design=stage.DESIGN, units=units,
                        runtime={}, runtime_differences=[], reuse_keep=reuse, source_sha256={}, code_sha256={}))
                    stage.c.publish_identical(root / "manifest.json", plan)
                    real_collect = stage.collect
                    args = (root, root, dict(design=stage.s100.DESIGN), root, root, {}, {}, {}, source, actor, self.observer, plan)
                    calls_before = self.observer.calls
                    with (patch.object(stage.e63.source_experiment, "installed", side_effect=lambda *_: contextlib.nullcontext("native")),
                          patch.object(stage, "collect", side_effect=lambda *a: real_collect(*a, image_reader=self.reader)),
                          contextlib.redirect_stdout(io.StringIO())):
                        stage.run(*args)
                        stage.check_results(root)
                        self.assertEqual(self.observer.calls-calls_before, stage.budget(plan)[2])
                        with patch.object(self.observer, "capture", side_effect=AssertionError("Repeated inference")):
                            stage.run(*args)
                    report = stage.c.c66.sealed(root / "analysis/stage101_decision.json", stage.SCHEMA)
                    self.assertEqual(report["maximum_keep_source_box_error_px"], 0.)
                    self.assertEqual(report["new_search_calls"], stage.budget(plan)[2])
                    path = next((root / "points").glob("*.npz"))
                    path.write_bytes(b"tampered")
                    with self.assertRaises(ValueError):
                        stage.check_results(root)

    def test_video_aggregation_averages_units_before_videos_and_handles_invalid_gt(self):
        triggers = []
        for sequence, seed, count, gain in (("a", 1, 3, 1.), ("a", 2, 1, 0.), ("b", 1, 1, 0.)):
            triggers.extend(dict(cohort="historical", sequence=sequence, seed=seed, keep_best_iou=0.,
                skip_best_iou=gain, skip_minus_keep_iou=gain) for _ in range(count))
        summary, videos, branches = stage.aggregate(triggers, [])
        self.assertAlmostEqual(summary[0]["mean_iou_gain_video"], .25)
        self.assertEqual(branches, [])
        sources = self.sources()
        invalid = dict(self.entry, ground_truth=np.zeros((13, 4)))
        metrics = stage.window_metrics(sources["pulse"], invalid, stage.DESIGN)
        self.assertIsNone(metrics["mean_iou"])
        self.assertEqual(metrics["failure_events"], 0)


if __name__ == "__main__":
    unittest.main()
