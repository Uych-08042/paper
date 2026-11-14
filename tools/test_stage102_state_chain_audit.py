"""Exercise cached dependency replay, local deletion effects and recovery accounting."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage102_state_chain_audit as stage
import test_stage101_trigger_replay as fixture


class StateChain(unittest.TestCase):
    setUp = fixture.TriggerReplay.setUp
    sources = fixture.TriggerReplay.sources

    def test_accepted_wrong_writes_references_and_label_isolation(self):
        sources = self.sources()
        unit = dict(sequence="test", seed=2171)
        wrong = dict(self.entry, ground_truth=np.tile([160., 160., 10., 10.], (13, 1)))
        with patch.object(stage.prior, "collect", side_effect=AssertionError("New inference")):
            rows, refs = stage.describe_trace(sources["pulse"], wrong, unit, "historical", "pulse", self.old, stage.DESIGN)
            other, deleted = stage.describe_trace(sources["pulse"], self.entry, unit, "historical", "pulse", self.old, stage.DESIGN)
        self.assertTrue(any(r["low_scale_write"] for r in rows))
        self.assertTrue(any(r["reads_low_scale"] for r in rows))
        self.assertTrue(all(r["reference_frame"] < r["frame"] for r in refs))
        self.assertTrue(any(r["kind"] == "scale" and r["drop_crop_changed"] for r in refs))
        for a, b in zip(rows, other):
            for key in ("scale_refs", "appearance_refs", "selected", "ppo_selected", "memory_write", "scale_written"):
                self.assertEqual(a[key], b[key])
        for a, b in zip(refs, deleted):
            for key in ("frame", "reference_frame", "kind", "drop_crop_changed", "drop_selected", "drop_used_history"):
                self.assertEqual(a[key], b[key])
        changed = {k: v.copy() for k, v in sources["pulse"].items()}
        first = int(np.flatnonzero(changed["scale_active"])[0])
        changed["scale_refs"][first, 0] = first+1
        with self.assertRaises(AssertionError):
            stage.describe_trace(changed, self.entry, unit, "historical", "pulse", self.old, stage.DESIGN)

    def test_mechanisms_and_two_level_video_weights(self):
        def state(best, final):
            return dict(gt_valid=True, best_iou=best, iou=final)
        examples = ((.8,.8,.1,.1,"skip_removes_good_supply"),
                    (.1,.1,.8,.8,"skip_restores_good_supply"),
                    (.1,.1,.2,.2,"neither_has_good_supply"),
                    (.8,.1,.8,.8,"skip_rescues_choice"),
                    (.8,.8,.8,.1,"skip_loses_choice"),
                    (.8,.1,.8,.1,"both_miss_available_good"),
                    (.8,.8,.8,.8,"both_select_good"))
        for a,b,c,d,expected in examples:
            self.assertEqual(stage.mechanism(state(a,b), state(c,d), stage.DESIGN), expected)
        self.assertEqual(stage.mechanism(dict(gt_valid=False), dict(gt_valid=False), stage.DESIGN), "invalid_gt")
        rows = []
        for name, seed, count, yes in (("a",1,3,True),("a",2,1,False),("b",1,1,False)):
            rows.extend(dict(cohort="historical", sequence=name, seed=seed,
                mechanism="skip_rescues_choice" if yes else "both_select_good") for _ in range(count))
        out = next(r for r in stage.trigger_summary(rows) if r["cohort"] == "all" and r["mechanism"] == "skip_rescues_choice")
        self.assertEqual(out["states"], 3)
        self.assertAlmostEqual(out["fraction_video"], .25)

    def test_nonfinite_descriptor_is_not_an_effective_appearance_read(self):
        design = dict(stage.DESIGN, spacing=1)
        original_pool = stage.s93.pool
        count = 0
        def pool(*args):
            nonlocal count
            count += 1
            value = original_pool(*args)
            return np.full_like(value, np.nan) if count == 3 else value
        with patch.object(stage.s93, "pool", side_effect=pool):
            data = stage.s100.collect(self.observer, self.entry, self.ap, fixture.choose, "pulse", design, image_reader=self.reader)
        self.assertTrue(data["memory_write"][1])
        self.assertFalse(np.isfinite(data["queries"][1, 0]).all())
        rows, refs = stage.describe_trace(data, self.entry, dict(sequence="test", seed=2171), "historical", "pulse", self.old, design)
        affected = [r for r in rows if r["used_history"] and "2" in r["appearance_refs"].split("|")]
        self.assertTrue(affected)
        self.assertTrue(all("2" not in r["effective_appearance_refs"].split("|") for r in affected))
        self.assertFalse(any(r["kind"] == "appearance" and r["reference_frame"] == 2 for r in refs))

    def test_recovery_inclusive_source_boundary_and_invalid_labels(self):
        truth = np.tile([0.,0.,10.,10.], (31,1))
        data = dict(frame_id=np.arange(5,31), boxes=np.tile([100.,100.,10.,10.], (26,1)))
        # Last allowed start is reference+15; preserve this existing implementation.
        data["boxes"][20:23] = [0.,0.,10.,10.]
        result = stage.recovery_match(data, dict(ground_truth=truth), 10, stage.DESIGN)
        self.assertEqual(result["matched_good_frame"], 25)
        self.assertTrue(result["complete_search_window"])
        truth[26] = [0.,0.,0.,0.]
        self.assertFalse(stage.recovery_match(data, dict(ground_truth=truth), 10, stage.DESIGN)["aligned_recovery"])
        short = {k:v[:6] for k,v in data.items()}
        self.assertFalse(stage.recovery_match(short, dict(ground_truth=truth), 10, stage.DESIGN)["complete_search_window"])
        with self.assertRaises(ValueError):
            stage.compare_recorded([dict(value="nan")], [dict(value=1.)])

    def test_full_cached_pipeline_shared_prefix_reproduction_resume_and_tamper(self):
        sources = self.sources()
        unit = dict(sequence="test", seed=2171, origin="native", policy="ppo", fold=0)
        spec = dict(unit=unit, cohort="historical")
        source = dict(original=self.old, report=dict(native_head_sha256="native"))
        actor = dict(models={}, prior={})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, output = root / "stage101", root / "stage102"
            with (patch.object(stage.prior, "load_unit", return_value=(self.entry, sources)),
                  patch.object(stage.prior.e74, "learned_pack"),
                  patch.object(stage.prior.e74, "choose", side_effect=lambda r,h,*_: fixture.choose(r,h))):
                units = stage.prior.prepare_units(upstream, dict(units=[spec],design=stage.s100.DESIGN),
                    root, root, {}, {}, {}, source, actor, set())
                # Include an actual recovery role to exercise cached branch evaluation.
                anchor = units[0]["anchors"][0]
                anchor["roles"].append(dict(role="newly_lost_ppo_recovery", target_frame=anchor["frame"], reference_recovery=anchor["frame"]))
                plan101 = stage.io.seal(dict(schema=stage.prior.SCHEMA, design=stage.prior.DESIGN, units=units,
                    runtime={}, runtime_differences=[], reuse_keep=True, source_sha256={}, code_sha256={}))
                stage.c.publish_identical(upstream / "manifest.json", plan101)
                real_collect = stage.prior.collect
                with (patch.object(stage.prior.e63.source_experiment, "installed", side_effect=lambda *_: contextlib.nullcontext("native")),
                      patch.object(stage.prior, "collect", side_effect=lambda *a: real_collect(*a, image_reader=self.reader)),
                      contextlib.redirect_stdout(io.StringIO())):
                    stage.prior.run(upstream, root, {}, root, root, {}, {}, {}, source, actor, self.observer, plan101)
                before = self.observer.calls
                with (patch.object(stage.prior, "collect", side_effect=AssertionError("New inference")),
                      patch.object(stage.prior.e63, "checked_image", side_effect=AssertionError("Image read")),
                      contextlib.redirect_stdout(io.StringIO())):
                    tables = stage.run(upstream, root, plan101, dict(design=stage.s100.DESIGN), root, root,
                                       {}, {}, {}, source, actor, stage.DESIGN)
                self.assertEqual(self.observer.calls, before)
                self.assertEqual(len(tables[4]), 2)
                self.assertEqual(len(tables[9]), 24)
                self.assertTrue(tables[10])
                self.assertTrue(all(r["frame"] <= r["case_target"] for r in tables[10]))
                self.assertTrue(any("read_dependency" in r["evidence_roles"] for r in tables[10]))
                self.assertEqual(sum(r["reference_reads"] for r in tables[11]), len(tables[7]))
                expected = dict(units=1, points=len(units[0]["points"]), anchors=len(units[0]["anchors"]),
                    roles=sum(len(a["roles"]) for a in units[0]["anchors"]), recovery_rows=2,
                    branch_frames=sum(2*(a["last"]-a["frame"]+1) for a in units[0]["anchors"]), state_frames=24)
                plan = stage.io.seal(dict(schema=stage.SCHEMA, expected=expected, source_runtime={}, source_sha256={}, code_sha256={}))
                stage.c.publish_identical(output / "manifest.json", plan)
                stage.publish(output, plan, tables)
                stage.publish(output, plan, tables)
                stage.check_results(output)
                (output / "analysis/stage102_cases.csv").write_text("tamper", encoding="utf-8")
                with self.assertRaises(ValueError):
                    stage.check_results(output)


if __name__ == "__main__":
    unittest.main()
