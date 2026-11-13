"""Exercise scale-memory reconstruction, prefix reads and no-inference publication."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage98_scale_memory_audit as stage
import test_stage97_trusted_scale as fixture


class MemoryAudit(unittest.TestCase):
    def fixture(self):
        old = dict(root="fixture", search_factor=5., search_size=100, window=np.ones((5, 5), np.float32),
                   actor_design=dict(stage.e63.source_common.DESIGN))
        initial = np.array([80., 80., 20., 20.])
        entry = dict(initial_box=initial, frames=[{}]*9, ground_truth=np.tile([160., 160., 10., 10.], (9, 1)))
        empty = dict(boxes=np.empty((0, 4)), anchor=np.array([1., 0.]))
        observer = fixture.Observer()
        reader = lambda *_: np.zeros((200, 200, 3), np.uint8)
        with patch.object(stage.s93.s91, "cpu_array", side_effect=lambda x: x.copy()):
            source = stage.s97.collect(observer, entry, empty, 1, 8, fixture.choose, "control", old,
                                      stage.s97.DESIGN, image_reader=reader)
            trusted = stage.s97.collect(observer, entry, source, 3, 8, fixture.choose, "trusted_scale", old,
                                       stage.s97.DESIGN, image_reader=reader)
        event = dict(sequence="test", seed="2171", outcome="missed", first_audit_frame="3",
                     last_audit_frame="8", ppo_recovery_frame="5", matched_good_frame="")
        return old, entry, source, trusted, event

    def test_wrong_writes_prefix_references_and_drop_sensitivity(self):
        old, entry, source, trusted, event = self.fixture()
        stats, rows, references = stage.audit_branch(event, trusted, source, entry, "trusted_scale", old, stage.s97.DESIGN)
        self.assertEqual(stats["low_iou_writes"], 6)
        self.assertEqual(stats["frames_reading_low_iou_refs"], 6)
        self.assertTrue(any(r["reference_from_prefix"] for r in references))
        self.assertGreater(stats["crop_sensitive_drops"], 0)
        self.assertTrue(all(r["reference_frame"] < r["frame"] for r in references))
        self.assertTrue(all(r["iou"] == 0 for r in rows))
        altered = {k: v.copy() for k, v in trusted.items()}
        altered["scale_refs"][0, 0] = 3
        with self.assertRaises(AssertionError):
            stage.audit_branch(event, altered, source, entry, "trusted_scale", old, stage.s97.DESIGN)

    def test_complete_cached_run_and_identical_publication(self):
        old, entry, source, trusted, event = self.fixture()
        control = stage.s97.control_slice(source, 3, 8)
        unit = dict(sequence="test", seed=2171, origin="native", policy="ppo", fold=0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            window = dict(event)
            for arm, data in (("control", control), ("trusted_scale", trusted)):
                metrics, _ = stage.s97.describe(data, entry, event, stage.s97.DESIGN)
                window.update({f"{arm}_{k}": v for k, v in metrics.items()})
                path = root / "branches" / (stage.s97.case_key(event)+"__"+arm+".npz")
                stage.c.save(path, data, dict(plan_fingerprint="fixture97", unit=unit,
                    case=stage.s97.case_key(event), arm=arm))
            (root / "analysis").mkdir(exist_ok=True)
            stage.s97.s86.publish_table(root / "analysis/stage97_windows.csv", [window])
            plan = dict(fingerprint="fixture97", design=stage.s97.DESIGN, events=[event])
            plan89 = dict(units=[dict(unit=unit)])
            with (patch.object(stage.s95, "load_unit", return_value=(entry, dict(gated=source))),
                  patch.object(stage.s97, "collect", side_effect=AssertionError("New inference")),
                  contextlib.redirect_stdout(io.StringIO())):
                tables = stage.run(root, root, root, plan, {}, plan89, {}, dict(original=old))
                repeat = stage.run(root, root, root, plan, {}, plan89, {}, dict(original=old))
            self.assertEqual([len(t) for t in tables[:3]], [1, 2, 12])
            self.assertEqual(tables[0][0]["transition"], "unchanged")
            for name, first, second in zip(stage.TABLES, tables, repeat):
                path = root / f"stage98_{name}.csv"
                stage.s97.s86.publish_table(path, first)
                stage.s97.s86.publish_table(path, second)
            np.testing.assert_equal(stage.input_geometry(np.array([0., 0., 20., 20.]), np.zeros(4), [200,200],
                                                       stage.e63.c.actor_plan(old)), (None, None))


if __name__ == "__main__":
    unittest.main()
