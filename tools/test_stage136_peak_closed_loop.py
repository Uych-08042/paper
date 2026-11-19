"""Exercise candidate parity, own-history feedback and one-search accounting."""
import csv
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

import stage136_peak_closed_loop as stage


class PeakClosedLoop(unittest.TestCase):
    def setUp(self):
        self.parent = np.array([40., 40., 16., 16.])
        self.alternate = np.array([8., 8., 16., 16.])
        self.ap = dict(root=None, search_size=64, window=np.ones((4, 4), np.float32))

    def observe(self, _observer, _image, parent, frame, _ap):
        raw = np.full((4, 4), .01, np.float32)
        raw[0, 0], raw[2, 2] = (.90 if frame == 1 else .10), 1.
        return dict(boxes=parent.copy(), parent=parent, raw=raw, size=np.full((2, 4, 4), .25, np.float32),
                    offset=np.zeros((2, 4, 4), np.float32), factor=1., image_hw=(128, 128),
                    layer=7, head_calls=1, selector_calls=1, block_calls=[1] * 7 + [0] * 5)

    def collect(self, arm):
        observe = Mock(side_effect=self.observe)
        trace = stage.collect(Mock(), list(range(4)), self.parent, self.ap, arm,
                              image_reader=lambda _root, item: item, observe=observe)
        self.assertEqual(observe.call_count, 3)
        stage.audit_trace(trace, 3, self.parent, arm)
        return trace

    def test_same_stage134_runner_and_own_history(self):
        score_arm = stage.ARMS[1]
        trace = self.collect(score_arm)
        archived_choice = stage.prior.choices(self.observe(None, None, self.parent, 1, self.ap), self.alternate, self.ap)
        np.testing.assert_array_equal(trace["candidates"][0], archived_choice["boxes"][:2])
        np.testing.assert_array_equal(trace["switched"], [True, False, False])
        np.testing.assert_array_equal(trace["parent"][1:], np.tile(self.alternate, (2, 1)))
        for arm in (stage.ARMS[0], stage.ARMS[2]):
            control = self.collect(arm)
            self.assertFalse(control["switched"].any())
            np.testing.assert_array_equal(control["boxes"], np.tile(self.parent, (3, 1)))
        trace["parent"][1] = self.parent
        with self.assertRaises(AssertionError):
            stage.audit_trace(trace, 3, self.parent, score_arm)

    def test_reports_budget_and_resumes(self):
        job = dict(spec=dict(unit=dict(sequence="fixture", origin="native"), cohort="historical"), frames=3)
        entry = dict(sequence="fixture", frames=list(range(4)), initial_box=self.parent,
                     ground_truth=np.tile(self.parent, (4, 1)))
        plan = stage.io.seal(dict(schema=stage.SCHEMA, design=stage.DESIGN, budget=9, jobs=[job], runtime={},
                                  runtime_differences=["fixture"], source_stage135=stage.SOURCE135))
        traces = {arm: self.collect(arm) for arm in stage.ARMS}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            stage.c.publish_identical(root / "manifest.json", plan)
            with (patch.object(stage, "collect", side_effect=lambda *args: traces[args[-1]]) as collect,
                  patch.object(stage.base.e63.source_experiment, "installed", side_effect=lambda *_: nullcontext("head"))):
                for _ in range(2):
                    stage.run(root, None, {"native_head_sha256": "head"}, None, plan,
                              lambda _: entry, Mock(), self.ap)
                self.assertEqual(collect.call_count, 3)
            stage.check_results(root)
            with (root / "stage136_video_summary.csv").open(newline="") as handle:
                by_arm = {row["arm"]: row for row in csv.DictReader(handle)}
            self.assertEqual(by_arm[stage.ARMS[1]]["new_failure_on_native_stable"], "True")
            self.assertEqual(by_arm[stage.ARMS[1]]["first_switch_frame"], "1")


if __name__ == "__main__":
    unittest.main()
