"""Exercise one-box substitution, native continuation and archived controls."""
import unittest
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np

import stage134_native_peak_continuation as stage
from stage27_uav_common import crop_geometry


class NativePeakContinuation(unittest.TestCase):
    def test_choose_stable_window_deterministically(self):
        self.assertEqual(stage.healthy_frame(np.ones(9), 3), 5)
        ious = np.array([.9, .9, .9, .9, 0, .9, .9, .9, .9])
        self.assertEqual(stage.healthy_frame(ious, 3), 6)
        self.assertIsNone(stage.healthy_frame(ious, 4))
        self.assertIsNone(stage.healthy_frame(np.ones(3), 3))

    def setUp(self):
        self.parent = np.array([40., 40., 16., 16.])
        self.truth = np.array([8., 8., 16., 16.])
        self.hw = (128, 128)
        window = np.full((4, 4), .2, np.float32)
        window[2, 2] = 1.
        self.ap = dict(root=None, search_factor=4., search_size=64, window=window, design={"box_atol": 1e-4})
        self.case = dict(case_id="fixture", sequence="fixture", kind="onset", frame=1, successors=3)
        self.entry = dict(frames=list(range(5)), initial_box=self.parent,
                          ground_truth=np.vstack((self.parent, np.tile(self.truth, (4, 1)))))
        self.raw = np.full((4, 4), .01, np.float32)
        self.raw[0, 0], self.raw[2, 2] = .9, .8
        crop, _, _ = crop_geometry(self.parent, self.hw, 4., 64)
        self.trace = dict(candidate_parents=np.tile(self.parent, (4, 1, 1)), crop=np.tile(crop, (4, 1, 1)),
                          crop_sha256=np.full((4, 1), "fixture"), image_hw=np.tile(self.hw, (4, 1)),
                          raw=np.tile(self.raw, (4, 1, 1, 1)), boxes=np.tile(self.parent, (4, 1)))

    def collect_fixture(self):
        def observe(_observer, _image, parent, _frame, ap):
            crop, _, factor = crop_geometry(parent, self.hw, ap["search_factor"], ap["search_size"])
            return dict(boxes=parent.copy(), parent=parent, crop=crop, crop_sha256="fixture",
                        raw=self.raw, size=np.full((2, 4, 4), .25, np.float32),
                        offset=np.zeros((2, 4, 4), np.float32), factor=factor, image_hw=self.hw,
                        layer=7, head_calls=1, selector_calls=1, block_calls=[1] * 7 + [0] * 5)

        observer = Mock()
        forward = Mock(side_effect=observe)
        data = stage.collect(observer, self.case, self.entry, self.parent, self.ap,
                             image_reader=lambda _root, frame: frame, observe=forward)
        observer.initialize.assert_called_once()
        self.assertEqual(forward.call_count, 13)
        return data

    def test_shared_anchor_and_native_tail(self):
        data = self.collect_fixture()
        self.assertEqual(stage.verify_case(data, self.case, self.entry, self.trace, self.ap, True), 0.)
        rows, timeline = stage.describe(data, self.case, self.entry)
        self.assertEqual(len(timeline), 16)
        self.assertEqual([r["anchor_iou"] for r in rows], [0., 1., 1., 1.])
        self.assertEqual([r["tail_mean_iou"] for r in rows], [0., 1., 1., 1.])
        self.assertEqual([r["first_tail_good_run"] for r in rows], [None, 2, 2, 2])
        summary = stage.summarize([dict(row, live_native_low_anchor=True) for row in rows])
        dense = [r for r in summary if r["kind"] == "onset_live_low" and r["arm"] == "dense_oracle"]
        self.assertEqual(len(dense), 1)
        self.assertEqual(dense[0]["new_tail_good_runs"], 1)

    def test_reject_broken_parent_chain(self):
        data = self.collect_fixture()
        data["parent"][1, 1, 0] += 1.
        with self.assertRaises(AssertionError):
            stage.verify_case(data, self.case, self.entry, self.trace, self.ap, True)

    def test_publication_reuses_verified_case_and_rejects_table_change(self):
        data = self.collect_fixture()
        plan = stage.io.seal(dict(schema=stage.SCHEMA, design=stage.DESIGN, source_stage133=stage.SOURCE133,
                                  cases=[self.case], runtime={}, runtime_differences=[], native_head_sha256="head",
                                  budget=13))
        contexts = {"fixture": (dict(spec={"unit": {"origin": "native"}}), self.entry, self.trace)}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            stage.c.publish_identical(root / "manifest.json", plan)
            with (patch.object(stage, "collect", return_value=data) as collect,
                  patch.object(stage.base.e63.source_experiment, "installed", side_effect=lambda *_: nullcontext("head"))):
                stage.run(root, plan, contexts, Mock(), self.ap)
                stage.run(root, plan, contexts, Mock(), self.ap)
            collect.assert_called_once()
            stage.check_results(root)
            (root / "stage134_windows.csv").write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                stage.check_results(root)


if __name__ == "__main__":
    unittest.main()
