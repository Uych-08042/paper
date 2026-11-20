"""Check the geometry cross and first-successor diagnosis with fixed responses."""
import unittest

import numpy as np

import stage151_write_geometry_cross as stage
from test_stage143_fixed_pool_quality import fixture


class WriteGeometryCross(unittest.TestCase):
    def setUp(self):
        self.case = dict(case_id="fixture", sequence="uav0000003_00000_s", source="uav0000003",
                         frame=1, onset=True, successors=3,
                         anchor_boxes=[[40., 40., 16., 16.], [22., 24., 24., 20.]])

    def test_cross_preserves_the_other_component(self):
        boxes = np.array(stage.cross(self.case)["anchor_boxes"])
        centers = boxes[:, :2] + .5 * boxes[:, 2:]
        np.testing.assert_array_equal(boxes[stage.CONTROL], self.case["anchor_boxes"])
        np.testing.assert_array_equal(centers[[0, 1, 2, 3]], centers[[2, 3, 0, 1]])
        np.testing.assert_array_equal(boxes[[0, 1, 2, 3], 2:], boxes[[1, 0, 3, 2], 2:])

    def test_controls_reproduce_and_audit_each_branch(self):
        raw, size, offset, _, _, hw, _, _, ap = fixture([40., 40., 16., 16.])
        ap.update(root="fixture", search_factor=4.)
        model = dict(mean=np.zeros(9), scale=np.ones(9), coef=np.zeros(9), intercept=0.)

        def observe(observer, image, parent, frame, settings):
            crop, usable, factor = stage.prior.crop_geometry(parent, hw, settings["search_factor"], settings["search_size"])
            decoded = stage.prior.fixed.dense_boxes(size, offset, parent, factor, settings["search_size"], hw)
            return dict(raw=raw, size=size, offset=offset, parent=parent, boxes=decoded[raw.argmax()],
                        crop=crop, usable=usable, factor=factor, image_hw=hw, crop_sha256=str(crop),
                        layer=9, head_calls=1, selector_calls=1,
                        block_calls=np.array([1]*6 + [0, 0, 1, 0, 0, 0]))

        kwargs = dict(read_image=lambda root, frame:frame, observe=observe)
        args = list(range(5)), [40., 40., 16., 16.], model, ap
        old = stage.prior.collect(self, self.case, *args, **kwargs)
        mixed = stage.cross(self.case)
        data = stage.collect(self, mixed, *args, **kwargs)
        self.assertEqual(stage.audit(data, mixed, old, True, 16), dict(max_control_box_error=0., control_rows_equal=6))
        truth = np.array([[40., 40., 16., 16.]]*5)
        first, windows, trace = stage.describe(data, mixed, truth, ap)
        self.assertEqual(first[0]["outcome"], "selected_good_box")
        self.assertEqual(first[0]["coverage"], 1.)
        self.assertEqual(windows[0]["first_good_run"], 2)
        self.assertEqual(len(trace), 12)
        data["parent"][1, 1] = data["boxes"][0, 0]
        with self.assertRaises(AssertionError):
            stage.audit(data, mixed, old, True, 16)

    def initialize(self, image, box):
        self.assertEqual(image, 0)

    def test_terminal_anchor_retains_four_empty_windows(self):
        case = dict(self.case, successors=0)
        ap = {"search_size":256}
        data = stage.collect(None, stage.cross(case), [], None, None, ap)
        old = stage.prior.collect(None, case, [], None, None, ap)
        self.assertEqual(stage.audit(data, stage.cross(case), old, True, 256)["control_rows_equal"], 0)
        search, windows, frames = stage.describe(data, stage.cross(case), np.ones((2,4)), ap)
        self.assertEqual(search, [])
        self.assertEqual(frames, [])
        self.assertEqual(len(windows), 8)
        self.assertTrue(all(r["successor_mean_iou"] is None for r in windows))

    def test_paired_interaction_is_source_weighted(self):
        rows = []
        for case, source, outcomes in (("a", "s1", [.5,.6,.7,.9]), ("b", "s1", [.5,.5,.5,.5]),
                                        ("c", "s2", [.5,.5,.5,.3])):
            rows.extend(dict(case_id=case, source=source, horizon=8, arm=arm, full_labeled_window=True,
                             successor_mean_iou=value) for arm,value in zip(stage.ARMS,outcomes))
        paired = {r["contrast"]:r for r in stage.paired(rows) if r["horizon"] == 8}
        self.assertAlmostEqual(paired["center_by_size_interaction"]["source_mean_iou_delta"], -.075)
        self.assertAlmostEqual(paired["joint_write"]["source_mean_iou_delta"], 0.)


if __name__ == "__main__":
    unittest.main()
