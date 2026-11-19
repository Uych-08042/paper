"""Check the audit on fixed native boxes and response maps with known alternatives."""
import unittest

import numpy as np

import stage133_native_failure_audit as stage
from stage27_uav_common import crop_geometry, success_auc, tracking_iou


class NativeFailureAudit(unittest.TestCase):
    def setUp(self):
        self.parent = np.array([40., 40., 16., 16.])
        self.hw = (128, 128)
        self.n = 6
        self.window = np.full((4, 4), .2, np.float32)
        self.window[2, 2] = 1.
        self.ap = dict(search_factor=4., search_size=64, window=self.window,
                       design={"box_atol": 1e-4})
        crop, _, _ = crop_geometry(self.parent, self.hw, 4., 64)
        raw = np.full((4, 4), .01, np.float32)
        raw[1, 1], raw[2, 2] = .9, .8
        self.trace = dict(frame_id=np.arange(1, self.n + 1), selected=np.zeros(self.n, int),
                          unique_search_calls=np.ones(self.n, int), boxes=np.tile(self.parent, (self.n, 1)),
                          candidate_parents=np.tile(self.parent, (self.n, 1, 1)),
                          candidate_boxes=np.tile(self.parent, (self.n, 1, 1)),
                          crop=np.tile(crop, (self.n, 1, 1)), image_hw=np.tile(self.hw, (self.n, 1)),
                          raw=np.tile(raw, (self.n, 1, 1, 1)),
                          size=np.full((self.n, 1, 2, 4, 4), .25, np.float32),
                          offset=np.zeros((self.n, 1, 2, 4, 4), np.float32))

    def test_raw_peak_repairs_onset_but_harms_healthy_frames(self):
        raw_box = np.array([24., 24., 16., 16.])
        truth = np.vstack((self.parent, np.tile(raw_box, (3, 1)), np.tile(self.parent, (3, 1))))
        entry = dict(sequence="fixture", initial_box=self.parent, ground_truth=truth)
        archived = dict(cohort="historical", failure_events=1,
                        auc=success_auc(tracking_iou(self.trace["boxes"], truth[1:])))
        video, events, anchors, counts = stage.diagnose(self.trace, entry, self.ap, archived)
        self.assertEqual((video["low_to_good_frames"], video["good_to_low_frames"]), (3, 3))
        self.assertEqual(events[0]["onset_frame"], 1)
        self.assertEqual(events[0]["recovery_frame"], 4)
        self.assertEqual(events[0]["category"], "raw_top1_recovers_good")
        self.assertTrue(events[0]["raw_top5_gt_recall"])
        self.assertTrue(all(1 <= r["frame"] <= self.n for r in anchors))
        self.assertEqual(counts, {"raw_top1_recovers_good": 3, "native_good": 3})

    def test_distinguishes_missing_crop_from_dense_alternative(self):
        for gt, expected in (([104., 104., 16., 16.], "no_crop_overlap_no_good"),
                             ([56., 56., 16., 16.], "other_dense_cell_has_good"),
                             ([21., 21., 29., 29.], "adequate_crop_no_good")):
            with self.subTest(expected=expected):
                row = stage.frame_metrics(self.trace, 0, np.asarray(gt), self.ap, self.window, False)
                self.assertEqual(row["category"], expected)

    def test_inexact_hann_decode_rejects_cache(self):
        self.trace["candidate_boxes"][0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "Native decode mismatch"):
            stage.frame_metrics(self.trace, 0, self.parent, self.ap, self.window, False)


if __name__ == "__main__":
    unittest.main()
