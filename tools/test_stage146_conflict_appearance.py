"""Protect the matched candidate menu and calibration fallback."""
import unittest

import numpy as np

import stage146_conflict_appearance as stage


class ConflictAppearance(unittest.TestCase):
    def test_shared_candidate_menu_ignores_iou_labels(self):
        row = dict(pools={"dense":np.array([0,1,2,3])},
                   boxes=np.array([[0,0,10,10],[20,0,10,10],[1,1,10,10],[40,0,10,10]]),
                   x=np.zeros((4,9)))
        row["x"][:,2] = [.5,1.,.8,.7]
        cues = dict(recent_hist=np.array([.1,.9,.8,np.nan]), template_ncc=np.array([.5,.3,.4,.8]))
        values = np.array([.6,.57,.56,.55])
        anchor, geometry, eligible, scores = stage.candidate_menu(row, cues, values)
        self.assertEqual(anchor, 0)
        np.testing.assert_array_equal(geometry, [1,3])
        np.testing.assert_array_equal(eligible, [1])
        np.testing.assert_allclose([scores[arm][0] for arm in stage.METHODS], [.02,.10,0.], atol=1e-12)
        cues["recent_hist"][0] = np.nan
        self.assertEqual(len(stage.candidate_menu(row, cues, values)[2]), 0)

    def test_calibration_keeps_baseline_if_gain_needs_an_onset_loss(self):
        arm = stage.METHODS[0]

        def row(margin, incumbent, proposal, stable=False, onset=False):
            return dict(source="fit", anchor_iou=incumbent, stable_window=stable, onset=onset,
                        **{arm+"_margin":margin,arm+"_best_iou":proposal})

        keep, grid = stage.calibrate([row(.08,.1,.8),row(.09,.7,.3,onset=True)], arm)
        self.assertIsNone(keep["threshold"])
        self.assertTrue(any(trial["calibration_onset_losses"] for trial in grid))
        switch, _ = stage.calibrate([row(.06,.1,.8),row(.02,.7,0.,stable=True)], arm)
        self.assertEqual(switch["threshold"], .05)
        self.assertEqual(switch["calibration_switches"], 1)


if __name__ == "__main__":
    unittest.main()
