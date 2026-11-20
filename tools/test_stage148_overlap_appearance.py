"""Check that opening the menu is distinct from accepting a new prediction."""
import unittest

import numpy as np

import stage148_overlap_appearance as stage


class WiderOverlap(unittest.TestCase):
    def test_new_candidate_is_admitted_and_threshold_modes_may_disagree(self):
        x = np.zeros((2,9))
        x[:,1] = [.70,.66]
        x[:,2] = [.3,1.]
        row = dict(x=x, boxes=np.array([[0.,0.,12.,12.],[6.,0.,12.,12.]]), native_iou=.12,
                   y=np.array([.42,.67]), cells=np.array([6,14]), pools={"dense":np.array([0,1])})
        baseline = dict(source="uav1", sequence="uav1_00000_s", frame=10, dense_quality_cell=6, onset=True,
                        stable_window=0,state="native", dense_only_good=True, grid_iou=.67)
        features = dict(recent_hist=np.array([.2,.8]), template_ncc=np.array([.4,.3]))
        model = dict(mean=[0.]*9, scale=[1.]*9, coef=[0.,1.]+[0.]*7, intercept=0.)
        limits = {"frozen_thresholds":{a:.10 for a in stage.METHODS}, "recalibrated":{a:.05 for a in stage.METHODS}}
        case = row, baseline, features
        reference = {k:"" if v is None else str(v) for k,v in
                     stage.previous.evaluate(case, model, limits["frozen_thresholds"]).items()}
        frozen, adjusted = stage.evaluate(case, model, limits, reference, ([14],[14]))
        self.assertEqual(frozen["matched_cells"], "14")
        self.assertFalse(frozen["dense_with_recent_color_switch"])
        self.assertTrue(adjusted["dense_with_recent_color_switch"])
        self.assertFalse(adjusted["dense_response_geometry_switch"])
        counts = next(r for r in stage.funnel([adjusted]) if r["population"]=="native_onsets" and r["arm"]=="with_recent_color")
        self.assertEqual((counts["newly_accessible"],counts["scored_first"],counts["accepted_good"]),(1,1,1))

    def test_rejects_held_source_inside_calibration(self):
        row = dict(chosen="True", held_source="a", arm=stage.METHODS[0],calibration_sources="a|b", threshold="0.025")
        with self.assertRaisesRegex(ValueError, "provenance"):
            stage.stage146_thresholds([row], {"a","b"})


if __name__ == "__main__":
    unittest.main()
