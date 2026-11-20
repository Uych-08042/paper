"""Keep candidate admission fixed while varying the acceptance score."""
import unittest

import numpy as np

import stage149_score_ablation as stage


class ScoreAblation(unittest.TestCase):
    def test_appearance_only_changes_acceptance_without_changing_the_menu(self):
        x = np.zeros((2,9))
        x[:,1] = [.72,.60]
        x[:,2] = [.95,.35]
        row = dict(x=x,boxes=np.array([[0.,0.,12.,12.],[6.,0.,12.,12.]]),native_iou=.12,
                   y=np.array([.42,.67]),cells=np.array([6,14]),pools={"dense":np.array([0,1])})
        baseline = dict(source="uav1",sequence="uav1_00000_s",frame=10,dense_quality_cell=6,
                        onset=True,stable_window=0,state="native",dense_only_good=True,grid_iou=.67)
        cues = dict(recent_hist=np.array([.1,.8]),template_ncc=np.array([.30,.50]))
        model = dict(mean=[0.]*9,scale=[1.]*9,coef=[0.,1.]+[0.]*7,intercept=0.)
        case = row,baseline,cues
        original_menu = stage.previous.evidence(case,model,design=stage.wider.DESIGN)
        amended = stage.evidence(case,model)
        for k in ("geometry","eligible"):
            np.testing.assert_array_equal(original_menu[k],amended[k])
        np.testing.assert_allclose([amended["margins"][a][0] for a in (
            "with_recent_color","recent_hist_without_regression","recent_hist_without_raw","recent_hist_only"
        )], [-.11,.01,-.05,.07])
        thresholds = dict.fromkeys(stage.METHODS,.025)
        archive = stage.previous.evaluate(case,model,{a:thresholds[a] for a in stage.previous.METHODS},design=stage.wider.DESIGN)
        archive.update({a+"_threshold":thresholds[a] for a in stage.previous.METHODS})
        reference = {k:"" if v is None else str(v) for k,v in archive.items()}
        calibrated,zero = stage.evaluate(case,model,thresholds,reference)
        self.assertEqual(calibrated["matched_cells"],"14")
        self.assertFalse(calibrated["dense_with_recent_color_switch"])
        self.assertTrue(calibrated["dense_recent_hist_only_switch"])
        self.assertFalse(calibrated["dense_recent_hist_without_regression_switch"])
        self.assertTrue(zero["dense_recent_hist_without_regression_switch"])
        self.assertFalse(zero["dense_recent_hist_without_raw_switch"])
        reference["with_recent_color_threshold"] = "0.01"
        with self.assertRaisesRegex(ValueError,"recalibration changed"):
            stage.evaluate(case,model,thresholds,reference)


if __name__ == "__main__":
    unittest.main()
