"""Check causal references, usable crop boundaries and matched score availability."""
import unittest

import numpy as np

import stage145_fixed_pool_appearance as stage


def example(frame, ious=(.1,.8), *, agreement=True):
    boxes = np.array([[0.,0.,10.,10.],[20.,0.,10.,10.]])
    x = np.zeros((2, len(stage.fixed.FEATURES)))
    x[:,0] = [.9,.1] if agreement else [.1,.9]
    x[:,1] = [.9,.1]
    return dict(sequence="uav0000001_00000_s", source="uav0000001", state="native", case="", branch="",
                frame=frame, stable_window=0, study_window=0, relapse=False, onset=False,
                native_iou=float(ious[0]), grid_iou=float(max(ious)), budget=2, x=x, y=np.array(ious),
                pools={"peak":np.array([0,1]), "dense":np.array([0,1])}, cells=np.array([5,9]),
                boxes=boxes, parent=boxes[0], usable=np.array([0.,0.,32.,16.]))


class FrozenAppearance(unittest.TestCase):
    def test_reference_never_uses_current_future_or_untrusted_frame(self):
        history = {f:example(f, agreement=f!=14) for f in (1,5,8,11,13,14,15,16)}
        refs = stage.recent_rows(15, history)
        self.assertEqual([r["frame"] for r in refs], [13,8,5])
        self.assertTrue(stage.trusted(history[13]))
        history[13] = example(13, ious=(0.,0.))
        self.assertTrue(stage.trusted(history[13]))

    def test_same_pair_color_cue_can_rescue_and_missing_texture_uses_baseline(self):
        row = example(1)
        template = np.zeros((16,32,3), dtype=np.uint8)
        template[:10,:10,0] = 255
        current = np.zeros_like(template)
        current[:10,:10,2] = 255
        current[:10,20:30,0] = 255
        model = dict(mean=[0.]*9, scale=[1.]*9, coef=[1.]+[0.]*8, intercept=0.)
        methods = {k:model for k in stage.prior.METHODS}
        expected = stage.prior.evaluate(row, model, methods)
        ref = {(stage.context_key(row), 1):{k:str(v) for k,v in expected.items()}}
        entry = dict(initial_box=[0.,0.,10.,10.], frames=[0,1])
        decisions, pairs, reads = stage.read_sequence(entry, None, [row], {stage.context_key(row):{1:row}}, ref,
                                                      model, methods, reader=lambda _,f:(template,current)[f])
        self.assertEqual(reads, 2)
        self.assertEqual((decisions[0]["dense_template_hist_cell"], decisions[0]["dense_template_ncc_cell"]), (9,5))
        self.assertTrue(decisions[0]["dense_template_hist_available"])
        self.assertFalse(decisions[0]["dense_template_ncc_available"])
        self.assertGreater(pairs[0]["template_hist_margin"], 0.)
        self.assertIsNone(pairs[0]["template_ncc_margin"])
        self.assertIsNone(stage.candidate_descriptor(current, row["boxes"][1], np.array([0.,0.,21.,16.]))["hist"])


if __name__ == "__main__":
    unittest.main()
