"""Check peak placement, protected slots, and physical search accounting."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

import stage129_coarse_to_fine as stage
import test_stage113_frozen_closed_loop as fixture


class CoarseToFine(unittest.TestCase):
    def test_keeps_wide_and_q0_and_skips_peak_on_padding(self):
        hw = (200,400)
        ap = dict(search_factor=4.,search_size=256)
        old = np.array([stage.parent_at(c,s,4.) for c,s in
                        [([120,100],44),([200,100],900),([140,100],44),([160,100],44)]])
        trimmed = np.array([stage.source.trim_padding(p,hw,4.,256) for p in old])
        raw = np.zeros((4,8,8))
        raw[1,0,0] = 100.  # Top row falls inside vertical padding.
        raw[1,2,1] = .9
        raw[1,4,6] = .8
        result,wide,chosen,side = stage.zoom_layout(trimmed,old,raw,hw,ap)
        self.assertEqual((wide,side),(1,100))
        self.assertEqual(chosen,{2:17,3:38})
        np.testing.assert_array_equal(result[:2],trimmed[:2])
        self.assertEqual(len({tuple(stage.crop_geometry(p,hw,4.,256)[0]) for p in result}),4)
        untouched,wide,chosen,side = stage.zoom_layout(trimmed,trimmed,raw,hw,ap)
        np.testing.assert_array_equal(untouched,trimmed)
        self.assertEqual((wide,chosen,side),(-1,{},0))

    def test_when_q0_is_the_only_wide_crop_q1_is_preserved(self):
        hw = (160,240)
        ap = dict(search_factor=4.,search_size=256)
        old = np.array([stage.parent_at(c,s,4.) for c,s in
                        [([120,80],500),([25,25],40),([60,60],40),([90,90],40)]])
        trimmed = np.array([stage.source.trim_padding(p,hw,4.,256) for p in old])
        result,wide,chosen,side = stage.zoom_layout(trimmed,old,np.ones((4,8,8)),hw,ap)
        self.assertEqual((wide,set(chosen),side),(0,{2,3},60))
        np.testing.assert_array_equal(result[:2],trimmed[:2])
        centers = result[2:,:2]+result[2:,2:]/2
        self.assertGreaterEqual(np.linalg.norm(centers[1]-centers[0]),30)


class MatchedObservation(unittest.TestCase):
    setUp = fixture.FrozenClosedLoop.setUp

    def test_control_and_zoom_share_retained_outputs_and_labels_only_evaluate(self):
        item = self.entry["frames"][0]
        image = self.reader(self.ap["root"],item)
        hw, factor = image.shape[:2],self.ap["search_factor"]
        initial = np.asarray(self.entry["initial_box"],float)
        center = np.array(hw[::-1])/2
        old = np.array([initial,stage.parent_at(center,3*max(hw),factor),
                        stage.parent_at(center+5,32,factor),stage.parent_at(center-5,32,factor)])
        trimmed = np.array([stage.source.trim_padding(p,hw,factor,self.ap["search_size"]) for p in old])
        self.observer.initialize(image,initial)
        control = stage.base.s112.observe(self.observer,image,[initial],1,self.ap,trimmed)
        zoom,wide,peaks,side = stage.zoom_layout(trimmed,old,control["raw"],hw,self.ap)
        self.assertEqual((wide,len(peaks)),(1,2))
        ident = dict(sequence="fixture",source_seed=2312,frame=1)
        with tempfile.TemporaryDirectory() as root:
            bank,n = stage.source.observe_bank(Path(root),self.observer,zoom,control,1,hw,[initial],
                self.ap,lambda:image,lambda:None,dict(ident,arm="zoom"),{})
            self.assertEqual(n,2)
            np.testing.assert_array_equal(bank["candidate_boxes"][:2],control["candidate_boxes"][:2])
            anchor = dict(source_small_covered="True")
            for gt in (initial, initial+np.array([5.,5.,0.,0.])):
                summary,crops = stage.report_point(ident,control,bank,wide,peaks,side,gt,self.model,
                    [initial],dict(trim=4,zoom=n),self.ap,hw,anchor)
                self.assertEqual(summary["zoom_search_slots"],2)
                self.assertEqual(len(crops),8)


if __name__=="__main__":
    unittest.main()
