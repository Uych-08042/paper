"""Check that the cap preserves image pixels, including SGLA border conventions."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

import stage128_padding_trim as stage
import test_stage113_frozen_closed_loop as fixture


class PaddingTrim(unittest.TestCase):
    def test_only_oversized_nonempty_crops_are_trimmed(self):
        for parent in ([4.,3.,16.,16.],[5000.,5000.,250.,250.]):
            np.testing.assert_array_equal(stage.trim_padding(np.array(parent),(100,120),4.,256),parent)
        parent = np.array([-80.,-70.,260.,220.])
        new = stage.trim_padding(parent,(100,120),4.,256)
        raw,usable,_ = stage.crop_geometry(new,(100,120),4.,256)
        self.assertEqual(raw[2],120)
        np.testing.assert_array_equal(usable,[0,0,119,99])
        self.assertAlmostEqual(new[2]/new[3],parent[2]/parent[3])

    def test_preserves_all_original_image_pixels_across_aspect_ratios_and_borders(self):
        rng = np.random.default_rng(128)
        changes = 0
        for hw in ((240,640),(720,360),(513,513)):
            for i in range(300):
                parent = np.r_[rng.uniform(-800,800,2),rng.uniform(10,750,2)]
                new = stage.trim_padding(parent,hw,4.,256)
                crop0, image0, _ = stage.crop_geometry(parent,hw,4.,256)
                crop1, image1, _ = stage.crop_geometry(new,hw,4.,256)
                if np.array_equal(new,parent):
                    self.assertTrue(crop0[2]<=max(hw) or 0 in image0[2:])
                    continue
                changes += 1
                self.assertEqual(crop1[2],max(hw))
                self.assertTrue((image1[:2]<=image0[:2]).all())
                self.assertTrue((image1[:2]+image1[2:]>=image0[:2]+image0[2:]).all())
        self.assertGreater(changes,100)


class Matching(unittest.TestCase):
    setUp = fixture.FrozenClosedLoop.setUp

    def test_reuses_identical_searches_and_runs_only_changed_parents(self):
        item = self.entry["frames"][0]
        image = self.reader(self.ap["root"],item)
        hw = image.shape[:2]
        initial = np.asarray(self.entry["initial_box"],float)
        large = np.r_[-np.asarray(hw,dtype=float),3*np.asarray(hw,dtype=float)]
        old = np.array([initial,initial,large,large])
        new = np.array([stage.trim_padding(p,hw,self.ap["search_factor"],self.ap["search_size"]) for p in old])
        self.observer.initialize(image,initial)
        ident = dict(sequence="fixture",source_seed=2312,frame=1)
        with tempfile.TemporaryDirectory() as path:
            evidence = {}
            root = Path(path)
            common = (1,hw,[initial],self.ap,lambda:image,lambda:None)
            original,n = stage.observe_bank(root,self.observer,old,None,*common,dict(ident,arm="original"),evidence)
            trimmed,m = stage.observe_bank(root,self.observer,new,original,*common,dict(ident,arm="trim"),evidence)
            self.assertEqual(n,2)
            self.assertEqual(m,1)
            self.assertEqual(len(evidence),4)
            np.testing.assert_array_equal(original["candidate_boxes"][:2],trimmed["candidate_boxes"][:2])
            result,crops = stage.readouts(ident,dict(original=original,trim=trimmed),dict(original=0,trim=0),
                initial,hw,self.ap,0,dict(max_short_side_in_adequate_crop="",good_output="False"),
                dict(original=n,trim=m))
            self.assertEqual(result["changed_slots"],2)
            self.assertEqual(len(crops),8)


if __name__=="__main__":
    unittest.main()
