"""Check the four-slot protection rule and the pre-search location vote."""
import unittest

import numpy as np

import stage130_preserve_q3 as stage


class PreserveQ3(unittest.TestCase):
    def test_spares_q0_q3_and_the_wide_slot(self):
        self.assertEqual([stage.replacement_slot(w) for w in range(4)],[2,2,1,2])

    def test_vote_only_uses_pre_zoom_observations(self):
        control = dict(candidate_parents=np.arange(16,dtype=float).reshape(4,4),
                       candidate_boxes=np.array([[48.,38.,4.,4.],[4.,4.,4.,4.],[5.,5.,4.,4.],[50.,38.,4.,4.]]),
                       features=np.arange(96,dtype=float).reshape(4,24))
        zoom = {k:np.array(v,copy=True) for k,v in control.items()}
        zoom["candidate_parents"][2:] = [stage.source.parent_at(c,20,4.) for c in ([19,19],[50,40])]
        hw = (100,100)
        ap = {"search_factor":4.,"search_size":256}
        results = stage.finite_choices(control,zoom,1,[2,3],hw,ap)
        self.assertEqual(results["coverage_vote"][1:],(2,[0,2]))
        for bank,_,_ in results.values():
            for key in bank:
                np.testing.assert_array_equal(bank[key][[0,1,3]],control[key][[0,1,3]])
        zoom["candidate_boxes"][:] = 90.
        self.assertEqual(stage.finite_choices(control,zoom,1,[2,3],hw,ap)["coverage_vote"][1:],(2,[0,2]))
        self.assertEqual(stage.finite_choices(control,zoom,1,[],hw,ap)["rank1"][1:],(None,[]))

    def test_tied_votes_use_first_response(self):
        boxes = np.tile([70.,70.,10.,10.],(4,1))
        parents = np.tile([13.,13.,5.,5.],(4,1))
        control = dict(candidate_parents=parents,candidate_boxes=boxes,features=np.zeros((4,24)))
        self.assertEqual(stage.finite_choices(control,control,1,[2,3],(128,128),
            {"search_factor":4.,"search_size":256})["coverage_vote"][1:],(1,[0,0]))


if __name__=="__main__":
    unittest.main()
