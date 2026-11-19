"""Check first-event attribution with matched and already diverged histories."""
import unittest

import numpy as np

import stage132_first_divergence as stage
from stage27_uav_common import crop_geometry, tracking_iou


class FirstDivergence(unittest.TestCase):
    def setUp(self):
        self.n = 8
        self.hw = (120,160)
        self.ap = dict(search_factor=4.,search_size=256,design={"history":5})
        self.parents = np.array([[60.,60.,8.,8.],[20.,20.,8.,8.],[100.,60.,8.,8.],[30.,80.,8.,8.]])
        self.gt = self.parents[1]
        self.entry = dict(ground_truth=np.tile(self.gt,(self.n+1,1)))
        self.ids = dict(sequence="fixture",seed=2312,reference_arm="alternate_shared",treatment_arm="trim",focus=False)

    def make_trace(self):
        crops = np.array([crop_geometry(p,self.hw,4.,256)[0] for p in self.parents])
        return dict(frame_id=np.arange(1,self.n+1),image_hw=np.tile(self.hw,(self.n,1)),
            candidate_parents=np.tile(self.parents,(self.n,1,1)),untrimmed_parents=np.tile(self.parents,(self.n,1,1)),
            candidate_boxes=np.tile(self.parents,(self.n,1,1)),boxes=np.tile(self.gt,(self.n,1)),scores=np.tile([.30,.70,.2,.1],(self.n,1)),
            selected=np.ones(self.n,int),forecast_x=np.zeros((self.n,32)),features=np.zeros((self.n,4,24)),
            crop=np.tile(crops,(self.n,1,1)),alternate_requested=np.zeros(self.n,bool),alternate_used=np.zeros(self.n,bool))

    def diagnose(self,a,b):
        auc = [stage.success_auc(tracking_iou(d["boxes"],self.gt)) for d in (a,b)]
        return stage.diagnose(a,b,self.entry,self.ap,self.ids,auc)

    def test_same_box_can_change_request_before_any_output_changes(self):
        a,b = self.make_trace(),self.make_trace()
        b["scores"][1,0] = .68
        b["alternate_requested"][2] = True
        b["alternate_used"][2] = True
        result,events,_ = self.diagnose(a,b)
        self.assertEqual(result["first_request_frame"],2)
        self.assertEqual(result["first_incoming_state_frame"],3)
        self.assertIsNone(result["first_output_frame"])
        self.assertEqual(result["matched_same_output_request_changes"],1)
        row = next(r for r in events if r["frame"]==2)
        self.assertEqual(row["mechanism"],"same_box_request_changed")
        self.assertTrue(row["matched_before_search"])

    def test_good_supply_loss_and_later_unmatched_miss_are_kept_separate(self):
        a,b = self.make_trace(),self.make_trace()
        b["candidate_boxes"][2,1] = self.parents[0]
        b["boxes"][2] = self.parents[0]
        b["crop"][2,1,2:] += 1
        b["scores"][3] = [.1,.2,.7,.1]
        b["selected"][3],b["boxes"][3] = 2,self.parents[2]
        result,events,_ = self.diagnose(a,b)
        self.assertEqual(result["first_layout_frame"],3)
        self.assertEqual(result["first_matched_good_supply_lost_frame"],3)
        self.assertIsNone(result["first_matched_missed_good_frame"])
        self.assertEqual(result["first_missed_good_frame"],4)
        self.assertEqual(next(r["mechanism"] for r in events if r["frame"]==4),
                         "history_or_proposal_already_different")
        self.assertEqual(next(r["mechanism"] for r in events if r["frame"]==3),"good_candidate_lost")

    def test_unchanged_boxes_can_rerank_after_a_search_change(self):
        a,b = self.make_trace(),self.make_trace()
        b["scores"][0] = [.1,.2,.8,.1]
        b["selected"][0],b["boxes"][0] = 2,self.parents[2]
        b["crop"][0,3,:2] += 2
        result,events,_ = self.diagnose(a,b)
        self.assertEqual(result["first_matched_missed_good_frame"],1)
        self.assertEqual(events[0]["mechanism"],"good_candidate_missed")
        self.assertTrue(events[0]["retained_reranked"])


if __name__=="__main__":
    unittest.main()
