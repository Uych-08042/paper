"""Exercise active zoom, own history, and matched full-sequence reporting."""
import unittest
from unittest.mock import patch

import numpy as np

import stage131_zoom_closed_loop as stage
import test_stage113_frozen_closed_loop as fixture


class ZoomClosedLoop(unittest.TestCase):
    setUp = fixture.FrozenClosedLoop.setUp

    def test_without_alternate_trigger_layout_never_observes_a_wide_response(self):
        hw = (200,400)
        ap = {"search_factor":4.,"search_size":256}
        originals = np.array([stage.coarse.parent_at(c,s,4.) for c,s in
                              [([120,100],44),([200,100],900),([140,100],44),([160,100],44)]])
        def forbidden(*_):
            self.fail("Read a response without an active alternate trigger")
        parents,trimmed,slot,wide,cell = stage.layout(dict(candidate_parents=originals,alternate_used=False),
            hw,ap,"rank1",forbidden)
        self.assertEqual((slot,wide,cell),(-1,-1,-1))
        self.assertFalse(np.array_equal(originals,parents))
        np.testing.assert_array_equal(parents,trimmed)

    def test_every_trajectory_starts_from_frame_one_keeps_q3_and_stays_within_budget(self):
        reference = stage.s121.forecast
        def activation(history,previous,hw,ap,generator):
            result = reference(history,previous,hw,ap,generator)
            if previous is not None:
                result["candidate_parents"][1] = stage.coarse.parent_at(np.array(hw[::-1])/2,3*max(hw),ap["search_factor"])
                result.update(alternate_used=True,alternate_requested=True)
            return result

        def choose(row, *_):
            return np.array([.0,.2,1.,.3]),2

        generator = self.package["generator"]
        traces = {}
        with (patch.object(stage.s121,"forecast",side_effect=activation),patch.object(stage.base,"choose",side_effect=choose)):
            for arm in ("trim","rank1"):
                before = self.observer.calls
                data = stage.collect(self.observer,self.entry,self.old,arm,generator,self.model,self.reader)
                stage.audit(data,self.entry,self.old,arm,generator,self.model)
                traces[arm] = data
                np.testing.assert_array_equal(data["frame_id"],np.arange(1,len(self.entry["frames"])))
                np.testing.assert_array_equal(data["history_latest"][1:],data["boxes"][:-1])
                np.testing.assert_array_equal(data["candidate_parents"][:,3],data["trimmed_parents"][:,3])
                self.assertEqual(self.observer.calls-before,int(data["unique_search_calls"].sum()))
                self.assertLessEqual(int(data["unique_search_calls"].max()),4)
            self.assertEqual(stage.prefix(traces["rank1"],traces["trim"]),2)
            self.assertTrue(np.any(traces["rank1"]["boxes"][1:]!=traces["trim"]["boxes"][1:]))
            changed = dict(self.entry,ground_truth=np.full_like(self.entry["ground_truth"],999))
            repeated = stage.collect(self.observer,changed,self.old,"rank1",generator,self.model,self.reader)
            for key in stage.FIELDS:
                np.testing.assert_array_equal(traces["rank1"][key],repeated[key])

    def test_reports_full_paired_video_and_controls(self):
        generator = self.package["generator"]
        native = stage.base.collect(self.observer,self.entry,self.old,"native",None,None,image_reader=self.reader)
        alternate = stage.s121.collect(self.observer,self.entry["frames"],self.entry["initial_box"],self.old,
                                      generator,self.model,image_reader=self.reader)
        trim = stage.collect(self.observer,self.entry,self.old,"trim",generator,self.model,self.reader)
        rank1 = stage.collect(self.observer,self.entry,self.old,"rank1",generator,self.model,self.reader)
        traces = {("native",0):native}
        for arm,data in (("alternate_shared",alternate),("trim",trim),("rank1",rank1)):
            traces.update({(arm,seed):data for seed in stage.base.DESIGN["seeds"]})
        jobs = [dict(arm=a,seed=s,outer_fold=0,spec=dict(cohort="fixture")) for a,s in traces]
        tables = {name:[] for name in stage.TABLES}
        stage.report_video(tables,jobs,traces,self.entry,True)
        self.assertEqual(len(tables["scope"]),7)
        self.assertEqual(sum(r["provenance"]=="new" for r in tables["scope"]),4)
        summary,_,by_seed,protection = stage.summarize(tables["seed_video"],tables["recovery"])
        self.assertEqual(len(summary),4)
        self.assertEqual(set(by_seed),{"2312","2313"})
        self.assertIn("trim",protection)


if __name__=="__main__":
    unittest.main()
