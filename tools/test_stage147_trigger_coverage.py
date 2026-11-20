"""Verify condition attribution and frame-level counting on overlapping opportunities."""
import unittest

import stage147_trigger_coverage as stage


class TriggerCoverage(unittest.TestCase):
    def test_two_candidates_must_not_jointly_satisfy_one_candidate_rule(self):
        states = [dict(state_id=0, state="native", native_iou=.1, grid_iou=.8, onset=True,
                       dense_only_good=True, stable_window=0, relapse=False),
                  dict(state_id=1, state="native", native_iou=.8, grid_iou=.9, onset=False,
                       dense_only_good=False, stable_window=1, relapse=False)]

        def record(state, gap, overlap, good=False, bad=False, recent=True):
            return dict(state_id=state, source=str(state), quality_gap=gap, box_overlap=overlap,
                        original_exclusion=stage.blocked_by(gap, overlap), recent_color=recent, template_gray=True,
                        new_good=good, stable_bad=bad, stable_severe=bad)

        candidates = [record(0,.1,.3,good=True,recent=False), record(0,.2,.1,good=True),
                      record(1,.3,.1,bad=True), record(1,.4,.2,bad=True)]
        coverage, by_reason = stage.summarize(states, candidates)
        rows = {(r["population"],r["gap_limit"],r["overlap_limit"],r["appearance"]):r for r in coverage}
        self.assertEqual(rows["native_onsets",.15,.25,"none"]["new_good_states"], 0)
        self.assertEqual(rows["native_onsets",.15,None,"none"]["new_good_states"], 1)
        self.assertEqual(rows["native_onsets",.15,None,"both"]["new_good_states"], 0)
        self.assertEqual(rows["native_onsets",None,.25,"both"]["new_good_states"], 1)
        self.assertEqual(rows["native_onsets",None,None,"none"]["new_good_candidates"], 2)
        self.assertEqual(rows["native_stable_windows",None,None,"none"]["stable_severe_candidates"], 2)
        self.assertEqual(rows["native_stable_windows",None,None,"none"]["stable_severe_states"], 1)
        excluded = {r["original_exclusion"]:r["new_good_candidates"] for r in by_reason if r["population"]=="native_onsets"}
        self.assertEqual(excluded, {"gap_only":1,"overlap_only":1,"both":0,"neither":0})

    def test_original_boundaries_are_inclusive(self):
        self.assertEqual(stage.blocked_by(.15,.25), "neither")
        self.assertEqual(stage.blocked_by(.150001,.25), "gap_only")
        self.assertEqual(stage.blocked_by(.15,.250001), "overlap_only")
        self.assertEqual(stage.blocked_by(.150001,.250001), "both")


if __name__ == "__main__":
    unittest.main()
