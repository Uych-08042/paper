"""Verify one-step intervention, cached control and factorial accounting."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage126_factorial_continuation as stage
import test_stage113_frozen_closed_loop as fixture


class FactorialContinuation(unittest.TestCase):
    setUp = fixture.FrozenClosedLoop.setUp

    def test_only_allowed_nonduplicate_crop_changes(self):
        boxes = np.array([[10., 10., 10., 10.], [60., 10., 10., 10.],
                          [23., 10., 10., 10.], [40., 10., 10., 10.]])
        parents = np.array([boxes[1], [72., 10., 10., 10.], [84., 10., 10., 10.], [96., 10., 10., 10.]])
        row = {name: None for name in stage.prior.FIELDS}
        row["candidate_boxes"] = boxes
        with (patch.object(stage.base,"write_eligible",return_value=False),
              patch.object(stage.base,"forecast_parents",side_effect=lambda *_:dict(candidate_parents=parents.copy()))):
            variants = stage.variants(row,[0,1,1], [boxes[0]],(128,128),self.ap,None)
            self.assertEqual(len(variants),4)
            off, on = (variants[1, allow] for allow in (False,True))
            np.testing.assert_array_equal(off[0]["boxes"],on[0]["boxes"])
            self.assertFalse(off[1]["alternate_requested"])
            self.assertTrue(on[1]["alternate_used"])
            np.testing.assert_array_equal(on[1]["candidate_parents"][:3],parents[:3])
            np.testing.assert_array_equal(on[1]["candidate_parents"][3],boxes[0])
            self.assertFalse(variants[0,True][1]["alternate_requested"])
            parents[3] = boxes[0]
            variants = stage.variants(row,[1],[boxes[0]],(128,128),self.ap,None)
            self.assertTrue(variants[1,True][1]["alternate_requested"])
            self.assertFalse(variants[1,True][1]["alternate_used"])

    def test_roll_point_reuses_control_and_replays_only_required_counterfactuals(self):
        prior = stage.prior
        generator = self.package["generator"]
        entry = self.entry
        trace = stage.s121.collect(self.observer,entry["frames"],entry["initial_box"],self.old,
                                   generator,self.model,image_reader=self.reader)
        frame = 4
        history = list(prior.prior.history(trace,entry["initial_box"],frame-1,self.ap["design"]["history"]))
        start = prior.snapshot(trace,frame-1)
        scores, q = start["scores"], int(start["selected"])
        alternative = next(j for j in range(4) if j!=q)
        score = np.array([.1,.1,.1,.1])
        score[alternative] = .9
        actions = {("recorded",0):(scores,q)} | {
            (arm,seed):(score,alternative) for arm in prior.source.ARMS for seed in prior.SEEDS}
        point = dict(sequence=entry["sequence"],source_seed=2312,frame=frame)
        ident = dict(point,plan_fingerprint="fixture")
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(prior,"DESIGN",dict(prior.DESIGN,future_frames=4)),
              patch.object(stage,"DESIGN",dict(stage.DESIGN,future_frames=4))):
            root = Path(directory)
            source = root/"source"
            source_fingerprint = "stage125-fixture"
            natural = {k:prior.commit(start,*v) for k,v in actions.items()}
            continuations = {prior.continuation_key(natural["recorded",0]):
                             {k:trace[k][frame:frame+4] for k in prior.FIELDS}}
            alternate_start = natural["independent",prior.SEEDS[0]]
            alternate_key = prior.continuation_key(alternate_start)
            data = prior.rollout(self.observer,entry["frames"],history,alternate_start,self.ap,
                                 generator,self.model,self.reader,lambda:None)
            continuations[alternate_key] = data
            stage.c.save(source/"branches"/f"{point['sequence']}__s2312__f{frame:06d}__q{alternate_key[0]}__a{int(alternate_key[1])}.npz",
                         data,dict(ident,plan_fingerprint=source_fingerprint,action=alternate_key[0],next_request=alternate_key[1]))
            gt = np.asarray(entry["ground_truth"])[frame:frame+5]
            rows = {k:dict(prior.outcome(s,continuations[prior.continuation_key(s)],gt),
                          action=str(s["selected"]),next_alternate_requested=str(prior.continuation_key(s)[1]))
                    for k,s in natural.items()}
            for r in rows.values():
                if r["first_sustained_good_offset"] is None:
                    r["first_sustained_good_offset"] = ""
            branches = {k:{"cached_future":str(k!=alternate_key)} for k in continuations}
            arguments = (root,source,dict(fingerprint=source_fingerprint),dict(decisions=rows,branches=branches),
                         point,entry,trace,history,start,actions,self.observer,self.ap,generator,self.model,
                         self.reader,lambda:None)
            previous_calls = self.observer.calls
            cells, effects, coverage = stage.roll_point(*arguments,{},dict(fingerprint="fixture",runtime_differences=[]),0)
            self.assertEqual(len(cells),4)
            self.assertEqual(len(effects),8)
            self.assertEqual(coverage["actual_actions"],2)
            self.assertEqual(self.observer.calls-previous_calls,sum(r["new_branch_search_calls"] for r in cells))
            self.assertIn("stage125_original",[r["provenance"] for r in cells])
            self.assertIn("stage125_branch",[r["provenance"] for r in cells])
            parsed = stage.summarize(effects)
            self.assertTrue(parsed)
            for row in effects:
                self.assertAlmostEqual(row["natural_total"], row["commit_at_disabled"]+row["natural_request_contribution"])
            previous_calls = self.observer.calls
            fresh, _, new_coverage = stage.roll_point(*arguments,{},dict(fingerprint="fresh",runtime_differences=[{}]),0)
            self.assertEqual(self.observer.calls-previous_calls,sum(r["new_branch_search_calls"] for r in fresh))
            self.assertEqual(new_coverage["physical_branches"],coverage["physical_branches"])
            for fresh_cell, prior_cell in zip(fresh,cells):
                self.assertNotIn(fresh_cell["provenance"],("stage125_original","stage125_branch"))
                for key in ("current_iou", "future_mean_iou", "future_low_frames"):
                    self.assertEqual(fresh_cell[key],prior_cell[key])

    def test_factorial_difference_of_differences(self):
        effects = stage.signed_effects(.1,.2,.3,.6,.2,.6)
        self.assertAlmostEqual(effects["commit_at_disabled"],.2)
        self.assertAlmostEqual(effects["allow_on_original"],.1)
        self.assertAlmostEqual(effects["allow_on_retrained"],.3)
        self.assertAlmostEqual(effects["interaction"],.2)
        self.assertAlmostEqual(effects["natural_request_contribution"],.2)
        self.assertAlmostEqual(effects["natural_total"],.4)

    def test_budget_excludes_q0_requests_and_already_recorded_branches(self):
        decisions = {("recorded",0):dict(action="0"),("independent",2312):dict(action="2")}
        indexed = dict(decisions={"point":decisions},branches={"point":{(0,False):{},(2,True):{}}})
        horizon = stage.DESIGN["future_frames"]
        self.assertEqual(stage.search_ceiling(indexed,False),4*horizon)
        self.assertEqual(stage.search_ceiling(indexed,True),7*4*horizon+4)


if __name__ == "__main__":
    unittest.main()
