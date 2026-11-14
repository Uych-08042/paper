"""Verify candidate coverage, frozen pair equivalence and actual full-run call budgets."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage103_candidate_verification as stage
import test_stage100_online_scale as previous
from test_stage97_trusted_scale import choose


class Verification(unittest.TestCase):
    setUp = previous.OnlineScale.setUp

    def decision_fixture(self):
        features=np.zeros((4,len(stage.e63.source_common.prior.FEATURES)))
        features[:,stage.e63.source_common.prior.FEATURES.index("hann_peak")]=[.90,.50,.89,.89]
        row=dict(selected=2,features=features,history_latest=np.array([80.,80.,20.,20.]),
            candidate_boxes=np.array([[80.,80.,20.,20.],[81.,80.,20.,20.],[200.,200.,20.,20.],[82.,80.,20.,20.]]))
        row["candidate_parents"]=row["candidate_boxes"].copy()
        x=np.array([.1,1.,0.,.05]);queries=np.c_[x,np.sqrt(1-x*x)]
        return row,queries,np.array([1.,0.]),[(1,np.array([1.,0.])),(4,np.array([1.,0.]))]

    def test_hidden_good_candidate_coverage_and_distinct_domain_control(self):
        row,q,anchor,memory=self.decision_fixture()
        pair=stage.decide(row,q,anchor,list(memory),7,"pair",stage.DESIGN)
        four=stage.decide(row,q,anchor,list(memory),7,"four_matched",stage.DESIGN)
        per=stage.decide(row,q,anchor,list(memory),7,"four_pairwise",stage.DESIGN)
        self.assertEqual(pair["selected"],0)
        self.assertEqual(four["selected"],1)
        self.assertFalse(per["support_mask"][1])
        self.assertEqual(per["selected"],0)
        self.assertEqual(pair["used_history"],four["used_history"])
        np.testing.assert_array_equal(pair["refs"],four["refs"])
        # All four candidates and references are fixed before GT is ever evaluated.
        self.assertEqual(row["selected"],2)
        with self.assertRaises(ValueError):
            stage.decide(row,q,anchor,[(7,anchor)],7,"four_matched",stage.DESIGN)

    def test_exact_tie_finite_candidates_and_missing_pair_support(self):
        row,q,anchor,memory=self.decision_fixture()
        q[1]=q[0]
        self.assertEqual(stage.decide(row,q,anchor,list(memory),7,"four_matched",stage.DESIGN)["selected"],0)
        q[1]=np.nan
        d=stage.decide(row,q,anchor,list(memory),7,"four_matched",stage.DESIGN)
        self.assertFalse(d["support_mask"][1]);self.assertEqual(d["selected"],0)
        q[0]=np.nan;q[1]=[1.,0.]
        d=stage.decide(row,q,anchor,list(memory),7,"four_matched",stage.DESIGN)
        self.assertEqual(d["selected"],2)
        self.assertEqual(int(d["support_mask"].sum()),1)

    def baselines(self):
        with (patch.object(stage.s93,"checked_image",side_effect=self.reader),
              patch.object(stage.s93.s85,"checked_image",side_effect=self.reader),
              patch.object(stage.s86,"checked_image",side_effect=self.reader)):
            native=stage.s93.s85.collect(self.observer,self.entry,"native_sgla",self.ap,None,{})
            ppo=stage.s86.collect(self.observer,self.entry,np.asarray([self.entry["initial_box"]]),np.arange(1,13),0.,self.ap,choose)
            pair=stage.s93.collect(self.observer,self.entry,self.ap,choose,"gated",stage.DESIGN)
        pulse=stage.prior.s100.collect(self.observer,self.entry,self.ap,choose,"pulse",stage.prior.s100.DESIGN,image_reader=self.reader)
        return native,dict(ppo=ppo,gated=pair),pulse

    def test_full_collection_pair_equivalence_own_history_and_gt_isolation(self):
        _,refs,_=self.baselines()
        old=stage.convert_pair(refs["gated"],self.entry,self.old,choose,stage.DESIGN)
        for mode in stage.MODES:
            before=self.observer.calls
            data=stage.collect(self.observer,self.entry,self.old,choose,mode,stage.DESIGN,image_reader=self.reader)
            self.assertEqual(self.observer.calls-before,48)
            stage.audit(data,self.entry,self.old,choose,mode,stage.DESIGN)
            changed=dict(self.entry,ground_truth=np.full((13,4),999.))
            repeat=stage.collect(self.observer,changed,self.old,choose,mode,stage.DESIGN,image_reader=self.reader)
            for key in stage.FIELDS:np.testing.assert_array_equal(data[key],repeat[key])
            np.testing.assert_array_equal(data["history_latest"][1:],data["boxes"][:-1])
            self.assertEqual(data["all_queries"].shape,(12,4,2))
            if mode=="pair":
                for key in stage.s86.FIELDS+tuple(k for k in stage.EXTRA if k not in ("all_queries","scores","gains")):
                    np.testing.assert_array_equal(data[key],old[key])
            corrupt={k:v.copy() for k,v in data.items()};corrupt["refs"][5,0]=6
            with self.assertRaises(AssertionError):stage.audit(corrupt,self.entry,self.old,choose,mode,stage.DESIGN)

    def test_complete_mixed_cohort_both_cards_budget_resume_and_tamper(self):
        native,refs,pulse=self.baselines()
        unit=dict(sequence="test",seed=2171,origin="native",policy="ppo",fold=0)
        specs=[dict(unit=unit,cohort="historical"),dict(unit=dict(unit,seed=2172),cohort="historical"),
               dict(unit=dict(unit,sequence="other"),cohort="expanded")]
        points=[dict(sequence="test",seed=2171,frame=5,roles="first_pulse"),
                dict(sequence="other",seed=2171,frame=7,roles="first_observed_harm")]
        source=dict(original=self.old,report=dict(native_head_sha256="native"))
        actor=dict(models={},prior={})
        original_collect,original_point=stage.collect,stage.matched_point
        for reuse in (True,False):
            with self.subTest(reuse=reuse),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                calls=(288 if reuse else 600)+8
                plan=stage.io.seal(dict(schema=stage.SCHEMA,design=dict(stage.DESIGN,bootstrap_samples=16),
                    units=specs,points=points,runtime={},runtime_differences=[],reuse_sources=reuse,
                    trajectories=14,new_trajectories=6 if reuse else 14,new_search_calls=calls,
                    source_sha256={},code_sha256={}))
                stage.c.publish_identical(root/"manifest.json",plan)
                args=(root,root,{},root,root,{}, {},{},source,actor,self.observer,plan)
                before=self.observer.calls
                with (patch.object(stage,"load_source_unit",side_effect=lambda *a:(dict(self.entry,sequence=a[-1]["unit"]["sequence"]),refs,pulse)),
                      patch.object(stage,"load_native",side_effect=lambda *a:(dict(a[-2]["unit"],seed=0,policy="q0"),native)),
                      patch.object(stage.e74,"learned_pack"),patch.object(stage.e74,"choose",side_effect=lambda r,h,*_:choose(r,h)),
                      patch.object(stage.e63.source_experiment,"installed",side_effect=lambda *_:contextlib.nullcontext("native")),
                      patch.object(stage.s93.s85,"checked_image",side_effect=self.reader),patch.object(stage.s86,"checked_image",side_effect=self.reader),
                      patch.object(stage,"collect",side_effect=lambda *a:original_collect(*a,image_reader=self.reader)),
                      patch.object(stage,"matched_point",side_effect=lambda *a:original_point(*a,image_reader=self.reader)),
                      contextlib.redirect_stdout(io.StringIO())):
                    stage.run(*args)
                    self.assertEqual(self.observer.calls-before,calls)
                    with patch.object(self.observer,"capture",side_effect=AssertionError("Repeated inference")):stage.run(*args)
                stage.check_results(root)
                report=stage.c.c66.sealed(root/"analysis/stage103_decision.json",stage.SCHEMA)
                self.assertEqual(report["new_full_trajectories"],6 if reuse else 14)
                self.assertEqual(report["maximum_source_control_box_error_px"],0.)
                next((root/"points").glob("*.npz")).write_bytes(b"tampered")
                with self.assertRaises(ValueError):stage.check_results(root)

    def test_directional_statistics_do_not_treat_more_failures_as_wins(self):
        result=stage.contrast([1.,2.,-1.],"x","failure_events",dict(stage.DESIGN,bootstrap_samples=16))
        self.assertEqual(result["favorable_direction"],"negative")
        self.assertEqual(result["wins"],1);self.assertEqual(result["losses"],2)
        self.assertEqual(result["worst_video_delta"],2.)


if __name__=="__main__":unittest.main()
