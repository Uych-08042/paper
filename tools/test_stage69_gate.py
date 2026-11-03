"""Actual paired values versus current shadows, plus immutable real CENTER caches."""
import copy
import csv
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import stage69_gate_common as c
import stage69_gate_audit as a
from test_stage68_advantage import old_pack


def pack():
    def model(target,bias):return dict(target=target,center=[0.]*128,scale=[1.]*128,weights=[0.]*128,bias=bias,pairs=1)
    return dict(source=old_pack(),immediate=model("immediate",.1),future=model("future",-.1),rate_margin=dict(threshold=.005))


class Semantics(unittest.TestCase):
    def test_true_paired_value_not_current_or_an_unobserved_alternative(self):
        self.assertEqual(c.paired_outcome_metrics(.2,.8,"allow"),dict(chosen_mean_iou=.2,gain_vs_veto=-.6000000000000001,oracle_binary_regret=.6000000000000001))
        self.assertEqual(c.paired_outcome_metrics(.2,.8,"veto"),dict(chosen_mean_iou=.8,gain_vs_veto=0.,oracle_binary_regret=0.))
        with self.assertRaisesRegex(ValueError,"No stored future"):c.paired_outcome_metrics(.2,.8,"hann_other")
        stats=c.prediction_stats([.1,-.1,0],[.2,.2,-.1],[1,2,1],1e-6)
        self.assertEqual(stats["positive_sign_accuracy"],.5);self.assertEqual(stats["positive_precision"],1.)
        self.assertAlmostEqual(stats["mae"],.2)
        self.assertIsNone(c.prediction_stats([0],[.1],[1],1e-6)["positive_precision"])
        for w in ([0],[-1],[np.nan]):
            with self.assertRaises(ValueError):c.prediction_stats([0],[.1],w,1e-6)

    def test_causal_scoring_gt_free_support_and_available_but_missed_candidates(self):
        x=np.zeros((3,128));x[:,0]=[0,1,2];m=pack();m["future"]["scale"][1:]=[1e-6]*127
        ref=c.reference(x,[1,1,2],m["future"])
        support=c.support_features(np.array([x[0],x[2]+3]),ref)
        self.assertEqual(support["outside_active"].tolist(),[0.,1.]);self.assertGreater(support["max_abs_z"][1],1e6)
        f=np.zeros((2,4,64));f[:,1,0]=.01
        data=dict(features=np.zeros((2,4,24)),choice_features=f,
            candidate_boxes=np.array([[[0,0,4,4],[20,0,4,4],[20,0,4,4],[20,0,4,4]],
                                      [[20,0,4,4],[0,0,4,4],[20,0,4,4],[20,0,4,4]]],float))
        plan=dict(design=c.c68.DESIGN,controller_design=c.e68.c66.DESIGN)
        initial=c.score(data,m,plan)
        metrics,detail=c.state_metrics(data,dict(ground_truth=[[0,0,4,4]]*3),m,plan,ref)
        self.assertEqual(metrics["immediate"]["q0_good_to_low"].tolist(),[True,False])
        self.assertEqual(metrics["future"]["missed_available"].tolist(),[False,True])
        self.assertEqual(detail["current_Q_advantage"].tolist(),[-1.,1.])
        self.assertEqual(detail["eligible"].tolist(),[True,True])
        c.state_metrics(data,dict(ground_truth=[[80,0,4,4]]*3),m,plan,ref)
        after=c.score(data,m,plan)
        for rule in c.RULES:np.testing.assert_array_equal(initial[0][rule],after[0][rule])
        self.assertEqual(a.first_difference(np.array([[1,2],[3,4]]),np.array([[1,2],[5,4]])),2)
        self.assertIsNone(a.first_difference(np.ones(2),np.ones(2)))

    def test_missing_event_not_replaced_and_uniform_csv(self):
        source=dict(plan=dict(units=[dict(sequence="a",policy="future")]))
        data=dict(frame_id=np.arange(1,7),boxes=np.tile([0.,0.,4.,4.],(6,1)))
        entry=dict(ground_truth=[[0.,0.,4.,4.]]*7)
        specs=(dict(case_id="none",sequence="a",anchor_policy="future",selector="first_failure"),
               dict(case_id="fixed",sequence="a",anchor_policy="future",selector="fixed",frame=2))
        with patch.object(c,"CONFIG",specs),patch.object(c,"read_unit",return_value=(data,entry)):
            rows=c.cases(source)
        self.assertIsNone(rows[0]["frame"]);self.assertEqual(rows[0]["status"],"no_declared_source_event");self.assertEqual(rows[1]["frame"],2)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path=Path(temp)/"table.csv";a.publish_table(path,[dict(case_id="none"),dict(case_id="fixed",frame=2)])
            with path.open(encoding="utf8",newline="") as stream:self.assertEqual(list(csv.DictReader(stream))[1]["frame"],"2")
            payload=path.read_bytes();a.publish_table(path,[dict(case_id="none"),dict(case_id="fixed",frame=2)]);self.assertEqual(path.read_bytes(),payload)
            with self.assertRaisesRegex(ValueError,"differs"):a.publish_table(path,[dict(case_id="other")])


try:
    import torch
    import cv2
except ImportError:torch=cv2=None


@unittest.skipUnless(torch is not None and cv2 is not None,"Real CPU Torch/OpenCV required; skips are not validation")
class Workflow(unittest.TestCase):
    def test_full_center_source_chain_cached_transfer_resume_and_tamper(self):
        from test_stage68_advantage import Workflow as Previous
        import stage68_advantage_summary as s68
        from stage48_stability_runtime import digest
        saved=s68.summarize;saved_evaluate=c.e68.evaluate;captured={}
        def capture(*args,**kwargs):
            captured["observer"]=args[3]
            return saved_evaluate(*args,**kwargs)
        def extend(root68):
            report=saved(root68)
            if captured.get("done"):return report
            captured["done"]=True;root68=Path(root68);root69=root68.parent/"stage69"
            plan,_=c.c68.load_plan(root68);seq=plan["entries"][0]["sequence"]
            before={str(p):p.read_bytes() for p in root68.parent.rglob("*") if p.is_file()}
            observer=captured["observer"];state=digest(observer.rt.net)
            specs=(dict(case_id="micro",sequence=seq,anchor_policy="future",selector="fixed",frame=2),)
            # The source fixture really used Torch/CV2 for CENTER observations;
            # this diagnostic must work after those libraries are made unavailable.
            with patch.object(c,"CONFIG",specs),patch.dict(c.DESIGN,before=1,after=1), \
                 patch.dict("sys.modules",torch=None,cv2=None), \
                 patch.object(observer,"capture",side_effect=AssertionError("No new inference")), \
                 patch.object(c.e68,"checked_image",side_effect=AssertionError("No new image reads")), \
                 patch.object(c.e68.e67,"checked_image",side_effect=AssertionError("No new image reads")):
                result=a.summarize(root68,root69);self.assertEqual(result,a.summarize(root68,root69))
                self.assertEqual(result["videos"],2);self.assertEqual(result["source_fit_pairs"],2)
                self.assertEqual(result["transferred_paired_starts"],1);self.assertEqual(result["source_frames"],56)
                for k in ("new_models","new_image_reads","new_search_calls","new_full_trajectories","new_counterfactual_branches"):self.assertEqual(result[k],0)
                self.assertFalse(result["training_performed"]);self.assertFalse(result["method_adoption_gate"])
                def read(name):
                    with (root69/"analysis"/("stage69_"+name+".csv")).open(encoding="utf8",newline="") as stream:return list(csv.DictReader(stream))
                self.assertEqual(len(read("same_state_summary")),49);self.assertEqual(len(read("matched_summary")),21)
                self.assertEqual(len(read("focus")),7);self.assertEqual(len(read("feature_shift")),14)
                pair=read("paired_case")[0];actions=read("paired_policy")
                for row in actions:
                    value=float(pair[row["action"]+"_mean_iou_Q"])
                    self.assertAlmostEqual(float(row["chosen_mean_iou"]),value)
                source=c.load_source(root68);refs,_,_,_=a.fit_diagnostics(source)
                transfer,_=a.paired_transfer(source,refs)
                self.assertAlmostEqual(float(pair["future_observed_Q"]),transfer[0]["future_observed_Q"])
                self.assertTrue(all(Path(p).read_bytes()==b for p,b in before.items()))
                p=root69/"analysis/stage69_paired_case.csv";payload=p.read_bytes();p.write_bytes(payload+b"tamper")
                with self.assertRaisesRegex(ValueError,"differs"):a.publish_table(p,transfer)
                p.write_bytes(payload)
                for p in (root68/"analysis/stage68_scope.csv",root68/"models/training.json",
                          c.c68.eval_path(root68,plan["units"][0])):
                    payload=p.read_bytes();p.write_bytes(payload+b"tamper")
                    with self.assertRaises((ValueError,AssertionError)):c.load_source(root68)
                    p.write_bytes(payload)
            self.assertEqual(digest(observer.rt.net),state);return report
        with patch.object(c.e68,"evaluate",side_effect=capture),patch.object(s68,"summarize",side_effect=extend):
            Previous().test_real_fit_isolation_paired_calls_controls_resume_and_immutability()
        self.assertTrue(captured.get("done"))


if __name__=="__main__":unittest.main()
