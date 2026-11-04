"""Current candidate versus history support, terminal targets, and immutable CPU audit."""
import copy
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import stage72_ppo_common as c
import stage72_ppo_audit as a


def controller_pack(sign):
    state={}
    for branch in ("actor","critic"):
        w=np.zeros((2,128));w[0,0]=1.
        state.update({branch+".0.weight":w.tolist(),branch+".0.bias":[0.,0.],
                      branch+".2.weight":[[sign if branch=="actor" else 0.,0.]],branch+".2.bias":[0.]})
    return dict(state=state,norm=dict(center=[0.]*128,scale=[1.]*128),rate_margin=dict(threshold=.005))


def fixture():
    f=np.zeros((4,4,64));f[:,1,0]=[1.,-1.,1.,1.]
    boxes=np.tile([20.,0.,4.,4.],(4,4,1));boxes[0,0]=[0.,0.,4.,4.];boxes[1,1]=[0.,0.,4.,4.]
    boxes[2]=[0.,0.,4.,4.];boxes[3,1]=[40.,0.,4.,4.];boxes[3,2]=[0.,0.,4.,4.]
    return dict(choice_features=f,candidate_boxes=boxes,quality_choice=np.ones(4,int),quality_gain=np.full(4,.01)),dict(ppo=controller_pack(1.),myopic=controller_pack(-1.))


class Semantics(unittest.TestCase):
    def test_GT_free_same_state_selection_separates_binary_and_public_support(self):
        data,pack=fixture();scored=c.score(data,pack,c.c71.DESIGN)
        self.assertEqual(scored["eligible"].tolist(),[True,True,False,True])
        self.assertEqual(scored["choices"]["ppo"].tolist(),[1,0,0,1]);self.assertEqual(scored["choices"]["myopic"].tolist(),[0,1,0,0])
        entry=dict(ground_truth=[[0.,0.,4.,4.]]*5)
        metrics,detail=c.label_states(data,entry,scored)
        self.assertEqual(metrics["ppo"]["q0_good_to_low"].tolist(),[True,False,False,False])
        self.assertEqual(metrics["ppo"]["missed_binary_good"].tolist(),[True,True,False,False])
        self.assertEqual(metrics["ppo"]["missed_public_good"].tolist(),[True,True,False,True])
        self.assertEqual(metrics["ppo"]["binary_current_regret"].tolist(),[1.,1.,0.,0.])
        self.assertEqual(metrics["myopic"]["selected_iou"].tolist(),[1.,1.,1.,0.])
        self.assertTrue(all(np.all(v["public_current_regret"]>=0) for v in metrics.values()))
        c.label_states(data,dict(ground_truth=[[80.,0.,4.,4.]]*5),scored)
        after=c.score(data,pack,c.c71.DESIGN)
        for rule in c.RULES:np.testing.assert_array_equal(scored["choices"][rule],after["choices"][rule])
        zero=controller_pack(0.);all_ties=c.controller(scored["x"],scored["eligible"],zero,c.c71.DESIGN)
        self.assertFalse(all_ties["accept"].any())
        with self.assertRaises(ValueError):c.controller(np.zeros((3,2)),np.ones(3,bool),zero,c.c71.DESIGN)

    def test_discounted_behavior_GAE_and_zero_variance_diagnostics(self):
        np.testing.assert_array_equal(c.discounted_return([0.,0.,1.],.5),[.25,.5,1.])
        self.assertEqual(c.explained_variance([1.,2.,3.],[1.,2.,3.]),1.)
        self.assertEqual(c.explained_variance([0.,0.,0.],[1.,2.,3.]),0.)
        self.assertIsNone(c.explained_variance([1.,2.],[0.,0.]))
        data,pack=fixture();p=pack["ppo"]
        x=c.score(data,pack,c.c71.DESIGN)["x"]
        metrics=a.episode_metrics(dict(x=x,eligible=np.zeros(4,bool),action=np.zeros(4,int),log_prob=np.zeros(4),
            reward=np.zeros(4),value=np.zeros(4),myopic_label=np.zeros(4)),p,p,c.c71.DESIGN,"ppo",{})
        self.assertEqual(metrics["actor_parameter_step_l2"],0.)
        self.assertIsNone(metrics["before_mean_probability"])
        self.assertIsNone(metrics["post_episode_behavior_ratio_clip_fraction"])
        self.assertIsNone(metrics["before_value_behavior_MC_explained_variance"])
        self.assertEqual(metrics["after_value_fixed_GAE_mse"],0.)

    def test_video_first_seed_aggregation_missing_event_and_stable_export(self):
        rows=[dict(sequence="a",provider="ppo",metric=0.),dict(sequence="a",provider="ppo",metric=1.),
              dict(sequence="b",provider="ppo",metric=1.)]
        videos=c.aggregate(rows,("sequence","provider"),("metric",));macro=a.macro(videos,("provider",),("metric",))
        self.assertEqual(macro[0]["metric"],.75);self.assertEqual(macro[0]["videos"],2)
        source=dict(plan=dict(units=[dict(sequence="a",origin="native",policy="ppo",seed=2171)]))
        data=dict(frame_id=np.arange(1,5),boxes=np.tile([0.,0.,4.,4.],(4,1)))
        specs=(dict(case_id="none",sequence="a",policy="ppo",seed=2171,selector="first_failure"),)
        with patch.object(c,"CONFIG",specs),patch.object(c,"read_unit",return_value=(data,dict(ground_truth=[[0.,0.,4.,4.]]*5))):
            anchors=c.resolve_cases(source)
        self.assertIsNone(anchors[0]["frame"]);self.assertEqual(anchors[0]["status"],"no_declared_event")
        self.assertIsNone(a.first_difference(np.ones(3),np.ones(3)))
        self.assertEqual(a.first_difference(np.array([1,2]),np.array([1,3])),2)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path=Path(temp)/"table.csv";a.publish_table(path,[dict(case_id="none"),dict(case_id="fixed",frame=2)])
            with path.open(encoding="utf-8",newline="") as f:self.assertEqual(list(csv.DictReader(f))[1]["frame"],"2")
            original=path.read_bytes();a.publish_table(path,[dict(case_id="none"),dict(case_id="fixed",frame=2)]);self.assertEqual(path.read_bytes(),original)
            with self.assertRaises(ValueError):a.publish_table(path,[dict(case_id="other")])


try:
    import torch
    import cv2
except ImportError:torch=cv2=None


@unittest.skipUnless(torch is not None and cv2 is not None,"Real CENTER source fixture requires CPU Torch/OpenCV; skips do not validate")
class Workflow(unittest.TestCase):
    def test_full_real_source_cached_audit_no_training_images_torch_or_inference(self):
        from test_stage71_ppo import Workflow as Previous
        import stage71_ppo_summary as s71
        from stage48_stability_runtime import digest
        original=s71.summarize;saved_eval=c.e71.evaluate;captured={}
        def capture(*args,**kwargs):
            captured["observer"]=args[3];return saved_eval(*args,**kwargs)
        def extend(root71):
            report=original(root71)
            if captured.get("done"):return report
            captured["done"]=True;root71=Path(root71);root72=root71.parent/"stage72";observer=captured["observer"]
            state=digest(observer.rt.net);before={str(p):p.read_bytes() for p in root71.parent.rglob("*") if p.is_file()}
            p71,old=c.c71.load_plan(root71);seq=p71["entries"][0]["sequence"]
            cases=(dict(case_id="micro",sequence=seq,policy="ppo",seed=p71["design"]["seeds"][0],selector="fixed",frame=2),
                   dict(case_id="missing",sequence=seq,policy="q0",seed=0,selector="first_recovery"))
            with patch.object(c,"CONFIG",cases),patch.dict(c.DESIGN,before=1,after=1),patch.dict("sys.modules",torch=None,cv2=None), \
                 patch.object(observer,"capture",side_effect=AssertionError("No new SGLA calls")), \
                 patch.object(c.e71,"checked_image",side_effect=AssertionError("No image reads")), \
                 patch.object(c.e71.learning,"update",side_effect=AssertionError("No new controller training")):
                result=a.summarize(root71,root72);self.assertEqual(result,a.summarize(root71,root72))
                self.assertEqual(result["source_evaluation_traces"],22);self.assertEqual(result["source_evaluation_frames"],88)
                self.assertEqual(result["scored_eval_state_occurrences"],128);self.assertEqual(result["shadow_rule_decisions"],640)
                self.assertEqual(result["source_training_episodes"],8);self.assertEqual(result["source_training_frames"],32)
                for k in ("new_models","new_search_calls","new_image_reads","new_full_trajectories","new_counterfactual_branches"):self.assertEqual(result[k],0)
                self.assertFalse(result["training_performed"]);self.assertFalse(result["reinforcement_learning_performed"]);self.assertFalse(result["method_adoption_gate"])
                def read(name):
                    with (root72/"analysis"/("stage72_"+name+".csv")).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
                self.assertEqual(len(read("same_state_summary")),40);self.assertEqual(len(read("matched_summary")),16)
                self.assertEqual(len(read("training_episode")),16);self.assertEqual(len(read("training_summary")),16)
                self.assertEqual(len(read("optimization_summary")),8);self.assertEqual(len(read("divergences")),4)
                self.assertTrue(all(x["videos"]=="2" for x in read("same_state_summary")))
                self.assertTrue(all(Path(path).read_bytes()==blob for path,blob in before.items()))
                job=p71["jobs"][0];path=c.c71.trace_path(root71,job,0);blob=path.read_bytes();path.write_bytes(blob+b"tamper")
                with self.assertRaises(ValueError):c.load_source(root71)
                path.write_bytes(blob)
                path=c.c71.step_path(root71,job,1);blob=path.read_bytes();path.write_bytes(blob+b"tamper")
                with self.assertRaises(ValueError):c.load_source(root71)
                path.write_bytes(blob)
                path=root71/"analysis/stage71_scope.csv";blob=path.read_bytes();path.write_bytes(blob+b"tamper")
                with self.assertRaises(ValueError):c.load_source(root71)
                path.write_bytes(blob)
                path=root72/"analysis/stage72_focus.csv";blob=path.read_bytes();path.write_bytes(blob+b"tamper")
                with self.assertRaisesRegex(ValueError,"differs"):a.summarize(root71,root72)
                path.write_bytes(blob)
                with self.assertRaisesRegex(ValueError,"separate"):a.summarize(root71,root71)
            self.assertEqual(digest(observer.rt.net),state);return report
        with patch.object(c.e71,"evaluate",side_effect=capture),patch.object(s71,"summarize",side_effect=extend):
            Previous().test_real_live_PPO_matched_updates_resumption_GT_isolation_and_frozen_controls()
        self.assertTrue(captured.get("done"))


if __name__=="__main__":unittest.main()
