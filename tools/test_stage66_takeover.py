"""Takeover semantics, fit isolation and genuine CENTER Stage-64/65/66 workflow."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import stage66_takeover_common as c
import stage66_takeover_experiment as e
import stage66_takeover_summary as s


class Semantics(unittest.TestCase):
    def pack(self,bias=.8):
        return dict(quality=dict(target="quality",scale=[1.]*64,weights=[1.]+[0.]*63),
                    critic=dict(bias=bias,center=[0.]*64,scale=[1.]*64,weights=[0.]*64),margin=dict(threshold=.008))

    def test_absolute_state_guard_and_weak_state_noncurrent_recovery_route(self):
        base=np.zeros((4,24));f=np.zeros((4,64));f[1,0]=.01;pack=self.pack()
        self.assertEqual(c.decide("quality",base,f,pack,c.DESIGN)["selected"],1)
        self.assertEqual(c.decide("margin",base,f,pack,c.DESIGN)["selected"],1)
        d=c.decide("protect",base,f,pack,c.DESIGN);self.assertEqual(d["selected"],0);self.assertTrue(d["veto"])
        self.assertEqual(c.decide("protect",base,f,self.pack(.2),c.DESIGN)["selected"],1)
        f[1,0]=.03;self.assertEqual(c.decide("protect",base,f,pack,c.DESIGN)["selected"],1)
        f[:]=0;self.assertEqual(c.decide("protect",base,f,pack,c.DESIGN)["selected"],0)
        self.assertEqual(c.decide("q0",base,f,pack,c.DESIGN)["selected"],0)
        train=np.zeros((8,4,64));train[:,0,2]=np.linspace(-1,1,8)
        labels=np.tile(np.linspace(.1,.9,8)[:,None],(1,4));critic=c.fit_absolute(train,labels,np.ones(8),.01)
        self.assertLess(c.absolute(train[0],critic),c.absolute(train[-1],critic))
        bad=train.copy();bad[0,0,0]=np.nan
        with self.assertRaises(ValueError):c.fit_absolute(bad,labels,np.ones(8),.01)

    def test_weighted_acceptance_control_ties_and_empty_eligible(self):
        gains=np.array([.01,.01,.03,.04]);w=np.array([.2,.1,.3,.4]);yes=np.ones(4,bool)
        r=c.match_margin(gains,yes,np.array([1,0,0,1],bool),w)
        self.assertAlmostEqual(r["protected_fit_acceptance"],.6);self.assertAlmostEqual(r["margin_fit_acceptance"],.7)
        self.assertAlmostEqual(r["threshold"],.01)
        self.assertLessEqual(r["acceptance_mismatch"],r["largest_tied_gain_mass"]+1e-12)
        r=c.match_margin(np.zeros(4),~yes,~yes,w);self.assertEqual(r["margin_fit_acceptance"],0.)
        lo=np.nextafter(1.,np.inf);hi=np.nextafter(lo,np.inf)
        adjacent=c.match_margin([hi,lo],[True,True],[True,False],[.5,.5])
        self.assertEqual(adjacent["margin_fit_acceptance"],.5)
        with self.assertRaises(ValueError):c.match_margin(gains,~yes,yes,w)

    def test_receipt_schema_model_binding_and_tamper(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            p=Path(temp)/"a.npz";meta=dict(model="fixed")
            c.save(p,dict(x=np.arange(4)),meta)
            np.testing.assert_array_equal(c.read_cache(p,meta)["x"],np.arange(4))
            with self.assertRaises(ValueError):c.read_cache(p,dict(model="other"))
            p.write_bytes(p.read_bytes()+b"changed")
            with self.assertRaises(ValueError):c.read_cache(p,meta)


try:
    import torch
    import cv2
except ImportError:torch=cv2=None


@unittest.skipUnless(torch is not None and cv2 is not None,"Real CPU Torch/OpenCV required; skips are not execution validation")
class Workflow(unittest.TestCase):
    def test_real_source_chain_fit_only_gt_isolation_controls_prefix_resume_and_immutability(self):
        from test_stage64_reliability import Workflow as Previous
        import stage64_reliability_summary as s64
        import stage65_selection_common as c65
        import stage65_selection_audit as a65
        from stage48_stability_runtime import digest
        saved_summary=s64.summarize;saved_execute=c.e64.execute;captured={}

        def capture(*args,**kwargs):
            captured["observer"]=args[3]
            return saved_execute(*args,**kwargs)

        def extend(root64):
            result=saved_summary(root64)
            if captured.get("done"):return result
            captured["done"]=True;observer=captured["observer"]
            oldplan,_=c.c64.load_plan(root64);first=oldplan["entries"][0]["sequence"]
            spec=dict(case_id="fixture",sequence=first,origin="native",seed=0,anchor_policy="reliability",anchor="fixed",frame=2)
            root65=Path(root64).parent/"stage65";root66=Path(root64).parent/"stage66"
            with patch.object(c65,"CASES",(spec,)):
                a65.audit(root64,root65)
                before={str(p):p.read_bytes() for p in Path(root64).parent.rglob("*") if p.is_file()}
                e.prepare(root65,root66,dict(device="NVIDIA L40",gpu_uuid="stage66-cpu-fixture"))
                plan,source=c.load_plan(root66)
                self.assertEqual(len(plan["units"]),10);self.assertEqual(plan["formal_search_calls"],160)
                original_examples=c.e64.examples
                def checked_examples(p,old,fold):
                    split=next(x for x in p["folds"] if x["fold"]==fold)
                    with patch.object(c.c64.prior.prior,"read_trace",wraps=c.c64.prior.prior.read_trace) as reader:
                        data=original_examples(p,old,fold)
                    self.assertTrue(all(call.args[1]["origin"]=="native" and call.args[1]["sequence"] in split["fit"] for call in reader.call_args_list))
                    return data
                with patch.object(c.e64,"examples",side_effect=checked_examples):models=e.fit(plan,root66,source)
                split=source["s64"]["plan"]["folds"][0];original=original_examples(source["s64"]["plan"],source["s64"]["old"],split["fold"])
                changed=copy.deepcopy(source["s64"]["old"])
                for entry in changed["l40"]["plan"]["entries"]:
                    if entry["sequence"] in split["held_out"]:entry["ground_truth"][1:]=[[1.,1.,1.,1.]]*4
                alternate=original_examples(source["s64"]["plan"],changed,split["fold"])
                for a,b in zip(original[:3],alternate[:3]):np.testing.assert_array_equal(a,b)
                state=digest(observer.rt.net);real_capture=type(observer).capture.__get__(observer,type(observer))
                # The old completed-stage fixture blocks extra calls; Stage-66 intentionally runs NEW policies.
                with patch.dict("sys.modules",torch=torch,cv2=cv2),patch.object(observer,"capture",wraps=real_capture) as calls:
                    e.execute(root66,plan,source,observer,plan["runtime_identity"],True);self.assertEqual(calls.call_count,80)
                    calls.reset_mock();e.execute(root66,plan,source,observer,plan["runtime_identity"]);self.assertEqual(calls.call_count,80)
                    u=next(u for u in plan["units"] if u["policy"]=="protect")
                    entry=next(x for x in plan["entries"] if x["sequence"]==u["sequence"])
                    committed=c.read_cache(e.path_for(root66,u),c.metadata(plan,u,"evaluation",models))
                    changedplan=copy.deepcopy(plan)
                    for x in changedplan["entries"]:x["ground_truth"][1:]=[[1.,1.,1.,1.]]*4
                    replay=e.run_actor(observer,entry["frames"],entry["initial_box"],changedplan,u,models["models"][str(u["fold"])] ,4)
                    for k in e.FIELDS:np.testing.assert_array_equal(replay[k],committed[k])
                with patch.object(observer,"capture",side_effect=AssertionError("No inference on resume")):
                    e.execute(root66,plan,source,observer,plan["runtime_identity"],True)
                    e.execute(root66,plan,source,observer,plan["runtime_identity"])
                    report=s.summarize(root66);self.assertEqual(report,s.summarize(root66))
                self.assertFalse(report["sgla_training_performed"]);self.assertEqual(report["new_critics"],2)
                self.assertEqual(report["formal_search_calls"],160);self.assertEqual(digest(observer.rt.net),state)
                self.assertTrue(all(Path(p).read_bytes()==payload for p,payload in before.items()))
                p=e.path_for(root66,u);saved=p.read_bytes();p.write_bytes(saved+b"tamper")
                with self.assertRaises(ValueError):s.summarize(root66)
                p.write_bytes(saved)
                p=root66/"models/training.json";saved=p.read_bytes();p.write_bytes(saved+b"tamper")
                with self.assertRaises(ValueError):e.load_models(plan,root66,source)
                p.write_bytes(saved)
                p=root66/"analysis/stage66_paired.csv";saved=p.read_bytes();p.write_bytes(saved+b"tamper")
                with self.assertRaisesRegex(ValueError,"differs"):s.summarize(root66)
                p.write_bytes(saved)
                p=root65/"analysis/stage65_scope.csv";saved=p.read_bytes();p.write_bytes(saved+b"tamper")
                with self.assertRaises(ValueError):c.load_source(root65)
                p.write_bytes(saved)
            return result

        with patch.dict(c.DESIGN,smoke_frames=4,bootstrap_samples=30),patch.object(c.e64,"execute",side_effect=capture),patch.object(s64,"summarize",side_effect=extend):
            Previous().test_real_source_chain_fit_isolation_controls_resume_and_frozen_weights()
        self.assertTrue(captured.get("done"))


if __name__=="__main__":unittest.main()
