"""Labels, causal features and a real CPU Stage-59..64 full workflow."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import stage64_reliability_common as c
import stage64_reliability_experiment as e
import stage64_reliability_summary as s


class Semantics(unittest.TestCase):
    def test_causal_features_labels_and_current_ties(self):
        box=np.array([40.,40.,20.,20.]);boxes=np.array([box,box+[1,0,0,0],box+[0,2,0,0],box+[0,0,2,2]])
        f=c.choice_features(np.ones((4,24))*.1,boxes,[box])
        self.assertEqual(f.shape,(4,64));self.assertTrue(np.isfinite(f).all())
        # No external objects, ids or GT are part of the feature interface.
        np.testing.assert_array_equal(f,c.choice_features(np.ones((4,24))*.1,boxes,[box]))
        train=np.stack([f,f]);y=np.array([[.6,.1,.4,.8],[.4,.6,.1,.3]])
        q=c.fit_selector(train,y,np.ones(2),"quality",.01);r=c.fit_selector(train,y,np.ones(2),"reliability",.01)
        self.assertFalse(np.array_equal(q["weights"],r["weights"]))
        self.assertEqual(c.choose("reliability",np.ones((4,24)),np.zeros((4,64)),r,1e-6)[0],0)
        self.assertEqual(c.choose("q0",np.ones((4,24)),f,None,1e-6)[0],0)
        bad=boxes.copy();bad[0,2]=0
        with self.assertRaises(ValueError):c.choice_features(np.ones((4,24)),bad,[box])

    def test_cache_model_binding_and_tamper(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            p=Path(directory)/"x.npz";m=dict(model="frozen")
            c.save(p,dict(x=np.arange(3)),m)
            np.testing.assert_array_equal(c.read_cache(p,m)["x"],np.arange(3))
            with self.assertRaises(ValueError):c.read_cache(p,dict(model="other"))
            p.write_bytes(p.read_bytes()+b"tamper")
            with self.assertRaises(ValueError):c.read_cache(p,m)


try:
    import torch
    import cv2
except ImportError:torch=cv2=None


@unittest.skipUnless(torch is not None and cv2 is not None,"Real CPU Torch/OpenCV required; skips are not execution validation")
class Workflow(unittest.TestCase):
    def test_real_source_chain_fit_isolation_controls_resume_and_frozen_weights(self):
        from test_stage63_choice import RealCenterTests
        from stage48_stability_runtime import digest
        import stage63_choice_experiment as e63
        import stage63_choice_summary as s63
        captured={};original_execute=e63.execute;original_summary=s63.summarize

        def capture(root,plan,inputs,observer,records,identity,*args,**kwargs):
            captured.update(observer=observer,records=records,identity=identity)
            return original_execute(root,plan,inputs,observer,records,identity,*args,**kwargs)

        def after_stage63(root):
            result=original_summary(root)
            if captured.get("stage64_done"):return result
            captured["stage64_done"]=True
            observer=captured["observer"];records=captured["records"]
            frozen={str(p):p.read_bytes() for p in Path(root).parent.rglob("*") if p.is_file()}
            root64=Path(root).parent/"stage64";hardware=dict(device="NVIDIA L40",gpu_uuid="stage64-cpu-fixture")
            e.prepare(root,root64,hardware);plan,source=c.load_plan(root64)
            self.assertEqual(plan["new_full_trajectories"],16);self.assertEqual(plan["formal_search_calls"],256)
            # Every fit trace must be native, and come only from this fold's fit groups.
            split=plan["folds"][0]
            with patch.object(c.prior.prior,"read_trace",wraps=c.prior.prior.read_trace) as reader:
                a=e.examples(plan,source,split["fold"])
            self.assertTrue(all(call.args[1]["origin"]=="native" and call.args[1]["sequence"] in split["fit"] for call in reader.call_args_list))
            changed=copy.deepcopy(source)
            for entry in changed["l40"]["plan"]["entries"]:
                if entry["sequence"] in split["held_out"]:entry["ground_truth"][1:]=[[130.,120.,1.,1.]]*4
            b=e.examples(plan,changed,split["fold"])
            for x,y in zip(a[:3],b[:3]):np.testing.assert_array_equal(x,y)
            models=e.fit(plan,root64,source);self.assertEqual(models["scorers"],4)
            before=digest(observer.rt.net)
            with patch.object(observer,"capture",wraps=observer.capture) as calls:
                e.execute(root64,plan,source,observer,records,plan["runtime_identity"],True)
                self.assertEqual(calls.call_count,128)
            with patch.object(observer,"capture",wraps=observer.capture) as calls:
                e.execute(root64,plan,source,observer,records,plan["runtime_identity"])
                self.assertEqual(calls.call_count,128) # First video's entire prefix already committed.
            unit=next(u for u in plan["units"] if u["origin"]=="native" and u["policy"]=="reliability")
            entry=next(x for x in plan["entries"] if x["sequence"]==unit["sequence"])
            data=c.read_cache(e.path_for(root64,unit),c.meta(plan,unit,"evaluation",models))
            changed_plan=copy.deepcopy(plan)
            for item in changed_plan["entries"]:item["ground_truth"][1:]=[[1.,1.,1.,1.]]*4
            replay=e.run_actor(observer,entry["frames"],entry["initial_box"],changed_plan,unit,e.selected_model(models,unit),4)
            for key in e.FIELDS:np.testing.assert_array_equal(data[key],replay[key])
            with patch.object(observer,"capture",side_effect=AssertionError("Committed inference must not repeat")):
                e.execute(root64,plan,source,observer,records,plan["runtime_identity"],True)
                e.execute(root64,plan,source,observer,records,plan["runtime_identity"])
                report=s.summarize(root64);self.assertEqual(report,s.summarize(root64))
            self.assertEqual(report["formal_search_calls"],256);self.assertEqual(report["new_selectors"],4)
            self.assertFalse(report["sgla_training_performed"]);self.assertEqual(digest(observer.rt.net),before)
            self.assertTrue(all(Path(p).read_bytes()==payload for p,payload in frozen.items()))
            path=e.path_for(root64,unit);saved=path.read_bytes();path.write_bytes(saved+b"tamper")
            with self.assertRaises(ValueError):s.summarize(root64)
            path.write_bytes(saved)
            path=root64/"models/training.json";saved=path.read_bytes();path.write_bytes(saved+b"tamper")
            with self.assertRaises(ValueError):e.load_models(plan,root64)
            path.write_bytes(saved)
            path=root64/"analysis/stage64_paired.csv";path.write_bytes(path.read_bytes()+b"tamper")
            with self.assertRaisesRegex(ValueError,"differs"):s.summarize(root64)
            return result

        def extend_fixture(root):
            # Stage-63 invokes its summary with Torch/OpenCV imports blocked. Stage-64 deliberately
            # creates NEW image replays, so restore the real CPU dependencies inside this extension.
            with patch.dict("sys.modules",torch=torch,cv2=cv2):return after_stage63(root)

        with patch.dict(c.DESIGN,smoke_frames=4,bootstrap_samples=30),patch.object(e63,"execute",side_effect=capture),patch.object(s63,"summarize",side_effect=extend_fixture):
            RealCenterTests().test_full_sealed_source_chain_l40_binding_resume_immutability_and_tamper()
        self.assertTrue(captured.get("stage64_done"))


if __name__=="__main__":unittest.main()
