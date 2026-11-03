"""Anchor support, forced-state isolation, and a real CENTER Stage-66/67 chain."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import stage67_veto_common as c
import stage67_veto_experiment as e
import stage67_veto_summary as s


class Semantics(unittest.TestCase):
    def trace(self):
        boxes=np.tile([0.,0.,4.,4.],(8,4,1));boxes[:,1,0]=2
        return dict(frame_id=np.arange(1,9),candidate_boxes=boxes,quality_choice=np.ones(8,int),
                    veto=np.array([0,1,0,1,0,0,1,0],bool))

    def test_fixed_anchor_support_never_falls_back_or_shortens(self):
        data=self.trace();d=dict(horizon=3,event_lookback=3)
        self.assertEqual(c.pick(dict(selector="first_changed_veto"),data,[],d),(2,"supported"))
        event=[dict(start=5,recovery=7)]
        self.assertEqual(c.pick(dict(selector="last_veto_before_first_failure"),data,event,d),(4,"supported"))
        self.assertEqual(c.pick(dict(selector="last_veto_before_first_recovery"),data,event,d),(None,"insufficient_full_horizon"))
        self.assertEqual(c.pick(dict(selector="last_veto_before_first_failure"),data,[],d),(None,"missing_source_event"))
        data["candidate_boxes"][:,1]=data["candidate_boxes"][:,0]
        self.assertFalse(c.changed_veto(data).any())
        self.assertEqual(c.pick(dict(selector="first_changed_veto"),data,[],d),(None,"no_changed_veto_in_declared_window"))

    def test_forced_choice_and_successor_history_are_independent_copies(self):
        shared=dict(initial_history=np.array([[0.,0.,3.,3.],[1.,0.,3.,3.]]),candidate_boxes=self.trace()["candidate_boxes"][:1])
        before=copy.deepcopy(shared);case=dict(allow_choice=1);plan=dict(actor_design=dict(history=2))
        a=e.successor_history(shared,case,"allow",plan);v=e.successor_history(shared,case,"veto",plan)
        self.assertFalse(np.array_equal(a[-1],v[-1]));a[0][0]=999;a[-1][0]=999
        for k in shared:np.testing.assert_array_equal(shared[k],before[k])
        with self.assertRaises(ValueError):e.forced_choice(case,"oracle")
        m=s.measures(np.array([.1,.6,.7,.8]),np.arange(40,44));self.assertEqual(m["first_three_good_frame"],41)

    def test_receipts_reject_mixed_metadata_and_tampering(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            p=Path(temp)/"trace.npz";meta=dict(source="frozen")
            c.save(p,dict(x=np.arange(4)),meta)
            np.testing.assert_array_equal(c.read_cache(p,meta)["x"],np.arange(4))
            with self.assertRaises(ValueError):c.read_cache(p,dict(source="other"))
            with self.assertRaises(ValueError):c.save(p,dict(x=np.arange(4)),meta)
            p.write_bytes(p.read_bytes()+b"tamper")
            with self.assertRaises(ValueError):c.read_cache(p,meta)


try:
    import torch
    import cv2
except ImportError:torch=cv2=None


@unittest.skipUnless(torch is not None and cv2 is not None,"Real CPU Torch/OpenCV required; skips are not validation")
class Workflow(unittest.TestCase):
    def test_real_source_diagonal_gt_isolation_smoke_reuse_resume_and_immutability(self):
        from test_stage66_takeover import Workflow as Previous
        import stage66_takeover_summary as s66
        from stage48_stability_runtime import digest
        saved_summary=s66.summarize;saved_execute=c.e66.execute;captured={}
        def capture(*args,**kwargs):
            captured["observer"]=args[3]
            return saved_execute(*args,**kwargs)
        def extend(root66):
            report=saved_summary(root66)
            if captured.get("done"):return report
            captured["done"]=True;observer=captured["observer"]
            source=c.load_source(root66);first=source["plan"]["entries"][0]["sequence"]
            spec=dict(case_id="fixture",sequence=first,selector="first_changed_veto")
            with patch.object(c,"CONFIG",(spec,)),patch.dict(c.DESIGN,horizon=2,event_lookback=2):
                before={str(p):p.read_bytes() for p in Path(root66).parent.rglob("*") if p.is_file()}
                root=Path(root66).parent/"stage67"
                e.prepare(root66,root,dict(device="NVIDIA L40",gpu_uuid="stage67-cpu-fixture"))
                plan,source=c.load_plan(root);self.assertEqual(len(plan["cases"]),1)
                self.assertEqual(plan["formal_search_calls"],20)
                state=digest(observer.rt.net);real_capture=type(observer).capture.__get__(observer,type(observer))
                with patch.dict("sys.modules",torch=torch,cv2=cv2),patch.object(observer,"capture",wraps=real_capture) as calls:
                    e.execute(root,plan,source,observer,plan["runtime_identity"],True);self.assertEqual(calls.call_count,8)
                    calls.reset_mock();e.execute(root,plan,source,observer,plan["runtime_identity"]);self.assertEqual(calls.call_count,12)
                    case=plan["cases"][0];shared=c.read_cache(c.path_for(root,case),c.metadata(plan,case))
                    _,entry,_=c.read_unit(source,first,"protect");pack=source["models"]["models"][str(case["unit"]["fold"])]
                    changed=copy.deepcopy(entry);changed["ground_truth"]=[[1.,1.,1.,1.]]*len(entry["ground_truth"])
                    data=e.collect_tail(observer,changed,shared,case,"quality","allow",plan,pack,e.checked_image)
                    original=c.read_cache(c.path_for(root,case,"quality","allow"),c.metadata(plan,case,"quality","allow"))
                    for k in e.FIELDS:np.testing.assert_array_equal(data[k],original[k])
                with patch.object(observer,"capture",side_effect=AssertionError("No inference on resume")):
                    e.execute(root,plan,source,observer,plan["runtime_identity"],True)
                    e.execute(root,plan,source,observer,plan["runtime_identity"])
                    result=s.summarize(root);self.assertEqual(result,s.summarize(root))
                self.assertFalse(result["training_performed"]);self.assertEqual(result["formal_search_calls"],20)
                self.assertEqual(digest(observer.rt.net),state)
                self.assertTrue(all(Path(p).read_bytes()==payload for p,payload in before.items()))
                path=c.path_for(root,case,"quality","allow");payload=path.read_bytes();path.write_bytes(payload+b"tamper")
                with self.assertRaises(ValueError):s.summarize(root)
                path.write_bytes(payload)
                path=root/"analysis/stage67_contrasts.csv";payload=path.read_bytes();path.write_bytes(payload+b"tamper")
                with self.assertRaisesRegex(ValueError,"differs"):s.summarize(root)
                path.write_bytes(payload)
                path=Path(root66)/"analysis/stage66_scope.csv";payload=path.read_bytes();path.write_bytes(payload+b"tamper")
                with self.assertRaises(ValueError):c.load_source(root66)
                path.write_bytes(payload)
            return report
        # Tiny four-frame fixture needs deliberate veto exposure. Only the synthetic
        # Stage-66 source design is changed; real Stage-67 controller parameters
        # are always copied unchanged from its sealed source manifest.
        with patch.dict(c.prior.DESIGN,q0_good_threshold=0.,protected_gain=1e6), \
             patch.object(c.e66,"execute",side_effect=capture),patch.object(s66,"summarize",side_effect=extend):
            Previous().test_real_source_chain_fit_only_gt_isolation_controls_prefix_resume_and_immutability()
        self.assertTrue(captured.get("done"))


if __name__=="__main__":unittest.main()
