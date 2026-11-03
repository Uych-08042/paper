"""Paired-label semantics, GT-free fit sampling, and a real CENTER source chain."""
import copy
import csv
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import stage68_advantage_common as c
import stage68_advantage_experiment as e
import stage68_advantage_summary as s


def old_pack(axis=0):
    weights=np.zeros(64);weights[axis]=1
    return dict(quality=dict(target="quality",scale=[1.]*64,weights=weights.tolist()),
        critic=dict(bias=.8,center=[0.]*64,scale=[1.]*64,weights=[0.]*64),margin=dict(threshold=.008))


class Semantics(unittest.TestCase):
    def test_paired_future_uses_actual_tails_and_can_reverse_current_gain(self):
        shared=dict(candidate_boxes=np.array([[[1.,0.,2.,2.],[0.,0.,2.,2.],[0.,0.,2.,2.],[0.,0.,2.,2.]]]))
        tails=dict(allow=dict(boxes=np.array([[10.,0.,2.,2.]])),veto=dict(boxes=np.array([[0.,0.,2.,2.]])))
        entry=dict(ground_truth=[[0.,0.,2.,2.]]*3);case=dict(frame=1,allow_choice=1)
        label=e.label_pair(shared,tails,entry,case,dict(design=dict(horizon=2)))
        self.assertAlmostEqual(label["immediate"],2/3);self.assertAlmostEqual(label["future"],-1/6)
        self.assertEqual(label["low_frame_delta"],1)
        f=np.zeros((4,64));f[1,0]=.01
        model=lambda t,b:dict(target=t,center=[0.]*128,scale=[1.]*128,weights=[0.]*128,bias=b,pairs=1)
        pack=dict(source=old_pack(),immediate=model("immediate",.1),future=model("future",-.1),rate_margin=dict(threshold=.005))
        self.assertEqual(c.decide("immediate",np.zeros((4,24)),f,pack,c.DESIGN,e.c66.DESIGN)["selected"],1)
        self.assertEqual(c.decide("future",np.zeros((4,24)),f,pack,c.DESIGN,e.c66.DESIGN)["selected"],0)
        self.assertEqual(c.decide("quality",np.zeros((4,24)),f,pack,c.DESIGN,e.c66.DESIGN)["selected"],1)
        x=np.zeros((6,128));x[:,0]=np.linspace(-1,1,6);y=np.linspace(-.5,.5,6)
        fitted=c.fit_regressor(x,y,np.ones(6),.01,"future")
        self.assertLess(c.predict(x[0],fitted),c.predict(x[-1],fitted))
        with self.assertRaises(ValueError):c.pair_feature(f,0)
        with self.assertRaises(ValueError):c.fit_regressor(x,np.full(6,np.nan),np.ones(6),.01,"future")

    def test_fit_sampling_ignores_stored_oof_q_and_gt_and_recomputes_fold_local_q(self):
        f=np.zeros((3,4,64));f[:,1,0]=.01;f[:,2,1]=.02
        boxes=np.tile([0.,0.,4.,4.],(3,4,1));boxes[:,1,0]=1;boxes[:,2,0]=2
        trace=dict(frame_id=np.arange(1,4),features=np.zeros((3,4,24)),choice_features=f,candidate_boxes=boxes,
                   quality_choice=np.full(3,3),q0_pred_iou=np.full(3,999.))
        split0=dict(fold=0,fit=["a"],held_out=["b"]);split1=dict(fold=1,fit=["b"],held_out=["a"])
        source=dict(s66=dict(root=Path("unused"),plan=dict(folds=[split0,split1],design=e.c66.DESIGN),
            models=dict(models={"0":dict(old_pack(),**split0),"1":dict(old_pack(1),**split1)}),
            run=dict(observed_heads={e.c66.key(dict(sequence=n,origin="native",seed=0,policy="hann")):"native" for n in ("a","b")})))
        def read(_,seq,policy):
            self.assertEqual(policy,"hann")
            return trace,dict(ground_truth=object()),dict(sequence=seq,origin="native",seed=0,policy="hann")
        with patch.object(c.prior,"read_unit",side_effect=read),patch.object(c,"file_hash",return_value="frozen"),patch.dict(c.DESIGN,horizon=2,anchors_per_video=1):
            cases,support=c.resolve_pairs(source)
        self.assertEqual([(x["fold"],x["sequence"],x["allow_choice"]) for x in cases],[(0,"a",1),(1,"b",2)])
        self.assertEqual([x["selected_pairs"] for x in support],[1,1])

    def test_uniform_export_receipts_tamper_and_unknown_run_metadata(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path=Path(temp)/"table.csv";s.publish_table(path,[dict(a=1),dict(a=2,frame=664)])
            with path.open(encoding="utf8",newline="") as stream:rows=list(csv.DictReader(stream))
            self.assertEqual(rows,[dict(a="1",frame=""),dict(a="2",frame="664")])
            before=path.read_bytes();s.publish_table(path,[dict(a=1),dict(a=2,frame=664)]);self.assertEqual(path.read_bytes(),before)
            p=Path(temp)/"cache.npz";meta=dict(fold=0);c.save(p,dict(x=np.arange(3)),meta)
            np.testing.assert_array_equal(c.read_cache(p,meta)["x"],np.arange(3))
            with self.assertRaises(ValueError):c.read_cache(p,dict(fold=1))
            p.write_bytes(p.read_bytes()+b"tamper")
            with self.assertRaises(ValueError):c.read_cache(p,meta)


try:
    import torch
    import cv2
except ImportError:torch=cv2=None


@unittest.skipUnless(torch is not None and cv2 is not None,"Real CPU Torch/OpenCV required; skips are not validation")
class Workflow(unittest.TestCase):
    def test_real_fit_isolation_paired_calls_controls_resume_and_immutability(self):
        from test_stage67_summary_repair import Workflow as Previous
        import stage67_summary_repair as r67
        from stage48_stability_runtime import digest
        saved_summary=r67.summarize;saved_execute=e.e67.execute;captured={}
        saved_pack=c.fit_pack;exposed={}
        def exposed_fit_pack(source,fold):
            # Four-frame synthetic fits can have no learned Q proposal. Expose
            # fit-only branch paths with deterministic causal test coefficients;
            # original sources and evaluation-control Q coefficients stay intact.
            if fold in exposed:return exposed[fold]
            pack=copy.deepcopy(saved_pack(source,fold))
            for seq in pack["fit"]:
                trace,_,_=c.prior.read_unit(source["s66"],seq,"hann")
                for i in range(len(trace["frame_id"])-c.DESIGN["horizon"]+1):
                    f=trace["choice_features"][i]
                    for q in (1,2,3):
                        v=(f[q]-f[0])/np.asarray(pack["quality"]["scale"])
                        pack["quality"]["weights"]=v.tolist()
                        chosen,_=e.c66.c64.choose("quality",trace["features"][i],f,pack["quality"],c.DESIGN["tie_epsilon"])
                        if chosen and not np.array_equal(trace["candidate_boxes"][i,chosen],trace["candidate_boxes"][i,0]):
                            exposed[fold]=pack;return pack
            self.fail("Real CENTER fixture has no changed fit candidate")
        def capture(*args,**kwargs):
            captured["observer"]=args[3]
            return saved_execute(*args,**kwargs)
        def extend(root67):
            report=saved_summary(root67)
            if captured.get("done"):return report
            captured["done"]=True;observer=captured["observer"];root=Path(root67).parent/"stage68"
            before={str(p):p.read_bytes() for p in Path(root67).parent.rglob("*") if p.is_file()}
            e.prepare(root67,root,dict(device="NVIDIA L40",gpu_uuid="stage68-cpu-fixture"))
            plan,source=c.load_plan(root);self.assertEqual(len(plan["pairs"]),2)
            self.assertEqual(plan["pair_search_calls"],24);self.assertEqual(plan["formal_eval_calls"],224)
            state=digest(observer.rt.net);real_capture=type(observer).capture.__get__(observer,type(observer))
            with patch.dict("sys.modules",torch=torch,cv2=cv2),patch.object(observer,"capture",wraps=real_capture) as calls:
                e.collect(root,plan,source,observer,plan["runtime_identity"],True);self.assertEqual(calls.call_count,12)
                calls.reset_mock();e.collect(root,plan,source,observer,plan["runtime_identity"]);self.assertEqual(calls.call_count,12)
                with patch.object(e,"label_pair",wraps=e.label_pair) as labeller:
                    models=e.fit(plan,root,source)
                for call in labeller.call_args_list:
                    case=call.args[3];split=next(f for f in plan["folds"] if f["fold"]==case["fold"])
                    self.assertIn(case["sequence"],split["fit"]);self.assertNotIn(case["sequence"],split["held_out"])
                fold=plan["folds"][0];changed=copy.deepcopy(source)
                for x in changed["s66"]["plan"]["entries"]:
                    if x["sequence"] in fold["held_out"]:x["ground_truth"][1:]=[[99.,99.,1.,1.]]*4
                self.assertEqual(e.build_fold(plan,root,source,fold["fold"]),e.build_fold(plan,root,changed,fold["fold"]))
                calls.reset_mock();e.evaluate(root,plan,source,observer,plan["runtime_identity"],True);self.assertEqual(calls.call_count,112)
                calls.reset_mock();e.evaluate(root,plan,source,observer,plan["runtime_identity"]);self.assertEqual(calls.call_count,112)
                u=next(x for x in plan["units"] if x["policy"]=="future");entry=copy.deepcopy(next(x for x in plan["entries"] if x["sequence"]==u["sequence"]))
                entry["ground_truth"][1:]=[[99.,99.,1.,1.]]*4
                replay=e.run_actor(observer,entry,u,plan,models["models"][str(u["fold"])],4)
                committed=c.read_cache(c.eval_path(root,u),c.eval_meta(plan,u,models))
                for k in e.FIELDS:np.testing.assert_array_equal(replay[k],committed[k])
                case=plan["pairs"][0];trace,entry,pack=e.context(plan,source,case)
                changed=copy.deepcopy(entry);changed["ground_truth"][1:]=[[99.,99.,1.,1.]]*4
                shared=c.read_cache(c.pair_path(root,case),c.pair_meta(plan,case))
                replay=e.e67.collect_tail(observer,changed,shared,case,"quality","allow",plan,pack,e.checked_image)
                committed=c.read_cache(c.pair_path(root,case,"allow"),c.pair_meta(plan,case,"allow"))
                for k in e.e67.FIELDS:np.testing.assert_array_equal(replay[k],committed[k])
            with patch.object(observer,"capture",side_effect=AssertionError("No inference on resume or summary")):
                e.collect(root,plan,source,observer,plan["runtime_identity"],True);e.collect(root,plan,source,observer,plan["runtime_identity"])
                e.evaluate(root,plan,source,observer,plan["runtime_identity"],True);e.evaluate(root,plan,source,observer,plan["runtime_identity"])
                result=s.summarize(root);self.assertEqual(result,s.summarize(root))
            self.assertEqual(result["new_predictors"],4);self.assertFalse(result["sgla_training_performed"])
            self.assertEqual(digest(observer.rt.net),state)
            self.assertTrue(all(Path(p).read_bytes()==b for p,b in before.items()))
            p=c.pair_path(root,case,"allow");payload=p.read_bytes();p.write_bytes(payload+b"tamper")
            with self.assertRaises(ValueError):s.summarize(root)
            p.write_bytes(payload)
            p=root/"analysis/stage68_scope.csv";payload=p.read_bytes();p.write_bytes(payload+b"tamper")
            with self.assertRaisesRegex(ValueError,"differs"):s.summarize(root)
            p.write_bytes(payload)
            return report
        with patch.dict(c.DESIGN,horizon=2,anchors_per_video=1,smoke_frames=4,bootstrap_samples=30), \
             patch.object(c,"fit_pack",side_effect=exposed_fit_pack), \
             patch.object(e.e67,"execute",side_effect=capture),patch.object(r67,"summarize",side_effect=extend):
            Previous().test_real_frozen_branch_chain_repaired_without_source_changes_or_inference()
        self.assertTrue(captured.get("done"))


if __name__=="__main__":unittest.main()
