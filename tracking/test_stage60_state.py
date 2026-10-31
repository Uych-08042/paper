"""Causal candidate construction, fit isolation and original CENTER actor tests."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import stage60_state_common as c
import stage60_state_runtime as actor
import stage60_state_experiment as experiment
import stage60_state_summary as summary
try:
    import torch
    import cv2
except ImportError:
    torch=cv2=None

class MathTests(unittest.TestCase):
    def test_causal_components_fixed_current_and_action_value_reference(self):
        history=np.array([[1.2345678901,2.3,17.9,23.7],[3.,5.,40.,60.],[9.,4.,22.,24.]])
        parents=c.history_parents(history,c.DESIGN)
        np.testing.assert_array_equal(parents[0],history[-1])
        np.testing.assert_allclose(parents[1,:2]+parents[1,2:]/2,history[-1,:2]+history[-1,2:]/2)
        np.testing.assert_allclose(parents[2,:2]+parents[2,2:]/2,history[-2,:2]+history[-2,2:]/2)
        np.testing.assert_allclose(parents[3,2:],np.exp(np.median(np.log(history[:,2:]),axis=0)))
        with self.assertRaises(ValueError): c.history_parents([[1,2,-1,5]],c.DESIGN)
        features=np.zeros((80,4,len(c.FEATURES)))
        features[:,:,0]=np.random.default_rng(3).uniform(0,1,(80,4))
        returns=.2+.7*features[:,:,0]
        model=c.fit_value(features,returns,c.DESIGN)
        for f,y in zip(features,returns):
            score=c.scores(f,model)
            self.assertEqual(score[0],0.)
            selected,_=c.choose("future_value",f,{"future_value":model},c.DESIGN)
            self.assertEqual(selected,int(y.argmax()))
        selected,_=c.choose("future_value",np.zeros((4,len(c.FEATURES))),{"future_value":model},c.DESIGN)
        self.assertEqual(selected,0)
        self.assertEqual(model["training_examples"],240)
        bad=returns.copy(); bad[0,0]=np.nan
        with self.assertRaises(ValueError): c.fit_value(features,bad,c.DESIGN)
        # Float64 multiplication would flip these tied native Hann peaks.
        raw=np.full((4,4),.01,np.float32); window=np.zeros((4,4),np.float32)
        raw[0,0]=.6; window[0,0]=.9; raw[0,3]=.54; window[0,3]=1.
        self.assertEqual(int((raw*window).argmax()),0)
        self.assertEqual(int((raw.astype(float)*window).argmax()),3)
        parent=np.array([40.,40.,20.,20.]); geometry=c.crop_geometry(parent,(160,180),4.,256)
        row=dict(parent=parent,boxes=parent,raw=raw,crop=geometry[0],usable=geometry[1],image_hw=np.array([160,180]))
        value=actor.feature_vector(row,[parent],dict(design=c.DESIGN,window=window.tolist()))
        self.assertEqual(value[c.FEATURES.index("raw_hann_distance")],0.)

    def test_group_isolation_and_sealed_cache_tamper(self):
        source=dict(entries=[dict(sequence="a",source_group_proxy="a"),dict(sequence="b",source_group_proxy="b")],
            folds=[dict(fold=0,fit=["a"],held_out=["b"]),dict(fold=1,fit=["b"],held_out=["a"])])
        c.validate_folds(source)
        bad=copy.deepcopy(source); bad["entries"][1]["source_group_proxy"]="a"
        with self.assertRaisesRegex(ValueError,"leakage"): c.validate_folds(bad)
        bad=copy.deepcopy(source); bad["folds"][0]["fit"].append("b")
        with self.assertRaisesRegex(ValueError,"leakage"): c.validate_folds(bad)
        with tempfile.TemporaryDirectory(prefix="stage60-cache-",dir=Path(__file__).parent) as name:
            path=Path(name)/"cache.npz"; metadata=dict(test=True)
            c.save(path,dict(x=np.arange(3)),metadata)
            np.testing.assert_array_equal(c.read_cache(path,metadata)["x"],np.arange(3))
            with self.assertRaisesRegex(ValueError,"changed"): c.read_cache(path,dict(test=False))
            path.write_bytes(path.read_bytes()+b"changed")
            with self.assertRaisesRegex(ValueError,"changed"): c.read_cache(path,metadata)

@unittest.skipUnless(torch is not None and cv2 is not None,"CPU Torch/OpenCV required; skips do not pass the actor audit")
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def fixture(self, folder):
        from test_stage50_crossed import TorchTests as Fixture
        from test_stage52_branch import actual_small_head
        from stage48_stability_runtime import make_head,digest
        from stage58_gradient_runtime import configure,frozen_digest
        from stage54_rollout_runtime import temporary_head
        from stage27_uav_common import checked_image
        import stage59_parent_collect as e59
        with patch("test_stage50_crossed.tiny_head",side_effect=actual_small_head):
            old,_,native,observer,_=Fixture().fixture(folder)
        source=dict(copy.deepcopy(old["plan"]),entries=[dict(copy.deepcopy(old["focus"]["fixture"][0]),sequence="v"+str(i),source_group_proxy="g"+str(i)) for i in range(2)],
            folds=[dict(fold=0,fit=["v1"],held_out=["v0"]),dict(fold=1,fit=["v0"],held_out=["v1"])])
        records=[]
        for fold in range(2):
            for seed in c.prior.prior.SEEDS:
                for i,arm in enumerate(c.prior.ARMS):
                    head=make_head(native.rt); _,names=configure(head,arm); frozen=frozen_digest(head,names)
                    with torch.no_grad(): head.conv5_size.bias.add_(.01*(i+1)+.003*fold+.001*(seed-2132)); head.conv5_offset.bias.add_(.002*(i+1))
                    records.append((dict(fold=fold,seed=seed,arm=arm,trainable_names=names,initial_frozen_sha256=frozen,
                        final_frozen_sha256=frozen,final_head_sha256=digest(head)),copy.deepcopy(head.state_dict())))
        arrays={}
        for entry in source["entries"]:
            images={i:checked_image(source["root"],r) for i,r in enumerate(entry["frames"])}
            native_trace=e59.collect_branch(observer,images,entry["initial_box"],entry["initial_box"],range(1,len(entry["frames"])))
            own={"native":dict(native_trace,previous=native_trace["parents"])}
            fold=next(f["fold"] for f in source["folds"] if entry["sequence"] in f["held_out"])
            for record,state in records:
                if record["fold"]!=fold: continue
                with temporary_head(native,state):
                    trace=e59.collect_branch(observer,images,entry["initial_box"],entry["initial_box"],range(1,len(entry["frames"])))
                own[record["arm"]+"_seed"+str(record["seed"])]=dict(trace,previous=trace["parents"])
            arrays[entry["sequence"]]=(entry,own)
        previous=folder/"stage58"; previous.mkdir(); marker=folder/"source.marker"; marker.write_bytes(b"frozen")
        inputs=dict(root=previous,plan=source,report=dict(fingerprint="source58decision"),run=dict(fingerprint="source58run"),
            bindings={str(marker.resolve()):c.file_hash(marker)},arrays=arrays,records=[r for r,_ in records])
        return inputs,native,observer,records

    def test_real_center_actor_history_gt_isolation_and_continuation_labels(self):
        from stage48_stability_runtime import digest
        with tempfile.TemporaryDirectory(prefix="stage60-actor-",dir=Path(__file__).parent) as name:
            source,native,observer,records=self.fixture(Path(name)); plan=dict(source["plan"],design=dict(c.DESIGN,horizon=2,anchors_per_video=2))
            entry=plan["entries"][0]; record,state=experiment.record_for(records,0,2132); original=digest(native.rt.net)
            model=dict(scale=[1.]*len(c.FEATURES),weights=[0.]*len(c.FEATURES))
            model["weights"][c.FEATURES.index("parent_log_w")]=-2.
            models={t:model for t in c.TARGETS}
            with experiment.installed(observer,record,state):
                trace=actor.run_actor(observer,entry["frames"],entry["initial_box"],plan["root"],"future_value",models,plan)
                self.assertTrue((trace["selected"][1:]!=0).any())
                self.assertEqual(int(trace["head_calls"].sum()),16)
                experiment.audit_evaluation(trace,entry,plan,"future_value",models,None)
                baseline=actor.run_actor(observer,entry["frames"],entry["initial_box"],plan["root"],"current",{},plan)
                first=actor.collect_anchor(observer,entry,baseline,2,plan)
                changed=copy.deepcopy(entry); changed["ground_truth"][1:]=[[130.,120.,10.,10.]]*(len(entry["frames"])-1)
                second=actor.collect_anchor(observer,changed,baseline,2,plan)
                for key in ("boxes","parents","features","initial_parents"):
                    np.testing.assert_array_equal(first[key],second[key])
                self.assertFalse(np.array_equal(first["future_returns"],second["future_returns"]))
                self.assertLess(float(trace["decode_error"].max()),.001)
                broken=copy.deepcopy(trace); broken["candidate_parents"][2,1,0]+=5
                with self.assertRaises(AssertionError): experiment.audit_evaluation(broken,entry,plan,"future_value",models,None)
            self.assertEqual(digest(native.rt.net),original)
            observer.handle.remove(); observer.pre_handle.remove()

    def test_real_stage59_source_prepare_smoke_fit_oof_resume_and_tamper(self):
        import stage59_parent_collect as e59
        import stage59_parent_summary as s59
        from stage48_stability_runtime import digest
        with tempfile.TemporaryDirectory(prefix="stage60-full-",dir=Path(__file__).parent) as name,contextlib.redirect_stdout(io.StringIO()):
            folder=Path(name); source,native,observer,records=self.fixture(folder)
            config=[dict(case_id="T01",sequence="v0",frame=2,purpose="fixture",anchor_method="endpoint_control_seed2132",anchor_kind="fixed_frame")]
            with patch.object(c.prior,"load_source",return_value=source),patch.object(c.prior,"CONFIG",config),\
                    patch.dict(c.prior.DESIGN,horizon=2),patch.dict(c.DESIGN,horizon=2,anchors_per_video=2,smoke_frames=3,bootstrap_samples=50):
                root59=folder/"stage59"; e59.prepare(source["root"],root59); p59,i59=c.prior.load_plan(root59)
                banks={f["fold"]:e59.HeadBank(native,records,f["fold"]) for f in source["plan"]["folds"]}
                e59.execute(root59,p59,i59,observer,banks,source["plan"]["runtime_identity"],True)
                e59.execute(root59,p59,i59,observer,banks,source["plan"]["runtime_identity"]); s59.summarize(root59)
                checked=c.load_source(root59); self.assertEqual(checked["stage59_report"]["status"],"COMPLETE_PARENT_COMPONENT_CONTINUATION_DIAGNOSTIC")
                root=folder/"stage60"; experiment.prepare(root59,root); plan,inputs=c.load_plan(root)
                self.assertEqual(plan["expected_evaluation_calls"],208); self.assertEqual(plan["expected_branch_calls"],64)
                original=digest(native.rt.net)
                experiment.smoke(plan,root,inputs,observer,records)
                cached={str(p):p.read_bytes() for p in (root/"fit").rglob("*") if p.is_file()}
                experiment.collect_fit(plan,root,observer,records)
                self.assertTrue(all(Path(p).read_bytes()==v for p,v in cached.items()))
                training=experiment.fit_models(plan,root,records); self.assertEqual(training["new_scorers"],8)
                models,_=experiment.load_models(plan,root,records)
                self.assertEqual(models[0,2132]["fit"],["v1"])
                changed=copy.deepcopy(plan)
                for e in changed["entries"]:
                    if e["sequence"]=="v0": e["ground_truth"]=[[120.,120.,10.,10.]]*len(e["frames"])
                record,_=experiment.record_for(records,0,2132)
                values,bindings=experiment.examples(changed,root,0,2132,record)
                self.assertEqual(models[0,2132],experiment.model_body(changed,0,2132,record,values,bindings))
                with patch.object(observer,"capture",side_effect=AssertionError("Committed fit inference must be reused")):
                    experiment.collect_fit(plan,root,observer,records); experiment.fit_models(plan,root,records)
                experiment.evaluate(plan,root,inputs,observer,records)
                with patch.object(observer,"capture",side_effect=AssertionError("Committed OOF must be reused")):
                    experiment.evaluate(plan,root,inputs,observer,records)
                report=summary.summarize(root)
                self.assertEqual(report["formal_search_calls"],288); self.assertEqual(report["additional_smoke_search_calls"],39)
                self.assertEqual(report,summary.summarize(root)); self.assertEqual(digest(native.rt.net),original)
                path=root/"models/fold0__seed2132.json"; saved=path.read_bytes(); path.write_bytes(saved+b" ")
                with self.assertRaises(ValueError): experiment.load_models(plan,root,records)
                path.write_bytes(saved)
                path=root59/"analysis/stage59_focus.csv"; saved=path.read_bytes(); path.write_bytes(saved+b"changed")
                with self.assertRaises(ValueError): c.load_plan(root)
                path.write_bytes(saved)
                path=root/"analysis/stage60_policy_summary.csv"; path.write_bytes(path.read_bytes()+b"changed")
                with self.assertRaisesRegex(ValueError,"differs"): summary.summarize(root)
                observer.handle.remove(); observer.pre_handle.remove()

if __name__=="__main__": unittest.main()
