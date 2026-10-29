"""Native-head loss routing, OOF isolation, resume and end-to-end CPU fixtures."""
import contextlib
import copy
import inspect
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import stage48_stability_common as c
import stage48_stability_experiment as experiment

try:
    import torch
    import cv2
except ImportError:
    torch=cv2=None


class PlanTests(unittest.TestCase):
    def test_group_oof_and_paired_parent_budget(self):
        entries=[dict(sequence="v"+str(i),source_group_proxy="g"+str(i)) for i in range(12)]
        folds=c.fold_design(entries)
        self.assertEqual({n for f in folds for n in f["held_out"]},{e["sequence"] for e in entries})
        for fold in folds:
            self.assertFalse(set(fold["fit"])&set(fold["held_out"]))
            self.assertEqual((len(fold["fit"]),len(fold["held_out"])),(6,6))
        with self.assertRaisesRegex(ValueError,"distinct"):
            c.fold_design(entries[:-1]+[entries[0]])
        pair=c.pair_parents([40,40,20,20],"fixture",12)
        self.assertAlmostEqual(np.linalg.norm(pair[1,:2]-pair[0,:2]),1.)
        np.testing.assert_array_equal(pair[:,2:],[[20,20],[20,20]])

    def test_completed_source_binding_and_mutation_rejection(self):
        with tempfile.TemporaryDirectory(prefix="stage48-source-",dir=Path(__file__).parent) as folder:
            root=Path(folder); old=root/"old"; old.mkdir(); previous=root/"stage46"; previous.mkdir()
            marker=root/"source.txt"; marker.write_text("frozen",encoding="utf-8")
            entries=[dict(sequence="v"+str(i),source_group_proxy="g"+str(i)) for i in range(12)]
            run=c.prior.seal(dict(schema=c.prior.SCHEMA,window=np.ones((4,4)).tolist()))
            c.prior.motion.write_json(old/"run.json",run)
            c.prior.motion.write_json(old/"manifest.json",dict(fixture=True))
            oldplan=dict(fingerprint="plan47",expected_head_calls=400,source_input=str(previous),events=[],
                source_file_sha256={str(marker):c.file_hash(marker)})
            report=c.prior.seal(dict(schema=c.prior.SCHEMA,status="COMPLETE_MATCHED_SEARCH_RESPONSE_DIAGNOSTIC",
                plan_fingerprint="plan47",run_fingerprint=run["fingerprint"],training_performed=False,
                replay_head_calls=400,cache_sha256={},table_sha256={}))
            c.prior.motion.write_json(old/"analysis/stage47_diagnostic.json",report)
            c.prior.motion.write_json(previous/"manifest.json",c.prior.motion.seal(dict(schema=c.prior.motion.SCHEMA,sequences=[e["sequence"] for e in entries])))
            inputs=dict(traces={e["sequence"]:(e,{}) for e in entries},source=dict(runtime=dict(lens_x=16,embed_dim=3),
                manifest=dict(root=str(root)),search_factor=4.,search_size=256),old_run=dict(runtime=dict(fixture=True)))
            with patch.object(c.prior,"load_plan",return_value=(oldplan,inputs)):
                plan=c.make_plan(old)
                self.assertFalse(plan["independent_validation"])
                self.assertEqual(len(plan["folds"]),2)
                marker.write_text("changed",encoding="utf-8")
                with self.assertRaisesRegex(ValueError,"changed"): c.make_plan(old)


def tiny_head():
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ctr=torch.nn.Conv2d(3,1,1)
            self.sz=torch.nn.Conv2d(3,2,1)
            self.off=torch.nn.Conv2d(3,2,1)
            self.register_buffer("fixed_buffer",torch.tensor(1.))
            with torch.no_grad():
                self.ctr.weight.zero_(); self.ctr.weight[:,0]=.1; self.ctr.bias.zero_()
                self.sz.weight.zero_(); self.sz.bias.fill_(float(np.log(.275/.725)))
                self.off.weight.zero_(); self.off.bias.zero_()
        def forward(self,x):
            raw=self.ctr(x).sigmoid(); size=self.sz(x).sigmoid(); offset=self.off(x).sigmoid()*.01
            index=raw.flatten(1).argmax(1)
            b=torch.arange(len(x),device=x.device)
            selected=size.flatten(2)[b,:,index]
            off=offset.flatten(2)[b,:,index]
            xy=torch.stack([index%4,index//4],1).to(x.dtype)
            box=torch.cat([(xy+off)/4,selected],1)
            return raw,box,size,offset
    return Head().eval()


def fixture_feature():
    value=np.zeros((1,3,4,4),np.float32); value[0,0,2,2]=1
    return value


@unittest.skipUnless(torch is not None,"CPU Torch/OpenCV needed for training tests")
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_auxiliary_gradients_are_scores_or_sizes_only_and_empty_support_is_zero(self):
        from stage48_stability_runtime import head_output,loss_terms
        head=tiny_head()
        output=head_output(head,torch.tensor(np.repeat(fixture_feature(),2,0)))
        pos=np.zeros((1,2,4,4),bool); pos[:,:,2,2]=True
        negative=~pos
        batch=dict(target=np.tile([.375,.375,.25,.25],(1,2,1)),positive=pos,negative=negative,
            size_cell=np.full((1,2),10),size_valid=np.ones((1,2),bool),crop=np.tile([10,10,80,80],(1,2,1)),gt_wh=np.full((1,2,2),20.))
        ordinary,rank,shape,stats=loss_terms(torch,output,batch,np.ones((4,4)))
        g=torch.autograd.grad(rank,[output["score_map"],output["size_map"],output["offset_map"]],allow_unused=True,retain_graph=True)
        self.assertGreater(float(g[0].abs().sum()),0); self.assertIsNone(g[1]); self.assertIsNone(g[2])
        g=torch.autograd.grad(shape,[output["score_map"],output["size_map"],output["offset_map"]],allow_unused=True,retain_graph=True)
        self.assertIsNone(g[0]); self.assertGreater(float(g[1].abs().sum()),0); self.assertIsNone(g[2])
        self.assertEqual((stats["rank_views"],stats["size_views"]),(2,2))
        empty=copy.deepcopy(batch); empty["positive"][:]=False; empty["size_valid"][:]=False
        _,r,s,stats=loss_terms(torch,output,empty,np.ones((4,4)))
        self.assertEqual((float(r),float(s)),(0.,0.)); self.assertEqual(stats["rank_views"],0)
        self.assertTrue(torch.isfinite(ordinary))

    def test_training_head_clone_excludes_bound_native_instrumentation(self):
        from stage48_stability_runtime import make_head,digest
        head=tiny_head()
        class Owner:
            def __init__(self): self.unpickleable_runtime=torch; self.calls=0
            def observed(self,module,args,output): self.calls+=1
        owner=Owner(); handle=head.register_forward_hook(owner.observed)
        trained=make_head(types.SimpleNamespace(net=types.SimpleNamespace(box_head=head)))
        self.assertEqual(digest(head),digest(trained))
        trained(torch.tensor(fixture_feature()))
        self.assertEqual(owner.calls,0)
        head(torch.tensor(fixture_feature()))
        self.assertEqual(owner.calls,1)
        self.assertFalse(trained.training)
        handle.remove()

    def fixture(self,root):
        from stage48_stability_runtime import head_output
        from stage28_response_common import dense_boxes
        from stage27_uav_common import checked_image,tracking_iou
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.box_head=tiny_head()
        net=Net().eval()
        tracker=types.SimpleNamespace(state=None)
        def decode(tracker,output,factor,h,w):
            raw=output["score_map"].detach().numpy()[0,0]
            dense=dense_boxes(output["size_map"].detach().numpy()[0],output["offset_map"].detach().numpy()[0],tracker.state,factor,256,(h,w))
            return dense[int(np.argmax(raw))].copy()
        rt=types.SimpleNamespace(torch=torch,net=net,tracker=tracker,s1=types.SimpleNamespace(_decode_box=decode))
        class Native:
            def __init__(self):
                from stage46_motion_runtime import state_digest
                self.rt=rt; self.digest=state_digest(net)
            def initialize(self,image,box): tracker.state=list(box)
            def step(self,image,parent):
                tracker.state=list(parent)
                with torch.no_grad(): out=head_output(net.box_head,torch.tensor(fixture_feature()))
                factor=256/c.crop_geometry(parent,image.shape[:2],4.,256)[0][2]
                box=decode(tracker,out,factor,*image.shape[:2])
                blocks=np.zeros(12,int); blocks[:6]=1; blocks[8]=1
                return dict(box=box,layer=9,peak=float(out["score_map"].max()),head_calls=1,block_calls=blocks)
            def assert_frozen(self):
                from stage46_motion_runtime import state_digest
                if self.digest!=state_digest(net): raise ValueError("Fixture model changed")
        native=Native(); frames=[]
        for i in range(9):
            path=root/(str(i)+".png"); cv2.imwrite(str(path),np.zeros((160,180,3),np.uint8))
            frames.append(dict(file=path.name,height=160,width=180,sha256=c.file_hash(path)))
        entries=[dict(sequence="v"+str(i),source_group_proxy="g"+str(i),frames=frames,initial_box=[40,40,20,20],ground_truth=[[40,40,20,20]]*9) for i in range(12)]
        marker=root/"source.txt"; marker.write_text("frozen",encoding="utf-8")
        plan=c.seal(dict(schema=c.SCHEMA,entries=entries,folds=c.fold_design(entries),root=str(root),
            runtime=dict(lens_x=16,embed_dim=3),runtime_identity=dict(fixture=True),search_size=256,search_factor=4.,
            window=np.ones((4,4)).tolist(),source_file_sha256={str(marker):c.file_hash(marker)}))
        references={}
        for e in entries:
            previous=np.array(e["initial_box"],float); parents=[]; boxes=[]; peaks=[]
            for frame in range(1,9):
                parents.append(previous.copy()); result=native.step(np.zeros((160,180,3)),previous)
                boxes.append(result["box"]); previous=result["box"].copy(); peaks.append(result["peak"])
            references[e["sequence"]]=dict(previous=np.asarray(parents),boxes=np.asarray(boxes),peak=np.asarray(peaks),layer=np.full(8,9,int))
            rows=[]
            for frame in c.sample_frames(e):
                views=[]
                for parent in c.pair_parents(parents[frame-1],e["sequence"],frame):
                    with torch.no_grad(): out=head_output(net.box_head,torch.tensor(fixture_feature()))
                    raw=out["score_map"].numpy()[0,0]; size=out["size_map"].numpy()[0]; off=out["offset_map"].numpy()[0]
                    crop,_,factor=c.crop_geometry(parent,(160,180),4.,256)
                    dense=dense_boxes(size,off,parent,factor,256,(160,180)); box=dense[int(np.argmax(raw))].copy()
                    gt=np.array([40,40,20,20],float); target=np.r_[(gt[:2]-crop[:2])/crop[2],gt[2:]/crop[2]]
                    overlap=tracking_iou(dense,gt).reshape(4,4); xy=np.clip(np.round((target[:2]+target[2:]/2)*4).astype(int),0,3)
                    views.append(dict(features=fixture_feature()[0],parents=parent,crop=crop,raw=raw,size=size,offset=off,boxes=box,layer=9,
                        target=target,gt_wh=gt[2:],positive=overlap>=.5,negative=overlap<.1,size_cell=int(xy[1]*4+xy[0]),size_valid=True))
                rows.append(views)
            data={k:np.asarray([[v[k] for v in pair] for pair in rows]) for k in rows[0][0]}; data["frame_id"]=np.asarray(c.sample_frames(e),int)
            c.audit_features(data,plan,e,references[e["sequence"]])
            c.save_cache(c.cache_path(root,e),data,c.feature_meta(plan,e))
        bindings={str(p):c.file_hash(p) for p in (root/"features").glob("*")}
        publish=c.seal(dict(schema=c.SCHEMA,status="COMPLETE_FROZEN_NATIVE_FEATURES",plan_fingerprint=plan["fingerprint"],file_sha256=bindings,backbone_and_head_frozen=True))
        c.publish_identical(root/"features/stage48_features.json",publish)
        return native,plan,references

    def test_full_training_resume_oof_rollout_probes_and_checkpoint_tamper(self):
        from stage48_stability_runtime import digest
        import stage48_stability_summary as summary
        with tempfile.TemporaryDirectory(prefix="stage48-test-",dir=Path(__file__).parent) as folder:
            root=Path(folder)
            with patch.dict(c.DESIGN,updates=2,smoke_updates=2,save_every=1,batch_pairs=1,samples_per_video=4,smoke_frames=4,
                minimum_rank_pairs=2,minimum_rank_videos=2,bootstrap_samples=50),contextlib.redirect_stdout(io.StringIO()):
                native,plan,references=self.fixture(root)
                source=lambda p,e: references[e["sequence"]]
                before=digest(native.rt.net)
                with patch.object(c,"load_plan",return_value=plan),patch.object(c,"source_native",side_effect=source):
                    held=set(plan["folds"][0]["held_out"]); seen=[]; reader=c.read_cache
                    def guarded(path,meta):
                        if meta.get("sequence") in held: raise AssertionError("Held-out arrays loaded by fit")
                        seen.append(meta.get("sequence")); return reader(path,meta)
                    with patch.object(c,"read_cache",side_effect=guarded): c.fit_pool(root,plan,0)
                    self.assertEqual(set(seen),set(plan["folds"][0]["fit"]))
                    smoke=experiment.train(native,plan,root,c.SEEDS[0],True)
                    self.assertEqual(smoke["status"],"PASS_HEAD_STABILITY_SMOKE")
                    for seed in c.SEEDS:
                        report=experiment.train(native,plan,root,seed)
                        self.assertEqual(len(report["records"]),8)
                        self.assertEqual(report,experiment.train(native,plan,root,seed))
                    experiment.evaluate(native,plan,root)
                    report=summary.summarize(root)
                    self.assertEqual(report["full_oof_trajectories"],96)
                    self.assertEqual(report["trained_models"],16)
                    self.assertFalse(report["external_generalization_established"])
                    self.assertEqual(digest(native.rt.net),before)
                    self.assertEqual(report,summary.summarize(root))
                    record=smoke["records"][0]; path=Path(record["checkpoint"])
                    with path.open("ab") as handle: handle.write(b"changed")
                    with self.assertRaisesRegex(ValueError,"changed"):
                        experiment.smoke_gate(plan,root)
                self.assertNotIn("gt",inspect.signature(__import__("stage48_stability_runtime").run_trace).parameters)

    def test_feature_collection_reproduces_frozen_head_labels(self):
        from stage48_stability_runtime import collect_features,head_output
        with tempfile.TemporaryDirectory(prefix="stage48-feature-",dir=Path(__file__).parent) as folder:
            root=Path(folder)
            with patch.dict(c.DESIGN,samples_per_video=4):
                native,plan,references=self.fixture(root)
                class Observer:
                    def initialize(self,image,box): native.initialize(image,box)
                    def assert_frozen(self): native.assert_frozen()
                    def capture(self,image,parent,frame):
                        result=native.step(image,parent)
                        with torch.no_grad(): out=head_output(native.rt.net.box_head,torch.tensor(fixture_feature()))
                        return dict(features=fixture_feature()[0],boxes=result["box"],layer=result["layer"],
                            raw=out["score_map"].numpy()[0,0],size=out["size_map"].numpy()[0],offset=out["offset_map"].numpy()[0])
                e=plan["entries"][0]
                got=collect_features(Observer(),plan,e,references[e["sequence"]])
                expected=c.read_cache(c.cache_path(root,e),c.feature_meta(plan,e))
                for key in got: np.testing.assert_array_equal(got[key],expected[key])


if __name__=="__main__": unittest.main()
