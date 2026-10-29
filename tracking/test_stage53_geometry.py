"""Native geometry gradients, fit-only qualification and end-to-end CPU checks."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import stage53_geometry_common as c
import stage53_geometry_experiment as experiment
try:
    import torch
    import cv2
except ImportError:
    torch=cv2=None

class GeometryTests(unittest.TestCase):
    def sample(self):
        return dict(sequence="fit",frame=1,raw=np.ones((3,1,4,4),np.float32),size=np.full((3,2,4,4),.25,np.float32),
            offset=np.zeros((3,2,4,4),np.float32),target=np.array([[.375,.375,.25,.25]]*3,np.float32))

    def test_continuous_hann_qualification_mass_and_cache_rejection(self):
        sample=self.sample(); window=np.zeros((4,4),np.float32); window[2,2]=1
        sample["raw"][0,0,0,0]=10 # raw argmax differs from Hann-selected cell.
        sample["target"][2]=[10,10,.25,.25]
        first=c.teacher_state(sample,0,window,c.DESIGN)
        self.assertEqual(first["cell"],10); self.assertEqual(first["iou"],1.)
        support,rows=c.qualify_pool([sample],window,c.DESIGN)
        self.assertEqual(support["reliable_fraction"],.5)
        self.assertEqual(sum(r["reliable"]*support["reliable_scale"] for r in rows),len(rows))
        sample["target"][2]=sample["target"][0]
        with self.assertRaisesRegex(ValueError,"both reliable"): c.qualify_pool([sample],window,c.DESIGN)
        with tempfile.TemporaryDirectory(prefix="stage53-cache-",dir=Path(__file__).parent) as folder:
            p=Path(folder)/"trace.npz"; c.save_cache(p,dict(a=np.arange(3)),dict(fixture=True))
            np.testing.assert_array_equal(c.read_cache(p,dict(fixture=True))["a"],np.arange(3))
            p.write_bytes(p.read_bytes()+b"changed")
            with self.assertRaisesRegex(ValueError,"changed"): c.read_cache(p,dict(fixture=True))

@unittest.skipUnless(torch is not None,"CPU Torch/OpenCV needed; a skip does not verify training")
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_real_center_anchor_gradients_and_gating(self):
        from test_stage52_branch import actual_small_head
        from test_stage50_crossed import feature
        from stage48_stability_runtime import head_output
        from stage53_geometry_runtime import objective,configure,frozen_digest
        head=actual_small_head(); configure(head,"anchor_reliable")
        with torch.no_grad():
            for name,module in head.named_modules():
                if isinstance(module,torch.nn.Conv2d) and name.split(".")[0].endswith("_size"):
                    module.weight[:,0,module.weight.shape[2]//2,module.weight.shape[3]//2]=.5
        features=torch.cat([feature(),feature(),feature()*2],0)
        with torch.no_grad(): ref=head_output(head,features)
        wh=ref["size_map"][0,:,2,2].numpy(); target=np.array([np.r_[.5-wh/2,wh]]*3,np.float32); target[2]=[10,10,.2,.2]
        sample=dict(sequence="fit",frame=1,features=features.numpy(),target=target,
            raw=ref["score_map"].numpy(),size=ref["size_map"].numpy(),offset=ref["offset_map"].numpy())
        plan=dict(window=np.ones((4,4)).tolist(),design=dict(c.DESIGN)); support,_=c.qualify_pool([sample],plan["window"],plan["design"])
        _,_,zero,_=objective(torch,head,[sample],plan,"anchor_reliable",support)
        self.assertEqual(float(zero.detach()),0.)
        with torch.no_grad(): head.conv5_size.bias.add_(.05); head.conv5_offset.bias.add_(.01)
        before=frozen_digest(head,[n for n,p in head.named_parameters() if p.requires_grad])
        total,stats,penalty,audit=objective(torch,head,[sample],plan,"anchor_reliable",support)
        self.assertEqual(audit["views"],2); self.assertEqual(audit["reliable"],1); self.assertEqual(audit["weight_mass"],2.)
        self.assertEqual(int(stats["supervised_count"]),2); self.assertGreater(float(penalty.detach()),0)
        penalty.backward()
        for suffix in ("size","offset"):
            self.assertGreater(sum(float(p.grad.abs().sum()) for n,p in head.named_parameters() if suffix in n and p.grad is not None),0)
        self.assertTrue(all(p.grad is None for n,p in head.named_parameters() if "_ctr" in n))
        changed=copy.deepcopy(sample); changed["target"][2]=[20,20,.2,.2]
        self.assertEqual(float(objective(torch,head,[changed],plan,"anchor_reliable",support)[2].detach()),float(penalty.detach()))
        opt=torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],lr=1e-4); opt.step()
        self.assertEqual(before,frozen_digest(head,[n for n,p in head.named_parameters() if p.requires_grad]))

    def test_complete_source52_fit_isolation_resume_and_oof(self):
        import stage51_state_common as old
        import stage52_branch_common as branch
        import stage52_branch_experiment as branch_experiment
        import stage52_branch_summary as branch_summary
        import stage53_geometry_runtime as runtime
        import stage53_geometry_summary as summary
        from test_stage52_branch import TorchTests as SourceFixture,actual_small_head
        from stage48_stability_runtime import digest
        with tempfile.TemporaryDirectory(prefix="stage53-full-",dir=Path(__file__).parent) as name:
            folder=Path(name)
            with patch.dict(old.DESIGN,updates=3,smoke_updates=2,save_every=1,batch_pairs=1,smoke_frames=2,
                    bootstrap_samples=50,minimum_changed_pairs=1,minimum_changed_videos=1),patch.dict(old.prior.DESIGN,samples_per_video=4),\
                    patch("test_stage50_crossed.tiny_head",side_effect=actual_small_head),patch.dict(branch.DESIGN,bootstrap_samples=50,focus=["v0","v1"]),\
                    patch.dict(c.DESIGN,bootstrap_samples=50,focus=["v0","v1"]),contextlib.redirect_stdout(io.StringIO()):
                native,source48,source48root,refs,bindings,root51=SourceFixture().fixture(folder)
                with patch.object(old,"load_source",return_value=(source48,source48root,"report50",bindings)),\
                        patch.object(old.prior,"source_native",side_effect=lambda p,e:refs[e["sequence"]]):
                    root52=folder/"stage52"; branch_experiment.prepare(root51,root52); plan52,inputs52=branch.load_plan(root52)
                    branch_experiment.train(native,plan52,inputs52,root52,c.SEEDS[0],True)
                    for seed in c.SEEDS: branch_experiment.train(native,plan52,inputs52,root52,seed)
                    branch_experiment.evaluate(native,plan52,root52); branch_summary.summarize(root52)
                    source=c.load_source(root52); self.assertEqual(len(source["records"]),8)
                    output=folder/"stage53"; experiment.prepare(root52,output); plan,inputs=c.load_plan(output)
                    self.assertTrue(all(0<r["reliable_fraction"]<1 for r in plan["support"]))
                    frozen=digest(native.rt.net); held=set(plan["folds"][0]["held_out"]); seen=set()
                    reader=old.read_cache; oldreader=old.prior.read_cache
                    def guard(fn):
                        def checked(path,meta):
                            self.assertNotIn(meta.get("sequence"),held); seen.add(meta.get("sequence")); return fn(path,meta)
                        return checked
                    with patch.object(old,"read_cache",side_effect=guard(reader)),patch.object(old.prior,"read_cache",side_effect=guard(oldreader)):
                        pool,pool_bindings=c.fit_pool(plan,inputs,0); support,_=c.qualify_pool(pool,plan["window"],plan["design"])
                    self.assertEqual(seen,set(plan["folds"][0]["fit"]))
                    self.assertEqual(dict(fold=0,**support),plan["support"][0])
                    smoke=experiment.train(native,plan,inputs,output,c.SEEDS[0],True)
                    self.assertEqual(smoke["status"],"PASS_GEOMETRY_PRESERVATION_SMOKE")
                    real_save=runtime.save_step
                    def interrupt(torch,path,state): real_save(torch,path,state); raise RuntimeError("simulated anchor interruption")
                    with patch.object(runtime,"save_step",side_effect=interrupt):
                        with self.assertRaisesRegex(RuntimeError,"simulated"):
                            runtime.train_arm(native,plan,output,pool,pool_bindings,0,c.SEEDS[0],c.ARMS[0])
                    resumed=runtime.train_arm(native,plan,output,pool,pool_bindings,0,c.SEEDS[0],c.ARMS[0])
                    direct=runtime.train_arm(native,plan,folder/"direct",pool,pool_bindings,0,c.SEEDS[0],c.ARMS[0])
                    self.assertEqual(resumed["final_head_sha256"],direct["final_head_sha256"])
                    for seed in c.SEEDS:
                        r=experiment.train(native,plan,inputs,output,seed); self.assertEqual(r,experiment.train(native,plan,inputs,output,seed))
                    experiment.evaluate(native,plan,output)
                    with patch("stage48_stability_runtime.run_trace",side_effect=AssertionError("Cached trajectories must be reused")):
                        experiment.evaluate(native,plan,output)
                    result=summary.summarize(output); self.assertEqual(result,summary.summarize(output))
                    self.assertEqual(result["new_models"],8); self.assertEqual(result["inference_head_calls"],48*4)
                    self.assertEqual(result["reused_full_trajectories"],60); self.assertFalse(result["new_feature_collection"])
                    self.assertEqual(digest(native.rt.net),frozen)
                    table=output/"analysis/stage53_policy_summary.csv"; original=table.read_bytes(); table.write_bytes(original+b"changed")
                    with self.assertRaisesRegex(ValueError,"differs"): summary.summarize(output)
                    table.write_bytes(original)
                    checkpoint=Path(resumed["checkpoint"]); checkpoint.write_bytes(checkpoint.read_bytes()+b"changed")
                    with self.assertRaisesRegex(ValueError,"changed"): experiment.model_records(plan,output)
                    path=root52/"analysis/stage52_seed_video_summary.csv"; path.write_bytes(path.read_bytes()+b"changed")
                    with self.assertRaisesRegex(ValueError,"changed"): c.load_source(root52)

if __name__=="__main__": unittest.main()
