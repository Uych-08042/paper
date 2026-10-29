"""Current-state advancement, endpoint gradients and complete CPU OOF fixture."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import stage54_rollout_common as c
import stage54_rollout_experiment as experiment
try:
    import torch
    import cv2
except ImportError:
    torch=cv2=None

class PlanTests(unittest.TestCase):
    def test_stage_cache_identity_and_corruption(self):
        with tempfile.TemporaryDirectory(prefix="stage54-cache-",dir=Path(__file__).parent) as folder:
            p=Path(folder)/"trace.npz"; c.save_cache(p,dict(a=np.arange(3)),dict(fixture=True))
            np.testing.assert_array_equal(c.read_cache(p,dict(fixture=True))["a"],np.arange(3))
            with self.assertRaisesRegex(ValueError,"changed"): c.read_cache(p,dict(fixture=False))
            p.write_bytes(p.read_bytes()+b"changed")
            with self.assertRaisesRegex(ValueError,"changed"): c.read_cache(p,dict(fixture=True))

@unittest.skipUnless(torch is not None,"CPU Torch/OpenCV required; skipped is not a passed rollout audit")
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_actor_endpoint_gradient_and_future_gt_independence(self):
        from test_stage50_crossed import TorchTests as NativeFixture
        from test_stage52_branch import actual_small_head
        from stage48_stability_runtime import make_head,digest
        from stage54_rollout_runtime import configure,rollout_batch,objective,decode_image_boxes,temporary_head
        with tempfile.TemporaryDirectory(prefix="stage54-actor-",dir=Path(__file__).parent) as name,\
                patch("test_stage50_crossed.tiny_head",side_effect=actual_small_head):
            inputs,_,native,capture,_=NativeFixture().fixture(Path(name)); capture.handle.remove(); capture.pre_handle.remove()
            entry,arrays=inputs["focus"]["fixture"]; plan=dict(inputs["plan"],design=dict(c.DESIGN))
            anchor=dict(sequence=entry["sequence"],frame=1,parents=[entry["initial_box"]]*2,native_expected=arrays["native"]["boxes"][:2].tolist())
            head=make_head(native.rt); configure(head,c.ARMS[1]); original=digest(native.rt.net)
            rows,ends,audit,_=rollout_batch(native,head,plan,{entry["sequence"]:entry},[anchor])
            self.assertEqual(audit["reference_calls"],4); self.assertEqual(audit["student_calls"],4)
            self.assertLess(float(objective(torch,head,rows,ends,plan,c.ARMS[1])[2].detach()),1e-6)
            with torch.no_grad(): head.conv5_size.bias.add_(.02); head.conv5_offset.bias.add_(.02)
            rows,ends,audit,_=rollout_batch(native,head,plan,{entry["sequence"]:entry},[anchor])
            np.testing.assert_array_equal(rows[1]["parent"],rows[0]["boxes"])
            self.assertGreater(audit["changed_successor_parents"],0)
            loss,stats,penalty,error,response_error=objective(torch,head,rows,ends,plan,c.ARMS[1])
            self.assertGreater(float(penalty.detach()),0); self.assertLess(error,.001); self.assertEqual(response_error,0)
            penalty.backward()
            for suffix in ("size","offset"):
                self.assertGreater(sum(float(p.grad.abs().sum()) for n,p in head.named_parameters() if suffix in n and p.grad is not None),0)
            self.assertTrue(all(p.grad is None for n,p in head.named_parameters() if "_ctr" in n))
            changed=copy.deepcopy(entry); changed["ground_truth"][1:]=[[10000,10000,20,20] for _ in changed["ground_truth"][1:]]
            again,other,_,_=rollout_batch(native,head,plan,{entry["sequence"]:changed},[anchor])
            for a,b in zip(rows,again): np.testing.assert_array_equal(a["boxes"],b["boxes"])
            self.assertFalse(any(e["qualified"] for e in other)); self.assertEqual(digest(native.rt.net),original)
            # Endpoint gradients do not differentiate the first-frame parent.
            size=torch.full((2,2,4,4),.25,requires_grad=True); offset=torch.zeros(2,2,4,4,requires_grad=True)
            decoded=decode_image_boxes(torch,dict(size_map=size,offset_map=offset),rows[:2],256)
            decoded[1].sum().backward(); self.assertEqual(float(size.grad[0].abs().sum()),0); self.assertEqual(float(offset.grad[0].abs().sum()),0)
            with self.assertRaisesRegex(RuntimeError,"rollback"):
                with temporary_head(native,head.state_dict()): raise RuntimeError("rollback")
            self.assertEqual(digest(native.rt.net),original)

    def test_complete_source53_fit_isolation_resume_and_live_budgets(self):
        import stage51_state_common as old
        import stage52_branch_common as b52
        import stage52_branch_experiment as e52
        import stage52_branch_summary as s52
        import stage53_geometry_common as b53
        import stage53_geometry_experiment as e53
        import stage53_geometry_summary as s53
        import stage54_rollout_runtime as runtime
        import stage54_rollout_summary as summary
        from test_stage52_branch import TorchTests as SourceFixture,actual_small_head
        from stage48_stability_runtime import digest
        with tempfile.TemporaryDirectory(prefix="stage54-full-",dir=Path(__file__).parent) as name:
            folder=Path(name)
            with patch.dict(old.DESIGN,updates=3,smoke_updates=2,save_every=1,batch_pairs=1,smoke_frames=2,
                    bootstrap_samples=50,minimum_changed_pairs=1,minimum_changed_videos=1),patch.dict(old.prior.DESIGN,samples_per_video=4),\
                    patch("test_stage50_crossed.tiny_head",side_effect=actual_small_head),patch.dict(b52.DESIGN,bootstrap_samples=50,focus=["v0","v1"]),\
                    patch.dict(b53.DESIGN,bootstrap_samples=50,focus=["v0","v1"]),patch.dict(c.DESIGN,bootstrap_samples=50,focus=["v0","v1"]),\
                    contextlib.redirect_stdout(io.StringIO()):
                native,source48,source48root,refs,bindings,root51=SourceFixture().fixture(folder)
                with patch.object(old,"load_source",return_value=(source48,source48root,"report50",bindings)),\
                        patch.object(old.prior,"source_native",side_effect=lambda p,e:refs[e["sequence"]]):
                    root52=folder/"stage52"; e52.prepare(root51,root52); p52,i52=b52.load_plan(root52)
                    e52.train(native,p52,i52,root52,c.SEEDS[0],True)
                    for seed in c.SEEDS: e52.train(native,p52,i52,root52,seed)
                    e52.evaluate(native,p52,root52); s52.summarize(root52)
                    root53=folder/"stage53"; e53.prepare(root52,root53); p53,i53=b53.load_plan(root53)
                    e53.train(native,p53,i53,root53,c.SEEDS[0],True)
                    for seed in c.SEEDS: e53.train(native,p53,i53,root53,seed)
                    e53.evaluate(native,p53,root53); s53.summarize(root53)
                    self.assertEqual(len(c.load_source(root53)["records"]),8)
                    output=folder/"stage54"; experiment.prepare(root53,output); plan,inputs=c.load_plan(output)
                    self.assertEqual(plan["anchor_pools"][0]["pairs"],18) # 6 videos x (4 samples minus final frame)
                    held=set(plan["folds"][0]["held_out"]); seen=set(); read=old.read_cache; oldread=old.prior.read_cache
                    def guard(fn):
                        def checked(path,meta):
                            self.assertNotIn(meta.get("sequence"),held); seen.add(meta.get("sequence")); return fn(path,meta)
                        return checked
                    with patch.object(old,"read_cache",side_effect=guard(read)),patch.object(old.prior,"read_cache",side_effect=guard(oldread)):
                        pool,pool_bindings=c.fit_pool(plan,inputs,0)
                    self.assertEqual(seen,set(plan["folds"][0]["fit"]))
                    original=digest(native.rt.net)
                    self.assertEqual(experiment.train(native,plan,inputs,output,c.SEEDS[0],True)["status"],"PASS_SHORT_ROLLOUT_SMOKE")
                    real_save=runtime.save_step
                    def interrupt(torch,path,state): real_save(torch,path,state); raise RuntimeError("simulated rollout interruption")
                    with patch.object(runtime,"save_step",side_effect=interrupt):
                        with self.assertRaisesRegex(RuntimeError,"simulated"):
                            runtime.train_arm(native,plan,output,pool,pool_bindings,0,c.SEEDS[0],c.ARMS[0])
                    resumed=runtime.train_arm(native,plan,output,pool,pool_bindings,0,c.SEEDS[0],c.ARMS[0])
                    direct=runtime.train_arm(native,plan,folder/"direct",pool,pool_bindings,0,c.SEEDS[0],c.ARMS[0])
                    self.assertEqual(resumed["final_head_sha256"],direct["final_head_sha256"])
                    for seed in c.SEEDS:
                        report=experiment.train(native,plan,inputs,output,seed)
                        self.assertEqual(report,experiment.train(native,plan,inputs,output,seed))
                    experiment.evaluate(native,plan,output)
                    with patch("stage48_stability_runtime.run_trace",side_effect=AssertionError("Reuse committed traces")):
                        experiment.evaluate(native,plan,output)
                    result=summary.summarize(output); self.assertEqual(result,summary.summarize(output))
                    self.assertEqual(result["new_models"],8); self.assertEqual(result["inference_head_calls"],48*4)
                    self.assertEqual(result["training_network_calls"],8*3*8); self.assertEqual(result["training_head_only_calls"],8*3*4)
                    self.assertEqual(digest(native.rt.net),original)
                    table=output/"analysis/stage54_policy_summary.csv"; saved=table.read_bytes(); table.write_bytes(saved+b"changed")
                    with self.assertRaisesRegex(ValueError,"differs"): summary.summarize(output)
                    table.write_bytes(saved)
                    p=Path(resumed["checkpoint"]); p.write_bytes(p.read_bytes()+b"changed")
                    with self.assertRaisesRegex(ValueError,"changed"): experiment.model_records(plan,output)
                    p=root53/"analysis/stage53_seed_video_summary.csv"; p.write_bytes(p.read_bytes()+b"changed")
                    with self.assertRaisesRegex(ValueError,"changed"): c.load_source(root53)

if __name__=="__main__": unittest.main()
