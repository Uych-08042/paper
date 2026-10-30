"""Current-state advancement, endpoint gradients and complete CPU OOF fixture."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import stage56_transition_common as c
import stage56_transition_experiment as experiment
try:
    import torch
    import cv2
except ImportError:
    torch=cv2=None

class PlanTests(unittest.TestCase):
    def test_stage_cache_identity_and_corruption(self):
        with tempfile.TemporaryDirectory(prefix="stage56-cache-",dir=Path(__file__).parent) as folder:
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
        from stage56_transition_runtime import configure,rollout_batch,objective,decode_image_boxes,temporary_head
        with tempfile.TemporaryDirectory(prefix="stage56-actor-",dir=Path(__file__).parent) as name,\
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
            loss,stats,penalty,first_penalty,error,response_error=objective(torch,head,rows,ends,plan,c.ARMS[1])
            self.assertGreater(float(first_penalty.detach()),0)
            self.assertGreater(float(penalty.detach()),0); self.assertLess(error,.001); self.assertEqual(response_error,0)
            losses={arm:objective(torch,head,rows,ends,plan,arm) for arm in c.ARMS}
            torch.testing.assert_close(losses["transition_preserve"][0]-losses["endpoint_half"][0],.5*first_penalty)
            torch.testing.assert_close(losses["endpoint_control"][0]-losses["endpoint_half"][0],.5*penalty)
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

    def test_direct_node_gradients_and_strength_control(self):
        from stage56_transition_runtime import node_penalty
        plan={"design":dict(c.DESIGN)}
        end=[dict(index=1,target=[10.,10.,20.,20.],qualified=True,
                  first_index=0,first_target=[10.,10.,20.,20.],first_qualified=True)]
        boxes=torch.tensor([[12.,11.,24.,22.],[13.,12.,23.,21.]],dtype=torch.float64,requires_grad=True)
        ep=node_penalty(torch,boxes,end,plan)
        first=node_penalty(torch,boxes,end,plan,True)
        grad_end=torch.autograd.grad(ep,boxes,retain_graph=True)[0]
        grad_first=torch.autograd.grad(first,boxes,retain_graph=True)[0]
        self.assertEqual(float(grad_end[0].abs().sum()),0.)
        self.assertGreater(float(grad_first[0].abs().sum()),0.)
        self.assertEqual(float(grad_first[1].abs().sum()),0.)
        control=.5*ep; treated=.5*ep+.5*first
        torch.testing.assert_close(treated-control,.5*first)
        end[0]["first_qualified"]=False
        self.assertEqual(float(node_penalty(torch,boxes,end,plan,True)),0.)
        self.assertEqual(c.WEIGHTS,{"endpoint_control":(1.,0.),"endpoint_half":(.5,0.),"transition_preserve":(.5,.5)})

    def test_complete_source54_fit_isolation_resume_budgets_and_parity(self):
        import stage51_state_common as old
        import stage52_branch_common as b52
        import stage52_branch_experiment as e52
        import stage52_branch_summary as s52
        import stage53_geometry_common as b53
        import stage53_geometry_experiment as e53
        import stage53_geometry_summary as s53
        import stage54_rollout_common as b54
        import stage54_rollout_experiment as e54
        import stage54_rollout_summary as s54
        import stage56_transition_runtime as runtime
        import stage56_transition_summary as summary
        from test_stage52_branch import TorchTests as SourceFixture,actual_small_head
        from stage48_stability_runtime import digest
        with tempfile.TemporaryDirectory(prefix="stage56-full-",dir=Path(__file__).parent) as name:
            folder=Path(name)
            with patch.dict(old.DESIGN,updates=2,smoke_updates=2,save_every=1,batch_pairs=1,smoke_frames=2,
                    bootstrap_samples=50,minimum_changed_pairs=1,minimum_changed_videos=1),patch.dict(old.prior.DESIGN,samples_per_video=4), \
                    patch("test_stage50_crossed.tiny_head",side_effect=actual_small_head),patch.dict(b52.DESIGN,bootstrap_samples=50,focus=["v0","v1"]), \
                    patch.dict(b53.DESIGN,bootstrap_samples=50,focus=["v0","v1"]),patch.dict(b54.DESIGN,bootstrap_samples=50,focus=["v0","v1"]), \
                    patch.dict(c.DESIGN,bootstrap_samples=50,focus=["v0","v1"]),contextlib.redirect_stdout(io.StringIO()):
                native,source48,source48root,refs,bindings,root51=SourceFixture().fixture(folder)
                with patch.object(old,"load_source",return_value=(source48,source48root,"report50",bindings)), \
                        patch.object(old.prior,"source_native",side_effect=lambda p,e:refs[e["sequence"]]):
                    previous=root51
                    for number,common,exp,summarizer in ((52,b52,e52,s52),(53,b53,e53,s53),(54,b54,e54,s54)):
                        root=folder/("stage"+str(number)); exp.prepare(previous,root); plan,inputs=common.load_plan(root)
                        exp.train(native,plan,inputs,root,c.SEEDS[0],True)
                        for seed in c.SEEDS: exp.train(native,plan,inputs,root,seed)
                        exp.evaluate(native,plan,root); summarizer.summarize(root); previous=root
                    self.assertEqual(len(c.load_source(previous)["records"]),8)
                    output=folder/"stage56"; experiment.prepare(previous,output); plan,inputs=c.load_plan(output)
                    held=set(plan["folds"][0]["held_out"]); seen=set(); reader=old.read_cache
                    def read_fit(path,meta):
                        self.assertNotIn(meta.get("sequence"),held); seen.add(meta.get("sequence")); return reader(path,meta)
                    with patch.object(old,"read_cache",side_effect=read_fit):
                        pool,pool_bindings=c.fit_pool(plan,inputs,0)
                    self.assertEqual(seen,set(plan["folds"][0]["fit"]))
                    original=digest(native.rt.net)
                    self.assertEqual(experiment.train(native,plan,inputs,output,c.SEEDS[0],True)["status"],"PASS_SHORT_ROLLOUT_SMOKE")
                    real_save=runtime.save_step
                    def interrupt(torch,path,state):
                        real_save(torch,path,state); raise RuntimeError("simulated first-transition interruption")
                    with patch.object(runtime,"save_step",side_effect=interrupt):
                        with self.assertRaisesRegex(RuntimeError,"simulated"):
                            runtime.train_arm(native,plan,output,pool,pool_bindings,0,c.SEEDS[0],c.ARMS[0])
                    resumed=runtime.train_arm(native,plan,output,pool,pool_bindings,0,c.SEEDS[0],c.ARMS[0])
                    direct=runtime.train_arm(native,plan,folder/"direct",pool,pool_bindings,0,c.SEEDS[0],c.ARMS[0])
                    self.assertEqual(resumed["final_head_sha256"],direct["final_head_sha256"])
                    self.assertTrue(resumed["reproduces_stage54_control"])
                    for seed in c.SEEDS:
                        report=experiment.train(native,plan,inputs,output,seed)
                        self.assertEqual(report,experiment.train(native,plan,inputs,output,seed))
                    experiment.evaluate(native,plan,output)
                    with patch("stage48_stability_runtime.run_trace",side_effect=AssertionError("Reuse committed traces")):
                        experiment.evaluate(native,plan,output)
                    result=summary.summarize(output); self.assertEqual(result,summary.summarize(output))
                    self.assertEqual(result["new_models"],12); self.assertEqual(result["inference_head_calls"],72*4)
                    self.assertEqual(result["training_network_calls"],12*2*8); self.assertEqual(result["training_head_only_calls"],12*2*4)
                    records,_=experiment.model_records(plan,output)
                    self.assertEqual(sum(r["reproduces_stage54_control"] is True for r,_ in records),4)
                    self.assertTrue(all(r["maximum_first_gradient"]>0 for r,_ in records))
                    self.assertEqual(digest(native.rt.net),original)
                    table=output/"analysis/stage56_policy_summary.csv"; saved=table.read_bytes(); table.write_bytes(saved+b"changed")
                    with self.assertRaisesRegex(ValueError,"differs"): summary.summarize(output)
                    table.write_bytes(saved)
                    model=Path(resumed["checkpoint"]); saved=model.read_bytes(); model.write_bytes(saved+b"changed")
                    with self.assertRaisesRegex(ValueError,"changed"): experiment.model_records(plan,output)
                    model.write_bytes(saved)
                    source_table=previous/"analysis/stage54_seed_video_summary.csv"
                    saved=source_table.read_bytes(); source_table.write_bytes(saved+b"changed")
                    with self.assertRaises(ValueError): c.load_source(previous)
                    source_table.write_bytes(saved)

if __name__=="__main__": unittest.main()
