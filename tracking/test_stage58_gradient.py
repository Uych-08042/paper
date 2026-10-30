"""Current-state advancement, endpoint gradients and complete CPU OOF fixture."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import stage58_gradient_common as c
import stage58_gradient_experiment as experiment
try:
    import torch
    import cv2
except ImportError:
    torch=cv2=None

class PlanTests(unittest.TestCase):
    def test_stage_cache_identity_and_corruption(self):
        with tempfile.TemporaryDirectory(prefix="stage58-cache-",dir=Path(__file__).parent) as folder:
            p=Path(folder)/"trace.npz"; c.save_cache(p,dict(a=np.arange(3)),dict(fixture=True))
            np.testing.assert_array_equal(c.read_cache(p,dict(fixture=True))["a"],np.arange(3))
            with self.assertRaisesRegex(ValueError,"changed"): c.read_cache(p,dict(fixture=False))
            p.write_bytes(p.read_bytes()+b"changed")
            with self.assertRaisesRegex(ValueError,"changed"): c.read_cache(p,dict(fixture=True))

@unittest.skipUnless(torch is not None,"CPU Torch/OpenCV required; skipped is not a passed rollout audit")
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_projection_math_zero_cases_and_adamw_limit(self):
        from stage58_gradient_runtime import gradient_rule
        ordinary=torch.tensor([1.,0.]); anchor=torch.tensor([-1.,1.])
        projected,audit,error=gradient_rule(torch,ordinary,anchor,"projected_anchor",c.DESIGN)
        matched,_,_=gradient_rule(torch,ordinary,anchor,"norm_matched",c.DESIGN)
        torch.testing.assert_close(projected,torch.tensor([0.,1.]))
        torch.testing.assert_close(matched,anchor/2**.5)
        self.assertAlmostEqual(float(matched.norm()),float(projected.norm()),places=6)
        self.assertLess(float(torch.dot(ordinary,matched)),0)
        self.assertEqual(float(torch.dot(ordinary,projected)),0)
        self.assertEqual(audit["conflict_batches"],1)
        self.assertEqual(audit["actual_modified_batches"],1)
        self.assertLess(max(error.values()),c.DESIGN["gradient_audit_rtol"])
        for gs,gp in (([0.,0.],[-1.,2.]),([1.,0.],[0.,0.]),([1.,0.],[1.,1.])):
            gs,gp=torch.tensor(gs),torch.tensor(gp)
            for arm in c.ARMS:
                got,stats,_=gradient_rule(torch,gs,gp,arm,c.DESIGN)
                torch.testing.assert_close(got,gp); self.assertEqual(stats["conflict_batches"],0)
        zero,_,_=gradient_rule(torch,ordinary,-ordinary,"projected_anchor",c.DESIGN)
        self.assertEqual(float(zero.abs().sum()),0)
        with self.assertRaises(ValueError): gradient_rule(torch,ordinary,torch.tensor([float("nan"),1.]),c.ARMS[-1],c.DESIGN)
        with self.assertRaises(ValueError): gradient_rule(torch,ordinary,anchor[:1],c.ARMS[-1],c.DESIGN)
        # Old momentum can still oppose gS despite projected gP dot gS == 0.
        param=torch.nn.Parameter(torch.zeros(2)); opt=torch.optim.AdamW([param],lr=.01,weight_decay=0.)
        param.grad=torch.tensor([-10.,0.]); opt.step(); before=param.detach().clone()
        param.grad=ordinary+projected; opt.step()
        self.assertLess(-float(torch.dot(ordinary,param.detach()-before)),0)

    def test_noop_projection_preserves_original_backward_bits(self):
        from stage58_gradient_runtime import gradient_vectors, gradient_rule, backward_gradient
        for arm in c.ARMS:
            p=torch.nn.Parameter(torch.tensor([.2,.3,.7],dtype=torch.float32))
            shared=p.sin().square()*torch.tensor([.003,1.,1000.])
            ordinary=shared.sum(); penalty=(shared*2).sum(); loss=ordinary+1.*penalty
            expected=torch.autograd.grad(loss,p,retain_graph=True)[0]
            gs,gp=gradient_vectors(torch,ordinary,penalty,[p])
            applied,audit,_=gradient_rule(torch,gs,gp,arm,c.DESIGN)
            self.assertEqual(audit["actual_modified_batches"],0)
            backward_gradient(loss,[p],gs,applied,audit["actual_modified_batches"])
            torch.testing.assert_close(p.grad,expected,rtol=0,atol=0)

    def test_real_actor_objective_gradients_and_fit_gt_isolation(self):
        from test_stage50_crossed import TorchTests as NativeFixture
        from test_stage52_branch import actual_small_head
        from stage48_stability_runtime import make_head,digest
        from stage58_gradient_runtime import configure,rollout_batch,objective,gradient_vectors,gradient_rule,node_penalty
        from stage57_quality_runtime import objective as control_objective
        with tempfile.TemporaryDirectory(prefix="stage58-actor-",dir=Path(__file__).parent) as name, \
                patch("test_stage50_crossed.tiny_head",side_effect=actual_small_head):
            inputs,_,native,capture,_=NativeFixture().fixture(Path(name)); capture.handle.remove(); capture.pre_handle.remove()
            entry,arrays=inputs["focus"]["fixture"]; plan=dict(inputs["plan"],design=dict(c.DESIGN))
            anchor=dict(sequence=entry["sequence"],frame=1,parents=[entry["initial_box"]]*2,native_expected=arrays["native"]["boxes"][:2].tolist())
            head=make_head(native.rt); configure(head,"projected_anchor"); original=digest(native.rt.net)
            with torch.no_grad(): head.conv5_size.bias.add_(.02); head.conv5_offset.bias.add_(.02)
            rows,ends,audit,_=rollout_batch(native,head,plan,{entry["sequence"]:entry},[anchor])
            np.testing.assert_array_equal(rows[1]["parent"],rows[0]["boxes"])
            self.assertEqual(audit["student_calls"],4); self.assertEqual(audit["reference_calls"],4)
            forced=copy.deepcopy(ends)
            for e in forced: e["qualified"]=True
            head_calls=[]
            hook=head.register_forward_hook(lambda *args:head_calls.append(1))
            loss,stats,ordinary,penalty,de,re=objective(torch,head,rows,forced,plan,"projected_anchor")
            hook.remove(); self.assertEqual(len(head_calls),len(rows))
            previous=control_objective(torch,head,rows,forced,plan,"endpoint_control",[.9,.1])
            self.assertEqual(float(loss),float(previous[0])); self.assertEqual(stats,previous[1])
            self.assertLessEqual(de,plan["design"]["decode_atol"]); self.assertEqual(re,0.)
            parameters=[p for p in head.parameters() if p.requires_grad]
            gs,gp=gradient_vectors(torch,ordinary,penalty,parameters)
            self.assertGreater(float(gs.norm()),0); self.assertGreater(float(gp.norm()),0)
            for arm in c.ARMS:
                applied,_,_=gradient_rule(torch,gs,gp,arm,c.DESIGN)
                self.assertTrue(torch.isfinite(applied).all())
            names=configure(head,"endpoint_control")[1]
            loss.backward()
            self.assertTrue(all(p.grad is None for n,p in head.named_parameters() if n not in names))
            boxes=torch.tensor([[12.,11.,24.,22.],[13.,12.,23.,21.]],dtype=torch.float64,requires_grad=True)
            nodes=[dict(index=1,target=[10.,10.,20.,20.],qualified=True)]
            gradient=torch.autograd.grad(node_penalty(torch,boxes,nodes,plan),boxes)[0]
            self.assertEqual(float(gradient[0].abs().sum()),0); self.assertGreater(float(gradient[1].abs().sum()),0)
            changed=copy.deepcopy(entry)
            changed["ground_truth"][1:]=[[1000.,1000.,2.,2.]]*(len(changed["ground_truth"])-1)
            other,_,_,_=rollout_batch(native,head,plan,{entry["sequence"]:changed},[anchor])
            for x,y in zip(rows,other):
                for key in ("boxes","parent","raw","features"):
                    np.testing.assert_array_equal(x[key],y[key])
            self.assertEqual(digest(native.rt.net),original)

    def test_complete_source57_fit_isolation_resume_budgets_and_parity(self):
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
        import stage56_transition_common as b56
        import stage56_transition_experiment as e56
        import stage56_transition_summary as s56
        import stage57_quality_common as b57
        import stage57_quality_experiment as e57
        import stage57_quality_summary as s57
        import stage58_gradient_runtime as runtime
        import stage58_gradient_summary as summary
        from test_stage52_branch import TorchTests as SourceFixture,actual_small_head
        from stage48_stability_runtime import digest
        with tempfile.TemporaryDirectory(prefix="stage58-full-",dir=Path(__file__).parent) as name:
            folder=Path(name)
            with patch.dict(old.DESIGN,updates=2,smoke_updates=2,save_every=1,batch_pairs=1,smoke_frames=2,
                    bootstrap_samples=50,minimum_changed_pairs=1,minimum_changed_videos=1),patch.dict(old.prior.DESIGN,samples_per_video=4), \
                    patch("test_stage50_crossed.tiny_head",side_effect=actual_small_head),patch.dict(b52.DESIGN,bootstrap_samples=50,focus=["v0","v1"]), \
                    patch.dict(b53.DESIGN,bootstrap_samples=50,focus=["v0","v1"]),patch.dict(b54.DESIGN,bootstrap_samples=50,focus=["v0","v1"]), \
                    patch.dict(b56.DESIGN,bootstrap_samples=50,focus=["v0","v1"]),patch.dict(b57.DESIGN,bootstrap_samples=50,focus=["v0","v1"]),patch.dict(c.DESIGN,bootstrap_samples=50,focus=["v0","v1"]),contextlib.redirect_stdout(io.StringIO()):
                native,source48,source48root,refs,bindings,root51=SourceFixture().fixture(folder)
                with patch.object(old,"load_source",return_value=(source48,source48root,"report50",bindings)), \
                        patch.object(old.prior,"source_native",side_effect=lambda p,e:refs[e["sequence"]]):
                    previous=root51
                    for number,common,exp,summarizer in ((52,b52,e52,s52),(53,b53,e53,s53),(54,b54,e54,s54),(56,b56,e56,s56),(57,b57,e57,s57)):
                        root=folder/("stage"+str(number)); exp.prepare(previous,root); plan,inputs=common.load_plan(root)
                        exp.train(native,plan,inputs,root,c.SEEDS[0],True)
                        for seed in c.SEEDS: exp.train(native,plan,inputs,root,seed)
                        exp.evaluate(native,plan,root); summarizer.summarize(root); previous=root
                    self.assertEqual(len(c.load_source(previous)["records"]),12)
                    output=folder/"stage58"; experiment.prepare(previous,output); plan,inputs=c.load_plan(output)
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
                    for arm in ("endpoint_control","projected_anchor"):
                        with patch.object(runtime,"save_step",side_effect=interrupt):
                            with self.assertRaisesRegex(RuntimeError,"simulated"):
                                runtime.train_arm(native,plan,output,pool,pool_bindings,0,c.SEEDS[0],arm)
                        resumed=runtime.train_arm(native,plan,output,pool,pool_bindings,0,c.SEEDS[0],arm)
                        direct=runtime.train_arm(native,plan,folder/"direct",pool,pool_bindings,0,c.SEEDS[0],arm)
                        self.assertEqual(resumed["final_head_sha256"],direct["final_head_sha256"])
                        self.assertEqual(resumed["gradient_totals"],direct["gradient_totals"])
                        if arm=="endpoint_control": self.assertTrue(resumed["reproduces_stage57_control"])
                        x=torch.load(resumed["checkpoint"],weights_only=False)["optimizer"]["state"]
                        y=torch.load(direct["checkpoint"],weights_only=False)["optimizer"]["state"]
                        for i in x:
                            for key in x[i]: torch.testing.assert_close(x[i][key],y[i][key],rtol=0,atol=0)
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
                    self.assertEqual(sum(r["reproduces_stage57_control"] is True for r,_ in records),4)
                    self.assertTrue(all(r["maximum_endpoint_gradient"]>0 for r,_ in records))
                    self.assertTrue(all(r["quality_totals"]["retained_endpoints"]==r["totals"]["qualified_endpoints"] for r,_ in records))
                    self.assertEqual(result["role"],"consumed_development_projected_endpoint_gradient")
                    altered=copy.deepcopy(records[0][0]); altered["quality_totals"]["retained_endpoints"]-=1
                    with self.assertRaisesRegex(ValueError,"scope/mass/audit"):
                        experiment.validate_gradients(plan,altered,{})
                    self.assertEqual(digest(native.rt.net),original)
                    table=output/"analysis/stage58_policy_summary.csv"; saved=table.read_bytes(); table.write_bytes(saved+b"changed")
                    with self.assertRaisesRegex(ValueError,"differs"): summary.summarize(output)
                    table.write_bytes(saved)
                    model=Path(resumed["checkpoint"]); saved=model.read_bytes(); model.write_bytes(saved+b"changed")
                    with self.assertRaisesRegex(ValueError,"changed"): experiment.model_records(plan,output)
                    model.write_bytes(saved)
                    source_table=previous/"analysis/stage57_seed_video_summary.csv"
                    saved=source_table.read_bytes(); source_table.write_bytes(saved+b"changed")
                    with self.assertRaises(ValueError): c.load_source(previous)
                    source_table.write_bytes(saved)

if __name__=="__main__": unittest.main()
