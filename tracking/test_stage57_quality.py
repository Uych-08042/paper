"""Current-state advancement, endpoint gradients and complete CPU OOF fixture."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import stage57_quality_common as c
import stage57_quality_experiment as experiment
try:
    import torch
    import cv2
except ImportError:
    torch=cv2=None

class PlanTests(unittest.TestCase):
    def test_stage_cache_identity_and_corruption(self):
        with tempfile.TemporaryDirectory(prefix="stage57-cache-",dir=Path(__file__).parent) as folder:
            p=Path(folder)/"trace.npz"; c.save_cache(p,dict(a=np.arange(3)),dict(fixture=True))
            np.testing.assert_array_equal(c.read_cache(p,dict(fixture=True))["a"],np.arange(3))
            with self.assertRaisesRegex(ValueError,"changed"): c.read_cache(p,dict(fixture=False))
            p.write_bytes(p.read_bytes()+b"changed")
            with self.assertRaisesRegex(ValueError,"changed"): c.read_cache(p,dict(fixture=True))

    def test_quality_masks_count_matching_ties_and_invalid_labels(self):
        ends=[dict(qualified=True,reference_iou=.9,student_iou=.5),
              dict(qualified=True,reference_iou=.6,student_iou=.8),
              dict(qualified=False,reference_iou=1.,student_iou=0.)]
        quality,qa=c.selection_masks(ends,[.9,.1,0.],"quality_select")
        random,ra=c.selection_masks(ends,[.9,.1,0.],"count_matched")
        np.testing.assert_array_equal(quality,[True,False,False])
        np.testing.assert_array_equal(random,[False,True,False])
        self.assertEqual(qa["actual_kept"],ra["actual_kept"])
        self.assertEqual(ra["mask_disagreements"],2)
        ends[0]["reference_iou"]=.5
        selected,audit=c.selection_masks(ends,[.9,.1,0.],"quality_select")
        self.assertEqual(int(selected.sum()),0); self.assertEqual(audit["quality_tied"],1)
        ends[0]["reference_iou"]=float("nan")
        with self.assertRaisesRegex(ValueError,"Invalid"): c.selection_masks(ends,[.9,.1,0.],"quality_select")
        with self.assertRaises(ValueError): c.selection_masks(ends,[0.],"count_matched")


@unittest.skipUnless(torch is not None,"CPU Torch/OpenCV required; skipped is not a passed rollout audit")
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_actor_quality_is_fit_label_only_and_real_objective(self):
        from test_stage50_crossed import TorchTests as NativeFixture
        from test_stage52_branch import actual_small_head
        from stage48_stability_runtime import make_head,digest
        from stage57_quality_runtime import configure,rollout_batch,objective,temporary_head
        with tempfile.TemporaryDirectory(prefix="stage57-actor-",dir=Path(__file__).parent) as name, \
                patch("test_stage50_crossed.tiny_head",side_effect=actual_small_head):
            inputs,_,native,capture,_=NativeFixture().fixture(Path(name)); capture.handle.remove(); capture.pre_handle.remove()
            entry,arrays=inputs["focus"]["fixture"]; plan=dict(inputs["plan"],design=dict(c.DESIGN))
            anchor=dict(sequence=entry["sequence"],frame=1,parents=[entry["initial_box"]]*2,native_expected=arrays["native"]["boxes"][:2].tolist())
            head=make_head(native.rt); configure(head,"quality_select"); original=digest(native.rt.net)
            with torch.no_grad(): head.conv5_size.bias.add_(.02); head.conv5_offset.bias.add_(.02)
            rows,ends,audit,_=rollout_batch(native,head,plan,{entry["sequence"]:entry},[anchor])
            np.testing.assert_array_equal(rows[1]["parent"],rows[0]["boxes"])
            self.assertEqual(audit["student_calls"],4); self.assertEqual(audit["reference_calls"],4)
            for end in ends:
                self.assertTrue(0<=end["reference_iou"]<=1 and 0<=end["student_iou"]<=1)
            # Explicit mixed-quality labels isolate whether selection enters the
            # REAL head objective, independent of this tiny image's IoU ties.
            forced=copy.deepcopy(ends)
            forced[0].update(qualified=True,reference_iou=.9,student_iou=.5)
            forced[1].update(qualified=True,reference_iou=.6,student_iou=.8)
            values={arm:objective(torch,head,rows,forced,plan,arm,[.9,.1]) for arm in c.ARMS}
            full=values["endpoint_control"]; quality=values["quality_select"]; matched=values["count_matched"]
            torch.testing.assert_close(full[0]-quality[0],full[2]-quality[3])
            torch.testing.assert_close(full[0]-matched[0],full[2]-matched[3])
            self.assertEqual(quality[6]["kept_not_better"],0)
            self.assertEqual(matched[6]["kept_not_better"],1)
            self.assertGreater(float(quality[3].detach()),0)
            quality[3].backward()
            for suffix in ("size","offset"):
                self.assertGreater(sum(float(v.grad.abs().sum()) for n,v in head.named_parameters() if suffix in n and v.grad is not None),0)
            self.assertTrue(all(v.grad is None for n,v in head.named_parameters() if "_ctr" in n))
            changed=copy.deepcopy(entry); changed["ground_truth"][1:]=[[10000,10000,20,20] for _ in changed["ground_truth"][1:]]
            again,new_ends,_,_=rollout_batch(native,head,plan,{entry["sequence"]:changed},[anchor])
            for old,new in zip(rows,again): np.testing.assert_array_equal(old["boxes"],new["boxes"])
            self.assertFalse(any(e["qualified"] for e in new_ends))
            self.assertEqual(digest(native.rt.net),original)
            with self.assertRaisesRegex(RuntimeError,"rollback"):
                with temporary_head(native,head.state_dict()): raise RuntimeError("rollback")
            self.assertEqual(digest(native.rt.net),original)

    def test_selected_node_gradient_and_empty_mask(self):
        from stage57_quality_runtime import node_penalty
        plan={"design":dict(c.DESIGN)}
        ends=[dict(index=i,target=[10.,10.,20.,20.],qualified=True) for i in range(2)]
        boxes=torch.tensor([[12.,11.,24.,22.],[13.,12.,23.,21.]],dtype=torch.float64,requires_grad=True)
        loss=node_penalty(torch,boxes,ends,plan,[True,False])
        gradient=torch.autograd.grad(loss,boxes,retain_graph=True)[0]
        self.assertGreater(float(gradient[0].abs().sum()),0)
        self.assertEqual(float(gradient[1].abs().sum()),0)
        empty=node_penalty(torch,boxes,ends,plan,[False,False])
        self.assertEqual(float(empty.detach()),0)
        self.assertEqual(float(torch.autograd.grad(empty,boxes)[0].abs().sum()),0)

    def test_complete_source56_fit_isolation_resume_budgets_and_parity(self):
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
        import stage57_quality_runtime as runtime
        import stage57_quality_summary as summary
        from test_stage52_branch import TorchTests as SourceFixture,actual_small_head
        from stage48_stability_runtime import digest
        with tempfile.TemporaryDirectory(prefix="stage57-full-",dir=Path(__file__).parent) as name:
            folder=Path(name)
            with patch.dict(old.DESIGN,updates=2,smoke_updates=2,save_every=1,batch_pairs=1,smoke_frames=2,
                    bootstrap_samples=50,minimum_changed_pairs=1,minimum_changed_videos=1),patch.dict(old.prior.DESIGN,samples_per_video=4), \
                    patch("test_stage50_crossed.tiny_head",side_effect=actual_small_head),patch.dict(b52.DESIGN,bootstrap_samples=50,focus=["v0","v1"]), \
                    patch.dict(b53.DESIGN,bootstrap_samples=50,focus=["v0","v1"]),patch.dict(b54.DESIGN,bootstrap_samples=50,focus=["v0","v1"]), \
                    patch.dict(b56.DESIGN,bootstrap_samples=50,focus=["v0","v1"]),patch.dict(c.DESIGN,bootstrap_samples=50,focus=["v0","v1"]),contextlib.redirect_stdout(io.StringIO()):
                native,source48,source48root,refs,bindings,root51=SourceFixture().fixture(folder)
                with patch.object(old,"load_source",return_value=(source48,source48root,"report50",bindings)), \
                        patch.object(old.prior,"source_native",side_effect=lambda p,e:refs[e["sequence"]]):
                    previous=root51
                    for number,common,exp,summarizer in ((52,b52,e52,s52),(53,b53,e53,s53),(54,b54,e54,s54),(56,b56,e56,s56)):
                        root=folder/("stage"+str(number)); exp.prepare(previous,root); plan,inputs=common.load_plan(root)
                        exp.train(native,plan,inputs,root,c.SEEDS[0],True)
                        for seed in c.SEEDS: exp.train(native,plan,inputs,root,seed)
                        exp.evaluate(native,plan,root); summarizer.summarize(root); previous=root
                    self.assertEqual(len(c.load_source(previous)["records"]),12)
                    output=folder/"stage57"; experiment.prepare(previous,output); plan,inputs=c.load_plan(output)
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
                    self.assertTrue(resumed["reproduces_stage56_control"])
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
                    self.assertEqual(sum(r["reproduces_stage56_control"] is True for r,_ in records),4)
                    self.assertTrue(all(r["maximum_endpoint_gradient"]>0 for r,_ in records))
                    self.assertEqual(digest(native.rt.net),original)
                    table=output/"analysis/stage57_policy_summary.csv"; saved=table.read_bytes(); table.write_bytes(saved+b"changed")
                    with self.assertRaisesRegex(ValueError,"differs"): summary.summarize(output)
                    table.write_bytes(saved)
                    model=Path(resumed["checkpoint"]); saved=model.read_bytes(); model.write_bytes(saved+b"changed")
                    with self.assertRaisesRegex(ValueError,"changed"): experiment.model_records(plan,output)
                    model.write_bytes(saved)
                    source_table=previous/"analysis/stage56_seed_video_summary.csv"
                    saved=source_table.read_bytes(); source_table.write_bytes(saved+b"changed")
                    with self.assertRaises(ValueError): c.load_source(previous)
                    source_table.write_bytes(saved)

if __name__=="__main__": unittest.main()
