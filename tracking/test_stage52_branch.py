"""Actual small-channel CENTER scope/gradient tests and full CPU source-to-OOF fixture."""
import ast
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import stage52_branch_common as c
import stage52_branch_experiment as experiment

try:
    import torch
    import cv2
except ImportError:
    torch=cv2=None


class PlanTests(unittest.TestCase):
    def test_own_schema_receipt_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory(prefix="stage52-cache-",dir=Path(__file__).parent) as folder:
            p=Path(folder)/"trace.npz"; meta=dict(source="fixture")
            c.save_cache(p,dict(a=np.arange(4)),meta)
            np.testing.assert_array_equal(c.read_cache(p,meta)["a"],np.arange(4))
            with self.assertRaisesRegex(ValueError,"changed"): c.read_cache(p,dict(source="wrong"))
            p.write_bytes(p.read_bytes()+b"changed")
            with self.assertRaisesRegex(ValueError,"changed"): c.read_cache(p,meta)


def actual_small_head():
    # Execute the unmodified repository conv/CENTER definitions without importing
    # the GPU tracker package through lib.models.__init__.
    path=Path(__file__).resolve().parents[1]/"lib/models/layers/head.py"
    tree=ast.parse(path.read_text(encoding="utf-8"))
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ("conv","CenterPredictor")]
    namespace=dict(torch=torch,nn=torch.nn,FrozenBatchNorm2d=None)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),namespace)
    class Head(namespace["CenterPredictor"]):
        # Properties let the existing source fixture vary donor output biases;
        # only the real conv1..5_* modules appear in state_dict/named_children.
        @property
        def ctr(self): return self.conv5_ctr
        @property
        def sz(self): return self.conv5_size
        @property
        def off(self): return self.conv5_offset
    torch.manual_seed(2151)
    head=Head(inplanes=3,channel=8,feat_sz=4,stride=64).eval()
    with torch.no_grad():
        for name,module in head.named_modules():
            if isinstance(module,torch.nn.Conv2d):
                module.weight.zero_(); module.bias.zero_()
                if name.split(".")[0].endswith("_ctr"):
                    module.weight[:,0,module.weight.shape[2]//2,module.weight.shape[3]//2]=.5
        head.conv5_size.bias.fill_(float(np.log(.27/.73)))
    return head


@unittest.skipUnless(torch is not None,"CPU Torch/OpenCV needed; skipped is not a passed branch audit")
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_actual_center_freezing_and_hard_argmax_gradient_routes(self):
        from stage18_tracking_loss import tracking_loss
        from stage48_stability_runtime import head_output
        from stage52_branch_runtime import branches,configure,frozen_digest
        from test_stage50_crossed import feature
        head=actual_small_head().requires_grad_(True); groups=branches(head)
        target=torch.tensor([[.38,.38,.24,.24]])
        output=head_output(head,feature())
        # Remove focal's graph: actual box loss has no response-tower gradient.
        geometry_loss,_=tracking_loss(torch,dict(output,score_map=output["score_map"].detach()),target)
        geometry_loss.backward()
        self.assertTrue(all(p.grad is None or float(p.grad.abs().sum())==0 for n,p in head.named_parameters() if n in groups["response"]))
        self.assertGreater(sum(float(p.grad.abs().sum()) for n,p in head.named_parameters() if n in groups["geometry"] and p.grad is not None),0)
        for arm in c.ARMS:
            head=actual_small_head(); _,chosen=configure(head,arm); before=frozen_digest(head,chosen)
            reference=head_output(head,feature()); opt=torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],lr=1e-3)
            out=head_output(head,feature()); loss,_=tracking_loss(torch,out,target); loss.backward(); opt.step()
            self.assertEqual(before,frozen_digest(head,chosen))
            after=head_output(head,feature()); keys=("size_map","offset_map") if arm=="response_only" else ("score_map",)
            for key in keys: torch.testing.assert_close(reference[key],after[key],atol=0,rtol=0)
            self.assertTrue(all(p.grad is None for n,p in head.named_parameters() if n not in chosen))
        head=actual_small_head(); head.shared=torch.nn.Linear(3,3)
        with self.assertRaisesRegex(ValueError,"original"): branches(head)

    def fixture(self,folder):
        import stage51_state_common as old
        import stage51_state_experiment as old_experiment
        import stage51_state_summary as old_summary
        from test_stage51_state_source import TorchTests as SourceFixture
        native,source48,source48root,refs,bindings=SourceFixture().fixture(folder)
        root=folder/"stage51"
        with patch.object(old,"load_source",return_value=(source48,source48root,"report50",bindings)),\
                patch.object(old.prior,"source_native",side_effect=lambda p,e:refs[e["sequence"]]):
            old_experiment.prepare(folder/"stage50",root); source,_=old.load_plan(root)
            old_experiment.collect(native,source,source48,root)
            old_experiment.train(native,source,source48,root,old.SEEDS[0],True)
            for seed in old.SEEDS: old_experiment.train(native,source,source48,root,seed)
            old_experiment.evaluate(native,source,root); old_summary.summarize(root)
        return native,source48,source48root,refs,bindings,root

    def test_full_source_audit_resume_scope_oof_and_tamper(self):
        import stage51_state_common as old
        from stage48_stability_runtime import digest
        import stage52_branch_runtime as runtime
        import stage52_branch_summary as summary
        with tempfile.TemporaryDirectory(prefix="stage52-full-",dir=Path(__file__).parent) as name:
            folder=Path(name)
            with patch.dict(old.DESIGN,updates=3,smoke_updates=2,save_every=1,batch_pairs=1,smoke_frames=2,
                    bootstrap_samples=50,minimum_changed_pairs=1,minimum_changed_videos=1),\
                    patch.dict(old.prior.DESIGN,samples_per_video=4),patch("test_stage50_crossed.tiny_head",side_effect=actual_small_head),\
                    patch.dict(c.DESIGN,bootstrap_samples=50,focus=["v0","v1"]),contextlib.redirect_stdout(io.StringIO()):
                native,source48,source48root,refs,bindings,source_root=self.fixture(folder)
                with patch.object(old,"load_source",return_value=(source48,source48root,"report50",bindings)),\
                        patch.object(old.prior,"source_native",side_effect=lambda p,e:refs[e["sequence"]]):
                    # This is the actual Stage-52 loader, not a mocked completion gate.
                    source=c.load_source(source_root); self.assertEqual(source["report"]["new_models"],8)
                    frozen=digest(native.rt.net); output=folder/"stage52"
                    experiment.prepare(source_root,output); plan,inputs=c.load_plan(output)
                    held=set(plan["folds"][0]["held_out"]); seen=set(); reader=old.read_cache; oldreader=old.prior.read_cache
                    def guard(fn):
                        def wrapped(path,meta):
                            self.assertNotIn(meta.get("sequence"),held); seen.add(meta.get("sequence")); return fn(path,meta)
                        return wrapped
                    with patch.object(old,"read_cache",side_effect=guard(reader)),patch.object(old.prior,"read_cache",side_effect=guard(oldreader)):
                        pool,pool_bindings=c.fit_pool(plan,inputs,0)
                    self.assertEqual(seen,set(plan["folds"][0]["fit"]))
                    smoke=experiment.train(native,plan,inputs,output,c.SEEDS[0],True)
                    self.assertEqual(smoke["status"],"PASS_BRANCH_SCOPE_SMOKE")
                    real_save=runtime.save_step
                    def interrupt(torch,path,state):
                        real_save(torch,path,state); raise RuntimeError("simulated branch interruption")
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
                    with patch("stage48_stability_runtime.run_trace",side_effect=AssertionError("Do not rerun committed trajectories")):
                        experiment.evaluate(native,plan,output)
                    result=summary.summarize(output)
                    self.assertEqual(result["new_models"],8); self.assertEqual(result["inference_head_calls"],48*4)
                    self.assertFalse(result["new_feature_collection"]); self.assertIsNone(result["adoption_gate"])
                    self.assertEqual(result,summary.summarize(output)); self.assertEqual(digest(native.rt.net),frozen)
                    table=output/"analysis/stage52_scope_summary.csv"; before=table.read_bytes(); table.write_bytes(before+b"changed")
                    with self.assertRaisesRegex(ValueError,"differs"): summary.summarize(output)
                    table.write_bytes(before)
                    p=Path(resumed["checkpoint"]); p.write_bytes(p.read_bytes()+b"changed")
                    with self.assertRaisesRegex(ValueError,"changed"): experiment.model_records(plan,output)
                    source_table=source_root/"analysis/stage51_video_summary.csv"; source_table.write_bytes(source_table.read_bytes()+b"changed")
                    with self.assertRaisesRegex(ValueError,"changed"): c.load_source(source_root)


if __name__=="__main__": unittest.main()
