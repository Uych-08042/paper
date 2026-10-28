"""Matched gradients, native selection, exact resume, OOF and frozen-source checks."""

import contextlib
import copy
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from stage4_common import file_hash
import stage44_matched_common as common
import stage44_matched_experiment as experiment
import stage44_matched_summary as summary

HERE = Path(__file__).parent
TORCH = importlib.util.find_spec("torch") is not None


class AnalysisTests(unittest.TestCase):
    def test_factorial_video_pairing_and_invalid_control_are_separate_from_gap(self):
        plan = dict(fit=[dict(sequence=str(i)) for i in range(8)])
        rows = []
        auc = dict(six_initial=60.,retained_initial=62.,six_adapted=63.,retained_adapted=64.)
        for e in plan["fit"]:
            for seed in common.SEEDS:
                for method in common.METHODS:
                    rows.append(dict(sequence=e["sequence"],seed=seed,method=method,auc=auc[method],
                        low_iou_fraction=.02,failure_events=0,longest_failure_fraction=0.))
        with mock.patch.dict(common.DESIGN,bootstrap_samples=64):
            tables,good = summary.analyze(rows,plan)
            self.assertTrue(good["compression_gap_supported"])
            self.assertEqual(good["primary"]["videos"],8)
            self.assertEqual(good["primary"]["mean_delta"],1.)
            self.assertEqual(good["interaction"]["mean_delta"],-1.)
            damaged = copy.deepcopy(rows)
            for r in damaged:
                if r["method"] == "retained_initial":
                    r["auc"] = 70.
            rejected = summary.analyze(damaged,plan)[1]
            self.assertTrue(rejected["raw_gap_gate_pass"])
            self.assertFalse(rejected["compression_gap_supported"])
            self.assertEqual(rejected["status"],"MATCHED_RETAINED_CONTROL_DEGRADED")
            no_gap = copy.deepcopy(rows)
            for r in no_gap:
                if r["method"] == "retained_adapted":
                    r["auc"] = 63.
            self.assertEqual(summary.analyze(no_gap,plan)[1]["status"],"NO_MATCHED_COMPRESSION_GAP_ESTABLISHED")
            with self.assertRaisesRegex(ValueError,"complete unmixed"):
                summary.analyze(rows+rows[:1],plan)


@unittest.skipUnless(TORCH,"CPU PyTorch needed for real matched training")
class TorchTests(unittest.TestCase):
    def test_exact_native_forward_shared_parameters_and_candidate_gradient_flow(self):
        import torch
        from test_stage36_same_crop import canonical_runtime
        from stage35_preserve_runtime import make_student
        from stage44_matched_runtime import make_retained,forward_retained,shared_prefix_identity,parameters,frozen_digest
        torch.set_num_threads(1)
        torch.manual_seed(2123)
        rt = canonical_runtime(torch)
        image = np.zeros((64,64,3),np.uint8)
        rt.initialize(image,[20.,20.,12.,12.])
        rt.tracker.state = [20.,20.,12.,12.]
        search,_ = rt.prepare(image)
        net = make_retained(rt)
        self.assertEqual(len(net.backbone.blocks),12)
        shared = shared_prefix_identity(rt,net,make_student(rt))
        self.assertGreater(shared["retained_stored_parameters"],shared["six_stored_parameters"])
        before = frozen_digest(net)
        output,layer = forward_retained(rt,net,search)
        with torch.no_grad():
            tokens = rt.prefix(search)
            original_layer = rt.selected_layer(rt.selector(tokens))
            original = rt.candidate(tokens,original_layer)
        self.assertEqual(layer,original_layer)
        for k in ("score_map","size_map","offset_map","pred_boxes"):
            torch.testing.assert_close(output[k],original[k],atol=1e-6,rtol=0)
        loss = output["score_map"].sum()+output["pred_boxes"].sum()
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in parameters(net).values()))
        self.assertTrue(all(p.grad is None for n,p in net.named_parameters() if not n.startswith(("backbone.blocks.4.","backbone.blocks.5."))))
        self.assertEqual(frozen_digest(net),before)

    def test_real_training_resume_heldout_isolation_own_rollout_and_tamper_rejection(self):
        import torch
        import stage40_crop_common as c40
        import stage40_crop_experiment as e40
        import stage40_crop_summary as s40
        from test_stage40_crop_augmentation import fixture
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=HERE) as temporary,contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(torch.cuda,"is_available",return_value=False), \
                mock.patch.dict(c40.DESIGN,updates=3,smoke_updates=2,batch_size=1,save_every=1,bootstrap_samples=64), \
                mock.patch.dict(c40.prior.DESIGN,horizon=3,smoke_horizon=2), \
                mock.patch.dict(common.DESIGN,smoke_frames=4,bootstrap_samples=64):
            root = Path(temporary)
            old,rt,reader,adapter,old_bindings = fixture(root,torch)
            r40,r43,out = root / "stage40",root / "stage43",root / "stage44"
            with mock.patch.object(c40,"source_info",side_effect=adapter):
                p40 = c40.prepare(old,r40)
                e40.smoke(rt,p40,r40,reader)
                for seed in c40.SEEDS:
                    e40.train(rt,p40,r40,seed,image_reader=reader)
                e40.evaluate(rt,p40,r40,image_reader=reader)
                s40.summarize(r40)
                frozen = {str(p):file_hash(p) for p in r40.rglob("*") if p.is_file()}
                def completed_source(unused):
                    # Only archived Stage-43 completion is a synthetic boundary.
                    # All Stage-40 controls, new training, models and trajectories are real.
                    common.verify_bindings(frozen)
                    return dict(p40,source_stage40=str(r40)),dict(fingerprint="synthetic-complete-negative-stage43"),dict(frozen)
                with mock.patch.object(common,"source_info",side_effect=completed_source):
                    plan = common.prepare(r43,out)
                    self.assertEqual(plan,common.load_plan(out))
                    with self.assertRaises(FileNotFoundError):
                        experiment.train(rt,plan,out,2110,image_reader=reader)
                    smoke = experiment.smoke(rt,plan,out,reader)
                    self.assertEqual(smoke["initial_native_box_error"],0.)
                    experiment.smoke_gate(out,plan)
                    seen = set()
                    def training_reader(data_root,frame):
                        seen.add(Path(frame["file"]).parent.name)
                        return reader(data_root,frame)
                    direct = experiment.train_arm(rt,plan,root / "direct",0,2110,image_reader=training_reader)
                    self.assertTrue(seen <= set(plan["folds"][0]["fit"]))
                    self.assertFalse(seen & set(plan["folds"][0]["held_out"]))
                    saved = experiment.save_step
                    def interrupt(torch_api,directory,value):
                        result = saved(torch_api,directory,value)
                        if value["step"] == 2:
                            raise RuntimeError("test interruption")
                        return result
                    with mock.patch.object(experiment,"save_step",side_effect=interrupt),self.assertRaisesRegex(RuntimeError,"test interruption"):
                        experiment.train_arm(rt,plan,root / "resumed",0,2110,image_reader=reader)
                    resumed = experiment.train_arm(rt,plan,root / "resumed",0,2110,image_reader=reader)
                    a = experiment.load_checkpoint(torch,direct["checkpoint"])
                    b = experiment.load_checkpoint(torch,resumed["checkpoint"])
                    for k in a["parameters"]:
                        self.assertTrue(torch.equal(a["parameters"][k],b["parameters"][k]))
                    self.assertEqual(a["history"],b["history"])
                    self.assertEqual(direct["head_calls"],3*1*2)
                    self.assertEqual(direct["block_calls"],7*direct["head_calls"])
                    experiment.train(rt,plan,out,2110,image_reader=reader)
                    with self.assertRaises(FileNotFoundError):
                        experiment.evaluate(rt,plan,out,image_reader=reader)
                    experiment.train(rt,plan,out,2111,image_reader=reader)
                    entry = plan["fit"][0]
                    experiment.evaluate(rt,plan,out,entry["sequence"],reader)
                    with self.assertRaisesRegex(ValueError,"16 complete"):
                        summary.summarize(out)
                    models,_ = experiment.models_for_evaluation(rt,plan,out)
                    net = models[experiment.fold_for(plan,entry),2110]
                    old_read = experiment.read_boxes
                    with mock.patch.object(rt,"prepare",wraps=rt.prepare) as spy:
                        def after_frames(path):
                            self.assertEqual(spy.call_count,len(entry["frames"])-1)
                            return old_read(path)
                        with mock.patch.object(experiment,"read_boxes",side_effect=after_frames):
                            before = experiment.collect_trace(rt,plan,entry,net,image_reader=reader)
                    future = Path(plan["root"]) / "annotations/other_future.txt"
                    future.write_text("20,20,12,12\n"+"\n".join(["50,50,5,5"]*12),encoding="utf-8")
                    other = dict(entry,annotation="annotations/other_future.txt",annotation_sha256=file_hash(future))
                    after = experiment.collect_trace(rt,plan,other,net,image_reader=reader)
                    for k in before:
                        if k not in ("gt","iou"):
                            np.testing.assert_array_equal(before[k],after[k])
                    self.assertFalse(np.array_equal(before["gt"],after["gt"]))
                    experiment.evaluate(rt,plan,out,image_reader=reader)
                    report = summary.summarize(out)
                    self.assertEqual(report["new_models"],8)
                    self.assertEqual(report["new_full_trajectories"],16)
                    self.assertEqual(report["retained_inference_block_calls"],7*report["retained_inference_head_calls"])
                    self.assertFalse(report["novel_method_established"])
                    self.assertEqual(report,summary.summarize(out))
                    common.verify_bindings(frozen)
                    common.verify_bindings(old_bindings)
                    with mock.patch.object(rt,"prepare",wraps=rt.prepare) as spy:
                        experiment.evaluate(rt,plan,out,image_reader=reader)
                        self.assertEqual(spy.call_count,0)
                    path = common.trace_path(out,entry,2110)
                    path.with_suffix(".json").unlink()
                    experiment.evaluate(rt,plan,out,entry["sequence"],reader)
                    data = path.read_bytes()
                    path.write_bytes(data+b"tamper")
                    with self.assertRaisesRegex(ValueError,"Stage-44 trace changed"):
                        summary.summarize(out)
                    path.write_bytes(data)
                    new_checkpoint = next((out / "training").rglob("step000003.pt"))
                    saved_bytes = new_checkpoint.read_bytes()
                    new_checkpoint.write_bytes(saved_bytes+b"tamper checkpoint")
                    with self.assertRaisesRegex(ValueError,"Frozen Stage-35 artifact changed"):
                        summary.summarize(out)
                    new_checkpoint.write_bytes(saved_bytes)
                    old_checkpoint = next((r40 / "training").rglob("step000003.pt"))
                    old_checkpoint.write_bytes(old_checkpoint.read_bytes()+b"tamper source")
                    with self.assertRaisesRegex(ValueError,"Frozen Stage-35 artifact changed"):
                        common.load_plan(out)


if __name__ == "__main__":
    unittest.main()
