"""Source isolation, real CPU head training, resume, and closed-loop OOF plumbing."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import stage51_state_common as c
import stage51_state_experiment as experiment

try:
    import torch
    import cv2
except ImportError:
    torch=cv2=None


class PlanTests(unittest.TestCase):
    def test_completed_stage50_and_mutation_rejection(self):
        with tempfile.TemporaryDirectory(prefix="stage51-source-",dir=Path(__file__).parent) as folder:
            root=Path(folder); marker=root/"cache.npz"; marker.write_bytes(b"frozen")
            old=dict(fingerprint="p50",expected_inputs=95,expected_head_forwards=855,source_input=str(root/"stage48"),source_file_sha256={})
            run=c.replay.seal(dict(schema=c.replay.SCHEMA,plan_fingerprint="p50"))
            c.publish_identical(root/"run.json",run); c.publish_identical(root/"manifest.json",dict(fixture=True))
            hashes={}
            for name in ("responses","cell_positions","crossed_matrix","focus"):
                p=root/"analysis"/("stage50_"+name+".csv"); p.parent.mkdir(exist_ok=True); p.write_bytes(b"header\n")
                hashes[p.name]=c.file_hash(p)
            report=c.replay.seal(dict(schema=c.replay.SCHEMA,status="COMPLETE_SHARED_PARENT_HEAD_DIAGNOSTIC",
                plan_fingerprint="p50",run_fingerprint=run["fingerprint"],shared_inputs=95,total_head_forwards=855,
                table_sha256=hashes,replay_cache_sha256={str(marker):c.file_hash(marker)}))
            c.publish_identical(root/"analysis/stage50_diagnostic.json",report)
            with patch.object(c.replay,"load_plan",return_value=(old,dict(plan=dict(fingerprint="p48")))):
                self.assertEqual(c.load_source(root)[0]["fingerprint"],"p48")
                marker.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError,"changed"): c.load_source(root)

    def test_donor_uses_fit_fold_not_complementary_oof_fold(self):
        plan=dict(fingerprint="p",folds=[dict(fold=0,fit=["a"],held_out=["b"]),dict(fold=1,fit=["b"],held_out=["a"])],
            donors=[dict(checkpoint_sha256="d0",fit=["a"],held_out=["b"]),dict(checkpoint_sha256="d1",fit=["b"],held_out=["a"])])
        self.assertEqual(c.fit_fold(plan,"a"),0)
        self.assertEqual(c.meta(plan,dict(sequence="a"),"fit")["donor_sha256"],"d0")
        plan["donors"][0]["held_out"]=["a"]
        with self.assertRaisesRegex(ValueError,"Held-out"): c.meta(plan,dict(sequence="a"),"fit")


@unittest.skipUnless(torch is not None,"CPU Torch/OpenCV required; skipped is not a passed training check")
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def fixture(self,folder):
        from test_stage50_crossed import TorchTests as ReplayFixture
        from stage48_stability_runtime import collect_features,digest,run_trace
        from stage51_state_runtime import installed_head
        inputs,_,native,capture,banks=ReplayFixture().fixture(folder)
        capture.handle.remove(); capture.pre_handle.remove()
        e=inputs["focus"]["fixture"][0]
        entries=[dict(copy.deepcopy(e),sequence="v"+str(i),source_group_proxy="g"+str(i)) for i in range(12)]
        source=dict(copy.deepcopy(inputs["plan"]),entries=entries,folds=c.prior.fold_design(entries),source_file_sha256={})
        source_root=folder/"stage48"; source_root.mkdir(); references={}
        for entry in entries:
            ref=run_trace(native,entry["frames"],entry["initial_box"],source["root"])
            references[entry["sequence"]]=ref
            observer=__import__("stage48_stability_runtime").FeatureCapture(native)
            try: pack=collect_features(observer,source,entry,ref)
            finally: observer.handle.remove(); observer.pre_handle.remove()
            c.prior.save_cache(c.prior.cache_path(source_root,entry),pack,c.prior.feature_meta(source,entry))
        states={}; records=[]
        for fold in range(2):
            head=__import__("stage48_stability_runtime").make_head(native.rt)
            with torch.no_grad(): head.sz.bias.add_(.03+.02*fold)
            states[fold]=copy.deepcopy(head.state_dict())
            metadata=dict(plan_fingerprint=source["fingerprint"],fold=fold,seed=2132,arm="supervised",mode="full",
                updates=c.prior.DESIGN["updates"],fit=source["folds"][fold]["fit"],held_out=source["folds"][fold]["held_out"])
            p=source_root/"training/seed2132"/("fold{}".format(fold))/"supervised/step000256.pt"; p.parent.mkdir(parents=True)
            torch.save(dict(metadata=metadata,step=c.prior.DESIGN["updates"],head=states[fold]),p)
            c.publish_identical(p.with_suffix(".receipt.json"),c.prior.seal(dict(schema=c.prior.SCHEMA,metadata=metadata,
                step=c.prior.DESIGN["updates"],sha256=c.file_hash(p))))
            records.append(dict(fold=fold,seed=2132,arm="supervised",checkpoint=str(p.resolve()),checkpoint_sha256=c.file_hash(p),
                receipt_sha256=c.file_hash(p.with_suffix(".receipt.json")),final_head_sha256=digest(head)))
        c.publish_identical(source_root/"training/seed2132/stage48_training.json",c.prior.seal(dict(schema=c.prior.SCHEMA,
            status="COMPLETE_FROZEN_HEAD_TRAINING",plan_fingerprint=source["fingerprint"],records=records)))
        oldrun=c.prior.seal(dict(schema=c.prior.SCHEMA,plan_fingerprint=source["fingerprint"]))
        c.publish_identical(source_root/"evaluation/run.json",oldrun)
        for entry in entries:
            # Frozen OOF reference uses the opposite orientation from new fit-state generation.
            fold=next(f["fold"] for f in source["folds"] if entry["sequence"] in f["held_out"])
            with installed_head(native,states[fold]): data=run_trace(native,entry["frames"],entry["initial_box"],source["root"])
            for seed in c.SEEDS:
                name="supervised_seed"+str(seed)
                c.prior.save_cache(source_root/"evaluation/traces"/(entry["sequence"]+"__"+name+".npz"),data,
                    dict(run_fingerprint=oldrun["fingerprint"],sequence=entry["sequence"],method=name))
        marker=folder/"frozen.marker"
        bindings={str(marker):c.file_hash(marker)}
        bindings.update({str(p):c.file_hash(p) for p in source_root.rglob("*") if p.is_file()})
        return native,source,source_root,references,bindings

    def test_ordinary_loss_changes_only_with_selected_view_and_keeps_offcrop_negatives(self):
        from test_stage50_crossed import tiny_head,feature
        from stage51_state_runtime import objective
        head=tiny_head().requires_grad_(True)
        features=np.repeat(feature().numpy(),3,axis=0)
        target=np.array([[.4,.4,.2,.2]]*3,np.float32)
        sample=dict(features=features,target=target)
        base=float(objective(torch,head,[sample],"native_replay")[0])
        unused=copy.deepcopy(sample); unused["target"][1]=[10,10,.2,.2]; unused["features"][1]+=5
        self.assertEqual(base,float(objective(torch,head,[unused],"native_replay")[0]))
        self.assertEqual(float(objective(torch,head,[sample],"state_mix")[0]),float(objective(torch,head,[unused],"state_mix")[0]))
        changed=copy.deepcopy(sample); changed["target"][2]=[10,10,.2,.2]
        self.assertEqual(base,float(objective(torch,head,[changed],"native_replay")[0]))
        loss,stats=objective(torch,head,[changed],"state_mix")
        self.assertEqual(int(stats["supervised_count"]),2)
        self.assertEqual(int(stats["center_count"]),1); self.assertEqual(int(stats["regression_count"]),1)
        self.assertNotEqual(base,float(loss)); loss.backward()
        self.assertGreater(float(head.ctr.weight.grad.abs().sum()),0)

    def test_full_workflow_resume_scope_isolation_and_tamper(self):
        from stage48_stability_runtime import digest
        from stage51_state_runtime import train_arm
        import stage51_state_runtime as runtime
        import stage51_state_summary as summary
        with tempfile.TemporaryDirectory(prefix="stage51-test-",dir=Path(__file__).parent) as name:
            folder=Path(name)
            with patch.dict(c.DESIGN,updates=3,smoke_updates=2,save_every=1,batch_pairs=1,smoke_frames=2,
                    bootstrap_samples=50,minimum_changed_pairs=1,minimum_changed_videos=1),\
                    patch.dict(c.prior.DESIGN,samples_per_video=4),contextlib.redirect_stdout(io.StringIO()):
                native,source,source_root,refs,bindings=self.fixture(folder)
                frozen=digest(native.rt.net); output=folder/"stage51"
                with patch.object(c,"load_source",return_value=(source,source_root,"audit50",bindings)),\
                        patch.object(c.prior,"source_native",side_effect=lambda p,e:refs[e["sequence"]]):
                    experiment.prepare(folder/"stage50",output); plan,_=c.load_plan(output)
                    r=experiment.collect(native,plan,source,output)
                    self.assertTrue(r["nontrivial_exposure_pass"])
                    for entry in plan["entries"]:
                        data=c.read_cache(c.path_for(output,entry,"donor_traces"),c.meta(plan,entry,"fit_only_donor_trajectory"))
                        oldrun=c.prior.read_sealed(source_root/"evaluation/run.json")
                        name="supervised_seed2132"
                        old=c.prior.read_cache(source_root/"evaluation/traces"/(entry["sequence"]+"__"+name+".npz"),
                            dict(run_fingerprint=oldrun["fingerprint"],sequence=entry["sequence"],method=name))
                        self.assertFalse(np.array_equal(data["boxes"],old["boxes"]),"Do not reuse the complementary OOF rollout as training states")
                    with patch.object(runtime,"collect_features",side_effect=AssertionError("Completed features must be reused")):
                        self.assertEqual(r,experiment.collect(native,plan,source,output))
                    held=set(plan["folds"][0]["held_out"]); seen=set(); read=c.read_cache; oldread=c.prior.read_cache
                    def guard(reader):
                        def wrapped(path,meta):
                            self.assertNotIn(meta.get("sequence"),held); seen.add(meta.get("sequence")); return reader(path,meta)
                        return wrapped
                    with patch.object(c,"read_cache",side_effect=guard(read)),patch.object(c.prior,"read_cache",side_effect=guard(oldread)):
                        pool,pool_bindings=c.fit_pool(output,plan,source,0)
                    self.assertEqual(seen,set(plan["folds"][0]["fit"]))
                    smoke=experiment.train(native,plan,source,output,c.SEEDS[0],True)
                    self.assertEqual(smoke["status"],"PASS_STATE_SOURCE_SMOKE")
                    real_save=runtime.save_step
                    def interrupted(torch,path,state):
                        real_save(torch,path,state)
                        raise RuntimeError("simulated interruption after committed optimizer state")
                    with patch.object(runtime,"save_step",side_effect=interrupted):
                        with self.assertRaisesRegex(RuntimeError,"simulated"):
                            train_arm(native,plan,output,pool,pool_bindings,0,c.SEEDS[0],c.ARMS[0])
                    resumed=train_arm(native,plan,output,pool,pool_bindings,0,c.SEEDS[0],c.ARMS[0])
                    uninterrupted=train_arm(native,plan,folder/"uninterrupted",pool,pool_bindings,0,c.SEEDS[0],c.ARMS[0])
                    self.assertEqual(resumed["final_head_sha256"],uninterrupted["final_head_sha256"])
                    for seed in c.SEEDS:
                        record=experiment.train(native,plan,source,output,seed)
                        self.assertEqual(record,experiment.train(native,plan,source,output,seed))
                    experiment.evaluate(native,plan,output)
                    with patch("stage48_stability_runtime.run_trace",side_effect=AssertionError("Committed OOF traces must be reused")):
                        experiment.evaluate(native,plan,output)
                    report=summary.summarize(output)
                    self.assertEqual(report["new_models"],8); self.assertEqual(report["new_oof_trajectories"],48)
                    self.assertEqual(report["inference_head_calls"],48*4)
                    self.assertFalse(report["held_out_examples_used_for_fitting"])
                    self.assertEqual(report,summary.summarize(output)); self.assertEqual(digest(native.rt.net),frozen)
                    table=output/"analysis/stage51_policy_summary.csv"; original=table.read_bytes(); table.write_bytes(original+b"changed")
                    with self.assertRaisesRegex(ValueError,"differs"): summary.summarize(output)
                    table.write_bytes(original)
                    path=Path(resumed["checkpoint"]); path.write_bytes(path.read_bytes()+b"changed")
                    with self.assertRaisesRegex(ValueError,"changed"): experiment.model_records(plan,output)


if __name__=="__main__": unittest.main()
