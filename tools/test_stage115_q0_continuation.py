"""Test paired current observations, one action change and real own-history futures."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import stage115_q0_continuation as stage
import test_stage113_frozen_closed_loop as fixture


class OneQ0Continuation(unittest.TestCase):
    def setUp(self):
        base=fixture.FrozenClosedLoop();base.setUp();self.addCleanup(base.doCleanups);self.b=base
        self.b.observer.assert_frozen=lambda:None  # Synthetic observer has no Torch parameters.

    def test_role_selection_uses_all_states_and_records_unsupported_first_event(self):
        n=8;good=np.array([50.,50.,10.,10.]);bad=good+[50,0,0,0];medium=good+[5,0,0,0]
        candidates=np.tile(good,(n,4,1));selected=np.ones(n,int)
        candidates[0,1]=bad;candidates[1,0]=medium;candidates[1,1]=bad;candidates[2,0]=bad
        trace=dict(frame_id=np.arange(1,n+1),candidate_boxes=candidates,selected=selected,boxes=candidates[np.arange(n),selected])
        entry=dict(ground_truth=np.tile(good,(n+1,1)))
        rows,anchors=stage.select_anchors(trace,entry,dict(stage.DESIGN,horizon=4))
        self.assertEqual([r["first_frame"] for r in rows],[1,2,3]);self.assertEqual([a["frame"] for a in anchors],[1,2,3])
        self.assertEqual([a["roles"] for a in anchors],[[role] for role in stage.ROLES])
        short={k:v[-2:].copy() for k,v in trace.items()};short["candidate_boxes"][1,1]=bad;short["boxes"][1]=bad
        rows,anchors=stage.select_anchors(short,dict(ground_truth=np.tile(good,(3,1))),dict(stage.DESIGN,horizon=4))
        self.assertEqual(rows[0]["first_frame"],2);self.assertFalse(rows[0]["full_horizon_supported"]);self.assertEqual(anchors,[])

    def test_single_q0_action_changes_actual_futures_without_gt_leakage(self):
        b=self.b;s=stage.s;trace=b.collect("generated_shared");anchor=dict(frame=2,last=7)
        generator=b.package["generator"];before=b.observer.calls
        shared=stage.point(b.observer,trace,b.entry,2,b.old,generator,b.model,image_reader=b.reader)
        calls=int(shared["unique_search_calls"].sum());self.assertEqual(calls,b.observer.calls-before)
        outputs={}
        for treatment in stage.TREATMENTS:
            before=b.observer.calls
            data=stage.collect(b.observer,trace,shared,b.entry,anchor,treatment,b.old,generator,b.model,image_reader=b.reader)
            stage.audit(data,trace,shared,b.entry,anchor,treatment,b.old,generator,b.model)
            queries=int((data["head_calls"]*data["new_executed"]).sum());self.assertEqual(queries,b.observer.calls-before);calls+=queries
            self.assertFalse(data["new_executed"][0].any());np.testing.assert_array_equal(data["selected"][1:],data["policy_selected"][1:])
            np.testing.assert_array_equal(data["intervention"],np.array([treatment=="q0_once"]+[False]*5))
            altered=dict(b.entry,ground_truth=np.full_like(b.entry["ground_truth"],999.))
            other=stage.collect(b.observer,trace,shared,altered,anchor,treatment,b.old,generator,b.model,image_reader=b.reader)
            for k in stage.FIELDS:
                if k not in ("pipeline_ms","search_ms"):np.testing.assert_array_equal(data[k],other[k])
            outputs[treatment]=data
        stage.bridge(outputs["policy"],trace,anchor,True)
        self.assertEqual(outputs["q0_once"]["selected"][0],0)
        self.assertTrue(np.any(outputs["policy"]["boxes"][0]!=outputs["q0_once"]["boxes"][0]))
        self.assertTrue(np.any(outputs["policy"]["candidate_parents"][1:]!=outputs["q0_once"]["candidate_parents"][1:]))
        for k in ("candidate_boxes","candidate_parents","features","scores","raw","size","offset"):
            np.testing.assert_array_equal(outputs["policy"][k][0],outputs["q0_once"][k][0])
        broken={k:v.copy() for k,v in outputs["q0_once"].items()};broken["new_executed"][0]=True
        with self.assertRaises(AssertionError):stage.audit(broken,trace,shared,b.entry,anchor,"q0_once",b.old,generator,b.model)

    def test_metrics_exclude_current_frame_and_weight_seeds_inside_videos(self):
        data=dict(frame_id=np.arange(1,5),boxes=np.tile([50.,50.,10.,10.],(4,1)),candidate_boxes=np.tile([50.,50.,10.,10.],(4,4,1)))
        entry=dict(ground_truth=np.tile([50.,50.,10.,10.],(5,1)))
        first=stage.metrics(data,entry);entry["ground_truth"][1,:2]+=100
        second=stage.metrics(data,entry);self.assertNotEqual(first["current_iou"],second["current_iou"]);self.assertEqual(first["future_mean_iou"],second["future_mean_iou"])
        anchors=[];branches=[]
        for seq,seed,count in (("a",2312,3),("a",2313,1),("b",2312,1)):
            for f in range(count):
                ident=dict(anchor_id=f"{seq}_{seed}_{f}",sequence=seq,seed=seed,frame=f+1,cohort="historical" if seq=="a" else "expanded",roles=[stage.ROLES[0]])
                anchors.append(ident)
                for treatment in stage.TREATMENTS:
                    value=float(seq=="a" and seed==2312 and treatment=="q0_once")
                    branches.append(dict(ident,treatment=treatment,**{m:value for m in stage.METRICS}))
        _,_,pairs=stage.aggregate(anchors,branches);r=next(r for r in pairs if r["cohort"]=="all" and r["role"]=="all_unique_anchors")
        self.assertAlmostEqual(r["future_mean_iou_delta"],.25)

    def test_first_card_change_and_resume_identity_remain_separate(self):
        a=dict(tracker=dict(device="NVIDIA L20",dtype="float32"),gpu_uuid="a");b=dict(tracker=dict(device="NVIDIA L40",dtype="float32"),gpu_uuid="b")
        self.assertEqual(len(stage.runtime_binding(a,b,b)),2)
        with self.assertRaisesRegex(ValueError,"STAGE115_ROOT"):stage.runtime_binding(a,b,a)
        changed=copy.deepcopy(b);changed["tracker"]["dtype"]="float16"
        with self.assertRaises(ValueError):stage.runtime_binding(a,changed)

    def test_full_pipeline_new_call_ledger_shared_receipts_resume_and_tamper(self):
        b=self.b;s=stage.s;trace=b.collect("generated_shared")
        jobs=[];models={};anchors=[]
        for n,cohort in ((9,"historical"),(10,"expanded")):
            for seed in (2312,2313):
                job=dict(spec=dict(cohort=cohort,unit=dict(sequence=f"uav{n:07d}_00000_s",seed=2171,origin="native",policy="ppo",fold=0)),outer_fold=0,arm="generated_shared",seed=seed)
                jobs.append(job);anchors.append(dict(anchor_id=stage.key(job)+"__f000002",context=stage.key(job),cohort=cohort,sequence=job["spec"]["unit"]["sequence"],seed=seed,frame=2,last=5,roles=[stage.ROLES[0]]))
                model=copy.deepcopy(b.model);model["seed"]=seed;models[0,"generated_4",seed]=s.io.seal({k:v for k,v in model.items() if k!="fingerprint"})
        models[0,"generator",0]=b.package
        signature=s.io.seal(dict(models={str(k):v for k,v in models.items()}))["fingerprint"]
        with tempfile.TemporaryDirectory() as folder,patch.object(stage,"DESIGN",dict(stage.DESIGN,horizon=4)),contextlib.redirect_stdout(io.StringIO()):
            base=Path(folder);r113=base/"source";root=base/"output";r113.mkdir();root.mkdir()
            p113=s.io.seal(dict(fingerprint="unused",jobs=jobs))
            for job in jobs:s.c.save(r113/"evaluation"/(stage.key(job)+".npz"),trace,stage.source_meta(p113,job,models))
            snapshot={str(p):p.read_bytes() for p in r113.rglob("*") if p.is_file()}
            plan=s.io.seal(dict(schema=stage.SCHEMA,design=stage.DESIGN,anchors=anchors,sampling_scope=[],runtime={},runtime_differences=[],native_head_sha256="native",
                source_stage114_fingerprint=stage.SOURCE114,source_sha256={p:s.c.file_hash(Path(p)) for p in snapshot},code_sha256={}))
            s.c.publish_identical(root/"manifest.json",plan)
            metric=stage.metrics
            def checked_metric(*args):
                self.assertEqual(len(list(root.rglob("*.receipt.json")))%3,0)
                return metric(*args)
            with (patch.object(s.e63.source_experiment,"installed",side_effect=lambda *_:contextlib.nullcontext("native")),
                  patch.object(stage,"metrics",side_effect=checked_metric),patch.object(s.common,"train",side_effect=AssertionError("New scorer training")),
                  patch.object(s.common,"fit_generator",side_effect=AssertionError("New generator training"))):
                before=b.observer.calls;args=(root,r113,p113,plan,models,lambda _:b.entry,b.old,b.observer)
                stage.run(*args,image_reader=b.reader);report=s.c.c66.sealed(root/"analysis/stage115_decision.json",stage.SCHEMA)
                self.assertEqual(report["new_search_calls"],b.observer.calls-before);self.assertEqual(report["new_block_calls"],7*report["new_search_calls"])
                out={str(p):p.read_bytes() for p in root.rglob("*") if p.is_file()}
                with patch.object(b.observer,"capture",side_effect=AssertionError("Repeated inference")):stage.run(*args,image_reader=b.reader)
                self.assertEqual(out,{str(p):p.read_bytes() for p in root.rglob("*") if p.is_file()})
            self.assertEqual(snapshot,{str(p):p.read_bytes() for p in r113.rglob("*") if p.is_file()})
            self.assertEqual(signature,s.io.seal(dict(models={str(k):v for k,v in models.items()}))["fingerprint"])
            stage.check_results(root);next((root/"branches").glob("*.npz")).write_bytes(b"changed")
            with self.assertRaises(ValueError):stage.check_results(root)


if __name__=="__main__":unittest.main()
