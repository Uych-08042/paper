"""Test frozen own-history rollouts, cold start, GT exclusion, recovery and receipts."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import stage113_frozen_closed_loop as stage
import test_stage110_proposals as fixture
import test_stage112_shared_scoring as fitting


class FrozenClosedLoop(unittest.TestCase):
    def setUp(self):
        parent=fixture.SearchProposals(); parent.setUp(); self.addCleanup(parent.doCleanups)
        self.parent=parent; self.old=parent.old; self.entry=parent.entry; self.observer=parent.observer
        self.reader=parent.reader; self.ap=parent.ap
        rows=fitting.rows(); groups=sorted({r["group"] for r in rows})
        self.split=dict(fold=0,train_groups=groups[:9],held_out_groups=groups[9:])
        self.package=stage.common.fit_generator(rows,self.split)
        rng=np.random.Generator(np.random.PCG64(13))
        examples=[dict(r,x=rng.normal(size=(4,64)),y=rng.uniform(size=4)) for r in rows if r["group"] in groups[:9]]
        self.model=stage.common.train(examples,self.split,"shared",2312,dict(stage.common.DESIGN,epochs=2))

    def collect(self,arm,entry=None,scorer=None):
        return stage.collect(self.observer,self.entry if entry is None else entry,self.old,arm,self.package["generator"],
                             self.model if scorer is None else scorer,image_reader=self.reader)

    def test_full_own_history_cold_start_call_budget_and_gt_exclusion(self):
        for arm in stage.ARMS:
            before=self.observer.calls; data=self.collect(arm)
            self.assertEqual(self.observer.calls-before,int(data["unique_search_calls"].sum()))
            self.assertEqual(int((data["head_calls"]*data["executed"]).sum()),self.observer.calls-before)
            self.assertEqual(int((data["block_calls"]*data["executed"][...,None]).sum()),7*(self.observer.calls-before))
            stage.audit(data,self.entry,self.old,arm,self.package["generator"],self.model)
            np.testing.assert_array_equal(data["history_latest"][1:],data["boxes"][:-1])
            self.assertFalse(data["generated_active"][0])
            self.assertEqual(data["generated_active"][1:].all(),arm.startswith("generated"))
            changed=dict(self.entry,ground_truth=np.full_like(self.entry["ground_truth"],999.))
            repeated=self.collect(arm,changed)
            for key in data:
                if key not in ("pipeline_ms","search_ms"): np.testing.assert_array_equal(data[key],repeated[key])
            corrupt={k:v.copy() for k,v in data.items()}; corrupt["history_latest"][3,0]+=1
            with self.assertRaises(AssertionError): stage.audit(corrupt,self.entry,self.old,arm,self.package["generator"],self.model)

    def test_forecast_reads_own_previous_response_and_trust_only(self):
        data=self.collect("generated_shared"); history=[self.entry["initial_box"].copy()]
        for i in range(1,len(data["frame_id"])):
            history=(history+[data["boxes"][i-1]])[-5:]
            expected=stage.p.online_input(history,data["features"][i-1],int(data["selected"][i-1]),bool(data["memory_write"][i-1]))
            np.testing.assert_array_equal(data["forecast_x"][i],expected)
        row={k:data[k][0] for k in data}
        self.assertEqual(bool(data["memory_write"][0]),stage.s103.s93.s92.trusted(stage.s103.s93.action_bank(row),0,stage.s103.DESIGN))
        changed={k:v.copy() for k,v in data.items()}; changed["forecast_x"][4,0]+=1
        with self.assertRaises(AssertionError): stage.audit(changed,self.entry,self.old,"generated_shared",self.package["generator"],self.model)

    def test_changed_scores_change_future_own_crops(self):
        # Deliberately score a target-size choice differently, without changing GT.
        original=stage.choose
        def forced(row,history,arm,scorer):
            score,choice=original(row,history,arm,scorer)
            if arm=="generated_shared":
                choice=int(np.argmax(row["candidate_parents"][:,2:].prod(1)))
            return score,choice
        baseline=self.collect("generated_shared")
        with patch.object(stage,"choose",side_effect=forced): changed=self.collect("generated_shared")
        self.assertTrue(np.any(changed["boxes"]!=baseline["boxes"]))
        self.assertTrue(np.any(changed["candidate_parents"][2:]!=baseline["candidate_parents"][2:]))
        np.testing.assert_array_equal(changed["history_latest"][1:],changed["boxes"][:-1])

    def test_recovery_matching_preserves_events_and_detects_a_lost_recovery(self):
        reference=np.array([.1]*4+[.8]*5+[.1]*4+[.8]*5)
        identity=dict(cohort="historical",sequence="test",arm="generated_shared",seed=2312)
        matched=stage.recovery_rows(reference,reference,identity,"native")
        self.assertEqual(len(matched),2); self.assertTrue(all(r["matched_good_frame"] is not None for r in matched))
        lost=stage.recovery_rows(reference,np.full(len(reference),.1),identity,"native")
        self.assertTrue(all(r["matched_good_frame"] is None for r in lost))

    def test_runtime_allows_first_card_change_and_binds_resume_identity(self):
        first=dict(tracker=dict(device="NVIDIA L20",dtype="float32"),gpu_uuid="a")
        current=dict(tracker=dict(device="NVIDIA L40",dtype="float32"),gpu_uuid="b")
        self.assertEqual(len(stage.runtime_binding(first,current,current)),2)
        with self.assertRaisesRegex(ValueError,"STAGE113_ROOT"): stage.runtime_binding(first,current,first)
        bad=copy.deepcopy(current); bad["tracker"]["dtype"]="float16"
        with self.assertRaisesRegex(ValueError,"software"): stage.runtime_binding(first,bad)

    def test_model_loader_preserves_frozen_outer_folds_and_rejects_leakage(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(stage.common,"DESIGN",self.model["design"]):
            root=Path(directory); (root/"models").mkdir()
            stage.c.publish_identical(root/"models/generator_f0.json",self.package)
            for bank in ("fixed_4","generated_4"):
                for seed in stage.DESIGN["seeds"]:
                    model=dict(self.model,seed=seed); model=stage.io.seal({k:v for k,v in model.items() if k!="fingerprint"})
                    stage.c.publish_identical(stage.s112.model_path(root,0,bank,"shared",seed),model)
            before={str(q):q.read_bytes() for q in root.rglob("*") if q.is_file()}
            models,scope=stage.load_frozen_models(root,dict(folds=[self.split]))
            self.assertEqual(len(models),5); self.assertEqual(len(scope),5)
            self.assertEqual(before,{str(q):q.read_bytes() for q in root.rglob("*") if q.is_file()})
            bad=copy.deepcopy(self.split); bad["held_out_groups"].append(bad["train_groups"][0])
            with self.assertRaisesRegex(ValueError,"leakage"): stage.load_frozen_models(root,dict(folds=[bad]))

    def test_end_to_end_frozen_models_complete_seeds_resume_and_tamper(self):
        specs=[dict(cohort=cohort,unit=dict(sequence=f"uav{n:07d}_00000_s",seed=2171,origin="native",policy="ppo",fold=0))
               for n,cohort in ((9,"historical"),(10,"expanded"))]
        owners=stage.owners(specs,[self.split])
        for owner in owners: owner["frames"]=len(self.entry["frames"])-1
        models={(0,"generator",0):self.package}; model_scope=[dict(kind="generator",outer_fold=0)]
        for bank in ("fixed_4","generated_4"):
            for seed in stage.DESIGN["seeds"]:
                model=copy.deepcopy(self.model); model["seed"]=seed
                model=stage.io.seal({k:v for k,v in model.items() if k!="fingerprint"})
                models[0,bank,seed]=model; model_scope.append(dict(kind="shared",bank=bank,seed=seed,outer_fold=0))
        native=self.collect("native")
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            plan=stage.io.seal(dict(schema=stage.SCHEMA,design=stage.DESIGN,entries=owners,jobs=stage.jobs_for(owners),folds=[self.split],
                source_stage112_fingerprint=stage.SOURCE112,native_head_sha256="native",runtime={},runtime_differences=[],source_sha256={},code_sha256={}))
            stage.c.publish_identical(root/"manifest.json",plan)
            original_metric=stage.s103.s93.s85.metrics
            def metric(*args):
                self.assertEqual(len(list((root/"evaluation").glob("*.receipt.json")))%7,0)
                return original_metric(*args)
            signature=stage.io.seal(dict(models={str(k):v for k,v in models.items()}))["fingerprint"]
            with (patch.object(stage.e63.source_experiment,"installed",side_effect=lambda *_:contextlib.nullcontext("native")),
                  patch.object(stage.s103.s93.s85,"metrics",side_effect=metric),
                  patch.object(stage.common,"train",side_effect=AssertionError("New scorer training")),
                  patch.object(stage.common,"fit_generator",side_effect=AssertionError("New generator fitting")),
                  contextlib.redirect_stdout(io.StringIO())):
                before=self.observer.calls
                args=(root,plan,models,model_scope,lambda _:self.entry,lambda _:native,self.old,self.observer)
                stage.run(*args,image_reader=self.reader)
                report=stage.c.c66.sealed(root/"analysis/stage113_decision.json",stage.SCHEMA)
                self.assertEqual(report["new_search_calls"],self.observer.calls-before)
                self.assertEqual(report["trajectories"],14)
                snapshot={str(q):q.read_bytes() for q in root.rglob("*") if q.is_file()}
                with patch.object(self.observer,"capture",side_effect=AssertionError("Repeated GPU inference")):
                    stage.run(*args,image_reader=self.reader)
                self.assertEqual(snapshot,{str(q):q.read_bytes() for q in root.rglob("*") if q.is_file()})
            stage.check_results(root)
            self.assertEqual(signature,stage.io.seal(dict(models={str(k):v for k,v in models.items()}))["fingerprint"])
            target=next((root/"evaluation").glob("*.npz")); target.write_bytes(b"changed")
            with self.assertRaises(ValueError): stage.check_results(root)


if __name__=="__main__": unittest.main()
