"""Check cached diagnostic semantics, chronology, no-inference replay and provenance."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import stage114_cached_failure_chain as stage
import test_stage113_frozen_closed_loop as fixture


class CachedFailureChain(unittest.TestCase):
    def setUp(self):
        base=fixture.FrozenClosedLoop();base.setUp();self.addCleanup(base.doCleanups);self.base=base

    def test_quality_categories_keep_bad_supply_separate_from_bad_choice(self):
        n=6;gt=np.tile([50.,50.,10.,10.],(n+1,1));gt[-1]=0
        good=np.array([50.,50.,10.,10.]);bad=good+[50.,0,0,0];medium=good+[5.,0,0,0]
        boxes=np.tile(bad,(n,4,1));boxes[0,0]=good;boxes[1,1]=good;boxes[3,0]=good;boxes[3,1]=medium
        selected=np.array([1,1,1,1,1,1]);entry=dict(initial_box=good,ground_truth=gt)
        data=dict(candidate_boxes=boxes,selected=selected,boxes=boxes[np.arange(n),selected],frame_id=np.arange(1,n+1),
            candidate_parents=np.tile(good,(n,4,1)),image_hw=np.tile([200,200],(n,1)),generated_active=np.zeros(n,bool),
            scores=np.ones((n,4)),memory_write=np.zeros(n,bool))
        a=stage.analyze(data,entry,"fixed_hann",None,None,self.base.old)
        np.testing.assert_array_equal(a["state"],[3,1,2,3,2,0])
        np.testing.assert_array_equal(a["harmful_q0"],[1,0,0,0,0,0]);np.testing.assert_array_equal(a["beneficial_q0"],[0,1,0,0,0,0])
        np.testing.assert_array_equal(a["severe_miss"],[1,0,0,0,0,0]);self.assertTrue(a["missed_good"][3]);self.assertFalse(a["all_low"][-1])
        self.assertEqual(stage.run_start([False,True,True,True],3),2)
        self.assertIsNone(stage.run_start([True,True],3))
        self.assertIsNone(stage.finite_mean([None,np.nan]));self.assertEqual(stage.finite_mean([None,2]),2.)

    def test_labels_do_not_change_actions_support_or_one_commit_geometry(self):
        b=self.base;data=b.collect("generated_shared");before={k:v.copy() for k,v in data.items()}
        with patch.object(stage.source.e63.actor,"observe",side_effect=AssertionError("New inference")):
            a=stage.analyze(data,b.entry,"generated_shared",b.package["generator"],b.model,b.old)
            changed=dict(b.entry,ground_truth=np.full_like(b.entry["ground_truth"],999.))
            other=stage.analyze(data,changed,"generated_shared",b.package["generator"],b.model,b.old)
            np.testing.assert_array_equal(a["forecast_standardized_max"],other["forecast_standardized_max"])
            np.testing.assert_array_equal(a["scorer_standardized_max"],other["scorer_standardized_max"])
            target=stage.source.p.target(np.asarray(b.entry["ground_truth"])[2],data["history_latest"][1])
            ratio=(np.abs(target-data["forecast_mean"][1])/np.maximum(data["conditional_radius"][1],1e-12)).max()
            self.assertAlmostEqual(a["range_ratio"][1],ratio);self.assertTrue(np.isnan(a["range_ratio"][0]))
            self.assertEqual(stage.geometry_shadow(data,b.entry,3,"generated_shared",b.package["generator"],b.old),
                             stage.geometry_shadow(data,changed,3,"generated_shared",b.package["generator"],b.old))
            self.assertFalse(stage.geometry_shadow(data,b.entry,len(data["frame_id"])-1,"generated_shared",b.package["generator"],b.old)["next_shadow_available"])
        for k in data:np.testing.assert_array_equal(before[k],data[k])
        phases=stage.support_rows(a,dict(arm="generated_shared"))
        self.assertEqual(sum(r["frames"] for r in phases),len(data["frame_id"]))
        self.assertEqual(sum(r["frames"] for r in phases if r["phase"]=="short_history"),3)

    def test_prefix_first_difference_and_censoring_use_exact_frame_indices(self):
        good=np.array([50.,50.,10.,10.]);bad=good+[30.,0,0,0];n=4
        reference=dict(frame_id=np.arange(1,n+1),boxes=np.tile(good,(n,1)),history_latest=np.tile(good,(n,1)),
            candidate_parents=np.tile(good,(n,4,1)),candidate_boxes=np.tile(good,(n,4,1)),crop=np.ones((n,4,4)))
        candidate={k:v.copy() for k,v in reference.items()};candidate["boxes"][1]=bad;candidate["history_latest"][2]=bad
        a=dict(selected_iou=np.array([1,0,1,1]),q0_iou=np.ones(n),best_iou=np.ones(n),harmful_q0=np.array([0,1,0,0],bool))
        r=stage.prefix(candidate,reference,a,np.ones(n),{},"native")
        self.assertEqual(r["first_output_difference"],2);self.assertEqual(r["first_history_difference"],3);self.assertTrue(r["at_first_harmful_q0"])
        candidate["candidate_boxes"][0,0]=bad
        with self.assertRaises(AssertionError):stage.prefix(candidate,reference,a,np.ones(n),{},"native")
        simple=dict(frame_id=np.arange(1,6),harmful_q0=np.array([0,1,0,0,0],bool),all_low=np.array([0,0,1,1,1],bool),
                    best_iou=np.ones(5),max_coverage=np.ones(5),range_contains_truth=np.ones(5))
        self.assertEqual(stage.local_window(simple,4)["local_order"],"harmful_first")
        simple["all_low"][0]=True;self.assertEqual(stage.local_window(simple,4)["local_order"],"left_censored")

    def test_complete_cached_pipeline_zero_images_models_unchanged_resume_and_tamper(self):
        b=self.base;s=stage.source
        specs=[dict(cohort=cohort,unit=dict(sequence=f"uav{n:07d}_00000_s",seed=2171,origin="native",policy="ppo",fold=0))
               for n,cohort in ((9,"historical"),(10,"expanded"))]
        entries=s.owners(specs,[b.split])
        for e in entries:e["frames"]=len(b.entry["frames"])-1
        models={(0,"generator",0):b.package};model_scope=[dict(kind="generator",outer_fold=0)]
        for bank in ("fixed_4","generated_4"):
            for seed in s.DESIGN["seeds"]:
                m=copy.deepcopy(b.model);m["seed"]=seed;models[0,bank,seed]=s.io.seal({k:v for k,v in m.items() if k!="fingerprint"})
                model_scope.append(dict(kind="shared",bank=bank,seed=seed,outer_fold=0))
        native=b.collect("native");signature=s.io.seal(dict(models={str(k):v for k,v in models.items()}))["fingerprint"]
        # Exercise real failure, recovery and terminal-censor table branches while
        # keeping every actor input unchanged; GT is evaluation-only.
        entry=copy.deepcopy(b.entry)
        truth=np.vstack((np.asarray(entry["initial_box"]),native["boxes"])).copy()
        truth[3:6,:2]+=500;truth[-3:,:2]+=500;entry["ground_truth"]=truth
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            base=Path(directory);r113=base/"source";r113.mkdir();root=base/"audit";root.mkdir()
            p113=s.io.seal(dict(schema=s.SCHEMA,design=s.DESIGN,entries=entries,jobs=s.jobs_for(entries),folds=[b.split],
                source_stage112_fingerprint=s.SOURCE112,native_head_sha256="native",runtime={},runtime_differences=[],source_sha256={},code_sha256={}))
            s.c.publish_identical(r113/"manifest.json",p113)
            with patch.object(s.e63.source_experiment,"installed",side_effect=lambda *_:contextlib.nullcontext("native")):
                s.run(r113,p113,models,model_scope,lambda _:entry,lambda _:native,b.old,b.observer,image_reader=b.reader)
            old_snapshot={str(p):p.read_bytes() for p in r113.rglob("*") if p.is_file()}
            oldrows=s.s103.prior.read_csv(r113/"analysis/stage113_seed_video.csv");recoveries=s.s103.prior.read_csv(r113/"analysis/stage113_recovery.csv")
            report113=s.c.c66.sealed(r113/"analysis/stage113_decision.json",s.SCHEMA)
            expected=dict(trajectories=14,frames=sum(j["frames"] for j in p113["jobs"]),recoveries=len(recoveries),source_search_calls=report113["new_search_calls"],failures=int(sum(float(r["failure_events"]) for r in oldrows)),prefixes=18)
            self.assertGreater(expected["recoveries"],0);self.assertGreater(expected["failures"],0)
            plan=s.io.seal(dict(schema=stage.SCHEMA,design=stage.DESIGN,expected=expected,source_stage113_fingerprint=stage.SOURCE113,source_runtime={},
                source_sha256={p:s.c.file_hash(Path(p)) for p in old_snapshot},code_sha256={}))
            s.c.publish_identical(root/"manifest.json",plan)
            with (patch.object(s.e63.actor,"observe",side_effect=AssertionError("New network inference")),
                  patch.object(s.s105,"checked_image",side_effect=AssertionError("Image read")),
                  patch.object(s.common,"train",side_effect=AssertionError("New training")),
                  patch.object(s.common,"fit_generator",side_effect=AssertionError("New fitting"))):
                stage.run(root,r113,p113,plan,models,lambda _:entry,b.old)
                stage.check_results(root)
                snapshot={str(p):p.read_bytes() for p in root.rglob("*") if p.is_file()}
                with patch.object(stage,"analyze",side_effect=AssertionError("Recomputing committed diagnostic cache")):
                    stage.run(root,r113,p113,plan,models,lambda _:entry,b.old)
                self.assertEqual(snapshot,{str(p):p.read_bytes() for p in root.rglob("*") if p.is_file()})
            self.assertEqual(old_snapshot,{str(p):p.read_bytes() for p in r113.rglob("*") if p.is_file()})
            self.assertEqual(signature,s.io.seal(dict(models={str(k):v for k,v in models.items()}))["fingerprint"])
            cache=next((root/"audit").glob("*.npz"));cache.write_bytes(b"tampered")
            with self.assertRaises(ValueError):stage.check_results(root)


if __name__=="__main__":unittest.main()
