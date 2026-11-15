"""Check random sampling, frozen choices, matched rollouts, resume and evidence integrity."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage109_sampled_commit as stage
import test_stage105_commit_branches as f105
import test_stage106_commit_gate as f106
import test_stage108_relational_gate as f108


class SampledCommit(unittest.TestCase):
    setUp = f105.CommitBranches.setUp

    def models(self, group="uav0000001"):
        rows = f108.RelationalGate().rows()
        return {(group,a,h):(stage.s106.fit_gate(rows,group,h) if a == "baseline" else stage.s108.fit_gate(rows,group,a,h))
                for a in stage.ARMS for h in stage.s106.HEADS}

    def test_sampler_deterministic_availability_only_full_horizon_and_short_pools(self):
        frames = np.arange(1,201); eligible = frames%3 != 0
        source = dict(frame_id=frames,eligible=eligible,ground_truth=np.ones((200,4)),future=np.ones(200),role="failure")
        def select(data):
            return stage.sample_frames(data["frame_id"],data["eligible"],[2,4,7,160],"context")
        original = select(source)
        changed = dict(source,ground_truth=np.full((200,4),999),future=-np.ones(200),role="benefit")
        self.assertEqual(original,select(changed))
        self.assertEqual(original,select(source));self.assertEqual(len(original["selected"]),12)
        self.assertFalse(set(original["selected"]) & {2,4,7,160})
        self.assertTrue(all(f <= 169 and f%3 for f in original["pool"]))
        self.assertEqual(len(set(original["selected"])),12)
        self.assertNotEqual(original["selected"],stage.sample_frames(frames,eligible,[2,4,7,160],"other")["selected"])
        for length,expected in ((10,0),(33,2),(35,4)):
            record=stage.sample_frames(np.arange(1,length+1),np.ones(length,bool),[],"short")
            self.assertEqual(len(record["selected"]),expected)
            self.assertEqual(record["inclusion_probability"],1. if expected else 0.)
        bad=copy.deepcopy(original);bad["selected"][0]=999
        with self.assertRaises(ValueError):stage.validate_selection(dict(sampling=bad))
        with self.assertRaises(ValueError):stage.sample_frames(frames,np.full(200,np.nan),[],"bad")

    def test_runtime_allows_first_l20_l40_but_binds_resume_and_software(self):
        expected=dict(gpu_uuid="old",tracker=dict(device="NVIDIA L20",dtype="float32"),checkpoint="same")
        actual=copy.deepcopy(expected);actual["gpu_uuid"]="new";actual["tracker"]["device"]="NVIDIA L40"
        self.assertEqual(len(stage.check_runtime(expected,actual)),2)
        self.assertEqual(len(stage.check_runtime(expected,actual,actual)),2)
        with self.assertRaisesRegex(ValueError,"new STAGE109_ROOT"):stage.check_runtime(expected,actual,expected)
        bad=copy.deepcopy(actual);bad["tracker"]["dtype"]="float16"
        with self.assertRaisesRegex(ValueError,"software"):stage.check_runtime(expected,bad)

    def test_source_pool_preparation_and_readonly_frozen_model_loading(self):
        spec=dict(cohort="historical",unit=dict(sequence="uav0000001_00000_s",seed=2171,origin="native",policy="ppo",fold=0))
        models=self.models();frames=np.arange(1,51)
        with tempfile.TemporaryDirectory() as directory:
            r103,r106,r108=[Path(directory)/name for name in ("s103","s106","s108")]
            for root in (r103,r106,r108):root.mkdir()
            p103=stage.io.seal(dict(schema=stage.s103.SCHEMA,units=[spec]))
            p106=stage.io.seal(dict(schema=stage.s106.SCHEMA,groups=["uav0000001"]))
            known=[]
            for mode in stage.s103.MODES[1:]:
                key=stage.c.key(spec["unit"])+"__"+mode
                stage.c.save(r103/"evaluation"/(key+".npz"),dict(frame_id=frames),
                    dict(plan_fingerprint=p103["fingerprint"],unit=spec["unit"],arm=mode))
                stage.c.save(r106/"exposure"/(key+".npz"),dict(frame_id=frames,eligible=np.isin(frames,[1,2,3,40]),
                    offline_current_commit_delta=np.full(50,999.)),
                    dict(plan_fingerprint=p106["fingerprint"],unit=spec["unit"],source_arm=mode,kind="frozen_source_exposure"))
                known.append(dict(sequence=spec["unit"]["sequence"],seed="2171",source_arm=mode,frame="2",future="999",roles="failure"))
            with contextlib.redirect_stdout(io.StringIO()):
                jobs,bindings=stage.prepare(r103,r106,p103,p106,known)
                changed=[dict(r,future="-999",roles="benefit") for r in known]
                self.assertEqual((jobs,bindings),stage.prepare(r103,r106,p103,p106,changed))
            self.assertEqual(len(bindings),8)
            for job in jobs:
                self.assertEqual(job["sampling"]["selected"],[1,3]);stage.validate_selection(job)
            for (group,arm,head),model in models.items():
                path=(r106/"models"/(group+"__"+head+".json") if arm=="baseline" else r108/"models"/(group+"__relation_mode__"+head+".json"))
                path.parent.mkdir(exist_ok=True);stage.c.publish_identical(path,model)
            before={str(p):p.read_bytes() for p in Path(directory).rglob("*") if p.is_file()}
            with (patch.object(stage.s106,"fit_gate",side_effect=AssertionError("New fit")),
                  patch.object(stage.s108,"fit_gate",side_effect=AssertionError("New fit"))):
                actual,hashes=stage.load_models(r106,r108,["uav0000001"])
            self.assertEqual(len(hashes),4);self.assertEqual(models,actual)
            self.assertEqual(before,{str(p):p.read_bytes() for p in Path(directory).rglob("*") if p.is_file()})

    def test_rate_controls_macro_weighting_and_missing_future_not_zero(self):
        samples,predictions=[],[]
        for frame,delta,eligible,commit in ((1,.6,True,True),(2,-.3,True,False),(3,None,True,True),(4,.1,False,False)):
            r=dict(cohort="historical",sequence="uav0000001_00000_s",group="uav0000001",seed=1,
                source_arm="four_matched",frame=frame,future=delta,eligible=eligible)
            samples.append(r)
            for a in stage.ARMS:
                for h in stage.s106.HEADS:predictions.append(dict(r,representation=a,head=h,commit=commit))
        rows=stage.decisions(samples,predictions)
        rates=[r for r in rows if r["policy"]=="relation_mode__rate_expected"]
        self.assertAlmostEqual(rates[0]["commit_probability"],2/3)
        self.assertEqual(rates[-1]["commit_probability"],0.)
        self.assertTrue(all(r["utility"] is None for r in rows if r["frame"]==3))
        summary,paired=stage.summaries(rows)
        target=next(r for r in summary if r["cohort"]==r["source_arm"]=="all" and r["policy"]=="relation_mode__future")
        self.assertAlmostEqual(target["utility"],.2)
        self.assertEqual(len(paired),24)
        # Duplicating every observation in a context preserves its source-group weight.
        self.assertEqual(stage.s106.macro(rows,"utility"),stage.s106.macro(rows+rows,"utility"))

    def test_full_pipeline_exact_budget_prediction_before_labels_resume_and_tamper(self):
        self.entry=dict(self.entry,frames=self.entry["frames"]*4,ground_truth=np.tile(self.entry["initial_box"],(52,1)))
        models=self.models();spec=dict(cohort="historical",unit=dict(sequence="uav0000001_00000_s",seed=2171,origin="native",policy="ppo",fold=0))
        traces={mode:stage.s103.collect(self.observer,self.entry,self.old,f106.choose,mode,stage.s103.DESIGN,image_reader=self.reader)
                for mode in stage.s103.MODES[1:]}
        jobs=[]
        for mode in traces:
            key=stage.c.key(spec["unit"])+"__"+mode
            frames=traces[mode]["frame_id"]
            sampling=stage.sample_frames(frames,frames==3,[],key)
            jobs.append(dict(spec=spec,source_arm=mode,sampling=sampling))
        original_point,original_collect=stage.s105.point,stage.s105.collect
        for changed_card in (False,True):
            with self.subTest(changed_card=changed_card),tempfile.TemporaryDirectory() as directory:
                upstream,root=Path(directory)/"stage103",Path(directory)/"stage109"
                upstream.mkdir();root.mkdir()
                p103=stage.io.seal(dict(schema=stage.s103.SCHEMA,design=stage.s103.DESIGN))
                for mode,data in traces.items():
                    stage.c.save(upstream/"evaluation"/(stage.c.key(spec["unit"])+"__"+mode+".npz"),data,
                        dict(plan_fingerprint=p103["fingerprint"],unit=spec["unit"],arm=mode))
                plan=stage.io.seal(dict(schema=stage.SCHEMA,design=stage.DESIGN,units=jobs,groups=["uav0000001"],runtime={},
                    runtime_differences=[dict(field="runtime.gpu_uuid",expected="old",actual="new")] if changed_card else [],
                    native_head_sha256="native",source_sha256={},code_sha256={}))
                stage.c.publish_identical(root/"manifest.json",plan)
                before_sources={str(p):p.read_bytes() for p in upstream.rglob("*") if p.is_file()}
                def collect(*args):
                    first,mode=args[4],args[7]
                    receipt=list((root/"predictions").glob(f"*__{mode}__f{first:06d}.receipt.json"))
                    self.assertEqual(len(receipt),1)
                    return original_collect(*args,image_reader=self.reader)
                calls=self.observer.calls
                with (patch.object(stage.e63.source_experiment,"installed",side_effect=lambda *_:contextlib.nullcontext("native")),
                      patch.object(stage.s105,"point",side_effect=lambda *a:original_point(*a,image_reader=self.reader)),
                      patch.object(stage.s105,"collect",side_effect=collect),
                      patch.object(stage.s106,"fit_gate",side_effect=AssertionError("New fit")),
                      patch.object(stage.s108,"fit_gate",side_effect=AssertionError("New fit")),
                      contextlib.redirect_stdout(io.StringIO())):
                    stage.run(root,upstream,p103,plan,lambda _:self.entry,lambda _:f106.choose,self.old,self.observer,models)
                    self.assertEqual(self.observer.calls-calls,504)
                    with patch.object(self.observer,"capture",side_effect=AssertionError("Repeated inference")):
                        stage.run(root,upstream,p103,plan,lambda _:self.entry,lambda _:f106.choose,self.old,self.observer,models)
                stage.check_results(root)
                report=stage.c.c66.sealed(root/"analysis/stage109_decision.json",stage.SCHEMA)
                self.assertEqual(report["budget"]["logical_search_calls"],512)
                self.assertEqual(report["budget"]["anchors"],2)
                self.assertEqual(report["fresh_eligibility_changes"],2)
                self.assertEqual(report["maximum_keep_source_box_error_px"],0.)
                self.assertFalse((root/"models").exists())
                self.assertEqual(before_sources,{str(p):p.read_bytes() for p in upstream.rglob("*") if p.is_file()})
                # Predictions use own-past state; later GT cannot change them.
                mode=stage.s103.MODES[1];trace=traces[mode];first=3
                point_path=next(p for p in (root/"points").glob("*.npz") if mode in p.name)
                with np.load(point_path,allow_pickle=False) as z:shared={k:z[k] for k in z.files if k!="metadata_json"}
                a=stage.predict(shared,trace,self.entry,first,mode,self.old,f106.choose,stage.s103.DESIGN,models,"uav0000001")
                b=stage.predict(shared,trace,dict(self.entry,ground_truth=np.full((52,4),999.)),first,mode,self.old,f106.choose,stage.s103.DESIGN,models,"uav0000001")
                for k in a:np.testing.assert_array_equal(a[k],b[k])
                next((root/"points").glob("*.receipt.json")).write_bytes(b"tampered")
                with self.assertRaises(ValueError):stage.check_results(root)


if __name__=="__main__":unittest.main()
