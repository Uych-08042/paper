"""Check learned proposal isolation, calibration, equal budgets and real shared-image replay."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage110_search_proposals as stage
import test_stage105_commit_branches as fixture
import test_stage106_commit_gate as f106

p=stage.p


def fitting_rows():
    rng=np.random.Generator(np.random.PCG64(9));rows=[]
    for g in range(1,8):
        for mode in stage.s103.MODES[1:]:
            for f in range(1,33):
                x=np.zeros(32);x[0]=float(f%2);x[1]=rng.normal()
                y=np.array([.15*x[1],0,0,0])+rng.normal(size=4)*(.01+.5*x[0])
                rows.append(dict(group=f"uav{g:07d}",sequence=f"uav{g:07d}_00000_s",seed=1,source_arm=mode,frame=f,x=x.tolist(),y=y.tolist()))
    return rows


class SearchProposals(unittest.TestCase):
    setUp=fixture.CommitBranches.setUp

    def test_past_only_whitelist_and_gt_independent_sampling(self):
        trace=stage.s103.collect(self.observer,self.entry,self.old,f106.choose,"four_matched",stage.s103.DESIGN,image_reader=self.reader)
        x,h=p.input_at(trace,self.entry,7,self.old)
        altered={k:v.copy() for k,v in trace.items()}
        for k in ("boxes","features","selected","memory_write"):altered[k][6:]=999
        entry=dict(self.entry,ground_truth=np.full((13,4),999.))
        a,b=p.input_at(altered,entry,7,self.old)
        np.testing.assert_array_equal(x,a);np.testing.assert_array_equal(h,b)
        design=dict(p.DESIGN,fit_per_context=3,eval_per_context=2)
        sample=p.sample_frames(trace["frame_id"],"context",design)
        self.assertEqual(sample,p.sample_frames(altered["frame_id"],"context",design))
        self.assertFalse(set(sample["fit_frames"])&set(sample["eval_frames"]))
        self.assertEqual(len(sample["eval_frames"]),2)
        self.assertEqual(x.shape,(32,));self.assertEqual(p.sample_frames(np.arange(1,4),"short",design)["eval_frames"],[])

    def test_group_isolation_inner_residuals_and_calibration(self):
        rows=fitting_rows();group="uav0000001";model=p.fit_model(rows,group,p.DESIGN)
        changed=copy.deepcopy(rows)
        for r in changed:
            if r["group"]==group:r["x"]=[999.]*32;r["y"]=[-999.]*4
        self.assertEqual(model,p.fit_model(changed,group,p.DESIGN))
        self.assertNotIn(group,model["fit_groups"]+model["calibration_groups"])
        self.assertFalse(set(model["fit_groups"])&set(model["calibration_groups"]))
        changed_cal=copy.deepcopy(rows)
        for r in changed_cal:
            if r["group"] in model["calibration_groups"]:r["y"]=[10*v for v in r["y"]]
        recalibrated=p.fit_model(changed_cal,group,p.DESIGN)
        self.assertEqual(model["mean"],recalibrated["mean"]);self.assertEqual(model["variance"],recalibrated["variance"])
        self.assertNotEqual(model["conditional_quantile"],recalibrated["conditional_quantile"])
        for fold in model["inner_folds"]:
            self.assertFalse(set(fold["fit_groups"])&set(fold["residual_groups"]))
        cal=[r for r in rows if r["group"] in model["calibration_groups"]]
        mean,sigma=p.forecast(model,np.asarray([r["x"] for r in cal]));err=np.abs(np.asarray([r["y"] for r in cal])-mean)
        covered=(err<=sigma*model["conditional_quantile"]).all(1)
        self.assertGreaterEqual(float(p.weights(cal)@covered),p.DESIGN["nominal_coverage"]-1e-12)
        lo=np.zeros(32);hi=lo.copy();hi[0]=1
        self.assertGreater(p.forecast(model,hi)[1].mean(),p.forecast(model,lo)[1].mean())
        bad=copy.deepcopy(model);bad["mean"]["beta"][0][0]+=.1
        with self.assertRaises(ValueError):p.model_audit(bad,rows)

    def test_runtime_first_card_allowed_and_resume_identity_bound(self):
        old=dict(tracker=dict(device="NVIDIA L20",dtype="float32"),gpu_uuid="old")
        current=dict(tracker=dict(device="NVIDIA L40",dtype="float32"),gpu_uuid="new")
        self.assertEqual(len(stage.runtime_binding(old,current,current)),2)
        with self.assertRaisesRegex(ValueError,"STAGE110_ROOT"):stage.runtime_binding(old,current,old)
        bad=copy.deepcopy(current);bad["tracker"]["dtype"]="float16"
        with self.assertRaisesRegex(ValueError,"software"):stage.runtime_binding(old,bad)

    def test_algorithm_banks_joint_center_scale_q0_and_determinism(self):
        model=p.fit_model(fitting_rows(),"uav0000001",p.DESIGN)
        history=np.array([[75+i,80,15+i,20+i] for i in range(5)],float)
        x=np.zeros(32);x[1]=.5
        a=p.proposal_banks(history,(200,200),self.ap,x,model,"context",p.DESIGN)
        b=p.proposal_banks(history,(200,200),self.ap,x,model,"context",p.DESIGN)
        for k in a:np.testing.assert_array_equal(a[k],b[k])
        self.assertEqual(a["candidate_parents"].shape,(6,4,4))
        np.testing.assert_array_equal(a["candidate_parents"][:,0],np.tile(history[-1],(6,1)))
        self.assertTrue(np.any(a["candidate_parents"][4,1:,2:]!=a["candidate_parents"][0,1:,2:]))
        self.assertTrue(np.any(a["candidate_parents"][4,1:,:2]!=a["candidate_parents"][0,1:,:2]))
        self.assertTrue(np.isfinite(a["candidate_parents"]).all())
        # Random and greedy controls receive the same state hypotheses and proposal pool.
        means=a["forecast_mean"];sigma=a["conditional_radius"]
        hypotheses=p.parents_from_states(means+p.scenarios(p.DESIGN)*sigma,history[-1],(200,200))
        options=np.vstack((history[-1],p.parents_from_states(means[None],history[-1],(200,200)),hypotheses))
        for ai in (4,5):np.testing.assert_array_equal(a["candidate_parents"][ai],options[a["allocation_indices"][ai]])

    def test_real_decoder_unique_call_budget_and_observation_tamper(self):
        model=p.fit_model(fitting_rows(),"uav0000001",p.DESIGN)
        h=np.tile(self.entry["initial_box"],(5,1));x=np.zeros(32)
        bank=p.proposal_banks(h,(200,200),self.ap,x,model,"context",p.DESIGN)
        self.observer.initialize(self.reader(),self.entry["initial_box"])
        before=self.observer.calls
        data=stage.observe_banks(self.observer,self.reader(),h,7,self.ap,bank["candidate_parents"])
        self.assertEqual(self.observer.calls-before,int(data["unique_search_calls"]))
        self.assertLessEqual(int(data["unique_search_calls"]),19)
        stage.audit_observations(data,bank["candidate_parents"],h,7,(200,200),self.ap)
        data["candidate_boxes"][3,1,0]+=1
        with self.assertRaises(AssertionError):stage.audit_observations(data,bank["candidate_parents"],h,7,(200,200),self.ap)

    def test_full_pipeline_resume_sources_predictions_before_gt_and_results_tamper(self):
        design=dict(p.DESIGN,fit_per_context=3,eval_per_context=2)
        spec=dict(cohort="historical",unit=dict(sequence="uav0000001_00000_s",seed=2171,origin="native",policy="ppo",fold=0))
        traces={mode:stage.s103.collect(self.observer,self.entry,self.old,f106.choose,mode,stage.s103.DESIGN,image_reader=self.reader) for mode in stage.s103.MODES[1:]}
        rows=fitting_rows()
        with patch.object(p,"DESIGN",design),tempfile.TemporaryDirectory() as directory:
            root,upstream=Path(directory)/"stage110",Path(directory)/"stage103";root.mkdir();upstream.mkdir()
            p103=stage.io.seal(dict(schema=stage.s103.SCHEMA,design=stage.s103.DESIGN))
            jobs=[]
            for mode,data in traces.items():
                context=stage.c.key(spec["unit"])+"__"+mode
                stage.c.save(upstream/"evaluation"/(context+".npz"),data,dict(plan_fingerprint=p103["fingerprint"],unit=spec["unit"],arm=mode))
                jobs.append(dict(spec=spec,source_arm=mode,context=context,source_states=len(data["frame_id"]),sampling=p.sample_frames(data["frame_id"],context,design)))
            plan=stage.io.seal(dict(schema=p.SCHEMA,design=design,units=jobs,groups=["uav0000001"],runtime={},runtime_differences=[],
                native_head_sha256="native",fit_input_fingerprint=stage.io.seal(dict(rows=rows))["fingerprint"],source_sha256={},code_sha256={}))
            stage.c.publish_identical(root/"manifest.json",plan)
            before_sources={str(q):q.read_bytes() for q in upstream.rglob("*") if q.is_file()}
            before=self.observer.calls
            original_observe=stage.observe_banks
            def observed(*args):
                frame=args[3]
                self.assertTrue(any((root/"proposals").glob(f"*__f{frame:06d}.receipt.json")))
                return original_observe(*args)
            with (patch.object(stage.e63.source_experiment,"installed",side_effect=lambda *_:contextlib.nullcontext("native")),
                  patch.object(stage.s105,"checked_image",side_effect=self.reader),patch.object(stage,"observe_banks",side_effect=observed),
                  contextlib.redirect_stdout(io.StringIO())):
                stage.run(root,upstream,p103,plan,rows,lambda _:self.entry,lambda _:f106.choose,self.old,self.observer)
                actual_calls=self.observer.calls-before
                with patch.object(self.observer,"capture",side_effect=AssertionError("Repeated inference")):
                    stage.run(root,upstream,p103,plan,rows,lambda _:self.entry,lambda _:f106.choose,self.old,self.observer)
            stage.check_results(root)
            report=stage.c.c66.sealed(root/"analysis/stage110_decision.json",p.SCHEMA)
            self.assertEqual(report["new_search_calls"],actual_calls);self.assertEqual(report["logical_search_calls"],4*24)
            self.assertEqual(report["counts"]["state_supply"],24);self.assertEqual(report["maximum_fixed_source_box_error_px"],0.)
            self.assertEqual(before_sources,{str(q):q.read_bytes() for q in upstream.rglob("*") if q.is_file()})
            # Deployment inputs and proposals are exactly unchanged when evaluation GT is replaced.
            trace=next(iter(traces.values()));frame=7
            altered=dict(self.entry,ground_truth=np.full((13,4),999.))
            a,h=p.input_at(trace,self.entry,frame,self.old);b,hh=p.input_at(trace,altered,frame,self.old)
            np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(h,hh)
            next((root/"observations").glob("*.npz")).write_bytes(b"tampered")
            with self.assertRaises(ValueError):stage.check_results(root)


if __name__=="__main__":unittest.main()
