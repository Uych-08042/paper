"""Check nested source exclusion, real MLP gradients, K invariance and sealed replay."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import stage112_shared_scoring as stage
import test_stage110_proposals as fixture
import test_stage106_commit_gate as f106
import test_stage111_slot_audit as actor_fixture

c = stage.common


def rows(groups=12):
    rng = np.random.Generator(np.random.PCG64(112)); out = []
    for group in range(groups):
        for frame in range(6):
            x = rng.normal(size=32); y = np.r_[.1*x[:2],.02*x[2:4]]
            out.append(dict(group=f"uav{group:07d}",sequence=f"uav{group:07d}_00000_s",seed=2171,source_arm="four_matched",frame=frame,x=x.tolist(),y=y.tolist()))
    return out


class SharedScoring(unittest.TestCase):
    def test_real_shared_and_concat_gradients(self):
        rng = np.random.Generator(np.random.PCG64(8))
        x = rng.normal(size=(3,4,64)); y = rng.uniform(size=(3,4)); weights = np.array([.4,1.,1.6])
        for architecture in ("shared","concat"):
            state = c.initial(9,architecture,c.DESIGN)
            _,gradient = c.loss_gradient(state,x,y,weights,architecture)
            for key,index in (("w1",(0,0)),("b1",(0,)),("w2",(0,0)),("b2",(0,))):
                delta = 1e-6; state[key][index] += delta
                hi = c.loss_gradient(state,x,y,weights,architecture)[0]
                state[key][index] -= 2*delta
                lo = c.loss_gradient(state,x,y,weights,architecture)[0]
                state[key][index] += delta
                self.assertAlmostEqual(float(gradient[key][index]),(hi-lo)/(2*delta),places=8)

    def test_shared_features_and_scorer_equivariant_for_all_budgets(self):
        rng = np.random.Generator(np.random.PCG64(7)); history = np.tile([50.,50.,20.,20.],(5,1))
        norm = dict(center=[0.]*64,scale=[1.]*64)
        model = dict(architecture="shared",state={k:v.tolist() for k,v in c.initial(1,"shared",c.DESIGN).items()},norm=norm,design=c.DESIGN)
        for count in (2,4,6,8):
            boxes = np.c_[rng.uniform(20,100,(count,2)),rng.uniform(10,30,(count,2))]; base = rng.normal(size=(count,24))
            original = c.features(base,boxes,history); score = c.scores(model,original)
            order = np.r_[0,rng.permutation(np.arange(1,count))]
            changed = c.features(base[order],boxes[order],history)
            np.testing.assert_allclose(changed,original[order],rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(c.scores(model,changed),score[order],rtol=1e-12,atol=1e-12)
            np.testing.assert_array_equal(boxes[c.physical_choice(score,boxes)],boxes[order][c.physical_choice(c.scores(model,changed),boxes[order])])
            if count == 4: np.testing.assert_array_equal(original,c.prior.c74.c66.c64.choice_features(base,boxes,history))
        self.assertEqual(c.physical_choice(np.ones(8),boxes),0)

    def test_nested_generator_and_scorer_exclude_every_evaluation_group(self):
        data = rows(); groups = sorted({r["group"] for r in data})
        split = dict(fold=0,train_groups=groups[:9],held_out_groups=groups[9:])
        first = c.fit_generator(data,split); changed = copy.deepcopy(data)
        for row in changed:
            if row["group"] in split["held_out_groups"]: row["x"]=[999.]*32; row["y"]=[-999.]*4
        self.assertEqual(first,c.fit_generator(changed,split))
        generator = first["generator"]
        self.assertFalse(set(split["held_out_groups"])&set(generator["fit_groups"]+generator["calibration_groups"]))
        rng = np.random.Generator(np.random.PCG64(5))
        examples = [dict(r,x=rng.normal(size=(4,64)),y=rng.uniform(size=4)) for r in data if r["group"] in split["train_groups"]]
        design = dict(c.DESIGN,epochs=3)
        for architecture in ("shared","concat"):
            model = c.train(examples,split,architecture,1,design)
            self.assertLess(model["final_fit_mse"],model["first_fit_mse"])
            self.assertEqual(model,c.train(examples,split,architecture,1,design))
            poisoned = examples+[dict(data[-1],x=np.zeros((4,64)),y=np.ones(4))]
            with self.assertRaisesRegex(ValueError,"leakage"): c.train(poisoned,split,architecture,1,design)

    def test_nested_prefix_budgets_and_gt_free_forecasting(self):
        parent = fixture.SearchProposals(); parent.setUp(); self.addCleanup(parent.doCleanups)
        generator = c.fit_generator(rows(),dict(fold=0,train_groups=[f"uav{i:07d}" for i in range(9)],held_out_groups=[f"uav{i:07d}" for i in range(9,12)]))["generator"]
        history = np.array([[75+i,80,15+i,20+i] for i in range(5)],float)
        x = np.zeros(32)
        bank = c.search_parents(history,(200,200),parent.ap,x,generator,8)
        for count in (2,4,6,8):
            other = c.search_parents(history,(200,200),parent.ap,x,generator,count)
            np.testing.assert_array_equal(other["candidate_parents"][4:],bank["candidate_parents"][4:4+count])
        np.testing.assert_array_equal(bank["candidate_parents"][0],bank["candidate_parents"][4])
        parent.observer.initialize(parent.reader(),parent.entry["initial_box"])
        before = parent.observer.calls
        observed = stage.observe(parent.observer,parent.reader(),history,7,parent.ap,bank["candidate_parents"])
        self.assertEqual(parent.observer.calls-before,int(observed["unique_search_calls"]))
        self.assertLessEqual(int(observed["unique_search_calls"]),11)
        stage.audit_observation(observed,bank["candidate_parents"],history,7,(200,200),parent.ap)
        observed["candidate_boxes"][2,0] += 1
        with self.assertRaises(AssertionError): stage.audit_observation(observed,bank["candidate_parents"],history,7,(200,200),parent.ap)

    def test_complete_pipeline_resume_no_retraining_no_inference_and_tamper(self):
        parent = fixture.SearchProposals(); parent.setUp(); self.addCleanup(parent.doCleanups)
        actor = actor_fixture.SlotAudit(); actor.setUp()
        old = dict(parent.old,design=actor.design)
        design = dict(c.DESIGN,outer_folds=2,fit_per_context=2,seeds=[2312],epochs=2,minibatch=4)
        data_rows = rows(); jobs=[]; traces={}
        for i in range(12):
            unit = dict(sequence=f"uav{i:07d}_00000_s",seed=2171,origin="native",policy="ppo",fold=0)
            for mode in stage.s103.MODES[1:]:
                trace = stage.s103.collect(parent.observer,parent.entry,parent.old,f106.choose,mode,stage.s103.DESIGN,image_reader=parent.reader)
                spec = dict(cohort="historical" if i<6 else "expanded",unit=unit)
                context = stage.c.key(unit)+"__"+mode
                jobs.append(dict(spec=spec,source_arm=mode,context=context,sampling=dict(fit_frames=[5,6],eval_frames=[7])))
                traces[context] = trace
        with patch.object(c,"DESIGN",design),tempfile.TemporaryDirectory() as directory:
            root,r103 = Path(directory)/"run",Path(directory)/"source"; root.mkdir(); r103.mkdir()
            p103 = stage.io.seal(dict(schema=stage.s103.SCHEMA))
            for job in jobs:
                stage.c.save(r103/"evaluation"/(job["context"]+".npz"),traces[job["context"]],
                    dict(plan_fingerprint=p103["fingerprint"],unit=job["spec"]["unit"],arm=job["source_arm"]))
            bindings = {str(q):stage.c.file_hash(q) for q in r103.rglob("*") if q.is_file()}
            plan = stage.io.seal(dict(schema=c.SCHEMA,design=design,units=jobs,folds=c.folds(jobs,2),source_stage111_fingerprint=c.SOURCE111,
                runtime={},runtime_differences=[],native_head_sha256="native",fit_input_fingerprint=stage.io.seal(dict(rows=data_rows))["fingerprint"],
                source_sha256=bindings,code_sha256={}))
            stage.c.publish_identical(root/"manifest.json",plan)
            before_sources = {str(q):q.read_bytes() for q in r103.rglob("*") if q.is_file()}
            original_evaluate = stage.evaluate
            def evaluated(data,pred,truth,ident):
                self.assertTrue(any((root/"predictions").glob(f"*__f{ident['frame']:06d}.receipt.json")))
                return original_evaluate(data,pred,truth,ident)
            with (patch.object(stage.e63.source_experiment,"installed",side_effect=lambda *_:contextlib.nullcontext("native")),
                  patch.object(stage.s105,"checked_image",side_effect=parent.reader),patch.object(stage,"evaluate",side_effect=evaluated),
                  contextlib.redirect_stdout(io.StringIO())):
                args = (root,r103,p103,plan,data_rows,lambda _:parent.entry,lambda _:actor.pack,old,parent.observer)
                before = parent.observer.calls; stage.run(*args); calls = parent.observer.calls-before
                report = stage.c.c66.sealed(root/"analysis/stage112_decision.json",c.SCHEMA)
                self.assertEqual(report["new_search_calls"],calls)
                self.assertEqual(report["new_selector_models"],8); self.assertEqual(report["counts"]["collection_scope"],72)
                with (patch.object(parent.observer,"capture",side_effect=AssertionError("Repeated search")),
                      patch.object(c,"train",side_effect=AssertionError("Repeated scorer training"))): stage.run(*args)
            stage.check_results(root)
            self.assertEqual(before_sources,{str(q):q.read_bytes() for q in r103.rglob("*") if q.is_file()})
            output = next((root/"predictions").glob("*.npz")); original=output.read_bytes(); output.write_bytes(b"changed")
            with self.assertRaises(ValueError): stage.check_results(root)
            output.write_bytes(original)
            next((root/"models").glob("*shared*.json")).write_bytes(b"changed model")
            with self.assertRaises(ValueError): stage.check_results(root)


if __name__=="__main__": unittest.main()
