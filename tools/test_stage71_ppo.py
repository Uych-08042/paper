"""PPO credit/legality/optimizer persistence and real CENTER source-to-OOF workflow."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import stage71_ppo_common as c
import stage71_ppo_runtime as r
import stage71_ppo_experiment as e
import stage71_ppo_summary as s


class Semantics(unittest.TestCase):
    def test_terminal_GAE_causal_features_and_illegal_actions(self):
        a, returns = c.gae([0.,0.,1.], [0.,0.,0.], 1., 1.)
        np.testing.assert_array_equal(a, [1.,1.,1.]);np.testing.assert_array_equal(returns,a)
        a, _ = c.gae([1.,2.], [.5,.25], .5, 1.)
        np.testing.assert_array_equal(a, [1.5,1.75])
        f=np.zeros((4,64));f[0]=2;f[1]=3
        norm=dict(center=[0.]*128,scale=[1.]*128)
        np.testing.assert_array_equal(c.normalized(f,1,norm,c.DESIGN),np.r_[np.ones(64),np.full(64,2.)])
        np.testing.assert_array_equal(c.normalized(f,0,norm,c.DESIGN),np.r_[np.zeros(64),np.full(64,2.)])
        self.assertEqual(c.probability(999.,False),0.)
        self.assertEqual(c.log_probability(0,999.,False),0.)
        with self.assertRaises(ValueError):c.log_probability(1,0.,False)
        with self.assertRaises(ValueError):c.normalized(f,4,norm,c.DESIGN)

    def test_production_budget_and_fold_isolation(self):
        names=["v"+str(i) for i in range(12)];frames=[1127]*11+[1128]
        old=dict(entries=[dict(sequence=n,frames=list(range(k+1))) for n,k in zip(names,frames)],
            folds=[dict(fold=0,fit=names[:6],held_out=names[6:]),dict(fold=1,fit=names[6:],held_out=names[:6])],
            runtime_identity=dict(tracker=dict(device="NVIDIA L40"),gpu_uuid="old"),root="unused",search_size=256,search_factor=4.,window=[1.],actor_design={},controller_design={})
        source=dict(plan=old,root=Path("unused"),report=dict(fingerprint="r"),models=dict(fingerprint="m"),bindings={})
        with patch.object(c,"load_source",return_value=source):plan,_=c.make_plan("unused",dict(device="NVIDIA L40",gpu_uuid="new"))
        self.assertEqual(len(plan["jobs"]),4);self.assertEqual(sum(len(j["episodes"]) for j in plan["jobs"]),96)
        self.assertEqual(plan["training_environment_steps"],108200);self.assertEqual(plan["training_search_calls"],432800)
        self.assertEqual(plan["optimizer_updates_per_arm"],1920)
        self.assertEqual(len(plan["units"]),132);self.assertEqual(plan["formal_eval_calls"],595100)
        self.assertEqual(plan["eval_smoke_calls_reused"],1408);self.assertEqual(plan["source_bridge_search_calls"],8)
        self.assertEqual(plan["training_search_calls"]+plan["formal_eval_calls"]+plan["source_bridge_search_calls"],1027908)
        self.assertEqual(plan["jobs"][0]["episodes"],plan["jobs"][1]["episodes"])
        with self.assertRaisesRegex(ValueError,"held-out"):c.check_fit(plan,plan["jobs"][0],names[-1])
        for unit in plan["units"]:self.assertIn(unit["sequence"],next(f for f in plan["folds"] if f["fold"]==unit["fold"])["held_out"])


try:
    import torch
    import cv2
except ImportError:torch=cv2=None


@unittest.skipUnless(torch is not None and cv2 is not None,"Real Torch/OpenCV required; skips do not validate")
class TorchMath(unittest.TestCase):
    def test_clipped_loss_numpy_actor_and_serialized_Adam_continuation(self):
        torch.set_num_threads(1)
        design=dict(c.DESIGN,hidden=4,epochs=2,minibatch=2)
        norm=dict(center=[0.]*128,scale=[1.]*128)
        cpu_rng=torch.random.get_rng_state().clone()
        with patch.object(torch,"manual_seed",side_effect=AssertionError("Do not reseed SGLA CUDA")), \
             patch.object(torch.cuda,"manual_seed_all",side_effect=AssertionError("Do not reseed SGLA CUDA")):
            pack=r.initial(torch,2171,norm,design)
        self.assertTrue(torch.equal(cpu_rng,torch.random.get_rng_state()))
        net=r.restore(torch,pack["state"],design)
        x=np.random.default_rng(4).normal(size=(4,128))
        for branch in ("actor","critic"):
            np.testing.assert_allclose(c.forward(x,pack["state"],branch),net[branch](torch.tensor(x)).detach().numpy()[:,0],rtol=0,atol=1e-12)
        data=dict(x=x,eligible=np.array([True,True,False,True]),action=np.array([1,0,0,1]),
            log_prob=np.array([-np.log(2),-np.log(2),0.,-np.log(2)]),reward=np.array([0.,0.,.2,1.]),value=np.zeros(4),myopic_label=np.array([0.,1.,0.,0.]))
        packs=dict(ppo=pack,myopic=copy.deepcopy(pack))
        updated,stats=r.update(torch,packs,data,design,99)
        self.assertEqual(stats["ppo"]["updates"],stats["myopic"]["updates"])
        self.assertGreater(stats["ppo"]["maximum_actor_gradient"],0.)
        self.assertGreater(stats["myopic"]["maximum_actor_gradient"],0.)
        self.assertNotEqual(updated["ppo"]["state"],updated["myopic"]["state"])
        import json
        restored=json.loads(json.dumps(updated))
        a,_=r.update(torch,updated,data,design,100);b,_=r.update(torch,restored,data,design,100)
        self.assertEqual(a,b)
        # Current p=.9 versus old p=.5, A=+1: clipping binds at 1.2.
        with torch.no_grad():net["actor"][2].bias.fill_(np.log(9.))
        batch=dict(x=torch.zeros((1,128),dtype=torch.float64),eligible=torch.tensor([True]),action=torch.tensor([1.]),
            log_prob=torch.tensor([-np.log(2.)],dtype=torch.float64),advantage=torch.tensor([1.]),label=torch.tensor([0.]),
            **{"return":torch.zeros(1,dtype=torch.float64)})
        loss,metrics=r.objective(torch,net,batch,dict(design,value_coefficient=0.,entropy_coefficient=0.),"ppo")
        self.assertAlmostEqual(float(loss),-1.2,places=10);self.assertEqual(float(metrics["clip_fraction"]),1.)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path=Path(temp)/"trace.npz";c.save(path,dict(x=x),dict(fold=0))
            with self.assertRaises(ValueError):c.read_cache(path,dict(fold=1))
            with self.assertRaises(ValueError):c.save(path,dict(x=x),dict(fold=0))
            path.write_bytes(path.read_bytes()+b"tamper")
            with self.assertRaises(ValueError):c.read_cache(path,dict(fold=0))


@unittest.skipUnless(torch is not None and cv2 is not None,"Real Torch/OpenCV required; skips do not validate")
class Workflow(unittest.TestCase):
    def test_real_live_PPO_matched_updates_resumption_GT_isolation_and_frozen_controls(self):
        from test_stage70_history import Workflow as Previous
        import stage70_history_summary as s70
        from stage48_stability_runtime import digest
        saved_summary=s70.summarize;saved_bridge=e.e70.source_bridge;captured={}
        def capture(*args,**kwargs):
            captured["observer"]=args[3];return saved_bridge(*args,**kwargs)
        def extend(root70):
            original=saved_summary(root70)
            if captured.get("done"):return original
            captured["done"]=True;observer=captured["observer"]
            real_capture=type(observer).capture.__get__(observer,type(observer))
            root71=Path(root70).parent/"stage71";before={str(p):p.read_bytes() for p in Path(root70).parent.rglob("*") if p.is_file()}
            frozen=digest(observer.rt.net)
            with patch.dict("sys.modules",torch=torch,cv2=cv2),patch.object(observer,"capture",wraps=real_capture) as calls:
                e.prepare(root70,root71,dict(device="NVIDIA L40",gpu_uuid="stage70-cpu-fixture"))
                plan,source=c.load_plan(root71)
                self.assertEqual(plan["training_search_calls"],128);self.assertEqual(plan["formal_eval_calls"],352)
                e.source_bridge(root71,plan,source,observer,plan["runtime_identity"]);self.assertEqual(calls.call_count,8)
                calls.reset_mock();e.train(root71,plan,source,observer,plan["runtime_identity"],True);self.assertEqual(calls.call_count,16)
                calls.reset_mock()
                # Simulate disconnect after cache commit but before its optimizer checkpoint.
                original_update=r.update
                with patch.object(r,"update",side_effect=RuntimeError("interrupted before checkpoint")),self.assertRaisesRegex(RuntimeError,"interrupted"):
                    e.train(root71,plan,source,observer,plan["runtime_identity"])
                self.assertEqual(calls.call_count,16)
                calls.reset_mock();models=e.train(root71,plan,source,observer,plan["runtime_identity"]);self.assertEqual(calls.call_count,96)
                self.assertEqual(models["environment_steps"],32);self.assertEqual(len(models["controllers"]),4)
                job=plan["jobs"][0];entry=c.entry_for(plan,job["episodes"][0]["sequence"])
                init=c.sealed(c.step_path(root71,job,0));committed=c.read_cache(c.trace_path(root71,job,0),e.train_meta(plan,job,0,init))
                unit=dict(sequence=entry["sequence"],origin="native",fold=job["fold"],seed=job["seed"],policy="ppo")
                # Future GT changes reward/labels, never the same RNG's actor outputs or histories.
                changed=copy.deepcopy(entry);changed["ground_truth"][1:]=[[99.,99.,1.,1.]]*4
                replay=e.run_actor(observer,changed,unit,plan,source,init["packs"]["ppo"],np.random.default_rng(job["seed"]+100003*job["fold"]),fit=True)
                for k in e.FIELDS:np.testing.assert_array_equal(replay[k],committed[k])
                self.assertFalse(np.array_equal(replay["reward"],committed["reward"]))
                # Matched control consumes the exact stored PPO state matrix, not its own history or held-out labels.
                after=c.sealed(c.step_path(root71,job,1))
                recomputed,stats=original_update(torch,init["packs"],committed,plan["design"],job["seed"])
                self.assertEqual(recomputed,after["packs"])
                changed_plan=copy.deepcopy(plan)
                split=next(s for s in plan["folds"] if s["fold"]==job["fold"])
                for other in changed_plan["entries"]:
                    if other["sequence"] in split["held_out"]:other["ground_truth"]=object()
                replay_fit=e.run_actor(observer,entry,unit,changed_plan,source,init["packs"]["ppo"],np.random.default_rng(job["seed"]+100003*job["fold"]),fit=True)
                other,_=original_update(torch,init["packs"],replay_fit,plan["design"],job["seed"])
                self.assertEqual(other,after["packs"])
                calls.reset_mock();e.evaluate(root71,plan,source,observer,plan["runtime_identity"],True);self.assertEqual(calls.call_count,88)
                calls.reset_mock();e.evaluate(root71,plan,source,observer,plan["runtime_identity"]);self.assertEqual(calls.call_count,264)
                unit=next(u for u in plan["units"] if u["policy"]=="ppo")
                changed=copy.deepcopy(c.entry_for(plan,unit["sequence"]));changed["ground_truth"][1:]=[[99.,99.,1.,1.]]*4
                replay=e.run_actor(observer,changed,unit,plan,source,e.learned_pack(models,unit))
                trace=c.read_cache(c.eval_path(root71,unit),e.eval_meta(plan,unit,models))
                for k in e.FIELDS:np.testing.assert_array_equal(trace[k],replay[k])
            with patch.object(observer,"capture",side_effect=AssertionError("No new inference on committed resume")),patch.object(r,"update",side_effect=AssertionError("No updates after fixed budget")):
                with patch.dict("sys.modules",torch=torch,cv2=cv2):
                    e.source_bridge(root71,plan,source,observer,plan["runtime_identity"])
                    self.assertEqual(models,e.train(root71,plan,source,observer,plan["runtime_identity"]))
                with patch.dict("sys.modules",torch=None,cv2=None):
                    e.evaluate(root71,plan,source,observer,plan["runtime_identity"],True)
                    e.evaluate(root71,plan,source,observer,plan["runtime_identity"])
                    result=s.summarize(root71);self.assertEqual(result,s.summarize(root71))
            self.assertEqual(result["total_new_search_calls"],488);self.assertTrue(result["reinforcement_learning_performed"])
            self.assertFalse(result["sgla_training_performed"]);self.assertEqual(result["full_evaluation_trajectories"],22)
            self.assertEqual(digest(observer.rt.net),frozen)
            self.assertTrue(all(Path(path).read_bytes()==blob for path,blob in before.items()))
            path=c.step_path(root71,job,1);blob=path.read_bytes();path.write_bytes(blob+b"tamper")
            with self.assertRaises(ValueError):e.load_models(root71,plan,source)
            path.write_bytes(blob)
            path=c.trace_path(root71,job,0);blob=path.read_bytes();path.write_bytes(blob+b"tamper")
            with self.assertRaises(ValueError):e.load_models(root71,plan,source)
            path.write_bytes(blob)
            path=Path(root70)/"analysis/stage70_scope.csv";blob=path.read_bytes();path.write_bytes(blob+b"tamper")
            with self.assertRaises(ValueError):c.load_source(root70)
            path.write_bytes(blob)
            path=root71/"analysis/stage71_scope.csv";blob=path.read_bytes();path.write_bytes(blob+b"tamper")
            with self.assertRaises(ValueError):s.summarize(root71)
            path.write_bytes(blob)
            return original
        with patch.dict(c.DESIGN,passes=2,epochs=2,minibatch=2,hidden=4,smoke_frames=2,bootstrap_samples=30), \
             patch.object(e.e70,"source_bridge",side_effect=capture),patch.object(s70,"summarize",side_effect=extend):
            Previous().test_real_fit_histories_pairs_controls_GT_isolation_resume_and_frozen_sources()
        self.assertTrue(captured.get("done"))


if __name__=="__main__":unittest.main()
