"""Separate norm mechanics, unchanged objectives, durable on-policy training and full source replays."""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import stage73_ppo_common as c
import stage73_ppo_runtime as r
import stage73_ppo_experiment as e
import stage73_ppo_summary as s


class Semantics(unittest.TestCase):
    def test_production_budget_folds_and_single_treatment_constraint(self):
        names=["v"+str(i) for i in range(12)];frames=[1127]*11+[1128]
        old=dict(entries=[dict(sequence=n,frames=list(range(k+1))) for n,k in zip(names,frames)],
            folds=[dict(fold=0,fit=names[:6],held_out=names[6:]),dict(fold=1,fit=names[6:],held_out=names[:6])],
            runtime_identity=dict(tracker=dict(device="NVIDIA L40"),gpu_uuid="old"),root="unused",search_size=256,
            search_factor=4.,window=[1.],actor_design={},controller_design={})
        base_source=dict(plan=old,root=Path("unused"),report=dict(fingerprint="r"),models=dict(fingerprint="m"),bindings={})
        with patch.object(c.base,"load_source",return_value=base_source):old_plan,_=c.base.make_plan("unused",dict(device="NVIDIA L40",gpu_uuid="old"))
        source=dict(stage71=dict(plan=old_plan,report=dict(fingerprint="r71"),models=dict(fingerprint="m71")),diagnostic_report=dict(fingerprint="r72"),bindings={})
        with patch.object(c,"load_source",return_value=source):
            plan,_=c.make_plan("unused",dict(device="NVIDIA L40",gpu_uuid="new"))
            with patch.dict(c.DESIGN,learning_rate=.1),self.assertRaisesRegex(ValueError,"Only actor/critic clipping"):
                c.make_plan("unused",dict(device="NVIDIA L40",gpu_uuid="new"))
        self.assertEqual(plan["jobs"],old_plan["jobs"])
        self.assertEqual(plan["training_environment_steps"],108200);self.assertEqual(plan["training_search_calls"],432800)
        self.assertEqual(plan["optimizer_updates_per_arm"],old_plan["optimizer_updates_per_arm"])
        self.assertEqual(len(plan["units"]),204);self.assertEqual(plan["formal_eval_calls"],919700)
        self.assertEqual(plan["eval_smoke_calls_reused"],2176);self.assertEqual(plan["source_bridge_search_calls"],8)
        self.assertEqual(plan["training_search_calls"]+plan["formal_eval_calls"]+plan["source_bridge_search_calls"],1352508)
        self.assertEqual(sum(u["policy"] in c.CONTROLS+c.REUSED for u in plan["units"]),132)
        self.assertEqual(sum(u["policy"] in c.LEARNED for u in plan["units"]),72)
        for unit in plan["units"]:self.assertIn(unit["sequence"],next(f for f in plan["folds"] if f["fold"]==unit["fold"])["held_out"])
        with self.assertRaisesRegex(ValueError,"held-out"):c.check_fit(plan,plan["jobs"][0],names[-1])
        self.assertEqual(c.source_unit(dict(policy="joint_rate",seed=2171)),dict(policy="ppo_rate",seed=2171))
        with self.assertRaisesRegex(ValueError,"separate"):e.prepare("unused","unused",{})


try:
    import torch
    import cv2
except ImportError:torch=cv2=None


@unittest.skipUnless(torch is not None,"Real CPU Torch required; skips do not validate")
class TorchMath(unittest.TestCase):
    def test_branch_clipping_not_joint_clipping_and_same_loss(self):
        torch.set_num_threads(1)
        design=dict(c.DESIGN,hidden=4,epochs=2,minibatch=2)
        norm=dict(center=[0.]*128,scale=[1.]*128)
        cpu_rng=torch.random.get_rng_state().clone()
        with patch.object(torch,"manual_seed",side_effect=AssertionError("No CUDA reseeding")),patch.object(torch.cuda,"manual_seed_all",side_effect=AssertionError("No CUDA reseeding")):
            pack=r.initial(torch,2171,norm,design)
        self.assertTrue(torch.equal(cpu_rng,torch.random.get_rng_state()))
        self.assertEqual(pack,r.prior.initial(torch,2171,norm,design))
        separate=r.restore(torch,pack["state"],design);joint=r.restore(torch,pack["state"],design)
        for net in (separate,joint):
            for param in net.parameters():param.grad=torch.zeros_like(param)
            net["actor"][2].bias.grad.fill_(3.)
            net["critic"][2].bias.grad.fill_(40.)
        diagnostics=r.clip_branches(torch,separate,design)
        torch.nn.utils.clip_grad_norm_(joint.parameters(),1.)
        self.assertAlmostEqual(diagnostics["actor_gradient"],3.);self.assertAlmostEqual(diagnostics["critic_gradient"],40.)
        self.assertLessEqual(diagnostics["actor_post_clip_gradient"],1.);self.assertLessEqual(diagnostics["critic_post_clip_gradient"],1.)
        self.assertGreater(float(separate["actor"][2].bias.grad),.999)
        self.assertLess(float(joint["actor"][2].bias.grad),.075)
        # Only the clipping operation changes. Losses, architecture, initialization and Adam remain shared code.
        self.assertIs(r.objective,r.prior.objective);self.assertIs(r.optimizer,r.prior.optimizer)
        x=np.random.default_rng(9).normal(size=(4,128))
        data=dict(x=x,eligible=np.array([True,True,False,True]),action=np.array([1,0,0,1]),
            log_prob=np.array([-np.log(2),-np.log(2),0.,-np.log(2)]),reward=np.array([0.,0.,.2,1.]),value=np.zeros(4),myopic_label=np.array([0.,1.,0.,0.]))
        packs=dict(ppo=pack,myopic=copy.deepcopy(pack))
        updated,stats=r.update(torch,packs,data,design,99)
        self.assertEqual(stats["ppo"]["updates"],4);self.assertEqual(stats["myopic"]["updates"],4)
        for arm in packs:
            self.assertGreater(stats[arm]["maximum_actor_gradient"],0.)
            self.assertLessEqual(stats[arm]["actor_post_clip_gradient"],1.)
            self.assertLessEqual(stats[arm]["critic_post_clip_gradient"],1.)
            net=r.restore(torch,updated[arm]["state"],design)
            np.testing.assert_allclose(c.forward(x,updated[arm]["state"],"actor"),net["actor"](torch.tensor(x)).detach().numpy()[:,0],atol=1e-12,rtol=0)
        persisted=json.loads(json.dumps(updated))
        uninterrupted,_=r.update(torch,updated,data,design,100)
        resumed,_=r.update(torch,persisted,data,design,100)
        self.assertEqual(uninterrupted,resumed)


@unittest.skipUnless(torch is not None and cv2 is not None,"Real CENTER source fixture requires Torch/OpenCV; skips do not validate")
class Workflow(unittest.TestCase):
    def test_full_source_to_split_clip_resume_and_exact_old_policies(self):
        from test_stage71_ppo import Workflow as Previous
        import stage71_ppo_summary as s71
        import stage72_ppo_audit as a72
        from stage48_stability_runtime import digest
        original=s71.summarize;saved_eval=e.source_experiment.evaluate;captured={}
        def capture(*args,**kwargs):
            captured["observer"]=args[3];return saved_eval(*args,**kwargs)
        def extend(root71):
            report=original(root71)
            if captured.get("done"):return report
            captured["done"]=True;root71=Path(root71);root72=root71.parent/"stage72";root73=root71.parent/"stage73"
            observer=captured["observer"];p71,_=c.base.load_plan(root71)
            seq=p71["entries"][0]["sequence"]
            cases=(dict(case_id="micro",sequence=seq,policy="ppo",seed=p71["design"]["seeds"][0],selector="fixed",frame=2),)
            design=dict(p71["design"],gradient_clip_mode="separate_actor_critic")
            with patch.object(c.diagnostic,"CONFIG",cases),patch.dict(c.diagnostic.DESIGN,before=1,after=1),patch.dict(c.DESIGN,design,clear=True):
                a72.summarize(root71,root72)
                before={str(p):p.read_bytes() for p in root71.parent.rglob("*") if p.is_file()}
                frozen=digest(observer.rt.net)
                real_capture=type(observer).capture.__get__(observer,type(observer))
                with patch.object(observer,"capture",wraps=real_capture) as calls:
                    e.prepare(root72,root73,dict(device="NVIDIA L40",gpu_uuid="stage73-cpu-fixture"))
                    plan,source=c.load_plan(root73)
                    self.assertEqual(plan["training_search_calls"],128);self.assertEqual(plan["formal_eval_calls"],544)
                    e.source_bridge(root73,plan,source,observer,plan["runtime_identity"]);self.assertEqual(calls.call_count,8)
                    calls.reset_mock();e.train(root73,plan,source,observer,plan["runtime_identity"],True);self.assertEqual(calls.call_count,16)
                    calls.reset_mock()
                    with patch.object(r,"update",side_effect=RuntimeError("interrupted before split checkpoint")),self.assertRaisesRegex(RuntimeError,"interrupted"):
                        e.train(root73,plan,source,observer,plan["runtime_identity"])
                    self.assertEqual(calls.call_count,16)
                    calls.reset_mock();models=e.train(root73,plan,source,observer,plan["runtime_identity"]);self.assertEqual(calls.call_count,96)
                    self.assertEqual(models["environment_steps"],32);self.assertEqual(len(models["controllers"]),4)
                    job=plan["jobs"][0];entry=c.entry_for(plan,job["episodes"][0]["sequence"])
                    init=c.sealed(c.step_path(root73,job,0));committed=c.read_cache(c.trace_path(root73,job,0),e.train_meta(plan,job,0,init))
                    original_init=c.base.sealed(c.base.step_path(root71,job,0));self.assertEqual(init["packs"],original_init["packs"])
                    after=c.sealed(c.step_path(root73,job,1))
                    recomputed,_=r.update(torch,init["packs"],committed,plan["design"],job["seed"])
                    self.assertEqual(recomputed,after["packs"])
                    unit=dict(sequence=entry["sequence"],origin="native",fold=job["fold"],seed=job["seed"],policy="ppo")
                    changed=copy.deepcopy(entry);changed["ground_truth"][1:]=[[99.,99.,1.,1.]]*4
                    replay=e.run_actor(observer,changed,unit,plan,source,init["packs"]["ppo"],np.random.default_rng(job["seed"]+100003*job["fold"]),fit=True)
                    for k in e.FIELDS:np.testing.assert_array_equal(replay[k],committed[k])
                    self.assertFalse(np.array_equal(replay["reward"],committed["reward"]))
                    changed_plan=copy.deepcopy(plan)
                    held=next(f["held_out"] for f in plan["folds"] if f["fold"]==job["fold"])
                    for item in changed_plan["entries"]:
                        if item["sequence"] in held:item["ground_truth"]=object()
                    replay=e.run_actor(observer,entry,unit,changed_plan,source,init["packs"]["ppo"],np.random.default_rng(job["seed"]+100003*job["fold"]),fit=True)
                    other,_=r.update(torch,init["packs"],replay,plan["design"],job["seed"]);self.assertEqual(other,after["packs"])
                    calls.reset_mock();e.evaluate(root73,plan,source,observer,plan["runtime_identity"],True);self.assertEqual(calls.call_count,136)
                    calls.reset_mock();e.evaluate(root73,plan,source,observer,plan["runtime_identity"]);self.assertEqual(calls.call_count,408)
                    learned=next(u for u in plan["units"] if u["policy"]=="ppo")
                    changed=copy.deepcopy(c.entry_for(plan,learned["sequence"]));changed["ground_truth"][1:]=[[99.,99.,1.,1.]]*4
                    replay=e.run_actor(observer,changed,learned,plan,source,e.learned_pack(models,learned,source))
                    actual=c.read_cache(c.eval_path(root73,learned),e.eval_meta(plan,learned,models))
                    for k in e.FIELDS:np.testing.assert_array_equal(replay[k],actual[k])
                with patch.object(observer,"capture",side_effect=AssertionError("No new inference on committed resume")),patch.object(r,"update",side_effect=AssertionError("No new optimization on resume")):
                    self.assertEqual(models,e.train(root73,plan,source,observer,plan["runtime_identity"]))
                    with patch.dict("sys.modules",torch=None,cv2=None):
                        e.source_bridge(root73,plan,source,observer,plan["runtime_identity"])
                        e.evaluate(root73,plan,source,observer,plan["runtime_identity"],True)
                        e.evaluate(root73,plan,source,observer,plan["runtime_identity"])
                        result=s.summarize(root73);self.assertEqual(result,s.summarize(root73))
                self.assertEqual(result["total_new_search_calls"],680);self.assertEqual(result["full_evaluation_trajectories"],34)
                self.assertEqual(result["source_evaluation_trajectories_retested"],22);self.assertEqual(result["new_policy_evaluation_trajectories"],12)
                self.assertTrue(result["all_source_policies_exactly_reproduced"]);self.assertFalse(result["sgla_training_performed"])
                def read(name):
                    with (root73/"analysis"/("stage73_"+name+".csv")).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
                self.assertEqual(len(read("policy_summary")),11);self.assertEqual(len(read("paired")),44)
                self.assertEqual(len(read("training_episode")),32);self.assertEqual(len(read("training_summary")),32)
                opt=read("optimization_summary");self.assertEqual(len(opt),16)
                self.assertTrue(all(x["actor_post_clip_gradient"]=="" for x in opt if x["condition"]=="joint"))
                self.assertTrue(all(0<=float(x["actor_post_clip_gradient"])<=1 for x in opt if x["condition"]=="separate"))
                video={(x["sequence"],x["policy"]):x for x in read("video_summary")}
                for influence in read("video_influence"):
                    n=influence["sequence"]
                    delta=float(video[n,"ppo"]["auc"])-float(video[n,"joint_ppo"]["auc"])
                    self.assertAlmostEqual(float(influence["joint_ppo_auc_gain"]),delta)
                    others=[float(x["auc"])-float(video[seq,"joint_ppo"]["auc"]) for (seq,policy),x in video.items() if policy=="ppo" and seq!=n]
                    self.assertAlmostEqual(float(influence["primary_leave_one_out"]),float(np.mean(others)))
                self.assertTrue(all(Path(path).read_bytes()==blob for path,blob in before.items()))
                self.assertEqual(digest(observer.rt.net),frozen)
                for path,operation in ((root72/"analysis/stage72_scope.csv",lambda:c.load_source(root72)),
                    (c.base.step_path(root71,p71["jobs"][0],1),lambda:c.load_source(root72)),
                    (c.step_path(root73,job,1),lambda:e.load_models(root73,plan,source)),
                    (c.trace_path(root73,job,0),lambda:e.load_models(root73,plan,source)),
                    (root73/"analysis/stage73_scope.csv",lambda:s.summarize(root73))):
                    blob=path.read_bytes();path.write_bytes(blob+b"tamper")
                    try:
                        with self.assertRaises(ValueError):operation()
                    finally:path.write_bytes(blob)
                with self.assertRaisesRegex(ValueError,"separate"):e.prepare(root72,root72,{})
            return report
        def enabled_extend(root71):
            # The preceding stage invokes summary with Torch/OpenCV imports disabled.
            # Re-enable the actual CPU fixture for this NEW live experiment only.
            with patch.dict("sys.modules",torch=torch,cv2=cv2):return extend(root71)
        with patch.object(e.source_experiment,"evaluate",side_effect=capture),patch.object(s71,"summarize",side_effect=enabled_extend):
            Previous().test_real_live_PPO_matched_updates_resumption_GT_isolation_and_frozen_controls()
        self.assertTrue(captured.get("done"))


if __name__=="__main__":unittest.main()
