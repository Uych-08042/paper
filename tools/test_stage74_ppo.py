"""Direct four-action legality, PPO/CE objectives and real frozen CENTER integration."""
import copy
import csv
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import stage74_ppo_common as c
import stage74_ppo_runtime as r
import stage74_ppo_experiment as e
import stage74_ppo_summary as s


class Semantics(unittest.TestCase):
    def test_masked_actions_ties_and_direct_access_outside_Q(self):
        boxes=np.array([[0.,0.,4.,4.],[1.,0.,4.,4.],[1.,0.,4.,4.],[3.,0.,4.,4.]])
        mask=c.action_mask(boxes);np.testing.assert_array_equal(mask,[1,1,0,1])
        p,lp=c.distribution(np.zeros(4),mask);np.testing.assert_allclose(p,[1/3,1/3,0.,1/3])
        self.assertTrue(np.isneginf(lp[2]));self.assertEqual(c.best_label([.1,.7,.7,.3],mask,1e-6),1)
        self.assertEqual(c.best_label([.7,.70000001,.1,.2],mask,1e-6),0)
        with self.assertRaises(ValueError):c.log_probability(2,np.zeros(4),mask)
        with self.assertRaises(ValueError):c.distribution([np.nan,0.,0.,0.],mask)
        allsame=np.repeat(boxes[:1],4,axis=0);p,_=c.distribution(np.zeros(4),c.action_mask(allsame));np.testing.assert_array_equal(p,[1,0,0,0])
        # Q's proposal does not restrict the new action. Fixed actor may select q2 even when Q says q1.
        design=dict(c.DESIGN,hidden=2);state={}
        for name,out in (("actor",4),("critic",1)):
            state.update({name+'.0.weight':np.zeros((2,256)).tolist(),name+'.0.bias':[0.,0.],name+'.2.weight':np.zeros((out,2)).tolist(),name+'.2.bias':([0.,0.,3.,0.] if out==4 else [0.])})
        pack=dict(state=state,norm=dict(center=[0.]*256,scale=[1.]*256))
        row=dict(features=np.zeros((4,24)),candidate_boxes=np.array([[i,0.,4.,4.] for i in range(4)]),candidate_parents=np.repeat(boxes[:1],4,axis=0))
        unit=dict(policy='ppo',fold=0)
        with patch.object(c,'fit_pack',return_value={}),patch.object(c.c66,'decide',return_value=dict(quality_choice=1)):
            selected=e.choose(copy.deepcopy(row),[boxes[0]],unit,dict(design=design,controller_design={}),{},pack)
        self.assertEqual(selected['selected'],2);self.assertEqual(selected['quality_choice'],1)
        self.assertEqual(selected['x'].shape,(256,))
        with patch.object(c,'fit_pack',return_value={}),patch.object(c.c66,'decide',return_value=dict(quality_choice=3)):
            changed=e.choose(copy.deepcopy(row),[boxes[0]],unit,dict(design=design,controller_design={}),{},pack)
        np.testing.assert_array_equal(selected['probabilities'],changed['probabilities'])

    def test_production_budget_and_frozen_source_design(self):
        names=['v'+str(i) for i in range(12)];lengths=[1127]*11+[1128]
        folds=[dict(fold=0,fit=names[:6],held_out=names[6:]),dict(fold=1,fit=names[6:],held_out=names[:6])]
        jobs=[dict(fold=f['fold'],seed=seed,episodes=[dict(sequence=n,pass_index=p) for p in range(4) for n in f['fit']]) for f in folds for seed in (2171,2172)]
        old=dict(design=dict(c.previous.DESIGN),entries=[dict(sequence=n,frames=list(range(k+1))) for n,k in zip(names,lengths)],folds=folds,jobs=jobs,
            runtime_identity=dict(tracker=dict(device='NVIDIA L40'),gpu_uuid='old'),training_environment_steps=108200,training_search_calls=432800,
            optimizer_updates_per_arm=1920,source_bridge_search_calls=8,train_smoke_calls_reused=4508,controller_design={},root='unused',search_size=256,search_factor=4.,window=[1.],actor_design={})
        source=dict(stage73=dict(plan=old,report=dict(fingerprint='r'),models=dict(fingerprint='m')),bindings={})
        with patch.object(c,'load_source',return_value=source):
            plan,_=c.make_plan('unused',dict(device='NVIDIA L40',gpu_uuid='new'))
            with patch.dict(c.DESIGN,passes=5),self.assertRaisesRegex(ValueError,'Keep source optimizer'):
                c.make_plan('unused',dict(device='NVIDIA L40',gpu_uuid='new'))
        self.assertEqual(plan['jobs'],jobs);self.assertEqual(len(plan['units']),144)
        self.assertEqual(plan['training_search_calls'],432800);self.assertEqual(plan['formal_eval_calls'],649200)
        self.assertEqual(plan['eval_smoke_calls_reused'],1536)
        self.assertEqual(plan['training_search_calls']+plan['formal_eval_calls']+8,1082008)
        self.assertEqual(sum(u['policy'] in c.CONTROLS+c.REUSED for u in plan['units']),96)
        self.assertEqual(sum(u['policy'] in c.LEARNED for u in plan['units']),48)
        for u in plan['units']:self.assertIn(u['sequence'],next(f for f in folds if f['fold']==u['fold'])['held_out'])
        with self.assertRaises(ValueError):c.check_fit(plan,jobs[0],names[-1])
        with self.assertRaisesRegex(ValueError,'separate'):e.prepare('unused','unused',{})


try:
    import torch
    import cv2
except ImportError:torch=cv2=None


@unittest.skipUnless(torch is not None,'Actual CPU Torch required; skips do not validate')
class TorchMath(unittest.TestCase):
    def test_categorical_clipping_CE_forced_action_and_Adam_persistence(self):
        torch.set_num_threads(1);design=dict(c.DESIGN,hidden=4,epochs=2,minibatch=2)
        norm=dict(center=[0.]*256,scale=[1.]*256);saved_rng=torch.random.get_rng_state().clone()
        with patch.object(torch,'manual_seed',side_effect=AssertionError('Do not seed CUDA')),patch.object(torch.cuda,'manual_seed_all',side_effect=AssertionError('Do not seed CUDA')):
            pack=r.initial(torch,2171,norm,design)
        self.assertTrue(torch.equal(saved_rng,torch.random.get_rng_state()))
        net=r.restore(torch,pack['state'],design);x=np.random.default_rng(9).normal(size=(4,256))
        for branch in ('actor','critic'):
            expected=net[branch](torch.tensor(x)).detach().numpy()
            np.testing.assert_allclose(c.forward(x,pack['state'],branch),expected if branch=='actor' else expected[:,0],atol=1e-12,rtol=0)
        masks=np.array([[1,1,1,1],[1,1,0,1],[1,0,0,0],[1,1,1,1]],bool)
        actions=np.array([2,1,0,3]);lps=np.array([c.log_probability(int(a),np.zeros(4),m) for a,m in zip(actions,masks)])
        data=dict(x=x,action_mask=masks,eligible=masks.sum(1)>1,action=actions,log_prob=lps,reward=np.array([0.,.3,.2,1.]),value=np.zeros(4),myopic_label=np.array([3,3,0,1]))
        packs=dict(ppo=pack,myopic=copy.deepcopy(pack));updated,stats=r.update(torch,packs,data,design,99)
        for arm in packs:
            self.assertEqual(stats[arm]['updates'],4);self.assertGreater(stats[arm]['maximum_actor_gradient'],0.)
            self.assertLessEqual(stats[arm]['actor_post_clip_gradient'],1.);self.assertLessEqual(stats[arm]['critic_post_clip_gradient'],1.)
        self.assertNotEqual(updated['ppo']['state'],updated['myopic']['state'])
        restored=json.loads(json.dumps(updated));left,_=r.update(torch,updated,data,design,100);right,_=r.update(torch,restored,data,design,100);self.assertEqual(left,right)
        one=dict(x=torch.zeros((1,256),dtype=torch.float64),action_mask=torch.ones((1,4),dtype=torch.bool),action=torch.tensor([0.]),
            log_prob=torch.tensor([-np.log(4.)],dtype=torch.float64),advantage=torch.ones(1,dtype=torch.float64),label=torch.tensor([2.]),**{'return':torch.zeros(1,dtype=torch.float64)})
        simple=dict(design,value_coefficient=0.,entropy_coefficient=0.)
        loss,_=r.objective(torch,net,one,simple,'myopic');self.assertAlmostEqual(float(loss),np.log(4.),places=12)
        with torch.no_grad():net['actor'][2].bias[0]=np.log(9.)
        loss,metrics=r.objective(torch,net,one,simple,'ppo');self.assertAlmostEqual(float(loss),-1.2,places=12);self.assertEqual(float(metrics['clip_fraction']),1.)
        one['action_mask']=torch.tensor([[True,False,False,False]]);one['label']=torch.tensor([0.])
        for arm in ('ppo','myopic'):
            loss,_=r.objective(torch,net,one,simple,arm);net.zero_grad();loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()))
            self.assertAlmostEqual(float(loss),0.)


@unittest.skipUnless(torch is not None and cv2 is not None,'Actual CENTER source fixture required; skips do not validate')
class Workflow(unittest.TestCase):
    def test_complete_source_direct_rollouts_no_GT_leak_resume_and_immutability(self):
        from test_stage73_ppo import Workflow as Previous
        import stage73_ppo_summary as s73
        from stage48_stability_runtime import digest
        original=s73.summarize;saved_eval=e.source_experiment.evaluate;captured={}
        def capture(*args,**kwargs):captured['observer']=args[3];return saved_eval(*args,**kwargs)
        def extend(root73):
            report=original(root73)
            if captured.get('done'):return report
            captured['done']=True;root73=Path(root73);root74=root73.parent/'stage74';observer=captured['observer']
            p73,_=c.previous.load_plan(root73);design=dict(p73['design'],input_dim=256,actions=4)
            before={str(p):p.read_bytes() for p in root73.parent.rglob('*') if p.is_file()};frozen=digest(observer.rt.net)
            actual_capture=type(observer).capture.__get__(observer,type(observer))
            with patch.dict(c.DESIGN,design,clear=True),patch.object(observer,'capture',wraps=actual_capture) as calls:
                e.prepare(root73,root74,dict(device='NVIDIA L40',gpu_uuid='stage74-cpu-fixture'))
                plan,source=c.load_plan(root74);self.assertEqual(plan['training_search_calls'],128);self.assertEqual(plan['formal_eval_calls'],384)
                e.source_bridge(root74,plan,source,observer,plan['runtime_identity']);self.assertEqual(calls.call_count,8)
                calls.reset_mock();e.train(root74,plan,source,observer,plan['runtime_identity'],True);self.assertEqual(calls.call_count,16)
                calls.reset_mock()
                with patch.object(r,'update',side_effect=RuntimeError('interrupted categorical update')),self.assertRaisesRegex(RuntimeError,'interrupted'):
                    e.train(root74,plan,source,observer,plan['runtime_identity'])
                self.assertEqual(calls.call_count,16)
                calls.reset_mock();models=e.train(root74,plan,source,observer,plan['runtime_identity']);self.assertEqual(calls.call_count,96)
                job=plan['jobs'][0];entry=c.entry_for(plan,job['episodes'][0]['sequence']);initial=c.sealed(c.step_path(root74,job,0))
                cache=c.read_cache(c.trace_path(root74,job,0),e.train_meta(plan,job,0,initial));after=c.sealed(c.step_path(root74,job,1))
                recomputed,_=r.update(torch,initial['packs'],cache,plan['design'],job['seed']);self.assertEqual(recomputed,after['packs'])
                self.assertEqual(initial['packs']['ppo'],initial['packs']['myopic']);self.assertEqual(len(initial['packs']['ppo']['norm']['center']),256)
                unit=dict(sequence=entry['sequence'],origin='native',fold=job['fold'],seed=job['seed'],policy='ppo')
                changed=copy.deepcopy(entry);changed['ground_truth'][1:]=[[99.,99.,1.,1.]]*4
                replay=e.run_actor(observer,changed,unit,plan,source,initial['packs']['ppo'],np.random.default_rng(job['seed']+100003*job['fold']),fit=True)
                for k in e.FIELDS:np.testing.assert_array_equal(replay[k],cache[k])
                self.assertFalse(np.array_equal(replay['reward'],cache['reward']))
                changed_plan=copy.deepcopy(plan);split=next(f for f in plan['folds'] if f['fold']==job['fold'])
                for item in changed_plan['entries']:
                    if item['sequence'] in split['held_out']:item['ground_truth']=object()
                self.assertEqual(c.fit_normalization(changed_plan,source,job['fold']),initial['packs']['ppo']['norm'])
                replay=e.run_actor(observer,entry,unit,changed_plan,source,initial['packs']['ppo'],np.random.default_rng(job['seed']+100003*job['fold']),fit=True)
                result,_=r.update(torch,initial['packs'],replay,plan['design'],job['seed']);self.assertEqual(result,after['packs'])
                calls.reset_mock();e.evaluate(root74,plan,source,observer,plan['runtime_identity'],True);self.assertEqual(calls.call_count,96)
                calls.reset_mock();e.evaluate(root74,plan,source,observer,plan['runtime_identity']);self.assertEqual(calls.call_count,288)
                learned=next(u for u in plan['units'] if u['policy']=='ppo');entry=c.entry_for(plan,learned['sequence']);changed=copy.deepcopy(entry)
                changed['ground_truth'][1:]=[[99.,99.,1.,1.]]*4
                replay=e.run_actor(observer,changed,learned,plan,source,e.learned_pack(models,learned,source))
                cache=c.read_cache(c.eval_path(root74,learned),e.eval_meta(plan,learned,models))
                for k in e.FIELDS:np.testing.assert_array_equal(cache[k],replay[k])
            with patch.dict(c.DESIGN,design,clear=True),patch.object(observer,'capture',side_effect=AssertionError('No repeated inference')),patch.object(r,'update',side_effect=AssertionError('No repeated update')):
                self.assertEqual(e.train(root74,plan,source,observer,plan['runtime_identity']),models)
                with patch.dict('sys.modules',torch=None,cv2=None):
                    e.source_bridge(root74,plan,source,observer,plan['runtime_identity'])
                    e.evaluate(root74,plan,source,observer,plan['runtime_identity'],True);e.evaluate(root74,plan,source,observer,plan['runtime_identity'])
                    result=s.summarize(root74);self.assertEqual(result,s.summarize(root74))
                self.assertEqual(result['total_new_search_calls'],520);self.assertEqual(result['full_evaluation_trajectories'],24)
                self.assertEqual(result['source_evaluation_trajectories_retested'],16);self.assertEqual(result['new_policy_evaluation_trajectories'],8)
                self.assertFalse(result['recovery_branch_present']);self.assertEqual(result['new_recovery_search_calls'],0)
                def read(name):
                    with (root74/'analysis'/('stage74_'+name+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
                self.assertEqual(len(read('policy_summary')),8);self.assertEqual(len(read('paired')),28);self.assertEqual(len(read('candidate_summary')),8)
                self.assertEqual(len(read('training_episode')),16);self.assertEqual(len(read('training_summary')),16);self.assertEqual(len(read('optimization_summary')),8)
                for row in read('policy_summary'):self.assertAlmostEqual(sum(float(row['action_q'+str(q)+'_fraction']) for q in range(4)),1.)
                for row in read('candidate_summary'):self.assertAlmostEqual(float(row['selected_good_fraction'])+float(row['missed_public_good_fraction']),float(row['any_public_good_fraction']))
                video={(x['sequence'],x['policy']):float(x['auc']) for x in read('video_summary')}
                for row in read('video_influence'):
                    n=row['sequence'];other=[video[k,'ppo']-video[k,'myopic'] for k,p in video if p=='ppo' and k!=n]
                    self.assertAlmostEqual(float(row['primary_leave_one_out']),float(np.mean(other)))
                self.assertTrue(all(Path(p).read_bytes()==blob for p,blob in before.items()));self.assertEqual(digest(observer.rt.net),frozen)
                for path,operation in ((root73/'analysis/stage73_scope.csv',lambda:c.load_source(root73)),
                    (c.step_path(root74,job,1),lambda:e.load_models(root74,plan,source)),(c.trace_path(root74,job,0),lambda:e.load_models(root74,plan,source)),
                    (root74/'analysis/stage74_scope.csv',lambda:s.summarize(root74))):
                    blob=path.read_bytes();path.write_bytes(blob+b'tamper')
                    try:
                        with self.assertRaises(ValueError):operation()
                    finally:path.write_bytes(blob)
                with self.assertRaisesRegex(ValueError,'separate'):e.prepare(root73,root73,{})
            return report
        def enabled(root73):
            with patch.dict('sys.modules',torch=torch,cv2=cv2):return extend(root73)
        with patch.object(e.source_experiment,'evaluate',side_effect=capture),patch.object(s73,'summarize',side_effect=enabled):
            Previous().test_full_source_to_split_clip_resume_and_exact_old_policies()
        self.assertTrue(captured.get('done'))


if __name__=='__main__':unittest.main()
