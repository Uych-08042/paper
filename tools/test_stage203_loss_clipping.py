"""Factorial clipping, legacy replay, nonfinite handling and complete CPU tests."""
import contextlib
import csv
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'tools'),str(REPO/'tracking')]
import stage203_loss_clipping as run


class Progress:
    def __init__(self):self.done,self.reused=0,0
    def advance(self,count=1):self.done+=count


class PureTests(unittest.TestCase):
    def test_factorial_effects_have_declared_signs(self):
        policies,errors=[],[]
        levels=dict(point_clip=.2,point_no_clip=.3,regret_clip=.25,regret_no_clip=.55)
        for arm,value in levels.items():
            objective,clipping=run.common.flags(arm)
            self.assertEqual(clipping,arm.endswith('_clip') and not arm.endswith('_no_clip'))
            for pool in ('fit','held'):
                for seed in (2132,2133):
                    for i in range(38):
                        base=dict(arm=arm,pool=pool,seed=seed,group=f'g{i:02d}')
                        for policy in run.POLICIES:
                            policies.append(dict(base,policy=policy,mean_iou=value if policy=='trained' else .1,state_alignment_gain=value/10 if policy=='trained' else 0.))
                        errors.append(dict(base,predictor='trained',interaction_mse=value,within_video_mse=value,
                                           regret_margin_loss=value,gap_weighted_pair_accuracy=value))
        rows=run.common.paired_reports(policies,errors)
        self.assertEqual(len(rows),92)
        interaction=next(r for r in rows if r['pool']=='held' and r['metric']=='mean_iou' and r['kind']=='interaction')
        self.assertAlmostEqual(interaction['mean_delta'],.2)
        with self.assertRaises(ValueError):run.common.flags('regret_unknown')

    def test_reviewed_source202_and_committed_cache_integrity(self):
        d,p,tables=run.load202(run.DEFAULT_BUNDLE)
        self.assertEqual(d['fingerprint'],run.SOURCE202)
        self.assertEqual(p['fingerprint'],run.PLAN202)
        self.assertEqual(len(tables['frame_predictions']),14592)
        sizes={r['group']:48 for r in tables['frame_predictions']}
        for fold in p['folds']:
            for seed in p['seeds']:
                draws=run.common.pair_draws(fold,seed,sizes)
                scopes=[r for r in tables['model_scope'] if int(r['fold'])==fold['fold'] and int(r['seed'])==seed]
                self.assertTrue(all(r['pair_draws_sha256']==run.fingerprint(draws) for r in scopes))
                self.assertEqual(sum(len(run.common.flatten(batch)) for batch in draws),1024)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'x.npz'
            run.cache(path,dict(plan='test'),lambda:dict(value=np.ones(3)))
            _,reused=run.cache(path,dict(plan='test'),lambda:self.fail('Must reuse'))
            self.assertTrue(reused)
            with self.assertRaises(ValueError):run.cache(path,dict(plan='changed'),lambda:{})
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):run.cache(path,dict(plan='test'),lambda:{})
            bundle=Path(tmp)/'changed.txt';bundle.write_bytes(b'changed')
            with self.assertRaises(ValueError):run.load202(bundle)

    def test_float64_bound_and_near_argmax_tie(self):
        rng=np.random.default_rng(202)
        y,p=rng.uniform(0,1,(50,6)),rng.normal(size=(50,6))
        terms=run.common.decision_terms(y,p)
        self.assertTrue(np.all(terms['regret']<=terms['bound']+run.metrics.EPS+1e-12))
        np.testing.assert_allclose(run.common.decision_terms(y,y+.37)['bound'],0,atol=1e-15)
        np.testing.assert_allclose(terms['bound'],run.common.decision_terms(y,p+np.arange(50)[:,None])['bound'],atol=1e-14)
        y=np.array([[1.,0.,0.,0.,0.,0.]])
        p=np.array([[.1,0.,.1-5e-9,0.,0.,0.]])
        self.assertEqual(int(run.metrics.choose(p)[0]),2)
        term=run.common.decision_terms(y,p)
        self.assertGreater(term['regret'][0],term['bound'][0])
        self.assertEqual(float(term['violation'][0]),0.)


@unittest.skipUnless(importlib.util.find_spec('torch'),'CPU Torch required')
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch;torch.set_num_threads(1)

    def bank(self):
        groups=[f'uav{i:07d}' for i in range(38)]
        rng=np.random.default_rng(202);bank={}
        for g in groups:
            tokens=rng.normal(size=(2,6,4)).astype(np.float32)
            bank[g]=dict(tokens=tokens,pooled=run.architecture.pooling.pool(self.torch,self.torch.as_tensor(tokens),2).numpy(),
                         labels=rng.uniform(.1,.9,(2,6)),frame_id=np.array([1,2]),sequence=g+'_test')
        return groups,bank

    def test_margin_gradient_and_exact_mse_control(self):
        torch=self.torch
        s=torch.tensor([[0.,2.,-3.,-3.,-3.,-3.]]*2,dtype=torch.float64,requires_grad=True)
        y=torch.tensor([[.9,.1,.2,.3,.4,.5]]*2,dtype=torch.float64)
        terms=run.common.objectives(torch,s,y)
        self.assertAlmostEqual(float(terms['regret_margin'].detach()),2.8)
        grad=torch.autograd.grad(terms['regret_margin'],s,retain_graph=True)[0]
        torch.testing.assert_close(grad,torch.tensor([[-.5,.5,0,0,0,0]]*2,dtype=torch.float64))
        old=run.common.previous.pair_source.objectives(torch,s,y)['point']
        torch.testing.assert_close(old,terms['point'],rtol=0,atol=0)
        ga=torch.autograd.grad(old,s,retain_graph=True)[0]
        gb=torch.autograd.grad(terms['point'],s)[0]
        torch.testing.assert_close(ga,gb,rtol=0,atol=0)
        zero=run.common.objectives(torch,y,y)
        self.assertEqual(float(zero['regret_margin']),0.)
        shifted=run.common.objectives(torch,y+.37,y)
        self.assertLess(abs(float(shifted['regret_margin'])),1e-15)
        with self.assertRaises(ValueError):run.common.objectives(torch,s[:,:5],y[:,:5])

    def test_real_parameter_count_initial_weights_and_prior(self):
        torch=self.torch
        stats=dict(prior=np.array([.7,.6,.5,.4,.3,.2]),mean_tokens=np.zeros((320,192)),mean_pooled=np.zeros(768))
        models=[run.architecture.build(torch,stats,64,'interaction_residual') for _ in run.ARMS]
        self.assertEqual(len({run.source.model_digest(m) for m in models}),1)
        for model in models:
            self.assertEqual(sum(p.numel() for p in model.parameters()),6720)
            self.assertEqual(model.readout.in_features,1088)
            np.testing.assert_allclose(model(torch.zeros(1,320,192)).detach().numpy()[0],stats['prior'],atol=3e-8)

    def test_matched_training_all_four_arms_legacy_replay_and_tamper(self):
        groups,bank=self.bank();fold=run.protocol.split_groups(groups)[0]
        stats=run.architecture.fit_statistics(bank,fold['fit'])
        draws=run.common.pair_draws(fold,2132,{g:2 for g in groups})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'203';scopes=[];progress=Progress();artifacts={}
            for arm in run.ARMS:
                model,scope,history=run.fit(self.torch,root,dict(fingerprint='test'),bank,stats,fold,2132,arm,draws,2,progress,artifacts)
                self.assertEqual(len(history),256);self.assertEqual(scope['examples'],1024)
                self.assertEqual(scope['pairs'],512);self.assertTrue(scope['fixed_fit_buffers_unchanged'])
                self.assertGreater(scope['maximum_projection_gradient'],0)
                self.assertTrue(all(not p.requires_grad for p in model.parameters()));scopes.append(scope)
                objective,clipping=run.common.flags(arm)
                if not clipping:
                    self.assertEqual(scope['clipped_updates'],0)
                    self.assertTrue(all(r['gradient_scale']==1 and not r['gradient_was_clipped'] for r in history))
                    self.assertTrue(all(r['gradient_norm']==r['gradient_norm_after'] for r in history))
                else:
                    _,old,_=run.previous.fit(self.torch,Path(tmp)/'202',dict(fingerprint='old'),bank,stats,fold,2132,objective,draws,2,Progress(),{})
                    self.assertEqual(scope['final_sha256'],old['final_sha256'])
                self.assertTrue(all(r['parameter_update_norm']>0 for r in history))
            for k in ('initial_sha256','pair_draws_sha256','examples','trainable_parameters','input_features'):
                self.assertEqual(len({scope[k] for scope in scopes}),1)
            _,again,_=run.fit(self.torch,root,dict(fingerprint='test'),bank,stats,fold,2132,'regret_no_clip',draws,2,progress,artifacts)
            self.assertEqual(again['final_sha256'],scopes[-1]['final_sha256']);self.assertEqual(progress.reused,256)
            checkpoint=root/'training/fold0/seed2132/regret_no_clip.pt';checkpoint.write_bytes(b'changed')
            with self.assertRaises(ValueError):run.fit(self.torch,root,dict(fingerprint='test'),bank,stats,fold,2132,'regret_no_clip',draws,2,progress,artifacts)

    def test_gradient_off_does_not_mutate_and_on_matches_legacy(self):
        torch=self.torch
        for clipping in (False,True):
            p=torch.nn.Parameter(torch.zeros(2,dtype=torch.float64));p.grad=torch.tensor([3.,4.],dtype=torch.float64)
            original=p.grad.clone();q=torch.nn.Parameter(p.detach().clone());q.grad=original.clone()
            value=run.common.gradient_rule(torch,[p],clipping)
            if clipping:
                torch.nn.utils.clip_grad_norm_([q],1.,error_if_nonfinite=True)
                torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
                self.assertTrue(value['gradient_was_clipped']);self.assertLessEqual(value['gradient_norm_after'],1.)
            else:
                torch.testing.assert_close(p.grad,original,rtol=0,atol=0)
                self.assertFalse(value['gradient_was_clipped']);self.assertEqual(value['gradient_norm_after'],5.)
            self.assertEqual(value['gradient_norm'],5.)
            p.grad.fill_(float('nan'))
            with self.assertRaises(FloatingPointError):run.common.gradient_rule(torch,[p],clipping)

    def test_nonfinite_failure_is_sealed_and_prevents_retry(self):
        groups,bank=self.bank();fold=run.protocol.split_groups(groups)[0]
        stats=run.architecture.fit_statistics(bank,fold['fit'])
        draws=run.common.pair_draws(fold,2132,{g:2 for g in groups})
        for g in fold['fit']:bank[g]['labels'].fill(float('nan'))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'failure'
            with self.assertRaises(FloatingPointError):run.fit(self.torch,root,dict(fingerprint='test'),bank,stats,fold,2132,'regret_no_clip',draws,2,Progress(),{})
            failure=run.source.read_sealed(root/'diagnostics/nonfinite_failure.json')
            self.assertEqual(failure['status'],'FAILED_NONFINITE_TRAINING');self.assertEqual(failure['step'],1)
            self.assertEqual(failure['field'],'loss');self.assertFalse((root/'stage203_decision.json').exists())
            with self.assertRaises(ValueError):run.fit(self.torch,root,dict(fingerprint='test'),bank,stats,fold,2132,'point_clip',draws,2,Progress(),{})
            with self.assertRaises(ValueError):run.verify_results(root)

    def test_synthetic_full_pipeline_budgets_bounds_resume_and_no_cuda(self):
        groups,bank=self.bank();folds=run.protocol.split_groups(groups)
        entries=[dict(group=g,sequence=bank[g]['sequence'],sample_frames=[1,2]) for g in groups]
        p194=dict(fingerprint=run.inputs.SOURCE_PLAN,entries=entries,folds=folds,runtime=dict(tracker=dict(lens_z=2)))
        with tempfile.TemporaryDirectory() as tmp:
            r194,out,bundle=Path(tmp)/'194',Path(tmp)/'output',Path(tmp)/'source/202.txt'
            r194.mkdir();bundle.parent.mkdir();bundle.write_bytes(b'fixture');artifacts={}
            for e in entries:
                g,seq=e['group'],e['sequence']
                for relative,data,meta in (
                    (f'features/{seq}.npz',dict(tokens=bank[g]['tokens'],frame_id=bank[g]['frame_id']),dict(plan_fingerprint=p194['fingerprint'],sequence=seq,kind='native_parent_prefix')),
                    (f'reference/{g}.npz',dict(iou=bank[g]['labels']),dict(plan_fingerprint=p194['fingerprint'],kind='fixed_layer_fit_labels',group=g))):
                    path=r194/relative;run.source.cache(path,meta,lambda data=data:data);run.source.record(r194,path,artifacts)
            old_fit=[dict(fold=str(f['fold']),**{f'prior_L{k}':str(run.architecture.fit_statistics(bank,f['fit'])['prior'][k-7]) for k in range(7,13)}) for f in folds]
            old_frames=[];old_scopes=[]
            for fold in folds:
                prior=run.architecture.fit_statistics(bank,fold['fit'])['prior'].astype(np.float32)
                for seed in run.protocol.DESIGN['seeds']:
                    for objective in ('point','regret_margin'):
                        old_scopes.append(dict(fold=str(fold['fold']),seed=str(seed),arm=objective,final_sha256='f'*64))
                        for g in groups:
                            for i in range(2):
                                old_frames.append(dict(fold=str(fold['fold']),seed=str(seed),arm=objective,group=g,frame=str(i+1),
                                    selected_layer=str(int(run.metrics.choose(prior[None])[0])+7),
                                    **{f'prediction_L{k}':str(float(prior[k-7])) for k in range(7,13)},
                                    **{f'label_L{k}':str(bank[g]['labels'][i,k-7]) for k in range(7,13)}))
            d194=dict(fingerprint=run.inputs.SOURCE194,table_sha256={},artifact_sha256=artifacts)
            fixtures=(p194,d194,run.inputs.VerifiedSource(r194,d194),{},dict(fingerprint=run.SOURCE202),run.load202(run.DEFAULT_BUNDLE)[1],dict(frame_predictions=old_frames,fit_summary=old_fit,model_scope=old_scopes))
            stdout=io.StringIO()
            with patch.object(run,'load_sources',return_value=fixtures),patch.object(sys,'argv',['stage203','--stage194-root',str(r194),'--stage202-bundle',str(bundle),'--output-dir',str(out)]),patch.dict(os.environ,{}),patch.object(self.torch.cuda,'_lazy_init',side_effect=AssertionError('No CUDA')),contextlib.redirect_stdout(stdout):
                run.main();self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'');run.verify_results(out)
                with patch.object(run,'load_sources',side_effect=AssertionError('Must reuse')):run.main()
            d=run.source.read_sealed(out/'stage203_decision.json')
            self.assertEqual(d['training_runs'],16);self.assertEqual(d['training_updates'],4096)
            self.assertEqual(d['predicted_states'],1216);self.assertEqual(d['training_state_references'],16384)
            self.assertEqual(d['training_pair_uses'],8192);self.assertEqual(d['pair_schedule_rows'],2048)
            self.assertEqual(len(d['primary_contrasts']),5);self.assertEqual(d['new_encoder_calls'],0)
            self.assertFalse(d['actual_transformer_calls_measured']);self.assertIn('100.00%',stdout.getvalue())
            def read(name):
                with (out/f'stage203_{name}.csv').open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
            expected=dict(model_scope=16,fit_summary=2,source_bridge=38,pair_schedule=2048,pair_support=76,
                error_summary=24,error_video=1824,policy_summary=40,policy_video=3040,paired=92,frame_predictions=1216,training_history=4096,seed_summary=80,control_replay=8)
            for name,n in expected.items():self.assertEqual(len(read(name)),n)
            counts={}
            for row in read('pair_schedule'):
                for side in ('a','b'):
                    key=row['fold'],row['seed'],row['group'],row['frame_'+side];counts[key]=counts.get(key,0)+1
            for row in read('frame_predictions'):
                if row['pool']=='fit':self.assertEqual(int(row['training_draw_count']),counts.get((row['fold'],row['seed'],row['group'],row['frame']),0))
                else:self.assertEqual(row['training_draw_count'],'')
                self.assertLessEqual(float(row['selection_regret']),float(row['regret_margin_bound'])+1e-8+1e-12)
            for name,sha in artifacts.items():self.assertEqual(run.file_hash(r194/name),sha)
            (out/'stage203_policy_summary.csv').write_text('changed',encoding='utf8')
            with self.assertRaises(ValueError):run.verify_results(out)


if __name__=='__main__':unittest.main()
