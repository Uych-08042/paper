"""Regret bound, matched MSE control, cache integrity and full CPU pipeline tests."""
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
import stage202_regret_routing as run


class Progress:
    def __init__(self):self.done,self.reused=0,0
    def advance(self,count=1):self.done+=count


class PureTests(unittest.TestCase):
    def test_reviewed_source200_and_committed_cache_integrity(self):
        d,p,tables=run.bundle_source.load_bundle(run.DEFAULT_BUNDLE)
        self.assertEqual(d['fingerprint'],run.SOURCE200)
        self.assertEqual(p['fingerprint'],run.PLAN200)
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
            with self.assertRaises(ValueError):run.bundle_source.load_bundle(bundle)

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
        old=run.common.pair_source.objectives(torch,s,y)['point']
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

    def test_matched_training_control_exact_replay_and_model_tamper(self):
        groups,bank=self.bank();fold=run.protocol.split_groups(groups)[0]
        stats=run.architecture.fit_statistics(bank,fold['fit'])
        draws=run.common.pair_draws(fold,2132,{g:2 for g in groups})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'202';scopes=[];progress=Progress();artifacts={}
            for arm in run.ARMS:
                model,scope,history=run.fit(self.torch,root,dict(fingerprint='test'),bank,stats,fold,2132,arm,draws,2,progress,artifacts)
                self.assertEqual(len(history),256);self.assertEqual(scope['examples'],1024)
                self.assertEqual(scope['pairs'],512);self.assertTrue(scope['fixed_fit_buffers_unchanged'])
                self.assertGreater(scope['maximum_projection_gradient'],0)
                self.assertTrue(all(not p.requires_grad for p in model.parameters()));scopes.append(scope)
            for k in ('initial_sha256','pair_draws_sha256','examples','trainable_parameters','input_features'):
                self.assertEqual(scopes[0][k],scopes[1][k])
            # Original Stage-200 MSE computation must produce the same learned weights.
            _,baseline,_=run.previous.fit(self.torch,Path(tmp)/'200',dict(fingerprint='old'),bank,stats,fold,2132,'point',draws,2,Progress(),{})
            self.assertEqual(scopes[0]['final_sha256'],baseline['final_sha256'])
            _,again,_=run.fit(self.torch,root,dict(fingerprint='test'),bank,stats,fold,2132,'regret_margin',draws,2,progress,artifacts)
            self.assertEqual(again['final_sha256'],scopes[1]['final_sha256']);self.assertEqual(progress.reused,256)
            checkpoint=root/'training/fold0/seed2132/regret_margin.pt';checkpoint.write_bytes(b'changed')
            with self.assertRaises(ValueError):run.fit(self.torch,root,dict(fingerprint='test'),bank,stats,fold,2132,'regret_margin',draws,2,progress,artifacts)

    def test_synthetic_full_pipeline_budgets_bounds_resume_and_no_cuda(self):
        groups,bank=self.bank();folds=run.protocol.split_groups(groups)
        entries=[dict(group=g,sequence=bank[g]['sequence'],sample_frames=[1,2]) for g in groups]
        p194=dict(fingerprint=run.inputs.SOURCE_PLAN,entries=entries,folds=folds,runtime=dict(tracker=dict(lens_z=2)))
        with tempfile.TemporaryDirectory() as tmp:
            r194,out,bundle=Path(tmp)/'194',Path(tmp)/'output',Path(tmp)/'source/200.txt'
            r194.mkdir();bundle.parent.mkdir();bundle.write_bytes(b'fixture');artifacts={}
            for e in entries:
                g,seq=e['group'],e['sequence']
                for relative,data,meta in (
                    (f'features/{seq}.npz',dict(tokens=bank[g]['tokens'],frame_id=bank[g]['frame_id']),dict(plan_fingerprint=p194['fingerprint'],sequence=seq,kind='native_parent_prefix')),
                    (f'reference/{g}.npz',dict(iou=bank[g]['labels']),dict(plan_fingerprint=p194['fingerprint'],kind='fixed_layer_fit_labels',group=g))):
                    path=r194/relative;run.source.cache(path,meta,lambda data=data:data);run.source.record(r194,path,artifacts)
            old_fit=[dict(fold=str(f['fold']),**{f'prior_L{k}':str(run.architecture.fit_statistics(bank,f['fit'])['prior'][k-7]) for k in range(7,13)}) for f in folds]
            old_frames=[dict(group=g,frame=str(i+1),**{f'label_L{k}':str(bank[g]['labels'][i,k-7]) for k in range(7,13)}) for g in groups for i in range(2)]
            d194=dict(fingerprint=run.inputs.SOURCE194,table_sha256={},artifact_sha256=artifacts)
            fixtures=(p194,d194,run.inputs.VerifiedSource(r194,d194),{},dict(fingerprint=run.SOURCE200),dict(frame_predictions=old_frames,fit_summary=old_fit))
            stdout=io.StringIO()
            with patch.object(run,'load_sources',return_value=fixtures),patch.object(sys,'argv',['stage202','--stage194-root',str(r194),'--stage200-bundle',str(bundle),'--output-dir',str(out)]),patch.dict(os.environ,{}),patch.object(self.torch.cuda,'_lazy_init',side_effect=AssertionError('No CUDA')),contextlib.redirect_stdout(stdout):
                run.main();self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'');run.verify_results(out)
                with patch.object(run,'load_sources',side_effect=AssertionError('Must reuse')):run.main()
            d=run.source.read_sealed(out/'stage202_decision.json')
            self.assertEqual(d['training_runs'],8);self.assertEqual(d['training_updates'],2048)
            self.assertEqual(d['predicted_states'],608);self.assertEqual(d['training_state_references'],8192)
            self.assertEqual(d['training_pair_uses'],4096);self.assertEqual(d['pair_schedule_rows'],2048)
            self.assertEqual(len(d['primary_contrasts']),3);self.assertEqual(d['new_encoder_calls'],0)
            self.assertFalse(d['actual_transformer_calls_measured']);self.assertIn('100.00%',stdout.getvalue())
            def read(name):
                with (out/f'stage202_{name}.csv').open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
            expected=dict(model_scope=8,fit_summary=2,source_bridge=38,pair_schedule=2048,pair_support=76,
                error_summary=12,error_video=912,policy_summary=20,policy_video=1520,paired=40,frame_predictions=608,training_history=2048,seed_summary=40)
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
            (out/'stage202_policy_summary.csv').write_text('changed',encoding='utf8')
            with self.assertRaises(ValueError):run.verify_results(out)


if __name__=='__main__':unittest.main()
