"""Fit-only relation inputs, matched parameters and complete CPU experiment tests."""
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
import stage204_target_relation as run


class Progress:
    def __init__(self):self.done,self.reused=0,0
    def advance(self,count=1):self.done+=count


class PureTests(unittest.TestCase):
    def test_reviewed_source203_and_committed_cache_integrity(self):
        d,p,tables=run.load203(run.DEFAULT_BUNDLE)
        self.assertEqual(d['fingerprint'],run.SOURCE203)
        self.assertEqual(p['fingerprint'],run.PLAN203)
        self.assertEqual(len(tables['frame_predictions']),29184)
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
            with self.assertRaises(ValueError):run.load203(bundle)

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

    def test_relation_shapes_donors_and_held_label_isolation(self):
        groups,bank=self.bank();fold=run.protocol.split_groups(groups)[0]
        snapshots=[]
        for arm in run.ARMS:
            data,stats,mapping=run.relation.prepare(bank,fold,arm)
            self.assertTrue(all(mapping[g] in fold['fit'] and mapping[g]!=g for g in groups))
            model=run.relation.build(self.torch,stats,2,arm)
            snapshots.append({n:p.detach().clone() for n,p in model.named_parameters()})
            video=data[groups[0]];x=self.torch.as_tensor(video['tokens']);pooled=self.torch.as_tensor(video['pooled']);ref=self.torch.as_tensor(video['reference'][None])
            self.torch.testing.assert_close(model.pool(x,ref),pooled,rtol=0,atol=0)
            self.torch.testing.assert_close(model(x,pooled,ref),model(x,None,ref),rtol=0,atol=0)
            fit_before=stats['prior'].copy();g=fold['held'][0];old=bank[g]['labels'].copy();bank[g]['labels'].fill(0)
            changed,st,_=run.relation.prepare(bank,fold,arm)
            np.testing.assert_array_equal(st['prior'],fit_before);np.testing.assert_array_equal(changed[g]['pooled'],data[g]['pooled'])
            bank[g]['labels'][:]=old
            first=data[g]['pooled'][0].copy()
            saved=bank[g]['pooled'][1].copy();bank[g]['pooled'][1]+=100
            future,st,_=run.relation.prepare(bank,fold,arm)
            np.testing.assert_array_equal(future[g]['pooled'][0],first)
            np.testing.assert_array_equal(st['prior'],fit_before)
            bank[g]['pooled'][1]=saved
        for v in snapshots[1:]:
            for name in v:self.torch.testing.assert_close(v[name],snapshots[0][name],rtol=0,atol=0)
        real=dict(prior=np.array([.7,.6,.5,.4,.3,.2]),mean_tokens=np.zeros((320,192)),mean_pooled=np.zeros(768),scale_pooled=np.ones(768))
        self.assertEqual(sum(p.numel() for p in run.relation.build(self.torch,real,64,'concat').parameters()),6720)

    def test_train_all_inputs_and_resume(self):
        groups,bank=self.bank();fold=run.protocol.split_groups(groups)[0]
        draws=run.common.pair_draws(fold,2132,{g:2 for g in groups});scopes=[]
        with tempfile.TemporaryDirectory() as tmp:
            for arm in run.ARMS:
                data,stats,_=run.relation.prepare(bank,fold,arm)
                model,scope,history=run.fit(self.torch,Path(tmp),dict(fingerprint='test'),data,stats,fold,2132,arm,draws,2,Progress(),{})
                self.assertEqual(len(history),256);self.assertTrue(scope['fixed_fit_buffers_unchanged']);scopes.append(scope)
                output=run.predict(self.torch,model,data[groups[0]],Progress())
                self.assertEqual(float(output['maximum_error']),0.)
            self.assertEqual(len({s['initial_sha256'] for s in scopes}),1)
            _,again,_=run.fit(self.torch,Path(tmp),dict(fingerprint='test'),data,stats,fold,2132,arm,draws,2,Progress(),{})
            self.assertEqual(again['final_sha256'],scopes[-1]['final_sha256'])

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
            fixtures=(p194,d194,run.inputs.VerifiedSource(r194,d194),{},dict(fingerprint=run.SOURCE203),run.load203(run.DEFAULT_BUNDLE)[1],dict(frame_predictions=old_frames,fit_summary=old_fit,model_scope=old_scopes))
            stdout=io.StringIO()
            with patch.object(run,'load_sources',return_value=fixtures),patch.object(sys,'argv',['stage204','--stage194-root',str(r194),'--stage203-bundle',str(bundle),'--output-dir',str(out)]),patch.dict(os.environ,{}),patch.object(self.torch.cuda,'_lazy_init',side_effect=AssertionError('No CUDA')),contextlib.redirect_stdout(stdout):
                run.main();self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'');run.verify_results(out)
                with patch.object(run,'load_sources',side_effect=AssertionError('Must reuse')):run.main()
            d=run.source.read_sealed(out/'stage204_decision.json')
            self.assertEqual(d['training_runs'],12);self.assertEqual(d['training_updates'],3072)
            self.assertEqual(d['predicted_states'],912);self.assertEqual(d['training_state_references'],12288)
            self.assertEqual(d['training_pair_uses'],6144);self.assertEqual(d['pair_schedule_rows'],2048)
            self.assertEqual(len(d['primary_contrasts']),4);self.assertEqual(d['new_encoder_calls'],0)
            self.assertFalse(d['actual_transformer_calls_measured']);self.assertIn('100.00%',stdout.getvalue())
            def read(name):
                with (out/f'stage204_{name}.csv').open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
            expected=dict(model_scope=12,fit_summary=6,source_bridge=38,pair_schedule=2048,pair_support=76,
                error_summary=18,error_video=1368,policy_summary=30,policy_video=2280,paired=66,frame_predictions=912,training_history=3072,seed_summary=60,descriptor_audit=228)
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
            (out/'stage204_policy_summary.csv').write_text('changed',encoding='utf8')
            with self.assertRaises(ValueError):run.verify_results(out)


if __name__=='__main__':unittest.main()
