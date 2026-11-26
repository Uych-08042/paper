"""Strict inner-source isolation, analytic trust, frozen controls and CPU pipeline."""
import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'tools'),str(REPO/'tracking')]
import stage205_crossfit_shrinkage as run


class PureTests(unittest.TestCase):
    def test_reviewed_204_bundle_and_tamper_rejection(self):
        d,p,t = run.load204(run.DEFAULT_BUNDLE)
        self.assertEqual(d['fingerprint'],run.SOURCE204)
        self.assertEqual(p['fingerprint'],run.PLAN204)
        self.assertEqual(len(t),14)
        self.assertEqual(len(t['frame_predictions']),21888)
        held = run.outer_records(t,p['folds'])
        self.assertEqual(len(held),228)
        self.assertEqual(sum(len(v['labels']) for v in held.values()),10944)
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp)/'changed.txt';bad.write_bytes(run.DEFAULT_BUNDLE.read_bytes()+b'changed')
            with self.assertRaises(ValueError):run.load204(bad)

    def test_analytic_alpha_equal_sources_boundaries_and_score_endpoints(self):
        common = run.common
        prior = np.full(6,.5)
        h = np.asarray([.1,-.1,.1,-.1,.1,-.1])
        videos = {g:dict(prediction=np.tile(prior+h,(n,1)),labels=np.tile(prior+a*h,(n,1)),prior=prior)
                  for g,n,a in [('short',1,.2),('long',10,.8)]}
        result = common.estimate_alpha(videos,list(videos))
        self.assertAlmostEqual(result['alpha'],.5,places=14)
        self.assertLessEqual(result['calibration_mse_shrunk'],min(result['calibration_mse_raw'],result['calibration_mse_prior']))
        self.assertGreater(abs(result['alpha']-(.2+10*.8)/11),.1)
        for gain,expected in [(-.5,0.),(2.,1.),(.3,.3)]:
            v = dict(prediction=(prior+h)[None],labels=(prior+gain*h)[None],prior=prior)
            self.assertAlmostEqual(common.estimate_alpha({'g':v},['g'])['alpha'],expected)
        v = dict(prediction=prior[None],labels=(prior+h)[None],prior=prior)
        self.assertEqual(common.estimate_alpha({'g':v},['g'])['alpha'],0.)
        self.assertTrue(common.estimate_alpha({'g':v},['g'])['zero_denominator'])
        with self.assertRaises(ValueError):common.estimate_alpha(videos,['short'])
        p = np.array([[.1,0,.1-5e-9,0,0,0]],np.float32)
        np.testing.assert_array_equal(common.shrink(p,prior,1.),p)
        np.testing.assert_array_equal(run.metrics.choose(common.shrink(p,prior,1.)),run.metrics.choose(p))
        np.testing.assert_array_equal(common.shrink(p,prior,0.),prior[None])
        np.testing.assert_allclose(common.shrink(p+.15,prior,.4),common.shrink(p,prior,.4),atol=1e-8)
        with self.assertRaises(ValueError):common.shrink(p,prior,np.nan)


@unittest.skipUnless(importlib.util.find_spec('torch'),'CPU Torch required')
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch;torch.set_num_threads(1)

    def bank(self):
        groups = [f'uav{i:07d}' for i in range(38)]
        rng = np.random.default_rng(205);bank = {}
        for g in groups:
            tokens = rng.normal(size=(2,6,4)).astype(np.float32)
            bank[g] = dict(tokens=tokens,pooled=run.old.architecture.pooling.pool(self.torch,self.torch.as_tensor(tokens),2).numpy(),
                labels=rng.uniform(.1,.9,(2,6)),frame_id=np.array([1,2]),sequence=g+'_fixture')
        return groups,bank

    def test_inner_prior_features_scales_donors_and_draws_exclude_dropped(self):
        groups,bank = self.bank();outer = run.protocol.split_groups(groups)[0];drop=outer['fit'][0]
        fold = run.common.inner_fold(outer,drop)
        self.assertEqual(len(fold['fit']),18)
        with self.assertRaises(ValueError):run.common.inner_fold(outer,outer['held'][0])
        for arm in run.common.ARMS:
            innerbank = {g:bank[g] for g in outer['fit']}
            prepared,stats,mapping = run.old.relation.prepare(innerbank,fold,arm)
            self.assertTrue(all(g in fold['fit'] and g!=s for s,g in mapping.items()))
            changed = {g:dict(v) for g,v in innerbank.items()}
            changed[drop] = dict(changed[drop],labels=np.zeros_like(changed[drop]['labels']),
                                 tokens=changed[drop]['tokens']+100,pooled=changed[drop]['pooled']+100)
            altered,stats2,_ = run.old.relation.prepare(changed,fold,arm)
            for name in stats:np.testing.assert_array_equal(stats[name],stats2[name])
            for g in fold['fit']:np.testing.assert_array_equal(prepared[g]['pooled'],altered[g]['pooled'])
            np.testing.assert_array_equal(prepared[drop]['reference'],altered[drop]['reference'])
        draws = run.old.common.pair_draws(fold,2132,{g:2 for g in outer['fit']})
        self.assertEqual(len(draws),256)
        self.assertTrue(all(g in fold['fit'] and g!=drop for b in draws for g,a,c in b))
        real = dict(prior=np.full(6,.5),mean_tokens=np.zeros((320,192)),mean_pooled=np.zeros(768),scale_pooled=np.ones(768))
        self.assertEqual(sum(p.numel() for p in run.old.relation.build(self.torch,real,64,'matched_relation').parameters()),6720)

    def fixture(self,tmp):
        groups,bank = self.bank();folds=run.protocol.split_groups(groups)
        entries = [dict(group=g,sequence=bank[g]['sequence'],sample_frames=[1,2]) for g in groups]
        p194 = dict(fingerprint=run.old.inputs.SOURCE_PLAN,entries=entries,folds=folds,runtime=dict(tracker=dict(lens_z=2)))
        r194 = Path(tmp)/'194';r194.mkdir();artifacts={}
        for e in entries:
            g,seq=e['group'],e['sequence']
            for rel,data,meta in (
                (f'features/{seq}.npz',dict(tokens=bank[g]['tokens'],frame_id=bank[g]['frame_id']),dict(plan_fingerprint=p194['fingerprint'],sequence=seq,kind='native_parent_prefix')),
                (f'reference/{g}.npz',dict(iou=bank[g]['labels']),dict(plan_fingerprint=p194['fingerprint'],kind='fixed_layer_fit_labels',group=g))):
                path=r194/rel;run.source.cache(path,meta,lambda data=data:data);run.source.record(r194,path,artifacts)
        tables = dict(frame_predictions=[],fit_summary=[],policy_video=[])
        for f in folds:
            prior=run.old.architecture.fit_statistics(bank,f['fit'])['prior']
            for arm in run.common.ARMS:
                tables['fit_summary'].append(dict(fold=str(f['fold']),arm=arm,**{f'prior_L{k}':str(prior[k-7]) for k in range(7,13)}))
                for seed in run.protocol.DESIGN['seeds']:
                    for g in groups:
                        y=bank[g]['labels'];p=np.asarray(prior+.07*(y-y.mean(1,keepdims=True)),np.float32)
                        actions = run.metrics.choose(p)
                        for i,frame in enumerate(bank[g]['frame_id']):
                            tables['frame_predictions'].append(dict(fold=str(f['fold']),seed=str(seed),arm=arm,group=g,frame=str(frame),
                                sequence=bank[g]['sequence'],selected_layer=str(int(actions[i])+7),
                                **{f'prediction_L{k}':str(float(p[i,k-7])) for k in range(7,13)},
                                **{f'label_L{k}':str(y[i,k-7]) for k in range(7,13)}))
                        for name,a in [('trained',actions),('prior_only',run.metrics.choose(np.broadcast_to(prior,p.shape)))]:
                            tables['policy_video'].append(dict(fold=str(f['fold']),seed=str(seed),arm=arm,group=g,pool='fit' if g in f['fit'] else 'held',
                                policy=name,**run.old.predictor.diagnosis.correspondence(y,a),mean_selected_layer=float(np.mean(a+7))))
        d194 = dict(fingerprint=run.old.inputs.SOURCE194,artifact_sha256=artifacts,table_sha256={})
        real_d,real_p,_=run.load204(run.DEFAULT_BUNDLE)
        return r194,bank,(p194,d194,run.old.inputs.VerifiedSource(r194,d194),real_d,real_p,tables),artifacts

    def test_complete_228_fit_pipeline_interruption_reuse_lock_before_held_and_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            r194,bank,fixtures,source_artifacts=self.fixture(tmp)
            root=Path(tmp)/'output';bundle=Path(tmp)/'source/204.txt';bundle.parent.mkdir();bundle.write_bytes(b'fixture')
            argv=['stage205','--stage194-root',str(r194),'--stage204-bundle',str(bundle),'--output-dir',str(root)]
            actual_fit,actual_predict,actual_evaluate=run.old.fit,run.old.predict,run.evaluate_outer
            calls=[0]
            def interrupted_predict(*args):
                calls[0]+=1
                if calls[0]==2:raise InterruptedError('test normal interruption after model commit')
                return actual_predict(*args)
            def checked_fit(torch,innerroot,plan,prepared,stats,fold,seed,arm,*args):
                outer=next(f for f in plan['folds'] if f['fold']==fold['fold'])
                self.assertEqual(set(prepared),set(outer['fit']))
                self.assertFalse(set(outer['held'])&set(prepared))
                self.assertEqual(len(fold['fit']),18)
                return actual_fit(torch,innerroot,plan,prepared,stats,fold,seed,arm,*args)
            def checked_evaluate(root,plan,tables,bank,locked,rows,progress):
                self.assertEqual(len(list((root/'calibration').glob('*.json'))),12)
                self.assertEqual(len(locked),12)
                return actual_evaluate(root,plan,tables,bank,locked,rows,progress)
            with patch.object(run,'load_sources',return_value=fixtures),patch.object(sys,'argv',argv),patch.dict(os.environ,{}),\
                 patch.object(self.torch.cuda,'_lazy_init',side_effect=AssertionError('No CUDA')),contextlib.redirect_stdout(io.StringIO()):
                with patch.object(run.old,'predict',side_effect=interrupted_predict),patch.object(run.old,'fit',side_effect=checked_fit):
                    with self.assertRaises(InterruptedError):run.main()
                committed={str(p):run.file_hash(p) for p in root.rglob('*.pt')}
                self.assertEqual(len(committed),2)
                with patch.object(run.old,'fit',side_effect=checked_fit),patch.object(run,'evaluate_outer',side_effect=checked_evaluate):run.main()
                for path,sha in committed.items():self.assertEqual(run.file_hash(path),sha)
                decision=run.source.read_sealed(root/'stage205_decision.json')
                self.assertEqual(decision['training_runs'],228)
                self.assertEqual(decision['training_updates'],58368)
                self.assertEqual(decision['inner_predicted_states'],456)
                self.assertEqual(decision['outer_held_states'],456)
                self.assertEqual(len(decision['primary_contrasts']),5)
                self.assertEqual(len(decision['artifact_sha256']),1152)
                self.assertEqual(len(decision['table_sha256']),16)
                with patch.object(run.old,'fit',side_effect=AssertionError('Do not retrain completed run')):run.main()
            for relative,sha in source_artifacts.items():self.assertEqual(run.file_hash(r194/relative),sha)
            def read(name):
                with (root/f'stage205_{name}.csv').open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
            predictions=read('inner_predictions');calibration=read('calibration')
            for row in calibration:
                selected=[r for r in predictions if all(r[k]==row[k] for k in ('fold','seed','arm'))]
                videos={}
                for g in sorted({r['group'] for r in selected}):
                    cell=[r for r in selected if r['group']==g]
                    videos[g]=dict(prediction=np.asarray([[float(r[f'prediction_L{k}']) for k in range(7,13)] for r in cell]),
                        labels=np.asarray([[float(r[f'label_L{k}']) for k in range(7,13)] for r in cell]),prior=np.asarray([float(cell[0][f'prior_L{k}']) for k in range(7,13)]))
                self.assertAlmostEqual(run.common.estimate_alpha(videos,list(videos))['alpha'],float(row['alpha']),places=14)
            for row in read('frame_predictions'):
                p=np.asarray([[float(row[f'raw_score_L{k}']) for k in range(7,13)]])
                mu=np.asarray([float(row[f'prior_L{k}']) for k in range(7,13)])
                expected=run.common.shrink(p,mu,float(row['alpha']))
                np.testing.assert_array_equal(expected[0],[float(row[f'shrunk_score_L{k}']) for k in range(7,13)])
                self.assertEqual(int(run.metrics.choose(expected)[0])+7,int(row['shrunk_layer']))
            with self.assertRaises(ValueError):
                run.old.cache(next((root/'inner').rglob('statistics/*.npz')),dict(changed=True),lambda:{})
            (root/'stage205_policy_summary.csv').write_text('tampered',encoding='utf-8')
            with self.assertRaises(ValueError):run.verify_results(root)
            failure=root/'inner/fold0/drop_test/diagnostics/nonfinite_failure.json';failure.parent.mkdir(parents=True);failure.write_text('{}')
            with self.assertRaises(ValueError):run.failure_guard(root)


if __name__=='__main__':unittest.main()
