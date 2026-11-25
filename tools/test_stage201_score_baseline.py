"""Causality, offset invariance, source integrity and complete cache-only reports."""
import builtins
import contextlib
import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'tools'),str(REPO/'tracking')]
import stage201_score_baseline as run


def fixture():
    groups = [f'uav{i:07d}' for i in range(38)]
    folds = run.protocol.split_groups(groups)
    rng = np.random.default_rng(201)
    labels = {g:rng.uniform(.1,.9,(48,6)) for g in groups}
    tables = dict(model_scope=[],frame_predictions=[],fit_summary=[],policy_video=[])
    for fold in folds:
        prior = np.stack([labels[g].mean(0) for g in fold['fit']]).mean(0)
        tables['fit_summary'].append(dict(fold=fold['fold'],**{f'prior_L{k}':str(prior[k-7]) for k in range(7,13)}))
        for seed in run.protocol.DESIGN['seeds']:
            for arm in run.ARMS:
                tables['model_scope'].append(dict(fold=fold['fold'],seed=seed,arm=arm,fit_groups='|'.join(fold['fit']),
                    held_groups='|'.join(fold['held']),final_sha256=run.fingerprint([fold['fold'],seed,arm]),source='synthetic',pair_draws_sha256='fixture'))
                for g in groups:
                    y=labels[g]
                    p=(y*.1+rng.uniform(-.06,.06,(1,6))+.4).astype(np.float32)
                    actions=run.metrics.choose(p)
                    base=dict(fold=fold['fold'],seed=seed,arm=arm,group=g,sequence=g+'_test',pool='fit' if g in fold['fit'] else 'held')
                    tables['policy_video'].append(dict(base,policy='trained',**run.correspondence.correspondence(y,actions)))
                    for i in range(48):
                        tables['frame_predictions'].append(dict(base,frame=i*3+1,selected_layer=int(actions[i])+7,
                            selected_iou=float(y[i,actions[i]]),**{f'prediction_L{k}':str(p[i,k-7]) for k in range(7,13)},
                            **{f'label_L{k}':str(y[i,k-7]) for k in range(7,13)}))
    return dict(predicted_states=14592,table_sha256={}),dict(folds=folds,seeds=[2132,2133]),tables


class Tests(unittest.TestCase):
    def test_scores_exact_unclipped_and_label_free(self):
        p=np.array([[4.,0,1,2,3,-2],[2,1,0,3,4,-1],[3,2,1,4,5,0]])
        mu=np.arange(6)/10
        s=run.baseline_scores(p,mu)
        np.testing.assert_array_equal(s['causal_past_mean'][0],mu)
        np.testing.assert_allclose(s['causal_past_mean'][1],mu+p[1]-p[0])
        np.testing.assert_allclose(s['causal_past_mean'][2],mu+p[2]-p[:2].mean(0))
        np.testing.assert_allclose(s['offline_video_mean'].mean(0),mu,atol=1e-15)
        self.assertLess(s['causal_past_mean'].min(),0)
        self.assertGreater(s['offline_video_mean'].max(),1)
        with self.assertRaises(ValueError):run.baseline_scores(np.empty((0,6)),mu)
        with self.assertRaises(ValueError):run.baseline_scores(np.full((2,6),np.nan),mu)
        np.testing.assert_array_equal(run.baseline_scores(p[:1],mu)['causal_past_mean'],mu[None])

    def test_causal_prefix_ignores_future_and_first_action_ignores_current(self):
        rng=np.random.default_rng(2);p=rng.normal(size=(12,6));mu=np.arange(6)/30
        before=run.baseline_scores(p,mu)
        changed=p.copy();changed[6:]+=rng.normal(size=(6,6))*100
        after=run.baseline_scores(changed,mu)
        np.testing.assert_array_equal(before['causal_past_mean'][:6],after['causal_past_mean'][:6])
        self.assertFalse(np.allclose(before['offline_video_mean'][:6],after['offline_video_mean'][:6]))
        for n in range(1,13):
            np.testing.assert_array_equal(run.baseline_scores(p[:n],mu)['causal_past_mean'],before['causal_past_mean'][:n])
        np.testing.assert_array_equal(run.baseline_scores(changed[6:7],mu)['causal_past_mean'][0],mu)

    def test_video_constant_layer_shift_cancels(self):
        rng=np.random.default_rng(3);p=rng.normal(size=(9,6));mu=np.arange(6)/20;c=rng.normal(size=6)*10
        a,b=run.baseline_scores(p,mu),run.baseline_scores(p+c,mu)
        for mode in ('offline_video_mean','causal_past_mean'):
            np.testing.assert_allclose(a[mode],b[mode],atol=1e-14,rtol=0)
            np.testing.assert_array_equal(run.metrics.choose(a[mode]),run.metrics.choose(b[mode]))

    def test_real_source_integrity_and_no_unknown_bundle(self):
        d,p,t=run.load_bundle(run.DEFAULT_BUNDLE)
        self.assertEqual(d['fingerprint'],run.SOURCE)
        self.assertEqual(len(t['frame_predictions']),14592)
        self.assertEqual(len(list(run.models(d,p,t))),8)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fake.txt';path.write_bytes(b'changed')
            with self.assertRaises(ValueError):run.load_bundle(path)

    def test_complete_synthetic_reports_reuse_tampering_and_no_torch(self):
        source=fixture()
        original_import=builtins.__import__
        def guarded(name,*a,**kw):
            if name.split('.')[0] in ('torch','cv2','PIL'):
                raise AssertionError('No model/image import allowed')
            return original_import(name,*a,**kw)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'out'
            argv=['stage201_score_baseline','--output-dir',str(root)]
            with patch.object(run,'load_bundle',return_value=source),patch.object(sys,'argv',argv),patch('builtins.__import__',side_effect=guarded),contextlib.redirect_stdout(io.StringIO()):
                run.main()
            decision=run.verify_results(root)
            self.assertEqual(decision['evaluated_states'],14592)
            self.assertEqual(decision['causal_baseline_violations'],0)
            self.assertEqual(len(decision['primary_contrasts']),3)
            expected=dict(source_audit=8,calibration_video=304,policy_video=1824,policy_summary=24,seed_summary=48,paired=38,error_video=1216,error_summary=16,frame_predictions=14592)
            for name,count in expected.items():
                with (root/f'stage201_{name}.csv').open(newline='') as f:
                    self.assertEqual(len(list(run.csv.DictReader(f))),count)
            before={p.name:p.read_bytes() for p in root.iterdir() if p.is_file()}
            with patch.object(run,'load_bundle',side_effect=AssertionError('Must reuse')),patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):run.main()
            self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir() if p.is_file()})
            path=root/'stage201_policy_summary.csv';path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaises(ValueError):run.verify_results(root)

    def test_held_labels_do_not_change_fit_prior_or_transforms(self):
        d,p,t=fixture();first=next(run.models(d,p,t));_,fold,seed,arm,prior,videos=first
        g=fold['held'][0];before=run.baseline_scores(videos[g]['prediction'],prior)
        changed=copy.deepcopy(t)
        # Change the same held source in every model, preserving cross-model labels.
        for r in changed['frame_predictions']:
            if r['group']==g:
                for k in range(7,13):r[f'label_L{k}']='0.0'
                r['selected_iou']=0.
        after=next(run.models(d,p,changed));mu,bank=after[-2:]
        np.testing.assert_array_equal(mu,prior)
        for mode,score in before.items():np.testing.assert_array_equal(score,run.baseline_scores(bank[g]['prediction'],mu)[mode])
        broken=copy.deepcopy(t);broken['frame_predictions'][1]['frame']=1
        with self.assertRaises(ValueError):list(run.models(d,p,broken))


if __name__=='__main__':unittest.main()
