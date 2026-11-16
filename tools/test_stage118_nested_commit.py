"""Verify provider isolation, label exclusion, matched controls and CPU artifact reuse."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import stage118_nested_commit as stage


def fixture():
    groups=[f'uav{i:07d}' for i in range(1,7)]
    splits=[dict(fold=f,held_out_groups=groups[3*f:3*f+3],train_groups=groups[3*(1-f):3*(1-f)+3]) for f in range(2)]
    models={}
    for split in splits:
        f=split['fold']
        models[f,'generator',0]=stage.io.seal(dict(split=split,generator=dict(fit_groups=split['train_groups'][:2],calibration_groups=split['train_groups'][2:])))
        for seed in stage.s.DESIGN['seeds']:models[f,'generated_4',seed]=stage.io.seal(dict(split=split,seed=seed))
    providers=stage.provider_audit(splits,models);owners={};records=[]
    for i,g in enumerate(groups):
        seq=g+'_00000_s';fold=i//3;cohort='historical' if i%2==0 else 'expanded'
        owners[g]=dict(group=g,outer_fold=fold,sequence=seq)
        for seed in (2312,2313):
            for j in range(2):
                x={name:0. for name in stage.FEATURES};x['hann_advantage']=(-1.)**j;x['forecast_advantage']=.2*(i-2)
                x['continuity_advantage']=.3 if j==0 else -.3
                p=.3+.02*i;q=p+(.1 if j==0 else -.08)
                uid=f'stage117__{g}_{seed}_{j}'
                records.append(dict(x,uid=uid,anchor_id=uid,context=f'{g}_{seed}',sequence=seq,group=g,outer_fold=fold,seed=seed,frame=10+j,
                    cohort=cohort,origin='stage117',memberships='uniform|stratified' if j==0 else 'uniform',uniform_weight=.5,stratified_weight=1. if j==0 else 0.,
                    policy_current=.5,q0_current=.52 if j==0 else .48,policy_future=p,q0_future=q,future_gain=q-p,
                    policy_low=.2,q0_low=.1 if j==0 else .4,current_role='other',outcome=stage.ev.outcome(q,p)))
        pressure=copy.deepcopy(records[-1]);pressure.update(uid='stage115__'+g,anchor_id='pressure_'+g,origin='stage115',memberships='pressure',
            uniform_weight=0.,stratified_weight=0.,current_role=('good_q0_lost','medium_q0_lost','q0_rescue')[i%3])
        records.append(pressure)
    return records,owners,providers,splits,models


class NestedCommit(unittest.TestCase):
    def test_provider_partition_and_upstream_train_exclusion(self):
        records,owners,providers,splits,models=fixture();held=owners['uav0000001'];provider=providers[0]
        fit=stage.training_rows(records,held,provider)
        self.assertEqual({r['group'] for r in fit},{'uav0000002','uav0000003'})
        self.assertEqual(len(fit),8);self.assertTrue(all(r['outer_fold']==0 and r['origin']=='stage117' for r in fit))
        bad=copy.deepcopy(models);bad[0,'generator',0]['generator']['calibration_groups']=['uav0000001']
        with self.assertRaisesRegex(ValueError,'Generator source leakage'):stage.provider_audit(splits,bad)
        bad=copy.deepcopy(models);bad[0,'generated_4',2312]['split']=splits[1]
        with self.assertRaisesRegex(ValueError,'Scorer source leakage'):stage.provider_audit(splits,bad)
        bad=copy.deepcopy(records);next(r for r in bad if r['group']=='uav0000002')['outer_fold']=1
        with self.assertRaisesRegex(ValueError,'same-provider'):stage.training_rows(bad,held,provider)
        with self.assertRaisesRegex(ValueError,'upstream provider'):stage.training_rows(records,held,providers[1])

    def test_held_pressure_and_other_provider_labels_cannot_change_a_fit(self):
        records,owners,providers,_,_=fixture();held=owners['uav0000001'];provider=providers[0]
        fit=stage.training_rows(records,held,provider);package=stage.fit_package(fit,held,provider)
        poisoned=copy.deepcopy(records)
        for r in poisoned:
            if r['group']==held['group'] or r['origin']=='stage115' or r['outer_fold']!=held['outer_fold']:
                for name in ('q0_current','policy_current','future_gain','policy_future','q0_future'):r[name]=10000.
        refit=stage.fit_package(stage.training_rows(poisoned,held,provider),held,provider)
        self.assertEqual(package,refit)
        x=np.array([[r[k] for k in stage.FEATURES] for r in records if r['group']==held['group']])
        np.testing.assert_array_equal(stage.predict(package,x),stage.predict(refit,x))
        changed=copy.deepcopy(fit);changed[0]['future_gain']+=.5
        self.assertNotEqual(package['fit_input_fingerprint'],stage.fit_package(changed,held,provider)['fit_input_fingerprint'])
        broken=copy.deepcopy(package);broken['norm']['scale'][0]=0.
        with self.assertRaisesRegex(ValueError,'normalization'):stage.audit_package(broken,fit,held,provider)
        with self.assertRaisesRegex(ValueError,'public gate input'):stage.predict(package,np.full((1,len(stage.FEATURES)),np.nan))

    def test_ridge_response_to_known_signal_and_equal_capacity_controls(self):
        records,owners,providers,_,_=fixture();held=owners['uav0000001'];provider=providers[0]
        rows=stage.training_rows(records,held,provider)
        for i,r in enumerate(rows):
            value=1. if i%2 else -1.
            for k in stage.FEATURES:r[k]=0.
            r.update(hann_advantage=value,q0_current=.5+.04*value,policy_current=.5,future_gain=.1*value)
        package=stage.fit_package(rows,held,provider);stage.audit_package(package,rows,held,provider)
        x=np.zeros((3,len(stage.FEATURES)));x[:,0]=[-1,0,1]
        score=stage.predict(package,x)
        # With balanced +/-1 evidence, ridge=1 halves the true slopes; intercept is unpenalized.
        np.testing.assert_allclose(score[:,:2],[[-.02,-.05],[0,0],[.02,.05]],atol=1e-14)
        np.testing.assert_allclose(score[:,2:],0.,atol=1e-14)
        self.assertEqual(np.asarray(package['coefficients']).shape,(14,2))

    def test_study_weights_same_rate_expected_and_pressure_remain_separate(self):
        records,_,_,_,_=fixture();ordinary=copy.deepcopy(records[:2])
        for i,r in enumerate(ordinary):
            r.update(memberships='uniform|stratified',uniform_weight=.5,stratified_weight=.9 if i==0 else .1,
                policy_future=.1 if i==0 else .9,q0_future=.8 if i==0 else .1,future_gain=.7 if i==0 else -.8)
        pressure=copy.deepcopy(ordinary[0]);pressure.update(uid='pressure',origin='stage115',memberships='pressure',current_role='q0_rescue',
            policy_future=.9,q0_future=.1,future_gain=-.8)
        rows=ordinary+[pressure]
        scores=[dict(uid=r['uid'],myopic=-1.,future=1. if i==0 or i==2 else -1.,myopic_constant=-1.,future_constant=-1.) for i,r in enumerate(rows)]
        choices=stage.evaluate(rows,scores);summary,paired,support,_=stage.summarize(choices)
        def result(study,rule):return next(r for r in summary if r['study']==study and r['cohort']=='all' and r['current_role']=='all' and r['rule']==rule)
        self.assertAlmostEqual(result('ordinary_uniform','future')['realized_gain'],.35)
        self.assertAlmostEqual(result('ordinary_stratified','future')['realized_gain'],.63)
        self.assertAlmostEqual(result('ordinary_uniform','future__rate_expected')['protect_probability'],.5)
        self.assertAlmostEqual(result('ordinary_stratified','future__rate_expected')['protect_probability'],.9)
        self.assertAlmostEqual(result('pressure','future')['realized_gain'],-.8)
        self.assertEqual(result('pressure','future')['large_harm_anchors'],1)
        for study,delta in (('ordinary_uniform',.375),('ordinary_stratified',.135)):
            pair=next(r for r in paired if r['study']==study and r['cohort']=='all' and r['contrast']=='future_minus_future__rate_expected')
            self.assertAlmostEqual(pair['mean_delta'],delta);self.assertEqual(pair['rate_informative_anchors'],2)
        self.assertTrue(any(r['current_role']=='q0_rescue' and r['study']=='pressure' and r['anchors']==1 for r in support))
        for r in choices:self.assertAlmostEqual(r['regret'],r['harm_cost']+r['missed_gain'])

    def test_csv_owner_binding_and_unit_interval_rejection(self):
        records,_,_,_,_=fixture();r=copy.deepcopy(records[0])
        job=dict(arm='generated_shared',seed=r['seed'],outer_fold=r['outer_fold'],spec=dict(cohort=r['cohort'],unit=dict(sequence=r['sequence'])))
        got=stage.attach(r,'stage117',job);self.assertEqual(got['future_gain'],r['future_gain'])
        wrong=dict(r,q0_future=2.)
        with self.assertRaisesRegex(ValueError,'unit interval'):stage.attach(wrong,'stage117',job)
        wrong=dict(r,stratified_weight=0.)
        with self.assertRaisesRegex(ValueError,'study weights'):stage.attach(wrong,'stage117',job)

    def test_complete_cpu_workflow_receipts_resume_and_tamper(self):
        records,owners,providers,_,_=fixture()
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            base=Path(directory);root=base/'out';root.mkdir();source=base/'source.txt';source.write_text('frozen source')
            expected=dict(groups=6,unique_points=30,membership_points=42,uniform_fit_points=24,pressure_points=6)
            plan=stage.io.seal(dict(schema=stage.SCHEMA,design=stage.DESIGN,features=list(stage.FEATURES),expected=expected,
                source_stage117_fingerprint=stage.SOURCE117,source_sha256={str(source):stage.c.file_hash(source)},code_sha256={}))
            stage.c.publish_identical(root/'manifest.json',plan)
            original=stage.evaluate
            def evaluate(*args):
                self.assertEqual(len(list((root/'models').glob('*.json'))),6)
                self.assertEqual(len(list((root/'predictions').glob('*.receipt.json'))),6)
                for p in (root/'predictions').glob('*.npz'):
                    with np.load(p,allow_pickle=False) as d:self.assertEqual(set(d.files),{'x','scores','metadata_json'})
                return original(*args)
            with (patch.object(stage,'evaluate',side_effect=evaluate),
                  patch.object(stage.s.e63.actor,'observe',side_effect=AssertionError('New SGLA inference')),
                  patch.object(stage.s.s105,'checked_image',side_effect=AssertionError('Image read')),
                  patch.object(stage.s.common,'train',side_effect=AssertionError('Scorer training')),
                  patch.object(stage.s.common,'fit_generator',side_effect=AssertionError('Generator fitting'))):
                stage.run(root,plan,records,owners,providers);stage.check_results(root)
                report=stage.c.c66.sealed(root/'analysis/stage118_decision.json',stage.SCHEMA)
                self.assertEqual(report['new_models'],24);self.assertEqual(report['fitted_ridge_heads'],12)
                self.assertEqual(report['primary']['contrast'],'future_minus_myopic');self.assertEqual(report['image_reads'],0)
                snapshot={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
                with patch.object(stage,'fit_package',side_effect=AssertionError('Repeated fitting')):stage.run(root,plan,records,owners,providers)
                self.assertEqual(snapshot,{str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()})
            self.assertEqual(source.read_text(),'frozen source')
            next((root/'predictions').glob('*.npz')).write_bytes(b'tampered')
            with self.assertRaises(ValueError):stage.check_results(root)


if __name__=='__main__':unittest.main()
