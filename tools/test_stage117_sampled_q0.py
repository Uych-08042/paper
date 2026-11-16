"""Check outcome-free sampling, population weights and actual matched branch replay."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import stage117_sampled_q0 as stage
import test_stage113_frozen_closed_loop as fixture


class SampledQ0(unittest.TestCase):
    def test_uniform_and_sign_strata_are_deterministic_and_keep_weights_separate(self):
        n=100;frames=np.arange(1,n+1);physical=frames%11!=0;x=np.zeros((n,len(stage.evidence.FEATURES)))
        for i in range(n):
            for name,bit in zip(('hann_advantage','continuity_advantage','forecast_advantage'),(1,2,4)):
                x[i,stage.evidence.FEATURES.index(name)]=1 if i%8&bit else -1
        first=stage.sample_frames(frames,physical,x,[8,9,10],'context')
        second=stage.sample_frames(frames,physical,x,[10,8,9,8],'context');self.assertEqual(first,second)
        support,anchors=first;self.assertEqual(len(support),9);self.assertLessEqual(len(anchors),16)
        self.assertEqual(support[0]['draws'],8);self.assertEqual(sum(r['draws'] for r in support[1:]),8)
        self.assertTrue(all(5<=a['frame']<=n-31 and a['frame'] not in (8,9,10) and a['frame']%11 for a in anchors))
        self.assertAlmostEqual(sum(a['uniform_weight'] for a in anchors),1.)
        self.assertAlmostEqual(sum(a['stratified_weight'] for a in anchors),1.)
        self.assertEqual(len({a['frame'] for a in anchors}),len(anchors))
        for a in anchors:
            for study in stage.STUDIES:self.assertEqual(study in a['memberships'],a[study+'_weight']>0)
        empty,selected=stage.sample_frames(frames,np.zeros(n,bool),x,[],'empty')
        self.assertFalse(selected);self.assertTrue(all(r['draws']==0 and r['inclusion_probability']==0 for r in empty))

    def test_sampling_ignores_gt_outcomes_and_excludes_known_starts_across_seeds(self):
        n=70;trace=dict(frame_id=np.arange(1,n+1),physical=np.ones(n,bool),x=np.zeros((n,len(stage.evidence.FEATURES))))
        jobs=[dict(arm='generated_shared',seed=seed,outer_fold=0,spec=dict(cohort='historical',unit=dict(sequence='a'))) for seed in (2312,2313)]
        known=[dict(sequence='a',seed=2312,frame=10),dict(sequence='a',seed=2313,frame=20)]
        a=stage.sampling(jobs,lambda _:trace,known)
        poison=dict(trace,ground_truth=np.full((n,4),np.nan),future_gain=np.arange(n),valid=np.zeros(n,bool),q0_iou=np.ones(n))
        self.assertEqual(a,stage.sampling(jobs,lambda _:poison,known))
        self.assertFalse(any(r['frame'] in (10,20) for r in a[1]))
        self.assertEqual([r['excluded_ids'] for r in a[0]],[[10,20]]*18)

    def test_weighted_strata_rate_control_and_primary_comparison(self):
        choices=[];samples=[]
        for i,(q,p) in enumerate(((.8,.1),(.1,.9))):
            r=dict(anchor_id=str(i),context='a',sequence='a',seed=2312,cohort='historical',frame=i+1,memberships='uniform|stratified',
                uniform_weight=.5,stratified_weight=.9 if i==0 else .1,policy_future=p,q0_future=q,future_gain=q-p,
                policy_low=1-p,q0_low=1-q,current_role='other',outcome=stage.evidence.outcome(q,p))
            samples.append(r)
            choices.extend(dict(r,rule=rule,protect_probability=float(rule=='q0_all' or rule=='forecast_pair' and i==0)) for rule in stage.RULES)
        actions,summary,pairs,support=stage.summaries(samples,choices)
        def get(study,rule):return next(r for r in summary if r['study']==study and r['cohort']=='all' and r['rule']==rule)
        self.assertAlmostEqual(get('uniform','forecast_pair')['realized_gain'],.35)
        self.assertAlmostEqual(get('stratified','forecast_pair')['realized_gain'],.63)
        self.assertAlmostEqual(get('uniform','forecast_pair__rate_expected')['protect_probability'],.5)
        self.assertAlmostEqual(get('stratified','forecast_pair__rate_expected')['protect_probability'],.9)
        for study,gain in (('uniform',.375),('stratified',.135)):
            r=next(r for r in pairs if r['study']==study and r['cohort']=='all' and r['contrast']=='forecast_pair_minus_forecast_pair__rate_expected')
            self.assertAlmostEqual(r['mean_delta'],gain);self.assertEqual(r['rate_informative_contexts'],1)
        self.assertEqual(stage.current_role(.7,0),'good_q0_lost');self.assertEqual(stage.current_role(.3,.1),'medium_q0_lost')
        self.assertEqual(stage.current_role(0,.8),'q0_rescue');self.assertEqual(stage.current_role(.6,.7),'other')

    def test_first_card_change_and_resume_identity(self):
        a=dict(tracker=dict(device='NVIDIA L20',dtype='float32'),gpu_uuid='a')
        b=dict(tracker=dict(device='NVIDIA L40',dtype='float32'),gpu_uuid='b')
        self.assertEqual(len(stage.runtime_binding(a,b,b)),2)
        with self.assertRaisesRegex(ValueError,'STAGE117_ROOT'):stage.runtime_binding(a,b,a)
        broken=copy.deepcopy(b);broken['tracker']['dtype']='float16'
        with self.assertRaises(ValueError):stage.runtime_binding(a,broken)

    def test_actual_futures_ledger_receipts_before_labels_source_unchanged_and_resume(self):
        b=fixture.FrozenClosedLoop();b.setUp();self.addCleanup(b.doCleanups);s=stage.s;prior=stage.prior
        b.observer.assert_frozen=lambda:None;trace=b.collect('generated_shared');models={(0,'generator',0):b.package};jobs=[]
        for n,cohort in ((9,'historical'),(10,'expanded')):
            for seed in (2312,2313):
                m=copy.deepcopy(b.model);m['seed']=seed;models[0,'generated_4',seed]=s.io.seal({k:v for k,v in m.items() if k!='fingerprint'})
                jobs.append(dict(arm='generated_shared',seed=seed,outer_fold=0,frames=len(trace['frame_id']),
                    spec=dict(cohort=cohort,unit=dict(sequence=f'uav{n:07d}_00000_s',seed=2171,origin='native',policy='ppo',fold=0))))
        public=stage.evidence.diagnose(trace,b.entry,b.model)
        signature=s.io.seal(dict(models={str(k):v for k,v in models.items()}))['fingerprint']
        design=dict(stage.DESIGN,horizon=4,first_frame=2,uniform_per_context=2)
        with tempfile.TemporaryDirectory() as directory,patch.object(stage,'DESIGN',design),contextlib.redirect_stdout(io.StringIO()):
            base=Path(directory);root=base/'out';r113=base/'source';root.mkdir();r113.mkdir()
            p113=s.io.seal(dict(jobs=jobs))
            for job in jobs:s.c.save(r113/'evaluation'/(prior.key(job)+'.npz'),trace,prior.source_meta(p113,job,models))
            support,anchors=stage.sampling(jobs,lambda _:public,[])
            self.assertGreater(len(anchors),4)
            before_files={str(p):p.read_bytes() for p in r113.rglob('*') if p.is_file()}
            plan=s.io.seal(dict(schema=stage.SCHEMA,design=design,sampling_scope=support,anchors=anchors,source_stage116_fingerprint=stage.SOURCE116,
                source_anchor_evidence={a['anchor_id']:public['x'][a['frame']-1].tolist() for a in anchors},runtime={},runtime_differences=[],native_head_sha256='native',
                source_sha256={p:s.c.file_hash(Path(p)) for p in before_files},code_sha256={}))
            s.c.publish_identical(root/'manifest.json',plan)
            original=stage.measured_rows
            def measured(*args):
                self.assertEqual(len(list(root.rglob('*.receipt.json')))%3,0)
                return original(*args)
            with (patch.object(s.e63.source_experiment,'installed',side_effect=lambda *_:contextlib.nullcontext('native')),
                  patch.object(stage,'measured_rows',side_effect=measured),patch.object(s.common,'train',side_effect=AssertionError('New training')),
                  patch.object(s.common,'fit_generator',side_effect=AssertionError('New generator fitting'))):
                args=(root,r113,p113,plan,models,lambda _:b.entry,b.old,b.observer)
                calls=b.observer.calls;stage.run(*args,image_reader=b.reader)
                report=s.c.c66.sealed(root/'analysis/stage117_decision.json',stage.SCHEMA)
                self.assertEqual(report['new_search_calls'],b.observer.calls-calls);self.assertEqual(report['new_block_calls'],7*report['new_search_calls'])
                self.assertEqual(report['primary']['contrast'],stage.DESIGN['primary']);self.assertEqual(report['training_updates'],0)
                self.assertEqual(report['counts']['branches'],2*len(anchors));self.assertEqual(report['counts']['branch_frames'],8*len(anchors))
                snapshot={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
                with patch.object(b.observer,'capture',side_effect=AssertionError('Repeated inference')):stage.run(*args,image_reader=b.reader)
                self.assertEqual(snapshot,{str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()})
            self.assertEqual(before_files,{str(p):p.read_bytes() for p in r113.rglob('*') if p.is_file()})
            self.assertEqual(signature,s.io.seal(dict(models={str(k):v for k,v in models.items()}))['fingerprint'])
            stage.check_results(root);next((root/'branches').glob('*.npz')).write_bytes(b'changed')
            with self.assertRaises(ValueError):stage.check_results(root)


if __name__=='__main__':unittest.main()
