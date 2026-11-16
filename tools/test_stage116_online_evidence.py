"""Verify online evidence isolation, matched utility, weighting and zero-inference replay."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import stage116_online_evidence as stage
import test_stage113_frozen_closed_loop as fixture


class OnlineEvidence(unittest.TestCase):
    def test_auc_ties_macro_weights_and_no_positive_support(self):
        self.assertEqual(stage.weighted_auc([0,1,1,2],[0,1,0,1],[.25]*4),.875)
        self.assertEqual(stage.weighted_auc([0,0],[0,1],[.25,.75]),.5)
        self.assertIsNone(stage.weighted_auc([1,2],[0,0],[.5,.5]))
        rows=[dict(sequence='a',seed=1,x=0.),dict(sequence='a',seed=1,x=0.),dict(sequence='a',seed=2,x=1.),dict(sequence='b',seed=1,x=1.)]
        np.testing.assert_array_equal(stage.balanced(rows),[.125,.125,.25,.5])
        self.assertEqual(stage.macro(rows,'x'),dict(a=.5,b=1.))
        self.assertEqual(stage.outcome(.15,.1),'both_low')
        self.assertEqual(stage.outcome(.6,.2),'q0_better');self.assertEqual(stage.outcome(.2,.6),'policy_better')
        self.assertEqual(stage.outcome(.3,.3),'tie')

    def test_rules_use_only_evidence_and_rate_control_uses_whole_context(self):
        e={name:0. for name in stage.FEATURES}
        np.testing.assert_array_equal(stage.decisions(e,True),[0,1,0,0,0])
        e.update(hann_advantage=1.,raw_advantage=1.,continuity_advantage=1.)
        np.testing.assert_array_equal(stage.decisions(e,True),[0,1,1,1,1])
        np.testing.assert_array_equal(stage.decisions(e,False),np.zeros(5))
        anchors=[];rows=[]
        for i,(q,p,flag) in enumerate(((.8,.1,1),(.1,.9,0))):
            a=dict(anchor_id=str(i),cohort='historical',sequence='a',seed=2312,frame=i+1,
                roles=stage.prior.ROLES[0 if i==0 else 2],policy_future=p,q0_future=q,future_gain=q-p,policy_low=1-p,q0_low=1-q)
            anchors.append(a)
            rows.extend(dict(a,rule=rule,protect_probability=float(flag if rule in stage.RULES[2:] else rule=='q0_all')) for rule in stage.RULES)
        decisions,summary,paired,videos=stage.matched_summary(anchors,rows)
        random=[r for r in decisions if r['rule']=='hann_pair__rate_expected']
        self.assertEqual([r['protect_probability'] for r in random],[.5,.5])
        selected=next(r for r in summary if r['cohort']=='all' and r['role']=='all_unique_anchors' and r['rule']=='hann_pair')
        self.assertAlmostEqual(selected['future_gain_realized'],.35)
        controlled=next(r for r in paired if r['cohort']=='all' and r['role']=='all_unique_anchors' and r['contrast']=='hann_pair_minus_hann_pair__rate_expected')
        self.assertAlmostEqual(controlled['mean_delta'],.375)
        self.assertEqual(controlled['rate_informative_contexts'],1);self.assertEqual(controlled['rate_informative_anchors'],2)
        # Subgroups retain the original .5 inclusion chance, not their own labels/rate.
        group=next(r for r in summary if r['cohort']=='all' and r['role']==stage.prior.ROLES[0] and r['rule']=='hann_pair__rate_expected')
        self.assertEqual(group['protect_probability'],.5)

    def test_observation_whitelist_gt_replacement_and_invalid_forecast(self):
        b=fixture.FrozenClosedLoop();b.setUp();self.addCleanup(b.doCleanups)
        trace=b.collect('generated_shared');i=2
        row={k:v[i].copy() for k,v in trace.items()};h=stage.prior.prior.history_at(trace,b.entry,i)
        expected=stage.online_evidence(row,h,b.model)
        poison=dict(row,ground_truth=np.full((4,),np.nan),future_iou=999.,sequence='secret')
        self.assertEqual(expected,stage.online_evidence(poison,h,b.model))
        with (patch.object(stage.s.e63.actor,'observe',side_effect=AssertionError('New inference')),
              patch.object(stage.s.s105,'checked_image',side_effect=AssertionError('New images'))):
            original=stage.diagnose(trace,b.entry,b.model)
            changed=stage.diagnose(trace,dict(b.entry,ground_truth=np.full_like(b.entry['ground_truth'],999.)),b.model)
        for name in ('x','actions','physical','frame_id'):np.testing.assert_array_equal(original[name],changed[name])
        self.assertFalse(np.array_equal(original['q0_iou'],changed['q0_iou']))
        corrupt=dict(row,conditional_radius=np.zeros(4),generated_active=True)
        with self.assertRaisesRegex(ValueError,'radius'):stage.online_evidence(corrupt,h,b.model)

    def test_complete_cached_workflow_source_immutability_resume_and_tamper(self):
        b=fixture.FrozenClosedLoop();b.setUp();self.addCleanup(b.doCleanups);s=stage.s;prior=stage.prior
        b.observer.assert_frozen=lambda:None
        trace=b.collect('generated_shared');jobs=[];anchors=[];models={(0,'generator',0):b.package}
        for n,cohort in ((9,'historical'),(10,'expanded')):
            for seed in (2312,2313):
                model=copy.deepcopy(b.model);model['seed']=seed
                models[0,'generated_4',seed]=s.io.seal({k:v for k,v in model.items() if k!='fingerprint'})
                job=dict(arm='generated_shared',seed=seed,outer_fold=0,
                    spec=dict(cohort=cohort,unit=dict(sequence=f'uav{n:07d}_00000_s',seed=2171,origin='native',policy='ppo',fold=0)))
                jobs.append(job);anchors.append(dict(context=prior.key(job),cohort=cohort,sequence=job['spec']['unit']['sequence'],seed=seed,
                    frame=2,last=5,roles=[prior.ROLES[0]],anchor_id=prior.key(job)+'__f000002'))
        signature=s.io.seal(dict(models={str(k):v for k,v in models.items()}))['fingerprint']
        with tempfile.TemporaryDirectory() as directory,patch.object(prior,'DESIGN',dict(prior.DESIGN,horizon=4)),contextlib.redirect_stdout(io.StringIO()):
            base=Path(directory);r113=base/'stage113';r115=base/'stage115';root=base/'audit'
            for r in (r113,r115,root):r.mkdir()
            p113=s.io.seal(dict(jobs=jobs));
            for job in jobs:s.c.save(r113/'evaluation'/(prior.key(job)+'.npz'),trace,prior.source_meta(p113,job,models))
            p115=s.io.seal(dict(schema=prior.SCHEMA,design=prior.DESIGN,anchors=anchors,sampling_scope=[],runtime={},runtime_differences=[],
                native_head_sha256='native',source_sha256={},code_sha256={}))
            c=stage.c;c.publish_identical(r115/'manifest.json',p115)
            with patch.object(s.e63.source_experiment,'installed',side_effect=lambda *_:contextlib.nullcontext('native')):
                prior.run(r115,r113,p113,p115,models,lambda _:b.entry,b.old,b.observer,image_reader=b.reader)
            snapshots={str(p):p.read_bytes() for r in (r113,r115) for p in r.rglob('*') if p.is_file()}
            plan=s.io.seal(dict(schema=stage.SCHEMA,design=stage.DESIGN,features=list(stage.FEATURES),rules=list(stage.RULES),
                expected=dict(contexts=4,frames=4*len(trace['frame_id']),anchors=4),source_stage115_fingerprint=stage.SOURCE115,
                source_sha256={p:c.file_hash(Path(p)) for p in snapshots},code_sha256={}))
            c.publish_identical(root/'manifest.json',plan)
            args=(root,r113,p113,r115,p115,plan,models,lambda _:b.entry,b.old)
            with (patch.object(s.e63.actor,'observe',side_effect=AssertionError('New network inference')),
                  patch.object(s.s105,'checked_image',side_effect=AssertionError('Image read')),
                  patch.object(s.common,'train',side_effect=AssertionError('Scorer training')),
                  patch.object(s.common,'fit_generator',side_effect=AssertionError('Generator fitting'))):
                stage.run(*args);stage.check_results(root)
                report=c.c66.sealed(root/'analysis/stage116_decision.json',stage.SCHEMA)
                self.assertEqual(report['counts']['population_scope'],20);self.assertEqual(report['counts']['anchor_decisions'],36)
                self.assertEqual(report['image_reads'],0);self.assertEqual(len(report['artifact_sha256']),8)
                snapshot={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
                with patch.object(stage,'diagnose',side_effect=AssertionError('Rebuilding committed cache')):stage.run(*args)
                self.assertEqual(snapshot,{str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()})
            self.assertEqual(snapshots,{str(p):p.read_bytes() for r in (r113,r115) for p in r.rglob('*') if p.is_file()})
            self.assertEqual(signature,s.io.seal(dict(models={str(k):v for k,v in models.items()}))['fingerprint'])
            next((root/'audit').glob('*.npz')).write_bytes(b'tampered')
            with self.assertRaises(ValueError):stage.check_results(root)


if __name__=='__main__':unittest.main()
