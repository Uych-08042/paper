"""Verify causal RGB evidence, isolated support expansion and factorial commit comparisons."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import stage120_commit_factorial as stage
import test_stage119_commit_support as fixture


def pixels():
    rng=np.random.default_rng(12);image=np.zeros((32,32,3),np.uint8)
    image[4:12,4:12]=rng.integers(0,80,(8,8,3),dtype=np.uint8);image[4:12,4:12,0]+=150
    image[20:28,20:28]=rng.integers(0,80,(8,8,3),dtype=np.uint8);image[20:28,20:28,2]+=150
    return image


class SupportAppearance(unittest.TestCase):
    def setUp(self):
        self.records,self.owners,self.providers=fixture.fixture();self.held=self.owners['uav0000001']
        rng=np.random.default_rng(45);self.bank={r['uid']:rng.normal(size=len(stage.appearance.FEATURES)) for r in self.records}
        fit=stage.gate.training_rows(self.records,self.held,self.providers[0]);self.baseline=stage.gate.fit_package(fit,self.held,self.providers[0])

    def test_rgb_identity_cues_and_unavailable_flat_partial_roi_masks(self):
        image=pixels();task=dict(frame=12,initial_box=[4,4,8,8],boxes=[[4,4,8,8],[20,20,8,8]],recent_frame=8,recent_box=[4,4,8,8])
        x=stage.appearance.appearance_vector(image,image,image,task);self.assertEqual(x.shape,(24,))
        named=dict(zip(stage.appearance.FEATURES,x));self.assertAlmostEqual(named['template_hist_q0'],1.)
        self.assertGreater(named['template_hist_advantage'],.2);self.assertGreater(named['template_ncc_advantage'],.2)
        self.assertAlmostEqual(named['recent_age'],4/30);self.assertEqual(named['recent_exists'],1.)
        flat=np.full_like(image,100);d=stage.appearance.describe(flat,[4,4,8,8]);self.assertIsNone(d['ncc']);self.assertAlmostEqual(float(d['hist']@d['hist']),1.)
        missing=dict(task,recent_frame=-1,recent_box=None,boxes=[[-10,-10,2,2],[20,20,8,8]])
        values=dict(zip(stage.appearance.FEATURES,stage.appearance.appearance_vector(image,image,None,missing)))
        self.assertEqual(values['recent_exists'],0.);self.assertEqual(values['template_hist_q0_valid'],0.);self.assertTrue(np.isfinite(list(values.values())).all())
        with self.assertRaisesRegex(ValueError,'reference'):stage.appearance.appearance_vector(image,image,image,dict(task,recent_frame=12))

    def test_reference_and_tasks_ignore_future_gt_future_memory_and_future_boxes(self):
        a=stage.appearance;frame=12;bank=np.array([[4,4,8,8],[3,3,8,8],[20,20,8,8],[2,2,8,8]],float)
        trace=dict(candidate_boxes=np.tile(bank,(20,1,1)),selected=np.full(20,2),boxes=np.tile(bank[2],(20,1)),memory_write=np.zeros(20,bool))
        trace['memory_write'][3]=trace['memory_write'][7]=trace['memory_write'][15]=True
        entry=dict(initial_box=[4,4,8,8],frames=[dict(file=str(f),sha256='fixed',height=32,width=32) for f in range(21)],ground_truth=np.zeros((21,4)))
        point=dict(frame_id=np.array([frame]),candidate_boxes=bank[None],selected=np.array([2]))
        task=a.make_task('u',frame,point,trace,entry);self.assertEqual(task['recent_frame'],8)
        poison=copy.deepcopy(trace);poison['boxes'][frame:]=999.;poison['memory_write'][frame:]=True
        self.assertEqual(task,a.make_task('u',frame,dict(point,ground_truth=np.full((1,4),999.)),poison,dict(entry,ground_truth=np.full((21,4),999.))))
        reads=[]
        def reader(root,item):reads.append(item['file']);return pixels()
        result=a.collect('root',[task,task],reader)
        self.assertEqual(set(reads),{'0','8','12'});self.assertEqual(int(result['image_reads']),3);self.assertEqual(result['x'].shape,(2,24))
        wrong=copy.deepcopy(trace);wrong['candidate_boxes'][frame-1,0,0]+=1
        with self.assertRaisesRegex(ValueError,'drifted'):a.make_task('u',frame,point,wrong,entry)

    def test_expanded_support_keeps_provider_and_all_test_video_labels_out(self):
        provider=self.providers[0];fit=stage.fitting_rows(self.records,self.held,provider,'expanded')
        self.assertEqual(len(fit),10);self.assertEqual({r['group'] for r in fit},{'uav0000002','uav0000003'})
        self.assertEqual(sum(r['origin']=='stage115' for r in fit),2)
        w=stage.weights(fit);self.assertAlmostEqual(w.sum(),1.);self.assertAlmostEqual(sum(v for r,v in zip(fit,w) if r['origin']=='stage115'),.25)
        for group in {r['group'] for r in fit}:self.assertAlmostEqual(sum(v for r,v in zip(fit,w) if r['group']==group),.5)
        original=stage.package_for(self.records,self.bank,self.held,provider,self.baseline)
        poison=copy.deepcopy(self.records);bank=copy.deepcopy(self.bank)
        for r in poison:
            if r['group']==self.held['group'] or r['outer_fold']!=self.held['outer_fold']:
                r['future_gain']=1000.;bank[r['uid']]=np.full(24,1000.)
        self.assertEqual(original,stage.package_for(poison,bank,self.held,provider,self.baseline))
        changed=copy.deepcopy(self.records);next(r for r in changed if r['origin']=='stage115' and r['group']=='uav0000002')['future_gain']+=.4
        altered=stage.package_for(changed,self.bank,self.held,provider,self.baseline)
        self.assertEqual(original['heads']['uniform_appearance'],altered['heads']['uniform_appearance'])
        self.assertNotEqual(original['heads']['expanded_appearance']['input_fingerprint'],altered['heads']['expanded_appearance']['input_fingerprint'])

    def test_prepare_reads_original_point_receipts_and_binds_only_used_images(self):
        a=stage.appearance;s=stage.s;c=stage.c
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);r113=base/'113';r115=base/'115';r117=base/'117';image_root=base/'images'
            jobs=[dict(arm='generated_shared',outer_fold=0,seed=seed,spec=dict(cohort='historical',unit=dict(sequence='uav0000001_00000_s'))) for seed in (2312,2313)]
            models={(0,'generator',0):dict(generator={},fingerprint='generator')}
            for seed in (2312,2313):models[0,'generated_4',seed]=dict(fingerprint=str(seed))
            p113=stage.io.seal(dict(jobs=jobs));p115=stage.io.seal(dict(anchors=[]));p117=stage.io.seal(dict(anchors=[]));records=[]
            bank=np.array([[4,4,8,8],[3,3,8,8],[20,20,8,8],[2,2,8,8]],float)
            trace=dict(frame_id=np.arange(1,21),candidate_boxes=np.tile(bank,(20,1,1)),selected=np.full(20,2),boxes=np.tile(bank[2],(20,1)),memory_write=np.zeros(20,bool))
            trace['memory_write'][7]=True
            entry=dict(sequence='uav0000001_00000_s',initial_box=[4,4,8,8],frames=[dict(file=str(f)+'.png',sha256='sha'+str(f),height=32,width=32) for f in range(21)],ground_truth=np.full((21,4),999.))
            for job in jobs:
                context=a.branch.key(job);c.save(r113/'evaluation'/(context+'.npz'),trace,a.branch.source_meta(p113,job,models))
                for origin,frame,p in [('stage115',10,p115),('stage117',11,p117)]:
                    anchor=dict(anchor_id=context+f'__f{frame:06d}',frame=frame);p['anchors'].append(anchor)
                    records.append(dict(uid=origin+'__'+anchor['anchor_id'],origin=origin,anchor_id=anchor['anchor_id'],context=context,frame=frame))
            p115=stage.io.seal({k:v for k,v in p115.items() if k!='fingerprint'});p117=stage.io.seal({k:v for k,v in p117.items() if k!='fingerprint'})
            for r in records:
                source,p=(r115,p115) if r['origin']=='stage115' else (r117,p117);anchor=next(x for x in p['anchors'] if x['anchor_id']==r['anchor_id'])
                point={k:v[r['frame']-1:r['frame']] for k,v in trace.items()}
                c.save(source/'points'/(r['anchor_id']+'.npz'),point,dict(plan_fingerprint=p['fingerprint'],anchor=anchor,kind='shared_point'))
            tasks,images=a.prepare(records,r113,p113,models,r115,p115,r117,p117,lambda _:entry,image_root)
            self.assertEqual(len(tasks),2);self.assertEqual(sum(len(v) for v in tasks.values()),4)
            self.assertEqual({Path(k).name for k in images},{'0.png','8.png','10.png','11.png'})
            self.assertTrue(all(t['recent_frame']==8 for values in tasks.values() for t in values))
            self.assertFalse(any('ground_truth' in t for values in tasks.values() for t in values))

    def test_shuffle_is_label_free_within_video_and_matches_capacity_normalization(self):
        fit=stage.fitting_rows(self.records,self.held,self.providers[0],'expanded');indices=stage.permutation(fit,self.held['group'],'expanded')
        self.assertEqual(sorted(indices.tolist()),list(range(len(fit))))
        self.assertTrue(all(fit[i]['group']==fit[j]['group'] for i,j in enumerate(indices)))
        a=stage.fit_head(fit,self.bank,self.held['group'],'expanded','appearance');b=stage.fit_head(fit,self.bank,self.held['group'],'expanded','shuffled')
        self.assertEqual(a['center'],b['center']);self.assertEqual(a['scale'],b['scale']);self.assertEqual(len(a['beta']),38);self.assertEqual(len(b['beta']),38)
        self.assertGreater(b['shuffled_value_change_fraction'],0.)
        poison=copy.deepcopy(fit)
        for r in poison:r['future_gain']+=1000.
        np.testing.assert_array_equal(indices,stage.permutation(poison,self.held['group'],'expanded'))
        with self.assertRaisesRegex(ValueError,'inference'):stage.predict(a,np.full((1,37),np.nan))

    def test_factorial_interaction_study_weights_and_same_rate_controls(self):
        records=copy.deepcopy(self.records[:2]);pred=[]
        for i,r in enumerate(records):
            r.update(memberships='uniform|stratified',uniform_weight=.5,stratified_weight=.9 if i==0 else .1,policy_future=.1 if i==0 else .9,
                q0_future=.8 if i==0 else .1,future_gain=.7 if i==0 else -.8,myopic=-1.)
            values={name:-1. for name in stage.SCORES};values['expanded_appearance']=1. if i==0 else -1.;values['uniform_relative']=-1. if i==0 else 1.
            pred.append(dict(uid=r['uid'],**values))
        decisions=stage.choices(records,pred);summary,pairs,_,_=stage.summaries(decisions)
        for study,gain,interaction,rate in [('ordinary_uniform',.35,-.05,.5),('ordinary_stratified',.63,.55,.9)]:
            get=lambda rule:next(r for r in summary if r['study']==study and r['cohort']=='all' and r['rule']==rule)
            self.assertAlmostEqual(get('expanded_appearance')['realized_gain'],gain)
            self.assertAlmostEqual(get('expanded_appearance__rate_expected')['protect_probability'],rate)
            pair=next(r for r in pairs if r['study']==study and r['cohort']=='all' and r['contrast']=='support_by_appearance_interaction')
            self.assertAlmostEqual(pair['mean_delta'],interaction)
        for r in decisions:self.assertAlmostEqual(r['regret'],r['harm_cost']+r['missed_gain'])

    def test_complete_pixel_and_crossfit_workflow_frozen_baseline_resume_and_tamper(self):
        p=stage.gate
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            base=Path(directory);source=base/'stage118';root=base/'stage120';source.mkdir();root.mkdir()
            expected118=dict(groups=6,unique_points=30,membership_points=42,uniform_fit_points=24,pressure_points=6)
            p118=p.io.seal(dict(schema=p.SCHEMA,design=p.DESIGN,features=list(p.FEATURES),expected=expected118,owners=self.owners,providers=self.providers,
                source_stage117_fingerprint=p.SOURCE117,source_sha256={},code_sha256={}))
            p.c.publish_identical(source/'manifest.json',p118);p.run(source,p118,self.records,self.owners,self.providers)
            p118=p.c.c66.sealed(source/'manifest.json',p.SCHEMA);records=stage.audit.load_records(source,p118);tasks={}
            for r in records:
                f=r['frame'];task=dict(uid=r['uid'],frame=f,initial_box=[4,4,8,8],boxes=[[4,4,8,8],[20,20,8,8]],recent_frame=f-1,recent_box=[4,4,8,8],
                    images={str(i):dict(file=str(i),sha256='toy',height=32,width=32) for i in (0,f,f-1)})
                tasks.setdefault(r['context'],[]).append(task)
            for part in tasks.values():part.sort(key=lambda t:t['uid'])
            image_reads=sum(len({f for t in part for f in t['images']}) for part in tasks.values())
            before={str(f):f.read_bytes() for f in source.rglob('*') if f.is_file()}
            plan=p.io.seal(dict(schema=stage.SCHEMA,design=stage.DESIGN,appearance_design=stage.appearance.DESIGN,source_stage119_fingerprint=stage.SOURCE119,
                expected=dict(groups=6,points=30,memberships=42,contexts=12,image_reads=image_reads),appearance_tasks=tasks,image_root='toy',image_sha256={},
                source_sha256={f:p.c.file_hash(Path(f)) for f in before},code_sha256={}))
            p.c.publish_identical(root/'manifest.json',plan);reader=lambda *_:pixels();original=stage.choices
            def evaluated(*args):
                self.assertEqual(len(list((root/'predictions').glob('*.receipt.json'))),6)
                for path in (root/'predictions').glob('*.npz'):
                    with np.load(path,allow_pickle=False) as data:self.assertEqual(set(data.files),{'base','appearance','scores','metadata_json'})
                return original(*args)
            with (patch.object(stage,'choices',side_effect=evaluated),patch.object(p,'fit_package',side_effect=AssertionError('Old gate refit')),
                  patch.object(p.s.e63.actor,'observe',side_effect=AssertionError('New SGLA inference')),
                  patch.object(p.s.common,'train',side_effect=AssertionError('Scorer training')),
                  patch.object(p.s.common,'fit_generator',side_effect=AssertionError('Generator training'))):
                stage.run(root,source,p118,plan,records,reader);stage.check_results(root)
                report=p.c.c66.sealed(root/'analysis/stage120_decision.json',stage.SCHEMA)
                self.assertEqual(report['new_models'],36);self.assertEqual(report['baseline_prediction_error'],0.);self.assertEqual(report['counts']['policy_decisions'],42*19)
                self.assertEqual(report['primary']['contrast'],stage.DESIGN['primary'])
                snapshot={str(f):f.read_bytes() for f in root.rglob('*') if f.is_file()}
                with patch.object(stage,'fit_head',side_effect=AssertionError('Repeated training')):
                    stage.run(root,source,p118,plan,records,lambda *_:self.fail('Repeated image read'))
                self.assertEqual(snapshot,{str(f):f.read_bytes() for f in root.rglob('*') if f.is_file()})
            self.assertEqual(before,{str(f):f.read_bytes() for f in source.rglob('*') if f.is_file()})
            next((root/'appearance').glob('*.npz')).write_bytes(b'tampered')
            with self.assertRaises(ValueError):stage.check_results(root)


if __name__=='__main__':unittest.main()
