"""Check label-free support geometry and exact auditing of frozen commit models."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import stage119_commit_support as stage
import test_stage118_nested_commit as source_fixture


def fixture():
    records,owners,providers,_,_=source_fixture.fixture()
    for r in records:
        r['context']=f"{r['sequence']}__generated_shared__s{r['seed']}"
        r['anchor_id']=r['context']+f"__f{r['frame']:06d}";r['uid']=r['origin']+'__'+r['anchor_id']
        if r['origin']=='stage115':
            old,q0={'good_q0_lost':(.1,.8),'medium_q0_lost':(.1,.3),'q0_rescue':(.8,.1)}[r['current_role']]
            r.update(policy_current=old,q0_current=q0)
    return records,owners,providers


class FixedCommitSupport(unittest.TestCase):
    def setUp(self):
        self.records,self.owners,self.providers=fixture();self.held=self.owners['uav0000001']
        self.fit=stage.prior.training_rows(self.records,self.held,self.providers[0])
        self.package=stage.prior.fit_package(self.fit,self.held,self.providers[0])

    def test_neighbor_ids_ignore_labels_scores_and_duplicate_source_seeds(self):
        z=stage.public_geometry(self.package,self.fit);query=np.zeros(len(stage.prior.FEATURES))
        expected=stage.neighbors(self.fit,z,query)
        self.assertEqual(len(expected),2);self.assertEqual(len({self.fit[i]['group'] for i,_ in expected}),2)
        poisoned=copy.deepcopy(self.fit)
        for r in poisoned:r.update(future_gain=999.,q0_current=-999.,myopic=999.,future=999.,current_role='poison')
        self.assertEqual(expected,stage.neighbors(poisoned,z,query))
        # Exclude the entire video, including its second seed and other frames.
        excluded=self.fit[0]['group'];remaining=stage.neighbors(self.fit,z,z[0],excluded)
        self.assertTrue(all(self.fit[i]['group']!=excluded for i,_ in remaining));self.assertEqual(len(remaining),1)
        tied=np.zeros_like(z);expected=stage.neighbors(self.fit,tied,query)
        for i,_ in expected:self.assertEqual(self.fit[i]['uid'],min(r['uid'] for r in self.fit if r['group']==self.fit[i]['group']))

    def test_reference_is_fit_only_leave_source_out_and_uses_unclipped_distance(self):
        z,k,p95=stage.support_reference(self.package,self.fit);self.assertEqual(k,1)
        independent=[]
        for i,r in enumerate(self.fit):
            allowed=[j for j,b in enumerate(self.fit) if b['group']!=r['group']]
            independent.append(min(np.linalg.norm(z[i]-z[j])/np.sqrt(z.shape[1]) for j in allowed))
        self.assertAlmostEqual(p95,float(np.quantile(independent,.95)))
        poison=copy.deepcopy(self.fit)
        for r in poison:r.update(future_gain=-1000.,policy_current=999.)
        other=stage.support_reference(self.package,poison)
        np.testing.assert_array_equal(z,other[0]);self.assertEqual((k,p95),other[1:])
        query=np.full(z.shape[1],1000.)
        self.assertGreater(stage.neighbors(self.fit,z,query)[0][1],900.)

    def test_fixed_normal_equation_signed_sources_and_feature_terms(self):
        package=copy.deepcopy(self.package);snapshot=copy.deepcopy(package)
        _,y,_,_,mapping,residual=stage.normal_audit(package,self.fit);self.assertLess(residual,1e-10)
        z,k,p95=stage.support_reference(package,self.fit)
        row=copy.deepcopy(self.records[0]);x=np.array([[row[f] for f in stage.prior.FEATURES]])
        score=stage.prior.predict(package,x)[0]
        row.update({name:float(score[i]) for i,name in enumerate(stage.prior.SCORES)})
        details,nearest,sources,features=stage.audit_point(row,package,self.fit,z,k,p95,mapping,y)
        for target in stage.prior.TARGETS:
            self.assertAlmostEqual(sum(r[target+'_contribution'] for r in sources),row[target])
            self.assertAlmostEqual(sum(r[target+'_contribution'] for r in features),row[target])
        self.assertAlmostEqual(sum(r['signed_kernel_sum'] for r in sources),1.)
        self.assertEqual(len(nearest),2);self.assertEqual(len(features),14);self.assertEqual(package,snapshot)
        broken=copy.deepcopy(package);broken['coefficients'][0][1]+=.1
        with self.assertRaisesRegex(ValueError,'normal equation'):stage.normal_audit(broken,self.fit)
        row['future']+=.1
        with self.assertRaisesRegex(ValueError,'decomposition'):stage.audit_point(row,package,self.fit,z,k,p95,mapping,y)

    def test_studies_weights_error_subsets_and_pressure_stay_separate(self):
        rows=[]
        for i,(p,q,score) in enumerate(((.1,.8,1.),(.9,.1,-1.))):
            r=copy.deepcopy(self.records[i]);r.update(memberships='uniform|stratified',uniform_weight=.5,stratified_weight=.9 if i==0 else .1,
                policy_future=p,q0_future=q,future_gain=q-p,myopic=-1.,future=score,myopic_constant=0.,future_constant=0.,
                distance_beyond_reference=False,gate_clip_fraction=0.,neighbor_mean_distance=1.,neighbor_sign_mixed=True,neighbor_material_mixed=True,
                neighbor_future_mean=0.,future_prediction_error=score-(q-p),future_realized_gain=(q-p) if score>0 else 0.,wrong_commit=False,missed_gain=False)
            rows.append(r)
        pressure=copy.deepcopy(rows[0]);pressure.update(origin='stage115',memberships='pressure',current_role='q0_rescue',policy_future=.9,q0_future=.1,
            future_gain=-.8,future_realized_gain=-.8,wrong_commit=True,future_prediction_error=1.8)
        rows.append(pressure);general,support=stage.summaries(rows)
        def get(study):return next(r for r in general if r['study']==study and r['cohort']=='all' and r['target']=='future')
        self.assertAlmostEqual(get('ordinary_uniform')['realized_gain'],.35)
        self.assertAlmostEqual(get('ordinary_stratified')['realized_gain'],.63)
        self.assertAlmostEqual(get('pressure')['realized_gain'],-.8)
        self.assertTrue(any(r['study']=='pressure' and r['cohort']=='all' and r['subset']=='wrong_commit' and r['anchors']==1 for r in support))
        empty=next(r for r in support if r['study']=='ordinary_uniform' and r['cohort']=='all' and r['subset']=='wrong_commit')
        self.assertEqual(empty['anchors'],0);self.assertIsNone(empty['neighbor_future_mean'])

    def test_zero_clip_normalization_and_malformed_public_geometry(self):
        row=copy.deepcopy(self.records[0]);row[stage.prior.FEATURES[0]]=1000.
        score=stage.prior.predict(self.package,np.array([[row[k] for k in stage.prior.FEATURES]]))[0]
        row.update({name:float(score[i]) for i,name in enumerate(stage.prior.SCORES)})
        _,y,_,_,mapping,_=stage.normal_audit(self.package,self.fit);z,k,p95=stage.support_reference(self.package,self.fit)
        details,_,_,_=stage.audit_point(row,self.package,self.fit,z,k,p95,mapping,y)
        self.assertGreater(details['gate_clip_fraction'],0.);self.assertTrue(details['distance_beyond_reference'])
        with self.assertRaisesRegex(ValueError,'geometry'):stage.neighbors(self.fit,z,np.full(len(stage.prior.FEATURES),np.nan))

    def test_complete_fixed_artifact_workflow_no_training_source_mutation_or_resume_drift(self):
        p=stage.prior
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            base=Path(directory);source=base/'source118';root=base/'audit119';source.mkdir();root.mkdir()
            expected118=dict(groups=6,unique_points=30,membership_points=42,uniform_fit_points=24,pressure_points=6)
            p118=p.io.seal(dict(schema=p.SCHEMA,design=p.DESIGN,features=list(p.FEATURES),expected=expected118,owners=self.owners,providers=self.providers,
                source_stage117_fingerprint=p.SOURCE117,source_sha256={},code_sha256={}))
            p.c.publish_identical(source/'manifest.json',p118);p.run(source,p118,self.records,self.owners,self.providers)
            p118=p.c.c66.sealed(source/'manifest.json',p.SCHEMA);records=stage.load_records(source,p118)
            before={str(f):f.read_bytes() for f in source.rglob('*') if f.is_file()}
            expected=dict(groups=6,unique_points=30,fit_rows=48,neighbor_rows=60)
            plan=p.io.seal(dict(schema=stage.SCHEMA,design=stage.DESIGN,source_stage118_fingerprint=stage.SOURCE118,expected=expected,
                source_sha256={f:p.c.file_hash(Path(f)) for f in before},code_sha256={}))
            p.c.publish_identical(root/'manifest.json',plan)
            with (patch.object(p,'fit_package',side_effect=AssertionError('New gate fitting')),
                  patch.object(p.s.e63.actor,'observe',side_effect=AssertionError('SGLA inference')),
                  patch.object(p.s.s105,'checked_image',side_effect=AssertionError('Image read')),
                  patch.object(p.s.common,'train',side_effect=AssertionError('New scorer training')),
                  patch.object(p.s.common,'fit_generator',side_effect=AssertionError('New generator training'))):
                stage.run(root,source,p118,plan,records);stage.check_results(root)
                report=p.c.c66.sealed(root/'analysis/stage119_decision.json',stage.SCHEMA)
                self.assertEqual(report['new_models'],0);self.assertFalse(report['training_performed'])
                self.assertEqual(report['fixed_cpu_prediction_rows_recomputed'],78);self.assertEqual(report['counts']['feature_terms'],420)
                saved={str(f):f.read_bytes() for f in root.rglob('*') if f.is_file()}
                stage.run(root,source,p118,plan,records)
                self.assertEqual(saved,{str(f):f.read_bytes() for f in root.rglob('*') if f.is_file()})
            self.assertEqual(before,{str(f):f.read_bytes() for f in source.rglob('*') if f.is_file()})
            (root/'analysis/stage119_scope.csv').write_bytes(b'tampered')
            with self.assertRaises(ValueError):stage.check_results(root)


if __name__=='__main__':unittest.main()
