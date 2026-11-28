"""Boundary categories, source weighting and sealed cache integration, no torch."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tools'),str(Path(__file__).resolve().parents[1]/'tracking')]
import stage222_localization_evidence as run


def trace(n=6):
    box=np.tile([90.,90.,20.,20.],(n,1))
    return dict(frame_id=np.arange(1,n+1),parent=box.copy(),reference_boxes=box.copy(),boxes=box.copy(),
                factor=np.full(n,256/80),image_hw=np.tile([200,200],(n,1)),fully_covered=np.ones(n,bool),
                dense_best_iou=np.ones(n),gt_side_network_px=np.full(n,64.))


class Tests(unittest.TestCase):
    def test_reviewed_source_lineage_and_search_config(self):
        d215,p215=run.source.prior.bundle(run.source.prior.DEFAULT_BUNDLE)
        _,p216=run.source.source216(run.source.BUNDLE216)
        self.assertEqual(p216['source_stage215'],d215['fingerprint'])
        self.assertEqual(p215['entries'],p216['entries'])
        self.assertEqual(len(p215['entries']),38)
        self.assertEqual(len(p216['selection']),1824)
        self.assertEqual(p215['runtime']['tracker']['config']['TEST']['SEARCH_SIZE'],256)

    def test_partition_boundaries_and_precedence(self):
        self.assertEqual(run.category(.2,0,True,1),'raw_not_severe')
        self.assertEqual(run.category(0,.5,False,1),'adapted_good')
        self.assertEqual(run.category(0,.49,False,1),'not_fully_covered')
        self.assertEqual(run.category(0,.49,True,.499),'covered_no_good_grid')
        self.assertEqual(run.category(0,.49,True,.5),'covered_readout_miss')

    def test_integer_coverage_and_invalid_support(self):
        tr=trace();gt=np.tile([90.,90.,20.,20.],(7,1));tr['reference_boxes'][0,0]=140
        rr=run.observations(tr,[1,2],gt,256)
        self.assertEqual(rr[0]['category'],'adapted_good')
        self.assertEqual(rr[1]['raw_iou'],1.)
        tr['fully_covered'][0]=False
        with self.assertRaises(ValueError):run.observations(tr,[1],gt,256)
        tr['fully_covered'][0]=True;gt[1,2]=0
        rr=run.observations(tr,[1,2],gt,256)
        self.assertFalse(rr[0]['valid_current']);self.assertIsNone(rr[0]['raw_iou'])
        self.assertEqual(run.reduce_cells(rr)['invalid_current'],1)
        # Touching the nominal crop but falling into excluded right/bottom pixels.
        self.assertFalse(run.covered([180,90,20,20],[160,90,20,20],256/80,[200,200],256))

    def test_source_weighting_zero_denominator_and_empty_primary(self):
        def row(g,k):
            return dict(group=g,origin=run.ORIGINS[0],valid_current=True,raw_iou=0.,adapted_iou=0.,dense_best_iou=1.,
                        readout_gap=1.,raw_error_center_l2=1.,gt_side_network_px=64.,category=k)
        cells=[row('a',run.CLASSES[0])]+[row('b',run.CLASSES[1]) for _ in range(9)]
        good=row('c','raw_not_severe');good['raw_iou']=1.;cells.append(good)
        _,p=run.summarize(cells)
        self.assertEqual(p[0]['mean_fraction'],.5);self.assertEqual(p[1]['mean_fraction'],.5)
        self.assertEqual(p[0]['unsupported_groups'],'c')
        _,p=run.summarize([good]);self.assertTrue(all(r['mean_fraction'] is None for r in p))

    def test_bound_cache_workflow_and_tamper(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);out=root/'out';out.mkdir();r215=root/'215';r216=root/'216'
            groups=['a','b'];specs=[dict(group=g,sequence='seq'+g) for g in groups]
            folds=[dict(fold=0,fit=['a'],held=['b']),dict(fold=1,fit=['b'],held=['a'])]
            p215=run.old.sealed(dict(entries=specs,folds=folds));p216=run.old.sealed(dict(selection=[dict(group=g,sequence='seq'+g,frame=t) for g in groups for t in range(1,7)]))
            decisions={215:dict(artifact_sha256={}),216:dict(artifact_sha256={})}
            gt=np.tile([90.,90.,20.,20.],(8,1));entries={s['sequence']:([],gt.copy()) for s in specs};ref={};files=[]
            def save(k,rel,meta,values):
                rt={215:r215,216:r216}[k];p=rt/rel;run.old.cache(p,meta,lambda:values)
                run.old.record(rt,p,decisions[k]['artifact_sha256']);files.append(p)
            for spec in specs:
                g,seq=spec['group'],spec['sequence'];fold=next(f for f in folds if g in f['held'])
                for seed in (2132,2133):
                    for origin in run.ORIGINS:
                        v=trace(7);frames=list(range(1,7))
                        v['reference_boxes'][:4,:2]=140
                        v['boxes'][1:4,:2]=140
                        v['parent'][1,:2]=150;v['fully_covered'][1]=False
                        v['dense_best_iou'][2]=.4;v['dense_best_iou'][3]=.8
                        meta=dict(plan_fingerprint=p215['fingerprint'],sequence=seq,fold=fold['fold'],seed=seed,arm=origin,phase='full')
                        path=run.source.prior.source.previous.path_for(r215,'full',seq,seed,origin)
                        save(215,path.relative_to(r215).as_posix(),meta,v)
                        meta=dict(plan_fingerprint=p216['fingerprint'],sequence=seq,origin=origin,seed=seed,frames=frames)
                        save(216,f'full/{seq}__s{seed}__{origin}.npz',meta,dict(current_reference=v['reference_boxes'][:6],current_box=v['boxes'][:6]))
                        for t in frames:
                            e=run.source.label(v['reference_boxes'][t-1:t],gt[t:t+1])[0]
                            ref[g,seed,origin,t]=dict(valid_current='True',**{'error_'+a:str(float(z)) for a,z in zip(('dx','dy','log_side'),e)})
            plan=run.old.sealed(dict(schema=run.SCHEMA,entries=specs,folds=folds,search_size=256,code_sha256={},protocol_sha256=run.file_hash(run.PROTOCOL)))
            run.publish_identical(out/'manifest.json',plan);before={p:run.file_hash(p) for p in files}
            # This path never imports/loads a torch model, image or optimizer.
            with patch.dict(sys.modules,{'torch':None}):
                run.experiment(out,plan,dict(stage215_root=str(r215),stage216_root=str(r216)),entries,p215,decisions[215],p216,decisions[216],ref)
            self.assertEqual(before,{p:run.file_hash(p) for p in files})
            run.verify(out)
            decision=run.old.read_sealed(out/'stage222_decision.json');self.assertEqual(len(decision['table_sha256']),7)
            self.assertTrue(all(r['mean_fraction']==.25 and r['supported_groups']==2 for r in decision['primary']))
            with (out/'stage222_cells.csv').open('a') as f:f.write('tamper')
            with self.assertRaises(ValueError):run.verify(out)
            files[0].write_bytes(b'changed')
            with self.assertRaises(ValueError):run.source.bound(r215,decisions[215],files[0].relative_to(r215).as_posix(),{}, {})


if __name__=='__main__':unittest.main()
