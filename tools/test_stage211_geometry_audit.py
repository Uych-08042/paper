"""Center/size construction, factorial statistics and actual spawned CPU audit."""
import contextlib,io,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tools'),str(Path(__file__).resolve().parents[1]/'tracking')]
import stage211_geometry_audit as run
import test_stage210_recovery_audit as f


def cpu_worker(config,sequences,phase,worker,channel):
    try:
        root=Path(config['root']);plan=run.old.read_sealed(root/'manifest.json');native=f.make_native();models=f.models_for(native,plan['folds'])
        entries=f.fixture_entries(root,plan['entries'],config['truths']);progress=run.runtime.WorkerProgress(channel,worker)
        for seq in sequences:
            spec=next(e for e in plan['entries'] if e['sequence']==seq)
            ends=run.endpoints(config,spec,config['source_plan'],config['source_decision'],config['source_tables'],{})
            chosen=[r for r in plan['selection'] if r['sequence']==seq]
            run.audit_sequence(root,plan,native,models,spec,entries[seq],None,ends,chosen,progress,phase)
        progress.flush();channel.put(('complete',worker,0,0))
    except BaseException:
        import traceback
        channel.put(('error',worker,traceback.format_exc(),0));raise


class Tests(unittest.TestCase):
    def test_source_and_center_size_not_top_left(self):
        d,p,t=run.c.source210(run.DEFAULT_BUNDLE)
        self.assertEqual(d['fingerprint'],run.c.SOURCE210);self.assertEqual(len(t),10)
        a=np.array([10.125,30.75,50.5,22.25]);b=np.array([2.,6.,3.,8.])
        for mode in run.c.MODES:
            parent=run.c.parent(a,b,mode);center=a if mode[0]=='n' else b;size=a if mode[1]=='n' else b
            np.testing.assert_allclose(parent[:2]+parent[2:]/2,center[:2]+center[2:]/2,rtol=0,atol=1e-13)
            np.testing.assert_array_equal(parent[2:],size[2:])
        np.testing.assert_array_equal(run.c.parent(a,b,'nn'),a);np.testing.assert_array_equal(run.c.parent(a,b,'mm'),b)
        self.assertFalse(np.array_equal(run.c.parent(a,b,'nm')[:2],a[:2]))
        with self.assertRaises(ValueError):run.c.parent(a,[0,0,0,1],'nm')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'changed.txt';path.write_bytes(run.DEFAULT_BUNDLE.read_bytes()+b'x')
            with self.assertRaises(ValueError):run.c.source210(path)

    def test_factorial_decomposition_and_empty_support(self):
        cells=[]
        for g in ('a','b','c'):
            for seed in run.c.SEEDS:
                for mode in run.c.MODES:
                    x,y=int(mode[0]=='n'),int(mode[1]=='n');out=.3+.25*x+.1*y+.1*x*y
                    for offset in range(3):cells.append(dict(sequence=g,group=g,seed=seed,kind='recovery',event=0,stratum='lost',offset=offset,
                        mode=mode,output_iou=out,dense_best_iou=.9,fully_covered=True,gt_width_network_px=10.,gt_height_network_px=20.,
                        category=run.base.c.category(True,.9,out)))
        e,s,contrasts=run.c.summarize(cells,[{}, {}, {}, {}])
        def delta(name):return next(r for r in contrasts if r['stratum']=='all_recovery' and r['contrast']==name and r['metric']=='mean_output_iou')['mean_delta']
        self.assertAlmostEqual(delta('center_order_average'),.3);self.assertAlmostEqual(delta('size_order_average'),.15)
        self.assertAlmostEqual(delta('center_by_size_interaction'),.1)
        self.assertAlmostEqual(delta('center_order_average')+delta('size_order_average'),delta('total_native_parent'))
        self.assertEqual(next(r for r in s if r['stratum']=='normal')['supported_sources'],0)

    def test_full_endpoint_binding_and_spawned_workflow(self):
        native=f.make_native();groups=[f'uav{i:07d}' for i in range(38)];folds=run.runtime.common.source.common.protocol.split_groups(groups)
        models=f.models_for(native,folds);rng=np.random.default_rng(211)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'211';source=Path(tmp)/'210';(root/'fixture').mkdir(parents=True)
            specs=[dict(group=g,sequence=g+'_00000_s') for g in groups];truths={};bank={};events=[];selection=[];support=[]
            for spec in specs:
                seq=spec['sequence'];images=rng.normal(size=(11,20,8)).astype(np.float32);np.save(root/'fixture'/(seq+'.npy'),images)
                files=[str(i) for i in range(11)];reader=lambda dataset,path,images=images:images[int(path)]
                nt=run.base.previous.prior.previous.trace(native,None,None,files,[2.,2.,1.,1.],reader,None,run.base.NoProgress())
                fold=next(f['fold'] for f in folds if spec['group'] in f['held']);mt={}
                for seed in run.c.SEEDS:mt[seed]=run.base.previous.prior.previous.trace(native,models[fold,seed,'matched'],None,files,[2.,2.,1.,1.],reader,None,run.base.NoProgress())
                gt=np.vstack(([2.,2.,1.,1.],nt['boxes']));gt[1:4]=[1000.,1000.,1.,1.]
                truths[seq]=gt.tolist();bank[seq]=(nt,mt)
                e,s,u=run.base.c.select(spec,nt,mt,gt);events.extend(e);selection.extend(s);support.append(u)
            entries=f.fixture_entries(root,specs,truths);source_plan=run.old.sealed(dict(entries=specs,folds=folds,selection=selection))
            artifacts={};scopes=[dict(fold=str(fold['fold']),seed=str(seed),state_sha256=run.old.model_digest(models[fold['fold'],seed,'matched'])) for fold in folds for seed in run.c.SEEDS]
            for spec in specs:
                nt,mt=bank[spec['sequence']];chosen=[r for r in selection if r['sequence']==spec['sequence']]
                paths=run.base.audit_sequence(source,source_plan,native,models,spec,entries[spec['sequence']],None,nt,mt,chosen,run.base.NoProgress())
                for path,*_ in paths:run.old.record(source,path,artifacts)
            decision=dict(artifact_sha256=artifacts);tables=dict(model_scope=scopes)
            config=dict(root=str(root),diagnostic_root=str(source),source_plan=source_plan,source_decision=decision,source_tables=tables,truths=truths)
            end_bank={spec['sequence']:run.endpoints(config,spec,source_plan,decision,tables,{}) for spec in specs}
            plan=run.old.sealed(dict(entries=specs,folds=folds,selection=selection,events=events,source_support=support,
                same_source_runtime=True,workers=2,code_sha256={},protocol_sha256=run.file_hash(run.PROTOCOL)))
            run.publish_identical(root/'manifest.json',plan)
            source_before={p:run.file_hash(p) for p in source.rglob('*') if p.is_file()}
            actual=run.runtime.run_workers
            def workers(config,seqs,phase,n,progress,target=None):return actual(config,seqs,phase,n,progress,target=cpu_worker)
            with patch.object(run.runtime,'run_workers',side_effect=workers),contextlib.redirect_stdout(io.StringIO()):
                run.experiment(root,plan,native,models,entries,None,end_bank,config,scopes)
            d=run.old.read_sealed(root/'stage211_decision.json')
            self.assertEqual(d['diagnostic_searches'],1824);self.assertEqual(d['new_training_updates'],0);self.assertEqual(len(d['table_sha256']),10)
            with contextlib.redirect_stdout(io.StringIO()):run.verify(root)
            self.assertTrue(all(run.file_hash(p)==sha for p,sha in source_before.items()))
            # A source-bound NPZ must reject a different expected state identity.
            bad=dict(model_scope=[dict(r,state_sha256='wrong') for r in scopes])
            with self.assertRaises(ValueError):run.endpoints(config,specs[0],source_plan,decision,bad,{})
            progress=run.base.NoProgress()
            with patch.object(native,'step',side_effect=AssertionError('Must reuse')):
                run.audit_sequence(root,plan,native,models,specs[0],entries[specs[0]['sequence']],None,end_bank[specs[0]['sequence']],
                    [r for r in selection if r['sequence']==specs[0]['sequence']],progress)
            (root/'stage211_summary.csv').write_text('changed')
            with self.assertRaises(ValueError):run.verify(root)


if __name__=='__main__':unittest.main()
