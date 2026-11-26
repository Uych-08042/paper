"""Natural-event selection, exact public grid decode, and spawned CPU workflow."""
import contextlib,io,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tools'),str(Path(__file__).resolve().parents[1]/'tracking')]
import stage210_recovery_audit as run


def make_native():
    import torch
    import test_stage206_spatial_adaptation as f
    from lib.utils.box_ops import clip_box
    torch.set_num_threads(1);helper=f.TorchTests();helper.torch=torch;native=helper.native();rt=native.rt
    f.sample_target=lambda image,*a,**kw:(image,4.,np.zeros(image.shape[:2],bool));f.clip_box=clip_box
    def map_back(self,box,factor):
        cx,cy,w,h=box;origin=np.asarray(self.state,float);half=.5*self.params.search_size/factor
        return [cx+origin[0]+origin[2]/2-half-w/2,cy+origin[1]+origin[3]/2-half-h/2,w,h]
    rt.tracker.map_box_back=types.MethodType(map_back,rt.tracker)
    original=native.step
    def step(image,parent):
        h=[];handle=rt.net.box_head.register_forward_hook(lambda m,a,o:h.append(o))
        try:out=original(image,parent)
        finally:handle.remove()
        grid=run.dense_boxes(rt,h[0][2],h[0][3],4.,parent,image.shape[:2]);cell=int(h[0][0].flatten().argmax())
        out['box']=grid[cell];rt.tracker.state=grid[cell].copy();return out
    native.step=step;return native


def models_for(native,folds):
    result={}
    for fold in folds:
        for seed in run.c.SEEDS:
            base=run.previous.prior.spatial.build(native.rt.torch,8,4,4,seed,'matched_spatial')
            with native.rt.torch.no_grad():base.output.weight.copy_(native.rt.torch.eye(8)*.1)
            result[fold['fold'],seed,'matched']=run.previous.c.view(native.rt.torch,native.rt,base.state_dict(),seed,'matched')
    return result


def fixture_entries(root,specs,truths):
    result={}
    for e in specs:
        seq=e['sequence'];images=np.load(Path(root)/'fixture'/(seq+'.npy'))
        files=[f'{seq}/{i}' for i in range(len(images))]
        reader=lambda dataset,path,images=images:images[int(path.split('/')[1])]
        result[seq]=(files,np.asarray(truths[seq]),None,reader,'fixture')
    return result


def cpu_worker(config,sequences,phase,worker,channel):
    try:
        root=Path(config['root']);plan=run.old.read_sealed(root/'manifest.json');native=make_native();models=models_for(native,plan['folds'])
        entries=fixture_entries(root,plan['entries'],config['truths']);progress=run.previous.prior.WorkerProgress(channel,worker)
        for seq in sequences:
            spec=next(e for e in plan['entries'] if e['sequence']==seq);nt,mt=config['bank'][seq]
            selection=[r for r in plan['selection'] if r['sequence']==seq]
            run.audit_sequence(root,plan,native,models,spec,entries[seq],None,nt,mt,selection,progress,phase)
        progress.flush();channel.put(('complete',worker,0,0))
    except BaseException:
        import traceback
        channel.put(('error',worker,traceback.format_exc(),0));raise


class Tests(unittest.TestCase):
    def test_recovery_indexing_lost_retained_and_source_summary(self):
        gt=np.tile([0.,0.,10.,10.],(10,1));good=gt[1:].copy();bad=good.copy();bad[:,:2]=100.
        native={'boxes':np.vstack((bad[:3],good[3:]))};matched={run.c.SEEDS[0]:{'boxes':bad},run.c.SEEDS[1]:native}
        events,selected,support=run.c.select(dict(sequence='s',group='g'),native,matched,gt)
        self.assertEqual(events[0]['recovery'],4);self.assertEqual(support['controls'],1)
        lost=[r for r in selected if r['kind']=='recovery' and r['stratum']=='lost']
        retained=[r for r in selected if r['kind']=='recovery' and r['stratum']=='retained']
        self.assertEqual([r['frame'] for r in lost],[4,5,6]);self.assertEqual([r['frame'] for r in retained],[4,5,6])
        cells=[]
        for r in selected:
            for parent in ('native','matched'):
                for model in ('native','matched'):
                    out=.8 if parent=='native' else .1
                    cells.append(dict(**r,parent_origin=parent,model=model,output_iou=out,dense_best_iou=.9,
                        fully_covered=True,category=run.c.category(True,.9,out)))
        event_rows,summaries,contrasts=run.c.summarize(cells,[dict(group='g'),dict(group='unsupported')])
        summary=next(r for r in summaries if r['stratum']=='all_recovery' and r['parent_origin']=='matched' and r['model']=='native')
        self.assertEqual(summary['roster_sources'],2);self.assertEqual(summary['supported_sources'],1)
        self.assertEqual(summary['readout_miss_fraction'],1.)
        contrast=next(r for r in contrasts if r['stratum']=='all_recovery' and r['contrast']=='parent_effect_native_model' and r['metric']=='mean_output_iou')
        self.assertAlmostEqual(contrast['mean_delta'],.7);self.assertEqual(contrast['ci_low'],'')

    def test_bundle_categories_and_empty_support(self):
        d,p,t=run.c.source209(run.DEFAULT_BUNDLE)
        self.assertEqual(d['fingerprint'],run.c.SOURCE209);self.assertEqual(len(t),10)
        self.assertEqual(run.c.category(True,.8,.1),'good_grid_missed')
        self.assertEqual(run.c.category(True,.1,.1),'covered_no_good_grid')
        self.assertEqual(run.c.category(False,.1,.1),'not_fully_covered_no_good_grid')
        self.assertEqual(run.c.category(False,.1,.8),'good_output')
        gt=np.tile([0.,0.,10.,10.],(5,1));trace={'boxes':np.tile([100.,100.,10.,10.],(4,1))}
        events,selection,support=run.c.select(dict(sequence='s',group='g'),trace,{s:trace for s in run.c.SEEDS},gt)
        self.assertEqual(support['native_unrecovered'],1);self.assertFalse(selection);self.assertFalse(support['has_diagnostic_support'])

    def test_full_crossed_workflow(self):
        import torch
        native=make_native();groups=[f'uav{i:07d}' for i in range(38)];folds=run.previous.prior.common.source.common.protocol.split_groups(groups)
        models=models_for(native,folds)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'fixture').mkdir();specs=[dict(group=g,sequence=g+'_00000_s') for g in groups]
            rng=np.random.default_rng(210);truths={};bank={};selection=[];support=[];events=[]
            progress=run.NoProgress()
            for spec in specs:
                seq=spec['sequence'];images=rng.normal(size=(11,20,8)).astype(np.float32);np.save(root/'fixture'/(seq+'.npy'),images)
                files=[str(i) for i in range(11)];reader=lambda dataset,path,images=images:images[int(path)]
                nt=run.previous.prior.previous.trace(native,None,None,files,[2.,2.,1.,1.],reader,None,progress)
                fold=next(f['fold'] for f in folds if spec['group'] in f['held']);mt={}
                for seed in run.c.SEEDS:mt[seed]=run.previous.prior.previous.trace(native,models[fold,seed,'matched'],None,files,[2.,2.,1.,1.],reader,None,progress)
                gt=np.vstack(([2.,2.,1.,1.],nt['boxes']));gt[1:4]=[1000.,1000.,1.,1.]
                truths[seq]=gt.tolist();bank[seq]=(nt,mt)
                e,s,u=run.c.select(spec,nt,mt,gt);events.extend(e);selection.extend(s);support.append(u)
            self.assertEqual(len(events),38);self.assertTrue(all(r['native_recoveries']==1 for r in support))
            self.assertTrue(all(r['controls']==1 for r in support))
            entries=fixture_entries(root,specs,truths)
            plan=run.old.sealed(dict(entries=specs,folds=folds,selection=selection,source_support=support,events=events,workers=2,
                same_source_runtime=True,code_sha256={},protocol_sha256=run.file_hash(run.PROTOCOL)))
            run.publish_identical(root/'manifest.json',plan);config=dict(root=str(root),truths=truths,bank=bank)
            actual=run.previous.prior.run_workers
            def workers(config,seqs,phase,n,progress,target=None):return actual(config,seqs,phase,n,progress,target=cpu_worker)
            with patch.object(run.previous.prior,'run_workers',side_effect=workers),contextlib.redirect_stdout(io.StringIO()):
                run.experiment(root,plan,native,models,entries,None,bank,config,[dict(fold=f['fold'],seed=s,new_training_updates=0) for f in folds for s in run.c.SEEDS])
            d=run.old.read_sealed(root/'stage210_decision.json');self.assertEqual(d['new_training_updates'],0)
            self.assertEqual(d['diagnostic_searches'],38*2*2*3*4);self.assertEqual(len(d['table_sha256']),10)
            with contextlib.redirect_stdout(io.StringIO()):run.verify(root)
            # Exact public grid consistency and cleanup remain checked after the workflow.
            spec=specs[0];nt,mt=bank[spec['sequence']];r=entries[spec['sequence']];image=r[3](None,r[0][4])
            result=run.capture(native,None,image,r[3](None,r[0][0]),r[1][0],nt['parent'][3],4)
            np.testing.assert_array_equal(result['grid'][result['cell']],result['box'])
            self.assertEqual(len(native.rt.net.box_head._forward_hooks),0)
            (root/'stage210_summary.csv').write_text('changed')
            with self.assertRaises(ValueError):run.verify(root)


if __name__=='__main__':unittest.main()
