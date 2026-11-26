"""Fixed causal inputs, geometry/loss gradients, fit-only training and spawned OOF."""
import contextlib,io,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tools'),str(Path(__file__).resolve().parents[1]/'tracking')]
import stage214_search_state as run


class Silent:
    reused=0
    def advance(self,*a,**k):pass


def cpu_worker(config,sequences,phase,worker,channel):
    try:
        import test_stage210_recovery_audit as f
        root=Path(config['root']);plan=run.old.read_sealed(root/'manifest.json');native=f.make_native();adapters=f.models_for(native,plan['folds'])
        updaters=run.load_updaters(root,plan,native.rt.torch,next(native.rt.net.parameters()).device)
        entries=f.fixture_entries(root,plan['entries'],config['truths']);progress=run.runtime.WorkerProgress(channel,worker)
        for seq in sequences:run.evaluate_video(root,plan,native,adapters,updaters,next(e for e in plan['entries'] if e['sequence']==seq),entries[seq],None,progress,phase)
        progress.flush();channel.put(('complete',worker,0,0))
    except BaseException:
        import traceback
        channel.put(('error',worker,traceback.format_exc(),0));raise


class Tests(unittest.TestCase):
    def test_reviewed_source_and_state_geometry(self):
        d,p,t=run.c.source213(run.DEFAULT_BUNDLE);self.assertEqual(d['fingerprint'],run.c.SOURCE213);self.assertEqual(len(t),7)
        raw=np.asarray([11.5,13.25,20.5,10.25]);z=np.zeros(3)
        np.testing.assert_array_equal(run.c.parent(raw,z,[100,200]),raw)
        shifted=run.c.parent(raw,[.5,-.25,np.log(2)],[100,200])
        np.testing.assert_allclose(shifted[2:],raw[2:]*2)
        np.testing.assert_allclose(shifted[:2]+shifted[2:]/2,raw[:2]+raw[2:]/2+np.sqrt(raw[2:].prod())*np.asarray([.5,-.25]))
        self.assertAlmostEqual(shifted[2]/shifted[3],raw[2]/raw[3])
        bounded=run.c.parent(raw,[1e9,-1e9,1e9],[100,200]);self.assertTrue(np.isfinite(bounded).all())
        np.testing.assert_allclose(bounded[:2]+bounded[2:]/2,[200,0],atol=1e-10)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.txt';path.write_bytes(run.DEFAULT_BUNDLE.read_bytes()+b'x')
            with self.assertRaises(ValueError):run.c.source213(path)

    def test_zero_update_loss_gradient_and_label_noninterference(self):
        import torch
        import test_stage210_recovery_audit as f
        torch.set_num_threads(1);native=f.make_native();folds=[dict(fold=0,fit=[],held=[])];adapter=f.models_for(native,folds)[0,2132,'matched']
        model=run.c.build(torch,24,np.zeros(24),np.ones(24),2132).eval().requires_grad_(False)
        images=np.random.default_rng(214).normal(size=(4,20,8)).astype(np.float32);files=list(map(str,range(4)));reader=lambda d,p:images[int(p)]
        before={i:len(b._forward_hooks) for i,b in enumerate(native.rt.backbone.blocks)}
        a=run.rollout(native,adapter,'nn',None,files,[2.,2.,1.,1.],reader,None,Silent())
        b=run.rollout(native,adapter,'forecast',model,files,[2.,2.,1.,1.],reader,None,Silent())
        for k in ('boxes','reference_boxes','parent','next_parent','block_calls','head_calls','layer'):np.testing.assert_array_equal(a[k],b[k])
        run.c.audit(b,[2,2,1,1],3,'forecast')
        gt=np.tile([1.,1.,1.,1.],(4,1));labelled=run.rollout(native,adapter,'forecast',model,files,[2.,2.,1.,1.],reader,None,Silent(),truth=gt)
        for k in a:
            if k not in ('pipeline_ms','updater_calls'):np.testing.assert_array_equal(labelled[k],b[k])
        self.assertEqual(before,{i:len(bl._forward_hooks) for i,bl in enumerate(native.rt.backbone.blocks)})
        self.assertEqual(len(native.rt.net.box_head._forward_hooks),0)
        # Repeat with input-dependent nonzero actions; zero initialization cannot mask leakage.
        with torch.no_grad():model.network[-1].weight.fill_(.01)
        other_gt=gt.copy();other_gt[1:,:2]+=50
        p=run.rollout(native,adapter,'forecast',model,files,[2.,2.,1.,1.],reader,None,Silent(),truth=gt)
        q=run.rollout(native,adapter,'forecast',model,files,[2.,2.,1.,1.],reader,None,Silent(),truth=other_gt)
        for k in ('boxes','parent','reference_boxes','next_parent','state_delta','layer'):np.testing.assert_array_equal(p[k],q[k])
        delta=torch.zeros((1,3),requires_grad=True);raw=torch.tensor([[0.,0.,10.,10.]])
        target=torch.tensor([[80.,50.,10.,10.]]);hw=torch.tensor([[100.,100.]])
        f0,reg,cover=run.c.loss(torch,delta,raw,target,hw,4.,'forecast');f1,_,_=run.c.loss(torch,delta,raw,target,hw,4.,'coverage')
        self.assertGreater(float(cover),0);self.assertAlmostEqual(float(f1-f0),float(cover),places=5)
        f1.backward();self.assertGreater(float(delta.grad.abs().max()),0)
        same,_,inactive=run.c.loss(torch,torch.zeros((1,3)),raw,raw,hw,4.,'coverage')
        self.assertEqual(float(inactive),0.);self.assertEqual(float(same),0.)
        bad=dict(b,parent=b['parent'].copy());bad['parent'][1,0]+=.1
        with self.assertRaises(AssertionError):run.c.audit(bad,[2,2,1,1],3,'forecast')

    def test_complete_fit_and_spawned_pipeline(self):
        import test_stage210_recovery_audit as f
        native=f.make_native();groups=[f'uav{i:07d}' for i in range(38)];folds=run.c.protocol.split_groups(groups);adapters=f.models_for(native,folds)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'fixture').mkdir();specs=[dict(group=g,sequence=g+'_00000_s') for g in groups];truths={};source_bank={};rng=np.random.default_rng(214)
            for spec in specs:
                seq=spec['sequence'];images=rng.normal(size=(4,20,8)).astype(np.float32);np.save(root/'fixture'/(seq+'.npy'),images)
                truths[seq]=np.tile([2.,2.,1.,1.],(4,1)).tolist();files=list(map(str,range(4)));reader=lambda d,p,images=images:images[int(p)]
                fold=next(fold['fold'] for fold in folds if spec['group'] in fold['held'])
                vv={('native',0):run.runtime.previous.trace(native,None,None,files,truths[seq][0],reader,None,Silent())}
                for seed in run.c.SEEDS:
                    adapter=adapters[fold,seed,'matched']
                    for mode in ('nn','mm'):vv[mode,seed]=run.previous.dual_trace(native,adapter,mode,files,truths[seq][0],reader,None,Silent())
                source_bank[seq]=vv
            entries=f.fixture_entries(root,specs,truths)
            plan=run.old.sealed(dict(entries=specs,folds=folds,workers=2,search_factor=4.,code_sha256={},protocol_sha256=run.file_hash(run.PROTOCOL)))
            run.publish_identical(root/'manifest.json',plan);config=dict(root=str(root),truths=truths,stage212_root=str(root/'fake_source'))
            actual=run.runtime.run_workers
            def workers(config,seqs,phase,n,progress,target=None):return actual(config,seqs,phase,n,progress,target=cpu_worker)
            def source(root,p,d,t,spec,gt,verified):return source_bank[spec['sequence']]
            with patch.object(run.runtime,'run_workers',side_effect=workers),patch.object(run.c.source,'read_traces',side_effect=source),contextlib.redirect_stdout(io.StringIO()):
                run.experiment(root,plan,native,adapters,[dict(fold=f['fold'],seed=s) for f in folds for s in run.c.SEEDS],entries,None,config,({}, {}, {}))
                run.verify(root)
            d=run.old.read_sealed(root/'stage214_decision.json');self.assertEqual(d['training_runs'],8);self.assertEqual(d['training_updates'],2048)
            self.assertEqual(d['fit_provider_trajectories'],152);self.assertEqual(d['full_evaluation_trajectories'],342);self.assertEqual(len(d['primary_contrasts']),1)
            self.assertEqual(len(d['table_sha256']),12)
            import csv
            def table(name):
                with (root/f'stage214_{name}.csv').open(newline='',encoding='utf8') as h:return list(csv.DictReader(h))
            self.assertEqual(len(table('smoke')),18);self.assertEqual(len(table('source_replay')),190);self.assertEqual(len(table('paired')),42)
            self.assertTrue(all(r['in_fit']=='True' and r['in_held']=='False' for r in table('fit_support')))
            self.assertTrue(all(int(r['updater_calls'])==int(r['frames']) for r in table('calls') if r['arm'] in run.c.TRAINED))
            updaters=run.load_updaters(root,plan,native.rt.torch,next(native.rt.net.parameters()).device)
            with patch.object(run,'rollout',side_effect=AssertionError('Must reuse committed trace')):
                run.evaluate_video(root,plan,native,adapters,updaters,specs[0],entries[specs[0]['sequence']],None,Silent(),'full')
            # A held key is rejected before fitting, regardless of label values.
            with self.assertRaises(ValueError):run.train(root,plan,native,{folds[0]['held'][0]:{}},folds[0],2132,'forecast',Silent(),{})
            (root/'stage214_policy_summary.csv').write_text('changed')
            with self.assertRaises(ValueError):run.verify(root)


if __name__=='__main__':unittest.main()
