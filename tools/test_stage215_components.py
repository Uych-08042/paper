"""Frozen output masks, bound endpoints, true CENTER, and spawned complete OOF."""
import contextlib,io,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tools'),str(Path(__file__).resolve().parents[1]/'tracking')]
import stage215_component_experiment as run


class Quiet:
    reused=0
    def advance(self,*a,**k):pass


def cpu_worker(config,sequences,phase,worker,channel):
    try:
        import test_stage210_recovery_audit as f
        root=Path(config['root']);plan=run.old.read_sealed(root/'manifest.json');native=f.make_native();adapters=f.models_for(native,plan['folds'])
        updaters=run.previous.load_updaters(Path(config['state_root']),config['source_plan'],native.rt.torch,next(native.rt.net.parameters()).device)
        entries=f.fixture_entries(root,plan['entries'],config['truths']);progress=run.runtime.WorkerProgress(channel,worker)
        for seq in sequences:run.evaluate_video(root,plan,native,adapters,updaters,next(e for e in plan['entries'] if e['sequence']==seq),entries[seq],None,progress,phase)
        progress.flush();channel.put(('complete',worker,0,0))
    except BaseException:
        import traceback
        channel.put(('error',worker,traceback.format_exc(),0));raise


def updater_fixture(native,folds):
    torch=native.rt.torch;result={}
    for fold in folds:
        for seed in run.c.SEEDS:
            for a in run.c.FAMILIES:
                m=run.previous.c.build(torch,24,np.zeros(24),np.ones(24),seed).eval()
                with torch.no_grad():m.network[-1].weight.fill_(.001);m.network[-1].bias.copy_(torch.tensor([.05,-.04,.03 if a=='forecast' else .2]))
                result[fold['fold'],seed,a]=m.requires_grad_(False)
    return result


class Tests(unittest.TestCase):
    def test_reviewed_source_and_masks(self):
        d,p,t=run.c.source214(run.DEFAULT_BUNDLE);self.assertEqual(d['fingerprint'],run.c.SOURCE214);self.assertEqual(len(t),12)
        import torch
        m=run.previous.c.build(torch,3,np.zeros(3),np.ones(3),2132).eval()
        with torch.no_grad():m.network[-1].bias.copy_(torch.tensor([.3,-.2,.4]))
        x=torch.ones((1,3));original=m(x);sha=run.old.model_digest(m)
        for mode in run.c.MASKS:
            v=run.c.Masked(m,mode)(x)
            for i,on in enumerate(run.c.MASKS[mode]):self.assertEqual(float(v[0,i]),float(original[0,i]) if on else 0.)
            self.assertEqual(run.old.model_digest(m),sha)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.txt';path.write_bytes(run.DEFAULT_BUNDLE.read_bytes()+b'x')
            with self.assertRaises(ValueError):run.c.source214(path)

    def test_same_parent_endpoints_and_label_independence(self):
        import test_stage210_recovery_audit as f
        native=f.make_native();folds=[dict(fold=0,fit=[],held=[])];a=f.models_for(native,folds)[0,2132,'matched'];u=updater_fixture(native,folds)[0,2132,'coverage']
        images=np.random.default_rng(215).normal(size=(4,20,8)).astype(np.float32);reader=lambda d,p:images[int(p)];files=list(map(str,range(4)));initial=[2.,2.,1.,1.]
        before=[len(b._forward_hooks) for b in native.rt.backbone.blocks];sha=run.old.model_digest(u)
        old_none=run.previous.rollout(native,a,'nn',None,files,initial,reader,None,Quiet())
        old_joint=run.previous.rollout(native,a,'coverage',u,files,initial,reader,None,Quiet())
        for mode in run.c.MASKS:
            v=run.previous.rollout(native,a,'coverage',run.c.Masked(u,mode),files,initial,reader,None,Quiet());run.c.audit(v,initial,3,'coverage__'+mode)
            if mode in ('none','joint'):
                ref=old_none if mode=='none' else old_joint
                for k in ref:
                    if k not in ('pipeline_ms','updater_calls'):np.testing.assert_array_equal(v[k],ref[k])
            if mode=='center':np.testing.assert_array_equal(v['next_parent'][:,2:],v['reference_boxes'][:,2:])
            if mode=='scale':np.testing.assert_allclose(v['next_parent'][:,:2]+v['next_parent'][:,2:]/2,v['reference_boxes'][:,:2]+v['reference_boxes'][:,2:]/2,atol=1e-12)
        gt=np.tile([1.,1.,1.,1.],(4,1));other=gt.copy();other[1:,:2]+=50
        vv=[run.previous.rollout(native,a,'coverage',run.c.Masked(u,'center'),files,initial,reader,None,Quiet(),truth=g) for g in (gt,other)]
        for k in ('boxes','parent','next_parent','state_delta'):np.testing.assert_array_equal(vv[0][k],vv[1][k])
        self.assertEqual(sha,run.old.model_digest(u));self.assertEqual(before,[len(b._forward_hooks) for b in native.rt.backbone.blocks])

    def test_bound_complete_spawned_workflow(self):
        import torch
        import test_stage210_recovery_audit as f
        native=f.make_native();groups=[f'uav{i:07d}' for i in range(38)];folds=run.c.protocol.split_groups(groups);adapters=f.models_for(native,folds);updaters=updater_fixture(native,folds)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'new';root.mkdir();src=Path(tmp)/'old';src.mkdir();(root/'fixture').mkdir();truths={};specs=[dict(group=g,sequence=g+'_00000_s') for g in groups]
            sp=run.old.sealed(dict(entries=specs,folds=folds));artifacts={};scope=[];frozen_scope=[];rng=np.random.default_rng(215)
            for (fold,seed,a),m in updaters.items():
                meta=dict(plan_fingerprint=sp['fingerprint'],fold=fold,seed=seed,arm=a,fit=folds[fold]['fit'],held=folds[fold]['held'],updates=256)
                path=src/'training'/f'f{fold}__s{seed}__{a}.pt';sha=run.old.model_digest(m)
                run.old.save_model(torch,path,dict(state=m.state_dict(),scope=dict(state_sha256=sha)),meta);run.old.record(src,path,artifacts)
                scope.append(dict(fold=str(fold),seed=str(seed),arm=a,state_sha256=sha))
            for fold in folds:
                for seed in run.c.SEEDS:frozen_scope.append(dict(fold=str(fold['fold']),seed=str(seed),state_sha256=run.old.model_digest(adapters[fold['fold'],seed,'matched'])))
            for spec in specs:
                seq=spec['sequence'];images=rng.normal(size=(3,20,8)).astype(np.float32);np.save(root/'fixture'/(seq+'.npy'),images)
                gt=np.tile([2.,2.,1.,1.],(3,1));truths[seq]=gt.tolist();files=list(map(str,range(3)));reader=lambda d,p,images=images:images[int(p)]
                fold=next(f['fold'] for f in folds if spec['group'] in f['held'])
                for arm,seed in [('native',0)]+[(a,s) for s in run.c.SEEDS for a in ('nn','mm',*run.c.FAMILIES)]:
                    a=None if arm=='native' else adapters[fold,seed,'matched'];u=updaters.get((fold,seed,arm))
                    v=run.runtime.previous.trace(native,None,None,files,gt[0],reader,None,Quiet()) if arm=='native' else run.previous.rollout(native,a,arm,u,files,gt[0],reader,None,Quiet(),truth=gt)
                    path=run.previous.path_for(src,'full',seq,seed,arm);meta=run.previous.metadata(sp,seq,fold,seed,arm,a,u,'full')
                    run.old.cache(path,meta,lambda:v);run.old.record(src,path,artifacts)
            tables=dict(frozen_scope=frozen_scope,scope_summary=scope);sd=dict(artifact_sha256=artifacts)
            entries=f.fixture_entries(root,specs,truths);plan=run.old.sealed(dict(entries=specs,folds=folds,workers=2,code_sha256={},protocol_sha256=run.file_hash(run.PROTOCOL),masks={k:list(v) for k,v in run.c.MASKS.items()}))
            run.publish_identical(root/'manifest.json',plan);run.publish_identical(root/'manifest.json',plan)
            config=dict(root=str(root),truths=truths,state_root=str(src),source_plan=sp);actual=run.runtime.run_workers
            def workers(config,seqs,phase,n,progress,target=None):return actual(config,seqs,phase,n,progress,target=cpu_worker)
            with patch.object(run.runtime,'run_workers',side_effect=workers),patch.object(torch.optim.AdamW,'__init__',side_effect=AssertionError('No training')),contextlib.redirect_stdout(io.StringIO()):
                run.experiment(root,plan,native,adapters,updaters,scope,entries,None,config,(sp,sd,tables));run.verify(root)
            d=run.old.read_sealed(root/'stage215_decision.json');self.assertEqual(d['new_training_updates'],0);self.assertEqual(d['new_full_trajectories'],608);self.assertEqual(d['reported_quality_sequences'],722);self.assertEqual(len(d['primary_contrasts']),1)
            import csv
            def table(name):
                with (root/f'stage215_{name}.csv').open(newline='',encoding='utf8') as h:return list(csv.DictReader(h))
            for n,count in dict(smoke=32,source_replay=418,policy_summary=10,paired=132,factor_summary=36,seed_video_summary=722,search_geometry=684,calls=722).items():self.assertEqual(len(table(n)),count,n)
            self.assertTrue(all(int(r['updater_calls'])==int(r['frames']) for r in table('calls') if r['provenance']=='new'))
            self.assertTrue(all(run.file_hash(src/n)==h for n,h in artifacts.items()))
            with patch.object(run.previous,'rollout',side_effect=AssertionError('Reuse committed trace')):run.evaluate_video(root,plan,native,adapters,updaters,specs[0],entries[specs[0]['sequence']],None,Quiet(),'full')
            with self.assertRaises(ValueError):run.source_trace(config,sp,sd,tables,specs[0],'nn',999,{})
            (root/'stage215_policy_summary.csv').write_text('changed')
            with self.assertRaises(ValueError):run.verify(root)


if __name__=='__main__':unittest.main()
