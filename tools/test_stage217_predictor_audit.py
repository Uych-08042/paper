"""Frozen model/cache audit with real receipts, loss and CPU model loading."""
import contextlib,csv,io,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tools'),str(Path(__file__).resolve().parents[1]/'tracking')]
import stage217_predictor_audit as run


def fixture(root,torch):
    groups=list('abcd');folds=[dict(fold=0,fit=groups[:2],held=groups[2:]),dict(fold=1,fit=groups[2:],held=groups[:2])]
    specs=[dict(group=g,sequence='seq'+g,tracked_frames=4) for g in groups]
    p214=run.old.sealed(dict(entries=specs,folds=folds,search_factor=4.))
    p215=run.old.sealed(dict(entries=specs,folds=folds,search_factor=4.,source_stage214='fixture'))
    p216=run.old.sealed(dict(entries=specs,folds=folds,selection=run.prior.selection(p215)))
    paths={str(k):root/str(k) for k in (214,215,216)}
    for p in paths.values():p.mkdir()
    decisions={str(k):dict(artifact_sha256={}) for k in (214,215,216)}
    t214=dict(frozen_scope=[],scope_summary=[]);entries={};saved=[]
    def forbid(*a,**kw):raise AssertionError('Images/GPU/optimization forbidden')
    def cache(k,relative,meta,values):
        path=paths[str(k)]/relative;run.old.cache(path,meta,lambda:values)
        run.old.record(paths[str(k)],path,decisions[str(k)]['artifact_sha256']);saved.append(path)
    def data(g,seed,origin):
        rng=np.random.default_rng(seed+ord(g)+origin)
        raw=np.tile([60.,10.,4.,8.],(4,1));raw[:,0]+=np.arange(4)
        hw=np.c_[80.+10*np.arange(4),100.+100*np.arange(4)]
        x=rng.normal(size=(3,392)).astype(np.float32)
        return dict(reference_boxes=raw,boxes=raw+np.array([.1,.2,0,0]),image_hw=hw,sample_frame=np.arange(1,4),sample_features=x)
    for g in groups:
        gt=np.tile([65.,12.,6.,10.],(5,1));gt[:,0]+=np.arange(5)
        entries['seq'+g]=(list(map(str,range(5))),gt,None,forbid,'fixture')
    for fold in folds:
        for seed in run.c.SEEDS:
            bank={};adapter=f"adapter{fold['fold']}_{seed}"
            t214['frozen_scope'].append(dict(fold=fold['fold'],seed=seed,state_sha256=adapter))
            for g in fold['fit']:
                pieces=[]
                for i,o in enumerate(('nn','mm')):
                    v=data(g,seed,i);meta=dict(plan_fingerprint=p214['fingerprint'],fold=fold['fold'],seed=seed,group=g,mode=o,fit=fold['fit'],held=fold['held'],samples=[1,2,3],adapter_sha256=adapter,role='outer_fit_only')
                    cache(214,f"fit_rollouts/f{fold['fold']}__s{seed}__seq{g}__{o}.npz",meta,v)
                    pieces.append(dict(x=v['sample_features'],raw=v['reference_boxes'][:3],target=entries['seq'+g][1][2:],image_hw=v['image_hw'][:3]))
                bank[g]={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
            xx=np.concatenate([bank[g]['x'] for g in fold['fit']]);mean=xx.mean(0,dtype=np.float64).astype(np.float32);std=np.maximum(xx.std(0,dtype=np.float64),1e-6).astype(np.float32)
            draws=run.prior.source.c.protocol.draws_for(fold,seed,{g:len(bank[g]['x']) for g in fold['fit']});models={}
            for i,f in enumerate(run.c.TRAINED):
                model=run.c.build(torch,392,mean,std,seed)
                with torch.no_grad():model.network[-1].bias.copy_(torch.tensor([10.,-5.,3.])+i)
                models[f]=model.eval().requires_grad_(False);sha=run.old.model_digest(model)
                t214['scope_summary'].append(dict(fold=fold['fold'],seed=seed,arm=f,state_sha256=sha))
                meta=dict(plan_fingerprint=p214['fingerprint'],fold=fold['fold'],seed=seed,arm=f,fit=fold['fit'],held=fold['held'],updates=256,draws_sha256=run.fingerprint(draws),normalization_sha256=run.fingerprint(dict(mean=mean.tolist(),std=std.tolist())))
                path=paths['214']/'training'/f"f{fold['fold']}__s{seed}__{f}.pt"
                run.old.save_model(torch,path,dict(state=model.state_dict(),scope=dict(state_sha256=sha)),meta)
                run.old.record(paths['214'],path,decisions['214']['artifact_sha256']);saved.append(path)
            for g in fold['held']:
                for i,o in enumerate(run.prior.ORIGINS):
                    v=data(g,seed,i);seq='seq'+g
                    meta=dict(plan_fingerprint=p215['fingerprint'],sequence=seq,fold=fold['fold'],seed=seed,arm=o,phase='full')
                    path=run.prior.source.previous.path_for(paths['215'],'full',seq,seed,o)
                    cache(215,path.relative_to(paths['215']).as_posix(),meta,v)
                    with torch.no_grad():delta=np.stack([models[f](torch.tensor(v['sample_features'])).numpy() for f in run.c.TRAINED],axis=1)
                    value=dict(features=v['sample_features'],current_reference=v['reference_boxes'][:3],current_box=v['boxes'][:3],deltas=delta,
                        next_image_hw=np.repeat(v['image_hw'][1:,None,:],len(run.prior.ACTIONS),axis=1))
                    meta=dict(plan_fingerprint=p216['fingerprint'],sequence=seq,origin=o,seed=seed,frames=[1,2,3],updaters={f:run.old.model_digest(m) for f,m in models.items()})
                    cache(216,f'full/{seq}__s{seed}__{o}.npz',meta,value)
    config={f'stage{k}_root':str(p) for k,p in paths.items()}
    plan=run.old.sealed(dict(schema=run.SCHEMA,entries=specs,search_factor=4.,code_sha256={},protocol_sha256=run.file_hash(run.PROTOCOL)))
    output=root/'out';output.mkdir();run.publish_identical(output/'manifest.json',plan)
    args=(output,plan,config,entries,p214,decisions['214'],t214,p215,decisions['215'],p216,decisions['216'],torch)
    return args,saved


class Tests(unittest.TestCase):
    def test_reviewed_export_code_and_newline_binding(self):
        d,p=run.source216(run.BUNDLE216);self.assertEqual(d['states'],14592)
        compatibility=run.check_source_code(p);self.assertEqual(len(compatibility),1)
        self.assertEqual(compatibility[0]['permitted_difference'],'bundle reader only')
        read=Path.read_bytes;entry=run.REPO/'tools/stage216_successor_quality.py'
        changed=read(entry).replace(b"ORIGINS=('coverage__none'",b"ORIGINS=('changed'",1)
        with patch.object(Path,'read_bytes',lambda path:changed if path==entry else read(path)):
            with self.assertRaises(ValueError):run.check_source_code(p)
        d214,p214,t214=run.prior.source.c.source214(run.prior.source.DEFAULT_BUNDLE)
        self.assertEqual(len(t214['scope_summary']),8);self.assertEqual(p214['folds'],p['folds'])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bundle.txt';raw=run.BUNDLE216.read_bytes().replace(b'\r\n',b'\n')
            for b in (raw,raw.replace(b'\n',b'\r\n')):
                path.write_bytes(b);self.assertEqual(run.source216(path),(d,p))
            for b in (raw+b'\n',raw+b'x',b'\xef\xbb\xbf'+raw):
                path.write_bytes(b)
                with self.assertRaises(ValueError):run.source216(path)

    def test_exact_original_loss_clipping_and_decomposition(self):
        import torch
        raw=np.array([[60.,10.,4.,8.],[1.,2.,2.,2.]])
        target=np.array([[65.,12.,6.,10.],[4.,5.,4.,4.]])
        data=dict(raw=raw,target=target,image_hw=np.array([[80.,100.],[20.,30.]]))
        z=np.array([[10.,-5.,3.],[-10.,10.,-9.]])
        # Independent NumPy clipping and Smooth-L1 arithmetic.
        side=np.sqrt(raw[:,2:].prod(1));rc=raw[:,:2]+raw[:,2:]/2;tc=target[:,:2]+target[:,2:]/2;ts=np.sqrt(target[:,2:].prod(1))
        center=np.clip(rc+z[:,:2]*side[:,None],0,data['image_hw'][:,::-1]);logs=np.clip(np.log(side)+z[:,2],0,np.log(data['image_hw'].max(1)))
        pred=np.c_[(center-rc)/side[:,None],logs-np.log(side)];diff=np.abs(pred-run.label(raw,target));reg=np.where(diff<1,.5*diff**2,diff-.5).mean()
        cover=np.maximum((np.abs(center-tc)+target[:,2:]/2-4*np.exp(logs[:,None])/2)/ts[:,None],0).mean()
        for family in run.c.TRAINED:
            out=run.evaluate(torch,z,data,family,4.)
            self.assertAlmostEqual(out['regression'],reg,places=5);self.assertAlmostEqual(out['coverage'],cover,places=5)
            self.assertAlmostEqual(out['total'],reg+(cover if family=='coverage' else 0),places=5)
        wrong=dict(data,image_hw=data['image_hw']*2)
        self.assertNotEqual(run.evaluate(torch,z,data,'forecast',4.),run.evaluate(torch,z,wrong,'forecast',4.))

    def test_constant_fit_roster_and_equal_source_weight(self):
        one=lambda n,dx:dict(raw=np.tile([0.,0.,2.,2.],(n,1)),target=np.tile([dx,0.,2.,2.],(n,1)))
        bank=dict(a=one(1,0),b=one(9,4));np.testing.assert_array_equal(run.constant_for_fit(bank,['a','b']),[1,0,0])
        with self.assertRaises(ValueError):run.constant_for_fit(dict(bank,held=one(1,100)),['a','b'])
        before=run.constant_for_fit(bank,['a','b']);held=one(8,999);held['target'][:]=1
        np.testing.assert_array_equal(before,run.constant_for_fit(bank,['a','b']))
        self.assertEqual(run.loss_contrast([-9.,1.,2.])['worst_leave_one_video_out'],1.5)

    def test_complete_frozen_cache_workflow_and_tamper_rejection(self):
        import torch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            args,files=fixture(Path(tmp),torch);hashes={str(p):run.file_hash(p) for p in files}
            with patch.object(torch.optim.AdamW,'__init__',side_effect=AssertionError('no optimizer')),patch.object(torch.cuda,'init',side_effect=AssertionError('no CUDA')),contextlib.redirect_stdout(io.StringIO()):
                run.audit(*args)
            root=args[0];decision=run.old.read_sealed(root/'stage217_decision.json');run.verify(root)
            self.assertEqual(len(decision['table_sha256']),9);self.assertEqual(len(decision['primary']),1)
            counts=dict(support=48,model_scope=8,constants=4,normalization=4,group_metrics=288,policy_summary=36,paired=73,prediction_replay=64,execution=1)
            for name,count in counts.items():
                rows=list(csv.DictReader(io.StringIO((root/f'stage217_{name}.csv').read_text(encoding='utf8'))));self.assertEqual(len(rows),count,name)
            replay=list(csv.DictReader(io.StringIO((root/'stage217_prediction_replay.csv').read_text(encoding='utf8'))))
            self.assertTrue(all(float(r['cpu_gpu_max_abs_delta'])==0 for r in replay))
            self.assertEqual(hashes,{str(p):run.file_hash(p) for p in files})
            # Corruption cannot be silently accepted as a cache resume.
            (root/'stage217_constants.csv').write_text('changed',encoding='utf8')
            with self.assertRaises(ValueError):run.verify(root)
            files[0].write_bytes(files[0].read_bytes()+b'x')
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(ValueError):run.audit(*args)


if __name__=='__main__':unittest.main()
