"""Exact label identities, invalid support and frozen CPU replay without optimizer."""
import contextlib,csv,io,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tools'),str(Path(__file__).resolve().parents[1]/'tracking')]
import stage221_target_components as run
import test_stage217_predictor_audit as source_fixture

def fixture(root,torch):
    args,original_files=source_fixture.fixture(root,torch)
    _,_,config,entries,p214,d214,_,p215,d215,p216,d216,_=args
    plans=(p214,p215,p216);catalog={f'stage{k}':d['artifact_sha256'] for k,d in ((214,d214),(215,d215),(216,d216))}
    # Current GT invalid at t=1, next targets t>=2 remain valid. It must remain in cells.
    entries['seqa'][1][1,2]=0
    p220=run.old.sealed(dict(schema=run.source.SCHEMA,entries=p214['entries'],folds=p214['folds'],search_factor=4.))
    r220=root/'220';r220.mkdir();artifacts={};tables={'scope':[],'group_metrics':[],'actions':[]}
    for fold in p214['folds']:
        for seed in run.c.SEEDS:
            bank,items=run.source.data_for(config,plans,catalog,entries,fold,seed,{})
            xx=np.concatenate([bank[g]['x'] for g in fold['fit']]);mean=xx.mean(0).astype(np.float32);std=np.maximum(xx.std(0),1e-6).astype(np.float32);models={}
            for arm in run.source.ARMS:
                m=run.source.build(torch,arm,392,mean,std,seed)
                with torch.no_grad():m.network[-1].bias.copy_(torch.tensor([.01,.02,.03]))
                m.eval().requires_grad_(False);models[arm]=m
                scope=dict(fold=fold['fold'],seed=seed,arm=arm,state_sha256=run.old.model_digest(m));tables['scope'].append(scope)
                meta=dict(plan_fingerprint=p220['fingerprint'],fold=fold['fold'],seed=seed,arm=arm,fit=fold['fit'],held=fold['held'],updates=256)
                path=r220/'training'/f"f{fold['fold']}__s{seed}__{arm}.pt";run.old.save_model(torch,path,dict(state=m.state_dict(),scope=scope),meta);run.old.record(r220,path,artifacts)
            for split,g,origin,data in items:
                for arm in run.source.METHODS:
                    with torch.no_grad():z=np.zeros((len(data['x']),3),np.float32) if arm=='zero' else data['velocity'] if arm=='velocity' else models[arm](torch.tensor(data['x'])).numpy()
                    row=dict(split=split,group=g,seed=seed,origin=origin,arm=arm)
                    tables['group_metrics'].append(dict(row,**run.source.source.evaluate(torch,z,data,'forecast',4.)))
                    tables['actions'].append(dict(row,mean_center_delta=float(np.linalg.norm(z[:,:2],axis=1).mean()),mean_abs_log_side_delta=float(np.abs(z[:,2]).mean()),nonzero_fraction=float((np.abs(z).max(1)>1e-9).mean())))
    config.update(stage220_root=str(r220),source220_plan=p220['fingerprint'],source220_artifacts=artifacts)
    out=root/'221';out.mkdir();p=run.old.sealed(dict(schema=run.SCHEMA,entries=p214['entries'],folds=p214['folds'],search_factor=4.,code_sha256={},protocol_sha256=run.file_hash(run.PROTOCOL)));run.publish_identical(out/'manifest.json',p)
    return (out,p,config,plans,catalog,entries,tables,torch),original_files


class Tests(unittest.TestCase):
    def test_real_bundle(self):
        d,p,t=run.bundle(run.BUNDLE);self.assertEqual(len(t['group_metrics']),2280)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'packet';raw=run.BUNDLE.read_bytes().replace(b'\r\n',b'\n')
            for b in (raw,raw.replace(b'\n',b'\r\n')):
                path.write_bytes(b);self.assertEqual(run.bundle(path),(d,p,t))
            path.write_bytes(raw+b'x')
            with self.assertRaises(ValueError):run.bundle(path)

    def test_exact_decomposition_cancellation_and_invalid_labels(self):
        raw=np.tile([0.,0.,2.,2.],(3,1));cur=raw.copy();cur[0,0]=4;cur[1,2:]=4;cur[1,:2]=-1;cur[2,2]=0
        nxt=raw.copy();nxt[1,2:]=8;nxt[1,:2]=-3
        vc,vn,mask,e,m,y=run.components(raw,cur,nxt)
        self.assertEqual(mask.tolist(),[True,True,False]);self.assertTrue(vn.all());np.testing.assert_allclose(e[mask]+m[mask],y[mask])
        self.assertEqual((e[0,0],m[0,0],y[0,0]),(2,-2,0))
        np.testing.assert_allclose([e[1,2],m[1,2],y[1,2]],[np.log(2),np.log(2),np.log(4)])
        v=run.component_metrics(e[:1],m[:1],y[:1]);self.assertEqual(v['center_interaction'],-8);self.assertEqual(v['label_center_energy'],0)
        self.assertEqual(v['error_center_energy']+v['motion_center_energy']+v['center_interaction'],v['label_center_energy'])
        self.assertTrue(all(v is None for v in run.component_metrics(e[:0],m[:0],y[:0]).values()))
        self.assertIsNone(run.contrast([])['ci_low']);self.assertIsNone(run.contrast([1])['worst_leave_one_video_out'])

    def test_effective_parent_bounds_and_zero_cosine_support(self):
        import torch
        raw=np.array([[50.,10.,4.,4.]]);hw=np.array([[20.,60.]])
        effective=run.applied_delta(torch,np.array([[100.,-100.,100.]]),raw,hw)
        np.testing.assert_allclose(effective[0,:2],[2,-3],atol=1e-6);self.assertAlmostEqual(effective[0,2],np.log(15),places=5)
        rr=run.action_metrics(np.zeros((1,3)),np.ones((1,3)),np.ones((1,3)),np.ones((1,3))*2)
        self.assertEqual(rr['center_cosine_error_samples'],0);self.assertIsNone(rr['center_cosine_to_error'])

    def test_frozen_end_to_end_no_optimizer_and_tamper(self):
        import torch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            args,files=fixture(Path(tmp),torch);before={str(f):run.file_hash(f) for f in files}
            with patch.object(torch.optim.AdamW,'__init__',side_effect=AssertionError('No optimizer')),patch.object(torch.cuda,'init',side_effect=AssertionError('No CUDA')):run.experiment(*args)
            root=args[0];d=run.old.read_sealed(root/'stage221_decision.json');self.assertEqual(d['new_training_updates'],0);self.assertEqual(len(d['primary']),1)
            counts=dict(support=48,scope=12,cells=144,actions=240,group_metrics=48,policy_summary=6,paired=2,source_replay=240,execution=1)
            for name,n in counts.items():self.assertEqual(len(list(csv.DictReader(io.StringIO((root/f'stage221_{name}.csv').read_text())))),n,name)
            cells=list(csv.DictReader(io.StringIO((root/'stage221_cells.csv').read_text())))
            bad=[r for r in cells if r['valid_current']=='False'];self.assertEqual(len(bad),12);self.assertTrue(all(r['error_dx']==r['motion_dx']=='' and r['label_dx']!='' for r in bad))
            self.assertEqual(before,{str(f):run.file_hash(f) for f in files});run.experiment(*args)
            (root/'stage221_cells.csv').write_text('changed')
            with self.assertRaises(ValueError):run.verify(root)
            files[0].write_bytes(files[0].read_bytes()+b'x')
            with self.assertRaises(ValueError):run.experiment(*( (root.parent/'new',)+args[1:] ))

if __name__=='__main__':unittest.main()
