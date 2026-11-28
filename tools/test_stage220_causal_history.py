"""Causality, fit-only negative control and full cached CPU workflow."""
import contextlib,csv,io,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tools'),str(Path(__file__).resolve().parents[1]/'tracking')]
import stage220_causal_history as run
import test_stage217_predictor_audit as fixture

def read_table(root,stage,name):return list(csv.DictReader(io.StringIO((root/f'stage{stage}_{name}.csv').read_text())))

class Tests(unittest.TestCase):
    def test_reviewed_source(self):
        d,p,t=run.bundle(run.BUNDLE);self.assertEqual(len(t['group_metrics']),1824)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'packet';raw=run.BUNDLE.read_bytes().replace(b'\r\n',b'\n')
            for b in (raw,raw.replace(b'\n',b'\r\n')):
                path.write_bytes(b);self.assertEqual(run.bundle(path),(d,p,t))
            path.write_bytes(raw+b'x')
            with self.assertRaises(ValueError):run.bundle(path)

    def test_causal_differences_warmup_and_scale(self):
        initial=np.array([0.,0.,4.,4.]);refs=np.array([[2.,1.,4.,4.],[4.,2.,4.,4.],[6.,3.,4.,4.]])
        v,a=run.motion(refs,initial,[1,2,3]);np.testing.assert_array_equal(v,np.tile([.5,.25,0],(3,1)));np.testing.assert_array_equal(a,0)
        changed=refs.copy();changed[2]=np.nan
        np.testing.assert_array_equal(run.motion(changed,initial,[1,2])[0],v[:2])
        quadratic=np.array([[1.,0.,4.,4.],[4.,0.,4.,4.],[9.,0.,4.,4.]])
        np.testing.assert_array_equal(run.motion(quadratic,initial,[2,3])[1][:,0],[.5,.5])
        side=np.exp(.1*np.arange(4)**2);boxes=np.c_[-side/2,-side/2,side,side]
        vv,aa=run.motion(boxes[1:],boxes[0],[1,2,3]);np.testing.assert_allclose(aa[:,2],[0,.2,.2],atol=1e-7);np.testing.assert_allclose(vv[:,2],[.1,.3,.5],atol=1e-7)

    def test_shuffle_changes_only_fit_history_columns(self):
        bank={'g':dict(x=np.arange(3*392).reshape(3,392).astype(np.float32),raw=np.ones((3,4)),target=np.ones((3,4))*2,image_hw=np.ones((3,2))*50)}
        batch=[('g',0),('g',2)];perm={'g':np.array([2,0,1])}
        actual=run.batch_values(bank,batch,'history',perm);wrong=run.batch_values(bank,batch,'shuffled',perm)
        for k in ('raw','target','image_hw'):np.testing.assert_array_equal(actual[k],wrong[k])
        np.testing.assert_array_equal(actual['x'][:,:381],wrong['x'][:,:381]);np.testing.assert_array_equal(actual['x'][:,384:],wrong['x'][:,384:]);np.testing.assert_array_equal(wrong['x'][:,381:384],bank['g']['x'][[2,1],381:384])

    def test_complete_training_source_bridge_resume_and_tamper(self):
        import torch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            args,files=fixture.fixture(Path(tmp),torch);run.source.audit(*args)
            out,_,config,entries,p214,d214,_,p215,d215,p216,d216,_=args
            plans=(p214,p215,p216);catalog={f'stage{k}':d['artifact_sha256'] for k,d in ((214,d214),(215,d215),(216,d216))}
            def plan_for(stage,protocol):
                root=Path(tmp)/str(stage);root.mkdir();p=run.old.sealed(dict(schema=stage,entries=p214['entries'],folds=p214['folds'],search_factor=4.,code_sha256={},protocol_sha256=run.file_hash(protocol)));return root,p
            r218,p218=plan_for(218,run.base.PROTOCOL);p218=run.old.sealed(dict(p218,schema=run.base.SCHEMA));p218=run.old.sealed({k:v for k,v in p218.items() if k!='fingerprint'});run.publish_identical(r218/'manifest.json',p218)
            run.base.experiment(r218,p218,config,plans,catalog,entries,{'group_metrics':read_table(out,217,'group_metrics')},torch)
            r219,p219=plan_for(219,run.prior.PROTOCOL);p219=run.old.sealed({**{k:v for k,v in p219.items() if k!='fingerprint'},'schema':run.prior.SCHEMA});run.publish_identical(r219/'manifest.json',p219)
            run.prior.experiment(r219,p219,config,plans,catalog,entries,{k:read_table(r218,218,k) for k in ('scope','group_metrics')},torch)
            d219=run.old.read_sealed(r219/'stage219_decision.json');config.update(stage219_root=str(r219),source219_artifacts=d219['artifact_sha256'])
            root,p=plan_for(220,run.PROTOCOL);p=run.old.sealed({**{k:v for k,v in p.items() if k!='fingerprint'},'schema':run.SCHEMA});run.publish_identical(root/'manifest.json',p)
            tables={k:read_table(r219,219,k) for k in ('scope','group_metrics')};before={str(f):run.file_hash(f) for f in files}
            with patch.object(torch.cuda,'init',side_effect=AssertionError('No CUDA')):run.experiment(root,p,config,plans,catalog,entries,tables,torch)
            d=run.old.read_sealed(root/'stage220_decision.json');self.assertEqual(d['training_updates'],3072);self.assertEqual(len(d['primary']),1)
            expected=dict(support=48,scope=12,training=3072,group_metrics=240,policy_summary=30,paired=12,source_replay=96,actions=240,execution=1)
            for k,n in expected.items():self.assertEqual(len(read_table(root,220,k)),n,k)
            ss=read_table(root,220,'scope')
            for f in p['folds']:
                for seed in run.c.SEEDS:
                    rr=[r for r in ss if int(r['fold'])==f['fold'] and int(r['seed'])==seed]
                    for key in ('initial_network_sha256','normalization_sha256','draws_sha256','permutation_sha256'):self.assertEqual(len({r[key] for r in rr}),1)
            self.assertEqual(before,{str(f):run.file_hash(f) for f in files})
            bank,items=run.data_for(config,plans,catalog,entries,p['folds'][0],2132,{})
            for _,_,_,item in items:np.testing.assert_array_equal(item['x'][:,381:384],item['acceleration'])
            prog=run.OverallProgress('resume',256)
            with patch.object(torch.optim.AdamW,'__init__',side_effect=AssertionError('No repeated training')):
                run.train(root,p,torch,bank,p['folds'][0],2132,'history',prog);run.experiment(root,p,config,plans,catalog,entries,tables,torch)
            self.assertEqual(prog.reused,256)
            with self.assertRaises(ValueError):run.train(root,p,torch,dict(bank,held=next(iter(bank.values()))),p['folds'][0],2132,'history',prog)
            (root/'stage220_scope.csv').write_text('changed')
            with self.assertRaises(ValueError):run.verify(root)
            files[0].write_bytes(files[0].read_bytes()+b'x')
            with self.assertRaises(ValueError):run.data_for(config,plans,catalog,entries,p['folds'][0],2132,{})

if __name__=='__main__':unittest.main()
