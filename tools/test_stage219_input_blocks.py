"""Block invariance, full-model bridge, matched CPU training and cache seals."""
import contextlib,csv,io,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tools'),str(Path(__file__).resolve().parents[1]/'tracking')]
import stage219_input_blocks as run
import test_stage217_predictor_audit as fixture

class Tests(unittest.TestCase):
    def test_reviewed_bundle(self):
        d,p,t=run.bundle(run.BUNDLE);self.assertEqual(len(t['group_metrics']),2280)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'input.txt';raw=run.BUNDLE.read_bytes().replace(b'\r\n',b'\n')
            for b in (raw,raw.replace(b'\n',b'\r\n')):
                path.write_bytes(b);self.assertEqual(run.bundle(path),(d,p,t))
            for b in (raw+b'x',raw+b'\n',b'\xef\xbb\xbf'+raw):
                path.write_bytes(b)
                with self.assertRaises(ValueError):run.bundle(path)

    def test_exact_block_invariance_and_same_initialization(self):
        import torch
        mean=np.linspace(-1,1,392).astype(np.float32);std=np.linspace(.5,2,392).astype(np.float32)
        models={a:run.build(torch,a,392,mean,std,2132) for a in run.ARMS}
        self.assertEqual(len({run.old.model_digest(m) for m in models.values()}),1)
        x=torch.tensor(mean[None]+.03);self.assertEqual(x.shape,(1,392))
        for mode,m in models.items():
            with torch.no_grad():m.network[-1].weight.fill_(.1)
            y=x.clone()
            if mode=='geometry':y[:,:384]+=1000
            elif mode=='visual':y[:,384:]+=1000
            else:continue
            np.testing.assert_array_equal(m(x).detach().numpy(),m(y).detach().numpy())
            explicit=x.clone();explicit[:,:384]=torch.tensor(mean[:384]) if mode=='geometry' else explicit[:,:384]
            if mode=='visual':explicit[:,384:]=torch.tensor(mean[384:])
            full=models['full'];np.testing.assert_array_equal(m(x).detach().numpy(),full(explicit).detach().numpy())
        with self.assertRaises(ValueError):run.build(torch,'geometry',8,mean,std,2132)

    def test_complete_workflow_bridge_resume_and_tampering(self):
        import torch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            args,files=fixture.fixture(Path(tmp),torch);run.source.audit(*args)
            out,_,config,entries,p214,d214,_,p215,d215,p216,d216,_=args
            catalog={f'stage{k}':d['artifact_sha256'] for k,d in ((214,d214),(215,d215),(216,d216))};plans=(p214,p215,p216)
            prior=Path(tmp)/'218';prior.mkdir();pp=run.old.sealed(dict(schema=run.base.SCHEMA,entries=p214['entries'],folds=p214['folds'],search_factor=4.,code_sha256={},protocol_sha256=run.file_hash(run.base.PROTOCOL)));run.publish_identical(prior/'manifest.json',pp)
            tables={'group_metrics':list(csv.DictReader(io.StringIO((out/'stage217_group_metrics.csv').read_text())))}
            run.base.experiment(prior,pp,config,plans,catalog,entries,tables,torch)
            t218={k:list(csv.DictReader(io.StringIO((prior/f'stage218_{k}.csv').read_text()))) for k in ('scope','group_metrics')}
            root=Path(tmp)/'219';root.mkdir();p=run.old.sealed(dict(schema=run.SCHEMA,entries=p214['entries'],folds=p214['folds'],search_factor=4.,code_sha256={},protocol_sha256=run.file_hash(run.PROTOCOL)));run.publish_identical(root/'manifest.json',p)
            frozen={str(f):run.file_hash(f) for f in files}
            with patch.object(torch.cuda,'init',side_effect=AssertionError('No CUDA')):run.experiment(root,p,config,plans,catalog,entries,t218,torch)
            d=run.old.read_sealed(root/'stage219_decision.json');self.assertEqual(d['training_updates'],3072);self.assertEqual(len(d['primary']),1)
            expected=dict(support=48,scope=12,training=3072,group_metrics=192,policy_summary=24,paired=10,source_replay=96,execution=1)
            for k,n in expected.items():self.assertEqual(len(list(csv.DictReader(io.StringIO((root/f'stage219_{k}.csv').read_text())))),n,k)
            self.assertEqual(frozen,{str(f):run.file_hash(f) for f in files})
            bank,_=run.base.data_for(config,plans,catalog,entries,p['folds'][0],2132,{})
            prog=run.OverallProgress('resume',256)
            with patch.object(torch.optim.AdamW,'__init__',side_effect=AssertionError('No retraining completed model')):
                run.train(root,p,torch,bank,p['folds'][0],2132,'geometry',prog)
                run.experiment(root,p,config,plans,catalog,entries,t218,torch)
            self.assertEqual(prog.reused,256)
            with self.assertRaises(ValueError):run.train(root,p,torch,dict(bank,held=next(iter(bank.values()))),p['folds'][0],2132,'geometry',prog)
            (root/'stage219_scope.csv').write_text('corrupt')
            with self.assertRaises(ValueError):run.verify(root)
            files[0].write_bytes(files[0].read_bytes()+b'x')
            with self.assertRaises(ValueError):run.base.data_for(config,plans,catalog,entries,p['folds'][0],2132,{})

if __name__=='__main__':unittest.main()
