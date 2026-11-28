"""Actual CPU optimization and bound cache evaluation of all capacity arms."""
import contextlib,io,csv,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tools'),str(Path(__file__).resolve().parents[1]/'tracking')]
import stage218_capacity as run
import test_stage217_predictor_audit as fixture

class Tests(unittest.TestCase):
    def test_source_packet(self):
        d,p,t=run.bundle(run.BUNDLE);self.assertEqual(len(t['group_metrics']),2736)
        with tempfile.TemporaryDirectory() as tmp:
            pp=Path(tmp)/'bad.txt';pp.write_bytes(run.BUNDLE.read_bytes()+b'x')
            with self.assertRaises(ValueError):run.bundle(pp)

    def test_complete_training_evaluation_resume_and_source_tamper(self):
        import torch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            args,files=fixture.fixture(Path(tmp),torch)
            run.source.audit(*args)
            out,p217,config,entries,p214,d214,t214,p215,d215,p216,d216,_=args
            catalog={f'stage{k}':d['artifact_sha256'] for k,d in ((214,d214),(215,d215),(216,d216))}
            root=Path(tmp)/'218';root.mkdir()
            p=run.old.sealed(dict(schema=run.SCHEMA,entries=p214['entries'],folds=p214['folds'],search_factor=4.,code_sha256={},protocol_sha256=run.file_hash(run.PROTOCOL)))
            run.publish_identical(root/'manifest.json',p)
            t={'group_metrics':list(csv.DictReader(io.StringIO((out/'stage217_group_metrics.csv').read_text())))}
            hashes={str(f):run.file_hash(f) for f in files}
            with patch.object(torch.cuda,'init',side_effect=AssertionError('No CUDA')):
                run.experiment(root,p,config,(p214,p215,p216),catalog,entries,t,torch)
            d=run.old.read_sealed(root/'stage218_decision.json');self.assertEqual((d['training_runs'],d['training_updates']),(12,3072))
            scope=list(csv.DictReader(io.StringIO((root/'stage218_scope.csv').read_text())))
            self.assertEqual({r['arm']:int(r['parameters']) for r in scope},{'bias':3,'linear':1179,'mlp':12675})
            for fold in (0,1):
                for seed in (2132,2133):
                    rr=[r for r in scope if int(r['fold'])==fold and int(r['seed'])==seed]
                    self.assertEqual(len({r['draws_sha256'] for r in rr}),1);self.assertEqual(len({r['normalization_sha256'] for r in rr}),1)
            before={n:run.file_hash(root/n) for n in d['artifact_sha256'] if n.endswith('.pt')}
            with patch.object(torch.optim.AdamW,'__init__',side_effect=AssertionError('Resume must not train')):
                run.experiment(root,p,config,(p214,p215,p216),catalog,entries,t,torch)
            self.assertEqual(before,{n:run.file_hash(root/n) for n in before});self.assertEqual(hashes,{str(f):run.file_hash(f) for f in files})
            bank,_=run.data_for(config,(p214,p215,p216),catalog,entries,p['folds'][0],2132,{})
            progress=run.OverallProgress('Resume test',256)
            with patch.object(torch.optim.AdamW,'__init__',side_effect=AssertionError('Must reuse completed model')):
                run.train(root,p,torch,bank,p['folds'][0],2132,'linear',progress)
            self.assertEqual(progress.reused,256)
            with self.assertRaises(ValueError):run.train(root,p,torch,dict(bank,held=next(iter(bank.values()))),p['folds'][0],2132,'bias',None)
            (root/'stage218_scope.csv').write_text('changed')
            with self.assertRaises(ValueError):run.verify(root)
            files[0].write_bytes(files[0].read_bytes()+b'x')
            with self.assertRaises(ValueError):run.data_for(config,(p214,p215,p216),catalog,entries,p['folds'][0],2132,{})

if __name__=='__main__':unittest.main()
