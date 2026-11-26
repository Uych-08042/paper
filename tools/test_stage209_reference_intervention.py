"""CPU validation: frozen views, bound source, same-parent and spawned full workflow."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tools'),str(Path(__file__).resolve().parents[1]/'tracking')]
import stage209_reference_intervention as run
import test_stage208_state_interaction as fixture208


def cpu_worker(config,sequences,phase,worker,channel):
    import torch
    import test_stage206_spatial_adaptation as f
    try:
        torch.set_num_threads(1);helper=f.TorchTests();helper.torch=torch;native=helper.native()
        root=Path(config['root']);plan=run.old.read_sealed(root/'manifest.json')
        models,refs,_,_=run.c.load_models(Path(config['source_root']),config['source_decision'],config['source_plan'],config['source_tables'],native,{})
        entries=fixture208.fixture_entries(root,plan['entries']);progress=run.prior.WorkerProgress(channel,worker)
        for seq in sequences:
            spec=next(e for e in plan['entries'] if e['sequence']==seq)
            run.evaluate_video(root,plan,native,models,refs,spec,entries[seq],None,progress,phase)
        progress.flush();channel.put(('complete',worker,0,0))
    except BaseException:
        import traceback
        channel.put(('error',worker,traceback.format_exc(),0));raise


class Tests(unittest.TestCase):
    def test_source_bundle(self):
        d,p,t=run.c.source208(run.DEFAULT_BUNDLE)
        self.assertEqual(d['fingerprint'],run.c.SOURCE208);self.assertEqual(len(t),14)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.txt';path.write_bytes(run.DEFAULT_BUNDLE.read_bytes()+b'x')
            with self.assertRaises(ValueError):run.c.source208(path)

    def test_equation_same_weights_and_identity(self):
        import torch
        import test_stage206_spatial_adaptation as f
        torch.set_num_threads(1);helper=f.TorchTests();helper.torch=torch;native=helper.native();rt=native.rt
        m=run.prior.spatial.build(torch,8,4,4,2132,'matched_spatial')
        with torch.no_grad():m.output.weight.copy_(torch.eye(8))
        x=torch.randn(1,20,8);ref=torch.randn(1,4,8);outputs=[]
        for mode in run.c.MODES:
            model=run.c.view(torch,rt,m.state_dict(),2132,mode)
            self.assertEqual(run.old.model_digest(model),run.old.model_digest(m))
            y=model.adapt(x,ref);torch.testing.assert_close(y[:,:4],x[:,:4],rtol=0,atol=0)
            if mode=='zero':self.assertTrue(torch.equal(y,x))
            else:
                kv=x[:,:4] if mode=='matched' else ref if mode=='wrong' else torch.nn.functional.avg_pool2d(x[:,4:].transpose(1,2).reshape(1,8,4,4),2,2).flatten(2).transpose(1,2)
                norm=lambda v:torch.nn.functional.layer_norm(v,(8,))
                attention=(m.query(norm(x[:,4:]))@m.key(norm(kv)).transpose(-1,-2)/(8**.5)).softmax(-1)
                torch.testing.assert_close(y[:,4:],x[:,4:]+m.output(attention@m.value(norm(kv))))
                outputs.append(y)
        self.assertFalse(torch.equal(outputs[0],outputs[1]));self.assertFalse(torch.equal(outputs[0],outputs[2]))

    def test_bound_full_workflow_without_training(self):
        import torch
        import test_stage206_spatial_adaptation as f
        torch.set_num_threads(1);helper=f.TorchTests();helper.torch=torch;native=helper.native();rt=native.rt
        groups=[f'uav{i:07d}' for i in range(38)];folds=run.prior.common.source.common.protocol.split_groups(groups)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'209';src=Path(tmp)/'208';(root/'fixture').mkdir(parents=True);src.mkdir()
            specs=[];rng=np.random.default_rng(209)
            for g in groups:
                seq=g+'_00000_s';images=rng.normal(size=(3,20,8)).astype(np.float32)
                np.save(root/'fixture'/(seq+'.npy'),images);specs.append(dict(sequence=seq,group=g))
            entries=fixture208.fixture_entries(root,specs)
            sp=run.old.sealed(dict(entries=specs,folds=folds));tables=dict(scope_summary=[],reference_audit=[]);artifacts={}
            run.publish_identical(src/'manifest.json',sp)
            for fold in folds:
                for seed in run.c.DESIGN['seeds']:
                    m=run.prior.spatial.build(torch,8,4,4,seed,'matched_spatial')
                    with torch.no_grad():m.output.weight.fill_(.01)
                    scope=dict(state_sha256=run.old.model_digest(m),fold=fold['fold'],seed=seed,arm='matched_spatial')
                    meta=dict(plan_fingerprint=run.prior.training_plan(sp,'own_pair')['fingerprint'],fold=fold['fold'],seed=seed,
                              arm='matched_spatial',fit=fold['fit'],held=fold['held'],updates=256)
                    # Synthetic source receipt fixture, not a claim that this test trains 256 steps.
                    path=run.prior.model_path(src,fold['fold'],seed,'own_pair','matched_spatial')
                    run.old.save_model(torch,path,dict(state=m.state_dict(),scope=scope),meta);run.old.record(src,path,artifacts)
                    tables['scope_summary'].append(dict(training_condition='own_pair',**{k:str(v) for k,v in scope.items()}))
                    path=src/'references'/f"f{fold['fold']}__s{seed}.npz"
                    run.old.cache(path,dict(plan_fingerprint=sp['fingerprint'],fold=fold['fold'],seed=seed,fit=fold['fit']),
                        lambda:dict(groups=np.asarray(groups),reference=rng.normal(size=(38,4,8)).astype(np.float32)))
                    run.old.record(src,path,artifacts)
                    for g in groups:
                        donor=next(v for v in fold['fit'] if v!=g)
                        tables['reference_audit'].append(dict(fold=str(fold['fold']),seed=str(seed),group=g,donor=donor))
            sd=dict(artifact_sha256=artifacts);models,refs,scope,rr=run.c.load_models(src,sd,sp,tables,native,{})
            # Build the actual public-path source trajectories used by the production replay audit.
            for spec in specs:
                seq=spec['sequence'];fold,work=run.jobs(sp,spec,models);files,gt,_,reader,_=entries[seq]
                for mode,seed,model in work:
                    if mode not in ('native','matched'):continue
                    ref=None if model is None else torch.as_tensor(refs[fold,seed][spec['group']][None])
                    path=src/'evaluation'/f"{seq}__s{seed}__{'native' if mode=='native' else 'own_pair__matched_spatial'}.npz"
                    meta=run.metadata(sp,seq,fold,seed,mode,model,'full');meta['arm']='native' if mode=='native' else 'own_pair__matched_spatial'
                    run.old.cache(path,meta,lambda model=model,ref=ref:run.prior.previous.trace(native,model,ref,files,gt[0],reader,None,fixture208.Progress()))
                    run.old.record(src,path,artifacts)
            sd=dict(artifact_sha256=artifacts);source_before={p:run.file_hash(p) for p in src.rglob('*') if p.is_file()}
            plan=run.old.sealed(dict(entries=specs,folds=folds,workers=2,same_source_runtime=True,code_sha256={},protocol_sha256=run.file_hash(run.PROTOCOL)))
            run.publish_identical(root/'manifest.json',plan)
            config=dict(root=str(root),source_root=str(src),source_plan=sp,source_decision=sd,source_tables=tables)
            actual=run.prior.run_workers
            def workers(config,seqs,phase,n,progress,target=None):return actual(config,seqs,phase,n,progress,target=cpu_worker)
            with (patch.object(run.prior,'run_workers',side_effect=workers),patch.object(torch.optim.AdamW,'__init__',side_effect=AssertionError('No optimizer')),
                  contextlib.redirect_stdout(io.StringIO())):
                run.experiment(root,plan,native,models,refs,scope,rr,entries,None,config,sp,sd)
            d=run.old.read_sealed(root/'stage209_decision.json')
            self.assertEqual(d['full_evaluation_trajectories'],342);self.assertEqual(d['new_training_updates'],0);self.assertEqual(len(d['table_sha256']),10)
            self.assertEqual(len(d['primary_contrasts']),2)
            with contextlib.redirect_stdout(io.StringIO()):run.verify(root)
            self.assertTrue(all(run.file_hash(p)==sha for p,sha in source_before.items()))
            progress=fixture208.Progress()
            with patch.object(native,'step',side_effect=AssertionError('Must reuse')):
                run.evaluate_video(root,plan,native,models,refs,specs[0],entries[specs[0]['sequence']],None,progress,'full')
            self.assertEqual(progress.reused,18)
            trace={'boxes':np.zeros((2,4)),'pipeline_ms':np.zeros(2)};other=dict(trace,boxes=np.ones((2,4)))
            with self.assertRaises(AssertionError):run.equal_quality(trace,other)
            (root/'stage209_policy_summary.csv').write_text('changed')
            with self.assertRaises(ValueError):run.verify(root)


if __name__=='__main__':unittest.main()
