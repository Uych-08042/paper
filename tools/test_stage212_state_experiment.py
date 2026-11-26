"""Online state chains, real CENTER decode, budget parity and spawned CPU OOF."""
import contextlib,io,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tools'),str(Path(__file__).resolve().parents[1]/'tracking')]
import stage212_state_experiment as run
import test_stage210_recovery_audit as f


class Progress:
    reused=0
    def advance(self,*a,**kw):pass


def cpu_worker(config,sequences,phase,worker,channel):
    try:
        root=Path(config['root']);plan=run.old.read_sealed(root/'manifest.json');native=f.make_native();models=f.models_for(native,plan['folds'])
        entries=f.fixture_entries(root,plan['entries'],config['truths']);progress=run.runtime.WorkerProgress(channel,worker)
        for seq in sequences:
            spec=next(e for e in plan['entries'] if e['sequence']==seq)
            run.evaluate_video(root,plan,native,models,spec,entries[seq],None,progress,phase)
        progress.flush();channel.put(('complete',worker,0,0))
    except BaseException:
        import traceback
        channel.put(('error',worker,traceback.format_exc(),0));raise


class Tests(unittest.TestCase):
    def test_source_and_online_geometry(self):
        d,p,t=run.c.source211(run.DEFAULT_BUNDLE)
        self.assertEqual(d['fingerprint'],run.c.SOURCE211);self.assertEqual(len(t),10)
        raw=[20.,30.,10.,12.];adapt=[1.,2.,4.,6.]
        for mode in run.c.MODES:
            state=run.c.next_state(raw,adapt,mode);center=raw if mode[0]=='n' else adapt;size=raw if mode[1]=='n' else adapt
            np.testing.assert_allclose(state[:2]+state[2:]/2,np.asarray(center)[:2]+np.asarray(center)[2:]/2)
            np.testing.assert_array_equal(state[2:],size[2:])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.txt';path.write_bytes(run.DEFAULT_BUNDLE.read_bytes()+b'x')
            with self.assertRaises(ValueError):run.c.source211(path)

    def test_state_chain_decode_and_hook_cleanup(self):
        native=f.make_native();folds=[dict(fold=0,fit=[],held=[])];models=f.models_for(native,folds);model=models[0,2132,'matched']
        images=np.random.default_rng(212).normal(size=(4,20,8)).astype(np.float32);files=['0','1','2','3'];reader=lambda dataset,path:images[int(path)]
        before={k:len(v._forward_hooks) for k,v in enumerate(native.rt.backbone.blocks)}
        baseline=run.runtime.previous.trace(native,None,None,files,[2,2,1,1],reader,None,Progress())
        single=run.runtime.previous.trace(native,model,None,files,[2,2,1,1],reader,None,Progress())
        values={('native',0):baseline}
        for seed in run.c.SEEDS:
            m=models[0,seed,'matched'];values['matched_single',seed]=run.runtime.previous.trace(native,m,None,files,[2,2,1,1],reader,None,Progress())
            for mode in run.c.MODES:
                v=run.dual_trace(native,m,mode,files,[2,2,1,1],reader,None,Progress());run.c.audit(v,[2,2,1,1],3,mode);values[mode,seed]=v
        run.check_controls(values,'s','test')
        self.assertEqual({k:len(v._forward_hooks) for k,v in enumerate(native.rt.backbone.blocks)},before)
        self.assertEqual(len(native.rt.net.box_head._forward_hooks),0)
        bad={k:v.copy() for k,v in values['nm',2132].items()};bad['parent'][1,0]+=.1
        with self.assertRaises(AssertionError):run.c.audit(bad,[2,2,1,1],3,'nm')
        bad={k:v.copy() for k,v in values['mm',2132].items()};bad['head_calls'][:]=1
        with self.assertRaises(AssertionError):run.c.audit(bad,[2,2,1,1],3,'mm')
        # Three-frame recovery loss remains measured at the reference recovery time.
        gt=np.tile([0.,0.,10.,10.],(10,1));badbox=np.tile([100.,100.,10.,10.],(9,1));good=gt[1:].copy()
        ref=dict(boxes=np.vstack((badbox[:3],good[3:])),pipeline_ms=np.ones(9));trace=dict(boxes=badbox,pipeline_ms=np.ones(9))
        row=run.old.describe_trace(trace,ref,gt,'uav0000001_00000_s','test',2132,0)
        self.assertEqual(row['missed_native_recoveries'],1)

    def test_complete_spawned_pipeline_and_statistics(self):
        import torch
        native=f.make_native();groups=[f'uav{i:07d}' for i in range(38)];folds=run.c.PROTOCOL.split_groups(groups);models=f.models_for(native,folds)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'fixture').mkdir();specs=[];truths={};rng=np.random.default_rng(212)
            for g in groups:
                seq=g+'_00000_s';specs.append(dict(sequence=seq,group=g));images=rng.normal(size=(3,20,8)).astype(np.float32)
                np.save(root/'fixture'/(seq+'.npy'),images);truths[seq]=np.tile([2.,2.,1.,1.],(3,1)).tolist()
            entries=f.fixture_entries(root,specs,truths)
            plan=run.old.sealed(dict(entries=specs,folds=folds,workers=2,code_sha256={},protocol_sha256=run.file_hash(run.PROTOCOL)))
            run.publish_identical(root/'manifest.json',plan);config=dict(root=str(root),truths=truths)
            scope=[dict(fold=fold['fold'],seed=seed,state_sha256=run.old.model_digest(models[fold['fold'],seed,'matched'])) for fold in folds for seed in run.c.SEEDS]
            actual=run.runtime.run_workers
            def workers(config,seqs,phase,n,progress,target=None):return actual(config,seqs,phase,n,progress,target=cpu_worker)
            with patch.object(run.runtime,'run_workers',side_effect=workers),patch.object(torch.optim.AdamW,'__init__',side_effect=AssertionError('No training')),contextlib.redirect_stdout(io.StringIO()):
                run.experiment(root,plan,native,models,scope,entries,None,config)
            d=run.old.read_sealed(root/'stage212_decision.json')
            self.assertEqual(d['full_evaluation_trajectories'],418);self.assertEqual(d['new_training_updates'],0);self.assertEqual(len(d['primary_contrasts']),2)
            with contextlib.redirect_stdout(io.StringIO()):run.verify(root)
            import csv
            with (root/'stage212_seed_video_summary.csv').open(encoding='utf8',newline='') as h:rows=list(csv.DictReader(h))
            typed=[]
            for r in rows:
                r=dict(r)
                for m in run.c.METRICS:r[m]=float(r[m])
                r['seed']=int(r['seed']);r['introduced_failure']=r['introduced_failure']=='True';r['introduced_matched_failure']=r['introduced_matched_failure']=='True';typed.append(r)
            s,p,factors=run.c.summarize(typed);self.assertEqual(len(s),6);self.assertEqual(len(factors),21)
            self.assertEqual(len([r for r in p if r['primary']]),2)
            for r in s:
                if r['arm']=='mm':self.assertEqual(r['missed_matched_recoveries'],0.);self.assertEqual(r['matched_good_frame_loss'],0.)
            for r in p:
                if r['primary']:self.assertLessEqual(r['family_ci_low'],r['ci_low']);self.assertGreaterEqual(r['family_ci_high'],r['ci_high'])
            with patch.object(native,'step',side_effect=AssertionError('Must reuse committed trace')):
                run.evaluate_video(root,plan,native,models,specs[0],entries[specs[0]['sequence']],None,Progress(),'full')
            (root/'stage212_policy_summary.csv').write_text('changed')
            with self.assertRaises(ValueError):run.verify(root)


if __name__=='__main__':unittest.main()
