"""Real CENTER/loss CPU integration, factorial controls and spawned video workers."""
import contextlib
import csv
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'tools'),str(REPO/'tracking')]
import stage208_state_interaction as run


class Progress:
    def __init__(self):self.done=0;self.reused=0
    def advance(self,message='',count=1):self.done+=count


def fixture_entries(root,specs):
    entries={}
    for e in specs:
        seq=e['sequence'];images=np.load(Path(root)/'fixture'/(seq+'.npy'),allow_pickle=False)
        files=[f'{seq}/{i}' for i in range(len(images))];truth=np.tile([20.,20.,20.,20.],(len(images),1))
        reader=lambda dataset,path,images=images:images[int(path.split('/')[1])]
        entries[seq]=(files,truth,None,reader,'fixture')
    return entries


def fixture_worker(config,sequences,phase,worker,channel):
    """Spawn actual CPU models and run the production load/evaluate/cache path."""
    import torch
    import test_stage206_spatial_adaptation as fixture
    torch.set_num_threads(1)
    try:
        root=Path(config['root']);plan=run.old.read_sealed(root/'manifest.json')
        helper=fixture.TorchTests();helper.torch=torch;native=helper.native()
        models,refs=run.load_students(root,plan,native)
        entries=fixture_entries(root,plan['entries']);progress=run.WorkerProgress(channel,worker)
        for seq in sequences:
            spec=next(e for e in plan['entries'] if e['sequence']==seq)
            run.evaluate_video(root,plan,native,models,refs,spec,entries[seq],None,progress,phase)
        progress.flush();channel.put(('complete',worker,0,0))
    except BaseException:
        import traceback
        channel.put(('error',worker,traceback.format_exc(),0));raise


def failing_worker(config,sequences,phase,worker,channel):
    channel.put(('error',worker,'deliberate worker failure',0))


class PureTests(unittest.TestCase):
    def test_reviewed_input_and_family_analysis(self):
        d,p,t=run.common.source207(run.DEFAULT_BUNDLE)
        self.assertEqual(d['fingerprint'],run.common.SOURCE207);self.assertEqual(len(t),12)
        self.assertEqual(p['fingerprint'],run.common.PLAN207)
        with tempfile.TemporaryDirectory() as tmp:
            changed=Path(tmp)/'bundle.txt';changed.write_bytes(run.DEFAULT_BUNDLE.read_bytes()+b' ')
            with self.assertRaises(ValueError):run.common.source207(changed)
        rows=[]
        for g in range(38):
            for arm in run.common.ARMS:
                for seed in ([0] if arm=='native' else run.common.DESIGN['seeds']):
                    value=1+g/100+(arm.startswith('own_pair'))+(arm.endswith('matched_spatial') and not arm.endswith('mismatched_spatial'))
                    rows.append(dict(group=str(g),arm=arm,seed=seed,introduced_failure=False,
                                     **{m:value for m in run.common.METRICS}))
        summary,pairs,factors=run.common.summarize(rows)
        self.assertEqual(len(summary),7);self.assertEqual(len(pairs),65);self.assertEqual(len(factors),25)
        primary=[r for r in pairs if r['primary']]
        self.assertEqual(len(primary),2)
        for r in primary:
            self.assertGreaterEqual(r['ci_low'],r['family_ci_low']);self.assertLessEqual(r['ci_high'],r['family_ci_high'])
        for r in factors:
            if r['contrast'].startswith('state_source_by'):self.assertEqual(r['mean_delta'],0.)
        with self.assertRaises(ValueError):run.common.summarize(rows[:-1])

    def test_worker_failure_is_not_hang(self):
        with self.assertRaisesRegex(RuntimeError,'deliberate worker failure'):
            run.run_workers({},['a','b'],'smoke',2,Progress(),target=failing_worker)
        with self.assertRaises(ValueError):run.run_workers({},['a','a'],'full',2,Progress())


@unittest.skipUnless(importlib.util.find_spec('torch'),'CPU Torch required')
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch;torch.set_num_threads(1)

    def test_frozen_provider_bank_receipts_and_exact_support(self):
        import test_stage206_spatial_adaptation as fixture
        helper=fixture.TorchTests();helper.torch=self.torch;native=helper.native()
        model=run.spatial.build(self.torch,8,4,4,2132,'matched_spatial').eval().requires_grad_(False)
        with tempfile.TemporaryDirectory() as tmp,patch.object(fixture,'sample_target',
                side_effect=lambda image,*a,**k:(image,4.,np.zeros(image.shape[:2],bool))):
            root=Path(tmp);entries={};bank={};values={};specs=[]
            folds=[dict(fold=0,fit=['a'],held=['b']),dict(fold=1,fit=['b'],held=['a'])]
            for j,g in enumerate(('a','b')):
                seq=g+'_seq';images=np.random.default_rng(j).normal(size=(3,20,8)).astype(np.float32)
                files=[str(i) for i in range(3)];truth=np.tile([2.,2.,1.,1.],(3,1))
                reader=lambda dataset,path,images=images:images[int(path)]
                entries[seq]=(files,truth,None,reader,'fixture');specs.append(dict(group=g,sequence=seq))
                value=run.source.collect_provider(native,model,files,truth[0],reader,None,np.array([1,2]),Progress())
                values[g]=value;bank[g]=run.source.attach_truth(value,truth,256)
            plan=run.old.sealed(dict(entries=specs,folds=folds,same_source_runtime=True));artifacts={};table={'provider_scope':[],'support':[],'sampling':[]}
            for fold in folds:
                for seed in run.common.DESIGN['seeds']:
                    table['provider_scope'].append(dict(fold=str(fold['fold']),seed=str(seed),model_sha256='frozen-provider'))
                    for g in fold['fit']:
                        path=root/'providers'/f"f{fold['fold']}__s{seed}__{g}_seq.npz"
                        meta=dict(plan_fingerprint=plan['fingerprint'],fold=fold['fold'],seed=seed,group=g,
                                  provider_sha256='frozen-provider',role='outer_fit_only')
                        run.old.cache(path,meta,lambda g=g:values[g]);run.old.record(root,path,artifacts)
                    a={g:bank[g] for g in fold['fit']}
                    _,_,s,m=run.source.common.paired_bank(a,a,256)
                    table['support'].extend({k:str(v) for k,v in dict(fold=fold['fold'],seed=seed,**r).items()} for r in s)
                    table['sampling'].extend({k:str(v) for k,v in dict(fold=fold['fold'],seed=seed,**r).items()} for r in m)
            decision=dict(artifact_sha256=artifacts);verified={}
            b,conditions,s,m=run.common.paired_inputs(root,decision,plan,table,bank,entries,256,verified)
            self.assertIs(b,bank);self.assertEqual(len(conditions),4);self.assertEqual(len(verified),8)
            for cell in conditions.values():
                for g in cell['native_pair']:np.testing.assert_array_equal(cell['native_pair'][g]['tokens'],cell['own_pair'][g]['tokens'])
            table['sampling'][0]['paired']='False'
            with self.assertRaisesRegex(ValueError,'support/sampling changed'):
                run.common.paired_inputs(root,decision,plan,table,bank,entries,256,{})
            rel=next(k for k in artifacts if k.endswith('.npz'))
            with self.assertRaisesRegex(ValueError,'receipt'):
                run.common.bound_arrays(root,decision,rel,dict(plan_fingerprint='wrong'),{})

    def test_factorial_full_pipeline_control_replay_spawn_resume_and_tamper(self):
        import test_stage206_spatial_adaptation as fixture
        helper=fixture.TorchTests();helper.torch=self.torch;native=helper.native();rt=native.rt
        groups=[f'uav{i:07d}' for i in range(38)];folds=run.common.source.common.protocol.split_groups(groups)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'208';root.mkdir();(root/'fixture').mkdir();bank={};specs=[]
            rng=np.random.default_rng(208)
            for g in groups:
                seq=g+'_00000_s';images=rng.normal(size=(3,20,8)).astype(np.float32)
                np.save(root/'fixture'/(seq+'.npy'),images)
                tokens=self.torch.as_tensor(images[1:])
                with self.torch.no_grad():
                    for block in rt.backbone.blocks[:6]:tokens=block(tokens)
                bank[g]=dict(frame_id=np.array([1,2]),tokens=tokens.numpy(),truth=np.tile([20.,20.,20.,20.],(2,1)),
                    parent=np.tile([20.,20.,20.,20.],(2,1)),factor=np.full(2,4.),image_hw=np.full((2,2),200),
                    target=np.tile([.4,.4,.2,.2],(2,1)),native_box=np.tile([20.,20.,20.,20.],(2,1)),
                    native_layer=run.spatial.chosen(rt,tokens).numpy())
                specs.append(dict(sequence=seq,group=g))
            entries=fixture_entries(root,specs);conditions={};support=[];sampling=[]
            for fold in folds:
                for seed in run.common.DESIGN['seeds']:
                    a={g:bank[g] for g in fold['fit']}
                    b={g:{k:v.copy() for k,v in a[g].items()} for g in a}
                    for g in b:b[g]['tokens']+=np.float32(.01)
                    a,b,s,m=run.source.common.paired_bank(a,b,256)
                    self.assertTrue(all(r['paired_states']==2 for r in s))
                    conditions[fold['fold'],seed]={'native_pair':a,'own_pair':b}
                    support.extend(dict(fold=fold['fold'],seed=seed,**r) for r in s)
                    sampling.extend(dict(fold=fold['fold'],seed=seed,**r) for r in m)
            plan=run.old.sealed(dict(schema=run.SCHEMA,entries=specs,folds=folds,same_source_runtime=True,workers=2,
                code_sha256={},protocol_sha256=run.file_hash(run.PROTOCOL_FILE)))
            run.publish_identical(root/'manifest.json',plan)
            controls={};source_root=Path(tmp)/'207';artifacts={};source_scopes=[]
            for fold in folds:
                for seed in run.common.DESIGN['seeds']:
                    refs,mapping=run.common.references(conditions[fold['fold'],seed]['native_pair'],fold,groups,4)
                    self.assertTrue(all(v in fold['fit'] and v!=g for g,v in mapping.items()))
                    for origin in run.common.ORIGINS:
                        model,scope,_=run.previous.fit(source_root/'conditions'/origin,plan,native,
                            conditions[fold['fold'],seed][origin],refs,fold,seed,'matched_spatial',Progress(),{})
                        controls[fold['fold'],seed,origin]={k:scope[k] for k in ('state_sha256','draws_sha256','initial_sha256')}
                        source_scopes.append(dict(training_condition=origin,**{k:str(v) for k,v in scope.items()}))
            # Exercise exact source student receipt/scope binding, not just the in-memory fixture controls.
            for path in source_root.rglob('*'):
                if path.is_file():artifacts[path.relative_to(source_root).as_posix()]=run.file_hash(path)
            loaded=run.common.control_states(self.torch,rt,source_root,dict(artifact_sha256=artifacts),plan,
                                             dict(scope_summary=source_scopes),{})
            self.assertEqual(loaded,controls)
            source_before={p:run.file_hash(p) for p in source_root.rglob('*') if p.is_file()}
            config=dict(root=str(root))
            actual_run_workers=run.run_workers
            def cpu_workers(config,sequences,phase,workers,progress):
                return actual_run_workers(config,sequences,phase,workers,progress,target=fixture_worker)
            with contextlib.redirect_stdout(io.StringIO()),patch.object(run,'run_workers',side_effect=cpu_workers),\
                 patch.object(self.torch.cuda,'_lazy_init',side_effect=AssertionError('No CUDA')):
                run.experiment(root,plan,native,entries,None,bank,conditions,support,sampling,controls,config)
            d=run.old.read_sealed(root/'stage208_decision.json')
            self.assertEqual(d['training_runs'],24);self.assertEqual(d['training_updates'],6144)
            self.assertEqual(d['full_evaluation_trajectories'],494);self.assertEqual(len(d['table_sha256']),14)
            def read(name):
                with (root/f'stage208_{name}.csv').open(encoding='utf8',newline='') as h:return list(csv.DictReader(h))
            self.assertEqual(len(read('parallel_smoke')),26)
            self.assertTrue(all(r['exact_quality_and_call_match']=='True' for r in read('parallel_smoke')))
            self.assertTrue(all(r['exact_state_match']=='True' for r in read('control_replay')))
            for fold in folds:
                for seed in run.common.DESIGN['seeds']:
                    scope=[r for r in read('scope_summary') if int(r['fold'])==fold['fold'] and int(r['seed'])==seed]
                    self.assertEqual(len(scope),6)
                    for key in ('initial_sha256','draws_sha256','trainable_parameters'):self.assertEqual(len({r[key] for r in scope}),1)
            # Finished verification is read-only and binds every evaluation cache and all models.
            with contextlib.redirect_stdout(io.StringIO()):run.verify(root)
            models,refs=run.load_students(root,plan,native);progress=Progress()
            with patch.object(native,'step',side_effect=AssertionError('Must reuse committed trajectory')):
                run.evaluate_video(root,plan,native,models,refs,specs[0],entries[specs[0]['sequence']],None,progress,'full')
            self.assertEqual(progress.done,26);self.assertEqual(progress.reused,26)
            self.assertTrue(all(run.file_hash(p)==sha for p,sha in source_before.items()))
            # Inject an internally sealed but numerically different smoke result; quality comparison must reject it.
            path=root/'smoke_parallel/evaluation'/f"{specs[0]['sequence']}__s0__native.npz"
            receipt=run.old.read_sealed(path.with_suffix('.receipt.json'))
            with np.load(path,allow_pickle=False) as f:data={k:f[k] for k in f.files if k!='metadata_json'}
            data['boxes']=data['boxes'].copy();data['boxes'][0,0]+=.01
            run.old.write_cache(path,data,receipt['metadata']);receipt['sha256']=run.file_hash(path)
            path.with_suffix('.receipt.json').write_text(__import__('json').dumps(run.old.sealed({k:v for k,v in receipt.items() if k!='fingerprint'})))
            with self.assertRaises((ValueError,AssertionError)):run.compare_smoke(root,plan,models,specs[:2],entries,{})
            with self.assertRaises(ValueError):run.verify(root)
            failure=root/'conditions/own_pair/nonfinite_failure.json';failure.write_text('{}')
            with self.assertRaises(ValueError):run.guard(root)


if __name__=='__main__':unittest.main()
