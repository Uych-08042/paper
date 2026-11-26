"""Actual crop support, provider identity, raw-prefix capture and full CPU workflow."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'tools'),str(REPO/'tracking')]
import stage207_state_training as run


class Progress:
    def __init__(self):self.done=0;self.reused=0
    def advance(self,*args,**kwargs):self.done+=kwargs.get('count',1)


class PureTests(unittest.TestCase):
    def test_reviewed_bundle_and_integer_padding_support(self):
        d,p,t=run.source206(run.DEFAULT_BUNDLE)
        self.assertEqual(d['fingerprint'],run.SOURCE206);self.assertEqual(len(t),9)
        # Original sample_target excludes the final row/column when padding.
        self.assertTrue(run.common.covered([4,4,2,2],[3,3,2,2],4,[20,20],16))
        self.assertFalse(run.common.covered([18,4,2,2],[17,3,2,2],4,[20,20],16))
        self.assertTrue(run.common.covered([0,0,1,1],[0,0,1,1],4,[20,20],16))
        self.assertFalse(run.common.covered([-1,0,1,1],[0,0,1,1],4,[20,20],16))
        with self.assertRaises(ValueError):run.common.covered([0,0,1,1],[0,0,1,1],float('nan'),[20,20],16)
        # Round(1.5)=2, so a truth ending at x=2 is outside this actual crop.
        self.assertFalse(run.common.covered([1.5,2,1,1],[2.5,3,2,2],4,[20,20],16))

    def test_pair_identity_and_no_silent_empty_source(self):
        def data(parents):
            return dict(frame_id=np.array([1,2]),truth=np.array([[4,4,1,1],[4,4,1,1]]),parent=np.asarray(parents),
                factor=np.array([4.,4.]),image_hw=np.array([[20,20],[20,20]]),tokens=np.zeros((2,5,3)),target=np.zeros((2,4)))
        native={'a':data([[4,4,1,1],[4,4,1,1]]),'b':data([[4,4,1,1],[4,4,1,1]])}
        own={'a':data([[4,4,1,1],[15,15,1,1]]),'b':data([[15,15,1,1],[15,15,1,1]])}
        a,b,support,rows=run.common.paired_bank(native,own,16)
        np.testing.assert_array_equal(a['a']['frame_id'],b['a']['frame_id']);self.assertEqual(len(a['a']['frame_id']),1)
        self.assertEqual(support[1]['paired_states'],0);self.assertIn('b',a)
        fold=dict(fold=0,fit=['a'],held=['b'])
        self.assertEqual(run.common.protocol.draws_for(fold,2132,{'a':1}),run.common.protocol.draws_for(fold,2132,{'a':1}))


@unittest.skipUnless(importlib.util.find_spec('torch'),'CPU Torch required')
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch;torch.set_num_threads(1)

    def setup_corpus(self):
        import test_stage206_spatial_adaptation as fixture
        helper=fixture.TorchTests();helper.torch=self.torch;native=helper.native();rt=native.rt
        groups=[f'uav{i:07d}' for i in range(38)];rng=np.random.default_rng(207);entries={};specs=[];bank={}
        zero=run.spatial.build(self.torch,8,4,4,2132,'matched_spatial').eval().requires_grad_(False)
        for g in groups:
            seq=g+'_00000_s';images=rng.normal(size=(3,20,8)).astype(np.float32)
            files=[f'{seq}/{i}' for i in range(3)];truth=np.tile([2.,2.,1.,1.],(3,1))
            reader=lambda dataset,path,images=images:images[int(path.split('/')[1])]
            entries[seq]=(files,truth,None,reader,'fixture');specs.append(dict(group=g,sequence=seq,sample_frames=[1,2]))
            value=run.collect_provider(native,zero,files,truth[0],reader,None,np.array([1,2]),Progress())
            bank[g]=run.attach_truth(value,truth,256)
        return fixture,native,entries,specs,bank,run.common.protocol.split_groups(groups)

    def test_raw_provider_capture_and_future_truth_not_actor_input(self):
        import test_stage206_spatial_adaptation as fixture
        with patch.object(fixture,'sample_target',side_effect=lambda image,*a,**k:(image,4.,np.zeros(image.shape[:2],bool))):
            helper=fixture.TorchTests();helper.torch=self.torch;native=helper.native();rt=native.rt
            images=np.random.default_rng(7).normal(size=(3,20,8)).astype(np.float32)
            model=run.spatial.build(self.torch,8,4,4,2132,'matched_spatial')
            with self.torch.no_grad():model.output.weight.copy_(self.torch.eye(8))
            files=['0','1','2'];reader=lambda dataset,path:images[int(path)]
            value=run.collect_provider(native,model,files,[2,2,1,1],reader,None,np.array([1,2]),Progress())
            tokens=self.torch.as_tensor(images[1:2])
            with self.torch.no_grad():
                for block in rt.backbone.blocks[:6]:tokens=block(tokens)
            np.testing.assert_array_equal(tokens.numpy()[0],value['sample_tokens'][0])
            self.assertFalse(self.torch.equal(model.adapt(tokens),tokens))
            gt=np.tile([2.,2.,1.,1.],(3,1));changed=gt.copy();changed[1:,:2]+=1000
            first=run.attach_truth(value,gt,256);second=run.attach_truth(value,changed,256)
            np.testing.assert_array_equal(first['tokens'],second['tokens']);np.testing.assert_array_equal(first['parent'],second['parent'])
            self.assertFalse(np.array_equal(first['target'],second['target']))
            self.assertEqual(len(rt.backbone.blocks[5]._forward_hooks),0)
            with tempfile.TemporaryDirectory() as tmp:
                plan=dict(fingerprint='current-runtime-test',entries=[dict(group='g',sequence='seq')])
                entries={'seq':(files,gt,None,reader,'fixture')};artifacts={}
                replay,ledger=run.replay_native(Path(tmp),plan,native,entries,None,{'g':first},Progress(),artifacts)
                np.testing.assert_array_equal(replay['g']['parent'],first['parent'])
                np.testing.assert_array_equal(replay['g']['target'],first['target'])
                self.assertTrue(ledger[0]['recomputed']);self.assertEqual(len(artifacts),2)
                with patch.object(native,'step',side_effect=AssertionError('Must reuse current-GPU replay')):
                    run.replay_native(Path(tmp),plan,native,entries,None,{'g':first},Progress(),{})

    def test_real_center_students_provider_binding_complete_pipeline_and_support_stop(self):
        import test_stage206_spatial_adaptation as fixture
        with patch.object(fixture,'sample_target',side_effect=lambda image,*a,**k:(image,4.,np.zeros(image.shape[:2],bool))),\
             patch.object(self.torch.cuda,'_lazy_init',side_effect=AssertionError('No CUDA')):
            _,native,entries,specs,bank,folds=self.setup_corpus()
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'207';source=Path(tmp)/'206';source.mkdir();artifacts={};providers={};scopes=[]
                source_plan=run.old.sealed(dict(folds=folds));source_rows=[]
                refs={g:np.zeros((4,8),np.float32) for g in bank}
                for fold in folds:
                    for seed in run.common.DESIGN['seeds']:
                        model,scope,_=run.previous.fit(source,source_plan,native,bank,refs,fold,seed,'matched_spatial',Progress(),artifacts)
                        source_rows.append({k:str(v) for k,v in scope.items()})
                decision=dict(artifact_sha256=artifacts)
                verified={}
                for fold in folds:
                    for seed in run.common.DESIGN['seeds']:
                        model,scope=run.load_provider(self.torch,native.rt,source,decision,source_plan,dict(scope_summary=source_rows),fold,seed,verified)
                        providers[fold['fold'],seed]=model;scopes.append(scope)
                self.assertEqual(len(verified),8)
                plan=run.old.sealed(dict(schema=run.SCHEMA,folds=folds,entries=specs,same_source_runtime=True,code_sha256={},protocol_sha256=run.file_hash(run.PROTOCOL_FILE)))
                run.publish_identical(root/'manifest.json',plan)
                with contextlib.redirect_stdout(io.StringIO()):run.experiment(root,plan,native,entries,None,bank,providers,scopes)
                d=run.old.read_sealed(root/'stage207_decision.json')
                self.assertEqual(d['training_runs'],12);self.assertEqual(d['training_updates'],3072)
                self.assertEqual(d['provider_fit_trajectories'],76);self.assertEqual(d['full_evaluation_trajectories'],342)
                self.assertEqual(len(d['table_sha256']),12)
                self.assertEqual(d['primary_contrast']['contrast'],'own_pair_minus_native_pair')
                import csv
                def read(name):
                    with (root/f'stage207_{name}.csv').open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
                self.assertTrue(all(v['exact_state_match']=='True' for v in read('control_replay')))
                for fold in folds:
                    for seed in run.common.DESIGN['seeds']:
                        cell=[r for r in read('scope_summary') if int(r['fold'])==fold['fold'] and int(r['seed'])==seed]
                        pair=[r for r in cell if r['training_condition'] in ('native_pair','own_pair')]
                        self.assertEqual(pair[0]['draws_sha256'],pair[1]['draws_sha256'])
                original={p:run.file_hash(p) for p in source.rglob('*') if p.is_file()}
                with contextlib.redirect_stdout(io.StringIO()):run.verify(root)
                # A no-support preflight must collect/report, never train or silently drop sources.
                stopped=Path(tmp)/'stopped';run.publish_identical(stopped/'manifest.json',plan)
                with patch.object(run.common,'covered',return_value=False),patch.object(run.previous,'fit',side_effect=AssertionError('Must not train')),contextlib.redirect_stdout(io.StringIO()):
                    run.experiment(stopped,plan,native,entries,None,bank,providers,scopes);run.verify(stopped)
                self.assertEqual(run.old.read_sealed(stopped/'stage207_support_decision.json')['status'],'INSUFFICIENT_PAIRED_SUPPORT')
                self.assertFalse((stopped/'stage207_decision.json').exists())
                for path,sha in original.items():self.assertEqual(run.file_hash(path),sha)
                (root/'stage207_policy_summary.csv').write_text('changed')
                with self.assertRaises(ValueError):run.verify(root)
                failure=root/'conditions/own_pair/nonfinite_failure.json';failure.write_text('{}')
                with self.assertRaises(ValueError):run.guard(root)


if __name__=='__main__':unittest.main()
