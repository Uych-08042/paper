"""Spatial equation, routing isolation, real CENTER learning and full OOF orchestration."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'tools'),str(REPO/'tracking')]
import stage206_spatial_adaptation as run


def sample_target(image,*args,**kwargs):return image


class PureTests(unittest.TestCase):
    def test_reviewed_source_and_split_only_reference(self):
        d,p=run.source205(run.DEFAULT_BUNDLE)
        self.assertEqual(d['fingerprint'],run.SOURCE205)
        groups=[f'uav{i:07d}' for i in range(38)];fold=run.common.old.split_groups(groups)[0]
        bank={g:dict(tokens=np.full((2,5,3),i,dtype=np.float32)) for i,g in enumerate(groups)}
        refs,mapping=run.common.references(bank,fold,1)
        self.assertTrue(all(v in fold['fit'] and v!=g for g,v in mapping.items()))
        for g in fold['held']:bank[g]['tokens']+=1000
        again,_=run.common.references(bank,fold,1)
        for g in groups:np.testing.assert_array_equal(refs[g],again[g])
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'bad.txt';file.write_text('changed')
            with self.assertRaises(ValueError):run.source205(file)


@unittest.skipUnless(importlib.util.find_spec('torch'),'CPU Torch required')
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch;torch.set_num_threads(1)

    def native(self):
        torch=self.torch
        package=ModuleType('lib.models');package.__path__=[str(REPO/'lib/models')]
        with patch.dict(sys.modules,{'lib.models':package}):
            from lib.models.layers.head import CenterPredictor
        torch.manual_seed(206)
        bb=torch.nn.Module();bb.blocks=torch.nn.ModuleList([torch.nn.Linear(8,8) for _ in range(12)])
        bb.MLP=torch.nn.Linear(20,6);bb.norm=torch.nn.LayerNorm(8);bb.cat_mode='direct'
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__();self.backbone=bb;self.box_head=CenterPredictor(8,channel=16,feat_sz=4)
            def forward_head(self,feature,gt_score_map=None):
                x=feature[:,-16:].transpose(1,2).reshape(-1,8,4,4)
                score,box,size,offset=self.box_head(x,gt_score_map)
                return dict(score_map=score,pred_boxes=box[:,None],size_map=size,offset_map=offset)
        net=Net().eval().requires_grad_(False)
        class Tracker:
            def track(self,image):return sample_target(image)
        tracker=Tracker();tracker.params=SimpleNamespace(search_size=256);tracker.output_window=torch.ones(4,4)
        rt=SimpleNamespace(torch=torch,net=net,backbone=bb,lens_z=4,lens_x=16,search_side=4,identity=dict(embed_dim=8),tracker=tracker,
            s1=SimpleNamespace(recover_tokens=lambda x,*a,**kw:x))
        rt.selector=lambda x:bb.MLP(x[:,:,0].clone())
        rt.selected_layer=lambda p:int(torch.topk(p,1,dim=1).indices[0,0])+7
        def candidate(x,layer):return net.forward_head(bb.norm(bb.blocks[layer-1](x.clone())))
        rt.candidate=candidate
        class Native:
            def __init__(self):self.rt=rt;self.initial=run.old.model_digest(net)
            def initialize(self,image,box):tracker.frame_id=0;tracker.state=np.asarray(box,float)
            def assert_frozen(self):
                if run.old.model_digest(net)!=self.initial:raise AssertionError('Native state changed')
            def step(self,image,parent):
                tracker.track(image);tracker.frame_id+=1
                with torch.no_grad():
                    x=torch.as_tensor(image[None],dtype=torch.float32)
                    for block in bb.blocks[:6]:x=block(x)
                    layer=rt.selected_layer(rt.selector(x));output=candidate(x,layer)
                    grid=run.common.old.dense_image_boxes(output['size_map'][0].numpy(),output['offset_map'][0].numpy(),parent,4.,256,[200,200])
                    cell=int(output['score_map'].flatten(1).argmax(1)[0]);box=grid[cell]
                calls=np.zeros(12,int);calls[:6]=1;calls[layer-1]=1;tracker.state=box
                return dict(box=box,layer=layer,block_calls=calls,head_calls=1,selector_calls=1)
        return Native()

    def test_equation_zero_identity_parameters_and_native_mlp_input(self):
        torch=self.torch;native=self.native();rt=native.rt;x=torch.randn(2,20,8)
        snapshots=[]
        for arm in run.spatial.ARMS:
            model=run.spatial.build(torch,8,4,4,2132,arm);ref=x[:,:4].clone()
            self.assertTrue(torch.equal(model.adapt(x,ref),x));snapshots.append(run.old.model_digest(model))
            with torch.no_grad():model.output.weight.copy_(torch.eye(8))
            result=model.adapt(x,ref)
            torch.testing.assert_close(result[:,:4],x[:,:4],rtol=0,atol=0)
            # Independent explicit attention calculation, including the pooled-search control.
            source=rt.torch.nn.functional.avg_pool2d(x[:,4:].transpose(1,2).reshape(2,8,4,4),2,2).flatten(2).transpose(1,2) if arm=='search_control' else ref
            norm=lambda v:torch.nn.functional.layer_norm(v,(8,))
            attention=(model.query(norm(x[:,4:]))@model.key(norm(source)).transpose(-1,-2)/(8**.5)).softmax(-1)
            torch.testing.assert_close(result[:,4:],x[:,4:]+model.output(attention@model.value(norm(source))))
        self.assertEqual(len(set(snapshots)),1)
        real=run.spatial.build(torch,192,64,16,2132,'matched_spatial')
        self.assertEqual(sum(p.numel() for p in real.parameters()),147456)
        model=run.spatial.build(torch,8,4,4,2132,'matched_spatial')
        with torch.no_grad():model.output.weight.fill_(10.)
        image=np.random.default_rng(3).normal(size=(20,8)).astype(np.float32)
        with torch.no_grad():
            tokens=torch.as_tensor(image[None]);original=tokens
            for block in rt.backbone.blocks[:6]:original=block(original)
            expected=rt.selector(original)
            native.initialize(image,[20,20,20,20])
            with run.spatial.Installed(native,model,None) as installed:
                native.step(image,[20,20,20,20])
                torch.testing.assert_close(installed.last_scores,expected,rtol=0,atol=0)
                self.assertEqual(installed.route_calls,1)
        self.assertEqual(len(rt.backbone.blocks[5]._forward_hooks),0)
        self.assertEqual(len(rt.backbone.MLP._forward_pre_hooks),0)

    def test_complete_cpu_12_model_266_trace_pipeline_and_reuse(self):
        torch=self.torch;native=self.native();rt=native.rt
        groups=[f'uav{i:07d}' for i in range(38)];rng=np.random.default_rng(206);bank={};entries={};specs=[]
        images={}
        for g in groups:
            seq=g+'_00000_s';images[seq]=rng.normal(size=(3,20,8)).astype(np.float32)
            files=[f'{seq}/{i}' for i in range(3)];gt=np.tile([20.,20.,20.,20.],(3,1))
            reader=lambda dataset,path:images[path.split('/')[0]][int(path.split('/')[1])]
            entries[seq]=(files,gt,None,reader,'fixture');specs.append(dict(group=g,sequence=seq))
            tokens=torch.as_tensor(images[seq][1:])
            with torch.no_grad():
                for block in rt.backbone.blocks[:6]:tokens=block(tokens)
            bank[g]=dict(tokens=tokens.numpy(),frame_id=np.array([1,2]),target=np.tile([.4,.4,.2,.2],(2,1)),
                parent=np.tile(gt[0],(2,1)),factor=np.full(2,4.),image_hw=np.full((2,2),200),truth=gt[1:],native_box=gt[1:],
                native_layer=run.spatial.chosen(rt,tokens).numpy())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=run.old.sealed(dict(schema=run.SCHEMA,entries=specs,folds=run.common.old.split_groups(groups),
                code_sha256={},protocol_sha256=run.file_hash(run.PROTOCOL_FILE)))
            run.publish_identical(root/'manifest.json',plan)
            with contextlib.redirect_stdout(io.StringIO()),patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('No CUDA')):
                run.experiment(root,plan,native,entries,Path(tmp),bank)
            d=run.old.read_sealed(root/'stage206_decision.json')
            self.assertEqual(d['training_runs'],12);self.assertEqual(d['training_updates'],3072);self.assertEqual(d['full_trajectories'],266)
            self.assertEqual(len(d['primary_contrasts']),3);self.assertEqual(len(d['table_sha256']),9)
            # Resume the actual fit helper, binding the same original fixture model and draws.
            refs,_=run.common.references(bank,plan['folds'][0],4)
            progress=SimpleNamespace(done=0,reused=0)
            def advance(*a,**kw):progress.done+=kw.get('count',1)
            progress.advance=advance
            model,scope,h=run.fit(root,plan,native,bank,refs,plan['folds'][0],2132,'matched_spatial',progress,{})
            self.assertEqual(progress.reused,256);self.assertTrue(scope['frozen_native_unchanged']);self.assertEqual(len(h),256)
            value=run.old.read_sealed(root/'training/f0__s2132__matched_spatial.receipt.json')
            self.assertEqual(scope['state_sha256'],run.old.model_digest(model));self.assertEqual(value['metadata']['fit'],plan['folds'][0]['fit'])
            self.assertTrue(all(p.grad is None for p in rt.net.parameters()))
            native.assert_frozen()
            (root/'stage206_policy_summary.csv').write_text('changed')
            with self.assertRaises(ValueError):run.verify(root)
            with self.assertRaises(FloatingPointError):run.nonfinite(root,dict(test=True),1)
            with self.assertRaises(ValueError):run.guard(root)


if __name__=='__main__':unittest.main()
