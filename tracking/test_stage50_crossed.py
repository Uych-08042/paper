"""Provenance, cell alignment and CPU instrumented shared-feature replay fixtures."""
import ast
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import stage50_crossed_common as c
import stage50_crossed_summary as summary

try:
    import torch
    import cv2
except ImportError:
    torch=cv2=None


class SourceTests(unittest.TestCase):
    def test_fixed_anchor_validation_and_exact_input_budget(self):
        with tempfile.TemporaryDirectory(prefix="stage50-plan-",dir=Path(__file__).parent) as folder:
            focus={e["sequence"]:(dict(frames=[{}]*2001),{h:{} for h in c.HEADS}) for e in c.CONFIG}
            inputs=dict(plan=dict(fingerprint="plan48"),run=dict(fingerprint="run48"),source49_fingerprint="audit49",
                focus=focus,bindings={},onsets=[dict(sequence=e["sequence"],method=e["anchor_method"],
                    **{e["anchor_kind"]:str(e["anchor_frame"])}) for e in c.CONFIG])
            with patch.object(c,"load_source",return_value=inputs):
                plan,_=c.make_plan(Path(folder))
                self.assertEqual((plan["expected_inputs"],plan["expected_head_forwards"]),(95,855))
                self.assertEqual(len({p["point_id"] for p in c.points(plan)}),95)
                self.assertEqual(len(c.points(plan,True)),15)
                inputs["onsets"][0]["onset_frame"]="553"
                with self.assertRaisesRegex(ValueError,"anchor"): c.make_plan(Path(folder))

    def test_budget_and_completed_stage49_binding(self):
        self.assertEqual(sum(len(e["frames"])*len(e["histories"]) for e in c.CONFIG),95)
        self.assertEqual(len(c.HEADS),9)
        from test_stage49_trajectory_audit import WorkflowTests
        with tempfile.TemporaryDirectory(prefix="stage50-source-",dir=Path(__file__).parent) as folder:
            root,plan,refs=WorkflowTests().fixture(Path(folder))
            with patch.object(c.fit,"load_plan",return_value=plan),patch.object(c.fit,"source_native",side_effect=lambda p,e:refs[e["sequence"]]),contextlib.redirect_stdout(io.StringIO()):
                c.prior.audit(root)
                inputs=c.load_source(root)
                self.assertEqual(len(inputs["focus"]),6)
                path=root/"analysis/trajectory_audit_v1/stage49_focus.csv"
                path.write_bytes(path.read_bytes()+b"changed")
                with self.assertRaisesRegex(ValueError,"input/output changed"): c.load_source(root)


# Minimal public tracker API with real Torch heads and actual NativeObserver hooks.
def sample_target(image,parent,search_factor,output_sz):
    crop,_,factor=c.crop_geometry(parent,image.shape[:2],search_factor,output_sz)
    return image.copy(),factor,np.zeros(image.shape[:2],bool)


def feature():
    x=torch.zeros(1,3,4,4); x[0,0,2,2]=1.; x[0,1,0,0]=.7
    return x


def tiny_head():
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.ctr=torch.nn.Conv2d(3,1,1); self.sz=torch.nn.Conv2d(3,2,1); self.off=torch.nn.Conv2d(3,2,1)
            with torch.no_grad():
                self.ctr.weight.zero_(); self.ctr.weight[0,0]=.8; self.ctr.bias.fill_(-1.)
                self.sz.weight.zero_(); self.sz.bias.fill_(float(np.log(.25/.75)))
                self.off.weight.zero_(); self.off.bias.zero_()
        def cal_bbox(self,raw,size,offset):
            index=raw.flatten(1).argmax(1); b=torch.arange(len(raw))
            xy=torch.stack([index%4,index//4],1).float()
            wh=size.flatten(2)[b,:,index]; off=offset.flatten(2)[b,:,index]
            return torch.cat([(xy+off)/4,wh],1)
        def forward(self,x):
            raw=self.ctr(x).sigmoid(); size=self.sz(x).sigmoid(); offset=self.off(x)
            return raw,self.cal_bbox(raw,size,offset),size,offset
    return Head().eval()


@unittest.skipUnless(torch is not None,"CPU Torch/OpenCV needed for shared-feature replay tests")
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def fixture(self,folder):
        from stage46_motion_runtime import NativeObserver,state_digest
        from stage48_stability_runtime import digest,head_output
        from stage27_uav_common import checked_image
        from stage28_response_common import clip_boxes
        from stage50_crossed_collect import NativeCapture,HeadBank
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.box_head=tiny_head()
                self.backbone=torch.nn.Module(); self.backbone.blocks=torch.nn.ModuleList([torch.nn.Identity() for _ in range(12)])
                self.backbone.MLP=torch.nn.Identity()
            def forward(self):
                x=feature(); self.backbone.MLP(x)
                for i in [0,1,2,3,4,5,8]: x=self.backbone.blocks[i](x)
                return head_output(self.box_head,x)
        net=Net().eval()
        clip=lambda box,h,w,margin=10:clip_boxes(np.asarray(box,float),(h,w),margin).tolist()
        path=Path(__file__).with_name("stage1_validate.py")
        fn=next(n for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n,ast.FunctionDef) and n.name=="_decode_box")
        namespace=dict(clip_box=clip)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),"exec"),namespace)
        decoder=namespace["_decode_box"]
        class Tracker:
            def __init__(self):
                self.network=net; self.params=types.SimpleNamespace(search_size=256,search_factor=4.)
                self.output_window=torch.ones(4,4); self.state=None; self.frame_id=0
            def map_box_back(self,box,factor):
                old=np.asarray(self.state); center=old[:2]+old[2:]/2
                b=np.asarray(box); xy=b[:2]+center-.5*256/factor
                return np.r_[xy-b[2:]/2,b[2:]].tolist()
            def track(self,image):
                self.frame_id+=1
                _,factor,_=sample_target(image,self.state,4.,output_sz=256)
                output=net(); self.state=decoder(self,output,factor,*image.shape[:2])
                return dict(target_bbox=self.state)
        tracker=Tracker()
        def initialize(image,box): tracker.state=list(box); tracker.frame_id=0
        rt=types.SimpleNamespace(torch=torch,net=net,tracker=tracker,search_side=4,initialize=initialize,
            s1=types.SimpleNamespace(_decode_box=decoder,clip_box=clip))
        class Observer(NativeObserver):
            def synchronize(self): pass
        native=Observer(rt); original=copy.deepcopy(net.box_head.state_dict()); records=[]
        for seed in c.fit.SEEDS:
            for index,arm in enumerate(c.fit.ARMS):
                head=tiny_head(); head.load_state_dict(original)
                with torch.no_grad(): head.sz.bias.add_(.005*(index+1)+(seed-2132)*.002)
                state=copy.deepcopy(head.state_dict())
                records.append((dict(fold=0,seed=seed,arm=arm,final_head_sha256=digest(head)),state))
        dataset=folder/"images"; dataset.mkdir(); frames=[]
        for i in range(5):
            p=dataset/("img{:07d}.png".format(i)); cv2.imwrite(str(p),np.full((160,180,3),30+i,np.uint8))
            frames.append(dict(file=p.name,height=160,width=180,sha256=c.file_hash(p)))
        e=dict(sequence="fixture",source_group_proxy="fixture",initial_box=[40.,40.,20.,20.],ground_truth=[[40.,40.,20.,20.]]*5,frames=frames)
        states={"native":original,**{r["arm"]+"_seed"+str(r["seed"]):s for r,s in records}}
        arrays={}
        for name,state in states.items():
            net.box_head.load_state_dict(state); native.digest=state_digest(net)
            native.initialize(checked_image(dataset,frames[0]),e["initial_box"]); parent=np.asarray(e["initial_box"])
            rows=[]
            for f in range(1,5):
                result=native.step(checked_image(dataset,frames[f]),parent)
                rows.append(dict(previous=parent.copy(),boxes=result["box"],peak=result["peak"],layer=result["layer"]))
                parent=result["box"].copy()
            arrays[name]={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
        net.box_head.load_state_dict(original); native.digest=state_digest(net)
        marker=folder/"frozen.marker"; marker.write_bytes(b"original provenance")
        source=dict(fingerprint="source48",root=str(dataset),runtime=dict(lens_x=16,embed_dim=3),runtime_identity=dict(fixture=True),
            search_size=256,search_factor=4.,window=np.ones((4,4)).tolist(),folds=[dict(fold=0,fit=[],held_out=["fixture"])])
        inputs=dict(plan=source,run=dict(fingerprint="run48"),source49_fingerprint="audit49",focus={"fixture":(e,arrays)},
            bindings={str(marker):c.file_hash(marker)},onsets=[dict(sequence="fixture",method="native",onset_frame="1",recovery_frame="3")])
        histories=["native","rank_seed2132","shape_seed2133"]
        config=[dict(case_id="T01",sequence="fixture",purpose="fixture",frames=[1,2],focus_frames=[2],histories=histories,
                    anchor_method="native",anchor_kind="onset_frame",anchor_frame=1),
                dict(case_id="T02",sequence="fixture",purpose="fixture_recovery",frames=[3,4],focus_frames=[3],histories=histories,
                    anchor_method="native",anchor_kind="recovery_frame",anchor_frame=3)]
        bank=HeadBank(native,records,0); capture=NativeCapture(native)
        return inputs,config,native,capture,{0:bank}

    def test_full_prepare_smoke_collect_diagonal_summary_resume_and_tamper(self):
        import stage50_crossed_collect as collect
        with tempfile.TemporaryDirectory(prefix="stage50-test-",dir=Path(__file__).parent) as name:
            folder=Path(name); inputs,config,native,capture,banks=self.fixture(folder)
            source=folder/"stage48"; source.mkdir(); output=folder/"stage50"
            old_digest=native.digest
            with patch.object(c,"CONFIG",config),patch.object(c,"load_source",return_value=inputs),contextlib.redirect_stdout(io.StringIO()):
                collect.prepare(source,output); plan,_=c.load_plan(output)
                collect.execute(output,plan,inputs,capture,banks,inputs["plan"]["runtime_identity"],True)
                smoke_hashes={str(p):p.read_bytes() for p in (output/"responses").glob("*")}
                collect.execute(output,plan,inputs,capture,banks,inputs["plan"]["runtime_identity"])
                self.assertTrue(all(Path(p).read_bytes()==b for p,b in smoke_hashes.items()))
                with patch.object(capture,"capture",side_effect=AssertionError("Committed replay must not run again")):
                    collect.execute(output,plan,inputs,capture,banks,inputs["plan"]["runtime_identity"])
                report=summary.summarize(output)
                self.assertEqual(report["shared_inputs"],12); self.assertEqual(report["head_only_forwards"],96)
                self.assertEqual(report["maximum_diagonal_box_error"],0.)
                self.assertFalse(report["continuation_performed"])
                self.assertEqual(report,summary.summarize(output)); self.assertEqual(native.digest,old_digest); native.assert_frozen()
                path=output/"analysis/stage50_focus.csv"; path.write_bytes(path.read_bytes()+b"changed")
                with self.assertRaisesRegex(ValueError,"differs"): summary.summarize(output)

    def test_cell_disagreement_gt_outside_and_diagonal_mutation(self):
        import stage50_crossed_collect as collect
        with tempfile.TemporaryDirectory(prefix="stage50-cell-",dir=Path(__file__).parent) as name:
            inputs,config,native,capture,banks=self.fixture(Path(name))
            with patch.object(c,"CONFIG",config),patch.object(c,"load_source",return_value=inputs):
                plan,_=c.make_plan(Path(name)/"source")
            point=c.points(plan)[0]; e,arrays=inputs["focus"]["fixture"]
            from stage27_uav_common import checked_image
            capture.initialize(checked_image(inputs["plan"]["root"],e["frames"][0]),e["initial_box"])
            data=collect.collect_point(capture,banks[0],checked_image(inputs["plan"]["root"],e["frames"][1]),arrays["native"]["previous"][0],1)
            c.audit_point(data,point,inputs,plan)
            altered={k:v.copy() for k,v in data.items()}; altered["boxes"][0,0]+=1
            with self.assertRaises((ValueError,AssertionError)): c.audit_point(altered,point,inputs,plan)
            # Offline test of genuinely distinct raw/Hann cells and unsupported GT cell.
            shifted=copy.deepcopy(inputs); shifted["focus"]["fixture"][0]["ground_truth"][1]=[1000.,1000.,20.,20.]
            changed={k:v.copy() for k,v in data.items()}; changed["raw"][:,0,0]=.99
            shifted["plan"]["window"][0][0]=0.
            from stage28_response_common import dense_boxes
            for h in range(len(c.HEADS)):
                dense=dense_boxes(changed["size"][h],changed["offset"][h],changed["parent"],float(changed["factor"]),256,changed["image_hw"])
                changed["raw_boxes"][h]=dense[0]
                changed["raw_normalized"][h]=np.r_[changed["offset"][h].reshape(2,-1)[:,0]/4,changed["size"][h].reshape(2,-1)[:,0]]
            rows,cells,_,_=summary.analyze(changed,point,plan,shifted)
            self.assertTrue(all(r["raw_hann_differ"] for r in rows))
            self.assertTrue(all(r["gt_cell"] is None and r["gt_cell_unclipped_log_size_error"] is None for r in rows))
            self.assertEqual(len(cells),2*len(c.HEADS))


if __name__=="__main__": unittest.main()
