"""Geometry, real OpenCV motion, native-call audit and resumable four-arm pipeline."""
import ast
import contextlib
import copy
import inspect
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import stage46_motion_common as c
from stage46_motion_runtime import run_video

try:
    import cv2
except ImportError:
    cv2 = None
try:
    import torch
except ImportError:
    torch = None


class StateTests(unittest.TestCase):
    def test_camera_coordinates_and_relative_velocity(self):
        state = c.MotionState([40,40,20,20])
        m = np.array([[1.,0.,4.],[0.,1.,0.]])
        state.advance([44,40,20,20],m,True)
        np.testing.assert_array_equal(state.relative,[0,0])
        accelerated = np.array([[1.,0.,8.],[0.,1.,0.]])
        camera = state.propose("camera_velocity",accelerated,True,(150,150))
        velocity = state.propose("velocity",accelerated,True,(150,150))
        np.testing.assert_array_equal(camera["shift"],[8,0])
        np.testing.assert_array_equal(velocity["shift"],[2,0])
        # Residual velocity is expressed in the previous image and must rotate.
        state.relative = np.array([2.,0.])
        rotation = np.array([[0.,-1.,100.],[1.,0.,0.]])
        parent = state.propose("camera_velocity",rotation,True,(150,150))
        np.testing.assert_allclose(parent["proposal"],c.warp(rotation,c.center(state.previous))+[0,2])

    def test_native_cap_and_failed_fit_fallback(self):
        state = c.MotionState([40,40,20,20])
        state.raw = np.array([60.,80.])
        state.relative = np.array([9.,9.])
        native = state.propose("sgla",c.IDENTITY,False,(100,100))
        np.testing.assert_array_equal(native["parent"],state.previous)
        fallback = state.propose("camera_velocity",c.IDENTITY,False,(100,100))
        control = state.propose("velocity",c.IDENTITY,False,(100,100))
        np.testing.assert_array_equal(fallback["parent"],control["parent"])
        self.assertAlmostEqual(np.linalg.norm(fallback["shift"]),10.)
        self.assertTrue(fallback["cap_hit"])
        state.advance([42,40,20,20],c.IDENTITY,False)
        np.testing.assert_array_equal(state.relative,[0,0])
        with self.assertRaisesRegex(ValueError,"initial box"):
            c.MotionState([0,0,-1,5])
        self.assertNotIn("gt",inspect.signature(run_video).parameters)
        self.assertNotIn("labels",inspect.signature(run_video).parameters)


@unittest.skipUnless(cv2 is not None,"OpenCV required for image motion tests")
class FlowTests(unittest.TestCase):
    def test_real_lk_affine_with_rounded_resize_and_own_target_mask(self):
        rng = np.random.default_rng(17)
        gray = rng.integers(0,256,size=(721,1083),dtype=np.uint8)
        gray = cv2.GaussianBlur(gray,(3,3),0)
        image = np.repeat(gray[:,:,None],3,axis=2)
        angle = np.deg2rad(1.)
        known = np.array([[np.cos(angle),-np.sin(angle),9.],[np.sin(angle),np.cos(angle),-4.]])
        shifted = cv2.warpAffine(image,known,(image.shape[1],image.shape[0]))
        a,b = c.gray_image(image),c.gray_image(shifted)
        result = c.estimate_camera(a,b,[440,300,40,40],image.shape[:2],2127)
        self.assertTrue(result["valid"],str(result))
        for point in ([100,100],[500,400],[850,550]):
            np.testing.assert_allclose(c.warp(result["matrix"],point),c.warp(known,point),atol=1.,rtol=0)
        mask = c.background_mask((100,100),[40,40,20,20],np.ones(2))
        self.assertEqual(mask[50,50],0)
        self.assertEqual(mask[5,5],255)
        empty = c.background_mask((100,100),[-100,-100,10,10],np.ones(2))
        self.assertTrue((empty == 255).all())

    def test_textureless_and_size_change_reject_to_identity(self):
        blank = c.gray_image(np.zeros((160,160,3),np.uint8))
        value = c.estimate_camera(blank,blank,[30,30,20,20],(160,160),1)
        self.assertFalse(value["valid"])
        np.testing.assert_array_equal(value["matrix"],c.IDENTITY)
        other = c.gray_image(np.zeros((160,180,3),np.uint8))
        self.assertEqual(c.estimate_camera(blank,other,[30,30,20,20],(160,180),1)["reason"],1)


class FixtureObserver:
    def __init__(self):
        self.calls = 0
    def initialize(self,image,box):
        self.frame = 0
    def synchronize(self):
        pass
    def step(self,image,parent):
        self.frame += 1
        self.calls += 1
        blocks = np.zeros(12,int)
        blocks[:6],blocks[8] = 1,1
        return dict(box=np.array([40.+self.frame,40.,20.,20.]),layer=9,peak=.8,
                    block_calls=blocks,head_calls=1,selector_calls=1,network_ms=1.)
    def assert_frozen(self):
        pass


@unittest.skipUnless(cv2 is not None,"OpenCV required for pipeline fixtures")
class PipelineTests(unittest.TestCase):
    def fixture(self,root):
        frames = []
        for index in range(5):
            image = np.zeros((128,128,3),np.uint8)
            cv2.rectangle(image,(40+index,40),(60+index,60),(255,255,255),-1)
            path = root/(str(index)+".png")
            cv2.imwrite(str(path),image)
            frames.append(dict(file=path.name,sha256=c.file_hash(path),height=128,width=128))
        return frames

    def test_rollout_audit_call_budget_receipt_and_tamper(self):
        with tempfile.TemporaryDirectory(prefix="stage46-test-",dir=Path(__file__).parent) as folder:
            root = Path(folder)
            frames = self.fixture(root)
            def estimator(*args):
                return dict(matrix=c.IDENTITY.copy(),valid=False,stats=np.zeros(6),reason=2)
            def accepted(*args):
                return dict(matrix=np.array([[1.,0.,2.],[0.,1.,0.]]),valid=True,
                            stats=np.array([100,80,70,.875,.2,12]),reason=0)
            active = run_video(FixtureObserver(),root,frames,[40,40,20,20],"fixture","camera_velocity",estimator=accepted)
            c.audit_video(active,frames,[40,40,20,20],"camera_velocity")
            active["motion_stats"][0] = 0
            with self.assertRaisesRegex(ValueError,"quality support"):
                c.audit_video(active,frames,[40,40,20,20],"camera_velocity")
            for method in c.METHODS:
                data = run_video(FixtureObserver(),root,frames,[40,40,20,20],"fixture",method,estimator=estimator)
                c.audit_video(data,frames,[40,40,20,20],method)
                wrong = copy.deepcopy(data)
                wrong["parents"][1,0] += 1
                with self.assertRaises(AssertionError):
                    c.audit_video(wrong,frames,[40,40,20,20],method)
                wrong = copy.deepcopy(data)
                wrong["head_calls"][1] = 2
                with self.assertRaises(AssertionError):
                    c.audit_video(wrong,frames,[40,40,20,20],method)
                path = root/(method+".npz")
                meta = dict(method=method)
                c.save_result(path,data,meta)
                restored = c.load_result(path,meta)
                np.testing.assert_array_equal(restored["boxes"],data["boxes"])
                # NPZ still loads after a suffix is appended; the raw-byte receipt must reject it.
                with path.open("ab") as f:
                    f.write(b"changed")
                with self.assertRaisesRegex(ValueError,"trace changed"):
                    c.load_result(path,meta)

    def test_full_prepare_smoke_resume_four_arm_summary_and_source_freeze(self):
        import stage46_motion_experiment as experiment
        import stage46_motion_runtime as runtime
        import stage46_motion_summary as summary
        with tempfile.TemporaryDirectory(prefix="stage46-test-",dir=Path(__file__).parent) as folder:
            root = Path(folder)
            frames = self.fixture(root)
            source = root/"source"
            source.mkdir()
            c.write_json(source/"run.json",dict(fixture=True))
            traces = {}
            for index in range(12):
                name = "uav{:07d}_00000_s".format(index)
                boxes = np.array([[40.+i,40,20,20] for i in range(1,5)])
                entry = dict(sequence=name,source_group_proxy="uav{:07d}".format(index),frames=frames,
                    initial_box=[40,40,20,20],ground_truth=[[40,40,20,20],*boxes.tolist()])
                traces[name] = entry,dict(boxes=boxes,raw_score_peak=np.full(4,.8),layer=np.full(4,9,int))
            original = dict(fingerprint="source",manifest=dict(root=str(root)),search_factor=4.,search_size=256)
            inputs = (source,original,traces,{"fixture":"sha"})
            output = root/"out"
            with patch.object(c,"load_source",return_value=inputs),patch.dict(c.DESIGN,smoke_frames=4):
                with contextlib.redirect_stdout(io.StringIO()):
                    experiment.prepare(source,output)
                    plan,_ = c.load_plan(output)
                    observer = FixtureObserver()
                    identity = dict(fixture=True)
                    experiment.execute(plan,inputs,output,observer,identity,smoke=True)
                    experiment.execute(plan,inputs,output,observer,identity)
                    calls = observer.calls
                    experiment.execute(plan,inputs,output,observer,identity)
                    self.assertEqual(observer.calls,calls)
                    summary.summarize(output)
                decision = c.load_sealed(output/"analysis/stage46_decision.json")
                self.assertEqual(decision["full_trajectories"],48)
                self.assertEqual(decision["max_native_box_error"],0.)
                self.assertFalse(decision["external_generalization_established"])
                self.assertEqual(decision["primary"]["mean_delta"],0.)
                self.assertEqual(decision["status"],"LIMITED_BACKGROUND_MOTION_SUPPORT")
                path = c.cache_path(output,plan["sequences"][0],"velocity")
                path.unlink()
                with self.assertRaisesRegex(ValueError,"trace changed"):
                    with contextlib.redirect_stdout(io.StringIO()):
                        summary.summarize(output)
                c.write_json(source/"run.json",dict(fixture=False))
                with self.assertRaisesRegex(ValueError,"source/code/design changed"):
                    c.load_plan(output)


@unittest.skipUnless(torch is not None,"Torch needed for live native-forward call hooks")
class NativeCallTests(unittest.TestCase):
    def test_original_sgla_backbone_forward_uses_exactly_six_plus_one(self):
        from stage46_motion_runtime import NativeObserver
        from torch import nn
        path = Path(__file__).resolve().parents[1]/"lib/models/sglatrack/base_backbone.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        backbone = next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name == "BaseBackbone")
        method = next(x for x in backbone.body if isinstance(x,ast.FunctionDef) and x.name == "forward_test")
        namespace = dict(torch=torch,start_layer=5,enabled_layer_num=1,
            combine_tokens=lambda z,x,mode:torch.cat([z,x],1),recover_tokens=lambda x,z,n,mode:x)
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),"exec"),namespace)
        class Selector(nn.Module):
            def forward(self,x):
                return torch.tensor([[0.,0.,1.,0.,0.,0.]])
        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = nn.Identity()
                self.pos_embed_z = nn.Parameter(torch.zeros(1,1,3),requires_grad=False)
                self.pos_embed_x = nn.Parameter(torch.zeros(1,1,3),requires_grad=False)
                self.pos_drop = self.norm = nn.Identity()
                self.blocks = nn.ModuleList([nn.Identity() for _ in range(12)])
                self.MLP,self.cat_mode = Selector(),"direct"
            forward = namespace["forward_test"]
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone,self.box_head = Backbone(),nn.Identity()
            def forward(self):
                tokens,_ = self.backbone(torch.ones(1,1,3),torch.ones(1,1,3))
                return dict(score_map=self.box_head(tokens))
        class Tracker:
            def __init__(self,net):
                self.net,self.state = net,None
            def track(self,image):
                self.net()
                return dict(target_bbox=self.state)
        net = Net().eval()
        rt = types.SimpleNamespace(net=net,tracker=Tracker(net),torch=torch)
        observer = NativeObserver(rt)
        observer.synchronize = lambda:None
        result = observer.step(np.zeros((8,8,3)),[1,1,4,4])
        self.assertEqual(result["layer"],9)
        self.assertEqual(result["block_calls"].sum(),7)
        self.assertEqual(result["head_calls"],1)
        observer.assert_frozen()


class SummaryTests(unittest.TestCase):
    def test_low_iou_win_direction(self):
        from stage46_motion_summary import paired
        draws = np.array([[0,1],[1,0]])
        result = paired([-.1,-.2],"effect","low_iou_fraction",draws)
        self.assertEqual((result["wins"],result["losses"],result["favorable_direction"]),(2,0,"negative"))


if __name__ == "__main__":
    unittest.main()
