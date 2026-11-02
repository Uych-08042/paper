"""Exercise the actual legacy factory contract without requiring a CUDA device."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
import stage61_runtime_replay as helper
from stage46_motion_experiment import make_runtime
import stage46_motion_runtime as native_module


class RuntimeReplayTests(unittest.TestCase):
    def fixture(self,directory):
        root=Path(directory); checkpoint=root/"original.pth"; checkpoint.write_bytes(b"original frozen checkpoint fixture")
        cfg=dict(TEST=dict(TEMPLATE_FACTOR=2.,TEMPLATE_SIZE=128,SEARCH_FACTOR=4.,SEARCH_SIZE=256),
                 MODEL=dict(BACKBONE=dict(TYPE="deit_tiny_distilled_patch16",STRIDE=16)))
        raw=dict(parameter="deit_distilled",checkpoint_sha256=helper.file_hash(checkpoint),config=copy.deepcopy(cfg),
                 torch="fixture-torch",cuda="fixture-cuda",device="fixture-device",lens_z=64,lens_x=256,embed_dim=192)
        identity=dict(tracker=raw,opencv="fixture-cv2",numpy=np.__version__,gpu_uuid="fixture-uuid",model_state_sha256="frozen-net")
        plan=dict(fingerprint="sealed-stage61-plan",runtime=copy.deepcopy(raw),runtime_identity=copy.deepcopy(identity))
        current=copy.deepcopy(cfg); current["TEST"]["SEARCH_FACTOR"]=5.
        params=SimpleNamespace(parameters=Mock(return_value=SimpleNamespace(cfg=current)))
        built=[]

        class Runtime:
            def __init__(self,parameter,gpu,path):
                self.params=params.parameters(parameter)
                self.identity=copy.deepcopy(raw); self.identity["config"]=self.params.cfg
                built.append(self)

        runtime_module=SimpleNamespace(Runtime=Runtime)
        fake_torch=SimpleNamespace(__version__="fixture-torch",version=SimpleNamespace(cuda="fixture-cuda"),manual_seed=Mock(),
            cuda=SimpleNamespace(is_available=lambda:True,manual_seed_all=Mock(),get_device_name=lambda gpu:"fixture-device",
                                 get_device_properties=lambda gpu:SimpleNamespace(uuid="fixture-uuid")),
            backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=True,deterministic=False),cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))))
        fake_cv2=SimpleNamespace(__version__="fixture-cv2",setNumThreads=Mock())
        args=SimpleNamespace(mode="smoke",parameter="deit_distilled",gpu=0,checkpoint=str(checkpoint))
        return root,plan,args,params,runtime_module,built,fake_torch,fake_cv2

    def test_frozen_config_restoration_and_real_legacy_factory(self):
        with tempfile.TemporaryDirectory(prefix="stage61-runtime-",dir=Path(__file__).parent) as directory:
            root,plan,args,params,module,built,torch,cv2=self.fixture(directory)
            frozen=copy.deepcopy(plan); original_params=params.parameters; original_runtime=module.Runtime
            with patch.dict("sys.modules",torch=torch,cv2=cv2,stage4_runtime=module),\
                 patch.object(native_module,"NativeObserver",side_effect=lambda rt:SimpleNamespace(rt=rt,digest="frozen-net")),\
                 contextlib.redirect_stdout(io.StringIO()):
                native,identity=helper.checked_runtime(args,plan,root,make_runtime,params,module,SimpleNamespace)
            self.assertEqual(identity,plan["runtime_identity"])
            self.assertEqual(built[0].params.search_factor,4.)
            self.assertEqual(built[0].params.cfg.TEST.SEARCH_FACTOR,4.)
            self.assertEqual(built[0].params.checkpoint,str(Path(args.checkpoint).resolve()))
            self.assertEqual(plan,frozen); self.assertIs(params.parameters,original_params); self.assertIs(module.Runtime,original_runtime)
            diagnostic=json.loads((root/"diagnostics/stage61_runtime_smoke.json").read_text())
            self.assertEqual(diagnostic["status"],"EXACT_SOURCE_RUNTIME_VERIFIED")
            self.assertEqual(diagnostic["current_yaml_differences"],[dict(field="config.TEST.SEARCH_FACTOR",expected=4.,actual=5.)])
            self.assertTrue(diagnostic["frozen_config_restored"])

    def test_checkpoint_and_software_mismatch_rejected_before_model_construction(self):
        for mismatch in ("checkpoint","torch"):
            with self.subTest(mismatch=mismatch),tempfile.TemporaryDirectory(prefix="stage61-runtime-",dir=Path(__file__).parent) as directory:
                root,plan,args,params,module,built,torch,cv2=self.fixture(directory)
                if mismatch=="checkpoint": Path(args.checkpoint).write_bytes(b"a different checkpoint")
                else: torch.__version__="different-torch"
                with patch.dict("sys.modules",torch=torch,cv2=cv2,stage4_runtime=module):
                    with self.assertRaisesRegex(ValueError,"checkpoint_sha256" if mismatch=="checkpoint" else "tracker.torch"):
                        helper.checked_runtime(args,plan,root,make_runtime,params,module,SimpleNamespace)
                self.assertEqual(built,[])
                diagnostic=json.loads((root/"diagnostics/stage61_runtime_smoke.json").read_text())
                self.assertEqual(diagnostic["status"],"RUNTIME_MISMATCH")

    def test_model_hash_and_gpu_identity_still_rejected(self):
        for mismatch in ("model","uuid"):
            with self.subTest(mismatch=mismatch),tempfile.TemporaryDirectory(prefix="stage61-runtime-",dir=Path(__file__).parent) as directory:
                root,plan,args,params,module,built,torch,cv2=self.fixture(directory)
                if mismatch=="uuid": torch.cuda.get_device_properties=lambda gpu:SimpleNamespace(uuid="other-uuid")
                digest="changed-net" if mismatch=="model" else "frozen-net"
                with patch.dict("sys.modules",torch=torch,cv2=cv2,stage4_runtime=module),\
                     patch.object(native_module,"NativeObserver",side_effect=lambda rt:SimpleNamespace(rt=rt,digest=digest)),\
                     contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(ValueError,"model_state_sha256" if mismatch=="model" else "gpu_uuid"):
                        helper.checked_runtime(args,plan,root,make_runtime,params,module,SimpleNamespace)

    def test_runtime_dimensions_and_sealed_identity_binding_are_not_relaxed(self):
        with tempfile.TemporaryDirectory(prefix="stage61-runtime-",dir=Path(__file__).parent) as directory:
            root,plan,args,params,module,built,torch,cv2=self.fixture(directory)
            real=module.Runtime
            def different_shape(*values,**keywords):
                rt=real(*values,**keywords); rt.identity["lens_x"]=196; return rt
            module.Runtime=different_shape
            with patch.dict("sys.modules",torch=torch,cv2=cv2,stage4_runtime=module),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,"tracker.lens_x"):
                    helper.checked_runtime(args,plan,root,make_runtime,params,module,SimpleNamespace)
            plan["runtime_identity"]["tracker"]["config"]["TEST"]["SEARCH_SIZE"]=384
            with patch.dict("sys.modules",torch=torch,cv2=cv2,stage4_runtime=module):
                with self.assertRaisesRegex(ValueError,"sealed.runtime.config.TEST.SEARCH_SIZE"):
                    helper.checked_runtime(args,plan,root,make_runtime,params,module,SimpleNamespace)


if __name__=="__main__": unittest.main()
