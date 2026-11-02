"""CPU source-chain and real public-path tests for the separate L40 comparison."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import stage61_l40_common as c
import stage61_l40_experiment as experiment
import stage61_l40_summary as summary
import stage61_search_common as source61
try:
    import torch
    import cv2
except ImportError:
    torch=cv2=None


class DeviceTests(unittest.TestCase):
    def test_only_hardware_can_change_and_l40_is_required(self):
        plan=dict(runtime=dict(torch="source-torch",cuda="source-cuda"))
        fake=SimpleNamespace(__version__="source-torch",version=SimpleNamespace(cuda="source-cuda"),
            cuda=SimpleNamespace(is_available=lambda:True,get_device_name=lambda gpu:"NVIDIA L40",
                                get_device_properties=lambda gpu:SimpleNamespace(uuid="l40-uuid")))
        self.assertEqual(c.device_binding(plan,0,fake),dict(device="NVIDIA L40",gpu_uuid="l40-uuid"))
        fake.cuda.get_device_name=lambda gpu:"NVIDIA L20"
        with self.assertRaisesRegex(ValueError,"requires NVIDIA L40"): c.device_binding(plan,0,fake)
        fake.cuda.get_device_name=lambda gpu:"NVIDIA L40"; fake.__version__="other-torch"
        with self.assertRaisesRegex(ValueError,"software differs"): c.device_binding(plan,0,fake)

    def test_old_device_receipts_cannot_fill_a_new_l40_cache(self):
        with tempfile.TemporaryDirectory(prefix="stage61-l40-cache-",dir=Path(__file__).parent) as directory:
            path=Path(directory)/"trace.npz"; metadata=dict(kind="old_L20_control")
            source61.save(path,dict(boxes=np.ones((2,4))),metadata)
            with self.assertRaisesRegex(ValueError,"L40 sealed content changed"): c.read_cache(path,metadata)


@unittest.skipUnless(torch is not None and cv2 is not None,"CPU Torch/OpenCV required; skipped tests do not validate the workflow")
class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_complete_six_arm_runtime_local_baselines_source_freeze_resume_and_tamper(self):
        from test_stage61_search import TorchTests as Fixture
        import stage59_parent_collect as e59
        import stage59_parent_summary as s59
        import stage60_state_common as c60
        import stage60_state_experiment as e60
        import stage60_state_summary as s60
        import stage61_search_experiment as e61
        from stage48_stability_runtime import digest
        with tempfile.TemporaryDirectory(prefix="stage61-l40-",dir=Path(__file__).parent) as directory,contextlib.redirect_stdout(io.StringIO()):
            folder=Path(directory); old,native,observer,records=Fixture().fixture(folder)
            raw=old["plan"]["runtime"]
            raw.update(device="NVIDIA L20",torch=str(torch.__version__),cuda=str(torch.version.cuda))
            old["plan"]["runtime_identity"]=dict(tracker=copy.deepcopy(raw),gpu_uuid="original-l20-uuid",
                model_state_sha256=native.digest,opencv=cv2.__version__,numpy=np.__version__)
            config=[dict(case_id="T01",sequence="v0",frame=2,purpose="fixture",anchor_method="endpoint_control_seed2132",anchor_kind="fixed_frame")]
            with patch.object(c60.prior,"load_source",return_value=old),patch.object(c60.prior,"CONFIG",config),\
                 patch.dict(c60.prior.DESIGN,horizon=2),patch.dict(c60.DESIGN,horizon=2,anchors_per_video=2,smoke_frames=3,bootstrap_samples=50),\
                 patch.dict(source61.DESIGN,smoke_frames=3,bootstrap_samples=50):
                root59=folder/"stage59"; e59.prepare(old["root"],root59); p59,i59=c60.prior.load_plan(root59)
                banks={f["fold"]:e59.HeadBank(native,records,f["fold"]) for f in old["plan"]["folds"]}
                e59.execute(root59,p59,i59,observer,banks,old["plan"]["runtime_identity"],True)
                e59.execute(root59,p59,i59,observer,banks,old["plan"]["runtime_identity"]); s59.summarize(root59)
                root60=folder/"stage60"; e60.prepare(root59,root60); p60,i60=c60.load_plan(root60)
                e60.smoke(p60,root60,i60,observer,records); e60.collect_fit(p60,root60,observer,records)
                e60.fit_models(p60,root60,records); e60.evaluate(p60,root60,i60,observer,records); s60.summarize(root60)
                root61=folder/"stage61"; e61.prepare(root60,root61)
                frozen={str(p):p.read_bytes() for d in (root59,root60,root61) for p in d.rglob("*") if p.is_file()}
                root=folder/"l40"; experiment.prepare(root61,root,dict(device="NVIDIA L40",gpu_uuid="new-l40-uuid"))
                plan,source=c.load_plan(root)
                self.assertEqual((plan["new_full_trajectories"],plan["reused_full_trajectories"],plan["formal_search_calls"]),(18,0,216))
                self.assertEqual(plan["smoke_search_calls"],90)
                self.assertEqual(plan["runtime"]["device"],"NVIDIA L40")
                self.assertEqual(plan["source_runtime_identity"]["tracker"]["device"],"NVIDIA L20")
                self.assertEqual(plan["source_runtime_identity"]["model_state_sha256"],plan["runtime_identity"]["model_state_sha256"])
                self.assertEqual({r["field"] for r in plan["declared_runtime_differences"]},{"runtime.tracker.device","runtime.gpu_uuid"})
                original=digest(native.rt.net)
                gate=experiment.smoke(plan,root,source,observer,records)
                self.assertEqual(gate["maximum_public_control_box_error"],0)
                experiment.evaluate(plan,root,source,observer,records)
                with patch.object(observer,"capture",side_effect=AssertionError("Committed inference must not repeat")),\
                     patch.object(native,"step",side_effect=AssertionError("Committed public prefix must not repeat")):
                    experiment.smoke(plan,root,source,observer,records)
                    experiment.evaluate(plan,root,source,observer,records)
                report=summary.summarize(root)
                self.assertEqual(report,summary.summarize(root))
                self.assertEqual(report["formal_search_calls"],216)
                self.assertEqual(report["reused_full_trajectories"],0)
                self.assertFalse(report["training_performed"])
                self.assertEqual(report["comparison_runtime"]["tracker"]["device"],"NVIDIA L40")
                # Poison only old-device comparison arrays: main effects/cohorts must remain local to L40.
                shifted=copy.deepcopy(source)
                for trace in shifted["reused"].values(): trace["boxes"]=trace["boxes"]+50
                with patch.object(c,"load_plan",return_value=(plan,shifted)),patch.object(summary,"publish_table") as tables,\
                     patch.object(c,"publish_identical"):
                    shifted_report=summary.summarize(root)
                for field in ("primary","overall","retention_guard","pilot_pass","origin_by_proposal_interaction"):
                    self.assertEqual(shifted_report[field],report[field])
                bridge=next(call.args[1] for call in tables.call_args_list if call.args[0].name=="stage61_l40_device_bridge.csv")
                self.assertTrue(all(r["changed_box_frames"]>0 for r in bridge))
                import csv
                with (root/"analysis/stage61_l40_policy_summary.csv").open(newline="") as handle: policies=list(csv.DictReader(handle))
                self.assertEqual(len(policies),6)
                self.assertEqual(sum(int(r["newly_executed_head_calls"]) for r in policies),216)
                self.assertTrue(all(int(r["newly_executed_head_calls"])==int(r["logical_head_calls"]) for r in policies))
                self.assertEqual(digest(native.rt.net),original)
                self.assertTrue(all(Path(p).read_bytes()==v for p,v in frozen.items()))
                path=next((root/"evaluation/traces").glob("*.npz")); saved=path.read_bytes(); path.write_bytes(saved+b"changed")
                with self.assertRaisesRegex(ValueError,"changed"): experiment.completed(plan,root)
                path.write_bytes(saved)
                path=root61/"manifest.json"; saved=path.read_bytes(); path.write_bytes(saved+b"changed")
                with self.assertRaises(ValueError): c.load_plan(root)
                path.write_bytes(saved)
                path=root/"analysis/stage61_l40_policy_summary.csv"; path.write_bytes(path.read_bytes()+b"changed")
                with self.assertRaisesRegex(ValueError,"differs"): summary.summarize(root)
                observer.handle.remove(); observer.pre_handle.remove()


if __name__=="__main__": unittest.main()
