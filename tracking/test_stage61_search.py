"""Actual CENTER/public-decode, equal-call control, GT isolation and sealed workflow."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import stage61_search_common as c
import stage61_search_runtime as actor
import stage61_search_experiment as experiment
import stage61_search_summary as summary
try:
    import torch
    import cv2
except ImportError:
    torch=cv2=None

class MathTests(unittest.TestCase):
    def test_control_components_current_bytes_and_directional_statistics(self):
        latest=np.array([1.12345678901,4.,21.,33.]); output=np.array([7.,9.,30.,19.])
        p=c.recrop_parents(latest,output)
        np.testing.assert_array_equal(p[0],latest); np.testing.assert_array_equal(p[3],output)
        np.testing.assert_allclose(p[1,:2]+p[1,2:]/2,latest[:2]+latest[2:]/2)
        np.testing.assert_array_equal(p[1,2:],output[2:])
        np.testing.assert_allclose(p[2,:2]+p[2,2:]/2,output[:2]+output[2:]/2)
        np.testing.assert_array_equal(p[2,2:],latest[2:])
        with self.assertRaises(ValueError): c.recrop_parents(latest,[-1,2,-3,4])
        config=dict(c.DESIGN,bootstrap_samples=50)
        low=summary.contrast([-.4,.1,.2],"test","low_iou_fraction",config)
        self.assertEqual((low["worst_video_delta"],low["wins"],low["losses"]),(.2,1,2))
        auc=summary.contrast([-.4,.1,.2],"test","auc",config,True)
        self.assertEqual(auc["worst_video_delta"],-.4)
        self.assertLessEqual(auc["family_ci_low"],auc["ci_low"])

@unittest.skipUnless(torch is not None and cv2 is not None,"CPU Torch/OpenCV required; skipped tests are not validation")
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def fixture(self,folder):
        from test_stage60_state import TorchTests as PreviousFixture
        source,native,observer,records=PreviousFixture().fixture(folder)
        source["plan"]["runtime_identity"]["model_state_sha256"]=native.digest
        return source,native,observer,records

    def test_real_actor_four_calls_order_and_accepted_state_isolation(self):
        from stage27_uav_common import checked_image
        from stage48_stability_runtime import digest
        with tempfile.TemporaryDirectory(prefix="stage61-actor-",dir=Path(__file__).parent) as directory:
            old,native,observer,records=self.fixture(Path(directory)); plan=dict(old["plan"],design=dict(c.DESIGN))
            entry=plan["entries"][0]; original=digest(native.rt.net)
            for policy in c.POLICIES:
                with patch.object(observer,"capture",wraps=observer.capture) as calls:
                    trace=actor.run_actor(observer,entry["frames"],entry["initial_box"],plan["root"],policy,plan)
                expected=(1 if policy=="current" else 4)*(len(entry["frames"])-1)
                self.assertEqual(calls.call_count,expected); self.assertEqual(int(trace["head_calls"].sum()),expected)
                reference=old["arrays"][entry["sequence"]][1]["native"] if policy=="current" else None
                actor.audit(trace,entry,plan,policy,reference)
                self.assertLess(float(trace["decode_error"].max()),.001)
                changed=copy.deepcopy(entry); changed["ground_truth"][1:]=[[120.,100.,10.,10.]]*4
                actor.audit(trace,changed,plan,policy,reference)  # No evaluation GT enters the actor audit.
                changed_plan=copy.deepcopy(plan)
                for e in changed_plan["entries"]:
                    e["ground_truth"][1:]=[[120.,100.,10.,10.]]*(len(e["frames"])-1)
                replay=actor.run_actor(observer,entry["frames"],entry["initial_box"],plan["root"],policy,changed_plan)
                for key in ("boxes","candidate_parents","features","scores","selected"):
                    np.testing.assert_array_equal(trace[key],replay[key])
                if policy=="recrop":
                    np.testing.assert_array_equal(trace["candidate_parents"][:,3],trace["candidate_boxes"][:,0])
                    bad=copy.deepcopy(trace); bad["candidate_parents"][1,3,0]+=3
                    with self.assertRaises(AssertionError): actor.audit(bad,entry,plan,policy)
            image0=checked_image(plan["root"],entry["frames"][0]); image=checked_image(plan["root"],entry["frames"][1])
            parents=c.recrop_parents(entry["initial_box"],[43.,44.,21.,23.])
            results=[]
            for order in ((0,1,2,3),(3,2,1,0)):
                observer.initialize(image0,entry["initial_box"])
                results.append({i:observer.capture(image,p,1) for i in order for p in [parents[i]]})
            for i in range(4):
                for key in ("boxes","raw","size","offset","block_calls"):
                    np.testing.assert_array_equal(results[0][i][key],results[1][i][key])
            self.assertEqual(digest(native.rt.net),original)
            observer.handle.remove(); observer.pre_handle.remove()

    def test_complete_real_stage59_to61_source_reuse_resume_and_tamper(self):
        import stage59_parent_collect as e59
        import stage59_parent_summary as s59
        import stage60_state_common as c60
        import stage60_state_experiment as e60
        import stage60_state_summary as s60
        from stage48_stability_runtime import digest
        with tempfile.TemporaryDirectory(prefix="stage61-workflow-",dir=Path(__file__).parent) as directory,contextlib.redirect_stdout(io.StringIO()):
            folder=Path(directory); old,native,observer,records=self.fixture(folder)
            config=[dict(case_id="T01",sequence="v0",frame=2,purpose="fixture",anchor_method="endpoint_control_seed2132",anchor_kind="fixed_frame")]
            with patch.object(c60.prior,"load_source",return_value=old),patch.object(c60.prior,"CONFIG",config),\
                 patch.dict(c60.prior.DESIGN,horizon=2),patch.dict(c60.DESIGN,horizon=2,anchors_per_video=2,smoke_frames=3,bootstrap_samples=50),\
                 patch.dict(c.DESIGN,smoke_frames=3,bootstrap_samples=50):
                root59=folder/"stage59"; e59.prepare(old["root"],root59); p59,i59=c60.prior.load_plan(root59)
                banks={f["fold"]:e59.HeadBank(native,records,f["fold"]) for f in old["plan"]["folds"]}
                e59.execute(root59,p59,i59,observer,banks,old["plan"]["runtime_identity"],True)
                e59.execute(root59,p59,i59,observer,banks,old["plan"]["runtime_identity"]); s59.summarize(root59)
                root60=folder/"stage60"; e60.prepare(root59,root60); p60,i60=c60.load_plan(root60)
                e60.smoke(p60,root60,i60,observer,records); e60.collect_fit(p60,root60,observer,records)
                e60.fit_models(p60,root60,records); e60.evaluate(p60,root60,i60,observer,records); s60.summarize(root60)
                frozen={str(p):p.read_bytes() for p in root60.rglob("*") if p.is_file()}
                root=folder/"stage61"; experiment.prepare(root60,root); plan,source=c.load_plan(root)
                self.assertEqual((plan["new_full_trajectories"],plan["reused_full_trajectories"],plan["formal_search_calls"]),(8,10,128))
                original=digest(native.rt.net)
                gate=experiment.smoke(plan,root,source,observer,records)
                self.assertEqual(gate["extra_smoke_search_calls"],81); self.assertEqual(gate["maximum_source_box_error"],0.)
                experiment.evaluate(plan,root,source,observer,records)
                with patch.object(observer,"capture",side_effect=AssertionError("Committed GPU calls must not repeat")):
                    experiment.smoke(plan,root,source,observer,records)
                    experiment.evaluate(plan,root,source,observer,records)
                report=summary.summarize(root)
                self.assertEqual(report["formal_search_calls"],128)
                self.assertEqual(report["new_models"],0); self.assertFalse(report["training_performed"])
                self.assertEqual(report["native_deterministic_runs_per_video"],1)
                self.assertEqual(report,summary.summarize(root)); self.assertEqual(digest(native.rt.net),original)
                self.assertTrue(all(Path(p).read_bytes()==v for p,v in frozen.items()))
                # Sealed source checks reject mixed reports before any further inference.
                path=root60/"analysis/stage60_paired.csv"; saved=path.read_bytes(); path.write_bytes(saved+b"changed")
                with self.assertRaises(ValueError): c.load_plan(root)
                path.write_bytes(saved)
                path=next((root/"evaluation/traces").glob("*.npz")); saved=path.read_bytes(); path.write_bytes(saved+b"changed")
                with self.assertRaises(ValueError): experiment.completed(plan,root)
                path.write_bytes(saved)
                path=root/"analysis/stage61_policy_summary.csv"; path.write_bytes(path.read_bytes()+b"changed")
                with self.assertRaisesRegex(ValueError,"differs"): summary.summarize(root)
                observer.handle.remove(); observer.pre_handle.remove()

if __name__=="__main__": unittest.main()
