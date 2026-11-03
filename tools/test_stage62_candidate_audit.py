"""Candidate-label semantics, explicit support, geometry and a sealed real-CPU workflow."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import stage62_candidate_common as c
import stage62_candidate_audit as audit
from stage27_uav_common import crop_geometry, scene_geometry
from stage4_common import write_json, fingerprint


def fixture_arrays():
    good=np.array([40.,40.,20.,20.]); low=np.array([110.,110.,20.,20.]); middle=np.array([50.,40.,20.,20.])
    boxes=np.array([[good,low,low,low],[low,good,low,low],[low,low,low,low],
                    [middle,good,good,low],[good,good,low,low],[middle,middle,middle,middle]])
    selected=np.array([1,1,0,0,0,0]); n=len(boxes)
    parents=np.tile(good,(n,4,1)); hw=np.tile([200,200],(n,1))
    crops=np.array([[crop_geometry(p,hw[i],4.,256)[0] for p in parents[i]] for i in range(n)])
    features=np.zeros((n,4,len(c.prior.original.prior.FEATURES)))
    features[:,:,1]=.1; features[np.arange(n),selected,1]=.9; features[:,:,0]=.8
    data=dict(frame_id=np.arange(1,n+1),candidate_boxes=boxes,selected=selected,boxes=boxes[np.arange(n),selected].copy(),
        candidate_parents=parents,crop=crops,image_hw=hw,features=features,layer=np.full((n,4),9))
    return data,dict(ground_truth=[good.tolist()]*(n+1))


class CandidateTests(unittest.TestCase):
    def test_decoded_availability_selection_damage_rescue_and_official_iou(self):
        data,entry=fixture_arrays(); saved=copy.deepcopy(data)
        metrics,_=c.frame_metrics(data,entry); result=c.summarize_metrics(metrics,np.ones(6,bool))
        self.assertEqual(result["has_good_decoded_count"],4)
        self.assertEqual(result["no_good_decoded_count"],2)
        self.assertEqual(result["available_selected_miss_count"],2)
        self.assertEqual(result["miss_given_good_decoded"],.5)
        self.assertEqual(result["q0_good_to_low_count"],1)
        self.assertEqual(result["loss_given_good_q0"],.5)
        self.assertEqual(result["rescue_opportunity_count"],2)
        self.assertEqual(result["rescue_given_opportunity"],.5)
        self.assertEqual(result["all_decoded_low_count"],1)
        self.assertTrue(np.all(metrics["decoded_regret"]>=0))
        # Offline GT changes labels, not frozen outputs, incoming parents, scores or actions.
        changed=copy.deepcopy(entry); changed["ground_truth"][1:]=[[110.,110.,20.,20.]]*6
        different,_=c.frame_metrics(data,changed)
        self.assertFalse(np.array_equal(metrics["q0_iou"],different["q0_iou"]))
        for key in data: np.testing.assert_array_equal(data[key],saved[key])
        fractional=copy.deepcopy(data); fractional["candidate_boxes"][0,0]=[40.9,40.9,20.9,20.9]
        labels,_=c.frame_metrics(fractional,entry)
        self.assertEqual(labels["q0_iou"][0],1.)  # Predictions truncate exactly as the established evaluator.

    def test_geometry_matches_frozen_usable_crop_including_padding(self):
        data,entry=fixture_arrays()
        parents=np.array([[40.,40.,20.,20.],[-10.,-10.,20.,20.],[190.,195.,10.,10.],[0.,0.,2.,3.]])
        data["candidate_parents"]=np.tile(parents,(6,1,1))
        for i in range(6):
            for q,p in enumerate(parents): data["crop"][i,q]=crop_geometry(p,data["image_hw"][i],4.,256)[0]
        _,detail=c.frame_metrics(data,entry)
        for i in range(6):
            for q,p in enumerate(parents):
                reference=scene_geometry(p,entry["ground_truth"][i+1],data["image_hw"][i],4.,256)
                self.assertAlmostEqual(detail["coverage"][i,q],reference["coverage"])
                self.assertEqual(detail["center_inside"][i,q],reference["center_in_usable"])

    def test_empty_denominators_remain_undefined_and_video_weights_are_equal(self):
        data,entry=fixture_arrays(); metrics,_=c.frame_metrics(data,entry)
        empty=c.summarize_metrics(metrics,np.zeros(6,bool))
        self.assertIsNone(empty["miss_given_good_decoded"])
        self.assertIsNone(empty["has_good_decoded_fraction"])
        miss=c.summarize_metrics(metrics,np.array([1,0,0,0,0,0],bool))
        success=c.summarize_metrics(metrics,np.array([0,1,0,0,0,0],bool)); success["frame_states"]=100
        rows=[dict(sequence="a",arm="native_history",cohort="all",**miss),
              dict(sequence="a",arm="native_history",cohort="all",**empty),
              dict(sequence="b",arm="native_history",cohort="all",**success)]
        videos=c.aggregate(rows,("sequence","arm","cohort"),"seeds")
        self.assertEqual(videos[0]["miss_given_good_decoded_supported_seeds"],1)
        policies=c.aggregate(videos,("arm","cohort"),"videos")
        self.assertEqual(policies[0]["miss_given_good_decoded"],.5)
        self.assertEqual(policies[0]["miss_given_good_decoded_supported_videos"],2)
        json.dumps(policies,allow_nan=False)
        masks,events=c.cohort_masks(metrics,np.ones(6))
        self.assertFalse(events); self.assertFalse(masks["pre_first_failure"].any())


try:
    import torch
    import cv2
except ImportError:
    torch=cv2=None


@unittest.skipUnless(torch is not None and cv2 is not None,"CPU Torch/OpenCV needed to generate a genuine small public-path source fixture")
class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_complete_cache_only_source_chain_immutability_repeat_and_tamper(self):
        from test_stage61_search import TorchTests as Fixture
        import stage59_parent_collect as e59
        import stage59_parent_summary as s59
        import stage60_state_common as c60
        import stage60_state_experiment as e60
        import stage60_state_summary as s60
        import stage61_search_common as c61
        import stage61_search_experiment as e61
        import stage61_l40_common as l40
        import stage61_l40_experiment as e40
        import stage61_l40_summary as s40
        with tempfile.TemporaryDirectory(prefix="stage62-cache-",dir=Path(__file__).parent) as directory,contextlib.redirect_stdout(io.StringIO()):
            folder=Path(directory); old,native,observer,records=Fixture().fixture(folder)
            # A failure from the first tracked frame in v0, while v1 retains reliable support.
            old["plan"]["entries"][0]["ground_truth"][1:]=[[1.,1.,1.,1.]]*4
            raw=old["plan"]["runtime"]; raw.update(device="NVIDIA L20",torch=str(torch.__version__),cuda=str(torch.version.cuda))
            old["plan"]["runtime_identity"]=dict(tracker=copy.deepcopy(raw),gpu_uuid="source-l20",model_state_sha256=native.digest,
                opencv=cv2.__version__,numpy=np.__version__)
            config=[dict(case_id="T01",sequence="v0",frame=2,purpose="fixture",anchor_method="endpoint_control_seed2132",anchor_kind="fixed_frame")]
            cases=(c.case("T01","v0","native",0,"history",1,"onset","fixture"),)
            with patch.object(c60.prior,"load_source",return_value=old),patch.object(c60.prior,"CONFIG",config),\
                 patch.dict(c60.prior.DESIGN,horizon=2),patch.dict(c60.DESIGN,horizon=2,anchors_per_video=2,smoke_frames=3,bootstrap_samples=30),\
                 patch.dict(c61.DESIGN,smoke_frames=3,bootstrap_samples=30),patch.object(c,"CASES",cases):
                root59=folder/"stage59"; e59.prepare(old["root"],root59); p59,i59=c60.prior.load_plan(root59)
                banks={f["fold"]:e59.HeadBank(native,records,f["fold"]) for f in old["plan"]["folds"]}
                e59.execute(root59,p59,i59,observer,banks,old["plan"]["runtime_identity"],True)
                e59.execute(root59,p59,i59,observer,banks,old["plan"]["runtime_identity"]); s59.summarize(root59)
                root60=folder/"stage60"; e60.prepare(root59,root60); p60,i60=c60.load_plan(root60)
                e60.smoke(p60,root60,i60,observer,records); e60.collect_fit(p60,root60,observer,records)
                e60.fit_models(p60,root60,records); e60.evaluate(p60,root60,i60,observer,records); s60.summarize(root60)
                root61=folder/"stage61"; e61.prepare(root60,root61)
                root=folder/"l40"; e40.prepare(root61,root,dict(device="NVIDIA L40",gpu_uuid="fixture-l40")); plan,source=l40.load_plan(root)
                e40.smoke(plan,root,source,observer,records); e40.evaluate(plan,root,source,observer,records); s40.summarize(root)
                for mode in ("smoke","evaluate"):
                    write_json(root/"diagnostics"/("stage61_runtime_"+mode+".json"),dict(schema="stage61_runtime_replay_diagnostic_v1",
                        mode=mode,status="EXACT_SOURCE_RUNTIME_VERIFIED",plan_fingerprint=plan["fingerprint"],mismatches=[],
                        verified_runtime_fingerprint=fingerprint(plan["runtime_identity"]),launcher_sha256=plan["code_sha256"]["stage61_runtime_replay.py"]))
                frozen={str(p):p.read_bytes() for d in (root59,root60,root61,root) for p in d.rglob("*") if p.is_file()}
                original_open=Path.open
                def no_images(path,*values,**keywords):
                    if path.suffix.lower() in (".png",".jpg",".jpeg"): raise AssertionError("Audit cannot read images")
                    return original_open(path,*values,**keywords)
                output=folder/"audit"
                with patch.dict("sys.modules",torch=None,cv2=None),patch.object(Path,"open",no_images),\
                     patch.object(audit.source_actor,"run_actor",side_effect=AssertionError("No inference")),\
                     patch.object(native,"step",side_effect=AssertionError("No public forward")),\
                     patch.object(observer,"capture",side_effect=AssertionError("No candidate forward")):
                    report=audit.audit(root,output)
                    self.assertEqual(report,audit.audit(root,output))
                self.assertEqual(report["source_trajectories"],18)
                self.assertEqual(report["cached_frame_states"],72)
                self.assertEqual(report["cached_decoded_candidates"],216)
                self.assertEqual(report["new_network_calls"],0)
                self.assertFalse(report["training_performed"])
                self.assertEqual(report["focus_rows"],3)
                self.assertEqual(report["focus_timeline_rows"],12)
                self.assertEqual(report["focus_candidate_rows"],36)
                self.assertEqual(len(report["table_sha256"]),8)
                self.assertTrue(all(Path(p).read_bytes()==value for p,value in frozen.items()))
                for path in (root/"analysis/stage61_l40_paired.csv",next((root/"evaluation/traces").glob("*.npz"))):
                    saved=path.read_bytes(); path.write_bytes(saved+b"tamper")
                    with self.assertRaises(ValueError): audit.audit(root,output)
                    path.write_bytes(saved)
                path=output/"analysis/stage62_policy_summary.csv"; path.write_bytes(path.read_bytes()+b"tamper")
                with self.assertRaisesRegex(ValueError,"differs"): audit.audit(root,output)
                observer.handle.remove(); observer.pre_handle.remove()


if __name__=="__main__": unittest.main()
