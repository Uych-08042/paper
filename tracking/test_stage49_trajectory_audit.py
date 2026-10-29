"""Pure NumPy tests of geometry, temporal boundaries and frozen Stage-48 audit IO."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage49_trajectory_audit as a


def entry(name,n=65):
    return dict(sequence=name,source_group_proxy=name.split("_")[0],initial_box=[80.,60.,20.,20.],
        ground_truth=[[80.,60.,20.,20.]]*(n+1),
        frames=[dict(file="unopened/img{:07d}.jpg".format(i),height=300,width=400) for i in range(n+1)])


def trace(e,boxes):
    n=len(boxes); blocks=np.zeros((n,12),int); blocks[:,:6]=1; blocks[:,8]=1
    data=dict(boxes=np.asarray(boxes,float),previous=np.vstack([e["initial_box"],np.asarray(boxes,float)[:-1]]),
        layer=np.full(n,9,int),peak=np.full(n,.6),head_calls=np.ones(n,int),block_calls=blocks,image_hw=np.tile([300,400],(n,1)))
    a.audit_trace(data,e)
    return data


def geometric_plan():
    return dict(search_factor=4.,search_size=256)


class TemporalTests(unittest.TestCase):
    def test_center_and_size_probes_are_separate_and_crop_equality_is_not_parent_equality(self):
        e=entry("fixture",6); good=[80,60,20,20]
        native=trace(e,[good]*6)
        # Center correct but predicted box 6x wider; GT-size replacement fixes it.
        large=[30,50,120,40]
        rows=a.frame_rows(e,trace(e,[large]*6),native,geometric_plan())
        self.assertEqual(rows[0]["output_center_error_gt_units"],0.)
        self.assertAlmostEqual(rows[0]["gt_size_at_pred_center_iou"],1.)
        self.assertLess(rows[0]["pred_size_at_gt_center_iou"],.2)
        shifted=trace(e,[[200,60,20,20]]*6)
        moved=a.frame_rows(e,shifted,native,geometric_plan())
        self.assertEqual(moved[0]["gt_size_at_pred_center_iou"],0.)
        self.assertEqual(moved[0]["pred_size_at_gt_center_iou"],1.)
        subpixel=trace(e,[[80.1,60,20,20]]*6)
        series=dict(left=a.frame_rows(e,subpixel,native,geometric_plan()),right=a.frame_rows(e,native,native,geometric_plan()))
        pair=a.compare_pair("fixture","left","right",series)
        self.assertEqual(pair["shared_parent_frames"],1)
        self.assertEqual(pair["shared_crop_frames"],6)
        self.assertEqual(pair["shared_parent_changed_output_frames"],1)
        self.assertIsNone(pair["first_official_divergence"])

    def test_onset_confirmation_recovery_local_order_and_left_censoring(self):
        e=entry("fixture",45); good=[80,60,20,20]
        native=trace(e,[good]*45)
        boxes=[good]*5+[[200,60,20,20]]*4+[[200,30,120,80]]*4+[good]*32
        data=trace(e,boxes); rows=a.frame_rows(e,data,native,geometric_plan())
        summary,events=a.summarize("fixture","rank_shape_seed2132",rows,e,data,native)
        r=a.event_row("fixture","rank_shape_seed2132",1,events[0],rows)
        self.assertEqual((r["onset_frame"],r["confirmation_frame"],r["end_frame"],r["recovery_frame"]),(6,8,13,14))
        self.assertEqual(r["center_vs_size_order"],"center_first")
        self.assertEqual((r["local_first_sustained_center_frame"],r["local_first_sustained_size_frame"]),(6,10))
        self.assertEqual(summary["longest_failure_frames"],8)
        self.assertIsNone(summary["first_long_failure_frame"])
        all_large=a.frame_rows(e,trace(e,[[30,50,120,40]]*45),native,geometric_plan())
        ev=a.failure_events([r["iou"] for r in all_large])[0]
        self.assertEqual(a.event_row("fixture","shape_seed2132",1,ev,all_large)["center_vs_size_order"],"left_censored")
        self.assertEqual(a.first([False,True,True,False,True,True,True],3),5)


class WorkflowTests(unittest.TestCase):
    def fixture(self,folder):
        root=folder/"stage48"; root.mkdir()
        marker=root/"source.marker"; marker.write_bytes(b"frozen source, no images")
        names=list(a.CONFIG["cases"])+["uav{:07d}_00000_s".format(i) for i in range(100,106)]
        entries=[entry(name) for name in names]
        plan=a.prior.seal(dict(schema=a.prior.SCHEMA,entries=entries,folds=a.prior.fold_design(entries),
            source_file_sha256={str(marker):a.file_hash(marker)},**geometric_plan()))
        a.prior.prior.motion.write_json(root/"manifest.json",plan)
        model_bindings={}
        for seed in a.prior.SEEDS:
            records=[]
            for fold in range(2):
                for arm in a.prior.ARMS:
                    path=root/"training"/("seed{}".format(seed))/("fold{}".format(fold))/arm/"step000256.pt"
                    path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes((arm+str(seed)+str(fold)).encode())
                    meta=dict(plan_fingerprint=plan["fingerprint"],mode="full",updates=256,fold=fold,seed=seed,arm=arm,
                        fit=plan["folds"][fold]["fit"],held_out=plan["folds"][fold]["held_out"])
                    receipt=a.prior.seal(dict(schema=a.prior.SCHEMA,metadata=meta,step=256,sha256=a.file_hash(path)))
                    a.prior.prior.motion.write_json(path.with_suffix(".receipt.json"),receipt)
                    history=[dict(step=s,total=1.,ordinary=1.,applied_rank=0.,applied_shape=0.,gradient_norm=2.,rank_views=2,size_views=2) for s in (1,128,256)]
                    records.append(dict(fold=fold,arm=arm,checkpoint=str(path),checkpoint_sha256=a.file_hash(path),
                        receipt_sha256=a.file_hash(path.with_suffix(".receipt.json")),buffers_unchanged=True,initial_map_error=0.,history=history))
                    for p in (path,path.with_suffix(".receipt.json")): model_bindings[str(p)]=a.file_hash(p)
            path=root/"training"/("seed{}".format(seed))/"stage48_training.json"
            report=a.prior.seal(dict(schema=a.prior.SCHEMA,plan_fingerprint=plan["fingerprint"],seed=seed,
                status="COMPLETE_FROZEN_HEAD_TRAINING",records=records))
            a.prior.prior.motion.write_json(path,report); model_bindings[str(path)]=a.file_hash(path)
        run=a.prior.seal(dict(schema=a.prior.SCHEMA,plan_fingerprint=plan["fingerprint"],
            role="native_head_own_prediction_oof_evaluation",model_sha256=model_bindings,feature_sha256={str(marker):a.file_hash(marker)}))
        a.prior.prior.motion.write_json(root/"evaluation/run.json",run)
        references={}; stats=[]; bindings={}
        for e in entries:
            name=e["sequence"]; n=len(e["frames"])-1; good=[80,60,20,20]
            boxes=[good]*n
            if name==a.CONFIG["cases"][3]: boxes=boxes[:9]+[[250,60,20,20]]*(n-9)
            if name==a.CONFIG["cases"][5]: boxes[29:35]=[[200,60,20,20]]*6
            native=trace(e,boxes); references[name]=native
            stats.append(dict(sequence=name,arm="native",seed=0,**a.video_metrics(native["boxes"],e,native["boxes"])))
            for seed in a.prior.SEEDS:
                for arm in a.prior.ARMS:
                    boxes=native["boxes"].tolist()
                    if name==a.CONFIG["cases"][0]: boxes[14:]=[[200,60,20,20]]*(n-14)
                    if name==a.CONFIG["cases"][1]: boxes[19:]=[[30,50,120,40]]*(n-19)
                    if name==a.CONFIG["cases"][2] and ((arm=="rank" and seed==2133) or (arm=="rank_shape" and seed==2132)):
                        boxes[5:11]=[[200,60,20,20]]*6; boxes[11:]=[[200,30,120,80]]*(n-11)
                    if name==a.CONFIG["cases"][3] and arm=="rank_shape" and seed==2133: boxes=[good]*n
                    if name==a.CONFIG["cases"][4]: boxes[24:27]=[[200,60,20,20]]*3
                    data=trace(e,boxes); method=arm+"_seed"+str(seed)
                    path=root/"evaluation/traces"/(name+"__"+method+".npz")
                    a.prior.save_cache(path,data,dict(run_fingerprint=run["fingerprint"],sequence=name,method=method))
                    for p in (path,path.with_suffix(".receipt.json")): bindings[str(p)]=a.file_hash(p)
                    stats.append(dict(sequence=name,arm=arm,seed=seed,**a.video_metrics(data["boxes"],e,native["boxes"])))
        video=[]; policies=[]
        for arm in ("native",)+a.prior.ARMS:
            for name in names:
                selected=[r for r in stats if r["sequence"]==name and r["arm"]==arm]
                values={k:float(np.mean([r[k] for r in selected])) for k in
                    ("auc","low_iou_fraction","failure_events","reliable_auc","mean_abs_log_size_error","oversize3_fraction")}
                video.append(dict(sequence=name,arm=arm,**values,introduced_failure=any(r["introduced_failure"] for r in selected)))
            selected=[r for r in video if r["arm"]==arm]
            policies.append(dict(arm=arm,videos=12,**{k:float(np.mean([r[k] for r in selected])) for k in
                ("auc","low_iou_fraction","failure_events","mean_abs_log_size_error","oversize3_fraction")},
                introduced_failure_videos=sum(r["introduced_failure"] for r in selected)))
        tables=dict(policy_summary=policies,video_summary=video,seed_video_summary=stats,
            paired=[dict(fixture=True)],probe_summary=[dict(fixture=True)],probe_policy_summary=[dict(fixture=True)])
        directory=root/"analysis"; directory.mkdir()
        for k,v in tables.items(): a.publish_table(directory/("stage48_"+k+".csv"),v)
        decision=a.prior.seal(dict(schema=a.prior.SCHEMA,plan_fingerprint=plan["fingerprint"],run_fingerprint=run["fingerprint"],
            role="consumed_development_native_head_two_factor_oof",status="REVIEW_NATIVE_HEAD_STABILITY_PILOT",videos=12,
            seeds=list(a.prior.SEEDS),trained_models=16,full_oof_trajectories=96,training_performed=True,exact_native_call_budget=True,
            inference_head_calls=96*65,inference_block_calls=96*65*7,model_sha256=model_bindings,source_file_sha256=bindings,
            table_sha256={"stage48_"+k+".csv":a.file_hash(directory/("stage48_"+k+".csv")) for k in tables}))
        a.prior.prior.motion.write_json(directory/"stage48_decision.json",decision)
        return root,plan,references

    def test_complete_audit_reuse_source_preservation_and_output_tamper(self):
        import builtins
        with tempfile.TemporaryDirectory(prefix="stage49-test-",dir=Path(__file__).parent) as folder:
            root,plan,refs=self.fixture(Path(folder)); original={str(p):p.read_bytes() for p in root.rglob("*") if p.is_file()}
            real_import=builtins.__import__
            def guarded(name,*args,**kwargs):
                if name.split(".")[0] in ("torch","cv2"): raise AssertionError("No model/image dependency imports")
                return real_import(name,*args,**kwargs)
            with patch.object(a.prior,"load_plan",return_value=plan),patch.object(a.prior,"source_native",side_effect=lambda p,e:refs[e["sequence"]]), \
                 patch("builtins.__import__",side_effect=guarded),contextlib.redirect_stdout(io.StringIO()):
                report=a.audit(root)
                self.assertEqual(report["status"],"COMPLETE_CACHED_HEAD_TRAJECTORY_AUDIT")
                self.assertEqual(report["focus_trajectories"],54)
                self.assertFalse(report["images_opened"]); self.assertFalse(report["model_weights_loaded"])
                self.assertEqual(report,a.audit(root))
                for name,payload in original.items(): self.assertEqual(Path(name).read_bytes(),payload)
                directory=root/"analysis/trajectory_audit_v1"
                rows=a.table_rows(directory/"stage49_timeline.csv",("sequence","method","frame_id"))
                self.assertEqual(len(rows),report["table_rows"]["timeline"])
                self.assertFalse(any("peak_cell" in r or "top5" in r for r in rows.values()))
                anchors=a.table_rows(directory/"stage49_anchors.csv",("sequence","anchor_id"))
                self.assertTrue(any("recovery_e" in r["kinds"] for r in anchors.values()))
                path=directory/"stage49_focus.csv"; path.write_bytes(path.read_bytes()+b"tampered")
                with self.assertRaisesRegex(ValueError,"input/output changed"): a.audit(root)

    def test_mixed_decision_checkpoint_and_trace_tamper(self):
        with tempfile.TemporaryDirectory(prefix="stage49-test-",dir=Path(__file__).parent) as folder:
            root,plan,refs=self.fixture(Path(folder)); decision_path=root/"analysis/stage48_decision.json"
            saved=a.prior.read_sealed(decision_path)
            with patch.object(a.prior,"load_plan",return_value=plan),patch.object(a.prior,"source_native",side_effect=lambda p,e:refs[e["sequence"]]),contextlib.redirect_stdout(io.StringIO()):
                changed={k:v for k,v in saved.items() if k!="fingerprint"}; changed["run_fingerprint"]="mixed"
                a.prior.prior.motion.write_json(decision_path,a.prior.seal(changed))
                with self.assertRaisesRegex(ValueError,"complete, unmixed"): a.load_completed(root)
                a.prior.prior.motion.write_json(decision_path,saved)
                checkpoint=next((root/"training").rglob("*.pt")); payload=checkpoint.read_bytes(); checkpoint.write_bytes(payload+b"changed")
                with self.assertRaisesRegex(ValueError,"input/output changed"): a.load_completed(root)
                checkpoint.write_bytes(payload)
                path=next((root/"evaluation/traces").glob("*.npz")); path.write_bytes(path.read_bytes()+b"changed")
                with self.assertRaisesRegex(ValueError,"input/output changed"): a.load_completed(root)

    def test_resealed_summary_disagreement_and_unsafe_output(self):
        with tempfile.TemporaryDirectory(prefix="stage49-test-",dir=Path(__file__).parent) as folder:
            root,plan,refs=self.fixture(Path(folder))
            with self.assertRaisesRegex(ValueError,"dedicated"): a.audit(root,root/"evaluation")
            path=root/"analysis/stage48_seed_video_summary.csv"
            rows=list(a.table_rows(path,("sequence","arm","seed")).values()); rows[0]["auc"]=float(rows[0]["auc"])+1
            path.unlink(); a.publish_table(path,rows)
            decision_path=root/"analysis/stage48_decision.json"; saved=a.prior.read_sealed(decision_path)
            changed={k:v for k,v in saved.items() if k!="fingerprint"}
            changed["table_sha256"][path.name]=a.file_hash(path)
            a.prior.prior.motion.write_json(decision_path,a.prior.seal(changed))
            with patch.object(a.prior,"load_plan",return_value=plan),patch.object(a.prior,"source_native",side_effect=lambda p,e:refs[e["sequence"]]),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,"summary/cache mismatch"): a.load_completed(root)


if __name__=="__main__": unittest.main()
