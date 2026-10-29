"""NumPy-only temporal diagnosis of completed native-SGLA head OOF trajectories."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint
from stage27_uav_collect import writer_lock
from stage27_uav_common import (failure_events, nonempty_path, publish_identical,
    scene_geometry, tracking_iou)
from stage46_shift_audit import publish_table, verify_files
from stage48_stability_experiment import audit_trace
from stage48_stability_summary import video_metrics
import stage48_stability_common as prior

SCHEMA = "stage49_cached_head_trajectory_audit_v1"
CONFIG = dict(cases=["uav0000304_00253_s", "uav0000307_04531_s", "uav0000226_05370_s",
    "uav0000176_00000_s", "uav0000170_00000_s", "uav0000204_00000_s"],
    before=10, after=30, ordering_before=30, sustained_frames=3,
    center_error_gt_sides=.5, oversize_ratio=3., undersize_ratio=1/3,
    coverage_threshold=.9, long_event_frames=30, float_atol=1e-9,
    fixed_quartiles=[1,2,3],
    selection="Six consumed cases selected after Stage-48 outcomes; descriptive temporal audit only")
TABLES = ("case_summary", "onsets", "pairs", "anchors", "phases", "timeline", "training_log", "focus")
METRICS = ("iou", "coverage", "parent_center_error_gt_units", "output_center_error_gt_units",
    "width_ratio", "height_ratio", "oversize3", "undersize_third", "gt_size_at_pred_center_iou",
    "pred_size_at_gt_center_iou", "raw_peak_amplitude")


def seal(body):
    return dict(body, fingerprint=fingerprint(body))


def read_report(path):
    value=json.loads(Path(path).read_text(encoding="utf-8"))
    body={k:v for k,v in value.items() if k!="fingerprint"}
    if value.get("schema")!=SCHEMA or value.get("fingerprint")!=fingerprint(body):
        raise ValueError("Stage-49 schema/fingerprint mismatch: "+str(path))
    return value


def table_rows(path, keys):
    with path.open(encoding="utf-8",newline="") as handle:
        rows=list(csv.DictReader(handle))
    result={tuple(r[k] for k in keys):r for r in rows}
    if len(result)!=len(rows): raise ValueError("Duplicate source summary rows: "+str(path))
    return result


def check_values(record, expected):
    for key,value in expected.items():
        if key not in record: raise ValueError("Source summary column missing: "+key)
        actual=record[key]
        if isinstance(value,(bool,np.bool_)):
            good=actual==str(bool(value))
        elif value is None:
            good=actual==""
        else:
            good=bool(np.isfinite(float(actual)) and abs(float(actual)-value)<=CONFIG["float_atol"])
        if not good: raise ValueError("Source summary/cache mismatch: "+key)


def load_completed(root):
    """Validate existing files/receipts and independently recompute all video metrics."""
    plan=prior.load_plan(root)
    run=prior.read_sealed(root/"evaluation/run.json")
    decision=prior.read_sealed(root/"analysis/stage48_decision.json")
    expected_tables={"stage48_"+n+".csv" for n in
        ("policy_summary","video_summary","seed_video_summary","paired","probe_summary","probe_policy_summary")}
    if (run.get("role")!="native_head_own_prediction_oof_evaluation" or
        run.get("plan_fingerprint")!=plan["fingerprint"] or
        decision.get("plan_fingerprint")!=plan["fingerprint"] or decision.get("run_fingerprint")!=run["fingerprint"] or
        decision.get("role")!="consumed_development_native_head_two_factor_oof" or
        decision.get("status") not in {"PROMISING_NATIVE_HEAD_STABILITY_PILOT","REVIEW_NATIVE_HEAD_STABILITY_PILOT"} or
        decision.get("training_performed") is not True or decision.get("exact_native_call_budget") is not True or
        decision.get("videos")!=12 or decision.get("seeds")!=list(prior.SEEDS) or
        decision.get("trained_models")!=16 or decision.get("full_oof_trajectories")!=96 or
        decision.get("model_sha256")!=run.get("model_sha256") or set(decision.get("table_sha256",{}))!=expected_tables or
        not set(CONFIG["cases"])<=set(e["sequence"] for e in plan["entries"])):
        raise ValueError("Need complete, unmixed Stage-48 summary and evaluation")
    bindings=dict(plan["source_file_sha256"])
    for name in ("model_sha256","feature_sha256"): bindings.update(run[name])
    bindings.update(decision["source_file_sha256"])
    bindings.update({str(root/"analysis"/n):h for n,h in decision["table_sha256"].items()})
    for name in ("manifest.json","evaluation/run.json","analysis/stage48_decision.json"):
        bindings[str(root/name)]=file_hash(root/name)
    verify_files(bindings)
    logs=[]
    for seed in prior.SEEDS:
        report_path=root/"training"/("seed{}".format(seed))/"stage48_training.json"
        record=prior.read_sealed(report_path)
        wanted={(f,a) for f in range(2) for a in prior.ARMS}
        if (str(report_path) not in run["model_sha256"] or record.get("plan_fingerprint")!=plan["fingerprint"] or
            record.get("seed")!=seed or record.get("status")!="COMPLETE_FROZEN_HEAD_TRAINING" or
            len(record["records"])!=8 or {(r["fold"],r["arm"]) for r in record["records"]}!=wanted):
            raise ValueError("Incomplete frozen OOF training records")
        for r in record["records"]:
            path=Path(r["checkpoint"]); receipt_path=path.with_suffix(".receipt.json")
            receipt=prior.read_sealed(receipt_path); m=receipt["metadata"]
            if (run["model_sha256"].get(str(path))!=r["checkpoint_sha256"] or
                run["model_sha256"].get(str(receipt_path))!=r["receipt_sha256"] or receipt["sha256"]!=r["checkpoint_sha256"] or
                receipt["step"]!=prior.DESIGN["updates"] or m.get("plan_fingerprint")!=plan["fingerprint"] or
                m.get("mode")!="full" or m.get("updates")!=prior.DESIGN["updates"] or
                (m.get("fold"),m.get("seed"),m.get("arm"))!=(r["fold"],seed,r["arm"]) or
                m.get("fit")!=plan["folds"][r["fold"]]["fit"] or m.get("held_out")!=plan["folds"][r["fold"]]["held_out"] or
                r.get("buffers_unchanged") is not True or r["initial_map_error"]>1e-6):
                raise ValueError("Frozen checkpoint/OOF receipt mismatch")
            for row in r["history"]:
                logs.append(dict(fold=r["fold"],seed=seed,arm=r["arm"],**row,
                    exceeds_clip_threshold=row["gradient_norm"]>prior.DESIGN["gradient_clip"]))
    seed_rows=table_rows(root/"analysis/stage48_seed_video_summary.csv",("sequence","arm","seed"))
    video_rows=table_rows(root/"analysis/stage48_video_summary.csv",("sequence","arm"))
    policy_rows=table_rows(root/"analysis/stage48_policy_summary.csv",("arm",))
    if (len(seed_rows),len(video_rows),len(policy_rows))!=(108,60,5): raise ValueError("Incomplete source summary tables")
    expected_files={e["sequence"]+"__"+a+"_seed"+str(s)+".npz" for e in plan["entries"] for a in prior.ARMS for s in prior.SEEDS}
    if {p.name for p in (root/"evaluation/traces").glob("*.npz")}!=expected_files:
        raise ValueError("Need exactly 96 complete Stage-48 traces")
    focus={}; recomputed=[]; heads=blocks=0
    for entry in plan["entries"]:
        sequence=entry["sequence"]; native=prior.source_native(plan,entry)
        arrays={"native":{k:native[k] for k in ("previous","boxes","layer","peak","head_calls","block_calls","image_hw")}}
        for seed in prior.SEEDS:
            for arm in prior.ARMS:
                method=arm+"_seed"+str(seed)
                path=root/"evaluation/traces"/(sequence+"__"+method+".npz")
                for p in (path,path.with_suffix(".receipt.json")):
                    if str(p) not in decision["source_file_sha256"]: raise ValueError("Unbound committed trace")
                arrays[method]=prior.read_cache(path,dict(run_fingerprint=run["fingerprint"],sequence=sequence,method=method))
                if arrays[method] is None: raise ValueError("Missing committed Stage-48 trace")
                heads+=int(arrays[method]["head_calls"].sum()); blocks+=int(arrays[method]["block_calls"].sum())
        for method,data in arrays.items():
            audit_trace(data,entry)
            arm,seed=("native",0) if method=="native" else (method.rsplit("_seed",1)[0],int(method.rsplit("_seed",1)[1]))
            values=video_metrics(data["boxes"],entry,native["boxes"])
            check_values(seed_rows[sequence,arm,str(seed)],values)
            recomputed.append(dict(sequence=sequence,arm=arm,seed=seed,**values))
        if sequence in CONFIG["cases"]: focus[sequence]=(entry,arrays)
        print("[Stage-48 traces audited] "+sequence,flush=True)
    if (heads,blocks)!=(decision["inference_head_calls"],decision["inference_block_calls"]):
        raise ValueError("Source inference budget mismatch")
    averaged=[]
    for (sequence,arm),record in video_rows.items():
        selected=[r for r in recomputed if r["sequence"]==sequence and r["arm"]==arm]
        values={k:(None if any(r[k] is None for r in selected) else float(np.mean([r[k] for r in selected])))
            for k in ("auc","low_iou_fraction","failure_events","mean_abs_log_size_error","oversize3_fraction","reliable_auc")}
        values["introduced_failure"]=any(r["introduced_failure"] for r in selected)
        check_values(record,values); averaged.append(dict(arm=arm,**values))
    for (arm,),record in policy_rows.items():
        selected=[r for r in averaged if r["arm"]==arm]
        values={k:float(np.mean([r[k] for r in selected])) for k in
            ("auc","low_iou_fraction","failure_events","mean_abs_log_size_error","oversize3_fraction")}
        check_values(record,dict(videos=12,**values,introduced_failure_videos=sum(r["introduced_failure"] for r in selected)))
    return plan,run,decision,focus,logs,bindings


def first(mask, run=1):
    count=0
    for i,value in enumerate(mask):
        count=count+1 if value else 0
        if count==run: return i-run+2  # one-based first frame, not confirmation
    return None


def center(box):
    value=np.asarray(box,float)
    return value[..., :2]+value[..., 2:]/2


def frame_rows(entry,data,native,plan):
    gt=np.asarray(entry["ground_truth"],float)[1:]; boxes=data["boxes"]
    iou=tracking_iou(boxes,gt); base_iou=tracking_iou(native["boxes"],gt)
    pc,gc,oc=center(data["previous"]),center(gt),center(boxes)
    side=np.sqrt(gt[:,2:].prod(1)); ratio=boxes[:,2:]/gt[:,2:]
    gt_size=np.column_stack([oc-gt[:,2:]/2,gt[:,2:]])
    pred_size=np.column_stack([gc-boxes[:,2:]/2,boxes[:,2:]])
    a,b=tracking_iou(gt_size,gt),tracking_iou(pred_size,gt)
    rows=[]
    for i in range(len(boxes)):
        geom=scene_geometry(data["previous"][i],gt[i],data["image_hw"][i],plan["search_factor"],plan["search_size"])
        native_geom=scene_geometry(native["previous"][i],gt[i],data["image_hw"][i],plan["search_factor"],plan["search_size"])
        error=float(np.linalg.norm(oc[i]-gc[i])/side[i])
        row=dict(frame_id=i+1,source_file=entry["frames"][i+1]["file"],iou=float(iou[i]),native_iou=float(base_iou[i]),
            coverage=float(geom["coverage"]),projected_gt_side_px=float(np.sqrt(geom["input_wh"].prod())),
            parent_center_error_gt_units=float(np.linalg.norm(pc[i]-gc[i])/side[i]),output_center_error_gt_units=error,
            dx_gt_width=float((oc[i,0]-gc[i,0])/gt[i,2]),dy_gt_height=float((oc[i,1]-gc[i,1])/gt[i,3]),
            width_ratio=float(ratio[i,0]),height_ratio=float(ratio[i,1]),
            oversize3=bool((ratio[i]>CONFIG["oversize_ratio"]).any()),
            undersize_third=bool((ratio[i]<CONFIG["undersize_ratio"]).any()),center_bad=error>=CONFIG["center_error_gt_sides"],
            coverage_bad=bool(geom["coverage"]<CONFIG["coverage_threshold"]),
            gt_size_at_pred_center_iou=float(a[i]),pred_size_at_gt_center_iou=float(b[i]),
            parent_differs_native=bool(np.max(np.abs(data["previous"][i]-native["previous"][i]))>CONFIG["float_atol"]),
            crop_differs_native=bool(np.any(geom["crop"]!=native_geom["crop"])),
            output_differs_native=bool(np.max(np.abs(boxes[i]-native["boxes"][i]))>CONFIG["float_atol"]),
            official_box_differs_native=bool(np.any(np.trunc(boxes[i])!=np.trunc(native["boxes"][i]))),
            layer=int(data["layer"][i]),native_layer=int(native["layer"][i]),
            raw_peak_amplitude=float(data["peak"][i]),native_raw_peak_amplitude=float(native["peak"][i]))
        for prefix,value in (("output",boxes[i]),("parent",data["previous"][i]),("gt",gt[i]),("crop",geom["crop"])):
            row.update({prefix+"_"+key:float(v) for key,v in zip(("x","y","w","h"),value)})
        rows.append(row)
    return rows


def size_bad(row):
    return row["oversize3"] or row["undersize_third"]


def summarize(sequence,method,rows,entry,data,native):
    events=failure_events([r["iou"] for r in rows]); n=len(rows)
    longest=max((e["end"]-e["start"]+1 for e in events),default=0)
    long=[e for e in events if e["end"]-e["start"]+1>=CONFIG["long_event_frames"]]
    arm,seed=("native",0) if method=="native" else (method.rsplit("_seed",1)[0],int(method.rsplit("_seed",1)[1]))
    summary=dict(sequence=sequence,method=method,arm=arm,seed=seed,tracked_frames=n,
        **video_metrics(data["boxes"],entry,native["boxes"]),low_iou_frames=sum(r["iou"]<.2 for r in rows),
        longest_failure_frames=longest,longest_failure_fraction=longest/n,
        first_failure_frame=None if not events else events[0]["start"]+1,
        first_long_failure_frame=None if not long else long[0]["start"]+1,
        first_parent_divergence=first([r["parent_differs_native"] for r in rows]),
        first_crop_divergence=first([r["crop_differs_native"] for r in rows]),
        first_output_divergence=first([r["output_differs_native"] for r in rows]),
        first_official_divergence=first([r["official_box_differs_native"] for r in rows]),
        first_sustained_center_bad=first([r["center_bad"] for r in rows],CONFIG["sustained_frames"]),
        first_sustained_size_bad=first([size_bad(r) for r in rows],CONFIG["sustained_frames"]),
        first_sustained_coverage_bad=first([r["coverage_bad"] for r in rows],CONFIG["sustained_frames"]))
    return summary,events


def event_row(sequence,method,number,event,rows):
    start,end=event["start"],event["end"]; lo=max(0,start-CONFIG["ordering_before"])
    hi=min(len(rows),start+CONFIG["after"]+1); local=rows[lo:hi]
    times={}
    for key,fn in (("center",lambda r:r["center_bad"]),("size",size_bad),("coverage",lambda r:r["coverage_bad"])):
        frame=first([fn(r) for r in local],CONFIG["sustained_frames"])
        times[key]=None if frame is None else lo+frame
    left_censored=bool(local[0]["center_bad"] or size_bad(local[0]))
    a,b=times["center"],times["size"]
    order=("left_censored" if left_censored else "neither" if a is None and b is None else
        "size_only" if a is None else "center_only" if b is None else "same_frame" if a==b else "center_first" if a<b else "size_first")
    row=dict(sequence=sequence,method=method,event=number,onset_frame=start+1,confirmation_frame=event["confirmation"]+1,
        end_frame=end+1,recovery_frame=None if event["recovery"] is None else event["recovery"]+1,censor=event["censor"],
        duration_frames=end-start+1,low_iou_frames=sum(r["iou"]<.2 for r in rows[start:end+1]),
        local_window_start=lo+1,local_window_end=hi,center_vs_size_order=order,
        center_bad_at_window_start=local[0]["center_bad"],size_bad_at_window_start=size_bad(local[0]),
        **{"local_first_sustained_"+k+"_frame":v for k,v in times.items()},
        **{"at_onset_"+k:rows[start][k] for k in METRICS})
    return row


def compare_pair(sequence,left,right,series):
    a,b=series[left],series[right]
    def boxes(rows,prefix): return np.asarray([[r[prefix+"_"+k] for k in ("x","y","w","h")] for r in rows])
    pa,pb=boxes(a,"parent"),boxes(b,"parent"); oa,ob=boxes(a,"output"),boxes(b,"output")
    ca,cb=boxes(a,"crop"),boxes(b,"crop")
    parent=np.max(np.abs(pa-pb),1)>CONFIG["float_atol"]; output=np.max(np.abs(oa-ob),1)>CONFIG["float_atol"]
    crop=np.any(ca!=cb,1); official=np.any(np.trunc(oa)!=np.trunc(ob),1)
    at=first(official); t=None if at is None else at-1
    return dict(sequence=sequence,left=left,right=right,first_parent_divergence=first(parent),first_crop_divergence=first(crop),
        first_output_divergence=first(output),first_official_divergence=at,
        shared_parent_frames=int((~parent).sum()),shared_crop_frames=int((~crop).sum()),
        shared_parent_changed_output_frames=int((~parent&output).sum()),
        shared_crop_changed_output_frames=int((~crop&output).sum()),
        left_iou_at_first_official=None if t is None else a[t]["iou"],right_iou_at_first_official=None if t is None else b[t]["iou"],
        parents_equal_at_first_official=None if t is None else not bool(parent[t]),
        first_layer_divergence=first([x["layer"]!=y["layer"] for x,y in zip(a,b)]))


def analyze_case(entry,arrays,plan):
    sequence=entry["sequence"]; native=arrays["native"]
    series={m:frame_rows(entry,d,native,plan) for m,d in arrays.items()}
    parts={k:[] for k in TABLES if k!="training_log"}; anchors={}
    for method,rows in series.items():
        summary,events=summarize(sequence,method,rows,entry,arrays[method],native)
        parts["case_summary"].append(summary)
        for number,event in enumerate(events,1):
            parts["onsets"].append(event_row(sequence,method,number,event,rows))
            anchors.setdefault((method,event["start"]+1),[]).append("failure_e"+str(number))
            if event["recovery"] is not None:
                anchors.setdefault((method,event["recovery"]+1),[]).append("recovery_e"+str(number))
        for key in ("first_output_divergence","first_official_divergence","first_failure_frame","first_long_failure_frame",
                    "first_sustained_center_bad","first_sustained_size_bad","first_sustained_coverage_bad"):
            frame=summary[key]
            if frame is not None: anchors.setdefault((method,frame),[]).append(key)
    n=len(next(iter(series.values())))
    for q in CONFIG["fixed_quartiles"]: anchors.setdefault(("native",max(1,n*q//4)),[]).append("fixed_q"+str(q))
    membership={}
    for ordinal,((owner,frame),kinds) in enumerate(sorted(anchors.items()),1):
        anchor=dict(sequence=sequence,anchor_id=ordinal,owner=owner,frame_id=frame,kinds="|".join(kinds))
        parts["anchors"].append(anchor)
        phases=dict(before=range(max(1,frame-CONFIG["before"]),frame),anchor=range(frame,frame+1),
            after=range(frame+1,min(n,frame+CONFIG["after"])+1))
        for phase,frames in phases.items():
            for f in frames: membership.setdefault(f,[]).append(str(ordinal)+":"+phase)
            if not frames: continue
            for method,rows in series.items():
                picked=[rows[f-1] for f in frames]
                parts["phases"].append(dict(**anchor,evaluator=method,phase=phase,frames=len(picked),
                    first_frame=picked[0]["frame_id"],last_frame=picked[-1]["frame_id"],
                    **{"mean_"+k:float(np.mean([r[k] for r in picked])) for k in METRICS}))
    for method,rows in series.items():
        for f in sorted(membership): parts["timeline"].append(dict(sequence=sequence,method=method,
            anchor_phase_memberships="|".join(membership[f]),**rows[f-1]))
    for seed in prior.SEEDS:
        for left,right in (("supervised","native"),("rank_shape","native"),("rank","supervised"),("shape","supervised"),
            ("rank_shape","supervised"),("rank_shape","rank"),("rank_shape","shape")):
            l=left+"_seed"+str(seed); r=right if right=="native" else right+"_seed"+str(seed)
            parts["pairs"].append(compare_pair(sequence,l,r,series))
    for arm in prior.ARMS: parts["pairs"].append(compare_pair(sequence,arm+"_seed2132",arm+"_seed2133",series))
    by_method={r["method"]:r for r in parts["case_summary"]}
    row=dict(sequence=sequence,native_auc=by_method["native"]["auc"],native_first_failure=by_method["native"]["first_failure_frame"])
    for seed in prior.SEEDS:
        r=by_method["rank_shape_seed"+str(seed)]
        for key in ("auc","low_iou_frames","first_failure_frame","first_long_failure_frame","longest_failure_frames"):
            row["combination_"+str(seed)+"_"+key]=r[key]
    parts["focus"].append(row)
    return parts


def audit(root,output_dir=None):
    root=Path(root).expanduser().resolve(); dedicated=root/"analysis/trajectory_audit_v1"
    output=Path(output_dir).expanduser().resolve() if output_dir else dedicated
    if output==root or output in root.parents or (root in output.parents and output!=dedicated):
        raise ValueError("Use dedicated trajectory_audit_v1 or a separate new audit directory")
    plan,run,decision,focus,logs,bindings=load_completed(root)
    code={name:file_hash(Path(__file__).with_name(name)) for name in
        ("stage49_trajectory_audit.py","test_stage49_trajectory_audit.py","stage48_stability_summary.py",
         "stage48_stability_experiment.py","stage46_shift_audit.py","stage27_uav_common.py","validation_metrics.py")}
    identity=dict(schema=SCHEMA,configuration=CONFIG,source_plan_fingerprint=plan["fingerprint"],
        source_run_fingerprint=run["fingerprint"],source_decision_fingerprint=decision["fingerprint"],
        source_file_sha256=bindings,code_sha256=code)
    with writer_lock(output):
        report_path=output/"stage49_diagnostic.json"
        if report_path.exists():
            saved=read_report(report_path)
            if any(saved.get(k)!=v for k,v in identity.items()) or read_report(output/"manifest.json")!=seal(identity):
                raise ValueError("Completed Stage-49 source/code/configuration changed")
            verify_files({str(output/n):h for n,h in saved["table_sha256"].items()})
            print("Stage-49: COMPLETE_CACHED_HEAD_TRAJECTORY_AUDIT (reuse, audited)")
            print("Reports: "+str(output))
            return saved
        allowed={"collect.lock","manifest.json"}|{"stage49_"+n+".csv" for n in TABLES}
        allowed|={n+".tmp" for n in allowed if n.endswith(".csv")}
        if any(p.name not in allowed for p in output.iterdir()): raise ValueError("Use an empty or matching audit directory")
        publish_identical(output/"manifest.json",seal(identity))
        tables={k:[] for k in TABLES}; tables["training_log"]=logs
        for sequence in CONFIG["cases"]:
            entry,arrays=focus[sequence]
            for name,rows in analyze_case(entry,arrays,plan).items(): tables[name].extend(rows)
            print("[temporal case audited] "+sequence,flush=True)
        for name,rows in tables.items(): publish_table(output/("stage49_"+name+".csv"),rows)
        verify_files(bindings)
        report=seal(dict(**identity,status="COMPLETE_CACHED_HEAD_TRAJECTORY_AUDIT",
            role="posthoc_consumed_development_temporal_diagnosis",source_videos_audited=12,
            source_adapted_trajectories_audited=96,source_native_trajectories_audited=12,
            focus_videos=len(focus),focus_trajectories=len(tables["case_summary"]),
            table_rows={k:len(v) for k,v in tables.items()},
            table_sha256={"stage49_"+k+".csv":file_hash(output/("stage49_"+k+".csv")) for k in TABLES},
            training_performed=False,new_inference_performed=False,images_opened=False,model_weights_loaded=False,adoption_gate=None,
            warnings=["Six cases/anchors use consumed outcomes; no new method or independent statistical validation.",
                "All trajectories keep their own recorded parents. Cross-method windows are not matched-state causal interventions.",
                "Frame IDs index tracked images starting at one; initialization is image zero. Source filenames are retained in the timeline.",
                "Native failure/recovery rules are reused. A >=30-frame event may include intermediate non-low-IoU frames.",
                "Center/size/coverage thresholds and their ordering are descriptive. Left-censored local starts do not establish which error began first.",
                "Coverage does not establish visibility; geometric error does not establish semantic distractor identity.",
                "No response location, Top-K retention or dense maps can be reconstructed from full-trajectory boxes/peak amplitudes.",
                "GT-size/GT-center replacements are offline non-additive geometric probes; temporal ordering alone does not establish causality.",
                "Equal rounded crops can have different floating mapping origins; shared-parent/crop counts do not prove identical internal responses.",
                "Only logged training steps are shown. Pre-clip norms do not measure clipping frequency across all 256 updates.",
                "Checkpoint bytes and receipts are verified without deserializing model weights. No branch-weight or optimizer-state attribution is performed."],
            next_step="Use onset and recovery timing to select a few frozen matched-parent replays or a controlled own-state training experiment; preserve all Stage-48 outcomes"))
        publish_identical(report_path,report)
    print("Stage-49: COMPLETE_CACHED_HEAD_TRAJECTORY_AUDIT; no training, images, model loading or GPU inference")
    for r in tables["focus"]:
        print(r["sequence"],"native failure",r["native_first_failure"],"combination first/longest",
            *[(r["combination_"+str(s)+"_first_failure_frame"],r["combination_"+str(s)+"_longest_failure_frames"]) for s in prior.SEEDS])
    print("Reports: "+str(output))
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=nonempty_path,required=True,help="Completed Stage-48 root")
    parser.add_argument("--output_dir",type=nonempty_path,help="Optional separate new diagnostic output")
    args=parser.parse_args(); audit(args.input,args.output_dir)
