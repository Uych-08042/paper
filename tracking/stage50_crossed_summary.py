"""Describe frozen-parent x head matrices and raw/Hann/GT-cell geometry."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stage27_uav_collect import writer_lock
from stage27_uav_common import nonempty_path,publish_identical,scene_geometry,tracking_iou
from stage28_response_common import dense_boxes,response_metrics
from stage46_shift_audit import publish_table,verify_files
import stage50_crossed_common as c


def measures(box,gt):
    center=lambda b:b[:2]+b[2:]/2
    return dict(iou=float(tracking_iou(box,gt)),center_error_gt_units=float(np.linalg.norm(center(box)-center(gt))/np.sqrt(gt[2:].prod())),
        width_ratio=float(box[2]/gt[2]),height_ratio=float(box[3]/gt[3]))


def analyze(data,point,plan,inputs):
    entry,_=inputs["focus"][point["sequence"]]; source=inputs["plan"]
    case=next(e for e in plan["cases"] if e["case_id"]==point["case_id"])
    gt=np.asarray(entry["ground_truth"][point["frame"]],float); side=int(source["runtime"]["lens_x"]**.5)
    geom=scene_geometry(data["parent"],gt,data["image_hw"],source["search_factor"],source["search_size"])
    gt_xy=(gt[:2]+gt[2:]/2-data["crop"][:2])/data["crop"][2]
    valid=bool(((gt_xy>=0)&(gt_xy<1)).all()); xy=np.clip(np.round(gt_xy*side).astype(int),0,side-1)
    gi=int(xy[1]*side+xy[0]) if valid else None
    window=np.asarray(source["window"],np.float32)
    responses=[]; cells=[]; matrix=dict(**point,purpose=case["purpose"],coverage=float(geom["coverage"]))
    for h,name in enumerate(c.HEADS):
        raw=data["raw"][h]; weighted=raw*window; ri=int(np.argmax(raw)); wi=int(np.argmax(weighted))
        dense=dense_boxes(data["size"][h],data["offset"][h],data["parent"],float(data["factor"]),source["search_size"],data["image_hw"])
        dense[ri]=data["raw_boxes"][h]; dense[wi]=data["boxes"][h]
        metrics,_,_=response_metrics(raw,window,dense,gt,native_box=data["boxes"][h])
        row=dict(**point,purpose=case["purpose"],evaluator=name,on_diagonal=name==point["history"],
            coverage=float(geom["coverage"]),projected_gt_side_px=float(np.sqrt(geom["input_wh"].prod())),
            parent_center_error_gt_units=measures(data["parent"],gt)["center_error_gt_units"],
            raw_cell=ri,hann_cell=wi,gt_cell=gi,gt_cell_valid=valid,
            raw_hann_differ=ri!=wi,hann_gt_cell_differ=None if gi is None else wi!=gi,
            raw_iou=measures(data["raw_boxes"][h],gt)["iou"],
            **{"output_"+k:v for k,v in measures(data["boxes"][h],gt).items()},
            dense_best_gt_iou=metrics["dense_best_gt_iou"],weighted_top5_gt_recall=metrics["weighted_top5_gt_recall"],
            weighted_gt_rank_in_top5=metrics["weighted_gt_rank_in_top5"],weighted_gt_score_margin=metrics["weighted_gt_score_margin"])
        for label,index in (("raw",ri),("hann",wi),("gt",gi)):
            if index is None:
                row["gt_cell_unclipped_log_size_error"]=None
                continue
            physical=data["size"][h].reshape(2,-1)[:,index].astype(float)*float(data["crop"][2])
            offset=data["offset"][h].reshape(2,-1)[:,index]
            error=float(np.abs(np.log(physical/gt[2:])).mean())
            box=data["raw_boxes"][h] if label=="raw" else data["boxes"][h] if label=="hann" else dense[index]
            cells.append(dict(**point,evaluator=name,position=label,cell=index,cell_x=index%side,cell_y=index//side,
                raw_response=float(raw.ravel()[index]),hann_response=float(weighted.ravel()[index]),
                offset_x=float(offset[0]),offset_y=float(offset[1]),
                unclipped_width=float(physical[0]),unclipped_height=float(physical[1]),
                unclipped_width_ratio=float(physical[0]/gt[2]),unclipped_height_ratio=float(physical[1]/gt[3]),
                unclipped_log_size_error=error,**{"decoded_"+k:v for k,v in measures(box,gt).items()}))
            row[label+"_cell_unclipped_log_size_error"]=error
        responses.append(row); matrix[name+"_iou"]=row["output_iou"]
    own=next(r for r in responses if r["on_diagonal"])
    focus_fields=("coverage","parent_center_error_gt_units","raw_cell","hann_cell","gt_cell","gt_cell_valid",
        "raw_hann_differ","hann_gt_cell_differ","raw_iou","output_iou","output_center_error_gt_units",
        "output_width_ratio","output_height_ratio","raw_cell_unclipped_log_size_error","hann_cell_unclipped_log_size_error",
        "gt_cell_unclipped_log_size_error","dense_best_gt_iou","weighted_top5_gt_recall","weighted_gt_rank_in_top5")
    focus=dict(**point,purpose=case["purpose"],**{k:own[k] for k in focus_fields})
    selected=point["frame"] in case["focus_frames"]
    return responses,cells,[matrix] if selected else [],[focus] if selected else []


def summarize(root):
    root=Path(root).expanduser().resolve(); plan,inputs=c.load_plan(root); run=c.read_sealed(root/"run.json")
    gate=c.read_sealed(root/"smoke/stage50_smoke.json")
    if (run.get("plan_fingerprint")!=plan["fingerprint"] or run.get("role")!="independent_shared_feature_head_replay" or
        run.get("runtime")!=inputs["plan"]["runtime_identity"] or run.get("heads")!=list(c.HEADS) or
        run.get("window")!=inputs["plan"]["window"] or gate.get("run_fingerprint")!=run["fingerprint"] or
        gate.get("status")!="PASS_SHARED_PARENT_HEAD_SMOKE"):
        raise ValueError("Need unchanged Stage-50 run and smoke")
    tables={k:[] for k in ("responses","cell_positions","crossed_matrix","focus")}; bindings={}; checks=[]; missing=[]
    points=c.points(plan)
    expected={c.path_for(root,p).name for p in points}
    if {p.name for p in (root/"responses").glob("*.npz")}-expected: raise ValueError("Unexpected replay cache")
    for point in points:
        path=c.path_for(root,point); data=c.read_cache(path,c.metadata(run,point))
        if data is None: missing.append(point["point_id"]); continue
        checks.append(c.audit_point(data,point,inputs,plan))
        for key,rows in zip(tables,analyze(data,point,plan,inputs)): tables[key].extend(rows)
        for p in (path,path.with_suffix(".receipt.json")): bindings[str(p)]=c.file_hash(p)
    if missing:
        print("Stage-50: REPLAY_INCOMPLETE; {} missing shared inputs".format(len(missing))); return None
    directory=root/"analysis"
    with writer_lock(directory):
        for name,rows in tables.items(): publish_table(directory/("stage50_"+name+".csv"),rows)
        verify_files(bindings); verify_files(plan["source_file_sha256"])
        report=c.seal(dict(schema=c.SCHEMA,status="COMPLETE_SHARED_PARENT_HEAD_DIAGNOSTIC",
            role="posthoc_frozen_head_by_predicted_parent_mechanism_diagnosis",plan_fingerprint=plan["fingerprint"],run_fingerprint=run["fingerprint"],
            shared_inputs=len(points),full_sgla_forwards=len(points),head_only_forwards=len(points)*(len(c.HEADS)-1),
            total_head_forwards=len(points)*len(c.HEADS),full_backbone_block_calls=len(points)*7,
            **{key:max(r[key] for r in checks) for key in checks[0]},
            replay_cache_sha256=bindings,table_rows={k:len(v) for k,v in tables.items()},
            table_sha256={"stage50_"+k+".csv":c.file_hash(directory/("stage50_"+k+".csv")) for k in tables},
            training_performed=False,weights_frozen=True,new_inference_performed=True,continuation_performed=False,
            new_closed_loop_policy_evaluated=False,adoption_gate=None,
            warnings=["Cases/frames were selected after consumed development outcomes; this is not independent method validation.",
                "All heads share one exact original feature tensor per input; histories remain frozen predicted parents, never GT parents.",
                "Own-model diagonals reproduce their source outputs; off-diagonals are independent current-frame responses, not recovery rollouts.",
                "Raw/Hann cells are response-grid indices; the GT cell uses actual rounded-crop coordinates and is diagnostic only.",
                "Different cells do not alone establish harmful training mismatch. Compare scores, geometry and retained GT candidates.",
                "Physical size-map ratios precede image clipping; decoded-box ratios and official integer IoU can differ.",
                "GT coverage does not establish visibility. A geometric non-target box is not a confirmed semantic distractor.",
                "All-cell matches and fixed Top-5 recall are hindsight diagnostics. No new accuracy, speed or causal trajectory gain is established."],
            next_step="Compare shared-parent matrices and raw/Hann/GT-cell geometry; choose an execution-aligned geometry or own-state recovery training control based on the observed results"))
        publish_identical(directory/"stage50_diagnostic.json",report)
    print("Stage-50: COMPLETE_SHARED_PARENT_HEAD_DIAGNOSTIC; {} full forwards + {} head-only forwards".format(
        report["full_sgla_forwards"],report["head_only_forwards"]))
    print("Reports: "+str(directory))
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--input",required=True,type=nonempty_path)
    summarize(parser.parse_args().input)
