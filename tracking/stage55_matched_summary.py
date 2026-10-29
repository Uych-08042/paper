"""Current-frame model/history accounting and matched-start branch descriptions."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from stage27_uav_collect import writer_lock
from stage27_uav_common import nonempty_path,scene_geometry,tracking_iou,failure_events
from stage28_response_common import dense_boxes,response_metrics
from stage46_shift_audit import publish_table
from stage50_crossed_summary import measures
import stage55_matched_common as c

def frame_measures(box,parent,gt,hw,source):
    m=measures(box,gt); geometry=scene_geometry(parent,gt,hw,source["search_factor"],source["search_size"])
    center=box[:2]+box[2:]/2; gtcenter=gt[:2]+gt[2:]/2
    m.update(coverage=float(geometry["coverage"]),parent_center_error_gt_units=measures(parent,gt)["center_error_gt_units"],
        gt_size_at_pred_center_iou=float(tracking_iou(np.r_[center-gt[2:]/2,gt[2:]],gt)),
        pred_size_at_gt_center_iou=float(tracking_iou(np.r_[gtcenter-box[2:]/2,box[2:]],gt)),
        log_size_error=float(np.abs(np.log(box[2:]/gt[2:])).mean()))
    return m

def analyze_point(data,item,inputs):
    source=inputs["plan"]; entry,_=inputs["arrays"][item["sequence"]]; gt=np.asarray(entry["ground_truth"][item["frame"]],float)
    rows=[]; matrix=dict(**item,coverage=float(scene_geometry(data["parent"],gt,data["image_hw"],source["search_factor"],source["search_size"])["coverage"]))
    for i,name in enumerate(c.HEADS):
        raw=data["raw"][i]; wi=int(np.argmax(raw*np.asarray(source["window"],np.float32))); ri=int(np.argmax(raw))
        dense=dense_boxes(data["size"][i],data["offset"][i],data["parent"],float(data["factor"]),source["search_size"],data["image_hw"])
        dense[wi]=data["boxes"][i]; dense[ri]=data["raw_boxes"][i]
        metrics,_,_=response_metrics(raw,np.asarray(source["window"],np.float32),dense,gt,native_box=data["boxes"][i])
        row=dict(**item,evaluator=name,on_diagonal=name==item["history"],layer=int(data["layer"]),raw_cell=ri,hann_cell=wi,
            **frame_measures(data["boxes"][i],data["parent"],gt,data["image_hw"],source),
            dense_best_gt_iou=metrics["dense_best_gt_iou"],weighted_top5_gt_recall=metrics["weighted_top5_gt_recall"],
            weighted_gt_rank_in_top5=metrics["weighted_gt_rank_in_top5"],weighted_gt_score_margin=metrics["weighted_gt_score_margin"])
        rows.append(row); matrix[name+"_iou"]=row["iou"]
    return rows,matrix

def analyze_branch(data,item,inputs):
    source=inputs["plan"]; entry,_=inputs["arrays"][item["sequence"]]; timeline=[]
    for j,f in enumerate(data["frame_id"]):
        gt=np.asarray(entry["ground_truth"][int(f)],float)
        timeline.append(dict(**item,step=j,source_frame=int(f),layer=int(data["layer"][j]),
            hann_cell=int(np.argmax(data["raw"][j]*np.asarray(source["window"],np.float32))),
            **frame_measures(data["boxes"][j],data["parents"][j],gt,data["image_hw"][j],source)))
    iou=np.array([r["iou"] for r in timeline]); events=failure_events(iou)
    summary=dict(**item,horizon=len(iou),start_iou=float(iou[0]),endpoint_iou=float(iou[-1]),mean_iou=float(iou.mean()),
        low_iou_fraction=float((iou<.2).mean()),left_censored_low_start=bool(iou[0]<.2),
        first_local_failure_frame=int(data["frame_id"][events[0]["start"]]) if events else None,
        first_local_recovery_frame=next((int(data["frame_id"][e["recovery"]]) for e in events if e["recovery"] is not None),None),
        window_unrecovered_events=sum(e["censor"]=="video_end" for e in events))
    for k in ("coverage","center_error_gt_units","parent_center_error_gt_units","log_size_error"):
        summary["mean_"+k]=float(np.mean([r[k] for r in timeline]))
    return summary,timeline

def first_difference(left,right,frames,atol):
    flags=np.abs(left-right)>atol
    indices=np.flatnonzero(flags.reshape(len(frames),-1).any(1))
    return int(frames[indices[0]]) if len(indices) else None

def summarize(root):
    root=Path(root).expanduser().resolve(); plan,inputs=c.load_plan(root); run=c.read_sealed(root/"run.json")
    gate=c.read_sealed(root/"smoke/stage55_smoke.json")
    if (run.get("plan_fingerprint")!=plan["fingerprint"] or run.get("role")!="frozen_shared_head_and_matched_branch" or
        run.get("runtime")!=inputs["plan"]["runtime_identity"] or run.get("heads")!=list(c.HEADS) or
        gate.get("status")!="PASS_MATCHED_BRANCH_SMOKE" or gate.get("run_fingerprint")!=run["fingerprint"]):
        raise ValueError("Need unchanged run and passing smoke")
    tables=dict(point_details=[],crossed_matrix=[],rollout_summary=[],branch_contrasts=[],focus=[],timelines=[])
    bindings={}; checks=[]; branch_checks=[]; point_data={}; branch_data={}; branch_rows={}; missing=[]
    full=head_only=blocks=selectors=0
    items=c.points(plan)+c.branches(plan)
    for directory,key in (("responses","point_id"),("branches","branch_id")):
        wanted={item[key]+".npz" for item in items if key in item}
        if {p.name for p in (root/directory).glob("*.npz")}-wanted: raise ValueError("Unexpected diagnostic cache")
    for item in c.points(plan):
        path=c.path_for(root,item); data=c.read_cache(path,c.metadata(run,item))
        if data is None: missing.append(path.name); continue
        checks.append(c.audit_point(data,item,plan,inputs)); details,matrix=analyze_point(data,item,inputs)
        tables["point_details"].extend(details)
        if item["offset"]==0:
            tables["crossed_matrix"].append(matrix); point_data[item["case_id"],item["history"]]=data
        full+=1; head_only+=int(data["head_forward_calls"][1:].sum()); blocks+=int(data["full_block_calls"].sum()); selectors+=int(data["selector_calls"])
        for p in (path,path.with_suffix(".receipt.json")): bindings[str(p)]=c.file_hash(p)
    for item in c.branches(plan):
        path=c.path_for(root,item); data=c.read_cache(path,c.metadata(run,item)); start=point_data.get((item["case_id"],item["history"]))
        if data is None or start is None: missing.append(path.name); continue
        branch_checks.append(c.audit_branch(data,item,start,plan,inputs)); row,timeline=analyze_branch(data,item,inputs)
        tables["rollout_summary"].append(row); tables["timelines"].extend(timeline)
        key=(item["case_id"],item["history"],item["evaluator"]); branch_data[key]=data; branch_rows[key]=row
        full+=int(data["head_calls"].sum()); blocks+=int(data["block_calls"].sum()); selectors+=int(data["selector_calls"].sum())
        for p in (path,path.with_suffix(".receipt.json")): bindings[str(p)]=c.file_hash(p)
    if missing:
        print("Stage-55: DIAGNOSTIC_INCOMPLETE; {} missing caches".format(len(missing))); return None
    if (full,head_only,blocks,selectors)!=(plan["expected_full_forwards"],plan["expected_head_only_forwards"],plan["expected_full_forwards"]*7,plan["expected_full_forwards"]):
        raise ValueError("Exact native/head-only diagnostic budget changed")
    maximum_residual=0.
    for case in plan["cases"]:
        cid=case["case_id"]
        for seed in c.prior.SEEDS:
            sn="rollout_supervised_seed"+str(seed); pn="rollout_preserve_seed"+str(seed)
            for history in c.HEADS:
                left=branch_data[cid,history,pn]; right=branch_data[cid,history,sn]
                l=branch_rows[cid,history,pn]; r=branch_rows[cid,history,sn]
                row=dict(case_id=cid,sequence=case["sequence"],frame=case["frame"],history=history,seed=seed,
                    start_iou_gain=l["start_iou"]-r["start_iou"],mean_iou_gain=l["mean_iou"]-r["mean_iou"],
                    endpoint_iou_gain=l["endpoint_iou"]-r["endpoint_iou"],low_iou_fraction_delta=l["low_iou_fraction"]-r["low_iou_fraction"],
                    first_box_divergence_frame=first_difference(left["boxes"],right["boxes"],left["frame_id"],1e-6),
                    first_crop_divergence_frame=first_difference(left["crop"],right["crop"],left["frame_id"],0),
                    first_response_divergence_frame=first_difference(left["raw"],right["raw"],left["frame_id"],plan["design"]["response_atol"]))
                tables["branch_contrasts"].append(row)
            focus=dict(case_id=cid,sequence=case["sequence"],frame=case["frame"],seed=seed)
            # Two accounting paths; head effects in branch means include subsequent own-state evolution.
            for metric,label in (("start_iou","immediate"),("mean_iou","branch_mean"),("endpoint_iou","branch_end")):
                ss=branch_rows[cid,sn,sn][metric]; ps=branch_rows[cid,sn,pn][metric]
                pp=branch_rows[cid,pn,pn][metric]; sp=branch_rows[cid,pn,sn][metric]
                focus[label+"_own_gap"]=pp-ss
                focus[label+"_model_gain_at_supervised_start"]=ps-ss
                focus[label+"_parent_gain_for_preserve"]=pp-ps
                if label=="immediate":
                    focus[label+"_model_gain_at_preserve_start"]=pp-sp
                    focus[label+"_parent_gain_for_supervised"]=sp-ss
                maximum_residual=max(maximum_residual,abs((pp-ss)-(ps-ss)-(pp-ps)),abs((pp-ss)-(pp-sp)-(sp-ss)))
            matched=next(r for r in tables["branch_contrasts"] if r["case_id"]==cid and r["seed"]==seed and r["history"]==sn)
            for key in ("first_box_divergence_frame","first_crop_divergence_frame","first_response_divergence_frame"): focus[key]=matched[key]
            tables["focus"].append(focus)
    if maximum_residual>1e-9: raise ValueError("Parent/model accounting residual changed")
    with writer_lock(root/"analysis"):
        for name,rows in tables.items(): publish_table(root/"analysis"/("stage55_"+name+".csv"),rows)
        c.verify_files(bindings); c.verify_files(plan["source_file_sha256"])
        report=c.seal(dict(schema=c.SCHEMA,status="COMPLETE_MATCHED_PARENT_CONTINUATION_DIAGNOSTIC",
            role="posthoc_consumed_case_parent_head_mechanism_audit",plan_fingerprint=plan["fingerprint"],run_fingerprint=run["fingerprint"],
            cases=len(plan["cases"]),shared_inputs=len(checks),branches=len(branch_checks),branch_horizon=plan["design"]["horizon"],
            full_network_calls=full,head_only_calls=head_only,block_calls=blocks,selector_calls=selectors,total_head_calls=full+head_only,
            **{k:max(r[k] for r in checks) for k in checks[0]},**{k:max(r[k] for r in branch_checks) for k in branch_checks[0]},
            maximum_accounting_residual=maximum_residual,cache_sha256=bindings,table_rows={k:len(v) for k,v in tables.items()},
            table_sha256={"stage55_"+k+".csv":c.file_hash(root/"analysis"/("stage55_"+k+".csv")) for k in tables},
            training_performed=False,weights_frozen=True,new_inference_performed=True,new_method_evaluated=False,adoption_gate=None,
            warnings=["Cases/anchors use consumed Stage-54 outcomes; this is mechanism diagnosis, not independent validation.",
                "Only the first-frame template uses GT. Shared parents are recorded predictions; branch actors use their OWN outputs thereafter.",
                "Fixed-state response equality is a scope check; subsequent response/selector changes can follow different input histories.",
                "Single-frame decompositions are accounting identities. Matched-start multi-frame head effects include downstream state evolution.",
                "Window mean IoU is not full-video AUC. Overlapping frames, branches and seeds are not independent videos.",
                "Local failure/recovery counters reset at the branch boundary and may be left censored; window ends are not observed terminal-video outcomes.",
                "No GT correction, action selection, new online recovery policy, deployment speedup or external generalization is established."],
            next_step="Use common-parent geometry, ranking and matched branch divergence to choose a specific training change; preserve fixed source results"))
        c.publish_identical(root/"analysis/stage55_diagnostic.json",report)
    print("Stage-55: "+report["status"]+"; no training, no method adoption gate")
    for r in tables["focus"]: print("{} seed{} immediate={:+.4f} branch_mean={:+.4f}".format(r["case_id"],r["seed"],r["immediate_own_gap"],r["branch_mean_own_gap"]))
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--input",required=True,type=nonempty_path)
    summarize(p.parse_args().input)
