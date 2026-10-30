"""Conditional center/size/head effects in matched 32-frame continuations."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from stage27_uav_collect import writer_lock
from stage27_uav_common import nonempty_path,scene_geometry,tracking_iou
from stage46_shift_audit import publish_table
from stage28_response_common import dense_boxes,response_metrics
from stage55_matched_summary import frame_measures,first_difference
import stage59_parent_common as c


def analyze_point(data,item,inputs):
    source=inputs["plan"]; entry,_=inputs["arrays"][item["sequence"]]; gt=np.asarray(entry["ground_truth"][item["frame"]],float)
    parent=data["parent"]; factor=float(data["factor"])
    matrix=dict(**item,coverage=float(scene_geometry(parent,gt,data["image_hw"],source["search_factor"],source["search_size"])["coverage"]),
        parent_center_error_gt_units=frame_measures(parent,parent,gt,data["image_hw"],source)["center_error_gt_units"],
        parent_width=float(parent[2]),parent_height=float(parent[3]),rounded_crop_side=float(source["search_size"]/factor),
        gt_width_search_px=float(gt[2]*factor),gt_height_search_px=float(gt[3]*factor),layer=int(data["layer"]))
    rows=[]
    for i,name in enumerate(c.point_heads(item)):
        raw=data["raw"][i]; wi=int(np.argmax(raw*np.asarray(source["window"],np.float32))); ri=int(np.argmax(raw))
        dense=dense_boxes(data["size"][i],data["offset"][i],parent,factor,source["search_size"],data["image_hw"])
        dense[wi]=data["boxes"][i]; dense[ri]=data["raw_boxes"][i]
        metrics,_,_=response_metrics(raw,np.asarray(source["window"],np.float32),dense,gt,native_box=data["boxes"][i])
        row=dict(**item,evaluator=name,layer=int(data["layer"]),raw_cell=ri,hann_cell=wi,
            **frame_measures(data["boxes"][i],parent,gt,data["image_hw"],source),
            dense_best_gt_iou=metrics["dense_best_gt_iou"],weighted_top5_gt_recall=metrics["weighted_top5_gt_recall"],
            weighted_gt_rank_in_top5=metrics["weighted_gt_rank_in_top5"],weighted_gt_score_margin=metrics["weighted_gt_score_margin"])
        rows.append(row)
        matrix[("native" if name=="native" else "E" if name.startswith(c.ARMS[0]+"_") else "P")+"_iou"]=row["iou"]
    return rows,matrix


def analyze_branch(data,item,inputs):
    from stage55_matched_summary import analyze_branch as previous
    return previous(data,item,inputs)


def factor_effects(values):
    """values[C,S,H]; effects are conditional on one anchor/seed, no pooling CI."""
    y=np.asarray(values,float)
    if y.shape!=(2,2,2) or not np.isfinite(y).all(): raise ValueError("Need a finite complete 2x2x2 factorial")
    head=float(y[0,0,1]-y[0,0,0]); center=float(y[1,0,1]-y[0,0,1]); size=float(y[1,1,1]-y[1,0,1])
    size_first=float(y[0,1,1]-y[0,0,1]); center_last=float(y[1,1,1]-y[0,1,1]); gap=float(y[1,1,1]-y[0,0,0])
    result=dict(own_gap=gap,head_at_EE=head,center_after_head=center,size_after_center=size,
        size_after_head=size_first,center_after_size=center_last,
        center_order_average=float((y[1,:,:]-y[0,:,:]).mean()),size_order_average=float((y[:,1,:]-y[:,0,:]).mean()),
        head_order_average=float((y[:,:,1]-y[:,:,0]).mean()),
        center_size_interaction=float((y[1,1,:]-y[1,0,:]-y[0,1,:]+y[0,0,:]).mean()),
        center_head_interaction=float((y[1,:,1]-y[1,:,0]-y[0,:,1]+y[0,:,0]).mean()),
        size_head_interaction=float((y[:,1,1]-y[:,1,0]-y[:,0,1]+y[:,0,0]).mean()))
    result["maximum_accounting_residual"]=max(abs(gap-head-center-size),abs(gap-head-size_first-center_last))
    if result["maximum_accounting_residual"]>1e-9: raise ValueError("Center/size/head accounting changed")
    return result


def summarize(root):
    root=Path(root).expanduser().resolve(); plan,inputs=c.load_plan(root); run=c.read_sealed(root/"run.json")
    gate=c.read_sealed(root/"smoke/stage59_smoke.json")
    if (run.get("plan_fingerprint")!=plan["fingerprint"] or run.get("role")!="frozen_parent_center_size_factorial" or
        run.get("runtime")!=inputs["plan"]["runtime_identity"] or run.get("arms")!=list(c.ARMS) or
        gate.get("status")!="PASS_PARENT_FACTORIAL_SMOKE" or gate.get("run_fingerprint")!=run["fingerprint"]):
        raise ValueError("Need unchanged run and passing factorial smoke")
    tables=dict(point_details=[],crossed_matrix=[],rollout_summary=[],branch_contrasts=[],focus=[],factor_summary=[],timelines=[])
    bindings={}; checks=[]; branch_checks=[]; point_data={}; branch_data={}; branch_rows={}; missing=[]
    full=head_only=blocks=selectors=0
    for directory,key,items in (("responses","point_id",c.points(plan)),("branches","branch_id",c.branches(plan))):
        wanted={item[key]+".npz" for item in items}
        if {p.name for p in (root/directory).glob("*.npz")}-wanted: raise ValueError("Unexpected diagnostic cache")
    for item in c.points(plan):
        path=c.path_for(root,item); data=c.read_cache(path,c.metadata(run,item))
        if data is None: missing.append(path.name); continue
        checks.append(c.audit_point(data,item,plan,inputs)); details,matrix=analyze_point(data,item,inputs)
        tables["point_details"].extend(details); tables["crossed_matrix"].append(matrix); point_data[item["point_id"]]=data
        full+=1; head_only+=int(data["head_forward_calls"][1:].sum()); blocks+=int(data["full_block_calls"].sum()); selectors+=int(data["selector_calls"])
        for p in (path,path.with_suffix(".receipt.json")): bindings[str(p)]=c.file_hash(p)
    for item in c.branches(plan):
        path=c.path_for(root,item); data=c.read_cache(path,c.metadata(run,item)); start=point_data.get(item["point_id"])
        if data is None or start is None: missing.append(path.name); continue
        branch_checks.append(c.audit_branch(data,item,start,plan,inputs)); row,timeline=analyze_branch(data,item,inputs)
        tables["rollout_summary"].append(row); tables["timelines"].extend(timeline)
        key=(item["case_id"],item["seed"],item["center_donor"],item["size_donor"],item["evaluator"])
        branch_data[key]=data; branch_rows[key]=row
        full+=int(data["head_calls"].sum()); blocks+=int(data["block_calls"].sum()); selectors+=int(data["selector_calls"].sum())
        for p in (path,path.with_suffix(".receipt.json")): bindings[str(p)]=c.file_hash(p)
    if missing:
        print("Stage-59: DIAGNOSTIC_INCOMPLETE; {} missing caches".format(len(missing))); return None
    if (full,head_only,blocks,selectors)!=(plan["expected_full_forwards"],plan["expected_head_only_forwards"],plan["expected_full_forwards"]*7,plan["expected_full_forwards"]):
        raise ValueError("Exact factorial forward budget changed")
    for case in plan["cases"]:
        for seed in c.prior.SEEDS:
            cid=case["case_id"]; focus=dict(case_id=cid,sequence=case["sequence"],frame=case["frame"],seed=seed)
            for metric,label in (("start_iou","immediate"),("mean_iou","branch_mean"),("endpoint_iou","branch_end")):
                y=np.array([branch_rows[cid,seed,center,size,c.evaluator(head,seed)][metric]
                    for center in c.DONORS for size in c.DONORS for head in c.DONORS]).reshape(2,2,2)
                effect=factor_effects(y)
                tables["factor_summary"].append(dict(case_id=cid,sequence=case["sequence"],frame=case["frame"],seed=seed,metric=metric,**effect))
                for k in ("own_gap","head_at_EE","center_after_head","size_after_center","size_after_head","center_after_size"):
                    focus[label+"_"+k]=effect[k]
            tables["focus"].append(focus)
            for center in c.DONORS:
                for size in c.DONORS:
                    key=(cid,seed,center,size); left=branch_rows[key+(c.evaluator("P",seed),)]; right=branch_rows[key+(c.evaluator("E",seed),)]
                    ld=branch_data[key+(c.evaluator("P",seed),)]; rd=branch_data[key+(c.evaluator("E",seed),)]
                    tables["branch_contrasts"].append(dict(case_id=cid,sequence=case["sequence"],frame=case["frame"],seed=seed,center_donor=center,size_donor=size,
                        start_iou_gain=left["start_iou"]-right["start_iou"],mean_iou_gain=left["mean_iou"]-right["mean_iou"],
                        endpoint_iou_gain=left["endpoint_iou"]-right["endpoint_iou"],low_iou_fraction_delta=left["low_iou_fraction"]-right["low_iou_fraction"],
                        first_crop_divergence_frame=first_difference(ld["crop"],rd["crop"],ld["frame_id"],0),
                        first_response_divergence_frame=first_difference(ld["raw"],rd["raw"],ld["frame_id"],plan["design"]["response_atol"])))
    maximum_residual=max(r["maximum_accounting_residual"] for r in tables["factor_summary"])
    # Rounded-input exposure is descriptive, not an adoption gate.
    exposure=[]
    for case in plan["cases"]:
        for seed in c.prior.SEEDS:
            points=[p for p in c.points(plan) if p["case_id"]==case["case_id"] and p["seed"]==seed]
            ds=[point_data[p["point_id"]] for p in points]
            exposure.append(dict(case_id=case["case_id"],seed=seed,
                distinct_parents=len({d["parent"].tobytes() for d in ds}),distinct_rounded_crops=len({d["crop"].tobytes() for d in ds}),
                distinct_crop_pixels=len({str(d["crop_sha256"].item()) for d in ds})))
    tables["exposure"]=exposure
    with writer_lock(root/"analysis"):
        for name,rows in tables.items(): publish_table(root/"analysis"/("stage59_"+name+".csv"),rows)
        c.verify_files(bindings); c.verify_files(plan["source_file_sha256"])
        report=c.seal(dict(schema=c.SCHEMA,status="COMPLETE_PARENT_COMPONENT_CONTINUATION_DIAGNOSTIC",
            role="posthoc_consumed_case_predicted_parent_factorial",plan_fingerprint=plan["fingerprint"],run_fingerprint=run["fingerprint"],
            cases=len(plan["cases"]),shared_inputs=len(checks),branches=len(branch_checks),branch_horizon=plan["design"]["horizon"],
            full_network_calls=full,head_only_calls=head_only,block_calls=blocks,selector_calls=selectors,total_head_calls=full+head_only,
            **{k:max(r[k] for r in checks) for k in checks[0]},**{k:max(r[k] for r in branch_checks) for k in branch_checks[0]},
            maximum_accounting_residual=maximum_residual,cache_sha256=bindings,table_rows={k:len(v) for k,v in tables.items()},
            table_sha256={"stage59_"+k+".csv":c.file_hash(root/"analysis"/("stage59_"+k+".csv")) for k in tables},
            training_performed=False,weights_frozen=True,new_inference_performed=True,new_method_evaluated=False,adoption_gate=None,
            warnings=["Seven anchors use consumed Stage-58 outcomes. No population bootstrap, independent validation or deployable method claim.",
                "Center and size donors are recorded previous predictions from matched seed E/P; no GT hybrid or teacher update after the first step.",
                "Only the initial parent is crossed. Each head then follows its own outputs for 32 frames, without future source states or GT reset.",
                "Size swaps alter search extent/resolution and may alter response ranking; they do not isolate size-regression error alone.",
                "Two component accounting orders are exact conditional identities, not unique causal fractions. Multi-frame head effects include later state evolution.",
                "Full homogeneous diagonals must reproduce source traces; shared-feature first maps and native call budgets are audited.",
                "First-frame GT template only enters actors. Other GT enters offline geometry/ranking and source-event checks, never online selection.",
                "Window IoU is not full-video AUC. Local events may be censored at both branch boundaries.",
                "No new training, online recovery policy, compression gain or deployment speedup is established."],
            next_step="Use center/size/head interactions and conditional continuations to identify a specific state-control or training treatment; preserve fixed cases and source results"))
        c.publish_identical(root/"analysis/stage59_diagnostic.json",report)
    print("Stage-59: "+report["status"]+"; no training or method adoption gate")
    for r in tables["focus"]: print("{} seed{} immediate={:+.4f} branch_mean={:+.4f}".format(r["case_id"],r["seed"],r["immediate_own_gap"],r["branch_mean_own_gap"]))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--input",required=True,type=nonempty_path)
    summarize(p.parse_args().input)
