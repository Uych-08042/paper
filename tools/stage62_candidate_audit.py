"""Audit completed Stage-61 L40 candidate caches, without images, training or inference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import stage62_candidate_common as c
import stage61_l40_experiment as source_experiment
import stage61_search_runtime as source_actor
from stage4_common import file_hash, fingerprint, write_json
from stage27_uav_common import failure_events, nonempty_path, publish_identical
from stage27_uav_collect import writer_lock
from stage46_shift_audit import publish_table
from validation_metrics import tracking_iou, IOU_PROTOCOL

SCHEMA="stage62_cached_decoded_candidate_selection_v1"
FILES=("stage62_candidate_common.py","stage62_candidate_audit.py","test_stage62_candidate_audit.py","run_stage62_candidate_audit.sh")
SOURCE_TABLES={"stage61_l40_"+name+".csv" for name in ("device_bridge","policy_summary","paired","video_summary",
    "seed_video_summary","scope_summary","proposal_summary","video_influence","events")}


def seal(body): return dict(body,fingerprint=fingerprint(body))


def load_source(root):
    root=Path(root).expanduser().resolve()
    plan,source=c.prior.load_plan(root)
    gate=source_experiment.smoke_gate(plan,root); run=source_experiment.completed(plan,root)
    report=c.prior.read_sealed(root/"analysis/stage61_l40_decision.json")
    if (report.get("status") not in ("REVIEW_L40_NATIVE_HISTORY_RECROP_PILOT","PROMISING_L40_NATIVE_HISTORY_RECROP_PILOT") or
        report.get("role")!=plan["role"] or report.get("plan_fingerprint")!=plan["fingerprint"] or
        report.get("run_fingerprint")!=run["fingerprint"] or report.get("comparison_runtime")!=plan["runtime_identity"] or
        report.get("evidence_sha256")!=run["cache_sha256"] or set(report.get("table_sha256",{}))!=SOURCE_TABLES or
        report.get("formal_search_calls")!=plan["formal_search_calls"] or report.get("new_full_trajectories")!=len(plan["units"]) or
        report.get("reused_full_trajectories")!=0 or report.get("training_performed") is not False or
        report.get("sgla_weights_unchanged") is not True or report.get("exact_four_pass_budget") is not True or
        report.get("maximum_public_control_box_error")!=0 or report.get("maximum_decode_error")!=run["maximum_decode_error"]):
        raise ValueError("Need the complete unmixed Stage-61 L40 result")
    bindings=dict(plan["source_file_sha256"])
    bindings.update(run["cache_sha256"]); bindings.update(gate["cache_sha256"])
    bindings.update({str(root/"analysis"/name):sha for name,sha in report["table_sha256"].items()})
    for name in ("manifest.json","run.json","evaluation/run.json","analysis/stage61_l40_decision.json","smoke/stage61_l40_smoke.json"):
        bindings[str(root/name)]=file_hash(root/name)
    for mode in ("smoke","evaluate"):
        path=root/"diagnostics"/("stage61_runtime_"+mode+".json")
        diag=json.loads(path.read_text(encoding="utf-8"))
        if (diag.get("schema")!="stage61_runtime_replay_diagnostic_v1" or diag.get("status")!="EXACT_SOURCE_RUNTIME_VERIFIED" or
            diag.get("mode")!=mode or diag.get("plan_fingerprint")!=plan["fingerprint"] or
            diag.get("verified_runtime_fingerprint")!=fingerprint(plan["runtime_identity"]) or diag.get("mismatches")!=[] or
            diag.get("launcher_sha256")!=plan["code_sha256"]["stage61_runtime_replay.py"]):
            raise ValueError("Need matching L40 startup identities: "+str(path))
        bindings[str(path)]=file_hash(path)
    c.prior.verify_files(bindings)
    return dict(root=root,plan=plan,report=report,run=run,gate=gate,bindings=bindings,
                records=[(r,None) for r in source["records"]])


def read_trace(source,unit):
    plan=source["plan"]; path=source["root"]/"evaluation/traces"/(c.prior.key(unit)+".npz")
    identity=source_experiment.model_identity(plan,source["records"],unit)
    data=c.prior.read_cache(path,dict(c.prior.meta(plan,unit,"full_evaluation"),model_identity=identity))
    entry=next(e for e in plan["entries"] if e["sequence"]==unit["sequence"])
    if data is None or len(data["frame_id"])!=len(entry["frames"])-1: raise ValueError("Missing complete source trace")
    source_actor.audit(data,entry,plan,unit["policy"])
    head=identity.split(":",1)[1] if unit["origin"]=="endpoint" else source["gate"]["observed_head_sha256"]["native_0"]
    if source["run"]["observed_head_sha256"][c.prior.key(unit)]!=head: raise ValueError("Wrong frozen source head")
    return data,entry,path,head


def transition_row(metrics,events):
    row={"first_"+name:first for name in ("available_selected_miss","q0_good_lost","q0_good_to_low","no_good_decoded","all_decoded_low","selected_low")
         for first in (c.first_frame(metrics[name]),)}
    row["first_all_gt_centers_outside"]=c.first_frame(~metrics["any_gt_center_inside"])
    event=events[0] if events else None
    row.update(first_failure_onset=None if event is None else event["start"]+1,
        first_failure_confirmation=None if event is None else event["confirmation"]+1,
        first_recovery=None if event is None or event["recovery"] is None else event["recovery"]+1,
        failure_events=len(events),terminal_unrecovered_events=sum(e["censor"]=="video_end" for e in events))
    if event is not None:
        before=np.arange(len(metrics["selected_iou"]))<event["start"]
        row["first_q0_good_to_low_before_failure"]=c.first_frame(metrics["q0_good_to_low"]&before)
        row["first_available_miss_before_failure"]=c.first_frame(metrics["available_selected_miss"]&before)
    else: row.update(first_q0_good_to_low_before_failure=None,first_available_miss_before_failure=None)
    return row


def scalar_row(metrics,index):
    return {name:np.asarray(value)[index].item() for name,value in metrics.items()}


def focus_rows(spec,unit,data,metrics,detail,feature_names):
    frame=spec["frame"]; count=len(metrics["frame_id"])
    if not 1<=frame<=count: raise ValueError("Focus anchor outside trace")
    first=max(1,frame-c.DESIGN["before"]); last=min(count,frame+c.DESIGN["after"])
    common=dict(case_id=spec["case_id"],purpose=spec["purpose"],anchor_frame=frame,
        anchor_policy=spec["anchor_policy"],anchor_kind=spec["anchor_kind"],**unit)
    idx=frame-1
    focus=dict(common,window_first=first,window_last=last,
        **{"at_anchor_"+k:v for k,v in scalar_row(metrics,idx).items()})
    mask=np.zeros(count,bool); mask[first-1:last]=True
    focus.update({"window_"+k:v for k,v in c.summarize_metrics(metrics,mask).items()})
    timeline=[]; candidates=[]
    for i in range(first-1,last):
        timeline.append(dict(common,offset=i+1-frame,**scalar_row(metrics,i)))
        for q,box in enumerate(data["candidate_boxes"][i]):
            features={"feature_"+name:float(data["features"][i,q,j]) for j,name in enumerate(feature_names)}
            parent=data["candidate_parents"][i,q]; crop=data["crop"][i,q]
            row=dict(common,frame=i+1,offset=i+1-frame,candidate=q,selected=bool(q==int(data["selected"][i])),
                iou=float(detail["iou"][i,q]),gt_coverage=float(detail["coverage"][i,q]),gt_center_inside=bool(detail["center_inside"][i,q]),
                parent_x=float(parent[0]),parent_y=float(parent[1]),parent_w=float(parent[2]),parent_h=float(parent[3]),
                output_x=float(box[0]),output_y=float(box[1]),output_w=float(box[2]),output_h=float(box[3]),
                parent_dx_gt_width=float(detail["parent_error"][i,q,0]),parent_dy_gt_height=float(detail["parent_error"][i,q,1]),
                output_dx_gt_width=float(detail["output_error"][i,q,0]),output_dy_gt_height=float(detail["output_error"][i,q,1]),
                width_ratio=float(detail["size_ratio"][i,q,0]),height_ratio=float(detail["size_ratio"][i,q,1]),
                crop_x=float(crop[0]),crop_y=float(crop[1]),crop_side=float(crop[2]),layer=int(data["layer"][i,q]),**features)
            candidates.append(row)
    return focus,timeline,candidates


def audit(source_root,output_root):
    source_root,output_root=Path(source_root).expanduser().resolve(),Path(output_root).expanduser().resolve()
    if source_root==output_root or source_root in output_root.parents or output_root in source_root.parents:
        raise ValueError("Use a separate Stage-62 output directory")
    source=load_source(source_root); old=source["plan"]
    cases=list(c.CASES)
    if len({s["case_id"] for s in cases})!=len(cases): raise ValueError("Duplicate focus case")
    code={str(Path(__file__).with_name(name)):file_hash(Path(__file__).with_name(name)) for name in FILES}
    plan=seal(dict(schema=SCHEMA,role="consumed_development_same_state_decoded_candidate_diagnostic",
        source_input=str(source_root),source_plan_fingerprint=old["fingerprint"],source_run_fingerprint=source["run"]["fingerprint"],
        source_report_fingerprint=source["report"]["fingerprint"],source_file_sha256=source["bindings"],code_sha256=code,
        design=dict(c.DESIGN),cases=cases,cohorts=list(c.COHORTS),iou_protocol=IOU_PROTOCOL,
        unit_definition="one actual accepted prefix and current-frame decoded candidates; no counterfactual continuation"))
    with writer_lock(output_root):
        if not (output_root/"manifest.json").exists() and any(p.name!="collect.lock" for p in output_root.iterdir()):
            raise ValueError("Use an empty Stage-62 directory")
        publish_identical(output_root/"manifest.json",plan)
        print("[source verified] Stage-61 L40 complete; NumPy cache audit only",flush=True)
        native={}
        for unit in old["units"]:
            if unit["origin"]=="native" and unit["policy"]=="current":
                data,entry,_,_=read_trace(source,unit)
                native[unit["sequence"]]=tracking_iou(data["boxes"],np.asarray(entry["ground_truth"])[1:])
        seed_rows=[]; scope=[]; transitions=[]; focus=[]; timeline=[]; candidates=[]; checked_cases=set()
        cached_calls=0; state_count=0
        for unit in old["units"]:
            data,entry,path,head=read_trace(source,unit)
            metrics,detail=c.frame_metrics(data,entry); masks,events=c.cohort_masks(metrics,native[unit["sequence"]])
            label=dict(**unit,arm=unit["origin"]+"_"+unit["policy"])
            for cohort in c.COHORTS:
                seed_rows.append(dict(label,cohort=cohort,**c.summarize_metrics(metrics,masks[cohort])))
            calls=int(data["head_calls"].sum()); count=len(data["frame_id"])
            cached_calls+=calls; state_count+=count
            scope.append(dict(label,frame_states=count,decoded_candidates=int(data["candidate_boxes"].shape[1]),
                cached_source_search_calls=calls,maximum_cached_decode_error=float(data["decode_error"].max()),
                head_sha256=head,cache_sha256=file_hash(path),receipt_sha256=file_hash(path.with_suffix(".receipt.json")),
                new_network_calls=0,new_image_reads=0,training_performed=False))
            transitions.append(dict(label,**transition_row(metrics,events)))
            for spec in cases:
                if (spec["sequence"],spec["origin"],spec["seed"])!=(unit["sequence"],unit["origin"],unit["seed"]): continue
                if unit["policy"]==spec["anchor_policy"]:
                    if not events or spec["anchor_kind"] not in ("onset","recovery"): raise ValueError("Missing source focus event")
                    e=events[0]; value=e["start"] if spec["anchor_kind"]=="onset" else e["recovery"]
                    if value is None or value+1!=spec["frame"]: raise ValueError("Fixed focus anchor differs: "+spec["case_id"])
                    checked_cases.add(spec["case_id"])
                row,times,options=focus_rows(spec,unit,data,metrics,detail,c.prior.original.prior.FEATURES)
                focus.append(row); timeline.extend(times); candidates.extend(options)
            print("[cached audited] "+c.prior.key(unit),flush=True)
        if cached_calls!=old["formal_search_calls"] or len(scope)!=old["new_full_trajectories"]:
            raise ValueError("Cached source exposure differs from completed run")
        if checked_cases!={s["case_id"] for s in cases} or len(focus)!=len(cases)*len(c.prior.POLICIES):
            raise ValueError("Missing a predeclared focus policy or anchor")
        videos=c.aggregate(seed_rows,("sequence","origin","policy","arm","cohort"),"seeds")
        policies=c.aggregate(videos,("origin","policy","arm","cohort"),"videos")
        order={p:i for i,p in enumerate(c.prior.POLICIES)}
        focus.sort(key=lambda r:(r["case_id"],order[r["policy"]]))
        timeline.sort(key=lambda r:(r["case_id"],order[r["policy"]],r["frame_id"]))
        candidates.sort(key=lambda r:(r["case_id"],order[r["policy"]],r["frame"],r["candidate"]))
        tables=dict(policy_summary=policies,video_summary=videos,seed_summary=seed_rows,scope=scope,
                    transitions=transitions,focus=focus,timeline=timeline,candidates=candidates)
        c.prior.verify_files(source["bindings"]); c.prior.verify_files(code)
        analysis=output_root/"analysis"; analysis.mkdir(exist_ok=True)
        with writer_lock(analysis):
            for name,rows in tables.items():
                keys=list(dict.fromkeys(k for r in rows for k in r))
                publish_table(analysis/("stage62_"+name+".csv"),[{k:r.get(k) for k in keys} for r in rows])
            all_policies=[{k:r[k] for k in ("arm","videos","frame_states_total","has_good_decoded_fraction", "available_selected_miss_fraction",
                "miss_given_good_decoded","q0_good_to_low_fraction","rescued_fraction","mean_decoded_regret")} for r in policies if r["cohort"]=="all"]
            report=seal(dict(schema=SCHEMA,status="COMPLETE_CACHED_CANDIDATE_SELECTION_DIAGNOSTIC",role=plan["role"],
                plan_fingerprint=plan["fingerprint"],source_plan_fingerprint=old["fingerprint"],
                source_report_fingerprint=source["report"]["fingerprint"],source_bindings_fingerprint=fingerprint(source["bindings"]),
                source_status=source["report"]["status"],source_runtime=old["runtime_identity"],source_trajectories=len(scope),
                videos=len(old["entries"]),endpoint_seeds=old["seeds"],cached_frame_states=state_count,cached_decoded_candidates=cached_calls,
                source_search_calls_already_paid=cached_calls,new_network_calls=0,new_full_trajectories=0,new_image_reads=0,
                training_performed=False,new_models=0,offline_development_gt_loaded=True,original_checking_labels_loaded=False,
                full_response_grids_available=False,counterfactual_continuations_performed=False,novel_method_established=False,
                all_policy_summary=all_policies,focus_cases=len(cases),focus_rows=len(focus),focus_timeline_rows=len(timeline),
                focus_candidate_rows=len(candidates),table_sha256={"stage62_"+n+".csv":file_hash(analysis/("stage62_"+n+".csv")) for n in tables},
                warnings=["All twelve source groups and focus outcomes are consumed development evidence; no method adoption test.",
                    "Decoded candidate availability is not dense-grid availability or guaranteed future recovery.",
                    "GT-best instantaneous IoU is conditional on each actual prefix; no oracle tracking trajectory or policy AUC is constructed.",
                    "Each policy owns its incoming history; same source-frame comparisons across policies are not matched-start causal effects.",
                    "Supported seeds average within a video, then supported videos average equally; empty conditional denominators remain undefined and support is explicit.",
                    "Own-outcome cohorts and overlapping focus windows are offline descriptions, not independent samples or online triggers.",
                    "Cross-crop peak reliability is a hypothesis; this audit fits no calibration, scorer, threshold or controller."],
                next_step="Inspect pre-failure availability and wrong selection in existing cache; predeclare any later matched-start continuation or controller test"))
            c.prior.verify_files(source["bindings"]); c.prior.verify_files(code)
            publish_identical(analysis/"stage62_diagnostic.json",report)
        print("Stage-62: "+report["status"]+"; zero new training, image reads or GPU inference")
        print("Reports: "+str(analysis))
        return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True,type=nonempty_path)
    parser.add_argument("--output_dir",required=True,type=nonempty_path)
    args=parser.parse_args(); audit(args.input,args.output_dir)
