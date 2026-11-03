"""Cached veto exposure and conditional allow-minus-veto continuation effects."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import stage67_veto_common as c
import stage67_veto_experiment as e
from stage27_uav_common import tracking_iou, failure_events, nonempty_path
from stage46_shift_audit import publish_table
from stage27_uav_collect import writer_lock


def first_true(mask, frames):
    found = np.flatnonzero(mask)
    return int(frames[found[0]]) if len(found) else None


def measures(values, frames):
    return dict(anchor_iou=float(values[0]),mean_iou=float(values.mean()),successor_mean_iou=float(values[1:].mean()),
        endpoint_iou=float(values[-1]),low_iou_frames=int((values<.2).sum()),good_iou_frames=int((values>=.5).sum()),
        first_three_good_frame=first_true(np.convolve((values>=.5).astype(int),np.ones(3,int),"valid")==3,frames)
            if len(values)>=3 else None)


def cached_audit(source, plan):
    rows, divergences, detail = [], [], []
    for entry in source["plan"]["entries"]:
        seq = entry["sequence"]; traces = {}
        gt = np.asarray(entry["ground_truth"])[1:]
        for p in c.prior.POLICIES:
            data,_,_ = c.read_unit(source,seq,p); traces[p] = data
            y = tracking_iou(data["candidate_boxes"],gt[:,None]); q = data["quality_choice"].astype(int)
            qy = y[np.arange(len(y)),q]; veto = data["veto"].astype(bool); changed = c.changed_veto(data)
            rows.append(dict(sequence=seq,policy=p,frame_states=len(y),veto_frames=int(veto.sum()),changed_box_veto_frames=int(changed.sum()),
                first_changed_veto=first_true(changed,data["frame_id"]),
                veto_q0_good_proposal_low=int((veto & (y[:,0]>=.5) & (qy<.2)).sum()),
                veto_q0_low_proposal_good=int((veto & (y[:,0]<.2) & (qy>=.5)).sum()),
                mean_immediate_allow_gain_at_veto=float((qy-y[:,0])[veto].mean()) if veto.any() else None,
                new_network_calls=0,new_image_reads=0))
        p,q = traces["protect"],traces["quality"]; frames = p["frame_id"]
        boxdiff = np.any(p["boxes"] != q["boxes"],axis=1)
        cropdiff = np.any(p["crop"] != q["crop"],axis=tuple(range(1,p["crop"].ndim)))
        first = first_true(boxdiff,frames)
        divergences.append(dict(sequence=seq,first_output_divergence=first,first_crop_divergence=first_true(cropdiff,frames),
            first_selected_index_divergence=first_true(p["selected"]!=q["selected"],frames),
            protect_changed_veto_at_first_output=None if first is None else bool(c.changed_veto(p)[first-1]),
            common_candidate_boxes_at_first_output=None if first is None else bool(np.array_equal(p["candidate_boxes"][first-1],q["candidate_boxes"][first-1]))))
        relevant = [x for x in plan["cases"] if x["sequence"]==seq]
        for case in relevant:
            start=max(1,case["frame"]-8); end=min(len(frames),case["frame"]+plan["design"]["horizon"]-1)
            y=tracking_iou(p["candidate_boxes"],gt[:,None])
            for i in range(start-1,end):
                choice=int(p["quality_choice"][i])
                detail.append(dict(case_id=case["case_id"],sequence=seq,frame=int(frames[i]),selected=int(p["selected"][i]),
                    quality_choice=choice,q0_pred_iou=float(p["q0_pred_iou"][i]),quality_gain=float(p["quality_gain"][i]),
                    veto=bool(p["veto"][i]),changed_box_veto=bool(c.changed_veto(p)[i]),
                    q0_iou=float(y[i,0]),proposed_iou=float(y[i,choice]),**{"candidate_"+str(j)+"_iou":float(y[i,j]) for j in range(4)}))
    return rows,divergences,detail


def summarize(root):
    root=Path(root).resolve();plan,source=c.load_plan(root);run=e.completed(plan,root)
    cached,divergences,detail=cached_audit(source,plan)
    branches,contrasts,scope,timeline,focus=[],[],[],[],[]
    for case in plan["cases"]:
        trace,entry,_=c.read_unit(source,case["sequence"],"protect");quality,_,_=c.read_unit(source,case["sequence"],"quality")
        pack=source["models"]["models"][str(case["unit"]["fold"])]
        shared=c.read_cache(c.path_for(root,case),c.metadata(plan,case))
        e.audit_shared(shared,trace,entry,case,plan,pack)
        frame=case["frame"];horizon=plan["design"]["horizon"];frames=np.arange(frame,frame+horizon)
        gt=np.asarray(entry["ground_truth"])[frames]
        common=np.array_equal(shared["initial_history"],e.e63.history_at(quality,entry["initial_box"],frame,plan))
        lookup={};quality_diagonal=False
        scope.append(dict(case_id=case["case_id"],kind="shared",controller="protect",action="source",frames=1,
            search_calls=int(shared["head_calls"].sum()),block_calls=int(shared["block_calls"].sum()),source_diagonal=True))
        for controller in c.CONTROLLERS:
            for action in c.ACTIONS:
                data=c.read_cache(c.path_for(root,case,controller,action),c.metadata(plan,case,controller,action))
                e.audit_tail(data,shared,trace,case,controller,action,plan,pack)
                forced=e.forced_choice(case,action)
                boxes=np.concatenate([shared["candidate_boxes"][:,forced],data["boxes"]]);values=tracking_iou(boxes,gt)
                m=measures(values,frames);lookup[controller,action]=m
                diagonal=(controller,action)==("protect","veto")
                if common and (controller,action)==("quality","allow"):
                    np.testing.assert_array_equal(shared["candidate_boxes"][0],quality["candidate_boxes"][frame-1])
                    if forced != int(quality["selected"][frame-1]):raise ValueError("Common-history quality anchor differs")
                    e.exact_source(data,quality,frame+1);diagonal=quality_diagonal=True
                branches.append(dict(case_id=case["case_id"],sequence=case["sequence"],frame=frame,controller=controller,action=action,
                    forced_choice=forced,common_quality_incoming_history=bool(common),**m))
                scope.append(dict(case_id=case["case_id"],kind="tail",controller=controller,action=action,frames=len(data["frame_id"]),
                    search_calls=int(data["head_calls"].sum()),block_calls=int(data["block_calls"].sum()),source_diagonal=diagonal))
                for i,f in enumerate(frames):timeline.append(dict(case_id=case["case_id"],sequence=case["sequence"],frame=int(f),
                    controller=controller,action=action,forced_anchor=i==0,iou=float(values[i]),
                    selected=forced if i==0 else int(data["selected"][i-1])))
        metrics=("anchor_iou","mean_iou","successor_mean_iou","endpoint_iou","low_iou_frames","good_iou_frames")
        gains={}
        for controller in c.CONTROLLERS:
            gains[controller]={k:lookup[controller,"allow"][k]-lookup[controller,"veto"][k] for k in metrics}
            contrasts.append(dict(case_id=case["case_id"],sequence=case["sequence"],frame=frame,
                contrast="allow_minus_veto",continuation=controller,**gains[controller]))
        contrasts.append(dict(case_id=case["case_id"],sequence=case["sequence"],frame=frame,
            contrast="allow_effect_quality_minus_protect",continuation="interaction",
            **{k:gains["quality"][k]-gains["protect"][k] for k in metrics}))
        focus.append(dict(case_id=case["case_id"],sequence=case["sequence"],frame=frame,selector=case["selector"],
            common_quality_incoming_history=bool(common),quality_source_diagonal=quality_diagonal,
            immediate_allow_gain=gains["protect"]["anchor_iou"],allow_gain_under_protect=gains["protect"]["mean_iou"],
            allow_gain_under_quality=gains["quality"]["mean_iou"],
            continuation_interaction=gains["quality"]["mean_iou"]-gains["protect"]["mean_iou"]))
    if sum(r["search_calls"] for r in scope)!=plan["formal_search_calls"]:raise ValueError("Summary search budget differs")
    tables=dict(anchors=plan["requests"],case_summary=focus,branch_summary=branches,contrasts=contrasts,scope=scope,
                cached_video=cached,divergences=divergences,veto_frames=detail,timelines=timeline)
    out=root/"analysis";out.mkdir(exist_ok=True)
    with writer_lock(out):
        c.VERIFY(plan["source_file_sha256"]);c.VERIFY(plan["code_sha256"])
        for n,rows in tables.items():publish_table(out/("stage67_"+n+".csv"),rows)
        report=c.seal(dict(schema=c.SCHEMA,status="COMPLETE_MATCHED_VETO_CONTINUATION_DIAGNOSTIC",role=plan["role"],
            plan_fingerprint=plan["fingerprint"],run_fingerprint=run["fingerprint"],source_report_fingerprint=source["report"]["fingerprint"],
            source_training_fingerprint=source["models"]["fingerprint"],runtime=plan["runtime_identity"],
            declared_runtime_differences=plan["declared_runtime_differences"],requested_anchors=len(plan["requests"]),supported_anchors=len(plan["cases"]),
            unsupported_requests=[r for r in plan["requests"] if r["status"]!="supported"],counterfactual_branches=plan["counterfactual_branches"],
            formal_search_calls=plan["formal_search_calls"],formal_block_calls=7*plan["formal_search_calls"],smoke_calls_reused=plan["smoke_calls_reused"],
            additional_smoke_calls=0,maximum_source_box_error=0.,training_performed=False,new_models=0,new_full_trajectories=0,
            original_checking_labels_loaded=False,external_generalization_established=False,novel_method_established=False,
            evidence_sha256=run["cache_sha256"],table_sha256={"stage67_"+n+".csv":c.file_hash(out/("stage67_"+n+".csv")) for n in tables},
            warnings=["Consumed development cases, partly selected near already observed events; no population confidence interval or adoption gate.",
                "Anchor veto compares q0 to the frozen quality proposal, not to an oracle or a refitted scorer.",
                "Four branches share one incoming protect history and four actual anchor searches. Only the anchor choice is forced; tails follow own accepted predictions.",
                "Allow-minus-veto within one continuation controller is a local intervention effect. Controller interaction is a separate factorial effect.",
                "Thirty-two-frame results do not establish avoidance of a later full-video failure. Overlapping anchors are not independent samples.",
                "Cached shadow candidate IoUs describe immediate opportunities; they are not observed counterfactual futures.",
                "SGLA V1 weights/template/selector and both CPU critics remain frozen. GT after initialization enters source anchor selection/evaluation only.",
                "First three good frames is a window-local run, not proof of full-trajectory recovery; event-local rules use source events offline.",
                "All four searches per state are charged; smoke reuses its committed branch. No speedup or model adoption claim."],
            next_step="Use case-level matched continuation effects and common-prefix divergence to define a specific future takeover target before any new fitting"))
        c.publish_identical(out/"stage67_diagnostic.json",report)
    print("Stage-67: "+report["status"]+"; no training or method adoption gate")
    for r in focus:print("{} f{} allow-veto: immediate={:+.4f}, protect={:+.4f}, quality={:+.4f}".format(
        r["case_id"],r["frame"],r["immediate_allow_gain"],r["allow_gain_under_protect"],r["allow_gain_under_quality"]))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",required=True,type=nonempty_path);summarize(p.parse_args().input)
