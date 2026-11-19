"""Locate the first layout, selection and feedback changes in Stage-131 histories."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

import stage131_zoom_closed_loop as source
from stage27_uav_common import scene_geometry, success_auc, tracking_iou

prior, base, c, io = source.prior, source.base, source.c, source.io
SCHEMA = "stage132_first_search_divergence_v1"
STATUS = "COMPLETE_CACHED_FIRST_DIVERGENCE_AUDIT"
SOURCE131 = "eaea870fdba9b820dd02f414f757d8f161386b4d27c9224cfc08b6f70db23d38"
PAIRS = (("alternate_shared","trim"),("trim","rank1"))
FOCUS = {"uav0000070","uav0000076","uav0000085","uav0000174","uav0000176"}
DESIGN = dict(expected_pairs=112, good=.5, score_atol=1e-12, history_length=5)
TABLES = ("trajectories","first_events","summary","recovery")
WORK = dict(new_search_calls=0,image_reads=0,training_updates=0,new_full_trajectories=0)


def changed_rows(a,b):
    a,b = np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or not a.ndim:
        raise ValueError("Paired arrays do not align")
    return np.any(a!=b,axis=tuple(range(1,a.ndim)))


def same_box_history(box_change, length):
    # On tracked frame i the preceding length frames exclude frame i itself.
    padded = np.pad(box_change[:-1],(length,0),constant_values=False)
    return ~np.lib.stride_tricks.sliding_window_view(padded,length).any(axis=1)


def next_request(trace):
    return np.array([source.s121.request_alternate(dict(selected=q,candidate_boxes=b,scores=s))
                     for q,b,s in zip(trace["selected"],trace["candidate_boxes"],trace["scores"])],bool)


def compare(a,b,ref_arm,history_length):
    n = len(a["frame_id"])
    if n<1:
        raise ValueError("Empty trajectory")
    for d in (a,b):
        np.testing.assert_array_equal(d["frame_id"],np.arange(1,n+1))
    np.testing.assert_array_equal(a["image_hw"],b["image_hw"])
    req_a,req_b = next_request(a),next_request(b)
    for trace,req in ((a,req_a),(b,req_b)):
        np.testing.assert_array_equal(trace["alternate_requested"][1:],req[:-1])
    box = changed_rows(a["boxes"],b["boxes"])
    geometry_changed = changed_rows(a["crop"],b["crop"])
    parents_changed = changed_rows(a["candidate_parents"],b["candidate_parents"])
    forecast = changed_rows(a["forecast_x"],b["forecast_x"])
    common = (same_box_history(box,history_length) & ~forecast
              & ~changed_rows(a["alternate_requested"],b["alternate_requested"])
              & ~changed_rows(a["alternate_used"],b["alternate_used"])
              & ~changed_rows(a["candidate_parents"] if ref_arm=="alternate_shared" else a["untrimmed_parents"],
                             b["untrimmed_parents"]))
    if ref_arm=="trim":
        common &= ~changed_rows(a["candidate_parents"],b["trimmed_parents"])
    following_forecast = np.r_[forecast[1:],False]
    request = req_a!=req_b
    request[-1] = False  # No successor at the last frame.
    carry = req_a & req_b & changed_rows(a["candidate_boxes"][:,0],b["candidate_boxes"][:,0])
    carry[-1] = False
    next_state = following_forecast | request | carry | np.r_[box[:-1],False]
    masks = dict(layout=geometry_changed,parents=parents_changed,slot=a["selected"]!=b["selected"],output=box,
                 request=request,carry=carry,next_forecast=following_forecast,next_state=next_state,
                 incoming_state=np.r_[False,next_state[:-1]],matched_crop_change=common & geometry_changed)
    return common,masks,req_a,req_b


def winner(scores, boxes, slots):
    if not len(slots):
        return None
    best = max(scores[q] for q in slots)
    tied = [int(q) for q in slots if scores[q]>=best-base.common.DESIGN["tie_atol"]]
    return 0 if 0 in tied else min(tied,key=lambda q:tuple(boxes[q]))


def measure(a,b,i,ga,gb,gt,ap,common,masks):
    converted = (a,b)
    parents = [d["candidate_parents"][i] for d in converted]
    boxes = [d["candidate_boxes"][i] for d in converted]
    scores = [d["scores"][i] for d in converted]
    selected = [int(d["selected"][i]) for d in converted]
    changed_crops = np.flatnonzero((a["crop"][i]!=b["crop"][i]).any(axis=1))
    retained = np.flatnonzero((parents[0]==parents[1]).all(axis=1)&(boxes[0]==boxes[1]).all(axis=1)
        &(a["features"][i]==b["features"][i]).all(axis=1)&(a["crop"][i]==b["crop"][i]).all(axis=1))
    wins = [winner(s,v,retained) for s,v in zip(scores,boxes)]
    if common:
        for j,d in enumerate(converted):
            if base.common.physical_choice(scores[j],boxes[j])!=int(d["selected"][i]):
                raise ValueError("Cached scorer ordering differs from the selected slot")
    values = [ga[i],gb[i]]
    best = [float(v.max()) for v in values]
    outputs = [float(v[q]) for v,q in zip(values,selected)]
    good = DESIGN["good"]
    if not common:
        mechanism = "history_or_proposal_already_different"
    elif best[0]>=good>best[1]:
        mechanism = "good_candidate_lost"
    elif outputs[0]>=good>outputs[1]:
        mechanism = "good_candidate_missed"
    elif not masks["output"][i] and masks["request"][i]:
        mechanism = "same_box_request_changed"
    elif not masks["output"][i] and masks["next_state"][i]:
        mechanism = "same_box_other_feedback_changed"
    elif best[1]>=good>best[0]:
        mechanism = "good_candidate_gained"
    elif outputs[1]>=good>outputs[0]:
        mechanism = "good_selection_gained"
    elif masks["output"][i]:
        mechanism = "output_changed_without_good_threshold_crossing"
    else:
        mechanism = "no_measured_output_or_feedback_change"
    coverage = [[scene_geometry(p,gt,d["image_hw"][i],ap["search_factor"],ap["search_size"])["coverage"]
                 for p in parent] for d,parent in zip(converted,parents)]
    vec = lambda x:"|".join(map(str,x))
    zoom_slot = int(b["zoom_slot"][i]) if "zoom_slot" in b else -1
    return dict(mechanism=mechanism,matched_before_search=bool(common),output_changed=bool(masks["output"][i]),
                request_changed=bool(masks["request"][i]),next_forecast_changed=bool(masks["next_forecast"][i]),
                changed_crop_slots=vec(changed_crops),treatment_zoom_slot=zoom_slot if zoom_slot>=0 else None,
                retained_slots=vec(retained),retained_top_reference=wins[0],retained_top_treatment=wins[1],
                retained_reranked=wins[0]!=wins[1],
                reference_slot=selected[0],treatment_slot=selected[1],
                reference_best_iou=best[0],treatment_best_iou=best[1],
                reference_iou=outputs[0],treatment_iou=outputs[1],
                reference_candidate_ious=vec(values[0]),treatment_candidate_ious=vec(values[1]),
                reference_scores=vec(scores[0]),treatment_scores=vec(scores[1]),
                reference_coverage=vec(coverage[0]),treatment_coverage=vec(coverage[1]))


def first(mask):
    indices = np.flatnonzero(mask)
    return int(indices[0])+1 if len(indices) else None


def diagnose(a,b,entry,ap,identity,reported_auc):
    common,masks,ra,rb = compare(a,b,identity["reference_arm"],ap["design"]["history"])
    gt = np.asarray(entry["ground_truth"],float)[1:]
    aa,bb = (tracking_iou(d["candidate_boxes"],gt[:,None,:]) for d in (a,b))
    if aa.shape!=bb.shape or aa.shape!=(len(gt),4) or (aa<0).any() or (bb<0).any():
        raise ValueError("Invalid four-candidate geometry")
    selected_a = aa[np.arange(len(gt)),a["selected"]]
    selected_b = bb[np.arange(len(gt)),b["selected"]]
    np.testing.assert_allclose([success_auc(selected_a),success_auc(selected_b)],reported_auc,rtol=0,atol=1e-12)
    good = DESIGN["good"]
    masks.update(good_supply_lost=(aa.max(axis=1)>=good)&(bb.max(axis=1)<good),
                 good_selection_lost=(selected_a>=good)&(selected_b<good),
                 missed_good=(selected_a>=good)&(selected_b<good)&(bb.max(axis=1)>=good))
    for name in ("good_supply_lost","missed_good","request","output","next_state"):
        masks["matched_"+name] = masks[name]&common
    masks["score"] = np.max(np.abs(a["scores"]-b["scores"]),axis=1)>DESIGN["score_atol"]
    firsts = {name:first(mask) for name,mask in masks.items()}
    recovery = base.recovery_rows(selected_a,selected_b,identity,identity["reference_arm"])
    roles = {}
    for name,frame in firsts.items():
        if frame is not None:
            roles.setdefault(frame,[]).append(name)
    for r in recovery:
        if r["matched_good_frame"] is None:
            roles.setdefault(r["reference_recovery_frame"],[]).append("unmatched_recovery")
    events = []
    for frame,names in sorted(roles.items()):
        i = frame-1
        has_successor = frame<len(gt)
        events.append(dict(identity,frame=frame,roles="|".join(names),
            next_reference_requested=bool(ra[i]) if has_successor else None,
            next_treatment_requested=bool(rb[i]) if has_successor else None,
            next_reference_used=bool(a["alternate_used"][i+1]) if has_successor else None,
            next_treatment_used=bool(b["alternate_used"][i+1]) if has_successor else None,
            **measure(a,b,i,aa,bb,gt[i],ap,common[i],masks)))
    summary = dict(identity,frames=len(gt),matched_frames=int(common.sum()),
        matched_crop_changes=int(masks["matched_crop_change"].sum()),
        matched_good_supply_losses=int(masks["matched_good_supply_lost"].sum()),
        matched_good_misses=int(masks["matched_missed_good"].sum()),
        matched_same_output_request_changes=int((masks["matched_request"]&~masks["output"]).sum()),
        reference_auc=reported_auc[0],treatment_auc=reported_auc[1],auc_delta=reported_auc[1]-reported_auc[0],
        lost_recoveries=sum(r["matched_good_frame"] is None for r in recovery),
        **{"first_"+k+"_frame":v for k,v in firsts.items()})
    return summary,events,recovery


def aggregate(rows):
    summary = []
    for reference,treatment in PAIRS:
        part = [r for r in rows if r["reference_arm"]==reference]
        summary.append(dict(reference_arm=reference,treatment_arm=treatment,pairs=len(part),
            matched_crop_changes=sum(r["matched_crop_changes"] for r in part),
            matched_good_supply_losses=sum(r["matched_good_supply_losses"] for r in part),
            matched_good_misses=sum(r["matched_good_misses"] for r in part),
            matched_same_output_request_changes=sum(r["matched_same_output_request_changes"] for r in part),
            lost_recoveries=sum(r["lost_recoveries"] for r in part),
            pairs_with_output_difference=sum(r["first_output_frame"] is not None for r in part)))
    return summary


def check_results(root):
    p = c.c66.sealed(root/"manifest.json",SCHEMA)
    d = c.c66.sealed(root/"stage132_decision.json",SCHEMA)
    if (p["design"]!=DESIGN or d["plan_fingerprint"]!=p["fingerprint"] or d["status"]!=STATUS
            or d["source_stage131_fingerprint"]!=SOURCE131 or d["pairs"]!=DESIGN["expected_pairs"]
            or set(d["table_sha256"])!={f"stage132_{name}.csv" for name in TABLES}
            or any(d[k]!=v for k,v in WORK.items())):
        raise ValueError("Incomplete first-divergence audit")
    io.verify({str(root/name):digest for name,digest in d["table_sha256"].items()})


def run(root,r121,p121,r131,p131,ap,load_entry,plan):
    tr,events,recovery = [],[],[]
    index = {(r["sequence"],r["arm"],int(r["seed"])):r for r in
             prior.source.read_table(r131/"analysis/stage131_seed_video.csv")}
    jobs = {(j["spec"]["unit"]["sequence"],j["arm"],j["seed"]):j for j in p131["jobs"] if j["arm"]!="native"}
    archived = prior.source.read_table(r131/"analysis/stage131_recovery.csv")
    entries = {}
    for sequence,arm,seed in sorted(jobs):
        if arm!=source.s121.TREATMENT:
            continue
        j = jobs[sequence,arm,seed]
        if sequence not in entries:
            entries[sequence] = load_entry(j["spec"])
        entry = entries[sequence]
        if entry["sequence"]!=sequence or len(entry["ground_truth"])!=j["frames"]+1:
            raise ValueError("Source video or frame count changed")
        traces = {arm:prior.prior.load_trace(r121,p121,j)}
        for mode in ("trim","rank1"):
            j_mode = jobs[sequence,mode,seed]
            path = r131/"evaluation"/(base.job_key(j_mode)+".npz")
            trace = c.read_cache(path,dict(plan_fingerprint=p131["fingerprint"],job=j_mode))
            if trace is None:
                raise ValueError(f"Missing Stage-131 trace: {path}")
            traces[mode] = trace
        for reference,treatment in PAIRS:
            ident = dict(sequence=sequence,seed=seed,reference_arm=reference,treatment_arm=treatment,
                         focus=base.s109.s106.source_group(sequence) in FOCUS)
            auc = [float(index[sequence,mode,seed]["auc"]) for mode in (reference,treatment)]
            actual,frames,local = diagnose(traces[reference],traces[treatment],entry,ap,ident,auc)
            tr.append(actual)
            events.extend(frames)
            recovery.extend(local)
        print(f"[Stage-132] {sequence} s{seed}: both pairs",flush=True)
    recovery_key = lambda r:(r["sequence"],int(r["seed"]),int(r["reference_failure_frame"]),
        int(r["reference_recovery_frame"]),int(r["matched_good_frame"]) if r["matched_good_frame"] not in ("",None) else None)
    assert_source = [r for r in recovery if r["reference_arm"]=="trim"]
    assert_archive = [r for r in archived if r["reference_arm"]=="trim"]
    if len(tr)!=DESIGN["expected_pairs"] or Counter(map(recovery_key,assert_source))!=Counter(map(recovery_key,assert_archive)):
        raise ValueError("Source Stage-131 trajectory or recovery inventory changed")
    summary = aggregate(tr)
    hashes = {}
    for name,rows in dict(trajectories=tr,first_events=events,summary=summary,recovery=recovery).items():
        path = root/f"stage132_{name}.csv"
        base.s103.s86.publish_table(path,rows)
        hashes[path.name] = c.file_hash(path)
    c.publish_identical(root/"stage132_decision.json",io.seal(dict(schema=SCHEMA,status=STATUS,
        plan_fingerprint=plan["fingerprint"],source_stage131_fingerprint=SOURCE131,pairs=len(tr),
        event_rows=len(events),summary=summary,table_sha256=hashes,**WORK)))
    check_results(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76","stage89","stage93","stage121","stage131")
    for name in names:
        parser.add_argument("--"+name,type=Path)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--check-results",action="store_true")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not all(getattr(args,name) for name in names):
        parser.error("Supply the frozen Stage-76/89/93/121/131 directories")
    r76,r89,r93,r121,r131 = dirs = [getattr(args,k).expanduser().resolve() for k in names]
    if any(root==p or root in p.parents or p in root.parents for p in dirs):
        parser.error("Use a separate Stage-132 directory")
    if not (r131/"analysis/stage131_decision.json").is_file():
        parser.error("Need the Stage-131 server trajectories")
    with base.s109.s106.s93.writer_lock(root):
        if (root/"stage132_decision.json").is_file():
            check_results(root)
            return
        source.check_results(r131)
        p131 = c.c66.sealed(r131/"manifest.json",source.SCHEMA)
        d131 = c.c66.sealed(r131/"analysis/stage131_decision.json",source.SCHEMA)
        p121 = c.c66.sealed(r121/"manifest.json",source.s121.SCHEMA)
        d121 = c.c66.sealed(r121/"analysis/stage121_decision.json",source.s121.SCHEMA)
        if (d131["fingerprint"]!=SOURCE131 or d121["fingerprint"]!=source.SOURCE121
                or d121["plan_fingerprint"]!=p121["fingerprint"] or not p131["reuse_controls"]
                or p131["runtime"]!=p121["runtime"] or p131["jobs"]!=source.source_jobs(p121)):
            raise ValueError("Need the original same-device Stage-131 and Stage-121 trajectories")
        paths = [f"evaluation/{base.job_key(j)}{suffix}" for j in p121["jobs"] if j["arm"]==source.s121.TREATMENT
                 for suffix in (".npz",".receipt.json")]
        io.verify({str(r121/p):d121["artifact_sha256"][p] for p in paths})
        s95 = base.s103.prior.s95
        p93,_,p89,report89,origin = s95.s94.load_run(r93,r76,r89)
        def load_entry(spec):
            return s95.load_unit(r93,r89,p93,p89,report89,origin,spec)[0]

        ap = base.e63.c.actor_plan(origin["original"])
        if ap["design"]["history"]!=DESIGN["history_length"]:
            raise ValueError("Unexpected source history length")
        plan = io.seal(dict(schema=SCHEMA,design=DESIGN,source_stage131_fingerprint=SOURCE131,
                            source_stage121_fingerprint=source.SOURCE121,code_sha256=c.file_hash(Path(__file__))))
        c.publish_identical(root/"manifest.json",plan)
        run(root,r121,p121,r131,p131,ap,load_entry,plan)


if __name__=="__main__":
    main()
