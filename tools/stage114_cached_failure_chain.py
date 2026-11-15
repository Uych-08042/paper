"""Audit all frozen Stage-113 histories without images, GPU inference or training."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np

import stage113_frozen_closed_loop as source
from stage27_uav_common import scene_geometry, failure_events, tracking_iou

c,io=source.c,source.io
SCHEMA="stage114_cached_generated_candidate_failure_chain_v1"
STATUS="COMPLETE_CACHED_GENERATED_CANDIDATE_FAILURE_CHAIN"
SOURCE113="11cada1ae2be6ad882acb3f6a55e4a0073ea44dc9957b70dfe1c69a799d93d0c"
DESIGN=dict(good_iou=.5,low_iou=.2,supply_run=3,long_failure=32,window_before=16,window_after=16,standardized_level=10.)
TABLES=("policy_summary","seed_video","cases","failure_events","recovery","prefix","support_summary","scope","chain_excerpt")
STATES=("invalid_gt","selected_good","no_good_candidate","missed_good_candidate")
RATES=("physical_nonq0","harmful_q0","beneficial_q0","missed_good","severe_miss","all_low","no_good","selected_low")
MEANS=("q0_iou","selected_iou","best_iou","q0_coverage","selected_coverage","max_coverage","center_error",
       "width_ratio","height_ratio","forecast_standardized_max","forecast_standardized_rms","range_ratio",
       "range_contains_truth","scorer_standardized_max","scorer_clip_fraction")


def finite_mean(values):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    return float(x.mean()) if len(x) else None


def first(mask):
    ix=np.flatnonzero(mask)
    return int(ix[0])+1 if len(ix) else None


def run_start(mask,width):
    m=np.asarray(mask,bool)
    if len(m)<width:return None
    return first(np.lib.stride_tricks.sliding_window_view(m,width).all(1))


def history_at(data,entry,index):
    return ([np.asarray(entry["initial_box"],float)]+list(data["boxes"][:index]))[-5:]


def analyze(data,entry,arm,generator,scorer,old):
    """Attach diagnostic labels only; never change a recorded action or infer its future."""
    ap=source.e63.c.actor_plan(old);gt=np.asarray(entry["ground_truth"],float)[1:];n=len(gt)
    if len(data["frame_id"])!=n:raise ValueError("Incomplete labeled trajectory")
    valid=np.isfinite(gt).all(1)&(gt[:,2:]>0).all(1)
    bank=tracking_iou(data["candidate_boxes"],gt[:,None,:]);ix=np.arange(n);chosen=data["selected"]
    selected=bank[ix,chosen];q0=bank[:,0];best=bank.max(1)
    changed=np.any(data["boxes"]!=data["candidate_boxes"][:,0],axis=1)
    good,low=DESIGN["good_iou"],DESIGN["low_iou"]
    out=dict(frame_id=data["frame_id"].copy(),gt_valid=valid,candidate_iou=bank,q0_iou=q0,selected_iou=selected,best_iou=best,
        physical_nonq0=changed,harmful_q0=valid&(q0>=good)&(selected<low),beneficial_q0=valid&(q0<low)&(selected>=good),
        missed_good=valid&(best>=good)&(selected<good),severe_miss=valid&(best>=good)&(selected<low),
        all_low=valid&(best<low),no_good=valid&(best<good),selected_low=valid&(selected<low),
        state=np.select((~valid,selected>=good,best<good),(0,1,2),default=3).astype(int))
    for name in MEANS[3:]:out[name]=np.full(n,np.nan)
    for i in range(n):
        h=history_at(data,entry,i)
        if scorer is not None:
            x=source.common.features(data["features"][i],data["candidate_boxes"][i],h)
            z=(x-scorer["norm"]["center"])/scorer["norm"]["scale"]
            out["scorer_standardized_max"][i]=np.abs(z).max()
            out["scorer_clip_fraction"][i]=(np.abs(z)>scorer["design"]["feature_clip"]).mean()
        if generator is not None and data["generated_active"][i]:
            z=(data["forecast_x"][i]-generator["mean"]["center"])/generator["mean"]["scale"]
            out["forecast_standardized_max"][i]=np.abs(z).max()
            out["forecast_standardized_rms"][i]=np.sqrt(np.mean(z*z))
        if not valid[i]:continue
        coverage=[scene_geometry(p,gt[i],data["image_hw"][i],ap["search_factor"],ap["search_size"])["coverage"] for p in data["candidate_parents"][i]]
        out["q0_coverage"][i]=coverage[0];out["selected_coverage"][i]=coverage[chosen[i]];out["max_coverage"][i]=max(coverage)
        box=data["boxes"][i]
        out["center_error"][i]=np.linalg.norm(box[:2]+box[2:]/2-gt[i,:2]-gt[i,2:]/2)/np.sqrt(gt[i,2:].prod())
        out["width_ratio"][i],out["height_ratio"][i]=box[2:]/gt[i,2:]
        if generator is not None and data["generated_active"][i]:
            target=source.p.target(gt[i],data["history_latest"][i])
            ratio=np.abs(target-data["forecast_mean"][i])/np.maximum(data["conditional_radius"][i],1e-12)
            out["range_ratio"][i]=ratio.max();out["range_contains_truth"][i]=float((ratio<=1.).all())
    return out


def frame_record(data,a,index):
    def val(x):return float(x) if np.isfinite(x) else None
    selected=int(data["selected"][index]);names=source.e63.source_common.prior.FEATURES
    r=dict(frame=index+1,state=STATES[int(a["state"][index])],gt_valid=bool(a["gt_valid"][index]),selected=int(data["selected"][index]),
        generated_active=bool(data["generated_active"][index]),memory_write_eligible=bool(data["memory_write"][index]),
        candidate_iou="|".join(f"{x:.12g}" for x in a["candidate_iou"][index]),scores="|".join(f"{x:.12g}" for x in data["scores"][index]),
        score_gain_over_q0=float(data["scores"][index,selected]-data["scores"][index,0]),
        selected_raw_peak=float(data["features"][index,selected,names.index("raw_peak")]),q0_raw_peak=float(data["features"][index,0,names.index("raw_peak")]),
        selected_hann_peak=float(data["features"][index,selected,names.index("hann_peak")]),q0_hann_peak=float(data["features"][index,0,names.index("hann_peak")]),
        forecast_mean="|".join(f"{x:.12g}" for x in data["forecast_mean"][index]),conditional_radius="|".join(f"{x:.12g}" for x in data["conditional_radius"][index]))
    r.update({k:bool(a[k][index]) for k in RATES});r.update({k:val(a[k][index]) for k in MEANS})
    return r


def geometry_shadow(data,entry,index,arm,generator,old):
    """Change one committed box, then recompute parents only. No alternate outputs."""
    empty=dict(next_shadow_available=False,next_q0_commit_changes_crops=None,next_q0_commit_parent_delta=None)
    if index+1>=len(data["frame_id"]):return empty
    ap=source.e63.c.actor_plan(old);h=history_at(data,entry,index);row={k:v[index] for k,v in data.items()}
    actual=(h+[row["boxes"]])[-5:]
    expected=source.forecast_parents(actual,row,data["image_hw"][index+1],ap,arm,generator)["candidate_parents"]
    np.testing.assert_array_equal(expected,data["candidate_parents"][index+1])
    alternative=dict(row,selected=0,boxes=row["candidate_boxes"][0],parents=row["candidate_parents"][0])
    alternative["memory_write"]=False if arm=="native" else source.write_eligible(alternative)
    other=source.forecast_parents((h+[alternative["boxes"]])[-5:],alternative,data["image_hw"][index+1],ap,arm,generator)["candidate_parents"]
    crop=lambda p:source.s105.crop_geometry(p,data["image_hw"][index+1],ap["search_factor"],ap["search_size"])[0]
    return dict(next_shadow_available=True,next_q0_commit_changes_crops=bool(any(not np.array_equal(crop(x),crop(y)) for x,y in zip(expected,other))),
        next_q0_commit_parent_delta=float(np.max(np.abs(expected-other))))


def prefix(candidate,reference,a,reference_iou,ident,reference_arm):
    n=len(candidate["frame_id"])
    if len(reference["frame_id"])!=n:raise ValueError("Mismatched reference length")
    diff=np.any(candidate["boxes"]!=reference["boxes"],axis=1);f=first(diff);stop=f if f is not None else n
    # Until and including the first output difference the past boxes are identical.
    np.testing.assert_array_equal(candidate["history_latest"][:stop],reference["history_latest"][:stop])
    np.testing.assert_array_equal(candidate["candidate_parents"][:stop,0],reference["candidate_parents"][:stop,0])
    np.testing.assert_array_equal(candidate["candidate_boxes"][:stop,0],reference["candidate_boxes"][:stop,0])
    history_diff=first(np.any(candidate["history_latest"]!=reference["history_latest"],axis=1))
    if history_diff!=(f+1 if f is not None and f<n else None):raise ValueError("Own-history divergence is not the preceding output")
    return dict(ident,reference_arm=reference_arm,frames=n,first_output_difference=f,first_history_difference=history_diff,
        first_q0_crop_difference=first(np.any(candidate["crop"][:,0]!=reference["crop"][:,0],axis=1)),
        identical_past_q0_verified=True,at_first_candidate_iou=float(a["selected_iou"][f-1]) if f else None,
        at_first_reference_iou=float(reference_iou[f-1]) if f else None,
        at_first_q0_iou=float(a["q0_iou"][f-1]) if f else None,at_first_best_iou=float(a["best_iou"][f-1]) if f else None,
        at_first_harmful_q0=bool(a["harmful_q0"][f-1]) if f else None)


def case_rows(data,a,entry,ident,arm,generator,old,roles):
    cases=[];excerpt=set();n=len(data["frame_id"])
    for role,frame,reference in sorted(set(roles),key=lambda r:(r[1],r[0],r[2])):
        i=frame-1;start=max(0,i-DESIGN["window_before"]);stop=min(n,i+DESIGN["window_after"]+1)
        last=lambda key:next((j+1 for j in range(i,-1,-1) if a[key][j]),None)
        cases.append(dict(ident,role=role,reference=reference,window_start=start+1,window_end=stop,
            last_harmful_q0=last("harmful_q0"),last_severe_miss=last("severe_miss"),last_all_low=last("all_low"),
            local_harmful_q0_frames=int(a["harmful_q0"][start:stop].sum()),local_all_low_frames=int(a["all_low"][start:stop].sum()),
            **frame_record(data,a,i),**geometry_shadow(data,entry,i,arm,generator,old)))
        excerpt.update(range(start,stop))
    return cases,[dict(ident,**frame_record(data,a,i)) for i in sorted(excerpt)]


def local_window(a,frame):
    i=frame-1;lo=max(0,i-DESIGN["window_before"]);hi=min(len(a["frame_id"]),i+DESIGN["window_after"]+1)
    harmful=first(a["harmful_q0"][lo:hi]);supply=run_start(a["all_low"][lo:hi],DESIGN["supply_run"])
    hf=lo+harmful if harmful else None;sf=lo+supply if supply else None
    order="left_censored" if bool(a["all_low"][lo]) else "neither" if hf is None and sf is None else "harmful_only" if sf is None else "supply_only" if hf is None else "harmful_first" if hf<sf else "supply_first" if sf<hf else "same_frame"
    return dict(local_window_start=lo+1,local_window_end=hi,local_first_harmful_q0=hf,local_first_sustained_all_low=sf,
        local_order=order,local_mean_best_iou=finite_mean(a["best_iou"][lo:hi]),
        local_mean_max_coverage=finite_mean(a["max_coverage"][lo:hi]),local_mean_range_contains_truth=finite_mean(a["range_contains_truth"][lo:hi]))


def aggregate(units):
    out=[]
    for cohort in ("all","historical","expanded"):
        for arm in source.ARMS:
            part=[r for r in units if r["arm"]==arm and (cohort=="all" or r["cohort"]==cohort)]
            if not part:continue
            videos=defaultdict(list)
            for r in part:videos[r["sequence"]].append(r)
            keys=[k+"_fraction" for k in RATES]+list(MEANS)
            out.append(dict(cohort=cohort,arm=arm,videos=len(videos),trajectories=len(part),frames=sum(r["frames"] for r in part),
                **{k+"_frames":sum(r[k+"_frames"] for r in part) for k in RATES},
                **{k:finite_mean([finite_mean([r[k] for r in local if r[k] is not None]) for local in videos.values()]) for k in keys}))
    return out


def support_rows(a,ident):
    rows=[];n=len(a["frame_id"])
    for phase,mask in (("cold",np.arange(n)==0),("short_history",(np.arange(n)>=1)&(np.arange(n)<4)),("from_frame5",np.arange(n)>=4)):
        for state in range(len(STATES)):
            take=mask&(a["state"]==state)
            if not take.any():continue
            rows.append(dict(ident,phase=phase,state=STATES[state],frames=int(take.sum()),
                **{k:finite_mean(a[k][take]) for k in MEANS},harmful_q0_frames=int(a["harmful_q0"][take].sum())))
    return rows


def compare_recorded(path,rows):
    source.s103.source102.compare_recorded(source.s103.prior.read_csv(path),rows)


def run(root,root113,plan113,plan,models,load_entry,old):
    tables={k:[] for k in TABLES};evidence={};source_seeds=[];source_recoveries=[]
    groups=defaultdict(list)
    for job in plan113["jobs"]:groups[job["spec"]["unit"]["sequence"]].append(job)
    for sequence,jobs in groups.items():
        entry=load_entry(jobs[0]["spec"]);traces={};audits={};gt=np.asarray(entry["ground_truth"],float)[1:]
        for job in jobs:
            arm,seed=job["arm"],job["seed"];generator,scorer=source.job_models(job,models)
            meta=dict(plan_fingerprint=plan113["fingerprint"],job=job,
                generator_fingerprint=models[job["outer_fold"],"generator",0]["fingerprint"] if generator is not None else None,
                scorer_fingerprint=scorer["fingerprint"] if scorer is not None else None)
            data=c.read_cache(root113/"evaluation"/(source.job_key(job)+".npz"),meta)
            if data is None:raise ValueError("Missing sealed full trajectory")
            source.audit(data,entry,old,arm,generator,scorer)
            derived_meta=dict(plan_fingerprint=plan["fingerprint"],job=job,truth_fingerprint=io.seal(dict(gt=gt.tolist()))["fingerprint"])
            path=root/"audit"/(source.job_key(job)+".npz")
            a=source.s109.cache(root,path,derived_meta,lambda:analyze(data,entry,arm,generator,scorer,old),evidence)
            np.testing.assert_array_equal(a["frame_id"],data["frame_id"])
            traces[arm,seed]=data;audits[arm,seed]=a
        native=traces["native",0];native_iou=audits["native",0]["selected_iou"]
        for job in jobs:
            arm,seed=job["arm"],job["seed"];data=traces[arm,seed];a=audits[arm,seed];generator,_=source.job_models(job,models)
            ident=dict(cohort=job["spec"]["cohort"],sequence=sequence,arm=arm,seed=seed,outer_fold=job["outer_fold"])
            overlap=a["selected_iou"];events=failure_events(overlap)
            metric=source.s103.s93.s85.metrics(data,entry,native["boxes"],source.s103.DESIGN)
            timing=data["pipeline_ms"][source.DESIGN["latency_warmup"]:]
            nr=source.recovery_rows(native_iou,overlap,ident,"native") if arm!="native" else []
            fr=source.recovery_rows(audits["fixed_shared",seed]["selected_iou"],overlap,ident,"fixed_shared") if arm=="generated_shared" else []
            metric.update(failure_events=len(events),new_on_fixed_stable=not failure_events(audits["fixed_shared",seed]["selected_iou"]) and bool(events) if arm=="generated_shared" else False,
                native_recoveries_lost=sum(r["matched_good_frame"] is None for r in nr),fixed_recoveries_lost=sum(r["matched_good_frame"] is None for r in fr),
                median_pipeline_ms=float(np.median(timing)) if len(timing) else None,p95_pipeline_ms=float(np.quantile(timing,.95)) if len(timing) else None)
            source_seeds.append(dict(ident,**{k:metric[k] for k in (*source.METRICS,*source.FLAGS)}));source_recoveries.extend(nr+fr)
            unit=dict(ident,frames=len(gt),valid_frames=int(a["gt_valid"].sum()),
                **{k+"_frames":int(a[k].sum()) for k in RATES},
                **{k+"_fraction":finite_mean(a[k] if k=="physical_nonq0" else a[k][a["gt_valid"]]) for k in RATES},**{k:finite_mean(a[k][a["gt_valid"]]) for k in MEANS},
                **{"first_"+k:first(a[k]) for k in ("physical_nonq0","harmful_q0","severe_miss","all_low")},
                first_sustained_all_low=run_start(a["all_low"],DESIGN["supply_run"]),first_failure=events[0]["start"]+1 if events else None,
                first_long_failure=next((ev["start"]+1 for ev in events if ev["end"]-ev["start"]+1>=DESIGN["long_failure"]),None))
            tables["seed_video"].append(unit);tables["support_summary"].extend(support_rows(a,ident))
            roles=[]
            for name in ("physical_nonq0","harmful_q0","severe_miss","all_low"):
                f=first(a[name])
                if f:roles.append(("first_"+name,f,""))
            f=run_start(a["all_low"],DESIGN["supply_run"])
            if f:roles.append(("first_sustained_all_low",f,""))
            if events:roles.append(("first_failure",events[0]["start"]+1,""))
            long=next((ev for ev in events if ev["end"]-ev["start"]+1>=DESIGN["long_failure"]),None)
            if long:roles.append(("first_long_failure",long["start"]+1,""))
            refs=[] if arm=="native" else [("native",0)]
            if arm.startswith("generated"):refs.append(("fixed_shared" if arm.endswith("shared") else "fixed_hann",seed))
            for ref,rs in refs:
                pr=prefix(data,traces[ref,rs],a,audits[ref,rs]["selected_iou"],ident,ref);tables["prefix"].append(pr)
                if pr["first_output_difference"]:roles.append(("first_output_difference",pr["first_output_difference"],ref))
            for index,ev in enumerate(events,1):
                frame=ev["start"]+1
                tables["failure_events"].append(dict(ident,event=index,onset_frame=frame,confirmation_frame=ev["confirmation"]+1,
                    end_frame=ev["end"]+1,recovery_frame=ev["recovery"]+1 if ev["recovery"] is not None else None,censor=ev["censor"],
                    duration_frames=ev["end"]-ev["start"]+1,**frame_record(data,a,frame-1),**local_window(a,frame)))
            for rec in nr+fr:
                frame=rec["reference_recovery_frame"]
                tables["recovery"].append(dict(rec,**frame_record(data,a,frame-1),**local_window(a,frame)))
                if rec["matched_good_frame"] is None:roles.append(("unmatched_reference_recovery",frame,rec["reference_arm"]))
            cases,excerpt=case_rows(data,a,entry,ident,arm,generator,old,roles)
            tables["cases"].extend(cases);tables["chain_excerpt"].extend(excerpt)
            tables["scope"].append(dict(ident,frames=len(gt),source_physical_search_calls=int(data["unique_search_calls"].sum()),
                generated_frames=int(data["generated_active"].sum()),source_actions_replayed=True,images_read=0,new_search_calls=0))
        print(f"[Stage-114] {sequence}: seven frozen histories audited",flush=True)
    compare_recorded(root113/"analysis/stage113_seed_video.csv",source_seeds)
    compare_recorded(root113/"analysis/stage113_recovery.csv",source_recoveries)
    policies,paired,videos=source.summarize(source_seeds)
    for name,rows in (("policy_summary",policies),("paired",paired),("video_summary",videos)):compare_recorded(root113/f"analysis/stage113_{name}.csv",rows)
    tables["policy_summary"]=aggregate(tables["seed_video"])
    publish(root,plan,tables,evidence)


def publish(root,plan,tables,evidence):
    expected=plan["expected"];counts={k:len(v) for k,v in tables.items()}
    if (set(tables)!=set(TABLES) or counts["seed_video"]!=expected["trajectories"] or counts["scope"]!=expected["trajectories"]
        or sum(r["frames"] for r in tables["scope"])!=expected["frames"] or counts["recovery"]!=expected["recoveries"]
        or sum(r["source_physical_search_calls"] for r in tables["scope"])!=expected["source_search_calls"]
        or counts["failure_events"]!=expected["failures"] or counts["prefix"]!=expected["prefixes"] or len(evidence)!=2*expected["trajectories"]):raise ValueError("Incomplete cached diagnostic")
    io.verify(plan["source_sha256"]);io.verify(plan["code_sha256"])
    out=root/"analysis";out.mkdir(exist_ok=True);hashes={}
    for name,rows in tables.items():
        path=out/f"stage114_{name}.csv";source.s103.s86.publish_table(path,rows);hashes[path.name]=c.file_hash(path)
    c.publish_identical(out/"stage114_decision.json",io.seal(dict(schema=SCHEMA,status=STATUS,plan_fingerprint=plan["fingerprint"],
        source_stage113_fingerprint=SOURCE113,source_runtime=plan["source_runtime"],counts=counts,source_summary_reproduction_pass=True,
        cached_frames=expected["frames"],source_search_calls_audited=expected["source_search_calls"],
        own_history_action_replay_pass=True,identical_past_q0_pass=True,one_commit_geometry_shadow_pass=True,
        image_reads=0,new_search_calls=0,new_block_calls=0,training_updates=0,new_models=0,new_full_trajectories=0,confirmation_inference_calls=0,
        table_sha256=hashes,artifact_sha256=evidence,warnings=[
            "All source groups are consumed development. COMPLETE describes execution, not method adoption.",
            "Current q0/oracle comparisons condition on the actual history; their unused futures are not observed.",
            "One-commit shadows compute next parents/crops only; they do not produce counterfactual IoU or recovery.",
            "Standardized features and empirical forecast containment are descriptive, not calibrated OOD probabilities.",
            "Local event order is descriptive and may be censored; image-level environmental causes remain unobserved."])))
    check_results(root)


def check_results(root):
    plan=c.c66.sealed(root/"manifest.json",SCHEMA);r=c.c66.sealed(root/"analysis/stage114_decision.json",SCHEMA);e=plan["expected"]
    if (plan["design"]!=DESIGN or plan["source_stage113_fingerprint"]!=SOURCE113 or r["status"]!=STATUS or r["plan_fingerprint"]!=plan["fingerprint"] or r["source_stage113_fingerprint"]!=SOURCE113
        or r["source_runtime"]!=plan["source_runtime"] or set(r["table_sha256"])!={f"stage114_{n}.csv" for n in TABLES}
        or r["cached_frames"]!=e["frames"] or r["source_search_calls_audited"]!=e["source_search_calls"]
        or r["counts"]["scope"]!=e["trajectories"] or r["counts"]["seed_video"]!=e["trajectories"] or r["counts"]["prefix"]!=e["prefixes"]
        or r["counts"]["recovery"]!=e["recoveries"] or r["counts"]["failure_events"]!=e["failures"] or len(r["artifact_sha256"])!=2*e["trajectories"]
        or not all(r[k] for k in ("source_summary_reproduction_pass","own_history_action_replay_pass","identical_past_q0_pass","one_commit_geometry_shadow_pass"))
        or any(r[k]!=0 for k in ("image_reads","new_search_calls","new_block_calls","training_updates","new_models","new_full_trajectories","confirmation_inference_calls"))):raise ValueError("Incomplete Stage-114 audit")
    io.verify(plan["source_sha256"]);io.verify(plan["code_sha256"])
    io.verify({str(root/"analysis"/n):h for n,h in r["table_sha256"].items()});io.verify({str(root/n):h for n,h in r["artifact_sha256"].items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__);names=("stage76","stage89","stage93","stage112","stage113")
    for name in (*names,"output_dir"):parser.add_argument("--"+name,type=Path,required=name=="output_dir")
    parser.add_argument("--check-results",action="store_true");args=parser.parse_args();root=args.output_dir.expanduser().resolve()
    if args.check_results:check_results(root);return
    if not all(getattr(args,n) for n in names):parser.error("Supply Stage-76/89/93/112/113 source directories")
    r76,r89,r93,r112,r113=sources=[getattr(args,n).expanduser().resolve() for n in names]
    if any(root==q or root in q.parents or q in root.parents for q in sources):parser.error("Use a separate Stage-114 root")
    with source.s109.s106.s93.writer_lock(root):
        if (root/"analysis/stage114_decision.json").exists():check_results(root);return
        if not (root/"manifest.json").exists() and any(q.name!="collect.lock" for q in root.iterdir()):raise ValueError("Use an empty Stage-114 root")
        source.check_results(r113);source.s112.check_results(r112)
        p113=c.c66.sealed(r113/"manifest.json",source.SCHEMA);r=c.c66.sealed(r113/"analysis/stage113_decision.json",source.SCHEMA)
        p112=c.c66.sealed(r112/"manifest.json",source.common.SCHEMA)
        if r["fingerprint"]!=SOURCE113 or len(p113["jobs"])!=196:raise ValueError("Need the complete frozen Stage-113 result")
        p93,_,p89,r89report,origin=source.s103.prior.s95.s94.load_run(r93,r76,r89)
        models,_=source.load_frozen_models(r112,p112)
        seed_rows=source.s103.prior.read_csv(r113/"analysis/stage113_seed_video.csv")
        expected=dict(trajectories=196,frames=sum(j["frames"] for j in p113["jobs"]),recoveries=r["table_counts"]["recovery"],source_search_calls=r["new_search_calls"],
            failures=int(sum(float(z["failure_events"]) for z in seed_rows)),prefixes=9*len(p113["entries"]))
        if expected!=dict(trajectories=196,frames=203329,recoveries=102,source_search_calls=725283,failures=147,prefixes=252):raise ValueError("Frozen Stage-113 cohort changed")
        bindings=dict(p113["source_sha256"])
        for q in (r113/"manifest.json",r113/"analysis/stage113_decision.json",r112/"manifest.json"):bindings[str(q)]=c.file_hash(q)
        bindings.update({str(r113/n):h for n,h in r["artifact_sha256"].items()});bindings.update({str(r113/"analysis"/n):h for n,h in r["table_sha256"].items()})
        code=[Path(__file__).resolve(),Path(__file__).with_name("run_stage114_failure_chain.sh").resolve()]
        plan=io.seal(dict(schema=SCHEMA,design=DESIGN,expected=expected,source_stage113_fingerprint=SOURCE113,source_runtime=r["runtime"],source_sha256=bindings,
            code_sha256=p113["code_sha256"]|{str(q):c.file_hash(q) for q in code}))
        c.publish_identical(root/"manifest.json",plan)
        def load_entry(spec):return source.s103.prior.s95.load_unit(r93,r89,p93,p89,r89report,origin,spec)[0]
        print("Stage-114: all 196 cached histories; no image reads, GPU inference or training",flush=True)
        run(root,r113,p113,plan,models,load_entry,origin["original"])
        print("Stage-114: "+STATUS+"; no method adoption gate",flush=True)


if __name__=="__main__":main()
