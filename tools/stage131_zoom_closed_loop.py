"""Run the Stage-130 q3-preserving rank-one search on complete own histories."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import stage130_preserve_q3 as source
from stage27_uav_common import crop_geometry, failure_events, tracking_iou

prior, base, c, io = source.prior, source.base, source.c, source.io
s121 = prior.s121
coarse = source.source
trim_source = coarse.source
SCHEMA = "stage131_q3_zoom_closed_loop_v1"
SOURCE121 = trim_source.audit.SOURCE121
SOURCE130 = "1a02bace8724c99c6e275b8dc97d1ad1508fc5437058424a46422788660acbcd"
DESIGN = dict(bootstrap_seed=23131, bootstrap_samples=20000, practical_auc=.5)
ARMS = ("native","alternate_shared","trim","rank1")
FIELDS = tuple(k for k in s121.FIELDS if k not in ("search_ms","pipeline_ms","crop_sha256")) + (
    "untrimmed_parents","trimmed_parents","zoom_slot","wide_slot","wide_cell")
METRICS = (*s121.METRICS, "best_candidate_iou", "selected_good_fraction", "missed_good_fraction")
TABLES = ("policy_summary","paired","seed_video","scope","recovery")
WORK = dict(new_models=0, training_updates=0, confirmation_inference_calls=0)


def layout(proposal, hw, ap, arm, get_response):
    original = proposal["candidate_parents"]
    trimmed = np.array([trim_source.trim_padding(p,hw,ap["search_factor"],ap["search_size"]) for p in original])
    if arm=="trim" or not proposal["alternate_used"]:
        return trimmed,trimmed,-1,-1,-1
    changed = np.flatnonzero(np.any(trimmed!=original,axis=1))
    if not changed.size:
        return trimmed,trimmed,-1,-1,-1
    wide = max(map(int,changed),key=lambda q:crop_geometry(original[q],hw,ap["search_factor"],ap["search_size"])[0][2])
    options,_,peaks,_ = coarse.zoom_layout(trimmed,original,{wide:get_response(wide,trimmed[wide])},hw,ap)
    if not peaks:
        return trimmed,trimmed,-1,-1,-1
    first = min(peaks)
    slot = source.replacement_slot(wide)
    parents = trimmed.copy()
    parents[slot] = options[first]
    return parents,trimmed,slot,wide,peaks[first]


def collect(observer, entry, old, arm, generator, scorer, image_reader=base.s105.checked_image):
    ap = base.e63.c.actor_plan(old)
    frames = entry["frames"]
    history = [np.asarray(entry["initial_box"],float)]
    observer.initialize(image_reader(ap["root"],frames[0]),history[0])
    previous, rows = None,[]
    for frame,item in enumerate(frames[1:],1):
        image = image_reader(ap["root"],item)
        hw = image.shape[:2]
        views = {}
        proposal = s121.forecast(history,previous,hw,ap,generator)
        def observe(parent):
            key = tuple(parent)
            if key not in views:
                views[key] = base.e63.actor.observe(observer,image,parent,frame,ap)
            return views[key]

        parents,trimmed,slot,wide,cell = layout(proposal,hw,ap,arm,lambda q,p:observe(p)["raw"])
        ordered = [observe(parent) for parent in parents]
        executed = np.array([not any(np.array_equal(parent,p) for p in parents[:i])
                             for i,parent in enumerate(parents)],bool)
        if len(views)!=executed.sum() or len(views)>4:
            raise ValueError("Preliminary search did not fit the four-slot budget")
        row = dict(proposal,candidate_parents=parents,untrimmed_parents=proposal["candidate_parents"],
                   trimmed_parents=trimmed,zoom_slot=slot,wide_slot=wide,wide_cell=cell,
                   candidate_boxes=np.asarray([v["boxes"] for v in ordered]),
                   features=np.asarray([base.e63.actor.feature_vector(v,history,ap) for v in ordered]),
                   image_hw=hw,frame_id=frame,unique_search_calls=len(views),history_latest=history[-1],executed=executed)
        row.update({k:np.asarray([v[k] for v in ordered]) for k in base.s112.MAPS if k!="crop_sha256"})
        scores,selected = base.choose(row,history,"generated_shared",scorer)
        row.update(scores=scores,selected=selected,boxes=row["candidate_boxes"][selected].copy(),parents=parents[selected].copy())
        row["memory_write"] = base.write_eligible(row)
        history = [*history,row["boxes"]][-ap["design"]["history"]:]
        observer.rt.tracker.state = row["boxes"].tolist()
        previous = row
        rows.append(row)
        if frame%500==0:
            print(f"  [Stage-131] {arm} {frame}/{len(frames)-1}",flush=True)
    return {k:np.asarray([r[k] for r in rows]) for k in FIELDS}


def audit(data,entry,old,arm,generator,scorer):
    ap = base.e63.c.actor_plan(old)
    n = len(entry["frames"])-1
    if set(data)!=set(FIELDS) or any(len(v)!=n for v in data.values()):
        raise ValueError("Incomplete closed-loop trajectory")
    np.testing.assert_array_equal(data["frame_id"],np.arange(1,n+1))
    base.e63.source_common.prior.audit_calls(data,(n,4))
    if not np.isfinite(data["decode_error"]).all() or data["decode_error"].max()>ap["design"]["box_atol"]:
        raise ValueError("Native decoder drift")
    history,previous = [np.asarray(entry["initial_box"],float)],None
    for i,frame in enumerate(data["frame_id"]):
        row = {k:v[i] for k,v in data.items()}
        item = entry["frames"][frame]
        hw = item["height"],item["width"]
        np.testing.assert_array_equal(row["image_hw"],hw)
        np.testing.assert_array_equal(row["history_latest"],history[-1])
        forecast = s121.forecast(history,previous,hw,ap,generator)
        np.testing.assert_array_equal(row["untrimmed_parents"],forecast["candidate_parents"])
        for k in set(forecast)-{"candidate_parents"}:
            np.testing.assert_array_equal(row[k],forecast[k])
        parents,trimmed,q,w,cell = layout(forecast,hw,ap,arm,lambda j,p:row["raw"][j])
        np.testing.assert_equal((row["zoom_slot"],row["wide_slot"],row["wide_cell"]),(q,w,cell))
        np.testing.assert_array_equal(row["candidate_parents"],parents)
        np.testing.assert_array_equal(row["trimmed_parents"],trimmed)
        np.testing.assert_array_equal(row["candidate_parents"][3],trimmed[3])
        expected = np.array([not any(np.array_equal(parent,p) for p in parents[:j]) for j,parent in enumerate(parents)])
        np.testing.assert_array_equal(row["executed"],expected)
        np.testing.assert_equal(row["unique_search_calls"],int(expected.sum()))
        for j,parent in enumerate(parents):
            rect,usable,factor = crop_geometry(parent,hw,ap["search_factor"],ap["search_size"])
            np.testing.assert_array_equal(row["crop"][j],rect)
            view = dict(parent=parent,boxes=row["candidate_boxes"][j],crop=rect,usable=usable,image_hw=hw,
                        **{k:row[k][j] for k in ("raw","size","offset")})
            np.testing.assert_array_equal(row["features"][j],base.e63.actor.feature_vector(view,history,ap))
            dense = base.e63.dense_boxes(view["size"],view["offset"],parent,factor,ap["search_size"],hw)
            selected_peak = int(np.argmax(view["raw"]*np.asarray(ap["window"],np.float32)))
            np.testing.assert_allclose(dense[selected_peak],view["boxes"],atol=ap["design"]["box_atol"],rtol=0)
        scores,selected = base.choose(row,history,"generated_shared",scorer)
        np.testing.assert_allclose(row["scores"],scores,atol=1e-12,rtol=1e-9)
        np.testing.assert_equal(row["selected"],selected)
        np.testing.assert_array_equal(row["boxes"],row["candidate_boxes"][selected])
        np.testing.assert_array_equal(row["parents"],parents[selected])
        np.testing.assert_equal(row["memory_write"],base.write_eligible(row))
        history,previous = [*history,row["boxes"]][-ap["design"]["history"]:],row


def prefix(rank1,trim):
    active = np.flatnonzero(rank1["zoom_slot"]>=0)
    i = int(active[0]) if active.size else len(trim["frame_id"])
    for k in set(FIELDS)-{"zoom_slot","wide_slot","wide_cell"}:
        np.testing.assert_array_equal(rank1[k][:i],trim[k][:i],err_msg=k)
    if i==len(trim["frame_id"]):
        return None
    np.testing.assert_array_equal(rank1["trimmed_parents"][i],trim["candidate_parents"][i])
    slot = int(rank1["zoom_slot"][i])
    others = [q for q in range(4) if q!=slot]
    for k in ("candidate_parents","candidate_boxes","features","crop"):
        np.testing.assert_array_equal(rank1[k][i,others],trim[k][i,others],err_msg=k)
    return i+1


def source_jobs(p121):
    jobs = [job for job in p121["jobs"] if job["arm"] in ARMS[:2]]
    jobs.extend(dict(job,arm=arm) for arm in ARMS[2:] for job in p121["jobs"] if job["arm"]==s121.TREATMENT)
    return jobs


def summarize(rows,recovery):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["sequence"],r["arm"]].append(r)
    macro = {}
    for k,items in grouped.items():
        if sorted(r["seed"] for r in items)!=([0] if k[1]=="native" else base.DESIGN["seeds"]):
            raise ValueError("Missing comparison seeds")
        macro[k] = {m:float(np.mean(v)) if (v:=[r[m] for r in items if r[m] is not None]) else None for m in METRICS}
    sequences = sorted({k[0] for k in macro})
    summary,paired = [],[]
    for arm in ARMS:
        summary.append(dict(arm=arm,videos=len(sequences),**{
            m:float(np.mean(v)) if (v:=[macro[s,arm][m] for s in sequences if macro[s,arm][m] is not None]) else None
            for m in METRICS}))
    for arm,ref in (("rank1","trim"),("rank1","alternate_shared"),("rank1","native"),("trim","alternate_shared")):
        for m in METRICS:
            values = [macro[s,arm][m]-macro[s,ref][m] for s in sequences
                      if macro[s,arm][m] is not None and macro[s,ref][m] is not None]
            if values:
                paired.append(base.s103.contrast(values,arm+"_minus_"+ref,m,DESIGN))
    seeds = {(r["sequence"],r["arm"],r["seed"]):r["auc"] for r in rows}
    seed_delta = {str(k):float(np.mean([seeds[s,"rank1",k]-seeds[s,"trim",k] for s in sequences]))
                  for k in base.DESIGN["seeds"]}
    protection = {ref:dict(new_failure_videos=len({r["sequence"] for r in rows if r["arm"]=="rank1"
                      and r["new_failure_vs_"+ref]}),
                      lost_recoveries=sum(r["reference_arm"]==ref and r["matched_good_frame"] is None for r in recovery))
                  for ref in ARMS[:3]}
    return summary,paired,seed_delta,protection


def report_video(tables,jobs,traces,entry,reuse):
    gt = np.asarray(entry["ground_truth"],float)[1:]
    overlaps = {k:tracking_iou(r["boxes"],gt) for k,r in traces.items()}
    events = {k:failure_events(iou) for k,iou in overlaps.items()}
    for job in jobs:
        arm,seed = key = job["arm"],job["seed"]
        d = traces[key]
        ident = dict(sequence=entry["sequence"],cohort=job["spec"]["cohort"],arm=arm,seed=seed,
                     outer_fold=job["outer_fold"])
        metric = base.s103.s93.s85.metrics(d,entry,traces["native",0]["boxes"],base.s103.DESIGN)
        output = dict(ident,**{m:metric[m] for m in METRICS if m!="failure_events"},failure_events=len(events[key]))
        first = None
        for ref in ARMS[:3]:
            ref_key = ref,0 if ref=="native" else seed
            output["new_failure_vs_"+ref] = bool(events[key]) and not events[ref_key] if arm=="rank1" else None
            if arm=="rank1":
                tables["recovery"].extend(base.recovery_rows(overlaps[ref_key],overlaps[key],ident,ref))
        active = d["zoom_slot"]>=0 if arm=="rank1" else np.zeros(len(gt),bool)
        if arm=="rank1":
            first = prefix(d,traces["trim",seed])
        tables["seed_video"].append(output)
        calls = int(d["unique_search_calls"].sum())
        blocks = int((d["block_calls"]*d["executed"][...,None]).sum())
        heads = int((d["head_calls"]*d["executed"]).sum())
        logical = len(gt)*(1 if arm=="native" else 4)
        if not len(gt)<=calls<=logical or blocks!=7*calls or heads!=calls:
            raise ValueError("Search budget mismatch")
        tables["scope"].append(dict(ident,frames=len(gt),provenance="stage121_cache" if reuse and arm in ARMS[:2] else "new",
            logical_calls=logical,physical_calls=calls,head_calls=heads,block_calls=blocks,
            alternate_frames=int(d["alternate_used"].sum()) if arm!="native" else 0,
            trimmed_frames=int(np.any(d["trimmed_parents"]!=d["untrimmed_parents"],axis=(1,2)).sum()) if arm in ARMS[2:] else 0,
            zoom_frames=int(active.sum()),zoom_selected_frames=int((active & (d["selected"]==d["zoom_slot"])).sum()) if arm=="rank1" else 0,
            first_zoom_frame=first))


def check_results(root):
    plan = c.c66.sealed(root/"manifest.json",SCHEMA)
    report = c.c66.sealed(root/"analysis/stage131_decision.json",SCHEMA)
    new = [j for j in plan["jobs"] if not plan["reuse_controls"] or j["arm"] in ARMS[2:]]
    if (plan["design"]!=DESIGN or report["plan_fingerprint"]!=plan["fingerprint"] or report["runtime"]!=plan["runtime"]
            or report["source_stage130_fingerprint"]!=SOURCE130 or report["new_full_trajectories"]!=len(new)
            or report["evaluated_trajectories"]!=196
            or not sum(j["frames"] for j in new)<=report["new_search_calls"]<=plan["maximum_new_search_calls"]
            or len(report["artifact_sha256"])!=2*len(new)
            or set(report["table_sha256"])!={f"stage131_{n}.csv" for n in TABLES}
            or any(report[k]!=v for k,v in WORK.items())):
        raise ValueError("Incomplete Stage-131 result")
    io.verify({str(root/"analysis"/k):v for k,v in report["table_sha256"].items()})
    io.verify({str(root/k):v for k,v in report["artifact_sha256"].items()})


def run(root,r121,p121,plan,old,models,load_entry,observer):
    tables = {k:[] for k in TABLES}
    evidence,grouped = {},defaultdict(list)
    reuse = plan["reuse_controls"]
    for j in plan["jobs"]:
        grouped[j["spec"]["unit"]["sequence"]].append(j)
    for seq,jobs in sorted(grouped.items()):
        entry = load_entry(jobs[0]["spec"])
        if entry["sequence"]!=seq or any(j["frames"]!=len(entry["frames"])-1 for j in jobs):
            raise ValueError("Source video or frame count changed")
        traces = {}
        with base.e63.source_experiment.installed(observer,[],jobs[0]["spec"]["unit"]) as head:
            if head!=p121["native_head_sha256"]:
                raise ValueError("Native SGLA head changed")
            for job in jobs:
                arm = job["arm"]
                generator,scorer = base.job_models(dict(job,arm="native" if arm=="native" else "generated_shared"),models)
                if reuse and arm in ARMS[:2]:
                    data = prior.prior.load_trace(r121,p121,job)
                else:
                    path = root/"evaluation"/(base.job_key(job)+".npz")
                    meta = dict(plan_fingerprint=plan["fingerprint"],job=job)
                    if arm=="native":
                        data = base.s109.cache(root,path,meta,lambda:base.collect(observer,entry,old,arm,None,None),evidence)
                        base.audit(data,entry,old,arm,None,None)
                    elif arm==s121.TREATMENT:
                        data = base.s109.cache(root,path,meta,lambda:s121.collect(
                            observer,entry["frames"],entry["initial_box"],old,generator,scorer),evidence)
                        s121.audit(data,entry,old,generator,scorer)
                    else:
                        data = base.s109.cache(root,path,meta,lambda:collect(
                            observer,entry,old,arm,generator,scorer),evidence)
                        audit(data,entry,old,arm,generator,scorer)
                traces[arm,job["seed"]] = data
                print(f"[Stage-131] {seq} {arm} seed={job['seed']}: {len(data['frame_id'])} frames",flush=True)
        report_video(tables,jobs,traces,entry,reuse)
    tables["policy_summary"],tables["paired"],by_seed,protection = summarize(tables["seed_video"],tables["recovery"])
    primary = next(r for r in tables["paired"] if r["contrast"]=="rank1_minus_trim" and r["metric"]=="auc")
    checks = dict(auc_gain=primary["mean_delta"]>=DESIGN["practical_auc"] and primary["ci_low"]>0,
                  positive_both_seeds=all(v>0 for v in by_seed.values()),
                  preserves_trim=not any(protection["trim"].values()))
    out = root/"analysis"
    out.mkdir(exist_ok=True)
    hashes = {}
    for k,rows in tables.items():
        path = out/f"stage131_{k}.csv"
        base.s103.s86.publish_table(path,rows)
        hashes[path.name] = c.file_hash(path)
    new = [r for r in tables["scope"] if r["provenance"]=="new"]
    status = "PROMISING_DEVELOPMENT" if all(checks.values()) else "REVIEW_DEVELOPMENT"
    c.publish_identical(out/"stage131_decision.json",io.seal(dict(schema=SCHEMA,status=status,
        plan_fingerprint=plan["fingerprint"],source_stage130_fingerprint=SOURCE130,runtime=plan["runtime"],
        reuse_controls=reuse,primary=primary,protection=protection,auc_delta_by_seed=by_seed,development_checks=checks,
        new_full_trajectories=len(new),evaluated_trajectories=len(tables["scope"]),
        new_search_calls=sum(r["physical_calls"] for r in new),table_sha256=hashes,artifact_sha256=evidence,**WORK)))
    check_results(root)
    print(f"Stage-131: {status}. Paired full-video AUC delta {primary['mean_delta']:+.4f}",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76","stage89","stage93","stage112","stage121","stage130")
    for name in names:
        parser.add_argument("--"+name,type=Path)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--check-results",action="store_true")
    parser.set_defaults(mode="stage131",parameter="deit_distilled")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not args.checkpoint or not all(getattr(args,k) for k in names):
        parser.error("Supply the frozen source directories and original checkpoint")
    r76,r89,r93,r112,r121,r130 = paths = [getattr(args,k).expanduser().resolve() for k in names]
    if any(root==p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate Stage-131 directory")
    if not (r121/"manifest.json").is_file() or not (r130/"stage130_decision.json").is_file():
        parser.error("Need the completed Stage-121 and Stage-130 server runs")
    with base.s109.s106.s93.writer_lock(root):
        if (root/"analysis/stage131_decision.json").is_file():
            check_results(root)
            return
        source.check_results(r130)
        d130 = c.c66.sealed(r130/"stage130_decision.json",source.SCHEMA)
        p121 = c.c66.sealed(r121/"manifest.json",s121.SCHEMA)
        d121 = c.c66.sealed(r121/"analysis/stage121_decision.json",s121.SCHEMA)
        if d130["fingerprint"]!=SOURCE130 or d121["fingerprint"]!=SOURCE121 or d121["plan_fingerprint"]!=p121["fingerprint"]:
            raise ValueError("Wrong Stage-121/130 source")
        p112 = c.c66.sealed(r112/"manifest.json",prior.source.common.SCHEMA)
        files = [r112/"manifest.json",r112/"analysis/stage112_decision.json",
                 *(r112/"models").glob("*__generated_4__shared__*.json"),*(r112/"models").glob("generator_*.json")]
        io.verify({str(p):p121["source_sha256"][str(p)] for p in files})
        models,_ = base.load_frozen_models(r112,p112)
        s95 = base.s103.prior.s95
        p93,_,p89,report89,origin = s95.s94.load_run(r93,r76,r89)
        def load_entry(spec):
            return s95.load_unit(r93,r89,p93,p89,report89,origin,spec)[0]

        import torch
        torch.set_num_threads(1)
        old = origin["original"]
        runtime = base.s103.s76.device_runtime(old,torch,args.gpu)
        diffs = base.s109.check_runtime(p121["runtime"],runtime)
        manifest = root/"manifest.json"
        if manifest.is_file() and c.c66.sealed(manifest,SCHEMA)["runtime"]!=runtime:
            raise ValueError("GPU/runtime changed on resume. Set STAGE131_ROOT to a new directory.")
        reuse = not diffs
        jobs = source_jobs(p121)
        if len(jobs)!=196 or len({j["spec"]["unit"]["sequence"] for j in jobs})!=28:
            raise ValueError("Expected 28 complete development videos")
        if reuse:
            links = {str(r121/n):d121["artifact_sha256"][n] for j in jobs if j["arm"] in ARMS[:2]
                     for ext in (".npz",".receipt.json") for n in ["evaluation/"+base.job_key(j)+ext]}
            io.verify(links)
        budget = sum(j["frames"]*(1 if j["arm"]=="native" else 4) for j in jobs if not reuse or j["arm"] in ARMS[2:])
        plan = io.seal(dict(schema=SCHEMA,design=DESIGN,runtime=runtime,runtime_differences=diffs,reuse_controls=reuse,
             jobs=jobs,source_stage130_fingerprint=SOURCE130,source_stage121_fingerprint=SOURCE121,
             code_sha256=c.file_hash(Path(__file__)),maximum_new_search_calls=budget))
        c.publish_identical(manifest,plan)
        frozen = dict(old,fingerprint=plan["fingerprint"],runtime_identity=runtime,runtime=runtime["tracker"])
        native,_ = base.s103.s76.startup.checked_runtime(args,frozen,root)
        observer = base.e63.Capture(native)
        try:
            run(root,r121,p121,plan,old,models,load_entry,observer)
        finally:
            observer.handle.remove()


if __name__=="__main__":
    main()
