"""Trim oversized search crops without discarding their original image content."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import stage127_crop_coverage as audit
from stage27_uav_common import crop_geometry, scene_geometry, tracking_iou

prior, base, c, io = audit.trigger, audit.base, audit.c, audit.io
SCHEMA = "stage128_padding_trim_v1"
STATUS = "COMPLETE_MATCHED_PADDING_TRIM_PILOT"
SOURCE127 = "73a425f26f7b4519285a678838475fa45cbdc17262d9c42fa28c561f3a728092"
DESIGN = dict(points=346, good=.5, adequate_coverage=.9, source_small_side=4.,
              bootstrap_seed=23128, bootstrap_samples=20000)
TABLES = ("points", "crops", "summary", "paired")
WORK = dict(training_updates=0, new_full_trajectories=0, confirmation_inference_calls=0)
OBSERVATIONS = ("candidate_boxes", "features", "crop", "layer", "head_calls", "block_calls", "selector_calls")


def trim_padding(parent, hw, factor, search_size):
    """Keep the original real-pixel rectangle and cap the square at the image long side."""
    crop, usable, _ = crop_geometry(parent,hw,factor,search_size)
    limit = max(map(int,hw))
    if crop[2]<=limit or not (usable[2:]>0).all():
        return np.asarray(parent,dtype=float).copy()
    center = parent[:2]+parent[2:]/2
    # These integer origin bounds contain the old usable rectangle. The original
    # SGLA bottom/right padding rule is also used to verify the realized crop.
    lower = usable[:2]+usable[2:]-limit
    upper = usable[:2]
    origin = np.clip(np.rint(center-limit/2),lower,upper)
    new_center = origin+limit/2
    size = parent[2:]*((limit-.5)/(factor*np.sqrt(parent[2:].prod())))
    changed = np.r_[new_center-size/2,size]
    realized, area, _ = crop_geometry(changed,hw,factor,search_size)
    np.testing.assert_array_equal(realized,np.r_[origin,limit,limit])
    if (area[:2]>usable[:2]).any() or (area[:2]+area[2:]<usable[:2]+usable[2:]).any():
        raise ValueError("Trimming removed original image content")
    return changed


def observe_bank(root, observer, wanted, original, frame, hw, history, ap, read_image, sync, meta, evidence):
    lookup = {tuple(p):i for i,p in enumerate(original["candidate_parents"])} if original is not None else {}
    pending = list(dict.fromkeys(tuple(p) for p in wanted if tuple(p) not in lookup))
    fresh = {}
    if pending:
        stem = f"{meta['sequence']}__s{meta['source_seed']}__f{frame:06d}__{meta['arm']}"
        data = base.s109.cache(root,root/"frames"/f"{stem}.npz",meta,
            lambda:base.s112.observe(observer,read_image(),history,frame,ap,np.asarray(pending),sync),evidence)
        np.testing.assert_array_equal(data["candidate_parents"],pending)
        np.testing.assert_array_equal(data["frame_id"],frame)
        np.testing.assert_array_equal(data["image_hw"],hw)
        np.testing.assert_equal(data["unique_search_calls"],len(pending))
        base.e63.source_common.prior.audit_calls(data,(len(pending),))
        if not np.isfinite(data["decode_error"]).all() or data["decode_error"].max()>ap["design"]["box_atol"]:
            raise ValueError("SGLA decode drift")
        fresh = {p:{k:data[k][i] for k in OBSERVATIONS} for i,p in enumerate(pending)}
    pieces = []
    for parent in wanted:
        key = tuple(parent)
        row = fresh[key] if key in fresh else {k:original[k][lookup[key]] for k in OBSERVATIONS}
        np.testing.assert_array_equal(row["crop"],crop_geometry(parent,hw,ap["search_factor"],ap["search_size"])[0])
        pieces.append(row)
    return {"candidate_parents":wanted, **{k:np.asarray([row[k] for row in pieces]) for k in OBSERVATIONS}},len(pending)


def readouts(ident, banks, choices, gt, hw, ap, past, source_point, calls):
    values, crops = {}, []
    for arm, record in banks.items():
        ious = tracking_iou(record["candidate_boxes"],gt)
        if ious.shape!=(4,) or (ious<0).any():
            raise ValueError("Invalid target or four-candidate bank")
        selected = choices[arm]
        local = []
        for q,parent in enumerate(record["candidate_parents"]):
            geom = scene_geometry(parent,gt,hw,ap["search_factor"],ap["search_size"])
            local.append(dict(ident,arm=arm,slot=q,side=int(geom["crop"][2]),padding_fraction=geom["padding_fraction"],
                coverage=geom["coverage"],input_short_side=float(geom["input_wh"].min()),
                output_iou=float(ious[q]),selected=q==selected))
        crops.extend(local)
        values[arm] = dict(best=float(ious.max()),selected=float(ious[selected]),
            coverage=max(r["coverage"] for r in local),
            short_side=max((r["input_short_side"] for r in local if r["coverage"]>=DESIGN["adequate_coverage"]),
                           default=None))
    before, after = values["original"],values["trim"]
    if after["coverage"]+1e-12<before["coverage"]:
        raise ValueError("Trimmed bank lost source coverage")
    moved = banks["trim"]["crop"][:,:2]+banks["trim"]["crop"][:,2:]/2-banks["original"]["crop"][:,:2]-banks["original"]["crop"][:,2:]/2
    side = source_point["max_short_side_in_adequate_crop"]
    changed = banks["original"]["crop"][:,2]!=banks["trim"]["crop"][:,2]
    return dict(ident, changed_slots=int(changed.sum()), original_slot=choices["original"], trim_slot=choices["trim"],
        max_crop_center_shift=float(np.linalg.norm(moved,axis=1).max()),
        prior_commits_over_image_long_side=past, original_good=before["best"]>=DESIGN["good"],
        trim_good=after["best"]>=DESIGN["good"], best_gain=after["best"]-before["best"],
        selected_gain=after["selected"]-before["selected"], original_calls=calls["original"], trim_calls=calls["trim"],
        source_small_covered=bool(side and float(side)<DESIGN["source_small_side"] and source_point["good_output"]=="False"),
        **{f"{arm}_{k}":v for arm,metrics in values.items() for k,v in metrics.items()}),crops


def summarize(rows):
    summary, paired = [], []
    populations = dict(all=rows, affected=[r for r in rows if r["changed_slots"]],
                       small_covered=[r for r in rows if r["source_small_covered"]])
    for name, points in populations.items():
        videos = sorted({r["sequence"] for r in points})
        summary.append(dict(population=name,points=len(points),videos=len(videos),
            new_good=sum(r["trim_good"] and not r["original_good"] for r in points),
            lost_good=sum(r["original_good"] and not r["trim_good"] for r in points),
            new_selected_good=sum(r["trim_selected"]>=DESIGN["good"]>r["original_selected"] for r in points),
            lost_selected_good=sum(r["original_selected"]>=DESIGN["good"]>r["trim_selected"] for r in points),
            changed_slots=sum(r["changed_slots"] for r in points)))
        for metric in ("best_gain","selected_gain"):
            effects = []
            for sequence in videos:
                unit = defaultdict(list)
                for r in points:
                    if r["sequence"]==sequence:
                        unit[r["source_seed"]].append(r[metric])
                effects.append(float(np.mean([np.mean(v) for v in unit.values()])))
            if effects:
                paired.append(dict(population=name,**base.s103.contrast(
                    effects,"trim_minus_original",metric,DESIGN)))
    return summary, paired


def check_results(root):
    plan = c.c66.sealed(root/"manifest.json",SCHEMA)
    report = c.c66.sealed(root/"stage128_decision.json",SCHEMA)
    if (report["status"]!=STATUS or report["plan_fingerprint"]!=plan["fingerprint"] or plan["design"]!=DESIGN
            or report["source_stage127_fingerprint"]!=SOURCE127 or report["runtime"]!=plan["runtime"]
            or report["row_counts"]["points"]!=DESIGN["points"] or report["row_counts"]["crops"]!=8*DESIGN["points"]
            or not 0<=report["new_search_calls"]<=plan["maximum_new_search_calls"]
            or set(report["table_sha256"])!={f"stage128_{name}.csv" for name in TABLES}
            or any(report[k]!=v for k,v in WORK.items())):
        raise ValueError("Incomplete Stage-128 result")
    io.verify({str(root/k):v for k,v in (report["table_sha256"]|report["cache_sha256"]).items()})


def run(root,r121,p121,p125,r127,plan,old,models,load_entry,observer,sync):
    points, crops, evidence = [], [], {}
    ap = base.e63.c.actor_plan(old)
    stage127 = {(r["sequence"],int(r["source_seed"]),int(r["frame"])):r
                for r in prior.source.read_table(r127/"stage127_points.csv")}
    by_job = defaultdict(list)
    for p in p125["points"]:
        if p["full_window"]:
            by_job[p["sequence"],p["source_seed"]].append(p)
    jobs = prior.reference_jobs(p121)
    for (sequence,seed), group in sorted(by_job.items()):
        job = jobs[sequence,seed]
        entry = load_entry(job["spec"])
        trace = prior.prior.load_trace(r121,p121,job)
        _,scorer = base.job_models(dict(job,arm=prior.prior.CONTROL),models)
        with base.e63.source_experiment.installed(observer,[],job["spec"]["unit"]) as head:
            if head!=p125["native_head_sha256"]:
                raise ValueError("Original SGLA head changed")
            observer.initialize(base.s105.checked_image(ap["root"],entry["frames"][0]),entry["initial_box"])
            for p in group:
                frame = p["frame"]
                i = frame-1
                ident = dict(sequence=sequence,source_seed=seed,frame=frame)
                source_point = stage127[sequence,seed,frame]
                original, _ = audit.inspect(trace,entry,p,ap)
                for k in ("best_output_iou","max_coverage_full_gt"):
                    np.testing.assert_allclose(original[k],float(source_point[k]),atol=1e-12,rtol=0)
                hw = trace["image_hw"][i]
                history = list(prior.prior.history(trace,entry["initial_box"],i,ap["design"]["history"]))
                old_parents = trace["candidate_parents"][i]
                parents = np.array([trim_padding(a,hw,ap["search_factor"],ap["search_size"]) for a in old_parents])
                frame_image = None
                def read_image():
                    nonlocal frame_image
                    if frame_image is None:
                        frame_image = base.s105.checked_image(ap["root"],entry["frames"][frame])
                    return frame_image

                banks, choices, calls = {}, {}, {}
                cached = None if plan["runtime_differences"] else {
                    k:trace[k][i] for k in (*OBSERVATIONS,"candidate_parents")}
                for arm, wanted in (("original",old_parents),("trim",parents)):
                    meta = dict(ident,plan_fingerprint=plan["fingerprint"],arm=arm)
                    bank,calls[arm] = observe_bank(root,observer,wanted,cached,frame,hw,history,ap,read_image,sync,meta,evidence)
                    scores,choices[arm] = base.choose(bank,history,"generated_shared",scorer)
                    if arm=="original":
                        if not plan["runtime_differences"]:
                            np.testing.assert_equal(choices[arm],trace["selected"][i])
                            np.testing.assert_allclose(scores,trace["scores"][i],atol=1e-12,rtol=1e-9)
                        cached = bank
                    banks[arm] = bank
                past = sum(crop_geometry(box,hw,ap["search_factor"],ap["search_size"])[0][2]>max(hw) for box in history)
                row,local = readouts(ident,banks,choices,entry["ground_truth"][frame],hw,ap,int(past),source_point,calls)
                points.append(row)
                crops.extend(local)
        print(f"[Stage-128] {sequence} s{seed}: {len(group)} matched frames",flush=True)
    ids = {(r["sequence"],r["source_seed"],r["frame"]) for r in points}
    if (len(points)!=DESIGN["points"] or ids!=set(stage127)
            or sum(r["source_small_covered"] for r in points)!=226):
        raise ValueError("Frozen population changed")
    summary,paired = summarize(points)
    tables = dict(points=points,crops=crops,summary=summary,paired=paired)
    hashes = {}
    for name,rows in tables.items():
        path = root/f"stage128_{name}.csv"
        base.s103.s86.publish_table(path,rows)
        hashes[path.name] = c.file_hash(path)
    n = sum(r["original_calls"]+r["trim_calls"] for r in points)
    if len(evidence)!=2*sum(bool(r[f"{a}_calls"]) for r in points for a in ("original","trim")):
        raise ValueError("Missing search accounting receipts")
    c.publish_identical(root/"stage128_decision.json",io.seal(dict(schema=SCHEMA,status=STATUS,
        source_stage127_fingerprint=SOURCE127,plan_fingerprint=plan["fingerprint"],runtime=plan["runtime"],
        row_counts={k:len(v) for k,v in tables.items()},new_search_calls=n,table_sha256=hashes,cache_sha256=evidence,
        primary=summary[0], **WORK)))
    check_results(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76","stage89","stage93","stage112","stage121","stage125","stage127")
    for name in names:
        parser.add_argument("--"+name,type=Path)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--check-results",action="store_true")
    parser.set_defaults(mode="stage128",parameter="deit_distilled")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not args.checkpoint or not all(getattr(args,k) for k in names):
        parser.error("Supply the frozen source directories and original checkpoint")
    r76,r89,r93,r112,r121,r125,r127 = paths = [getattr(args,k).expanduser().resolve() for k in names]
    if any(root==p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate Stage-128 directory")
    if not (r127/"stage127_decision.json").is_file():
        parser.error("Need the server Stage-127 coverage report and Stage-121 caches")
    with base.s109.s106.s93.writer_lock(root):
        if (root/"stage128_decision.json").is_file():
            check_results(root)
            return
        d127 = audit.check_results(r127)
        p127 = c.c66.sealed(r127/"manifest.json",audit.SCHEMA)
        if d127["fingerprint"]!=SOURCE127:
            raise ValueError("Wrong Stage-127 source")
        for p in (r121/"manifest.json",r125/"manifest.json"):
            if p127["inputs"].get(str(p))!=c.file_hash(p):
                raise ValueError(f"Manifest changed from Stage-127: {p}")
        p121 = c.c66.sealed(r121/"manifest.json",prior.s121.SCHEMA)
        p125 = c.c66.sealed(r125/"manifest.json",prior.SCHEMA)
        jobs = prior.reference_jobs(p121)
        source_keys = {(p["sequence"],p["source_seed"]) for p in p125["points"] if p["full_window"]}
        traces = [r121/"evaluation"/(base.job_key(jobs[key])+suffix)
                  for key in sorted(source_keys) for suffix in (".npz",".receipt.json")]
        io.verify({str(path):p127["inputs"][str(path)] for path in traces})
        if p125["inputs"].get(str(r112/"manifest.json"))!=c.file_hash(r112/"manifest.json"):
            raise ValueError("Stage-112 model lineage changed")
        p112 = c.c66.sealed(r112/"manifest.json",prior.source.common.SCHEMA)
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
            raise ValueError("GPU/runtime changed on resume. Set STAGE128_ROOT to a new directory.")
        plan = io.seal(dict(schema=SCHEMA,design=DESIGN,runtime=runtime,runtime_differences=diffs,
            code_sha256=c.file_hash(Path(__file__)),source_stage127_fingerprint=SOURCE127,
            source_stage127_plan=p127["fingerprint"],maximum_new_search_calls=DESIGN["points"]*(8 if diffs else 4)))
        c.publish_identical(manifest,plan)
        frozen = dict(old,fingerprint=plan["fingerprint"],runtime_identity=runtime,runtime=runtime["tracker"])
        native,_ = base.s103.s76.startup.checked_runtime(args,frozen,root)
        observer = base.e63.Capture(native)
        try:
            run(root,r121,p121,p125,r127,plan,old,models,load_entry,observer,lambda:torch.cuda.synchronize(args.gpu))
        finally:
            observer.handle.remove()


if __name__=="__main__":
    main()
