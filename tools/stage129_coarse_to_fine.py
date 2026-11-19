"""Use a wide SGLA response to place two fine searches within a four-slot budget."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import stage128_padding_trim as source
from stage27_uav_common import crop_geometry, scene_geometry, tracking_iou

prior, base, c, io = source.prior, source.base, source.c, source.io
SCHEMA = "stage129_coarse_to_fine_search_v1"
STATUS = "COMPLETE_MATCHED_COARSE_TO_FINE_PILOT"
SOURCE128 = "d3cd24867082954a7701dd3789f7de8331c4fc89b1753665fc4ab4e9a64ce6a7"
DESIGN = dict(points=346, good=.5, coverage=.9, zoom_divisor=4, minimum_side=16,
              minimum_peak_distance=.5, bootstrap_seed=23129, bootstrap_samples=20000)
SEARCH_LIMIT = 346*4 + 265*3
TABLES = ("points", "crops", "summary", "paired")
WORK = dict(training_updates=0, new_full_trajectories=0, confirmation_inference_calls=0)


def parent_at(center, side, factor):
    size = (side-.5)/factor
    return np.r_[np.asarray(center,dtype=float)-size/2,[size,size]]


def zoom_layout(parents, original, responses, hw, ap):
    """Use only causal search geometry and the wide response. Labels never enter."""
    changed = np.flatnonzero(np.any(parents!=original,axis=1))
    if not changed.size:
        return parents.copy(), -1, {}, 0
    factor, input_size = ap["search_factor"],ap["search_size"]
    wide = max(map(int,changed),key=lambda q:crop_geometry(original[q],hw,factor,input_size)[0][2])
    retained = {wide,0} if wide else {0,1}
    available = sorted(set(range(4))-retained)
    broad, usable, _ = crop_geometry(parents[wide],hw,factor,input_size)
    side = max(DESIGN["minimum_side"],int(np.ceil(broad[2]/DESIGN["zoom_divisor"])))
    if side>=broad[2]:
        return parents.copy(), -1, {}, 0
    raw = np.asarray(responses[wide])
    if raw.ndim!=2 or not raw.size or not np.isfinite(raw).all():
        raise ValueError("Invalid oversize SGLA response map")
    dy,dx = np.indices(raw.shape)
    centers = broad[:2]+(np.stack((dx,dy),axis=-1).reshape(-1,2)+.5)*broad[2]/np.array(raw.shape[::-1])
    valid = ((centers>=usable[:2]) & (centers<usable[:2]+usable[2:])).all(axis=1)
    order = np.argsort(-raw.ravel(),kind="stable")
    # Exclude all legacy geometry so fallback slots cannot become duplicates.
    used = {tuple(crop_geometry(p,hw,factor,input_size)[0]) for p in parents}
    chosen, result = {}, parents.copy()
    for cell in order[valid[order]]:
        center = centers[cell]
        if any(np.linalg.norm(center-centers[old])<side*DESIGN["minimum_peak_distance"] for old in chosen.values()):
            continue
        parent = parent_at(center,side,factor)
        rect = tuple(crop_geometry(parent,hw,factor,input_size)[0])
        if rect[2]!=side:
            raise ValueError("Zoom crop scale changed")
        if rect in used:
            continue
        slot = available[len(chosen)]
        chosen[slot], result[slot] = int(cell),parent
        used.add(rect)
        if len(chosen)==len(available):
            break
    return result,wide,chosen,side


def report_point(ident, control, treatment, wide, peaks, side, gt, scorer, history, calls, ap, hw, anchor):
    rows, crops = {}, []
    for arm, bank in (("trim",control),("zoom",treatment)):
        iou = tracking_iou(bank["candidate_boxes"],gt)
        if iou.shape!=(4,) or (iou<0).any():
            raise ValueError("Expected valid ground truth and four predictions")
        _,selected = base.choose(bank,history,"generated_shared",scorer)
        local = []
        for q,parent in enumerate(bank["candidate_parents"]):
            geom = scene_geometry(parent,gt,hw,ap["search_factor"],ap["search_size"])
            np.testing.assert_array_equal(geom["crop"],bank["crop"][q])
            local.append(dict(ident,arm=arm,slot=q,selected=q==selected,side=int(geom["crop"][2]),
                input_target_short_side=float(geom["input_wh"].min()),coverage=geom["coverage"],
                iou=float(iou[q]),local_search=arm=="zoom" and q in peaks,
                response_cell=peaks.get(q) if arm=="zoom" else None))
        crops.extend(local)
        fine = [r for r in local if r["local_search"]]
        rows[arm] = dict(best=float(iou.max()),selected=float(iou[selected]),selected_slot=selected,
                         max_coverage=max(r["coverage"] for r in local),
                         fine_coverage=max((r["coverage"] for r in fine),default=None),
                         fine_best=max((r["iou"] for r in fine),default=None))
    t,z = rows["trim"],rows["zoom"]
    return dict(ident,wide_slot=wide,zoom_search_slots=len(peaks),zoom_side=side or None,
        source_small_covered=anchor["source_small_covered"]=="True",
        trim_good=t["best"]>=DESIGN["good"],zoom_good=z["best"]>=DESIGN["good"],
        best_gain=z["best"]-t["best"],selected_gain=z["selected"]-t["selected"],
        trim_control_calls=calls["trim"],zoom_calls=calls["zoom"],
        **{f"{arm}_{k}":v for arm,d in rows.items() for k,v in d.items() if k!="fine_coverage" and k!="fine_best"},
        local_max_coverage=z["fine_coverage"],local_best_iou=z["fine_best"]),crops


def summarize(rows):
    summary,paired = source.summarize([dict(r,changed_slots=r["zoom_search_slots"],original_good=r["trim_good"],
        trim_good=r["zoom_good"],original_selected=r["trim_selected"],trim_selected=r["zoom_selected"]) for r in rows])
    cohorts = (rows,[r for r in rows if r["zoom_search_slots"]],[r for r in rows if r["source_small_covered"]])
    for summary_row, cohort in zip(summary,cohorts):
        labeled = [r for r in cohort if r["oracle_iou"] is not None]
        summary_row.update(frames_with_oracle=len(labeled),
            oracle_good=sum(r["oracle_iou"]>=DESIGN["good"] for r in labeled),
            local_full_coverage=sum(r["local_max_coverage"] is not None and r["local_max_coverage"]>=DESIGN["coverage"]
                                    for r in cohort),
            local_good=sum(r["local_best_iou"] is not None and r["local_best_iou"]>=DESIGN["good"] for r in cohort),
            oracle_good_without_local_coverage=sum(r["oracle_iou"]>=DESIGN["good"]
                and (r["local_max_coverage"] is None or r["local_max_coverage"]<DESIGN["coverage"]) for r in labeled))
    for row in paired:
        row["contrast"] = "zoom_minus_trim"
    return summary,paired


def check_results(root):
    plan = c.c66.sealed(root/"manifest.json",SCHEMA)
    report = c.c66.sealed(root/"stage129_decision.json",SCHEMA)
    if (plan["design"]!=DESIGN or report["status"]!=STATUS or report["plan_fingerprint"]!=plan["fingerprint"]
            or report["source_stage128_fingerprint"]!=SOURCE128 or report["runtime"]!=plan["runtime"]
            or report["row_counts"]["points"]!=DESIGN["points"] or report["row_counts"]["crops"]!=8*DESIGN["points"]
            or not 0<=report["new_search_calls"]<=SEARCH_LIMIT
            or set(report["table_sha256"])!={f"stage129_{n}.csv" for n in TABLES}
            or any(report[k]!=v for k,v in WORK.items())):
        raise ValueError("Incomplete Stage-129 result")
    io.verify({str(root/k):v for k,v in (report["table_sha256"]|report["cache_sha256"]).items()})


def run(root,r121,p121,p125,r128,p128,plan,old,models,load_entry,observer,sync):
    points,crops,evidence = [],[],{}
    ap = base.e63.c.actor_plan(old)
    source_points = {(r["sequence"],int(r["source_seed"]),int(r["frame"])):r
                     for r in prior.source.read_table(r128/"stage128_points.csv")}
    grouped = defaultdict(list)
    for p in p125["points"]:
        if p["full_window"]:
            grouped[p["sequence"],p["source_seed"]].append(p["frame"])
    jobs = prior.reference_jobs(p121)
    same_runtime = plan["runtime"]==p128["runtime"]
    for (seq,seed), frames in sorted(grouped.items()):
        job = jobs[seq,seed]
        entry = load_entry(job["spec"])
        trace = prior.prior.load_trace(r121,p121,job)
        _,scorer = base.job_models(dict(job,arm=prior.prior.CONTROL),models)
        with base.e63.source_experiment.installed(observer,[],job["spec"]["unit"]) as head:
            if head!=p125["native_head_sha256"]:
                raise ValueError("Original SGLA head changed")
            observer.initialize(base.s105.checked_image(ap["root"],entry["frames"][0]),entry["initial_box"])
            for frame in frames:
                i = frame-1
                if int(trace["frame_id"][i])!=frame or not trace["alternate_used"][i]:
                    raise ValueError("Stage-121 trigger identity changed")
                ident = dict(sequence=seq,source_seed=seed,frame=frame)
                meta = dict(ident,plan_fingerprint=plan["fingerprint"])
                stem = f"{seq}__s{seed}__f{frame:06d}"
                anchor = source_points[seq,seed,frame]
                hw = trace["image_hw"][i]
                original = trace["candidate_parents"][i]
                parents = np.array([source.trim_padding(parent,hw,ap["search_factor"],ap["search_size"]) for parent in original])
                np.testing.assert_equal(np.any(parents!=original,axis=1).sum(),int(anchor["changed_slots"]))
                history = list(prior.prior.history(trace,entry["initial_box"],i,ap["design"]["history"]))
                image = None
                def read_image():
                    nonlocal image
                    if image is None:
                        image = base.s105.checked_image(ap["root"],entry["frames"][frame])
                    return image

                control = base.s109.cache(root,root/"frames"/f"{stem}__trim.npz",dict(meta,arm="trim"),
                    lambda:base.s112.observe(observer,read_image(),history,frame,ap,parents,sync),evidence)
                np.testing.assert_array_equal(control["candidate_parents"],parents)
                np.testing.assert_array_equal(control["image_hw"],hw)
                np.testing.assert_equal(control["frame_id"],frame)
                np.testing.assert_equal(control["unique_search_calls"],len({tuple(p) for p in parents}))
                base.e63.source_common.prior.audit_calls(control,(4,))
                if not np.isfinite(control["decode_error"]).all() or control["decode_error"].max()>ap["design"]["box_atol"]:
                    raise ValueError("SGLA decode drift")
                zoom,wide,peaks,side = zoom_layout(parents,original,control["raw"],hw,ap)
                bank,calls = source.observe_bank(root,observer,zoom,control,frame,hw,history,ap,
                    read_image,sync,dict(meta,arm="zoom"),evidence)
                np.testing.assert_equal(calls,len(peaks))
                counts = dict(trim=int(control["unique_search_calls"]),zoom=calls)
                # The entire operational treatment has completed before reading its evaluation label.
                gt = np.asarray(entry["ground_truth"][frame],float)
                row,local = report_point(ident,control,bank,wide,peaks,side,gt,scorer,history,counts,ap,hw,anchor)
                if same_runtime:
                    for k in ("best","selected","max_coverage"):
                        original_key = "trim_coverage" if k=="max_coverage" else "trim_"+k
                        np.testing.assert_allclose(row["trim_"+k],float(anchor[original_key]),atol=1e-9,rtol=1e-9)
                    np.testing.assert_equal(row["trim_selected_slot"],int(anchor["trim_slot"]))
                row.update(oracle_iou=None,oracle_coverage=None,oracle_input_short_side=None,oracle_calls=0)
                if wide>=0:
                    oracle = np.array([parent_at(gt[:2]+gt[2:]/2,side,ap["search_factor"])])
                    evaluated,n = source.observe_bank(root,observer,oracle,None,frame,hw,history,ap,
                        read_image,sync,dict(meta,arm="oracle"),evidence)
                    geo = scene_geometry(oracle[0],gt,hw,ap["search_factor"],ap["search_size"])
                    row.update(oracle_iou=float(tracking_iou(evaluated["candidate_boxes"][0],gt)),
                               oracle_coverage=geo["coverage"],oracle_input_short_side=float(geo["input_wh"].min()),
                               oracle_calls=n)
                points.append(row)
                crops.extend(local)
        print(f"[Stage-129] {seq} s{seed}: {len(frames)} matched frames",flush=True)
    ids = {(r["sequence"],r["source_seed"],r["frame"]) for r in points}
    if len(points)!=DESIGN["points"] or ids!=set(source_points) or sum(r["source_small_covered"] for r in points)!=226:
        raise ValueError("Frozen Stage-128 population changed")
    summary,paired = summarize(points)
    tables = dict(points=points,crops=crops,summary=summary,paired=paired)
    hashes = {}
    for name,rows in tables.items():
        path = root/f"stage129_{name}.csv"
        base.s103.s86.publish_table(path,rows)
        hashes[path.name] = c.file_hash(path)
    calls = sum(r["trim_control_calls"]+r["zoom_calls"]+r["oracle_calls"] for r in points)
    if len(evidence)!=2*sum(1+bool(r["zoom_calls"])+bool(r["oracle_calls"]) for r in points):
        raise ValueError("Missing search accounting receipts")
    c.publish_identical(root/"stage129_decision.json",io.seal(dict(schema=SCHEMA,status=STATUS,
        source_stage128_fingerprint=SOURCE128,plan_fingerprint=plan["fingerprint"],runtime=plan["runtime"],
        row_counts={k:len(v) for k,v in tables.items()},new_search_calls=calls,table_sha256=hashes,cache_sha256=evidence,
        primary=summary[0],**WORK)))
    check_results(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76","stage89","stage93","stage112","stage121","stage125","stage127","stage128")
    for name in names:
        parser.add_argument("--"+name,type=Path)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--check-results",action="store_true")
    parser.set_defaults(mode="stage129",parameter="deit_distilled")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not args.checkpoint or not all(getattr(args,k) for k in names):
        parser.error("Supply the frozen source directories and original checkpoint")
    r76,r89,r93,r112,r121,r125,r127,r128 = paths = [getattr(args,k).expanduser().resolve() for k in names]
    if any(root==p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate Stage-129 directory")
    if not (r128/"stage128_decision.json").is_file():
        parser.error("Need the server Stage-128 matched result and original images")
    with base.s109.s106.s93.writer_lock(root):
        if (root/"stage129_decision.json").is_file():
            check_results(root)
            return
        source.check_results(r128)
        p128 = c.c66.sealed(r128/"manifest.json",source.SCHEMA)
        d128 = c.c66.sealed(r128/"stage128_decision.json",source.SCHEMA)
        p127 = c.c66.sealed(r127/"manifest.json",source.audit.SCHEMA)
        if d128["fingerprint"]!=SOURCE128 or p128["source_stage127_plan"]!=p127["fingerprint"]:
            raise ValueError("Need the archived Stage-128/127 result")
        for p in (r121/"manifest.json",r125/"manifest.json"):
            if p127["inputs"].get(str(p))!=c.file_hash(p):
                raise ValueError(f"Manifest changed from Stage-127: {p}")
        p121 = c.c66.sealed(r121/"manifest.json",prior.s121.SCHEMA)
        p125 = c.c66.sealed(r125/"manifest.json",prior.SCHEMA)
        jobs = prior.reference_jobs(p121)
        keys = {(p["sequence"],p["source_seed"]) for p in p125["points"] if p["full_window"]}
        traces = [r121/"evaluation"/(base.job_key(jobs[key])+suffix)
                  for key in sorted(keys) for suffix in (".npz",".receipt.json")]
        io.verify({str(path):p127["inputs"][str(path)] for path in traces})
        if p125["inputs"].get(str(r112/"manifest.json"))!=c.file_hash(r112/"manifest.json"):
            raise ValueError("Stage-112 scorer lineage changed")
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
            raise ValueError("GPU/runtime changed on resume. Set STAGE129_ROOT to a new directory.")
        plan = io.seal(dict(schema=SCHEMA,design=DESIGN,runtime=runtime,runtime_differences=diffs,
            code_sha256=c.file_hash(Path(__file__)),source_stage128_fingerprint=SOURCE128))
        c.publish_identical(manifest,plan)
        frozen = dict(old,fingerprint=plan["fingerprint"],runtime_identity=runtime,runtime=runtime["tracker"])
        native,_ = base.s103.s76.startup.checked_runtime(args,frozen,root)
        observer = base.e63.Capture(native)
        try:
            run(root,r121,p121,p125,r128,p128,plan,old,models,load_entry,observer,lambda:torch.cuda.synchronize(args.gpu))
        finally:
            observer.handle.remove()


if __name__=="__main__":
    main()
