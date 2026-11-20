"""Fit matched current-IoU and future-IoU six-way SGLA selectors."""

import argparse
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
import stage155_adapt_serial_l10 as source
import stage158_layer_routing as prior
from stage27_uav_common import source_group, tracking_iou
from stage28_response_common import cpu_array


base, c, io = prior.base, prior.c, prior.io
probe, LAYERS, HORIZON = prior.probe, prior.LAYERS, prior.HORIZON
SCHEMA = "stage159_sgla_layer_supervision_v1"
STATUS = "COMPLETE_SIX_WAY_SELECTOR_SUPERVISION_PILOT"
SOURCES = ("stage76", "stage89", "stage93", "stage121")
DESIGN = dict(uniform=8, historical_low=4, minimum_spacing=12, history=4, projection=16, projection_seed=23159,
              ridge=1., successor_frames=HORIZON)
OBJECTIVES = {"anchor_supervised":"anchor_iou", "future_supervised":"tail32"}
TABLES = ("summary", "windows", "choices", "fit", "paired", "support")
HISTORY_FEATURES = 8
ANCHOR_WORK = 1 + len(LAYERS)*HORIZON
FIT_WORK = 2*len(OBJECTIVES)
POST_WORK = FIT_WORK+1


def sample_frames(truth, peaks):
    truth = np.asarray(truth,float)
    if len(truth) != len(peaks)+1 or not np.isfinite(peaks).all():
        raise ValueError("Invalid chronological native trajectory")
    valid = np.isfinite(truth).all(axis=1)&(truth[:,2:] > 0).all(axis=1)
    windows = np.lib.stride_tricks.sliding_window_view(valid,HORIZON+1).all(axis=1)
    candidates = [int(f) for f in np.flatnonzero(windows) if f > DESIGN["history"]]
    if not candidates:
        raise ValueError("No complete 32-frame labeled window")
    choices = {}

    def admissible(f):
        return all(abs(f-old) >= DESIGN["minimum_spacing"] for old in choices)

    for target in np.linspace(candidates[0],candidates[-1],DESIGN["uniform"]):
        available = [f for f in candidates if admissible(f)]
        if available:
            choices[min(available,key=lambda f:(abs(f-target),f))] = "uniform"
    h = DESIGN["history"]
    hard = sorted(candidates,key=lambda f:(np.mean(peaks[f-1-h:f-1]),f))
    for f in hard:
        if admissible(f):
            choices[f] = "historical_low"
            if sum(k == "historical_low" for k in choices.values()) == DESIGN["historical_low"]:
                break
    return sorted(choices.items())


def history_features(boxes, peaks, frame):
    end = frame-1
    recent = boxes[end-DESIGN["history"]:end]
    confidence = peaks[end-DESIGN["history"]:end]
    centers = recent[:,:2]+.5*recent[:,2:]
    shift = np.linalg.norm(np.diff(centers,axis=0),axis=1)/np.sqrt(recent[:-1,2:].prod(axis=1))
    scale = np.abs(np.diff(np.log(recent[:,2:]),axis=0)).mean(axis=1)
    return np.array([confidence[-1],confidence.mean(),confidence.std(),confidence[-1]-confidence[0],
                     shift[-1],shift.mean(),scale[-1],scale.mean()],float)


def prepare(r121, p121, d121, jobs, load_entry, ap):
    cases, refs, entries, horizons = [], {}, {}, {}
    for job in jobs:
        entry = load_entry(job["spec"])
        seq = entry["sequence"]
        entries[seq] = entry
        key = f"evaluation/{base.job_key(job)}"
        io.verify({str(r121/k):d121["artifact_sha256"][k] for k in (key+".npz",key+".receipt.json")})
        trace = source.source.prior.prior.load_trace(r121,p121,job)
        boxes = trace["boxes"]
        np.testing.assert_array_equal(trace["frame_id"],np.arange(1,len(boxes)+1))
        np.testing.assert_array_equal(trace["candidate_parents"][:,0],np.vstack((entry["initial_box"],boxes[:-1])))
        peaks = np.max(trace["raw"][:,0]*np.asarray(ap["window"],np.float32),axis=(1,2))
        picks = sample_frames(entry["ground_truth"],peaks)
        horizons[seq] = Counter(kind for _,kind in picks)
        for frame,kind in picks:
            i = frame-1
            cid = f"{seq}__{frame:06d}"
            cases.append(dict(case_id=cid,sequence=seq,frame=frame,kind=kind,group=kind,successors=HORIZON))
            segment = slice(i,i+HORIZON+1)
            refs[cid] = dict(boxes=boxes[segment].copy()[None], layer=trace["layer"][segment,0].copy()[None],
                             parent=trace["candidate_parents"][i,0].copy(), image_hw=trace["image_hw"][i].copy(),
                             crop_sha256=str(trace["crop_sha256"][i,0]), history=history_features(boxes,peaks,frame))
    return cases, refs, entries, horizons


def projected_prefix(rt, pool):
    flat = cpu_array(rt.pool(pool,grid=2))[0].astype(float)
    rng = np.random.default_rng(DESIGN["projection_seed"])
    projection = rng.standard_normal((len(flat),DESIGN["projection"]))/np.sqrt(len(flat))
    return flat@projection


def anchor_observation(native, entry, case, ref, root, reader=probe.checked_image):
    rt = native.rt
    frames = entry["frames"]
    native.initialize(reader(root,frames[0]),entry["initial_box"])
    prefix = []
    handle = rt.backbone.blocks[5].register_forward_hook(lambda _, args, out: prefix.append(projected_prefix(rt,out)))
    try:
        result = probe.collect(rt,reader(root,frames[case["frame"]]),ref["parent"],case["frame"])
    finally:
        handle.remove()
    if len(prefix) != 1:
        raise ValueError("Expected a single shared L6 representation")
    return dict(anchor_boxes=result["boxes"],anchor_parent=result["parent"],anchor_layer=result["selected"],
                anchor_crop=result["crop_sha256"],anchor_hw=result["image_hw"],anchor_calls=result["counts"],
                features=np.r_[prefix[0],ref["history"]])


def audit_exposure(data, ref, case, same_runtime, atol):
    anchors = data["anchor_boxes"]
    if anchors.shape != (len(LAYERS),4) or not np.isfinite(anchors).all() or (anchors[:,2:] <= 0).any():
        raise ValueError("Invalid anchor candidate boxes")
    np.testing.assert_array_equal(data["anchor_calls"],[1]*13+[6])
    np.testing.assert_array_equal(data["anchor_parent"],ref["parent"])
    np.testing.assert_array_equal(data["anchor_hw"],ref["image_hw"])
    if str(data["anchor_crop"]) != ref["crop_sha256"]:
        raise ValueError("Native crop differs from Stage-121")
    if data["features"].shape != (DESIGN["projection"]+HISTORY_FEATURES,) or not np.isfinite(data["features"]).all():
        raise ValueError("Invalid L6 or causal features")
    np.testing.assert_array_equal(data["features"][-HISTORY_FEATURES:],ref["history"])
    layer = int(data["anchor_layer"])
    if same_runtime:
        if layer != int(ref["layer"][0,0]):
            raise ValueError("Native selected layer differs")
        np.testing.assert_allclose(anchors[LAYERS.index(layer)],ref["boxes"][0,0],atol=atol,rtol=0)
    prior.audit_trace(data,anchors,case,ref,layer,same_runtime,atol)


def ridge(features, targets, weights):
    mu = np.average(features,axis=0,weights=weights)
    scale = np.sqrt(np.average((features-mu)**2,axis=0,weights=weights))
    scale = np.where(scale > 1e-9,scale,1.)
    x = (features-mu)/scale
    bias = np.average(targets,axis=0,weights=weights)
    weighted = weights[:,None]*x
    coef = np.linalg.solve(x.T@weighted+DESIGN["ridge"]*weights.sum()*np.eye(x.shape[1]),weighted.T@(targets-bias))
    return mu, scale, bias, coef


def train_selectors(points, folds, progress):
    choices = [{} for _ in points]
    models, fit_rows = {}, []
    x = np.stack([p["features"] for p in points])
    groups = [source_group(p["sequence"]) for p in points]
    native = np.array([LAYERS.index(p["native_layer"]) for p in points])
    for fold in folds:
        i = fold["fold"]
        fit = np.isin(groups,fold["fit"])
        held = np.isin(groups,fold["held"])
        if not np.array_equal(fit,~held) or not fit.any() or not held.any():
            raise ValueError("Source folds overlap or have no support")
        counts = Counter(g for g,use in zip(groups,fit) if use)
        weights = np.array([1./counts[g] for g,use in zip(groups,fit) if use])
        for arm,target in OBJECTIVES.items():
            y = np.stack([p[target] for p in points])
            relative = y-y[np.arange(len(points)),native,None]
            mu,scale,bias,coef = ridge(x[fit],relative[fit],weights)
            standardized = (x[held]-mu)/scale
            pred = standardized@coef+bias
            if not np.isfinite(pred).all():
                raise ValueError("Invalid held-out routing score")
            for j,score in zip(np.flatnonzero(held),pred,strict=True):
                choices[j][arm] = prior.best_layer(score,points[j]["native_layer"])
            models[f"fold{i}_{arm}"] = dict(mean=mu.tolist(),scale=scale.tolist(),bias=bias.tolist(),coef=coef.tolist())
            fit_rows.append(dict(fold=i,arm=arm,target=target,fit_sources=len(counts),fit_cases=int(fit.sum()),
                                 held_cases=int(held.sum()),ridge=DESIGN["ridge"],features=x.shape[1],parameters=coef.size+bias.size))
            progress.advance()
    if any(set(r) != set(OBJECTIVES) for r in choices):
        raise ValueError("Incomplete held-out selector predictions")
    return choices, models, fit_rows


def paired(choices, points):
    per_arm = {(r["case_id"],r["arm"]):r for r in choices}
    output = []
    for control in ("native","best_fixed_held_out","anchor_supervised"):
        for kind in ("all","native_stable"):
            cases = [p for p in points if kind == "all" or p["native_stable"]]
            if not cases:
                continue
            test = [per_arm[p["case_id"],"future_supervised"] for p in cases]
            ref = [per_arm[p["case_id"],control] for p in cases]
            deltas = prior.grouped_mean(cases,[a["tail32"]-b["tail32"] for a,b in zip(test,ref,strict=True)])
            output.append(dict(contrast=f"future_supervised_minus_{control}",kind=kind,cases=len(cases),videos=len(deltas),
                               mean_tail32_delta=float(deltas.mean()),wins=int((deltas>1e-8).sum()),
                               ties=int((np.abs(deltas)<=1e-8).sum()),losses=int((deltas< -1e-8).sum()),
                               worst_leave_one_out=float((deltas.sum()-deltas.max())/(len(deltas)-1)) if len(deltas)>1 else None,
                               new_low_vs_native=sum(r["new_low_run"] for r in test),lost_recovery_vs_native=sum(r["lost_good_run"] for r in test)))
    return output


def run(root, plan, refs, entries, sample_counts, native, ap, interval):
    out = root/"branches"
    out.mkdir(exist_ok=True)
    cases = plan["cases"]
    paths, cache = {}, {}
    for case in cases:
        cid = case["case_id"]
        paths[cid] = out/f"{cid}.npz"
        cache[cid] = c.read_cache(paths[cid],dict(plan_fingerprint=plan["fingerprint"],case_id=cid))
    reused = sum(x is not None for x in cache.values())
    progress = OverallProgress("Stage-159",len(cases)*ANCHOR_WORK+POST_WORK,reused*ANCHOR_WORK,interval,"work")
    progress.report("Collecting")
    points, windows, hashes = [], [], {}
    same = not plan["runtime_differences"]
    atol = ap["design"]["box_atol"]
    for case in cases:
        cid, seq, frame = case["case_id"],case["sequence"],case["frame"]
        entry, ref = entries[seq],refs[cid]
        path, data = paths[cid],cache[cid]
        fresh = data is None
        if fresh:
            anchor = anchor_observation(native,entry,case,ref,ap["root"])
            progress.advance()
            with ExitStack() as stack:
                observer = prior.DepthObserver(native,stack)
                data = prior.collect_tails(observer,entry,case,anchor["anchor_boxes"],ap["root"],progress,"")|anchor
        audit_exposure(data,ref,case,same,atol)
        layer = int(data["anchor_layer"])
        point = prior.measure(case,data,data["anchor_boxes"],entry["ground_truth"],layer)
        n = LAYERS.index(layer)
        gt = np.asarray(entry["ground_truth"],float)
        stable = point["anchor_iou"][n] >= .5 and (tracking_iou(data["boxes"][n],gt[frame+1:frame+HORIZON+1])>=.5).all()
        points.append(dict(point,features=data["features"],native_stable=bool(stable)))
        for i,k in enumerate(LAYERS):
            windows.append(dict(case,layer=k,native_layer=layer,native_stable=bool(stable),anchor_iou=float(point["anchor_iou"][i]),
                                tail8=float(point["tail8"][i]),tail32=float(point["tail32"][i]),
                                good32=bool(point["good32"][i]),low32=bool(point["low32"][i])))
        if fresh:
            c.save(path,data,dict(plan_fingerprint=plan["fingerprint"],case_id=cid))
        for item in (path,path.with_suffix(".receipt.json")):
            hashes[item.relative_to(root).as_posix()] = c.file_hash(item)
    native.assert_frozen()
    progress.report("Fitting")
    policies, models, fit = train_selectors(points,plan["folds"],progress)
    progress.report("Evaluating")
    summary, choices, fixed = prior.reports(points,"tail32",policies,kinds=("all","uniform","historical_low"))
    support = [dict(sequence=seq,uniform=n["uniform"],historical_low=n["historical_low"],
                    native_stable=sum(p["native_stable"] for p in points if p["sequence"] == seq))
               for seq,n in sample_counts.items()]
    for row in fit:
        row["best_fixed_layer"] = fixed[row["fold"]]["layer"]
    tables = prior.publish(root,dict(summary=summary,windows=windows,choices=choices,fit=fit,
                                    paired=paired(choices,points),support=support),stage="stage159")
    model_path = root/"stage159_models.json"
    c.publish_identical(model_path,io.seal(dict(schema=SCHEMA,plan_fingerprint=plan["fingerprint"],models=models)))
    hashes[model_path.name] = c.file_hash(model_path)
    c.publish_identical(root/"stage159_decision.json",io.seal(dict(schema=SCHEMA,status=STATUS,
        plan_fingerprint=plan["fingerprint"],source_stage121=plan["source_stage121"],runtime=plan["runtime"],
        runtime_differences=plan["runtime_differences"],cases=len(cases),videos=len(entries),
        successor_searches=len(cases)*len(LAYERS)*HORIZON,fitted_models=len(models),work=progress.total,
        new_full_trajectories=0,confirmation_inference_calls=0,table_sha256=tables,artifact_sha256=hashes)))
    check_results(root)
    progress.phase = "Complete"
    progress.advance()
    if progress.done != progress.total:
        raise ValueError("Incomplete progress accounting")
    progress.report()


def check_results(root):
    plan = c.c66.sealed(root/"manifest.json",SCHEMA)
    d = c.c66.sealed(root/"stage159_decision.json",SCHEMA)
    n = len(plan["cases"])
    if (d["status"] != STATUS or d["plan_fingerprint"] != plan["fingerprint"] or d["cases"] != n
            or d["source_stage121"] != source.source.SOURCE121 or d["videos"] != 28
            or d["successor_searches"] != n*len(LAYERS)*HORIZON or d["work"] != n*ANCHOR_WORK+POST_WORK
            or d["fitted_models"] != FIT_WORK or len(d["artifact_sha256"]) != 2*n+1
            or set(d["table_sha256"]) != {f"stage159_{k}.csv" for k in TABLES}):
        raise ValueError("Incomplete Stage-159 result")
    io.verify({str(root/k):h for k,h in (d["table_sha256"]|d["artifact_sha256"]).items()})
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for k in SOURCES:
        parser.add_argument("--"+k,type=Path)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--progress-interval",type=float,default=15)
    parser.add_argument("--check-results",action="store_true")
    parser.set_defaults(mode="stage159_layer_selector",parameter="deit_distilled")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if args.checkpoint is None or any(getattr(args,k) is None for k in SOURCES) or args.progress_interval <= 0:
        parser.error("Need the frozen source directories, original checkpoint and positive progress interval")
    r76,r89,r93,r121 = paths = [getattr(args,k).expanduser().resolve() for k in SOURCES]
    if any(not p.is_dir() or root == p or p in root.parents or root in p.parents for p in paths):
        parser.error("Use existing source directories and a separate Stage-159 directory")
    with prior.prior.writer_lock(root):
        if (root/"stage159_decision.json").is_file():
            work = check_results(root)["work"]
            OverallProgress("Stage-159",work,work,args.progress_interval,"work").report("Complete")
            return
        p121 = c.c66.sealed(r121/"manifest.json",source.source.s121.SCHEMA)
        d121 = c.c66.sealed(r121/"analysis/stage121_decision.json",source.source.s121.SCHEMA)
        if d121["fingerprint"] != source.source.SOURCE121 or d121["plan_fingerprint"] != p121["fingerprint"]:
            raise ValueError("Need the Stage-121 development native trajectories")
        jobs = sorted((j for j in p121["jobs"] if j["arm"] == "native"),key=base.job_key)
        if len(jobs) != 28 or sum(j["frames"] for j in jobs) != source.DESIGN["tracked_frames"]:
            raise ValueError("Stage-121 native trajectory roster changed")
        s95 = base.s103.prior.s95
        p93,_,p89,d89,origin = s95.s94.load_run(r93,r76,r89)
        old = origin["original"]
        ap = base.e63.c.actor_plan(old)

        def load_entry(spec):
            return s95.load_unit(r93,r89,p93,p89,d89,origin,spec)[0]

        cases, refs, entries, counts = prepare(r121,p121,d121,jobs,load_entry,ap)
        folds = source.split_sources(entries)
        import torch
        torch.set_num_threads(1)
        runtime = base.s103.s76.device_runtime(old,torch,args.gpu)
        diffs = base.s109.check_runtime(p121["runtime"],runtime)
        code = (Path(__file__),Path(prior.__file__),Path(__file__).with_name("experiment_progress.py"))
        plan = io.seal(dict(schema=SCHEMA,design=DESIGN,cases=cases,folds=folds,layers=list(LAYERS),
                            source_stage121=d121["fingerprint"],runtime=runtime,runtime_differences=diffs,
                            code_sha256={p.name:c.file_hash(p) for p in code}))
        c.publish_identical(root/"manifest.json",plan)
        snapshot = dict(old,fingerprint=plan["fingerprint"],runtime_identity=runtime,runtime=runtime["tracker"])
        args.parameter = runtime["tracker"]["parameter"]
        args.checkpoint = args.checkpoint.expanduser().resolve()
        native,_ = base.s103.s76.startup.checked_runtime(args,snapshot,root)
        run(root,plan,refs,entries,counts,native,ap,args.progress_interval)


if __name__ == "__main__":
    main()
