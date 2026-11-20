"""Evaluate six independent SGLA candidate blocks and their native continuations."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from experiment_progress import ExperimentProgress
import stage137_shared_crop_layers as probe
import stage156_depth_routing_audit as archives
from stage153_adaptive_depth import DepthObserver
from stage155_adapt_serial_l10 import split_sources
from stage27_uav_common import source_group, tracking_iou


prior, base, c, io = probe.prior, probe.base, probe.c, probe.io
SCHEMA = "stage158_independent_layer_continuation_v1"
STATUS = "COMPLETE_SIX_LAYER_CONTINUATION_DIAGNOSTIC"
LAYERS = probe.LAYERS
HORIZON = 32
SOURCES = ("stage76", "stage89", "stage93", "stage134", "stage137")
TABLES = ("summary", "windows", "choices", "folds")
DEFAULT_BUNDLE = (Path(__file__).resolve().parents[1]/"docs/experiments/06_sgla_uav_tracking/results"
                  /"stage137_results_20261119T123230Z_433433.txt")


def grouped_mean(points, values):
    videos = defaultdict(list)
    for p, value in zip(points, values, strict=True):
        videos[p["sequence"]].append(value)
    return np.stack([np.mean(rows, axis=0) for rows in videos.values()])


def best_layer(values, native=9):
    indices = np.flatnonzero(np.abs(np.asarray(values)-np.max(values)) <= 1e-8)
    return native if LAYERS.index(native) in indices else LAYERS[indices[0]]


def comparison(points, key):
    """Hold out source groups when choosing each fold's best fixed layer."""
    folds = split_sources(sorted({p["sequence"] for p in points}))
    fixed, scope = {}, []
    for fold in folds:
        fit = [p for p in points if source_group(p["sequence"]) in fold["fit"]]
        means = grouped_mean(fit, [p[key] for p in fit]).mean(axis=0)
        layer = best_layer(means)
        fixed.update({g:layer for g in fold["held"]})
        scope.append(dict(fold=fold["fold"], fit="|".join(fold["fit"]), held="|".join(fold["held"]),
                          objective=key, fit_cases=len(fit), layer=layer,
                          **{f"fit_L{k}":float(v) for k,v in zip(LAYERS,means,strict=True)}))
    choices = []
    for p in points:
        native = p["native_layer"]
        arm = {"native":native, **{f"fixed_L{k}":k for k in LAYERS},
               "best_fixed_held_out":fixed[source_group(p["sequence"])],
               "oracle_anchor_iou":best_layer(p["anchor_iou"],native)}
        if key == "tail32":
            n = LAYERS.index(native)
            good, low = p["good32"], p["low32"]
            preserves = (~good[n] | good) & (low[n] | ~low)
            if p["anchor_iou"][n] >= .5:
                preserves &= p["anchor_iou"] >= .5
            arm["oracle_tail32"] = best_layer(p[key],native)
            arm["oracle_tail32_protected"] = best_layer(np.where(preserves,p[key],-1.),native)
        choices.append(arm)
    return choices, scope


def reports(points, key):
    choices, scope = comparison(points, key)
    rows = []
    tracked = key == "tail32"
    for p, arms in zip(points, choices, strict=True):
        native, fixed = (LAYERS.index(arms[k]) for k in ("native","best_fixed_held_out"))
        for arm, layer in arms.items():
            j = LAYERS.index(layer)
            row = dict(case_id=p["case_id"], sequence=p["sequence"], frame=p["frame"], kind=p["kind"],
                       group=p["group"], arm=arm, layer=layer, native_layer=p["native_layer"],
                       anchor_iou=float(p["anchor_iou"][j]), anchor_delta=float(p["anchor_iou"][j]-p["anchor_iou"][native]))
            if tracked:
                row.update(tail8=float(p["tail8"][j]), tail32=float(p["tail32"][j]),
                           tail8_delta=float(p["tail8"][j]-p["tail8"][native]),
                           tail32_delta=float(p["tail32"][j]-p["tail32"][native]),
                           tail32_delta_vs_fixed=float(p["tail32"][j]-p["tail32"][fixed]),
                           new_good_run=bool(p["good32"][j] and not p["good32"][native]),
                           lost_good_run=bool(p["good32"][native] and not p["good32"][j]),
                           new_low_run=bool(p["low32"][j] and not p["low32"][native]))
            rows.append(row)
    summary = []
    for arm in choices[0]:
        for kind in ("all", "onset", "healthy"):
            group = [r for r in rows if r["arm"] == arm and (kind == "all" or r["kind"] == kind)]
            fields = ["anchor_iou", "anchor_delta"]
            if tracked:
                fields += ["tail8", "tail32", "tail8_delta", "tail32_delta", "tail32_delta_vs_fixed"]
            values = grouped_mean(group, [[r[k] for k in fields] for r in group])
            occupancy = Counter(r["layer"] for r in group)
            result = dict(arm=arm, kind=kind, cases=len(group), videos=len(values),
                          good_anchor_boxes=sum(r["anchor_iou"] >= .5 for r in group),
                          **{f"L{k}_count":occupancy[k] for k in LAYERS},
                          **{k:float(v) for k,v in zip(fields,values.mean(axis=0),strict=True)})
            if tracked:
                delta = values[:, fields.index("tail32_delta_vs_fixed")]
                result.update(new_good_runs=sum(r["new_good_run"] for r in group),
                              lost_good_runs=sum(r["lost_good_run"] for r in group),
                              new_low_runs=sum(r["new_low_run"] for r in group),
                              videos_beating_fixed=int((delta > 1e-8).sum()),
                              worst_leave_one_video_out_vs_fixed=float((delta.sum()-delta.max())/(len(delta)-1)))
            summary.append(result)
    return summary, rows, scope


def collect_tails(observer, entry, case, anchors, dataset_root, progress, label, reader=probe.checked_image):
    first = case["frame"]
    ids = np.arange(first, first+HORIZON+1)
    frames = entry["frames"]
    observer.initialize(reader(dataset_root, frames[0]), entry["initial_box"], "native")
    paths = [[] for _ in LAYERS]
    parents = anchors.copy()
    for offset, frame in enumerate(ids[1:], 1):
        image = reader(dataset_root, frames[frame])
        for i, layer in enumerate(LAYERS):
            row = observer.step(image, parents[i], int(frame))
            parents[i] = row["boxes"]
            paths[i].append(row)
            progress.advance(f"{label} | L{layer} branch | successor {offset}/{HORIZON}")
    fields = ("boxes", "parent", "selected", "added", "block_calls", "crop_calls", "head_calls", "selector_calls")
    data = {k:np.asarray([[r[k] for r in path] for path in paths]) for k in fields}
    return dict(data, frame_id=ids)


def audit_trace(data, anchors, case, reference, native, same_runtime, atol):
    shape = (len(LAYERS), HORIZON)
    np.testing.assert_array_equal(data["frame_id"], np.arange(case["frame"],case["frame"]+HORIZON+1))
    boxes = data["boxes"]
    if boxes.shape != (*shape,4) or not np.isfinite(boxes).all() or (boxes[...,2:] <= 0).any():
        raise ValueError("Invalid continuation boxes")
    np.testing.assert_array_equal(data["parent"][:,0], anchors)
    np.testing.assert_array_equal(data["parent"][:,1:], boxes[:,:-1])
    layer = data["selected"]
    if layer.shape != shape or not np.isin(layer,LAYERS).all():
        raise ValueError("Invalid successor layer trace")
    calls = np.zeros((*shape,12),int)
    calls[...,:6] = 1
    np.put_along_axis(calls,layer[...,None]-1,1,axis=2)
    np.testing.assert_array_equal(data["block_calls"],calls)
    np.testing.assert_array_equal(data["added"],np.zeros(shape,int))
    for k in ("crop_calls","head_calls","selector_calls"):
        np.testing.assert_array_equal(data[k],np.ones(shape,int))
    ref = reference["boxes"][0,1:]
    if same_runtime:
        n = LAYERS.index(native)
        np.testing.assert_allclose(boxes[n],ref,atol=atol,rtol=0)
        np.testing.assert_array_equal(layer[n],reference["layer"][0,1:])
    return float(np.abs(boxes[LAYERS.index(native)]-ref).max())


def measure(case, data, anchors, ground_truth, native):
    boxes = np.concatenate((anchors[:,None],data["boxes"]),axis=1)
    iou = tracking_iou(boxes,np.asarray(ground_truth,float)[data["frame_id"]])
    if not np.isfinite(iou).all() or (iou < 0).any():
        raise ValueError(f"Expected labeled Stage-158 window: {case['case_id']}")
    tail = iou[:,1:]
    windows = np.lib.stride_tricks.sliding_window_view(tail,3,axis=1)
    return dict(case,native_layer=native,anchor_iou=iou[:,0],tail8=tail[:,:8].mean(axis=1),tail32=tail.mean(axis=1),
                good32=(windows >= .5).all(-1).any(-1),low32=(windows < .2).all(-1).any(-1))


def publish(root, tables):
    hashes = {}
    for name, rows in tables.items():
        path = root/f"stage158_{name}.csv"
        prior.publish_table(path, rows)
        hashes[path.name] = c.file_hash(path)
    return hashes


def cached_anchors(root, bundle):
    _, tables = archives.bundle(bundle,137)
    roster = tables["stage137_cases.csv"]
    matrix = defaultdict(dict)
    for row in tables["stage137_layers.csv"]:
        matrix[row["case_id"]][int(row["layer"])] = float(row["top1_iou"])
    points = [dict(r, frame=int(r["frame"]),native_layer=int(r["native_layer"]),
                   anchor_iou=np.array([matrix[r["case_id"]][k] for k in LAYERS])) for r in roster]
    summary, choices, folds = reports(points,"anchor_iou")
    publish(root, dict(anchor_summary=summary, anchor_choices=choices, anchor_folds=folds))
    print(f"[Stage-158] {len(points)}/{len(points)} anchors (100.0%) | Stage-137 cached anchor comparison", flush=True)


def check_results(root):
    plan = c.c66.sealed(root/"manifest.json",SCHEMA)
    d = c.c66.sealed(root/"stage158_decision.json",SCHEMA)
    cases = len(plan["cases"])
    if (d["status"] != STATUS or d["plan_fingerprint"] != plan["fingerprint"] or d["cases"] != cases
            or d["successor_searches"] != cases*len(LAYERS)*HORIZON or d["source_stage137"] != archives.FINGERPRINTS[137]
            or len(d["artifact_sha256"]) != 2*cases or set(d["table_sha256"]) != {f"stage158_{k}.csv" for k in TABLES}):
        raise ValueError("Incomplete Stage-158 evaluation")
    io.verify({str(root/k):h for k,h in (d["table_sha256"]|d["artifact_sha256"]).items()})


def run(root, plan, sources, native, load_entry, ap, interval):
    r134, r137, p134, p137 = sources
    cases = plan["cases"]
    same = not plan["runtime_differences"]
    per_case = len(LAYERS)*HORIZON
    out = root/"branches"
    out.mkdir(exist_ok=True)
    cache = {}
    for case in cases:
        cid = case["case_id"]
        cache[cid] = c.read_cache(out/f"{cid}.npz",dict(plan_fingerprint=plan["fingerprint"],case_id=cid))
    reused = sum(v is not None for v in cache.values())
    progress = ExperimentProgress("Stage-158",len(cases)*per_case,reused*per_case,interval)
    progress.report(f"starting | completed cases {reused}/{len(cases)} | six branches per case | anchors "
                    f"{'reused from Stage-137' if same else 'recomputed on current GPU'}")
    entries, points, windows, hashes = {}, [], [], {}
    for index,case in enumerate(cases,1):
        cid, seq = case["case_id"],case["sequence"]
        label = f"case {index}/{len(cases)} {cid}"
        progress.report(f"{label} | {'cached' if cache[cid] is not None else 'preparing shared anchor'}")
        if seq not in entries:
            entries[seq] = load_entry(seq)
        entry = entries[seq]
        ref = c.read_cache(r134/"branches"/f"{cid}.npz",dict(plan_fingerprint=p134["fingerprint"],case_id=cid))
        if ref is None or entry["sequence"] != seq:
            raise ValueError(f"Missing Stage-134 native control for {cid}")
        data = cache[cid]
        if data is None:
            if same:
                anchor = c.read_cache(r137/"observations"/f"{cid}.npz",dict(plan_fingerprint=p137["fingerprint"],case_id=cid))
                if anchor is None:
                    raise ValueError(f"Missing Stage-137 shared crop for {cid}")
            else:
                frames = entry["frames"]
                native.initialize(probe.checked_image(ap["root"],frames[0]),entry["initial_box"])
                anchor = probe.collect(native.rt,probe.checked_image(ap["root"],frames[case["frame"]]),ref["parent"][0,0],case["frame"])
            with ExitStack() as stack:
                observer = DepthObserver(native,stack)
                data = collect_tails(observer,entry,case,anchor["boxes"],ap["root"],progress,label)
        else:
            anchor = {k.removeprefix("anchor__"):v for k,v in data.items() if k.startswith("anchor__")}
        row, _ = probe.score_case(anchor,case,ref,entry["ground_truth"][case["frame"]],ap,same)
        replay_error = audit_trace(data,anchor["boxes"],case,ref,row["native_layer"],same,ap["design"]["box_atol"])
        point = measure(case,data,anchor["boxes"],entry["ground_truth"],row["native_layer"])
        points.append(point)
        for i,layer in enumerate(LAYERS):
            windows.append(dict(case,layer=layer,native_layer=row["native_layer"],
                                anchor_iou=float(point["anchor_iou"][i]),tail8=float(point["tail8"][i]),
                                tail32=float(point["tail32"][i]),good32=bool(point["good32"][i]),
                                low32=bool(point["low32"][i]),source_native_max_box_error=replay_error))
        path = out/f"{cid}.npz"
        if cache[cid] is None:
            packed = data|{f"anchor__{k}":v for k,v in anchor.items()}
            c.save(path,packed,dict(plan_fingerprint=plan["fingerprint"],case_id=cid))
        for item in (path,path.with_suffix(".receipt.json")):
            hashes[item.relative_to(root).as_posix()] = c.file_hash(item)
        progress.report(f"{label} | {'verified cached result' if cache[cid] is not None else 'saved all six branches'}")
    native.assert_frozen()
    if progress.done != progress.total:
        raise ValueError("Incomplete successor search budget")
    progress.report("all branches checked | writing reports")
    summary, choices, folds = reports(points,"tail32")
    tables = publish(root,dict(summary=summary,windows=windows,choices=choices,folds=folds))
    c.publish_identical(root/"stage158_decision.json",io.seal(dict(schema=SCHEMA,status=STATUS,
        plan_fingerprint=plan["fingerprint"],source_stage137=archives.FINGERPRINTS[137],runtime=plan["runtime"],
        runtime_differences=plan["runtime_differences"],cases=len(cases),successor_searches=progress.total,
        new_anchor_crops=0 if same else len(cases),evaluated_anchor_outputs=len(cases)*len(LAYERS),
        training_updates=0,new_full_trajectories=0,confirmation_inference_calls=0,
        table_sha256=tables,artifact_sha256=hashes)))
    check_results(root)
    progress.report("complete | reports verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for k in SOURCES:
        parser.add_argument("--"+k,type=Path)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--progress-interval",type=float,default=10)
    parser.add_argument("--anchor-bundle",type=Path,default=DEFAULT_BUNDLE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cached-anchors",action="store_true")
    mode.add_argument("--check-results",action="store_true")
    parser.set_defaults(mode="stage158_layer_routing",parameter="deit_distilled")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        print(f"Stage-158: {STATUS}",flush=True)
        return
    if args.cached_anchors:
        with prior.writer_lock(root):
            cached_anchors(root,args.anchor_bundle)
        return
    if args.checkpoint is None or any(getattr(args,k) is None for k in SOURCES) or args.progress_interval <= 0:
        parser.error("Need frozen source directories, original checkpoint and a positive progress interval")
    r76,r89,r93,r134,r137 = paths = [getattr(args,k).expanduser().resolve() for k in SOURCES]
    if any(not p.is_dir() or root == p or p in root.parents or root in p.parents for p in paths):
        parser.error("Use existing source directories and a separate Stage-158 directory")
    with prior.writer_lock(root):
        if (root/"stage158_decision.json").is_file():
            check_results(root)
            total = sum(probe.GROUPS.values())*len(LAYERS)*HORIZON
            ExperimentProgress("Stage-158",total,total,args.progress_interval).report("complete | reused verified reports")
            return
        print("[Stage-158] verifying source receipts and frozen runtime",flush=True)
        prior.check_results(r134)
        probe.check_results(r137)
        p134 = c.c66.sealed(r134/"manifest.json",prior.SCHEMA)
        p137 = c.c66.sealed(r137/"manifest.json",probe.SCHEMA)
        d134 = c.c66.sealed(r134/"stage134_decision.json",prior.SCHEMA)
        d137 = c.c66.sealed(r137/"stage137_decision.json",probe.SCHEMA)
        cases = p137["cases"]
        if (d137["fingerprint"] != archives.FINGERPRINTS[137] or d137["source_stage134"] != d134["fingerprint"]
                or p137["runtime"] != p134["runtime"] or Counter(r["group"] for r in cases) != probe.GROUPS
                or any(r["successors"] != HORIZON for r in cases)):
            raise ValueError("Need the matched 47-case Stage-134/137 development archive")
        s95 = base.s103.prior.s95
        p93,_,p89,d89,origin = s95.s94.load_run(r93,r76,r89)
        old = origin["original"]
        if p137["source_roster_fingerprints"] != [p93["fingerprint"],p89["fingerprint"],old["fingerprint"]]:
            raise ValueError("Stage-137 source roster changed")
        specs = {s["unit"]["sequence"]:s for s in p89["units"]}

        def load_entry(seq):
            return s95.load_unit(r93,r89,p93,p89,d89,origin,specs[seq])[0]

        import torch
        torch.set_num_threads(1)
        runtime = base.s103.s76.device_runtime(old,torch,args.gpu)
        diffs = base.s109.check_runtime(p137["runtime"],runtime)
        plan = io.seal(dict(schema=SCHEMA,cases=cases,layers=list(LAYERS),horizon=HORIZON,runtime=runtime,
                            runtime_differences=diffs,source_stage137=d137["fingerprint"],source_stage134=d134["fingerprint"],
                            code_sha256=c.file_hash(Path(__file__))))
        c.publish_identical(root/"manifest.json",plan)
        frozen = dict(old,fingerprint=plan["fingerprint"],runtime_identity=runtime,runtime=runtime["tracker"])
        args.parameter = runtime["tracker"]["parameter"]
        args.checkpoint = args.checkpoint.expanduser().resolve()
        native,_ = base.s103.s76.startup.checked_runtime(args,frozen,root)
        run(root,plan,(r134,r137,p134,p137),native,load_entry,base.e63.c.actor_plan(old),args.progress_interval)
    print(f"Stage-158: {STATUS}",flush=True)


if __name__ == "__main__":
    main()
