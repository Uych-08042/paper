"""Calibrate when the Stage-159 selector should keep the native SGLA layer."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
import stage159_layer_selector as trained
from stage27_uav_common import source_group


prior, c, io = trained.prior, trained.c, trained.io
LAYERS = trained.LAYERS
SOURCE159 = "d82414dfb4ca044e71bb8e24c0fdab3c364b5e04dca29e2951ae885dd6825a50"
SCHEMA = "stage160_native_layer_veto_v1"
EPSILON = 1e-8


def collect_points(windows, actions):
    by_case = defaultdict(dict)
    for r in windows:
        cid, layer = r["case_id"],int(r["layer"])
        if layer in by_case[cid]:
            raise ValueError("Duplicate layer outcome")
        by_case[cid][layer] = r
    selected = {r["case_id"]:int(r["layer"]) for r in actions if r["arm"] == "future_supervised"}
    if len(by_case) != 328 or set(by_case) != set(selected):
        raise ValueError("Expected the matched Stage-159 development roster")
    points = []
    for cid, candidates in by_case.items():
        order = [candidates[k] for k in LAYERS]
        first = order[0]
        item = {k:first[k] for k in ("case_id","sequence","kind")}
        for key in ("sequence","native_layer","native_stable","frame","kind"):
            if any(r[key] != first[key] for r in order):
                raise ValueError(f"Divergent {key} at {cid}")
        points.append(dict(item,frame=int(first["frame"]),native_layer=int(first["native_layer"]),
                           native_stable=first["native_stable"]=="True",future=selected[cid],
                           **{k:np.array([float(r[k]) for r in order]) for k in ("anchor_iou","tail8","tail32")},
                           **{k:np.array([r[k]=="True" for r in order]) for k in ("good32","low32")}))
    return points


def effects(point, layer):
    n, j = (LAYERS.index(i) for i in (point["native_layer"],layer))
    good, low = point["good32"],point["low32"]
    return dict(anchor=float(point["anchor_iou"][j]),tail8=float(point["tail8"][j]),tail32=float(point["tail32"][j]),
                delta32=float(point["tail32"][j]-point["tail32"][n]),
                new_good=bool(good[j] and not good[n]),lost_good=bool(good[n] and not good[j]),
                new_low=bool(low[j] and not low[n]),switched=bool(j != n))


def gate(point, candidate, margin, threshold):
    return candidate if margin > threshold+EPSILON else point["native_layer"]


def summarize(points, actions):
    rows = []
    regions = ("all","uniform","historical_low","native_stable")
    for method,layers in actions.items():
        for region in regions:
            selected = [(p,effects(p,k),effects(p,p["future"])) for p,k in zip(points,layers,strict=True)
                        if region in ("all",p["kind"]) or region == "native_stable" and p["native_stable"]]
            if not selected:
                continue
            stats = prior.grouped_mean([p for p,_,_ in selected],[[e["tail32"],e["delta32"],e["tail32"]-baseline["tail32"]]
                                          for _,e,baseline in selected])
            n = len(stats)
            deltas = stats[:,1]
            rows.append(dict(arm=method,group=region,cases=len(selected),videos=n,tail32=float(stats[:,0].mean()),
                             delta_vs_native=float(deltas.mean()),delta_vs_stage159=float(stats[:,2].mean()),
                             worst_leave_one_out_vs_native=float((deltas.sum()-deltas.max())/(n-1)) if n>1 else None,
                             wins_vs_native=int((deltas>EPSILON).sum()),ties_vs_native=int((np.abs(deltas)<=EPSILON).sum()),
                             losses_vs_native=int((deltas< -EPSILON).sum()),
                             **{f"{k}_count":sum(e[k] for _,e,_ in selected) for k in ("switched","new_good","lost_good","new_low")}))
    return rows


def pick_threshold(points, layers, margin):
    """Fit a single conservative threshold using only inner-held predictions."""
    gains = np.array([effects(p,k)["delta32"] for p,k in zip(points,layers,strict=True)])
    harmful = np.array([effects(p,k)["lost_good"] or effects(p,k)["new_low"] for p,k in zip(points,layers,strict=True)])
    groups = [source_group(p["sequence"]) for p in points]
    n = Counter(groups)
    weights = np.array([1/n[g]/len(n) for g in groups])
    stats = []
    for threshold in sorted({0.,*(float(a) for a in margin if a>EPSILON)}):
        accept = margin > threshold+EPSILON
        stats.append(dict(threshold=threshold,switches=int(accept.sum()),mean_tail32_delta=float(weights@(accept*gains)),
                          new_low_or_lost=int((accept&harmful).sum())))
    accepted = max((r for r in stats if r["new_low_or_lost"]==0),key=lambda r:(r["mean_tail32_delta"],r["threshold"]))
    return accepted, stats


def predictions(points, scores):
    if not np.isfinite(scores).all():
        raise ValueError("Invalid routing score")
    layers = [prior.best_layer(s,p["native_layer"]) for p,s in zip(points,scores,strict=True)]
    margin = np.array([s[LAYERS.index(k)]-s[LAYERS.index(p["native_layer"])]
                       for p,k,s in zip(points,layers,scores,strict=True)])
    return layers,margin


def calibrated(points, features, folds, models, progress):
    groups = np.array([source_group(p["sequence"]) for p in points])
    targets = np.stack([p["tail32"]-p["tail32"][LAYERS.index(p["native_layer"])] for p in points])
    margins, positive, chosen = np.full(len(points),np.nan), [None]*len(points), [None]*len(points)
    details, calibration = [], []
    for fold in folds:
        i = fold["fold"]
        outer_fit = np.isin(groups,fold["fit"])
        held = np.isin(groups,fold["held"])
        if not np.array_equal(outer_fit,~held) or not held.any():
            raise ValueError("Invalid outer source partition")
        inner_scores = np.full_like(targets,np.nan)
        for group in fold["fit"]:
            val = outer_fit&(groups==group)
            fit = outer_fit&~val
            counts = Counter(groups[fit])
            weights = np.array([1./counts[g] for g in groups[fit]])
            mean,scale,bias,coef = trained.ridge(features[fit],targets[fit],weights)
            inner_scores[val] = ((features[val]-mean)/scale)@coef+bias
            progress.advance()
        fit_points = [p for p,include in zip(points,outer_fit) if include]
        inner_layer,inner_gap = predictions(fit_points,inner_scores[outer_fit])
        selected, curve = pick_threshold(fit_points,inner_layer,inner_gap)
        threshold = selected["threshold"]
        details.append(dict(fold=i,fit_sources=len(fold["fit"]),fit_cases=int(outer_fit.sum()),held_cases=int(held.sum()),**selected))
        calibration.extend(dict(fold=i,chosen=r is selected,**r) for r in curve)
        model = {k:np.asarray(v,float) for k,v in models[f"fold{i}_future_supervised"].items()}
        scores = ((features[held]-model["mean"])/model["scale"])@model["coef"]+model["bias"]
        layers,margin = predictions([p for p,include in zip(points,held) if include],scores)
        for j,k,gap in zip(np.flatnonzero(held),layers,margin,strict=True):
            if k != points[j]["future"]:
                raise ValueError(f"Archived Stage-159 selector did not replay at {points[j]['case_id']}")
            margins[j] = gap
            positive[j] = gate(points[j],k,gap,0.)
            chosen[j] = gate(points[j],k,gap,threshold)
    if not np.isfinite(margins).all() or any(k is None for k in chosen):
        raise ValueError("Incomplete held-out choices")
    return dict(positive_margin=positive,calibrated_margin=chosen),margins,details,calibration


def reports(root, points, extra, margins=None, fit=None, curve=None):
    actions = {"native":[p["native_layer"] for p in points],"future_supervised":[p["future"] for p in points]}
    protected = []
    for p in points:
        e = effects(p,p["future"])
        protected.append(p["future"] if e["delta32"]>EPSILON and not (e["lost_good"] or e["new_low"]) else p["native_layer"])
    actions["hindsight_veto"] = protected
    actions.update(extra)
    summary = summarize(points,actions)
    choices = [dict(case_id=p["case_id"],sequence=p["sequence"],frame=p["frame"],kind=p["kind"],
                    native_stable=p["native_stable"],native_layer=p["native_layer"],arm=arm,layer=k,
                    stage159_layer=p["future"],predicted_margin=float(margins[i]) if margins is not None else None,
                    **effects(p,k)) for arm,layers in actions.items() for i,(p,k) in enumerate(zip(points,layers,strict=True))]
    tables = dict(summary=summary,choices=choices)
    if fit is not None:
        tables.update(fit=fit,calibration=curve)
    return prior.publish(root,tables,stage="stage160")


def load(root):
    d = trained.check_results(root)
    plan = c.c66.sealed(root/"manifest.json",trained.SCHEMA)
    model = c.c66.sealed(root/"stage159_models.json",trained.SCHEMA)
    if d["fingerprint"] != SOURCE159 or model["plan_fingerprint"] != plan["fingerprint"]:
        raise ValueError("Need the archived Stage-159 caches")
    points = collect_points(prior.prior.read_table(root/"stage159_windows.csv"),prior.prior.read_table(root/"stage159_choices.csv"))
    return plan,model["models"],points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stage159-root",type=Path)
    mode.add_argument("--cached-bound",type=Path,metavar="STAGE159_BUNDLE")
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--progress-interval",type=float,default=15.)
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    source = args.stage159_root.expanduser().resolve() if args.stage159_root else None
    if source is not None and (source == root or source in root.parents or root in source.parents):
        parser.error("Use a separate Stage-160 result directory")
    status = "COMPLETE_STAGE160_MARGIN_AUDIT" if source is not None else "COMPLETE_STAGE160_CACHED_BOUND"
    with prior.prior.writer_lock(root):
        decision = root/"stage160_decision.json"
        if decision.exists():
            d = c.c66.sealed(decision,SCHEMA)
            if d["source_stage159"] != SOURCE159 or d["status"] != status:
                raise ValueError("Use separate result directories for the cached bound and the margin audit")
            io.verify({str(root/k):h for k,h in d["table_sha256"].items()})
            OverallProgress("Stage-160",d["work"],d["work"],args.progress_interval,"work").report("Complete")
            return
        if source is None:
            _, tables = prior.archives.bundle(args.cached_bound,159,SOURCE159)
            points = collect_points(tables["stage159_windows.csv"],tables["stage159_choices.csv"])
            progress = OverallProgress("Stage-160",1,interval=args.progress_interval,unit="work")
            progress.report("Cached bound")
            hashes = reports(root,points,{})
        else:
            plan,models,points = load(source)
            total = len(points)+sum(len(f["fit"]) for f in plan["folds"])+1
            progress = OverallProgress("Stage-160",total,interval=args.progress_interval,unit="work")
            progress.report("Checking cached features")
            features = []
            for p in points:
                cid = p["case_id"]
                data = c.read_cache(source/"branches"/f"{cid}.npz",dict(plan_fingerprint=plan["fingerprint"],case_id=cid))
                if data is None or int(data["anchor_layer"]) != p["native_layer"] or data["features"].shape != (24,):
                    raise ValueError(f"Missing or invalid Stage-159 features for {cid}")
                features.append(data["features"])
                progress.advance()
            progress.report("Calibrating and evaluating")
            choices,margins,fit,curve = calibrated(points,np.stack(features),plan["folds"],models,progress)
            hashes = reports(root,points,choices,margins,fit,curve)
        c.publish_identical(decision,io.seal(dict(schema=SCHEMA,status=status,source_stage159=SOURCE159,
            cases=len(points),work=progress.total,table_sha256=hashes,gpu_searches=0,new_full_trajectories=0)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
