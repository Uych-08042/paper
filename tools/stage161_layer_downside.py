"""Compare gain and downside-aware vetoes at the Stage-160 switch budget."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
import stage160_layer_veto as veto


prior, c, io = veto.prior, veto.c, veto.io
LAYERS = veto.LAYERS
SOURCE160 = "0e9eaccc4fcf3ee5ff392d1c3cda6d6f9694b00f250a3776968e1bb964eca299"
SCHEMA = "stage161_layer_downside_v1"
PENALTY = 1.


def attach_choices(points, rows):
    direct = {r["case_id"]:r for r in rows if r["arm"] == "future_supervised"}
    control = {r["case_id"]:r for r in rows if r["arm"] == "calibrated_margin"}
    if set(direct) != set(control) or set(direct) != {p["case_id"] for p in points}:
        raise ValueError("Inconsistent Stage-159/160 case sets")
    for p in points:
        cid, native = p["case_id"], p["native_layer"]
        a, b = direct[cid],control[cid]
        layer, margin = int(b["layer"]),float(a["predicted_margin"])
        if (int(a["layer"]) != p["future"] or layer not in (native,p["future"])
                or float(b["predicted_margin"]) != margin or not np.isfinite(margin)
                or any(r["sequence"] != p["sequence"] or int(r["frame"]) != p["frame"]
                       or int(r["native_layer"]) != native for r in (a,b))):
            raise ValueError(f"Stage-160 policy differs from the archived candidate for {cid}")
        p.update(stage160=layer,margin=margin)


def gain(indexed, points):
    return np.array([points[j]["margin"] for j in indexed])


def risk_prediction(points, features, fit, held):
    groups = [veto.source_group(p["sequence"]) for p in points]
    counts = Counter(g for g,use in zip(groups,fit,strict=True) if use)
    weights = np.array([1./counts[g] for g,use in zip(groups,fit,strict=True) if use])
    targets = np.stack([np.maximum(0.,p["tail32"][LAYERS.index(p["native_layer"])]-p["tail32"]) for p in points])
    mu,scale,bias,coef = veto.trained.ridge(features[fit],targets[fit],weights)
    score = np.maximum(0.,((features[held]-mu)/scale)@coef+bias)
    if not np.isfinite(score).all():
        raise ValueError("Invalid downside predictions")
    return [float(row[LAYERS.index(points[j]["future"])]) for row,j in zip(score,np.flatnonzero(held),strict=True)], targets[fit]


def top(points, indices, scores, limit):
    if limit > len(indices):
        raise ValueError("Switch budget exceeds available nonnative proposals")
    ranked = sorted(zip(indices,scores,strict=True),key=lambda pair:(-pair[1],-points[pair[0]]["margin"],points[pair[0]]["case_id"]))
    return {i for i,_ in ranked[:limit]}


def compare(points, folds, progress, features=None):
    groups = np.array([veto.source_group(p["sequence"]) for p in points])
    native = [p["native_layer"] for p in points]
    actions = dict(native=native,future_supervised=[p["future"] for p in points],
                   calibrated_margin=[p["stage160"] for p in points],matched_hindsight=native.copy())
    downside, risk_score = [None]*len(points), [None]*len(points)
    if features is not None:
        actions["gain_and_downside_matched"] = native.copy()
    scope = []
    coverage = np.zeros(len(points),int)
    for fold in folds:
        i = fold["fold"]
        fit, held = np.isin(groups,fold["fit"]), np.isin(groups,fold["held"])
        if not fit.any() or not held.any() or not np.array_equal(fit,~held):
            raise ValueError("Invalid Stage-159 outer folds")
        coverage += held
        available = [j for j in np.flatnonzero(held) if points[j]["future"] != native[j]]
        accepted = {j for j in available if points[j]["stage160"] != native[j]}
        budget = len(accepted)
        if top(points,available,gain(available,points),budget) != accepted:
            raise ValueError("Matched gain ranking must replay the Stage-160 gate")
        n = Counter(groups[held])
        eligible, actual = [], []
        for j in available:
            outcome = veto.effects(points[j],points[j]["future"])
            if outcome["delta32"] > veto.EPSILON and not (outcome["lost_good"] or outcome["new_low"]):
                eligible.append(j)
                actual.append(outcome["delta32"]/n[groups[j]])
        oracle = top(points,eligible,actual,min(budget,len(eligible)))
        for j in oracle:
            actions["matched_hindsight"][j] = points[j]["future"]
        row = dict(fold=i,fit_cases=int(fit.sum()),held_cases=int(held.sum()),available=len(available),
                   switch_budget=budget,hindsight_switches=len(oracle))
        if features is not None:
            predicted, target = risk_prediction(points,features,fit,held)
            for j, risk in zip(np.flatnonzero(held),predicted,strict=True):
                downside[j] = risk
                risk_score[j] = points[j]["margin"]-PENALTY*risk
            reranked = top(points,available,[risk_score[j] for j in available],budget)
            for j in reranked:
                actions["gain_and_downside_matched"][j] = points[j]["future"]
            row.update(risk_fit_nonzero_candidates=int((target>veto.EPSILON).sum()),
                       risk_fit_mean=float(target.mean()),risk_switches=len(reranked),
                       replaced_switches=len(reranked-accepted))
        scope.append(row)
        progress.advance()
    np.testing.assert_array_equal(coverage,np.ones(len(points),int))
    return actions,scope,downside,risk_score


def paired(points, actions):
    rows = []
    baseline = actions["calibrated_margin"]
    for arm in ("matched_hindsight","gain_and_downside_matched"):
        if arm not in actions:
            continue
        for group in ("all","uniform","historical_low","native_stable"):
            records = [(p,veto.effects(p,a),veto.effects(p,b)) for p,a,b in zip(points,actions[arm],baseline,strict=True)
                       if group in ("all",p["kind"]) or group == "native_stable" and p["native_stable"]]
            if not records:
                continue
            delta = prior.grouped_mean([p for p,_,_ in records],[a["tail32"]-b["tail32"] for _,a,b in records])
            rows.append(dict(arm=arm,baseline="calibrated_margin",group=group,cases=len(records),videos=len(delta),
                             mean_tail32_delta=float(delta.mean()),wins=int((delta>veto.EPSILON).sum()),
                             ties=int((abs(delta)<=veto.EPSILON).sum()),losses=int((delta< -veto.EPSILON).sum()),
                             worst_leave_one_out=float((delta.sum()-delta.max())/(len(delta)-1)) if len(delta)>1 else None,
                             **{f"delta_{k}":sum(int(a[k])-int(b[k]) for _,a,b in records)
                                for k in ("switched","new_good","lost_good","new_low")}))
    return rows


def report(root, points, actions, scope, downside, risk_score):
    rows = [dict(case_id=p["case_id"],sequence=p["sequence"],frame=p["frame"],kind=p["kind"],
                 native_stable=p["native_stable"],arm=arm,layer=k,native_layer=p["native_layer"],
                 proposed_layer=p["future"],predicted_gain=p["margin"],predicted_downside=downside[i],risk_score=risk_score[i],
                 **veto.effects(p,k)) for arm,layers in actions.items() for i,(p,k) in enumerate(zip(points,layers,strict=True))]
    return prior.publish(root,dict(summary=veto.summarize(points,actions),paired=paired(points,actions),
                                  folds=scope,choices=rows),stage="stage161")


def read_server(stage159, stage160):
    plan,_,points = veto.load(stage159)
    decision = c.c66.sealed(stage160/"stage160_decision.json",veto.SCHEMA)
    if decision["fingerprint"] != SOURCE160 or decision["source_stage159"] != veto.SOURCE159:
        raise ValueError("Need the exact Stage-160 held-out results")
    table = "stage160_choices.csv"
    io.verify({str(stage160/table):decision["table_sha256"][table]})
    attach_choices(points,prior.prior.read_table(stage160/table))
    return plan,points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage159-root",type=Path)
    parser.add_argument("--stage160-root",type=Path)
    parser.add_argument("--cached-bound",nargs=2,type=Path,metavar=("STAGE159_BUNDLE","STAGE160_BUNDLE"))
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--progress-interval",type=float,default=15.)
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    sources = [p.expanduser().resolve() for p in (args.stage159_root,args.stage160_root) if p is not None]
    if (args.cached_bound and sources) or (not args.cached_bound and len(sources)!=2) or args.progress_interval <= 0:
        parser.error("Provide both server roots or both result bundles, and a positive progress interval")
    if any(root == p or root in p.parents or p in root.parents for p in sources):
        parser.error("Use a separate Stage-161 directory")
    status = "COMPLETE_STAGE161_CACHED_BOUND" if args.cached_bound else "COMPLETE_STAGE161_MATCHED_DOWNSIDE_AUDIT"
    with prior.prior.writer_lock(root):
        decision = root/"stage161_decision.json"
        if decision.exists():
            d = c.c66.sealed(decision,SCHEMA)
            if d["status"] != status or d["source_stage160"] != SOURCE160 or d["source_stage159"] != veto.SOURCE159:
                raise ValueError("Use separate result directories for cached bound and matched audit")
            io.verify({str(root/k):h for k,h in d["table_sha256"].items()})
            OverallProgress("Stage-161",d["work"],d["work"],args.progress_interval,"work").report("Complete")
            return
        features = None
        if args.cached_bound:
            _, t159 = prior.archives.bundle(args.cached_bound[0],159,veto.SOURCE159)
            _, t160 = prior.archives.bundle(args.cached_bound[1],160,SOURCE160)
            points = veto.collect_points(t159["stage159_windows.csv"],t159["stage159_choices.csv"])
            attach_choices(points,t160["stage160_choices.csv"])
            folds = veto.trained.source.split_sources(sorted({p["sequence"] for p in points}))
            progress = OverallProgress("Stage-161",len(folds)+1,interval=args.progress_interval,unit="work")
            progress.report("Cached bound")
        else:
            plan,points = read_server(*sources)
            folds = plan["folds"]
            progress = OverallProgress("Stage-161",len(points)+len(folds)+1,interval=args.progress_interval,unit="work")
            progress.report("Checking cached features")
            vectors = []
            for p in points:
                cid = p["case_id"]
                data = c.read_cache(sources[0]/"branches"/f"{cid}.npz",dict(plan_fingerprint=plan["fingerprint"],case_id=cid))
                if (data is None or int(data["anchor_layer"]) != p["native_layer"] or data["features"].shape != (24,)
                        or not np.isfinite(data["features"]).all()):
                    raise ValueError(f"Missing or invalid Stage-159 features for {cid}")
                vectors.append(data["features"])
                progress.advance()
            features = np.stack(vectors)
            progress.report("Fitting and evaluating")
        actions,scope,downside,risk_score = compare(points,folds,progress,features)
        hashes = report(root,points,actions,scope,downside,risk_score)
        c.publish_identical(decision,io.seal(dict(schema=SCHEMA,status=status,source_stage159=veto.SOURCE159,
            source_stage160=SOURCE160,cases=len(points),work=progress.total,table_sha256=hashes,
            switch_budget=sum(r["switch_budget"] for r in scope),downside_penalty=PENALTY,
            gpu_searches=0,new_full_trajectories=0)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
