"""Test longer causal histories at the frozen Stage-161 layer-switch budgets."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
import stage161_layer_downside as risk


veto, prior, c, io = risk.veto, risk.prior, risk.c, risk.io
trained = veto.trained
SOURCE161 = "f7ec6f0c5f66211e8503b564ffaaf01f2f9f20e653ed02f23d6917a711054faa"
SCHEMA = "stage162_history_ablation_v1"
BASELINE = "history4_base"
FEATURES = {BASELINE:(4,False),"history16_base":(16,False),"history4_trend":(4,True),"history16_trend":(16,True)}
FOCUS = {("uav0000176",163),("uav0000226",174),("uav0000348",540),("uav0000080",171)}


def features_from_history(boxes, peaks, frame, window, trend):
    end = frame-1
    begin = max(0,end-window)
    recent = np.asarray(boxes[begin:end],float)
    confidence = np.asarray(peaks[begin:end],float)
    if len(recent) < 4 or len(confidence) != len(recent) or (recent[:,2:] <= 0).any():
        raise ValueError("Invalid causal history")
    centers = recent[:,:2]+.5*recent[:,2:]
    shift = np.linalg.norm(np.diff(centers,axis=0),axis=1)/np.sqrt(recent[:-1,2:].prod(axis=1))
    scale = np.abs(np.diff(np.log(recent[:,2:]),axis=0)).mean(axis=1)
    base = [confidence[-1],confidence.mean(),confidence.std(),confidence[-1]-confidence[0],
            shift[-1],shift.mean(),scale[-1],scale.mean()]
    if trend:
        reverse = (np.diff(confidence)<0)[::-1]
        falling_run = len(reverse) if reverse.all() else int(np.flatnonzero(~reverse)[0])
        base.extend([falling_run/len(reverse),shift[-1]-np.median(shift[:-1]),scale.std()])
    result = np.asarray(base,float)
    if not np.isfinite(result).all():
        raise ValueError("Invalid history features")
    return result


def attach_stage161(points, rows):
    choices = [dict(r,predicted_margin=r["predicted_gain"]) for r in rows]
    risk.attach_choices(points,choices)
    by_id = {r["case_id"]:r for r in rows if r["arm"] == "gain_and_downside_matched"}
    if set(by_id) != {p["case_id"] for p in points}:
        raise ValueError("Incomplete Stage-161 downside choices")
    for p in points:
        r = by_id[p["case_id"]]
        layer, downside = int(r["layer"]),float(r["predicted_downside"])
        if (layer not in (p["native_layer"],p["future"]) or not np.isfinite(downside) or downside < 0
                or float(r["predicted_gain"]) != p["margin"] or int(r["proposed_layer"]) != p["future"]):
            raise ValueError("Stage-161 downside choices changed")
        p.update(previous_risk=downside,previous_layer=layer)


def native_jobs(root):
    origin = trained.source.source
    plan = c.c66.sealed(root/"manifest.json",origin.s121.SCHEMA)
    d = c.c66.sealed(root/"analysis/stage121_decision.json",origin.s121.SCHEMA)
    if d["fingerprint"] != origin.SOURCE121 or d["plan_fingerprint"] != plan["fingerprint"]:
        raise ValueError("Need the exact Stage-121 native trajectories used in Stage-159")
    return plan,d,{j["spec"]["unit"]["sequence"]:j for j in plan["jobs"] if j["arm"] == "native"}


def prepare(points, plan159, root159, root121, progress):
    from lib.test.utils.hann import hann2d
    import torch

    plan,d,jobs = native_jobs(root121)
    side = int(np.sqrt(plan["runtime"]["tracker"]["lens_x"]))
    window = hann2d(torch.tensor([side,side],dtype=torch.long),centered=True).numpy().reshape(side,side)
    grouped = defaultdict(list)
    for i,p in enumerate(points):
        grouped[p["sequence"]].append((i,p))
    vectors = {name:[None]*len(points) for name in FEATURES}
    support = []
    for seq, records in grouped.items():
        job = jobs[seq]
        key = f"evaluation/{trained.base.job_key(job)}"
        io.verify({str(root121/k):d["artifact_sha256"][k] for k in (key+".npz",key+".receipt.json")})
        trace = trained.source.source.prior.prior.load_trace(root121,plan,job)
        boxes = trace["boxes"]
        raw = trace["raw"][:,0]
        if raw.shape != (len(boxes),side,side):
            raise ValueError(f"Invalid native Hann maps for {seq}")
        np.testing.assert_array_equal(trace["frame_id"],np.arange(1,len(boxes)+1))
        peaks = np.max(raw*window,axis=(1,2))
        for i,p in records:
            cid, frame = p["case_id"],p["frame"]
            cache = c.read_cache(root159/"branches"/f"{cid}.npz",dict(plan_fingerprint=plan159["fingerprint"],case_id=cid))
            if cache is None or cache["features"].shape != (24,) or not np.isfinite(cache["features"]).all():
                raise ValueError(f"Missing or invalid Stage-159 features for {cid}")
            if frame > len(boxes) or int(cache["frame_id"][0]) != frame or int(cache["anchor_layer"]) != p["native_layer"]:
                raise ValueError(f"Inconsistent Stage-159 anchor for {cid}")
            np.testing.assert_array_equal(cache["anchor_parent"],trace["candidate_parents"][frame-1,0])
            rebuilt = features_from_history(boxes,peaks,frame,4,False)
            np.testing.assert_allclose(rebuilt,cache["features"][16:],atol=2e-7,rtol=0,
                                       err_msg=f"Archived 4-frame history does not replay for {cid}")
            for name,(length,trend) in FEATURES.items():
                history = features_from_history(boxes,peaks,frame,length,trend)
                # Reuse the exact archived four-frame summary for both short-history models.
                if length == 4:
                    history[:8] = cache["features"][16:]
                vectors[name][i] = np.r_[cache["features"][:16],history]
            support.append(dict(case_id=cid,sequence=seq,frame=frame,history16_available=min(16,frame-1)))
            progress.advance()
    return {name:np.stack(rows) for name,rows in vectors.items()},support


def fit(points, folds, features, progress):
    actions = {"native":[p["native_layer"] for p in points],"stage160":[p["stage160"] for p in points]}
    scores, scopes = {}, []
    for name in FEATURES:
        choices, scope, downside, score = risk.compare(points,folds,progress,features[name])
        layer = choices["gain_and_downside_matched"]
        if name == BASELINE:
            np.testing.assert_array_equal(layer,[p["previous_layer"] for p in points])
            np.testing.assert_allclose(downside,[p["previous_risk"] for p in points],atol=1e-12,rtol=0)
        actions[name] = layer
        scores[name] = (downside,score)
        scopes.extend(dict(arm=name,history=FEATURES[name][0],trends=FEATURES[name][1],**r) for r in scope)
    return actions,scores,scopes


def paired(points, actions, baseline):
    rows = []
    for name,layers in actions.items():
        if name == baseline:
            continue
        for group in ("all","uniform","historical_low","native_stable"):
            records = [(p,veto.effects(p,a),veto.effects(p,b))
                       for p,a,b in zip(points,layers,actions[baseline],strict=True)
                       if group in ("all",p["kind"]) or group == "native_stable" and p["native_stable"]]
            if not records:
                continue
            delta = prior.grouped_mean([p for p,_,_ in records],[a["tail32"]-b["tail32"] for _,a,b in records])
            rows.append(dict(arm=name,baseline=baseline,group=group,cases=len(records),videos=len(delta),
                             mean_tail32_delta=float(delta.mean()),worst_leave_one_out=float((delta.sum()-delta.max())/(len(delta)-1))
                             if len(delta)>1 else None,wins=int((delta>veto.EPSILON).sum()),
                             ties=int((abs(delta)<=veto.EPSILON).sum()),losses=int((delta< -veto.EPSILON).sum()),
                             **{f"delta_{k}":sum(int(a[k])-int(b[k]) for _,a,b in records)
                                for k in ("switched","new_good","lost_good","new_low")}))
    return rows


def cached_diagnostic(points, folds):
    """Retrospectively recover missed recoveries with the same per-fold count."""
    original = [p["previous_layer"] for p in points]
    chosen = original.copy()
    swaps = []
    for fold in folds:
        held = [i for i,p in enumerate(points) if veto.source_group(p["sequence"]) in fold["held"]]
        drops, recoveries = [], []
        for i in held:
            p = points[i]
            observed = veto.effects(p,original[i])
            offered = veto.effects(p,p["future"])
            if observed["switched"] and not observed["new_good"]:
                drops.append((observed["delta32"],i))
            if not observed["switched"] and offered["new_good"] and not (offered["new_low"] or offered["lost_good"]):
                recoveries.append((-offered["delta32"],i))
        for (_,restore),(_,drop) in zip(sorted(recoveries),sorted(drops)):
            chosen[restore] = points[restore]["future"]
            chosen[drop] = points[drop]["native_layer"]
            swaps.append(dict(fold=fold["fold"],recovery=points[restore]["case_id"],replaced=points[drop]["case_id"],
                              recovery_delta32=veto.effects(points[restore],chosen[restore])["delta32"],
                              replaced_delta32=veto.effects(points[drop],original[drop])["delta32"]))
    return {BASELINE:original,"hindsight_recovery_swaps":chosen},swaps


def publish(root, points, actions, scores, extra):
    choices = []
    for name,layers in actions.items():
        downside, ranking = scores.get(name,([None]*len(points),[None]*len(points)))
        choices.extend(dict(arm=name,case_id=p["case_id"],sequence=p["sequence"],frame=p["frame"],kind=p["kind"],
                            layer=k,native_layer=p["native_layer"],proposed_layer=p["future"],predicted_gain=p["margin"],
                            predicted_downside=risk_value,ranking_score=score,**veto.effects(p,k))
                       for p,k,risk_value,score in zip(points,layers,downside,ranking,strict=True))
    focus = [r for r in choices if (veto.source_group(r["sequence"]),r["frame"]) in FOCUS]
    tables = dict(summary=veto.summarize(points,actions),paired=paired(points,actions,BASELINE),
                  choices=choices,focus=focus,**extra)
    return prior.publish(root,{name:rows for name,rows in tables.items() if rows},stage="stage162")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage121","stage159","stage161")
    for name in names:
        parser.add_argument("--"+name+"-root",type=Path)
    parser.add_argument("--cached-diagnostic",nargs=2,type=Path,metavar=("STAGE159_BUNDLE","STAGE161_BUNDLE"))
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--progress-interval",type=float,default=15.)
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    sources = [p.expanduser().resolve() for n in names if (p:=getattr(args,n+"_root")) is not None]
    if (args.cached_diagnostic and sources) or (not args.cached_diagnostic and len(sources)!=3) or args.progress_interval <= 0:
        parser.error("Provide all server roots or both result bundles, and a positive interval")
    if any(root == p or root in p.parents or p in root.parents for p in sources):
        parser.error("Use a separate Stage-162 result directory")
    status = "COMPLETE_STAGE162_CACHED_DIAGNOSTIC" if args.cached_diagnostic else "COMPLETE_STAGE162_HISTORY_ABLATION"
    with prior.prior.writer_lock(root):
        destination = root/"stage162_decision.json"
        if destination.exists():
            saved = c.c66.sealed(destination,SCHEMA)
            if saved["status"] != status or saved["source_stage161"] != SOURCE161:
                raise ValueError("Use a separate directory for the cached diagnostic and history ablation")
            io.verify({str(root/k):h for k,h in saved["table_sha256"].items()})
            OverallProgress("Stage-162",saved["work"],saved["work"],args.progress_interval,"work").report("Complete")
            return
        if args.cached_diagnostic:
            _, a = prior.archives.bundle(args.cached_diagnostic[0],159,veto.SOURCE159)
            _, b = prior.archives.bundle(args.cached_diagnostic[1],161,SOURCE161)
            points = veto.collect_points(a["stage159_windows.csv"],a["stage159_choices.csv"])
            attach_stage161(points,b["stage161_choices.csv"])
            folds = trained.source.split_sources(sorted({p["sequence"] for p in points}))
            progress = OverallProgress("Stage-162",1,interval=args.progress_interval,unit="work")
            progress.report("Cached diagnostic")
            actions,swaps = cached_diagnostic(points,folds)
            hashes = publish(root,points,actions,{},dict(swaps=swaps))
        else:
            r121,r159,r161 = sources
            plan,_,points = veto.load(r159)
            d = c.c66.sealed(r161/"stage161_decision.json",risk.SCHEMA)
            if d["fingerprint"] != SOURCE161 or d["source_stage159"] != veto.SOURCE159:
                raise ValueError("Need the exact Stage-161 matched-risk run")
            table = "stage161_choices.csv"
            io.verify({str(r161/table):d["table_sha256"][table]})
            attach_stage161(points,prior.prior.read_table(r161/table))
            folds = plan["folds"]
            progress = OverallProgress("Stage-162",len(points)+len(folds)*len(FEATURES)+1,
                                       interval=args.progress_interval,unit="work")
            progress.report("Reading causal histories")
            features,support = prepare(points,plan,r159,r121,progress)
            progress.report("Fitting and evaluating")
            actions,scores,scope = fit(points,folds,features,progress)
            hashes = publish(root,points,actions,scores,dict(folds=scope,support=support))
        c.publish_identical(destination,io.seal(dict(schema=SCHEMA,status=status,source_stage159=veto.SOURCE159,
            source_stage161=SOURCE161,cases=len(points),work=progress.total,table_sha256=hashes,
            gpu_searches=0,new_full_trajectories=0)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
