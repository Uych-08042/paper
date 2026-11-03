"""Collect fit-only matched pairs, fit two ridge gates, and run seven full policies."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import stage68_advantage_common as c
from stage27_uav_common import checked_image, crop_geometry, nonempty_path, tracking_iou
from stage27_uav_collect import writer_lock

e67=c.e67; e63=e67.e63; c66=c.prior.prior; e66=c.prior.e66
FIELDS=e66.FIELDS+("immediate_gain","future_gain")
OBSERVATIONS=("frame_id","history_latest","candidate_parents","candidate_boxes","features","choice_features",
              "crop","crop_sha256","image_hw","layer","block_calls","head_calls","selector_calls")


def prepare(source,root,hardware):
    source,root=Path(source).resolve(),Path(root).resolve()
    if source==root or source in root.parents or root in source.parents:raise ValueError("Use separate Stage-68 directory")
    p,_=c.make_plan(source,hardware)
    with writer_lock(root):
        if not (root/"manifest.json").exists() and any(x.name!="collect.lock" for x in root.iterdir()):raise ValueError("Use empty Stage-68 directory")
        c.publish_identical(root/"manifest.json",p)
    print("Stage-68 prepared: {} fit pairs / {} pair searches; {} full trajectories / {} evaluation searches".format(
        len(p["pairs"]),p["pair_search_calls"],len(p["units"]),p["formal_eval_calls"]),flush=True)
    for row in p["pair_support"]:print("[fit support] "+str(row),flush=True)


def context(plan,source,case,cache=None):
    split=next(f for f in plan["folds"] if f["fold"]==case["fold"])
    if case["sequence"] not in split["fit"] or case["sequence"] in split["held_out"]:raise ValueError("Fit pair crosses held-out group")
    if cache is None or case["sequence"] not in cache:
        observed=c.prior.read_unit(source["s66"],case["sequence"],"hann")
        if cache is not None:cache[case["sequence"]]=observed
    else:observed=cache[case["sequence"]]
    trace,entry,_=observed
    pack=c.fit_pack(source,case["fold"])
    if pack["fit"]!=split["fit"] or pack["held_out"]!=split["held_out"]:raise ValueError("Pair continuation Q is not fit-local")
    return trace,entry,pack


def audit_pair(shared,tails,trace,entry,case,plan,pack):
    history=e63.history_at(trace,entry["initial_box"],case["frame"],plan)
    np.testing.assert_array_equal(shared["initial_history"],history)
    e67.audit_segment(shared,history,case["frame"],1,"quality",plan,pack)
    for k in OBSERVATIONS:
        np.testing.assert_array_equal(shared[k],trace[k][case["frame"]-1:case["frame"]],err_msg="Native Hann observation differs: "+k)
    if int(shared["quality_choice"][0])!=case["allow_choice"] or case["allow_choice"]==0:raise ValueError("Frozen Q proposal changed")
    if np.array_equal(shared["candidate_boxes"][0,0],shared["candidate_boxes"][0,case["allow_choice"]]):raise ValueError("No actual pair intervention")
    for action,data in tails.items():
        e67.audit_segment(data,e67.successor_history(shared,case,action,plan),case["frame"]+1,
                          plan["design"]["horizon"]-1,"quality",plan,pack)


def fit_run(plan):return c.seal(dict(schema=c.SCHEMA,plan_fingerprint=plan["fingerprint"],runtime=plan["runtime_identity"],continuation="fit_local_frozen_quality"))
def eval_run(plan,models):return c.seal(dict(schema=c.SCHEMA,plan_fingerprint=plan["fingerprint"],runtime=plan["runtime_identity"],training_fingerprint=models["fingerprint"]))


def pair_gate(plan,root):
    r=c.sealed(Path(root)/"smoke/stage68_pair_smoke.json")
    if r.get("status")!="PASS_FIT_PAIR_SMOKE" or r.get("run_fingerprint")!=fit_run(plan)["fingerprint"] or r.get("search_calls")!=plan["pair_smoke_calls_reused"]:raise ValueError("Need complete fit-pair smoke")
    c.VERIFY(r["cache_sha256"]);return r


def collect(root,plan,source,observer,identity,smoke=False,image_reader=checked_image):
    if identity!=plan["runtime_identity"]:raise ValueError("Need bound L40 runtime")
    if not smoke:pair_gate(plan,root)
    c.publish_identical(Path(root)/"fit_runtime.json",fit_run(plan));hashes={};heads={};calls=0;contexts={}
    for case in plan["pairs"][:1] if smoke else plan["pairs"]:
        trace,entry,pack=context(plan,source,case,contexts);shared_path=c.pair_path(root,case);meta=c.pair_meta(plan,case)
        with e63.source_experiment.installed(observer,[],case["unit"]) as head:
            if head!=case["head_sha256"]:raise ValueError("Original native head differs")
            heads[case["case_id"]]=head;shared=c.read_cache(shared_path,meta)
            if shared is None:
                observer.initialize(image_reader(plan["root"],entry["frames"][0]),entry["initial_box"])
                history=e63.history_at(trace,entry["initial_box"],case["frame"],plan)
                row=e67.observe(observer,image_reader(plan["root"],entry["frames"][case["frame"]]),history,case["frame"],"quality",plan,pack)
                shared=dict(e67.stack([row]),initial_history=history.copy())
                audit_pair(shared,{},trace,entry,case,plan,pack);c.save(shared_path,shared,meta)
            tails={}
            for action in ("veto","allow"):
                path=c.pair_path(root,case,action);meta=c.pair_meta(plan,case,action);data=c.read_cache(path,meta)
                if data is None:
                    data=e67.collect_tail(observer,entry,shared,case,"quality",action,plan,pack,image_reader)
                    audit_pair(shared,{action:data},trace,entry,case,plan,pack);c.save(path,data,meta)
                tails[action]=data;e63.source_experiment.bind(hashes,path);calls+=int(data["head_calls"].sum())
            audit_pair(shared,tails,trace,entry,case,plan,pack)
        e63.source_experiment.bind(hashes,shared_path);calls+=int(shared["head_calls"].sum())
        print("[fit pair audited] "+case["case_id"],flush=True)
    observer.assert_frozen();c.VERIFY(plan["source_file_sha256"]);c.VERIFY(plan["code_sha256"])
    if calls!=(plan["pair_smoke_calls_reused"] if smoke else plan["pair_search_calls"]):raise ValueError("Fit pair call budget differs")
    r=c.seal(dict(schema=c.SCHEMA,status="PASS_FIT_PAIR_SMOKE" if smoke else "COMPLETE_FIT_PAIRED_BRANCHES",
        plan_fingerprint=plan["fingerprint"],run_fingerprint=fit_run(plan)["fingerprint"],search_calls=calls,block_calls=7*calls,
        pairs=1 if smoke else len(plan["pairs"]),observed_heads=heads,cache_sha256=hashes,training_performed=False))
    c.publish_identical(Path(root)/("smoke/stage68_pair_smoke.json" if smoke else "collection/run.json"),r);return r


def pair_completed(plan,root):
    pair_gate(plan,root)
    if c.sealed(Path(root)/"fit_runtime.json")!=fit_run(plan):raise ValueError("Fit runtime record changed")
    r=c.sealed(Path(root)/"collection/run.json")
    expected={str(c.pair_path(root,x,a).with_suffix(s).resolve()) for x in plan["pairs"] for a in (None,"allow","veto") for s in (".npz",".receipt.json")}
    if (r.get("status")!="COMPLETE_FIT_PAIRED_BRANCHES" or r.get("run_fingerprint")!=fit_run(plan)["fingerprint"] or
        r.get("plan_fingerprint")!=plan["fingerprint"] or r.get("search_calls")!=plan["pair_search_calls"] or
        r.get("block_calls")!=7*plan["pair_search_calls"] or r.get("pairs")!=len(plan["pairs"]) or
        r.get("observed_heads")!={x["case_id"]:x["head_sha256"] for x in plan["pairs"]} or set(r.get("cache_sha256",{}))!=expected):raise ValueError("Need all unmixed fit pairs")
    c.VERIFY(r["cache_sha256"]);return r


def label_pair(shared,tails,entry,case,plan):
    """Offline labels only, after actual own-history continuations have finished."""
    frames=np.arange(case["frame"],case["frame"]+plan["design"]["horizon"])
    gt=np.asarray(entry["ground_truth"],float)[frames];values={}
    for action in ("allow","veto"):
        q=e67.forced_choice(case,action)
        boxes=np.concatenate([shared["candidate_boxes"][:,q],tails[action]["boxes"]])
        values[action]=tracking_iou(boxes,gt)
    delta=values["allow"]-values["veto"]
    return dict(immediate=float(delta[0]),future=float(delta.mean()),
        allow_mean_iou=float(values["allow"].mean()),veto_mean_iou=float(values["veto"].mean()),
        low_frame_delta=int((values["allow"]<.2).sum()-(values["veto"]<.2).sum()))


def examples(plan,root,source,fold):
    split=next(f for f in plan["folds"] if f["fold"]==fold)
    cases=[x for x in plan["pairs"] if x["fold"]==fold];counts={seq:sum(x["sequence"]==seq for x in cases) for seq in split["fit"]}
    supported=sum(v>0 for v in counts.values());features=[];labels=[];weights=[];rows=[];contexts={}
    for case in cases:
        trace,entry,pack=context(plan,source,case,contexts)
        shared=c.read_cache(c.pair_path(root,case),c.pair_meta(plan,case))
        tails={a:c.read_cache(c.pair_path(root,case,a),c.pair_meta(plan,case,a)) for a in ("allow","veto")}
        audit_pair(shared,tails,trace,entry,case,plan,pack)
        label=label_pair(shared,tails,entry,case,plan);weight=1/(supported*counts[case["sequence"]])
        features.append(c.pair_feature(shared["choice_features"][0],case["allow_choice"]));labels.append([label[t] for t in c.TARGETS]);weights.append(weight)
        rows.append(dict(case_id=case["case_id"],fold=fold,sequence=case["sequence"],frame=case["frame"],
            allow_choice=case["allow_choice"],quality_gain=float(shared["quality_gain"][0]),weight=weight,**label))
    return np.asarray(features),np.asarray(labels),np.asarray(weights),rows


def build_fold(plan,root,source,fold):
    x,y,w,rows=examples(plan,root,source,fold);split=next(f for f in plan["folds"] if f["fold"]==fold)
    predictors={t:c.fit_regressor(x,y[:,j],w,plan["design"]["ridge"],t) for j,t in enumerate(c.TARGETS)}
    # Calibrate on ALL cached fit-Hann states, with equal video mass. This
    # requires no extra labels or network calls and is not a held-out rate fit.
    gains=[];eligible=[];accepted=[];calibration_weights=[]
    for seq in split["fit"]:
        trace,_,_=c.prior.read_unit(source["s66"],seq,"hann")
        old=c.fit_pack(source,fold)
        for base,f in zip(trace["features"],trace["choice_features"]):
            decision=c66.decide("quality",base,f,old,plan["controller_design"]);q=decision["selected"]
            gains.append(decision["quality_gain"]);eligible.append(q!=0)
            accepted.append(bool(q and c.predict(c.pair_feature(f,q),predictors["future"])>plan["design"]["tie_epsilon"]))
            calibration_weights.append(1/(len(split["fit"])*len(trace["frame_id"])))
    margin=c66.match_margin(gains,eligible,accepted,calibration_weights)
    margin["future_fit_acceptance"]=margin.pop("protected_fit_acceptance")
    margin["calibration_states"]=len(gains);margin["calibration_origin"]="all_fit_native_hann_frames"
    pack=dict(fit=split["fit"],held_out=split["held_out"],source=source["s66"]["models"]["models"][str(fold)],rate_margin=margin,**predictors)
    return pack,rows


def build_models(plan,root,source):
    run=pair_completed(plan,root);packs={};support=[]
    for f in plan["folds"]:
        pack,rows=build_fold(plan,root,source,f["fold"]);packs[str(f["fold"])]=pack;support.extend(rows)
    return c.seal(dict(schema=c.SCHEMA,status="COMPLETE_FIT_PAIRED_ADVANTAGE_MODELS",plan_fingerprint=plan["fingerprint"],
        pair_run_fingerprint=run["fingerprint"],source_training_fingerprint=source["s66"]["models"]["fingerprint"],
        models=packs,support=support,new_predictors=2*len(packs),sgla_training_performed=False))


def fit(plan,root,source):
    models=build_models(plan,root,source);c.VERIFY(plan["source_file_sha256"]);c.VERIFY(plan["code_sha256"])
    with writer_lock(Path(root)/"models"):c.publish_identical(Path(root)/"models/training.json",models)
    print("Stage-68 CPU fit: {} matched immediate/future predictors".format(models["new_predictors"]));return models


def load_models(plan,root,source):
    m=c.sealed(Path(root)/"models/training.json");pair_run=pair_completed(plan,root)
    if (m.get("status")!="COMPLETE_FIT_PAIRED_ADVANTAGE_MODELS" or m.get("plan_fingerprint")!=plan["fingerprint"] or
        m.get("pair_run_fingerprint")!=pair_run["fingerprint"] or m.get("source_training_fingerprint")!=source["s66"]["models"]["fingerprint"] or
        set(m.get("models",{}))!={str(f["fold"]) for f in plan["folds"]}):raise ValueError("Need bound paired advantage models")
    for f in plan["folds"]:
        p=m["models"][str(f["fold"])]
        if p["fit"]!=f["fit"] or p["held_out"]!=f["held_out"] or p["source"]!=source["s66"]["models"]["models"][str(f["fold"])]:raise ValueError("Fit isolation or frozen Q changed")
        for t in c.TARGETS:
            model=p[t]
            if model["target"]!=t or any(len(model[k])!=128 for k in ("center","scale","weights")) or not all(np.isfinite(model[k]).all() for k in ("center","scale","weights","bias")) or np.any(np.asarray(model["scale"])<=0):raise ValueError("Invalid gain predictor")
        if not np.isfinite(p["rate_margin"]["threshold"]) or p["rate_margin"]["threshold"]<0:raise ValueError("Invalid fit-rate control")
    return m


def audit_eval(data,entry,unit,plan,models):
    n=len(data["frame_id"]);e63.source_common.prior.audit_trace(data,n,4)
    if not 0<n<=len(entry["frames"])-1 or np.max(data["decode_error"])>plan["actor_design"]["box_atol"]:raise ValueError("Invalid evaluation segment")
    history=[np.asarray(entry["initial_box"],float)];pack=models["models"][str(unit["fold"])]
    for i in range(n):
        parents=e63.source_common.prior.history_parents(history,plan["actor_design"])
        np.testing.assert_array_equal(data["candidate_parents"][i],parents)
        f=c66.c64.choice_features(data["features"][i],data["candidate_boxes"][i],history)
        np.testing.assert_array_equal(data["choice_features"][i],f)
        for k,v in c.decide(unit["policy"],data["features"][i],f,pack,plan["design"],plan["controller_design"]).items():np.testing.assert_array_equal(data[k][i],v)
        np.testing.assert_array_equal(data["history_latest"][i],history[-1])
        for q,p in enumerate(parents):np.testing.assert_array_equal(data["crop"][i,q],crop_geometry(p,data["image_hw"][i],plan["search_factor"],plan["search_size"])[0])
        history.append(data["boxes"][i]);history=history[-plan["actor_design"]["history"]:]


def run_actor(observer,entry,unit,plan,pack,limit,prefix=None,image_reader=checked_image):
    observer.initialize(image_reader(plan["root"],entry["frames"][0]),entry["initial_box"])
    history=[np.asarray(entry["initial_box"],float)];rows=[];start=1
    if prefix is not None:
        start=len(prefix["frame_id"])+1
        for b in prefix["boxes"]:history.append(b);history=history[-plan["actor_design"]["history"]:]
    for frame in range(start,limit+1):
        row=e63.probe(observer,image_reader(plan["root"],entry["frames"][frame]),history,frame,"history",e67.actor_plan(plan))
        f=c66.c64.choice_features(row["features"],row["candidate_boxes"],history)
        d=c.decide(unit["policy"],row["features"],f,pack,plan["design"],plan["controller_design"]);q=d["selected"]
        row.update(d,choice_features=f,boxes=row["candidate_boxes"][q].copy(),parents=row["candidate_parents"][q].copy())
        rows.append(row);history.append(row["boxes"].copy());history=history[-plan["actor_design"]["history"]:]
        observer.rt.tracker.state=row["boxes"].tolist()
    data={k:np.asarray([r[k] for r in rows]) for k in FIELDS} if rows else {k:prefix[k][:0] for k in FIELDS}
    return data if prefix is None else {k:np.concatenate([prefix[k],data[k]]) for k in FIELDS}


def check_control(data,source,unit):
    if unit["policy"] not in ("hann","q0","quality","protect"):return
    trace,_,_=c.prior.read_unit(source["s66"],unit["sequence"],unit["policy"])
    e67.exact_source(data,trace,1)


def eval_gate(plan,root,models):
    r=c.sealed(Path(root)/"smoke/stage68_eval_smoke.json")
    if r.get("status")!="PASS_ADVANTAGE_EVAL_SMOKE" or r.get("run_fingerprint")!=eval_run(plan,models)["fingerprint"] or r.get("search_calls")!=plan["eval_smoke_calls_reused"]:raise ValueError("Need passing seven-policy smoke")
    c.VERIFY(r["cache_sha256"]);return r


def evaluate(root,plan,source,observer,identity,smoke=False,image_reader=checked_image):
    if identity!=plan["runtime_identity"]:raise ValueError("Need bound L40 runtime")
    models=load_models(plan,root,source)
    if not smoke:eval_gate(plan,root,models)
    c.publish_identical(Path(root)/"run.json",eval_run(plan,models));hashes={};heads={};calls=0
    units=[u for u in plan["units"] if not smoke or u["sequence"]==plan["entries"][0]["sequence"]]
    for u in units:
        entry=next(e for e in plan["entries"] if e["sequence"]==u["sequence"]);limit=len(entry["frames"])-1
        if smoke:limit=min(limit,plan["design"]["smoke_frames"])
        path=c.eval_path(root,u,smoke);meta=c.eval_meta(plan,u,models,smoke);data=c.read_cache(path,meta)
        with e63.source_experiment.installed(observer,[],u) as head:
            expected=source["s66"]["run"]["observed_heads"][c66.key(dict(u,policy="hann"))]
            if head!=expected:raise ValueError("Original native head differs")
            if data is None:
                prefix=None
                if not smoke and u["sequence"]==plan["entries"][0]["sequence"]:
                    prefix=c.read_cache(c.eval_path(root,u,True),c.eval_meta(plan,u,models,True))
                    if prefix is None:raise ValueError("Missing reusable evaluation smoke")
                data=run_actor(observer,entry,u,plan,models["models"][str(u["fold"])],limit,prefix,image_reader)
                audit_eval(data,entry,u,plan,models);check_control(data,source,u);c.save(path,data,meta)
            audit_eval(data,entry,u,plan,models);check_control(data,source,u)
        e63.source_experiment.bind(hashes,path);heads[c66.key(u)]=head;calls+=int(data["head_calls"].sum())
        print("[advantage evaluated] "+c66.key(u),flush=True)
    observer.assert_frozen();c.VERIFY(plan["source_file_sha256"]);c.VERIFY(plan["code_sha256"])
    if calls!=(plan["eval_smoke_calls_reused"] if smoke else plan["formal_eval_calls"]):raise ValueError("Evaluation call budget differs")
    r=c.seal(dict(schema=c.SCHEMA,status="PASS_ADVANTAGE_EVAL_SMOKE" if smoke else "COMPLETE_ADVANTAGE_OOF",
        plan_fingerprint=plan["fingerprint"],run_fingerprint=eval_run(plan,models)["fingerprint"],search_calls=calls,
        block_calls=7*calls,trajectories=len(units),cache_sha256=hashes,observed_heads=heads,maximum_source_box_error=0.))
    c.publish_identical(Path(root)/("smoke/stage68_eval_smoke.json" if smoke else "evaluation/run.json"),r);return r


def completed(plan,root,source,models):
    eval_gate(plan,root,models)
    if c.sealed(Path(root)/"run.json")!=eval_run(plan,models):raise ValueError("Evaluation runtime/model record changed")
    r=c.sealed(Path(root)/"evaluation/run.json")
    expected={str(c.eval_path(root,u).with_suffix(s).resolve()) for u in plan["units"] for s in (".npz",".receipt.json")}
    heads={c66.key(u):source["s66"]["run"]["observed_heads"][c66.key(dict(u,policy="hann"))] for u in plan["units"]}
    if (r.get("status")!="COMPLETE_ADVANTAGE_OOF" or r.get("run_fingerprint")!=eval_run(plan,models)["fingerprint"] or
        r.get("plan_fingerprint")!=plan["fingerprint"] or r.get("search_calls")!=plan["formal_eval_calls"] or
        r.get("block_calls")!=7*plan["formal_eval_calls"] or r.get("trajectories")!=len(plan["units"]) or
        set(r.get("cache_sha256",{}))!=expected or r.get("observed_heads")!=heads or r.get("maximum_source_box_error")!=0):raise ValueError("Need all unmixed complete policies")
    c.VERIFY(r["cache_sha256"]);return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--mode",required=True,choices=("prepare","pair_smoke","collect","fit","smoke","evaluate"))
    p.add_argument("--input",required=True,type=nonempty_path);p.add_argument("--output_dir",type=nonempty_path)
    p.add_argument("--checkpoint",default=os.environ.get("CHECKPOINT"));p.add_argument("--gpu",type=int,default=0);p.add_argument("--parameter",default="deit_distilled")
    args=p.parse_args()
    if args.mode=="prepare":
        if not args.output_dir:p.error("prepare requires --output_dir")
        import torch
        source=c.load_source(args.input)
        hardware=c66.c64.prior.prior.c.prior.device_binding(source["s66"]["plan"],args.gpu,torch)
        prepare(args.input,args.output_dir,hardware);return
    root=Path(args.input).resolve();plan,source=c.load_plan(root)
    if args.mode=="fit":fit(plan,root,source);return
    if not args.checkpoint:p.error("Need original SGLA V1 checkpoint")
    with writer_lock(root):
        native,identity=e63.startup.checked_runtime(args,plan,root);observer=e63.Capture(native)
        try:
            if args.mode in ("pair_smoke","collect"):collect(root,plan,source,observer,identity,args.mode=="pair_smoke")
            else:evaluate(root,plan,source,observer,identity,args.mode=="smoke")
        finally:observer.handle.remove()


if __name__=="__main__":main()
