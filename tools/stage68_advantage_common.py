"""Fit-only one-decision advantages under frozen Q, with matched current controls."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
import stage67_veto_common as prior
import stage67_veto_experiment as e67
import stage67_summary_repair as repair
from stage4_common import file_hash, fingerprint, write_cache
from stage27_uav_common import publish_identical

SCHEMA = "stage68_fit_paired_takeover_advantage_v1"
FILES = ("stage68_advantage_common.py", "stage68_advantage_experiment.py", "stage68_advantage_summary.py",
         "test_stage68_advantage.py", "run_stage68_advantage.sh")
DESIGN = dict(horizon=32, anchors_per_video=32, ridge=.01, tie_epsilon=1e-6, smoke_frames=32,
              bootstrap_seed=2168, bootstrap_samples=20000, practical_gain=.5, reliable_floor=-1.)
POLICIES = ("hann", "q0", "quality", "protect", "rate_margin", "immediate", "future")
TARGETS = ("immediate", "future")
seal, VERIFY = prior.seal, prior.VERIFY


def sealed(path): return prior.prior.sealed(path, SCHEMA)


def load_source(root):
    root = Path(root).expanduser().resolve(); plan, s66 = prior.load_plan(root)
    run = e67.completed(plan, root); report = prior.sealed(root/"analysis/stage67_diagnostic.json")
    fix = prior.prior.sealed(root/"analysis/stage67_summary_repair.json", repair.SCHEMA)
    tables = {"stage67_"+n+".csv" for n in ("anchors","case_summary","branch_summary","contrasts","scope",
                                               "cached_video","divergences","veto_frames","timelines")}
    if (report.get("status") != "COMPLETE_MATCHED_VETO_CONTINUATION_DIAGNOSTIC" or
        report.get("plan_fingerprint") != plan["fingerprint"] or report.get("run_fingerprint") != run["fingerprint"] or
        report.get("source_training_fingerprint") != s66["models"]["fingerprint"] or
        report.get("evidence_sha256") != run["cache_sha256"] or report.get("runtime") != plan["runtime_identity"] or
        report.get("formal_search_calls") != plan["formal_search_calls"] or report.get("training_performed") is not False or
        set(report.get("table_sha256",{})) != tables or fix.get("plan_fingerprint") != plan["fingerprint"] or
        fix.get("run_fingerprint") != run["fingerprint"] or fix.get("table_sha256") != report["table_sha256"] or
        fix.get("diagnostic_sha256") != file_hash(root/"analysis/stage67_diagnostic.json") or
        fix.get("original_code_sha256") != plan["code_sha256"] or fix.get("anchor_rows") != len(plan["requests"]) or
        any(fix.get(k) is not False for k in ("manifest_edited","source_hash_checks_relaxed","cache_metadata_edited","training_performed")) or
        fix.get("new_image_reads") != 0 or fix.get("new_search_calls") != 0):
        raise ValueError("Need complete Stage-67 diagnostic and its export-only repair")
    bindings = dict(plan["source_file_sha256"]); bindings.update(plan["code_sha256"]); bindings.update(run["cache_sha256"])
    bindings.update(fix["repair_code_sha256"])
    bindings.update({str(root/"analysis"/n):h for n,h in report["table_sha256"].items()})
    for name in ("manifest.json","run.json","collection/run.json","smoke/stage67_smoke.json",
                 "analysis/stage67_diagnostic.json","analysis/stage67_summary_repair.json"):
        bindings[str(root/name)] = file_hash(root/name)
    VERIFY(bindings)
    return dict(root=root,plan67=plan,report=report,s66=s66,bindings=bindings)


def fit_pack(source,fold):
    return source["s66"]["models"]["models"][str(fold)]


def resolve_pairs(source):
    """Only native Hann histories; the current fold's Q picks eligible times, without GT."""
    old = source["s66"]; plan = old["plan"]; cases, support = [], []
    for split in plan["folds"]:
        pack = fit_pack(source,split["fold"])
        if pack["fit"] != split["fit"] or pack["held_out"] != split["held_out"]: raise ValueError("Source fold differs")
        for seq in split["fit"]:
            data,entry,unit = prior.read_unit(old,seq,"hann")
            # This trajectory was chosen by fixed Hann, not by its stored OOF Q.
            # Only the state/candidate observation is used; recompute Q with THIS fold.
            choices = [prior.prior.decide("quality",b,f,pack,plan["design"])["selected"]
                       for b,f in zip(data["features"],data["choice_features"])]
            eligible = [i for i,q in enumerate(choices) if q!=0 and i+DESIGN["horizon"]<=len(data["frame_id"]) and
                        not np.array_equal(data["candidate_boxes"][i,q],data["candidate_boxes"][i,0])]
            selected = np.linspace(0,len(eligible)-1,min(DESIGN["anchors_per_video"],len(eligible)),dtype=int) if eligible else []
            support.append(dict(fold=split["fold"],sequence=seq,source_policy="hann",eligible_states=len(eligible),
                                selected_pairs=len(selected),status="supported" if len(selected) else "no_full_horizon_Q_proposal"))
            for j in selected:
                i = eligible[j]
                cases.append(dict(case_id="f{}__{}__t{:06d}".format(split["fold"],seq,i+1),sequence=seq,fold=split["fold"],
                    frame=i+1,allow_choice=choices[i],unit=dict(sequence=seq,origin="native",seed=0,fold=split["fold"],policy="quality"),
                    source_trace_sha256=file_hash(prior.e66.path_for(old["root"],unit)),
                    head_sha256=old["run"]["observed_heads"][prior.prior.key(unit)]))
        if not any(x["fold"]==split["fold"] for x in cases): raise ValueError("No eligible fit pair in fold; do not replace with held-out events")
    return cases,support


def make_plan(root, hardware):
    source = load_source(root); old = source["s66"]["plan"]
    if set(hardware)!={"device","gpu_uuid"} or hardware["device"]!="NVIDIA L40" or not hardware["gpu_uuid"]:
        raise ValueError("Need observed L40 binding")
    identity=copy.deepcopy(old["runtime_identity"]);identity["gpu_uuid"]=hardware["gpu_uuid"]
    changes=e67.e63.startup.differences(old["runtime_identity"],identity)
    if any(r["field"]!="runtime.gpu_uuid" for r in changes):raise ValueError("Only physical L40 UUID may change")
    if DESIGN["horizon"]<2 or DESIGN["anchors_per_video"]<1:raise ValueError("Invalid fixed pair budget")
    pairs,support=resolve_pairs(source)
    units=[dict(sequence=e["sequence"],origin="native",seed=0,
        fold=next(f["fold"] for f in old["folds"] if e["sequence"] in f["held_out"]),policy=p)
        for e in old["entries"] for p in POLICIES]
    eval_calls=sum(4*(len(e["frames"])-1)*len(POLICIES) for e in old["entries"])
    return seal(dict(schema=SCHEMA,role="consumed_development_fit_paired_takeover_advantage",source_input=str(source["root"]),
        source_report_fingerprint=source["report"]["fingerprint"],source_file_sha256=source["bindings"],design=dict(DESIGN),
        controller_design=copy.deepcopy(old["design"]),actor_design=copy.deepcopy(old["actor_design"]),
        hardware=hardware,runtime=identity["tracker"],runtime_identity=identity,declared_runtime_differences=changes,
        pairs=pairs,pair_support=support,units=units,policies=list(POLICIES),
        pair_search_calls=len(pairs)*(4+8*(DESIGN["horizon"]-1)),pair_smoke_calls_reused=4+8*(DESIGN["horizon"]-1),
        formal_eval_calls=eval_calls,eval_smoke_calls_reused=4*min(DESIGN["smoke_frames"],len(old["entries"][0]["frames"])-1)*len(POLICIES),
        **{k:copy.deepcopy(old[k]) for k in ("entries","folds","root","search_size","search_factor","window")},
        code_sha256={str(Path(__file__).with_name(n)):file_hash(Path(__file__).with_name(n)) for n in FILES})),source


def load_plan(root):
    p=sealed(Path(root)/"manifest.json");expected,source=make_plan(p["source_input"],p["hardware"])
    if p!=expected:raise ValueError("Frozen Stage-68 source/code/fit sampling changed")
    return p,source


def pair_feature(f,q):
    f=np.asarray(f,float)
    if f.shape!=(4,64) or q not in (1,2,3) or not np.isfinite(f).all():raise ValueError("Need a causal non-q0 Q proposal")
    return np.r_[f[q]-f[0],f[0]]


def fit_regressor(x,y,w,ridge,target):
    x,y,w=np.asarray(x,float),np.asarray(y,float),np.asarray(w,float)
    if x.ndim!=2 or x.shape[1]!=128 or y.shape!=(len(x),) or w.shape!=y.shape or not len(x):raise ValueError("Invalid paired fit dimensions")
    if target not in TARGETS or not all(np.isfinite(v).all() for v in (x,y,w)) or np.any(w<=0) or np.any(np.abs(y)>1+1e-12):raise ValueError("Invalid paired fit values")
    w=w/w.sum();center=w@x;scale=np.maximum(np.sqrt(w@((x-center)**2)),1e-6);z=(x-center)/scale;bias=float(w@y)
    coef=np.linalg.solve(z.T@(w[:,None]*z)+ridge*np.eye(128),z.T@(w*(y-bias)))
    return dict(target=target,center=center.tolist(),scale=scale.tolist(),weights=coef.tolist(),bias=bias,pairs=len(x))


def predict(x,m):
    return float(m["bias"]+((np.asarray(x,float)-np.asarray(m["center"]))/np.asarray(m["scale"]))@np.asarray(m["weights"]))


def decide(policy,base,f,pack,design,controller_design):
    source=pack["source"]
    d=prior.prior.decide(policy if policy in ("hann","q0","quality","protect") else "quality",base,f,source,controller_design)
    q=d["quality_choice"]
    immediate,future=(predict(pair_feature(f,q),pack[t]) for t in TARGETS) if q else (0.,0.)
    if policy in ("immediate","future","rate_margin"):
        gain={"immediate":immediate,"future":future,"rate_margin":d["quality_gain"]}[policy]
        threshold=pack["rate_margin"]["threshold"] if policy=="rate_margin" else design["tie_epsilon"]
        d["selected"]=q if q and gain>threshold else 0
        d["veto"]=bool(q and d["selected"]==0)
    elif policy not in ("hann","q0","quality","protect"):raise ValueError("Unknown Stage-68 policy")
    return dict(d,immediate_gain=immediate,future_gain=future)


def pair_path(root,case,action=None):return Path(root)/"collection/traces"/(case["case_id"]+"__"+(action or "shared")+".npz")
def pair_meta(plan,case,action=None):return dict(plan_fingerprint=plan["fingerprint"],case=case,kind="fit_pair",action=action)
def eval_path(root,unit,smoke=False):return Path(root)/("smoke/traces" if smoke else "evaluation/traces")/(prior.prior.key(unit)+".npz")
def eval_meta(plan,unit,models,smoke=False):return dict(plan_fingerprint=plan["fingerprint"],unit=unit,kind="smoke" if smoke else "evaluation",training_fingerprint=models["fingerprint"])


def save(path,data,meta):
    path=Path(path)
    if path.with_suffix(".receipt.json").exists():raise ValueError("Never overwrite committed Stage-68 cache")
    if path.exists():
        with np.load(path,allow_pickle=False) as f:
            if json.loads(str(f["metadata_json"].item()))!=meta:raise ValueError("Unknown orphan cache")
    write_cache(path,data,meta)
    publish_identical(path.with_suffix(".receipt.json"),seal(dict(schema=SCHEMA,metadata=meta,sha256=file_hash(path))))


def read_cache(path,meta):
    path=Path(path)
    if not path.with_suffix(".receipt.json").exists():return None
    r=sealed(path.with_suffix(".receipt.json"))
    if r.get("metadata")!=meta or not path.is_file() or file_hash(path)!=r["sha256"]:raise ValueError("Stage-68 cache changed")
    with np.load(path,allow_pickle=False) as f:
        if json.loads(str(f["metadata_json"].item()))!=meta:raise ValueError("Mixed Stage-68 cache")
        return {k:f[k] for k in f.files if k!="metadata_json"}
