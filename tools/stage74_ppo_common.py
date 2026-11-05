"""Direct four-candidate PPO design; frozen SGLA and fixed source training budget."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import stage71_ppo_common as base
import stage73_ppo_common as previous
import stage73_ppo_experiment as e73
import stage72_ppo_common as diagnostic
import stage70_history_common as prior
import stage70_history_experiment as e70
from stage4_common import file_hash, fingerprint

SCHEMA = "stage74_direct_four_candidate_ppo_v1"
FILES = ("stage74_ppo_common.py", "stage74_ppo_runtime.py", "stage74_ppo_experiment.py",
         "stage74_ppo_summary.py", "test_stage74_ppo.py", "run_stage74_ppo.sh")
DESIGN = dict(previous.DESIGN, input_dim=256, actions=4)
CONTROLS = ("q0", "hann", "quality", "future")
LEARNED = ("ppo", "myopic")
REUSED = ("binary_ppo", "binary_myopic")
SOURCE_POLICIES = dict(binary_ppo="ppo", binary_myopic="myopic")
c68, c66, e63 = prior.c68, prior.c66, prior.e63
seal, VERIFY, publish_identical = prior.seal, prior.VERIFY, prior.publish_identical


def sealed(path): return c66.sealed(path, SCHEMA)


def load_source(root):
    root=Path(root).expanduser().resolve();plan,old=previous.load_plan(root)
    models=e73.load_models(root,plan,old);run=e73.completed(root,plan,old,models)
    report=previous.sealed(root/"analysis/stage73_decision.json");bridge=e73.bridge_gate(root,plan)
    names=("policy_summary","paired","factor_summary","calibration","training_scope","fit_support","focus","scope","video_influence",
           "seed_effects","seed_video_summary","video_summary","events","training_episode","training_summary","optimization_summary","training_log")
    if (report.get("status") not in ("REVIEW_SEPARATE_CLIP_PPO_PILOT","PROMISING_SEPARATE_CLIP_PPO_PILOT") or
        report.get("plan_fingerprint")!=plan["fingerprint"] or report.get("training_fingerprint")!=models["fingerprint"] or
        report.get("run_fingerprint")!=run["fingerprint"] or report.get("runtime")!=plan["runtime_identity"] or
        report.get("source_report_fingerprint")!=old["diagnostic_report"]["fingerprint"] or
        report.get("source_training_fingerprint")!=old["stage71"]["models"]["fingerprint"] or
        report.get("source_bindings_fingerprint")!=fingerprint(plan["source_file_sha256"]) or
        report.get("training_evidence_fingerprint")!=fingerprint(models["training_evidence_sha256"]) or
        report.get("eval_evidence_fingerprint")!=fingerprint(run["cache_sha256"]) or
        report.get("source_bridge_fingerprint")!=bridge["fingerprint"] or report.get("sgla_training_performed") is not False or
        report.get("reinforcement_learning_performed") is not True or report.get("maximum_source_control_error")!=0. or
        any(report.get(k) is not True for k in ("source_quality_unchanged","source_joint_models_unchanged","all_source_policies_exactly_reproduced")) or
        set(report.get("table_sha256",{}))!={"stage73_"+n+".csv" for n in names}):
        raise ValueError("Need complete Stage-73 result with immutable training/evaluation and upstream sources")
    bindings=dict(plan["source_file_sha256"]);bindings.update(plan["code_sha256"])
    bindings.update(models["training_evidence_sha256"]);bindings.update(run["cache_sha256"]);bindings.update(bridge["cache_sha256"])
    bindings.update({str(root/"analysis"/n):h for n,h in report["table_sha256"].items()})
    bindings.update({str(root/n):file_hash(root/n) for n in ("manifest.json","models/training.json","evaluation/run.json",
        "analysis/stage73_decision.json","smoke/train.json","smoke/evaluate.json","smoke/bridge.json")})
    VERIFY(bindings)
    return dict(old,stage73=dict(root=root,plan=plan,old=old,models=models,run=run,report=report),bindings=bindings)


def entry_for(plan, seq): return next(e for e in plan["entries"] if e["sequence"] == seq)
def fit_pack(source, fold): return prior.fit_pack(source["upstream"], fold)
def head_for(source, unit): return prior.head_for(source["upstream"], dict(unit, seed=0))


def make_plan(root,hardware):
    source=load_source(root);old=source["stage73"]["plan"]
    if set(hardware)!={"device","gpu_uuid"} or hardware["device"]!="NVIDIA L40" or not hardware["gpu_uuid"]:raise ValueError("Need observed physical L40 binding")
    identity=copy.deepcopy(old["runtime_identity"]);identity["gpu_uuid"]=hardware["gpu_uuid"]
    changes=e63.startup.differences(old["runtime_identity"],identity)
    if any(x["field"]!="runtime.gpu_uuid" for x in changes):raise ValueError("Only physical L40 UUID may change")
    if {k:v for k,v in DESIGN.items() if k not in ("input_dim","actions")}!=old["design"] or DESIGN["input_dim"]!=256 or DESIGN["actions"]!=4:
        raise ValueError("Keep source optimizer, reward, seeds and budget; only direct four-candidate representation/actions change")
    jobs=copy.deepcopy(old["jobs"])
    units=[dict(sequence=x["sequence"],origin="native",fold=next(f["fold"] for f in old["folds"] if x["sequence"] in f["held_out"]),policy=p,seed=seed)
        for x in old["entries"] for p in CONTROLS+REUSED+LEARNED for seed in ([0] if p in CONTROLS else DESIGN["seeds"])]
    frames={x["sequence"]:len(x["frames"])-1 for x in old["entries"]}
    return seal(dict(schema=SCHEMA,role="consumed_development_direct_four_candidate_ppo",source_input=str(Path(root).resolve()),
        source_report_fingerprint=source["stage73"]["report"]["fingerprint"],source_training_fingerprint=source["stage73"]["models"]["fingerprint"],
        source_file_sha256=source["bindings"],design=copy.deepcopy(DESIGN),jobs=jobs,units=units,controls=list(CONTROLS),reused=list(REUSED),learned=list(LEARNED),
        training_environment_steps=old["training_environment_steps"],training_search_calls=old["training_search_calls"],optimizer_updates_per_arm=old["optimizer_updates_per_arm"],
        formal_eval_calls=sum(4*frames[u["sequence"]] for u in units),source_bridge_search_calls=old["source_bridge_search_calls"],train_smoke_calls_reused=old["train_smoke_calls_reused"],
        eval_smoke_calls_reused=4*min(DESIGN["smoke_frames"],frames[old["entries"][0]["sequence"]])*(len(CONTROLS)+(len(REUSED)+len(LEARNED))*len(DESIGN["seeds"])),
        hardware=hardware,runtime=identity["tracker"],runtime_identity=identity,declared_runtime_differences=changes,
        **{k:copy.deepcopy(old[k]) for k in ("controller_design","entries","folds","root","search_size","search_factor","window","actor_design")},
        code_sha256={str(Path(__file__).with_name(n)):file_hash(Path(__file__).with_name(n)) for n in FILES})),source


def load_plan(root):
    plan = sealed(Path(root) / "manifest.json"); expected, source = make_plan(plan["source_input"], plan["hardware"])
    if plan != expected: raise ValueError("Frozen Stage-74 source/code/design changed")
    return plan, source


def check_fit(plan, job, seq):
    split = next(s for s in plan["folds"] if s["fold"] == job["fold"])
    if seq not in split["fit"] or seq in split["held_out"]: raise ValueError("Training crosses held-out source group")



def normalized(features,norm,design):
    x=np.asarray(features,float).reshape(-1);center=np.asarray(norm["center"],float);scale=np.asarray(norm["scale"],float)
    if x.shape!=(256,) or center.shape!=x.shape or scale.shape!=x.shape or not np.isfinite(np.r_[x,center,scale]).all() or np.any(scale<=0):
        raise ValueError("Need fit-normalized four-by-64 causal state")
    return np.clip((x-center)/scale,-design["feature_clip"],design["feature_clip"])


def forward(x,state,branch):
    x=np.asarray(x,float);h=np.tanh(x@np.asarray(state[branch+".0.weight"]).T+np.asarray(state[branch+".0.bias"]))
    out=h@np.asarray(state[branch+".2.weight"]).T+np.asarray(state[branch+".2.bias"])
    return out if branch=="actor" else np.squeeze(out,axis=-1)


def action_mask(boxes):
    b=np.asarray(boxes,float)
    if b.shape!=(4,4) or not np.isfinite(b).all() or np.any(b[:,2:]<=0):raise ValueError("Need four finite public boxes")
    return np.array([not any(np.array_equal(b[i],b[j]) for j in range(i)) for i in range(4)],bool)


def distribution(logits,mask):
    z,m=np.asarray(logits,float),np.asarray(mask,bool)
    if z.shape!=(4,) or m.shape!=(4,) or not m[0] or not np.isfinite(z).all():raise ValueError("Invalid categorical decision")
    maximum=z[m].max();total=float(np.exp(z[m]-maximum).sum());lp=np.full(4,-np.inf)
    lp[m]=z[m]-maximum-np.log(total);p=np.zeros(4);p[m]=np.exp(lp[m])
    return p,lp


def gate(features,boxes,pack,design):
    x=normalized(features,pack["norm"],design);mask=action_mask(boxes)
    logits=forward(x,pack["state"],"actor");probs,_=distribution(logits,mask)
    return dict(x=x,action_mask=mask,logits=logits,probabilities=probs,value=float(forward(x,pack["state"],"critic")),eligible=bool(mask.sum()>1))


def log_probability(action,logits,mask):
    if not isinstance(action,(int,np.integer)) or not 0<=action<4 or not mask[action]:raise ValueError("Illegal categorical action")
    return float(distribution(logits,mask)[1][action])


def best_label(overlaps,mask,epsilon):
    y,m=np.asarray(overlaps,float),np.asarray(mask,bool)
    if y.shape!=(4,) or m.shape!=(4,) or not m[0] or not np.isfinite(y).all() or np.any((y<0)|(y>1)):raise ValueError("Invalid offline candidate labels")
    return int(np.flatnonzero(m & (y>=y[m].max()-epsilon))[0])


gae=base.gae


def key(unit): return c66.key(unit)
def job_key(job): return "f{}__s{}".format(job["fold"], job["seed"])
def trace_path(root, job, index): return Path(root)/"training"/job_key(job)/("episode{:03d}.npz".format(index))
def step_path(root, job, index): return Path(root)/"training"/job_key(job)/("step{:03d}.json".format(index))
def eval_path(root, unit, smoke=False): return Path(root)/("smoke/evaluation" if smoke else "evaluation/traces")/(key(unit)+".npz")
def save(path, data, meta):
    # Identical immutable NPZ+receipt format, in a new namespace/schema.
    path = Path(path)
    if path.with_suffix(".receipt.json").exists(): raise ValueError("Never overwrite committed Stage-74 cache")
    if path.exists():
        import json
        with np.load(path, allow_pickle=False) as old:
            if json.loads(str(old["metadata_json"].item())) != meta: raise ValueError("Unknown orphan Stage-74 cache")
    from stage4_common import write_cache
    write_cache(path, data, meta)
    publish_identical(path.with_suffix(".receipt.json"), seal(dict(schema=SCHEMA, metadata=meta, sha256=file_hash(path))))


def read_cache(path, meta):
    path = Path(path)
    if not path.with_suffix(".receipt.json").exists(): return None
    r = sealed(path.with_suffix(".receipt.json"))
    if r.get("metadata") != meta or file_hash(path) != r["sha256"]: raise ValueError("Stage-74 cache changed")
    import json
    with np.load(path, allow_pickle=False) as f:
        if json.loads(str(f["metadata_json"].item())) != meta: raise ValueError("Mixed Stage-74 cache")
        return {k: f[k] for k in f.files if k != "metadata_json"}


def source_unit(unit):
    return dict(unit,policy=SOURCE_POLICIES.get(unit["policy"],unit["policy"]))


def source_trace(source,unit):
    old=source["stage73"];u=source_unit(unit)
    data=previous.read_cache(previous.eval_path(old["root"],u),e73.eval_meta(old["plan"],u,old["models"]))
    if data is None:raise ValueError("Missing immutable Stage-73 control")
    return data




def fit_normalization(plan,source,fold):
    split=next(f for f in plan["folds"] if f["fold"]==fold);parts=[]
    for seq in split["fit"]:
        if seq in split["held_out"]:raise ValueError("Normalization crosses held-out groups")
        data,_,_=c68.prior.read_unit(source["upstream"]["s68"]["old"]["s66"],seq,"hann")
        x=np.asarray(data["choice_features"],float).reshape(-1,256)
        parts.append((x,1/(len(split["fit"])*len(x))))
    center=sum(x.sum(0)*w for x,w in parts);variance=sum(((x-center)**2).sum(0)*w for x,w in parts)
    return dict(center=center.tolist(),scale=np.maximum(np.sqrt(variance),1e-6).tolist())
