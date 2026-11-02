"""Original V1 versus E: own-history proposals and matched four-pass recropping."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from stage4_common import file_hash, fingerprint, write_cache
from stage27_uav_common import publish_identical, crop_geometry
from stage46_shift_audit import verify_files
import stage60_state_common as prior
import stage60_state_experiment as previous

SCHEMA = "stage61_v1_history_matched_recrop_v1"
POLICIES = ("current", "history", "recrop")
ORIGINS = ("native", "endpoint")
DESIGN = dict(history=5, tie_epsilon=1e-6, feature_clip=10., box_atol=.001,
              source_box_atol=0., smoke_frames=32, bootstrap_seed=2161, bootstrap_samples=20000,
              practical_auc_gain=.5, reliable_ci_floor=-1.)
FILES = ("stage61_search_common.py", "stage61_search_runtime.py", "stage61_search_experiment.py",
         "stage61_search_summary.py", "test_stage61_search.py")

def seal(body): return dict(body, fingerprint=fingerprint(body))

def read_sealed(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("fingerprint") != fingerprint({k:v for k,v in value.items() if k!="fingerprint"}):
        raise ValueError("Stage-61 sealed content changed: " + str(path))
    return value

def load_source(root):
    """Read completed Stage-60 evidence; never rerun its summary or inference."""
    root = Path(root).expanduser().resolve()
    plan, source = prior.load_plan(root)
    previous.smoke_gate(plan, root)
    collection = previous.collection_gate(plan, root)
    records = [(r,None) for r in source["records"]]
    # The scorers are provenance objects only; verify sealed packs without fitting them again.
    models, training = previous.load_models(plan, root, records, recompute=False)
    run = prior.read_sealed(root/"evaluation/run.json")
    report = prior.read_sealed(root/"analysis/stage60_decision.json")
    tables = {"stage60_"+n+".csv" for n in ("policy_summary","video_summary","seed_video_summary","paired",
              "scope_summary","fit_support","retention_summary","focus","events")}
    if (run.get("status") != "COMPLETE_CAUSAL_OOF" or run.get("plan_fingerprint") != plan["fingerprint"] or
        run.get("run_fingerprint") != previous.run_record(plan)["fingerprint"] or
        run.get("training_fingerprint") != training["fingerprint"] or
        run.get("full_trajectories") != plan["expected_evaluation_trajectories"] or
        run.get("full_search_calls") != plan["expected_evaluation_calls"] or
        report.get("status") not in ("REVIEW_CAUSAL_SEARCH_STATE_PILOT","PROMISING_CAUSAL_SEARCH_STATE_PILOT") or
        report.get("role") != "consumed_development_causal_search_state_value" or
        report.get("plan_fingerprint") != plan["fingerprint"] or report.get("run_fingerprint") != run["fingerprint"] or
        report.get("evidence_sha256") != run.get("cache_sha256") or
        report.get("model_sha256") != training["model_sha256"] or set(report.get("table_sha256",{})) != tables or
        report.get("maximum_control_box_error") != 0 or not 0 <= report.get("maximum_decode_error",float("inf")) <= .001 or
        report.get("sgla_weights_unchanged") is not True):
        raise ValueError("Need complete unmixed Stage-60 source")
    bindings = dict(source["bindings"])
    for mapping in (plan["source_file_sha256"], run["cache_sha256"], collection["cache_sha256"], training["model_sha256"]):
        bindings.update(mapping)
    bindings.update({str(root/"analysis"/n):h for n,h in report["table_sha256"].items()})
    for name in ("manifest.json","evaluation/run.json","analysis/stage60_decision.json","smoke/stage60_smoke.json",
                 "fit/collection.json","models/training.json"):
        bindings[str(root/name)] = file_hash(root/name)
    verify_files(bindings)
    reused = {}
    for entry in plan["entries"]:
        name = entry["sequence"]
        fold = next(f["fold"] for f in plan["folds"] if name in f["held_out"])
        native = source["arrays"][name][1]["native"]
        reused[name,"native","current",0] = native
        for seed in plan["seeds"]:
            model = models[fold,seed]
            for policy, label in (("current","current"),("response","history")):
                path = root/"evaluation/traces"/(name+"__s{}__{}.npz".format(seed,policy))
                data = prior.read_cache(path,prior.meta(plan,"oof",sequence=name,fold=fold,seed=seed,
                                        policy=policy,model_fingerprint=model["fingerprint"]))
                if data is None or len(data["boxes"]) != len(entry["frames"])-1: raise ValueError("Missing full reused trace")
                previous.audit_evaluation(data,entry,plan,policy,model["models"],source["arrays"][name][1]["endpoint_control_seed"+str(seed)])
                reused[name,"endpoint",label,seed] = data
    return dict(stage60=root, plan60=plan, old=source, records=source["records"], reused=reused, bindings=bindings, report=report)

def make_plan(root):
    source = load_source(root); old = source["plan60"]
    prior.validate_folds(old)
    units = []
    for ordinal, entry in enumerate(old["entries"]):
        name = entry["sequence"]
        fold = next(f["fold"] for f in old["folds"] if name in f["held_out"])
        own = [dict(sequence=name,origin="native",policy=p,seed=0,fold=-1) for p in ("history","recrop")]
        own += [dict(sequence=name,origin="endpoint",policy="recrop",seed=s,fold=fold) for s in old["seeds"]]
        offset = ordinal % len(own)
        units.extend(own[offset:]+own[:offset])
    frames = {e["sequence"]:len(e["frames"])-1 for e in old["entries"]}
    calls = 4*sum(frames[u["sequence"]] for u in units)
    body = dict(schema=SCHEMA,role="consumed_development_v1_history_vs_same_frame_recrop",source_input=str(source["stage60"]),
        source_decision_fingerprint=source["report"]["fingerprint"],source_file_sha256=source["bindings"],design=dict(DESIGN),
        origins=list(ORIGINS),policies=list(POLICIES),units=units,seeds=old["seeds"],new_full_trajectories=len(units),
        formal_search_calls=calls,reused_full_trajectories=len(source["reused"]),
        smoke_search_calls=min(DESIGN["smoke_frames"],frames[old["entries"][0]["sequence"]])*9*(1+len(old["seeds"])),
        **{k:old[k] for k in ("entries","folds","root","runtime","runtime_identity","search_size","search_factor","window")},
        code_sha256={n:file_hash(Path(__file__).with_name(n)) for n in FILES+prior.FILES+
                     ("stage47_search_collect.py","stage54_rollout_runtime.py","stage48_stability_runtime.py")})
    return seal(body),source

def load_plan(root):
    plan = read_sealed(Path(root)/"manifest.json")
    expected,source = make_plan(plan["source_input"])
    if expected != plan: raise ValueError("Frozen Stage-61 source/code/design changed")
    return plan,source

def recrop_parents(latest, first_box):
    """Current-image q0 output provides the second center/size; no amplitude knob."""
    latest, first_box = np.asarray(latest,float), np.asarray(first_box,float)
    if latest.shape!=(4,) or first_box.shape!=(4,) or not np.isfinite([latest,first_box]).all() or np.any(np.array([latest,first_box])[:,2:]<=0):
        raise ValueError("Invalid decoded parent components")
    a = latest[:2]+latest[2:]/2; b = first_box[:2]+first_box[2:]/2
    return np.array([latest.copy(),np.r_[a-first_box[2:]/2,first_box[2:]],
                     np.r_[b-latest[2:]/2,latest[2:]],first_box.copy()])

def key(unit): return "{}__{}__s{}__{}".format(unit["sequence"],unit["origin"],unit["seed"],unit["policy"])
def meta(plan, unit, kind): return dict(plan_fingerprint=plan["fingerprint"],kind=kind,**unit)

def save(path,data,metadata):
    path = Path(path)
    if path.with_suffix(".receipt.json").exists(): raise ValueError("Never overwrite committed Stage-61 cache")
    if path.exists():
        with np.load(path,allow_pickle=False) as f:
            if json.loads(str(f["metadata_json"].item())) != metadata: raise ValueError("Unidentified orphan cache")
    write_cache(path,data,metadata)
    publish_identical(path.with_suffix(".receipt.json"),seal(dict(schema=SCHEMA,metadata=metadata,sha256=file_hash(path))))

def read_cache(path,metadata):
    path = Path(path); receipt = path.with_suffix(".receipt.json")
    if not receipt.exists(): return None
    value = read_sealed(receipt)
    if value.get("metadata") != metadata or not path.is_file() or file_hash(path) != value["sha256"]:
        raise ValueError("Committed Stage-61 cache changed")
    with np.load(path,allow_pickle=False) as f:
        if json.loads(str(f["metadata_json"].item())) != metadata: raise ValueError("Mixed Stage-61 cache")
        return {k:f[k] for k in f.files if k!="metadata_json"}
