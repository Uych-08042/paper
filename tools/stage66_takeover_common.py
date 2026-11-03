"""Native SGLA takeover protection with an absolute q0 critic and a rate control."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
import stage65_selection_audit as prior
from stage4_common import file_hash, fingerprint, write_cache
from stage27_uav_common import publish_identical
from validation_metrics import IOU_PROTOCOL

c64, e64 = prior.c64, prior.e64
SCHEMA = "stage66_native_absolute_q0_takeover_v1"
FILES = ("stage66_takeover_common.py", "stage66_takeover_experiment.py", "stage66_takeover_summary.py",
         "test_stage66_takeover.py", "run_stage66_takeover.sh")
DESIGN = dict(ridge=.01, q0_good_threshold=.5, protected_gain=.02, tie_epsilon=1e-6,
              smoke_frames=32, bootstrap_seed=2166, bootstrap_samples=20000, practical_gain=.5, reliable_floor=-1.)
POLICIES = ("hann", "q0", "quality", "margin", "protect")
seal = c64.seal
VERIFY = e64.VERIFY


def sealed(path, schema=SCHEMA):
    v = json.loads(Path(path).read_text(encoding="utf8"))
    if v.get("schema") != schema or v.get("fingerprint") != fingerprint({k:x for k,x in v.items() if k != "fingerprint"}):
        raise ValueError("Sealed content changed: "+str(path))
    return v


def load_source(root):
    root = Path(root).expanduser().resolve(); p65 = sealed(root/"manifest.json", prior.c.SCHEMA)
    s64 = prior.load_source(p65["source_input"])
    code = {str(Path(prior.__file__).with_name(n)):file_hash(Path(prior.__file__).with_name(n)) for n in prior.c.FILES}
    expected = seal(dict(schema=prior.c.SCHEMA, role="consumed_development_cached_common_state_selector_audit",
        source_input=str(s64["root"]), source_plan_fingerprint=s64["plan"]["fingerprint"], source_report_fingerprint=s64["report"]["fingerprint"],
        source_training_fingerprint=s64["models"]["fingerprint"], source_file_sha256=s64["bindings"], code_sha256=code,
        design=dict(prior.c.DESIGN), cases=prior.cases(s64), iou_protocol=IOU_PROTOCOL, rules=list(prior.c.RULES),
        unit_definition="four selectors on the same stored predicted history and four public boxes; no state advancement"))
    if p65 != expected: raise ValueError("Frozen Stage-65 source or design differs")
    report = sealed(root/"analysis/stage65_diagnostic.json", prior.c.SCHEMA)
    ref = sealed(root/"reference.json", prior.c.SCHEMA)
    tables = {"stage65_"+n+".csv" for n in ("policy_summary", "matched_summary", "feature_shift", "scope", "focus", "video_summary",
              "seed_summary", "matched_video", "feature_video", "feature_seed", "focus_frames")}
    if (report.get("status") != "COMPLETE_CACHED_SAME_STATE_SELECTOR_DIAGNOSTIC" or report.get("plan_fingerprint") != p65["fingerprint"] or
        report.get("source_report_fingerprint") != s64["report"]["fingerprint"] or report.get("source_training_fingerprint") != s64["models"]["fingerprint"] or
        report.get("reference_fingerprint") != ref["fingerprint"] or ref.get("plan_fingerprint") != p65["fingerprint"] or
        report.get("source_bindings_fingerprint") != fingerprint(s64["bindings"]) or report.get("source_runtime") != s64["plan"]["runtime_identity"] or
        set(report.get("table_sha256", {})) != tables or report.get("training_performed") is not False or
        any(report.get(k) != 0 for k in ("new_network_calls", "new_image_reads", "new_models", "new_full_trajectories"))):
        raise ValueError("Need the complete unmixed Stage-65 diagnostic")
    bindings = dict(s64["bindings"]); bindings.update(code)
    bindings.update({str(root/"analysis"/n):h for n,h in report["table_sha256"].items()})
    for n in ("manifest.json", "reference.json", "analysis/stage65_diagnostic.json"): bindings[str(root/n)] = file_hash(root/n)
    VERIFY(bindings)
    return dict(root=root, report=report, s64=s64, bindings=bindings)


def make_plan(root, hardware):
    source = load_source(root); old = source["s64"]["plan"]
    if set(hardware) != {"device", "gpu_uuid"} or hardware["device"] != "NVIDIA L40" or not hardware["gpu_uuid"]:
        raise ValueError("Need an observed L40 binding")
    identity = copy.deepcopy(old["runtime_identity"]); identity["gpu_uuid"] = hardware["gpu_uuid"]
    changes = c64.e63.startup.differences(old["runtime_identity"], identity)
    if any(r["field"] != "runtime.gpu_uuid" for r in changes): raise ValueError("Only physical L40 UUID may change")
    units = [dict(sequence=e["sequence"], origin="native", seed=0,
                  fold=next(f["fold"] for f in old["folds"] if e["sequence"] in f["held_out"]), policy=p)
             for e in old["entries"] for p in POLICIES]
    counts = {e["sequence"]:len(e["frames"])-1 for e in old["entries"]}
    return seal(dict(schema=SCHEMA, role="consumed_development_native_q0_takeover_protection", source_input=str(source["root"]),
        source_report_fingerprint=source["report"]["fingerprint"], source_file_sha256=source["bindings"], design=dict(DESIGN),
        hardware=hardware, runtime=identity["tracker"], runtime_identity=identity, declared_runtime_differences=changes,
        units=units, policies=list(POLICIES), formal_search_calls=sum(4*counts[u["sequence"]] for u in units),
        smoke_calls_reused=4*min(DESIGN["smoke_frames"], counts[old["entries"][0]["sequence"]])*len(POLICIES),
        actor_design=copy.deepcopy(old["actor_design"]),
        **{k:copy.deepcopy(old[k]) for k in ("entries", "folds", "root", "search_factor", "search_size", "window")},
        code_sha256={str(Path(__file__).with_name(n)):file_hash(Path(__file__).with_name(n)) for n in FILES})), source


def load_plan(root):
    p = sealed(Path(root)/"manifest.json"); expected, source = make_plan(p["source_input"], p["hardware"])
    if p != expected: raise ValueError("Frozen Stage-66 source/code/design changed")
    return p, source


def fit_absolute(f, overlaps, weights, ridge):
    f, y, w = np.asarray(f, float), np.asarray(overlaps, float), np.asarray(weights, float)
    if f.ndim != 3 or f.shape[1:] != (4,64) or y.shape != f.shape[:2] or w.shape != (len(f),) or not len(f):
        raise ValueError("Need fit-only current-q0 examples")
    if not all(np.isfinite(v).all() for v in (f,y,w)) or np.any(w<=0) or np.any((y<0)|(y>1)): raise ValueError("Invalid critic labels")
    w = w/w.sum(); x = f[:,0]; center = w@x; scale = np.maximum(np.sqrt(w@((x-center)**2)), 1e-6)
    z = (x-center)/scale; bias = float(w@y[:,0])
    coef = np.linalg.solve(z.T@(w[:,None]*z)+ridge*np.eye(64), z.T@(w*(y[:,0]-bias)))
    return dict(target="current_q0_iou", center=center.tolist(), scale=scale.tolist(), weights=coef.tolist(), bias=bias, states=len(f))


def absolute(f, critic):
    v = np.asarray(f, float)
    if v.shape != (4,64) or not np.isfinite(v).all(): raise ValueError("Invalid causal critic input")
    return float(np.clip(critic["bias"]+((v[0]-np.asarray(critic["center"]))/np.asarray(critic["scale"]))@np.asarray(critic["weights"]), 0., 1.))


def match_margin(gains, eligible, protected, weights):
    gains, eligible, protected, w = np.asarray(gains,float), np.asarray(eligible,bool), np.asarray(protected,bool), np.asarray(weights,float)
    if not (gains.shape==eligible.shape==protected.shape==w.shape) or not np.isfinite(gains).all() or np.any(w<=0) or np.any(protected & ~eligible):
        raise ValueError("Invalid fit acceptance calibration")
    w = w/w.sum(); target = float(w[protected].sum())
    values = np.unique(gains[eligible])[::-1]
    masses = np.array([float(w[eligible & (gains==v)].sum()) for v in values])
    totals = np.r_[0.,np.cumsum(masses)]; k = int(np.argmin(np.abs(totals-target)))
    # Strict gain > cutoff accepts exactly the chosen prefix, even for adjacent floats.
    threshold = (0. if k==len(values) else float(values[k])) if len(values) else 0.
    accepted = eligible & (gains>threshold)
    achieved = float(w[accepted].sum())
    return dict(threshold=threshold, protected_fit_acceptance=target, margin_fit_acceptance=achieved,
                acceptance_mismatch=abs(achieved-target), largest_tied_gain_mass=float(masses.max()) if len(masses) else 0.,
                quality_fit_acceptance=float(w[eligible].sum()))


def decide(policy, base, f, pack, design):
    q, quality_scores = c64.choose("quality", base, f, pack["quality"], design["tie_epsilon"])
    gain = float(quality_scores[q]); prediction = absolute(f, pack["critic"])
    qualified = prediction >= design["q0_good_threshold"]
    if policy in ("hann", "q0"):
        selected, scores = c64.choose(policy, base, f, None, design["tie_epsilon"])
    elif policy in ("quality", "margin", "protect"):
        selected, scores = q, quality_scores
        if policy=="margin" and gain<=pack["margin"]["threshold"]: selected=0
        if policy=="protect" and qualified and gain<=design["protected_gain"]: selected=0
    else: raise ValueError("Unknown takeover policy")
    return dict(selected=selected, scores=scores, q0_pred_iou=prediction, q0_qualified=qualified,
                quality_choice=q, quality_gain=gain, veto=bool(policy in ("margin","protect") and q!=0 and selected==0))


key = c64.key
def metadata(plan, unit, kind, models):
    return dict(plan_fingerprint=plan["fingerprint"], unit=unit, kind=kind, training_fingerprint=models["fingerprint"])


def save(path, data, meta):
    path=Path(path)
    if path.with_suffix(".receipt.json").exists(): raise ValueError("Never overwrite committed Stage-66 cache")
    if path.exists():
        with np.load(path,allow_pickle=False) as f:
            if json.loads(str(f["metadata_json"].item())) != meta: raise ValueError("Unknown orphan cache")
    write_cache(path,data,meta)
    publish_identical(path.with_suffix(".receipt.json"), seal(dict(schema=SCHEMA, metadata=meta, sha256=file_hash(path))))


def read_cache(path, meta):
    path=Path(path)
    if not path.with_suffix(".receipt.json").exists(): return None
    receipt=sealed(path.with_suffix(".receipt.json"))
    if receipt.get("metadata") != meta or not path.is_file() or file_hash(path)!=receipt["sha256"]: raise ValueError("Stage-66 cache changed")
    with np.load(path,allow_pickle=False) as f:
        if json.loads(str(f["metadata_json"].item())) != meta: raise ValueError("Mixed Stage-66 cache")
        return {k:f[k] for k in f.files if k!="metadata_json"}
