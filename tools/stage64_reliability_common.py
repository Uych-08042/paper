"""Fit-only native-state quality/reliability selectors; frozen SGLA weights."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import stage63_choice_common as prior
import stage63_choice_experiment as e63
from stage4_common import file_hash, fingerprint, write_cache
from stage27_uav_common import publish_identical
from stage28_response_common import continuous_iou

SCHEMA = "stage64_causal_candidate_reliability_v1"
DESIGN = dict(ridge=.01, good_iou=.5, low_iou=.2, tie_epsilon=1e-6, smoke_frames=32,
              bootstrap_seed=2164, bootstrap_samples=20000, practical_gain=.5, reliable_floor=-1.)
FILES = ("stage64_reliability_common.py", "stage64_reliability_experiment.py", "stage64_reliability_summary.py",
         "test_stage64_reliability.py", "run_stage64_reliability.sh")
POLICIES = ("hann", "q0", "quality", "reliability")
TARGETS = ("quality", "reliability")
EXTRA = ("consensus_mean", "consensus_max", "q0_agreement", "velocity_residual", "median_size_error",
         "center_disagreement", "size_disagreement", "hann_rank")
BASE = tuple(e63.source_common.prior.FEATURES)+EXTRA
FEATURES = BASE+tuple(n+"_square" for n in BASE)
seal = prior.seal


def read_sealed(path):
    return prior.read_sealed(path, SCHEMA)


def load_source(root):
    root = Path(root).expanduser().resolve()
    plan, inputs = prior.load_plan(root)
    run = e63.completed(plan, root)
    report = prior.read_sealed(root/"analysis/stage63_diagnostic.json")
    tables = {"stage63_"+n+".csv" for n in ("case_summary", "branch_summary", "contrasts", "scope", "dense_focus", "timelines")}
    if (report.get("status") != "COMPLETE_MATCHED_ONE_CHOICE_CONTINUATION_DIAGNOSTIC" or
        report.get("plan_fingerprint") != plan["fingerprint"] or report.get("run_fingerprint") != run["fingerprint"] or
        report.get("evidence_sha256") != run["cache_sha256"] or report.get("runtime") != plan["runtime_identity"] or
        set(report.get("table_sha256", {})) != tables or report.get("formal_search_calls") != plan["formal_search_calls"] or
        report.get("training_performed") is not False or report.get("maximum_source_box_error") != 0):
        raise ValueError("Need complete unmixed Stage-63 results")
    bindings = dict(plan["source_file_sha256"]); bindings.update(plan["code_sha256"]); bindings.update(run["cache_sha256"])
    bindings.update({str(root/"analysis"/n):v for n,v in report["table_sha256"].items()})
    for name in ("manifest.json", "run.json", "collection/run.json", "smoke/stage63_smoke.json", "analysis/stage63_diagnostic.json"):
        bindings[str(root/name)] = file_hash(root/name)
    prior.prior.c.prior.verify_files(bindings)
    return dict(root=root, plan63=plan, report=report, inputs62=inputs, l40=inputs["source"], bindings=bindings)


def make_plan(root, hardware):
    source = load_source(root); old = source["l40"]["plan"]
    if hardware.get("device") != "NVIDIA L40" or set(hardware) != {"device", "gpu_uuid"} or not hardware["gpu_uuid"]:
        raise ValueError("Need an observed L40 hardware binding")
    e63.source_common.prior.validate_folds(old)
    identity = copy.deepcopy(old["runtime_identity"]); identity["gpu_uuid"] = hardware["gpu_uuid"]
    changes = e63.startup.differences(old["runtime_identity"], identity)
    if any(r["field"] != "runtime.gpu_uuid" for r in changes): raise ValueError("Only L40 UUID may change")
    units = []
    for entry in old["entries"]:
        fold = next(f["fold"] for f in old["folds"] if entry["sequence"] in f["held_out"])
        units.extend(dict(sequence=entry["sequence"], origin="native", seed=0, fold=fold, policy=p) for p in POLICIES)
        units.extend(dict(sequence=entry["sequence"], origin="endpoint", seed=s, fold=fold, policy=p)
                     for s in old["seeds"] for p in ("hann", "reliability"))
    counts = {r["sequence"]:len(r["frames"])-1 for r in old["entries"]}
    plan = seal(dict(schema=SCHEMA, role="consumed_development_gt_free_candidate_reliability", source_input=str(source["root"]),
        source_report_fingerprint=source["report"]["fingerprint"], source_file_sha256=source["bindings"],
        hardware=hardware, runtime=identity["tracker"], runtime_identity=identity, source_runtime=old["runtime_identity"],
        declared_runtime_differences=changes, design=dict(DESIGN), actor_design=copy.deepcopy(old["design"]),
        features=list(FEATURES), units=units, new_full_trajectories=len(units),
        formal_search_calls=sum(4*counts[u["sequence"]] for u in units),
        smoke_calls_reused=4*min(DESIGN["smoke_frames"], counts[old["entries"][0]["sequence"]])*sum(u["sequence"]==old["entries"][0]["sequence"] for u in units),
        **{k:copy.deepcopy(old[k]) for k in ("entries", "folds", "seeds", "root", "search_factor", "search_size", "window")},
        code_sha256={str(Path(__file__).with_name(n)):file_hash(Path(__file__).with_name(n)) for n in FILES}))
    return plan, source


def load_plan(root):
    plan = read_sealed(Path(root)/"manifest.json"); expected, source = make_plan(plan["source_input"], plan["hardware"])
    if plan != expected: raise ValueError("Frozen Stage-64 source/code/design changed")
    return plan, source


def choice_features(base, boxes, history):
    base, boxes, h = np.asarray(base, float), np.asarray(boxes, float), np.asarray(history, float)
    if base.shape != (4, 24) or boxes.shape != (4, 4) or h.ndim != 2 or h.shape[1] != 4 or not len(h):
        raise ValueError("Need four causal candidates and nonempty predicted history")
    if not all(np.isfinite(x).all() for x in (base,boxes,h)) or np.any(boxes[:,2:]<=0) or np.any(h[:,2:]<=0):
        raise ValueError("Invalid causal geometry")
    centers = boxes[:,:2]+boxes[:,2:]/2; hc = h[:,:2]+h[:,2:]/2
    predicted = hc[-1]+(hc[-1]-hc[-2] if len(h)>1 else 0)
    side = np.sqrt(h[-1,2:].prod())
    agreement = np.array([continuous_iou(b, boxes) for b in boxes]); np.fill_diagonal(agreement, 0.)
    delta = np.linalg.norm(centers[:,None]-centers[None], axis=2)/side
    scale = np.abs(np.log(boxes[:,None,2:]/boxes[None,:,2:])).mean(2)
    peak = base[:,1]
    extra = np.column_stack([agreement.sum(1)/3, agreement.max(1), np.array([continuous_iou(b, boxes[:1])[0] for b in boxes]),
        np.linalg.norm(centers-predicted,axis=1)/side, np.abs(np.log(boxes[:,2:])-np.median(np.log(h[:,2:]),axis=0)).mean(1),
        delta.sum(1)/3, scale.sum(1)/3, (peak[:,None]>peak[None]).mean(1)])
    x = np.clip(np.c_[base, extra], -10., 10.)
    return np.c_[x,x*x]


def fit_selector(features, overlaps, weights, target, ridge):
    f, overlaps, w = np.asarray(features,float), np.asarray(overlaps,float), np.asarray(weights,float)
    if f.ndim!=3 or f.shape[1:]!=(4,len(FEATURES)) or overlaps.shape!=f.shape[:2] or w.shape!=(len(f),) or not len(f):
        raise ValueError("Invalid fit-only examples")
    if not all(np.isfinite(x).all() for x in (f,overlaps,w)) or np.any(w<=0) or np.any((overlaps<0)|(overlaps>1)):
        raise ValueError("Invalid training values")
    if target not in TARGETS: raise ValueError("Unknown fixed label")
    labels = overlaps if target=="quality" else (overlaps>=.5).astype(float)-(overlaps<.2).astype(float)
    x = (f-f[:,:1])[:,1:].reshape(-1,len(FEATURES)); y = (labels-labels[:,:1])[:,1:].ravel()
    weight = np.repeat(w/np.sum(w),3)/3
    scale = np.maximum(np.sqrt((weight[:,None]*x*x).sum(0)),1e-6); z = x/scale
    coef = np.linalg.solve(z.T@(weight[:,None]*z)+ridge*np.eye(len(FEATURES)),z.T@(weight*y))
    if not np.isfinite(coef).all(): raise ValueError("Nonfinite selector")
    return dict(target=target, scale=scale.tolist(), weights=coef.tolist(), states=len(f),
                informative_pairs=int((np.abs(y)>1e-6).sum()), weighted_rmse=float(np.sqrt((weight*(z@coef-y)**2).sum())))


def choose(policy, base, features, model, epsilon):
    if policy=="q0": return 0, np.zeros(4)
    if policy=="hann": return e63.source_common.prior.choose("response",base,{},dict(tie_epsilon=epsilon))
    if policy not in TARGETS or model["target"]!=policy: raise ValueError("Mismatched selector target")
    values=(features-features[:1])/np.asarray(model["scale"])@np.asarray(model["weights"])
    if not np.isfinite(values).all(): raise ValueError("Nonfinite online scores")
    indices=np.flatnonzero(values>values[0]+epsilon)
    return (int(indices[np.argmax(values[indices])]) if len(indices) else 0), values


def key(unit): return "{}__{}__s{}__{}".format(unit["sequence"],unit["origin"],unit["seed"],unit["policy"])
def meta(plan, unit, kind, models):
    return dict(plan_fingerprint=plan["fingerprint"], unit=unit, kind=kind, training_fingerprint=models["fingerprint"])


def save(path, data, metadata):
    path=Path(path)
    if path.with_suffix(".receipt.json").exists(): raise ValueError("Never overwrite committed Stage-64 cache")
    if path.exists():
        with np.load(path,allow_pickle=False) as f:
            import json
            if json.loads(str(f["metadata_json"].item()))!=metadata: raise ValueError("Unidentified orphan")
    write_cache(path,data,metadata)
    publish_identical(path.with_suffix(".receipt.json"),seal(dict(schema=SCHEMA,metadata=metadata,sha256=file_hash(path))))


def read_cache(path, metadata):
    import json
    path=Path(path)
    if not path.with_suffix(".receipt.json").exists(): return None
    r=read_sealed(path.with_suffix(".receipt.json"))
    if r.get("metadata")!=metadata or not path.exists() or file_hash(path)!=r["sha256"]: raise ValueError("Stage-64 cache changed")
    with np.load(path,allow_pickle=False) as f:
        if json.loads(str(f["metadata_json"].item()))!=metadata: raise ValueError("Mixed cache")
        return {k:f[k] for k in f.files if k!="metadata_json"}
