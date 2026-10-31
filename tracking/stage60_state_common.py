"""Fit-only action values for causal SGLA search-parent proposals."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from stage4_common import file_hash, fingerprint, write_cache
from stage27_uav_common import publish_identical, crop_geometry
from stage46_shift_audit import verify_files
import stage59_parent_common as prior

SCHEMA = "stage60_causal_search_state_value_v1"
PROPOSALS = ("current", "history_size", "lag_center", "lag_center_history_size")
POLICIES = ("current", "response", "immediate_value", "future_value")
TARGETS = ("immediate_value", "future_value")
DESIGN = dict(history=5, horizon=8, anchors_per_video=24, ridge=.01, tie_epsilon=1e-6,
              feature_clip=10., box_atol=.001, source_box_atol=0., smoke_anchors=2, smoke_frames=32,
              bootstrap_seed=2160, bootstrap_samples=20000)
FEATURES = ("raw_peak", "hann_peak", "hann_margin", "raw_margin", "entropy", "psr",
            "raw_hann_distance", "hann_radius", "output_dx", "output_dy", "output_log_w", "output_log_h",
            "parent_dx", "parent_dy", "parent_log_w", "parent_log_h", "relative_image_w", "relative_image_h",
            "padding", "speed_innovation_interaction", "spread_scale_interaction", "center_scale_interaction",
            "innovation_scale_interaction", "peak_size_interaction")
FILES = ("stage60_state_common.py", "stage60_state_runtime.py", "stage60_state_experiment.py",
         "stage60_state_summary.py", "test_stage60_state.py")

def seal(body): return dict(body, fingerprint=fingerprint(body))

def read_sealed(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("fingerprint") != fingerprint({k:v for k,v in value.items() if k!="fingerprint"}):
        raise ValueError("Stage-60 sealed content changed: " + str(path))
    return value

def load_source(root):
    root = Path(root).expanduser().resolve()
    p59, source = prior.load_plan(root)
    report = prior.read_sealed(root/"analysis/stage59_diagnostic.json")
    run = prior.read_sealed(root/"run.json")
    gate = prior.read_sealed(root/"smoke/stage59_smoke.json")
    tables = {"stage59_"+n+".csv" for n in ("point_details","crossed_matrix","rollout_summary","branch_contrasts","focus","factor_summary","timelines","exposure")}
    if (report.get("status") != "COMPLETE_PARENT_COMPONENT_CONTINUATION_DIAGNOSTIC" or
        report.get("role") != "posthoc_consumed_case_predicted_parent_factorial" or
        report.get("plan_fingerprint") != p59["fingerprint"] or run.get("plan_fingerprint") != p59["fingerprint"] or
        report.get("run_fingerprint") != run["fingerprint"] or gate.get("run_fingerprint") != run["fingerprint"] or
        gate.get("status") != "PASS_PARENT_FACTORIAL_SMOKE" or set(report.get("table_sha256",{})) != tables or
        report.get("shared_inputs") != p59["expected_points"] or report.get("branches") != p59["expected_branches"] or
        report.get("full_network_calls") != p59["expected_full_forwards"] or report.get("head_only_calls") != p59["expected_head_only_forwards"] or
        report.get("block_calls") != p59["expected_full_forwards"]*7 or report.get("selector_calls") != p59["expected_full_forwards"] or
        report.get("total_head_calls") != p59["expected_full_forwards"]+p59["expected_head_only_forwards"] or
        report.get("cases") != len(p59["cases"]) or report.get("branch_horizon") != p59["design"]["horizon"] or
        len(report.get("cache_sha256",{})) != 2*(p59["expected_points"]+p59["expected_branches"])):
        raise ValueError("Need complete unmixed Stage-59 diagnostic")
    for k, bound in (("maximum_diagonal_error",.001),("maximum_branch_diagonal_error",.001),("maximum_decode_error",.001),
                     ("maximum_first_frame_error",.001),("maximum_first_map_error",1e-6),
                     ("maximum_fixed_response_error",1e-6),("maximum_branch_decode_error",.001),
                     ("maximum_accounting_residual",1e-10)):
        if not 0 <= float(report.get(k,float("inf"))) <= bound: raise ValueError("Failed Stage-59 audit: "+k)
    bindings = dict(source["bindings"])
    bindings.update(report["cache_sha256"])
    bindings.update({str(root/"analysis"/n):h for n,h in report["table_sha256"].items()})
    for name in ("manifest.json","run.json","analysis/stage59_diagnostic.json","smoke/stage59_smoke.json"):
        bindings[str(root/name)] = file_hash(root/name)
    verify_files(bindings)
    return dict(source, stage59=root, stage59_report=report, bindings=bindings)

def anchors(entry, design):
    last = len(entry["frames"])-design["horizon"]
    if last < 1: raise ValueError("Video too short for fixed fit horizon")
    return np.unique(np.linspace(1,last,min(design["anchors_per_video"],last),dtype=int)).tolist()

def validate_folds(source):
    names = {e["sequence"] for e in source["entries"]}
    if len(names) != len(source["entries"]): raise ValueError("Duplicate source sequences")
    if [f["fold"] for f in source["folds"]] != list(range(len(source["folds"]))): raise ValueError("Source fold order changed")
    held = []
    for fold in source["folds"]:
        fit, test = set(fold["fit"]), set(fold["held_out"])
        if not fit or not test or fit & test or fit | test != names: raise ValueError("Fit/held-out leakage")
        held.extend(test)
    if sorted(held) != sorted(names): raise ValueError("Need one OOF evaluator per source group")
    groups = {e["sequence"]:e["source_group_proxy"] for e in source["entries"]}
    for fold in source["folds"]:
        if {groups[n] for n in fold["fit"]} & {groups[n] for n in fold["held_out"]}:
            raise ValueError("Source-group leakage")

def make_plan(root):
    inputs = load_source(root); source = inputs["plan"]; validate_folds(source)
    design = dict(DESIGN); seeds = list(prior.prior.SEEDS)
    entries = {e["sequence"]:e for e in source["entries"]}
    units = [dict(fold=f["fold"], seed=s, sequence=n) for f in source["folds"] for s in seeds for n in f["fit"]]
    fit_frames = sum(len(entries[u["sequence"]]["frames"])-1 for u in units)
    count = sum(len(anchors(entries[u["sequence"]],design)) for u in units)
    test_frames = len(seeds)*sum(len(e["frames"])-1 for e in entries.values())
    body = dict(schema=SCHEMA, role="consumed_development_causal_search_state_value", source_input=str(inputs["stage59"]),
        source_diagnostic_fingerprint=inputs["stage59_report"]["fingerprint"], source_file_sha256=inputs["bindings"],
        design=design, seeds=seeds, policies=list(POLICIES), proposals=list(PROPOSALS), features=list(FEATURES),
        base_arm="endpoint_control", fit_units=units, expected_fit_frames=fit_frames, expected_fit_anchors=count,
        expected_branch_calls=count*4*design["horizon"], expected_evaluation_calls=test_frames*13,
        expected_evaluation_trajectories=len(entries)*len(seeds)*len(POLICIES),
        **{k:source[k] for k in ("entries","folds","root","runtime","runtime_identity","search_size","search_factor","window")},
        code_sha256={n:file_hash(Path(__file__).with_name(n)) for n in FILES+("stage47_search_collect.py","stage54_rollout_runtime.py","stage48_stability_runtime.py")})
    return seal(body), inputs

def load_plan(root):
    plan = read_sealed(Path(root)/"manifest.json"); expected, inputs = make_plan(plan["source_input"])
    if expected != plan: raise ValueError("Frozen Stage-60 source/code/design changed")
    return plan, inputs

def save(path, data, metadata):
    path = Path(path)
    if path.with_suffix(".receipt.json").exists(): raise ValueError("Never overwrite committed Stage-60 cache")
    if path.exists():
        with np.load(path,allow_pickle=False) as orphan:
            if json.loads(str(orphan["metadata_json"].item()))!=metadata: raise ValueError("Unidentified orphan cache")
    write_cache(path,data,metadata)
    publish_identical(path.with_suffix(".receipt.json"),seal(dict(schema=SCHEMA,metadata=metadata,sha256=file_hash(path))))

def read_cache(path, metadata):
    path = Path(path); receipt = path.with_suffix(".receipt.json")
    if not receipt.exists(): return None
    value = read_sealed(receipt)
    if value.get("metadata") != metadata or not path.is_file() or file_hash(path) != value["sha256"]:
        raise ValueError("Committed Stage-60 cache changed")
    with np.load(path,allow_pickle=False) as f:
        if json.loads(str(f["metadata_json"].item())) != metadata: raise ValueError("Mixed Stage-60 cache")
        return {k:f[k] for k in f.files if k!="metadata_json"}

def key(unit): return "f{}__s{}__{}".format(unit["fold"],unit["seed"],unit["sequence"])

def meta(plan, kind, **kwargs): return dict(plan_fingerprint=plan["fingerprint"],kind=kind,**kwargs)

def history_parents(history, design):
    h = np.asarray(history,dtype=float)[-design["history"]:]
    if h.ndim!=2 or h.shape[1]!=4 or not len(h) or not np.isfinite(h).all() or np.any(h[:,2:]<=0):
        raise ValueError("Need valid own predicted history only")
    latest = h[-1]; center = latest[:2]+latest[2:]/2
    older = h[-2] if len(h)>1 else latest
    lag = older[:2]+older[2:]/2
    size = np.exp(np.median(np.log(h[:,2:]),axis=0))
    if len(h)==1: size=latest[2:].copy()
    # The original candidate is copied byte-for-byte. No predicted output is smoothed.
    return np.array([latest.copy(),np.r_[center-size/2,size],np.r_[lag-latest[2:]/2,latest[2:]],np.r_[lag-size/2,size]])

def fit_value(features, returns, design):
    features, returns = np.asarray(features,float), np.asarray(returns,float)
    if features.ndim!=3 or not len(features) or features.shape[1:]!=(4,len(FEATURES)) or returns.shape!=features.shape[:2] or not np.isfinite(features).all() or not np.isfinite(returns).all() or np.any((returns<0)|(returns>1)):
        raise ValueError("Invalid fit-only action examples")
    x = (features-features[:,:1])[:,1:].reshape(-1,len(FEATURES))
    y = (returns-returns[:,:1])[:,1:].reshape(-1)
    scale = np.maximum(np.sqrt(np.mean(x*x,axis=0)),1e-6)
    z=x/scale
    weights = np.linalg.solve(z.T@z/len(z)+design["ridge"]*np.eye(z.shape[1]),z.T@y/len(z))
    if not np.isfinite(weights).all(): raise ValueError("Invalid ridge model")
    return dict(scale=scale.tolist(),weights=weights.tolist(),training_states=len(features),training_examples=len(y),
                nonzero_targets=int((np.abs(y)>1e-6).sum()),target_mean=float(y.mean()),target_rms=float(np.sqrt(np.mean(y*y))),
                fit_rmse=float(np.sqrt(np.mean((z@weights-y)**2))))

def scores(features, model):
    f = np.asarray(features,float); weights=np.asarray(model["weights"],float); scale=np.asarray(model["scale"],float)
    if f.shape!=(4,len(FEATURES)) or weights.shape!=(len(FEATURES),) or scale.shape!=weights.shape or np.any(scale<=0):
        raise ValueError("Scorer shape changed")
    result=(f-f[:1])/scale@weights
    if not np.isfinite(result).all(): raise ValueError("Nonfinite scorer output")
    return result

def choose(policy, features, models, design):
    if policy=="current": return 0,np.zeros(4)
    if policy=="response": values=np.asarray(features,float)[:,FEATURES.index("hann_peak")]
    elif policy in TARGETS: values=scores(features,models[policy])
    else: raise ValueError("Unknown online policy")
    # Current first on ties; other ties follow the fixed proposal order.
    candidates=np.flatnonzero(values>values[0]+design["tie_epsilon"])
    return (int(candidates[np.argmax(values[candidates])]) if len(candidates) else 0), values

def audit_calls(data, leading):
    shapes=dict(block_calls=leading+(12,),layer=leading,head_calls=leading,selector_calls=leading)
    for name,shape in shapes.items():
        if np.asarray(data[name]).shape!=shape: raise ValueError("Call accounting shape changed: "+name)
    for b,l,h,s in zip(np.asarray(data["block_calls"]).reshape(-1,12),np.asarray(data["layer"]).ravel(),np.asarray(data["head_calls"]).ravel(),np.asarray(data["selector_calls"]).ravel()):
        prior.native_calls(b,l,h,s)

def audit_trace(data, count, proposals):
    if (data["boxes"].shape!=(count,4) or data["parents"].shape!=(count,4) or
        data["candidate_boxes"].shape!=(count,proposals,4) or data["features"].shape!=(count,proposals,len(FEATURES)) or
        not np.array_equal(data["frame_id"],np.arange(1,count+1))): raise ValueError("Incomplete own trajectory")
    audit_calls(data,(count,proposals))
    if not all(np.isfinite(data[k]).all() for k in ("boxes","parents","features","candidate_boxes","candidate_parents")):
        raise ValueError("Nonfinite trajectory")
    if np.any(data["boxes"][:,2:]<=0): raise ValueError("Invalid output boxes")
    selected=np.asarray(data["selected"],int)
    if np.any((selected<0)|(selected>=proposals)): raise ValueError("Invalid selection")
    np.testing.assert_array_equal(data["boxes"],data["candidate_boxes"][np.arange(count),selected])
    np.testing.assert_array_equal(data["parents"],data["candidate_parents"][np.arange(count),selected])
    if count>1: np.testing.assert_array_equal(data["history_latest"][1:],data["boxes"][:-1])
