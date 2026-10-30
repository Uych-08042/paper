"""Frozen Stage-54 source and a controlled first-transition preservation pilot."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from stage4_common import file_hash, fingerprint, write_cache
from stage27_uav_common import publish_identical
from stage46_shift_audit import verify_files
import stage54_rollout_common as prior
from stage55_matched_common import load_source

SCHEMA = "stage56_first_transition_preservation_v1"
ARMS = ("endpoint_control", "endpoint_half", "transition_preserve")
SEEDS = prior.SEEDS
# These three contrasts are fixed interventions, not an outcome-driven weight sweep.
WEIGHTS = {"endpoint_control": (1., 0.), "endpoint_half": (.5, 0.),
           "transition_preserve": (.5, .5)}  # endpoint, first transition
DESIGN = dict(prior.DESIGN, bootstrap_seed=2156,
    gradient="Detached crop/rounding/argmax and parents; direct geometry gradients at BOTH visited frames",
    first_transition_quality="Both native branch IoUs >=0.5 and native first center in usable first student crop",
    coefficients={a: list(w) for a, w in WEIGHTS.items()})
FILES = ("stage56_transition_common.py", "stage56_transition_runtime.py",
         "stage56_transition_experiment.py", "stage56_transition_summary.py", "test_stage56_transition.py")

def seal(body):
    return dict(body, fingerprint=fingerprint(body))

def read_sealed(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("fingerprint") != fingerprint({k: v for k, v in value.items() if k != "fingerprint"}):
        raise ValueError("Stage-56 schema/fingerprint changed: " + str(path))
    return value

def native_trace(inputs, entry):
    return inputs["arrays"][entry["sequence"]][1]["native"]

def source_trace(inputs, entry, arm, seed):
    return prior.source_trace(inputs["source53"], entry, arm, seed)

def fit_pool(plan, inputs, fold):
    return prior.fit_pool(plan, inputs["source53"], fold)

def initial_record(plan, fold, seed):
    return prior.initial_record(plan, fold, seed)

def endpoint_record(inputs, fold, seed):
    return next(r for r in inputs["records"] if (r["fold"], r["seed"], r["arm"]) == (fold, seed, "rollout_preserve"))

def make_plan(root):
    inputs = load_source(root)
    source = inputs["plan"]
    body = dict(schema=SCHEMA, role="fit_only_first_transition_preservation", source_input=str(inputs["root"]),
        source_plan_fingerprint=source["fingerprint"], source_decision_fingerprint=inputs["report"]["fingerprint"],
        source_file_sha256=inputs["bindings"], design=dict(DESIGN), training=dict(source["training"]),
        seeds=list(SEEDS), arms=list(ARMS), endpoint_controls=[endpoint_record(inputs, f["fold"], s) for f in source["folds"] for s in SEEDS],
        **{k: source[k] for k in ("initial_records", "entries", "folds", "root", "runtime", "runtime_identity", "search_size", "search_factor", "window", "anchor_pools")},
        code_sha256={n: file_hash(Path(__file__).with_name(n)) for n in FILES + prior.FILES + ("stage55_matched_common.py",)})
    for fold in body["folds"]:
        pool, _ = fit_pool(body, inputs, fold["fold"])
        if dict(fold=fold["fold"], pairs=len(pool), fingerprint=fingerprint(pool)) != body["anchor_pools"][fold["fold"]]:
            raise ValueError("Original fit anchor pool changed")
    return seal(body), inputs

def load_plan(root):
    plan = read_sealed(Path(root)/"manifest.json")
    expected, inputs = make_plan(plan["source_input"])
    if plan != expected:
        raise ValueError("Frozen Stage-56 source/code/design changed")
    return plan, inputs

def save_cache(path, data, metadata):
    write_cache(path, data, metadata)
    publish_identical(path.with_suffix(".receipt.json"), seal(dict(schema=SCHEMA, metadata=metadata, sha256=file_hash(path))))

def read_cache(path, metadata):
    if not path.with_suffix(".receipt.json").exists():
        return None
    receipt = read_sealed(path.with_suffix(".receipt.json"))
    if receipt.get("metadata") != metadata or not path.is_file() or receipt["sha256"] != file_hash(path):
        raise ValueError("Committed Stage-56 cache changed")
    with np.load(path, allow_pickle=False) as arrays:
        if json.loads(str(arrays["metadata_json"].item())) != metadata:
            raise ValueError("Mixed Stage-56 cache")
        return {k: arrays[k] for k in arrays.files if k != "metadata_json"}
