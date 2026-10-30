"""Completed Stage-56 source and fit-only reference-quality selection."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from stage4_common import file_hash, fingerprint, write_cache
from stage27_uav_common import publish_identical
from stage46_shift_audit import verify_files
import stage56_transition_common as prior
from stage49_trajectory_audit import table_rows, check_values
from stage51_state_summary import describe
from stage27_uav_common import tracking_iou
import stage56_transition_experiment as previous
from stage56_transition_summary import METRICS

SCHEMA = "stage57_quality_selective_endpoint_v1"
ARMS = ("endpoint_control", "count_matched", "quality_select")
SEEDS = prior.SEEDS
DESIGN = dict(prior.DESIGN, bootstrap_seed=2157,
    gradient="Detached fit GT quality gate; endpoint geometry gradient only; no first-step penalty or BPTT",
    quality_rule="Reference endpoint IoU > detached current endpoint IoU + 1e-6, inside the original qualified cohort",
    matched_rule="Uniform subset of qualified endpoints with the same k as quality selection on OWN current state",
    gate_seed_offset=571000, coefficients={a: [1., 0.] for a in ARMS})

FILES = ("stage57_quality_common.py", "stage57_quality_runtime.py",
         "stage57_quality_experiment.py", "stage57_quality_summary.py", "test_stage57_quality.py")

def selection_masks(ends, priorities, arm):
    """Detached fit-label masks. Random control matches k on its OWN states."""
    if arm not in ARMS:
        raise ValueError("Unknown quality-selection arm")
    eligible = np.asarray([e["qualified"] for e in ends], bool)
    reference = np.asarray([e["reference_iou"] for e in ends], float)
    student = np.asarray([e["student_iou"] for e in ends], float)
    priorities = np.asarray(priorities, float)
    if (priorities.shape != eligible.shape or not np.isfinite(priorities).all() or
        not np.isfinite(reference).all() or not np.isfinite(student).all() or
        ((reference < 0) | (reference > 1) | (student < 0) | (student > 1)).any()):
        raise ValueError("Invalid detached quality labels/priorities")
    gap = reference-student
    better = eligible & (gap > DESIGN["epsilon"])
    random = np.zeros_like(eligible)
    ids = np.flatnonzero(eligible)
    chosen = ids[np.argsort(priorities[ids], kind="stable")[:int(better.sum())]]
    random[chosen] = True
    selected = {"endpoint_control": eligible, "quality_select": better, "count_matched": random}[arm]
    audit = dict(eligible_endpoints=int(eligible.sum()), quality_better=int(better.sum()),
        quality_tied=int((eligible & (np.abs(gap) <= DESIGN["epsilon"])).sum()),
        quality_worse=int((eligible & (gap < -DESIGN["epsilon"])).sum()),
        random_kept=int(random.sum()), actual_kept=int(selected.sum()),
        kept_not_better=int((selected & ~better).sum()),
        withheld_better=int((better & ~selected).sum()),
        mask_disagreements=int((better != random).sum()),
        mixed_quality_batches=int(0 < better.sum() < eligible.sum()),
        eligible_iou_gap_sum=float(gap[eligible].sum()), kept_iou_gap_sum=float(gap[selected].sum()))
    if audit["random_kept"] != audit["quality_better"]:
        raise ValueError("Count-matched selection changed mask mass")
    return selected, audit

def seal(body):
    return dict(body, fingerprint=fingerprint(body))

def read_sealed(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("fingerprint") != fingerprint({k: v for k, v in value.items() if k != "fingerprint"}):
        raise ValueError("Stage-57 schema/fingerprint changed: " + str(path))
    return value

def native_trace(inputs, entry):
    return inputs["arrays"][entry["sequence"]][1]["native"]

def source_trace(inputs, entry, arm, seed):
    return prior.source_trace(inputs["source54"], entry, arm, seed)

def fit_pool(plan, inputs, fold):
    return prior.fit_pool(plan, inputs["source54"], fold)

def initial_record(plan, fold, seed):
    return prior.initial_record(plan, fold, seed)

def endpoint_record(inputs, fold, seed):
    return next(r for r in inputs["records"] if (r["fold"], r["seed"], r["arm"]) == (fold, seed, "endpoint_control"))

def load_source(root):
    root=Path(root).expanduser().resolve(); plan,source54=prior.load_plan(root)
    previous.smoke_gate(plan,root); records,models=previous.model_records(plan,root)
    run=prior.read_sealed(root/"evaluation/run.json"); report=prior.read_sealed(root/"analysis/stage56_decision.json")
    tables={"stage56_"+n+".csv" for n in ("policy_summary","paired","scope_summary","retention_summary","video_summary","seed_video_summary","focus","events")}
    if (report.get("status") not in ("REVIEW_FIRST_TRANSITION_PRESERVATION_PILOT","PROMISING_FIRST_TRANSITION_PRESERVATION_PILOT") or
        report.get("role")!="consumed_development_first_transition_preservation" or report.get("plan_fingerprint")!=plan["fingerprint"] or
        run.get("plan_fingerprint")!=plan["fingerprint"] or report.get("run_fingerprint")!=run["fingerprint"] or
        run.get("role")!="native_short_rollout_oof" or run.get("model_sha256")!=models or report.get("model_sha256")!=models or
        report.get("videos")!=12 or report.get("new_models")!=12 or report.get("new_full_trajectories")!=72 or
        report.get("seeds")!=list(prior.SEEDS) or set(report.get("table_sha256",{}))!=tables):
        raise ValueError("Need complete unmixed Stage-56 source")
    bindings=dict(plan["source_file_sha256"]); bindings.update(models); bindings.update(report["source_file_sha256"])
    bindings.update({str(root/"analysis"/n):h for n,h in report["table_sha256"].items()})
    for name in ("manifest.json","evaluation/run.json","analysis/stage56_decision.json","smoke/stage56_smoke.json"):
        bindings[str(root/name)]=file_hash(root/name)
    verify_files(bindings)
    seed_rows=table_rows(root/"analysis/stage56_seed_video_summary.csv",("sequence","arm","seed"))
    video_rows=table_rows(root/"analysis/stage56_video_summary.csv",("sequence","arm"))
    policy_rows=table_rows(root/"analysis/stage56_policy_summary.csv",("arm",))
    if (len(seed_rows),len(video_rows),len(policy_rows))!=(132,72,6): raise ValueError("Incomplete Stage-56 summaries")
    arrays={}; recomputed=[]; calls=0
    for entry in plan["entries"]:
        native=prior.native_trace(source54,entry)
        control={s:prior.source_trace(source54,entry,"geometry_only",s) for s in prior.SEEDS}
        mask=np.any([tracking_iou(v["boxes"],np.asarray(entry["ground_truth"],float)[1:])<.2 for v in control.values()],axis=0)
        values=[("native",0,native)]+[("geometry_control",s,v) for s,v in control.items()]
        values.extend(("static_anchor",s,prior.source_trace(source54,entry,"anchor_reliable",s)) for s in prior.SEEDS)
        own={"native":native}
        for seed in prior.SEEDS:
            for arm in prior.ARMS:
                method=arm+"_seed"+str(seed); path=root/"evaluation/traces"/(entry["sequence"]+"__"+method+".npz")
                data=prior.read_cache(path,dict(run_fingerprint=run["fingerprint"],sequence=entry["sequence"],method=method))
                if data is None: raise ValueError("Missing completed Stage-56 trace")
                from stage48_stability_experiment import audit_trace
                audit_trace(data,entry); calls+=int(data["head_calls"].sum()); own[method]=data; values.append((arm,seed,data))
        arrays[entry["sequence"]]=(entry,own)
        for arm,seed,data in values:
            m,_=describe(data,entry,native["boxes"],mask); m["control_failure_auc"]=m.pop("source_supervised_failure_auc")
            check_values(seed_rows[entry["sequence"],arm,str(seed)],m)
            recomputed.append(dict(sequence=entry["sequence"],arm=arm,seed=seed,**m))
    for (name,arm),row in video_rows.items():
        chosen=[r for r in recomputed if (r["sequence"],r["arm"])==(name,arm)]
        check_values(row,{k:float(np.mean([r[k] for r in chosen])) if all(r[k] is not None for r in chosen) else None for k in METRICS})
        check_values(row,dict(introduced_failure=any(r["introduced_failure"] for r in chosen)))
    for (arm,),row in policy_rows.items():
        selected=[r for (n,a),r in video_rows.items() if a==arm]
        expected={k:float(np.mean([float(r[k]) for r in selected if r[k]!=""])) if any(r[k]!="" for r in selected) else None for k in METRICS}
        check_values(row,dict(videos=12,introduced_failure_videos=sum(r["introduced_failure"]=="True" for r in selected),**expected))
    if (report.get("inference_head_calls"),report.get("inference_block_calls"))!=(calls,calls*7): raise ValueError("Stage-56 source call budget changed")
    return dict(root=root,plan=plan,report=report,run=run,source54=source54,records=[r for r,_ in records],arrays=arrays,bindings=bindings)


def make_plan(root):
    inputs = load_source(root)
    source = inputs["plan"]
    body = dict(schema=SCHEMA, role="fit_only_quality_selective_endpoint", source_input=str(inputs["root"]),
        source_plan_fingerprint=source["fingerprint"], source_decision_fingerprint=inputs["report"]["fingerprint"],
        source_file_sha256=inputs["bindings"], design=dict(DESIGN), training=dict(source["training"]),
        seeds=list(SEEDS), arms=list(ARMS), endpoint_controls=[endpoint_record(inputs, f["fold"], s) for f in source["folds"] for s in SEEDS],
        **{k: source[k] for k in ("initial_records", "entries", "folds", "root", "runtime", "runtime_identity", "search_size", "search_factor", "window", "anchor_pools")},
        code_sha256={n: file_hash(Path(__file__).with_name(n)) for n in FILES + prior.FILES})
    for fold in body["folds"]:
        pool, _ = fit_pool(body, inputs, fold["fold"])
        if dict(fold=fold["fold"], pairs=len(pool), fingerprint=fingerprint(pool)) != body["anchor_pools"][fold["fold"]]:
            raise ValueError("Original fit anchor pool changed")
    return seal(body), inputs

def load_plan(root):
    plan = read_sealed(Path(root)/"manifest.json")
    expected, inputs = make_plan(plan["source_input"])
    if plan != expected:
        raise ValueError("Frozen Stage-57 source/code/design changed")
    return plan, inputs

def save_cache(path, data, metadata):
    write_cache(path, data, metadata)
    publish_identical(path.with_suffix(".receipt.json"), seal(dict(schema=SCHEMA, metadata=metadata, sha256=file_hash(path))))

def read_cache(path, metadata):
    if not path.with_suffix(".receipt.json").exists():
        return None
    receipt = read_sealed(path.with_suffix(".receipt.json"))
    if receipt.get("metadata") != metadata or not path.is_file() or receipt["sha256"] != file_hash(path):
        raise ValueError("Committed Stage-57 cache changed")
    with np.load(path, allow_pickle=False) as arrays:
        if json.loads(str(arrays["metadata_json"].item())) != metadata:
            raise ValueError("Mixed Stage-57 cache")
        return {k: arrays[k] for k in arrays.files if k != "metadata_json"}
