"""Frozen Stage-51 state pool and original-CENTER branch adaptation protocol."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np

from stage4_common import file_hash,fingerprint,write_cache
from stage27_uav_common import publish_identical,tracking_iou
from stage46_shift_audit import verify_files
from stage48_stability_experiment import audit_trace
from stage49_trajectory_audit import check_values
import stage51_state_common as prior
import stage51_state_experiment as previous
from stage51_state_summary import describe

SCHEMA="stage52_fixed_state_branch_adaptation_v1"
ARMS=("response_only","geometry_only")
SEEDS=(2132,2133)
DESIGN=dict(bootstrap_samples=5000,bootstrap_seed=2150,frozen_map_atol=1e-6,
    training="Original CENTER: ctr versus size+offset; ordinary loss; source state_mix views and draws",
    focus=["uav0000304_00253_s","uav0000307_04531_s","uav0000176_00000_s","uav0000226_05370_s","uav0000204_00000_s"])
FILES=("stage52_branch_common.py","stage52_branch_runtime.py","stage52_branch_experiment.py",
       "stage52_branch_summary.py","test_stage52_branch.py")


def seal(body): return dict(body,fingerprint=fingerprint(body))


def read_sealed(path):
    value=json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema")!=SCHEMA or value.get("fingerprint")!=fingerprint({k:v for k,v in value.items() if k!="fingerprint"}):
        raise ValueError("Stage-52 schema/fingerprint mismatch: "+str(path))
    return value


def source_trace(root,run,entry,arm,seed):
    method=arm+"_seed"+str(seed); path=Path(root)/"evaluation/traces"/(entry["sequence"]+"__"+method+".npz")
    value=prior.read_cache(path,dict(run_fingerprint=run["fingerprint"],sequence=entry["sequence"],method=method))
    if value is None: raise ValueError("Missing completed Stage-51 trajectory")
    audit_trace(value,entry)
    return value,path


def load_source(root):
    root=Path(root).expanduser().resolve(); source,source48=prior.load_plan(root)
    previous.smoke_gate(source,root); features=previous.feature_gate(source,root)
    records,models=previous.model_records(source,root)
    run=prior.read_sealed(root/"evaluation/run.json"); report=prior.read_sealed(root/"analysis/stage51_decision.json")
    required={"stage51_"+n+".csv" for n in ("policy_summary","video_summary","seed_video_summary","paired","video_influence")}
    if (report.get("status") not in ("PROMISING_FIT_STATE_SOURCE_PILOT","REVIEW_FIT_STATE_SOURCE_PILOT") or
        report.get("role")!="consumed_development_fixed_donor_state_source" or report.get("plan_fingerprint")!=source["fingerprint"] or
        run.get("plan_fingerprint")!=source["fingerprint"] or report.get("run_fingerprint")!=run["fingerprint"] or
        run.get("role")!="native_state_source_oof_evaluation" or run.get("model_sha256")!=models or report.get("model_sha256")!=models or
        run.get("collection_fingerprint")!=features["fingerprint"] or report.get("new_models")!=8 or report.get("new_oof_trajectories")!=48 or
        report.get("videos")!=12 or report.get("seeds")!=list(SEEDS) or
        not required<=set(report.get("table_sha256",{}))<=required|{"stage51_events.csv"}):
        raise ValueError("Need complete unmixed Stage-51 result")
    bindings=dict(source["source_file_sha256"]); bindings.update(features["file_sha256"]); bindings.update(models)
    bindings.update(report["source_file_sha256"])
    bindings.update({str(root/"analysis"/n):h for n,h in report["table_sha256"].items()})
    for p in (root/"manifest.json",root/"evaluation/run.json",root/"analysis/stage51_decision.json",
            root/"collection/stage51_collection.json",root/"collection/stage51_state_exposure.csv",root/"collection/stage51_state_support.csv"):
        bindings[str(p)]=file_hash(p)
    verify_files(bindings)
    with (root/"analysis/stage51_seed_video_summary.csv").open(encoding="utf-8",newline="") as h: rows=list(csv.DictReader(h))
    table={(r["sequence"],r["arm"],r["seed"]):r for r in rows}
    expected={(e["sequence"],a,str(s)) for e in source["entries"] for a in ("native","source_supervised")+prior.ARMS
              for s in ((0,) if a=="native" else SEEDS)}
    if len(table)!=len(rows) or set(table)!=expected: raise ValueError("Incomplete or duplicate Stage-51 video summaries")
    oldroot=Path(source["source_stage48"]); oldrun=prior.prior.read_sealed(oldroot/"evaluation/run.json")
    calls=0
    for entry in source["entries"]:
        native=prior.prior.source_native(source48,entry); refs={}
        for seed in SEEDS:
            method="supervised_seed"+str(seed); p=oldroot/"evaluation/traces"/(entry["sequence"]+"__"+method+".npz")
            refs[seed]=prior.prior.read_cache(p,dict(run_fingerprint=oldrun["fingerprint"],sequence=entry["sequence"],method=method))
            if refs[seed] is None: raise ValueError("Missing frozen reference")
            audit_trace(refs[seed],entry)
        gt=np.asarray(entry["ground_truth"],float)[1:]
        mask=np.any([tracking_iou(v["boxes"],gt)<.2 for v in refs.values()],axis=0)
        values=[("native",0,native)]+[("source_supervised",s,v) for s,v in refs.items()]
        for seed in SEEDS:
            for arm in prior.ARMS:
                data,_=source_trace(root,run,entry,arm,seed); calls+=int(data["head_calls"].sum()); values.append((arm,seed,data))
        for arm,seed,data in values:
            metrics,_=describe(data,entry,native["boxes"],mask)
            check_values(table[entry["sequence"],arm,str(seed)],metrics)
    if report.get("inference_head_calls")!=calls or report.get("inference_block_calls")!=calls*7:
        raise ValueError("Stage-51 complete trajectory budget changed")
    return dict(plan=source,source48=source48,run=run,report=report,records=[r for r,_ in records],bindings=bindings)


def make_plan(root):
    root=Path(root).expanduser().resolve(); inputs=load_source(root); source=inputs["plan"]
    keys=("entries","folds","root","runtime","runtime_identity","search_size","search_factor","window")
    training={k:source["design"][k] for k in ("updates","smoke_updates","batch_pairs","save_every","learning_rate","weight_decay","gradient_clip","smoke_frames")}
    return seal(dict(schema=SCHEMA,role="fixed_state_source_branch_training_mechanism",source_input=str(root),
        source_plan_fingerprint=source["fingerprint"],source_decision_fingerprint=inputs["report"]["fingerprint"],
        source_file_sha256=inputs["bindings"],joint_records=[r for r in inputs["records"] if r["arm"]=="state_mix"],
        design=dict(DESIGN),training=training,seeds=list(SEEDS),arms=list(ARMS),**{k:source[k] for k in keys},
        head_implementation_sha256=file_hash(Path(__file__).resolve().parents[1]/"lib/models/layers/head.py"),
        code_sha256={n:file_hash(Path(__file__).with_name(n)) for n in FILES+(
            "stage51_state_common.py","stage51_state_runtime.py","stage51_state_experiment.py","stage51_state_summary.py",
            "stage18_tracking_loss.py","stage48_stability_runtime.py")})),inputs


def load_plan(root):
    plan=read_sealed(Path(root)/"manifest.json"); expected,inputs=make_plan(plan["source_input"])
    if plan!=expected: raise ValueError("Frozen Stage-52 source/code/design changed")
    return plan,inputs


def joint_record(plan,fold,seed):
    values=[r for r in plan["joint_records"] if r["fold"]==fold and r["seed"]==seed]
    if len(values)!=1: raise ValueError("Need unique fixed joint-training reference")
    return values[0]


def fit_pool(plan,inputs,fold):
    return prior.fit_pool(Path(plan["source_input"]),inputs["plan"],inputs["source48"],fold)


def save_cache(path,data,metadata):
    write_cache(path,data,metadata)
    publish_identical(path.with_suffix(".receipt.json"),seal(dict(schema=SCHEMA,metadata=metadata,sha256=file_hash(path))))


def read_cache(path,metadata):
    if not path.with_suffix(".receipt.json").exists(): return None
    record=read_sealed(path.with_suffix(".receipt.json"))
    if record.get("metadata")!=metadata or not path.is_file() or record["sha256"]!=file_hash(path): raise ValueError("Committed Stage-52 cache changed")
    with np.load(path,allow_pickle=False) as a:
        if json.loads(str(a["metadata_json"].item()))!=metadata: raise ValueError("Mixed Stage-52 trace")
        return {k:a[k] for k in a.files if k!="metadata_json"}
