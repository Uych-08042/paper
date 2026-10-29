"""Two-frame current-policy fit branches and paired native endpoint targets."""
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
from stage51_state_summary import describe
import stage51_state_common as states
import stage53_geometry_common as prior
import stage53_geometry_experiment as previous

SCHEMA="stage54_short_rollout_endpoint_v1"
ARMS=("rollout_supervised","rollout_preserve")
SEEDS=prior.SEEDS
DESIGN=dict(horizon=2,batch_pairs=1,endpoint_weight=1.,reliable_iou=.5,epsilon=1e-6,
    decode_atol=.001,bootstrap_samples=5000,bootstrap_seed=2152,frozen_map_atol=1e-6,
    focus=list(prior.DESIGN["focus"]),
    gradient="Detached native crop/rounding/argmax and branch parents; endpoint gradient enters final-frame geometry only")
FILES=("stage54_rollout_common.py","stage54_rollout_runtime.py","stage54_rollout_experiment.py",
       "stage54_rollout_summary.py","test_stage54_rollout.py")

def seal(value): return dict(value,fingerprint=fingerprint(value))

def read_sealed(path):
    value=json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema")!=SCHEMA or value.get("fingerprint")!=fingerprint({k:v for k,v in value.items() if k!="fingerprint"}):
        raise ValueError("Stage-54 schema/fingerprint changed: "+str(path))
    return value

def native_trace(inputs,entry): return prior.native_trace(inputs["source52"],entry)

def source_trace(inputs,entry,arm,seed):
    if arm in ("geometry_only","joint"): return prior.source_trace(inputs["source52"],entry,arm,seed)
    method=arm+"_seed"+str(seed); path=inputs["root"]/"evaluation/traces"/(entry["sequence"]+"__"+method+".npz")
    data=prior.read_cache(path,dict(run_fingerprint=inputs["run"]["fingerprint"],sequence=entry["sequence"],method=method))
    if data is None: raise ValueError("Missing completed Stage-53 trace")
    audit_trace(data,entry); return data

def load_source(root):
    root=Path(root).expanduser().resolve(); plan,source52=prior.load_plan(root)
    previous.smoke_gate(plan,root); records,models=previous.model_records(plan,root)
    run=prior.read_sealed(root/"evaluation/run.json"); report=prior.read_sealed(root/"analysis/stage53_decision.json")
    tables={"stage53_"+n+".csv" for n in ("policy_summary","paired","scope_summary","retention_summary","video_summary","seed_video_summary","focus","events")}
    if (report.get("status") not in ("REVIEW_NATIVE_GEOMETRY_PRESERVATION_PILOT","PROMISING_NATIVE_GEOMETRY_PRESERVATION_PILOT") or
        report.get("role")!="consumed_development_native_geometry_preservation" or report.get("plan_fingerprint")!=plan["fingerprint"] or
        run.get("plan_fingerprint")!=plan["fingerprint"] or report.get("run_fingerprint")!=run["fingerprint"] or
        run.get("role")!="native_geometry_preservation_oof" or run.get("model_sha256")!=models or report.get("model_sha256")!=models or
        report.get("new_models")!=8 or report.get("new_full_trajectories")!=48 or report.get("videos")!=12 or
        report.get("seeds")!=list(SEEDS) or set(report.get("table_sha256",{}))!=tables):
        raise ValueError("Need complete unmixed Stage-53 source")
    bindings=dict(plan["source_file_sha256"]); bindings.update(models); bindings.update(report["source_file_sha256"])
    bindings.update({str(root/"analysis"/n):h for n,h in report["table_sha256"].items()})
    for p in (root/"manifest.json",root/"evaluation/run.json",root/"analysis/stage53_decision.json",root/"smoke/stage53_smoke.json"):
        bindings[str(p)]=file_hash(p)
    verify_files(bindings)
    result=dict(root=root,plan=plan,source52=source52,records=[r for r,_ in records],run=run,report=report,bindings=bindings)
    with (root/"analysis/stage53_seed_video_summary.csv").open(encoding="utf-8",newline="") as h: rows=list(csv.DictReader(h))
    table={(r["sequence"],r["arm"],r["seed"]):r for r in rows}
    expected={(e["sequence"],a,str(s)) for e in plan["entries"] for a in ("native","geometry_control","joint")+prior.ARMS for s in ((0,) if a=="native" else SEEDS)}
    if len(table)!=len(rows) or set(table)!=expected: raise ValueError("Incomplete Stage-53 seed summaries")
    calls=0
    for entry in plan["entries"]:
        native=native_trace(result,entry); control={s:source_trace(result,entry,"geometry_only",s) for s in SEEDS}
        mask=np.any([tracking_iou(v["boxes"],np.asarray(entry["ground_truth"],float)[1:])<.2 for v in control.values()],axis=0)
        values=[("native",0,native)]+[("geometry_control",s,v) for s,v in control.items()]+[("joint",s,source_trace(result,entry,"joint",s)) for s in SEEDS]
        for seed in SEEDS:
            for arm in prior.ARMS:
                data=source_trace(result,entry,arm,seed); calls+=int(data["head_calls"].sum()); values.append((arm,seed,data))
        for arm,seed,data in values:
            metrics,_=describe(data,entry,native["boxes"],mask); metrics["control_failure_auc"]=metrics.pop("source_supervised_failure_auc")
            check_values(table[entry["sequence"],arm,str(seed)],metrics)
    if report.get("inference_head_calls")!=calls or report.get("inference_block_calls")!=calls*7: raise ValueError("Stage-53 budget changed")
    return result

def fit_pool(plan,inputs,fold):
    # Reuse source parents, not source model outputs as new training targets.
    source51=inputs["source52"]["source51"]; p51=source51["plan"]
    root51=Path(inputs["source52"]["plan"]["source_input"])
    pool=[]; bindings={}
    for entry in plan["entries"]:
        if entry["sequence"] not in plan["folds"][fold]["fit"]: continue
        if states.fit_fold(p51,entry["sequence"])!=fold: raise ValueError("Held-out branch generation forbidden")
        tp=states.path_for(root51,entry,"donor_traces"); fp=states.path_for(root51,entry,"features")
        trace=states.read_cache(tp,states.meta(p51,entry,"fit_only_donor_trajectory"))
        own=states.read_cache(fp,states.meta(p51,entry,"fit_only_original_features"))
        if trace is None or own is None: raise ValueError("Need frozen fit-only parents")
        audit_trace(trace,entry); states.audit_features(own,p51,entry,trace)
        base=states.source_pack(Path(p51["source_stage48"]),source51["source48"],entry)
        native=native_trace(inputs,entry)
        for p in (tp,fp,states.prior.cache_path(Path(p51["source_stage48"]),entry)):
            for q in (p,p.with_suffix(".receipt.json")): bindings[str(q)]=file_hash(q)
        for i,frame in enumerate(own["frame_id"]):
            frame=int(frame)
            if frame+plan["design"]["horizon"]>len(entry["frames"]): continue
            pool.append(dict(sequence=entry["sequence"],frame=frame,
                parents=np.stack([base["parents"][i,0],own["parents"][i]]).tolist(),
                native_expected=native["boxes"][frame-1:frame-1+plan["design"]["horizon"]].tolist()))
    if not pool: raise ValueError("No fit-only short branches")
    return pool,bindings

def make_plan(root):
    inputs=load_source(root); source=inputs["plan"]
    training=dict(source["training"],batch_pairs=DESIGN["batch_pairs"])
    body=dict(schema=SCHEMA,role="fit_only_current_policy_short_rollout",source_input=str(inputs["root"]),
        source_plan_fingerprint=source["fingerprint"],source_decision_fingerprint=inputs["report"]["fingerprint"],
        source_file_sha256=inputs["bindings"],design=dict(DESIGN),training=training,seeds=list(SEEDS),arms=list(ARMS),
        initial_records=source["controls"],**{k:source[k] for k in ("entries","folds","root","runtime","runtime_identity","search_size","search_factor","window")},
        code_sha256={n:file_hash(Path(__file__).with_name(n)) for n in FILES+("stage53_geometry_common.py","stage53_geometry_experiment.py",
            "stage53_geometry_runtime.py","stage53_geometry_summary.py","stage48_stability_runtime.py","stage18_tracking_loss.py")})
    pools=[fit_pool(body,inputs,f["fold"])[0] for f in body["folds"]]
    body["anchor_pools"]=[dict(fold=f["fold"],pairs=len(pool),fingerprint=fingerprint(pool)) for f,pool in zip(body["folds"],pools)]
    return seal(body),inputs

def load_plan(root):
    plan=read_sealed(Path(root)/"manifest.json"); expected,inputs=make_plan(plan["source_input"])
    if plan!=expected: raise ValueError("Frozen Stage-54 source/code/design changed")
    return plan,inputs

def initial_record(plan,fold,seed): return next(r for r in plan["initial_records"] if r["fold"]==fold and r["seed"]==seed)

def save_cache(path,data,metadata):
    write_cache(path,data,metadata); publish_identical(path.with_suffix(".receipt.json"),seal(dict(schema=SCHEMA,metadata=metadata,sha256=file_hash(path))))

def read_cache(path,metadata):
    if not path.with_suffix(".receipt.json").exists(): return None
    r=read_sealed(path.with_suffix(".receipt.json"))
    if r.get("metadata")!=metadata or not path.is_file() or r["sha256"]!=file_hash(path): raise ValueError("Committed Stage-54 cache changed")
    with np.load(path,allow_pickle=False) as a:
        if json.loads(str(a["metadata_json"].item()))!=metadata: raise ValueError("Mixed Stage-54 cache")
        return {k:a[k] for k in a.files if k!="metadata_json"}
