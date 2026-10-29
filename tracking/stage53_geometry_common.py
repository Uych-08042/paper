"""Fixed-state, fit-qualified native geometry preservation protocol."""
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
import stage52_branch_common as prior
import stage52_branch_experiment as previous

SCHEMA="stage53_native_geometry_preservation_v1"
ARMS=("anchor_all","anchor_reliable")
SEEDS=prior.SEEDS
DESIGN=dict(reliable_iou=.5,anchor_weight=1.,epsilon=1e-6,bootstrap_samples=5000,bootstrap_seed=2151,
    frozen_map_atol=1e-6,focus=list(prior.DESIGN["focus"]),
    anchor="Frozen native Hann cell: mean abs log-size ratio plus mean abs center change / native size",
    normalization="Reliable indicator / full fit-pool reliable fraction; all-state weight=1")
FILES=("stage53_geometry_common.py","stage53_geometry_runtime.py","stage53_geometry_experiment.py",
    "stage53_geometry_summary.py","test_stage53_geometry.py")

def seal(value): return dict(value,fingerprint=fingerprint(value))

def read_sealed(path):
    value=json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema")!=SCHEMA or value.get("fingerprint")!=fingerprint({k:v for k,v in value.items() if k!="fingerprint"}):
        raise ValueError("Stage-53 schema/fingerprint changed: "+str(path))
    return value

def native_trace(inputs,entry): return prior.prior.prior.source_native(inputs["source51"]["source48"],entry)

def source_trace(inputs,entry,arm,seed):
    if arm=="joint":
        return prior.source_trace(Path(inputs["plan"]["source_input"]),inputs["source51"]["run"],entry,"state_mix",seed)[0]
    method=arm+"_seed"+str(seed)
    path=inputs["root"] / "evaluation/traces"/(entry["sequence"]+"__"+method+".npz")
    value=prior.read_cache(path,dict(run_fingerprint=inputs["run"]["fingerprint"],sequence=entry["sequence"],method=method))
    if value is None: raise ValueError("Missing completed Stage-52 trajectory")
    audit_trace(value,entry); return value

def load_source(root):
    root=Path(root).expanduser().resolve(); plan,source51=prior.load_plan(root)
    previous.smoke_gate(plan,root); records,models=previous.model_records(plan,root)
    run=prior.read_sealed(root/"evaluation/run.json"); report=prior.read_sealed(root/"analysis/stage52_diagnostic.json")
    tables={"stage52_"+x+".csv" for x in ("policy_summary","paired","factor_summary","scope_summary","video_summary","seed_video_summary","focus","events")}
    if (report.get("status")!="COMPLETE_FIXED_STATE_BRANCH_DIAGNOSTIC" or report.get("role")!="consumed_development_branch_update_factorial" or
        report.get("plan_fingerprint")!=plan["fingerprint"] or run.get("plan_fingerprint")!=plan["fingerprint"] or
        run.get("role")!="native_branch_adaptation_oof" or report.get("run_fingerprint")!=run["fingerprint"] or
        run.get("model_sha256")!=models or report.get("model_sha256")!=models or report.get("new_models")!=8 or
        report.get("new_full_trajectories")!=48 or report.get("videos")!=12 or report.get("seeds")!=list(SEEDS) or set(report.get("table_sha256",{}))!=tables):
        raise ValueError("Need complete unmixed Stage-52 source")
    bindings=dict(plan["source_file_sha256"]); bindings.update(models); bindings.update(report["source_file_sha256"])
    bindings.update({str(root/"analysis"/n):h for n,h in report["table_sha256"].items()})
    for p in (root/"manifest.json",root/"evaluation/run.json",root/"analysis/stage52_diagnostic.json",root/"smoke/stage52_smoke.json"):
        bindings[str(p)]=file_hash(p)
    verify_files(bindings)
    result=dict(root=root,plan=plan,source51=source51,run=run,report=report,records=[r for r,_ in records],bindings=bindings)
    with (root/"analysis/stage52_seed_video_summary.csv").open(encoding="utf-8",newline="") as h: rows=list(csv.DictReader(h))
    table={(r["sequence"],r["arm"],r["seed"]):r for r in rows}
    expected={(e["sequence"],a,str(s)) for e in plan["entries"] for a in ("native",)+prior.ARMS+("joint",) for s in ((0,) if a=="native" else SEEDS)}
    if len(table)!=len(rows) or set(table)!=expected: raise ValueError("Incomplete Stage-52 seed summaries")
    calls=0
    for entry in plan["entries"]:
        native=native_trace(result,entry); joints={s:source_trace(result,entry,"joint",s) for s in SEEDS}
        mask=np.any([tracking_iou(v["boxes"],np.asarray(entry["ground_truth"],float)[1:])<.2 for v in joints.values()],axis=0)
        values=[("native",0,native)]+[("joint",s,v) for s,v in joints.items()]
        for seed in SEEDS:
            for arm in prior.ARMS:
                data=source_trace(result,entry,arm,seed); calls+=int(data["head_calls"].sum()); values.append((arm,seed,data))
        for arm,seed,data in values:
            metrics,_=describe(data,entry,native["boxes"],mask); metrics["joint_failure_auc"]=metrics.pop("source_supervised_failure_auc")
            check_values(table[entry["sequence"],arm,str(seed)],metrics)
    if report.get("inference_head_calls")!=calls or report.get("inference_block_calls")!=calls*7: raise ValueError("Stage-52 inference budget changed")
    return result

def fit_pool(plan,inputs,fold): return prior.fit_pool(inputs["plan"],inputs["source51"],fold)

def teacher_state(sample,view,window,design):
    raw=np.asarray(sample["raw"][view]); h,w=raw.shape[-2:]
    weights=np.asarray(window,dtype=raw.dtype).reshape(-1)
    if h!=w or weights.size!=h*w or not np.isfinite(weights).all(): raise ValueError("Need native square Hann window")
    size=np.asarray(sample["size"][view],float).reshape(2,-1); off=np.asarray(sample["offset"][view],float).reshape(2,-1)
    cell=int(np.argmax(raw.reshape(-1)*weights)); wh=size[:,cell]; center=(np.array([cell%w,cell//w])+off[:,cell])/w
    target=np.asarray(sample["target"][view],float)
    if not np.isfinite(np.r_[wh,center,target]).all() or (wh<=0).any() or (target[2:]<=0).any(): raise ValueError("Invalid native geometry/fit label")
    xy=center-wh/2; intersection=np.maximum(0,np.minimum(xy+wh,target[:2]+target[2:])-np.maximum(xy,target[:2])).prod()
    iou=float(intersection/(wh.prod()+target[2:].prod()-intersection))
    return dict(cell=cell,center=center.tolist(),size=wh.tolist(),offset=off[:,cell].tolist(),side=w,iou=iou,reliable=iou>=design["reliable_iou"])

def qualify_pool(pool,window,design):
    rows=[dict(sequence=s["sequence"],frame=s["frame"],view=v,**teacher_state(s,v,window,design)) for s in pool for v in (0,2)]
    reliable=sum(r["reliable"] for r in rows); total=len(rows)
    if not 0<reliable<total: raise ValueError("Need both reliable and unreliable native fit states for the matched anchor comparison")
    return dict(views=total,reliable_views=reliable,reliable_fraction=reliable/total,
        reliable_scale=total/reliable,gate_fingerprint=fingerprint(rows)),rows

def control_record(plan,fold,seed):
    values=[r for r in plan["controls"] if r["fold"]==fold and r["seed"]==seed]
    if len(values)!=1: raise ValueError("Need unique frozen geometry control")
    return values[0]

def make_plan(root):
    inputs=load_source(root); source=inputs["plan"]; keys=("entries","folds","root","runtime","runtime_identity","search_size","search_factor","window","training")
    support=[]
    for f in source["folds"]:
        pool,_=fit_pool(source,inputs,f["fold"]); stats,_=qualify_pool(pool,source["window"],DESIGN)
        support.append(dict(fold=f["fold"],**stats))
    plan=seal(dict(schema=SCHEMA,role="fit_only_native_geometry_anchor_comparison",source_input=str(inputs["root"]),
        source_plan_fingerprint=source["fingerprint"],source_decision_fingerprint=inputs["report"]["fingerprint"],
        source_file_sha256=inputs["bindings"],design=dict(DESIGN),support=support,seeds=list(SEEDS),arms=list(ARMS),
        controls=[r for r in inputs["records"] if r["arm"]=="geometry_only"],**{k:source[k] for k in keys},
        code_sha256={n:file_hash(Path(__file__).with_name(n)) for n in FILES+("stage52_branch_common.py","stage52_branch_experiment.py",
            "stage52_branch_runtime.py","stage52_branch_summary.py","stage51_state_runtime.py","stage18_tracking_loss.py")}))
    return plan,inputs

def load_plan(root):
    plan=read_sealed(Path(root)/"manifest.json"); expected,inputs=make_plan(plan["source_input"])
    if plan!=expected: raise ValueError("Frozen Stage-53 source/code/design changed")
    return plan,inputs

def save_cache(path,data,metadata):
    write_cache(path,data,metadata); publish_identical(path.with_suffix(".receipt.json"),seal(dict(schema=SCHEMA,metadata=metadata,sha256=file_hash(path))))

def read_cache(path,metadata):
    if not path.with_suffix(".receipt.json").exists(): return None
    r=read_sealed(path.with_suffix(".receipt.json"))
    if r.get("metadata")!=metadata or not path.is_file() or r["sha256"]!=file_hash(path): raise ValueError("Committed Stage-53 cache changed")
    with np.load(path,allow_pickle=False) as a:
        if json.loads(str(a["metadata_json"].item()))!=metadata: raise ValueError("Mixed Stage-53 cache")
        return {k:a[k] for k in a.files if k!="metadata_json"}
