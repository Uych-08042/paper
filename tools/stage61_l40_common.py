"""Separate L40 execution of the fixed Stage-61 six-arm comparison."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
import stage61_runtime_replay as startup  # Also sets up repository/tracking imports.
import stage61_search_common as original
from stage4_common import file_hash, fingerprint, write_cache
from stage27_uav_common import publish_identical

SCHEMA="stage61_l40_device_matched_recrop_v1"
PREFIX="stage61_l40"
ORIGINS=original.ORIGINS
POLICIES=original.POLICIES
FILES=("stage61_l40_common.py","stage61_l40_experiment.py","stage61_l40_summary.py",
       "test_stage61_l40.py","stage61_runtime_replay.py","run_stage61_l40.sh")
verify_files=original.verify_files
key=original.key


def seal(body): return dict(body,fingerprint=fingerprint(body))


def read_sealed(path):
    value=json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema")!=SCHEMA or value.get("fingerprint")!=fingerprint({k:v for k,v in value.items() if k!="fingerprint"}):
        raise ValueError("L40 sealed content changed: "+str(path))
    return value


def device_binding(plan,gpu,torch):
    if not torch.cuda.is_available(): raise ValueError("L40 execution requires CUDA")
    device=torch.cuda.get_device_name(gpu)
    if device!="NVIDIA L40": raise ValueError("This separately declared run requires NVIDIA L40, got "+device)
    for field,value in (("torch",str(torch.__version__)),("cuda",str(torch.version.cuda))):
        if value!=plan["runtime"][field]: raise ValueError("Source software differs: "+field)
    return dict(device=device,gpu_uuid=str(getattr(torch.cuda.get_device_properties(gpu),"uuid","unavailable")))


def make_plan(source_root,hardware):
    source_root=Path(source_root).expanduser().resolve()
    old,source=original.load_plan(source_root)
    if set(hardware)!={"device","gpu_uuid"} or hardware["device"]!="NVIDIA L40" or not hardware["gpu_uuid"]:
        raise ValueError("Need an observed L40 device and UUID")
    if old["runtime_identity"]["tracker"]!=old["runtime"]: raise ValueError("Source runtime binding differs")
    runtime=copy.deepcopy(old["runtime"]); runtime["device"]=hardware["device"]
    identity=copy.deepcopy(old["runtime_identity"])
    identity["tracker"]=runtime; identity["gpu_uuid"]=hardware["gpu_uuid"]
    changed=startup.differences(old["runtime_identity"],identity)
    if any(r["field"] not in ("runtime.tracker.device","runtime.gpu_uuid") for r in changed):
        raise ValueError("Only physical GPU identity can change in this migration")
    units=[]
    for ordinal,entry in enumerate(old["entries"]):
        name=entry["sequence"]
        fold=next(f["fold"] for f in old["folds"] if name in f["held_out"])
        own=[dict(sequence=name,origin="native",policy=p,seed=0,fold=-1) for p in POLICIES]
        own += [dict(sequence=name,origin="endpoint",policy=p,seed=s,fold=fold) for s in old["seeds"] for p in POLICIES]
        offset=ordinal%len(own); units.extend(own[offset:]+own[:offset])
    frames={e["sequence"]:len(e["frames"])-1 for e in old["entries"]}
    calls=sum((1 if u["policy"]=="current" else 4)*frames[u["sequence"]] for u in units)
    smoke_frames=min(old["design"]["smoke_frames"],frames[old["entries"][0]["sequence"]])
    bindings=dict(old["source_file_sha256"])
    bindings[str(source_root/"manifest.json")]=file_hash(source_root/"manifest.json")
    body=dict(schema=SCHEMA,role="consumed_development_l40_all_six_arms_remeasured",
        source_stage61=str(source_root),source_stage61_plan_fingerprint=old["fingerprint"],
        source_file_sha256=bindings,source_runtime_identity=copy.deepcopy(old["runtime_identity"]),
        hardware=dict(hardware),declared_runtime_differences=changed,runtime=runtime,runtime_identity=identity,
        comparison_cohort="new L40 trajectories only; original L20 traces are provenance/bridge descriptions",
        design=copy.deepcopy(old["design"]),units=units,new_full_trajectories=len(units),reused_full_trajectories=0,
        formal_search_calls=calls,smoke_policy_calls=smoke_frames*9*(1+len(old["seeds"])),
        smoke_public_calls=smoke_frames*(1+len(old["seeds"])),
        **{k:copy.deepcopy(old[k]) for k in ("entries","folds","root","seeds","search_size","search_factor","window")},
        code_sha256={n:file_hash(Path(__file__).with_name(n)) for n in FILES})
    body["smoke_search_calls"]=body["smoke_policy_calls"]+body["smoke_public_calls"]
    return seal(body),source


def load_plan(root):
    plan=read_sealed(Path(root)/"manifest.json")
    expected,source=make_plan(plan["source_stage61"],plan["hardware"])
    if expected!=plan: raise ValueError("Frozen L40 source/code/design changed")
    return plan,source


def meta(plan,unit,kind):
    return dict(plan_fingerprint=plan["fingerprint"],runtime_fingerprint=fingerprint(plan["runtime_identity"]),kind=kind,**unit)


def save(path,data,metadata):
    path=Path(path)
    if path.with_suffix(".receipt.json").exists(): raise ValueError("Never overwrite a committed L40 cache")
    if path.exists():
        with np.load(path,allow_pickle=False) as f:
            if json.loads(str(f["metadata_json"].item()))!=metadata: raise ValueError("Unidentified orphan L40 cache")
    write_cache(path,data,metadata)
    publish_identical(path.with_suffix(".receipt.json"),seal(dict(schema=SCHEMA,metadata=metadata,sha256=file_hash(path))))


def read_cache(path,metadata):
    path=Path(path); receipt=path.with_suffix(".receipt.json")
    if not receipt.exists(): return None
    value=read_sealed(receipt)
    if value.get("metadata")!=metadata or not path.is_file() or file_hash(path)!=value["sha256"]:
        raise ValueError("Committed L40 cache changed")
    with np.load(path,allow_pickle=False) as f:
        if json.loads(str(f["metadata_json"].item()))!=metadata: raise ValueError("Mixed L40 cache")
        return {k:f[k] for k in f.files if k!="metadata_json"}
