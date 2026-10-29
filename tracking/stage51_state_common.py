"""Frozen outer-fold training-state source comparison; no Torch import."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from stage4_common import file_hash, fingerprint, write_cache
from stage27_uav_common import crop_geometry, publish_identical, tracking_iou
from stage28_response_common import dense_boxes
from stage46_shift_audit import verify_files
import stage48_stability_common as prior
import stage50_crossed_common as replay

SCHEMA = "stage51_fit_only_state_source_oof_v1"
ARMS = ("native_replay", "state_mix")
SEEDS = (2132, 2133)
DESIGN = dict(donor_arm="supervised", donor_seed=2132, updates=256, smoke_updates=8,
    batch_pairs=2, save_every=128, learning_rate=1e-5, weight_decay=1e-4, gradient_clip=1.,
    smoke_frames=32, bootstrap_samples=5000, bootstrap_seed=2142,
    practical_auc_gain=.5, reliable_mean_floor=-.5, minimum_changed_pairs=16,
    minimum_changed_videos=2, views="two identical native views versus native plus fixed fit-only donor parent; +/-1px is reference only",
    labels="All uniformly sampled states kept; ordinary loss handles off-crop GT without box regression")
FILES = ("stage51_state_common.py", "stage51_state_runtime.py", "stage51_state_experiment.py",
         "stage51_state_summary.py", "test_stage51_state_source.py")


def seal(body): return dict(body, fingerprint=fingerprint(body))


def read_sealed(path):
    value=json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema")!=SCHEMA or value.get("fingerprint")!=fingerprint({k:v for k,v in value.items() if k!="fingerprint"}):
        raise ValueError("Stage-51 schema/fingerprint mismatch: "+str(path))
    return value


def load_source(root):
    root=Path(root).expanduser().resolve(); plan50,inputs=replay.load_plan(root)
    report=replay.read_sealed(root/"analysis/stage50_diagnostic.json")
    run=replay.read_sealed(root/"run.json")
    if (report.get("status")!="COMPLETE_SHARED_PARENT_HEAD_DIAGNOSTIC" or
        report.get("plan_fingerprint")!=plan50["fingerprint"] or report.get("run_fingerprint")!=run["fingerprint"] or
        run.get("plan_fingerprint")!=plan50["fingerprint"] or report.get("shared_inputs")!=plan50["expected_inputs"] or
        report.get("total_head_forwards")!=plan50["expected_head_forwards"] or
        set(report.get("table_sha256",{}))!={"stage50_"+n+".csv" for n in ("responses","cell_positions","crossed_matrix","focus")}):
        raise ValueError("Need completed unchanged Stage-50 replay")
    bindings=dict(plan50["source_file_sha256"])
    bindings.update(report["replay_cache_sha256"])
    bindings.update({str(root/"analysis"/n):h for n,h in report["table_sha256"].items()})
    for p in (root/"manifest.json",root/"run.json",root/"analysis/stage50_diagnostic.json"):
        bindings[str(p)]=file_hash(p)
    verify_files(bindings)
    return inputs["plan"],Path(plan50["source_input"]),report["fingerprint"],bindings


def donor_record(source, source_root, fold):
    report=prior.read_sealed(source_root/"training"/("seed{}".format(DESIGN["donor_seed"]))/"stage48_training.json")
    records=[r for r in report["records"] if r["fold"]==fold and r["arm"]==DESIGN["donor_arm"] and r["seed"]==DESIGN["donor_seed"]]
    if (report.get("status")!="COMPLETE_FROZEN_HEAD_TRAINING" or report.get("plan_fingerprint")!=source["fingerprint"] or len(records)!=1):
        raise ValueError("Need fixed supervised donor, not a selected best model")
    record=records[0]; path=Path(record["checkpoint"])
    receipt=prior.read_sealed(path.with_suffix(".receipt.json")); m=receipt["metadata"]
    split=source["folds"][fold]
    if (m["fold"]!=fold or m["arm"]!=DESIGN["donor_arm"] or m["seed"]!=DESIGN["donor_seed"] or
        m["fit"]!=split["fit"] or m["held_out"]!=split["held_out"] or m["plan_fingerprint"]!=source["fingerprint"] or
        m["mode"]!="full" or m["updates"]!=prior.DESIGN["updates"] or receipt["step"]!=prior.DESIGN["updates"]):
        raise ValueError("Donor fit/held-out/budget mismatch")
    verify_files({str(path):record["checkpoint_sha256"],str(path.with_suffix(".receipt.json")):record["receipt_sha256"]})
    return dict(record,fit=split["fit"],held_out=split["held_out"])


def make_plan(root):
    source,source_root,report,bindings=load_source(root)
    donors=[donor_record(source,source_root,f["fold"]) for f in source["folds"]]
    for fold in source["folds"]:
        if set(fold["fit"])&set(fold["held_out"]) or set(fold["fit"])|set(fold["held_out"])!={e["sequence"] for e in source["entries"]}:
            raise ValueError("Invalid outer source-group split")
    keys=("entries","folds","runtime","runtime_identity","root","search_factor","search_size","window")
    body=dict(schema=SCHEMA,role="consumed_development_matched_fit_state_source",source_input=str(Path(root).resolve()),
        source_stage48=str(source_root),source_stage48_plan_fingerprint=source["fingerprint"],source_stage50_fingerprint=report,
        source_file_sha256=bindings,donors=donors,design=dict(DESIGN),seeds=list(SEEDS),arms=list(ARMS),
        **{k:source[k] for k in keys},
        code_sha256={n:file_hash(Path(__file__).with_name(n)) for n in FILES+(
            "stage48_stability_common.py","stage48_stability_runtime.py","stage48_stability_experiment.py",
            "stage48_stability_summary.py","stage18_tracking_loss.py","stage50_crossed_common.py")})
    return seal(body),source


def load_plan(root):
    plan=read_sealed(Path(root)/"manifest.json"); expected,source=make_plan(plan["source_input"])
    if plan!=expected: raise ValueError("Frozen Stage-51 source/code/design changed")
    return plan,source


def fit_fold(plan,sequence):
    values=[f["fold"] for f in plan["folds"] if sequence in f["fit"]]
    if len(values)!=1: raise ValueError("Need exactly one donor training fold per video")
    return values[0]


def meta(plan,entry,role):
    fold=fit_fold(plan,entry["sequence"]); donor=plan["donors"][fold]
    if entry["sequence"] in donor["held_out"] or entry["sequence"] not in donor["fit"]:
        raise ValueError("Held-out sequence must not enter donor state generation")
    return dict(plan_fingerprint=plan["fingerprint"],sequence=entry["sequence"],fold=fold,role=role,donor_sha256=donor["checkpoint_sha256"])


def path_for(root,entry,kind): return Path(root)/kind/(entry["sequence"]+".npz")


def save_cache(path,data,metadata):
    write_cache(path,data,metadata)
    publish_identical(path.with_suffix(".receipt.json"),seal(dict(schema=SCHEMA,metadata=metadata,sha256=file_hash(path))))


def read_cache(path,metadata):
    if not path.with_suffix(".receipt.json").exists(): return None
    record=read_sealed(path.with_suffix(".receipt.json"))
    if record.get("metadata")!=metadata or not path.is_file() or record.get("sha256")!=file_hash(path):
        raise ValueError("Committed Stage-51 cache changed: "+str(path))
    with np.load(path,allow_pickle=False) as a:
        if json.loads(str(a["metadata_json"].item()))!=metadata: raise ValueError("Mixed Stage-51 cache")
        return {k:a[k] for k in a.files if k!="metadata_json"}


def audit_features(data,plan,entry,trace):
    n=len(prior.sample_frames(entry)); s=int(plan["runtime"]["lens_x"]**.5); dim=plan["runtime"]["embed_dim"]
    shapes=dict(frame_id=(n,),features=(n,dim,s,s),parents=(n,4),crop=(n,4),target=(n,4),
        raw=(n,s,s),size=(n,2,s,s),offset=(n,2,s,s),boxes=(n,4),layer=(n,))
    if set(data)!=set(shapes) or any(data[k].shape!=shape for k,shape in shapes.items()) or not all(np.isfinite(v).all() for v in data.values()):
        raise ValueError("Invalid fit-state feature pack")
    np.testing.assert_array_equal(data["frame_id"],prior.sample_frames(entry))
    for i,frame in enumerate(data["frame_id"]):
        parent=trace["previous"][int(frame)-1]; gt=np.asarray(entry["ground_truth"][int(frame)],float)
        hw=[entry["frames"][int(frame)]["height"],entry["frames"][int(frame)]["width"]]
        crop,_,factor=crop_geometry(parent,hw,plan["search_factor"],plan["search_size"])
        np.testing.assert_array_equal(data["parents"][i],parent); np.testing.assert_array_equal(data["crop"][i],crop)
        np.testing.assert_allclose(data["target"][i],np.r_[(gt[:2]-crop[:2])/crop[2],gt[2:]/crop[2]],atol=1e-10,rtol=0)
        if data["layer"][i]!=trace["layer"][int(frame)-1]: raise ValueError("Frozen selector changed on fit parent")
        dense=dense_boxes(data["size"][i],data["offset"][i],parent,factor,plan["search_size"],hw)
        cell=int(np.argmax(data["raw"][i]*np.asarray(plan["window"],np.float32)))
        if np.max(np.abs(dense[cell]-data["boxes"][i]))>.001: raise ValueError("Fit-state decode mismatch")


def source_pack(source_root,source,entry):
    pack=prior.read_cache(prior.cache_path(source_root,entry),prior.feature_meta(source,entry))
    if pack is None: raise ValueError("Need frozen Stage-48 native features")
    prior.audit_features(pack,source,entry,prior.source_native(source,entry))
    return pack


def fit_pool(root,plan,source,fold):
    pool=[]; bindings={}
    for entry in plan["entries"]:
        if entry["sequence"] not in plan["folds"][fold]["fit"]: continue
        if fit_fold(plan,entry["sequence"])!=fold: raise ValueError("Complementary OOF donor would leak held-out labels")
        tp=path_for(root,entry,"donor_traces"); fp=path_for(root,entry,"features")
        trace=read_cache(tp,meta(plan,entry,"fit_only_donor_trajectory"))
        own=read_cache(fp,meta(plan,entry,"fit_only_original_features"))
        if trace is None or own is None: raise ValueError("Collect all fit-only states before training")
        from stage48_stability_experiment import audit_trace
        audit_trace(trace,entry); audit_features(own,plan,entry,trace)
        base=source_pack(Path(plan["source_stage48"]),source,entry)
        for p in (tp,fp,prior.cache_path(Path(plan["source_stage48"]),entry)):
            for q in (p,p.with_suffix(".receipt.json")): bindings[str(q)]=file_hash(q)
        for i,frame in enumerate(own["frame_id"]):
            pool.append(dict(sequence=entry["sequence"],frame=int(frame),
                features=np.concatenate([base["features"][i],own["features"][i:i+1]],axis=0),
                target=np.concatenate([base["target"][i],own["target"][i:i+1]],axis=0),
                raw=np.concatenate([base["raw"][i],own["raw"][i:i+1]],axis=0),
                size=np.concatenate([base["size"][i],own["size"][i:i+1]],axis=0),
                offset=np.concatenate([base["offset"][i],own["offset"][i:i+1]],axis=0)))
    if not pool: raise ValueError("Empty fit-only training pool")
    return pool,bindings
