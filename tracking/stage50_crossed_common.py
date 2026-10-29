"""Frozen predicted-parent x head replay plan, NumPy audits and provenance."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint, write_cache
from stage27_uav_common import crop_geometry, tracking_iou, publish_identical
from stage28_response_common import dense_boxes
from stage46_shift_audit import verify_files
import stage49_trajectory_audit as prior
import stage48_stability_common as fit

SCHEMA="stage50_shared_parent_frozen_heads_v1"
HEADS=("native",)+tuple(a+"_seed"+str(s) for s in fit.SEEDS for a in fit.ARMS)
CONFIG=[
    dict(case_id="C01",sequence="uav0000304_00253_s",purpose="early_center",frames=[552,553,554,555,556],
        focus_frames=[554],histories=["native","rank_seed2132","shape_seed2133"],
        anchor_method="rank_seed2132",anchor_kind="onset_frame",anchor_frame=554),
    dict(case_id="C02",sequence="uav0000304_00253_s",purpose="late_size",frames=[1076,1078,1081,1083,1085],
        focus_frames=[1078,1081],histories=["native","rank_shape_seed2132","rank_shape_seed2133"],
        anchor_method="rank_shape_seed2132",anchor_kind="onset_frame",anchor_frame=1078),
    dict(case_id="C03",sequence="uav0000204_00000_s",purpose="center_jump",frames=[662,663,664,665,666],
        focus_frames=[664],histories=["native","rank_shape_seed2132","rank_shape_seed2133"],
        anchor_method="native",anchor_kind="onset_frame",anchor_frame=664),
    dict(case_id="C04",sequence="uav0000226_05370_s",purpose="recovery_split",frames=[328,329,332,338,339],
        focus_frames=[338],histories=["native","supervised_seed2132","rank_seed2133","rank_shape_seed2132"],
        anchor_method="native",anchor_kind="recovery_frame",anchor_frame=338),
    dict(case_id="C05",sequence="uav0000176_00000_s",purpose="recovery_gain",frames=[238,239,240,241,242],
        focus_frames=[240],histories=["native","rank_shape_seed2132","rank_shape_seed2133"],
        anchor_method="rank_shape_seed2133",anchor_kind="recovery_frame",anchor_frame=240),
    dict(case_id="C06",sequence="uav0000307_04531_s",purpose="recovery_relapse",frames=[470,471,474,486,487],
        focus_frames=[471,487],histories=["native","rank_shape_seed2132","rank_shape_seed2133"],
        anchor_method="rank_shape_seed2132",anchor_kind="recovery_frame",anchor_frame=471)]
FILES=("stage50_crossed_common.py","stage50_crossed_collect.py","stage50_crossed_summary.py","test_stage50_crossed.py")


def seal(body): return dict(body,fingerprint=fingerprint(body))


def read_sealed(path):
    value=json.loads(Path(path).read_text(encoding="utf-8")); body={k:v for k,v in value.items() if k!="fingerprint"}
    if value.get("schema")!=SCHEMA or value.get("fingerprint")!=fingerprint(body):
        raise ValueError("Stage-50 schema/fingerprint mismatch: "+str(path))
    return value


def code_hashes():
    return {n:file_hash(Path(__file__).with_name(n)) for n in FILES+(
        "stage49_trajectory_audit.py","test_stage49_trajectory_audit.py","stage48_stability_runtime.py",
        "stage47_search_collect.py","stage28_response_common.py")}


def load_source(root):
    root=Path(root).expanduser().resolve()
    plan,run,decision,focus,logs,bindings=prior.load_completed(root)
    directory=root/"analysis/trajectory_audit_v1"
    report=prior.read_report(directory/"stage49_diagnostic.json")
    manifest=prior.read_report(directory/"manifest.json")
    if (report.get("status")!="COMPLETE_CACHED_HEAD_TRAJECTORY_AUDIT" or
        report.get("source_plan_fingerprint")!=plan["fingerprint"] or report.get("source_run_fingerprint")!=run["fingerprint"] or
        report.get("source_decision_fingerprint")!=decision["fingerprint"] or report.get("source_file_sha256")!=bindings or
        report.get("configuration")!=prior.CONFIG or any(report.get(k)!=v for k,v in manifest.items() if k!="fingerprint") or
        report.get("source_adapted_trajectories_audited")!=96 or report.get("source_native_trajectories_audited")!=12 or
        set(report.get("table_sha256",{}))!={"stage49_"+n+".csv" for n in prior.TABLES}):
        raise ValueError("Need unchanged completed Stage-49 diagnostic")
    verify_files({str(Path(__file__).parent/n):h for n,h in report["code_sha256"].items()})
    bindings.update({str(directory/n):h for n,h in report["table_sha256"].items()})
    for n in ("manifest.json","stage49_diagnostic.json"): bindings[str(directory/n)]=file_hash(directory/n)
    verify_files(bindings)
    with (directory/"stage49_onsets.csv").open(encoding="utf-8",newline="") as handle:
        import csv
        onsets=list(csv.DictReader(handle))
    return dict(plan=plan,run=run,decision=decision,focus=focus,bindings=bindings,
        source49_fingerprint=report["fingerprint"],onsets=onsets)


def make_plan(root):
    root=Path(root).expanduser().resolve(); inputs=load_source(root); cases=json.loads(json.dumps(CONFIG))
    for case in cases:
        if case["sequence"] not in inputs["focus"]: raise ValueError("Replay case absent from frozen source")
        entry,arrays=inputs["focus"][case["sequence"]]
        if (min(case["frames"])<1 or max(case["frames"])>=len(entry["frames"]) or
            not set(case["histories"])<=set(arrays) or not set(case["focus_frames"])<=set(case["frames"])):
            raise ValueError("Invalid fixed replay frames/histories")
        if not any(r["sequence"]==case["sequence"] and r["method"]==case["anchor_method"] and
            r[case["anchor_kind"]]==str(case["anchor_frame"]) for r in inputs["onsets"]):
            raise ValueError("Replay anchor does not match the completed Stage-49 event")
    count=sum(len(e["frames"])*len(e["histories"]) for e in cases)
    return seal(dict(schema=SCHEMA,role="posthoc_frozen_parent_by_head_replay",source_input=str(root),cases=cases,
        heads=list(HEADS),source_stage48_plan_fingerprint=inputs["plan"]["fingerprint"],
        source_stage48_run_fingerprint=inputs["run"]["fingerprint"],source_stage49_fingerprint=inputs["source49_fingerprint"],
        source_file_sha256=inputs["bindings"],code_sha256=code_hashes(),
        expected_inputs=count,expected_full_forwards=count,expected_head_only_forwards=count*(len(HEADS)-1),
        expected_head_forwards=count*len(HEADS),box_atol=.001,peak_atol=.00001,
        training_performed=False,continuation_performed=False)),inputs


def load_plan(root):
    plan=read_sealed(Path(root)/"manifest.json"); expected,inputs=make_plan(plan["source_input"])
    if plan!=expected: raise ValueError("Frozen Stage-50 source/code/configuration changed")
    return plan,inputs


def points(plan,smoke=False):
    result=[]
    for case in plan["cases"][:1] if smoke else plan["cases"]:
        for frame in case["frames"]:
            for index,history in enumerate(case["histories"]):
                result.append(dict(case_id=case["case_id"],sequence=case["sequence"],frame=frame,history=history,
                    point_id=case["case_id"]+"__f{:06d}__h{:02d}".format(frame,index)))
    return result


def path_for(root,point): return Path(root)/"responses"/(point["point_id"]+".npz")


def metadata(run,point): return dict(run_fingerprint=run["fingerprint"],**point)


def save(path,data,meta):
    write_cache(path,data,meta)
    publish_identical(path.with_suffix(".receipt.json"),seal(dict(schema=SCHEMA,metadata=meta,sha256=file_hash(path))))


def read_cache(path,meta):
    receipt=path.with_suffix(".receipt.json")
    if not receipt.exists(): return None
    record=read_sealed(receipt)
    if record.get("metadata")!=meta or not path.is_file() or file_hash(path)!=record.get("sha256"):
        raise ValueError("Committed Stage-50 replay changed: "+str(path))
    with np.load(path,allow_pickle=False) as archive:
        if json.loads(str(archive["metadata_json"].item()))!=meta: raise ValueError("Mixed Stage-50 replay")
        return {k:archive[k] for k in archive.files if k!="metadata_json"}


def audit_point(data,point,inputs,plan):
    source=inputs["plan"]; s=int(source["runtime"]["lens_x"]**.5); k=len(HEADS)
    shapes=dict(frame_id=(),parent=(4,),crop=(4,),image_hw=(2,),factor=(),layer=(),
        raw=(k,s,s),size=(k,2,s,s),offset=(k,2,s,s),boxes=(k,4),raw_boxes=(k,4),raw_normalized=(k,4),
        full_block_calls=(12,),selector_calls=(),head_forward_calls=(k,),crop_sha256=(),mask_sha256=(),feature_sha256=())
    if set(data)!=set(shapes) or any(data[n].shape!=shape for n,shape in shapes.items()):
        raise ValueError("Stage-50 replay shape mismatch")
    for name,value in data.items():
        if name.endswith("sha256"):
            text=str(value.item())
            if len(text)!=64 or not set(text)<=set("0123456789abcdef"): raise ValueError("Invalid replay digest")
        elif not np.isfinite(value).all(): raise ValueError("Non-finite replay")
    if (data["raw"]<=0).any() or (data["raw"]>=1).any() or (data["size"]<=0).any():
        raise ValueError("Invalid CENTER response/size probabilities")
    entry,arrays=inputs["focus"][point["sequence"]]; index=point["frame"]-1; own=arrays[point["history"]]
    hw=np.asarray([entry["frames"][point["frame"]]["height"],entry["frames"][point["frame"]]["width"]])
    np.testing.assert_array_equal(data["frame_id"],point["frame"])
    np.testing.assert_array_equal(data["parent"],own["previous"][index]); np.testing.assert_array_equal(data["image_hw"],hw)
    crop,_,factor=crop_geometry(data["parent"],hw,source["search_factor"],source["search_size"])
    np.testing.assert_array_equal(data["crop"],crop)
    if abs(float(data["factor"])-factor)>1e-12: raise ValueError("Replay crop resize mismatch")
    layer=int(data["layer"]); blocks=np.zeros(12,int); blocks[:6]=1
    if layer not in range(7,13) or layer!=own["layer"][index]: raise ValueError("Frozen selector replay changed")
    blocks[layer-1]=1
    np.testing.assert_array_equal(data["full_block_calls"],blocks)
    np.testing.assert_array_equal(data["selector_calls"],1)
    np.testing.assert_array_equal(data["head_forward_calls"],np.ones(k,int))
    window=np.asarray(source["window"],np.float32); max_decode=0.
    for h in range(k):
        dense=dense_boxes(data["size"][h],data["offset"][h],data["parent"],factor,source["search_size"],hw)
        ri=int(np.argmax(data["raw"][h])); wi=int(np.argmax(data["raw"][h]*window))
        max_decode=max(max_decode,float(np.max(np.abs(dense[ri]-data["raw_boxes"][h]))),float(np.max(np.abs(dense[wi]-data["boxes"][h]))))
        norm=np.r_[(np.array([ri%s,ri//s],np.float32)+data["offset"][h].reshape(2,-1)[:,ri])/np.float32(s),
            data["size"][h].reshape(2,-1)[:,ri]]
        np.testing.assert_allclose(data["raw_normalized"][h],norm,atol=1e-6,rtol=0)
    if max_decode>plan["box_atol"]: raise ValueError("Dense/actual decoder mismatch")
    h=HEADS.index(point["history"])
    error=float(np.max(np.abs(data["boxes"][h]-own["boxes"][index])))
    peak_error=abs(float(data["raw"][h].max())-float(own["peak"][index]))
    if error>plan["box_atol"] or peak_error>plan["peak_atol"]: raise ValueError("Own-model diagonal replay failed")
    np.testing.assert_array_equal(np.trunc(data["boxes"][h]),np.trunc(own["boxes"][index]))
    gt=np.asarray(entry["ground_truth"][point["frame"]],float)
    np.testing.assert_array_equal(tracking_iou(data["boxes"][h],gt),tracking_iou(own["boxes"][index],gt))
    return dict(maximum_diagonal_box_error=error,maximum_diagonal_peak_error=peak_error,maximum_dense_decode_error=max_decode)
