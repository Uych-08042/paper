"""Frozen Stage-58 heads and a factorial intervention on predicted search parents."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from stage4_common import file_hash,fingerprint,write_cache
from stage27_uav_common import crop_geometry,failure_events,tracking_iou,publish_identical
from stage28_response_common import dense_boxes
from stage46_shift_audit import verify_files
from stage49_trajectory_audit import table_rows,check_values
from stage51_state_summary import describe
import stage58_gradient_common as prior
import stage58_gradient_experiment as previous
from stage58_gradient_summary import METRICS

SCHEMA="stage59_parent_component_continuation_v1"
ARMS=("endpoint_control","projected_anchor")
DONORS=("E","P")
DESIGN=dict(horizon=32,box_atol=.001,peak_atol=1e-5,response_atol=1e-6)
CONFIG=[
    dict(case_id="C01",sequence="uav0000226_05370_s",purpose="lost_recovery",frame=338,anchor_method="endpoint_control_seed2132",anchor_kind="recovery_frame"),
    dict(case_id="C02",sequence="uav0000176_00000_s",purpose="preserved_recovery",frame=238,anchor_method="projected_anchor_seed2133",anchor_kind="recovery_frame"),
    dict(case_id="C03",sequence="uav0000176_00000_s",purpose="delayed_failure",frame=315,anchor_method="projected_anchor_seed2132",anchor_kind="onset_frame"),
    dict(case_id="C04",sequence="uav0000304_00253_s",purpose="early_persistent_failure",frame=555,anchor_method="projected_anchor_seed2133",anchor_kind="onset_frame"),
    dict(case_id="C05",sequence="uav0000304_00253_s",purpose="late_persistent_failure",frame=1086,anchor_method="projected_anchor_seed2132",anchor_kind="onset_frame"),
    dict(case_id="C06",sequence="uav0000307_04531_s",purpose="recovery_not_retained",frame=511,anchor_method="endpoint_control_seed2133",anchor_kind="recovery_frame"),
    dict(case_id="C07",sequence="uav0000204_00000_s",purpose="stable_branch_gain",frame=664,anchor_method="endpoint_control_seed2132",anchor_kind="onset_frame")]
FILES=("stage59_parent_common.py","stage59_parent_collect.py","stage59_parent_summary.py","test_stage59_parent.py")


def seal(body): return dict(body,fingerprint=fingerprint(body))


def read_sealed(path):
    value=json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema")!=SCHEMA or value.get("fingerprint")!=fingerprint({k:v for k,v in value.items() if k!="fingerprint"}):
        raise ValueError("Stage-59 schema/fingerprint changed: "+str(path))
    return value


def load_source(root):
    root=Path(root).expanduser().resolve(); plan,source57=prior.load_plan(root)
    previous.smoke_gate(plan,root); records,models=previous.model_records(plan,root)
    run=prior.read_sealed(root/"evaluation/run.json"); report=prior.read_sealed(root/"analysis/stage58_decision.json")
    tables={"stage58_"+n+".csv" for n in ("policy_summary","paired","scope_summary","retention_summary","video_summary","seed_video_summary","focus","events")}
    if (report.get("status") not in ("REVIEW_PROJECTED_ENDPOINT_GRADIENT_PILOT","PROMISING_PROJECTED_ENDPOINT_GRADIENT_PILOT") or
        report.get("role")!="consumed_development_projected_endpoint_gradient" or report.get("plan_fingerprint")!=plan["fingerprint"] or
        run.get("plan_fingerprint")!=plan["fingerprint"] or report.get("run_fingerprint")!=run["fingerprint"] or
        run.get("role")!="native_short_rollout_oof" or run.get("model_sha256")!=models or report.get("model_sha256")!=models or
        report.get("videos")!=12 or report.get("new_models")!=12 or report.get("new_full_trajectories")!=72 or
        report.get("seeds")!=list(prior.SEEDS) or set(report.get("table_sha256",{}))!=tables):
        raise ValueError("Need complete unmixed Stage-58 source")
    bindings=dict(plan["source_file_sha256"]); bindings.update(models); bindings.update(report["source_file_sha256"])
    bindings.update({str(root/"analysis"/n):h for n,h in report["table_sha256"].items()})
    for name in ("manifest.json","evaluation/run.json","analysis/stage58_decision.json","smoke/stage58_smoke.json"):
        bindings[str(root/name)]=file_hash(root/name)
    verify_files(bindings)
    seed_rows=table_rows(root/"analysis/stage58_seed_video_summary.csv",("sequence","arm","seed"))
    video_rows=table_rows(root/"analysis/stage58_video_summary.csv",("sequence","arm"))
    policy_rows=table_rows(root/"analysis/stage58_policy_summary.csv",("arm",))
    if (len(seed_rows),len(video_rows),len(policy_rows))!=(132,72,6): raise ValueError("Incomplete Stage-58 summaries")
    arrays={}; recomputed=[]; calls=0
    for entry in plan["entries"]:
        native=prior.native_trace(source57,entry)
        control={s:prior.source_trace(source57,entry,"geometry_only",s) for s in prior.SEEDS}
        mask=np.any([tracking_iou(v["boxes"],np.asarray(entry["ground_truth"],float)[1:])<.2 for v in control.values()],axis=0)
        values=[("native",0,native)]+[("geometry_control",s,v) for s,v in control.items()]
        values.extend(("static_anchor",s,prior.source_trace(source57,entry,"anchor_reliable",s)) for s in prior.SEEDS)
        own={"native":native}
        for seed in prior.SEEDS:
            for arm in prior.ARMS:
                method=arm+"_seed"+str(seed); path=root/"evaluation/traces"/(entry["sequence"]+"__"+method+".npz")
                data=prior.read_cache(path,dict(run_fingerprint=run["fingerprint"],sequence=entry["sequence"],method=method))
                if data is None: raise ValueError("Missing completed Stage-58 trace")
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
    if (report.get("inference_head_calls"),report.get("inference_block_calls"))!=(calls,calls*7): raise ValueError("Stage-58 source call budget changed")
    return dict(root=root,plan=plan,report=report,run=run,source57=source57,records=[r for r,_ in records],arrays=arrays,bindings=bindings)


def evaluator(donor,seed):
    return ARMS[DONORS.index(donor)]+"_seed"+str(seed)


def point_heads(item): return ("native",evaluator("E",item["seed"]),evaluator("P",item["seed"]))


def hybrid_parent(center_box,size_box,same=False):
    center_box,size_box=np.asarray(center_box,float),np.asarray(size_box,float)
    if center_box.shape!=(4,) or size_box.shape!=(4,) or not np.isfinite([center_box,size_box]).all() or np.any(center_box[2:]<=0) or np.any(size_box[2:]<=0):
        raise ValueError("Invalid predicted parent components")
    if same:
        np.testing.assert_array_equal(center_box,size_box)
        return center_box.copy()  # Preserve diagonal source bytes, no rebuild.
    center=center_box[:2]+center_box[2:]/2
    return np.r_[center-size_box[2:]/2,size_box[2:]]


def parent_for(inputs,item):
    arrays=inputs["arrays"][item["sequence"]][1]; i=item["frame"]-1
    left=arrays[evaluator(item["center_donor"],item["seed"])]["previous"][i]
    right=arrays[evaluator(item["size_donor"],item["seed"])]["previous"][i]
    return hybrid_parent(left,right,item["center_donor"]==item["size_donor"])


def points(plan,smoke=False):
    result=[]
    for case in plan["cases"][:1] if smoke else plan["cases"]:
        for seed in prior.SEEDS:
            for center in DONORS:
                for size in DONORS:
                    identity="{}__s{}__c{}__z{}".format(case["case_id"],seed,center,size)
                    result.append(dict(case_id=case["case_id"],sequence=case["sequence"],frame=case["frame"],purpose=case["purpose"],
                        seed=seed,center_donor=center,size_donor=size,point_id=identity))
    return result


def branches(plan,smoke=False):
    return [dict(p,evaluator=evaluator(d,p["seed"]),branch_id=p["point_id"]+"__"+d) for p in points(plan,smoke) for d in DONORS]


def make_plan(root):
    inputs=load_source(root); source=inputs["plan"]; cases=json.loads(json.dumps(CONFIG))
    for case in cases:
        entry,arrays=inputs["arrays"][case["sequence"]]; frame=case["frame"]
        if frame<1 or frame+DESIGN["horizon"]>len(entry["frames"]): raise ValueError("Need fixed frame and complete horizon")
        events=failure_events(tracking_iou(arrays[case["anchor_method"]]["boxes"],np.asarray(entry["ground_truth"],float)[1:]))
        key={"onset_frame":"start","recovery_frame":"recovery"}.get(case["anchor_kind"])
        if key and not any(e[key] is not None and e[key]+1==frame for e in events): raise ValueError("Case anchor changed")
        if key is None and case["anchor_kind"]!="fixed_frame": raise ValueError("Unknown case anchor kind")
    plan=dict(schema=SCHEMA,role="posthoc_predicted_parent_component_factorial",source_input=str(inputs["root"]),
        source_plan_fingerprint=source["fingerprint"],source_decision_fingerprint=inputs["report"]["fingerprint"],
        source_file_sha256=inputs["bindings"],cases=cases,seeds=list(prior.SEEDS),donors=list(DONORS),design=dict(DESIGN),
        code_sha256={n:file_hash(Path(__file__).with_name(n)) for n in FILES+prior.FILES+("stage55_matched_collect.py","stage50_crossed_collect.py","stage47_search_collect.py","stage55_matched_summary.py","stage50_crossed_summary.py")})
    n=len(points(plan)); b=len(branches(plan))
    plan.update(expected_points=n,expected_branches=b,expected_full_forwards=n+b*DESIGN["horizon"],expected_head_only_forwards=n*2)
    for p in points(plan): parent_for(inputs,p)
    return seal(plan),inputs


def load_plan(root):
    plan=read_sealed(Path(root)/"manifest.json"); expected,inputs=make_plan(plan["source_input"])
    if plan!=expected: raise ValueError("Frozen Stage-59 code/source/design changed")
    return plan,inputs


def path_for(root,item):
    return Path(root)/("branches" if "branch_id" in item else "responses")/(item.get("branch_id",item["point_id"])+".npz")


def metadata(run,item): return dict(run_fingerprint=run["fingerprint"],**item)

def save(path,data,meta):
    write_cache(path,data,meta); publish_identical(path.with_suffix(".receipt.json"),seal(dict(schema=SCHEMA,metadata=meta,sha256=file_hash(path))))
def read_cache(path,meta):
    if not path.with_suffix(".receipt.json").exists(): return None
    r=read_sealed(path.with_suffix(".receipt.json"))
    if r.get("metadata")!=meta or not path.is_file() or r["sha256"]!=file_hash(path): raise ValueError("Committed Stage-59 cache changed")
    with np.load(path,allow_pickle=False) as a:
        if json.loads(str(a["metadata_json"].item()))!=meta: raise ValueError("Mixed Stage-59 cache")
        return {k:a[k] for k in a.files if k!="metadata_json"}

def native_calls(blocks,layer,head,selector):
    expected=np.zeros(12,int); expected[:6]=1
    if int(layer) not in range(7,13): raise ValueError("Invalid native selector layer")
    expected[int(layer)-1]=1
    np.testing.assert_array_equal(blocks,expected)
    if int(head)!=1 or int(selector)!=1: raise ValueError("Not one native head/selector call")

def audit_point(data,item,plan,inputs):
    source=inputs["plan"]; entry,arrays=inputs["arrays"][item["sequence"]]; i=item["frame"]-1; heads=point_heads(item)
    side=int(source["runtime"]["lens_x"]**.5); k=len(heads)
    shapes=dict(frame_id=(),parent=(4,),crop=(4,),image_hw=(2,),factor=(),layer=(),raw=(k,side,side),size=(k,2,side,side),
        offset=(k,2,side,side),boxes=(k,4),raw_boxes=(k,4),raw_normalized=(k,4),full_block_calls=(12,),selector_calls=(),
        head_forward_calls=(k,),crop_sha256=(),mask_sha256=(),feature_sha256=())
    if set(data)!=set(shapes) or any(data[n].shape!=s for n,s in shapes.items()): raise ValueError("Replay shape changed")
    for key,v in data.items():
        if key.endswith("sha256"):
            if len(str(v.item()))!=64 or not set(str(v.item()))<=set("0123456789abcdef"): raise ValueError("Invalid replay hash")
        elif not np.isfinite(v).all(): raise ValueError("Non-finite replay")
    np.testing.assert_array_equal(data["frame_id"],item["frame"]); np.testing.assert_array_equal(data["parent"],parent_for(inputs,item))
    np.testing.assert_array_equal(data["image_hw"],[entry["frames"][item["frame"]]["height"],entry["frames"][item["frame"]]["width"]])
    crop,_,factor=crop_geometry(data["parent"],data["image_hw"],source["search_factor"],source["search_size"])
    np.testing.assert_array_equal(data["crop"],crop)
    if abs(float(data["factor"])-factor)>1e-12: raise ValueError("Crop/selector replay changed")
    native_calls(data["full_block_calls"],data["layer"],data["head_forward_calls"][0],data["selector_calls"])
    np.testing.assert_array_equal(data["head_forward_calls"],np.ones(k,int))
    response=float(np.max(np.abs(data["raw"]-data["raw"][:1])))
    if response>plan["design"]["response_atol"]: raise ValueError("Frozen response differs on identical features")
    decode=0.
    for h in range(k):
        dense=dense_boxes(data["size"][h],data["offset"][h],data["parent"],factor,source["search_size"],data["image_hw"])
        ri=int(np.argmax(data["raw"][h])); wi=int(np.argmax(data["raw"][h]*np.asarray(source["window"],np.float32)))
        decode=max(decode,float(np.max(np.abs(dense[wi]-data["boxes"][h]))),float(np.max(np.abs(dense[ri]-data["raw_boxes"][h]))))
    diagonal=0.
    if item["center_donor"]==item["size_donor"]:
        method=evaluator(item["center_donor"],item["seed"]); own=arrays[method]; h=heads.index(method)
        diagonal=float(np.max(np.abs(data["boxes"][h]-own["boxes"][i])))
        np.testing.assert_array_equal(np.trunc(data["boxes"][h]),np.trunc(own["boxes"][i]))
        if int(data["layer"])!=own["layer"][i] or abs(float(data["raw"][h].max())-own["peak"][i])>plan["design"]["peak_atol"]:
            raise ValueError("Own parent selector/peak changed")
    if max(decode,diagonal)>plan["design"]["box_atol"]: raise ValueError("Own-parent/dense decoder replay changed")
    return dict(maximum_diagonal_error=diagonal,maximum_decode_error=decode,maximum_fixed_response_error=response)

def audit_branch(data,item,point,plan,inputs):
    source=inputs["plan"]; entry,arrays=inputs["arrays"][item["sequence"]]; n=plan["design"]["horizon"]; s=int(source["runtime"]["lens_x"]**.5)
    shapes=dict(frame_id=(n,),parents=(n,4),crop=(n,4),image_hw=(n,2),factor=(n,),boxes=(n,4),layer=(n,),peak=(n,),
        raw=(n,s,s),size=(n,2,s,s),offset=(n,2,s,s),block_calls=(n,12),head_calls=(n,),selector_calls=(n,),crop_sha256=(n,),mask_sha256=(n,))
    if set(data)!=set(shapes) or any(data[k].shape!=v for k,v in shapes.items()): raise ValueError("Branch shape changed")
    for k,v in data.items():
        if k.endswith("sha256"):
            if any(len(str(x))!=64 or not set(str(x))<=set("0123456789abcdef") for x in v): raise ValueError("Invalid branch crop hash")
        elif not np.isfinite(v).all(): raise ValueError("Non-finite branch")
    frames=np.arange(item["frame"],item["frame"]+n); np.testing.assert_array_equal(data["frame_id"],frames)
    np.testing.assert_array_equal(data["parents"][0],parent_for(inputs,item))
    np.testing.assert_array_equal(data["parents"][1:],data["boxes"][:-1])
    first=float(np.max(np.abs(data["boxes"][0]-point["boxes"][point_heads(item).index(item["evaluator"])])))
    first_maps=max(float(np.max(np.abs(data[k][0]-point[k][point_heads(item).index(item["evaluator"])]))) for k in ("raw","size","offset"))
    if first_maps>plan["design"]["response_atol"]: raise ValueError("Shared-feature/actor first maps changed")
    if float(np.max(np.abs(data["peak"]-data["raw"].reshape(n,-1).max(1))))>plan["design"]["peak_atol"]:
        raise ValueError("Branch response peak changed")
    decode=0.
    for j,f in enumerate(frames):
        hw=np.array([entry["frames"][f]["height"],entry["frames"][f]["width"]]); np.testing.assert_array_equal(data["image_hw"][j],hw)
        crop,_,factor=crop_geometry(data["parents"][j],hw,source["search_factor"],source["search_size"])
        np.testing.assert_array_equal(crop,data["crop"][j]); native_calls(data["block_calls"][j],data["layer"][j],data["head_calls"][j],data["selector_calls"][j])
        if abs(factor-data["factor"][j])>1e-12: raise ValueError("Branch factor changed")
        dense=dense_boxes(data["size"][j],data["offset"][j],data["parents"][j],factor,source["search_size"],hw)
        wi=int(np.argmax(data["raw"][j]*np.asarray(source["window"],np.float32)))
        decode=max(decode,float(np.max(np.abs(dense[wi]-data["boxes"][j]))))
    diagonal=0.
    if item["center_donor"]==item["size_donor"] and item["evaluator"]==evaluator(item["center_donor"],item["seed"]):
        own=arrays[item["evaluator"]]; ix=frames-1
        diagonal=float(np.max(np.abs(data["boxes"]-own["boxes"][ix])))
        np.testing.assert_array_equal(np.trunc(data["boxes"]),np.trunc(own["boxes"][ix])); np.testing.assert_array_equal(data["layer"],own["layer"][ix])
    if max(first,decode,diagonal)>plan["design"]["box_atol"]: raise ValueError("Matched first-frame/native diagonal branch changed")
    return dict(maximum_first_frame_error=first,maximum_first_map_error=first_maps,maximum_branch_decode_error=decode,maximum_branch_diagonal_error=diagonal)
