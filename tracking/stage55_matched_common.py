"""Completed Stage-54 provenance and fixed parent/head/continuation diagnosis."""
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
import stage54_rollout_common as prior
import stage54_rollout_experiment as previous
from stage54_rollout_summary import METRICS

SCHEMA="stage55_matched_parent_continuation_v1"
HEADS=("native",)+tuple(a+"_seed"+str(s) for s in prior.SEEDS for a in prior.ARMS)
DESIGN=dict(horizon=32,offsets=[-1,0,1],box_atol=.001,peak_atol=1e-5,response_atol=1e-6)
CONFIG=[
    dict(case_id="C01",sequence="uav0000204_00000_s",purpose="lost_stable_branch",frame=664,
         anchor_method="rollout_preserve_seed2132",anchor_kind="onset_frame"),
    dict(case_id="C02",sequence="uav0000304_00253_s",purpose="early_seed_split",frame=554,
         anchor_method="rollout_supervised_seed2132",anchor_kind="onset_frame"),
    dict(case_id="C03",sequence="uav0000304_00253_s",purpose="late_remaining_failure",frame=1078,
         anchor_method="rollout_preserve_seed2132",anchor_kind="onset_frame"),
    dict(case_id="C04",sequence="uav0000307_04531_s",purpose="shared_failure_start",frame=389,
         anchor_method="rollout_preserve_seed2132",anchor_kind="onset_frame"),
    dict(case_id="C05",sequence="uav0000307_04531_s",purpose="recovery_followup",frame=471,
         anchor_method="rollout_preserve_seed2132",anchor_kind="recovery_frame")]
FILES=("stage55_matched_common.py","stage55_matched_collect.py","stage55_matched_summary.py","test_stage55_matched.py")

def seal(body): return dict(body,fingerprint=fingerprint(body))

def read_sealed(path):
    value=json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema")!=SCHEMA or value.get("fingerprint")!=fingerprint({k:v for k,v in value.items() if k!="fingerprint"}):
        raise ValueError("Stage-55 schema/fingerprint changed: "+str(path))
    return value

def load_source(root):
    root=Path(root).expanduser().resolve(); plan,source53=prior.load_plan(root)
    previous.smoke_gate(plan,root); records,models=previous.model_records(plan,root)
    run=prior.read_sealed(root/"evaluation/run.json"); report=prior.read_sealed(root/"analysis/stage54_decision.json")
    tables={"stage54_"+n+".csv" for n in ("policy_summary","paired","scope_summary","retention_summary","video_summary","seed_video_summary","focus","events")}
    if (report.get("status") not in ("REVIEW_SHORT_ROLLOUT_ENDPOINT_PILOT","PROMISING_SHORT_ROLLOUT_ENDPOINT_PILOT") or
        report.get("role")!="consumed_development_current_policy_endpoint" or report.get("plan_fingerprint")!=plan["fingerprint"] or
        run.get("plan_fingerprint")!=plan["fingerprint"] or report.get("run_fingerprint")!=run["fingerprint"] or
        run.get("role")!="native_short_rollout_oof" or run.get("model_sha256")!=models or report.get("model_sha256")!=models or
        report.get("videos")!=12 or report.get("new_models")!=8 or report.get("new_full_trajectories")!=48 or
        report.get("seeds")!=list(prior.SEEDS) or set(report.get("table_sha256",{}))!=tables):
        raise ValueError("Need complete unmixed Stage-54 source")
    bindings=dict(plan["source_file_sha256"]); bindings.update(models); bindings.update(report["source_file_sha256"])
    bindings.update({str(root/"analysis"/n):h for n,h in report["table_sha256"].items()})
    for name in ("manifest.json","evaluation/run.json","analysis/stage54_decision.json","smoke/stage54_smoke.json"):
        bindings[str(root/name)]=file_hash(root/name)
    verify_files(bindings)
    seed_rows=table_rows(root/"analysis/stage54_seed_video_summary.csv",("sequence","arm","seed"))
    video_rows=table_rows(root/"analysis/stage54_video_summary.csv",("sequence","arm"))
    policy_rows=table_rows(root/"analysis/stage54_policy_summary.csv",("arm",))
    if (len(seed_rows),len(video_rows),len(policy_rows))!=(108,60,5): raise ValueError("Incomplete Stage-54 summaries")
    arrays={}; recomputed=[]; calls=0
    for entry in plan["entries"]:
        native=prior.native_trace(source53,entry)
        control={s:prior.source_trace(source53,entry,"geometry_only",s) for s in prior.SEEDS}
        mask=np.any([tracking_iou(v["boxes"],np.asarray(entry["ground_truth"],float)[1:])<.2 for v in control.values()],axis=0)
        values=[("native",0,native)]+[("geometry_control",s,v) for s,v in control.items()]
        values.extend(("static_anchor",s,prior.source_trace(source53,entry,"anchor_reliable",s)) for s in prior.SEEDS)
        own={"native":native}
        for seed in prior.SEEDS:
            for arm in prior.ARMS:
                method=arm+"_seed"+str(seed); path=root/"evaluation/traces"/(entry["sequence"]+"__"+method+".npz")
                data=prior.read_cache(path,dict(run_fingerprint=run["fingerprint"],sequence=entry["sequence"],method=method))
                if data is None: raise ValueError("Missing completed Stage-54 trace")
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
    if (report.get("inference_head_calls"),report.get("inference_block_calls"))!=(calls,calls*7): raise ValueError("Stage-54 source call budget changed")
    return dict(root=root,plan=plan,report=report,run=run,source53=source53,records=[r for r,_ in records],arrays=arrays,bindings=bindings)

def make_plan(root):
    inputs=load_source(root); cases=json.loads(json.dumps(CONFIG)); source=inputs["plan"]
    for case in cases:
        entry,arrays=inputs["arrays"][case["sequence"]]; frame=case["frame"]
        if min(frame+o for o in DESIGN["offsets"])<1 or max(frame+o for o in DESIGN["offsets"])>=len(entry["frames"]) or frame+DESIGN["horizon"]>len(entry["frames"]):
            raise ValueError("Case requires all fixed frames and complete horizon")
        events=failure_events(tracking_iou(arrays[case["anchor_method"]]["boxes"],np.asarray(entry["ground_truth"],float)[1:]))
        key={"onset_frame":"start","recovery_frame":"recovery"}.get(case["anchor_kind"])
        if key is not None and not any(e[key] is not None and e[key]+1==frame for e in events): raise ValueError("Case anchor no longer matches source event")
        if key is None and case["anchor_kind"]!="fixed_frame": raise ValueError("Unknown anchor kind")
    n=len(cases)*len(DESIGN["offsets"])*len(HEADS); b=len(cases)*len(HEADS)**2
    body=dict(schema=SCHEMA,role="posthoc_frozen_parent_head_short_continuation",source_input=str(inputs["root"]),
        source_plan_fingerprint=source["fingerprint"],source_decision_fingerprint=inputs["report"]["fingerprint"],
        source_file_sha256=inputs["bindings"],cases=cases,heads=list(HEADS),design=dict(DESIGN),
        expected_points=n,expected_branches=b,expected_full_forwards=n+b*DESIGN["horizon"],expected_head_only_forwards=n*(len(HEADS)-1),
        code_sha256={n:file_hash(Path(__file__).with_name(n)) for n in FILES+prior.FILES+("stage50_crossed_collect.py","stage50_crossed_summary.py","stage47_search_collect.py")})
    return seal(body),inputs

def load_plan(root):
    plan=read_sealed(Path(root)/"manifest.json"); expected,inputs=make_plan(plan["source_input"])
    if plan!=expected: raise ValueError("Frozen Stage-55 source/code/design changed")
    return plan,inputs

def points(plan,smoke=False):
    return [dict(case_id=c["case_id"],sequence=c["sequence"],frame=c["frame"]+offset,offset=offset,history=h,
        point_id=c["case_id"]+"__f{:06d}__h{:02d}".format(c["frame"]+offset,i))
        for c in (plan["cases"][:1] if smoke else plan["cases"]) for offset in plan["design"]["offsets"] for i,h in enumerate(HEADS)]

def branches(plan,smoke=False):
    return [dict(case_id=c["case_id"],sequence=c["sequence"],frame=c["frame"],history=h,evaluator=e,
        branch_id=c["case_id"]+"__h{:02d}__e{:02d}".format(i,j))
        for c in (plan["cases"][:1] if smoke else plan["cases"]) for i,h in enumerate(HEADS[:1] if smoke else HEADS) for j,e in enumerate(HEADS)]

def path_for(root,item): return Path(root)/("branches" if "branch_id" in item else "responses")/((item.get("branch_id") or item["point_id"])+".npz")
def metadata(run,item): return dict(run_fingerprint=run["fingerprint"],**item)
def save(path,data,meta):
    write_cache(path,data,meta); publish_identical(path.with_suffix(".receipt.json"),seal(dict(schema=SCHEMA,metadata=meta,sha256=file_hash(path))))
def read_cache(path,meta):
    if not path.with_suffix(".receipt.json").exists(): return None
    r=read_sealed(path.with_suffix(".receipt.json"))
    if r.get("metadata")!=meta or not path.is_file() or r["sha256"]!=file_hash(path): raise ValueError("Committed Stage-55 cache changed")
    with np.load(path,allow_pickle=False) as a:
        if json.loads(str(a["metadata_json"].item()))!=meta: raise ValueError("Mixed Stage-55 cache")
        return {k:a[k] for k in a.files if k!="metadata_json"}

def native_calls(blocks,layer,head,selector):
    expected=np.zeros(12,int); expected[:6]=1
    if int(layer) not in range(7,13): raise ValueError("Invalid native selector layer")
    expected[int(layer)-1]=1
    np.testing.assert_array_equal(blocks,expected)
    if int(head)!=1 or int(selector)!=1: raise ValueError("Not one native head/selector call")

def audit_point(data,item,plan,inputs):
    source=inputs["plan"]; entry,arrays=inputs["arrays"][item["sequence"]]; own=arrays[item["history"]]; i=item["frame"]-1
    side=int(source["runtime"]["lens_x"]**.5); k=len(HEADS)
    shapes=dict(frame_id=(),parent=(4,),crop=(4,),image_hw=(2,),factor=(),layer=(),raw=(k,side,side),size=(k,2,side,side),
        offset=(k,2,side,side),boxes=(k,4),raw_boxes=(k,4),raw_normalized=(k,4),full_block_calls=(12,),selector_calls=(),
        head_forward_calls=(k,),crop_sha256=(),mask_sha256=(),feature_sha256=())
    if set(data)!=set(shapes) or any(data[n].shape!=s for n,s in shapes.items()): raise ValueError("Replay shape changed")
    for key,v in data.items():
        if key.endswith("sha256"):
            if len(str(v.item()))!=64 or not set(str(v.item()))<=set("0123456789abcdef"): raise ValueError("Invalid replay hash")
        elif not np.isfinite(v).all(): raise ValueError("Non-finite replay")
    np.testing.assert_array_equal(data["frame_id"],item["frame"]); np.testing.assert_array_equal(data["parent"],own["previous"][i])
    np.testing.assert_array_equal(data["image_hw"],[entry["frames"][item["frame"]]["height"],entry["frames"][item["frame"]]["width"]])
    crop,_,factor=crop_geometry(data["parent"],data["image_hw"],source["search_factor"],source["search_size"])
    np.testing.assert_array_equal(data["crop"],crop)
    if abs(float(data["factor"])-factor)>1e-12 or int(data["layer"])!=own["layer"][i]: raise ValueError("Crop/selector replay changed")
    native_calls(data["full_block_calls"],data["layer"],data["head_forward_calls"][0],data["selector_calls"])
    np.testing.assert_array_equal(data["head_forward_calls"],np.ones(k,int))
    response=float(np.max(np.abs(data["raw"]-data["raw"][:1])))
    if response>plan["design"]["response_atol"]: raise ValueError("Frozen response differs on identical features")
    decode=0.
    for h in range(k):
        dense=dense_boxes(data["size"][h],data["offset"][h],data["parent"],factor,source["search_size"],data["image_hw"])
        ri=int(np.argmax(data["raw"][h])); wi=int(np.argmax(data["raw"][h]*np.asarray(source["window"],np.float32)))
        decode=max(decode,float(np.max(np.abs(dense[wi]-data["boxes"][h]))),float(np.max(np.abs(dense[ri]-data["raw_boxes"][h]))))
    h=HEADS.index(item["history"]); diagonal=float(np.max(np.abs(data["boxes"][h]-own["boxes"][i])))
    if decode>plan["design"]["box_atol"] or diagonal>plan["design"]["box_atol"] or abs(float(data["raw"][h].max())-own["peak"][i])>plan["design"]["peak_atol"]:
        raise ValueError("Own-history/dense decoder replay changed")
    np.testing.assert_array_equal(np.trunc(data["boxes"][h]),np.trunc(own["boxes"][i]))
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
    np.testing.assert_array_equal(data["parents"][0],arrays[item["history"]]["previous"][item["frame"]-1])
    np.testing.assert_array_equal(data["parents"][1:],data["boxes"][:-1])
    first=float(np.max(np.abs(data["boxes"][0]-point["boxes"][HEADS.index(item["evaluator"])])))
    first_maps=max(float(np.max(np.abs(data[k][0]-point[k][HEADS.index(item["evaluator"])]))) for k in ("raw","size","offset"))
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
    if item["history"]==item["evaluator"]:
        own=arrays[item["history"]]; ix=frames-1
        diagonal=float(np.max(np.abs(data["boxes"]-own["boxes"][ix])))
        np.testing.assert_array_equal(np.trunc(data["boxes"]),np.trunc(own["boxes"][ix])); np.testing.assert_array_equal(data["layer"],own["layer"][ix])
    if max(first,decode,diagonal)>plan["design"]["box_atol"]: raise ValueError("Matched first-frame/native diagonal branch changed")
    return dict(maximum_first_frame_error=first,maximum_first_map_error=first_maps,maximum_branch_decode_error=decode,maximum_branch_diagonal_error=diagonal)
