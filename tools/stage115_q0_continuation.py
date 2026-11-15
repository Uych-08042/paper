"""Measure one-frame q0 actions in matched frozen generated-search continuations."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import time
import numpy as np

import stage114_cached_failure_chain as prior
from stage27_uav_common import tracking_iou, failure_events

s,c,io=prior.source,prior.c,prior.io
SCHEMA="stage115_matched_one_q0_action_continuation_v1"
STATUS="COMPLETE_MATCHED_ONE_Q0_ACTION_CONTINUATION"
SOURCE114="cb1665a027fbb6e7db9c5394eb6d14969134c946d3001552ce41c65ff90ae282"
DESIGN=dict(horizon=32,good_iou=.5,low_iou=.2,bootstrap_seed=23150,bootstrap_samples=10000)
ROLES=("first_good_q0_lost","first_medium_q0_lost","first_bad_q0_rescued")
TREATMENTS=("policy","q0_once")
FIELDS=s.FIELDS+("policy_selected","intervention","new_executed")
METRICS=("current_iou","future_mean_iou","future_low_iou_fraction","future_good_candidate_fraction",
         "future_missed_good_fraction","endpoint_iou")
TABLES=("sampling_scope","anchors","scope","branches","contrasts","summary","paired","bridge","branch_frames")


def key(job):return s.job_key(job)


def source_meta(plan,job,models):
    generator,scorer=s.job_models(job,models)
    return dict(plan_fingerprint=plan["fingerprint"],job=job,
        generator_fingerprint=models[job["outer_fold"],"generator",0]["fingerprint"],scorer_fingerprint=scorer["fingerprint"])


def required_trace(root,plan,job,models):
    trace=c.read_cache(root/"evaluation"/(key(job)+".npz"),source_meta(plan,job,models))
    if trace is None:raise ValueError("Missing frozen generated_shared trajectory")
    return trace


def select_anchors(trace,entry,design=DESIGN):
    """Consumed-development GT selects diagnostic roles, never branch actor actions."""
    gt=np.asarray(entry["ground_truth"],float)[1:];n=len(gt)
    valid=np.isfinite(gt).all(1)&(gt[:,2:]>0).all(1)
    bank=tracking_iou(trace["candidate_boxes"],gt[:,None]);q0=bank[:,0];actual=bank[np.arange(n),trace["selected"]]
    changed=np.any(trace["boxes"]!=trace["candidate_boxes"][:,0],axis=1);good,low=design["good_iou"],design["low_iou"]
    masks=(valid&changed&(q0>=good)&(actual<low),valid&changed&(q0>=low)&(q0<good)&(actual<low),valid&changed&(q0<low)&(actual>=good))
    sampling=[];anchors={}
    for role,mask in zip(ROLES,masks):
        first=prior.first(mask);supported=first is not None and first+design["horizon"]-1<=n
        sampling.append(dict(role=role,source_frames=n,matching_frames=int(mask.sum()),first_frame=first,full_horizon_supported=bool(supported)))
        if supported:
            a=anchors.setdefault(first,dict(frame=first,last=first+design["horizon"]-1,roles=[]))
            a["roles"].append(role)
    return sampling,[anchors[f] for f in sorted(anchors)]


def initial_state(trace,entry,first):
    history=[np.asarray(x,float).copy() for x in prior.history_at(trace,entry,first-1)]
    previous={k:v[first-2].copy() for k,v in trace.items()} if first>1 else None
    return dict(history=history,previous=previous)


def observe(observer,image,state,frame,old,generator,scorer,synchronize=lambda:None):
    ap=s.e63.c.actor_plan(old);synchronize();started=time.perf_counter()
    proposal=s.forecast_parents(state["history"],state["previous"],image.shape[:2],ap,"generated_shared",generator)
    row=s.s112.observe(observer,image,state["history"],frame,ap,proposal["candidate_parents"],synchronize)
    scores,selected=s.choose(row,state["history"],"generated_shared",scorer)
    executed=np.array([not any(np.array_equal(p,q) for q in row["candidate_parents"][:i]) for i,p in enumerate(row["candidate_parents"])],bool)
    row.update(history_latest=state["history"][-1].copy(),scores=scores,selected=selected,
        boxes=row["candidate_boxes"][selected].copy(),parents=row["candidate_parents"][selected].copy(),executed=executed)
    row["memory_write"]=s.write_eligible(row);row.update({k:v for k,v in proposal.items() if k!="candidate_parents"})
    synchronize();row["pipeline_ms"]=1000*(time.perf_counter()-started)
    return row


def point(observer,trace,entry,first,old,generator,scorer,synchronize=lambda:None,image_reader=s.s105.checked_image):
    ap=s.e63.c.actor_plan(old);observer.initialize(image_reader(ap["root"],entry["frames"][0]),entry["initial_box"])
    row=observe(observer,image_reader(ap["root"],entry["frames"][first]),initial_state(trace,entry,first),first,old,generator,scorer,synchronize)
    return {k:np.asarray(row[k])[None] for k in s.FIELDS}


def advance(observation,state,treatment,first):
    row={k:np.asarray(v).copy() for k,v in observation.items()};choice=int(row["selected"])
    intervene=treatment=="q0_once" and bool(first)
    row.update(policy_selected=choice,selected=0 if intervene else choice,intervention=bool(intervene),
        new_executed=np.zeros(4,bool) if first else row["executed"].copy())
    selected=int(row["selected"]);row["boxes"]=row["candidate_boxes"][selected].copy();row["parents"]=row["candidate_parents"][selected].copy()
    row["memory_write"]=s.write_eligible(row)
    state["history"]=(state["history"]+[row["boxes"].copy()])[-5:];state["previous"]=row
    return row


def collect(observer,trace,shared,entry,anchor,treatment,old,generator,scorer,synchronize=lambda:None,image_reader=s.s105.checked_image):
    ap=s.e63.c.actor_plan(old);observer.initialize(image_reader(ap["root"],entry["frames"][0]),entry["initial_box"])
    state=initial_state(trace,entry,anchor["frame"]);rows=[]
    for frame in range(anchor["frame"],anchor["last"]+1):
        observation=({k:shared[k][0].copy() for k in s.FIELDS} if frame==anchor["frame"] else
            observe(observer,image_reader(ap["root"],entry["frames"][frame]),state,frame,old,generator,scorer,synchronize))
        row=advance(observation,state,treatment,frame==anchor["frame"]);observer.rt.tracker.state=row["boxes"].tolist();rows.append(row)
    return {k:np.asarray([r[k] for r in rows]) for k in FIELDS}


def audit(data,trace,shared,entry,anchor,treatment,old,generator,scorer):
    ap=s.e63.c.actor_plan(old);first,last=anchor["frame"],anchor["last"];n=last-first+1
    if set(data)!=set(FIELDS) or treatment not in TREATMENTS:raise ValueError("Incomplete/unknown matched branch")
    np.testing.assert_array_equal(data["frame_id"],np.arange(first,last+1));s.e63.source_common.prior.audit_calls(data,(n,4))
    if any(data[k].shape!=shape for k,shape in dict(candidate_boxes=(n,4,4),features=(n,4,24),new_executed=(n,4),policy_selected=(n,),intervention=(n,)).items()):raise ValueError("Branch shapes changed")
    if not np.isfinite(data["decode_error"]).all() or data["decode_error"].max()>ap["design"]["box_atol"]:raise ValueError("Branch decoder changed")
    state=initial_state(trace,entry,first)
    for i,frame in enumerate(data["frame_id"]):
        row={k:v[i] for k,v in data.items()};h=state["history"]
        hw=(entry["frames"][frame]["height"],entry["frames"][frame]["width"])
        np.testing.assert_array_equal(row["image_hw"],hw);np.testing.assert_array_equal(row["history_latest"],h[-1])
        proposal=s.forecast_parents(h,state["previous"],hw,ap,"generated_shared",generator)
        for k,v in proposal.items():np.testing.assert_array_equal(row[k],v)
        for j,parent in enumerate(row["candidate_parents"]):
            crop,usable,factor=s.s105.crop_geometry(parent,hw,ap["search_factor"],ap["search_size"])
            np.testing.assert_array_equal(row["crop"][j],crop)
            view=dict(parent=parent,boxes=row["candidate_boxes"][j],crop=crop,usable=usable,image_hw=hw,**{k:row[k][j] for k in ("raw","size","offset")})
            np.testing.assert_array_equal(row["features"][j],s.e63.actor.feature_vector(view,h,ap))
            dense=s.e63.dense_boxes(view["size"],view["offset"],parent,factor,ap["search_size"],hw)
            cell=int(np.argmax(view["raw"]*np.asarray(ap["window"],np.float32)))
            np.testing.assert_allclose(dense[cell],view["boxes"],rtol=0,atol=ap["design"]["box_atol"])
        scores,chosen=s.choose(row,h,"generated_shared",scorer)
        np.testing.assert_allclose(scores,row["scores"],rtol=1e-9,atol=1e-12)
        observation={k:row[k].copy() for k in s.FIELDS};observation["selected"]=chosen
        expected_mask=np.array([not any(np.array_equal(p,q) for q in row["candidate_parents"][:j]) for j,p in enumerate(row["candidate_parents"])],bool)
        np.testing.assert_array_equal(row["executed"],expected_mask)
        if int(row["unique_search_calls"])!=int(expected_mask.sum()):raise ValueError("Unique search budget changed")
        if i==0:
            for k in s.FIELDS:
                if k not in ("selected","boxes","parents","memory_write"):np.testing.assert_array_equal(row[k],shared[k][0])
        expected=advance(observation,state,treatment,i==0)
        for k in FIELDS:
            np.testing.assert_array_equal(row[k],expected[k])


def bridge(data,trace,anchor,strict):
    errors={};first,last=anchor["frame"]-1,anchor["last"]
    for name in s.FIELDS:
        if name in ("search_ms","pipeline_ms"):continue
        a,b=data[name],trace[name][first:last]
        if a.shape!=b.shape:raise ValueError("Source bridge length changed")
        if strict:np.testing.assert_array_equal(a,b,err_msg="Same-runtime policy control drift: "+name)
        if name in ("boxes","candidate_boxes","features"):errors[name+"_max_error"]=float(np.abs(a-b).max())
    errors["source_action_changes"]=int((data["selected"]!=trace["selected"][first:last]).sum())
    return errors


def metrics(data,entry):
    gt=np.asarray(entry["ground_truth"],float)[data["frame_id"]];bank=tracking_iou(data["candidate_boxes"],gt[:,None]);actual=tracking_iou(data["boxes"],gt)
    valid=np.isfinite(gt).all(1)&(gt[:,2:]>0).all(1)&(actual>=0);future=valid.copy();future[0]=False;best=bank.max(1)
    mean=lambda x:float(x[future].mean()) if future.any() else None
    return dict(current_iou=float(actual[0]) if valid[0] else None,future_valid_frames=int(future.sum()),
        future_mean_iou=mean(actual),future_low_iou_fraction=mean(actual<DESIGN["low_iou"]),
        future_good_candidate_fraction=mean(best>=DESIGN["good_iou"]),future_missed_good_fraction=mean((best>=DESIGN["good_iou"])&(actual<DESIGN["good_iou"])),
        endpoint_iou=float(actual[-1]) if valid[-1] else None,window_failure_events=len(failure_events(actual)))


def aggregate(anchors,branches):
    indexed=defaultdict(dict)
    for row in branches:indexed[row["anchor_id"]][row["treatment"]]=row
    effects=[]
    for a in anchors:
        bank=indexed[a["anchor_id"]]
        if set(bank)!=set(TREATMENTS):raise ValueError("Missing a paired continuation")
        effects.append(dict(anchor_id=a["anchor_id"],cohort=a["cohort"],sequence=a["sequence"],seed=a["seed"],frame=a["frame"],roles="|".join(a["roles"]),
            **{m+"_delta":bank["q0_once"][m]-bank["policy"][m] if all(bank[t][m] is not None for t in TREATMENTS) else None for m in METRICS}))
    summary=[];paired=[]
    for cohort in ("all","historical","expanded"):
        for role in ("all_unique_anchors",*ROLES):
            part=[a for a in anchors if (cohort=="all" or a["cohort"]==cohort) and (role=="all_unique_anchors" or role in a["roles"])]
            if not part:continue
            for treatment in TREATMENTS:
                selected=[indexed[a["anchor_id"]][treatment] for a in part];record=dict(cohort=cohort,role=role,treatment=treatment,anchors=len(part),videos=len({a["sequence"] for a in part}))
                for m in METRICS:
                    units=defaultdict(list)
                    for row in selected:
                        if row[m] is not None:units[row["sequence"],row["seed"]].append(row[m])
                    videos=defaultdict(list)
                    for (seq,seed),values in units.items():videos[seq].append(float(np.mean(values)))
                    record[m]=float(np.mean([np.mean(v) for v in videos.values()])) if videos else None
                summary.append(record)
            ids={a["anchor_id"] for a in part};record=dict(cohort=cohort,role=role,contrast="q0_once_minus_policy",anchors=len(part))
            for m in METRICS:
                units=defaultdict(list)
                for row in effects:
                    if row["anchor_id"] in ids and row[m+"_delta"] is not None:units[row["sequence"],row["seed"]].append(row[m+"_delta"])
                videos=defaultdict(list)
                for (seq,seed),values in units.items():videos[seq].append(float(np.mean(values)))
                values=np.array([np.mean(videos[seq]) for seq in sorted(videos)])
                record[m+"_delta"]=float(values.mean()) if len(values) else None;record[m+"_supported_videos"]=len(values)
                if m=="future_mean_iou" and len(values):
                    draws=np.random.default_rng(DESIGN["bootstrap_seed"]).integers(0,len(values),(DESIGN["bootstrap_samples"],len(values)))
                    boot=values[draws].mean(1)
                    record.update(ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),wins=int((values>1e-12).sum()),
                        ties=int((np.abs(values)<=1e-12).sum()),losses=int((values< -1e-12).sum()))
            paired.append(record)
    return effects,summary,paired


def budget(anchors):
    n=len(anchors);h=DESIGN["horizon"]
    return dict(anchors=n,short_branches=2*n,shared_points=n,branch_frame_exposures=2*n*h,
        unique_search_cap=n*(4+8*(h-1)),minimum_search_calls=n*(1+2*(h-1)),logical_branch_searches=8*n*h,template_initializations=3*n,artifacts=6*n)


def run(root,r113,p113,plan,models,load_entry,old,observer,synchronize=lambda:None,image_reader=s.s105.checked_image):
    evidence={};tables={n:[] for n in TABLES};tables["sampling_scope"]=plan["sampling_scope"];tables["anchors"]=plan["anchors"]
    jobs={key(j):j for j in p113["jobs"] if j["arm"]=="generated_shared"};grouped=defaultdict(list)
    for a in plan["anchors"]:grouped[a["context"]].append(a)
    for context,anchors in grouped.items():
        job=jobs[context];entry=load_entry(job["spec"]);trace=required_trace(r113,p113,job,models);generator,scorer=s.job_models(job,models)
        s.audit(trace,entry,old,"generated_shared",generator,scorer)
        with s.e63.source_experiment.installed(observer,[],job["spec"]["unit"]) as head:
            if head!=plan["native_head_sha256"]:raise ValueError("Native SGLA head changed")
            for a in anchors:
                meta=dict(plan_fingerprint=plan["fingerprint"],anchor=a);path=root/"points"/(a["anchor_id"]+".npz")
                shared=s.s109.cache(root,path,dict(meta,kind="shared_point"),lambda:point(observer,trace,entry,a["frame"],old,generator,scorer,synchronize,image_reader),evidence)
                outputs={}
                for treatment in TREATMENTS:
                    path=root/"branches"/(a["anchor_id"]+"__"+treatment+".npz")
                    data=s.s109.cache(root,path,dict(meta,kind="branch",treatment=treatment),
                        lambda:collect(observer,trace,shared,entry,a,treatment,old,generator,scorer,synchronize,image_reader),evidence)
                    audit(data,trace,shared,entry,a,treatment,old,generator,scorer);outputs[treatment]=data
                # Both branches and receipts are committed before outcome labels.
                for treatment,data in outputs.items():
                    ident=dict(anchor_id=a["anchor_id"],cohort=a["cohort"],sequence=a["sequence"],seed=a["seed"],frame=a["frame"],treatment=treatment)
                    tables["branches"].append(dict(ident,effective_first_action_change=bool(np.any(data["boxes"][0]!=shared["boxes"][0])),**metrics(data,entry)))
                    queries=int((data["head_calls"]*data["new_executed"]).sum());blocks=int((data["block_calls"]*data["new_executed"][...,None]).sum())
                    tables["scope"].append(dict(ident,frames=len(data["frame_id"]),new_search_calls=queries,new_block_calls=blocks,shared_initial_calls=int(shared["unique_search_calls"][0])))
                    overlap=tracking_iou(data["boxes"],np.asarray(entry["ground_truth"])[data["frame_id"]]);bank=tracking_iou(data["candidate_boxes"],np.asarray(entry["ground_truth"])[data["frame_id"],None])
                    for i,f in enumerate(data["frame_id"]):tables["branch_frames"].append(dict(ident,tracked_frame=int(f),selected=int(data["selected"][i]),policy_selected=int(data["policy_selected"][i]),
                        intervention=bool(data["intervention"][i]),iou=float(overlap[i]),best_iou=float(bank[i].max())))
                row=bridge(outputs["policy"],trace,a,not plan["runtime_differences"])
                for field in ("boxes","candidate_parents","crop"):
                    difference=np.any((outputs["policy"][field]!=outputs["q0_once"][field]).reshape(DESIGN["horizon"],-1),axis=1)
                    at=np.flatnonzero(difference)
                    row["first_between_branch_"+field+"_difference"]=int(outputs["policy"]["frame_id"][at[0]]) if len(at) else None
                gt=np.asarray(entry["ground_truth"])[a["frame"]];source_iou=tracking_iou(trace["candidate_boxes"][a["frame"]-1],gt)
                matched_iou=tracking_iou(shared["candidate_boxes"][0],gt)
                q0=float(matched_iou[0]);actual=float(matched_iou[int(shared["selected"][0])]);good,low=DESIGN["good_iou"],DESIGN["low_iou"]
                preserved=[role for role,keep in zip(ROLES,(q0>=good and actual<low,low<=q0<good and actual<low,q0<low and actual>=good)) if keep and role in a["roles"]]
                tables["bridge"].append(dict(anchor_id=a["anchor_id"],cohort=a["cohort"],sequence=a["sequence"],seed=a["seed"],frame=a["frame"],
                    source_selected=int(trace["selected"][a["frame"]-1]),matched_selected=int(shared["selected"][0]),source_q0_iou=float(source_iou[0]),
                    source_policy_iou=float(source_iou[int(trace["selected"][a["frame"]-1])]),matched_q0_iou=float(matched_iou[0]),
                    matched_policy_iou=float(matched_iou[int(shared["selected"][0])]),matched_roles="|".join(preserved),all_roles_preserved=len(preserved)==len(a["roles"]),
                    point_search_calls=int(shared["unique_search_calls"][0]),**row))
                print(f"[Stage-115] {a['anchor_id']}: two own-history continuations saved",flush=True)
            observer.assert_frozen()
    tables["contrasts"],tables["summary"],tables["paired"]=aggregate(plan["anchors"],tables["branches"])
    publish(root,plan,tables,evidence)


def publish(root,plan,tables,evidence):
    expected=budget(plan["anchors"]);n=expected["anchors"];counts={k:len(v) for k,v in tables.items()}
    if not n or set(tables)!=set(TABLES) or counts["branches"]!=2*n or counts["scope"]!=2*n or counts["bridge"]!=n or counts["contrasts"]!=n or counts["branch_frames"]!=expected["branch_frame_exposures"] or len(evidence)!=6*n:raise ValueError("Incomplete paired branches")
    io.verify(plan["source_sha256"]);io.verify(plan["code_sha256"]);out=root/"analysis";out.mkdir(exist_ok=True);hashes={}
    for name,rows in tables.items():
        path=out/f"stage115_{name}.csv"
        # Flatten roles for the report; manifest retains the exact list.
        packed=[dict(r,roles="|".join(r["roles"])) if isinstance(r.get("roles"),list) else r for r in rows]
        s.s103.s86.publish_table(path,packed);hashes[path.name]=c.file_hash(path)
    calls=sum(r["new_search_calls"] for r in tables["scope"])+sum(r["point_search_calls"] for r in tables["bridge"])
    blocks=sum(r["new_block_calls"] for r in tables["scope"])+7*sum(r["point_search_calls"] for r in tables["bridge"])
    if not expected["minimum_search_calls"]<=calls<=expected["unique_search_cap"] or blocks!=7*calls:raise ValueError("New physical search budget differs")
    c.publish_identical(out/"stage115_decision.json",io.seal(dict(schema=SCHEMA,status=STATUS,plan_fingerprint=plan["fingerprint"],source_stage114_fingerprint=SOURCE114,
        runtime=plan["runtime"],runtime_differences=plan["runtime_differences"],budget=expected,counts=counts,new_search_calls=calls,new_block_calls=blocks,
        weights_frozen=True,training_updates=0,new_models=0,new_full_trajectories=0,confirmation_inference_calls=0,table_sha256=hashes,artifact_sha256=evidence,
        warnings=["Anchors use consumed-development GT and fixed roles; this is a conditional diagnostic, not an online gate or confirmation.",
            "Only the first action differs; box, selected response and write qualification feed back consistently. This is not a pure center-only intervention.",
            "Future means exclude the intervention frame. Thirty-two-frame continuations are not full-video AUC or global recovery guarantees.",
            "Both branches share one newly observed first bank; copied observations do not incur a second search charge.",
            "A first L20/L40 GPU change is allowed and bridged; every point/control is newly observed, and role identities are never reselected from new outcomes."])))
    check_results(root)


def check_results(root):
    plan=c.c66.sealed(root/"manifest.json",SCHEMA);r=c.c66.sealed(root/"analysis/stage115_decision.json",SCHEMA);b=budget(plan["anchors"]);n=b["anchors"]
    if (plan["design"]!=DESIGN or r["status"]!=STATUS or r["plan_fingerprint"]!=plan["fingerprint"] or r["source_stage114_fingerprint"]!=SOURCE114
        or r["runtime"]!=plan["runtime"] or r["runtime_differences"]!=plan["runtime_differences"] or r["budget"]!=b or not r["weights_frozen"]
        or r["counts"]["branches"]!=2*n or r["counts"]["bridge"]!=n or r["counts"]["branch_frames"]!=b["branch_frame_exposures"]
        or not b["minimum_search_calls"]<=r["new_search_calls"]<=b["unique_search_cap"] or r["new_block_calls"]!=7*r["new_search_calls"]
        or set(r["table_sha256"])!={f"stage115_{k}.csv" for k in TABLES} or len(r["artifact_sha256"])!=6*n
        or any(r[k]!=0 for k in ("training_updates","new_models","new_full_trajectories","confirmation_inference_calls"))):raise ValueError("Incomplete Stage-115 diagnostic")
    io.verify(plan["source_sha256"]);io.verify(plan["code_sha256"]);io.verify({str(root/"analysis"/k):v for k,v in r["table_sha256"].items()});io.verify({str(root/k):v for k,v in r["artifact_sha256"].items()})


def runtime_binding(expected,actual,previous=None):
    differences=s.s109.check_runtime(expected,actual)
    if previous is not None and previous!=actual:raise ValueError("Stage-115 runtime changed; use a new STAGE115_ROOT")
    return differences


def main():
    parser=argparse.ArgumentParser(description=__doc__);names=("stage76","stage89","stage93","stage112","stage113","stage114")
    for name in (*names,"output_dir"):parser.add_argument("--"+name,type=Path,required=name=="output_dir")
    parser.add_argument("--checkpoint");parser.add_argument("--parameter",default="deit_distilled");parser.add_argument("--gpu",type=int,default=0);parser.add_argument("--check-results",action="store_true");parser.set_defaults(mode="stage115")
    args=parser.parse_args();root=args.output_dir.expanduser().resolve()
    if args.check_results:check_results(root);return
    if not args.checkpoint or not all(getattr(args,n) for n in names):parser.error("Supply original checkpoint and Stage-76/89/93/112/113/114 directories")
    r76,r89,r93,r112,r113,r114=sources=[getattr(args,n).expanduser().resolve() for n in names]
    if any(root==p or root in p.parents or p in root.parents for p in sources):parser.error("Use a separate Stage-115 directory")
    with s.s109.s106.s93.writer_lock(root):
        if (root/"analysis/stage115_decision.json").exists():check_results(root);return
        if not (root/"manifest.json").exists() and any(p.name!="collect.lock" for p in root.iterdir()):raise ValueError("Use an empty Stage-115 root")
        prior.check_results(r114);s.check_results(r113);s.s112.check_results(r112)
        p114=c.c66.sealed(r114/"manifest.json",prior.SCHEMA);r=c.c66.sealed(r114/"analysis/stage114_decision.json",prior.SCHEMA)
        if r["fingerprint"]!=SOURCE114:raise ValueError("Need the frozen Stage-114 diagnostic")
        p113=c.c66.sealed(r113/"manifest.json",s.SCHEMA);p112=c.c66.sealed(r112/"manifest.json",s.common.SCHEMA)
        if c.c66.sealed(r113/"analysis/stage113_decision.json",s.SCHEMA)["fingerprint"]!=prior.SOURCE113:raise ValueError("Frozen Stage-113 source differs")
        p93,_,p89,r89report,origin=s.s103.prior.s95.s94.load_run(r93,r76,r89);models,_=s.load_frozen_models(r112,p112)
        def load_entry(spec):return s.s103.prior.s95.load_unit(r93,r89,p93,p89,r89report,origin,spec)[0]
        sampling=[];anchors=[]
        for job in p113["jobs"]:
            if job["arm"]!="generated_shared":continue
            entry=load_entry(job["spec"]);trace=required_trace(r113,p113,job,models);generator,scorer=s.job_models(job,models)
            s.audit(trace,entry,origin["original"],"generated_shared",generator,scorer)
            rows,found=select_anchors(trace,entry);ident=dict(context=key(job),cohort=job["spec"]["cohort"],sequence=job["spec"]["unit"]["sequence"],seed=job["seed"])
            sampling.extend(dict(ident,**row) for row in rows)
            anchors.extend(dict(ident,**a,anchor_id=key(job)+f"__f{a['frame']:06d}") for a in found)
        if len(sampling)!=168 or not anchors:raise ValueError("Incomplete 56-context diagnostic population")
        import torch
        torch.set_num_threads(1);runtime=s.s103.s76.device_runtime(origin["original"],torch,args.gpu)
        previous=c.c66.sealed(root/"manifest.json",SCHEMA) if (root/"manifest.json").exists() else None
        differences=runtime_binding(p113["runtime"],runtime,previous["runtime"] if previous else None)
        bindings=dict(p114["source_sha256"])
        for q in (r114/"manifest.json",r114/"analysis/stage114_decision.json",r113/"manifest.json",r112/"manifest.json"):bindings[str(q)]=c.file_hash(q)
        bindings.update({str(r114/"analysis"/k):v for k,v in r["table_sha256"].items()});bindings.update({str(r114/k):v for k,v in r["artifact_sha256"].items()})
        code=[Path(__file__).resolve(),Path(__file__).with_name("run_stage115_q0_continuation.sh").resolve()]
        plan=io.seal(dict(schema=SCHEMA,design=DESIGN,sampling_scope=sampling,anchors=anchors,source_stage114_fingerprint=SOURCE114,runtime=runtime,runtime_differences=differences,
            native_head_sha256=origin["report"]["native_head_sha256"],source_sha256=bindings,code_sha256=p114["code_sha256"]|{str(q):c.file_hash(q) for q in code}))
        c.publish_identical(root/"manifest.json",plan);print("Stage-115 frozen before inference:",budget(anchors),flush=True)
        frozen=dict(origin["original"],fingerprint=plan["fingerprint"],runtime_identity=runtime,runtime=runtime["tracker"])
        native,_=s.s103.s76.startup.checked_runtime(args,frozen,root);observer=s.e63.Capture(native)
        try:run(root,r113,p113,plan,models,load_entry,origin["original"],observer,lambda:torch.cuda.synchronize(args.gpu))
        finally:observer.handle.remove()


if __name__=="__main__":main()
