"""Sample q0 interventions without outcome labels and measure actual matched futures."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import numpy as np

import stage116_online_evidence as evidence

prior,s,c,io=evidence.prior,evidence.s,evidence.c,evidence.io
SCHEMA="stage117_gt_free_sampled_q0_continuation_v1"
STATUS="COMPLETE_GT_FREE_SAMPLED_Q0_CONTINUATION"
SOURCE116="e15637be797024fd188b10dd2f02783ba6b51f381b5e9e81c35b687ea71d2075"
DESIGN=dict(horizon=32,first_frame=5,uniform_per_context=8,stratum_per_context=1,sampler_seed=23170,
    bootstrap_seed=23171,bootstrap_samples=10000,epsilon=1e-12,
    primary_study="uniform",primary="forecast_pair_minus_continuity_pair")
STUDIES=("uniform","stratified")
RULES=evidence.RULES+("forecast_pair",)
CANDIDATES=RULES[2:]
POLICIES=RULES+tuple(r+"__rate_expected" for r in CANDIDATES)+("future_oracle",)
TABLES=("sampling_scope","anchors","scope","branches","samples","sample_decisions","support",
        "policy_summary","paired","bridge","branch_frames")


def seed_for(context,stream):
    return int.from_bytes(hashlib.sha256(f"{DESIGN['sampler_seed']}:{context}:{stream}".encode()).digest()[:16],"big")


def sample_frames(frame_ids,physical,x,excluded,context):
    """Sampling whitelist; no GT, quality tags, outcomes, or fitted gate are accepted."""
    frames=np.asarray(frame_ids);mask=np.asarray(physical);features=np.asarray(x,float);n=len(frames)
    np.testing.assert_array_equal(frames,np.arange(1,n+1))
    if mask.shape!=(n,) or not np.isin(mask,(0,1)).all() or features.shape!=(n,len(evidence.FEATURES)) or not np.isfinite(features).all():raise ValueError("Invalid public sampling exposure")
    eligible=frames[mask.astype(bool)&(frames>=DESIGN["first_frame"])&(frames+DESIGN["horizon"]-1<=n)]
    excluded=sorted(set(int(f) for f in excluded));pool=sorted(set(eligible.tolist())-set(excluded))
    indices=[evidence.FEATURES.index(k) for k in ("hann_advantage","continuity_advantage","forecast_advantage")]
    strata=((features[:,indices]>DESIGN["epsilon"])*np.array([1,2,4])).sum(1)
    rows=[];chosen={}
    for study,stratum,values,quota in [("uniform",-1,pool,DESIGN["uniform_per_context"])]+[
            ("stratified",j,[f for f in pool if strata[f-1]==j],DESIGN["stratum_per_context"]) for j in range(8)]:
        seed=seed_for(context,f"{study}:{stratum}");count=min(quota,len(values))
        selected=sorted(int(f) for f in np.random.Generator(np.random.PCG64(seed)).choice(values,size=count,replace=False)) if count else []
        mass=(len(values)/len(pool) if pool else 0.) if study=="stratified" else float(bool(pool))
        rows.append(dict(context=context,study=study,stratum=stratum,source_frames=n,full_horizon_physical=len(eligible),
            removed_known=len(set(eligible.tolist())&set(excluded)),eligible_pool=len(pool),excluded_ids=excluded,
            stratum_pool=len(values),draws=count,selected=selected,sampler_seed=str(seed),
            inclusion_probability=count/len(values) if values else 0.,population_mass=mass))
        for f in selected:
            a=chosen.setdefault(f,dict(frame=f,last=f+DESIGN["horizon"]-1,stratum=int(strata[f-1]),memberships=[],uniform_weight=0.,stratified_weight=0.))
            a["memberships"].append(study);a[study+"_weight"]=mass/count
    return rows,[chosen[f] for f in sorted(chosen)]


def required_exposure(root,p116,job):
    data=c.read_cache(root/"audit"/(prior.key(job)+".npz"),dict(plan_fingerprint=p116["fingerprint"],job=job))
    if data is None:raise ValueError("Missing complete Stage-116 sampling exposure")
    return {k:data[k] for k in ("frame_id","physical","x")}


def sampling(jobs,load_exposure,known):
    rows=[];anchors=[]
    for job in jobs:
        context=prior.key(job);sequence=job["spec"]["unit"]["sequence"]
        # Exclude only known start IDs, across both scorer seeds for the same video.
        excluded=[a["frame"] for a in known if a["sequence"]==sequence]
        public=load_exposure(job);support,found=sample_frames(public["frame_id"],public["physical"],public["x"],excluded,context)
        ident=dict(context=context,cohort=job["spec"]["cohort"],sequence=sequence,seed=job["seed"],outer_fold=job["outer_fold"])
        rows.extend(dict(ident,**r) for r in support)
        anchors.extend(dict(ident,**a,anchor_id=context+f"__f{a['frame']:06d}") for a in found)
    return rows,anchors


def budget(anchors):
    n=len(anchors);h=DESIGN["horizon"]
    return dict(anchors=n,shared_points=n,short_branches=2*n,branch_frame_exposures=2*n*h,
        maximum_search_calls=n*(4+8*(h-1)),minimum_search_calls=n*(1+2*(h-1)),logical_branch_searches=8*n*h,
        template_initializations=3*n,artifacts=6*n)


def runtime_binding(expected,actual,previous=None):
    changed=s.s109.check_runtime(expected,actual)
    if previous is not None and previous!=actual:raise ValueError("Stage-117 runtime changed; use a new STAGE117_ROOT")
    return changed


def decisions(e):
    return np.r_[evidence.decisions(e,True),e["forecast_advantage"]>DESIGN["epsilon"]].astype(bool)


def current_role(q0,policy):
    if q0>=.5 and policy<.2:return "good_q0_lost"
    if .2<=q0<.5 and policy<.2:return "medium_q0_lost"
    if q0<.2 and policy>=.5:return "q0_rescue"
    return "other"


def measured_rows(anchor,shared,trace,scorer,outputs,entry):
    row={k:v[0] for k,v in shared.items()};history=prior.initial_state(trace,entry,anchor["frame"])["history"]
    public=evidence.online_evidence(row,history,scorer)
    physical=bool(np.any(row["boxes"]!=row["candidate_boxes"][0]));flags=decisions(public)&physical
    # Rule decisions use public observations before evaluating either measured future.
    p,q=(prior.metrics(outputs[t],entry) for t in prior.TREATMENTS)
    if any(z["future_mean_iou"] is None for z in (p,q)):raise ValueError("No valid paired future; retain unsupported identity and inspect source")
    record=dict(anchor_id=anchor["anchor_id"],context=anchor["context"],cohort=anchor["cohort"],sequence=anchor["sequence"],seed=anchor["seed"],frame=anchor["frame"],
        memberships="|".join(anchor["memberships"]),stratum=anchor["stratum"],uniform_weight=anchor["uniform_weight"],stratified_weight=anchor["stratified_weight"],
        effective_intervention=physical,**public,policy_future=p["future_mean_iou"],q0_future=q["future_mean_iou"],
        policy_current=p["current_iou"],q0_current=q["current_iou"],policy_low=p["future_low_iou_fraction"],q0_low=q["future_low_iou_fraction"],
        policy_endpoint=p["endpoint_iou"],q0_endpoint=q["endpoint_iou"],future_gain=q["future_mean_iou"]-p["future_mean_iou"],
        current_role=current_role(q["current_iou"],p["current_iou"]),outcome=evidence.outcome(q["future_mean_iou"],p["future_mean_iou"]))
    return record,[dict(record,rule=rule,protect_probability=float(flags[j])) for j,rule in enumerate(RULES)]


def weighted_macro(rows,name):
    units=defaultdict(list)
    for r in rows:units[r["sequence"],r["seed"]].append(r)
    videos=defaultdict(list)
    for (sequence,_),values in units.items():
        weight=np.array([r["analysis_weight"] for r in values]);weight=weight/weight.sum()
        videos[sequence].append(float(weight@np.array([r[name] for r in values])))
    return {sequence:float(np.mean(values)) for sequence,values in videos.items()}


def summaries(samples,choices):
    all_rows=[];summary=[];pairs=[];support=[]
    for study in STUDIES:
        local=[dict(r,study=study,analysis_weight=r[study+"_weight"]) for r in choices if study in r["memberships"].split("|")]
        for rule in CANDIDATES:
            units=defaultdict(list)
            for r in local:
                if r["rule"]==rule:units[r["sequence"],r["seed"]].append(r)
            for rows in units.values():
                probability=sum(r["analysis_weight"]*r["protect_probability"] for r in rows)/sum(r["analysis_weight"] for r in rows)
                local.extend(dict(r,rule=rule+"__rate_expected",protect_probability=probability) for r in rows)
        local.extend(dict(r,rule="future_oracle",protect_probability=float(r["future_gain"]>DESIGN["epsilon"])) for r in list(local) if r["rule"]=="policy")
        for r in local:
            a=r["protect_probability"];r["future_iou"]=r["policy_future"]+a*r["future_gain"]
            r["low_fraction"]=r["policy_low"]+a*(r["q0_low"]-r["policy_low"])
            r["realized_gain"]=a*r["future_gain"];r["regret"]=max(r["future_gain"],0.)-a*r["future_gain"]
        all_rows.extend(local)
        for cohort in ("all","historical","expanded"):
            rows=[r for r in local if cohort=="all" or r["cohort"]==cohort]
            if not rows:continue
            base=[r for r in rows if r["rule"]=="policy"]
            for label in ("all","good_q0_lost","medium_q0_lost","q0_rescue","other"):
                subset=[r for r in base if label=="all" or r["current_role"]==label]
                support.append(dict(study=study,cohort=cohort,current_role=label,anchors=len(subset),contexts=len({r["context"] for r in subset}),videos=len({r["sequence"] for r in subset}),
                    positive=sum(r["future_gain"]>1e-12 for r in subset),ties=sum(abs(r["future_gain"])<=1e-12 for r in subset),negative=sum(r["future_gain"]< -1e-12 for r in subset),
                    both_low=sum(r["outcome"]=="both_low" for r in subset)))
            for rule in POLICIES:
                part=[r for r in rows if r["rule"]==rule]
                means={m:weighted_macro(part,m) for m in ("future_iou","low_fraction","protect_probability","realized_gain","regret")}
                summary.append(dict(study=study,cohort=cohort,rule=rule,anchors=len(part),videos=len(means["future_iou"]),**{m:float(np.mean(list(v.values()))) for m,v in means.items()}))
            for rule in CANDIDATES:
                treatment=weighted_macro([r for r in rows if r["rule"]==rule],"future_iou")
                informative=[r for r in rows if r["rule"]==rule+"__rate_expected" and 0<r["protect_probability"]<1]
                for control in ("policy","q0_all","continuity_pair",rule+"__rate_expected"):
                    if rule==control:continue
                    reference=weighted_macro([r for r in rows if r["rule"]==control],"future_iou");groups=sorted(treatment.keys()&reference.keys())
                    delta=np.array([treatment[g]-reference[g] for g in groups]);rng=np.random.default_rng(DESIGN["bootstrap_seed"])
                    boot=delta[rng.integers(0,len(delta),(DESIGN["bootstrap_samples"],len(delta)))].mean(1)
                    pairs.append(dict(study=study,cohort=cohort,contrast=rule+"_minus_"+control,videos=len(delta),mean_delta=float(delta.mean()),
                        ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),wins=int((delta>1e-12).sum()),ties=int((np.abs(delta)<=1e-12).sum()),losses=int((delta< -1e-12).sum()),
                        rate_informative_contexts=len({r["context"] for r in informative}),rate_informative_anchors=len(informative)))
    return all_rows,summary,pairs,support


def run(root,r113,p113,plan,models,load_entry,old,observer,synchronize=lambda:None,image_reader=s.s105.checked_image):
    tables={n:[] for n in TABLES};tables["sampling_scope"]=plan["sampling_scope"];tables["anchors"]=plan["anchors"];artifacts={}
    jobs={prior.key(j):j for j in p113["jobs"] if j["arm"]=="generated_shared"};by_context=defaultdict(list)
    for a in plan["anchors"]:by_context[a["context"]].append(a)
    for context,anchors in by_context.items():
        job=jobs[context];entry=load_entry(job["spec"]);trace=prior.required_trace(r113,p113,job,models);generator,scorer=s.job_models(job,models)
        s.audit(trace,entry,old,"generated_shared",generator,scorer)
        with s.e63.source_experiment.installed(observer,[],job["spec"]["unit"]) as head:
            if head!=plan["native_head_sha256"]:raise ValueError("Native head changed")
            for a in anchors:
                meta=dict(plan_fingerprint=plan["fingerprint"],anchor=a)
                shared=s.s109.cache(root,root/"points"/(a["anchor_id"]+".npz"),dict(meta,kind="shared_point"),
                    lambda:prior.point(observer,trace,entry,a["frame"],old,generator,scorer,synchronize,image_reader),artifacts)
                outputs={}
                for t in prior.TREATMENTS:
                    data=s.s109.cache(root,root/"branches"/(a["anchor_id"]+"__"+t+".npz"),dict(meta,kind="branch",treatment=t),
                        lambda:prior.collect(observer,trace,shared,entry,a,t,old,generator,scorer,synchronize,image_reader),artifacts)
                    prior.audit(data,trace,shared,entry,a,t,old,generator,scorer);outputs[t]=data
                # Both measured futures/receipts exist before labels or rule utility.
                record,actions=measured_rows(a,shared,trace,scorer,outputs,entry)
                tables["samples"].append(record);tables["sample_decisions"].extend(actions)
                for t,data in outputs.items():
                    ident={k:a[k] for k in ("anchor_id","context","cohort","sequence","seed","frame")};ident["treatment"]=t
                    tables["branches"].append(dict(ident,**prior.metrics(data,entry)))
                    calls=int((data["head_calls"]*data["new_executed"]).sum());blocks=int((data["block_calls"]*data["new_executed"][...,None]).sum())
                    tables["scope"].append(dict(ident,frames=len(data["frame_id"]),new_search_calls=calls,new_block_calls=blocks))
                    gt=np.asarray(entry["ground_truth"])[data["frame_id"]]
                    actual=prior.tracking_iou(data["boxes"],gt);best=prior.tracking_iou(data["candidate_boxes"],gt[:,None]).max(1)
                    tables["branch_frames"].extend(dict(ident,tracked_frame=int(f),selected=int(data["selected"][i]),policy_selected=int(data["policy_selected"][i]),
                        intervention=bool(data["intervention"][i]),iou=float(actual[i]),best_iou=float(best[i])) for i,f in enumerate(data["frame_id"]))
                bridge=prior.bridge(outputs["policy"],trace,a,not plan["runtime_differences"])
                for field in ("boxes","candidate_parents","crop"):
                    diff=np.any((outputs["policy"][field]!=outputs["q0_once"][field]).reshape(DESIGN["horizon"],-1),axis=1);at=np.flatnonzero(diff)
                    bridge["first_"+field+"_difference"]=int(outputs["policy"]["frame_id"][at[0]]) if len(at) else None
                source_x=plan["source_anchor_evidence"][a["anchor_id"]]
                tables["bridge"].append(dict(anchor_id=a["anchor_id"],point_search_calls=int(shared["unique_search_calls"][0]),
                    source_stratum=a["stratum"],matched_stratum=int(sum((record[k]>DESIGN["epsilon"])*v for k,v in zip(
                        ("hann_advantage","continuity_advantage","forecast_advantage"),(1,2,4)))),
                    public_evidence_max_error=float(np.max(np.abs(np.array([record[k] for k in evidence.FEATURES])-source_x))),
                    effective_intervention=record["effective_intervention"],**bridge))
                print("[Stage-117] "+a["anchor_id"]+": two matched futures saved",flush=True)
            observer.assert_frozen()
    tables["sample_decisions"],tables["policy_summary"],tables["paired"],tables["support"]=summaries(tables["samples"],tables["sample_decisions"])
    publish(root,plan,tables,artifacts)


def publish(root,plan,tables,artifacts):
    b=budget(plan["anchors"]);n=b["anchors"];counts={k:len(v) for k,v in tables.items()}
    memberships=sum(len(a["memberships"]) for a in plan["anchors"])
    if (not n or counts["samples"]!=n or counts["bridge"]!=n or counts["branches"]!=2*n or counts["scope"]!=2*n
        or counts["branch_frames"]!=b["branch_frame_exposures"] or counts["sample_decisions"]!=memberships*len(POLICIES) or len(artifacts)!=6*n):raise ValueError("Incomplete sampled paired futures")
    out=root/"analysis";out.mkdir(exist_ok=True);hashes={}
    for name,rows in tables.items():
        packed=[{k:("|".join(str(x) for x in v) if isinstance(v,list) else v) for k,v in row.items()} for row in rows]
        path=out/f"stage117_{name}.csv";s.s103.s86.publish_table(path,packed);hashes[path.name]=c.file_hash(path)
    calls=sum(r["new_search_calls"] for r in tables["scope"])+sum(r["point_search_calls"] for r in tables["bridge"])
    blocks=sum(r["new_block_calls"] for r in tables["scope"])+7*sum(r["point_search_calls"] for r in tables["bridge"])
    if not b["minimum_search_calls"]<=calls<=b["maximum_search_calls"] or blocks!=7*calls:raise ValueError("Sampled call ledger changed")
    primary=next(r for r in tables["paired"] if r["study"]==DESIGN["primary_study"] and r["cohort"]=="all" and r["contrast"]==DESIGN["primary"])
    io.verify(plan["source_sha256"]);io.verify(plan["code_sha256"])
    c.publish_identical(out/"stage117_decision.json",io.seal(dict(schema=SCHEMA,status=STATUS,plan_fingerprint=plan["fingerprint"],source_stage116_fingerprint=SOURCE116,
        runtime=plan["runtime"],runtime_differences=plan["runtime_differences"],budget=b,counts=counts,membership_exposures=memberships,primary=primary,
        new_search_calls=calls,new_block_calls=blocks,weights_frozen=True,training_updates=0,new_models=0,new_full_trajectories=0,confirmation_inference_calls=0,
        table_sha256=hashes,artifact_sha256=artifacts,warnings=[
            "GT-free sampling of remaining physical replacements; excludes known Stage-115 start IDs, not all overlapping windows.",
            "Uniform and evidence-stratified studies are separate. Stratified estimates use source population masses; no pooled headline.",
            "This is consumed development under frozen upstream models, not fresh source-group confirmation or a fitted gate.",
            "Only one action differs; means exclude its frame. No complete-video AUC or continuously deployed rule is measured.",
            "Labels cannot guarantee sampled rescue support; report all current-role and outcome counts without reselecting.",
            "A new GPU may change public evidence or make an intervention a no-op; retain every presealed anchor and bridge it.",
            "Rate controls match expected population-weighted protection within video/seed; a single-anchor context has no selection contrast."])))
    check_results(root)


def check_results(root):
    plan=c.c66.sealed(root/"manifest.json",SCHEMA);r=c.c66.sealed(root/"analysis/stage117_decision.json",SCHEMA);b=budget(plan["anchors"]);n=b["anchors"]
    if (plan["design"]!=DESIGN or plan["source_stage116_fingerprint"]!=SOURCE116 or r["status"]!=STATUS or r["plan_fingerprint"]!=plan["fingerprint"] or r["source_stage116_fingerprint"]!=SOURCE116
        or r["runtime"]!=plan["runtime"] or r["runtime_differences"]!=plan["runtime_differences"] or r["budget"]!=b
        or r["counts"]["samples"]!=n or r["counts"]["anchors"]!=n or r["counts"]["sampling_scope"]!=len(plan["sampling_scope"])
        or r["counts"]["bridge"]!=n or r["counts"]["branches"]!=2*n or r["counts"]["scope"]!=2*n or r["counts"]["branch_frames"]!=b["branch_frame_exposures"]
        or r["membership_exposures"]!=sum(len(a["memberships"]) for a in plan["anchors"])
        or r["primary"]["study"]!=DESIGN["primary_study"] or r["primary"]["cohort"]!="all" or r["primary"]["contrast"]!=DESIGN["primary"]
        or r["counts"]["sample_decisions"]!=r["membership_exposures"]*len(POLICIES) or len(r["artifact_sha256"])!=6*n or not r["weights_frozen"]
        or not b["minimum_search_calls"]<=r["new_search_calls"]<=b["maximum_search_calls"] or r["new_block_calls"]!=7*r["new_search_calls"]
        or set(r["table_sha256"])!={f"stage117_{k}.csv" for k in TABLES}
        or any(r[k]!=0 for k in ("training_updates","new_models","new_full_trajectories","confirmation_inference_calls"))):raise ValueError("Incomplete Stage-117 result")
    io.verify(plan["source_sha256"]);io.verify(plan["code_sha256"])
    io.verify({str(root/"analysis"/k):v for k,v in r["table_sha256"].items()});io.verify({str(root/k):v for k,v in r["artifact_sha256"].items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__);names=("stage76","stage89","stage93","stage112","stage113","stage115","stage116")
    for name in (*names,"output_dir"):parser.add_argument("--"+name,type=Path,required=name=="output_dir")
    parser.add_argument("--checkpoint");parser.add_argument("--parameter",default="deit_distilled");parser.add_argument("--gpu",type=int,default=0);parser.add_argument("--check-results",action="store_true");parser.set_defaults(mode="stage117")
    args=parser.parse_args();root=args.output_dir.expanduser().resolve()
    if args.check_results:check_results(root);return
    if not args.checkpoint or not all(getattr(args,n) for n in names):parser.error("Supply original checkpoint and Stage-76/89/93/112/113/115/116 roots")
    r76,r89,r93,r112,r113,r115,r116=paths=[getattr(args,n).expanduser().resolve() for n in names]
    if any(root==p or root in p.parents or p in root.parents for p in paths):parser.error("Use a separate Stage-117 root")
    with s.s109.s106.s93.writer_lock(root):
        if (root/"analysis/stage117_decision.json").exists():check_results(root);return
        if not (root/"manifest.json").exists() and any(p.name!="collect.lock" for p in root.iterdir()):raise ValueError("Use an empty Stage-117 root")
        evidence.check_results(r116);prior.check_results(r115);s.check_results(r113);s.s112.check_results(r112)
        p116=c.c66.sealed(r116/"manifest.json",evidence.SCHEMA);r116report=c.c66.sealed(r116/"analysis/stage116_decision.json",evidence.SCHEMA)
        p115=c.c66.sealed(r115/"manifest.json",prior.SCHEMA);p113=c.c66.sealed(r113/"manifest.json",s.SCHEMA);p112=c.c66.sealed(r112/"manifest.json",s.common.SCHEMA)
        for path,schema,expected in ((r116/"analysis/stage116_decision.json",evidence.SCHEMA,SOURCE116),(r115/"analysis/stage115_decision.json",prior.SCHEMA,evidence.SOURCE115),
                (r113/"analysis/stage113_decision.json",s.SCHEMA,prior.prior.SOURCE113),(r112/"analysis/stage112_decision.json",s.common.SCHEMA,s.SOURCE112)):
            if c.c66.sealed(path,schema)["fingerprint"]!=expected:raise ValueError("Need the frozen Stage-112/113/115/116 sources")
        jobs=[j for j in p113["jobs"] if j["arm"]=="generated_shared"]
        support,anchors=sampling(jobs,lambda j:required_exposure(r116,p116,j),p115["anchors"])
        if len(jobs)!=56 or len(support)!=504 or not 0<len(anchors)<=896:raise ValueError("Frozen sampled population differs")
        source_x={}
        for job in jobs:
            public=required_exposure(r116,p116,job)
            for a in anchors:
                if a["context"]==prior.key(job):source_x[a["anchor_id"]]=public["x"][a["frame"]-1].tolist()
        p93,_,p89,r89report,origin=s.s103.prior.s95.s94.load_run(r93,r76,r89);models,_=s.load_frozen_models(r112,p112)
        import torch
        torch.set_num_threads(1);runtime=s.s103.s76.device_runtime(origin["original"],torch,args.gpu)
        previous=c.c66.sealed(root/"manifest.json",SCHEMA) if (root/"manifest.json").exists() else None
        changed=runtime_binding(p113["runtime"],runtime,previous["runtime"] if previous else None)
        bindings=dict(p116["source_sha256"])
        for p in (r116/"manifest.json",r116/"analysis/stage116_decision.json"):bindings[str(p)]=c.file_hash(p)
        bindings.update({str(r116/k):v for k,v in r116report["artifact_sha256"].items()});bindings.update({str(r116/"analysis"/k):v for k,v in r116report["table_sha256"].items()})
        code=[Path(__file__).resolve(),Path(__file__).with_name("run_stage117_sampled_q0.sh").resolve()]
        plan=io.seal(dict(schema=SCHEMA,design=DESIGN,sampling_scope=support,anchors=anchors,source_anchor_evidence=source_x,
            source_stage116_fingerprint=SOURCE116,runtime=runtime,runtime_differences=changed,native_head_sha256=origin["report"]["native_head_sha256"],
            source_sha256=bindings,code_sha256=p116["code_sha256"]|{str(p):c.file_hash(p) for p in code}))
        c.publish_identical(root/"manifest.json",plan);print("Stage-117 frozen before inference:",budget(anchors),flush=True)
        frozen=dict(origin["original"],fingerprint=plan["fingerprint"],runtime_identity=runtime,runtime=runtime["tracker"])
        native,_=s.s103.s76.startup.checked_runtime(args,frozen,root);observer=s.e63.Capture(native)
        def load_entry(spec):return s.s103.prior.s95.load_unit(r93,r89,p93,p89,r89report,origin,spec)[0]
        try:run(root,r113,p113,plan,models,load_entry,origin["original"],observer,lambda:torch.cuda.synchronize(args.gpu))
        finally:observer.handle.remove()


if __name__=="__main__":main()
