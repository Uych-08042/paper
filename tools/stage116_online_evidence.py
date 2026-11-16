"""Audit causal q0 evidence against frozen current outputs and measured paired futures."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np

import stage115_q0_continuation as prior
from stage27_uav_common import tracking_iou

s,c,io=prior.s,prior.c,prior.io
SCHEMA="stage116_cached_online_q0_evidence_v1"
STATUS="COMPLETE_CACHED_ONLINE_Q0_EVIDENCE"
SOURCE115="141baa4bae77280cf89cf063665b8b4fa03545ba3ae3c98e52ce0137f0b90edc"
DESIGN=dict(epsilon=1e-12,low_iou=.2,good_iou=.5,bootstrap_seed=23160,bootstrap_samples=10000)
EVIDENCE=("hann_advantage","raw_advantage","psr_advantage","entropy_advantage","continuity_advantage",
          "size_stability_advantage","forecast_advantage","agreement_advantage","score_advantage")
CONTEXT=("center_conflict","log_size_conflict","radius_max","scorer_clip_fraction")
FEATURES=EVIDENCE+CONTEXT
RULES=("policy","q0_all","hann_pair","continuity_pair","corroborated")
POLICIES=RULES+tuple(r+"__rate_expected" for r in RULES[2:])+("future_oracle",)
OUTCOMES=("q0_better","policy_better","both_low","tie")
TABLES=("population_scope","population_summary","anchor_evidence","evidence_summary","discrimination",
        "anchor_decisions","policy_summary","paired","video_effects")


def online_evidence(observation,history,scorer):
    """Explicit whitelist: current four observations, prior boxes and frozen norm only."""
    row={k:np.asarray(observation[k]) for k in ("candidate_boxes","features","scores","selected",
        "forecast_mean","conditional_radius","generated_active")}
    boxes,base,h=np.asarray(row["candidate_boxes"],float),np.asarray(row["features"],float),np.asarray(history,float)
    j=int(row["selected"])
    if (boxes.shape!=(4,4) or base.shape!=(4,24) or row["scores"].shape!=(4,) or h.ndim!=2 or h.shape[1]!=4
        or not 1<=len(h)<=5 or j not in range(4) or (boxes[:,2:]<=0).any() or (h[:,2:]<=0).any()
        or not all(np.isfinite(x).all() for x in (boxes,base,h,*row.values()))):raise ValueError("Invalid public observations")
    names=s.e63.source_common.prior.FEATURES;f=lambda name:base[0,names.index(name)]-base[j,names.index(name)]
    centers=boxes[:,:2]+boxes[:,2:]/2;hc=h[:,:2]+h[:,2:]/2;unit=np.sqrt(h[-1,2:].prod())
    predicted=hc[-1]+(hc[-1]-hc[-2] if len(h)>1 else 0.)
    distance=np.linalg.norm(centers-predicted,axis=1)/unit
    size=np.abs(np.log(boxes[:,2:])-np.median(np.log(h[:,2:]),axis=0)).mean(1)
    overlap=s.common.prior.c74.c66.c64.continuous_iou
    # Exclude exact duplicate public boxes from support to avoid repeated votes.
    unique=[]
    for k,b in enumerate(boxes):
        if not any(np.array_equal(b,boxes[v]) for v in unique):unique.append(k)
    support=np.array([np.mean([overlap(b,boxes[v:v+1])[0] for v in unique if not np.array_equal(b,boxes[v])])
                      if any(not np.array_equal(b,boxes[v]) for v in unique) else 0. for b in boxes])
    radius=np.asarray(row["conditional_radius"],float);active=bool(row["generated_active"])
    if radius.shape!=(4,) or row["forecast_mean"].shape!=(4,) or (active and (radius<=0).any()):raise ValueError("Invalid forecast radius")
    compatible=np.array([np.max(np.abs(s.p.target(b,h[-1])-row["forecast_mean"])/np.maximum(radius,1e-12)) for b in boxes]) if active else np.zeros(4)
    x=s.common.features(base,boxes,h);z=(x-np.asarray(scorer["norm"]["center"]))/np.asarray(scorer["norm"]["scale"])
    result=dict(hann_advantage=f("hann_peak"),raw_advantage=f("raw_peak"),psr_advantage=f("psr"),entropy_advantage=-f("entropy"),
        continuity_advantage=distance[j]-distance[0],size_stability_advantage=size[j]-size[0],
        forecast_advantage=compatible[j]-compatible[0],agreement_advantage=support[0]-support[j],score_advantage=float(row["scores"][0]-row["scores"][j]),
        center_conflict=float(np.linalg.norm(centers[j]-centers[0])/unit),log_size_conflict=float(np.abs(np.log(boxes[j,2:]/boxes[0,2:])).mean()),
        radius_max=float(radius.max()) if active else 0.,scorer_clip_fraction=float((np.abs(z)>scorer["design"]["feature_clip"]).mean()))
    if set(result)!=set(FEATURES) or not np.isfinite(list(result.values())).all():raise ValueError("Nonfinite public evidence")
    return {k:float(result[k]) for k in FEATURES}


def decisions(evidence,physical):
    h=evidence["hann_advantage"]>DESIGN["epsilon"];m=evidence["continuity_advantage"]>DESIGN["epsilon"]
    return np.array([False,physical,physical and h,physical and m,
                     physical and h and m and evidence["raw_advantage"]>DESIGN["epsilon"]],bool)


def diagnose(trace,entry,scorer):
    """Compute decisions first; labels never enter online_evidence or decisions."""
    n=len(trace["frame_id"]);x=[];actions=[];physical=[]
    for i in range(n):
        row={k:v[i] for k,v in trace.items()};history=prior.prior.history_at(trace,entry,i)
        e=online_evidence(row,history,scorer);changed=bool(np.any(row["boxes"]!=row["candidate_boxes"][0]))
        x.append([e[k] for k in FEATURES]);actions.append(decisions(e,changed));physical.append(changed)
    truth=np.asarray(entry["ground_truth"],float)[1:];bank=tracking_iou(trace["candidate_boxes"],truth[:,None])
    valid=np.isfinite(truth).all(1)&(truth[:,2:]>0).all(1);actual=bank[np.arange(n),trace["selected"]]
    return dict(frame_id=trace["frame_id"].copy(),x=np.asarray(x),actions=np.asarray(actions),physical=np.asarray(physical),
                valid=valid,q0_iou=bank[:,0],policy_iou=actual,best_iou=bank.max(1))


def macro(rows,name):
    units=defaultdict(list)
    for r in rows:
        if r.get(name) is not None:units[r["sequence"],r["seed"]].append(float(r[name]))
    videos=defaultdict(list)
    for (seq,_),values in units.items():videos[seq].append(float(np.mean(values)))
    return {seq:float(np.mean(values)) for seq,values in videos.items()}


def weighted_auc(values,positive,weights):
    x,y,w=np.asarray(values,float),np.asarray(positive,bool),np.asarray(weights,float)
    if x.shape!=y.shape or x.shape!=w.shape or not np.isfinite(np.r_[x,w]).all() or (w<0).any():raise ValueError("Invalid AUC rows")
    a,b=w[y].sum(),w[~y].sum()
    if a<=0 or b<=0:return None
    order=np.argsort(x,kind="stable");x,y,w=x[order],y[order],w[order];negative=0.;total=0.;i=0
    while i<len(x):
        end=i+1
        while end<len(x) and x[end]==x[i]:end+=1
        pos=w[i:end][y[i:end]].sum();neg=w[i:end][~y[i:end]].sum()
        total+=pos*(negative+.5*neg);negative+=neg;i=end
    return float(total/(a*b))


def balanced(rows):
    """Each video equal; then available seeds; then rows within a seed."""
    counts=defaultdict(int);seeds=defaultdict(set)
    for r in rows:counts[r["sequence"],r["seed"]]+=1;seeds[r["sequence"]].add(r["seed"])
    return np.array([1/(len(seeds)*len(seeds[r["sequence"]])*counts[r["sequence"],r["seed"]]) for r in rows])


def population_rows(data,ident):
    valid=data["valid"];q0,old=data["q0_iou"],data["policy_iou"];rows=[]
    harm=valid&(q0>=DESIGN["good_iou"])&(old<DESIGN["low_iou"])
    rescue=valid&(q0<DESIGN["low_iou"])&(old>=DESIGN["good_iou"])
    for j,rule in enumerate(RULES):
        protect=data["actions"][:,j];new=np.where(protect,q0,old)
        rows.append(dict(ident,rule=rule,frames=len(valid),valid_frames=int(valid.sum()),physical_nonq0=int(data["physical"].sum()),
            protected_frames=int(protect.sum()),protection_fraction=float(protect.mean()),
            shadow_current_iou=float(new[valid].mean()) if valid.any() else None,
            shadow_current_gain=float((new-old)[valid].mean()) if valid.any() else None,
            harmful_q0_frames=int(harm.sum()),harmful_q0_blocked=int((harm&protect).sum()),
            rescue_frames=int(rescue.sum()),rescue_blocked=int((rescue&protect).sum())))
    return rows


def outcome(q0,policy):
    if max(q0,policy)<DESIGN["low_iou"]:return "both_low"
    if q0-policy>DESIGN["epsilon"]:return "q0_better"
    if policy-q0>DESIGN["epsilon"]:return "policy_better"
    return "tie"


def anchor_records(anchor,shared,trace,scorer,entry,outputs):
    first=int(anchor["frame"]);history=prior.initial_state(trace,entry,first)["history"]
    observation={k:v[0] for k,v in shared.items()};e=online_evidence(observation,history,scorer)
    p,q=(prior.metrics(outputs[t],entry) for t in prior.TREATMENTS)
    flag=decisions(e,bool(np.any(observation["boxes"]!=observation["candidate_boxes"][0])))
    ident={k:anchor[k] for k in ("anchor_id","cohort","sequence","seed","frame")}
    record=dict(ident,roles="|".join(anchor["roles"]),**e,outcome=outcome(q["future_mean_iou"],p["future_mean_iou"]),
        policy_future=p["future_mean_iou"],q0_future=q["future_mean_iou"],future_gain=q["future_mean_iou"]-p["future_mean_iou"],
        policy_low=p["future_low_iou_fraction"],q0_low=q["future_low_iou_fraction"],policy_current=p["current_iou"],q0_current=q["current_iou"])
    decisions_rows=[]
    for j,rule in enumerate(RULES):
        decisions_rows.append(dict(record,rule=rule,protect_probability=float(flag[j])))
    return record,decisions_rows


def matched_summary(anchors,decisions_rows):
    rows=list(decisions_rows)
    for rule in RULES[2:]:
        units=defaultdict(list)
        for r in decisions_rows:
            if r["rule"]==rule:units[r["sequence"],r["seed"]].append(r)
        for values in units.values():
            rate=float(np.mean([r["protect_probability"] for r in values]))
            rows.extend(dict(r,rule=rule+"__rate_expected",protect_probability=rate) for r in values)
    rows.extend(dict(r,rule="future_oracle",protect_probability=float(r["future_gain"]>DESIGN["epsilon"]))
                for r in decisions_rows if r["rule"]=="policy")
    for r in rows:
        a=r["protect_probability"];r["future_iou"]=r["policy_future"]+a*r["future_gain"]
        r["low_fraction"]=r["policy_low"]+a*(r["q0_low"]-r["policy_low"])
        r["future_gain_realized"]=a*r["future_gain"];r["regret"]=max(r["future_gain"],0.)-a*r["future_gain"]
    summary=[];paired=[];videos=[]
    for cohort in ("all","historical","expanded"):
        for role in ("all_unique_anchors",*prior.ROLES):
            local=[r for r in rows if (cohort=="all" or r["cohort"]==cohort) and (role=="all_unique_anchors" or role in r["roles"].split("|"))]
            if not local:continue
            for rule in POLICIES:
                part=[r for r in local if r["rule"]==rule];means={m:macro(part,m) for m in ("future_iou","low_fraction","protect_probability","future_gain_realized","regret")}
                summary.append(dict(cohort=cohort,role=role,rule=rule,anchors=len(part),videos=len(means["future_iou"]),
                    **{m:float(np.mean(list(v.values()))) for m,v in means.items()}))
                for seq in sorted(means["future_iou"]):videos.append(dict(cohort=cohort,role=role,rule=rule,sequence=seq,**{m:v[seq] for m,v in means.items()}))
            for rule in RULES[2:]:
                treatment=macro([r for r in local if r["rule"]==rule],"future_iou")
                rate_rows=[r for r in local if r["rule"]==rule+"__rate_expected" and 0<r["protect_probability"]<1]
                for control in ("policy","q0_all",rule+"__rate_expected"):
                    reference=macro([r for r in local if r["rule"]==control],"future_iou");keys=sorted(treatment.keys()&reference.keys())
                    delta=np.array([treatment[k]-reference[k] for k in keys]);rng=np.random.default_rng(DESIGN["bootstrap_seed"])
                    boots=delta[rng.integers(0,len(delta),(DESIGN["bootstrap_samples"],len(delta)))].mean(1)
                    paired.append(dict(cohort=cohort,role=role,contrast=rule+"_minus_"+control,videos=len(delta),mean_delta=float(delta.mean()),
                        rate_informative_contexts=len({(r["sequence"],r["seed"]) for r in rate_rows}),rate_informative_anchors=len(rate_rows),
                        ci_low=float(np.quantile(boots,.025)),ci_high=float(np.quantile(boots,.975)),wins=int((delta>1e-12).sum()),ties=int((np.abs(delta)<=1e-12).sum()),losses=int((delta< -1e-12).sum())))
    return rows,summary,paired,videos


def evidence_summaries(anchors):
    summary=[];ranking=[]
    for cohort in ("all","historical","expanded"):
        part=[r for r in anchors if cohort=="all" or r["cohort"]==cohort]
        if not part:continue
        for kind in ("all",*OUTCOMES):
            selected=[r for r in part if kind=="all" or r["outcome"]==kind]
            for name in FEATURES:
                v=macro(selected,name)
                summary.append(dict(cohort=cohort,outcome=kind,feature=name,anchors=len(selected),videos=len(v),mean=float(np.mean(list(v.values()))) if v else None,
                    minimum=min((r[name] for r in selected),default=None),maximum=max((r[name] for r in selected),default=None)))
        # Ties excluded; both-low cases with a nonzero gain remain in sign discrimination.
        selected=[r for r in part if abs(r["future_gain"])>DESIGN["epsilon"]];w=balanced(selected)
        for name in EVIDENCE:
            auc=weighted_auc([r[name] for r in selected],[r["future_gain"]>0 for r in selected],w)
            ranking.append(dict(cohort=cohort,feature=name,anchors=len(selected),positive=sum(r["future_gain"]>0 for r in selected),
                negative=sum(r["future_gain"]<0 for r in selected),videos=len({r["sequence"] for r in selected}),weighted_auc=auc))
    return summary,ranking


def run(root,r113,p113,r115,p115,plan,models,load_entry,old):
    tables={name:[] for name in TABLES};evidence={}
    by_context=defaultdict(list)
    for a in p115["anchors"]:by_context[a["context"]].append(a)
    for job in p113["jobs"]:
        if job["arm"]!="generated_shared":continue
        context=prior.key(job);entry=load_entry(job["spec"]);trace=prior.required_trace(r113,p113,job,models);generator,scorer=s.job_models(job,models)
        s.audit(trace,entry,old,"generated_shared",generator,scorer)
        ident=dict(context=context,cohort=job["spec"]["cohort"],sequence=job["spec"]["unit"]["sequence"],seed=job["seed"])
        data=s.s109.cache(root,root/"audit"/(context+".npz"),dict(plan_fingerprint=plan["fingerprint"],job=job),lambda:diagnose(trace,entry,scorer),evidence)
        tables["population_scope"].extend(population_rows(data,ident))
        for a in by_context[context]:
            meta=dict(plan_fingerprint=p115["fingerprint"],anchor=a)
            shared=c.read_cache(r115/"points"/(a["anchor_id"]+".npz"),dict(meta,kind="shared_point"))
            outputs={t:c.read_cache(r115/"branches"/(a["anchor_id"]+"__"+t+".npz"),dict(meta,kind="branch",treatment=t)) for t in prior.TREATMENTS}
            if shared is None or any(v is None for v in outputs.values()):raise ValueError("Missing matched measured branches")
            for t,d in outputs.items():prior.audit(d,trace,shared,entry,a,t,old,generator,scorer)
            prior.bridge(outputs["policy"],trace,a,not p115["runtime_differences"])
            record,actions=anchor_records(a,shared,trace,scorer,entry,outputs)
            tables["anchor_evidence"].append(record);tables["anchor_decisions"].extend(actions)
        print("[Stage-116 cached] "+context,flush=True)
    population=tables["population_scope"]
    for cohort in ("all","historical","expanded"):
        for rule in RULES:
            part=[r for r in population if r["rule"]==rule and (cohort=="all" or r["cohort"]==cohort)]
            if part:tables["population_summary"].append(dict(cohort=cohort,rule=rule,contexts=len(part),videos=len({r["sequence"] for r in part}),
                **{k:float(np.mean(list(macro(part,k).values()))) for k in ("protection_fraction","shadow_current_iou","shadow_current_gain")},
                **{k:sum(r[k] for r in part) for k in ("frames","valid_frames","physical_nonq0","protected_frames","harmful_q0_frames","harmful_q0_blocked","rescue_frames","rescue_blocked")}))
    anchors=tables["anchor_evidence"]
    tables["evidence_summary"],tables["discrimination"]=evidence_summaries(anchors)
    tables["anchor_decisions"],tables["policy_summary"],tables["paired"],tables["video_effects"]=matched_summary(anchors,tables["anchor_decisions"])
    publish(root,plan,tables,evidence)


def publish(root,plan,tables,evidence):
    expected=plan["expected"];counts={k:len(v) for k,v in tables.items()}
    contexts=[r for r in tables["population_scope"] if r["rule"]=="policy"]
    if (counts["population_scope"]!=expected["contexts"]*len(RULES) or counts["anchor_evidence"]!=expected["anchors"]
        or counts["anchor_decisions"]!=expected["anchors"]*len(POLICIES) or len(evidence)!=2*expected["contexts"]
        or len({r["context"] for r in contexts})!=expected["contexts"] or sum(r["frames"] for r in contexts)!=expected["frames"]):raise ValueError("Incomplete evidence population")
    out=root/"analysis";out.mkdir(exist_ok=True);hashes={}
    for name,rows in tables.items():
        path=out/f"stage116_{name}.csv";s.s103.s86.publish_table(path,rows);hashes[path.name]=c.file_hash(path)
    io.verify(plan["source_sha256"]);io.verify(plan["code_sha256"])
    report=io.seal(dict(schema=SCHEMA,status=STATUS,plan_fingerprint=plan["fingerprint"],source_stage115_fingerprint=SOURCE115,
        expected=expected,counts=counts,training_updates=0,new_models=0,image_reads=0,new_search_calls=0,new_block_calls=0,new_full_trajectories=0,confirmation_inference_calls=0,
        population_frame_exposures=sum(r["frames"] for r in contexts),
        table_sha256=hashes,artifact_sha256=evidence,warnings=[
            "Consumed-development, GT-selected anchors: descriptive evidence only; no fitted gate or adoption test.",
            "Population decisions are one-frame shadows on frozen source histories, not rolled-out policies or AUC.",
            "Matched future utility selects an already measured one-action branch; excludes the current frame.",
            "Rate controls match within source video/seed over ALL anchors before role subgrouping.",
            "AUC orientation and zero-comparison rules are fixed; no threshold or feature selection is performed.",
            "Future oracle uses labels as an upper bound only. No fresh appearance descriptors exist in these caches."]))
    c.publish_identical(out/"stage116_decision.json",report);check_results(root)


def check_results(root):
    plan=c.c66.sealed(root/"manifest.json",SCHEMA);r=c.c66.sealed(root/"analysis/stage116_decision.json",SCHEMA);e=plan["expected"]
    if (plan["design"]!=DESIGN or plan["features"]!=list(FEATURES) or plan["rules"]!=list(RULES) or plan["source_stage115_fingerprint"]!=SOURCE115 or r["status"]!=STATUS
        or r["plan_fingerprint"]!=plan["fingerprint"] or r["source_stage115_fingerprint"]!=SOURCE115 or r["expected"]!=e
        or r["counts"]["population_scope"]!=e["contexts"]*len(RULES) or r["counts"]["anchor_evidence"]!=e["anchors"]
        or r["population_frame_exposures"]!=e["frames"]
        or r["counts"]["anchor_decisions"]!=e["anchors"]*len(POLICIES) or len(r["artifact_sha256"])!=2*e["contexts"]
        or set(r["table_sha256"])!={f"stage116_{n}.csv" for n in TABLES}
        or any(r[k]!=0 for k in ("training_updates","new_models","image_reads","new_search_calls","new_block_calls","new_full_trajectories","confirmation_inference_calls"))):raise ValueError("Incomplete Stage-116 evidence audit")
    io.verify(plan["source_sha256"]);io.verify(plan["code_sha256"])
    io.verify({str(root/"analysis"/k):v for k,v in r["table_sha256"].items()});io.verify({str(root/k):v for k,v in r["artifact_sha256"].items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__);names=("stage76","stage89","stage93","stage112","stage113","stage115")
    for name in (*names,"output_dir"):parser.add_argument("--"+name,type=Path,required=name=="output_dir")
    parser.add_argument("--check-results",action="store_true");args=parser.parse_args();root=args.output_dir.expanduser().resolve()
    if args.check_results:check_results(root);return
    if not all(getattr(args,n) for n in names):parser.error("Supply frozen Stage-76/89/93/112/113/115 roots")
    r76,r89,r93,r112,r113,r115=paths=[getattr(args,n).expanduser().resolve() for n in names]
    if any(root==p or root in p.parents or p in root.parents for p in paths):parser.error("Use a separate Stage-116 root")
    with s.s109.s106.s93.writer_lock(root):
        if (root/"analysis/stage116_decision.json").exists():check_results(root);return
        if not (root/"manifest.json").exists() and any(p.name!="collect.lock" for p in root.iterdir()):raise ValueError("Use an empty Stage-116 root")
        prior.check_results(r115);s.check_results(r113);s.s112.check_results(r112)
        p115=c.c66.sealed(r115/"manifest.json",prior.SCHEMA);r115report=c.c66.sealed(r115/"analysis/stage115_decision.json",prior.SCHEMA)
        p113=c.c66.sealed(r113/"manifest.json",s.SCHEMA);p112=c.c66.sealed(r112/"manifest.json",s.common.SCHEMA)
        if r115report["fingerprint"]!=SOURCE115:raise ValueError("Need the frozen complete Stage-115 result")
        if (c.c66.sealed(r113/"analysis/stage113_decision.json",s.SCHEMA)["fingerprint"]!=prior.prior.SOURCE113
            or c.c66.sealed(r112/"analysis/stage112_decision.json",s.common.SCHEMA)["fingerprint"]!=s.SOURCE112):raise ValueError("Frozen generator/scorer source differs")
        jobs=[j for j in p113["jobs"] if j["arm"]=="generated_shared"]
        expected=dict(contexts=len(jobs),frames=sum(j["frames"] for j in jobs),anchors=len(p115["anchors"]))
        if expected!=dict(contexts=56,frames=58094,anchors=34):raise ValueError("Frozen evidence cohort changed")
        p93,_,p89,r89report,origin=s.s103.prior.s95.s94.load_run(r93,r76,r89);models,_=s.load_frozen_models(r112,p112)
        bindings=dict(p115["source_sha256"])
        for p in (r115/"manifest.json",r115/"analysis/stage115_decision.json"):bindings[str(p)]=c.file_hash(p)
        bindings.update({str(r115/k):v for k,v in r115report["artifact_sha256"].items()});bindings.update({str(r115/"analysis"/k):v for k,v in r115report["table_sha256"].items()})
        code=[Path(__file__).resolve(),Path(__file__).with_name("run_stage116_online_evidence.sh").resolve()]
        plan=io.seal(dict(schema=SCHEMA,design=DESIGN,features=list(FEATURES),rules=list(RULES),expected=expected,source_stage115_fingerprint=SOURCE115,
            source_sha256=bindings,code_sha256=p115["code_sha256"]|{str(p):c.file_hash(p) for p in code}))
        c.publish_identical(root/"manifest.json",plan)
        def load_entry(spec):return s.s103.prior.s95.load_unit(r93,r89,p93,p89,r89report,origin,spec)[0]
        print("Stage-116: cached online evidence; no images, GPU inference or fitting",flush=True)
        run(root,r113,p113,r115,p115,plan,models,load_entry,origin["original"])
        print("Stage-116: "+STATUS+"; no method adoption gate",flush=True)


if __name__=="__main__":main()
