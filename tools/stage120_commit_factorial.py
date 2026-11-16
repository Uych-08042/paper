"""Cross-fit training-support x RGB-appearance commit heads on frozen measured futures."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import platform
import numpy as np
import stage119_commit_support as audit
import stage120_appearance as appearance

gate=audit.prior;s,c,io=gate.s,gate.c,gate.io
SCHEMA="stage120_support_appearance_commit_factorial_v1"
STATUS="COMPLETE_SUPPORT_APPEARANCE_COMMIT_FACTORIAL"
SOURCE119="81cc50e2b8a74be60274f9a1f02b793176125e3fcaae21e1dd8a30e9bbbaf5ad"
DESIGN=dict(ridge=1.,minimum_scale=1e-6,feature_clip=10.,epsilon=1e-12,pressure_mass=.5,shuffle_seed=23201,
    bootstrap_seed=23200,bootstrap_samples=10000,primary_study="ordinary_uniform",primary="expanded_appearance_minus_expanded_relative")
ARMS=tuple(support+"_"+representation for support in ("uniform","expanded") for representation in ("relative","appearance","shuffled"))
SCORES=ARMS+("uniform_constant","expanded_constant")
RULES=("policy","q0_all","continuity_pair","old_myopic")+SCORES
POLICIES=RULES+tuple(a+"__rate_expected" for a in ARMS)+("future_oracle",)
TABLES=("appearance_scope","appearance_evidence","model_scope","fit_metrics","sample_predictions","policy_summary","paired","support","policy_decisions","video_effects")


def fitting_rows(records,held,provider,support):
    uniform=gate.training_rows(records,held,provider)
    if support=="uniform":return uniform
    if support!="expanded":raise ValueError("Unknown fitting support")
    allowed=set(provider["upstream_held_groups"])-{held["group"]}
    pressure=[r for r in records if r["origin"]=="stage115" and r["group"] in allowed]
    rows=sorted(uniform+pressure,key=lambda r:r["uid"])
    if any(r["outer_fold"]!=held["outer_fold"] or r["group"]==held["group"] for r in rows):raise ValueError("Cross-provider/test labels entered fit")
    return rows


def weights(rows):
    """Video -> seed equal; split uniform/pressure mass only in contexts with pressure."""
    groups=defaultdict(set);counts=defaultdict(int)
    for r in rows:groups[r["group"]].add(r["seed"]);counts[r["group"],r["seed"],r["origin"]]+=1
    result=[]
    for r in rows:
        key=(r["group"],r["seed"]);u=counts[key+("stage117",)];p=counts[key+("stage115",)]
        if not u:raise ValueError("Every fitting context needs original uniform support")
        mass=(DESIGN["pressure_mass"] if r["origin"]=="stage115" else 1-DESIGN["pressure_mass"]) if p else 1.
        result.append(mass/(len(groups)*len(groups[r["group"]])*counts[key+(r["origin"],)]))
    return np.array(result)


def matrix(rows,bank,representation):
    base=np.array([[r[k] for k in gate.FEATURES] for r in rows])
    return base if representation=="relative" else np.c_[base,np.array([bank[r["uid"]] for r in rows])]


def permutation(rows,held,support):
    result=np.arange(len(rows))
    for group in sorted({r["group"] for r in rows}):
        indices=np.array([i for i,r in enumerate(rows) if r["group"]==group])
        seed=int.from_bytes(hashlib.sha256(f"{DESIGN['shuffle_seed']}:{held}:{support}:{group}".encode()).digest()[:16],"big")
        result[indices]=np.random.Generator(np.random.PCG64(seed)).permutation(indices)
    return result


def fit_head(rows,bank,held,support,representation):
    x=matrix(rows,bank,representation);w=weights(rows);y=np.array([r["future_gain"] for r in rows]);center=w@x;scale=np.sqrt(w@((x-center)**2))
    scale[scale<DESIGN["minimum_scale"]]=1.;used=x.copy();indices=np.arange(len(rows))
    if representation=="shuffled":indices=permutation(rows,held,support);used[:,len(gate.FEATURES):]=x[indices,len(gate.FEATURES):]
    z=np.c_[np.clip((used-center)/scale,-DESIGN["feature_clip"],DESIGN["feature_clip"]),np.ones(len(rows))]
    normal=z.T@(w[:,None]*z)+DESIGN["ridge"]*np.diag([1.]*(z.shape[1]-1)+[0.]);rhs=z.T@(w*y);beta=np.linalg.solve(normal,rhs)
    error=float(np.max(np.abs(normal@beta-rhs)))
    if not np.isfinite(beta).all() or error>1e-10:raise ValueError("Factorial head solve failed")
    return io.seal(dict(schema=SCHEMA,support=support,representation=representation,held_group=held,fit_groups=sorted({r["group"] for r in rows}),
        fit_ids=[r["uid"] for r in rows],fit_weight=w.tolist(),permuted_appearance_ids=[rows[i]["uid"] for i in indices],
        input_fingerprint=io.seal(dict(ids=[r["uid"] for r in rows],x=x.tolist(),y=y.tolist(),w=w.tolist()))["fingerprint"],
        center=center.tolist(),scale=scale.tolist(),beta=beta.tolist(),constant=float(w@y),fit_mse=float(w@((z@beta-y)**2)),normal_equation_error=error,
        shuffled_value_change_fraction=float(np.mean(np.any(used[:,len(gate.FEATURES):]!=x[:,len(gate.FEATURES):],axis=1))) if representation=="shuffled" else 0.))


def predict(head,x):
    x=np.asarray(x,float)
    if x.ndim!=2 or x.shape[1]!=len(head["center"]) or not np.isfinite(x).all():raise ValueError("Invalid public inference vector")
    z=np.c_[np.clip((x-np.asarray(head["center"]))/head["scale"],-DESIGN["feature_clip"],DESIGN["feature_clip"]),np.ones(len(x))]
    return z@np.asarray(head["beta"])


def package_for(records,bank,held,provider,source_model):
    heads={}
    for arm in ARMS:
        if arm=="uniform_relative":continue
        support,representation=arm.split("_",1);fit=fitting_rows(records,held,provider,support)
        heads[arm]=fit_head(fit,bank,held["group"],support,representation)
    return io.seal(dict(schema=SCHEMA,design=DESIGN,appearance_design=appearance.DESIGN,held_group=held["group"],provider=provider,
        baseline_model_fingerprint=source_model["fingerprint"],heads=heads,expanded_constant=heads["expanded_relative"]["constant"]))


def package_input_audit(package,records,bank,held,provider,source_model):
    if (package["design"]!=DESIGN or package["appearance_design"]!=appearance.DESIGN or package["held_group"]!=held["group"]
        or package["provider"]!=provider or package["baseline_model_fingerprint"]!=source_model["fingerprint"] or set(package["heads"])!=set(ARMS)-{"uniform_relative"}):raise ValueError("Factorial model scope changed")
    for arm,head in package["heads"].items():
        support,representation=arm.split("_",1);fit=fitting_rows(records,held,provider,support);x=matrix(fit,bank,representation);y=[r["future_gain"] for r in fit];w=weights(fit)
        fp=io.seal(dict(ids=[r["uid"] for r in fit],x=x.tolist(),y=y,w=w.tolist()))["fingerprint"]
        if head["input_fingerprint"]!=fp or head["held_group"]!=held["group"] or head["fit_ids"]!=[r["uid"] for r in fit] or held["group"] in head["fit_groups"]:raise ValueError("Frozen fitting inputs changed")
        if head["support"]!=support or head["representation"]!=representation or head["fit_groups"]!=sorted({r["group"] for r in fit}):raise ValueError("Head treatment changed")
        indices=permutation(fit,held["group"],support) if representation=="shuffled" else np.arange(len(fit))
        if head["permuted_appearance_ids"]!=[fit[i]["uid"] for i in indices]:raise ValueError("Appearance shuffle changed")
        center=w@x;scale=np.sqrt(w@((x-center)**2));scale[scale<DESIGN["minimum_scale"]]=1.
        audit.close(center,head["center"],"Head normalization changed");audit.close(scale,head["scale"],"Head normalization changed")
        audit.close(w,head["fit_weight"],"Head weights changed")
        used=x.copy()
        if representation=="shuffled":used[:,len(gate.FEATURES):]=x[indices,len(gate.FEATURES):]
        z=np.c_[np.clip((used-center)/scale,-DESIGN["feature_clip"],DESIGN["feature_clip"]),np.ones(len(fit))]
        normal=z.T@(w[:,None]*z)+DESIGN["ridge"]*np.diag([1.]*(z.shape[1]-1)+[0.])
        if not np.isfinite(head["beta"]).all() or len(head["beta"])!=x.shape[1]+1:raise ValueError("Malformed head")
        audit.close(normal@np.asarray(head["beta"]),z.T@(w*np.asarray(y)),"Head normal equation changed")
        audit.close(head["constant"],w@np.asarray(y),"Head constant changed")
    audit.close(package["expanded_constant"],package["heads"]["expanded_relative"]["constant"],"Expanded constant changed")


def choices(records,predictions):
    lookup={r["uid"]:r for r in predictions};rows=[]
    for r in records:
        score=lookup[r["uid"]];flags=[False,True,r["continuity_advantage"]>DESIGN["epsilon"],r["myopic"]>DESIGN["epsilon"]]+[score[a]>DESIGN["epsilon"] for a in SCORES]
        for membership in r["memberships"].split("|"):
            study={"uniform":"ordinary_uniform","stratified":"ordinary_stratified","pressure":"pressure"}[membership]
            weight=1. if membership=="pressure" else r[membership+"_weight"]
            rows.extend(dict(r,study=study,analysis_weight=weight,rule=rule,protect_probability=float(flag)) for rule,flag in zip(RULES,flags))
    expanded=list(rows)
    for arm in ARMS:
        contexts=defaultdict(list)
        for r in rows:
            if r["rule"]==arm:contexts[r["study"],r["sequence"],r["seed"]].append(r)
        for part in contexts.values():
            p=sum(r["analysis_weight"]*r["protect_probability"] for r in part)/sum(r["analysis_weight"] for r in part)
            expanded.extend(dict(r,rule=arm+"__rate_expected",protect_probability=p) for r in part)
    expanded.extend(dict(r,rule="future_oracle",protect_probability=float(r["future_gain"]>DESIGN["epsilon"])) for r in rows if r["rule"]=="policy")
    for r in expanded:
        p=r["protect_probability"];d=r["future_gain"]
        r.update(future_iou=r["policy_future"]+p*d,low_fraction=r["policy_low"]+p*(r["q0_low"]-r["policy_low"]),realized_gain=p*d,
            harm_cost=p*max(-d,0.),missed_gain=(1-p)*max(d,0.),regret=max(d,0.)-p*d)
    return expanded


def summaries(rows):
    summary=[];paired=[];support=[];video=[];macro=gate.prior.weighted_macro
    contrasts=[("expanded_appearance_minus_"+control,{"expanded_appearance":1.,control:-1.}) for control in (
        "expanded_relative","uniform_appearance","uniform_relative","expanded_shuffled","policy","q0_all","continuity_pair","old_myopic","expanded_constant","expanded_appearance__rate_expected")]
    contrasts += [("expanded_relative_minus_uniform_relative",dict(expanded_relative=1.,uniform_relative=-1.)),
        ("uniform_appearance_minus_uniform_relative",dict(uniform_appearance=1.,uniform_relative=-1.)),
        ("uniform_appearance_minus_uniform_shuffled",dict(uniform_appearance=1.,uniform_shuffled=-1.)),
        ("support_by_appearance_interaction",dict(expanded_appearance=1.,expanded_relative=-1.,uniform_appearance=-1.,uniform_relative=1.))]
    for study in gate.STUDIES:
        for cohort in ("all","historical","expanded"):
            pool=[r for r in rows if r["study"]==study and (cohort=="all" or r["cohort"]==cohort)]
            if not pool:continue
            for role in ("all","good_q0_lost","medium_q0_lost","q0_rescue","other"):
                local=[r for r in pool if r["rule"]=="policy" and (role=="all" or r["current_role"]==role)]
                support.append(dict(study=study,cohort=cohort,current_role=role,anchors=len(local),videos=len({r["sequence"] for r in local})))
            for role in (("all","good_q0_lost","medium_q0_lost","q0_rescue") if study=="pressure" else ("all",)):
                part=[r for r in pool if role=="all" or r["current_role"]==role]
                if not part:continue
                values={}
                for rule in POLICIES:
                    local=[r for r in part if r["rule"]==rule];v={k:macro(local,k) for k in ("future_iou","low_fraction","protect_probability","realized_gain","harm_cost","missed_gain","regret")};values[rule]=v["future_iou"]
                    summary.append(dict(study=study,cohort=cohort,current_role=role,rule=rule,anchors=len(local),videos=len(v["future_iou"]),
                        large_harm_anchors=sum(r["realized_gain"]<-.1 for r in local),worst_anchor_gain=min(r["realized_gain"] for r in local),**{k:float(np.mean(list(a.values()))) for k,a in v.items()}))
                    video.extend(dict(study=study,cohort=cohort,current_role=role,rule=rule,sequence=seq,**{k:a[seq] for k,a in v.items()}) for seq in sorted(v["future_iou"]))
                for name,terms in contrasts:
                    groups=sorted(values[next(iter(terms))]);delta=np.array([sum(weight*values[rule][g] for rule,weight in terms.items()) for g in groups]);rng=np.random.default_rng(DESIGN["bootstrap_seed"])
                    boot=delta[rng.integers(0,len(delta),(DESIGN["bootstrap_samples"],len(delta)))].mean(1)
                    paired.append(dict(study=study,cohort=cohort,current_role=role,contrast=name,videos=len(groups),mean_delta=float(delta.mean()),median_delta=float(np.median(delta)),
                        ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),wins=int((delta>1e-12).sum()),ties=int((np.abs(delta)<=1e-12).sum()),losses=int((delta< -1e-12).sum()),
                        worst_video_delta=float(delta.min()),minimum_leave_one_video_out=float(np.min((delta.sum()-delta)/(len(delta)-1))) if len(delta)>1 else None))
    return summary,paired,support,video


def run(root,r118,p118,plan,records,reader=appearance.checked_image):
    tables={k:[] for k in TABLES};artifacts={};bank={};image_reads=0
    for context,tasks in sorted(plan["appearance_tasks"].items()):
        data=s.s109.cache(root,root/"appearance"/(context+".npz"),dict(plan_fingerprint=plan["fingerprint"],context=context,uids=[t["uid"] for t in tasks]),
            lambda:appearance.collect(plan["image_root"],tasks,reader),artifacts)
        if data["x"].shape!=(len(tasks),len(appearance.FEATURES)) or not np.isfinite(data["x"]).all():raise ValueError("Incomplete appearance cache")
        np.testing.assert_array_equal(data["recent_frame"],[t["recent_frame"] for t in tasks]);image_reads+=int(data["image_reads"])
        tables["appearance_scope"].append(dict(context=context,points=len(tasks),image_reads=int(data["image_reads"]),recent_supported=sum(t["recent_frame"]>=1 for t in tasks)))
        for i,t in enumerate(tasks):
            bank[t["uid"]]=data["x"][i];tables["appearance_evidence"].append(dict(uid=t["uid"],recent_frame=t["recent_frame"],**{k:float(data["x"][i,j]) for j,k in enumerate(appearance.FEATURES)}))
    providers={int(k):v for k,v in p118["providers"].items()}
    for group,held in sorted(p118["owners"].items()):
        provider=providers[held["outer_fold"]];baseline=c.c66.sealed(r118/"models"/(group+".json"),gate.SCHEMA)
        gate.audit_package(baseline,gate.training_rows(records,held,provider),held,provider)
        path=root/"models"/(group+".json")
        if not path.exists():c.publish_identical(path,package_for(records,bank,held,provider,baseline))
        pack=c.c66.sealed(path,SCHEMA);package_input_audit(pack,records,bank,held,provider,baseline);artifacts[path.relative_to(root).as_posix()]=c.file_hash(path)
        for name in SCORES:
            support=name.split("_",1)[0];fit=fitting_rows(records,held,provider,support);w=weights(fit);y=np.array([r["future_gain"] for r in fit]);pressure=[r["origin"]=="stage115" for r in fit]
            if name=="uniform_relative":head=None;parameters=14;loss=float(w@((gate.predict(baseline,matrix(fit,bank,"relative"))[:,1]-y)**2));shuffled=0.
            elif name.endswith("constant"):head=None;parameters=1;constant=baseline["constant"][1] if support=="uniform" else pack["expanded_constant"];loss=float(w@((constant-y)**2));shuffled=0.
            else:head=pack["heads"][name];parameters=len(head["beta"]);loss=head["fit_mse"];shuffled=head["shuffled_value_change_fraction"]
            tables["model_scope"].append(dict(held_group=group,arm=name,fit_groups="|".join(sorted({r["group"] for r in fit})),fit_examples=len(fit),uniform_examples=sum(not p for p in pressure),
                pressure_examples=sum(pressure),pressure_weight=float(w@pressure),
                pressure_good_q0_lost=sum(r["origin"]=="stage115" and r["current_role"]=="good_q0_lost" for r in fit),
                pressure_medium_q0_lost=sum(r["origin"]=="stage115" and r["current_role"]=="medium_q0_lost" for r in fit),
                pressure_q0_rescue=sum(r["origin"]=="stage115" and r["current_role"]=="q0_rescue" for r in fit),
                parameters=parameters,new_model=name not in ("uniform_relative","uniform_constant"),no_test_group_in_fit=all(r["group"]!=group for r in fit)))
            tables["fit_metrics"].append(dict(held_group=group,arm=name,fit_mse=loss,fit_examples=len(fit),shuffled_value_change_fraction=shuffled))
        part=sorted([r for r in records if r["group"]==group],key=lambda r:r["uid"]);base=matrix(part,bank,"relative");rich=matrix(part,bank,"appearance")
        if any(r["model_fingerprint"]!=baseline["fingerprint"] for r in part):raise ValueError("Baseline point binding changed")
        audit.close(gate.predict(baseline,base)[:,1],[r["future"] for r in part],"Original future head changed")
        def infer():
            scores={"uniform_relative":np.array([r["future"] for r in part]),"uniform_constant":np.array([r["future_constant"] for r in part]),"expanded_constant":np.full(len(part),pack["expanded_constant"])}
            for name,head in pack["heads"].items():scores[name]=predict(head,base if name.endswith("relative") else rich)
            return dict(base=base,appearance=rich[:,len(gate.FEATURES):],scores=np.column_stack([scores[k] for k in SCORES]))
        prediction=s.s109.cache(root,root/"predictions"/(group+".npz"),dict(plan_fingerprint=plan["fingerprint"],model_fingerprint=pack["fingerprint"],uids=[r["uid"] for r in part]),infer,artifacts)
        np.testing.assert_array_equal(prediction["base"],base);np.testing.assert_array_equal(prediction["appearance"],rich[:,len(gate.FEATURES):])
        if prediction["scores"].shape!=(len(part),len(SCORES)) or not np.isfinite(prediction["scores"]).all():raise ValueError("Malformed frozen factorial predictions")
        np.testing.assert_array_equal(prediction["scores"][:,0],[r["future"] for r in part])
        for i,r in enumerate(part):tables["sample_predictions"].append(dict(r,**{k:float(prediction["scores"][i,j]) for j,k in enumerate(SCORES)}))
        print(f"[Stage-120] {group}: held-out factorial heads/predictions saved",flush=True)
    tables["policy_decisions"]=choices(records,tables["sample_predictions"])
    tables["policy_summary"],tables["paired"],tables["support"],tables["video_effects"]=summaries(tables["policy_decisions"])
    publish(root,plan,tables,artifacts,image_reads)


def publish(root,plan,tables,artifacts,image_reads):
    e=plan["expected"];counts={k:len(v) for k,v in tables.items()}
    if (counts["appearance_scope"]!=e["contexts"] or counts["appearance_evidence"]!=e["points"] or counts["sample_predictions"]!=e["points"]
        or counts["model_scope"]!=8*e["groups"] or counts["fit_metrics"]!=8*e["groups"] or counts["policy_decisions"]!=e["memberships"]*len(POLICIES)
        or len(artifacts)!=2*e["contexts"]+3*e["groups"] or image_reads!=e["image_reads"]):raise ValueError("Incomplete support/appearance factorial")
    out=root/"analysis";out.mkdir(exist_ok=True);hashes={}
    for name,rows in tables.items():
        path=out/f"stage120_{name}.csv";s.s103.s86.publish_table(path,rows);hashes[path.name]=c.file_hash(path)
    io.verify(plan["source_sha256"]);io.verify(plan["image_sha256"]);io.verify(plan["code_sha256"])
    primary=next(r for r in tables["paired"] if r["study"]==DESIGN["primary_study"] and r["cohort"]=="all" and r["current_role"]=="all" and r["contrast"]==DESIGN["primary"])
    c.publish_identical(out/"stage120_decision.json",io.seal(dict(schema=SCHEMA,status=STATUS,plan_fingerprint=plan["fingerprint"],source_stage119_fingerprint=SOURCE119,
        expected=e,counts=counts,primary=primary,training_performed=True,new_ridge_heads=5*e["groups"],new_constant_controls=e["groups"],new_models=6*e["groups"],
        reused_ridge_heads=e["groups"],reused_constant_controls=e["groups"],image_reads=image_reads,sgla_training_updates=0,new_search_calls=0,new_block_calls=0,new_full_trajectories=0,confirmation_inference_calls=0,
        baseline_prediction_error=0.,table_sha256=hashes,artifact_sha256=artifacts,warnings=["Only other groups within the same upstream provider-held fold fit each model; the evaluation video, both seeds and all its pressure labels are excluded.",
            "Pressure-supported training is deliberate consumed-development resampling, not an estimate of the natural population. Evaluation studies remain separate.",
            "RGB histograms/NCC are lightweight appearance cues, not validated semantic identity embeddings. Trusted past predictions may be wrong.",
            "Shuffled controls preserve within-training-video appearance marginals and model size; they do not prove unique causal attribution.",
            "One measured q0 action only. No new SGLA inference, complete-video AUC, adoption gate or independent confirmation."])))
    check_results(root)


def check_results(root):
    p=c.c66.sealed(root/"manifest.json",SCHEMA);r=c.c66.sealed(root/"analysis/stage120_decision.json",SCHEMA);e=p["expected"];n=r["counts"]
    if (p["design"]!=DESIGN or p["appearance_design"]!=appearance.DESIGN or r["status"]!=STATUS or p["source_stage119_fingerprint"]!=SOURCE119 or r["source_stage119_fingerprint"]!=SOURCE119
        or r["plan_fingerprint"]!=p["fingerprint"] or r["expected"]!=e or n["appearance_scope"]!=e["contexts"] or n["appearance_evidence"]!=e["points"] or n["sample_predictions"]!=e["points"]
        or n["model_scope"]!=8*e["groups"] or n["fit_metrics"]!=8*e["groups"] or n["policy_decisions"]!=e["memberships"]*len(POLICIES)
        or r["new_ridge_heads"]!=5*e["groups"] or r["new_constant_controls"]!=e["groups"] or r["new_models"]!=6*e["groups"] or not r["training_performed"] or r["image_reads"]!=e["image_reads"]
        or r["reused_ridge_heads"]!=e["groups"] or r["reused_constant_controls"]!=e["groups"]
        or r["primary"]["study"]!=DESIGN["primary_study"] or r["primary"]["contrast"]!=DESIGN["primary"] or r["primary"]["cohort"]!="all" or r["primary"]["current_role"]!="all"
        or len(r["artifact_sha256"])!=2*e["contexts"]+3*e["groups"] or set(r["table_sha256"])!={f"stage120_{k}.csv" for k in TABLES}
        or any(r[k]!=0 for k in ("sgla_training_updates","new_search_calls","new_block_calls","new_full_trajectories","confirmation_inference_calls","baseline_prediction_error"))):raise ValueError("Incomplete Stage-120 result")
    io.verify(p["source_sha256"]);io.verify(p["image_sha256"]);io.verify(p["code_sha256"])
    io.verify({str(root/"analysis"/k):v for k,v in r["table_sha256"].items()});io.verify({str(root/k):v for k,v in r["artifact_sha256"].items()})


def main():
    names=("stage76","stage89","stage93","stage112","stage113","stage115","stage117","stage118","stage119")
    parser=argparse.ArgumentParser(description=__doc__)
    for name in (*names,"output_dir"):parser.add_argument("--"+name,type=Path,required=name=="output_dir")
    parser.add_argument("--check-results",action="store_true");args=parser.parse_args();root=args.output_dir.expanduser().resolve()
    if args.check_results:check_results(root);return
    if not all(getattr(args,k) for k in names):parser.error("Supply the frozen source roots; no checkpoint/GPU is needed")
    sources={k:getattr(args,k).expanduser().resolve() for k in names}
    if any(root==p or root in p.parents or p in root.parents for p in sources.values()):parser.error("Use a separate Stage-120 root")
    with s.s109.s106.s93.writer_lock(root):
        if (root/"analysis/stage120_decision.json").exists():check_results(root);return
        if not (root/"manifest.json").exists() and any(p.name!="collect.lock" for p in root.iterdir()):raise ValueError("Use an empty Stage-120 root")
        r118=sources["stage118"];r119=sources["stage119"];audit.check_results(r119);gate.check_results(r118)
        p119=c.c66.sealed(r119/"manifest.json",audit.SCHEMA);report=c.c66.sealed(r119/"analysis/stage119_decision.json",audit.SCHEMA)
        if report["fingerprint"]!=SOURCE119:raise ValueError("Frozen Stage-119 result differs")
        p118=c.c66.sealed(r118/"manifest.json",gate.SCHEMA);p113=c.c66.sealed(sources["stage113"]/"manifest.json",s.SCHEMA);p112=c.c66.sealed(sources["stage112"]/"manifest.json",s.common.SCHEMA)
        if c.c66.sealed(r118/"analysis/stage118_decision.json",gate.SCHEMA)["fingerprint"]!=audit.SOURCE118:raise ValueError("Frozen Stage-118 result differs")
        branch=appearance.branch;sampled=appearance.sampled
        for name,module in (("stage113",s),("stage112",s.s112),("stage115",branch),("stage117",sampled)):module.check_results(sources[name])
        for name,schema,expected in (("stage113",s.SCHEMA,branch.prior.SOURCE113),("stage112",s.common.SCHEMA,s.SOURCE112),
            ("stage115",branch.SCHEMA,gate.ev.SOURCE115),("stage117",sampled.SCHEMA,gate.SOURCE117)):
            if c.c66.sealed(sources[name]/"analysis"/(name+"_decision.json"),schema)["fingerprint"]!=expected:raise ValueError("Frozen pixel/trajectory source differs")
        p115=c.c66.sealed(sources["stage115"]/"manifest.json",branch.SCHEMA);p117=c.c66.sealed(sources["stage117"]/"manifest.json",sampled.SCHEMA)
        _,_,p89,r89,origin=s.s103.prior.s95.s94.load_run(sources["stage93"],sources["stage76"],sources["stage89"])
        def load_entry(spec):
            sequence=spec["unit"]["sequence"]
            if spec["cohort"]=="historical":return c.entry_for(origin["original"],sequence)
            path=sources["stage89"]/"entries"/(sequence+".json");io.verify({str(path):r89["evidence_sha256"]["entries/"+sequence+".json"]})
            bound=c.c66.sealed(path,s.s103.prior.s95.s94.s89.SCHEMA)
            if bound["plan_fingerprint"]!=p89["fingerprint"]:raise ValueError("Expanded source entry changed")
            return bound["entry"]
        records=audit.load_records(r118,p118);models,_=s.load_frozen_models(sources["stage112"],p112);image_root=s.e63.c.actor_plan(origin["original"])["root"]
        tasks,image_hashes=appearance.prepare(records,sources["stage113"],p113,models,sources["stage115"],p115,sources["stage117"],p117,load_entry,image_root)
        expected=dict(groups=28,points=778,memberships=791,contexts=56,image_reads=sum(len({f for t in part for f in t["images"]}) for part in tasks.values()))
        if len(records)!=778 or len(tasks)!=56 or len(p118["owners"])!=28:raise ValueError("Frozen evaluation cohort differs")
        bindings=dict(p119["source_sha256"])
        for path in (r119/"manifest.json",r119/"analysis/stage119_decision.json"):bindings[str(path)]=c.file_hash(path)
        bindings.update({str(r119/"analysis"/k):v for k,v in report["table_sha256"].items()})
        code=[Path(__file__).resolve(),Path(appearance.__file__).resolve(),Path(__file__).with_name("run_stage120_commit_factorial.sh").resolve()]
        plan=io.seal(dict(schema=SCHEMA,design=DESIGN,appearance_design=appearance.DESIGN,source_stage119_fingerprint=SOURCE119,expected=expected,
            appearance_features=list(appearance.FEATURES),appearance_tasks=tasks,image_root=str(image_root),image_sha256=image_hashes,
            analysis_runtime=dict(python=platform.python_version(),numpy=np.__version__),source_sha256=bindings,code_sha256=p119["code_sha256"]|{str(p):c.file_hash(p) for p in code}))
        io.verify(image_hashes);c.publish_identical(root/"manifest.json",plan);run(root,r118,p118,plan,records);print("Stage-120: "+STATUS+"; CPU training/image reads, no GPU inference",flush=True)


if __name__=="__main__":main()
