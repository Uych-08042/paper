"""Fit provider-isolated CPU commit heads; audit ordinary utility and rescue pressure."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import platform
import numpy as np

import stage117_sampled_q0 as prior

ev,s,c,io=prior.evidence,prior.s,prior.c,prior.io
SCHEMA="stage118_provider_isolated_commit_learning_v1"
STATUS="COMPLETE_PROVIDER_ISOLATED_COMMIT_LEARNING"
SOURCE117="59f645ac852adc35080cf94dbce9dd8708b4118ee27d8a6bad879ee3e476d3cc"
FEATURES=ev.FEATURES
TARGETS=("myopic","future")
SCORES=TARGETS+("myopic_constant","future_constant")
STUDIES=("ordinary_uniform","ordinary_stratified","pressure")
RULES=prior.RULES+SCORES
POLICIES=RULES+("myopic__rate_expected","future__rate_expected","future_oracle")
DESIGN=dict(ridge=1.,minimum_scale=1e-6,feature_clip=10.,epsilon=1e-12,bootstrap_seed=23180,bootstrap_samples=10000,
    primary_study="ordinary_uniform",primary="future_minus_myopic",training_population="stage117_uniform_only")
TABLES=("fold_scope","model_scope","fit_metrics","sample_predictions","policy_decisions","policy_summary","paired","support","video_effects")


def source_group(sequence):return s.s109.s106.source_group(sequence)


def provider_audit(splits,models):
    providers={}
    for split in splits:
        fold=split["fold"];train=set(split["train_groups"]);held=set(split["held_out_groups"])
        if fold in providers or not train or not held or train&held:raise ValueError("Overlapping provider groups")
        package=models[fold,"generator",0];generator=package["generator"]
        if package["split"]!=split or not set(generator["fit_groups"]+generator["calibration_groups"])<=train:raise ValueError("Generator source leakage")
        signatures=[package["fingerprint"]]
        for seed in s.DESIGN["seeds"]:
            model=models[fold,"generated_4",seed]
            if model["split"]!=split:raise ValueError("Scorer source leakage")
            signatures.append(model["fingerprint"])
        providers[fold]=dict(fold=fold,upstream_train_groups=sorted(train),upstream_held_groups=sorted(held),fingerprints=signatures)
    universe=set().union(*(set(p["upstream_held_groups"]) for p in providers.values()))
    if (sum(len(p["upstream_held_groups"]) for p in providers.values())!=len(universe)
        or any(set(p["upstream_train_groups"])!=universe-set(p["upstream_held_groups"]) for p in providers.values())):raise ValueError("Provider folds must partition the same source groups")
    return providers


def attach(row,origin,job):
    record={k:row[k] for k in ("anchor_id","sequence","cohort")}
    record.update(seed=int(row["seed"]),frame=int(row["frame"]),context=prior.prior.key(job),outer_fold=job["outer_fold"],
        group=source_group(row["sequence"]),origin=origin,uid=origin+"__"+row["anchor_id"])
    for k in (*FEATURES,"policy_future","q0_future","policy_current","q0_current","policy_low","q0_low","future_gain"):
        record[k]=float(row[k])
    if not np.isfinite([record[k] for k in (*FEATURES,"policy_future","q0_future","policy_current","q0_current","policy_low","q0_low","future_gain")]).all():raise ValueError("Nonfinite measured evidence")
    if any(not 0<=record[k]<=1 for k in ("policy_future","q0_future","policy_current","q0_current","policy_low","q0_low")):raise ValueError("Measured IoU/fraction outside unit interval")
    if not np.isclose(record["future_gain"],record["q0_future"]-record["policy_future"],rtol=0,atol=1e-12):raise ValueError("Paired outcome changed")
    record["current_role"]=prior.current_role(record["q0_current"],record["policy_current"])
    record["outcome"]=ev.outcome(record["q0_future"],record["policy_future"])
    if origin=="stage117":
        record.update(memberships=row["memberships"],uniform_weight=float(row["uniform_weight"]),stratified_weight=float(row["stratified_weight"]))
        memberships=record["memberships"].split("|")
        if (not memberships or len(set(memberships))!=len(memberships) or not set(memberships)<={"uniform","stratified"}
            or any(not np.isfinite(record[k+"_weight"]) or not 0<=record[k+"_weight"]<=1 or (record[k+"_weight"]>0)!=(k in memberships) for k in ("uniform","stratified"))):raise ValueError("Invalid independent study weights")
    else:record.update(memberships="pressure",uniform_weight=0.,stratified_weight=0.)
    if record["sequence"]!=job["spec"]["unit"]["sequence"] or record["seed"]!=job["seed"] or record["cohort"]!=job["spec"]["cohort"]:raise ValueError("Wrong source owner")
    return record


def load_records(root117,root116,p113):
    jobs={prior.prior.key(j):j for j in p113["jobs"] if j["arm"]=="generated_shared"}
    read=s.s103.prior.read_csv;records=[]
    for row in read(root117/"analysis/stage117_samples.csv"):
        if row["effective_intervention"]!="True":raise ValueError("Need the frozen physical-intervention cohort")
        records.append(attach(row,"stage117",jobs[row["context"]]))
    for row in read(root116/"analysis/stage116_anchor_evidence.csv"):
        context=row["anchor_id"].rsplit("__f",1)[0]
        records.append(attach(row,"stage115",jobs[context]))
    if len({r["uid"] for r in records})!=len(records):raise ValueError("Repeated measured point")
    return records


def training_rows(records,held,provider):
    group=held["group"];fold=held["outer_fold"];allowed=set(provider["upstream_held_groups"])-{group}
    if provider["fold"]!=fold or group not in provider["upstream_held_groups"] or group in provider["upstream_train_groups"]:raise ValueError("Evaluation group used by upstream provider")
    rows=[r for r in records if r["origin"]=="stage117" and "uniform" in r["memberships"].split("|") and r["group"] in allowed]
    if {r["group"] for r in rows}!=allowed or any(r["outer_fold"]!=fold for r in rows):raise ValueError("Need same-provider uniform fits; no cross-fold gate fitting")
    if any(r["group"]==group or r["origin"]!="stage117" for r in rows):raise ValueError("Gate fitting leakage")
    return sorted(rows,key=lambda r:r["uid"])


def arrays(rows):
    x=np.array([[r[k] for k in FEATURES] for r in rows],float)
    y=np.array([[r["q0_current"]-r["policy_current"],r["future_gain"]] for r in rows],float)
    if not rows or x.shape!=(len(rows),len(FEATURES)) or not np.isfinite(np.r_[x.ravel(),y.ravel()]).all():raise ValueError("Invalid gate fitting rows")
    return x,y,ev.balanced(rows)


def fit_signature(rows):
    x,y,w=arrays(rows)
    return io.seal(dict(ids=[r["uid"] for r in rows],x=x.tolist(),y=y.tolist(),weight=w.tolist()))["fingerprint"]


def fit_package(rows,held,provider):
    x,y,w=arrays(rows);center=w@x;scale=np.sqrt(w@((x-center)**2));scale[scale<DESIGN["minimum_scale"]]=1.
    z=np.clip((x-center)/scale,-DESIGN["feature_clip"],DESIGN["feature_clip"]);z=np.c_[z,np.ones(len(z))]
    penalty=DESIGN["ridge"]*np.diag([1.]*len(FEATURES)+[0.]);normal=z.T@(w[:,None]*z)+penalty
    coefficients=np.linalg.solve(normal,z.T@(w[:,None]*y));constant=w@y
    residual=np.max(np.abs(normal@coefficients-z.T@(w[:,None]*y)))
    if not np.isfinite(coefficients).all() or residual>1e-10:raise ValueError("Ridge normal equation failed")
    return io.seal(dict(schema=SCHEMA,design=DESIGN,features=list(FEATURES),held_group=held["group"],outer_fold=held["outer_fold"],provider=provider,
        fit_groups=sorted({r["group"] for r in rows}),fit_ids=[r["uid"] for r in rows],fit_examples=len(rows),fit_input_fingerprint=fit_signature(rows),
        norm=dict(center=center.tolist(),scale=scale.tolist()),coefficients=coefficients.tolist(),constant=constant.tolist(),normal_equation_max_error=float(residual)))


def predict(package,x):
    """Inference accepts only frozen fit parameters and public feature vectors."""
    x=np.asarray(x,float)
    if x.ndim!=2 or x.shape[1]!=len(FEATURES) or not np.isfinite(x).all():raise ValueError("Invalid public gate input")
    z=np.clip((x-np.asarray(package["norm"]["center"]))/package["norm"]["scale"],-DESIGN["feature_clip"],DESIGN["feature_clip"])
    linear=np.c_[z,np.ones(len(x))]@np.asarray(package["coefficients"])
    return np.c_[linear,np.tile(package["constant"],(len(x),1))]


def audit_package(package,rows,held,provider):
    if (package["schema"]!=SCHEMA or package["design"]!=DESIGN or package["features"]!=list(FEATURES) or package["held_group"]!=held["group"]
        or package["outer_fold"]!=held["outer_fold"] or package["provider"]!=provider or package["fit_input_fingerprint"]!=fit_signature(rows)
        or package["fit_ids"]!=[r["uid"] for r in rows] or package["fit_examples"]!=len(rows)
        or package["fit_groups"]!=sorted({r["group"] for r in rows}) or not np.isfinite(package["normal_equation_max_error"])
        or package["normal_equation_max_error"]>1e-10):raise ValueError("Gate fitting scope changed")
    for name,shape in (("coefficients",(len(FEATURES)+1,2)),("constant",(2,))):
        if np.asarray(package[name]).shape!=shape or not np.isfinite(package[name]).all():raise ValueError("Malformed CPU head")
    for name in ("center","scale"):
        if np.asarray(package["norm"][name]).shape!=(len(FEATURES),) or not np.isfinite(package["norm"][name]).all():raise ValueError("Malformed fit normalization")
    if (np.asarray(package["norm"]["scale"])<=0).any():raise ValueError("Malformed fit normalization")


def evaluate(records,prediction_rows):
    scores={r["uid"]:r for r in prediction_rows};base=[]
    for r in records:
        # Labels do not enter learned scores or any threshold.
        flags=np.r_[prior.decisions(r),[scores[r["uid"]][name]>DESIGN["epsilon"] for name in SCORES]]
        for study in (r["memberships"].split("|") if r["origin"]=="stage117" else ["pressure"]):
            study={"uniform":"ordinary_uniform","stratified":"ordinary_stratified"}.get(study,study)
            weight=r["uniform_weight"] if study=="ordinary_uniform" else r["stratified_weight"] if study=="ordinary_stratified" else 1.
            base.extend(dict(r,study=study,analysis_weight=weight,rule=rule,protect_probability=float(flags[j])) for j,rule in enumerate(RULES))
    expanded=list(base)
    for study in STUDIES:
        for head in TARGETS:
            units=defaultdict(list)
            for row in base:
                if row["study"]==study and row["rule"]==head:units[row["sequence"],row["seed"]].append(row)
            for rows in units.values():
                probability=sum(r["analysis_weight"]*r["protect_probability"] for r in rows)/sum(r["analysis_weight"] for r in rows)
                expanded.extend(dict(r,rule=head+"__rate_expected",protect_probability=probability) for r in rows)
    expanded.extend(dict(r,rule="future_oracle",protect_probability=float(r["future_gain"]>DESIGN["epsilon"])) for r in base if r["rule"]=="policy")
    for r in expanded:
        p=r["protect_probability"];d=r["future_gain"]
        r.update(future_iou=r["policy_future"]+p*d,low_fraction=r["policy_low"]+p*(r["q0_low"]-r["policy_low"]),
            realized_gain=p*d,harm_cost=p*max(-d,0.),missed_gain=(1-p)*max(d,0.),regret=max(d,0.)-p*d)
    return expanded


def summarize(decisions):
    summary=[];paired=[];support=[];video=[]
    for study in STUDIES:
        for cohort in ("all","historical","expanded"):
            pool=[r for r in decisions if r["study"]==study and (cohort=="all" or r["cohort"]==cohort)]
            if not pool:continue
            baseline=[r for r in pool if r["rule"]=="policy"]
            for label in ("all","good_q0_lost","medium_q0_lost","q0_rescue","other"):
                local=[r for r in baseline if label=="all" or r["current_role"]==label]
                support.append(dict(study=study,cohort=cohort,current_role=label,anchors=len(local),groups=len({r["group"] for r in local}),
                    positive=sum(r["future_gain"]>1e-12 for r in local),ties=sum(abs(r["future_gain"])<=1e-12 for r in local),negative=sum(r["future_gain"]< -1e-12 for r in local),
                    both_low=sum(r["outcome"]=="both_low" for r in local)))
            for label in (("all","good_q0_lost","medium_q0_lost","q0_rescue") if study=="pressure" else ("all",)):
                rows=[r for r in pool if label=="all" or r["current_role"]==label]
                if not rows:continue
                for rule in POLICIES:
                    local=[r for r in rows if r["rule"]==rule]
                    values={name:prior.weighted_macro(local,name) for name in ("future_iou","low_fraction","protect_probability","realized_gain","harm_cost","missed_gain","regret")}
                    summary.append(dict(study=study,cohort=cohort,current_role=label,rule=rule,anchors=len(local),videos=len(values["future_iou"]),
                        large_harm_anchors=sum(r["realized_gain"]< -.1 for r in local),worst_anchor_gain=min(r["realized_gain"] for r in local),
                        **{name:float(np.mean(list(v.values()))) for name,v in values.items()}))
                    video.extend(dict(study=study,cohort=cohort,current_role=label,rule=rule,sequence=seq,**{name:v[seq] for name,v in values.items()}) for seq in sorted(values["future_iou"]))
                treatment=prior.weighted_macro([r for r in rows if r["rule"]=="future"],"future_iou")
                rate=[r for r in rows if r["rule"]=="future__rate_expected" and 0<r["protect_probability"]<1]
                for control in ("myopic","policy","q0_all","continuity_pair","future_constant","future__rate_expected"):
                    reference=prior.weighted_macro([r for r in rows if r["rule"]==control],"future_iou");groups=sorted(treatment.keys()&reference.keys())
                    delta=np.array([treatment[g]-reference[g] for g in groups]);rng=np.random.default_rng(DESIGN["bootstrap_seed"])
                    boot=delta[rng.integers(0,len(delta),(DESIGN["bootstrap_samples"],len(delta)))].mean(1)
                    paired.append(dict(study=study,cohort=cohort,current_role=label,contrast="future_minus_"+control,videos=len(delta),mean_delta=float(delta.mean()),
                        median_delta=float(np.median(delta)),worst_video_delta=float(delta.min()),minimum_leave_one_video_out=float(np.min((delta.sum()-delta)/(len(delta)-1))) if len(delta)>1 else None,
                        ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),wins=int((delta>1e-12).sum()),ties=int((np.abs(delta)<=1e-12).sum()),losses=int((delta< -1e-12).sum()),
                        rate_informative_contexts=len({r["context"] for r in rate}),rate_informative_anchors=len(rate)))
    return summary,paired,support,video


def run(root,plan,records,owners,providers):
    artifacts={};tables={k:[] for k in TABLES};groups=sorted(owners)
    uniform=[r for r in records if r["origin"]=="stage117" and "uniform" in r["memberships"].split("|")]
    for group in groups:
        held=owners[group];provider=providers[held["outer_fold"]];fit=training_rows(uniform,held,provider)
        path=root/"models"/(group+".json")
        if not path.exists():c.publish_identical(path,fit_package(fit,held,provider))
        package=c.c66.sealed(path,SCHEMA);audit_package(package,fit,held,provider);artifacts[path.relative_to(root).as_posix()]=c.file_hash(path)
        tables["fold_scope"].append(dict(held_group=group,outer_fold=held["outer_fold"],fit_groups="|".join(package["fit_groups"]),fit_examples=len(fit),
            upstream_train_groups="|".join(provider["upstream_train_groups"]),upstream_held_groups="|".join(provider["upstream_held_groups"]),
            no_test_group_in_upstream_fit=True,no_test_group_in_gate_fit=True,no_pressure_in_gate_fit=True,model_fingerprint=package["fingerprint"]))
        x,y,w=arrays(fit);fitted=predict(package,x)
        for j,target in enumerate(TARGETS):
            for kind,params in (("ridge",len(FEATURES)+1),("constant",1)):
                index=j if kind=="ridge" else j+2
                tables["model_scope"].append(dict(held_group=group,outer_fold=held["outer_fold"],target=target,kind=kind,parameters=params,fit_examples=len(fit),fit_groups=len(package["fit_groups"])))
            tables["fit_metrics"].append(dict(held_group=group,target=target,fit_examples=len(fit),fit_mse=float(w@((fitted[:,j]-y[:,j])**2)),
                constant_mse=float(w@((fitted[:,j+2]-y[:,j])**2)),fit_positive_fraction=float(w@(y[:,j]>1e-12)),fit_negative_fraction=float(w@(y[:,j]< -1e-12)),
                normal_equation_max_error=package["normal_equation_max_error"]))
        held_rows=sorted([r for r in records if r["group"]==group],key=lambda r:r["uid"])
        x=np.array([[r[k] for k in FEATURES] for r in held_rows]);ids=[r["uid"] for r in held_rows]
        prediction=s.s109.cache(root,root/"predictions"/(group+".npz"),dict(plan_fingerprint=plan["fingerprint"],model_fingerprint=package["fingerprint"],ids=ids),
            lambda:dict(x=x,scores=predict(package,x)),artifacts)
        np.testing.assert_array_equal(prediction["x"],x)
        if prediction["scores"].shape!=(len(ids),4) or not np.isfinite(prediction["scores"]).all():raise ValueError("Invalid frozen predictions")
        # Predictions are committed before attaching held-out outcomes/roles.
        for i,row in enumerate(held_rows):
            tables["sample_predictions"].append(dict(row,model_fingerprint=package["fingerprint"],**{k:float(prediction["scores"][i,j]) for j,k in enumerate(SCORES)}))
        print("[Stage-118] "+group+": provider-isolated CPU heads and predictions saved",flush=True)
    tables["policy_decisions"]=evaluate(records,tables["sample_predictions"])
    tables["policy_summary"],tables["paired"],tables["support"],tables["video_effects"]=summarize(tables["policy_decisions"])
    publish(root,plan,tables,artifacts)


def publish(root,plan,tables,artifacts):
    e=plan["expected"];counts={k:len(v) for k,v in tables.items()}
    if (counts["fold_scope"]!=e["groups"] or counts["model_scope"]!=4*e["groups"] or counts["fit_metrics"]!=2*e["groups"]
        or counts["sample_predictions"]!=e["unique_points"] or counts["policy_decisions"]!=e["membership_points"]*len(POLICIES)
        or len(artifacts)!=3*e["groups"]
        or sum(r["origin"]=="stage115" for r in tables["sample_predictions"])!=e["pressure_points"]
        or sum(r["origin"]=="stage117" and "uniform" in r["memberships"].split("|") for r in tables["sample_predictions"])!=e["uniform_fit_points"]):raise ValueError("Incomplete provider-isolated gate audit")
    out=root/"analysis";out.mkdir(exist_ok=True);hashes={}
    for name,rows in tables.items():
        path=out/f"stage118_{name}.csv";s.s103.s86.publish_table(path,rows);hashes[path.name]=c.file_hash(path)
    io.verify(plan["source_sha256"]);io.verify(plan["code_sha256"])
    primary=next(r for r in tables["paired"] if r["study"]==DESIGN["primary_study"] and r["cohort"]=="all" and r["current_role"]=="all" and r["contrast"]==DESIGN["primary"])
    c.publish_identical(out/"stage118_decision.json",io.seal(dict(schema=SCHEMA,status=STATUS,plan_fingerprint=plan["fingerprint"],source_stage117_fingerprint=SOURCE117,
        expected=e,counts=counts,primary=primary,training_performed=True,fitted_ridge_heads=2*e["groups"],fitted_constant_controls=2*e["groups"],new_models=4*e["groups"],
        sgla_training_updates=0,optimizer_steps=0,image_reads=0,new_search_calls=0,new_block_calls=0,new_full_trajectories=0,confirmation_inference_calls=0,
        table_sha256=hashes,artifact_sha256=artifacts,warnings=[
            "Only Stage-117 uniform rows fit heads. Pressure labels never fit or calibrate a head.",
            "Each gate fits the other groups within its provider-held fold; its test group was excluded from that provider and the gate.",
            "This is consumed-development cross-fitting under fixed upstream providers, not fresh independent confirmation.",
            "Pressure is GT-selected and previously reviewed; it is reported separately and never pooled into ordinary utility.",
            "Gates choose one already measured action; no continuous deployment or complete-video AUC is measured.",
            "Regression scores are gain estimates, not calibrated correctness probabilities. No thresholds or hyperparameters are selected here."])))
    check_results(root)


def check_results(root):
    plan=c.c66.sealed(root/"manifest.json",SCHEMA);r=c.c66.sealed(root/"analysis/stage118_decision.json",SCHEMA);e=plan["expected"]
    if (plan["design"]!=DESIGN or plan["features"]!=list(FEATURES) or r["status"]!=STATUS or r["plan_fingerprint"]!=plan["fingerprint"]
        or plan["source_stage117_fingerprint"]!=SOURCE117 or r["source_stage117_fingerprint"]!=SOURCE117 or r["expected"]!=e or r["counts"]["fold_scope"]!=e["groups"]
        or r["counts"]["model_scope"]!=4*e["groups"] or r["counts"]["fit_metrics"]!=2*e["groups"]
        or r["counts"]["sample_predictions"]!=e["unique_points"] or r["counts"]["policy_decisions"]!=e["membership_points"]*len(POLICIES)
        or r["new_models"]!=4*e["groups"] or r["fitted_ridge_heads"]!=2*e["groups"] or r["fitted_constant_controls"]!=2*e["groups"]
        or r["primary"]["study"]!=DESIGN["primary_study"] or r["primary"]["contrast"]!=DESIGN["primary"] or r["primary"]["cohort"]!="all" or r["primary"]["current_role"]!="all"
        or not r["training_performed"] or len(r["artifact_sha256"])!=3*e["groups"] or set(r["table_sha256"])!={f"stage118_{k}.csv" for k in TABLES}
        or any(r[k]!=0 for k in ("sgla_training_updates","optimizer_steps","image_reads","new_search_calls","new_block_calls","new_full_trajectories","confirmation_inference_calls"))):raise ValueError("Incomplete Stage-118 result")
    io.verify(plan["source_sha256"]);io.verify(plan["code_sha256"])
    io.verify({str(root/"analysis"/k):v for k,v in r["table_sha256"].items()});io.verify({str(root/k):v for k,v in r["artifact_sha256"].items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__);names=("stage112","stage113","stage115","stage116","stage117")
    for name in (*names,"output_dir"):parser.add_argument("--"+name,type=Path,required=name=="output_dir")
    parser.add_argument("--check-results",action="store_true");args=parser.parse_args();root=args.output_dir.expanduser().resolve()
    if args.check_results:check_results(root);return
    if not all(getattr(args,k) for k in names):parser.error("Supply Stage-112/113/115/116/117 frozen roots")
    r112,r113,r115,r116,r117=paths=[getattr(args,k).expanduser().resolve() for k in names]
    if any(root==p or root in p.parents or p in root.parents for p in paths):parser.error("Use a separate Stage-118 root")
    with s.s109.s106.s93.writer_lock(root):
        if (root/"analysis/stage118_decision.json").exists():check_results(root);return
        if not (root/"manifest.json").exists() and any(p.name!="collect.lock" for p in root.iterdir()):raise ValueError("Use an empty Stage-118 root")
        prior.check_results(r117);ev.check_results(r116);prior.prior.check_results(r115);s.check_results(r113);s.s112.check_results(r112)
        p117=c.c66.sealed(r117/"manifest.json",prior.SCHEMA);r117report=c.c66.sealed(r117/"analysis/stage117_decision.json",prior.SCHEMA)
        p113=c.c66.sealed(r113/"manifest.json",s.SCHEMA);p112=c.c66.sealed(r112/"manifest.json",s.common.SCHEMA)
        for path,schema,expected in ((r117/"analysis/stage117_decision.json",prior.SCHEMA,SOURCE117),(r116/"analysis/stage116_decision.json",ev.SCHEMA,prior.SOURCE116),
            (r115/"analysis/stage115_decision.json",prior.prior.SCHEMA,ev.SOURCE115),(r113/"analysis/stage113_decision.json",s.SCHEMA,prior.prior.prior.SOURCE113),
            (r112/"analysis/stage112_decision.json",s.common.SCHEMA,s.SOURCE112)):
            if c.c66.sealed(path,schema)["fingerprint"]!=expected:raise ValueError("Frozen gate sources differ")
        models,_=s.load_frozen_models(r112,p112);providers=provider_audit(p112["folds"],models)
        records=load_records(r117,r116,p113);owners={}
        for row in records:
            key=row["group"];value=dict(group=key,outer_fold=row["outer_fold"],sequence=row["sequence"])
            if key in owners and owners[key]!=value:raise ValueError("Ambiguous source-group owner")
            owners[key]=value
        uniform=[r for r in records if r["origin"]=="stage117" and "uniform" in r["memberships"].split("|")]
        if (len(owners)!=28 or len(providers)!=4 or any(len(p["upstream_held_groups"])!=7 for p in providers.values()) or len(records)!=778 or len(uniform)!=448):raise ValueError("Frozen ordinary/pressure support differs")
        expected=dict(groups=28,unique_points=778,membership_points=791,uniform_fit_points=448,pressure_points=34)
        bindings=dict(p117["source_sha256"])
        for p in (r117/"manifest.json",r117/"analysis/stage117_decision.json"):bindings[str(p)]=c.file_hash(p)
        bindings.update({str(r117/"analysis"/k):v for k,v in r117report["table_sha256"].items()});bindings.update({str(r117/k):v for k,v in r117report["artifact_sha256"].items()})
        code=[Path(__file__).resolve(),Path(__file__).with_name("run_stage118_nested_commit.sh").resolve()]
        plan=io.seal(dict(schema=SCHEMA,design=DESIGN,features=list(FEATURES),expected=expected,owners=owners,providers=providers,source_stage117_fingerprint=SOURCE117,
            analysis_runtime=dict(python=platform.python_version(),numpy=np.__version__),source_sha256=bindings,code_sha256=p117["code_sha256"]|{str(p):c.file_hash(p) for p in code}))
        c.publish_identical(root/"manifest.json",plan);print("Stage-118 frozen: 28 provider-isolated fits; no image/GPU calls",flush=True)
        run(root,plan,records,owners,providers);print("Stage-118: "+STATUS+"; no method adoption gate",flush=True)


if __name__=="__main__":main()
