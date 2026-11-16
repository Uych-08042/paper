"""Audit fixed Stage-118 commit heads, fit support and signed score contributions."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import platform
import numpy as np
import stage118_nested_commit as prior

c,io=prior.c,prior.io
SCHEMA="stage119_fixed_commit_support_diagnostic_v1"
STATUS="COMPLETE_FIXED_COMMIT_SUPPORT_DIAGNOSTIC"
SOURCE118="8efa6dcf197a9249a3b034f5a70eba0f5c80b5eeeb4cbcccb2b2ee216ecede64"
DESIGN=dict(neighbor_groups=3,support_quantile=.95,material_gain=.1,epsilon=1e-12,atol=1e-10,rtol=1e-9,
    distance="fit_normalized_unclipped_RMS_one_nearest_point_per_distinct_source_group",
    reference="fit_only_leave_source_group_out_neighbor_distance",
    decomposition="fixed_normalization_and_normal_matrix_no_group_deletion_or_refit")
TABLES=("scope","fit_quality","fit_support","sample_audit","nearest_groups","score_sources","feature_terms","generalization","support_summary","focus")
LABELS=("policy_future","q0_future","policy_current","q0_current","policy_low","q0_low","future_gain")


def close(actual,expected,message):
    if not np.allclose(actual,expected,atol=DESIGN["atol"],rtol=DESIGN["rtol"]):raise ValueError(message)


def load_records(source,p118):
    rows=prior.s.s103.prior.read_csv(source/"analysis/stage118_sample_predictions.csv");records=[]
    for row in rows:
        r=dict(row)
        for k in (*prior.FEATURES,*prior.SCORES,*LABELS,"uniform_weight","stratified_weight"):r[k]=float(r[k])
        for k in ("seed","frame","outer_fold"):r[k]=int(r[k])
        if not np.isfinite([r[k] for k in (*prior.FEATURES,*prior.SCORES,*LABELS,"uniform_weight","stratified_weight")]).all():raise ValueError("Nonfinite source row")
        if r["origin"] not in ("stage117","stage115") or r["uid"]!=r["origin"]+"__"+r["anchor_id"]:raise ValueError("Source identity changed")
        owner=p118["owners"][r["group"]]
        if (r["group"]!=prior.source_group(r["sequence"]) or owner!=dict(group=r["group"],outer_fold=r["outer_fold"],sequence=r["sequence"])
            or r["context"]!=f"{r['sequence']}__generated_shared__s{r['seed']}"):raise ValueError("Source owner changed")
        if any(not 0<=r[k]<=1 for k in LABELS[:-1]):raise ValueError("Source metric outside unit interval")
        close(r["future_gain"],r["q0_future"]-r["policy_future"],"Changed paired future")
        if r["current_role"]!=prior.prior.current_role(r["q0_current"],r["policy_current"]) or r["outcome"]!=prior.ev.outcome(r["q0_future"],r["policy_future"]):raise ValueError("Source label changed")
        if r["origin"]=="stage117":
            memberships=r["memberships"].split("|")
            if not set(memberships)<={"uniform","stratified"} or len(set(memberships))!=len(memberships):raise ValueError("Invalid source membership")
            if any(not 0<=r[k+"_weight"]<=1 or (r[k+"_weight"]>0)!=(k in memberships) for k in ("uniform","stratified")):raise ValueError("Invalid source weights")
        elif r["memberships"]!="pressure" or r["uniform_weight"]!=0 or r["stratified_weight"]!=0:raise ValueError("Pressure entered ordinary study")
        records.append(r)
    e=p118["expected"]
    if (len(records)!=e["unique_points"] or len({r["uid"] for r in records})!=len(records)
        or sum(r["origin"]=="stage115" for r in records)!=e["pressure_points"]
        or sum(r["origin"]=="stage117" and "uniform" in r["memberships"].split("|") for r in records)!=e["uniform_fit_points"]
        or sum(len(r["memberships"].split("|")) for r in records)!=e["membership_points"]):raise ValueError("Incomplete source support")
    return records


def public_geometry(package,fit):
    """Only training public features and stored normalization enter distances."""
    x=np.array([[r[k] for k in prior.FEATURES] for r in fit]);norm=package["norm"]
    return (x-np.asarray(norm["center"]))/norm["scale"]


def neighbors(fit,z,query,exclude_group=None):
    """Distances and tie breaks never use current/future labels or policy scores."""
    query=np.asarray(query,float)
    if z.shape!=(len(fit),len(prior.FEATURES)) or query.shape!=(len(prior.FEATURES),) or not np.isfinite(np.r_[z.ravel(),query]).all():raise ValueError("Invalid public neighbor geometry")
    distance=np.sqrt(np.mean((z-query)**2,axis=1));nearest={}
    for i,r in enumerate(fit):
        if r["group"]==exclude_group:continue
        key=(float(distance[i]),r["uid"])
        if r["group"] not in nearest or key<nearest[r["group"]][0]:nearest[r["group"]]=(key,i)
    return [(i,key[0]) for key,i in sorted(nearest.values(),key=lambda pair:pair[0])]


def support_reference(package,fit):
    z=public_geometry(package,fit);groups={r["group"] for r in fit}
    k=min(DESIGN["neighbor_groups"],len(groups)-1)
    if k<1:raise ValueError("Need at least two fitting source groups")
    distances=[float(np.mean([d for _,d in neighbors(fit,z,z[i],r["group"])[:k]])) for i,r in enumerate(fit)]
    return z,k,float(np.quantile(distances,DESIGN["support_quantile"]))


def normal_audit(package,fit):
    x,y,w=prior.arrays(fit);z=np.c_[np.clip(public_geometry(package,fit),-prior.DESIGN["feature_clip"],prior.DESIGN["feature_clip"]),np.ones(len(fit))]
    center=w@x;scale=np.sqrt(w@((x-center)**2));scale[scale<prior.DESIGN["minimum_scale"]]=1.
    close(center,package["norm"]["center"],"Changed fit center");close(scale,package["norm"]["scale"],"Changed fit scale")
    normal=z.T@(w[:,None]*z)+prior.DESIGN["ridge"]*np.diag([1.]*len(prior.FEATURES)+[0.]);rhs=z.T@(w[:,None]*y)
    residual=float(np.max(np.abs(normal@np.asarray(package["coefficients"])-rhs)))
    close(normal@np.asarray(package["coefficients"]),rhs,"Fixed normal equation failed")
    close(package["constant"],w@y,"Changed fit constant")
    # This matrix audits the existing coefficients; it never fits replacement coefficients.
    mapping=np.linalg.solve(normal,z.T*w)
    return x,y,w,z,mapping,residual


def fit_quality(group,fit,y,w,predicted):
    rows=[]
    for j,target in enumerate(prior.TARGETS):
        label=y[:,j];valid=np.abs(label)>DESIGN["epsilon"];mass=float(w@valid)
        rows.append(dict(held_group=group,target=target,fit_examples=len(fit),fit_mse=float(w@((predicted[:,j]-label)**2)),
            constant_mse=float(w@((predicted[:,j+2]-label)**2)),label_mean=float(w@label),predicted_mean=float(w@predicted[:,j]),
            nonzero_labels=int(valid.sum()),positive_labels=int((label>DESIGN["epsilon"]).sum()),negative_labels=int((label< -DESIGN["epsilon"]).sum()),
            sign_accuracy_nontie=float(w@(valid&((predicted[:,j]>DESIGN["epsilon"])==(label>DESIGN["epsilon"]))))/mass if mass else None))
    return rows


def audit_point(row,package,fit,z,k,reference_p95,mapping,y):
    x=np.array([row[name] for name in prior.FEATURES]);raw=(x-np.asarray(package["norm"]["center"]))/package["norm"]["scale"]
    clipped=np.clip(raw,-prior.DESIGN["feature_clip"],prior.DESIGN["feature_clip"]);query=np.r_[clipped,1.]
    ordered=neighbors(fit,z,raw);local=ordered[:k];indices=[i for i,_ in local];gain=np.array([fit[i]["future_gain"] for i in indices])
    if len(ordered)!=len({r["group"] for r in fit}):raise ValueError("Missing distinct-group neighbors")
    distance=float(np.mean([d for _,d in local]));kernel=query@mapping;contribution=kernel[:,None]*y
    parts={g:contribution[[i for i,r in enumerate(fit) if r["group"]==g]].sum(0) for g in sorted({r["group"] for r in fit})}
    terms=query[:,None]*np.asarray(package["coefficients"]);score=np.array([row[t] for t in prior.TARGETS])
    close(kernel.sum(),1.,"Unpenalized intercept identity failed")
    close(contribution.sum(0),score,"Training-label score decomposition failed");close(terms.sum(0),score,"Feature score decomposition failed")
    errors=float(max(np.max(np.abs(contribution.sum(0)-score)),np.max(np.abs(terms.sum(0)-score))))
    positive=max(parts,key=lambda g:(parts[g][1],g));negative=min(parts,key=lambda g:(parts[g][1],g))
    names=(*prior.FEATURES,"intercept");pos_term=int(np.argmax(terms[:,1]));neg_term=int(np.argmin(terms[:,1]));d=row["future_gain"];accept=row["future"]>DESIGN["epsilon"]
    details=dict(row,neighbor_k=k,neighbor_mean_distance=distance,nearest_distance=ordered[0][1],fit_reference_p95=reference_p95,
        distance_beyond_reference=bool(distance>reference_p95+DESIGN["epsilon"]),distance_margin=distance-reference_p95,
        gate_clip_fraction=float((np.abs(raw)>prior.DESIGN["feature_clip"]).mean()),maximum_abs_standardized_feature=float(np.abs(raw).max()),
        neighbor_future_mean=float(gain.mean()),neighbor_future_min=float(gain.min()),neighbor_future_max=float(gain.max()),
        neighbor_sign_mixed=bool((gain>DESIGN["epsilon"]).any() and (gain< -DESIGN["epsilon"]).any()),
        neighbor_material_mixed=bool((gain>DESIGN["material_gain"]).any() and (gain< -DESIGN["material_gain"]).any()),
        neighbor_material_positive=int((gain>DESIGN["material_gain"]).sum()),neighbor_material_negative=int((gain< -DESIGN["material_gain"]).sum()),
        nearest_uids="|".join(fit[i]["uid"] for i in indices),nearest_future_gains="|".join(format(v,".12g") for v in gain),
        future_accept=accept,future_prediction_error=row["future"]-d,future_realized_gain=float(accept*d),
        wrong_commit=bool(accept and d< -DESIGN["material_gain"]),missed_gain=bool(not accept and d>DESIGN["material_gain"]),
        largest_group=positive,largest_group_contribution=float(parts[positive][1]),smallest_group=negative,smallest_group_contribution=float(parts[negative][1]),
        largest_feature=names[pos_term],largest_feature_contribution=float(terms[pos_term,1]),smallest_feature=names[neg_term],smallest_feature_contribution=float(terms[neg_term,1]),
        feature_absolute_sum=float(np.abs(terms[:,1]).sum()),group_absolute_sum=float(np.abs(list(parts.values()))[:,1].sum()),decomposition_max_error=errors)
    nearest_rows=[dict(uid=row["uid"],rank=rank,source_group=fit[i]["group"],neighbor_uid=fit[i]["uid"],neighbor_sequence=fit[i]["sequence"],neighbor_seed=fit[i]["seed"],
        neighbor_frame=fit[i]["frame"],distance=dist,in_local_k=rank<=k,neighbor_current_gain=fit[i]["q0_current"]-fit[i]["policy_current"],neighbor_future_gain=fit[i]["future_gain"])
        for rank,(i,dist) in enumerate(ordered,1)]
    sources=[dict(uid=row["uid"],source_group=g,myopic_contribution=float(part[0]),future_contribution=float(part[1]),
        signed_kernel_sum=float(kernel[[i for i,r in enumerate(fit) if r["group"]==g]].sum())) for g,part in parts.items()]
    features=[dict(uid=row["uid"],feature=name,standardized_value=float(query[j]),myopic_coefficient=package["coefficients"][j][0],future_coefficient=package["coefficients"][j][1],
        myopic_contribution=float(terms[j,0]),future_contribution=float(terms[j,1])) for j,name in enumerate(names)]
    return details,nearest_rows,sources,features


def exposure(samples):
    rows=[]
    for r in samples:
        for membership in r["memberships"].split("|"):
            study={"uniform":"ordinary_uniform","stratified":"ordinary_stratified","pressure":"pressure"}[membership]
            rows.append(dict(r,study=study,analysis_weight=1. if membership=="pressure" else r[membership+"_weight"]))
    return rows


def summaries(samples):
    generalization=[];support=[];expanded=exposure(samples);macro=prior.prior.weighted_macro
    subsets=("all","material_positive","material_negative","wrong_commit","missed_gain","current_rescue")
    for study in prior.STUDIES:
        for cohort in ("all","historical","expanded"):
            pool=[r for r in expanded if r["study"]==study and (cohort=="all" or r["cohort"]==cohort)]
            if not pool:continue
            for target in prior.TARGETS:
                local=[]
                for r in pool:
                    label=r["future_gain"] if target=="future" else r["q0_current"]-r["policy_current"]
                    local.append(dict(r,mse=(r[target]-label)**2,constant_mse=(r[target+"_constant"]-label)**2,
                        realized_gain=float(r[target]>DESIGN["epsilon"])*r["future_gain"],harm_cost=float(r[target]>DESIGN["epsilon"])*max(-r["future_gain"],0.),
                        lost_gain=float(r[target]<=DESIGN["epsilon"])*max(r["future_gain"],0.)))
                values={name:float(np.mean(list(macro(local,name).values()))) for name in ("mse","constant_mse","realized_gain","harm_cost","lost_gain")}
                generalization.append(dict(study=study,cohort=cohort,target=target,anchors=len(pool),videos=len({r["sequence"] for r in pool}),
                    **values,mse_relative_to_constant=values["mse"]/values["constant_mse"] if values["constant_mse"] else None))
            for subset in subsets:
                local=[r for r in pool if subset=="all" or (subset=="material_positive" and r["future_gain"]>DESIGN["material_gain"])
                    or (subset=="material_negative" and r["future_gain"]< -DESIGN["material_gain"])
                    or (subset in ("wrong_commit","missed_gain") and r[subset]) or (subset=="current_rescue" and r["current_role"]=="q0_rescue")]
                values={name:float(np.mean(list(macro(local,name).values()))) if local else None for name in (
                    "distance_beyond_reference","gate_clip_fraction","neighbor_mean_distance","neighbor_sign_mixed","neighbor_material_mixed","neighbor_future_mean","future_prediction_error","future_realized_gain")}
                support.append(dict(study=study,cohort=cohort,subset=subset,anchors=len(local),videos=len({r["sequence"] for r in local}),**values))
    return generalization,support


def run(root,source,p118,plan,records):
    tables={k:[] for k in TABLES};recomputed=fit_total=0
    providers={int(k):v for k,v in p118["providers"].items()}
    for group,held in sorted(p118["owners"].items()):
        fit=prior.training_rows(records,held,providers[held["outer_fold"]]);package=c.c66.sealed(source/"models"/(group+".json"),prior.SCHEMA)
        prior.audit_package(package,fit,held,providers[held["outer_fold"]]);x,y,w,z,mapping,residual=normal_audit(package,fit)
        fit_pred=prior.predict(package,x);tables["fit_quality"].extend(fit_quality(group,fit,y,w,fit_pred));raw,k,p95=support_reference(package,fit)
        part=sorted([r for r in records if r["group"]==group],key=lambda r:r["uid"]);xh=np.array([[r[k] for k in prior.FEATURES] for r in part])
        if any(r["model_fingerprint"]!=package["fingerprint"] for r in part):raise ValueError("Changed point model binding")
        metadata=dict(plan_fingerprint=p118["fingerprint"],model_fingerprint=package["fingerprint"],ids=[r["uid"] for r in part])
        saved=c.read_cache(source/"predictions"/(group+".npz"),metadata)
        if saved is None:raise ValueError("Missing fixed predictions")
        np.testing.assert_array_equal(saved["x"],xh);np.testing.assert_array_equal(saved["scores"],[[r[k] for k in prior.SCORES] for r in part])
        predicted=prior.predict(package,xh);close(predicted,saved["scores"],"Frozen CPU score replay failed")
        fit_total+=len(fit);recomputed+=len(fit)+len(part)
        tables["scope"].append(dict(held_group=group,outer_fold=held["outer_fold"],fit_groups="|".join(package["fit_groups"]),fit_examples=len(fit),
            evaluated_points=len(part),model_fingerprint=package["fingerprint"],maximum_replay_error=float(np.max(np.abs(predicted-saved["scores"]))),
            normal_equation_max_error=residual,no_test_group_in_fit=group not in package["fit_groups"],no_pressure_in_fit=all(r["origin"]=="stage117" for r in fit)))
        tables["fit_support"].append(dict(held_group=group,fit_examples=len(fit),fit_groups=len(package["fit_groups"]),neighbor_k=k,fit_reference_p95=p95,
            material_positive=sum(r["future_gain"]>DESIGN["material_gain"] for r in fit),material_negative=sum(r["future_gain"]< -DESIGN["material_gain"] for r in fit),
            material_positive_groups=len({r["group"] for r in fit if r["future_gain"]>DESIGN["material_gain"]}),
            material_negative_groups=len({r["group"] for r in fit if r["future_gain"]< -DESIGN["material_gain"]}),
            current_rescue=sum(r["current_role"]=="q0_rescue" for r in fit),fit_clip_fraction=float((np.abs(raw)>prior.DESIGN["feature_clip"]).mean())))
        for row in part:
            detail,nearest,sources,features=audit_point(row,package,fit,raw,k,p95,mapping,y)
            tables["sample_audit"].append(detail);tables["nearest_groups"].extend(nearest);tables["score_sources"].extend(sources);tables["feature_terms"].extend(features)
            reasons=[name for name in ("wrong_commit","missed_gain") if detail[name]]
            if row["origin"]=="stage115" and row["current_role"]=="q0_rescue":reasons.append("rescue_pressure")
            if reasons:tables["focus"].append(dict(detail,focus_reason="|".join(reasons)))
        print(f"[Stage-119] {group}: fixed models, distinct-group support and score terms audited",flush=True)
    tables["generalization"],tables["support_summary"]=summaries(tables["sample_audit"])
    publish(root,plan,tables,recomputed,fit_total)


def publish(root,plan,tables,recomputed,fit_total):
    e=plan["expected"];counts={k:len(v) for k,v in tables.items()}
    if (counts["scope"]!=e["groups"] or counts["fit_support"]!=e["groups"] or counts["fit_quality"]!=2*e["groups"] or counts["sample_audit"]!=e["unique_points"]
        or counts["feature_terms"]!=(len(prior.FEATURES)+1)*e["unique_points"] or counts["nearest_groups"]!=e["neighbor_rows"] or counts["score_sources"]!=e["neighbor_rows"]
        or fit_total!=e["fit_rows"] or recomputed!=fit_total+e["unique_points"]):raise ValueError("Incomplete fixed-head diagnostic")
    out=root/"analysis";out.mkdir(exist_ok=True);hashes={}
    for name,rows in tables.items():
        path=out/f"stage119_{name}.csv";prior.s.s103.s86.publish_table(path,rows);hashes[path.name]=c.file_hash(path)
    io.verify(plan["source_sha256"]);io.verify(plan["code_sha256"])
    c.publish_identical(out/"stage119_decision.json",io.seal(dict(schema=SCHEMA,status=STATUS,plan_fingerprint=plan["fingerprint"],source_stage118_fingerprint=SOURCE118,
        expected=e,counts=counts,fixed_cpu_prediction_rows_recomputed=recomputed,maximum_decomposition_error=max(r["decomposition_max_error"] for r in tables["sample_audit"]),
        new_models=0,training_performed=False,training_updates=0,image_reads=0,new_search_calls=0,new_block_calls=0,new_full_trajectories=0,confirmation_inference_calls=0,
        table_sha256=hashes,warnings=["Distances and fit-only support references never use held-out outcome labels.",
            "Neighbors summarize training support, not a new decision rule or a proof of feature indistinguishability.",
            "Signed contributions are exact algebra under fixed normalization and normal matrix, not causal effects of deleting a group.",
            "Material errors and pressure roles use GT only for diagnostic reporting. No thresholds or models are selected.",
            "Consumed development only. No adoption gate, new complete-video AUC or confirmation run."])))
    check_results(root)


def check_results(root):
    plan=c.c66.sealed(root/"manifest.json",SCHEMA);report=c.c66.sealed(root/"analysis/stage119_decision.json",SCHEMA);e=plan["expected"];n=report["counts"]
    if (plan["design"]!=DESIGN or plan["source_stage118_fingerprint"]!=SOURCE118 or report["source_stage118_fingerprint"]!=SOURCE118
        or report["plan_fingerprint"]!=plan["fingerprint"] or report["status"]!=STATUS or report["expected"]!=e or report["training_performed"]
        or n["scope"]!=e["groups"] or n["fit_support"]!=e["groups"] or n["fit_quality"]!=2*e["groups"] or n["sample_audit"]!=e["unique_points"]
        or n["nearest_groups"]!=e["neighbor_rows"] or n["score_sources"]!=e["neighbor_rows"] or n["feature_terms"]!=14*e["unique_points"]
        or report["fixed_cpu_prediction_rows_recomputed"]!=e["fit_rows"]+e["unique_points"] or not np.isfinite(report["maximum_decomposition_error"]) or report["maximum_decomposition_error"]>DESIGN["atol"]
        or set(report["table_sha256"])!={f"stage119_{k}.csv" for k in TABLES}
        or any(report[k]!=0 for k in ("new_models","training_updates","image_reads","new_search_calls","new_block_calls","new_full_trajectories","confirmation_inference_calls"))):raise ValueError("Incomplete Stage-119 diagnostic")
    io.verify(plan["source_sha256"]);io.verify(plan["code_sha256"]);io.verify({str(root/"analysis"/k):v for k,v in report["table_sha256"].items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--stage118",type=Path);parser.add_argument("--output_dir",type=Path,required=True)
    parser.add_argument("--check-results",action="store_true");args=parser.parse_args();root=args.output_dir.expanduser().resolve()
    if args.check_results:check_results(root);return
    if not args.stage118:parser.error("Supply the completed frozen Stage-118 root")
    source=args.stage118.expanduser().resolve()
    if root==source or root in source.parents or source in root.parents:parser.error("Use a separate Stage-119 root")
    with prior.s.s109.s106.s93.writer_lock(root):
        if (root/"analysis/stage119_decision.json").exists():check_results(root);return
        if not (root/"manifest.json").exists() and any(p.name!="collect.lock" for p in root.iterdir()):raise ValueError("Use an empty Stage-119 root")
        prior.check_results(source);p118=c.c66.sealed(source/"manifest.json",prior.SCHEMA);r118=c.c66.sealed(source/"analysis/stage118_decision.json",prior.SCHEMA)
        if r118["fingerprint"]!=SOURCE118:raise ValueError("Frozen Stage-118 result differs")
        records=load_records(source,p118);owners=p118["owners"]
        if len(owners)!=28 or len(records)!=778:raise ValueError("Need the declared 28-group Stage-118 run")
        expected=dict(groups=28,unique_points=778,fit_rows=2688,neighbor_rows=4668)
        bindings=dict(p118["source_sha256"])
        for path in (source/"manifest.json",source/"analysis/stage118_decision.json"):bindings[str(path)]=c.file_hash(path)
        bindings.update({str(source/"analysis"/k):v for k,v in r118["table_sha256"].items()});bindings.update({str(source/k):v for k,v in r118["artifact_sha256"].items()})
        code=[Path(__file__).resolve(),Path(__file__).with_name("run_stage119_commit_support.sh").resolve()]
        plan=io.seal(dict(schema=SCHEMA,design=DESIGN,source_stage118_fingerprint=SOURCE118,expected=expected,
            analysis_runtime=dict(python=platform.python_version(),numpy=np.__version__),source_sha256=bindings,code_sha256=p118["code_sha256"]|{str(p):c.file_hash(p) for p in code}))
        c.publish_identical(root/"manifest.json",plan);run(root,source,p118,plan,records);print("Stage-119: "+STATUS+"; no new training or GPU inference",flush=True)


if __name__=="__main__":main()
