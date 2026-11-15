"""Fixed relational-feature x mode-interaction experiment on cached commit evidence."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import stage107_gate_support_audit as s107


s106, c, io = s107.s106, s107.c, s107.io
SCHEMA = "stage108_relational_mode_commit_gate_v1"
STATUS = "COMPLETE_RELATIONAL_COMMIT_GATE_DEVELOPMENT"
SOURCE107 = "7782ea245f6f17fe2b0bd3575edcadcee3da12a23a9bafbb2987d24a465cd0ba"
ARMS = ("baseline", "relation", "mode", "relation_mode")
RELATIONS = ("center_conflict", "size_conflict", "motion_preference", "shape_preference",
    "center_x_anchor", "center_x_reference", "center_x_hann", "center_x_ppo",
    "size_x_anchor", "size_x_reference", "motion_x_anchor", "motion_x_reference",
    "shape_x_anchor", "shape_x_reference", "write_x_anchor", "write_x_reference")
DESIGN = dict(ridge=1., threshold=0., minimum_scale=1e-8, feature_version=1,
    split="leave_one_source_group_out", weighting=s106.DESIGN["weighting"],
    mode_product="signed_mode_times_fit_standardized_nonmode_stem",
    primary="relation_mode__future_minus_baseline__future")
POLICIES = ("keep", "always_pair", "future_oracle")+tuple(a+"__"+h for a in ARMS for h in s106.HEADS)+tuple(a+"__rate_expected" for a in ARMS)
TABLES = ("model_scope", "training_summary", "training_macro", "policy_summary", "paired", "state_predictions",
          "sample_decisions", "focus", "exposure_units", "exposure_summary")


def stem_features(arm):
    if arm not in ARMS:
        raise ValueError("Unknown representation")
    return list(s106.FEATURES)+(list(RELATIONS) if arm in ("relation", "relation_mode") else [])


def feature_names(arm):
    stem = stem_features(arm)
    return stem+(["mode_x_"+n for n in stem if n != "mode_pairwise"] if arm in ("mode", "relation_mode") else [])


def relation_features(x):
    """Current-state whitelist only; no labels, role, group or future argument."""
    x = np.asarray(x, float)
    if x.ndim not in (1, 2) or x.shape[-1] != len(s106.FEATURES) or not np.isfinite(x).all():
        raise ValueError("Invalid original feature vector")
    get = lambda name: x[..., s106.FEATURES.index(name)]
    center = np.log1p(np.hypot(get("pair_minus_four_dx"), get("pair_minus_four_dy")))
    size = np.log1p(np.hypot(get("pair_over_four_log_w"), get("pair_over_four_log_h")))
    motion = np.tanh(get("four_extrapolation_error"))-np.tanh(get("pair_extrapolation_error"))
    shape = np.tanh(np.hypot(get("four_vs_median_log_w"), get("four_vs_median_log_h")))-np.tanh(np.hypot(get("pair_vs_median_log_w"), get("pair_vs_median_log_h")))
    anchor, reference = get("anchor_score_delta"), get("reference_score_delta")
    write = get("pair_memory_write")-get("four_memory_write")
    out = np.stack((center, size, motion, shape, center*anchor, center*reference, center*get("hann_peak_delta"),
        center*get("ppo_probability_delta"), size*anchor, size*reference, motion*anchor, motion*reference,
        shape*anchor, shape*reference, write*anchor, write*reference), axis=-1)
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite relational feature")
    return out


def stem(x, arm):
    x = np.asarray(x, float)
    if x.ndim not in (1, 2) or x.shape[-1] != len(s106.FEATURES) or not np.isfinite(x).all():
        raise ValueError("Invalid original feature vector")
    mode = x[..., s106.FEATURES.index("mode_pairwise")]
    if not np.isin(mode, (0., 1.)).all():
        raise ValueError("Invalid verifier mode bit")
    return np.concatenate((x, relation_features(x)), axis=-1) if arm in ("relation", "relation_mode") else x.copy()


def mapping_parameters(fit, arm):
    values = stem(np.asarray([r["x"] for r in fit]), arm)
    w = s106.balanced_weights(fit)
    mean = w @ values; scale = np.sqrt(w @ ((values-mean)**2))
    scale[scale < DESIGN["minimum_scale"]] = 1.
    return dict(center=mean.tolist(), scale=scale.tolist())


def transform(x, arm, mapping):
    values = stem(x, arm)
    if arm not in ("mode", "relation_mode"):
        return values
    if np.asarray(mapping["center"]).shape != (len(stem_features(arm)),) or np.asarray(mapping["scale"]).shape != (len(stem_features(arm)),):
        raise ValueError("Incorrect fit-only interaction mapping")
    z = (values-np.asarray(mapping["center"]))/mapping["scale"]
    indices = [i for i,n in enumerate(stem_features(arm)) if n != "mode_pairwise"]
    signed = 2*values[..., s106.FEATURES.index("mode_pairwise")]-1
    return np.concatenate((values, signed[..., None]*z[..., indices]), axis=-1)


def training_rows(samples, group):
    return s107.common_support([r for r in samples if r["group"] != group])


def fit_inputs(samples, group, arm, head):
    if arm not in ARMS[1:] or head not in s106.HEADS:
        raise ValueError("Fit new representations only; baseline is reused")
    fit = training_rows(samples, group)
    if not fit:
        raise ValueError("No common training support")
    mapping = mapping_parameters(fit, arm)
    inputs = [{k:r[k] for k in ("group", "sequence", "seed", "source_arm", "frame")} |
              dict(x=r["x"].tolist(), y=float(r[head])) for r in fit]
    receipt = io.seal(dict(design=DESIGN, representation=arm, head=head, held_out_group=group,
        features=feature_names(arm), fit_rows=inputs))["fingerprint"]
    return fit, mapping, receipt


def fit_gate(samples, group, arm, head, prepared=None):
    fit, mapping, receipt = fit_inputs(samples,group,arm,head) if prepared is None else prepared
    x = transform(np.asarray([r["x"] for r in fit]), arm, mapping)
    y, w = np.asarray([r[head] for r in fit]), s106.balanced_weights(fit)
    mean = w @ x; scale = np.sqrt(w @ ((x-mean)**2)); scale[scale < DESIGN["minimum_scale"]] = 1.
    z = (x-mean)/scale; intercept = float(w @ y)
    beta = np.linalg.solve(z.T @ (w[:, None]*z)+DESIGN["ridge"]*np.eye(x.shape[1]), z.T @ (w*(y-intercept)))
    return io.seal(dict(schema=SCHEMA, representation=arm, head=head, held_out_group=group,
        features=feature_names(arm), design=DESIGN, supported=True, mapping=mapping, input_fingerprint=receipt,
        fit_rows=len(fit), fit_groups=sorted({r["group"] for r in fit}),
        fit_contexts=len({(r["group"],r["seed"],r["source_arm"]) for r in fit}),
        center=mean.tolist(), scale=scale.tolist(), minimum=x.min(0).tolist(), maximum=x.max(0).tolist(),
        beta=beta.tolist(), intercept=intercept))


def predict_batch(model, raw, eligible, arm):
    raw, eligible = np.asarray(raw, float), np.asarray(eligible, bool)
    if raw.ndim != 2 or raw.shape[0] != len(eligible):
        raise ValueError("Prediction shape mismatch")
    values = transform(raw, arm, model.get("mapping", {}))
    if values.shape[1] != len(model["features"]) or model["features"] != feature_names(arm):
        raise ValueError("Model feature schema mismatch")
    z = (values-model["center"])/model["scale"]
    score = np.where(eligible, z @ model["beta"]+model["intercept"], 0.)
    return dict(score=score, commit=eligible & (score > DESIGN["threshold"]),
        outside_fit_range=eligible & ((values < model["minimum"]) | (values > model["maximum"])).any(1),
        max_abs_z=np.where(eligible, np.abs(z).max(1), 0.))


def audit_fit(model, prepared, arm, head):
    fit, mapping, _ = prepared
    for key in ("center","scale"):
        s107.close(model["mapping"][key],mapping[key])
    x=transform(np.asarray([r["x"] for r in fit]),arm,model["mapping"])
    w=s106.balanced_weights(fit);y=np.asarray([r[head] for r in fit])
    mean=w @ x;scale=np.sqrt(w @ ((x-mean)**2));scale[scale<DESIGN["minimum_scale"]]=1.
    s107.close(model["center"],mean);s107.close(model["scale"],scale)
    s107.close(model["minimum"],x.min(0));s107.close(model["maximum"],x.max(0));s107.close(model["intercept"],float(w @ y))
    z=(x-model["center"])/model["scale"]
    rhs=z.T @ (w*(y-model["intercept"]))
    residual=(z.T @ (w[:,None]*z)+DESIGN["ridge"]*np.eye(x.shape[1])) @ np.asarray(model["beta"])-rhs
    value=float(np.abs(residual).max()/max(1.,np.abs(rhs).max()))
    if not np.isfinite(value) or value>s107.DESIGN["verification_rtol"]:raise ValueError("New gate violates its fixed normal equations")
    return value


def evaluate(model, rows, arm):
    valid = s107.common_support(rows)
    result = dict(rows=len(rows), eligible_rows=len(valid), groups=len({r["group"] for r in valid}))
    if not valid:
        return result | {m:None for m in s107.EVAL_METRICS}
    w = s106.balanced_weights(valid)
    score = predict_batch(model, [r["x"] for r in valid], np.ones(len(valid),bool), arm)["score"]
    y = np.asarray([r[model["head"]] for r in valid]); future = np.asarray([r["future"] for r in valid])
    p = (score > 0).astype(float); variance = float(w @ ((y-w @ y)**2)); mse = float(w @ ((score-y)**2))
    bins = defaultdict(list)
    for i,r in enumerate(valid): bins[r["group"],r["seed"],r["source_arm"]].append(i)
    rate = np.zeros(len(valid))
    for indices in bins.values(): rate[indices] = p[indices].mean()
    conditional = lambda mask: float(w[mask] @ p[mask]/w[mask].sum()) if mask.any() else None
    oracle = float(w @ np.maximum(future,0))
    return result | dict(target_mse=mse, intercept_mse=float(w @ ((model["intercept"]-y)**2)),
        target_mae=float(w @ np.abs(score-y)), target_r2=1-mse/variance if variance > 0 else None,
        target_positive_acceptance=conditional(y>0), target_negative_acceptance=conditional(y<0),
        future_positive_acceptance=conditional(future>0), future_negative_acceptance=conditional(future<0),
        future_utility=float(w @ (p*future)), always_future_utility=float(w @ future), rate_future_utility=float(w @ (rate*future)),
        oracle_future_utility=oracle, future_regret=oracle-float(w @ (p*future)), commit_rate=float(w @ p))


def build_models(root, source, samples, groups):
    baseline = s107.load_models(source, groups)
    models, rows, training, hashes = {}, [], [], {}
    for group in groups:
        for arm in ARMS:
            for head in s106.HEADS:
                if arm == "baseline":
                    model = baseline[group, head]
                    _,_,residual = s107.audit_model(model, samples)
                    path = source/"models"/(group+"__"+head+".json")
                else:
                    prepared = fit_inputs(samples,group,arm,head)
                    path = root/"models"/(group+"__"+arm+"__"+head+".json")
                    if path.exists():
                        model = c.c66.sealed(path, SCHEMA)
                        expected=dict(representation=arm,head=head,held_out_group=group,features=feature_names(arm),design=DESIGN,
                            input_fingerprint=prepared[2],fit_rows=len(prepared[0]),
                            fit_groups=sorted({r["group"] for r in prepared[0]}))
                        if any(model[k]!=v for k,v in expected.items()): raise ValueError("Committed model input changed")
                    else:
                        computed = fit_gate(samples,group,arm,head,prepared)
                        path.parent.mkdir(exist_ok=True); c.publish_identical(path, computed); model = computed
                    residual=audit_fit(model,prepared,arm,head)
                    hashes[path.relative_to(root).as_posix()] = c.file_hash(path)
                if group in model["fit_groups"]: raise ValueError("Held-out source entered fitting")
                models[group,arm,head] = model
                rows.append(dict(held_out_group=group, representation=arm, head=head, features=len(model["features"]),
                    provenance="reused_frozen_source" if arm == "baseline" else "new_cpu_ridge_fit",
                    fit_groups=len(model["fit_groups"]), fit_rows=model["fit_rows"], fit_contexts=model["fit_contexts"],
                    input_fingerprint=model["input_fingerprint"], model_fingerprint=model["fingerprint"],
                    held_out_excluded=True, normal_equation_residual=residual, model_path=str(path)))
                held = [r for r in samples if r["group"] == group]
                for phase, part in (("fit_resubstitution", training_rows(samples,group)), ("held_out",held)):
                    training.append(dict(held_out_group=group, representation=arm, head=head, phase=phase, **evaluate(model,part,arm)))
            print(f"[Stage-108 fit] {group} {arm}: paired targets complete", flush=True)
    return models, rows, training, hashes


def predict_samples(samples, models):
    rows = []
    for group in sorted({r["group"] for r in samples}):
        held = [r for r in samples if r["group"] == group]
        for arm in ARMS:
            for head in s106.HEADS:
                model = models[group,arm,head]
                prediction = predict_batch(model,[r["x"] for r in held],[r["eligible"] for r in held],arm)
                for i,r in enumerate(held):
                    rows.append({k:r[k] for k in s107.IDENTITY} | dict(representation=arm,head=head,eligible=r["eligible"],reason=r["reason"],
                        score=float(prediction["score"][i]),commit=bool(prediction["commit"][i]),
                        outside_fit_range=bool(prediction["outside_fit_range"][i]),max_abs_z=float(prediction["max_abs_z"][i]),
                        measured_future=float(r["future"]) if np.isfinite(r["future"]) else None,
                        measured_myopic=float(r["myopic"]) if np.isfinite(r["myopic"]) else None))
    return rows


def decisions(samples, predictions):
    bank = {(s107.sample_key(r),r["representation"],r["head"]):r for r in predictions}
    contexts = defaultdict(list)
    for r in samples: contexts[r["group"],r["sequence"],r["seed"],r["source_arm"]].append(r)
    out = []
    for part in contexts.values():
        rates = {a:sum(bank[s107.sample_key(r),a,"future"]["commit"] for r in part)/sum(r["eligible"] for r in part)
                 if any(r["eligible"] for r in part) else 0. for a in ARMS}
        for r in part:
            key = s107.sample_key(r); valid = np.isfinite(r["future"]); delta = float(r["future"]) if valid else None
            choices = dict(keep=0.,always_pair=float(r["eligible"]),future_oracle=float(valid and r["eligible"] and delta>0))
            choices.update({a+"__"+h:float(bank[key,a,h]["commit"]) for a in ARMS for h in s106.HEADS})
            choices.update({a+"__rate_expected":rates[a]*float(r["eligible"]) for a in ARMS})
            for policy in POLICIES:
                p = choices[policy]
                out.append({k:r[k] for k in s107.IDENTITY} | dict(policy=policy,eligible=r["eligible"],commit_probability=p,measured_delta=delta,
                    utility=p*delta if valid else None,harmful_commit=p*float(delta<0) if valid else None,
                    missed_positive_utility=(1-p)*max(delta,0) if valid else None,
                    oracle_regret=float(r["eligible"])*max(delta,0)-p*delta if valid else None))
    return out


def summaries(rows):
    metrics = ("eligible","commit_probability","utility","harmful_commit","missed_positive_utility","oracle_regret")
    policy, _ = s106.summarize(rows,POLICIES,metrics)
    paired = []
    for cohort in ("all","historical","expanded"):
        for mode in ("all",*s106.s103.MODES[1:]):
            local = [r for r in rows if (cohort=="all" or r["cohort"]==cohort) and (mode=="all" or r["source_arm"]==mode)]
            if not local: continue
            values = {p:s106.macro([r for r in local if r["policy"]==p],"utility") for p in POLICIES}
            contrasts = {"relation_mode__future_minus_"+p:{"relation_mode__future":1.,p:-1.}
                for p in ("baseline__future","relation__future","mode__future","relation_mode__myopic","always_pair","relation_mode__rate_expected","keep")}
            contrasts.update(relation_by_mode_interaction={"relation_mode__future":1.,"relation__future":-1.,"mode__future":-1.,"baseline__future":1.},
                relation_order_average={"relation__future":.5,"baseline__future":-.5,"relation_mode__future":.5,"mode__future":-.5},
                mode_order_average={"mode__future":.5,"baseline__future":-.5,"relation_mode__future":.5,"relation__future":-.5})
            for name, coefficients in contrasts.items():
                groups = sorted(set.intersection(*(set(values[p]) for p in coefficients)))
                delta = np.asarray([sum(v*values[p][g] for p,v in coefficients.items()) for g in groups])
                paired.append(dict(cohort=cohort,source_arm=mode,contrast=name,groups=len(delta),mean_delta=float(delta.mean()) if len(delta) else None,
                    median_delta=float(np.median(delta)) if len(delta) else None,wins=int((delta>1e-12).sum()),ties=int((np.abs(delta)<=1e-12).sum()),
                    losses=int((delta < -1e-12).sum()),worst_group_delta=float(delta.min()) if len(delta) else None,
                    minimum_leave_one_group_out=float(min((delta.sum()-x)/(len(delta)-1) for x in delta)) if len(delta)>1 else None))
    return policy, paired


def exposure(root, source, old, report, models, plan):
    rows, hashes = [], {}
    source_units = {(r["sequence"],int(r["seed"]),r["source_arm"],r["policy"]):r for r in s107.read_csv(source/"analysis/stage106_exposure_units.csv")}
    states = 0
    for relative in sorted(n for n in report["artifact_sha256"] if n.startswith("exposure/") and n.endswith(".npz")):
        path = source/relative
        meta = c.c66.sealed(path.with_suffix(".receipt.json"),c.SCHEMA)["metadata"]
        if meta["plan_fingerprint"] != old["fingerprint"] or meta["kind"] != "frozen_source_exposure": raise ValueError("Mixed source exposure")
        data = s106.required(path,meta); unit,mode=meta["unit"],meta["source_arm"]
        if path.name != c.key(unit)+"__"+mode+".npz": raise ValueError("Source exposure identity differs")
        group = s106.source_group(unit["sequence"]); n=len(data["frame_id"]); states+=n
        np.testing.assert_array_equal(data["frame_id"],np.arange(1,n+1))
        bindings = {a+"__"+h:models[group,a,h]["fingerprint"] for a in ARMS for h in s106.HEADS}
        cache_path = root/"exposure"/path.name
        metadata = dict(plan_fingerprint=plan["fingerprint"],unit=unit,source_arm=mode,models=bindings,kind="frozen_source_predictions")
        cached = c.read_cache(cache_path,metadata)
        fresh = cached is None
        if fresh: cached = dict(frame_id=data["frame_id"].copy(),eligible=data["eligible"].copy())
        np.testing.assert_array_equal(cached["frame_id"],data["frame_id"]); np.testing.assert_array_equal(cached["eligible"],data["eligible"])
        for arm in ARMS:
            for head in s106.HEADS:
                p = predict_batch(models[group,arm,head],data["x"],data["eligible"],arm)
                for key,value in p.items():
                    name=arm+"__"+head+"_"+key
                    if fresh: cached[name]=value
                    else:
                        if key in ("score","max_abs_z"): s107.close(cached[name],value)
                        else: np.testing.assert_array_equal(cached[name],value)
                    if arm=="baseline":
                        if key in ("score","max_abs_z"): s107.close(data[head+"_"+key],value)
                        else: np.testing.assert_array_equal(data[head+"_"+key],value)
                eligible=data["eligible"].astype(bool)
                source_row=source_units[unit["sequence"],unit["seed"],mode,head]
                rows.append(dict(cohort=source_row["cohort"],sequence=unit["sequence"],group=group,seed=unit["seed"],source_arm=mode,
                    representation=arm,head=head,policy=arm+"__"+head,states=n,eligible=float(eligible.mean()),commit_probability=float(p["commit"].mean()),
                    outside_fit_range=float(p["outside_fit_range"][eligible].mean()) if eligible.any() else None,
                    mean_max_abs_z=float(p["max_abs_z"][eligible].mean()) if eligible.any() else None,
                    p95_max_abs_z=float(np.quantile(p["max_abs_z"][eligible],.95)) if eligible.any() else None))
        if fresh: c.save(cache_path,cached,metadata)
        hashes[cache_path.relative_to(root).as_posix()]=c.file_hash(cache_path)
        print(f"[Stage-108 exposure] {path.stem}: {n} original states",flush=True)
    if states!=plan["exposure_states"] or len(hashes)!=plan["exposure_contexts"]: raise ValueError("Exposure coverage differs")
    summary,_=s106.summarize(rows,tuple(a+"__"+h for a in ARMS for h in s106.HEADS),
        ("eligible","commit_probability","outside_fit_range","mean_max_abs_z","p95_max_abs_z"))
    return rows,summary,hashes


def run(root,source,old,report,plan):
    samples=s107.load_samples(source,old)
    if len(s107.common_support(samples))!=plan["eligible_samples"]: raise ValueError("Common support differs")
    models,scope,training,artifacts=build_models(root,source,samples,old["groups"])
    prediction=predict_samples(samples,models)
    baseline=[dict(r,head=r["head"]) for r in prediction if r["representation"]=="baseline"]
    s107.audit_cached_decisions(source,samples,baseline)
    decision=decisions(samples,prediction); policy,paired=summaries(decision)
    training_macro=[]
    for arm in ARMS:
        for head in s106.HEADS:
            for phase in ("fit_resubstitution","held_out"):
                local=[r for r in training if r["representation"]==arm and r["head"]==head and r["phase"]==phase]
                record=dict(representation=arm,head=head,phase=phase,models=len(local))
                for metric in s107.EVAL_METRICS:
                    record[metric]=s106.mean(local,metric); record[metric+"_supported_models"]=sum(r[metric] is not None for r in local)
                training_macro.append(record)
    lookup={tuple(f[1:]):f[0] for f in plan["focus"]}
    if set(lookup)-{s107.sample_key(r) for r in samples}: raise ValueError("Missing fixed focus sample")
    focus=[dict(r,case_id=lookup[s107.sample_key(r)]) for r in prediction if s107.sample_key(r) in lookup]
    units,summary,cache_hashes=exposure(root,source,old,report,models,plan); artifacts.update(cache_hashes)
    tables=dict(zip(TABLES,(scope,training,training_macro,policy,paired,prediction,decision,focus,units,summary)))
    publish(root,plan,tables,artifacts)


def expected_counts(plan):
    g,n,contexts=len(plan["groups"]),plan["samples"],plan["exposure_contexts"]
    return dict(model_scope=8*g,training_summary=16*g,training_macro=16,state_predictions=8*n,
        sample_decisions=len(POLICIES)*n,focus=8*len(plan["focus"]),exposure_units=8*contexts)


def publish(root,plan,tables,artifacts):
    if set(tables)!=set(TABLES) or any(len(tables[n])!=v for n,v in expected_counts(plan).items()): raise ValueError("Incomplete factorial gate experiment")
    if len(artifacts)!=plan["new_models"]+plan["exposure_contexts"]: raise ValueError("Incomplete trained models or exposure caches")
    io.verify(plan["source_sha256"]);io.verify(plan["code_sha256"])
    out=root/"analysis";out.mkdir(exist_ok=True);hashes={}
    for name in TABLES:
        path=out/f"stage108_{name}.csv";s106.s103.s86.publish_table(path,tables[name]);hashes[path.name]=c.file_hash(path)
    primary=next(r for r in tables["paired"] if r["cohort"]==r["source_arm"]=="all" and r["contrast"]==DESIGN["primary"])
    c.publish_identical(out/"stage108_decision.json",io.seal(dict(schema=SCHEMA,status=STATUS,
        role="consumed_development_fixed_representation_factorial",plan_fingerprint=plan["fingerprint"],source_stage107_fingerprint=SOURCE107,
        counts={n:len(v) for n,v in tables.items()},primary=primary,reused_gate_models=2*len(plan["groups"]),new_gate_models=plan["new_models"],
        exposure_states=plan["exposure_states"],sgla_training_updates=0,ppo_training_updates=0,new_image_reads=0,new_search_calls=0,
        new_full_trajectories=0,confirmation_inference_calls=0,exact_baseline_replay=True,held_out_labels_excluded=True,
        table_sha256=hashes,artifact_sha256=artifacts,
        warnings=["Feature design uses consumed Stage-107 outcomes; leave-group-out is development evidence, not fresh confirmation.",
                  "Same data/targets/weights/ridge, different dimensions and capacity; effective regularization and CPU cost may differ.",
                  "Matched utility is one commit followed by the original policy; exposure is on old trajectories, not a new full policy.",
                  "Rate and future oracle are offline controls. No method adoption or full-trajectory protection gate."])))
    check_results(root);print(f"Stage-108: {STATUS}; {plan['new_models']} new CPU gates; no new GPU inference",flush=True)


def check_results(root):
    plan=c.c66.sealed(root/"manifest.json",SCHEMA);r=c.c66.sealed(root/"analysis/stage108_decision.json",SCHEMA)
    if (plan["design"]!=DESIGN or r["status"]!=STATUS or r["plan_fingerprint"]!=plan["fingerprint"]
            or r["source_stage107_fingerprint"]!=SOURCE107 or r["new_gate_models"]!=plan["new_models"]
            or plan["new_models"]!=6*len(plan["groups"]) or r["reused_gate_models"]!=2*len(plan["groups"])
            or r["exposure_states"]!=plan["exposure_states"] or not r["exact_baseline_replay"] or not r["held_out_labels_excluded"]
            or any(r["counts"][n]!=v for n,v in expected_counts(plan).items())
            or set(r["table_sha256"])!={f"stage108_{n}.csv" for n in TABLES}
            or len(r["artifact_sha256"])!=plan["new_models"]+plan["exposure_contexts"]
            or any(r[n]!=0 for n in ("sgla_training_updates","ppo_training_updates","new_image_reads","new_search_calls","new_full_trajectories","confirmation_inference_calls"))):
        raise ValueError("Incomplete Stage-108 result")
    io.verify(plan["source_sha256"]);io.verify(plan["code_sha256"])
    io.verify({str(root/"analysis"/n):h for n,h in r["table_sha256"].items()});io.verify({str(root/n):h for n,h in r["artifact_sha256"].items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage106",type=Path);parser.add_argument("--stage107",type=Path)
    parser.add_argument("--output_dir",required=True,type=Path);parser.add_argument("--check-results",action="store_true")
    args=parser.parse_args();root=args.output_dir.expanduser().resolve()
    if args.check_results: check_results(root);return
    if not args.stage106 or not args.stage107: parser.error("Supply complete Stage-106 and Stage-107 sources")
    source,audit=args.stage106.expanduser().resolve(),args.stage107.expanduser().resolve()
    if any(root==p or root in p.parents or p in root.parents for p in (source,audit)): parser.error("Use a separate Stage-108 directory")
    with s106.s93.writer_lock(root):
        if (root/"analysis/stage108_decision.json").exists(): check_results(root);return
        if not (root/"manifest.json").exists() and any(p.name!="collect.lock" for p in root.iterdir()): raise ValueError("Use an empty Stage-108 directory")
        print("Stage-108: checking frozen sources before any new fitting",flush=True)
        s107.check_results(audit);s106.check_results(source)
        p107=c.c66.sealed(audit/"manifest.json",s107.SCHEMA);r107=c.c66.sealed(audit/"analysis/stage107_decision.json",s107.SCHEMA)
        old=c.c66.sealed(source/"manifest.json",s106.SCHEMA);report=c.c66.sealed(source/"analysis/stage106_decision.json",s106.SCHEMA)
        if r107["fingerprint"]!=SOURCE107 or report["fingerprint"]!=s107.SOURCE106 or old["anchors"]!=259 or len(old["groups"])!=28:
            raise ValueError("Need the declared complete sources")
        bindings=dict(p107["source_sha256"])
        for path in (audit/"manifest.json",audit/"analysis/stage107_decision.json",source/"manifest.json",source/"analysis/stage106_decision.json"):
            bindings[str(path)]=c.file_hash(path)
        bindings.update({str(audit/"analysis"/n):h for n,h in r107["table_sha256"].items()})
        code=[Path(__file__).resolve(),Path(__file__).with_name("run_stage108_relational_gate.sh").resolve()]
        plan=io.seal(dict(schema=SCHEMA,design=DESIGN,groups=old["groups"],samples=259,eligible_samples=224,new_models=168,
            dimensions={a:len(feature_names(a)) for a in ARMS},focus=p107["focus"],exposure_contexts=old["exposure_contexts"],exposure_states=old["exposure_states"],
            source_sha256=bindings,code_sha256=p107["code_sha256"]|{str(p):c.file_hash(p) for p in code}))
        c.publish_identical(root/"manifest.json",plan);run(root,source,old,report,plan)


if __name__=="__main__": main()
