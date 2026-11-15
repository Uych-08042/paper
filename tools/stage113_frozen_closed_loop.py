"""Run frozen Stage-112 generators/scorers on complete own-history SGLA trajectories."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import time
import numpy as np

import stage112_shared_scoring as s112
from stage27_uav_common import failure_events, tracking_iou

common = s112.common
p,c,io,s103,s105,e63,s109 = common.p,common.c,common.io,common.s103,common.s105,common.e63,common.s109
SCHEMA = "stage113_frozen_generated_shared_closed_loop_v1"
SOURCE112 = "4622ef840a90ccfc658ff91b86652ac460a1371a3da9579e37b08425d8f3803f"
ARMS = ("native","fixed_hann","generated_hann","fixed_shared","generated_shared")
DESIGN = dict(seeds=[2312,2313], search_budget=4, bootstrap_seed=23130, bootstrap_samples=20000,
              practical_auc=.5, reliable_auc_tolerance=1., latency_warmup=10,
              primary="generated_shared_minus_fixed_shared", minimum_usage_videos=4,
              critical_controls=["fixed_shared","native","generated_hann"])
METRICS = ("auc","low_iou_fraction","longest_failure_fraction","reliable_auc","failure_events",
           "recovered_events","unrecovered_events","best_candidate_iou","selected_iou","mean_selection_gap",
           "good_candidate_fraction","selected_good_fraction","missed_good_fraction",
           "median_pipeline_ms","p95_pipeline_ms")
FLAGS = ("introduced_failure","new_on_fixed_stable","native_recoveries_lost","fixed_recoveries_lost")
TABLES = ("model_scope","policy_summary","paired","video_summary","seed_video","scope","recovery","seed_effects")
FIELDS = ("candidate_parents","candidate_boxes","features","image_hw","frame_id","unique_search_calls","search_ms",*s112.MAPS,
          "history_latest","selected","scores","boxes","parents","executed","memory_write","forecast_x","forecast_mean",
          "forecast_sigma","conditional_radius","generated_active","changed_crop","pipeline_ms")


def owners(units, splits):
    grouped = {}
    for spec in units: grouped.setdefault(spec["unit"]["sequence"],spec)
    out = []
    for sequence,spec in sorted(grouped.items()):
        group = s109.s106.source_group(sequence)
        matches = [s for s in splits if group in s["held_out_groups"]]
        if len(matches)!=1: raise ValueError("Need exactly one outer held-out generator/scorer")
        out.append(dict(spec=spec,outer_fold=matches[0]["fold"]))
    return out


def jobs_for(entries):
    return [dict(owner,arm=arm,seed=seed) for owner in entries for arm in ARMS
            for seed in (DESIGN["seeds"] if arm.endswith("shared") else [0])]


def job_key(job): return f"{job['spec']['unit']['sequence']}__{job['arm']}__s{job['seed']}"


def runtime_binding(expected,actual,previous=None):
    if actual["tracker"]["device"] not in ("NVIDIA L20","NVIDIA L40"): raise ValueError("Stage-113 requires a separately bound L20 or L40")
    differences=s109.check_runtime(expected,actual)
    if previous is not None and previous!=actual: raise ValueError("Stage-113 runtime changed on resume; use a new STAGE113_ROOT")
    return differences


def write_eligible(row):
    # Same causal boolean used by Stage-103, not an extra appearance re-selection.
    return bool(s103.s93.s92.trusted(s103.s93.action_bank(row),0,s103.DESIGN))


def forecast_parents(history, previous, hw, ap, arm, generator):
    k = 1 if arm=="native" else 4
    fixed = e63.source_common.prior.history_parents(history,ap["design"])
    active = arm.startswith("generated") and previous is not None
    x,mean,sigma,radius = np.zeros(32),np.zeros(4),np.zeros(4),np.zeros(4)
    parents = fixed[:k].copy()
    if active:
        x = p.online_input(history,previous["features"],int(previous["selected"]),bool(previous["memory_write"]))
        proposal = common.search_parents(history,hw,ap,x,generator,4)
        parents = proposal["candidate_parents"][4:]
        mean,sigma,radius = (proposal[n] for n in ("forecast_mean","forecast_sigma","conditional_radius"))
    changed = any(not np.array_equal(s105.crop_geometry(a,hw,ap["search_factor"],ap["search_size"])[0],
                                     s105.crop_geometry(b,hw,ap["search_factor"],ap["search_size"])[0]) for a,b in zip(parents,fixed))
    return dict(candidate_parents=parents,forecast_x=x,forecast_mean=mean,forecast_sigma=sigma,
                conditional_radius=radius,generated_active=bool(active),changed_crop=bool(changed))


def choose(row, history, arm, scorer):
    if arm=="native": scores=np.ones(1); selected=0
    elif arm.endswith("hann"):
        scores=row["features"][:,e63.source_common.prior.FEATURES.index("hann_peak")]
        selected=common.physical_choice(scores,row["candidate_boxes"])
    else:
        x=common.features(row["features"],row["candidate_boxes"],history)
        scores=common.scores(scorer,x); selected=common.physical_choice(scores,row["candidate_boxes"])
    return np.asarray(scores),int(selected)


def collect(observer, entry, old, arm, generator, scorer, synchronize=lambda:None, image_reader=s105.checked_image):
    ap=e63.c.actor_plan(old); history=[np.asarray(entry["initial_box"],float).copy()]; previous=None; rows=[]
    observer.initialize(image_reader(ap["root"],entry["frames"][0]),history[0])
    for frame,item in enumerate(entry["frames"][1:],1):
        image=image_reader(ap["root"],item); synchronize(); started=time.perf_counter()
        proposal=forecast_parents(history,previous,image.shape[:2],ap,arm,generator)
        row=s112.observe(observer,image,history,frame,ap,proposal["candidate_parents"],synchronize)
        scores,selected=choose(row,history,arm,scorer)
        executed=np.asarray([not any(np.array_equal(parent,other) for other in row["candidate_parents"][:i])
                             for i,parent in enumerate(row["candidate_parents"])],bool)
        row.update(history_latest=history[-1].copy(),selected=selected,scores=scores,
                   boxes=row["candidate_boxes"][selected].copy(),parents=row["candidate_parents"][selected].copy(),executed=executed)
        row["memory_write"]=False if arm=="native" else write_eligible(row)
        row.update({k:v for k,v in proposal.items() if k!="candidate_parents"})
        history=(history+[row["boxes"].copy()])[-ap["design"]["history"]:]
        observer.rt.tracker.state=row["boxes"].tolist(); synchronize()
        row["pipeline_ms"]=1000*(time.perf_counter()-started)
        rows.append(row); previous=row
        if frame%500==0: print(f"  [Stage-113 {arm}] {frame}/{len(entry['frames'])-1}",flush=True)
    if not rows: raise ValueError("Empty full trajectory")
    return {k:np.asarray([r[k] for r in rows]) for k in FIELDS}


def audit(data,entry,old,arm,generator,scorer):
    n=len(entry["frames"])-1; k=1 if arm=="native" else 4; ap=e63.c.actor_plan(old)
    if set(data)!=set(FIELDS): raise ValueError("Full trajectory schema changed")
    np.testing.assert_array_equal(data["frame_id"],np.arange(1,n+1))
    e63.source_common.prior.audit_calls(data,(n,k))
    shapes=dict(candidate_boxes=(n,k,4),candidate_parents=(n,k,4),features=(n,k,24),scores=(n,k),
                boxes=(n,4),parents=(n,4),executed=(n,k),selected=(n,),unique_search_calls=(n,),
                memory_write=(n,),generated_active=(n,),changed_crop=(n,),pipeline_ms=(n,),
                forecast_x=(n,32),forecast_mean=(n,4),forecast_sigma=(n,4),conditional_radius=(n,4))
    if any(data[name].shape!=shape for name,shape in shapes.items()): raise ValueError("Incomplete full trajectory")
    if not all(np.isfinite(data[name]).all() for name in ("raw","size","offset",*shapes)):
        raise ValueError("Nonfinite trajectory")
    if (data["candidate_boxes"][...,2:]<=0).any() or (data["candidate_parents"][...,2:]<=0).any() or data["selected"].dtype.kind not in "iu": raise ValueError("Invalid boxes/actions")
    if not np.isfinite(data["decode_error"]).all() or data["decode_error"].max()>ap["design"]["box_atol"]: raise ValueError("Decode error")
    if not np.isfinite(data["search_ms"]).all() or (data["search_ms"]<0).any() or (data["pipeline_ms"]<0).any(): raise ValueError("Invalid timing")
    history=[np.asarray(entry["initial_box"],float).copy()]; previous=None
    for i,frame in enumerate(data["frame_id"]):
        hw=(entry["frames"][frame]["height"],entry["frames"][frame]["width"])
        proposal=forecast_parents(history,previous,hw,ap,arm,generator)
        for name,value in proposal.items(): np.testing.assert_array_equal(data[name][i],value)
        np.testing.assert_array_equal(data["history_latest"][i],history[-1]); np.testing.assert_array_equal(data["image_hw"][i],hw)
        row={name:data[name][i] for name in data}; selected=int(row["selected"])
        score,choice=choose(row,history,arm,scorer)
        if choice!=selected: raise ValueError("Saved scoring action changed")
        np.testing.assert_allclose(row["scores"],score,rtol=1e-9,atol=1e-12)
        np.testing.assert_array_equal(row["boxes"],row["candidate_boxes"][selected]); np.testing.assert_array_equal(row["parents"],row["candidate_parents"][selected])
        expected_mask=np.asarray([not any(np.array_equal(parent,other) for other in row["candidate_parents"][:j])
                                  for j,parent in enumerate(row["candidate_parents"])],bool)
        np.testing.assert_array_equal(row["executed"],expected_mask)
        if int(row["unique_search_calls"])!=int(expected_mask.sum()): raise ValueError("Physical search accounting differs")
        if bool(row["memory_write"])!=(False if arm=="native" else write_eligible(row)): raise ValueError("Past trust qualification differs")
        for j,parent in enumerate(row["candidate_parents"]):
            crop,usable,factor=s105.crop_geometry(parent,hw,ap["search_factor"],ap["search_size"])
            np.testing.assert_array_equal(row["crop"][j],crop)
            view=dict(parent=parent,boxes=row["candidate_boxes"][j],crop=crop,usable=usable,image_hw=hw,
                      **{name:row[name][j] for name in ("raw","size","offset")})
            np.testing.assert_array_equal(row["features"][j],e63.actor.feature_vector(view,history,ap))
            dense=e63.dense_boxes(view["size"],view["offset"],parent,factor,ap["search_size"],hw)
            cell=int(np.argmax(view["raw"]*np.asarray(ap["window"],np.float32)))
            np.testing.assert_allclose(dense[cell],view["boxes"],rtol=0,atol=ap["design"]["box_atol"])
        history=(history+[row["boxes"].copy()])[-ap["design"]["history"]:]; previous=row


def recovery_rows(reference,candidate,identity,reference_arm):
    result=[]
    for r in s103.s93.recovery_comparison(reference,candidate,s103.DESIGN):
        result.append(dict(identity,reference_arm=reference_arm,reference_failure_frame=r["ppo_failure_frame"],
                           reference_recovery_frame=r["ppo_recovery_frame"],matched_good_frame=r["matched_good_frame"]))
    return result


def summarize(seed_rows):
    grouped=defaultdict(list)
    for r in seed_rows: grouped[r["cohort"],r["sequence"],r["arm"]].append(r)
    videos=[]
    for (cohort,sequence,arm),rows in sorted(grouped.items()):
        expected=DESIGN["seeds"] if arm.endswith("shared") else [0]
        if sorted(r["seed"] for r in rows)!=expected: raise ValueError("Missing trained seed")
        videos.append(dict(cohort=cohort,sequence=sequence,arm=arm,
            **{m:float(np.mean(v)) if (v:=[r[m] for r in rows if r[m] is not None]) else None for m in METRICS},
            introduced_failure=any(r["introduced_failure"] for r in rows),new_on_fixed_stable=any(r["new_on_fixed_stable"] for r in rows),
            native_recoveries_lost=sum(r["native_recoveries_lost"] for r in rows),fixed_recoveries_lost=sum(r["fixed_recoveries_lost"] for r in rows)))
    policies,paired=[],[]
    comparisons=(("generated_shared","fixed_shared"),("generated_shared","native"),("generated_shared","generated_hann"),
                 ("fixed_shared","fixed_hann"),("generated_hann","fixed_hann"),("fixed_shared","native"))
    negative={"low_iou_fraction","longest_failure_fraction","failure_events","unrecovered_events","mean_selection_gap","missed_good_fraction","median_pipeline_ms","p95_pipeline_ms"}
    design=dict(s103.DESIGN,bootstrap_seed=DESIGN["bootstrap_seed"],bootstrap_samples=DESIGN["bootstrap_samples"])
    for cohort in ("all","historical","expanded"):
        local=[r for r in videos if cohort=="all" or r["cohort"]==cohort]
        if not local: continue
        lookup={(r["sequence"],r["arm"]):r for r in local}; names=sorted({r["sequence"] for r in local})
        for arm in ARMS:
            part=[r for r in local if r["arm"]==arm]
            policies.append(dict(cohort=cohort,arm=arm,videos=len(part),
                **{m:float(np.mean(v)) if (v:=[r[m] for r in part if r[m] is not None]) else None for m in METRICS},
                **{f:sum(r[f] for r in part) for f in FLAGS}))
        for a,b in comparisons:
            for metric in METRICS:
                delta=[lookup[n,a][metric]-lookup[n,b][metric] for n in names if lookup[n,a][metric] is not None and lookup[n,b][metric] is not None]
                if not delta: continue
                stats=s103.contrast(delta,a+"_minus_"+b,metric,design,
                    family=False)
                sign=-1 if metric in negative else 1; d=np.asarray(delta)
                if cohort=="all" and a=="generated_shared" and b in DESIGN["critical_controls"] and metric=="auc":
                    draws=np.random.default_rng(DESIGN["bootstrap_seed"]).integers(0,len(d),(DESIGN["bootstrap_samples"],len(d)))
                    boot=d[draws].mean(1); tail=.05/(2*len(DESIGN["critical_controls"]))
                    stats.update(family_ci_low=float(np.quantile(boot,tail)),family_ci_high=float(np.quantile(boot,1-tail)))
                stats.update(wins=int((sign*d>1e-12).sum()),losses=int((sign*d< -1e-12).sum()),favorable_direction="negative" if sign<0 else "positive",
                             worst_video_delta=float(d.max() if sign<0 else d.min()),
                             worst_leave_one_video_out_delta=float(((d.sum()-d)/(len(d)-1)).max() if sign<0 else ((d.sum()-d)/(len(d)-1)).min()) if len(d)>1 else None)
                paired.append(dict(cohort=cohort,**stats))
    return policies,paired,videos


def load_frozen_models(root112,plan112):
    models,scope={},[]
    for split in plan112["folds"]:
        fold=split["fold"]; path=root112/"models"/f"generator_f{fold}.json"
        package=c.c66.sealed(path,common.SCHEMA)
        if package["split"]!=split or set(split["held_out_groups"])&set(package["generator"]["fit_groups"]+package["generator"]["calibration_groups"]): raise ValueError("Generator outer leakage")
        models[fold,"generator",0]=package
        scope.append(dict(outer_fold=fold,kind="generator",bank="generated_4",seed=0,fingerprint=package["fingerprint"],held_out_groups="|".join(split["held_out_groups"])))
        for bank in ("fixed_4","generated_4"):
            for seed in DESIGN["seeds"]:
                path=s112.model_path(root112,fold,bank,"shared",seed); model=c.c66.sealed(path,common.SCHEMA)
                if model["split"]!=split or model["design"]!=common.DESIGN or model["architecture"]!="shared" or model["seed"]!=seed: raise ValueError("Frozen shared scorer differs")
                models[fold,bank,seed]=model
                scope.append(dict(outer_fold=fold,kind="shared",bank=bank,seed=seed,fingerprint=model["fingerprint"],held_out_groups="|".join(split["held_out_groups"])))
    return models,scope


def job_models(job,models):
    generator=models[job["outer_fold"],"generator",0]["generator"] if job["arm"].startswith("generated") else None
    scorer=models[job["outer_fold"],"generated_4" if job["arm"]=="generated_shared" else "fixed_4",job["seed"]] if job["arm"].endswith("shared") else None
    return generator,scorer


def run(root,plan,models,model_scope,load_entry,load_native,old,observer,synchronize=lambda:None,image_reader=s105.checked_image):
    evidence,seed_rows,scope,recoveries={},[],[],[]
    by_sequence={}
    for job in plan["jobs"]: by_sequence.setdefault(job["spec"]["unit"]["sequence"],[]).append(job)
    for sequence,jobs in by_sequence.items():
        entry=load_entry(jobs[0]["spec"]); native_source=load_native(jobs[0]["spec"]); traces={}
        with e63.source_experiment.installed(observer,[],jobs[0]["spec"]["unit"]) as head:
            if head!=plan["native_head_sha256"]: raise ValueError("Native SGLA head changed")
            for job in jobs:
                arm,seed=job["arm"],job["seed"]; generator,scorer=job_models(job,models)
                meta=dict(plan_fingerprint=plan["fingerprint"],job=job,
                          generator_fingerprint=models[job["outer_fold"],"generator",0]["fingerprint"] if generator is not None else None,
                          scorer_fingerprint=scorer["fingerprint"] if scorer is not None else None)
                path=root/"evaluation"/(job_key(job)+".npz")
                data=s109.cache(root,path,meta,lambda:collect(observer,entry,old,arm,generator,scorer,synchronize,image_reader),evidence)
                audit(data,entry,old,arm,generator,scorer)
                if arm=="native":
                    error=float(np.abs(data["boxes"]-native_source["boxes"]).max())
                    if not plan["runtime_differences"] and error!=0.: raise ValueError("Same-runtime native trajectory differs")
                else: error=None
                traces[arm,seed]=data
                ident=dict(cohort=job["spec"]["cohort"],sequence=sequence,arm=arm,seed=seed,outer_fold=job["outer_fold"])
                physical=int(data["unique_search_calls"].sum()); n=len(data["frame_id"])
                scope.append(dict(ident,frames=n,physical_search_calls=physical,logical_search_calls=n*(1 if arm=="native" else 4),
                    physical_block_calls=int((data["block_calls"]*data["executed"][...,None]).sum()),
                    physical_head_calls=int((data["head_calls"]*data["executed"]).sum()),source_native_box_error=error,
                    generated_frames=int(data["generated_active"].sum()),changed_crop_frames=int(data["changed_crop"].sum()),
                    nonq0_output_frames=int(np.any(data["boxes"]!=data["candidate_boxes"][:,0],axis=1).sum()),trajectory_saved_before_labels=True))
                print(f"[Stage-113] {job_key(job)}: {n} frames / {physical} unique searches",flush=True)
        # Only now read post-initialization GT and compute metrics/recovery comparisons.
        gt=np.asarray(entry["ground_truth"],float)[1:]; native=traces["native",0]
        native_iou=tracking_iou(native["boxes"],gt)
        for job in jobs:
            arm,seed=job["arm"],job["seed"]; data=traces[arm,seed]
            ident=dict(cohort=job["spec"]["cohort"],sequence=sequence,arm=arm,seed=seed,outer_fold=job["outer_fold"])
            metric=s103.s93.s85.metrics(data,entry,native["boxes"],s103.DESIGN)
            overlap=tracking_iou(data["boxes"],gt); nr=[]; fr=[]
            if arm!="native": nr=recovery_rows(native_iou,overlap,ident,"native")
            if arm=="generated_shared":
                ref=tracking_iou(traces["fixed_shared",seed]["boxes"],gt)
                fr=recovery_rows(ref,overlap,ident,"fixed_shared")
                new_on_fixed=not failure_events(ref) and bool(failure_events(overlap))
            else: new_on_fixed=False
            timing=data["pipeline_ms"][DESIGN["latency_warmup"]:]
            metric.update(failure_events=len(failure_events(overlap)),new_on_fixed_stable=new_on_fixed,
                          native_recoveries_lost=sum(r["matched_good_frame"] is None for r in nr),
                          fixed_recoveries_lost=sum(r["matched_good_frame"] is None for r in fr),
                          median_pipeline_ms=float(np.median(timing)) if len(timing) else None,
                          p95_pipeline_ms=float(np.quantile(timing,.95)) if len(timing) else None)
            seed_rows.append(dict(ident,**{m:metric[m] for m in (*METRICS,*FLAGS)})); recoveries.extend(nr+fr)
    policies,paired,videos=summarize(seed_rows)
    effects=[]
    for seed in DESIGN["seeds"]:
        for baseline in DESIGN["critical_controls"]:
            delta=[]
            for sequence,jobs in by_sequence.items():
                a=next(r for r in seed_rows if r["sequence"]==sequence and r["arm"]=="generated_shared" and r["seed"]==seed)
                b=next(r for r in seed_rows if r["sequence"]==sequence and r["arm"]==baseline and r["seed"]==(seed if baseline=="fixed_shared" else 0))
                delta.append(a["auc"]-b["auc"])
            effects.append(dict(seed=seed,contrast="generated_shared_minus_"+baseline,videos=len(delta),mean_auc_delta=float(np.mean(delta))))
    publish(root,plan,dict(zip(TABLES,(model_scope,policies,paired,videos,seed_rows,scope,recoveries,effects))),evidence)


def development_checks(tables):
    policies={r["arm"]:r for r in tables["policy_summary"] if r["cohort"]=="all"}
    effects={r["contrast"]:r for r in tables["paired"] if r["cohort"]=="all" and r["metric"]=="auc"}
    own=policies["generated_shared"]
    reliable=next((r["ci_low"] for r in tables["paired"] if r["cohort"]=="all" and r["contrast"]=="generated_shared_minus_native" and r["metric"]=="reliable_auc"),None)
    return dict(practical_auc=all(effects["generated_shared_minus_"+b]["mean_delta"]>=DESIGN["practical_auc"] for b in DESIGN["critical_controls"]),
        family_ci_positive=all(effects["generated_shared_minus_"+b]["family_ci_low"]>0 for b in DESIGN["critical_controls"]),
        both_seeds_positive=all(r["mean_auc_delta"]>0 for r in tables["seed_effects"]),
        low_iou_not_worse=all(own["low_iou_fraction"]<=policies[b]["low_iou_fraction"] for b in ("fixed_shared","native")),
        longest_failure_not_worse=all(own["longest_failure_fraction"]<=policies[b]["longest_failure_fraction"] for b in ("fixed_shared","native")),
        reliable_auc_protected=reliable is not None and reliable>=-DESIGN["reliable_auc_tolerance"],
        native_stable_protected=own["introduced_failure"]==0,fixed_stable_protected=own["new_on_fixed_stable"]==0,
        native_recovery_protected=own["native_recoveries_lost"]==0,fixed_recovery_protected=own["fixed_recoveries_lost"]==0,
        nontrivial_usage=len({r["sequence"] for r in tables["scope"] if r["arm"]=="generated_shared" and r["nonq0_output_frames"]>0 and r["changed_crop_frames"]>0})>=DESIGN["minimum_usage_videos"])


def publish(root,plan,tables,evidence):
    trajectories=len(plan["jobs"]); sequences=len(plan["entries"])
    if (set(tables)!=set(TABLES) or len(tables["scope"])!=trajectories or len(tables["seed_video"])!=trajectories
            or len(evidence)!=2*trajectories or len(tables["video_summary"])!=sequences*len(ARMS)
            or len(tables["model_scope"])!=len(plan["folds"])*(1+2*len(DESIGN["seeds"])) or len(tables["seed_effects"])!=len(DESIGN["critical_controls"])*len(DESIGN["seeds"])): raise ValueError("Incomplete full cohort")
    io.verify(plan["source_sha256"]); io.verify(plan["code_sha256"])
    checks=development_checks(tables); out=root/"analysis"; out.mkdir(exist_ok=True); hashes={}
    for name in TABLES:
        path=out/f"stage113_{name}.csv"; s103.s86.publish_table(path,tables[name]); hashes[path.name]=c.file_hash(path)
    calls=sum(r["physical_search_calls"] for r in tables["scope"])
    blocks=sum(r["physical_block_calls"] for r in tables["scope"])
    primary=next(r for r in tables["paired"] if r["cohort"]=="all" and r["contrast"]==DESIGN["primary"] and r["metric"]=="auc")
    c.publish_identical(out/"stage113_decision.json",io.seal(dict(schema=SCHEMA,
        status="PROMISING_FROZEN_GENERATED_SHARED_CLOSED_LOOP_DEVELOPMENT" if all(checks.values()) else "REVIEW_FROZEN_GENERATED_SHARED_CLOSED_LOOP_DEVELOPMENT",
        plan_fingerprint=plan["fingerprint"],source_stage112_fingerprint=SOURCE112,runtime=plan["runtime"],runtime_differences=plan["runtime_differences"],
        primary=primary,development_checks=checks,videos=sequences,trajectories=trajectories,new_full_trajectories=trajectories,
        new_search_calls=calls,new_block_calls=blocks,logical_search_calls=sum(r["logical_search_calls"] for r in tables["scope"]),
        maximum_source_native_box_error=max(r["source_native_box_error"] for r in tables["scope"] if r["arm"]=="native"),
        weights_frozen=True,own_history_audited=True,training_updates=0,new_models=0,confirmation_inference_calls=0,
        table_counts={n:len(r) for n,r in tables.items()},table_sha256=hashes,artifact_sha256=evidence,warnings=[
            "All 28 source groups are consumed development. Fixed models and conditional video bootstrap do not constitute independent confirmation.",
            "Every arm follows its own accepted boxes; no source-state replay, teacher forcing, GT reset, recovery branch or online template update.",
            "The first tracked frame uses fixed cold-start inputs. Generated forecasts start after a previous predicted frame exists.",
            "Past write eligibility is recomputed with the unchanged Stage-103 response/geometry trust rule; it does not add an appearance selector.",
            "Both scorer seeds are retained. No K search, checkpoint choice, generator/scorer refitting or PPO training.",
            "Exact parent duplicates share one search within a frame; logical and physical call budgets are reported separately.",
            "Pipeline timing excludes image reads, initialization and artifact IO; native audit features also consume CPU time. No deployment speedup claim."])))
    check_results(root)
    for r in tables["policy_summary"]:
        if r["cohort"]=="all": print(f"{r['arm']:18s} AUC={r['auc']:.4f} low-IoU={100*r['low_iou_fraction']:.2f}%",flush=True)


def check_results(root):
    plan=c.c66.sealed(root/"manifest.json",SCHEMA); report=c.c66.sealed(root/"analysis/stage113_decision.json",SCHEMA)
    n=len(plan["jobs"]); v=len(plan["entries"])
    logical=sum(j["frames"]*(1 if j["arm"]=="native" else 4) for j in plan["jobs"])
    if (plan["design"]!=DESIGN or report["plan_fingerprint"]!=plan["fingerprint"] or report["source_stage112_fingerprint"]!=SOURCE112
            or plan["source_stage112_fingerprint"]!=SOURCE112 or report["runtime"]!=plan["runtime"] or report["runtime_differences"]!=plan["runtime_differences"]
            or report["trajectories"]!=n or report["new_full_trajectories"]!=n or report["videos"]!=v or report["logical_search_calls"]!=logical
            or not sum(j["frames"] for j in plan["jobs"])<=report["new_search_calls"]<=logical or report["new_block_calls"]!=7*report["new_search_calls"]
            or len(report["artifact_sha256"])!=2*n or set(report["table_sha256"])!={f"stage113_{name}.csv" for name in TABLES}
            or report["table_counts"]["scope"]!=n or report["table_counts"]["seed_video"]!=n or report["table_counts"]["video_summary"]!=v*len(ARMS)
            or not report["weights_frozen"] or not report["own_history_audited"]
            or any(report[k]!=0 for k in ("training_updates","new_models","confirmation_inference_calls"))
            or (not plan["runtime_differences"] and report["maximum_source_native_box_error"]!=0.)): raise ValueError("Incomplete Stage-113 result")
    expected_status="PROMISING_FROZEN_GENERATED_SHARED_CLOSED_LOOP_DEVELOPMENT" if all(report["development_checks"].values()) else "REVIEW_FROZEN_GENERATED_SHARED_CLOSED_LOOP_DEVELOPMENT"
    if report["status"]!=expected_status: raise ValueError("Development decision changed")
    io.verify(plan["source_sha256"]); io.verify(plan["code_sha256"])
    io.verify({str(root/"analysis"/name):h for name,h in report["table_sha256"].items()}); io.verify({str(root/name):h for name,h in report["artifact_sha256"].items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__); names=("stage76","stage89","stage93","stage103","stage112")
    for name in (*names,"output_dir"): parser.add_argument("--"+name,type=Path,required=name=="output_dir")
    parser.add_argument("--checkpoint"); parser.add_argument("--parameter",default="deit_distilled"); parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--check-results",action="store_true"); parser.set_defaults(mode="stage113")
    args=parser.parse_args(); root=args.output_dir.expanduser().resolve()
    if args.check_results: check_results(root); return
    if not all(getattr(args,n) for n in names) or not args.checkpoint: parser.error("Supply Stage-76/89/93/103/112 and original checkpoint")
    r76,r89,r93,r103,r112=sources=[getattr(args,n).expanduser().resolve() for n in names]
    if any(root==q or root in q.parents or q in root.parents for q in sources): parser.error("Use a separate Stage-113 root")
    with s109.s106.s93.writer_lock(root):
        if (root/"analysis/stage113_decision.json").exists(): check_results(root); return
        if not (root/"manifest.json").exists() and any(q.name!="collect.lock" for q in root.iterdir()): raise ValueError("Use an empty Stage-113 root")
        s112.check_results(r112); s103.check_results(r103)
        p112=c.c66.sealed(r112/"manifest.json",common.SCHEMA); report112=c.c66.sealed(r112/"analysis/stage112_decision.json",common.SCHEMA)
        p103=c.c66.sealed(r103/"manifest.json",s103.SCHEMA)
        if report112["fingerprint"]!=SOURCE112 or len(p112["folds"])!=4: raise ValueError("Need the frozen complete Stage-112 result")
        p93,_,p89,report89,source=s103.prior.s95.s94.load_run(r93,r76,r89)
        if p103["units"]!=p89["units"]: raise ValueError("Development source units changed")
        def load_entry(spec): return s103.prior.s95.load_unit(r93,r89,p93,p89,report89,source,spec)[0]
        def load_native(spec):
            owner=dict(spec["unit"],seed=0,policy="q0")
            return s109.s106.required(r103/"evaluation"/(c.key(owner)+"__native.npz"),dict(plan_fingerprint=p103["fingerprint"],unit=owner,arm="native"))
        entries=owners(p103["units"],p112["folds"])
        for owner in entries: owner["frames"]=len(load_entry(owner["spec"])["frames"])-1
        jobs=jobs_for(entries); models,model_scope=load_frozen_models(r112,p112)
        if len(entries)!=28 or len(jobs)!=196 or len(model_scope)!=20: raise ValueError("Need the complete 28-group/two-scorer-seed cohort")
        import torch
        torch.set_num_threads(1); runtime=s103.s76.device_runtime(source["original"],torch,args.gpu)
        previous=c.c66.sealed(root/"manifest.json",SCHEMA) if (root/"manifest.json").exists() else None
        differences=runtime_binding(p112["runtime"],runtime,previous["runtime"] if previous else None)
        bindings=dict(p112["source_sha256"])
        for base,number in ((r112,112),(r103,103)):
            for path in (base/"manifest.json",base/f"analysis/stage{number}_decision.json"): bindings[str(path)]=c.file_hash(path)
        for path in (r112/"models").glob("*.json"):
            if "generator_" in path.name or "__shared__" in path.name: bindings[str(path)]=c.file_hash(path)
        for owner in entries:
            native_owner=dict(owner["spec"]["unit"],seed=0,policy="q0"); path=r103/"evaluation"/(c.key(native_owner)+"__native.npz")
            for q in (path,path.with_suffix(".receipt.json")): bindings[str(q)]=c.file_hash(q)
        code=[Path(__file__).resolve(),Path(__file__).with_name("run_stage113_closed_loop.sh").resolve()]
        plan=io.seal(dict(schema=SCHEMA,design=DESIGN,source_stage112_fingerprint=SOURCE112,entries=entries,jobs=jobs,folds=p112["folds"],
            runtime=runtime,runtime_differences=differences,native_head_sha256=source["report"]["native_head_sha256"],source_sha256=bindings,
            code_sha256=p112["code_sha256"]|p103["code_sha256"]|{str(q):c.file_hash(q) for q in code}))
        c.publish_identical(root/"manifest.json",plan)
        print(f"Stage-113 frozen: {len(jobs)} complete own-history trajectories; no new model training",flush=True)
        frozen=dict(source["original"],fingerprint=plan["fingerprint"],runtime_identity=runtime,runtime=runtime["tracker"])
        native,_=s103.s76.startup.checked_runtime(args,frozen,root); observer=e63.Capture(native)
        try: run(root,plan,models,model_scope,load_entry,load_native,source["original"],observer,lambda:torch.cuda.synchronize(args.gpu))
        finally: observer.handle.remove()


if __name__=="__main__": main()
