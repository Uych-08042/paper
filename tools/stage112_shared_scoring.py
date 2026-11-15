"""Jointly isolate search generation and matched quality scoring; audit K=2/4/6/8."""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np

import stage112_scoring_common as common

p,c,io,s103,s105,e63,s109 = common.p,common.c,common.io,common.s103,common.s105,common.e63,common.s109
METRICS = ("iou","best_iou","missed_good","harmful_q0_replacement","oracle_regret","logical_search_calls","unique_search_calls","search_ms")
TABLES = ("fold_support","sampling_scope","training_scope","collection_scope","state_decisions","policy_summary","paired")
MAPS = ("crop","crop_sha256","layer","block_calls","head_calls","selector_calls","decode_error","raw","size","offset")


def observe(observer,image,history,frame,ap,parents,synchronize=lambda:None):
    views,times = {},{}
    for parent in parents:
        key = tuple(parent)
        if key not in views:
            synchronize(); started = time.perf_counter()
            views[key] = e63.actor.observe(observer,image,parent,frame,ap)
            synchronize(); times[key] = 1000*(time.perf_counter()-started)
    ordered = [views[tuple(parent)] for parent in parents]
    data = dict(candidate_parents=parents.copy(),candidate_boxes=np.asarray([v["boxes"] for v in ordered]),
        features=np.asarray([e63.actor.feature_vector(v,history,ap) for v in ordered]),
        image_hw=np.asarray(image.shape[:2]),frame_id=np.array(frame),unique_search_calls=np.array(len(views)),
        search_ms=np.asarray([times[tuple(parent)] for parent in parents]))
    data.update({name:np.asarray([v[name] for v in ordered]) for name in MAPS})
    return data


def audit_observation(data,parents,history,frame,hw,ap):
    n = len(parents)
    np.testing.assert_array_equal(data["candidate_parents"],parents)
    np.testing.assert_array_equal(data["frame_id"],frame); np.testing.assert_array_equal(data["image_hw"],hw)
    e63.source_common.prior.audit_calls(data,(n,))
    if data["candidate_boxes"].shape != (n,4) or data["features"].shape != (n,24): raise ValueError("Incomplete observation")
    if not np.isfinite(data["search_ms"]).all() or (data["search_ms"]<0).any(): raise ValueError("Invalid instrumented cost")
    if int(data["unique_search_calls"]) != len({tuple(v) for v in parents}): raise ValueError("Search accounting differs")
    for i,parent in enumerate(parents):
        crop,usable,factor = s105.crop_geometry(parent,hw,ap["search_factor"],ap["search_size"])
        np.testing.assert_array_equal(data["crop"][i],crop)
        view = dict(parent=parent,boxes=data["candidate_boxes"][i],crop=crop,usable=usable,image_hw=hw,
                    **{k:data[k][i] for k in ("raw","size","offset")})
        np.testing.assert_array_equal(data["features"][i],e63.actor.feature_vector(view,history,ap))
        dense = e63.dense_boxes(view["size"],view["offset"],parent,factor,ap["search_size"],hw)
        cell = int(np.argmax(view["raw"]*np.asarray(ap["window"],np.float32)))
        np.testing.assert_allclose(dense[cell],view["boxes"],rtol=0,atol=ap["design"]["box_atol"])
    if not np.isfinite(data["decode_error"]).all() or data["decode_error"].max()>ap["design"]["box_atol"]: raise ValueError("Decoder drift")
    np.testing.assert_array_equal(data["candidate_boxes"][0],data["candidate_boxes"][4])


def banks(data):
    return {"fixed_4":np.arange(4)} | {f"generated_{k}":np.arange(4,4+k) for k in common.DESIGN["budgets"]}


def scorer_examples(data,history,truth,ident):
    truth = np.asarray(truth,float)
    if not np.isfinite(truth).all() or (truth[2:]<=0).any(): return {}
    out = {}
    for bank,indices in banks(data).items():
        if bank not in ("fixed_4","generated_4"): continue
        out[bank] = dict(ident,x=common.features(data["features"][indices],data["candidate_boxes"][indices],history),
                         y=s105.tracking_iou(data["candidate_boxes"][indices],truth))
    return out


def specifications():
    out = []
    for condition in common.CONDITIONS:
        for policy in ("q0","hann"): out.append((condition,policy,0))
        if condition.endswith("_4"):
            out.append((condition,"old_ppo",0))
            for seed in common.DESIGN["seeds"]: out.append((condition,"concat",seed))
        for seed in common.DESIGN["seeds"]: out.append((condition,"shared",seed))
    return out


def predict(data,history,models,old_pack,old_design):
    selected,scores = [],[]
    bank_indices = banks(data)
    hann_column = e63.source_common.prior.FEATURES.index("hann_peak")
    matrices = {condition:common.features(data["features"][ix],data["candidate_boxes"][ix],history) for condition,ix in bank_indices.items()}
    for condition,policy,seed in specifications():
        ix = bank_indices[condition]; boxes = data["candidate_boxes"][ix]
        if policy == "q0": value = np.r_[1.,np.zeros(len(ix)-1)]; action = 0
        elif policy == "hann":
            value = data["features"][ix,hann_column]; action = common.physical_choice(value,boxes)
        elif policy == "old_ppo":
            decision = common.prior.c74.gate(matrices[condition],boxes,old_pack,old_design)
            value = decision["probabilities"]; action = int(np.argmax(value))
        else:
            model_bank = "fixed_4" if condition == "fixed_4" else "generated_4"
            value = common.scores(models[model_bank,policy,seed],matrices[condition])
            action = common.physical_choice(value,boxes)
        selected.append(action); scores.append(np.pad(value,(0,max(common.DESIGN["budgets"])-len(value))))
    return dict(selected=np.asarray(selected,int),scores=np.asarray(scores))


def evaluate(data,prediction,truth,ident):
    truth = np.asarray(truth,float); valid = bool(np.isfinite(truth).all() and (truth[2:]>0).all())
    rows = []; lookup = banks(data)
    def row(condition,policy,seed,action):
        ix = lookup[condition]; boxes = data["candidate_boxes"][ix]
        overlaps = s105.tracking_iou(boxes,truth) if valid else np.full(len(ix),np.nan)
        if action is None: action = int(np.argmax(overlaps)) if valid else 0
        best,value,q0 = (float(overlaps.max()),float(overlaps[action]),float(overlaps[0])) if valid else (None,None,None)
        unique = list(dict.fromkeys(tuple(parent) for parent in data["candidate_parents"][ix]))
        cost = sum(float(data["search_ms"][ix[next(i for i,p in enumerate(data["candidate_parents"][ix]) if tuple(p)==u)]]) for u in unique)
        return dict(ident,condition=condition,policy=policy,model_seed=seed,selected=action,valid_gt=valid,iou=value,best_iou=best,q0_iou=q0,
            missed_good=float(best>=.5 and value<.5) if valid else None,
            harmful_q0_replacement=float(q0>=.5 and value<.2) if valid else None,oracle_regret=best-value if valid else None,
            logical_search_calls=len(ix),unique_search_calls=len(unique),search_ms=cost)
    for i,(condition,policy,seed) in enumerate(specifications()): rows.append(row(condition,policy,seed,int(prediction["selected"][i])))
    for condition in common.CONDITIONS: rows.append(row(condition,"oracle",0,None))
    return rows


def summaries(rows):
    policies,paired = [],[]
    for condition in common.CONDITIONS:
        local = [r for r in rows if r["condition"] == condition]
        names = ("q0","hann","old_ppo","concat","shared","oracle") if condition.endswith("_4") else ("q0","hann","shared","oracle")
        summary,_ = s109.s106.summarize(local,names,METRICS)
        policies.extend(dict(r,condition=condition) for r in summary)
    comparisons = [(condition,"shared",condition,control) for condition in common.CONDITIONS
                   for control in (("q0","hann","old_ppo","concat") if condition.endswith("_4") else ("q0","hann"))]
    comparisons += [("generated_4","shared","fixed_4","shared")]
    comparisons += [(f"generated_{k}","shared","generated_4","shared") for k in (2,6,8)]
    for cohort in ("all","historical","expanded"):
        for mode in ("all",*s103.MODES[1:]):
            local = [r for r in rows if (cohort=="all" or r["cohort"]==cohort) and (mode=="all" or r["source_arm"]==mode)]
            if not local: continue
            for target,policy,control,baseline in comparisons:
                for metric in ("iou","harmful_q0_replacement","missed_good"):
                    a = s109.s106.macro([r for r in local if r["condition"]==target and r["policy"]==policy],metric)
                    b = s109.s106.macro([r for r in local if r["condition"]==control and r["policy"]==baseline],metric)
                    groups = sorted(a.keys()&b.keys()); d = np.asarray([a[g]-b[g] for g in groups])
                    name = policy+"_minus_"+baseline if target==control else target+"__"+policy+"_minus_"+control+"__"+baseline
                    paired.append(dict(cohort=cohort,source_arm=mode,condition=target,contrast=name,metric=metric,groups=len(d),
                        mean_delta=float(d.mean()) if len(d) else None,median_delta=float(np.median(d)) if len(d) else None,
                        wins=int((d>1e-12).sum()),ties=int((np.abs(d)<=1e-12).sum()),losses=int((d< -1e-12).sum()),
                        minimum_leave_one_group_out=float(min((d.sum()-v)/(len(d)-1) for v in d)) if len(d)>1 else None))
    return policies,paired


def model_path(root,fold,bank,architecture,seed): return root/"models"/f"f{fold}__{bank}__{architecture}__s{seed}.json"


def fit_scorers(root,split,examples,evidence):
    models,scope = {},[]
    for bank in ("fixed_4","generated_4"):
        _,_,_,signature = common.training_arrays(examples[bank],split)
        for architecture in ("concat","shared"):
            for seed in common.DESIGN["seeds"]:
                path = model_path(root,split["fold"],bank,architecture,seed)
                if path.exists():
                    model = c.c66.sealed(path,common.SCHEMA)
                    if (model["input_fingerprint"]!=signature or model["design"]!=common.DESIGN or model["split"]!=split
                            or model["architecture"]!=architecture or model["seed"]!=seed): raise ValueError("Scorer fitting evidence changed")
                else:
                    model = common.train(examples[bank],split,architecture,seed,common.DESIGN)
                    c.publish_identical(path,model)
                models[bank,architecture,seed] = model; evidence[path.relative_to(root).as_posix()] = c.file_hash(path)
                scope.append(dict(fold=split["fold"],bank=bank,architecture=architecture,model_seed=seed,examples=model["examples"],
                    parameters=model["parameters"],optimizer_steps=model["optimizer_steps"],first_fit_mse=model["first_fit_mse"],final_fit_mse=model["final_fit_mse"],
                    input_fingerprint=signature,model_fingerprint=model["fingerprint"],held_out_excluded=True))
    return models,scope


def run(root,r103,p103,plan,fit_rows,load_unit,pack_for,old,observer,synchronize=lambda:None):
    evidence,fold_rows,sampling,training,collection,decisions = {},[],[],[],[],[]
    input_path = root/"generator_fit_inputs.json"
    if io.seal(dict(rows=fit_rows))["fingerprint"] != plan["fit_input_fingerprint"]: raise ValueError("Source predictor fitting input changed")
    c.publish_identical(input_path,io.seal(dict(schema=common.SCHEMA,rows=fit_rows))); evidence[input_path.name] = c.file_hash(input_path)
    ap = e63.c.actor_plan(old)
    for split in plan["folds"]:
        fold = split["fold"]; package = common.fit_generator(fit_rows,split)
        path = root/"models"/f"generator_f{fold}.json"; path.parent.mkdir(exist_ok=True)
        c.publish_identical(path,package); evidence[path.relative_to(root).as_posix()] = c.file_hash(path)
        generator = package["generator"]
        fold_rows.append(dict(fold=fold,held_out_groups="|".join(split["held_out_groups"]),scorer_fit_groups="|".join(split["train_groups"]),
            generator_fit_groups="|".join(generator["fit_groups"]),calibration_groups="|".join(generator["calibration_groups"]),
            generator_fit_rows=generator["fit_rows"],calibration_rows=generator["calibration_rows"],generator_fingerprint=package["fingerprint"],outer_groups_excluded=True))
        examples = {bank:[] for bank in ("fixed_4","generated_4")}
        # All observations precede scoring fits; evaluation labels never enter this fitting loop.
        for job in plan["units"]:
            spec,mode = job["spec"],job["source_arm"]; unit = spec["unit"]
            role,frames = common.frame_selection(job,split,common.DESIGN)
            sampling.append(dict(fold=fold,context=job["context"],role=role,group=s109.s106.source_group(unit["sequence"]),
                                 frames="|".join(map(str,frames)),states=len(frames)))
            entry = load_unit(spec); trace = s105.prior.required_trace(r103,p103,unit,mode)
            with e63.source_experiment.installed(observer,[],unit) as head:
                if head != plan["native_head_sha256"]: raise ValueError("Original SGLA head changed")
                for frame in frames:
                    ident = s109.s106.identity(spec,mode,frame); x,history = p.input_at(trace,entry,frame,old)
                    hw = (entry["frames"][frame]["height"],entry["frames"][frame]["width"])
                    stem = f"f{fold}__{job['context']}__f{frame:06d}"
                    meta = dict(plan_fingerprint=plan["fingerprint"],fold=fold,unit=unit,source_arm=mode,frame=frame,role=role,generator_fingerprint=package["fingerprint"])
                    computed = common.search_parents(history,hw,ap,x,generator,max(common.DESIGN["budgets"]))
                    proposed = s109.cache(root,root/"proposals"/(stem+".npz"),dict(meta,kind="parents"),lambda:computed,evidence)
                    for k in computed: np.testing.assert_array_equal(proposed[k],computed[k])
                    def collect():
                        observer.initialize(s105.checked_image(ap["root"],entry["frames"][0]),entry["initial_box"])
                        image = s105.checked_image(ap["root"],entry["frames"][frame])
                        return observe(observer,image,history,frame,ap,proposed["candidate_parents"],synchronize)
                    data = s109.cache(root,root/"observations"/(stem+".npz"),dict(meta,kind="searches"),collect,evidence)
                    audit_observation(data,proposed["candidate_parents"],history,frame,hw,ap)
                    error = float(np.abs(data["candidate_boxes"][:4]-trace["candidate_boxes"][frame-1]).max())
                    if not plan["runtime_differences"]:
                        for name in ("candidate_boxes","features","crop","crop_sha256","layer"):
                            np.testing.assert_array_equal(data[name][:4],trace[name][frame-1])
                    collection.append(dict(ident,fold=fold,role=role,unique_search_calls=int(data["unique_search_calls"]),logical_search_calls=len(data["candidate_parents"]),
                                           source_fixed_box_error=error,parents_saved_before_images=True,shared_q0_pass=True))
                    if role == "fit":
                        for bank,row in scorer_examples(data,history,entry["ground_truth"][frame],ident).items(): examples[bank].append(row)
            print(f"[Stage-112 collect] outer{fold} {job['context']} {role}: {len(frames)} states",flush=True)
        models,fit_scope = fit_scorers(root,split,examples,evidence); training.extend(fit_scope)
        for job in plan["units"]:
            spec,mode = job["spec"],job["source_arm"]; unit = spec["unit"]
            role,frames = common.frame_selection(job,split,common.DESIGN)
            if role != "evaluate": continue
            entry,old_pack = load_unit(spec),pack_for(unit); trace = s105.prior.required_trace(r103,p103,unit,mode)
            for frame in frames:
                ident = s109.s106.identity(spec,mode,frame); _,history = p.input_at(trace,entry,frame,old)
                stem = f"f{fold}__{job['context']}__f{frame:06d}"
                meta = dict(plan_fingerprint=plan["fingerprint"],fold=fold,unit=unit,source_arm=mode,frame=frame,role=role,generator_fingerprint=package["fingerprint"])
                data = s109.s106.required(root/"observations"/(stem+".npz"),dict(meta,kind="searches"))
                computed = predict(data,history,models,old_pack,old["design"])
                prediction = s109.cache(root,root/"predictions"/(stem+".npz"),dict(meta,kind="selection"),lambda:computed,evidence)
                np.testing.assert_array_equal(prediction["selected"],computed["selected"])
                np.testing.assert_allclose(prediction["scores"],computed["scores"],rtol=1e-9,atol=1e-12)
                decisions.extend(evaluate(data,prediction,entry["ground_truth"][frame],ident))
    policies,paired = summaries(decisions)
    publish(root,plan,dict(zip(TABLES,(fold_rows,sampling,training,collection,decisions,policies,paired))),evidence)


def counts(plan):
    n = sum(len(j["sampling"]["eval_frames"]) for j in plan["units"])
    points = sum(len(common.frame_selection(j,f,common.DESIGN)[1]) for f in plan["folds"] for j in plan["units"])
    settings = {(cohort,mode) for j in plan["units"] for cohort in ("all",j["spec"]["cohort"]) for mode in ("all",j["source_arm"])}
    models = len(plan["folds"])*4*len(common.DESIGN["seeds"])
    return dict(fold_support=len(plan["folds"]),sampling_scope=len(plan["folds"])*len(plan["units"]),training_scope=models,
                collection_scope=points,state_decisions=n*(len(specifications())+len(common.CONDITIONS)),
                policy_summary=24*len(settings),paired=54*len(settings))


def publish(root,plan,tables,evidence):
    expected = counts(plan); n = sum(len(j["sampling"]["eval_frames"]) for j in plan["units"])
    artifacts = 1+len(plan["folds"])+expected["training_scope"]+4*expected["collection_scope"]+2*n
    if set(tables)!=set(TABLES) or any(len(tables[k])!=v for k,v in expected.items()) or len(evidence)!=artifacts: raise ValueError("Incomplete nested scoring run")
    io.verify(plan["source_sha256"]); io.verify(plan["code_sha256"])
    out = root/"analysis"; out.mkdir(exist_ok=True); hashes = {}
    for name in TABLES:
        path = out/f"stage112_{name}.csv"; s103.s86.publish_table(path,tables[name]); hashes[path.name] = c.file_hash(path)
    primary = next(r for r in tables["paired"] if r["cohort"]==r["source_arm"]=="all" and r["condition"]==common.DESIGN["primary_condition"]
                   and r["contrast"]==common.DESIGN["primary"] and r["metric"]==common.DESIGN["primary_metric"])
    calls = sum(r["unique_search_calls"] for r in tables["collection_scope"])
    c.publish_identical(out/"stage112_decision.json",io.seal(dict(schema=common.SCHEMA,status=common.STATUS,plan_fingerprint=plan["fingerprint"],
        source_stage111_fingerprint=common.SOURCE111,role="consumed_development_nested_candidate_quality_and_budget",runtime=plan["runtime"],runtime_differences=plan["runtime_differences"],
        counts=expected,primary=primary,new_generator_packages=len(plan["folds"]),new_selector_models=expected["training_scope"],
        selector_optimizer_steps=sum(r["optimizer_steps"] for r in tables["training_scope"]),new_search_calls=calls,new_block_calls=7*calls,
        template_initializations=expected["collection_scope"],logical_search_calls=sum(r["logical_search_calls"] for r in tables["collection_scope"]),
        maximum_fixed_source_box_error=max(r["source_fixed_box_error"] for r in tables["collection_scope"]),
        outer_groups_excluded=True,shared_q0_pass=True,nested_budget_prefixes=True,decisions_saved_before_labels=True,
        sgla_training_updates=0,ppo_training_updates=0,new_full_trajectories=0,confirmation_inference_calls=0,
        table_sha256=hashes,artifact_sha256=evidence,warnings=[
            "28 consumed development groups; outer isolation applies to new generator/calibration/scorer, not independent upstream confirmation.",
            "Four is the primary matched budget. K=2/6/8 reuse the same K=4-trained shared scorer, without K-specific fitting or selecting an optimal K.",
            "All decisions use source histories. Current-IoU quality fitting and extra same-state budget searches are not PPO training or full-video AUC.",
            "Search timing sums synchronized observed parent calls; excludes image reads, proposal allocation, templates and scorer cost. No deployment speed claim.",
            "q0 is charged the entire observed bank for equal-budget diagnostics; a standalone q0 implementation would need only its own search.",
            "New outer generators differ from Stage-110 single-group packages. All controls are reobserved on the same bound runtime.",
            "Shared and concatenated MLPs have matched hidden width, labels, examples and update counts, but different parameter counts.",
            "Shared normalization is fitted only on training groups. q0 and Hann are required controls; ordering invariance alone does not ensure accuracy."])))
    check_results(root); print(f"Stage-112: {common.STATUS}; {n} evaluation states, {calls} unique searches",flush=True)


def check_results(root):
    plan = c.c66.sealed(root/"manifest.json",common.SCHEMA); r = c.c66.sealed(root/"analysis/stage112_decision.json",common.SCHEMA)
    expected = counts(plan); n = sum(len(j["sampling"]["eval_frames"]) for j in plan["units"])
    artifacts = 1+len(plan["folds"])+expected["training_scope"]+4*expected["collection_scope"]+2*n
    if (plan["design"]!=common.DESIGN or r["status"]!=common.STATUS or r["plan_fingerprint"]!=plan["fingerprint"]
            or plan["source_stage111_fingerprint"]!=common.SOURCE111 or r["source_stage111_fingerprint"]!=common.SOURCE111 or r["counts"]!=expected
            or r["runtime"]!=plan["runtime"] or r["runtime_differences"]!=plan["runtime_differences"]
            or r["new_generator_packages"]!=len(plan["folds"]) or r["new_selector_models"]!=expected["training_scope"]
            or r["new_block_calls"]!=7*r["new_search_calls"] or not expected["collection_scope"]<=r["new_search_calls"]<=11*expected["collection_scope"]
            or r["logical_search_calls"]!=12*expected["collection_scope"] or r["template_initializations"]!=expected["collection_scope"]
            or len(r["artifact_sha256"])!=artifacts or set(r["table_sha256"])!={f"stage112_{name}.csv" for name in TABLES}
            or any(r[k]!=0 for k in ("sgla_training_updates","ppo_training_updates","new_full_trajectories","confirmation_inference_calls"))
            or not all(r[k] for k in ("outer_groups_excluded","shared_q0_pass","nested_budget_prefixes","decisions_saved_before_labels"))
            or (not plan["runtime_differences"] and r["maximum_fixed_source_box_error"]!=0.)): raise ValueError("Incomplete Stage-112 result")
    if plan["folds"]!=common.folds(plan["units"],common.DESIGN["outer_folds"]): raise ValueError("Outer split changed")
    io.verify(plan["source_sha256"]); io.verify(plan["code_sha256"])
    io.verify({str(root/"analysis"/name):h for name,h in r["table_sha256"].items()}); io.verify({str(root/name):h for name,h in r["artifact_sha256"].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__); names = ("stage76","stage89","stage93","stage103","stage110","stage111")
    for name in (*names,"output_dir"): parser.add_argument("--"+name,type=Path,required=name=="output_dir")
    parser.add_argument("--checkpoint"); parser.add_argument("--parameter",default="deit_distilled"); parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--check-results",action="store_true"); parser.set_defaults(mode="stage112")
    args = parser.parse_args(); root = args.output_dir.expanduser().resolve()
    if args.check_results: check_results(root); return
    if not all(getattr(args,n) for n in names) or not args.checkpoint: parser.error("Supply Stage-76/89/93/103/110/111 and original checkpoint")
    r76,r89,r93,r103,r110,r111 = sources = [getattr(args,n).expanduser().resolve() for n in names]
    if any(root==q or root in q.parents or q in root.parents for q in sources): parser.error("Use a separate Stage-112 root")
    with s109.s106.s93.writer_lock(root):
        if (root/"analysis/stage112_decision.json").exists(): check_results(root); return
        if not (root/"manifest.json").exists() and any(q.name!="collect.lock" for q in root.iterdir()): raise ValueError("Use an empty Stage-112 root")
        common.prior.check_results(r111); common.prior.s110.check_results(r110); s103.check_results(r103)
        p111 = c.c66.sealed(r111/"manifest.json",common.prior.SCHEMA); report111 = c.c66.sealed(r111/"analysis/stage111_decision.json",common.prior.SCHEMA)
        p110 = c.c66.sealed(r110/"manifest.json",p.SCHEMA); p103 = c.c66.sealed(r103/"manifest.json",s103.SCHEMA)
        if report111["fingerprint"]!=common.SOURCE111 or p110["units"]!=p111["units"] or len(p110["groups"])!=28: raise ValueError("Need fixed complete source results")
        p93,_,p89,report89,source = s103.prior.s95.s94.load_run(r93,r76,r89)
        actor = io.load_source(Path(source["plan"]["source_input"]))
        if actor["plan"]!=source["original"] or actor["models"]["fingerprint"]!=source["report"]["source_training_fingerprint"] or p103["units"]!=p89["units"]: raise ValueError("Frozen PPO/source changed")
        def load_unit(spec): return s103.prior.s95.load_unit(r93,r89,p93,p89,report89,source,spec)[0]
        def pack_for(unit): return s103.e74.learned_pack(actor["models"],unit,actor["prior"])
        jobs,fit_rows,bindings = common.prior.s110.prepare(r103,p103,load_unit,source["original"])
        for a,b in zip(jobs,p110["units"]):
            if any(a[k]!=b[k] for k in a): raise ValueError("Source sampling changed")
        splits = common.folds(jobs,common.DESIGN["outer_folds"])
        if len(jobs)!=80 or len(splits)!=4 or any(len(f["held_out_groups"])!=7 for f in splits): raise ValueError("Need full balanced development outer folds")
        import torch
        torch.set_num_threads(1); runtime = s103.s76.device_runtime(source["original"],torch,args.gpu)
        previous = c.c66.sealed(root/"manifest.json",common.SCHEMA) if (root/"manifest.json").exists() else None
        differences = common.prior.s110.runtime_binding(p110["runtime"],runtime,previous["runtime"] if previous else None)
        bindings.update(p111["source_sha256"]); bindings.update(actor["bindings"])
        for base,number in ((r110,110),(r111,111),(r103,103)):
            for q in (base/"manifest.json",base/f"analysis/stage{number}_decision.json"): bindings[str(q)] = c.file_hash(q)
        code = [Path(__file__).resolve(),Path(common.__file__).resolve(),Path(__file__).with_name("run_stage112_shared_scoring.sh").resolve()]
        plan = io.seal(dict(schema=common.SCHEMA,design=common.DESIGN,units=jobs,folds=splits,source_stage111_fingerprint=common.SOURCE111,
            native_head_sha256=source["report"]["native_head_sha256"],runtime=runtime,runtime_differences=differences,
            fit_input_fingerprint=io.seal(dict(rows=fit_rows))["fingerprint"],source_sha256=bindings,
            code_sha256=p111["code_sha256"]|p103["code_sha256"]|{str(q):c.file_hash(q) for q in code}))
        c.publish_identical(root/"manifest.json",plan)
        print("Stage-112 frozen: four outer folds, shared/concat fitting, nested 2/4/6/8 search budgets",flush=True)
        frozen = dict(source["original"],fingerprint=plan["fingerprint"],runtime_identity=runtime,runtime=runtime["tracker"])
        native,_ = s103.s76.startup.checked_runtime(args,frozen,root); observer = e63.Capture(native)
        try: run(root,r103,p103,plan,fit_rows,load_unit,pack_for,source["original"],observer,lambda:torch.cuda.synchronize(args.gpu))
        finally: observer.handle.remove()


if __name__=="__main__": main()
