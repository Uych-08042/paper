"""Sample unlabelled eligible development states; freeze gates and measure paired futures."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path

import numpy as np

import stage108_relational_commit_gate as s108

s107, s106 = s108.s107, s108.s106
s105, c, io = s106.s105, s108.c, s108.io
s103, e63 = s105.s103, s105.e63
SCHEMA = "stage109_gt_free_sampled_commit_branches_v1"
STATUS = "COMPLETE_GT_FREE_SAMPLED_COMMIT_BRANCHES"
SOURCE108 = "e8cf2e2928d42715b69956355858e04d2952eedd0b0b4ac05c7c541a1d9397a9"
DESIGN = dict(horizon=32, quota_per_context=12, sampler_seed=2309,
    generator="PCG64", population="remaining_eligible_full_horizon_source_states",
    exclusion="all_previously_labelled_stage106_anchor_identities",
    threshold=0., primary="relation_mode__future_minus_baseline__future")
ARMS = ("baseline", "relation_mode")
TREATMENTS = ("keep", "both_pair")
POLICIES = ("keep", "always_pair", "future_oracle")+tuple(a+"__"+h for a in ARMS for h in s106.HEADS)+tuple(a+"__rate_expected" for a in ARMS)
TABLES = ("sampling_scope", "scope", "branches", "samples", "state_predictions", "sample_decisions", "policy_summary", "paired")


def sample_frames(frame_ids, eligible, excluded_ids, context_key, design=DESIGN):
    """Whitelist: identities, online eligibility, availability and fixed random seed only."""
    frames, mask = np.asarray(frame_ids), np.asarray(eligible)
    if frames.ndim != 1 or mask.shape != frames.shape or not np.isin(mask, (0, 1)).all():
        raise ValueError("Invalid sampling population")
    np.testing.assert_array_equal(frames, np.arange(1, len(frames)+1))
    eligible_frames = frames[mask.astype(bool)].astype(int)
    full = eligible_frames[eligible_frames <= len(frames)-design["horizon"]+1]
    excluded = sorted(set(int(f) for f in excluded_ids))
    pool = sorted(set(full.tolist())-set(excluded))
    seed = int.from_bytes(hashlib.sha256(f"{design['sampler_seed']}:{context_key}".encode()).digest()[:16], "big")
    rng = np.random.Generator(np.random.PCG64(seed))
    count = min(design["quota_per_context"], len(pool))
    selected = sorted(int(f) for f in rng.choice(pool, size=count, replace=False))
    return dict(context=context_key, source_states=len(frames), eligible_states=len(eligible_frames),
        full_horizon_eligible=len(full), removed_known=len(set(full.tolist()) & set(excluded)),
        excluded_ids=excluded, pool=pool, selected=selected, seed=str(seed),
        inclusion_probability=count/len(pool) if pool else 0.)


def validate_selection(job, design=DESIGN):
    s = job["sampling"]
    pool, selected = s["pool"], s["selected"]
    if "spec" in job and s["context"] != c.key(job["spec"]["unit"])+"__"+job["source_arm"]:
        raise ValueError("Sampling context identity changed")
    expected = sorted(int(f) for f in np.random.Generator(np.random.PCG64(int(s["seed"]))).choice(
        pool, size=min(design["quota_per_context"], len(pool)), replace=False))
    seed = int.from_bytes(hashlib.sha256(f"{design['sampler_seed']}:{s['context']}".encode()).digest()[:16], "big")
    if (pool != sorted(set(pool)) or selected != expected or int(s["seed"]) != seed
            or set(pool) & set(s["excluded_ids"]) or any(f < 1 or f+design["horizon"]-1 > s["source_states"] for f in pool)
            or s["full_horizon_eligible"] != len(pool)+s["removed_known"]
            or s["inclusion_probability"] != (len(selected)/len(pool) if pool else 0.)):
        raise ValueError("Frozen sampling identities changed")


def budget(jobs):
    n = sum(len(j["sampling"]["selected"]) for j in jobs)
    length = DESIGN["horizon"]
    return dict(contexts=len(jobs), anchors=n, short_branches=2*n, branch_frames=2*n*length,
        new_search_calls=n*(4+8*(length-1)), logical_search_calls=n*8*length, data_caches=4*n)


def check_runtime(expected, actual, previous=None):
    if actual["tracker"]["device"] not in ("NVIDIA L20", "NVIDIA L40"):
        raise ValueError("Stage-109 supports a separately bound L20 or L40 run")
    differences = s103.s76.startup.differences(expected, actual)
    if any(r["field"] not in ("runtime.gpu_uuid", "runtime.tracker.device") for r in differences):
        raise ValueError("Frozen software, weights or configuration changed")
    if previous is not None and actual != previous:
        raise ValueError("Runtime changed on resume; use a new STAGE109_ROOT for this GPU/runtime")
    return differences


def prepare(root103, root106, p103, p106, known):
    jobs, bindings = [], {}
    for spec in p103["units"]:
        unit = spec["unit"]
        for mode in s103.MODES[1:]:
            key = c.key(unit)+"__"+mode
            path = root106/"exposure"/(key+".npz")
            meta = dict(plan_fingerprint=p106["fingerprint"], unit=unit, source_arm=mode, kind="frozen_source_exposure")
            data = s106.required(path, meta)
            trace = s105.prior.required_trace(root103, p103, unit, mode)
            np.testing.assert_array_equal(data["frame_id"], trace["frame_id"])
            excluded = [int(r["frame"]) for r in known if r["sequence"] == unit["sequence"]
                and int(r["seed"]) == unit["seed"] and r["source_arm"] == mode]
            sampling = sample_frames(data["frame_id"], data["eligible"], excluded, key)
            jobs.append(dict(spec=spec, source_arm=mode, sampling=sampling))
            for artifact in (path, root103/"evaluation"/(key+".npz")):
                for bound in (artifact, artifact.with_suffix(".receipt.json")):
                    bindings[str(bound)] = c.file_hash(bound)
            print(f"[Stage-109 sample] {key}: {len(sampling['selected'])}/{len(sampling['pool'])} unlabelled eligible starts", flush=True)
    if not budget(jobs)["anchors"]:
        raise ValueError("No unlabelled full-horizon eligible states; no inference started")
    return jobs, bindings


def load_models(root106, root108, groups):
    bank, bindings = {}, {}
    for group in groups:
        for arm in ARMS:
            for head in s106.HEADS:
                path = (root106/"models"/(group+"__"+head+".json") if arm == "baseline" else
                        root108/"models"/(group+"__relation_mode__"+head+".json"))
                model = c.c66.sealed(path, s106.SCHEMA if arm == "baseline" else s108.SCHEMA)
                if (model["head"] != head or model["held_out_group"] != group or group in model["fit_groups"]
                        or model["features"] != s108.feature_names(arm)
                        or (arm != "baseline" and model["representation"] != arm)):
                    raise ValueError("Wrong frozen gate identity")
                bank[group, arm, head] = model
                bindings[str(path)] = c.file_hash(path)
    return bank, bindings


def predict(shared, trace, entry, first, mode, old, choose, design, models, group):
    state = s105.initial_state(trace, entry, first, old, design)
    observation = {k: shared[k][0].copy() for k in s105.POINT_FIELDS}
    x, eligible, reason, actual, pair = s106.decision_features(observation, state, shared["anchor"], first, mode, choose, design)
    data = dict(x=x[None], eligible=np.array([eligible]), reason=np.array([reason]),
        four_selected=np.array([actual["selected"]]), pair_selected=np.array([pair["selected"]]))
    for arm in ARMS:
        for head in s106.HEADS:
            result = s108.predict_batch(models[group, arm, head], x[None], np.array([eligible]), arm)
            data.update({arm+"__"+head+"_"+k:v for k,v in result.items()})
    return data


def decisions(samples, predictions):
    bank = {(r["sequence"], r["seed"], r["source_arm"], r["frame"], r["representation"], r["head"]):r for r in predictions}
    bins = defaultdict(list)
    for r in samples:
        bins[r["sequence"], r["seed"], r["source_arm"]].append(r)
    rows = []
    for local in bins.values():
        def lookup(r, arm, head):
            return bank[r["sequence"], r["seed"], r["source_arm"], r["frame"], arm, head]
        eligible = [r for r in local if r["eligible"]]
        rates = {a:float(np.mean([lookup(r,a,"future")["commit"] for r in eligible])) if eligible else 0. for a in ARMS}
        for r in local:
            delta = r["future"]; valid = delta is not None
            choice = dict(keep=0., always_pair=float(r["eligible"]), future_oracle=float(valid and r["eligible"] and delta > 0))
            choice.update({a+"__"+h:float(lookup(r,a,h)["commit"]) for a in ARMS for h in s106.HEADS})
            choice.update({a+"__rate_expected":rates[a]*float(r["eligible"]) for a in ARMS})
            for policy in POLICIES:
                p = choice[policy]
                rows.append({k:r[k] for k in s107.IDENTITY} | dict(policy=policy, eligible=r["eligible"],
                    commit_probability=p, measured_delta=delta, utility=p*delta if valid else None,
                    harmful_commit=p*float(delta < 0) if valid else None,
                    missed_positive_utility=(1-p)*max(delta,0) if valid else None,
                    oracle_regret=float(r["eligible"])*max(delta,0)-p*delta if valid else None))
    return rows


def summaries(rows):
    summary, _ = s106.summarize(rows, POLICIES,
        ("eligible", "commit_probability", "utility", "harmful_commit", "missed_positive_utility", "oracle_regret"))
    paired = []
    controls = ("baseline__future", "baseline__myopic", "relation_mode__myopic", "always_pair", "relation_mode__rate_expected", "keep")
    for cohort in ("all", "historical", "expanded"):
        for mode in ("all", *s103.MODES[1:]):
            local = [r for r in rows if (cohort == "all" or r["cohort"] == cohort) and (mode == "all" or r["source_arm"] == mode)]
            if not local:
                continue
            values = {p:s106.macro([r for r in local if r["policy"] == p], "utility") for p in POLICIES}
            for control in controls:
                a, b = values["relation_mode__future"], values[control]
                groups = sorted(a.keys() & b.keys())
                d = np.array([a[g]-b[g] for g in groups])
                paired.append(dict(cohort=cohort, source_arm=mode, contrast="relation_mode__future_minus_"+control,
                    groups=len(d), mean_delta=float(d.mean()) if len(d) else None,
                    median_delta=float(np.median(d)) if len(d) else None, wins=int((d > 1e-12).sum()),
                    ties=int((np.abs(d) <= 1e-12).sum()), losses=int((d < -1e-12).sum()),
                    worst_group_delta=float(d.min()) if len(d) else None,
                    minimum_leave_one_group_out=float(min((d.sum()-v)/(len(d)-1) for v in d)) if len(d)>1 else None))
    return summary, paired


def cache(root, path, metadata, compute, evidence):
    data = c.read_cache(path, metadata)
    if data is None:
        data = compute()
        c.save(path, data, metadata)
    for bound in (path, path.with_suffix(".receipt.json")):
        evidence[bound.relative_to(root).as_posix()] = c.file_hash(bound)
    return data


def run(root, root103, p103, plan, load_unit, choose_for, old, observer, models):
    evidence, sampling_rows, scopes, branches, samples, predictions = {}, [], [], [], [], []
    design = p103["design"]
    for job in plan["units"]:
        validate_selection(job)
        spec, mode, sampling = job["spec"], job["source_arm"], job["sampling"]
        unit = spec["unit"]; group = s106.source_group(unit["sequence"])
        sampling_rows.append(s106.identity(spec,mode,0) | {k:v for k,v in sampling.items() if k not in ("pool", "selected", "excluded_ids")} |
            dict(pool_size=len(sampling["pool"]), sampled=len(sampling["selected"]), selected_frames="|".join(map(str,sampling["selected"]))))
        if not sampling["selected"]:
            continue
        entry, choose = load_unit(spec), choose_for(unit)
        trace = s105.prior.required_trace(root103,p103,unit,mode)
        s103.audit(trace,entry,old,choose,mode,design)
        with e63.source_experiment.installed(observer, [], unit) as head:
            if head != plan["native_head_sha256"]:
                raise ValueError("Original SGLA head changed")
            for first in sampling["selected"]:
                last = first+DESIGN["horizon"]-1
                ident = s106.identity(spec,mode,first) | dict(last=last)
                stem = sampling["context"]+f"__f{first:06d}"
                meta = dict(plan_fingerprint=plan["fingerprint"],unit=unit,source_arm=mode,first=first,last=last)
                shared = cache(root,root/"points"/(stem+".npz"),dict(meta,kind="point"),
                    lambda:s105.point(observer,trace,entry,first,old,choose,design),evidence)
                s105.audit_point(shared,trace,entry,first,mode,old,choose,design)
                bindings = {a+"__"+h:models[group,a,h]["fingerprint"] for a in ARMS for h in s106.HEADS}
                computed = predict(shared,trace,entry,first,mode,old,choose,design,models,group)
                pred = cache(root,root/"predictions"/(stem+".npz"),dict(meta,kind="frozen_gate_predictions",models=bindings),lambda:computed,evidence)
                if set(pred) != set(computed):
                    raise ValueError("Frozen prediction schema changed")
                for k in computed:
                    if k.endswith(("_score", "_max_abs_z")):
                        s107.close(pred[k],computed[k])
                    else:
                        np.testing.assert_array_equal(pred[k],computed[k])
                # Persist all four decisions before evaluating any new branch labels.
                local = {}
                for treatment in TREATMENTS:
                    data = cache(root,root/"branches"/(stem+"__"+treatment+".npz"),dict(meta,kind="branch",treatment=treatment),
                        lambda:s105.collect(observer,trace,shared,entry,first,last,treatment,mode,old,choose,design),evidence)
                    s105.audit(data,trace,shared,entry,first,last,treatment,mode,old,choose,design)
                    local[treatment] = data
                keep = local["keep"]
                source = {k:trace[k][first-1:last] for k in s103.FIELDS}
                error = float(np.abs(keep["boxes"]-source["boxes"]).max())
                if not plan["runtime_differences"]:
                    s105.compare_fields(keep,source,s103.FIELDS)
                s105.compare_fields({k:local["both_pair"][k][:1] for k in s103.FIELDS},
                    {k:keep[k][:1] for k in s103.FIELDS},s103.FIELDS)
                values = {t:s105.metrics(data,entry,design) for t,data in local.items()}
                for t,v in values.items():
                    branches.append(dict(ident,treatment=t,**v))
                v0,v1 = values["keep"],values["both_pair"]
                np.testing.assert_equal(v0["current_iou"],v1["current_iou"])
                future = v1["future_mean_iou"]-v0["future_mean_iou"] if all(v["future_mean_iou"] is not None for v in values.values()) else None
                myopic = v0["current_pair_iou"]-v0["current_iou"] if v0["current_iou"] is not None else None
                eligible = bool(pred["eligible"][0])
                samples.append(dict(ident,eligible=eligible,reason=str(pred["reason"][0]),future=future,myopic=myopic,
                    four_selected=int(pred["four_selected"][0]),pair_selected=int(pred["pair_selected"][0]),
                    future_valid_frames=v0["future_valid_frames"], inclusion_probability=sampling["inclusion_probability"],
                    x=s105.prior.encoded(pred["x"][0])))
                for arm in ARMS:
                    for h in s106.HEADS:
                        predictions.append(dict(ident,representation=arm,head=h,eligible=eligible,
                            **{k:pred[arm+"__"+h+"_"+k][0].item() for k in ("score","commit","outside_fit_range","max_abs_z")}))
                scopes.append(dict(ident,source_keep_box_error_px=error,shared_first_pass=True,sampled_cached_eligible=True,
                    fresh_eligible=eligible,eligibility_changed=not eligible,geometry_intervention_effective=bool(local["both_pair"]["geometry_commit_changed"][0]),
                    memory_intervention_effective=bool(local["both_pair"]["memory_commit_changed"][0]),
                    new_search_calls=4+8*(last-first),logical_search_calls=8*(last-first+1)))
                print(f"[Stage-109] {stem}: keep/pair {first}..{last}; fresh eligible={eligible}",flush=True)
    decision = decisions(samples,predictions)
    summary,paired = summaries(decision)
    publish(root,plan,dict(zip(TABLES,(sampling_rows,scopes,branches,samples,predictions,decision,summary,paired))),evidence)


def expected_counts(plan):
    b = budget(plan["units"]); n = b["anchors"]
    return dict(sampling_scope=b["contexts"],scope=n,branches=2*n,samples=n,state_predictions=4*n,sample_decisions=len(POLICIES)*n)


def publish(root,plan,tables,evidence):
    b = budget(plan["units"])
    if (set(tables) != set(TABLES) or any(len(tables[k]) != n for k,n in expected_counts(plan).items())
            or len(evidence) != 2*b["data_caches"] or sum(r["new_search_calls"] for r in tables["scope"]) != b["new_search_calls"]
            or sum(r["logical_search_calls"] for r in tables["scope"]) != b["logical_search_calls"]):
        raise ValueError("Incomplete sampled matched branches")
    io.verify(plan["source_sha256"]);io.verify(plan["code_sha256"])
    out=root/"analysis";out.mkdir(exist_ok=True);hashes={}
    for name in TABLES:
        path=out/f"stage109_{name}.csv";s103.s86.publish_table(path,tables[name]);hashes[path.name]=c.file_hash(path)
    primary=next(r for r in tables["paired"] if r["cohort"]==r["source_arm"]=="all" and r["contrast"]==DESIGN["primary"])
    c.publish_identical(out/"stage109_decision.json",io.seal(dict(schema=SCHEMA,status=STATUS,
        role="consumed_development_gt_free_sampled_one_step_commit_evidence",plan_fingerprint=plan["fingerprint"],
        source_stage108_fingerprint=SOURCE108,runtime=plan["runtime"],runtime_differences=plan["runtime_differences"],
        budget=b,new_block_calls=7*b["new_search_calls"],counts={k:len(v) for k,v in tables.items()},primary=primary,
        sampled_groups=len({r["group"] for r in tables["samples"]}),eligible_sampled_groups=len({r["group"] for r in tables["samples"] if r["eligible"]}),
        fresh_eligibility_changes=sum(r["eligibility_changed"] for r in tables["scope"]),
        labelled_future_samples=sum(r["future"] is not None for r in tables["samples"]),
        maximum_keep_source_box_error_px=max(r["source_keep_box_error_px"] for r in tables["scope"]),
        sampling_gt_free=True,decisions_before_branch_labels=True,matched_first_output_pass=True,
        one_step_intervention_only=True,weights_frozen=True,reused_gate_models=4*len(plan["groups"]),
        new_gate_models=0,training_updates=0,new_full_trajectories=0,confirmation_inference_calls=0,
        table_sha256=hashes,cache_sha256=evidence,warnings=[
            "Uniform within remaining eligible full-horizon contexts; not representative of all frames or independent confirmation.",
            "Previously labelled identities are excluded by availability; windows may overlap old and new windows.",
            "Source histories and old gates are fixed; fresh eligibility drift is retained, never resampled.",
            "One initial commit then the original verifier; utility is future IoU, not full-video AUC or repeated-gate performance.",
            "Own-rate expectation and future oracle are offline controls. No method adoption gate or frame-independent CI.",
            "Different-card source replay error is a bridge diagnostic; keep/pair are matched on the current bound runtime."])))
    check_results(root)
    print(f"Stage-109: {STATUS}; {b['anchors']} starts, {b['short_branches']} branches, {b['new_search_calls']} searches",flush=True)


def check_results(root):
    plan=c.c66.sealed(root/"manifest.json",SCHEMA);r=c.c66.sealed(root/"analysis/stage109_decision.json",SCHEMA)
    for job in plan["units"]:
        validate_selection(job)
    if (plan["design"] != DESIGN or r["status"] != STATUS or r["plan_fingerprint"] != plan["fingerprint"]
            or r["source_stage108_fingerprint"] != SOURCE108 or r["budget"] != budget(plan["units"])
            or r["runtime"] != plan["runtime"] or r["runtime_differences"] != plan["runtime_differences"]
            or r["new_block_calls"] != 7*r["budget"]["new_search_calls"] or r["reused_gate_models"] != 4*len(plan["groups"])
            or any(r["counts"][k] != n for k,n in expected_counts(plan).items())
            or set(r["table_sha256"]) != {f"stage109_{n}.csv" for n in TABLES}
            or len(r["cache_sha256"]) != 2*r["budget"]["data_caches"]
            or (not plan["runtime_differences"] and r["maximum_keep_source_box_error_px"] != 0.)
            or not all(r[k] for k in ("sampling_gt_free","decisions_before_branch_labels","matched_first_output_pass","one_step_intervention_only","weights_frozen"))
            or any(r[k] != 0 for k in ("new_gate_models","training_updates","new_full_trajectories","confirmation_inference_calls"))):
        raise ValueError("Incomplete Stage-109 result")
    io.verify(plan["source_sha256"]);io.verify(plan["code_sha256"])
    io.verify({str(root/"analysis"/n):h for n,h in r["table_sha256"].items()})
    io.verify({str(root/n):h for n,h in r["cache_sha256"].items()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    names=("stage76","stage89","stage93","stage103","stage106","stage108")
    for name in (*names,"output_dir"):
        p.add_argument("--"+name,type=Path,required=name=="output_dir")
    p.add_argument("--checkpoint");p.add_argument("--parameter",default="deit_distilled");p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--check-results",action="store_true");p.set_defaults(mode="stage109")
    args=p.parse_args();root=args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root);return
    if not all(getattr(args,n) for n in names) or not args.checkpoint:
        p.error("Supply Stage-76/89/93/103/106/108 and the original checkpoint")
    r76,r89,r93,r103,r106,r108=sources=[getattr(args,n).expanduser().resolve() for n in names]
    if any(root==s or root in s.parents or s in root.parents for s in sources):
        p.error("Use a separate Stage-109 output directory")
    with s106.s93.writer_lock(root):
        if (root/"analysis/stage109_decision.json").exists():
            check_results(root);return
        if not (root/"manifest.json").exists() and any(x.name!="collect.lock" for x in root.iterdir()):
            raise ValueError("Use an empty Stage-109 directory")
        print("Stage-109: checking sources and freezing GT-free sample identities before inference",flush=True)
        s108.check_results(r108);s106.check_results(r106);s103.check_results(r103)
        p108=c.c66.sealed(r108/"manifest.json",s108.SCHEMA);report108=c.c66.sealed(r108/"analysis/stage108_decision.json",s108.SCHEMA)
        p106=c.c66.sealed(r106/"manifest.json",s106.SCHEMA)
        report106=c.c66.sealed(r106/"analysis/stage106_decision.json",s106.SCHEMA)
        p103=c.c66.sealed(r103/"manifest.json",s103.SCHEMA);report103=c.c66.sealed(r103/"analysis/stage103_decision.json",s103.SCHEMA)
        if (report108["fingerprint"]!=SOURCE108 or report106["fingerprint"]!=s107.SOURCE106
                or report103["fingerprint"]!=s105.prior.SOURCE103 or p108["groups"]!=p106["groups"]
                or len(p103["units"])!=40 or len(p106["groups"])!=28):
            raise ValueError("Need the declared complete development sources")
        p93,_,p89,report89,source=s103.prior.s95.s94.load_run(r93,r76,r89)
        actor=io.load_source(Path(source["plan"]["source_input"]))
        if p103["units"]!=p89["units"] or actor["plan"]!=source["original"] or actor["models"]["fingerprint"]!=source["report"]["source_training_fingerprint"]:
            raise ValueError("Frozen development units or PPO changed")
        known=s107.read_csv(r106/"analysis/stage106_samples.csv")
        if len(known)!=259:
            raise ValueError("Prior label-availability ledger changed")
        jobs,bindings=prepare(r103,r106,p103,p106,known)
        models,model_bindings=load_models(r106,r108,p106["groups"]);bindings.update(model_bindings)
        import torch
        torch.set_num_threads(1)
        runtime=s103.s76.device_runtime(source["original"],torch,args.gpu)
        previous=c.c66.sealed(root/"manifest.json",SCHEMA) if (root/"manifest.json").exists() else None
        differences=check_runtime(report103["runtime"],runtime,previous["runtime"] if previous else None)
        bindings.update(p108["source_sha256"]);bindings.update(actor["bindings"])
        for source_root,number in ((r103,103),(r106,106),(r108,108)):
            for path in (source_root/"manifest.json",source_root/f"analysis/stage{number}_decision.json"):
                bindings[str(path)]=c.file_hash(path)
        bindings[str(r106/"analysis/stage106_samples.csv")]=c.file_hash(r106/"analysis/stage106_samples.csv")
        code=[Path(__file__).resolve(),Path(__file__).with_name("run_stage109_sampled_commit.sh").resolve()]
        plan=io.seal(dict(schema=SCHEMA,design=DESIGN,units=jobs,groups=p106["groups"],runtime=runtime,runtime_differences=differences,
            native_head_sha256=source["report"]["native_head_sha256"],source_sha256=bindings,
            code_sha256=p108["code_sha256"]|p103["code_sha256"]|{str(x):c.file_hash(x) for x in code}))
        c.publish_identical(root/"manifest.json",plan)
        print("Stage-109 frozen:",budget(jobs),"runtime differences:",differences,flush=True)
        def load_unit(spec):
            return s103.prior.s95.load_unit(r93,r89,p93,p89,report89,source,spec)[0]
        def choose_for(unit):
            pack=s103.e74.learned_pack(actor["models"],unit,actor["prior"])
            return lambda row,history:s103.e74.choose(row,history,unit,source["original"],actor["prior"],pack)
        frozen=dict(source["original"],fingerprint=plan["fingerprint"],runtime_identity=runtime,runtime=runtime["tracker"])
        native,_=s103.s76.startup.checked_runtime(args,frozen,root);observer=e63.Capture(native)
        try:
            run(root,r103,p103,plan,load_unit,choose_for,source["original"],observer,models)
        finally:
            observer.handle.remove()


if __name__=="__main__":
    main()
