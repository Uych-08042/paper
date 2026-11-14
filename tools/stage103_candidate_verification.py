"""Compare paired and four-candidate appearance verification on full SGLA trajectories."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import stage102_state_chain_audit as source102
from stage27_uav_common import checked_image, crop_geometry, failure_events, tracking_iou


prior = source102.prior
c, io, s93, s86, e63, e74, s76 = prior.c, prior.io, prior.s93, prior.s86, prior.e63, prior.e74, prior.s76
SCHEMA = "stage103_four_candidate_appearance_verification_v1"
SOURCE102 = "2385b6d968cb5aa69ae89ad055a87b02a0591472eedcdb8a9e5fada4454fcf4b"
DESIGN = dict(s93.DESIGN, bootstrap_seed=2203, bootstrap_samples=10000)
MODES = ("pair", "four_matched", "four_pairwise")
ARMS = ("native", "ppo")+MODES
EXTRA = ("all_queries", "ppo_selected", "pair_selected", "legacy_challenger", "legacy_in_domain",
         "legacy_used_history", "refs", "used_history", "reference_kind", "support_mask", "scores", "gains", "memory_write")
FIELDS = s86.FIELDS+EXTRA
METRICS = s93.METRICS+("failure_events", "good_candidate_fraction", "selected_good_fraction", "missed_good_fraction")
FLAGS = ("new_on_native_stable", "new_on_ppo_stable", "new_on_pair_stable", "ppo_recoveries_lost", "pair_recoveries_lost")
TABLES = ("policy_summary", "paired", "video_summary", "seed_video", "scope", "recovery", "matched", "matched_candidates")
NEGATIVE = {"low_iou_fraction", "longest_failure_fraction", "failure_events", "unrecovered_events", "missed_good_fraction"}


def pair_queries(row, queries, design):
    alt, _ = s93.appearance_pair(row, design)
    out = np.full((2, queries.shape[1]), np.nan)
    out[0] = queries[int(row["selected"])]
    if alt >= 0:
        out[1] = queries[alt]
    return out


def decide(row, queries, anchor, memory, frame, mode, design):
    """Use own past descriptors only; geometry commits remain unconditional in all arms."""
    if mode not in MODES or queries.shape != (4, len(anchor)):
        raise ValueError("Unknown verifier or incomplete candidate descriptors")
    ppo = int(row["selected"])
    legacy = s93.choose_write(row, pair_queries(row, queries, design), anchor, list(memory), frame, "gated", design)
    reference = s93.select_memory(memory, frame, design)
    usable = [q for _, q in reference if np.isfinite(q).all()]
    legal = c.action_mask(row["candidate_boxes"])
    finite = np.isfinite(queries).all(1)
    mask = np.zeros(4, bool)
    mask[ppo] = True
    selected, used = legacy["selected"], bool(legacy["used_history"])
    if mode == "pair":
        alt = legacy["challenger"]
        if legacy["in_domain"] and alt >= 0 and finite[[ppo, alt]].all():
            mask[alt] = True
    elif mode == "four_matched":
        # Keep the original pair's domain, finite-query gate and reference choice.
        if legacy["in_domain"] and np.isfinite(pair_queries(row, queries, design)).all():
            mask |= legal & finite & np.any(row["candidate_boxes"] != row["candidate_boxes"][ppo], axis=1)
    else:
        peaks = row["features"][:, e63.source_common.prior.FEATURES.index("hann_peak")]
        distance = np.linalg.norm(s86.centers(row["candidate_boxes"])-s86.centers(row["candidate_boxes"][ppo]), axis=1)
        distance /= np.sqrt(np.prod(row["history_latest"][2:]))
        supported = (distance < design["center_separation"]) | (np.abs(peaks-peaks[ppo]) <= design["hann_gap"])
        if finite[ppo]:
            mask |= legal & finite & supported & np.any(row["candidate_boxes"] != row["candidate_boxes"][ppo], axis=1)
        used = bool(mask.sum() > 1 and len(usable) >= design["min_refs"])
    vector = np.mean(usable, axis=0) if used else anchor
    scores = queries @ vector
    gains = (queries-queries[ppo]) @ vector
    if mode != "pair":
        selected = ppo
        # Preserve the old challenger on exact alternative ties.
        order = list(dict.fromkeys([legacy["challenger"]]+list(range(4))))
        alternatives = [q for q in order if q >= 0 and q != ppo and mask[q] and np.isfinite(gains[q])]
        if alternatives:
            best = max(alternatives, key=lambda q: gains[q])
            if gains[best] > design["switch_margin"]:
                selected = best
    write = s93.s92.trusted(s93.action_bank(dict(row, selected=selected)), 0, design)
    if write:
        memory.append((frame, queries[selected].copy()))
    memory[:] = [(f, q) for f, q in memory if frame-f < design["lookback"]]
    return dict(selected=int(selected), ppo_selected=ppo, pair_selected=int(legacy["selected"]),
        legacy_challenger=int(legacy["challenger"]), legacy_in_domain=bool(legacy["in_domain"]),
        legacy_used_history=bool(legacy["used_history"]), refs=legacy["refs"], used_history=used,
        reference_kind="history" if used else "anchor", support_mask=mask, scores=scores, gains=gains, memory_write=bool(write))


def observe(observer, image, history, parents, frame, ap, choose, design):
    tokens = []
    def capture(_module, _args, output):
        tokens.append(s93.s91.cpu_array(output)[0])
    with observer.rt.net.backbone.patch_embed.register_forward_hook(capture):
        row = choose(prior.s97.probe(observer, image, history, parents, frame, ap), history)
    if len(tokens) != 8 or any(len(t) != (observer.rt.lens_z if i % 2 == 0 else observer.rt.lens_x) for i,t in enumerate(tokens)):
        raise ValueError("Expected exactly four template/search grids")
    queries = np.asarray([s93.pool(image, row["candidate_boxes"][q], parents[q], ap["search_factor"], ap["search_size"], tokens[2*q+1], design)
                          for q in range(4)])
    return row, queries, tokens[0]


def collect(observer, entry, old, choose, mode, design, image_reader=checked_image):
    ap = e63.c.actor_plan(old)
    template = image_reader(ap["root"], entry["frames"][0])
    initial = np.asarray(entry["initial_box"], float)
    observer.initialize(template, initial)
    history, memory, rows, anchor = [initial], [], [], None
    for frame, item in enumerate(entry["frames"][1:], 1):
        image = image_reader(ap["root"], item)
        parents = e63.source_common.prior.history_parents(history, ap["design"])
        row, queries, template_grid = observe(observer, image, history, parents, frame, ap, choose, design)
        if anchor is None:
            params = observer.rt.tracker.params
            anchor = s93.pool(template, initial, initial, params.template_factor, params.template_size, template_grid, design)
        decision = decide(row, queries, anchor, memory, frame, mode, design)
        box = row["candidate_boxes"][decision["selected"]].copy()
        rows.append(row | decision | dict(boxes=box, all_queries=queries))
        history = (history+[box])[-ap["design"]["history"]:]
        observer.rt.tracker.state = box.tolist()
        if frame % 500 == 0:
            print(f"  [Stage-103] {mode} {frame}/{len(entry['frames'])-1}", flush=True)
    return {k: np.asarray([r[k] for r in rows]) for k in FIELDS} | dict(anchor=anchor)


def convert_pair(trace, entry, old, choose, design):
    """Reuse the old pair without fabricating descriptors for its unobserved candidates."""
    ap = e63.c.actor_plan(old)
    history, memory, rows = [np.asarray(entry["initial_box"], float)], [], []
    for i, box in enumerate(trace["boxes"]):
        row = {k: trace[k][i].copy() for k in s86.FIELDS}
        row = choose(row, history)
        queries = np.full((4,len(trace["anchor"])), np.nan)
        alt = int(trace["challenger"][i])
        queries[int(row["selected"])] = trace["queries"][i,0]
        if alt >= 0:
            queries[alt] = trace["queries"][i,1]
        decision = decide(row, queries, trace["anchor"], memory, i+1, "pair", design)
        np.testing.assert_array_equal(decision["selected"], trace["selected"][i])
        for key in ("ppo_selected", "memory_write", "used_history", "refs"):
            np.testing.assert_array_equal(decision[key], trace[key][i])
        rows.append(row | decision | dict(boxes=box.copy(), all_queries=queries))
        history = (history+[box])[-ap["design"]["history"]:]
    return {k:np.asarray([r[k] for r in rows]) for k in FIELDS} | dict(anchor=trace["anchor"].copy())


def audit(data, entry, old, choose, mode, design):
    n, ap = len(entry["frames"])-1, e63.c.actor_plan(old)
    if set(data) != set(FIELDS) | {"anchor"} or data["all_queries"].shape != (n,4,len(data["anchor"])):
        raise ValueError("Incomplete four-candidate verifier cache")
    np.testing.assert_array_equal(data["frame_id"], np.arange(1,n+1))
    e63.source_common.prior.audit_calls(data, (n,4))
    if not np.isfinite(data["decode_error"]).all() or data["decode_error"].max() > ap["design"]["box_atol"]:
        raise ValueError("Decode drift")
    history, memory = [np.asarray(entry["initial_box"], float)], []
    for i, box in enumerate(data["boxes"]):
        hw = [entry["frames"][i+1]["height"], entry["frames"][i+1]["width"]]
        np.testing.assert_array_equal(data["image_hw"][i], hw)
        parents = e63.source_common.prior.history_parents(history, ap["design"])
        np.testing.assert_array_equal(data["candidate_parents"][i], parents)
        np.testing.assert_array_equal(data["crop"][i], [crop_geometry(p,hw,ap["search_factor"],ap["search_size"])[0] for p in parents])
        np.testing.assert_array_equal(data["history_latest"][i], history[-1])
        row = choose({k:data[k][i].copy() for k in ("features","candidate_boxes","candidate_parents")}, history)
        row["history_latest"] = history[-1]
        decision = decide(row, data["all_queries"][i], data["anchor"], memory, i+1, mode, design)
        for key, value in decision.items():
            if key in ("scores","gains"):
                np.testing.assert_allclose(data[key][i],value,rtol=0,atol=1e-12,equal_nan=True)
            else:
                np.testing.assert_array_equal(data[key][i],value)
        np.testing.assert_array_equal(box, data["candidate_boxes"][i,int(decision["selected"])])
        history = (history+[box])[-ap["design"]["history"]:]


def load_source_unit(root100, plan100, root93, root89, plan93, plan89, report89, source, spec):
    entry, refs = prior.s95.load_unit(root93, root89, plan93, plan89, report89, source, spec)
    unit = spec["unit"]
    pulse = prior.s95.s94.required_cache(root100 / "evaluation" / f"{c.key(unit)}__pulse.npz",
        dict(plan_fingerprint=plan100["fingerprint"],unit=unit,arm="pulse"))
    return entry, refs, pulse


def load_native(root100, plan100, entry, spec, ap):
    unit = dict(spec["unit"],seed=0,policy="q0")
    data = prior.s95.s94.required_cache(root100 / "evaluation" / f"{c.key(unit)}__native.npz",
        dict(plan_fingerprint=plan100["fingerprint"],unit=unit,arm="native"))
    return unit, prior.compat.native_view(data,entry,ap)


def first_commit_difference(data, baseline):
    changed = np.flatnonzero(np.any(data["boxes"] != baseline["boxes"],axis=1))
    stop = int(changed[0]) if len(changed) else len(data["boxes"])
    for key in ("candidate_parents","candidate_boxes","features","crop"):
        np.testing.assert_array_equal(data[key][:stop+1],baseline[key][:stop+1],err_msg="Pre-commit drift:"+key)
    return stop+1 if len(changed) else None


def contrast(values, label, metric, design, family=False):
    values = np.asarray(values,float)
    draws = np.random.default_rng(design["bootstrap_seed"]).integers(0,len(values),(design["bootstrap_samples"],len(values)))
    boot = values[draws].mean(1)
    sign = -1 if metric in NEGATIVE else 1
    loo = (values.sum()-values)/(len(values)-1) if len(values)>1 else np.array([])
    return dict(contrast=label,metric=metric,videos=len(values),mean_delta=float(values.mean()),median_delta=float(np.median(values)),
        ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),
        family_ci_low=float(np.quantile(boot,.0125)) if family else None,
        family_ci_high=float(np.quantile(boot,.9875)) if family else None,
        wins=int((sign*values>1e-12).sum()),ties=int((np.abs(values)<=1e-12).sum()),losses=int((sign*values< -1e-12).sum()),
        favorable_direction="positive" if sign>0 else "negative",
        worst_video_delta=float(values.min() if sign>0 else values.max()),
        worst_leave_one_video_out_delta=float(loo.min() if sign>0 else loo.max()) if len(loo) else None)


def summarize(seeds, design):
    grouped = defaultdict(list)
    for r in seeds:
        grouped[r["cohort"],r["sequence"],r["arm"]].append(r)
    videos = []
    for (cohort,seq,arm),rows in sorted(grouped.items()):
        videos.append(dict(cohort=cohort,sequence=seq,arm=arm,
            **{k:float(np.mean(v)) if (v:=[r[k] for r in rows if r[k] is not None]) else None for k in METRICS},
            **{k:(sum(v) if k.endswith("recoveries_lost") else any(v)) if (v:=[r[k] for r in rows if r[k] is not None]) else None for k in FLAGS}))
    policies, pairs = [], []
    contrasts = [(a,b) for a in MODES[1:] for b in ("pair","ppo","native")]+[("four_pairwise","four_matched")]
    for cohort in ("all","historical","expanded"):
        local = [r for r in videos if cohort=="all" or r["cohort"]==cohort]
        indexed = {(r["sequence"],r["arm"]):r for r in local}
        names = sorted({r["sequence"] for r in local})
        for arm in ARMS:
            rows = [r for r in local if r["arm"]==arm]
            if rows:
                policies.append(dict(cohort=cohort,arm=arm,videos=len(rows),
                    **{k:float(np.mean(v)) if (v:=[r[k] for r in rows if r[k] is not None]) else None for k in METRICS},
                    **{k:sum(v) if (v:=[r[k] for r in rows if r[k] is not None]) else None for k in FLAGS}))
        for a,b in contrasts:
            for metric in METRICS:
                delta = [indexed[n,a][metric]-indexed[n,b][metric] for n in names
                         if indexed[n,a][metric] is not None and indexed[n,b][metric] is not None]
                if delta:
                    pairs.append(dict(cohort=cohort,**contrast(delta,f"{a}_minus_{b}",metric,design,
                        family=cohort=="all" and b=="pair" and metric=="auc")))
    return policies,pairs,videos


def matches_by_key(recoveries, arm, reference):
    return {(r["sequence"],r["seed"],r["ppo_failure_frame"],r["ppo_recovery_frame"])
            for r in recoveries if r["arm"]==arm and r["reference_arm"]==reference and r["matched_good_frame"] is not None}


def publish(root,plan,evidence,seeds,scope,recovery,matched,candidates):
    if len(scope)!=plan["trajectories"] or len(seeds)!=len(scope) or len(evidence)!=len(scope)+len(plan["points"]):
        raise ValueError("Incomplete Stage-103 coverage")
    policies,pairs,videos = summarize(seeds,plan["design"])
    macro = {r["arm"]:r for r in policies if r["cohort"]=="all"}
    effects = {r["contrast"]:r for r in pairs if r["cohort"]=="all" and r["metric"]=="auc"}
    checks = {}
    def damaged(arm, flag):
        return {(r["sequence"],r["seed"]) for r in seeds if r["arm"]==arm and r[flag]}
    for arm in MODES[1:]:
        effect = effects[arm+"_minus_pair"]
        a,b = macro[arm],macro["pair"]
        checks[arm] = dict(practical_auc=effect["mean_delta"]>=plan["design"]["practical_auc"],
            family_ci_positive=effect["family_ci_low"]>0,
            low_iou_not_worse=a["low_iou_fraction"]<=b["low_iou_fraction"],
            longest_failure_not_worse=a["longest_failure_fraction"]<=b["longest_failure_fraction"],
            reliable_auc_not_worse=a["reliable_auc"]>=b["reliable_auc"],
            pair_stable_protected=a["new_on_pair_stable"]==0,
            no_new_native_stable_loss=not (damaged(arm,"new_on_native_stable")-damaged("pair","new_on_native_stable")),
            no_new_ppo_stable_loss=not (damaged(arm,"new_on_ppo_stable")-damaged("pair","new_on_ppo_stable")),
            pair_recovery_protected=a["pair_recoveries_lost"]==0,
            no_new_ppo_recovery_loss=not (matches_by_key(recovery,"pair","ppo")-matches_by_key(recovery,arm,"ppo")),
            nontrivial_extra_selection=sum(r["outside_pair_frames"] for r in scope if r["arm"]==arm)>0)
    io.verify(plan["source_sha256"]);io.verify(plan["code_sha256"])
    out=root/"analysis";out.mkdir(exist_ok=True)
    hashes={}
    for name,rows in zip(TABLES,(policies,pairs,videos,seeds,scope,recovery,matched,candidates)):
        path=out/f"stage103_{name}.csv";s86.publish_table(path,rows);hashes[path.name]=c.file_hash(path)
    calls=sum(r["new_search_calls"] for r in scope)+4*len(plan["points"])
    c.publish_identical(out/"stage103_decision.json",io.seal(dict(schema=SCHEMA,
        status="PROMISING_FOUR_CANDIDATE_VERIFICATION_DEVELOPMENT" if any(all(v.values()) for v in checks.values()) else "REVIEW_FOUR_CANDIDATE_VERIFICATION_DEVELOPMENT",
        plan_fingerprint=plan["fingerprint"],source_stage102_fingerprint=SOURCE102,runtime=plan["runtime"],
        runtime_differences=plan["runtime_differences"],reuse_sources=plan["reuse_sources"],
        trajectories=len(scope),new_full_trajectories=sum(r["new_search_calls"]>0 for r in scope),
        matched_states=len(plan["points"]),new_search_calls=calls,new_block_calls=7*calls,
        development_checks=checks,training_updates=0,confirmation_inference_calls=0,
        maximum_source_control_box_error_px=max(r["source_box_error_px"] for r in scope if r["source_box_error_px"] is not None),
        shared_pre_commit_pass=True,matched_query_checks_pass=True,weights_frozen=True,
        geometry_commit_unchanged=True,table_sha256=hashes,cache_sha256=evidence)))
    check_results(root)
    for arm in ARMS:
        print(f"{arm:16s} AUC={macro[arm]['auc']:.4f} low-IoU={100*macro[arm]['low_iou_fraction']:.2f}%")


def check_results(root):
    plan=c.c66.sealed(root/"manifest.json",SCHEMA)
    report=c.c66.sealed(root/"analysis/stage103_decision.json",SCHEMA)
    if (report["plan_fingerprint"]!=plan["fingerprint"] or report["source_stage102_fingerprint"]!=SOURCE102
            or report["status"] not in ("REVIEW_FOUR_CANDIDATE_VERIFICATION_DEVELOPMENT","PROMISING_FOUR_CANDIDATE_VERIFICATION_DEVELOPMENT")
            or report["runtime"]!=plan["runtime"] or report["runtime_differences"]!=plan["runtime_differences"]
            or report["reuse_sources"]!=plan["reuse_sources"]
            or (plan["reuse_sources"] and report["maximum_source_control_box_error_px"]!=0.)
            or report["trajectories"]!=plan["trajectories"] or report["new_full_trajectories"]!=plan["new_trajectories"]
            or report["new_search_calls"]!=plan["new_search_calls"] or report["new_block_calls"]!=7*plan["new_search_calls"]
            or len(report["cache_sha256"])!=plan["trajectories"]+len(plan["points"])
            or report["matched_states"]!=len(plan["points"])
            or any(report[k]!=0 for k in ("training_updates","confirmation_inference_calls"))
            or not all(report[k] for k in ("shared_pre_commit_pass","matched_query_checks_pass","weights_frozen","geometry_commit_unchanged"))
            or set(report["table_sha256"])!={f"stage103_{n}.csv" for n in TABLES}):
        raise ValueError("Incomplete Stage-103 result")
    io.verify({str(root/p):h for p,h in report["cache_sha256"].items()})
    io.verify({str(root/"analysis"/p):h for p,h in report["table_sha256"].items()})
    io.verify(plan["source_sha256"]);io.verify(plan["code_sha256"])


def matched_point(observer,entry,trace,frame,old,choose,design,image_reader=checked_image):
    ap=e63.c.actor_plan(old)
    observer.initialize(image_reader(ap["root"],entry["frames"][0]),entry["initial_box"])
    history=list(e63.history_at(trace,entry["initial_box"],frame,old))
    row,queries,_=observe(observer,image_reader(ap["root"],entry["frames"][frame]),history,
        trace["candidate_parents"][frame-1].copy(),frame,ap,choose,design)
    return {k:np.asarray(row[k])[None] for k in s86.FIELDS} | dict(all_queries=queries[None])


def run(root,root100,plan100,root93,root89,plan93,plan89,report89,source,actor,observer,plan):
    old,design=source["original"],plan["design"]
    ap=e63.c.actor_plan(old)
    evidence,seeds,scope,recovery,matched,candidates={},[],[],[],[],[]
    native_by_sequence={}
    for spec in plan["units"]:
        unit,cohort=spec["unit"],spec["cohort"]
        entry,refs,pulse=load_source_unit(root100,plan100,root93,root89,plan93,plan89,report89,source,spec)
        pack=e74.learned_pack(actor["models"],unit,actor["prior"])
        choose=lambda row,history:e74.choose(row,history,unit,old,actor["prior"],pack)
        native_unit,source_native=load_native(root100,plan100,entry,spec,ap)
        traces={}
        arms=ARMS if unit["sequence"] not in native_by_sequence else ARMS[1:]
        with e63.source_experiment.installed(observer,[],unit) as head:
            if head!=source["report"]["native_head_sha256"]:raise ValueError("SGLA head changed")
            for arm in arms:
                owner=native_unit if arm=="native" else unit
                path=root/"evaluation"/f"{c.key(owner)}__{arm}.npz"
                meta=dict(plan_fingerprint=plan["fingerprint"],unit=owner,arm=arm)
                data=c.read_cache(path,meta);fresh=data is None
                reused=plan["reuse_sources"] and arm in ("native","ppo","pair")
                original=source_native if arm=="native" else refs.get("gated" if arm=="pair" else arm)
                if fresh:
                    if reused:
                        data=convert_pair(original,entry,old,choose,design) if arm=="pair" else original
                    elif arm=="native":data=s93.s85.collect(observer,entry,"native_sgla",ap,None,{})
                    elif arm=="ppo":data=s86.collect(observer,entry,np.asarray([entry["initial_box"]]),np.arange(1,len(entry["frames"])),0.,ap,choose)
                    else:data=collect(observer,entry,old,choose,arm,design)
                if arm=="native":
                    s93.s85.audit(data,entry,"native_sgla",ap,None,{})
                    native_by_sequence[unit["sequence"]]=data
                elif arm=="ppo":s86.audit({k:data[k] for k in s86.FIELDS},np.asarray([entry["initial_box"]]),np.arange(1,len(entry["frames"])),0.,ap,choose)
                else:audit(data,entry,old,choose,arm,design)
                if fresh:c.save(path,data,meta)
                evidence[path.relative_to(root).as_posix()]=c.file_hash(path)
                traces[arm]=data
                drift=float(np.abs(data["boxes"]-original["boxes"]).max()) if original is not None else None
                if reused and drift!=0:raise ValueError("Reused source drifted")
                first=first_commit_difference(data,traces["pair"]) if arm in MODES[1:] else None
                scope.append(dict(cohort=cohort,sequence=unit["sequence"],seed=owner["seed"],arm=arm,frames=len(data["frame_id"]),
                    new_search_calls=0 if reused else len(data["frame_id"])*(1 if arm=="native" else 4),
                    source_box_error_px=drift,first_output_difference_from_pair=first,
                    native_source_slots=int(source_native.get("native_source_slots",1)) if arm=="native" else None,
                    native_source_head_calls=int(source_native.get("native_source_head_calls",len(data["frame_id"]))) if arm=="native" else None,
                    outside_pair_frames=int(np.count_nonzero((data["selected"]!=data["ppo_selected"]) & (data["selected"]!=data["legacy_challenger"]))) if arm in MODES else 0))
            native=native_by_sequence[unit["sequence"]]
            gt=np.asarray(entry["ground_truth"],float)[1:]
            pair_iou=tracking_iou(traces["pair"]["boxes"],gt)
            for arm in arms:
                owner=native_unit if arm=="native" else unit
                data=traces[arm]
                row,events=s93.describe_arm(entry,cohort,owner,arm,data,native,traces["ppo"],design)
                overlap=tracking_iou(data["boxes"],gt)
                bank=tracking_iou(data["candidate_boxes"],gt[:,None])
                pr=s93.recovery_comparison(pair_iou,overlap,design) if arm!="native" else []
                row.update(failure_events=len(failure_events(overlap)),good_candidate_fraction=float((bank.max(1)>=design["good_iou"]).mean()),
                    selected_good_fraction=float((overlap>=design["good_iou"]).mean()),
                    missed_good_fraction=float(((bank.max(1)>=design["good_iou"]) & (overlap<design["good_iou"])).mean()),
                    new_on_pair_stable=not failure_events(pair_iou) and bool(failure_events(overlap)) if arm!="native" else None,
                    pair_recoveries_lost=sum(r["matched_good_frame"] is None for r in pr) if arm!="native" else None)
                seeds.append(row);recovery.extend(dict(r,reference_arm="ppo") for r in events)
                recovery.extend(dict(cohort=cohort,sequence=unit["sequence"],seed=owner["seed"],arm=arm,reference_arm="pair",**r) for r in pr)
            for point in (p for p in plan["points"] if p["sequence"]==unit["sequence"] and p["seed"]==unit["seed"]):
                f=point["frame"]
                path=root/"points"/f"{c.key(unit)}__f{f:06d}.npz"
                meta=dict(plan_fingerprint=plan["fingerprint"],unit=unit,frame=f)
                data=c.read_cache(path,meta);fresh=data is None
                if fresh:data=matched_point(observer,entry,pulse,f,old,choose,design)
                if set(data)!=set(s86.FIELDS)|{"all_queries"} or data["all_queries"].shape!=(1,4,len(pulse["anchor"])):
                    raise ValueError("Incomplete matched descriptors")
                np.testing.assert_array_equal(data["frame_id"],[f])
                e63.source_common.prior.audit_calls(data,(1,4))
                if not np.isfinite(data["decode_error"]).all() or data["decode_error"].max()>ap["design"]["box_atol"]:
                    raise ValueError("Matched decode drift")
                np.testing.assert_array_equal(data["candidate_parents"][0],pulse["candidate_parents"][f-1])
                if plan["reuse_sources"]:
                    for key in ("candidate_boxes","features","crop","crop_sha256"):
                        np.testing.assert_array_equal(data[key][0],pulse[key][f-1])
                if fresh:c.save(path,data,meta)
                evidence[path.relative_to(root).as_posix()]=c.file_hash(path)
                history=list(e63.history_at(pulse,entry["initial_box"],f,old))
                memory,_=prior.s97.starting_memory(pulse,f,design)
                row=choose({k:data[k][0].copy() for k in s86.FIELDS},history)
                np.testing.assert_array_equal(data["selected"][0],row["selected"])
                np.testing.assert_array_equal(data["history_latest"][0],history[-1])
                if plan["reuse_sources"]:
                    np.testing.assert_allclose(pair_queries(row,data["all_queries"][0],design),pulse["queries"][f-1],rtol=0,atol=1e-12,equal_nan=True)
                q=tracking_iou(row["candidate_boxes"],entry["ground_truth"][f])
                for mode in MODES:
                    d=decide(row,data["all_queries"][0],pulse["anchor"],list(memory),f,mode,design)
                    if mode=="pair" and plan["reuse_sources"]:
                        np.testing.assert_array_equal(d["selected"],pulse["selected"][f-1])
                    matched.append(dict(cohort=cohort,**point,arm=mode,selected=d["selected"],pair_selected=d["pair_selected"],
                        iou=float(q[d["selected"]]),best_iou=float(q.max()),used_history=d["used_history"],
                        reference_kind=d["reference_kind"],support_mask="|".join(str(int(v)) for v in d["support_mask"]),
                        source_selected=int(pulse["selected"][f-1]),
                        candidate_max_source_box_error_px=float(np.abs(row["candidate_boxes"]-pulse["candidate_boxes"][f-1]).max())))
                    for j in range(4):
                        candidates.append(dict(**point,arm=mode,candidate=j,iou=float(q[j]),eligible=bool(d["support_mask"][j]),
                            descriptor_finite=bool(np.isfinite(data["all_queries"][0,j]).all()),
                            appearance_score=float(d["scores"][j]) if np.isfinite(d["scores"][j]) else None))
        print(f"[Stage-103] {unit['sequence']} seed{unit['seed']} complete",flush=True)
    publish(root,plan,evidence,seeds,scope,recovery,matched,candidates)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("stage76","stage89","stage93","stage100","stage102","output_dir"):
        p.add_argument("--"+name,type=Path,required=name=="output_dir")
    p.add_argument("--checkpoint");p.add_argument("--parameter",default="deit_distilled");p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--check-results",action="store_true");p.set_defaults(mode="stage103")
    args=p.parse_args();root=args.output_dir.expanduser().resolve()
    if args.check_results:check_results(root);return
    names=("stage76","stage89","stage93","stage100","stage102")
    if not all(getattr(args,n) for n in names) or not args.checkpoint:p.error("Supply Stage-76/89/93/100/102 and original checkpoint")
    root76,root89,root93,root100,root102=sources=[getattr(args,n).expanduser().resolve() for n in names]
    if any(root==s or root in s.parents or s in root.parents for s in sources):p.error("Use separate Stage-103 output")
    with s93.writer_lock(root):
        if (root/"analysis/stage103_decision.json").exists():check_results(root);return
        if not (root/"manifest.json").exists() and any(x.name!="collect.lock" for x in root.iterdir()):raise ValueError("Use an empty Stage-103 directory")
        source102.check_results(root102)
        report102=c.c66.sealed(root102/"analysis/stage102_decision.json",source102.SCHEMA)
        plan102=c.c66.sealed(root102/"manifest.json",source102.SCHEMA)
        if report102["fingerprint"]!=SOURCE102:raise ValueError("Need frozen Stage-102 result")
        prior.compat.check_results(root100)
        plan100=c.c66.sealed(root100/"manifest.json",prior.s100.SCHEMA)
        report100=c.c66.sealed(root100/"analysis/stage100_decision.json",prior.s100.SCHEMA)
        if report100["fingerprint"]!=prior.SOURCE100:raise ValueError("Stage-100 changed")
        plan93,_,plan89,report89,source=prior.s95.s94.load_run(root93,root76,root89)
        actor=io.load_source(Path(source["plan"]["source_input"]))
        if actor["plan"]!=source["original"] or actor["models"]["fingerprint"]!=source["report"]["source_training_fingerprint"]:raise ValueError("Frozen PPO changed")
        import torch
        torch.set_num_threads(1)
        runtime=s76.device_runtime(source["original"],torch,args.gpu)
        differences=s76.startup.differences(report100["runtime"],runtime)
        if any(r["field"] not in ("runtime.gpu_uuid","runtime.tracker.device") for r in differences):raise ValueError("Frozen software or weights changed")
        roles=prior.read_csv(root102/"analysis/stage102_cases.csv");points={}
        for r in roles:
            key=(r["sequence"],int(r["seed"]),int(r["target_frame"]))
            row=points.setdefault(key,dict(sequence=key[0],seed=key[1],frame=key[2],roles=[]))
            row["roles"].append(r["role"])
        points=[dict(v,roles="|".join(sorted(v["roles"]))) for _,v in sorted(points.items())]
        if len(roles)!=64 or len(plan89["units"])!=40:raise ValueError("Development coverage changed")
        code=[Path(__file__).resolve(),Path(__file__).with_name("run_stage103_verification.sh").resolve()]
        paths=[root102/"manifest.json",root102/"analysis/stage102_decision.json",root102/"analysis/stage102_cases.csv"]
        bindings=plan102["source_sha256"] | {str(x):c.file_hash(x) for x in paths}
        reuse=not differences
        plan=io.seal(dict(schema=SCHEMA,role="consumed_development_full_video_candidate_verification",design=DESIGN,
            units=plan89["units"],points=points,runtime=runtime,runtime_differences=differences,reuse_sources=reuse,
            trajectories=188,new_trajectories=80 if reuse else 188,
            new_search_calls=(340576 if reuse else 710199)+4*len(points),
            source_sha256=bindings,code_sha256=plan102["code_sha256"] | {str(x):c.file_hash(x) for x in code}))
        c.publish_identical(root/"manifest.json",plan)
        print(f"Stage-103 frozen: {plan['new_trajectories']} new full trajectories, {len(points)} matched states, {plan['new_search_calls']} searches",flush=True)
        frozen=dict(source["original"],fingerprint=plan["fingerprint"],runtime_identity=runtime,runtime=runtime["tracker"])
        native,_=s76.startup.checked_runtime(args,frozen,root);observer=e63.Capture(native)
        try:run(root,root100,plan100,root93,root89,plan93,plan89,report89,source,actor,observer,plan)
        finally:observer.handle.remove()


if __name__=="__main__":main()
