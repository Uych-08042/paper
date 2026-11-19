"""Cross the committed box and the next-frame alternate search at Stage-125 triggers."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import stage125_trigger_continuation as prior
from stage27_uav_common import tracking_iou

base, c, io, s121 = prior.base, prior.c, prior.io, prior.s121
SCHEMA = "stage126_factorial_continuation_v1"
STATUS = "COMPLETE_COMMIT_BY_ALTERNATE_DIAGNOSTIC"
SOURCE125 = "5f569cdeaede3ffc4e5440cc25362b6948f3fffcd4bb1737262abbe2121b83d9"
DESIGN = dict(prior.DESIGN, bootstrap_seed=23126)
TABLES = ("cells", "effects", "paired", "coverage")
WORK = dict(training_updates=0, new_full_trajectories=0, confirmation_inference_calls=0)
METRICS = ("future_mean_iou", "future_low_frames")
EFFECTS = ("commit_at_disabled", "allow_on_original", "allow_on_retrained", "interaction",
           "natural_request_contribution", "natural_total")


def forced_start(row, action, allow):
    # After committing, only Stage-121's next-request rule reads these scores.
    # Equal scores let the non-q0 and displacement checks decide eligibility.
    scores = np.zeros(4)
    if not allow:
        scores[action] = 1
    return prior.commit(row, scores, action)


def variants(row, actions, history, hw, ap, generator):
    cells = {}
    for action in sorted(set(actions)):
        for allow in (False, True):
            start = forced_start(row, action, allow)
            state = [*history, start["boxes"]][-ap["design"]["history"]:]
            forecast = s121.forecast(state, start, hw, ap, generator)
            cells[action, allow] = (start, forecast)
        off, on = (cells[action, allow][1] for allow in (False, True))
        if off["alternate_requested"] or off["alternate_used"]:
            raise ValueError("Disabled branch requested a replacement")
        if not on["alternate_used"]:
            np.testing.assert_array_equal(off["candidate_parents"], on["candidate_parents"])
    return cells


def index_source(root, report):
    indexed = {}
    for table in ("decisions", "branches"):
        rows = prior.source.read_table(root / f"analysis/stage125_{table}.csv")
        by_start = defaultdict(dict)
        for row in rows:
            ident = (row["sequence"], int(row["source_seed"]), int(row["frame"]))
            if table == "decisions":
                key = (row["arm"], int(row["model_seed"]))
            else:
                key = (int(row["action"]), row["next_request"] == "True")
            if key in by_start[ident]:
                raise ValueError("Duplicate Stage-125 branch or decision")
            by_start[ident][key] = row
        if len(rows) != report["row_counts"][table]:
            raise ValueError("Incomplete Stage-125 table")
        indexed[table] = by_start
    if set(indexed["decisions"]) != set(indexed["branches"]):
        raise ValueError("Stage-125 decisions and continuations do not match")
    expected = {(arm,seed) for arm in prior.source.ARMS for seed in prior.SEEDS} | {("recorded",0)}
    for ident, decisions in indexed["decisions"].items():
        natural = {(int(r["action"]), r["next_alternate_requested"]=="True") for r in decisions.values()}
        if set(decisions)!=expected or natural!=set(indexed["branches"][ident]):
            raise ValueError("Stage-125 natural branches are incomplete")
    return indexed


def search_ceiling(source_index, changed_runtime):
    if changed_runtime:
        # Four actions, two settings for each, with q0 structurally unable to request.
        return len(source_index["decisions"])*(7*4*DESIGN["future_frames"]+4)
    missing = 0
    for ident, decisions in source_index["decisions"].items():
        actions = {int(r["action"]) for r in decisions.values()}
        prospective = {(action,request) for action in actions for request in (False,True) if action or not request}
        missing += len(prospective - set(source_index["branches"][ident]))
    return missing*4*DESIGN["future_frames"]


def inherit(stem, meta, source_path, source_plan, source_rows, trace, frame, action, request, used):
    # When the crop guard blocks the replacement, both flag values lead to the
    # same next candidate parents. The reported request remains arm-specific.
    options = (request,) if used else (request, not request)
    for flag in options:
        if (action, flag) not in source_rows:
            continue
        row = source_rows[action, flag]
        if row["cached_future"] == "True":
            future = {k:trace[k][frame:frame + DESIGN["future_frames"]] for k in prior.FIELDS}
        else:
            path = source_path / "branches" / f"{stem}__q{action}__a{int(flag)}.npz"
            future = c.read_cache(path, dict(meta, plan_fingerprint=source_plan["fingerprint"],
                                             action=action, next_request=flag))
            if future is None:
                raise ValueError(f"Missing Stage-125 continuation {path}")
        if bool(future["alternate_used"][0]) != used:
            continue
        return future, "stage125_original" if row["cached_future"] == "True" else "stage125_branch"
    return None, None


def with_request(data, request):
    if bool(data["alternate_requested"][0]) == request:
        return data
    flags = data["alternate_requested"].copy()
    flags[0] = request
    return dict(data, alternate_requested=flags)


def signed_effects(off0, allow0, off1, allow1, natural0, natural1):
    commit = off1 - off0
    allow_original = allow0 - off0
    allow_retrained = allow1 - off1
    return dict(zip(EFFECTS, (commit, allow_original, allow_retrained,
                              allow_retrained - allow_original, (natural1 - off1) - (natural0 - off0),
                              natural1 - natural0)))


def effect_rows(ident, starts, cells):
    original, original_request = starts["recorded", 0]
    for (arm, seed), (action, request) in starts.items():
        if arm == "recorded":
            continue
        off0, allow0, off1, allow1 = (cells[k] for k in ((original,False), (original,True),
                                                        (action,False), (action,True)))
        natural0, natural1 = cells[original,original_request], cells[action,request]
        for metric in METRICS:
            contrasts = signed_effects(*(cell[metric] for cell in (off0,allow0,off1,allow1,natural0,natural1)))
            yield dict(ident, arm=arm, model_seed=seed, metric=metric, original_action=original, action=action,
                       original_request=original_request, natural_request=request,
                       original_can_replace=allow0["used"], retrained_can_replace=allow1["used"],
                       **contrasts)


def roll_point(root, source_path, source_plan, source_rows, point, entry, trace, history, shared, actions,
               observer, ap, generator, scorer, image_reader, synchronize, evidence, plan, reobserved_calls):
    ident = {k:point[k] for k in ("sequence", "source_seed", "frame")}
    frame = point["frame"]
    stem = f"{ident['sequence']}__s{ident['source_seed']}__f{frame:06d}"
    meta = dict(ident, plan_fingerprint=plan["fingerprint"])
    key_by_policy = {k:prior.continuation_key(prior.commit(shared,*v)) for k,v in actions.items()}
    next_frame = entry["frames"][frame+1]
    hw = next_frame["height"], next_frame["width"]
    forks = variants(shared, [k[0] for k in key_by_policy.values()], history, hw, ap, generator)
    futures, cells = {}, {}
    gt = np.asarray(entry["ground_truth"])[frame:frame+DESIGN["future_frames"]+1]
    good_candidates = int((tracking_iou(shared["candidate_boxes"], np.broadcast_to(gt[0], (4, 4)))>=DESIGN["good"]).sum())
    for (action, allow), (start, forecast) in forks.items():
        request, used = (bool(forecast["alternate_"+k]) for k in ("requested", "used"))
        physical = action, used
        calls = 0
        if physical in futures:
            future, provenance = futures[physical], "same_parent_branch"
        else:
            future, provenance = (None, None) if plan["runtime_differences"] else inherit(
                stem, meta, source_path, source_plan, source_rows["branches"], trace, frame, action, request, used)
            if future is None:
                path = root/"branches"/f"{stem}__q{action}__u{int(used)}.npz"
                future = base.s109.cache(root,path,dict(meta, action=action, first_replacement=used),
                    lambda:prior.rollout(observer,entry["frames"],history,start,ap,generator,scorer,
                                         image_reader,synchronize), evidence)
                calls = int(future["unique_search_calls"].sum())
                provenance = "stage126"
            futures[physical] = future
        future = with_request(future, request)
        prior.audit(future,entry["frames"],history,start,ap,generator,scorer)
        np.testing.assert_equal(future["alternate_used"][0], used)
        np.testing.assert_array_equal(future["candidate_parents"][0], forecast["candidate_parents"])
        next_ious = tracking_iou(future["candidate_boxes"][0], np.broadcast_to(gt[1], (4, 4)))
        cells[action,allow] = dict(ident, action=action, allow=allow, requested=request, used=used,
            provenance=provenance, branch_search_calls=int(future["unique_search_calls"].sum()),
            new_branch_search_calls=calls, current_good_candidates=good_candidates,
            next_best_iou=float(next_ious.max()), next_chosen_iou=float(tracking_iou(future["boxes"][0],gt[1])),
            **prior.outcome(start,future,gt))
    for policy, (action, request) in key_by_policy.items():
        cell = cells[action, request]
        if request != cell["requested"]:
            raise ValueError("Natural policy lacks its requested factorial cell")
        if not plan["runtime_differences"]:
            old = source_rows["decisions"][policy]
            if action != int(old["action"]) or request != (old["next_alternate_requested"] == "True"):
                raise ValueError("Stage-125 action or request changed")
            for k in ("current_iou", "final_iou", *METRICS):
                np.testing.assert_allclose(cell[k], float(old[k]), rtol=1e-9, atol=1e-12)
            offset = int(old["first_sustained_good_offset"]) if old["first_sustained_good_offset"] else None
            np.testing.assert_equal(cell["first_sustained_good_offset"], offset)
    return list(cells.values()), list(effect_rows(ident,key_by_policy,cells)), dict(ident,
        actual_actions=len({k[0] for k in key_by_policy.values()}), cells=len(cells), physical_branches=len(futures),
        feasible_actions=sum(row["used"] for row in cells.values()), current_good_candidates=good_candidates,
        source_action=int(source_rows["decisions"]["recorded",0]["action"]), observed_action=key_by_policy["recorded",0][0],
        max_candidate_box_difference=float(np.max(np.abs(shared["candidate_boxes"] -
                                                          trace["candidate_boxes"][frame-1]))),
        new_start_search_calls=reobserved_calls)


def summarize(rows):
    result = []
    subsets = {"all":lambda r:True, "changed_action":lambda r:r["action"]!=r["original_action"],
               "both_replace":lambda r:r["original_can_replace"] and r["retrained_can_replace"]
                                       and r["action"]!=r["original_action"]}
    for name, include in subsets.items():
        for arm in prior.source.ARMS:
            for model_seed in ("all", *prior.SEEDS):
                for metric in METRICS:
                    selected = [r for r in rows if include(r) and r["arm"]==arm and r["metric"]==metric
                                and (model_seed=="all" or r["model_seed"]==model_seed)]
                    if not selected:
                        continue
                    for effect in EFFECTS:
                        points = defaultdict(list)
                        for row in selected:
                            points[row["sequence"],row["source_seed"],row["frame"]].append(row[effect])
                        sources, videos = defaultdict(list), defaultdict(list)
                        for (seq, seed, _), values in sorted(points.items()):
                            sources[seq,seed].append(float(np.mean(values)))
                        for (seq,_), values in sorted(sources.items()):
                            videos[seq].append(float(np.mean(values)))
                        values = [float(np.mean(v)) for _,v in sorted(videos.items())]
                        stats = base.s103.contrast(values,effect,
                            "low_iou_fraction" if metric=="future_low_frames" else metric,DESIGN)
                        result.append(dict(population=name, arm=arm, model_seed=model_seed, points=len(points),
                                           point_model_pairs=len(selected), **dict(stats,metric=metric)))
    return result


def run(root, source_path, source_plan, source_index, plan, r121, p121, r124, models, load_entry, old,
        observer, synchronize=lambda:None, image_reader=base.s105.checked_image):
    evidence = {}
    tables = {name:[] for name in TABLES}
    ap = base.e63.c.actor_plan(old)
    jobs = prior.reference_jobs(p121)
    groups = defaultdict(list)
    for point in source_plan["points"]:
        if point["full_window"]:
            groups[point["sequence"],point["source_seed"]].append(point)
    for (sequence, seed), points in sorted(groups.items()):
        job = jobs[sequence,seed]
        entry, trace = load_entry(job["spec"]), prior.prior.load_trace(r121,p121,job)
        generator, scorer = base.job_models(dict(job,arm=prior.prior.CONTROL),models)
        group = base.s109.s106.source_group(sequence)
        fitted = {(arm,s):c.c66.sealed(r124/"models"/f"{group}__{arm}__s{s}.json",prior.source.common.SCHEMA)
                  for arm in prior.source.ARMS for s in prior.SEEDS}
        for (_,s), model in fitted.items():
            if model["split"]["held_out_groups"] != [group] or model["seed"] != s or model["architecture"] != "shared":
                raise ValueError("Stage-124 held-out model changed")
        with base.e63.source_experiment.installed(observer,[],job["spec"]["unit"]) as head:
            if head != source_plan["native_head_sha256"]:
                raise ValueError("Native SGLA head changed")
            observer.initialize(image_reader(ap["root"],entry["frames"][0]),entry["initial_box"])
            for point in points:
                frame = point["frame"]
                i = frame-1
                if i < 1 or not trace["alternate_used"][i] or int(trace["frame_id"][i])!=frame:
                    raise ValueError("Stage-121 trigger changed")
                history = list(prior.prior.history(trace,entry["initial_box"],i,ap["design"]["history"]))
                shared = prior.snapshot(trace,i)
                calls = 0
                if plan["runtime_differences"]:
                    meta = dict(plan_fingerprint=plan["fingerprint"],sequence=sequence,source_seed=seed,frame=frame)
                    def recollect():
                        row = prior.observe(observer,image_reader(ap["root"],entry["frames"][frame]),history,
                                            prior.snapshot(trace,i-1),frame,ap,generator,synchronize)
                        return prior.commit(row,*base.choose(row,history,"generated_shared",scorer))
                    shared = base.s109.cache(root,root/"starts"/f"{sequence}__s{seed}__f{frame:06d}.npz",
                                             meta,recollect,evidence)
                    np.testing.assert_array_equal(shared["candidate_parents"],trace["candidate_parents"][i])
                    if not shared["alternate_used"]:
                        raise ValueError("Re-observed start lost its original trigger")
                    calls = int(shared["unique_search_calls"])
                actions = prior.choices(shared,history,scorer,fitted)
                key = sequence,seed,frame
                source_rows = {name:table[key] for name,table in source_index.items()}
                cells, effects, coverage = roll_point(root,source_path,source_plan,source_rows,point,entry,trace,
                    history,shared,actions,observer,ap,generator,scorer,image_reader,synchronize,evidence,plan,calls)
                tables["cells"].extend(cells)
                tables["effects"].extend(effects)
                tables["coverage"].append(coverage)
                print(f"[Stage-126] {sequence} s{seed} f{frame}: {coverage['physical_branches']} continuations",flush=True)
    if len(tables["coverage"])!=plan["windows"] or len(tables["effects"])!=plan["windows"]*4*len(METRICS):
        raise ValueError("Incomplete factorial continuations")
    tables["paired"] = summarize(tables["effects"])
    calls = sum(r["new_branch_search_calls"] for r in tables["cells"]) + sum(r["new_start_search_calls"] for r in tables["coverage"])
    n_new = sum(r["provenance"]=="stage126" for r in tables["cells"])
    if len(evidence) != 2*(n_new + (plan["windows"] if plan["runtime_differences"] else 0)):
        raise ValueError("Incomplete cache receipts")
    ceiling = plan["maximum_new_search_calls"]
    if calls>ceiling or any(r["branch_search_calls"]>4*DESIGN["future_frames"] for r in tables["cells"]):
        raise ValueError("Search budget exceeded")
    out = root/"analysis"
    out.mkdir(exist_ok=True)
    hashes = {}
    for name, rows in tables.items():
        path = out/f"stage126_{name}.csv"
        base.s103.s86.publish_table(path,rows)
        hashes[path.name] = c.file_hash(path)
    primary = [r for r in tables["paired"] if r["population"]=="all" and r["metric"]=="future_mean_iou"
               and r["model_seed"]=="all"]
    c.publish_identical(out/"stage126_decision.json",io.seal(dict(schema=SCHEMA,status=STATUS,
        plan_fingerprint=plan["fingerprint"],source_stage125_fingerprint=SOURCE125,runtime=plan["runtime"],
        runtime_differences=plan["runtime_differences"],windows=plan["windows"],primary=primary,
        new_search_calls=calls,maximum_new_search_calls=ceiling,new_branches=n_new,
        row_counts={k:len(v) for k,v in tables.items()},artifact_sha256=evidence,table_sha256=hashes,**WORK)))
    check_results(root)
    print(f"Stage-126: {STATUS}; {plan['windows']} windows, {calls} new SGLA searches.",flush=True)


def check_results(root):
    plan = c.c66.sealed(root/"manifest.json",SCHEMA)
    report = c.c66.sealed(root/"analysis/stage126_decision.json",SCHEMA)
    if (plan["design"]!=DESIGN or report["status"]!=STATUS or report["source_stage125_fingerprint"]!=SOURCE125
            or report["plan_fingerprint"]!=plan["fingerprint"] or report["runtime"]!=plan["runtime"]
            or report["runtime_differences"]!=plan["runtime_differences"] or report["windows"]!=plan["windows"]
            or report["maximum_new_search_calls"]!=plan["maximum_new_search_calls"]
            or not 0<=report["new_search_calls"]<=plan["maximum_new_search_calls"]
            or report["row_counts"]["effects"]!=plan["windows"]*4*len(METRICS)
            or report["row_counts"]["coverage"]!=plan["windows"] or any(report[k]!=v for k,v in WORK.items())
            or set(report["table_sha256"])!={f"stage126_{name}.csv" for name in TABLES}):
        raise ValueError("Incomplete Stage-126 result")
    io.verify(plan["inputs"])
    io.verify({str(root/"analysis"/k):v for k,v in report["table_sha256"].items()})
    io.verify({str(root/k):v for k,v in report["artifact_sha256"].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76","stage89","stage93","stage112","stage121","stage124","stage125")
    for name in names:
        parser.add_argument("--"+name,type=Path)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--check-results",action="store_true")
    parser.set_defaults(mode="stage126",parameter="deit_distilled")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not args.checkpoint or not all(getattr(args,n) for n in names):
        parser.error("Supply Stage-76/89/93/112/121/124/125 and the original checkpoint")
    r76,r89,r93,r112,r121,r124,r125 = paths = [getattr(args,n).expanduser().resolve() for n in names]
    if any(root==p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate Stage-126 output directory")
    if not (r125/"analysis/stage125_decision.json").is_file():
        parser.error("Need the completed Stage-125 server run with its branch caches")
    with base.s109.s106.s93.writer_lock(root):
        if (root/"analysis/stage126_decision.json").is_file():
            check_results(root)
            return
        prior.check_results(r125)
        p125 = c.c66.sealed(r125/"manifest.json",prior.SCHEMA)
        d125 = c.c66.sealed(r125/"analysis/stage125_decision.json",prior.SCHEMA)
        if d125["fingerprint"]!=SOURCE125 or p125["runtime_differences"] or d125["windows"]!=346:
            raise ValueError("Need the archived Stage-125 same-runtime run")
        for p in (r112/"manifest.json", r121/"manifest.json", r124/"manifest.json"):
            if p125["inputs"].get(str(p))!=c.file_hash(p):
                raise ValueError(f"Source differs from Stage-125: {p}")
        p112 = c.c66.sealed(r112/"manifest.json",prior.source.common.SCHEMA)
        p121 = c.c66.sealed(r121/"manifest.json",s121.SCHEMA)
        models,_ = base.load_frozen_models(r112,p112)
        source_index = index_source(r125,d125)
        if len(source_index["decisions"])!=d125["windows"]:
            raise ValueError("Stage-125 complete windows differ from the sealed report")
        p93,_,p89,report89,origin = base.s103.prior.s95.s94.load_run(r93,r76,r89)
        def load_entry(spec):
            return base.s103.prior.s95.load_unit(r93,r89,p93,p89,report89,origin,spec)[0]

        import torch
        torch.set_num_threads(1)
        old = origin["original"]
        runtime = base.s103.s76.device_runtime(old,torch,args.gpu)
        differences = base.s109.check_runtime(p125["runtime"],runtime)
        manifest = root/"manifest.json"
        if manifest.is_file() and c.c66.sealed(manifest,SCHEMA)["runtime"]!=runtime:
            raise ValueError("GPU/runtime changed on resume. Set STAGE126_ROOT to a new directory.")
        inputs = dict(p125["inputs"])
        for path in (r125/"manifest.json",r125/"analysis/stage125_decision.json",Path(__file__).resolve(),
                     Path(__file__).with_name("run_stage126_factorial_continuation.sh").resolve()):
            inputs[str(path)] = c.file_hash(path)
        inputs.update({str(r125/"analysis"/k):v for k,v in d125["table_sha256"].items()})
        inputs.update({str(r125/k):v for k,v in d125["artifact_sha256"].items()})
        plan = io.seal(dict(schema=SCHEMA,design=DESIGN,source_stage125_fingerprint=SOURCE125,windows=d125["windows"],
                            maximum_new_search_calls=search_ceiling(source_index,bool(differences)),
                            runtime=runtime,runtime_differences=differences,inputs=inputs))
        c.publish_identical(manifest,plan)
        frozen = dict(old,fingerprint=plan["fingerprint"],runtime_identity=runtime,runtime=runtime["tracker"])
        native,_ = base.s103.s76.startup.checked_runtime(args,frozen,root)
        observer = base.e63.Capture(native)
        try:
            run(root,r125,p125,source_index,plan,r121,p121,r124,models,load_entry,old,observer,
                lambda:torch.cuda.synchronize(args.gpu))
        finally:
            observer.handle.remove()


if __name__=="__main__":
    main()
