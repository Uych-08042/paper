"""Run matched short continuations after a single alternate-trigger scoring decision."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import stage124_candidate_independence as source
from stage27_uav_common import tracking_iou

prior, s121, base, c, io = source.prior, source.s121, source.base, source.c, source.io
SCHEMA = "stage125_alternate_trigger_continuation_v1"
STATUS = "COMPLETE_ALTERNATE_TRIGGER_CONTINUATION_DIAGNOSTIC"
SOURCE124 = "5087297fef1fbf5d2783caa71018f252f68f275bc75d43ea7378b6d0c28f8f9e"
DESIGN = dict(future_frames=31, good=.5, low=.2, recovery_run=3,
              bootstrap_seed=23125, bootstrap_samples=20000)
SEEDS = tuple(source.common.DESIGN["seeds"])
POLICIES = ("recorded", *source.ARMS)
FIELDS = ("frame_id", "image_hw", "candidate_boxes", "candidate_parents", "features", "crop",
          "history_latest", "scores", "selected", "boxes", "memory_write", "alternate_requested",
          "alternate_used", "unique_search_calls", "executed", "layer", "head_calls", "block_calls", "selector_calls")
TABLES = ("sampling", "starts", "branches", "decisions", "summary", "paired")
WORK = dict(training_updates=0, new_full_trajectories=0, confirmation_inference_calls=0)


def reference_jobs(plan):
    return {(j["spec"]["unit"]["sequence"], j["seed"]):j for j in plan["jobs"] if j["arm"] == s121.TREATMENT}


def select_points(decisions, jobs):
    groups = defaultdict(dict)
    for row in decisions:
        if row["alternate_activation"] != "True":
            continue
        key = (row["sequence"], int(row["seed"]), int(row["frame"]))
        seed = int(row["model_seed"])
        if seed in groups[key]:
            raise ValueError("Duplicate Stage-124 trigger prediction")
        groups[key][seed] = row
    points = []
    for (sequence, seed, frame), models in sorted(groups.items()):
        n = jobs[sequence, seed]["frames"]
        if set(models) != set(SEEDS) or not 1 <= frame <= n:
            raise ValueError("Incomplete Stage-124 trigger prediction")
        first = models[SEEDS[0]]
        second = models[SEEDS[1]]
        fields = ("candidate_ious", "recorded_action", "recorded_iou", "fold", "new_good_added")
        if any(first[k] != second[k] for k in fields):
            raise ValueError("Trigger bank differs between fitted seeds")
        points.append(dict(sequence=sequence, source_seed=seed, frame=frame, available_future=n-frame,
                           full_window=n-frame>=DESIGN["future_frames"], added_good=first["new_good_added"]=="True",
                           stage124_predictions={str(i): row for i, row in models.items()}))
    return points


def snapshot(data, i):
    return {k: data[k][i].copy() for k in FIELDS}


def observe(observer, image, history, previous, frame, ap, generator, synchronize):
    proposal = s121.forecast(history, previous, image.shape[:2], ap, generator)
    parents = proposal["candidate_parents"]
    row = base.s112.observe(observer, image, history, frame, ap, parents, synchronize)
    row.update(proposal, history_latest=history[-1].copy())
    row["executed"] = np.array([not any(np.array_equal(p, q) for q in parents[:i])
                                for i,p in enumerate(parents)], bool)
    return row


def commit(row, scores, action):
    updated = dict(row, scores=scores, selected=action, boxes=row["candidate_boxes"][action].copy())
    updated["memory_write"] = base.write_eligible(updated)
    return {k:updated[k] for k in FIELDS}


def choices(row, history, original, fitted):
    result = {("recorded", 0):base.choose(row, history, "generated_shared", original)}
    x = source.common.features(row["features"], row["candidate_boxes"], history)
    for arm in source.ARMS:
        for seed in SEEDS:
            values = source.common.scores(fitted[arm,seed], source.inputs(x, arm))
            action = source.common.physical_choice(values, row["candidate_boxes"])
            result[arm,seed] = (values, action)
    return result


def continuation_key(row):
    # The successor reads the chosen slot's features/write flag and the alternate request.
    # This request can differ even when two scorers publish the same box.
    return int(row["selected"]), s121.request_alternate(row)


def rollout(observer, frames, history, start, ap, generator, original, read_image, synchronize):
    state = [*history, start["boxes"]][-ap["design"]["history"]:]
    previous, rows = start, []
    observer.rt.tracker.state = start["boxes"].tolist()
    start_frame = int(start["frame_id"])
    for frame in range(start_frame+1, start_frame+DESIGN["future_frames"]+1):
        row = observe(observer, read_image(ap["root"], frames[frame]), state, previous, frame, ap, generator, synchronize)
        scores, action = base.choose(row, state, "generated_shared", original)
        previous = commit(row, scores, action)
        state = [*state, previous["boxes"]][-ap["design"]["history"]:]
        observer.rt.tracker.state = previous["boxes"].tolist()
        rows.append(previous)
    return {k:np.array([r[k] for r in rows]) for k in FIELDS}


def audit(data, frames, history, start, ap, generator, original):
    start_frame = int(start["frame_id"])
    horizon = DESIGN["future_frames"]
    if set(data) != set(FIELDS) or any(len(v) != horizon for v in data.values()):
        raise ValueError("Incomplete short trajectory")
    np.testing.assert_array_equal(data["frame_id"], np.arange(start_frame+1, start_frame+horizon+1))
    base.e63.source_common.prior.audit_calls(data, (horizon, 4))
    state = [*history, start["boxes"]][-ap["design"]["history"]:]
    previous = start
    for i, frame in enumerate(data["frame_id"]):
        row = snapshot(data, i)
        hw = (frames[frame]["height"], frames[frame]["width"])
        np.testing.assert_array_equal(row["image_hw"], hw)
        np.testing.assert_array_equal(row["history_latest"], state[-1])
        proposal = s121.forecast(state, previous, hw, ap, generator)
        for k in ("candidate_parents", "alternate_requested", "alternate_used"):
            np.testing.assert_array_equal(row[k], proposal[k])
        parents = row["candidate_parents"]
        expected = np.array([not any(np.array_equal(p, q) for q in parents[:j])
                             for j,p in enumerate(parents)], bool)
        np.testing.assert_array_equal(row["executed"], expected)
        np.testing.assert_equal(row["unique_search_calls"], expected.sum())
        for j,parent in enumerate(parents):
            np.testing.assert_array_equal(row["crop"][j], base.s105.crop_geometry(
                parent, hw, ap["search_factor"], ap["search_size"])[0])
        scores, action = base.choose(row, state, "generated_shared", original)
        np.testing.assert_allclose(row["scores"], scores, rtol=1e-9, atol=1e-12)
        np.testing.assert_equal(row["selected"], action)
        np.testing.assert_array_equal(row["boxes"], row["candidate_boxes"][action])
        np.testing.assert_equal(row["memory_write"], base.write_eligible(row))
        state = [*state, row["boxes"]][-ap["design"]["history"]:]
        previous = row


def verify_start_choices(recorded, point, actions):
    np.testing.assert_equal(actions["recorded",0][1], int(recorded["selected"]))
    np.testing.assert_allclose(actions["recorded",0][0], recorded["scores"], rtol=1e-9, atol=1e-12)
    for arm in source.ARMS:
        for seed in SEEDS:
            entry = point["stage124_predictions"][str(seed)]
            scores, action = actions[arm,seed]
            np.testing.assert_allclose(scores, list(map(float, entry[arm+"_scores"].split("|"))), rtol=1e-9, atol=1e-12)
            np.testing.assert_equal(action, int(entry[arm+"_action"]))


def outcome(start, future, gt):
    current = float(tracking_iou(start["boxes"], gt[0]))
    iou = tracking_iou(future["boxes"], gt[1:])
    if current < 0 or (iou < 0).any():
        raise ValueError("Invalid cached evaluation ground truth")
    good = iou>=DESIGN["good"]
    recovered = np.flatnonzero(np.lib.stride_tricks.sliding_window_view(good, DESIGN["recovery_run"]).all(axis=1))
    return dict(current_iou=current, future_mean_iou=float(iou.mean()),
                future_low_frames=int((iou<DESIGN["low"]).sum()),
                first_sustained_good_offset=int(recovered[0])+1 if recovered.size else None,
                final_iou=float(iou[-1]))


def report_point(points, futures, entry, ident, point, recorded):
    n = DESIGN["future_frames"]
    gt = np.asarray(entry["ground_truth"])[point["frame"]:point["frame"]+n+1]
    cached_actions = point["stage124_predictions"][str(SEEDS[0])]
    source_rescue = float(cached_actions["q0_iou"])<DESIGN["low"] and float(cached_actions["recorded_iou"])>=DESIGN["good"]
    actual = {k:outcome(row, futures[continuation_key(row)], gt) for k,row in points.items()}
    old = actual["recorded",0]
    decisions = []
    for (arm,seed), row in points.items():
        metrics = actual[arm,seed]
        stable = old["first_sustained_good_offset"] is not None
        decisions.append(dict(ident, arm=arm, model_seed=seed, action=int(row["selected"]),
                              next_alternate_requested=s121.request_alternate(row), **metrics,
                              added_good=point["added_good"], source_rescue=source_rescue,
                              actual_orig_good=old["current_iou"]>=DESIGN["good"],
                              lost_current_good=old["current_iou"]>=DESIGN["good"]>metrics["current_iou"],
                              lost_sustained_good=stable and metrics["first_sustained_good_offset"] is None))
    start = dict(ident, source_rescue=source_rescue, added_good=point["added_good"],
                 source_action=int(recorded["selected"]), rechecked_action=int(points["recorded",0]["selected"]),
                 max_candidate_box_difference=float(np.max(np.abs(
                     points["recorded",0]["candidate_boxes"]-recorded["candidate_boxes"]))))
    return start, decisions


def run(root, plan, r121, p121, r124, models, load_entry, old, observer, synchronize=lambda: None,
        image_reader=base.s105.checked_image):
    tables = {k:[] for k in TABLES}
    evidence = {}
    jobs = reference_jobs(p121)
    ap = base.e63.c.actor_plan(old)
    grouped = defaultdict(list)
    for point in plan["points"]:
        tables["sampling"].append({k:v for k,v in point.items() if k!="stage124_predictions"})
        if point["full_window"]:
            grouped[point["sequence"],point["source_seed"]].append(point)
    same_runtime = not plan["runtime_differences"]
    for key, points in grouped.items():
        job = jobs[key]
        sequence, source_seed = key
        entry = load_entry(job["spec"])
        trace = prior.load_trace(r121, p121, job)
        np.testing.assert_array_equal(trace["frame_id"], np.arange(1, job["frames"]+1))
        generator, original = base.job_models(dict(job, arm=prior.CONTROL), models)
        group = base.s109.s106.source_group(sequence)
        fitted = {(arm,seed):c.c66.sealed(r124/"models"/f"{group}__{arm}__s{seed}.json", source.common.SCHEMA)
                  for arm in source.ARMS for seed in SEEDS}
        for arm, seed in fitted:
            model = fitted[arm,seed]
            if model["split"]["held_out_groups"] != [group] or model["seed"] != seed or model["architecture"]!="shared":
                raise ValueError("Stage-124 held-out model identity changed")
        with base.e63.source_experiment.installed(observer, [], job["spec"]["unit"]) as head:
            if head != plan["native_head_sha256"]:
                raise ValueError("Native SGLA head identity changed")
            observer.initialize(image_reader(ap["root"], entry["frames"][0]), entry["initial_box"])
            for point in points:
                frame = point["frame"]
                i = frame-1
                if i < 1 or not bool(trace["alternate_used"][i]):
                    raise ValueError("Expected actual Stage-121 alternate activation")
                history = list(prior.history(trace, entry["initial_box"], i, ap["design"]["history"]))
                recorded = snapshot(trace, i)
                cached = choices(recorded, history, original, fitted)
                verify_start_choices(recorded, point, cached)
                ident = dict(sequence=sequence, source_seed=source_seed, frame=frame)
                stem = f"{sequence}__s{source_seed}__f{frame:06d}"
                meta = dict(plan_fingerprint=plan["fingerprint"], **ident)
                if same_runtime:
                    shared, actions = recorded, cached
                else:
                    def recollect():
                        image = image_reader(ap["root"], entry["frames"][frame])
                        row = observe(observer, image, history, snapshot(trace,i-1), frame, ap, generator, synchronize)
                        return commit(row, *base.choose(row, history, "generated_shared", original))
                    shared = base.s109.cache(root, root/"starts"/(stem+".npz"), meta, recollect, evidence)
                    actions = choices(shared, history, original, fitted)
                np.testing.assert_array_equal(shared["candidate_parents"], recorded["candidate_parents"])
                if not shared["alternate_used"]:
                    raise ValueError("Re-observation changed the trigger")
                starts = {k:commit(shared, *v) for k,v in actions.items()}
                ref_key = continuation_key(starts["recorded",0])
                futures = {}
                for start in starts.values():
                    branch = continuation_key(start)
                    if branch in futures:
                        continue
                    if same_runtime and branch==ref_key:
                        data = {k:trace[k][frame:frame+DESIGN["future_frames"]] for k in FIELDS}
                        cached_future = True
                    else:
                        path = root/"branches"/f"{stem}__q{branch[0]}__a{int(branch[1])}.npz"
                        data = base.s109.cache(root, path, dict(meta, action=branch[0], next_request=branch[1]),
                                              lambda:rollout(observer, entry["frames"], history, start, ap,
                                                             generator, original, image_reader, synchronize), evidence)
                        cached_future = False
                    audit(data, entry["frames"], history, start, ap, generator, original)
                    futures[branch] = data
                    count = int(data["unique_search_calls"].sum())
                    tables["branches"].append(dict(ident, action=branch[0], next_request=branch[1],
                                                   cached_future=cached_future, future_search_calls=count,
                                                   newly_computed_search_calls=0 if cached_future else count))
                start, rows = report_point(starts, futures, entry, ident, point, recorded)
                start["shared_start_search_calls"] = 0 if same_runtime else int(shared["unique_search_calls"])
                tables["starts"].append(start)
                tables["decisions"].extend(rows)
                print(f"[Stage-125] {stem}: {len(futures)} distinct continuations", flush=True)
    publish(root, plan, tables, evidence)


def summarize(rows):
    by_point = defaultdict(dict)
    for r in rows:
        by_point[r["sequence"],r["source_seed"],r["frame"]][r["arm"],r["model_seed"]] = r
    summaries, comparisons = [], []
    for name in ("all", "added_good", "source_rescue", "actual_orig_good"):
        local = [p for p in by_point.values() if name=="all" or p["recorded",0][name]]
        for arm in POLICIES:
            selected = [r for p in local for (a,_),r in p.items() if a==arm]
            summaries.append(dict(population=name, arm=arm, points=len(local), decisions=len(selected),
                                  mean_future_iou=float(np.mean([r["future_mean_iou"] for r in selected])) if selected else None,
                                  mean_low_frames=float(np.mean([r["future_low_frames"] for r in selected])) if selected else None,
                                  lost_current_good=sum(r["lost_current_good"] for r in selected),
                                  lost_sustained_good=sum(r["lost_sustained_good"] for r in selected)))
        if not local:
            continue
        contrasts = (("independent", "recorded"), ("independent", "relation"), ("relation", "recorded"))
        for arm, ref in contrasts:
            for seed in (0,*SEEDS) if name=="all" else (0,):
                for metric in ("future_mean_iou","future_low_frames"):
                    unit = defaultdict(list)
                    for p in local:
                        used = SEEDS if seed==0 else (seed,)
                        diff = np.mean([p[arm,s][metric] - p[ref,0 if ref=="recorded" else s][metric] for s in used])
                        r = p["recorded",0]
                        unit[r["sequence"],r["source_seed"]].append(float(diff))
                    videos = defaultdict(list)
                    for (seq,_),values in sorted(unit.items()):
                        videos[seq].append(float(np.mean(values)))
                    values = [float(np.mean(v)) for _,v in sorted(videos.items())]
                    # For signed deltas, use the existing loss-metric convention only for low-frame counts.
                    stats = base.s103.contrast(values, arm+"_minus_"+ref,
                                              "low_iou_fraction" if metric=="future_low_frames" else metric, DESIGN)
                    stats["metric"] = metric
                    comparisons.append(dict(population=name, model_seed="all" if seed==0 else seed, **stats))
    return summaries, comparisons


def publish(root, plan, tables, evidence):
    n = sum(p["full_window"] for p in plan["points"])
    if len(tables["sampling"])!=len(plan["points"]) or len(tables["starts"])!=n or len(tables["decisions"])!=n*5:
        raise ValueError("Incomplete triggered branch evaluation")
    tables["summary"], tables["paired"] = summarize(tables["decisions"])
    actual = sum(r["newly_computed_search_calls"] for r in tables["branches"])
    actual += sum(r["shared_start_search_calls"] for r in tables["starts"])
    same_runtime = not plan["runtime_differences"]
    ceiling = n*4*((4 if same_runtime else 5)*DESIGN["future_frames"] + (not same_runtime))
    if not 0 <= actual <= ceiling:
        raise ValueError("Short-branch search budget exceeded")
    new_branches = sum(not r["cached_future"] for r in tables["branches"])
    if len(evidence) != 2*(new_branches + (0 if same_runtime else n)):
        raise ValueError("Incomplete re-observation or continuation receipts")
    out = root/"analysis"
    out.mkdir(exist_ok=True)
    hashes = {}
    for key in TABLES:
        path = out/f"stage125_{key}.csv"
        base.s103.s86.publish_table(path, tables[key])
        hashes[path.name] = c.file_hash(path)
    primary = [r for r in tables["paired"] if r["population"]=="all" and r["model_seed"]=="all"
               and r["metric"]=="future_mean_iou" and r["contrast"].startswith("independent_minus_")]
    report = io.seal(dict(schema=SCHEMA, status=STATUS, source_stage124_fingerprint=SOURCE124,
                         plan_fingerprint=plan["fingerprint"], runtime=plan["runtime"], windows=n,
                         censored=len(plan["points"])-n, new_search_calls=actual, maximum_new_search_calls=ceiling,
                         new_branches=new_branches, primary=primary,
                         row_counts={k:len(v) for k,v in tables.items()},
                         artifact_sha256=evidence, table_sha256=hashes, **WORK))
    c.publish_identical(out/"stage125_decision.json", report)
    check_results(root)
    print(f"Stage-125: {STATUS}. {n} full windows; {actual} new SGLA searches.", flush=True)


def check_results(root):
    plan = c.c66.sealed(root/"manifest.json", SCHEMA)
    report = c.c66.sealed(root/"analysis/stage125_decision.json", SCHEMA)
    n = sum(p["full_window"] for p in plan["points"])
    if (plan["design"]!=DESIGN or report["status"]!=STATUS or report["source_stage124_fingerprint"]!=SOURCE124
            or plan["fingerprint"]!=report["plan_fingerprint"] or plan["runtime"]!=report["runtime"]
            or report["windows"]!=n or report["censored"]!=len(plan["points"])-n
            or report["row_counts"]["decisions"]!=n*5 or any(report[k]!=v for k,v in WORK.items())
            or set(report["table_sha256"])!={f"stage125_{key}.csv" for key in TABLES}):
        raise ValueError("Incomplete Stage-125 result")
    io.verify(plan["inputs"])
    io.verify({str(root/"analysis"/k):v for k,v in report["table_sha256"].items()})
    io.verify({str(root/k):v for k,v in report["artifact_sha256"].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76","stage89","stage93","stage112","stage121","stage124")
    for name in names:
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--check-results",action="store_true")
    parser.set_defaults(mode="stage125", parameter="deit_distilled")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not args.checkpoint or not all(getattr(args,n) for n in names):
        parser.error("Supply Stage-76/89/93/112/121/124 directories and original checkpoint")
    r76,r89,r93,r112,r121,r124 = sources = [getattr(args,n).expanduser().resolve() for n in names]
    if any(root==p or root in p.parents or p in root.parents for p in sources):
        parser.error("Choose a separate Stage-125 directory")
    if not (r121/"manifest.json").is_file() or not (r124/"manifest.json").is_file():
        parser.error("Need sealed Stage-121 and Stage-124 server runs")
    with base.s109.s106.s93.writer_lock(root):
        if (root/"analysis/stage125_decision.json").exists():
            check_results(root)
            return
        source.check_results(r124)
        p124 = c.c66.sealed(r124/"manifest.json", source.SCHEMA)
        d124 = c.c66.sealed(r124/"analysis/stage124_decision.json", source.SCHEMA)
        p121 = c.c66.sealed(r121/"manifest.json", s121.SCHEMA)
        p112 = c.c66.sealed(r112/"manifest.json", source.common.SCHEMA)
        if d124["fingerprint"]!=SOURCE124 or p121["reuse_controls"]:
            raise ValueError("Need the archived Stage-124 held-out scoring run and Stage-121 L40 source")
        if any(p124["inputs"][str(p)] != c.file_hash(p) for p in (r112/"manifest.json",r121/"manifest.json")):
            raise ValueError("Source manifest does not match Stage-124")
        models, _ = base.load_frozen_models(r112,p112)
        decisions = source.read_table(r124/"analysis/stage124_decisions.csv")
        points = select_points(decisions, reference_jobs(p121))
        if (len(points),sum(p["full_window"] for p in points),
                sum(p["full_window"] and p["added_good"] for p in points)) != (358,346,9):
            raise ValueError("Frozen trigger population differs from the archived result")
        p93,_,p89,report89,origin = base.s103.prior.s95.s94.load_run(r93,r76,r89)

        def load_entry(spec):
            return base.s103.prior.s95.load_unit(r93,r89,p93,p89,report89,origin,spec)[0]

        import torch
        torch.set_num_threads(1)
        old = origin["original"]
        runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
        manifest = root/"manifest.json"
        if manifest.is_file() and c.c66.sealed(manifest,SCHEMA)["runtime"]!=runtime:
            raise ValueError("GPU/runtime changed on resume. Set STAGE125_ROOT to a new directory.")
        differences = base.s109.check_runtime(p121["runtime"],runtime)
        bindings = dict(p124["inputs"])
        bound = (r124/"manifest.json",r124/"analysis/stage124_decision.json",Path(__file__).resolve(),
                 Path(__file__).with_name("run_stage125_trigger_continuation.sh").resolve())
        bindings.update({str(p):c.file_hash(p) for p in bound})
        bindings.update({str(r124/"analysis"/k):v for k,v in d124["table_sha256"].items()})
        bindings.update({str(r124/k):v for k,v in d124["model_sha256"].items()})
        plan = io.seal(dict(schema=SCHEMA, design=DESIGN, runtime=runtime, runtime_differences=differences,
                            native_head_sha256=p121["native_head_sha256"], points=points,
                            source_stage124_fingerprint=SOURCE124, inputs=bindings))
        c.publish_identical(manifest,plan)
        frozen = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native,_ = base.s103.s76.startup.checked_runtime(args, frozen, root)
        observer = base.e63.Capture(native)
        try:
            run(root,plan,r121,p121,r124,models,load_entry,old,observer,lambda:torch.cuda.synchronize(args.gpu))
        finally:
            observer.handle.remove()


if __name__=="__main__":
    main()
