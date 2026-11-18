"""Evaluate one-frame alternate search state on the frozen Stage-113 cohort."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import time

import numpy as np

import stage113_frozen_closed_loop as base
from stage114_cached_failure_chain import SOURCE113
from stage27_uav_common import failure_events, tracking_iou

c, io = base.c, base.io
SCHEMA = "stage121_alternate_search_state_v1"
TREATMENT = "alternate_shared"
REFERENCES = ("generated_shared", "fixed_shared", "native")
DESIGN = dict(search_budget=4, slot=3, maximum_score_gap=.05, minimum_center_distance=1.,
              bootstrap_seed=23121, bootstrap_samples=20000, practical_auc=.5,
              reliable_auc_tolerance=1.)
FIELDS = base.FIELDS + ("alternate_requested", "alternate_used")
METRICS = ("auc", "low_iou_fraction", "longest_failure_fraction", "reliable_auc",
           "good_candidate_fraction", "failure_events", "recovered_events")
TABLES = ("policy_summary", "paired", "seed_video", "scope", "recovery")


def request_alternate(previous):
    if previous is None or int(previous["selected"]) == 0:
        return False
    chosen = int(previous["selected"])
    q0, winner = previous["candidate_boxes"][[0, chosen]]
    gap = float(previous["scores"][chosen] - previous["scores"][0])
    center_delta = (winner[:2] + winner[2:] / 2) - (q0[:2] + q0[2:] / 2)
    unit = np.sqrt(max(q0[2:].prod(), winner[2:].prod()))
    return bool(gap <= DESIGN["maximum_score_gap"]
                and np.linalg.norm(center_delta) >= DESIGN["minimum_center_distance"] * unit)


def forecast(history, previous, hw, ap, generator):
    proposal = base.forecast_parents(history, previous, hw, ap, "generated_shared", generator)
    requested = request_alternate(previous)
    used = False
    if requested:
        parents = proposal["candidate_parents"]
        slot = DESIGN["slot"]
        alternate = previous["candidate_boxes"][0]

        def crop(parent):
            rect, _, _ = base.s105.crop_geometry(parent, hw, ap["search_factor"], ap["search_size"])
            return tuple(rect)

        original_crops = [crop(parent) for parent in parents]
        # Replacing a unique crop with a different unique crop preserves this state's physical budget.
        if original_crops.count(original_crops[slot]) == 1 and crop(alternate) not in original_crops:
            parents[slot] = alternate
            used = True
            fixed = base.e63.source_common.prior.history_parents(history, ap["design"])
            proposal["changed_crop"] = any(crop(a) != crop(b) for a, b in zip(parents, fixed))
    return dict(proposal, alternate_requested=requested, alternate_used=used)


def collect(observer, frames, initial_box, old, generator, scorer, synchronize=lambda: None,
            image_reader=base.s105.checked_image):
    ap = base.e63.c.actor_plan(old)
    history, previous, rows = [np.asarray(initial_box, float).copy()], None, []
    observer.initialize(image_reader(ap["root"], frames[0]), history[0])
    for frame, item in enumerate(frames[1:], 1):
        image = image_reader(ap["root"], item)
        synchronize()
        started = time.perf_counter()
        proposal = forecast(history, previous, image.shape[:2], ap, generator)
        parents = proposal["candidate_parents"]
        row = base.s112.observe(observer, image, history, frame, ap, parents, synchronize)
        scores, selected = base.choose(row, history, "generated_shared", scorer)
        executed = np.array([tuple(parent) not in {tuple(p) for p in parents[:i]}
                             for i, parent in enumerate(parents)], bool)
        row.update(proposal, history_latest=history[-1].copy(), scores=scores, selected=selected,
                   boxes=row["candidate_boxes"][selected].copy(), parents=parents[selected].copy(),
                   executed=executed)
        row["memory_write"] = base.write_eligible(row)
        history = (history + [row["boxes"].copy()])[-ap["design"]["history"]:]
        observer.rt.tracker.state = row["boxes"].tolist()
        synchronize()
        row["pipeline_ms"] = 1000 * (time.perf_counter() - started)
        rows.append(row)
        previous = row
        if frame % 500 == 0:
            print(f"  [Stage-121] {frame}/{len(frames)-1}", flush=True)
    return {key: np.asarray([row[key] for row in rows]) for key in FIELDS}


def audit(data, entry, old, generator, scorer):
    ap = base.e63.c.actor_plan(old)
    n = len(entry["frames"]) - 1
    if set(data) != set(FIELDS) or any(len(array) != n for array in data.values()):
        raise ValueError("Incomplete alternate trajectory")
    np.testing.assert_array_equal(data["frame_id"], np.arange(1, n + 1))
    base.e63.source_common.prior.audit_calls(data, (n, DESIGN["search_budget"]))
    if not np.isfinite(data["decode_error"]).all() or data["decode_error"].max() > ap["design"]["box_atol"]:
        raise ValueError("CENTER decode differs")
    if any(not np.isfinite(data[k]).all() or (data[k] < 0).any() for k in ("search_ms", "pipeline_ms")):
        raise ValueError("Invalid timing")
    history, previous = [np.asarray(entry["initial_box"], float)], None
    for i, frame in enumerate(data["frame_id"]):
        row = {key: array[i] for key, array in data.items()}
        hw = (entry["frames"][frame]["height"], entry["frames"][frame]["width"])
        for key, value in forecast(history, previous, hw, ap, generator).items():
            np.testing.assert_array_equal(row[key], value)
        np.testing.assert_array_equal(row["history_latest"], history[-1])
        np.testing.assert_array_equal(row["candidate_parents"][0], history[-1])
        np.testing.assert_array_equal(row["image_hw"], hw)
        parents = row["candidate_parents"]
        executed = np.array([not any(np.array_equal(p, q) for q in parents[:j])
                             for j, p in enumerate(parents)], bool)
        np.testing.assert_array_equal(row["executed"], executed)
        if int(row["unique_search_calls"]) != executed.sum():
            raise ValueError("Unique search count differs")
        for j, parent in enumerate(parents):
            crop, usable, factor = base.s105.crop_geometry(parent, hw, ap["search_factor"], ap["search_size"])
            np.testing.assert_array_equal(row["crop"][j], crop)
            view = dict(parent=parent, boxes=row["candidate_boxes"][j], crop=crop, usable=usable,
                        image_hw=hw, **{key: row[key][j] for key in ("raw", "size", "offset")})
            np.testing.assert_array_equal(row["features"][j], base.e63.actor.feature_vector(view, history, ap))
            dense = base.e63.dense_boxes(view["size"], view["offset"], parent, factor, ap["search_size"], hw)
            cell = int(np.argmax(view["raw"] * np.asarray(ap["window"], np.float32)))
            np.testing.assert_allclose(dense[cell], view["boxes"], rtol=0, atol=ap["design"]["box_atol"])
        scores, selected = base.choose(row, history, "generated_shared", scorer)
        np.testing.assert_allclose(row["scores"], scores, rtol=1e-9, atol=1e-12)
        np.testing.assert_equal(row["selected"], selected)
        np.testing.assert_array_equal(row["boxes"], row["candidate_boxes"][selected])
        np.testing.assert_array_equal(row["parents"], parents[selected])
        np.testing.assert_equal(row["memory_write"], base.write_eligible(row))
        history = (history + [row["boxes"].copy()])[-ap["design"]["history"]:]
        previous = row


def paired_prefix(treatment, reference):
    used = np.flatnonzero(treatment["alternate_used"])
    end = int(used[0]) if used.size else len(treatment["frame_id"])
    for key in base.FIELDS:
        if key not in ("search_ms", "pipeline_ms"):
            np.testing.assert_array_equal(treatment[key][:end], reference[key][:end],
                                          err_msg="Trajectories differ before alternate search: " + key)
    if used.size:
        slot = DESIGN["slot"]
        np.testing.assert_array_equal(treatment["candidate_parents"][end, :slot],
                                      reference["candidate_parents"][end, :slot])
        np.testing.assert_equal(treatment["unique_search_calls"][end], reference["unique_search_calls"][end])
    return int(treatment["frame_id"][end]) if used.size else None


def report_video(tables, jobs, traces, entry, reuse):
    gt = np.asarray(entry["ground_truth"], float)[1:]
    native = traces["native", 0]
    n = len(gt)
    ious = {key: tracking_iou(trace["boxes"], gt) for key, trace in traces.items()}
    treatment_by_seed = {}
    for job in jobs:
        arm, seed = job["arm"], job["seed"]
        data = traces[arm, seed]
        ident = dict(cohort=job["spec"]["cohort"], sequence=entry["sequence"], arm=arm, seed=seed,
                     outer_fold=job["outer_fold"])
        metric = base.s103.s93.s85.metrics(data, entry, native["boxes"], base.s103.DESIGN)
        overlap = ious[arm, seed]
        events = failure_events(overlap)
        tables["seed_video"].append(dict(ident, **{m: metric[m] for m in METRICS if m != "failure_events"},
                                         failure_events=len(events)))
        first = None
        used = np.zeros(n, bool)
        if arm == TREATMENT:
            first = paired_prefix(data, traces["generated_shared", seed])
            used = data["alternate_used"]
            for ref in REFERENCES:
                ref_iou = ious[ref, 0 if ref == "native" else seed]
                recovery = base.recovery_rows(ref_iou, overlap, ident, ref)
                tables["recovery"].extend(recovery)
                treatment_by_seed.setdefault(seed, {}).update({
                    f"new_failure_vs_{ref}": bool(events) and not failure_events(ref_iou),
                    f"lost_recoveries_vs_{ref}": sum(r["matched_good_frame"] is None for r in recovery)})
        calls = int(data["unique_search_calls"].sum())
        blocks = int((data["block_calls"] * data["executed"][..., None]).sum())
        heads = int((data["head_calls"] * data["executed"]).sum())
        if not n <= calls <= n * (1 if arm == "native" else DESIGN["search_budget"]) or blocks != 7 * calls or heads != calls:
            raise ValueError("Search, block or head budget differs")
        tables["scope"].append(dict(ident, frames=n, provenance="stage113_cache" if reuse and arm != TREATMENT else "new_full_trajectory",
                                    logical_search_calls=n * (1 if arm == "native" else DESIGN["search_budget"]),
                                    physical_search_calls=calls, physical_block_calls=blocks, physical_head_calls=heads,
                                    alternate_requested_frames=int(data["alternate_requested"].sum()) if arm == TREATMENT else 0,
                                    alternate_used_frames=int(used.sum()), first_alternate_frame=first,
                                    alternate_selected_frames=int((used & (data["selected"] == DESIGN["slot"])).sum())))
    empty_protection = dict.fromkeys(next(iter(treatment_by_seed.values())))
    for row in tables["seed_video"][-len(jobs):]:
        row.update(treatment_by_seed[row["seed"]] if row["arm"] == TREATMENT else empty_protection)


def summarize(tables):
    grouped = defaultdict(list)
    for row in tables["seed_video"]:
        grouped[row["sequence"], row["arm"]].append(row)
    macro = {}
    for key, rows in grouped.items():
        expected = [0] if key[1] == "native" else base.DESIGN["seeds"]
        if sorted(row["seed"] for row in rows) != expected:
            raise ValueError("Missing matched seeds")
        macro[key] = {m: float(np.mean(values)) if (values := [r[m] for r in rows if r[m] is not None]) else None
                      for m in METRICS}
    sequences = sorted({seq for seq, _ in macro})
    for arm in (*REFERENCES, TREATMENT):
        rows = [macro[seq, arm] for seq in sequences]
        tables["policy_summary"].append(dict(arm=arm, videos=len(rows), **{
            m: float(np.mean(values)) if (values := [r[m] for r in rows if r[m] is not None]) else None for m in METRICS}))
    for ref in REFERENCES:
        for m in METRICS:
            pairs = [(macro[seq, TREATMENT][m], macro[seq, ref][m]) for seq in sequences]
            values = [a - b for a, b in pairs if a is not None and b is not None]
            if values:
                tables["paired"].append(base.s103.contrast(values, TREATMENT + "_minus_" + ref, m, DESIGN))
    rows = [r for r in tables["seed_video"] if r["arm"] == TREATMENT]
    controls = {(r["sequence"], r["seed"]): r for r in tables["seed_video"] if r["arm"] == "generated_shared"}
    seed_gain = {str(seed): float(np.mean([r["auc"] - controls[r["sequence"], seed]["auc"]
                                         for r in rows if r["seed"] == seed])) for seed in base.DESIGN["seeds"]}
    protection = {ref: dict(new_failure_videos=len({r["sequence"] for r in rows if r[f"new_failure_vs_{ref}"]}),
                            lost_recoveries=sum(r[f"lost_recoveries_vs_{ref}"] for r in rows)) for ref in REFERENCES}
    return seed_gain, protection


def run(root, source_root, source_plan, plan, models, load_entry, old, observer,
        synchronize=lambda: None, image_reader=base.s105.checked_image):
    tables = {key: [] for key in TABLES}
    evidence = {}
    jobs_by_sequence = defaultdict(list)
    for job in plan["jobs"]:
        jobs_by_sequence[job["spec"]["unit"]["sequence"]].append(job)
    for sequence, jobs in jobs_by_sequence.items():
        entry = load_entry(jobs[0]["spec"])
        if entry["sequence"] != sequence or any(j["frames"] != len(entry["frames"]) - 1 for j in jobs):
            raise ValueError("Source video or frame count changed")
        traces = {}
        with base.e63.source_experiment.installed(observer, [], jobs[0]["spec"]["unit"]) as head:
            if head != plan["native_head_sha256"]:
                raise ValueError("Native head identity changed")
            for job in jobs:
                arm, seed = job["arm"], job["seed"]
                original = dict(job, arm="generated_shared") if arm == TREATMENT else job
                generator, scorer = base.job_models(original, models)
                if plan["reuse_controls"] and arm != TREATMENT:
                    meta = dict(plan_fingerprint=source_plan["fingerprint"], job=job,
                                generator_fingerprint=models[job["outer_fold"], "generator", 0]["fingerprint"] if generator is not None else None,
                                scorer_fingerprint=scorer["fingerprint"] if scorer is not None else None)
                    path = source_root / "evaluation" / (base.job_key(job) + ".npz")
                    data = c.read_cache(path, meta)
                    if data is None:
                        raise ValueError(f"Missing frozen Stage-113 control: {path}")
                else:
                    path = root / "evaluation" / (base.job_key(job) + ".npz")
                    meta = dict(plan_fingerprint=plan["fingerprint"], job=job)
                    if arm == TREATMENT:
                        data = base.s109.cache(root, path, meta, lambda: collect(
                            observer, entry["frames"], entry["initial_box"], old, generator, scorer, synchronize, image_reader), evidence)
                        audit(data, entry, old, generator, scorer)
                    else:
                        data = base.s109.cache(root, path, meta, lambda: base.collect(
                            observer, entry, old, arm, generator, scorer, synchronize, image_reader), evidence)
                        base.audit(data, entry, old, arm, generator, scorer)
                traces[arm, seed] = data
                print(f"[Stage-121] {sequence} {arm} seed={seed}: {len(data['frame_id'])} frames", flush=True)
        report_video(tables, jobs, traces, entry, plan["reuse_controls"])
    publish(root, plan, tables, evidence)


def publish(root, plan, tables, evidence):
    seed_gain, protection = summarize(tables)
    metrics = {(r["contrast"], r["metric"]): r for r in tables["paired"]}
    primary = metrics[TREATMENT + "_minus_generated_shared", "auc"]
    new = [r for r in tables["scope"] if r["provenance"] == "new_full_trajectory"]
    usage = {r["sequence"] for r in new if r["alternate_used_frames"]}
    reliable = metrics.get((TREATMENT + "_minus_native", "reliable_auc"))
    checks = dict(practical_gain=primary["mean_delta"] >= DESIGN["practical_auc"] and primary["ci_low"] > 0,
                  both_seeds_positive=all(g > 0 for g in seed_gain.values()),
                  longest_failure_reduced=metrics[TREATMENT + "_minus_generated_shared", "longest_failure_fraction"]["mean_delta"] < 0,
                  low_iou_not_worse=metrics[TREATMENT + "_minus_generated_shared", "low_iou_fraction"]["mean_delta"] <= 0,
                  stable_and_recovery_protected=all(not any(values.values()) for values in protection.values()),
                  reliable_auc_protected=reliable is not None and reliable["ci_low"] >= -DESIGN["reliable_auc_tolerance"],
                  native_and_fixed_auc_matched=all(metrics[TREATMENT + "_minus_" + ref, "auc"]["mean_delta"] >= 0
                                                 for ref in ("native", "fixed_shared")), alternate_used=bool(usage))
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    hashes = {}
    for key, rows in tables.items():
        path = out / f"stage121_{key}.csv"
        base.s103.s86.publish_table(path, rows)
        hashes[path.name] = c.file_hash(path)
    report = io.seal(dict(schema=SCHEMA, status="PROMISING_DEVELOPMENT" if all(checks.values()) else "REVIEW_DEVELOPMENT",
                          plan_fingerprint=plan["fingerprint"], primary=primary, development_checks=checks,
                          auc_delta_by_seed=seed_gain, protection=protection, runtime=plan["runtime"],
                          reuse_controls=plan["reuse_controls"], videos=primary["videos"],
                          new_full_trajectories=len(new), evaluated_trajectories=len(tables["scope"]),
                          videos_using_alternate=len(usage), new_search_calls=sum(r["physical_search_calls"] for r in new),
                          new_block_calls=sum(r["physical_block_calls"] for r in new),
                          new_logical_search_slots=sum(r["logical_search_calls"] for r in new),
                          new_models=0, training_updates=0, confirmation_inference_calls=0,
                          table_sha256=hashes, artifact_sha256=evidence))
    c.publish_identical(out / "stage121_decision.json", report)
    check_results(root)
    print(f"Stage-121: {report['status']}, paired AUC delta {primary['mean_delta']:+.4f}", flush=True)


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    report = c.c66.sealed(root / "analysis/stage121_decision.json", SCHEMA)
    new = [j for j in plan["jobs"] if not plan["reuse_controls"] or j["arm"] == TREATMENT]
    logical = sum(j["frames"] * (1 if j["arm"] == "native" else DESIGN["search_budget"]) for j in new)
    expected_artifacts = {f"evaluation/{base.job_key(j)}.{suffix}" for j in new for suffix in ("npz", "receipt.json")}
    if (plan["design"] != DESIGN or plan["source_stage113_fingerprint"] != SOURCE113
            or plan["fingerprint"] != report["plan_fingerprint"] or plan["runtime"] != report["runtime"]
            or plan["reuse_controls"] != report["reuse_controls"] or report["new_full_trajectories"] != len(new)
            or report["evaluated_trajectories"] != len(plan["jobs"]) or report["new_logical_search_slots"] != logical
            or report["new_block_calls"] != 7 * report["new_search_calls"]
            or not sum(j["frames"] for j in new) <= report["new_search_calls"] <= logical
            or set(report["artifact_sha256"]) != expected_artifacts
            or set(report["table_sha256"]) != {f"stage121_{key}.csv" for key in TABLES}
            or any(report[k] for k in ("new_models", "training_updates", "confirmation_inference_calls"))):
        raise ValueError("Incomplete or mismatched Stage-121 result")
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])
    io.verify({str(root / "analysis" / key): value for key, value in report["table_sha256"].items()})
    io.verify({str(root / key): value for key, value in report["artifact_sha256"].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76", "stage89", "stage93", "stage112", "stage113")
    for name in names:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage121")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not args.checkpoint or not all(getattr(args, name) for name in names):
        parser.error("Supply the original checkpoint and Stage-76/89/93/112/113 directories")
    r76, r89, r93, r112, r113 = sources = [getattr(args, name).expanduser().resolve() for name in names]
    if any(root == source or root in source.parents or source in root.parents for source in sources):
        parser.error("Choose a separate Stage-121 output directory")
    if not (r113 / "manifest.json").is_file():
        parser.error(f"Stage-113 manifest not found: {r113 / 'manifest.json'}")
    with base.s109.s106.s93.writer_lock(root):
        if (root / "analysis/stage121_decision.json").exists():
            check_results(root)
            return
        if not (root / "manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()):
            raise ValueError("Choose an empty Stage-121 output directory")
        base.check_results(r113)
        p113 = c.c66.sealed(r113 / "manifest.json", base.SCHEMA)
        r113_report = c.c66.sealed(r113 / "analysis/stage113_decision.json", base.SCHEMA)
        if r113_report["fingerprint"] != SOURCE113 or len(p113["entries"]) != 28:
            raise ValueError("Need the archived 28-video Stage-113 run")
        p112 = c.c66.sealed(r112 / "manifest.json", base.common.SCHEMA)
        if p112["folds"] != p113["folds"]:
            raise ValueError("Frozen Stage-112 source folds differ")
        bindings = dict(p113["source_sha256"])
        for path in (r113 / "manifest.json", r113 / "analysis/stage113_decision.json", r112 / "manifest.json"):
            bindings[str(path)] = c.file_hash(path)
        for path in (r112 / "models").glob("*.json"):
            if "generator_" in path.name or "__shared__" in path.name:
                if bindings.get(str(path)) != c.file_hash(path):
                    raise ValueError(f"Stage-112 model differs from Stage-113 binding: {path}")
        models, _ = base.load_frozen_models(r112, p112)
        p93, _, p89, report89, origin = base.s103.prior.s95.s94.load_run(r93, r76, r89)

        def load_entry(spec):
            return base.s103.prior.s95.load_unit(r93, r89, p93, p89, report89, origin, spec)[0]

        import torch
        torch.set_num_threads(1)
        old = origin["original"]
        runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
        manifest = root / "manifest.json"
        if manifest.is_file() and c.c66.sealed(manifest, SCHEMA)["runtime"] != runtime:
            raise ValueError("GPU/runtime changed on resume. Set STAGE121_ROOT to a new directory.")
        differences = base.s109.check_runtime(p113["runtime"], runtime)
        reuse = not differences
        jobs = [job for job in p113["jobs"] if job["arm"] in REFERENCES]
        jobs.extend(dict(job, arm=TREATMENT) for job in p113["jobs"] if job["arm"] == "generated_shared")
        if len(jobs) != 196:
            raise ValueError("Incomplete four-arm/two-seed comparison")
        if reuse:
            for job in jobs:
                if job["arm"] == TREATMENT:
                    continue
                for suffix in ("npz", "receipt.json"):
                    relative = f"evaluation/{base.job_key(job)}.{suffix}"
                    bindings[str(r113 / relative)] = r113_report["artifact_sha256"][relative]
        code = [Path(__file__).resolve(), Path(__file__).with_name("run_stage121_search_state.sh").resolve()]
        plan = io.seal(dict(schema=SCHEMA, design=DESIGN, jobs=jobs, source_stage113_fingerprint=SOURCE113,
                            source_sha256=bindings, code_sha256=p113["code_sha256"] | {str(p): c.file_hash(p) for p in code},
                            native_head_sha256=p113["native_head_sha256"], runtime=runtime,
                            runtime_differences=differences, reuse_controls=reuse))
        c.publish_identical(manifest, plan)
        print(f"Stage-121 frozen plan: {56 if reuse else len(jobs)} new full trajectories", flush=True)
        frozen = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = base.s103.s76.startup.checked_runtime(args, frozen, root)
        observer = base.e63.Capture(native)
        try:
            run(root, r113, p113, plan, models, load_entry, old, observer, lambda: torch.cuda.synchronize(args.gpu))
        finally:
            observer.handle.remove()


if __name__ == "__main__":
    main()
