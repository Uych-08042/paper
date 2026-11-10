"""Run the frozen Stage-92 appearance comparison on full development trajectories."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np

import stage85_motion_closed_loop as s85
import stage92_recent_appearance as s92
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, crop_geometry, failure_events, tracking_iou


s91, s89, s86, c, io = s92.s91, s92.s89, s92.s86, s92.c, s92.io
e63, e74, s76, s88 = s92.e63, s92.e74, s92.s76, s92.s89.s88
SCHEMA = "stage93_appearance_closed_loop_v1"
STATUS = "COMPLETE_APPEARANCE_CLOSED_LOOP_DEVELOPMENT"
SOURCE92 = "0d364d32914ca6608ec17f231c890fbb10a2c94dadb7d743b97bc8c82fe91b54"
BUNDLE92 = Path(__file__).resolve().parents[1] / "docs/experiments/06_sgla_uav_tracking/results/stage92_results_20261110T053910Z_1124695.txt"
DESIGN = dict(s92.DESIGN, practical_auc=.5, recovery_window=15, bootstrap_seed=2193)
ARMS = ("native", "ppo", "patch", "gated")
FIELDS = ("queries", "ppo_selected", "challenger", "in_domain", "refs", "memory_write", "used_history", "patch_margin", "margin")
TABLES = ("policy_summary", "paired", "video_summary", "seed_video", "scope", "recovery")
METRICS = ("auc", "low_iou_fraction", "longest_failure_fraction", "reliable_auc", "recovered_events", "unrecovered_events")


def action_bank(row):
    return {k: np.asarray(row[k])[None] for k in
            ("candidate_boxes", "candidate_parents", "history_latest", "features", "selected")}


def appearance_pair(row, design):
    alt, scores = s88.challenger(action_bank(row))
    if alt is None:
        return -1, False
    boxes = row["candidate_boxes"]
    separation = np.linalg.norm(s86.centers(boxes[row["selected"]]) - s86.centers(boxes[alt]))
    separation /= np.sqrt(np.prod(row["history_latest"][2:]))
    return alt, bool(separation < design["center_separation"] or abs(scores["hann"]) <= design["hann_gap"])


def select_memory(memory, first, design):
    recent = []
    for frame, feature in reversed(memory):
        if frame >= first:
            raise ValueError("Current or future appearance entered memory")
        if first - frame > design["lookback"]:
            break
        if not recent or recent[-1][0] - frame >= design["spacing"]:
            recent.append((frame, feature))
            if len(recent) == design["max_refs"]:
                break
    return recent


def choose_write(row, queries, anchor, memory, frame, arm, design):
    ppo = int(row["selected"])
    alt, supported = appearance_pair(row, design)
    reference = select_memory(memory, frame, design) if arm == "gated" else []
    usable = [feature for _, feature in reference if np.isfinite(feature).all()]
    used_history = supported and np.isfinite(queries).all() and len(usable) >= design["min_refs"]
    patch_margin = margin = np.nan
    if supported and np.isfinite(queries).all():
        delta = queries[1] - queries[0]
        patch_margin = float(delta @ anchor) if np.isfinite(anchor).all() else np.nan
        margin = float(delta @ np.mean(usable, axis=0)) if used_history else patch_margin
    selected = alt if np.isfinite(margin) and margin > design["switch_margin"] else ppo
    write = arm == "gated" and s92.trusted(action_bank(dict(row, selected=selected)), 0, design)
    if write:
        memory.append((frame, queries[int(selected != ppo)].copy()))
    memory[:] = [(f, q) for f, q in memory if frame - f < design["lookback"]]
    refs = [f for f, _ in reference]
    return dict(selected=selected, ppo_selected=ppo, challenger=alt, in_domain=supported, memory_write=write,
                refs=np.asarray(refs + [-1] * (design["max_refs"] - len(refs))), used_history=used_history,
                patch_margin=patch_margin, margin=margin)


def pool(image, box, parent, factor, size, grid, design):
    crop, usable, _ = crop_geometry(parent, image.shape[:2], factor, size)
    roi, coverage = s91.visible_roi(box, usable)
    if coverage < design["min_roi_coverage"]:
        return np.full(grid.shape[1], np.nan)
    value = s91.descriptor(grid, crop, roi)
    return value if value is not None else np.full(grid.shape[1], np.nan)


def collect(observer, entry, ap, choose, arm, design):
    initial = np.asarray(entry["initial_box"], float)
    template_image = checked_image(ap["root"], entry["frames"][0])
    observer.initialize(template_image, initial)
    rt = observer.rt
    params = rt.tracker.params
    history, memory, rows, tokens = [initial], [], [], []
    anchor = None

    def capture(_module, _args, output):
        # Position embeddings are added in place immediately after this hook.
        grid = s91.cpu_array(output)
        if grid.ndim != 3 or grid.shape[0] != 1:
            raise ValueError("Expected one patch grid")
        tokens.append(grid[0])

    with rt.net.backbone.patch_embed.register_forward_hook(capture):
        for frame, image_entry in enumerate(entry["frames"][1:], 1):
            image = checked_image(ap["root"], image_entry)
            tokens.clear()
            row = choose(e63.probe(observer, image, history, frame, "history", ap), history)
            if len(tokens) != 8 or any(len(grid) != (rt.lens_z if k % 2 == 0 else rt.lens_x) for k, grid in enumerate(tokens)):
                raise ValueError("Expected four native template/search pairs")
            if anchor is None:
                anchor = pool(template_image, initial, initial, params.template_factor, params.template_size, tokens[0], design)
            alt, _ = appearance_pair(row, design)
            queries = np.full((2, len(anchor)), np.nan)
            for k, slot in enumerate((int(row["selected"]), alt)):
                if slot >= 0:
                    queries[k] = pool(image, row["candidate_boxes"][slot], row["candidate_parents"][slot],
                                      ap["search_factor"], ap["search_size"], tokens[2 * slot + 1], design)
            write = choose_write(row, queries, anchor, memory, frame, arm, design)
            box = row["candidate_boxes"][write["selected"]].copy()
            rows.append(row | write | dict(boxes=box, queries=queries))
            history = (history + [box])[-ap["design"]["history"]:]
            observer.rt.tracker.state = box.tolist()
            if frame % 500 == 0:
                print(f"  [Stage-93] {arm} {frame}/{len(entry['frames']) - 1}", flush=True)
    return dict({k: np.asarray([row[k] for row in rows]) for k in s86.FIELDS + FIELDS}, anchor=anchor)


def audit(data, entry, ap, choose, arm, design):
    prefix = np.asarray([entry["initial_box"]], float)
    frames = np.arange(1, len(entry["frames"]))
    memory, index = [], 0
    anchor = data["anchor"]
    if anchor.ndim != 1 or data["queries"].shape != (len(frames), 2, len(anchor)):
        raise ValueError("Incomplete cached appearance")

    def audited_choice(row, history):
        nonlocal index
        decision = choose(row, history)
        decision["history_latest"] = history[-1]
        write = choose_write(decision, data["queries"][index], anchor, memory, index + 1, arm, design)
        for key, value in write.items():
            np.testing.assert_array_equal(data[key][index], value)
        index += 1
        return dict(write, boxes=decision["candidate_boxes"][write["selected"]])

    s86.audit({k: data[k] for k in s86.FIELDS}, prefix, frames, 0., ap, audited_choice)


def recovery_comparison(reference, candidate, design):
    result = []
    for event in failure_events(reference):
        start = event["recovery"]
        if start is None:
            continue
        width = design["recovery_run"]
        search = candidate[start:min(len(candidate), start + design["recovery_window"] + width)]
        good = np.lib.stride_tricks.sliding_window_view(search >= design["good_iou"], width).all(axis=1)
        matches = np.flatnonzero(good)
        result.append(dict(ppo_failure_frame=event["start"] + 1, ppo_recovery_frame=start + 1,
                           matched_good_frame=int(start + matches[0] + 1) if len(matches) else None))
    return result


def summarize(rows, design):
    videos, policies, paired = [], [], []
    for (cohort, sequence, arm), entries in sorted(group_rows(rows).items()):
        stats = {k: float(np.mean([r[k] for r in entries])) for k in METRICS if k != "reliable_auc"}
        valid = [r["reliable_auc"] for r in entries if r["reliable_auc"] is not None]
        stats["reliable_auc"] = float(np.mean(valid)) if valid else None
        videos.append(dict(cohort=cohort, sequence=sequence, arm=arm, **stats,
                           new_on_native_stable=any(r["new_on_native_stable"] for r in entries),
                           new_on_ppo_stable=any(r["new_on_ppo_stable"] for r in entries) if arm != "native" else None,
                           additional_on_native_stable=any(r["additional_on_native_stable"] for r in entries) if arm != "native" else None,
                           ppo_recoveries_lost=sum(r["ppo_recoveries_lost"] for r in entries) if arm != "native" else None))
    for cohort in ("all", "historical", "expanded"):
        group = [r for r in videos if cohort == "all" or r["cohort"] == cohort]
        if not group:
            continue
        indexed = {(r["sequence"], r["arm"]): r for r in group}
        for arm in ARMS:
            items = [r for r in group if r["arm"] == arm]
            policy = dict(cohort=cohort, arm=arm, videos=len(items))
            for k in METRICS:
                values = [r[k] for r in items if r[k] is not None]
                policy[k] = float(np.mean(values)) if values else None
            for k in ("new_on_native_stable", "new_on_ppo_stable", "additional_on_native_stable", "ppo_recoveries_lost"):
                values = [r[k] for r in items if r[k] is not None]
                policy[k] = sum(values) if values else None
            policies.append(policy)
        for arm, base in (("ppo", "native"), ("patch", "ppo"), ("gated", "native"), ("gated", "ppo"), ("gated", "patch")):
            for metric in METRICS:
                diff = []
                for seq in sorted({r["sequence"] for r in group}):
                    a, b = indexed[seq, arm][metric], indexed[seq, base][metric]
                    if a is not None and b is not None:
                        diff.append(a - b)
                if diff:
                    paired.append(dict(s86.contrast(diff, f"{arm}_minus_{base}", metric, design), cohort=cohort))
    return policies, paired, videos


def group_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["cohort"], row["sequence"], row["arm"]].append(row)
    return groups


def cached_track(root, plan, unit, arm, build, verify, evidence):
    path = root / "evaluation" / (c.key(unit) + "__" + arm + ".npz")
    meta = dict(plan_fingerprint=plan["fingerprint"], unit=unit, arm=arm)
    trace = c.read_cache(path, meta)
    fresh = trace is None
    if fresh:
        trace = build()
    verify(trace)
    if fresh:
        c.save(path, trace, meta)
    evidence[path.name] = c.file_hash(path)
    return trace


def describe_arm(entry, cohort, unit, arm, trace, native, ppo, design):
    truth = np.asarray(entry["ground_truth"], float)[1:]
    native_iou, ppo_iou = (tracking_iou(d["boxes"], truth) for d in (native, ppo))
    scores, events = s85.describe(trace, entry, native["boxes"], native_iou < design["low_iou"])
    ppo_stable, native_stable = not failure_events(ppo_iou), not failure_events(native_iou)
    recovery = recovery_comparison(ppo_iou, tracking_iou(trace["boxes"], truth), design) if arm != "native" else []
    ident = dict(cohort=cohort, sequence=unit["sequence"], seed=unit["seed"], arm=arm)
    return (dict(ident, **{k: scores[k] for k in METRICS}, new_on_native_stable=native_stable and bool(events),
                 new_on_ppo_stable=ppo_stable and bool(events) if arm != "native" else None,
                 additional_on_native_stable=native_stable and ppo_stable and bool(events) if arm != "native" else None,
                 ppo_recovery_opportunities=len(recovery) if arm != "native" else None,
                 ppo_recoveries_lost=sum(r["matched_good_frame"] is None for r in recovery) if arm != "native" else None),
            [dict(ident, **row) for row in recovery])


def collect_run(root, plan, source, actor, root89, plan89, observer):
    design = plan["design"]
    old = source["original"]
    ap = e74.e67.actor_plan(old)
    evidence, seeds, scope, recoveries = {}, [], [], []
    grouped = defaultdict(list)
    for spec in plan89["units"]:
        grouped[spec["unit"]["sequence"]].append(spec)
    for sequence, specs in grouped.items():
        cohort = specs[0]["cohort"]
        if cohort == "expanded":
            bound = c.c66.sealed(root89 / "entries" / (sequence + ".json"), s89.SCHEMA)
            if bound["plan_fingerprint"] != plan89["fingerprint"] or bound["entry"]["sequence"] != sequence:
                raise ValueError("Stage-89 expanded entry changed")
            entry = bound["entry"]
        else:
            entry = c.entry_for(old, sequence)
        frames = np.arange(1, len(entry["frames"]))
        initial = np.asarray([entry["initial_box"]], float)
        seed_unit = specs[0]["unit"]
        native_unit = dict(seed_unit, policy="q0", seed=0)
        native_cached = cohort == "historical" and plan["reuse_native_historical"]
        if native_cached:
            _, native = s86.previous.read_source_trace(source, dict(unit=native_unit, frames=[1], expected_first_failure=None))
            if np.any(native["selected"] != 0):
                raise ValueError("Historical native trace is not q0")
        else:
            with e63.source_experiment.installed(observer, [], native_unit) as head:
                if head != source["report"]["native_head_sha256"]:
                    raise ValueError("Original SGLA head changed")
                native = cached_track(root, plan, native_unit, "native",
                    lambda: s85.collect(observer, entry, "native_sgla", ap, None, {}),
                    lambda d: s85.audit(d, entry, "native_sgla", ap, None, {}), evidence)
        baseline, _ = describe_arm(entry, cohort, native_unit, "native", native, native, native, design)
        seeds.append(baseline)
        scope.append(dict(cohort=cohort, sequence=sequence, seed=0, arm="native", frames=len(frames),
                          logical_search_calls=len(frames), new_search_calls=0 if native_cached else len(frames),
                          switches=0, first_switch_frame=None, eligible_frames=0, frames_using_history=0))
        for spec in specs:
            unit = spec["unit"]
            pack = e74.learned_pack(actor["models"], unit, actor["prior"])

            def choose(row, history):
                return e74.choose(row, history, unit, old, actor["prior"], pack)

            with e63.source_experiment.installed(observer, [], unit) as head:
                if head != source["report"]["native_head_sha256"]:
                    raise ValueError("Original SGLA head changed")
                if plan["reuse_ppo"]:
                    if cohort == "historical" and plan89["reuse_historical"]:
                        _, ppo = s86.previous.read_source_trace(source, dict(unit=unit, frames=[1], expected_first_failure=None))
                    else:
                        ppo = c.read_cache(root89 / "baseline" / (c.key(unit) + ".npz"),
                                           dict(plan_fingerprint=plan89["fingerprint"], unit=unit))
                        if ppo is None:
                            raise ValueError("Missing frozen PPO baseline")
                    s86.audit({k: ppo[k] for k in s86.FIELDS}, initial, frames, 0., ap, choose)
                else:
                    ppo = cached_track(root, plan, unit, "ppo",
                        lambda: s86.collect(observer, entry, initial, frames, 0., ap, choose),
                        lambda d: s86.audit(d, initial, frames, 0., ap, choose), evidence)
                data = {"ppo": ppo}
                for arm in ("patch", "gated"):
                    data[arm] = cached_track(root, plan, unit, arm,
                        lambda: collect(observer, entry, ap, choose, arm, design),
                        lambda d: audit(d, entry, ap, choose, arm, design), evidence)
                for arm, trace in data.items():
                    scores, events = describe_arm(entry, cohort, unit, arm, trace, native, ppo, design)
                    seeds.append(scores)
                    recoveries.extend(events)
                    switched = trace["selected"] != trace.get("ppo_selected", trace["selected"])
                    equal_prefix = int(np.flatnonzero(switched)[0]) if switched.any() else len(frames)
                    if arm != "ppo":
                        # The first alternative must arise at exactly the same observation state.
                        through_first = min(len(frames), equal_prefix + 1)
                        for k in ("candidate_boxes", "crop_sha256", "features"):
                            np.testing.assert_array_equal(trace[k][:through_first], ppo[k][:through_first])
                        np.testing.assert_array_equal(trace["boxes"][:equal_prefix], ppo["boxes"][:equal_prefix])
                    scope.append(dict(cohort=cohort, sequence=sequence, seed=unit["seed"], arm=arm, frames=len(frames),
                        logical_search_calls=4 * len(frames), new_search_calls=0 if arm == "ppo" and plan["reuse_ppo"] else 4 * len(frames),
                        switches=int(switched.sum()), first_switch_frame=equal_prefix + 1 if switched.any() else None,
                        eligible_frames=int(np.count_nonzero(trace.get("in_domain", False))),
                        frames_using_history=int(np.count_nonzero(trace.get("used_history", False)))))
            print(f"[Stage-93] {sequence} seed={unit['seed']} completed", flush=True)
    publish(root, plan, evidence, seeds, scope, recoveries)


def publish(root, plan, evidence, seeds, scope, recoveries):
    if len(seeds) != plan["trajectories"] or len(scope) != len(seeds) or len(evidence) != plan["new_trajectories"]:
        raise ValueError("Incomplete Stage-93 trajectory coverage")
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])
    policies, paired, videos = summarize(seeds, plan["design"])
    macro = {r["arm"]: r for r in policies if r["cohort"] == "all"}
    au = {r["contrast"]: r for r in paired if r["cohort"] == "all" and r["metric"] == "auc"}
    proposed = macro["gated"]
    switches = sum(r["switches"] for r in scope if r["arm"] == "gated")
    checks = dict(practical_auc_vs_ppo=au["gated_minus_ppo"]["mean_delta"] >= plan["design"]["practical_auc"],
        positive_ci_vs_ppo=au["gated_minus_ppo"]["ci_low"] > 0,
        positive_ci_vs_fixed_template=au["gated_minus_patch"]["ci_low"] > 0,
        low_iou_not_worse=proposed["low_iou_fraction"] <= macro["ppo"]["low_iou_fraction"],
        longest_failure_not_worse=proposed["longest_failure_fraction"] <= macro["ppo"]["longest_failure_fraction"],
        no_new_on_ppo_stable=proposed["new_on_ppo_stable"] == 0,
        no_ppo_recovery_lost=proposed["ppo_recoveries_lost"] == 0,
        recovery_supported=any(r["arm"] == "gated" for r in recoveries), nontrivial_usage=switches > 0)
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    hashes = {}
    for name, data in zip(TABLES, (policies, paired, videos, seeds, scope, recoveries)):
        path = out / f"stage93_{name}.csv"
        s86.publish_table(path, data)
        hashes[path.name] = c.file_hash(path)
    calls = sum(r["new_search_calls"] for r in scope)
    c.publish_identical(out / "stage93_decision.json", io.seal(dict(schema=SCHEMA, status=STATUS,
        plan_fingerprint=plan["fingerprint"], source_stage92_fingerprint=SOURCE92, runtime=plan["runtime"],
        trajectories=len(scope), new_search_calls=calls, new_block_calls=7 * calls,
        logical_search_calls=sum(r["logical_search_calls"] for r in scope), gated_switches=switches,
        training_updates=0, confirmation_inference_calls=0, development_checks=checks, development_pass=all(checks.values()),
        cache_sha256=evidence, table_sha256=hashes)))
    print(f"Stage-93: {STATUS} ({len(scope)} trajectories, {calls} new searches)", flush=True)


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    report = c.c66.sealed(root / "analysis/stage93_decision.json", SCHEMA)
    if (report["status"] != STATUS or report["plan_fingerprint"] != plan["fingerprint"]
            or report["trajectories"] != plan["trajectories"] or report["new_block_calls"] != 7 * report["new_search_calls"]
            or set(report["table_sha256"]) != {f"stage93_{name}.csv" for name in TABLES}
            or len(report["cache_sha256"]) != plan["new_trajectories"]):
        raise ValueError("Incomplete Stage-93 report")
    for folder, key in (("analysis", "table_sha256"), ("evaluation", "cache_sha256")):
        io.verify({str(root / folder / name): sha for name, sha in report[key].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("stage76", "stage89", "output_dir"):
        parser.add_argument("--" + flag, type=Path, required=flag == "output_dir")
    parser.add_argument("--bundle92", type=Path, default=BUNDLE92)
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage93")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not all((args.stage76, args.stage89, args.checkpoint)):
        parser.error("Supply Stage-76, Stage-89 and the original checkpoint")
    root76, root89 = (p.expanduser().resolve() for p in (args.stage76, args.stage89))
    if any(root == p or p in root.parents or root in p.parents for p in (root76, root89)):
        parser.error("Use a separate Stage-93 output directory")
    import torch

    torch.set_num_threads(1)
    with writer_lock(root):
        if (root / "analysis/stage93_decision.json").exists():
            check_results(root)
            return
        if not (root / "manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-93 directory")
        archive = args.bundle92.read_bytes()
        decision = json.loads(archive.split(b"===== stage92_decision.json =====", 1)[1].split(b"===== ", 1)[0])
        if (decision.pop("fingerprint") != SOURCE92 or c.fingerprint(decision) != SOURCE92 or decision["status"] != s92.STATUS):
            raise ValueError("Need the frozen Stage-92 result bundle")
        s89.check_results(root89)
        plan89 = c.c66.sealed(root89 / "manifest.json", s89.SCHEMA)
        report89 = c.c66.sealed(root89 / "analysis/stage89_decision.json", s89.SCHEMA)
        if report89["fingerprint"] != s92.s90.SOURCE_FINGERPRINT:
            raise ValueError("Need the Stage-89 cohort used in Stage-92")
        roster = c.c66.sealed(s86.ROSTER, s86.confirmation.SCHEMA)
        held = {r["source_group_proxy"] for r in roster["sequences"]}
        units = plan89["units"]
        groups = {s89.source_group(r["unit"]["sequence"]) for r in units}
        if (roster["fingerprint"] != plan89["confirmation_fingerprint"] or held & groups or len(held) != 28
                or len(groups) != 28 or len(units) != 40):
            raise ValueError("Development/confirmation cohort changed")
        source = s86.previous.load_source(root76)
        if source["report"]["fingerprint"] != plan89["source_stage76_fingerprint"]:
            raise ValueError("Stage-76 source changed")
        actor = io.load_source(Path(source["plan"]["source_input"]))
        if actor["plan"] != source["original"] or actor["models"]["fingerprint"] != source["report"]["source_training_fingerprint"]:
            raise ValueError("Stage-74 actor changed")
        runtime = s76.device_runtime(source["original"], torch, args.gpu)
        if any(r["field"] not in ("runtime.gpu_uuid", "runtime.tracker.device")
               for r in s76.startup.differences(report89["runtime"], runtime)):
            raise ValueError("Frozen SGLA runtime software or weights changed")
        device = runtime["tracker"]["device"]
        reuse_ppo = device == report89["runtime"]["tracker"]["device"]
        reuse_native = device == source["plan"]["runtime"]["tracker"]["device"]
        code = [Path(__file__), Path(s92.__file__), Path(s91.__file__), Path(e74.__file__), Path(s85.__file__),
                Path(__file__).with_name("run_stage93_appearance_closed_loop.sh")]
        paths = (args.bundle92.resolve(), s86.ROSTER, root89 / "manifest.json", root89 / "analysis/stage89_decision.json")
        plan = io.seal(dict(schema=SCHEMA, design=DESIGN, runtime=runtime, source_stage92_fingerprint=SOURCE92,
            trajectories=len(groups) + 3 * len(units), new_trajectories=len(groups) + (2 if reuse_ppo else 3) * len(units) - (12 if reuse_native else 0),
            reuse_ppo=reuse_ppo, reuse_native_historical=reuse_native,
            source_sha256={str(path): c.file_hash(path) for path in paths} | source["evidence"] | actor["bindings"],
            code_sha256={str(path.resolve()): c.file_hash(path) for path in code}))
        c.publish_identical(root / "manifest.json", plan)
        frozen = dict(source["original"], fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = s76.startup.checked_runtime(args, frozen, root)
        observer = e63.Capture(native)
        try:
            collect_run(root, plan, source, actor, root89, plan89, observer)
        finally:
            observer.handle.remove()


if __name__ == "__main__":
    main()
