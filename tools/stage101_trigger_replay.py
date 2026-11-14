"""Replay every Stage-100 pulse and fixed continuations at matched own states."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import os
from pathlib import Path

import numpy as np

import stage100_native_compat as compat
from stage27_uav_common import checked_image, crop_geometry, failure_events, scene_geometry, tracking_iou


s100 = compat.original
s97, s93, s95, c, io = s100.s97, s100.s93, s100.s95, s100.c, s100.io
e63, e74, s76, s86 = s100.e63, s100.e74, s100.s76, s100.s86
SCHEMA = "stage101_matched_online_scale_trigger_v1"
STATUS = "COMPLETE_MATCHED_SCALE_TRIGGER_DIAGNOSTIC"
SOURCE100 = "9b237b9c93ad05f7e9fbcf38d203c38ad33cfad432ff184c6a3069f353054915"
DESIGN = dict(s100.DESIGN, horizon=16)
TREATMENTS = ("keep", "skip")
TABLES = ("trigger_summary", "trigger_video", "anchors", "branches", "branch_summary", "triggers", "candidates")
WINDOW_METRICS = ("mean_iou", "endpoint_iou", "low_iou_fraction", "good_candidate_fraction")


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_unit(root100, plan100, root93, root89, plan93, plan89, report89, source, spec):
    entry, _ = s95.load_unit(root93, root89, plan93, plan89, report89, source, spec)
    unit = spec["unit"]
    traces = {arm: s95.s94.required_cache(root100 / "evaluation" / f"{c.key(unit)}__{arm}.npz",
              dict(plan_fingerprint=plan100["fingerprint"], unit=unit, arm=arm)) for arm in ("gated", "pulse")}
    return entry, traces


def select_anchors(spec, pulse, gated, entry, lost, design):
    """Select source-only diagnostic anchors before any counterfactual inference."""
    triggers = np.flatnonzero(pulse["scale_active"]).astype(int)+1
    if not len(triggers):
        return [], []
    n = len(pulse["frame_id"])
    truth = np.asarray(entry["ground_truth"], float)[1:]
    p, g = (tracking_iou(t["boxes"], truth) for t in (pulse, gated))
    anchors = {}

    def add(target, role, recovery=None):
        available = triggers[triggers <= target]
        if not len(available):
            raise ValueError("Outcome diverged before any pulse")
        frame = int(available[-1])
        row = anchors.setdefault(frame, dict(frame=frame, last=min(n, frame+design["horizon"]-1), roles=[]))
        row["roles"].append(dict(role=role, target_frame=int(target), reference_recovery=recovery))
        if recovery is not None:
            row["last"] = min(n, max(row["last"], recovery+design["recovery_window"]+design["recovery_run"]-1))

    add(int(triggers[0]), "first_pulse")
    for role, mask in (("first_observed_harm", (g >= design["good_iou"]) & (p >= 0) & (p < design["low_iou"])),
                       ("first_observed_gain", (p >= design["good_iou"]) & (g >= 0) & (g < design["low_iou"]))):
        positions = np.flatnonzero(mask)
        if len(positions):
            add(int(positions[0])+1, role)
    for sequence, seed, _onset, recovery in sorted(lost):
        if (sequence, seed) == (spec["unit"]["sequence"], spec["unit"]["seed"]):
            add(recovery, "newly_lost_ppo_recovery", recovery)
    return triggers.tolist(), [anchors[f] for f in sorted(anchors)]


def source_slice(trace, first, last):
    return {k: trace[k][first-1:last].copy() for k in s100.FIELDS} | dict(anchor=trace["anchor"].copy())


def initial_state(trace, entry, first, old, design):
    memory, sizes = s97.starting_memory(trace, first, design)
    return dict(history=list(e63.history_at(trace, entry["initial_box"], first, old)), memory=memory,
                sizes=sizes, previous=bool(trace["scale_request"][first-2]) if first > 1 else False)


def row_mode(treatment, frame, first):
    if treatment not in TREATMENTS:
        raise ValueError("Unknown matched treatment")
    return "gated" if treatment == "skip" and frame == first else "pulse"


def advance(row, state, anchor, frame, first, treatment, ap, choose, design):
    parents, refs, request, episode = s100.parents_for(state["history"], state["sizes"], frame,
        row_mode(treatment, frame, first), state["previous"], row["image_hw"], ap, design)
    for key, value in dict(candidate_parents=parents, history_latest=state["history"][-1],
            scale_refs=refs, scale_request=request, scale_episode_start=episode,
            scale_active=bool((refs >= 0).any())).items():
        np.testing.assert_array_equal(row[key], value, err_msg=f"{frame}:{key}")
    np.testing.assert_array_equal(row["crop"], [crop_geometry(p, row["image_hw"], ap["search_factor"], ap["search_size"])[0]
                                                 for p in parents])
    np.testing.assert_array_equal([row["scale_changed"], row["scale_crop_changed"]],
                                 s97.scale_changes(state["history"], parents, row["image_hw"], ap))
    decision_row = choose({k: row[k].copy() for k in ("features", "candidate_boxes", "candidate_parents")}, state["history"])
    decision_row["history_latest"] = state["history"][-1]
    decision = s93.choose_write(decision_row, row["queries"], anchor, state["memory"], frame, "gated", design)
    for key, value in decision.items():
        if key in ("patch_margin", "margin"):
            np.testing.assert_allclose(row[key], value, rtol=0, atol=1e-12, equal_nan=True)
        else:
            np.testing.assert_array_equal(row[key], value)
    box = decision_row["candidate_boxes"][decision["selected"]]
    np.testing.assert_array_equal(row["boxes"], box)
    accepted = s97.scale_write(decision, row["queries"])
    np.testing.assert_array_equal(row["scale_written"], accepted)
    s97.update_sizes(state["sizes"], frame, box, accepted, design)
    state["history"] = (state["history"]+[box])[-ap["design"]["history"]:]
    state["previous"] = request


def observe(observer, image, state, anchor, frame, first, treatment, ap, choose, design):
    parents, refs, request, episode = s100.parents_for(state["history"], state["sizes"], frame,
        row_mode(treatment, frame, first), state["previous"], image.shape[:2], ap, design)
    tokens = []

    def capture(_module, _args, output):
        grid = s93.s91.cpu_array(output)
        if grid.ndim != 3 or grid.shape[0] != 1:
            raise ValueError("Expected one patch embedding grid")
        tokens.append(grid[0])

    with observer.rt.net.backbone.patch_embed.register_forward_hook(capture):
        row = choose(s97.probe(observer, image, state["history"], parents, frame, ap), state["history"])
    if len(tokens) != 8 or any(len(t) != (observer.rt.lens_z if i % 2 == 0 else observer.rt.lens_x)
                               for i, t in enumerate(tokens)):
        raise ValueError("Expected four template/search pairs")
    alt, _ = s93.appearance_pair(row, design)
    queries = np.full((2, len(anchor)), np.nan)
    for slot, q in enumerate((int(row["selected"]), alt)):
        if q >= 0:
            queries[slot] = s93.pool(image, row["candidate_boxes"][q], parents[q], ap["search_factor"],
                                    ap["search_size"], tokens[2*q+1], design)
    decision = s93.choose_write(row, queries, anchor, list(state["memory"]), frame, "gated", design)
    box = row["candidate_boxes"][decision["selected"]].copy()
    changed, crop_changed = s97.scale_changes(state["history"], parents, image.shape[:2], ap)
    return row | decision | dict(boxes=box, queries=queries, scale_refs=refs, scale_request=request,
        scale_episode_start=episode, scale_active=bool((refs >= 0).any()),
        scale_written=s97.scale_write(decision, queries), scale_changed=changed, scale_crop_changed=crop_changed)


def collect(observer, trace, entry, first, last, treatment, old, choose, design, shared=None, image_reader=checked_image):
    ap = e63.c.actor_plan(old)
    observer.initialize(image_reader(ap["root"], entry["frames"][0]), entry["initial_box"])
    state = initial_state(trace, entry, first, old, design)
    anchor, rows = trace["anchor"].copy(), []
    for frame in range(first, last+1):
        row = ({k: shared[k][0].copy() for k in s100.FIELDS} if frame == first and shared is not None else
               observe(observer, image_reader(ap["root"], entry["frames"][frame]), state, anchor,
                       frame, first, treatment, ap, choose, design))
        advance(row, state, anchor, frame, first, treatment, ap, choose, design)
        rows.append(row)
        observer.rt.tracker.state = row["boxes"].tolist()
    return {k: np.asarray([r[k] for r in rows]) for k in s100.FIELDS} | dict(anchor=anchor)


def audit(data, trace, entry, first, last, treatment, old, choose, design):
    if set(data) != set(s100.FIELDS) | {"anchor"}:
        raise ValueError("Incomplete matched continuation")
    np.testing.assert_array_equal(data["frame_id"], np.arange(first, last+1))
    np.testing.assert_array_equal(data["anchor"], trace["anchor"])
    e63.source_common.prior.audit_calls(data, (last-first+1, 4))
    if not np.isfinite(data["decode_error"]).all() or data["decode_error"].max() > old["actor_design"]["box_atol"]:
        raise ValueError("Decode drift")
    ap, state = e63.c.actor_plan(old), initial_state(trace, entry, first, old, design)
    for i, frame in enumerate(range(first, last+1)):
        row = {k: data[k][i].copy() for k in s100.FIELDS}
        np.testing.assert_array_equal(row["image_hw"], [entry["frames"][frame]["height"], entry["frames"][frame]["width"]])
        advance(row, state, data["anchor"], frame, first, treatment, ap, choose, design)


def frame_metrics(data, truth, design):
    bank = tracking_iou(data["candidate_boxes"][0], truth)
    selected = int(data["selected"][0])
    valid = bool(np.isfinite(truth).all() and np.all(truth[2:] > 0))
    return dict(iou=float(bank[selected]) if valid else None, best_iou=float(bank.max()) if valid else None,
        selected=selected, ppo_selected=int(data["ppo_selected"][0]), memory_write=bool(data["memory_write"][0]),
        good_candidate=bool(bank.max() >= design["good_iou"]) if valid else None,
        missed_good=bool(bank.max() >= design["good_iou"] and bank[selected] < design["good_iou"]) if valid else None)


def window_metrics(data, entry, design):
    truth = np.asarray(entry["ground_truth"], float)[data["frame_id"]]
    iou = tracking_iou(data["boxes"], truth)
    best = tracking_iou(data["candidate_boxes"], truth[:, None]).max(1)
    valid = np.isfinite(truth).all(1) & (truth[:, 2:] > 0).all(1) & (iou >= 0) & (best >= 0)
    return dict(mean_iou=float(iou[valid].mean()) if valid.any() else None,
        endpoint_iou=float(iou[-1]) if valid[-1] else None,
        low_iou_fraction=float((iou[valid] < design["low_iou"]).mean()) if valid.any() else None,
        good_candidate_fraction=float((best[valid] >= design["good_iou"]).mean()) if valid.any() else None,
        failure_events=len(failure_events(np.where(valid, iou, -1.))), active_frames=int(data["scale_active"].sum()))


def aggregate(triggers, branches):
    units = defaultdict(list)
    for row in triggers:
        if row["skip_minus_keep_iou"] is not None:
            units[row["cohort"], row["sequence"], row["seed"]].append(row)
    grouped = defaultdict(list)
    for (cohort, sequence, seed), rows in units.items():
        grouped[cohort, sequence].append(dict(seed=seed, states=len(rows),
            iou_gain=float(np.mean([r["skip_minus_keep_iou"] for r in rows])),
            best_gain=float(np.mean([r["skip_best_iou"]-r["keep_best_iou"] for r in rows]))))
    videos = [dict(cohort=k[0], sequence=k[1], units=len(v), states=sum(r["states"] for r in v),
        iou_gain=float(np.mean([r["iou_gain"] for r in v])), best_gain=float(np.mean([r["best_gain"] for r in v])))
        for k, v in sorted(grouped.items())]
    summary = []
    for cohort in ("all", "historical", "expanded"):
        local = [r for r in videos if cohort == "all" or r["cohort"] == cohort]
        states = [r for r in triggers if (cohort == "all" or r["cohort"] == cohort) and r["skip_minus_keep_iou"] is not None]
        if local:
            summary.append(dict(cohort=cohort, states=len(states), videos=len(local),
                mean_iou_gain_video=float(np.mean([r["iou_gain"] for r in local])),
                mean_best_gain_video=float(np.mean([r["best_gain"] for r in local])),
                skip_iou_wins=sum(r["skip_minus_keep_iou"] > 1e-12 for r in states),
                skip_iou_ties=sum(abs(r["skip_minus_keep_iou"]) <= 1e-12 for r in states),
                skip_iou_losses=sum(r["skip_minus_keep_iou"] < -1e-12 for r in states)))
    branch_summary = []
    for role in sorted({role for r in branches for role in r["roles"].split("|")}):
        local = [r for r in branches if role in r["roles"].split("|")]
        by_video = defaultdict(lambda: defaultdict(list))
        for row in local:
            by_video[row["sequence"]][row["seed"]].append(row)
        result = dict(role=role, anchors=len(local), videos=len(by_video))
        for key in WINDOW_METRICS:
            video_values = []
            for seeds in by_video.values():
                seed_values = []
                for rows in seeds.values():
                    delta = [r["skip_"+key]-r["keep_"+key] for r in rows
                             if r["skip_"+key] is not None and r["keep_"+key] is not None]
                    if delta:
                        seed_values.append(float(np.mean(delta)))
                if seed_values:
                    video_values.append(float(np.mean(seed_values)))
            result[key+"_skip_gain_video"] = float(np.mean(video_values)) if video_values else None
        branch_summary.append(result)
    return summary, videos, branch_summary


def budget(plan):
    states = sum(len(u["points"]) for u in plan["units"])
    anchors = sum(len(u["anchors"]) for u in plan["units"])
    tail = sum(a["last"]-a["frame"] for u in plan["units"] for a in u["anchors"])
    return states, anchors, 4*(states+tail)*(1 if plan["reuse_keep"] else 2)


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    report = c.c66.sealed(root / "analysis/stage101_decision.json", SCHEMA)
    states, anchors, calls = budget(plan)
    if (report["status"] != STATUS or report["plan_fingerprint"] != plan["fingerprint"]
            or report["matched_trigger_states"] != states or report["short_branches"] != 2*anchors
            or len(report["cache_sha256"]) != 2*(states+anchors) or report["new_search_calls"] != calls
            or report["new_block_calls"] != 7*calls or set(report["table_sha256"]) != {f"stage101_{n}.csv" for n in TABLES}):
        raise ValueError("Incomplete Stage-101 diagnostic")
    io.verify({str(root / p): h for p, h in report["cache_sha256"].items()})
    io.verify({str(root / "analysis" / p): h for p, h in report["table_sha256"].items()})
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])


def prepare_units(root100, plan100, root93, root89, plan93, plan89, report89, source, actor, lost):
    result = []
    ap = e63.c.actor_plan(source["original"])
    for spec in plan100["units"]:
        entry, traces = load_unit(root100, plan100, root93, root89, plan93, plan89, report89, source, spec)
        unit = spec["unit"]
        pack = e74.learned_pack(actor["models"], unit, actor["prior"])
        choose = lambda row, history: e74.choose(row, history, unit, source["original"], actor["prior"], pack)
        for arm, trace in traces.items():
            s100.audit(trace, entry, ap, choose, arm, plan100["design"])
        points, anchors = select_anchors(spec, traces["pulse"], traces["gated"], entry, lost, DESIGN)
        result.append(dict(spec=spec, points=points, anchors=anchors))
    return result


def run(root, root100, plan100, root93, root89, plan93, plan89, report89, source, actor, observer, plan):
    triggers, candidates, anchors, branches, evidence, drift = [], [], [], [], {}, []
    old, design = source["original"], plan["design"]
    ap = e63.c.actor_plan(old)
    for spec101 in plan["units"]:
        spec = spec101["spec"]
        entry, sources = load_unit(root100, plan100, root93, root89, plan93, plan89, report89, source, spec)
        trace, unit = sources["pulse"], spec["unit"]
        pack = e74.learned_pack(actor["models"], unit, actor["prior"])
        choose = lambda row, history: e74.choose(row, history, unit, old, actor["prior"], pack)
        point_data = {}

        def cached(kind, first, last, treatment, shared=None):
            name = f"{c.key(unit)}__f{first:06d}__{treatment}"
            path = root / kind / (name+".npz")
            meta = dict(plan_fingerprint=plan["fingerprint"], unit=unit, kind=kind,
                        first=first, last=last, treatment=treatment)
            data = c.read_cache(path, meta)
            fresh = data is None
            if fresh:
                data = (source_slice(trace, first, last) if treatment == "keep" and plan["reuse_keep"] else
                        collect(observer, trace, entry, first, last, treatment, old, choose, design, shared))
            audit(data, trace, entry, first, last, treatment, old, choose, design)
            if fresh:
                c.save(path, data, meta)
            evidence[path.relative_to(root).as_posix()] = c.file_hash(path)
            if treatment == "keep":
                drift.append(float(np.abs(data["boxes"]-trace["boxes"][first-1:last]).max()))
            return data

        with e63.source_experiment.installed(observer, [], unit) as head:
            if head != source["report"]["native_head_sha256"]:
                raise ValueError("Original SGLA head changed")
            for number, frame in enumerate(spec101["points"], 1):
                actual = {t: cached("points", frame, frame, t) for t in TREATMENTS}
                point_data[frame] = actual
                for key in ("candidate_parents", "candidate_boxes", "features", "crop", "crop_sha256", "layer", "block_calls", "head_calls", "selector_calls"):
                    np.testing.assert_array_equal(actual["keep"][key][0, [0, 2]], actual["skip"][key][0, [0, 2]],
                                                  err_msg=f"Unchanged q0/q2 drift:{key}")
                if not actual["keep"]["scale_active"][0] or actual["skip"]["scale_active"][0]:
                    raise ValueError("Matched frame did not apply/remove the requested pulse")
                ident = dict(cohort=spec["cohort"], sequence=unit["sequence"], seed=unit["seed"], frame=frame)
                truth = np.asarray(entry["ground_truth"][frame], float)
                row = dict(ident)
                for treatment, data in actual.items():
                    row.update({treatment+"_"+k: v for k, v in frame_metrics(data, truth, design).items()})
                    valid = np.isfinite(truth).all() and np.all(truth[2:] > 0)
                    for q in range(4):
                        parent, box = data["candidate_parents"][0, q], data["candidate_boxes"][0, q]
                        features = data["features"][0, q]
                        candidates.append(dict(ident, treatment=treatment, candidate=q,
                            selected=q == int(data["selected"][0]), ppo_selected=q == int(data["ppo_selected"][0]),
                            iou=float(tracking_iou(box, truth)) if valid else None,
                            raw_peak=float(features[e63.source_common.prior.FEATURES.index("raw_peak")]),
                            hann_peak=float(features[e63.source_common.prior.FEATURES.index("hann_peak")]),
                            coverage=scene_geometry(parent, truth, data["image_hw"][0], ap["search_factor"], ap["search_size"])["coverage"] if valid else None,
                            **{f"{label}_{k}": float(values[j]) for label, values in (("parent", parent), ("box", box))
                               for j, k in enumerate(("x", "y", "w", "h"))}))
                row["skip_minus_keep_iou"] = row["skip_iou"]-row["keep_iou"] if row["skip_iou"] is not None else None
                row["scale_refs"] = "|".join(str(int(f)) for f in actual["keep"]["scale_refs"][0] if f >= 0)
                triggers.append(row)
                if number % 100 == 0:
                    print(f"[Stage-101] {c.key(unit)} {number}/{len(spec101['points'])} triggers", flush=True)
            for case in spec101["anchors"]:
                first, last = case["frame"], case["last"]
                ident = dict(cohort=spec["cohort"], sequence=unit["sequence"], seed=unit["seed"], frame=first,
                             last=last, roles="|".join(sorted({r["role"] for r in case["roles"]})))
                anchors.append(dict(ident, targets="|".join(str(r["target_frame"]) for r in case["roles"])))
                row = dict(ident)
                for treatment in TREATMENTS:
                    data = cached("branches", first, last, treatment, point_data[first][treatment])
                    for key in s100.FIELDS:
                        np.testing.assert_array_equal(data[key][0], point_data[first][treatment][key][0],
                                                      err_msg="Shared first frame changed:"+key)
                    row.update({treatment+"_"+k: v for k, v in window_metrics(data, entry, design).items()})
                branches.append(row)
        print(f"[Stage-101] {c.key(unit)} complete", flush=True)
    summary, videos, branch_summary = aggregate(triggers, branches)
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    hashes = {}
    for name, rows in zip(TABLES, (summary, videos, anchors, branches, branch_summary, triggers, candidates)):
        path = out / f"stage101_{name}.csv"
        s86.publish_table(path, rows)
        hashes[path.name] = c.file_hash(path)
    states, count, calls = budget(plan)
    c.publish_identical(out / "stage101_decision.json", io.seal(dict(schema=SCHEMA, status=STATUS,
        plan_fingerprint=plan["fingerprint"], source_stage100_fingerprint=SOURCE100, runtime=plan["runtime"],
        runtime_differences=plan["runtime_differences"], reuse_keep=plan["reuse_keep"],
        matched_trigger_states=states, anchors=count, short_branches=2*count, new_search_calls=calls, new_block_calls=7*calls,
        maximum_keep_source_box_error_px=max(drift) if drift else 0., unchanged_q0_q2_checks_pass=True,
        first_frame_reuse_pass=True, training_updates=0, new_full_trajectories=0, confirmation_inference_calls=0,
        table_sha256=hashes, cache_sha256=evidence)))
    check_results(root)
    print(f"Stage-101: {STATUS}; {states} matched triggers, {count*2} short branches, {calls} new searches")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage76", "stage89", "stage93", "stage100", "output_dir"):
        parser.add_argument(f"--{name}", type=Path, required=name == "output_dir")
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage101")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not all((args.stage76, args.stage89, args.stage93, args.stage100, args.checkpoint)):
        parser.error("Supply Stage-76, 89, 93, 100 and the original checkpoint")
    root76, root89, root93, root100 = [p.expanduser().resolve() for p in
                                     (args.stage76, args.stage89, args.stage93, args.stage100)]
    if any(root == p or root in p.parents or p in root.parents for p in (root76, root89, root93, root100)):
        parser.error("Use a separate Stage-101 directory")
    with s93.writer_lock(root):
        if (root / "analysis/stage101_decision.json").exists():
            check_results(root)
            return
        if not (root / "manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-101 directory")
        compat.check_results(root100)
        plan100 = c.c66.sealed(root100 / "manifest.json", s100.SCHEMA)
        report100 = c.c66.sealed(root100 / "analysis/stage100_decision.json", s100.SCHEMA)
        if report100["fingerprint"] != SOURCE100:
            raise ValueError("Need the frozen completed Stage-100 result")
        plan93, _, plan89, report89, source = s95.s94.load_run(root93, root76, root89)
        actor = io.load_source(Path(source["plan"]["source_input"]))
        if actor["plan"] != source["original"] or actor["models"]["fingerprint"] != source["report"]["source_training_fingerprint"]:
            raise ValueError("Frozen PPO changed")
        import torch

        torch.set_num_threads(1)
        recovery = read_csv(root100 / "analysis/stage100_recovery.csv")
        typed = [dict(r, seed=int(r["seed"]), ppo_failure_frame=int(r["ppo_failure_frame"]),
                      ppo_recovery_frame=int(r["ppo_recovery_frame"]), matched_good_frame=r["matched_good_frame"] or None) for r in recovery]
        units = prepare_units(root100, plan100, root93, root89, plan93, plan89, report89, source, actor,
                              s100.newly_lost_ppo_recoveries(typed))
        if len(units) != 40 or sum(len(u["points"]) for u in units) != 1834:
            raise ValueError("Stage-100 trigger coverage changed")
        runtime = s76.device_runtime(source["original"], torch, args.gpu)
        differences = s76.startup.differences(report100["runtime"], runtime)
        if any(r["field"] not in ("runtime.gpu_uuid", "runtime.tracker.device") for r in differences):
            raise ValueError("Frozen software or weights changed")
        paths = [root100 / "manifest.json", root100 / "analysis/stage100_decision.json", root100 / compat.RECEIPT]
        code = [Path(__file__), Path(__file__).with_name("run_stage101_trigger_replay.sh"), Path(compat.__file__)]
        receipt100 = c.c66.sealed(root100 / compat.RECEIPT, compat.SCHEMA)
        bindings = {str(p): c.file_hash(p) for p in paths} | plan100["source_sha256"] | actor["bindings"]
        bindings.update({str(root100 / "analysis" / p): h for p, h in report100["table_sha256"].items()})
        bindings.update({str(root100 / "evaluation" / p): h for p, h in report100["cache_sha256"].items()})
        plan = io.seal(dict(schema=SCHEMA, role="consumed_development_matched_trigger_diagnostic",
            design=DESIGN, units=units, runtime=runtime, runtime_differences=differences, reuse_keep=not differences,
            source_sha256=bindings, code_sha256={str(p.resolve()): c.file_hash(p) for p in code}
                | plan100["code_sha256"] | receipt100["adapter_code_sha256"]))
        c.publish_identical(root / "manifest.json", plan)
        states, count, calls = budget(plan)
        print(f"Stage-101 frozen: {states} triggers, {count} anchors, {calls} new searches", flush=True)
        frozen = dict(source["original"], fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = s76.startup.checked_runtime(args, frozen, root)
        observer = e63.Capture(native)
        try:
            run(root, root100, plan100, root93, root89, plan93, plan89, report89, source, actor, observer, plan)
        finally:
            observer.handle.remove()


if __name__ == "__main__":
    main()
