"""Test an own-history scale update in matched Stage-93 continuation windows."""
from __future__ import annotations

import argparse
from collections import defaultdict
import os
from pathlib import Path

import numpy as np

import stage96_geometry_cross as s96
from stage27_uav_common import checked_image, crop_geometry, tracking_iou


s95, c, io, e63, s76, s86 = s96.s95, s96.c, s96.io, s96.e63, s96.s76, s96.s86
s93, e74 = s95.s93, s95.s93.e74
SCHEMA = "stage97_own_trusted_scale_continuation_v1"
STATUS = "COMPLETE_OWN_TRUSTED_SCALE_CONTINUATION_PILOT"
SOURCE96 = "7185c1dc58f0f1695039a3082507e103e0fa212e6f1e12031c81305583a6a28c"
DESIGN = dict(s93.DESIGN, scale_max_refs=5)
ARMS = ("control", "trusted_scale")
FIELDS = s86.FIELDS + s93.FIELDS + ("scale_refs", "scale_active", "scale_written", "scale_changed", "scale_crop_changed")
TABLES = ("windows", "video_summary", "summary", "frames")
METRICS = ("mean_iou", "endpoint_iou", "good_candidate_fraction", "low_iou_fraction",
           "reaches_sustained_good", "unresolved_at_window_end", "aligned_recovery")


def scale_write(decision, queries):
    slot = int(decision["selected"] != decision["ppo_selected"])
    return bool(decision["memory_write"] and np.isfinite(queries[slot]).all())


def starting_memory(trace, first, design):
    appearance, sizes = [], []
    for frame in range(max(1, first - design["lookback"]), first):
        i = frame - 1
        if trace["memory_write"][i]:
            slot = int(trace["selected"][i] != trace["ppo_selected"][i])
            appearance.append((frame, trace["queries"][i, slot].copy()))
            if np.isfinite(trace["queries"][i, slot]).all():
                sizes.append((frame, trace["boxes"][i, 2:].copy()))
    return appearance, sizes


def parents_at(history, sizes, frame, arm, ap, design):
    parents = e63.source_common.prior.history_parents(history, ap["design"])
    if arm not in ARMS:
        raise ValueError("Unknown scale arm")
    if any(f >= frame for f, _ in sizes):
        raise ValueError("Current or future size entered search input")
    recent = [(f, size) for f, size in sizes if frame-f <= design["lookback"]][-design["scale_max_refs"]:]
    refs = []
    if arm == "trusted_scale" and len(recent) >= design["min_refs"]:
        refs = [f for f, _ in recent]
        values = np.asarray([size for _, size in recent], float)
        if values.shape[1:] != (2,) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("Invalid historical sizes")
        size = np.exp(np.median(np.log(values), axis=0))
        for q in (1, 3):
            center = parents[q, :2] + parents[q, 2:] / 2
            parents[q] = np.r_[center-size/2, size]
    return parents, np.asarray(refs + [-1]*(design["scale_max_refs"]-len(refs)), int)


def update_sizes(sizes, frame, box, accepted, design):
    if accepted:
        sizes.append((frame, np.asarray(box[2:], float).copy()))
    sizes[:] = [(f, size) for f, size in sizes if frame-f < design["lookback"]]


def scale_changes(history, parents, image_hw, ap):
    ordinary = e63.source_common.prior.history_parents(history, ap["design"])
    changed = not np.array_equal(ordinary, parents)
    changed_crop = any(not np.array_equal(
        crop_geometry(a, image_hw, ap["search_factor"], ap["search_size"])[0],
        crop_geometry(b, image_hw, ap["search_factor"], ap["search_size"])[0])
        for a, b in zip(ordinary, parents))
    return changed, changed_crop


def probe(observer, image, history, parents, frame, ap):
    views = [e63.actor.observe(observer, image, parent, frame, ap) for parent in parents]
    return dict(frame_id=frame, history_latest=history[-1].copy(), candidate_parents=parents,
        candidate_boxes=np.asarray([v["boxes"] for v in views]),
        features=np.asarray([e63.actor.feature_vector(v, history, ap) for v in views]),
        image_hw=np.asarray(image.shape[:2]), decode_error=max(v["decode_error"] for v in views),
        **{key: np.asarray([v[key] for v in views]) for key in
           ("crop", "crop_sha256", "layer", "block_calls", "head_calls", "selector_calls")})


def collect(observer, entry, source_trace, first, last, choose, arm, old, design, image_reader=checked_image):
    ap = e63.c.actor_plan(old)
    template = image_reader(ap["root"], entry["frames"][0])
    observer.initialize(template, entry["initial_box"])
    history = list(e63.history_at(source_trace, entry["initial_box"], first, old))
    memory, sizes = starting_memory(source_trace, first, design)
    anchor = source_trace["anchor"].copy()
    tokens, rows = [], []

    def capture(_module, _args, output):
        grid = s93.s91.cpu_array(output)
        if grid.ndim != 3 or grid.shape[0] != 1:
            raise ValueError("Expected one patch embedding grid")
        tokens.append(grid[0])

    with observer.rt.net.backbone.patch_embed.register_forward_hook(capture):
        for frame in range(first, last+1):
            image = image_reader(ap["root"], entry["frames"][frame])
            tokens.clear()
            parents, refs = parents_at(history, sizes, frame, arm, ap, design)
            row = choose(probe(observer, image, history, parents, frame, ap), history)
            if len(tokens) != 8 or any(len(t) != (observer.rt.lens_z if k % 2 == 0 else observer.rt.lens_x)
                                       for k, t in enumerate(tokens)):
                raise ValueError("Expected four template/search pairs")
            alt, _ = s93.appearance_pair(row, design)
            queries = np.full((2, len(anchor)), np.nan)
            for slot, q in enumerate((int(row["selected"]), alt)):
                if q >= 0:
                    queries[slot] = s93.pool(image, row["candidate_boxes"][q], parents[q],
                        ap["search_factor"], ap["search_size"], tokens[2*q+1], design)
            decision = s93.choose_write(row, queries, anchor, memory, frame, "gated", design)
            box = row["candidate_boxes"][decision["selected"]].copy()
            accepted = scale_write(decision, queries)
            changed, changed_crop = scale_changes(history, parents, image.shape[:2], ap)
            rows.append(row | decision | dict(boxes=box, queries=queries, scale_refs=refs,
                scale_active=bool((refs >= 0).any()), scale_written=accepted,
                scale_changed=changed, scale_crop_changed=changed_crop))
            update_sizes(sizes, frame, box, accepted, design)
            history = (history + [box])[-ap["design"]["history"]:]
            observer.rt.tracker.state = box.tolist()
            if (frame-first+1) % 200 == 0:
                print(f"  [Stage-97] {arm} {frame-first+1}/{last-first+1}", flush=True)
    return {key: np.asarray([row[key] for row in rows]) for key in FIELDS} | dict(anchor=anchor)


def control_slice(trace, first, last):
    data = {k: trace[k][first-1:last].copy() for k in s86.FIELDS+s93.FIELDS}
    data["anchor"] = trace["anchor"].copy()
    count = last-first+1
    data.update(scale_refs=np.full((count, DESIGN["scale_max_refs"]), -1, int), scale_active=np.zeros(count, bool))
    data.update(scale_changed=np.zeros(count, bool), scale_crop_changed=np.zeros(count, bool))
    data["scale_written"] = np.asarray([scale_write({k: data[k][i] for k in
        ("selected", "ppo_selected", "memory_write")}, data["queries"][i]) for i in range(count)])
    return data


def audit(data, source_trace, entry, first, last, choose, arm, old, design):
    count = last-first+1
    if set(data) != set(FIELDS) | {"anchor"}:
        raise ValueError("Incomplete scale continuation")
    np.testing.assert_array_equal(data["frame_id"], np.arange(first, last+1))
    np.testing.assert_array_equal(data["anchor"], source_trace["anchor"])
    e63.source_common.prior.audit_calls(data, (count, 4))
    if not np.isfinite(data["decode_error"]).all() or np.max(data["decode_error"]) > old["actor_design"]["box_atol"]:
        raise ValueError("Native decode drift")
    ap = e63.c.actor_plan(old)
    history = list(e63.history_at(source_trace, entry["initial_box"], first, old))
    memory, sizes = starting_memory(source_trace, first, design)
    for i, frame in enumerate(range(first, last+1)):
        parents, refs = parents_at(history, sizes, frame, arm, ap, design)
        np.testing.assert_array_equal(data["candidate_parents"][i], parents)
        np.testing.assert_array_equal(data["history_latest"][i], history[-1])
        np.testing.assert_array_equal(data["scale_refs"][i], refs)
        np.testing.assert_array_equal(data["scale_active"][i], (refs >= 0).any())
        changed, changed_crop = scale_changes(history, parents, data["image_hw"][i], ap)
        np.testing.assert_array_equal(data["scale_changed"][i], changed)
        np.testing.assert_array_equal(data["scale_crop_changed"][i], changed_crop)
        for q in range(4):
            crop = crop_geometry(parents[q], data["image_hw"][i], ap["search_factor"], ap["search_size"])[0]
            np.testing.assert_array_equal(data["crop"][i, q], crop)
        row = choose({k: data[k][i].copy() for k in ("features", "candidate_boxes", "candidate_parents")}, history)
        row["history_latest"] = history[-1]
        decision = s93.choose_write(row, data["queries"][i], data["anchor"], memory, frame, "gated", design)
        for key, value in decision.items():
            if key in ("margin", "patch_margin"):
                np.testing.assert_allclose(data[key][i], value, rtol=0, atol=1e-12, equal_nan=True)
            else:
                np.testing.assert_array_equal(data[key][i], value)
        box = row["candidate_boxes"][decision["selected"]]
        np.testing.assert_array_equal(data["boxes"][i], box)
        accepted = scale_write(decision, data["queries"][i])
        np.testing.assert_array_equal(data["scale_written"][i], accepted)
        update_sizes(sizes, frame, box, accepted, design)
        history = (history+[box])[-ap["design"]["history"]:]


def describe(data, entry, event, design):
    first, last, recovery = (int(event[k]) for k in ("first_audit_frame", "last_audit_frame", "ppo_recovery_frame"))
    truth = np.asarray(entry["ground_truth"], float)[first:last+1]
    metrics = s86.metrics(data, truth, design)
    iou = tracking_iou(data["boxes"], truth)
    index = recovery-first
    good = np.lib.stride_tricks.sliding_window_view(iou[index:] >= design["good_iou"], design["recovery_run"]).all(axis=1)
    matches = np.flatnonzero(good)
    metrics.update(aligned_recovery=bool(len(matches)), matched_good_frame=recovery+int(matches[0]) if len(matches) else None,
        scale_active_frames=int(data["scale_active"].sum()), scale_writes=int(data["scale_written"].sum()),
        scale_changed_frames=int(data["scale_changed"].sum()), scale_crop_changed_frames=int(data["scale_crop_changed"].sum()))
    return metrics, iou


def case_key(event):
    return f"{event['sequence']}__s{event['seed']}__r{event['ppo_recovery_frame']}"


def summarize(rows):
    grouped, videos = defaultdict(list), []
    for row in rows:
        grouped[row["outcome"], row["sequence"]].append(row)
    for (outcome, sequence), group in sorted(grouped.items()):
        videos.append(dict(outcome=outcome, sequence=sequence, events=len(group),
            **{key: float(np.mean([float(r["trusted_scale_"+key])-float(r["control_"+key]) for r in group])) for key in METRICS}))
    summaries = []
    for outcome in ("missed", "retained"):
        group = [r for r in rows if r["outcome"] == outcome]
        local = [r for r in videos if r["outcome"] == outcome]
        if not group:
            continue
        summaries.append(dict(outcome=outcome, events=len(group), videos=len(local),
            **{key+"_delta_video_mean": float(np.mean([r[key] for r in local])) for key in METRICS},
            control_aligned_recoveries=sum(r["control_aligned_recovery"] for r in group),
            trusted_scale_aligned_recoveries=sum(r["trusted_scale_aligned_recovery"] for r in group),
            aligned_recoveries_gained=sum(not r["control_aligned_recovery"] and r["trusted_scale_aligned_recovery"] for r in group),
            aligned_recoveries_lost=sum(r["control_aligned_recovery"] and not r["trusted_scale_aligned_recovery"] for r in group)))
    return videos, summaries


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    report = c.c66.sealed(root / "analysis/stage97_decision.json", SCHEMA)
    frames = sum(int(e["last_audit_frame"])-int(e["first_audit_frame"])+1 for e in plan["events"])
    calls = frames*4*(1 if plan["reuse_control"] else 2)
    if (report["status"] != STATUS or report["plan_fingerprint"] != plan["fingerprint"]
            or report["branches"] != 2*len(plan["events"]) or len(report["cache_sha256"]) != report["branches"]
            or report["new_search_calls"] != calls or report["new_block_calls"] != 7*calls
            or set(report["table_sha256"]) != {f"stage97_{name}.csv" for name in TABLES}):
        raise ValueError("Incomplete Stage-97 pilot")
    io.verify({str(root / name): sha for name, sha in report["cache_sha256"].items()})
    io.verify({str(root / "analysis" / name): sha for name, sha in report["table_sha256"].items()})
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])


def run(root, plan, root93, root89, plan93, plan89, report89, source, actor, observer):
    old, design = source["original"], plan["design"]
    specs = {(s["unit"]["sequence"], s["unit"]["seed"]): s for s in plan89["units"]}
    grouped = defaultdict(list)
    for event in plan["events"]:
        grouped[event["sequence"], int(event["seed"])].append(event)
    rows, frames, evidence, drift = [], [], {}, []
    for key, events in grouped.items():
        spec = specs[key]
        unit = spec["unit"]
        entry, traces = s95.load_unit(root93, root89, plan93, plan89, report89, source, spec)
        trace = traces["gated"]
        pack = e74.learned_pack(actor["models"], unit, actor["prior"])

        def choose(row, history):
            return e74.choose(row, history, unit, old, actor["prior"], pack)

        with e63.source_experiment.installed(observer, [], unit) as head:
            if head != source["report"]["native_head_sha256"]:
                raise ValueError("Native SGLA head changed")
            for event in events:
                first, last = int(event["first_audit_frame"]), int(event["last_audit_frame"])
                name = case_key(event)
                row = dict(event)
                for arm in ARMS:
                    path = root / "branches" / (name+"__"+arm+".npz")
                    meta = dict(plan_fingerprint=plan["fingerprint"], unit=unit, case=name, arm=arm)
                    data = c.read_cache(path, meta)
                    fresh = data is None
                    if fresh:
                        data = (control_slice(trace, first, last) if arm == "control" and plan["reuse_control"] else
                                collect(observer, entry, trace, first, last, choose, arm, old, design))
                    audit(data, trace, entry, first, last, choose, arm, old, design)
                    if fresh:
                        c.save(path, data, meta)
                    evidence[path.relative_to(root).as_posix()] = c.file_hash(path)
                    metrics, overlaps = describe(data, entry, event, design)
                    row.update({arm+"_"+k: v for k, v in metrics.items()})
                    if arm == "control":
                        drift.append(float(np.abs(data["boxes"]-trace["boxes"][first-1:last]).max()))
                        expected = int(event["matched_good_frame"]) if event["matched_good_frame"] else None
                        if plan["reuse_control"] and metrics["matched_good_frame"] != expected:
                            raise ValueError("Source recovery timing does not reconstruct")
                    for i, frame in enumerate(range(first, last+1)):
                        frames.append(dict(case=name, sequence=key[0], seed=key[1], outcome=event["outcome"], arm=arm,
                            frame=frame, iou=float(overlaps[i]), scale_active=bool(data["scale_active"][i]),
                            scale_written=bool(data["scale_written"][i]),
                            scale_changed=bool(data["scale_changed"][i]), scale_crop_changed=bool(data["scale_crop_changed"][i]),
                            scale_refs="|".join(str(f) for f in data["scale_refs"][i] if f >= 0),
                            selected=int(data["selected"][i]), ppo_selected=int(data["ppo_selected"][i])))
                rows.append(row)
                print(f"[Stage-97] {name} frames={last-first+1}", flush=True)
    videos, summary = summarize(rows)
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    hashes = {}
    for name, items in zip(TABLES, (rows, videos, summary, frames)):
        path = out / f"stage97_{name}.csv"
        s86.publish_table(path, items)
        hashes[path.name] = c.file_hash(path)
    window_frames = sum(int(e["last_audit_frame"])-int(e["first_audit_frame"])+1 for e in plan["events"])
    calls = window_frames*4*(1 if plan["reuse_control"] else 2)
    c.publish_identical(out / "stage97_decision.json", io.seal(dict(
        schema=SCHEMA, status=STATUS, plan_fingerprint=plan["fingerprint"], source_stage96_fingerprint=SOURCE96,
        runtime=plan["runtime"], runtime_differences=plan["runtime_differences"], reuse_control=plan["reuse_control"],
        branches=len(evidence), window_frames=window_frames, logical_search_calls=window_frames*8,
        new_search_calls=calls, new_block_calls=calls*7, maximum_control_source_box_error_px=max(drift),
        control_source_correspondence_pass=max(drift) <= old["actor_design"]["box_atol"],
        training_updates=0, new_full_trajectories=0, confirmation_inference_calls=0,
        cache_sha256=evidence, table_sha256=hashes)))
    check_results(root)
    print(f"Stage-97: {STATUS} ({len(evidence)} branches, {calls} new searches)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage76", "stage89", "stage93", "stage96", "output_dir"):
        parser.add_argument(f"--{name}", type=Path, required=name == "output_dir")
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage97")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not all((args.stage76, args.stage89, args.stage93, args.stage96, args.checkpoint)):
        parser.error("Supply Stage-76, 89, 93, 96 and the original checkpoint")
    root93, root76, root89, root96 = [p.expanduser().resolve() for p in
                                     (args.stage93, args.stage76, args.stage89, args.stage96)]
    if any(root == p or root in p.parents or p in root.parents for p in (root93, root76, root89, root96)):
        parser.error("Use a separate Stage-97 output directory")
    with s93.writer_lock(root):
        if (root / "analysis/stage97_decision.json").exists():
            check_results(root)
            return
        if not (root / "manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-97 directory")
        s96.check_results(root96)
        plan96 = c.c66.sealed(root96 / "manifest.json", s96.SCHEMA)
        report96 = c.c66.sealed(root96 / "analysis/stage96_decision.json", s96.SCHEMA)
        if report96["fingerprint"] != SOURCE96:
            raise ValueError("Need the frozen Stage-96 result")
        plan93, report93, plan89, report89, source = s95.s94.load_run(root93, root76, root89)
        # The fixed recovery windows live in the Stage-95 manifest, whose hash is bound by Stage-96.
        reports95 = [Path(p) for p in plan96["source_sha256"] if Path(p).name == "stage95_decision.json"]
        if len(reports95) != 1:
            raise ValueError("Ambiguous Stage-95 source")
        root95 = reports95[0].parent.parent
        prior95 = c.c66.sealed(root95 / "manifest.json", s95.SCHEMA)
        events = prior95["events"]
        if len(events) != 25 or sum(int(e["last_audit_frame"])-int(e["first_audit_frame"])+1 for e in events) != 3162:
            raise ValueError("Fixed continuation windows changed")
        actor = io.load_source(Path(source["plan"]["source_input"]))
        if actor["plan"] != source["original"] or actor["models"]["fingerprint"] != source["report"]["source_training_fingerprint"]:
            raise ValueError("Frozen PPO source changed")
        import torch

        torch.set_num_threads(1)
        runtime = s76.device_runtime(source["original"], torch, args.gpu)
        differences = s76.startup.differences(report93["runtime"], runtime)
        if any(r["field"] not in ("runtime.gpu_uuid", "runtime.tracker.device") for r in differences):
            raise ValueError("Source software or weights changed")
        code = [Path(__file__), Path(__file__).with_name("run_stage97_trusted_scale.sh"),
                Path(s96.__file__), Path(s95.__file__), Path(s93.__file__), Path(s93.s92.__file__), Path(e74.__file__)]
        paths = [root96 / "manifest.json", root96 / "analysis/stage96_decision.json", root95 / "manifest.json"]
        plan = io.seal(dict(schema=SCHEMA, design=DESIGN, events=events, runtime=runtime, runtime_differences=differences,
            reuse_control=not differences, source_stage96_fingerprint=SOURCE96,
            source_sha256={str(p): c.file_hash(p) for p in paths} | plan96["source_sha256"] | actor["bindings"],
            code_sha256={str(p.resolve()): c.file_hash(p) for p in code} | plan96["code_sha256"]))
        c.publish_identical(root / "manifest.json", plan)
        frozen = dict(source["original"], fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = s76.startup.checked_runtime(args, frozen, root)
        observer = e63.Capture(native)
        try:
            run(root, plan, root93, root89, plan93, plan89, report89, source, actor, observer)
        finally:
            observer.handle.remove()


if __name__ == "__main__":
    main()
