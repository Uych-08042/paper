"""Compare one size-input pulse with continued updates at a matched full state."""
from __future__ import annotations

import argparse
from collections import defaultdict
import os
from pathlib import Path

import numpy as np

import stage98_scale_memory_audit as s98
from stage27_uav_common import checked_image, crop_geometry


s97, c, io = s98.s97, s98.c, s98.io
s95, s93, e63, e74, s76, s86 = s97.s95, s97.s93, s97.e63, s97.e74, s97.s76, s97.s86
SCHEMA = "stage99_matched_single_scale_pulse_v1"
STATUS = "COMPLETE_MATCHED_SINGLE_SCALE_PULSE_PILOT"
SOURCE98 = "a63817e102f8ec431c3a6cc3eba2560433e4e172df04e27421ca9091f8e71f30"
ARMS = ("control", "once", "continuous")
TABLES = ("windows", "video_summary", "summary")


def rule_at(mode, frame, anchor):
    if mode not in ARMS or frame < anchor:
        raise ValueError("Invalid pulse schedule")
    return "trusted_scale" if mode == "continuous" or (mode == "once" and frame == anchor) else "control"


def matched_snapshot(source, control, trusted, anchor):
    first = int(control["frame_id"][0])
    np.testing.assert_array_equal(control["frame_id"], trusted["frame_id"])
    if not first <= anchor <= int(control["frame_id"][-1]):
        raise ValueError("Anchor outside its source branch")
    stop = anchor-first
    for key in ("boxes", "candidate_parents", "features", "queries", "memory_write", "selected", "ppo_selected"):
        np.testing.assert_array_equal(control[key][:stop], trusted[key][:stop], err_msg=f"Unmatched pre-anchor state: {key}")
    np.testing.assert_array_equal(control["anchor"], trusted["anchor"])
    np.testing.assert_array_equal(control["anchor"], source["anchor"])
    snapshot = {key: np.concatenate((source[key][:first-1], control[key][:stop])) for key in
                ("boxes", "queries", "memory_write", "selected", "ppo_selected")}
    snapshot["anchor"] = control["anchor"].copy()
    if len(snapshot["boxes"]) != anchor-1:
        raise ValueError("Snapshot includes an incomplete or future prefix")
    return snapshot


def tail(data, first):
    start = first-int(data["frame_id"][0])
    if not 0 <= start < len(data["frame_id"]):
        raise ValueError("Invalid source tail")
    return {key: data[key][start:].copy() for key in s97.FIELDS} | dict(anchor=data["anchor"].copy())


def collect(observer, entry, snapshot, first, last, choose, mode, old, design, image_reader=checked_image):
    ap = e63.c.actor_plan(old)
    observer.initialize(image_reader(ap["root"], entry["frames"][0]), entry["initial_box"])
    history = list(e63.history_at(snapshot, entry["initial_box"], first, old))
    memory, sizes = s97.starting_memory(snapshot, first, design)
    anchor = snapshot["anchor"].copy()
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
            rule = rule_at(mode, frame, first)
            parents, refs = s97.parents_at(history, sizes, frame, rule, ap, design)
            row = choose(s97.probe(observer, image, history, parents, frame, ap), history)
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
            accepted = s97.scale_write(decision, queries)
            changed, changed_crop = s97.scale_changes(history, parents, image.shape[:2], ap)
            rows.append(row | decision | dict(boxes=box, queries=queries, scale_refs=refs,
                scale_active=bool((refs >= 0).any()), scale_written=accepted,
                scale_changed=changed, scale_crop_changed=changed_crop))
            s97.update_sizes(sizes, frame, box, accepted, design)
            history = (history+[box])[-ap["design"]["history"]:]
            observer.rt.tracker.state = box.tolist()
            if (frame-first+1) % 200 == 0:
                print(f"  [Stage-99] {mode} {frame-first+1}/{last-first+1}", flush=True)
    return {key: np.asarray([row[key] for row in rows]) for key in s97.FIELDS} | dict(anchor=anchor)


def audit(data, snapshot, entry, first, last, choose, mode, old, design):
    if set(data) != set(s97.FIELDS) | {"anchor"}:
        raise ValueError("Incomplete pulse continuation")
    np.testing.assert_array_equal(data["frame_id"], np.arange(first, last+1))
    np.testing.assert_array_equal(data["anchor"], snapshot["anchor"])
    e63.source_common.prior.audit_calls(data, (last-first+1, 4))
    if not np.isfinite(data["decode_error"]).all() or np.max(data["decode_error"]) > old["actor_design"]["box_atol"]:
        raise ValueError("Native decode drift")
    ap = e63.c.actor_plan(old)
    history = list(e63.history_at(snapshot, entry["initial_box"], first, old))
    memory, sizes = s97.starting_memory(snapshot, first, design)
    for i, frame in enumerate(range(first, last+1)):
        parents, refs = s97.parents_at(history, sizes, frame, rule_at(mode, frame, first), ap, design)
        np.testing.assert_array_equal(data["candidate_parents"][i], parents)
        np.testing.assert_array_equal(data["history_latest"][i], history[-1])
        np.testing.assert_array_equal(data["scale_refs"][i], refs)
        np.testing.assert_array_equal(data["scale_active"][i], (refs >= 0).any())
        np.testing.assert_array_equal([data["scale_changed"][i], data["scale_crop_changed"][i]],
                                     s97.scale_changes(history, parents, data["image_hw"][i], ap))
        for q, parent in enumerate(parents):
            crop = crop_geometry(parent, data["image_hw"][i], ap["search_factor"], ap["search_size"])[0]
            np.testing.assert_array_equal(data["crop"][i, q], crop)
        row = choose({key: data[key][i].copy() for key in ("features", "candidate_boxes", "candidate_parents")}, history)
        row["history_latest"] = history[-1]
        decision = s93.choose_write(row, data["queries"][i], data["anchor"], memory, frame, "gated", design)
        for key, value in decision.items():
            if key in ("margin", "patch_margin"):
                np.testing.assert_allclose(data[key][i], value, rtol=0, atol=1e-12, equal_nan=True)
            else:
                np.testing.assert_array_equal(data[key][i], value)
        box = row["candidate_boxes"][decision["selected"]]
        np.testing.assert_array_equal(data["boxes"][i], box)
        accepted = s97.scale_write(decision, data["queries"][i])
        np.testing.assert_array_equal(data["scale_written"][i], accepted)
        s97.update_sizes(sizes, frame, box, accepted, design)
        history = (history+[box])[-ap["design"]["history"]:]


def summarize(rows):
    contrasts = (("once", "control"), ("continuous", "once"), ("continuous", "control"))
    grouped, videos, summary = defaultdict(list), [], []
    for row in rows:
        grouped[row["outcome"], row["sequence"]].append(row)
    for (outcome, sequence), events in sorted(grouped.items()):
        for arm, base in contrasts:
            videos.append(dict(outcome=outcome, sequence=sequence, events=len(events), contrast=f"{arm}_minus_{base}",
                **{key: float(np.mean([float(r[f"{arm}_{key}"])-float(r[f"{base}_{key}"]) for r in events]))
                   for key in s97.METRICS}))
    for outcome in ("missed", "retained"):
        events = [r for r in rows if r["outcome"] == outcome]
        if not events:
            continue
        for arm, base in contrasts:
            local = [r for r in videos if r["outcome"] == outcome and r["contrast"] == f"{arm}_minus_{base}"]
            summary.append(dict(outcome=outcome, contrast=f"{arm}_minus_{base}", events=len(events), videos=len(local),
                **{key+"_delta_video_mean": float(np.mean([r[key] for r in local])) for key in s97.METRICS},
                base_aligned_recoveries=sum(r[f"{base}_aligned_recovery"] for r in events),
                arm_aligned_recoveries=sum(r[f"{arm}_aligned_recovery"] for r in events),
                aligned_gained=sum(not r[f"{base}_aligned_recovery"] and r[f"{arm}_aligned_recovery"] for r in events),
                aligned_lost=sum(r[f"{base}_aligned_recovery"] and not r[f"{arm}_aligned_recovery"] for r in events)))
    return videos, summary


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    report = c.c66.sealed(root / "analysis/stage99_decision.json", SCHEMA)
    frames = sum(int(case["event"]["last_audit_frame"])-case["anchor"]+1 for case in plan["cases"])
    calls = frames*(4 if plan["reuse_sources"] else 12)
    if (report["status"] != STATUS or report["plan_fingerprint"] != plan["fingerprint"]
            or report["branches"] != 3*len(plan["cases"]) or len(report["cache_sha256"]) != report["branches"]
            or report["new_search_calls"] != calls or report["new_block_calls"] != 7*calls
            or set(report["table_sha256"]) != {f"stage99_{name}.csv" for name in TABLES}):
        raise ValueError("Incomplete Stage-99 pilot")
    io.verify({str(root / name): sha for name, sha in report["cache_sha256"].items()})
    io.verify({str(root / "analysis" / name): sha for name, sha in report["table_sha256"].items()})
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])


def run(root, root97, root93, root89, plan, plan97, plan93, plan89, report89, source, actor, observer):
    old = source["original"]
    specs = {(r["unit"]["sequence"], r["unit"]["seed"]): r for r in plan89["units"]}
    groups = defaultdict(list)
    for case in plan["cases"]:
        groups[case["event"]["sequence"], int(case["event"]["seed"])].append(case)
    rows, evidence, drift = [], {}, []
    source_windows = {s97.case_key(r): r for r in s98.read_csv(root97 / "analysis/stage97_windows.csv")}
    for key, cases in groups.items():
        spec = specs[key]
        unit = spec["unit"]
        entry, traces = s95.load_unit(root93, root89, plan93, plan89, report89, source, spec)
        pack = e74.learned_pack(actor["models"], unit, actor["prior"])

        def choose(row, history):
            return e74.choose(row, history, unit, old, actor["prior"], pack)

        with e63.source_experiment.installed(observer, [], unit) as head:
            if head != source["report"]["native_head_sha256"]:
                raise ValueError("Native SGLA head changed")
            for case in cases:
                event, first = case["event"], case["anchor"]
                last, name = int(event["last_audit_frame"]), s97.case_key(event)
                originals = {}
                for mode, arm in (("control", "control"), ("continuous", "trusted_scale")):
                    path = root97 / "branches" / f"{name}__{arm}.npz"
                    originals[mode] = s95.s94.required_cache(path, dict(plan_fingerprint=plan97["fingerprint"],
                                                              unit=unit, case=name, arm=arm))
                snapshot = matched_snapshot(traces["gated"], originals["control"], originals["continuous"], first)
                window = dict(event, source_first_frame=event["first_audit_frame"], first_audit_frame=str(first),
                              source_transition=case["transition"])
                row, branches = dict(window), {}
                for mode in ARMS:
                    path = root / "branches" / f"{name}__{mode}.npz"
                    meta = dict(plan_fingerprint=plan["fingerprint"], unit=unit, case=name, anchor=first, mode=mode)
                    data = c.read_cache(path, meta)
                    fresh = data is None
                    if fresh:
                        data = (tail(originals[mode], first) if mode in originals and plan["reuse_sources"] else
                                collect(observer, entry, snapshot, first, last, choose, mode, old, plan["design"]))
                    audit(data, snapshot, entry, first, last, choose, mode, old, plan["design"])
                    if fresh:
                        c.save(path, data, meta)
                    evidence[path.relative_to(root).as_posix()] = c.file_hash(path)
                    branches[mode] = data
                    metrics, _ = s97.describe(data, entry, window, plan["design"])
                    row.update({mode+"_"+k: v for k, v in metrics.items()})
                    if mode in originals:
                        reference = tail(originals[mode], first)
                        drift.append(float(np.abs(data["boxes"]-reference["boxes"]).max()))
                        if plan["reuse_sources"]:
                            original_arm = "control" if mode == "control" else "trusted_scale"
                            expected = source_windows[name][original_arm+"_matched_good_frame"]
                            observed = metrics["matched_good_frame"]
                            if ("" if observed is None else str(observed)) != expected:
                                raise ValueError("Source recovery timing changed")
                for field in s97.FIELDS:
                    np.testing.assert_array_equal(branches["once"][field][0], branches["continuous"][field][0],
                                                  err_msg=f"First intervention differs: {field}")
                rows.append(row)
                print(f"[Stage-99] {name} anchor={first} tail={last-first+1}", flush=True)
    videos, summary = summarize(rows)
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    hashes = {}
    for name, items in zip(TABLES, (rows, videos, summary)):
        path = out / f"stage99_{name}.csv"
        s86.publish_table(path, items)
        hashes[path.name] = c.file_hash(path)
    frames = sum(int(case["event"]["last_audit_frame"])-case["anchor"]+1 for case in plan["cases"])
    calls = frames*(4 if plan["reuse_sources"] else 12)
    c.publish_identical(out / "stage99_decision.json", io.seal(dict(
        schema=SCHEMA, status=STATUS, plan_fingerprint=plan["fingerprint"], source_stage98_fingerprint=SOURCE98,
        runtime=plan["runtime"], runtime_differences=plan["runtime_differences"], reuse_sources=plan["reuse_sources"],
        branches=len(evidence), tail_frames=frames, logical_search_calls=frames*12,
        new_search_calls=calls, new_block_calls=7*calls, maximum_source_tail_box_error_px=max(drift),
        source_correspondence_pass=max(drift) <= old["actor_design"]["box_atol"],
        matched_prefix_checks_pass=True, equal_first_intervention_pass=True,
        training_updates=0, new_full_trajectories=0, confirmation_inference_calls=0,
        cache_sha256=evidence, table_sha256=hashes)))
    check_results(root)
    print(f"Stage-99: {STATUS} ({len(evidence)} branches, {calls} new searches)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage76", "stage89", "stage93", "stage97", "stage98", "output_dir"):
        parser.add_argument(f"--{name}", type=Path, required=name == "output_dir")
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage99")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not all((args.stage76, args.stage89, args.stage93, args.stage97, args.stage98, args.checkpoint)):
        parser.error("Supply Stage-76, 89, 93, 97, 98 and the original checkpoint")
    root93, root76, root89, root97, root98 = [p.expanduser().resolve() for p in
                                            (args.stage93, args.stage76, args.stage89, args.stage97, args.stage98)]
    if any(root == p or root in p.parents or p in root.parents for p in (root93, root76, root89, root97, root98)):
        parser.error("Use a separate Stage-99 output directory")
    with s93.writer_lock(root):
        if (root / "analysis/stage99_decision.json").exists():
            check_results(root)
            return
        if not (root / "manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-99 directory")
        s98.check_results(root98)
        report98 = c.c66.sealed(root98 / "stage98_decision.json", s98.SCHEMA)
        if report98["fingerprint"] != SOURCE98:
            raise ValueError("Need the frozen Stage-98 audit")
        s97.check_results(root97)
        plan97 = c.c66.sealed(root97 / "manifest.json", s97.SCHEMA)
        report97 = c.c66.sealed(root97 / "analysis/stage97_decision.json", s97.SCHEMA)
        plan93, _, plan89, report89, source = s95.s94.load_run(root93, root76, root89)
        by_case = {s97.case_key(event): event for event in plan97["events"]}
        cases = []
        for row in s98.read_csv(root98 / "stage98_cases.csv"):
            anchor = int(row["first_parent_difference"])
            event = by_case[row["case"]]
            if not int(event["first_audit_frame"]) <= anchor <= int(event["ppo_recovery_frame"]):
                raise ValueError("Invalid matched anchor")
            cases.append(dict(event=event, anchor=anchor, transition=row["transition"]))
        if len(cases) != 25 or sum(int(case["event"]["last_audit_frame"])-case["anchor"]+1 for case in cases) != 2879:
            raise ValueError("Frozen intervention cohort changed")
        actor = io.load_source(Path(source["plan"]["source_input"]))
        if actor["plan"] != source["original"] or actor["models"]["fingerprint"] != source["report"]["source_training_fingerprint"]:
            raise ValueError("Frozen PPO changed")
        import torch

        torch.set_num_threads(1)
        runtime = s76.device_runtime(source["original"], torch, args.gpu)
        differences = s76.startup.differences(report97["runtime"], runtime)
        if any(r["field"] not in ("runtime.gpu_uuid", "runtime.tracker.device") for r in differences):
            raise ValueError("Source software or weights changed")
        paths = [root98 / "stage98_decision.json", root98 / "stage98_cases.csv", root97 / "manifest.json",
                 root97 / "analysis/stage97_decision.json"]
        code = [Path(__file__), Path(__file__).with_name("run_stage99_scale_pulse.sh")]
        plan = io.seal(dict(schema=SCHEMA, source_stage98_fingerprint=SOURCE98, design=plan97["design"],
            cases=cases, runtime=runtime, runtime_differences=differences, reuse_sources=not differences,
            source_sha256={str(p): c.file_hash(p) for p in paths} | report98["source_sha256"],
            code_sha256={str(p.resolve()): c.file_hash(p) for p in code} | report98["code_sha256"]))
        c.publish_identical(root / "manifest.json", plan)
        frozen = dict(source["original"], fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = s76.startup.checked_runtime(args, frozen, root)
        observer = e63.Capture(native)
        try:
            run(root, root97, root93, root89, plan, plan97, plan93, plan89, report89, source, actor, observer)
        finally:
            observer.handle.remove()


if __name__ == "__main__":
    main()
