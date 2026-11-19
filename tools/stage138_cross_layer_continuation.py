"""Compare native and cached cross-layer anchor boxes through native SGLA continuation."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import stage137_shared_crop_layers as probe
from stage135_peak_gate import SOURCE134
from stage27_uav_common import checked_image, tracking_iou


prior, base, c, io = probe.prior, probe.base, probe.c, probe.io
SCHEMA = "stage138_cross_layer_continuation_v1"
STATUS = "COMPLETE_CROSS_LAYER_CONTINUATION_DIAGNOSTIC"
SOURCE137 = "a092d092f5fc2ff684f888c243197eefe6a8c70f430268d820c8475d2c38e89c"
HORIZON = 32
ARMS = ("native", "peak_pool_oracle", "dense_oracle", "cross_layer_top1_oracle", "cross_layer_max_hann")
TABLES = ("summary", "windows", "scope", "timeline")


def freeze_cases(plan137, rows137, rows134):
    pool = {r["case_id"]: float(r["anchor_iou"]) for r in rows134 if r["arm"] == ARMS[1]}
    by_case = {r["case_id"]: r for r in rows137}
    cases = []
    for case in plan137["cases"]:
        row = by_case[case["case_id"]]
        kind = case["kind"]
        extra_top1 = row["adds_good_standard"] == "True" and pool[case["case_id"]] < probe.GOOD
        harms_good = row["peak_choice_loses_good"] == "True"
        if (kind == "onset" and extra_top1) or (kind == "healthy" and harms_good):
            if (row["sequence"] != case["sequence"] or int(row["frame"]) != case["frame"]
                    or case["successors"] != HORIZON):
                raise ValueError("Stage-137 matched case differs")
            cases.append(case)
    if (len({r["case_id"] for r in cases}) != 6
            or [sum(r["kind"] == kind for r in cases) for kind in ("onset", "healthy")] != [3, 3]):
        raise ValueError("Need three cross-layer gains and three healthy harm controls")
    return cases


def anchor_boxes(data, case, reference, truth, ap):
    row, _ = probe.score_case(data, case, reference, truth, ap, same_runtime=True)
    native = row["native_layer"]
    original = reference["boxes"][[prior.ARMS.index(arm) for arm in ARMS[:3]], 0]
    best = row["best_other_layer"] if row["best_other_top1_iou"] > row["native_top1_iou"] else native
    peak = row["peak_choice_layer"]
    by_layer = dict(zip(probe.LAYERS, data["boxes"]))
    by_layer[native] = original[0]
    boxes = np.vstack((original, by_layer[best], by_layer[peak]))
    ious = tracking_iou(boxes, truth)
    if case["kind"] == "onset":
        if not (ious[0] < .20 and ious[1] < probe.GOOD <= ious[3]):
            raise ValueError("Cross-layer anchor gain no longer matches source")
    elif not ious[4] < probe.GOOD <= ious[0]:
        raise ValueError("Stable anchor harm no longer matches source")
    return boxes, np.array([native, native, native, best, peak], int)


def collect(observer, entry, case, anchors, ap, image_reader=checked_image, observe=base.e63.actor.observe):
    first = case["frame"]
    ids = np.arange(first, first + case["successors"] + 1)
    frames = entry["frames"]
    observer.initialize(image_reader(ap["root"], frames[0]), entry["initial_box"])
    boxes = np.empty((len(ARMS), len(ids), 4), float)
    boxes[:, 0] = anchors
    steps = [[] for _ in ARMS]
    for j, frame in enumerate(ids[1:], 1):
        image = image_reader(ap["root"], frames[frame])
        for k, branch in enumerate(steps):
            observed = observe(observer, image, boxes[k, j - 1].copy(), int(frame), ap)
            boxes[k, j] = observed["boxes"]
            branch.append(observed)
    fields = ("parent", "crop", "crop_sha256", "block_calls", "head_calls", "selector_calls", "layer")
    return dict(boxes=boxes, frame_id=ids, **{key: np.asarray([[r[key] for r in branch] for branch in steps]) for key in fields})


def verify_trace(trace, case, anchors, reference, same_runtime, atol):
    first, tail = case["frame"], case["successors"]
    np.testing.assert_array_equal(trace["frame_id"], np.arange(first, first + tail + 1))
    boxes = trace["boxes"]
    if boxes.shape != (len(ARMS), tail + 1, 4) or not np.isfinite(boxes).all() or (boxes[..., 2:] <= 0).any():
        raise ValueError("Invalid Stage-138 boxes")
    np.testing.assert_array_equal(boxes[:, 0], anchors)
    np.testing.assert_array_equal(trace["parent"], boxes[:, :-1])
    base.e63.source_common.prior.audit_calls(trace, (len(ARMS), tail))
    for k in range(1, len(ARMS)):
        for j in range(k):
            if np.array_equal(anchors[j], anchors[k]):
                np.testing.assert_allclose(boxes[j], boxes[k], rtol=0, atol=atol)
    slots = [prior.ARMS.index(arm) for arm in ARMS[:3]]
    cached = reference["boxes"][slots]
    error = float(np.abs(boxes[:3] - cached).max())
    if same_runtime:
        np.testing.assert_allclose(boxes[:3], cached, rtol=0, atol=atol)
        np.testing.assert_array_equal(trace["layer"][:3], reference["layer"][slots, 1:])
    return error


def describe(trace, case, truth, layers):
    ious = tracking_iou(trace["boxes"], np.asarray(truth, float)[trace["frame_id"]][None])
    tails = ious[:, 1:]
    good = np.lib.stride_tricks.sliding_window_view(tails >= probe.GOOD, 3, axis=1).all(-1)
    low = (tails < .2).sum(axis=1)
    rows, timeline = [], []
    for k, arm in enumerate(ARMS):
        indices = np.flatnonzero(good[k])
        end = float(tails[k, -1])
        rows.append(dict(case, arm=arm, anchor_layer=int(layers[k]), anchor_iou=float(ious[k, 0]),
                         live_native_low_anchor=bool(ious[0, 0] < .20), anchor_delta=float(ious[k, 0] - ious[0, 0]),
                         active=not np.array_equal(trace["boxes"][k, 0], trace["boxes"][0, 0]),
                         first_tail_good_run=int(trace["frame_id"][indices[0] + 1]) if indices.size else None,
                         new_tail_good_run=bool(good[k].any() and not good[0].any()),
                         lost_tail_good_run=bool(good[0].any() and not good[k].any()),
                         tail_low_frames=int(low[k]), tail_low_frame_delta=int(low[k] - low[0]),
                         new_tail_low=bool(low[k] > 0 and low[0] == 0),
                         tail_mean_iou=float(tails[k].mean()), tail_end_iou=end,
                         tail_mean_delta=float((tails[k] - tails[0]).mean()),
                         tail_delta_vs_peak_pool=float((tails[k] - tails[1]).mean()),
                         tail_delta_vs_dense=float((tails[k] - tails[2]).mean())))
        timeline.extend(dict(case_id=case["case_id"], sequence=case["sequence"], arm=arm,
                             frame=int(frame), offset=j, iou=float(ious[k, j]))
                        for j, frame in enumerate(trace["frame_id"]))
    return rows, timeline


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    result = c.c66.sealed(root / "stage138_decision.json", SCHEMA)
    if (result["status"] != STATUS or result["plan_fingerprint"] != plan["fingerprint"]
            or result["source_stage137"] != SOURCE137 or result["new_successor_searches"] != plan["budget"]
            or result["cases"] != len(plan["cases"]) or len(result["artifact_sha256"]) != 2 * len(plan["cases"])
            or set(result["table_sha256"]) != {f"stage138_{n}.csv" for n in TABLES}):
        raise ValueError("Incomplete Stage-138 continuation")
    io.verify({str(root / k): h for k, h in (result["table_sha256"] | result["artifact_sha256"]).items()})


def run(root, p134, p137, r134, r137, plan, observer, ap, load_entry):
    windows, traces, scope, evidence = [], [], [], {}
    out = root / "branches"
    out.mkdir(exist_ok=True)
    for case in plan["cases"]:
        entry = load_entry(case["sequence"])
        cid, frame = case["case_id"], case["frame"]
        ref = c.read_cache(r134 / "branches" / f"{cid}.npz", dict(plan_fingerprint=p134["fingerprint"], case_id=cid))
        observed = c.read_cache(r137 / "observations" / f"{cid}.npz", dict(plan_fingerprint=p137["fingerprint"], case_id=cid))
        if ref is None or observed is None or int(ref["frame_id"][0]) != frame or entry["sequence"] != case["sequence"]:
            raise ValueError(f"Missing or mismatched Stage-134/137 anchor: {cid}")
        anchors, layers = anchor_boxes(observed, case, ref, entry["ground_truth"][frame], ap)
        path, meta = out / f"{cid}.npz", dict(plan_fingerprint=plan["fingerprint"], case_id=cid)
        trace = c.read_cache(path, meta)
        computed = trace is None
        if computed:
            trace = collect(observer, entry, case, anchors, ap)
        error = verify_trace(trace, case, anchors, ref, not plan["runtime_differences"], ap["design"]["box_atol"])
        if computed:
            c.save(path, trace, meta)
        for item in (path, path.with_suffix(".receipt.json")):
            evidence[item.relative_to(root).as_posix()] = c.file_hash(item)
        rows, points = describe(trace, case, entry["ground_truth"], layers)
        windows.extend(rows)
        traces.extend(points)
        scope.append(dict(case, source_control_max_box_error=error, successor_searches=int(trace["head_calls"].sum())))
        print(f"[Stage-138] {cid}: {case['successors']} successors in each of {len(ARMS)} branches", flush=True)
    searches = sum(r["successor_searches"] for r in scope)
    if searches != plan["budget"]:
        raise ValueError("Stage-138 search budget changed")
    summary = [r for r in prior.summarize(windows) if r["kind"] != "onset_live_low"]
    for row in summary:
        grouped = [r for r in windows if (r["kind"], r["arm"]) == (row["kind"], row["arm"])]
        for key in ("tail_delta_vs_peak_pool", "tail_delta_vs_dense"):
            row["mean_" + key] = float(np.mean([r[key] for r in grouped]))
    tables = {}
    for name, rows in zip(TABLES, (summary, windows, scope, traces)):
        path = root / f"stage138_{name}.csv"
        prior.publish_table(path, rows)
        tables[path.name] = c.file_hash(path)
    c.publish_identical(root / "stage138_decision.json", io.seal(dict(
        schema=SCHEMA, status=STATUS, role="selected_development_cross_layer_one_box_intervention",
        plan_fingerprint=plan["fingerprint"], source_stage137=SOURCE137, source_stage134=SOURCE134,
        runtime=plan["runtime"], runtime_differences=plan["runtime_differences"], cases=len(scope),
        branches=len(windows), new_anchor_searches=0, new_successor_searches=searches,
        training_updates=0, new_full_trajectories=0, confirmation_inference_calls=0,
        table_sha256=tables, artifact_sha256=evidence)))
    check_results(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76", "stage89", "stage93", "stage134", "stage137")
    for name in names:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage138", parameter="deit_distilled")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not args.checkpoint or any(not getattr(args, k) for k in names):
        parser.error("Supply the frozen source directories and original checkpoint")
    r76, r89, r93, r134, r137 = paths = [getattr(args, k).expanduser().resolve() for k in names]
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate Stage-138 directory")
    if not (r137 / "stage137_decision.json").is_file():
        parser.error("Need the completed Stage-137 server caches")
    with prior.writer_lock(root):
        if (root / "stage138_decision.json").is_file():
            check_results(root)
            return
        prior.check_results(r134)
        probe.check_results(r137)
        p134 = c.c66.sealed(r134 / "manifest.json", prior.SCHEMA)
        p137 = c.c66.sealed(r137 / "manifest.json", probe.SCHEMA)
        d134 = c.c66.sealed(r134 / "stage134_decision.json", prior.SCHEMA)
        d137 = c.c66.sealed(r137 / "stage137_decision.json", probe.SCHEMA)
        if (d137["fingerprint"] != SOURCE137 or d134["fingerprint"] != SOURCE134
                or d137["source_stage134"] != d134["fingerprint"] or p137["runtime"] != p134["runtime"]):
            raise ValueError("Need the matched Stage-134 and Stage-137 archive")
        cases = freeze_cases(p137, prior.read_table(r137 / "stage137_cases.csv"), prior.read_table(r134 / "stage134_windows.csv"))
        s95 = base.s103.prior.s95
        p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
        old = origin["original"]
        if p137["source_roster_fingerprints"] != [p93["fingerprint"], p89["fingerprint"], old["fingerprint"]]:
            raise ValueError("Stage-137 source roster changed")
        specs = {s["unit"]["sequence"]: s for s in p89["units"]}

        def load_entry(seq):
            return s95.load_unit(r93, r89, p93, p89, d89, origin, specs[seq])[0]

        import torch
        torch.set_num_threads(1)
        runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
        differences = base.s109.check_runtime(p134["runtime"], runtime)
        plan = io.seal(dict(schema=SCHEMA, cases=cases, arms=list(ARMS), source_stage137=SOURCE137,
                            source_stage134=d134["fingerprint"], runtime=runtime, runtime_differences=differences,
                            budget=sum(r["successors"] for r in cases) * len(ARMS), code_sha256=c.file_hash(Path(__file__))))
        manifest = root / "manifest.json"
        if manifest.is_file() and c.c66.sealed(manifest, SCHEMA)["runtime"] != runtime:
            raise ValueError("GPU/runtime changed on resume. Set STAGE138_ROOT to a new directory.")
        c.publish_identical(manifest, plan)
        frozen = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = base.s103.s76.startup.checked_runtime(args, frozen, root)
        observer = base.e63.Capture(native)
        try:
            with base.e63.source_experiment.installed(observer, [], {"origin": "native"}) as head:
                if head != p134["native_head_sha256"]:
                    raise ValueError("Frozen native head differs")
                run(root, p134, p137, r134, r137, plan, observer, base.e63.c.actor_plan(old), load_entry)
        finally:
            observer.handle.remove()
    print(f"Stage-138: {STATUS}", flush=True)


if __name__ == "__main__":
    main()
