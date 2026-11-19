"""Audit crop support and replay native responses on frozen Stage-138 branch states."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import stage138_cross_layer_continuation as source
from stage27_uav_common import checked_image, scene_geometry, tracking_iou
from stage28_response_common import dense_boxes, peak_candidates


prior, base, c, io = source.prior, source.base, source.c, source.io
SCHEMA = "stage139_post_write_diagnostic_v1"
STATUS = "COMPLETE_POST_WRITE_CROP_RESPONSE_DIAGNOSTIC"
SOURCE138 = "96ab86bf173e52b6ea02b5e04408155104ef7db061487dade40843e7faa1ffb9"
DESIGN = dict(early_frames=8, good_iou=.50, low_iou=.20, min_coverage=.90, consecutive_frames=3)
ARMS = source.ARMS
RELAPSE_CASE = ("uav0000089_00920_s", 1159)
RELAPSE_ARM = "cross_layer_top1_oracle"
TABLES = ("mechanism_summary", "replay", "coverage", "scope")
MECHANISMS = ("output_good", "good_grid_not_selected", "limited_crop_no_good_grid", "covered_no_good_grid")


def relapse_offset(tail):
    width = DESIGN["consecutive_frames"]
    runs = np.lib.stride_tricks.sliding_window_view(np.asarray(tail, float), width)
    good = np.flatnonzero((runs >= DESIGN["good_iou"]).all(axis=1))
    if good.size:
        low = np.flatnonzero((runs < DESIGN["low_iou"]).all(axis=1))
        later = low[low >= good[0] + width]
        if later.size:
            return int(later[0] + 1)
    return None


def replay_plan(cases, timeline):
    indexed = defaultdict(dict)
    for row in timeline:
        key = (row["case_id"], row["arm"])
        offset = int(row["offset"])
        if offset in indexed[key]:
            raise ValueError("Duplicate frozen Stage-138 IoU")
        indexed[key][offset] = float(row["iou"])
    result = []
    for case in cases:
        last = case["successors"]
        if last != 32 or any(set(indexed[case["case_id"], arm]) != set(range(last + 1)) for arm in ARMS):
            raise ValueError("Missing complete Stage-138 trajectory")
        relapse = None
        if (case["sequence"], case["frame"]) == RELAPSE_CASE:
            trace = indexed[case["case_id"], RELAPSE_ARM]
            relapse = relapse_offset([trace[j] for j in range(1, last + 1)])
            if relapse is None:
                raise ValueError("The frozen 0089 relapse could not be located")
        around = [j for j in range(relapse - 1, relapse + 2) if 1 <= j <= last] if relapse is not None else []
        offsets = sorted(set(range(1, DESIGN["early_frames"] + 1)) | set(around))
        result.append(dict(case, offsets=offsets, relapse_offset=relapse, relapse_offsets=around))
    if len(result) != 6 or sum(r["relapse_offset"] is not None for r in result) != 1:
        raise ValueError("Stage-138 case roster differs")
    return result, indexed


def collect(observer, entry, case, trace, ap, image_reader=checked_image, observe=base.e63.actor.observe):
    frames = entry["frames"]
    observer.initialize(image_reader(ap["root"], frames[0]), entry["initial_box"])
    rows, executed = [], []
    for offset in case["offsets"]:
        frame = case["frame"] + offset
        image = image_reader(ap["root"], frames[frame])
        observed, arms, calls = {}, [], []
        for parent in trace["parent"][:, offset - 1]:
            key = tuple(parent)
            calls.append(key not in observed)
            if calls[-1]:
                observed[key] = observe(observer, image, parent, frame, ap)
            arms.append(observed[key])
        rows.append(arms)
        executed.append(calls)
    fields = ("boxes", "parent", "crop", "crop_sha256", "raw", "size", "offset", "factor", "image_hw",
              "head_calls", "selector_calls", "block_calls", "layer")
    arrays = {k: np.asarray([[row[k] for row in arms] for arms in rows]) for k in fields}
    return dict(arrays, executed=np.asarray(executed), offsets=np.asarray(case["offsets"]))


def verify_replay(data, case, trace, same_runtime, atol):
    offsets = np.asarray(case["offsets"])
    np.testing.assert_array_equal(data["offsets"], offsets)
    indices = offsets - 1
    for k in ("parent", "crop", "crop_sha256"):
        np.testing.assert_array_equal(data[k], np.swapaxes(trace[k][:, indices], 0, 1), err_msg=k)
    base.e63.source_common.prior.audit_calls(data, (len(offsets), len(ARMS)))
    live, archived = data["boxes"], np.swapaxes(trace["boxes"][:, offsets], 0, 1)
    if live.shape != archived.shape or not np.isfinite(live).all() or (live[..., 2:] <= 0).any():
        raise ValueError("Invalid replayed box")
    executed = np.array([[tuple(parent) not in {tuple(p) for p in parents[:i]}
                          for i, parent in enumerate(parents)] for parents in data["parent"]])
    np.testing.assert_array_equal(data["executed"], executed)
    error = float(np.abs(live - archived).max())
    if same_runtime:
        np.testing.assert_allclose(live, archived, rtol=0, atol=atol)
        np.testing.assert_array_equal(data["layer"], np.swapaxes(trace["layer"][:, indices], 0, 1))
    return error, int((data["head_calls"] * executed).sum())


def diagnose(raw, size, offset, parent, factor, hw, box, gt, geometry, ap):
    weighted = raw * np.asarray(ap["window"], np.float32)
    cell = int(weighted.argmax())
    grid = dense_boxes(size, offset, parent, factor, ap["search_size"], hw)
    np.testing.assert_allclose(grid[cell], box, rtol=0, atol=ap["design"]["box_atol"])
    grid[cell] = box
    ious = tracking_iou(grid, gt)
    limit = DESIGN["good_iou"]
    best = float(ious.max())
    peaks = set(peak_candidates(raw, grid)) | set(peak_candidates(weighted, grid))
    covered = geometry["center_in_usable"] and geometry["coverage"] >= DESIGN["min_coverage"]
    if ious[cell] >= limit:
        mechanism = MECHANISMS[0]
    elif best >= limit:
        mechanism = MECHANISMS[1]
    else:
        mechanism = MECHANISMS[3] if covered else MECHANISMS[2]
    return dict(mechanism=mechanism, replay_iou=float(ious[cell]), dense_best_iou=best,
                peak_pool_best_iou=float(ious[sorted(peaks)].max()), hann_cell=cell,
                hann_peak=float(weighted.flat[cell]), dense_good_cells=int((ious >= limit).sum()))


def score_case(data, trace, entry, case, ap, indexed):
    gt = np.asarray(entry["ground_truth"], float)
    rows, coverage = [], []
    first = case["frame"]
    for k, arm in enumerate(ARMS):
        ident = dict(case_id=case["case_id"], sequence=case["sequence"], kind=case["kind"], arm=arm)
        all_coverage, all_centers, all_sizes = [], [], []
        for j, parent in enumerate(trace["parent"][k], 1):
            frame = first + j
            item = entry["frames"][frame]
            hw = (item["height"], item["width"])
            geometry = scene_geometry(parent, gt[frame], hw, ap["search_factor"], ap["search_size"])
            np.testing.assert_array_equal(geometry["crop"], trace["crop"][k, j - 1])
            all_coverage.append(float(geometry["coverage"]))
            all_centers.append(bool(geometry["center_in_usable"]))
            all_sizes.append(float(min(geometry["input_wh"])))
        all_iou = np.asarray([indexed[case["case_id"], arm][j] for j in range(1, case["successors"] + 1)])
        np.testing.assert_allclose(all_iou, tracking_iou(trace["boxes"][k, 1:], gt[first+1:first+1+len(all_iou)]),
                                   rtol=0, atol=1e-12)
        low = all_iou < DESIGN["low_iou"]
        covered = np.asarray(all_centers) & (np.asarray(all_coverage) >= DESIGN["min_coverage"])
        coverage.append(dict(ident, tail_frames=len(low), low_frames=int(low.sum()),
                             center_outside_frames=all_centers.count(False), low_adequately_covered=int((low & covered).sum()),
                             low_limited_coverage=int((low & ~covered).sum()),
                             minimum_coverage=min(all_coverage), minimum_input_target_short_side=min(all_sizes)))
        for z, j in enumerate(case["offsets"]):
            frame = first + j
            hw = data["image_hw"][z, k]
            parent = data["parent"][z, k]
            geometry = scene_geometry(parent, gt[frame], hw, ap["search_factor"], ap["search_size"])
            np.testing.assert_allclose(float(data["factor"][z, k]), geometry["factor"], atol=1e-12, rtol=0)
            raw, size, shift = (data[key][z, k] for key in ("raw", "size", "offset"))
            result = diagnose(raw, size, shift, parent, geometry["factor"], hw, data["boxes"][z, k], gt[frame], geometry, ap)
            rows.append(dict(ident, frame=frame, offset=j, in_early=j <= DESIGN["early_frames"],
                             in_relapse_neighborhood=j in case["relapse_offsets"], coverage=all_coverage[j-1],
                             target_center_inside=all_centers[j-1], input_target_short_side=all_sizes[j-1],
                             archived_iou=float(all_iou[j-1]), layer=int(data["layer"][z, k]), **result))
    return rows, coverage


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["in_early"]:
            groups["early", row["kind"], row["arm"]].append(row)
        if row["in_relapse_neighborhood"]:
            groups["relapse", row["kind"], row["arm"]].append(row)
    summary = []
    for (phase, kind, arm), subset in sorted(groups.items()):
        counts = Counter(r["mechanism"] for r in subset)
        summary.append(dict(phase=phase, kind=kind, arm=arm, frames=len(subset),
                            low_iou=sum(r["replay_iou"] < DESIGN["low_iou"] for r in subset),
                            mean_replay_iou=float(np.mean([r["replay_iou"] for r in subset])),
                            **{label: counts[label] for label in MECHANISMS}))
    return summary


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    report = c.c66.sealed(root / "stage139_decision.json", SCHEMA)
    if (report["status"] != STATUS or report["plan_fingerprint"] != plan["fingerprint"]
            or report["source_stage138"] != SOURCE138 or report["logical_replays"] != plan["logical_budget"]
            or not 0 < report["new_search_calls"] <= plan["logical_budget"]
            or len(report["artifact_sha256"]) != 2 * len(plan["cases"])
            or set(report["table_sha256"]) != {f"stage139_{k}.csv" for k in TABLES}):
        raise ValueError("Incomplete Stage-139 diagnostic")
    io.verify({str(root / k): h for k, h in (report["table_sha256"] | report["artifact_sha256"]).items()})


def run(root, plan, p138, r138, indexed, observer, ap, load_entry):
    replay, coverage, scope, evidence = [], [], [], {}
    out = root / "replays"
    out.mkdir(exist_ok=True)
    for case in plan["cases"]:
        cid = case["case_id"]
        entry = load_entry(case["sequence"])
        trace = c.read_cache(r138 / "branches" / f"{cid}.npz", dict(plan_fingerprint=p138["fingerprint"], case_id=cid))
        if (trace is None or entry["sequence"] != case["sequence"]
                or int(trace["frame_id"][0]) != case["frame"]):
            raise ValueError(f"Missing or mismatched Stage-138 trace: {cid}")
        path, key = out / f"{cid}.npz", dict(plan_fingerprint=plan["fingerprint"], case_id=cid)
        data = c.read_cache(path, key)
        computed = data is None
        if computed:
            data = collect(observer, entry, case, trace, ap)
        error, calls = verify_replay(data, case, trace, not plan["runtime_differences"], ap["design"]["box_atol"])
        rows, geometry = score_case(data, trace, entry, case, ap, indexed)
        if computed:
            c.save(path, data, key)
        for item in (path, path.with_suffix(".receipt.json")):
            evidence[item.relative_to(root).as_posix()] = c.file_hash(item)
        replay.extend(rows)
        coverage.extend(geometry)
        scope.append(dict(case_id=cid, sequence=case["sequence"], kind=case["kind"], logical_replays=len(rows),
                          physical_replays=calls, archived_max_box_error=error, relapse_offset=case["relapse_offset"]))
        print(f"[Stage-139] {cid}: {calls} searches for {len(rows)} matched branch frames", flush=True)
    tables = {}
    for name, rows in zip(TABLES, (summarize(replay), replay, coverage, scope)):
        path = root / f"stage139_{name}.csv"
        prior.publish_table(path, rows)
        tables[path.name] = c.file_hash(path)
    c.publish_identical(root / "stage139_decision.json", io.seal(dict(
        schema=SCHEMA, status=STATUS, role="consumed_development_fixed_parent_response_diagnostic",
        plan_fingerprint=plan["fingerprint"], source_stage138=SOURCE138, design=DESIGN,
        runtime=plan["runtime"], runtime_differences=plan["runtime_differences"], cases=len(scope),
        logical_replays=len(replay), new_search_calls=sum(r["physical_replays"] for r in scope),
        geometry_branch_frames=sum(r["tail_frames"] for r in coverage), state_updates=0, training_updates=0,
        confirmation_inference_calls=0, table_sha256=tables, artifact_sha256=evidence)))
    check_results(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76", "stage89", "stage93", "stage138")
    for name in names:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage139", parameter="deit_distilled")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not args.checkpoint or any(not getattr(args, k) for k in names):
        parser.error("Supply the frozen source directories and original checkpoint")
    r76, r89, r93, r138 = paths = [getattr(args, k).expanduser().resolve() for k in names]
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate Stage-139 directory")
    if not (r138 / "stage138_decision.json").is_file():
        parser.error("Need the completed Stage-138 server run")
    with prior.writer_lock(root):
        if (root / "stage139_decision.json").is_file():
            check_results(root)
            return
        source.check_results(r138)
        p138 = c.c66.sealed(r138 / "manifest.json", source.SCHEMA)
        d138 = c.c66.sealed(r138 / "stage138_decision.json", source.SCHEMA)
        if d138["fingerprint"] != SOURCE138 or p138["arms"] != list(ARMS):
            raise ValueError("Need the archived Stage-138 result")
        cases, indexed = replay_plan(p138["cases"], prior.read_table(r138 / "stage138_timeline.csv"))
        s95 = base.s103.prior.s95
        p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
        old = origin["original"]
        specs = {s["unit"]["sequence"]: s for s in p89["units"]}

        def load_entry(seq):
            return s95.load_unit(r93, r89, p93, p89, d89, origin, specs[seq])[0]

        import torch
        torch.set_num_threads(1)
        runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
        differences = base.s109.check_runtime(p138["runtime"], runtime)
        plan = io.seal(dict(schema=SCHEMA, cases=cases, design=DESIGN, source_stage138=SOURCE138,
                            runtime=runtime, runtime_differences=differences,
                            logical_budget=sum(len(r["offsets"]) for r in cases) * len(ARMS),
                            code_sha256=c.file_hash(Path(__file__))))
        manifest = root / "manifest.json"
        if manifest.is_file() and c.c66.sealed(manifest, SCHEMA)["runtime"] != runtime:
            raise ValueError("GPU/runtime changed on resume. Set STAGE139_ROOT to a new directory.")
        c.publish_identical(manifest, plan)
        frozen = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = base.s103.s76.startup.checked_runtime(args, frozen, root)
        observer = base.e63.Capture(native)
        try:
            observer.assert_frozen()
            run(root, plan, p138, r138, indexed, observer, base.e63.c.actor_plan(old), load_entry)
            observer.assert_frozen()
        finally:
            observer.handle.remove()
    print(f"Stage-139: {STATUS}", flush=True)


if __name__ == "__main__":
    main()
