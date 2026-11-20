"""Replay all calibrated Stage-149 native grayscale switches as single box writes."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

import stage149_score_ablation as ablation
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, crop_geometry, source_group, tracking_iou
from stage46_shift_audit import publish_table


fixed, c, io = ablation.fixed, ablation.c, ablation.io
audit = fixed.prior.native_audit
base, source = audit.base, audit.source
SCHEMA = "stage150_gray_write_continuation_v1"
STATUS = "COMPLETE_CALIBRATED_GRAY_WRITE_CONTINUATION"
SOURCE149 = "5dcfaca88b2511cf550c431ddf745a98b2d4b17063aa7200cc8950c5b1394549"
SOURCE133 = "b748935be8765678a92156c9000ee79fd0e308bc951ed29a19790a5b7f23e814"
CUE = "template_ncc_only"
ARMS = ("regression", "gray_once")
DESIGN = dict(threshold=.025, cases=14, horizons=[8, 32], good_iou=.5, low_iou=.2, run_length=3)
WORK = dict(anchor_search_calls=0, training_updates=0, subsequent_gray_decisions=0,
            confirmation_inference_calls=0, new_full_trajectories=0)
TABLES = ("summary", "windows", "timeline")
SOURCES = ("stage76", "stage89", "stage93", "stage121", "stage133", "stage143", "stage149")
FIELDS = ("boxes", "parent", "crop", "crop_sha256", "cell", "layer", "head_calls", "selector_calls", "block_calls")


def switches(rows):
    by_sequence = defaultdict(list)
    seen = set()
    for row in rows:
        if row["threshold_mode"] != "calibrated" or row[f"dense_{CUE}_switch"] != "True":
            continue
        sequence, frame = row["sequence"], int(row["frame"])
        threshold, margin = float(row[f"{CUE}_threshold"]), float(row[f"{CUE}_margin"])
        anchor, gray = int(row["anchor_cell"]), int(row[f"dense_{CUE}_cell"])
        if (row["state"] != "native" or row["source"] != source_group(sequence)
                or (sequence, frame) in seen or frame < 1
                or threshold != DESIGN["threshold"] or not np.isfinite(margin) or margin <= threshold
                or anchor != int(row["dense_quality_cell"]) or gray == anchor
                or gray not in map(int, row["matched_cells"].split("|"))):
            raise ValueError(f"Unexpected Stage-149 switch: {sequence}/{frame}")
        seen.add((sequence, frame))
        by_sequence[sequence].append(dict(
            case_id=f"{sequence}__f{frame:06d}", sequence=sequence, source=row["source"], frame=frame,
            onset=row["onset"] == "True", cells=[anchor, gray], margin=margin,
            anchor_ious=[float(row["anchor_iou"]), float(row[f"dense_{CUE}_iou"])]))
    if len(seen) != DESIGN["cases"]:
        raise ValueError(f"Expected all {DESIGN['cases']} calibrated Stage-149 switches")
    return dict(sorted(by_sequence.items()))


def choose(row, model):
    dense = row["pools"]["dense"]
    values = fixed.score(model, row["x"])[dense]
    if not np.isfinite(values).all():
        raise ValueError("Invalid frozen regression score")
    return int(row["cells"][dense[values.argmax()]])


def freeze_case(case, trace, entry, model, ap):
    frame = case["frame"]
    frames = entry["frames"]
    if not 1 <= frame < len(frames) or len(entry["ground_truth"]) != len(frames):
        raise ValueError(f"Bad anchor frame or annotation length for {case['case_id']}")
    i = frame - 1
    parent, hw = trace["candidate_parents"][i, 0], trace["image_hw"][i]
    crop, _, factor = crop_geometry(parent, hw, ap["search_factor"], ap["search_size"])
    np.testing.assert_array_equal(crop, trace["crop"][i, 0])
    raw, size, offset = (trace[k][i, 0] for k in ("raw", "size", "offset"))
    row, grid, _ = fixed.candidate_inputs(raw, size, offset, parent, factor, hw, trace["boxes"][i], ap)
    cells = case["cells"]
    if cells[0] != choose(row, model) or not np.isin(cells, row["cells"][row["pools"]["dense"]]).all():
        raise ValueError(f"Stage-149 starting cells differ for {case['case_id']}")
    boxes = grid[cells]
    np.testing.assert_allclose(tracking_iou(boxes, entry["ground_truth"][frame]), case["anchor_ious"],
                               atol=1e-12, rtol=0, err_msg=case["case_id"])
    successors = min(DESIGN["horizons"][-1], len(frames) - frame - 1)
    return dict(case, successors=successors, anchor_boxes=boxes.tolist(), anchor_parent=parent.tolist(),
                anchor_crop=crop.tolist(), anchor_crop_sha256=str(trace["crop_sha256"][i, 0]))


def collect(observer, case, frames, initial_box, model, ap, *,
            read_image=checked_image, observe=base.e63.actor.observe):
    """The cached writes differ at t. Both branches use regression at t+1 onward."""
    first, n = case["frame"], case["successors"]
    frame_ids = np.arange(first + 1, first + n + 1)
    rows = [[] for _ in ARMS]
    if n:
        observer.initialize(read_image(ap["root"], frames[0]), initial_box)
        parents = np.array(case["anchor_boxes"], float)
        for frame in frame_ids:
            image = read_image(ap["root"], frames[frame])
            for j, parent in enumerate(parents):
                obs = observe(observer, image, parent.copy(), int(frame), ap)
                maps = (obs[k] for k in ("raw", "size", "offset"))
                inputs, grid, _ = fixed.candidate_inputs(
                    *maps, obs["parent"], obs["factor"], obs["image_hw"], obs["boxes"], ap)
                cell = choose(inputs, model)
                rows[j].append(dict(obs, cell=cell, boxes=grid[cell].copy()))
                parents[j] = grid[cell]
    data = {field: np.asarray([[row[field] for row in branch] for branch in rows]) for field in FIELDS}
    return dict(data, frame_id=frame_ids)


def audit_branch(data, case):
    n = case["successors"]
    np.testing.assert_array_equal(data["frame_id"], np.arange(case["frame"] + 1, case["frame"] + n + 1))
    shape = (len(ARMS), n)
    if not n:
        if any(data[k].shape != shape for k in FIELDS):
            raise ValueError("A terminal anchor cannot have successors")
        return
    for name in ("boxes", "parent", "crop"):
        value = data[name]
        if value.shape != (*shape, 4) or not np.isfinite(value).all() or (value[..., 2:] <= 0).any():
            raise ValueError(f"Invalid Stage-150 {name}")
    for name in ("cell", "crop_sha256", "layer", "head_calls", "selector_calls"):
        if data[name].shape != shape:
            raise ValueError(f"Stage-150 shape changed: {name}")
    np.testing.assert_array_equal(data["parent"][:, 0], case["anchor_boxes"])
    np.testing.assert_array_equal(data["parent"][:, 1:], data["boxes"][:, :-1])
    layers = data["layer"]
    if not np.isin(layers, np.arange(7, 13)).all():
        raise ValueError("Stage-150 selected layer outside 7–12")
    expected = np.zeros((*shape, 12), int)
    expected[:, :, :6] = 1
    np.put_along_axis(expected, layers[..., None] - 1, 1, axis=2)
    np.testing.assert_array_equal(data["block_calls"], expected)
    for name in ("head_calls", "selector_calls"):
        np.testing.assert_array_equal(data[name], np.ones(shape, int))


def first_run(values, first_frame):
    length = DESIGN["run_length"]
    if len(values) >= length:
        starts = np.flatnonzero(np.lib.stride_tricks.sliding_window_view(values, length).all(axis=1))
        if starts.size:
            return first_frame + int(starts[0])
    return None


def describe(data, case, truth):
    first, n = case["frame"], case["successors"]
    anchor = np.array(case["anchor_ious"])
    tail = np.array([tracking_iou(boxes, truth[data["frame_id"]]) for boxes in data["boxes"]]) if n else np.empty((2, 0))
    valid = (tail >= 0.).all(axis=0)
    diff = np.flatnonzero(data["crop_sha256"][0] != data["crop_sha256"][1])
    labels = dict(case_id=case["case_id"], sequence=case["sequence"], source=case["source"],
                  frame=first, onset=case["onset"], gray_margin=case["margin"])
    windows = []
    for horizon in DESIGN["horizons"]:
        observed = min(n, horizon)
        mask = valid[:observed]
        group = dict(labels, horizon=horizon, successors=observed, valid_successors=int(mask.sum()),
                     full_labeled_window=observed == horizon and mask.all(),
                     anchor_delta=float(anchor[1] - anchor[0]),
                     first_different_crop_pixels=first + 1 + int(diff[0]) if diff.size and diff[0] < observed else None)
        for j, arm in enumerate(ARMS):
            x = tail[j, :observed]
            group.update({
                f"{arm}_anchor_iou": float(anchor[j]),
                f"{arm}_successor_iou": float(x[mask].mean()) if mask.any() else None,
                f"{arm}_last_iou": float(x[-1]) if observed and mask[-1] else None,
                f"{arm}_low_fraction": float((x[mask] < DESIGN["low_iou"]).mean()) if mask.any() else None,
                f"{arm}_first_good_run": first_run(mask & (x >= DESIGN["good_iou"]), first + 1),
                f"{arm}_first_low_run": first_run(mask & (x < DESIGN["low_iou"]), first + 1)})
        group["delta_successor_iou"] = float((tail[1, :observed] - tail[0, :observed])[mask].mean()) if mask.any() else None
        windows.append(group)

    timeline = []
    for j, arm in enumerate(ARMS):
        context = dict(labels, arm=arm)
        timeline.append(dict(context, offset=0, video_frame=first, iou=float(anchor[j]), selected_cell=case["cells"][j],
                             crop_sha256=case["anchor_crop_sha256"], head_calls=0, selector_calls=0, block_calls=0))
        for i, frame in enumerate(data["frame_id"]):
            timeline.append(dict(context, offset=i+1, video_frame=int(frame), iou=float(tail[j,i]) if valid[i] else None,
                                 selected_cell=int(data["cell"][j,i]), crop_sha256=str(data["crop_sha256"][j,i]),
                                 head_calls=int(data["head_calls"][j,i]), selector_calls=int(data["selector_calls"][j,i]),
                                 block_calls=int(data["block_calls"][j,i].sum())))
    return windows, timeline


def summarize(windows):
    for horizon in DESIGN["horizons"]:
        rows = [r for r in windows if r["horizon"] == horizon]
        full = [r for r in rows if r["full_labeled_window"]]
        by_source = defaultdict(list)
        for row in full:
            by_source[row["source"]].append(row["delta_successor_iou"])
        good = DESIGN["good_iou"]
        yield dict(horizon=horizon, cases=len(rows), complete_labeled_cases=len(full), complete_sources=len(by_source),
                   partial_cases=sum(0 < r["successors"] < horizon for r in rows),
                   anchor_only_cases=sum(r["successors"] == 0 for r in rows),
                   anchor_mean_delta=float(np.mean([r["anchor_delta"] for r in rows])),
                   anchor_new_good=sum(r["regression_anchor_iou"] < good <= r["gray_once_anchor_iou"] for r in rows),
                   successor_source_mean_delta=float(np.mean([np.mean(v) for v in by_source.values()])) if full else None,
                   better_cases=sum(r["delta_successor_iou"] > 0 for r in full),
                   worse_cases=sum(r["delta_successor_iou"] < 0 for r in full),
                   added_good_run=sum(r["regression_first_good_run"] is None and r["gray_once_first_good_run"] is not None for r in full),
                   lost_good_run=sum(r["regression_first_good_run"] is not None and r["gray_once_first_good_run"] is None for r in full),
                   added_low_run=sum(r["regression_first_low_run"] is None and r["gray_once_first_low_run"] is not None for r in full),
                   removed_low_run=sum(r["regression_first_low_run"] is not None and r["gray_once_first_low_run"] is None for r in full))


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    report = c.c66.sealed(root / "stage150_decision.json", SCHEMA)
    if (plan["design"] != DESIGN or report["status"] != STATUS or plan["source_stage149"] != SOURCE149
            or report["plan_fingerprint"] != plan["fingerprint"] or report["cases"] != DESIGN["cases"]
            or report["new_search_calls"] != plan["search_budget"]
            or any(report[key] != value for key, value in WORK.items())
            or len(report["artifact_sha256"]) != 2 * DESIGN["cases"]
            or set(report["table_sha256"]) != {f"stage150_{name}.csv" for name in TABLES}):
        raise ValueError("Incomplete Stage-150 continuation")
    hashes = report["table_sha256"] | report["artifact_sha256"]
    io.verify({str(root / name): value for name, value in hashes.items()})


def run(root, plan, contexts, observer, ap):
    windows, timeline, artifacts = [], [], {}
    (root / "branches").mkdir(exist_ok=True)
    for case in plan["cases"]:
        key = dict(plan_fingerprint=plan["fingerprint"], case_id=case["case_id"])
        path = root / "branches" / f"{case['case_id']}.npz"
        entry, model = contexts[case["sequence"]]
        data = c.read_cache(path, key)
        if data is None:
            data = collect(observer, case, entry["frames"], entry["initial_box"], model, ap)
            audit_branch(data, case)
            observer.assert_frozen()
            c.save(path, data, key)
        else:
            audit_branch(data, case)
        for item in (path, path.with_suffix(".receipt.json")):
            artifacts[item.relative_to(root).as_posix()] = c.file_hash(item)
        rows, frames = describe(data, case, np.asarray(entry["ground_truth"], float))
        windows.extend(rows)
        timeline.extend(frames)
        print(f"[Stage-150] {case['case_id']}: {case['successors']} successors per arm", flush=True)
    calls = sum(row["head_calls"] for row in timeline)
    if calls != plan["search_budget"]:
        raise ValueError("Stage-150 search budget changed")
    tables = {}
    for name, rows in zip(TABLES, (list(summarize(windows)), windows, timeline)):
        path = root / f"stage150_{name}.csv"
        publish_table(path, rows)
        tables[path.name] = c.file_hash(path)
    c.publish_identical(root / "stage150_decision.json", io.seal(dict(
        schema=SCHEMA, status=STATUS, role="consumed_development_single_write_diagnostic",
        plan_fingerprint=plan["fingerprint"], source_stage149=SOURCE149, runtime=plan["runtime"],
        stage121_runtime_differences=plan["stage121_runtime_differences"],
        cases=len(plan["cases"]), new_search_calls=calls, table_sha256=tables, artifact_sha256=artifacts, **WORK)))
    check_results(root)


def load_cases(roots):
    r76, r89, r93, r121, r133, r143, r149 = roots
    ablation.check_results(r149)
    fixed.check_results(r143)
    audit.check_results(r133)
    p121 = c.c66.sealed(r121 / "manifest.json", source.s121.SCHEMA)
    reports = (
        (r149 / "stage149_decision.json", ablation.SCHEMA, SOURCE149),
        (r143 / "stage143_decision.json", fixed.SCHEMA, ablation.prior.SOURCE143),
        (r133 / "stage133_decision.json", audit.SCHEMA, SOURCE133),
        (r121 / "analysis/stage121_decision.json", source.s121.SCHEMA, source.SOURCE121))
    _, _, d133, d121 = verified = [c.c66.sealed(path, schema) for path, schema, _ in reports]
    if (any(r["fingerprint"] != expected for r, (_, _, expected) in zip(verified, reports))
            or d121["plan_fingerprint"] != p121["fingerprint"]):
        raise ValueError("Stage-150 source results differ from reviewed runs")
    io.verify({str(r121 / name): value for name, value in d133["native_artifact_sha256"].items()})
    models = c.c66.sealed(r143 / "stage143_models.json", fixed.SCHEMA)["models"]
    with (r149 / "stage149_frames.csv").open(encoding="utf-8", newline="") as handle:
        selected = switches(csv.DictReader(handle))
    jobs = {j["spec"]["unit"]["sequence"]: j for j in p121["jobs"] if j["arm"] == "native"}
    s95 = base.s103.prior.s95
    p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
    old = origin["original"]
    ap = base.e63.c.actor_plan(old)
    cases, contexts = [], {}
    for sequence, rows in selected.items():
        job = jobs[sequence]
        entry = s95.load_unit(r93, r89, p93, p89, d89, origin, job["spec"])[0]
        trace = source.prior.prior.load_trace(r121, p121, job)
        model = models[source_group(sequence)]
        if (entry["sequence"] != sequence or len(entry["frames"]) != job["frames"] + 1
                or model["held_source"] != source_group(sequence) or model["held_source"] in model["fit_sources"]):
            raise ValueError(f"Stage-150 source grouping differs: {sequence}")
        np.testing.assert_array_equal(trace["frame_id"], np.arange(1, job["frames"] + 1))
        cases.extend(freeze_case(row, trace, entry, model, ap) for row in sorted(rows, key=lambda r:r["frame"]))
        contexts[sequence] = entry, model
    return p121, old, ap, cases, contexts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in SOURCES:
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage150_gray_write")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if args.checkpoint is None or any(getattr(args, key) is None for key in SOURCES):
        parser.error("Supply the frozen source directories and SGLA checkpoint")
    roots = [getattr(args, key).expanduser().resolve() for key in SOURCES]
    for key, path in zip(SOURCES, roots):
        if not path.is_dir():
            parser.error(f"Missing --{key} source directory: {path}")
        if path == root or path in root.parents or root in path.parents:
            parser.error("Use a separate Stage-150 output directory")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    with writer_lock(root):
        if (root / "stage150_decision.json").is_file():
            check_results(root)
            return
        p121, old, ap, cases, contexts = load_cases(roots)
        import torch
        torch.set_num_threads(1)
        runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
        differences = base.s109.check_runtime(p121["runtime"], runtime)
        manifest = root / "manifest.json"
        if manifest.is_file() and c.c66.sealed(manifest, SCHEMA)["runtime"] != runtime:
            raise ValueError("Runtime changed on resume; set STAGE150_ROOT to a fresh directory")
        plan = io.seal(dict(schema=SCHEMA, design=DESIGN, source_stage149=SOURCE149,
                            source_stage121=p121["fingerprint"], source_stage143=ablation.prior.SOURCE143,
                            cases=cases, arms=list(ARMS), runtime=runtime, stage121_runtime_differences=differences,
                            search_budget=len(ARMS) * sum(r["successors"] for r in cases),
                            code_sha256={Path(p).name:c.file_hash(Path(p)) for p in (__file__, fixed.__file__)}))
        c.publish_identical(manifest, plan)
        frozen = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        args.parameter = runtime["tracker"]["parameter"]
        native, _ = base.s103.s76.startup.checked_runtime(args, frozen, root)
        observer = base.e63.Capture(native)
        try:
            run(root, plan, contexts, observer, ap)
        finally:
            observer.handle.remove()
    print(f"Stage-150: {STATUS}", flush=True)


if __name__ == "__main__":
    main()
