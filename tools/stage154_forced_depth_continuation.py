"""Measure one actual serial block addition and follow its box through native SGLA."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np

import stage137_shared_crop_layers as roster
from stage150_gray_write_continuation import first_run
from stage153_adaptive_depth import DepthObserver
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, tracking_iou
from stage46_shift_audit import publish_table


prior = roster.prior
base, c, io = prior.base, prior.c, prior.io
SCHEMA = "stage154_forced_serial_depth_continuation_v1"
STATUS = "COMPLETE_FORCED_SERIAL_DEPTH_CONTINUATION"
SOURCE137 = "a092d092f5fc2ff684f888c243197eefe6a8c70f430268d820c8475d2c38e89c"
SOURCES = ("stage76", "stage89", "stage93", "stage134", "stage137")
ARMS = ("native", "serial_extra")
DESIGN = dict(horizons=[8, 32], good_iou=.5, low_iou=.2, run_length=3, groups=roster.GROUPS)
TABLES = ("summary", "anchors", "windows", "timeline")
WORK = dict(training_updates=0, new_full_trajectories=0, confirmation_inference_calls=0)
FIELDS = ("boxes", "parent", "selected", "added", "block_calls", "crop_calls", "head_calls", "selector_calls")


def collect(observer, entry, case, parent, dataset_root, reader=checked_image):
    """Force the extra block only at the paired anchor; each path carries its own box."""
    frames = entry["frames"]
    observer.initialize(reader(dataset_root, frames[0]), entry["initial_box"], "native")
    first = case["frame"]
    frame_ids = np.arange(first, first+case["successors"]+1)
    first_image = reader(dataset_root, frames[first])
    paths = [[observer.step(first_image, parent.copy(), first, force_extra=arm == "serial_extra",
                            capture_crop=True)] for arm in ARMS]
    for frame in frame_ids[1:]:
        image = reader(dataset_root, frames[frame])
        for path in paths:
            path.append(observer.step(image, path[-1]["boxes"].copy(), int(frame)))
    data = {k:np.asarray([[row[k] for row in path] for path in paths]) for k in FIELDS}
    return dict(data, frame_id=frame_ids, anchor_crop_sha256=np.array([path[0]["crop_sha256"] for path in paths]))


def audit(data, case, reference, same_runtime, atol):
    first = case["frame"]
    count = case["successors"]+1
    shape = (len(ARMS), count)
    np.testing.assert_array_equal(data["frame_id"], reference["frame_id"])
    np.testing.assert_array_equal(data["frame_id"], np.arange(first, first+count))
    boxes = data["boxes"]
    if boxes.shape != (*shape, 4) or not np.isfinite(boxes).all() or (boxes[..., 2:] <= 0).any():
        raise ValueError("Invalid Stage-154 box trace")
    np.testing.assert_array_equal(data["parent"][:, 0], np.broadcast_to(reference["parent"][0, 0], (len(ARMS), 4)))
    np.testing.assert_array_equal(data["parent"][:, 1:], boxes[:, :-1])
    np.testing.assert_array_equal(data["anchor_crop_sha256"], [reference["crop_sha256"][0, 0]]*len(ARMS))
    selected = data["selected"]
    if selected.shape != shape or not np.isin(selected, np.arange(7, 13)).all() or selected[0, 0] != selected[1, 0]:
        raise ValueError("Both paths must use the same native selector at the anchor")
    expected_added = np.zeros(shape, int)
    if selected[0, 0] < 12:
        expected_added[1, 0] = selected[0, 0]+1
    np.testing.assert_array_equal(data["added"], expected_added)
    blocks = np.zeros((*shape, 12), int)
    blocks[..., :6] = 1
    np.put_along_axis(blocks, selected[..., None]-1, 1, axis=2)
    if expected_added[1, 0]:
        blocks[1, 0, expected_added[1, 0]-1] = 1
    np.testing.assert_array_equal(data["block_calls"], blocks)
    for k in ("crop_calls", "head_calls", "selector_calls"):
        np.testing.assert_array_equal(data[k], np.ones(shape, int))
    if not expected_added[1, 0] or np.array_equal(boxes[1, 0], boxes[0, 0]):
        np.testing.assert_array_equal(boxes[0], boxes[1])
    error = float(np.abs(boxes[0]-reference["boxes"][0]).max())
    if same_runtime:
        np.testing.assert_allclose(boxes[0], reference["boxes"][0], atol=atol, rtol=0)
        np.testing.assert_array_equal(selected[0], reference["layer"][0])
    return error


def describe(data, case, truth, source_error):
    frame = case["frame"]
    ids = data["frame_id"]
    ious = tracking_iou(data["boxes"], np.asarray(truth, float)[ids])
    valid = (ious >= 0.).all(axis=0)
    good, low = DESIGN["good_iou"], DESIGN["low_iou"]
    labels = {k:case[k] for k in ("case_id", "sequence", "frame", "kind", "group")}
    anchor = dict(labels, primary_layer=int(data["selected"][0, 0]),
                  extra_layer=int(data["added"][1, 0]) or None, evaluated=bool(data["added"][1, 0]),
                  native_iou=float(ious[0, 0]) if valid[0] else None,
                  serial_iou=float(ious[1, 0]) if valid[0] else None,
                  anchor_delta=float(ious[1, 0]-ious[0, 0]) if valid[0] else None,
                  new_good=bool(valid[0] and ious[0, 0] < good <= ious[1, 0]),
                  lost_good=bool(valid[0] and ious[1, 0] < good <= ious[0, 0]),
                  changed_box=not np.array_equal(data["boxes"][0, 0], data["boxes"][1, 0]),
                  native_source_max_box_error=source_error, crop_sha256=str(data["anchor_crop_sha256"][0]))
    windows = []
    for h in DESIGN["horizons"]:
        mask = valid[1:h+1]
        tail = ious[:, 1:h+1]
        row = dict(labels, evaluated=anchor["evaluated"], horizon=h, valid_successors=int(mask.sum()),
                   tail_mean_delta=float(np.mean((tail[1]-tail[0])[mask])) if mask.any() else None)
        for arm, u in zip(ARMS, tail):
            row.update({f"{arm}_mean_iou":float(u[mask].mean()) if mask.any() else None,
                        f"{arm}_first_good_run":first_run(mask & (u >= good), frame+1),
                        f"{arm}_first_low_run":first_run(mask & (u < low), frame+1)})
        windows.append(row)
    timeline = [dict(labels, arm=arm, video_frame=int(f), offset=i, iou=float(ious[j, i]) if valid[i] else None,
                     selected_layer=int(data["selected"][j, i]), extra_layer=int(data["added"][j, i]) or None)
                for j, arm in enumerate(ARMS) for i, f in enumerate(ids)]
    return anchor, windows, timeline


def summarize(anchors, windows):
    for group in ("all_onsets", *roster.GROUPS):
        subset = {r["case_id"]:r for r in anchors
                  if r["group"] == group or (group == "all_onsets" and r["kind"] == "onset")}
        for h in DESIGN["horizons"]:
            matched = [r for r in windows if r["horizon"] == h and r["case_id"] in subset]
            complete = [r for r in matched if r["evaluated"] and r["valid_successors"] == h]
            by_video = defaultdict(list)
            for row in complete:
                by_video[row["sequence"]].append(row["tail_mean_delta"])
            yield dict(group=group, horizon=h, cases=len(subset),
                       evaluated=sum(r["evaluated"] for r in subset.values()), complete_cases=len(complete),
                       complete_videos=len(by_video), new_anchor_good=sum(r["new_good"] for r in subset.values()),
                       lost_anchor_good=sum(r["lost_good"] for r in subset.values()),
                       video_mean_tail_iou_delta=float(np.mean([np.mean(x) for x in by_video.values()])) if by_video else None,
                       new_good_run=sum(r["native_first_good_run"] is None and r["serial_extra_first_good_run"] is not None for r in complete),
                       lost_good_run=sum(r["native_first_good_run"] is not None and r["serial_extra_first_good_run"] is None for r in complete),
                       new_low_run=sum(r["native_first_low_run"] is None and r["serial_extra_first_low_run"] is not None for r in complete))


def check_results(root):
    plan = c.c66.sealed(root/"manifest.json", SCHEMA)
    report = c.c66.sealed(root/"stage154_decision.json", SCHEMA)
    cases = len(plan["cases"])
    searches = len(ARMS)*sum(case["successors"]+1 for case in plan["cases"])
    if (plan["design"] != DESIGN or plan["source_stage137"] != SOURCE137 or report["status"] != STATUS
            or report["plan_fingerprint"] != plan["fingerprint"] or report["cases"] != cases
            or report["new_search_calls"] != searches or not 0 <= report["extra_blocks"] <= cases
            or report["new_block_calls"] != 7*searches+report["extra_blocks"]
            or len(report["artifact_sha256"]) != 2*cases or any(report[k] != v for k,v in WORK.items())
            or set(report["table_sha256"]) != {f"stage154_{name}.csv" for name in TABLES}):
        raise ValueError("Incomplete Stage-154 receipt")
    io.verify({str(root/k):v for k,v in (report["table_sha256"]|report["artifact_sha256"]).items()})


def run(root, r134, p134, plan, observer, ap, load_entry):
    anchors, windows, timeline, entries, hashes = [], [], [], {}, {}
    (root/"branches").mkdir(exist_ok=True)
    for case in plan["cases"]:
        cid, seq, first = (case[k] for k in ("case_id", "sequence", "frame"))
        if seq not in entries:
            entries[seq] = load_entry(seq)
        entry = entries[seq]
        reference = c.read_cache(r134/"branches"/f"{cid}.npz", dict(plan_fingerprint=p134["fingerprint"], case_id=cid))
        if (reference is None or entry["sequence"] != seq or len(entry["frames"]) != len(entry["ground_truth"])
                or not 1 <= first <= len(entry["frames"])-case["successors"]-1):
            raise ValueError(f"Missing frames or reference for {cid}")
        path = root/"branches"/f"{cid}.npz"
        meta = dict(plan_fingerprint=plan["fingerprint"], case_id=cid)
        data = c.read_cache(path, meta)
        fresh = data is None
        if fresh:
            data = collect(observer, entry, case, reference["parent"][0, 0], ap["root"])
        error = audit(data, case, reference, not plan["runtime_differences"], ap["design"]["box_atol"])
        if fresh:
            c.save(path, data, meta)
        for item in (path, path.with_suffix(".receipt.json")):
            hashes[item.relative_to(root).as_posix()] = c.file_hash(item)
        anchor, pair, frames = describe(data, case, entry["ground_truth"], error)
        anchors.append(anchor)
        windows.extend(pair)
        timeline.extend(frames)
        print(f"[Stage-154] {cid}: L{anchor['primary_layer']} extra={anchor['extra_layer']}", flush=True)
    observer.native.assert_frozen()
    tables = {}
    for name, rows in zip(TABLES, (list(summarize(anchors, windows)), anchors, windows, timeline)):
        path = root/f"stage154_{name}.csv"
        publish_table(path, rows)
        tables[path.name] = c.file_hash(path)
    extra = sum(row["evaluated"] for row in anchors)
    c.publish_identical(root/"stage154_decision.json", io.seal(dict(
        schema=SCHEMA, status=STATUS, role="consumed_development_forced_depth_anchor_and_native_continuation",
        plan_fingerprint=plan["fingerprint"], source_stage134=plan["source_stage134"], source_stage137=SOURCE137,
        runtime=plan["runtime"], runtime_differences=plan["runtime_differences"], cases=len(anchors),
        new_search_calls=len(timeline), new_block_calls=7*len(timeline)+extra, extra_blocks=extra,
        table_sha256=tables, artifact_sha256=hashes, **WORK)))
    check_results(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in SOURCES:
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage154_forced_serial")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if args.checkpoint is None or any(getattr(args, k) is None for k in SOURCES):
        parser.error("Supply all source directories and the original SGLA checkpoint")
    sources = [getattr(args, k).expanduser().resolve() for k in SOURCES]
    for name, path in zip(SOURCES, sources):
        if not path.is_dir():
            parser.error(f"Missing --{name} source directory: {path}")
        if root == path or root in path.parents or path in root.parents:
            parser.error("Use a separate Stage-154 output directory")
    r76,r89,r93,r134,r137 = sources
    with writer_lock(root):
        if (root/"stage154_decision.json").is_file():
            check_results(root)
            return
        prior.check_results(r134)
        roster.check_results(r137)
        p134 = c.c66.sealed(r134/"manifest.json", prior.SCHEMA)
        p137 = c.c66.sealed(r137/"manifest.json", roster.SCHEMA)
        d134 = c.c66.sealed(r134/"stage134_decision.json", prior.SCHEMA)
        d137 = c.c66.sealed(r137/"stage137_decision.json", roster.SCHEMA)
        cases = p137["cases"]
        archived = {r["case_id"]:r for r in p134["cases"]}
        if (d137["fingerprint"] != SOURCE137 or d134["fingerprint"] != p137["source_stage134"]
                or Counter(r["group"] for r in cases) != roster.GROUPS
                or set(archived) != {r["case_id"] for r in cases}
                or any(any(r[k] != v for k,v in archived[r["case_id"]].items())
                       or r["successors"] != DESIGN["horizons"][-1] for r in cases)):
            raise ValueError("Expected the 47 reviewed Stage-134 and Stage-137 anchors")
        s95 = base.s103.prior.s95
        p93,_,p89,d89,origin = s95.s94.load_run(r93,r76,r89)
        old = origin["original"]
        specs = {s["unit"]["sequence"]:s for s in p89["units"]}

        def load_entry(seq):
            return s95.load_unit(r93,r89,p93,p89,d89,origin,specs[seq])[0]

        import torch
        torch.set_num_threads(1)
        runtime = base.s103.s76.device_runtime(old,torch,args.gpu)
        diffs = base.s109.check_runtime(p134["runtime"],runtime)
        manifest = root/"manifest.json"
        if manifest.is_file() and c.c66.sealed(manifest,SCHEMA)["runtime"] != runtime:
            raise ValueError("GPU/runtime changed on resume; set STAGE154_ROOT to a new directory")
        plan = io.seal(dict(schema=SCHEMA, design=DESIGN, arms=list(ARMS), cases=cases,
                            runtime=runtime, runtime_differences=diffs,
                            source_stage134=d134["fingerprint"], source_stage137=SOURCE137,
                            code_sha256={p.name:c.file_hash(p) for p in (
                                Path(__file__), Path(__file__).with_name("stage153_adaptive_depth.py"))}))
        c.publish_identical(manifest, plan)
        frozen = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        args.parameter = runtime["tracker"]["parameter"]
        args.checkpoint = args.checkpoint.expanduser().resolve()
        native,_ = base.s103.s76.startup.checked_runtime(args,frozen,root)
        ap = base.e63.c.actor_plan(old)
        with ExitStack() as stack:
            run(root,r134,p134,plan,DepthObserver(native,stack),ap,load_entry)
    print(f"Stage-154: {STATUS}", flush=True)


if __name__ == "__main__":
    main()
