"""Audit search coverage at the frozen Stage-125 trigger frames using Stage-121 caches."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import stage125_trigger_continuation as trigger
from stage27_uav_common import scene_geometry, tracking_iou

base, c, io = trigger.base, trigger.c, trigger.io
SCHEMA = "stage127_trigger_crop_coverage_v1"
SOURCE121 = "a83daf985bd2d0fd2ff00dd91793c12bb4960936db881400186b6abf8ac0d142"
SOURCE125 = "5f569cdeaede3ffc4e5440cc25362b6948f3fffcd4bb1737262abbe2121b83d9"
DESIGN = dict(good_iou=.5, near_miss=.1, adequate_coverage=.9, expected_windows=346, expected_no_good=333)
GROUPS = ("target_outside_image", "zero_crop_overlap", "under_10_percent", "partial_coverage", "adequate_coverage")
TABLES = ("summary", "points", "crops")
STATUS = "COMPLETE_TRIGGER_CROP_COVERAGE_AUDIT"
WORK = dict(image_reads=0, new_search_calls=0, training_updates=0)


def classify(image_fraction, coverage):
    if image_fraction == 0:
        return GROUPS[0]
    if coverage == 0:
        return GROUPS[1]
    if coverage < DESIGN["near_miss"]:
        return GROUPS[2]
    if coverage < DESIGN["adequate_coverage"]:
        return GROUPS[3]
    return GROUPS[4]


def inspect(data, entry, point, ap):
    frame = point["frame"]
    i = frame - 1
    meta = entry["frames"][frame]
    hw = (meta["height"], meta["width"])
    if data["frame_id"][i] != frame or not data["alternate_used"][i]:
        raise ValueError("The cached frame is not an actual Stage-121 trigger")
    np.testing.assert_array_equal(data["image_hw"][i], hw)
    gt = np.asarray(entry["ground_truth"][frame], dtype=float)
    ious = tracking_iou(data["candidate_boxes"][i], gt)
    archived = point["stage124_predictions"][str(trigger.SEEDS[0])]
    np.testing.assert_allclose(ious, list(map(float, archived["candidate_ious"].split("|"))), atol=1e-12, rtol=0)
    if ious.shape != (4,) or (ious < 0).any():
        raise ValueError("Expected four valid predictions and one valid target")
    np.testing.assert_equal(data["selected"][i], int(archived["recorded_action"]))

    ident = dict(sequence=point["sequence"], source_seed=point["source_seed"], frame=frame)
    crops = []
    for q, parent in enumerate(data["candidate_parents"][i]):
        geometry = scene_geometry(parent, gt, hw, ap["search_factor"], ap["search_size"])
        np.testing.assert_array_equal(geometry["crop"], data["crop"][i, q])
        full, image = geometry["coverage"], geometry["gt_image_fraction"]
        visible = min(1., full/image) if image > 0 else None
        # In addition to coverage, retain target size on the resized search input.
        x, y, side, _ = geometry["crop"]
        width, height = geometry["input_wh"]
        crops.append(dict(ident, slot=q, selected=q==int(data["selected"][i]), output_iou=float(ious[q]),
                          crop_x=int(x), crop_y=int(y), crop_side=int(side), coverage_full_gt=full,
                          gt_image_fraction=image, coverage_visible_gt=visible,
                          center_in_crop=geometry["center_in_usable"], input_target_width=float(width),
                          input_target_height=float(height), padding_fraction=geometry["padding_fraction"]))
    if len(crops) != 4:
        raise ValueError("Expected four cached crops")
    crop = max(crops, key=lambda r:r["coverage_full_gt"])
    bucket = classify(crop["gt_image_fraction"], crop["coverage_visible_gt"])
    best = float(ious.max())
    return dict(ident, best_output_iou=best, good_output=best>=DESIGN["good_iou"],
                gt_image_fraction=crop["gt_image_fraction"], max_coverage_full_gt=crop["coverage_full_gt"],
                max_coverage_visible_gt=crop["coverage_visible_gt"], best_coverage_slot=crop["slot"],
                center_in_any_crop=any(r["center_in_crop"] for r in crops),
                max_short_side_in_adequate_crop=max((min(r["input_target_width"],r["input_target_height"])
                    for r in crops if r["coverage_visible_gt"] is not None
                    and r["coverage_visible_gt"]>=DESIGN["adequate_coverage"]), default=None),
                bucket=bucket), crops


def summarize(points):
    rows = []
    for population, part in (("all", points), ("no_good_output", [r for r in points if not r["good_output"]])):
        counts = Counter(r["bucket"] for r in part)
        for key in GROUPS:
            subset = [r for r in part if r["bucket"]==key]
            rows.append(dict(population=population, bucket=key, points=counts[key], total=len(part),
                             fraction=counts[key]/len(part), videos=len({r["sequence"] for r in subset})))
    return rows


def check_results(root):
    plan = c.c66.sealed(root/"manifest.json", SCHEMA)
    report = c.c66.sealed(root/"stage127_decision.json", SCHEMA)
    n = DESIGN["expected_windows"]
    if (report["status"]!=STATUS or report["plan_fingerprint"]!=plan["fingerprint"] or plan["design"]!=DESIGN
            or report["source_stage125_fingerprint"]!=SOURCE125 or plan["source_stage121_fingerprint"]!=SOURCE121
            or report["row_counts"]!={"points":n,"crops":4*n,"summary":2*len(GROUPS)}
            or set(report["table_sha256"])!={f"stage127_{t}.csv" for t in TABLES}
            or any(report[k]!=v for k,v in WORK.items())):
        raise ValueError("Incomplete or mismatched crop coverage result")
    io.verify({str(root/name):value for name,value in report["table_sha256"].items()})
    return report


def run(root, source121, p121, p125, jobs, load_entry, ap, plan):
    points, crops = [], []
    grouped = defaultdict(list)
    for point in p125["points"]:
        if point["full_window"]:
            grouped[point["sequence"],point["source_seed"]].append(point)
    entries = {}
    for key, group in sorted(grouped.items()):
        job = jobs[key]
        sequence, _ = key
        if sequence not in entries:
            entries[sequence] = load_entry(job["spec"])
        entry = entries[sequence]
        if entry["sequence"] != sequence or len(entry["ground_truth"]) != job["frames"]+1:
            raise ValueError("Label identity or frame numbering changed")
        data = trigger.prior.load_trace(source121,p121,job)
        for point in group:
            row, local = inspect(data,entry,point,ap)
            points.append(row)
            crops.extend(local)
    counts = (len(points), sum(not r["good_output"] for r in points))
    ids = {(r["sequence"],r["source_seed"],r["frame"]) for r in points}
    if len(ids) != len(points) or counts != (DESIGN["expected_windows"], DESIGN["expected_no_good"]):
        raise ValueError(f"Stage-125 frozen point counts changed: {counts}")
    summary = summarize(points)
    tables = dict(summary=summary, points=points, crops=crops)
    hashes = {}
    for name, rows in tables.items():
        path = root/f"stage127_{name}.csv"
        base.s103.s86.publish_table(path, rows)
        hashes[path.name] = c.file_hash(path)
    c.publish_identical(root/"stage127_decision.json", io.seal(dict(schema=SCHEMA,
        status=STATUS,plan_fingerprint=plan["fingerprint"],
        source_stage125_fingerprint=SOURCE125,row_counts={k:len(v) for k,v in tables.items()},
        table_sha256=hashes, no_good_output=[r for r in summary if r["population"]=="no_good_output"], **WORK)))
    check_results(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76", "stage89", "stage93", "stage121", "stage125")
    for name in names:
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-results", action="store_true")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not all(getattr(args,n) for n in names):
        parser.error("Supply Stage-76/89/93/121/125 source directories")
    r76,r89,r93,r121,r125 = sources = [getattr(args,n).expanduser().resolve() for n in names]
    if any(root==p or root in p.parents or p in root.parents for p in sources):
        parser.error("Use a separate Stage-127 directory")
    if not all((path/"manifest.json").is_file() for path in (r121,r125)):
        parser.error("Need Stage-121 trajectory caches and the Stage-125 manifest")
    with base.s109.s106.s93.writer_lock(root):
        if (root/"stage127_decision.json").is_file():
            check_results(root)
            return
        p121 = c.c66.sealed(r121/"manifest.json",trigger.s121.SCHEMA)
        p125 = c.c66.sealed(r125/"manifest.json",trigger.SCHEMA)
        d121 = c.c66.sealed(r121/"analysis/stage121_decision.json",trigger.s121.SCHEMA)
        d125 = c.c66.sealed(r125/"analysis/stage125_decision.json",trigger.SCHEMA)
        if (d121["fingerprint"]!=SOURCE121 or d125["fingerprint"]!=SOURCE125 or p125["runtime_differences"]
                or d121["plan_fingerprint"]!=p121["fingerprint"] or d125["plan_fingerprint"]!=p125["fingerprint"]
                or d121["runtime"]!=d125["runtime"]
                or p125["inputs"].get(str(r121/"manifest.json"))!=c.file_hash(r121/"manifest.json")):
            raise ValueError("Need the frozen Stage-121/125 source lineage")
        jobs = trigger.reference_jobs(p121)
        selected = {(p["sequence"],p["source_seed"]) for p in p125["points"] if p["full_window"]}
        inputs = {str(path):c.file_hash(path) for path in (r121/"manifest.json",r125/"manifest.json",
                  r121/"analysis/stage121_decision.json",r125/"analysis/stage125_decision.json",Path(__file__).resolve())}
        traces = {}
        for key in sorted(selected):
            stem = f"evaluation/{base.job_key(jobs[key])}"
            for suffix in (".npz",".receipt.json"):
                name = stem+suffix
                traces[str(r121/name)] = d121["artifact_sha256"][name]
        io.verify(traces)
        s95 = base.s103.prior.s95
        p93,_,p89,report89,origin = s95.s94.load_run(r93,r76,r89)
        def load_entry(spec):
            return s95.load_unit(r93,r89,p93,p89,report89,origin,spec)[0]

        ap = base.e63.c.actor_plan(origin["original"])
        plan = io.seal(dict(schema=SCHEMA,design=DESIGN,inputs=inputs|traces,
                            source_stage121_fingerprint=SOURCE121,source_stage125_fingerprint=SOURCE125))
        c.publish_identical(root/"manifest.json",plan)
        run(root,r121,p121,p125,jobs,load_entry,ap,plan)


if __name__=="__main__":
    main()
