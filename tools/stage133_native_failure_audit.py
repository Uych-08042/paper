"""Inspect failure onsets and raw versus Hann choices on frozen native SGLA crops."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

import stage132_first_divergence as previous
from stage27_uav_collect import writer_lock
from stage27_uav_common import failure_events, scene_geometry, success_auc, tracking_iou
from stage28_response_common import dense_boxes, response_metrics
from stage46_shift_audit import publish_table


source, base, c, io = previous.source, previous.base, previous.c, previous.io
SCHEMA = "stage133_native_failure_audit_v1"
STATUS = "COMPLETE_CACHED_NATIVE_FAILURE_AUDIT"
DESIGN = dict(videos=28, frames=29047, events=19, good_iou=.5, low_iou=.2,
              adequate_coverage=.9, onset_offsets=[-3, -1, 0, 1, 2])
WORK = dict(new_search_calls=0, image_reads=0, training_updates=0, new_full_trajectories=0)
TABLES = ("video", "events", "anchors", "summary")


def category(native, raw, best, geometry):
    if native >= DESIGN["good_iou"]:
        return "native_good"
    if native >= DESIGN["low_iou"]:
        return "native_intermediate"
    if raw >= DESIGN["good_iou"]:
        return "raw_top1_recovers_good"
    if best >= DESIGN["good_iou"]:
        return "other_dense_cell_has_good"
    if geometry["coverage"] == 0:
        return "no_crop_overlap_no_good"
    if geometry["coverage"] < DESIGN["adequate_coverage"] or not geometry["center_in_usable"]:
        return "limited_crop_no_good"
    return "adequate_crop_no_good"


def frame_metrics(trace, i, gt, ap, window, inspect_peaks):
    parent, box = (trace[key][i, 0] for key in ("candidate_parents", "candidate_boxes"))
    raw, size, offset = (trace[key][i, 0] for key in ("raw", "size", "offset"))
    hw = trace["image_hw"][i]
    geo = scene_geometry(parent, gt, hw, ap["search_factor"], ap["search_size"])
    np.testing.assert_array_equal(geo["crop"], trace["crop"][i, 0])
    dense = dense_boxes(size, offset, parent, geo["factor"], ap["search_size"], hw)
    raw_cell = int(np.argmax(raw))
    hann_cell = int(np.argmax(raw * window))
    box_error = float(np.abs(dense[hann_cell] - box).max())
    if box_error > ap["design"]["box_atol"]:
        raise ValueError(f"Native decode mismatch at tracked frame {i + 1}: {box_error}")
    # Preserve the original box at Hann top-1, including near-integer coordinates.
    dense[hann_cell] = box
    ious = tracking_iou(dense, gt)
    native, unweighted, best = float(ious[hann_cell]), float(ious[raw_cell]), float(ious.max())
    peak_fields = {}
    if inspect_peaks:
        peak, _, _ = response_metrics(raw, window, dense, gt)
        peak_fields = {k: peak[k] for k in (
            "raw_top5_gt_recall", "weighted_top5_gt_recall",
            "raw_gt_rank_in_top5", "weighted_gt_rank_in_top5",
        )}
    return dict(native_iou=native, raw_top1_iou=unweighted, dense_best_iou=best,
                category=category(native, unweighted, best, geo), raw_cell=raw_cell,
                hann_cell=hann_cell, coverage=float(geo["coverage"]),
                gt_center_inside=geo["center_in_usable"],
                gt_image_fraction=float(geo["gt_image_fraction"]),
                input_gt_short_side=float(min(geo["input_wh"])),
                padding_fraction=float(geo["padding_fraction"]), decode_error=box_error,
                **peak_fields)


def anchors_for(events, frames):
    by_frame = {}
    for number, event in enumerate(events, start=1):
        start, recovery = event["start"], event["recovery"]
        pairs = [(start + d, f"onset{d:+d}") for d in DESIGN["onset_offsets"]]
        if recovery is not None:
            pairs.extend(((recovery - 1, "recovery_previous"), (recovery, "recovery")))
        for index, role in pairs:
            if 0 <= index < frames:
                by_frame.setdefault(index, []).append(f"{number}:{role}")
    return by_frame


def diagnose(trace, entry, ap, archived):
    truth = np.asarray(entry["ground_truth"], float)
    n = len(truth) - 1
    np.testing.assert_array_equal(trace["frame_id"], np.arange(1, n + 1))
    np.testing.assert_array_equal(trace["selected"], np.zeros(n, int))
    np.testing.assert_array_equal(trace["unique_search_calls"], np.ones(n, int))
    np.testing.assert_array_equal(trace["candidate_boxes"][:, 0], trace["boxes"])
    np.testing.assert_array_equal(trace["candidate_parents"][:, 0],
                                  np.vstack((entry["initial_box"], trace["boxes"][:-1])))
    native = tracking_iou(trace["boxes"], truth[1:])
    events = failure_events(native)
    native_auc = success_auc(native)
    np.testing.assert_allclose(native_auc, float(archived["auc"]), rtol=0, atol=1e-12)
    if len(events) != int(archived["failure_events"]):
        raise ValueError("Source failure count changed")
    marks = anchors_for(events, n)
    window = np.asarray(ap["window"], np.float32)
    if trace["raw"].shape != (n, 1, *window.shape):
        raise ValueError("Expected a single native response map per frame")
    points = [frame_metrics(trace, i, truth[i + 1], ap, window, i in marks) for i in range(n)]
    np.testing.assert_array_equal([p["native_iou"] for p in points], native)
    raw = np.asarray([p["raw_top1_iou"] for p in points])
    raw_auc = success_auc(raw)
    good, low = DESIGN["good_iou"], DESIGN["low_iou"]
    identity = dict(sequence=entry["sequence"], cohort=archived["cohort"])
    video = dict(identity, frames=n, events=len(events), native_auc=native_auc,
                 fixed_state_raw_auc=raw_auc, fixed_state_raw_delta=raw_auc - native_auc,
                 changed_peak_frames=sum(p["raw_cell"] != p["hann_cell"] for p in points),
                 native_low_iou_frames=int((native < low).sum()), raw_low_iou_frames=int((raw < low).sum()),
                 low_to_good_frames=int(((native < low) & (raw >= good)).sum()),
                 good_to_low_frames=int(((native >= good) & (raw < low)).sum()),
                 max_native_decode_error=max(p["decode_error"] for p in points))
    event_rows = [
        dict(identity, event=k, onset_frame=e["start"] + 1, confirmation_frame=e["confirmation"] + 1,
             end_frame=e["end"] + 1, recovery_frame=None if e["recovery"] is None else e["recovery"] + 1,
             censor=e["censor"], duration_frames=e["end"] - e["start"] + 1, **points[e["start"]])
        for k, e in enumerate(events, start=1)
    ]
    anchor_rows = [dict(identity, frame=i + 1, roles="|".join(roles), **points[i])
                   for i, roles in sorted(marks.items())]
    return video, event_rows, anchor_rows, Counter(p["category"] for p in points)


def check_results(root):
    report = c.c66.sealed(root / "stage133_decision.json", SCHEMA)
    if (report["status"] != STATUS or report["source_stage131_fingerprint"] != previous.SOURCE131
            or report["source_stage121_fingerprint"] != source.SOURCE121 or report["design"] != DESIGN
            or any(report[k] != DESIGN[k] for k in ("videos", "frames", "events"))
            or any(report[k] != count for k, count in WORK.items())
            or set(report["table_sha256"]) != {f"stage133_{name}.csv" for name in TABLES}):
        raise ValueError("Incomplete native failure audit")
    io.verify({str(root / key): value for key, value in report["table_sha256"].items()})


def run(root, r121, p121, r131, d131, jobs, load_entry, ap, artifacts):
    index = {r["sequence"]: r for r in source.prior.source.read_table(r131 / "analysis/stage131_seed_video.csv")
             if r["arm"] == "native"}
    if len(jobs) != DESIGN["videos"] or {j["spec"]["unit"]["sequence"] for j in jobs} != set(index):
        raise ValueError("Unexpected native video list")
    videos, event_rows, anchors = [], [], []
    counts = Counter()
    for job in jobs:
        entry = load_entry(job["spec"])
        sequence = job["spec"]["unit"]["sequence"]
        if entry["sequence"] != sequence or len(entry["ground_truth"]) != job["frames"] + 1:
            raise ValueError("Source video or annotation length changed")
        trace = source.prior.prior.load_trace(r121, p121, job)
        video, events, points, count = diagnose(trace, entry, ap, index[sequence])
        videos.append(video)
        event_rows.extend(events)
        anchors.extend(points)
        counts.update(count)
        print(f"[Stage-133] {sequence}: {video['frames']} frames, {len(events)} onsets", flush=True)
    frames = sum(r["frames"] for r in videos)
    if frames != DESIGN["frames"] or len(event_rows) != DESIGN["events"]:
        raise ValueError("Unexpected native frame or failure count")
    onset_counts = Counter(r["category"] for r in event_rows)
    summary = [dict(category=key, tracked_frames=count, onset_events=onset_counts[key],
                    onset_videos=len({r["sequence"] for r in event_rows if r["category"] == key}))
               for key, count in sorted(counts.items())]
    hashes = {}
    for name, rows in zip(TABLES, (videos, event_rows, anchors, summary)):
        path = root / f"stage133_{name}.csv"
        publish_table(path, rows)
        hashes[path.name] = c.file_hash(path)
    native_auc = float(np.mean([r["native_auc"] for r in videos]))
    raw_auc = float(np.mean([r["fixed_state_raw_auc"] for r in videos]))
    c.publish_identical(root / "stage133_decision.json", io.seal(dict(
        schema=SCHEMA, status=STATUS, source_stage131_fingerprint=d131["fingerprint"],
        source_stage121_fingerprint=source.SOURCE121, design=DESIGN, videos=len(videos),
        frames=frames, events=len(event_rows), anchors=len(anchors), native_auc=native_auc,
        fixed_state_raw_auc=raw_auc, fixed_state_raw_delta=raw_auc - native_auc,
        low_to_good_frames=sum(r["low_to_good_frames"] for r in videos),
        good_to_low_frames=sum(r["good_to_low_frames"] for r in videos),
        onset_categories=dict(sorted(onset_counts.items())), native_artifact_sha256=artifacts,
        table_sha256=hashes, code_sha256=c.file_hash(Path(__file__)), **WORK)))
    check_results(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76", "stage89", "stage93", "stage121", "stage131")
    for name in names:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-results", action="store_true")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if any(getattr(args, k) is None for k in names):
        parser.error("Supply the frozen Stage-76/89/93/121/131 directories")
    r76, r89, r93, r121, r131 = dirs = [getattr(args, k).expanduser().resolve() for k in names]
    if any(root == p or root in p.parents or p in root.parents for p in dirs):
        parser.error("Use a separate Stage-133 directory")
    if not (r131 / "analysis/stage131_decision.json").is_file():
        parser.error("Need the Stage-131 server results and linked native Stage-121 caches")
    with writer_lock(root):
        if (root / "stage133_decision.json").is_file():
            check_results(root)
            return
        p131 = c.c66.sealed(r131 / "manifest.json", source.SCHEMA)
        d131 = c.c66.sealed(r131 / "analysis/stage131_decision.json", source.SCHEMA)
        p121 = c.c66.sealed(r121 / "manifest.json", source.s121.SCHEMA)
        d121 = c.c66.sealed(r121 / "analysis/stage121_decision.json", source.s121.SCHEMA)
        if (d131["fingerprint"] != previous.SOURCE131 or d121["fingerprint"] != source.SOURCE121
                or d131["plan_fingerprint"] != p131["fingerprint"] or d121["plan_fingerprint"] != p121["fingerprint"]
                or not p131["reuse_controls"] or p131["runtime"] != p121["runtime"]
                or p131["jobs"] != source.source_jobs(p121)):
            raise ValueError("Need the original Stage-131 run and reused Stage-121 native caches")
        io.verify({str(r131 / "analysis/stage131_seed_video.csv"): d131["table_sha256"]["stage131_seed_video.csv"]})
        jobs = sorted((j for j in p121["jobs"] if j["arm"] == "native"), key=base.job_key)
        artifacts = {f"evaluation/{base.job_key(j)}{ext}": d121["artifact_sha256"][f"evaluation/{base.job_key(j)}{ext}"]
                     for j in jobs for ext in (".npz", ".receipt.json")}
        io.verify({str(r121 / path): digest for path, digest in artifacts.items()})
        s95 = base.s103.prior.s95
        p93, _, p89, report89, origin = s95.s94.load_run(r93, r76, r89)

        def load_entry(spec):
            return s95.load_unit(r93, r89, p93, p89, report89, origin, spec)[0]

        ap = base.e63.c.actor_plan(origin["original"])
        run(root, r121, p121, r131, d131, jobs, load_entry, ap, artifacts)


if __name__ == "__main__":
    main()
