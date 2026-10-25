"""Describe UAV failure regimes; do not select a new policy or claim causality."""

import argparse
import csv
import os
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_json
from stage27_uav_common import (DESIGN, SCHEMA, coverage_stratum, failure_events,
    load_manifest, load_sealed, load_trace, motion_stratum, nonempty_path,
    seal, size_stratum, success_auc)


EVENT_FIELDS = ("sequence", "source_group_proxy", "event", "onset_frame", "confirmed_frame",
    "end_frame", "recovery_frame", "censor", "onset_coverage", "onset_coverage_stratum",
    "onset_projected_target_side_px", "onset_background_motion_proxy",
    "first_outside_offset_in_review", "preceding_iou", "event_low_iou_frames",
    "review_start_frame", "review_stop_frame_exclusive", "manual_cause", "visibility")


def write_table(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (None if isinstance(value, (float, np.floating))
                                   and not np.isfinite(value) else value) for key, value in row.items()})
    os.replace(temporary, path)


def interval(values, rng, samples):
    values = np.asarray(values, float)
    if len(values) < 2:
        return None, None
    draws = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(1)
    return tuple(float(x) for x in np.quantile(draws, [.025, .975]))


def summarize_scenarios(traces, samples=5000, seed=2093):
    groups = {}
    for entry, data in traces:
        n = len(data["iou"])
        sizes = np.sqrt(data["input_wh"].prod(1))
        cov = np.array([coverage_stratum(v) for v in data["coverage"]])
        size = np.array([size_stratum(v) for v in sizes])
        motion = np.array([motion_stratum(v) for v in data["motion"][:, 0]])
        masks = {("all", "all"): np.ones(n, bool)}
        for axis, values in (("coverage", cov), ("projected_input_size", size), ("background_motion_proxy", motion)):
            masks.update({(axis, value): values == value for value in sorted(set(values))})
        for s in sorted(set(size)):
            for m in sorted(set(motion)):
                # Restrict the compound representation probe to targets substantially in the usable crop.
                masks[("inside_size_x_motion", s + "|" + m)] = (size == s) & (motion == m) & (data["coverage"] >= .9)
        for key, mask in masks.items():
            if mask.any():
                values = data["iou"][mask]
                groups.setdefault(key, []).append(dict(auc=success_auc(values),
                    low_ratio=float((values < .2).mean()), frames=len(values), low_frames=int((values < .2).sum())))
    rng = np.random.default_rng(seed)
    rows = []
    for (axis, stratum), videos in sorted(groups.items()):
        auc, low = [v["auc"] for v in videos], [v["low_ratio"] for v in videos]
        auc_ci, low_ci = interval(auc, rng, samples), interval(low, rng, samples)
        rows.append(dict(axis=axis, stratum=stratum, videos=len(videos),
            tracked_frames=sum(v["frames"] for v in videos), low_iou_frames=sum(v["low_frames"] for v in videos),
            macro_auc=float(np.mean(auc)), auc_ci_low=auc_ci[0], auc_ci_high=auc_ci[1],
            macro_low_iou_fraction=float(np.mean(low)), low_ci_low=low_ci[0], low_ci_high=low_ci[1]))
    return rows


def event_rows(entry, data):
    rows = []
    for number, event in enumerate(failure_events(data["iou"]), 1):
        start, confirmation, end = event["start"], event["confirmation"], event["end"]
        before = max(0, start - DESIGN["review_before"])
        outside = np.flatnonzero(data["coverage"][before:start + 1] < .1)
        rows.append(dict(sequence=entry["sequence"], source_group_proxy=entry["source_group_proxy"], event=number,
            onset_frame=int(data["frame_id"][start]), confirmed_frame=int(data["frame_id"][confirmation]),
            end_frame=int(data["frame_id"][end]),
            recovery_frame=None if event["recovery"] is None else int(data["frame_id"][event["recovery"]]),
            censor=event["censor"], onset_coverage=float(data["coverage"][start]),
            onset_coverage_stratum=coverage_stratum(data["coverage"][start]),
            onset_projected_target_side_px=float(np.sqrt(data["input_wh"][start].prod())),
            onset_background_motion_proxy=float(data["motion"][start, 0]),
            first_outside_offset_in_review=None if not len(outside) else int(before + outside[0] - start),
            preceding_iou=None if start == 0 else float(data["iou"][start - 1]),
            event_low_iou_frames=int((data["iou"][start:end + 1] < .2).sum()),
            review_start_frame=max(0, int(data["frame_id"][start]) - 10),
            review_stop_frame_exclusive=min(len(entry["frames"]), int(data["frame_id"][confirmation]) + 31),
            manual_cause="unreviewed", visibility="unknown"))
    return rows


def frame_rows(entry, data):
    rows = []
    for i, frame_id in enumerate(data["frame_id"]):
        row = dict(sequence=entry["sequence"], source_group_proxy=entry["source_group_proxy"],
            frame_id=int(frame_id), source_file=entry["frames"][int(frame_id)]["file"],
            iou=float(data["iou"][i]), layer=int(data["layer"][i]),
            coverage=float(data["coverage"][i]), coverage_stratum=coverage_stratum(data["coverage"][i]),
            gt_image_fraction=float(data["gt_image_fraction"][i]),
            projected_target_side_px=float(np.sqrt(data["input_wh"][i].prod())),
            raw_score_peak=float(data["raw_score_peak"][i]),
            background_motion_proxy=float(data["motion"][i, 0]),
            target_residual_motion_proxy=float(data["motion"][i, 1]),
            motion_tracks=int(data["motion"][i, 2]), motion_inliers=int(data["motion"][i, 3]),
            motion_inlier_fraction=float(data["motion"][i, 4]),
            target_laplacian_variance=float(data["sharpness"][i]),
            padding_fraction=float(data["padding_fraction"][i]), visibility="unknown")
        for key in ("previous", "boxes", "gt", "crop", "usable"):
            row.update({key + "_" + axis: float(value) for axis, value in zip("xywh", data[key][i])})
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True)
    parser.add_argument("--allow_incomplete", action="store_true")
    args = parser.parse_args()
    directory = Path(args.input).expanduser().resolve()
    run = load_sealed(directory / "run.json")
    if run.get("role") != "original_sgla_descriptive_uav_diagnosis":
        raise ValueError("Expected a Stage-27 original-SGLA collection")
    manifest = load_manifest(run["manifest"])
    traces, missing, hashes = [], [], {}
    expected = {e["sequence"] + "__stage27_uav.npz" for e in manifest["sequences"]}
    extras = [p.name for p in (directory / "traces").glob("*.npz") if p.name not in expected]
    if extras:
        raise ValueError("Unexpected trace files: " + str(extras))
    for entry in manifest["sequences"]:
        path = directory / "traces" / (entry["sequence"] + "__stage27_uav.npz")
        if not path.is_file():
            missing.append(entry["sequence"])
        else:
            traces.append((entry, load_trace(path, entry, run)))
            hashes[path.name] = file_hash(path)
    if not traces or (missing and not args.allow_incomplete):
        raise ValueError("{} videos collected, {} missing; pilot only: --allow_incomplete".format(len(traces), len(missing)))
    output = directory / ("analysis_pilot" if args.allow_incomplete else "analysis")
    frames, events, videos = [], [], []
    for entry, data in traces:
        frames.extend(frame_rows(entry, data))
        ev = event_rows(entry, data)
        events.extend(ev)
        videos.append(dict(sequence=entry["sequence"], source_group_proxy=entry["source_group_proxy"],
            tracked_frames=len(data["iou"]), auc=success_auc(data["iou"]),
            low_iou_frames=int((data["iou"] < .2).sum()), low_iou_fraction=float((data["iou"] < .2).mean()),
            events=len(ev), background_motion_known_fraction=float(np.isfinite(data["motion"][:, 0]).mean())))
    write_table(output / "stage27_frame_diagnostic.csv", frames)
    write_table(output / "stage27_sequence_summary.csv", videos)
    scenarios = summarize_scenarios(traces)
    write_table(output / "stage27_scenario_summary.csv", scenarios)
    write_table(output / "stage27_failure_events.csv", events, EVENT_FIELDS)
    summary = seal(dict(schema=SCHEMA, role="descriptive_development_failure_audit",
        status="PILOT_INCOMPLETE" if missing else "COMPLETE_DIAGNOSTIC",
        run_fingerprint=run["fingerprint"], manifest_fingerprint=manifest["fingerprint"],
        audited_sequences=len(traces), missing_sequences=missing,
        audited_source_groups=len({e["source_group_proxy"] for e, _ in traces}),
        tracked_frames=len(frames), failure_events=len(events), design=DESIGN,
        bootstrap_samples=5000, bootstrap_seed=2093, bootstrap_unit="source-group-distinct video; one clip/group",
        trace_sha256=hashes, analysis_code_sha256=file_hash(Path(__file__)),
        visibility="unknown; positive boxes do not prove visibility",
        warnings=["Associations are descriptive, not causal attribution.",
                  "Motion and Laplacian variance are proxies; unavailable motion is unknown, not zero.",
                  "Input size is projected GT size, not necessarily visible target pixels.",
                  "Strata may contain different videos; subgroup gaps are not paired treatment effects.",
                  "No compression comparison, search-centering intervention, learning, or speed claim was performed."],
        next_step="Review onset windows and data coverage before choosing a mechanism probe; no automatic method selection"))
    write_json(output / "stage27_diagnostic.json", summary)
    print("Audited {} / {} videos, {} tracked frames, {} low-IoU events.".format(
        len(traces), len(manifest["sequences"]), len(frames), len(events)))
    print("axis                        stratum                              videos    AUC   low-IoU%")
    for row in scenarios:
        if row["axis"] != "inside_size_x_motion":
            print("{axis:<27s} {stratum:<36s} {videos:>6d} {macro_auc:>7.3f} {low:>9.2f}".format(
                **row, low=100 * row["macro_low_iou_fraction"]))
    print("Status: " + summary["status"])
    print("Reports: " + str(output))
    print("Review evidence before assigning causes. No model was trained or selected.")


if __name__ == "__main__":
    main()
