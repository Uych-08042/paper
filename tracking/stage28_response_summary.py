"""Describe candidate retention and window effects on frozen response replays."""

import argparse
import html
import json
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_json
from stage27_uav_common import checked_image, nonempty_path, seal
from stage27_uav_summary import interval, write_table
from stage28_response_common import (DESIGN, SCHEMA, bind_native_top1, dense_boxes,
    load_response, load_sealed, load_source, observation_label, response_metrics, validate_plan)


def describe_frames(video, entry, trace, data, source):
    rows, candidates = [], []
    for index, task in enumerate(video["frames"]):
        frame = task["frame_id"]
        old = frame - 1
        boxes = dense_boxes(data["size"][index], data["offset"][index], data["previous"][index],
                            data["factor"][index], source["search_size"], data["image_hw"][index])
        boxes = bind_native_top1(data["raw"][index], data["window"], boxes, data["native_boxes"][index])
        metrics, details, overlap = response_metrics(data["raw"][index], data["window"], boxes, data["gt"][index])
        gt_side = float(np.sqrt(data["gt"][index, 2:].prod()))
        parent_side = float(np.sqrt(data["previous"][index, 2:].prod()))
        row = dict(sequence=entry["sequence"], source_group_proxy=entry["source_group_proxy"], frame_id=frame,
            event_memberships=json.dumps(task["events"], separators=(",", ":")),
            control_quartiles=json.dumps(task["controls"]),
            is_onset=any(e["offset"] == 0 for e in task["events"]),
            is_pre_onset=any(e["offset"] < 0 for e in task["events"]),
            is_post_onset=any(e["offset"] > 0 for e in task["events"]),
            is_fixed_control=bool(task["controls"]),
            coverage=float(trace["coverage"][old]), raw_gt_side_px=gt_side,
            projected_input_side_px=float(np.sqrt(trace["input_wh"][old].prod())),
            parent_to_gt_side_ratio=parent_side / gt_side, source_iou=float(trace["iou"][old]),
            source_motion_proxy=float(trace["motion"][old, 0]), executed_layer=int(data["layer"][index]),
            replay_box_error=float(data["replay_box_error"][index]),
            replay_peak_error=float(data["replay_peak_error"][index]),
            decode_box_error=float(data["decode_box_error"][index]), **metrics)
        row["observation"] = observation_label(metrics, row["coverage"])
        rows.append(row)
        for kind, detail in details.items():
            for rank, cell in enumerate(detail["indices"], 1):
                candidate = dict(sequence=entry["sequence"], frame_id=frame, ranking=kind, rank=rank,
                    grid_y=cell // data["raw"].shape[1], grid_x=cell % data["raw"].shape[1],
                    score=float(detail["score"][rank - 1]), gt_iou=float(overlap[cell]),
                    label="gt_matching_box" if overlap[cell] >= .5 else "non_target_box" if overlap[cell] < .1 else "ambiguous_box")
                candidate.update({axis: float(value) for axis, value in zip("xywh", boxes[cell])})
                candidates.append(candidate)
    return rows, candidates


def grouped_summary(rows):
    fields = ("raw_top1_iou", "weighted_top1_iou", "raw_top5_gt_recall", "weighted_top5_gt_recall",
              "dense_best_gt_iou", "raw_gt_score_margin", "weighted_gt_score_margin")
    rng, output = np.random.default_rng(2094), []
    masks = dict(all=lambda r: True, event_onsets=lambda r: r["is_onset"],
                 pre_onset=lambda r: r["is_pre_onset"], post_onset=lambda r: r["is_post_onset"],
                 fixed_time_controls=lambda r: r["is_fixed_control"],
                 inside_low_iou=lambda r: r["coverage"] >= .9 and r["weighted_top1_iou"] < .2)
    for phase, select in masks.items():
        selected = [row for row in rows if select(row)]
        for field in fields:
            by_video = {}
            for row in selected:
                value = row[field]
                if value is not None and np.isfinite(value):
                    by_video.setdefault(row["source_group_proxy"], []).append(float(value))
            if not by_video:
                continue
            values = [float(np.mean(value)) for value in by_video.values()]
            low, high = interval(values, rng, 5000)
            output.append(dict(phase=phase, metric=field, source_groups=len(values),
                frames=sum(len(v) for v in by_video.values()), macro_mean=float(np.mean(values)),
                ci_low=low, ci_high=high))
    return output


def scale_temporal_rows(plan, traces):
    rows = []
    for event in plan["events"]:
        entry, trace = traces[event["sequence"]]
        onset = event["onset_frame"]
        first, stop = event["first_frame"], event["stop_frame_exclusive"]
        indices = np.arange(first - 1, stop - 1)
        projected = np.sqrt(trace["input_wh"][indices].prod(1))
        raw_side = np.sqrt(trace["gt"][indices, 2:].prod(1))
        raw_before = np.sqrt(np.asarray(entry["ground_truth"])[indices, 2:].prod(1))
        parent = np.sqrt(trace["previous"][indices, 2:].prod(1))
        small = indices[projected < 32]
        coverage_drop = indices[trace["coverage"][indices] < .9]
        start = onset - 1
        gt_side = float(np.sqrt(trace["gt"][start, 2:].prod()))
        row = dict(sequence=event["sequence"], event=event["event"], onset_frame=onset,
            onset_raw_gt_side_px=gt_side,
            onset_projected_input_side_px=float(np.sqrt(trace["input_wh"][start].prod())),
            onset_parent_to_gt_side_ratio=float(np.sqrt(trace["previous"][start, 2:].prod()) / gt_side),
            first_projected_lt32_offset=None if not len(small) else int(small[0] + 1 - onset),
            first_coverage_lt09_offset=None if not len(coverage_drop) else int(coverage_drop[0] + 1 - onset),
            review_raw_gt_side_min=float(raw_side.min()), review_raw_gt_side_max=float(raw_side.max()),
            review_parent_to_gt_side_ratio_max=float((parent / raw_side).max()),
            review_true_gt_side_step_ratio_min=float((raw_side / raw_before).min()),
            note="observed time ordering only; no causal or visibility label")
        rows.append(row)
    return rows


def all_source_size_summary(traces):
    groups = {}
    for entry, data in traces.values():
        raw = np.sqrt(data["gt"][:, 2:].prod(1))
        ratio = np.sqrt(data["previous"][:, 2:].prod(1)) / raw
        # Raw size and crop-scale mismatch are kept separate. Never call projected size raw resolution.
        for axis, values, edges, labels in (
            ("raw_gt_side_px", raw, [16, 32, 64], ["lt16", "16to32", "32to64", "ge64"]),
            ("parent_to_current_gt_side_ratio", ratio, [.5, 2], ["lt0.5", "0.5to2", "ge2"])):
            category = np.searchsorted(edges, values, side="right")
            for i, label in enumerate(labels):
                mask = category == i
                if mask.any():
                    groups.setdefault((axis, label), []).append((int(mask.sum()), int((data["iou"][mask] < .2).sum())))
    return [dict(axis=axis, stratum=label, videos=len(values), frames=sum(v[0] for v in values),
                 low_iou_frames=sum(v[1] for v in values),
                 macro_low_iou_fraction=float(np.mean([v[1] / v[0] for v in values])))
            for (axis, label), values in sorted(groups.items())]


def render_event(source, entry, trace, data, event, output, image_reader=checked_image):
    from PIL import Image, ImageDraw
    from stage27_uav_review import GREEN, RED, place_view, review_font, search_patch
    chosen = [event["onset_frame"] + offset for offset in (-1, 0, 1, 10)
              if event["onset_frame"] + offset in set(data["frame_id"].tolist())]
    sheet = Image.new("RGB", (960, 380 * len(chosen)), "white")
    font = review_font()
    side = data["raw"].shape[1]
    for r, frame in enumerate(chosen):
        index = int(np.flatnonzero(data["frame_id"] == frame)[0])
        old = frame - 1
        image = image_reader(source["manifest"]["root"], entry["frames"][frame])
        crop, usable = trace["crop"][old], trace["usable"][old]
        patch = search_patch(image, crop, usable)
        boxes = dense_boxes(data["size"][index], data["offset"][index], data["previous"][index],
                            data["factor"][index], source["search_size"], data["image_hw"][index])
        boxes = bind_native_top1(data["raw"][index], data["window"], boxes, data["native_boxes"][index])
        metrics, details, ious = response_metrics(data["raw"][index], data["window"], boxes, data["gt"][index])
        tile = Image.new("RGB", (960, 380), "white")
        draw = ImageDraw.Draw(tile)
        draw.text((10, 6), "{} frame {} onset {:+d}; native IoU {:.3f}; raw {:.3f}; dense best {:.3f}".format(
            entry["sequence"], frame, frame - event["onset_frame"], metrics["weighted_top1_iou"],
            metrics["raw_top1_iou"], metrics["dense_best_gt_iou"]), fill="black", font=font)
        colored = [(data["gt"][index], GREEN), (data["native_boxes"][index], RED)]
        relative = [(np.r_[b[:2] - crop[:2], b[2:]], color) for b, color in colored]
        place_view(tile, patch, (10, 55, 300, 280), relative)
        draw.text((10, 344), "GT green; native red (display view)", fill="black", font=font)
        for column, kind in enumerate(("raw", "weighted"), 1):
            response = data["raw"][index] * (1 if kind == "raw" else data["window"])
            value = (response - response.min()) / max(1e-12, float(response.max() - response.min()))
            colors = np.stack([value, np.sqrt(value), 1 - value], axis=-1)
            heat = Image.fromarray((colors * 255).astype(np.uint8)).resize((280, 280),
                getattr(Image, "Resampling", Image).NEAREST)
            hd = ImageDraw.Draw(heat)
            for cell in np.flatnonzero(ious >= .5):
                y, x = divmod(int(cell), side)
                hd.rectangle((x * 280 / side, y * 280 / side, (x + 1) * 280 / side - 1,
                              (y + 1) * 280 / side - 1), outline=GREEN, width=2)
            for rank, cell in enumerate(details[kind]["indices"], 1):
                y, x = divmod(cell, side)
                hd.text((x * 280 / side + 2, y * 280 / side + 1), str(rank),
                        fill=RED if rank == 1 else "white", stroke_width=1, stroke_fill="black", font=font)
            left = column * 320 + 20
            tile.paste(heat, (left, 55))
            draw.text((left, 31), kind + ": own color normalization", fill="black", font=font)
            margin = metrics[kind + "_gt_score_margin"]
            draw.text((left, 344), "GT margin {}; recall@5 {}".format(
                "n/a" if margin is None else "{:.4f}".format(margin), metrics[kind + "_top5_gt_recall"]),
                fill="black", font=font)
        sheet.paste(tile, (0, 380 * r))
    path = output / "heatmaps" / (entry["sequence"] + "__e{:03d}.png".format(event["event"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True)
    parser.add_argument("--allow_incomplete", action="store_true")
    parser.add_argument("--render", action="store_true", help="CPU heatmaps and crop views; requires Pillow/OpenCV")
    args = parser.parse_args()
    directory = Path(args.input).expanduser().resolve()
    run = load_sealed(directory / "run.json")
    if run.get("role") != "frozen_response_replay":
        raise ValueError("Expected a Stage-28 response replay run")
    plan = load_sealed(directory / "plan.json")
    source_info = load_source(plan["source_input"])
    validate_plan(plan, source_info)
    if run["plan"] != plan:
        raise ValueError("Run and plan differ")
    _, source, traces, _ = source_info
    rows, candidates, missing, caches, hashes = [], [], [], {}, {}
    expected = {v["sequence"] + "__stage28_response.npz" for v in plan["videos"]}
    extras = {p.name for p in (directory / "responses").glob("*.npz")} - expected
    if extras:
        raise ValueError("Unexpected response files: " + str(extras))
    for video in plan["videos"]:
        name = video["sequence"]
        path = directory / "responses" / (name + "__stage28_response.npz")
        if not path.is_file():
            missing.append(name)
            continue
        entry, trace = traces[name]
        data = load_response(path, entry, trace, video, source, run)
        np.testing.assert_array_equal(data["window"], np.asarray(run["window"], np.float32))
        frame_rows, peak_rows = describe_frames(video, entry, trace, data, source)
        rows.extend(frame_rows)
        candidates.extend(peak_rows)
        caches[name] = data
        hashes[path.name] = file_hash(path)
    if not rows or (missing and not args.allow_incomplete):
        raise ValueError("{} videos collected, {} missing; pilot only: --allow_incomplete".format(len(caches), len(missing)))
    output = directory / ("analysis_pilot" if args.allow_incomplete else "analysis")
    write_table(output / "stage28_frame_responses.csv", rows)
    write_table(output / "stage28_candidate_boxes.csv", candidates)
    summaries = grouped_summary(rows)
    write_table(output / "stage28_phase_summary.csv", summaries)
    write_table(output / "stage28_source_raw_size_summary.csv", all_source_size_summary(traces))
    temporal = scale_temporal_rows(plan, traces)
    write_table(output / "stage28_scale_temporal.csv", temporal,
                None if temporal else ["sequence", "event", "onset_frame"])
    onsets = [row for row in rows if row["is_onset"]]
    write_table(output / "stage28_onset_responses.csv", onsets, list(rows[0]))
    observations = []
    for label in sorted({row["observation"] for row in onsets}):
        selected = [row for row in onsets if row["observation"] == label]
        observations.append(dict(observation=label, events=len(selected),
                                 source_groups=len({row["source_group_proxy"] for row in selected})))
    write_table(output / "stage28_onset_observations.csv", observations,
                ["observation", "events", "source_groups"])
    pictures = []
    if args.render:
        for event in plan["events"]:
            name = event["sequence"]
            if name in caches:
                entry, trace = traces[name]
                path = render_event(source, entry, trace, caches[name], event, output)
                pictures.append(dict(file=path.relative_to(output).as_posix(), sha256=file_hash(path)))
                print("[heatmap] " + path.name, flush=True)
        document = ["<!doctype html><meta charset='utf-8'><title>Response competition audit</title>",
                    "<p>All response ranks are diagnostics. Green grid cells decode to GT IoU>=0.5. "
                    "Colors are normalized separately in each map. No counterfactual output advances tracking.</p>"]
        document.extend('<p>{}</p><img style="max-width:100%" src="{}">'.format(
            html.escape(p["file"]), html.escape(p["file"], quote=True)) for p in pictures)
        (output / "index.html").write_text("\n".join(document), encoding="utf-8")
    report = seal(dict(schema=SCHEMA, role="descriptive_response_competition_audit",
        status="PILOT_INCOMPLETE" if missing else "COMPLETE_RESPONSE_DIAGNOSTIC",
        source_run_fingerprint=source["fingerprint"], plan_fingerprint=plan["fingerprint"],
        audited_videos=len(caches), missing_videos=missing, replayed_unique_frames=len(rows),
        audited_event_onsets=len(onsets), planned_event_onsets=len(plan["events"]),
        max_replay_box_error=max(row["replay_box_error"] for row in rows),
        max_replay_score_error=max(row["replay_peak_error"] for row in rows),
        max_dense_decode_error=max(row["decode_box_error"] for row in rows),
        cache_sha256=hashes, pictures=pictures, analysis_code_sha256=file_hash(Path(__file__)),
        bootstrap="5000 draws, seed 2094, source-group macro; phase overlap and outcome conditioning remain",
        warnings=["Top-5 retention depends on the fixed local-peak and box-NMS rules.",
                  "Dense grid GT match is a hindsight localization ceiling, not a policy.",
                  "Non-target decoded boxes are not automatically semantic distractors.",
                  "Raw argmax is a same-state decode diagnostic, not an evaluated tracking policy.",
                  "Projected input size is not raw image resolution; scale time ordering is descriptive."],
        training_performed=False, causal_gain_established=False,
        next_step="Review candidate retention, margins, decoder effects and raw scale before method selection"))
    write_json(output / "stage28_response_diagnostic.json", report)
    print("Audited {} / {} videos, {} unique replay frames, {} event onsets.".format(
        len(caches), len(plan["videos"]), len(rows), len(onsets)))
    print("Replay max box error: {:.6g}; dense decode error: {:.6g}".format(
        report["max_replay_box_error"], report["max_dense_decode_error"]))
    for row in summaries:
        if row["phase"] == "event_onsets":
            print("onsets {:<30s} {:.6f} ({} source groups)".format(
                row["metric"], row["macro_mean"], row["source_groups"]))
    for row in observations:
        print("{}: {} event(s)".format(row["observation"], row["events"]))
    print("Status: " + report["status"])
    print("Reports: " + str(output))


if __name__ == "__main__":
    main()
