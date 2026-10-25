"""Describe target-candidate retention at the corrected branch's first bad output."""

import argparse
import html
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_json
from stage27_uav_common import nonempty_path, scene_geometry, seal, tracking_iou
from stage27_uav_summary import write_table
from stage28_response_common import bind_native_top1, dense_boxes, observation_label, response_metrics
from stage30_branch_response_common import (SCHEMA, load_inputs, load_point, load_sealed,
    point_filename, point_source, validate_plan)


def describe_point(point, data, source):
    boxes = dense_boxes(data["size"], data["offset"], data["previous"], float(data["factor"]),
                        source["search_size"], data["image_hw"])
    boxes = bind_native_top1(data["raw"], data["window"], boxes, data["native_box"])
    metrics, details, overlaps = response_metrics(data["raw"], data["window"], boxes, data["gt"])
    gt, previous, predicted = data["gt"], data["previous"], data["native_box"]
    geometry = scene_geometry(previous, gt, data["image_hw"], source["search_factor"], source["search_size"])
    center_error = np.linalg.norm(((predicted[:2] + predicted[2:] / 2) - (gt[:2] + gt[2:] / 2)) / gt[2:])
    row = dict(sequence=point["sequence"], event=point["event"], frame_id=point["frame_id"],
        branch_rank=point["branch_rank"], branch_offset=point["branch_offset"], purpose=point["purpose"],
        output_was_overridden=bool(data["output_was_overridden"]),
        recorded_branch_iou=float(tracking_iou(data["recorded_branch_box"], gt)),
        model_top1_iou=metrics["weighted_top1_iou"], coverage=geometry["coverage"],
        raw_gt_side_px=float(np.sqrt(gt[2:].prod())),
        projected_gt_side_px=float(np.sqrt(geometry["input_wh"].prod())),
        parent_to_gt_side_ratio=float(np.sqrt(previous[2:].prod() / gt[2:].prod())),
        model_center_error_gt_units=float(center_error), model_width_ratio=float(predicted[2] / gt[2]),
        model_height_ratio=float(predicted[3] / gt[3]), executed_layer=int(data["layer"]),
        replay_box_error=float(data["replay_box_error"]), replay_peak_error=float(data["replay_peak_error"]),
        decode_box_error=float(data["decode_box_error"]), **metrics)
    row["observation"] = observation_label(metrics, geometry["coverage"])
    rows = []
    for kind, detail in details.items():
        for rank, cell in enumerate(detail["indices"], 1):
            r = dict(sequence=point["sequence"], event=point["event"], frame_id=point["frame_id"],
                branch_rank=point["branch_rank"], branch_offset=point["branch_offset"], ranking=kind, rank=rank,
                grid_y=cell // data["raw"].shape[1], grid_x=cell % data["raw"].shape[1],
                score=float(detail["score"][rank - 1]), gt_iou=float(overlaps[cell]),
                label="gt_matching_box" if overlaps[cell] >= .5 else "non_target_box" if overlaps[cell] < .1 else "ambiguous_box")
            r.update({k: float(v) for k, v in zip("xywh", boxes[cell])})
            rows.append(r)
    return row, rows, details, overlaps


def render_point(point, data, inputs, output, image_reader=None):
    from PIL import Image, ImageDraw
    from stage27_uav_common import checked_image
    from stage27_uav_review import GREEN, RED, place_view, review_font, search_patch
    reader = image_reader or checked_image
    entry, _, _ = point_source(inputs, point)
    image = reader(inputs["source"]["manifest"]["root"], entry["frames"][point["frame_id"]])
    patch = search_patch(image, data["crop"], data["usable"])
    row, _, details, overlaps = describe_point(point, data, inputs["source"])
    sheet = Image.new("RGB", (1200, 850), "white")
    draw, font = ImageDraw.Draw(sheet), review_font()
    draw.text((10, 5), "{} frame {} | corrected rank {} +{} | {}".format(point["sequence"],
        point["frame_id"], point["branch_rank"], point["branch_offset"], point["purpose"]), fill="black", font=font)
    draw.text((10, 25), "Model IoU {:.3f}; recorded branch IoU {:.3f}; coverage {:.3f}; Stage-29 override at this point: {}".format(
        row["model_top1_iou"], row["recorded_branch_iou"], row["coverage"], row["output_was_overridden"]), fill="black", font=font)
    colored = [(data["gt"], GREEN), (data["native_box"], RED)]
    if bool(data["output_was_overridden"]):
        colored.append((data["recorded_branch_box"], (255, 165, 0)))
    place_view(sheet, image, (10, 65, 570, 300), colored)
    relative = [(np.r_[b[:2] - data["crop"][:2], b[2:]], color) for b, color in colored]
    place_view(sheet, patch, (610, 65, 570, 300), relative)
    draw.text((10, 375), "Full image | green GT; red native model; orange injected output if present", fill="black", font=font)
    draw.text((610, 375), "Actual corrected-branch search crop (display resize)", fill="black", font=font)
    side = data["raw"].shape[1]
    for left, kind in ((75, "raw"), (650, "weighted")):
        response = data["raw"] * (1 if kind == "raw" else data["window"])
        value = (response - response.min()) / max(1e-12, float(response.max() - response.min()))
        colors = np.stack([value, np.sqrt(value), 1 - value], axis=-1)
        heat = Image.fromarray((colors * 255).astype(np.uint8)).resize((360, 360),
            getattr(Image, "Resampling", Image).NEAREST)
        hd = ImageDraw.Draw(heat)
        for cell in np.flatnonzero(overlaps >= .5):
            y, x = divmod(int(cell), side)
            hd.rectangle((x * 360 / side, y * 360 / side, (x + 1) * 360 / side - 1,
                          (y + 1) * 360 / side - 1), outline=GREEN, width=2)
        for rank, cell in enumerate(details[kind]["indices"], 1):
            y, x = divmod(cell, side)
            hd.text((x * 360 / side + 2, y * 360 / side + 1), str(rank),
                    fill=RED if rank == 1 else "white", stroke_width=1, stroke_fill="black", font=font)
        sheet.paste(heat, (left, 435))
        draw.text((left, 412), kind + " (independent color normalization)", fill="black", font=font)
        draw.text((left, 805), "GT recall@5 {}; dense best IoU {:.3f}".format(
            row[kind + "_top5_gt_recall"], row["dense_best_gt_iou"]), fill="black", font=font)
    draw.text((10, 830), "Green heatmap cells decode to GT IoU>=0.5. Rankings are diagnostics; non-target boxes need visual review.",
              fill="black", font=font)
    path = output / "figures" / (point_filename(point).replace(".npz", ".png"))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True)
    parser.add_argument("--allow_incomplete", action="store_true")
    parser.add_argument("--render", action="store_true", help="CPU crop/full-image overlays and response maps; Pillow/OpenCV")
    args = parser.parse_args()
    directory = Path(args.input).expanduser().resolve()
    plan, run = load_sealed(directory / "plan.json"), load_sealed(directory / "run.json")
    inputs = load_inputs(plan["source_input"])
    validate_plan(plan, inputs)
    if run.get("role") != "corrected_branch_single_frame_response_replay" or run.get("plan") != plan:
        raise ValueError("Stage-30 run and plan differ")
    expected = {point_filename(p) for p in plan["points"]}
    extras = {p.name for p in (directory / "responses").glob("*.npz")} - expected
    if extras:
        raise ValueError("Unexpected response files: " + str(extras))
    rows, candidates, missing, hashes, saved = [], [], [], {}, []
    for point in plan["points"]:
        path = directory / "responses" / point_filename(point)
        if not path.is_file():
            missing.append(path.name)
            continue
        data = load_point(path, point, inputs, run)
        row, boxes, _, _ = describe_point(point, data, inputs["source"])
        rows.append(row)
        candidates.extend(boxes)
        hashes[path.name] = file_hash(path)
        saved.append((point, data))
    if not rows or (missing and not args.allow_incomplete):
        raise ValueError("{} points collected, {} missing; engineering pilot only: --allow_incomplete".format(len(rows), len(missing)))
    output = directory / ("analysis_pilot" if args.allow_incomplete else "analysis")
    write_table(output / "stage30_point_responses.csv", rows)
    critical = [r for r in rows if r["purpose"] == "first_low_iou"]
    write_table(output / "stage30_failure_responses.csv", critical, list(rows[0]))
    write_table(output / "stage30_candidate_boxes.csv", candidates)
    pictures = []
    if args.render:
        for point, data in saved:
            path = render_point(point, data, inputs, output)
            pictures.append(dict(file=path.relative_to(output).as_posix(), sha256=file_hash(path)))
            print("[figure] " + path.name, flush=True)
        parts = ["<!doctype html><meta charset='utf-8'><title>Corrected branch response audit</title>",
                 "<p>Green: GT. Red: native model output. Orange: the historical injected output, only at offset 0. "
                 "Response colors are normalized independently; candidate ranks are offline diagnostics.</p>"]
        parts.extend('<p>{}</p><img style="max-width:100%" src="{}">'.format(html.escape(p["file"]),
                     html.escape(p["file"], quote=True)) for p in pictures)
        (output / "index.html").write_text("\n".join(parts), encoding="utf-8")
    report = seal(dict(schema=SCHEMA, role="descriptive_corrected_branch_response_audit",
        status="PILOT_INCOMPLETE" if missing else "COMPLETE_CORRECTED_BRANCH_RESPONSE_AUDIT",
        plan_fingerprint=plan["fingerprint"], source_stage29_decision_fingerprint=inputs["report"]["fingerprint"],
        source_stage29_status=inputs["report"]["status"], audited_points=len(rows), expected_points=len(plan["points"]),
        missing_points=missing, audited_failure_points=len(critical),
        failure_observations=[dict(sequence=r["sequence"], frame_id=r["frame_id"],
            branch_offset=r["branch_offset"], model_iou=r["model_top1_iou"], coverage=r["coverage"],
            dense_best_gt_iou=r["dense_best_gt_iou"], weighted_top5_gt_recall=r["weighted_top5_gt_recall"],
            observation=r["observation"]) for r in critical],
        max_replay_box_error=max(r["replay_box_error"] for r in rows),
        max_replay_score_error=max(r["replay_peak_error"] for r in rows),
        max_dense_decode_error=max(r["decode_box_error"] for r in rows),
        cache_sha256=hashes, pictures=pictures, analysis_code_sha256=file_hash(Path(__file__)),
        training_performed=False, new_policy_evaluated=False, causal_gain_established=False,
        warnings=["These two failure frames were selected after the consumed development outcomes.",
            "GT coverage does not establish visibility. A low-IoU box does not identify a semantic distractor.",
            "At offset 0 the native output precedes the Stage-29 override; it need not match the injected box.",
            "Top-5 retention depends on the unchanged local-peak/NMS rule; all-cell matching is a hindsight ceiling.",
            "No new rollout, method pass/fail gate, speed claim or external generalization test is performed."],
        next_step="Review corrected-crop overlays and target retention before choosing identity association or localization work"))
    write_json(output / "stage30_response_diagnostic.json", report)
    print("Audited {} / {} points; {} first-low-IoU points.".format(len(rows), len(plan["points"]), len(critical)))
    for r in critical:
        print("{} frame {} +{}: model IoU {:.4f}; dense best {:.4f}; weighted recall@5 {}; {}".format(
            r["sequence"], r["frame_id"], r["branch_offset"], r["model_top1_iou"], r["dense_best_gt_iou"],
            r["weighted_top5_gt_recall"], r["observation"]))
    print("Status: " + report["status"])
    print("Reports: " + str(output))


if __name__ == "__main__":
    main()
