"""Read-only onset review of a completed Stage-27 run; no tracker inference."""

import argparse
import csv
import html
import json
import math
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_json
from stage27_uav_common import (SCHEMA, checked_image, coverage_stratum, failure_events,
    load_manifest, load_sealed, load_trace, nonempty_path, publish_identical,
    seal, success_auc)
from stage27_uav_summary import write_table


OFFSETS = (-10, -3, -1, 0, 1, 3, 10, 30)
GREEN, RED, BLUE = (0, 150, 40), (225, 35, 35), (35, 100, 230)
LABEL_FIELDS = ("sequence", "event", "onset_frame", "visibility", "mechanism",
                "review_confidence", "reviewer", "notes")
VISIBILITY = ("unreviewed", "visible", "partial_occlusion", "full_occlusion", "out_of_view", "uncertain")
MECHANISMS = ("unreviewed", "search_exclusion", "localization_shift", "scale_error",
              "distractor_switch", "occlusion_associated", "annotation_issue", "uncertain")


def box_errors(predicted, gt):
    predicted, gt = np.asarray(predicted, float), np.asarray(gt, float)
    if (predicted.shape != (4,) or gt.shape != (4,) or not np.isfinite([predicted, gt]).all()
            or min(*predicted[2:], *gt[2:]) <= 0):
        raise ValueError("Expected finite positive xywh boxes")
    delta = predicted[:2] + predicted[2:] / 2 - gt[:2] - gt[2:] / 2
    return dict(center_error_gt_units=float(np.linalg.norm(delta) / np.sqrt(gt[2:].prod())),
                width_ratio=float(predicted[2] / gt[2]), height_ratio=float(predicted[3] / gt[3]),
                area_ratio=float(predicted[2:].prod() / gt[2:].prod()))


def sample_frames(onset, count):
    if not 1 <= onset < count:
        raise ValueError("Invalid tracked-frame onset")
    return sorted({onset + offset for offset in OFFSETS if 0 <= onset + offset < count})


def finite_stat(values, operation):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(operation(values)) if len(values) else None


def describe_event(entry, data, event, number):
    start = event["start"]
    frame = int(data["frame_id"][start])
    begin = max(0, start - 10)
    before = data["motion"][begin:start, 0]
    known = before[np.isfinite(before)]
    gt = data["gt"][start]
    errors = box_errors(data["boxes"][start], gt)
    row = dict(sequence=entry["sequence"], source_group_proxy=entry["source_group_proxy"],
        event=number, onset_frame=frame, confirmed_frame=int(data["frame_id"][event["confirmation"]]),
        onset_source_file=entry["frames"][frame]["file"],
        recovery_frame=None if event["recovery"] is None else int(data["frame_id"][event["recovery"]]),
        censor=event["censor"], previous_iou=None if start == 0 else float(data["iou"][start - 1]),
        onset_iou=float(data["iou"][start]), coverage=float(data["coverage"][start]),
        coverage_stratum=coverage_stratum(data["coverage"][start]),
        raw_gt_width_px=float(gt[2]), raw_gt_height_px=float(gt[3]),
        raw_gt_side_px=float(np.sqrt(gt[2:].prod())),
        projected_input_side_px=float(np.sqrt(data["input_wh"][start].prod())),
        gt_image_fraction=float(data["gt_image_fraction"][start]),
        **errors, executed_layer=int(data["layer"][start]),
        previous_executed_layer=None if start == 0 else int(data["layer"][start - 1]),
        raw_score_peak=float(data["raw_score_peak"][start]),
        previous_raw_score_peak=None if start == 0 else float(data["raw_score_peak"][start - 1]),
        onset_background_motion_proxy=float(data["motion"][start, 0]),
        previous_10_motion_known_fraction=None if start == 0 else float(np.isfinite(before).mean()),
        previous_10_background_motion_max=finite_stat(before, np.max),
        previous_10_high_motion_fraction=None if not len(known) else float((known >= .02).mean()),
        onset_target_residual_motion_proxy=float(data["motion"][start, 1]),
        previous_10_target_residual_max=finite_stat(data["motion"][begin:start, 1], np.max),
        event_low_iou_frames=int((data["iou"][start:event["end"] + 1] < .2).sum()),
        contact_sheet="contacts/{}__e{:03d}.png".format(entry["sequence"], number),
        manual_cause="unreviewed", visibility="unknown")
    return row


def aggregate(traces, rows):
    counts = [int((data["iou"] < .2).sum()) for _, data in traces]
    frames = sum(len(data["iou"]) for _, data in traces)
    total_low = sum(counts)
    top = sorted(zip(traces, counts), key=lambda item: (-item[1], item[0][0]["sequence"]))[:4]
    return dict(videos=len(traces), tracked_frames=frames,
        macro_auc=float(np.mean([success_auc(data["iou"]) for _, data in traces])),
        low_iou_frames=total_low, frame_weighted_low_iou_fraction=total_low / frames,
        video_macro_low_iou_fraction=float(np.mean([count / len(data["iou"])
                                                   for (_, data), count in zip(traces, counts)])),
        videos_with_any_low_iou=sum(count > 0 for count in counts), events=len(rows),
        events_with_coverage_ge_09=sum(row["coverage"] >= .9 for row in rows),
        events_with_low_onset_motion=sum(bool(np.isfinite(row["onset_background_motion_proxy"]))
                                        and row["onset_background_motion_proxy"] < .005 for row in rows),
        events_with_projected_size_ge_32=sum(row["projected_input_side_px"] >= 32 for row in rows),
        events_censored_at_video_end=sum(row["censor"] == "video_end" for row in rows),
        top_four_videos=[dict(sequence=item[0][0]["sequence"], low_iou_frames=item[1]) for item in top],
        top_four_low_frame_share=None if not total_low else sum(item[1] for item in top) / total_low)


def search_patch(image, crop, usable, max_pixels=100_000_000):
    """Reconstruct the recorded integer crop before display resizing."""
    x, y, width, height = map(int, crop)
    sx, sy, sw, sh = map(int, usable)
    if min(width, height) <= 0 or width * height > max_pixels:
        raise ValueError("Invalid/oversized review crop")
    patch = np.zeros((height, width, 3), dtype=np.uint8)
    if sw and sh:
        if not (0 <= sx < sx + sw <= image.shape[1] and 0 <= sy < sy + sh <= image.shape[0]
                and 0 <= sx - x <= width - sw and 0 <= sy - y <= height - sh):
            raise ValueError("Usable crop is inconsistent with the image")
        patch[sy - y:sy - y + sh, sx - x:sx - x + sw] = image[sy:sy + sh, sx:sx + sw]
    return patch


def review_font():
    from PIL import ImageFont
    for name in ("DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, 14)
        except OSError:
            pass
    return ImageFont.load_default()


def place_view(canvas, rgb, rectangle, boxes):
    from PIL import Image, ImageDraw
    x, y, width, height = rectangle
    scale = min(width / rgb.shape[1], height / rgb.shape[0])
    resized = Image.fromarray(rgb).resize((max(1, round(rgb.shape[1] * scale)),
                                           max(1, round(rgb.shape[0] * scale))),
                                          getattr(Image, "Resampling", Image).BILINEAR)
    left, top = x + (width - resized.width) // 2, y + (height - resized.height) // 2
    view = Image.new("RGB", (width, height), (35, 35, 35))
    view.paste(resized, (left - x, top - y))
    draw = ImageDraw.Draw(view)
    # Draw on the individual viewport so out-of-crop boxes cannot leak into text/neighbour panels.
    for box, color in boxes:
        bx, by, bw, bh = np.asarray(box, float)
        coordinates = [left - x + bx * scale, top - y + by * scale,
                       left - x + (bx + bw) * scale, top - y + (by + bh) * scale]
        draw.rectangle(coordinates, outline=color, width=2)
    canvas.paste(view, (x, y))


def render_contact(root, entry, data, row, destination, reader=checked_image):
    from PIL import Image, ImageDraw
    frames = sample_frames(row["onset_frame"], len(entry["frames"]))
    width, height = 920, 440
    sheet = Image.new("RGB", (width * 2, height * math.ceil(len(frames) / 2)), "white")
    font = review_font()
    for position, frame_id in enumerate(frames):
        image = reader(root, entry["frames"][frame_id])
        tile = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(tile)
        gt = np.asarray(entry["ground_truth"][frame_id], float)
        if frame_id == 0:
            box, crop, usable = gt, None, None
            detail, extra = "INITIAL GT: no tracked prediction", ""
        else:
            index = frame_id - 1
            box, crop, usable = (data[key][index] for key in ("boxes", "crop", "usable"))
            metrics = box_errors(box, gt)
            detail = "IoU={:.3f} coverage={:.3f} raw GT={:.0f}x{:.0f}px projected side={:.1f}px".format(
                data["iou"][index], data["coverage"][index], gt[2], gt[3], np.sqrt(data["input_wh"][index].prod()))
            extra = "center error={:.2f} GT units; width x{:.2f}, height x{:.2f}; motion={:.5f}".format(
                metrics["center_error_gt_units"], metrics["width_ratio"], metrics["height_ratio"],
                data["motion"][index, 0])
        draw.text((12, 8), "{} | frame {} ({}) | onset {:+d}".format(
            entry["sequence"], frame_id, Path(entry["frames"][frame_id]["file"]).name,
            frame_id - row["onset_frame"]), fill="black", font=font)
        draw.text((12, 29), detail, fill="black", font=font)
        draw.text((12, 50), extra, fill="black", font=font)
        overview = [(gt, GREEN), (box, RED)] + ([] if crop is None else [(crop, BLUE)])
        place_view(tile, image, (12, 85, 610, 310), overview)
        if crop is not None:
            patch = search_patch(image, crop, usable)
            relative = [(np.r_[b[:2] - crop[:2], b[2:]], color) for b, color in ((gt, GREEN), (box, RED))]
            place_view(tile, patch, (640, 85, 268, 310), relative)
        draw.text((12, 410), "Full frame: green=GT, red=prediction, blue=search", fill="black", font=font)
        draw.text((640, 410), "Search view (display resize)", fill="black", font=font)
        sheet.paste(tile, ((position % 2) * width, (position // 2) * height))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="PNG")


def manual_template(path, rows):
    identities = [(r["sequence"], str(r["event"]), str(r["onset_frame"])) for r in rows]
    path = Path(path)
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            existing = list(reader)
            if reader.fieldnames != list(LABEL_FIELDS):
                raise ValueError("Manual-label header changed: " + str(path))
        if [(r["sequence"], r["event"], r["onset_frame"]) for r in existing] != identities:
            raise ValueError("Manual labels do not match these events; existing file preserved")
        return
    entries = [dict(sequence=r["sequence"], event=r["event"], onset_frame=r["onset_frame"],
                    visibility="unreviewed", mechanism="unreviewed", review_confidence="", reviewer="", notes="")
               for r in rows]
    write_table(path, entries, LABEL_FIELDS)


def build_review(input_dir, output_dir, tables_only=False, reader=checked_image):
    directory, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if output == directory or output == directory / "analysis" or output == directory / "traces":
        raise ValueError("Use a separate review output directory")
    run = load_sealed(directory / "run.json")
    if run.get("role") != "original_sgla_descriptive_uav_diagnosis":
        raise ValueError("Expected an original-SGLA Stage-27 run")
    manifest = load_manifest(run["manifest"])
    traces, rows, hashes = [], [], {}
    expected = {e["sequence"] + "__stage27_uav.npz" for e in manifest["sequences"]}
    found = {p.name for p in (directory / "traces").glob("*.npz")}
    if expected != found:
        raise ValueError("Review requires exactly the complete frozen traces; missing={}, extra={}".format(
            sorted(expected - found), sorted(found - expected)))
    for entry in manifest["sequences"]:
        path = directory / "traces" / (entry["sequence"] + "__stage27_uav.npz")
        data = load_trace(path, entry, run)
        hashes[path.name] = file_hash(path)
        traces.append((entry, data))
        rows.extend(describe_event(entry, data, event, number)
                    for number, event in enumerate(failure_events(data["iou"]), 1))
    # Newly derived review artifacts only. The frozen manifest/traces/analysis are never edited.
    contract = seal(dict(schema=SCHEMA, role="read_only_event_onset_review", source_run_fingerprint=run["fingerprint"],
        source_manifest_fingerprint=manifest["fingerprint"], source_trace_sha256=hashes,
        source_run_sha256=file_hash(directory / "run.json"), tables_only=bool(tables_only),
        code_sha256=file_hash(Path(__file__)), offsets=list(OFFSETS), events=len(rows),
        rendering="Pillow display interpolation, not inference; original input tensor is not regenerated",
        selection="all frozen low-IoU events; no top-case selection",
        allowed_visibility=list(VISIBILITY), allowed_mechanisms=list(MECHANISMS)))
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "review_run.json").exists() and any(output.iterdir()):
        raise ValueError("Use a new empty review directory; existing files were preserved")
    publish_identical(output / "review_run.json", contract)
    summary = aggregate(traces, rows)
    # Seal metadata cannot contain NaN, although proxy fields in the CSV may be unavailable.
    summary = seal(dict(schema=SCHEMA, role="descriptive_event_review_summary",
        source_review_fingerprint=contract["fingerprint"], **summary,
        warning="Event onset associations and visual labels do not establish a deployable intervention or causal gain"))
    write_json(output / "review_summary.json", summary)
    fields = list(rows[0]) if rows else ["sequence", "event", "onset_frame", "manual_cause", "visibility"]
    write_table(output / "stage27_event_onset_diagnostic.csv", rows, fields)
    manual_template(output / "manual_review.csv", rows)
    pictures = []
    if not tables_only:
        for entry, data in traces:
            for row in (r for r in rows if r["sequence"] == entry["sequence"]):
                destination = output / row["contact_sheet"]
                render_contact(manifest["root"], entry, data, row, destination, reader)
                pictures.append(dict(file=row["contact_sheet"], sha256=file_hash(destination)))
                print("[review image] {} event {}".format(entry["sequence"], row["event"]), flush=True)
    document = ["<!doctype html><meta charset='utf-8'><title>Stage-27 onset review</title>",
        "<style>body{font:16px sans-serif;max-width:1900px;margin:24px auto}img{width:100%}section{margin:32px 0}</style>",
        "<h1>Stage-27 event onset review</h1>",
        "<p>Green: GT. Red: prediction. Blue: recorded search region. All event causes remain unreviewed. "
        "Coverage does not imply visibility. Display resizing is not an inference operation.</p>",
        "<p>Edit manual_review.csv. Preserve unknown/uncertain cases; do not infer identity swaps from IoU alone.</p>"]
    for row in rows:
        document.append("<section><h2>{} / event {} / onset {}</h2>".format(
            html.escape(row["sequence"]), row["event"], row["onset_frame"]))
        if not tables_only:
            document.append('<img loading="lazy" src="{}" alt="Event onset contact sheet">'.format(html.escape(row["contact_sheet"], quote=True)))
        else:
            document.append("<p>Tables-only mode: images not rendered.</p>")
        document.append("</section>")
    (output / "index.html").write_text("\n".join(document), encoding="utf-8")
    write_json(output / "review_outputs.json", seal(dict(schema=SCHEMA,
        role="event_review_outputs", review_fingerprint=contract["fingerprint"], pictures=pictures)))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True)
    parser.add_argument("--output_dir", type=nonempty_path, required=True)
    parser.add_argument("--tables_only", action="store_true", help="No image decoding or Pillow dependency")
    args = parser.parse_args()
    summary = build_review(args.input, args.output_dir, args.tables_only)
    print("Reviewed {} videos / {} tracked frames: {} events; {} onset coverage>=0.9.".format(
        summary["videos"], summary["tracked_frames"], summary["events"], summary["events_with_coverage_ge_09"]))
    print("Macro AUC: {:.4f}; low-IoU frames: {} ({:.2f}% frame weighted).".format(
        summary["macro_auc"], summary["low_iou_frames"], 100 * summary["frame_weighted_low_iou_fraction"]))
    print("Review folder: " + str(Path(args.output_dir).resolve()))
    print("No GPU inference, training, or original artifact changes. Causes require human review.")


if __name__ == "__main__":
    main()
