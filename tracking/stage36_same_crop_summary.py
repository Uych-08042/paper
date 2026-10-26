"""Describe score/geometry swaps and crossed-parent effects; never select a policy."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from stage4_common import file_hash
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, nonempty_path, publish_identical, scene_geometry, seal, tracking_iou
from stage27_uav_summary import write_table
from stage28_response_common import bind_native_top1, dense_boxes, observation_label, response_metrics
from stage36_same_crop_collect import load_response
from stage36_same_crop_common import METHODS, SCHEMA, load_inputs, load_sealed, verify_bindings


def describe(video, data, source):
    responses, hybrids, decomposition = [], [], []
    sequence = video["sequence"]
    for row, task in enumerate(video["frames"]):
        gt = data["gt"][row]
        # Keep all parent/model outputs: no GT candidate selection or future feedback.
        current_iou = np.empty((4, 4))
        for parent, owner in enumerate(METHODS):
            geom = scene_geometry(data["previous"][row, parent], gt, data["image_hw"][row], source["search_factor"], source["search_size"])
            all_boxes, peaks = [], []
            for evaluator, method in enumerate(METHODS):
                raw = data["raw"][row, parent, evaluator]
                native = data["native_boxes"][row, parent, evaluator]
                boxes = dense_boxes(data["size"][row, parent, evaluator], data["offset"][row, parent, evaluator],
                    data["previous"][row, parent], data["factor"][row, parent], source["search_size"], data["image_hw"][row])
                boxes = bind_native_top1(raw, data["window"], boxes, native)
                metrics, _, _ = response_metrics(raw, data["window"], boxes, gt)
                current_iou[parent, evaluator] = metrics["weighted_top1_iou"]
                center_error = np.linalg.norm(native[:2] + native[2:] / 2 - gt[:2] - gt[2:] / 2) / np.sqrt(gt[2:].prod())
                responses.append(dict(sequence=sequence, frame_id=task["frame_id"], parent_method=owner, evaluator=method,
                    own_parent=parent == evaluator, fixed_time_control=bool(task["controls"]),
                    anchor_frame=any(a["offset"] == 0 for a in task["anchors"]),
                    window_memberships=json.dumps(task["anchors"], separators=(",", ":")),
                    coverage=float(geom["coverage"]), projected_gt_side_px=float(np.sqrt(geom["input_wh"].prod())),
                    parent_to_gt_side_ratio=float(np.sqrt(data["previous"][row, parent, 2:].prod() / gt[2:].prod())),
                    center_error_gt_units=float(center_error), width_ratio=float(native[2] / gt[2]), height_ratio=float(native[3] / gt[3]),
                    executed_layer=int(data["layer"][row, parent, evaluator]), **metrics,
                    observation=observation_label(metrics, geom["coverage"])))
                all_boxes.append(boxes)
                peaks.append(int(np.argmax(raw * data["window"])))
            for ref in range(2):
                for student in range(2, 4):
                    # Scores pick locations; geometry comes from the OTHER model at that location.
                    selected = np.stack([all_boxes[ref][peaks[ref]], all_boxes[ref][peaks[student]],
                        all_boxes[student][peaks[ref]], all_boxes[student][peaks[student]]])
                    values = tracking_iou(selected, gt)
                    base, score_swap, geometry_swap, full = map(float, values)
                    hybrids.append(dict(sequence=sequence, frame_id=task["frame_id"], parent_method=owner,
                        reference=METHODS[ref], student=METHODS[student], baseline_iou=base, score_swap_iou=score_swap,
                        geometry_swap_iou=geometry_swap, student_iou=full,
                        score_only_gain=score_swap - base, geometry_only_gain=geometry_swap - base,
                        full_gain=full - base, interaction=full - score_swap - geometry_swap + base))
        for ref in range(2):
            for student in range(2, 4):
                own_gap = current_iou[student, student] - current_iou[ref, ref]
                model_at_reference = current_iou[ref, student] - current_iou[ref, ref]
                parent_for_student = current_iou[student, student] - current_iou[ref, student]
                model_at_student = current_iou[student, student] - current_iou[student, ref]
                parent_for_reference = current_iou[student, ref] - current_iou[ref, ref]
                residual = max(abs(own_gap - model_at_reference - parent_for_student), abs(own_gap - model_at_student - parent_for_reference))
                if residual > 1e-12:
                    raise ValueError("Crossed-parent decomposition identity failed")
                decomposition.append(dict(sequence=sequence, frame_id=task["frame_id"], reference=METHODS[ref],
                    student=METHODS[student], own_trajectory_iou_gap=float(own_gap),
                    model_change_at_reference_parent=float(model_at_reference), parent_change_for_student=float(parent_for_student),
                    model_change_at_student_parent=float(model_at_student), parent_change_for_reference=float(parent_for_reference),
                    decomposition_max_residual=float(residual)))
    return responses, hybrids, decomposition


def means(rows, keys, metrics):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        result = dict(zip(keys, key))
        result["sampled_frames"] = len(values)
        for metric in metrics:
            valid = [v[metric] for v in values if v[metric] is not None]
            result["mean_" + metric] = float(np.mean(valid)) if valid else None
            result[metric + "_supported_frames"] = len(valid)
        output.append(result)
    return output


def render_matrix(video, data, row, source, entry, path, image_reader=checked_image):
    from PIL import Image, ImageDraw
    from stage27_uav_review import place_view, review_font, search_patch
    image = image_reader(source["root"], entry["frames"][int(data["frame_id"][row])])
    sheet = Image.new("RGB", (1520, 1260), "white")
    font, draw = review_font(), ImageDraw.Draw(sheet)
    draw.text((10, 8), "{} frame {} | rows: crop parents; columns: evaluators".format(video["sequence"], int(data["frame_id"][row])), fill="black", font=font)
    draw.text((10, 30), "GT green; native red; other fixed top-5 yellow. Weighted heatmaps normalize separately. GT match cells green.", fill="black", font=font)
    side = data["window"].shape[0]
    for parent, owner in enumerate(METHODS):
        geom = scene_geometry(data["previous"][row, parent], data["gt"][row], data["image_hw"][row], source["search_factor"], source["search_size"])
        patch = search_patch(image, geom["crop"], geom["usable"])
        for evaluator, method in enumerate(METHODS):
            left, top = evaluator * 380, 60 + parent * 300
            raw = data["raw"][row, parent, evaluator]
            native = data["native_boxes"][row, parent, evaluator]
            boxes = dense_boxes(data["size"][row, parent, evaluator], data["offset"][row, parent, evaluator],
                data["previous"][row, parent], data["factor"][row, parent], source["search_size"], data["image_hw"][row])
            boxes = bind_native_top1(raw, data["window"], boxes, native)
            metrics, details, overlap = response_metrics(raw, data["window"], boxes, data["gt"][row])
            colored = [(boxes[cell], "yellow") for cell in details["weighted"]["indices"][1:]]
            colored += [(data["gt"][row], (0, 190, 0)), (native, (230, 0, 0))]
            relative = [(np.r_[box[:2] - geom["crop"][:2], box[2:]], color) for box, color in colored]
            draw.text((left + 8, top + 4), "parent: " + owner, fill="black", font=font)
            draw.text((left + 8, top + 23), "model: " + method, fill="black", font=font)
            place_view(sheet, patch, (left + 8, top + 45, 224, 224), relative)
            weighted = raw * data["window"]
            value = (weighted - weighted.min()) / max(1e-12, float(weighted.max() - weighted.min()))
            heat = Image.fromarray((np.stack([value, np.sqrt(value), 1 - value], -1) * 255).astype(np.uint8))
            heat = heat.resize((120, 120), getattr(Image, "Resampling", Image).NEAREST)
            hd = ImageDraw.Draw(heat)
            for cell in np.flatnonzero(overlap >= .5):
                y, x = divmod(int(cell), side)
                hd.rectangle((x * 120 / side, y * 120 / side, (x + 1) * 120 / side - 1, (y + 1) * 120 / side - 1), outline=(0, 190, 0), width=1)
            for rank, cell in enumerate(details["weighted"]["indices"], 1):
                y, x = divmod(cell, side)
                hd.text((x * 120 / side, y * 120 / side), str(rank), fill="white", stroke_width=1, stroke_fill="black", font=font)
            sheet.paste(heat, (left + 244, top + 50))
            lines = ["IoU {:.3f}".format(metrics["weighted_top1_iou"]), "dense {:.3f}".format(metrics["dense_best_gt_iou"]),
                "GT rank {}".format(metrics["weighted_gt_rank_in_top5"]), "cover {:.2f}".format(geom["coverage"])]
            for line, text in enumerate(lines):
                draw.text((left + 242, top + 185 + 20 * line), text, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    sheet.save(temporary, format="PNG")
    temporary.replace(path)


def summarize(output, image_reader=checked_image):
    output = Path(output).resolve()
    plan, source, traces = load_inputs(output)
    run = load_sealed(output / "run.json")
    if run.get("plan_fingerprint") != plan["fingerprint"] or run.get("methods") != list(METHODS) or run.get("parents") != list(METHODS):
        raise ValueError("Replay run differs from frozen plan")
    expected = {v["sequence"] + "__stage36_same_crop.npz" for v in plan["videos"]}
    actual = {p.name for p in (output / "responses").glob("*.npz")}
    if actual - expected:
        raise ValueError("Unexpected replay cache")
    missing = sorted(expected - actual)
    if missing and (output / "analysis/stage36_diagnostic.json").exists():
        raise ValueError("A previously completed replay is now incomplete")
    selected = [v for v in plan["videos"] if v["sequence"] + "__stage36_same_crop.npz" in actual]
    if not selected:
        raise ValueError("No replay cache: collect at least one planned case")
    entries = {e["sequence"]: e for e in source["fit"]}
    all_responses, all_hybrids, all_decomposition, figures, hashes, integrity = [], [], [], [], {}, []
    destination = output / ("analysis_pilot" if missing else "analysis")
    with writer_lock(destination):
        completed = destination / "stage36_diagnostic.json"
        if completed.exists() and not missing:
            saved = load_sealed(completed)
            current = {name: file_hash(output / "responses" / name) for name in sorted(expected)}
            if (saved.get("status") != "COMPLETE_SAME_CROP_DIAGNOSTIC" or
                    saved.get("plan_fingerprint") != plan["fingerprint"] or saved.get("run_fingerprint") != run["fingerprint"] or
                    saved.get("response_cache_sha256") != current):
                raise ValueError("Completed Stage-36 response/source changed")
            for name, digest in saved["table_sha256"].items():
                if file_hash(destination / name) != digest:
                    raise ValueError("Completed Stage-36 table changed")
            for picture in saved["figures"]:
                if file_hash(destination / picture["file"]) != picture["sha256"]:
                    raise ValueError("Completed Stage-36 figure changed")
            print("Stage-36: COMPLETE_SAME_CROP_DIAGNOSTIC (reuse, audited)")
            return saved
        for video in selected:
            name = video["sequence"]
            path = output / "responses" / (name + "__stage36_same_crop.npz")
            hashes[path.name] = file_hash(path)
            data = load_response(path, source, entries[name], traces[name], video, run)
            integrity.append(dict(sequence=name, replayed_images=len(data["frame_id"]),
                maximum_own_parent_box_error=float(data["replay_box_error"].max()),
                maximum_dense_decode_box_error=float(data["decode_box_error"].max()),
                model_head_calls=data["model_head_calls"].tolist()))
            r, h, d = describe(video, data, source)
            all_responses.extend(r)
            all_hybrids.extend(h)
            all_decomposition.extend(d)
            centers = {a["frame_id"] for a in video["anchors"]} or {f["frame_id"] for f in video["frames"] if f["controls"]}
            for frame in sorted(centers):
                row = int(np.flatnonzero(data["frame_id"] == frame)[0])
                picture = destination / "figures" / (name + "__frame{:06d}.png".format(frame))
                render_matrix(video, data, row, source, entries[name], picture, image_reader)
                figures.append(dict(file=picture.relative_to(destination).as_posix(), sha256=file_hash(picture)))
        tables = dict(responses=all_responses, hybrids=all_hybrids, parent_decomposition=all_decomposition)
        tables["same_crop_summary"] = means(all_responses, ("sequence", "parent_method", "evaluator"),
            ("weighted_top1_iou", "dense_best_gt_iou", "weighted_top5_gt_recall", "raw_top5_gt_recall", "weighted_gt_score_margin", "coverage"))
        tables["hybrid_summary"] = means(all_hybrids, ("sequence", "parent_method", "reference", "student"),
            ("baseline_iou", "score_swap_iou", "geometry_swap_iou", "student_iou", "score_only_gain", "geometry_only_gain", "full_gain", "interaction"))
        tables["parent_summary"] = means(all_decomposition, ("sequence", "reference", "student"),
            ("own_trajectory_iou_gap", "model_change_at_reference_parent", "parent_change_for_student", "model_change_at_student_parent", "parent_change_for_reference"))
        phase_rows = []
        for phase, key in (("fixed_time_controls", "fixed_time_control"), ("anchor_frames", "anchor_frame")):
            selected_responses = [r for r in all_responses if r[key]]
            if selected_responses:
                phase_rows.extend(dict(phase=phase, **r) for r in means(selected_responses,
                    ("sequence", "parent_method", "evaluator"), ("weighted_top1_iou", "dense_best_gt_iou", "weighted_top5_gt_recall", "coverage")))
        tables["phase_summary"] = phase_rows
        table_hashes = {}
        for label, values in tables.items():
            path = destination / ("stage36_" + label + ".csv")
            write_table(path, values)
            table_hashes[path.name] = file_hash(path)
        verify_bindings(plan["source_file_sha256"])
        for name, digest in hashes.items():
            if file_hash(output / "responses" / name) != digest:
                raise ValueError("Replay cache changed during analysis")
        report = seal(dict(schema=SCHEMA, role="posthoc_development_same_crop_mechanism_audit",
            status="PILOT_REPLAY_ONLY" if missing else "COMPLETE_SAME_CROP_DIAGNOSTIC",
            plan_fingerprint=plan["fingerprint"], run_fingerprint=run["fingerprint"],
            audited_cases=len(selected), missing_cases=missing, response_cache_sha256=hashes, figures=figures,
            table_sha256=table_hashes, integrity=integrity,
            sampled_frame_images=sum(len(v["frames"]) for v in selected),
            maximum_parent_decomposition_residual=max(d["decomposition_max_residual"] for d in all_decomposition),
            training_performed=False, learned_policy_evaluated=False, adoption_gate=None,
            warnings=["Cases, anchors and windows use consumed outcomes; this is not independent validation.",
                "All replays reset to recorded predicted parents; no GT parent, output correction or branch continuation is used.",
                "Matrix diagonals reproduce original outputs; off-diagonals are single-state counterfactual responses.",
                "Parent decompositions are exact single-frame accounting identities, not unique causal attribution of a whole trajectory.",
                "Score/geometry swaps are offline decoder probes, not separately trained modules or evaluated policies.",
                "Dense GT matches are hindsight ceilings; geometric non-target boxes are not confirmed semantic distractors.",
                "Selected-window means are not full-video AUC. Overlapping windows and frames are not independent videos.",
                "No new method pass/fail, compression improvement, speedup or external generalization is established."],
            next_step="Interpret source-consistent overlays, score/geometry probes and crop-history effects before designing a new training mechanism"))
        # A partial pilot can grow; completed reports remain immutable.
        report_path = destination / "stage36_diagnostic.json"
        if missing:
            from stage4_common import write_json
            write_json(report_path, report)
        else:
            publish_identical(report_path, report)
    print("Stage-36: {} ({} / 4 cases)".format(report["status"], len(selected)))
    print("Reports: " + str(destination))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True, help="Stage-36 output root")
    args = parser.parse_args()
    summarize(Path(args.input).expanduser())


if __name__ == "__main__":
    main()
