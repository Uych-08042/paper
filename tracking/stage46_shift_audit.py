"""CPU-only temporal audit of completed Stage-46 search shifts; no image/model calls."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint
from stage27_uav_collect import writer_lock
from stage27_uav_common import (crop_geometry, failure_events, nonempty_path,
    publish_identical, scene_geometry, success_auc, tracking_iou)
import stage46_motion_common as c


SCHEMA = "stage46b_cached_search_shift_audit_v1"
CONFIG = dict(cases=["uav0000176_00000_s", "uav0000204_00000_s", "uav0000304_00253_s"],
    before=10, after=30, long_event_frames=30, controls=[1, 2, 3],
    float_difference_atol=1e-9, accounting_atol_px=1e-8,
    selection="Three cases chosen after Stage-46 outcomes; descriptive development diagnosis")
TABLES = ("case_summary", "onsets", "anchors", "phase_summary", "timeline", "transitions")
TERMS = ("previous_center_change", "applied_shift_change", "crop_scale_origin_change",
         "mapped_output_remainder_change", "negative_gt_motion")


def seal(body):
    return dict(body, fingerprint=fingerprint(body))


def read_report(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: v for k, v in value.items() if k != "fingerprint"}
    if value.get("schema") != SCHEMA or value.get("fingerprint") != fingerprint(body):
        raise ValueError("Stage-46B schema/fingerprint mismatch: " + str(path))
    return value


def verify_files(bindings):
    for name, digest in bindings.items():
        if not Path(name).is_file() or file_hash(name) != digest:
            raise ValueError("Frozen audit input/output changed: " + str(name))


def load_completed(root):
    """Audit all 48 committed trajectories before selecting the three known cases."""
    root = Path(root).expanduser().resolve()
    plan, inputs = c.load_plan(root)
    directory, source, traces, source_hashes = inputs
    run = c.load_sealed(root / "run.json")
    decision = c.load_sealed(root / "analysis/stage46_decision.json")
    expected = {c.cache_path(root, name, method).name for name in plan["sequences"] for method in c.METHODS}
    if (run.get("plan_fingerprint") != plan["fingerprint"] or
        run.get("role") != "frozen_native_sgla_four_policy_motion_run" or run.get("methods") != list(c.METHODS) or
        decision.get("plan_fingerprint") != plan["fingerprint"] or decision.get("run_fingerprint") != run["fingerprint"] or
        decision.get("role") != "consumed_development_native_sgla_motion_prototype" or
        decision.get("status") not in {"REVIEW_CAMERA_SEARCH_PILOT", "PROMISING_CAMERA_SEARCH_PILOT", "LIMITED_BACKGROUND_MOTION_SUPPORT"} or
        decision.get("training_performed") is not False or decision.get("weights_unchanged") is not True or
        decision.get("exact_native_call_budget") is not True or
        decision.get("videos") != len(plan["sequences"]) or decision.get("full_trajectories") != len(expected) or
        set(decision.get("cache_sha256", {})) != expected):
        raise ValueError("Need the complete, unmixed Stage-46 summary and run")
    if not set(CONFIG["cases"]) <= set(plan["sequences"]):
        raise ValueError("The three fixed audit cases must belong to the Stage-46 plan")
    if {p.name for p in (root / "traces").glob("*.npz")} != expected:
        raise ValueError("Need exactly the completed Stage-46 traces")
    bindings = {str(root / name): file_hash(root / name)
                for name in ("manifest.json", "run.json", "analysis/stage46_decision.json")}
    bindings[str(directory / "run.json")] = plan["source_run_sha256"]
    bindings.update({str(directory / "traces" / name): digest for name, digest in source_hashes.items()})
    cached = {}
    for name in plan["sequences"]:
        entry, native = traces[name]
        for method in c.METHODS:
            path = c.cache_path(root, name, method)
            data = c.load_result(path, c.metadata(run, name, method, False))
            if data is None or file_hash(path) != decision["cache_sha256"][path.name]:
                raise ValueError("Incomplete trace or Stage-46 decision/cache mismatch: " + str(path))
            c.audit_video(data, entry["frames"], entry["initial_box"], method, native)
            bindings[str(path)] = file_hash(path)
            receipt = path.with_suffix(".receipt.json")
            bindings[str(receipt)] = file_hash(receipt)
            if name in CONFIG["cases"]:
                cached[name, method] = data
        print("[source audited] " + name, flush=True)
    verify_files(bindings)
    return plan, source, traces, run, decision, cached, bindings


def first_frame(mask):
    indices = np.flatnonzero(mask)
    return None if not len(indices) else int(indices[0]) + 1


def vector_columns(prefix, value):
    return {prefix + "_x": float(value[0]), prefix + "_y": float(value[1])}


def frame_rows(entry, data, native, factor, search_size):
    """Ground truth appears only in offline measurements of already-saved predictions."""
    gt_all = np.asarray(entry["ground_truth"], float)
    overlap = tracking_iou(data["boxes"], gt_all[1:])
    native_iou = tracking_iou(native["boxes"], gt_all[1:])
    rows = []
    for i, frame_id in enumerate(data["frame_id"]):
        hw, parent, previous, box = data["image_hw"][i], data["parents"][i], data["previous"][i], data["boxes"][i]
        gt, old_gt = gt_all[i + 1], gt_all[i]
        gt_side, own_side = np.sqrt(gt[2:].prod()), np.sqrt(previous[2:].prod())
        geom = scene_geometry(parent, gt, hw, factor, search_size)
        own_crop = crop_geometry(previous, hw, factor, search_size)[0]
        native_crop = crop_geometry(native["parents"][i], hw, factor, search_size)[0]
        pc, gc, oc = c.center(parent), c.center(gt), c.center(box)
        prevc = c.center(previous)
        origin = pc - geom["crop"][2] / 2
        # This remainder includes CENTER decode arithmetic and final clipping.
        # Score/size/offset maps and the peak location were not saved in Stage-46.
        remainder = oc - origin
        camera, shift = data["camera_displacement"][i], data["shift"][i]
        relative = data["matrix"][i, :, :2] @ data["relative_velocity"][i]
        required = gc - prevc
        required_norm = np.linalg.norm(required)
        stats = data["motion_stats"][i]
        row = dict(frame_id=int(frame_id), source_file=entry["frames"][int(frame_id)]["file"],
            iou=float(overlap[i]), native_iou=float(native_iou[i]), coverage=float(geom["coverage"]),
            projected_gt_side_px=float(np.sqrt(geom["input_wh"].prod())),
            parent_error_gt_units=float(np.linalg.norm(pc - gc) / gt_side),
            previous_error_gt_units=float(np.linalg.norm(prevc - gc) / gt_side),
            output_error_gt_units=float(np.linalg.norm(oc - gc) / gt_side),
            center_error_reduction=float((np.linalg.norm(prevc - gc) - np.linalg.norm(pc - gc)) / gt_side),
            gt_side_px=float(gt_side), previous_side_px=float(own_side),
            crop_x=int(geom["crop"][0]), crop_y=int(geom["crop"][1]), crop_side=int(geom["crop"][2]),
            crop_changed_from_own_previous=bool(np.any(geom["crop"] != own_crop)),
            crop_differs_native=bool(np.any(geom["crop"] != native_crop)),
            parent_differs_native=bool(np.max(np.abs(parent - native["parents"][i])) > CONFIG["float_difference_atol"]),
            previous_differs_native=bool(np.max(np.abs(previous - native["previous"][i])) > CONFIG["float_difference_atol"]),
            output_differs_native=bool(np.max(np.abs(box - native["boxes"][i])) > CONFIG["float_difference_atol"]),
            official_box_differs_native=bool(np.any(np.trunc(box) != np.trunc(native["boxes"][i]))),
            layer=int(data["layer"][i]), native_layer=int(native["layer"][i]),
            raw_peak_amplitude=float(data["peak"][i]), native_raw_peak_amplitude=float(native["peak"][i]),
            peak_amplitude_delta=float(data["peak"][i] - native["peak"][i]),
            motion_valid=bool(data["motion_valid"][i]), motion_reason=int(data["motion_reason"][i]),
            corners=int(stats[0]), valid_tracks=int(stats[1]), inliers=int(stats[2]),
            inlier_ratio=float(stats[3]), median_fit_residual_scaled_px=float(stats[4]), occupied_cells=int(stats[5]),
            cap_hit=bool(data["cap_hit"][i]), shift_target_sides=float(np.linalg.norm(shift) / own_side),
            camera_target_sides=float(np.linalg.norm(camera) / own_side),
            transported_relative_target_sides=float(np.linalg.norm(relative) / own_side),
            camera_plus_relative_target_sides=float(np.linalg.norm(camera + relative) / own_side),
            camera_relative_cosine=(float(np.dot(camera, relative) / np.linalg.norm(camera) / np.linalg.norm(relative))
                if min(np.linalg.norm(camera), np.linalg.norm(relative)) > 1e-12 else None),
            shift_along_required_gt_units=(float(np.dot(shift, required) / required_norm / gt_side)
                if required_norm > 1e-12 else None),
            mapped_camera_gt_prediction_error=float(np.linalg.norm(c.warp(data["matrix"][i], c.center(old_gt)) - gc) / gt_side)
                if data["motion_valid"][i] else None)
        for prefix, value in (("previous_center", prevc), ("parent_center", pc), ("output_center", oc),
            ("gt_center", gc), ("decoder_origin", origin), ("mapped_output_remainder", remainder),
            ("decoder_origin_minus_crop_origin", origin - geom["crop"][:2]),
            ("shift", shift), ("camera", camera), ("raw_velocity", data["raw_velocity"][i]),
            ("relative_velocity", data["relative_velocity"][i]), ("transported_relative", relative)):
            row.update(vector_columns(prefix, value))
        row.update({"affine_{}{}".format(a, b): float(data["matrix"][i, a, b]) for a in range(2) for b in range(3)})
        rows.append(row)
    return rows


def summarize_case(sequence, method, rows):
    events = failure_events([r["iou"] for r in rows])
    long = [e for e in events if e["end"] - e["start"] + 1 >= CONFIG["long_event_frames"]]
    summary = dict(sequence=sequence, method=method, tracked_frames=len(rows),
        auc=success_auc([r["iou"] for r in rows]),
        auc_delta_native=success_auc([r["iou"] for r in rows]) - success_auc([r["native_iou"] for r in rows]),
        low_iou_frames=sum(r["iou"] < .2 for r in rows), failure_events=len(events),
        first_applied_shift=first_frame([np.hypot(r["shift_x"], r["shift_y"]) > CONFIG["float_difference_atol"] for r in rows]),
        first_own_rounded_crop_change=first_frame([r["crop_changed_from_own_previous"] for r in rows]),
        first_parent_divergence=first_frame([r["parent_differs_native"] for r in rows]),
        first_rounded_crop_divergence=first_frame([r["crop_differs_native"] for r in rows]),
        first_output_divergence=first_frame([r["output_differs_native"] for r in rows]),
        first_official_box_divergence=first_frame([r["official_box_differs_native"] for r in rows]),
        first_layer_divergence=first_frame([r["layer"] != r["native_layer"] for r in rows]),
        first_peak_amplitude_divergence=first_frame([abs(r["peak_amplitude_delta"]) > CONFIG["float_difference_atol"] for r in rows]),
        first_low_iou_frame=first_frame([r["iou"] < .2 for r in rows]),
        first_failure_onset=None if not events else events[0]["start"] + 1,
        first_failure_confirmation=None if not events else events[0]["confirmation"] + 1,
        first_long_failure_onset=None if not long else long[0]["start"] + 1,
        valid_motion_fraction=float(np.mean([r["motion_valid"] for r in rows])),
        same_native_crop_output_changed_frames=sum(not r["crop_differs_native"] and r["output_differs_native"] for r in rows),
        mean_center_error_reduction=float(np.mean([r["center_error_reduction"] for r in rows])))
    return summary, events


def make_anchors(sequence, summaries):
    anchors = {}
    for summary in summaries:
        method, n = summary["method"], summary["tracked_frames"]
        requested = {key: summary[key] for key in ("first_output_divergence", "first_rounded_crop_divergence",
                                                  "first_failure_onset", "first_long_failure_onset")}
        requested.update({"fixed_q" + str(q): max(1, n * q // 4) for q in CONFIG["controls"]})
        for kind, frame in requested.items():
            if frame is not None:
                anchors.setdefault((method, frame), []).append(kind)
    return [dict(sequence=sequence, anchor_id=i + 1, owner=method, frame_id=frame, kinds="|".join(kinds))
            for i, ((method, frame), kinds) in enumerate(sorted(anchors.items()))]


def transition(before, after):
    """Exact vector accounting, with an unresolved decoded/clipped output remainder."""
    def vector(row, prefix):
        return np.array([row[prefix + "_x"], row[prefix + "_y"]])
    terms = dict(previous_center_change=vector(after, "previous_center") - vector(before, "previous_center"),
        applied_shift_change=vector(after, "shift") - vector(before, "shift"),
        crop_scale_origin_change=np.full(2, -.5 * (after["crop_side"] - before["crop_side"])),
        mapped_output_remainder_change=vector(after, "mapped_output_remainder") - vector(before, "mapped_output_remainder"),
        negative_gt_motion=vector(before, "gt_center") - vector(after, "gt_center"))
    error_change = ((vector(after, "output_center") - vector(after, "gt_center")) -
                    (vector(before, "output_center") - vector(before, "gt_center")))
    residual = float(np.max(np.abs(sum(terms.values()) - error_change)))
    if residual > CONFIG["accounting_atol_px"]:
        raise ValueError("Output-center vector accounting failed")
    norm, scale = float(np.linalg.norm(error_change)), after["gt_side_px"]
    result = dict(from_frame=before["frame_id"], to_frame=after["frame_id"],
        previous_iou=before["iou"], current_iou=after["iou"],
        error_vector_change_gt_units=norm / scale, accounting_residual_px=residual)
    result.update(vector_columns("error_change", error_change))
    for name, value in terms.items():
        result.update(vector_columns(name, value))
        result[name + "_along_error_gt_units"] = float(np.dot(value, error_change) / norm / scale) if norm > 1e-12 else None
    if norm > 1e-12:
        projected = sum(result[t + "_along_error_gt_units"] for t in TERMS)
        if abs(projected - norm / scale) * scale > CONFIG["accounting_atol_px"]:
            raise ValueError("Projected vector accounting failed")
    return result


def analyze_case(sequence, entry, arrays, factor, search_size):
    series = {method: frame_rows(entry, arrays[method], arrays["sgla"], factor, search_size) for method in c.METHODS}
    summaries, onsets = [], []
    for method, rows in series.items():
        summary, events = summarize_case(sequence, method, rows)
        summaries.append(summary)
        for number, event in enumerate(events, 1):
            row = rows[event["start"]]
            onsets.append(dict(sequence=sequence, method=method, event=number,
                onset_frame=event["start"] + 1, confirmed_frame=event["confirmation"] + 1,
                end_frame=event["end"] + 1, recovery_frame=None if event["recovery"] is None else event["recovery"] + 1,
                event_duration_frames=event["end"] - event["start"] + 1,
                event_low_iou_frames=sum(r["iou"] < .2 for r in rows[event["start"]:event["end"] + 1]),
                first_event=number == 1, censor=event["censor"],
                **{k: row[k] for k in ("iou", "native_iou", "coverage", "parent_error_gt_units", "output_error_gt_units",
                    "center_error_reduction", "shift_target_sides", "camera_target_sides", "transported_relative_target_sides",
                    "camera_plus_relative_target_sides", "camera_relative_cosine", "crop_differs_native",
                    "motion_valid", "inlier_ratio", "median_fit_residual_scaled_px", "layer", "native_layer", "raw_peak_amplitude")}))
    anchors = make_anchors(sequence, summaries)
    phases, memberships = [], {}
    for anchor in anchors:
        t = anchor["frame_id"]
        n = len(series[anchor["owner"]])
        indices = dict(before=range(max(1, t - CONFIG["before"]), t), anchor=range(t, t + 1),
                       after=range(t + 1, min(n, t + CONFIG["after"]) + 1))
        for phase, frames in indices.items():
            for frame in frames:
                memberships.setdefault(frame, []).append("{}:{}".format(anchor["anchor_id"], phase))
            for method, rows in series.items():
                selected = [rows[f - 1] for f in frames]
                if not selected:
                    continue
                metrics = ("iou", "coverage", "parent_error_gt_units", "output_error_gt_units", "center_error_reduction",
                    "shift_target_sides", "camera_target_sides", "transported_relative_target_sides", "motion_valid",
                    "crop_differs_native", "cap_hit", "inlier_ratio", "median_fit_residual_scaled_px", "raw_peak_amplitude")
                phases.append(dict(**anchor, evaluator=method, phase=phase, frames=len(selected),
                    first_frame=selected[0]["frame_id"], last_frame=selected[-1]["frame_id"],
                    **{"mean_" + key: float(np.mean([r[key] for r in selected])) for key in metrics}))
    timeline, transitions = [], []
    for method, rows in series.items():
        for frame in sorted(memberships):
            i = frame - 1
            timeline.append(dict(sequence=sequence, method=method, anchor_phase_memberships="|".join(memberships[frame]), **rows[i]))
            if i > 0:
                transitions.append(dict(sequence=sequence, method=method, **transition(rows[i - 1], rows[i])))
    return dict(case_summary=summaries, onsets=onsets, anchors=anchors, phase_summary=phases,
                timeline=timeline, transitions=transitions)


def publish_table(path, rows):
    """Stable CSV bytes; an existing differing report is never overwritten."""
    stream = io.StringIO(newline="")
    fields = list(rows[0]) if rows else ["sequence", "method", "event", "onset_frame"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    payload = stream.getvalue().encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("Existing audit table differs; use a new output directory: " + str(path))
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def audit(root, output_dir=None):
    root = Path(root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve() if output_dir else root / "analysis/search_shift_audit_v1"
    # Only the dedicated child or a separate directory may receive audit artifacts.
    dedicated = root / "analysis/search_shift_audit_v1"
    if root == output or output in root.parents or (root in output.parents and output != dedicated):
        raise ValueError("Use the dedicated search_shift_audit_v1 directory or a separate output directory")
    plan, source, traces, run, decision, cached, bindings = load_completed(root)
    files = [Path(__file__), Path(__file__).with_name("test_stage46_shift_audit.py"),
             Path(__file__).with_name("stage27_uav_common.py"), Path(__file__).with_name("validation_metrics.py")]
    code = {str(p.relative_to(Path(__file__).resolve().parents[1])): file_hash(p) for p in files}
    code.update({"tracking/" + name: digest for name, digest in c.code_hashes().items()})
    identity = dict(schema=SCHEMA, configuration=CONFIG, source_plan_fingerprint=plan["fingerprint"],
        source_run_fingerprint=run["fingerprint"], source_decision_fingerprint=decision["fingerprint"],
        source_file_sha256=bindings, code_sha256=code)
    with writer_lock(output):
        report_path = output / "stage46b_diagnostic.json"
        if report_path.exists():
            saved = read_report(report_path)
            if any(saved.get(key) != value for key, value in identity.items()):
                raise ValueError("Completed Stage-46B source/code/configuration changed; use a new output directory")
            if read_report(output / "manifest.json") != seal(identity):
                raise ValueError("Completed Stage-46B manifest changed")
            verify_files({str(output / name): digest for name, digest in saved["table_sha256"].items()})
            print("Stage-46B: COMPLETE_CACHED_SEARCH_SHIFT_AUDIT (reuse, audited)")
            print("Reports: " + str(output))
            return saved
        allowed = {"collect.lock", "manifest.json"} | {"stage46b_" + name + ".csv" for name in TABLES}
        allowed |= {name + ".tmp" for name in allowed if name.endswith(".csv")}
        if any(p.name not in allowed for p in output.iterdir()):
            raise ValueError("Use an empty or matching Stage-46B output directory")
        publish_identical(output / "manifest.json", seal(identity))
        tables = {name: [] for name in TABLES}
        for sequence in CONFIG["cases"]:
            entry = traces[sequence][0]
            parts = analyze_case(sequence, entry, {m: cached[sequence, m] for m in c.METHODS},
                                 source["search_factor"], source["search_size"])
            for name in TABLES:
                tables[name].extend(parts[name])
            print("[case audited] " + sequence, flush=True)
        for name, rows in tables.items():
            publish_table(output / ("stage46b_" + name + ".csv"), rows)
        verify_files(bindings)
        report = seal(dict(**identity, role="posthoc_consumed_development_cached_search_shift_diagnosis",
            status="COMPLETE_CACHED_SEARCH_SHIFT_AUDIT", source_videos_audited=len(plan["sequences"]),
            source_trajectories_audited=len(plan["sequences"]) * len(c.METHODS),
            focus_videos=len(CONFIG["cases"]), focus_trajectories=len(tables["case_summary"]),
            failure_events_audited=len(tables["onsets"]),
            maximum_vector_accounting_residual_px=max((r["accounting_residual_px"] for r in tables["transitions"]), default=0.),
            table_rows={name: len(rows) for name, rows in tables.items()},
            table_sha256={"stage46b_" + name + ".csv": file_hash(output / ("stage46b_" + name + ".csv")) for name in TABLES},
            training_performed=False, new_inference_performed=False, images_opened=False,
            source_artifacts_modified=False, adoption_gate=None,
            warnings=["The three cases and failure anchors use consumed outcomes; no new method test or independent validation.",
                "Frame IDs are tracked image indices: source images are indexed with the initialization image at zero.",
                "All evaluators follow their own recorded trajectories, including their own masks and background estimates; not matched-parent responses.",
                "Rounded crop origin and floating-parent decoder origin differ in native SGLA. Equal crop rectangles do not force equal mapped boxes.",
                "Raw peak is amplitude only; no peak position, score/size/offset maps or Top-K retention can be recovered from Stage-46 caches.",
                "Center-transition terms are exact accounting; the mapped-output remainder includes model response, geometry, arithmetic and clipping.",
                "Accepted affine fits do not validate physical camera motion. GT projection errors and coverage are offline geometric diagnostics, not online gates or visibility labels.",
                "First long failure means an event lasting at least 30 frames; it need not contain 30 consecutive low-IoU frames.",
                "Anchor windows overlap, fixed quartiles need not be successful, and sampled-window means are not full-video AUC or independent samples."],
            next_step="Inspect first divergence and failure windows; specify a matched-parent response replay only if peak/geometry attribution is needed"))
        publish_identical(report_path, report)
    print("Stage-46B: COMPLETE_CACHED_SEARCH_SHIFT_AUDIT; no training, image reads or GPU inference")
    print("sequence             method             crop_diff output_diff first_failure first_long")
    for r in tables["case_summary"]:
        print("{} {:<16} {:>9} {:>11} {:>13} {:>10}".format(r["sequence"], r["method"],
            str(r["first_rounded_crop_divergence"]), str(r["first_output_divergence"]),
            str(r["first_failure_onset"]), str(r["first_long_failure_onset"])))
    print("Reports: " + str(output))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True, help="Completed Stage-46 root")
    parser.add_argument("--output_dir", type=nonempty_path, help="Optional separate new audit directory")
    args = parser.parse_args()
    audit(args.input, args.output_dir)


if __name__ == "__main__":
    main()
