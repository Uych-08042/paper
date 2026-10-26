"""CPU-only response/offset/size factorial audit of completed Stage-36 caches."""

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

from stage4_common import file_hash
from stage27_uav_collect import writer_lock
from stage27_uav_common import nonempty_path, publish_identical, scene_geometry, seal, tracking_iou
from stage27_uav_summary import write_table
from stage28_response_common import clip_boxes, continuous_iou
from stage34_identity_common import load_sealed as read_sealed
from stage36_same_crop_collect import load_response
from stage36_same_crop_common import METHODS, load_inputs, load_sealed, verify_bindings


SCHEMA = "stage36b_cached_geometry_factor_audit_v1"
CONFIG = dict(factors=["response", "offset", "size"], official_iou="truncate_prediction_as_original",
    floating_iou="untruncated_prediction_secondary_only", endpoint_atol=1e-12,
    decomposition_atol=1e-12, native_binding_atol=.001,
    anchors="All original first_sustained_discrepancy and first_failure anchors; unchanged windows",
    phases=["before", "anchor", "after"])
CORNERS = tuple(itertools.product((0, 1), repeat=3))
SUMMARY_FIELDS = ("response_only_gain", "offset_only_gain", "size_only_gain", "geometry_gain", "full_gain",
    "response_offset_interaction", "response_size_interaction", "offset_size_interaction", "three_way_interaction",
    "order_averaged_response_gain", "order_averaged_offset_gain", "order_averaged_size_gain",
    "offset_given_reference_response", "size_given_reference_response",
    "offset_given_student_response", "size_given_student_response",
    "floating_full_gain", "floating_geometry_gain", "floating_offset_given_reference_response", "floating_size_given_reference_response",
    "floating_order_averaged_offset_gain", "floating_order_averaged_size_gain")
STATE_FIELDS = ("iou", "coverage", "projected_gt_side_px", "parent_center_error_gt_units", "parent_side_ratio",
    "center_error_gt_units", "width_ratio", "height_ratio", "unclipped_center_error_gt_units",
    "unclipped_width_ratio", "unclipped_height_ratio", "clip_max_abs_delta")


def decode_cell(size, offset, cell, previous, factor, search_size, image_hw):
    """Same float32 arithmetic and floating parent mapping as dense_boxes, at one cell."""
    size, offset = np.asarray(size, np.float32), np.asarray(offset, np.float32)
    if size.ndim != 3 or size.shape != offset.shape or size.shape[0] != 2 or size.shape[1] != size.shape[2]:
        raise ValueError("Invalid size/offset grid")
    side = size.shape[1]
    if not 0 <= cell < side * side or factor <= 0 or not np.isfinite(size).all() or not np.isfinite(offset).all() or (size < 0).any():
        raise ValueError("Invalid geometry/cell")
    y, x = divmod(int(cell), side)
    center = np.array([np.float32(x) + offset[0, y, x], np.float32(y) + offset[1, y, x]], np.float32) / np.float32(side)
    normalized = np.concatenate([center, size[:, y, x]])
    physical = ((normalized * np.float32(search_size)) / np.float32(factor)).astype(np.float64)
    parent = np.asarray(previous, float)
    image_center = physical[:2] + parent[:2] + parent[2:] / 2 - .5 * search_size / factor
    unclipped = np.r_[image_center - physical[2:] / 2, physical[2:]]
    return unclipped, clip_boxes(unclipped, image_hw)


def box_metrics(box, unclipped, clipped, gt):
    box, gt = np.asarray(box, float), np.asarray(gt, float)
    scale = np.sqrt(gt[2:].prod())
    gt_center = gt[:2] + gt[2:] / 2
    return dict(iou=float(tracking_iou(box, gt)), floating_iou=float(continuous_iou(box, gt)),
        center_error_gt_units=float(np.linalg.norm(box[:2] + box[2:] / 2 - gt_center) / scale),
        width_ratio=float(box[2] / gt[2]), height_ratio=float(box[3] / gt[3]),
        unclipped_center_error_gt_units=float(np.linalg.norm(unclipped[:2] + unclipped[2:] / 2 - gt_center) / scale),
        unclipped_width_ratio=float(unclipped[2] / gt[2]), unclipped_height_ratio=float(unclipped[3] / gt[3]),
        clip_max_abs_delta=float(np.max(np.abs(clipped - unclipped))),
        native_binding_max_abs_delta=float(np.max(np.abs(box - clipped))))


def factorial(data, row, parent, reference, student, search_size):
    """R chooses the peak, O the sub-cell center offset, Z width/height; no GT choice."""
    methods = (reference, student)
    raw = [data["raw"][row, parent, m] for m in methods]
    sizes = [data["size"][row, parent, m] for m in methods]
    offsets = [data["offset"][row, parent, m] for m in methods]
    natives = [data["native_boxes"][row, parent, m] for m in methods]
    peaks = [int(np.argmax(r * data["window"])) for r in raw]
    values, floating, details = np.empty((2, 2, 2)), np.empty((2, 2, 2)), {}
    for r, o, z in CORNERS:
        cell = peaks[r]
        y, x = divmod(cell, raw[0].shape[1])
        unclipped, clipped = decode_cell(sizes[z], offsets[o], cell, data["previous"][row, parent],
            data["factor"][row, parent], search_size, data["image_hw"][row])
        box = clipped.copy()
        bound = ""
        # Preserve exact source-native decoding only for IDENTICAL cell inputs.
        # This also keeps a literal identity intervention exactly zero, with no GT snapping.
        # Preserve Stage-36's existing four endpoints exactly. A same-owner
        # geometry pair uses only that owner's native binding; mixed pairs
        # may bind either source when their selected cell inputs are identical.
        candidates = (o,) if o == z else range(2)
        for index in candidates:
            if (cell == peaks[index] and np.array_equal(sizes[z][:, y, x], sizes[index][:, y, x]) and
                    np.array_equal(offsets[o][:, y, x], offsets[index][:, y, x])):
                if np.max(np.abs(box - natives[index])) > CONFIG["native_binding_atol"]:
                    raise ValueError("Single-cell decode disagrees with frozen native output")
                box, bound = natives[index].copy(), METHODS[methods[index]]
                break
        metrics = box_metrics(box, unclipped, clipped, data["gt"][row])
        values[r, o, z], floating[r, o, z] = metrics["iou"], metrics["floating_iou"]
        details[(r, o, z)] = dict(response_from=METHODS[methods[r]], offset_from=METHODS[methods[o]],
            size_from=METHODS[methods[z]], selected_cell=cell, native_binding_source=bound,
            x=float(box[0]), y=float(box[1]), width=float(box[2]), height=float(box[3]), **metrics)
    return values, floating, details


def split_values(values):
    values = np.asarray(values, float)
    if values.shape != (2, 2, 2) or not np.isfinite(values).all():
        raise ValueError("Need eight finite factorial corners")
    v = lambda r, o, z: float(values[r, o, z])
    base = v(0, 0, 0)
    single = [v(1, 0, 0) - base, v(0, 1, 0) - base, v(0, 0, 1) - base]
    ro = v(1, 1, 0) - v(1, 0, 0) - v(0, 1, 0) + base
    rz = v(1, 0, 1) - v(1, 0, 0) - v(0, 0, 1) + base
    oz = v(0, 1, 1) - v(0, 1, 0) - v(0, 0, 1) + base
    triple = v(1, 1, 1) - v(1, 1, 0) - v(1, 0, 1) - v(0, 1, 1) + v(1, 0, 0) + v(0, 1, 0) + v(0, 0, 1) - base
    ordered = np.zeros(3)
    for order in itertools.permutations(range(3)):
        state = [0, 0, 0]
        previous = base
        for factor in order:
            state[factor] = 1
            current = float(values[tuple(state)])
            ordered[factor] += current - previous
            previous = current
    ordered /= 6
    offset_ref = .5 * (v(0, 1, 0) - base + v(0, 1, 1) - v(0, 0, 1))
    size_ref = .5 * (v(0, 0, 1) - base + v(0, 1, 1) - v(0, 1, 0))
    offset_student = .5 * (v(1, 1, 0) - v(1, 0, 0) + v(1, 1, 1) - v(1, 0, 1))
    size_student = .5 * (v(1, 0, 1) - v(1, 0, 0) + v(1, 1, 1) - v(1, 1, 0))
    full, geometry = v(1, 1, 1) - base, v(0, 1, 1) - base
    residual = max(abs(ordered.sum() - full), abs(sum(single) + ro + rz + oz + triple - full),
        abs(offset_ref + size_ref - geometry), abs(offset_student + size_student - (v(1, 1, 1) - v(1, 0, 0))))
    if residual > CONFIG["decomposition_atol"]:
        raise ValueError("Factorial accounting identity failed")
    return dict(response_only_gain=single[0], offset_only_gain=single[1], size_only_gain=single[2],
        geometry_gain=geometry, full_gain=full, response_offset_interaction=ro, response_size_interaction=rz,
        offset_size_interaction=oz, three_way_interaction=triple,
        order_averaged_response_gain=float(ordered[0]), order_averaged_offset_gain=float(ordered[1]), order_averaged_size_gain=float(ordered[2]),
        offset_given_reference_response=offset_ref, size_given_reference_response=size_ref,
        offset_given_student_response=offset_student, size_given_student_response=size_student,
        max_accounting_residual=float(residual))


def anchor_pairs(anchor):
    if anchor["kind"] == "first_sustained_discrepancy":
        return [(METHODS.index(anchor["reference"]), METHODS.index(anchor["owner"]))]
    owner = METHODS.index(anchor["owner"])
    return [(owner, s) for s in (2, 3)] if owner < 2 else [(r, owner) for r in (0, 1)]


def analyze_video(video, data, source, legacy):
    factorial_rows, corner_rows, timeline = [], [], []
    selected_points = {a["frame_id"] for a in video["anchors"]} | {t["frame_id"] for t in video["frames"] if t["controls"]}
    membership = {f["frame_id"]: f for f in video["frames"]}
    for row, frame in enumerate(data["frame_id"]):
        frame = int(frame)
        task = membership[frame]
        for parent in range(4):
            for reference in range(2):
                for student in (2, 3):
                    key = (video["sequence"], frame, METHODS[parent], METHODS[reference], METHODS[student])
                    if key not in legacy:
                        raise ValueError("Missing original Stage-36 hybrid row")
                    values, floating, details = factorial(data, row, parent, reference, student, source["search_size"])
                    endpoint = [values[0, 0, 0], values[1, 0, 0], values[0, 1, 1], values[1, 1, 1]]
                    expected = [float(legacy[key][k]) for k in ("baseline_iou", "score_swap_iou", "geometry_swap_iou", "student_iou")]
                    error = float(np.max(np.abs(np.asarray(endpoint) - expected)))
                    if error > CONFIG["endpoint_atol"]:
                        raise ValueError("Eight-corner audit changed an original Stage-36 endpoint")
                    metrics, float_metrics = split_values(values), split_values(floating)
                    identity = dict(sequence=video["sequence"], frame_id=frame, parent_method=METHODS[parent],
                        reference=METHODS[reference], student=METHODS[student])
                    result = dict(**identity, fixed_time_control=bool(task["controls"]), anchor_frame=frame in {a["frame_id"] for a in video["anchors"]},
                        **{"iou_{}{}{}".format(*c): float(values[c]) for c in CORNERS}, **metrics,
                        floating_full_gain=float_metrics["full_gain"], floating_geometry_gain=float_metrics["geometry_gain"],
                        floating_offset_given_reference_response=float_metrics["offset_given_reference_response"],
                        floating_size_given_reference_response=float_metrics["size_given_reference_response"],
                        floating_order_averaged_offset_gain=float_metrics["order_averaged_offset_gain"],
                        floating_order_averaged_size_gain=float_metrics["order_averaged_size_gain"], legacy_endpoint_max_error=error)
                    factorial_rows.append(result)
                    if frame in selected_points:
                        for corner, detail in details.items():
                            corner_rows.append(dict(**identity, corner="{}{}{}".format(*corner), **detail))
        for anchor in video["anchors"]:
            if not any(m["anchor_id"] == anchor["anchor_id"] for m in task["anchors"]):
                continue
            for reference, student in anchor_pairs(anchor):
                for parent in (reference, student):
                    geom = scene_geometry(data["previous"][row, parent], data["gt"][row], data["image_hw"][row], source["search_factor"], source["search_size"])
                    gt, previous = data["gt"][row], data["previous"][row, parent]
                    for evaluator in (reference, student):
                        cell = int(np.argmax(data["raw"][row, parent, evaluator] * data["window"]))
                        unclipped, clipped = decode_cell(data["size"][row, parent, evaluator], data["offset"][row, parent, evaluator], cell,
                            previous, data["factor"][row, parent], source["search_size"], data["image_hw"][row])
                        metrics = box_metrics(data["native_boxes"][row, parent, evaluator], unclipped, clipped, gt)
                        offset = frame - anchor["frame_id"]
                        timeline.append(dict(sequence=video["sequence"], anchor_id=anchor["anchor_id"], anchor_kind=anchor["kind"],
                            anchor_owner=anchor["owner"], direction=anchor["direction"], anchor_frame=anchor["frame_id"], frame_id=frame,
                            relative_frame=offset, phase="before" if offset < 0 else "after" if offset > 0 else "anchor",
                            reference=METHODS[reference], student=METHODS[student], parent_method=METHODS[parent], evaluator=METHODS[evaluator],
                            coverage=float(geom["coverage"]), projected_gt_side_px=float(np.sqrt(geom["input_wh"].prod())),
                            parent_center_error_gt_units=float(np.linalg.norm(previous[:2] + previous[2:] / 2 - gt[:2] - gt[2:] / 2) / np.sqrt(gt[2:].prod())),
                            parent_side_ratio=float(np.sqrt(previous[2:].prod() / gt[2:].prod())), **metrics))
    return factorial_rows, corner_rows, timeline


def average_rows(rows, keys, metrics):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key not in groups:
            groups[key] = []
        groups[key].append(row)
    output = []
    for key, values in sorted(groups.items()):
        item = dict(zip(keys, key))
        item["sampled_frames"] = len(values)
        item.update({"mean_" + k: float(np.mean([v[k] for v in values])) for k in metrics})
        output.append(item)
    return output


def load_completed(root):
    root = Path(root).resolve()
    plan, source, traces = load_inputs(root)
    run = load_sealed(root / "run.json")
    diagnostic_path = root / "analysis/stage36_diagnostic.json"
    diagnostic = load_sealed(diagnostic_path)
    if (diagnostic.get("status") != "COMPLETE_SAME_CROP_DIAGNOSTIC" or diagnostic.get("audited_cases") != 4 or
            diagnostic.get("missing_cases") != [] or diagnostic.get("plan_fingerprint") != plan["fingerprint"] or
            diagnostic.get("run_fingerprint") != run["fingerprint"] or run.get("plan_fingerprint") != plan["fingerprint"] or
            run.get("methods") != list(METHODS) or run.get("parents") != list(METHODS)):
        raise ValueError("Need the complete, unmixed original four-case Stage-36 result")
    bindings = dict(plan["source_file_sha256"])
    bindings.update({str(root / "plan.json"): file_hash(root / "plan.json"), str(root / "run.json"): file_hash(root / "run.json"),
        str(diagnostic_path): file_hash(diagnostic_path)})
    expected = {v["sequence"] + "__stage36_same_crop.npz" for v in plan["videos"]}
    if set(diagnostic["response_cache_sha256"]) != expected or {p.name for p in (root / "responses").glob("*.npz")} != expected:
        raise ValueError("Need all four original Stage-36 replay caches")
    bindings.update({str(root / "responses" / name): digest for name, digest in diagnostic["response_cache_sha256"].items()})
    bindings.update({str(root / "analysis" / name): digest for name, digest in diagnostic["table_sha256"].items()})
    bindings.update({str(root / "analysis" / p["file"]): p["sha256"] for p in diagnostic["figures"]})
    verify_bindings(bindings)
    legacy_path = root / "analysis/stage36_hybrids.csv"
    if str(legacy_path) not in bindings:
        raise ValueError("Original hybrid table was not frozen")
    with legacy_path.open(newline="", encoding="utf-8") as handle:
        legacy_rows = list(csv.DictReader(handle))
    legacy = {}
    for row in legacy_rows:
        key = (row["sequence"], int(row["frame_id"]), row["parent_method"], row["reference"], row["student"])
        if key in legacy:
            raise ValueError("Duplicate original hybrid row")
        legacy[key] = row
    expected_rows = 16 * sum(len(v["frames"]) for v in plan["videos"])
    if len(legacy) != expected_rows:
        raise ValueError("Original hybrid table coverage differs from plan")
    return plan, source, traces, run, diagnostic, legacy, bindings


def audit(root):
    root = Path(root).resolve()
    plan, source, traces, run, original, legacy, bindings = load_completed(root)
    output = root / "analysis/geometry_factor_audit_v1"
    code = {p.name: file_hash(p) for p in (Path(__file__), Path(__file__).with_name("test_stage36_geometry_audit.py"))}
    source_digest = dict(schema=SCHEMA, configuration=CONFIG, source_plan_fingerprint=plan["fingerprint"],
        source_diagnostic_fingerprint=original["fingerprint"], source_file_sha256=bindings, code_sha256=code)
    with writer_lock(output):
        report_path = output / "stage36b_diagnostic.json"
        if report_path.exists():
            saved = read_sealed(report_path, SCHEMA)
            if any(saved.get(k) != v for k, v in source_digest.items()):
                raise ValueError("Completed geometry audit source/code/configuration changed")
            verify_bindings({str(output / name): digest for name, digest in saved["table_sha256"].items()})
            print("Stage-36B: COMPLETE_CACHED_GEOMETRY_AUDIT (reuse, audited)")
            return saved
    entries = {e["sequence"]: e for e in source["fit"]}
    all_factors, all_corners, all_timeline = [], [], []
    for video in plan["videos"]:
        name = video["sequence"]
        data = load_response(root / "responses" / (name + "__stage36_same_crop.npz"), source, entries[name], traces[name], video, run)
        factors, corners, timeline = analyze_video(video, data, source, legacy)
        all_factors.extend(factors)
        all_corners.extend(corners)
        all_timeline.extend(timeline)
        print("[geometry factors] {}: {} cached images".format(name, len(data["frame_id"])), flush=True)
    keys = ("sequence", "parent_method", "reference", "student")
    factor_summary = average_rows(all_factors, keys, SUMMARY_FIELDS)
    focus = []
    for video in plan["videos"]:
        for reference in METHODS[:2]:
            rows = [r for r in all_factors if r["sequence"] == video["sequence"] and r["parent_method"] == reference and r["reference"] == reference]
            focus.append(dict(sequence=video["sequence"], reference=reference,
                sampled_frames=len({r["frame_id"] for r in rows}), seed_repeats=len({r["student"] for r in rows}),
                **{"mean_" + k: float(np.mean([r[k] for r in rows])) for k in SUMMARY_FIELDS}))
    phase_summary = []
    for phase, flag in (("fixed_time_controls", "fixed_time_control"), ("anchor_frames", "anchor_frame")):
        phase_summary.extend(dict(phase=phase, **r) for r in average_rows([r for r in all_factors if r[flag]], keys, SUMMARY_FIELDS))
    time_keys = ("sequence", "anchor_id", "anchor_kind", "anchor_owner", "direction", "anchor_frame",
        "reference", "student", "parent_method", "evaluator", "phase")
    timeline_summary = average_rows(all_timeline, time_keys, STATE_FIELDS)
    tables = dict(factorial_frames=all_factors, factor_summary=factor_summary, geometry_focus=focus, phase_factor_summary=phase_summary,
        corner_metrics=all_corners, timeline=all_timeline, timeline_summary=timeline_summary,
        anchor_points=[r for r in all_timeline if r["relative_frame"] == 0])
    with writer_lock(output):
        report_path = output / "stage36b_diagnostic.json"
        if report_path.exists():
            saved = read_sealed(report_path, SCHEMA)
            if any(saved.get(k) != v for k, v in source_digest.items()):
                raise ValueError("Completed geometry audit source/code/configuration changed")
            verify_bindings({str(output / name): digest for name, digest in saved["table_sha256"].items()})
            print("Stage-36B: COMPLETE_CACHED_GEOMETRY_AUDIT (reuse, audited)")
            return saved
        table_hashes = {}
        for label, values in tables.items():
            path = output / ("stage36b_" + label + ".csv")
            if values:
                write_table(path, values)
                table_hashes[path.name] = file_hash(path)
        verify_bindings(bindings)
        report = seal(dict(**source_digest, status="COMPLETE_CACHED_GEOMETRY_AUDIT", role="posthoc_cached_response_offset_size_diagnosis",
            audited_cases=len(plan["videos"]), sampled_frame_images=sum(len(v["frames"]) for v in plan["videos"]),
            factorial_pairs=len(all_factors), timeline_rows=len(all_timeline), table_sha256=table_hashes,
            maximum_original_endpoint_error=max(r["legacy_endpoint_max_error"] for r in all_factors),
            maximum_factorial_accounting_residual=max(r["max_accounting_residual"] for r in all_factors),
            training_performed=False, new_inference_performed=False, adoption_gate=None,
            warnings=["Same consumed four cases, frames and anchors as Stage-36; no new validation sample or method test.",
                "Response chooses a grid cell; offset is only the sub-cell center correction, not the entire center prediction.",
                "Size changes can change final centers through image-boundary clipping; pre-clipping and post-clipping geometry are both reported.",
                "Six-order averaged allocations are accounting conventions for interactions, not unique physical causal contributions.",
                "Official IoU truncates predicted boxes; untruncated IoU is a secondary quantization-sensitivity diagnostic.",
                "Temporal windows are outcome-conditioned and may overlap. Before/after means do not establish causal time ordering.",
                "IoU gains use 0-1 units, not AUC points. No strategy, speedup or novel method PASS is assigned."],
            next_step="Use center-offset, size and inherited-anchor timelines to specify a targeted hypothesis; preserve all original failures"))
        publish_identical(report_path, report)
    print("Stage-36B: COMPLETE_CACHED_GEOMETRY_AUDIT; no training or GPU inference")
    print("Reference-parent mean IoU gains; offset/size allocations keep the reference response fixed:")
    print("sequence                 reference       offset       size   geometry       full")
    for r in focus:
        print("{:<24s} {:<12s} {:+.6f} {:+.6f} {:+.6f} {:+.6f}".format(r["sequence"], r["reference"],
            r["mean_offset_given_reference_response"], r["mean_size_given_reference_response"],
            r["mean_geometry_gain"], r["mean_full_gain"]))
    print("Reports: " + str(output))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True, help="Completed Stage-36 root")
    args = parser.parse_args()
    audit(Path(args.input).expanduser())


if __name__ == "__main__":
    main()
