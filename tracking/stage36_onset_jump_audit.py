"""CPU-only peak/offset/crop accounting at two consumed Stage-36 failure onsets."""

import argparse
from pathlib import Path

import numpy as np

from stage4_common import file_hash
from stage27_uav_collect import writer_lock
from stage27_uav_common import nonempty_path, publish_identical, scene_geometry, seal
from stage27_uav_summary import write_table
from stage28_response_common import bind_native_top1, dense_boxes, observation_label, response_metrics
from stage34_identity_common import load_sealed as read_sealed
from stage36_same_crop_common import METHODS, verify_bindings
from stage36_same_crop_collect import load_response
import stage36_geometry_audit as geometry


SCHEMA = "stage36c_cached_onset_jump_v1"
CONFIG = dict(events=[
    dict(sequence="uav0000089_00920_s", student="supervised_seed2110", onset=724),
    dict(sequence="uav0000089_00920_s", student="supervised_seed2111", onset=641)],
    before=2, after=2, center_accounting_atol_px=1e-9, projection_zero_atol_px=1e-12,
    geometry="Floating-parent decoder center; signed x/y accounting, no new policy")
TERMS = ("parent_translation", "crop_scale", "response_grid", "offset_field", "offset_resampling",
    "float32_arithmetic", "boundary_clipping", "native_decode", "gt_motion")


def frame_state(data, row, parent, evaluator, search_size):
    raw, window = data["raw"][row, parent, evaluator], data["window"]
    side = raw.shape[1]
    cell, raw_cell = int(np.argmax(raw * window)), int(np.argmax(raw))
    y, x = divmod(cell, side)
    selected_offset = data["offset"][row, parent, evaluator, :, y, x].astype(float)
    previous, factor = data["previous"][row, parent], float(data["factor"][row, parent])
    parent_center = previous[:2] + previous[2:] / 2
    stride = search_size / factor / side
    grid = np.array([x, y], float)
    nominal = parent_center + stride * (grid + selected_offset - side / 2)
    unclipped, clipped = geometry.decode_cell(data["size"][row, parent, evaluator], data["offset"][row, parent, evaluator],
        cell, previous, factor, search_size, data["image_hw"][row])
    native = data["native_boxes"][row, parent, evaluator]
    center = lambda b: b[:2] + b[2:] / 2
    state = dict(cell=cell, raw_cell=raw_cell, grid=grid, side=side, selected_offset=selected_offset,
        offset_map=data["offset"][row, parent, evaluator], parent_center=parent_center, stride=stride,
        native_center=center(native), float32_delta=center(unclipped) - nominal,
        clipping_delta=center(clipped) - center(unclipped), native_delta=center(native) - center(clipped),
        gt=data["gt"][row], image_hw=data["image_hw"][row])
    boxes = dense_boxes(data["size"][row, parent, evaluator], data["offset"][row, parent, evaluator], previous,
        factor, search_size, data["image_hw"][row])
    boxes = bind_native_top1(raw, window, boxes, native)
    metrics, candidates, overlap = response_metrics(raw, window, boxes, data["gt"][row])
    shape_metrics = geometry.box_metrics(native, unclipped, clipped, data["gt"][row])
    return state, metrics, shape_metrics, boxes, candidates, overlap


def center_accounting(before, after):
    if before["side"] != after["side"] or not np.array_equal(before["image_hw"], after["image_hw"]):
        raise ValueError("Temporal grid/image resolution changed; fixed pixel-coordinate accounting is unsupported")
    def offset(state, grid):
        x, y = map(int, grid)
        return np.asarray(state["offset_map"][:, y, x], float)
    old_grid, new_grid = before["grid"], after["grid"]
    oo, on = offset(before, old_grid), offset(before, new_grid)
    no, nn = offset(after, old_grid), offset(after, new_grid)
    field = .5 * ((no - oo) + (nn - on))
    resampling = .5 * ((nn - no) + (on - oo))
    gt_center = lambda s: s["gt"][:2] + s["gt"][2:] / 2
    gt_scale = float(np.sqrt(after["gt"][2:].prod()))
    vectors = dict(parent_translation=after["parent_center"] - before["parent_center"],
        crop_scale=(after["stride"] - before["stride"]) * (old_grid + oo - before["side"] / 2),
        response_grid=after["stride"] * (new_grid - old_grid),
        offset_field=after["stride"] * field, offset_resampling=after["stride"] * resampling,
        float32_arithmetic=after["float32_delta"] - before["float32_delta"],
        boundary_clipping=after["clipping_delta"] - before["clipping_delta"],
        native_decode=after["native_delta"] - before["native_delta"], gt_motion=-(gt_center(after) - gt_center(before)))
    output_change = after["native_center"] - before["native_center"]
    error_change = output_change + vectors["gt_motion"]
    residual = max(float(np.max(np.abs(sum(vectors[t] for t in TERMS[:-1]) - output_change))),
        float(np.max(np.abs(sum(vectors.values()) - error_change))),
        float(after["stride"] * np.max(np.abs(field + resampling - (nn - oo)))))
    if residual > CONFIG["center_accounting_atol_px"]:
        raise ValueError("Center-transition vector accounting failed")
    length = float(np.linalg.norm(error_change))
    components = []
    for term in TERMS:
        v = vectors[term]
        components.append(dict(term=term, dx_px=float(v[0]), dy_px=float(v[1]),
            magnitude_gt_units=float(np.linalg.norm(v) / gt_scale),
            along_error_gt_units=float(np.dot(v, error_change) / length / gt_scale) if length > CONFIG["projection_zero_atol_px"] else None))
    projection_residual = None
    if length > CONFIG["projection_zero_atol_px"]:
        projection_residual = abs(sum(c["along_error_gt_units"] for c in components) - length / gt_scale)
        if projection_residual * gt_scale > CONFIG["center_accounting_atol_px"]:
            raise ValueError("Signed projection accounting failed")
    return dict(peak_changed=bool(np.any(new_grid != old_grid)),
        peak_grid_distance=float(np.linalg.norm(new_grid - old_grid)),
        native_center_displacement_gt_units=float(np.linalg.norm(output_change) / gt_scale),
        error_vector_change_gt_units=length / gt_scale, accounting_residual_px=residual,
        projection_residual_gt_units=projection_residual), components


def analyze_event(event, data, source, video):
    required = list(range(event["onset"] - CONFIG["before"], event["onset"] + CONFIG["after"] + 1))
    rows = {int(f): i for i, f in enumerate(data["frame_id"])}
    if not set(required) <= set(rows):
        raise ValueError("Required onset context is absent from frozen Stage-36 cache; do not add GPU replay")
    if event["student"] not in METHODS[2:] or not any(a["kind"] == "first_failure" and a["owner"] == event["student"] and
            a["frame_id"] == event["onset"] for a in video["anchors"]):
        raise ValueError("Requested onset is not the owner's original planned first failure")
    frame_rows, candidate_rows, transition_rows, component_rows, states = [], [], [], [], {}
    for frame in required:
        row = rows[frame]
        for parent, parent_name in enumerate(METHODS):
            geom = scene_geometry(data["previous"][row, parent], data["gt"][row], data["image_hw"][row], source["search_factor"], source["search_size"])
            for evaluator, method in enumerate(METHODS):
                state, metrics, shapes, boxes, candidates, overlap = frame_state(data, row, parent, evaluator, source["search_size"])
                states[(frame, parent, evaluator)] = state
                identity = dict(sequence=event["sequence"], event_student=event["student"], onset_frame=event["onset"],
                    frame_id=frame, relative_frame=frame - event["onset"], parent_method=parent_name, evaluator=method)
                selected = data["size"][row, parent, evaluator].reshape(2, -1)[:, state["cell"]]
                frame_rows.append(dict(**identity, weighted_peak_cell=state["cell"], raw_peak_cell=state["raw_cell"],
                    grid_x=int(state["grid"][0]), grid_y=int(state["grid"][1]),
                    selected_offset_x=float(state["selected_offset"][0]), selected_offset_y=float(state["selected_offset"][1]),
                    selected_size_w=float(selected[0]), selected_size_h=float(selected[1]),
                    decoder_stride_px=state["stride"], coverage=float(geom["coverage"]),
                    projected_gt_side_px=float(np.sqrt(geom["input_wh"].prod())),
                    parent_center_error_gt_units=float(np.linalg.norm(state["parent_center"] - data["gt"][row, :2] - data["gt"][row, 2:] / 2) / np.sqrt(data["gt"][row, 2:].prod())),
                    **shapes, **metrics, observation=observation_label(metrics, geom["coverage"])))
                for ranking, detail in candidates.items():
                    for rank, cell in enumerate(detail["indices"], 1):
                        candidate_rows.append(dict(**identity, ranking=ranking, rank=rank, cell=cell,
                            score=float(detail["score"][rank - 1]), gt_iou=float(overlap[cell]),
                            x=float(boxes[cell, 0]), y=float(boxes[cell, 1]), width=float(boxes[cell, 2]), height=float(boxes[cell, 3])))
    for first, second in zip(required[:-1], required[1:]):
        if second != first + 1:
            raise ValueError("Only consecutive-frame transitions are supported")
        for parent, parent_name in enumerate(METHODS):
            for evaluator, method in enumerate(METHODS):
                before, after = states[(first, parent, evaluator)], states[(second, parent, evaluator)]
                accounting, components = center_accounting(before, after)
                identity = dict(sequence=event["sequence"], event_student=event["student"], onset_frame=event["onset"],
                    from_frame=first, to_frame=second, parent_method=parent_name, evaluator=method)
                transition_rows.append(dict(**identity, from_peak_cell=before["cell"], to_peak_cell=after["cell"],
                    from_offset_x=float(before["selected_offset"][0]), from_offset_y=float(before["selected_offset"][1]),
                    to_offset_x=float(after["selected_offset"][0]), to_offset_y=float(after["selected_offset"][1]), **accounting))
                component_rows.extend(dict(**identity, **c) for c in components)
    native = lambda r: r["parent_method"] == event["student"] and r["evaluator"] == event["student"]
    transition = next(r for r in transition_rows if native(r) and r["to_frame"] == event["onset"])
    previous = next(r for r in frame_rows if native(r) and r["frame_id"] == event["onset"] - 1)
    current = next(r for r in frame_rows if native(r) and r["frame_id"] == event["onset"])
    sgla = next(r for r in frame_rows if r["parent_method"] == event["student"] and r["evaluator"] == "sgla" and r["frame_id"] == event["onset"])
    components = {r["term"]: r for r in component_rows if native(r) and r["to_frame"] == event["onset"]}
    projection = lambda term: components[term]["along_error_gt_units"]
    def combined(terms):
        values = [projection(t) for t in terms]
        return sum(values) if all(v is not None for v in values) else None
    summary = dict(sequence=event["sequence"], student=event["student"], from_frame=event["onset"] - 1, to_frame=event["onset"],
        previous_iou=previous["weighted_top1_iou"], onset_iou=current["weighted_top1_iou"],
        from_peak_cell=transition["from_peak_cell"], to_peak_cell=transition["to_peak_cell"], peak_changed=transition["peak_changed"],
        error_jump_gt_units=transition["error_vector_change_gt_units"],
        grid_along_error_gt_units=projection("response_grid"), offset_field_along_error_gt_units=projection("offset_field"),
        offset_resampling_along_error_gt_units=projection("offset_resampling"),
        parent_and_scale_along_error_gt_units=combined(("parent_translation", "crop_scale")),
        decoder_along_error_gt_units=combined(("float32_arithmetic", "boundary_clipping", "native_decode")),
        gt_motion_along_error_gt_units=projection("gt_motion"),
        previous_center_error_gt_units=previous["center_error_gt_units"], onset_center_error_gt_units=current["center_error_gt_units"],
        onset_coverage=current["coverage"], previous_weighted_top5_gt_recall=previous["weighted_top5_gt_recall"],
        onset_weighted_top5_gt_recall=current["weighted_top5_gt_recall"], onset_dense_best_gt_iou=current["dense_best_gt_iou"],
        sgla_same_crop_iou=sgla["weighted_top1_iou"], sgla_same_crop_top5_gt_recall=sgla["weighted_top5_gt_recall"])
    return summary, frame_rows, candidate_rows, transition_rows, component_rows


def load_source(root):
    plan, source, traces, run, diagnostic, _, bindings = geometry.load_completed(root)
    prior_path = root / "analysis/geometry_factor_audit_v1/stage36b_diagnostic.json"
    prior = read_sealed(prior_path, geometry.SCHEMA)
    if (prior.get("status") != "COMPLETE_CACHED_GEOMETRY_AUDIT" or prior.get("configuration") != geometry.CONFIG or
            prior.get("source_plan_fingerprint") != plan["fingerprint"] or
            prior.get("source_diagnostic_fingerprint") != diagnostic["fingerprint"] or
            prior.get("training_performed") is not False or prior.get("new_inference_performed") is not False):
        raise ValueError("Need the frozen completed Stage-36B geometry audit")
    verify_bindings(prior["source_file_sha256"])
    for name, digest in prior["code_sha256"].items():
        if file_hash(Path(__file__).with_name(name)) != digest:
            raise ValueError("Frozen Stage-36B code changed")
    bindings.update(prior["source_file_sha256"])
    bindings[str(prior_path)] = file_hash(prior_path)
    bindings.update({str(prior_path.parent / name): digest for name, digest in prior["table_sha256"].items()})
    verify_bindings(bindings)
    return plan, source, traces, run, prior, bindings


def audit(root):
    root = Path(root).resolve()
    plan, source, traces, run, prior, bindings = load_source(root)
    output = root / "analysis/onset_jump_audit_v1"
    code = {p.name: file_hash(p) for p in (Path(__file__), Path(__file__).with_name("test_stage36_onset_jump.py"))}
    identity = dict(schema=SCHEMA, configuration=CONFIG, source_plan_fingerprint=plan["fingerprint"],
        source_geometry_audit_fingerprint=prior["fingerprint"], source_file_sha256=bindings, code_sha256=code)
    entries = {e["sequence"]: e for e in source["fit"]}
    videos = {v["sequence"]: v for v in plan["videos"]}
    tables = dict(events=[], frame_states=[], candidates=[], transitions=[], components=[], same_crop_onset=[])
    cached = {}
    with writer_lock(output):
        report_path = output / "stage36c_diagnostic.json"
        if report_path.exists():
            saved = read_sealed(report_path, SCHEMA)
            if any(saved.get(k) != v for k, v in identity.items()):
                raise ValueError("Completed onset audit source/code/configuration changed")
            verify_bindings({str(output / n): d for n, d in saved["table_sha256"].items()})
            print("Stage-36C: COMPLETE_CACHED_ONSET_AUDIT (reuse, audited)")
            return saved
        seen = set()
        for event in CONFIG["events"]:
            key = (event["sequence"], event["student"], event["onset"])
            if key in seen:
                raise ValueError("Duplicate target event")
            seen.add(key)
            name = event["sequence"]
            if name not in videos:
                raise ValueError("Target event is outside the frozen four cases")
            if name not in cached:
                path = root / "responses" / (name + "__stage36_same_crop.npz")
                cached[name] = load_response(path, source, entries[name], traces[name], videos[name], run)
            summary, frames, candidates, transitions, components = analyze_event(event, cached[name], source, videos[name])
            tables["events"].append(summary)
            tables["frame_states"].extend(frames)
            tables["candidates"].extend(candidates)
            tables["transitions"].extend(transitions)
            tables["components"].extend(components)
            tables["same_crop_onset"].extend(r for r in frames if r["relative_frame"] == 0)
        if not tables["events"]:
            raise ValueError("No fixed target event")
        table_hashes = {}
        for name, values in tables.items():
            path = output / ("stage36c_" + name + ".csv")
            write_table(path, values)
            table_hashes[path.name] = file_hash(path)
        verify_bindings(bindings)
        report = seal(dict(**identity, role="posthoc_consumed_failure_onset_center_accounting",
            status="COMPLETE_CACHED_ONSET_AUDIT", events=tables["events"],
            source_videos=len({e["sequence"] for e in CONFIG["events"]}),
            distinct_frame_images=len({(r["sequence"], r["frame_id"]) for r in tables["frame_states"]}),
            cached_model_crop_frame_views=len(tables["frame_states"]),
            maximum_accounting_residual_px=max(r["accounting_residual_px"] for r in tables["transitions"]),
            table_sha256=table_hashes, training_performed=False, new_inference_performed=False, adoption_gate=None,
            warnings=["These two events were chosen after consumed outcomes; no independent validation or new method test.",
                "Temporal fixed-grid offset changes do not compare the same physical pixels when crops change.",
                "Grid, offset resampling and offset-field allocations interact; accounting is not a unique causal explanation.",
                "Signed projections sum to the error-vector displacement; component magnitudes do not sum and projections can cancel.",
                "GT motion includes object/camera/image effects and is used only offline.",
                "GT candidate matching does not establish visibility or semantic distractor identity.",
                "No new output, continued trajectory, recovery policy, speed claim or method PASS is evaluated."],
            next_step="Inspect peak movement, fixed-grid offset changes and target retention before specifying a new training hypothesis"))
        publish_identical(report_path, report)
    print("Stage-36C: COMPLETE_CACHED_ONSET_AUDIT; signed contributions in GT-side units")
    print("student               frames       peak       grid  offset_field  resampling  parent+scale  total_error_jump  target_top5")
    for r in tables["events"]:
        fmt = lambda value: "null" if value is None else "{:+.5f}".format(value)
        print("{} {}->{} {}->{} {} {} {} {} {:.5f} {}->{}".format(r["student"], r["from_frame"], r["to_frame"],
            r["from_peak_cell"], r["to_peak_cell"], fmt(r["grid_along_error_gt_units"]), fmt(r["offset_field_along_error_gt_units"]),
            fmt(r["offset_resampling_along_error_gt_units"]), fmt(r["parent_and_scale_along_error_gt_units"]),
            r["error_jump_gt_units"], r["previous_weighted_top5_gt_recall"], r["onset_weighted_top5_gt_recall"]))
    print("Reports: " + str(output))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True, help="Completed Stage-36 root, including Stage-36B")
    args = parser.parse_args()
    audit(Path(args.input).expanduser())


if __name__ == "__main__":
    main()
