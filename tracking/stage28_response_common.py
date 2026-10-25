"""Frozen spatial-response diagnostics on the original Stage-27 crop states."""

import json
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint
from stage27_uav_common import (failure_events, load_manifest, load_sealed as load_stage27,
                               load_trace, seal, tracking_iou)


SCHEMA = "stage28_response_competition_v1"
DESIGN = dict(event_before=10, event_after=30, control_quartiles=[1, 2, 3],
              topk=5, local_peak_radius=1, box_nms_iou=0.5,
              gt_match_iou=0.5, non_target_iou=0.1,
              replay_box_atol=0.001, replay_score_atol=0.00001,
              decode_box_atol=0.001, seed=2092)


def load_sealed(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: v for k, v in value.items() if k != "fingerprint"}
    if value.get("schema") != SCHEMA or value.get("fingerprint") != fingerprint(body):
        raise ValueError("Stage-28 schema/fingerprint mismatch: " + str(path))
    return value


def load_source(directory):
    directory = Path(directory).expanduser().resolve()
    run = load_stage27(directory / "run.json")
    if run.get("role") != "original_sgla_descriptive_uav_diagnosis":
        raise ValueError("Expected the original completed Stage-27 run")
    manifest = load_manifest(run["manifest"])
    for name, digest in run["code_sha256"].items():
        if file_hash(Path(__file__).parent / name) != digest:
            raise ValueError("Frozen Stage-27 dependency changed: " + name)
    expected = {e["sequence"] + "__stage27_uav.npz" for e in manifest["sequences"]}
    found = {p.name for p in (directory / "traces").glob("*.npz")}
    if expected != found:
        raise ValueError("Need the complete, unmixed Stage-27 traces")
    traces, hashes = {}, {}
    for entry in manifest["sequences"]:
        path = directory / "traces" / (entry["sequence"] + "__stage27_uav.npz")
        traces[entry["sequence"]] = (entry, load_trace(path, entry, run))
        hashes[path.name] = file_hash(path)
    return directory, run, traces, hashes


def plan_video(entry, data):
    n = len(data["frame_id"])
    selected, events = {}, []
    for quartile in DESIGN["control_quartiles"]:
        frame = max(1, n * quartile // 4)
        selected.setdefault(frame, dict(frame_id=frame, controls=[], events=[]))["controls"].append(quartile)
    for number, event in enumerate(failure_events(data["iou"]), 1):
        onset = int(data["frame_id"][event["start"]])
        first, stop = max(1, onset - DESIGN["event_before"]), min(n + 1, onset + DESIGN["event_after"] + 1)
        events.append(dict(sequence=entry["sequence"], source_group_proxy=entry["source_group_proxy"],
                           event=number, onset_frame=onset, first_frame=first, stop_frame_exclusive=stop))
        for frame in range(first, stop):
            item = selected.setdefault(frame, dict(frame_id=frame, controls=[], events=[]))
            item["events"].append(dict(event=number, onset_frame=onset, offset=frame - onset))
    return dict(sequence=entry["sequence"], source_group_proxy=entry["source_group_proxy"],
                frames=[selected[f] for f in sorted(selected)]), events


def make_plan(directory, source, traces, hashes):
    videos, events = [], []
    for entry, data in traces.values():
        video, ev = plan_video(entry, data)
        videos.append(video)
        events.extend(ev)
    return seal(dict(schema=SCHEMA, role="frozen_descriptive_response_replay_plan",
        source_input=str(directory), source_run_fingerprint=source["fingerprint"],
        source_run_sha256=file_hash(directory / "run.json"),
        source_manifest_fingerprint=source["manifest"]["fingerprint"], source_trace_sha256=hashes,
        design=DESIGN, videos=videos, events=events,
        unique_replay_frames=sum(len(v["frames"]) for v in videos),
        controls="fixed video-time quartiles, no success filtering; not necessarily stable controls",
        interpretation="outcome-conditioned development diagnosis, not policy AUC or an independent validation",
        code_sha256={name: file_hash(Path(__file__).parent / name)
                     for name in ("stage28_response_common.py", "stage28_response_collect.py")}))


def validate_plan(plan, source_info):
    directory, source, traces, hashes = source_info
    if plan != make_plan(directory, source, traces, hashes):
        raise ValueError("Plan/source/code changed; use the frozen configuration and a new output path")


def cpu_array(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    return value.detach().float().cpu().numpy().copy()


def clip_boxes(boxes, image_hw, margin=10):
    """Vectorized copy of lib.utils.box_ops.clip_box, including minimum size."""
    boxes = np.asarray(boxes, float)
    h, w = image_hw
    if min(h, w) < margin:
        raise ValueError("Image smaller than original decoder margin")
    lo = np.minimum(np.maximum(boxes[..., :2], 0), [w - margin, h - margin])
    hi = np.minimum(np.maximum(boxes[..., :2] + boxes[..., 2:], margin), [w, h])
    return np.concatenate([lo, np.maximum(margin, hi - lo)], axis=-1)


def dense_boxes(size_map, offset_map, previous, factor, search_size, image_hw):
    """Decode every CENTER grid cell using the original floating parent center.

    Do not map using rounded crop x/y: the native decoder uses previous center.
    Float32 normalized-box arithmetic precedes Python/float64 image mapping.
    """
    size_map, offset_map = np.asarray(size_map, np.float32), np.asarray(offset_map, np.float32)
    if (size_map.ndim != 3 or size_map.shape != offset_map.shape or size_map.shape[0] != 2
            or size_map.shape[1] != size_map.shape[2] or factor <= 0
            or not np.isfinite(size_map).all() or not np.isfinite(offset_map).all()
            or (size_map < 0).any()):
        raise ValueError("Invalid CENTER size/offset maps")
    side = size_map.shape[1]
    yy, xx = np.indices((side, side), dtype=np.float32)
    center = np.stack([xx + offset_map[0], yy + offset_map[1]], axis=-1) / np.float32(side)
    normalized = np.concatenate([center, size_map.transpose(1, 2, 0)], axis=-1)
    physical = ((normalized * np.float32(search_size)) / np.float32(factor)).astype(np.float64)
    parent = np.asarray(previous, float)
    image_center = physical[..., :2] + parent[:2] + parent[2:] / 2 - 0.5 * search_size / factor
    boxes = np.concatenate([image_center - physical[..., 2:] / 2, physical[..., 2:]], axis=-1)
    return clip_boxes(boxes, image_hw).reshape(-1, 4)


def continuous_iou(box, others):
    box, others = np.asarray(box, float), np.asarray(others, float)
    lo = np.maximum(box[:2], others[..., :2])
    hi = np.minimum(box[:2] + box[2:], others[..., :2] + others[..., 2:])
    inter = np.maximum(0, hi - lo).prod(-1)
    return inter / np.maximum(np.finfo(float).tiny, box[2:].prod() + others[..., 2:].prod(-1) - inter)


def local_peak_mask(scores):
    scores = np.asarray(scores)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("Expected a finite response map")
    h, w = scores.shape
    padded = np.pad(scores.astype(float), 1, constant_values=-np.inf)
    neighbour = np.maximum.reduce([padded[y:y + h, x:x + w] for y in range(3) for x in range(3)])
    # Equal-score plateaus use deterministic row-major tie breaking downstream.
    return (scores >= neighbour) & (scores > 0)


def peak_candidates(scores, boxes):
    scores = np.asarray(scores)
    flat = scores.ravel()
    peaks = np.flatnonzero(local_peak_mask(scores))
    order = peaks[np.lexsort((peaks, -flat[peaks]))]
    selected = []
    for index in order:
        if selected and (continuous_iou(boxes[index], boxes[selected]) > DESIGN["box_nms_iou"]).any():
            continue
        selected.append(int(index))
        if len(selected) == DESIGN["topk"]:
            break
    if not selected or selected[0] != int(np.argmax(flat)):
        raise ValueError("Peak extraction did not preserve native row-major argmax")
    return selected


def bind_native_top1(raw, window, boxes, native_box):
    boxes = np.asarray(boxes, float).copy()
    boxes[int(np.argmax(np.asarray(raw) * np.asarray(window)))] = native_box
    return boxes


def response_metrics(raw, window, boxes, gt, native_box=None):
    raw, window = np.asarray(raw, np.float32), np.asarray(window, np.float32)
    if (raw.ndim != 2 or raw.shape != window.shape or not np.isfinite(raw).all()
            or not np.isfinite(window).all() or (raw <= 0).any() or (window < 0).any()
            or not (window > 0).any()):
        raise ValueError("Invalid raw/window response")
    if native_box is not None:
        boxes = bind_native_top1(raw, window, boxes, native_box)
    overlap = tracking_iou(boxes, gt)
    matched, other = overlap >= DESIGN["gt_match_iou"], overlap < DESIGN["non_target_iou"]
    result = dict(dense_best_gt_iou=float(overlap.max()), dense_gt_match_cells=int(matched.sum()))
    details = {}
    for kind, score in (("raw", raw), ("weighted", raw * window)):
        flat = score.ravel()
        indices = peak_candidates(score, boxes)
        ious = overlap[indices]
        order = np.lexsort((np.arange(len(flat)), -flat))
        gt_ranks = np.flatnonzero(matched[order])
        candidates = np.flatnonzero(ious >= DESIGN["gt_match_iou"])
        best_gt = None if not matched.any() else float(flat[matched].max())
        best_other = None if not other.any() else float(flat[other].max())
        result.update({kind + "_top1_iou": float(ious[0]), kind + "_candidate_count": len(indices),
            kind + "_top5_gt_recall": bool(len(candidates)),
            kind + "_gt_rank_in_top5": None if not len(candidates) else int(candidates[0] + 1),
            kind + "_best_gt_dense_score_rank": None if not len(gt_ranks) else int(gt_ranks[0] + 1),
            kind + "_gt_local_peak_present": bool((matched & local_peak_mask(score).ravel()).any()),
            kind + "_best_gt_score": best_gt, kind + "_best_non_target_score": best_other,
            kind + "_gt_score_margin": None if best_gt is None or best_other is None else best_gt - best_other})
        details[kind] = dict(indices=indices, iou=ious, score=flat[indices])
    return result, details, overlap


def observation_label(metrics, coverage):
    if metrics["weighted_top1_iou"] >= .5:
        return "gt_matched_native_top1"
    if metrics["weighted_top1_iou"] >= .2:
        return "intermediate_native_localization"
    if coverage < .9:
        return "low_iou_with_low_crop_coverage"
    if metrics["dense_best_gt_iou"] < .5:
        return "inside_no_gt_matching_dense_box"
    if metrics["raw_top1_iou"] >= .5:
        return "inside_window_demotes_correct_raw_top1"
    if metrics["weighted_top5_gt_recall"]:
        return "inside_gt_in_weighted_top5_not_top1"
    if metrics["raw_top5_gt_recall"]:
        return "inside_gt_only_in_raw_top5"
    return "inside_gt_box_exists_not_retained_by_fixed_top5"


def audit_response(data, entry, trace, video, source_run):
    frame_ids = np.array([item["frame_id"] for item in video["frames"]], dtype=np.int64)
    n = len(frame_ids)
    side = int(source_run["runtime"]["lens_x"] ** .5)
    shapes = dict(frame_id=(n,), raw=(n, side, side), size=(n, 2, side, side),
        offset=(n, 2, side, side), window=(side, side), previous=(n, 4), gt=(n, 4),
        factor=(n,), image_hw=(n, 2), native_boxes=(n, 4), raw_boxes=(n, 4),
        layer=(n,), replay_box_error=(n,), replay_peak_error=(n,), decode_box_error=(n,))
    if set(data) != set(shapes) or any(np.shape(data[k]) != shape for k, shape in shapes.items()):
        raise ValueError("Response cache shape/coverage mismatch: " + entry["sequence"])
    if not all(np.isfinite(v).all() for v in data.values()):
        raise ValueError("Nonfinite response cache")
    np.testing.assert_array_equal(data["frame_id"], frame_ids)
    index = frame_ids - 1
    for field in ("previous", "gt", "factor", "image_hw", "layer"):
        np.testing.assert_array_equal(data[field], trace[field][index])
    native_errors = np.max(np.abs(data["native_boxes"] - trace["boxes"][index]), axis=1)
    peak_errors = np.abs(data["raw"].max(axis=(1, 2)) - trace["raw_score_peak"][index])
    np.testing.assert_allclose(data["replay_box_error"], native_errors, atol=1e-12, rtol=0)
    np.testing.assert_allclose(data["replay_peak_error"], peak_errors, atol=1e-12, rtol=0)
    if native_errors.max() > DESIGN["replay_box_atol"] or peak_errors.max() > DESIGN["replay_score_atol"]:
        raise ValueError("Frozen source replay failed")
    np.testing.assert_allclose(tracking_iou(data["native_boxes"], data["gt"]), trace["iou"][index],
                               atol=1e-12, rtol=0, err_msg="Replay changed the official integer-box IoU")
    for i in range(n):
        boxes = dense_boxes(data["size"][i], data["offset"][i], data["previous"][i],
                            data["factor"][i], source_run["search_size"], data["image_hw"][i])
        wi = int(np.argmax(data["raw"][i] * data["window"]))
        ri = int(np.argmax(data["raw"][i]))
        error = float(np.max(np.abs(boxes[wi] - data["native_boxes"][i])))
        np.testing.assert_allclose(data["decode_box_error"][i], error, atol=1e-12, rtol=0)
        if error > DESIGN["decode_box_atol"]:
            raise ValueError("Dense decode disagrees with original CENTER decoder")
        boxes = bind_native_top1(data["raw"][i], data["window"], boxes, data["native_boxes"][i])
        response_metrics(data["raw"][i], data["window"], boxes, data["gt"][i])
        np.testing.assert_allclose(data["raw_boxes"][i], boxes[ri], atol=1e-12, rtol=0)


def load_response(path, entry, trace, video, source_run, run):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata_json"].item()))
        data = {k: archive[k] for k in archive.files if k != "metadata_json"}
    if meta != dict(run_fingerprint=run["fingerprint"], sequence=entry["sequence"]):
        raise ValueError("Response cache identity mismatch")
    audit_response(data, entry, trace, video, source_run)
    return data
