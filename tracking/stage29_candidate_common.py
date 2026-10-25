"""One-output spatial-candidate interventions followed by native SGLA rollouts."""

import json
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint
from stage27_uav_common import scene_geometry, seal, tracking_iou
from stage28_response_common import (DESIGN as RESPONSE_DESIGN, bind_native_top1, dense_boxes,
    load_response, load_sealed as load_stage28, load_source, response_metrics, validate_plan)


SCHEMA = "stage29_spatial_candidate_branch_v1"
DESIGN = dict(horizon=40, primary_expected_events=3, minimum_success_events=2,
              minimum_post_iou_gain=0.05, failure_iou=0.2, recovery_iou=0.5,
              recovery_run=3, inside_coverage=0.9, gt_match_iou=0.5, seed=2092)


def load_sealed(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: v for k, v in value.items() if k != "fingerprint"}
    if value.get("schema") != SCHEMA or value.get("fingerprint") != fingerprint(body):
        raise ValueError("Stage-29 schema/fingerprint mismatch: " + str(path))
    return value


def load_inputs(directory):
    """Audit all Stage-28 responses and their complete, unchanged Stage-27 source."""
    directory = Path(directory).expanduser().resolve()
    run = load_stage28(directory / "run.json")
    plan = load_stage28(directory / "plan.json")
    report = load_stage28(directory / "analysis" / "stage28_response_diagnostic.json")
    if (run.get("role") != "frozen_response_replay" or run.get("plan") != plan
            or report.get("role") != "descriptive_response_competition_audit"
            or report.get("status") != "COMPLETE_RESPONSE_DIAGNOSTIC"):
        raise ValueError("Need the complete Stage-28 diagnostic, not a pilot")
    info = load_source(plan["source_input"])
    validate_plan(plan, info)
    _, source, traces, _ = info
    expected = {v["sequence"] + "__stage28_response.npz" for v in plan["videos"]}
    if {p.name for p in (directory / "responses").glob("*.npz")} != expected:
        raise ValueError("Incomplete or mixed Stage-28 responses")
    caches, hashes = {}, {}
    for video in plan["videos"]:
        name = video["sequence"]
        entry, trace = traces[name]
        path = directory / "responses" / (name + "__stage28_response.npz")
        data = load_response(path, entry, trace, video, source, run)
        np.testing.assert_array_equal(data["window"], np.asarray(run["window"], np.float32))
        caches[name] = data
        hashes[path.name] = file_hash(path)
    if (report.get("cache_sha256") != hashes or report.get("missing_videos") != []
            or report.get("audited_videos") != len(caches)
            or report.get("replayed_unique_frames") != plan["unique_replay_frames"]
            or report.get("audited_event_onsets") != len(plan["events"])
            or report.get("planned_event_onsets") != len(plan["events"])
            or report.get("source_run_fingerprint") != source["fingerprint"]
            or report.get("plan_fingerprint") != plan["fingerprint"]):
        raise ValueError("Stage-28 completion report does not match audited caches")
    return dict(directory=directory, run=run, response_plan=plan, report=report,
                source_info=info, caches=caches, hashes=hashes)


def event_candidates(event, source, trace, data):
    frame = event["onset_frame"]
    found = np.flatnonzero(data["frame_id"] == frame)
    if len(found) != 1:
        raise ValueError("Missing unique onset response")
    i = int(found[0])
    boxes = dense_boxes(data["size"][i], data["offset"][i], data["previous"][i],
                        data["factor"][i], source["search_size"], data["image_hw"][i])
    boxes = bind_native_top1(data["raw"][i], data["window"], boxes, data["native_boxes"][i])
    metrics, details, overlaps = response_metrics(data["raw"][i], data["window"], boxes, data["gt"][i])
    candidates = [dict(rank=r + 1, grid_index=int(cell), box=boxes[cell].tolist(),
        score=float(details["weighted"]["score"][r]), onset_gt_iou=float(overlaps[cell]))
        for r, cell in enumerate(details["weighted"]["indices"])]
    best = int(np.argmax([c["onset_gt_iou"] for c in candidates]))
    gt_rank = best + 1 if candidates[best]["onset_gt_iou"] >= DESIGN["gt_match_iou"] else None
    coverage = float(trace["coverage"][frame - 1])
    primary = (coverage >= DESIGN["inside_coverage"] and metrics["weighted_top1_iou"] < DESIGN["failure_iou"]
               and gt_rank is not None)
    return candidates, gt_rank, bool(primary), coverage


def make_plan(inputs):
    directory, old_plan = inputs["directory"], inputs["response_plan"]
    _, source, traces, _ = inputs["source_info"]
    events = []
    for old in old_plan["events"]:
        name, frame = old["sequence"], old["onset_frame"]
        entry, trace = traces[name]
        if frame + DESIGN["horizon"] >= len(entry["frames"]):
            raise ValueError("Full H40 unavailable at {} event {}; no silent horizon truncation".format(name, old["event"]))
        candidates, gt_rank, primary, coverage = event_candidates(old, source, trace, inputs["caches"][name])
        events.append(dict(sequence=name, source_group_proxy=old["source_group_proxy"], event=old["event"],
            onset_frame=frame, future_first_frame=frame + 1,
            future_stop_frame_exclusive=frame + DESIGN["horizon"] + 1,
            onset_coverage=coverage, candidates=candidates,
            onset_gt_best_rank=gt_rank, primary_eligible=primary))
    primary = [e for e in events if e["primary_eligible"]]
    if (len(primary) != DESIGN["primary_expected_events"]
            or len({e["source_group_proxy"] for e in primary}) != DESIGN["primary_expected_events"]):
        raise ValueError("Expected exactly three Stage-28 correctable events; do not change the primary cohort after outcomes")
    branches = sum(len(e["candidates"]) for e in events)
    return seal(dict(schema=SCHEMA, role="frozen_spatial_candidate_intervention_plan",
        source_input=str(directory), source_stage28_run_fingerprint=inputs["run"]["fingerprint"],
        source_stage28_plan_fingerprint=old_plan["fingerprint"],
        source_diagnostic_fingerprint=inputs["report"]["fingerprint"],
        source_stage27_run_fingerprint=source["fingerprint"], source_response_sha256=inputs["hashes"],
        design=DESIGN, candidate_design=RESPONSE_DESIGN, events=events, branches=branches,
        onset_replay_forwards=branches, future_model_forwards=branches * DESIGN["horizon"],
        primary_selection="inside onset, native IoU<0.2, weighted top-5 contains IoU>=0.5; frozen before branch outcomes",
        primary_candidate="maximum onset GT IoU, rank breaks ties; no future-return selection",
        code_sha256={n: file_hash(Path(__file__).parent / n)
                     for n in ("stage29_candidate_common.py", "stage29_candidate_collect.py")}))


def validate_stage29_plan(plan, inputs):
    if plan != make_plan(inputs):
        raise ValueError("Stage-29 plan/source/code changed; preserve the old run and use a new path")


def event_filename(event):
    return "{}__e{:03d}__stage29_candidate.npz".format(event["sequence"], event["event"])


def event_metadata(run, event):
    return dict(run_fingerprint=run["fingerprint"], sequence=event["sequence"],
                event=event["event"], onset_frame=event["onset_frame"])


def audit_event(data, event, entry, trace, source):
    c, t = len(event["candidates"]), DESIGN["horizon"] + 1
    shapes = dict(frame_id=(t,), candidate_rank=(c,), candidate_cell=(c,), image_hw=(t, 2), gt=(t, 4),
        previous=(c, t, 4), boxes=(c, t, 4), model_boxes=(c, t, 4), crop=(c, t, 4), usable=(c, t, 4),
        factor=(c, t), layer=(c, t), raw_score_peak=(c, t), coverage=(c, t), iou=(c, t),
        output_override=(c, t))
    if set(data) != set(shapes) or any(np.shape(data[k]) != shape for k, shape in shapes.items()):
        raise ValueError("Branch cache shape/coverage mismatch")
    if not all(np.isfinite(a).all() for a in data.values()):
        raise ValueError("Nonfinite branch data")
    for k in ("frame_id", "candidate_rank", "candidate_cell", "image_hw", "layer"):
        if not np.issubdtype(data[k].dtype, np.integer):
            raise ValueError("Expected integer " + k)
    if data["output_override"].dtype != np.bool_ or (data["boxes"][..., 2:] <= 0).any():
        raise ValueError("Invalid override mask or boxes")
    onset = event["onset_frame"]
    frames = np.arange(onset, onset + t)
    old = frames - 1
    np.testing.assert_array_equal(data["frame_id"], frames)
    np.testing.assert_array_equal(data["candidate_rank"], np.arange(1, c + 1))
    np.testing.assert_array_equal(data["candidate_cell"], [v["grid_index"] for v in event["candidates"]])
    np.testing.assert_array_equal(data["gt"], np.asarray(entry["ground_truth"])[frames])
    np.testing.assert_array_equal(data["image_hw"], trace["image_hw"][old])
    np.testing.assert_array_equal(data["previous"][:, 0], np.tile(trace["previous"][old[0]], (c, 1)))
    np.testing.assert_array_equal(data["previous"][:, 1:], data["boxes"][:, :-1])
    np.testing.assert_array_equal(data["boxes"][:, 1:], data["model_boxes"][:, 1:])
    np.testing.assert_array_equal(data["boxes"][0, 0], data["model_boxes"][0, 0])
    if c > 1:
        np.testing.assert_array_equal(data["boxes"][1:, 0], [v["box"] for v in event["candidates"][1:]])
    mask = np.zeros((c, t), bool)
    mask[1:, 0] = True
    np.testing.assert_array_equal(data["output_override"], mask)
    if not ((data["layer"] >= 7) & (data["layer"] <= 12)).all():
        raise ValueError("Continuation must use the native candidate-layer selector")
    np.testing.assert_allclose(data["model_boxes"][:, 0], np.tile(trace["boxes"][old[0]], (c, 1)),
                               atol=RESPONSE_DESIGN["replay_box_atol"], rtol=0)
    for k in ("factor", "layer"):
        np.testing.assert_array_equal(data[k][:, 0], np.repeat(trace[k][old[0]], c))
    np.testing.assert_allclose(data["raw_score_peak"][:, 0], trace["raw_score_peak"][old[0]],
                               atol=RESPONSE_DESIGN["replay_score_atol"], rtol=0)
    np.testing.assert_allclose(data["boxes"][0], trace["boxes"][old],
                               atol=RESPONSE_DESIGN["replay_box_atol"], rtol=0,
                               err_msg="Native branch no longer replays the frozen original trajectory")
    np.testing.assert_array_equal(data["factor"][0], trace["factor"][old])
    np.testing.assert_array_equal(data["layer"][0], trace["layer"][old])
    np.testing.assert_allclose(data["raw_score_peak"][0], trace["raw_score_peak"][old],
                               atol=RESPONSE_DESIGN["replay_score_atol"], rtol=0)
    np.testing.assert_allclose(data["iou"], tracking_iou(data["boxes"], data["gt"][None]), atol=1e-12, rtol=0)
    np.testing.assert_allclose(data["iou"][:, 0], [v["onset_gt_iou"] for v in event["candidates"]],
                               atol=1e-12, rtol=0)
    np.testing.assert_allclose(data["iou"][0], trace["iou"][old], atol=1e-12, rtol=0,
                               err_msg="Native branch changed official integer-box IoU")
    for branch in range(c):
        for step in range(t):
            geometry = scene_geometry(data["previous"][branch, step], data["gt"][step], data["image_hw"][step],
                                      source["search_factor"], source["search_size"])
            for k in ("crop", "usable", "factor", "coverage"):
                np.testing.assert_allclose(data[k][branch, step], geometry[k], atol=1e-12, rtol=0)


def load_event(path, event, entry, trace, source, run):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata_json"].item()))
        data = {k: archive[k] for k in archive.files if k != "metadata_json"}
    if meta != event_metadata(run, event):
        raise ValueError("Branch cache identity mismatch")
    audit_event(data, event, entry, trace, source)
    return data
