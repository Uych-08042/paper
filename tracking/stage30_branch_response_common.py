"""Frozen response probes on the corrected Stage-29 branch, never its native parent."""

import json
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint
from stage27_uav_common import scene_geometry, seal, tracking_iou
from stage28_response_common import (DESIGN as RESPONSE_DESIGN, bind_native_top1, dense_boxes, response_metrics)
from stage29_candidate_common import (event_filename, load_event, load_inputs as load_stage29_sources,
    load_sealed as load_stage29, validate_stage29_plan)
from stage29_candidate_summary import make_decision, summarize_event


SCHEMA = "stage30_corrected_branch_response_v1"
DESIGN = dict(failure_iou=0.2, preceding_frames=1, stable_control_offsets=[1, 3],
              expected_failure_events=2, expected_stable_events=1, seed=2092)


def load_sealed(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: v for k, v in value.items() if k != "fingerprint"}
    if value.get("schema") != SCHEMA or value.get("fingerprint") != fingerprint(body):
        raise ValueError("Stage-30 schema/fingerprint mismatch: " + str(path))
    return value


def load_inputs(directory):
    directory = Path(directory).expanduser().resolve()
    plan, run = load_stage29(directory / "plan.json"), load_stage29(directory / "run.json")
    report = load_stage29(directory / "analysis" / "stage29_decision.json")
    upstream = load_stage29_sources(plan["source_input"])
    validate_stage29_plan(plan, upstream)
    if run.get("role") != "one_output_intervention_native_continuation" or run.get("plan") != plan:
        raise ValueError("Stage-29 run/plan mismatch")
    _, source, traces, _ = upstream["source_info"]
    expected = {event_filename(e) for e in plan["events"]}
    if {p.name for p in (directory / "branches").glob("*.npz")} != expected:
        raise ValueError("Need complete, unmixed Stage-29 event caches")
    branches, hashes, rows = {}, {}, []
    for event in plan["events"]:
        path = directory / "branches" / event_filename(event)
        entry, trace = traces[event["sequence"]]
        data = load_event(path, event, entry, trace, source, run)
        branches[path.name] = data
        hashes[path.name] = file_hash(path)
        rows.append(summarize_event(event, data, trace)[1])
    # Reproduce the existing decision without editing it or changing its gate.
    if report != make_decision(plan, rows, [], run, hashes):
        raise ValueError("Stage-29 completion decision differs from audited branches/code")
    return dict(directory=directory, plan=plan, run=run, report=report, upstream=upstream,
                source=source, traces=traces, branches=branches, hashes=hashes)


def make_plan(inputs):
    points, failures, stable = [], 0, 0
    for event in inputs["plan"]["events"]:
        if not event["primary_eligible"]:
            continue
        rank = event["onset_gt_best_rank"]
        branch = inputs["branches"][event_filename(event)]
        bad = np.flatnonzero(branch["iou"][rank - 1, 1:] < DESIGN["failure_iou"])
        if len(bad):
            first = int(bad[0]) + 1
            failures += 1
            choices = [(max(0, first - DESIGN["preceding_frames"]), "pre_failure_context"),
                       (first, "first_low_iou")]
        else:
            first = None
            stable += 1
            choices = [(offset, "stable_control") for offset in DESIGN["stable_control_offsets"]]
        for offset, purpose in choices:
            points.append(dict(sequence=event["sequence"], source_group_proxy=event["source_group_proxy"],
                event=event["event"], branch_rank=rank, branch_offset=offset,
                onset_frame=event["onset_frame"], frame_id=event["onset_frame"] + offset,
                purpose=purpose, first_low_iou_offset=first,
                recorded_branch_iou=float(branch["iou"][rank - 1, offset]),
                output_was_overridden=bool(branch["output_override"][rank - 1, offset])))
    if failures != DESIGN["expected_failure_events"] or stable != DESIGN["expected_stable_events"]:
        raise ValueError("Expected two failing corrected events and one stable event; do not rescope after new responses")
    return seal(dict(schema=SCHEMA, role="frozen_corrected_branch_response_plan",
        source_input=str(inputs["directory"]), source_run_fingerprint=inputs["run"]["fingerprint"],
        source_plan_fingerprint=inputs["plan"]["fingerprint"], source_decision_fingerprint=inputs["report"]["fingerprint"],
        source_branch_sha256=inputs["hashes"], design=DESIGN, response_design=RESPONSE_DESIGN,
        points=points, replay_frames=len(points),
        selection="Outcome-conditioned diagnosis: first future IoU<0.2 of each onset-GT-selected branch, preceding frame, stable offsets 1/3",
        semantics="Restore branch.previous at each point; inspect native model output without any new override or rollout",
        code_sha256={name: file_hash(Path(__file__).parent / name)
                     for name in ("stage30_branch_response_common.py", "stage30_branch_response_collect.py")}))


def validate_plan(plan, inputs):
    if plan != make_plan(inputs):
        raise ValueError("Stage-30 plan/source/code changed; preserve the frozen run")


def point_filename(point):
    return "{}__e{:03d}__r{:02d}__p{:03d}__stage30_response.npz".format(
        point["sequence"], point["event"], point["branch_rank"], point["branch_offset"])


def point_source(inputs, point):
    event = next(e for e in inputs["plan"]["events"]
                 if e["sequence"] == point["sequence"] and e["event"] == point["event"])
    entry, trace = inputs["traces"][point["sequence"]]
    return entry, trace, inputs["branches"][event_filename(event)]


def point_metadata(run, point):
    return dict(run_fingerprint=run["fingerprint"], sequence=point["sequence"], event=point["event"],
                branch_rank=point["branch_rank"], branch_offset=point["branch_offset"], frame_id=point["frame_id"])


def audit_point(data, point, branch, source):
    side = int(source["runtime"]["lens_x"] ** .5)
    shapes = dict(raw=(side, side), window=(side, side), size=(2, side, side), offset=(2, side, side),
        previous=(4,), gt=(4,), native_box=(4,), recorded_branch_box=(4,), crop=(4,), usable=(4,),
        factor=(), image_hw=(2,), layer=(), frame_id=(), branch_offset=(), output_was_overridden=(),
        replay_box_error=(), replay_peak_error=(), decode_box_error=())
    if set(data) != set(shapes) or any(np.shape(data[k]) != shape for k, shape in shapes.items()):
        raise ValueError("Response point shape mismatch")
    if not all(np.isfinite(v).all() for v in data.values()):
        raise ValueError("Nonfinite response point")
    for key in ("image_hw", "layer", "frame_id", "branch_offset"):
        if not np.issubdtype(data[key].dtype, np.integer):
            raise ValueError("Expected integer " + key)
    b, t = point["branch_rank"] - 1, point["branch_offset"]
    for key, expected in dict(previous=branch["previous"][b, t], gt=branch["gt"][t],
        recorded_branch_box=branch["boxes"][b, t], image_hw=branch["image_hw"][t],
        factor=branch["factor"][b, t], layer=branch["layer"][b, t], frame_id=point["frame_id"],
        branch_offset=t, output_was_overridden=branch["output_override"][b, t]).items():
        np.testing.assert_array_equal(data[key], expected)
    if data["output_was_overridden"].dtype != np.bool_:
        raise ValueError("Expected boolean override indicator")
    error = float(np.max(np.abs(data["native_box"] - branch["model_boxes"][b, t])))
    peak_error = float(abs(float(data["raw"].max()) - branch["raw_score_peak"][b, t]))
    for key, value in (("replay_box_error", error), ("replay_peak_error", peak_error)):
        np.testing.assert_allclose(data[key], value, atol=1e-12, rtol=0)
    if error > RESPONSE_DESIGN["replay_box_atol"] or peak_error > RESPONSE_DESIGN["replay_score_atol"]:
        raise ValueError("Replayed model output differs from Stage-29 corrected-branch model output")
    np.testing.assert_allclose(tracking_iou(data["native_box"], data["gt"]),
        tracking_iou(branch["model_boxes"][b, t], branch["gt"][t]), atol=1e-12, rtol=0)
    geometry = scene_geometry(data["previous"], data["gt"], data["image_hw"], source["search_factor"], source["search_size"])
    for key in ("factor", "crop", "usable"):
        np.testing.assert_allclose(data[key], geometry[key], atol=1e-12, rtol=0)
    boxes = dense_boxes(data["size"], data["offset"], data["previous"], float(data["factor"]),
                        source["search_size"], data["image_hw"])
    cell = int(np.argmax(data["raw"] * data["window"]))
    error = float(np.max(np.abs(boxes[cell] - data["native_box"])))
    np.testing.assert_allclose(data["decode_box_error"], error, atol=1e-12, rtol=0)
    if error > RESPONSE_DESIGN["decode_box_atol"]:
        raise ValueError("Dense CENTER decode disagrees with native output")
    boxes = bind_native_top1(data["raw"], data["window"], boxes, data["native_box"])
    response_metrics(data["raw"], data["window"], boxes, data["gt"])


def load_point(path, point, inputs, run):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata_json"].item()))
        data = {k: archive[k] for k in archive.files if k != "metadata_json"}
    if meta != point_metadata(run, point):
        raise ValueError("Response point identity mismatch")
    _, _, branch = point_source(inputs, point)
    audit_point(data, point, branch, inputs["source"])
    np.testing.assert_array_equal(data["window"], np.asarray(run["window"], np.float32))
    return data
