"""Frozen-state search interventions and integrity checks for Stage-47."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint, write_cache
from stage27_uav_common import crop_geometry, publish_identical, tracking_iou
from stage28_response_common import dense_boxes
import stage46_motion_common as motion
import stage46_shift_audit as prior


SCHEMA = "stage47_matched_history_search_response_v1"
HISTORIES = motion.METHODS
TREATMENTS = ("no_shift", "raw_velocity", "camera_only", "relative_only", "camera_relative")
EXECUTED = dict(sgla="no_shift", velocity="raw_velocity", camera_only="camera_only", camera_velocity="camera_relative")
CONFIG = dict(events=[
    dict(case_id="E01", sequence="uav0000176_00000_s", frame=158, owner="camera_velocity"),
    dict(case_id="E02", sequence="uav0000204_00000_s", frame=664, owner="sgla"),
    dict(case_id="E03", sequence="uav0000304_00253_s", frame=555, owner="camera_only"),
    dict(case_id="E04", sequence="uav0000304_00253_s", frame=1087, owner="camera_velocity")],
    offsets=[-2, -1, 0, 1, 2], dense_decode_atol=.001,
    semantics="Independent current-frame interventions from each frozen own-history state; no continuation")
FILES = ("stage47_search_common.py", "stage47_search_collect.py", "stage47_search_summary.py", "test_stage47_search.py")


def seal(body):
    return dict(body, fingerprint=fingerprint(body))


def read_sealed(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: v for k, v in value.items() if k != "fingerprint"}
    if value.get("schema") != SCHEMA or value.get("fingerprint") != fingerprint(body):
        raise ValueError("Stage-47 schema/fingerprint mismatch: " + str(path))
    return value


def code_hashes():
    names = FILES + ("stage46_shift_audit.py", "test_stage46_shift_audit.py", "stage28_response_common.py", "stage27_uav_common.py")
    return {name: file_hash(Path(__file__).with_name(name)) for name in names}


def load_source(root):
    root = Path(root).expanduser().resolve()
    old_plan, source, traces, old_run, decision, cached, bindings = prior.load_completed(root)
    directory = root / "analysis/search_shift_audit_v1"
    report = prior.read_report(directory / "stage46b_diagnostic.json")
    manifest = prior.read_report(directory / "manifest.json")
    if (report.get("status") != "COMPLETE_CACHED_SEARCH_SHIFT_AUDIT" or
        report.get("configuration") != prior.CONFIG or
        report.get("source_plan_fingerprint") != old_plan["fingerprint"] or
        report.get("source_run_fingerprint") != old_run["fingerprint"] or
        report.get("source_decision_fingerprint") != decision["fingerprint"] or
        report.get("source_file_sha256") != bindings or
        any(manifest.get(k) != report.get(k) for k in manifest if k != "fingerprint") or
        set(report.get("table_sha256", {})) != {"stage46b_" + n + ".csv" for n in prior.TABLES}):
        raise ValueError("Need the matching completed Stage-46B audit")
    repository = Path(__file__).resolve().parents[1]
    prior.verify_files({str(repository / name): digest for name, digest in report["code_sha256"].items()})
    bindings.update({str(directory / name): digest for name, digest in report["table_sha256"].items()})
    bindings.update({str(directory / name): file_hash(directory / name) for name in ("manifest.json", "stage46b_diagnostic.json")})
    prior.verify_files(bindings)
    with (directory / "stage46b_onsets.csv").open(encoding="utf-8", newline="") as handle:
        onsets = list(csv.DictReader(handle))
    events = []
    for event in CONFIG["events"]:
        if not any(r["sequence"] == event["sequence"] and r["method"] == event["owner"] and
            int(r["onset_frame"]) == event["frame"] and r["first_event"] == "True" for r in onsets):
            raise ValueError("Fixed event is not its recorded first failure: " + event["case_id"])
        frames = [event["frame"] + offset for offset in CONFIG["offsets"]]
        if min(frames) < 1 or max(frames) >= len(traces[event["sequence"]][0]["frames"]):
            raise ValueError("Fixed context is outside the recorded video")
        events.append(dict(event, frames=frames))
    plan = seal(dict(schema=SCHEMA, role="posthoc_frozen_history_search_intervention_plan",
        source_input=str(root), source_stage46_plan_fingerprint=old_plan["fingerprint"],
        source_stage46_run_fingerprint=old_run["fingerprint"], source_stage46b_fingerprint=report["fingerprint"],
        source_file_sha256=bindings, code_sha256=code_hashes(), configuration=CONFIG,
        histories=list(HISTORIES), treatments=list(TREATMENTS), events=events,
        expected_head_calls=sum(len(e["frames"]) for e in events) * len(HISTORIES) * len(TREATMENTS),
        training_performed=False, new_closed_loop_policy_evaluated=False))
    return plan, dict(source=source, traces=traces, old_run=old_run, cached=cached)


def load_plan(root):
    plan = read_sealed(Path(root) / "manifest.json")
    expected, inputs = load_source(plan["source_input"])
    if plan != expected:
        raise ValueError("Frozen Stage-47 source/code/configuration changed")
    return plan, inputs


def proposals(data, index, hw):
    """Only saved predictions and past motion state enter current-frame proposals."""
    state = motion.MotionState(data["previous"][index])
    state.raw = data["raw_velocity"][index].copy()
    state.relative = data["relative_velocity"][index].copy()
    matrix, valid = data["matrix"][index], bool(data["motion_valid"][index])
    methods = dict(no_shift="sgla", raw_velocity="velocity", camera_only="camera_only", camera_relative="camera_velocity")
    result = {name: state.propose(method, matrix, valid, hw)["parent"] for name, method in methods.items()}
    relative = matrix[:, :2] @ state.relative if valid else np.zeros(2)
    bounded, _ = motion.bounded(relative, motion.DESIGN["max_shift_side"] * np.sqrt(state.previous[2:].prod()))
    old_center = motion.center(state.previous)
    new_center = np.clip(old_center + bounded, [0., 0.], np.asarray(hw[::-1]) - 1.)
    result["relative_only"] = state.previous.copy()
    result["relative_only"][:2] += new_center - old_center
    return np.asarray([result[name] for name in TREATMENTS])


def path_for(root, event):
    return Path(root) / "responses" / (event["case_id"] + "__stage47_response.npz")


def metadata(run, event):
    return dict(schema=SCHEMA, run_fingerprint=run["fingerprint"], case_id=event["case_id"])


def save(path, data, meta):
    write_cache(path, data, meta)
    publish_identical(path.with_suffix(".receipt.json"), seal(dict(schema=SCHEMA, metadata=meta, sha256=file_hash(path))))


def read_cache(path, meta):
    receipt = path.with_suffix(".receipt.json")
    if not receipt.exists():
        return None
    record = read_sealed(receipt)
    if record.get("metadata") != meta or not path.is_file() or record.get("sha256") != file_hash(path):
        raise ValueError("Committed Stage-47 response changed: " + str(path))
    with np.load(path, allow_pickle=False) as archive:
        if json.loads(str(archive["metadata_json"].item())) != meta:
            raise ValueError("Stage-47 response metadata mismatch")
        return {k: archive[k] for k in archive.files if k != "metadata_json"}


def audit_response(data, event, inputs, window):
    frames, side = event["frames"], len(window)
    if (np.shape(window) != (side, side) or side * side != inputs["source"]["runtime"]["lens_x"] or
        not np.isfinite(window).all() or (np.asarray(window) < 0).any() or not (np.asarray(window) > 0).any()):
        raise ValueError("Invalid native Hann window")
    k = (len(frames), len(HISTORIES), len(TREATMENTS))
    shapes = dict(frame_id=(k[0],), image_hw=(k[0], 2), parents=k + (4,), crop=k + (4,), factor=k,
        boxes=k + (4,), raw=k + (side, side), size=k + (2, side, side), offset=k + (2, side, side),
        layer=k, peak=k, block_calls=k + (12,), head_calls=k, selector_calls=k, crop_sha256=k, mask_sha256=k)
    if set(data) != set(shapes) or any(data[n].shape != s for n, s in shapes.items()):
        raise ValueError("Stage-47 response shape/coverage mismatch")
    for key, values in data.items():
        if key.endswith("sha256"):
            if values.dtype.kind != "U" or not all(len(s) == 64 and set(s) <= set("0123456789abcdef") for s in values.flat):
                raise ValueError("Invalid image/mask digest")
        elif not np.isfinite(values).all():
            raise ValueError("Non-finite response field " + key)
    for key in ("frame_id", "image_hw", "layer", "block_calls", "head_calls", "selector_calls"):
        if not np.issubdtype(data[key].dtype, np.integer):
            raise ValueError("Invalid integer response field " + key)
    if (data["raw"] <= 0).any() or (data["size"] <= 0).any() or not np.isin(data["layer"], range(7, 13)).all():
        raise ValueError("Invalid CENTER map or selected layer")
    np.testing.assert_array_equal(data["frame_id"], frames)
    np.testing.assert_array_equal(data["head_calls"], np.ones(k, int))
    np.testing.assert_array_equal(data["selector_calls"], np.ones(k, int))
    expected = np.zeros(k + (12,), int)
    expected[..., :6] = 1
    for indices in np.ndindex(k):
        expected[indices + (int(data["layer"][indices]) - 1,)] = 1
    np.testing.assert_array_equal(data["block_calls"], expected)
    source = inputs["source"]
    entry = inputs["traces"][event["sequence"]][0]
    maximum_box = maximum_peak = maximum_decode = 0.
    for f, frame in enumerate(frames):
        hw = [entry["frames"][frame]["height"], entry["frames"][frame]["width"]]
        np.testing.assert_array_equal(data["image_hw"][f], hw)
        for h, history in enumerate(HISTORIES):
            saved = inputs["cached"][event["sequence"], history]
            parents = proposals(saved, frame - 1, hw)
            np.testing.assert_allclose(data["parents"][f, h], parents, atol=1e-9, rtol=0)
            executed = TREATMENTS.index(EXECUTED[history])
            np.testing.assert_array_equal(parents[executed], saved["parents"][frame - 1])
            for t in range(len(TREATMENTS)):
                ix = (f, h, t)
                parent = data["parents"][ix]
                crop, _, factor = crop_geometry(parent, hw, source["search_factor"], source["search_size"])
                np.testing.assert_array_equal(data["crop"][ix], crop)
                if data["factor"][ix] != factor:
                    raise ValueError("Resize factor changed")
                raw, size, offset = (data[n][ix] for n in ("raw", "size", "offset"))
                if abs(float(raw.max()) - data["peak"][ix]) > 1e-7:
                    raise ValueError("Captured response does not match native peak")
                boxes = dense_boxes(size, offset, parent, factor, source["search_size"], hw)
                error = float(np.max(np.abs(boxes[int(np.argmax(raw * window))] - data["boxes"][ix])))
                maximum_decode = max(maximum_decode, error)
                if error > CONFIG["dense_decode_atol"]:
                    raise ValueError("Dense decode differs from original tracker output")
            ix = (f, h, executed)
            box_error = float(np.max(np.abs(data["boxes"][ix] - saved["boxes"][frame - 1])))
            peak_error = float(abs(data["peak"][ix] - saved["peak"][frame - 1]))
            if box_error > motion.DESIGN["replay_box_atol"] or peak_error > motion.DESIGN["replay_peak_atol"] or data["layer"][ix] != saved["layer"][frame - 1]:
                raise ValueError("Recorded treatment failed Stage-46 output replay")
            np.testing.assert_array_equal(np.trunc(data["boxes"][ix]), np.trunc(saved["boxes"][frame - 1]))
            maximum_box, maximum_peak = max(maximum_box, box_error), max(maximum_peak, peak_error)
    return dict(maximum_recorded_box_error=maximum_box, maximum_recorded_peak_error=maximum_peak,
                maximum_dense_decode_error=maximum_decode, head_calls=int(data["head_calls"].sum()),
                block_calls=int(data["block_calls"].sum()))
