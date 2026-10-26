"""Frozen, consumed-development 2x2 response/geometry closed-loop diagnostic."""

import json
from pathlib import Path

import numpy as np

from stage4_common import file_hash
from stage27_uav_common import crop_geometry, publish_identical, read_boxes, seal, tracking_iou
from stage28_response_common import dense_boxes
from stage34_identity_common import load_sealed as read_sealed
from stage35_preserve_common import DESIGN as TRAINING
from stage35_supervised_audit import verify_bindings
import stage36_onset_jump_audit as onset


SCHEMA = "stage37_frozen_response_geometry_closed_loop_v1"
ARMS = ("student", "teacher_response", "teacher_geometry", "sgla")
DESIGN = dict(seeds=[2110, 2111], runtime_seed=2116, smoke_frames=64,
    bootstrap_samples=5000, bootstrap_seed=2116, reliable_iou=.5,
    minimum_conditional_frames=10, decode_atol_px=.001, replay_atol_px=.001,
    replay_iou_atol=1e-12, dense_gt_match_iou=.5,
    primary="teacher_response_minus_student: video-macro full-video AUC",
    geometry="offset and size are swapped together; original window and decoder",
    gate="Source/decoder/causality checks only; no new-method adoption gate")


def methods():
    return [a + "_seed" + str(s) for s in DESIGN["seeds"] for a in ARMS[:-1]] + ["sgla"]


def arm_seed(method):
    if method == "sgla":
        return "sgla", DESIGN["seeds"][0]  # Unused student is fixed for the common teacher control.
    arm, seed = method.rsplit("_seed", 1)
    if arm not in ARMS[:-1] or int(seed) not in DESIGN["seeds"]:
        raise ValueError("Unknown frozen diagnostic method")
    return arm, int(seed)


def channels(arm):
    # 0=student, 1=SGLA; rows of the 2x2 design are exact source endpoints.
    return dict(student=(0, 0), teacher_response=(1, 0), teacher_geometry=(0, 1), sgla=(1, 1))[arm]


def compose(outputs, arm):
    response, geometry = channels(arm)
    return dict(score_map=outputs[response]["score_map"],
        size_map=outputs[geometry]["size_map"], offset_map=outputs[geometry]["offset_map"])


def code_hashes():
    names = ("stage37_crossed_common.py", "stage37_crossed_experiment.py", "stage37_crossed_summary.py",
        "test_stage37_crossed.py", "stage36_same_crop_common.py", "stage36_same_crop_collect.py",
        "stage36_onset_jump_audit.py", "stage28_response_common.py")
    return {n: file_hash(Path(__file__).with_name(n)) for n in names}


def source_info(root):
    root = Path(root).resolve()
    original_plan, source, _, _, prior, bindings = onset.load_source(root)
    path = root / "analysis/onset_jump_audit_v1/stage36c_diagnostic.json"
    report = read_sealed(path, onset.SCHEMA)
    if (report.get("status") != "COMPLETE_CACHED_ONSET_AUDIT" or report.get("configuration") != onset.CONFIG or
            report.get("source_plan_fingerprint") != original_plan["fingerprint"] or
            report.get("source_geometry_audit_fingerprint") != prior["fingerprint"] or
            report.get("training_performed") is not False or report.get("new_inference_performed") is not False):
        raise ValueError("Need the completed frozen Stage-36C audit")
    for name, digest in report["code_sha256"].items():
        if file_hash(Path(__file__).with_name(name)) != digest:
            raise ValueError("Frozen Stage-36C code changed")
    bindings.update(report["source_file_sha256"])
    bindings.update({str(path.parent / n): d for n, d in report["table_sha256"].items()})
    bindings[str(path)] = file_hash(path)
    if len(source["fit"]) != 8 or len({e["source_group_proxy"] for e in source["fit"]}) != 8 or DESIGN["seeds"] != TRAINING["seeds"]:
        raise ValueError("Need the original eight OOF source groups and both original seeds")
    for entry in source["fit"]:
        for frame in entry["frames"]:
            bindings[str(Path(source["root"]) / frame["file"])] = frame["sha256"]
    verify_bindings(bindings)
    return source, original_plan["source_input"], report["fingerprint"], bindings


def make_plan(root):
    source, source_root, diagnostic, bindings = source_info(root)
    return seal(dict(schema=SCHEMA, role="consumed_fit_only_frozen_model_crossed_closed_loop",
        source_stage36=str(Path(root).resolve()), source_input=str(Path(source_root).resolve()),
        source_plan_fingerprint=source["fingerprint"], source_onset_fingerprint=diagnostic,
        source_file_sha256=bindings, design=DESIGN, code_sha256=code_hashes(),
        fit=source["fit"], folds=source["folds"], root=source["root"], runtime=source["runtime"],
        search_factor=source["search_factor"], search_size=source["search_size"], methods=methods(),
        initialization="Original first-frame box only; each method uses its own predicted parents",
        training_performed=False, original_checking_labels_loaded=False, adoption_gate=None))


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    plan = make_plan(root)
    for frozen in (root, Path(plan["source_input"])):
        if frozen == output or frozen in output.parents or output in frozen.parents:
            raise ValueError("Use a separate Stage-37 output tree outside the frozen source trees")
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "manifest.json").exists() and any(p.name != "collect.lock" for p in output.iterdir()):
        raise ValueError("Use an empty Stage-37 output directory")
    publish_identical(output / "manifest.json", plan)
    print("Stage-37: 8 OOF videos; 2 seeds; 7 unique trajectories/video = 56 full traces")
    print("Plan fingerprint: " + plan["fingerprint"])
    return plan


def load_plan(root):
    plan = read_sealed(Path(root) / "manifest.json", SCHEMA)
    if plan.get("design") != DESIGN or plan.get("code_sha256") != code_hashes():
        raise ValueError("Frozen Stage-37 code/design changed")
    if make_plan(plan["source_stage36"]) != plan:
        raise ValueError("Frozen Stage-37 source changed")
    return plan


def fold_for(plan, entry):
    matches = [f for f in plan["folds"] if entry["sequence"] in f["held_out"]]
    if len(matches) != 1 or entry["sequence"] in matches[0]["fit"]:
        raise ValueError("Wrong OOF fold / training-video leakage")
    return matches[0]["fold"]


def metadata(plan, run, entry, method):
    return dict(schema=SCHEMA, run_fingerprint=run["fingerprint"], sequence=entry["sequence"],
        fold=fold_for(plan, entry), method=method)


def audit_trace(data, plan, entry, method, limit=None):
    arm, _ = arm_seed(method)
    n = min(len(entry["frames"]) - 1, limit) if limit else len(entry["frames"]) - 1
    side = int(plan["runtime"]["lens_x"] ** .5)
    shapes = dict(frame_id=(n,), previous=(n, 4), boxes=(n, 4), gt=(n, 4), iou=(n,),
        elapsed_ms=(n,), model_head_calls=(2,), raw=(n, 2, side, side),
        size=(n, 2, 2, side, side), offset=(n, 2, 2, side, side),
        factor=(n,), image_hw=(n, 2), layer=(n,), window=(side, side), decode_error=(n,))
    if set(data) != set(shapes) or any(data[k].shape != s for k, s in shapes.items()):
        raise ValueError("Crossed closed-loop trace shape/coverage mismatch")
    if not all(np.isfinite(v).all() for v in data.values()) or (data["boxes"][:, 2:] <= 0).any():
        raise ValueError("Nonfinite or invalid crossed trace")
    if ((data["raw"] <= 0).any() or (data["raw"] > 1).any() or (data["size"] < 0).any()
            or (data["factor"] <= 0).any() or not np.isin(data["layer"], np.arange(7, 13)).all()):
        raise ValueError("Invalid source response, geometry or selected layer")
    np.testing.assert_array_equal(data["frame_id"], np.arange(1, n + 1))
    np.testing.assert_array_equal(data["previous"], np.vstack([entry["initial_box"], data["boxes"][:-1]]))
    np.testing.assert_array_equal(data["model_head_calls"], [n, n])
    np.testing.assert_allclose(data["iou"], tracking_iou(data["boxes"], data["gt"]), atol=1e-12, rtol=0)
    r, g = channels(arm)
    for i in range(n):
        hw = [entry["frames"][i + 1]["height"], entry["frames"][i + 1]["width"]]
        np.testing.assert_array_equal(data["image_hw"][i], hw)
        factor = crop_geometry(data["previous"][i], hw, plan["search_factor"], plan["search_size"])[2]
        np.testing.assert_allclose(data["factor"][i], factor, rtol=0, atol=1e-12)
        boxes = dense_boxes(data["size"][i, g], data["offset"][i, g], data["previous"][i],
            factor, plan["search_size"], hw)
        cell = int(np.argmax(data["raw"][i, r] * data["window"]))
        error = float(np.max(np.abs(boxes[cell] - data["boxes"][i])))
        np.testing.assert_allclose(data["decode_error"][i], error, atol=1e-12, rtol=0)
        if error > DESIGN["decode_atol_px"]:
            raise ValueError("Original dense/native crossed decoder mismatch")
    if (data["elapsed_ms"] < 0).any() or (data["window"] < 0).any():
        raise ValueError("Invalid timing/window")


def native_parity(data, plan, entry, method):
    arm, seed = arm_seed(method)
    if arm not in ("student", "sgla"):
        return None
    old_method = "sgla" if arm == "sgla" else "supervised_seed" + str(seed)
    path = Path(plan["source_input"]) / "oof" / (entry["sequence"] + "__" + old_method + ".npz")
    with np.load(path, allow_pickle=False) as a:
        n = len(data["iou"])
        np.testing.assert_array_equal(data["gt"], a["gt"][:n])
        error = float(np.max(np.abs(data["boxes"] - a["boxes"][:n])))
        parent_error = float(np.max(np.abs(data["previous"] - a["previous"][:n])))
        np.testing.assert_allclose(data["iou"], a["iou"][:n], rtol=0, atol=DESIGN["replay_iou_atol"],
            err_msg="Native control changed original integer-box IoU")
    if max(error, parent_error) > DESIGN["replay_atol_px"]:
        raise ValueError("Native control differs from frozen Stage-35 trajectory")
    return dict(max_box_error=error, max_parent_error=parent_error)


def load_trace(path, plan, run, entry, method, limit=None, allow_orphan=False):
    path = Path(path)
    with np.load(path, allow_pickle=False) as a:
        if json.loads(str(a["metadata_json"].item())) != metadata(plan, run, entry, method):
            raise ValueError("Mixed Stage-37 trace metadata")
        data = {k: a[k] for k in a.files if k != "metadata_json"}
    receipt_path = path.with_suffix(".json")
    receipt = seal(dict(schema=SCHEMA, metadata=metadata(plan, run, entry, method),
        trace_sha256=file_hash(path), frames=len(data["iou"])))
    if receipt_path.exists():
        if read_sealed(receipt_path, SCHEMA) != receipt:
            raise ValueError("Completed Stage-37 trace changed")
    elif not allow_orphan:
        raise ValueError("Uncommitted Stage-37 trace; resume collector to audit it")
    audit_trace(data, plan, entry, method, limit)
    np.testing.assert_array_equal(data["window"], np.asarray(run["window"], np.float32))
    annotation = Path(plan["root"]) / entry["annotation"]
    verify_bindings({str(annotation): entry["annotation_sha256"]})
    np.testing.assert_array_equal(data["gt"], read_boxes(annotation)[1:len(data["iou"]) + 1])
    native_parity(data, plan, entry, method)
    if not receipt_path.exists():
        publish_identical(receipt_path, receipt)
    return data
