"""Frozen student-response size/offset factorial; reuse Stage-37 full controls."""

import json
from pathlib import Path

import numpy as np

from stage4_common import file_hash
from stage27_uav_common import publish_identical, seal
from stage34_identity_common import load_sealed as read_sealed
from stage35_supervised_audit import verify_bindings
import stage37_crossed_common as prior


SCHEMA = "stage38_frozen_size_offset_closed_loop_v1"
ARMS = ("student", "teacher_size", "teacher_offset", "teacher_geometry")
NEW_ARMS = ("teacher_size", "teacher_offset")
REFERENCE = "sgla_reference"
DESIGN = dict(seeds=[2110, 2111], smoke_frames=64, bootstrap_samples=5000, bootstrap_seed=2117,
    co_primary=["teacher_size_minus_student", "teacher_offset_minus_student"],
    family_size=2, reliable_iou=.5, minimum_conditional_frames=10,
    decode_atol_px=.001, endpoint_box_atol_px=.001, endpoint_iou_atol=1e-12,
    response="Student windowed response at every frame, on this arm's own crop",
    inference="Frozen original models; first-frame GT only; no new training or threshold search")


def methods(mode):
    arms = ARMS if mode == "smoke" else NEW_ARMS
    return [a + "_seed" + str(s) for s in DESIGN["seeds"] for a in arms]


def arm_seed(method):
    arm, seed = method.rsplit("_seed", 1)
    if arm not in ARMS or int(seed) not in DESIGN["seeds"]:
        raise ValueError("Unknown frozen geometry method")
    return arm, int(seed)


def channels(arm):
    # Both dimensions are independent source choices: (size, offset), 0=student, 1=SGLA.
    return dict(student=(0, 0), teacher_size=(1, 0), teacher_offset=(0, 1), teacher_geometry=(1, 1))[arm]


def compose(outputs, arm):
    size, offset = channels(arm)
    return dict(score_map=outputs[0]["score_map"], size_map=outputs[size]["size_map"],
        offset_map=outputs[offset]["offset_map"])


def code_hashes():
    names = ("stage38_geometry_common.py", "stage38_geometry_experiment.py", "stage38_geometry_summary.py",
        "test_stage38_geometry.py", "stage37_crossed_common.py", "stage37_crossed_experiment.py",
        "stage37_crossed_summary.py", "stage36_same_crop_common.py")
    return {n: file_hash(Path(__file__).with_name(n)) for n in names}


def source_info(root):
    root = Path(root).resolve()
    old = prior.load_plan(root)
    run = read_sealed(root / "full/run.json", prior.SCHEMA)
    report_path = root / "analysis/stage37_diagnostic.json"
    report = read_sealed(report_path, prior.SCHEMA)
    from stage37_crossed_experiment import smoke_gate
    smoke_gate(root, old)
    smoke = read_sealed(root / "smoke/run.json", prior.SCHEMA)
    expected = {e["sequence"] + "__" + m + ".npz" for e in old["fit"] for m in prior.methods()}
    if (run.get("mode") != "full" or run.get("methods") != prior.methods() or
            run.get("plan_fingerprint") != old["fingerprint"] or
            report.get("status") != "COMPLETE_FROZEN_CROSSED_DIAGNOSTIC" or
            report.get("plan_fingerprint") != old["fingerprint"] or report.get("run_fingerprint") != run["fingerprint"] or
            report.get("videos") != 8 or len(old["fit"]) != 8 or report.get("seeds") != DESIGN["seeds"] or
            old["design"]["seeds"] != DESIGN["seeds"] or report.get("unique_full_trajectories") != 56 or
            report.get("training_performed") is not False or report.get("original_checking_labels_loaded") is not False or
            set(report["cache_sha256"]) != expected or {p.name for p in (root / "full/traces").glob("*.npz")} != expected or
            any(run.get(k) != smoke.get(k) for k in ("runtime", "window", "frozen_neural_model_sha256"))):
        raise ValueError("Need the unchanged complete Stage-37 eight-video / two-seed result")
    bindings = dict(old["source_file_sha256"])
    bindings.update(report["source_file_sha256"])
    bindings[str(report_path)] = file_hash(report_path)
    bindings.update({str(root / "full/traces" / n): d for n, d in report["cache_sha256"].items()})
    bindings.update({str(root / "analysis" / n): d for n, d in report["table_sha256"].items()})
    verify_bindings(bindings)
    return old, run, report, bindings


def make_plan(root):
    root = Path(root).resolve()
    old, run, report, bindings = source_info(root)
    return seal(dict(schema=SCHEMA, role="consumed_development_fixed_student_response_geometry_factorial",
        source_stage37=str(root), source_input=old["source_input"], source_stage36=old["source_stage36"],
        source_plan_fingerprint=old["fingerprint"], source_model_plan_fingerprint=old["source_plan_fingerprint"],
        source_diagnostic_fingerprint=report["fingerprint"], source_file_sha256=bindings,
        source_runtime=run["runtime"], source_frozen_neural_model_sha256=run["frozen_neural_model_sha256"],
        window=run["window"], runtime_seed=old["design"]["runtime_seed"],
        design=DESIGN, code_sha256=code_hashes(), fit=old["fit"], folds=old["folds"], root=old["root"],
        runtime=old["runtime"], search_factor=old["search_factor"], search_size=old["search_size"],
        full_methods=methods("full"), smoke_methods=methods("smoke"),
        training_performed=False, original_checking_labels_loaded=False, adoption_gate=None))


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    plan = make_plan(root)
    for frozen in (root, Path(plan["source_input"]), Path(plan["source_stage36"])):
        if output == frozen or output in frozen.parents or frozen in output.parents:
            raise ValueError("Use a separate Stage-38 output tree outside the frozen source trees")
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "manifest.json").exists() and any(p.name != "collect.lock" for p in output.iterdir()):
        raise ValueError("Use an empty Stage-38 output directory")
    publish_identical(output / "manifest.json", plan)
    print("Stage-38: 32 NEW full trajectories; reuse 32 Stage-37 endpoint controls and 8 SGLA references")
    print("Plan fingerprint: " + plan["fingerprint"])
    return plan


def load_plan(root):
    plan = read_sealed(Path(root) / "manifest.json", SCHEMA)
    if plan.get("design") != DESIGN or plan.get("code_sha256") != code_hashes():
        raise ValueError("Frozen Stage-38 code/design changed")
    if make_plan(plan["source_stage37"]) != plan:
        raise ValueError("Frozen Stage-38 source changed")
    return plan


def source_trace(plan, entry, arm, seed):
    root = Path(plan["source_stage37"])
    old = read_sealed(root / "manifest.json", prior.SCHEMA)
    run = read_sealed(root / "full/run.json", prior.SCHEMA)
    if arm not in ("student", "teacher_geometry", REFERENCE):
        raise ValueError("Only frozen endpoint controls / SGLA can be reused")
    method = "sgla" if arm == REFERENCE else arm + "_seed" + str(seed)
    path = root / "full/traces" / (entry["sequence"] + "__" + method + ".npz")
    verify_bindings({str(path): plan["source_file_sha256"][str(path)]})
    data = prior.load_trace(path, old, run, entry, method)
    return data, path


def audit_trace(data, plan, entry, method, limit=None):
    arm, seed = arm_seed(method)
    n = min(len(entry["frames"]) - 1, limit) if limit else len(entry["frames"]) - 1
    side = int(plan["runtime"]["lens_x"] ** .5)
    shape = (n, 2, 2, side, side)
    if data["size"].shape != shape or data["offset"].shape != shape or not all(np.isfinite(v).all() for v in data.values()):
        raise ValueError("Invalid geometry source maps")
    if (data["size"] < 0).any():
        raise ValueError("Negative source size")
    size, offset = channels(arm)
    # The legacy auditor checks a canonical student-response view. Only temporary map
    # selections change; the on-disk two-source maps and the executed trajectory do not.
    view = dict(data, size=data["size"][:, [size, size]], offset=data["offset"][:, [offset, offset]])
    prior.audit_trace(view, plan, entry, "student_seed" + str(seed), limit)
    if data["decode_error"].max() > DESIGN["decode_atol_px"]:
        raise ValueError("Size/offset crossed decoder mismatch")


def endpoint_parity(data, plan, entry, method):
    arm, seed = arm_seed(method)
    if arm not in ("student", "teacher_geometry"):
        return None
    reference, path = source_trace(plan, entry, arm, seed)
    n = len(data["iou"])
    box_error = float(np.max(np.abs(data["boxes"] - reference["boxes"][:n])))
    parent_error = float(np.max(np.abs(data["previous"] - reference["previous"][:n])))
    np.testing.assert_array_equal(data["gt"], reference["gt"][:n])
    np.testing.assert_allclose(data["iou"], reference["iou"][:n], atol=DESIGN["endpoint_iou_atol"], rtol=0)
    if max(box_error, parent_error) > DESIGN["endpoint_box_atol_px"]:
        raise ValueError("Smoke endpoint differs from frozen Stage-37 trajectory")
    return dict(source_path=str(path), max_box_error=box_error, max_parent_error=parent_error)


def metadata(plan, run, entry, method):
    return dict(schema=SCHEMA, run_fingerprint=run["fingerprint"], sequence=entry["sequence"],
        fold=prior.fold_for(plan, entry), method=method)


def load_trace(path, plan, run, entry, method, limit=None, allow_orphan=False):
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        if json.loads(str(archive["metadata_json"].item())) != metadata(plan, run, entry, method):
            raise ValueError("Mixed Stage-38 trace metadata")
        data = {k: archive[k] for k in archive.files if k != "metadata_json"}
    receipt_path = path.with_suffix(".json")
    receipt = seal(dict(schema=SCHEMA, metadata=metadata(plan, run, entry, method),
        trace_sha256=file_hash(path), frames=len(data["iou"])))
    if receipt_path.exists():
        if read_sealed(receipt_path, SCHEMA) != receipt:
            raise ValueError("Completed Stage-38 trace changed")
    elif not allow_orphan:
        raise ValueError("Uncommitted Stage-38 trace; resume the collector")
    audit_trace(data, plan, entry, method, limit)
    np.testing.assert_array_equal(data["window"], np.asarray(run["window"], np.float32))
    annotation = Path(plan["root"]) / entry["annotation"]
    verify_bindings({str(annotation): entry["annotation_sha256"]})
    np.testing.assert_array_equal(data["gt"], prior.read_boxes(annotation)[1:len(data["iou"]) + 1])
    endpoint_parity(data, plan, entry, method)
    if not receipt_path.exists():
        publish_identical(receipt_path, receipt)
    return data
