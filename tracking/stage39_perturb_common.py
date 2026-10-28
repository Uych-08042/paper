"""Matched predicted-parent perturbations of frozen native and compact trackers."""

import json
from pathlib import Path

import numpy as np

from stage4_common import file_hash
from stage27_uav_common import crop_geometry, publish_identical, read_boxes, seal, tracking_iou
from stage28_response_common import dense_boxes
from stage34_identity_common import load_sealed as read_sealed
from stage35_supervised_audit import verify_bindings
import stage38_geometry_common as prior


SCHEMA = "stage39_matched_parent_perturbation_v1"
MODELS = ("sgla", "fast_initial", "supervised_seed2110", "supervised_seed2111")
PERTURBATIONS = ("clean", "x_minus", "x_plus", "y_minus", "y_plus", "scale_down", "scale_up")
DESIGN = dict(horizon=20, smoke_horizon=4, anchor_quartiles=[1, 2, 3],
    translation_parent_side=.15, scales=[.8, 1.25], bootstrap_samples=5000, bootstrap_seed=2118,
    primary="Fast minus SGLA excess clean-minus-perturbed mean IoU loss at offsets 2..20",
    minimum_excess_iou=.02, minimum_positive_videos=6, minimum_ci_low=0.,
    minimum_leave_one_out=0., decode_atol_px=.001, replay_atol_px=.001,
    replay_iou_atol=1e-12, parent_reference="Frozen native SGLA predicted parent; never GT",
    continuation="One initial parent intervention; autonomous predictions thereafter; no resets")


def perturb(parent, name):
    parent = np.asarray(parent, float)
    if parent.shape != (4,) or not np.isfinite(parent).all() or (parent[2:] <= 0).any():
        raise ValueError("Invalid intervention parent")
    if name not in PERTURBATIONS:
        raise ValueError("Unknown fixed perturbation")
    result = parent.copy()
    side = float(np.sqrt(parent[2] * parent[3]))
    if name.startswith(("x_", "y_")):
        result[0 if name.startswith("x_") else 1] += (1 if name.endswith("plus") else -1) * DESIGN["translation_parent_side"] * side
    elif name.startswith("scale_"):
        factor = DESIGN["scales"][0 if name == "scale_down" else 1]
        result[2:] = parent[2:] * factor
        result[:2] = parent[:2] + (parent[2:] - result[2:]) / 2
    # Keep this exact xywh parent, including out-of-image extents. Native sample_target
    # pads crops; predicted output boxes still use the unchanged native clipping.
    return result


def anchors(entry):
    available = len(entry["frames"]) - DESIGN["horizon"]
    values = [max(1, available * q // 4) for q in DESIGN["anchor_quartiles"]]
    if available < 4 or len(set(values)) != 3:
        raise ValueError("Video too short for three distinct fixed anchors and full horizon")
    return values


def code_hashes():
    names = ("stage39_perturb_common.py", "stage39_perturb_experiment.py", "stage39_perturb_summary.py",
        "test_stage39_perturbation.py", "stage38_geometry_common.py", "stage35_preserve_runtime.py",
        "stage36_same_crop_common.py", "stage4_runtime.py", "stage28_response_common.py")
    return {n: file_hash(Path(__file__).with_name(n)) for n in names}


def make_plan(root):
    root = Path(root).resolve()
    old = prior.load_plan(root)
    path = root / "analysis/stage38_diagnostic.json"
    report = read_sealed(path, prior.SCHEMA)
    expected = {e["sequence"] + "__" + m + ".npz" for e in old["fit"] for m in prior.methods("full")}
    if (report.get("status") != "COMPLETE_FROZEN_SIZE_OFFSET_DIAGNOSTIC" or
            report.get("plan_fingerprint") != old["fingerprint"] or report.get("videos") != 8 or
            report.get("seeds") != [2110, 2111] or report.get("training_performed") is not False or
            report.get("original_checking_labels_loaded") is not False or set(report["cache_sha256"]) != expected):
        raise ValueError("Need unchanged complete Stage-38 development diagnostic")
    bindings = dict(old["source_file_sha256"])
    bindings.update(report["source_file_sha256"])
    bindings[str(path)] = file_hash(path)
    bindings.update({str(root / "analysis" / n): d for n, d in report["table_sha256"].items()})
    bindings.update({str(root / "full/traces" / n): d for n, d in report["cache_sha256"].items()})
    verify_bindings(bindings)
    return seal(dict(schema=SCHEMA, source_stage38=str(root), source_input=old["source_input"],
        source_stage37=old["source_stage37"], source_model_plan_fingerprint=old["source_model_plan_fingerprint"],
        source_diagnostic_fingerprint=report["fingerprint"], source_file_sha256=bindings,
        source_runtime=old["source_runtime"], source_frozen_neural_model_sha256=old["source_frozen_neural_model_sha256"],
        window=old["window"], runtime_seed=old["runtime_seed"], fit=old["fit"], folds=old["folds"], root=old["root"],
        runtime=old["runtime"], search_factor=old["search_factor"], search_size=old["search_size"],
        tasks={e["sequence"]: anchors(e) for e in old["fit"]}, models=list(MODELS), perturbations=list(PERTURBATIONS),
        design=DESIGN, code_sha256=code_hashes(), training_performed=False, original_checking_labels_loaded=False,
        role="consumed_development_frozen_surgical_compression_sensitivity_screen"))


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    plan = make_plan(root)
    for frozen in (root, Path(plan["source_input"]), Path(plan["source_stage37"])):
        if output == frozen or output in frozen.parents or frozen in output.parents:
            raise ValueError("Use a separate Stage-39 output tree")
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "manifest.json").exists() and any(p.name != "collect.lock" for p in output.iterdir()):
        raise ValueError("Use an empty Stage-39 output directory")
    publish_identical(output / "manifest.json", plan)
    print("Stage-39: 8 videos x 3 fixed anchors x 4 models x 7 conditions x 20 frames = 13440 branch frames")
    print("Plan fingerprint: " + plan["fingerprint"])
    return plan


def load_plan(root):
    plan = read_sealed(Path(root) / "manifest.json", SCHEMA)
    if plan.get("design") != DESIGN or plan.get("code_sha256") != code_hashes():
        raise ValueError("Frozen Stage-39 code/design changed")
    if make_plan(plan["source_stage38"]) != plan:
        raise ValueError("Frozen Stage-39 source changed")
    return plan


def source_trace(plan, entry):
    source_plan = read_sealed(Path(plan["source_stage38"]) / "manifest.json", prior.SCHEMA)
    data, path = prior.source_trace(source_plan, entry, prior.REFERENCE, 2110)
    return data


def task_path(root, mode, entry, anchor):
    return Path(root) / mode / "branches" / (entry["sequence"] + "__a{:06d}.npz".format(anchor))


def tasks(plan, mode):
    entries = plan["fit"][:1] if mode == "smoke" else plan["fit"]
    return [(e, a) for e in entries for a in (plan["tasks"][e["sequence"]][:1] if mode == "smoke" else plan["tasks"][e["sequence"]])]


def metadata(plan, run, entry, anchor):
    return dict(schema=SCHEMA, run_fingerprint=run["fingerprint"], sequence=entry["sequence"], anchor=anchor)


def audit(data, plan, entry, anchor, horizon, reference):
    side = int(plan["runtime"]["lens_x"] ** .5)
    shape = (len(MODELS), len(PERTURBATIONS), horizon)
    shapes = dict(boxes=shape + (4,), previous=shape + (4,), raw=shape + (side, side),
        size=shape + (2, side, side), offset=shape + (2, side, side), factor=shape,
        layer=shape, decode_error=shape, image_hw=(horizon, 2), gt=(horizon, 4), iou=shape,
        window=(side, side), head_calls=(len(MODELS),), fast_parity_error=(len(PERTURBATIONS),))
    if set(data) != set(shapes) or any(data[k].shape != s for k, s in shapes.items()):
        raise ValueError("Branch shape/coverage changed")
    if not all(np.isfinite(v).all() for v in data.values()) or (data["boxes"][..., 2:] <= 0).any():
        raise ValueError("Nonfinite or invalid branch")
    if ((data["raw"] <= 0).any() or (data["raw"] > 1).any() or (data["size"] < 0).any() or
            (data["factor"] <= 0).any() or (data["window"] < 0).any() or
            data["fast_parity_error"].max() > 1e-6):
        raise ValueError("Invalid frozen branch maps or Fast-prefix equivalence")
    np.testing.assert_array_equal(data["head_calls"], np.full(len(MODELS), horizon * len(PERTURBATIONS)))
    np.testing.assert_array_equal(data["image_hw"], [[entry["frames"][f]["height"], entry["frames"][f]["width"]]
        for f in range(anchor, anchor + horizon)])
    for m, model in enumerate(MODELS):
        legal = range(7, 13) if model == "sgla" else [0]
        if not np.isin(data["layer"][m], list(legal)).all():
            raise ValueError("Invalid executed layer")
        for p, name in enumerate(PERTURBATIONS):
            expected = np.vstack([perturb(reference["previous"][anchor - 1], name), data["boxes"][m, p, :-1]])
            np.testing.assert_array_equal(data["previous"][m, p], expected,
                err_msg="Continuation must use its own predictions after one intervention")
            np.testing.assert_allclose(data["iou"][m, p], tracking_iou(data["boxes"][m, p], data["gt"]), atol=1e-12, rtol=0)
            for t in range(horizon):
                factor = crop_geometry(expected[t], data["image_hw"][t], plan["search_factor"], plan["search_size"])[2]
                np.testing.assert_allclose(data["factor"][m, p, t], factor, atol=1e-12, rtol=0)
                dense = dense_boxes(data["size"][m, p, t], data["offset"][m, p, t], expected[t],
                    factor, plan["search_size"], data["image_hw"][t])
                cell = int(np.argmax(data["raw"][m, p, t] * data["window"]))
                error = float(np.max(np.abs(dense[cell] - data["boxes"][m, p, t])))
                np.testing.assert_allclose(data["decode_error"][m, p, t], error, atol=1e-12, rtol=0)
                if error > DESIGN["decode_atol_px"]:
                    raise ValueError("Native/dense decoder mismatch")
    sl = slice(anchor - 1, anchor - 1 + horizon)
    np.testing.assert_allclose(data["boxes"][0, 0], reference["boxes"][sl], atol=DESIGN["replay_atol_px"], rtol=0)
    np.testing.assert_allclose(data["iou"][0, 0], reference["iou"][sl], atol=DESIGN["replay_iou_atol"], rtol=0)
    np.testing.assert_array_equal(data["layer"][0, 0], reference["layer"][sl])


def load_branch(path, plan, run, entry, anchor, reference, allow_orphan=False):
    path = Path(path)
    with np.load(path, allow_pickle=False) as a:
        if json.loads(str(a["metadata_json"].item())) != metadata(plan, run, entry, anchor):
            raise ValueError("Mixed Stage-39 branch metadata")
        data = {k: a[k] for k in a.files if k != "metadata_json"}
    receipt_path = path.with_suffix(".json")
    receipt = seal(dict(schema=SCHEMA, metadata=metadata(plan, run, entry, anchor), trace_sha256=file_hash(path)))
    if receipt_path.exists():
        if read_sealed(receipt_path, SCHEMA) != receipt:
            raise ValueError("Completed Stage-39 branch changed")
    elif not allow_orphan:
        raise ValueError("Uncommitted branch; resume collector to audit it")
    horizon = DESIGN["smoke_horizon"] if run["mode"] == "smoke" else DESIGN["horizon"]
    audit(data, plan, entry, anchor, horizon, reference)
    np.testing.assert_array_equal(data["window"], np.asarray(run["window"], np.float32))
    annotation = Path(plan["root"]) / entry["annotation"]
    verify_bindings({str(annotation): entry["annotation_sha256"]})
    np.testing.assert_array_equal(data["gt"], read_boxes(annotation)[anchor:anchor + horizon])
    if not receipt_path.exists():
        publish_identical(receipt_path, receipt)
    return data
