"""Fixed matched crop-augmentation control following the negative Stage-39 screen."""

import json
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint
from stage27_uav_common import crop_geometry, publish_identical, read_boxes, seal, tracking_iou
from stage28_response_common import dense_boxes
from stage34_identity_common import load_sealed as read_sealed
from stage34_identity_experiment import audit_trace
from stage35_preserve_common import SCHEMA as SOURCE_SCHEMA, fold_samples
from stage35_supervised_audit import verify_bindings
import stage39_perturb_common as prior


SCHEMA = "stage40_matched_crop_augmentation_oof_v1"
ARMS = ("clean", "augmented")
SEEDS = (2110, 2111)
DESIGN = dict(seeds=list(SEEDS), folds=4, updates=256, smoke_updates=8, batch_size=2,
    save_every=64, learning_rate=.0001, weight_decay=.0001, gradient_clip=1.,
    queries_per_sample=2, objective="Mean of two ordinary GT tracking losses; no KD or extra loss",
    training_state="Frozen original SGLA predicted parents in the original fold's fit videos only",
    primary="augmented minus clean full-video AUC; seeds averaged within video",
    minimum_auc_gain=.3, minimum_positive_videos=6, minimum_reliable_frames=10,
    reliable_ci_floor=-.5, reliable_worst_video_floor=-1., minimum_reliable_videos=4,
    bootstrap_samples=5000, bootstrap_seed=2119, source_design=prior.DESIGN,
    stress_role="Secondary paired loss at original fixed anchors; no perturbation-specific search")


def methods():
    return [a + "_seed" + str(s) for s in SEEDS for a in ARMS]


def arm_seed(method):
    arm, seed = method.rsplit("_seed", 1)
    if arm not in ARMS or int(seed) not in SEEDS:
        raise ValueError("Unknown matched crop method")
    return arm, int(seed)


def target_roi(gt, parent, hw, plan):
    crop, _, factor = crop_geometry(parent, hw, plan["search_factor"], plan["search_size"])
    gt = np.asarray(gt, float)
    return np.r_[(gt[:2] - crop[:2]) / crop[2], gt[2:] / crop[2]], factor


def schedule(plan, fold, seed, updates):
    samples, weights = fold_samples(plan, fold)
    rng = np.random.default_rng(seed + 100 * fold)
    draws = rng.choice(len(samples), (updates, DESIGN["batch_size"]), p=weights)
    perturbations = np.arange(draws.size) % 6
    np.random.default_rng(seed + 100 * fold + 10000).shuffle(perturbations)
    return samples, draws, perturbations.reshape(draws.shape)


def code_hashes():
    names = ("stage40_crop_common.py", "stage40_crop_experiment.py", "stage40_crop_summary.py",
        "test_stage40_crop_augmentation.py", "stage39_perturb_common.py", "stage39_perturb_experiment.py",
        "stage35_preserve_common.py", "stage35_preserve_runtime.py", "stage35_preserve_experiment.py",
        "stage18_tracking_loss.py", "stage4_runtime.py")
    return {n: file_hash(Path(__file__).with_name(n)) for n in names}


def source_info(root):
    root = Path(root).resolve()
    old = prior.load_plan(root)
    from stage39_perturb_experiment import smoke_gate, validate_run
    smoke = smoke_gate(root, old)
    run = read_sealed(root / "full/run.json", prior.SCHEMA)
    validate_run(run, old, "full")
    report_path = root / "analysis/stage39_decision.json"
    report = read_sealed(report_path, prior.SCHEMA)
    expected = {prior.task_path(root, "full", e, a).name for e, a in prior.tasks(old, "full")}
    if (report.get("status") != "NO_EXCESS_SENSITIVITY_ESTABLISHED" or
            report.get("plan_fingerprint") != old["fingerprint"] or report.get("run_fingerprint") != run["fingerprint"] or
            report.get("videos") != 8 or report.get("anchors") != 24 or report.get("training_performed") is not False or
            report.get("original_checking_labels_loaded") is not False or set(report["cache_sha256"]) != expected or
            run["frozen_neural_model_sha256"] != smoke["frozen_neural_model_sha256"]):
        raise ValueError("Need the unchanged completed negative Stage-39 screen")
    bindings = dict(report["source_file_sha256"])
    bindings[str(report_path)] = file_hash(report_path)
    bindings.update({str(root / "analysis" / n): d for n, d in report["table_sha256"].items()})
    bindings.update({str(root / "full/branches" / n): d for n, d in report["cache_sha256"].items()})
    source = read_sealed(Path(old["source_input"]) / "manifest.json", SOURCE_SCHEMA)
    if any(source[k] != old[k] for k in ("fit", "folds", "root", "runtime", "search_factor", "search_size")):
        raise ValueError("Original eight-video OOF training source changed")
    verify_bindings(bindings)
    return old, source, report, bindings


def make_plan(root):
    root = Path(root).resolve()
    old, source, report, bindings = source_info(root)
    return seal(dict(schema=SCHEMA, source_stage39=str(root), source_input=old["source_input"],
        source_diagnostic_fingerprint=report["fingerprint"], source_file_sha256=bindings,
        source_runtime=old["source_runtime"], source_frozen_neural_model_sha256=old["source_frozen_neural_model_sha256"],
        runtime_seed=old["runtime_seed"], window=old["window"], tasks=old["tasks"],
        fit=source["fit"], samples=source["samples"], folds=source["folds"], root=source["root"],
        runtime=source["runtime"], search_factor=source["search_factor"], search_size=source["search_size"],
        design=DESIGN, code_sha256=code_hashes(), role="consumed_development_matched_augmentation_control",
        original_checking_labels_loaded=False))


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    plan = make_plan(root)
    for frozen in (root, Path(plan["source_input"])):
        if output == frozen or output in frozen.parents or frozen in output.parents:
            raise ValueError("Use a separate Stage-40 output tree")
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "manifest.json").exists() and any(p.name != "collect.lock" for p in output.iterdir()):
        raise ValueError("Use an empty Stage-40 output directory")
    publish_identical(output / "manifest.json", plan)
    print("Stage-40: 4 folds x 2 seeds x 2 arms = 16 NEW models; 32 full OOF traces + 672 stress branches")
    print("Plan fingerprint: " + plan["fingerprint"])
    return plan


def load_plan(root):
    plan = read_sealed(Path(root) / "manifest.json", SCHEMA)
    if plan.get("design") != DESIGN or plan.get("code_sha256") != code_hashes() or make_plan(plan["source_stage39"]) != plan:
        raise ValueError("Frozen Stage-40 source/code/design changed")
    return plan


def baseline(plan, entry, method):
    if method not in ("sgla", "fast_initial"):
        raise ValueError("Unknown original control")
    path = Path(plan["source_input"]) / "oof" / (entry["sequence"] + "__" + method + ".npz")
    verify_bindings({str(path): plan["source_file_sha256"][str(path)]})
    with np.load(path, allow_pickle=False) as a:
        data = {k: a[k] for k in a.files if k != "metadata_json"}
    audit_trace(data, entry)
    return data


def audit_stress(data, plan, entry, anchor, horizon, reference):
    side = int(plan["runtime"]["lens_x"] ** .5)
    shape = (7, horizon)
    shapes = dict(previous=shape + (4,), boxes=shape + (4,), raw=shape + (side, side),
        size=shape + (2, side, side), offset=shape + (2, side, side), factor=shape,
        layer=shape, decode_error=shape, gt=(horizon, 4), iou=shape,
        image_hw=(horizon, 2), head_calls=(7,), window=(side, side))
    if set(data) != set(shapes) or any(data[k].shape != v for k, v in shapes.items()):
        raise ValueError("Stress shape/coverage changed")
    if (not all(np.isfinite(v).all() for v in data.values()) or (data["boxes"][..., 2:] <= 0).any() or
            (data["raw"] <= 0).any() or (data["raw"] > 1).any() or (data["size"] < 0).any() or
            (data["factor"] <= 0).any() or data["layer"].any()):
        raise ValueError("Invalid compact stress arrays")
    np.testing.assert_array_equal(data["head_calls"], np.full(7, horizon))
    np.testing.assert_array_equal(data["window"], np.asarray(plan["window"], np.float32))
    np.testing.assert_array_equal(data["image_hw"], [[entry["frames"][f]["height"], entry["frames"][f]["width"]]
        for f in range(anchor, anchor + horizon)])
    for p, label in enumerate(prior.PERTURBATIONS):
        parents = np.vstack([prior.perturb(reference["previous"][anchor - 1], label), data["boxes"][p, :-1]])
        np.testing.assert_array_equal(data["previous"][p], parents)
        np.testing.assert_allclose(data["iou"][p], tracking_iou(data["boxes"][p], data["gt"]), atol=1e-12, rtol=0)
        for t in range(horizon):
            factor = crop_geometry(parents[t], data["image_hw"][t], plan["search_factor"], plan["search_size"])[2]
            np.testing.assert_allclose(data["factor"][p, t], factor, atol=1e-12, rtol=0)
            dense = dense_boxes(data["size"][p, t], data["offset"][p, t], parents[t], factor,
                plan["search_size"], data["image_hw"][t])
            cell = int(np.argmax(data["raw"][p, t] * data["window"]))
            error = float(np.max(np.abs(dense[cell] - data["boxes"][p, t])))
            np.testing.assert_allclose(data["decode_error"][p, t], error, atol=1e-12, rtol=0)
            if error > prior.DESIGN["decode_atol_px"]:
                raise ValueError("Stress native/dense decoder mismatch")


def load_result(path, plan, run, entry, method, anchor=None, allow_orphan=False):
    path = Path(path)
    meta = dict(schema=SCHEMA, run_fingerprint=run["fingerprint"], sequence=entry["sequence"], method=method, anchor=anchor)
    with np.load(path, allow_pickle=False) as a:
        if json.loads(str(a["metadata_json"].item())) != meta:
            raise ValueError("Mixed Stage-40 result metadata")
        data = {k: a[k] for k in a.files if k != "metadata_json"}
    receipt = seal(dict(schema=SCHEMA, metadata=meta, trace_sha256=file_hash(path)))
    if path.with_suffix(".json").exists():
        if read_sealed(path.with_suffix(".json"), SCHEMA) != receipt:
            raise ValueError("Completed Stage-40 result changed")
    elif not allow_orphan:
        raise ValueError("Uncommitted result; resume evaluation")
    if anchor is None:
        audit_trace(data, entry)
        expected = slice(1, len(entry["frames"]))
    else:
        horizon = prior.DESIGN["smoke_horizon"] if run["mode"] == "smoke" else prior.DESIGN["horizon"]
        audit_stress(data, plan, entry, anchor, horizon, baseline(plan, entry, "sgla"))
        expected = slice(anchor, anchor + horizon)
    annotation = Path(plan["root"]) / entry["annotation"]
    verify_bindings({str(annotation): entry["annotation_sha256"]})
    np.testing.assert_array_equal(data["gt"], read_boxes(annotation)[expected])
    if not path.with_suffix(".json").exists():
        publish_identical(path.with_suffix(".json"), receipt)
    return data


def result_path(root, entry, method, anchor=None, mode="evaluation"):
    name = entry["sequence"] + "__" + method
    if anchor is not None:
        name += "__a{:06d}".format(anchor)
    return Path(root) / mode / ("traces" if anchor is None else "stress") / (name + ".npz")
