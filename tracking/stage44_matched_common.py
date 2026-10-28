"""Matched retained-structure adaptation control after negative recovery results."""

import json
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint
from stage27_uav_common import publish_identical, read_boxes, seal, tracking_iou
from stage34_identity_common import load_sealed as read_sealed
from stage35_supervised_audit import verify_bindings
import stage40_crop_common as source
import stage43_recovery_common as recovery


SCHEMA = "stage44_matched_structure_adaptation_v1"
SEEDS = source.SEEDS
METHODS = ("six_initial", "retained_initial", "six_adapted", "retained_adapted")
DESIGN = dict(bootstrap_samples=5000, bootstrap_seed=2123, smoke_frames=32,
    minimum_gap=.5, minimum_positive_videos=5, retained_mean_floor=-.5, retained_ci_floor=-1.,
    primary="retained_adapted minus six_adapted full-video AUC",
    treatment="Delete independent candidate blocks and selector versus native SGLA hard selection",
    adaptation="Only common prefix L5-L6 trainable; both structures use identical clean two-view supervised training",
    role="Compression-gap prerequisite test, not a new compression method")


def code_hashes():
    names = ("stage44_matched_common.py", "stage44_matched_runtime.py", "stage44_matched_experiment.py",
        "stage44_matched_summary.py", "test_stage44_matched_structure.py", "stage40_crop_common.py",
        "stage40_crop_experiment.py", "stage35_preserve_runtime.py", "stage18_tracking_loss.py", "stage4_runtime.py")
    return {n:file_hash(Path(__file__).with_name(n)) for n in names}


def source_info(root):
    root = Path(root).resolve()
    old43 = recovery.load_plan(root)
    report_path, evidence_path = root / "analysis/stage43_decision.json", root / "analysis/stage43_evidence.json"
    report, evidence = read_sealed(report_path, recovery.SCHEMA), read_sealed(evidence_path, recovery.SCHEMA)
    from stage43_recovery_experiment import smoke_gate, validate_run
    smoke_gate(root, old43)
    run = read_sealed(root / "full/run.json", recovery.SCHEMA)
    validate_run(run, old43, "full")
    if (report.get("status") != "NO_GT_FREE_RECOVERY_GAIN_ESTABLISHED" or report.get("pilot_pass") is not False or
            report.get("plan_fingerprint") != old43["fingerprint"] or report.get("run_fingerprint") != run["fingerprint"] or
            report.get("full_trajectories") != 128 or report.get("training_performed") is not False or
            report.get("original_checking_labels_loaded") is not False or report.get("videos") != 8 or
            report.get("seeds") != list(SEEDS) or file_hash(evidence_path) != report.get("evidence_sha256") or
            evidence.get("table_sha256") != report.get("table_sha256")):
        raise ValueError("Need unchanged complete negative Stage-43 result")
    bindings = dict(old43["source_file_sha256"])
    verify_bindings(bindings)
    bindings.update(evidence["source_file_sha256"])
    for task in recovery.tasks(old43,"full"):
        path = recovery.result_path(root,"full",*task)
        if any(str(p) not in bindings for p in (path,path.with_suffix(".json"))):
            raise ValueError("Incomplete Stage-43 evidence")
    bindings.update({str(root / "analysis" / n):d for n,d in report["table_sha256"].items()})
    for path in (root / "manifest.json",root / "full/run.json",report_path,evidence_path):
        bindings[str(path)] = file_hash(path)
    old40 = source.load_plan(old43["source_stage40"])
    if any(old40[k] != old43[k] for k in ("fit","folds","root","runtime","search_factor","search_size")):
        raise ValueError("Matched Stage-40 split/runtime changed")
    old40 = dict(old40, source_stage40=old43["source_stage40"])
    verify_bindings(bindings)
    return old40, report, bindings


def make_plan(root):
    old, report, bindings = source_info(root)
    return seal(dict(schema=SCHEMA, source_stage43=str(Path(root).resolve()), source_stage40=old["source_stage40"],
        source_decision_fingerprint=report["fingerprint"], source_model_plan_fingerprint=old["fingerprint"],
        source_teacher_sha256=old["source_frozen_neural_model_sha256"]["sgla"],
        source_file_sha256=bindings, **{k:old[k] for k in ("fit","folds","root","runtime","source_runtime",
            "search_factor","search_size","samples","window","runtime_seed")},
        design=DESIGN, training_design=source.DESIGN, code_sha256=code_hashes(), methods=list(METHODS),
        original_checking_labels_loaded=False, novel_method_established=False))


def prepare(root, output):
    plan = make_plan(root)
    output = Path(output).resolve()
    for frozen in (Path(root).resolve(),Path(plan["source_stage40"])):
        if output == frozen or output in frozen.parents or frozen in output.parents:
            raise ValueError("Use a separate Stage-44 output tree")
    output.mkdir(parents=True,exist_ok=True)
    if not (output / "manifest.json").exists() and any(p.name != "collect.lock" for p in output.iterdir()):
        raise ValueError("Use an empty Stage-44 output directory")
    publish_identical(output / "manifest.json",plan)
    print("Stage-44: 8 NEW retained-SGLA models; reuse 8 frozen clean six-block models")
    print("4 folds x 2 seeds; {} updates/model; only common L5-L6 trainable".format(source.DESIGN["updates"]))
    print("16 new full-video trajectories; three original factorial corners reused with source hashes")
    print("Plan fingerprint: " + plan["fingerprint"])
    return plan


def load_plan(root):
    plan = read_sealed(Path(root) / "manifest.json",SCHEMA)
    if (plan.get("design") != DESIGN or plan.get("training_design") != source.DESIGN or
            plan.get("code_sha256") != code_hashes() or make_plan(plan["source_stage43"]) != plan):
        raise ValueError("Frozen Stage-44 source/code/design changed")
    return plan


def original_plan(plan):
    original = source.load_plan(plan["source_stage40"])
    if original["fingerprint"] != plan["source_model_plan_fingerprint"]:
        raise ValueError("Original matched-training plan changed")
    return original


def reused_trace(plan, entry, seed, method, original=None):
    old = original if original is not None else original_plan(plan)
    if method == "six_initial":
        return source.baseline(old,entry,"fast_initial")
    if method == "retained_initial":
        return source.baseline(old,entry,"sgla")
    if method != "six_adapted":
        raise ValueError("Not a reused factorial corner")
    run = read_sealed(Path(plan["source_stage40"]) / "evaluation/run.json",source.SCHEMA)
    name = "clean_seed" + str(seed)
    return source.load_result(source.result_path(plan["source_stage40"],entry,name),old,run,entry,name)


def checkpoint_meta(plan, rt, fold, seed, smoke, initial):
    updates = source.DESIGN["smoke_updates"] if smoke else source.DESIGN["updates"]
    _,draws,_ = source.schedule(plan,fold,seed,updates)
    return json.loads(json.dumps(dict(schema=SCHEMA,plan_fingerprint=plan["fingerprint"],runtime=rt.identity,
        mode="smoke" if smoke else "full",fold=fold,seed=seed,updates=updates,initial_sha256=initial,
        draws_fingerprint=fingerprint(draws.tolist()),fit_videos=plan["folds"][fold]["fit"],
        held_out_videos=plan["folds"][fold]["held_out"])))


def trace_path(root, entry, seed, smoke=False):
    return Path(root) / ("smoke" if smoke else "evaluation") / "traces" / (entry["sequence"] + "__seed{}.npz".format(seed))


def audit_trace(data, plan, entry, smoke=False):
    n = min(len(entry["frames"]) - 1,DESIGN["smoke_frames"]) if smoke else len(entry["frames"]) - 1
    shapes = dict(previous=(n,4),boxes=(n,4),layer=(n,),head_calls=(n,),block_calls=(n,),
        decode_error=(n,),gt=(n,4),iou=(n,))
    if set(data) != set(shapes) or any(data[k].shape != v for k,v in shapes.items()) or not all(np.isfinite(v).all() for v in data.values()):
        raise ValueError("Incomplete/invalid retained trajectory")
    if (data["boxes"][:,2:] <= 0).any() or not np.isin(data["layer"],range(7,13)).all():
        raise ValueError("Retained native selection/boxes invalid")
    np.testing.assert_array_equal(data["previous"],np.vstack([entry["initial_box"],data["boxes"][:-1]]))
    np.testing.assert_array_equal(data["head_calls"],np.ones(n))
    np.testing.assert_array_equal(data["block_calls"],np.full(n,7))
    if data["decode_error"].max() > .001:
        raise ValueError("Retained native/dense decoding differs")
    np.testing.assert_allclose(data["iou"],tracking_iou(data["boxes"],data["gt"]),atol=1e-12,rtol=0)
    annotation = Path(plan["root"]) / entry["annotation"]
    verify_bindings({str(annotation):entry["annotation_sha256"]})
    np.testing.assert_array_equal(data["gt"],read_boxes(annotation)[1:n+1])


def trace_meta(run,entry,seed):
    return dict(schema=SCHEMA,run_fingerprint=run["fingerprint"],sequence=entry["sequence"],seed=seed)


def load_trace(path,plan,run,entry,seed,smoke=False,allow_orphan=False):
    path = Path(path)
    with np.load(path,allow_pickle=False) as a:
        if json.loads(str(a["metadata_json"].item())) != trace_meta(run,entry,seed):
            raise ValueError("Mixed Stage-44 trace metadata")
        data = {k:a[k] for k in a.files if k != "metadata_json"}
    receipt = seal(dict(schema=SCHEMA,metadata=trace_meta(run,entry,seed),trace_sha256=file_hash(path)))
    if path.with_suffix(".json").exists():
        if read_sealed(path.with_suffix(".json"),SCHEMA) != receipt:
            raise ValueError("Completed Stage-44 trace changed")
    elif not allow_orphan:
        raise ValueError("Uncommitted Stage-44 trace; resume evaluation")
    audit_trace(data,plan,entry,smoke)
    if not path.with_suffix(".json").exists():
        publish_identical(path.with_suffix(".json"),receipt)
    return data
