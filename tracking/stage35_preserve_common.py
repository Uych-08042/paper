"""Frozen fit-only grouped OOF design for corrective distillation."""

import copy
import json
from pathlib import Path

import numpy as np

from stage4_common import file_hash
from stage27_uav_common import nonempty_path, publish_identical, seal
from stage34_identity_common import load_plan as load_source_plan, load_sealed as read_sealed, sample_weights


SCHEMA = "stage35_preserve_correct_oof_v1"
ARMS = ("supervised", "preserve", "correct", "preserve_correct", "distill_all_correct")
BASELINES = ("fast_initial", "sgla")
DESIGN = dict(fold_seed=2110, folds=4, seeds=[2110, 2111], updates=256,
    smoke_updates=8, batch_size=2, save_every=128, learning_rate=.0001,
    weight_decay=.0001, gradient_clip=1., reliable_iou=.5,
    kd_temperature=2., kd_weight=1., box_kd_weight=5., correction_weight=.25,
    contrast_margin=.2, hard_negatives=8, minimum_fit_cohort_frames=16,
    minimum_fit_cohort_videos=2, minimum_eval_cohort_frames=10,
    minimum_eval_cohort_videos=4, minimum_auc_gain=.3,
    reliable_macro_noninferiority=-.5, reliable_worst_video_floor=-1.,
    bootstrap_samples=5000, bootstrap_seed=2112)


def code_hashes():
    names = ("stage35_preserve_common.py", "stage35_preserve_runtime.py",
        "stage35_preserve_experiment.py", "stage35_preserve_summary.py",
        "stage34_identity_common.py", "stage34_identity_runtime.py",
        "stage34_identity_experiment.py", "stage18_tracking_loss.py", "stage4_runtime.py",
        "stage4_common.py", "stage20_compression_common.py", "stage20_student_train.py",
        "stage27_uav_common.py", "stage27_uav_collect.py", "stage27_uav_summary.py",
        "stage28_response_collect.py", "validation_metrics.py")
    return {name: file_hash(Path(__file__).with_name(name)) for name in names}


def load_sealed(path):
    return read_sealed(path, SCHEMA)


def fold_design(entries, samples):
    groups = sorted(e["source_group_proxy"] for e in entries)
    if len(groups) != 8 or len(set(groups)) != 8:
        raise ValueError("Need eight distinct original fit source groups")
    order = np.random.default_rng(DESIGN["fold_seed"]).permutation(groups).tolist()
    folds = []
    for index in range(DESIGN["folds"]):
        held = order[2 * index:2 * index + 2]
        fit = [e["sequence"] for e in entries if e["source_group_proxy"] not in held]
        checking = [e["sequence"] for e in entries if e["source_group_proxy"] in held]
        selected = [s for s in samples if s["sequence"] in fit]
        support = {}
        for cohort in ("easy", "hard"):
            rows = [s for s in selected if s["cohort"] == cohort]
            support[cohort] = dict(frames=len(rows), videos=len({r["sequence"] for r in rows}))
        passed = all(s["frames"] >= DESIGN["minimum_fit_cohort_frames"] and
                     s["videos"] >= DESIGN["minimum_fit_cohort_videos"] for s in support.values())
        folds.append(dict(fold=index, fit=sorted(fit), held_out=sorted(checking),
                          support=support, support_pass=passed))
    return folds


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Use a separate Stage-35 output tree")
    if (output / "manifest.json").exists():
        plan = load_plan(output)
        if plan["source_stage34"] != str(source):
            raise ValueError("Existing plan belongs to another source")
        print("[reuse, audited] " + plan["fingerprint"], flush=True)
        return plan
    old = load_source_plan(source)
    result_path = source / "analysis" / "stage34_decision.json"
    result = read_sealed(result_path)
    if (result.get("status") != "NO_SPATIAL_REPRESENTATION_GAIN_ESTABLISHED" or
            result.get("plan_fingerprint") != old["fingerprint"]):
        raise ValueError("Need the frozen completed negative Stage-34 decision")
    entries, samples = copy.deepcopy(old["fit"]), copy.deepcopy(old["samples"])
    folds = fold_design(entries, samples)
    support = seal(dict(schema=SCHEMA, role="fit_only_support_preflight", folds=folds,
        status="PASS_FOLD_SUPPORT" if all(f["support_pass"] for f in folds) else "INSUFFICIENT_FOLD_SUPPORT"))
    publish_identical(output / "support_audit.json", support)
    if not all(f["support_pass"] for f in folds):
        raise ValueError("Insufficient fixed fold support; preserve audit, do not relax the gate")
    plan = seal(dict(schema=SCHEMA, role="consumed_fit_only_group_oof_corrective_distillation",
        source_stage34=str(source), source_plan_fingerprint=old["fingerprint"],
        source_decision_sha256=file_hash(result_path), design=DESIGN, code_sha256=code_hashes(),
        root=old["root"], fit=entries, samples=samples, folds=folds,
        runtime=old["runtime"], search_factor=old["search_factor"], search_size=old["search_size"],
        excluded_consumed_checking_groups=sorted({e["source_group_proxy"] for e in old["checking"]} |
            set(old["excluded_checking_groups"])),
        initialization="Original SGLA checkpoint only; no Stage-34 trained weights",
        training_state="Frozen original SGLA parents; GT reliability gates are training-only",
        gate_source="easy/hard sample labels from original SGLA executed-box IoU >=0.5 / <0.5",
        inference="Physical six-block Fast path; original head/decoder; no identity module or teacher",
        statistical_role="Previously consumed development fit videos; OOF is not untouched validation",
        original_checking_labels_loaded=False))
    publish_identical(output / "manifest.json", plan)
    print("Stage-35: 4 folds x 2 seeds x 5 arms = 40 models; 8 OOF videos / 96 full traces", flush=True)
    print("Plan fingerprint: " + plan["fingerprint"], flush=True)
    return plan


def load_plan(root):
    plan = load_sealed(Path(root) / "manifest.json")
    if plan.get("design") != DESIGN or plan.get("code_sha256") != code_hashes():
        raise ValueError("Frozen Stage-35 code/design changed")
    old = load_source_plan(plan["source_stage34"])
    if (old["fingerprint"] != plan["source_plan_fingerprint"] or
            file_hash(Path(plan["source_stage34"]) / "analysis/stage34_decision.json") != plan["source_decision_sha256"] or
            plan["fit"] != old["fit"] or plan["samples"] != old["samples"] or
            plan["folds"] != fold_design(plan["fit"], plan["samples"])):
        raise ValueError("Frozen source or grouped split changed")
    return plan


def fold_samples(plan, fold):
    spec = plan["folds"][fold]
    samples = [s for s in plan["samples"] if s["sequence"] in spec["fit"]]
    if set(spec["fit"]) & set(spec["held_out"]) or len(spec["fit"]) != 6 or len(spec["held_out"]) != 2:
        raise ValueError("Invalid OOF group isolation")
    return samples, sample_weights(samples)
