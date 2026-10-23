"""CPU audit of matched fixed-student training and autonomous paired rollouts."""

import argparse
import json
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_csv, write_json
from stage20_compression_common import (ARMS, METHODS, audit_trace, factorial_effect,
                                        load_contract, nonempty_path, paired_interval)


def audit_training(directory, contract):
    data = json.loads((directory / "stage20_artifacts.json").read_text(encoding="utf-8"))
    if data.get("contract_fingerprint") != contract["fingerprint"] or not data.get("teacher_unchanged"):
        raise ValueError("Teacher/contract integrity missing")
    artifacts = {(a["seed"], a["method"]): a for a in data["checkpoints"]}
    keys = {(seed, arm) for seed in contract["seeds"] for arm in ARMS}
    if len(artifacts) != len(data["checkpoints"]) or set(artifacts) != keys:
        raise ValueError("Missing, repeated or unexpected trained arms/seeds")
    settings = contract["training"]
    fit = contract["sources"]["fit"]
    epochs = settings["epochs"]
    frames = epochs * sum(s["stop"] - s["start"] - 1 for s in fit)
    valid = epochs * sum(sum(s["valid"][1:]) for s in fit)
    updates = 0
    step = settings["accumulation_frames"]
    for source in fit:
        mask = source["valid"][1:]
        updates += epochs * sum(any(mask[i:i + step]) for i in range(0, len(mask), step))
    for item in artifacts.values():
        # Resolve from this run, not from a possibly moved server's absolute path.
        path = directory / "checkpoints" / ("seed{}".format(item["seed"])) / item["method"] / "last.pt"
        if not path.is_file() or file_hash(path) != item["sha256"]:
            raise ValueError("Checkpoint missing/changed: " + str(path))
        if ((item["epoch"], item["frames"], item["valid_frames"], item["updates"])
                != (epochs, frames, valid, updates)):
            raise ValueError("Training epochs/frame exposure/updates differ across arms")
        if item["final_state_sha256"] == contract["initial_state_sha256"]:
            raise ValueError("Student did not change")
        if item["method"].startswith("kd_") and (item["teacher_accepted"] < 1 or item["kd_loss_sum"] <= 0):
            raise ValueError("KD arm had no active teacher supervision")
    return artifacts


def read_scores(directory, contract, artifacts):
    rows = []
    for seed in contract["seeds"]:
        for source in contract["sources"]["check"]:
            for method in METHODS:
                path = directory / "evaluation" / ("seed{}".format(seed)) / method / (source["sequence"] + ".npz")
                with np.load(path, allow_pickle=False) as archive:
                    metadata = json.loads(str(archive["metadata_json"].item()))
                    data = {k: archive[k] for k in archive.files if k != "metadata_json"}
                digest = (contract["runtime"]["checkpoint_sha256"] if method == "teacher" else
                          contract["initial_state_sha256"] if method == "compact_initial" else
                          artifacts[seed, method]["sha256"])
                expected = dict(contract_fingerprint=contract["fingerprint"], seed=seed, method=method,
                                sequence=source["sequence"], checkpoint_sha256=digest)
                if metadata != expected:
                    raise ValueError("Trace identity mismatch: " + str(path))
                score = audit_trace(data, source["gt"][0], np.arange(source["start"] + 1, source["stop"]))
                gt = np.asarray(source["gt"][1:], dtype=float)
                if (not np.allclose(gt, data["gt"], rtol=0, atol=0, equal_nan=True)
                        or not np.array_equal(source["valid"][1:], data["valid"])):
                    raise ValueError("Evaluation labels differ from frozen clip")
                warm = contract["analysis"]["latency_warmup_frames"]
                latency = data["latency_ms"][warm:]
                if not len(latency):
                    raise ValueError("No timed frames after warm-up")
                rows.append(dict(seed=seed, sequence=source["sequence"], method=method, auc=score,
                                 valid_frames=int(data["valid"].sum()),
                                 failure_ratio=float((data["iou"][data["valid"]] < .2).mean()),
                                 timed_frames=len(latency), latency_sum_ms=float(latency.sum()),
                                 p95_ms=float(np.percentile(latency, 95))))
    return rows


def summarize(rows, contract):
    seeds = contract["seeds"]
    names = [s["sequence"] for s in contract["sources"]["check"]]
    index = {(r["seed"], r["sequence"], r["method"]): r for r in rows}
    scores = {m: np.asarray([np.mean([index[s, name, m]["auc"] for s in seeds]) for name in names])
              for m in METHODS}
    analyses = contract["analysis"]
    contrast_values = dict(
        kd_student_minus_kd_reference=scores["kd_student"] - scores["kd_reference"],
        kd_student_minus_supervised_student=scores["kd_student"] - scores["supervised_student"],
        correction_interaction=factorial_effect(scores),
        kd_student_minus_teacher=scores["kd_student"] - scores["teacher"],
        kd_student_minus_compact_initial=scores["kd_student"] - scores["compact_initial"])
    paired = [dict(contrast=name, num_videos=len(names), num_seeds=len(seeds),
                   **paired_interval(values, analyses["bootstrap_samples"], analyses["seed"]))
              for name, values in contrast_values.items()]
    policies, seed_rows = [], []
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        timing = sum(r["latency_sum_ms"] for r in selected) / sum(r["timed_frames"] for r in selected)
        policies.append(dict(method=method, num_videos=len(names), num_seeds=len(seeds),
                             auc=float(scores[method].mean()),
                             gain_vs_teacher=float((scores[method] - scores["teacher"]).mean()),
                             failure_ratio=float(np.mean([r["failure_ratio"] for r in selected])),
                             mean_crop_model_decode_ms=timing,
                             mean_sequence_p95_ms=float(np.mean([r["p95_ms"] for r in selected]))))
        for seed in seeds:
            seed_rows.append(dict(seed=seed, method=method, auc=float(np.mean(
                [index[seed, name, method]["auc"] for name in names]))))
    effect = {r["contrast"]: r for r in paired}
    latency = {r["method"]: r["mean_crop_model_decode_ms"] for r in policies}
    checks = dict(
        teacher_correction_lift=all(effect[k]["ci_low"] > 0 for k in analyses["primary_contrasts"]),
        interaction_lift=effect["correction_interaction"]["ci_low"] > 0,
        teacher_noninferiority=effect["kd_student_minus_teacher"]["ci_low"] > -analyses["teacher_auc_noninferiority_margin"],
        parameters=contract["architecture"]["parameter_reduction"] >= analyses["target_parameter_reduction"],
        latency=latency["kd_student"] <= latency["teacher"] * (1 - analyses["target_latency_reduction"]))
    if contract["mode"] == "smoke":
        status = "PASS_MATCHED_TRAINING_SMOKE"
    else:
        status = "PASS_DEVELOPMENT_FEASIBILITY" if all(checks.values()) else "FAIL_DEVELOPMENT_FEASIBILITY"
    decision = dict(schema="stage20_fixed_student_decision_v1", status=status,
                    contract_fingerprint=contract["fingerprint"], checks=checks,
                    role="engineering_only" if contract["mode"] == "smoke" else "development_only",
                    checkpoint_selection="all_declared_seeds_fixed_final_epoch",
                    paired_unit="video_after_seed_averaging_not_seed_times_video",
                    warning="Smoke checks execution only. These reused development videos cannot establish novelty or external generalization. Timing includes crop, preprocessing, model and decode, not disk I/O or initialization.",
                    next_step=("Inspect smoke integrity before explicitly starting full training."
                               if contract["mode"] == "smoke" else
                               "Freeze this development result. Do not choose another pruning ratio, KD weight or epoch on the check outcomes."))
    return policies, paired, seed_rows, decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True)
    args = parser.parse_args()
    try:
        directory, contract = load_contract(args.input)
        artifacts = audit_training(directory, contract)
        rows = read_scores(directory, contract, artifacts)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, "Stage-20 audit incomplete or failed: {}\nResume the exact training command; per-arm progress is in checkpoints/seed*/ARM/history.csv.\n".format(error))
    policies, paired, seed_rows, decision = summarize(rows, contract)
    output = directory / "analysis"
    for name, values in (("policy_summary", policies), ("paired", paired), ("seed_summary", seed_rows),
                         ("sequence_summary", rows)):
        write_csv(output / ("stage20_" + name + ".csv"), values)
    write_json(output / "stage20_decision.json", decision)
    print("Audited matched 2x2 training; all rollouts use each student's own predictions.")
    print("method                       AUC    vs_teacher    crop_model_decode_ms")
    for row in policies:
        print("{method:<26s} {auc:7.4f} {gain_vs_teacher:13.4f} {mean_crop_model_decode_ms:23.3f}".format(**row))
    print("\nVideo-paired contrasts (seeds averaged before bootstrap):")
    for row in paired:
        print("{contrast:<38s} {mean_delta:8.4f} [{ci_low:.4f},{ci_high:.4f}]".format(**row))
    print("\nDecision: " + decision["status"])
    print("Reports: " + str(output))


if __name__ == "__main__":
    main()
