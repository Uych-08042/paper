"""Compact decision plus a separate full provenance ledger for matched augmentation."""

import argparse
from pathlib import Path

import numpy as np

from stage4_common import file_hash
from stage27_uav_collect import writer_lock
from stage27_uav_common import failure_events, nonempty_path, publish_identical, seal, success_auc
from stage27_uav_summary import write_table
from stage40_crop_common import (ARMS, DESIGN, SCHEMA, SEEDS, arm_seed, baseline, load_plan,
    load_result, methods, prior, read_sealed, result_path, verify_bindings)


def paired(values, smaller_better=False):
    values = np.asarray(values, float)
    if not len(values):
        return dict(videos=0, mean_delta=None, ci_low=None, ci_high=None, worst_video_delta=None,
            positive_videos=0, zero_videos=0, negative_videos=0, minimum_leave_one_video_out=None)
    rng = np.random.default_rng(DESIGN["bootstrap_seed"])
    means = values[rng.integers(0, len(values), (DESIGN["bootstrap_samples"], len(values)))].mean(1)
    low, high = map(float, np.quantile(means, [.025, .975])) if len(values) > 1 else (None, None)
    return dict(videos=len(values), mean_delta=float(values.mean()), ci_low=low, ci_high=high,
        worst_video_delta=float(values.max() if smaller_better else values.min()), positive_videos=int((values > 1e-9).sum()),
        zero_videos=int((np.abs(values) <= 1e-9).sum()), negative_videos=int((values < -1e-9).sum()),
        minimum_leave_one_video_out=float(min(np.delete(values, i).mean() for i in range(len(values)))) if len(values) > 1 else None)


def full_row(entry, arm, seed, data, fast):
    reliable = fast["iou"] >= .5
    events = failure_events(data["iou"])
    return dict(sequence=entry["sequence"], arm=arm, seed=seed, auc=success_auc(data["iou"]),
        low_iou_fraction=float((data["iou"] < .2).mean()), failure_events=len(events),
        longest_failure_fraction=max((e["end"] - e["start"] + 1 for e in events), default=0) / len(data["iou"]),
        reliable_auc=success_auc(data["iou"][reliable]) if reliable.sum() >= DESIGN["minimum_reliable_frames"] else None)


METRICS = ("auc", "low_iou_fraction", "failure_events", "longest_failure_fraction", "reliable_auc")


def analyze(rows, stress, plan):
    names = [e["sequence"] for e in plan["fit"]]
    expected = {(n, a, s) for n in names for a in ARMS for s in SEEDS} | {(n, b, "shared") for n in names for b in ("fast_initial", "sgla")}
    lookup = {(r["sequence"], r["arm"], r["seed"]): r for r in rows}
    expected_stress = {(n, m, a) for n in names for m in methods() for a in plan["tasks"][n]}
    if (len(names) != 8 or len({e["source_group_proxy"] for e in plan["fit"]}) != 8 or
            set(lookup) != expected or len(rows) != len(expected) or
            {(r["sequence"], r["method"], r["anchor"]) for r in stress} != expected_stress or len(stress) != len(expected_stress)):
        raise ValueError("Need complete unmixed OOF/stress video/seed/arm coverage")
    videos, index = [], {}
    for name in names:
        for arm in ARMS + ("fast_initial", "sgla"):
            selected = [lookup[(name, arm, s)] for s in (SEEDS if arm in ARMS else ("shared",))]
            value = dict(sequence=name, arm=arm, seed_repeats=len(selected),
                **{k: float(np.mean([v[k] for v in selected])) if all(v[k] is not None for v in selected) else None for k in METRICS})
            videos.append(value)
            index[(name, arm)] = value
    comparisons = []
    for metric in METRICS:
        for left, right in (("augmented", "clean"), ("augmented", "fast_initial"), ("augmented", "sgla"), ("clean", "fast_initial")):
            supported = [n for n in names if index[(n, left)][metric] is not None and index[(n, right)][metric] is not None]
            smaller = metric in ("low_iou_fraction", "failure_events", "longest_failure_fraction")
            comparisons.append(dict(contrast=left + "_minus_" + right, metric=metric,
                favorable_direction="negative" if smaller else "positive",
                **paired([index[(n, left)][metric] - index[(n, right)][metric] for n in supported], smaller)))
    def comparison(metric, right):
        return next(r for r in comparisons if r["metric"] == metric and r["contrast"] == "augmented_minus_" + right)
    primary = comparison("auc", "clean")
    reliable = comparison("reliable_auc", "fast_initial")
    seed_gains = [float(np.mean([lookup[(n, "augmented", s)]["auc"] - lookup[(n, "clean", s)]["auc"] for n in names])) for s in SEEDS]
    stability, influence = [], []
    for name in names:
        delta = index[(name, "augmented")]["auc"] - index[(name, "clean")]["auc"]
        rest = [index[(n, "augmented")]["auc"] - index[(n, "clean")]["auc"] for n in names if n != name]
        influence.append(dict(sequence=name, auc_gain_vs_clean=delta, leave_one_video_out=float(np.mean(rest))))
        for seed in SEEDS:
            for ref in ("clean", "fast_initial"):
                base = lookup[(name, ref, seed if ref == "clean" else "shared")]
                value = lookup[(name, "augmented", seed)]
                stability.append(dict(sequence=name, seed=seed, reference=ref, original_stable=base["failure_events"] == 0,
                    introduced_failure=base["failure_events"] == 0 and value["failure_events"] > 0,
                    auc_delta=value["auc"] - base["auc"], low_iou_delta=value["low_iou_fraction"] - base["low_iou_fraction"]))
    stable_fast = {r["sequence"] for r in stability if r["reference"] == "fast_initial" and r["original_stable"]}
    introduced_fast = {r["sequence"] for r in stability if r["reference"] == "fast_initial" and r["introduced_failure"]}
    introduced_clean = {r["sequence"] for r in stability if r["reference"] == "clean" and r["introduced_failure"]}
    support = reliable["videos"] >= DESIGN["minimum_reliable_videos"] and bool(stable_fast)
    primary_pass = (primary["mean_delta"] >= DESIGN["minimum_auc_gain"] and primary["ci_low"] > 0 and
        primary["positive_videos"] >= DESIGN["minimum_positive_videos"] and primary["minimum_leave_one_video_out"] > 0 and min(seed_gains) > 0)
    guard = bool(support and reliable["ci_low"] is not None and reliable["ci_low"] >= DESIGN["reliable_ci_floor"] and
        reliable["worst_video_delta"] >= DESIGN["reliable_worst_video_floor"] and not introduced_fast and not introduced_clean and
        comparison("low_iou_fraction", "clean")["mean_delta"] <= 0 and comparison("low_iou_fraction", "fast_initial")["mean_delta"] <= 0 and
        comparison("auc", "fast_initial")["mean_delta"] >= 0)
    stress_videos = []
    for name in names:
        for arm in ARMS:
            values = [r for r in stress if r["sequence"] == name and arm_seed(r["method"])[0] == arm]
            stress_videos.append(dict(sequence=name, arm=arm,
                **{k: float(np.mean([r[k] for r in values])) for k in ("clean_iou", "perturbed_iou", "post_iou_loss")}))
    si = {(r["sequence"], r["arm"]): r for r in stress_videos}
    stress_pair = paired([si[(n, "augmented")]["post_iou_loss"] - si[(n, "clean")]["post_iou_loss"] for n in names], True)
    stress_supported = stress_pair["ci_high"] is not None and stress_pair["ci_high"] < 0
    passed = primary_pass and guard
    decision = dict(status="INSUFFICIENT_RETENTION_SUPPORT" if not support else
        "PASS_MATCHED_AUGMENTATION_DEVELOPMENT_SCREEN" if passed else "NO_MATCHED_AUGMENTATION_GAIN_ESTABLISHED",
        pilot_pass=bool(passed), primary_auc_pass=bool(primary_pass), retention_guard_pass=guard, support_pass=support,
        primary=primary, augmented_minus_fast=comparison("auc", "fast_initial"), reliable_vs_fast=reliable,
        primary_seed_gains=seed_gains, original_fast_stable_videos=len(stable_fast),
        introduced_fast_stable_videos=len(introduced_fast), introduced_clean_stable_videos=len(introduced_clean),
        stress_augmented_minus_clean=stress_pair, stress_reduction_supported=bool(stress_supported),
        augmentation_stability_mechanism_supported=bool(passed and stress_supported))
    policies = [dict(arm=a, videos=8, **{k: float(np.mean([index[(n, a)][k] for n in names])) for k in METRICS[:-1]})
        for a in ARMS + ("fast_initial", "sgla")]
    return dict(policy_summary=policies, seed_video_summary=rows, video_summary=videos, paired=comparisons,
        stability=stability, video_influence=influence, stress_anchor_summary=stress, stress_video_summary=stress_videos), decision


def summarize(root):
    root = Path(root).resolve()
    plan = load_plan(root)
    from stage40_crop_experiment import smoke_gate
    smoke_gate(root, plan)
    run = read_sealed(root / "evaluation/run.json", SCHEMA)
    expected_states = {"{}:{}".format(f, m) for f in range(DESIGN["folds"]) for m in methods()}
    if (run.get("plan_fingerprint") != plan["fingerprint"] or run.get("mode") != "evaluation" or run.get("methods") != methods() or
            run.get("runtime") != plan["source_runtime"] or run.get("teacher_sha256") != plan["source_frozen_neural_model_sha256"]["sgla"] or
            set(run.get("model_state_sha256", {})) != expected_states):
        raise ValueError("Mixed evaluation run/models")
    verify_bindings(run["model_file_sha256"])
    expected = {result_path(root, e, m, a).resolve() for e in plan["fit"] for m in methods() for a in [None] + plan["tasks"][e["sequence"]]}
    if {p.resolve() for p in (root / "evaluation").rglob("*.npz")} != expected:
        raise ValueError("Need all 32 full traces and 96 complete stress packs")
    bindings = dict(plan["source_file_sha256"])
    bindings.update(run["model_file_sha256"])
    for path in (root / "manifest.json", root / "evaluation/run.json", root / "smoke/stage40_smoke.json"):
        bindings[str(path)] = file_hash(path)
    rows, stress = [], []
    for entry in plan["fit"]:
        fast = baseline(plan, entry, "fast_initial")
        for label in ("fast_initial", "sgla"):
            rows.append(full_row(entry, label, "shared", fast if label == "fast_initial" else baseline(plan, entry, label), fast))
        for method in methods():
            arm, seed = arm_seed(method)
            for anchor in [None] + plan["tasks"][entry["sequence"]]:
                path = result_path(root, entry, method, anchor)
                data = load_result(path, plan, run, entry, method, anchor)
                for p in (path, path.with_suffix(".json")):
                    bindings[str(p)] = file_hash(p)
                if anchor is None:
                    rows.append(full_row(entry, arm, seed, data, fast))
                else:
                    clean = data["iou"][0, 1:]
                    perturbed = data["iou"][1:, 1:]
                    stress.append(dict(sequence=entry["sequence"], method=method, anchor=anchor, clean_iou=float(clean.mean()),
                        perturbed_iou=float(perturbed.mean()), post_iou_loss=float((clean[None] - perturbed).mean())))
    tables, decision = analyze(rows, stress, plan)
    output = root / "analysis"
    with writer_lock(output):
        path = output / "stage40_decision.json"
        if path.exists():
            saved = read_sealed(path, SCHEMA)
            evidence = read_sealed(output / "stage40_evidence.json", SCHEMA)
            if file_hash(output / "stage40_evidence.json") != saved["evidence_sha256"] or evidence["source_file_sha256"] != bindings:
                raise ValueError("Completed Stage-40 evidence changed")
            verify_bindings({str(output / n): d for n, d in saved["table_sha256"].items()})
            print("Stage-40: " + saved["status"] + " (reuse, audited)")
            return saved
        table_hashes = {}
        for label, values in tables.items():
            target = output / ("stage40_" + label + ".csv")
            write_table(target, values)
            table_hashes[target.name] = file_hash(target)
        verify_bindings(bindings)
        evidence = seal(dict(schema=SCHEMA, plan_fingerprint=plan["fingerprint"], run_fingerprint=run["fingerprint"],
            source_file_sha256=bindings, table_sha256=table_hashes))
        publish_identical(output / "stage40_evidence.json", evidence)
        report = seal(dict(schema=SCHEMA, role="consumed_development_matched_augmentation_screen", **decision,
            plan_fingerprint=plan["fingerprint"], run_fingerprint=run["fingerprint"], videos=8, seeds=list(SEEDS),
            new_models=16, new_full_trajectories=32, stress_branch_frames=8 * 3 * 4 * 7 * prior.DESIGN["horizon"],
            evidence_sha256=file_hash(output / "stage40_evidence.json"), table_sha256=table_hashes,
            training_performed=True, original_checking_labels_loaded=False, novel_method_established=False,
            next_step="Freeze this result and design a distinct mechanism before new-group validation" if decision["pilot_pass"] else
                "Preserve the fixed result; do not search steps, perturbation amplitudes, folds or checkpoints on these outcomes",
            warnings=["Ordinary crop augmentation is established prior work; a positive result is an augmentation control, not a novel method.",
                "These eight videos and prior outcomes informed design; grouped OOF is development evidence, not external confirmation.",
                "Both arms use two supervised views and identical frame draws, initialization, optimizer and update count; only the second crop changes.",
                "Only L5-L6 are trained; targets use actual rounded crop coordinates. Neither arm rolls its training states forward.",
                "Evaluation uses own predicted crops; stress uses shared frozen SGLA starts. Neither uses future GT in the controller.",
                "Seeds and anchors are averaged inside videos; nominal bootstrap is conditional on fixed overlapping OOF fits.",
                "Stress directions/amplitudes also occur in training; this is held-out-video transfer, not unseen-perturbation generalization.",
                "Reliable cohorts are offline baseline-defined; a noninferiority margin is a practical tolerance, not zero damage.",
                "No deployment speedup, UAV-specific mechanism or external generalization claim is established."]))
        publish_identical(path, report)
    for row in tables["policy_summary"]:
        print("{:<14s} AUC={:.4f} low-IoU={:.2%}".format(row["arm"], row["auc"], row["low_iou_fraction"]))
    print("Stage-40: " + report["status"])
    print("Reports: " + str(output))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True)
    args = parser.parse_args()
    summarize(Path(args.input).expanduser())


if __name__ == "__main__":
    main()
