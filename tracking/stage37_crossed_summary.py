"""Video-paired descriptive summary of frozen crossed-decoder full rollouts."""

import argparse
from pathlib import Path

import numpy as np

from stage4_common import file_hash
from stage27_uav_collect import writer_lock
from stage27_uav_common import failure_events, nonempty_path, publish_identical, seal, success_auc
from stage27_uav_summary import interval, write_table
from stage37_crossed_common import (ARMS, DESIGN, SCHEMA, load_plan, load_trace, methods,
    native_parity, read_sealed, verify_bindings)


METRICS = ("auc", "low_iou_fraction", "failure_events", "student_reliable_auc", "sgla_reliable_auc")


def metrics(data, student, teacher):
    result = dict(auc=success_auc(data["iou"]), low_iou_fraction=float(np.mean(data["iou"] < .2)),
        failure_events=len(failure_events(data["iou"])))
    for key, baseline in (("student_reliable_auc", student), ("sgla_reliable_auc", teacher)):
        mask = baseline["iou"] >= DESIGN["reliable_iou"]
        result[key] = success_auc(data["iou"][mask]) if mask.sum() >= DESIGN["minimum_conditional_frames"] else None
    return result


def analyze(rows):
    names = sorted({r["sequence"] for r in rows})
    lookup = {(r["sequence"], r["seed"], r["arm"]): r for r in rows}
    expected = {(n, s, a) for n in names for s in DESIGN["seeds"] for a in ARMS}
    if len(rows) != len(expected) or set(lookup) != expected:
        raise ValueError("Missing or duplicate seed/video/arm rows")
    averaged = []
    for name in names:
        for arm in ARMS:
            values = [lookup[(name, s, arm)] for s in DESIGN["seeds"]]
            record = dict(sequence=name, arm=arm, seed_repeats=len(values))
            for metric in METRICS:
                record[metric] = float(np.mean([v[metric] for v in values])) if all(v[metric] is not None for v in values) else None
            averaged.append(record)
    index = {(r["sequence"], r["arm"]): r for r in averaged}
    rng = np.random.default_rng(DESIGN["bootstrap_seed"])
    pairs = []
    comparisons = [("teacher_response", "student"), ("teacher_geometry", "student"),
        ("teacher_response", "teacher_geometry"), ("teacher_response", "sgla")]
    for metric in METRICS:
        for left, right in comparisons:
            supported = [n for n in names if index[(n, left)][metric] is not None and index[(n, right)][metric] is not None]
            delta = [index[(n, left)][metric] - index[(n, right)][metric] for n in supported]
            lo, hi = interval(delta, rng, DESIGN["bootstrap_samples"])
            seed_deltas = [float(np.mean([lookup[(n, s, left)][metric] - lookup[(n, s, right)][metric]
                for n in supported])) for s in DESIGN["seeds"]] if supported else []
            direction = -1 if metric in ("low_iou_fraction", "failure_events") else 1
            pairs.append(dict(contrast=left + "_minus_" + right, metric=metric, videos=len(supported),
                mean_delta=float(np.mean(delta)) if delta else None, ci_low=lo, ci_high=hi,
                worst_video_delta=float(min(delta) if direction > 0 else max(delta)) if delta else None,
                wins=sum(direction * d > 1e-9 for d in delta), ties=sum(abs(d) <= 1e-9 for d in delta),
                losses=sum(direction * d < -1e-9 for d in delta),
                seed_mean_deltas="|".join("{:.8f}".format(d) for d in seed_deltas),
                favorable_direction="positive" if direction > 0 else "negative"))
    delta = [index[(n, "sgla")]["auc"] - index[(n, "teacher_geometry")]["auc"]
        - index[(n, "teacher_response")]["auc"] + index[(n, "student")]["auc"] for n in names]
    lo, hi = interval(delta, rng, DESIGN["bootstrap_samples"])
    pairs.append(dict(contrast="response_by_geometry_interaction", metric="auc", videos=len(names),
        mean_delta=float(np.mean(delta)), ci_low=lo, ci_high=hi, worst_video_delta=float(min(delta)),
        wins=sum(d > 1e-9 for d in delta), ties=sum(abs(d) <= 1e-9 for d in delta),
        losses=sum(d < -1e-9 for d in delta), seed_mean_deltas="", favorable_direction="positive"))
    stability, influence = [], []
    primary = np.array([index[(n, "teacher_response")]["auc"] - index[(n, "student")]["auc"] for n in names])
    for i, name in enumerate(names):
        influence.append(dict(sequence=name, primary_auc_delta=float(primary[i]),
            macro_mean_contribution=float(primary[i] / len(names)),
            leave_one_video_out_mean_delta=float(np.delete(primary, i).mean()) if len(names) > 1 else None))
        for seed in DESIGN["seeds"]:
            for arm in ("teacher_response", "teacher_geometry"):
                for reference in ("student", "sgla"):
                    baseline, value = lookup[(name, seed, reference)], lookup[(name, seed, arm)]
                    key = reference + "_reliable_auc"
                    reliable = value[key] - baseline[key] if value[key] is not None and baseline[key] is not None else None
                    stability.append(dict(sequence=name, seed=seed, arm=arm, reference=reference,
                        original_stable=baseline["failure_events"] == 0,
                        introduced_failure=baseline["failure_events"] == 0 and value["failure_events"] > 0,
                        auc_delta=value["auc"] - baseline["auc"],
                        low_iou_fraction_delta=value["low_iou_fraction"] - baseline["low_iou_fraction"], reliable_auc_delta=reliable))
    policies = [dict(arm=a, videos=len(names), auc=float(np.mean([index[(n, a)]["auc"] for n in names])),
        low_iou_fraction=float(np.mean([index[(n, a)]["low_iou_fraction"] for n in names])),
        failure_events=float(np.mean([index[(n, a)]["failure_events"] for n in names]))) for a in ARMS]
    return policies, averaged, pairs, stability, influence


def summarize(root):
    root = Path(root).resolve()
    plan = load_plan(root)
    directory = root / "full"
    run = read_sealed(directory / "run.json", SCHEMA)
    from stage37_crossed_experiment import smoke_gate
    smoke_gate(root, plan)
    smoke_run = read_sealed(root / "smoke/run.json", SCHEMA)
    if (run.get("mode") != "full" or run.get("plan_fingerprint") != plan["fingerprint"] or
            run.get("methods") != methods() or any(run.get(k) != smoke_run.get(k)
            for k in ("runtime", "frozen_neural_model_sha256", "window"))):
        raise ValueError("Full runtime/models differ from the frozen smoke")
    expected = {e["sequence"] + "__" + m + ".npz" for e in plan["fit"] for m in methods()}
    if {p.name for p in (directory / "traces").glob("*.npz")} != expected:
        raise ValueError("Need all 56 full OOF traces; use --mode status for incomplete progress")
    bindings = {str(root / "manifest.json"): file_hash(root / "manifest.json"),
        str(directory / "run.json"): file_hash(directory / "run.json"),
        str(root / "smoke/run.json"): file_hash(root / "smoke/run.json"),
        str(root / "smoke/stage37_smoke.json"): file_hash(root / "smoke/stage37_smoke.json")}
    rows, parity, hashes = [], [], {}
    for entry in plan["fit"]:
        traces = {}
        for method in methods():
            path = directory / "traces" / (entry["sequence"] + "__" + method + ".npz")
            data = load_trace(path, plan, run, entry, method)
            traces[method] = data
            hashes[path.name] = file_hash(path)
            bindings[str(path)] = hashes[path.name]
            bindings[str(path.with_suffix(".json"))] = file_hash(path.with_suffix(".json"))
            result = native_parity(data, plan, entry, method)
            if result is not None:
                parity.append(dict(sequence=entry["sequence"], method=method, **result))
        for seed in DESIGN["seeds"]:
            student, teacher = traces["student_seed" + str(seed)], traces["sgla"]
            for arm in ARMS:
                data = teacher if arm == "sgla" else traces[arm + "_seed" + str(seed)]
                rows.append(dict(sequence=entry["sequence"], seed=seed, arm=arm,
                    tracked_frames=len(data["iou"]), **metrics(data, student, teacher)))
    policies, averaged, pairs, stability, influence = analyze(rows)
    primary = next(r for r in pairs if r["contrast"] == "teacher_response_minus_student" and r["metric"] == "auc")
    output = root / "analysis"
    with writer_lock(output):
        target = output / "stage37_diagnostic.json"
        if target.exists():
            saved = read_sealed(target, SCHEMA)
            if saved["source_file_sha256"] != bindings:
                raise ValueError("Completed Stage-37 summary source changed")
            verify_bindings({str(output / n): d for n, d in saved["table_sha256"].items()})
            print("Stage-37: COMPLETE_FROZEN_CROSSED_DIAGNOSTIC (reuse, audited)")
            return saved
        tables = dict(policy_summary=policies, seed_video_summary=rows, video_summary=averaged,
            paired=pairs, stability=stability, video_influence=influence, native_parity=parity)
        table_hashes = {}
        for name, values in tables.items():
            path = output / ("stage37_" + name + ".csv")
            write_table(path, values)
            table_hashes[path.name] = file_hash(path)
        verify_bindings(bindings)
        verify_bindings(plan["source_file_sha256"])
        report = seal(dict(schema=SCHEMA, role="consumed_development_frozen_crossed_decoder_mechanism",
            status="COMPLETE_FROZEN_CROSSED_DIAGNOSTIC", plan_fingerprint=plan["fingerprint"],
            run_fingerprint=run["fingerprint"], primary=primary, videos=len(plan["fit"]),
            seeds=DESIGN["seeds"], unique_full_trajectories=len(expected), cache_sha256=hashes,
            native_source_parity=parity, source_file_sha256=bindings, table_sha256=table_hashes,
            training_performed=False, learned_policy_evaluated=False, original_checking_labels_loaded=False,
            adoption_gate=None, warnings=[
                "All eight fit groups and both seeds have informed design; this is not independent external validation.",
                "Seeds are averaged within video before video bootstrap; common SGLA is collected once/video, not twice.",
                "OOF fits overlap; nominal conditional-on-fit intervals omit retraining uncertainty and multiple-comparison adjustment.",
                "All arms use their own predicted crops. Crossed output channels are decoder interventions, not separately trained modules.",
                "Both models run in every arm for same-state auditing. No learned compression, latency or speedup claim is established.",
                "Reliable cohorts use baseline GT outcomes only offline; they are not online triggers.",
                "A positive replacement effect does not establish that a compact student can learn that effect."],
            next_step="Interpret full-video response replacement, geometry replacement and stability before designing compact training"))
        publish_identical(target, report)
    for row in policies:
        print("{:<19s} AUC={:.4f} low-IoU={:.2%}".format(row["arm"], row["auc"], row["low_iou_fraction"]))
    print("Primary response replacement: {:+.4f} [{:+.4f},{:+.4f}]".format(primary["mean_delta"], primary["ci_low"], primary["ci_high"]))
    print("Stage-37: COMPLETE_FROZEN_CROSSED_DIAGNOSTIC; mechanism diagnostic, no method PASS")
    print("Reports: " + str(output))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True)
    args = parser.parse_args()
    summarize(Path(args.input).expanduser())


if __name__ == "__main__":
    main()
