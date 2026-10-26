"""Paired full-video size/offset diagnostic with reused frozen endpoints."""

import argparse
from pathlib import Path

import numpy as np

from stage4_common import file_hash
from stage27_uav_collect import writer_lock
from stage27_uav_common import failure_events, nonempty_path, publish_identical, seal
from stage27_uav_summary import write_table
from stage37_crossed_summary import metrics as old_metrics
from stage38_geometry_common import (ARMS, NEW_ARMS, REFERENCE, DESIGN, SCHEMA, load_plan,
    load_trace, methods, read_sealed, source_trace, verify_bindings)


METRICS = ("auc", "low_iou_fraction", "failure_events", "longest_failure_fraction",
    "student_reliable_auc", "sgla_reliable_auc")


def metrics(data, student, teacher):
    result = old_metrics(data, student, teacher)
    events = failure_events(data["iou"])
    longest = max((e["end"] - e["start"] + 1 for e in events), default=0)
    return dict(result, longest_failure_frames=longest, longest_failure_fraction=float(longest / len(data["iou"])))


def paired(values, rng, family=False):
    values = np.asarray(values, float)
    if len(values) < 2:
        return None, None, None, None
    means = values[rng.integers(0, len(values), (DESIGN["bootstrap_samples"], len(values)))].mean(1)
    lo, hi = map(float, np.quantile(means, [.025, .975]))
    alpha = .05 / (2 * DESIGN["family_size"])
    flo, fhi = map(float, np.quantile(means, [alpha, 1 - alpha])) if family else (None, None)
    return lo, hi, flo, fhi


def analyze(rows):
    names = sorted({r["sequence"] for r in rows})
    arms = ARMS + (REFERENCE,)
    lookup = {(r["sequence"], r["seed"], r["arm"]): r for r in rows}
    expected = {(n, s, a) for n in names for s in DESIGN["seeds"] for a in arms}
    if len(rows) != len(expected) or set(lookup) != expected:
        raise ValueError("Missing or duplicate size/offset seed-video-arm rows")
    averaged = []
    for name in names:
        for arm in arms:
            values = [lookup[(name, s, arm)] for s in DESIGN["seeds"]]
            record = dict(sequence=name, arm=arm, seed_repeats=len(values))
            for metric in METRICS:
                record[metric] = float(np.mean([v[metric] for v in values])) if all(v[metric] is not None for v in values) else None
            averaged.append(record)
    index = {(r["sequence"], r["arm"]): r for r in averaged}
    rng = np.random.default_rng(DESIGN["bootstrap_seed"])
    pairs = []
    specs = [(a, "student") for a in ARMS[1:]] + [("teacher_size", "teacher_offset"),
        ("teacher_geometry", "teacher_size"), ("teacher_geometry", "teacher_offset")]
    specs += [(a, REFERENCE) for a in ARMS[1:]]
    for metric in METRICS:
        for left, right in specs:
            supported = [n for n in names if index[(n, left)][metric] is not None and index[(n, right)][metric] is not None]
            delta = [index[(n, left)][metric] - index[(n, right)][metric] for n in supported]
            contrast = left + "_minus_" + right
            lo, hi, flo, fhi = paired(delta, rng, metric == "auc" and contrast in DESIGN["co_primary"])
            seeds = [float(np.mean([lookup[(n, s, left)][metric] - lookup[(n, s, right)][metric]
                for n in supported])) for s in DESIGN["seeds"]] if supported else []
            direction = -1 if metric in ("low_iou_fraction", "failure_events", "longest_failure_fraction") else 1
            pairs.append(dict(contrast=contrast, metric=metric, videos=len(supported),
                mean_delta=float(np.mean(delta)) if delta else None, ci_low=lo, ci_high=hi,
                family_ci_low=flo, family_ci_high=fhi,
                worst_video_delta=float(min(delta) if direction > 0 else max(delta)) if delta else None,
                wins=sum(direction * d > 1e-9 for d in delta), ties=sum(abs(d) <= 1e-9 for d in delta),
                losses=sum(direction * d < -1e-9 for d in delta),
                seed_mean_deltas="|".join("{:.8f}".format(d) for d in seeds),
                favorable_direction="positive" if direction > 0 else "negative"))
    factors, residual = [], 0.
    effects = {"size_order_average": [], "offset_order_average": [], "size_offset_interaction": []}
    for name in names:
        base, size, offset, both = [index[(name, a)]["auc"] for a in ARMS]
        s, o = .5 * ((size - base) + (both - offset)), .5 * ((offset - base) + (both - size))
        residual = max(residual, abs(s + o - (both - base)))
        effects["size_order_average"].append(s)
        effects["offset_order_average"].append(o)
        effects["size_offset_interaction"].append(both - size - offset + base)
    if residual > 1e-10:
        raise ValueError("Full-trajectory factorial accounting failed")
    for label, values in effects.items():
        lo, hi, _, _ = paired(values, rng)
        factors.append(dict(effect=label, videos=len(names), mean_auc_delta=float(np.mean(values)), ci_low=lo, ci_high=hi))
    stability, influence = [], []
    for name in names:
        for seed in DESIGN["seeds"]:
            for arm in ARMS[1:]:
                for reference in ("student", REFERENCE):
                    value, baseline = lookup[(name, seed, arm)], lookup[(name, seed, reference)]
                    key = "student_reliable_auc" if reference == "student" else "sgla_reliable_auc"
                    reliable = value[key] - baseline[key] if value[key] is not None and baseline[key] is not None else None
                    stability.append(dict(sequence=name, seed=seed, arm=arm, reference=reference,
                        original_stable=baseline["failure_events"] == 0,
                        introduced_failure=baseline["failure_events"] == 0 and value["failure_events"] > 0,
                        auc_delta=value["auc"] - baseline["auc"], reliable_auc_delta=reliable,
                        low_iou_fraction_delta=value["low_iou_fraction"] - baseline["low_iou_fraction"],
                        longest_failure_fraction_delta=value["longest_failure_fraction"] - baseline["longest_failure_fraction"]))
    for arm in ARMS[1:]:
        for reference in ("student", REFERENCE):
            values = np.array([index[(n, arm)]["auc"] - index[(n, reference)]["auc"] for n in names])
            positive = np.maximum(values, 0).sum()
            for i, name in enumerate(names):
                influence.append(dict(sequence=name, arm=arm, reference=reference, auc_delta=float(values[i]),
                    macro_mean_contribution=float(values[i] / len(names)),
                    positive_gain_share=float(max(values[i], 0) / positive) if positive > 0 else None,
                    leave_one_video_out_mean_delta=float(np.delete(values, i).mean()) if len(names) > 1 else None))
    policies = [dict(arm=a, role="frozen_reference" if a == REFERENCE else "factorial_arm", videos=len(names),
        **{m: float(np.mean([index[(n, a)][m] for n in names])) for m in METRICS[:4]}) for a in arms]
    focus = []
    for arm in ARMS[1:]:
        auc = next(p for p in pairs if p["contrast"] == arm + "_minus_student" and p["metric"] == "auc")
        reliable = next(p for p in pairs if p["contrast"] == arm + "_minus_student" and p["metric"] == "student_reliable_auc")
        low = next(p for p in pairs if p["contrast"] == arm + "_minus_student" and p["metric"] == "low_iou_fraction")
        stable = [r for r in stability if r["arm"] == arm and r["reference"] == "student"]
        impacts = [r for r in influence if r["arm"] == arm and r["reference"] == "student"]
        shares = [r["positive_gain_share"] for r in impacts if r["positive_gain_share"] is not None]
        focus.append(dict(arm=arm, auc=next(p["auc"] for p in policies if p["arm"] == arm),
            gain_vs_student=auc["mean_delta"], ci_low=auc["ci_low"], ci_high=auc["ci_high"],
            family_ci_low=auc["family_ci_low"], family_ci_high=auc["family_ci_high"],
            wins=auc["wins"], ties=auc["ties"], losses=auc["losses"], seed_mean_deltas=auc["seed_mean_deltas"],
            low_iou_fraction_delta=low["mean_delta"], reliable_auc_delta=reliable["mean_delta"],
            reliable_supported_videos=reliable["videos"],
            reliable_ci_low=reliable["ci_low"], reliable_ci_high=reliable["ci_high"],
            original_stable_videos=len({r["sequence"] for r in stable if r["original_stable"]}),
            introduced_failure_videos=len({r["sequence"] for r in stable if r["introduced_failure"]}),
            largest_positive_gain_share=max(shares) if shares else None,
            minimum_leave_one_video_out_gain=min(r["leave_one_video_out_mean_delta"] for r in impacts) if len(names) > 1 else None))
    return dict(policy_summary=policies, focus=focus, seed_video_summary=rows, video_summary=averaged,
        paired=pairs, factor_summary=factors, stability=stability, video_influence=influence), residual


def summarize(root):
    root = Path(root).resolve()
    plan = load_plan(root)
    from stage38_geometry_experiment import smoke_gate
    smoke_gate(root, plan)
    run = read_sealed(root / "full/run.json", SCHEMA)
    if (run.get("mode") != "full" or run.get("plan_fingerprint") != plan["fingerprint"] or run.get("methods") != methods("full") or
            run.get("runtime") != plan["source_runtime"] or run.get("window") != plan["window"] or
            run.get("frozen_neural_model_sha256") != plan["source_frozen_neural_model_sha256"]):
        raise ValueError("Mixed full size/offset run")
    expected = {e["sequence"] + "__" + m + ".npz" for e in plan["fit"] for m in methods("full")}
    directory = root / "full/traces"
    if {p.name for p in directory.glob("*.npz")} != expected:
        raise ValueError("Need all 32 NEW full trajectories; use --mode status for progress")
    bindings = dict(plan["source_file_sha256"])
    for path in (root / "manifest.json", root / "full/run.json", root / "smoke/run.json", root / "smoke/stage38_smoke.json"):
        bindings[str(path)] = file_hash(path)
    rows, controls, hashes = [], [], {}
    for entry in plan["fit"]:
        teacher, teacher_path = source_trace(plan, entry, REFERENCE, DESIGN["seeds"][0])
        controls.append(dict(sequence=entry["sequence"], arm=REFERENCE, seed="shared", source_path=str(teacher_path),
            source_sha256=file_hash(teacher_path), tracked_frames=len(teacher["iou"])))
        for seed in DESIGN["seeds"]:
            data_by_arm = {REFERENCE: teacher}
            for arm in ("student", "teacher_geometry"):
                data, path = source_trace(plan, entry, arm, seed)
                data_by_arm[arm] = data
                controls.append(dict(sequence=entry["sequence"], arm=arm, seed=seed, source_path=str(path),
                    source_sha256=file_hash(path), tracked_frames=len(data["iou"])))
            for arm in NEW_ARMS:
                method = arm + "_seed" + str(seed)
                path = directory / (entry["sequence"] + "__" + method + ".npz")
                data_by_arm[arm] = load_trace(path, plan, run, entry, method)
                hashes[path.name] = file_hash(path)
                bindings[str(path)] = hashes[path.name]
                bindings[str(path.with_suffix(".json"))] = file_hash(path.with_suffix(".json"))
            for arm in ARMS + (REFERENCE,):
                data = data_by_arm[arm]
                rows.append(dict(sequence=entry["sequence"], seed=seed, arm=arm, tracked_frames=len(data["iou"]),
                    **metrics(data, data_by_arm["student"], teacher)))
    tables, residual = analyze(rows)
    tables["source_controls"] = controls
    output = root / "analysis"
    with writer_lock(output):
        target = output / "stage38_diagnostic.json"
        if target.exists():
            saved = read_sealed(target, SCHEMA)
            if saved["source_file_sha256"] != bindings:
                raise ValueError("Completed Stage-38 summary source changed")
            verify_bindings({str(output / n): d for n, d in saved["table_sha256"].items()})
            print("Stage-38: COMPLETE_FROZEN_SIZE_OFFSET_DIAGNOSTIC (reuse, audited)")
            return saved
        hashes_of_tables = {}
        for label, values in tables.items():
            path = output / ("stage38_" + label + ".csv")
            write_table(path, values)
            hashes_of_tables[path.name] = file_hash(path)
        verify_bindings(bindings)
        report = seal(dict(schema=SCHEMA, role="consumed_development_fixed_response_size_offset_closed_loop",
            status="COMPLETE_FROZEN_SIZE_OFFSET_DIAGNOSTIC", plan_fingerprint=plan["fingerprint"],
            run_fingerprint=run["fingerprint"], videos=len(plan["fit"]), seeds=DESIGN["seeds"],
            new_full_trajectories=32, reused_endpoint_trajectories=32, shared_reference_trajectories=8,
            focus=tables["focus"], factorial_maximum_residual=residual, source_file_sha256=bindings,
            cache_sha256=hashes, table_sha256=hashes_of_tables, training_performed=False,
            learned_policy_evaluated=False, original_checking_labels_loaded=False, adoption_gate=None,
            warnings=["These eight fit groups and prior geometry outcomes informed this design; not independent external evidence.",
                "Average seeds within video; shared SGLA references are paired reuse, not extra independent videos.",
                "Two co-primary AUC contrasts have additional Bonferroni percentile intervals; all inference remains descriptive and conditional on fixed overlapping OOF fits.",
                "Every arm uses student responses computed on its own crop; score maps are not held numerically equal across trajectories.",
                "Size/offset swaps include subsequent crop feedback; order-average effects are an accounting convention, not unique causal attribution.",
                "Reliable cohorts and failure duration use GT only offline; fewer failure events do not establish shorter failures.",
                "Both frozen models run for diagnostic interventions; no compact learned model or speedup is established."],
            next_step="Interpret size-only, offset-only, interaction and reliable-state damage before selecting compact training"))
        publish_identical(target, report)
    for row in tables["policy_summary"]:
        print("{:<19s} AUC={:.4f} low-IoU={:.2%}".format(row["arm"], row["auc"], row["low_iou_fraction"]))
    print("arm                 gain_student       paired_95%       reliable_delta  new_failure_videos")
    for row in tables["focus"]:
        reliable = "missing" if row["reliable_auc_delta"] is None else "{:+.4f}".format(row["reliable_auc_delta"])
        print("{:<19s} {:+.4f} [{:+.4f},{:+.4f}] {} {}".format(row["arm"], row["gain_vs_student"],
            row["ci_low"], row["ci_high"], reliable, row["introduced_failure_videos"]))
    print("Stage-38: COMPLETE_FROZEN_SIZE_OFFSET_DIAGNOSTIC; no new-method adoption gate")
    print("Reports: " + str(output))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True)
    args = parser.parse_args()
    summarize(Path(args.input).expanduser())


if __name__ == "__main__":
    main()
