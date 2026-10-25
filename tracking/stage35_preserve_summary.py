"""Fixed OOF screen: correction gains must coexist with reliable-state retention."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from stage4_common import file_hash
from stage27_uav_common import failure_events, publish_identical, read_boxes, seal, success_auc
from stage27_uav_summary import interval, write_table
from stage34_identity_experiment import audit_trace
from stage35_preserve_common import ARMS, BASELINES, DESIGN, SCHEMA, load_plan, load_sealed, nonempty_path


def compare_rows(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["sequence"], r["method"])].append(r)
    averaged = []
    fields = ("auc", "low_iou_fraction", "failure_events", "instrumented_ms", "fast_reliable_auc", "teacher_failure_auc")
    for (name, method), values in sorted(grouped.items()):
        result = dict(sequence=name, method=method, seeds=len(values),
            fast_reliable_frames=values[0]["fast_reliable_frames"], teacher_failure_frames=values[0]["teacher_failure_frames"])
        for field in fields:
            result[field] = float(np.mean([v[field] for v in values])) if values[0][field] is not None else None
        averaged.append(result)
    index = {(r["sequence"], r["method"]): r for r in averaged}
    names = sorted({r["sequence"] for r in rows})
    rng = np.random.default_rng(DESIGN["bootstrap_seed"])
    comparisons = []
    specs = [("auc", right) for right in ("supervised", "preserve", "correct", "distill_all_correct", "fast_initial", "sgla")]
    specs += [("fast_reliable_auc", "correct"), ("fast_reliable_auc", "fast_initial"), ("teacher_failure_auc", "preserve")]
    for metric, right in specs:
        supported = [n for n in names if index[(n, "preserve_correct")][metric] is not None]
        delta = [index[(n, "preserve_correct")][metric] - index[(n, right)][metric] for n in supported]
        lo, hi = interval(delta, rng, DESIGN["bootstrap_samples"]) if len(delta) >= 2 else (None, None)
        comparisons.append(dict(contrast="preserve_correct_minus_" + right, metric=metric, videos=len(delta),
            mean_delta=float(np.mean(delta)) if delta else None, ci_low=lo, ci_high=hi,
            worst_video_delta=float(min(delta)) if delta else None,
            wins=sum(d > 1e-9 for d in delta), ties=sum(abs(d) <= 1e-9 for d in delta), losses=sum(d < -1e-9 for d in delta)))
    delta = [index[(n, "preserve_correct")]["auc"] - index[(n, "correct")]["auc"]
        - index[(n, "preserve")]["auc"] + index[(n, "supervised")]["auc"] for n in names]
    lo, hi = interval(delta, rng, DESIGN["bootstrap_samples"])
    comparisons.append(dict(contrast="preservation_by_correction_interaction", metric="auc", videos=len(delta),
        mean_delta=float(np.mean(delta)), ci_low=lo, ci_high=hi, worst_video_delta=float(min(delta)),
        wins=sum(d > 1e-9 for d in delta), ties=sum(abs(d) <= 1e-9 for d in delta), losses=sum(d < -1e-9 for d in delta)))
    return averaged, comparisons


def decide(rows, averaged, comparisons):
    lookup = {(r["metric"], r["contrast"]): r for r in comparisons}
    def contrast(metric, right):
        return lookup[(metric, "preserve_correct_minus_" + right)]
    gain = contrast("auc", "fast_initial")
    preserve = contrast("fast_reliable_auc", "correct")
    correct = contrast("teacher_failure_auc", "preserve")
    retention = contrast("fast_reliable_auc", "fast_initial")
    unconditional = contrast("auc", "distill_all_correct")
    positive_ci = lambda r: r["ci_low"] is not None and r["ci_low"] > 0
    enough = lambda r: r["videos"] >= DESIGN["minimum_eval_cohort_videos"]
    index = {(r["sequence"], r["method"]): r for r in averaged}
    names = sorted({r["sequence"] for r in rows})
    stable = [n for n in names if index[(n, "fast_initial")]["failure_events"] == 0]
    introduced = sum(any(r["failure_events"] > 0 for r in rows if r["sequence"] == n and r["method"] == "preserve_correct") for n in stable)
    low_delta = float(np.mean([index[(n, "preserve_correct")]["low_iou_fraction"] - index[(n, "fast_initial")]["low_iou_fraction"] for n in names]))
    seed_gains = []
    for seed in DESIGN["seeds"]:
        seed_gains.append(float(np.mean([next(r["auc"] for r in rows if r["sequence"] == n and r["method"] == "preserve_correct" and r["seed"] == seed)
            - index[(n, "fast_initial")]["auc"] for n in names])))
    support = enough(preserve) and enough(correct) and enough(retention) and bool(stable)
    mechanism = bool(support and positive_ci(preserve) and positive_ci(correct) and positive_ci(gain)
        and gain["mean_delta"] >= DESIGN["minimum_auc_gain"] and all(g > 0 for g in seed_gains)
        and contrast("auc", "supervised")["mean_delta"] > 0 and positive_ci(unconditional))
    # Noninferiority is a practical tolerance, NOT a proof of exact preservation.
    guard = bool(support and retention["ci_low"] is not None and
        retention["ci_low"] >= DESIGN["reliable_macro_noninferiority"] and
        retention["worst_video_delta"] >= DESIGN["reliable_worst_video_floor"] and
        introduced == 0 and low_delta <= 0)
    passed = mechanism and guard
    return dict(status=("INSUFFICIENT_OOF_SUPPORT" if not support else
        "PASS_PRESERVE_CORRECT_OOF_SCREEN" if passed else "NO_PRESERVE_CORRECT_GAIN_ESTABLISHED"),
        pilot_pass=passed, support_pass=support, mechanism_pass=mechanism, retention_guard_pass=guard,
        overall_vs_fast=gain, preservation_effect=preserve, correction_effect=correct,
        conditional_vs_unconditional_distillation=unconditional,
        reliable_frame_retention=retention, primary_seed_gains_vs_fast=seed_gains,
        low_iou_fraction_delta_vs_fast=low_delta, fast_stable_videos=len(stable),
        introduced_failure_videos=introduced), passed


def summarize(root):
    root = Path(root).resolve()
    plan = load_plan(root)
    directory = root / "oof"
    run = load_sealed(directory / "run.json")
    methods = list(BASELINES) + sorted(a + "_seed" + str(s) for a in ARMS for s in DESIGN["seeds"])
    if run.get("plan_fingerprint") != plan["fingerprint"] or run.get("methods") != methods:
        raise ValueError("OOF run differs from frozen plan")
    for path, digest in run["frozen_model_sha256"].items():
        if file_hash(path) != digest:
            raise ValueError("Frozen model changed after OOF freeze")
    expected = {e["sequence"] + "__" + m + ".npz" for e in plan["fit"] for m in methods}
    if {p.name for p in directory.glob("*.npz")} != expected:
        raise ValueError("Need all 96 whole-video OOF traces, no partial gain decision")
    rows, hashes = [], {}
    for entry in plan["fit"]:
        name = entry["sequence"]
        annotation = Path(plan["root"]) / entry["annotation"]
        if file_hash(annotation) != entry["annotation_sha256"]:
            raise ValueError("Fit OOF annotations changed")
        gt = read_boxes(annotation)[1:]
        fold = next(f["fold"] for f in plan["folds"] if name in f["held_out"])
        traces = {}
        for method in methods:
            path = directory / (name + "__" + method + ".npz")
            with np.load(path, allow_pickle=False) as a:
                meta = json.loads(str(a["metadata_json"].item()))
                data = {k: a[k] for k in a.files if k != "metadata_json"}
            if meta != dict(schema=SCHEMA, run_fingerprint=run["fingerprint"], sequence=name, fold=fold, method=method):
                raise ValueError("Mixed OOF trace")
            audit_trace(data, entry)
            np.testing.assert_array_equal(data["gt"], gt)
            traces[method] = data
            hashes[path.name] = file_hash(path)
        reliable = traces["fast_initial"]["iou"] >= DESIGN["reliable_iou"]
        failure = traces["sgla"]["iou"] < DESIGN["reliable_iou"]
        for method, data in traces.items():
            arm, seed = method.rsplit("_seed", 1) if "_seed" in method else (method, "baseline")
            def conditional(mask):
                return success_auc(data["iou"][mask]) if int(mask.sum()) >= DESIGN["minimum_eval_cohort_frames"] else None
            timing = data["elapsed_ms"][5:] if len(data["elapsed_ms"]) > 5 else data["elapsed_ms"]
            rows.append(dict(sequence=name, source_group_proxy=entry["source_group_proxy"], fold=fold,
                method=arm, seed=int(seed) if seed != "baseline" else seed,
                tracked_frames=len(data["iou"]), auc=success_auc(data["iou"]),
                low_iou_fraction=float(np.mean(data["iou"] < .2)), failure_events=len(failure_events(data["iou"])),
                instrumented_ms=float(timing.mean()), fast_reliable_frames=int(reliable.sum()),
                teacher_failure_frames=int(failure.sum()), fast_reliable_auc=conditional(reliable), teacher_failure_auc=conditional(failure)))
    averaged, paired = compare_rows(rows)
    decision, passed = decide(rows, averaged, paired)
    policies = []
    for method in BASELINES + ARMS:
        values = [r for r in averaged if r["method"] == method]
        policies.append(dict(method=method, videos=len(values), auc=float(np.mean([r["auc"] for r in values])),
            low_iou_fraction=float(np.mean([r["low_iou_fraction"] for r in values])),
            instrumented_ms=float(np.mean([r["instrumented_ms"] for r in values]))))
    output = root / "analysis"
    report = seal(dict(schema=SCHEMA, role="consumed_development_fit_only_group_oof_screen", **decision,
        plan_fingerprint=plan["fingerprint"], oof_run_fingerprint=run["fingerprint"], videos=len(plan["fit"]),
        seeds=DESIGN["seeds"], cache_sha256=hashes, original_checking_labels_loaded=False,
        next_step=("Freeze this screen and predeclare a genuinely new checking study before any claim" if passed else
            "Preserve this result; no loss-weight, fold, step or checkpoint search on these OOF outcomes"),
        warnings=["These eight fit groups have informed method design; OOF is development evidence, not external validation.",
            "Seeds are averaged within video before video bootstrap; seeds/frames are not independent videos.",
            "Folds have overlapping training sets. Nominal conditional-on-fit bootstrap does not include retraining uncertainty.",
            "Baseline-defined reliable/failure strata are outcome-conditioned diagnostics, not online triggers.",
            "Only L5-L6 are trained; six physical blocks and no inference auxiliaries do not alone prove speedup.",
            "Conditional KD and hard-negative supervision are hypotheses to test, not an established novel method."]))
    publish_identical(output / "stage35_decision.json", report)
    for label, values in (("seed_video_summary", rows), ("video_summary", averaged), ("paired", paired), ("policy_summary", policies)):
        write_table(output / ("stage35_" + label + ".csv"), values)
    for row in policies:
        print("{:<18s} AUC={:.4f} low-IoU={:.2%}".format(row["method"], row["auc"], row["low_iou_fraction"]))
    print("Stage-35: " + report["status"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True)
    args = parser.parse_args()
    summarize(Path(args.input).expanduser())


if __name__ == "__main__":
    main()
