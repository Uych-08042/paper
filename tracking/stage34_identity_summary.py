"""Matched fresh-group whole-video comparison; preserve the predeclared decision."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_json
from stage27_uav_common import failure_events, read_boxes, seal, success_auc
from stage27_uav_summary import interval, write_table
from stage34_identity_common import ARMS, DESIGN, SCHEMA, load_plan, load_sealed, nonempty_path
from stage34_identity_experiment import audit_trace


def compare_rows(rows):
    by_video = defaultdict(dict)
    for row in rows:
        by_video[row["sequence"]].setdefault(row["method"], []).append(row)
    averaged = []
    for sequence, methods in sorted(by_video.items()):
        for method, values in methods.items():
            averaged.append(dict(sequence=sequence, method=method, seeds=len(values),
                auc=float(np.mean([v["auc"] for v in values])),
                low_iou_fraction=float(np.mean([v["low_iou_fraction"] for v in values])),
                failure_events=float(np.mean([v["failure_events"] for v in values])),
                instrumented_ms=float(np.mean([v["instrumented_ms"] for v in values]))))
    index = {(r["sequence"], r["method"]): r for r in averaged}
    names = sorted(by_video)
    comparisons = []
    rng = np.random.default_rng(DESIGN["bootstrap_seed"])
    for right in ("representation_standard", "readout_contrast", "fast_initial", "sgla"):
        deltas = [index[(name, "representation_contrast")]["auc"] - index[(name, right)]["auc"] for name in names]
        low, high = interval(deltas, rng, DESIGN["bootstrap_samples"])
        comparisons.append(dict(contrast="representation_contrast_minus_" + right, videos=len(names),
            mean_delta=float(np.mean(deltas)), ci_low=low, ci_high=high,
            wins=sum(v > 1e-9 for v in deltas), ties=sum(abs(v) <= 1e-9 for v in deltas), losses=sum(v < -1e-9 for v in deltas)))
    interaction = []
    for name in names:
        value = (index[(name, "representation_contrast")]["auc"] - index[(name, "representation_standard")]["auc"]
                 - index[(name, "readout_contrast")]["auc"] + index[(name, "readout_standard")]["auc"])
        interaction.append(value)
    low, high = interval(interaction, rng, DESIGN["bootstrap_samples"])
    comparisons.append(dict(contrast="representation_by_contrast_interaction", videos=len(names),
        mean_delta=float(np.mean(interaction)), ci_low=low, ci_high=high,
        wins=sum(v > 1e-9 for v in interaction), ties=sum(abs(v) <= 1e-9 for v in interaction), losses=sum(v < -1e-9 for v in interaction)))
    return averaged, comparisons


def decide(rows, averaged, comparisons):
    lookup = {r["contrast"]: r for r in comparisons}
    primary = lookup["representation_contrast_minus_representation_standard"]
    interaction = lookup["representation_by_contrast_interaction"]
    readout = lookup["representation_contrast_minus_readout_contrast"]
    initial = lookup["representation_contrast_minus_fast_initial"]
    by_video = {(r["sequence"], r["method"]): r for r in averaged}
    names = sorted({r["sequence"] for r in averaged})
    low_delta = float(np.mean([by_video[(n, "representation_contrast")]["low_iou_fraction"]
                             - by_video[(n, "fast_initial")]["low_iou_fraction"] for n in names]))
    stable = [n for n in names if by_video[(n, "fast_initial")]["failure_events"] == 0]
    introduced = sum(any(r["failure_events"] > 0 for r in rows if r["sequence"] == n and r["method"] == "representation_contrast") for n in stable)
    seed_lifts = []
    for seed in DESIGN["seeds"]:
        per = {(r["sequence"], r["method"]): r["auc"] for r in rows if r["seed"] == seed}
        seed_lifts.append(float(np.mean([per[(n, "representation_contrast")] - per[(n, "representation_standard")] for n in names])))
    mechanism = bool(primary["mean_delta"] >= DESIGN["minimum_auc_gain"] and primary["ci_low"] is not None
        and primary["ci_low"] > 0 and interaction["ci_low"] is not None and interaction["ci_low"] > 0
        and readout["mean_delta"] > 0 and initial["mean_delta"] >= DESIGN["minimum_auc_gain"]
        and initial["ci_low"] is not None and initial["ci_low"] > 0 and all(v > 0 for v in seed_lifts))
    guard = bool(stable) and introduced == 0 and low_delta <= 0
    return dict(status="PASS_SPATIAL_REPRESENTATION_PILOT" if mechanism and guard else "NO_SPATIAL_REPRESENTATION_GAIN_ESTABLISHED",
        mechanism_pass=mechanism, guard_pass=guard, primary=primary, interaction=interaction, initial_comparison=initial,
        primary_seed_lifts=seed_lifts, low_iou_fraction_delta_vs_initial=low_delta,
        initial_stable_videos=len(stable), introduced_failures_on_initial_stable=introduced)


def summarize(root):
    root = Path(root).resolve()
    plan = load_plan(root)
    directory = root / "checking"
    run = load_sealed(directory / "run.json")
    expected_methods = ["sgla", "fast_initial"] + sorted(arm + "_seed" + str(seed) for arm in ARMS for seed in DESIGN["seeds"])
    if run.get("plan_fingerprint") != plan["fingerprint"] or run.get("methods") != expected_methods:
        raise ValueError("Checking run differs from frozen plan")
    for path, digest in run["frozen_model_sha256"].items():
        if file_hash(path) != digest:
            raise ValueError("Trained model changed after checking freeze")
    expected = {e["sequence"] + "__" + m + ".npz" for e in plan["checking"] for m in expected_methods}
    if {p.name for p in directory.glob("*.npz")} != expected:
        raise ValueError("Need all 84 matched full-video traces; no incomplete method decision")
    rows, hashes = [], {}
    for entry in plan["checking"]:
        annotation = Path(plan["root"]) / entry["annotation"]
        if file_hash(annotation) != entry["annotation_sha256"]:
            raise ValueError("Fresh checking annotations changed")
        gt = read_boxes(annotation)[1:]
        for method in expected_methods:
            path = directory / (entry["sequence"] + "__" + method + ".npz")
            with np.load(path, allow_pickle=False) as a:
                meta = json.loads(str(a["metadata_json"].item()))
                data = {k: a[k] for k in a.files if k != "metadata_json"}
            expected_meta = dict(schema=SCHEMA, run_fingerprint=run["fingerprint"], sequence=entry["sequence"], method=method)
            if meta != expected_meta:
                raise ValueError("Mixed checking trace")
            audit_trace(data, entry)
            np.testing.assert_array_equal(data["gt"], gt)
            timing = data["elapsed_ms"][5:] if len(data["elapsed_ms"]) > 5 else data["elapsed_ms"]
            arm, seed = method.rsplit("_seed", 1) if "_seed" in method else (method, "baseline")
            rows.append(dict(sequence=entry["sequence"], source_group_proxy=entry["source_group_proxy"], method=arm,
                seed=int(seed) if seed != "baseline" else seed, auc=success_auc(data["iou"]),
                low_iou_fraction=float(np.mean(data["iou"] < .2)), failure_events=len(failure_events(data["iou"])),
                instrumented_ms=float(timing.mean()), tracked_frames=len(data["iou"])))
            hashes[path.name] = file_hash(path)
    averaged, comparisons = compare_rows(rows)
    decision = decide(rows, averaged, comparisons)
    policies = []
    for method in ("sgla", "fast_initial") + ARMS:
        values = [r for r in averaged if r["method"] == method]
        policies.append(dict(method=method, videos=len(values), auc=float(np.mean([v["auc"] for v in values])),
            low_iou_fraction=float(np.mean([v["low_iou_fraction"] for v in values])),
            instrumented_ms=float(np.mean([v["instrumented_ms"] for v in values]))))
    output = root / "analysis"
    for name, values in (("stage34_seed_video_summary", rows), ("stage34_video_summary", averaged),
                         ("stage34_policy_summary", policies), ("stage34_paired", comparisons)):
        write_table(output / (name + ".csv"), values)
    report = seal(dict(schema=SCHEMA, role="fresh_group_development_matched_representation_pilot", **decision,
        plan_fingerprint=plan["fingerprint"], checking_run_fingerprint=run["fingerprint"],
        videos=len(plan["checking"]), seeds=DESIGN["seeds"], cache_sha256=hashes,
        warning="Fresh source-group proxy development check; not an untouched external benchmark or deployment speed claim",
        next_step="Preserve the fixed result; no rate/layer/loss-weight/checkpoint search on these checking outcomes"))
    write_json(output / "stage34_decision.json", report)
    for row in policies:
        print("{:<26s} AUC={:.4f} low-IoU={:.2%}".format(row["method"], row["auc"], row["low_iou_fraction"]))
    print("Stage-34: " + report["status"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True)
    args = parser.parse_args()
    summarize(Path(args.input).expanduser())


if __name__ == "__main__":
    main()
