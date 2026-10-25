"""Matched full-video checking gate for the frozen GT-free identity baseline."""

import argparse
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_json
from stage27_uav_common import failure_events, nonempty_path, seal, success_auc
from stage27_uav_summary import interval, write_table
from stage31_identity_common import DESIGN, METHODS, SCHEMA, load_contract, load_sealed
from stage31_identity_engine import load_video
from stage31_identity_fit import load_model


def video_metrics(data):
    times = data["elapsed_ms"][5:] if len(data["elapsed_ms"]) > 5 else data["elapsed_ms"]
    return dict(auc=success_auc(data["iou"]), mean_iou=float(data["iou"].mean()),
        low_iou_fraction=float((data["iou"] < .2).mean()), failure_events=len(failure_events(data["iou"])),
        tracked_frames=len(data["iou"]), alternative_frames=int((data["chosen"] > 0).sum()),
        alternative_ratio=float((data["chosen"] > 0).mean()),
        mean_crop_model_decode_association_ms=float(times.mean()),
        p95_crop_model_decode_association_ms=float(np.percentile(times, 95)))


def decide(per_video, pairs, complete):
    by_name = {}
    for row in per_video:
        by_name.setdefault(row["sequence"], {})[row["method"]] = row
    complete_videos = [v for v in by_name.values() if set(v) == set(METHODS)]
    stable = [v for v in complete_videos if v["sgla"]["failure_events"] == 0]
    introduced = sum(v["identity"]["failure_events"] > 0 for v in stable)
    association = [v["identity"] for v in complete_videos]
    ratio = (sum(v["alternative_frames"] for v in association) / max(1, sum(v["tracked_frames"] for v in association)))
    used_videos = sum(v["alternative_frames"] > 0 for v in association)
    primary = next((p for p in pairs if p["right"] == "sgla"), None)
    primary_pass = primary is not None and primary["ci_low"] is not None and primary["ci_low"] > 0
    control_pass = len(pairs) == len(METHODS) - 1 and all(p["mean_delta"] > 0 for p in pairs if p["right"] != "sgla")
    low_reduction = float(np.mean([v["sgla"]["low_iou_fraction"] - v["identity"]["low_iou_fraction"]
                                   for v in complete_videos])) if complete_videos else None
    guard = bool(stable) and introduced == 0
    usage = ratio >= DESIGN["minimum_alternative_ratio"] and used_videos >= DESIGN["minimum_alternative_videos"]
    passed = complete and primary_pass and control_pass and guard and usage and low_reduction >= 0
    return dict(status="PILOT_INCOMPLETE" if not complete else "PASS_GT_FREE_IDENTITY_PILOT" if passed else "FAIL_GT_FREE_IDENTITY_PILOT",
        pilot_pass=bool(passed), primary_paired_auc_ci_pass=bool(primary_pass),
        beats_motion_appearance_static_in_mean=bool(control_pass),
        low_iou_fraction_reduction=low_reduction, original_stable_videos=len(stable),
        stable_videos_with_introduced_failure=introduced, stable_video_guard_pass=guard,
        alternative_ratio=ratio, alternative_videos=int(used_videos), nontrivial_usage_pass=bool(usage))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True)
    parser.add_argument("--allow_incomplete", action="store_true")
    args = parser.parse_args()
    directory = Path(args.input).expanduser().resolve()
    contract = load_contract(directory / "manifest.json")
    model = load_model(directory, contract)
    run = load_sealed(directory / "checking_run.json")
    if (run.get("role") != "frozen_full_video_identity_checking" or run.get("contract_fingerprint") != contract["fingerprint"]
            or run.get("model_fingerprint") != model["fingerprint"] or run.get("methods") != list(METHODS)):
        raise ValueError("Checking run does not match frozen model/contract")
    entries = [e for e in contract["sequences"] if e["split"] == "checking"]
    expected = {e["sequence"] + "__" + method + ".npz" for e in entries for method in METHODS}
    extras = {p.name for p in (directory / "checking").glob("*.npz")} - expected
    if extras:
        raise ValueError("Unexpected checking files: " + str(extras))
    rows, missing, hashes = [], [], {}
    for entry in entries:
        for method in METHODS:
            path = directory / "checking" / (entry["sequence"] + "__" + method + ".npz")
            if not path.is_file():
                missing.append(path.name)
                continue
            data = load_video(path, contract, entry, method, run, model)
            rows.append(dict(sequence=entry["sequence"], source_group_proxy=entry["source_group_proxy"],
                             method=method, **video_metrics(data)))
            hashes[path.name] = file_hash(path)
    if not rows or (missing and not args.allow_incomplete):
        raise ValueError("{} matched policy files missing; incomplete status only: --allow_incomplete".format(len(missing)))
    grouped = {}
    for row in rows:
        grouped.setdefault(row["sequence"], {})[row["method"]] = row
    paired_rows, pairs = [], []
    rng = np.random.default_rng(2097)
    for right in (m for m in METHODS if m != "identity"):
        delta = []
        for name, methods in grouped.items():
            if "identity" in methods and right in methods:
                d = methods["identity"]["auc"] - methods[right]["auc"]
                delta.append(d)
                paired_rows.append(dict(sequence=name, left="identity", right=right, auc_delta=d))
        if delta:
            low, high = interval(delta, rng, DESIGN["bootstrap_samples"])
            pairs.append(dict(left="identity", right=right, videos=len(delta), mean_delta=float(np.mean(delta)),
                median_delta=float(np.median(delta)), ci_low=low, ci_high=high,
                wins=int((np.asarray(delta) > 1e-9).sum()), ties=int((np.abs(delta) <= 1e-9).sum()),
                losses=int((np.asarray(delta) < -1e-9).sum())))
    policies = []
    for method in METHODS:
        values = [r for r in rows if r["method"] == method]
        if values:
            policies.append(dict(method=method, videos=len(values), macro_auc=float(np.mean([v["auc"] for v in values])),
                macro_low_iou_fraction=float(np.mean([v["low_iou_fraction"] for v in values])),
                alternative_ratio=sum(v["alternative_frames"] for v in values) / sum(v["tracked_frames"] for v in values),
                mean_crop_model_decode_association_ms=float(np.mean([v["mean_crop_model_decode_association_ms"] for v in values]))))
    output = directory / ("analysis_pilot" if args.allow_incomplete else "analysis")
    write_table(output / "stage31_video_summary.csv", rows)
    write_table(output / "stage31_policy_summary.csv", policies)
    write_table(output / "stage31_paired.csv", pairs, ["left", "right", "videos", "mean_delta", "median_delta", "ci_low", "ci_high", "wins", "ties", "losses"])
    write_table(output / "stage31_video_paired.csv", paired_rows, ["sequence", "left", "right", "auc_delta"])
    decision = seal(dict(schema=SCHEMA, role="fresh_group_development_only_identity_feasibility", **decide(rows, pairs, not missing),
        contract_fingerprint=contract["fingerprint"], model_fingerprint=model["fingerprint"],
        checking_run_fingerprint=run["fingerprint"], missing_files=missing, cache_sha256=hashes,
        checking_videos=len(entries), excluded_diagnostic_groups=len(contract["excluded_groups"]),
        primary_rule="Identity-SGLA paired video-bootstrap 95% lower bound>0; positive mean vs motion, appearance and equally trained static ranker; no new failures on original-stable videos; nontrivial use; no mean low-IoU regression",
        statistical_role="Six source-group proxy videos; pilot feasibility, not confirmatory external generalization",
        bootstrap_samples=DESIGN["bootstrap_samples"], bootstrap_seed=2097,
        analysis_code_sha256=file_hash(Path(__file__)),
        warnings=["Candidate association is established prior work; this pilot establishes no novel compression method.",
            "The original backbone and prediction head are unchanged. No model compression or speedup is claimed.",
            "Fit states were produced by native SGLA; closed-loop checking tests the resulting state-distribution shift.",
            "Timing is instrumented crop/model/decode/association wall time, excluding image reads and label logs.",
            "Checking is consumed after this run; do not adjust thresholds or train on its outcomes."],
        next_step="Preserve the frozen result; establish GT-free association value before designing relation-preserving compression"))
    write_json(output / "stage31_decision.json", decision)
    print("method         videos     AUC   low_IoU%  alternative%  instrumented_ms")
    for p in policies:
        print("{:<14s} {:>6d} {:>7.3f} {:>10.2f} {:>13.2f} {:>16.3f}".format(p["method"], p["videos"],
            p["macro_auc"], 100 * p["macro_low_iou_fraction"], 100 * p["alternative_ratio"], p["mean_crop_model_decode_association_ms"]))
    for p in pairs:
        print("identity - {}: {:+.4f} [{}, {}]".format(p["right"], p["mean_delta"], p["ci_low"], p["ci_high"]))
    print("Decision: " + decision["status"])
    print("Reports: " + str(output))


if __name__ == "__main__":
    main()
