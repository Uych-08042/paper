"""Audit and decide the frozen Stage-11 OTB external validation."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from stage3_closed_loop_summary import load_closed_loop
from stage4_common import fingerprint, write_csv, write_json
from stage6_temporal_stability import segment_auc
from stage11_external_validate import (
    DATASET_NAME,
    EXPECTED_PHYSICAL_VIDEOS,
    EXPECTED_SEQUENCE_OBJECTS,
    FROZEN_POLICIES,
    NONINFERIORITY_MARGIN_AUC,
    PRIMARY_POLICY,
    SCHEMA,
    SECONDARY_POLICY,
    expected_candidate_blocks,
)


def load_contract(input_dir):
    input_dir = Path(input_dir).expanduser().resolve()
    path = input_dir / "run.json"
    if not path.is_file():
        raise FileNotFoundError("Stage-11 run metadata missing: {}".format(path))
    run = json.loads(path.read_text(encoding="utf-8"))
    digest = run.get("fingerprint")
    body = {key: value for key, value in run.items() if key != "fingerprint"}
    if fingerprint(body) != digest:
        raise ValueError("Stage-11 run fingerprint mismatch")
    if run.get("schema") != SCHEMA:
        raise ValueError("Unexpected Stage-11 schema: {}".format(run.get("schema")))
    if tuple(run.get("policies", ())) != FROZEN_POLICIES:
        raise ValueError("Stage-11 policy set or order is not frozen")
    if run.get("dataset") != DATASET_NAME:
        raise ValueError("Stage-11 requires the OTB dataset")
    expected_sequences = run.get("expected_sequences", {})
    sequence_clusters = run.get("sequence_clusters", {})
    if (
        run.get("expected_sequence_objects") != EXPECTED_SEQUENCE_OBJECTS
        or len(expected_sequences) != EXPECTED_SEQUENCE_OBJECTS
    ):
        raise ValueError("Stage-11 requires all 100 OTB sequence objects")
    if (
        set(sequence_clusters) != set(expected_sequences)
        or len(set(sequence_clusters.values())) != EXPECTED_PHYSICAL_VIDEOS
    ):
        raise ValueError("Stage-11 requires the 98 physical-video OTB clusters")
    if run.get("primary_policy") != PRIMARY_POLICY:
        raise ValueError("Stage-11 primary policy changed")
    if run.get("secondary_policy") != SECONDARY_POLICY:
        raise ValueError("Stage-11 secondary policy changed")
    if float(run.get("noninferiority_margin_auc")) != NONINFERIORITY_MARGIN_AUC:
        raise ValueError("Stage-11 non-inferiority margin changed")
    return input_dir, run


def audit_rows(paths, rows, run):
    expected_frames = {
        name: int(frames) for name, frames in run["expected_sequences"].items()
    }
    expected_keys = {
        (DATASET_NAME, name, policy)
        for name in expected_frames
        for policy in FROZEN_POLICIES
    }
    expected_files = len(expected_keys)
    if len(paths) != expected_files:
        raise ValueError(
            "Stage-11 is incomplete: expected {} CSV files, got {}".format(
                expected_files, len(paths)
            )
        )

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["sequence"], row["policy"])].append(row)
    actual_keys = set(grouped)
    if actual_keys != expected_keys:
        missing = expected_keys.difference(actual_keys)
        extra = actual_keys.difference(expected_keys)
        raise ValueError(
            "Stage-11 coverage mismatch: {} missing, {} extra groups".format(
                len(missing), len(extra)
            )
        )

    warmup = int(run["latency_warmup_frames"])
    audited = {}
    for key in sorted(expected_keys):
        _, sequence, policy = key
        policy_rows = sorted(grouped[key], key=lambda row: row["frame_id"])
        frame_count = expected_frames[sequence]
        frame_ids = [row["frame_id"] for row in policy_rows]
        if frame_ids != list(range(frame_count)):
            raise ValueError(
                "Frame coverage mismatch for {}/{}".format(sequence, policy)
            )
        tracked = policy_rows[1:]
        actual_blocks = int(sum(row["candidate_blocks"] for row in tracked))
        expected_blocks = expected_candidate_blocks(frame_count - 1, policy)
        if actual_blocks != expected_blocks:
            raise ValueError(
                "Candidate cost mismatch for {}/{}: {} != {}".format(
                    sequence, policy, actual_blocks, expected_blocks
                )
            )
        latency_count = sum(
            row["model_decode_latency_ms"] is not None for row in tracked
        )
        expected_latency_count = max((frame_count - 1) - warmup, 0)
        if latency_count != expected_latency_count:
            raise ValueError(
                "Latency coverage mismatch for {}/{}: {} != {}".format(
                    sequence, policy, latency_count, expected_latency_count
                )
            )
        audited[key] = policy_rows
    return audited


def summarize_policies(grouped, run):
    names = list(run["expected_sequences"])
    sequence_auc = {policy: {} for policy in FROZEN_POLICIES}
    rows = []
    for policy in FROZEN_POLICIES:
        policy_rows = [
            row
            for name in names
            for row in grouped[(DATASET_NAME, name, policy)]
        ]
        tracked = [row for row in policy_rows if row["frame_id"] > 0]
        for name in names:
            sequence_auc[policy][name] = segment_auc(
                [
                    row["iou"]
                    for row in grouped[(DATASET_NAME, name, policy)]
                ]
            )
        latencies = np.asarray(
            [
                row["model_decode_latency_ms"]
                for row in tracked
                if row["model_decode_latency_ms"] is not None
            ],
            dtype=np.float64,
        )
        valid_ious = np.asarray(
            [row["iou"] for row in policy_rows if row["gt_valid"] == 1],
            dtype=np.float64,
        )
        blocks = int(sum(row["candidate_blocks"] for row in tracked))
        rows.append(
            {
                "policy": policy,
                "num_sequences": len(names),
                "frames": len(policy_rows),
                "tracked_frames": len(tracked),
                "candidate_blocks": blocks,
                "candidate_blocks_per_frame": blocks / len(tracked),
                "auc": float(np.mean(list(sequence_auc[policy].values()))),
                "mean_iou_valid": float(valid_ious.mean()),
                "mean_latency_ms": float(latencies.mean()),
                "median_latency_ms": float(np.median(latencies)),
                "p95_latency_ms": float(np.percentile(latencies, 95.0)),
                "fps_from_mean_latency": 1000.0 / float(latencies.mean()),
            }
        )

    by_policy = {row["policy"]: row for row in rows}
    fast_auc = by_policy["fast"]["auc"]
    sgla_auc = by_policy["sgla"]["auc"]
    sgla_latency = by_policy["sgla"]["mean_latency_ms"]
    for row in rows:
        row["gain_vs_fast"] = row["auc"] - fast_auc
        row["gain_vs_sgla"] = row["auc"] - sgla_auc
        row["recovery_of_sgla_gain"] = ""
        row["latency_reduction_vs_sgla"] = (
            (sgla_latency - row["mean_latency_ms"]) / sgla_latency
        )
    return rows, sequence_auc


def paired_contrast(
    contrast,
    left,
    right,
    sequence_auc,
    names,
    sequence_clusters,
    bootstrap_samples,
    seed,
    role,
    threshold,
):
    delta = np.asarray(
        [sequence_auc[left][name] - sequence_auc[right][name] for name in names],
        dtype=np.float64,
    )
    cluster_order = list(dict.fromkeys(sequence_clusters[name] for name in names))
    cluster_sums = np.asarray(
        [
            delta[
                [
                    index
                    for index, name in enumerate(names)
                    if sequence_clusters[name] == cluster
                ]
            ].sum()
            for cluster in cluster_order
        ],
        dtype=np.float64,
    )
    cluster_counts = np.asarray(
        [
            sum(sequence_clusters[name] == cluster for name in names)
            for cluster in cluster_order
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(cluster_order),
        size=(bootstrap_samples, len(cluster_order)),
    )
    means = (
        cluster_sums[indices].sum(axis=1)
        / cluster_counts[indices].sum(axis=1)
    )
    return {
        "contrast": contrast,
        "role": role,
        "left_policy": left,
        "right_policy": right,
        "num_sequences": len(delta),
        "num_physical_video_clusters": len(cluster_order),
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "wins": int(np.sum(delta > 1e-12)),
        "ties": int(np.sum(np.abs(delta) <= 1e-12)),
        "losses": int(np.sum(delta < -1e-12)),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "decision_threshold": threshold,
        "probability_above_threshold": float(np.mean(means > threshold)),
        "passes_predeclared_threshold": int(
            float(np.percentile(means, 2.5)) > threshold
        ),
    }


def build_contrasts(
    sequence_auc,
    names,
    sequence_clusters,
    bootstrap_samples,
    seed,
):
    specifications = (
        (
            "sgla_minus_fast",
            "sgla",
            "fast",
            "external_headroom_diagnostic",
            0.0,
        ),
        (
            "p30_minus_fast",
            PRIMARY_POLICY,
            "fast",
            "primary_superiority",
            0.0,
        ),
        (
            "p30_minus_sgla",
            PRIMARY_POLICY,
            "sgla",
            "primary_noninferiority",
            -NONINFERIORITY_MARGIN_AUC,
        ),
        (
            "p20_minus_fast",
            SECONDARY_POLICY,
            "fast",
            "secondary_descriptive",
            0.0,
        ),
        (
            "p20_minus_sgla",
            SECONDARY_POLICY,
            "sgla",
            "secondary_descriptive",
            -NONINFERIORITY_MARGIN_AUC,
        ),
        (
            "p30_minus_p20",
            PRIMARY_POLICY,
            SECONDARY_POLICY,
            "descriptive",
            0.0,
        ),
    )
    return [
        paired_contrast(
            contrast,
            left,
            right,
            sequence_auc,
            names,
            sequence_clusters,
            bootstrap_samples,
            seed + index,
            role,
            threshold,
        )
        for index, (contrast, left, right, role, threshold) in enumerate(
            specifications
        )
    ]


def apply_recovery(policy_rows, contrast_rows):
    policies = {row["policy"]: row for row in policy_rows}
    contrasts = {row["contrast"]: row for row in contrast_rows}
    if contrasts["sgla_minus_fast"]["passes_predeclared_threshold"]:
        fast_auc = policies["fast"]["auc"]
        headroom = policies["sgla"]["auc"] - fast_auc
        for row in policy_rows:
            row["recovery_of_sgla_gain"] = (
                (row["auc"] - fast_auc) / headroom
            )
    return policy_rows


def primary_decision(policy_rows, contrast_rows):
    policies = {row["policy"]: row for row in policy_rows}
    contrasts = {row["contrast"]: row for row in contrast_rows}
    p30_superior = bool(
        contrasts["p30_minus_fast"]["passes_predeclared_threshold"]
    )
    p30_noninferior = bool(
        contrasts["p30_minus_sgla"]["passes_predeclared_threshold"]
    )
    headroom_confirmed = bool(
        contrasts["sgla_minus_fast"]["passes_predeclared_threshold"]
    )
    cost_valid = (
        0.295
        <= policies[PRIMARY_POLICY]["candidate_blocks_per_frame"]
        <= 0.305
    )
    latency_improves = (
        policies[PRIMARY_POLICY]["mean_latency_ms"]
        < policies["sgla"]["mean_latency_ms"]
    )
    if p30_superior and p30_noninferior and cost_valid and latency_improves:
        status = "PASS"
        interpretation = (
            "The frozen P30 policy transfers as an external accuracy-efficiency "
            "operating point. Stop experimentation and write the result."
        )
    elif not headroom_confirmed:
        status = "NO_CONFIRMED_HEADROOM"
        interpretation = (
            "SGLA itself does not establish external improvement over Fast, so "
            "this benchmark cannot confirm recovery of an SGLA gain. Do not tune "
            "the schedule on OTB."
        )
    else:
        status = "FAIL_TRANSFER"
        interpretation = (
            "SGLA has external headroom but the frozen P30 policy does not meet "
            "the predeclared Pareto criteria. Do not search another OTB schedule."
        )
    return {
        "status": status,
        "analysis_role": "predeclared_untouched_external_confirmation",
        "primary_rule": (
            "P30-Fast paired 95% CI lower bound > 0; P30-SGLA paired "
            "95% CI lower bound > -0.5 AUC; P30 block ratio is 30%; "
            "P30 measured latency is below SGLA"
        ),
        "p30_superiority_pass": p30_superior,
        "p30_noninferiority_pass": p30_noninferior,
        "external_sgla_headroom_confirmed": headroom_confirmed,
        "candidate_cost_pass": cost_valid,
        "latency_direction_pass": latency_improves,
        "interpretation": interpretation,
        "prohibited_follow_up": (
            "No OTB-driven rate, layer, phase, threshold, or policy search."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2041)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap_samples must be positive")

    input_dir, run = load_contract(args.input)
    _, paths, rows = load_closed_loop(input_dir)
    grouped = audit_rows(paths, rows, run)
    policy_rows, sequence_auc = summarize_policies(grouped, run)
    names = list(run["expected_sequences"])
    contrast_rows = build_contrasts(
        sequence_auc,
        names,
        run["sequence_clusters"],
        args.bootstrap_samples,
        args.seed,
    )
    policy_rows = apply_recovery(policy_rows, contrast_rows)
    decision = primary_decision(policy_rows, contrast_rows)
    decision.update(
        {
            "run_fingerprint": run["fingerprint"],
            "dataset_fingerprint": run["dataset_fingerprint"],
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        }
    )

    by_policy = {row["policy"]: row for row in policy_rows}
    sequence_rows = []
    for name in names:
        for policy in FROZEN_POLICIES:
            sequence_rows.append(
                {
                    "dataset": DATASET_NAME,
                    "sequence": name,
                    "policy": policy,
                    "auc": sequence_auc[policy][name],
                    "delta_vs_fast": (
                        sequence_auc[policy][name]
                        - sequence_auc["fast"][name]
                    ),
                    "delta_vs_sgla": (
                        sequence_auc[policy][name]
                        - sequence_auc["sgla"][name]
                    ),
                }
            )

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_dir / "analysis"
    )
    policy_path = output_dir / "stage11_external_policy_summary.csv"
    contrast_path = output_dir / "stage11_external_paired.csv"
    sequence_path = output_dir / "stage11_external_sequence_summary.csv"
    decision_path = output_dir / "stage11_external_decision.json"
    write_csv(policy_path, policy_rows)
    write_csv(contrast_path, contrast_rows)
    write_csv(sequence_path, sequence_rows)
    write_json(decision_path, decision)

    print(
        "Audited {} OTB sequence objects and {} frozen policy files.".format(
            len(names), len(paths)
        )
    )
    print(
        "policy             blocks/f       auc  gain_fast  gain_sgla  "
        "recovery  latency_ms"
    )
    for policy in FROZEN_POLICIES:
        row = by_policy[policy]
        recovery = row["recovery_of_sgla_gain"]
        recovery_text = (
            "n/a" if recovery == "" else "{:.2%}".format(recovery)
        )
        print(
            "{policy:<18s} {candidate_blocks_per_frame:>8.2%}  "
            "{auc:>8.4f}  {gain_vs_fast:>9.4f}  {gain_vs_sgla:>9.4f}  "
            "{recovery:>8s}  {mean_latency_ms:>10.3f}".format(
                recovery=recovery_text, **row
            )
        )
    print("\nPaired sequence contrasts:")
    print("contrast                 mean_delta  paired_95%           W/T/L  pass")
    for row in contrast_rows:
        interval = "[{:.4f},{:.4f}]".format(row["ci_low"], row["ci_high"])
        result = "{}/{}/{}".format(row["wins"], row["ties"], row["losses"])
        print(
            "{contrast:<24s} {mean_delta:>10.4f}  {interval:<19s} "
            "{result:<11s} {passes_predeclared_threshold}".format(
                interval=interval, result=result, **row
            )
        )
    print("\nPredeclared decision: {}".format(decision["status"]))
    print(decision["interpretation"])
    print("Policy summary CSV: {}".format(policy_path))
    print("Paired CSV: {}".format(contrast_path))
    print("Sequence CSV: {}".format(sequence_path))
    print("Decision JSON: {}".format(decision_path))


if __name__ == "__main__":
    main()
