"""Audit and summarize the frozen Stage-10 closed-loop transfer diagnostic."""

import argparse
import json
from pathlib import Path

import numpy as np

from stage4_common import fingerprint, write_csv, write_json
from stage6_switched_compare import _load_run, compare
from stage10_closed_loop_transfer import TRANSFER_POLICIES


EXPECTED_SCHEMA = "stage10_closed_loop_transfer_v1"


def load_contract(path, expected_policy):
    directory = Path(path).expanduser().resolve()
    run_path = directory / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError("Stage-10 run metadata missing: {}".format(run_path))
    run = json.loads(run_path.read_text(encoding="utf-8"))
    digest = run.get("fingerprint")
    body = {key: value for key, value in run.items() if key != "fingerprint"}
    if fingerprint(body) != digest:
        raise ValueError("Stage-10 run fingerprint mismatch under {}".format(directory))
    if run.get("schema") != EXPECTED_SCHEMA:
        raise ValueError("Unexpected Stage-10 schema under {}".format(directory))
    if run.get("policy") != expected_policy:
        raise ValueError(
            "Expected policy {} under {}, got {}".format(
                expected_policy, directory, run.get("policy")
            )
        )
    return run


def audit_run_coverage(run, contract, expected_policy):
    if run["method"] != expected_policy:
        raise ValueError(
            "CSV policy under {} is {}, expected {}".format(
                run["path"], run["method"], expected_policy
            )
        )
    expected = set(contract.get("expected_sequences", []))
    if not expected:
        raise ValueError("Stage-10 run contract has no expected sequence list")
    actual = {
        sequence
        for dataset, sequence in run["grouped"]
        if dataset == contract["dataset"]
    }
    foreign = {
        dataset
        for dataset, _ in run["grouped"]
        if dataset != contract["dataset"]
    }
    if foreign or actual != expected:
        raise ValueError(
            "Incomplete or foreign Stage-10 coverage for {}: expected {}, got {}".format(
                expected_policy, len(expected), len(actual)
            )
        )


def audit_contracts(contracts):
    shared_fields = (
        "analysis_role",
        "policy_origin",
        "manifest_fingerprint",
        "manifest_purpose",
        "dataset",
        "split",
        "sequence_filter",
        "expected_sequences",
        "tracker_name",
        "tracker_parameter",
        "checkpoint_sha256",
        "latency_warmup_frames",
        "iou_protocol",
        "reporting_boundary_fraction",
        "code_sha256",
    )
    reference = contracts["fast"]
    for policy in TRANSFER_POLICIES[1:]:
        candidate = contracts[policy]
        changed = [
            field
            for field in shared_fields
            if candidate.get(field) != reference.get(field)
        ]
        if changed:
            raise ValueError(
                "Stage-10 run contracts differ for {}: {}".format(
                    policy, ", ".join(changed)
                )
            )
    return reference


def mean_latency_ms(run):
    values = [
        row["model_decode_latency_ms"]
        for rows in run["grouped"].values()
        for row in rows
        if row["model_decode_latency_ms"] is not None
    ]
    return float(np.mean(values)) if values else float("nan")


def add_reference_metrics(summaries, runs):
    by_method = {row["method"]: row for row in summaries}
    fast_auc = by_method["fast"]["auc"]
    sgla_auc = by_method["sgla"]["auc"]
    denominator = sgla_auc - fast_auc
    rows = []
    for policy in TRANSFER_POLICIES:
        source = by_method[policy]
        auc = source["auc"]
        recovery = (
            (auc - fast_auc) / denominator
            if abs(denominator) > 1e-12
            else float("nan")
        )
        rows.append(
            {
                **source,
                "gain_vs_fast": auc - fast_auc,
                "gain_vs_sgla": auc - sgla_auc,
                "recovery_of_sgla_gain": recovery,
                "mean_model_decode_latency_ms": mean_latency_ms(runs[policy]),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", required=True)
    parser.add_argument("--periodic", required=True)
    parser.add_argument("--sgla", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2036)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap_samples must be positive")

    paths = {
        "fast": args.fast,
        "periodic_l9_20": args.periodic,
        "sgla": args.sgla,
    }
    contracts = {
        policy: load_contract(path, policy) for policy, path in paths.items()
    }
    shared = audit_contracts(contracts)
    runs = {policy: _load_run(path) for policy, path in paths.items()}
    for policy in TRANSFER_POLICIES:
        audit_run_coverage(runs[policy], contracts[policy], policy)
    summaries, paired = compare(
        runs["periodic_l9_20"],
        [runs["fast"], runs["sgla"]],
        args.bootstrap_samples,
        args.seed,
        tolerance=0.0,
        check_prefix=False,
    )
    summaries = add_reference_metrics(summaries, runs)

    output_dir = Path(args.output_dir).expanduser().resolve()
    summary_path = output_dir / "stage10_closed_loop_summary.csv"
    paired_path = output_dir / "stage10_closed_loop_paired.csv"
    metadata_path = output_dir / "stage10_closed_loop_analysis.json"
    write_csv(summary_path, summaries)
    write_csv(paired_path, paired)
    write_json(
        metadata_path,
        {
            "schema": "stage10_closed_loop_analysis_v1",
            "analysis_role": "descriptive_post_stage9_transfer_diagnostic",
            "manifest_fingerprint": shared["manifest_fingerprint"],
            "split": shared["split"],
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "policies": list(TRANSFER_POLICIES),
            "claim_boundary": (
                "The Stage-9 clips were already inspected; this result tests "
                "closed-loop transfer and is not an independent confirmation."
            ),
        },
    )

    print(
        "Audited matched Stage-10 {} runs on manifest {}.".format(
            shared["split"], shared["manifest_fingerprint"]
        )
    )
    print("Result status: descriptive post-Stage-9 transfer diagnostic")
    print(
        "policy             blocks/f       auc  gain_fast  gain_sgla  recovery  latency_ms"
    )
    for row in summaries:
        print(
            "{method:<18s} {candidate_blocks_per_full_frame:>8.2%}  "
            "{auc:>8.4f}  {gain_vs_fast:>9.4f}  {gain_vs_sgla:>9.4f}  "
            "{recovery_of_sgla_gain:>8.2%}  "
            "{mean_model_decode_latency_ms:>10.3f}".format(**row)
        )
    print("\nPaired periodic_l9_20 minus reference:")
    print("reference                  mean_delta  paired_95%           W/T/L")
    for row in paired:
        interval = "[{:.4f},{:.4f}]".format(row["ci_low"], row["ci_high"])
        result = "{}/{}/{}".format(row["wins"], row["ties"], row["losses"])
        print(
            "{control:<26s} {mean_delta:>10.4f}  {interval:<19s} {result}".format(
                interval=interval, result=result, **row
            )
        )
    print("Summary CSV: {}".format(summary_path))
    print("Paired CSV: {}".format(paired_path))
    print("Analysis metadata: {}".format(metadata_path))


if __name__ == "__main__":
    main()
