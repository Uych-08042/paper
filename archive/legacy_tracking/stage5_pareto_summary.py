"""Combine Stage-5 closed-loop accuracy with independent latency profiles."""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

from stage3_periodic import PERIODIC_L9_BUDGETS


POLICY_ORDER = (
    "fast",
    "periodic_l9_10",
    "periodic_l9_20",
    "periodic_l9_30",
    "periodic_l9_50",
    "sgla",
)
REQUIRED_OPERATIONS = ("fast", "fixed_l9", "sgla")


def load_accuracy(path):
    path = Path(path).expanduser().resolve()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = {row["policy"]: row for row in csv.DictReader(handle)}
    missing = set(POLICY_ORDER).difference(rows)
    if missing:
        raise ValueError(
            "Closed-loop summary is missing policies: {}".format(
                ", ".join(sorted(missing))
            )
        )
    return {
        policy: {
            "auc": float(rows[policy]["auc"]),
            "candidate_blocks": float(rows[policy]["candidate_blocks"]),
        }
        for policy in POLICY_ORDER
    }


def aggregate_profiles(profile_dirs):
    totals = defaultdict(float)
    samples = defaultdict(int)
    paths = []
    for value in profile_dirs:
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            path = path / "benchmark_summary.csv"
        if not path.is_file():
            raise FileNotFoundError("Benchmark summary missing: {}".format(path))
        paths.append(path)
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row["scope"] != "integrated":
                    continue
                operation = row["operation"]
                count = int(row["samples"])
                totals[operation] += float(row["mean_ms"]) * count
                samples[operation] += count
    missing = set(REQUIRED_OPERATIONS).difference(samples)
    if missing:
        raise ValueError(
            "Latency profiles are missing integrated operations: {}".format(
                ", ".join(sorted(missing))
            )
        )
    return {
        operation: totals[operation] / samples[operation]
        for operation in samples
    }, paths, dict(samples)


def build_pareto(accuracy, latency):
    fast_auc = accuracy["fast"]["auc"]
    sgla_auc = accuracy["sgla"]["auc"]
    full_gain = sgla_auc - fast_auc
    if full_gain <= 0.0:
        raise ValueError("SGLA AUC must exceed Fast AUC for recovery reporting")

    fast_ms = latency["fast"]
    fixed_l9_ms = latency["fixed_l9"]
    sgla_ms = latency["sgla"]
    if min(fast_ms, fixed_l9_ms, sgla_ms) <= 0.0:
        raise ValueError("Latency means must be positive")

    output = []
    for policy in POLICY_ORDER:
        if policy == "fast":
            budget = 0.0
            latency_ms = fast_ms
            latency_source = "measured_integrated"
        elif policy == "sgla":
            budget = 1.0
            latency_ms = sgla_ms
            latency_source = "measured_integrated"
        else:
            budget = PERIODIC_L9_BUDGETS[policy] / 100.0
            latency_ms = fast_ms + budget * (fixed_l9_ms - fast_ms)
            latency_source = "measured_path_mixture"
        auc = accuracy[policy]["auc"]
        output.append(
            {
                "policy": policy,
                "candidate_blocks_per_frame": accuracy[policy][
                    "candidate_blocks"
                ],
                "auc": auc,
                "gain_vs_fast": auc - fast_auc,
                "gain_vs_sgla": auc - sgla_auc,
                "recovery_of_sgla_gain": (auc - fast_auc) / full_gain,
                "latency_ms": latency_ms,
                "fps": 1000.0 / latency_ms,
                "latency_increase_vs_fast": latency_ms / fast_ms - 1.0,
                "latency_reduction_vs_sgla": 1.0 - latency_ms / sgla_ms,
                "speedup_vs_sgla": sgla_ms / latency_ms,
                "latency_source": latency_source,
            }
        )
    return output


def _format(value):
    if isinstance(value, float):
        return "{:.8f}".format(value)
    return value


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format(value) for key, value in row.items()})
    os.replace(str(temporary), str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closed_loop_summary", required=True)
    parser.add_argument("--profile_dirs", nargs="+", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    accuracy = load_accuracy(args.closed_loop_summary)
    latency, profile_paths, samples = aggregate_profiles(args.profile_dirs)
    rows = build_pareto(accuracy, latency)
    accuracy_path = Path(args.closed_loop_summary).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else accuracy_path.parent / "stage5_periodic_pareto.csv"
    )
    write_csv(output, rows)

    print(
        "Loaded {} latency profile(s), {} integrated samples per required "
        "operation.".format(
            len(profile_paths), min(samples[key] for key in REQUIRED_OPERATIONS)
        )
    )
    print(
        "Measured means: Fast={:.4f} ms, fixed-L9={:.4f} ms, "
        "SGLA={:.4f} ms".format(
            latency["fast"], latency["fixed_l9"], latency["sgla"]
        )
    )
    print(
        "policy             blocks/f      auc  recovery  latency_ms       fps  "
        "latency_save"
    )
    for row in rows:
        print(
            "{policy:<18s} {candidate_blocks_per_frame:>8.2f}  "
            "{auc:>7.4f}  {recovery_of_sgla_gain:>8.2%}  "
            "{latency_ms:>10.4f}  {fps:>8.2f}  "
            "{latency_reduction_vs_sgla:>11.2%}".format(**row)
        )
    if "native_sgla" in latency:
        print("Native SGLA diagnostic: {:.4f} ms".format(latency["native_sgla"]))
    if "fast_then_sgla" in latency:
        print(
            "Fast-then-SGLA diagnostic: {:.4f} ms".format(
                latency["fast_then_sgla"]
            )
        )
    print("Pareto CSV: {}".format(output))


if __name__ == "__main__":
    main()
