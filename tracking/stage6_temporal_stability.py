"""Audit whether early sequence policy preferences transfer to later frames.

The early segment still uses ground truth, so this is a necessary-condition
oracle rather than a deployable controller. A budget allocation learned from
early AUC is evaluated on disjoint later frames under the same global block
budget and compared with both a fixed policy and a late-ground-truth oracle.
"""

import argparse
import csv
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import stage2_candidate_oracle as stage2
from stage3_closed_loop_summary import load_closed_loop
from stage6_budget_oracle import POLICY_ORDER, solve_multiple_choice


def segment_auc(overlaps):
    overlaps = np.asarray(overlaps, dtype=np.float64)
    if overlaps.ndim != 1 or overlaps.size == 0:
        raise ValueError("A non-empty one-dimensional overlap array is required")
    thresholds = np.asarray(stage2.OVERLAP_THRESHOLDS, dtype=np.float64)
    return 100.0 * float(
        np.mean(overlaps[:, None] > thresholds[None, :])
    )


def build_temporal_sequences(rows, policies, fraction):
    if not 0.0 < fraction < 1.0:
        raise ValueError("Temporal fraction must lie strictly between 0 and 1")
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["policy"] in policies and row["frame_id"] > 0:
            grouped[(row["dataset"], row["sequence"])][row["policy"]].append(
                row
            )
    if not grouped:
        raise ValueError("No tracked rows found for requested policies")

    output = []
    required = set(policies)
    for key in sorted(grouped):
        policy_rows = grouped[key]
        missing = required.difference(policy_rows)
        if missing:
            raise ValueError(
                "{}/{} is missing policies: {}".format(
                    key[0], key[1], ", ".join(sorted(missing))
                )
            )
        ordered = {
            policy: sorted(policy_rows[policy], key=lambda row: row["frame_id"])
            for policy in policies
        }
        reference_ids = [row["frame_id"] for row in ordered[policies[0]]]
        if len(reference_ids) < 2:
            raise ValueError(
                "Sequence {}/{} needs at least two tracked frames".format(*key)
            )
        for policy in policies[1:]:
            frame_ids = [row["frame_id"] for row in ordered[policy]]
            if frame_ids != reference_ids:
                raise ValueError(
                    "Policies have different frame IDs for {}/{}".format(*key)
                )
        split = int(math.floor(len(reference_ids) * fraction + 1e-12))
        split = min(max(split, 1), len(reference_ids) - 1)
        records = {}
        for policy in policies:
            policy_data = ordered[policy]
            early = policy_data[:split]
            late = policy_data[split:]
            records[policy] = {
                "early_auc": segment_auc([row["iou"] for row in early]),
                "late_auc": segment_auc([row["iou"] for row in late]),
                "early_candidate_blocks": int(
                    sum(row["candidate_blocks"] for row in early)
                ),
                "late_candidate_blocks": int(
                    sum(row["candidate_blocks"] for row in late)
                ),
            }
        output.append(
            {
                "dataset": key[0],
                "sequence": key[1],
                "early_frames": split,
                "late_frames": len(reference_ids) - split,
                "records": records,
            }
        )
    return output


def temporal_matrices(sequences, policies):
    early_values = np.asarray(
        [
            [sequence["records"][policy]["early_auc"] for policy in policies]
            for sequence in sequences
        ],
        dtype=np.float64,
    )
    late_values = np.asarray(
        [
            [sequence["records"][policy]["late_auc"] for policy in policies]
            for sequence in sequences
        ],
        dtype=np.float64,
    )
    late_costs = np.asarray(
        [
            [
                sequence["records"][policy]["late_candidate_blocks"]
                for policy in policies
            ]
            for sequence in sequences
        ],
        dtype=np.int64,
    )
    return early_values, late_values, late_costs


def bootstrap_mean(values, samples, rng):
    values = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "probability_positive": float(np.mean(means > 0.0)),
    }


def _distribution(indices, policies):
    counts = Counter(policies[index] for index in indices.tolist())
    return "|".join(
        "{}:{}".format(policy, counts.get(policy, 0)) for policy in policies
    )


def analyze_fraction(sequences, policies, budgets, bootstrap_samples, rng):
    early_values, late_values, late_costs = temporal_matrices(
        sequences, policies
    )
    num_sequences = len(sequences)
    total_late_frames = sum(sequence["late_frames"] for sequence in sequences)
    capacities = [
        int(math.floor(total_late_frames * budget / 100.0 + 1e-12))
        for budget in budgets
    ]
    early_solutions = solve_multiple_choice(
        early_values, late_costs, capacities
    )
    late_solutions = solve_multiple_choice(late_values, late_costs, capacities)
    fixed = []
    for policy_index, policy in enumerate(policies):
        fixed.append(
            {
                "policy": policy,
                "total_cost": int(late_costs[:, policy_index].sum()),
                "auc": float(late_values[:, policy_index].mean()),
                "policy_index": policy_index,
            }
        )

    summary = []
    assignments = []
    sequence_indices = np.arange(num_sequences)
    for budget, capacity in zip(budgets, capacities):
        early_solution = early_solutions[capacity]
        late_solution = late_solutions[capacity]
        early_selected = early_solution["policy_indices"]
        late_selected = late_solution["policy_indices"]
        selected_late = late_values[sequence_indices, early_selected]
        late_oracle_auc = late_solution["total_value"] / num_sequences
        feasible_fixed = [row for row in fixed if row["total_cost"] <= capacity]
        best_fixed = max(
            feasible_fixed,
            key=lambda row: (row["auc"], -row["total_cost"]),
        )
        fixed_late = late_values[:, best_fixed["policy_index"]]
        gain = selected_late - fixed_late
        bootstrap = bootstrap_mean(gain, bootstrap_samples, rng)
        selected_late_auc = float(selected_late.mean())
        headroom = late_oracle_auc - best_fixed["auc"]
        summary.append(
            {
                "budget_percent": float(budget),
                "num_sequences": num_sequences,
                "early_frames": sum(
                    sequence["early_frames"] for sequence in sequences
                ),
                "late_frames": total_late_frames,
                "budget_candidate_blocks": capacity,
                "early_selected_blocks": early_solution["total_cost"],
                "early_selected_block_ratio": early_solution["total_cost"]
                / total_late_frames,
                "early_selected_late_auc": selected_late_auc,
                "best_fixed_late_policy": best_fixed["policy"],
                "best_fixed_late_auc": best_fixed["auc"],
                "gain_vs_best_fixed_late": selected_late_auc
                - best_fixed["auc"],
                "gain_ci_low": bootstrap["ci_low"],
                "gain_ci_high": bootstrap["ci_high"],
                "gain_probability_positive": bootstrap[
                    "probability_positive"
                ],
                "late_oracle_blocks": late_solution["total_cost"],
                "late_oracle_block_ratio": late_solution["total_cost"]
                / total_late_frames,
                "late_oracle_auc": late_oracle_auc,
                "gap_to_late_oracle": late_oracle_auc - selected_late_auc,
                "recovery_of_late_oracle_headroom": (
                    (selected_late_auc - best_fixed["auc"]) / headroom
                    if headroom > 1e-12
                    else float("nan")
                ),
                "assignment_exact_match_rate": float(
                    np.mean(early_selected == late_selected)
                ),
                "early_selection_distribution": _distribution(
                    early_selected, policies
                ),
                "late_oracle_distribution": _distribution(
                    late_selected, policies
                ),
            }
        )
        for sequence_index, sequence in enumerate(sequences):
            early_policy_index = int(early_selected[sequence_index])
            late_policy_index = int(late_selected[sequence_index])
            assignments.append(
                {
                    "budget_percent": float(budget),
                    "dataset": sequence["dataset"],
                    "sequence": sequence["sequence"],
                    "early_frames": sequence["early_frames"],
                    "late_frames": sequence["late_frames"],
                    "early_selected_policy": policies[early_policy_index],
                    "late_oracle_policy": policies[late_policy_index],
                    "early_selected_policy_early_auc": early_values[
                        sequence_index, early_policy_index
                    ],
                    "early_selected_policy_late_auc": late_values[
                        sequence_index, early_policy_index
                    ],
                    "best_fixed_late_policy": best_fixed["policy"],
                    "best_fixed_policy_late_auc": fixed_late[sequence_index],
                    "gain_vs_best_fixed_late": gain[sequence_index],
                    "late_oracle_policy_late_auc": late_values[
                        sequence_index, late_policy_index
                    ],
                    "gap_to_late_oracle": late_values[
                        sequence_index, late_policy_index
                    ]
                    - late_values[sequence_index, early_policy_index],
                }
            )
    return summary, assignments


def _format(value):
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        return "{:.8f}".format(float(value))
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: _format(value) for key, value in row.items()}
            )
    os.replace(str(temporary), str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--policies", nargs="+", choices=POLICY_ORDER, default=list(POLICY_ORDER)
    )
    parser.add_argument(
        "--fractions", nargs="+", type=float, default=[0.25, 0.5]
    )
    parser.add_argument(
        "--budgets", nargs="+", type=float, default=[10, 20, 30, 50]
    )
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    policies = tuple(dict.fromkeys(args.policies))
    if "fast" not in policies or "sgla" not in policies:
        raise ValueError("--policies must include fast and sgla")
    fractions = sorted(set(args.fractions))
    budgets = sorted(set(args.budgets))
    if any(not 0.0 < value < 1.0 for value in fractions):
        raise ValueError("--fractions must lie strictly between 0 and 1")
    if not budgets or budgets[0] < 0.0 or budgets[-1] > 100.0:
        raise ValueError("--budgets must lie in [0, 100]")
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap_samples must be positive")

    input_path, paths, rows = load_closed_loop(args.input)
    output = Path(args.output_dir).expanduser().resolve()
    rng = np.random.default_rng(args.seed)
    summaries = []
    assignments = []
    for fraction in fractions:
        sequences = build_temporal_sequences(rows, policies, fraction)
        fraction_summary, fraction_assignments = analyze_fraction(
            sequences, policies, budgets, args.bootstrap_samples, rng
        )
        for row in fraction_summary:
            row["early_fraction"] = fraction
        for row in fraction_assignments:
            row["early_fraction"] = fraction
        summaries.extend(fraction_summary)
        assignments.extend(fraction_assignments)

    summary_path = output / "stage6_temporal_stability.csv"
    assignment_path = output / "stage6_temporal_stability_assignments.csv"
    write_csv(summary_path, summaries)
    write_csv(assignment_path, assignments)

    print(
        "Loaded {} rows from {} policy CSV files under {}.".format(
            len(rows), len(paths), input_path
        )
    )
    print(
        "early  budget  transfer_auc  best_fixed  gain  paired_95%           "
        "late_oracle  recovery  match"
    )
    for row in summaries:
        interval = "[{:.4f},{:.4f}]".format(
            row["gain_ci_low"], row["gain_ci_high"]
        )
        print(
            "{early_fraction:>5.0%}  {budget_percent:>5.1f}%  "
            "{early_selected_late_auc:>12.4f}  "
            "{best_fixed_late_policy:<18s} "
            "{gain_vs_best_fixed_late:>6.4f}  {interval:<19s} "
            "{late_oracle_auc:>11.4f}  "
            "{recovery_of_late_oracle_headroom:>8.2%}  "
            "{assignment_exact_match_rate:>6.2%}".format(
                interval=interval, **row
            )
        )
    print("Temporal stability CSV: {}".format(summary_path))
    print("Assignment CSV: {}".format(assignment_path))
    print(
        "Warning: early selection uses early-segment ground truth and each "
        "policy's pre-existing closed-loop trajectory. It is a transfer "
        "diagnostic, not a deployable switching result."
    )


if __name__ == "__main__":
    main()
