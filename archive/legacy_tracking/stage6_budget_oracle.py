"""Measure sequence-level adaptive-budget headroom under exact block budgets.

This is a ground-truth oracle, not a deployable policy. It selects one complete
closed-loop Stage-5 policy per sequence while constraining the total number of
candidate-block executions across all tracked frames.
"""

import argparse
import csv
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np


POLICY_ORDER = (
    "fast",
    "periodic_l9_10",
    "periodic_l9_20",
    "periodic_l9_30",
    "periodic_l9_50",
    "sgla",
)
REQUIRED_FIELDS = {
    "dataset",
    "sequence",
    "policy",
    "candidate_blocks",
    "num_frames",
    "auc",
}


def load_sequence_results(path, policies=POLICY_ORDER):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("Sequence summary missing: {}".format(path))

    by_sequence = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_FIELDS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Sequence summary is missing fields: {}".format(
                    ", ".join(sorted(missing))
                )
            )
        for row in reader:
            policy = row["policy"]
            if policy not in policies:
                continue
            key = (row["dataset"], row["sequence"])
            records = by_sequence.setdefault(key, {})
            if policy in records:
                raise ValueError(
                    "Duplicate sequence-policy row: {}/{}".format(
                        row["sequence"], policy
                    )
                )
            num_frames = int(row["num_frames"])
            if num_frames < 2:
                raise ValueError("Each sequence must have a tracked frame")
            tracked_frames = num_frames - 1
            block_ratio = float(row["candidate_blocks"])
            raw_blocks = block_ratio * tracked_frames
            candidate_blocks = int(round(raw_blocks))
            if abs(raw_blocks - candidate_blocks) > 1e-3:
                raise ValueError(
                    "Non-integral candidate count for {}/{}: {}".format(
                        row["sequence"], policy, raw_blocks
                    )
                )
            records[policy] = {
                "num_frames": num_frames,
                "tracked_frames": tracked_frames,
                "candidate_blocks": candidate_blocks,
                "auc": float(row["auc"]),
            }

    if not by_sequence:
        raise ValueError("No requested policies found in sequence summary")
    required = set(policies)
    output = []
    for key in sorted(by_sequence):
        records = by_sequence[key]
        missing = required.difference(records)
        if missing:
            raise ValueError(
                "{}/{} is missing policies: {}".format(
                    key[0], key[1], ", ".join(sorted(missing))
                )
            )
        frame_counts = {records[policy]["num_frames"] for policy in policies}
        if len(frame_counts) != 1:
            raise ValueError(
                "Policies have different frame counts for {}/{}".format(*key)
            )
        output.append(
            {
                "dataset": key[0],
                "sequence": key[1],
                "tracked_frames": records[policies[0]]["tracked_frames"],
                "records": records,
            }
        )
    return path, output


def sequence_matrices(sequences, policies):
    values = np.asarray(
        [
            [sequence["records"][policy]["auc"] for policy in policies]
            for sequence in sequences
        ],
        dtype=np.float64,
    )
    costs = np.asarray(
        [
            [
                sequence["records"][policy]["candidate_blocks"]
                for policy in policies
            ]
            for sequence in sequences
        ],
        dtype=np.int64,
    )
    return values, costs


def solve_multiple_choice(values, costs, capacities):
    """Solve all requested multiple-choice knapsack capacities in one DP."""
    values = np.asarray(values, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.int64)
    capacities = [int(value) for value in capacities]
    if values.ndim != 2 or values.shape != costs.shape:
        raise ValueError("values and costs must be equally shaped matrices")
    if values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("At least one sequence and one policy are required")
    if np.any(costs < 0) or any(value < 0 for value in capacities):
        raise ValueError("Costs and capacities must be non-negative")
    if values.shape[1] > np.iinfo(np.int8).max:
        raise ValueError("Too many policy options for compact backtracking")

    max_capacity = max(capacities)
    num_sequences, num_policies = values.shape
    choices = np.full(
        (num_sequences, max_capacity + 1), -1, dtype=np.int8
    )
    dp = np.full(max_capacity + 1, -np.inf, dtype=np.float64)
    dp[0] = 0.0

    for sequence_index in range(num_sequences):
        updated = np.full_like(dp, -np.inf)
        selected = choices[sequence_index]
        option_order = sorted(
            range(num_policies),
            key=lambda index: (int(costs[sequence_index, index]), index),
        )
        for policy_index in option_order:
            cost = int(costs[sequence_index, policy_index])
            if cost > max_capacity:
                continue
            candidate = (
                dp[: max_capacity - cost + 1]
                + values[sequence_index, policy_index]
            )
            target = updated[cost:]
            better = candidate > target + 1e-12
            target[better] = candidate[better]
            selected[cost:][better] = policy_index
        dp = updated

    solutions = {}
    for capacity in capacities:
        exact_cost = int(np.argmax(dp[: capacity + 1]))
        if not np.isfinite(dp[exact_cost]):
            raise RuntimeError("No feasible allocation for capacity {}".format(capacity))
        policy_indices = np.empty(num_sequences, dtype=np.int64)
        remaining = exact_cost
        for sequence_index in range(num_sequences - 1, -1, -1):
            policy_index = int(choices[sequence_index, remaining])
            if policy_index < 0:
                raise RuntimeError("Oracle backtracking reached an invalid state")
            policy_indices[sequence_index] = policy_index
            remaining -= int(costs[sequence_index, policy_index])
        if remaining != 0:
            raise RuntimeError("Oracle backtracking did not return to zero cost")
        solutions[capacity] = {
            "policy_indices": policy_indices,
            "total_cost": exact_cost,
            "total_value": float(dp[exact_cost]),
        }
    return solutions


def fixed_policy_summary(sequences, policies, values, costs):
    total_frames = sum(sequence["tracked_frames"] for sequence in sequences)
    return [
        {
            "policy": policy,
            "uses_ground_truth": 0,
            "num_sequences": len(sequences),
            "tracked_frames": total_frames,
            "total_candidate_blocks": int(costs[:, index].sum()),
            "candidate_blocks_per_frame": float(costs[:, index].sum())
            / total_frames,
            "auc": float(values[:, index].mean()),
        }
        for index, policy in enumerate(policies)
    ]


def analyze(sequences, policies, budgets):
    values, costs = sequence_matrices(sequences, policies)
    total_frames = sum(sequence["tracked_frames"] for sequence in sequences)
    capacities = [
        int(math.floor(total_frames * budget / 100.0 + 1e-12))
        for budget in budgets
    ]
    solutions = solve_multiple_choice(values, costs, capacities)
    fixed_rows = fixed_policy_summary(sequences, policies, values, costs)
    fixed_by_policy = {row["policy"]: row for row in fixed_rows}
    fast_auc = fixed_by_policy["fast"]["auc"]
    sgla_auc = fixed_by_policy["sgla"]["auc"]
    sgla_gain = sgla_auc - fast_auc

    oracle_rows = []
    assignment_rows = []
    for budget, capacity in zip(budgets, capacities):
        solution = solutions[capacity]
        selected = solution["policy_indices"]
        actual_cost = solution["total_cost"]
        auc = solution["total_value"] / len(sequences)
        feasible_fixed = [
            row
            for row in fixed_rows
            if row["total_candidate_blocks"] <= capacity
        ]
        best_fixed = max(
            feasible_fixed,
            key=lambda row: (row["auc"], -row["total_candidate_blocks"]),
        )
        sequence_distribution = Counter(
            policies[index] for index in selected.tolist()
        )
        frame_distribution = Counter()
        for sequence, policy_index in zip(sequences, selected.tolist()):
            frame_distribution[policies[policy_index]] += sequence[
                "tracked_frames"
            ]
        distribution_text = "|".join(
            "{}:{}".format(policy, sequence_distribution.get(policy, 0))
            for policy in policies
        )
        frame_distribution_text = "|".join(
            "{}:{:.2%}".format(
                policy, frame_distribution.get(policy, 0) / total_frames
            )
            for policy in policies
        )
        oracle_rows.append(
            {
                "budget_percent": float(budget),
                "uses_ground_truth": 1,
                "num_sequences": len(sequences),
                "tracked_frames": total_frames,
                "budget_candidate_blocks": capacity,
                "used_candidate_blocks": actual_cost,
                "candidate_blocks_per_frame": actual_cost / total_frames,
                "budget_utilization": (
                    actual_cost / capacity if capacity > 0 else 0.0
                ),
                "oracle_auc": auc,
                "gain_vs_fast": auc - fast_auc,
                "gain_vs_sgla": auc - sgla_auc,
                "recovery_of_sgla_gain": (
                    (auc - fast_auc) / sgla_gain
                    if sgla_gain > 0.0
                    else float("nan")
                ),
                "best_fixed_feasible": best_fixed["policy"],
                "best_fixed_feasible_auc": best_fixed["auc"],
                "gain_vs_best_fixed_feasible": auc - best_fixed["auc"],
                "sequence_policy_distribution": distribution_text,
                "frame_policy_distribution": frame_distribution_text,
            }
        )
        for sequence_index, (sequence, policy_index) in enumerate(
            zip(sequences, selected.tolist())
        ):
            policy = policies[policy_index]
            record = sequence["records"][policy]
            assignment_rows.append(
                {
                    "budget_percent": float(budget),
                    "dataset": sequence["dataset"],
                    "sequence": sequence["sequence"],
                    "tracked_frames": sequence["tracked_frames"],
                    "selected_policy": policy,
                    "candidate_blocks": record["candidate_blocks"],
                    "candidate_blocks_per_frame": record["candidate_blocks"]
                    / sequence["tracked_frames"],
                    "selected_auc": record["auc"],
                    "fast_auc": values[sequence_index, policies.index("fast")],
                    "sgla_auc": values[sequence_index, policies.index("sgla")],
                    "gain_vs_fast": record["auc"]
                    - values[sequence_index, policies.index("fast")],
                    "gain_vs_sgla": record["auc"]
                    - values[sequence_index, policies.index("sgla")],
                }
            )
    return fixed_rows, oracle_rows, assignment_rows


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
        "--budgets", nargs="+", type=float, default=[0, 10, 20, 30, 50, 100]
    )
    args = parser.parse_args()

    policies = tuple(dict.fromkeys(args.policies))
    if "fast" not in policies or "sgla" not in policies:
        raise ValueError("--policies must include fast and sgla")
    budgets = sorted(set(args.budgets))
    if not budgets or budgets[0] < 0.0 or budgets[-1] > 100.0:
        raise ValueError("--budgets must lie in [0, 100]")

    input_path, sequences = load_sequence_results(args.input, policies)
    fixed_rows, oracle_rows, assignment_rows = analyze(
        sequences, policies, budgets
    )
    output = Path(args.output_dir).expanduser().resolve()
    fixed_path = output / "stage6_fixed_policy_summary.csv"
    oracle_path = output / "stage6_budget_oracle.csv"
    assignment_path = output / "stage6_budget_oracle_assignments.csv"
    write_csv(fixed_path, fixed_rows)
    write_csv(oracle_path, oracle_rows)
    write_csv(assignment_path, assignment_rows)

    print(
        "Loaded {} sequences from {}.".format(len(sequences), input_path)
    )
    print("Fixed closed-loop policies:")
    print("policy               blocks/f       auc")
    for row in fixed_rows:
        print(
            "{policy:<20s} {candidate_blocks_per_frame:>8.4f}  "
            "{auc:>8.4f}".format(**row)
        )
    print("\nGround-truth sequence-budget oracle:")
    print(
        "budget  actual  oracle_auc  gain_fast  best_fixed  headroom  distribution"
    )
    for row in oracle_rows:
        print(
            "{budget_percent:>5.1f}%  {candidate_blocks_per_frame:>6.2%}  "
            "{oracle_auc:>10.4f}  {gain_vs_fast:>9.4f}  "
            "{best_fixed_feasible:<18s} "
            "{gain_vs_best_fixed_feasible:>8.4f}  "
            "{sequence_policy_distribution}".format(**row)
        )
    print("Fixed summary CSV: {}".format(fixed_path))
    print("Budget oracle CSV: {}".format(oracle_path))
    print("Assignments CSV: {}".format(assignment_path))
    print(
        "Warning: this oracle uses each sequence's ground-truth AUC and is an "
        "upper bound, not a deployable result."
    )


if __name__ == "__main__":
    main()
