"""Create cost-matched control assignments for Stage-6 switching."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from stage3_periodic import PERIODIC_L9_BUDGETS, periodic_l9_active
from stage6_budget_oracle import solve_multiple_choice, write_csv
from stage6_switch_common import ALLOWED_POLICIES, load_assignments


DEFAULT_RANDOM_POLICIES = ("fast", "periodic_l9_30", "sgla")


def policy_cost(policy, early_frames, late_frames):
    if policy not in ALLOWED_POLICIES:
        raise ValueError("Unsupported policy: {}".format(policy))
    if early_frames < 1 or late_frames < 1:
        raise ValueError("Early and late frame counts must be positive")
    if policy == "fast":
        return 0
    if policy == "sgla":
        return int(late_frames)
    if policy in PERIODIC_L9_BUDGETS:
        last_frame = early_frames + late_frames
        return sum(
            int(periodic_l9_active(policy, frame_id))
            for frame_id in range(early_frames + 1, last_frame + 1)
        )
    raise ValueError("No cost rule for policy: {}".format(policy))


def cost_matrix(assignments, policies):
    keys = sorted(assignments)
    costs = np.asarray(
        [
            [
                policy_cost(
                    policy,
                    assignments[key]["early_frames"],
                    assignments[key]["late_frames"],
                )
                for policy in policies
            ]
            for key in keys
        ],
        dtype=np.int64,
    )
    return keys, costs


def uniform_selection(assignments, policy):
    return {key: policy for key in sorted(assignments)}


def random_budget_selection(assignments, policies, target_blocks, seed):
    policies = tuple(policies)
    if "fast" not in policies:
        raise ValueError("Random controls require the Fast action")
    keys, costs = cost_matrix(assignments, policies)
    rng = np.random.default_rng(seed)
    values = rng.random(costs.shape)
    values[:, policies.index("fast")] = 0.0
    solution = solve_multiple_choice(values, costs, [target_blocks])[target_blocks]
    return {
        key: policies[int(policy_index)]
        for key, policy_index in zip(keys, solution["policy_indices"])
    }


def assignment_rows(
    assignments,
    selected,
    early_fraction,
    budget_percent,
    control_name,
    seed="",
):
    rows = []
    for key in sorted(assignments):
        source = assignments[key]
        policy = selected[key]
        rows.append(
            {
                "early_fraction": float(early_fraction),
                "budget_percent": float(budget_percent),
                "dataset": source["dataset"],
                "sequence": source["sequence"],
                "fold": source["fold"],
                "early_frames": source["early_frames"],
                "late_frames": source["late_frames"],
                "router_policy": policy,
                "router_candidate_blocks": policy_cost(
                    policy,
                    source["early_frames"],
                    source["late_frames"],
                ),
                "control_name": control_name,
                "control_seed": seed,
                "source_router_policy": source["router_policy"],
            }
        )
    return rows


def summarize(rows, source_blocks, total_tracked, total_late):
    total_blocks = sum(row["router_candidate_blocks"] for row in rows)
    distribution = Counter(row["router_policy"] for row in rows)
    return {
        "control_name": rows[0]["control_name"],
        "control_seed": rows[0]["control_seed"],
        "num_sequences": len(rows),
        "tracked_frames": total_tracked,
        "late_frames": total_late,
        "candidate_blocks": total_blocks,
        "candidate_blocks_per_full_frame": total_blocks / total_tracked,
        "candidate_blocks_per_late_frame": total_blocks / total_late,
        "source_router_candidate_blocks": source_blocks,
        "block_difference_vs_router": total_blocks - source_blocks,
        "assignment_distribution": "|".join(
            "{}:{}".format(policy, distribution[policy])
            for policy in sorted(distribution)
        ),
    }


def _safe_name(value):
    return "".join(
        character if character.isalnum() else "_" for character in value
    ).strip("_")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--early_fraction", type=float, default=0.5)
    parser.add_argument("--budget_percent", type=float, default=10.0)
    parser.add_argument("--uniform_policy", default="periodic_l9_10")
    parser.add_argument("--random_seeds", type=int, nargs="*", default=[])
    parser.add_argument(
        "--random_policies",
        nargs="+",
        default=list(DEFAULT_RANDOM_POLICIES),
    )
    args = parser.parse_args()
    if args.uniform_policy not in ALLOWED_POLICIES:
        raise ValueError("Unsupported uniform policy: {}".format(args.uniform_policy))
    invalid_random = set(args.random_policies).difference(ALLOWED_POLICIES)
    if invalid_random:
        raise ValueError(
            "Unsupported random policies: {}".format(
                ", ".join(sorted(invalid_random))
            )
        )

    source_path, assignments = load_assignments(
        args.input, args.early_fraction, args.budget_percent
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_blocks = sum(
        assignment["router_candidate_blocks"]
        for assignment in assignments.values()
    )
    total_late = sum(
        assignment["late_frames"] for assignment in assignments.values()
    )
    total_tracked = sum(
        assignment["early_frames"] + assignment["late_frames"]
        for assignment in assignments.values()
    )

    outputs = []
    summaries = []
    uniform_name = "uniform_late_{}".format(
        args.uniform_policy.replace("periodic_l9_", "p")
    )
    uniform_rows = assignment_rows(
        assignments,
        uniform_selection(assignments, args.uniform_policy),
        args.early_fraction,
        args.budget_percent,
        uniform_name,
    )
    uniform_path = output_dir / "{}_assignments.csv".format(
        _safe_name(uniform_name)
    )
    write_csv(uniform_path, uniform_rows)
    outputs.append(uniform_path)
    summaries.append(
        summarize(uniform_rows, source_blocks, total_tracked, total_late)
    )

    for seed in args.random_seeds:
        name = "random_budget_seed{}".format(seed)
        selected = random_budget_selection(
            assignments,
            args.random_policies,
            source_blocks,
            seed,
        )
        rows = assignment_rows(
            assignments,
            selected,
            args.early_fraction,
            args.budget_percent,
            name,
            seed=seed,
        )
        path = output_dir / "{}_assignments.csv".format(_safe_name(name))
        write_csv(path, rows)
        outputs.append(path)
        summaries.append(summarize(rows, source_blocks, total_tracked, total_late))

    summary_path = output_dir / "stage6_matched_assignment_summary.csv"
    write_csv(summary_path, summaries)
    print("Source assignments: {}".format(source_path))
    print(
        "Router target: {} blocks over {} tracked frames ({:.4%})".format(
            source_blocks, total_tracked, source_blocks / total_tracked
        )
    )
    print("control                    blocks    full%    late%    delta  distribution")
    for row in summaries:
        print(
            "{control_name:<26s} {candidate_blocks:>7d}  "
            "{candidate_blocks_per_full_frame:>7.2%}  "
            "{candidate_blocks_per_late_frame:>7.2%}  "
            "{block_difference_vs_router:>7d}  "
            "{assignment_distribution}".format(**row)
        )
    for path in outputs:
        print("Assignment CSV: {}".format(path))
    print("Summary CSV: {}".format(summary_path))


if __name__ == "__main__":
    main()
