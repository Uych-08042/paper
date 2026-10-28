"""Audit Stage-16A continuous behavior data before any offline-RL training."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint, write_csv, write_json
from stage8_intervention_common import OVERLAP_THRESHOLDS
from stage16_behavior_common import (
    ACTION_NAMES,
    behavior_cache_path,
    load_and_audit_cache,
    load_behavior_run,
)


SCHEMA = "stage16_behavior_audit_v1"


def _window_auc(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return 100.0 * float(
        np.mean(values[:, None] > OVERLAP_THRESHOLDS[None, :])
    )


def _safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()) if values.size else float("nan")


def audit_behavior_dataset(directory, allow_incomplete=False):
    directory, run = load_behavior_run(directory)
    reports = []
    trajectory_chunks = defaultdict(
        lambda: {
            "sequences": set(),
            "iou": [],
            "gain": [],
            "actions": np.zeros(len(ACTION_NAMES), dtype=np.int64),
            "effect_count": 0,
            "candidate_count": 0,
            "divergence_count": 0,
            "divergence_total": 0,
        }
    )
    split_actions = defaultdict(lambda: np.zeros(len(ACTION_NAMES), dtype=np.int64))
    split_valid = defaultdict(lambda: [0, 0])
    missing = []

    for entry in run["manifest"]["sequences"]:
        path = behavior_cache_path(directory, entry)
        if not path.is_file():
            missing.append(path)
            continue
        report, data, _ = load_and_audit_cache(path, entry, run)
        reports.append(report)
        split = entry["split"]
        valid = data["valid_gt"]
        split_valid[split][0] += int(valid.sum()) * len(run["behaviors"])
        split_valid[split][1] += int(valid.size) * len(run["behaviors"])
        fast_episode = next(
            index
            for index, spec in enumerate(run["behaviors"])
            if spec["candidate_rate_percent"] == 0.0
        )
        fast_state = data["previous_state"][fast_episode]
        for episode, spec in enumerate(run["behaviors"]):
            key = (split, spec["behavior_id"])
            chunk = trajectory_chunks[key]
            chunk["sequences"].add(entry["sequence"])
            chunk["iou"].append(data["executed_iou"][episode, valid])
            chunk["gain"].append(data["reward_gain_vs_fast"][episode, valid])
            actions = data["behavior_action"][episode]
            counts = np.bincount(actions, minlength=len(ACTION_NAMES))
            chunk["actions"] += counts
            split_actions[split] += counts
            candidate = actions > 0
            effects = (
                np.max(
                    np.abs(
                        data["executed_box"][episode]
                        - data["fast_box"][episode]
                    ),
                    axis=1,
                )
                > 1e-7
            )
            chunk["effect_count"] += int(np.sum(effects & candidate))
            chunk["candidate_count"] += int(candidate.sum())
            if episode != fast_episode:
                divergent = (
                    np.max(
                        np.abs(data["previous_state"][episode] - fast_state),
                        axis=1,
                    )
                    > 1e-7
                )
                chunk["divergence_count"] += int(divergent.sum())
                chunk["divergence_total"] += len(divergent)

    expected = len(run["manifest"]["sequences"])
    if missing and not allow_incomplete:
        raise ValueError(
            "Stage-16 cache is incomplete: {} of {} files missing".format(
                len(missing), expected
            )
        )
    if not reports:
        raise ValueError("No Stage-16 behavior caches were loaded")

    trajectory_rows = []
    for split, spec_id in sorted(trajectory_chunks):
        chunk = trajectory_chunks[(split, spec_id)]
        spec = next(item for item in run["behaviors"] if item["behavior_id"] == spec_id)
        iou = np.concatenate(chunk["iou"]) if chunk["iou"] else np.asarray([])
        gain = np.concatenate(chunk["gain"]) if chunk["gain"] else np.asarray([])
        transitions = int(chunk["actions"].sum())
        candidates = int(chunk["actions"][1:].sum())
        trajectory_rows.append(
            {
                "split": split,
                "behavior_id": spec_id,
                "target_candidate_percent": spec["candidate_rate_percent"],
                "num_sequences": len(chunk["sequences"]),
                "transitions": transitions,
                "candidate_transitions": candidates,
                "actual_candidate_percent": 100.0 * candidates / transitions,
                "mean_iou": _safe_mean(iou),
                "success_auc": _window_auc(iou),
                "mean_immediate_gain_vs_fast": _safe_mean(gain),
                "candidate_action_effect_ratio": (
                    chunk["effect_count"] / chunk["candidate_count"]
                    if chunk["candidate_count"]
                    else 0.0
                ),
                "state_divergence_vs_fast_ratio": (
                    chunk["divergence_count"] / chunk["divergence_total"]
                    if chunk["divergence_total"]
                    else 0.0
                ),
                **{
                    "{}_count".format(name): int(chunk["actions"][index])
                    for index, name in enumerate(ACTION_NAMES)
                },
            }
        )

    coverage_rows = []
    for split in sorted(split_actions):
        counts = split_actions[split]
        total = int(counts.sum())
        for action, name in enumerate(ACTION_NAMES):
            coverage_rows.append(
                {
                    "split": split,
                    "action": name,
                    "action_index": action,
                    "count": int(counts[action]),
                    "fraction": float(counts[action] / total),
                    "total_transitions": total,
                }
            )

    sequence_rows = sorted(reports, key=lambda row: (row["split"], row["sequence"]))
    observed_splits = sorted(split_actions)
    complete = not missing
    full_split_coverage = complete and set(observed_splits) == {"train", "val"}
    minimum_counts = {
        split: int(split_actions[split].min()) for split in observed_splits
    }
    minimum_fractions = {
        split: float((split_actions[split] / split_actions[split].sum()).min())
        for split in observed_splits
    }
    effect_values = [
        row["candidate_action_effect_ratio"]
        for row in sequence_rows
        if row["candidate_action_ratio"] > 0.0
    ]
    minimum_effect = min(effect_values) if effect_values else 0.0
    valid_ratios = {
        split: split_valid[split][0] / split_valid[split][1]
        for split in observed_splits
    }

    if complete:
        coverage_pass = (
            full_split_coverage
            and all(value >= 100 for value in minimum_counts.values())
            and all(value >= 0.02 for value in minimum_fractions.values())
        )
        valid_pass = all(value >= 0.8 for value in valid_ratios.values())
        effect_pass = minimum_effect >= 0.5
        status = (
            "PASS_BEHAVIOR_DATA"
            if coverage_pass and valid_pass and effect_pass
            else "INSUFFICIENT_BEHAVIOR_DATA"
        )
    else:
        all_observed_actions = all(
            np.all(split_actions[split] > 0) for split in observed_splits
        )
        status = (
            "PILOT_PASS"
            if all_observed_actions and minimum_effect >= 0.5
            else "PILOT_FAIL"
        )

    decision = {
        "schema": SCHEMA,
        "status": status,
        "analysis_role": "environment_and_behavior_data_integrity_gate",
        "source_run_fingerprint": run["fingerprint"],
        "audited_clips": len(reports),
        "expected_clips": expected,
        "missing_clips": len(missing),
        "episodes_per_clip": len(run["behaviors"]),
        "checks": {
            "schedule_reconstruction_and_state_continuity": True,
            "complete": complete,
            "minimum_action_count_by_split": minimum_counts,
            "minimum_action_fraction_by_split": minimum_fractions,
            "valid_label_ratio_by_split": valid_ratios,
            "minimum_candidate_action_effect_ratio_by_clip": minimum_effect,
            "full_coverage_rule": "each action >=100 samples and >=2% in train and val",
            "minimum_valid_label_ratio": 0.8,
            "minimum_candidate_action_effect_ratio": 0.5,
        },
        "next_step": (
            "Freeze this cache and implement Stage-16B conservative recurrent offline Q-learning."
            if status == "PASS_BEHAVIOR_DATA"
            else (
                "Resume the identical frozen collection; do not train on an incomplete pilot."
                if status == "PILOT_PASS"
                else "Stop before policy training and repair behavior coverage or transition semantics."
            )
        ),
        "warning": (
            "Stage-16A validates randomized continuous transition data only. "
            "Trajectory AUCs are descriptive behavior-policy outcomes, not a learned-policy result."
        ),
        "code_sha256": {
            "stage16_behavior_audit.py": file_hash(Path(__file__).resolve()),
            "stage16_behavior_common.py": file_hash(
                Path(__file__).resolve().parent / "stage16_behavior_common.py"
            ),
        },
    }
    decision["fingerprint"] = fingerprint(decision)
    return run, sequence_rows, trajectory_rows, coverage_rows, decision, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--allow_incomplete", action="store_true")
    args = parser.parse_args()

    run, sequences, trajectories, coverage, decision, missing = audit_behavior_dataset(
        args.input, args.allow_incomplete
    )
    output = Path(args.output_dir).expanduser().resolve()
    write_csv(output / "stage16_behavior_sequence_summary.csv", sequences)
    write_csv(output / "stage16_behavior_trajectory_summary.csv", trajectories)
    write_csv(output / "stage16_behavior_action_coverage.csv", coverage)
    write_json(output / "stage16_behavior_decision.json", decision)

    print(
        "Audited {} of {} Stage-16 clips ({} missing); {} episodes per clip.".format(
            decision["audited_clips"],
            decision["expected_clips"],
            decision["missing_clips"],
            decision["episodes_per_clip"],
        )
    )
    print("split  behavior     target  actual  sequences  transitions  mean_iou  auc")
    for row in trajectories:
        print(
            "{split:<6s} {behavior_id:<11s} {target_candidate_percent:>6.1f}% "
            "{actual_candidate_percent:>6.2f}% {num_sequences:>10d} "
            "{transitions:>12d} {mean_iou:>9.4f} {success_auc:>7.3f}".format(**row)
        )
    print("\nAction coverage:")
    for split in sorted({row["split"] for row in coverage}):
        rows = [row for row in coverage if row["split"] == split]
        print(
            "{}: {}".format(
                split,
                " | ".join(
                    "{}={:.2f}%".format(row["action"], 100.0 * row["fraction"])
                    for row in rows
                ),
            )
        )
    print("\nPre-training decision: {}".format(decision["status"]))
    print("Decision JSON: {}".format(output / "stage16_behavior_decision.json"))
    print("Reports: {}".format(output))
    if missing:
        print("Pilot/incomplete audit only; resume the exact run before Stage-16B.")


if __name__ == "__main__":
    main()
