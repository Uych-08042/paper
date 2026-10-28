"""Audit and summarize Stage-9 equal-cost schedule interventions."""

import argparse
from pathlib import Path

import numpy as np

from stage4_common import write_csv, write_json
from stage6_temporal_stability import bootstrap_mean
from stage9_schedule_common import (
    POLICIES,
    PRIMARY_COMPARISON,
    PRIMARY_HORIZON,
    load_and_audit_cache,
    load_schedule_run,
    schedule_cache_path,
    schedules_from_run,
)


COMPARISONS = (
    ("burst4", "fast"),
    ("paired4", "fast"),
    ("uniform4", "fast"),
    ("paired4", "burst4"),
    ("uniform4", "burst4"),
    ("uniform4", "paired4"),
)


def load_records(directory, require_complete=False):
    directory, run = load_schedule_run(directory)
    records = []
    missing = []
    for entry in run["manifest"]["sequences"]:
        path = schedule_cache_path(directory, entry)
        if not path.is_file():
            missing.append(path)
            continue
        report, data, metadata = load_and_audit_cache(path, entry, run)
        records.append(
            {
                "entry": entry,
                "path": path,
                "report": report,
                "data": data,
                "metadata": metadata,
            }
        )
    if require_complete and missing:
        raise FileNotFoundError(
            "Stage-9 cache incomplete: {} of {} clips missing; first: {}".format(
                len(missing), len(run["manifest"]["sequences"]), missing[0]
            )
        )
    if not records:
        raise ValueError("No Stage-9 cache files found")
    return directory, run, records, missing


def _scopes(records):
    output = []
    for name in ("train", "val"):
        selected = [row for row in records if row["entry"]["split"] == name]
        if selected:
            output.append((name, selected))
    if records:
        output.append(("pooled", records))
    return output


def _finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def _sequence_values(records, field, policy_index, horizon_index=None):
    values = []
    names = []
    for record in records:
        array = record["data"][field]
        selected = (
            array[:, policy_index]
            if horizon_index is None
            else array[:, policy_index, horizon_index]
        )
        value = _finite_mean(selected)
        if np.isfinite(value):
            names.append(record["entry"]["sequence"])
            values.append(value)
    return names, np.asarray(values, dtype=np.float64)


def _paired_sequence_values(records, field, left, right, horizon_index=None):
    left_names, left_values = _sequence_values(
        records, field, left, horizon_index
    )
    right_names, right_values = _sequence_values(
        records, field, right, horizon_index
    )
    if left_names != right_names:
        raise RuntimeError("Paired sequence sets changed")
    return left_names, left_values, right_values, left_values - right_values


def _bootstrap(values, samples, rng):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "probability_positive": float("nan"),
        }
    return bootstrap_mean(values, samples, rng)


def policy_summary(records, run, bootstrap_samples, rng):
    schedules = schedules_from_run(run)
    rows = []
    for split, selected in _scopes(records):
        for horizon_index, horizon in enumerate(run["evaluation_horizons"]):
            _, fast_auc = _sequence_values(selected, "policy_auc", 0, horizon_index)
            for policy_index, policy in enumerate(POLICIES):
                names, auc = _sequence_values(
                    selected, "policy_auc", policy_index, horizon_index
                )
                _, mean_iou = _sequence_values(
                    selected, "policy_mean_iou", policy_index, horizon_index
                )
                if len(auc) != len(fast_auc):
                    raise RuntimeError("Policy/Fast sequence count changed")
                delta = auc - fast_auc
                interval = _bootstrap(delta, bootstrap_samples, rng)
                blocks = sum(value < horizon for value in schedules[policy])
                rows.append(
                    {
                        "split": split,
                        "horizon": horizon,
                        "policy": policy,
                        "num_sequences": len(names),
                        "candidate_blocks_per_anchor": blocks,
                        "candidate_blocks_per_frame": blocks / horizon,
                        "mean_iou_macro": float(mean_iou.mean()),
                        "auc_macro": float(auc.mean()),
                        "gain_vs_fast": float(delta.mean()),
                        "gain_vs_fast_ci_low": interval["ci_low"],
                        "gain_vs_fast_ci_high": interval["ci_high"],
                        "gain_vs_fast_probability_positive": interval[
                            "probability_positive"
                        ],
                    }
                )
    return rows


def paired_summary(records, run, bootstrap_samples, rng):
    schedules = schedules_from_run(run)
    rows = []
    for split, selected in _scopes(records):
        for horizon_index, horizon in enumerate(run["evaluation_horizons"]):
            for left_name, right_name in COMPARISONS:
                left = POLICIES.index(left_name)
                right = POLICIES.index(right_name)
                names, left_values, right_values, delta = _paired_sequence_values(
                    selected, "policy_auc", left, right, horizon_index
                )
                interval = _bootstrap(delta, bootstrap_samples, rng)
                tolerance = 1e-12
                left_blocks = sum(value < horizon for value in schedules[left_name])
                right_blocks = sum(value < horizon for value in schedules[right_name])
                rows.append(
                    {
                        "split": split,
                        "horizon": horizon,
                        "left_policy": left_name,
                        "right_policy": right_name,
                        "is_primary": int(
                            split == "val"
                            and horizon == PRIMARY_HORIZON
                            and (left_name, right_name) == PRIMARY_COMPARISON
                        ),
                        "equal_cost_at_horizon": int(left_blocks == right_blocks),
                        "left_candidate_blocks": left_blocks,
                        "right_candidate_blocks": right_blocks,
                        "num_sequences": len(names),
                        "left_auc_macro": float(left_values.mean()),
                        "right_auc_macro": float(right_values.mean()),
                        "mean_delta": float(delta.mean()),
                        "median_delta": float(np.median(delta)),
                        "wins": int(np.sum(delta > tolerance)),
                        "ties": int(np.sum(np.abs(delta) <= tolerance)),
                        "losses": int(np.sum(delta < -tolerance)),
                        "ci_low": interval["ci_low"],
                        "ci_high": interval["ci_high"],
                        "probability_positive": interval[
                            "probability_positive"
                        ],
                    }
                )
    return rows


def post_action_summary(records, bootstrap_samples, rng):
    rows = []
    metrics = (
        ("common_tail_auc", "common_tail_16_20"),
        ("persistence_auc", "persistence_20_40"),
    )
    for split, selected in _scopes(records):
        for field, window in metrics:
            for left_name, right_name in COMPARISONS:
                left = POLICIES.index(left_name)
                right = POLICIES.index(right_name)
                names, left_values, right_values, delta = _paired_sequence_values(
                    selected, field, left, right
                )
                interval = _bootstrap(delta, bootstrap_samples, rng)
                tolerance = 1e-12
                rows.append(
                    {
                        "split": split,
                        "window": window,
                        "left_policy": left_name,
                        "right_policy": right_name,
                        "num_sequences": len(names),
                        "left_auc_macro": float(left_values.mean()),
                        "right_auc_macro": float(right_values.mean()),
                        "mean_delta": float(delta.mean()),
                        "median_delta": float(np.median(delta)),
                        "wins": int(np.sum(delta > tolerance)),
                        "ties": int(np.sum(np.abs(delta) <= tolerance)),
                        "losses": int(np.sum(delta < -tolerance)),
                        "ci_low": interval["ci_low"],
                        "ci_high": interval["ci_high"],
                        "probability_positive": interval[
                            "probability_positive"
                        ],
                    }
                )
    return rows


def step_effect_summary(records, run, bootstrap_samples, rng):
    schedules = schedules_from_run(run)
    rows = []
    for split, selected in _scopes(records):
        for left_name, right_name in COMPARISONS:
            left = POLICIES.index(left_name)
            right = POLICIES.index(right_name)
            for step in range(run["horizon"]):
                values = []
                for record in selected:
                    valid = record["data"]["valid_gt"][:, step]
                    if not valid.any():
                        continue
                    delta = (
                        record["data"]["policy_iou"][:, left, step]
                        - record["data"]["policy_iou"][:, right, step]
                    )
                    values.append(float(delta[valid].mean()))
                values = np.asarray(values, dtype=np.float64)
                interval = _bootstrap(values, bootstrap_samples, rng)
                rows.append(
                    {
                        "split": split,
                        "step": step,
                        "frame_from_anchor": step + 1,
                        "left_policy": left_name,
                        "right_policy": right_name,
                        "left_uses_l9": int(step in schedules[left_name]),
                        "right_uses_l9": int(step in schedules[right_name]),
                        "num_sequences": len(values),
                        "mean_iou_delta": float(values.mean()),
                        "ci_low": interval["ci_low"],
                        "ci_high": interval["ci_high"],
                        "probability_positive": interval[
                            "probability_positive"
                        ],
                    }
                )
    return rows


def primary_decision(paired_rows, missing):
    if missing:
        return {
            "status": "PENDING_INCOMPLETE",
            "reason": "Confirmatory cache is incomplete",
        }
    lookup = {
        (row["split"], row["horizon"], row["left_policy"], row["right_policy"]): row
        for row in paired_rows
    }
    layout = lookup[("val", 20, "uniform4", "burst4")]
    benefit = lookup[("val", 20, "uniform4", "fast")]
    passed = layout["ci_low"] > 0.0 and benefit["ci_low"] > 0.0
    return {
        "status": "PASS" if passed else "FAIL",
        "rule": (
            "Confirmatory H20 Uniform-4 must beat both Burst-4 and Fast "
            "with clustered 95% CI lower bounds above zero"
        ),
        "uniform_minus_burst_mean": layout["mean_delta"],
        "uniform_minus_burst_ci_low": layout["ci_low"],
        "uniform_minus_burst_ci_high": layout["ci_high"],
        "uniform_minus_fast_mean": benefit["mean_delta"],
        "uniform_minus_fast_ci_low": benefit["ci_low"],
        "uniform_minus_fast_ci_high": benefit["ci_high"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2035)
    parser.add_argument("--require_complete", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap_samples must be positive")

    directory, run, records, missing = load_records(
        args.input, args.require_complete
    )
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else directory
    )
    rng = np.random.default_rng(args.seed)
    policies = policy_summary(records, run, args.bootstrap_samples, rng)
    paired = paired_summary(records, run, args.bootstrap_samples, rng)
    post_action = post_action_summary(records, args.bootstrap_samples, rng)
    steps = step_effect_summary(records, run, args.bootstrap_samples, rng)
    decision = primary_decision(paired, missing)
    cache_rows = [record["report"] for record in records]

    paths = {
        "cache": output / "stage9_schedule_cache_summary.csv",
        "policy": output / "stage9_schedule_policy_summary.csv",
        "paired": output / "stage9_schedule_pairwise.csv",
        "post_action": output / "stage9_schedule_post_action.csv",
        "steps": output / "stage9_schedule_step_effects.csv",
        "decision": output / "stage9_schedule_decision.json",
    }
    write_csv(paths["cache"], cache_rows)
    write_csv(paths["policy"], policies)
    write_csv(paths["paired"], paired)
    write_csv(paths["post_action"], post_action)
    write_csv(paths["steps"], steps)
    write_json(
        paths["decision"],
        {
            "run_fingerprint": run["fingerprint"],
            "audited_clips": len(records),
            "expected_clips": len(run["manifest"]["sequences"]),
            "missing_clips": len(missing),
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "primary": decision,
        },
    )

    print(
        "Audited {} of {} Stage-9 clips ({} missing).".format(
            len(records), len(run["manifest"]["sequences"]), len(missing)
        )
    )
    if missing:
        print("Incomplete run: all displayed intervals are diagnostic only.")
    print("\nH20 policy macro AUC:")
    print("split   policy       blocks/f       auc  gain_fast  paired_95%")
    for row in policies:
        if row["horizon"] != 20 or row["split"] == "pooled":
            continue
        interval = "[{:.4f},{:.4f}]".format(
            row["gain_vs_fast_ci_low"], row["gain_vs_fast_ci_high"]
        )
        print(
            "{split:<7s} {policy:<11s} {candidate_blocks_per_frame:>8.2%} "
            "{auc_macro:>9.4f} {gain_vs_fast:>10.4f}  {interval}".format(
                interval=interval, **row
            )
        )
    print("\nEqual-cost H20 paired layout comparisons:")
    print("split   comparison                 delta  paired_95%           W/T/L")
    for row in paired:
        if (
            row["horizon"] != 20
            or not row["equal_cost_at_horizon"]
            or row["split"] == "pooled"
        ):
            continue
        interval = "[{:.4f},{:.4f}]".format(row["ci_low"], row["ci_high"])
        print(
            "{split:<7s} {left_policy:<11s}-{right_policy:<11s} "
            "{mean_delta:>7.4f}  {interval:<19s} "
            "{wins}/{ties}/{losses}".format(interval=interval, **row)
        )
    print("\nPredeclared confirmatory decision: {}".format(decision["status"]))
    print("Policy summary CSV: {}".format(paths["policy"]))
    print("Paired summary CSV: {}".format(paths["paired"]))
    print("Post-action CSV: {}".format(paths["post_action"]))
    print("Step-effect CSV: {}".format(paths["steps"]))
    print("Decision JSON: {}".format(paths["decision"]))
    print(
        "Note: pooled rows are descriptive; the frozen confirmatory claim uses "
        "only the val split."
    )


if __name__ == "__main__":
    main()
