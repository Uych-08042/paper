"""Evaluate a causal per-video budget controller for Fast-versus-L9 routing.

The controller uses only scores observed earlier in the same trajectory.  A
token bucket enforces a prefix budget while a rolling empirical quantile moves
eligible high-score frames earlier than a score-agnostic periodic baseline.
This is still a frozen-trajectory diagnostic; no tracker or GPU is used.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint, write_csv, write_json
from stage4_l9_gate_stability import (
    audit_saved_val_scores,
    bootstrap_statistics,
    group_policy_rows,
    load_gate_model,
    model_scores,
    policy_statistics,
    read_csv,
    sequence_records,
    verify_fingerprint,
)
from stage4_l9_gate_train import L9_DELTA_OFFSET, l9_latency
from stage4_router_train import (
    _finite_or_blank,
    balanced_group_weights,
    load_cache_dataset,
    subset_data,
)


DEFAULT_MODELS = ("history_only__signed", "history_only__positive")
TOLERANCE = 1e-10


def grouped_indices(data):
    groups = []
    for sequence in sorted(set(data["sequence"].tolist())):
        for trajectory in sorted(set(data["trajectory"].tolist())):
            indices = np.flatnonzero(
                (data["sequence"] == sequence)
                & (data["trajectory"] == trajectory))
            if len(indices):
                order = np.argsort(data["frame_id"][indices], kind="mergesort")
                groups.append(indices[order])
    return groups


def periodic_schedule(data, budget_percent):
    rate = float(budget_percent) / 100.0
    deep = np.zeros(len(data["frame_id"]), dtype=np.bool_)
    for indices in grouped_indices(data):
        credit = 0.0
        used = 0
        for step, index in enumerate(indices):
            credit += rate
            if credit + TOLERANCE >= 1.0:
                deep[index] = True
                credit -= 1.0
                used += 1
            if used > math.floor((step + 1) * rate + TOLERANCE):
                raise AssertionError("Periodic controller exceeded prefix budget")
    return deep


def matched_periodic_schedule(data, reference_deep):
    """Spread exactly the reference decision count over each trajectory."""
    reference_deep = np.asarray(reference_deep, dtype=np.bool_)
    deep = np.zeros(len(reference_deep), dtype=np.bool_)
    for indices in grouped_indices(data):
        target = int(reference_deep[indices].sum())
        if target == 0:
            continue
        rate = target / float(len(indices))
        credit = 0.0
        for index in indices:
            credit += rate
            if credit + TOLERANCE >= 1.0:
                deep[index] = True
                credit -= 1.0
        if int(deep[indices].sum()) != target:
            raise AssertionError("Matched periodic controller count mismatch")
    return deep


def causal_quantile_schedule(data, scores, budget_percent, window,
                             min_history, bucket_capacity):
    """Route from past-score quantiles with bounded deferred budget credit."""
    rate = float(budget_percent) / 100.0
    scores = np.asarray(scores, dtype=np.float64)
    deep = np.zeros(len(scores), dtype=np.bool_)
    for indices in grouped_indices(data):
        history = []
        credit = 0.0
        used = 0
        for step, index in enumerate(indices):
            credit = min(float(bucket_capacity), credit + rate)
            eligible = False
            if len(history) >= min_history:
                reference = history[-window:] if window > 0 else history
                threshold = float(np.quantile(reference, 1.0 - rate))
                eligible = bool(scores[index] > threshold)
            forced = credit >= float(bucket_capacity) - TOLERANCE
            if credit + TOLERANCE >= 1.0 and (eligible or forced):
                deep[index] = True
                credit -= 1.0
                used += 1
            history.append(float(scores[index]))
            if used > math.floor((step + 1) * rate + TOLERANCE):
                raise AssertionError("Causal controller exceeded prefix budget")
    return deep


def paired_sequence_records(data, first_deep, second_deep):
    delta = np.asarray(
        data["delta"][:, L9_DELTA_OFFSET], dtype=np.float64)
    records = []
    for sequence in sorted(set(data["sequence"].tolist())):
        first_gains = []
        second_gains = []
        for trajectory in sorted(set(data["trajectory"].tolist())):
            mask = ((data["sequence"] == sequence)
                    & (data["trajectory"] == trajectory))
            if not np.any(mask):
                continue
            first_gains.append(float(np.mean(np.where(
                first_deep[mask], delta[mask], 0.0))))
            second_gains.append(float(np.mean(np.where(
                second_deep[mask], delta[mask], 0.0))))
        records.append({
            "sequence": str(sequence),
            "first_gain": float(np.mean(first_gains)),
            "second_gain": float(np.mean(second_gains)),
        })
    return records


def paired_bootstrap(records, samples, seed):
    generator = np.random.RandomState(seed)
    count = len(records)
    differences = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        selected = generator.randint(0, count, size=count)
        differences[sample] = float(np.mean([
            records[index]["first_gain"] - records[index]["second_gain"]
            for index in selected
        ]))
    return {
        "gain_vs_periodic_matched_ci_low": float(
            np.quantile(differences, 0.025)),
        "gain_vs_periodic_matched_ci_high": float(
            np.quantile(differences, 0.975)),
        "gain_vs_periodic_matched_probability_positive": float(
            np.mean(differences > 0.0)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Complete audited Stage-4B cache")
    parser.add_argument("--router_dir", required=True, help="Completed Stage-4D output")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--budgets", nargs="+", type=float, default=[10, 20, 30, 50])
    parser.add_argument("--window", type=int, default=32,
                        help="Past-score window; 0 uses all previous scores")
    parser.add_argument("--min_history", type=int, default=10)
    parser.add_argument("--bucket_capacity", type=float, default=2.0)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (any(value <= 0.0 or value >= 100.0 for value in args.budgets)
            or args.window < 0 or args.min_history < 1
            or (args.window and args.min_history > args.window)
            or args.bucket_capacity < 1.0
            or args.bootstrap_samples < 100 or args.batch_size < 1):
        raise ValueError("Invalid controller, budget, bootstrap or batch-size argument")
    if len(set(args.models)) != len(args.models) or len(set(args.budgets)) != len(args.budgets):
        raise ValueError("Models and budgets must be unique")

    input_dir = Path(args.input).expanduser().resolve()
    router_dir = Path(args.router_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    try:
        output_dir.relative_to(input_dir)
    except ValueError:
        pass
    else:
        raise ValueError("--output_dir must be outside the audited cache directory")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError("Use a new --output_dir or pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment = json.loads(
        (router_dir / "stage4_l9_gate_experiment.json").read_text(encoding="utf-8"))
    verify_fingerprint(experiment, "Stage-4D experiment")
    data, run = load_cache_dataset(input_dir)
    if experiment.get("cache_fingerprint") != run["fingerprint"]:
        raise ValueError("Stage-4D output and cache fingerprints disagree")
    val = subset_data(data, data["split"] == "val")
    weights = balanced_group_weights(val["sequence"], val["trajectory"])
    costs = experiment.get("profile_means_ms")
    exact_lookup = {
        (row["model_id"], float(row["budget_percent"])): row
        for row in read_csv(router_dir / "stage4_l9_gate_budget.csv")
    }

    summary_rows = []
    group_rows = []
    bootstrap_rows = []
    model_reports = []
    print("Loaded {} validation examples from {} videos.".format(
        len(val["delta"]), len(set(val["sequence"].tolist()))))
    for model_index, model_name in enumerate(args.models):
        model_path, metadata, arrays = load_gate_model(
            router_dir, model_name, run)
        scores = model_scores(val, metadata, arrays, args.batch_size)
        audit_saved_val_scores(router_dir, model_name, val, scores)
        model_reports.append({
            "model_id": model_name,
            "model_fingerprint": metadata["fingerprint"],
            "model_sha256": file_hash(model_path),
        })
        for budget_index, budget in enumerate(args.budgets):
            periodic_budget = periodic_schedule(val, budget)
            causal = causal_quantile_schedule(
                val, scores, budget, args.window, args.min_history,
                args.bucket_capacity)
            periodic_matched = matched_periodic_schedule(val, causal)
            periodic_budget_stats = policy_statistics(
                val, periodic_budget, weights)
            periodic_matched_stats = policy_statistics(
                val, periodic_matched, weights)
            causal_stats = policy_statistics(val, causal, weights)
            exact = exact_lookup.get((model_name, float(budget)))
            if exact is None:
                raise ValueError("Missing exact-top-k row for {} at {}".format(
                    model_name, budget))
            for controller, deep, statistics in (
                    ("periodic_budget", periodic_budget,
                     periodic_budget_stats),
                    ("periodic_matched", periodic_matched,
                     periodic_matched_stats),
                    ("causal_quantile", causal, causal_stats)):
                latency_variant = (
                    "history_only" if controller.startswith("periodic")
                    else metadata["variant"])
                latency = l9_latency(latency_variant, deep, costs)
                summary_rows.append({
                    "model_id": model_name,
                    "variant": metadata["variant"],
                    "objective": metadata["objective"],
                    "controller": controller,
                    "budget_percent": float(budget),
                    "window": args.window if controller == "causal_quantile" else "",
                    "min_history": (args.min_history
                                    if controller == "causal_quantile" else ""),
                    "bucket_capacity": (args.bucket_capacity
                                        if controller == "causal_quantile" else ""),
                    "deep_frames": statistics["deep_frames"],
                    "deep_ratio": statistics["deep_ratio"],
                    "current_state_mean_iou": statistics["current_state_mean_iou"],
                    "gain_vs_fast": statistics["gain_vs_fast"],
                    "random_l9_expected_gain": statistics["random_l9_expected_gain"],
                    "lift_over_random_l9": statistics["lift_over_random_l9"],
                    "gain_vs_periodic_budget": (
                        statistics["gain_vs_fast"]
                        - periodic_budget_stats["gain_vs_fast"]),
                    "gain_vs_periodic_matched": (
                        statistics["gain_vs_fast"]
                        - periodic_matched_stats["gain_vs_fast"]),
                    "selected_beneficial_ratio": _finite_or_blank(
                        statistics["selected_beneficial_ratio"]),
                    "exact_topk_gain_vs_fast": float(exact["gain_vs_fast"]),
                    "gain_gap_to_exact_topk": (statistics["gain_vs_fast"]
                                               - float(exact["gain_vs_fast"])),
                    "estimated_latency_ms_excluding_controller": _finite_or_blank(
                        latency),
                    "estimated_fps_excluding_controller": (
                        _finite_or_blank(1000.0 / latency)
                        if np.isfinite(latency) and latency > 0.0 else ""),
                })
                current_groups = group_policy_rows(
                    model_name, metadata, val, deep, budget)
                for row in current_groups:
                    row["controller"] = controller
                group_rows.extend(current_groups)

            causal_groups = group_policy_rows(
                model_name, metadata, val, causal, budget)
            records = sequence_records(causal_groups)
            causal_bootstrap = bootstrap_statistics(
                records, args.bootstrap_samples,
                args.seed + model_index * 1009 + budget_index * 37)
            paired = paired_bootstrap(
                paired_sequence_records(val, causal, periodic_matched),
                args.bootstrap_samples,
                args.seed + 100000 + model_index * 1009 + budget_index * 37)
            bootstrap_rows.append({
                "model_id": model_name,
                "variant": metadata["variant"],
                "objective": metadata["objective"],
                "budget_percent": float(budget),
                "sequences": len(records),
                "bootstrap_samples": args.bootstrap_samples,
                **causal_bootstrap,
                **paired,
            })

    paths = {
        "summary": output_dir / "stage4_l9_online_controller_summary.csv",
        "groups": output_dir / "stage4_l9_online_controller_groups.csv",
        "bootstrap": output_dir / "stage4_l9_online_controller_bootstrap.csv",
    }
    for key, rows in (("summary", summary_rows), ("groups", group_rows),
                      ("bootstrap", bootstrap_rows)):
        write_csv(paths[key], rows)
    report = {
        "schema": "stage4_l9_online_controller_v1",
        "cache_fingerprint": run["fingerprint"],
        "stage4d_experiment_fingerprint": experiment["fingerprint"],
        "script_sha256": file_hash(Path(__file__).resolve()),
        "configuration": vars(args),
        "models": model_reports,
        "outputs": {key: str(path) for key, path in paths.items()},
        "warning": (
            "The controller uses only earlier scores in each trajectory and "
            "never reads validation labels, but reported accuracy remains a "
            "frozen-trajectory current-state diagnostic."),
    }
    report["fingerprint"] = fingerprint(report)
    write_json(output_dir / "stage4_l9_online_controller.json", report)

    bootstrap_lookup = {
        (row["model_id"], row["budget_percent"]): row for row in bootstrap_rows
    }
    print("\nCausal online controller versus periodic L9:")
    print("model                          budget  actual  gain_fast  gain_matched  "
          "paired_95%              p_positive  exact_gap  latency")
    for row in summary_rows:
        if row["controller"] != "causal_quantile":
            continue
        boot = bootstrap_lookup[(row["model_id"], row["budget_percent"])]
        interval = "[{:.5f},{:.5f}]".format(
            boot["gain_vs_periodic_matched_ci_low"],
            boot["gain_vs_periodic_matched_ci_high"])
        print("{model_id:<30} {budget_percent:>5.1f}%  {deep_ratio:>6.2%}  "
              "{gain_vs_fast:>9.6f}  {gain_vs_periodic_matched:>12.6f}  "
              "{interval:<21}  {probability:>10.3f}  {gap:>9.6f}  {latency!s}".format(
                  interval=interval,
                  probability=boot[
                      "gain_vs_periodic_matched_probability_positive"],
                  gap=row["gain_gap_to_exact_topk"],
                  latency=row["estimated_latency_ms_excluding_controller"], **row))
    print("Reports: {}".format(output_dir))


if __name__ == "__main__":
    main()
