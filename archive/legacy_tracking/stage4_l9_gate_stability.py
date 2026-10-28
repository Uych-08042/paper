"""Validate fixed-threshold Fast/L9 gates and their video-level stability.

The script does not retrain a router or run the tracker.  It loads deployable
Stage-4D models, calibrates score thresholds from training scores only, and
applies those fixed thresholds to untouched validation trajectories.  It also
reports sequence-level bootstrap intervals and circular-shift permutation
tests that preserve each trajectory's temporal score pattern.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint, write_csv, write_json
from stage4_l9_gate_train import (
    MODEL_SCHEMA,
    L9_DELTA_OFFSET,
    build_variant_low_features,
    empty_projection,
    l9_latency,
    uses_tokens,
    weighted_average_precision,
)
from stage4_router_train import (
    _finite_or_blank,
    _weighted_binary_auc,
    _weighted_mean,
    _weighted_pearson,
    balanced_group_weights,
    load_cache_dataset,
    subset_data,
    transform_combined,
    transform_token_pca,
)


DEFAULT_MODELS = (
    "history_only__signed",
    "history_only__positive",
    "fast_only__signed",
    "fast_only__positive",
)
TOLERANCE = 1e-12


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_fingerprint(value, description):
    recorded = value.get("fingerprint")
    body = {key: item for key, item in value.items() if key != "fingerprint"}
    if recorded is None or fingerprint(body) != recorded:
        raise ValueError("{} fingerprint mismatch".format(description))


def load_gate_model(router_dir, model_name, run):
    path = Path(router_dir) / "stage4_l9_gate_{}.npz".format(model_name)
    if not path.is_file():
        raise FileNotFoundError("Missing Stage-4D model: {}".format(path))
    with np.load(str(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {key: archive[key].copy() for key in archive.files
                  if key != "metadata_json"}
    verify_fingerprint(metadata, "Model {}".format(model_name))
    if (metadata.get("schema") != MODEL_SCHEMA
            or metadata.get("model_id") != model_name
            or metadata.get("cache_fingerprint") != run["fingerprint"]
            or metadata.get("manifest_fingerprint") != run["manifest"]["fingerprint"]
            or metadata.get("action") != "layer9"):
        raise ValueError("Model/cache identity mismatch: {}".format(model_name))
    return path, metadata, arrays


def model_scores(data, metadata, arrays, batch_size=1024):
    variant = metadata["variant"]
    low, low_names = build_variant_low_features(data, variant)
    stored_names = tuple(arrays["low_feature_names"].astype(str).tolist())
    if tuple(low_names) != stored_names:
        raise ValueError("Low-feature schema mismatch for {}".format(
            metadata["model_id"]))
    components = arrays["pca_components"]
    if uses_tokens(variant):
        if components.ndim != 2 or components.shape[0] == 0:
            raise ValueError("Token model has no PCA components")
        projection = transform_token_pca(data["token_features"], {
            "mean": arrays["token_mean"],
            "scale": arrays["token_scale"],
            "components": components,
        }, batch_size)
    else:
        if components.shape[0] != 0:
            raise ValueError("Token-free model unexpectedly contains PCA components")
        projection = empty_projection(len(low))
    features = transform_combined(projection, low, {
        "mean": arrays["combined_mean"],
        "scale": arrays["combined_scale"],
    })
    coefficient = np.asarray(arrays["ridge_coefficient"], dtype=np.float64)
    intercept = float(np.asarray(
        arrays["ridge_intercept"], dtype=np.float64).item())
    if coefficient.ndim != 1 or coefficient.shape[0] != features.shape[1]:
        raise ValueError("Ridge coefficient shape mismatch")
    scores = features.dot(coefficient) + intercept
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite gate scores")
    return scores


def audit_saved_val_scores(router_dir, model_name, val, scores):
    rows = read_csv(Path(router_dir) / "stage4_l9_gate_val_predictions.csv")
    rows = [row for row in rows if row["model_id"] == model_name]
    lookup = {
        (row["sequence"], row["trajectory"], int(row["frame_id"])):
        float(row["score"])
        for row in rows
    }
    keys = [(str(sequence), str(trajectory), int(frame_id))
            for sequence, trajectory, frame_id in zip(
                val["sequence"], val["trajectory"], val["frame_id"])]
    if len(rows) != len(lookup) or set(keys) != set(lookup):
        raise ValueError("Saved validation predictions do not match cache: {}".format(
            model_name))
    saved = np.asarray([lookup[key] for key in keys], dtype=np.float64)
    if not np.allclose(saved, scores, rtol=1e-10, atol=1e-12):
        raise ValueError("Reconstructed predictions differ from Stage-4D CSV")


def budget_threshold(scores, maximum_percent, natural_threshold):
    """Choose a strict fixed threshold using only the calibration score array."""
    scores = np.asarray(scores, dtype=np.float64)
    maximum = float(maximum_percent) / 100.0
    if maximum <= 0.0:
        return float("inf")
    count = int(math.floor(len(scores) * maximum + 1e-12))
    if count <= 0:
        return float("inf")
    if count >= len(scores):
        return float(natural_threshold)
    ordered = np.sort(scores)[::-1]
    upper = float(ordered[count - 1])
    lower = float(ordered[count])
    boundary = 0.5 * (upper + lower) if upper > lower else upper
    return max(float(natural_threshold), boundary)


def policy_statistics(data, deep, weights):
    deep = np.asarray(deep, dtype=np.bool_)
    delta = np.asarray(data["delta"][:, L9_DELTA_OFFSET], dtype=np.float64)
    fast = np.asarray(data["iou"][:, 0], dtype=np.float64)
    selected_delta = np.where(deep, delta, 0.0)
    gain = _weighted_mean(selected_delta, weights)
    fixed_gain = _weighted_mean(delta, weights)
    ratio = float(np.mean(deep))
    random_gain = ratio * fixed_gain
    selected_weights = weights[deep]
    return {
        "deep_frames": int(deep.sum()),
        "deep_ratio": ratio,
        "fast_mean_iou": _weighted_mean(fast, weights),
        "current_state_mean_iou": _weighted_mean(fast + selected_delta, weights),
        "gain_vs_fast": gain,
        "fixed_l9_gain_vs_fast": fixed_gain,
        "gain_vs_fixed_l9": gain - fixed_gain,
        "random_l9_expected_gain": random_gain,
        "lift_over_random_l9": gain - random_gain,
        "selected_beneficial_ratio": (
            _weighted_mean(delta[deep] > 0.0, selected_weights)
            if np.any(deep) else float("nan")),
    }


def score_statistics(model_name, metadata, data, scores, split):
    rows = []
    for trajectory in ("all",) + tuple(sorted(set(data["trajectory"].tolist()))):
        mask = (np.ones(len(scores), dtype=np.bool_) if trajectory == "all"
                else data["trajectory"] == trajectory)
        current = subset_data(data, mask)
        current_scores = scores[mask]
        weights = balanced_group_weights(
            current["sequence"], current["trajectory"])
        delta = current["delta"][:, L9_DELTA_OFFSET]
        mean_score = _weighted_mean(current_scores, weights)
        variance = _weighted_mean((current_scores - mean_score) ** 2, weights)
        rows.append({
            "model_id": model_name,
            "variant": metadata["variant"],
            "objective": metadata["objective"],
            "split": split,
            "trajectory": trajectory,
            "examples": len(delta),
            "sequences": len(set(current["sequence"].tolist())),
            "l9_beneficial_ratio": _weighted_mean(delta > 0.0, weights),
            "l9_benefit_auroc": _weighted_binary_auc(
                delta > 0.0, current_scores, weights),
            "l9_benefit_average_precision": weighted_average_precision(
                delta > 0.0, current_scores, weights),
            "score_delta_pearson": _weighted_pearson(
                current_scores, delta, weights),
            "score_mean": mean_score,
            "score_std": math.sqrt(max(variance, 0.0)),
        })
    return rows


def group_policy_rows(model_name, metadata, data, deep, budget):
    rows = []
    delta = np.asarray(data["delta"][:, L9_DELTA_OFFSET], dtype=np.float64)
    for sequence in sorted(set(data["sequence"].tolist())):
        for trajectory in sorted(set(data["trajectory"].tolist())):
            mask = ((data["sequence"] == sequence)
                    & (data["trajectory"] == trajectory))
            if not np.any(mask):
                continue
            current_delta = delta[mask]
            current_deep = deep[mask]
            ratio = float(np.mean(current_deep))
            fixed_gain = float(np.mean(current_delta))
            gain = float(np.mean(np.where(current_deep, current_delta, 0.0)))
            rows.append({
                "model_id": model_name,
                "variant": metadata["variant"],
                "objective": metadata["objective"],
                "budget_percent": float(budget),
                "sequence": str(sequence),
                "trajectory": str(trajectory),
                "frames": int(mask.sum()),
                "deep_frames": int(current_deep.sum()),
                "deep_ratio": ratio,
                "gain_vs_fast": gain,
                "fixed_l9_gain_vs_fast": fixed_gain,
                "random_l9_expected_gain": ratio * fixed_gain,
                "lift_over_random_l9": gain - ratio * fixed_gain,
                "selected_beneficial_ratio": (
                    float(np.mean(current_delta[current_deep] > 0.0))
                    if np.any(current_deep) else ""),
            })
    return rows


def sequence_records(group_rows):
    records = []
    for sequence in sorted(set(row["sequence"] for row in group_rows)):
        current = [row for row in group_rows if row["sequence"] == sequence]
        records.append({
            "sequence": sequence,
            "gain": float(np.mean([row["gain_vs_fast"] for row in current])),
            "fixed_gain": float(np.mean([
                row["fixed_l9_gain_vs_fast"] for row in current])),
            "frames": int(sum(row["frames"] for row in current)),
            "deep_frames": int(sum(row["deep_frames"] for row in current)),
        })
    return records


def bootstrap_statistics(records, samples, seed):
    generator = np.random.RandomState(seed)
    count = len(records)
    gains = np.empty(samples, dtype=np.float64)
    lifts = np.empty(samples, dtype=np.float64)
    ratios = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        chosen = generator.randint(0, count, size=count)
        current = [records[index] for index in chosen]
        gain = float(np.mean([record["gain"] for record in current]))
        fixed_gain = float(np.mean([
            record["fixed_gain"] for record in current]))
        ratio = (sum(record["deep_frames"] for record in current)
                 / max(sum(record["frames"] for record in current), 1))
        gains[sample] = gain
        ratios[sample] = ratio
        lifts[sample] = gain - ratio * fixed_gain
    return {
        "gain_ci_low": float(np.quantile(gains, 0.025)),
        "gain_ci_high": float(np.quantile(gains, 0.975)),
        "gain_probability_positive": float(np.mean(gains > 0.0)),
        "lift_ci_low": float(np.quantile(lifts, 0.025)),
        "lift_ci_high": float(np.quantile(lifts, 0.975)),
        "lift_probability_positive": float(np.mean(lifts > 0.0)),
        "deep_ratio_ci_low": float(np.quantile(ratios, 0.025)),
        "deep_ratio_ci_high": float(np.quantile(ratios, 0.975)),
    }


def circular_shift_test(data, deep, observed_lift, permutations, seed):
    generator = np.random.RandomState(seed)
    delta = np.asarray(data["delta"][:, L9_DELTA_OFFSET], dtype=np.float64)
    groups = []
    for sequence in sorted(set(data["sequence"].tolist())):
        for trajectory in sorted(set(data["trajectory"].tolist())):
            mask = ((data["sequence"] == sequence)
                    & (data["trajectory"] == trajectory))
            if np.any(mask):
                groups.append((delta[mask], np.asarray(deep[mask], dtype=np.bool_)))
    fixed_gain = float(np.mean([values.mean() for values, _ in groups]))
    total_frames = sum(len(values) for values, _ in groups)
    total_deep = sum(int(mask.sum()) for _, mask in groups)
    random_gain = (total_deep / max(total_frames, 1)) * fixed_gain
    null_lifts = np.empty(permutations, dtype=np.float64)
    for permutation in range(permutations):
        gains = []
        for values, mask in groups:
            shift = int(generator.randint(0, len(mask))) if len(mask) > 1 else 0
            shifted = np.roll(mask, shift)
            gains.append(float(np.mean(np.where(shifted, values, 0.0))))
        null_lifts[permutation] = float(np.mean(gains)) - random_gain
    return {
        "null_lift_mean": float(null_lifts.mean()),
        "null_lift_p95": float(np.quantile(null_lifts, 0.95)),
        "one_sided_p_value": float(
            (1 + np.sum(null_lifts >= observed_lift - TOLERANCE))
            / (permutations + 1)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Complete audited Stage-4B cache")
    parser.add_argument("--router_dir", required=True, help="Completed Stage-4D output")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--budgets", nargs="+", type=float, default=[10, 20, 30, 50])
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (any(value <= 0.0 or value > 100.0 for value in args.budgets)
            or args.bootstrap_samples < 100 or args.permutations < 100
            or args.batch_size < 1):
        raise ValueError("Invalid budget, resampling or batch-size argument")
    if len(set(args.models)) != len(args.models) or len(set(args.budgets)) != len(args.budgets):
        raise ValueError("Models and budgets must be unique")

    router_dir = Path(args.router_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    input_dir = Path(args.input).expanduser().resolve()
    try:
        output_dir.relative_to(input_dir)
    except ValueError:
        pass
    else:
        raise ValueError("--output_dir must be outside the audited cache directory")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError("Use a new --output_dir or pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_path = router_dir / "stage4_l9_gate_experiment.json"
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    verify_fingerprint(experiment, "Stage-4D experiment")
    data, run = load_cache_dataset(input_dir)
    if experiment.get("cache_fingerprint") != run["fingerprint"]:
        raise ValueError("Stage-4D output and cache fingerprints disagree")
    train = subset_data(data, data["split"] == "train")
    val = subset_data(data, data["split"] == "val")
    train_weights = balanced_group_weights(train["sequence"], train["trajectory"])
    val_weights = balanced_group_weights(val["sequence"], val["trajectory"])
    exact_rows = read_csv(router_dir / "stage4_l9_gate_budget.csv")
    exact_lookup = {
        (row["model_id"], float(row["budget_percent"])): row
        for row in exact_rows
    }
    costs = experiment.get("profile_means_ms")

    score_rows = []
    online_rows = []
    group_rows = []
    bootstrap_rows = []
    permutation_rows = []
    model_reports = []
    print("Loaded {} train / {} val examples from {} videos.".format(
        len(train["delta"]), len(val["delta"]),
        len(set(data["sequence"].tolist()))))

    for model_index, model_name in enumerate(args.models):
        model_path, metadata, arrays = load_gate_model(router_dir, model_name, run)
        train_scores = model_scores(train, metadata, arrays, args.batch_size)
        val_scores = model_scores(val, metadata, arrays, args.batch_size)
        audit_saved_val_scores(router_dir, model_name, val, val_scores)
        score_rows.extend(score_statistics(
            model_name, metadata, train, train_scores, "train"))
        score_rows.extend(score_statistics(
            model_name, metadata, val, val_scores, "val"))
        model_reports.append({
            "model_id": model_name,
            "model_fingerprint": metadata["fingerprint"],
            "model_sha256": file_hash(model_path),
        })

        for budget_index, budget in enumerate(args.budgets):
            threshold = budget_threshold(
                train_scores, budget, metadata["natural_score_threshold"])
            train_deep = train_scores > threshold
            val_deep = val_scores > threshold
            train_stats = policy_statistics(train, train_deep, train_weights)
            val_stats = policy_statistics(val, val_deep, val_weights)
            exact = exact_lookup.get((model_name, float(budget)))
            if exact is None:
                raise ValueError("Stage-4D exact budget row missing for {} at {}".format(
                    model_name, budget))
            latency = l9_latency(metadata["variant"], val_deep, costs)
            online_rows.append({
                "model_id": model_name,
                "variant": metadata["variant"],
                "objective": metadata["objective"],
                "budget_percent": float(budget),
                "training_score_threshold": _finite_or_blank(threshold),
                "train_deep_ratio": train_stats["deep_ratio"],
                "val_deep_ratio": val_stats["deep_ratio"],
                "val_budget_drift_percentage_points": 100.0 * (
                    val_stats["deep_ratio"] - float(budget) / 100.0),
                "val_deep_frames": val_stats["deep_frames"],
                "val_current_state_mean_iou": val_stats["current_state_mean_iou"],
                "val_gain_vs_fast": val_stats["gain_vs_fast"],
                "val_gain_vs_fixed_l9": val_stats["gain_vs_fixed_l9"],
                "val_random_l9_expected_gain": val_stats["random_l9_expected_gain"],
                "val_lift_over_random_l9": val_stats["lift_over_random_l9"],
                "val_selected_beneficial_ratio": _finite_or_blank(
                    val_stats["selected_beneficial_ratio"]),
                "exact_topk_val_gain_vs_fast": float(exact["gain_vs_fast"]),
                "exact_topk_val_lift_over_random_l9": float(
                    exact["lift_over_random_l9"]),
                "online_gain_gap_to_exact_topk": (
                    val_stats["gain_vs_fast"] - float(exact["gain_vs_fast"])),
                "estimated_latency_ms_excluding_gate": _finite_or_blank(latency),
                "estimated_fps_excluding_gate": (
                    _finite_or_blank(1000.0 / latency)
                    if np.isfinite(latency) and latency > 0.0 else ""),
            })
            current_groups = group_policy_rows(
                model_name, metadata, val, val_deep, budget)
            group_rows.extend(current_groups)
            records = sequence_records(current_groups)
            bootstrap = bootstrap_statistics(
                records, args.bootstrap_samples,
                args.seed + model_index * 1009 + budget_index * 37)
            bootstrap_rows.append({
                "model_id": model_name,
                "variant": metadata["variant"],
                "objective": metadata["objective"],
                "budget_percent": float(budget),
                "sequences": len(records),
                "bootstrap_samples": args.bootstrap_samples,
                **bootstrap,
            })
            permutation = circular_shift_test(
                val, val_deep, val_stats["lift_over_random_l9"],
                args.permutations,
                args.seed + 100000 + model_index * 1009 + budget_index * 37)
            permutation_rows.append({
                "model_id": model_name,
                "variant": metadata["variant"],
                "objective": metadata["objective"],
                "budget_percent": float(budget),
                "permutations": args.permutations,
                "observed_lift_over_random_l9": val_stats["lift_over_random_l9"],
                **permutation,
            })

    paths = {
        "score": output_dir / "stage4_l9_gate_score_stability.csv",
        "online": output_dir / "stage4_l9_gate_online_budget.csv",
        "groups": output_dir / "stage4_l9_gate_online_groups.csv",
        "bootstrap": output_dir / "stage4_l9_gate_bootstrap.csv",
        "permutation": output_dir / "stage4_l9_gate_permutation.csv",
    }
    for key, rows in (("score", score_rows), ("online", online_rows),
                      ("groups", group_rows), ("bootstrap", bootstrap_rows),
                      ("permutation", permutation_rows)):
        write_csv(paths[key], rows)
    report = {
        "schema": "stage4_l9_gate_stability_v1",
        "cache_fingerprint": run["fingerprint"],
        "stage4d_experiment_fingerprint": experiment["fingerprint"],
        "script_sha256": file_hash(Path(__file__).resolve()),
        "configuration": vars(args),
        "models": model_reports,
        "outputs": {key: str(path) for key, path in paths.items()},
        "warning": (
            "Thresholds use training scores only. Validation labels are used for "
            "evaluation, video bootstrap, and circular-shift significance tests. "
            "Results remain frozen-trajectory current-state diagnostics."),
    }
    report["fingerprint"] = fingerprint(report)
    write_json(output_dir / "stage4_l9_gate_stability.json", report)

    print("\nFixed training-threshold validation:")
    print("model                            budget  train%    val%  gain_fast  "
          "lift_random  boot_lift_95%          p_shift  latency")
    bootstrap_lookup = {
        (row["model_id"], row["budget_percent"]): row for row in bootstrap_rows
    }
    permutation_lookup = {
        (row["model_id"], row["budget_percent"]): row for row in permutation_rows
    }
    for row in online_rows:
        key = (row["model_id"], row["budget_percent"])
        boot = bootstrap_lookup[key]
        permutation = permutation_lookup[key]
        interval = "[{:.5f},{:.5f}]".format(
            boot["lift_ci_low"], boot["lift_ci_high"])
        print("{model_id:<32} {budget_percent:>5.1f}%  {train_deep_ratio:>6.2%}  "
              "{val_deep_ratio:>6.2%}  {val_gain_vs_fast:>9.6f}  "
              "{val_lift_over_random_l9:>11.6f}  {interval:<21}  "
              "{p_value:>7.4f}  {latency!s}".format(
                  interval=interval,
                  p_value=permutation["one_sided_p_value"],
                  latency=row["estimated_latency_ms_excluding_gate"], **row))
    print("Reports: {}".format(output_dir))


if __name__ == "__main__":
    main()
