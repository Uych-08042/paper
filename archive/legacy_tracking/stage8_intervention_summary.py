"""Audit and summarize Stage-8B single-L9 causal intervention caches."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from stage6_budget_oracle import write_csv
from stage6_temporal_stability import bootstrap_mean
from stage8_intervention_common import (
    intervention_cache_path,
    load_and_audit_cache,
    load_intervention_run,
)


def load_records(directory, require_complete=False):
    directory, run = load_intervention_run(directory)
    entries = run["manifest"]["sequences"]
    expected = {intervention_cache_path(directory, entry) for entry in entries}
    found = set(directory.rglob("*.npz"))
    extra = found.difference(expected)
    missing = expected.difference(found)
    if extra:
        raise ValueError(
            "Intervention directory has {} unexpected cache(s): {}".format(
                len(extra), [str(path) for path in sorted(extra)[:3]]
            )
        )
    if require_complete and missing:
        raise ValueError(
            "Intervention directory is incomplete: {} of {} cache(s) missing".format(
                len(missing), len(expected)
            )
        )
    records = []
    for entry in entries:
        path = intervention_cache_path(directory, entry)
        if not path.is_file():
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
    if not records:
        raise ValueError("No completed Stage-8B caches found")
    return directory, run, records, len(missing), len(expected)


def anchor_rows(records, horizons):
    output = []
    for record in records:
        entry = record["entry"]
        data = record["data"]
        for anchor_index, offset in enumerate(data["anchor_offset"]):
            for horizon_index, horizon in enumerate(horizons):
                output.append(
                    {
                        "split": entry["split"],
                        "sequence": entry["sequence"],
                        "anchor_index": anchor_index,
                        "frame_id": int(data["frame_id"][anchor_index]),
                        "anchor_offset": int(offset),
                        "anchor_fraction": float(
                            data["anchor_fraction"][anchor_index]
                        ),
                        "horizon": int(horizon),
                        "valid_frames": int(
                            data["valid_frames_by_horizon"][
                                anchor_index, horizon_index
                            ]
                        ),
                        "control_mean_iou": float(
                            data["control_mean_iou"][
                                anchor_index, horizon_index
                            ]
                        ),
                        "treatment_mean_iou": float(
                            data["treatment_mean_iou"][
                                anchor_index, horizon_index
                            ]
                        ),
                        "mean_iou_delta": float(
                            data["mean_iou_delta"][
                                anchor_index, horizon_index
                            ]
                        ),
                        "control_auc": float(
                            data["control_auc"][anchor_index, horizon_index]
                        ),
                        "treatment_auc": float(
                            data["treatment_auc"][anchor_index, horizon_index]
                        ),
                        "auc_delta": float(
                            data["auc_delta"][anchor_index, horizon_index]
                        ),
                        "post_anchor_mean_iou_delta": float(
                            data["post_anchor_mean_iou_delta"][
                                anchor_index, horizon_index
                            ]
                        ),
                        "post_anchor_auc_delta": float(
                            data["post_anchor_auc_delta"][
                                anchor_index, horizon_index
                            ]
                        ),
                        "terminal_iou_delta": float(
                            data["terminal_iou_delta"][
                                anchor_index, horizon_index
                            ]
                        ),
                        "failure_reduction_02": float(
                            data["failure_reduction_02"][
                                anchor_index, horizon_index
                            ]
                        ),
                        "selector_selected_layer": int(
                            data["selected_layer"][anchor_index]
                        ),
                    }
                )
    return output


def _scopes(rows):
    splits = sorted({row["split"] for row in rows})
    output = [(split, [row for row in rows if row["split"] == split]) for split in splits]
    if len(splits) > 1:
        output.append(("all", rows))
    return output


def _finite(values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _sequence_means(rows, field):
    grouped = defaultdict(list)
    for row in rows:
        value = row[field]
        if np.isfinite(value):
            grouped[row["sequence"]].append(value)
    return {
        sequence: float(np.mean(values))
        for sequence, values in grouped.items()
        if values
    }


def summarize_effects(rows, horizons, bootstrap_samples, seed):
    output = []
    rng = np.random.default_rng(seed)
    for scope, scope_rows in _scopes(rows):
        for horizon in horizons:
            selected = [row for row in scope_rows if row["horizon"] == horizon]
            sequence_auc = _sequence_means(selected, "auc_delta")
            sequence_iou = _sequence_means(selected, "mean_iou_delta")
            sequence_post = _sequence_means(selected, "post_anchor_auc_delta")
            if not sequence_auc:
                continue
            auc_values = np.asarray(list(sequence_auc.values()), dtype=np.float64)
            iou_values = np.asarray(
                [sequence_iou[name] for name in sequence_auc], dtype=np.float64
            )
            post_values = np.asarray(
                [
                    sequence_post.get(name, float("nan"))
                    for name in sequence_auc
                ],
                dtype=np.float64,
            )
            bootstrap = bootstrap_mean(auc_values, bootstrap_samples, rng)
            finite_post = post_values[np.isfinite(post_values)]
            post_bootstrap = (
                bootstrap_mean(finite_post, bootstrap_samples, rng)
                if finite_post.size
                else {
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "probability_positive": float("nan"),
                }
            )
            anchor_auc = _finite([row["auc_delta"] for row in selected])
            failure_values = _finite(
                [row["failure_reduction_02"] for row in selected]
            )
            control_auc = _sequence_means(selected, "control_auc")
            treatment_auc = _sequence_means(selected, "treatment_auc")
            output.append(
                {
                    "split": scope,
                    "horizon": int(horizon),
                    "num_sequences": len(sequence_auc),
                    "num_anchors": len(selected),
                    "num_valid_anchors": int(anchor_auc.size),
                    "control_auc_macro": float(np.mean(list(control_auc.values()))),
                    "treatment_auc_macro": float(
                        np.mean(list(treatment_auc.values()))
                    ),
                    "auc_delta_macro": float(auc_values.mean()),
                    "auc_delta_median_sequence": float(np.median(auc_values)),
                    "auc_delta_p10_sequence": float(np.percentile(auc_values, 10.0)),
                    "auc_delta_p90_sequence": float(np.percentile(auc_values, 90.0)),
                    "auc_delta_ci_low": bootstrap["ci_low"],
                    "auc_delta_ci_high": bootstrap["ci_high"],
                    "auc_delta_probability_positive": bootstrap[
                        "probability_positive"
                    ],
                    "mean_iou_delta_macro": float(iou_values.mean()),
                    "post_anchor_auc_delta_macro": (
                        float(finite_post.mean())
                        if finite_post.size
                        else float("nan")
                    ),
                    "post_anchor_auc_delta_ci_low": post_bootstrap["ci_low"],
                    "post_anchor_auc_delta_ci_high": post_bootstrap["ci_high"],
                    "post_anchor_auc_delta_probability_positive": post_bootstrap[
                        "probability_positive"
                    ],
                    "positive_anchor_ratio": float(np.mean(anchor_auc > 0.0)),
                    "negative_anchor_ratio": float(np.mean(anchor_auc < 0.0)),
                    "positive_sequence_ratio": float(np.mean(auc_values > 0.0)),
                    "mean_failure_reduction_02": float(
                        failure_values.mean()
                    ) if failure_values.size else float("nan"),
                }
            )
    return output


def summarize_temporal_bins(rows, horizons, bins, bootstrap_samples, seed):
    output = []
    rng = np.random.default_rng(seed)
    for scope, scope_rows in _scopes(rows):
        for horizon in horizons:
            horizon_rows = [row for row in scope_rows if row["horizon"] == horizon]
            for bin_index in range(bins):
                lower = bin_index / bins
                upper = (bin_index + 1) / bins
                selected = [
                    row
                    for row in horizon_rows
                    if lower < row["anchor_fraction"] <= upper
                ]
                sequence_auc = _sequence_means(selected, "auc_delta")
                if not sequence_auc:
                    continue
                values = np.asarray(list(sequence_auc.values()), dtype=np.float64)
                bootstrap = bootstrap_mean(values, bootstrap_samples, rng)
                output.append(
                    {
                        "split": scope,
                        "horizon": int(horizon),
                        "bin": bin_index + 1,
                        "normalized_start_exclusive": lower,
                        "normalized_end_inclusive": upper,
                        "num_sequences": len(sequence_auc),
                        "num_anchors": len(selected),
                        "auc_delta_macro": float(values.mean()),
                        "auc_delta_median_sequence": float(np.median(values)),
                        "auc_delta_ci_low": bootstrap["ci_low"],
                        "auc_delta_ci_high": bootstrap["ci_high"],
                        "auc_delta_probability_positive": bootstrap[
                            "probability_positive"
                        ],
                    }
                )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument("--require_complete", action="store_true")
    args = parser.parse_args()
    if args.bins < 2:
        raise ValueError("--bins must be at least 2")
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap_samples must be positive")

    directory, run, records, missing, expected = load_records(
        args.input, args.require_complete
    )
    horizons = tuple(run["horizons"])
    anchors = anchor_rows(records, horizons)
    effects = summarize_effects(
        anchors, horizons, args.bootstrap_samples, args.seed
    )
    temporal = summarize_temporal_bins(
        anchors,
        horizons,
        args.bins,
        args.bootstrap_samples,
        args.seed + 1,
    )
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else directory
    )
    cache_path = output / "stage8_intervention_cache_summary.csv"
    anchor_path = output / "stage8_intervention_anchor_summary.csv"
    effect_path = output / "stage8_intervention_effect_summary.csv"
    temporal_path = output / "stage8_intervention_temporal_summary.csv"
    write_csv(cache_path, [record["report"] for record in records])
    write_csv(anchor_path, anchors)
    write_csv(effect_path, effects)
    write_csv(temporal_path, temporal)

    print(
        "Audited {} of {} intervention clips ({} missing).".format(
            len(records), expected, missing
        )
    )
    if missing:
        print("Pilot summary only; use --require_complete for the final development run.")
    print(
        "split  horizon  sequences  anchors  control_auc  treatment_auc  "
        "delta  clustered_95%       post_delta  post_95%"
    )
    for row in effects:
        if row["split"] == "all":
            continue
        interval = "[{:.4f},{:.4f}]".format(
            row["auc_delta_ci_low"], row["auc_delta_ci_high"]
        )
        post = row["post_anchor_auc_delta_macro"]
        post_text = "n/a" if not np.isfinite(post) else "{:.4f}".format(post)
        post_interval = (
            "n/a"
            if not np.isfinite(row["post_anchor_auc_delta_ci_low"])
            else "[{:.4f},{:.4f}]".format(
                row["post_anchor_auc_delta_ci_low"],
                row["post_anchor_auc_delta_ci_high"],
            )
        )
        print(
            "{split:<6s} {horizon:>7d} {num_sequences:>10d} {num_anchors:>8d} "
            "{control_auc_macro:>11.4f} {treatment_auc_macro:>14.4f} "
            "{auc_delta_macro:>7.4f}  {interval:<19s} {post_text:>10s}  "
            "{post_interval}".format(
                interval=interval,
                post_text=post_text,
                post_interval=post_interval,
                **row
            )
        )
    print("Cache summary CSV: {}".format(cache_path))
    print("Anchor summary CSV: {}".format(anchor_path))
    print("Effect summary CSV: {}".format(effect_path))
    print("Temporal summary CSV: {}".format(temporal_path))


if __name__ == "__main__":
    main()
