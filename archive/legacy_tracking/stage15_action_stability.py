"""Audit whether Stage-14 hindsight layer values persist across disjoint time windows.

This is a mechanism diagnostic, not a deployable selector.  The primary test
selects an L7-L12 action with frames 1-10 after an intervention and evaluates
that frozen action on frames 11-20.  Selection therefore still uses future
ground truth, but evaluation uses a disjoint outcome window and detects whether
the Stage-14 oracle headroom is temporally persistent rather than a local max.
"""

import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint, write_csv, write_json
from stage6_temporal_stability import bootstrap_mean
from stage14_multiaction_common import (
    CANDIDATE_LAYERS,
    load_and_audit_cache,
    load_multiaction_run,
    multiaction_cache_path,
)


SCHEMA = "stage15_action_stability_v1"
PRIMARY_TRANSFER = "early_to_late"


def _json_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def load_stability_dataset(directory, allow_incomplete=False):
    directory, run = load_multiaction_run(directory)
    chunks = {
        "split": [],
        "sequence": [],
        "anchor_offset": [],
        "selected_layer": [],
        "delta_iou": [],
        "valid_gt": [],
    }
    reports = []
    missing = []
    entries = run["manifest"]["sequences"]
    for entry in entries:
        path = multiaction_cache_path(directory, entry)
        if not path.is_file():
            missing.append(path)
            continue
        report, values, _ = load_and_audit_cache(path, entry, run)
        reports.append(report)
        eligible = values["eligible"]
        if not eligible.any():
            continue
        count = int(eligible.sum())
        chunks["split"].append(np.full(count, entry["split"], dtype="U8"))
        chunks["sequence"].append(
            np.full(count, entry["sequence"], dtype="U128")
        )
        for name in ("anchor_offset", "selected_layer", "delta_iou", "valid_gt"):
            chunks[name].append(values[name][eligible].copy())
    if missing and not allow_incomplete:
        raise ValueError(
            "Stage-14 cache is incomplete: {} of {} files missing".format(
                len(missing), len(entries)
            )
        )
    if not chunks["sequence"]:
        raise ValueError("No eligible Stage-14 anchors were loaded")
    data = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    if data["delta_iou"].shape[1:] != (
        len(CANDIDATE_LAYERS),
        run["horizon"],
    ):
        raise ValueError("Stage-14 delta_iou shape changed")
    if data["valid_gt"].shape != (
        len(data["sequence"]),
        run["horizon"],
    ):
        raise ValueError("Stage-14 valid_gt shape changed")
    if np.any(
        (data["selected_layer"] < CANDIDATE_LAYERS[0])
        | (data["selected_layer"] > CANDIDATE_LAYERS[-1])
    ):
        raise ValueError("Native selector contains an invalid candidate layer")
    valid_values = np.broadcast_to(
        data["valid_gt"][:, None, :], data["delta_iou"].shape
    )
    if not np.isfinite(data["delta_iou"][valid_values]).all():
        raise ValueError("Valid Stage-14 delta_iou values are non-finite")
    return directory, run, data, reports, missing, len(entries)


def discounted_window_values(
    delta_iou,
    valid_gt,
    frame_indices,
    gamma=0.9,
    min_valid_fraction=0.8,
):
    """Return normalized discounted values for one subset of outcome frames."""
    delta_iou = np.asarray(delta_iou, dtype=np.float64)
    valid_gt = np.asarray(valid_gt, dtype=bool)
    indices = np.asarray(frame_indices, dtype=np.int64)
    if delta_iou.ndim != 3 or delta_iou.shape[1] != len(CANDIDATE_LAYERS):
        raise ValueError("delta_iou must have shape [anchor,6,time]")
    if valid_gt.shape != (delta_iou.shape[0], delta_iou.shape[2]):
        raise ValueError("valid_gt must match delta_iou")
    if (
        indices.ndim != 1
        or indices.size == 0
        or len(np.unique(indices)) != len(indices)
        or np.any(indices < 0)
        or np.any(indices >= delta_iou.shape[2])
    ):
        raise ValueError("frame_indices must be unique and inside the horizon")
    if not 0.0 < gamma <= 1.0 or not 0.0 < min_valid_fraction <= 1.0:
        raise ValueError("Invalid gamma or minimum valid fraction")

    valid = valid_gt[:, indices]
    coverage = valid.mean(axis=1)
    eligible = coverage + 1e-12 >= min_valid_fraction
    weights = gamma ** indices.astype(np.float64)
    denominator = np.sum(valid * weights[None, :], axis=1)
    numerator = np.sum(
        np.where(valid[:, None, :], delta_iou[:, :, indices], 0.0)
        * weights[None, None, :],
        axis=2,
    )
    values = np.full((len(delta_iou), len(CANDIDATE_LAYERS)), np.nan)
    usable = eligible & (denominator > 0.0)
    values[usable] = numerator[usable] / denominator[usable, None]
    return values, usable, coverage


def _sequence_means(sequences, values):
    sequences = np.asarray(sequences)
    values = np.asarray(values, dtype=np.float64)
    if sequences.ndim != 1 or values.shape != (len(sequences),):
        raise ValueError("Expected aligned one-dimensional sequence values")
    if not np.isfinite(values).all():
        raise ValueError("Sequence values must be finite")
    unique = np.unique(sequences)
    means = np.asarray(
        [values[sequences == sequence].mean() for sequence in unique],
        dtype=np.float64,
    )
    return unique, means


def _paired_stats(sequences, values, bootstrap_samples, rng):
    unique, macro = _sequence_means(sequences, values)
    interval = bootstrap_mean(macro, bootstrap_samples, rng)
    tolerance = 1e-12
    return {
        "num_sequences": len(unique),
        "mean": float(macro.mean()),
        "median": float(np.median(macro)),
        "wins": int(np.sum(macro > tolerance)),
        "ties": int(np.sum(np.abs(macro) <= tolerance)),
        "losses": int(np.sum(macro < -tolerance)),
        **interval,
    }


def _macro_mean(sequences, values):
    return float(_sequence_means(sequences, values)[1].mean())


def _distribution(indices):
    counts = Counter(np.asarray(indices, dtype=np.int64).tolist())
    return "|".join(
        "l{}:{}".format(layer, counts.get(layer - 7, 0))
        for layer in CANDIDATE_LAYERS
    )


def _scope_masks(data):
    return (
        ("train", data["split"] == "train"),
        ("val", data["split"] == "val"),
        ("all", np.ones(len(data["sequence"]), dtype=bool)),
    )


def build_outcome_windows(data, run, split_point, min_valid_fraction):
    horizon = int(run["horizon"])
    if not 0 < split_point < horizon:
        raise ValueError("split_point must lie inside the collected horizon")
    index_sets = {
        "early": np.arange(0, split_point, dtype=np.int64),
        "late": np.arange(split_point, horizon, dtype=np.int64),
        "odd_positions": np.arange(0, horizon, 2, dtype=np.int64),
        "even_positions": np.arange(1, horizon, 2, dtype=np.int64),
    }
    windows = {}
    for name, indices in index_sets.items():
        values, eligible, coverage = discounted_window_values(
            data["delta_iou"],
            data["valid_gt"],
            indices,
            gamma=run["gamma"],
            min_valid_fraction=min_valid_fraction,
        )
        windows[name] = {
            "indices": indices,
            "values": values,
            "eligible": eligible,
            "coverage": coverage,
        }
    return windows


def transfer_summary_rows(
    data,
    windows,
    bootstrap_samples,
    seed,
):
    specs = (
        ("early_to_late", "early", "late", 1),
        ("late_to_early", "late", "early", 0),
        ("odd_to_even", "odd_positions", "even_positions", 0),
        ("even_to_odd", "even_positions", "odd_positions", 0),
    )
    rows = []
    rng = np.random.default_rng(seed)
    native_index = data["selected_layer"].astype(np.int64) - 7
    all_indices = np.arange(len(data["sequence"]))
    for scope, scope_mask in _scope_masks(data):
        for transfer, selection_name, evaluation_name, is_primary in specs:
            selection = windows[selection_name]
            evaluation = windows[evaluation_name]
            mask = scope_mask & selection["eligible"] & evaluation["eligible"]
            if not mask.any():
                raise ValueError("No eligible anchors for {}/{}".format(scope, transfer))
            indices = all_indices[mask]
            selection_values = selection["values"][mask]
            evaluation_values = evaluation["values"][mask]
            selected_action = np.argmax(selection_values, axis=1)
            evaluation_oracle = np.argmax(evaluation_values, axis=1)
            selected_gain = evaluation_values[
                np.arange(len(indices)), selected_action
            ]
            native_gain = evaluation_values[
                np.arange(len(indices)), native_index[mask]
            ]
            oracle_gain = evaluation_values.max(axis=1)
            difference = selected_gain - native_gain
            stats = _paired_stats(
                data["sequence"][mask], difference, bootstrap_samples, rng
            )
            rows.append(
                {
                    "scope": scope,
                    "transfer": transfer,
                    "is_primary": is_primary,
                    "selection_window": selection_name,
                    "evaluation_window": evaluation_name,
                    "num_sequences": stats["num_sequences"],
                    "num_anchors": int(mask.sum()),
                    "selected_gain": _macro_mean(
                        data["sequence"][mask], selected_gain
                    ),
                    "native_gain": _macro_mean(
                        data["sequence"][mask], native_gain
                    ),
                    "selected_minus_native": stats["mean"],
                    "median_sequence_delta": stats["median"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "probability_positive": stats["probability_positive"],
                    "wins": stats["wins"],
                    "ties": stats["ties"],
                    "losses": stats["losses"],
                    "evaluation_oracle_gain": _macro_mean(
                        data["sequence"][mask], oracle_gain
                    ),
                    "gap_to_evaluation_oracle": _macro_mean(
                        data["sequence"][mask], selected_gain - oracle_gain
                    ),
                    "selection_evaluation_oracle_agreement": float(
                        np.mean(selected_action == evaluation_oracle)
                    ),
                    "selected_positive_anchor_ratio": float(
                        np.mean(difference > 0.0)
                    ),
                    "selected_action_distribution": _distribution(selected_action),
                    "evaluation_oracle_distribution": _distribution(
                        evaluation_oracle
                    ),
                }
            )
    return rows


def horizon_summary_rows(
    data,
    run,
    horizons,
    min_valid_fraction,
    bootstrap_samples,
    seed,
):
    rows = []
    values_by_horizon = {}
    rng = np.random.default_rng(seed)
    native_index = data["selected_layer"].astype(np.int64) - 7
    all_indices = np.arange(len(data["sequence"]))
    for horizon in horizons:
        values, eligible, coverage = discounted_window_values(
            data["delta_iou"],
            data["valid_gt"],
            np.arange(horizon, dtype=np.int64),
            gamma=run["gamma"],
            min_valid_fraction=min_valid_fraction,
        )
        values_by_horizon[horizon] = {
            "values": values,
            "eligible": eligible,
            "coverage": coverage,
            "oracle": np.where(
                eligible,
                np.argmax(np.where(np.isfinite(values), values, -np.inf), axis=1),
                -1,
            ),
        }
        for scope, scope_mask in _scope_masks(data):
            mask = scope_mask & eligible
            if not mask.any():
                raise ValueError("No eligible {} H{} anchors".format(scope, horizon))
            indices = all_indices[mask]
            selected_values = values[mask]
            oracle_index = np.argmax(selected_values, axis=1)
            oracle_gain = selected_values.max(axis=1)
            native_gain = selected_values[
                np.arange(len(indices)), native_index[mask]
            ]
            headroom = oracle_gain - native_gain
            ordered = np.sort(selected_values, axis=1)
            margins = ordered[:, -1] - ordered[:, -2]
            stats = _paired_stats(
                data["sequence"][mask], headroom, bootstrap_samples, rng
            )
            fixed_gains = [
                _macro_mean(data["sequence"][mask], selected_values[:, index])
                for index in range(len(CANDIDATE_LAYERS))
            ]
            rows.append(
                {
                    "scope": scope,
                    "horizon": horizon,
                    "num_sequences": stats["num_sequences"],
                    "num_anchors": int(mask.sum()),
                    "best_fixed_layer": CANDIDATE_LAYERS[int(np.argmax(fixed_gains))],
                    "best_fixed_gain": float(max(fixed_gains)),
                    "native_gain": _macro_mean(
                        data["sequence"][mask], native_gain
                    ),
                    "oracle_gain": _macro_mean(
                        data["sequence"][mask], oracle_gain
                    ),
                    "oracle_minus_native": stats["mean"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "probability_positive": stats["probability_positive"],
                    "native_oracle_hit_rate": float(
                        np.mean(native_index[mask] == oracle_index)
                    ),
                    "mean_best_second_margin": float(margins.mean()),
                    "median_best_second_margin": float(np.median(margins)),
                    "margin_p10": float(np.percentile(margins, 10.0)),
                    "material_margin_ratio_001": float(np.mean(margins >= 0.001)),
                    "oracle_distribution": _distribution(oracle_index),
                }
            )
    return rows, values_by_horizon


def horizon_agreement_rows(
    data,
    values_by_horizon,
    bootstrap_samples,
    seed,
):
    rows = []
    rng = np.random.default_rng(seed)
    native_index = data["selected_layer"].astype(np.int64) - 7
    horizons = sorted(values_by_horizon)
    for selection_horizon_index, selection_horizon in enumerate(horizons[:-1]):
        for evaluation_horizon in horizons[selection_horizon_index + 1 :]:
            selection = values_by_horizon[selection_horizon]
            evaluation = values_by_horizon[evaluation_horizon]
            for scope, scope_mask in _scope_masks(data):
                mask = scope_mask & selection["eligible"] & evaluation["eligible"]
                selection_values = selection["values"][mask]
                evaluation_values = evaluation["values"][mask]
                selected_action = np.argmax(selection_values, axis=1)
                evaluation_oracle = np.argmax(evaluation_values, axis=1)
                selected_gain = evaluation_values[
                    np.arange(int(mask.sum())), selected_action
                ]
                native_gain = evaluation_values[
                    np.arange(int(mask.sum())), native_index[mask]
                ]
                stats = _paired_stats(
                    data["sequence"][mask],
                    selected_gain - native_gain,
                    bootstrap_samples,
                    rng,
                )
                rows.append(
                    {
                        "scope": scope,
                        "selection_horizon": selection_horizon,
                        "evaluation_horizon": evaluation_horizon,
                        "num_sequences": stats["num_sequences"],
                        "num_anchors": int(mask.sum()),
                        "oracle_action_agreement": float(
                            np.mean(selected_action == evaluation_oracle)
                        ),
                        "selection_gain_on_evaluation_horizon": _macro_mean(
                            data["sequence"][mask], selected_gain
                        ),
                        "native_gain_on_evaluation_horizon": _macro_mean(
                            data["sequence"][mask], native_gain
                        ),
                        "selected_minus_native": stats["mean"],
                        "ci_low": stats["ci_low"],
                        "ci_high": stats["ci_high"],
                        "probability_positive": stats["probability_positive"],
                    }
                )
    return rows


def make_decision(source_run, transfer_rows, horizon_rows, args, sequences):
    primary = next(
        row
        for row in transfer_rows
        if row["scope"] == "val" and row["transfer"] == PRIMARY_TRANSFER
    )
    supporting = next(
        row
        for row in transfer_rows
        if row["scope"] == "val" and row["transfer"] == "odd_to_even"
    )
    full_horizon = next(
        row
        for row in horizon_rows
        if row["scope"] == "val" and row["horizon"] == source_run["horizon"]
    )
    primary_pass = primary["ci_low"] > 0.0
    status = "PASS_STABILITY_SCREEN" if primary_pass else "FAIL_STABILITY_SCREEN"
    next_step = (
        "Collect randomized on-policy L7-L12 transitions with a richer recurrent state, "
        "then run one frozen offline-RL screen."
        if primary_pass
        else
        "Stop L7-L12 RL for this one-step intervention design: the hindsight-best "
        "action does not transfer to an independent later outcome window."
    )
    directory = Path(__file__).resolve().parent
    decision = {
        "schema": SCHEMA,
        "status": status,
        "analysis_role": "development_only_disjoint_outcome_stability_screen",
        "source_run_fingerprint": source_run["fingerprint"],
        "num_sequences": len(set(np.asarray(sequences).tolist())),
        "num_anchors": len(sequences),
        "horizon": source_run["horizon"],
        "gamma": source_run["gamma"],
        "split_point": args.split_point,
        "primary": {
            "scope": "val",
            "transfer": PRIMARY_TRANSFER,
            "rule": (
                "H1-H{} hindsight-selected layer must beat the native selector on "
                "disjoint H{}-H{} outcomes with a sequence-clustered 95% CI lower "
                "bound above zero"
            ).format(
                args.split_point,
                args.split_point + 1,
                source_run["horizon"],
            ),
            "selected_minus_native": primary["selected_minus_native"],
            "ci_low": primary["ci_low"],
            "ci_high": primary["ci_high"],
            "probability_positive": primary["probability_positive"],
            "oracle_action_agreement": primary[
                "selection_evaluation_oracle_agreement"
            ],
            "pass": primary_pass,
        },
        "supporting": {
            "odd_to_even_selected_minus_native": supporting[
                "selected_minus_native"
            ],
            "odd_to_even_ci_low": supporting["ci_low"],
            "odd_to_even_ci_high": supporting["ci_high"],
            "full_horizon_oracle_minus_native": full_horizon[
                "oracle_minus_native"
            ],
            "full_horizon_mean_best_second_margin": full_horizon[
                "mean_best_second_margin"
            ],
        },
        "next_step": next_step,
        "warning": (
            "Action selection still uses future ground truth and is not deployable. "
            "Passing only establishes temporal persistence of counterfactual action value."
        ),
        "code_sha256": {
            "stage15_action_stability.py": file_hash(Path(__file__)),
            "stage14_multiaction_common.py": file_hash(
                directory / "stage14_multiaction_common.py"
            ),
        },
    }
    decision["fingerprint"] = fingerprint(decision)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_point", type=int, default=10)
    parser.add_argument(
        "--prefix_horizons", nargs="+", type=int, default=[1, 5, 10, 20]
    )
    parser.add_argument("--min_valid_fraction", type=float, default=0.8)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2045)
    parser.add_argument("--allow_incomplete", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (
        args.split_point < 1
        or not args.prefix_horizons
        or any(horizon < 1 for horizon in args.prefix_horizons)
        or not 0.0 < args.min_valid_fraction <= 1.0
        or args.bootstrap_samples < 1
    ):
        raise ValueError("Invalid Stage-15 analysis arguments")

    input_dir, source_run, data, reports, missing, expected = load_stability_dataset(
        args.input, args.allow_incomplete
    )
    horizon = int(source_run["horizon"])
    horizons = tuple(sorted(set(args.prefix_horizons)))
    if horizons[-1] > horizon or horizon not in horizons:
        raise ValueError("prefix_horizons must include and not exceed the source horizon")
    if args.split_point >= horizon:
        raise ValueError("split_point must be below the source horizon")

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError("Use a new --output_dir or pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "stage15_cache_summary.csv", reports)
    print(
        "Audited {} of {} Stage-14 clips ({} missing); {} eligible anchors.".format(
            len(reports), expected, len(missing), len(data["sequence"])
        )
    )
    print(
        "Primary: H1-H{} selects one layer; disjoint H{}-H{} evaluates it.".format(
            args.split_point, args.split_point + 1, horizon
        )
    )

    windows = build_outcome_windows(
        data, source_run, args.split_point, args.min_valid_fraction
    )
    transfers = transfer_summary_rows(
        data, windows, args.bootstrap_samples, args.seed + 100
    )
    horizon_rows, values_by_horizon = horizon_summary_rows(
        data,
        source_run,
        horizons,
        args.min_valid_fraction,
        args.bootstrap_samples,
        args.seed + 200,
    )
    agreements = horizon_agreement_rows(
        data, values_by_horizon, args.bootstrap_samples, args.seed + 300
    )
    decision = make_decision(
        source_run, transfers, horizon_rows, args, data["sequence"]
    )

    write_csv(output / "stage15_transfer_summary.csv", transfers)
    write_csv(output / "stage15_horizon_summary.csv", horizon_rows)
    write_csv(output / "stage15_horizon_agreement.csv", agreements)
    write_json(output / "stage15_decision.json", decision)

    print("\nDisjoint-window action transfer:")
    print("scope  transfer          selected  native    delta    paired_95%          agree")
    for row in transfers:
        if row["transfer"] not in ("early_to_late", "odd_to_even"):
            continue
        print(
            "{scope:<6s} {transfer:<17s} {selected_gain:>8.6f} {native_gain:>8.6f} "
            "{selected_minus_native:>8.6f} [{ci_low:.6f},{ci_high:.6f}] "
            "{selection_evaluation_oracle_agreement:>6.2%}".format(**row)
        )
    print("\nPrefix-horizon oracle stability:")
    print("scope  H   oracle   native  headroom  paired_95%          hit")
    for row in horizon_rows:
        if row["scope"] not in ("train", "val"):
            continue
        print(
            "{scope:<6s} {horizon:>2d} {oracle_gain:>8.6f} {native_gain:>8.6f} "
            "{oracle_minus_native:>8.6f} [{ci_low:.6f},{ci_high:.6f}] "
            "{native_oracle_hit_rate:>6.2%}".format(**row)
        )
    print("\nPredeclared Stage-15 decision: {}".format(decision["status"]))
    print(decision["next_step"])
    print("Decision JSON: {}".format(output / "stage15_decision.json"))
    print("Reports: {}".format(output))
    print("Source cache: {}".format(input_dir))


if __name__ == "__main__":
    main()
