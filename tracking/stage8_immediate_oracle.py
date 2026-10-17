"""Budget oracle for the immediate causal value of one fixed-L9 action."""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from stage4_common import write_csv
from stage6_temporal_stability import bootstrap_mean
from stage8_intervention_summary import load_records


DEFAULT_BUDGETS = (0, 5, 10, 20, 30, 50, 100)


def immediate_rows(records, horizons):
    """Flatten valid intervention anchors without using post-action features."""
    horizons = tuple(int(value) for value in horizons)
    if 1 not in horizons:
        raise ValueError("Stage-8C requires horizon 1 labels")
    horizon_index = horizons.index(1)
    rows = []
    for record in records:
        entry = record["entry"]
        data = record["data"]
        for anchor_index in np.flatnonzero(data["valid_gt"][:, 0]):
            rows.append(
                {
                    "split": entry["split"],
                    "sequence": entry["sequence"],
                    "anchor_index": int(anchor_index),
                    "anchor_fraction": float(
                        data["anchor_fraction"][anchor_index]
                    ),
                    "iou_delta": float(
                        data["delta_iou"][anchor_index, 0]
                    ),
                    "auc_delta": float(
                        data["auc_delta"][anchor_index, horizon_index]
                    ),
                }
            )
    if not rows:
        raise ValueError("No valid immediate intervention anchors")
    return rows


def _sequence_values(rows, selected, field):
    sums = defaultdict(float)
    counts = defaultdict(int)
    for index, row in enumerate(rows):
        sequence = row["sequence"]
        counts[sequence] += 1
        if selected[index]:
            sums[sequence] += row[field]
    return {
        sequence: sums[sequence] / count
        for sequence, count in counts.items()
    }


def _bootstrap(values, samples, rng):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "probability_positive": float("nan"),
        }
    return bootstrap_mean(values, samples, rng)


def effect_distribution(rows):
    output = []
    for split in sorted({row["split"] for row in rows}):
        selected = [row for row in rows if row["split"] == split]
        iou = np.asarray([row["iou_delta"] for row in selected])
        auc = np.asarray([row["auc_delta"] for row in selected])
        output.append(
            {
                "split": split,
                "num_sequences": len({row["sequence"] for row in selected}),
                "num_anchors": len(selected),
                "always_l9_mean_iou_gain": float(iou.mean()),
                "always_l9_auc_gain": float(auc.mean()),
                "median_iou_delta": float(np.median(iou)),
                "iou_delta_p10": float(np.percentile(iou, 10.0)),
                "iou_delta_p90": float(np.percentile(iou, 90.0)),
                "positive_ratio": float(np.mean(iou > 0.0)),
                "negative_ratio": float(np.mean(iou < 0.0)),
                "material_positive_ratio_001": float(np.mean(iou > 0.01)),
                "material_harm_ratio_001": float(np.mean(iou < -0.01)),
                "positive_cap_mean_iou_gain": float(
                    np.maximum(iou, 0.0).mean()
                ),
                "positive_cap_auc_gain": float(
                    np.where(iou > 0.0, auc, 0.0).mean()
                ),
            }
        )
    return output


def budget_oracle(rows, budgets, bootstrap_samples, seed):
    output = []
    rng = np.random.default_rng(seed)
    for split in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        utility = np.asarray([row["iou_delta"] for row in split_rows])
        order = np.argsort(-utility, kind="mergesort")
        all_selected = np.ones(len(split_rows), dtype=bool)
        full_iou = _sequence_values(split_rows, all_selected, "iou_delta")
        full_auc = _sequence_values(split_rows, all_selected, "auc_delta")
        positive_selected = utility > 0.0
        positive_iou = _sequence_values(
            split_rows, positive_selected, "iou_delta"
        )
        positive_headroom = float(np.mean(list(positive_iou.values())))

        for budget in budgets:
            maximum = (
                0
                if float(budget) <= 0.0
                else min(
                    len(split_rows),
                    int(math.ceil(len(split_rows) * float(budget) / 100.0)),
                )
            )
            exact = np.zeros(len(split_rows), dtype=bool)
            exact[order[:maximum]] = True
            capped = exact & positive_selected
            for mode, selected in (
                ("exact_topk", exact),
                ("positive_cap", capped),
            ):
                count = int(selected.sum())
                ratio = count / len(split_rows)
                sequence_iou = _sequence_values(
                    split_rows, selected, "iou_delta"
                )
                sequence_auc = _sequence_values(
                    split_rows, selected, "auc_delta"
                )
                names = sorted(sequence_iou)
                iou_values = np.asarray(
                    [sequence_iou[name] for name in names], dtype=np.float64
                )
                auc_values = np.asarray(
                    [sequence_auc[name] for name in names], dtype=np.float64
                )
                random_iou = np.asarray(
                    [ratio * full_iou[name] for name in names],
                    dtype=np.float64,
                )
                random_auc = np.asarray(
                    [ratio * full_auc[name] for name in names],
                    dtype=np.float64,
                )
                lift = iou_values - random_iou
                gain_bootstrap = _bootstrap(
                    iou_values, bootstrap_samples, rng
                )
                lift_bootstrap = _bootstrap(lift, bootstrap_samples, rng)
                selected_utility = utility[selected]
                output.append(
                    {
                        "split": split,
                        "oracle_mode": mode,
                        "budget_percent": float(budget),
                        "num_sequences": len(names),
                        "num_anchors": len(split_rows),
                        "max_selected": maximum,
                        "num_selected": count,
                        "selected_ratio": ratio,
                        "mean_iou_gain": float(iou_values.mean()),
                        "gain_ci_low": gain_bootstrap["ci_low"],
                        "gain_ci_high": gain_bootstrap["ci_high"],
                        "gain_probability_positive": gain_bootstrap[
                            "probability_positive"
                        ],
                        "auc_gain": float(auc_values.mean()),
                        "random_matched_mean_iou_gain": float(
                            random_iou.mean()
                        ),
                        "random_matched_auc_gain": float(random_auc.mean()),
                        "lift_over_random_mean_iou": float(lift.mean()),
                        "lift_ci_low": lift_bootstrap["ci_low"],
                        "lift_ci_high": lift_bootstrap["ci_high"],
                        "lift_probability_positive": lift_bootstrap[
                            "probability_positive"
                        ],
                        "selected_positive_ratio": (
                            float(np.mean(selected_utility > 0.0))
                            if count
                            else float("nan")
                        ),
                        "selected_negative_ratio": (
                            float(np.mean(selected_utility < 0.0))
                            if count
                            else float("nan")
                        ),
                        "positive_headroom_mean_iou_gain": positive_headroom,
                        "positive_headroom_recovery": (
                            float(iou_values.mean() / positive_headroom)
                            if positive_headroom > 0.0
                            else float("nan")
                        ),
                    }
                )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--budgets", nargs="+", type=float, default=list(DEFAULT_BUDGETS)
    )
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--allow_incomplete", action="store_true")
    args = parser.parse_args()
    if (
        not args.budgets
        or any(value < 0.0 or value > 100.0 for value in args.budgets)
        or args.bootstrap_samples < 1
    ):
        raise ValueError("Budgets must be in [0,100] and bootstrap samples positive")

    directory, run, records, missing, expected = load_records(
        args.input, require_complete=not args.allow_incomplete
    )
    rows = immediate_rows(records, run["horizons"])
    distribution = effect_distribution(rows)
    budgets = budget_oracle(
        rows,
        tuple(sorted(set(float(value) for value in args.budgets))),
        args.bootstrap_samples,
        args.seed,
    )
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else directory
    )
    distribution_path = output / "stage8_immediate_effect_distribution.csv"
    budget_path = output / "stage8_immediate_budget_oracle.csv"
    write_csv(distribution_path, distribution)
    write_csv(budget_path, budgets)

    print(
        "Audited {} of {} clips ({} missing); {} valid immediate anchors.".format(
            len(records), expected, missing, len(rows)
        )
    )
    print("Immediate intervention distribution:")
    print(
        "split  anchors  all_l9_iou  all_l9_auc  positive  harmful  "
        "positive_headroom"
    )
    for row in distribution:
        print(
            "{split:<6s} {num_anchors:>7d} {always_l9_mean_iou_gain:>11.6f} "
            "{always_l9_auc_gain:>11.4f} {positive_ratio:>8.2%} "
            "{negative_ratio:>7.2%} {positive_cap_mean_iou_gain:>17.6f}".format(
                **row
            )
        )

    print("\nBudget oracle ranked by ground-truth immediate IoU delta:")
    print(
        "split  mode          budget  selected  iou_gain  random  lift_95%       "
        "auc_gain  recovery"
    )
    for row in budgets:
        if row["split"] != "val":
            continue
        interval = "[{:.6f},{:.6f}]".format(
            row["lift_ci_low"], row["lift_ci_high"]
        )
        recovery = row["positive_headroom_recovery"]
        recovery_text = (
            "n/a" if not np.isfinite(recovery) else "{:.2%}".format(recovery)
        )
        print(
            "{split:<6s} {oracle_mode:<13s} {budget_percent:>6.1f}% "
            "{selected_ratio:>8.2%} {mean_iou_gain:>9.6f} "
            "{random_matched_mean_iou_gain:>7.6f} {interval:<19s} "
            "{auc_gain:>8.4f} {recovery_text:>9s}".format(
                interval=interval, recovery_text=recovery_text, **row
            )
        )
    print("Distribution CSV: {}".format(distribution_path))
    print("Budget oracle CSV: {}".format(budget_path))
    print(
        "Warning: rankings use ground-truth intervention effects and are upper "
        "bounds, not deployable policies."
    )


if __name__ == "__main__":
    main()
