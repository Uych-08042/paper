"""Evaluate train-source-only structure rankings on separate development videos."""

import argparse
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint, write_csv, write_json
from stage6_temporal_stability import bootstrap_mean
from stage8_intervention_common import _window_auc
from stage19_compression_common import cache_path, load_and_audit_cache, load_run, nonempty_path


def sequence_scores(data, horizon):
    """Use the same valid anchors for H1, H5, and H20 and for both layouts."""
    valid = data["valid"]
    eligible = valid[:, 0] & (valid.mean(axis=1) >= 0.8) & (valid[:, :5].mean(axis=1) >= 0.8)
    if not eligible.any():
        raise ValueError("No anchors with valid first frame and >=80% valid H5/full-window outcomes")
    keep = valid[eligible, :horizon]
    reference = data["reference_iou"][eligible, :horizon]
    reference_mean = np.mean([row[mask].mean() for row, mask in zip(reference, keep)])
    output = {"eligible_anchors": int(eligible.sum()), "reference_mean_iou": float(reference_mean)}
    for kind in ("sync", "free"):
        values = data[kind + "_iou"][eligible, :, :horizon]
        mean = np.asarray([[row[mask].mean() for row in anchor]
                           for anchor, mask in zip(values, keep)]).mean(axis=0)
        auc = np.asarray([[_window_auc(row[mask]) for row in anchor]
                          for anchor, mask in zip(values, keep)]).mean(axis=0)
        output[kind + "_damage"] = reference_mean - mean
        output[kind + "_auc"] = auc
    output["feedback_damage"] = output["free_damage"] - output["sync_damage"]
    return output


def _interval(values, samples, rng):
    return bootstrap_mean(np.asarray(values, dtype=np.float64), samples, rng)


def analyze(records, run, missing=0, bootstrap_samples=5000, seed=2058):
    rng = np.random.default_rng(seed)
    horizons = sorted(set((1, 5, run["design"]["horizon"])))
    scores = {h: [sequence_scores(record["data"], h) for record in records] for h in horizons}
    sequence_rows, summaries = [], []
    for h in horizons:
        for i, record in enumerate(records):
            result = scores[h][i]
            for mask, spec in enumerate(run["masks"]):
                sequence_rows.append({"role": "calibration" if record["entry"]["split"] == "train" else "development_check",
                                      "sequence": record["entry"]["sequence"], "mask": spec["name"],
                                      "horizon": h, "eligible_anchors": result["eligible_anchors"],
                                      "sync_iou_damage": float(result["sync_damage"][mask]),
                                      "free_iou_damage": float(result["free_damage"][mask]),
                                      "feedback_iou_damage": float(result["feedback_damage"][mask])})
        for split in ("train", "val"):
            indices = [i for i, record in enumerate(records) if record["entry"]["split"] == split]
            if not indices:
                continue
            for mask, spec in enumerate(run["masks"]):
                feedback = [scores[h][i]["feedback_damage"][mask] for i in indices]
                interval = _interval(feedback, bootstrap_samples, rng)
                summaries.append({"role": "calibration" if split == "train" else "development_check",
                                  "mask": spec["name"], "layer": spec["layer"], "horizon": h,
                                  "num_sequences": len(indices), "removed_ffn_channels": len(spec["drop"]),
                                  "removed_parameters": int(records[indices[0]]["data"]["removed_parameters"][mask]),
                                  "sync_iou_damage": float(np.mean([scores[h][i]["sync_damage"][mask] for i in indices])),
                                  "free_iou_damage": float(np.mean([scores[h][i]["free_damage"][mask] for i in indices])),
                                  "feedback_iou_damage": float(np.mean(feedback)),
                                  "feedback_ci_low": interval["ci_low"], "feedback_ci_high": interval["ci_high"]})
    decision = {"schema": "stage19_compression_mechanism_decision_v1",
                "source_run_fingerprint": run["fingerprint"], "status": "PENDING_INCOMPLETE",
                "audited_clips": len(records), "missing_clips": int(missing),
                "analysis_role": "development_only_mechanism_pilot",
                "speed_claim_tested": False, "learned_compression_model_trained": False,
                "rule": "Choose one mask on calibration videos only; its check H20 IoU must beat both H1 and same-crop-H5 selections with sequence-clustered CI lower bounds >0",
                "warning": "Source-train videos only. H5 calibration and full-horizon checking use disjoint videos. Mask choices are individual models, not a combined pruning mask or an ensemble."}
    comparisons = []
    calibration = [i for i, record in enumerate(records) if record["entry"]["split"] == "train"]
    check = [i for i, record in enumerate(records) if record["entry"]["split"] == "val"]
    # Never select structures from a partial calibration cohort.
    if not missing and calibration and check:
        selections = {}
        for name, h, kind in (("single_frame", 1, "free"), ("same_crop_h5", 5, "sync"), ("closed_loop_h5", 5, "free")):
            damage = np.mean([scores[h][i][kind + "_damage"] for i in calibration], axis=0)
            selections[name] = int(np.argmin(damage))
        decision["calibration_selected_masks"] = {key: run["masks"][value]["name"] for key, value in selections.items()}
        primary_h = run["design"]["horizon"]
        selected = selections["closed_loop_h5"]
        for name in ("single_frame", "same_crop_h5"):
            alternative = selections[name]
            gains = np.asarray([scores[primary_h][i]["free_damage"][alternative]
                                - scores[primary_h][i]["free_damage"][selected] for i in check])
            interval = _interval(gains, bootstrap_samples, rng)
            comparisons.append({"reference": run["masks"][selected]["name"], "control_criterion": name,
                                "control_mask": run["masks"][alternative]["name"], "check_sequences": len(check),
                                "horizon": primary_h, "mean_iou_gain": float(gains.mean()),
                                "ci_low": interval["ci_low"], "ci_high": interval["ci_high"],
                                "wins": int((gains > 1e-10).sum()), "ties": int((np.abs(gains) <= 1e-10).sum()),
                                "losses": int((gains < -1e-10).sum())})
        decision["status"] = ("PILOT_TOO_SMALL" if min(len(calibration), len(check)) < 6 else
                              "MECHANISM_SUPPORTED" if all(row["ci_low"] > 0 for row in comparisons) else
                              "MECHANISM_NOT_ESTABLISHED")
        decision["check_comparisons"] = comparisons
    decision["next_step"] = ("Collect the remaining frozen clips before selecting any structure." if missing else
                             "Design and compare joint structure/weight learning under matched training and compute budgets."
                             if decision["status"] == "MECHANISM_SUPPORTED" else
                             "Inspect the paired mechanism reports; current evidence does not justify full training of this hypothesis.")
    return summaries, sequence_rows, comparisons, decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True)
    parser.add_argument("--output_dir", type=nonempty_path, required=True)
    parser.add_argument("--allow_incomplete", action="store_true")
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2058)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap_samples must be positive")
    directory, run = load_run(args.input)
    records, missing = [], []
    for entry in run["manifest"]["sequences"]:
        path = cache_path(directory, entry)
        if path.is_file():
            records.append({"entry": entry, "data": load_and_audit_cache(path, entry, run)})
        else:
            missing.append(str(path))
    if missing and not args.allow_incomplete:
        raise FileNotFoundError("{} clips missing; first {}. Use --allow_incomplete for integrity checks only.".format(len(missing), missing[0]))
    if not records:
        raise ValueError("No Stage-19 clips found")
    summary, sequences, paired, decision = analyze(records, run, len(missing), args.bootstrap_samples, args.seed)
    decision.update(bootstrap_samples=args.bootstrap_samples, seed=args.seed,
                    analysis_code_sha256=file_hash(__file__))
    decision["fingerprint"] = fingerprint(decision)
    output = Path(args.output_dir).expanduser().resolve()
    for name, rows in (("mask_summary", summary), ("sequence_summary", sequences), ("selection_comparison", paired)):
        if rows:
            write_csv(output / ("stage19_" + name + ".csv"), rows)
    write_json(output / "stage19_decision.json", decision)
    print("Audited {} of {} clips. Reference replay, identity surgery, module restoration and crop continuity passed.".format(
        len(records), len(run["manifest"]["sequences"])))
    print("role                mask           H  same-crop damage  closed-loop damage  feedback [95% CI]")
    for row in summary:
        if row["horizon"] == run["design"]["horizon"]:
            print("{role:<19} {mask:<13} {horizon:>2} {sync_iou_damage:>17.6f} {free_iou_damage:>19.6f} {feedback_iou_damage:>9.6f} [{feedback_ci_low:.6f},{feedback_ci_high:.6f}]".format(**row))
    for row in paired:
        print("Closed-loop selection minus {}: {:.6f} [{:.6f},{:.6f}]".format(
            row["control_criterion"], row["mean_iou_gain"], row["ci_low"], row["ci_high"]))
    print("Decision: {}\nReports: {}".format(decision["status"], output))
    print("This is a persistent-compression mechanism pilot, not a learned method or a speed result.")


if __name__ == "__main__":
    main()
