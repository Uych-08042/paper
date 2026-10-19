"""Audit and summarize the one-shot Stage-16C true closed-loop evaluation."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint, tracking_iou, write_csv, write_json
from stage6_temporal_stability import bootstrap_mean, segment_auc
from stage16_behavior_common import ACTION_NAMES
from stage16c_closed_loop_validate import (
    POLICIES,
    POLICY_NAME,
    SCHEMA,
    candidate_budget,
    output_path,
)
from stage16c_freeze_policy import load_policy_bundle


ANALYSIS_SCHEMA = "stage16c_closed_loop_analysis_v1"
NUMERIC_FLOAT_FIELDS = (
    "iou",
    "candidate_advantage_threshold",
    "predicted_advantage",
    "q_fast",
    "q_l7",
    "q_l8",
    "q_l9",
    "q_l10",
    "q_l11",
    "q_l12",
    "model_decode_latency_ms",
    "gt_x",
    "gt_y",
    "gt_w",
    "gt_h",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
)
NUMERIC_INT_FIELDS = (
    "frame_id",
    "decision_index",
    "gt_valid",
    "candidate_blocks",
    "action_index",
    "candidate_budget_total",
    "candidate_used_before",
    "candidate_used_after",
    "candidate_remaining_after",
)


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_run(input_dir):
    input_dir = Path(input_dir).expanduser().resolve()
    run_path = input_dir / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError("Stage-16C run metadata missing: {}".format(run_path))
    run = _load_json(run_path)
    digest = run.get("fingerprint")
    body = {key: value for key, value in run.items() if key != "fingerprint"}
    if run.get("schema") != SCHEMA or fingerprint(body) != digest:
        raise ValueError("Invalid Stage-16C run contract")
    for name, expected in run.get("code_sha256", {}).items():
        path = Path(__file__).resolve().parent / name
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError("Stage-16C source changed after collection: {}".format(name))
    _, bundle = load_policy_bundle(run["policy_bundle_path"])
    if bundle["fingerprint"] != run["policy_bundle_fingerprint"]:
        raise ValueError("Stage-16C run and frozen policy bundle disagree")
    return input_dir, run, bundle


def _parse_row(row, path):
    output = dict(row)
    for field in NUMERIC_INT_FIELDS:
        value = output.get(field, "").strip()
        output[field] = None if value == "" else int(value)
    for field in NUMERIC_FLOAT_FIELDS:
        value = output.get(field, "").strip()
        output[field] = None if value == "" else float(value)
    if output["iou"] is None or not np.isfinite(output["iou"]):
        raise ValueError("Invalid IoU in {}".format(path))
    return output


def load_policy_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Stage-16C output missing: {}".format(path))
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [_parse_row(row, path) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError("Empty Stage-16C output: {}".format(path))
    return rows


def _box(row, prefix):
    return np.asarray(
        [row["{}_{}".format(prefix, name)] for name in "xywh"],
        dtype=np.float64,
    )


def audit_policy_rows(rows, entry, policy, run, bundle):
    expected_frames = list(range(entry["start"], entry["stop"]))
    if [row["frame_id"] for row in rows] != expected_frames:
        raise ValueError("Frame IDs changed for {}/{}".format(entry["sequence"], policy))
    if any(
        row["dataset"] != run["dataset"]
        or row["split"] != entry["split"]
        or row["sequence"] != entry["sequence"]
        or row["policy"] != policy
        or row["manifest_fingerprint"] != run["manifest_fingerprint"]
        or row["policy_bundle_fingerprint"] != bundle["fingerprint"]
        for row in rows
    ):
        raise ValueError("Row provenance mismatch for {}/{}".format(entry["sequence"], policy))
    if rows[0]["decision_index"] != 0 or rows[0]["action"] != "init":
        raise ValueError("Initial row contract changed for {}/{}".format(entry["sequence"], policy))
    decisions = rows[1:]
    steps = len(decisions)
    budget = candidate_budget(steps, bundle["primary_budget_percent"])
    for index, row in enumerate(decisions, start=1):
        if row["decision_index"] != index:
            raise ValueError("Decision index changed for {}/{}".format(entry["sequence"], policy))
        expected_iou = float(
            tracking_iou(_box(row, "bbox"), _box(row, "gt"), bool(row["gt_valid"]))
        )
        if abs(expected_iou - row["iou"]) > 1e-10:
            raise ValueError("IoU mismatch for {}/{}".format(entry["sequence"], policy))
        action = row["action_index"]
        if action is None or not 0 <= action < len(ACTION_NAMES):
            raise ValueError("Invalid action for {}/{}".format(entry["sequence"], policy))
        if row["action"] != ACTION_NAMES[action] or row["candidate_blocks"] != int(action > 0):
            raise ValueError("Action/cost mismatch for {}/{}".format(entry["sequence"], policy))
        expected_layer = action + 6 if action > 0 else None
        observed_layer = row["selected_layer"].strip()
        if observed_layer != ("" if expected_layer is None else str(expected_layer)):
            raise ValueError("Selected layer mismatch for {}/{}".format(entry["sequence"], policy))

    actions = np.asarray([row["action_index"] for row in decisions], dtype=np.int64)
    candidates = actions > 0
    if policy == "fast" and candidates.any():
        raise ValueError("Fast spent candidate blocks")
    if policy == "sgla" and not candidates.all():
        raise ValueError("Native SGLA did not spend one block per decision")
    if policy == "uniform_l9_p30":
        if int(candidates.sum()) != budget or np.any(actions[candidates] != 3):
            raise ValueError("Matched-cost uniform L9 policy changed")
    if policy == POLICY_NAME:
        if int(candidates.sum()) > budget:
            raise ValueError("TAVRTrack exceeded the P30 hard budget")
        expected_threshold = float(bundle["candidate_advantage_threshold"])
        previous_used = 0
        for row in decisions:
            if (
                row["candidate_budget_total"] != budget
                or row["candidate_used_before"] != previous_used
                or row["candidate_used_after"]
                != previous_used + row["candidate_blocks"]
                or row["candidate_remaining_after"]
                != budget - row["candidate_used_after"]
                or row["candidate_advantage_threshold"] is None
                or abs(row["candidate_advantage_threshold"] - expected_threshold) > 1e-12
                or row["predicted_advantage"] is None
                or any(row["q_{}".format(name)] is None for name in ACTION_NAMES)
            ):
                raise ValueError("TAVRTrack recurrent/budget trace is inconsistent")
            previous_used = row["candidate_used_after"]
    return {
        "steps": steps,
        "budget": budget,
        "candidate_blocks": int(candidates.sum()),
        "actions": actions,
    }


def audit_complete_run(input_dir, run, bundle, allow_incomplete=False):
    required = tuple(POLICIES)
    missing_contract = set(required).difference(run["policies"])
    if missing_contract and not allow_incomplete:
        raise ValueError(
            "Final Stage-16C run contract is missing policies: {}".format(
                ", ".join(sorted(missing_contract))
            )
        )
    policies = tuple(policy for policy in required if policy in run["policies"])
    grouped = defaultdict(dict)
    audit = {}
    missing = []
    for entry in run["entries"]:
        for policy in policies:
            path = output_path(input_dir, entry, policy)
            if not path.is_file():
                missing.append(path)
                continue
            rows = load_policy_file(path)
            audit[(entry["sequence"], policy)] = audit_policy_rows(
                rows, entry, policy, run, bundle
            )
            grouped[policy][entry["sequence"]] = rows
    if missing and not allow_incomplete:
        raise FileNotFoundError(
            "Stage-16C is incomplete; first missing output: {}".format(missing[0])
        )
    complete_sequences = [
        entry["sequence"]
        for entry in run["entries"]
        if all(entry["sequence"] in grouped[policy] for policy in policies)
    ]
    if not complete_sequences:
        raise ValueError("No complete Stage-16C sequence-policy groups")
    for sequence in complete_sequences:
        reference = grouped[policies[0]][sequence]
        for policy in policies[1:]:
            candidate = grouped[policy][sequence]
            if len(candidate) != len(reference):
                raise ValueError("Policy row counts differ for {}".format(sequence))
            for left, right in zip(reference, candidate):
                if left["frame_id"] != right["frame_id"] or not np.allclose(
                    _box(left, "gt"), _box(right, "gt"), rtol=0, atol=0
                ) or left["gt_valid"] != right["gt_valid"]:
                    raise ValueError("Paired ground truth differs for {}".format(sequence))
    grouped = {
        policy: {sequence: grouped[policy][sequence] for sequence in complete_sequences}
        for policy in policies
    }
    return grouped, audit, missing


def _sequence_auc(rows):
    valid = [row["iou"] for row in rows if row["gt_valid"]]
    return segment_auc(valid)


def summarize_methods(grouped, bundle):
    summaries = []
    sequence_auc = {}
    for policy, sequences in grouped.items():
        values = {
            sequence: _sequence_auc(rows) for sequence, rows in sequences.items()
        }
        sequence_auc[policy] = values
        decisions = [row for rows in sequences.values() for row in rows[1:]]
        latencies = np.asarray(
            [
                row["model_decode_latency_ms"]
                for row in decisions
                if row["model_decode_latency_ms"] is not None
            ],
            dtype=np.float64,
        )
        actions = Counter(row["action"] for row in decisions)
        candidate_blocks = int(sum(row["candidate_blocks"] for row in decisions))
        active_layers = sum(actions.get("l{}".format(layer), 0) > 0 for layer in range(7, 13))
        summaries.append(
            {
                "method": policy,
                "num_sequences": len(sequences),
                "tracked_frames": len(decisions),
                "candidate_blocks": candidate_blocks,
                "candidate_blocks_per_frame": candidate_blocks / len(decisions),
                "budget_percent": (
                    float(bundle["primary_budget_percent"])
                    if policy in ("uniform_l9_p30", POLICY_NAME)
                    else 100.0 if policy == "sgla" else 0.0
                ),
                "auc": float(np.mean(list(values.values()))),
                "mean_model_decode_latency_ms": (
                    float(latencies.mean()) if len(latencies) else float("nan")
                ),
                "fps_from_mean_latency": (
                    1000.0 / float(latencies.mean()) if len(latencies) else float("nan")
                ),
                "active_candidate_layers": active_layers,
                "action_distribution": "|".join(
                    "{}:{}".format(name, actions.get(name, 0)) for name in ACTION_NAMES
                ),
            }
        )
    by_method = {row["method"]: row for row in summaries}
    fast_auc = by_method["fast"]["auc"]
    sgla_auc = by_method["sgla"]["auc"]
    denominator = sgla_auc - fast_auc
    for row in summaries:
        row["gain_vs_fast"] = row["auc"] - fast_auc
        row["gain_vs_sgla"] = row["auc"] - sgla_auc
        row["recovery_of_sgla_gain"] = (
            (row["auc"] - fast_auc) / denominator
            if abs(denominator) > 1e-12
            else float("nan")
        )
    return summaries, sequence_auc


def paired_comparisons(sequence_auc, bootstrap_samples, seed):
    primary = sequence_auc[POLICY_NAME]
    keys = sorted(primary)
    rng = np.random.default_rng(seed)
    paired = []
    sequence_rows = []
    for reference in ("fast", "uniform_l9_p30", "sgla"):
        left = np.asarray([primary[key] for key in keys], dtype=np.float64)
        right = np.asarray([sequence_auc[reference][key] for key in keys], dtype=np.float64)
        delta = left - right
        bootstrap = bootstrap_mean(delta, bootstrap_samples, rng)
        paired.append(
            {
                "method": POLICY_NAME,
                "reference": reference,
                "num_sequences": len(keys),
                "mean_delta": float(delta.mean()),
                "median_delta": float(np.median(delta)),
                "wins": int(np.sum(delta > 1e-12)),
                "ties": int(np.sum(np.abs(delta) <= 1e-12)),
                "losses": int(np.sum(delta < -1e-12)),
                "ci_low": bootstrap["ci_low"],
                "ci_high": bootstrap["ci_high"],
                "probability_positive": bootstrap["probability_positive"],
            }
        )
        for key, left_value, right_value in zip(keys, left, right):
            sequence_rows.append(
                {
                    "sequence": key,
                    "method": POLICY_NAME,
                    "reference": reference,
                    "method_auc": left_value,
                    "reference_auc": right_value,
                    "delta": left_value - right_value,
                }
            )
    return paired, sequence_rows


def make_decision(run, bundle, summaries, paired, complete, bootstrap_samples, seed):
    primary = next(row for row in summaries if row["method"] == POLICY_NAME)
    vs_fast = next(row for row in paired if row["reference"] == "fast")
    vs_uniform = next(
        row for row in paired if row["reference"] == "uniform_l9_p30"
    )
    cap = float(bundle["primary_budget_percent"]) / 100.0
    checks = {
        "complete_frozen_val_run": bool(complete),
        "budget_cap_pass": primary["candidate_blocks_per_frame"] <= cap + 1e-12,
        "minimum_active_candidate_layers": 3,
        "observed_active_candidate_layers": primary["active_candidate_layers"],
        "noncollapse_pass": primary["active_candidate_layers"] >= 3,
        "tavr_minus_fast": vs_fast["mean_delta"],
        "tavr_minus_fast_ci_low": vs_fast["ci_low"],
        "beats_fast_pass": vs_fast["ci_low"] > 0.0,
        "tavr_minus_uniform_l9_p30": vs_uniform["mean_delta"],
        "tavr_minus_uniform_l9_p30_ci_low": vs_uniform["ci_low"],
        "adaptive_value_pass": vs_uniform["ci_low"] > 0.0,
    }
    passed = all(
        checks[key]
        for key in (
            "complete_frozen_val_run",
            "budget_cap_pass",
            "noncollapse_pass",
            "beats_fast_pass",
            "adaptive_value_pass",
        )
    )
    if passed:
        status = "PASS_CLOSED_LOOP_DEVELOPMENT"
        next_step = (
            "Freeze this exact policy and evaluate it once on a new untouched external "
            "benchmark. OTB and UAV123 remain unavailable for policy selection."
        )
    elif not checks["complete_frozen_val_run"]:
        status = "PILOT_ONLY"
        next_step = "Complete the exact frozen Stage-16C run before drawing a policy conclusion."
    elif not checks["budget_cap_pass"] or not checks["noncollapse_pass"]:
        status = "FAIL_POLICY_INTEGRITY"
        next_step = "Stop Stage-16C and audit the frozen deployment implementation."
    elif not checks["beats_fast_pass"]:
        status = "FAIL_CLOSED_LOOP_GAIN"
        next_step = (
            "Stop this TAVRTrack-P30 variant. Do not retune its threshold or budget on "
            "these validation videos; a new method version requires new development data."
        )
    else:
        status = "FAIL_ADAPTIVE_VALUE"
        next_step = (
            "The policy may beat Fast but did not beat the same-cap uniform L9 control. "
            "Do not claim learned layer scheduling or retune on these videos."
        )
    decision = {
        "schema": ANALYSIS_SCHEMA,
        "status": status,
        "analysis_role": "one_shot_true_closed_loop_development_decision",
        "source_run_fingerprint": run["fingerprint"],
        "policy_bundle_fingerprint": bundle["fingerprint"],
        "manifest_fingerprint": run["manifest_fingerprint"],
        "split": run["split"],
        "num_sequences": primary["num_sequences"],
        "bootstrap_samples": int(bootstrap_samples),
        "seed": int(seed),
        "primary_rule": (
            "Frozen TAVRTrack-P30 must remain within budget, use at least three candidate "
            "layers, and beat both Fast and same-cap Uniform-L9-P30 with paired "
            "sequence-bootstrap 95% CI lower bounds above zero."
        ),
        "checks": checks,
        "next_step": next_step,
        "warning": (
            "These Stage-16 validation videos were used once for the Stage-16B screen. "
            "This is a development result, not an external generalization claim."
        ),
        "code_sha256": {
            "stage16c_closed_loop_summary.py": file_hash(Path(__file__).resolve()),
            "stage16c_closed_loop_validate.py": file_hash(
                Path(__file__).resolve().parent / "stage16c_closed_loop_validate.py"
            ),
        },
    }
    decision["fingerprint"] = fingerprint(decision)
    return decision


def analyze(input_dir, bootstrap_samples, seed, allow_incomplete=False):
    input_dir, run, bundle = load_run(input_dir)
    grouped, audit, missing = audit_complete_run(
        input_dir, run, bundle, allow_incomplete=allow_incomplete
    )
    required_complete = (
        run["split"] == "val"
        and len(run["entries"]) == 40
        and set(run["policies"]) == set(POLICIES)
        and not missing
        and all(len(grouped.get(policy, {})) == 40 for policy in POLICIES)
    )
    summaries, sequence_auc = summarize_methods(grouped, bundle)
    paired, sequence_rows = paired_comparisons(
        sequence_auc, bootstrap_samples, seed
    )
    decision = make_decision(
        run,
        bundle,
        summaries,
        paired,
        required_complete,
        bootstrap_samples,
        seed,
    )
    return input_dir, run, bundle, summaries, paired, sequence_rows, decision, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2050)
    parser.add_argument("--allow_incomplete", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    (
        input_dir,
        run,
        bundle,
        summaries,
        paired,
        sequence_rows,
        decision,
        missing,
    ) = analyze(
        args.input,
        args.bootstrap_samples,
        args.seed,
        allow_incomplete=args.allow_incomplete,
    )
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_dir / "analysis"
    )
    summary_path = output / "stage16c_policy_summary.csv"
    paired_path = output / "stage16c_paired.csv"
    sequence_path = output / "stage16c_sequence_comparison.csv"
    decision_path = output / "stage16c_decision.json"
    write_csv(summary_path, summaries)
    write_csv(paired_path, paired)
    write_csv(sequence_path, sequence_rows)
    write_json(decision_path, decision)

    print(
        "Audited {} complete Stage-16C sequences ({} missing policy files).".format(
            decision["num_sequences"], len(missing)
        )
    )
    print("method               blocks/f       auc  gain_fast  gain_sgla  layers  latency_ms")
    for row in summaries:
        print(
            "{method:<20s} {candidate_blocks_per_frame:>8.2%}  {auc:>8.4f} "
            "{gain_vs_fast:>10.4f} {gain_vs_sgla:>10.4f} "
            "{active_candidate_layers:>7d} {mean_model_decode_latency_ms:>11.3f}".format(
                **row
            )
        )
    print("\nPaired TAVRTrack-P30 contrasts:")
    print("reference             mean_delta  paired_95%           W/T/L")
    for row in paired:
        interval = "[{:.4f},{:.4f}]".format(row["ci_low"], row["ci_high"])
        result = "{}/{}/{}".format(row["wins"], row["ties"], row["losses"])
        print(
            "{reference:<20s} {mean_delta:>10.4f}  {interval:<19s} {result}".format(
                interval=interval, result=result, **row
            )
        )
    print("\nPredeclared Stage-16C decision: {}".format(decision["status"]))
    print("Decision JSON: {}".format(decision_path))
    print("Next: {}".format(decision["next_step"]))
    print("Policy summary CSV: {}".format(summary_path))
    print("Paired CSV: {}".format(paired_path))


if __name__ == "__main__":
    main()
