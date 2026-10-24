"""CPU integrity, matched-seed and 16-video development analysis for Stage-26."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_csv, write_json
from stage23_branch_ppo import single_writer
from stage23_development import read_sealed, sealed, valid_auc
from stage23_development_summary import audit_evaluation, paired_statistics
from stage26_value_common import (ANALYSIS_SEED, ARMS, BOOTSTRAPS, CONTROLS, MINIMUM_GAIN, SCHEMA,
    SEEDS, anchors_for, branch_actions, example_schedule, planned_jobs, validate_sample, verify_contract)


def load_training(directory, contract, seed, arm):
    root = Path(directory) / ("seed" + str(seed)) / arm
    complete = read_sealed(root / "complete.json")
    training = read_sealed(root / "training_complete.json")
    if (complete.get("schema") != SCHEMA or complete.get("contract_fingerprint") != contract["fingerprint"]
            or complete.get("seed") != seed or complete.get("arm") != arm or complete["training"] != training
            or training.get("schema") != SCHEMA or training.get("seed") != seed or training.get("arm") != arm
            or training.get("contract_fingerprint") != contract["fingerprint"]
            or file_hash(root / "last.pt") != training["checkpoint_sha256"]
            or file_hash(root / "history.csv") != training["history_sha256"]):
        raise ValueError("Completed arm/checkpoint/history does not match frozen identity")
    with (root / "history.csv").open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    jobs = planned_jobs(contract, seed)
    if len(history) != len(jobs) or training["attempts"] != len(jobs):
        raise ValueError("Incomplete training quota")
    prefix_id, head_id = training["initial_prefix_sha256"], training["initial_head_sha256"]
    eligibility = []
    for i, (row, job) in enumerate(zip(history, jobs)):
        name, round_index, anchor, actions = job
        path = root / "samples" / ("{:05d}.npz".format(i))
        if (int(row["attempt"]) != i or row["sequence"] != name or int(row["round"]) != round_index
                or int(row["anchor"]) != anchor or row["prefix_before_sha256"] != prefix_id
                or row["head_before_sha256"] != head_id or file_hash(path) != row["sample_sha256"]
                or int(row["branch_frames"]) != 240 or int(row["prefix_frames"]) != anchor + 1
                or int(row["replay_frames"]) != 19):
            raise ValueError("Training order, freshness or branch quota mismatch")
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata_json"].item()))
            data = {n: archive[n] for n in archive.files if n != "metadata_json"}
        validate_sample(data, meta)
        expected_identity = dict(contract_fingerprint=contract["fingerprint"], seed=seed, arm=arm,
            initial_prefix_sha256=training["initial_prefix_sha256"], initial_head_sha256=training["initial_head_sha256"])
        if (meta["identity"] != expected_identity or meta["attempt"] != i or meta["sequence"] != name
                or meta["round"] != round_index or meta["anchor"] != anchor
                or meta["teacher_prefix_sha256"] != prefix_id or meta["teacher_head_sha256"] != head_id
                or not np.array_equal(data["continuation"], branch_actions(actions, anchor, seed, name, round_index))):
            raise ValueError("Branches used another model version or sampling schedule")
        eligible = row["eligible"] == "True"
        if row["eligible"] not in ("True", "False") or eligible != meta["eligible"] or int(row["optimizer_steps"]) != int(eligible):
            raise ValueError("Eligibility and optimizer updates disagree")
        eligibility.append(eligible)
        for key in ("rank_prefix_gradient", "tracking_prefix_gradient", "rank_loss", "tracking_loss",
                    "maximum_gradient_norm", "seconds", "full_h20_replay_error", "differentiable_replay_error"):
            if not np.isfinite(float(row[key])) or float(row[key]) < 0:
                raise ValueError("Nonfinite/negative update diagnostic")
        if max(float(row["full_h20_replay_error"]), float(row["differentiable_replay_error"])) > 1e-6:
            raise ValueError("Training replay invariant failed")
        if arm == "stop_gradient" and float(row["rank_prefix_gradient"]) != 0:
            raise ValueError("Control leaked rank gradients into visual prefix")
        if not eligible and (row["prefix_after_sha256"] != prefix_id or row["head_after_sha256"] != head_id):
            raise ValueError("An ineligible anchor updated the model")
        prefix_id, head_id = row["prefix_after_sha256"], row["head_after_sha256"]
    if (prefix_id != training["final_prefix_sha256"] or head_id != training["final_head_sha256"]
            or training["optimizer_steps"] != sum(eligibility)
            or training["branch_frames"] != 240 * len(jobs)
            or training["prefix_changed"] != (prefix_id != training["initial_prefix_sha256"])
            or training["head_changed"] != (head_id != training["initial_head_sha256"])
            or training.get("frozen_runtime_and_memory_unchanged") is not True):
        raise ValueError("Final model/counts/frozen state do not match committed updates")
    for summary, column in (("maximum_rank_prefix_gradient", "rank_prefix_gradient"),
                             ("maximum_tracking_prefix_gradient", "tracking_prefix_gradient")):
        if training[summary] != max(float(r[column]) for r in history):
            raise ValueError("Gradient summary differs from actual update history")
    if training["maximum_replay_error"] != max(max(float(r["full_h20_replay_error"]),
                                                        float(r["differentiable_replay_error"])) for r in history):
        raise ValueError("Replay summary differs from actual update history")
    return root, training, complete, eligibility


def integrity_of_pair(directory, contract, seed):
    loaded = {arm: load_training(directory, contract, seed, arm) for arm in ARMS}
    a, b = (loaded[arm][1] for arm in ARMS)
    if (a["initial_prefix_sha256"] != b["initial_prefix_sha256"]
            or a["initial_head_sha256"] != b["initial_head_sha256"]
            or a["frozen_memory_sha256"] != b["frozen_memory_sha256"]
            or loaded[ARMS[0]][3] != loaded[ARMS[1]][3]):
        raise ValueError("A/B initializations or label/update quotas were not matched")
    return loaded


def summarize_smoke(directory, contract):
    loaded = integrity_of_pair(directory, contract, SEEDS[0])
    rows = [dict(arm=arm, **{k: v for k, v in loaded[arm][1].items() if k != "arm"}) for arm in ARMS]
    gradient_pass = (rows[0]["maximum_rank_prefix_gradient"] == 0 and rows[1]["maximum_rank_prefix_gradient"] > 0)
    training_pass = all(r["maximum_tracking_prefix_gradient"] > 0 and r["prefix_changed"] and
                        r["head_changed"] and r["optimizer_steps"] > 0 and r["maximum_replay_error"] <= 1e-6 for r in rows)
    if contract["mode"] != "smoke" or contract["checking"]:
        raise ValueError("Not a fit-only gradient smoke")
    return sealed(dict(schema=SCHEMA, contract_fingerprint=contract["fingerprint"],
        status="PASS_VALUE_REPRESENTATION_SMOKE" if gradient_pass and training_pass else "FAIL_VALUE_REPRESENTATION_SMOKE",
        gradient_intervention_pass=gradient_pass, training_and_replay_pass=training_pass,
        arms=rows, checking_labels_loaded=False, tune_labels_loaded=False, validation_labels_loaded=False,
        ppo_trained=False, warning="Runtime/gradient checks only; no learned routing gain or speed claim."))


def validate_route(record, method, frames):
    actions = np.asarray(record["actions"])
    boxes, valid, overlaps = np.asarray(record["boxes"]), np.asarray(record["valid"]), np.asarray(record["overlaps"])
    if (record["frames"] != frames or actions.shape != (frames,) or actions.dtype.kind not in "iu"
            or not np.isin(actions, range(7)).all() or boxes.shape != (frames, 4)
            or not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= 0)
            or valid.shape != (frames,) or valid.dtype != np.bool_):
        raise ValueError("Invalid checking trajectory geometry/action/visibility")
    expected = 3 * np.arange(1, frames + 1) // 10
    cost = (actions > 0).cumsum()
    if (not np.array_equal(cost, np.zeros_like(cost) if method == "memory_fast" else expected)
            or record["candidate_blocks"] != int(cost[-1])
            or not np.isclose(valid_auc(overlaps, valid), record["auc"], rtol=0, atol=1e-10)):
        raise ValueError("Checking metric or canonical call schedule does not reproduce")
    if method.startswith("l") and not np.isin(actions, (0, int(method[1:]) - 6)).all():
        raise ValueError("Fixed-layer control executed another layer")
    return valid


def load_evaluations(root, training, complete, contract, seed):
    references = complete["evaluations"]
    expected = {(name, method) for name in contract["checking"] for method in CONTROLS + ("branches",)}
    if len(references) != len(expected) or {(r["sequence"], r["method"]) for r in references} != expected:
        raise ValueError("Missing or duplicate final checking outputs")
    output, masks = {}, {}
    for ref in references:
        name, method = ref["sequence"], ref["method"]
        path = root / "checking" / method / (name + ".json")
        if path.relative_to(root).as_posix() != ref["file"] or file_hash(path) != ref["sha256"]:
            raise ValueError("Final checking output changed")
        record = read_sealed(path)
        if (record["contract_fingerprint"] != contract["fingerprint"] or record["sequence"] != name
                or record["method"] != method or record["policy_sha256"] != training["checkpoint_sha256"]):
            raise ValueError("Checking output belongs to another model")
        if method == "branches":
            q, score, eligible = np.asarray(record["post_values"]), np.asarray(record["predicted_values"]), np.asarray(record["eligible"])
            _, anchors = example_schedule(contract["clip_steps"][name], seed, name, 0)
            if (q.shape != (4, 6) or score.shape != (4, 6) or eligible.shape != (4,) or eligible.dtype != np.bool_
                    or not np.isfinite(q).all() or not np.isfinite(score).all() or np.any((q < 0) | (q > 1))
                    or record["anchors"] != anchors.tolist() or not eligible.any()
                    or not 0 <= record["maximum_replay_error"] <= 1e-6):
                raise ValueError("Invalid final-model H2-H20 support-state diagnostic")
        else:
            valid = validate_route(record, method, contract["clip_steps"][name])
            if name in masks and not np.array_equal(masks[name], valid):
                raise ValueError("Methods used different checking visibility masks")
            masks[name] = valid
        output[name, method] = record
    return output


def paired_row(left, right, label, family=1):
    ordinary = paired_statistics(left, right, BOOTSTRAPS, ANALYSIS_SEED)
    adjusted = paired_statistics(left, right, BOOTSTRAPS, ANALYSIS_SEED, .025 / family)
    return dict(contrast=label, **ordinary, family_ci_low=adjusted["ci_low"])


def progress(directory, contract):
    rows = []
    for seed in contract["seeds"]:
        for arm in ARMS:
            root = Path(directory) / ("seed" + str(seed)) / arm
            history = []
            if (root / "history.csv").exists():
                with (root / "history.csv").open(encoding="utf-8", newline="") as handle:
                    history = list(csv.DictReader(handle))
            rows.append(dict(seed=seed, arm=arm, reported_attempts=len(history),
                             expected_attempts=contract["expected_attempts"], complete=(root / "complete.json").exists()))
    return rows


def summarize_full(directory, contract):
    if contract["mode"] != "full" or contract["seeds"] != list(SEEDS) or len(contract["checking"]) != 16:
        raise ValueError("Require the complete predeclared three-seed 32/16 pilot")
    auc = {(arm, method): np.zeros((3, 16)) for arm in ARMS for method in CONTROLS}
    distributions = {(arm, method): np.zeros((3, 7), np.int64) for arm in ARMS for method in CONTROLS}
    support = {(arm, method): np.zeros((3, 16)) for arm in ARMS for method in ("value", "mean") + CONTROLS[2:8]}
    training_rows = []
    visibility = {}
    for s, seed in enumerate(SEEDS):
        loaded = integrity_of_pair(directory, contract, seed)
        for arm in ARMS:
            root, training, complete, eligible = loaded[arm]
            if sum(eligible) < .8 * len(eligible):
                raise ValueError("Fewer than 80% of planned training anchors had eligible H20 supervision")
            if (training["maximum_tracking_prefix_gradient"] <= 0 or not training["prefix_changed"]
                    or not training["head_changed"] or training["maximum_replay_error"] > 1e-6
                    or (arm == "value_aligned" and training["maximum_rank_prefix_gradient"] <= 0)):
                raise ValueError("Matched training did not satisfy the passed gradient/replay mechanism")
            data = load_evaluations(root, training, complete, contract, seed)
            training_rows.append(dict(seed=seed, arm=arm, attempts=training["attempts"],
                optimizer_steps=training["optimizer_steps"], branch_frames=training["branch_frames"],
                rank_prefix_gradient=training["maximum_rank_prefix_gradient"]))
            valid_anchors = 0
            for v, name in enumerate(contract["checking"]):
                mask = data[name, "value"]["valid"]
                if name in visibility and visibility[name] != mask:
                    raise ValueError("Seed/arm checking labels differ")
                visibility[name] = mask
                for method in CONTROLS:
                    auc[arm, method][s, v] = data[name, method]["auc"]
                    distributions[arm, method][s] += np.bincount(data[name, method]["actions"], minlength=7)
                record = data[name, "branches"]
                q, scores, valid = np.asarray(record["post_values"]), np.asarray(record["predicted_values"]), np.asarray(record["eligible"])
                valid_anchors += int(valid.sum())
                support[arm, "value"][s, v] = q[np.arange(4), scores.argmax(1)][valid].mean()
                support[arm, "mean"][s, v] = q[valid].mean()
                for a in range(6):
                    support[arm, "l" + str(a + 7)][s, v] = q[valid, a].mean()
            if valid_anchors < .8 * 4 * 16:
                raise ValueError("Insufficient checking support-state coverage")
    branch_rows = [paired_row(support[ARMS[1], "value"] - support[ARMS[1], "mean"],
        support[ARMS[0], "value"] - support[ARMS[0], "mean"], "B_minus_A_routing_excess"),
        paired_row(support[ARMS[1], "value"], support[ARMS[0], "value"], "B_minus_A_absolute_value")]
    route_rows = [paired_row(auc[ARMS[1], "value"], auc[ARMS[0], "value"], "B_value_minus_A_value")]
    for layer in range(7, 13):
        name = "l" + str(layer)
        branch_rows.append(paired_row(support[ARMS[1], "value"], support[ARMS[1], name], "B_value_minus_B_" + name, 6))
        route_rows.append(paired_row(auc[ARMS[1], "value"], auc[ARMS[1], name], "B_value_minus_B_" + name, 6))
    controls = read_sealed(Path(directory) / "original_controls.json")
    if controls["contract_fingerprint"] != contract["fingerprint"] or set(controls["references"]) != {"fast", "sgla"}:
        raise ValueError("Original reference controls differ from frozen contract")
    original = {}
    for method in ("fast", "sgla"):
        refs = controls["references"][method]
        if len(refs) != 16 or [r["sequence"] for r in refs] != contract["checking"]:
            raise ValueError("Incomplete original controls")
        values = []
        for ref in refs:
            path = Path(directory) / "original_controls" / method / (ref["sequence"] + ".json")
            if file_hash(path) != ref["sha256"]:
                raise ValueError("Original prediction control changed")
            record = audit_evaluation(path, contract, method, contract["bundle_fingerprint"])
            if record["valid"] != visibility[ref["sequence"]]:
                raise ValueError("Original control used different visibility labels")
            values.append(record["auc"])
        original[method] = np.tile(values, (3, 1))
        route_rows.append(paired_row(auc[ARMS[1], "value"], original[method], "B_value_minus_original_" + method))
    mechanism = (branch_rows[0]["ci_low"] > 0 and branch_rows[0]["positive_seed_means"] == 3
                 and all(r["family_ci_low"] > 0 and r["mean_delta"] >= MINIMUM_GAIN for r in branch_rows[2:]))
    deployment = route_rows[0]["ci_low"] > 0 and all(r["family_ci_low"] > 0 for r in route_rows[1:7])
    status = ("FAIL_VALUE_REPRESENTATION_PILOT" if not mechanism else
              "PASS_VALUE_REPRESENTATION_PILOT" if deployment else "MECHANISM_ONLY_NO_CLOSED_LOOP_GAIN")
    policy_rows = []
    for (arm, method), values in auc.items():
        counts = distributions[arm, method].sum(0)
        policy_rows.append(dict(arm=arm, method=method, mean_auc=float(values.mean()),
            seed_aucs="|".join("{:.6f}".format(v) for v in values.mean(1)),
            candidate_ratio=float(counts[1:].sum() / counts.sum()), active_layers=int((counts[1:] > 0).sum()),
            action_distribution="|".join(n + ":" + str(int(c)) for n, c in zip(("fast", "l7", "l8", "l9", "l10", "l11", "l12"), counts))))
    policy_rows.extend(dict(arm="original", method=method, mean_auc=float(values.mean()), seed_aucs="not_repeated_independent_videos",
                           candidate_ratio=0. if method == "fast" else 1., active_layers="native" if method == "sgla" else 0,
                           action_distribution="native_selector_not_inferred" if method == "sgla" else "fast_only")
                       for method, values in original.items())
    decision = sealed(dict(schema=SCHEMA, contract_fingerprint=contract["fingerprint"], status=status,
        value_representation_mechanism_pass=mechanism, own_model_closed_loop_guard_pass=deployment,
        train_videos=32, checking_videos=16, seeds=list(SEEDS), ppo_trained=False,
        original_tune_and_validation_labels_loaded=False,
        next_step="Design one matched PPO necessity smoke; no generalization or SGLA superiority claim" if status == "PASS_VALUE_REPRESENTATION_PILOT" else
                  "Stop this fixed-budget candidate; do not extend steps or search the excluded data",
        warning="Previously consumed development videos and pretrained representation. Seeds are paired repeats, not extra videos. "
                "Support-state H2-H20 values are not closed-loop AUC. No speed or external validation claim."))
    return decision, branch_rows, route_rows, policy_rows, training_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if not args.input.strip():
        parser.error("Explicit nonempty run directory required")
    directory = Path(args.input).expanduser().resolve()
    contract = verify_contract(directory)
    if args.status:
        for row in progress(directory, contract):
            print("{seed} {arm:<15s} {reported_attempts}/{expected_attempts} attempts; complete={complete}".format(**row))
        print("CSV can lag one atomic checkpoint. Resume the same command to reconcile it.")
        return
    with single_writer(directory):
        if contract["mode"] == "smoke":
            result = summarize_smoke(directory, contract)
            write_json(directory / "stage26_smoke_decision.json", result)
        else:
            result, branches, routes, policies, training = summarize_full(directory, contract)
            output = directory / "analysis"
            for name, rows in (("branch_paired", branches), ("closed_loop_paired", routes),
                               ("policy_summary", policies), ("training_summary", training)):
                write_csv(output / ("stage26_" + name + ".csv"), rows)
            write_json(output / "stage26_decision.json", result)
            for row in branches:
                print("{contrast}: {mean_delta:.6f} [{ci_low:.6f},{ci_high:.6f}]".format(**row))
        print("Decision: " + result["status"])


if __name__ == "__main__":
    main()
