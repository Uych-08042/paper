"""CPU-only paired-seed/video analysis; refuses efficacy claims on partial runs."""

import argparse
import csv
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_csv, write_json
from stage23_branch_credit import ARMS, ANCHORS, HORIZON, REPEATS
from stage23_development import CONTROLS, METRIC, ROLLOUT, SCHEMA, SEEDS, STEPS, read_sealed, sealed, valid_auc


def progress(directory):
    directory = Path(directory).expanduser().resolve()
    contract = read_sealed(directory / "stage23b_contract.json")
    if contract.get("schema") != SCHEMA:
        raise ValueError("Not a Stage-23B fixed-budget development run")
    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            path = directory / ("seed" + str(seed)) / arm
            history = []
            if (path / "history.csv").is_file():
                with (path / "history.csv").open(newline="", encoding="utf-8") as handle:
                    history = list(csv.DictReader(handle))
            last = history[-1] if history else None
            rows.append(dict(seed=seed, arm=arm, reported_steps=int(last["steps"]) if last else 0,
                             branch_frames=sum(int(r["branch_transitions"]) for r in history),
                             completed=(path / "complete.json").is_file()))
    return contract, rows


def paired_statistics(left, right, samples=5000, seed=2077, lower_tail=.025):
    """Average paired seed effects within video; never count seeds as more videos."""
    left, right = np.asarray(left, float), np.asarray(right, float)
    if (left.ndim != 2 or right.shape != left.shape or min(left.shape) < 1
            or not np.isfinite(left).all() or not np.isfinite(right).all()
            or samples < 2 or not 0 < lower_tail < .5):
        raise ValueError("Expected paired finite [seed, video] outcomes")
    delta = left - right
    per_video, per_seed = delta.mean(0), delta.mean(1)
    rng = np.random.default_rng(seed)
    video_draws = rng.integers(len(per_video), size=(samples, len(per_video)))
    video_bootstrap = per_video[video_draws].mean(1)
    seed_draws = rng.integers(len(per_seed), size=(samples, len(per_seed)))
    sampled_seeds = delta[seed_draws].mean(1)
    crossed = np.take_along_axis(sampled_seeds, video_draws, axis=1).mean(1)
    return dict(mean_delta=float(delta.mean()), median_video_delta=float(np.median(per_video)),
                ci_low=float(np.quantile(video_bootstrap, lower_tail)),
                ci_high=float(np.quantile(video_bootstrap, 1 - lower_tail)),
                seed_video_ci_low=float(np.quantile(crossed, lower_tail)),
                seed_video_ci_high=float(np.quantile(crossed, 1 - lower_tail)),
                bootstrap_positive_fraction=float((video_bootstrap > 0).mean()),
                positive_seed_means=int((per_seed > 0).sum()),
                seed_mean_deltas="|".join("{:.8f}".format(v) for v in per_seed),
                seed_mean_std=float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.,
                wins=int((per_video > 1e-8).sum()), ties=int((np.abs(per_video) <= 1e-8).sum()),
                losses=int((per_video < -1e-8).sum()))


def audit_evaluation(path, contract, method, policy_id):
    record = read_sealed(path)
    if (record.get("contract_fingerprint") != contract["fingerprint"] or record.get("method") != method
            or record.get("policy_sha256") != policy_id or record.get("metric") != METRIC):
        raise ValueError("Evaluation identity/metric mismatch: " + str(path))
    frames = record["frames"]
    actions = np.asarray(record["actions"])
    boxes, valid = np.asarray(record["boxes"], float), np.asarray(record["valid"])
    if (frames < 1 or actions.shape != (frames,) or actions.dtype.kind not in "iu"
            or boxes.shape != (frames, 4) or not np.isfinite(boxes).all() or (boxes[:, 2:] <= 0).any()
            or valid.shape != (frames,) or valid.dtype != np.bool_ or record["valid_frames"] != int(valid.sum())):
        raise ValueError("Malformed evaluation trajectory")
    if method == "sgla":
        if not (actions == -1).all():
            raise ValueError("Native selector actions are incorrectly labeled as candidate layers")
        candidate = np.ones(frames, bool)
    else:
        if not np.isin(actions, range(7)).all():
            raise ValueError("Unknown evaluation action")
        candidate = actions > 0
        if method != "fast" and np.any(candidate.cumsum() > 3 * np.arange(1, frames + 1) // 10):
            raise ValueError("Evaluation violated a causal budget prefix")
        if method in ("fast", "memory_fast") and candidate.any():
            raise ValueError("Fast control executed a candidate")
        if method.startswith("l") and not np.isin(actions, (0, int(method[1:]) - 6)).all():
            raise ValueError("Static-layer control executed another layer")
    auc = valid_auc(record["overlaps"], valid)
    if not np.isclose(auc, record["auc"], rtol=0, atol=1e-10) or record["candidate_blocks"] != int(candidate.sum()):
        raise ValueError("Evaluation AUC/candidate counts do not reproduce")
    if not np.isfinite(record["mean_crop_model_decode_ms"]) or record["mean_crop_model_decode_ms"] < 0:
        raise ValueError("Invalid timing observation")
    return record


def load_references(references, directory, contract, method, policy_id):
    expected = contract["tune"]
    if len(references) != len(expected) or {r["sequence"] for r in references} != set(expected):
        raise ValueError("Incomplete or duplicate tune-video references")
    by_name, rows = {r["sequence"]: r for r in references}, []
    for name in expected:
        if Path(name).name != name or "\\" in name:
            raise ValueError("Unsafe source video identity")
        path = directory / (name + ".json")
        if file_hash(path) != by_name[name]["sha256"]:
            raise ValueError("Frozen evaluation changed: " + str(path))
        row = audit_evaluation(path, contract, method, policy_id)
        if row["sequence"] != name:
            raise ValueError("Evaluation sequence mismatch")
        rows.append(row)
    return rows


def load_complete(directory, contract):
    controls = read_sealed(directory / "controls_complete.json")
    if controls["contract_fingerprint"] != contract["fingerprint"] or set(controls["evaluations"]) != set(CONTROLS):
        raise ValueError("Controls do not match the frozen plan")
    trajectories = {}
    for method in CONTROLS:
        trajectories[method] = [load_references(controls["evaluations"][method], directory / "controls" / method,
            contract, method, contract["bundle_fingerprint"])]
    costs = []
    for seed in SEEDS:
        initial_ids, update_counts, visibility_counts = set(), [], []
        for arm in ARMS:
            path = directory / ("seed" + str(seed)) / arm
            complete = read_sealed(path / "complete.json")
            if (complete.get("schema") != SCHEMA or complete.get("contract_fingerprint") != contract["fingerprint"]
                    or complete.get("seed") != seed or complete.get("arm") != arm or complete.get("steps") != STEPS
                    or not complete.get("frozen_representation_unchanged") or not complete.get("actor_changed")
                    or complete["initial_model_sha256"] == complete["final_model_sha256"]
                    or file_hash(path / "last.pt") != complete["checkpoint_sha256"]
                    or file_hash(path / "history.csv") != complete["history_sha256"]):
                raise ValueError("Final checkpoint/history integrity failure: " + str(path))
            initial_ids.add(complete["initial_model_sha256"])
            with (path / "history.csv").open(newline="", encoding="utf-8") as handle:
                history = list(csv.DictReader(handle))
            if len(history) != STEPS // ROLLOUT:
                raise ValueError("Incomplete training history")
            for index, row in enumerate(history, 1):
                if (int(row["steps"]) != index * ROLLOUT or int(row["seed"]) != seed or row["arm"] != arm
                        or int(row["anchor_count"]) != ANCHORS
                        or int(row["branch_transitions"]) != ANCHORS * REPEATS * 7 * HORIZON
                        or int(row["optimizer_steps"]) not in (0, 4)
                        or any(not np.isfinite(float(row[k])) or float(row[k]) > bound for k, bound in (
                            ("likelihood_replay_error", 1e-4), ("probability_replay_error", 1e-4),
                            ("chosen_replay_error", 1e-6)))):
                    raise ValueError("Training quota/replay violation")
            updates = sum(int(r["optimizer_steps"]) for r in history)
            update_counts.append(updates)
            visibility_counts.append(tuple(int(r["valid_transitions"]) for r in history))
            costs.append(dict(seed=seed, method=arm, live_frames=STEPS,
                branch_frames=sum(int(r["branch_transitions"]) for r in history), optimizer_steps=updates,
                live_candidate_blocks=sum(int(r["live_candidate_blocks"]) for r in history),
                branch_candidate_blocks=sum(int(r["branch_candidate_blocks"]) for r in history),
                valid_anchors=sum(int(r["valid_anchors"]) for r in history),
                train_seconds=sum(float(r["seconds"]) for r in history)))
            trajectories.setdefault(arm, []).append(load_references(complete["evaluations"], path / "evaluation",
                contract, "policy", complete["checkpoint_sha256"]))
        if len(initial_ids) != 1 or len(set(update_counts)) != 1 or len(set(visibility_counts)) != 1:
            raise ValueError("Seed-paired arms did not match initialization, update or label-visibility quotas")
    reference = trajectories["fast"][0]
    for runs in trajectories.values():
        for rows in runs:
            if any(r["frames"] != ref["frames"] or r["valid"] != ref["valid"] for r, ref in zip(rows, reference)):
                raise ValueError("Evaluation methods do not cover identical frames/visibility")
    return trajectories, costs


def summarize(trajectories, contract):
    matrices, summaries = {}, []
    for method in ARMS + CONTROLS:
        runs = trajectories[method]
        matrix = np.array([[r["auc"] for r in rows] for rows in runs])
        matrices[method] = np.repeat(matrix, len(SEEDS), axis=0) if len(runs) == 1 else matrix
        frames = sum(r["frames"] for rows in runs for r in rows)
        calls = sum(r["candidate_blocks"] for rows in runs for r in rows)
        layer_counts = np.bincount([a for rows in runs for r in rows for a in r["actions"] if a > 0], minlength=7)
        summaries.append(dict(method=method, auc=float(matrix.mean()), seeds=len(runs), videos=len(contract["tune"]),
            candidate_ratio=calls / frames, active_layers=None if method == "sgla" else int((layer_counts[1:] > 0).sum()),
            mean_crop_model_decode_ms=float(np.mean([r["mean_crop_model_decode_ms"] for rows in runs for r in rows]))))
    comparisons = []
    for left, right in ((ARMS[2], ARMS[0]), (ARMS[2], ARMS[1]), (ARMS[1], ARMS[0])) + tuple(
            (ARMS[2], method) for method in CONTROLS):
        row = dict(left=left, right=right, **paired_statistics(matrices[left], matrices[right],
            contract["bootstrap_samples"], contract["analysis_seed"]))
        row["static_family_ci_low"] = (paired_statistics(matrices[left], matrices[right],
            contract["bootstrap_samples"], contract["analysis_seed"], .025 / 6)["ci_low"] if right in CONTROLS[3:9] else None)
        comparisons.append(row)
    primary = comparisons[:2]
    credit = all(r["ci_low"] > 0 and r["positive_seed_means"] == len(SEEDS) for r in primary)
    static = all(r["static_family_ci_low"] > 0 for r in comparisons if r["static_family_ci_low"] is not None)
    fast = all(r["ci_low"] > 0 for r in comparisons if r["right"] in ("fast", "memory_fast"))
    status = ("DEVELOPMENT_POLICY_GAIN_SUPPORTED" if credit and static and fast else
              "CREDIT_EFFECT_ONLY_NOT_COMPETITIVE" if credit else "NO_CREDIT_EFFECT_ESTABLISHED")
    return summaries, comparisons, dict(status=status, credit_effect_supported=credit,
        beats_all_six_static_controls_bonferroni=static, beats_fast_and_memory_fast=fast,
        warning="Development only; tune has been used before. Three seeds are not 72 independent videos. No external or speed claim.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if not args.input.strip():
        parser.error("input must not be empty")
    directory = Path(args.input).expanduser().resolve()
    contract, rows = progress(directory)
    if args.status:
        print("seed  arm                reported_live  branch_frames  complete")
        for r in rows:
            print("{seed}  {arm:<19s} {reported_steps:>10d} {branch_frames:>14d}  {completed}".format(**r))
        print("CSV may lag one checkpoint. No efficacy analysis until all nine final policies are complete.")
        return
    if not all(r["completed"] for r in rows):
        raise RuntimeError("Only {}/9 completed final policies. Use --status; do not select a seed subset.".format(
            sum(r["completed"] for r in rows)))
    if (contract.get("seeds") != list(SEEDS) or contract.get("arms") != list(ARMS)
            or contract.get("live_steps") != STEPS or contract.get("metric") != METRIC):
        raise ValueError("Not the fixed three-seed protocol")
    for name, digest in dict(contract["source_code"], **contract["code_sha256"]).items():
        if Path(name).name != name or "\\" in name or file_hash(Path(__file__).parent / name) != digest:
            raise ValueError("Frozen development/analysis code changed: " + name)
    trajectories, costs = load_complete(directory, contract)
    summaries, comparisons, decision = summarize(trajectories, contract)
    output = directory / "analysis"
    write_csv(output / "stage23b_policy_summary.csv", summaries)
    write_csv(output / "stage23b_paired.csv", comparisons)
    write_csv(output / "stage23b_training_cost.csv", costs)
    seed_rows = []
    for method in ARMS:
        for seed, videos in zip(SEEDS, trajectories[method]):
            seed_rows.append(dict(seed=seed, method=method, auc=float(np.mean([v["auc"] for v in videos])),
                candidate_ratio=sum(v["candidate_blocks"] for v in videos) / sum(v["frames"] for v in videos)))
    write_csv(output / "stage23b_seed_summary.csv", seed_rows)
    write_json(output / "stage23b_decision.json", sealed(dict(schema=SCHEMA,
        contract_fingerprint=contract["fingerprint"], **decision)))
    print("method                  AUC  candidate%  layers")
    for row in summaries:
        print("{method:<20s} {auc:>8.4f} {ratio:>10.2%} {layers:>7s}".format(**row, ratio=row["candidate_ratio"],
            layers="native" if row["active_layers"] is None else str(row["active_layers"])))
    print("\nH20 minus control (paired video bootstrap after averaging seeds):")
    for row in comparisons:
        if row["left"] == ARMS[2]:
            print("{right:<16s} {mean_delta:>8.4f} [{ci_low:.4f},{ci_high:.4f}] positive seeds={positive_seed_means}/3".format(**row))
    print("Decision: " + decision["status"])
    print(decision["warning"])


if __name__ == "__main__":
    main()
