"""Fixed model/initial-state factorial and explicitly censored recovery diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np

from stage4_common import file_hash
from stage27_uav_collect import writer_lock
from stage27_uav_common import nonempty_path, publish_identical, scene_geometry, seal, tracking_iou
from stage27_uav_summary import write_table
from stage42_state_common import (ARMS, DESIGN, SCHEMA, load_plan, load_result, prior, read_sealed,
    source_traces, task_path, training, verify_bindings)


EFFECTS = ("augmented_clean_parent_minus_augmented_own", "clean_augmented_parent_minus_clean_own",
    "augmented_minus_clean_at_clean_parent", "augmented_minus_clean_at_augmented_parent",
    "native_augmented_minus_clean", "model_order_average", "parent_order_average", "model_parent_interaction")


def primary_key():
    return "followup_1_{}_mean_iou".format(DESIGN["followup_frames"])


def branch_metrics(iou):
    iou = np.asarray(iou, float)
    complete = len(iou) >= DESIGN["followup_frames"] + 1
    first_stable = prior.first_run(iou >= DESIGN["reliable_iou"], DESIGN["stable_run"])
    relapse = None
    if first_stable is not None:
        begin = first_stable + DESIGN["stable_run"]
        later = prior.first_run(iou[begin:] < DESIGN["low_iou"], DESIGN["stable_run"])
        relapse = begin + later if later is not None else None
    return dict(first_iou=float(iou[0]), post_available_mean_iou=float(iou[1:].mean()) if len(iou) > 1 else None,
        **{primary_key(): float(iou[1:DESIGN["followup_frames"] + 1].mean()) if complete else None},
        early_1_10_mean_iou=float(iou[1:11].mean()) if len(iou) >= 11 else None,
        late_21_40_mean_iou=float(iou[21:41].mean()) if len(iou) >= 41 else None,
        low_fraction_full_followup=float((iou[1:DESIGN["followup_frames"] + 1] < DESIGN["low_iou"]).mean()) if complete else None,
        first_three_reliable_offset=first_stable, first_relapse_offset=relapse)


def factorial(corners):
    if any(value is None for row in corners for value in row):
        return {label: None for label in EFFECTS}, 0.
    cc, ca, ac, aa = map(float, np.asarray(corners).ravel())
    model = .5 * ((ac - cc) + (aa - ca))
    parent = .5 * ((ca - cc) + (aa - ac))
    result = dict(zip(EFFECTS, (ac - aa, ca - cc, ac - cc, aa - ca,
        aa - cc, model, parent, aa - ac - ca + cc)))
    residual = abs(model + parent - (aa - cc))
    if residual > 1e-10:
        raise ValueError("Model/initial-parent factorial accounting failed")
    return result, residual


def task_rows(plan, entry, task, data):
    horizon = len(data["gt"])
    branches, metrics = [], {}
    for m, model in enumerate(ARMS):
        for p, parent in enumerate(ARMS):
            result = branch_metrics(data["iou"][m, p])
            metrics[(model, parent)] = result
            branches.append(dict(sequence=task["sequence"], seed=task["seed"], frame_id=task["frame_id"],
                model=model, initial_parent=parent, available_followup=horizon - 1,
                censored=horizon < DESIGN["followup_frames"] + 1, **result))
    row = dict(sequence=task["sequence"], seed=task["seed"], frame_id=task["frame_id"], frames=horizon,
        available_followup=horizon - 1, censored=horizon < DESIGN["followup_frames"] + 1,
        memberships=json.dumps(task["memberships"], sort_keys=True))
    for p, parent in enumerate(ARMS):
        box = data["previous"][0, p, 0]
        geometry = scene_geometry(box, data["gt"][0], data["image_hw"][0], plan["search_factor"], plan["search_size"])
        row[parent + "_parent_gt_iou"] = float(tracking_iou(box, data["gt"][0]))
        row[parent + "_initial_crop_coverage"] = float(geometry["coverage"])
    for model in ARMS:
        for parent in ARMS:
            row[model + "_from_" + parent + "_full_followup_iou"] = metrics[(model, parent)][primary_key()]
    effects, maximum_residual = {}, 0.
    for metric in (primary_key(), "first_iou", "post_available_mean_iou", "early_1_10_mean_iou", "late_21_40_mean_iou"):
        effect, residual = factorial([[metrics[(m, p)][metric] for p in ARMS] for m in ARMS])
        maximum_residual = max(maximum_residual, residual)
        effects[metric] = effect
    row.update(effects[primary_key()])
    for metric in ("first_iou", "post_available_mean_iou", "early_1_10_mean_iou", "late_21_40_mean_iou"):
        row["augmented_parent_swap_" + metric] = effects[metric][EFFECTS[0]]
    return row, branches, effects[primary_key()], maximum_residual


def groups(task):
    values = set()
    for member in task["memberships"]:
        if member["kind"] == "fixed_time":
            values.add(("fixed_time", None))
        else:
            values.add((member["owner"] + "_first_failure", member["lag"]))
    return values


def paired(values):
    if not values:
        return dict(videos=0, mean_delta=None, ci_low=None, ci_high=None, positive_videos=0, zero_videos=0, negative_videos=0)
    values = np.asarray(values, float)
    rng = np.random.default_rng(DESIGN["bootstrap_seed"])
    if len(values) > 1:
        means = values[rng.integers(0, len(values), (DESIGN["bootstrap_samples"], len(values)))].mean(1)
        low, high = map(float, np.quantile(means, [.025, .975]))
    else:
        low = high = None
    return dict(videos=len(values), mean_delta=float(values.mean()), ci_low=low, ci_high=high,
        positive_videos=int((values > 1e-9).sum()), zero_videos=int((np.abs(values) <= 1e-9).sum()), negative_videos=int((values < -1e-9).sum()))


def aggregate(plan, task_effects):
    lookup = {(t["sequence"], t["seed"], t["frame_id"]): (t, e) for t, e in task_effects}
    expected = {(t["sequence"], t["seed"], t["frame_id"]) for t in plan["tasks"]}
    if len(lookup) != len(task_effects) or set(lookup) != expected:
        raise ValueError("Need all deduplicated tasks exactly once")
    specs = [(owner + "_first_failure", lag) for owner in ARMS for lag in DESIGN["failure_lags"]] + [("fixed_time", None)]
    video_rows, summaries = [], []
    names = [e["sequence"] for e in plan["fit"]]
    for cohort, lag in specs:
        for effect in EFFECTS:
            video_values, supported = [], []
            for name in names:
                by_seed, counts = {}, {}
                for seed in training.SEEDS:
                    values = [e[effect] for t, e in task_effects if t["sequence"] == name and t["seed"] == seed
                        and (cohort, lag) in groups(t) and e[effect] is not None]
                    by_seed[seed] = float(np.mean(values)) if values else None
                    counts[seed] = len(values)
                complete = all(by_seed[s] is not None for s in training.SEEDS)
                mean = float(np.mean(list(by_seed.values()))) if complete else None
                video_rows.append(dict(sequence=name, cohort=cohort, lag=lag, metric=primary_key(), effect=effect,
                    supported_seed_count=sum(v is not None for v in by_seed.values()),
                    supported_anchor_counts="|".join(str(counts[s]) for s in training.SEEDS),
                    seed_mean_deltas="|".join("missing" if by_seed[s] is None else "{:+.8f}".format(by_seed[s]) for s in training.SEEDS),
                    mean_delta=mean))
                if complete:
                    video_values.append(mean)
                    supported.append(name)
            summaries.append(dict(cohort=cohort, lag=lag, metric=primary_key(), effect=effect,
                **paired(video_values), supported_videos="|".join(supported),
                excluded_videos="|".join(n for n in names if n not in supported)))
    return summaries, video_rows


def summarize(root):
    root = Path(root).resolve()
    plan = load_plan(root)
    from stage42_state_experiment import smoke_gate, validate_run
    smoke_gate(root, plan)
    run = read_sealed(root / "full/run.json", SCHEMA)
    validate_run(run, plan, "full")
    expected = {task_path(root, "full", t).name for t in plan["tasks"]}
    if {p.name for p in (root / "full/branches").glob("*.npz")} != expected:
        raise ValueError("Need all {} complete four-corner anchor packs; inspect --mode status".format(len(expected)))
    bindings = dict(plan["source_file_sha256"])
    for path in (root / "manifest.json", root / "full/run.json", root / "smoke/run.json", root / "smoke/stage42_smoke.json"):
        bindings[str(path)] = file_hash(path)
    tasks, branches, task_effects, references = [], [], [], {}
    max_residual = max_box_error = max_iou_error = max_decode_error = 0.
    for task in plan["tasks"]:
        entry = next(e for e in plan["fit"] if e["sequence"] == task["sequence"])
        key = (task["sequence"], task["seed"])
        if key not in references:
            references[key] = source_traces(plan, entry, task["seed"])
        path = task_path(root, "full", task)
        data, parity = load_result(path, plan, run, entry, task, references[key])
        row, branch, effects, residual = task_rows(plan, entry, task, data)
        tasks.append(row)
        branches.extend(branch)
        task_effects.append((task, effects))
        for p in (path, path.with_suffix(".json")):
            bindings[str(p)] = file_hash(p)
        max_residual = max(max_residual, residual)
        max_box_error = max(max_box_error, max(p["max_box_error"] for p in parity))
        max_iou_error = max(max_iou_error, max(p["max_iou_error"] for p in parity))
        max_decode_error = max(max_decode_error, float(data["decode_error"].max()))
    grouped, videos = aggregate(plan, task_effects)
    tables = dict(task_summary=tasks, branch_summary=branches, group_summary=grouped, video_effects=videos)
    output = root / "analysis"
    with writer_lock(output):
        path = output / "stage42_diagnostic.json"
        if path.exists():
            saved = read_sealed(path, SCHEMA)
            evidence_path = output / "stage42_evidence.json"
            evidence = read_sealed(evidence_path, SCHEMA)
            if file_hash(evidence_path) != saved["evidence_sha256"] or evidence["source_file_sha256"] != bindings:
                raise ValueError("Completed Stage-42 evidence changed")
            verify_bindings({str(output / n): d for n, d in saved["table_sha256"].items()})
            print("Stage-42: COMPLETE_FROZEN_STATE_SWAP_DIAGNOSTIC (reuse, audited)")
            return saved
        hashes = {}
        for label, rows in tables.items():
            target = output / ("stage42_" + label + ".csv")
            write_table(target, rows)
            hashes[target.name] = file_hash(target)
        verify_bindings(bindings)
        evidence = seal(dict(schema=SCHEMA, source_file_sha256=bindings, table_sha256=hashes))
        publish_identical(output / "stage42_evidence.json", evidence)
        focus = [r for r in grouped if r["cohort"] == "augmented_first_failure" and r["effect"] in
            (EFFECTS[0], "augmented_minus_clean_at_clean_parent")]
        report = seal(dict(schema=SCHEMA, status="COMPLETE_FROZEN_STATE_SWAP_DIAGNOSTIC", role="posthoc_frozen_initial_parent_intervention",
            plan_fingerprint=plan["fingerprint"], run_fingerprint=run["fingerprint"], videos=8, seeds=list(training.SEEDS),
            unique_anchor_packs=len(tasks), branches=len(branches), model_frames=4 * sum(t["frames"] for t in plan["tasks"]),
            complete_followup_anchors=sum(not r["censored"] for r in tasks), short_followup_anchors=sum(r["censored"] for r in tasks),
            skipped_after_video_end=len(plan["skipped_after_end"]), focus=focus,
            maximum_native_box_error=max_box_error, maximum_native_iou_error=max_iou_error,
            maximum_decode_error=max_decode_error, maximum_factorial_residual=max_residual,
            table_sha256=hashes, evidence_sha256=file_hash(output / "stage42_evidence.json"),
            training_performed=False, original_checking_labels_loaded=False, new_inference_performed=True, adoption_gate=None,
            next_step="Inspect parent-swap persistence, model effects on matched initial parents and donor quality before designing a recovery intervention",
            warnings=["All eight source groups and GT-defined failure onsets are consumed development outcomes; swaps are offline diagnostic interventions, not deployable triggers.",
                "Donor parents are frozen predictions, never GT. The clean parent is not guaranteed to be better; its quality is logged, not used to filter anchors.",
                "Only the initial predicted parent is swapped; every branch then follows its own outputs with original first-frame template. This does not reconstruct a different past trajectory.",
                "Both diagonal branches must reproduce original trajectories. Factorial effects include autonomous continuation and are conditional on the selected initial states, not unique attribution of the entire original histories.",
                "The first output at offset 0 is excluded from full follow-up 1..40. Short horizons remain missing for that metric; available-post means are separately labeled.",
                "Both seeds need valid full follow-up within each video/cohort for video bootstrap. No-failure or censored videos are explicitly excluded from those conditional summaries, never assigned zero effect.",
                "Failure-owner/lag groups can share a deduplicated anchor. Fixed-time controls are separate; frames, anchors and seeds are not independent videos.",
                "First-three-reliable and relapse offsets are descriptive GT evaluation, not online reset commands. Fewer low-IoU frames alone do not prove identity recovery.",
                "This experiment establishes no new compact method, full-video AUC gain, speedup or external generalization."]))
        publish_identical(path, report)
    print("Stage-42: COMPLETE_FROZEN_STATE_SWAP_DIAGNOSTIC; native replay max error {:.6g}px".format(max_box_error))
    for row in focus:
        if row["effect"] == EFFECTS[0]:
            value = "missing" if row["mean_delta"] is None else "{:+.5f}".format(row["mean_delta"])
            print("augmented onset +{}: clean-parent swap {} mean IoU; {} two-seed videos".format(row["lag"], value, row["videos"]))
    print("Reports: " + str(output))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True)
    args = parser.parse_args()
    summarize(Path(args.input).expanduser())


if __name__ == "__main__":
    main()
