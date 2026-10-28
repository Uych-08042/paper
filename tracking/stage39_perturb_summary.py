"""Video-paired excess perturbation sensitivity, separate from absolute accuracy."""

import argparse
from pathlib import Path

import numpy as np

from stage4_common import file_hash
from stage27_uav_collect import writer_lock
from stage27_uav_common import nonempty_path, publish_identical, seal
from stage27_uav_summary import write_table
from stage39_perturb_common import (DESIGN, MODELS, PERTURBATIONS, SCHEMA, load_branch,
    load_plan, read_sealed, source_trace, task_path, tasks, verify_bindings)


def branch_rows(entry, anchor, data):
    rows = []
    for m, model in enumerate(MODELS):
        clean = data["iou"][m, 0]
        for p, name in enumerate(PERTURBATIONS[1:], 1):
            value = data["iou"][m, p]
            rows.append(dict(sequence=entry["sequence"], source_group_proxy=entry["source_group_proxy"],
                anchor=anchor, model=model, perturbation=name, clean_first_iou=float(clean[0]),
                perturbed_first_iou=float(value[0]), first_iou_loss=float(clean[0] - value[0]),
                clean_post_iou=float(clean[1:].mean()), perturbed_post_iou=float(value[1:].mean()),
                post_iou_loss=float((clean[1:] - value[1:]).mean()),
                late_iou_loss=float((clean[horizon_tail(len(clean)):] - value[horizon_tail(len(clean)):]).mean()),
                post_low_iou_increase=float((value[1:] < .2).mean() - (clean[1:] < .2).mean())))
    return rows


def horizon_tail(horizon):
    return min(5, horizon - 1)  # offsets 6..20 in the frozen full experiment


METRICS = ("clean_first_iou", "perturbed_first_iou", "first_iou_loss", "clean_post_iou", "perturbed_post_iou",
    "post_iou_loss", "late_iou_loss", "post_low_iou_increase")


def paired(values):
    values = np.asarray(values, float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Need at least two finite independent video rows")
    rng = np.random.default_rng(DESIGN["bootstrap_seed"])
    sampled = values[rng.integers(0, len(values), (DESIGN["bootstrap_samples"], len(values)))].mean(1)
    low, high = map(float, np.quantile(sampled, [.025, .975]))
    return dict(videos=len(values), mean_delta=float(values.mean()), median_delta=float(np.median(values)),
        ci_low=low, ci_high=high, positive_videos=int((values > 1e-9).sum()),
        zero_videos=int((np.abs(values) <= 1e-9).sum()), negative_videos=int((values < -1e-9).sum()),
        minimum_leave_one_video_out=float(min(np.delete(values, i).mean() for i in range(len(values)))))


def analyze(rows, plan):
    expected = {(e["sequence"], a, m, p) for e, a in tasks(plan, "full") for m in MODELS for p in PERTURBATIONS[1:]}
    lookup = {(r["sequence"], r["anchor"], r["model"], r["perturbation"]): r for r in rows}
    if set(lookup) != expected or len(rows) != len(expected):
        raise ValueError("Need complete unmixed anchor/model/perturbation rows")
    names = [e["sequence"] for e in plan["fit"]]
    groups = [e["source_group_proxy"] for e in plan["fit"]]
    if len(names) != 8 or len(set(groups)) != 8:
        raise ValueError("Need eight original distinct source-group proxy videos")
    video = []
    for name in names:
        for model in MODELS:
            selected = [r for r in rows if r["sequence"] == name and r["model"] == model]
            video.append(dict(sequence=name, model=model, anchors=len(plan["tasks"][name]), perturbations=6,
                **{k: float(np.mean([r[k] for r in selected])) for k in METRICS}))
    index = {(r["sequence"], r["model"]): r for r in video}
    models = [dict(model=m, videos=8, **{k: float(np.mean([index[(n, m)][k] for n in names])) for k in METRICS}) for m in MODELS]
    contrasts = []
    specs = [("fast_initial", "sgla"), ("supervised_seed2110", "sgla"), ("supervised_seed2111", "sgla"),
        ("supervised_mean", "sgla"), ("supervised_mean", "fast_initial")]
    def value(name, model, metric):
        if model == "supervised_mean":
            return .5 * sum(index[(name, m)][metric] for m in MODELS[2:])
        return index[(name, model)][metric]
    for left, right in specs:
        for metric in ("first_iou_loss", "post_iou_loss", "late_iou_loss", "post_low_iou_increase"):
            delta = [value(n, left, metric) - value(n, right, metric) for n in names]
            contrasts.append(dict(contrast=left + "_minus_" + right, metric=metric,
                role="primary" if (left, right, metric) == ("fast_initial", "sgla", "post_iou_loss") else "secondary_descriptive",
                **paired(delta)))
    primary = next(r for r in contrasts if r["role"] == "primary")
    checks = dict(minimum_effect_pass=primary["mean_delta"] >= DESIGN["minimum_excess_iou"],
        interval_pass=primary["ci_low"] > DESIGN["minimum_ci_low"],
        cross_video_direction_pass=primary["positive_videos"] >= DESIGN["minimum_positive_videos"],
        leave_one_video_out_pass=primary["minimum_leave_one_video_out"] > DESIGN["minimum_leave_one_out"])
    influence = []
    values = [index[(n, "fast_initial")]["post_iou_loss"] - index[(n, "sgla")]["post_iou_loss"] for n in names]
    for i, name in enumerate(names):
        influence.append(dict(sequence=name, excess_iou_loss=values[i], leave_one_video_out=float(np.delete(values, i).mean()),
            fast_clean_post_iou=index[(name, "fast_initial")]["clean_post_iou"],
            sgla_clean_post_iou=index[(name, "sgla")]["clean_post_iou"]))
    # Perturbation-specific summaries are descriptive; they never select a winning direction/amplitude.
    condition = []
    for model in MODELS:
        for p in PERTURBATIONS[1:]:
            per_video = [float(np.mean([r["post_iou_loss"] for r in rows
                if r["sequence"] == n and r["model"] == model and r["perturbation"] == p])) for n in names]
            condition.append(dict(model=model, perturbation=p, **paired(per_video)))
    return dict(branch_summary=rows, video_summary=video, model_summary=models,
        paired=contrasts, primary_video_influence=influence, condition_summary=condition), primary, checks


def summarize(root):
    root = Path(root).resolve()
    plan = load_plan(root)
    from stage39_perturb_experiment import smoke_gate, validate_run
    smoke = smoke_gate(root, plan)
    run = read_sealed(root / "full/run.json", SCHEMA)
    validate_run(run, plan, "full")
    if run["frozen_neural_model_sha256"] != smoke["frozen_neural_model_sha256"]:
        raise ValueError("Full models differ from frozen smoke")
    expected = {task_path(root, "full", e, a).name for e, a in tasks(plan, "full")}
    if {p.name for p in (root / "full/branches").glob("*.npz")} != expected:
        raise ValueError("Need all 24 complete anchor packs; inspect --mode status")
    rows, hashes, bindings, references = [], {}, dict(plan["source_file_sha256"]), {}
    maximum_decode, maximum_fast_parity = 0., 0.
    for entry, anchor in tasks(plan, "full"):
        if entry["sequence"] not in references:
            references[entry["sequence"]] = source_trace(plan, entry)
        path = task_path(root, "full", entry, anchor)
        data = load_branch(path, plan, run, entry, anchor, references[entry["sequence"]])
        rows.extend(branch_rows(entry, anchor, data))
        hashes[path.name] = file_hash(path)
        bindings[str(path)], bindings[str(path.with_suffix(".json"))] = hashes[path.name], file_hash(path.with_suffix(".json"))
        maximum_decode = max(maximum_decode, float(data["decode_error"].max()))
        maximum_fast_parity = max(maximum_fast_parity, float(data["fast_parity_error"].max()))
    for path in (root / "manifest.json", root / "full/run.json", root / "smoke/run.json", root / "smoke/stage39_smoke.json"):
        bindings[str(path)] = file_hash(path)
    tables, primary, checks = analyze(rows, plan)
    supported = all(checks.values())
    status = "EXCESS_SENSITIVITY_SUPPORTED_ON_CONSUMED_DEVELOPMENT" if supported else "NO_EXCESS_SENSITIVITY_ESTABLISHED"
    output = root / "analysis"
    with writer_lock(output):
        target = output / "stage39_decision.json"
        if target.exists():
            saved = read_sealed(target, SCHEMA)
            if saved["source_file_sha256"] != bindings:
                raise ValueError("Completed Stage-39 summary source changed")
            verify_bindings({str(output / n): d for n, d in saved["table_sha256"].items()})
            print("Stage-39: " + saved["status"] + " (reuse, audited)")
            return saved
        table_hashes = {}
        for label, values in tables.items():
            path = output / ("stage39_" + label + ".csv")
            write_table(path, values)
            table_hashes[path.name] = file_hash(path)
        verify_bindings(bindings)
        report = seal(dict(schema=SCHEMA, role="consumed_development_frozen_surgical_compression_screen", status=status,
            plan_fingerprint=plan["fingerprint"], run_fingerprint=run["fingerprint"],
            videos=8, anchors=24, unique_branch_trajectories=672, branch_frames=672 * DESIGN["horizon"],
            primary=primary, screen=checks, excess_sensitivity_supported=supported,
            maximum_decode_error=maximum_decode, maximum_fast_prefix_output_error=maximum_fast_parity,
            source_file_sha256=bindings, cache_sha256=hashes, table_sha256=table_hashes,
            training_performed=False, learned_method_gain_established=False, original_checking_labels_loaded=False,
            next_step="Test a distinct training mechanism with matched training controls before fresh-group validation" if supported else
                "Keep this fixed result; compression-specific instability premise is not established under this intervention",
            warnings=["Eight consumed development groups; not external confirmation or a novel-method result.",
                "Primary comparison is exact six-block truncation versus original seven-executed-block SGLA with selector; shared pretrained weights, no matched new training.",
                "This isolates the deployed surgical change including removal of candidate computation and selection; not a universal depth/capacity/compression law.",
                "Fine-tuned students are secondary diagnostics; their difference also includes previous training and overlapping OOF fits.",
                "All starts use the native SGLA predicted parent; this is a shared-state stress test, not each model's natural anchor distribution.",
                "Anchors use timestamps, not failures. Perturbation amplitudes and directions are fixed; no GT reset or selected-on-outcome window.",
                "Offsets 2..20 include persistent input consequences, model errors and crop feedback; they do not identify feedback alone.",
                "Clean-minus-perturbed losses can be negative and depend on clean performance/floor effects; inspect both absolute clean and perturbed IoU.",
                "Average anchors/directions within videos; seeds and branch frames are not extra independent videos. Bootstrap remains conditional on fixed overlapping fits.",
                "No full-video AUC, accuracy gain, speedup, UAV specificity or deployment robustness claim is established."]))
        publish_identical(target, report)
    for row in tables["model_summary"]:
        print("{:<23s} clean={:.4f} perturbed={:.4f} loss={:+.4f}".format(row["model"], row["clean_post_iou"],
            row["perturbed_post_iou"], row["post_iou_loss"]))
    print("Primary excess IoU loss: {:+.5f} [{:+.5f},{:+.5f}], positive videos {}/8".format(
        primary["mean_delta"], primary["ci_low"], primary["ci_high"], primary["positive_videos"]))
    print("Stage-39: " + status)
    print("Reports: " + str(output))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True)
    args = parser.parse_args()
    summarize(Path(args.input).expanduser())


if __name__ == "__main__":
    main()
