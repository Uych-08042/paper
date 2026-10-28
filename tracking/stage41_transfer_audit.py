"""Read-only cached audit of Stage-40 stress transfer and full-trajectory tradeoffs."""

import argparse
import csv
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint
from stage27_uav_collect import writer_lock
from stage27_uav_common import failure_events, nonempty_path, publish_identical, scene_geometry, seal, success_auc
from stage27_uav_summary import write_table
from stage34_identity_common import load_sealed as read_sealed
from stage35_supervised_audit import verify_bindings
import stage40_crop_common as source


SCHEMA = "stage41_cached_crop_transfer_audit_v1"
CONFIG = dict(before=10, after=30, reliable_iou=.5, low_iou=.2,
    sustained_iou_difference=.2, sustained_frames=3, control_quartiles=[1, 2, 3],
    bootstrap_samples=5000, bootstrap_seed=2120, closure_atol=1e-10, sign_atol=1e-9)
COHORTS = ("both_reliable", "rescued_low_to_reliable", "harmed_reliable_to_low", "both_low", "other")
GEOMETRY = ("coverage", "projected_gt_side_px", "raw_gt_side_px", "parent_center_error_gt_units",
    "output_center_error_gt_units", "parent_side_ratio", "output_width_ratio", "output_height_ratio",
    "center_error_vector_jump_gt_units")
EVENT_FIELDS = ("sequence", "seed", "arm", "event", "onset_frame", "confirmed_frame", "end_frame",
    "recovery_frame", "duration_frames", "low_iou_frames", "censor", "onset_coverage",
    "onset_parent_center_error_gt_units", "onset_output_center_error_gt_units", "onset_center_error_vector_jump_gt_units")


def code_hashes():
    names = ("stage41_transfer_audit.py", "test_stage41_transfer_audit.py", "stage40_crop_common.py",
        "stage40_crop_experiment.py", "stage40_crop_summary.py", "stage27_uav_common.py", "validation_metrics.py")
    return {n: file_hash(Path(__file__).with_name(n)) for n in names}


def load_source(root):
    root = Path(root).resolve()
    plan = source.load_plan(root)
    from stage40_crop_experiment import smoke_gate
    smoke_gate(root, plan)
    report_path = root / "analysis/stage40_decision.json"
    evidence_path = root / "analysis/stage40_evidence.json"
    report = read_sealed(report_path, source.SCHEMA)
    evidence = read_sealed(evidence_path, source.SCHEMA)
    run = read_sealed(root / "evaluation/run.json", source.SCHEMA)
    statuses = {"NO_MATCHED_AUGMENTATION_GAIN_ESTABLISHED", "INSUFFICIENT_RETENTION_SUPPORT",
        "PASS_MATCHED_AUGMENTATION_DEVELOPMENT_SCREEN"}
    expected = {source.result_path(root, e, m, a).resolve() for e in plan["fit"] for m in source.methods()
        for a in [None] + plan["tasks"][e["sequence"]]}
    states = {"{}:{}".format(f, m) for f in range(source.DESIGN["folds"]) for m in source.methods()}
    if (report.get("status") not in statuses or report.get("plan_fingerprint") != plan["fingerprint"] or
            report.get("run_fingerprint") != run["fingerprint"] or report.get("videos") != 8 or
            report.get("seeds") != list(source.SEEDS) or report.get("new_models") != 16 or report.get("new_full_trajectories") != 32 or
            report.get("training_performed") is not True or report.get("original_checking_labels_loaded") is not False or
            report.get("novel_method_established") is not False or file_hash(evidence_path) != report.get("evidence_sha256") or
            evidence.get("plan_fingerprint") != plan["fingerprint"] or evidence.get("run_fingerprint") != run["fingerprint"] or
            evidence.get("table_sha256") != report["table_sha256"] or run.get("mode") != "evaluation" or
            run.get("methods") != source.methods() or run.get("runtime") != plan["source_runtime"] or
            run.get("teacher_sha256") != plan["source_frozen_neural_model_sha256"]["sgla"] or
            set(run.get("model_state_sha256", {})) != states or
            {p.resolve() for p in (root / "evaluation").rglob("*.npz")} != expected):
        raise ValueError("Need the unchanged complete Stage-40 result, evidence and all 128 committed caches")
    bindings = dict(evidence["source_file_sha256"])
    bindings.update(plan["source_file_sha256"])
    bindings.update(run["model_file_sha256"])
    bindings.update({str(root / "analysis" / n): d for n, d in report["table_sha256"].items()})
    for path in (report_path, evidence_path, root / "manifest.json", root / "evaluation/run.json", root / "smoke/stage40_smoke.json"):
        bindings[str(path)] = file_hash(path)
    verify_bindings(bindings)
    return plan, report, run, bindings


def frame_auc(iou):
    values = np.asarray(iou, float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Need finite valid frame IoUs")
    return (values[:, None] > np.linspace(0, 1, 21)).mean(1) * 100


def auc_partition(clean, augmented):
    clean, augmented = np.asarray(clean, float), np.asarray(augmented, float)
    if clean.shape != augmented.shape:
        raise ValueError("Paired trajectories need equal frame coverage")
    c, a = clean >= CONFIG["reliable_iou"], augmented >= CONFIG["reliable_iou"]
    cl, al = clean < CONFIG["low_iou"], augmented < CONFIG["low_iou"]
    masks = [c & a, cl & a, c & al, cl & al]
    masks.append(~np.logical_or.reduce(masks))
    np.testing.assert_array_equal(np.stack(masks).sum(0), np.ones(len(clean)))
    delta = frame_auc(augmented) - frame_auc(clean)
    rows = []
    for label, mask in zip(COHORTS, masks):
        count = int(mask.sum())
        rows.append(dict(cohort=label, frames=count, frame_fraction=count / len(clean),
            contribution_to_full_auc=float(delta[mask].sum() / len(clean)),
            conditional_auc_delta=float(delta[mask].mean()) if count else None,
            mean_iou_delta=float((augmented[mask] - clean[mask]).mean()) if count else None))
    error = abs(sum(r["contribution_to_full_auc"] for r in rows) - (success_auc(augmented) - success_auc(clean)))
    if error > CONFIG["closure_atol"]:
        raise ValueError("Frame partition does not reproduce the whole-video AUC difference")
    return rows, error


def stress_accounting(clean, augmented):
    """Models' [7 conditions,H] IoUs; keep loss reduction separate from absolute gain."""
    c, a = np.asarray(clean, float), np.asarray(augmented, float)
    if c.shape != a.shape or c.ndim != 2 or c.shape[0] != 7 or c.shape[1] < 2:
        raise ValueError("Need matched seven-condition branches")
    cc, ac = float(c[0, 1:].mean()), float(a[0, 1:].mean())
    cp, ap = float(c[1:, 1:].mean()), float(a[1:, 1:].mean())
    clean_delta, perturbed_delta = ac - cc, ap - cp
    loss_delta = (ac - ap) - (cc - cp)
    residual = abs(loss_delta - (clean_delta - perturbed_delta))
    if residual > CONFIG["closure_atol"]:
        raise ValueError("Stress accounting failed")
    return dict(clean_model_unperturbed_iou=cc, augmented_model_unperturbed_iou=ac,
        clean_model_perturbed_iou=cp, augmented_model_perturbed_iou=ap,
        unperturbed_iou_delta=clean_delta, perturbed_iou_delta=perturbed_delta,
        loss_delta=loss_delta, loss_reduction=-loss_delta,
        contribution_from_unperturbed_decline=-clean_delta, contribution_from_perturbed_gain=perturbed_delta), residual


def geometry_trace(entry, data, plan):
    gt, boxes, parents = data["gt"], data["boxes"], data["previous"]
    n = len(gt)
    gt_side = np.sqrt(gt[:, 2:].prod(1))
    center = lambda b: b[:, :2] + b[:, 2:] / 2
    errors = center(boxes) - center(gt)
    previous_errors = np.vstack([np.zeros(2), errors[:-1]])
    geometries = [scene_geometry(parents[i], gt[i], (entry["frames"][i + 1]["height"], entry["frames"][i + 1]["width"]),
        plan["search_factor"], plan["search_size"]) for i in range(n)]
    return dict(coverage=np.asarray([g["coverage"] for g in geometries]),
        projected_gt_side_px=np.asarray([np.sqrt(g["input_wh"].prod()) for g in geometries]),
        raw_gt_side_px=gt_side, parent_center_error_gt_units=np.linalg.norm(center(parents) - center(gt), axis=1) / gt_side,
        output_center_error_gt_units=np.linalg.norm(errors, axis=1) / gt_side,
        parent_side_ratio=np.sqrt(parents[:, 2:].prod(1)) / gt_side,
        output_width_ratio=boxes[:, 2] / gt[:, 2], output_height_ratio=boxes[:, 3] / gt[:, 3],
        center_error_vector_jump_gt_units=np.linalg.norm(errors - previous_errors, axis=1) / gt_side)


def first_run(mask, count):
    run = 0
    for i, value in enumerate(mask):
        run = run + 1 if value else 0
        if run == count:
            return i - count + 1
    return None


def anchors(clean, augmented):
    n, result = len(clean), []
    for arm, values in (("clean", clean), ("augmented", augmented)):
        events = failure_events(values)
        if events:
            result.append(dict(kind="first_failure", owner=arm, frame_id=int(events[0]["start"] + 1)))
    for sign, kind in ((1, "first_sustained_gain"), (-1, "first_sustained_damage")):
        start = first_run(sign * (augmented - clean) >= CONFIG["sustained_iou_difference"], CONFIG["sustained_frames"])
        if start is not None:
            result.append(dict(kind=kind, owner="paired", frame_id=int(start + 1)))
    for q in CONFIG["control_quartiles"]:
        result.append(dict(kind="fixed_time_q{}".format(q), owner="fixed", frame_id=max(1, n * q // 4)))
    return result


def trajectory_audit(entry, seed, clean, augmented, plan):
    np.testing.assert_array_equal(clean["frame_id"], augmented["frame_id"])
    np.testing.assert_array_equal(clean["gt"], augmented["gt"])
    name, n = entry["sequence"], len(clean["iou"])
    partitions, residual = auc_partition(clean["iou"], augmented["iou"])
    partitions = [dict(sequence=name, seed=seed, **r) for r in partitions]
    traces = dict(clean=clean, augmented=augmented)
    geometry = {arm: geometry_trace(entry, data, plan) for arm, data in traces.items()}
    event_rows, timeline, phases = [], [], []
    for arm, data in traces.items():
        g = geometry[arm]
        for index, event in enumerate(failure_events(data["iou"]), 1):
            start, end = event["start"], event["end"]
            event_rows.append(dict(sequence=name, seed=seed, arm=arm, event=index, onset_frame=start + 1,
                confirmed_frame=event["confirmation"] + 1, end_frame=end + 1,
                recovery_frame=event["recovery"] + 1 if event["recovery"] is not None else None,
                duration_frames=end - start + 1, low_iou_frames=int((data["iou"][start:end + 1] < CONFIG["low_iou"]).sum()),
                censor=event["censor"], onset_coverage=float(g["coverage"][start]),
                onset_parent_center_error_gt_units=float(g["parent_center_error_gt_units"][start]),
                onset_output_center_error_gt_units=float(g["output_center_error_gt_units"][start]),
                onset_center_error_vector_jump_gt_units=float(g["center_error_vector_jump_gt_units"][start])))
    for index, anchor in enumerate(anchors(clean["iou"], augmented["iou"])):
        center_row = anchor["frame_id"] - 1
        ids = np.arange(max(0, center_row - CONFIG["before"]), min(n, center_row + CONFIG["after"] + 1))
        offsets = ids - center_row
        for arm, data in traces.items():
            g = geometry[arm]
            base = dict(sequence=name, seed=seed, anchor_id=index, anchor_kind=anchor["kind"],
                anchor_owner=anchor["owner"], anchor_frame=anchor["frame_id"], arm=arm)
            for row, offset in zip(ids, offsets):
                timeline.append(dict(base, frame_id=int(row + 1), offset=int(offset), iou=float(data["iou"][row]),
                    paired_iou_delta=float(augmented["iou"][row] - clean["iou"][row]),
                    **{k: float(g[k][row]) for k in GEOMETRY}))
            outside = offsets[g["coverage"][ids] < .9]
            for label, mask in (("before", offsets < 0), ("onset", offsets == 0), ("after", offsets > 0)):
                rows = ids[mask]
                phases.append(dict(base, phase=label, sampled_frames=len(rows),
                    mean_iou=float(data["iou"][rows].mean()) if len(rows) else None,
                    first_coverage_lt09_offset=int(outside[0]) if len(outside) else None,
                    **{"mean_" + k: float(g[k][rows].mean()) if len(rows) else None for k in GEOMETRY}))
    base = dict(sequence=name, seed=seed, clean_auc=success_auc(clean["iou"]), augmented_auc=success_auc(augmented["iou"]),
        auc_delta=success_auc(augmented["iou"]) - success_auc(clean["iou"]),
        low_iou_fraction_delta=float((augmented["iou"] < .2).mean() - (clean["iou"] < .2).mean()),
        clean_failure_events=len(failure_events(clean["iou"])), augmented_failure_events=len(failure_events(augmented["iou"])))
    for label in COHORTS:
        base[label + "_auc_contribution"] = next(r["contribution_to_full_auc"] for r in partitions if r["cohort"] == label)
    for arm in traces:
        own = [r for r in event_rows if r["arm"] == arm]
        base[arm + "_first_failure_frame"] = own[0]["onset_frame"] if own else None
        base[arm + "_longest_failure_fraction"] = max((r["duration_frames"] for r in own), default=0) / n
    return base, partitions, event_rows, timeline, phases, residual


def paired(values):
    v = np.asarray(values, float)
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    means = v[rng.integers(0, len(v), (CONFIG["bootstrap_samples"], len(v)))].mean(1)
    low, high = map(float, np.quantile(means, [.025, .975]))
    return dict(videos=len(v), mean_delta=float(v.mean()), ci_low=low, ci_high=high)


def aggregate(seed_rows):
    names = sorted({r["sequence"] for r in seed_rows})
    lookup = {(r["sequence"], r["seed"]): r for r in seed_rows}
    if len(names) != 8 or len(seed_rows) != 16 or set(lookup) != {(n, s) for n in names for s in source.SEEDS}:
        raise ValueError("Need all eight videos and both paired seeds")
    numeric = ("clean_auc", "augmented_auc", "auc_delta", "low_iou_fraction_delta", "clean_failure_events", "augmented_failure_events",
        "clean_longest_failure_fraction", "augmented_longest_failure_fraction", "unperturbed_iou_delta", "perturbed_iou_delta", "loss_delta",
        "loss_reduction", "contribution_from_unperturbed_decline", "contribution_from_perturbed_gain")
    numeric += tuple(label + "_auc_contribution" for label in COHORTS)
    rows = []
    for name in names:
        values = [lookup[(name, s)] for s in source.SEEDS]
        effects = [v["auc_delta"] for v in values]
        row = dict(sequence=name, **{k: float(np.mean([v[k] for v in values])) for k in numeric})
        row.update(seed_auc_deltas="|".join("{:+.8f}".format(v) for v in effects),
            seed_direction="both_positive" if min(effects) > CONFIG["sign_atol"] else
                "both_negative" if max(effects) < -CONFIG["sign_atol"] else "mixed_or_tied",
            reduced_loss_without_absolute_perturbed_gain=bool(row["loss_delta"] < -CONFIG["sign_atol"] and row["perturbed_iou_delta"] <= CONFIG["sign_atol"]))
        rows.append(row)
    metrics = ["auc_delta", "unperturbed_iou_delta", "perturbed_iou_delta", "loss_delta"]
    metrics += [label + "_auc_contribution" for label in COHORTS]
    comparisons = [dict(metric=k, **paired([r[k] for r in rows])) for k in metrics]
    focus = {}
    for kind, row in (("largest_video_gain", max(rows, key=lambda r: r["auc_delta"])),
            ("largest_video_loss", min(rows, key=lambda r: r["auc_delta"]))):
        contributions = {c: row[c + "_auc_contribution"] for c in COHORTS}
        focus[kind] = dict(sequence=row["sequence"], auc_delta=row["auc_delta"], seed_auc_deltas=row["seed_auc_deltas"],
            seed_direction=row["seed_direction"], auc_contributions=contributions,
            unperturbed_iou_delta=row["unperturbed_iou_delta"], perturbed_iou_delta=row["perturbed_iou_delta"], loss_delta=row["loss_delta"])
    focus.update(seed_consistency={label: sum(r["seed_direction"] == label for r in rows)
        for label in ("both_positive", "both_negative", "mixed_or_tied")},
        reduced_loss_without_absolute_perturbed_gain_videos=sum(r["reduced_loss_without_absolute_perturbed_gain"] for r in rows))
    return rows, comparisons, focus


def audit(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if root == output or output in root.parents or root in output.parents:
        raise ValueError("Use a separate Stage-41 output directory outside the frozen Stage-40 tree")
    plan, original, run, bindings = load_source(root)
    hashes = code_hashes()
    identity = seal(dict(schema=SCHEMA, source_stage40=str(root), source_decision_fingerprint=original["fingerprint"],
        configuration=CONFIG, code_sha256=hashes, source_bindings_fingerprint=fingerprint(bindings)))
    target = output / "stage41_diagnostic.json"
    # Audit once; committed outputs are verified and reused without rewriting any table.
    if target.exists():
        saved = read_sealed(target, SCHEMA)
        evidence_path = output / "stage41_evidence.json"
        evidence = read_sealed(evidence_path, SCHEMA)
        if (read_sealed(output / "plan.json", SCHEMA) != identity or
                saved.get("configuration") != CONFIG or saved.get("code_sha256") != hashes or
                saved.get("source_decision_fingerprint") != original["fingerprint"] or
                file_hash(evidence_path) != saved["evidence_sha256"] or evidence["source_file_sha256"] != bindings):
            raise ValueError("Frozen Stage-41 source/code/configuration changed; use a new output directory")
        verify_bindings({str(output / n): d for n, d in saved["table_sha256"].items()})
        print("Stage-41: COMPLETE_CACHED_TRANSFER_AUDIT (reuse, audited)")
        return saved
    seed_rows, partitions, events, timeline, phases, stress_rows = [], [], [], [], [], []
    max_auc_residual = max_stress_residual = 0.
    for entry in plan["fit"]:
        for seed in source.SEEDS:
            full = {}
            for arm in source.ARMS:
                method = arm + "_seed" + str(seed)
                full[arm] = source.load_result(source.result_path(root, entry, method), plan, run, entry, method)
            row, p, e, t, ph, residual = trajectory_audit(entry, seed, full["clean"], full["augmented"], plan)
            partitions.extend(p)
            events.extend(e)
            timeline.extend(t)
            phases.extend(ph)
            max_auc_residual = max(max_auc_residual, residual)
            own_stress = []
            for anchor in plan["tasks"][entry["sequence"]]:
                data = {}
                for arm in source.ARMS:
                    method = arm + "_seed" + str(seed)
                    data[arm] = source.load_result(source.result_path(root, entry, method, anchor), plan, run, entry, method, anchor)
                info, residual = stress_accounting(data["clean"]["iou"], data["augmented"]["iou"])
                own_stress.append(info)
                stress_rows.append(dict(sequence=entry["sequence"], seed=seed, anchor=anchor, **info))
                max_stress_residual = max(max_stress_residual, residual)
            row.update({k: float(np.mean([v[k] for v in own_stress])) for k in own_stress[0]})
            seed_rows.append(row)
    videos, comparisons, focus = aggregate(seed_rows)
    # Reproduce frozen whole-video means and the two paired statistics, using every video/seed.
    index = {r["metric"]: r for r in comparisons}
    for metric, key in (("auc_delta", "primary"), ("loss_delta", "stress_augmented_minus_clean")):
        np.testing.assert_allclose(index[metric]["mean_delta"], original[key]["mean_delta"], atol=1e-10, rtol=0)
    with (root / "analysis/stage40_policy_summary.csv").open(encoding="utf-8", newline="") as handle:
        policies = {r["arm"]: r for r in csv.DictReader(handle)}
    for arm in source.ARMS:
        np.testing.assert_allclose(np.mean([r[arm + "_auc"] for r in videos]), float(policies[arm]["auc"]), atol=1e-10, rtol=0)
    tables = dict(video_summary=videos, seed_video_summary=seed_rows, auc_partition=partitions,
        event_summary=events, anchor_timeline=timeline, anchor_summary=phases, stress_accounting=stress_rows, paired=comparisons)
    verify_bindings(bindings)
    with writer_lock(output):
        if target.exists():
            raise ValueError("Another Stage-41 writer completed; rerun to audit the saved result")
        if not (output / "plan.json").exists() and any(p.name != "collect.lock" for p in output.iterdir()):
            raise ValueError("Use an empty Stage-41 output directory")
        publish_identical(output / "plan.json", identity)
        table_hashes = {}
        for label, rows in tables.items():
            path = output / ("stage41_" + label + ".csv")
            write_table(path, rows, EVENT_FIELDS if label == "event_summary" else None)
            table_hashes[path.name] = file_hash(path)
        evidence = seal(dict(schema=SCHEMA, source_stage40=str(root), source_file_sha256=bindings, table_sha256=table_hashes))
        publish_identical(output / "stage41_evidence.json", evidence)
        report = seal(dict(schema=SCHEMA, role="posthoc_consumed_development_cached_trajectory_and_stress_audit",
            status="COMPLETE_CACHED_TRANSFER_AUDIT", configuration=CONFIG, code_sha256=hashes,
            source_decision_fingerprint=original["fingerprint"], source_status=original["status"],
            videos=8, seeds=list(source.SEEDS), paired_trajectories=16, stress_anchor_pairs=48,
            focus=focus, paired=comparisons, maximum_auc_partition_residual=max_auc_residual,
            maximum_stress_accounting_residual=max_stress_residual, table_sha256=table_hashes,
            evidence_sha256=file_hash(output / "stage41_evidence.json"), training_performed=False,
            new_inference_performed=False, image_decoding_performed=False, original_checking_labels_loaded=False,
            adoption_gate=None, next_step="Inspect exact AUC partitions, seed consistency, absolute stress responses and onset geometry before choosing a training intervention",
            warnings=["This is a post-hoc audit of consumed outcomes; largest gain/loss cases and onset windows are selected after outcomes.",
                "AUC cohorts are disjoint outcome-conditioned frame sets, not online triggers or independent cohorts; contributions sum exactly to the whole-video difference.",
                "Stress loss reduction equals unperturbed decline plus perturbed gain; reduced loss alone does not establish higher absolute robustness.",
                "All geometry uses each recorded trajectory's own predicted parents. Coverage is not visibility and center-error jumps do not identify response peaks or semantic distractors.",
                "Full-video caches do not contain response maps. This audit cannot assign a failure to response, size or offset, or provide a same-crop counterfactual.",
                "Event and temporal-window ordering is descriptive; error propagation and crop feedback are not separately identified causal effects.",
                "Average seeds and anchors within video. Video bootstrap is exploratory, conditional on fixed overlapping OOF fits; frames/seeds are not additional independent videos.",
                "No new policy, compression method, speedup or external generalization is established."]))
        verify_bindings(bindings)
        publish_identical(target, report)
    for kind in ("largest_video_gain", "largest_video_loss"):
        value = focus[kind]
        print("{}: {} AUC={:+.4f}, seeds={}, {}".format(kind, value["sequence"], value["auc_delta"], value["seed_auc_deltas"], value["seed_direction"]))
    print("Reduced stress loss without absolute perturbed gain: {}/8 videos".format(focus["reduced_loss_without_absolute_perturbed_gain_videos"]))
    print("Stage-41: COMPLETE_CACHED_TRANSFER_AUDIT; no new method adoption gate")
    print("Reports: " + str(output))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True, help="Completed Stage-40 root")
    parser.add_argument("--output_dir", type=nonempty_path, required=True)
    args = parser.parse_args()
    root = Path(args.input).expanduser().resolve()
    if not (root / "analysis/stage40_decision.json").is_file():
        parser.error("Stage-40 decision missing: {}. Set STAGE40_ROOT from ~/workspace/paper.".format(root / "analysis/stage40_decision.json"))
    audit(root, Path(args.output_dir).expanduser())


if __name__ == "__main__":
    main()
