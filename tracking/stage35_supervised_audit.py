"""Post-hoc audit of existing Stage-35 supervised OOF traces; no training/inference."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from stage4_common import file_hash
from stage27_uav_collect import writer_lock
from stage27_uav_common import failure_events, publish_identical, read_boxes, seal, success_auc
from stage27_uav_summary import interval, write_table
from stage34_identity_experiment import audit_trace
from stage35_preserve_common import ARMS, BASELINES, DESIGN, load_plan, load_sealed, nonempty_path


SCHEMA = "stage35b_posthoc_supervised_baseline_audit_v1"
AUDIT = dict(bootstrap_samples=5000, bootstrap_seed=2113,
    reliable_iou=.5, minimum_conditional_frames=10)
METRICS = ("auc", "low_iou_fraction", "failure_events", "instrumented_ms",
    "fast_reliable_auc", "sgla_reliable_auc", "sgla_failure_auc")


def verify_bindings(bindings):
    for path, digest in bindings.items():
        if not Path(path).is_file() or file_hash(path) != digest:
            raise ValueError("Frozen Stage-35 artifact changed: " + str(path))


def load_source(root):
    """Bind all old models/caches/labels; never rewrite the original decision."""
    root = Path(root).resolve()
    plan = load_plan(root)
    decision_path, run_path = root / "analysis/stage35_decision.json", root / "oof/run.json"
    decision, run = load_sealed(decision_path), load_sealed(run_path)
    methods = list(BASELINES) + sorted(a + "_seed" + str(s) for a in ARMS for s in DESIGN["seeds"])
    if (decision.get("role") != "consumed_development_fit_only_group_oof_screen" or
            decision.get("plan_fingerprint") != plan["fingerprint"] or
            decision.get("oof_run_fingerprint") != run["fingerprint"] or
            decision.get("videos") != 8 or decision.get("seeds") != DESIGN["seeds"] or
            decision.get("original_checking_labels_loaded") is not False or
            run.get("plan_fingerprint") != plan["fingerprint"] or run.get("methods") != methods):
        raise ValueError("Need a completed, unmixed original Stage-35 summary")
    bindings = {str(root / "manifest.json"): file_hash(root / "manifest.json"),
        str(decision_path): file_hash(decision_path), str(run_path): file_hash(run_path)}
    model_bindings = run["frozen_model_sha256"]
    expected_models = set()
    for seed in DESIGN["seeds"]:
        path = root / ("training/seed{}/stage35_training_decision.json".format(seed))
        expected_models.add(str(path))
        report = load_sealed(path)
        pairs = {(f, a) for f in range(DESIGN["folds"]) for a in ARMS}
        if (report.get("status") != "COMPLETE_MATCHED_OOF_TRAINING" or report.get("seed") != seed or
                report.get("plan_fingerprint") != plan["fingerprint"] or len(report["arms"]) != len(pairs) or
                {(r["fold"], r["arm"]) for r in report["arms"]} != pairs):
            raise ValueError("Incomplete or mixed frozen training report")
        for record in report["arms"]:
            checkpoint = str(Path(record["checkpoint"]).resolve())
            expected_models.add(checkpoint)
            if model_bindings.get(checkpoint) != record["checkpoint_sha256"]:
                raise ValueError("Training report/checkpoint binding mismatch")
    if set(model_bindings) != expected_models:
        raise ValueError("Need both seed reports and all 40 frozen final checkpoints")
    bindings.update(model_bindings)
    expected_caches = {e["sequence"] + "__" + m + ".npz" for e in plan["fit"] for m in methods}
    if set(decision["cache_sha256"]) != expected_caches or {p.name for p in (root / "oof").glob("*.npz")} != expected_caches:
        raise ValueError("Need exactly the original 96 OOF caches")
    bindings.update({str(root / "oof" / name): digest for name, digest in decision["cache_sha256"].items()})
    verify_bindings(bindings)
    rows = []
    for entry in plan["fit"]:
        name = entry["sequence"]
        annotation = Path(plan["root"]) / entry["annotation"]
        bindings[str(annotation)] = entry["annotation_sha256"]
        verify_bindings({str(annotation): entry["annotation_sha256"]})
        gt = read_boxes(annotation)[1:]
        fold = next(f["fold"] for f in plan["folds"] if name in f["held_out"])
        traces = {}
        for method in methods:
            path = root / "oof" / (name + "__" + method + ".npz")
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata_json"].item()))
                data = {k: archive[k] for k in archive.files if k != "metadata_json"}
            if metadata != dict(schema=plan["schema"], run_fingerprint=run["fingerprint"], sequence=name, fold=fold, method=method):
                raise ValueError("Mixed source trace metadata")
            audit_trace(data, entry)
            np.testing.assert_array_equal(data["gt"], gt)
            traces[method] = data
        masks = dict(fast_reliable_auc=traces["fast_initial"]["iou"] >= AUDIT["reliable_iou"],
            sgla_reliable_auc=traces["sgla"]["iou"] >= AUDIT["reliable_iou"],
            sgla_failure_auc=traces["sgla"]["iou"] < AUDIT["reliable_iou"])
        for method, data in traces.items():
            arm, seed = method.rsplit("_seed", 1) if "_seed" in method else (method, "baseline")
            events = failure_events(data["iou"])
            timing = data["elapsed_ms"][5:] if len(data["elapsed_ms"]) > 5 else data["elapsed_ms"]
            row = dict(sequence=name, source_group_proxy=entry["source_group_proxy"], fold=fold,
                method=arm, seed=int(seed) if seed != "baseline" else seed,
                tracked_frames=len(gt), auc=success_auc(data["iou"]),
                low_iou_fraction=float(np.mean(data["iou"] < .2)), failure_events=len(events),
                first_failure_frame=int(events[0]["start"] + 1) if events else None,
                instrumented_ms=float(timing.mean()))
            for metric, mask in masks.items():
                count = int(mask.sum())
                row[metric + "_frames"] = count
                row[metric] = success_auc(data["iou"][mask]) if count >= AUDIT["minimum_conditional_frames"] else None
            rows.append(row)
    verify_bindings(bindings)
    return plan, decision, rows, bindings


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["sequence"], row["method"])].append(row)
    result = []
    names = sorted({r["sequence"] for r in rows})
    for name in names:
        for method in BASELINES + ARMS:
            values = groups[(name, method)]
            expected = set(DESIGN["seeds"]) if method in ARMS else {"baseline"}
            if len(values) != len(expected) or {r["seed"] for r in values} != expected:
                raise ValueError("Missing/duplicate seed-video rows")
            row = dict(sequence=name, method=method, seeds=len(values), fold=values[0]["fold"])
            for metric in METRICS:
                supported = [v[metric] is not None for v in values]
                if any(supported) != all(supported):
                    raise ValueError("Seed-dependent conditional cohort support")
                row[metric] = float(np.mean([v[metric] for v in values])) if all(supported) else None
            result.append(row)
    return result


def analyze(rows):
    averaged = aggregate(rows)
    index = {(r["sequence"], r["method"]): r for r in averaged}
    seeds = {(r["sequence"], r["method"], r["seed"]): r for r in rows}
    names = sorted({r["sequence"] for r in rows})
    rng = np.random.default_rng(AUDIT["bootstrap_seed"])
    specs = [("auc", r) for r in BASELINES + tuple(a for a in ARMS if a != "supervised")]
    specs += [(metric, r) for metric in ("low_iou_fraction", "failure_events") for r in BASELINES]
    specs += [("fast_reliable_auc", "fast_initial"), ("sgla_reliable_auc", "sgla"), ("sgla_failure_auc", "sgla")]
    paired, influence, stability, per_seed = [], [], [], []
    for metric, reference in specs:
        supported = [n for n in names if index[(n, "supervised")][metric] is not None and index[(n, reference)][metric] is not None]
        deltas = [index[(n, "supervised")][metric] - index[(n, reference)][metric] for n in supported]
        low, high = interval(deltas, rng, AUDIT["bootstrap_samples"])
        direction = -1 if metric in ("low_iou_fraction", "failure_events") else 1
        paired.append(dict(contrast="supervised_minus_" + reference, metric=metric, videos=len(deltas),
            mean_delta=float(np.mean(deltas)) if deltas else None,
            median_delta=float(np.median(deltas)) if deltas else None, ci_low=low, ci_high=high,
            worst_video_delta=float(min(deltas)) if deltas else None,
            wins=sum(direction * d > 1e-9 for d in deltas), ties=sum(abs(d) <= 1e-9 for d in deltas), losses=sum(direction * d < -1e-9 for d in deltas),
            favorable_direction="negative" if direction == -1 else "positive"))
    for reference in BASELINES:
        deltas = np.array([index[(n, "supervised")]["auc"] - index[(n, reference)]["auc"] for n in names])
        positive = np.maximum(deltas, 0).sum()
        for name, delta in zip(names, deltas):
            seed_deltas = [seeds[(name, "supervised", s)]["auc"] - index[(name, reference)]["auc"] for s in DESIGN["seeds"]]
            omitted = float((deltas.sum() - delta) / (len(names) - 1))
            influence.append(dict(sequence=name, fold=index[(name, "supervised")]["fold"], reference=reference,
                supervised_auc=index[(name, "supervised")]["auc"], reference_auc=index[(name, reference)]["auc"],
                auc_delta=float(delta), macro_mean_contribution=float(delta / len(names)),
                positive_gain_share=float(max(delta, 0) / positive) if positive > 0 else None,
                leave_one_video_out_mean_delta=omitted,
                mean_sign_changes_if_omitted=bool(float(deltas.mean()) * omitted < 0),
                seed_deltas="|".join("{:.8f}".format(d) for d in seed_deltas), worst_seed_delta=float(min(seed_deltas))))
        reliable_metric = "fast_reliable_auc" if reference == "fast_initial" else "sgla_reliable_auc"
        for name in names:
            baseline = index[(name, reference)]
            values = [seeds[(name, "supervised", s)] for s in DESIGN["seeds"]]
            reliable = [v[reliable_metric] - baseline[reliable_metric] for v in values] if baseline[reliable_metric] is not None else []
            stability.append(dict(sequence=name, reference=reference, original_stable=baseline["failure_events"] == 0,
                introduced_failure_any_seed=baseline["failure_events"] == 0 and any(v["failure_events"] > 0 for v in values),
                baseline_failure_events=baseline["failure_events"],
                supervised_failure_events_by_seed="|".join(str(v["failure_events"]) for v in values),
                first_supervised_failure_frame=min((v["first_failure_frame"] for v in values if v["first_failure_frame"] is not None), default=None),
                reliable_auc_mean_delta=float(np.mean(reliable)) if reliable else None,
                reliable_auc_worst_seed_delta=float(min(reliable)) if reliable else None,
                low_iou_fraction_delta=index[(name, "supervised")]["low_iou_fraction"] - baseline["low_iou_fraction"]))
        for seed in DESIGN["seeds"]:
            d = [seeds[(n, "supervised", seed)]["auc"] - index[(n, reference)]["auc"] for n in names]
            per_seed.append(dict(seed=seed, reference=reference, videos=len(names),
                supervised_auc=float(np.mean([seeds[(n, "supervised", seed)]["auc"] for n in names])),
                mean_auc_delta=float(np.mean(d)), worst_video_delta=float(min(d)),
                wins=sum(v > 1e-9 for v in d), ties=sum(abs(v) <= 1e-9 for v in d), losses=sum(v < -1e-9 for v in d)))
    return averaged, paired, influence, stability, per_seed


def audit(root):
    root = Path(root).resolve()
    plan, original, rows, bindings = load_source(root)
    averaged, paired, influence, stability, per_seed = analyze(rows)
    # Check reproduced macro scores against the existing summary, without rewriting it.
    policy_path = root / "analysis/stage35_policy_summary.csv"
    bindings[str(policy_path)] = file_hash(policy_path)
    with policy_path.open(newline="", encoding="utf-8") as handle:
        policies = list(csv.DictReader(handle))
    if len(policies) != len(BASELINES + ARMS) or {r["method"] for r in policies} != set(BASELINES + ARMS):
        raise ValueError("Original policy summary is incomplete/mixed")
    for policy in policies:
        values = [r for r in averaged if r["method"] == policy["method"]]
        if int(policy["videos"]) != len(plan["fit"]):
            raise ValueError("Original summary video count mismatch")
        for metric in ("auc", "low_iou_fraction", "instrumented_ms"):
            np.testing.assert_allclose(float(policy[metric]), np.mean([r[metric] for r in values]), rtol=0, atol=1e-9)
    summaries = {}
    for reference in BASELINES:
        p = next(r for r in paired if r["metric"] == "auc" and r["contrast"] == "supervised_minus_" + reference)
        impacts = [r for r in influence if r["reference"] == reference]
        stable = [r for r in stability if r["reference"] == reference]
        shares = sorted((r["positive_gain_share"] for r in impacts if r["positive_gain_share"] is not None), reverse=True)
        summaries[reference] = dict(paired_auc=p,
            seed_mean_auc_deltas=[r["mean_auc_delta"] for r in per_seed if r["reference"] == reference],
            positive_seed_means=sum(r["mean_auc_delta"] > 0 for r in per_seed if r["reference"] == reference),
            largest_single_video_positive_gain_share=shares[0] if shares else None,
            largest_two_video_positive_gain_share=sum(shares[:2]) if shares else None,
            leave_one_video_out_mean_range=[min(r["leave_one_video_out_mean_delta"] for r in impacts), max(r["leave_one_video_out_mean_delta"] for r in impacts)],
            original_stable_videos=sum(r["original_stable"] for r in stable),
            introduced_failure_videos=sum(r["introduced_failure_any_seed"] for r in stable))
    verify_bindings(bindings)
    code_files = (Path(__file__), Path(__file__).with_name("test_stage35_supervised_audit.py"))
    report = seal(dict(schema=SCHEMA, role="posthoc_descriptive_supervised_baseline_audit",
        status="COMPLETE_POSTHOC_SUPERVISED_AUDIT", configuration=AUDIT,
        source_plan_fingerprint=plan["fingerprint"], original_decision_fingerprint=original["fingerprint"],
        original_status=original["status"], summaries=summaries,
        source_file_sha256=bindings, code_sha256={p.name: file_hash(p) for p in code_files},
        videos=len(plan["fit"]), training_performed=False, new_inference_performed=False,
        original_checking_labels_loaded=False, adoption_gate=None,
        warnings=["Supervised was noticed after Stage-35 outcomes; all new contrasts are post-hoc development diagnostics.",
            "Average seeds within each source video before video bootstrap; two seeds are not extra independent videos.",
            "OOF folds share training groups; conditional-on-fit intervals omit retraining uncertainty.",
            "Leave-one-video-out values are influence diagnostics on fixed predictions, not retraining or permission to discard videos.",
            "Reliable/failure cohorts are defined by baseline outcomes; GT is used only for offline analysis.",
            "No confirmatory pass/fail, speedup, novel-method or external-generalization claim is established."],
        next_step="Keep the original Stage-35 failure frozen; interpret baseline consistency and contribution before designing a new study"))
    output = root / "analysis/supervised_baseline_audit_v1"
    with writer_lock(output):
        publish_identical(output / "stage35b_audit.json", report)
        for label, values in (("seed_video_summary", rows), ("video_summary", averaged), ("paired", paired),
                ("video_influence", influence), ("stability", stability), ("seed_summary", per_seed)):
            write_table(output / ("stage35b_" + label + ".csv"), values)
        verify_bindings(bindings)
    print("Supervised baseline: post-hoc video-paired AUC audit")
    for reference in BASELINES:
        p = summaries[reference]["paired_auc"]
        print("vs {:<12s} {:+.4f} [{:+.4f},{:+.4f}] W/T/L={}/{}/{}; positive seeds={}/2".format(
            reference, p["mean_delta"], p["ci_low"], p["ci_high"], p["wins"], p["ties"], p["losses"], summaries[reference]["positive_seed_means"]))
    print("Original Stage-35 status remains: " + original["status"])
    print("Reports: " + str(output))
    print("No new training/inference; no post-hoc method PASS is assigned.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True, help="Completed Stage-35 root")
    args = parser.parse_args()
    audit(Path(args.input).expanduser())


if __name__ == "__main__":
    main()
