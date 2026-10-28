"""Frozen full-video gates for online recovery; no onset-conditioned selection."""

import argparse
from pathlib import Path

import numpy as np

from stage4_common import file_hash
from stage27_uav_collect import writer_lock
from stage27_uav_common import failure_events, nonempty_path, publish_identical, seal, success_auc, tracking_iou
from stage27_uav_summary import write_table
from stage43_recovery_common import (DESIGN, MODELS, POLICIES, SCHEMA, SEEDS, load_plan,
    load_result, read_sealed, result_path, tasks, verify_bindings)


def paired(values):
    x = np.asarray(values, float)
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Paired effect needs at least two complete finite videos")
    rng = np.random.default_rng(DESIGN["bootstrap_seed"])
    means = x[rng.integers(len(x), size=(DESIGN["bootstrap_samples"], len(x)))].mean(1)
    low, high = np.quantile(means, [.025, .975])
    return dict(videos=len(x), mean_delta=float(x.mean()), ci_low=float(low), ci_high=float(high),
        wins=int((x > 1e-9).sum()), ties=int((np.abs(x) <= 1e-9).sum()), losses=int((x < -1e-9).sum()),
        worst_video_delta=float(x.min()), minimum_leave_one_video_out=float(((x.sum() - x) / (len(x) - 1)).min()))


def video_row(entry, seed, model, policy, data, baseline):
    iou, base = data["iou"], baseline["iou"]
    reliable = base >= .5
    events, old = failure_events(iou), failure_events(base)
    changed = np.max(np.abs(data["boxes"] - data["native_boxes"]), axis=1) > 1e-9
    return dict(sequence=entry["sequence"], seed=seed, model=model, policy=policy, tracked_frames=len(iou),
        auc=success_auc(iou), low_iou_fraction=float((iou < .2).mean()), failure_events=len(events),
        longest_failure_fraction=max((e["end"] - e["start"] + 1 for e in events), default=0) / len(iou),
        reliable_frames=int(reliable.sum()), reliable_auc=success_auc(iou[reliable]) if reliable.sum() >= DESIGN["minimum_reliable_frames"] else None,
        baseline_stable=not old, introduced_failure=not old and bool(events), alarm_frames=int(data["alarm"].sum()),
        interventions=int(data["eligible"].sum()), alternate_calls=int((data["head_calls"] - 1).sum()),
        total_head_calls=int(data["head_calls"].sum()), changed_frames=int(changed.sum()),
        changed_ratio=float(changed.mean()), alternate_ratio=float((data["head_calls"] - 1).mean()))


def analyze(rows, plan):
    names = [e["sequence"] for e in plan["fit"]]
    lookup = {(r["sequence"], r["seed"], r["model"], r["policy"]):r for r in rows}
    expected = {(n,s,m,p) for n in names for s in SEEDS for m in MODELS for p in POLICIES}
    if len(lookup) != len(rows) or set(lookup) != expected:
        raise ValueError("Need all complete unmixed Stage-43 video/seed/model/policy rows")
    videos, policies, contrasts = [], [], []
    metrics = ("auc", "low_iou_fraction", "failure_events", "longest_failure_fraction", "reliable_auc")
    averaged = {}
    for n in names:
        for m in MODELS:
            for p in POLICIES:
                rs = [lookup[n,s,m,p] for s in SEEDS]
                mean = {k:float(np.mean([r[k] for r in rs])) if all(r[k] is not None for r in rs) else None for k in metrics}
                averaged[n,m,p] = mean
                videos.append(dict(sequence=n, model=m, policy=p, seeds=len(SEEDS), **mean,
                    introduced_failure_any_seed=any(r["introduced_failure"] for r in rs),
                    changed_frames=sum(r["changed_frames"] for r in rs), alternate_calls=sum(r["alternate_calls"] for r in rs)))
    comparisons = [(p,"native") for p in POLICIES if p != "native"] + [("history_search",p) for p in ("motion_hold", "local_search")]
    for m in MODELS:
        for p in POLICIES:
            rs = [r for r in rows if r["model"] == m and r["policy"] == p]
            policies.append(dict(model=m, policy=p, videos=len(names), auc=float(np.mean([averaged[n,m,p]["auc"] for n in names])),
                low_iou_fraction=float(np.mean([averaged[n,m,p]["low_iou_fraction"] for n in names])),
                failure_events=float(np.mean([r["failure_events"] for r in rs])),
                alarm_frames=sum(r["alarm_frames"] for r in rs), interventions=sum(r["interventions"] for r in rs),
                alternate_calls=sum(r["alternate_calls"] for r in rs), total_head_calls=sum(r["total_head_calls"] for r in rs),
                changed_ratio=sum(r["changed_frames"] for r in rs) / sum(r["tracked_frames"] for r in rs)))
        for left,right in comparisons:
            for metric in metrics:
                support = [n for n in names if averaged[n,m,left][metric] is not None and averaged[n,m,right][metric] is not None]
                values = [averaged[n,m,left][metric] - averaged[n,m,right][metric] for n in support]
                if len(values) < 2:
                    result = dict(videos=len(values), mean_delta=float(np.mean(values)) if values else None, ci_low=None, ci_high=None)
                else:
                    result = paired(values)
                contrasts.append(dict(model=m, contrast=left + "_minus_" + right, metric=metric,
                    **result, supported_videos="|".join(support), excluded_videos="|".join(n for n in names if n not in support)))
    def get(m, contrast, metric="auc"):
        return next(r for r in contrasts if r["model"] == m and r["contrast"] == contrast and r["metric"] == metric)
    primary = get("augmented", "history_search_minus_native")
    reliable = get("augmented", "history_search_minus_native", "reliable_auc")
    low = get("augmented", "history_search_minus_native", "low_iou_fraction")
    controls = [get("augmented", "history_search_minus_" + p) for p in ("motion_hold", "local_search")]
    seed_gains = [float(np.mean([lookup[n,s,"augmented","history_search"]["auc"] - lookup[n,s,"augmented","native"]["auc"] for n in names])) for s in SEEDS]
    affected = [r for r in rows if r["model"] == "augmented" and r["policy"] == "history_search"]
    introduced = len({r["sequence"] for r in affected if r["introduced_failure"]})
    changed_videos = len({r["sequence"] for r in affected if r["changed_frames"]})
    changed_ratio = sum(r["changed_frames"] for r in affected) / sum(r["tracked_frames"] for r in affected)
    primary_pass = (primary["mean_delta"] >= DESIGN["minimum_auc_gain"] and primary["ci_low"] > 0 and
        primary["wins"] >= DESIGN["minimum_positive_videos"] and primary["minimum_leave_one_video_out"] > 0 and all(v > 0 for v in seed_gains))
    control_pass = all(r["mean_delta"] > 0 for r in controls)
    retention = (reliable["videos"] >= DESIGN["minimum_reliable_videos"] and reliable["ci_low"] is not None and
        reliable["mean_delta"] >= DESIGN["reliable_mean_floor"] and reliable["ci_low"] >= DESIGN["reliable_ci_floor"] and
        low["mean_delta"] <= 0 and introduced == 0)
    transfer = get("clean", "history_search_minus_native")
    transfer_low = get("clean", "history_search_minus_native", "low_iou_fraction")
    transfer_pass = (transfer["mean_delta"] >= DESIGN["reliable_mean_floor"] and transfer["ci_low"] >= DESIGN["reliable_ci_floor"] and
        transfer_low["mean_delta"] <= 0 and not any(r["introduced_failure"] for r in rows if r["model"] == "clean" and r["policy"] == "history_search"))
    use = changed_videos >= DESIGN["minimum_changed_videos"] and changed_ratio >= DESIGN["minimum_changed_ratio"]
    passed = primary_pass and control_pass and retention and transfer_pass and use
    decision = dict(status="PASS_ONLINE_RECOVERY_DEVELOPMENT_PILOT" if passed else "NO_GT_FREE_RECOVERY_GAIN_ESTABLISHED",
        pilot_pass=passed, primary_auc_pass=primary_pass, simple_control_pass=control_pass,
        retention_guard_pass=retention, clean_transfer_guard_pass=transfer_pass, nontrivial_usage_pass=use,
        primary=primary, reliable_guard=reliable, low_iou_effect=low, controls=controls, clean_transfer=transfer,
        primary_seed_gains=seed_gains, introduced_failure_videos=introduced, changed_videos=changed_videos, changed_ratio=changed_ratio)
    return dict(seed_video_summary=rows, video_summary=videos, policy_summary=policies, paired=contrasts), decision


def event_rows(entry, seed, model, policy, data):
    rows = []
    native_iou = tracking_iou(data["native_boxes"], data["gt"])
    for i in np.flatnonzero(data["eligible"]):
        complete = i + 41 <= len(data["iou"])
        rows.append(dict(sequence=entry["sequence"], seed=seed, model=model, policy=policy, frame_id=int(i + 1),
            native_proposal_iou=float(native_iou[i]), executed_iou=float(data["iou"][i]), choice=int(data["choice"][i]),
            extra_call=int(data["head_calls"][i] - 1), score_floor=float(data["score_floor"][i]),
            native_score=float(data["native_score"][i]), innovation=float(data["innovation"][i]),
            complete_followup=complete, post40_mean_iou=float(data["iou"][i + 1:i + 41].mean()) if complete else None,
            role="Own-trajectory descriptive event; not a matched-start intervention effect"))
    return rows


def summarize(root):
    from stage43_recovery_experiment import native_parity, smoke_gate, validate_run
    root = Path(root).resolve()
    plan = load_plan(root)
    smoke_gate(root, plan)
    run = read_sealed(root / "full/run.json", SCHEMA)
    validate_run(run, plan, "full")
    expected = {result_path(root, "full", *t).name for t in tasks(plan, "full")}
    if {p.name for p in (root / "full/traces").glob("*.npz")} != expected:
        raise ValueError("Need all 128 complete Stage-43 full-video traces; inspect --mode status")
    bindings = dict(plan["source_file_sha256"])
    for p in (root / "manifest.json", root / "full/run.json", root / "smoke/run.json", root / "smoke/stage43_smoke.json"):
        bindings[str(p)] = file_hash(p)
    rows, events, parities = [], [], []
    max_decode_error = 0.
    for entry in plan["fit"]:
        for seed in SEEDS:
            for model in MODELS:
                traces = {}
                for policy in POLICIES:
                    path = result_path(root, "full", entry, seed, model, policy)
                    traces[policy] = load_result(path, plan, run, entry, seed, model, policy)
                    for p in (path, path.with_suffix(".json")):
                        bindings[str(p)] = file_hash(p)
                parities.append(native_parity(traces["native"], plan, entry, seed, model))
                for policy,data in traces.items():
                    rows.append(video_row(entry, seed, model, policy, data, traces["native"]))
                    events.extend(event_rows(entry, seed, model, policy, data))
                    max_decode_error = max(max_decode_error, float(data["native_decode_error"].max()), float(data["alternate_decode_error"].max()))
    tables, decision = analyze(rows, plan)
    tables["interventions"] = events
    output = root / "analysis"
    with writer_lock(output):
        path = output / "stage43_decision.json"
        if path.exists():
            saved = read_sealed(path, SCHEMA)
            evidence = read_sealed(output / "stage43_evidence.json", SCHEMA)
            if (file_hash(output / "stage43_evidence.json") != saved["evidence_sha256"] or
                    evidence["source_file_sha256"] != bindings or evidence["table_sha256"] != saved["table_sha256"]):
                raise ValueError("Completed Stage-43 evidence changed")
            verify_bindings({str(output / n):d for n,d in saved["table_sha256"].items()})
            print("Stage-43: " + saved["status"] + " (reuse, audited)")
            return saved
        hashes = {}
        event_fields = ("sequence", "seed", "model", "policy", "frame_id", "native_proposal_iou", "executed_iou", "choice",
            "extra_call", "score_floor", "native_score", "innovation", "complete_followup", "post40_mean_iou", "role")
        for label,values in tables.items():
            target = output / ("stage43_" + label + ".csv")
            write_table(target, values, event_fields if label == "interventions" else None)
            hashes[target.name] = file_hash(target)
        verify_bindings(bindings)
        publish_identical(output / "stage43_evidence.json", seal(dict(schema=SCHEMA, source_file_sha256=bindings, table_sha256=hashes)))
        report = seal(dict(schema=SCHEMA, role="consumed_development_online_recovery_feasibility", **decision,
            plan_fingerprint=plan["fingerprint"], run_fingerprint=run["fingerprint"], videos=len(plan["fit"]), seeds=list(SEEDS),
            full_trajectories=len(rows), maximum_native_box_error=max(p["box_error"] for p in parities),
            maximum_native_iou_error=max(p["iou_error"] for p in parities), maximum_decode_error=max_decode_error,
            exact_call_and_prefix_budget_pass=True, training_performed=False, original_checking_labels_loaded=False,
            new_inference_performed=True, novel_method_established=False, table_sha256=hashes,
            evidence_sha256=file_hash(output / "stage43_evidence.json"),
            next_step="Freeze for a separately designed independent check" if decision["pilot_pass"] else
                "Preserve the result; do not search triggers, budgets, motion windows or checkpoints on these outcomes",
            warnings=["All eight videos and earlier outcomes informed design. This is development evidence, not external confirmation.",
                "The controller receives only its own previous predictions/current response; GT after initialization enters evaluation only.",
                "Accepted history means model-consistent, not GT-confirmed reliable. No UAV camera-motion compensation is used.",
                "Both students have six physical blocks. Search policies may run the SAME network twice; extra calls are explicitly charged.",
                "Local/history search share trigger, expansion, selection and prefix cap; own trajectories can yield different trigger counts, so realized compute is not necessarily identical.",
                "Seeds average inside videos; bootstrap is conditional on fixed overlapping OOF fits and cannot establish training uncertainty.",
                "Reliable cohorts and event IoUs are offline descriptions, never online triggers. Event follow-ups are not matched-start causal gains.",
                "Motion/search recovery are established ideas. No compression-specific novelty, deployment speedup or energy improvement is established."]))
        publish_identical(path, report)
    for r in tables["policy_summary"]:
        print("{:10s} {:15s} AUC={:.4f} low-IoU={:.2%} extra={}".format(r["model"], r["policy"], r["auc"], r["low_iou_fraction"], r["alternate_calls"]))
    print("Stage-43: " + report["status"])
    print("Reports: " + str(output))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True)
    args = parser.parse_args()
    summarize(Path(args.input).expanduser())


if __name__ == "__main__":
    main()
