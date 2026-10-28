"""Does deleting the native candidate structure leave a matched adaptation gap?"""

import argparse
from pathlib import Path

import numpy as np

from stage4_common import file_hash
from stage27_uav_collect import writer_lock
from stage27_uav_common import failure_events, nonempty_path, publish_identical, seal, success_auc
from stage27_uav_summary import write_table
from stage44_matched_common import (DESIGN, METHODS, SCHEMA, SEEDS, load_plan, load_trace,
    original_plan, read_sealed, reused_trace, trace_path, verify_bindings)


CONTRASTS = dict(initial_structure_gap=("retained_initial","six_initial"),
    adapted_structure_gap=("retained_adapted","six_adapted"),
    six_adaptation=("six_adapted","six_initial"),retained_adaptation=("retained_adapted","retained_initial"))


def paired(values):
    x = np.asarray(values,float)
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Need at least two finite paired video effects")
    rng = np.random.default_rng(DESIGN["bootstrap_seed"])
    draws = x[rng.integers(len(x),size=(DESIGN["bootstrap_samples"],len(x)))].mean(1)
    low,high = np.quantile(draws,[.025,.975])
    return dict(videos=len(x),mean_delta=float(x.mean()),ci_low=float(low),ci_high=float(high),
        wins=int((x>1e-9).sum()),ties=int((np.abs(x)<=1e-9).sum()),losses=int((x < -1e-9).sum()),
        worst_video_delta=float(x.min()),minimum_leave_one_video_out=float(((x.sum()-x)/(len(x)-1)).min()))


def analyze(rows,plan):
    names = [e["sequence"] for e in plan["fit"]]
    lookup = {(r["sequence"],r["seed"],r["method"]):r for r in rows}
    expected = {(n,s,m) for n in names for s in SEEDS for m in METHODS}
    if len(lookup) != len(rows) or set(lookup) != expected:
        raise ValueError("Need all complete unmixed factorial video/seed rows")
    videos,policies,pairs = [],[],[]
    metrics = ("auc","low_iou_fraction","failure_events","longest_failure_fraction")
    averaged = {}
    for n in names:
        for m in METHODS:
            rs = [lookup[n,s,m] for s in SEEDS]
            mean = {k:float(np.mean([r[k] for r in rs])) for k in metrics}
            averaged[n,m] = mean
            videos.append(dict(sequence=n,method=m,seeds=len(SEEDS),**mean))
    for m in METHODS:
        policies.append(dict(method=m,videos=len(names),**{k:float(np.mean([averaged[n,m][k] for n in names])) for k in metrics},
            provenance="new_matched_retained_training" if m == "retained_adapted" else "reused_frozen_source"))
    for contrast,(left,right) in CONTRASTS.items():
        for metric in metrics:
            values = [averaged[n,left][metric]-averaged[n,right][metric] for n in names]
            pairs.append(dict(contrast=contrast,metric=metric,**paired(values)))
    interaction = [averaged[n,"retained_adapted"]["auc"]-averaged[n,"six_adapted"]["auc"]-
        averaged[n,"retained_initial"]["auc"]+averaged[n,"six_initial"]["auc"] for n in names]
    pairs.append(dict(contrast="structure_by_adaptation_interaction",metric="auc",**paired(interaction)))
    def get(label,metric="auc"):
        return next(r for r in pairs if r["contrast"] == label and r["metric"] == metric)
    primary,retention = get("adapted_structure_gap"),get("retained_adaptation")
    seed_gaps = [float(np.mean([lookup[n,s,"retained_adapted"]["auc"]-lookup[n,s,"six_adapted"]["auc"] for n in names])) for s in SEEDS]
    stable = [(n,s) for n in names for s in SEEDS if lookup[n,s,"retained_initial"]["failure_events"] == 0]
    introduced = len({n for n,s in stable if lookup[n,s,"retained_adapted"]["failure_events"] > 0})
    control_valid = (retention["mean_delta"] >= DESIGN["retained_mean_floor"] and retention["ci_low"] >= DESIGN["retained_ci_floor"] and
        get("retained_adaptation","low_iou_fraction")["mean_delta"] <= 0 and introduced == 0)
    gap = (primary["mean_delta"] >= DESIGN["minimum_gap"] and primary["ci_low"] > 0 and
        primary["wins"] >= DESIGN["minimum_positive_videos"] and primary["minimum_leave_one_video_out"] > 0 and
        all(v > 0 for v in seed_gaps) and get("adapted_structure_gap","low_iou_fraction")["mean_delta"] <= 0)
    supported = bool(control_valid and gap)
    status = ("MATCHED_RETAINED_CONTROL_DEGRADED" if not control_valid else
        "DEVELOPMENT_COMPRESSION_GAP_SUPPORTED" if supported else "NO_MATCHED_COMPRESSION_GAP_ESTABLISHED")
    influence = []
    for i,n in enumerate(names):
        gain = averaged[n,"retained_adapted"]["auc"]-averaged[n,"six_adapted"]["auc"]
        others = [averaged[o,"retained_adapted"]["auc"]-averaged[o,"six_adapted"]["auc"] for o in names if o != n]
        influence.append(dict(sequence=n,adapted_gap=gain,leave_one_video_out=float(np.mean(others)),
            seed_gaps="|".join("{:+.8f}".format(lookup[n,s,"retained_adapted"]["auc"]-lookup[n,s,"six_adapted"]["auc"]) for s in SEEDS)))
    decision = dict(status=status,compression_gap_supported=supported,retained_control_valid=bool(control_valid),
        raw_gap_gate_pass=bool(gap),primary=primary,retained_adaptation_guard=retention,
        initial_gap=get("initial_structure_gap"),interaction=get("structure_by_adaptation_interaction"),
        primary_seed_gaps=seed_gaps,initial_stable_videos=len({n for n,s in stable}),introduced_failure_videos=introduced)
    return dict(seed_video_summary=rows,video_summary=videos,policy_summary=policies,paired=pairs,video_influence=influence),decision


def summarize(root):
    from stage44_matched_experiment import smoke_gate,validate_run
    root = Path(root).resolve()
    plan = load_plan(root)
    smoke_gate(root,plan)
    run = read_sealed(root / "evaluation/run.json",SCHEMA)
    validate_run(run,plan)
    old_plan = original_plan(plan)
    expected = {trace_path(root,e,s).name for e in plan["fit"] for s in SEEDS}
    if {p.name for p in (root / "evaluation/traces").glob("*.npz")} != expected:
        raise ValueError("Need all 16 complete retained OOF videos before summary")
    rows = []
    bindings = dict(plan["source_file_sha256"])
    bindings.update(run["file_sha256"])
    for p in (root / "manifest.json",root / "evaluation/run.json",root / "smoke/stage44_smoke.json"):
        bindings[str(p)] = file_hash(p)
    counts = np.zeros(6,np.int64)
    max_error,head_calls,block_calls = 0.,0,0
    for entry in plan["fit"]:
        for seed in SEEDS:
            path = trace_path(root,entry,seed)
            new = load_trace(path,plan,run,entry,seed)
            for p in (path,path.with_suffix(".json")):
                bindings[str(p)] = file_hash(p)
            head_calls += int(new["head_calls"].sum())
            block_calls += int(new["block_calls"].sum())
            max_error = max(max_error,float(new["decode_error"].max()))
            counts += np.bincount(new["layer"]-7,minlength=6)
            for method in METHODS:
                data = new if method == "retained_adapted" else reused_trace(plan,entry,seed,method,old_plan)
                events = failure_events(data["iou"])
                rows.append(dict(sequence=entry["sequence"],seed=seed,method=method,auc=success_auc(data["iou"]),
                    low_iou_fraction=float((data["iou"]<.2).mean()),failure_events=len(events),
                    longest_failure_fraction=max((e["end"]-e["start"]+1 for e in events),default=0)/len(data["iou"])))
    tables,decision = analyze(rows,plan)
    output = root / "analysis"
    with writer_lock(output):
        path = output / "stage44_decision.json"
        if path.exists():
            saved = read_sealed(path,SCHEMA)
            evidence_path = output / "stage44_evidence.json"
            evidence = read_sealed(evidence_path,SCHEMA)
            if (file_hash(evidence_path) != saved["evidence_sha256"] or evidence["source_file_sha256"] != bindings or
                    evidence["table_sha256"] != saved["table_sha256"]):
                raise ValueError("Completed Stage-44 evidence changed")
            verify_bindings({str(output/n):d for n,d in saved["table_sha256"].items()})
            print("Stage-44: " + saved["status"] + " (reuse, audited)")
            return saved
        hashes = {}
        for label,values in tables.items():
            target = output / ("stage44_"+label+".csv")
            write_table(target,values)
            hashes[target.name] = file_hash(target)
        verify_bindings(bindings)
        publish_identical(output / "stage44_evidence.json",seal(dict(schema=SCHEMA,source_file_sha256=bindings,table_sha256=hashes)))
        report = seal(dict(schema=SCHEMA,role="consumed_development_matched_structure_adaptation_prerequisite",**decision,
            plan_fingerprint=plan["fingerprint"],run_fingerprint=run["fingerprint"],videos=8,seeds=list(SEEDS),
            new_models=8,reused_six_adapted_models=8,new_full_trajectories=16,training_performed=True,
            retained_inference_head_calls=head_calls,retained_inference_block_calls=block_calls,
            retained_layer_distribution="|".join("l{}:{}".format(i+7,int(n)) for i,n in enumerate(counts)),
            maximum_dense_decode_error=max_error,original_checking_labels_loaded=False,novel_method_established=False,
            table_sha256=hashes,evidence_sha256=file_hash(output / "stage44_evidence.json"),
            next_step="Design an independently checked mechanism for the supported structural gap" if decision["compression_gap_supported"] else
                "Preserve result; no candidate-layer, step, fold, learning-rate or checkpoint search on these outcomes",
            warnings=["This is a prerequisite test, not a new method or a compression benchmark success.",
                "All eight groups and source fits are consumed development evidence; seeds average within videos before conditional bootstrap.",
                "Only common L5-L6 are adapted. This does not compare fully trained architecture optima or rule out other training budgets.",
                "Retained SGLA stores all original candidates and selector but executes exactly one candidate; it is not a sequential 12-block ViT.",
                "Candidate/selector weights stay frozen; prefix adaptation can change hard layer decisions. The treatment is the whole native candidate/selection structure, not pure depth.",
                "Both adaptations use identical frame draws, frozen SGLA crop parents, two clean views, optimizer and update counts; their FLOPs differ by design.",
                "The six-adapted checkpoint is reused byte-for-byte from Stage-40 clean; no after-result checkpoint selection or new six-model search.",
                "Evaluation follows own predictions after the initial GT box. No recovery module, KD, contrastive loss or oracle outputs are used.",
                "A degraded retained control makes the gap interpretation invalid, not proof that six layers are universally sufficient.",
                "Physical parameter counts and audited block calls alone do not establish deployment speed, memory savings or energy gains."]))
        publish_identical(path,report)
    for row in tables["policy_summary"]:
        print("{:20s} AUC={:.4f} low-IoU={:.2%}".format(row["method"],row["auc"],row["low_iou_fraction"]))
    p = report["primary"]
    print("Adapted structure gap: {:+.4f} [{:+.4f},{:+.4f}]".format(p["mean_delta"],p["ci_low"],p["ci_high"]))
    print("Stage-44: " + report["status"])
    print("Reports: " + str(output))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=nonempty_path,required=True)
    args = parser.parse_args()
    summarize(Path(args.input).expanduser())


if __name__ == "__main__":
    main()
