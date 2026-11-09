"""Test the sealed Stage-84 search layout with the frozen four-action PPO in closed loop."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

import stage84_motion_candidates as proposal
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, source_group, tracking_iou
from stage51_state_summary import describe


s76, c, io, e63 = proposal.s76, proposal.c, proposal.io, proposal.e63
e74 = s76.e
SCHEMA = "stage85_motion_candidate_closed_loop_v1"
STATUS = "COMPLETE_MOTION_CANDIDATE_CLOSED_LOOP_DEVELOPMENT"
ARMS = ("native_sgla", "fixed_ppo", "motion_ppo")
TABLES = ("policy_summary", "paired", "video_summary", "seed_video_summary", "scope")
DESIGN = dict(bootstrap_seed=2185, bootstrap_samples=20000, practical_auc=.5, good_iou=.5)
METRICS = ("auc", "low_iou_fraction", "longest_failure_fraction", "reliable_auc",
           "best_candidate_iou", "good_candidate_fraction", "selected_iou", "mean_selection_gap",
           "selected_good_fraction", "missed_good_fraction", "recovered_events", "unrecovered_events")
PAIRED = ("auc", "low_iou_fraction", "longest_failure_fraction", "reliable_auc",
          "best_candidate_iou", "good_candidate_fraction", "selected_iou", "selected_good_fraction")
MODEL_FIELDS = ("crop", "layer", "block_calls", "head_calls", "selector_calls")
ROSTER = Path(__file__).resolve().parents[1] / "docs/experiments/06_sgla_uav_tracking/results/stage79_confirmation_v1.json"


def stage84_binding(root, source):
    proposal.check_results(root)
    plan_path = root / "manifest.json"
    report_path = root / "analysis/stage84_decision.json"
    plan = c.c66.sealed(plan_path, proposal.SCHEMA)
    report = c.c66.sealed(report_path, proposal.SCHEMA)
    generator = Path(proposal.__file__).resolve()
    digest = c.file_hash(generator)
    passing = dict(best_iou_positive_ci=True, best_iou_video_wins=True, good_candidate_fraction_increased=True)
    if (not report["candidate_supply_pass"] or report["gates"] != passing
            or report["matched_states"] != 27050 or report["videos"] != 12 or report["histories"] != 24
            or report["fixed_source_max_box_error"] != 0. or report["fixed_source_layer_changes"]
            or report["confirmation_inference_calls"] or report["training_updates"] or report["new_full_trajectories"]
            or report["source_stage76_fingerprint"] != source["report"]["fingerprint"]
            or plan["source_stage76_fingerprint"] != source["report"]["fingerprint"]
            or plan["source_sha256"] != source["evidence"]
            or plan["design"] != proposal.DESIGN or plan["code_sha256"].get(str(generator)) != digest
            or plan["units"] != [u for u in source["plan"]["units"] if u["policy"] == "ppo"]):
        raise ValueError("Need the passing sealed Stage-84 result and its unchanged generator")
    bindings = {str(root / "analysis" / name): h for name, h in report["table_sha256"].items()}
    bindings.update({str(p): c.file_hash(p) for p in (plan_path, report_path)})
    return report, bindings


def parents_for(history, hw, arm, ap, generator_design):
    if arm == "native_sgla":
        return np.asarray(history[-1:]), False
    if arm == "fixed_ppo":
        return s76.branches.prior.history_parents(history, ap["design"]), False
    if arm != "motion_ppo":
        raise ValueError("Unknown Stage-85 arm")
    fixed, generated = proposal.layouts(history, hw, ap, generator_design)
    changed = any(not np.array_equal(proposal.crop(a, hw, ap), proposal.crop(b, hw, ap))
                  for a, b in zip(fixed, generated))
    return generated, changed


def collect(observer, entry, arm, ap, choose, generator_design):
    history = [np.asarray(entry["initial_box"], float)]
    observer.initialize(checked_image(ap["root"], entry["frames"][0]), history[0])
    rows = []
    for frame, image_entry in enumerate(entry["frames"][1:], 1):
        image = checked_image(ap["root"], image_entry)
        parents, changed = parents_for(history, image.shape[:2], arm, ap, generator_design)
        observed = [e63.actor.observe(observer, image, parent, frame, ap) for parent in parents]
        candidates = np.asarray([r["boxes"] for r in observed])
        row = dict(frame_id=frame, candidate_parents=parents, candidate_boxes=candidates, changed_crop=changed,
                   image_hw=np.asarray(image.shape[:2]), decode_error=max(r["decode_error"] for r in observed),
                   **{k: np.asarray([r[k] for r in observed]) for k in MODEL_FIELDS})
        if choose is None:
            row["selected"] = 0
        else:
            row["features"] = np.asarray([e63.actor.feature_vector(r, history, ap) for r in observed])
            row["selected"] = int(choose(row.copy(), history)["selected"])
        box = candidates[row["selected"]].copy()
        row["boxes"] = box
        rows.append(row)
        history.append(box)
        history = history[-ap["design"]["history"]:]
        observer.rt.tracker.state = box.tolist()
        if frame % 500 == 0:
            print(f"  frame {frame}/{len(entry['frames'])-1}", flush=True)
    return {k: np.asarray([row[k] for row in rows]) for k in rows[0]}


def audit(data, entry, arm, ap, choose, generator_design):
    n = len(entry["frames"])-1
    count = 1 if arm == "native_sgla" else 4
    s76.branches.prior.audit_calls(data, (n, count))
    np.testing.assert_array_equal(data["frame_id"], np.arange(1, n+1))
    candidates, selected = data["candidate_boxes"], data["selected"]
    if (candidates.shape != (n, count, 4) or selected.shape != (n,) or selected.dtype.kind not in "iu"
            or np.any((selected < 0) | (selected >= count)) or not np.isfinite(candidates).all()
            or np.any(candidates[..., 2:] <= 0) or not np.isfinite(data["decode_error"]).all()
            or data["decode_error"].max() > ap["design"]["box_atol"]):
        raise ValueError("Invalid Stage-85 trajectory")
    np.testing.assert_array_equal(data["boxes"], candidates[np.arange(n), selected])
    history = [np.asarray(entry["initial_box"], float)]
    for i, box in enumerate(data["boxes"]):
        frame = entry["frames"][i+1]
        hw = frame["height"], frame["width"]
        np.testing.assert_array_equal(data["image_hw"][i], hw)
        parents, changed = parents_for(history, hw, arm, ap, generator_design)
        np.testing.assert_array_equal(data["candidate_parents"][i], parents)
        np.testing.assert_array_equal(data["crop"][i], [proposal.crop(p, hw, ap) for p in parents])
        if changed != data["changed_crop"][i]:
            raise ValueError("Changed crop receipt differs")
        if choose is not None:
            row = {k: data[k][i] for k in ("candidate_parents", "candidate_boxes", "features")}
            if selected[i] != choose(row, history)["selected"]:
                raise ValueError("Frozen PPO action differs")
        history.append(box)
        history = history[-ap["design"]["history"]:]


def metrics(data, entry, native, design):
    gt = np.asarray(entry["ground_truth"], float)[1:]
    scores, _ = describe(data, entry, native, tracking_iou(native, gt) < .2)
    ious = tracking_iou(data["candidate_boxes"], gt[:, None, :])
    selected = ious[np.arange(len(gt)), data["selected"]]
    best = ious.max(axis=1)
    good = best >= design["good_iou"]
    selected_good = selected >= design["good_iou"]
    scores.update(best_candidate_iou=float(best.mean()), selected_iou=float(selected.mean()),
                  mean_selection_gap=float((best-selected).mean()), good_candidate_fraction=float(good.mean()),
                  selected_good_fraction=float(selected_good.mean()),
                  missed_good_fraction=float((good & ~selected_good).mean()))
    return scores


def mean_metrics(rows):
    return {k: float(np.mean(values)) if (values := [r[k] for r in rows if r[k] is not None]) else None
            for k in METRICS}


def summarize(seeds, design):
    videos, policies, paired = [], [], []
    names = sorted({r["sequence"] for r in seeds})
    for arm in ARMS:
        group = []
        for seq in names:
            rows = [r for r in seeds if r["sequence"] == seq and r["arm"] == arm]
            if sorted(r["seed"] for r in rows) != ([0] if arm == "native_sgla" else [2171, 2172]):
                raise ValueError("Incomplete video/seed/arm comparison")
            group.append(dict(sequence=seq, arm=arm, introduced_failure=any(r["introduced_failure"] for r in rows),
                              **mean_metrics(rows)))
        videos.extend(group)
        policies.append(dict(arm=arm, videos=len(group),
                             introduced_failure_videos=sum(r["introduced_failure"] for r in group),
                             **mean_metrics(group)))
    lookup = {(r["sequence"], r["arm"]): r for r in videos}
    for comparator in ("fixed_ppo", "native_sgla"):
        for key in PAIRED:
            values = [(lookup[n, "motion_ppo"][key], lookup[n, comparator][key]) for n in names]
            deltas = [a-b for a, b in values if a is not None and b is not None]
            if deltas:
                paired.append(s76.contrast(deltas, f"motion_ppo_minus_{comparator}", key, design))
    return policies, paired, videos


def run(root, plan, source, actor, observer):
    ap = e74.e67.actor_plan(source["original"])
    evidence, scope, seeds, native, fixed = {}, [], [], {}, {}
    for unit in plan["units"]:
        seq, arm = unit["sequence"], unit["arm"]
        old = {k: v for k, v in unit.items() if k != "arm"}
        entry, ref = proposal.source_io.read_source_trace(source, dict(unit=old, frames=[1], expected_first_failure=None))
        choose = None
        if arm != "native_sgla":
            pack = e74.learned_pack(actor["models"], old, actor["prior"])

            def choose(row, history):
                return e74.choose(row, history, old, source["original"], actor["prior"], pack)

        path = root / "evaluation" / (c.key(old) + "__" + arm + ".npz")
        meta = dict(plan_fingerprint=plan["fingerprint"], unit=unit)
        print(f"[Stage-85] {seq} {arm} seed{unit['seed']}", flush=True)
        data = c.read_cache(path, meta)
        fresh = data is None
        if fresh:
            with e63.source_experiment.installed(observer, [], old) as head:
                if head != source["report"]["native_head_sha256"]:
                    raise ValueError("Native SGLA head changed")
                data = collect(observer, entry, arm, ap, choose, plan["generator_design"])
        audit(data, entry, arm, ap, choose, plan["generator_design"])
        pair = (seq, unit["seed"])
        divergence = None
        if arm == "native_sgla":
            native[seq] = data["boxes"]
        elif arm == "fixed_ppo":
            fixed[pair] = data
        else:
            baseline = fixed.pop(pair)
            warmup = plan["generator_design"]["minimum_history"]-1
            for k in ("candidate_parents", "candidate_boxes", "selected", "boxes"):
                np.testing.assert_array_equal(data[k][:warmup], baseline[k][:warmup])
            changes = np.flatnonzero(np.any(data["boxes"] != baseline["boxes"], axis=1))
            divergence = int(changes[0]+1) if len(changes) else None
        if fresh:
            c.save(path, data, meta)
        s76.binding(evidence, path)
        scores = metrics(data, entry, native[seq], plan["design"])
        seeds.append(dict(unit, **scores))
        scope.append(dict(unit, frames=len(data["boxes"]), search_calls=int(data["head_calls"].sum()),
            block_calls=int(data["block_calls"].sum()), changed_crop_frames=int(data["changed_crop"].sum()),
            first_output_difference_frame=divergence,
            source_box_error=float(np.abs(data["boxes"]-ref["boxes"]).max()) if arm != "motion_ppo" else None,
            source_action_changes=int(np.count_nonzero(data["selected"] != ref["selected"])) if arm == "fixed_ppo" else None))

    policies, paired, videos = summarize(seeds, plan["design"])
    budget = sum(r["search_calls"] for r in scope)
    if budget != plan["expected_search_calls"] or sum(r["block_calls"] for r in scope) != 7*budget:
        raise ValueError("Full-track budget changed")
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, rows in zip(TABLES, (policies, paired, videos, seeds, scope)):
        table = analysis / f"stage85_{name}.csv"
        s76.publish_table(table, rows)
        hashes[table.name] = c.file_hash(table)
    _, control, motion = policies
    primary = paired[0]
    gates = dict(practical_auc=primary["mean_delta"] >= plan["design"]["practical_auc"],
                 paired_auc_positive=primary["ci_low"] > 0.,
                 low_iou_not_worse=motion["low_iou_fraction"] <= control["low_iou_fraction"],
                 longest_failure_not_worse=motion["longest_failure_fraction"] <= control["longest_failure_fraction"],
                 no_new_failure_on_native_stable=motion["introduced_failure_videos"] == 0)
    report = io.seal(dict(schema=SCHEMA, status=STATUS, plan_fingerprint=plan["fingerprint"],
        runtime=plan["runtime"], source_stage84_fingerprint=plan["source_stage84_fingerprint"],
        confirmation_fingerprint=plan["confirmation_fingerprint"], trajectories=len(scope),
        videos=motion["videos"], search_calls=budget, block_calls=7*budget, training_updates=0,
        confirmation_inference_calls=0, primary=primary, development_checks=gates, development_pass=all(gates.values()),
        changed_crop_frames=sum(r["changed_crop_frames"] for r in scope),
        max_source_box_error=max(r["source_box_error"] for r in scope if r["source_box_error"] is not None),
        source_action_changes=sum(r["source_action_changes"] or 0 for r in scope),
        evaluation_evidence_fingerprint=c.fingerprint(evidence), table_sha256=hashes))
    c.publish_identical(analysis / "stage85_decision.json", report)
    print(f"Stage-85: {STATUS} development_pass={report['development_pass']}", flush=True)


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    report = c.c66.sealed(root / "analysis/stage85_decision.json", SCHEMA)
    if (report["status"] != STATUS or report["plan_fingerprint"] != plan["fingerprint"]
            or report["trajectories"] != len(plan["units"]) or report["search_calls"] != plan["expected_search_calls"]
            or set(report["table_sha256"]) != {f"stage85_{n}.csv" for n in TABLES}):
        raise ValueError("Stage-85 result is incomplete")
    io.verify({str(root / "analysis" / name): digest for name, digest in report["table_sha256"].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage76", "stage84", "output_dir"):
        parser.add_argument("--"+name, type=Path, required=name == "output_dir")
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage85")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not all((args.stage76, args.stage84, args.checkpoint)):
        parser.error("Supply the Stage-76 and Stage-84 run roots and the original SGLA checkpoint")
    paths = [p.expanduser().resolve() for p in (args.stage76, args.stage84)]
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate Stage-85 directory")
    import torch

    torch.set_num_threads(1)
    with writer_lock(root):
        if not (root / "manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-85 directory")
        source = proposal.source_io.load_source(paths[0])
        passed, bindings = stage84_binding(paths[1], source)
        actor = io.load_source(Path(source["plan"]["source_input"]))
        if actor["plan"] != source["original"] or actor["models"]["fingerprint"] != source["report"]["source_training_fingerprint"]:
            raise ValueError("Stage-74 controller differs from the Stage-76 evaluation")
        cohort = source["plan"]["units"]
        units = [dict(u, arm="native_sgla") for u in cohort if u["policy"] == "q0"]
        units.extend(dict(u, arm=arm) for u in cohort if u["policy"] == "ppo" for arm in ARMS[1:])
        names = {u["sequence"] for u in units}
        roster = c.c66.sealed(ROSTER, "stage79_visdrone_confirmation_v1")
        held = {r["source_group_proxy"] for r in roster["sequences"]}
        if (len(names) != 12 or len(units) != 60 or len(held) != 28
                or {source_group(n) for n in names} & held):
            raise ValueError("Need the fixed development cohort disjoint from the confirmation roster")
        frames = {e["sequence"]: len(e["frames"])-1 for e in source["original"]["entries"]}
        budget = sum(frames[u["sequence"]] * (1 if u["arm"] == "native_sgla" else 4) for u in units)
        if budget != 229925:
            raise ValueError("Fixed development budget changed")
        bindings.update(source["evidence"])
        bindings.update(actor["bindings"])
        bindings[str(ROSTER)] = c.file_hash(ROSTER)
        runtime = s76.device_runtime(source["original"], torch, args.gpu)
        code = (__file__, proposal.__file__, e74.__file__, Path(__file__).with_name("run_stage85_motion_closed_loop.sh"),
                e63.actor.observe.__code__.co_filename, s76.startup.checked_runtime.__code__.co_filename)
        plan = io.seal(dict(schema=SCHEMA, role="consumed_development_own_history_motion_candidate_evaluation",
            units=units, expected_search_calls=budget, design=DESIGN, generator_design=proposal.DESIGN, runtime=runtime,
            source_stage84_fingerprint=passed["fingerprint"], confirmation_fingerprint=roster["fingerprint"],
            source_sha256=bindings, code_sha256={str(Path(p).resolve()): c.file_hash(p) for p in code}))
        c.publish_identical(root / "manifest.json", plan)
        frozen = dict(source["original"], fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = s76.startup.checked_runtime(args, frozen, root)
        observer = e63.Capture(native)
        try:
            run(root, plan, source, actor, observer)
        finally:
            observer.handle.remove()


if __name__ == "__main__":
    main()
