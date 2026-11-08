"""Test conflict-triggered retention of a pre-conflict SGLA search center."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

import stage82_same_history as s82
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, crop_geometry, tracking_iou


s80, s78, s76, c, io, e63, e74 = s82.s80, s82.s78, s82.s76, s82.c, s82.io, s82.e63, s82.e74
SCHEMA = "stage83_conflict_center_memory_v1"
STATUS = "COMPLETE_CONFLICT_CENTER_MEMORY_DEVELOPMENT"
ARMS = ("native_sgla", "original_ppo", "center_memory_ppo")
TABLES = s80.TABLES + ("memory_frames",)
DESIGN = dict(s78.DESIGN, bootstrap_seed=2183, bootstrap_samples=s80.DESIGN["bootstrap_samples"],
              near_tie_log_odds=s80.DESIGN["near_tie_log_odds"], practical_auc=s80.DESIGN["practical_auc"], anchor_frames=2,
              conflict_box_iou_max=.20, conflict_center_distance_min=.50)


def center(box):
    return box[:2] + box[2:] / 2


def conflict(boxes, mask, logits, selected, design):
    alternatives = np.flatnonzero(mask & (np.arange(len(mask)) != selected))
    if not len(alternatives):
        return False
    second = int(alternatives[np.argmax(logits[alternatives])])
    if logits[selected] - logits[second] > design["near_tie_log_odds"]:
        return False
    a, b = boxes[[selected, second]]
    area = a[2:].prod(), b[2:].prod()
    overlap = np.maximum(0., np.minimum(a[:2]+a[2:], b[:2]+b[2:]) - np.maximum(a[:2], b[:2])).prod()
    distance = np.linalg.norm(center(a) - center(b)) / np.mean(np.sqrt(area))
    return bool(overlap / (sum(area)-overlap) <= design["conflict_box_iou_max"]
                and distance >= design["conflict_center_distance_min"])


def search_parents(history, anchor, remaining, hw, ap):
    parents = s76.branches.prior.history_parents(history, ap["design"])
    # At t+1 the ordinary lag already points to the pre-conflict center.
    if remaining != 1 or np.array_equal(anchor, center(history[-2])):
        return parents, False
    shifted = parents[2:].copy()
    shifted[:, :2] = anchor - shifted[:, 2:] / 2
    for old, new in zip(parents[2:], shifted):
        old_rect = crop_geometry(old, hw, ap["search_factor"], ap["search_size"])[0]
        new_rect = crop_geometry(new, hw, ap["search_factor"], ap["search_size"])[0]
        if not np.array_equal(old_rect, new_rect):
            parents[2:] = shifted
            return parents, True
    return parents, False


def advance(history, box, anchor, remaining, trigger, ap, design):
    if remaining:
        remaining -= 1
    elif trigger:
        anchor, remaining = center(history[-1]), design["anchor_frames"]
    history.append(box.copy())
    del history[:-ap["design"]["history"]]
    return anchor, remaining


def collect(observer, entry, ap, choose, design):
    initial = np.asarray(entry["initial_box"], float)
    history, anchor, remaining = [initial], None, 0
    observer.initialize(checked_image(ap["root"], entry["frames"][0]), initial)
    rows = []
    for i, image_entry in enumerate(entry["frames"][1:], 1):
        image = checked_image(ap["root"], image_entry)
        parents, changed = search_parents(history, anchor, remaining, image.shape[:2], ap)
        observed = [e63.actor.observe(observer, image, parent, i, ap) for parent in parents]
        boxes = np.asarray([r["boxes"] for r in observed])
        decision = choose(dict(candidate_boxes=boxes, candidate_parents=parents,
                               features=np.asarray([e63.actor.feature_vector(r, history, ap) for r in observed])), history)
        selected, mask, logits = int(decision["selected"]), decision["action_mask"], decision["logits"]
        trigger = remaining == 0 and conflict(boxes, mask, logits, selected, design)
        box = boxes[selected]
        rows.append(dict(boxes=box, selected=selected, candidate_boxes=boxes, action_mask=mask, logits=logits,
                         trigger=trigger, remaining=remaining, changed=changed,
                         image_hw=np.array(image.shape[:2]), decode_error=max(r["decode_error"] for r in observed),
                         **{k: np.asarray([r[k] for r in observed])
                            for k in ("crop", "layer", "block_calls", "head_calls", "selector_calls")}))
        anchor, remaining = advance(history, box, anchor, remaining, trigger, ap, design)
        observer.rt.tracker.state = box.tolist()
    return {k: np.asarray([r[k] for r in rows]) for k in rows[0]}


def audit(data, entry, ap, design):
    n = len(entry["frames"]) - 1
    s76.branches.prior.audit_calls(data, (n, 4))
    if (data["boxes"].shape != (n, 4) or data["candidate_boxes"].shape != (n, 4, 4)
            or not np.isfinite(data["decode_error"]).all() or data["decode_error"].max() > ap["design"]["box_atol"]):
        raise ValueError("Invalid center-memory trajectory")
    history, anchor, remaining = [np.asarray(entry["initial_box"], float)], None, 0
    for i, boxes in enumerate(data["candidate_boxes"]):
        hw = data["image_hw"][i]
        parents, changed = search_parents(history, anchor, remaining, hw, ap)
        rects = [crop_geometry(p, hw, ap["search_factor"], ap["search_size"])[0] for p in parents]
        np.testing.assert_array_equal(rects, data["crop"][i])
        mask, logits = c.action_mask(boxes), data["logits"][i]
        np.testing.assert_array_equal(mask, data["action_mask"][i])
        selected = int(np.argmax(c.distribution(logits, mask)[0]))
        trigger = remaining == 0 and conflict(boxes, mask, logits, selected, design)
        if (selected, trigger, remaining, changed) != (data["selected"][i], data["trigger"][i], data["remaining"][i], data["changed"][i]):
            raise ValueError("Center-memory update or action changed")
        np.testing.assert_array_equal(boxes[selected], data["boxes"][i])
        anchor, remaining = advance(history, boxes[selected], anchor, remaining, trigger, ap, design)


def summary(seeds, design):
    video, macro, paired = [], [], []
    names = sorted({row["sequence"] for row in seeds})
    for seq in names:
        for arm in ARMS:
            rows = [r for r in seeds if r["sequence"] == seq and r["arm"] == arm]
            if len(rows) != (1 if arm == "native_sgla" else 2):
                raise ValueError("Incomplete matched seeds")
            video.append(dict(sequence=seq, arm=arm, introduced_failure=any(r["introduced_failure"] for r in rows),
                              **s80.averages(rows)))
    for arm in ARMS:
        rows = [r for r in video if r["arm"] == arm]
        macro.append(dict(arm=arm, videos=len(rows), introduced_failure_videos=sum(r["introduced_failure"] for r in rows),
                          **s80.averages(rows)))
    lookup = {(r["sequence"], r["arm"]): r for r in video}
    for comparator in ("original_ppo", "native_sgla"):
        for metric in s80.METRICS:
            values = [(lookup[n, "center_memory_ppo"][metric], lookup[n, comparator][metric]) for n in names]
            differences = [x-y for x, y in values if x is not None and y is not None]
            if differences:
                paired.append(s80.contrast(differences, "center_memory_ppo_minus_" + comparator, metric, design))
    return macro, paired, video


def run(root, plan, source, actor, observer):
    ap = e74.e67.actor_plan(source["original"])
    evidence, scope, seeds, native, memory = {}, [], [], {}, []
    for unit in plan["units"]:
        seq, arm = unit["sequence"], unit["arm"]
        old_unit = s82.source_unit(unit)
        entry, ref = s78.previous.read_source_trace(source, dict(unit=old_unit, frames=[1], expected_first_failure=None))
        action_unit, pack = s78.actor_pack(source, actor, old_unit)

        def choose(row, history):
            return e74.choose(row, history, action_unit, source["original"], actor["prior"], pack)

        path = root / "evaluation" / (c.key(old_unit) + "__" + arm + ".npz")
        meta = dict(plan_fingerprint=plan["fingerprint"], unit=unit)
        data = c.read_cache(path, meta)
        new = data is None
        modified = arm == "center_memory_ppo"
        if new:
            with e63.source_experiment.installed(observer, [], old_unit) as head:
                if head != source["report"]["native_head_sha256"]:
                    raise ValueError("Native SGLA head changed")
                data = collect(observer, entry, ap, choose, plan["design"]) if modified else s80.collect(
                    observer, entry, unit, ap, choose, plan["design"])
        if modified:
            audit(data, entry, ap, plan["design"])
        else:
            s80.audit(data, entry, unit, ap, plan["design"])
        if new:
            c.save(path, data, meta)
        s76.binding(evidence, path)
        if arm == "native_sgla":
            native[seq] = data["boxes"]
        gt = np.asarray(entry["ground_truth"])[1:]
        base = native[seq]
        metrics, _ = s80.describe(data, entry, base, tracking_iou(base, gt) < .2)
        metrics["native_failure_auc"] = metrics.pop("source_supervised_failure_auc")
        seeds.append(dict(unit, **metrics))
        only_reserved = 0
        if modified:
            ids = np.flatnonzero(data["trigger"] | (data["remaining"] > 0))
            for i in ids:
                iou = tracking_iou(data["candidate_boxes"][i], gt[i])
                good = plan["design"]["good_iou"]
                exclusive = bool(data["changed"][i] and iou[:2].max() < good <= iou[2:].max())
                only_reserved += int(exclusive)
                memory.append(dict(unit, frame=int(i+1), trigger=bool(data["trigger"][i]), remaining=int(data["remaining"][i]),
                    crop_changed=bool(data["changed"][i]), selected=int(data["selected"][i]),
                    selected_iou=float(iou[data["selected"][i]]), latest_best_iou=float(iou[:2].max()),
                    lag_or_reserved_best_iou=float(iou[2:].max()), only_reserved_pair_has_good=exclusive))
        scope.append(dict(unit, frames=len(gt), search_calls=int(data["head_calls"].sum()), block_calls=int(data["block_calls"].sum()),
            trigger_frames=int(np.sum(data.get("trigger", 0))), reserved_frames=int(np.count_nonzero(data.get("remaining", 0))),
            changed_crop_frames=int(np.sum(data.get("changed", 0))), only_reserved_pair_has_good_frames=only_reserved,
            source_box_error=float(np.abs(data["boxes"] - ref["boxes"]).max()) if not modified else None,
            source_action_changes=int(np.count_nonzero(data["selected"] != ref["selected"])) if arm == "original_ppo" else None))
        print(f"[Stage-83 full] {seq} {arm} seed{unit['seed']} {len(gt)} frames", flush=True)
    macro, paired, video = summary(seeds, plan["design"])
    budget = sum(r["search_calls"] for r in scope)
    if budget != plan["expected_search_calls"] or sum(r["block_calls"] for r in scope) != 7 * budget:
        raise ValueError("Full-track budget changed")
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])
    analysis = root / "analysis"
    analysis.mkdir(exist_ok=True)
    hashes = {}
    for name, rows in zip(TABLES, (macro, paired, video, seeds, scope, memory)):
        path = analysis / f"stage83_{name}.csv"
        s80.publish_table(path, rows)
        hashes[path.name] = c.file_hash(path)
    _, original, proposed = macro
    auc = paired[0]
    checks = dict(practical_auc=auc["mean_delta"] >= plan["design"]["practical_auc"], paired_auc_positive=auc["ci_low"] > 0,
        low_iou_not_worse=proposed["low_iou_fraction"] <= original["low_iou_fraction"],
        longest_failure_not_worse=proposed["longest_failure_fraction"] <= original["longest_failure_fraction"],
        no_new_failure_on_native_stable=proposed["introduced_failure_videos"] == 0)
    report = io.seal(dict(schema=SCHEMA, status=STATUS, plan_fingerprint=plan["fingerprint"], runtime=plan["runtime"],
        source_stage82_fingerprint=plan["source_stage82_fingerprint"], confirmation_fingerprint=plan["confirmation_fingerprint"],
        trajectories=len(scope), search_calls=budget, block_calls=7*budget, training_updates=0, confirmation_inference_calls=0,
        triggers=sum(r["trigger_frames"] for r in scope), changed_crop_frames=sum(r["changed_crop_frames"] for r in scope),
        only_reserved_pair_has_good_frames=sum(r["only_reserved_pair_has_good_frames"] for r in scope),
        max_source_box_error=max(r["source_box_error"] for r in scope if r["source_box_error"] is not None),
        source_action_changes=sum(r["source_action_changes"] or 0 for r in scope),
        development_checks=checks, development_pass=all(checks.values()),
        evaluation_evidence_fingerprint=c.fingerprint(evidence), table_sha256=hashes))
    c.publish_identical(analysis / "stage83_decision.json", report)
    print(f"Stage-83: {STATUS} development_pass={report['development_pass']}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("stage76", "stage77", "stage82", "output_dir"):
        p.add_argument("--" + name, type=Path, required=name == "output_dir")
    p.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    p.add_argument("--parameter", default="deit_distilled")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--check-results", action="store_true")
    p.set_defaults(mode="stage83")
    args = p.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        plan = c.c66.sealed(root / "manifest.json", SCHEMA)
        report = c.c66.sealed(root / "analysis/stage83_decision.json", SCHEMA)
        if (report["plan_fingerprint"] != plan["fingerprint"] or report["status"] != STATUS
                or report["search_calls"] != plan["expected_search_calls"] or report["trajectories"] != len(plan["units"])
                or set(report["table_sha256"]) != {f"stage83_{n}.csv" for n in TABLES}):
            raise ValueError("Stage-83 result is incomplete")
        io.verify({str(root / "analysis" / name): digest for name, digest in report["table_sha256"].items()})
        return
    if not all((args.stage76, args.stage77, args.stage82, args.checkpoint)):
        p.error("Supply Stage-76/77/82 and the native checkpoint")
    paths = [path.expanduser().resolve() for path in (args.stage76, args.stage77, args.stage82)]
    if any(root == path or root in path.parents or path in root.parents for path in paths):
        p.error("Use a separate Stage-83 directory")
    import torch

    torch.set_num_threads(1)
    with writer_lock(root):
        if not (root / "manifest.json").exists() and any(path.name != "collect.lock" for path in root.iterdir()):
            raise ValueError("Use an empty Stage-83 directory")
        source, actor, _ = s78.load_sources(*paths[:2])
        p82 = c.c66.sealed(paths[2] / "manifest.json", s82.SCHEMA)
        report_path = paths[2] / "analysis/stage82_decision.json"
        r82 = c.c66.sealed(report_path, s82.SCHEMA)
        if (r82["status"] != s82.STATUS or r82["plan_fingerprint"] != p82["fingerprint"]
                or r82["scanned_states"] != 54100 or not r82["factual_exact_bridge"]
                or r82["confirmation_inference_calls"] != 0 or r82["selector_retraining_supported"]
                or r82["confirmation_fingerprint"] != p82["confirmation_fingerprint"]
                or any(p82["source_sha256"].get(k) != v for k, v in source["evidence"].items())):
            raise ValueError("Need the completed Stage-82 development result")
        bindings = dict(source["evidence"])
        bindings.update({str(paths[2] / "analysis" / n): h for n, h in r82["table_sha256"].items()})
        bindings.update({str(p): c.file_hash(p) for p in (paths[2] / "manifest.json", report_path)})
        io.verify(bindings)
        dev = source["plan"]["units"]
        units = [dict(u, arm="native_sgla") for u in dev if u["policy"] == "q0"]
        units.extend(dict(u, arm=arm) for u in dev if u["policy"] == "ppo" for arm in ARMS[1:])
        if len(units) != 60 or len({u["sequence"] for u in units}) != 12:
            raise ValueError("Need the fixed 12 development videos and two PPO seeds")
        frames = {e["sequence"]: len(e["frames"])-1 for e in source["original"]["entries"]}
        budget = sum(frames[u["sequence"]] * (1 if u["arm"] == "native_sgla" else 4) for u in units)
        if budget != 229925:
            raise ValueError("Fixed development budget changed")
        runtime = s76.device_runtime(source["original"], torch, args.gpu)
        plan = io.seal(dict(schema=SCHEMA, role="consumed_development_full_search_center_memory", units=units, runtime=runtime,
            source_stage82_fingerprint=r82["fingerprint"], confirmation_fingerprint=r82["confirmation_fingerprint"],
            design=DESIGN, expected_search_calls=budget, source_sha256=bindings,
            code_sha256={str(Path(module).resolve()): c.file_hash(module)
                         for module in (__file__, s82.__file__, s80.__file__, s78.__file__, e74.__file__,
                                        Path(__file__).with_name("run_stage83_center_memory.sh"))}))
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
