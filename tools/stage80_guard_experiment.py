"""Full development evaluation of SGLA scale feedback with a consensus guard."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

import stage79_scale_cross as cross
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, crop_geometry, tracking_iou
from stage51_state_summary import describe
from stage61_l40_summary import contrast
from stage68_advantage_summary import publish_table


s78, c, io, s76, e63, e74 = cross.s78, cross.c, cross.io, cross.s76, cross.e63, cross.e74
SCHEMA = "stage80_scale_consensus_full_v1"
STATUS = "COMPLETE_SCALE_CONSENSUS_DEVELOPMENT"
ARMS = ("native_sgla", "original_ppo", "limited_ppo", "guarded_ppo")
TABLES = ("policy_summary", "paired", "video_summary", "seed_video_summary", "scope")
DESIGN = dict(s78.DESIGN, near_tie_log_odds=float(np.log(1.10)), consensus_gain=.20,
              bootstrap_seed=2180, bootstrap_samples=20000, practical_auc=.5)
METRICS = ("auc", "low_iou_fraction", "reliable_auc", "longest_failure_fraction")


def guard_action(boxes, mask, logits, selected, design):
    """Pick among near-tied distinct boxes using agreement with other current outputs."""
    peers = np.flatnonzero(mask)
    nearby = peers[logits[selected] - logits[peers] <= design["near_tie_log_odds"]]
    if len(peers) < 3 or len(nearby) < 2:
        return selected, False
    xy = boxes[:, :2]
    corner = xy + boxes[:, 2:]
    intersect = np.maximum(0, np.minimum(corner[:, None], corner) - np.maximum(xy[:, None], xy)).prod(axis=2)
    area = boxes[:, 2:].prod(axis=1)
    similarity = intersect / (area[:, None] + area - intersect)
    scores = (similarity[:, peers].sum(axis=1) - 1.) / (len(peers) - 1)
    preferred = int(nearby[np.argmax(scores[nearby])])
    return (preferred if scores[preferred] - scores[selected] >= design["consensus_gain"] else selected), True


def search_rects(history, hw, ap):
    parents = s76.branches.prior.history_parents(history, ap["design"])
    return np.array([crop_geometry(p, hw, ap["search_factor"], ap["search_size"])[0] for p in parents])


def collect(observer, entry, unit, ap, choose, design):
    initial = np.asarray(entry["initial_box"], float)
    raw, limited = [initial.copy()], [initial.copy()]
    arm = unit["arm"]
    stabilized = arm in ("limited_ppo", "guarded_ppo")
    observer.initialize(checked_image(ap["root"], entry["frames"][0]), initial)
    rows = []
    for i, image_entry in enumerate(entry["frames"][1:], 1):
        image = checked_image(ap["root"], image_entry)
        if arm == "native_sgla":
            data = e63.actor.observe(observer, image, raw[-1], i, ap)
            box = np.asarray(data["boxes"], float)
            row = dict(boxes=box, layer=[data["layer"]], block_calls=[data["block_calls"]],
                       head_calls=[data["head_calls"]], selector_calls=[data["selector_calls"]], decode_error=data["decode_error"])
        else:
            data = cross.observe(observer, image, i, raw, limited, "search_scale" if stabilized else "native", ap, choose)
            base = int(data["selected"])
            active = stabilized and not np.array_equal(data["crop"], search_rects(raw, image.shape[:2], ap))
            selected, eligible = base, False
            if arm == "guarded_ppo" and active:
                selected, eligible = guard_action(data["candidate_boxes"], data["action_mask"], data["logits"], base, design)
            box = data["candidate_boxes"][selected].copy()
            row = dict(boxes=box, selected=selected, ppo_selected=base, scale_active=active, guard_eligible=eligible,
                       **{k: data[k] for k in ("candidate_boxes", "action_mask", "logits", "crop", "image_hw",
                                               "layer", "block_calls", "head_calls", "selector_calls", "decode_error")})
        rows.append(row)
        s78.advance(raw, limited, box, ap["design"], design)
        observer.rt.tracker.state = box.tolist()
    return {k: np.asarray([r[k] for r in rows]) for k in rows[0]}


def audit(data, entry, unit, ap, design):
    n = len(entry["frames"]) - 1
    arm = unit["arm"]
    s76.branches.prior.audit_calls(data, (n, 1 if arm == "native_sgla" else 4))
    boxes = data["boxes"]
    if (boxes.shape != (n, 4) or not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= 0)
            or not np.isfinite(data["decode_error"]).all() or data["decode_error"].max() > ap["design"]["box_atol"]):
        raise ValueError("Invalid full trajectory")
    if arm == "native_sgla":
        return
    raw, limited = [np.asarray(entry["initial_box"], float)], [np.asarray(entry["initial_box"], float)]
    stabilized = arm in ("limited_ppo", "guarded_ppo")
    for i in range(n):
        candidates = data["candidate_boxes"][i]
        mask = data["action_mask"][i]
        logits = data["logits"][i]
        np.testing.assert_array_equal(mask, c.action_mask(candidates))
        base = int(np.argmax(c.distribution(logits, mask)[0]))
        np.testing.assert_array_equal(base, data["ppo_selected"][i])
        hw = data["image_hw"][i]
        actual = search_rects(limited if stabilized else raw, hw, ap)
        np.testing.assert_array_equal(actual, data["crop"][i])
        active = stabilized and not np.array_equal(actual, search_rects(raw, hw, ap))
        selected, eligible = (guard_action(candidates, mask, logits, base, design)
                              if active and arm == "guarded_ppo" else (base, False))
        if (selected != data["selected"][i] or eligible != data["guard_eligible"][i] or active != data["scale_active"][i]):
            raise ValueError("Frozen guard decision or scale changed")
        np.testing.assert_array_equal(boxes[i], candidates[selected])
        s78.advance(raw, limited, boxes[i], ap["design"], design)


def summarize(seeds, design):
    names = sorted({row["sequence"] for row in seeds})
    video, macro, paired = [], [], []
    for seq in names:
        for arm in ARMS:
            group = [r for r in seeds if r["sequence"] == seq and r["arm"] == arm]
            if len(group) != (1 if arm == "native_sgla" else 2):
                raise ValueError("Missing matched evaluation arms")
            video.append(dict(sequence=seq, arm=arm, introduced_failure=any(r["introduced_failure"] for r in group),
                              **averages(group)))
    for arm in ARMS:
        rows = [r for r in video if r["arm"] == arm]
        macro.append(dict(arm=arm, videos=len(rows), introduced_failure_videos=sum(r["introduced_failure"] for r in rows),
                          **averages(rows)))
    lookup = {(r["sequence"], r["arm"]): r for r in video}
    for a, b in (("guarded_ppo", "original_ppo"), ("guarded_ppo", "limited_ppo"),
                 ("limited_ppo", "original_ppo"), ("guarded_ppo", "native_sgla")):
        for metric in METRICS:
            values = [(lookup[n, a][metric], lookup[n, b][metric]) for n in names]
            deltas = [x - y for x, y in values if x is not None and y is not None]
            if deltas:
                paired.append(contrast(deltas, f"{a}_minus_{b}", metric, design))
    return video, macro, paired


def averages(rows):
    return {k: float(np.mean(values)) if (values := [r[k] for r in rows if r[k] is not None]) else None for k in METRICS}


def run(root, plan, source, actor, observer):
    ap = e74.e67.actor_plan(source["original"])
    evidence, scope, seeds, native = {}, [], [], {}
    for unit in plan["units"]:
        seq, arm = unit["sequence"], unit["arm"]
        source_unit = {k: v for k, v in unit.items() if k != "arm"}
        entry, ref = s78.previous.read_source_trace(source, dict(unit=source_unit, frames=[1], expected_first_failure=None))
        action_unit, pack = s78.actor_pack(source, actor, source_unit)

        def choose(row, history):
            return e74.choose(row, history, action_unit, source["original"], actor["prior"], pack)

        path = root / "evaluation" / (c.key(source_unit) + "__" + arm + ".npz")
        meta = dict(plan_fingerprint=plan["fingerprint"], unit=unit)
        data = c.read_cache(path, meta)
        new = data is None
        if new:
            with e63.source_experiment.installed(observer, [], source_unit) as head:
                if head != source["report"]["native_head_sha256"]:
                    raise ValueError("Native SGLA head changed")
                data = collect(observer, entry, unit, ap, choose, plan["design"])
        audit(data, entry, unit, ap, plan["design"])
        if new:
            c.save(path, data, meta)
        s76.binding(evidence, path)
        if arm == "native_sgla":
            native[seq] = data["boxes"]
        base = native[seq]
        gt = np.asarray(entry["ground_truth"])[1:]
        metrics, _ = describe(data, entry, base, tracking_iou(base, gt) < .2)
        metrics["native_failure_auc"] = metrics.pop("source_supervised_failure_auc")
        seeds.append(dict(unit, **metrics))
        bridge = arm in ("native_sgla", "original_ppo")
        scope.append(dict(unit, frames=len(gt), search_calls=int(data["head_calls"].sum()),
                          block_calls=int(data["block_calls"].sum()), scale_active_frames=int(np.sum(data.get("scale_active", 0))),
                          guard_eligible_frames=int(np.sum(data.get("guard_eligible", 0))),
                          guard_changes=int(np.count_nonzero(data.get("selected", 0) != data.get("ppo_selected", 0))),
                          source_box_error=float(np.abs(data["boxes"] - ref["boxes"]).max()) if bridge else None,
                          source_action_changes=int(np.count_nonzero(data["selected"] != ref["selected"])) if arm == "original_ppo" else None))
        print(f"[Stage-80 full] {seq} {arm} seed{unit['seed']} {len(gt)} frames", flush=True)
    video, macro, paired = summarize(seeds, plan["design"])
    budget = sum(r["search_calls"] for r in scope)
    if budget != plan["expected_search_calls"] or sum(r["block_calls"] for r in scope) != 7 * budget:
        raise ValueError("Full-video budget changed")
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])
    analysis = root / "analysis"
    analysis.mkdir(exist_ok=True)
    hashes = {}
    for name, rows in zip(TABLES, (macro, paired, video, seeds, scope)):
        path = analysis / f"stage80_{name}.csv"
        publish_table(path, rows)
        hashes[path.name] = c.file_hash(path)
    _, original, limited, guard = macro
    primary = paired[0]
    checks = dict(practical_auc=primary["mean_delta"] >= plan["design"]["practical_auc"],
        paired_auc_positive=primary["ci_low"] > 0, guard_not_worse_than_limit=guard["auc"] >= limited["auc"],
        low_iou_not_worse=guard["low_iou_fraction"] <= original["low_iou_fraction"],
        longest_failure_not_worse=guard["longest_failure_fraction"] <= original["longest_failure_fraction"],
        no_new_failure_on_native_stable=guard["introduced_failure_videos"] == 0)
    report = io.seal(dict(schema=SCHEMA, status=STATUS, plan_fingerprint=plan["fingerprint"], runtime=plan["runtime"],
        source_stage79_fingerprint=plan["source_stage79_fingerprint"], confirmation_fingerprint=plan["confirmation_fingerprint"],
        trajectories=len(scope), search_calls=budget, block_calls=7*budget, training_updates=0, confirmation_inference_calls=0,
        guard_changes=sum(r["guard_changes"] for r in scope if r["arm"] == "guarded_ppo"),
        max_source_box_error=max(r["source_box_error"] for r in scope if r["source_box_error"] is not None),
        source_action_changes=sum(r["source_action_changes"] or 0 for r in scope),
        development_checks=checks, development_pass=all(checks.values()),
        evaluation_evidence_fingerprint=c.fingerprint(evidence), table_sha256=hashes))
    c.publish_identical(analysis / "stage80_decision.json", report)
    print(f"Stage-80: {STATUS} development_pass={report['development_pass']}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("stage76", "stage77", "stage79", "output_dir"):
        p.add_argument("--" + name, type=Path, required=name == "output_dir")
    p.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    p.add_argument("--parameter", default="deit_distilled")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--check-results", action="store_true")
    p.set_defaults(mode="stage80")
    args = p.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        plan = c.c66.sealed(root / "manifest.json", SCHEMA)
        report = c.c66.sealed(root / "analysis/stage80_decision.json", SCHEMA)
        if (report["plan_fingerprint"] != plan["fingerprint"] or report["status"] != STATUS
                or report["search_calls"] != plan["expected_search_calls"] or report["trajectories"] != len(plan["units"])
                or set(report["table_sha256"]) != {f"stage80_{n}.csv" for n in TABLES}):
            raise ValueError("Stage-80 result is incomplete")
        io.verify({str(root / "analysis" / name): digest for name, digest in report["table_sha256"].items()})
        return
    if not all((args.stage76, args.stage77, args.stage79, args.checkpoint)):
        p.error("Supply Stage-76/77/79 and the native checkpoint")
    paths = [path.expanduser().resolve() for path in (args.stage76, args.stage77, args.stage79)]
    if any(root == path or root in path.parents or path in root.parents for path in paths):
        p.error("Use a separate Stage-80 directory")
    import torch

    torch.set_num_threads(1)
    with writer_lock(root):
        if not (root / "manifest.json").exists() and any(path.name != "collect.lock" for path in root.iterdir()):
            raise ValueError("Use an empty Stage-80 directory")
        source, actor, _ = s78.load_sources(*paths[:2])
        p79 = c.c66.sealed(paths[2] / "manifest.json", cross.SCHEMA)
        report_path = paths[2] / "analysis/stage79_decision.json"
        r79 = c.c66.sealed(report_path, cross.SCHEMA)
        if (r79["status"] != cross.STATUS or r79["plan_fingerprint"] != p79["fingerprint"]
                or r79["natural_exact_branches"] != 4 or r79["physical_search_calls"] != 672
                or r79["confirmation_inference_calls"] != 0):
            raise ValueError("Need exact Stage-79 development diagnosis")
        if any(p79["source_sha256"].get(k) != v for k, v in source["evidence"].items()):
            raise ValueError("Stage-79 source lineage changed")
        bindings = dict(source["evidence"])
        bindings.update({str(paths[2] / "analysis" / n): h for n, h in r79["table_sha256"].items()})
        bindings.update({str(p): c.file_hash(p) for p in (paths[2] / "manifest.json", report_path)})
        io.verify(bindings)
        dev = source["plan"]["units"]
        units = [dict(u, arm="native_sgla") for u in dev if u["policy"] == "q0"]
        units.extend(dict(u, arm=arm) for u in dev if u["policy"] == "ppo" for arm in ARMS[1:])
        if len(units) != 84 or len({u["sequence"] for u in units}) != 12:
            raise ValueError("Need the fixed 12 development videos and two PPO seeds")
        frames = {e["sequence"]: len(e["frames"]) - 1 for e in source["original"]["entries"]}
        budget = sum(frames[u["sequence"]] * (1 if u["arm"] == "native_sgla" else 4) for u in units)
        if budget != 338125:
            raise ValueError("Fixed development budget changed")
        runtime = s76.device_runtime(source["original"], torch, args.gpu)
        plan = io.seal(dict(schema=SCHEMA, role="consumed_development_full_scale_guard", units=units, runtime=runtime,
            source_stage79_fingerprint=r79["fingerprint"], confirmation_fingerprint=r79["confirmation_fingerprint"],
            design=DESIGN, expected_search_calls=budget, source_sha256=bindings,
            code_sha256={str(Path(module).resolve()): c.file_hash(module)
                         for module in (__file__, cross.__file__, s78.__file__, e74.__file__)}))
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
