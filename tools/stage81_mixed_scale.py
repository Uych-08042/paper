"""Evaluate raw and bounded SGLA scales within the same four search slots."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

import stage80_guard_experiment as s80
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, crop_geometry, tracking_iou


c, io, e63, e74, s76, s78 = s80.c, s80.io, s80.e63, s80.e74, s80.s76, s80.s78
SCHEMA = "stage81_mixed_scale_full_v1"
STATUS = "COMPLETE_MIXED_SCALE_DEVELOPMENT"
ARMS = ("native_sgla", "original_ppo", "mixed_ppo", "mixed_guard")
TABLES = s80.TABLES
DESIGN = dict(s80.DESIGN, bootstrap_seed=2181)


def mixed_parents(raw, limited, hw, ap):
    parents = s76.branches.prior.history_parents(raw, ap["design"])
    size = limited[-1][2:]
    mixed = np.zeros(4, bool)
    if np.array_equal(size, raw[-1][2:]):
        return parents, mixed

    def rect(parent):
        return crop_geometry(parent, hw, ap["search_factor"], ap["search_size"])[0]

    for anchor, slot in ((0, 1), (2, 3)):
        original = parents[anchor]
        bounded = np.r_[original[:2] + (original[2:] - size) / 2, size]
        box = rect(bounded)
        if not np.array_equal(box, rect(original)) and not np.array_equal(box, rect(parents[slot])):
            parents[slot] = bounded
            mixed[slot] = True
    return parents, mixed


def mixed_choice(unit, boxes, mask, logits, ppo, mixed, design):
    if unit["arm"] == "mixed_guard" and mixed.any():
        return s80.guard_action(boxes, mask, logits, ppo, design)
    return ppo, False


def collect_mixed(observer, entry, unit, ap, choose, design):
    initial = np.asarray(entry["initial_box"], float)
    raw, limited = [initial.copy()], [initial.copy()]
    observer.initialize(checked_image(ap["root"], entry["frames"][0]), initial)
    rows = []
    for i, image_entry in enumerate(entry["frames"][1:], 1):
        image = checked_image(ap["root"], image_entry)
        parents, mixed = mixed_parents(raw, limited, image.shape[:2], ap)
        observed = [e63.actor.observe(observer, image, parent, i, ap) for parent in parents]
        candidates = np.array([r["boxes"] for r in observed])
        row = choose(dict(candidate_boxes=candidates, candidate_parents=parents,
                          features=np.array([e63.actor.feature_vector(r, raw, ap) for r in observed])), raw)
        ppo, mask, logits = int(row["selected"]), row["action_mask"], row["logits"]
        selected, eligible = mixed_choice(unit, candidates, mask, logits, ppo, mixed, design)
        box = candidates[selected].copy()
        rows.append(dict(boxes=box, candidate_boxes=candidates, selected=selected, ppo_selected=ppo,
                         action_mask=mask, logits=logits, mixed=mixed, guard_eligible=eligible,
                         image_hw=np.array(image.shape[:2]), decode_error=max(r["decode_error"] for r in observed),
                         **{k: np.array([r[k] for r in observed]) for k in
                            ("crop", "layer", "block_calls", "head_calls", "selector_calls")}))
        s78.advance(raw, limited, box, ap["design"], design)
        observer.rt.tracker.state = box.tolist()
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def audit_mixed(data, entry, unit, ap, design):
    n = len(entry["frames"]) - 1
    s76.branches.prior.audit_calls(data, (n, 4))
    if (data["boxes"].shape != (n, 4) or data["candidate_boxes"].shape != (n, 4, 4)
            or not np.isfinite(data["decode_error"]).all()
            or data["decode_error"].max() > ap["design"]["box_atol"]):
        raise ValueError("Invalid full mixed-scale trajectory")
    raw, limited = [np.asarray(entry["initial_box"], float)], [np.asarray(entry["initial_box"], float)]
    for i, candidates in enumerate(data["candidate_boxes"]):
        mask, logits = c.action_mask(candidates), data["logits"][i]
        ppo = int(np.argmax(c.distribution(logits, mask)[0]))
        np.testing.assert_array_equal(mask, data["action_mask"][i])
        hw = data["image_hw"][i]
        parents, mixed = mixed_parents(raw, limited, hw, ap)
        rects = [crop_geometry(p, hw, ap["search_factor"], ap["search_size"])[0] for p in parents]
        np.testing.assert_array_equal(rects, data["crop"][i])
        np.testing.assert_array_equal(mixed, data["mixed"][i])
        selected, eligible = mixed_choice(unit, candidates, mask, logits, ppo, mixed, design)
        if (selected, ppo, eligible) != (data["selected"][i], data["ppo_selected"][i], data["guard_eligible"][i]):
            raise ValueError("Mixed-scale action changed")
        np.testing.assert_array_equal(candidates[selected], data["boxes"][i])
        s78.advance(raw, limited, candidates[selected], ap["design"], design)


def summary(seeds, design):
    video, macro, paired = [], [], []
    names = sorted({row["sequence"] for row in seeds})
    for seq in names:
        for arm in ARMS:
            rows = [r for r in seeds if r["sequence"] == seq and r["arm"] == arm]
            if len(rows) != (1 if arm == "native_sgla" else 2):
                raise ValueError("Incomplete full-track comparison")
            video.append(dict(sequence=seq, arm=arm, introduced_failure=any(r["introduced_failure"] for r in rows),
                              **s80.averages(rows)))
    for arm in ARMS:
        rows = [r for r in video if r["arm"] == arm]
        macro.append(dict(arm=arm, videos=len(rows), introduced_failure_videos=sum(r["introduced_failure"] for r in rows),
                          **s80.averages(rows)))
    lookup = {(r["sequence"], r["arm"]): r for r in video}
    for a, b in (("mixed_ppo", "original_ppo"), ("mixed_guard", "original_ppo"),
                 ("mixed_guard", "mixed_ppo"), ("mixed_ppo", "native_sgla"), ("mixed_guard", "native_sgla")):
        for metric in s80.METRICS:
            values = [(lookup[n, a][metric], lookup[n, b][metric]) for n in names]
            differences = [x-y for x, y in values if x is not None and y is not None]
            if differences:
                paired.append(s80.contrast(differences, f"{a}_minus_{b}", metric, design))
    return macro, paired, video


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

        mixed = arm in ARMS[2:]
        path = root / "evaluation" / (c.key(source_unit) + "__" + arm + ".npz")
        meta = dict(plan_fingerprint=plan["fingerprint"], unit=unit)
        data = c.read_cache(path, meta)
        new = data is None
        if new:
            with e63.source_experiment.installed(observer, [], source_unit) as head:
                if head != source["report"]["native_head_sha256"]:
                    raise ValueError("Native SGLA head changed")
                collector = collect_mixed if mixed else s80.collect
                data = collector(observer, entry, unit, ap, choose, plan["design"])
        (audit_mixed if mixed else s80.audit)(data, entry, unit, ap, plan["design"])
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
        slots = data["mixed"] if mixed else np.zeros((len(gt), 4), bool)
        candidates = data["candidate_boxes"] if arm != "native_sgla" else data["boxes"][:, None]
        good_support = np.any(tracking_iou(candidates, gt[:, None]) >= plan["design"]["good_iou"], axis=1)
        selected_good = tracking_iou(data["boxes"], gt) >= plan["design"]["good_iou"]
        scope.append(dict(unit, frames=len(gt), search_calls=int(data["head_calls"].sum()),
            block_calls=int(data["block_calls"].sum()), mixed_frames=int(slots.any(axis=1).sum()),
            capped_q1=int(slots[:, 1].sum()), capped_q3=int(slots[:, 3].sum()),
            candidate_good_frames=int(good_support.sum()), missed_good_frames=int((good_support & ~selected_good).sum()),
            selected_capped_frames=int(slots[np.arange(len(gt)), data["selected"]].sum()) if mixed else 0,
            guard_eligible_frames=int(np.sum(data.get("guard_eligible", 0))),
            guard_changes=int(np.count_nonzero(data.get("selected", 0) != data.get("ppo_selected", 0))),
            source_box_error=float(np.abs(data["boxes"] - ref["boxes"]).max()) if not mixed else None,
            source_action_changes=int(np.count_nonzero(data["selected"] != ref["selected"])) if arm == "original_ppo" else None))
        print(f"[Stage-81 full] {seq} {arm} seed{unit['seed']} {len(gt)} frames", flush=True)
    macro, paired, video = summary(seeds, plan["design"])
    budget = sum(r["search_calls"] for r in scope)
    if budget != plan["expected_search_calls"] or sum(r["block_calls"] for r in scope) != 7 * budget:
        raise ValueError("Full-video budget changed")
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])
    analysis = root / "analysis"
    analysis.mkdir(exist_ok=True)
    hashes = {}
    for name, rows in zip(TABLES, (macro, paired, video, seeds, scope)):
        path = analysis / f"stage81_{name}.csv"
        s80.publish_table(path, rows)
        hashes[path.name] = c.file_hash(path)
    _, original, mixed, guarded = macro
    checks = {}
    for result in (mixed, guarded):
        arm = result["arm"]
        auc = next(r for r in paired if r["contrast"] == arm + "_minus_original_ppo" and r["metric"] == "auc")
        checks[arm] = dict(practical_auc=auc["mean_delta"] >= plan["design"]["practical_auc"],
                           paired_auc_positive=auc["ci_low"] > 0,
                           low_iou_not_worse=result["low_iou_fraction"] <= original["low_iou_fraction"],
                           longest_failure_not_worse=result["longest_failure_fraction"] <= original["longest_failure_fraction"],
                           no_new_failure_on_native_stable=result["introduced_failure_videos"] == 0)
    report = io.seal(dict(schema=SCHEMA, status=STATUS, plan_fingerprint=plan["fingerprint"], runtime=plan["runtime"],
        source_stage80_fingerprint=plan["source_stage80_fingerprint"], confirmation_fingerprint=plan["confirmation_fingerprint"],
        trajectories=len(scope), search_calls=budget, block_calls=7*budget, training_updates=0, confirmation_inference_calls=0,
        max_source_box_error=max(r["source_box_error"] for r in scope if r["source_box_error"] is not None),
        source_action_changes=sum(r["source_action_changes"] or 0 for r in scope), development_checks=checks,
        primary_development_pass=all(checks["mixed_ppo"].values()),
        secondary_development_pass=all(checks["mixed_guard"].values()),
        evaluation_evidence_fingerprint=c.fingerprint(evidence), table_sha256=hashes))
    c.publish_identical(analysis / "stage81_decision.json", report)
    print(f"Stage-81: {STATUS} primary_development_pass={report['primary_development_pass']}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("stage76", "stage77", "stage80", "output_dir"):
        p.add_argument("--" + name, type=Path, required=name == "output_dir")
    p.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    p.add_argument("--parameter", default="deit_distilled")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--check-results", action="store_true")
    p.set_defaults(mode="stage81")
    args = p.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        plan = c.c66.sealed(root / "manifest.json", SCHEMA)
        report = c.c66.sealed(root / "analysis/stage81_decision.json", SCHEMA)
        if (report["plan_fingerprint"] != plan["fingerprint"] or report["status"] != STATUS
                or report["search_calls"] != plan["expected_search_calls"] or report["trajectories"] != len(plan["units"])
                or set(report["table_sha256"]) != {f"stage81_{n}.csv" for n in TABLES}):
            raise ValueError("Stage-81 result is incomplete")
        io.verify({str(root / "analysis" / name): digest for name, digest in report["table_sha256"].items()})
        return
    if not all((args.stage76, args.stage77, args.stage80, args.checkpoint)):
        p.error("Supply Stage-76/77/80 and the native checkpoint")
    paths = [path.expanduser().resolve() for path in (args.stage76, args.stage77, args.stage80)]
    if any(root == path or root in path.parents or path in root.parents for path in paths):
        p.error("Use a separate Stage-81 directory")
    import torch

    torch.set_num_threads(1)
    with writer_lock(root):
        if not (root / "manifest.json").exists() and any(path.name != "collect.lock" for path in root.iterdir()):
            raise ValueError("Use an empty Stage-81 directory")
        source, actor, _ = s78.load_sources(*paths[:2])
        p80 = c.c66.sealed(paths[2] / "manifest.json", s80.SCHEMA)
        report_path = paths[2] / "analysis/stage80_decision.json"
        r80 = c.c66.sealed(report_path, s80.SCHEMA)
        if (r80["status"] != s80.STATUS or r80["plan_fingerprint"] != p80["fingerprint"]
                or r80["trajectories"] != 84 or r80["search_calls"] != 338125
                or r80["confirmation_inference_calls"] != 0 or r80["development_pass"]
                or r80["confirmation_fingerprint"] != p80["confirmation_fingerprint"]):
            raise ValueError("Need the completed Stage-80 development result")
        if any(p80["source_sha256"].get(k) != v for k, v in source["evidence"].items()):
            raise ValueError("Stage-80 source lineage changed")
        bindings = dict(source["evidence"])
        bindings.update({str(paths[2] / "analysis" / n): h for n, h in r80["table_sha256"].items()})
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
        plan = io.seal(dict(schema=SCHEMA, role="consumed_development_full_mixed_scale", units=units, runtime=runtime,
            source_stage80_fingerprint=r80["fingerprint"], confirmation_fingerprint=r80["confirmation_fingerprint"],
            design=DESIGN, expected_search_calls=budget, source_sha256=bindings,
            code_sha256={str(Path(module).resolve()): c.file_hash(module)
                         for module in (__file__, s80.__file__, s78.__file__, e74.__file__)}))
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
