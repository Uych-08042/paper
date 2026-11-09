"""Compare fixed and motion-generated four-crop SGLA banks on identical histories."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

import stage77_module_diagnostic as source_io
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, crop_geometry, tracking_iou


s76, c, io, e63 = source_io.s76, source_io.c, source_io.io, source_io.e63
SCHEMA = "stage84_motion_candidate_supply_v1"
STATUS = "COMPLETE_MATCHED_FOUR_CANDIDATE_SUPPLY"
TABLES = ("scope", "video_summary", "paired", "frame_summary")
DESIGN = dict(minimum_history=4, velocity_limit_sides=1., jump_sides=.5,
              offset_min_sides=.25, offset_max_sides=1., mad_scale=1.4826,
              core_fraction=.25, normal_weights=[2., 1., 3., .5], jump_weights=[1., 3., 2., .5],
              good_iou=.5, bootstrap_seed=2184, bootstrap_samples=20000, minimum_wins=7)


def crop(parent, hw, ap):
    return crop_geometry(parent, hw, ap["search_factor"], ap["search_size"])[0]


def layouts(history, hw, ap, design):
    """Keep q0 and the four original sizes, and allocate the other centers by marginal coverage."""
    history = np.asarray(history, float)[-ap["design"]["history"]:]
    fixed = s76.branches.prior.history_parents(history, ap["design"])
    generated = fixed.copy()
    if len(history) < design["minimum_history"]:
        return np.stack((fixed, generated))

    centers = history[:, :2] + history[:, 2:] / 2
    unit = float(np.median(np.sqrt(history[:, 2:].prod(axis=1))))
    velocity = np.median(np.diff(centers, axis=0), axis=0)
    speed = np.linalg.norm(velocity)
    if speed > design["velocity_limit_sides"] * unit:
        velocity *= design["velocity_limit_sides"] * unit / speed
    projected = centers + np.arange(len(history), 0, -1)[:, None] * velocity
    forecast = np.median(projected, axis=0)
    latest_jump = np.linalg.norm(projected[-1] - np.median(projected[:-1], axis=0))
    jump = latest_jump > design["jump_sides"] * unit
    spread = design["mad_scale"] * np.median(np.abs(projected - forecast), axis=0)
    spread = np.clip(spread, design["offset_min_sides"] * unit, design["offset_max_sides"] * unit)
    dx, dy = np.diag(spread)
    points = np.vstack((centers[-1], centers[-2], forecast, forecast+dx, forecast-dx,
                        forecast+dy, forecast-dy))
    points = np.clip(points, 0., np.asarray(hw[::-1], float)-1.)
    weights = design["jump_weights"] if jump else design["normal_weights"]
    weights = np.array(weights[:3] + [weights[3]]*4)

    def coverage(rects):
        centers = rects[:, :2] + rects[:, 2:] / 2
        residual = (centers[:, None] - points[None, :]) / (rects[:, None, 2:] * design["core_fraction"])
        return np.exp(-.5 * np.sum(residual**2, axis=2))

    first = crop(fixed[0], hw, ap)
    used = {tuple(first)}
    covered = coverage(first[None, :])[0]
    for slot in range(1, 4):
        size = fixed[slot, 2:]
        options = np.column_stack((points-size/2, np.broadcast_to(size, points.shape)))
        # Keep the exact original float parent as a deterministic fallback.
        options = np.vstack((options, fixed[slot]))
        rects = np.asarray([crop(p, hw, ap) for p in options])
        novel = np.array([tuple(rect) not in used for rect in rects])
        support = coverage(rects)
        if novel.any():
            gain = np.maximum(support-covered, 0.) @ weights
            index = int(np.argmax(np.where(novel, gain, -np.inf)))
        else:
            index = len(options)-1
        generated[slot] = options[index]
        used.add(tuple(rects[index]))
        covered = np.maximum(covered, support[index])
    return np.stack((fixed, generated))


def prepare(entry, trace, ap, design):
    """Reconstruct each state from the source output prefix and bind fixed crop geometry."""
    frames = entry["frames"][1:]
    history = [np.asarray(entry["initial_box"], float)]
    parents, crops = [], []
    for frame, box in zip(frames, trace["boxes"], strict=True):
        hw = (frame["height"], frame["width"])
        pair = layouts(history, hw, ap, design)
        parents.append(pair)
        crops.append([[crop(p, hw, ap) for p in bank] for bank in pair])
        history.append(box)
        history = history[-ap["design"]["history"]:]
    parents, crops = np.asarray(parents), np.asarray(crops)
    np.testing.assert_array_equal(parents[:, 0], trace["candidate_parents"])
    np.testing.assert_array_equal(crops[:, 0], trace["crop"])
    np.testing.assert_array_equal(parents[:, 0, 0], parents[:, 1, 0])
    np.testing.assert_array_equal(parents[:, 0, :, 2:], parents[:, 1, :, 2:])
    return parents, crops


def collect(observer, entry, parents, ap):
    observer.initialize(checked_image(ap["root"], entry["frames"][0]), entry["initial_box"])
    boxes, layers, calls, hashes = [], [], [], []
    for frame, pair in enumerate(parents, 1):
        image = checked_image(ap["root"], entry["frames"][frame])
        observations = {}
        for parent in pair.reshape(-1, 4):
            key = tuple(parent)
            if key not in observations:
                observations[key] = e63.actor.observe(observer, image, parent, frame, ap)
        results = [observations[tuple(p)] for p in pair.reshape(-1, 4)]
        boxes.append([r["boxes"] for r in results])
        layers.append([r["layer"] for r in results])
        hashes.append([r["crop_sha256"] for r in results])
        calls.append(len(observations))
        if frame % 500 == 0:
            print(f"  frame {frame}/{len(parents)}", flush=True)
    n = len(parents)
    return dict(boxes=np.asarray(boxes).reshape(n, 2, 4, 4), layers=np.asarray(layers).reshape(n, 2, 4),
                crop_hash=np.asarray(hashes).reshape(n, 2, 4), calls=np.asarray(calls))


def measure(unit, entry, parents, crops, source, data, design):
    boxes = data["boxes"]
    n = len(parents)
    if (set(data) != {"boxes", "layers", "crop_hash", "calls"} or boxes.shape != (n, 2, 4, 4)
            or data["layers"].shape != (n, 2, 4) or data["crop_hash"].shape != (n, 2, 4)
            or not np.isfinite(boxes).all() or np.any(boxes[..., 2:] <= 0)
            or not np.isin(data["layers"], np.arange(7, 13)).all()):
        raise ValueError("Invalid candidate replay")
    np.testing.assert_array_equal(data["calls"], [len({tuple(p) for p in pair.reshape(-1, 4)}) for pair in parents])
    np.testing.assert_array_equal(data["crop_hash"][:, 0], source["crop_sha256"])
    np.testing.assert_array_equal(boxes[:, 0, 0], boxes[:, 1, 0])
    y = tracking_iou(boxes, np.asarray(entry["ground_truth"], float)[1:, None, None, :])
    best = y.max(axis=2)
    good = best >= design["good_iou"]
    gained = ~good[:, 0] & good[:, 1]
    lost = good[:, 0] & ~good[:, 1]
    changed = np.any(parents[:, 0] != parents[:, 1], axis=(1, 2))
    crop_changed = np.any(crops[:, 0] != crops[:, 1], axis=(1, 2))
    distinct = np.array([[len({tuple(r) for r in bank}) for bank in pair] for pair in crops])
    scope = dict(unit, frames=n, changed_states=int(changed.sum()), changed_crop_states=int(crop_changed.sum()),
                 logical_calls_per_arm=4*n, physical_search_calls=int(data["calls"].sum()),
                 fixed_source_max_box_error=float(np.abs(boxes[:, 0]-source["candidate_boxes"]).max()),
                 fixed_source_layer_changes=int(np.count_nonzero(data["layers"][:, 0] != source["layer"])),
                 gained_good_frames=int(gained.sum()), lost_good_frames=int(lost.sum()),
                 fixed_best_iou=float(best[:, 0].mean()), generated_best_iou=float(best[:, 1].mean()),
                 fixed_good_fraction=float(good[:, 0].mean()), generated_good_fraction=float(good[:, 1].mean()),
                 fixed_distinct_crops=float(distinct[:, 0].mean()), generated_distinct_crops=float(distinct[:, 1].mean()))
    rows = [dict(unit, frame=i+1, changed=bool(changed[i]), crop_changed=bool(crop_changed[i]),
                 fixed_ious="|".join(map(str, y[i, 0])), generated_ious="|".join(map(str, y[i, 1])),
                 fixed_best_iou=float(best[i, 0]), generated_best_iou=float(best[i, 1]),
                 gained_good=bool(gained[i]), lost_good=bool(lost[i]),
                 fixed_distinct_crops=int(distinct[i, 0]), generated_distinct_crops=int(distinct[i, 1]))
            for i in range(n)]
    return scope, rows


def summarize(scope, design):
    videos = []
    for sequence in sorted({r["sequence"] for r in scope}):
        seeds = [r for r in scope if r["sequence"] == sequence]
        if sorted(r["seed"] for r in seeds) != [2171, 2172]:
            raise ValueError("Need both fixed PPO seeds per development video")
        keys = [f"{arm}_{metric}" for arm in ("fixed", "generated")
                for metric in ("best_iou", "good_fraction", "distinct_crops")]
        row = dict(sequence=sequence, **{k: float(np.mean([r[k] for r in seeds])) for k in keys})
        for key in ("frames", "changed_states", "changed_crop_states", "gained_good_frames", "lost_good_frames"):
            row[key] = sum(r[key] for r in seeds)
        videos.append(row)
    paired = [s76.contrast([r["generated_"+m]-r["fixed_"+m] for r in videos],
                           "generated_minus_fixed", m, design) for m in ("best_iou", "good_fraction")]
    return videos, paired


def run(root, plan, source, observer):
    ap = s76.e.e67.actor_plan(source["original"])
    evidence, scope, frames = {}, [], []
    for unit in plan["units"]:
        entry, trace = source_io.read_source_trace(source, dict(unit=unit, frames=[1], expected_first_failure=None))
        parents, crops = prepare(entry, trace, ap, plan["design"])
        path = root / "collection" / (c.key(unit) + ".npz")
        meta = dict(plan_fingerprint=plan["fingerprint"], unit=unit)
        print(f"[Stage-84] {unit['sequence']} seed{unit['seed']} ({len(parents)} states)", flush=True)
        data = c.read_cache(path, meta)
        fresh = data is None
        if fresh:
            with e63.source_experiment.installed(observer, [], unit) as head:
                if head != source["report"]["native_head_sha256"]:
                    raise ValueError("Native SGLA head changed")
                data = collect(observer, entry, parents, ap)
        row, records = measure(unit, entry, parents, crops, trace, data, plan["design"])
        if fresh:
            c.save(path, data, meta)
        s76.binding(evidence, path)
        scope.append(row)
        frames.extend(records)
    videos, paired = summarize(scope, plan["design"])
    states = sum(r["frames"] for r in scope)
    logical = sum(r["logical_calls_per_arm"] for r in scope)
    physical = sum(r["physical_search_calls"] for r in scope)
    if states != plan["states"] or logical != 4*states or not states <= physical <= 7*states:
        raise ValueError("Matched search budget changed")
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, rows in zip(TABLES, (scope, videos, paired, frames)):
        table = out / f"stage84_{name}.csv"
        s76.publish_table(table, rows)
        hashes[table.name] = c.file_hash(table)
    best, good = paired
    gates = dict(best_iou_positive_ci=best["ci_low"] > 0.,
                 best_iou_video_wins=best["wins"] >= plan["design"]["minimum_wins"],
                 good_candidate_fraction_increased=good["mean_delta"] > 0.)
    report = io.seal(dict(schema=SCHEMA, status=STATUS, plan_fingerprint=plan["fingerprint"],
        runtime=plan["runtime"], source_stage76_fingerprint=plan["source_stage76_fingerprint"],
        videos=len(videos), histories=len(scope), matched_states=states, logical_search_calls_per_arm=logical,
        physical_search_calls=physical, block_calls=7*physical, training_updates=0, new_full_trajectories=0,
        confirmation_inference_calls=0, candidate_supply_pass=all(gates.values()), gates=gates,
        best_iou=best, good_candidate_fraction=good,
        gained_good_frames=sum(r["gained_good_frames"] for r in scope),
        lost_good_frames=sum(r["lost_good_frames"] for r in scope),
        changed_states=sum(r["changed_states"] for r in scope),
        changed_crop_states=sum(r["changed_crop_states"] for r in scope),
        fixed_source_max_box_error=max(r["fixed_source_max_box_error"] for r in scope),
        fixed_source_layer_changes=sum(r["fixed_source_layer_changes"] for r in scope),
        collection_evidence_fingerprint=c.fingerprint(evidence), table_sha256=hashes))
    c.publish_identical(out / "stage84_decision.json", report)
    print(f"Stage-84: {STATUS} candidate_supply_pass={report['candidate_supply_pass']}", flush=True)


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    report = c.c66.sealed(root / "analysis/stage84_decision.json", SCHEMA)
    if (report["plan_fingerprint"] != plan["fingerprint"] or report["status"] != STATUS
            or report["matched_states"] != plan["states"]
            or report["histories"] != len(plan["units"])
            or report["logical_search_calls_per_arm"] != 4*plan["states"]
            or set(report["table_sha256"]) != {f"stage84_{n}.csv" for n in TABLES}):
        raise ValueError("Stage-84 result is incomplete")
    io.verify({str(root / "analysis" / name): digest for name, digest in report["table_sha256"].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage76", type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage84")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not args.stage76 or not args.checkpoint:
        parser.error("Supply the Stage-76 run and the native checkpoint")
    stage76 = args.stage76.expanduser().resolve()
    if root == stage76 or root in stage76.parents or stage76 in root.parents:
        parser.error("Use a separate Stage-84 directory")
    import torch

    torch.set_num_threads(1)
    with writer_lock(root):
        if not (root / "manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-84 directory")
        source = source_io.load_source(stage76)
        units = [u for u in source["plan"]["units"] if u["policy"] == "ppo"]
        videos = {u["sequence"] for u in units}
        states = sum(len(c.entry_for(source["original"], u["sequence"])["frames"])-1 for u in units)
        if (len(videos) != 12 or len(units) != 24 or states != 27050
                or {(u["sequence"], u["seed"]) for u in units} != {(v,s) for v in videos for s in (2171, 2172)}):
            raise ValueError("Need the complete frozen Stage-76 PPO development cohort")
        runtime = s76.device_runtime(source["original"], torch, args.gpu)
        code = (Path(__file__), Path(__file__).with_name("run_stage84_motion_candidates.sh"),
                Path(source_io.__file__), Path(e63.actor.observe.__code__.co_filename),
                Path(s76.branches.prior.history_parents.__code__.co_filename),
                Path(crop_geometry.__code__.co_filename), Path(s76.startup.checked_runtime.__code__.co_filename))
        plan = io.seal(dict(schema=SCHEMA, role="consumed_development_same_history_candidate_supply",
            units=units, states=states, design=DESIGN, runtime=runtime, source_sha256=source["evidence"],
            source_stage76_fingerprint=source["report"]["fingerprint"],
            code_sha256={str(p.resolve()): c.file_hash(p) for p in code}))
        c.publish_identical(root / "manifest.json", plan)
        frozen = dict(source["original"], fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = s76.startup.checked_runtime(args, frozen, root)
        observer = e63.Capture(native)
        try:
            run(root, plan, source, observer)
        finally:
            observer.handle.remove()


if __name__ == "__main__":
    main()
