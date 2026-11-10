"""Isolate a one-frame search-center write after an identical published box."""
from __future__ import annotations

import argparse
from collections import defaultdict
import os
from pathlib import Path

import numpy as np

import stage77_module_diagnostic as previous
import stage79_confirmation as confirmation
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, crop_geometry, failure_events, tracking_iou
from stage61_l40_summary import contrast
from stage68_advantage_summary import publish_table


c, e63, io, s76 = previous.c, previous.e63, previous.io, previous.s76
e74 = s76.e
SCHEMA = "stage86_same_output_search_center_v1"
STATUS = "COMPLETE_SAME_OUTPUT_SEARCH_CENTER_DIAGNOSTIC"
DESIGN = dict(horizon=16, bins=3, lag_mix=.5, good_iou=.5, low_iou=.2, recovery_run=3,
              bootstrap_seed=2186, bootstrap_samples=10000)
ARMS = ("fixed", "midpoint")
STRATA = ("high", "low")
TABLES = ("windows", "video_summary", "paired", "scope")
FIELDS = ("frame_id", "history_latest", "candidate_parents", "candidate_boxes", "boxes", "selected", "features",
          "image_hw", "decode_error", "crop", "crop_sha256", "layer", "block_calls", "head_calls", "selector_calls")
ROSTER = Path(__file__).resolve().parents[1] / "docs/experiments/06_sgla_uav_tracking/results/stage79_confirmation_v1.json"


def centers(boxes):
    return boxes[..., :2] + boxes[..., 2:] / 2


def choose_windows(boxes, unit, design):
    """Only frozen predicted boxes determine eligibility and rank."""
    horizon = design["horizon"]
    moves = np.linalg.norm(np.diff(centers(boxes), axis=0), axis=1) / np.sqrt(boxes[:-1, 2:].prod(axis=1))
    windows = []
    for number, frames in enumerate(np.array_split(np.arange(1, len(boxes)), design["bins"]), 1):
        eligible = frames[frames <= frames[-1] - horizon]
        if not len(eligible):
            raise ValueError("Video too short for the fixed paired windows")
        high = int(eligible[np.argmax(moves[eligible - 1])])
        separate = eligible[np.abs(eligible - high) >= horizon]
        if not len(separate):
            raise ValueError("Need a disjoint low-displacement comparison window")
        low = int(separate[np.argmin(moves[separate - 1])])
        for stratum, anchor in zip(STRATA, (high, low)):
            windows.append(dict(case=f"{c.key(unit)}__b{number}_{stratum}", unit=unit, time_bin=number,
                                stratum=stratum, anchor=anchor, predicted_displacement=float(moves[anchor - 1])))
    return windows


def parents_at(history, design, mix):
    parents = s76.branches.prior.history_parents(history, design)
    if mix:
        parents[:2, :2] += mix * (centers(history[-2]) - centers(history[-1]))
    return parents


def collect(observer, entry, prefix, frames, mix, ap, choose):
    history = list(prefix.copy())
    root = ap["root"]
    observer.initialize(checked_image(root, entry["frames"][0]), entry["initial_box"])
    observer.rt.tracker.state = history[-1].tolist()
    rows = []
    for index, frame in enumerate(frames):
        image = checked_image(root, entry["frames"][frame])
        if index == 0 and mix:
            parents = parents_at(history, ap["design"], mix)
            views = [e63.actor.observe(observer, image, parent, frame, ap) for parent in parents]
            row = dict(frame_id=frame, history_latest=history[-1].copy(), candidate_parents=parents,
                       candidate_boxes=np.asarray([v["boxes"] for v in views]),
                       features=np.asarray([e63.actor.feature_vector(v, history, ap) for v in views]),
                       image_hw=np.asarray(image.shape[:2]), decode_error=max(v["decode_error"] for v in views),
                       **{k: np.asarray([v[k] for v in views]) for k in
                          ("crop", "crop_sha256", "layer", "block_calls", "head_calls", "selector_calls")})
        else:
            row = e63.probe(observer, image, history, frame, "history", ap)
        row = choose(row, history)
        rows.append(row)
        history = (history + [row["boxes"].copy()])[-ap["design"]["history"]:]
        observer.rt.tracker.state = row["boxes"].tolist()
    return {k: np.asarray([r[k] for r in rows]) for k in FIELDS}


def audit(data, prefix, frames, mix, ap, choose):
    if set(data) != set(FIELDS) or not np.array_equal(data["frame_id"], frames):
        raise ValueError("Incomplete branch")
    s76.branches.prior.audit_calls(data, (len(frames), 4))
    if not np.isfinite(data["decode_error"]).all() or data["decode_error"].max() > ap["design"]["box_atol"]:
        raise ValueError("Native decode drift")
    history = list(prefix.copy())
    for i in range(len(frames)):
        parents = parents_at(history, ap["design"], mix if i == 0 else 0.)
        np.testing.assert_array_equal(data["history_latest"][i], history[-1])
        np.testing.assert_array_equal(data["candidate_parents"][i], parents)
        for q, parent in enumerate(parents):
            crop = crop_geometry(parent, data["image_hw"][i], ap["search_factor"], ap["search_size"])[0]
            np.testing.assert_array_equal(data["crop"][i, q], crop)
        row = choose({k: data[k][i].copy() for k in ("features", "candidate_boxes", "candidate_parents")}, history)
        for k in ("selected", "boxes"):
            np.testing.assert_array_equal(data[k][i], row[k])
        history = (history + [data["boxes"][i]])[-ap["design"]["history"]:]


def metrics(data, truth, design):
    iou = tracking_iou(data["boxes"], truth)
    best = tracking_iou(data["candidate_boxes"], truth[:, None]).max(axis=1)
    if np.any(iou < 0) or np.any(best < 0):
        raise ValueError("Invalid evaluation box or label in continuation window")
    events = failure_events(iou)
    run = design["recovery_run"]
    good = np.lib.stride_tricks.sliding_window_view(iou >= design["good_iou"], run).all(axis=1)
    return dict(first_iou=float(iou[0]), first_best_iou=float(best[0]), first_action=int(data["selected"][0]),
                mean_iou=float(iou.mean()), endpoint_iou=float(iou[-1]), good_candidate_fraction=float(
                    (best >= design["good_iou"]).mean()), low_iou_fraction=float((iou < design["low_iou"]).mean()),
                sustained_low_iou=bool(events), unresolved_at_window_end=any(e["recovery"] is None for e in events),
                reaches_sustained_good=bool(good.any()))


def aggregate(rows, design):
    by_video = defaultdict(list)
    for row in rows:
        by_video[row["stratum"], row["sequence"]].append(row)
    fields = ("first_iou", "first_best_iou", "mean_iou", "endpoint_iou", "good_candidate_fraction", "low_iou_fraction",
              "sustained_low_iou", "unresolved_at_window_end", "reaches_sustained_good")
    videos, paired = [], []
    for (stratum, sequence), group in sorted(by_video.items()):
        videos.append(dict(stratum=stratum, sequence=sequence, windows=len(group),
                           **{f"delta_{k}": float(np.mean([float(r["midpoint_" + k]) - float(r["fixed_" + k])
                                                         for r in group])) for k in fields}))
    for stratum in STRATA:
        group = [r for r in videos if r["stratum"] == stratum]
        windows = [r for r in rows if r["stratum"] == stratum]
        for k in fields:
            # The repository helper uses 'low_iou_fraction' for metrics where less is better.
            direction = "low_iou_fraction" if k in ("sustained_low_iou", "unresolved_at_window_end") else k
            row = contrast([r["delta_" + k] for r in group], "midpoint_minus_fixed", direction, design)
            paired.append(dict(row, stratum=stratum, metric=k, windows=len(windows)))
    return videos, paired


def run(root, plan, source, actor, observer):
    old, design = source["original"], plan["design"]
    ap = e74.e67.actor_plan(old)
    evidence, windows, scope = {}, [], []
    for unit_windows in group_windows(plan["windows"]).values():
        unit = unit_windows[0]["unit"]
        entry, trace = previous.read_source_trace(source, dict(unit=unit, frames=[1], expected_first_failure=None))
        pack = e74.learned_pack(actor["models"], unit, actor["prior"])
        trajectory = np.vstack([entry["initial_box"], trace["boxes"]])
        same_model = plan["runtime"]["tracker"]["device"] == source["plan"]["runtime"]["tracker"]["device"]

        def choose(row, history):
            return e74.choose(row, history, unit, old, actor["prior"], pack)

        with e63.source_experiment.installed(observer, [], unit) as head:
            if head != source["report"]["native_head_sha256"]:
                raise ValueError("Native SGLA head changed")
            for window in unit_windows:
                t = window["anchor"]
                frames = np.arange(t + 1, t + 1 + design["horizon"])
                prefix = trajectory[max(0, t + 1 - ap["design"]["history"]):t + 1]
                data = {}
                for arm, mix in zip(ARMS, (0., design["lag_mix"])):
                    path = root / "collection" / (window["case"] + "__" + arm + ".npz")
                    meta = dict(plan_fingerprint=plan["fingerprint"], case=window["case"], arm=arm)
                    tail = c.read_cache(path, meta)
                    fresh = tail is None
                    if fresh:
                        tail = collect(observer, entry, prefix, frames, mix, ap, choose)
                    audit(tail, prefix, frames, mix, ap, choose)
                    if fresh:
                        c.save(path, tail, meta)
                    evidence[path.name] = c.file_hash(path)
                    data[arm] = tail
                fixed, midpoint = (data[arm] for arm in ARMS)
                np.testing.assert_array_equal(fixed["history_latest"][0], midpoint["history_latest"][0])
                np.testing.assert_array_equal(fixed["candidate_parents"][0, :, 2:], midpoint["candidate_parents"][0, :, 2:])
                for k in ("candidate_parents", "crop", "crop_sha256", "candidate_boxes", "layer"):
                    np.testing.assert_array_equal(fixed[k][0, 2:], midpoint[k][0, 2:])
                box_error = float(np.max(np.abs(fixed["boxes"] - trace["boxes"][frames - 1])))
                actions = int(np.count_nonzero(fixed["selected"] != trace["selected"][frames - 1]))
                crops = int(np.count_nonzero(fixed["crop_sha256"] != trace["crop_sha256"][frames - 1]))
                if same_model and (box_error or actions or crops):
                    raise ValueError(f"Same-model natural continuation drift at {window['case']}")
                scope.append(dict(case=window["case"], same_source_gpu_model=same_model, source_max_box_error=box_error,
                                  source_action_changes=actions, source_crop_changes=crops,
                                  first_changed_crop_pixels=int(np.count_nonzero(fixed["crop_sha256"][0] != midpoint["crop_sha256"][0])),
                                  search_calls=sum(int(a["head_calls"].sum()) for a in data.values()),
                                  block_calls=sum(int(a["block_calls"].sum()) for a in data.values())))

                # GT enters analysis only after both branches are fixed.
                truth = np.asarray(entry["ground_truth"], float)
                anchor_truth = truth[t - 1:t + 1]
                if not np.isfinite(anchor_truth).all() or np.any(anchor_truth[:, 2:] <= 0):
                    raise ValueError("Invalid anchor labels")
                gt_move = np.linalg.norm(centers(truth[t]) - centers(truth[t - 1])) / np.sqrt(truth[t - 1, 2:].prod())
                row = dict(case=window["case"], sequence=unit["sequence"], seed=unit["seed"], time_bin=window["time_bin"],
                           stratum=window["stratum"], anchor=t, predicted_displacement=window["predicted_displacement"],
                           applied_center_shift_px=float(np.linalg.norm(centers(fixed["candidate_parents"][0, 0]) -
                                                                        centers(midpoint["candidate_parents"][0, 0]))),
                           anchor_iou=float(tracking_iou(prefix[-1], truth[t])), gt_anchor_displacement=float(gt_move))
                for arm in ARMS:
                    row.update({arm + "_" + k: v for k, v in metrics(data[arm], truth[frames], design).items()})
                windows.append(row)
        print(f"[same-output replay] {unit['sequence']} seed={unit['seed']}", flush=True)
    publish(root, plan, windows, scope, evidence)


def group_windows(windows):
    groups = defaultdict(list)
    for window in windows:
        groups[c.key(window["unit"])].append(window)
    return groups


def event_support(windows, design):
    result = {}
    for stratum in STRATA:
        group = [r for r in windows if r["stratum"] == stratum]
        healthy = [r for r in group if r["anchor_iou"] >= design["good_iou"]]
        low = [r for r in group if 0 <= r["anchor_iou"] < design["low_iou"]]
        result[stratum] = dict(
            initially_good_windows=len(healthy), initially_low_windows=len(low),
            midpoint_only_failure=sum(r["midpoint_sustained_low_iou"] and not r["fixed_sustained_low_iou"] for r in healthy),
            fixed_only_failure=sum(r["fixed_sustained_low_iou"] and not r["midpoint_sustained_low_iou"] for r in healthy),
            midpoint_only_return_to_good=sum(r["midpoint_reaches_sustained_good"] and not r["fixed_reaches_sustained_good"] for r in low),
            fixed_only_return_to_good=sum(r["fixed_reaches_sustained_good"] and not r["midpoint_reaches_sustained_good"] for r in low))
    return result


def publish(root, plan, windows, scope, evidence):
    calls = sum(r["search_calls"] for r in scope)
    blocks = sum(r["block_calls"] for r in scope)
    if calls != plan["expected_search_calls"] or blocks != 7 * calls:
        raise ValueError("Unexpected search budget")
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])
    videos, paired = aggregate(windows, plan["design"])
    hashes = {}
    (root / "analysis").mkdir(exist_ok=True)
    for name, rows in zip(TABLES, (windows, videos, paired, scope)):
        path = root / "analysis" / f"stage86_{name}.csv"
        publish_table(path, rows)
        hashes[path.name] = c.file_hash(path)
    c.publish_identical(root / "analysis/stage86_decision.json", io.seal(dict(
        schema=SCHEMA, status=STATUS, plan_fingerprint=plan["fingerprint"], runtime=plan["runtime"],
        source_stage76_fingerprint=plan["source_stage76_fingerprint"], windows=len(windows), branches=len(evidence),
        search_calls=calls, block_calls=blocks, confirmation_inference_calls=0, training_updates=0,
        event_support=event_support(windows, plan["design"]),
        windows_with_changed_first_crop=sum(r["first_changed_crop_pixels"] > 0 for r in scope),
        source_max_box_error=max(r["source_max_box_error"] for r in scope), source_action_changes=sum(
            r["source_action_changes"] for r in scope), source_crop_changes=sum(r["source_crop_changes"] for r in scope),
        cache_sha256=evidence, table_sha256=hashes)))
    print(f"Stage-86: {STATUS} ({len(windows)} pairs, {calls} searches)", flush=True)


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    report = c.c66.sealed(root / "analysis/stage86_decision.json", SCHEMA)
    caches = {w["case"] + "__" + arm + ".npz" for w in plan["windows"] for arm in ARMS}
    if (report["status"] != STATUS or report["plan_fingerprint"] != plan["fingerprint"]
            or report["windows"] != len(plan["windows"]) or report["branches"] != len(caches)
            or report["search_calls"] != plan["expected_search_calls"] or report["block_calls"] != 7 * report["search_calls"]
            or set(report["cache_sha256"]) != caches or set(report["table_sha256"]) != {f"stage86_{n}.csv" for n in TABLES}):
        raise ValueError("Incomplete Stage-86 result")
    for directory, hashes in (("analysis", report["table_sha256"]), ("collection", report["cache_sha256"])):
        io.verify({str(root / directory / name): digest for name, digest in hashes.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage76", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage86")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not args.stage76 or not args.checkpoint:
        parser.error("Supply the Stage-76 root and original SGLA checkpoint")
    upstream = args.stage76.expanduser().resolve()
    if root == upstream or root in upstream.parents or upstream in root.parents:
        parser.error("Use a separate Stage-86 root")
    import torch

    torch.set_num_threads(1)
    with writer_lock(root):
        if (root / "analysis/stage86_decision.json").exists():
            check_results(root)
            return
        if not (root / "manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-86 directory")
        source = previous.load_source(upstream)
        actor = io.load_source(Path(source["plan"]["source_input"]))
        if actor["plan"] != source["original"] or actor["models"]["fingerprint"] != source["report"]["source_training_fingerprint"]:
            raise ValueError("Stage-74 actor changed")
        roster = c.c66.sealed(ROSTER, confirmation.SCHEMA)
        held = {r["source_group_proxy"] for r in roster["sequences"]}
        units = [u for u in source["plan"]["units"] if u["policy"] == "ppo"]
        names = {u["sequence"] for u in units}
        if (len(names) != 12 or len(units) != 24 or {(u["sequence"], u["seed"]) for u in units} !=
                {(name, seed) for name in names for seed in (2171, 2172)} or len(held) != 28
                or {confirmation.source_group(name) for name in names} & held):
            raise ValueError("Need the fixed development cohort disjoint from the confirmation roster")
        windows = []
        for unit in units:
            entry, trace = previous.read_source_trace(source, dict(unit=unit, frames=[1], expected_first_failure=None))
            windows.extend(choose_windows(np.vstack([entry["initial_box"], trace["boxes"]]), unit, DESIGN))
        if len(windows) != 144:
            raise ValueError("Fixed Stage-86 window count changed")
        runtime = s76.device_runtime(source["original"], torch, args.gpu)
        code = [Path(__file__), Path(__file__).with_name("run_stage86_search_center.sh")]
        code.extend(Path(m.__file__) for m in (previous, s76, e74, e63, e63.actor))
        plan = io.seal(dict(schema=SCHEMA, role="consumed_development_single_search_center_intervention",
            design=DESIGN, windows=windows, expected_search_calls=8 * DESIGN["horizon"] * len(windows), runtime=runtime,
            source_stage76_fingerprint=source["report"]["fingerprint"], confirmation_fingerprint=roster["fingerprint"],
            source_sha256=source["evidence"] | actor["bindings"] | {str(ROSTER): c.file_hash(ROSTER)},
            code_sha256={str(p.resolve()): c.file_hash(p) for p in code}))
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
