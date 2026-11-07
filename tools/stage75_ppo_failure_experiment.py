"""Diagnose Stage-74 decisions and replay one forced choice with the original PPO continuation."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
from pathlib import Path

import numpy as np

import stage75_ppo_failure_common as c
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, crop_geometry, nonempty_path, tracking_iou
from stage61_l40_common import device_binding

e74 = c.source_experiment
e63 = c.e63
prior = e63.source_common.prior
FIELDS = e74.FIELDS


def prepare(root, source):
    if not (root / "manifest.json").exists() and any(path.name != "collect.lock" for path in root.iterdir()):
        raise ValueError("Use an empty output directory for Stage-75")
    cases, tables = c.cached_tables(source)
    plan = c.make_plan(source, cases)
    c.publish_identical(root / "manifest.json", plan)
    c.verify(plan["source_file_sha256"])
    c.verify(plan["code_sha256"])
    table_hashes = c.publish_tables(root, tables)
    cached = c.seal(dict(schema=c.SCHEMA, status="COMPLETE_FOUR_CHOICE_CACHED_DIAGNOSTIC",
                         plan_fingerprint=plan["fingerprint"], source_report_fingerprint=plan["source_report_fingerprint"],
                         cases=len(cases), same_state_contexts=len(tables["same_state"]),
                         cached_timeline_rows=len(tables["timeline"]), table_sha256=table_hashes,
                         search_calls=0, training_updates=0, planned_replay_search_calls=plan["search_calls"],
                         maximum_replay_search_calls=plan["maximum_search_calls"]))
    c.publish_identical(root / "analysis/stage75_cached.json", cached)
    print(f"Stage-75 cached: {len(cases)} anchors, {cached['same_state_contexts']} shared states, "
          f"{plan['branch_count']} planned branches, {plan['search_calls']} planned searches", flush=True)
    return plan, cached


def observe_anchor(observer, entry, trace, case, source, pack, image_reader):
    plan = source["plan"]
    frame = case["frame"]
    history = e63.history_at(trace, entry["initial_box"], frame, plan)
    observer.initialize(image_reader(plan["root"], entry["frames"][0]), entry["initial_box"])
    image = image_reader(plan["root"], entry["frames"][frame])
    row = e63.probe(observer, image, history, frame, "history", e74.e67.actor_plan(plan))
    row = e74.choose(row, history, case["unit"], plan, source["prior"], pack)
    return {name: np.asarray([row[name]]) for name in FIELDS}


def collect_tail(observer, entry, trace, anchor, case, action, source, pack, image_reader):
    frame = case["frame"]
    # Stage-74 consumes only prefix boxes to reconstruct the causal history.
    prefix = {name: trace[name][:frame] for name in FIELDS}
    prefix["boxes"] = prefix["boxes"].copy()
    prefix["boxes"][-1] = anchor["candidate_boxes"][0, action]
    generated = e74.run_actor(observer, entry, case["unit"], source["plan"], source["prior"], pack,
                              limit=frame + case["horizon"] - 1, prefix=prefix, image_reader=image_reader)
    return {name: generated[name][frame:].copy() for name in FIELDS}


def audit_exact(data, trace, first, case_id):
    count = len(data["frame_id"])
    for name in FIELDS:
        if name != "decode_error":
            np.testing.assert_array_equal(data[name], trace[name][first - 1:first - 1 + count],
                                          err_msg=f"Stage-75 {case_id} source replay changed {name}")


def audit_anchor(data, trace, case, source):
    prior.audit_calls(data, (1, 4))
    audit_exact(data, trace, case["frame"], case["case_id"])
    if (len(data["frame_id"]) != 1 or not np.isfinite(data["decode_error"]).all()
            or data["decode_error"].max() > source["plan"]["actor_design"]["box_atol"]
            or int(data["selected"][0]) != case["recorded_action"]):
        raise ValueError("Shared observation does not reproduce the source PPO anchor")
    legal = np.flatnonzero(c.source_common.action_mask(data["candidate_boxes"][0])).tolist()
    if legal != case["legal_actions"]:
        raise ValueError("Duplicate-box equivalence changed at the shared anchor")


def audit_tail(data, trace, anchor, entry, case, action, source, pack):
    frame = case["frame"]
    count = case["horizon"] - 1
    np.testing.assert_array_equal(data["frame_id"], np.arange(frame + 1, frame + count + 1))
    prior.audit_calls(data, (count, 4))
    if action not in case["legal_actions"]:
        raise ValueError("Forced action must be a distinct public box")
    if count == 0:
        return
    plan = source["plan"]
    design = plan["actor_design"]
    if not np.isfinite(data["decode_error"]).all() or data["decode_error"].max() > design["box_atol"]:
        raise ValueError("Tail decode exceeds the source tolerance")
    history = list(e63.history_at(trace, entry["initial_box"], frame, plan))
    history.append(anchor["candidate_boxes"][0, action])
    history = history[-design["history"]:]
    for index in range(count):
        np.testing.assert_array_equal(data["history_latest"][index], history[-1])
        parents = prior.history_parents(history, design)
        np.testing.assert_array_equal(data["candidate_parents"][index], parents)
        for q, parent in enumerate(parents):
            crop = crop_geometry(parent, data["image_hw"][index], plan["search_factor"], plan["search_size"])[0]
            np.testing.assert_array_equal(data["crop"][index, q], crop)
        inputs = {name: data[name][index].copy() for name in ("features", "candidate_boxes", "candidate_parents")}
        decision = e74.choose(inputs, history, case["unit"], plan, source["prior"], pack)
        for name, value in decision.items():
            np.testing.assert_array_equal(data[name][index], value)
        history.append(data["boxes"][index])
        history = history[-design["history"]:]
    if action == case["recorded_action"]:
        audit_exact(data, trace, frame + 1, case["case_id"])


def summarize_branches(case, entry, anchor, tails):
    frame = case["frame"]
    end = frame + case["horizon"]
    truth = np.asarray(entry["ground_truth"], float)[frame:end]
    ious = {}
    for action, tail in tails.items():
        boxes = np.vstack([anchor["candidate_boxes"][0, action], tail["boxes"]])
        ious[action] = tracking_iou(boxes, truth)
    reference = ious[case["recorded_action"]]
    identity = {key: case[key] for key in ("case_id", "sequence", "seed", "frame", "purpose", "horizon")}
    good = c.DESIGN["good_iou"]
    low = c.DESIGN["low_iou"]
    rows, timeline = [], []
    for action, values in sorted(ious.items()):
        branch = dict(identity, forced_action=action, recorded_action=case["recorded_action"],
                      is_recorded_branch=action == case["recorded_action"])
        rows.append(dict(branch, anchor_iou=float(values[0]), mean_iou=float(values.mean()),
                         following_mean_iou=float(values[1:].mean()) if len(values) > 1 else None,
                         endpoint_iou=float(values[-1]), low_frames=int((values < low).sum()),
                         good_frames=int((values >= good).sum()), anchor_gain=float(values[0] - reference[0]),
                         mean_gain=float((values - reference).mean()),
                         following_mean_gain=float((values[1:] - reference[1:]).mean()) if len(values) > 1 else None,
                         low_frame_delta=int((values < low).sum() - (reference < low).sum())))
        selected = [action] + tails[action]["selected"].tolist()
        timeline.extend(dict(branch, tracked_frame=frame + index, selected=selected[index],
                             iou=float(value), recorded_iou=float(reference[index])) for index, value in enumerate(values))
    return rows, timeline


def runtime_record(plan, source, identity):
    differences = e63.startup.differences(source["plan"]["runtime_identity"], identity)
    if any(row["field"] != "runtime.gpu_uuid" for row in differences):
        raise ValueError("Replay requires the source L40 software, model and tracker configuration")
    return c.seal(dict(schema=c.SCHEMA, plan_fingerprint=plan["fingerprint"], runtime=identity,
                       declared_runtime_differences=differences))


def replay_case(root, plan, case, source, run, observer, image_reader):
    trace = c.read_trace(source, case["unit"])
    entry = c.source_common.entry_for(source["plan"], case["sequence"])
    pack = e74.learned_pack(source["models"], case["unit"], source["prior"])
    directory = root / "collection" / case["case_id"]
    metadata = dict(schema=c.SCHEMA, plan_fingerprint=plan["fingerprint"],
                    run_fingerprint=run["fingerprint"], case_id=case["case_id"])
    hashes = {}
    installed = (e63.source_experiment.installed(observer, [], case["unit"]) if observer is not None
                 else nullcontext(case["head_sha256"]))
    with installed as head:
        if head != case["head_sha256"]:
            raise ValueError("Frozen SGLA head differs from Stage-74")
        path = directory / "anchor.npz"
        meta = dict(metadata, kind="anchor")
        anchor = c.source_common.read_cache(path, meta)
        new_anchor = anchor is None
        if new_anchor:
            if observer is None:
                raise ValueError(f"Missing committed shared observation: {path}")
            anchor = observe_anchor(observer, entry, trace, case, source, pack, image_reader)
        audit_anchor(anchor, trace, case, source)
        if new_anchor:
            c.source_common.save(path, anchor, meta)
        e63.source_experiment.bind(hashes, path)
        tails = {}
        recorded = case["recorded_action"]
        actions = [recorded] + [action for action in case["legal_actions"] if action != recorded]
        for action in actions:
            path = directory / f"q{action}.npz"
            meta = dict(metadata, kind="continuation", forced_action=action)
            tail = c.source_common.read_cache(path, meta)
            new_tail = tail is None
            if new_tail:
                if observer is None:
                    raise ValueError(f"Missing committed replay: {path}")
                tail = collect_tail(observer, entry, trace, anchor, case, action, source, pack, image_reader)
            audit_tail(tail, trace, anchor, entry, case, action, source, pack)
            if new_tail:
                c.source_common.save(path, tail, meta)
            e63.source_experiment.bind(hashes, path)
            tails[action] = tail
    calls = int(anchor["head_calls"].sum()) + sum(int(tail["head_calls"].sum()) for tail in tails.values())
    if calls != case["search_calls"]:
        raise ValueError(f"Stage-75 search accounting differs for {case['case_id']}")
    rows, timeline = summarize_branches(case, entry, anchor, tails)
    return rows, timeline, hashes, calls


def replay(root, plan, source, cached, run, observer=None, image_reader=checked_image):
    if run != runtime_record(plan, source, run["runtime"]):
        raise ValueError("Replay runtime record changed")
    c.publish_identical(root / "run.json", run)
    tables = dict(branches=[], branch_frames=[])
    hashes = {}
    calls = 0
    for case in plan["cases"]:
        rows, timeline, evidence, case_calls = replay_case(root, plan, case, source, run, observer, image_reader)
        tables["branches"].extend(rows)
        tables["branch_frames"].extend(timeline)
        hashes.update(evidence)
        calls += case_calls
        print(f"[replayed] {case['case_id']} {case['sequence']} s{case['seed']} f{case['frame']}: "
              f"{len(rows)} branches, {case_calls} searches", flush=True)
    if calls != plan["search_calls"] or len(tables["branches"]) != plan["branch_count"]:
        raise ValueError("Incomplete Stage-75 replay budget")
    if observer is not None:
        observer.assert_frozen()
    c.verify(plan["source_file_sha256"])
    c.verify(plan["code_sha256"])
    table_hashes = dict(cached["table_sha256"], **c.publish_tables(root, tables))
    report = c.seal(dict(schema=c.SCHEMA, status="COMPLETE_FOUR_CHOICE_FAILURE_REPLAY",
                         role=plan["role"], plan_fingerprint=plan["fingerprint"],
                         run_fingerprint=run["fingerprint"], source_report_fingerprint=plan["source_report_fingerprint"],
                         source_training_fingerprint=plan["source_training_fingerprint"],
                         cached_fingerprint=cached["fingerprint"], runtime=run["runtime"],
                         declared_runtime_differences=run["declared_runtime_differences"],
                         cases=len(plan["cases"]), videos=len({case["sequence"] for case in plan["cases"]}),
                         exact_source_branches=len(plan["cases"]), branches=plan["branch_count"],
                         search_calls=calls, block_calls=7 * calls, maximum_search_calls=plan["maximum_search_calls"],
                         new_full_trajectories=0, training_updates=0, table_sha256=table_hashes,
                         cache_sha256=hashes))
    c.publish_identical(root / "analysis/stage75_diagnostic.json", report)
    print(f"Stage-75: {report['status']} ({calls} searches, {report['branches']} branches)", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cached", "replay"), default="replay")
    parser.add_argument("--input", required=True, type=nonempty_path)
    parser.add_argument("--output_dir", required=True, type=nonempty_path)
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--parameter", default="deit_distilled")
    args = parser.parse_args()
    root = Path(args.output_dir).expanduser().resolve()
    source_root = Path(args.input).expanduser().resolve()
    if root == source_root or root in source_root.parents or source_root in root.parents:
        parser.error("Stage-75 output must be separate from Stage-74")
    with writer_lock(root):
        print("[Stage-75] Checking the sealed Stage-74 source chain", flush=True)
        source = c.load_source(source_root)
        plan, cached = prepare(root, source)
        if args.mode == "cached":
            return
        if (root / "analysis/stage75_diagnostic.json").is_file():
            replay(root, plan, source, cached, c.sealed(root / "run.json"))
            return
        if not args.checkpoint:
            parser.error("Replay requires the original frozen SGLA checkpoint")
        import torch

        hardware = device_binding(source["plan"], args.gpu, torch)
        identity = dict(source["plan"]["runtime_identity"], gpu_uuid=hardware["gpu_uuid"])
        run = runtime_record(plan, source, identity)
        if (root / "run.json").is_file() and c.sealed(root / "run.json") != run:
            raise ValueError("Resume Stage-75 on the same bound L40")
        runtime_plan = dict(source["plan"], fingerprint=plan["fingerprint"], runtime_identity=identity, hardware=hardware)
        native, actual_identity = e63.startup.checked_runtime(args, runtime_plan, root)
        if actual_identity != identity:
            raise ValueError("Observed replay runtime differs from the bound L40")
        observer = e63.Capture(native)
        try:
            replay(root, plan, source, cached, run, observer)
        finally:
            observer.handle.remove()


if __name__ == "__main__":
    main()
