"""Replay the fixed Stage-75 choices on L20 from sealed Stage-74 histories."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
from pathlib import Path

import numpy as np

import stage75_ppo_failure_common as c
import stage75_ppo_failure_experiment as original
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, nonempty_path, tracking_iou


SCHEMA = "stage75_l20_four_choice_failure_replay_v1"
DEVICE = "NVIDIA L20"
startup = original.e63.startup
cache = c.source_common


def prepare(root, source):
    if not (root / "manifest.json").exists() and any(path.name != "collect.lock" for path in root.iterdir()):
        raise ValueError("Use a new output directory for Stage-75 L20")
    cases, tables = c.cached_tables(source)
    plan = c.make_plan(source, cases)
    plan.pop("fingerprint")
    plan["l40_planned_branches"] = plan.pop("branch_count")
    plan["l40_planned_search_calls"] = plan.pop("search_calls")
    plan["code_sha256"].update({str(path): c.file_hash(path) for path in (
        Path(__file__).resolve(), Path(__file__).resolve().with_name("run_stage75_l20_replay.sh"),
    )})
    plan = c.seal(dict(plan, schema=SCHEMA, role="consumed_development_l20_matched_choice_replay",
                       device=DEVICE, maximum_branches=4 * len(cases), cached_same_state_contexts=len(tables["same_state"]),
                       incoming_history="Stage-74 cached L40 PPO history",
                       comparison_baseline="Frozen PPO argmax on the shared L20 observation"))
    c.publish_identical(root / "manifest.json", plan)
    table_hashes = c.publish_tables(root, {f"l40_cached_{name}": rows for name, rows in tables.items()})
    return plan, table_hashes


def l20_uuid(source_plan, gpu, torch):
    visible = startup.gpu_inventory(source_plan, torch)["visible_cuda_devices"]
    selected = next((device for device in visible if device["index"] == gpu), None)
    if selected is None or selected["name"] != DEVICE:
        raise ValueError(f"Stage-75 L20 needs {DEVICE} at CUDA index {gpu}. Visible devices: {visible}")
    return selected["uuid"]


def runtime_record(plan, source, gpu_uuid):
    source_plan = source["plan"]
    old = source_plan["runtime_identity"]
    if old["tracker"] != source_plan["runtime"] or old["tracker"]["device"] != "NVIDIA L40":
        raise ValueError("Need the sealed L40 Stage-74 source runtime")
    if not isinstance(gpu_uuid, str) or gpu_uuid in ("", "unavailable", "None", old["gpu_uuid"]):
        raise ValueError("Need the observed physical L20 UUID")
    identity = dict(old, tracker=dict(old["tracker"], device=DEVICE), gpu_uuid=gpu_uuid)
    return c.seal(dict(schema=SCHEMA, plan_fingerprint=plan["fingerprint"], runtime=identity,
                       declared_runtime_differences=startup.differences(old, identity)))


def audit_anchor(anchor, trace, case, entry, source, pack):
    frame = case["frame"]
    source_plan = source["plan"]
    if set(anchor) != set(original.FIELDS):
        raise ValueError("Incomplete shared L20 observation")
    np.testing.assert_array_equal(anchor["frame_id"], [frame])
    original.prior.audit_calls(anchor, (1, 4))
    for name in ("history_latest", "candidate_parents", "image_hw", "crop", "crop_sha256"):
        np.testing.assert_array_equal(anchor[name], trace[name][frame - 1:frame],
                                      err_msg=f"Shared anchor input changed: {case['case_id']} {name}")
    error = anchor["decode_error"]
    if error.shape != (1,) or not np.isfinite(error).all() or not 0 <= error[0] <= source_plan["actor_design"]["box_atol"]:
        raise ValueError("L20 anchor decode exceeds the source tolerance")
    history = original.e63.history_at(trace, entry["initial_box"], frame, source_plan)
    inputs = {name: anchor[name][0].copy() for name in ("features", "candidate_boxes", "candidate_parents")}
    decision = original.e74.choose(inputs, history, case["unit"], source_plan, source["prior"], pack)
    for name, value in decision.items():
        np.testing.assert_array_equal(anchor[name][0], value)
    return dict(case, recorded_action=int(decision["selected"]), legal_actions=np.flatnonzero(decision["action_mask"]).tolist())


def bridge_row(case, local_case, entry, trace, anchor, baseline):
    first = case["frame"] - 1
    horizon = case["horizon"]
    reference = {name: trace[name][first:first + horizon] for name in original.FIELDS}
    local = {name: np.concatenate((anchor[name], baseline[name])) for name in original.FIELDS}
    truth = np.asarray(entry["ground_truth"], float)[first + 1:first + 1 + horizon]
    l40_iou = tracking_iou(reference["boxes"], truth)
    l20_iou = tracking_iou(local["boxes"], truth)
    differing = [name for name in original.FIELDS if name != "decode_error"
                 and not np.array_equal(reference[name], local[name])]
    first_action_change = c.first_difference(local["selected"], reference["selected"], horizon)
    candidate_iou = tracking_iou(anchor["candidate_boxes"][0], truth[0])
    return dict(case_id=case["case_id"], sequence=case["sequence"], seed=case["seed"], frame=case["frame"],
                horizon=horizon, l40_recorded_action=case["recorded_action"], l20_ppo_action=local_case["recorded_action"],
                l40_distinct_actions="|".join(map(str, case["legal_actions"])),
                l20_distinct_actions="|".join(map(str, local_case["legal_actions"])),
                anchor_max_candidate_error_px=float(np.abs(reference["candidate_boxes"][0] - anchor["candidate_boxes"][0]).max()),
                anchor_max_logit_delta=float(np.abs(reference["logits"][0] - anchor["logits"][0]).max()),
                l20_anchor_category=c.frame_category(candidate_iou, local_case["recorded_action"]),
                l40_window_mean_iou=float(l40_iou.mean()), l20_ppo_window_mean_iou=float(l20_iou.mean()),
                l40_window_low_frames=int((l40_iou < c.DESIGN["low_iou"]).sum()),
                l20_ppo_window_low_frames=int((l20_iou < c.DESIGN["low_iou"]).sum()),
                window_max_output_error_px=float(np.abs(reference["boxes"] - local["boxes"]).max()),
                first_ppo_action_difference=None if first_action_change is None else first + first_action_change,
                exact_source_window_match=not differing, differing_fields="|".join(differing),
                **{f"l20_q{q}_iou": float(value) for q, value in enumerate(candidate_iou)})


def replay_case(root, plan, source, run, case, observer, image_reader):
    trace = c.read_trace(source, case["unit"])
    entry = cache.entry_for(source["plan"], case["sequence"])
    pack = original.e74.learned_pack(source["models"], case["unit"], source["prior"])
    directory = root / "collection" / case["case_id"]
    metadata = dict(schema=SCHEMA, plan_fingerprint=plan["fingerprint"], run_fingerprint=run["fingerprint"],
                    case_id=case["case_id"])
    hashes = {}
    installed = (original.e63.source_experiment.installed(observer, [], case["unit"]) if observer is not None
                 else nullcontext(case["head_sha256"]))
    with installed as head:
        if head != case["head_sha256"]:
            raise ValueError("Frozen SGLA head differs from Stage-74")
        path = directory / "anchor.npz"
        meta = dict(metadata, kind="anchor")
        anchor = cache.read_cache(path, meta)
        new_anchor = anchor is None
        if new_anchor:
            if observer is None:
                raise ValueError(f"Missing committed L20 observation: {path}")
            anchor = original.observe_anchor(observer, entry, trace, case, source, pack, image_reader)
        local_case = audit_anchor(anchor, trace, case, entry, source, pack)
        if new_anchor:
            cache.save(path, anchor, meta)
        original.e63.source_experiment.bind(hashes, path)
        selected = local_case["recorded_action"]
        actions = [selected] + [action for action in local_case["legal_actions"] if action != selected]
        # The original audit still checks calls, causal inputs and PPO choices. L40 equality is measured separately.
        audit_case = dict(local_case, recorded_action=None)
        tails = {}
        for action in actions:
            path = directory / f"q{action}.npz"
            meta = dict(metadata, kind="continuation", forced_action=action)
            tail = cache.read_cache(path, meta)
            new_tail = tail is None
            if new_tail:
                if observer is None:
                    raise ValueError(f"Missing committed L20 continuation: {path}")
                tail = original.collect_tail(observer, entry, trace, anchor, local_case, action, source, pack, image_reader)
            if set(tail) != set(original.FIELDS):
                raise ValueError(f"Incomplete L20 continuation: {path}")
            original.audit_tail(tail, trace, anchor, entry, audit_case, action, source, pack)
            if new_tail:
                cache.save(path, tail, meta)
            original.e63.source_experiment.bind(hashes, path)
            tails[action] = tail

    calls = int(anchor["head_calls"].sum()) + sum(int(tail["head_calls"].sum()) for tail in tails.values())
    if calls != 4 * (1 + len(actions) * (case["horizon"] - 1)):
        raise ValueError(f"L20 search accounting differs for {case['case_id']}")
    rows, timeline = original.summarize_branches(local_case, entry, anchor, tails)
    for table in (rows, timeline):
        for row in table:
            row["l20_ppo_action"] = row.pop("recorded_action")
            row["is_l20_ppo_branch"] = row.pop("is_recorded_branch")
            row["l40_recorded_action"] = case["recorded_action"]
    for row in timeline:
        row["l20_ppo_iou"] = row.pop("recorded_iou")
    bridge = bridge_row(case, local_case, entry, trace, anchor, tails[selected])
    return rows, timeline, bridge, hashes, calls


def replay(root, plan, source, run, cached_tables, observer=None, image_reader=checked_image):
    if run != runtime_record(plan, source, run["runtime"]["gpu_uuid"]):
        raise ValueError("L20 runtime record changed")
    c.publish_identical(root / "run.json", run)
    tables = dict(l20_branches=[], l20_branch_frames=[], l20_device_bridge=[])
    hashes = {}
    calls = 0
    for case in plan["cases"]:
        rows, timeline, bridge, evidence, case_calls = replay_case(root, plan, source, run, case, observer, image_reader)
        tables["l20_branches"].extend(rows)
        tables["l20_branch_frames"].extend(timeline)
        tables["l20_device_bridge"].append(bridge)
        hashes.update(evidence)
        calls += case_calls
        print(f"[Stage-75 L20] {case['case_id']} {case['sequence']} s{case['seed']} f{case['frame']}: "
              f"L40 q{case['recorded_action']}, L20 q{bridge['l20_ppo_action']}, {len(rows)} branches, {case_calls} searches",
              flush=True)
    branches = len(tables["l20_branches"])
    if calls > plan["maximum_search_calls"] or branches > plan["maximum_branches"]:
        raise ValueError("L20 replay exceeded its fixed budget")
    if observer is not None:
        observer.assert_frozen()
    c.verify(plan["source_file_sha256"])
    c.verify(plan["code_sha256"])
    bridge = tables["l20_device_bridge"]
    report = c.seal(dict(schema=SCHEMA, status="COMPLETE_L20_MATCHED_FOUR_CHOICE_FAILURE_REPLAY",
                         role=plan["role"], plan_fingerprint=plan["fingerprint"], run_fingerprint=run["fingerprint"],
                         source_report_fingerprint=plan["source_report_fingerprint"],
                         source_training_fingerprint=plan["source_training_fingerprint"],
                         runtime=run["runtime"], declared_runtime_differences=run["declared_runtime_differences"],
                         comparison_baseline=plan["comparison_baseline"], incoming_history=plan["incoming_history"],
                         cases=len(bridge), videos=len({row["sequence"] for row in bridge}), branches=branches,
                         cached_same_state_contexts=plan["cached_same_state_contexts"], maximum_branches=plan["maximum_branches"],
                         anchor_ppo_action_matches=sum(row["l40_recorded_action"] == row["l20_ppo_action"] for row in bridge),
                         exact_source_windows=sum(row["exact_source_window_match"] for row in bridge),
                         search_calls=calls, block_calls=7 * calls, maximum_search_calls=plan["maximum_search_calls"],
                         new_full_trajectories=0, training_updates=0, new_recovery_search_calls=0, cache_sha256=hashes,
                         table_sha256=dict(cached_tables, **c.publish_tables(root, tables))))
    c.publish_identical(root / "analysis/stage75_l20_diagnostic.json", report)
    print(f"Stage-75 L20: {report['status']} ({calls} searches, {branches} branches)", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=nonempty_path)
    parser.add_argument("--output_dir", required=True, type=nonempty_path)
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--parameter", default="deit_distilled")
    parser.set_defaults(mode="replay")
    args = parser.parse_args()
    root = Path(args.output_dir).expanduser().resolve()
    source_root = Path(args.input).expanduser().resolve()
    if root == source_root or root in source_root.parents or source_root in root.parents:
        parser.error("Stage-75 L20 output must be separate from Stage-74")
    with writer_lock(root):
        print("[Stage-75 L20] Checking the sealed Stage-74 source chain", flush=True)
        source = c.load_source(source_root)
        plan, cached_tables = prepare(root, source)
        run_path = root / "run.json"
        if (root / "analysis/stage75_l20_diagnostic.json").is_file():
            run = cache.c66.sealed(run_path, SCHEMA)
            replay(root, plan, source, run, cached_tables)
            return
        if not args.checkpoint:
            parser.error("Replay requires the original frozen SGLA checkpoint")
        import torch

        run = runtime_record(plan, source, l20_uuid(source["plan"], args.gpu, torch))
        if run_path.exists() and cache.c66.sealed(run_path, SCHEMA) != run:
            raise ValueError("Resume with the L20 UUID bound in run.json or use a new L20 output directory")
        identity = run["runtime"]
        runtime_plan = dict(source["plan"], fingerprint=plan["fingerprint"],
                            runtime=identity["tracker"], runtime_identity=identity,
                            hardware=dict(device=DEVICE, gpu_uuid=identity["gpu_uuid"]))
        native, actual_identity = startup.checked_runtime(args, runtime_plan, root)
        if actual_identity != identity:
            raise ValueError("Observed replay runtime differs from the bound L20")
        observer = original.e63.Capture(native)
        try:
            replay(root, plan, source, run, cached_tables, observer)
        finally:
            observer.handle.remove()


if __name__ == "__main__":
    main()
