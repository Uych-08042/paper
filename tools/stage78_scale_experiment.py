"""Matched SGLA continuations with separate search and published box sizes."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

import stage77_module_diagnostic as previous
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, crop_geometry, failure_events, nonempty_path, scene_geometry, tracking_iou
from stage68_advantage_summary import publish_table


s76, c, e63, io = previous.s76, previous.c, previous.e63, previous.io
e74 = s76.e
SCHEMA = "stage78_sgla_scale_feedback_v1"
STATUS = "COMPLETE_MATCHED_SCALE_FEEDBACK_PILOT"
DESIGN = dict(horizon=32, scale_history=5, expansion_limit=1.25, good_iou=.5, low_iou=.2)
ARMS = ("native", "search_scale", "output_size")
TABLES = ("windows", "paired", "timeline")
FIELDS = ("frame_id", "boxes", "selected", "candidate_boxes", "candidate_parents", "features", "crop", "crop_sha256",
          "image_hw", "block_calls", "head_calls", "selector_calls", "layer", "decode_error")


def completed(root, schema, status, table_names):
    plan = c.c66.sealed(root / "manifest.json", schema)
    name = "stage77_diagnostic.json" if schema == previous.SCHEMA else "stage78_decision.json"
    report = c.c66.sealed(root / "analysis" / name, schema)
    if (report["plan_fingerprint"] != plan["fingerprint"] or report["status"] != status
            or set(report["table_sha256"]) != set(table_names)):
        raise ValueError(f"Need the complete sealed run: {root}")
    if schema == SCHEMA and (report["windows"] != len(plan["cases"])
                             or report["physical_search_calls"] != plan["expected_search_calls"]):
        raise ValueError("Completed Stage-78 budget differs from the plan")
    bindings = {str(root / "analysis" / n): digest for n, digest in report["table_sha256"].items()}
    bindings.update({str(path): c.file_hash(path) for path in (root / "manifest.json", root / "analysis" / name)})
    io.verify(bindings)
    return plan, report, bindings


def load_sources(stage76, stage77):
    source = previous.load_source(stage76)
    p77, r77, bindings = completed(stage77, previous.SCHEMA, "COMPLETE_SGLA_MODULE_DIAGNOSTIC",
                                   ("stage77_summary.csv", "stage77_states.csv", "stage77_candidates.csv"))
    if (p77["source_report_fingerprint"] != source["report"]["fingerprint"]
            or p77["source_evidence_fingerprint"] != c.fingerprint(source["evidence"])
            or r77["states"] != 105 or r77["search_calls"] != 420
            or r77["native_head_sha256"] != source["report"]["native_head_sha256"]):
        raise ValueError("Stage-77 diagnosis must match the frozen Stage-76 source")
    actor = io.load_source(Path(source["plan"]["source_input"]))
    if actor["plan"] != source["original"] or actor["models"]["fingerprint"] != source["report"]["source_training_fingerprint"]:
        raise ValueError("Stage-74 controller differs from the Stage-76 evaluation")
    source["evidence"].update(bindings)
    source["evidence"].update(actor["bindings"])
    return source, actor, r77


def cases(source):
    units = {(u["sequence"], u["policy"], u["seed"]): u for u in source["plan"]["units"]}
    main = previous.MAIN
    specs = (
        ("early_2171", main, "horizon_ce", 2171, 551, 556, "early_choice"),
        ("early_2172", main, "horizon_ce", 2172, 551, 554, "early_choice"),
        ("late_2172", main, "ppo", 2172, 1072, 1090, "primary"),
        ("late_2171", main, "ppo", 2171, 1072, 1090, "stable"),
        ("late_q0", main, "q0", 0, 1072, 1090, "stable"),
        ("choice_2171", "uav0000003_00000_s", "ppo", 2171, 134, 138, "choice"),
        ("choice_2172", "uav0000003_00000_s", "ppo", 2172, 136, 140, "choice"),
        ("recovery_0176", "uav0000176_00000_s", "ppo", 2172, 317, 321, "recovery"),
        ("recovery_0204", "uav0000204_00000_s", "ppo", 2172, 660, 664, "recovery"),
    )
    result = []
    for name, seq, policy, seed, first, focus, role in specs:
        entry = c.entry_for(source["original"], seq)
        last = min(first + DESIGN["horizon"] - 1, len(entry["frames"]) - 1)
        if not first <= focus <= last:
            raise ValueError(f"Missing fixed window: {name}")
        result.append(dict(case=name, role=role, unit=units[seq, policy, seed], first=first, last=last, focus=focus))
    return result


def actor_pack(source, actor, unit):
    if unit["policy"] not in s76.SUPERVISED:
        pack = None if unit["policy"] == "q0" else e74.learned_pack(actor["models"], unit, actor["prior"])
        return unit, pack
    job = c.job_key(unit)
    path = source["root"] / "models" / (job + ".json")
    record = c.c66.sealed(path, s76.SCHEMA)
    if (record["job"] != job or record["plan_fingerprint"] != source["plan"]["fingerprint"]
            or record["fingerprint"] != source["report"]["model_fingerprints"][job]):
        raise ValueError("Stage-76 supervised controller changed")
    return dict(unit, policy="myopic"), record["packs"][unit["policy"]]


def bounded_box(box, history, design):
    """Cap width and height growth relative to prior stabilized boxes, preserving center."""
    log_sizes = np.log(np.asarray(history, float)[-design["scale_history"]:, 2:])
    limit = np.exp(np.median(log_sizes, axis=0)) * design["expansion_limit"]
    width_height = np.minimum(box[2:], limit)
    if np.array_equal(width_height, box[2:]):
        return box.copy()
    return np.r_[box[:2] + (box[2:] - width_height) / 2, width_height]


def advance(raw, stabilized, box, actor_design, design):
    limited = bounded_box(box, stabilized, design)
    raw.append(box.copy())
    stabilized.append(limited)
    del raw[:-actor_design["history"]]
    del stabilized[:-max(actor_design["history"], design["scale_history"])]
    return limited


def collect(observer, frames, initial_box, prefix, frame_ids, unit, old, prior, pack, design, search_scale,
            image_reader=checked_image):
    # Only frame images, the initialization and accepted predictions enter the actor.
    ap = e74.e67.actor_plan(old)
    raw = list(prefix[-ap["design"]["history"]:])
    stabilized = list(prefix[-max(ap["design"]["history"], design["scale_history"]):])
    observer.initialize(image_reader(ap["root"], frames[0]), initial_box)
    rows = []
    for frame in frame_ids:
        active = stabilized[-ap["design"]["history"]:] if search_scale else raw
        parents = s76.branches.prior.history_parents(active, ap["design"])
        image = image_reader(ap["root"], frames[frame])
        observations = [e63.actor.observe(observer, image, parent, frame, ap) for parent in parents]
        # The stabilized sizes affect the crops. Policy history and normalization use raw boxes.
        row = dict(frame_id=frame, candidate_parents=parents,
                   candidate_boxes=np.asarray([r["boxes"] for r in observations]),
                   features=np.asarray([e63.actor.feature_vector(r, raw, ap) for r in observations]),
                   image_hw=np.array(image.shape[:2]), decode_error=max(r["decode_error"] for r in observations),
                   **{name: np.asarray([r[name] for r in observations])
                      for name in ("crop", "crop_sha256", "block_calls", "head_calls", "selector_calls", "layer")})
        row = e74.choose(row, raw, unit, old, prior, pack)
        row["limited"] = advance(raw, stabilized, row["boxes"], ap["design"], design)
        observer.rt.tracker.state = row["boxes"].tolist()
        rows.append(row)
    return {k: np.asarray([r[k] for r in rows]) for k in FIELDS + ("limited",)}


def audit(data, prefix, frame_ids, unit, old, prior, pack, design, search_scale):
    n = len(frame_ids)
    if set(data) != set(FIELDS) | {"limited"}:
        raise ValueError("Missing continuation fields")
    np.testing.assert_array_equal(data["frame_id"], frame_ids)
    s76.branches.prior.audit_calls(data, (n, 4))
    selected = data["selected"]
    if (selected.dtype.kind not in "iu" or selected.shape != (n,) or np.any((selected < 0) | (selected > 3))
            or data["candidate_boxes"].shape != (n, 4, 4) or not np.isfinite(data["candidate_boxes"]).all()
            or np.any(data["candidate_boxes"][..., 2:] <= 0) or not np.isfinite(data["decode_error"]).all()
            or data["decode_error"].max() > old["actor_design"]["box_atol"]):
        raise ValueError("Invalid continuation geometry or action")
    np.testing.assert_array_equal(data["boxes"], data["candidate_boxes"][np.arange(n), selected])
    actor_design = old["actor_design"]
    raw = list(prefix[-actor_design["history"]:])
    stabilized = list(prefix[-max(actor_design["history"], design["scale_history"]):])
    for i, box in enumerate(data["boxes"]):
        history = stabilized if search_scale else raw
        parents = s76.branches.prior.history_parents(history, actor_design)
        np.testing.assert_array_equal(data["candidate_parents"][i], parents)
        for q, parent in enumerate(parents):
            crop = crop_geometry(parent, data["image_hw"][i], old["search_factor"], old["search_size"])[0]
            np.testing.assert_array_equal(data["crop"][i, q], crop)
        policy_inputs = {name: data[name][i] for name in ("features", "candidate_boxes", "candidate_parents")}
        action = e74.choose(policy_inputs, raw, unit, old, prior, pack)["selected"]
        np.testing.assert_array_equal(selected[i], action)
        np.testing.assert_array_equal(data["limited"][i], advance(raw, stabilized, box, actor_design, design))


def measurements(case, entry, old, data):
    gt = np.asarray(entry["ground_truth"], float)
    timeline = []
    for arm in ARMS:
        trace = data["search_scale" if arm == "search_scale" else "native"]
        output = trace["limited"] if arm == "output_size" else trace["boxes"]
        for i, frame in enumerate(trace["frame_id"]):
            truth = gt[frame]
            action = int(trace["selected"][i])
            parent = trace["candidate_parents"][i, action]
            g = scene_geometry(parent, truth, trace["image_hw"][i], old["search_factor"], old["search_size"])
            best = float(tracking_iou(trace["candidate_boxes"][i], truth).max())
            timeline.append(dict(case=case["case"], arm=arm, frame=int(frame), action=action,
                iou=float(tracking_iou(output[i], truth)), raw_iou=float(tracking_iou(trace["boxes"][i], truth)),
                best_candidate_iou=best, crop_side=int(g["crop"][2]), gt_coverage=float(g["coverage"]),
                projected_gt_width=float(g["input_wh"][0]), published_width_ratio=float(output[i, 2] / truth[2]),
                raw_width_ratio=float(trace["boxes"][i, 2] / truth[2]),
                crop_changed=bool(np.any(trace["crop"][i] != data["native"]["crop"][i])),
                crop_pixels_changed=bool(np.any(trace["crop_sha256"][i] != data["native"]["crop_sha256"][i])),
                size_limited=bool(np.any(trace["limited"][i, 2:] < trace["boxes"][i, 2:]))))
    return timeline


def summarize(case, timeline, data, trace, design):
    scope = []
    ref = slice(case["first"] - 1, case["last"])
    base = data["native"]
    crop_changed = (np.any(base["crop_sha256"] != trace["crop_sha256"][ref], axis=1)
                    | np.any(base["crop"] != trace["crop"][ref], axis=(1, 2)))
    bridge = dict(source_action_disagreements=int(np.count_nonzero(base["selected"] != trace["selected"][ref])),
                  source_crop_disagreements=int(crop_changed.sum()),
                  source_max_box_error=float(np.abs(base["boxes"] - trace["boxes"][ref]).max()))
    for arm in ARMS:
        rows = [r for r in timeline if r["arm"] == arm]
        iou = np.array([r["iou"] for r in rows])
        events = failure_events(iou)
        changes = [r["frame"] for r in rows if r["crop_changed"]]
        focus = next(r for r in rows if r["frame"] == case["focus"])
        scope.append(dict(case=case["case"], role=case["role"], **case["unit"], arm=arm, first=case["first"],
                          last=case["last"], frames=len(rows), focus=case["focus"], focus_iou=focus["iou"],
                          mean_iou=float(iou.mean()), endpoint_iou=float(iou[-1]),
                          good_frames=int((iou >= design["good_iou"]).sum()), low_frames=int((iou < design["low_iou"]).sum()),
                          candidate_good_frames=sum(r["best_candidate_iou"] >= design["good_iou"] for r in rows),
                          first_local_failure=rows[events[0]["start"]]["frame"] if events else None,
                          recovered_events=sum(event["recovery"] is not None for event in events),
                          unrecovered_events=sum(event["recovery"] is None for event in events),
                          median_projected_gt_width=float(np.median([r["projected_gt_width"] for r in rows])),
                          median_published_width_ratio=float(np.median([r["published_width_ratio"] for r in rows])),
                          first_changed_crop=changes[0] if changes else None, changed_crop_frames=len(changes),
                          changed_crop_pixel_frames=sum(r["crop_pixels_changed"] for r in rows),
                          size_limit_frames=sum(r["size_limited"] for r in rows) if arm != "native" else 0,
                          logical_search_calls=4 * len(rows), physical_search_calls=4 * len(rows) if arm != "output_size" else 0,
                          **bridge))
    native = scope[0]
    paired = [dict(case=r["case"], role=r["role"], arm=r["arm"], mean_iou_delta=r["mean_iou"] - native["mean_iou"],
                   endpoint_iou_delta=r["endpoint_iou"] - native["endpoint_iou"], focus_iou_delta=r["focus_iou"] - native["focus_iou"],
                   candidate_good_frames_delta=r["candidate_good_frames"] - native["candidate_good_frames"],
                   low_frames_delta=r["low_frames"] - native["low_frames"], good_frames_delta=r["good_frames"] - native["good_frames"])
              for r in scope[1:]]
    return scope, paired


def replay(root, plan, source, actor, observer):
    evidence = {}
    tables = {name: [] for name in TABLES}
    old = source["original"]
    with e63.source_experiment.installed(observer, [], plan["cases"][0]["unit"]) as head:
        if head != source["report"]["native_head_sha256"]:
            raise ValueError("Native SGLA head changed")
        for case in plan["cases"]:
            frames = list(range(case["first"], case["last"] + 1))
            entry, trace = previous.read_source_trace(source, dict(unit=case["unit"], frames=frames, expected_first_failure=None))
            history_size = max(old["actor_design"]["history"], plan["design"]["scale_history"])
            prefix = np.vstack([entry["initial_box"], trace["boxes"][max(0, case["first"] - 1 - history_size):case["first"] - 1]])
            prefix = prefix[-history_size:]
            unit, pack = actor_pack(source, actor, case["unit"])
            data = {}
            for arm in ("native", "search_scale"):
                path = root / "collection" / case["case"] / (arm + ".npz")
                meta = dict(plan_fingerprint=plan["fingerprint"], case=case, arm=arm)
                cached = c.read_cache(path, meta)
                search_scale = arm == "search_scale"
                run = cached if cached is not None else collect(
                    observer, entry["frames"], entry["initial_box"], prefix, frames, unit, old, actor["prior"], pack, plan["design"], search_scale)
                audit(run, prefix, frames, unit, old, actor["prior"], pack, plan["design"], search_scale)
                if cached is None:
                    c.save(path, run, meta)
                s76.binding(evidence, path)
                data[arm] = run
            for field in ("boxes", "candidate_boxes", "candidate_parents", "features", "selected", "crop", "crop_sha256"):
                np.testing.assert_array_equal(data["native"][field][0], data["search_scale"][field][0],
                                              err_msg=f"Start is not matched: {case['case']} {field}")
            timeline = measurements(case, entry, old, data)
            scope, paired = summarize(case, timeline, data, trace, plan["design"])
            for name, rows in zip(TABLES, (scope, paired, timeline)):
                tables[name].extend(rows)
            print(f"[scale continuation] {case['case']} {len(frames)} frames per arm", flush=True)
    calls = sum(r["physical_search_calls"] for r in tables["windows"])
    if calls != plan["expected_search_calls"]:
        raise ValueError("Stage-78 search budget changed")
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])
    directory = root / "analysis"
    directory.mkdir(exist_ok=True)
    hashes = {}
    for name, rows in tables.items():
        path = directory / f"stage78_{name}.csv"
        publish_table(path, rows)
        hashes[path.name] = c.file_hash(path)
    native = [r for r in tables["windows"] if r["arm"] == "native"]
    report = io.seal(dict(schema=SCHEMA, status=STATUS, plan_fingerprint=plan["fingerprint"],
        source_diagnostic_fingerprint=plan["source_diagnostic_fingerprint"], runtime=plan["runtime"],
        declared_runtime_differences=plan["declared_runtime_differences"], design=plan["design"], windows=len(native),
        physical_search_calls=calls, block_calls=7 * calls, training_updates=0, new_full_trajectories=0,
        source_action_disagreements=sum(r["source_action_disagreements"] for r in native),
        source_crop_disagreements=sum(r["source_crop_disagreements"] for r in native),
        source_max_box_error=max(r["source_max_box_error"] for r in native),
        replay_evidence_fingerprint=c.fingerprint(evidence), table_sha256=hashes))
    c.publish_identical(directory / "stage78_decision.json", report)
    print(f"Stage-78: {STATUS} ({len(native)} windows, {calls} searches)", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path)
    parser.add_argument("--diagnostic", type=nonempty_path)
    parser.add_argument("--output_dir", required=True, type=nonempty_path)
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage78")
    args = parser.parse_args()
    root = Path(args.output_dir).expanduser().resolve()
    if args.check_results:
        completed(root, SCHEMA, STATUS, tuple(f"stage78_{n}.csv" for n in TABLES))
        return
    if not (args.input and args.diagnostic and args.checkpoint):
        parser.error("Supply Stage-76, Stage-77 and the original SGLA checkpoint")
    stage76, stage77 = (Path(p).expanduser().resolve() for p in (args.input, args.diagnostic))
    if any(root == p or root in p.parents or p in root.parents for p in (stage76, stage77)):
        parser.error("Use a separate Stage-78 directory")
    import torch

    torch.set_num_threads(1)
    with writer_lock(root):
        if not (root / "manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-78 directory")
        source, actor, report77 = load_sources(stage76, stage77)
        runtime = s76.device_runtime(source["original"], torch, args.gpu)
        windows = cases(source)
        code = [Path(__file__), Path(__file__).with_name("run_stage78_scale.sh"), Path(previous.__file__),
                Path(s76.__file__), Path(e63.__file__), Path(e74.__file__)]
        for item in windows:
            if item["unit"]["policy"] in s76.SUPERVISED:
                path = source["root"] / "models" / (c.job_key(item["unit"]) + ".json")
                source["evidence"][str(path)] = c.file_hash(path)
        plan = io.seal(dict(schema=SCHEMA, role="consumed_development_matched_scale_feedback",
            source_diagnostic_fingerprint=report77["fingerprint"], runtime=runtime,
            declared_runtime_differences=s76.startup.differences(source["plan"]["runtime"], runtime),
            design=dict(DESIGN), cases=windows, arms=list(ARMS),
            expected_search_calls=8 * sum(w["last"] - w["first"] + 1 for w in windows), source_sha256=source["evidence"],
            code_sha256={str(p.resolve()): c.file_hash(p) for p in code}))
        c.publish_identical(root / "manifest.json", plan)
        frozen = dict(source["original"], fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = s76.startup.checked_runtime(args, frozen, root)
        observer = e63.Capture(native)
        try:
            replay(root, plan, source, actor, observer)
        finally:
            observer.handle.remove()


if __name__ == "__main__":
    main()
