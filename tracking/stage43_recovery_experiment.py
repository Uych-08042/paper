"""Run complete causal recovery trajectories with frozen six-block students."""

import argparse
import os
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_cache
from stage20_compression_common import state_digest
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, nonempty_path, publish_identical, read_boxes, seal, tracking_iou
from stage28_response_collect import validate_runtime
from stage28_response_common import cpu_array, dense_boxes
from stage34_identity_experiment import seed_runtime
from stage35_preserve_runtime import forward_student
from stage37_crossed_common import fold_for
from stage43_recovery_common import (Controller, DESIGN, MODELS, POLICIES, SCHEMA, SEEDS,
    audit_control, load_plan, load_result, metadata, prepare, read_sealed, result_path, source, tasks, verify_bindings)


def observation(rt, plan, net, image, parent, frame):
    """No annotation, future frame, donor tracker or source trajectory is accepted."""
    rt.tracker.state, rt.tracker.frame_id = parent.tolist(), frame - 1
    search, factor = rt.prepare(image)
    saved = search.clone()
    output = forward_student(rt, net, search)
    if not rt.torch.equal(saved, search):
        raise ValueError("Frozen network modified its search tensor")
    box = np.asarray(rt.decode(output, factor, image), float)
    np.testing.assert_array_equal(rt.tracker.state, parent)
    side = rt.search_side
    raw = cpu_array(output["score_map"]).reshape(side, side)
    window = cpu_array(rt.tracker.output_window).reshape(side, side)
    cell = int(np.argmax(raw * window))
    dense = dense_boxes(cpu_array(output["size_map"]).reshape(2, side, side),
        cpu_array(output["offset_map"]).reshape(2, side, side), parent, factor, plan["search_size"], image.shape[:2])
    return box, float(raw.ravel()[cell]), float(np.max(np.abs(dense[cell] - box)))


def rollout(rt, plan, entry, net, policy, limit=None, image_reader=checked_image):
    n = min(len(entry["frames"]) - 1, limit) if limit else len(entry["frames"]) - 1
    vector = ("previous", "boxes", "native_boxes", "alternate_boxes", "alternate_parent", "forecast")
    scalar = ("native_score", "alternate_score", "score_floor", "innovation", "native_decode_error", "alternate_decode_error")
    flags = ("alarm", "eligible", "choice", "history_update", "interventions", "head_calls")
    data = {k:np.zeros((n, 4)) for k in vector}
    data.update({k:np.zeros(n) for k in scalar})
    data.update({k:np.zeros(n, np.int64) for k in flags})
    data["image_hw"] = np.zeros((n, 2), np.int64)
    initial = image_reader(plan["root"], entry["frames"][0])
    previous = np.asarray(entry["initial_box"], float).copy()
    controller, calls = Controller(previous, policy), [0]
    hook = net.box_head.register_forward_hook(lambda *unused:calls.__setitem__(0, calls[0] + 1))
    try:
        with rt.torch.no_grad():
            rt.initialize(initial, entry["initial_box"])
            for i in range(n):
                frame, before = i + 1, calls[0]
                image = image_reader(plan["root"], entry["frames"][frame])
                box, confidence, error = observation(rt, plan, net, image, previous, frame)
                view = controller.inspect(frame, previous, box, confidence, image.shape[:2])
                alternative = None
                if view["eligible"] and policy in ("local_search", "history_search"):
                    alt_box, alt_conf, alt_error = observation(rt, plan, net, image, view["alternate_parent"], frame)
                    alternative = (alt_box, alt_conf)
                    data["alternate_parent"][i], data["alternate_boxes"][i] = view["alternate_parent"], alt_box
                    data["alternate_score"][i], data["alternate_decode_error"][i] = alt_conf, alt_error
                selected, log = controller.finish(frame, previous, box, confidence, image.shape[:2], alternative)
                data["previous"][i], data["boxes"][i], data["native_boxes"][i] = previous, selected, box
                data["native_score"][i], data["native_decode_error"][i] = confidence, error
                data["image_hw"][i] = image.shape[:2]
                for key in ("alarm", "eligible", "choice", "history_update", "interventions", "score_floor", "innovation", "forecast"):
                    data[key][i] = log[key]
                data["head_calls"][i] = calls[0] - before
                if calls[0] - before != 1 + int(alternative is not None):
                    raise ValueError("Unaccounted model/head call")
                previous = selected.copy()
                if frame % 500 == 0:
                    print("[{}] {} {}/{} extra={} changes={}".format(policy, entry["sequence"], frame, n,
                        calls[0] - frame, int((data["choice"][:frame] != 0).sum())), flush=True)
    finally:
        hook.remove()
    # Future labels are used only now, after the entire autonomous trajectory.
    annotation = Path(plan["root"]) / entry["annotation"]
    verify_bindings({str(annotation):entry["annotation_sha256"]})
    data["gt"] = read_boxes(annotation)[1:n + 1]
    data["iou"] = tracking_iou(data["boxes"], data["gt"])
    audit_control(data, entry, policy)
    return data


def validate_run(run, plan, mode):
    if (run.get("mode") != mode or run.get("plan_fingerprint") != plan["fingerprint"] or
            run.get("runtime") != plan["source_runtime"] or run.get("window") != plan["window"] or
            run.get("model_state_sha256") != plan["source_model_states"] or
            run.get("model_file_sha256") != plan["source_model_files"] or
            run.get("teacher_sha256") != plan["source_teacher_sha256"]):
        raise ValueError("Mixed Stage-43 runtime/models/window")


def native_parity(data, plan, entry, seed, model):
    reference = source.source_traces(plan, entry, seed)[model]
    n = len(data["boxes"])
    box = float(np.max(np.abs(data["boxes"] - reference["boxes"][:n])))
    iou = float(np.max(np.abs(data["iou"] - reference["iou"][:n])))
    if box > source.DESIGN["native_box_atol"] or iou > source.DESIGN["native_iou_atol"]:
        raise ValueError("Stage-43 native differs from frozen Stage-40 full trajectory")
    return dict(sequence=entry["sequence"], seed=seed, model=model, box_error=box, iou_error=iou)


def smoke_gate(root, plan):
    directory = Path(root) / "smoke"
    run = read_sealed(directory / "run.json", SCHEMA)
    validate_run(run, plan, "smoke")
    report = read_sealed(directory / "stage43_smoke.json", SCHEMA)
    if report.get("status") != "PASS_CAUSAL_RECOVERY_SMOKE" or report.get("run_fingerprint") != run["fingerprint"]:
        raise ValueError("Need unchanged Stage-43 causal recovery smoke")
    hashes = {}
    for e,s,m,p in tasks(plan, "smoke"):
        path = result_path(root, "smoke", e, s, m, p)
        data = load_result(path, plan, run, e, s, m, p)
        if p == "native":
            native_parity(data, plan, e, s, m)
        hashes[path.name] = file_hash(path)
    if hashes != report["cache_sha256"]:
        raise ValueError("Stage-43 smoke evidence changed")


def collect(rt, plan, root, mode, sequence=None, image_reader=checked_image):
    from stage40_crop_experiment import evaluation_models
    validate_runtime(rt, plan)
    if mode == "full":
        smoke_gate(root, plan)
    original = source.training.load_plan(plan["source_stage40"])
    models, files = evaluation_models(rt, original, Path(plan["source_stage40"]))
    digests = {"{}:{}".format(*k):state_digest(net) for k,net in sorted(models.items())}
    run = seal(dict(schema=SCHEMA, mode=mode, plan_fingerprint=plan["fingerprint"], runtime=rt.identity,
        model_state_sha256=digests, model_file_sha256=files, teacher_sha256=state_digest(rt.net),
        window=cpu_array(rt.tracker.output_window).reshape(rt.search_side, rt.search_side).tolist(),
        timing_role="Explicit forward ledger only; no deployment speed or energy claim"))
    validate_run(run, plan, mode)
    selected = [t for t in tasks(plan, mode) if sequence is None or t[0]["sequence"] == sequence]
    if not selected or (mode == "smoke" and sequence):
        raise ValueError("Use fixed smoke video or a full original-video sequence")
    directory, hashes, parity = Path(root) / mode, {}, []
    with writer_lock(directory):
        if not (directory / "run.json").exists() and any(directory.rglob("*.npz")):
            raise ValueError("Unidentified Stage-43 traces")
        publish_identical(directory / "run.json", run)
        for entry, seed, model, policy in selected:
            path = result_path(root, mode, entry, seed, model, policy)
            if not path.exists():
                if path.with_suffix(".json").exists():
                    raise ValueError("Committed Stage-43 video missing")
                net = models[(fold_for(plan, entry), model + "_seed" + str(seed))]
                data = rollout(rt, plan, entry, net, policy, DESIGN["smoke_frames"] if mode == "smoke" else None, image_reader)
                write_cache(path, data, metadata(run, entry, seed, model, policy))
                action = "done"
            else:
                action = "reuse, audited"
            data = load_result(path, plan, run, entry, seed, model, policy, allow_orphan=True)
            if policy == "native":
                parity.append(native_parity(data, plan, entry, seed, model))
            hashes[path.name] = file_hash(path)
            print("[{}] {} calls={} changes={}".format(action, path.name, int(data["head_calls"].sum()),
                int((data["choice"] != 0).sum())), flush=True)
        after = {"{}:{}".format(*k):state_digest(net) for k,net in sorted(models.items())}
        if after != digests or state_digest(rt.net) != plan["source_teacher_sha256"]:
            raise ValueError("Frozen model changed during online recovery")
        verify_bindings(plan["source_file_sha256"])
        if mode == "smoke":
            report = seal(dict(schema=SCHEMA, status="PASS_CAUSAL_RECOVERY_SMOKE", run_fingerprint=run["fingerprint"],
                cache_sha256=hashes, native_parity=parity, training_performed=False,
                warning="Execution, baseline parity, causal state and call accounting only; may have zero real triggers"))
            publish_identical(directory / "stage43_smoke.json", report)
            print("Stage-43: " + report["status"])
    return run


def status(root, plan):
    for mode in ("smoke", "full"):
        complete = 0
        if (Path(root) / mode / "run.json").exists():
            run = read_sealed(Path(root) / mode / "run.json", SCHEMA)
            validate_run(run, plan, mode)
            for e,s,m,p in tasks(plan, mode):
                path = result_path(root, mode, e, s, m, p)
                if path.with_suffix(".json").exists():
                    load_result(path, plan, run, e, s, m, p)
                    complete += 1
        print("{}: {}/{} committed videos; unfinished video restarts from frame 1".format(mode, complete, len(tasks(plan, mode))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True, help="Stage-42 root for prepare; Stage-43 root otherwise")
    parser.add_argument("--mode", choices=("prepare", "smoke", "evaluate", "status"), required=True)
    parser.add_argument("--output_dir", type=nonempty_path)
    parser.add_argument("--checkpoint", type=nonempty_path)
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sequence")
    args = parser.parse_args()
    root = Path(args.input).expanduser().resolve()
    if not (root / "manifest.json").is_file():
        parser.error("--input must contain manifest.json; use the complete Stage-42 root for prepare, not analysis/")
    if args.mode == "prepare":
        if not args.output_dir:
            parser.error("prepare requires --output_dir")
        with writer_lock(Path(args.output_dir).expanduser().resolve()):
            prepare(root, args.output_dir)
        return
    plan = load_plan(root)
    if args.mode == "status":
        status(root, plan)
        return
    checkpoint = args.checkpoint or os.environ.get("CHECKPOINT") or None
    if checkpoint and file_hash(Path(checkpoint).expanduser()) != plan["runtime"]["checkpoint_sha256"]:
        parser.error("Use the original frozen SGLA checkpoint")
    import torch
    from stage4_runtime import Runtime
    seed_runtime(torch, plan["runtime_seed"])
    collect(Runtime(args.parameter, args.gpu, checkpoint), plan, root,
        "smoke" if args.mode == "smoke" else "full", args.sequence)


if __name__ == "__main__":
    main()
