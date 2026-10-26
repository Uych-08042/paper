"""Collect frozen 2x2 crossed-decoder rollouts on eight consumed OOF videos."""

import argparse
import os
import time
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_cache
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, nonempty_path, publish_identical, read_boxes, seal, tracking_iou
from stage28_response_collect import validate_runtime
from stage28_response_common import cpu_array, dense_boxes
from stage34_identity_experiment import seed_runtime
from stage35_preserve_runtime import forward_student
from stage36_same_crop_common import load_models
from stage37_crossed_common import (DESIGN, SCHEMA, arm_seed, audit_trace, channels, compose,
    fold_for, load_plan, load_trace, metadata, methods, native_parity, prepare, read_sealed, verify_bindings)


def collect_trace(rt, plan, entry, method, net, limit=None, image_reader=checked_image):
    torch, side = rt.torch, rt.search_side
    arm, _ = arm_seed(method)
    n = min(len(entry["frames"]) - 1, limit) if limit else len(entry["frames"]) - 1
    data = dict(frame_id=np.arange(1, n + 1), previous=np.empty((n, 4)), boxes=np.empty((n, 4)),
        elapsed_ms=np.empty(n), raw=np.empty((n, 2, side, side), np.float32),
        size=np.empty((n, 2, 2, side, side), np.float32), offset=np.empty((n, 2, 2, side, side), np.float32),
        factor=np.empty(n), image_hw=np.empty((n, 2), np.int64), layer=np.empty(n, np.int64),
        decode_error=np.empty(n), window=cpu_array(rt.tracker.output_window).reshape(side, side).astype(np.float32))
    calls = [0, 0]
    hooks = [h.register_forward_hook(lambda *unused, k=k: calls.__setitem__(k, calls[k] + 1))
        for k, h in enumerate((net.box_head, rt.net.box_head))]
    try:
        with torch.no_grad():
            rt.initialize(image_reader(plan["root"], entry["frames"][0]), entry["initial_box"])
            previous = np.asarray(entry["initial_box"], float)
            for row in range(n):
                index = row + 1
                image = image_reader(plan["root"], entry["frames"][index])
                rt.tracker.state, rt.tracker.frame_id = previous.tolist(), index - 1
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                search, factor = rt.prepare(image)  # Exactly one tensor, from THIS arm's own history.
                saved_search = search.clone()
                tokens = rt.prefix(search)
                layer = rt.selected_layer(rt.selector(tokens))
                teacher = rt.candidate(tokens, layer)
                student = forward_student(rt, net, search)
                if not torch.equal(saved_search, search):
                    raise ValueError("Frozen models modified the shared search input")
                outputs = [student, teacher]
                box = np.asarray(rt.decode(compose(outputs, arm), factor, image), float)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                data["elapsed_ms"][row] = (time.perf_counter() - start) * 1000
                np.testing.assert_array_equal(rt.tracker.state, previous)
                for model, output in enumerate(outputs):
                    data["raw"][row, model] = cpu_array(output["score_map"]).reshape(side, side)
                    data["size"][row, model] = cpu_array(output["size_map"]).reshape(2, side, side)
                    data["offset"][row, model] = cpu_array(output["offset_map"]).reshape(2, side, side)
                r, g = channels(arm)
                dense = dense_boxes(data["size"][row, g], data["offset"][row, g], previous,
                    factor, plan["search_size"], image.shape[:2])
                cell = int(np.argmax(data["raw"][row, r] * data["window"]))
                data["decode_error"][row] = np.max(np.abs(dense[cell] - box))
                data["previous"][row], data["boxes"][row] = previous, box
                data["factor"][row], data["image_hw"][row], data["layer"][row] = factor, image.shape[:2], layer
                previous = box.copy()
                rt.tracker.frame_id = index
                if index % 250 == 0:
                    print("[{}] {} {}/{}".format(method, entry["sequence"], index, n), flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    # Labels enter only the evaluator after this autonomous rollout is complete.
    annotation = Path(plan["root"]) / entry["annotation"]
    if file_hash(annotation) != entry["annotation_sha256"]:
        raise ValueError("OOF annotations changed")
    gt = read_boxes(annotation)
    if len(gt) != len(entry["frames"]):
        raise ValueError("OOF label coverage changed")
    data.update(gt=gt[1:n + 1], iou=tracking_iou(data["boxes"], gt[1:n + 1]), model_head_calls=np.asarray(calls))
    audit_trace(data, plan, entry, method, limit)
    native_parity(data, plan, entry, method)
    return data


def run_identity(rt, plan, models, mode):
    from stage20_compression_common import state_digest
    frozen = {"sgla": state_digest(rt.net)}
    frozen.update({"{}:{}".format(*key): state_digest(net) for key, net in sorted(models.items())})
    return seal(dict(schema=SCHEMA, role="frozen_two_model_crossed_decoder_integrity" if mode == "smoke" else
        "consumed_fit_only_autonomous_crossed_decoder_rollouts", plan_fingerprint=plan["fingerprint"],
        mode=mode, runtime=rt.identity, methods=methods(), frozen_neural_model_sha256=frozen,
        window=cpu_array(rt.tracker.output_window).reshape(rt.search_side, rt.search_side).tolist(),
        timing_role="Two frozen model forwards in EVERY arm for matched same-state audit; no compression or speed claim",
        gt_use="First-frame initialization; future GT only in offline source audit and post-rollout evaluation"))


def smoke_gate(root, plan):
    run = read_sealed(Path(root) / "smoke/run.json", SCHEMA)
    report = read_sealed(Path(root) / "smoke/stage37_smoke.json", SCHEMA)
    if (run["mode"] != "smoke" or run["plan_fingerprint"] != plan["fingerprint"] or
            report["run_fingerprint"] != run["fingerprint"] or report["status"] != "PASS_FROZEN_CROSSED_SMOKE"):
        raise ValueError("Complete the unchanged Stage-37 smoke before full evaluation")
    paths = {str(Path(root) / "smoke/traces" / n): d for n, d in report["cache_sha256"].items()}
    if len(paths) != len(methods()):
        raise ValueError("Incomplete smoke coverage")
    verify_bindings(paths)
    for method in methods():
        entry = plan["fit"][0]
        load_trace(Path(root) / "smoke/traces" / (entry["sequence"] + "__" + method + ".npz"),
            plan, run, entry, method, DESIGN["smoke_frames"])


def collect(rt, plan, root, mode, sequence=None, image_reader=checked_image):
    from stage20_compression_common import state_digest
    validate_runtime(rt, plan)
    if mode == "full":
        smoke_gate(root, plan)
    models = load_models(rt, dict(plan, fingerprint=plan["source_plan_fingerprint"]), plan["source_input"])
    run = run_identity(rt, plan, models, mode)
    if mode == "full":
        frozen_smoke = read_sealed(Path(root) / "smoke/run.json", SCHEMA)
        if any(run[k] != frozen_smoke[k] for k in ("runtime", "frozen_neural_model_sha256", "window")):
            raise ValueError("Full runtime/models/window differ from the frozen smoke")
    directory = Path(root) / mode
    entries = plan["fit"][:1] if mode == "smoke" else [e for e in plan["fit"] if sequence is None or e["sequence"] == sequence]
    if not entries or (mode == "smoke" and sequence is not None):
        raise ValueError("Smoke uses the fixed first fit video; full --sequence must belong to the original eight")
    with writer_lock(directory):
        if not (directory / "run.json").exists() and any(directory.rglob("*.npz")):
            raise ValueError("Unidentified crossed caches")
        publish_identical(directory / "run.json", run)
        hashes = {}
        for entry in entries:
            fold = fold_for(plan, entry)
            order = methods()
            ordinal = next(i for i, e in enumerate(plan["fit"]) if e["sequence"] == entry["sequence"])
            order = order[ordinal % len(order):] + order[:ordinal % len(order)]
            for method in order:
                _, seed = arm_seed(method)
                path = directory / "traces" / (entry["sequence"] + "__" + method + ".npz")
                limit = DESIGN["smoke_frames"] if mode == "smoke" else None
                if not path.exists():
                    if path.with_suffix(".json").exists():
                        raise ValueError("Committed trace file is missing")
                    data = collect_trace(rt, plan, entry, method, models[(fold, "supervised_seed" + str(seed))], limit, image_reader)
                    np.testing.assert_array_equal(data["window"], np.asarray(run["window"], np.float32))
                    write_cache(path, data, metadata(plan, run, entry, method))
                    action = "done"
                else:
                    action = "reuse, audited"
                data = load_trace(path, plan, run, entry, method, limit, allow_orphan=True)
                hashes[path.name] = file_hash(path)
                print("[{}] {}: {} frames".format(action, path.name, len(data["iou"])), flush=True)
        if state_digest(rt.net) != run["frozen_neural_model_sha256"]["sgla"] or any(
                state_digest(net) != run["frozen_neural_model_sha256"]["{}:{}".format(*key)] for key, net in models.items()):
            raise ValueError("A frozen neural model changed during evaluation")
        verify_bindings(plan["source_file_sha256"])
        if mode == "smoke":
            report = seal(dict(schema=SCHEMA, status="PASS_FROZEN_CROSSED_SMOKE", run_fingerprint=run["fingerprint"],
                sequence=entries[0]["sequence"], frames=min(DESIGN["smoke_frames"], len(entries[0]["frames"]) - 1),
                methods=methods(), cache_sha256=hashes, training_performed=False, learned_policy_evaluated=False,
                warning="Causality, source parity and decoder execution only; no tracking gain established"))
            publish_identical(directory / "stage37_smoke.json", report)
            print("Stage-37: " + report["status"])
    return run


def status(root, plan):
    for mode, entries in (("smoke", plan["fit"][:1]), ("full", plan["fit"])):
        directory = Path(root) / mode
        complete = 0
        if (directory / "run.json").exists():
            run = read_sealed(directory / "run.json", SCHEMA)
            if run["plan_fingerprint"] != plan["fingerprint"] or run["mode"] != mode:
                raise ValueError("Mixed progress run")
            for entry in entries:
                for method in methods():
                    path = directory / "traces" / (entry["sequence"] + "__" + method + ".npz")
                    if path.exists() and path.with_suffix(".json").exists():
                        load_trace(path, plan, run, entry, method, DESIGN["smoke_frames"] if mode == "smoke" else None)
                        complete += 1
                    elif path.with_suffix(".json").exists():
                        raise ValueError("Committed trace file is missing")
        print("{}: {}/{} committed whole-video traces".format(mode, complete, len(entries) * len(methods())))
    print("Interrupted unfinished trace is recomputed; completed audited traces are reused")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True, help="Stage-36 root for prepare; Stage-37 root otherwise")
    parser.add_argument("--mode", choices=("prepare", "smoke", "evaluate", "status"), required=True)
    parser.add_argument("--output_dir", type=nonempty_path)
    parser.add_argument("--checkpoint", type=nonempty_path, help="Original checkpoint; default CHECKPOINT environment variable, then tracker configuration")
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sequence")
    args = parser.parse_args()
    root = Path(args.input).expanduser().resolve()
    if args.mode == "prepare":
        if not args.output_dir:
            parser.error("prepare requires --output_dir")
        output = Path(args.output_dir).expanduser().resolve()
        with writer_lock(output):
            prepare(root, output)
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
    seed_runtime(torch, DESIGN["runtime_seed"])
    collect(Runtime(args.parameter, args.gpu, checkpoint), plan, root,
        "smoke" if args.mode == "smoke" else "full", args.sequence)


if __name__ == "__main__":
    main()
