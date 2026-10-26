"""Collect only the two new size-only / offset-only autonomous trajectory arms."""

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
from stage37_crossed_common import fold_for
from stage38_geometry_common import (DESIGN, SCHEMA, arm_seed, audit_trace, channels, compose,
    endpoint_parity, load_plan, load_trace, metadata, methods, prepare, read_sealed, verify_bindings)


def collect_trace(rt, plan, entry, method, net, limit=None, image_reader=checked_image):
    torch, side = rt.torch, rt.search_side
    arm, _ = arm_seed(method)
    n = min(len(entry["frames"]) - 1, limit) if limit else len(entry["frames"]) - 1
    data = dict(frame_id=np.arange(1, n + 1), previous=np.empty((n, 4)), boxes=np.empty((n, 4)),
        elapsed_ms=np.empty(n), raw=np.empty((n, 2, side, side), np.float32),
        size=np.empty((n, 2, 2, side, side), np.float32), offset=np.empty((n, 2, 2, side, side), np.float32),
        factor=np.empty(n), image_hw=np.empty((n, 2), np.int64), layer=np.empty(n, np.int64), decode_error=np.empty(n),
        window=cpu_array(rt.tracker.output_window).reshape(side, side).astype(np.float32))
    calls = [0, 0]
    hooks = [head.register_forward_hook(lambda *unused, k=k: calls.__setitem__(k, calls[k] + 1))
        for k, head in enumerate((net.box_head, rt.net.box_head))]
    try:
        with torch.no_grad():
            rt.initialize(image_reader(plan["root"], entry["frames"][0]), entry["initial_box"])
            previous = np.asarray(entry["initial_box"], float)
            for row in range(n):
                image = image_reader(plan["root"], entry["frames"][row + 1])
                rt.tracker.state, rt.tracker.frame_id = previous.tolist(), row
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                search, factor = rt.prepare(image)
                saved_search = search.clone()
                tokens = rt.prefix(search)
                layer = rt.selected_layer(rt.selector(tokens))
                teacher = rt.candidate(tokens, layer)
                student = forward_student(rt, net, search)
                if not torch.equal(search, saved_search):
                    raise ValueError("Frozen models changed the shared search tensor")
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
                size, offset = channels(arm)
                dense = dense_boxes(data["size"][row, size], data["offset"][row, offset], previous,
                    factor, plan["search_size"], image.shape[:2])
                cell = int(np.argmax(data["raw"][row, 0] * data["window"]))
                data["decode_error"][row] = np.max(np.abs(dense[cell] - box))
                data["previous"][row], data["boxes"][row] = previous, box
                data["factor"][row], data["image_hw"][row], data["layer"][row] = factor, image.shape[:2], layer
                previous = box.copy()
                rt.tracker.frame_id = row + 1
                if (row + 1) % 250 == 0:
                    print("[{}] {} {}/{}".format(method, entry["sequence"], row + 1, n), flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    annotation = Path(plan["root"]) / entry["annotation"]
    verify_bindings({str(annotation): entry["annotation_sha256"]})
    gt = read_boxes(annotation)
    if len(gt) != len(entry["frames"]):
        raise ValueError("Label coverage changed")
    data.update(gt=gt[1:n + 1], iou=tracking_iou(data["boxes"], gt[1:n + 1]), model_head_calls=np.asarray(calls))
    audit_trace(data, plan, entry, method, limit)
    endpoint_parity(data, plan, entry, method)
    return data


def smoke_gate(root, plan):
    root = Path(root)
    run = read_sealed(root / "smoke/run.json", SCHEMA)
    report = read_sealed(root / "smoke/stage38_smoke.json", SCHEMA)
    if (run["mode"] != "smoke" or run["methods"] != methods("smoke") or run["plan_fingerprint"] != plan["fingerprint"] or
            report["run_fingerprint"] != run["fingerprint"] or report["status"] != "PASS_FROZEN_SIZE_OFFSET_SMOKE"):
        raise ValueError("Complete the unchanged Stage-38 smoke before full collection")
    entry = plan["fit"][0]
    expected = {entry["sequence"] + "__" + m + ".npz" for m in methods("smoke")}
    if set(report["cache_sha256"]) != expected:
        raise ValueError("Incomplete size/offset smoke coverage")
    verify_bindings({str(root / "smoke/traces" / n): d for n, d in report["cache_sha256"].items()})
    for method in methods("smoke"):
        load_trace(root / "smoke/traces" / (entry["sequence"] + "__" + method + ".npz"),
            plan, run, entry, method, DESIGN["smoke_frames"])


def collect(rt, plan, root, mode, sequence=None, image_reader=checked_image):
    from stage20_compression_common import state_digest
    validate_runtime(rt, plan)
    if mode == "full":
        smoke_gate(root, plan)
    models = load_models(rt, dict(plan, fingerprint=plan["source_model_plan_fingerprint"]), plan["source_input"])
    frozen = {"sgla": state_digest(rt.net)}
    frozen.update({"{}:{}".format(*k): state_digest(net) for k, net in sorted(models.items())})
    window = cpu_array(rt.tracker.output_window).reshape(rt.search_side, rt.search_side).tolist()
    if rt.identity != plan["source_runtime"] or frozen != plan["source_frozen_neural_model_sha256"] or window != plan["window"]:
        raise ValueError("Runtime, models or window differ from frozen Stage-37")
    run = seal(dict(schema=SCHEMA, mode=mode, plan_fingerprint=plan["fingerprint"], runtime=rt.identity,
        methods=methods(mode), window=window, frozen_neural_model_sha256=frozen,
        gt_use="Own predicted parents; future GT only offline/post-rollout",
        timing_role="Both frozen models run for decoder intervention; no compact-model speed claim"))
    directory = Path(root) / mode
    entries = plan["fit"][:1] if mode == "smoke" else [e for e in plan["fit"] if sequence is None or e["sequence"] == sequence]
    if not entries or (mode == "smoke" and sequence is not None):
        raise ValueError("Smoke uses the fixed first fit video; full --sequence must belong to the original eight")
    with writer_lock(directory):
        if not (directory / "run.json").exists() and any(directory.rglob("*.npz")):
            raise ValueError("Unidentified Stage-38 caches")
        publish_identical(directory / "run.json", run)
        hashes, parity = {}, []
        for entry in entries:
            fold = fold_for(plan, entry)
            ordinal = next(i for i, e in enumerate(plan["fit"]) if e["sequence"] == entry["sequence"])
            order = methods(mode)
            order = order[ordinal % len(order):] + order[:ordinal % len(order)]
            for method in order:
                _, seed = arm_seed(method)
                path = directory / "traces" / (entry["sequence"] + "__" + method + ".npz")
                limit = DESIGN["smoke_frames"] if mode == "smoke" else None
                if not path.exists():
                    if path.with_suffix(".json").exists():
                        raise ValueError("Committed trace is missing")
                    data = collect_trace(rt, plan, entry, method, models[(fold, "supervised_seed" + str(seed))], limit, image_reader)
                    write_cache(path, data, metadata(plan, run, entry, method))
                    action = "done"
                else:
                    action = "reuse, audited"
                data = load_trace(path, plan, run, entry, method, limit, allow_orphan=True)
                hashes[path.name] = file_hash(path)
                result = endpoint_parity(data, plan, entry, method)
                if result is not None:
                    parity.append(dict(method=method, **result))
                print("[{}] {}: {} frames".format(action, path.name, len(data["iou"])), flush=True)
        if state_digest(rt.net) != frozen["sgla"] or any(state_digest(net) != frozen["{}:{}".format(*k)] for k, net in models.items()):
            raise ValueError("A frozen model changed")
        verify_bindings(plan["source_file_sha256"])
        if mode == "smoke":
            publish_identical(directory / "stage38_smoke.json", seal(dict(schema=SCHEMA,
                status="PASS_FROZEN_SIZE_OFFSET_SMOKE", run_fingerprint=run["fingerprint"], sequence=entries[0]["sequence"],
                methods=methods(mode), cache_sha256=hashes, endpoint_parity=parity, training_performed=False,
                warning="Execution and endpoint parity only; no learned geometry gain established")))
            print("Stage-38: PASS_FROZEN_SIZE_OFFSET_SMOKE")
    return run


def status(root, plan):
    for mode, entries in (("smoke", plan["fit"][:1]), ("full", plan["fit"])):
        count = 0
        directory = Path(root) / mode
        if (directory / "run.json").exists():
            run = read_sealed(directory / "run.json", SCHEMA)
            if run["plan_fingerprint"] != plan["fingerprint"] or run["mode"] != mode or run["methods"] != methods(mode):
                raise ValueError("Mixed Stage-38 progress")
            for entry in entries:
                for method in methods(mode):
                    path = directory / "traces" / (entry["sequence"] + "__" + method + ".npz")
                    if path.exists() and path.with_suffix(".json").exists():
                        load_trace(path, plan, run, entry, method, DESIGN["smoke_frames"] if mode == "smoke" else None)
                        count += 1
                    elif path.with_suffix(".json").exists():
                        raise ValueError("Committed trace is missing")
        print("{}: {}/{} committed new traces".format(mode, count, len(entries) * len(methods(mode))))
    print("Full controls are reused from frozen Stage-37; unfinished current trace restarts at frame 1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True, help="Completed Stage-37 root for prepare; Stage-38 root otherwise")
    parser.add_argument("--mode", choices=("prepare", "smoke", "evaluate", "status"), required=True)
    parser.add_argument("--output_dir", type=nonempty_path)
    parser.add_argument("--checkpoint", type=nonempty_path)
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
        parser.error("Use the frozen original SGLA checkpoint")
    import torch
    from stage4_runtime import Runtime
    seed_runtime(torch, plan["runtime_seed"])
    collect(Runtime(args.parameter, args.gpu, checkpoint), plan, root,
        "smoke" if args.mode == "smoke" else "full", args.sequence)


if __name__ == "__main__":
    main()
