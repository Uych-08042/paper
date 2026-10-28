"""Collect fixed, matched parent perturbations; no new model training."""

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
from stage35_preserve_runtime import forward_student, make_student
from stage36_same_crop_common import load_models
from stage37_crossed_common import fold_for
from stage39_perturb_common import (DESIGN, MODELS, PERTURBATIONS, SCHEMA, audit, load_branch,
    load_plan, metadata, perturb, prepare, read_sealed, source_trace, task_path, tasks, verify_bindings)


def run_branch(rt, plan, entry, anchor, images, initial_image, parent, model, net):
    """No future labels or reference outputs enter the autonomous controller."""
    torch, side, horizon = rt.torch, rt.search_side, len(images)
    data = dict(previous=np.empty((horizon, 4)), boxes=np.empty((horizon, 4)),
        raw=np.empty((horizon, side, side), np.float32), size=np.empty((horizon, 2, side, side), np.float32),
        offset=np.empty((horizon, 2, side, side), np.float32), factor=np.empty(horizon),
        layer=np.zeros(horizon, np.int64), decode_error=np.empty(horizon))
    calls = [0]
    head = rt.net.box_head if model == "sgla" else net.box_head
    hook = head.register_forward_hook(lambda *unused: calls.__setitem__(0, calls[0] + 1))
    parity = 0.
    try:
        with torch.no_grad():
            rt.initialize(initial_image, entry["initial_box"])
            previous = np.asarray(parent, float).copy()
            for t, image in enumerate(images):
                rt.tracker.state, rt.tracker.frame_id = previous.tolist(), anchor + t - 1
                search, factor = rt.prepare(image)
                saved = search.clone()
                if model == "sgla":
                    tokens = rt.prefix(search)
                    layer = rt.selected_layer(rt.selector(tokens))
                    output = rt.candidate(tokens, layer)
                    data["layer"][t] = layer
                else:
                    output = forward_student(rt, net, search)
                    if model == "fast_initial" and t == 0:
                        original_fast = rt.head(rt.prefix(search))
                        parity = max(float((output[k] - original_fast[k]).abs().max())
                            for k in ("score_map", "size_map", "offset_map"))
                        if parity > 1e-6:
                            raise ValueError("Six-block compact model differs from original Fast prefix")
                if not torch.equal(saved, search):
                    raise ValueError("Model modified its crop tensor")
                box = np.asarray(rt.decode(output, factor, image), float)
                np.testing.assert_array_equal(rt.tracker.state, previous)
                data["previous"][t], data["boxes"][t], data["factor"][t] = previous, box, factor
                for key, source in (("raw", "score_map"), ("size", "size_map"), ("offset", "offset_map")):
                    data[key][t] = cpu_array(output[source]).reshape(data[key].shape[1:])
                dense = dense_boxes(data["size"][t], data["offset"][t], previous,
                    factor, plan["search_size"], image.shape[:2])
                cell = int(np.argmax(data["raw"][t] * cpu_array(rt.tracker.output_window).reshape(side, side)))
                data["decode_error"][t] = np.max(np.abs(dense[cell] - box))
                previous = box.copy()
    finally:
        hook.remove()
    return data, calls[0], parity


def collect_pack(rt, plan, entry, anchor, models, reference, horizon, image_reader=checked_image):
    initial = image_reader(plan["root"], entry["frames"][0])
    images = [image_reader(plan["root"], entry["frames"][i]) for i in range(anchor, anchor + horizon)]
    packet, counts = {}, []
    parity = np.zeros(len(PERTURBATIONS))
    fold = fold_for(plan, entry)
    parent = reference["previous"][anchor - 1]
    for model in MODELS:
        net = None if model == "sgla" else models["fast_initial"] if model == "fast_initial" else models[(fold, model)]
        branches, calls = [], 0
        for p, name in enumerate(PERTURBATIONS):
            branch, used, error = run_branch(rt, plan, entry, anchor, images, initial,
                perturb(parent, name), model, net)
            branches.append(branch)
            calls += used
            if model == "fast_initial":
                parity[p] = error
        for key in branches[0]:
            packet.setdefault(key, []).append(np.stack([b[key] for b in branches]))
        counts.append(calls)
    data = {k: np.stack(v) for k, v in packet.items()}
    # This pack's evaluation annotations are read after all 28 autonomous branches.
    # Earlier source-integrity checks may read old labels; no future label enters run_branch.
    annotation = Path(plan["root"]) / entry["annotation"]
    verify_bindings({str(annotation): entry["annotation_sha256"]})
    gt = read_boxes(annotation)[anchor:anchor + horizon]
    data.update(gt=gt, iou=np.stack([np.stack([tracking_iou(b, gt) for b in row]) for row in data["boxes"]]),
        image_hw=np.asarray([im.shape[:2] for im in images], np.int64), head_calls=np.asarray(counts),
        fast_parity_error=parity, window=cpu_array(rt.tracker.output_window).reshape(rt.search_side, rt.search_side).astype(np.float32))
    audit(data, plan, entry, anchor, horizon, reference)
    return data


def frozen_models(rt, plan):
    models = load_models(rt, dict(plan, fingerprint=plan["source_model_plan_fingerprint"]), plan["source_input"])
    hashes = {"sgla": state_digest(rt.net)}
    hashes.update({"{}:{}".format(*k): state_digest(net) for k, net in sorted(models.items())})
    window = cpu_array(rt.tracker.output_window).reshape(rt.search_side, rt.search_side).tolist()
    if (rt.identity != plan["source_runtime"] or hashes != plan["source_frozen_neural_model_sha256"] or
            window != plan["window"]):
        raise ValueError("Runtime/models/window differ from frozen source")
    fast = make_student(rt).eval().requires_grad_(False)
    original = rt.net.state_dict()
    if len(fast.backbone.blocks) != 6 or hasattr(fast.backbone, "MLP") or not all(
            k in original and rt.torch.equal(value, original[k]) for k, value in fast.state_dict().items()):
        raise ValueError("Fast must be exact weight-preserving surgical truncation")
    models["fast_initial"] = fast
    hashes["fast_initial"] = state_digest(fast)
    return models, hashes


def run_identity(rt, plan, hashes, mode):
    return seal(dict(schema=SCHEMA, mode=mode, plan_fingerprint=plan["fingerprint"], runtime=rt.identity,
        frozen_neural_model_sha256=hashes, window=plan["window"], models=list(MODELS), perturbations=list(PERTURBATIONS),
        gt_use="Original initialization only; common start is a frozen prediction; future labels evaluated after branches",
        timing_role="Instrumented branch sensitivity audit; no latency, energy or speed claim"))


def validate_run(run, plan, mode):
    expected_hashes = dict(plan["source_frozen_neural_model_sha256"])
    # The newly constructed exact truncated model is sealed in smoke, then matched in full.
    if (run.get("mode") != mode or run.get("plan_fingerprint") != plan["fingerprint"] or
            run.get("runtime") != plan["source_runtime"] or run.get("window") != plan["window"] or
            run.get("models") != list(MODELS) or run.get("perturbations") != list(PERTURBATIONS) or
            set(run.get("frozen_neural_model_sha256", {})) != set(expected_hashes) | {"fast_initial"} or
            any(run["frozen_neural_model_sha256"].get(k) != v for k, v in expected_hashes.items())):
        raise ValueError("Mixed Stage-39 runtime/run")


def smoke_gate(root, plan):
    root = Path(root)
    run = read_sealed(root / "smoke/run.json", SCHEMA)
    validate_run(run, plan, "smoke")
    report = read_sealed(root / "smoke/stage39_smoke.json", SCHEMA)
    entry, anchor = tasks(plan, "smoke")[0]
    path = task_path(root, "smoke", entry, anchor)
    if (report.get("status") != "PASS_MATCHED_PARENT_SMOKE" or report.get("run_fingerprint") != run["fingerprint"] or
            report.get("cache_sha256") != {path.name: file_hash(path)}):
        raise ValueError("Need unchanged Stage-39 smoke before full collection")
    load_branch(path, plan, run, entry, anchor, source_trace(plan, entry))
    return run


def collect(rt, plan, root, mode, sequence=None, image_reader=checked_image):
    validate_runtime(rt, plan)
    smoke = smoke_gate(root, plan) if mode == "full" else None
    models, hashes = frozen_models(rt, plan)
    run = run_identity(rt, plan, hashes, mode)
    validate_run(run, plan, mode)
    if smoke and hashes != smoke["frozen_neural_model_sha256"]:
        raise ValueError("Frozen full models differ from smoke")
    selected = [(e, a) for e, a in tasks(plan, mode) if sequence is None or e["sequence"] == sequence]
    if not selected or (mode == "smoke" and sequence is not None):
        raise ValueError("Smoke uses fixed first video/anchor; full sequence must be in the original eight")
    directory = Path(root) / mode
    with writer_lock(directory):
        if not (directory / "run.json").exists() and any(directory.rglob("*.npz")):
            raise ValueError("Unidentified Stage-39 caches")
        publish_identical(directory / "run.json", run)
        references, digests = {}, {}
        for entry, anchor in selected:
            if entry["sequence"] not in references:
                references[entry["sequence"]] = source_trace(plan, entry)
            reference = references[entry["sequence"]]
            path = task_path(root, mode, entry, anchor)
            if not path.exists():
                if path.with_suffix(".json").exists():
                    raise ValueError("Committed branch file missing")
                horizon = DESIGN["smoke_horizon"] if mode == "smoke" else DESIGN["horizon"]
                data = collect_pack(rt, plan, entry, anchor, models, reference, horizon, image_reader)
                write_cache(path, data, metadata(plan, run, entry, anchor))
                action = "done"
            else:
                action = "reuse, audited"
            load_branch(path, plan, run, entry, anchor, reference, allow_orphan=True)
            digests[path.name] = file_hash(path)
            print("[{}] {} (28 branches)".format(action, path.name), flush=True)
        after = {"sgla": state_digest(rt.net), "fast_initial": state_digest(models["fast_initial"])}
        after.update({"{}:{}".format(*k): state_digest(net) for k, net in models.items() if isinstance(k, tuple)})
        if after != hashes:
            raise ValueError("Frozen model changed during branch evaluation")
        verify_bindings(plan["source_file_sha256"])
        if mode == "smoke":
            report = seal(dict(schema=SCHEMA, status="PASS_MATCHED_PARENT_SMOKE", run_fingerprint=run["fingerprint"],
                cache_sha256=digests, branches=28, horizon=DESIGN["smoke_horizon"], training_performed=False,
                warning="Execution, crop continuity, surgical weight identity and native replay only; no mechanism gain"))
            publish_identical(directory / "stage39_smoke.json", report)
            print("Stage-39: " + report["status"])
    return run


def status(root, plan):
    for mode in ("smoke", "full"):
        complete = 0
        path = Path(root) / mode / "run.json"
        if path.exists():
            run = read_sealed(path, SCHEMA)
            validate_run(run, plan, mode)
            references = {}
            for entry, anchor in tasks(plan, mode):
                path = task_path(root, mode, entry, anchor)
                if path.with_suffix(".json").exists():
                    if not path.exists():
                        raise ValueError("Committed branch file missing")
                    if entry["sequence"] not in references:
                        references[entry["sequence"]] = source_trace(plan, entry)
                    load_branch(path, plan, run, entry, anchor, references[entry["sequence"]])
                    complete += 1
        print("{}: {}/{} committed anchor packs; unfinished pack restarts all 28 branches".format(mode, complete, len(tasks(plan, mode))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True, help="Stage-38 root for prepare; Stage-39 otherwise")
    parser.add_argument("--mode", choices=("prepare", "smoke", "evaluate", "status"), required=True)
    parser.add_argument("--output_dir", type=nonempty_path)
    parser.add_argument("--checkpoint", type=nonempty_path)
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sequence")
    args = parser.parse_args()
    root = Path(args.input).expanduser().resolve()
    if not (root / "manifest.json").is_file():
        stage = "Stage-38" if args.mode == "prepare" else "Stage-39"
        variable = "STAGE38_ROOT" if args.mode == "prepare" else "STAGE39_ROOT"
        relative = "stage38/fixed_response_geometry_v1" if args.mode == "prepare" else "stage39/matched_parent_perturbation_v1"
        expected = Path(__file__).resolve().parents[1] / "output" / relative
        parser.error("{} manifest not found: {}. --input must be the result directory containing manifest.json. "
            "Check {} (an already-set environment variable may retain an old path). "
            "Default result location for this checkout: {}".format(stage, root / "manifest.json", variable, expected))
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
        parser.error("Use original frozen SGLA checkpoint")
    import torch
    from stage4_runtime import Runtime
    seed_runtime(torch, plan["runtime_seed"])
    collect(Runtime(args.parameter, args.gpu, checkpoint), plan, root,
        "smoke" if args.mode == "smoke" else "full", args.sequence)


if __name__ == "__main__":
    main()
