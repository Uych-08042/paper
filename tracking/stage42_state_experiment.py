"""One-time predicted-parent swaps with frozen Stage-40 models and own continuation."""

import argparse
import os
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_cache
from stage20_compression_common import state_digest
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, nonempty_path, publish_identical, read_boxes, seal, tracking_iou
from stage28_response_collect import validate_runtime
from stage28_response_common import cpu_array
from stage34_identity_experiment import seed_runtime
from stage37_crossed_common import fold_for
from stage39_perturb_experiment import run_branch
from stage42_state_common import (ARMS, DESIGN, SCHEMA, audit_data, load_plan, load_result, metadata, prepare,
    read_sealed, source_traces, task_path, training, verify_bindings)


def collect_unit(rt, plan, entry, task, nets, traces, limit=None, image_reader=checked_image):
    n = min(task["frames"], limit) if limit else task["frames"]
    start = task["frame_id"]
    initial = image_reader(plan["root"], entry["frames"][0])
    images = [image_reader(plan["root"], entry["frames"][i]) for i in range(start, start + n)]
    rows, counts = [], []
    for model in ARMS:
        branches, used = [], []
        for parent in ARMS:
            values, calls, _ = run_branch(rt, plan, entry, start, images, initial,
                traces[parent]["previous"][start - 1], model, nets[model])
            branches.append(values)
            used.append(calls)
        rows.append({k: np.stack([b[k] for b in branches]) for k in branches[0]})
        counts.append(used)
    data = {k: np.stack([row[k] for row in rows]) for k in rows[0]}
    # This unit's evaluation labels enter only after all four autonomous branches.
    # Old source audits may already have read labels; none enter run_branch.
    annotation = Path(plan["root"]) / entry["annotation"]
    verify_bindings({str(annotation): entry["annotation_sha256"]})
    gt = read_boxes(annotation)[start:start + n]
    data.update(gt=gt, iou=tracking_iou(data["boxes"], gt), image_hw=np.asarray([im.shape[:2] for im in images]),
        head_calls=np.asarray(counts), window=cpu_array(rt.tracker.output_window).reshape(rt.search_side, rt.search_side).astype(np.float32))
    audit_data(data, plan, entry, task, traces, limit)
    return data


def validate_run(run, plan, mode):
    if (run.get("mode") != mode or run.get("plan_fingerprint") != plan["fingerprint"] or
            run.get("runtime") != plan["source_runtime"] or run.get("window") != plan["window"] or
            run.get("model_state_sha256") != plan["source_model_states"] or
            run.get("model_file_sha256") != plan["source_model_files"] or
            run.get("teacher_sha256") != plan["source_teacher_sha256"]):
        raise ValueError("Mixed Stage-42 runtime/models/window")


def smoke_gate(root, plan):
    root = Path(root)
    run = read_sealed(root / "smoke/run.json", SCHEMA)
    validate_run(run, plan, "smoke")
    report = read_sealed(root / "smoke/stage42_smoke.json", SCHEMA)
    task = plan["smoke_task"]
    path = task_path(root, "smoke", task)
    if (report.get("status") != "PASS_FROZEN_STATE_SWAP_SMOKE" or report.get("run_fingerprint") != run["fingerprint"] or
            report.get("cache_sha256") != {path.name: file_hash(path)}):
        raise ValueError("Need unchanged frozen state-swap smoke before full evaluation")
    entry = next(e for e in plan["fit"] if e["sequence"] == task["sequence"])
    load_result(path, plan, run, entry, task, source_traces(plan, entry, task["seed"]), DESIGN["smoke_frames"])
    return run


def collect(rt, plan, root, mode, sequence=None, image_reader=checked_image):
    from stage40_crop_experiment import evaluation_models
    validate_runtime(rt, plan)
    if mode == "full":
        smoke_gate(root, plan)
    original = training.load_plan(plan["source_stage40"])
    models, files = evaluation_models(rt, original, Path(plan["source_stage40"]))
    digests = {"{}:{}".format(*k): state_digest(net) for k, net in sorted(models.items())}
    run = seal(dict(schema=SCHEMA, mode=mode, plan_fingerprint=plan["fingerprint"], runtime=rt.identity,
        model_state_sha256=digests, model_file_sha256=files, teacher_sha256=state_digest(rt.net),
        window=cpu_array(rt.tracker.output_window).reshape(rt.search_side, rt.search_side).tolist(),
        intervention="Initial recorded parent only; source template unchanged; own predictions thereafter",
        timing_role="Frozen mechanism intervention; no deployment speed claim"))
    validate_run(run, plan, mode)
    selected = [plan["smoke_task"]] if mode == "smoke" else [t for t in plan["tasks"] if sequence is None or t["sequence"] == sequence]
    if not selected or (mode == "smoke" and sequence is not None):
        raise ValueError("Use fixed smoke task or a full-evaluation sequence from the original eight")
    directory, hashes, parities = Path(root) / mode, {}, []
    references = {}
    with writer_lock(directory):
        if not (directory / "run.json").exists() and any(directory.rglob("*.npz")):
            raise ValueError("Unidentified Stage-42 branch files")
        publish_identical(directory / "run.json", run)
        for task in selected:
            entry = next(e for e in plan["fit"] if e["sequence"] == task["sequence"])
            key = (task["sequence"], task["seed"])
            if key not in references:
                references[key] = source_traces(plan, entry, task["seed"])
            traces = references[key]
            nets = {a: models[(fold_for(plan, entry), a + "_seed" + str(task["seed"]))] for a in ARMS}
            path = task_path(root, mode, task)
            limit = DESIGN["smoke_frames"] if mode == "smoke" else None
            if not path.exists():
                if path.with_suffix(".json").exists():
                    raise ValueError("Committed branch file missing")
                data = collect_unit(rt, plan, entry, task, nets, traces, limit, image_reader)
                write_cache(path, data, metadata(plan, run, task))
                action = "done"
            else:
                action = "reuse, audited"
            data, parity = load_result(path, plan, run, entry, task, traces, limit, allow_orphan=True)
            hashes[path.name] = file_hash(path)
            parities.extend(parity)
            print("[{}] {}: four branches x {} frames".format(action, path.name, len(data["gt"])), flush=True)
        if state_digest(rt.net) != run["teacher_sha256"] or any(state_digest(net) != digests["{}:{}".format(*k)] for k, net in models.items()):
            raise ValueError("Frozen source model changed")
        verify_bindings(plan["source_file_sha256"])
        verify_bindings(files)
        if mode == "smoke":
            report = seal(dict(schema=SCHEMA, status="PASS_FROZEN_STATE_SWAP_SMOKE", run_fingerprint=run["fingerprint"],
                cache_sha256=hashes, native_parity=parities, training_performed=False,
                warning="Four-corner execution/native replay only; no recovery or method gain established"))
            publish_identical(directory / "stage42_smoke.json", report)
            print("Stage-42: " + report["status"])
    return run


def status(root, plan):
    for mode, selected in (("smoke", [plan["smoke_task"]]), ("full", plan["tasks"])):
        complete = 0
        path = Path(root) / mode / "run.json"
        if path.exists():
            run = read_sealed(path, SCHEMA)
            validate_run(run, plan, mode)
            references = {}
            for task in selected:
                path = task_path(root, mode, task)
                if path.with_suffix(".json").exists():
                    entry = next(e for e in plan["fit"] if e["sequence"] == task["sequence"])
                    key = (task["sequence"], task["seed"])
                    if key not in references:
                        references[key] = source_traces(plan, entry, task["seed"])
                    load_result(path, plan, run, entry, task, references[key], DESIGN["smoke_frames"] if mode == "smoke" else None)
                    complete += 1
        print("{}: {}/{} committed four-corner anchor packs".format(mode, complete, len(selected)))
    print("Unfinished pack restarts; completed audited packs are reused")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True, help="Stage-41 output for prepare; Stage-42 otherwise")
    parser.add_argument("--mode", choices=("prepare", "smoke", "evaluate", "status"), required=True)
    parser.add_argument("--output_dir", type=nonempty_path)
    parser.add_argument("--checkpoint", type=nonempty_path)
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sequence")
    args = parser.parse_args()
    if args.sequence is not None and args.mode != "evaluate":
        parser.error("--sequence is only for full evaluate")
    root = Path(args.input).expanduser().resolve()
    expected = root / ("stage41_diagnostic.json" if args.mode == "prepare" else "manifest.json")
    if not expected.is_file():
        parser.error("Input file missing: {}. Reset STAGE41_ROOT / STAGE42_ROOT from ~/workspace/paper.".format(expected))
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
        parser.error("Use the original SGLA checkpoint")
    import torch
    from stage4_runtime import Runtime
    seed_runtime(torch, plan["runtime_seed"])
    collect(Runtime(args.parameter, args.gpu, checkpoint), plan, root, "smoke" if args.mode == "smoke" else "full", args.sequence)


if __name__ == "__main__":
    main()
