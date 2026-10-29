"""Native SGLA camera-search pilot: prepare, smoke, evaluate and resume."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from stage27_uav_collect import writer_lock
from stage27_uav_common import nonempty_path, publish_identical
import stage46_motion_common as c


def prepare(source, root):
    root = Path(root).expanduser().resolve()
    source = c.resolve_source(source)
    if root == source or source in root.parents or root in source.parents:
        raise ValueError("Use a separate new Stage-46 directory")
    plan, _ = c.make_plan(source)
    with writer_lock(root):
        if not (root/"manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-46 output directory")
        publish_identical(root/"manifest.json", plan)
    print("Stage-46: {} consumed development videos / {} tracked frames / four native-SGLA policies".format(
        len(plan["sequences"]), plan["tracked_frames"]))
    print("Plan: " + plan["fingerprint"])
    print("No training; no new data download. Next: --mode smoke")


def make_runtime(args, source_run):
    import cv2
    import torch
    from stage4_runtime import Runtime
    from stage46_motion_runtime import NativeObserver
    np.random.seed(c.DESIGN["seed"])
    torch.manual_seed(c.DESIGN["seed"])
    torch.cuda.manual_seed_all(c.DESIGN["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    cv2.setNumThreads(1)
    rt = Runtime(args.parameter, args.gpu, args.checkpoint)
    if rt.identity != source_run["runtime"]:
        raise ValueError("Need the original Stage-27 SGLA checkpoint/config/runtime; no students or new weights")
    observer = NativeObserver(rt)
    identity = dict(tracker=rt.identity, opencv=cv2.__version__, numpy=np.__version__,
        gpu_uuid=str(getattr(torch.cuda.get_device_properties(args.gpu), "uuid", "unavailable")),
        model_state_sha256=observer.digest)
    return observer, identity


def execute(plan, inputs, root, observer, identity, smoke=False, sequence=None):
    from stage46_motion_runtime import run_video
    _, source, traces, _ = inputs
    run = c.seal(dict(schema=c.SCHEMA, role="frozen_native_sgla_four_policy_motion_run",
        plan_fingerprint=plan["fingerprint"], runtime=identity, methods=list(c.METHODS)))
    if not smoke:
        record = c.load_sealed(root/"smoke/stage46_smoke_decision.json")
        if record["plan_fingerprint"] != plan["fingerprint"] or record["runtime"] != identity or record["status"] != "PASS_NATIVE_SGLA_MOTION_SMOKE":
            raise ValueError("Need a passing smoke from this exact plan/runtime")
    publish_identical(root/"run.json", run)
    names = plan["sequences"][:1] if smoke else plan["sequences"]
    if sequence:
        names = [n for n in names if n == sequence]
        if not names:
            raise ValueError("Sequence is not in the fixed selection")
    maximum = 0.
    for ordinal, name in enumerate(names):
        entry, native = traces[name]
        frames = entry["frames"][:c.DESIGN["smoke_frames"]+1] if smoke else entry["frames"]
        methods = list(c.METHODS)
        methods = methods[ordinal % 4:] + methods[:ordinal % 4]
        for method in methods:
            path = c.cache_path(root, name, method, smoke)
            meta = c.metadata(run, name, method, smoke)
            data = c.load_result(path, meta)
            if data is None:
                data = run_video(observer, source["manifest"]["root"], frames, entry["initial_box"], name, method)
                c.audit_video(data, frames, entry["initial_box"], method, native)
                c.save_result(path, data, meta)
            else:
                c.audit_video(data, frames, entry["initial_box"], method, native)
            if method == "sgla":
                maximum = max(maximum, float(np.max(np.abs(data["boxes"]-native["boxes"][:len(data["boxes"])]))))
            print("[done, audited]", name, method, len(data["boxes"]), flush=True)
    if smoke:
        decision = c.seal(dict(schema=c.SCHEMA, status="PASS_NATIVE_SGLA_MOTION_SMOKE",
            role="execution_only_not_method_gain", plan_fingerprint=plan["fingerprint"], runtime=identity,
            sequence=names[0], frames=len(frames)-1, methods=list(c.METHODS),
            max_native_box_error=maximum, exact_seven_blocks_and_one_head=True,
            no_gt_actor_inputs=True, weights_unchanged=True, training_performed=False,
            warning="Execution, continuity and native replay only; synthetic tests cover nonzero camera motion."))
        publish_identical(root/"smoke/stage46_smoke_decision.json", decision)
        print("Stage-46 smoke PASS; no tracking gain established")


def status(plan, root):
    run = c.load_sealed(root/"run.json") if (root/"run.json").exists() else None
    if run is not None and run["plan_fingerprint"] != plan["fingerprint"]:
        raise ValueError("Mixed run/plan")
    for method in c.METHODS:
        count = 0
        if run is not None:
            for name in plan["sequences"]:
                if c.load_result(c.cache_path(root,name,method), c.metadata(run,name,method,False)) is not None:
                    count += 1
        print("{}: {}/{} committed videos".format(method, count, len(plan["sequences"])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("prepare","smoke","evaluate","status"), required=True)
    p.add_argument("--input", type=nonempty_path, required=True, help="prepare: Stage-27 cache; otherwise Stage-46 root")
    p.add_argument("--output_dir", type=nonempty_path)
    p.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    p.add_argument("--parameter", default="deit_distilled")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--sequence")
    args = p.parse_args()
    if args.mode == "prepare":
        if not args.output_dir:
            p.error("prepare needs --output_dir")
        prepare(args.input, args.output_dir)
        return
    root = Path(args.input).expanduser().resolve()
    plan, inputs = c.load_plan(root)
    if args.mode == "status":
        status(plan, root)
        return
    if args.checkpoint is not None and not str(args.checkpoint).strip():
        p.error("Empty --checkpoint/CHECKPOINT; set original SGLA weights")
    with writer_lock(root):
        observer, identity = make_runtime(args, inputs[1])
        execute(plan, inputs, root, observer, identity, args.mode == "smoke", args.sequence)


if __name__ == "__main__":
    main()
