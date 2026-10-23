"""Profile frozen tracking paths under real, authorized single-GPU contention.

This is hardware calibration, not a PPO trainer or a tracking-quality result.
Use a fresh output directory. Interrupted raw journals are retained for audit.
"""

import argparse
import contextlib
import importlib.metadata
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint, load_manifest, write_json
from stage4_runtime import Runtime, add_runtime_arguments
from stage22_gpu_hardware import LoadProcess, TelemetrySampler, resolve_gpu
from stage22_resource_common import (
    CONDITIONS, OPERATIONS, RAW_FIELDS, SCHEMA, CsvJournal, calibration_sources,
    check_execution_counts, interleaved_order, load_phase, restore_call, run_operation, sealed,
)


CODE_FILES = ("stage22_device_calibrate.py", "stage22_resource_common.py",
              "stage22_gpu_hardware.py", "stage22_device_summary.py",
              "stage4_runtime.py", "stage1_validate.py", "stage4_common.py")


def verify_paths(rt, image, history):
    """Untimed hooks check actual execution, including the original native path."""
    counts, handles = {}, []
    def hook(name):
        def count(_module, _inputs, _output):
            counts[name] = counts.get(name, 0) + 1
        return count
    for index, block in enumerate(rt.backbone.blocks, 1):
        handles.append(block.register_forward_hook(hook("l%d" % index)))
    handles.append(rt.net.box_head.register_forward_hook(hook("head")))
    report = {}
    try:
        for operation in OPERATIONS:
            counts.clear()
            restore_call(rt, lambda: run_operation(rt, image, operation, history, 1 / 30))
            check_execution_counts(operation, counts)
            report[operation] = dict(counts)
    finally:
        for handle in handles:
            handle.remove()
    native = restore_call(rt, lambda: run_operation(rt, image, "sgla", history, 1 / 30))
    scripted = restore_call(rt, lambda: rt.run_image(image, "sgla"))
    if not np.allclose(native, scripted, rtol=0, atol=1e-5):
        raise ValueError("Native SGLA and scripted common-reference boxes disagree")
    report["native_scripted_bbox_max_abs_diff"] = float(np.max(np.abs(np.asarray(native) - scripted)))
    return report


def profile_clip(rt, source, config, condition, origin_ns, sampler, worker, journal, run_id):
    images = []
    for item in source["images"]:
        if file_hash(item["path"]) != item["sha256"]:
            raise ValueError("Source image changed during profiling: " + item["path"])
        images.append(rt.read(item["path"]))
    rt.initialize(images[0], source["init_box"])
    history = [(0., list(source["init_box"]))]
    verification = verify_paths(rt, images[1], history)
    for _ in range(config["warmup"]):
        for operation in OPERATIONS:
            restore_call(rt, lambda op=operation: run_operation(rt, images[1], op, history, 1 / 30))
    rt.torch.cuda.synchronize()
    for offset, image in enumerate(images[1:], 1):
        if worker is not None:
            worker.check()
        frame_id = source["start"] + offset
        state_id = fingerprint(dict(box=list(rt.tracker.state), frame_id=rt.tracker.frame_id))
        capture_time_s = offset / 30
        for repeat in range(config["repeats"]):
            order = interleaved_order(config["seed"], condition, source["sequence"], frame_id, repeat)
            for index, operation in enumerate(order):
                # Restoring the reference crop and pre-sync are outside timing.
                state, tracker_frame = list(rt.tracker.state), rt.tracker.frame_id
                if operation != "predict_only":
                    rt.torch.cuda.synchronize()
                lookup_start = time.monotonic_ns()
                sample = sampler.latest(lookup_start)
                lookup_end = time.monotonic_ns()
                start = time.monotonic_ns()
                try:
                    box = run_operation(rt, image, operation, history, capture_time_s)
                    if operation != "predict_only":
                        rt.torch.cuda.synchronize()
                    end = time.monotonic_ns()
                finally:
                    rt.tracker.state, rt.tracker.frame_id = state, tracker_frame
                if len(box) != 4 or not np.isfinite(box).all() or min(box[2:]) <= 0:
                    raise ValueError("Invalid decoded box: " + operation)
                elapsed = (start - origin_ns) / 1e9
                phase, duty = load_phase(condition, elapsed, config)
                journal.append(dict(run_fingerprint=run_id, condition=condition,
                    sequence=source["sequence"], frame_id=frame_id, repeat=repeat, order=index,
                    operation=operation, crop_state_fingerprint=state_id, start_ns=start, end_ns=end,
                    latency_ms=(end - start) / 1e6, condition_elapsed_s=elapsed,
                    phase=phase, nominal_duty=duty,
                    timing_basis="cpu_wall" if operation == "predict_only" else "cuda_synchronized_host_wall",
                    telemetry_sample_id=sample["sample_id"] if sample else "",
                    telemetry_end_ns=sample["end_ns"] if sample else "",
                    telemetry_age_ms=(start - sample["end_ns"]) / 1e6 if sample else "",
                    telemetry_lookup_ms=(lookup_end - lookup_start) / 1e6))
        # One untimed native SGLA call advances the common trajectory, not GT.
        box = rt.tracker.track(image)["target_bbox"]
        rt.torch.cuda.synchronize()
        history = (history + [(capture_time_s, list(box))])[-2:]
        journal.flush()
        if offset % 16 == 0 or offset == config["frames"]:
            print("[calibration] %s %s %d/%d" % (condition, source["sequence"], offset,
                                                config["frames"]), flush=True)
    return verification


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--bundle", required=True, help="Verified Stage-18 split bundle; weights are NOT loaded")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--exclusive_gpu_confirmed", action="store_true",
                        help="Confirm this full GPU is exclusively allocated to you")
    parser.add_argument("--require_telemetry", action="store_true")
    parser.add_argument("--warmup", type=int, default=3, help="Untimed repetitions of every path per clip/condition")
    parser.add_argument("--sustained_duty", type=float, default=.6)
    parser.add_argument("--switching_duty", type=float, default=.85)
    parser.add_argument("--switch_hold_s", type=float, default=1.)
    parser.add_argument("--load_period_ms", type=float, default=20.)
    parser.add_argument("--matrix_size", type=int, default=1024)
    parser.add_argument("--maximum_idle_kernel_ms", type=float, default=5.)
    parser.add_argument("--telemetry_interval_s", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=2070)
    args = parser.parse_args(argv)
    if any(not getattr(args, k) or not getattr(args, k).strip()
           for k in ("bundle", "manifest", "output_dir", "checkpoint")):
        parser.error("--bundle, --manifest, --checkpoint and --output_dir must be nonempty paths")
    numbers = (args.sustained_duty, args.switching_duty, args.switch_hold_s,
               args.load_period_ms, args.maximum_idle_kernel_ms, args.telemetry_interval_s)
    if (not all(np.isfinite(x) for x in numbers) or args.warmup < 1 or args.gpu < 0 or args.seed < 0
            or not 0 < args.sustained_duty <= .95 or not 0 < args.switching_duty <= .95
            or args.switch_hold_s < .1 or not 5 <= args.load_period_ms <= 100
            or not 128 <= args.matrix_size <= 2048 or not .1 <= args.maximum_idle_kernel_ms <= 10
            or not .1 <= args.telemetry_interval_s <= 5 or len(set(args.conditions)) != len(args.conditions)):
        parser.error("Invalid bounded workload/warmup/telemetry configuration")
    if not args.exclusive_gpu_confirmed:
        parser.error("Allocate an exclusive GPU, then pass --exclusive_gpu_confirmed; no worker was launched")
    return args


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Calibration never overwrites/resumes a run; choose a new output_dir: " + str(output))
    from stage18_freeze_representation import verify_representation_bundle
    _, bundle, _ = verify_representation_bundle(args.bundle)
    manifest = load_manifest(args.manifest)
    config = {k: getattr(args, k) for k in (
        "mode", "warmup", "sustained_duty", "switching_duty", "switch_hold_s", "load_period_ms",
        "matrix_size", "maximum_idle_kernel_ms", "telemetry_interval_s", "seed", "require_telemetry")}
    config.update(videos=1 if args.mode == "smoke" else 5,
                  frames=8 if args.mode == "smoke" else 64,
                  repeats=1 if args.mode == "smoke" else 3, deadline_ms=1000 / 30,
                  settle_s=max(2., 2 * args.switch_hold_s + args.load_period_ms / 1000))
    sources = calibration_sources(manifest, bundle, config["frames"], config["videos"])
    rt = Runtime(args.parameter, args.gpu, args.checkpoint)
    rt.torch.manual_seed(args.seed)
    rt.torch.cuda.manual_seed_all(args.seed)
    # The original checkpoint/config are frozen. A new GPU/software stack is a
    # legitimate timing target, so record it instead of requiring old hardware.
    for key in ("parameter", "checkpoint_sha256", "config", "lens_z", "lens_x", "embed_dim"):
        if rt.identity[key] != bundle["runtime"][key]:
            raise ValueError("Frozen original tracker identity changed: " + key)
    hardware = resolve_gpu(rt.torch, args.gpu)
    if hardware["other_compute_pids"]:
        if not hardware["current_process_pid_visible"]:
            raise RuntimeError("GPU UUID was resolved, but nvidia-smi does not show this process's PID aliases. "
                               "Unmatched GPU PIDs may be host-namespace aliases or other jobs; do not ignore/kill them. "
                               "Check the exclusive allocation on the host. Run: "
                               "python tracking/stage22_gpu_hardware.py --gpu %d. PIDs: %s"
                               % (args.gpu, hardware["other_compute_pids"]))
        raise RuntimeError("Other compute processes already use this GPU: " + str(hardware["other_compute_pids"]))
    repository = Path(__file__).resolve().parent.parent
    inference_files = set((repository / "lib/models").rglob("*.py"))
    inference_files.update((repository / "lib/test/tracker").rglob("*.py"))
    inference_files.update(repository / name for name in (
        "lib/train/data/processing_utils.py", "lib/test/evaluation/tracker.py",
        "lib/utils/box_ops.py"))
    packages = {}
    for package in ("numpy", "opencv-python", "timm", "torchvision"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    run = sealed(dict(schema=SCHEMA, config=config, conditions=args.conditions, sources=sources,
        source_bundle_fingerprint=bundle["fingerprint"], source_manifest_fingerprint=manifest["fingerprint"],
        runtime=rt.identity, hardware=hardware, python=platform.python_version(), platform=platform.platform(),
        package_versions=packages,
        inference_code_sha256={p.relative_to(repository).as_posix(): file_hash(p)
                               for p in sorted(inference_files)},
        execution=dict(torch_num_threads=rt.torch.get_num_threads(),
                       cudnn_version=rt.torch.backends.cudnn.version(),
                       cudnn_benchmark=rt.torch.backends.cudnn.benchmark,
                       cudnn_deterministic=rt.torch.backends.cudnn.deterministic,
                       cuda_matmul_allow_tf32=rt.torch.backends.cuda.matmul.allow_tf32),
        started_utc=datetime.now(timezone.utc).isoformat(),
        exclusive_gpu_confirmed=True, cuda_logical_device=args.gpu,
        code_sha256={name: file_hash(Path(__file__).parent / name) for name in CODE_FILES},
        timing="Host wall time includes crop/preprocess/model/decode; excludes disk IO, state restoration and pre-sync. "
               "Visual paths end with own-context CUDA sync; CPU prediction has no CUDA call. "
               "Native SGLA advances the untimed common reference once per frame. "
               "Telemetry lookup is separately timed. Polling/background startup/warmup occur outside timed paths, "
               "but can contend with them.",
        warning="No controller, streaming output, policy AUC or tracker-only energy is measured."))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run.json", run)
    print("Physical GPU: %s (%s)" % (hardware["name"], hardware["uuid"]), flush=True)
    print("Run fingerprint: " + run["fingerprint"], flush=True)
    conditions = []
    try:
        with contextlib.ExitStack() as stack:
            journal = CsvJournal(output / "latency_raw.csv", RAW_FIELDS)
            stack.callback(journal.close)
            sampler = TelemetrySampler(hardware["uuid"], output / "telemetry.csv", args.telemetry_interval_s)
            stack.callback(sampler.close)
            sampler.start()
            if args.require_telemetry:
                sample = sampler.latest(time.monotonic_ns())
                if not sample or sample["utilization.gpu"] == "" or sample["memory.free"] == "":
                    raise RuntimeError("Required GPU utilization/free-memory telemetry is unavailable")
            with rt.torch.no_grad():
                for condition in args.conditions:
                    worker = None
                    if condition != "idle":
                        worker = LoadProcess(config, condition, args.gpu, hardware["uuid"],
                                             output / ("load_%s.csv" % condition))
                    primary_error = None
                    try:
                        origin = worker.start() if worker else time.monotonic_ns()
                        row = dict(condition=condition, origin_ns=origin,
                                   worker=worker.ready if worker else None, execution_checks={})
                        # Let both load phases execute even in the short smoke.
                        # This is untimed apparatus warm-up, never tracker delay.
                        time.sleep(config["settle_s"])
                        if worker:
                            worker.check()
                        for source in sources:
                            checks = profile_clip(rt, source, config, condition, origin, sampler,
                                                  worker, journal, run["fingerprint"])
                            row["execution_checks"][source["sequence"]] = checks
                        if worker:
                            worker.check()
                        row["end_ns"] = time.monotonic_ns()
                        conditions.append(row)
                    except BaseException as error:
                        primary_error = error
                        raise
                    finally:
                        if worker:
                            try:
                                worker.close()
                            except BaseException as cleanup_error:
                                if primary_error is None:
                                    raise
                                write_json(output / ("worker_cleanup_%s.json" % condition),
                                           dict(primary=repr(primary_error), cleanup=repr(cleanup_error)))
            samples = journal.count
        write_json(output / "conditions.json", conditions)
        names = ["latency_raw.csv", "telemetry.csv", "conditions.json"]
        names += ["load_%s.csv" % c for c in args.conditions if c != "idle"]
        completion = sealed(dict(schema=SCHEMA + "_completion", run_fingerprint=run["fingerprint"],
                                 samples=samples, files={n: file_hash(output / n) for n in names}))
        write_json(output / "completion.json", completion)
    except BaseException as error:
        write_json(output / "interrupted.json", dict(error=repr(error), run_fingerprint=run["fingerprint"],
                   warning="Incomplete calibration. Preserve journals; rerun into a NEW directory."))
        raise
    from stage22_device_summary import summarize_run
    summarize_run(output)


if __name__ == "__main__":
    main()
