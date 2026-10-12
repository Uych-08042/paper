"""Measure synchronized component and integrated inference latency (Stage-4A)."""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from stage4_common import write_csv, write_json
from stage4_runtime import Runtime, add_runtime_arguments


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row["scope"], row["operation"])].append(row["latency_ms"])
    return [{"scope": scope, "operation": op, "samples": len(times),
             "mean_ms": float(np.mean(times)), "p50_ms": float(np.median(times)),
             "p95_ms": float(np.percentile(times, 95))}
            for (scope, op), times in sorted(groups.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--dataset_name", default="uav123")
    parser.add_argument("--sequence", default="0")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(args.frames, args.repeats, args.warmup) < 1:
        raise ValueError("frames/repeats/warmup must be positive")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.glob("benchmark_*")) and not args.overwrite:
        raise FileExistsError("Use a new output_dir or --overwrite")
    rt = Runtime(args.parameter, args.gpu, args.checkpoint)
    sequence = rt.s1.get_dataset(args.dataset_name)[rt.s1._parse_sequence_key(args.sequence)]
    if len(sequence.frames) < 2:
        raise ValueError("Need two or more video frames")
    rt.initialize(rt.read(sequence.frames[0]), sequence.init_info()["init_bbox"])
    modes = ("fast", "sgla", "fast_then_sgla")
    records = []

    def measure(fn):
        rt.torch.cuda.synchronize()
        start = time.perf_counter()
        value = fn()
        rt.torch.cuda.synchronize()
        return value, (time.perf_counter() - start) * 1000.0

    with rt.torch.no_grad():
        first = rt.read(sequence.frames[1])
        search, factor = rt.prepare(first)
        tokens = rt.prefix(search)
        fast = rt.head(tokens)
        # Warm both whole paths and each component; all updates stay outside timing.
        for _ in range(args.warmup):
            for mode in modes:
                rt.run_image(first, mode)
            state, frame = list(rt.tracker.state), rt.tracker.frame_id
            rt.tracker.track(first)
            rt.tracker.state, rt.tracker.frame_id = state, frame
            rt.pool(tokens)
            for layer in range(7, 13):
                rt.candidate(tokens, layer)
            rt.decode(fast, factor, first)
        rt.verify(search, rt.candidate(tokens, rt.selected_layer(rt.selector(tokens))))

        for frame_id, path in enumerate(sequence.frames[1:args.frames + 1], start=1):
            image = rt.read(path)
            search, factor = rt.prepare(image)
            tokens = rt.prefix(search)
            fast = rt.head(tokens)
            operations = [
                ("component", "crop_preprocess", lambda: rt.prepare(image)),
                ("component", "prefix_L1_L6", lambda: rt.prefix(search)),
                ("component", "norm_fast_head", lambda: rt.head(tokens)),
                ("component", "selector", lambda: rt.selector(tokens)),
                ("component", "pool_all_channels_grid4", lambda: rt.pool(tokens)),
                ("component", "decode_fast", lambda: rt.decode(fast, factor, image)),
            ]
            for layer in range(7, 13):
                operations.append(("component", "block_L{}".format(layer),
                                   lambda layer=layer: rt.backbone.blocks[layer - 1](tokens.clone())))
            operations.extend(("integrated", mode, lambda mode=mode: rt.run_image(image, mode)) for mode in modes)
            operations.append(("integrated", "native_sgla", lambda: rt.tracker.track(image)))
            for repeat in range(args.repeats):
                shift = (frame_id + repeat) % len(operations)
                for scope, operation, fn in operations[shift:] + operations[:shift]:
                    state, frame = list(rt.tracker.state), rt.tracker.frame_id
                    _, elapsed = measure(fn)
                    rt.tracker.state, rt.tracker.frame_id = state, frame
                    records.append({"sequence": sequence.name, "frame_id": frame_id,
                                    "repeat": repeat, "scope": scope, "operation": operation,
                                    "latency_ms": elapsed})
            # Every measured path uses the same SGLA state, isolating compute cost.
            rt.tracker.state = rt.run_image(image, "sgla")
            rt.tracker.frame_id = frame_id
            if frame_id % 10 == 0:
                print("[profile] {}/{}".format(frame_id, min(args.frames, len(sequence.frames) - 1)), flush=True)
    report = summarize(records)
    write_csv(output / "benchmark_raw.csv", records)
    write_csv(output / "benchmark_summary.csv", report)
    write_json(output / "benchmark_metadata.json", {
        "runtime": rt.identity, "dataset": args.dataset_name, "sequence": sequence.name,
        "warmup": args.warmup, "frames": frame_id, "repeats": args.repeats,
        "timing": "CUDA-synchronized host wall time; integrated includes crop/preprocess/decode, excludes disk IO. native_sgla includes its internal state update; restoring the common state is untimed.",
        "warning": "Component times include individual synchronization; do not sum them as integrated latency. No router is measured yet.",
    })
    print("scope       operation                       mean_ms    p50_ms    p95_ms")
    for row in report:
        print("{scope:<11} {operation:<30} {mean_ms:>9.3f} {p50_ms:>9.3f} {p95_ms:>9.3f}".format(**row))
    print("Benchmark reports: {}".format(output))


if __name__ == "__main__":
    main()
