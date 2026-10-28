"""Run frozen temporal policies on disjoint manifest clips in true closed loop."""

import argparse
import json
import time
from pathlib import Path

from stage4_common import IOU_PROTOCOL, file_hash, fingerprint, load_manifest, write_json
from stage7_holdout_common import (
    HOLDOUT_POLICIES,
    ManifestSequence,
    assignment_for_policy,
    resolve_policy,
    select_entries,
)


def publish_run(output_dir, run):
    """Prevent mixed manifests, checkpoints or policies across shards."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / "run.lock"
    deadline = time.monotonic() + 30.0
    while True:
        try:
            lock.open("x").close()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "run.lock persists; check for a running validator before removing it"
                )
            time.sleep(0.1)
    try:
        path = output_dir / "run.json"
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            if old != run:
                raise ValueError(
                    "Run configuration changed; use a new output_dir"
                )
        else:
            if any(output_dir.rglob("*_closed_loop.csv")):
                raise ValueError(
                    "Unidentified closed-loop files exist; use an empty output_dir"
                )
            write_json(path, run)
    finally:
        lock.unlink()


def make_run(args, manifest, checkpoint):
    directory = Path(__file__).resolve().parent
    files = (
        "stage7_temporal_holdout_validate.py",
        "stage7_holdout_common.py",
        "stage6_switched_validate.py",
        "stage6_temporal_layout.py",
        "stage3_closed_loop_validate.py",
    )
    run = {
        "schema": "stage7_temporal_holdout_v1",
        "manifest_fingerprint": manifest["fingerprint"],
        "dataset": manifest["dataset"],
        "split": args.split,
        "policy": args.policy,
        "temporal_boundary": "frozen_stage6_default",
        "tracker_name": args.tracker_name,
        "tracker_parameter": args.tracker_param,
        "checkpoint_sha256": file_hash(checkpoint),
        "latency_warmup_frames": args.latency_warmup_frames,
        "iou_protocol": IOU_PROTOCOL,
        "code_sha256": {name: file_hash(directory / name) for name in files},
    }
    run["fingerprint"] = fingerprint(run)
    return run


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracker_name", nargs="?", default="sglatrack")
    parser.add_argument("tracker_param", nargs="?", default="deit_distilled")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=("all", "train", "val"), default="val")
    parser.add_argument("--policy", choices=HOLDOUT_POLICIES, required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify_sgla", action="store_true")
    parser.add_argument("--latency_warmup_frames", type=int, default=20)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    return parser


def main():
    args = build_parser().parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Require 0 <= shard_id < num_shards")
    if args.latency_warmup_frames < 0:
        raise ValueError("--latency_warmup_frames must be non-negative")
    import stage1_validate as stage1
    from stage6_switched_validate import _run_sequence

    torch = stage1.torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the original SGLATrack tracker")

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    entries = select_entries(manifest, args.split, args.sequence)
    entries = entries[args.shard_id :: args.num_shards]
    if not entries:
        print("This shard has no clips.")
        return
    sequences = [ManifestSequence(manifest, entry) for entry in entries]
    assignments = {
        (sequence.dataset, sequence.name): assignment_for_policy(
            sequence, args.policy
        )
        for sequence in sequences
    }

    torch.cuda.set_device(args.gpu)
    tracker_info = stage1.Tracker(
        args.tracker_name, args.tracker_param, manifest["dataset"]
    )
    params = tracker_info.get_parameters()
    params.debug = 0
    if args.checkpoint:
        params.checkpoint = str(Path(args.checkpoint).expanduser().resolve())
    checkpoint = Path(params.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "Checkpoint missing: {} (or pass --checkpoint)".format(checkpoint)
        )
    tracker = tracker_info.create_tracker(params)

    output_dir = Path(args.output_dir).expanduser().resolve()
    publish_run(output_dir, make_run(args, manifest, checkpoint))
    total_frames = sum(len(sequence.frames) - 1 for sequence in sequences)
    total_blocks = sum(
        assignment["router_candidate_blocks"]
        for assignment in assignments.values()
    )
    print("Checkpoint: {}".format(checkpoint))
    print("Manifest: {}".format(manifest_path))
    print("Manifest fingerprint: {}".format(manifest["fingerprint"]))
    print("Stage-7 output directory: {}".format(output_dir))
    print("Frozen policy: {}".format(args.policy))
    print(
        "Expected cost: {} blocks / {} tracked frames ({:.4%})".format(
            total_blocks, total_frames, total_blocks / total_frames
        )
    )
    print(
        "Shard {}/{}: {} clip(s)".format(
            args.shard_id, args.num_shards, len(sequences)
        )
    )

    completed = 0
    for sequence in sequences:
        completed += int(
            _run_sequence(
                tracker_info,
                tracker,
                sequence,
                assignments[(sequence.dataset, sequence.name)],
                args.policy,
                output_dir,
                args.overwrite,
                args.verify_sgla,
                args.latency_warmup_frames,
                policy_resolver=resolve_policy,
            )
        )
    print("Newly completed Stage-7 clips: {}".format(completed))
    print(
        "Next: python tracking/stage6_switched_compare.py --reference "
        "<late-p10-exact-dir> --controls <fast-dir> <uniform-p5-dir> "
        "<sgla-dir> --skip_prefix_check --output_dir <comparison-dir>"
    )


if __name__ == "__main__":
    main()
