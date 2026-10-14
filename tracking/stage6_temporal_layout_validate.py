"""Run true closed-loop equal-cost temporal placement ablations."""

import argparse
from pathlib import Path

import stage1_validate as stage1
from stage6_switched_validate import _run_sequence
from stage6_temporal_layout import (
    LAYOUT_POLICIES,
    active_policy,
    expected_candidate_blocks,
    split_frames,
)


torch = stage1.torch
get_dataset = stage1.get_dataset
Tracker = stage1.Tracker


def _assignment(sequence, layout, early_fraction):
    tracked_frames = len(sequence.frames) - 1
    early_frames, late_frames = split_frames(tracked_frames, early_fraction)
    return {
        "dataset": sequence.dataset,
        "sequence": sequence.name,
        "early_frames": early_frames,
        "late_frames": late_frames,
        "router_policy": layout,
        "router_candidate_blocks": expected_candidate_blocks(
            tracked_frames, early_frames, layout
        ),
        "fold": -1,
    }


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracker_name", nargs="?", default="sglatrack")
    parser.add_argument("tracker_param", nargs="?", default="deit_distilled")
    parser.add_argument("--dataset_name", default="uav123")
    parser.add_argument("--layout", choices=sorted(LAYOUT_POLICIES), required=True)
    parser.add_argument("--early_fraction", type=float, default=0.5)
    parser.add_argument("--runid", type=int, default=None)
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
    args = _build_parser().parse_args()
    if args.num_shards < 1:
        raise ValueError("--num_shards must be at least 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard_id must be in [0, num_shards)")
    if args.latency_warmup_frames < 0:
        raise ValueError("--latency_warmup_frames must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the original SGLATrack tracker.")

    torch.cuda.set_device(args.gpu)
    dataset = get_dataset(args.dataset_name)
    sequence_key = stage1._parse_sequence_key(args.sequence)
    if sequence_key is not None:
        sequences = [dataset[sequence_key]]
    else:
        sequences = list(dataset)[args.shard_id :: args.num_shards]

    tracker_info = Tracker(
        args.tracker_name,
        args.tracker_param,
        args.dataset_name,
        args.runid,
    )
    params = tracker_info.get_parameters()
    params.debug = 0
    tracker = tracker_info.create_tracker(params)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    assignments = {
        (sequence.dataset, sequence.name): _assignment(
            sequence, args.layout, args.early_fraction
        )
        for sequence in sequences
    }
    total_frames = sum(
        assignment["early_frames"] + assignment["late_frames"]
        for assignment in assignments.values()
    )
    total_blocks = sum(
        assignment["router_candidate_blocks"]
        for assignment in assignments.values()
    )

    print("Checkpoint: {}".format(params.checkpoint))
    print("Temporal-layout output directory: {}".format(output_dir))
    print(
        "Layout: {} (first {:.0%} / remaining {:.0%})".format(
            args.layout, args.early_fraction, 1.0 - args.early_fraction
        )
    )
    print(
        "Expected cost: {} blocks / {} tracked frames ({:.4%})".format(
            total_blocks, total_frames, total_blocks / total_frames
        )
    )
    print(
        "Shard {}/{}: {} sequence(s)".format(
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
                args.layout,
                output_dir,
                args.overwrite,
                args.verify_sgla,
                args.latency_warmup_frames,
                policy_resolver=active_policy,
            )
        )
    print("Newly completed temporal-layout sequences: {}".format(completed))
    print(
        "Next: python tracking/stage6_switched_compare.py --reference "
        "<late-p10-dir> --controls \"{}\" --skip_prefix_check "
        "--output_dir <comparison-dir>".format(output_dir)
    )


if __name__ == "__main__":
    main()
