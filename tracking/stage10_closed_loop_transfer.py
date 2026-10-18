"""Run the frozen periodic-L9 policy on Stage-9 clips in true closed loop.

Stage-9 outcomes have already been inspected, so this is a descriptive
mechanism-to-trajectory transfer diagnostic rather than a new confirmatory
experiment.  The policy set is deliberately restricted to the Stage-5
periodic-L9-20 operating point and its Fast/SGLA references.
"""

import argparse
import json
import time
from pathlib import Path

from stage3_periodic import periodic_l9_active
from stage4_common import (
    IOU_PROTOCOL,
    file_hash,
    fingerprint,
    load_manifest,
    write_json,
)
from stage7_holdout_common import ManifestSequence, select_entries


TRANSFER_POLICIES = ("fast", "periodic_l9_20", "sgla")
STAGE9_PURPOSE = "stage9_equal_cost_schedule_intervention"


def validate_manifest(manifest):
    """Require the frozen Stage-9 cohort used by the schedule experiment."""
    if manifest.get("purpose") != STAGE9_PURPOSE:
        raise ValueError(
            "Stage-10 requires a frozen Stage-9 schedule manifest; got purpose={!r}".format(
                manifest.get("purpose")
            )
        )
    return manifest


def expected_candidate_blocks(tracked_frames, policy):
    if policy not in TRANSFER_POLICIES:
        raise ValueError("Unknown Stage-10 policy: {}".format(policy))
    if tracked_frames < 1:
        raise ValueError("A transfer clip must contain at least one tracked frame")
    if policy == "fast":
        return 0
    if policy == "sgla":
        return tracked_frames
    return sum(
        int(periodic_l9_active(policy, decision_index))
        for decision_index in range(1, tracked_frames + 1)
    )


def assignment_for_policy(sequence, policy):
    """Build the fixed-policy contract consumed by the closed-loop runner."""
    tracked_frames = len(sequence.frames) - 1
    reporting_boundary = tracked_frames // 2
    return {
        "dataset": sequence.dataset,
        "sequence": sequence.name,
        "early_frames": reporting_boundary,
        "late_frames": tracked_frames - reporting_boundary,
        "router_policy": policy,
        "router_candidate_blocks": expected_candidate_blocks(
            tracked_frames, policy
        ),
        "fold": -1,
    }


def resolve_policy(frame_id, early_frames, policy):
    """Return the frozen policy; the boundary is reporting-only."""
    del frame_id, early_frames
    if policy not in TRANSFER_POLICIES:
        raise ValueError("Unknown Stage-10 policy: {}".format(policy))
    return policy


def publish_run(output_dir, run):
    """Prevent mixed manifests, checkpoints, policies, or code across shards."""
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
                raise ValueError("Run configuration changed; use a new output_dir")
        else:
            if any(output_dir.rglob("*_closed_loop.csv")):
                raise ValueError(
                    "Unidentified closed-loop files exist; use an empty output_dir"
                )
            write_json(path, run)
    finally:
        lock.unlink()


def make_run(args, manifest, checkpoint, expected_sequences):
    directory = Path(__file__).resolve().parent
    files = (
        "stage10_closed_loop_transfer.py",
        "stage7_holdout_common.py",
        "stage6_switched_validate.py",
        "stage3_closed_loop_validate.py",
        "stage3_periodic.py",
    )
    run = {
        "schema": "stage10_closed_loop_transfer_v1",
        "analysis_role": "descriptive_post_stage9_transfer_diagnostic",
        "policy_origin": "frozen_stage5_periodic_l9_20",
        "manifest_fingerprint": manifest["fingerprint"],
        "manifest_purpose": manifest["purpose"],
        "dataset": manifest["dataset"],
        "split": args.split,
        "sequence_filter": args.sequence,
        "expected_sequences": list(expected_sequences),
        "policy": args.policy,
        "reporting_boundary_fraction": 0.5,
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
    parser.add_argument("--policy", choices=TRANSFER_POLICIES, required=True)
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
    manifest = validate_manifest(load_manifest(manifest_path))
    all_entries = select_entries(manifest, args.split, args.sequence)
    entries = all_entries[args.shard_id :: args.num_shards]
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
    publish_run(
        output_dir,
        make_run(
            args,
            manifest,
            checkpoint,
            [entry["sequence"] for entry in all_entries],
        ),
    )
    total_frames = sum(len(sequence.frames) - 1 for sequence in sequences)
    total_blocks = sum(
        assignment["router_candidate_blocks"]
        for assignment in assignments.values()
    )
    print("Checkpoint: {}".format(checkpoint))
    print("Manifest: {}".format(manifest_path))
    print("Manifest fingerprint: {}".format(manifest["fingerprint"]))
    print("Stage-10 output directory: {}".format(output_dir))
    print("Frozen policy: {}".format(args.policy))
    print("Analysis role: descriptive post-Stage-9 transfer diagnostic")
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
    print("Newly completed Stage-10 clips: {}".format(completed))
    print(
        "Next: python tracking/stage10_closed_loop_summary.py --fast <fast-dir> "
        "--periodic <periodic-l9-20-dir> --sgla <sgla-dir> "
        "--output_dir <comparison-dir>"
    )


if __name__ == "__main__":
    main()
