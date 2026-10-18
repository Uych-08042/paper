"""Run the frozen final external validation on the complete OTB-2015 set.

This stage is intentionally narrow. It evaluates exactly four policies on
every OTB sequence object and records a contract that prevents policy search,
subsetting, checkpoint changes, or mixed shards after outcomes are observed.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from stage3_periodic import periodic_l9_active
from stage4_common import IOU_PROTOCOL, file_hash, fingerprint, write_json


SCHEMA = "stage11_external_validation_v1"
DATASET_NAME = "otb"
EXPECTED_SEQUENCE_OBJECTS = 100
EXPECTED_PHYSICAL_VIDEOS = 98
FROZEN_POLICIES = (
    "fast",
    "periodic_l9_20",
    "periodic_l9_30",
    "sgla",
)
PRIMARY_POLICY = "periodic_l9_30"
SECONDARY_POLICY = "periodic_l9_20"
NONINFERIORITY_MARGIN_AUC = 0.5
LATENCY_WARMUP_FRAMES = 20


def expected_candidate_blocks(tracked_frames, policy):
    if tracked_frames < 1:
        raise ValueError("Each OTB sequence must contain a tracked frame")
    if policy == "fast":
        return 0
    if policy == "sgla":
        return tracked_frames
    if policy not in ("periodic_l9_20", "periodic_l9_30"):
        raise ValueError("Policy is not frozen for Stage-11: {}".format(policy))
    return sum(
        int(periodic_l9_active(policy, frame_id))
        for frame_id in range(1, tracked_frames + 1)
    )


def load_otb(dataset_root):
    """Load OTB from an explicit root without editing evaluation/local.py."""
    from lib.test.evaluation.otbdataset import OTBDataset

    dataset = OTBDataset()
    dataset.base_path = str(Path(dataset_root).expanduser().resolve())
    return list(dataset.get_sequence_list())


def audit_otb_sequences(
    sequences,
    dataset_root,
    expected_count=EXPECTED_SEQUENCE_OBJECTS,
):
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("OTB dataset root does not exist: {}".format(root))
    if len(sequences) != expected_count:
        raise ValueError(
            "Expected {} OTB sequence objects, got {}. Check extraction and "
            "the selected dataset root.".format(expected_count, len(sequences))
        )

    names = [sequence.name for sequence in sequences]
    if len(set(names)) != len(names):
        raise ValueError("OTB sequence object names must be unique")

    entries = []
    missing_frames = []
    for sequence in sequences:
        frames = [Path(path).expanduser().resolve() for path in sequence.frames]
        ground_truth = np.asarray(
            sequence.ground_truth_rect, dtype="<f8"
        )
        if ground_truth.ndim != 2 or ground_truth.shape[1] != 4:
            raise ValueError(
                "Invalid OTB annotation shape for {}: {}".format(
                    sequence.name, ground_truth.shape
                )
            )
        if len(frames) != len(ground_truth) or len(frames) < 2:
            raise ValueError(
                "Frame/annotation length mismatch for {}: {} frames, {} boxes".format(
                    sequence.name, len(frames), len(ground_truth)
                )
            )
        for frame in frames:
            if not frame.is_file():
                missing_frames.append(str(frame))
                if len(missing_frames) >= 10:
                    break
        if len(missing_frames) >= 10:
            break

        try:
            first_frame = frames[0].relative_to(root).as_posix()
            last_frame = frames[-1].relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(
                "Sequence {} resolves outside dataset root {}".format(
                    sequence.name, root
                )
            ) from error
        entries.append(
            {
                "sequence": sequence.name,
                "physical_video": Path(first_frame).parts[0],
                "frames": len(frames),
                "first_frame": first_frame,
                "last_frame": last_frame,
                "annotation_sha256": hashlib.sha256(
                    np.ascontiguousarray(ground_truth).tobytes()
                ).hexdigest(),
            }
        )

    if missing_frames:
        raise FileNotFoundError(
            "OTB extraction is incomplete; missing frame(s), first entries: {}".format(
                ", ".join(missing_frames)
            )
        )

    physical_videos = {entry["physical_video"] for entry in entries}
    if (
        expected_count == EXPECTED_SEQUENCE_OBJECTS
        and len(physical_videos) != EXPECTED_PHYSICAL_VIDEOS
    ):
        raise ValueError(
            "Expected {} physical OTB videos, got {}".format(
                EXPECTED_PHYSICAL_VIDEOS, len(physical_videos)
            )
        )

    body = {"dataset": DATASET_NAME, "sequences": entries}
    return {
        "dataset": DATASET_NAME,
        "sequence_objects": len(entries),
        "physical_videos": len(physical_videos),
        "physical_frame_references": sum(entry["frames"] for entry in entries),
        "tracked_frames": sum(entry["frames"] - 1 for entry in entries),
        "sequences": entries,
        "fingerprint": fingerprint(body),
    }


def publish_run(output_dir, run):
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
                    "run.lock persists; check for an active shard before removing it"
                )
            time.sleep(0.1)
    try:
        path = output_dir / "run.json"
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous != run:
                raise ValueError(
                    "Stage-11 contract changed; use a new empty output directory"
                )
        else:
            if any(output_dir.rglob("*_closed_loop.csv")):
                raise ValueError(
                    "Unidentified closed-loop files exist; use an empty output directory"
                )
            write_json(path, run)
    finally:
        lock.unlink()


def make_run(args, checkpoint, dataset_audit):
    directory = Path(__file__).resolve().parent
    code_files = (
        "stage11_external_validate.py",
        "stage3_closed_loop_validate.py",
        "stage3_periodic.py",
        "stage1_validate.py",
    )
    run = {
        "schema": SCHEMA,
        "analysis_role": "predeclared_untouched_external_confirmation",
        "dataset": DATASET_NAME,
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "dataset_fingerprint": dataset_audit["fingerprint"],
        "expected_sequence_objects": dataset_audit["sequence_objects"],
        "expected_sequences": {
            entry["sequence"]: entry["frames"]
            for entry in dataset_audit["sequences"]
        },
        "sequence_clusters": {
            entry["sequence"]: entry["physical_video"]
            for entry in dataset_audit["sequences"]
        },
        "policies": list(FROZEN_POLICIES),
        "primary_policy": PRIMARY_POLICY,
        "secondary_policy": SECONDARY_POLICY,
        "noninferiority_margin_auc": NONINFERIORITY_MARGIN_AUC,
        "tracker_name": "sglatrack",
        "tracker_parameter": "deit_distilled",
        "checkpoint_sha256": file_hash(checkpoint),
        "latency_warmup_frames": LATENCY_WARMUP_FRAMES,
        "num_shards": args.num_shards,
        "iou_protocol": IOU_PROTOCOL,
        "code_sha256": {
            name: file_hash(directory / name) for name in code_files
        },
    }
    run["fingerprint"] = fingerprint(run)
    return run


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument(
        "--audit_only",
        action="store_true",
        help="Verify the complete OTB extraction without running the tracker.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Require 0 <= shard_id < num_shards")

    sequences = load_otb(args.dataset_root)
    dataset_audit = audit_otb_sequences(sequences, args.dataset_root)
    print("OTB root: {}".format(Path(args.dataset_root).expanduser().resolve()))
    print(
        "Dataset audit: {} sequence objects / {} physical videos, {} frame "
        "references, {} tracked frames".format(
            dataset_audit["sequence_objects"],
            dataset_audit["physical_videos"],
            dataset_audit["physical_frame_references"],
            dataset_audit["tracked_frames"],
        )
    )
    print("Dataset fingerprint: {}".format(dataset_audit["fingerprint"]))
    if args.audit_only:
        print("Audit complete. No predictions were generated.")
        return

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --audit_only is used")
    if args.output_dir is None:
        raise ValueError("--output_dir is required unless --audit_only is used")
    import stage3_closed_loop_validate as stage3

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("Checkpoint missing: {}".format(checkpoint))
    if not stage3.torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the original SGLATrack tracker")

    stage3.torch.cuda.set_device(args.gpu)
    tracker_info = stage3.Tracker(
        "sglatrack", "deit_distilled", DATASET_NAME
    )
    params = tracker_info.get_parameters()
    params.debug = 0
    params.checkpoint = str(checkpoint)
    tracker = tracker_info.create_tracker(params)

    output_dir = Path(args.output_dir).expanduser().resolve()
    run = make_run(args, checkpoint, dataset_audit)
    publish_run(output_dir, run)
    shard_sequences = sequences[args.shard_id :: args.num_shards]
    print("Checkpoint: {}".format(checkpoint))
    print("Stage-11 output directory: {}".format(output_dir))
    print("Frozen policies: {}".format(", ".join(FROZEN_POLICIES)))
    print(
        "Primary rule: P30 > Fast and P30 non-inferior to SGLA at "
        "a {:.2f}-AUC margin; no policy tuning after this run".format(
            NONINFERIORITY_MARGIN_AUC
        )
    )
    print(
        "Shard {}/{}: {} sequence object(s)".format(
            args.shard_id, args.num_shards, len(shard_sequences)
        )
    )

    completed = 0
    for sequence in shard_sequences:
        for policy in FROZEN_POLICIES:
            completed += int(
                stage3._run_policy_sequence(
                    tracker_info,
                    tracker,
                    sequence,
                    policy,
                    output_dir,
                    False,
                    True,
                    LATENCY_WARMUP_FRAMES,
                )
            )
    print("Newly completed Stage-11 sequence-policy runs: {}".format(completed))
    print(
        "Next: python tracking/stage11_external_summary.py --input \"{}\"".format(
            output_dir
        )
    )


if __name__ == "__main__":
    main()
