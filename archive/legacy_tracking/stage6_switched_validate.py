"""Run a true Fast-prefix to OOF-assigned-policy closed-loop trajectory."""

import argparse
import csv
import os
import time
from collections import Counter
from pathlib import Path

import stage1_validate as stage1
import stage3_closed_loop_validate as stage3
from stage6_switch_common import active_policy, load_assignments


torch = stage1.torch
np = stage1.np
get_dataset = stage1.get_dataset
Tracker = stage1.Tracker
sample_target = stage1.sample_target


CSV_FIELDS = (
    "dataset",
    "sequence",
    "policy",
    "assigned_policy",
    "active_policy",
    "assignment_fold",
    "frame_id",
    "early_frames",
    "late_frames",
    "gt_valid",
    "iou",
    "candidate_blocks",
    "expected_sequence_candidate_blocks",
    "selected_layer",
    "model_decode_latency_ms",
    "gt_x",
    "gt_y",
    "gt_w",
    "gt_h",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
)


def _row(
    sequence,
    output_policy,
    assignment,
    frame_id,
    phase_policy,
    box,
    selected_layer,
    latency_ms,
):
    ground_truth = np.asarray(
        sequence.ground_truth_rect[frame_id], dtype=np.float64
    ).reshape(-1)
    gt_valid = stage1._is_valid_gt(
        ground_truth, stage1._target_visible(sequence, frame_id)
    )
    if frame_id == 0:
        iou = 1.0 if gt_valid else -1.0
        blocks = 0
    else:
        iou = stage1._iou_xywh(box, ground_truth, gt_valid)
        blocks = stage3._candidate_blocks(phase_policy, frame_id)
    return {
        "dataset": sequence.dataset,
        "sequence": sequence.name,
        "policy": output_policy,
        "assigned_policy": assignment["router_policy"],
        "active_policy": "init" if frame_id == 0 else phase_policy,
        "assignment_fold": assignment["fold"],
        "frame_id": frame_id,
        "early_frames": assignment["early_frames"],
        "late_frames": assignment["late_frames"],
        "gt_valid": int(gt_valid),
        "iou": iou,
        "candidate_blocks": blocks,
        "expected_sequence_candidate_blocks": assignment[
            "router_candidate_blocks"
        ],
        "selected_layer": "" if selected_layer is None else selected_layer,
        "model_decode_latency_ms": latency_ms,
        "gt_x": ground_truth[0],
        "gt_y": ground_truth[1],
        "gt_w": ground_truth[2],
        "gt_h": ground_truth[3],
        "bbox_x": box[0],
        "bbox_y": box[1],
        "bbox_w": box[2],
        "bbox_h": box[3],
    }


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _run_sequence(
    tracker_info,
    tracker,
    sequence,
    assignment,
    output_policy,
    output_dir,
    overwrite,
    verify_sgla,
    latency_warmup_frames,
    policy_resolver=active_policy,
):
    output_path = output_dir / "{}_{}_closed_loop.csv".format(
        stage1._safe_sequence_name(sequence.name),
        stage1._safe_sequence_name(output_policy),
    )
    if output_path.is_file() and not overwrite:
        print("[skip] {} -> {}".format(sequence.name, output_path))
        return False

    tracked_frames = len(sequence.frames) - 1
    if assignment["early_frames"] + assignment["late_frames"] != tracked_frames:
        raise ValueError(
            "Assignment length mismatch for {}: {}+{} != {}".format(
                sequence.name,
                assignment["early_frames"],
                assignment["late_frames"],
                tracked_frames,
            )
        )

    first_image = tracker_info._read_image(sequence.frames[0])
    tracker.initialize(first_image, dict(sequence.init_info()))
    rows = [
        _row(
            sequence,
            output_policy,
            assignment,
            0,
            "fast",
            list(tracker.state),
            None,
            "",
        )
    ]
    sgla_verified = False

    for frame_id, frame_path in enumerate(sequence.frames[1:], start=1):
        phase_policy = policy_resolver(
            frame_id,
            assignment["early_frames"],
            assignment["router_policy"],
        )
        image = tracker_info._read_image(frame_path)
        image_height, image_width, _ = image.shape
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image,
            tracker.state,
            tracker.params.search_factor,
            output_sz=tracker.params.search_size,
        )
        search = tracker.preprocessor.process(x_patch_arr, x_amask_arr)
        template_tensor = tracker.z_dict1.tensors
        search_tensor = search.tensors

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs, selected_layer = stage3._forward_policy(
                tracker.network,
                template_tensor,
                search_tensor,
                phase_policy,
                frame_id,
            )
            boxes = {
                layer: stage1._decode_box(
                    tracker,
                    output,
                    resize_factor,
                    image_height,
                    image_width,
                )
                for layer, output in outputs.items()
            }
            fused_box = stage3._fuse_boxes(
                boxes, phase_policy, image_height, image_width
            )
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        reported_latency = (
            "" if frame_id <= latency_warmup_frames else latency_ms
        )

        if verify_sgla and phase_policy == "sgla" and not sgla_verified:
            with torch.no_grad():
                stage1._verify_sgla_output(
                    tracker,
                    template_tensor,
                    search_tensor,
                    outputs[selected_layer],
                )
            sgla_verified = True

        rows.append(
            _row(
                sequence,
                output_policy,
                assignment,
                frame_id,
                phase_policy,
                fused_box,
                selected_layer,
                reported_latency,
            )
        )
        tracker.state = list(fused_box)
        tracker.frame_id = frame_id

    actual_blocks = sum(row["candidate_blocks"] for row in rows[1:])
    expected_blocks = assignment["router_candidate_blocks"]
    if actual_blocks != expected_blocks:
        raise RuntimeError(
            "Candidate-block mismatch for {}: actual {} != expected {}".format(
                sequence.name, actual_blocks, expected_blocks
            )
        )
    _write_rows(output_path, rows)
    print(
        "[done] {}/{}: {} frames, {} blocks -> {}".format(
            sequence.name,
            assignment["router_policy"],
            len(rows),
            actual_blocks,
            output_path,
        )
    )
    return True


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracker_name", nargs="?", default="sglatrack")
    parser.add_argument("tracker_param", nargs="?", default="deit_distilled")
    parser.add_argument("--dataset_name", default="uav123")
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--early_fraction", type=float, default=0.5)
    parser.add_argument("--budget_percent", type=float, default=10.0)
    parser.add_argument("--policy_name", default="oof_coarse50_b10")
    parser.add_argument("--runid", type=int, default=None)
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", default=None)
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

    assignment_path, assignments = load_assignments(
        args.assignments, args.early_fraction, args.budget_percent
    )
    torch.cuda.set_device(args.gpu)
    dataset = get_dataset(args.dataset_name)
    sequence_key = stage1._parse_sequence_key(args.sequence)
    if sequence_key is not None:
        sequences = [dataset[sequence_key]]
    else:
        sequences = list(dataset)[args.shard_id :: args.num_shards]

    selected_assignments = {}
    for sequence in sequences:
        key = (sequence.dataset, sequence.name)
        if key not in assignments:
            raise ValueError("Missing assignment for {}/{}".format(*key))
        selected_assignments[key] = assignments[key]

    tracker_info = Tracker(
        args.tracker_name,
        args.tracker_param,
        args.dataset_name,
        args.runid,
    )
    params = tracker_info.get_parameters()
    params.debug = 0
    tracker = tracker_info.create_tracker(params)

    if args.output_dir is None:
        output_dir = (
            Path(tracker_info.results_dir)
            / "stage6_switched_validation"
            / args.dataset_name
        )
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    distribution = Counter(
        assignment["router_policy"] for assignment in selected_assignments.values()
    )
    print("Checkpoint: {}".format(params.checkpoint))
    print("Assignments: {}".format(assignment_path))
    print("Switched output directory: {}".format(output_dir))
    print(
        "Frozen policy: first {:.0%} Fast, then OOF assignment at {:.1f}% "
        "late budget".format(args.early_fraction, args.budget_percent)
    )
    print(
        "Assignment distribution: {}".format(
            "|".join(
                "{}:{}".format(policy, distribution[policy])
                for policy in sorted(distribution)
            )
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
                selected_assignments[(sequence.dataset, sequence.name)],
                args.policy_name,
                output_dir,
                args.overwrite,
                args.verify_sgla,
                args.latency_warmup_frames,
            )
        )
    print("Newly completed switched sequences: {}".format(completed))
    print(
        "Next: python tracking/stage6_switched_summary.py --input \"{}\" "
        "--baseline_input <stage5-periodic-dir>".format(output_dir)
    )


if __name__ == "__main__":
    main()
