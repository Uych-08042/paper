"""Run closed-loop SGLATrack candidate-median policies.

Unlike the Stage-2/Stage-3 counterfactual CSV analyses, each policy in this
runner updates its own tracker state. Every later crop therefore follows that
policy's actual predicted trajectory.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import stage1_validate as stage1


torch = stage1.torch
np = stage1.np
combine_tokens = stage1.combine_tokens
recover_tokens = stage1.recover_tokens
start_layer = stage1.start_layer
enabled_layer_num = stage1.enabled_layer_num
get_dataset = stage1.get_dataset
Tracker = stage1.Tracker
sample_target = stage1.sample_target


CANDIDATE_LAYERS = tuple(range(start_layer + 2, 13))
POLICY_LAYERS = {
    "median3": (7, 9, 12),
    "median4": (7, 9, 11, 12),
    "median6": (7, 8, 9, 10, 11, 12),
}
POLICIES = ("sgla",) + tuple(POLICY_LAYERS)
CSV_FIELDS = (
    "dataset",
    "sequence",
    "policy",
    "frame_id",
    "gt_valid",
    "iou",
    "candidate_blocks",
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


def _features_from_tokens(backbone, tokens, lens_z, lens_x):
    recovered = recover_tokens(tokens, lens_z, lens_x, mode=backbone.cat_mode)
    return backbone.norm(recovered)


def _forward_policy(network, template, search, policy):
    if policy not in POLICIES:
        raise ValueError("Unknown policy: {}".format(policy))
    if enabled_layer_num != 1:
        raise RuntimeError(
            "Closed-loop validation requires enabled_layer_num == 1, got {}".format(
                enabled_layer_num
            )
        )

    backbone = network.backbone
    if len(backbone.blocks) != 12:
        raise RuntimeError(
            "Expected a 12-block backbone, got {}".format(len(backbone.blocks))
        )

    z = backbone.patch_embed(template)
    x = backbone.patch_embed(search)
    z = z + backbone.pos_embed_z
    x = x + backbone.pos_embed_x
    lens_z = backbone.pos_embed_z.shape[1]
    lens_x = backbone.pos_embed_x.shape[1]
    tokens = combine_tokens(z, x, mode=backbone.cat_mode)
    tokens = backbone.pos_drop(tokens)
    for block_index in range(start_layer + 1):
        tokens = backbone.blocks[block_index](tokens)

    selected_layer = None
    if policy == "sgla":
        probabilities = backbone.MLP(tokens[:, :, 0].clone())
        selected_offset = int(probabilities.argmax(dim=1).item())
        selected_layer = CANDIDATE_LAYERS[selected_offset]
        requested_layers = (selected_layer,)
    else:
        requested_layers = POLICY_LAYERS[policy]

    outputs = {}
    for layer in requested_layers:
        candidate_tokens = backbone.blocks[layer - 1](tokens.clone())
        features = _features_from_tokens(
            backbone, candidate_tokens, lens_z, lens_x
        )
        outputs[layer] = network.forward_head(features, None)
    return outputs, selected_layer


def _fuse_boxes(boxes, policy, image_height, image_width):
    if policy == "sgla":
        fused = list(next(iter(boxes.values())))
    else:
        fused = np.median(
            np.asarray(list(boxes.values()), dtype=np.float64), axis=0
        ).tolist()
    return stage1.clip_box(fused, image_height, image_width, margin=10)


def _initial_row(sequence, policy, initial_box):
    ground_truth = np.asarray(
        sequence.ground_truth_rect[0], dtype=np.float64
    ).reshape(-1)
    gt_valid = stage1._is_valid_gt(
        ground_truth, stage1._target_visible(sequence, 0)
    )
    row = {
        "dataset": sequence.dataset,
        "sequence": sequence.name,
        "policy": policy,
        "frame_id": 0,
        "gt_valid": int(gt_valid),
        "iou": 1.0 if gt_valid else -1.0,
        "candidate_blocks": 1 if policy == "sgla" else len(POLICY_LAYERS[policy]),
        "selected_layer": "",
        "model_decode_latency_ms": "",
        "gt_x": ground_truth[0],
        "gt_y": ground_truth[1],
        "gt_w": ground_truth[2],
        "gt_h": ground_truth[3],
        "bbox_x": initial_box[0],
        "bbox_y": initial_box[1],
        "bbox_w": initial_box[2],
        "bbox_h": initial_box[3],
    }
    return row


def _tracked_row(
    sequence,
    policy,
    frame_id,
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
    return {
        "dataset": sequence.dataset,
        "sequence": sequence.name,
        "policy": policy,
        "frame_id": int(frame_id),
        "gt_valid": int(gt_valid),
        "iou": stage1._iou_xywh(box, ground_truth, gt_valid),
        "candidate_blocks": 1 if policy == "sgla" else len(POLICY_LAYERS[policy]),
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


def _run_policy_sequence(
    tracker_info,
    tracker,
    sequence,
    policy,
    output_dir,
    overwrite,
    verify_sgla,
):
    output_path = output_dir / "{}_{}_closed_loop.csv".format(
        stage1._safe_sequence_name(sequence.name), policy
    )
    if output_path.is_file() and not overwrite:
        print("[skip] {}/{} -> {}".format(sequence.name, policy, output_path))
        return False

    first_image = tracker_info._read_image(sequence.frames[0])
    tracker.initialize(first_image, dict(sequence.init_info()))
    rows = [_initial_row(sequence, policy, list(tracker.state))]

    for frame_id, frame_path in enumerate(sequence.frames[1:], start=1):
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
            outputs, selected_layer = _forward_policy(
                tracker.network,
                template_tensor,
                search_tensor,
                policy,
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
            fused_box = _fuse_boxes(
                boxes, policy, image_height, image_width
            )
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        if verify_sgla and policy == "sgla" and frame_id == 1:
            with torch.no_grad():
                stage1._verify_sgla_output(
                    tracker,
                    template_tensor,
                    search_tensor,
                    outputs[selected_layer],
                )

        rows.append(
            _tracked_row(
                sequence,
                policy,
                frame_id,
                fused_box,
                selected_layer,
                latency_ms,
            )
        )
        tracker.state = list(fused_box)
        tracker.frame_id = frame_id

    _write_rows(output_path, rows)
    print(
        "[done] {}/{}: {} frames -> {}".format(
            sequence.name, policy, len(rows), output_path
        )
    )
    return True


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Run independent closed-loop SGLA median-fusion policies."
    )
    parser.add_argument("tracker_name", nargs="?", default="sglatrack")
    parser.add_argument("tracker_param", nargs="?", default="deit_distilled")
    parser.add_argument("--dataset_name", default="uav123")
    parser.add_argument("--runid", type=int, default=None)
    parser.add_argument(
        "--sequence",
        default=None,
        help="Sequence index or exact name. Omit for the whole dataset.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--policies", nargs="+", choices=POLICIES, default=["median4"]
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify_sgla", action="store_true")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    return parser


def main():
    args = _build_parser().parse_args()
    if args.num_shards < 1:
        raise ValueError("--num_shards must be at least 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard_id must be in [0, num_shards)")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the original SGLATrack tracker.")

    policies = tuple(dict.fromkeys(args.policies))
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

    if args.output_dir is None:
        output_dir = (
            Path(tracker_info.results_dir)
            / "stage3_closed_loop_validation"
            / args.dataset_name
        )
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Checkpoint: {}".format(params.checkpoint))
    print("Closed-loop output directory: {}".format(output_dir))
    print("Policies: {}".format(", ".join(policies)))
    print(
        "Shard {}/{}: {} sequence(s)".format(
            args.shard_id, args.num_shards, len(sequences)
        )
    )

    completed = 0
    for sequence in sequences:
        for policy in policies:
            completed += int(
                _run_policy_sequence(
                    tracker_info,
                    tracker,
                    sequence,
                    policy,
                    output_dir,
                    args.overwrite,
                    args.verify_sgla,
                )
            )
    print("Newly completed sequence-policy runs: {}".format(completed))
    print(
        "Next: python tracking/stage3_closed_loop_summary.py --input \"{}\"".format(
            output_dir
        )
    )


if __name__ == "__main__":
    main()
