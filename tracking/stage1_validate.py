"""Stage-1 Fast/SGLA/Full counterfactual validation for SGLATrack.

This runner is intentionally isolated from the normal tracking entry point. It
loads the unmodified tracker and checkpoint, executes three backbone paths on
the same template/search crop, advances the real trajectory with the SGLA
prediction, and writes one CSV row per frame.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

import types

# Compatibility for legacy SGLATrack code on modern PyTorch.
if "torch._six" not in sys.modules:
    torch_six = types.ModuleType("torch._six")
    torch_six.string_classes = (str, bytes)
    torch_six.int_classes = int
    sys.modules["torch._six"] = torch_six
    setattr(torch, "_six", torch_six)

# Compatibility for trusted checkpoints under PyTorch 2.6+.
_original_torch_load = torch.load


def _load_trusted_legacy_checkpoint(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _load_trusted_legacy_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from lib.models.sglatrack.base_backbone import enabled_layer_num, start_layer
from lib.models.sglatrack.utils import combine_tokens, recover_tokens
from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box


PATH_MODES = ("fast", "sgla", "full")
CSV_FIELDS = [
    "dataset",
    "sequence",
    "frame_id",
    "gt_valid",
    "iou_fast",
    "iou_sgla",
    "iou_full",
    "delta_full_fast",
    "sgla_selected_layer",
    "gt_x",
    "gt_y",
    "gt_w",
    "gt_h",
    "bbox_fast_x",
    "bbox_fast_y",
    "bbox_fast_w",
    "bbox_fast_h",
    "bbox_sgla_x",
    "bbox_sgla_y",
    "bbox_sgla_w",
    "bbox_sgla_h",
    "bbox_full_x",
    "bbox_full_y",
    "bbox_full_w",
    "bbox_full_h",
]


def _forward_backbone_path(backbone, template, search, mode):
    """Run one inference path while matching BaseBackbone.forward_test."""
    if mode not in PATH_MODES:
        raise ValueError("Unknown inference mode: {}".format(mode))

    z = backbone.patch_embed(template)
    x = backbone.patch_embed(search)
    z = z + backbone.pos_embed_z
    x = x + backbone.pos_embed_x

    lens_z = backbone.pos_embed_z.shape[1]
    lens_x = backbone.pos_embed_x.shape[1]
    x = combine_tokens(z, x, mode=backbone.cat_mode)
    x = backbone.pos_drop(x)

    selected_indices = None
    selected_layer = None

    if mode == "full":
        for block in backbone.blocks:
            x = block(x)
    else:
        for block_index, block in enumerate(backbone.blocks):
            if block_index <= start_layer:
                x = block(x)
                if block_index == start_layer and mode == "sgla":
                    probabilities = backbone.MLP(x[:, :, 0].clone())
                    topk_indices = torch.topk(
                        probabilities, enabled_layer_num, dim=1
                    ).indices
                    selected_indices = (
                        torch.sort(topk_indices, dim=1).values + start_layer + 1
                    )
                    selected_layer = int(selected_indices[0, 0].item()) + 1
                continue

            if mode == "fast":
                break

            batch_indices = torch.where(selected_indices[:, :] == block_index)[0]
            if len(batch_indices) > 0:
                # This is the exact inference operation used by the original
                # SGLA forward_test implementation for enabled_layer_num == 1.
                x[batch_indices] = block(x[batch_indices])
                break

    x = recover_tokens(x, lens_z, lens_x, mode=backbone.cat_mode)
    return backbone.norm(x), selected_layer


def _forward_path(network, template, search, mode):
    features, selected_layer = _forward_backbone_path(
        network.backbone, template, search, mode
    )
    return network.forward_head(features, None), selected_layer


def _decode_box(tracker, output, resize_factor, image_height, image_width):
    response = tracker.output_window * output["score_map"]
    pred_boxes = tracker.network.box_head.cal_bbox(
        response, output["size_map"], output["offset_map"]
    ).view(-1, 4)
    pred_box = (
        pred_boxes.mean(dim=0)
        * tracker.params.search_size
        / resize_factor
    ).tolist()
    mapped_box = tracker.map_box_back(pred_box, resize_factor)
    return clip_box(mapped_box, image_height, image_width, margin=10)


def _is_valid_gt(box, target_visible=True):
    values = np.asarray(box, dtype=np.float64).reshape(-1)
    return bool(
        target_visible
        and values.size == 4
        and np.isfinite(values).all()
        and values[2] > 0.0
        and values[3] > 0.0
    )


def _iou_xywh(prediction, ground_truth, gt_valid=True):
    """Match the inclusive-pixel IoU convention in extract_results.py."""
    if not gt_valid:
        return -1.0

    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    gt = np.asarray(ground_truth, dtype=np.float64).reshape(-1)
    if (
        pred.size != 4
        or gt.size != 4
        or not np.isfinite(pred).all()
        or pred[2] < 0.0
        or pred[3] < 0.0
    ):
        return -1.0

    # Normal SGLATrack evaluation saves predictions with astype(int) before
    # loading them in extract_results.py. Use the same truncation here so the
    # SGLA AUC in this CSV is directly comparable with the reported baseline.
    pred = pred.astype(np.int64).astype(np.float64)

    top_left = np.maximum(pred[:2], gt[:2])
    bottom_right = np.minimum(
        pred[:2] + pred[2:] - 1.0,
        gt[:2] + gt[2:] - 1.0,
    )
    size = np.maximum(bottom_right - top_left + 1.0, 0.0)
    intersection = float(size[0] * size[1])
    union = float(pred[2] * pred[3] + gt[2] * gt[3] - intersection)
    if union <= 0.0:
        return -1.0
    return intersection / union


def _frame_row(
    dataset_name,
    sequence_name,
    frame_id,
    ground_truth,
    boxes,
    selected_layer,
    target_visible=True,
):
    gt = np.asarray(ground_truth, dtype=np.float64).reshape(-1)
    gt_valid = _is_valid_gt(gt, target_visible)
    if frame_id == 0 and gt_valid:
        # extract_results.py explicitly replaces the first prediction with GT.
        ious = {mode: 1.0 for mode in PATH_MODES}
    else:
        ious = {
            mode: _iou_xywh(boxes[mode], gt, gt_valid)
            for mode in PATH_MODES
        }

    row = {
        "dataset": dataset_name,
        "sequence": sequence_name,
        "frame_id": int(frame_id),
        "gt_valid": int(gt_valid),
        "iou_fast": ious["fast"],
        "iou_sgla": ious["sgla"],
        "iou_full": ious["full"],
        "delta_full_fast": ious["full"] - ious["fast"],
        "sgla_selected_layer": "" if selected_layer is None else selected_layer,
        "gt_x": gt[0],
        "gt_y": gt[1],
        "gt_w": gt[2],
        "gt_h": gt[3],
    }
    for mode in PATH_MODES:
        box = boxes[mode]
        for field_name, value in zip(("x", "y", "w", "h"), box):
            row["bbox_{}_{}".format(mode, field_name)] = value
    return row


def _target_visible(sequence, frame_id):
    if sequence.target_visible is None:
        return True
    return bool(sequence.target_visible[frame_id])


def _write_rows(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary_path), str(csv_path))


def _safe_sequence_name(name):
    return str(name).replace("/", "_").replace("\\", "_")


def _verify_sgla_output(tracker, template, search, scripted_output):
    reference = tracker.network(
        template=template,
        search=search,
        ce_template_mask=tracker.box_mask_z,
    )
    keys = ("pred_boxes", "score_map", "size_map", "offset_map")
    for key in keys:
        if not torch.allclose(
            reference[key], scripted_output[key], rtol=1e-5, atol=1e-6
        ):
            difference = (reference[key] - scripted_output[key]).abs().max().item()
            raise RuntimeError(
                "Standalone SGLA path differs from the original path for {} "
                "(max abs difference: {:.8g}).".format(key, difference)
            )


def _run_sequence(tracker_info, tracker, sequence, output_dir, overwrite, verify_sgla):
    sequence_csv = output_dir / "{}_stage1.csv".format(
        _safe_sequence_name(sequence.name)
    )
    if sequence_csv.is_file() and not overwrite:
        print("[skip] {} -> {}".format(sequence.name, sequence_csv))
        return False

    first_image = tracker_info._read_image(sequence.frames[0])
    init_info = dict(sequence.init_info())
    tracker.initialize(first_image, init_info)

    initial_box = list(tracker.state)
    initial_boxes = {mode: list(initial_box) for mode in PATH_MODES}
    rows = [
        _frame_row(
            sequence.dataset,
            sequence.name,
            0,
            sequence.ground_truth_rect[0],
            initial_boxes,
            selected_layer=None,
            target_visible=_target_visible(sequence, 0),
        )
    ]

    for frame_id, frame_path in enumerate(sequence.frames[1:], start=1):
        image = tracker_info._read_image(frame_path)
        image_height, image_width, _ = image.shape

        # tracker.state is deliberately unchanged until all three modes have
        # decoded their boxes, so every mode uses the same crop and reference.
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image,
            tracker.state,
            tracker.params.search_factor,
            output_sz=tracker.params.search_size,
        )
        search = tracker.preprocessor.process(x_patch_arr, x_amask_arr)
        template_tensor = tracker.z_dict1.tensors
        search_tensor = search.tensors

        outputs = {}
        selected_layer = None
        with torch.no_grad():
            for mode in PATH_MODES:
                output, mode_selected_layer = _forward_path(
                    tracker.network,
                    template_tensor,
                    search_tensor,
                    mode,
                )
                outputs[mode] = output
                if mode == "sgla":
                    selected_layer = mode_selected_layer

            if verify_sgla and frame_id == 1:
                _verify_sgla_output(
                    tracker,
                    template_tensor,
                    search_tensor,
                    outputs["sgla"],
                )

        boxes = {
            mode: _decode_box(
                tracker,
                outputs[mode],
                resize_factor,
                image_height,
                image_width,
            )
            for mode in PATH_MODES
        }

        # The common trajectory follows the original SGLA prediction. Fast and
        # Full remain same-frame counterfactual observations.
        tracker.state = list(boxes["sgla"])
        tracker.frame_id = frame_id

        rows.append(
            _frame_row(
                sequence.dataset,
                sequence.name,
                frame_id,
                sequence.ground_truth_rect[frame_id],
                boxes,
                selected_layer,
                target_visible=_target_visible(sequence, frame_id),
            )
        )

    _write_rows(sequence_csv, rows)
    print(
        "[done] {}: {} frames -> {}".format(
            sequence.name, len(rows), sequence_csv
        )
    )
    return True


def _parse_sequence_key(value):
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Run isolated Fast/SGLA/Full stage-1 validation."
    )
    parser.add_argument("tracker_name", nargs="?", default="sglatrack")
    parser.add_argument("tracker_param", nargs="?", default="deit_distilled")
    parser.add_argument("--dataset_name", default="uav123")
    parser.add_argument("--runid", type=int, default=None)
    parser.add_argument(
        "--sequence",
        default=None,
        help="Sequence index or exact sequence name. Omit for the whole dataset.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--verify_sgla",
        action="store_true",
        help="On each sequence, compare the first tracked frame with the original SGLA forward path.",
    )
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

    torch.cuda.set_device(args.gpu)
    dataset = get_dataset(args.dataset_name)
    sequence_key = _parse_sequence_key(args.sequence)
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
            / "stage1_validation"
            / args.dataset_name
        )
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Checkpoint: {}".format(params.checkpoint))
    print("Stage-1 output directory: {}".format(output_dir))
    print(
        "Paths: Fast=L1-L{}, SGLA=L1-L{}+selected, Full=L1-L{}".format(
            start_layer + 1,
            start_layer + 1,
            len(tracker.network.backbone.blocks),
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
                output_dir,
                args.overwrite,
                args.verify_sgla,
            )
        )

    print("Newly completed sequences: {}".format(completed))
    print(
        "Next: python tracking/stage1_oracle.py --input \"{}\"".format(
            output_dir
        )
    )


if __name__ == "__main__":
    main()
