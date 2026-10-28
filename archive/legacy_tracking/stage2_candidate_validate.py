"""Stage-2 independent candidate-block validation for SGLATrack.

The original SGLA inference path runs blocks L1-L6 and then applies exactly
one selected block from L7-L12 to the L6 tokens. This runner evaluates every
candidate block independently from the same L6 feature. The real trajectory
continues to follow the original SGLA selector, while all other outputs are
same-frame counterfactual observations.
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Reuse Stage-1's compatibility shims and evaluation conventions without
# touching the model or tracker implementation.
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


CANDIDATE_BLOCK_INDICES = tuple(range(start_layer + 1, 12))
CANDIDATE_LAYERS = tuple(index + 1 for index in CANDIDATE_BLOCK_INDICES)
BOX_METHODS = (
    "fast",
    "sgla",
    "confidence_select",
    "ensemble_mean",
    "best_candidate",
) + tuple("layer{}".format(layer) for layer in CANDIDATE_LAYERS)


def _build_csv_fields():
    fields = [
        "dataset",
        "sequence",
        "frame_id",
        "gt_valid",
        "iou_fast",
        "iou_sgla",
        "iou_confidence_select",
        "iou_ensemble_mean",
        "iou_best_candidate",
        "delta_sgla_fast",
        "delta_confidence_fast",
        "delta_ensemble_fast",
        "delta_best_fast",
        "sgla_selected_layer",
        "confidence_selected_layer",
        "best_candidate_layer",
        "best_candidate_tie_count",
        "selector_hit",
        "confidence_hit",
        "selector_regret",
        "confidence_regret",
        "selector_max",
        "selector_margin",
        "selector_entropy",
        "selector_selected_probability",
        "confidence_selected_peak",
    ]
    for layer in CANDIDATE_LAYERS:
        fields.extend(
            [
                "iou_layer{}".format(layer),
                "delta_layer{}_fast".format(layer),
                "selector_prob_layer{}".format(layer),
                "response_peak_layer{}".format(layer),
            ]
        )
    fields.extend(["gt_x", "gt_y", "gt_w", "gt_h"])
    for method in BOX_METHODS:
        fields.extend(
            "bbox_{}_{}".format(method, coordinate)
            for coordinate in ("x", "y", "w", "h")
        )
    return fields


CSV_FIELDS = _build_csv_fields()


def _features_from_tokens(backbone, tokens, lens_z, lens_x):
    recovered = recover_tokens(tokens, lens_z, lens_x, mode=backbone.cat_mode)
    return backbone.norm(recovered)


def _forward_candidates(network, template, search):
    """Return Fast and all independent candidate-block head outputs."""
    backbone = network.backbone
    if enabled_layer_num != 1:
        raise RuntimeError(
            "Stage-2 validation requires enabled_layer_num == 1, got {}".format(
                enabled_layer_num
            )
        )
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

    layer6_tokens = tokens
    selector_probabilities = backbone.MLP(layer6_tokens[:, :, 0].clone())
    if selector_probabilities.shape[0] != 1:
        raise RuntimeError("Stage-2 runner currently requires batch size 1")
    if selector_probabilities.shape[1] != len(CANDIDATE_BLOCK_INDICES):
        raise RuntimeError(
            "Selector output width {} does not match {} candidates".format(
                selector_probabilities.shape[1],
                len(CANDIDATE_BLOCK_INDICES),
            )
        )

    fast_features = _features_from_tokens(
        backbone, layer6_tokens, lens_z, lens_x
    )
    fast_output = network.forward_head(fast_features, None)

    candidate_outputs = {}
    for block_index, layer in zip(
        CANDIDATE_BLOCK_INDICES, CANDIDATE_LAYERS
    ):
        candidate_tokens = backbone.blocks[block_index](layer6_tokens.clone())
        candidate_features = _features_from_tokens(
            backbone, candidate_tokens, lens_z, lens_x
        )
        candidate_outputs[layer] = network.forward_head(
            candidate_features, None
        )

    selected_offset = int(selector_probabilities.argmax(dim=1).item())
    selected_layer = CANDIDATE_LAYERS[selected_offset]
    return (
        fast_output,
        candidate_outputs,
        selector_probabilities[0],
        selected_layer,
    )


def _mean_output(candidate_outputs):
    outputs = list(candidate_outputs.values())
    return {
        key: torch.stack([output[key] for output in outputs], dim=0).mean(
            dim=0
        )
        for key in ("pred_boxes", "score_map", "size_map", "offset_map")
    }


def _response_peaks(tracker, candidate_outputs):
    layers = tuple(candidate_outputs)
    peaks = torch.stack(
        [
            (tracker.output_window * candidate_outputs[layer]["score_map"]).max()
            for layer in layers
        ]
    )
    values = peaks.detach().float().cpu().tolist()
    return {layer: float(value) for layer, value in zip(layers, values)}


def _selector_statistics(probabilities, selected_layer):
    values = tuple(float(value) for value in probabilities)
    sorted_values = sorted(values, reverse=True)
    maximum = sorted_values[0]
    margin = sorted_values[0] - sorted_values[1]

    total = max(sum(values), 1e-12)
    normalized = tuple(value / total for value in values)
    entropy = -sum(
        value * math.log(max(value, 1e-12)) for value in normalized
    ) / math.log(len(CANDIDATE_LAYERS))
    selected_offset = CANDIDATE_LAYERS.index(selected_layer)
    selected_probability = values[selected_offset]
    return maximum, margin, entropy, selected_probability


def _add_box_fields(row, method, box):
    for coordinate, value in zip(("x", "y", "w", "h"), box):
        row["bbox_{}_{}".format(method, coordinate)] = value


def _initial_row(sequence, initial_box):
    ground_truth = np.asarray(
        sequence.ground_truth_rect[0], dtype=np.float64
    ).reshape(-1)
    gt_valid = stage1._is_valid_gt(
        ground_truth, stage1._target_visible(sequence, 0)
    )
    initial_iou = 1.0 if gt_valid else -1.0
    row = {
        "dataset": sequence.dataset,
        "sequence": sequence.name,
        "frame_id": 0,
        "gt_valid": int(gt_valid),
        "iou_fast": initial_iou,
        "iou_sgla": initial_iou,
        "iou_confidence_select": initial_iou,
        "iou_ensemble_mean": initial_iou,
        "iou_best_candidate": initial_iou,
        "delta_sgla_fast": 0.0,
        "delta_confidence_fast": 0.0,
        "delta_ensemble_fast": 0.0,
        "delta_best_fast": 0.0,
        "sgla_selected_layer": "",
        "confidence_selected_layer": "",
        "best_candidate_layer": "",
        "best_candidate_tie_count": "",
        "selector_hit": "",
        "confidence_hit": "",
        "selector_regret": "",
        "confidence_regret": "",
        "selector_max": "",
        "selector_margin": "",
        "selector_entropy": "",
        "selector_selected_probability": "",
        "confidence_selected_peak": "",
        "gt_x": ground_truth[0],
        "gt_y": ground_truth[1],
        "gt_w": ground_truth[2],
        "gt_h": ground_truth[3],
    }
    for layer in CANDIDATE_LAYERS:
        row["iou_layer{}".format(layer)] = initial_iou
        row["delta_layer{}_fast".format(layer)] = 0.0
        row["selector_prob_layer{}".format(layer)] = ""
        row["response_peak_layer{}".format(layer)] = ""
    for method in BOX_METHODS:
        _add_box_fields(row, method, initial_box)
    return row


def _tracked_row(
    sequence,
    frame_id,
    boxes,
    selector_probabilities,
    response_peaks,
    selected_layer,
    confidence_layer,
):
    ground_truth = np.asarray(
        sequence.ground_truth_rect[frame_id], dtype=np.float64
    ).reshape(-1)
    gt_valid = stage1._is_valid_gt(
        ground_truth, stage1._target_visible(sequence, frame_id)
    )

    iou_fast = stage1._iou_xywh(boxes["fast"], ground_truth, gt_valid)
    candidate_ious = {
        layer: stage1._iou_xywh(
            boxes["layer{}".format(layer)], ground_truth, gt_valid
        )
        for layer in CANDIDATE_LAYERS
    }
    iou_sgla = candidate_ious[selected_layer]
    iou_confidence = candidate_ious[confidence_layer]
    iou_ensemble = stage1._iou_xywh(
        boxes["ensemble_mean"], ground_truth, gt_valid
    )

    if gt_valid:
        iou_best = max(candidate_ious.values())
        best_layers = tuple(
            layer
            for layer in CANDIDATE_LAYERS
            if iou_best - candidate_ious[layer] <= 1e-12
        )
        best_layer = best_layers[0]
        boxes["best_candidate"] = list(
            boxes["layer{}".format(best_layer)]
        )
        selector_regret = iou_best - iou_sgla
        confidence_regret = iou_best - iou_confidence
        selector_hit = int(selector_regret <= 1e-12)
        confidence_hit = int(confidence_regret <= 1e-12)
        best_tie_count = len(best_layers)
    else:
        best_layer = None
        iou_best = -1.0
        boxes["best_candidate"] = list(boxes["sgla"])
        selector_hit = ""
        confidence_hit = ""
        selector_regret = ""
        confidence_regret = ""
        best_tie_count = ""

    selector_max, selector_margin, selector_entropy, selected_probability = (
        _selector_statistics(selector_probabilities, selected_layer)
    )
    row = {
        "dataset": sequence.dataset,
        "sequence": sequence.name,
        "frame_id": int(frame_id),
        "gt_valid": int(gt_valid),
        "iou_fast": iou_fast,
        "iou_sgla": iou_sgla,
        "iou_confidence_select": iou_confidence,
        "iou_ensemble_mean": iou_ensemble,
        "iou_best_candidate": iou_best,
        "delta_sgla_fast": iou_sgla - iou_fast,
        "delta_confidence_fast": iou_confidence - iou_fast,
        "delta_ensemble_fast": iou_ensemble - iou_fast,
        "delta_best_fast": iou_best - iou_fast,
        "sgla_selected_layer": selected_layer,
        "confidence_selected_layer": confidence_layer,
        "best_candidate_layer": "" if best_layer is None else best_layer,
        "best_candidate_tie_count": best_tie_count,
        "selector_hit": selector_hit,
        "confidence_hit": confidence_hit,
        "selector_regret": selector_regret,
        "confidence_regret": confidence_regret,
        "selector_max": selector_max,
        "selector_margin": selector_margin,
        "selector_entropy": selector_entropy,
        "selector_selected_probability": selected_probability,
        "confidence_selected_peak": response_peaks[confidence_layer],
        "gt_x": ground_truth[0],
        "gt_y": ground_truth[1],
        "gt_w": ground_truth[2],
        "gt_h": ground_truth[3],
    }
    for layer in CANDIDATE_LAYERS:
        iou = candidate_ious[layer]
        row["iou_layer{}".format(layer)] = iou
        row["delta_layer{}_fast".format(layer)] = iou - iou_fast
        row["selector_prob_layer{}".format(layer)] = selector_probabilities[
            CANDIDATE_LAYERS.index(layer)
        ]
        row["response_peak_layer{}".format(layer)] = response_peaks[layer]
    for method in BOX_METHODS:
        _add_box_fields(row, method, boxes[method])
    return row


def _write_rows(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary_path), str(csv_path))


def _run_sequence(
    tracker_info, tracker, sequence, output_dir, overwrite, verify_sgla
):
    sequence_csv = output_dir / "{}_stage2_candidates.csv".format(
        stage1._safe_sequence_name(sequence.name)
    )
    if sequence_csv.is_file() and not overwrite:
        print("[skip] {} -> {}".format(sequence.name, sequence_csv))
        return False

    first_image = tracker_info._read_image(sequence.frames[0])
    tracker.initialize(first_image, dict(sequence.init_info()))
    rows = [_initial_row(sequence, list(tracker.state))]

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

        with torch.no_grad():
            (
                fast_output,
                candidate_outputs,
                selector_probabilities,
                selected_layer,
            ) = _forward_candidates(
                tracker.network, template_tensor, search_tensor
            )
            selector_probabilities = tuple(
                float(value)
                for value in selector_probabilities.detach().float().cpu().tolist()
            )
            sgla_output = candidate_outputs[selected_layer]
            ensemble_output = _mean_output(candidate_outputs)
            response_peaks = _response_peaks(tracker, candidate_outputs)
            confidence_layer = max(
                CANDIDATE_LAYERS,
                key=lambda layer: response_peaks[layer],
            )

            if verify_sgla and frame_id == 1:
                stage1._verify_sgla_output(
                    tracker,
                    template_tensor,
                    search_tensor,
                    sgla_output,
                )

        unique_outputs = {
            "fast": fast_output,
            "ensemble_mean": ensemble_output,
        }
        unique_outputs.update(
            {
                "layer{}".format(layer): output
                for layer, output in candidate_outputs.items()
            }
        )
        boxes = {
            method: stage1._decode_box(
                tracker,
                output,
                resize_factor,
                image_height,
                image_width,
            )
            for method, output in unique_outputs.items()
        }
        boxes["sgla"] = list(boxes["layer{}".format(selected_layer)])
        boxes["confidence_select"] = list(
            boxes["layer{}".format(confidence_layer)]
        )
        boxes["best_candidate"] = list(boxes["sgla"])

        rows.append(
            _tracked_row(
                sequence,
                frame_id,
                boxes,
                selector_probabilities,
                response_peaks,
                selected_layer,
                confidence_layer,
            )
        )

        # Preserve the baseline trajectory exactly: the original selector
        # chooses one independent candidate block for the next tracker state.
        tracker.state = list(boxes["sgla"])
        tracker.frame_id = frame_id

    _write_rows(sequence_csv, rows)
    print(
        "[done] {}: {} frames -> {}".format(
            sequence.name, len(rows), sequence_csv
        )
    )
    return True


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate L7-L12 independently from the same L6 feature and "
            "record selector/ensemble/Best-of-6 results."
        )
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
        help=(
            "For each sequence, compare the first tracked frame against the "
            "unmodified SGLA forward path."
        ),
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
            / "stage2_candidate_validation"
            / args.dataset_name
        )
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Checkpoint: {}".format(params.checkpoint))
    print("Stage-2 output directory: {}".format(output_dir))
    print(
        "Candidates: L{} base + independent {}".format(
            start_layer + 1,
            ", ".join("L{}".format(layer) for layer in CANDIDATE_LAYERS),
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
        "Next: python tracking/stage2_candidate_oracle.py --input \"{}\"".format(
            output_dir
        )
    )


if __name__ == "__main__":
    main()
