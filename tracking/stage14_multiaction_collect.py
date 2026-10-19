"""Collect one-step Fast/L7-L12 branches followed by a shared Fast rollout."""

import argparse
from pathlib import Path

import numpy as np

from stage4_collect import cpu_batch, publish_run
from stage4_common import (
    file_hash,
    fingerprint,
    frame_path,
    load_annotations,
    load_manifest,
    tracking_iou,
    write_cache,
)
from stage4_runtime import Runtime, add_runtime_arguments
from stage8_intervention_common import select_anchor_offsets
from stage14_multiaction_common import (
    ACTION_NAMES,
    CANDIDATE_LAYERS,
    discounted_policy_values,
    load_and_audit_cache,
    make_run,
    multiaction_cache_path,
)


def _fast_prediction(rt, image, state):
    rt.tracker.state = list(state)
    search, factor = rt.prepare(image)
    tokens = rt.prefix(search)
    output = rt.head(tokens)
    box = np.asarray(rt.decode(output, factor, image), dtype=np.float64)
    return {
        "tokens": tokens,
        "factor": factor,
        "fast_output": output,
        "fast_box": box,
    }


def _fast_rollout(rt, directory, dataset, absolute_frame, offset, first_box, horizon):
    boxes = [np.asarray(first_box, dtype=np.float64).copy()]
    state = boxes[0].copy()
    for step in range(1, horizon):
        image = rt.read(frame_path(directory, dataset, absolute_frame + step))
        rt.tracker.state = state.tolist()
        rt.tracker.frame_id = offset + step - 1
        state = np.asarray(rt.run_image(image, "fast"), dtype=np.float64)
        boxes.append(state.copy())
    return np.stack(boxes)


def collect_multiaction(
    rt,
    directory,
    dataset,
    entry,
    ground_truth,
    valid,
    pool_grid,
    anchor_count,
    horizon,
    gamma,
    min_valid_fraction,
    verify_actions=False,
):
    """Fork L7-L12 once from identical Fast states, then roll Fast to H."""
    start, stop = entry["start"], entry["stop"]
    tracked_frames = stop - start - 1
    offsets = select_anchor_offsets(tracked_frames, horizon, anchor_count)
    anchor_set = set(offsets.tolist())
    if stop > len(ground_truth) or not valid[start]:
        raise ValueError("Invalid clip bounds or initialization")

    initial = np.asarray(ground_truth[start], dtype=np.float64).copy()
    rt.initialize(rt.read(frame_path(directory, dataset, start)), initial)
    baseline_boxes = np.empty((tracked_frames, 4), dtype=np.float64)
    records = []
    control_probe_boxes = None
    actions_verified = False

    with rt.torch.no_grad():
        for offset in range(1, tracked_frames + 1):
            absolute_frame = start + offset
            image = rt.read(frame_path(directory, dataset, absolute_frame))
            previous_state = np.asarray(rt.tracker.state, dtype=np.float64).copy()
            prediction = _fast_prediction(rt, image, previous_state)
            fast_box = prediction["fast_box"]

            if offset in anchor_set:
                probabilities = rt.selector(prediction["tokens"])
                selected_layer = rt.selected_layer(probabilities)
                candidate_boxes = []
                for layer in CANDIDATE_LAYERS:
                    output = rt.candidate(prediction["tokens"], layer)
                    candidate_boxes.append(
                        np.asarray(
                            rt.decode(output, prediction["factor"], image),
                            dtype=np.float64,
                        )
                    )
                candidate_boxes = np.stack(candidate_boxes)

                if verify_actions and not actions_verified:
                    rt.tracker.state = previous_state.tolist()
                    rt.tracker.frame_id = offset - 1
                    integrated_fast = np.asarray(rt.run_image(image, "fast"), dtype=np.float64)
                    if not np.allclose(integrated_fast, fast_box, rtol=0, atol=1e-6):
                        raise RuntimeError("Integrated Fast path disagrees with branch output")
                    rt.tracker.state = previous_state.tolist()
                    rt.tracker.frame_id = offset - 1
                    integrated_sgla = np.asarray(rt.run_image(image, "sgla"), dtype=np.float64)
                    selected_box = candidate_boxes[selected_layer - 7]
                    if not np.allclose(integrated_sgla, selected_box, rtol=0, atol=1e-6):
                        raise RuntimeError("Integrated selector path disagrees with selected candidate")
                    actions_verified = True

                if control_probe_boxes is None:
                    control_probe_boxes = _fast_rollout(
                        rt,
                        directory,
                        dataset,
                        absolute_frame,
                        offset,
                        fast_box,
                        horizon,
                    )

                candidate_rollouts = []
                for candidate_box in candidate_boxes:
                    candidate_rollouts.append(
                        _fast_rollout(
                            rt,
                            directory,
                            dataset,
                            absolute_frame,
                            offset,
                            candidate_box,
                            horizon,
                        )
                    )
                records.append(
                    {
                        "frame_id": np.int64(absolute_frame),
                        "anchor_offset": np.int64(offset),
                        "anchor_fraction": np.float64(offset / tracked_frames),
                        "token_features": cpu_batch(
                            rt.pool(prediction["tokens"], pool_grid), np.float16
                        ),
                        "fast_score_map": cpu_batch(
                            prediction["fast_output"]["score_map"], np.float16
                        ),
                        "selector_prob": cpu_batch(probabilities, np.float32),
                        "selected_layer": np.int64(selected_layer),
                        "previous_state": previous_state,
                        "previous_fast_box": previous_state.copy(),
                        "fast_box": fast_box.copy(),
                        "candidate_boxes": candidate_boxes.copy(),
                        "image_hw": np.asarray(image.shape[:2], dtype=np.int64),
                        "resize_factor": np.float64(prediction["factor"]),
                        "candidate_rollouts": np.stack(candidate_rollouts),
                    }
                )
                print(
                    "[branch] {}: anchor {}/{} at {}/{}; actions={}".format(
                        entry["sequence"],
                        len(records),
                        anchor_count,
                        offset,
                        tracked_frames,
                        "/".join(ACTION_NAMES),
                    ),
                    flush=True,
                )

            baseline_boxes[offset - 1] = fast_box
            rt.tracker.state = fast_box.tolist()
            rt.tracker.frame_id = offset

    if len(records) != anchor_count:
        raise RuntimeError("Collected {} anchors, expected {}".format(len(records), anchor_count))
    baseline_gt = np.asarray(ground_truth[start + 1 : stop], dtype=np.float64).copy()
    baseline_valid = np.asarray(valid[start + 1 : stop], dtype=bool).copy()
    baseline_iou = tracking_iou(baseline_boxes, baseline_gt, baseline_valid)
    control_boxes = np.stack(
        [baseline_boxes[offset - 1 : offset - 1 + horizon] for offset in offsets]
    )
    candidate_rollouts = np.stack([record["candidate_rollouts"] for record in records])
    policy_boxes = np.concatenate((control_boxes[:, None], candidate_rollouts), axis=1)
    gt_windows = np.stack(
        [baseline_gt[offset - 1 : offset - 1 + horizon] for offset in offsets]
    )
    valid_windows = np.stack(
        [baseline_valid[offset - 1 : offset - 1 + horizon] for offset in offsets]
    )
    policy_iou = tracking_iou(
        policy_boxes, gt_windows[:, None], valid_windows[:, None]
    )
    values, deltas, coverage, eligible = discounted_policy_values(
        policy_iou,
        valid_windows,
        horizon,
        gamma,
        min_valid_fraction,
    )

    anchor_fields = (
        "frame_id",
        "anchor_offset",
        "anchor_fraction",
        "token_features",
        "fast_score_map",
        "selector_prob",
        "selected_layer",
        "previous_state",
        "previous_fast_box",
        "fast_box",
        "candidate_boxes",
        "image_hw",
        "resize_factor",
    )
    arrays = {key: np.stack([record[key] for record in records]) for key in anchor_fields}
    arrays.update(
        {
            "action_boxes": policy_boxes[:, :, 0].copy(),
            "baseline_boxes": baseline_boxes,
            "baseline_gt": baseline_gt,
            "baseline_valid_gt": baseline_valid,
            "baseline_iou": baseline_iou,
            "policy_boxes": policy_boxes,
            "control_probe_boxes": control_probe_boxes,
            "gt": gt_windows,
            "valid_gt": valid_windows,
            "policy_iou": policy_iou,
            "delta_iou": policy_iou[:, 1:] - policy_iou[:, :1],
            "valid_fraction": coverage,
            "eligible": eligible,
            "discounted_value": values,
            "discounted_delta": deltas,
        }
    )
    return arrays, initial.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pool_grid", type=int, default=4)
    parser.add_argument("--anchors", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--min_valid_fraction", type=float, default=0.8)
    parser.add_argument("--verify_actions", action="store_true")
    parser.add_argument("--split", choices=("all", "train", "val"), default="all")
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Require 0 <= shard_id < num_shards")
    if args.pool_grid < 1 or args.horizon < 2:
        raise ValueError("pool_grid must be positive and horizon must exceed one")
    if not 0.0 < args.gamma <= 1.0 or not 0.0 < args.min_valid_fraction <= 1.0:
        raise ValueError("Invalid gamma or min_valid_fraction")
    if args.max_sequences is not None and args.max_sequences < 1:
        raise ValueError("--max_sequences must be positive")

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    design = manifest.get("intervention_design", {})
    anchor_count = args.anchors or int(design.get("anchor_count", 12))
    if args.anchors is not None and design.get("anchor_count") != args.anchors:
        raise ValueError("--anchors must match the frozen intervention manifest")
    if design.get("horizons") and args.horizon not in design["horizons"]:
        raise ValueError("--horizon must be one of the frozen intervention horizons")

    entries = [
        entry
        for entry in manifest["sequences"]
        if (args.split == "all" or entry["split"] == args.split)
        and (args.sequence is None or entry["sequence"] == args.sequence)
    ]
    if not entries:
        raise ValueError("No manifest clips match --split/--sequence")
    if args.max_sequences is not None:
        entries = entries[: args.max_sequences]
    entries = entries[args.shard_id :: args.num_shards]
    if not entries:
        print("This shard has no clips.")
        return

    rt = Runtime(args.parameter, args.gpu, args.checkpoint)
    if args.pool_grid > rt.search_side:
        raise ValueError("pool_grid exceeds the search token grid")
    output = Path(args.output_dir).expanduser().resolve()
    run = make_run(
        manifest,
        rt.identity,
        args.pool_grid,
        anchor_count,
        args.horizon,
        args.gamma,
        args.min_valid_fraction,
        args.verify_actions,
    )
    publish_run(output, run)
    print("Manifest: {}".format(manifest_path))
    print("Cache: {}".format(output))
    print("Run fingerprint: {}".format(run["fingerprint"]))
    print(
        "Stage-14: {} anchors x H{}; gamma={:g}; actions={}".format(
            anchor_count, args.horizon, args.gamma, "/".join(ACTION_NAMES)
        )
    )
    print("Shard {}/{}: {} clip(s)".format(args.shard_id, args.num_shards, len(entries)))

    completed = 0
    for entry in entries:
        path = multiaction_cache_path(output, entry)
        if path.exists() and not args.overwrite:
            report, _, _ = load_and_audit_cache(path, entry, run)
            print("[skip, checked] {}: {} eligible anchors".format(path, report["eligible_anchors"]))
            continue
        directory, ground_truth, valid = load_annotations(
            manifest["root"], manifest["dataset"], entry["sequence"]
        )
        if (
            file_hash(directory / "groundtruth.txt") != entry["annotation_sha256"]
            or fingerprint(valid.tolist()) != entry["visible_sha256"]
        ):
            raise ValueError("Annotations changed since manifest creation: {}".format(entry["sequence"]))
        arrays, initial = collect_multiaction(
            rt,
            directory,
            manifest["dataset"],
            entry,
            ground_truth,
            valid,
            args.pool_grid,
            anchor_count,
            args.horizon,
            args.gamma,
            args.min_valid_fraction,
            args.verify_actions,
        )
        write_cache(
            path,
            arrays,
            {
                "run_fingerprint": run["fingerprint"],
                "entry": entry,
                "init_box": initial,
            },
        )
        load_and_audit_cache(path, entry, run)
        completed += 1
        print(
            "[done] {}: {} anchors x {} actions x H{} -> {}".format(
                entry["sequence"], anchor_count, len(ACTION_NAMES), args.horizon, path
            ),
            flush=True,
        )
    print("Newly completed Stage-14 clips: {}".format(completed))
    print(
        'Next: python tracking/stage14_multiaction_audit.py --input "{}" --output_dir <audit-dir>'.format(
            output
        )
    )


if __name__ == "__main__":
    main()
