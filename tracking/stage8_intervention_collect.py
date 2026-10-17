"""Collect true single-L9 branch interventions from a Fast trajectory."""

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
from stage8_intervention_common import (
    DEFAULT_HORIZONS,
    intervention_cache_path,
    intervention_labels,
    load_and_audit_cache,
    make_run,
    select_anchor_offsets,
)


def _fast_prediction(rt, image, state):
    rt.tracker.state = list(state)
    search, factor = rt.prepare(image)
    tokens = rt.prefix(search)
    fast_output = rt.head(tokens)
    fast_box = rt.decode(fast_output, factor, image)
    return {
        "tokens": tokens,
        "factor": factor,
        "fast_output": fast_output,
        "fast_box": np.asarray(fast_box, dtype=np.float64),
    }


def collect_interventions(
    rt,
    directory,
    dataset,
    entry,
    ground_truth,
    valid,
    pool_grid,
    anchor_count,
    horizons,
    verify_l9=False,
):
    """Run one Fast baseline and fork one L9 treatment at each fixed anchor."""
    start, stop = entry["start"], entry["stop"]
    horizons = tuple(int(value) for value in horizons)
    max_horizon = max(horizons)
    tracked_frames = stop - start - 1
    anchor_offsets = select_anchor_offsets(
        tracked_frames, max_horizon, anchor_count
    )
    anchor_set = set(anchor_offsets.tolist())
    if stop > len(ground_truth) or not valid[start]:
        raise ValueError("Invalid clip bounds or initialization")

    initial = np.asarray(ground_truth[start], dtype=np.float64).copy()
    rt.initialize(rt.read(frame_path(directory, dataset, start)), initial)
    baseline_boxes = np.empty((tracked_frames, 4), dtype=np.float64)
    anchor_records = []
    control_probe_boxes = None
    l9_verified = False

    with rt.torch.no_grad():
        for offset in range(1, tracked_frames + 1):
            absolute_frame = start + offset
            image = rt.read(frame_path(directory, dataset, absolute_frame))
            previous_state = np.asarray(rt.tracker.state, dtype=np.float64).copy()
            prediction = _fast_prediction(rt, image, previous_state)
            fast_box = prediction["fast_box"]

            if offset in anchor_set:
                probabilities = rt.selector(prediction["tokens"])
                l9_output = rt.candidate(prediction["tokens"], 9)
                l9_box = np.asarray(
                    rt.decode(l9_output, prediction["factor"], image),
                    dtype=np.float64,
                )
                if verify_l9 and not l9_verified:
                    rt.tracker.state = previous_state.tolist()
                    rt.tracker.frame_id = offset - 1
                    integrated_fast = np.asarray(
                        rt.run_image(image, "fast"), dtype=np.float64
                    )
                    if not np.allclose(
                        integrated_fast, fast_box, rtol=0, atol=1e-6
                    ):
                        raise RuntimeError(
                            "Integrated Fast path disagrees with branch output"
                        )
                    rt.tracker.state = previous_state.tolist()
                    rt.tracker.frame_id = offset - 1
                    integrated = np.asarray(
                        rt.run_image(image, "fixed_l9"), dtype=np.float64
                    )
                    if not np.allclose(
                        integrated, l9_box, rtol=0, atol=1e-6
                    ):
                        raise RuntimeError(
                            "Integrated fixed-L9 path disagrees with branch output"
                        )
                    l9_verified = True

                if control_probe_boxes is None:
                    replay = [fast_box.copy()]
                    replay_state = fast_box.copy()
                    for step in range(1, max_horizon):
                        future_image = rt.read(
                            frame_path(
                                directory,
                                dataset,
                                absolute_frame + step,
                            )
                        )
                        rt.tracker.state = replay_state.tolist()
                        rt.tracker.frame_id = offset + step - 1
                        replay_state = np.asarray(
                            rt.run_image(future_image, "fast"), dtype=np.float64
                        )
                        replay.append(replay_state.copy())
                    control_probe_boxes = np.stack(replay)

                treatment_boxes = [l9_box]
                treatment_state = l9_box.copy()
                for step in range(1, max_horizon):
                    future_image = rt.read(
                        frame_path(
                            directory,
                            dataset,
                            absolute_frame + step,
                        )
                    )
                    rt.tracker.state = treatment_state.tolist()
                    rt.tracker.frame_id = offset + step - 1
                    treatment_state = np.asarray(
                        rt.run_image(future_image, "fast"), dtype=np.float64
                    )
                    treatment_boxes.append(treatment_state.copy())

                anchor_records.append(
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
                        "selected_layer": np.int64(
                            rt.selected_layer(probabilities)
                        ),
                        "previous_state": previous_state,
                        "previous_fast_box": previous_state.copy(),
                        "fast_box": fast_box.copy(),
                        "l9_box": l9_box.copy(),
                        "image_hw": np.asarray(image.shape[:2], dtype=np.int64),
                        "resize_factor": np.float64(prediction["factor"]),
                        "treatment_boxes": np.stack(treatment_boxes),
                    }
                )
                print(
                    "[branch] {}: anchor {}/{} at {}/{}".format(
                        entry["sequence"],
                        len(anchor_records),
                        anchor_count,
                        offset,
                        tracked_frames,
                    ),
                    flush=True,
                )

            baseline_boxes[offset - 1] = fast_box
            rt.tracker.state = fast_box.tolist()
            rt.tracker.frame_id = offset

    if len(anchor_records) != anchor_count:
        raise RuntimeError(
            "Collected {} anchors, expected {}".format(
                len(anchor_records), anchor_count
            )
        )
    baseline_gt = np.asarray(
        ground_truth[start + 1 : stop], dtype=np.float64
    ).copy()
    baseline_valid = np.asarray(valid[start + 1 : stop], dtype=bool).copy()
    baseline_iou = tracking_iou(
        baseline_boxes, baseline_gt, baseline_valid
    )
    control_boxes = np.stack(
        [
            baseline_boxes[
                offset - 1 : offset - 1 + max_horizon
            ]
            for offset in anchor_offsets
        ]
    )
    treatment_boxes = np.stack(
        [record["treatment_boxes"] for record in anchor_records]
    )
    gt_windows = np.stack(
        [
            baseline_gt[offset - 1 : offset - 1 + max_horizon]
            for offset in anchor_offsets
        ]
    )
    valid_windows = np.stack(
        [
            baseline_valid[offset - 1 : offset - 1 + max_horizon]
            for offset in anchor_offsets
        ]
    )
    control_iou = tracking_iou(control_boxes, gt_windows, valid_windows)
    treatment_iou = tracking_iou(
        treatment_boxes, gt_windows, valid_windows
    )
    labels = intervention_labels(
        control_iou, treatment_iou, valid_windows, horizons
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
        "l9_box",
        "image_hw",
        "resize_factor",
    )
    arrays = {
        key: np.stack([record[key] for record in anchor_records])
        for key in anchor_fields
    }
    arrays.update(
        {
            "baseline_boxes": baseline_boxes,
            "baseline_gt": baseline_gt,
            "baseline_valid_gt": baseline_valid,
            "baseline_iou": baseline_iou,
            "control_boxes": control_boxes,
            "control_probe_boxes": control_probe_boxes,
            "treatment_boxes": treatment_boxes,
            "gt": gt_windows,
            "valid_gt": valid_windows,
            "control_iou": control_iou,
            "treatment_iou": treatment_iou,
            "delta_iou": treatment_iou - control_iou,
            "horizons": np.asarray(horizons, dtype=np.int64),
        }
    )
    arrays.update(labels)
    return arrays, initial.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pool_grid", type=int, default=4)
    parser.add_argument("--anchors", type=int, default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--verify_l9", action="store_true")
    parser.add_argument("--split", choices=("all", "train", "val"), default="all")
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Require 0 <= shard_id < num_shards")
    if args.pool_grid < 1:
        raise ValueError("--pool_grid must be positive")
    if args.max_sequences is not None and args.max_sequences < 1:
        raise ValueError("--max_sequences must be positive")

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    design = manifest.get("intervention_design", {})
    anchor_count = args.anchors or design.get("anchor_count", 12)
    horizons = tuple(
        sorted(
            set(
                args.horizons
                or design.get("horizons", list(DEFAULT_HORIZONS))
            )
        )
    )
    if not horizons or horizons[0] < 1:
        raise ValueError("--horizons must be positive")
    if args.anchors is not None and design.get("anchor_count") != args.anchors:
        raise ValueError("--anchors must match the frozen manifest design")
    if args.horizons is not None and list(horizons) != design.get("horizons"):
        raise ValueError("--horizons must match the frozen manifest design")

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
        horizons,
        args.verify_l9,
    )
    publish_run(output, run)
    print("Manifest: {}".format(manifest_path))
    print("Cache: {}".format(output))
    print("Run fingerprint: {}".format(run["fingerprint"]))
    print(
        "Intervention: {} anchors, horizons {}, one L9 then Fast".format(
            anchor_count, "/".join(str(value) for value in horizons)
        )
    )
    print(
        "Shard {}/{}: {} clip(s)".format(
            args.shard_id, args.num_shards, len(entries)
        )
    )

    completed = 0
    for entry in entries:
        path = intervention_cache_path(output, entry)
        if path.exists() and not args.overwrite:
            report, _, _ = load_and_audit_cache(path, entry, run)
            print(
                "[skip, checked] {}: {} anchors".format(
                    path, report["anchors"]
                ),
                flush=True,
            )
            continue
        directory, ground_truth, valid = load_annotations(
            manifest["root"], manifest["dataset"], entry["sequence"]
        )
        if (
            file_hash(directory / "groundtruth.txt")
            != entry["annotation_sha256"]
            or fingerprint(valid.tolist()) != entry["visible_sha256"]
        ):
            raise ValueError(
                "Annotations changed since manifest creation: {}".format(
                    entry["sequence"]
                )
            )
        arrays, initial = collect_interventions(
            rt,
            directory,
            manifest["dataset"],
            entry,
            ground_truth,
            valid,
            args.pool_grid,
            anchor_count,
            horizons,
            args.verify_l9,
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
            "[done] {}: {} anchors x {} frames -> {}".format(
                entry["sequence"], anchor_count, max(horizons), path
            ),
            flush=True,
        )
    print("Newly completed intervention clips: {}".format(completed))
    print(
        'Next: python tracking/stage8_intervention_summary.py --input "{}"'.format(
            output
        )
    )


if __name__ == "__main__":
    main()
