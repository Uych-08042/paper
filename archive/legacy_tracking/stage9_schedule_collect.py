"""Collect true equal-cost L9 schedule branches from a Fast trajectory."""

import argparse
from pathlib import Path

import numpy as np

from stage4_collect import publish_run
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
from stage9_schedule_common import (
    DEFAULT_EVALUATION_HORIZONS,
    DEFAULT_HORIZON,
    POLICIES,
    PRIMARY_COMPARISON,
    PRIMARY_HORIZON,
    load_and_audit_cache,
    make_run,
    schedule_cache_path,
    schedule_labels,
    validate_schedules,
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
        "fast_box": np.asarray(fast_box, dtype=np.float64),
    }


def _branch_policy(
    rt,
    directory,
    dataset,
    absolute_frame,
    tracker_offset,
    previous_state,
    schedule,
    horizon,
    first_l9_box,
):
    state = np.asarray(previous_state, dtype=np.float64).copy()
    boxes = []
    for step in range(horizon):
        if step == 0:
            state = np.asarray(first_l9_box, dtype=np.float64).copy()
        else:
            image = rt.read(
                frame_path(directory, dataset, absolute_frame + step)
            )
            rt.tracker.state = state.tolist()
            rt.tracker.frame_id = tracker_offset + step - 1
            mode = "fixed_l9" if step in schedule else "fast"
            state = np.asarray(rt.run_image(image, mode), dtype=np.float64)
        boxes.append(state.copy())
    return np.stack(boxes)


def _fast_replay(
    rt,
    directory,
    dataset,
    absolute_frame,
    tracker_offset,
    previous_state,
    horizon,
):
    state = np.asarray(previous_state, dtype=np.float64).copy()
    boxes = []
    for step in range(horizon):
        image = rt.read(frame_path(directory, dataset, absolute_frame + step))
        rt.tracker.state = state.tolist()
        rt.tracker.frame_id = tracker_offset + step - 1
        state = np.asarray(rt.run_image(image, "fast"), dtype=np.float64)
        boxes.append(state.copy())
    return np.stack(boxes)


def collect_schedules(
    rt,
    directory,
    dataset,
    entry,
    ground_truth,
    valid,
    anchor_count,
    horizon,
    evaluation_horizons,
    schedules,
    verify_l9=False,
):
    """Run one Fast baseline and fork every frozen schedule at each anchor."""
    action_mask = validate_schedules(schedules, horizon)
    start, stop = entry["start"], entry["stop"]
    tracked_frames = stop - start - 1
    anchor_offsets = select_anchor_offsets(tracked_frames, horizon, anchor_count)
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
                    integrated_l9 = np.asarray(
                        rt.run_image(image, "fixed_l9"), dtype=np.float64
                    )
                    if not np.allclose(
                        integrated_l9, l9_box, rtol=0, atol=1e-6
                    ):
                        raise RuntimeError(
                            "Integrated fixed-L9 path disagrees with branch output"
                        )
                    l9_verified = True

                if control_probe_boxes is None:
                    control_probe_boxes = _fast_replay(
                        rt,
                        directory,
                        dataset,
                        absolute_frame,
                        offset,
                        previous_state,
                        horizon,
                    )

                treatment_boxes = []
                for policy in POLICIES[1:]:
                    treatment_boxes.append(
                        _branch_policy(
                            rt,
                            directory,
                            dataset,
                            absolute_frame,
                            offset,
                            previous_state,
                            schedules[policy],
                            horizon,
                            l9_box,
                        )
                    )
                anchor_records.append(
                    {
                        "frame_id": np.int64(absolute_frame),
                        "anchor_offset": np.int64(offset),
                        "anchor_fraction": np.float64(offset / tracked_frames),
                        "previous_state": previous_state,
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
    baseline_iou = tracking_iou(baseline_boxes, baseline_gt, baseline_valid)
    control_boxes = np.stack(
        [
            baseline_boxes[offset - 1 : offset - 1 + horizon]
            for offset in anchor_offsets
        ]
    )
    treatment_boxes = np.stack(
        [record["treatment_boxes"] for record in anchor_records]
    )
    policy_boxes = np.concatenate((control_boxes[:, None], treatment_boxes), axis=1)
    gt_windows = np.stack(
        [
            baseline_gt[offset - 1 : offset - 1 + horizon]
            for offset in anchor_offsets
        ]
    )
    valid_windows = np.stack(
        [
            baseline_valid[offset - 1 : offset - 1 + horizon]
            for offset in anchor_offsets
        ]
    )
    policy_iou = tracking_iou(
        policy_boxes, gt_windows[:, None, :, :], valid_windows[:, None, :]
    )
    labels = schedule_labels(policy_iou, valid_windows, evaluation_horizons)
    arrays = {
        "frame_id": np.stack([record["frame_id"] for record in anchor_records]),
        "anchor_offset": np.stack(
            [record["anchor_offset"] for record in anchor_records]
        ),
        "anchor_fraction": np.stack(
            [record["anchor_fraction"] for record in anchor_records]
        ),
        "previous_state": np.stack(
            [record["previous_state"] for record in anchor_records]
        ),
        "baseline_boxes": baseline_boxes,
        "baseline_gt": baseline_gt,
        "baseline_valid_gt": baseline_valid,
        "baseline_iou": baseline_iou,
        "control_probe_boxes": control_probe_boxes,
        "action_mask": action_mask,
        "policy_boxes": policy_boxes,
        "gt": gt_windows,
        "valid_gt": valid_windows,
        "policy_iou": policy_iou,
        "horizons": np.asarray(evaluation_horizons, dtype=np.int64),
    }
    arrays.update(labels)
    return arrays, initial.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
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
    if args.max_sequences is not None and args.max_sequences < 1:
        raise ValueError("--max_sequences must be positive")

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    design = manifest.get("schedule_design", {})
    if (
        design.get("policies") != list(POLICIES)
        or design.get("horizon") != DEFAULT_HORIZON
        or design.get("evaluation_horizons")
        != list(DEFAULT_EVALUATION_HORIZONS)
        or design.get("primary_horizon") != PRIMARY_HORIZON
        or design.get("primary_comparison") != list(PRIMARY_COMPARISON)
    ):
        raise ValueError("Manifest does not contain the frozen Stage-9 policy set")
    schedules = {
        policy: tuple(int(value) for value in design["schedules"][policy])
        for policy in POLICIES
    }
    anchor_count = int(design["anchor_count"])
    horizon = int(design["horizon"])
    evaluation_horizons = tuple(
        int(value) for value in design["evaluation_horizons"]
    )
    validate_schedules(schedules, horizon)
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
    output = Path(args.output_dir).expanduser().resolve()
    run = make_run(
        manifest,
        rt.identity,
        anchor_count,
        horizon,
        evaluation_horizons,
        schedules,
        args.verify_l9,
    )
    publish_run(output, run)
    print("Manifest: {}".format(manifest_path))
    print("Cache: {}".format(output))
    print("Run fingerprint: {}".format(run["fingerprint"]))
    print(
        "Stage-9: {} anchors x H{}; treatment cost={} L9 blocks".format(
            anchor_count, horizon, run["equal_treatment_blocks"]
        )
    )
    print(
        "Schedules: {}".format(
            " | ".join(
                "{}={}".format(
                    policy,
                    ",".join(str(value) for value in schedules[policy]) or "none",
                )
                for policy in POLICIES
            )
        )
    )
    print(
        "Shard {}/{}: {} clip(s)".format(
            args.shard_id, args.num_shards, len(entries)
        )
    )

    completed = 0
    for entry in entries:
        path = schedule_cache_path(output, entry)
        if path.exists() and not args.overwrite:
            report, _, _ = load_and_audit_cache(path, entry, run)
            print(
                "[skip, checked] {}: {} anchors".format(path, report["anchors"]),
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
        arrays, initial = collect_schedules(
            rt,
            directory,
            manifest["dataset"],
            entry,
            ground_truth,
            valid,
            anchor_count,
            horizon,
            evaluation_horizons,
            schedules,
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
            "[done] {}: {} anchors x {} policies x H{} -> {}".format(
                entry["sequence"], anchor_count, len(POLICIES), horizon, path
            ),
            flush=True,
        )
    print("Newly completed Stage-9 clips: {}".format(completed))
    print(
        'Next: python tracking/stage9_schedule_summary.py --input "{}"'.format(
            output
        )
    )


if __name__ == "__main__":
    main()
