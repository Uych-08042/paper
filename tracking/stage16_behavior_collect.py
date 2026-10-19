"""Collect randomized continuous Fast/L7-L12 behavior trajectories.

Unlike Stage-14, every chosen action is written into the tracker state used to
crop the next frame. The resulting cache contains real sequential transitions
for a later conservative offline-RL screen.
"""

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
from stage16_behavior_common import (
    ACTION_NAMES,
    CANDIDATE_LAYERS,
    DEFAULT_BEHAVIOR_RATES,
    behavior_cache_path,
    behavior_seed,
    load_and_audit_cache,
    make_behavior_schedule,
    make_run,
)


def _spatial_state(rt, tokens, grid):
    if hasattr(rt, "spatial_state"):
        return rt.spatial_state(tokens, grid)
    template = tokens[:, : rt.lens_z]
    search = tokens[:, rt.lens_z :].transpose(1, 2).reshape(
        tokens.shape[0], tokens.shape[2], rt.search_side, rt.search_side
    )
    search = rt.torch.nn.functional.adaptive_avg_pool2d(search, (grid, grid))
    search = search.permute(0, 2, 3, 1)
    template_summary = rt.torch.stack(
        (template.mean(1), template.max(1).values), dim=1
    )
    return template_summary, search


def _predict_all_actions(rt, image, state, state_grid):
    rt.tracker.state = list(state)
    search, factor = rt.prepare(image)
    tokens = rt.prefix(search)
    fast_output = rt.head(tokens)
    probabilities = rt.selector(tokens)
    boxes = [np.asarray(rt.decode(fast_output, factor, image), dtype=np.float64)]
    for layer in CANDIDATE_LAYERS:
        output = rt.candidate(tokens, layer)
        boxes.append(np.asarray(rt.decode(output, factor, image), dtype=np.float64))
    template_summary, search_grid = _spatial_state(rt, tokens, state_grid)
    return {
        "search": search,
        "factor": factor,
        "tokens": tokens,
        "template_summary": template_summary,
        "search_token_grid": search_grid,
        "fast_output": fast_output,
        "selector_prob": probabilities,
        "selected_layer": rt.selected_layer(probabilities),
        "action_boxes": np.stack(boxes),
    }


def _rerun_candidate(rt, image, state, layer):
    rt.tracker.state = list(state)
    search, factor = rt.prepare(image)
    tokens = rt.prefix(search)
    output = rt.candidate(tokens, layer)
    return np.asarray(rt.decode(output, factor, image), dtype=np.float64)


def _verify_action_paths(rt, image, previous_state, prediction, frame_offset):
    rt.tracker.state = list(previous_state)
    rt.tracker.frame_id = frame_offset - 1
    integrated_fast = np.asarray(rt.run_image(image, "fast"), dtype=np.float64)
    if not np.allclose(integrated_fast, prediction["action_boxes"][0], rtol=0, atol=1e-6):
        raise RuntimeError("Integrated Fast path disagrees with Stage-16 action table")
    for action, layer in enumerate(CANDIDATE_LAYERS, start=1):
        rerun = _rerun_candidate(rt, image, previous_state, layer)
        if not np.allclose(rerun, prediction["action_boxes"][action], rtol=0, atol=1e-6):
            raise RuntimeError("Rerun L{} path disagrees with Stage-16 action table".format(layer))


def _collect_episode(
    rt,
    directory,
    dataset,
    entry,
    ground_truth,
    valid,
    state_grid,
    schedule,
    save_full_tokens=False,
    verify_actions=False,
):
    start, stop = entry["start"], entry["stop"]
    steps = stop - start - 1
    actions = np.asarray(schedule["action"], dtype=np.int64)
    if actions.shape != (steps,):
        raise ValueError("Behavior schedule length does not match the clip")
    initial = np.asarray(ground_truth[start], dtype=np.float64).copy()
    rt.initialize(rt.read(frame_path(directory, dataset, start)), initial)
    previous_fast = initial.copy()
    previous_action = -1
    rows = []

    with rt.torch.no_grad():
        for step, absolute_frame in enumerate(range(start + 1, stop)):
            image = rt.read(frame_path(directory, dataset, absolute_frame))
            previous_state = np.asarray(rt.tracker.state, dtype=np.float64).copy()
            prediction = _predict_all_actions(rt, image, previous_state, state_grid)
            if verify_actions and step == 0:
                _verify_action_paths(rt, image, previous_state, prediction, step + 1)

            action = int(actions[step])
            boxes = prediction["action_boxes"]
            action_iou = tracking_iou(
                boxes,
                np.asarray(ground_truth[absolute_frame], dtype=np.float64)[None, :],
                bool(valid[absolute_frame]),
            )
            executed_box = boxes[action].copy()
            row = {
                "template_summary": cpu_batch(prediction["template_summary"], np.float16),
                "search_token_grid": cpu_batch(prediction["search_token_grid"], np.float16),
                "fast_score_map": cpu_batch(prediction["fast_output"]["score_map"], np.float16),
                "selector_prob": cpu_batch(prediction["selector_prob"], np.float32),
                "selected_layer": np.int64(prediction["selected_layer"]),
                "previous_state": previous_state,
                "previous_fast_box": previous_fast.copy(),
                "previous_action": np.int64(previous_action),
                "fast_box": boxes[0].copy(),
                "action_boxes": boxes.copy(),
                "executed_box": executed_box,
                "action_iou": action_iou,
                "executed_iou": np.float64(action_iou[action]),
                "reward_gain_vs_fast": np.float64(action_iou[action] - action_iou[0]),
                "resize_factor": np.float64(prediction["factor"]),
            }
            if save_full_tokens:
                row["full_tokens"] = cpu_batch(prediction["tokens"], np.float16)
            rows.append(row)

            rt.tracker.state = executed_box.tolist()
            rt.tracker.frame_id = step + 1
            previous_fast = boxes[0].copy()
            previous_action = action

    return {key: np.stack([row[key] for row in rows]) for key in rows[0]}, initial


def collect_continuous_behavior(
    rt,
    directory,
    dataset,
    entry,
    ground_truth,
    valid,
    run,
):
    """Collect all frozen behavior episodes for one independently initialized clip."""
    start, stop = entry["start"], entry["stop"]
    steps = stop - start - 1
    if stop > len(ground_truth) or not valid[start]:
        raise ValueError("Invalid clip bounds or initialization")
    episode_arrays = []
    initial = None
    for episode, spec in enumerate(run["behaviors"]):
        seed = behavior_seed(
            run["behavior_seed"],
            run["manifest"]["fingerprint"],
            entry["sequence"],
            spec["behavior_id"],
        )
        schedule = make_behavior_schedule(
            steps, spec["candidate_rate_percent"], seed
        )
        values, episode_initial = _collect_episode(
            rt,
            directory,
            dataset,
            entry,
            ground_truth,
            valid,
            run["state_grid"],
            schedule,
            save_full_tokens=run["save_full_tokens"],
            verify_actions=run["verify_actions"] and episode == 0,
        )
        if initial is None:
            initial = episode_initial
        elif not np.allclose(initial, episode_initial, rtol=0, atol=0):
            raise RuntimeError("Behavior episodes did not share initialization")
        values.update(
            {
                "behavior_action": schedule["action"],
                "action_cost": schedule["action_cost"],
                "action_propensity": schedule["action_propensity"],
                "behavior_probability": schedule["behavior_probability"],
                "candidate_used_before": schedule["candidate_used_before"],
                "candidate_remaining_before": schedule["candidate_remaining_before"],
                "remaining_candidate_rate": schedule["remaining_candidate_rate"],
            }
        )
        episode_arrays.append(values)
        print(
            "[trajectory] {}: episode {}/{} {} ({:.2f}% candidate, {} actions)".format(
                entry["sequence"],
                episode + 1,
                len(run["behaviors"]),
                spec["behavior_id"],
                100.0 * np.mean(schedule["action"] > 0),
                steps,
            ),
            flush=True,
        )

    per_episode_fields = tuple(episode_arrays[0])
    arrays = {
        key: np.stack([values[key] for values in episode_arrays])
        for key in per_episode_fields
    }
    arrays.update(
        {
            "frame_id": np.arange(start + 1, stop, dtype=np.int64),
            "gt": np.asarray(ground_truth[start + 1 : stop], dtype=np.float64).copy(),
            "valid_gt": np.asarray(valid[start + 1 : stop], dtype=bool).copy(),
            "image_hw": np.stack(
                [
                    np.asarray(
                        rt.read(frame_path(directory, dataset, frame_id)).shape[:2],
                        dtype=np.int64,
                    )
                    for frame_id in range(start + 1, stop)
                ]
            ),
            "done": np.arange(steps, dtype=np.int64) == steps - 1,
        }
    )
    return arrays, initial.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--state_grid", type=int, default=8)
    parser.add_argument(
        "--behavior_rates",
        nargs="+",
        type=float,
        default=list(DEFAULT_BEHAVIOR_RATES),
    )
    parser.add_argument("--behavior_repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2046)
    parser.add_argument("--save_full_tokens", action="store_true")
    parser.add_argument("--verify_actions", action="store_true")
    parser.add_argument("--split", choices=("all", "train", "val"), default="all")
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not str(args.output_dir).strip():
        raise ValueError("--output_dir is empty; use a dedicated Stage-16 cache directory")
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Require 0 <= shard_id < num_shards")
    if args.state_grid < 1 or args.behavior_repeats < 1:
        raise ValueError("state_grid and behavior_repeats must be positive")
    if args.max_sequences is not None and args.max_sequences < 1:
        raise ValueError("--max_sequences must be positive")

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = load_manifest(manifest_path)
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

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and not (output / "run.json").is_file():
        unidentified = next(output.rglob("*.npz"), None)
        if unidentified is not None:
            raise ValueError(
                "Stage-16 output has cache files but no run.json: {}. Use a new dedicated directory.".format(output)
            )

    rt = Runtime(args.parameter, args.gpu, args.checkpoint)
    if args.state_grid > rt.search_side:
        raise ValueError("state_grid exceeds the search token grid")
    run = make_run(
        manifest,
        rt.identity,
        args.state_grid,
        args.behavior_rates,
        args.behavior_repeats,
        args.seed,
        args.verify_actions,
        args.save_full_tokens,
    )
    publish_run(output, run)
    print("Manifest: {}".format(manifest_path))
    print("Cache: {}".format(output))
    print("Run fingerprint: {}".format(run["fingerprint"]))
    print(
        "Stage-16A: {} continuous episodes per clip; actions={}; state_grid={}x{}".format(
            len(run["behaviors"]),
            "/".join(ACTION_NAMES),
            args.state_grid,
            args.state_grid,
        )
    )
    print("Shard {}/{}: {} clip(s)".format(args.shard_id, args.num_shards, len(entries)))

    completed = 0
    for entry in entries:
        path = behavior_cache_path(output, entry)
        if path.exists() and not args.overwrite:
            report, _, _ = load_and_audit_cache(path, entry, run)
            print(
                "[skip, checked] {}: {} transitions".format(path, report["transitions"]),
                flush=True,
            )
            continue
        directory, ground_truth, valid = load_annotations(
            manifest["root"], manifest["dataset"], entry["sequence"]
        )
        if (
            file_hash(directory / "groundtruth.txt") != entry["annotation_sha256"]
            or fingerprint(valid.tolist()) != entry["visible_sha256"]
        ):
            raise ValueError("Annotations changed since manifest creation: {}".format(entry["sequence"]))
        arrays, initial = collect_continuous_behavior(
            rt,
            directory,
            manifest["dataset"],
            entry,
            ground_truth,
            valid,
            run,
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
        report, _, _ = load_and_audit_cache(path, entry, run)
        completed += 1
        print(
            "[done] {}: {} episodes, {} transitions -> {}".format(
                entry["sequence"], report["episodes"], report["transitions"], path
            ),
            flush=True,
        )
    print("Newly completed Stage-16 clips: {}".format(completed))
    print(
        'Next: python tracking/stage16_behavior_audit.py --input "{}" --output_dir <audit-dir>'.format(output)
    )


if __name__ == "__main__":
    main()
