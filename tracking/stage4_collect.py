"""Collect frozen pre-candidate features and utility labels on training videos."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from stage4_common import (
    ACTIONS, IOU_PROTOCOL, SCHEMA, audit_cache, cache_path, file_hash, fingerprint,
    frame_path, load_annotations, load_manifest, tracking_iou, write_cache, write_json,
)
from stage4_runtime import Runtime, add_runtime_arguments


def publish_run(output, run):
    """Serialize the small run manifest, including concurrent shard startup."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "run.lock"
    deadline = time.monotonic() + 30
    while True:
        try:
            handle = lock.open("x")
            handle.close()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RuntimeError("run.lock persists; check for a running collector before removing it")
            time.sleep(0.1)
    try:
        path = output / "run.json"
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            if old != run:
                raise ValueError("Run configuration changed; use a NEW output_dir, even with --overwrite")
        else:
            if any(output.rglob("*.npz")):
                raise ValueError("Unidentified cache files exist; use an empty output_dir")
            write_json(path, run)
    finally:
        lock.unlink()


def cpu_batch(tensor, dtype=np.float32):
    return tensor.detach().float().cpu().numpy()[0].astype(dtype, copy=True)


def collect_clip(rt, directory, dataset, entry, gt, valid, trajectory, pool_grid,
                 save_full_tokens=False, verify_sgla=False):
    start, stop = entry["start"], entry["stop"]
    if trajectory not in ("sgla", "fast") or stop > len(gt) or not valid[start]:
        raise ValueError("Invalid rollout policy, clip range or initialization")
    initial = gt[start].tolist()
    rt.initialize(rt.read(frame_path(directory, dataset, start)), initial)
    previous_fast = np.asarray(initial, dtype=np.float64)
    previous_action = -1
    rows = []
    with rt.torch.no_grad():
        for frame_id in range(start + 1, stop):
            image = rt.read(frame_path(directory, dataset, frame_id))
            previous_state = np.asarray(rt.tracker.state, dtype=np.float64).copy()
            search, factor = rt.prepare(image)
            tokens = rt.prefix(search)
            fast = rt.head(tokens)
            probabilities = rt.selector(tokens)
            layer = rt.selected_layer(probabilities)
            # Capture inputs before evaluating candidates. GT is used only below
            # for labels, never for crops, history, features or action selection.
            token_features = cpu_batch(rt.pool(tokens, pool_grid), np.float16)
            fast_score_map = cpu_batch(fast["score_map"], np.float16)
            boxes = [rt.decode(fast, factor, image)]
            for candidate_layer in range(7, 13):
                candidate = rt.candidate(tokens, candidate_layer)
                if verify_sgla and frame_id == start + 1 and candidate_layer == layer:
                    rt.verify(search, candidate)
                boxes.append(rt.decode(candidate, factor, image))
            boxes = np.asarray(boxes, dtype=np.float64)
            executed = 0 if trajectory == "fast" else layer - 6
            iou = tracking_iou(boxes, gt[frame_id], bool(valid[frame_id]))
            row = {
                "frame_id": np.int64(frame_id), "token_features": token_features,
                "fast_score_map": fast_score_map, "selector_prob": cpu_batch(probabilities),
                "previous_state": previous_state, "previous_fast_box": previous_fast.copy(),
                "previous_action": np.int64(previous_action), "fast_box": boxes[0].copy(),
                "selected_layer": np.int64(layer), "action_boxes": boxes,
                "gt": np.asarray(gt[frame_id], dtype=np.float64).copy(),
                "valid_gt": np.bool_(valid[frame_id]), "iou": iou,
                "delta": iou[1:] - iou[0] if valid[frame_id] else np.full(6, np.nan),
                "executed_action": np.int64(executed),
                "image_hw": np.asarray(image.shape[:2], dtype=np.int64),
                "resize_factor": np.float64(factor),
            }
            if save_full_tokens:
                row["full_tokens"] = cpu_batch(tokens, np.float16)
            rows.append(row)
            rt.tracker.state = boxes[executed].tolist()
            rt.tracker.frame_id = frame_id - start
            previous_fast = boxes[0].copy()
            previous_action = executed
            if (frame_id - start) % 50 == 0:
                print("[collect] {}/{}: {}/{} decisions".format(
                    entry["sequence"], trajectory, frame_id - start, stop - start - 1), flush=True)
    return {key: np.stack([row[key] for row in rows]) for key in rows[0]}, initial


def make_run(manifest, runtime_identity, pool_grid, save_full_tokens, trajectories, verify_sgla):
    directory = Path(__file__).resolve().parent
    files = ("stage4_collect.py", "stage4_common.py", "stage4_runtime.py",
             "validation_metrics.py", "stage1_validate.py")
    run = {
        "schema": SCHEMA, "manifest": manifest, "runtime": runtime_identity,
        "pool_grid": pool_grid, "save_full_tokens": save_full_tokens,
        "trajectories": sorted(trajectories), "verify_sgla": verify_sgla,
        "actions": list(ACTIONS), "iou_protocol": IOU_PROTOCOL,
        "code_sha256": {name: file_hash(directory / name) for name in files},
        "feature_fields": ["token_features", "fast_score_map", "selector_prob", "previous_state",
                           "previous_fast_box", "previous_action", "fast_box", "image_hw", "resize_factor"],
        "label_fields": ["gt", "valid_gt", "iou", "delta", "action_boxes"],
    }
    if save_full_tokens:
        run["feature_fields"].append("full_tokens")
    run["fingerprint"] = fingerprint(run)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--trajectories", nargs="+", choices=("sgla", "fast"), default=["sgla", "fast"])
    parser.add_argument("--pool_grid", type=int, default=4)
    parser.add_argument("--save_full_tokens", action="store_true")
    parser.add_argument("--verify_sgla", action="store_true")
    parser.add_argument("--split", choices=("all", "train", "val"), default="all")
    parser.add_argument("--sequence", default=None, help="Exact training sequence name from manifest")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Require 0 <= shard_id < num_shards")
    if args.pool_grid < 1 or len(set(args.trajectories)) != len(args.trajectories):
        raise ValueError("Positive pool grid and unique trajectories required")
    manifest = load_manifest(args.manifest)
    entries = [entry for entry in manifest["sequences"]
               if (args.split == "all" or entry["split"] == args.split)
               and (args.sequence is None or entry["sequence"] == args.sequence)]
    if not entries:
        raise ValueError("No videos match --split/--sequence")
    entries = entries[args.shard_id::args.num_shards]
    if not entries:
        print("This shard has no videos.")
        return
    rt = Runtime(args.parameter, args.gpu, args.checkpoint)
    if args.pool_grid > rt.search_side:
        raise ValueError("pool_grid exceeds the search token grid")
    output = Path(args.output_dir).expanduser().resolve()
    run = make_run(manifest, rt.identity, args.pool_grid, args.save_full_tokens,
                   args.trajectories, args.verify_sgla)
    publish_run(output, run)
    completed = 0
    print("Cache: {}\nRun fingerprint: {}".format(output, run["fingerprint"]))
    for entry in entries:
        directory, gt, valid = load_annotations(manifest["root"], manifest["dataset"], entry["sequence"])
        if (file_hash(directory / "groundtruth.txt") != entry["annotation_sha256"]
                or fingerprint(valid.tolist()) != entry["visible_sha256"]):
            raise ValueError("Annotations changed since manifest creation: {}".format(entry["sequence"]))
        for trajectory in args.trajectories:
            path = cache_path(output, entry, trajectory)
            if path.exists() and not args.overwrite:
                audit_cache(path, entry, trajectory, run)
                print("[skip, checked] {}".format(path), flush=True)
                continue
            arrays, initial = collect_clip(rt, directory, manifest["dataset"], entry, gt, valid,
                                          trajectory, args.pool_grid, args.save_full_tokens, args.verify_sgla)
            write_cache(path, arrays, {"run_fingerprint": run["fingerprint"], "entry": entry,
                                       "trajectory": trajectory, "init_box": initial})
            audit_cache(path, entry, trajectory, run)
            completed += 1
            print("[done] {}: {} decisions -> {}".format(
                entry["sequence"], len(arrays["frame_id"]), path), flush=True)
    print("Newly completed video-policy clips: {}".format(completed))
    print('Next: python tracking/stage4_check_cache.py --input "{}"'.format(output))


if __name__ == "__main__":
    main()
