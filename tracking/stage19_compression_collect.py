"""Collect paired same-crop and autonomous persistent FFN compression branches."""

import argparse
from pathlib import Path

import numpy as np

from stage4_collect import publish_run
from stage4_common import file_hash, fingerprint, frame_path, load_annotations, tracking_iou, write_cache
from stage4_runtime import Runtime, add_runtime_arguments
from stage8_intervention_common import select_anchor_offsets
from stage19_compression_common import cache_path, load_and_audit_cache, make_run, nonempty_path, validate_manifest
from stage19_ffn_compression import compress_ffn


def predict(rt, image, previous, offset, verify=False):
    rt.tracker.state = np.asarray(previous, dtype=np.float64).tolist()
    rt.tracker.frame_id = int(offset) - 1
    search, factor = rt.prepare(image)
    tokens = rt.prefix(search)
    layer = rt.selected_layer(rt.selector(tokens))
    output = rt.candidate(tokens, layer)
    if verify:
        rt.verify(search, output)
    box = np.asarray(rt.decode(output, factor, image), dtype=np.float64)
    return box, layer


def rollout(rt, images, reference_states, offset, synchronized):
    state = reference_states[0].copy()
    states, boxes, layers = [], [], []
    for step, image in enumerate(images):
        previous = reference_states[step].copy() if synchronized else state.copy()
        state, layer = predict(rt, image, previous, offset + step)
        states.append(previous)
        boxes.append(state.copy())
        layers.append(layer)
    return np.stack(states), np.stack(boxes), np.asarray(layers, dtype=np.int64)


def collect_clip(rt, directory, dataset, entry, gt, valid, run, surgery=compress_ffn):
    a, h = run["design"]["anchors"], run["design"]["horizon"]
    masks = run["masks"]
    start, stop = entry["start"], entry["stop"]
    n = stop - start - 1
    if stop > len(gt) or not valid[start]:
        raise ValueError("Invalid clip bounds or initialization")
    offsets = select_anchor_offsets(n, h, a)
    initial = np.asarray(gt[start], dtype=np.float64).copy()
    baseline_boxes, baseline_layers = [], []
    rt.initialize(rt.read(frame_path(directory, dataset, start)), initial.tolist())
    with rt.torch.no_grad():
        state = initial.copy()
        for offset in range(1, n + 1):
            image = rt.read(frame_path(directory, dataset, start + offset))
            state, layer = predict(rt, image, state, offset, verify=(offset == 1))
            baseline_boxes.append(state.copy())
            baseline_layers.append(layer)
        baseline_boxes = np.stack(baseline_boxes)
        baseline_states = np.concatenate((initial[None], baseline_boxes[:-1]))
        states = np.stack([baseline_states[o - 1:o - 1 + h] for o in offsets])
        references = np.stack([baseline_boxes[o - 1:o - 1 + h] for o in offsets])
        result = {kind + field: np.empty((a, len(masks), h, 4), dtype=np.float64)
                  for kind in ("sync", "free") for field in ("_boxes", "_states")}
        result.update({kind + "_layers": np.empty((a, len(masks), h), dtype=np.int64)
                       for kind in ("sync", "free")})
        removed = np.zeros(len(masks), dtype=np.int64)
        errors = np.zeros(len(masks), dtype=np.float64)
        for anchor, offset in enumerate(offsets):
            images = [rt.read(frame_path(directory, dataset, start + int(offset) + step)) for step in range(h)]
            if anchor == 0:
                _, probe, _ = rollout(rt, images, states[anchor], offset, False)
                zero = dict(masks[0], drop=[])
                with surgery(rt, zero, verify=True):
                    _, identity_boxes, _ = rollout(rt, images, states[anchor], offset, False)
            for mask_index, spec in enumerate(masks):
                with surgery(rt, spec, verify=(anchor == 0)) as info:
                    removed[mask_index] = info["removed_parameters"]
                    if anchor == 0:
                        errors[mask_index] = info["masked_equivalence_max_abs_diff"]
                    for kind, synchronized in (("sync", True), ("free", False)):
                        branch_states, boxes, layers = rollout(rt, images, states[anchor], offset, synchronized)
                        result[kind + "_states"][anchor, mask_index] = branch_states
                        result[kind + "_boxes"][anchor, mask_index] = boxes
                        result[kind + "_layers"][anchor, mask_index] = layers
            if anchor == 0:
                _, restored, _ = rollout(rt, images, states[anchor], offset, False)
            print("[branch] {}: anchor {}/{}; {} persistent masks x 2 layouts x H{}".format(
                entry["sequence"], anchor + 1, a, len(masks), h), flush=True)
    baseline_gt = np.asarray(gt[start + 1:stop], dtype=np.float64)
    baseline_valid = np.asarray(valid[start + 1:stop], dtype=bool)
    windows_gt = np.stack([baseline_gt[o - 1:o - 1 + h] for o in offsets])
    windows_valid = np.stack([baseline_valid[o - 1:o - 1 + h] for o in offsets])
    result.update(baseline_boxes=baseline_boxes, baseline_layers=np.asarray(baseline_layers, dtype=np.int64),
                  baseline_gt=baseline_gt, baseline_valid=baseline_valid,
                  baseline_iou=tracking_iou(baseline_boxes, baseline_gt, baseline_valid),
                  anchor_offset=offsets, reference_states=states, reference_boxes=references,
                  gt=windows_gt, valid=windows_valid,
                  reference_iou=tracking_iou(references, windows_gt, windows_valid),
                  reference_probe_boxes=probe, post_restore_probe_boxes=restored,
                  identity_boxes=identity_boxes, removed_parameters=removed, surgery_error=errors)
    for kind in ("sync", "free"):
        result[kind + "_iou"] = tracking_iou(result[kind + "_boxes"], windows_gt[:, None], windows_valid[:, None])
    return result, initial.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--manifest", type=nonempty_path, required=True)
    parser.add_argument("--output_dir", type=nonempty_path, required=True)
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--split", choices=("all", "train", "val"), default="all")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--masks_per_layer", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2057)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        parser.error("Require 0 <= shard_id < num_shards")
    if args.checkpoint is not None and not args.checkpoint.strip():
        parser.error("Empty --checkpoint; check the exported CHECKPOINT variable")
    manifest = validate_manifest(args.manifest)
    entries = [entry for entry in manifest["sequences"]
               if (args.split == "all" or entry["split"] == args.split)
               and (args.sequence is None or entry["sequence"] == args.sequence)]
    if not entries:
        parser.error("No clips match --split/--sequence")
    entries = entries[args.shard_id::args.num_shards]
    if not entries:
        print("This shard has no clips.")
        return
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and not (output / "run.json").is_file() and next(output.rglob("*.npz"), None):
        raise ValueError("Unidentified caches exist; use a new dedicated Stage-19 cache directory")
    rt = Runtime(args.parameter, args.gpu, args.checkpoint)
    widths = []
    for block in rt.backbone.blocks[:6]:
        fc1, fc2 = block.mlp.fc1, block.mlp.fc2
        if fc1.bias is None or fc1.in_features != fc2.out_features:
            raise ValueError("Expected original biased FFN with equal input/output embedding widths")
        widths.append(fc1.out_features)
    run = make_run(manifest, rt.identity, widths, args.fraction, args.masks_per_layer, args.seed)
    publish_run(output, run)
    print("Stage-19 mechanism pilot. No learned compression model or speed claim is produced.")
    print("Cache: {}\nRun fingerprint: {}".format(output, run["fingerprint"]))
    for entry in entries:
        path = cache_path(output, entry)
        if path.exists():
            load_and_audit_cache(path, entry, run)
            print("[skip, audited] {}".format(path))
            continue
        directory, gt, valid = load_annotations(manifest["root"], manifest["dataset"], entry["sequence"])
        if (file_hash(directory / "groundtruth.txt") != entry["annotation_sha256"]
                or fingerprint(valid.tolist()) != entry["visible_sha256"]):
            raise ValueError("Annotations changed: {}".format(entry["sequence"]))
        arrays, initial = collect_clip(rt, directory, manifest["dataset"], entry, gt, valid, run)
        write_cache(path, arrays, {"run_fingerprint": run["fingerprint"], "entry": entry, "init_box": initial})
        load_and_audit_cache(path, entry, run)
        print("[done] {} -> {}".format(entry["sequence"], path), flush=True)
    print('Next: python tracking/stage19_compression_summary.py --input "{}" --output_dir <analysis-dir>'.format(output))


if __name__ == "__main__":
    main()
