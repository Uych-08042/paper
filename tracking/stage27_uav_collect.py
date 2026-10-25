"""Original SGLA full-video traces and offline scene proxies, without training."""

import argparse
import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_cache
from stage27_uav_common import (SCHEMA, audit_trace, checked_image, estimate_motion,
    load_manifest, load_trace, nonempty_path, publish_identical, scene_geometry,
    seal, sharpness_proxy, tracking_iou)


@contextmanager
def writer_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "collect.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError("Collector lock exists; check for an active process: " + str(lock)) from error
    try:
        with handle:
            import os
            handle.write(str(os.getpid()))
        yield
    finally:
        lock.unlink()


def code_hashes():
    directory = Path(__file__).parent
    names = ("stage27_uav_common.py", "stage27_uav_collect.py", "stage4_runtime.py",
             "stage1_validate.py", "validation_metrics.py", "stage4_common.py",
             "../lib/train/data/processing_utils.py", "../lib/test/tracker/data_utils.py",
             "../lib/test/tracker/sglatrack.py", "../lib/models/sglatrack/base_backbone.py",
             "../lib/models/sglatrack/deit.py", "../lib/models/sglatrack/vit.py",
             "../lib/models/sglatrack/sglatrack.py", "../lib/models/sglatrack/utils.py",
             "../lib/models/layers/head.py", "../lib/models/layers/patch_embed.py",
             "../lib/utils/box_ops.py")
    return {name: file_hash(directory / name) for name in names}


def make_run(rt, manifest):
    return seal(dict(schema=SCHEMA, role="original_sgla_descriptive_uav_diagnosis",
        manifest=manifest, runtime=rt.identity, code_sha256=code_hashes(),
        search_factor=float(rt.tracker.params.search_factor),
        search_size=int(rt.tracker.params.search_size), seed=2092,
        policy="original SGLA selector and checkpoint; no learned memory/controller/adapters",
        scene_features="post-prediction offline measurements; GT-masked background proxy is not actor state",
        visibility="unknown; IoU includes all geometrically valid GT rows, including possible occlusions",
        resource_claim="none; instrumented collection is not a speed benchmark",
        interventions_performed=False, training_performed=False))


def collect_video(rt, entry, run, image_reader=checked_image,
                  motion_estimator=estimate_motion, sharpness_estimator=sharpness_proxy):
    root = run["manifest"]["root"]
    gt = np.asarray(entry["ground_truth"], float)
    image_previous = image_reader(root, entry["frames"][0])
    rt.initialize(image_previous, gt[0].tolist())
    arrays = {key: [] for key in ("frame_id", "previous", "boxes", "gt", "image_hw",
        "crop", "usable", "factor", "coverage", "gt_image_fraction", "input_wh",
        "center_in_usable", "padding_fraction", "layer", "raw_score_peak", "motion", "sharpness")}
    with rt.torch.no_grad():
        for frame_id in range(1, len(gt)):
            image = image_reader(root, entry["frames"][frame_id])
            previous = np.asarray(rt.tracker.state, float).copy()
            search, factor = rt.prepare(image)
            tokens = rt.prefix(search)
            layer = rt.selected_layer(rt.selector(tokens))
            output = rt.candidate(tokens, layer)
            if frame_id == 1:
                rt.verify(search, output)
            box = np.asarray(rt.decode(output, factor, image), float)
            peak = float(output["score_map"].detach().amax().item())
            # Advance only with the executed model prediction. GT is read below for logging.
            rt.tracker.state = box.tolist()
            rt.tracker.frame_id += 1
            geometry = scene_geometry(previous, gt[frame_id], image.shape[:2],
                                      run["search_factor"], run["search_size"])
            if not np.isclose(factor, geometry["factor"], atol=1e-12, rtol=0):
                raise ValueError("Recorded geometry disagrees with actual crop resize")
            motion = motion_estimator(image_previous, image,
                                     [gt[frame_id - 1], previous], [gt[frame_id], box])
            if not np.array_equal(np.asarray(rt.tracker.state), box):
                raise ValueError("Offline diagnostics modified the live tracker")
            values = dict(frame_id=frame_id, previous=previous, boxes=box, gt=gt[frame_id],
                          image_hw=image.shape[:2], layer=layer, raw_score_peak=peak,
                          motion=motion, sharpness=sharpness_estimator(image, gt[frame_id]), **geometry)
            for key in arrays:
                arrays[key].append(values[key])
            image_previous = image
            if frame_id % 250 == 0:
                print("[tracking] {} {}/{}".format(entry["sequence"], frame_id, len(gt) - 1), flush=True)
    data = {key: np.asarray(values) for key, values in arrays.items()}
    for key in ("frame_id", "layer", "image_hw"):
        data[key] = data[key].astype(np.int64)
    data["iou"] = tracking_iou(data["boxes"], data["gt"])
    audit_trace(data, entry, run)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=nonempty_path, required=True)
    parser.add_argument("--checkpoint", type=nonempty_path, required=True)
    parser.add_argument("--output_dir", type=nonempty_path, required=True)
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sequence", default=None, help="One selected video; entire video, never a shortened clip")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    entries = [e for e in manifest["sequences"] if args.sequence is None or e["sequence"] == args.sequence]
    if not entries:
        raise ValueError("--sequence is not in the frozen manifest")
    # Heavy dependencies are deliberately below manifest/CLI validation.
    import cv2
    import torch
    from stage4_runtime import Runtime
    random.seed(2092)
    np.random.seed(2092)
    cv2.setRNGSeed(2092)
    torch.manual_seed(2092)
    torch.cuda.manual_seed_all(2092)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    rt = Runtime(args.parameter, args.gpu, args.checkpoint)
    run = make_run(rt, manifest)
    output = Path(args.output_dir).expanduser().resolve()
    with writer_lock(output):
        if not (output / "run.json").exists() and any(output.rglob("*.npz")):
            raise ValueError("Unidentified traces; use an empty output directory")
        publish_identical(output / "run.json", run)
        for entry in entries:
            path = output / "traces" / (entry["sequence"] + "__stage27_uav.npz")
            # Seed per video so a resumed/skipped earlier video cannot change RANSAC sampling.
            cv2.setRNGSeed(2092 + int(entry["sequence"][3:10]))
            if path.exists():
                load_trace(path, entry, run)
                print("[reuse, audited] " + entry["sequence"], flush=True)
                continue
            data = collect_video(rt, entry, run)
            write_cache(path, data, dict(run_fingerprint=run["fingerprint"], sequence=entry["sequence"]))
            load_trace(path, entry, run)
            print("[done] {}: {} tracked frames -> {}".format(entry["sequence"], len(data["iou"]), path), flush=True)
    print("Next: python tracking/stage27_uav_summary.py --input \"{}\"".format(output))


if __name__ == "__main__":
    main()
