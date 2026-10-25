"""Replay six corrected-branch points to inspect retained spatial target candidates."""

import argparse
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_cache
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, nonempty_path, publish_identical, scene_geometry, seal
from stage27_uav_summary import write_table
from stage28_response_collect import validate_runtime
from stage28_response_common import cpu_array, dense_boxes
from stage30_branch_response_common import (DESIGN, SCHEMA, audit_point, load_inputs,
    load_point, make_plan, point_filename, point_metadata, point_source)


def make_run(rt, inputs, plan):
    validate_runtime(rt, inputs["source"])
    window = cpu_array(rt.tracker.output_window).reshape(rt.search_side, rt.search_side).astype(np.float32)
    np.testing.assert_array_equal(window, np.asarray(inputs["upstream"]["run"]["window"], np.float32))
    return seal(dict(schema=SCHEMA, role="corrected_branch_single_frame_response_replay", plan=plan,
        runtime=rt.identity, window=window.tolist(), training_performed=False, new_policy_evaluated=False,
        semantics="Independent replays of corrected-branch parents, no new intervention or state advance"))


def collect_point(rt, point, inputs, run, image_reader=checked_image):
    source = inputs["source"]
    validate_runtime(rt, source)
    entry, _, branch = point_source(inputs, point)
    b, t, frame = point["branch_rank"] - 1, point["branch_offset"], point["frame_id"]
    root = source["manifest"]["root"]
    rt.initialize(image_reader(root, entry["frames"][0]), entry["initial_box"])
    previous = branch["previous"][b, t].copy()
    rt.tracker.state, rt.tracker.frame_id = previous.tolist(), frame - 1
    image = image_reader(root, entry["frames"][frame])
    with rt.torch.no_grad():
        search, factor = rt.prepare(image)
        tokens = rt.prefix(search)
        layer = rt.selected_layer(rt.selector(tokens))
        output = rt.candidate(tokens, layer)
        rt.verify(search, output)
        native = np.asarray(rt.decode(output, factor, image), float)
        side = rt.search_side
        raw = cpu_array(output["score_map"]).reshape(side, side).astype(np.float32)
        size = cpu_array(output["size_map"]).reshape(2, side, side).astype(np.float32)
        offset = cpu_array(output["offset_map"]).reshape(2, side, side).astype(np.float32)
    window = cpu_array(rt.tracker.output_window).reshape(side, side).astype(np.float32)
    np.testing.assert_array_equal(window, np.asarray(run["window"], np.float32))
    np.testing.assert_array_equal(rt.tracker.state, previous)
    np.testing.assert_equal(rt.tracker.frame_id, frame - 1)
    gt = branch["gt"][t]
    geometry = scene_geometry(previous, gt, image.shape[:2], source["search_factor"], source["search_size"])
    boxes = dense_boxes(size, offset, previous, factor, source["search_size"], image.shape[:2])
    cell = int(np.argmax(raw * window))
    data = dict(raw=raw, window=window, size=size, offset=offset, previous=previous, gt=gt.copy(), native_box=native,
        recorded_branch_box=branch["boxes"][b, t].copy(), crop=geometry["crop"], usable=geometry["usable"],
        factor=np.asarray(factor), image_hw=np.array(image.shape[:2], np.int64), layer=np.asarray(layer, np.int64),
        frame_id=np.asarray(frame, np.int64), branch_offset=np.asarray(t, np.int64),
        output_was_overridden=np.asarray(branch["output_override"][b, t], bool),
        replay_box_error=np.asarray(np.max(np.abs(native - branch["model_boxes"][b, t]))),
        replay_peak_error=np.asarray(abs(float(raw.max()) - branch["raw_score_peak"][b, t])),
        decode_box_error=np.asarray(np.max(np.abs(boxes[cell] - native))))
    audit_point(data, point, branch, source)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True, help="Complete Stage-29 root")
    parser.add_argument("--output_dir", type=nonempty_path, required=True)
    parser.add_argument("--plan_only", action="store_true")
    parser.add_argument("--checkpoint", type=nonempty_path)
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sequence", default=None, help="Engineering pilot: planned points of one video")
    args = parser.parse_args()
    inputs = load_inputs(args.input)
    plan = make_plan(inputs)
    output = Path(args.output_dir).expanduser().resolve()
    for directory in (inputs["directory"], inputs["upstream"]["directory"], inputs["upstream"]["source_info"][0]):
        if output == directory or directory in output.parents:
            raise ValueError("Use a directory separate from frozen Stage-27/28/29 artifacts")
    selected = [p for p in plan["points"] if args.sequence is None or p["sequence"] == args.sequence]
    if not selected:
        raise ValueError("--sequence has no planned points")
    with writer_lock(output):
        if not (output / "plan.json").exists() and any(p.name != "collect.lock" for p in output.iterdir()):
            raise ValueError("Use a new empty output directory")
        publish_identical(output / "plan.json", plan)
        write_table(output / "stage30_plan_summary.csv", plan["points"])
        print("Frozen plan: {} single-frame replays; corrected-branch parents only.".format(plan["replay_frames"]))
        for p in plan["points"]:
            print("{} frame {} rank {} offset +{}: {}".format(p["sequence"], p["frame_id"], p["branch_rank"], p["branch_offset"], p["purpose"]))
        print("Plan fingerprint: " + plan["fingerprint"])
        if args.plan_only:
            print("CPU plan only; no GPU initialization or response inference.")
            return
        if not args.checkpoint:
            parser.error("--checkpoint is required unless --plan_only")
        if file_hash(Path(args.checkpoint).expanduser()) != inputs["source"]["runtime"]["checkpoint_sha256"]:
            raise ValueError("Checkpoint differs from frozen original SGLA")
        import random
        import torch
        from stage4_runtime import Runtime
        random.seed(DESIGN["seed"])
        np.random.seed(DESIGN["seed"])
        torch.manual_seed(DESIGN["seed"])
        torch.cuda.manual_seed_all(DESIGN["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        rt = Runtime(args.parameter, args.gpu, args.checkpoint)
        run = make_run(rt, inputs, plan)
        if not (output / "run.json").exists() and any(output.rglob("*.npz")):
            raise ValueError("Unidentified response caches")
        publish_identical(output / "run.json", run)
        for point in selected:
            path = output / "responses" / point_filename(point)
            if path.exists():
                load_point(path, point, inputs, run)
                print("[reuse, audited] " + path.name, flush=True)
                continue
            data = collect_point(rt, point, inputs, run)
            write_cache(path, data, point_metadata(run, point))
            load_point(path, point, inputs, run)
            print("[done] {}: model-box replay error {:.6g}".format(path.name, float(data["replay_box_error"])), flush=True)
    print('Next: python tracking/stage30_branch_response_summary.py --input "{}" --render'.format(output))


if __name__ == "__main__":
    main()
