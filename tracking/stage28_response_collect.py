"""Replay frozen Stage-27 states to inspect raw versus windowed response competition."""

import argparse
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_cache
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, nonempty_path, publish_identical, seal
from stage27_uav_summary import write_table
from stage28_response_common import (DESIGN, SCHEMA, audit_response, bind_native_top1, cpu_array,
    dense_boxes, load_response, load_source, make_plan)


def validate_runtime(rt, source):
    for key in ("checkpoint_sha256", "parameter", "config", "lens_z", "lens_x", "embed_dim"):
        if key in source["runtime"] and rt.identity.get(key) != source["runtime"][key]:
            raise ValueError("Runtime differs from frozen Stage-27: " + key)
    if (rt.tracker.params.search_factor != source["search_factor"]
            or rt.tracker.params.search_size != source["search_size"]):
        raise ValueError("Search configuration changed")


def make_run(rt, source, plan):
    validate_runtime(rt, source)
    side = rt.search_side
    window = cpu_array(rt.tracker.output_window).reshape(side, side).astype(np.float32)
    return seal(dict(schema=SCHEMA, role="frozen_response_replay", plan=plan,
        runtime=rt.identity, window=window.tolist(),
        semantics="independent single-frame replays from recorded predicted parents; no new closed-loop trajectory",
        gt_use="post-forward box matching and offline rank labels only",
        training_performed=False, new_policy_evaluated=False, speed_claim_tested=False))


def collect_video(rt, entry, trace, video, source, run, image_reader=checked_image):
    validate_runtime(rt, source)
    side, root = rt.search_side, source["manifest"]["root"]
    rt.initialize(image_reader(root, entry["frames"][0]), entry["initial_box"])
    window = cpu_array(rt.tracker.output_window).reshape(side, side).astype(np.float32)
    np.testing.assert_array_equal(window, np.asarray(run["window"], np.float32))
    keys = ("frame_id", "raw", "size", "offset", "previous", "gt", "factor", "image_hw",
            "native_boxes", "raw_boxes", "layer", "replay_box_error", "replay_peak_error", "decode_box_error")
    result = {key: [] for key in keys}
    with rt.torch.no_grad():
        for ordinal, task in enumerate(video["frames"]):
            frame_id = task["frame_id"]
            index = frame_id - 1
            image = image_reader(root, entry["frames"][frame_id])
            # The parent comes from the frozen prediction, never the GT or a diagnostic decode.
            previous = trace["previous"][index].copy()
            rt.tracker.state = previous.tolist()
            rt.tracker.frame_id = frame_id - 1
            search, factor = rt.prepare(image)
            tokens = rt.prefix(search)
            layer = rt.selected_layer(rt.selector(tokens))
            output = rt.candidate(tokens, layer)
            if ordinal == 0:
                rt.verify(search, output)
            native = np.asarray(rt.decode(output, factor, image), float)
            raw = cpu_array(output["score_map"]).reshape(side, side).astype(np.float32)
            size = cpu_array(output["size_map"]).reshape(2, side, side).astype(np.float32)
            offset = cpu_array(output["offset_map"]).reshape(2, side, side).astype(np.float32)
            boxes = dense_boxes(size, offset, previous, factor, source["search_size"], image.shape[:2])
            weighted_box = boxes[int(np.argmax(raw * window))]
            replay_error = float(np.max(np.abs(native - trace["boxes"][index])))
            peak_error = float(abs(float(raw.max()) - trace["raw_score_peak"][index]))
            decode_error = float(np.max(np.abs(weighted_box - native)))
            boxes = bind_native_top1(raw, window, boxes, native)
            raw_box = boxes[int(np.argmax(raw))]
            if (layer != trace["layer"][index] or factor != trace["factor"][index]
                    or replay_error > DESIGN["replay_box_atol"]
                    or peak_error > DESIGN["replay_score_atol"]
                    or decode_error > DESIGN["decode_box_atol"]):
                raise ValueError("Replay mismatch at {} frame {}: layer={}, bbox={}, peak={}, dense={}".format(
                    entry["sequence"], frame_id, layer, replay_error, peak_error, decode_error))
            np.testing.assert_array_equal(rt.tracker.state, previous)
            values = dict(frame_id=frame_id, raw=raw, size=size, offset=offset, previous=previous,
                gt=trace["gt"][index], factor=factor, image_hw=image.shape[:2], native_boxes=native,
                raw_boxes=raw_box, layer=layer, replay_box_error=replay_error,
                replay_peak_error=peak_error, decode_box_error=decode_error)
            for key in keys:
                result[key].append(values[key])
    data = {key: np.asarray(value) for key, value in result.items()}
    data["window"] = window
    for key in ("frame_id", "layer", "image_hw"):
        data[key] = data[key].astype(np.int64)
    audit_response(data, entry, trace, video, source)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True, help="Completed Stage-27 cache directory")
    parser.add_argument("--output_dir", type=nonempty_path, required=True)
    parser.add_argument("--plan_only", action="store_true", help="CPU-only source audit and frozen replay schedule")
    parser.add_argument("--checkpoint", type=nonempty_path)
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sequence", default=None, help="Replay only one planned video for an engineering pilot")
    args = parser.parse_args()
    source_info = load_source(args.input)
    directory, source, traces, hashes = source_info
    output = Path(args.output_dir).expanduser().resolve()
    if output == directory or directory in output.parents:
        raise ValueError("Stage-28 output must be separate from the frozen Stage-27 cache")
    plan = make_plan(directory, source, traces, hashes)
    videos = [v for v in plan["videos"] if args.sequence is None or v["sequence"] == args.sequence]
    if not videos:
        raise ValueError("--sequence is not in the frozen plan")
    with writer_lock(output):
        if not (output / "plan.json").exists() and any(p.name != "collect.lock" for p in output.iterdir()):
            raise ValueError("Use a new empty output directory")
        publish_identical(output / "plan.json", plan)
        write_table(output / "stage28_plan_summary.csv", [dict(sequence=v["sequence"],
            source_group_proxy=v["source_group_proxy"], unique_replay_frames=len(v["frames"]),
            events=sum(e["sequence"] == v["sequence"] for e in plan["events"])) for v in plan["videos"]])
        print("Frozen plan: {} videos, {} events, {} unique replay frames.".format(
            len(plan["videos"]), len(plan["events"]), plan["unique_replay_frames"]))
        print("Plan fingerprint: " + plan["fingerprint"])
        if args.plan_only:
            print("CPU plan only. No GPU initialization or response inference.")
            return
        if not args.checkpoint:
            parser.error("--checkpoint is required unless --plan_only")
        if file_hash(Path(args.checkpoint).expanduser()) != source["runtime"]["checkpoint_sha256"]:
            raise ValueError("Checkpoint bytes differ from frozen Stage-27")
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
        run = make_run(rt, source, plan)
        if not (output / "run.json").exists() and any(output.rglob("*.npz")):
            raise ValueError("Unidentified response caches")
        publish_identical(output / "run.json", run)
        for video in videos:
            entry, trace = traces[video["sequence"]]
            path = output / "responses" / (entry["sequence"] + "__stage28_response.npz")
            if path.exists():
                load_response(path, entry, trace, video, source, run)
                print("[reuse, audited] " + entry["sequence"], flush=True)
                continue
            data = collect_video(rt, entry, trace, video, source, run)
            write_cache(path, data, dict(run_fingerprint=run["fingerprint"], sequence=entry["sequence"]))
            load_response(path, entry, trace, video, source, run)
            print("[done] {}: {} replayed frames; max source box error {:.6g}".format(
                entry["sequence"], len(data["frame_id"]), data["replay_box_error"].max()), flush=True)
    print("Next: python tracking/stage28_response_summary.py --input \"{}\"".format(output))


if __name__ == "__main__":
    main()
