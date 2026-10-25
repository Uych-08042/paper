"""Enumerate frozen weighted top-5 boxes; override once, then track natively for H40."""

import argparse
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_cache
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, nonempty_path, publish_identical, scene_geometry, seal, tracking_iou
from stage27_uav_summary import write_table
from stage28_response_collect import validate_runtime
from stage28_response_common import DESIGN as RESPONSE_DESIGN, cpu_array
from stage29_candidate_common import (DESIGN, SCHEMA, audit_event, event_filename, event_metadata,
                                      load_event, load_inputs, make_plan)


def make_run(rt, inputs, plan):
    _, source, _, _ = inputs["source_info"]
    validate_runtime(rt, source)
    window = cpu_array(rt.tracker.output_window).reshape(rt.search_side, rt.search_side)
    np.testing.assert_array_equal(window, np.asarray(inputs["run"]["window"], np.float32))
    return seal(dict(schema=SCHEMA, role="one_output_intervention_native_continuation", plan=plan,
        runtime=rt.identity, continuation="original frozen SGLA, own predicted crops, native L7-L12 selection",
        gt_use="initial template plus offline onset candidate labels and outcome evaluation only",
        intervention="each ranked box injected once at onset; rank 1 is the unmodified native replay",
        training_performed=False, learned_policy_evaluated=False, resource_claim="none"))


def collect_event(rt, event, entry, trace, responses, source, run, image_reader=checked_image):
    validate_runtime(rt, source)
    onset = event["onset_frame"]
    frame_ids = np.arange(onset, onset + DESIGN["horizon"] + 1, dtype=np.int64)
    root = source["manifest"]["root"]
    images = [image_reader(root, entry["frames"][int(f)]) for f in frame_ids]
    initial_image = image_reader(root, entry["frames"][0])
    response_index = int(np.flatnonzero(responses["frame_id"] == onset)[0])
    window = cpu_array(rt.tracker.output_window).reshape(rt.search_side, rt.search_side)
    np.testing.assert_array_equal(window, responses["window"])
    keys = ("previous", "boxes", "model_boxes", "crop", "usable", "factor", "layer", "raw_score_peak", "coverage")
    result = {k: [] for k in keys}
    with rt.torch.no_grad():
        for candidate in event["candidates"]:
            # Reinitialize the original template for every branch. The only branch-specific input is its box.
            rt.initialize(initial_image, entry["initial_box"])
            rt.tracker.state = trace["previous"][onset - 1].tolist()
            rt.tracker.frame_id = onset - 1
            branch = {k: [] for k in keys}
            for step, (frame, image) in enumerate(zip(frame_ids, images)):
                np.testing.assert_equal(rt.tracker.frame_id, int(frame) - 1)
                previous = np.asarray(rt.tracker.state, float).copy()
                search, factor = rt.prepare(image)
                tokens = rt.prefix(search)
                layer = rt.selected_layer(rt.selector(tokens))
                output = rt.candidate(tokens, layer)
                model_box = np.asarray(rt.decode(output, factor, image), float)
                peak = float(output["score_map"].detach().amax().item())
                if step == 0:
                    if candidate["rank"] == 1:
                        rt.verify(search, output)
                    for key, cached, shape in (("score_map", "raw", (rt.search_side, rt.search_side)),
                            ("size_map", "size", (2, rt.search_side, rt.search_side)),
                            ("offset_map", "offset", (2, rt.search_side, rt.search_side))):
                        np.testing.assert_allclose(cpu_array(output[key]).reshape(shape), responses[cached][response_index],
                            atol=RESPONSE_DESIGN["replay_score_atol"], rtol=0,
                            err_msg="Onset response differs from frozen Stage-28")
                    np.testing.assert_allclose(model_box, responses["native_boxes"][response_index],
                        atol=RESPONSE_DESIGN["replay_box_atol"], rtol=0)
                # No GT is used to choose an executed branch or any continuation action.
                box = np.asarray(candidate["box"], float).copy() if step == 0 and candidate["rank"] != 1 else model_box.copy()
                rt.tracker.state = box.tolist()
                rt.tracker.frame_id = int(frame)
                geometry = scene_geometry(previous, entry["ground_truth"][int(frame)], image.shape[:2],
                                          source["search_factor"], source["search_size"])
                np.testing.assert_allclose(factor, geometry["factor"], atol=1e-12, rtol=0)
                np.testing.assert_array_equal(rt.tracker.state, box)
                values = dict(previous=previous, boxes=box, model_boxes=model_box, crop=geometry["crop"],
                    usable=geometry["usable"], factor=factor, layer=layer, raw_score_peak=peak,
                    coverage=geometry["coverage"])
                for k in keys:
                    branch[k].append(values[k])
            for k in keys:
                result[k].append(branch[k])
            print("[branch] {} e{} rank {}/{}: {} native future frames".format(
                entry["sequence"], event["event"], candidate["rank"], len(event["candidates"]), DESIGN["horizon"]), flush=True)
    data = {k: np.asarray(v) for k, v in result.items()}
    data.update(frame_id=frame_ids, candidate_rank=np.arange(1, len(event["candidates"]) + 1, dtype=np.int64),
        candidate_cell=np.array([v["grid_index"] for v in event["candidates"]], np.int64),
        image_hw=np.array([im.shape[:2] for im in images], np.int64),
        gt=np.asarray(entry["ground_truth"], float)[frame_ids])
    data["layer"] = data["layer"].astype(np.int64)
    data["output_override"] = np.zeros(data["factor"].shape, bool)
    data["output_override"][1:, 0] = True
    data["iou"] = tracking_iou(data["boxes"], data["gt"][None])
    audit_event(data, event, entry, trace, source)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True, help="Complete Stage-28 root")
    parser.add_argument("--output_dir", type=nonempty_path, required=True)
    parser.add_argument("--plan_only", action="store_true")
    parser.add_argument("--checkpoint", type=nonempty_path)
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sequence", default=None, help="Engineering pilot: all planned events of one video")
    args = parser.parse_args()
    inputs = load_inputs(args.input)
    plan = make_plan(inputs)
    output = Path(args.output_dir).expanduser().resolve()
    for directory in (inputs["directory"], inputs["source_info"][0]):
        if output == directory or directory in output.parents:
            raise ValueError("Use an output directory separate from frozen Stage-27/28 artifacts")
    selected = [e for e in plan["events"] if args.sequence is None or e["sequence"] == args.sequence]
    if not selected:
        raise ValueError("--sequence has no event in the frozen plan")
    with writer_lock(output):
        if not (output / "plan.json").exists() and any(p.name != "collect.lock" for p in output.iterdir()):
            raise ValueError("Use a new empty output directory")
        publish_identical(output / "plan.json", plan)
        write_table(output / "stage29_plan_summary.csv", [dict(sequence=e["sequence"], event=e["event"],
            onset_frame=e["onset_frame"], candidates=len(e["candidates"]), primary_eligible=e["primary_eligible"],
            onset_gt_best_rank=e["onset_gt_best_rank"], future_frames=DESIGN["horizon"]) for e in plan["events"]])
        print("Frozen plan: {} events, {} candidate branches, {} primary events; H40 excludes onset.".format(
            len(plan["events"]), plan["branches"], sum(e["primary_eligible"] for e in plan["events"])))
        print("Planned model forwards: {} onset replays + {} future frames.".format(
            plan["onset_replay_forwards"], plan["future_model_forwards"]))
        print("Plan fingerprint: " + plan["fingerprint"])
        if args.plan_only:
            print("CPU plan only; no GPU initialization, intervention or branch inference.")
            return
        _, source, traces, _ = inputs["source_info"]
        if not args.checkpoint:
            parser.error("--checkpoint is required unless --plan_only")
        if file_hash(Path(args.checkpoint).expanduser()) != source["runtime"]["checkpoint_sha256"]:
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
            raise ValueError("Unidentified branch caches")
        publish_identical(output / "run.json", run)
        for event in selected:
            entry, trace = traces[event["sequence"]]
            path = output / "branches" / event_filename(event)
            if path.exists():
                load_event(path, event, entry, trace, source, run)
                print("[reuse, audited] " + path.name, flush=True)
                continue
            data = collect_event(rt, event, entry, trace, inputs["caches"][entry["sequence"]], source, run)
            write_cache(path, data, event_metadata(run, event))
            load_event(path, event, entry, trace, source, run)
            print("[done] " + path.name, flush=True)
    print('Next: python tracking/stage29_candidate_summary.py --input "{}"'.format(output))


if __name__ == "__main__":
    main()
