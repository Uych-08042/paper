"""Cross four frozen crop histories with four existing models; no new rollout."""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_cache
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, crop_geometry, nonempty_path, publish_identical, seal, tracking_iou
from stage28_response_collect import validate_runtime
from stage28_response_common import cpu_array, dense_boxes
from stage34_identity_experiment import seed_runtime
from stage35_preserve_runtime import forward_student
from stage36_same_crop_common import (DESIGN, METHODS, SCHEMA, load_inputs,
    load_models, load_sealed, prepare, verify_bindings)


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def collect_video(rt, source, entry, traces, video, models, run, image_reader=checked_image):
    torch, side = rt.torch, rt.search_side
    fold = next(f["fold"] for f in source["folds"] if entry["sequence"] in f["held_out"])
    students = {method: models[(fold, method)] for method in METHODS[2:]}
    rt.initialize(image_reader(source["root"], entry["frames"][0]), entry["initial_box"])
    window = cpu_array(rt.tracker.output_window).reshape(side, side).astype(np.float32)
    np.testing.assert_array_equal(window, np.asarray(run["window"], np.float32))
    n = len(video["frames"])
    data = dict(frame_id=np.array([f["frame_id"] for f in video["frames"]], np.int64),
        raw=np.empty((n, 4, 4, side, side), np.float32),
        size=np.empty((n, 4, 4, 2, side, side), np.float32),
        offset=np.empty((n, 4, 4, 2, side, side), np.float32),
        native_boxes=np.empty((n, 4, 4, 4)), previous=np.empty((n, 4, 4)), gt=np.empty((n, 4)),
        factor=np.empty((n, 4)), image_hw=np.empty((n, 2), np.int64), layer=np.zeros((n, 4, 4), np.int64),
        crop_sha256=np.empty((n, 4), dtype="U64"), replay_box_error=np.empty((n, 4)),
        decode_box_error=np.empty((n, 4, 4)), window=window)
    calls = [0, 0, 0]
    heads = [rt.net.box_head] + [students[m].box_head for m in METHODS[2:]]
    hooks = [h.register_forward_hook(lambda *unused, k=k: calls.__setitem__(k, calls[k] + 1)) for k, h in enumerate(heads)]
    try:
        with torch.no_grad():
            for row, task in enumerate(video["frames"]):
                frame_id = task["frame_id"]
                image = image_reader(source["root"], entry["frames"][frame_id])
                data["image_hw"][row] = image.shape[:2]
                # GT is logged after the forwards; it never creates a crop or selects a peak.
                for parent, owner in enumerate(METHODS):
                    previous = traces[owner]["previous"][frame_id - 1].copy()
                    rt.tracker.state, rt.tracker.frame_id = previous.tolist(), frame_id - 1
                    search, factor = rt.prepare(image)  # One SAME tensor for all four evaluators.
                    expected_factor = crop_geometry(previous, image.shape[:2], source["search_factor"], source["search_size"])[2]
                    np.testing.assert_allclose(factor, expected_factor, atol=1e-12, rtol=0)
                    digest = tensor_digest(search)
                    tokens = rt.prefix(search)
                    layer = rt.selected_layer(rt.selector(tokens))
                    outputs = [rt.head(tokens), rt.candidate(tokens, layer)]
                    outputs += [forward_student(rt, students[m], search) for m in METHODS[2:]]
                    if tensor_digest(search) != digest:
                        raise ValueError("A model modified the shared search tensor")
                    data["crop_sha256"][row, parent] = digest
                    data["previous"][row, parent], data["factor"][row, parent] = previous, factor
                    data["layer"][row, parent, 1] = layer
                    for evaluator, output in enumerate(outputs):
                        native = np.asarray(rt.decode(output, factor, image), float)
                        raw = cpu_array(output["score_map"]).reshape(side, side).astype(np.float32)
                        size = cpu_array(output["size_map"]).reshape(2, side, side).astype(np.float32)
                        offset = cpu_array(output["offset_map"]).reshape(2, side, side).astype(np.float32)
                        dense = dense_boxes(size, offset, previous, factor, source["search_size"], image.shape[:2])
                        error = float(np.max(np.abs(dense[int(np.argmax(raw * window))] - native)))
                        if error > DESIGN["decode_box_atol"]:
                            raise ValueError("Dense/native decoder mismatch")
                        data["raw"][row, parent, evaluator] = raw
                        data["size"][row, parent, evaluator] = size
                        data["offset"][row, parent, evaluator] = offset
                        data["native_boxes"][row, parent, evaluator] = native
                        data["decode_box_error"][row, parent, evaluator] = error
                    replay = float(np.max(np.abs(data["native_boxes"][row, parent, parent] - traces[owner]["boxes"][frame_id - 1])))
                    if replay > DESIGN["replay_box_atol"]:
                        raise ValueError("Own-parent replay mismatch: {} {} frame {} error {}".format(entry["sequence"], owner, frame_id, replay))
                    data["replay_box_error"][row, parent] = replay
                    np.testing.assert_array_equal(rt.tracker.state, previous)
                data["gt"][row] = traces["fast_initial"]["gt"][frame_id - 1]
                if (row + 1) % 25 == 0:
                    print("[crossed replay] {} {}/{} images".format(entry["sequence"], row + 1, n), flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    data["model_head_calls"] = np.asarray(calls, np.int64)
    audit_response(data, source, entry, traces, video, run)
    return data


def audit_response(data, source, entry, traces, video, run):
    frames = np.array([f["frame_id"] for f in video["frames"]], np.int64)
    n, side = len(frames), int(source["runtime"]["lens_x"] ** .5)
    shapes = dict(frame_id=(n,), raw=(n, 4, 4, side, side), size=(n, 4, 4, 2, side, side),
        offset=(n, 4, 4, 2, side, side), native_boxes=(n, 4, 4, 4), previous=(n, 4, 4), gt=(n, 4),
        factor=(n, 4), image_hw=(n, 2), layer=(n, 4, 4), crop_sha256=(n, 4),
        replay_box_error=(n, 4), decode_box_error=(n, 4, 4), window=(side, side), model_head_calls=(3,))
    if set(data) != set(shapes) or any(data[k].shape != shape for k, shape in shapes.items()):
        raise ValueError("Crossed response cache shape/coverage mismatch")
    if not all(np.isfinite(v).all() for k, v in data.items() if k != "crop_sha256"):
        raise ValueError("Nonfinite replay cache")
    if ((data["raw"] <= 0).any() or (data["raw"] > 1).any() or (data["size"] < 0).any()
            or (data["factor"] <= 0).any() or (data["native_boxes"][..., 2:] <= 0).any()):
        raise ValueError("Invalid replay response/geometry")
    if any(len(str(v)) != 64 or any(c not in "0123456789abcdef" for c in str(v)) for v in data["crop_sha256"].flat):
        raise ValueError("Missing shared crop digest")
    np.testing.assert_array_equal(data["frame_id"], frames)
    np.testing.assert_array_equal(data["window"], np.asarray(run["window"], np.float32))
    np.testing.assert_array_equal(data["model_head_calls"], [8 * n, 4 * n, 4 * n])
    np.testing.assert_array_equal(data["gt"], traces["fast_initial"]["gt"][frames - 1])
    if (data["layer"][:, :, [0, 2, 3]] != 0).any() or not np.isin(data["layer"][:, :, 1], np.arange(7, 13)).all():
        raise ValueError("Unexpected executed candidate layer")
    for parent, owner in enumerate(METHODS):
        np.testing.assert_array_equal(data["previous"][:, parent], traces[owner]["previous"][frames - 1])
        errors = np.max(np.abs(data["native_boxes"][:, parent, parent] - traces[owner]["boxes"][frames - 1]), axis=1)
        np.testing.assert_allclose(data["replay_box_error"][:, parent], errors, atol=1e-12, rtol=0)
        if errors.max() > DESIGN["replay_box_atol"]:
            raise ValueError("Frozen own-parent replay failed")
        np.testing.assert_allclose(tracking_iou(data["native_boxes"][:, parent, parent], data["gt"]),
            traces[owner]["iou"][frames - 1], atol=DESIGN["replay_iou_atol"], rtol=0,
            err_msg="Own-parent replay changed official integer-box IoU")
    for row, frame in enumerate(frames):
        expected_hw = [entry["frames"][frame]["height"], entry["frames"][frame]["width"]]
        np.testing.assert_array_equal(data["image_hw"][row], expected_hw)
        for parent in range(4):
            factor = crop_geometry(data["previous"][row, parent], expected_hw, source["search_factor"], source["search_size"])[2]
            np.testing.assert_allclose(data["factor"][row, parent], factor, atol=1e-12, rtol=0)
            for evaluator in range(4):
                dense = dense_boxes(data["size"][row, parent, evaluator], data["offset"][row, parent, evaluator],
                    data["previous"][row, parent], factor, source["search_size"], expected_hw)
                cell = int(np.argmax(data["raw"][row, parent, evaluator] * data["window"]))
                error = float(np.max(np.abs(dense[cell] - data["native_boxes"][row, parent, evaluator])))
                np.testing.assert_allclose(error, data["decode_box_error"][row, parent, evaluator], atol=1e-12, rtol=0)
                if error > DESIGN["decode_box_atol"]:
                    raise ValueError("Dense/native decoder mismatch in cached response")


def load_response(path, source, entry, traces, video, run):
    import json
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata_json"].item()))
        data = {k: archive[k] for k in archive.files if k != "metadata_json"}
    if meta != dict(schema=SCHEMA, run_fingerprint=run["fingerprint"], sequence=entry["sequence"]):
        raise ValueError("Mixed Stage-36 replay cache")
    audit_response(data, source, entry, traces, video, run)
    return data


def collect(rt, output, sequence=None, image_reader=checked_image):
    from stage20_compression_common import state_digest
    output = Path(output).resolve()
    plan, source, traces = load_inputs(output)
    validate_runtime(rt, source)
    models = load_models(rt, source, plan["source_input"])
    frozen = {"original": state_digest(rt.net)}
    frozen.update({str(k): state_digest(net) for k, net in models.items()})
    run = seal(dict(schema=SCHEMA, role="crossed_frozen_crop_response_replay", plan_fingerprint=plan["fingerprint"],
        runtime=rt.identity, window=cpu_array(rt.tracker.output_window).reshape(rt.search_side, rt.search_side).tolist(),
        frozen_neural_model_sha256=frozen, methods=list(METHODS), parents=list(METHODS)))
    entries = {e["sequence"]: e for e in source["fit"]}
    videos = [v for v in plan["videos"] if sequence is None or v["sequence"] == sequence]
    if not videos:
        raise ValueError("--sequence must be one of the four frozen cases")
    with writer_lock(output):
        if not (output / "run.json").exists() and any(output.rglob("*.npz")):
            raise ValueError("Unidentified replay caches")
        publish_identical(output / "run.json", run)
        for video in videos:
            name = video["sequence"]
            path = output / "responses" / (name + "__stage36_same_crop.npz")
            if path.exists():
                data = load_response(path, source, entries[name], traces[name], video, run)
                action = "reuse, audited"
            else:
                data = collect_video(rt, source, entries[name], traces[name], video, models, run, image_reader)
                write_cache(path, data, dict(schema=SCHEMA, run_fingerprint=run["fingerprint"], sequence=name))
                action = "done"
            print("[{}] {}: {} images x 4 parents x 4 models; own-parent max box error {:.6g}".format(
                action, name, len(data["frame_id"]), data["replay_box_error"].max()), flush=True)
        if frozen["original"] != state_digest(rt.net) or any(frozen[str(k)] != state_digest(net) for k, net in models.items()):
            raise ValueError("A frozen neural model changed during replay")
        verify_bindings(plan["source_file_sha256"])
    print("Next: python tracking/stage36_same_crop_summary.py --input \"{}\"".format(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True, help="Completed Stage-35 root")
    parser.add_argument("--output_dir", type=nonempty_path, required=True)
    parser.add_argument("--plan_only", action="store_true")
    parser.add_argument("--checkpoint", type=nonempty_path)
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sequence")
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    with writer_lock(output):
        plan = prepare(args.input, output)
    if args.plan_only:
        print("CPU planning only; no neural model initialized")
        return
    if not args.checkpoint or file_hash(Path(args.checkpoint).expanduser()) != plan["runtime"]["checkpoint_sha256"]:
        parser.error("Use --checkpoint with the frozen original SGLA weights")
    import torch
    from stage4_runtime import Runtime
    seed_runtime(torch, DESIGN["seed"])
    collect(Runtime(args.parameter, args.gpu, args.checkpoint), output, args.sequence)


if __name__ == "__main__":
    main()
