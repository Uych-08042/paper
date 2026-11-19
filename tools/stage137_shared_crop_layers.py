"""Compare independent SGLA blocks on the same crop and L6 tokens."""
from __future__ import annotations

import argparse
import hashlib
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

import stage134_native_peak_continuation as prior
import stage135_peak_gate as gate
import stage136_peak_closed_loop as full
from stage27_uav_common import checked_image, scene_geometry, tracking_iou
from stage28_response_common import cpu_array, dense_boxes
from stage48_stability_runtime import digest


base, c, io = prior.base, prior.c, prior.io
SCHEMA = "stage137_shared_crop_layers_v1"
STATUS = "COMPLETE_SHARED_CROP_LAYER_DIAGNOSTIC"
SOURCE136 = "1698fa5b98fb0a407f1c9e2519e8b51b0afdbf328d87fe3db99e31754e124883"
LAYERS = tuple(range(7, 13))
GOOD = .50
GROUPS = {"onset_no_good_native_grid": 8, "onset_with_good_native_grid": 11, "healthy": 28}
TABLES = ("summary", "cases", "layers")
CALLS = dict(search_crops=1, prefix_blocks=6, candidate_blocks=6, head_calls=6, selector_calls=1)


def freeze_cases(plan, events):
    by_frame = {(row["sequence"], int(row["onset_frame"])): row for row in events}
    cases, matched = [], set()
    for case in plan["cases"]:
        key = (case["sequence"], case["frame"])
        if case["kind"] == "onset":
            row = by_frame[key]
            matched.add(key)
            source_dense = float(row["dense_best_iou"])
            group = "onset_with_good_native_grid" if source_dense >= GOOD else "onset_no_good_native_grid"
            if group == "onset_no_good_native_grid" and row["category"] != "adequate_crop_no_good":
                raise ValueError("Unexpected no-good-grid onset category")
        else:
            group, source_dense = "healthy", None
        cases.append(dict(case, group=group, source_native_dense_iou=source_dense))
    if (matched != set(by_frame) or len(by_frame) != len(events)
            or len({r["case_id"] for r in cases}) != sum(GROUPS.values())
            or {g: sum(r["group"] == g for r in cases) for g in GROUPS} != GROUPS):
        raise ValueError("Stage-133/134 window roster changed")
    return cases


def collect(rt, image, parent, frame):
    """Execute only the shared prefix and six branch heads, without label inputs."""
    rt.tracker.state = np.asarray(parent, float).tolist()
    rt.tracker.frame_id = frame - 1
    counts = np.zeros(14, int)
    crop_hashes = []
    sample = rt.s1.sample_target

    def sample_once(*args, **kwargs):
        np.testing.assert_array_equal(args[1], parent)
        output = sample(*args, **kwargs)
        crop_hashes.append(hashlib.sha256(np.ascontiguousarray(output[0]).tobytes()).hexdigest())
        return output

    def hook(index):
        def count(*_):
            counts[index] += 1
        return count

    boxes, maps = [], {"raw": [], "size": [], "offset": []}
    with ExitStack() as stack:
        for index, module in enumerate([*rt.backbone.blocks, rt.backbone.MLP, rt.net.box_head]):
            stack.callback(module.register_forward_hook(hook(index)).remove)
        stack.enter_context(patch.object(rt.s1, "sample_target", sample_once))
        stack.enter_context(rt.torch.no_grad())
        search, factor = rt.prepare(image)
        tokens = rt.prefix(search)
        scores = rt.selector(tokens)
        selected = rt.selected_layer(scores)
        for layer in LAYERS:
            out = rt.candidate(tokens, layer)
            boxes.append(rt.decode(out, factor, image))
            for key, tensor in (("raw", "score_map"), ("size", "size_map"), ("offset", "offset_map")):
                shape = (rt.search_side,) * 2
                if key != "raw":
                    shape = (2, *shape)
                maps[key].append(cpu_array(out[tensor]).astype(np.float32).reshape(shape))
    np.testing.assert_array_equal(rt.tracker.state, parent)
    if len(crop_hashes) != 1:
        raise ValueError("Expected exactly one actual search crop")
    return dict(boxes=np.asarray(boxes, float), factor=np.asarray(factor), image_hw=np.asarray(image.shape[:2]),
                parent=np.asarray(parent, float), crop_sha256=np.asarray(crop_hashes[0]), counts=counts,
                selector=cpu_array(scores).reshape(-1), selected=np.asarray(selected),
                **{k: np.asarray(v) for k, v in maps.items()})


def score_case(data, case, reference, truth, ap, same_runtime):
    np.testing.assert_array_equal(data["counts"], [1] * 13 + [6])
    np.testing.assert_array_equal(data["parent"], reference["parent"][0, 0])
    np.testing.assert_array_equal(data["image_hw"], reference["anchor_hw"])
    np.testing.assert_array_equal(data["crop_sha256"], reference["crop_sha256"][0, 0])
    geo = scene_geometry(data["parent"], truth, data["image_hw"], ap["search_factor"], ap["search_size"])
    np.testing.assert_array_equal(geo["crop"], reference["crop"][0, 0])
    np.testing.assert_allclose(data["factor"], geo["factor"], atol=1e-12, rtol=0)
    boxes, scores = data["boxes"], data["selector"]
    if (boxes.shape != (6, 4) or scores.shape != (6,) or not np.isfinite(scores).all()
            or not np.isfinite(boxes).all() or (boxes[:, 2:] <= 0).any()):
        raise ValueError("Invalid six-block output")
    native = LAYERS.index(int(data["selected"]))
    if scores[native] != np.max(scores):
        raise ValueError("Replayed original selector did not pick a top score")
    other = np.flatnonzero(np.arange(len(LAYERS)) != native)
    second = int(other[np.argmax(scores[other])])
    ious = tracking_iou(boxes, truth)
    rows, dense_ious, peaks = [], [], []
    atol = ap["design"]["box_atol"]
    for i, layer in enumerate(LAYERS):
        raw = data["raw"][i]
        weighted = raw * np.asarray(ap["window"], np.float32)
        cell = int(weighted.argmax())
        grid = dense_boxes(data["size"][i], data["offset"][i], data["parent"], float(data["factor"]),
                           ap["search_size"], data["image_hw"])
        np.testing.assert_allclose(grid[cell], boxes[i], atol=atol, rtol=0)
        grid[cell] = boxes[i]
        dense_iou = float(tracking_iou(grid, truth).max())
        dense_ious.append(dense_iou)
        peaks.append(float(weighted.flat[cell]))
        rows.append(dict(case_id=case["case_id"], sequence=case["sequence"], frame=case["frame"], group=case["group"],
                         layer=layer, native=i == native, selector_second=i == second, selector_score=float(scores[i]),
                         hann_peak=peaks[-1], hann_cell=cell, top1_iou=float(ious[i]), dense_best_iou=dense_iou,
                         box_x=float(boxes[i, 0]), box_y=float(boxes[i, 1]), box_width=float(boxes[i, 2]), box_height=float(boxes[i, 3])))
    source_error = float(np.abs(boxes[native] - reference["boxes"][0, 0]).max())
    if same_runtime:
        if int(reference["layer"][0, 0]) != LAYERS[native] or rows[native]["hann_cell"] != int(reference["cells"][0]):
            raise ValueError("Native layer or peak differs from Stage-134")
        np.testing.assert_allclose(boxes[native], reference["boxes"][0, 0], atol=atol, rtol=0)
        for k in ("raw", "size", "offset"):
            np.testing.assert_allclose(data[k][native], reference[f"anchor_{k}"], atol=1e-6, rtol=0)
        if case["source_native_dense_iou"] is not None:
            if (dense_ious[native] >= GOOD) != (case["source_native_dense_iou"] >= GOOD):
                raise ValueError("Native dense good-box classification differs from Stage-133")
    best_alt = int(other[np.argmax(ious[other])])
    best_alt_dense = float(max(dense_ious[i] for i in other))
    peak_choice = native if peaks[native] >= max(peaks) else int(np.argmax(peaks))
    return dict(case, native_layer=LAYERS[native], native_top1_iou=float(ious[native]),
                native_dense_iou=dense_ious[native], source_native_box_error=source_error,
                native_grid_still_without_good=dense_ious[native] < GOOD,
                coverage=float(geo["coverage"]), input_target_short_side=float(min(geo["input_wh"])),
                selector_second_layer=LAYERS[second], selector_second_iou=float(ious[second]),
                raw_selector_score_gap=float(scores[native] - scores[second]),
                peak_choice_layer=LAYERS[peak_choice], peak_choice_iou=float(ious[peak_choice]),
                best_other_layer=LAYERS[best_alt], best_other_top1_iou=float(ious[best_alt]), best_other_dense_iou=best_alt_dense,
                adds_good_standard=bool(ious[native] < GOOD <= ious[best_alt]),
                adds_good_beyond_native_grid=bool(dense_ious[native] < GOOD <= ious[best_alt]),
                adds_good_only_in_other_grid=bool(dense_ious[native] < GOOD <= best_alt_dense and ious[best_alt] < GOOD),
                selector_second_adds_good=bool(ious[native] < GOOD <= ious[second]),
                peak_choice_adds_good=bool(ious[native] < GOOD <= ious[peak_choice]),
                peak_choice_loses_good=bool(ious[peak_choice] < GOOD <= ious[native]), **CALLS), rows


def summarize(cases):
    fields = ("native_grid_still_without_good", "adds_good_standard", "adds_good_beyond_native_grid",
              "adds_good_only_in_other_grid", "selector_second_adds_good", "peak_choice_adds_good", "peak_choice_loses_good")
    table = []
    for group in (*GROUPS, "all_onsets"):
        subset = [r for r in cases if r["group"] == group or group == "all_onsets" and r["kind"] == "onset"]
        if subset:
            table.append(dict(group=group, cases=len(subset), videos=len({r["sequence"] for r in subset}),
                              **{k: sum(r[k] for r in subset) for k in fields}))
    return table


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    report = c.c66.sealed(root / "stage137_decision.json", SCHEMA)
    if (report["plan_fingerprint"] != plan["fingerprint"] or report["status"] != STATUS
            or report["cases"] != len(plan["cases"]) or report["layer_outputs"] != 6 * len(plan["cases"])
            or report["source_stage136"] != SOURCE136 or plan["source_stage136"] != SOURCE136
            or report["calls"] != {k: len(plan["cases"]) * n for k, n in CALLS.items()}
            or len(report["artifact_sha256"]) != 2 * len(plan["cases"])
            or set(report["table_sha256"]) != {f"stage137_{k}.csv" for k in TABLES}):
        raise ValueError("Incomplete Stage-137 diagnostic")
    io.verify({str(root / k): v for k, v in (report["table_sha256"] | report["artifact_sha256"]).items()})


def run(root, r134, plan, p134, ap, load_entry, native):
    rt = native.rt
    rows, layers, artifacts, entries = [], [], {}, {}
    out = root / "observations"
    out.mkdir(exist_ok=True)
    native.assert_frozen()
    if digest(rt.net.box_head) != p134["native_head_sha256"]:
        raise ValueError("Frozen native head differs")
    for case in plan["cases"]:
        cid, seq, frame = (case[k] for k in ("case_id", "sequence", "frame"))
        if seq not in entries:
            entries[seq] = load_entry(seq)
        entry = entries[seq]
        reference = c.read_cache(r134 / "branches" / f"{cid}.npz", dict(plan_fingerprint=p134["fingerprint"], case_id=cid))
        if reference is None or entry["sequence"] != seq or int(reference["frame_id"][0]) != frame:
            raise ValueError(f"Frozen Stage-134 case differs: {cid}")
        path, meta = out / f"{cid}.npz", dict(plan_fingerprint=plan["fingerprint"], case_id=cid)
        data = c.read_cache(path, meta)
        computed = data is None
        if computed:
            rt.initialize(checked_image(ap["root"], entry["frames"][0]), entry["initial_box"])
            data = collect(rt, checked_image(ap["root"], entry["frames"][frame]), reference["parent"][0, 0], frame)
        row, candidates = score_case(data, case, reference, entry["ground_truth"][frame], ap, not plan["runtime_differences"])
        if computed:
            c.save(path, data, meta)
        for item in (path, path.with_suffix(".receipt.json")):
            artifacts[item.relative_to(root).as_posix()] = c.file_hash(item)
        rows.append(row)
        layers.extend(candidates)
        print(f"[Stage-137] {cid}: six blocks scored", flush=True)
    native.assert_frozen()
    summary = summarize(rows)
    tables = {}
    for name, values in zip(TABLES, (summary, rows, layers)):
        path = root / f"stage137_{name}.csv"
        prior.publish_table(path, values)
        tables[path.name] = c.file_hash(path)
    c.publish_identical(root / "stage137_decision.json", io.seal(dict(
        schema=SCHEMA, status=STATUS, role="consumed_development_fixed_crop_layer_supply_probe",
        plan_fingerprint=plan["fingerprint"], source_stage136=plan["source_stage136"], source_stage134=gate.SOURCE134,
        source_stage133=prior.SOURCE133, runtime=plan["runtime"], runtime_differences=plan["runtime_differences"],
        cases=len(rows), layer_outputs=len(layers), calls={k: sum(r[k] for r in rows) for k in CALLS},
        training_updates=0, new_full_trajectories=0, confirmation_inference_calls=0,
        table_sha256=tables, artifact_sha256=artifacts)))
    check_results(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76", "stage89", "stage93", "stage133", "stage134", "stage136")
    for name in names:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage137", parameter="deit_distilled")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not args.checkpoint or any(not getattr(args, k) for k in names):
        parser.error("Supply the frozen source directories and original checkpoint")
    r76, r89, r93, r133, r134, r136 = paths = [getattr(args, k).expanduser().resolve() for k in names]
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate Stage-137 directory")
    if not (r134 / "manifest.json").is_file():
        parser.error("Need the completed Stage-134 server caches")
    with prior.writer_lock(root):
        if (root / "stage137_decision.json").is_file():
            check_results(root)
            return
        prior.check_results(r134)
        prior.audit.check_results(r133)
        d133 = c.c66.sealed(r133 / "stage133_decision.json", prior.audit.SCHEMA)
        p134 = c.c66.sealed(r134 / "manifest.json", prior.SCHEMA)
        d134 = c.c66.sealed(r134 / "stage134_decision.json", prior.SCHEMA)
        d136 = c.c66.sealed(r136 / "stage136_decision.json", full.SCHEMA)
        if (d133["fingerprint"] != prior.SOURCE133 or d134["fingerprint"] != gate.SOURCE134
                or d136["fingerprint"] != SOURCE136
                or d136["status"] != full.STATUS or d136["source_stage121"] != p134["source_stage121"]):
            raise ValueError("Need the archived Stage-133/134/136 results")
        cases = freeze_cases(p134, prior.read_table(r133 / "stage133_events.csv"))
        s95 = base.s103.prior.s95
        p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
        old = origin["original"]
        roster = {r["sequence"] for r in cases}
        specs = {spec["unit"]["sequence"]: spec for spec in p89["units"] if spec["unit"]["sequence"] in roster}
        if set(specs) != roster:
            raise ValueError("Stage-89 video roster differs from the matched cases")

        def load_entry(seq):
            return s95.load_unit(r93, r89, p93, p89, d89, origin, specs[seq])[0]

        import torch
        torch.set_num_threads(1)
        runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
        diffs = base.s109.check_runtime(p134["runtime"], runtime)
        manifest = root / "manifest.json"
        if manifest.is_file() and c.c66.sealed(manifest, SCHEMA)["runtime"] != runtime:
            raise ValueError("GPU/runtime changed on resume. Set STAGE137_ROOT to a new directory.")
        plan = io.seal(dict(schema=SCHEMA, cases=cases, layers=list(LAYERS), good_iou=GOOD, calls_per_case=CALLS,
                            source_stage136=d136["fingerprint"], source_stage134=d134["fingerprint"],
                            source_roster_fingerprints=[p93["fingerprint"], p89["fingerprint"], old["fingerprint"]],
                            runtime=runtime, runtime_differences=diffs, code_sha256=c.file_hash(Path(__file__))))
        c.publish_identical(manifest, plan)
        frozen = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = base.s103.s76.startup.checked_runtime(args, frozen, root)
        run(root, r134, plan, p134, base.e63.c.actor_plan(old), load_entry, native)
    print(f"Stage-137: {STATUS}", flush=True)


if __name__ == "__main__":
    main()
