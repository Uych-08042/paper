"""Cross archived native and cross-layer crop center and size on matched frames."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import stage139_post_write_diagnostic as prior
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, scene_geometry
from stage46_shift_audit import publish_table


base, c, io = prior.base, prior.c, prior.io
SCHEMA = "stage140_crop_center_size_cross_v1"
STATUS = "COMPLETE_MATCHED_CROP_CENTER_SIZE_CROSS"
SOURCE139 = "ff5fba73d06336b895809a5d385d828dddd0e2540e7e81e47ec4846cdd29208b"
PRIMARY = ("uav0000237_00001_s", 243)
POINT_GROUPS = (
    (PRIMARY[0], range(243, 251), "cross_layer_top1_oracle", "target"),
    ("uav0000089_00920_s", (1165, 1166), "cross_layer_top1_oracle", "ranking_control"),
    ("uav0000349_02668_s", (409,), "cross_layer_max_hann", "healthy_control"),
)
ARMS = ("native_center_native_size", "cross_center_native_size", "native_center_cross_size", "cross_center_cross_size")
METRICS = ("replay_iou", "peak_pool_best_iou", "dense_best_iou")
TABLES = ("summary", "frames", "effects", "scope")


def freeze_points(plan):
    by_sequence = {case["sequence"]: case for case in plan["cases"]}
    points = []
    for sequence, frames, source_arm, group in POINT_GROUPS:
        case = by_sequence[sequence]
        for frame in frames:
            offset = frame - case["frame"]
            ordinal = case["offsets"].index(offset)
            points.append(dict(point_id=f"{sequence}__f{frame:06d}", sequence=sequence, frame=frame, offset=offset,
                               case_id=case["case_id"], ordinal=ordinal, cross_arm=source_arm,
                               group="primary" if (sequence, frame) == PRIMARY else group))
    return points


def cross_parents(native, cross):
    native, cross = np.asarray(native, float), np.asarray(cross, float)
    dims = np.stack((native[2:], cross[2:]))
    if np.array_equal(dims[0], dims[1]):
        return np.stack((native, cross, native, cross))
    origins = np.stack((native[:2], cross[:2]))
    centers = origins + .5 * dims
    if np.array_equal(centers[0], centers[1]):
        return np.stack((native, native, cross, cross))
    cross_center_native_size = np.r_[centers[1] - .5 * dims[0], dims[0]]
    native_center_cross_size = np.r_[centers[0] - .5 * dims[1], dims[1]]
    # Reuse the original endpoints verbatim to keep subpixel box identity.
    return np.stack((native, cross_center_native_size, native_center_cross_size, cross))


def unique_mask(parents):
    return np.asarray([tuple(parent) not in {tuple(p) for p in parents[:i]} for i, parent in enumerate(parents)])


def collect(observer, point, entry, parents, ap, image_reader=checked_image, observe=base.e63.actor.observe):
    frames = entry["frames"]
    observer.initialize(image_reader(ap["root"], frames[0]), entry["initial_box"])
    image = image_reader(ap["root"], frames[point["frame"]])
    seen, rows = {}, []
    for parent in parents:
        key = tuple(parent)
        if key not in seen:
            seen[key] = observe(observer, image, parent.copy(), point["frame"], ap)
        rows.append(seen[key])
    fields = ("boxes", "parent", "crop", "crop_sha256", "raw", "size", "offset", "factor", "image_hw",
              "head_calls", "selector_calls", "block_calls", "layer")
    return dict(executed=unique_mask(parents), **{k: np.asarray([r[k] for r in rows]) for k in fields})


def verify(data, frozen, ordinal, cross_index, parents, same_runtime, atol):
    original = [0, cross_index]
    endpoints = [0, 3]
    np.testing.assert_array_equal(data["parent"], parents)
    np.testing.assert_array_equal(data["executed"], unique_mask(parents))
    for k in ("parent", "crop", "crop_sha256", "image_hw"):
        np.testing.assert_array_equal(data[k][endpoints], frozen[k][ordinal, original], err_msg=k)
    base.e63.source_common.prior.audit_calls(data, (len(ARMS),))
    boxes = data["boxes"]
    if boxes.shape != (len(ARMS), 4) or not np.isfinite(boxes).all() or (boxes[:, 2:] <= 0).any():
        raise ValueError("Invalid crossed crop output")
    cached = frozen["boxes"][ordinal, original]
    error = float(np.abs(boxes[endpoints] - cached).max())
    if same_runtime:
        np.testing.assert_allclose(boxes[endpoints], cached, rtol=0, atol=atol)
        np.testing.assert_array_equal(data["layer"][endpoints], frozen["layer"][ordinal, original])
        for k in ("raw", "size", "offset"):
            np.testing.assert_allclose(data[k][endpoints], frozen[k][ordinal, original], rtol=0, atol=1e-6)
    return error, int((data["head_calls"] * data["executed"]).sum())


def score(data, point, truth, ap):
    rows = []
    for i, arm in enumerate(ARMS):
        parent, hw = data["parent"][i], data["image_hw"][i]
        geo = scene_geometry(parent, truth, hw, ap["search_factor"], ap["search_size"])
        np.testing.assert_array_equal(data["crop"][i], geo["crop"])
        np.testing.assert_allclose(data["factor"][i], geo["factor"], rtol=0, atol=1e-12)
        row = prior.diagnose(data["raw"][i], data["size"][i], data["offset"][i], parent, geo["factor"],
                             hw, data["boxes"][i], truth, geo, ap)
        rows.append(dict(point, arm=arm, layer=int(data["layer"][i]), center_x=float(parent[0] + .5 * parent[2]),
                         center_y=float(parent[1] + .5 * parent[3]), width=float(parent[2]), height=float(parent[3]),
                         crop_left=float(geo["crop"][0]), crop_top=float(geo["crop"][1]), crop_side=float(geo["crop"][2]),
                         crop_sha256=str(data["crop_sha256"][i]), coverage=float(geo["coverage"]),
                         target_center_inside=bool(geo["center_in_usable"]), input_target_short_side=float(min(geo["input_wh"])),
                         **row))
    return rows


def effects(rows, point):
    result = []
    for metric in METRICS:
        nn, xn, nx, xx = (r[metric] for r in rows)
        result.append(dict(point, metric=metric, native_center_native_size=nn, cross_center_native_size=xn,
                           native_center_cross_size=nx, cross_center_cross_size=xx,
                           cross_center_effect_native_size=xn - nn, cross_center_effect_cross_size=xx - nx,
                           cross_size_effect_native_center=nx - nn, cross_size_effect_cross_center=xx - xn,
                           restore_native_center=nx - xx, restore_native_size=xn - xx,
                           interaction=xx - xn - nx + nn))
    return result


def summarize(rows):
    groups = dict.fromkeys(row["group"] for row in rows)
    result = []
    for group in groups:
        for arm in ARMS:
            selected = [r for r in rows if r["group"] == group and r["arm"] == arm]
            result.append(dict(group=group, arm=arm, frames=len(selected),
                               fully_covered=sum(r["coverage"] == 1 for r in selected),
                               **{f"mean_{m}": float(np.mean([r[m] for r in selected])) for m in METRICS},
                               **{f"good_{m}_frames": sum(r[m] >= prior.DESIGN["good_iou"] for r in selected) for m in METRICS}))
    return result


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    report = c.c66.sealed(root / "stage140_decision.json", SCHEMA)
    if (report["status"] != STATUS or report["plan_fingerprint"] != plan["fingerprint"]
            or report["source_stage139"] != SOURCE139 or report["points"] != len(plan["points"])
            or report["logical_searches"] != len(ARMS) * len(plan["points"])
            or not 0 < report["physical_searches"] <= report["logical_searches"]
            or len(report["artifact_sha256"]) != 2 * len(plan["points"])
            or set(report["table_sha256"]) != {f"stage140_{name}.csv" for name in TABLES}):
        raise ValueError("Incomplete Stage-140 diagnostic")
    io.verify({str(root / k): v for k, v in (report["table_sha256"] | report["artifact_sha256"]).items()})


def run(root, plan, r139, p139, observer, ap, load_entry):
    rows, contrasts, scope, artifacts, entries, source_data = [], [], [], {}, {}, {}
    cache = root / "crosses"
    cache.mkdir(exist_ok=True)
    for point in plan["points"]:
        cid, seq, ordinal = (point[k] for k in ("case_id", "sequence", "ordinal"))
        if seq not in entries:
            entries[seq] = load_entry(seq)
        entry = entries[seq]
        if cid not in source_data:
            source_data[cid] = c.read_cache(r139 / "replays" / f"{cid}.npz",
                                            dict(plan_fingerprint=p139["fingerprint"], case_id=cid))
        frozen = source_data[cid]
        if frozen is None or entry["sequence"] != seq or int(frozen["offsets"][ordinal]) != point["offset"]:
            raise ValueError("Missing or mismatched archived Stage-139 point")
        index = prior.ARMS.index(point["cross_arm"])
        parents = cross_parents(*frozen["parent"][ordinal, [0, index]])
        point_id = point["point_id"]
        path, meta = cache / f"{point_id}.npz", dict(plan_fingerprint=plan["fingerprint"], point_id=point_id)
        data = c.read_cache(path, meta)
        computed = data is None
        if computed:
            data = collect(observer, point, entry, parents, ap)
        error, calls = verify(data, frozen, ordinal, index, parents, not plan["runtime_differences"], ap["design"]["box_atol"])
        point_rows = score(data, point, entry["ground_truth"][point["frame"]], ap)
        if computed:
            c.save(path, data, meta)
        for item in (path, path.with_suffix(".receipt.json")):
            artifacts[item.relative_to(root).as_posix()] = c.file_hash(item)
        rows.extend(point_rows)
        contrasts.extend(effects(point_rows, point))
        scope.append(dict(point, logical_searches=len(ARMS), physical_searches=calls,
                          distinct_pixel_crops=len(set(data["crop_sha256"])), source_endpoint_box_error=error))
        print(f"[Stage-140] {point_id}: {calls} physical searches", flush=True)
    tables = {}
    for name, records in zip(TABLES, (summarize(rows), rows, contrasts, scope)):
        path = root / f"stage140_{name}.csv"
        publish_table(path, records)
        tables[path.name] = c.file_hash(path)
    primary = [r for r in contrasts if r["group"] == "primary"]
    c.publish_identical(root / "stage140_decision.json", io.seal(dict(
        schema=SCHEMA, status=STATUS, role="consumed_development_fixed_frame_factorial_crop_diagnostic",
        plan_fingerprint=plan["fingerprint"], source_stage139=SOURCE139, primary=primary,
        runtime=plan["runtime"], runtime_differences=plan["runtime_differences"], points=len(scope),
        logical_searches=len(rows), physical_searches=sum(r["physical_searches"] for r in scope),
        state_updates=0, training_updates=0, confirmation_inference_calls=0,
        table_sha256=tables, artifact_sha256=artifacts)))
    check_results(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76", "stage89", "stage93", "stage139")
    for name in names:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage140", parameter="deit_distilled")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not args.checkpoint or any(not getattr(args, k) for k in names):
        parser.error("Supply the frozen source directories and original checkpoint")
    r76, r89, r93, r139 = paths = [getattr(args, k).expanduser().resolve() for k in names]
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate Stage-140 directory")
    if not (r139 / "stage139_decision.json").is_file():
        parser.error("Need the completed Stage-139 server replays")
    with writer_lock(root):
        if (root / "stage140_decision.json").is_file():
            check_results(root)
            return
        prior.check_results(r139)
        p139 = c.c66.sealed(r139 / "manifest.json", prior.SCHEMA)
        d139 = c.c66.sealed(r139 / "stage139_decision.json", prior.SCHEMA)
        if d139["fingerprint"] != SOURCE139:
            raise ValueError("Need the archived Stage-139 replay result")
        points = freeze_points(p139)
        s95 = base.s103.prior.s95
        p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
        old = origin["original"]
        specs = {s["unit"]["sequence"]: s for s in p89["units"]}

        def load_entry(seq):
            return s95.load_unit(r93, r89, p93, p89, d89, origin, specs[seq])[0]

        import torch
        torch.set_num_threads(1)
        runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
        differences = base.s109.check_runtime(p139["runtime"], runtime)
        manifest = root / "manifest.json"
        if manifest.is_file() and c.c66.sealed(manifest, SCHEMA)["runtime"] != runtime:
            raise ValueError("GPU/runtime changed on resume. Set STAGE140_ROOT to a new directory.")
        plan = io.seal(dict(schema=SCHEMA, points=points, arms=list(ARMS), source_stage139=SOURCE139,
                            runtime=runtime, runtime_differences=differences, code_sha256=c.file_hash(Path(__file__))))
        c.publish_identical(manifest, plan)
        frozen = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = base.s103.s76.startup.checked_runtime(args, frozen, root)
        observer = base.e63.Capture(native)
        try:
            observer.assert_frozen()
            run(root, plan, r139, p139, observer, base.e63.c.actor_plan(old), load_entry)
            observer.assert_frozen()
        finally:
            observer.handle.remove()
    print(f"Stage-140: {STATUS}", flush=True)


if __name__ == "__main__":
    main()
