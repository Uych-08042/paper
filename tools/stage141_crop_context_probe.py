"""Replace outer context while preserving the large crop and its target scale."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import stage140_crop_center_size_cross as source
from stage139_post_write_diagnostic import diagnose
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, crop_geometry, scene_geometry
from stage46_shift_audit import publish_table


base, c, io = source.base, source.c, source.io
SCHEMA = "stage141_crop_context_probe_v1"
STATUS = "COMPLETE_MATCHED_OUTER_CONTEXT_DIAGNOSTIC"
SOURCE140 = "d7d990ea00b5843e1ce1f33968cb80006bbf5ec91665fd0d1ccb806614eace46"
CENTERS = ("native", "cross")
ARMS = ("small_full", "large_full", "large_mean_fill", "large_edge_fill")
MEAN_RGB = (124, 116, 104)
TABLES = ("summary", "frames", "effects", "scope")


def crop_pairs(crops):
    pairs = np.asarray([sorted(pair, key=lambda i: crops[i, 2]) for pair in ((0, 2), (1, 3))])
    for a, b in pairs:
        small, large = crops[a], crops[b]
        if (small[2] >= large[2] or (large[:2] > small[:2]).any()
                or (small[:2] + small[2:] > large[:2] + large[2:]).any()):
            raise ValueError("Expected strictly nested source crops at each fixed center")
    return pairs


def fill_outer(image, support, mode):
    x, y, w, h = map(int, support)
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
        raise ValueError("Invalid unmasked source support")
    if mode == "mean":
        output = np.empty_like(image)
        output[...] = MEAN_RGB
        output[y:y+h, x:x+w] = image[y:y+h, x:x+w]
        return output
    if mode == "edge":
        ys = np.clip(np.arange(image.shape[0]), y, y+h-1)
        xs = np.clip(np.arange(image.shape[1]), x, x+w-1)
        return image[ys[:, None], xs[None, :]]
    raise ValueError(f"Unknown fill: {mode}")


def collect(observer, point, entry, frozen, pairs, ap, image_reader=checked_image, observe=base.e63.actor.observe):
    frames, frame = entry["frames"], point["frame"]
    observer.initialize(image_reader(ap["root"], frames[0]), entry["initial_box"])
    image = image_reader(ap["root"], frames[frame])
    values = []
    for a, b in pairs:
        small, large = frozen["parent"][[a, b]]
        _, tight_support, _ = crop_geometry(small, image.shape[:2], ap["search_factor"], ap["search_size"])
        values.append([observe(observer, pixels, parent.copy(), frame, ap) for pixels, parent in (
            (image, small), (image, large), (fill_outer(image, tight_support, "mean"), large),
            (fill_outer(image, tight_support, "edge"), large))])
    fields = ("boxes", "parent", "crop", "crop_sha256", "raw", "size", "offset", "factor", "image_hw",
              "head_calls", "selector_calls", "block_calls", "layer")
    return {k: np.asarray([[row[k] for row in arm_rows] for arm_rows in values]) for k in fields}


def verify(data, frozen, pairs, same_runtime, atol):
    base.e63.source_common.prior.audit_calls(data, (len(CENTERS), len(ARMS)))
    boxes = data["boxes"]
    if boxes.shape != (2, 4, 4) or not np.isfinite(boxes).all() or (boxes[..., 2:] <= 0).any():
        raise ValueError("Invalid Stage-141 boxes")
    for k in ("parent", "crop", "crop_sha256", "factor", "image_hw"):
        np.testing.assert_array_equal(data[k][:, :2], frozen[k][pairs], err_msg=k)
    for k in ("parent", "crop", "factor", "image_hw"):
        for i in (2, 3):
            np.testing.assert_array_equal(data[k][:, i], data[k][:, 1], err_msg=k)
    original = frozen["boxes"][pairs]
    if same_runtime:
        np.testing.assert_allclose(boxes[:, :2], original, rtol=0, atol=atol)
        np.testing.assert_array_equal(data["layer"][:, :2], frozen["layer"][pairs])
        for k in ("raw", "size", "offset"):
            np.testing.assert_allclose(data[k][:, :2], frozen[k][pairs], rtol=0, atol=1e-6)
    return float(np.abs(boxes[:, :2] - original).max())


def score(data, point, truth, ap):
    rows, effects = [], []
    for z, center in enumerate(CENTERS):
        ident = dict(point, center=center)
        subset = []
        tight_geo = scene_geometry(data["parent"][z, 0], truth, data["image_hw"][z, 0],
                                   ap["search_factor"], ap["search_size"])
        for i, arm in enumerate(ARMS):
            parent, hw = data["parent"][z, i], data["image_hw"][z, i]
            geo = scene_geometry(parent, truth, hw, ap["search_factor"], ap["search_size"])
            np.testing.assert_array_equal(data["crop"][z, i], geo["crop"])
            support = geo if i < 2 else tight_geo
            result = diagnose(data["raw"][z, i], data["size"][z, i], data["offset"][z, i], parent, geo["factor"],
                              hw, data["boxes"][z, i], truth, support, ap)
            subset.append(dict(ident, arm=arm, layer=int(data["layer"][z, i]), crop_side=int(geo["crop"][2]),
                               geometric_coverage=float(geo["coverage"]),
                               unmasked_target_coverage=float(support["coverage"]),
                               input_target_short_side=float(min(geo["input_wh"])),
                               crop_sha256=str(data["crop_sha256"][z, i]), **result))
        for metric in source.METRICS:
            small, large, mean_fill, edge_fill = (r[metric] for r in subset)
            effects.append(dict(ident, metric=metric, target_fully_kept=subset[0]["geometric_coverage"] == 1.,
                                small_full=small, large_full=large, large_mean_fill=mean_fill, large_edge_fill=edge_fill,
                                large_minus_small=large-small, mean_fill_minus_large=mean_fill-large,
                                edge_fill_minus_large=edge_fill-large, mean_fill_minus_small=mean_fill-small,
                                edge_fill_minus_small=edge_fill-small))
        rows.extend(subset)
    return rows, effects


def summarize(effects):
    groups = dict.fromkeys(r["group"] for r in effects)
    deltas = ("large_minus_small", "mean_fill_minus_large", "edge_fill_minus_large", "mean_fill_minus_small", "edge_fill_minus_small")
    result = []
    for group in groups:
        for metric in source.METRICS:
            rows = [r for r in effects if (r["group"], r["metric"]) == (group, metric)]
            kept = [r for r in rows if r["target_fully_kept"]]
            for support, subset in (("all", rows), ("fully_kept", kept)):
                if subset:
                    result.append(dict(group=group, metric=metric, support=support, pairs=len(subset),
                                       **{k: float(np.mean([r[k] for r in subset])) for k in deltas}))
    return result


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    result = c.c66.sealed(root / "stage141_decision.json", SCHEMA)
    if (result["status"] != STATUS or result["plan_fingerprint"] != plan["fingerprint"]
            or result["source_stage140"] != SOURCE140 or result["points"] != len(plan["points"])
            or result["searches"] != 8 * len(plan["points"]) or len(result["artifact_sha256"]) != 2 * len(plan["points"])
            or set(result["table_sha256"]) != {f"stage141_{n}.csv" for n in TABLES}):
        raise ValueError("Incomplete Stage-141 diagnostic")
    io.verify({str(root / k): h for k, h in (result["table_sha256"] | result["artifact_sha256"]).items()})


def run(root, plan, r140, p140, observer, ap, load_entry):
    rows, effects, scope, evidence, entries = [], [], [], {}, {}
    out = root / "responses"
    out.mkdir(exist_ok=True)
    for point in plan["points"]:
        pid, seq, frame = (point[k] for k in ("point_id", "sequence", "frame"))
        if seq not in entries:
            entries[seq] = load_entry(seq)
        entry = entries[seq]
        frozen = c.read_cache(r140 / "crosses" / f"{pid}.npz", dict(plan_fingerprint=p140["fingerprint"], point_id=pid))
        if frozen is None or entry["sequence"] != seq:
            raise ValueError(f"Missing or mismatched Stage-140 source: {pid}")
        pairs = crop_pairs(frozen["crop"])
        path, key = out / f"{pid}.npz", dict(plan_fingerprint=plan["fingerprint"], point_id=pid)
        data = c.read_cache(path, key)
        computed = data is None
        if computed:
            data = collect(observer, point, entry, frozen, pairs, ap)
        error = verify(data, frozen, pairs, not plan["runtime_differences"], ap["design"]["box_atol"])
        detail, contrasts = score(data, point, entry["ground_truth"][frame], ap)
        if computed:
            c.save(path, data, key)
        for item in (path, path.with_suffix(".receipt.json")):
            evidence[item.relative_to(root).as_posix()] = c.file_hash(item)
        rows.extend(detail)
        effects.extend(contrasts)
        scope.append(dict(point, source_max_box_error=error, searches=int(data["head_calls"].sum()),
                          mean_fill_changed_pairs=int((data["crop_sha256"][:, 2] != data["crop_sha256"][:, 1]).sum()),
                          edge_fill_changed_pairs=int((data["crop_sha256"][:, 3] != data["crop_sha256"][:, 1]).sum())))
        print(f"[Stage-141] {pid}: {scope[-1]['searches']} searches", flush=True)
    tables = {}
    for name, table in zip(TABLES, (summarize(effects), rows, effects, scope)):
        path = root / f"stage141_{name}.csv"
        publish_table(path, table)
        tables[path.name] = c.file_hash(path)
    c.publish_identical(root / "stage141_decision.json", io.seal(dict(
        schema=SCHEMA, status=STATUS, role="consumed_development_input_context_intervention",
        plan_fingerprint=plan["fingerprint"], source_stage140=SOURCE140, runtime=plan["runtime"],
        runtime_differences=plan["runtime_differences"], points=len(scope), searches=sum(r["searches"] for r in scope),
        training_updates=0, state_updates=0, confirmation_inference_calls=0,
        table_sha256=tables, artifact_sha256=evidence)))
    check_results(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76", "stage89", "stage93", "stage140")
    for name in names:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage141", parameter="deit_distilled")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not args.checkpoint or any(not getattr(args, k) for k in names):
        parser.error("Supply the frozen source directories and original checkpoint")
    r76, r89, r93, r140 = paths = [getattr(args, k).expanduser().resolve() for k in names]
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate Stage-141 directory")
    if not (r140 / "stage140_decision.json").is_file():
        parser.error("Need the completed Stage-140 server run")
    with writer_lock(root):
        if (root / "stage141_decision.json").is_file():
            check_results(root)
            return
        source.check_results(r140)
        p140 = c.c66.sealed(r140 / "manifest.json", source.SCHEMA)
        d140 = c.c66.sealed(r140 / "stage140_decision.json", source.SCHEMA)
        if d140["fingerprint"] != SOURCE140 or p140["arms"] != list(source.ARMS) or len(p140["points"]) != 11:
            raise ValueError("Need the archived Stage-140 crossed crop result")
        s95 = base.s103.prior.s95
        p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
        old = origin["original"]
        specs = {r["unit"]["sequence"]: r for r in p89["units"]}

        def load_entry(seq):
            return s95.load_unit(r93, r89, p93, p89, d89, origin, specs[seq])[0]

        import torch
        torch.set_num_threads(1)
        runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
        differences = base.s109.check_runtime(p140["runtime"], runtime)
        manifest = root / "manifest.json"
        if manifest.is_file() and c.c66.sealed(manifest, SCHEMA)["runtime"] != runtime:
            raise ValueError("GPU/runtime changed on resume. Set STAGE141_ROOT to a new directory.")
        plan = io.seal(dict(schema=SCHEMA, points=p140["points"], arms=list(ARMS), mean_rgb=list(MEAN_RGB),
                            source_stage140=SOURCE140, runtime=runtime, runtime_differences=differences,
                            code_sha256=c.file_hash(Path(__file__))))
        c.publish_identical(manifest, plan)
        frozen = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = base.s103.s76.startup.checked_runtime(args, frozen, root)
        observer = base.e63.Capture(native)
        try:
            observer.assert_frozen()
            run(root, plan, r140, p140, observer, base.e63.c.actor_plan(old), load_entry)
            observer.assert_frozen()
        finally:
            observer.handle.remove()
    print(f"Stage-141: {STATUS}", flush=True)


if __name__ == "__main__":
    main()
