"""Cross the center and size of the two frozen Stage-150 writes."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import stage150_gray_write_continuation as prior
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, tracking_iou
from stage46_shift_audit import publish_table

c, io, base = prior.c, prior.io, prior.base
SCHEMA = "stage151_write_geometry_cross_v1"
STATUS = "COMPLETE_GRAY_WRITE_GEOMETRY_CROSS"
SOURCE150 = "409db48355ab300463dd0c2ef225059fd8f7b65d8928d6f85f047f45caac22c0"
ARMS = ("regression", "gray_center", "gray_size", "gray_once")
CONTROL = [0, 3]
DESIGN = prior.DESIGN | dict(arms=list(ARMS), successors_per_arm=357)
TABLES = ("first_search", "windows", "paired", "scope", "timeline")
WORK = dict(prior.WORK)


def cross(case):
    """Use xywh boxes with the exchanged component and no extra clipping."""
    source = np.asarray(case["anchor_boxes"], float)
    sizes = source[:, 2:]
    centers = source[:, :2] + .5 * sizes
    width_height = sizes[[0, 0, 1, 1]]
    xy = centers[[0, 1, 0, 1]] - .5 * width_height
    parents = np.hstack((xy, width_height))
    parents[CONTROL] = source
    return dict(case, anchor_boxes=parents.tolist())


def collect(observer, case, frames, initial_box, model, ap, *,
            read_image=checked_image, observe=base.e63.actor.observe):
    """Each arm starts from a frozen parent and follows its own regression boxes."""
    frame_ids = np.arange(case["frame"] + 1, case["frame"] + case["successors"] + 1)
    branches = [[] for _ in ARMS]
    grids, pools, hann_cells, usable = [], [], [], []
    if frame_ids.size:
        observer.initialize(read_image(ap["root"], frames[0]), initial_box)
        parents = np.array(case["anchor_boxes"], float)
        for frame in frame_ids:
            image = read_image(ap["root"], frames[frame])
            for j, parent in enumerate(parents):
                obs = observe(observer, image, parent.copy(), int(frame), ap)
                inputs, grid, native, cell = prior.decode_search(obs, model, ap)
                branches[j].append(dict(obs, cell=cell, boxes=grid[cell].copy()))
                parents[j] = grid[cell]
                if frame == frame_ids[0]:
                    grids.append(grid)
                    dense = np.zeros(len(grid), dtype=bool)
                    dense[inputs["cells"][inputs["pools"]["dense"]]] = True
                    pools.append(dense)
                    hann_cells.append(native)
                    usable.append(obs["usable"])
    data = {field:np.asarray([[row[field] for row in branch] for branch in branches]) for field in prior.FIELDS}
    return dict(data, frame_id=frame_ids, first_grid=np.asarray(grids), first_pool=np.asarray(pools),
                first_hann_cell=np.asarray(hann_cells, dtype=int), first_usable=np.asarray(usable))


def audit(data, case, old, exact_runtime, grid_cells):
    prior.audit_branch(data, case)
    np.testing.assert_array_equal(data["frame_id"], old["frame_id"])
    n = case["successors"]
    grid, pool, hann, usable = (data["first_" + k] for k in ("grid", "pool", "hann_cell", "usable"))
    if not n:
        if any(a.size for a in (grid, pool, hann, usable)):
            raise ValueError("Unexpected search at the end of a sequence")
        return dict(max_control_box_error=None, control_rows_equal=0)
    count = len(ARMS)
    if (grid.shape != (count, grid_cells, 4) or pool.shape != (count, grid_cells) or pool.dtype != bool
            or usable.shape != (count, 4) or hann.shape != (count,)
            or not np.isfinite(grid).all() or not np.isfinite(usable).all()
            or not np.isin(hann, np.arange(grid_cells)).all()):
        raise ValueError("Invalid first-search evidence")
    chosen = data["cell"][:, 0]
    if not np.isin(chosen, np.arange(grid_cells)).all() or not pool[np.arange(count), chosen].all():
        raise ValueError("Regression selected outside its pool")
    np.testing.assert_array_equal(grid[np.arange(count), chosen], data["boxes"][:, 0])
    comparable = np.ones((len(CONTROL), n), dtype=bool)
    for key in prior.FIELDS:
        current = data[key][CONTROL]
        if exact_runtime:
            np.testing.assert_array_equal(current, old[key], err_msg=key)
        comparable &= (current == old[key]).reshape(len(CONTROL), n, -1).all(axis=2)
    return dict(max_control_box_error=float(np.max(np.abs(data["boxes"][CONTROL] - old["boxes"]))),
                control_rows_equal=int(comparable.sum()))


def usable_coverage(box, truth):
    lo = np.maximum(box[:2], truth[:2])
    hi = np.minimum(box[:2]+box[2:], truth[:2]+truth[2:])
    return float(np.maximum(hi-lo, 0.).prod()/truth[2:].prod())


def first_search(data, labels, truth):
    if not data["frame_id"].size:
        return []
    t = int(data["frame_id"][0])
    target = truth[t]
    output = []
    for k, arm in enumerate(ARMS):
        grid = tracking_iou(data["first_grid"][k], target)
        pool = grid[data["first_pool"][k]]
        chosen = int(data["cell"][k, 0])
        good = DESIGN["good_iou"]
        if grid.max() < 0.:
            outcome = "invalid_label"
        elif grid.max() < good:
            outcome = "grid_without_good_box"
        elif pool.max() < good:
            outcome = "pool_missed_good_box"
        elif grid[chosen] < good:
            outcome = "regression_missed_good_box"
        else:
            outcome = "selected_good_box"
        valid = outcome != "invalid_label"
        crop = data["crop"][k, 0]
        parent = data["parent"][k, 0]
        output.append(dict(
            labels, arm=arm, successor_frame=t,
            center_x=float(parent[0]+.5*parent[2]), center_y=float(parent[1]+.5*parent[3]),
            width=float(parent[2]), height=float(parent[3]), crop_side=float(crop[2]),
            same_crop_pixels_as_regression=data["crop_sha256"][k, 0] == data["crop_sha256"][0, 0],
            coverage=usable_coverage(data["first_usable"][k], target) if valid else None,
            target_short_side_in_input=float(min(target[2:]) * labels["search_size"] / crop[2]) if valid else None,
            hann_iou=float(grid[data["first_hann_cell"][k]]) if valid else None,
            grid_best_iou=float(grid.max()) if valid else None,
            pool_best_iou=float(pool.max()) if valid else None, pool_size=len(pool),
            selected_iou=float(grid[chosen]) if valid else None, outcome=outcome))
    return output


def describe(data, case, truth, ap):
    labels = {k:case[k] for k in ("case_id", "sequence", "source", "frame", "onset")}
    frame_ids = data["frame_id"]
    n = len(frame_ids)
    tail = tracking_iou(data["boxes"], truth[frame_ids]) if n else np.empty((len(ARMS), 0))
    anchor = tracking_iou(case["anchor_boxes"], truth[case["frame"]])
    valid = (tail >= 0.).all(axis=0)
    windows, timeline = [], []
    for k, arm in enumerate(ARMS):
        values = tail[k]
        row = dict(labels, arm=arm, anchor_iou=float(anchor[k]))
        for horizon in DESIGN["horizons"]:
            mask = valid[:horizon]
            v = values[:horizon]
            windows.append(dict(row, horizon=horizon, successors=len(v), valid_successors=int(mask.sum()),
                full_labeled_window=n >= horizon and bool(mask.all()),
                successor_mean_iou=float(v[mask].mean()) if mask.any() else None,
                successor_last_iou=float(v[-1]) if len(v) and mask[-1] else None,
                low_fraction=float((v[mask] < DESIGN["low_iou"]).mean()) if mask.any() else None,
                first_good_run=prior.first_run(mask & (v >= DESIGN["good_iou"]), case["frame"]+1),
                first_low_run=prior.first_run(mask & (v < DESIGN["low_iou"]), case["frame"]+1)))
        for i, frame in enumerate(frame_ids):
            timeline.append(dict(row, video_frame=int(frame), iou=float(values[i]) if valid[i] else None,
                                 cell=int(data["cell"][k,i]), crop_sha256=str(data["crop_sha256"][k,i]),
                                 head_calls=int(data["head_calls"][k,i])))
    return first_search(data, dict(labels, search_size=ap["search_size"]), truth), windows, timeline


def paired(windows):
    for horizon in DESIGN["horizons"]:
        cases = defaultdict(dict)
        for row in windows:
            if row["horizon"] == horizon and row["full_labeled_window"]:
                cases[row["case_id"]][row["arm"]] = row
        contrasts = {
            "center_at_regression_size": (-1, 1, 0, 0),
            "size_at_regression_center": (-1, 0, 1, 0),
            "center_at_gray_size": (0, 0, -1, 1),
            "size_at_gray_center": (0, -1, 0, 1),
            "center_by_size_interaction": (1, -1, -1, 1),
            "joint_write": (-1, 0, 0, 1)}
        for label, weights in contrasts.items():
            groups = defaultdict(list)
            gains = []
            for rows in cases.values():
                gains.append(float(sum(weight * rows[arm]["successor_mean_iou"] for arm, weight in zip(ARMS, weights))))
                groups[rows[ARMS[0]]["source"]].append(gains[-1])
            means = [np.mean(values) for values in groups.values()]
            yield dict(horizon=horizon, contrast=label, complete_cases=len(cases), sources=len(groups),
                       source_mean_iou_delta=float(np.mean(means)) if means else None,
                       better_cases=sum(v > 1e-12 for v in gains), worse_cases=sum(v < -1e-12 for v in gains))


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    report = c.c66.sealed(root / "stage151_decision.json", SCHEMA)
    if (plan["design"] != DESIGN or report["status"] != STATUS or plan["source_stage150"] != SOURCE150
            or report["plan_fingerprint"] != plan["fingerprint"] or report["cases"] != DESIGN["cases"]
            or report["new_search_calls"] != DESIGN["successors_per_arm"] * len(ARMS)
            or any(report[k] != v for k, v in WORK.items())
            or len(report["cache_sha256"]) != 2 * DESIGN["cases"]
            or set(report["table_sha256"]) != {f"stage151_{k}.csv" for k in TABLES}):
        raise ValueError("Incomplete Stage-151 result")
    io.verify({str(root / k):v for k, v in (report["cache_sha256"] | report["table_sha256"]).items()})


def run(root, r150, plan, p150, contexts, observer, ap):
    first, windows, scope, timeline, hashes = [], [], [], [], {}
    (root / "branches").mkdir(exist_ok=True)
    for original in plan["cases"]:
        case = cross(original)
        cid = case["case_id"]
        relative = f"branches/{cid}.npz"
        old = c.read_cache(r150 / relative, dict(plan_fingerprint=p150["fingerprint"], case_id=cid))
        if old is None:
            raise ValueError("Missing Stage-150 branch " + cid)
        prior.audit_branch(old, original)
        entry, model = contexts[case["sequence"]]
        meta = dict(plan_fingerprint=plan["fingerprint"], case_id=cid)
        path = root / relative
        data = c.read_cache(path, meta)
        fresh = data is None
        if fresh:
            data = collect(observer, case, entry["frames"], entry["initial_box"], model, ap)
        controls = audit(data, case, old, not plan["runtime_differences"], np.size(ap["window"]))
        if fresh:
            observer.assert_frozen()
            c.save(path, data, meta)
        for item in (path, path.with_suffix(".receipt.json")):
            hashes[item.relative_to(root).as_posix()] = c.file_hash(item)
        evidence, outcomes, frames = describe(data, case, np.asarray(entry["ground_truth"], float), ap)
        first.extend(evidence)
        windows.extend(outcomes)
        timeline.extend(frames)
        scope.append(dict(case_id=cid, successors_per_arm=case["successors"],
                          new_search_calls=sum(row["head_calls"] for row in frames), **controls))
        print(f"[Stage-151] {cid}: {case['successors']} successors per arm", flush=True)
    tables = {}
    for name, rows in zip(TABLES, (first, windows, list(paired(windows)), scope, timeline)):
        path = root / f"stage151_{name}.csv"
        publish_table(path, rows)
        tables[path.name] = c.file_hash(path)
    c.publish_identical(root / "stage151_decision.json", io.seal(dict(
        schema=SCHEMA, status=STATUS, role="consumed_development_center_and_size_diagnostic",
        plan_fingerprint=plan["fingerprint"], source_stage150=SOURCE150, cases=len(scope),
        runtime=plan["runtime"], runtime_differences=plan["runtime_differences"],
        new_search_calls=sum(r["new_search_calls"] for r in scope),
        table_sha256=tables, cache_sha256=hashes, **WORK)))
    check_results(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = (*prior.SOURCES, "stage150")
    for name in names:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage151_geometry_cross")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if args.checkpoint is None or any(getattr(args, k) is None for k in names):
        parser.error("Supply all frozen source directories and the original SGLA checkpoint")
    sources = [getattr(args, k).expanduser().resolve() for k in names]
    for name, path in zip(names, sources):
        if not path.is_dir():
            parser.error(f"Missing --{name} source directory: {path}")
        if root == path or root in path.parents or path in root.parents:
            parser.error("Use a separate Stage-151 output directory")
    with writer_lock(root):
        if (root / "stage151_decision.json").exists():
            check_results(root)
            return
        r150 = sources[-1]
        prior.check_results(r150)
        p150 = c.c66.sealed(r150 / "manifest.json", prior.SCHEMA)
        d150 = c.c66.sealed(r150 / "stage150_decision.json", prior.SCHEMA)
        p121, old, ap, cases, contexts = prior.load_cases(sources[:-1])
        if (d150["fingerprint"] != SOURCE150 or p150["cases"] != cases
                or p150["source_stage121"] != p121["fingerprint"]
                or sum(case["successors"] for case in cases) != DESIGN["successors_per_arm"]):
            raise ValueError("Need the reviewed Stage-150 roster and frozen writes")
        import torch
        torch.set_num_threads(1)
        runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
        differences = base.s109.check_runtime(p150["runtime"], runtime)
        manifest = root / "manifest.json"
        if manifest.is_file() and c.c66.sealed(manifest, SCHEMA)["runtime"] != runtime:
            raise ValueError("GPU/runtime changed on resume; use a new STAGE151_ROOT")
        plan = io.seal(dict(schema=SCHEMA, design=DESIGN, source_stage150=SOURCE150, cases=cases,
                            runtime=runtime, runtime_differences=differences,
                            code_sha256={Path(p).name:c.file_hash(Path(p)) for p in (
                                __file__, prior.__file__, prior.fixed.__file__)}))
        c.publish_identical(manifest, plan)
        frozen = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        args.parameter = runtime["tracker"]["parameter"]
        args.checkpoint = args.checkpoint.expanduser().resolve()
        native, _ = base.s103.s76.startup.checked_runtime(args, frozen, root)
        observer = base.e63.Capture(native)
        try:
            run(root, r150, plan, p150, contexts, observer, ap)
        finally:
            observer.handle.remove()
    print(f"Stage-151: {STATUS}", flush=True)


if __name__ == "__main__":
    main()
