"""Cross PPO/gated search centers and sizes at the frozen Stage-95 anchors."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import os
from pathlib import Path

import numpy as np

import stage95_response_diagnostic as s95
from stage27_uav_common import checked_image, crop_geometry, tracking_iou


c, io, e63, s76, s86 = s95.c, s95.io, s95.e63, s95.s76, s95.s86
SCHEMA = "stage96_search_center_size_cross_v1"
STATUS = "COMPLETE_SEARCH_CENTER_SIZE_CROSS_DIAGNOSTIC"
SOURCE95 = "62033ca759453fe0b40e7651ee17c418313bc4b4a87ee26f104ca052e14197dc"
CELLS = ("PP", "PG", "GP", "GG")
FIELDS = ("frame_id", "candidate_parents", "candidate_boxes", "raw", "size", "offset",
          "image_hw", "crop", "crop_sha256", "layer", "block_calls", "head_calls", "selector_calls", "decode_error")
METRICS = ("best_public_iou", "best_dense_iou", "maximum_gt_coverage")
TABLES = ("events", "states", "crops", "factors", "summary", "factor_summary")


def crossed_parents(ppo, gated, cell):
    """Each slot keeps its role. The first letter provides center, the second size."""
    if cell not in CELLS:
        raise ValueError("Unknown center/size cell")
    ppo, gated = np.asarray(ppo, float), np.asarray(gated, float)
    if any(p.shape != (4, 4) or not np.isfinite(p).all() or np.any(p[:, 2:] <= 0) for p in (ppo, gated)):
        raise ValueError("Need two valid four-slot parent banks")
    if cell == "PP":
        return ppo.copy()
    if cell == "GG":
        return gated.copy()
    center_bank = ppo if cell[0] == "P" else gated
    size = (ppo if cell[1] == "P" else gated)[:, 2:]
    center = center_bank[:, :2] + center_bank[:, 2:] / 2
    return np.concatenate((center - size / 2, size), axis=1)


def collect(observer, entry, parents, frame, old, image_reader=checked_image):
    ap = e63.c.actor_plan(old)
    observer.initialize(image_reader(ap["root"], entry["frames"][0]), entry["initial_box"])
    image = image_reader(ap["root"], entry["frames"][frame])
    views = [e63.actor.observe(observer, image, p.copy(), frame, ap) for p in parents]
    row = dict(frame_id=frame, candidate_parents=parents.copy(),
               candidate_boxes=np.asarray([v["boxes"] for v in views]), image_hw=np.asarray(image.shape[:2]),
               decode_error=max(v["decode_error"] for v in views),
               **{k: np.asarray([v[k] for v in views]) for k in
                  ("raw", "size", "offset", "crop", "crop_sha256", "layer", "block_calls", "head_calls", "selector_calls")})
    return {k: np.asarray([row[k]]) for k in FIELDS}


def audit(data, parents, frame, old):
    if set(data) != set(FIELDS) or data["candidate_boxes"].shape != (1, 4, 4):
        raise ValueError("Incomplete crossed observation")
    np.testing.assert_array_equal(data["frame_id"], [frame])
    np.testing.assert_array_equal(data["candidate_parents"], parents[None])
    e63.source_common.prior.audit_calls(data, (1, 4))
    for key in ("candidate_boxes", "raw", "size", "offset", "image_hw", "decode_error"):
        if not np.isfinite(data[key]).all():
            raise ValueError(f"Nonfinite crossed observation: {key}")
    errors = []
    for q, parent in enumerate(parents):
        crop, _, factor = crop_geometry(parent, data["image_hw"][0], old["search_factor"], old["search_size"])
        np.testing.assert_array_equal(data["crop"][0, q], crop)
        raw, size, offset = (data[k][0, q] for k in ("raw", "size", "offset"))
        dense = s95.dense_boxes(size, offset, parent, factor, old["search_size"], data["image_hw"][0])
        cell = int(np.argmax(raw * np.asarray(old["window"], np.float32)))
        errors.append(float(np.abs(dense[cell] - data["candidate_boxes"][0, q]).max()))
    np.testing.assert_allclose(data["decode_error"], [max(errors)], rtol=0, atol=1e-12)
    if max(errors) > old["actor_design"]["box_atol"]:
        raise ValueError("Crossed CENTER decode differs from native output")


def factorial(states):
    grouped = defaultdict(dict)
    for row in states:
        key = row["sequence"], row["seed"], row["frame"]
        if row["cell"] in grouped[key]:
            raise ValueError("Duplicate factorial cell")
        grouped[key][row["cell"]] = row
    rows = []
    for cells in grouped.values():
        if set(cells) != set(CELLS):
            raise ValueError("Incomplete geometry factorial")
        ref = cells["GG"]
        row = {k: ref[k] for k in ("sequence", "seed", "frame", "cohort", "outcome", "roles", "gt_valid")}
        for metric in METRICS:
            pp, pg, gp, gg = (cells[cell][metric] for cell in CELLS)
            for name, value in (
                ("center_at_gated_size", pg-gg), ("size_at_gated_center", gp-gg),
                ("center_average", ((pp-gp)+(pg-gg))/2), ("size_average", ((pp-pg)+(gp-gg))/2),
                ("interaction", pp-pg-gp+gg), ("ppo_minus_gated", pp-gg)):
                row[f"{metric}_{name}"] = value if ref["gt_valid"] else None
        rows.append(row)
    return rows


def summarize(events, states, factors):
    lookup = {(r["sequence"], r["seed"], r["frame"], r["cell"]): r for r in states}
    focus = []
    for event in events:
        row = dict(event)
        for cell in CELLS:
            state = lookup[event["sequence"], int(event["seed"]), int(event["ppo_recovery_frame"]), cell]
            for key in ("best_public_iou", "best_dense_iou", "maximum_gt_coverage", "weighted_peak_support",
                        "raw_peak_support", "gated_slot_iou", "ppo_slot_iou", "category_at_gated_slot"):
                row[f"{cell}_{key}"] = state[key]
        focus.append(row)
    groups, summary = defaultdict(list), []
    for row in states:
        groups[row["outcome"], row["cell"]].append(row)
    for (outcome, cell), rows in sorted(groups.items()):
        counts = Counter(r["category_at_gated_slot"] for r in rows)
        summary.extend(dict(outcome=outcome, cell=cell, category=k, states=v, all_anchor_states=len(rows))
                       for k, v in sorted(counts.items()))
    grouped_factors = defaultdict(list)
    for row in factors:
        grouped_factors[row["outcome"]].append(row)
    factor_summary = []
    for outcome, rows in sorted(grouped_factors.items()):
        for name in factors[0]:
            if any(name.startswith(metric + "_") for metric in METRICS):
                values = [r[name] for r in rows if r[name] is not None]
                factor_summary.append(dict(outcome=outcome, contrast=name, valid_anchor_states=len(values),
                                           mean_delta=float(np.mean(values)) if values else None))
    return focus, summary, factor_summary


def observe_cell(root, plan, observer, entry, unit, point, cell, reference, parents, old):
    path = root / "collection" / f"{c.key(unit)}__f{point['frame']:06d}__{cell}.npz"
    meta = dict(plan_fingerprint=plan["fingerprint"], unit=unit, frame=point["frame"], cell=cell)
    data = c.read_cache(path, meta)
    fresh = data is None
    reused = cell in ("PP", "GG") and plan["reuse_diagonals"]
    if fresh:
        data = {k: reference[k].copy() for k in FIELDS} if reused else collect(observer, entry, parents, point["frame"], old)
    audit(data, parents, point["frame"], old)
    if fresh:
        c.save(path, data, meta)
    return data, path


def describe(point, cell, data, baseline, entry, old, selected):
    gt = np.asarray(entry["ground_truth"][point["frame"]], float)
    gslot, pslot = selected["gated"], selected["ppo"]
    crops = s95.inspect(data, old, gt, gslot)
    iou = tracking_iou(data["candidate_boxes"][0], gt)
    diagonal = cell in ("PP", "GG")
    original = tracking_iou(baseline["candidate_boxes"][0], gt)
    ident = {k: point[k] for k in ("sequence", "seed", "frame", "cohort", "outcome")}
    ident.update(cell=cell, roles="|".join(point["roles"]))
    state = dict(ident, gt_valid=crops[0]["gt_valid"], category_at_gated_slot=s95.classify(crops, gslot),
        gated_slot=gslot, ppo_slot=pslot, gated_slot_iou=float(iou[gslot]), ppo_slot_iou=float(iou[pslot]),
        best_public_iou=float(iou.max()), best_dense_iou=max(r.get("dense_best_gt_iou", -1.) for r in crops),
        maximum_gt_coverage=max(r.get("gt_coverage", -1.) for r in crops),
        weighted_peak_support=any(r.get("weighted_top5_gt_recall", False) for r in crops),
        raw_peak_support=any(r.get("raw_top5_gt_recall", False) for r in crops),
        diagonal_box_error=float(np.abs(data["candidate_boxes"]-baseline["candidate_boxes"]).max()) if diagonal else None,
        diagonal_support_changed=bool(np.any((original >= .5) != (iou >= .5))) if diagonal else None)
    return state, [dict(ident, **r) for r in crops]


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    report = c.c66.sealed(root / "analysis/stage96_decision.json", SCHEMA)
    expected_new = (8 if plan["reuse_diagonals"] else 16) * len(plan["points"])
    if (report["status"] != STATUS or report["plan_fingerprint"] != plan["fingerprint"]
            or report["states"] != 4 * len(plan["points"]) or report["new_search_calls"] != expected_new
            or report["logical_search_calls"] != 4 * report["states"]
            or report["new_block_calls"] != 7 * expected_new or len(report["cache_sha256"]) != report["states"]
            or set(report["table_sha256"]) != {f"stage96_{n}.csv" for n in TABLES}):
        raise ValueError("Incomplete Stage-96 report")
    for folder, bindings in (("", report["cache_sha256"]), ("analysis", report["table_sha256"])):
        io.verify({str(root / folder / name): sha for name, sha in bindings.items()})
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])


def run(root, root95, plan, plan95, source, old, observer):
    states, crops, evidence = [], [], {}
    units = {(r["unit"]["sequence"], r["unit"]["seed"]): r["unit"] for r in plan["units"]}
    with (root95 / "analysis/stage95_states.csv").open(encoding="utf-8", newline="") as handle:
        slots = {(r["sequence"], int(r["seed"]), int(r["frame"]), r["history"]): int(r["source_selected"])
                 for r in csv.DictReader(handle)}
    for point in plan["points"]:
        unit = units[point["sequence"], point["seed"]]
        if point["cohort"] == "historical":
            entry = c.entry_for(old, point["sequence"])
        else:
            bound = c.c66.sealed(Path(plan["root89"]) / "entries" / (point["sequence"]+".json"), s95.s94.s89.SCHEMA)
            if bound["plan_fingerprint"] != plan["plan89_fingerprint"] or bound["entry"]["sequence"] != point["sequence"]:
                raise ValueError("Expanded entry changed")
            entry = bound["entry"]
        originals = {}
        for history in ("ppo", "gated"):
            path = root95 / "collection" / f"{c.key(unit)}__{history}__f{point['frame']:06d}.npz"
            originals[history] = s95.s94.required_cache(path, dict(plan_fingerprint=plan95["fingerprint"], unit=unit,
                                                                 history=history, frame=point["frame"]))
        ppo, gated = originals["ppo"], originals["gated"]
        for field in ("image_hw", "frame_id"):
            np.testing.assert_array_equal(ppo[field], gated[field])
        selected = {h: slots[point["sequence"], point["seed"], point["frame"], h] for h in originals}
        with e63.source_experiment.installed(observer, [], unit) as head:
            if head != source["report"]["native_head_sha256"]:
                raise ValueError("Native SGLA head changed")
            for cell in CELLS:
                parents = crossed_parents(ppo["candidate_parents"][0], gated["candidate_parents"][0], cell)
                reference = originals["ppo" if cell == "PP" else "gated"]
                data, path = observe_cell(root, plan, observer, entry, unit, point, cell, reference, parents, old)
                if cell in ("PP", "GG"):
                    for field in ("crop", "crop_sha256", "image_hw"):
                        np.testing.assert_array_equal(data[field], reference[field])
                state, views = describe(point, cell, data, reference, entry, old, selected)
                states.append(state)
                crops.extend(views)
                evidence[path.relative_to(root).as_posix()] = c.file_hash(path)
        print(f"[Stage-96] {point['sequence']} seed={point['seed']} frame={point['frame']}", flush=True)
    factors = factorial(states)
    focus, summary, factor_summary = summarize(plan95["events"], states, factors)
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    hashes = {}
    for name, rows in zip(TABLES, (focus, states, crops, factors, summary, factor_summary)):
        path = out / f"stage96_{name}.csv"
        s86.publish_table(path, rows)
        hashes[path.name] = c.file_hash(path)
    diagonal = [r for r in states if r["cell"] in ("PP", "GG")]
    calls = (8 if plan["reuse_diagonals"] else 16) * len(plan["points"])
    c.publish_identical(out / "stage96_decision.json", io.seal(dict(
        schema=SCHEMA, status=STATUS, plan_fingerprint=plan["fingerprint"], source_stage95_fingerprint=SOURCE95,
        runtime=plan["runtime"], runtime_differences=plan["runtime_differences"],
        reuse_diagonals=plan["reuse_diagonals"], states=len(states), logical_search_calls=4*len(states),
        new_search_calls=calls, new_block_calls=7*calls, maximum_diagonal_box_error_px=max(r["diagonal_box_error"] for r in diagonal),
        diagonal_support_changed_states=sum(r["diagonal_support_changed"] for r in diagonal),
        source_correspondence_pass=all(r["diagonal_box_error"] <= old["actor_design"]["box_atol"]
            and not r["diagonal_support_changed"] for r in diagonal),
        training_updates=0, new_full_trajectories=0, confirmation_inference_calls=0,
        cache_sha256=evidence, table_sha256=hashes)))
    check_results(root)
    print(f"Stage-96: {STATUS} ({len(states)} cells, {calls} new searches)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage76", "stage89", "stage93", "stage95", "output_dir"):
        parser.add_argument(f"--{name}", type=Path, required=name == "output_dir")
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage96")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not all((args.stage76, args.stage89, args.stage93, args.stage95, args.checkpoint)):
        parser.error("Supply Stage-76, 89, 93, 95 and the original SGLA checkpoint")
    root93, root76, root89, root95 = [p.expanduser().resolve() for p in
                                     (args.stage93, args.stage76, args.stage89, args.stage95)]
    if any(root == p or root in p.parents or p in root.parents for p in (root93, root76, root89, root95)):
        parser.error("Use a separate Stage-96 output directory")
    with s95.s93.writer_lock(root):
        if (root / "analysis/stage96_decision.json").exists():
            check_results(root)
            return
        if not (root / "manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-96 output directory")
        s95.check_results(root95)
        plan95 = c.c66.sealed(root95 / "manifest.json", s95.SCHEMA)
        report95 = c.c66.sealed(root95 / "analysis/stage95_decision.json", s95.SCHEMA)
        if report95["fingerprint"] != SOURCE95 or plan95["points"] != s95.points_for(plan95["events"]):
            raise ValueError("Need the frozen Stage-95 anchors and results")
        _, _, plan89, _, source = s95.s94.load_run(root93, root76, root89)
        import torch

        torch.set_num_threads(1)
        old = source["original"]
        runtime = s76.device_runtime(old, torch, args.gpu)
        differences = s76.startup.differences(report95["runtime"], runtime)
        if any(r["field"] not in ("runtime.gpu_uuid", "runtime.tracker.device") for r in differences):
            raise ValueError("Source software or weights changed")
        paths = [root95 / "manifest.json", root95 / "analysis/stage95_decision.json"]
        paths += [root95 / "analysis" / name for name in report95["table_sha256"]]
        code = [Path(__file__), Path(__file__).with_name("run_stage96_geometry_cross.sh"), Path(s95.__file__),
                Path(e63.actor.__file__), Path(e63.source_common.prior.__file__), Path(c.__file__), Path(io.__file__)]
        plan = io.seal(dict(schema=SCHEMA, source_stage95_fingerprint=SOURCE95, runtime=runtime,
            runtime_differences=differences, reuse_diagonals=not differences,
            points=plan95["points"], units=plan89["units"], root89=str(root89), plan89_fingerprint=plan89["fingerprint"],
            source_sha256={str(p): c.file_hash(p) for p in paths} | plan95["source_sha256"]
                | {str(root95 / p): h for p, h in report95["cache_sha256"].items()},
            code_sha256={str(p.resolve()): c.file_hash(p) for p in code} | plan95["code_sha256"]))
        c.publish_identical(root / "manifest.json", plan)
        frozen = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = s76.startup.checked_runtime(args, frozen, root)
        observer = e63.Capture(native)
        try:
            run(root, root95, plan, plan95, source, old, observer)
        finally:
            observer.handle.remove()


if __name__ == "__main__":
    main()
