"""Expand development support for matched first writes at causal conflict states."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path

import numpy as np

import stage88_single_layout as s88
from stage27_uav_collect import writer_lock
from stage27_uav_common import IMAGE_SUFFIXES, frame_number, inside, read_boxes, read_rgb, source_group


s86, c, io, e63, e74, s76 = s88.s86, s88.c, s88.io, s88.e63, s88.e74, s88.s76
SCHEMA = "stage89_conflict_sampled_write_support_v1"
STATUS = "COMPLETE_CONFLICT_SAMPLED_WRITE_SUPPORT"
DESIGN = dict(s86.DESIGN, extra_groups=16, windows_per_stratum=3, center_separation=.50, hann_gap=.05,
              material_gain=.05, material_low_frames=4, sampling_seed=2189, bootstrap_seed=2189)
STRATA = ("conflict", "ordinary")
TABLES = ("windows", "support", "paired", "scope")


def extra_sequences(inventory, historical, holdout, allowed, design):
    members = {}
    for item in sorted(inventory, key=lambda r: r["sequence"]):
        group = item["source_group_proxy"]
        if group in allowed and group not in historical | holdout and item["frames"] > design["horizon"]:
            members.setdefault(group, item["sequence"])
    ordered = sorted(members, key=lambda g: (c.fingerprint([design["sampling_seed"], g]), g))
    if len(ordered) < design["extra_groups"]:
        raise ValueError("Insufficient development-only source groups")
    return [members[g] for g in ordered[:design["extra_groups"]]]


def select_windows(trace, key, design):
    """Rank eligible positions by a frozen outcome-independent ordering."""
    pools = {name: [] for name in STRATA}
    for i in range(len(trace["boxes"]) - design["horizon"] + 1):
        bank = {name: trace[name][i:i + 1] for name in
                ("candidate_boxes", "candidate_parents", "history_latest", "features", "selected")}
        q, signals = s88.challenger(bank)
        if q is None:
            continue
        boxes = bank["candidate_boxes"][0]
        a = int(bank["selected"][0])
        separation = float(np.linalg.norm(s86.centers(boxes[a]) - s86.centers(boxes[q])) /
                           np.sqrt(bank["history_latest"][0, 2:].prod()))
        if separation >= design["center_separation"]:
            if abs(signals["hann"]) > design["hann_gap"]:
                continue
            stratum = "conflict"
        else:
            stratum = "ordinary"
        pools[stratum].append(dict(first=i + 1, stratum=stratum, action=q, signals=signals, separation=separation))

    picked = []
    for stratum, options in pools.items():
        order = sorted(options, key=lambda item: c.fingerprint([design["sampling_seed"], key, item["first"]]))
        count = 0
        for item in order:
            if all(abs(item["first"] - other["first"]) >= design["horizon"] for other in picked):
                picked.append(item)
                count += 1
                if count == design["windows_per_stratum"]:
                    break
    return picked, {name: len(items) for name, items in pools.items()}


def expanded_entry(root, plan, dataset, sequence):
    path = root / "entries" / (sequence + ".json")
    if path.exists():
        cached = c.c66.sealed(path, SCHEMA)
        if cached["plan_fingerprint"] != plan["fingerprint"] or cached["entry"]["sequence"] != sequence:
            raise ValueError("Expanded entry differs from the frozen roster")
        return cached["entry"], path
    folder = inside(dataset, "sequences/" + sequence)
    images = sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=frame_number)
    annotation = inside(dataset, "annotations/" + sequence + ".txt")
    annotation_hash = c.file_hash(annotation)
    gt = read_boxes(annotation)
    if len(images) != len(gt) or len(images) <= plan["design"]["horizon"] or c.file_hash(annotation) != annotation_hash:
        raise ValueError("Invalid expanded sequence annotation")
    frames = []
    for image in images:
        image = inside(dataset, image.relative_to(dataset))
        digest = c.file_hash(image)
        height, width = read_rgb(image).shape[:2]
        if c.file_hash(image) != digest:
            raise ValueError("Image changed during expanded inventory")
        frames.append(dict(file=image.relative_to(dataset).as_posix(), sha256=digest, height=height, width=width))
    entry = dict(sequence=sequence, frames=frames, initial_box=gt[0].tolist(), ground_truth=gt.tolist(),
                 annotation=annotation.relative_to(dataset).as_posix(), annotation_sha256=annotation_hash)
    c.publish_identical(path, io.seal(dict(schema=SCHEMA, plan_fingerprint=plan["fingerprint"], entry=entry)))
    return entry, path


def baseline(root, plan, source, actor, observer, unit, entry):
    path = root / "baseline" / (c.key(unit) + ".npz")
    meta = dict(plan_fingerprint=plan["fingerprint"], unit=unit)
    old = source["original"]
    ap = e74.e67.actor_plan(old)
    pack = e74.learned_pack(actor["models"], unit, actor["prior"])

    def choose(row, history):
        return e74.choose(row, history, unit, old, actor["prior"], pack)

    prefix = np.asarray([entry["initial_box"]], float)
    frames = np.arange(1, len(entry["frames"]))
    trace = c.read_cache(path, meta)
    fresh = trace is None
    if fresh:
        with e63.source_experiment.installed(observer, [], unit) as head:
            if head != source["report"]["native_head_sha256"]:
                raise ValueError("Original SGLA head changed")
            trace = s86.collect(observer, entry, prefix, frames, 0., ap, choose)
    s86.audit(trace, prefix, frames, 0., ap, choose)
    if fresh:
        c.save(path, trace, meta)
    return trace, path


def window_item(window, picked, entry, trace, history_length, horizon):
    first = picked["first"]
    i = first - 1
    frames = np.arange(first, first + horizon)
    previous = np.vstack([entry["initial_box"], trace["boxes"][max(0, i - history_length):i]])[-history_length:]
    fixed = {k: trace[k][i:i + horizon] for k in s86.FIELDS}
    np.testing.assert_array_equal(previous[-1], fixed["history_latest"][0])
    return dict(window=window, entry=entry, frames=frames, prefix=previous, arms=[fixed],
                action=picked["action"], signals=picked["signals"], box=fixed["candidate_boxes"][0, picked["action"]], ref="new")


def aggregate(rows, scope, design):
    support, paired = [], []
    low_limit = design["material_low_frames"]
    threshold = design["material_gain"]
    for cohort in ("all", "expanded", "historical"):
        for stratum in STRATA:
            group = [r for r in rows if r["stratum"] == stratum and (cohort == "all" or r["cohort"] == cohort)]
            units = [r for r in scope if cohort == "all" or r["cohort"] == cohort]
            groups = {r["sequence"] for r in group}
            categories = {
                "followup_gain": [r for r in group if r["followup_gain"] >= threshold],
                "followup_harm": [r for r in group if r["followup_gain"] <= -threshold],
                "low_iou_gain": [r for r in group if r["low_iou_delta_frames"] <= -low_limit],
                "low_iou_harm": [r for r in group if r["low_iou_delta_frames"] >= low_limit],
                "ppo_only_recovery": [r for r in group if r["anchor_iou"] < design["low_iou"] and r["ppo_return"] and not r["challenger_return"]],
                "challenger_only_recovery": [r for r in group if r["anchor_iou"] < design["low_iou"] and r["challenger_return"] and not r["ppo_return"]],
            }
            support.append(dict(cohort=cohort, stratum=stratum, windows=len(group), source_groups=len(groups),
                roster_source_groups=len({r["sequence"] for r in units}), eligible_frames=sum(r["eligible_" + stratum] for r in units),
                initially_low=sum(r["anchor_iou"] < design["low_iou"] for r in group),
                **{name + "_windows": len(items) for name, items in categories.items()},
                **{name + "_groups": len({r["sequence"] for r in items}) for name, items in categories.items()}))
            if group:
                videos = defaultdict(list)
                for row in group:
                    videos[row["sequence"]].append(row)
                for metric, field in (("followup_mean_iou", "followup_gain"), ("low_iou_fraction", "low_iou_delta_frames")):
                    scale = 1 if field == "followup_gain" else design["horizon"]
                    values = [np.mean([r[field] for r in videos[seq]]) / scale for seq in sorted(videos)]
                    paired.append(dict(s86.contrast(values, "challenger_minus_ppo", metric, design),
                                       cohort=cohort, stratum=stratum, windows=len(group)))
        by_video = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if cohort == "all" or r["cohort"] == cohort:
                by_video[r["sequence"]][r["stratum"]].append(r)
        matched = [groups for _, groups in sorted(by_video.items()) if all(s in groups for s in STRATA)]
        for metric, field, limit in (("material_iou_gap_rate", "followup_gain", threshold),
                                     ("material_low_iou_gap_rate", "low_iou_delta_frames", low_limit)):
            values = [np.mean([abs(r[field]) >= limit for r in g["conflict"]]) -
                      np.mean([abs(r[field]) >= limit for r in g["ordinary"]]) for g in matched]
            if values:
                paired.append(dict(s86.contrast(values, "conflict_minus_ordinary", metric, design),
                                   cohort=cohort, stratum="matched", windows=sum(len(g[s]) for g in matched for s in STRATA)))
    return support, paired


def execute(root, plan, source, actor, observer):
    design = plan["design"]
    old = source["original"]
    dataset = Path(plan["dataset"])
    evidence, rows, scope = {}, [], []
    for record in plan["units"]:
        unit, cohort = record["unit"], record["cohort"]
        sequence, key = unit["sequence"], c.key(unit)
        if cohort == "expanded":
            entry, path = expanded_entry(root, plan, dataset, sequence)
            annotation = inside(dataset, entry["annotation"])
            if c.file_hash(annotation) != entry["annotation_sha256"]:
                raise ValueError("Expanded annotation changed")
            evidence[path.relative_to(root).as_posix()] = c.file_hash(path)
            trace, path = baseline(root, plan, source, actor, observer, unit, entry)
        else:
            entry, trace = s86.previous.read_source_trace(source, dict(unit=unit, frames=[1], expected_first_failure=None))
            path = None
            if not plan["reuse_historical"]:
                trace, path = baseline(root, plan, source, actor, observer, unit, entry)
        baseline_calls = 0 if path is None else len(trace["boxes"]) * 4
        if path is not None:
            evidence[path.relative_to(root).as_posix()] = c.file_hash(path)
        windows, eligible = select_windows(trace, key, design)
        for picked in windows:
            first, stratum = picked["first"], picked["stratum"]
            case = f"{key}__f{first:06d}"
            w = dict(case=case, unit=unit, stratum=stratum, anchor=first - 1)
            item = window_item(w, picked, entry, trace, old["actor_design"]["history"], design["horizon"])
            tail, path = s88.resolve_tail(item, None, None, root, plan, source, actor, observer)
            evidence[path.relative_to(root).as_posix()] = c.file_hash(path)
            row = s88.record(item, tail, design)
            rows.append(dict(row, cohort=cohort, first=first, center_separation=picked["separation"],
                             low_iou_delta_frames=round(design["horizon"] * (row["challenger_low_iou"] - row["ppo_low_iou"]))))
        counts = Counter(w["stratum"] for w in windows)
        scope.append(dict(sequence=sequence, seed=unit["seed"], fold=unit["fold"], cohort=cohort,
                          **{"eligible_" + k: eligible[k] for k in STRATA}, **{"sampled_" + k: counts[k] for k in STRATA},
                          baseline_search_calls=baseline_calls, diagnostic_search_calls=len(windows) * design["horizon"] * 4))
        print(f"[Stage-89] {key} {cohort} conflict={counts['conflict']} ordinary={counts['ordinary']}", flush=True)

    support, paired = aggregate(rows, scope, design)
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])
    analysis = root / "analysis"
    analysis.mkdir(exist_ok=True)
    hashes = {}
    for name, table in zip(TABLES, (rows, support, paired, scope)):
        path = analysis / f"stage89_{name}.csv"
        s86.publish_table(path, table)
        hashes[path.name] = c.file_hash(path)
    baseline_calls = sum(r["baseline_search_calls"] for r in scope)
    diagnostic_calls = sum(r["diagnostic_search_calls"] for r in scope)
    c.publish_identical(analysis / "stage89_decision.json", io.seal(dict(schema=SCHEMA, status=STATUS,
        plan_fingerprint=plan["fingerprint"], runtime=plan["runtime"], windows=len(rows), histories=len(scope),
        baseline_search_calls=baseline_calls, diagnostic_search_calls=diagnostic_calls,
        search_calls=baseline_calls + diagnostic_calls, block_calls=7 * (baseline_calls + diagnostic_calls),
        training_updates=0, confirmation_inference_calls=0, table_sha256=hashes, evidence_sha256=evidence)))
    print(f"Stage-89: {STATUS} ({len(rows)} windows, {baseline_calls + diagnostic_calls} new searches)", flush=True)


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    report = c.c66.sealed(root / "analysis/stage89_decision.json", SCHEMA)
    if (report["status"] != STATUS or report["plan_fingerprint"] != plan["fingerprint"]
            or report["histories"] != len(plan["units"]) or report["runtime"] != plan["runtime"]
            or report["diagnostic_search_calls"] != report["windows"] * plan["design"]["horizon"] * 4
            or report["search_calls"] != report["baseline_search_calls"] + report["diagnostic_search_calls"]
            or report["block_calls"] != report["search_calls"] * 7
            or set(report["table_sha256"]) != {f"stage89_{n}.csv" for n in TABLES}):
        raise ValueError("Incomplete Stage-89 report")
    io.verify({str(inside(root, k)): v for k, v in report["evidence_sha256"].items()})
    io.verify({str(root / "analysis" / k): v for k, v in report["table_sha256"].items()})


def make_plan(source, actor, runtime):
    roster = c.c66.sealed(s86.ROSTER, s86.confirmation.SCHEMA)
    dataset = Path(source["original"]["root"]).resolve()
    inventory = s86.confirmation.inventory(dataset)
    if c.fingerprint(inventory) != roster["inventory_fingerprint"]:
        raise ValueError("VisDrone train inventory differs from the frozen confirmation roster")
    held = {r["source_group_proxy"] for r in roster["sequences"]}
    old_units = [u for u in source["plan"]["units"] if u["policy"] == "ppo"]
    historical = {source_group(u["sequence"]) for u in old_units}
    if len(historical) != 12 or len(old_units) != 24 or len(held) != 28 or historical & held:
        raise ValueError("Unexpected historical/confirmation roster")
    allowed = set().union(*map(set, roster["excluded_groups_by_source"].values()))
    expanded = extra_sequences(inventory, historical, held, allowed, DESIGN)
    units = [dict(unit=u, cohort="historical") for u in old_units]
    units += [dict(unit=dict(sequence=seq, origin="native", policy="ppo", fold=i % 2, seed=2171 + (i // 2) % 2),
                   cohort="expanded") for i, seq in enumerate(expanded)]
    code = [Path(__file__), Path(s88.__file__), Path(s86.__file__), Path(e74.__file__),
            Path(__file__).with_name("run_stage89_conflict_sampling.sh")]
    return io.seal(dict(schema=SCHEMA, design=DESIGN, dataset=str(dataset), units=units, runtime=runtime,
        reuse_historical=runtime["tracker"]["device"] == source["plan"]["runtime"]["tracker"]["device"],
        confirmation_fingerprint=roster["fingerprint"], source_stage76_fingerprint=source["report"]["fingerprint"],
        source_sha256=source["evidence"] | actor["bindings"] | {str(s86.ROSTER): c.file_hash(s86.ROSTER)},
        code_sha256={str(p.resolve()): c.file_hash(p) for p in code}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage76", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage89")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not args.stage76 or not args.checkpoint:
        parser.error("Supply Stage-76 and the original SGLA checkpoint")
    upstream = args.stage76.expanduser().resolve()
    if root == upstream or root in upstream.parents or upstream in root.parents:
        parser.error("Use a separate Stage-89 directory")
    import torch

    torch.set_num_threads(1)
    with writer_lock(root):
        if (root / "analysis/stage89_decision.json").exists():
            check_results(root)
            return
        if not (root / "manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-89 directory")
        source = s86.previous.load_source(upstream)
        actor = io.load_source(Path(source["plan"]["source_input"]))
        if actor["plan"] != source["original"] or actor["models"]["fingerprint"] != source["report"]["source_training_fingerprint"]:
            raise ValueError("Stage-74 actor changed")
        runtime = s76.device_runtime(source["original"], torch, args.gpu)
        plan = make_plan(source, actor, runtime)
        c.publish_identical(root / "manifest.json", plan)
        frozen = dict(source["original"], fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = s76.startup.checked_runtime(args, frozen, root)
        observer = e63.Capture(native)
        try:
            execute(root, plan, source, actor, observer)
        finally:
            observer.handle.remove()


if __name__ == "__main__":
    main()
