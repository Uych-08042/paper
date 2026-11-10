"""Cross Stage-86 search geometry with the two observed first-action slots."""
from __future__ import annotations

import argparse
from collections import defaultdict
import os
from pathlib import Path

import numpy as np

import stage86_search_center as s86
from stage27_uav_collect import writer_lock


c, io, e63, e74, s76 = s86.c, s86.io, s86.e63, s86.e74, s86.s76
SCHEMA = "stage87_search_geometry_action_cross_v1"
STATUS = "COMPLETE_SEARCH_GEOMETRY_ACTION_CROSS"
CELLS = ("ff", "fm", "mf", "mm")  # layout first, action source second
EFFECTS = {
    "geometry_given_fixed_action": (-1, 0, 1, 0),
    "action_given_midpoint_layout": (0, 0, -1, 1),
    "action_given_fixed_layout": (-1, 1, 0, 0),
    "geometry_given_midpoint_action": (0, -1, 0, 1),
    "interaction": (1, -1, -1, 1),
    "natural_total": (-1, 0, 0, 1),
}
METRICS = ("first_iou", "mean_iou", "followup_mean_iou", "low_iou_fraction", "sustained_low_iou",
           "unresolved_at_window_end", "reaches_sustained_good")
TABLES = ("first", "branches", "paired")


def cross(data):
    """Identify unique continuation states by exact published-box equality."""
    fixed, midpoint = data
    qf, qm = (int(a["selected"][0]) for a in data)
    for arm in data:
        if not np.array_equal(arm["boxes"][0], arm["candidate_boxes"][0, arm["selected"][0]]):
            raise ValueError("Natural first box does not match the recorded action")
    seeds = {"ff": fixed["boxes"][0], "mm": midpoint["boxes"][0]}
    if np.array_equal(seeds["ff"], seeds["mm"]):
        for k in ("boxes", "candidate_boxes", "selected", "crop_sha256"):
            np.testing.assert_array_equal(fixed[k][1:], midpoint[k][1:])
    cells = {}
    for name, bank, action in zip(CELLS, (fixed, fixed, midpoint, midpoint), (qf, qm, qf, qm)):
        box = bank["candidate_boxes"][0, action]
        ref = name if name in ("ff", "mm") else next((r for r, b in seeds.items() if np.array_equal(b, box)), name)
        seeds.setdefault(ref, box)
        cells[name] = dict(action=action, ref=ref, box=box)
    return cells


def open_input(root86, root76):
    s86.check_results(root86)
    manifest = c.c66.sealed(root86 / "manifest.json", s86.SCHEMA)
    report = c.c66.sealed(root86 / "analysis/stage86_decision.json", s86.SCHEMA)
    source = s86.previous.load_source(root76)
    if (manifest["design"] != s86.DESIGN or manifest["source_stage76_fingerprint"] != source["report"]["fingerprint"]
            or report["source_stage76_fingerprint"] != source["report"]["fingerprint"]
            or len(manifest["windows"]) != 144 or len({w["unit"]["sequence"] for w in manifest["windows"]}) != 12):
        raise ValueError("Need the complete Stage-86 development cohort and matching Stage-76 source")
    io.verify(manifest["code_sha256"])
    return manifest, report, source


def cases(root86, manifest, source):
    old = source["original"]
    for windows in s86.group_windows(manifest["windows"]).values():
        unit = windows[0]["unit"]
        entry, trace = s86.previous.read_source_trace(source, dict(unit=unit, frames=[1], expected_first_failure=None))
        history = np.vstack([entry["initial_box"], trace["boxes"]])
        for w in windows:
            first = w["anchor"] + 1
            frames = np.arange(first, first + manifest["design"]["horizon"])
            prefix = history[max(0, first - old["actor_design"]["history"]):first]
            arms = []
            for arm in s86.ARMS:
                path = root86 / "collection" / (w["case"] + "__" + arm + ".npz")
                data = c.read_cache(path, dict(plan_fingerprint=manifest["fingerprint"], case=w["case"], arm=arm))
                if data is None or set(data) != set(s86.FIELDS):
                    raise ValueError("Missing Stage-86 branch")
                np.testing.assert_array_equal(data["frame_id"], frames)
                np.testing.assert_array_equal(data["history_latest"][0], prefix[-1])
                arms.append(data)
            np.testing.assert_array_equal(arms[0]["candidate_boxes"][0, 2:], arms[1]["candidate_boxes"][0, 2:])
            yield w, entry, frames, prefix, arms, cross(arms)


def ids(w):
    return dict(case=w["case"], sequence=w["unit"]["sequence"], seed=w["unit"]["seed"],
                stratum=w["stratum"], anchor=w["anchor"])


def prepare(root, root86, manifest, report, source):
    first_rows, new = [], []
    for w, entry, frames, prefix, arms, cells in cases(root86, manifest, source):
        truth = np.asarray(entry["ground_truth"], float)
        first = int(frames[0])
        for name, cell in cells.items():
            bank = arms[name[0] == "m"]["candidate_boxes"][0]
            first_rows.append(dict(ids(w), cell=name, action=cell["action"], continuation=cell["ref"],
                                   anchor_iou=float(s86.tracking_iou(prefix[-1], truth[first - 1])),
                                   first_iou=float(s86.tracking_iou(cell["box"], truth[first])),
                                   best_iou=float(s86.tracking_iou(bank, truth[first]).max())))
            if name == cell["ref"] and name not in ("ff", "mm"):
                new.append(w["case"] + "__" + name)
    code = [Path(__file__), Path(s86.__file__), Path(e74.__file__), Path(e63.__file__),
            Path(e63.actor.__file__), Path(__file__).with_name("run_stage87_action_cross.sh")]
    plan = io.seal(dict(schema=SCHEMA, source_stage86_fingerprint=report["fingerprint"],
        source_stage86_plan_fingerprint=manifest["fingerprint"], source_stage76_fingerprint=source["report"]["fingerprint"],
        source_runtime=report["runtime"], design=dict(manifest["design"], bootstrap_seed=2187),
        windows=len(manifest["windows"]), new_branches=new, bridge_windows=len({name.rsplit("__", 1)[0] for name in new}),
        source_files={str(path): c.file_hash(path) for path in
                      (root86 / "manifest.json", root86 / "analysis/stage86_decision.json")},
        code_sha256={str(path.resolve()): c.file_hash(path) for path in code}))
    c.publish_identical(root / "manifest.json", plan)
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    s86.publish_table(out / "stage87_first.csv", first_rows)
    # Keep an immutable receipt even when only cached first-frame analysis is requested.
    c.publish_identical(out / "stage87_first_receipt.json", io.seal(dict(schema=SCHEMA,
        plan_fingerprint=plan["fingerprint"], rows=len(first_rows), sha256=c.file_hash(out / "stage87_first.csv"))))
    return plan


def branch_metrics(bank, continuation, cell, truth, design):
    data = dict(boxes=np.vstack([cell["box"], continuation["boxes"]]),
                selected=np.r_[cell["action"], continuation["selected"]],
                candidate_boxes=np.concatenate([bank["candidate_boxes"][:1], continuation["candidate_boxes"]]))
    stats = s86.metrics(data, truth, design)
    stats["followup_mean_iou"] = float(s86.tracking_iou(data["boxes"][1:], truth[1:]).mean())
    return stats


def summarize(rows, design):
    groups = defaultdict(dict)
    for r in rows:
        groups[r["stratum"], r["sequence"], r["case"]][r["cell"]] = r
    paired = []
    for stratum in s86.STRATA:
        for metric in METRICS:
            by_video = defaultdict(list)
            for (group, sequence, _), cells in groups.items():
                if group == stratum:
                    by_video[sequence].append([float(cells[key][metric]) for key in CELLS])
            values = np.asarray([np.mean(by_video[seq], axis=0) for seq in sorted(by_video)])
            direction = "low_iou_fraction" if metric in ("sustained_low_iou", "unresolved_at_window_end") else metric
            for effect, weights in EFFECTS.items():
                row = s86.contrast(values @ weights, effect, direction, design)
                paired.append(dict(row, metric=metric, stratum=stratum))
    return paired


def run(root, plan, inputs, source, actor, observer):
    old = source["original"]
    ap = e74.e67.actor_plan(old)
    design = plan["design"]
    rows, evidence, bridge_evidence = [], {}, {}
    for w, entry, frames, prefix, arms, cells in inputs:
        unit = w["unit"]
        continuations = dict(zip(("ff", "mm"), ({k: v[1:] for k, v in a.items()} for a in arms)))
        fresh_keys = [n for n, cell in cells.items() if cell["ref"] == n and n not in continuations]
        if fresh_keys:
            pack = e74.learned_pack(actor["models"], unit, actor["prior"])

            def choose(row, history):
                return e74.choose(row, history, unit, old, actor["prior"], pack)

            with e63.source_experiment.installed(observer, [], unit) as head:
                if head != source["report"]["native_head_sha256"]:
                    raise ValueError("Original SGLA head changed")
                bridge_path = root / "bridge" / (w["case"] + ".json")
                bridge = io.seal(dict(schema=SCHEMA, plan_fingerprint=plan["fingerprint"], case=w["case"], search_calls=8))
                if bridge_path.exists():
                    if c.c66.sealed(bridge_path, SCHEMA) != bridge:
                        raise ValueError("Mismatched replay bridge")
                else:
                    for mix, arm in zip((0., design["lag_mix"]), arms):
                        observed = s86.collect(observer, entry, prefix, frames[:1], mix, ap, choose)
                        s86.audit(observed, prefix, frames[:1], mix, ap, choose)
                        for k in s86.FIELDS:
                            np.testing.assert_array_equal(observed[k], arm[k][:1], err_msg=f"Stage-86 bridge changed {k}")
                    c.publish_identical(bridge_path, bridge)
                bridge_evidence[bridge_path.name] = c.file_hash(bridge_path)
                for name in fresh_keys:
                    key = w["case"] + "__" + name
                    path = root / "collection" / (key + ".npz")
                    meta = dict(plan_fingerprint=plan["fingerprint"], branch=key)
                    history = np.vstack([prefix, cells[name]["box"]])[-ap["design"]["history"]:]
                    tail = c.read_cache(path, meta)
                    missing = tail is None
                    if missing:
                        tail = s86.collect(observer, entry, history, frames[1:], 0., ap, choose)
                    s86.audit(tail, history, frames[1:], 0., ap, choose)
                    if missing:
                        c.save(path, tail, meta)
                    evidence[path.name] = c.file_hash(path)
                    continuations[name] = tail
                print(f"[Stage-87 cross] {w['case']} {len(evidence)}/{len(plan['new_branches'])} continuations", flush=True)
        truth = np.asarray(entry["ground_truth"], float)[frames]
        for name, cell in cells.items():
            stats = branch_metrics(arms[name[0] == "m"], continuations[cell["ref"]], cell, truth, design)
            rows.append(dict(ids(w), cell=name, continuation=cell["ref"], **stats))
    publish(root, plan, rows, evidence, bridge_evidence)


def publish(root, plan, rows, evidence, bridge_evidence):
    if (set(evidence) != {name + ".npz" for name in plan["new_branches"]}
            or len(bridge_evidence) != plan["bridge_windows"] or len(rows) != 4 * plan["windows"]):
        raise ValueError("Incomplete action cross")
    io.verify(plan["source_files"])
    io.verify(plan["code_sha256"])
    out = root / "analysis"
    s86.publish_table(out / "stage87_branches.csv", rows)
    s86.publish_table(out / "stage87_paired.csv", summarize(rows, plan["design"]))
    first = c.c66.sealed(out / "stage87_first_receipt.json", SCHEMA)
    io.verify({str(out / "stage87_first.csv"): first["sha256"]})
    runtime = c.c66.sealed(root / "runtime.json", SCHEMA)
    hashes = {f"stage87_{n}.csv": c.file_hash(out / f"stage87_{n}.csv") for n in TABLES}
    calls = len(evidence) * (plan["design"]["horizon"] - 1) * 4 + len(bridge_evidence) * 8
    report = io.seal(dict(schema=SCHEMA, status=STATUS, plan_fingerprint=plan["fingerprint"],
        source_stage86_fingerprint=plan["source_stage86_fingerprint"], runtime=runtime["runtime"],
        windows=plan["windows"], crossed_cells=len(rows), new_continuations=len(evidence), bridge_windows=len(bridge_evidence),
        search_calls=calls, block_calls=7*calls, confirmation_inference_calls=0, training_updates=0,
        cache_sha256=evidence, bridge_sha256=bridge_evidence, table_sha256=hashes))
    c.publish_identical(out / "stage87_decision.json", report)
    print(f"Stage-87: {STATUS} ({len(rows)} cells, {calls} new searches)", flush=True)


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    report = c.c66.sealed(root / "analysis/stage87_decision.json", SCHEMA)
    runtime = c.c66.sealed(root / "runtime.json", SCHEMA)
    calls = len(plan["new_branches"]) * (plan["design"]["horizon"] - 1) * 4 + plan["bridge_windows"] * 8
    if (report["status"] != STATUS or report["plan_fingerprint"] != plan["fingerprint"]
            or runtime["plan_fingerprint"] != plan["fingerprint"] or runtime["runtime"] != report["runtime"]
            or report["crossed_cells"] != 4 * plan["windows"] or report["search_calls"] != calls
            or report["block_calls"] != calls * 7 or report["new_continuations"] != len(plan["new_branches"])
            or set(report["cache_sha256"]) != {n + ".npz" for n in plan["new_branches"]}
            or set(report["bridge_sha256"]) != {n.rsplit("__", 1)[0] + ".json" for n in plan["new_branches"]}
            or set(report["table_sha256"]) != {f"stage87_{n}.csv" for n in TABLES}):
        raise ValueError("Incomplete Stage-87 report")
    for folder, records in (("analysis", "table_sha256"), ("collection", "cache_sha256"), ("bridge", "bridge_sha256")):
        io.verify({str(root / folder / name): sha for name, sha in report[records].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("stage86", "stage76", "output_dir"):
        parser.add_argument("--" + flag, type=Path, required=flag == "output_dir")
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--parameter", default="deit_distilled")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-results", action="store_true")
    modes.add_argument("--prepare-only", action="store_true")
    parser.set_defaults(mode="stage87")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not args.stage86 or not args.stage76 or (not args.prepare_only and not args.checkpoint):
        parser.error("Supply Stage-86, Stage-76 and the original checkpoint (checkpoint optional for prepare)")
    root86, root76 = (p.expanduser().resolve() for p in (args.stage86, args.stage76))
    if any(root == p or root in p.parents or p in root.parents for p in (root86, root76)):
        parser.error("Use a separate Stage-87 output directory")
    with writer_lock(root):
        if (root / "analysis/stage87_decision.json").exists():
            check_results(root)
            return
        if not (root / "manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-87 directory")
        manifest, report, source = open_input(root86, root76)
        plan = prepare(root, root86, manifest, report, source)
        print(f"Stage-87 cached: {plan['windows']} windows, {len(plan['new_branches'])} new continuations planned", flush=True)
        if args.prepare_only:
            return
        import torch

        torch.set_num_threads(1)
        runtime = s76.device_runtime(source["original"], torch, args.gpu)
        differences = s76.startup.differences(plan["source_runtime"], runtime)
        if any(row["field"] != "runtime.gpu_uuid" for row in differences):
            raise ValueError("Use a Stage-86 source from the same GPU model and software; rebuild Stage-86 in a separate root if needed")
        actor = io.load_source(Path(source["plan"]["source_input"]))
        if actor["plan"] != source["original"] or actor["models"]["fingerprint"] != source["report"]["source_training_fingerprint"]:
            raise ValueError("Stage-74 actor changed")
        c.publish_identical(root / "runtime.json", io.seal(dict(schema=SCHEMA, plan_fingerprint=plan["fingerprint"], runtime=runtime)))
        frozen = dict(source["original"], fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = s76.startup.checked_runtime(args, frozen, root)
        observer = e63.Capture(native)
        try:
            run(root, plan, cases(root86, manifest, source), source, actor, observer)
        finally:
            observer.handle.remove()


if __name__ == "__main__":
    main()
