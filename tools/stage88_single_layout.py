"""Audit one causal challenger per frozen fixed-layout PPO window."""
from __future__ import annotations

import argparse
from collections import defaultdict
import os
from pathlib import Path

import numpy as np

import stage87_action_cross as s87
from stage27_uav_collect import writer_lock


s86, c, io, e63, e74, s76 = s87.s86, s87.c, s87.io, s87.e63, s87.e74, s87.s76
SCHEMA = "stage88_single_layout_write_support_v1"
STATUS = "COMPLETE_SINGLE_LAYOUT_WRITE_SUPPORT_DIAGNOSTIC"
SIGNALS = ("hann", "velocity", "consensus")
METHODS = ("always_challenger",) + SIGNALS
DESIGN = dict(material_gain=.05, minimum_source_groups_per_sign=3, bootstrap_seed=2188)
TABLES = ("windows", "paired", "signal_support")


def challenger(fixed):
    boxes = fixed["candidate_boxes"][0]
    proposed = int(fixed["selected"][0])
    peaks = fixed["features"][0, :, s76.branches.prior.FEATURES.index("hann_peak")]
    legal = c.action_mask(boxes)
    alternatives = np.flatnonzero(legal & np.any(boxes != boxes[proposed], axis=1))
    if not len(alternatives):
        return None, {}
    q = int(alternatives[np.argmax(peaks[alternatives])])
    latest = fixed["history_latest"][0]
    center = s86.centers(latest)
    lag = s86.centers(fixed["candidate_parents"][0, 2])
    centers = s86.centers(boxes)
    side = np.sqrt(latest[2:].prod())
    velocity = np.linalg.norm(centers - (2 * center - lag), axis=1) / side
    consensus = np.linalg.norm(centers - np.median(centers[legal], axis=0), axis=1) / side
    return q, dict(hann=float(peaks[q] - peaks[proposed]),
                   velocity=float(velocity[proposed] - velocity[q]),
                   consensus=float(consensus[proposed] - consensus[q]))


def planned_cases(root86, manifest, source):
    for w, entry, frames, prefix, arms, cells in s87.cases(root86, manifest, source):
        fixed = arms[0]
        q, signals = challenger(fixed)
        box = fixed["boxes"][0] if q is None else fixed["candidate_boxes"][0, q]
        ref = next((cells[k]["ref"] for k in ("ff", "mm", "fm", "mf") if np.array_equal(cells[k]["box"], box)), "new")
        yield dict(window=w, entry=entry, frames=frames, prefix=prefix, arms=arms,
                   action=q, signals=signals, box=box, ref=ref)


def resolve_tail(item, root87, plan87, root, plan, source, actor, observer):
    ref = item["ref"]
    if ref in ("ff", "mm"):
        return {k: v[1:] for k, v in item["arms"][ref == "mm"].items()}, None
    w, frames = item["window"], item["frames"]
    case = w["case"]
    if ref != "new":
        key = case + "__" + ref
        tail = c.read_cache(root87 / "collection" / (key + ".npz"),
                            dict(plan_fingerprint=plan87["fingerprint"], branch=key))
        if tail is None:
            raise ValueError("Missing sealed Stage-87 continuation")
        np.testing.assert_array_equal(tail["frame_id"], frames[1:])
        np.testing.assert_array_equal(tail["history_latest"][0], item["box"])
        return tail, None

    old, unit = source["original"], w["unit"]
    ap = e74.e67.actor_plan(old)
    pack = e74.learned_pack(actor["models"], unit, actor["prior"])

    def choose(row, history):
        return e74.choose(row, history, unit, old, actor["prior"], pack)

    prefix = item["prefix"]
    history = np.vstack([prefix, item["box"]])[-ap["design"]["history"]:]
    path = root / "collection" / (case + ".npz")
    meta = dict(plan_fingerprint=plan["fingerprint"], case=case)
    tail = c.read_cache(path, meta)
    if tail is None:
        with e63.source_experiment.installed(observer, [], unit) as head:
            if head != source["report"]["native_head_sha256"]:
                raise ValueError("Original SGLA head changed")
            observed = s86.collect(observer, item["entry"], prefix, frames[:1], 0., ap, choose)
            for k in s86.FIELDS:
                np.testing.assert_array_equal(observed[k], item["arms"][0][k][:1],
                                              err_msg=f"Stage-86 fixed-layout replay changed {k}")
            tail = s86.collect(observer, item["entry"], history, frames[1:], 0., ap, choose)
        s86.audit(tail, history, frames[1:], 0., ap, choose)
        c.save(path, tail, meta)
    else:
        s86.audit(tail, history, frames[1:], 0., ap, choose)
    return tail, path


def record(item, tail, design):
    fixed = item["arms"][0]
    truth = np.asarray(item["entry"]["ground_truth"], float)
    frames = item["frames"]
    anchor_iou = float(s86.tracking_iou(item["prefix"][-1], truth[frames[0] - 1]))
    original = s86.metrics(fixed, truth[frames], design)
    q = int(fixed["selected"][0]) if item["action"] is None else item["action"]
    alternate = s87.branch_metrics(fixed, tail, dict(box=item["box"], action=q), truth[frames], design)
    base_followup = float(s86.tracking_iou(fixed["boxes"][1:], truth[frames[1:]]).mean())
    return dict(s87.ids(item["window"]), anchor_iou=anchor_iou,
        ppo_action=int(fixed["selected"][0]), challenger_action=item["action"], continuation=item["ref"],
        **{name + "_margin": item["signals"].get(name) for name in SIGNALS},
        ppo_followup_iou=base_followup, challenger_followup_iou=alternate["followup_mean_iou"],
        followup_gain=alternate["followup_mean_iou"] - base_followup,
        ppo_first_iou=original["first_iou"], challenger_first_iou=alternate["first_iou"],
        ppo_low_iou=original["low_iou_fraction"], challenger_low_iou=alternate["low_iou_fraction"],
        ppo_return=original["reaches_sustained_good"], challenger_return=alternate["reaches_sustained_good"])


def summarize(rows, design):
    paired, signal_support = [], []
    threshold = design["material_gain"]
    for stratum in s86.STRATA:
        cohort = [r for r in rows if r["stratum"] == stratum]
        eligible = [r for r in cohort if r["challenger_action"] is not None]
        helpful = [r for r in eligible if r["followup_gain"] >= threshold]
        harmful = [r for r in eligible if r["followup_gain"] <= -threshold]
        recoveries = [r for r in eligible if r["anchor_iou"] < design["low_iou"]
                      and r["ppo_return"] and not r["challenger_return"]]
        gains_in_recovery = [r for r in eligible if r["anchor_iou"] < design["low_iou"]
                             and r["challenger_return"] and not r["ppo_return"]]
        helpful_groups, harmful_groups = (len({s86.confirmation.source_group(r["sequence"]) for r in g})
                                            for g in (helpful, harmful))
        support = min(helpful_groups, harmful_groups) >= design["minimum_source_groups_per_sign"]
        for method in METHODS:
            scores = {r["case"]: (1. if method == "always_challenger" else r[method + "_margin"]) for r in eligible}
            selected = {key for key, value in scores.items() if value > 1e-12}
            gains, lows = defaultdict(list), defaultdict(list)
            for r in cohort:
                take = r["case"] in selected
                gains[r["sequence"]].append(r["followup_gain"] if take else 0.)
                lows[r["sequence"]].append(r["challenger_low_iou"] - r["ppo_low_iou"] if take else 0.)
            for metric, values in (("followup_mean_iou", gains), ("low_iou_fraction", lows)):
                paired.append(dict(s86.contrast([np.mean(values[k]) for k in sorted(values)], method + "_minus_ppo",
                                               metric, design), stratum=stratum, windows=len(cohort)))
            auc = None
            if support and method != "always_challenger":
                diff = np.array([scores[r["case"]] for r in helpful])[:, None] - [scores[r["case"]] for r in harmful]
                auc = float(np.mean((diff > 1e-12) + .5 * (np.abs(diff) <= 1e-12)))
            signal_support.append(dict(stratum=stratum, method=method, windows=len(cohort), eligible=len(eligible),
                material_helpful=len(helpful), helpful_source_groups=helpful_groups,
                material_harmful=len(harmful), harmful_source_groups=harmful_groups, rank_support=support, rank_auc=auc,
                switches=len(selected), helpful_switches=sum(r["case"] in selected for r in helpful),
                harmful_switches=sum(r["case"] in selected for r in harmful), ppo_only_recovery_windows=len(recoveries),
                ppo_only_recoveries_lost=sum(r["case"] in selected for r in recoveries),
                challenger_only_recovery_windows=len(gains_in_recovery),
                challenger_only_recoveries_gained=sum(r["case"] in selected for r in gains_in_recovery)))
    return paired, signal_support


def publish(root, plan, rows, caches):
    if len(rows) != plan["windows"] or set(caches) != {name + ".npz" for name in plan["new"]}:
        raise ValueError("Incomplete Stage-88 evidence")
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])
    paired, support = summarize(rows, plan["design"])
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    hashes = {}
    for name, table in zip(TABLES, (rows, paired, support)):
        path = out / f"stage88_{name}.csv"
        s86.publish_table(path, table)
        hashes[path.name] = c.file_hash(path)
    calls = len(caches) * plan["design"]["horizon"] * 4
    runtime_path = root / "runtime.json"
    runtime = c.c66.sealed(runtime_path, SCHEMA)["runtime"] if runtime_path.exists() else None
    c.publish_identical(out / "stage88_decision.json", io.seal(dict(schema=SCHEMA, status=STATUS,
        plan_fingerprint=plan["fingerprint"], source_stage87_fingerprint=plan["source_stage87_fingerprint"],
        windows=len(rows), eligible=sum(r["challenger_action"] is not None for r in rows), new_continuations=len(caches),
        runtime=runtime, search_calls=calls, block_calls=7*calls, training_updates=0, confirmation_inference_calls=0,
        support=support, table_sha256=hashes, cache_sha256=caches)))
    print(f"Stage-88: {STATUS} ({len(rows)} windows, {calls} new searches)", flush=True)


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    report = c.c66.sealed(root / "analysis/stage88_decision.json", SCHEMA)
    if (report["status"] != STATUS or report["plan_fingerprint"] != plan["fingerprint"]
            or report["windows"] != plan["windows"]
            or report["search_calls"] != len(plan["new"]) * plan["design"]["horizon"] * 4
            or report["block_calls"] != report["search_calls"] * 7
            or set(report["cache_sha256"]) != {key + ".npz" for key in plan["new"]}
            or set(report["table_sha256"]) != {f"stage88_{n}.csv" for n in TABLES}):
        raise ValueError("Incomplete Stage-88 report")
    for folder, field in (("analysis", "table_sha256"), ("collection", "cache_sha256")):
        io.verify({str(root / folder / key): sha for key, sha in report[field].items()})


def execute(root, inputs, root87, plan87, plan, source, actor, observer):
    rows, caches = [], {}
    for item in inputs:
        tail, path = resolve_tail(item, root87, plan87, root, plan, source, actor, observer)
        if path is not None:
            caches[path.name] = c.file_hash(path)
            print(f"[Stage-88] {item['window']['case']} {len(caches)}/{len(plan['new'])} continuations", flush=True)
        rows.append(record(item, tail, plan["design"]))
    publish(root, plan, rows, caches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("stage76", "stage86", "stage87", "output_dir"):
        parser.add_argument("--" + flag, type=Path, required=flag == "output_dir")
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-results", action="store_true")
    modes.add_argument("--plan-only", action="store_true")
    parser.set_defaults(mode="stage88")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not all((args.stage76, args.stage86, args.stage87)):
        parser.error("Supply the Stage-76, Stage-86 and Stage-87 output directories")
    root76, root86, root87 = (p.expanduser().resolve() for p in (args.stage76, args.stage86, args.stage87))
    if any(root == p or root in p.parents or p in root.parents for p in (root76, root86, root87)):
        parser.error("Use a separate Stage-88 output directory")
    with writer_lock(root):
        if (root / "analysis/stage88_decision.json").exists():
            check_results(root)
            return
        if not (root / "manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-88 directory")
        manifest86, report86, source = s87.open_input(root86, root76)
        s87.check_results(root87)
        plan87 = c.c66.sealed(root87 / "manifest.json", s87.SCHEMA)
        report87 = c.c66.sealed(root87 / "analysis/stage87_decision.json", s87.SCHEMA)
        if (plan87["source_stage86_plan_fingerprint"] != manifest86["fingerprint"]
                or report87["source_stage86_fingerprint"] != report86["fingerprint"]):
            raise ValueError("Stage-86 and Stage-87 caches come from different runs")
        inputs = list(planned_cases(root86, manifest86, source))
        new = [item["window"]["case"] for item in inputs if item["ref"] == "new"]
        code = [Path(__file__), Path(s87.__file__), Path(s86.__file__), Path(e74.__file__),
                Path(__file__).with_name("run_stage88_single_layout.sh")]
        files = [p / "manifest.json" for p in (root76, root86, root87)]
        files += [p / f"analysis/stage{i}_decision.json" for p, i in ((root76, 76), (root86, 86), (root87, 87))]
        plan = io.seal(dict(schema=SCHEMA, windows=len(inputs), new=new, design=manifest86["design"] | DESIGN,
            source_stage87_fingerprint=report87["fingerprint"], source_sha256={str(p): c.file_hash(p) for p in files},
            code_sha256={str(p.resolve()): c.file_hash(p) for p in code}))
        c.publish_identical(root / "manifest.json", plan)
        print(f"Stage-88 planned: {len(inputs)} windows, {len(new)} new continuations", flush=True)
        if args.plan_only:
            return
        if not new:
            execute(root, inputs, root87, plan87, plan, source, None, None)
            return
        if not args.checkpoint:
            parser.error("Supply the original checkpoint for new continuations")
        import torch

        torch.set_num_threads(1)
        runtime = s76.device_runtime(source["original"], torch, args.gpu)
        differences = s76.startup.differences(report86["runtime"], runtime)
        if any(row["field"] != "runtime.gpu_uuid" for row in differences):
            raise ValueError("Need a matching Stage-86 GPU model and software for new continuations")
        actor = io.load_source(Path(source["plan"]["source_input"]))
        if actor["plan"] != source["original"] or actor["models"]["fingerprint"] != source["report"]["source_training_fingerprint"]:
            raise ValueError("Stage-74 actor changed")
        c.publish_identical(root / "runtime.json", io.seal(dict(schema=SCHEMA, plan_fingerprint=plan["fingerprint"], runtime=runtime)))
        frozen = dict(source["original"], fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = s76.startup.checked_runtime(args, frozen, root)
        observer = e63.Capture(native)
        try:
            execute(root, inputs, root87, plan87, plan, source, actor, observer)
        finally:
            observer.handle.remove()


if __name__ == "__main__":
    main()
