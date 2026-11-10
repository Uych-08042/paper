"""Compare causal short-term appearance memory with the frozen first template."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

import stage91_template_appearance as s91
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image
from stage28_response_common import cpu_array


s90, s89, s86, c, io = s91.s90, s91.s89, s91.s86, s91.c, s91.io
e63, e74, s76 = s91.e63, s91.e74, s91.s76
SCHEMA = "stage92_recent_appearance_v1"
STATUS = "COMPLETE_RECENT_APPEARANCE_AUDIT"
SOURCE91 = "7eaffcb3c3903bca960a9b56b71db9176ba1a3c389513e7297847d614519a44c"
BUNDLE91 = Path(__file__).resolve().parents[1] / "docs/experiments/06_sgla_uav_tracking/results/stage91_results_20261110T052818Z_1112822.txt"
DESIGN = dict(s91.DESIGN, lookback=12, max_refs=3, min_refs=2, spacing=3, bootstrap_seed=2192)
METHODS = ("patch", "recent", "gated", "anchored")
TABLES = ("appearance", "signal_support", "paired")


def load_fixed(path, windows, report89):
    import csv
    from io import StringIO

    raw = path.read_bytes()
    pieces = re.split(rb"^===== (stage91_\S+) =====\r?\n", raw, flags=re.M)
    sections = {k.decode(): v.strip(b"\r\n") for k, v in zip(pieces[1::2], pieces[2::2])}
    if len(pieces) != 9 or set(sections) != {"stage91_decision.json", *(f"stage91_{n}.csv" for n in s91.TABLES)}:
        raise ValueError("Expected the original Stage-91 result bundle")
    report = json.loads(sections["stage91_decision.json"])
    body = {k: v for k, v in report.items() if k != "fingerprint"}
    if (report["status"] != s91.STATUS or report["schema"] != s91.SCHEMA
            or c.fingerprint(body) != report["fingerprint"] or report["fingerprint"] != SOURCE91
            or report["source_stage89_fingerprint"] != report89["fingerprint"]
            or set(report["table_sha256"]) != {f"stage91_{n}.csv" for n in s91.TABLES}):
        raise ValueError("Unexpected Stage-91 decision")
    for name, digest in report["table_sha256"].items():
        if hashlib.sha256(sections[name] + b"\r\n").hexdigest() != digest:
            raise ValueError("Stage-91 table bytes differ: " + name)
    rows = list(csv.DictReader(StringIO(sections["stage91_appearance.csv"].decode())))
    fixed = {r["case"]: r for r in rows}
    if len(rows) != len(fixed) or set(fixed) != {r["case"] for r in windows}:
        raise ValueError("Stage-91 window set changed")
    for row in windows:
        archived = fixed[row["case"]]
        if any(archived[k] != str(row[k]) for k in row):
            raise ValueError("Stage-91 matched-action fields changed")
        archived["patch_scores"] = [float(archived[k]) for k in ("patch_ppo", "patch_challenger")]
        if not np.isfinite(archived["patch_scores"]).all():
            raise ValueError("Missing Stage-91 fixed appearance")
    return fixed, hashlib.sha256(raw).hexdigest()


def trusted(trace, i, design):
    boxes, selected = trace["candidate_boxes"][i], int(trace["selected"][i])
    peaks = trace["features"][i, :, s76.branches.prior.FEATURES.index("hann_peak")]
    previous = trace["history_latest"][i]
    side = np.sqrt(previous[2:].prod())
    centers = s86.centers(boxes)
    if np.linalg.norm(centers[selected] - s86.centers(previous)) / side > design["center_separation"]:
        return False
    if peaks[c.action_mask(boxes)].max() - peaks[selected] > design["hann_gap"]:
        return False
    distance = np.linalg.norm(centers - centers[selected], axis=1) / side
    return not np.any(c.action_mask(boxes) & (distance >= design["center_separation"])
                      & (np.abs(peaks - peaks[selected]) <= design["hann_gap"]))


def reference_frames(trace, first, design):
    """Choose from earlier frames only, using the same budget and spacing in both arms."""
    refs = {}
    for mode in ("recent", "gated"):
        chosen = []
        for frame in range(first - 1, max(1, first - design["lookback"]) - 1, -1):
            if chosen and chosen[-1] - frame < design["spacing"]:
                continue
            if mode == "gated" and not trusted(trace, frame - 1, design):
                continue
            chosen.append(frame)
            if len(chosen) == design["max_refs"]:
                break
        refs[mode] = chosen
    return refs


def read_slot(observer, image, entry, trace, frame, slot, ap, design):
    i = frame - 1
    parent = trace["candidate_parents"][i, slot]
    history = np.vstack([entry["initial_box"], trace["boxes"][max(0, i - ap["design"]["history"]):i]])[-ap["design"]["history"]:]
    np.testing.assert_array_equal(history[-1], trace["history_latest"][i])
    embeddings = []

    def capture(_module, _args, output):
        array = cpu_array(output)
        if array.ndim != 3 or array.shape[0] != 1:
            raise ValueError("Expected one image's patch grid")
        embeddings.append(array[0])

    rt = observer.rt
    with rt.net.backbone.patch_embed.register_forward_hook(capture):
        out = e63.actor.observe(observer, image, parent, frame, ap)
    for key in ("crop", "crop_sha256", "head_calls", "block_calls", "selector_calls", "layer"):
        np.testing.assert_array_equal(out[key], trace[key][i, slot], err_msg=f"Stage-89 slot replay changed {key}")
    box = trace["candidate_boxes"][i, slot]
    np.testing.assert_array_equal(out["boxes"], box)
    np.testing.assert_array_equal(out["image_hw"], trace["image_hw"][i])
    np.testing.assert_array_equal(e63.actor.feature_vector(out, list(history), ap), trace["features"][i, slot])
    if len(embeddings) != 2 or (len(embeddings[0]), len(embeddings[1])) != (rt.lens_z, rt.lens_x):
        raise ValueError("Expected the original template/search patch pair")
    feature, _ = s91.region(image, box, parent, ap["search_factor"], ap["search_size"], embeddings[1], design)
    return feature[0], embeddings[0]


def collect(observer, entry, trace, first, slots, refs, expected, ap, design):
    initial = checked_image(ap["root"], entry["frames"][0])
    observer.initialize(initial, entry["initial_box"])
    current = checked_image(ap["root"], entry["frames"][first])
    queries, anchors = zip(*(read_slot(observer, current, entry, trace, first, q, ap, design) for q in slots))
    np.testing.assert_array_equal(anchors[0], anchors[1])
    if any(query is None for query in queries):
        raise ValueError("Stage-91 current candidate descriptor became unavailable")
    init = np.asarray(entry["initial_box"], float)
    params = observer.rt.tracker.params
    anchor, _ = s91.region(initial, init, init, params.template_factor, params.template_size, anchors[0], design)
    np.testing.assert_allclose([anchor[0] @ q for q in queries], expected, rtol=0., atol=1e-12,
                               err_msg="Stage-91 fixed appearance changed")
    features = {}
    for frame in sorted(set(refs["recent"]) | set(refs["gated"])):
        image = checked_image(ap["root"], entry["frames"][frame])
        slot = int(trace["selected"][frame - 1])
        features[frame], _ = read_slot(observer, image, entry, trace, frame, slot, ap, design)
    counts, margins = [], []
    fixed = expected[1] - expected[0]
    for mode in ("recent", "gated"):
        values = [features[frame] for frame in refs[mode] if features[frame] is not None]
        counts.append(len(values))
        margins.append(float((queries[1] - queries[0]) @ np.mean(values, axis=0)) if len(values) >= design["min_refs"] else fixed)
    margins += [(fixed + margins[1]) / 2]
    return dict(margins=np.asarray([fixed, *margins]), counts=np.asarray(counts))


def summarize(rows, design):
    support, paired = [], []
    flags = ("material_help", "material_harm", "tail_help", "tail_harm", "ppo_only_recovery", "challenger_only_recovery")
    for cohort in ("all", "expanded", "historical"):
        for stratum in s89.STRATA:
            group = [r for r in rows if r["stratum"] == stratum and (cohort == "all" or r["cohort"] == cohort)]
            if not group:
                continue
            good, bad = ([r for r in group if r[k]] for k in flags[:2])
            good_groups, bad_groups = (len({r["source_group"] for r in subset}) for subset in (good, bad))
            for method in METHODS:
                key = method + "_margin"
                take = {r["case"] for r in group if r[key] > design["switch_margin"]}
                auc = None
                if min(good_groups, bad_groups) >= design["minimum_source_groups_per_sign"]:
                    delta = np.array([r[key] for r in good])[:, None] - [r[key] for r in bad]
                    auc = float(np.mean((delta > 1e-12) + .5 * (np.abs(delta) <= 1e-12)))
                support.append(dict(cohort=cohort, stratum=stratum, method=method, windows=len(group),
                    memory_supported=sum(r[("recent" if method == "recent" else "gated") + "_usable"] >= design["min_refs"]
                                         for r in group) if method != "patch" else None,
                    switches=len(take), good_rank_sources=good_groups, bad_rank_sources=bad_groups, rank_auc=auc,
                    **{f"{k}_available": sum(r[k] for r in group) for k in flags},
                    **{f"{k}_switched": sum(r[k] and r["case"] in take for r in group) for k in flags}))
                for baseline in (("ppo",) if method == "patch" else ("ppo", "patch")):
                    by_video = defaultdict(list)
                    for r in group:
                        ref = baseline == "patch" and r["patch_margin"] > design["switch_margin"]
                        factor = int(r["case"] in take) - int(ref)
                        by_video[r["source_group"]].append((factor * r["followup_gain"],
                                                          factor * r["low_iou_delta_frames"] / design["horizon"]))
                    values = np.array([np.mean(by_video[g], axis=0) for g in sorted(by_video)])
                    for i, metric in enumerate(("followup_mean_iou", "low_iou_fraction")):
                        paired.append(dict(s86.contrast(values[:, i], method + "_minus_" + baseline, metric, design),
                                           cohort=cohort, stratum=stratum, windows=len(group)))
    return support, paired


def run(root, plan, source, root89, plan89, rows, fixed, observer):
    by_unit = defaultdict(list)
    for r in rows:
        by_unit[r["case"].rsplit("__f", 1)[0]].append(r)
    ap = e74.e67.actor_plan(source["original"])
    evidence, scored, calls = {}, [], 0
    for spec in plan89["units"]:
        unit = spec["unit"]
        key = c.key(unit)
        if key not in by_unit:
            continue
        if spec["cohort"] == "expanded":
            bound = c.c66.sealed(root89 / "entries" / (unit["sequence"] + ".json"), s89.SCHEMA)
            if bound["plan_fingerprint"] != plan89["fingerprint"] or bound["entry"]["sequence"] != unit["sequence"]:
                raise ValueError("Stage-89 expanded entry changed")
            entry = bound["entry"]
        else:
            entry, trace = s86.previous.read_source_trace(source, dict(unit=unit, frames=[1], expected_first_failure=None))
        if spec["cohort"] == "expanded" or not plan89["reuse_historical"]:
            trace = c.read_cache(root89 / "baseline" / (key + ".npz"), dict(plan_fingerprint=plan89["fingerprint"], unit=unit))
            if trace is None:
                raise ValueError("Missing Stage-89 natural trajectory")
        with e63.source_experiment.installed(observer, [], unit) as head:
            if head != source["report"]["native_head_sha256"]:
                raise ValueError("Original SGLA head changed")
            for row in by_unit.pop(key):
                first = int(row["first"])
                slots = tuple(int(row[k]) for k in ("ppo_action", "challenger_action"))
                bank = {k: trace[k][first - 1:first] for k in
                        ("candidate_boxes", "candidate_parents", "history_latest", "features", "selected")}
                if int(bank["selected"][0]) != slots[0] or s89.s88.challenger(bank)[0] != slots[1]:
                    raise ValueError("Stage-89 action pair changed")
                refs = reference_frames(trace, first, plan["design"])
                path = root / "collection" / (row["case"] + ".npz")
                meta = dict(plan_fingerprint=plan["fingerprint"], case=row["case"], refs=refs)
                data = c.read_cache(path, meta)
                if data is None:
                    data = collect(observer, entry, trace, first, slots, refs, fixed[row["case"]]["patch_scores"], ap, plan["design"])
                    c.save(path, data, meta)
                if data["margins"].shape != (4,) or not np.isfinite(data["margins"]).all() or data["counts"].shape != (2,):
                    raise ValueError("Incomplete appearance scores")
                evidence[path.name] = c.file_hash(path)
                searches = 2 + len(set(refs["recent"]) | set(refs["gated"]))
                calls += searches
                scored.append(dict(row, **{k + "_margin": float(v) for k, v in zip(METHODS, data["margins"])},
                    recent_frames="|".join(map(str, refs["recent"])), gated_frames="|".join(map(str, refs["gated"])),
                    recent_usable=int(data["counts"][0]), gated_usable=int(data["counts"][1]), searches=searches))
        print(f"[Stage-92] {key} {len(scored)}/{len(rows)}", flush=True)
    if by_unit or len(scored) != len(rows):
        raise ValueError("Incomplete Stage-92 window coverage")
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])
    support, paired = summarize(scored, plan["design"])
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    hashes = {}
    for name, data in zip(TABLES, (scored, support, paired)):
        path = out / f"stage92_{name}.csv"
        s86.publish_table(path, data)
        hashes[path.name] = c.file_hash(path)
    c.publish_identical(out / "stage92_decision.json", io.seal(dict(schema=SCHEMA, status=STATUS,
        plan_fingerprint=plan["fingerprint"], source_stage91_fingerprint=SOURCE91, runtime=plan["runtime"],
        windows=len(scored), search_calls=calls, block_calls=7 * calls, training_updates=0, new_continuations=0,
        confirmation_inference_calls=0, cache_sha256=evidence, table_sha256=hashes)))
    print(f"Stage-92: {STATUS} ({len(scored)} windows, {calls} diagnostic searches)", flush=True)


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    report = c.c66.sealed(root / "analysis/stage92_decision.json", SCHEMA)
    count = len(plan["cases"])
    if (report["status"] != STATUS or report["plan_fingerprint"] != plan["fingerprint"] or report["windows"] != count
            or not 2 * count <= report["search_calls"] <= (2 + 2 * plan["design"]["max_refs"]) * count
            or report["block_calls"] != 7 * report["search_calls"]
            or set(report["cache_sha256"]) != {f"{case}.npz" for case in plan["cases"]}
            or set(report["table_sha256"]) != {f"stage92_{name}.csv" for name in TABLES}):
        raise ValueError("Incomplete Stage-92 result")
    for folder, field in (("analysis", "table_sha256"), ("collection", "cache_sha256")):
        io.verify({str(root / folder / name): digest for name, digest in report[field].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("stage76", "stage89", "output_dir"):
        parser.add_argument("--" + flag, type=Path, required=flag == "output_dir")
    parser.add_argument("--bundle89", type=Path, default=s90.DEFAULT_INPUT)
    parser.add_argument("--bundle91", type=Path, default=BUNDLE91)
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage92")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not all((args.stage76, args.stage89, args.checkpoint)):
        parser.error("Supply Stage-76, Stage-89 and the original checkpoint")
    root76, root89 = (p.expanduser().resolve() for p in (args.stage76, args.stage89))
    if any(root == p or p in root.parents or root in p.parents for p in (root76, root89)):
        parser.error("Use a separate Stage-92 output directory")
    import torch

    torch.set_num_threads(1)
    with writer_lock(root):
        if (root / "analysis/stage92_decision.json").exists():
            check_results(root)
            return
        if not (root / "manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-92 directory")
        windows, report89, hash89 = s90.read_bundle(args.bundle89)
        rows = s90.decisions(windows, DESIGN)
        fixed, hash91 = load_fixed(args.bundle91, rows, report89)
        s89.check_results(root89)
        plan89 = c.c66.sealed(root89 / "manifest.json", s89.SCHEMA)
        if c.c66.sealed(root89 / "analysis/stage89_decision.json", s89.SCHEMA) != report89:
            raise ValueError("Stage-89 run differs from the archived bundle")
        source = s86.previous.load_source(root76)
        if plan89["source_stage76_fingerprint"] != source["report"]["fingerprint"]:
            raise ValueError("Stage-76 lineage differs from Stage-89")
        runtime = s76.device_runtime(source["original"], torch, args.gpu)
        if any(r["field"] not in ("runtime.gpu_uuid", "runtime.tracker.device")
               for r in s76.startup.differences(report89["runtime"], runtime)):
            raise ValueError("Frozen SGLA runtime software or weights changed")
        code = [Path(__file__), Path(s91.__file__), Path(s90.__file__), Path(e63.actor.__file__),
                Path(__file__).with_name("run_stage92_recent_appearance.sh")]
        plan = io.seal(dict(schema=SCHEMA, design=DESIGN, runtime=runtime, cases=[r["case"] for r in rows],
            source_stage91_fingerprint=SOURCE91,
            source_sha256={str(args.bundle89.resolve()): hash89, str(args.bundle91.resolve()): hash91,
                           str(root89 / "manifest.json"): c.file_hash(root89 / "manifest.json")},
            code_sha256={str(p.resolve()): c.file_hash(p) for p in code}))
        c.publish_identical(root / "manifest.json", plan)
        frozen = dict(source["original"], fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = s76.startup.checked_runtime(args, frozen, root)
        observer = e63.Capture(native)
        try:
            run(root, plan, source, root89, plan89, rows, fixed, observer)
        finally:
            observer.handle.remove()


if __name__ == "__main__":
    main()
