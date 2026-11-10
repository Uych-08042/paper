"""Read existing SGLA patch features at the frozen Stage-89 action pairs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import math
import os
from pathlib import Path

import numpy as np

import stage90_frozen_signal_audit as s90
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, crop_geometry
from stage28_response_common import cpu_array
from stage31_identity_common import roi_weights


s89, s88, s86, c, io = s90.s89, s90.s88, s90.s86, s90.c, s90.io
e63, e74, s76 = s89.e63, s89.e74, s89.s76
SCHEMA = "stage91_frozen_template_appearance_v1"
STATUS = "COMPLETE_FROZEN_TEMPLATE_APPEARANCE_AUDIT"
DESIGN = dict(s90.DESIGN, min_roi_coverage=.5, rgb_bins=8, bootstrap_seed=2191)
SIGNALS = ("patch", "rgb")
TABLES = ("appearance", "signal_support", "paired")


def visible_roi(box, usable):
    box, usable = np.asarray(box, float), np.asarray(usable, float)
    xy = np.maximum(box[:2], usable[:2])
    wh = np.maximum(0., np.minimum(box[:2] + box[2:], usable[:2] + usable[2:]) - xy)
    return np.r_[xy, wh], float(wh.prod() / box[2:].prod())


def descriptor(grid, crop, roi):
    side = math.isqrt(len(grid))
    if grid.ndim != 2 or side * side != len(grid) or not np.isfinite(grid).all():
        raise ValueError("Expected a square patch token grid")
    vector = roi_weights(roi[None], crop, side)[0].astype(float) @ grid
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else None


def histogram(image, roi, bins):
    left, top = np.floor(roi[:2]).astype(int)
    right, bottom = np.ceil(roi[:2] + roi[2:]).astype(int)
    pixels = image[top:bottom, left:right].reshape(-1, 3)
    if not len(pixels):
        return None
    indices = pixels.astype(int) * bins // 256
    counts = np.concatenate([np.bincount(indices[:, channel], minlength=bins) for channel in range(3)])
    return np.sqrt(counts / counts.sum())


def region(image, box, parent, factor, size, tokens, design):
    crop, usable, _ = crop_geometry(parent, image.shape[:2], factor, size)
    roi, coverage = visible_roi(box, usable)
    if coverage < design["min_roi_coverage"]:
        return (None, None), coverage
    return (descriptor(tokens, crop, roi), histogram(image, roi, design["rgb_bins"])), coverage


def observe(observer, entry, item, ap, choose, design):
    initial = checked_image(ap["root"], entry["frames"][0])
    first = int(item["frames"][0])
    image = checked_image(ap["root"], entry["frames"][first])
    history = list(item["prefix"])
    observer.initialize(initial, entry["initial_box"])
    embeddings = []

    def capture(_module, _args, output):
        # Copy before SGLA adds the positional embeddings in place.
        values = cpu_array(output)
        if values.ndim != 3 or values.shape[0] != 1:
            raise ValueError("Expected one image's patch embeddings")
        embeddings.append(values[0])

    with observer.rt.net.backbone.patch_embed.register_forward_hook(capture):
        row = e63.probe(observer, image, history, first, "history", ap)
        row = choose(row, history)
    fixed = item["arms"][0]
    for k in s86.FIELDS:
        np.testing.assert_array_equal(row[k], fixed[k][0], err_msg=f"Stage-89 first-frame replay changed {k}")
    rt = observer.rt
    if len(embeddings) != 8 or any(len(x) != (rt.lens_z if i % 2 == 0 else rt.lens_x)
                                  for i, x in enumerate(embeddings)):
        raise ValueError("Expected template/search embeddings from exactly four native searches")
    for tokens in embeddings[2::2]:
        np.testing.assert_array_equal(tokens, embeddings[0])
    init = np.asarray(entry["initial_box"], float)
    params = rt.tracker.params
    template, template_coverage = region(initial, init, init, params.template_factor, params.template_size, embeddings[0], design)
    scores, coverages = [], []
    for q in (int(row["selected"]), item["action"]):
        feature, coverage = region(image, row["candidate_boxes"][q], row["candidate_parents"][q],
                                   ap["search_factor"], ap["search_size"], embeddings[2 * q + 1], design)
        scores.append([float(a @ b) if a is not None and b is not None else np.nan for a, b in zip(template, feature)])
        coverages.append(coverage)
    return dict(scores=np.asarray(scores), coverage=np.asarray([template_coverage, *coverages]))


def summarize(rows, design):
    support, paired = [], []
    for cohort in ("all", "expanded", "historical"):
        for stratum in s89.STRATA:
            group = [r for r in rows if r["stratum"] == stratum and (cohort == "all" or r["cohort"] == cohort)]
            if not group:
                continue
            common = [r for r in group if all(r[name + "_margin"] is not None for name in SIGNALS)]
            good = [r for r in common if r["material_help"]]
            bad = [r for r in common if r["material_harm"]]
            good_groups, bad_groups = (len({r["source_group"] for r in subset}) for subset in (good, bad))
            for method in (*SIGNALS, *s88.SIGNALS):
                key = method + "_margin"
                eligible = [r for r in group if r[key] is not None]
                auc = None
                if min(good_groups, bad_groups) >= design["minimum_source_groups_per_sign"]:
                    differences = np.array([r[key] for r in good])[:, None] - [r[key] for r in bad]
                    auc = float(np.mean((differences > 1e-12) + .5 * (np.abs(differences) <= 1e-12)))
                take = {r["case"] for r in eligible if r[key] > design["switch_margin"]}
                videos = defaultdict(list)
                for r in group:
                    gain = (r["followup_gain"], r["low_iou_delta_frames"] / design["horizon"]) if r["case"] in take else (0., 0.)
                    videos[r["source_group"]].append(gain)
                values = np.array([np.mean(videos[name], axis=0) for name in sorted(videos)])
                flags = ("material_help", "material_harm", "tail_help", "tail_harm", "ppo_only_recovery", "challenger_only_recovery")
                support.append(dict(cohort=cohort, stratum=stratum, method=method, windows=len(group), eligible=len(eligible),
                    sources=len(values), switches=len(take), rank_windows=len(common), good_rank_sources=good_groups, bad_rank_sources=bad_groups,
                    rank_auc=auc, **{f"{k}_available": sum(r[k] for r in group) for k in flags},
                    **{f"{k}_switched": sum(r[k] and r["case"] in take for r in group) for k in flags}))
                for i, metric in enumerate(("followup_mean_iou", "low_iou_fraction")):
                    paired.append(dict(s86.contrast(values[:, i], method + "_minus_ppo", metric, design),
                                       cohort=cohort, stratum=stratum, windows=len(group)))
    return support, paired


def run(root, plan, source, actor, root89, plan89, windows, observer):
    by_unit = defaultdict(list)
    for r in windows:
        by_unit[r["case"].rsplit("__f", 1)[0]].append(r)
    old = source["original"]
    ap = e74.e67.actor_plan(old)
    evidence, scored = {}, []
    for record in plan89["units"]:
        unit = record["unit"]
        key = c.key(unit)
        if key not in by_unit:
            continue
        if record["cohort"] == "expanded":
            bound = c.c66.sealed(root89 / "entries" / (unit["sequence"] + ".json"), s89.SCHEMA)
            if bound["plan_fingerprint"] != plan89["fingerprint"] or bound["entry"]["sequence"] != unit["sequence"]:
                raise ValueError("Stage-89 expanded entry changed")
            entry = bound["entry"]
        else:
            entry, trace = s86.previous.read_source_trace(source, dict(unit=unit, frames=[1], expected_first_failure=None))
        if record["cohort"] == "expanded" or not plan89["reuse_historical"]:
            trace = c.read_cache(root89 / "baseline" / (key + ".npz"),
                                 dict(plan_fingerprint=plan89["fingerprint"], unit=unit))
            if trace is None:
                raise ValueError("Missing Stage-89 baseline trace")
        pack = e74.learned_pack(actor["models"], unit, actor["prior"])

        def choose(row, history):
            return e74.choose(row, history, unit, old, actor["prior"], pack)

        with e63.source_experiment.installed(observer, [], unit) as head:
            if head != source["report"]["native_head_sha256"]:
                raise ValueError("Original SGLA head changed")
            for row in by_unit.pop(key):
                first, action = int(row["first"]), int(row["challenger_action"])
                item = s89.window_item({}, dict(first=first, action=action, signals={}), entry, trace,
                                      ap["design"]["history"], plan["design"]["horizon"])
                fixed = item["arms"][0]
                if int(fixed["selected"][0]) != int(row["ppo_action"]) or s88.challenger(fixed)[0] != action:
                    raise ValueError("Stage-89 action pair changed")
                path = root / "collection" / (row["case"] + ".npz")
                meta = dict(plan_fingerprint=plan["fingerprint"], case=row["case"])
                data = c.read_cache(path, meta)
                if data is None:
                    data = observe(observer, entry, item, ap, choose, plan["design"])
                    c.save(path, data, meta)
                if data["scores"].shape != (2, 2) or data["coverage"].shape != (3,):
                    raise ValueError("Incomplete template scores")
                evidence[path.name] = c.file_hash(path)
                appearance = {}
                for i, name in enumerate(SIGNALS):
                    ppo, alt = [float(x) if np.isfinite(x) else None for x in data["scores"][:, i]]
                    appearance.update({name + "_ppo": ppo, name + "_challenger": alt,
                                       name + "_margin": alt - ppo if ppo is not None and alt is not None else None})
                scored.append(dict(row, **appearance, template_coverage=float(data["coverage"][0]),
                                   ppo_coverage=float(data["coverage"][1]), challenger_coverage=float(data["coverage"][2])))
        print(f"[Stage-91] {key} {len(scored)}/{len(windows)}", flush=True)
    if by_unit or len(scored) != len(windows):
        raise ValueError("Stage-89 window coverage changed")
    publish(root, plan, scored, evidence)


def publish(root, plan, scored, evidence):
    io.verify(plan["source_sha256"])
    io.verify(plan["code_sha256"])
    support, paired = summarize(scored, plan["design"])
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    hashes = {}
    for name, rows in zip(TABLES, (scored, support, paired)):
        path = out / f"stage91_{name}.csv"
        s86.publish_table(path, rows)
        hashes[path.name] = c.file_hash(path)
    calls = len(scored) * 4
    c.publish_identical(out / "stage91_decision.json", io.seal(dict(schema=SCHEMA, status=STATUS,
        plan_fingerprint=plan["fingerprint"], source_stage89_fingerprint=plan["source_stage89_fingerprint"],
        runtime=plan["runtime"], windows=len(scored), search_calls=calls, block_calls=calls * 7,
        training_updates=0, new_continuations=0, confirmation_inference_calls=0,
        cache_sha256=evidence, table_sha256=hashes)))
    print(f"Stage-91: {STATUS} ({len(scored)} windows, {calls} diagnostic searches)", flush=True)


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    report = c.c66.sealed(root / "analysis/stage91_decision.json", SCHEMA)
    if (report["status"] != STATUS or report["plan_fingerprint"] != plan["fingerprint"]
            or report["windows"] != len(plan["cases"]) or report["search_calls"] != 4 * len(plan["cases"])
            or report["block_calls"] != 7 * report["search_calls"]
            or set(report["cache_sha256"]) != {f"{case}.npz" for case in plan["cases"]}
            or set(report["table_sha256"]) != {f"stage91_{name}.csv" for name in TABLES}):
        raise ValueError("Incomplete Stage-91 result")
    for folder, field in (("analysis", "table_sha256"), ("collection", "cache_sha256")):
        io.verify({str(root / folder / name): sha for name, sha in report[field].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("stage76", "stage89", "output_dir"):
        parser.add_argument("--" + flag, type=Path, required=flag == "output_dir")
    parser.add_argument("--bundle", type=Path, default=s90.DEFAULT_INPUT)
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage91")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not all((args.stage76, args.stage89, args.checkpoint)):
        parser.error("Supply Stage-76, Stage-89 and the original checkpoint")
    root76, root89 = (p.expanduser().resolve() for p in (args.stage76, args.stage89))
    if any(root == p or root in p.parents or p in root.parents for p in (root76, root89)):
        parser.error("Use a separate Stage-91 output directory")
    import torch

    torch.set_num_threads(1)
    with writer_lock(root):
        if (root / "analysis/stage91_decision.json").exists():
            check_results(root)
            return
        if not (root / "manifest.json").exists() and any(p.name != "collect.lock" for p in root.iterdir()):
            raise ValueError("Use an empty Stage-91 directory")
        windows, report89, bundle_sha = s90.read_bundle(args.bundle)
        s89.check_results(root89)
        plan89 = c.c66.sealed(root89 / "manifest.json", s89.SCHEMA)
        if c.c66.sealed(root89 / "analysis/stage89_decision.json", s89.SCHEMA) != report89:
            raise ValueError("Stage-89 run differs from the archived bundle")
        source = s86.previous.load_source(root76)
        if plan89["source_stage76_fingerprint"] != source["report"]["fingerprint"]:
            raise ValueError("Stage-76 lineage differs from Stage-89")
        actor = io.load_source(Path(source["plan"]["source_input"]))
        if actor["plan"] != source["original"] or actor["models"]["fingerprint"] != source["report"]["source_training_fingerprint"]:
            raise ValueError("Stage-74 actor changed")
        runtime = s76.device_runtime(source["original"], torch, args.gpu)
        differences = s76.startup.differences(report89["runtime"], runtime)
        if any(r["field"] not in ("runtime.gpu_uuid", "runtime.tracker.device") for r in differences):
            raise ValueError("Frozen SGLA runtime software or weights changed")
        code = [Path(__file__), Path(s90.__file__), Path(s89.__file__), Path(e63.__file__),
                Path(roi_weights.__code__.co_filename), Path(__file__).with_name("run_stage91_template_appearance.sh")]
        plan = io.seal(dict(schema=SCHEMA, design=DESIGN, runtime=runtime, cases=[r["case"] for r in windows],
            source_stage89_fingerprint=report89["fingerprint"],
            source_sha256={str(args.bundle.resolve()): bundle_sha, str(root89 / "manifest.json"): c.file_hash(root89 / "manifest.json")},
            code_sha256={str(p.resolve()): c.file_hash(p) for p in code}))
        c.publish_identical(root / "manifest.json", plan)
        frozen = dict(source["original"], fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = s76.startup.checked_runtime(args, frozen, root)
        observer = e63.Capture(native)
        try:
            run(root, plan, source, actor, root89, plan89, s90.decisions(windows, DESIGN), observer)
        finally:
            observer.handle.remove()


if __name__ == "__main__":
    main()
