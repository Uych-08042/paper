"""Separate CENTER-map changes from coordinate mapping at fixed crop size and layer."""

import argparse
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
import stage183_search_feedback as source
from stage27_uav_common import crop_geometry
from stage28_response_common import dense_boxes


base, gain, c, io = source.base, source.gain, source.c, source.io
online, previous, risk = source.online, source.previous, source.risk
LAYERS, GOOD, LONG = source.LAYERS, source.GOOD, source.LONG
SOURCE183 = "2879714ac87a3a7b5cc0e0f750b02026469561fe4d591342c5fbbc33a027bed3"
SCHEMA = "stage184_center_mapping_v1"
SOURCES = (76, 89, 93, 121, 170, 182, 183)
REFERENCES = {182: source.REFERENCES[182], 183: (source.SCHEMA, SOURCE183)}
ARMS = ("current", "mapping_only", "maps_only", "full_center")
FACTORS = ((0, 0), (0, 1), (1, 0), (1, 1))
PRIMARY = 9
MAP_KEYS = ("raw", "size", "offset")


def bind_roster(read):
    cases = {r["id"]: r for r in read(183, "cases")}
    points = read(183, "points")
    controls = [r for r in read(182, "cases") if r["cohort"] == "stable"]
    if (len(cases) != 19 or len(points) != 38 or len(controls) != 9
            or len({(r["event_id"], r["phase"]) for r in points}) != len(points)):
        raise ValueError("Expected the sealed Stage-183 point roster and Stage-182 controls")
    result = []
    for p in points:
        if p["phase"] != "after":
            continue
        event = cases[p["event_id"]]
        result.append(dict(id=p["id"], sequence=p["sequence"], group=p["group"], cohort="supply_loss",
                           frame=int(p["frame"]), end=int(p["end"]), earlier=int(p["earlier_parent_frame"]),
                           prior_event=event["stage176_source_seen"] == "True" or event["stage182_event_seen"] == "True"))
    for p in controls:
        frame = int(p["frame"])
        result.append(dict(id=p["id"], sequence=p["sequence"], group=p["group"], cohort="stable",
                           frame=frame, end=int(p["end"]), earlier=frame - 2, prior_event=False))
    if len({r["id"] for r in result}) != len(result):
        raise ValueError("Duplicate Stage-184 point")
    return result


def check_control(case, truth):
    values = truth[case["frame"] - 1:case["end"]]
    return len(values) == LONG + 1 and bool((values >= 0.).all()) and values[0] >= GOOD and not online.failure_events(values)


def decompose(data, truth, window, search_size, search_factor):
    """Cross independently cached crop maps and floating decoding centers."""
    np.testing.assert_array_equal(data["counts"], np.tile([1] * 13 + [6], (2, 1)))
    np.testing.assert_array_equal(data["parent"][0, 2:], data["parent"][1, 2:])
    np.testing.assert_equal(data["factor"][0], data["factor"][1])
    np.testing.assert_array_equal(data["image_hw"][0], data["image_hw"][1])
    hw, factor = data["image_hw"][0], float(data["factor"][0])
    crops = [crop_geometry(p, hw, search_factor, search_size) for p in data["parent"]]
    np.testing.assert_allclose([geo[2] for geo in crops], [factor, factor], rtol=0., atol=1e-12)
    same_crop = np.array_equal(crops[0][0], crops[1][0])
    shift = data["parent"][1, :2] - data["parent"][0, :2]
    box_sets, reports = [], []
    for i, layer in enumerate(LAYERS):
        differences = {k: float(np.abs(data[k][0, i] - data[k][1, i]).max()) for k in MAP_KEYS}
        map_results = []
        for arm, (maps, mapping) in zip(ARMS, FACTORS, strict=True):
            raw, size, offset = (data[key][maps, i] for key in MAP_KEYS)
            cell = int(np.argmax(raw * window))
            grid = dense_boxes(size, offset, data["parent"][mapping], factor, search_size, hw)
            if maps == mapping:
                np.testing.assert_allclose(grid[cell], data["boxes"][maps, i], atol=source.BOX_ATOL, rtol=0.)
                grid[cell] = data["boxes"][maps, i]
            if (np.array_equal(data["parent"][0], data["parent"][1])
                    and not any(differences.values())):
                grid[cell] = data["boxes"][0, i]
            box = grid[cell]
            iou = online.tracking_iou(grid, truth)
            if (iou < 0.).any() or not np.isfinite(raw).all():
                raise ValueError("Invalid crossed CENTER box or response")
            map_results.append(box)
            reports.append(dict(layer=layer, arm=arm, crop_maps=maps, mapping_center=mapping, cell=cell,
                                output_iou=float(iou[cell]), dense_iou=float(iou.max()),
                                hann_peak=float((raw * window).flat[cell]), same_rounded_crop=same_crop,
                                same_crop_pixels=bool(data["crop_sha256"][0] == data["crop_sha256"][1]),
                                center_shift_x=float(shift[0]), center_shift_y=float(shift[1]),
                                box_x=float(box[0]), box_y=float(box[1]), box_w=float(box[2]), box_h=float(box[3]),
                                **{f"max_{k}_map_delta": d for k, d in differences.items()}))
        box_sets.append(map_results)
    return np.asarray(box_sets), reports


def factor_effects(rows, fields):
    arms = {r["arm"]: r for r in rows}
    effects = {}
    for metric in fields:
        baseline, mapping, changed_maps, full = [arms[arm][metric] for arm in ARMS]
        values = (None,) * 4
        if None not in (baseline, mapping, changed_maps, full):
            values = (mapping - baseline, changed_maps - baseline, full - baseline,
                      full - mapping - changed_maps + baseline)
        effects.update({f"{metric}_{name}": val for name, val in zip(("mapping", "maps", "full", "interaction"), values, strict=True)})
    return effects


def cache_reference(root, receipt, relative, metadata):
    path = root / relative
    io.verify({str(p): receipt["artifact_sha256"][p.relative_to(root).as_posix()]
               for p in (path, path.with_suffix(".receipt.json"))})
    data = c.read_cache(path, metadata)
    if data is None:
        raise ValueError(f"Missing sealed Stage-183 artifact: {path}")
    return data


def record_artifacts(root, path, report):
    for f in (path, path.with_suffix(".receipt.json")):
        report[f.relative_to(root).as_posix()] = c.file_hash(f)


def stage183_cache(root, receipt, case, kind):
    key = case["id"]
    return cache_reference(root, receipt, f"{kind}/{key}.npz",
                           dict(plan_fingerprint=receipt["plan_fingerprint"], kind=kind, key=key))


def run_paths(anchors, native_box, start, end, archive, recorded, compute, progress, cached=None):
    """Use exact cached trajectories for identical committed boxes."""
    count = len(anchors) * len(ARMS)
    wanted = anchors.reshape(count, 4)
    unique, inverse = np.unique(wanted, axis=0, return_inverse=True)
    known = {}
    if archive is not None:
        known[tuple(native_box)] = archive["boxes"][start - 1:end]
        if recorded is not None:
            known.update({tuple(path[0]): path for path in recorded["paths"]})
    missing = [i for i, box in enumerate(unique) if tuple(box) not in known]
    logical, physical = count * (end - start), len(missing) * (end - start)
    if cached is None:
        paths = np.empty((len(unique), end - start + 1, 4), float)
        for i, box in enumerate(unique):
            if tuple(box) in known:
                paths[i] = known[tuple(box)]
        if missing:
            paths[missing] = compute(unique[missing])
        progress.reused += (len(unique) - len(missing)) * (end - start)
        progress.advance(count=logical - physical)
        saved = dict(paths=paths, inverse=inverse, missing=np.asarray(missing, int))
    else:
        saved = cached
        progress.reused += logical
        progress.advance(count=logical)
    np.testing.assert_array_equal(saved["inverse"], inverse)
    np.testing.assert_array_equal(saved["missing"], missing)
    np.testing.assert_array_equal(saved["paths"][:, 0], unique)
    if saved["paths"].shape != (len(unique), end - start + 1, 4) or not np.isfinite(saved["paths"]).all():
        raise ValueError("Invalid continuation from the crossed CENTER output")
    for i, box in enumerate(unique):
        if tuple(box) in known:
            np.testing.assert_array_equal(saved["paths"][i], known[tuple(box)])
    return saved, saved["paths"][inverse], logical, physical


def summarize(cases, effects, continuations):
    by_case = {r["id"]: r for r in cases}
    summary = []
    for cohort in ("supply_loss", "stable", "supply_loss_without_previous_event"):
        for layer in (PRIMARY,) if cohort == "stable" else LAYERS:
            for arm in ARMS[1:]:
                selected = [r for r in continuations if r["layer"] == layer and r["arm"] == arm and r["valid"] and (
                    by_case[r["case_id"]]["cohort"] == cohort or
                    cohort == "supply_loss_without_previous_event" and by_case[r["case_id"]]["cohort"] == "supply_loss"
                    and not by_case[r["case_id"]]["prior_event"])]
                for length in ("all", "full128"):
                    rows = [r for r in selected if length == "all" or r["successors"] == LONG]
                    values = defaultdict(list)
                    for r in rows:
                        values[by_case[r["case_id"]]["group"]].append(r["delta128_or_shorter"])
                    gains = [float(np.mean(v)) for v in values.values()]
                    summary.append(dict(cohort=cohort, layer=layer, arm=arm, length=length, cases=len(rows),
                                        groups=len(values), mean_group_tail_delta=float(np.mean(gains)) if gains else None,
                                        tail_delta_without_best_group=float(np.mean(sorted(gains)[:-1])) if len(gains) > 1 else None,
                                        rescue_groups=len({r["group"] for r in rows if r["protected_rescue"]}),
                                        harm_groups=len({r["group"] for r in rows if r["event_harm"]})))
    front = []
    for cohort in ("supply_loss", "stable"):
        for layer in LAYERS:
            rows = [r for r in effects if r["layer"] == layer and r["cohort"] == cohort]
            m = ("output_iou_mapping", "output_iou_maps", "output_iou_full", "output_iou_interaction")
            front.append(dict(cohort=cohort, layer=layer, cases=len(rows),
                              **{f"mean_{k}": float(np.mean([r[k] for r in rows])) if rows else None for k in m},
                              **{f"restored_good_{k}": sum(r[f"restores_good_{k}"] for r in rows) for k in ("mapping", "maps", "full")},
                              **{f"lost_good_{k}": sum(r[f"loses_good_{k}"] for r in rows) for k in ("mapping", "maps", "full")}))
    return front, summary


def instrument(root, r183, d183, plan, cases, entries, archives, archive_iou, tracker, image_root, progress):
    rt = tracker.rt
    window = rt.tracker.output_window.detach().float().cpu().numpy().reshape(rt.search_side, rt.search_side)
    size, factor = int(rt.tracker.params.search_size), float(rt.tracker.params.search_factor)
    exact = not plan["runtime_differences"]
    table = dict(anchors=[], effects=[], tails=[], tail_effects=[])
    artifacts, recorded_crops, physical_crops, logical_calls, physical_calls = {}, 0, 0, 0, 0

    def read(seq, f):
        return online.checked_image(image_root, entries[seq]["frames"][f])

    with ExitStack() as stack:
        observer = online.RoutedObserver(tracker, stack)
        for case in cases:
            key, seq, start, end = (case[k] for k in ("id", "sequence", "frame", "end"))
            entry, archive = entries[seq], archives[seq]
            truth = np.asarray(entry["ground_truth"], float)
            original = archive["parent"][start - 1]
            earlier = archive["parent"][case["earlier"]]
            parents = source.cross.cross_parents(original, earlier)[:2]
            template = None

            def load_template():
                nonlocal template
                if template is None:
                    template = read(seq, 0)
                return template

            maps_path = root / "maps" / f"{key}.npz"
            maps_meta = dict(plan_fingerprint=plan["fingerprint"], kind="maps", case_id=key)
            source_maps = exact and case["cohort"] == "supply_loss"
            if source_maps:
                old = stage183_cache(r183, d183, case, "cross")
                data = {k: v[:2] for k, v in old.items()}
                recorded_crops += 2
                progress.reused += 2
            else:
                data = c.read_cache(maps_path, maps_meta)
                if data is None:
                    observer.initialize(load_template(), entry["initial_box"], "native", None)
                    image = read(seq, start)
                    first = source.probe.collect(rt, image, parents[0], start)
                    second = first if np.array_equal(*parents) else source.probe.collect(rt, image, parents[1], start)
                    data = {k: np.stack((v, second[k])) for k, v in first.items()}
                    c.save(maps_path, data, maps_meta)
                else:
                    progress.reused += 2
                record_artifacts(root, maps_path, artifacts)
                physical_crops += 1 if np.array_equal(*parents) else 2
            progress.advance(count=2)
            np.testing.assert_array_equal(data["parent"], parents)
            anchors, scores = decompose(data, truth[start], window, size, factor)
            native = int(data["selected"][0])
            if exact:
                n = LAYERS.index(native)
                np.testing.assert_equal(native, archive["layer"][start - 1])
                np.testing.assert_allclose(anchors[n, 0], archive["boxes"][start - 1], atol=source.BOX_ATOL, rtol=0.)
                same = (anchors == anchors[n, 0]).all(axis=-1)
                anchors[same] = archive["boxes"][start - 1]
            table["anchors"].extend(dict(case_id=key, cohort=case["cohort"], group=case["group"], **r) for r in scores)
            for layer in LAYERS:
                local = [r for r in scores if r["layer"] == layer]
                baseline = local[0]["output_iou"]
                table["effects"].append(dict(case_id=key, cohort=case["cohort"], group=case["group"], layer=layer,
                    **factor_effects(local, ("output_iou", "dense_iou")),
                    **{f"restores_good_{tag}": baseline < GOOD <= row["output_iou"]
                       for tag, row in zip(("mapping", "maps", "full"), local[1:], strict=True)},
                    **{f"loses_good_{tag}": row["output_iou"] < GOOD <= baseline
                       for tag, row in zip(("mapping", "maps", "full"), local[1:], strict=True)}))

            layers = LAYERS if case["cohort"] == "supply_loss" else (PRIMARY,)
            branches = anchors[[LAYERS.index(k) for k in layers]]
            recorded = stage183_cache(r183, d183, case, "tail") if source_maps else None
            path = root / "tail" / f"{key}.npz"
            meta = dict(plan_fingerprint=plan["fingerprint"], kind="tail", case_id=key)
            cached = c.read_cache(path, meta)

            def compute(unknown):
                return source.continue_from(observer, lambda f: read(seq, f), load_template(), entry["initial_box"],
                                            start, end, unknown, progress)

            saved, paths, logical, physical = run_paths(branches, anchors[LAYERS.index(native), 0], start, end,
                archive if exact else None, recorded, compute, progress, cached=cached)
            if cached is None:
                c.save(path, saved, meta)
            record_artifacts(root, path, artifacts)
            logical_calls += logical
            physical_calls += physical
            for j, layer in enumerate(layers):
                first = j * len(ARMS)
                indices = range(first, first + len(ARMS))
                tails = source.tail_summary(paths, truth[start:end + 1], archive_iou[seq][:start - 1], first, indices)
                results = [dict(case_id=key, group=case["group"], cohort=case["cohort"], arm=arm, layer=layer,
                                successors=end - start, **row) for arm, row in zip(ARMS, tails, strict=True)]
                table["tails"].extend(results)
                table["tail_effects"].append(dict(case_id=key, group=case["group"], cohort=case["cohort"], layer=layer,
                        successors=end - start, **factor_effects(results, ("delta32_or_shorter", "delta128_or_shorter"))))
    return table, artifacts, dict(reused_source_crop_observations=recorded_crops, required_crop_observations=physical_crops,
                                   logical_native_successor_calls=logical_calls, required_native_successor_calls=physical_calls)


def run(args, root, paths):
    r76, r89, r93, r121, r170, r182, r183 = paths
    p170 = c.c66.sealed(r170 / "manifest.json", previous.SCHEMA)
    d170 = c.c66.sealed(r170 / "stage170_decision.json", previous.SCHEMA)
    p183 = c.c66.sealed(r183 / "manifest.json", source.SCHEMA)
    d182 = c.c66.sealed(r182 / "stage182_decision.json", source.prior.SCHEMA)
    d183 = c.c66.sealed(r183 / "stage183_decision.json", source.SCHEMA)
    if (d170["fingerprint"] != source.event.matched.SOURCE170 or d170["plan_fingerprint"] != p170["fingerprint"]
            or d170["runtime"] != p170["runtime"] or d182["fingerprint"] != source.SOURCE182
            or d183["source_stage182"] != source.SOURCE182 or d183["fingerprint"] != SOURCE183
            or d183["plan_fingerprint"] != p183["fingerprint"] or d183["runtime"] != d170["runtime"]
            or d182["runtime"] != d170["runtime"] or d183["runtime_differences"]):
        raise ValueError("Need the sealed same-runtime Stage-170/182/183 sources")
    roots, receipts = {182: r182, 183: r183}, {182: d182, 183: d183}
    proposed = bind_roster(lambda s, k: risk.read_verified(roots[s], receipts[s], s, k))
    _, _, jobs = source.prior.history.native_jobs(r121)
    s95 = base.s103.prior.s95
    p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
    old = origin["original"]
    entries, archives, archive_iou = {}, {}, {}
    for seq in sorted({r["sequence"] for r in proposed}):
        entry = s95.load_unit(r93, r89, p93, p89, d89, origin, jobs[seq]["spec"])[0]
        frames = jobs[seq]["frames"]
        if (entry["sequence"] != seq or len(entry["ground_truth"]) != frames + 1 or len(entry["frames"]) != frames + 1
                or any(not 0 <= r["earlier"] < r["frame"] - 1 < r["end"] <= frames
                       for r in proposed if r["sequence"] == seq)):
            raise ValueError(f"Insufficient native history or unexpected horizon at {seq}")
        trace = previous.archive_trace(r170, p170, d170, seq, "native")
        previous.audit(trace, frames, entry["initial_box"], "native", 0.)
        entries[seq], archives[seq] = entry, trace
        archive_iou[seq] = online.tracking_iou(trace["boxes"], entry["ground_truth"][1:])
    support = [dict(case, selected=case["cohort"] == "supply_loss" or check_control(case, archive_iou[case["sequence"]]))
               for case in proposed]
    cases = [case for case in support if case["selected"]]

    import torch

    torch.set_num_threads(1)
    runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
    differences = base.s109.check_runtime(d183["runtime"], runtime)
    code = (Path(__file__), Path(source.__file__), Path(source.probe.__file__), Path(source.cross.__file__),
            Path(dense_boxes.__code__.co_filename))
    plan = io.seal(dict(schema=SCHEMA, source_stage183=SOURCE183, source_stage182=source.SOURCE182,
                        cases=cases, support=support, layers=list(LAYERS), arms=list(ARMS), stage170=d170["fingerprint"],
                        primary_layer=PRIMARY, runtime=runtime, runtime_differences=differences,
                        code_sha256={p.name: c.file_hash(p) for p in code}))
    c.publish_identical(root / "manifest.json", plan)
    for name in ("maps", "tail"):
        (root / name).mkdir(exist_ok=True)
    work = sum(2 + len(ARMS) * (len(LAYERS) if r["cohort"] == "supply_loss" else 1) * (r["end"] - r["frame"])
               for r in cases) + 1
    progress = OverallProgress("Stage-184", work, interval=args.progress_interval, unit="work")
    progress.report("Loading frozen SGLA")
    args.parameter = runtime["tracker"]["parameter"]
    tracker, _ = base.s103.s76.startup.checked_runtime(args, dict(old, fingerprint=plan["fingerprint"],
                 runtime_identity=runtime, runtime=runtime["tracker"]), root)
    progress.report("Crossing CENTER maps and mapping centers, then continuing native tracking")
    tables, artifacts, counts = instrument(root, r183, d183, plan, cases, entries, archives, archive_iou,
                                          tracker, old["root"], progress)
    tracker.assert_frozen()
    tables["anchor_summary"], tables["tail_summary"] = summarize(cases, tables["effects"], tables["tails"])
    tables["cases"], tables["support"] = cases, support
    if progress.done != work - 1:
        raise ValueError("Incomplete Stage-184 progress accounting")
    return dict(source_stage183=SOURCE183, plan_fingerprint=plan["fingerprint"], points=len(cases), primary_layer=PRIMARY,
                training_performed=False, new_full_video_policy_evaluated=False,
                stable_controls=sum(r["cohort"] == "stable" for r in cases), work=work,
                runtime=runtime, runtime_differences=differences, artifact_sha256=artifacts,
                table_sha256=gain.prior.publish(root, tables, stage="stage184"), **counts), progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", nargs=2, type=Path, metavar=("STAGE182", "STAGE183"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--progress-interval", type=float, default=15)
    parser.set_defaults(mode="stage184_center_mapping")
    for stage in SOURCES:
        parser.add_argument(f"--stage{stage}-root", type=Path)
    args = parser.parse_args()
    paths = [p.expanduser().resolve() for stage in SOURCES if (p := getattr(args, f"stage{stage}_root"))]
    if args.bundles:
        if paths or args.output_dir or args.checkpoint:
            parser.error("--bundles is a standalone CPU check of the crossed-point roster")
        tables = {s: gain.prior.archives.bundle(p, s, fp)[1]
                  for p, (s, (_, fp)) in zip(args.bundles, REFERENCES.items(), strict=True)}
        cases = bind_roster(lambda s, k: tables[s][f"stage{s}_{k}.csv"])
        print(f"Verified {sum(r['cohort'] == 'supply_loss' for r in cases)} supply-loss points and "
              f"{sum(r['cohort'] == 'stable' for r in cases)} potential stable controls")
        return
    if len(paths) != len(SOURCES) or args.output_dir is None or args.checkpoint is None or args.progress_interval <= 0:
        parser.error("Need all source directories, original checkpoint and output directory")
    root = args.output_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate output directory")
    with gain.prior.prior.writer_lock(root):
        out = root / "stage184_decision.json"
        if out.exists():
            saved = c.c66.sealed(out, SCHEMA)
            plan = c.c66.sealed(root / "manifest.json", SCHEMA)
            if saved["source_stage183"] != SOURCE183 or saved["plan_fingerprint"] != plan["fingerprint"]:
                raise ValueError("Stage-184 completed run has another source or plan")
            io.verify({str(root / k): h for k, h in (saved["artifact_sha256"] | saved["table_sha256"]).items()})
            OverallProgress("Stage-184", saved["work"], saved["work"]).report("Complete")
            return
        result, progress = run(args, root, paths)
        c.publish_identical(out, io.seal(dict(schema=SCHEMA, status="COMPLETE_STAGE184_CENTER_MAPPING_DIAGNOSTIC", **result)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
