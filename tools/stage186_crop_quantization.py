"""Separate floating center motion and integer crop changes on matched successor frames."""

import argparse
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
import stage185_writeback_components as prior
from stage27_uav_common import crop_geometry, scene_geometry
from stage28_response_common import dense_boxes


center, source, online, previous = prior.center, prior.source, prior.online, prior.previous
base, gain, c, io, risk = prior.base, prior.gain, prior.c, prior.io, prior.risk
SOURCE185 = "7101aaae1c4b38f14094df8f752546e9a55adb6d92f165cb3c68bad772a44027"
SCHEMA = "stage186_crop_quantization_v1"
SOURCES = (76, 89, 93, 121, 170, 185)
ARMS = ("native", "current", "within_crop", "horizontal", "vertical", "observed", "opposite_center", "opposite_pixels")
SCALE_POINT = "uav0000348_02415_s__failure000367__after"
MAP_KEYS = ("raw", "size", "offset")


def bind_roster(read):
    cases = [dict(r, frame=int(r["frame"]), end=int(r["end"]), earlier=int(r["earlier"]),
                  prior_event=r["prior_event"] == "True", selected=r["selected"] == "True") for r in read("cases")]
    windows = {(r["case_id"], r["arm"]): r for r in read("windows")}
    timing = {(r["case_id"], r["arm"]): r for r in read("timing")}
    response = {(r["case_id"], r["arm"], int(r["frame"])): r for r in read("response")}
    expected = {(r["id"], a) for r in cases for a in prior.ARMS}
    if (len(cases) != 27 or sum(r["cohort"] == "stable" for r in cases) != 8 or
            set(windows) != expected or set(timing) != expected or not any(r["id"] == SCALE_POINT for r in cases)):
        raise ValueError("Need the sealed Stage-185 windows and quantized crops")
    return cases, timing, response


def crop_arms(previous, case_id, hw, factor, size):
    original, moved = previous[[1, 2], 0]
    np.testing.assert_array_equal(original[2:], moved[2:])
    def rectangle(box):
        return crop_geometry(box, hw, factor, size)[0]

    baseline = rectangle(original)
    shift = moved[:2] - original[:2]
    pixel_shift = rectangle(moved)[:2] - baseline[:2]
    unrounded = original[:2] + original[2:] / 2. - baseline[2] / 2.
    within = original.copy()
    if np.any(pixel_shift):
        # Move along the observed direction, halfway to the first crossed boundary.
        changes = pixel_shift != 0.
        boundary = baseline[:2] + .5 * np.sign(shift)
        fraction = float(np.clip(np.min((boundary[changes] - unrounded[changes]) / shift[changes]), 0., 1.))
        within[:2] += .5 * fraction * shift
    else:
        within = moved.copy()
    horizontal, vertical, mirrored, reverse_pixels = (original.copy() for _ in range(4))
    horizontal[0], vertical[1] = moved[0], moved[1]
    mirrored[:2] -= shift
    reverse_pixels[:2] -= pixel_shift
    # At an exact half-pixel, ties-to-even need not commute with integer translation.
    desired = baseline[:2] - pixel_shift
    tied = rectangle(reverse_pixels)[:2] != desired
    reverse_pixels[:2][tied] = (desired + baseline[2] / 2. - original[2:] / 2.)[tied]
    boxes = np.stack((previous[0, 0], original, within, horizontal, vertical, moved, mirrored, reverse_pixels))
    crops = np.stack([rectangle(box) for box in boxes])
    np.testing.assert_array_equal(crops[2], baseline)
    np.testing.assert_array_equal(crops[3, :2] - baseline[:2], [pixel_shift[0], 0.])
    np.testing.assert_array_equal(crops[4, :2] - baseline[:2], [0., pixel_shift[1]])
    np.testing.assert_array_equal(crops[7, :2], desired)
    np.testing.assert_array_equal(crops[1:, 2:], np.tile(baseline[2:], (len(ARMS) - 1, 1)))
    arms = ARMS
    if case_id == SCALE_POINT:
        arms = (*ARMS, "size_control")
        boxes = np.vstack((boxes, previous[3, 0]))
        crops = np.vstack((crops, rectangle(previous[3, 0])))
    return arms, boxes, crops


class FixedCenterObserver(center.CenterObserver):
    def route(self, module, args, output):
        super().route(module, args, output)
        self.chosen = prior.LAYER
        self.switched = self.chosen != self.original
        if self.switched:
            forced = self.rt.torch.zeros_like(output)
            forced[:, source.LAYERS.index(self.chosen)] = 1
            return forced
        return None


def observe_crops(observer, template, initial, image, parents, frame):
    observer.initialize(template, initial, "native", None)
    cached = {}
    rows = []
    for p in parents:
        key = tuple(p)
        if key not in cached:
            observer.maps = None
            observer.rt.tracker.frame_id = frame - 1
            row = observer.step(image, p, frame)
            if observer.maps is None or row["layer"] != prior.LAYER or row["crop_calls"] != 1 or row["head_calls"] != 1:
                raise ValueError("Expected one crop and one fixed-layer CENTER output")
            cached[key] = dict(row, **observer.maps)
        rows.append(cached[key])
    return dict(calls=np.asarray(len(cached)), **{k: np.stack([row[k] for row in rows])
                for k in ("boxes", "parent", "layer", "hann_peak", *MAP_KEYS)})


def score_probe(data, arms, crops, target, hw, size, factor, window):
    reference = ARMS.index("current")
    baseline = crops[reference]
    ref_parent = data["parent"][reference]
    target_center = target[:2] + target[2:] / 2.
    rows = []
    for i, arm in enumerate(arms):
        raw, scale, offset = (data[k][i] for k in MAP_KEYS)
        if not np.isfinite(raw).all() or not np.isfinite(target).all() or (target[2:] <= 0.).any():
            raise ValueError("Invalid successor response or evaluation target")
        parent = data["parent"][i]
        geo = scene_geometry(parent, target, hw, factor, size)
        magnification = geo["factor"]
        np.testing.assert_array_equal(geo["crop"], crops[i])
        peak = int(np.argmax(raw * window))
        grid = dense_boxes(scale, offset, parent, magnification, size, hw)
        np.testing.assert_allclose(grid[peak], data["boxes"][i], atol=source.BOX_ATOL, rtol=0.)
        grid[peak] = data["boxes"][i]
        iou = online.tracking_iou(grid, target)
        deltas = {k: float(np.max(np.abs(data[k][i] - data[k][reference]))) for k in MAP_KEYS}
        same = np.array_equal(crops[i], baseline)
        if same and max(deltas.values()) > prior.MAP_ATOL:
            raise ValueError("Same rounded crop changed the fixed-layer CENTER maps")
        common_iou = None
        if crops[i, 2] == baseline[2]:
            common_grid = dense_boxes(scale, offset, ref_parent, magnification, size, hw)
            common_iou = float(online.tracking_iou(common_grid[peak], target))
        pix = crops[i, :2] - baseline[:2]
        rows.append(dict(arm=arm, same_rounded_crop=same, dx=float(parent[0] - ref_parent[0]),
                         dy=float(parent[1] - ref_parent[1]), pixel_dx=int(pix[0]), pixel_dy=int(pix[1]),
                         crop_left=int(crops[i, 0]), crop_top=int(crops[i, 1]), crop_side=int(crops[i, 2]),
                         target_center_distance=float(np.linalg.norm(crops[i, :2] + crops[i, 2:] / 2. - target_center)),
                         coverage=float(geo["coverage"]), input_short_side_px=float(min(geo["input_wh"])),
                         selected_cell=peak, hann_peak=float(data["hann_peak"][i]),
                         best_grid_iou=float(iou.max()), output_iou=float(iou[peak]),
                         output_iou_fixed_mapping=common_iou, **{f"{k}_map_delta": v for k, v in deltas.items()}))
    return rows


def continue_paths(anchors, known, successors, compute, progress, cached=None):
    unique, index = np.unique(anchors, axis=0, return_inverse=True)
    lookup = {tuple(p[0]): p for p in known}
    missing = [i for i, p in enumerate(unique) if tuple(p) not in lookup]
    work, required = len(anchors) * successors, len(missing) * successors
    if cached is None:
        paths = np.empty((len(unique), successors + 1, 4), float)
        paths[:, 0] = unique
        if missing and successors:
            paths[missing] = compute(unique[missing])
        for i, box in enumerate(unique):
            if tuple(box) in lookup:
                paths[i] = lookup[tuple(box)]
        progress.reused += (len(unique) - len(missing)) * successors
        progress.advance(count=work - required)
        recorded = dict(paths=paths, inverse=index, missing=np.asarray(missing, int))
    else:
        recorded = cached
        progress.reused += work
        progress.advance(count=work)
    paths = recorded["paths"]
    np.testing.assert_array_equal(recorded["inverse"], index)
    np.testing.assert_array_equal(recorded["missing"], missing)
    np.testing.assert_array_equal(paths[:, 0], unique)
    if (paths.shape != (len(unique), successors + 1, 4) or not np.isfinite(paths).all() or (paths[..., 2:] <= 0.).any()):
        raise ValueError("Invalid successor continuation")
    for i, box in enumerate(unique):
        if tuple(box) in lookup:
            np.testing.assert_array_equal(paths[i], lookup[tuple(box)])
    return recorded, paths[index], required


def summarize(cases, probes, windows):
    by_case = defaultdict(dict)
    for r in probes:
        by_case[r["case_id"]][r["arm"]] = r
    by_tail = {(r["case_id"], r["arm"]): r for r in windows}
    pairs = []
    for case in cases:
        arms = by_case[case["id"]]
        current, observed, reverse = (arms[a] for a in ("current", "observed", "opposite_pixels"))
        horizontal, vertical = (arms[a] for a in ("horizontal", "vertical"))
        same = arms["within_crop"]
        observed_tail, reverse_tail = (by_tail[case["id"], a] for a in ("observed", "opposite_pixels"))
        valid = observed_tail["valid"] and reverse_tail["valid"]
        pairs.append(dict(case_id=case["id"], group=case["group"], cohort=case["cohort"],
                         full128=case["end"] - case["frame"] == source.LONG,
                         observed_crosses_boundary=not observed["same_rounded_crop"],
                         horizontal_crosses_boundary=not horizontal["same_rounded_crop"],
                         vertical_crosses_boundary=not vertical["same_rounded_crop"],
                         opposite_center_crosses_boundary=not arms["opposite_center"]["same_rounded_crop"],
                         observed_moves_crop_toward_target=observed["target_center_distance"] < current["target_center_distance"],
                         within_crop_selected_cell_changed=same["selected_cell"] != current["selected_cell"],
                         observed_selected_cell_changed=observed["selected_cell"] != current["selected_cell"],
                         opposite_selected_cell_changed=reverse["selected_cell"] != current["selected_cell"],
                         observed_minus_opposite_iou=observed["output_iou"] - reverse["output_iou"],
                         observed_minus_opposite_fixed_mapping=observed["output_iou_fixed_mapping"] - reverse["output_iou_fixed_mapping"],
                         xy_interaction_fixed_mapping=observed["output_iou_fixed_mapping"] - horizontal["output_iou_fixed_mapping"]
                         - vertical["output_iou_fixed_mapping"] + current["output_iou_fixed_mapping"],
                         observed_minus_opposite_tail=(observed_tail["delta128_or_shorter"] - reverse_tail["delta128_or_shorter"])
                         if valid else None))
    summary = []
    for cohort in ("supply_loss", "stable", "supply_loss_without_previous_event"):
        for scope in ("all", "boundary_crossed", "boundary_crossed_full128"):
            selected = [p for p, r in zip(cases, pairs, strict=True) if (
                        cohort == p["cohort"] or cohort == "supply_loss_without_previous_event"
                        and p["cohort"] == "supply_loss" and not p["prior_event"]) and
                        (scope == "all" or r["observed_crosses_boundary"] and (scope == "boundary_crossed" or r["full128"]))]
            for arm in ARMS[2:]:
                per_group, harm, rescue = defaultdict(list), set(), set()
                ious, matching = [], []
                for p in selected:
                    current, rec = (by_case[p["id"]][a] for a in ("current", arm))
                    r, b = (by_tail[p["id"], a] for a in (arm, "current"))
                    ious.append(rec["output_iou"] - current["output_iou"])
                    matching.append(rec["output_iou_fixed_mapping"] - current["output_iou_fixed_mapping"])
                    if r["valid"]:
                        per_group[p["group"]].append(r["delta128_or_shorter"] - b["delta128_or_shorter"])
                        if r["protected_rescue"]:
                            rescue.add(p["group"])
                        if r["event_harm"]:
                            harm.add(p["group"])
                group_gains = [float(np.mean(v)) for v in per_group.values()]
                summary.append(dict(cohort=cohort, scope=scope, arm=arm, cases=len(selected), groups=len(group_gains),
                                    mean_anchor_delta=float(np.mean(ious)) if ious else None,
                                    mean_anchor_fixed_mapping_delta=float(np.mean(matching)) if matching else None,
                                    mean_tail_delta_from_current=float(np.mean(group_gains)) if group_gains else None,
                                    rescue_groups=len(rescue), harm_groups=len(harm)))
    return pairs, summary


def instrument(root, r185, d185, plan, cases, timing, response, entries, archive_iou, tracker, image_root, progress):
    rt = tracker.rt
    factor, size = float(rt.tracker.params.search_factor), int(rt.tracker.params.search_size)
    window = rt.tracker.output_window.detach().float().cpu().numpy().reshape(rt.search_side, rt.search_side)
    exact = not plan["runtime_differences"]
    artifacts, probes, windows = {}, [], []
    counts = dict(probe_calls=0, logical_successor_calls=0, required_successor_calls=0)

    def read(seq, f):
        return online.checked_image(image_root, entries[seq]["frames"][f])

    for case in cases:
        key, seq, start, end = (case[k] for k in ("id", "sequence", "frame", "end"))
        frame = start + 1
        entry = entries[seq]
        truth = np.asarray(entry["ground_truth"], float)
        data = prior.prior.cache_reference(r185, d185, f"tail/{key}.npz",
            dict(plan_fingerprint=d185["plan_fingerprint"], kind="tail", case_id=key))
        old = data["paths"][data["inverse"]]
        if old.shape != (len(prior.ARMS), end - start + 1, 4):
            raise ValueError("Stage-185 replay shape changed")
        img = entry["frames"][frame]
        hw = int(img["height"]), int(img["width"])
        arms, parents, crops = crop_arms(old, key, hw, factor, size)
        for i, arm in ((0, "native"), (1, "mapping_only"), (5, "full_center_position")):
            r = timing[key, arm]
            np.testing.assert_array_equal(crops[i, :3], [int(r[k]) for k in ("next_crop_left", "next_crop_top", "next_crop_side")])
        template = None

        def get_template():
            nonlocal template
            if template is None:
                template = read(seq, 0)
            return template

        path = root / "probe" / f"{key}.npz"
        meta = dict(plan_fingerprint=plan["fingerprint"], kind="probe", case_id=key)
        data = c.read_cache(path, meta)
        if data is None:
            with ExitStack() as stack:
                observer = FixedCenterObserver(tracker, stack)
                data = observe_crops(observer, get_template(), entry["initial_box"], read(seq, frame), parents, frame)
            c.save(path, data, meta)
        else:
            progress.reused += len(arms)
        np.testing.assert_array_equal(data["parent"], parents)
        np.testing.assert_array_equal(data["layer"], np.full(len(arms), prior.LAYER, int))
        prior.prior.record_artifacts(root, path, artifacts)
        rows = score_probe(data, arms, crops, truth[frame], hw, size, factor, window)
        if exact:
            bindings = [(0, "native"), (1, "mapping_only"), (5, "full_center_position")]
            if key == SCALE_POINT:
                bindings.append((len(arms) - 1, "full_center_size"))
            for i, name in bindings:
                ref = response[key, name, frame]
                if int(ref["selected_layer"]) == prior.LAYER:
                    j = prior.ARMS.index(name)
                    np.testing.assert_array_equal(data["boxes"][i], old[j, 1])
                    np.testing.assert_equal(rows[i]["selected_cell"], int(ref["hann_cell"]))
                    np.testing.assert_equal(rows[i]["hann_peak"], float(ref["hann_peak"]))
        probes.extend(dict(case_id=key, sequence=seq, group=case["group"], cohort=case["cohort"], frame=frame, **r)
                      for r in rows)
        counts["probe_calls"] += int(data["calls"])
        progress.advance(count=len(arms))

        path = root / "tail" / f"{key}.npz"
        meta = dict(plan_fingerprint=plan["fingerprint"], kind="tail", case_id=key)
        cached = c.read_cache(path, meta)

        def compute(anchors):
            with ExitStack() as stack:
                observer = online.RoutedObserver(tracker, stack)
                return source.continue_from(observer, lambda f: read(seq, f), get_template(), entry["initial_box"],
                                            frame, end, anchors, progress)

        saved, traced, required = continue_paths(data["boxes"], old[:, 1:] if exact else (),
                                                  end - frame, compute, progress, cached)
        if cached is None:
            c.save(path, saved, meta)
        prior.prior.record_artifacts(root, path, artifacts)
        logical = len(arms) * (end - frame)
        counts["required_successor_calls"] += required
        counts["logical_successor_calls"] += logical
        paths = np.concatenate((parents[:, None, :], traced), axis=1)
        prefix = archive_iou[seq][:start - 1]
        tail = source.tail_summary(paths, truth[start:end + 1], prefix, 0, range(len(arms)))
        windows.extend(dict(case_id=key, sequence=seq, group=case["group"], cohort=case["cohort"], arm=arm,
                            successors=end - start, **r) for arm, r in zip(arms, tail, strict=True))
    pairs, summary = summarize(cases, probes, windows)
    return dict(probes=probes, windows=windows, pairs=pairs, summary=summary,
                focus=[r for r in probes if r["group"] in {"uav0000091", "uav0000176", "uav0000348"}
                       and r["cohort"] == "supply_loss"]), artifacts, counts


def run(args, root, paths):
    r76, r89, r93, r121, r170, r185 = paths
    p170 = c.c66.sealed(r170 / "manifest.json", previous.SCHEMA)
    d170 = c.c66.sealed(r170 / "stage170_decision.json", previous.SCHEMA)
    p185 = c.c66.sealed(r185 / "manifest.json", prior.SCHEMA)
    d185 = c.c66.sealed(r185 / "stage185_decision.json", prior.SCHEMA)
    if (d170["fingerprint"] != source.event.matched.SOURCE170 or d170["plan_fingerprint"] != p170["fingerprint"]
            or p170["runtime"] != d170["runtime"] or d185["fingerprint"] != SOURCE185
            or d185["source_stage184"] != prior.SOURCE184 or d185["plan_fingerprint"] != p185["fingerprint"]
            or d185["runtime"] != d170["runtime"] or d185["runtime_differences"]):
        raise ValueError("Need the sealed same-runtime Stage-170 and Stage-185 sources")
    cases, timing, response = bind_roster(lambda k: risk.read_verified(r185, d185, 185, k))
    if cases != p185["cases"]:
        raise ValueError("Stage-185 planned and measured cases differ")
    _, _, jobs = source.prior.history.native_jobs(r121)
    s95 = base.s103.prior.s95
    p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
    old = origin["original"]
    entries, archive_iou = {}, {}
    for seq in sorted({r["sequence"] for r in cases}):
        entry = s95.load_unit(r93, r89, p93, p89, d89, origin, jobs[seq]["spec"])[0]
        frames = jobs[seq]["frames"]
        if (entry["sequence"] != seq or len(entry["ground_truth"]) != frames + 1 or len(entry["frames"]) != frames + 1
                or any(not 1 <= r["frame"] < r["end"] <= frames for r in cases if r["sequence"] == seq)):
            raise ValueError(f"Source video or horizon changed at {seq}")
        archive = previous.archive_trace(r170, p170, d170, seq, "native")
        previous.audit(archive, frames, entry["initial_box"], "native", 0.)
        entries[seq] = entry
        archive_iou[seq] = online.tracking_iou(archive["boxes"], entry["ground_truth"][1:])

    import torch

    torch.set_num_threads(1)
    runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
    differences = base.s109.check_runtime(d185["runtime"], runtime)
    code = (Path(__file__), Path(prior.__file__), Path(center.__file__), Path(source.__file__),
            Path(crop_geometry.__code__.co_filename), Path(dense_boxes.__code__.co_filename))
    plan = io.seal(dict(schema=SCHEMA, source_stage185=SOURCE185, source_stage170=d170["fingerprint"], cases=cases,
                        layer=prior.LAYER, arms=list(ARMS), size_control_point=SCALE_POINT,
                        runtime=runtime, runtime_differences=differences,
                        code_sha256={p.name: c.file_hash(p) for p in code}))
    c.publish_identical(root / "manifest.json", plan)
    for n in ("probe", "tail"):
        (root / n).mkdir(exist_ok=True)
    work = sum((len(ARMS) + int(case["id"] == SCALE_POINT)) * (case["end"] - case["frame"]) for case in cases) + 1
    progress = OverallProgress("Stage-186", work, interval=args.progress_interval, unit="work")
    progress.report("Loading frozen SGLA")
    args.parameter = runtime["tracker"]["parameter"]
    tracker, _ = base.s103.s76.startup.checked_runtime(args, dict(old, fingerprint=plan["fingerprint"],
                 runtime_identity=runtime, runtime=runtime["tracker"]), root)
    progress.report("Probing rounded crops and continuing each output")
    tables, artifacts, counts = instrument(root, r185, d185, plan, cases, timing, response, entries, archive_iou,
                                           tracker, old["root"], progress)
    tracker.assert_frozen()
    tables["cases"] = cases
    if progress.done != work - 1:
        raise ValueError("Incomplete Stage-186 progress accounting")
    return dict(source_stage185=SOURCE185, plan_fingerprint=plan["fingerprint"], points=len(cases), layer=prior.LAYER,
                stable_controls=sum(r["cohort"] == "stable" for r in cases), work=work, training_performed=False,
                runtime=runtime, runtime_differences=differences, artifact_sha256=artifacts,
                table_sha256=gain.prior.publish(root, tables, stage="stage186"), **counts), progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, help="Check the Stage-185 export on CPU")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--progress-interval", type=float, default=15)
    parser.set_defaults(mode="stage186_crop_quantization")
    for stage in SOURCES:
        parser.add_argument(f"--stage{stage}-root", type=Path)
    args = parser.parse_args()
    paths = [p.expanduser().resolve() for stage in SOURCES if (p := getattr(args, f"stage{stage}_root"))]
    if args.bundle:
        if paths or args.output_dir or args.checkpoint:
            parser.error("--bundle is a standalone CPU check")
        _, tables = gain.prior.archives.bundle(args.bundle, 185, SOURCE185)
        cases, _, _ = bind_roster(lambda k: tables[f"stage185_{k}.csv"])
        print(f"Verified {len(cases)} Stage-185 points, {sum(r['cohort'] == 'stable' for r in cases)} stable controls")
        return
    if len(paths) != len(SOURCES) or args.output_dir is None or args.checkpoint is None or args.progress_interval <= 0:
        parser.error("Need all source directories, original checkpoint and output directory")
    root = args.output_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate output directory")
    with gain.prior.prior.writer_lock(root):
        out = root / "stage186_decision.json"
        if out.exists():
            saved = c.c66.sealed(out, SCHEMA)
            plan = c.c66.sealed(root / "manifest.json", SCHEMA)
            if saved["source_stage185"] != SOURCE185 or saved["plan_fingerprint"] != plan["fingerprint"]:
                raise ValueError("Stage-186 completed run has another source or plan")
            io.verify({str(root / k): h for k, h in (saved["artifact_sha256"] | saved["table_sha256"]).items()})
            OverallProgress("Stage-186", saved["work"], saved["work"]).report("Complete")
            return
        result, progress = run(args, root, paths)
        c.publish_identical(out, io.seal(dict(schema=SCHEMA, status="COMPLETE_STAGE186_CROP_QUANTIZATION_DIAGNOSTIC", **result)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
