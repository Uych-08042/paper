"""Cross committed box center and size, and trace the resulting search feedback."""

import argparse
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
import stage180_center_match_audit as center
import stage184_center_mapping as prior
from stage27_uav_common import crop_geometry, scene_geometry
from stage28_response_common import dense_boxes


source, online, previous, risk = prior.source, prior.online, prior.previous, prior.risk
base, gain, c, io = prior.base, prior.gain, prior.c, prior.io
SOURCE184 = "5391afd90afa6a9199066b71330294f2257311df747b450210152e33ee3ce2e9"
SCHEMA = "stage185_writeback_components_v1"
SOURCES = (76, 89, 93, 121, 170, 184)
ARMS = ("native", "mapping_only", "full_center_position", "full_center_size", "full_center")
LAYER = prior.PRIMARY
LONG = prior.LONG
MAP_ATOL = 1e-6
WATCH = {"uav0000091", "uav0000176", "uav0000204"}


def bind_roster(read):
    cases = [dict(r, frame=int(r["frame"]), end=int(r["end"]), earlier=int(r["earlier"]),
                  prior_event=r["prior_event"] == "True", selected=r["selected"] == "True")
             for r in read("cases")]
    anchors = {(r["case_id"], r["arm"]): r for r in read("anchors") if int(r["layer"]) == LAYER}
    tails = {(r["case_id"], r["arm"]): r for r in read("tails") if int(r["layer"]) == LAYER}
    expected = {(r["id"], arm) for r in cases for arm in prior.ARMS}
    if (len(cases) != 27 or sum(r["cohort"] == "stable" for r in cases) != 8
            or len({r["id"] for r in cases}) != len(cases) or set(anchors) != expected or set(tails) != expected
            or any(int(tails[p, arm]["successors"]) < 1 for p, arm in expected)):
        raise ValueError("Need the complete Stage-184 L9 roster")
    return cases, anchors, tails


def input_paths(root, receipt, case, rows):
    key, steps = case["id"], case["end"] - case["frame"]
    data = prior.cache_reference(root, receipt, f"tail/{key}.npz",
                                 dict(plan_fingerprint=receipt["plan_fingerprint"], kind="tail", case_id=key))
    indices = data["inverse"].reshape(-1, len(prior.ARMS))
    expected = len(prior.LAYERS) if case["cohort"] == "supply_loss" else 1
    if indices.shape[0] != expected or data["paths"].shape[1:] != (steps + 1, 4):
        raise ValueError("Stage-184 tail layout changed")
    idx = prior.LAYERS.index(LAYER) if expected > 1 else 0
    paths = data["paths"][indices[idx]]
    for arm, path in zip(prior.ARMS, paths, strict=True):
        row = rows[key, arm]
        np.testing.assert_allclose(path[0], [float(row[k]) for k in ("box_x", "box_y", "box_w", "box_h")],
                                   rtol=0., atol=source.BOX_ATOL)
    return data["paths"], paths


def cross_anchors(paths):
    """Keep original endpoints exact and exchange one component of the two commits."""
    native = paths[0, 0]
    mapping_only, full_center = paths[1, 0], paths[3, 0]
    return np.vstack((native, source.cross.cross_parents(mapping_only, full_center)))


def complete_paths(anchors, known_paths, successors, compute, progress, cached=None):
    unique, inverse = np.unique(anchors, axis=0, return_inverse=True)
    lookup = {tuple(p[0]): p for p in known_paths}
    missing = [j for j, box in enumerate(unique) if tuple(box) not in lookup]
    work = len(ARMS) * successors
    required = len(missing) * successors
    if cached is None:
        paths = np.empty((len(unique), successors + 1, 4), float)
        for j, box in enumerate(unique):
            if tuple(box) in lookup:
                paths[j] = lookup[tuple(box)]
        if missing:
            paths[missing] = compute(unique[missing])
        progress.reused += (len(unique) - len(missing)) * successors
        progress.advance(count=work - required)
        data = dict(paths=paths, inverse=inverse, missing=np.asarray(missing, int))
    else:
        data = cached
        progress.reused += work
        progress.advance(count=work)
    np.testing.assert_array_equal(data["inverse"], inverse)
    np.testing.assert_array_equal(data["missing"], missing)
    np.testing.assert_array_equal(data["paths"][:, 0], unique)
    if (data["paths"].shape != (len(unique), successors + 1, 4) or not np.isfinite(data["paths"]).all()
            or (data["paths"][..., 2:] <= 0.).any()):
        raise ValueError("Invalid path after the crossed anchor")
    for i, box in enumerate(unique):
        if tuple(box) in lookup:
            np.testing.assert_array_equal(data["paths"][i], lookup[tuple(box)])
    return data, data["paths"][inverse], required


def first_true(mask, first_frame):
    ids = np.flatnonzero(mask)
    return int(first_frame + ids[0]) if ids.size else None


def trace_geometry(paths, frame, dimensions, factor, size):
    crop_arrays = np.asarray([[crop_geometry(parent, hw, factor, size)[0]
                              for parent, hw in zip(path[:-1], dimensions, strict=True)] for path in paths])
    output = paths[:, 1:]
    ref = ARMS.index("mapping_only")
    rows = []
    for i, arm in enumerate(ARMS):
        dx = np.abs(output[i] - output[ref]).max(axis=-1)
        crop = crop_arrays[i] != crop_arrays[ref]
        centers = crop_arrays[..., :2] + .5 * crop_arrays[..., 2:]
        rows.append(dict(arm=arm, next_crop_left=int(crop_arrays[i, 0, 0]), next_crop_top=int(crop_arrays[i, 0, 1]),
                         next_crop_side=int(crop_arrays[i, 0, 2]),
                         first_crop_difference=first_true(crop.any(axis=-1), frame + 1),
                         first_crop_center_difference=first_true((centers[i] != centers[ref]).any(axis=-1), frame + 1),
                         first_crop_size_difference=first_true(crop[:, 2], frame + 1),
                         first_float_output_difference=first_true(dx > source.BOX_ATOL, frame + 1),
                         first_integer_output_difference=first_true(
                             (np.trunc(output[i]) != np.trunc(output[ref])).any(axis=-1), frame + 1)))
    return crop_arrays, rows


def response_probe(observer, template, initial, read, paths, frame, cfg):
    """Observe the first successor and search for the first actual map change."""
    crop, hw, size, factor = (cfg[k] for k in ("crop", "dimensions", "size", "factor"))
    ref = ARMS.index("mapping_only")
    pending = set(range(len(ARMS))) - {ref}
    evidence = []
    counts = 0
    observer.initialize(template, initial, "native", None)
    for j in range(len(hw)):
        candidates = pending if j == 0 else {i for i in pending if not np.array_equal(crop[i, j], crop[ref, j])}
        if not candidates:
            if pending:
                continue
            break
        t = frame + j + 1
        image = read(t)
        seen = {}
        found = set()
        for i in sorted(candidates | {ref}):
            parent = paths[i, j]
            key = tuple(parent)
            if key not in seen:
                observer.maps = None
                observer.rt.tracker.frame_id = t - 1
                row = observer.step(image, parent, t)
                if observer.maps is None or row["crop_calls"] != 1 or row["head_calls"] != 1:
                    raise ValueError("Missing native successor CENTER maps or unexpected call budget")
                seen[key] = dict(row, **observer.maps)
                counts += 1
            obs = seen[key]
            np.testing.assert_array_equal(obs["boxes"], paths[i, j + 1])
        baseline = seen[tuple(paths[ref, j])]
        for i in sorted(candidates | ({ref} if j == 0 else set())):
            parent = paths[i, j]
            obs = seen[tuple(parent)]
            deltas = [float(np.abs(obs[k] - baseline[k]).max()) for k in ("raw", "size", "offset")]
            changed = max(deltas) > MAP_ATOL or obs["layer"] != baseline["layer"]
            if changed:
                found.add(i)
            if changed or j == 0:
                grid = dense_boxes(obs["size"], obs["offset"], parent, size / float(crop[i, j, 2]), size, hw[j])
                cell = int(np.argmax(obs["raw"] * cfg["window"]))
                np.testing.assert_allclose(grid[cell], obs["boxes"], atol=source.BOX_ATOL, rtol=0.)
                grid[cell] = obs["boxes"]
                gt = cfg["truth"][j]
                iou = online.tracking_iou(grid, gt)
                geometry = scene_geometry(parent, gt, hw[j], factor, size) if (iou >= 0.).all() else None
                evidence.append(dict(arm=ARMS[i], frame=t, first_changed=changed, selected_layer=int(obs["layer"]),
                                     hann_cell=cell, hann_peak=float(obs["hann_peak"]), raw_map_delta=deltas[0],
                                     size_map_delta=deltas[1], offset_map_delta=deltas[2],
                                     reference_layer=int(baseline["layer"]),
                                     reference_hann_cell=int(np.argmax(baseline["raw"] * cfg["window"])),
                                     reference_iou=float(online.tracking_iou(baseline["boxes"], gt)) if geometry is not None else None,
                                     coverage=geometry["coverage"] if geometry is not None else None,
                                     input_short_side_px=float(min(geometry["input_wh"])) if geometry is not None else None,
                                     dense_best_iou=float(iou.max()) if geometry is not None else None,
                                     output_iou=float(iou[cell]) if geometry is not None else None))
        pending -= found
    return evidence, counts


def treatment_summary(case, paths, target, prefix):
    rows = []
    frame, end = case["frame"], case["end"]
    labels = target[frame:end + 1]
    mapping = ARMS.index("mapping_only")
    pair = source.tail_summary(paths, labels, prefix, mapping, range(len(ARMS)))
    native = source.tail_summary(paths, labels, prefix, 0, range(len(ARMS)))
    values = online.tracking_iou(paths, labels[None])
    for i, arm in enumerate(ARMS):
        x, y, w, h = paths[i, 0]
        rows.append(dict(case_id=case["id"], sequence=case["sequence"], group=case["group"], cohort=case["cohort"],
                         arm=arm, anchor_iou=float(values[i, 0]), anchor_center_x=float(x + w / 2.),
                         anchor_center_y=float(y + h / 2.), anchor_width=float(w), anchor_height=float(h),
                         successors=end - frame, mean_tail_iou=float(values[i, 1:].mean()) if native[i]["valid"] else None,
                         **native[i], **{f"{k}_from_mapping": v for k, v in pair[i].items() if k != "valid"}))
    return rows


def summarize(cases, by_case):
    main = []
    for cohort in ("supply_loss", "stable", "supply_loss_without_previous_event"):
        for length in ("all", "full128"):
            used = [p for p in cases if (
                p["cohort"] == cohort or cohort == "supply_loss_without_previous_event" and p["cohort"] == "supply_loss"
                and not p["prior_event"]) and by_case[p["id"]][0]["valid"]
                and (length == "all" or p["end"] - p["frame"] == LONG)]
            for i, arm in enumerate(ARMS[2:], 2):
                rows = [by_case[p["id"]][i] for p in used]
                deltas = defaultdict(list)
                joint_groups = set()
                for case, row in zip(used, rows, strict=True):
                    deltas[row["group"]].append(row["delta128_or_shorter_from_mapping"])
                    if row["protected_rescue"] and by_case[case["id"]][-1]["protected_rescue"]:
                        joint_groups.add(row["group"])
                avg = [float(np.mean(v)) for v in deltas.values()]
                main.append(dict(cohort=cohort, length=length, arm=arm, points=len(rows), groups=len(avg),
                                 mean_group_delta_from_mapping=float(np.mean(avg)) if avg else None,
                                 delta_without_best_group=float(np.mean(sorted(avg)[:-1])) if len(avg) > 1 else None,
                                 reproduced_full_rescue_groups=len(joint_groups),
                                 rescue_groups=len({r["group"] for r in rows if r["protected_rescue"]}),
                                 harm_groups=len({r["group"] for r in rows if r["event_harm"]})))
    return main


def crossed_effects(cases, by_case):
    effects = []
    for case in cases:
        rows = by_case[case["id"]][1:]
        result = dict(case_id=case["id"], group=case["group"], cohort=case["cohort"], successors=case["end"] - case["frame"])
        for metric in ("anchor_iou", "mean_tail_iou"):
            baseline, position, size, joint = (r[metric] for r in rows)
            if None not in (baseline, position, size, joint):
                result.update({f"{metric}_position": position - baseline, f"{metric}_size": size - baseline,
                               f"{metric}_full": joint - baseline, f"{metric}_interaction": joint - size - position + baseline})
            else:
                result.update({f"{metric}_{k}": None for k in ("position", "size", "full", "interaction")})
        effects.append(result)
    return effects


def instrument(root, r184, d184, plan, cases, rows184, tails184, entries, archive_iou, tracker, image_root, progress):
    rt = tracker.rt
    factor, size = float(rt.tracker.params.search_factor), int(rt.tracker.params.search_size)
    window = rt.tracker.output_window.detach().float().cpu().numpy().reshape(rt.search_side, rt.search_side)
    exact = not plan["runtime_differences"]
    counts = dict(required_successor_calls=0, response_probe_calls=0)
    artifacts, outcomes, timings, audits, by_case = {}, [], [], [], {}

    def read(seq, f):
        return online.checked_image(image_root, entries[seq]["frames"][f])

    for case in cases:
        key, seq, frame, end = (case[k] for k in ("id", "sequence", "frame", "end"))
        entry = entries[seq]
        target = np.asarray(entry["ground_truth"], float)
        known, original = input_paths(r184, d184, case, rows184)
        anchors = cross_anchors(original)
        template = None

        def get_template():
            nonlocal template
            if template is None:
                template = read(seq, 0)
            return template

        def compute(candidates):
            with ExitStack() as stack:
                observer = online.RoutedObserver(tracker, stack)
                return source.continue_from(observer, lambda f: read(seq, f), get_template(), entry["initial_box"],
                                            frame, end, candidates, progress)

        path = root / "tail" / f"{key}.npz"
        meta = dict(plan_fingerprint=plan["fingerprint"], kind="tail", case_id=key)
        cached = c.read_cache(path, meta)
        saved, paths, needed = complete_paths(anchors, known if exact else [], end - frame, compute, progress, cached)
        counts["required_successor_calls"] += needed
        if cached is None:
            c.save(path, saved, meta)
        prior.record_artifacts(root, path, artifacts)
        if exact:
            np.testing.assert_array_equal(paths[[0, 1, 4]], original[[0, 1, 3]])
        scores = treatment_summary(case, paths, target, archive_iou[seq][:frame - 1])
        if exact:
            for label, i in (("current", 0), ("mapping_only", 1), ("full_center", 4)):
                saved_row = tails184[key, label]
                for name in ("delta128_or_shorter", "delta32_or_shorter"):
                    expected = float(saved_row[name]) if saved_row[name] else None
                    if scores[i][name] != expected:
                        raise ValueError(f"Stage-184 successor metric changed at {key} / {label} / {name}")
        outcomes.extend(scores)
        by_case[key] = scores

        dimensions = np.array([[r["height"], r["width"]] for r in entry["frames"][frame + 1:end + 1]])
        crop, timing = trace_geometry(paths, frame, dimensions, factor, size)
        report = root / "probe" / f"{key}.json"
        if report.exists():
            saved_probe = c.c66.sealed(report, SCHEMA)
            if saved_probe["plan_fingerprint"] != plan["fingerprint"] or saved_probe["case_id"] != key:
                raise ValueError("Response probe cache belongs to a different plan")
            observed, measured = saved_probe["observations"], saved_probe["calls"]
            progress.reused += 1
        else:
            with ExitStack() as stack:
                observer = center.CenterObserver(tracker, stack)
                observed, measured = response_probe(observer, get_template(), entry["initial_box"],
                    lambda f: read(seq, f), paths, frame, dict(crop=crop, dimensions=dimensions, size=size,
                    factor=factor, truth=target[frame + 1:end + 1], window=window))
            c.publish_identical(report, io.seal(dict(schema=SCHEMA, plan_fingerprint=plan["fingerprint"],
                                                    case_id=key, observations=observed, calls=measured)))
        counts["response_probe_calls"] += measured
        artifacts[report.relative_to(root).as_posix()] = c.file_hash(report)
        first_changed = {r["arm"]: r["frame"] for r in observed if r["first_changed"]}
        audits.extend(dict(case_id=key, group=case["group"], cohort=case["cohort"], **r) for r in observed)
        timings.extend(dict(case_id=key, group=case["group"], cohort=case["cohort"],
                            first_response_difference=first_changed.get(r["arm"]), **r) for r in timing)
        progress.advance()
    summary = summarize(cases, by_case)
    return dict(windows=outcomes, timing=timings, response=audits, summary=summary, effects=crossed_effects(cases, by_case),
                focus=[r for r in outcomes if r["group"] in WATCH and r["cohort"] == "supply_loss"]), artifacts, counts


def run(args, root, paths):
    r76, r89, r93, r121, r170, r184 = paths
    p170 = c.c66.sealed(r170 / "manifest.json", previous.SCHEMA)
    d170 = c.c66.sealed(r170 / "stage170_decision.json", previous.SCHEMA)
    p184 = c.c66.sealed(r184 / "manifest.json", prior.SCHEMA)
    d184 = c.c66.sealed(r184 / "stage184_decision.json", prior.SCHEMA)
    if (d170["fingerprint"] != source.event.matched.SOURCE170 or d170["plan_fingerprint"] != p170["fingerprint"]
            or p170["runtime"] != d170["runtime"] or d184["fingerprint"] != SOURCE184
            or d184["source_stage183"] != prior.SOURCE183 or d184["plan_fingerprint"] != p184["fingerprint"]
            or d184["runtime"] != d170["runtime"] or d184["runtime_differences"]):
        raise ValueError("Need the sealed same-runtime Stage-170 and Stage-184 sources")
    cases, anchors184, tails184 = bind_roster(lambda k: risk.read_verified(r184, d184, 184, k))
    if cases != p184["cases"]:
        raise ValueError("Stage-184 table and planned cases differ")
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
    differences = base.s109.check_runtime(d184["runtime"], runtime)
    code = (Path(__file__), Path(prior.__file__), Path(source.__file__), Path(center.__file__),
            Path(source.cross.__file__), Path(dense_boxes.__code__.co_filename))
    plan = io.seal(dict(schema=SCHEMA, source_stage184=SOURCE184, source_stage170=d170["fingerprint"], cases=cases,
                        layer=LAYER, arms=list(ARMS), map_atol=MAP_ATOL, runtime=runtime, runtime_differences=differences,
                        code_sha256={p.name: c.file_hash(p) for p in code}))
    c.publish_identical(root / "manifest.json", plan)
    for n in ("tail", "probe"):
        (root / n).mkdir(exist_ok=True)
    logical = len(ARMS) * sum(r["end"] - r["frame"] for r in cases)
    work = logical + len(cases) + 1
    progress = OverallProgress("Stage-185", work, interval=args.progress_interval, unit="work")
    progress.report("Loading frozen SGLA")
    args.parameter = runtime["tracker"]["parameter"]
    tracker, _ = base.s103.s76.startup.checked_runtime(args, dict(old, fingerprint=plan["fingerprint"],
                 runtime_identity=runtime, runtime=runtime["tracker"]), root)
    progress.report("Crossing box components and probing successor responses")
    tables, artifacts, counts = instrument(root, r184, d184, plan, cases, anchors184, tails184, entries, archive_iou,
                                           tracker, old["root"], progress)
    tracker.assert_frozen()
    tables["cases"] = cases
    if progress.done != work - 1:
        raise ValueError("Incomplete Stage-185 progress accounting")
    return dict(source_stage184=SOURCE184, plan_fingerprint=plan["fingerprint"], points=len(cases), layer=LAYER,
                stable_controls=sum(r["cohort"] == "stable" for r in cases), logical_successor_calls=logical,
                training_performed=False, work=work, runtime=runtime, runtime_differences=differences,
                artifact_sha256=artifacts, table_sha256=gain.prior.publish(root, tables, stage="stage185"), **counts), progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, help="Check the Stage-184 export on CPU")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--progress-interval", type=float, default=15)
    parser.set_defaults(mode="stage185_writeback_components")
    for stage in SOURCES:
        parser.add_argument(f"--stage{stage}-root", type=Path)
    args = parser.parse_args()
    paths = [p.expanduser().resolve() for stage in SOURCES if (p := getattr(args, f"stage{stage}_root"))]
    if args.bundle:
        if paths or args.output_dir or args.checkpoint:
            parser.error("--bundle is a standalone CPU check")
        _, tables = gain.prior.archives.bundle(args.bundle, 184, SOURCE184)
        cases, _, _ = bind_roster(lambda k: tables[f"stage184_{k}.csv"])
        print(f"Verified {len(cases)} Stage-184 L9 points, {sum(r['cohort'] == 'stable' for r in cases)} stable controls")
        return
    if len(paths) != len(SOURCES) or args.output_dir is None or args.checkpoint is None or args.progress_interval <= 0:
        parser.error("Need all source directories, original checkpoint and output directory")
    root = args.output_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate output directory")
    with gain.prior.prior.writer_lock(root):
        out = root / "stage185_decision.json"
        if out.exists():
            saved = c.c66.sealed(out, SCHEMA)
            plan = c.c66.sealed(root / "manifest.json", SCHEMA)
            if saved["source_stage184"] != SOURCE184 or saved["plan_fingerprint"] != plan["fingerprint"]:
                raise ValueError("Stage-185 completed run has another source or plan")
            io.verify({str(root / k): h for k, h in (saved["artifact_sha256"] | saved["table_sha256"]).items()})
            OverallProgress("Stage-185", saved["work"], saved["work"]).report("Complete")
            return
        result, progress = run(args, root, paths)
        c.publish_identical(out, io.seal(dict(schema=SCHEMA, status="COMPLETE_STAGE185_WRITEBACK_COMPONENTS_DIAGNOSTIC", **result)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
