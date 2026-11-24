"""Diagnose how crop feedback changes layer-level rescue opportunities around native failures."""

import argparse
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
import stage137_shared_crop_layers as probe
import stage140_crop_center_size_cross as cross
import stage182_layer_opportunity as prior
from stage27_uav_common import scene_geometry
from stage28_response_common import dense_boxes


base, gain, c, io = prior.base, prior.gain, prior.c, prior.io
event, online, previous, risk = prior.event, prior.online, prior.previous, prior.risk
LAYERS, LONG, SHORT = prior.LAYERS, prior.LONG, prior.SHORT
GOOD = online.TRACKING["recovery_iou"]
SOURCE182 = "47cb1378551c0b083f6787e35cba1b95d73ff8c51ed3870a771a163a532767fa"
SCHEMA = "stage183_search_feedback_v1"
SOURCES = (76, 89, 93, 121, 170, 176, 182)
REFERENCES = {170: prior.REFERENCES[170], 176: prior.REFERENCES[176], 182: (prior.SCHEMA, SOURCE182)}
ARMS = ("current", "earlier_center", "earlier_size", "earlier_parent")
BOX_ATOL = .001
GEOMETRY_ATOL = 1e-9


def bind_events(read):
    roster = {r["sequence"]: int(r["frames"]) for r in read(170, "video_summary") if r["arm"] == "native"}
    swept = {r["group"] for r in read(176, "cases")}
    windows = read(182, "cases")
    cases = []
    for r in read(170, "failures"):
        if r["arm"] != "native":
            continue
        seq, onset, end = r["sequence"], int(r["onset"]), int(r["end"])
        group = risk.SOURCE_GROUP(seq)
        cases.append(dict(id=f"{seq}__failure{onset:06d}", sequence=seq, group=group, onset=onset, event_end=end,
                          censor=r["censor"], recovery=int(r["recovery"]) if r["recovery"] else None,
                          stage176_source_seen=group in swept,
                          stage182_event_seen=any(w["sequence"] == seq and int(w["frame"]) <= end
                                                  and int(w["end"]) >= onset for w in windows)))
    if (len(roster) != online.DESIGN["videos"] or len(cases) != 19 or len({r["id"] for r in cases}) != len(cases)
            or len(windows) != 18 or any(not 1 <= r["onset"] <= r["event_end"] <= roster[r["sequence"]] for r in cases)):
        raise ValueError("Unexpected native failure or Stage-182 roster")
    return roster, cases


def scan_ranges(cases, archive_iou):
    """Use the last confirmed accurate frame and the actual failure confirmation."""
    ranges = []
    for case in cases:
        seq, onset = case["sequence"], case["onset"]
        iou = archive_iou[seq]
        events = online.failure_events(iou)
        current = next(r for r in events if r["start"] + 1 == onset)
        if ((current["end"] + 1, current["censor"]) != (case["event_end"], case["censor"])
                or case["recovery"] != (current["recovery"] + 1 if current["recovery"] is not None else None)):
            raise ValueError("Sealed source event does not reproduce")
        previous_end = max((r["end"] + 1 for r in events if r["end"] < current["start"]), default=0)
        prior_iou = iou[previous_end:onset - 1]
        invalid = np.flatnonzero(prior_iou < 0.)
        lower = previous_end + (int(invalid[-1]) + 1 if invalid.size else 0)
        before = iou[lower:onset - 1]
        k = online.TRACKING["recovery_run"]
        good = np.flatnonzero(np.lib.stride_tricks.sliding_window_view(before >= GOOD, k).all(-1)) if len(before) >= k else []
        first = lower + int(good[-1]) + k if len(good) else min(onset, lower + 1)
        ranges.append(dict(case, first=first, confirmation=current["confirmation"] + 1, confirmed_good_start=bool(len(good))))
    return ranges


def measure(data, truth, cfg, reference=None):
    np.testing.assert_array_equal(data["counts"], [1] * 13 + [6])
    geo = scene_geometry(data["parent"], truth, data["image_hw"], cfg["factor"], cfg["size"])
    np.testing.assert_allclose(data["factor"], geo["factor"], atol=1e-12, rtol=0.)
    visible = geo["gt_image_fraction"]
    scores = data["raw"] * cfg["window"]
    iou = online.tracking_iou(data["boxes"], truth)
    if (iou < 0.).any() or int(data["selected"]) not in LAYERS or not np.isfinite(scores).all():
        raise ValueError("Invalid anchor observation")
    dense_best, cells = [], []
    for i in range(len(LAYERS)):
        grid = dense_boxes(data["size"][i], data["offset"][i], data["parent"], geo["factor"], cfg["size"], data["image_hw"])
        cell = int(scores[i].argmax())
        np.testing.assert_allclose(grid[cell], data["boxes"][i], atol=BOX_ATOL, rtol=0.)
        grid[cell] = data["boxes"][i]
        cells.append(cell)
        dense_best.append(float(online.tracking_iou(grid, truth).max()))
    n = LAYERS.index(int(data["selected"]))
    if reference is not None:
        box, layer, peak = reference
        np.testing.assert_allclose(data["boxes"][n], box, atol=BOX_ATOL, rtol=0.)
        np.testing.assert_equal(data["selected"], layer)
        np.testing.assert_allclose(scores[n].max(), peak, atol=1e-6, rtol=0.)
    return dict(coverage_visible=float(np.clip(geo["coverage"] / visible, 0., 1.)) if visible else None,
                target_center_inside=geo["center_in_usable"], input_short_side_px=float(min(geo["input_wh"])),
                crop_side=int(geo["crop"][2]), selected=int(data["selected"]), native_iou=float(iou[n]),
                best_standard=float(iou.max()), best_dense=max(dense_best), native_dense=dense_best[n],
                standard=iou, dense=dense_best, cells=cells)


def find_transitions(case, frames, stride):
    """Report each marker; choose the first observed reduction in standard box supply."""
    transitions = []
    for earlier, now in zip(frames, frames[1:]):
        coverage = earlier["coverage_visible"], now["coverage_visible"]
        marks = (("standard_supply_loss", earlier["best_standard"] >= GOOD > now["best_standard"]),
                 ("dense_supply_loss", earlier["best_dense"] >= GOOD > now["best_dense"]),
                 ("readout_loss", earlier["native_iou"] >= GOOD > now["native_iou"] and now["native_dense"] >= GOOD),
                 ("target_center_exit", earlier["target_center_inside"] and not now["target_center_inside"]),
                 ("coverage_loss", None not in coverage and coverage[0] >= 1. - GEOMETRY_ATOL > coverage[1]),
                 ("below_one_patch", earlier["input_short_side_px"] >= stride > now["input_short_side_px"]))
        transitions.extend(dict(event_id=case["id"], frame=now["frame"], marker=label) for label, hit in marks if hit)
    chosen = next((r for r in transitions if r["marker"] == "standard_supply_loss"), None)
    if chosen is None:
        chosen = transitions[0] if transitions else dict(event_id=case["id"], frame=case["onset"], marker="observed_onset")
    return chosen, transitions


def continue_from(observer, read, template, initial_box, frame, end, anchors, progress):
    """Continue after the intervention with the original search and route."""
    observer.initialize(template, initial_box, "native", None)
    paths = np.empty((len(anchors), end - frame + 1, 4), float)
    paths[:, 0] = anchors
    for j, f in enumerate(range(frame + 1, end + 1), 1):
        image = read(f)
        for i in range(len(anchors)):
            observer.rt.tracker.frame_id = f - 1
            row = observer.step(image, paths[i, j - 1], f)
            np.testing.assert_equal(row["layer"], row["native_layer"])
            np.testing.assert_equal(row["head_calls"], 1)
            paths[i, j] = row["boxes"]
            progress.advance()
    return paths


def tail_summary(paths, labels, prefix, control, alternatives):
    values = online.tracking_iou(paths, labels[None])
    valid = values.shape[1] > 1 and bool((values >= 0.).all())
    result = []
    for i in alternatives:
        row = dict(valid=valid, delta128_or_shorter=None, delta32_or_shorter=None,
                   protected_rescue=None, event_harm=None, added_failure=None)
        if valid:
            delta = values[i, 1:] - values[control, 1:]
            long, short = float(delta.mean()), float(delta[:SHORT].mean())
            flags = event.event_labels(prefix, values[[control, i]])
            safe = not flags["event_harm"] and flags["added_failure"] == 0 and min(short, long) >= -prior.sweep.EPSILON
            row.update(delta128_or_shorter=long, delta32_or_shorter=short, event_harm=flags["event_harm"],
                       added_failure=flags["added_failure"], protected_rescue=safe and (
                           flags["prevented_terminal"] or flags["rescued_recovery"]))
        result.append(row)
    return result


def instrument(root, plan, cases, entries, archives, archive_iou, tracker, image_root, progress):
    rt = tracker.rt
    artifacts, scans, scan_layers, markers, points = {}, [], [], [], []
    cfg = dict(factor=rt.tracker.params.search_factor, size=rt.tracker.params.search_size,
               window=rt.tracker.output_window.detach().float().cpu().numpy().reshape(rt.search_side, rt.search_side))
    stride = cfg["size"] // rt.search_side
    if stride * rt.search_side != cfg["size"]:
        raise ValueError("Search patch stride is inconsistent with the response grid")
    exact = not plan["runtime_differences"]

    def get_cache(directory, key, compute, audit, work=0):
        path = root / directory / f"{key}.npz"
        meta = dict(plan_fingerprint=plan["fingerprint"], kind=directory, key=key)
        value = c.read_cache(path, meta)
        if value is None:
            value = compute()
            audit(value)
            c.save(path, value, meta)
        else:
            audit(value)
            progress.reused += work
        for f in (path, path.with_suffix(".receipt.json")):
            artifacts[f.relative_to(root).as_posix()] = c.file_hash(f)
        progress.advance(count=work)
        return value

    def read(seq, frame):
        return online.checked_image(image_root, entries[seq]["frames"][frame])

    def source_reference(archive, frame):
        return tuple(archive[k][frame - 1] for k in ("boxes", "layer", "hann_peak")) if exact else None

    progress.report("Scanning native failure transitions")
    for case in cases:
        seq, event_id = case["sequence"], case["id"]
        archive, truth = archives[seq], entries[seq]["ground_truth"]
        tracker.initialize(read(seq, 0), entries[seq]["initial_box"])
        observations = []
        for frame in range(case["first"], case["confirmation"] + 1):
            parent = archive["parent"][frame - 1]

            def collect_scan():
                return probe.collect(rt, read(seq, frame), parent, frame)

            def audit_scan(data):
                np.testing.assert_array_equal(data["parent"], parent)

            data = get_cache("scan", f"{seq}__f{frame:06d}", collect_scan, audit_scan, work=1)
            m = measure(data, truth[frame], cfg, source_reference(archive, frame))
            row = dict(event_id=event_id, sequence=seq, frame=frame,
                       **{k: v for k, v in m.items() if k not in ("standard", "dense", "cells")})
            observations.append(row)
            scans.append(row)
            scan_layers.extend(dict(event_id=event_id, frame=frame, layer=layer, selected=m["selected"] == layer,
                                    output_iou=float(m["standard"][i]), dense_iou=m["dense"][i], cell=m["cells"][i])
                               for i, layer in enumerate(LAYERS))
        chosen, changes = find_transitions(case, observations, stride)
        markers.extend(changes)
        for phase, frame in (("before", chosen["frame"] - 1), ("after", chosen["frame"])):
            if frame >= case["first"]:
                points.append(dict(id=f"{event_id}__{phase}", event_id=event_id, group=case["group"], sequence=seq,
                                   phase=phase, marker=chosen["marker"], frame=frame,
                                   end=min(frame + LONG, len(archive["boxes"])),
                                   earlier_parent_frame=max(0, chosen["frame"] - 2)))
    c.publish_identical(root / "collection.json", io.seal(dict(schema=SCHEMA, plan_fingerprint=plan["fingerprint"],
                                                               points=points, markers=markers)))
    artifacts["collection.json"] = c.file_hash(root / "collection.json")
    progress.total = progress.done + sum(len(ARMS) + 2 * len(LAYERS) * (p["end"] - p["frame"]) for p in points) + 1
    progress.report("Crossing crop state and layer, then replaying native successors")
    anchors, effects, tails, physical_tail_calls = [], [], [], 0
    with ExitStack() as stack:
        observer = online.RoutedObserver(tracker, stack)
        for point in points:
            seq, frame, end = (point[k] for k in ("sequence", "frame", "end"))
            archive, entry = archives[seq], entries[seq]
            parent = archive["parent"][frame - 1]
            earlier = archive["parent"][point["earlier_parent_frame"]]
            parents = cross.cross_parents(parent, earlier)
            template = read(seq, 0)
            observer.initialize(template, entry["initial_box"], "native", None)

            def collect_cross():
                image = read(seq, frame)
                observed = {}
                for item in parents:
                    key = tuple(item)
                    if key not in observed:
                        observed[key] = probe.collect(rt, image, item, frame)
                return {k: np.stack([observed[tuple(item)][k] for item in parents])
                        for k in observed[tuple(parents[0])]}

            def audit_cross(data):
                np.testing.assert_array_equal(data["parent"], parents)
                np.testing.assert_equal(data["counts"], np.tile([1] * 13 + [6], (len(ARMS), 1)))

            data = get_cache("cross", point["id"], collect_cross, audit_cross, work=len(ARMS))
            measured = []
            for j, arm in enumerate(ARMS):
                m = measure({k: v[j] for k, v in data.items()}, entry["ground_truth"][frame], cfg,
                            source_reference(archive, frame) if j == 0 else None)
                measured.append(m)
                for q, layer in enumerate(LAYERS):
                    anchors.append(dict(point_id=point["id"], arm=arm, layer=layer, selected=m["selected"] == layer,
                                        output_iou=float(m["standard"][q]), dense_iou=m["dense"][q], cell=m["cells"][q],
                                        coverage_visible=m["coverage_visible"], target_center_inside=m["target_center_inside"],
                                        input_short_side_px=m["input_short_side_px"], crop_side=m["crop_side"],
                                        crop_sha256=str(data["crop_sha256"][j])))
            a, x, y, b = measured
            for metric in ("best_standard", "best_dense", "native_iou", "coverage_visible", "input_short_side_px"):
                v = [m[metric] for m in (a, x, y, b)]
                effects.append(dict(point_id=point["id"], metric=metric, **dict(zip(ARMS, v, strict=True)),
                                    center_effect=v[1] - v[0] if None not in v else None,
                                    size_effect=v[2] - v[0] if None not in v else None,
                                    parent_effect=v[3] - v[0] if None not in v else None,
                                    interaction=v[3] - v[2] - v[1] + v[0] if None not in v else None))
            anchor_boxes = np.concatenate((data["boxes"][0], data["boxes"][-1])).copy()
            control = LAYERS.index(a["selected"])
            if exact:
                same = (anchor_boxes == anchor_boxes[control]).all(axis=1)
                anchor_boxes[same] = archive["boxes"][frame - 1]
            unique, inverse = np.unique(anchor_boxes, axis=0, return_inverse=True)
            logical = len(anchor_boxes) * (end - frame)

            def collect_tail():
                paths = continue_from(observer, lambda f: read(seq, f), template, entry["initial_box"], frame, end, unique, progress)
                progress.advance(count=logical - len(unique) * (end - frame))
                return dict(paths=paths, inverse=inverse)

            def audit_tail(data):
                np.testing.assert_array_equal(data["inverse"], inverse)
                np.testing.assert_array_equal(data["paths"][:, 0], unique)
                if data["paths"].shape != (len(unique), end - frame + 1, 4) or not np.isfinite(data["paths"]).all():
                    raise ValueError("Unexpected continuation")
                if exact:
                    np.testing.assert_array_equal(data["paths"][inverse[control]], archive["boxes"][frame - 1:end])

            meta_path = root / "tail" / f"{point['id']}.npz"
            cached = meta_path.with_suffix(".receipt.json").exists()
            replay = get_cache("tail", point["id"], collect_tail, audit_tail, work=logical if cached else 0)
            physical_tail_calls += len(unique) * (end - frame)
            paths = replay["paths"][inverse]
            truth = np.asarray(entry["ground_truth"], float)
            for j, row in enumerate(tail_summary(paths, truth[frame:end + 1], archive_iou[seq][:frame - 1],
                                                  control, range(len(anchor_boxes)))):
                tails.append(dict(point_id=point["id"], arm=ARMS[0] if j < len(LAYERS) else ARMS[-1],
                                  layer=LAYERS[j % len(LAYERS)], successors=end - frame, **row))
    return dict(scan=scans, scan_layers=scan_layers, transitions=markers, points=points,
                anchors=anchors, effects=effects, tails=tails), artifacts, physical_tail_calls


def summarize(cases, tables):
    effects = {(r["point_id"], r["metric"]): r for r in tables["effects"]}
    by_point = defaultdict(list)
    for r in tables["tails"]:
        by_point[r["point_id"]].append(r)
    rows = []
    for p in tables["points"]:
        standard = effects[p["id"], "best_standard"]
        dense = effects[p["id"], "best_dense"]
        available = by_point[p["id"]]
        rows.append(dict(p, crop_restores_good_standard=standard["current"] < GOOD <= standard["earlier_parent"],
                         crop_restores_good_dense=dense["current"] < GOOD <= dense["earlier_parent"],
                         current_rescue_layers=sum(r["protected_rescue"] is True for r in available if r["arm"] == "current"),
                         earlier_rescue_layers=sum(r["protected_rescue"] is True for r in available if r["arm"] == "earlier_parent"),
                         valid_tail=all(r["valid"] for r in available), full128=p["end"] - p["frame"] == LONG))
    case_info = {r["id"]: r for r in cases}
    summary = []
    for cohort in ("all", "without_previous_layer_sweep", "without_previous_event"):
        for phase in ("before", "after"):
            selected = [r for r in rows if r["phase"] == phase and (
                cohort == "all" or (not case_info[r["event_id"]]["stage176_source_seen"] and (
                    cohort == "without_previous_layer_sweep" or not case_info[r["event_id"]]["stage182_event_seen"])))]
            summary.append(dict(cohort=cohort, phase=phase, events=len(selected), groups=len({r["group"] for r in selected}),
                                full128=sum(r["full128"] and r["valid_tail"] for r in selected),
                                restored_standard_groups=len({r["group"] for r in selected if r["crop_restores_good_standard"]}),
                                restored_dense_groups=len({r["group"] for r in selected if r["crop_restores_good_dense"]}),
                                current_rescue_groups=len({r["group"] for r in selected if r["current_rescue_layers"]}),
                                earlier_rescue_groups=len({r["group"] for r in selected if r["earlier_rescue_layers"]})))
    return rows, summary


def run(args, root, paths):
    roots = dict(zip(SOURCES, paths, strict=True))
    r76, r89, r93, r121, r170 = (roots[s] for s in SOURCES[:5])
    receipts = {s: c.c66.sealed(roots[s] / f"stage{s}_decision.json", schema) for s, (schema, _) in REFERENCES.items()}
    p170 = c.c66.sealed(r170 / "manifest.json", previous.SCHEMA)
    d170, d182 = receipts[170], receipts[182]
    if (d170["plan_fingerprint"] != p170["fingerprint"] or p170["runtime"] != d170["runtime"]
            or d182["source_stage181"] != prior.SOURCE181 or not d182["replay_matches_source_runtime"]
            or any(receipts[s]["fingerprint"] != fp or receipts[s]["runtime"] != d170["runtime"]
                   for s, (_, fp) in REFERENCES.items())):
        raise ValueError("Need sealed Stage-170, Stage-176 and Stage-182 source reports")
    roster, source = bind_events(lambda s, k: risk.read_verified(roots[s], receipts[s], s, k))
    _, _, jobs = prior.history.native_jobs(r121)
    s95 = base.s103.prior.s95
    p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
    old = origin["original"]
    entries, archives, archive_iou = {}, {}, {}
    for seq in sorted({r["sequence"] for r in source}):
        entry = s95.load_unit(r93, r89, p93, p89, d89, origin, jobs[seq]["spec"])[0]
        count = roster[seq]
        if (entry["sequence"] != seq or len(entry["frames"]) != count + 1
                or len(entry["ground_truth"]) != count + 1):
            raise ValueError(f"Source video length changed for {seq}")
        trace = previous.archive_trace(r170, p170, d170, seq, "native")
        previous.audit(trace, count, entry["initial_box"], "native", 0.)
        entries[seq], archives[seq] = entry, trace
        archive_iou[seq] = online.tracking_iou(trace["boxes"], entry["ground_truth"][1:])
    cases = scan_ranges(source, archive_iou)

    import torch

    torch.set_num_threads(1)
    runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
    differences = base.s109.check_runtime(d170["runtime"], runtime)
    code = (Path(__file__), Path(probe.__file__), Path(cross.__file__), Path(event.__file__),
            Path(dense_boxes.__code__.co_filename), Path(online.__file__))
    plan = io.seal(dict(schema=SCHEMA, source_stage182=SOURCE182, cases=cases, layers=list(LAYERS), arms=list(ARMS),
                        horizon=LONG, good_iou=GOOD, source_reports={str(s): receipts[s]["fingerprint"] for s in receipts},
                        runtime=runtime, runtime_differences=differences, code_sha256={p.name: c.file_hash(p) for p in code}))
    c.publish_identical(root / "manifest.json", plan)
    for folder in ("scan", "cross", "tail"):
        (root / folder).mkdir(exist_ok=True)
    scan_work = sum(r["confirmation"] - r["first"] + 1 for r in cases)
    maximum = scan_work + 2 * len(cases) * (len(ARMS) + 2 * len(LAYERS) * LONG) + 1
    progress = OverallProgress("Stage-183", maximum, interval=args.progress_interval, unit="work")
    progress.report("Loading frozen SGLA")
    args.parameter = runtime["tracker"]["parameter"]
    tracker, _ = base.s103.s76.startup.checked_runtime(args, dict(old, fingerprint=plan["fingerprint"],
                   runtime_identity=runtime, runtime=runtime["tracker"]), root)
    tables, artifacts, physical = instrument(root, plan, cases, entries, archives, archive_iou, tracker, old["root"], progress)
    tracker.assert_frozen()
    tables["cases"] = cases
    tables["opportunities"], tables["summary"] = summarize(cases, tables)
    if progress.done != progress.total - 1:
        raise ValueError("Incomplete Stage-183 progress accounting")
    return dict(source_stage182=SOURCE182, plan_fingerprint=plan["fingerprint"], events=len(cases),
                sequences=len({r["sequence"] for r in cases}), scanned_frames=len(tables["scan"]),
                anchor_points=len(tables["points"]), cross_layer_rows=len(tables["anchors"]),
                logical_native_successor_calls=sum(r["successors"] for r in tables["tails"]),
                unique_native_successor_calls=physical, work=progress.total, runtime=runtime,
                runtime_differences=differences, artifact_sha256=artifacts,
                table_sha256=gain.prior.publish(root, tables, stage="stage183")), progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", nargs=3, type=Path, metavar=("STAGE170", "STAGE176", "STAGE182"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--progress-interval", type=float, default=15)
    parser.set_defaults(mode="stage183_search_feedback")
    for stage in SOURCES:
        parser.add_argument(f"--stage{stage}-root", type=Path)
    args = parser.parse_args()
    paths = [p.expanduser().resolve() for stage in SOURCES if (p := getattr(args, f"stage{stage}_root"))]
    if args.bundles:
        if paths or args.output_dir or args.checkpoint:
            parser.error("--bundles is a standalone CPU check of the native failure roster")
        tables = {s: gain.prior.archives.bundle(p, s, fp)[1]
                  for p, (s, (_, fp)) in zip(args.bundles, REFERENCES.items(), strict=True)}
        _, cases = bind_events(lambda s, k: tables[s][f"stage{s}_{k}.csv"])
        fresh = [r for r in cases if not r["stage176_source_seen"] and not r["stage182_event_seen"]]
        print(f"Verified {len(cases)} native events from {len({r['group'] for r in cases})} source groups; "
              f"{len(fresh)} events in sources outside Stage-176 and windows outside Stage-182")
        return
    if len(paths) != len(SOURCES) or args.output_dir is None or args.checkpoint is None or args.progress_interval <= 0:
        parser.error("Need all source directories, original checkpoint and output directory")
    root = args.output_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate output directory")
    with gain.prior.prior.writer_lock(root):
        out = root / "stage183_decision.json"
        if out.exists():
            saved = c.c66.sealed(out, SCHEMA)
            plan = c.c66.sealed(root / "manifest.json", SCHEMA)
            if saved["source_stage182"] != SOURCE182 or saved["plan_fingerprint"] != plan["fingerprint"]:
                raise ValueError("Stage-183 completed run has another source or plan")
            io.verify({str(root / k): h for k, h in (saved["artifact_sha256"] | saved["table_sha256"]).items()})
            OverallProgress("Stage-183", saved["work"], saved["work"]).report("Complete")
            return
        result, progress = run(args, root, paths)
        c.publish_identical(out, io.seal(dict(schema=SCHEMA, status="COMPLETE_STAGE183_SEARCH_FEEDBACK_DIAGNOSTIC", **result)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
