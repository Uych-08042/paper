"""Expand development windows and test causal history alongside the fixed crop selectors."""

import argparse
from collections import defaultdict
from contextlib import ExitStack
import csv
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np

from experiment_progress import OverallProgress
import stage79_confirmation as confirmation
import stage187_crop_consensus as fixed
from stage184_center_mapping import record_artifacts
from stage4_common import fingerprint
from stage27_uav_common import IMAGE_SUFFIXES, frame_number, inside, read_boxes, read_rgb


crop_stage = fixed.prior
geometry_stage = crop_stage.prior
source, online, previous = geometry_stage.source, geometry_stage.online, geometry_stage.previous
base, gain, c, io, risk = source.base, source.gain, source.c, source.io, source.risk
SCHEMA = "stage188_temporal_support_v1"
SOURCE187 = "a187e3bb24c84c0127675b54a0d510271d75ddd40401b6036695ab879fba9781"
SOURCE93 = "bf9e1e7f757399ebf4c16ce2a6b691fab80789a7de224e7599b8caff11051000"
ROSTER_SHA = "964470edee1409e5310bd79d59fc50f9b341fb9070243ee8436e4b612e559dc3"
ROSTER = Path(__file__).resolve().parents[1] / "docs/experiments/06_sgla_uav_tracking/results/stage79_confirmation_v1.json"
SOURCES = (76, 89, 93, 170, 187)
HISTORY, LONG = 16, fixed.LONG
METHODS = (*fixed.METHODS, "history_position")


def archived_stage93_roster(path):
    parts = re.split(rb"\n===== ([\w.]+) =====\r?\n", path.read_bytes())
    sections = dict(zip(parts[1::2], parts[2::2], strict=True))
    decision = json.loads(sections[b"stage93_decision.json"])
    if decision.pop("fingerprint") != SOURCE93 or fingerprint(decision) != SOURCE93:
        raise ValueError("Unexpected Stage-93 decision fingerprint")
    name = "stage93_video_summary.csv"
    section = sections[name.encode()]
    # The console export adds a blank line after the original CRLF CSV.
    original = section.removesuffix(b"\r\n\n") + b"\r\n" if section.endswith(b"\r\n\n") else section
    if sha256(original).hexdigest() != decision["table_sha256"][name]:
        raise ValueError("Stage-93 roster differs from the sealed table")
    return list(csv.DictReader(original.decode().splitlines()))


def eligible_groups(roster, stage93):
    excluded = set().union(*map(set, roster["excluded_groups_by_source"].values()))
    confirmation_groups = {r["source_group_proxy"] for r in roster["sequences"]}
    used = {risk.SOURCE_GROUP(r["sequence"]) for r in stage93}
    if (roster["fingerprint"] != ROSTER_SHA or len(confirmation_groups) != 28 or len(excluded) != 40
            or len(used) != 28 or not used <= excluded or excluded & confirmation_groups):
        raise ValueError("Unexpected confirmation or recent development membership")
    return excluded - used


def freeze_sequences(dataset, records, wanted):
    groups = defaultdict(list)
    for row in records:
        if row["source_group_proxy"] in wanted and row["frames"] >= HISTORY + LONG + 2:
            groups[row["source_group_proxy"]].append(row)
    clips = []
    for group, rows in sorted(groups.items()):
        pick = min(rows, key=lambda r: (-r["frames"], r["sequence"]))
        seq = pick["sequence"]
        directory = inside(dataset, f"sequences/{seq}")
        images = sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
                        key=frame_number)
        annotation = inside(dataset, f"annotations/{seq}.txt")
        if len(images) != pick["frames"]:
            raise ValueError(f"Inventory changed at {seq}")
        clips.append(dict(sequence=seq, group=group, tracked_frames=len(images) - 1,
                          annotation=annotation.relative_to(dataset).as_posix(), annotation_sha256=c.file_hash(annotation),
                          frames=[dict(file=p.relative_to(dataset).as_posix(), sha256=c.file_hash(p)) for p in images]))
    return clips


def checked_frame(dataset, record):
    path = inside(dataset, record["file"])
    if c.file_hash(path) != record["sha256"]:
        raise ValueError(f"Source pixels changed: {path}")
    return read_rgb(path)


def pick_windows(sequence, values):
    """Use annotated native events to place probes, never to rank outputs."""
    events = online.failure_events(values)
    stop = len(values) - LONG
    usable = [e["start"] + 1 for e in events if HISTORY < e["start"] + 1 <= stop
              and (values[e["start"] - HISTORY:e["start"] + LONG + 1] >= 0.).all()]
    onset = usable[0] if usable else None
    stable = []
    for f in range(HISTORY + 1, stop + 1):
        first, last = f - HISTORY, f + LONG
        if onset is not None and not (last < onset - HISTORY or first > onset + LONG):
            continue
        if (values[first - 1:last] >= .5).all():
            stable.append(f)
    control = min(stable, key=lambda f: (abs(f - onset), f)) if stable and onset is not None else (
        stable[len(stable) // 2] if stable else None)
    return [dict(id=f"{sequence}__{cohort}__f{f:06d}", sequence=sequence, group=risk.SOURCE_GROUP(sequence),
                 cohort=cohort, frame=f, end=f + LONG) for cohort, f in (("onset", onset), ("stable", control)) if f is not None]


def causal_center(history_boxes):
    if np.shape(history_boxes) != (HISTORY, 4) or not np.isfinite(history_boxes).all() or (history_boxes[:, 2:] <= 0.).any():
        raise ValueError("Invalid causal box history")
    centers = history_boxes[:, :2] + .5 * history_boxes[:, 2:]
    return centers[-1] + np.median(np.diff(centers, axis=0), axis=0)


def candidates(case, history_boxes, hw, factor, size):
    current, earlier = history_boxes[-1], history_boxes[-2]
    moved = source.cross.cross_parents(current, earlier)[1]
    seed = np.stack((current, current, moved, current, moved))[:, None]
    arms, parents, crops = crop_stage.crop_arms(seed, case["id"], hw, factor, size)
    probes = {}
    for arm, p, crop in zip(arms[1:], parents[1:], crops[1:], strict=True):
        probes[case["id"], arm] = dict(dx=float(p[0] - current[0]), dy=float(p[1] - current[1]),
                                       crop_left=int(crop[0]), crop_top=int(crop[1]), crop_side=int(crop[2]))
    pool = fixed.candidate_pool(case, probes)
    indices = [arms.index(k) for k in pool]
    return pool, parents[indices], probes


def choose(case, pool, probes, raw_boxes, previous_boxes):
    options, agreement = fixed.rank_arms(pool, probes, case["id"], raw_boxes)
    expected = causal_center(previous_boxes)
    prediction_error = {a: float(np.linalg.norm(raw_boxes[a][:2] + .5 * raw_boxes[a][2:] - expected) /
                                 np.sqrt(previous_boxes[-1, 2:].prod())) for a in pool}
    options["history_position"] = min(pool, key=lambda a: (prediction_error[a], a != "current", fixed.POOL.index(a)))
    return options, agreement, prediction_error


def summarize(cases, choices, availability):
    rows = []
    comparison = []
    index = {(r["case_id"], r["method"]): r for r in choices}
    for cohort in ("onset", "stable"):
        sample = [p for p in cases if p["cohort"] == cohort and all(index[p["id"], a]["valid"] for a in METHODS)]
        support = sum(availability[p["id"]] for p in sample)
        for arm in METHODS:
            observed = [index[p["id"], arm] for p in sample]
            rows.append(dict(cohort=cohort, method=arm, windows=len(observed),
                             groups=len({r["group"] for r in observed}), available_rescues=support,
                             chosen_rescues=sum(r["protected_rescue"] for r in observed),
                             event_harms=sum(r["event_harm"] for r in observed)))
        for treatment, control in (*(("history_position", a) for a in fixed.METHODS),
                                   ("crop_consensus", "highest_hann"), ("highest_hann", "current")):
            grouped = defaultdict(list)
            changes = rescue_gain = rescue_loss = harm_gain = harm_loss = 0
            for p in sample:
                a, b = (index[p["id"], k] for k in (treatment, control))
                grouped[p["group"]].append(a["delta128_or_shorter"] - b["delta128_or_shorter"])
                changes += a["arm"] != b["arm"]
                rescue_gain += a["protected_rescue"] and not b["protected_rescue"]
                rescue_loss += b["protected_rescue"] and not a["protected_rescue"]
                harm_gain += a["event_harm"] and not b["event_harm"]
                harm_loss += b["event_harm"] and not a["event_harm"]
            values = [float(np.mean(x)) for x in grouped.values()]
            comparison.append(dict(cohort=cohort, treatment=treatment, control=control,
                                   windows=len(sample), groups=len(values), choice_changes=changes,
                                   gained_rescues=rescue_gain, lost_rescues=rescue_loss,
                                   added_event_harms=harm_gain, removed_event_harms=harm_loss,
                                   group_wins=sum(x > 1e-9 for x in values), group_losses=sum(x < -1e-9 for x in values),
                                   mean_group_tail_delta=float(np.mean(values)) if values else None,
                                   mean_without_best_group=float(np.mean(sorted(values)[:-1])) if len(values) > 1 else None))
    return rows, comparison


def cached(root, plan, kind, key, report, collect):
    p = root / kind / f"{key}.npz"
    meta = dict(plan_fingerprint=plan["fingerprint"], kind=kind, key=key)
    data = c.read_cache(p, meta)
    fresh = data is None
    if fresh:
        data = collect()
        c.save(p, data, meta)
    record_artifacts(root, p, report)
    return data, fresh


def evaluate(root, plan, clips, tracker, dataset, progress):
    references, labels, ious, cases, artifacts, support = {}, {}, {}, [], {}, []
    progress.report("Collecting original SGLA traces for additional development sources")
    for entry in clips:
        seq = entry["sequence"]
        path = inside(dataset, entry["annotation"])
        if c.file_hash(path) != entry["annotation_sha256"]:
            raise ValueError("Annotation changed since roster was frozen")
        gt = read_boxes(path)
        if len(gt) != len(entry["frames"]):
            raise ValueError("Image and annotation counts differ")

        def native():
            with ExitStack() as stack:
                observer = online.RoutedObserver(tracker, stack)
                return online.collect(observer, entry["frames"], gt[0], dataset, "native", None, progress,
                                      reader=checked_frame)

        trace, fresh = cached(root, plan, "native", seq, artifacts, native)
        if not fresh:
            progress.reused += entry["tracked_frames"]
            progress.advance(count=entry["tracked_frames"])
        previous.audit(trace, entry["tracked_frames"], gt[0], "native", 0.)
        references[seq], labels[seq] = trace, gt
        ious[seq] = online.tracking_iou(trace["boxes"], gt[1:])
        selected = pick_windows(seq, ious[seq])
        cases.extend(selected)
        support.append(dict(sequence=seq, group=entry["group"], tracked_frames=entry["tracked_frames"],
                            native_failure_events=len(online.failure_events(ious[seq])),
                            onset_windows=sum(r["cohort"] == "onset" for r in selected),
                            stable_windows=sum(r["cohort"] == "stable" for r in selected)))
    c.publish_identical(root / "collection.json", io.seal(dict(schema=SCHEMA, plan_fingerprint=plan["fingerprint"],
                                                               cases=cases, support=support)))
    artifacts["collection.json"] = c.file_hash(root / "collection.json")
    per_case = 1 + len(fixed.POOL) * LONG
    progress.total = progress.done + len(cases) * per_case + 1
    progress.report("Replaying matched crop candidates and scoring earlier box motion")
    rows, choices, availability = [], [], {}
    probe_calls, tail_calls = 0, 0
    rt = tracker.rt
    factor, size = float(rt.tracker.params.search_factor), int(rt.tracker.params.search_size)
    entries = {r["sequence"]: r for r in clips}
    for case in cases:
        key, seq, frame, end = (case[k] for k in ("id", "sequence", "frame", "end"))
        entry, archive, truth = entries[seq], references[seq], labels[seq]
        previous_boxes = archive["boxes"][frame - HISTORY - 1:frame - 1]
        np.testing.assert_array_equal(previous_boxes[-1], archive["parent"][frame - 1])

        def read(f):
            return checked_frame(dataset, entry["frames"][f])

        def capture():
            image = read(frame)
            pool, parents, _ = candidates(case, previous_boxes, image.shape[:2], factor, size)
            with ExitStack() as stack:
                observer = crop_stage.FixedCenterObserver(tracker, stack)
                return dict(crop_stage.observe_crops(observer, read(0), truth[0], image, parents, frame),
                            names=np.asarray(list(pool)), image_hw=np.asarray(image.shape[:2]))

        observed, fresh = cached(root, plan, "probe", key, artifacts, capture)
        if not fresh:
            progress.reused += 1
        progress.advance()
        pool, parents, probes = candidates(case, previous_boxes, observed["image_hw"], factor, size)
        names = list(pool)
        np.testing.assert_array_equal(observed["names"], names)
        np.testing.assert_array_equal(observed["parent"], parents)
        np.testing.assert_array_equal(observed["layer"], [geometry_stage.LAYER] * len(pool))
        if archive["layer"][frame - 1] == geometry_stage.LAYER:
            np.testing.assert_array_equal(observed["boxes"][names.index("current")], archive["boxes"][frame - 1])
        for i, a in enumerate(names):
            probes[key, a]["hann_peak"] = float(observed["hann_peak"][i])
        raw_boxes = dict(zip(names, observed["boxes"], strict=True))
        options, agreement, errors = choose(case, pool, probes, raw_boxes, previous_boxes)
        probe_calls += int(observed["calls"])

        template = None

        def compute(anchors):
            nonlocal template
            if template is None:
                template = read(0)
            with ExitStack() as stack:
                observer = online.RoutedObserver(tracker, stack)
                return source.continue_from(observer, read, template, truth[0], frame, end, anchors, progress)

        def replay():
            saved, _, physical = crop_stage.continue_paths(observed["boxes"], [archive["boxes"][frame - 1:end]],
                                                           LONG, compute, progress)
            return dict(saved, physical=np.asarray(physical))

        saved, fresh = cached(root, plan, "tail", key, artifacts, replay)
        if not fresh:
            progress.reused += len(pool) * LONG
            progress.advance(count=len(pool) * LONG)
        progress.advance(count=(len(fixed.POOL) - len(pool)) * LONG)
        tail_calls += int(saved["physical"])
        traced = saved["paths"][saved["inverse"]]
        if traced.shape != (len(pool), LONG + 1, 4) or not np.isfinite(traced).all() or (traced[..., 2:] <= 0.).any():
            raise ValueError(f"Invalid continuation cache at {key}")
        np.testing.assert_array_equal(traced[:, 0], observed["boxes"])
        paths = np.concatenate((archive["boxes"][None, frame - 1:end], traced), axis=0)
        result = source.tail_summary(paths, truth[frame:end + 1], ious[seq][:frame - 1], 0, range(1, len(pool) + 1))
        labelled = {}
        for a, row in zip(names, result, strict=True):
            labelled[a] = dict(case_id=key, group=case["group"], cohort=case["cohort"], arm=a, distinct_crops=len(pool),
                               anchor_iou=float(online.tracking_iou(raw_boxes[a], truth[frame])),
                               hann_peak=probes[key, a]["hann_peak"], crop_agreement=agreement[a],
                               history_center_error=errors[a], aliases="|".join(pool[a]), **row)
            rows.append(labelled[a])
        availability[key] = any(r["protected_rescue"] is True for r in labelled.values())
        choices.extend(dict(labelled[arm], method=method) for method, arm in options.items())
    summary, paired = summarize(cases, choices, availability)
    tables = dict(cases=cases, support=support, candidates=rows, choices=choices, summary=summary, paired=paired)
    return tables, artifacts, dict(probe_calls=probe_calls, required_tail_calls=tail_calls)


def run(args, root, paths):
    r76, r89, r93, r170, r187 = paths
    d170 = c.c66.sealed(r170 / "stage170_decision.json", previous.SCHEMA)
    d187 = c.c66.sealed(r187 / "stage187_decision.json", fixed.SCHEMA)
    if d170["fingerprint"] != source.event.matched.SOURCE170 or d187["fingerprint"] != SOURCE187:
        raise ValueError("Need the sealed Stage-170 development roster and Stage-187 rule comparison")
    s95 = base.s103.prior.s95
    _, d93, _, _, origin = s95.s94.load_run(r93, r76, r89)
    if d93["fingerprint"] != SOURCE93:
        raise ValueError("Unexpected Stage-93 development roster")
    recent = risk.read_verified(r93 / "analysis", d93, 93, "video_summary")
    groups93 = {risk.SOURCE_GROUP(r["sequence"]) for r in recent}
    groups170 = {risk.SOURCE_GROUP(r["sequence"]) for r in risk.read_verified(r170, d170, 170, "video_summary")}
    if groups170 != groups93:
        raise ValueError("Stage-93 and Stage-170 source groups differ")
    roster = c.c66.sealed(ROSTER, confirmation.SCHEMA)
    wanted = eligible_groups(roster, recent)
    old = origin["original"]
    dataset = Path(old["root"]).resolve()
    records = confirmation.inventory(dataset)
    if fingerprint(records) != roster["inventory_fingerprint"]:
        raise ValueError("VisDrone inventory differs from the sealed Stage-79 inventory")
    existing = root / "manifest.json"
    if existing.exists():
        saved = c.c66.sealed(existing, SCHEMA)
        clips = saved["clips"]
    else:
        print("[Stage-188] Binding image and annotation hashes for additional development sources", flush=True)
        clips = freeze_sequences(dataset, records, wanted)
    if not clips or any(r["group"] not in wanted or risk.SOURCE_GROUP(r["sequence"]) != r["group"] for r in clips):
        raise ValueError("No permitted additional developmental sequences")

    import torch

    torch.set_num_threads(1)
    runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
    differences = base.s109.check_runtime(d170["runtime"], runtime)
    code = (Path(__file__), Path(crop_stage.__file__), Path(fixed.__file__), Path(source.__file__))
    plan = io.seal(dict(schema=SCHEMA, source_stage187=SOURCE187, source_stage170=d170["fingerprint"], source_stage93=SOURCE93,
                        confirmation_roster=ROSTER_SHA, clips=clips, eligible_groups=sorted(wanted), methods=list(METHODS),
                        horizon=LONG, causal_history=HISTORY, runtime=runtime, runtime_differences=differences,
                        code_sha256={p.name: c.file_hash(p) for p in code}))
    c.publish_identical(root / "manifest.json", plan)
    for n in ("native", "probe", "tail"):
        (root / n).mkdir(exist_ok=True)
    work = sum(r["tracked_frames"] for r in clips) + len(clips) * 2 * (1 + len(fixed.POOL) * LONG) + 1
    progress = OverallProgress("Stage-188", work, interval=args.progress_interval, unit="work")
    progress.report("Loading frozen SGLA")
    args.parameter = runtime["tracker"]["parameter"]
    tracker, _ = base.s103.s76.startup.checked_runtime(args, dict(old, fingerprint=plan["fingerprint"],
                     runtime_identity=runtime, runtime=runtime["tracker"]), root)
    tables, artifacts, counts = evaluate(root, plan, clips, tracker, dataset, progress)
    tracker.assert_frozen()
    if progress.done != progress.total - 1:
        raise ValueError("Incomplete Stage-188 progress accounting")
    return dict(source_stage187=SOURCE187, plan_fingerprint=plan["fingerprint"], additional_groups=len(clips),
                historical_groups_outside_recent_development=sorted(wanted),
                native_tracking_calls=sum(r["tracked_frames"] for r in clips), windows=len(tables["cases"]),
                confirmation_groups_opened=0, training_performed=False, work=progress.total, runtime=runtime,
                runtime_differences=differences, artifact_sha256=artifacts,
                table_sha256=gain.prior.publish(root, tables, stage="stage188"), **counts), progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster-audit", type=Path, metavar="STAGE93_BUNDLE")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--progress-interval", type=float, default=15)
    parser.set_defaults(mode="stage188_temporal_support")
    for stage in SOURCES:
        parser.add_argument(f"--stage{stage}-root", type=Path)
    args = parser.parse_args()
    paths = [p.expanduser().resolve() for stage in SOURCES if (p := getattr(args, f"stage{stage}_root"))]
    if args.roster_audit:
        if paths or args.output_dir or args.checkpoint:
            parser.error("--roster-audit is a standalone CPU roster check")
        groups = eligible_groups(c.c66.sealed(ROSTER, confirmation.SCHEMA), archived_stage93_roster(args.roster_audit))
        print(f"{len(groups)} previously exposed groups outside the recent 28-group development set and confirmation set")
        print(" ".join(sorted(groups)))
        return
    if len(paths) != len(SOURCES) or args.output_dir is None or args.checkpoint is None or args.progress_interval <= 0:
        parser.error("Need all source directories, original checkpoint and output directory")
    root = args.output_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate output directory")
    with gain.prior.prior.writer_lock(root):
        out = root / "stage188_decision.json"
        if out.exists():
            saved = c.c66.sealed(out, SCHEMA)
            plan = c.c66.sealed(root / "manifest.json", SCHEMA)
            if saved["source_stage187"] != SOURCE187 or saved["plan_fingerprint"] != plan["fingerprint"]:
                raise ValueError("Stage-188 completed run has another source or plan")
            io.verify({str(root / k): h for k, h in (saved["artifact_sha256"] | saved["table_sha256"]).items()})
            OverallProgress("Stage-188", saved["work"], saved["work"]).report("Complete")
            return
        result, progress = run(args, root, paths)
        c.publish_identical(out, io.seal(dict(schema=SCHEMA, status="COMPLETE_STAGE188_TEMPORAL_SUPPORT_DIAGNOSTIC", **result)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
