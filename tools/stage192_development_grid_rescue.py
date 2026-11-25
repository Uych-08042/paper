"""Measure grid supply and one-writeback effects across frozen development sources."""

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
import stage191_grid_writeback as paired
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, inside, read_boxes


probe, old, c, io, online = paired.probe, paired.old, paired.c, paired.io, paired.online
SCHEMA = "stage192_development_grid_rescue_v1"
SOURCE170 = old.source.event.matched.SOURCE170
SOURCE191 = "f033641b48174b1b8a42fb945ad947cc5d1dd082db58be5b44d7986bf4733db3"
SOURCES = (76, 89, 93, 121, 170, 188, 191)
LONG, GOOD = paired.LONG, probe.GOOD
PILOT = {"uav0000173", "uav0000107"}


def window_roster(sequence, group, cohort, values):
    """Set all anchors from the recorded native trajectory, before forward passes."""
    failures = online.failure_events(values)
    onsets = [int(e["start"] + 1) for e in failures]
    eligible = [f for f in onsets if f + LONG <= len(values) and (values[f - 1:f + LONG] >= 0.).all()]
    onset = eligible[0] if eligible else None
    stable = [f for f in range(1, len(values) - LONG + 1)
              if (onset is None or f + LONG < onset or f > onset + LONG)
              and (values[f - 1:f + LONG] >= GOOD).all()]
    control = (min(stable, key=lambda f: (abs(f - onset), f)) if onset is not None else stable[len(stable) // 2]) if stable else None
    rows = [dict(id=f"{sequence}__{kind}__f{f:06d}", sequence=sequence, group=group, cohort=cohort,
                 window=kind, frame=f, end=f + LONG, has_matched_control=onset is not None and control is not None)
            for kind, f in (("onset", onset), ("stable", control)) if f is not None]
    return rows, dict(sequence=sequence, group=group, cohort=cohort, native_events=len(onsets),
                      eligible_onsets=len(eligible), eligible_stable_starts=len(stable),
                      onset_frame=onset, stable_frame=control)


def load_sources(paths):
    d170 = c.c66.sealed(paths[170] / "stage170_decision.json", old.previous.SCHEMA)
    p170 = c.c66.sealed(paths[170] / "manifest.json", old.previous.SCHEMA)
    d188 = c.c66.sealed(paths[188] / "stage188_decision.json", old.SCHEMA)
    p188 = c.c66.sealed(paths[188] / "manifest.json", old.SCHEMA)
    d191 = c.c66.sealed(paths[191] / "stage191_decision.json", paired.SCHEMA)
    if (d170["fingerprint"] != SOURCE170 or d188["fingerprint"] != paired.prior.SOURCE188 or d191["fingerprint"] != SOURCE191
            or d191["source_stage190"] != paired.SOURCE190 or d170["runtime"] != d188["runtime"]
            or d170["plan_fingerprint"] != p170["fingerprint"] or d188["plan_fingerprint"] != p188["fingerprint"]
            or p170["runtime"] != d170["runtime"] or p188["runtime"] != d188["runtime"]):
        raise ValueError("Need the completed Stage-170, Stage-188 and Stage-191 sources")
    _, _, jobs = old.source.prior.history.native_jobs(paths[121])
    s95 = old.base.s103.prior.s95
    p93, d93, p89, d89, origin = s95.s94.load_run(paths[93], paths[76], paths[89])
    if d93["fingerprint"] != old.SOURCE93:
        raise ValueError("Unexpected Stage-93 source")
    dataset = Path(origin["original"]["root"]).resolve()
    entries = {}
    original = {r["sequence"] for r in old.risk.read_verified(paths[170], d170, 170, "video_summary") if r["arm"] == "native"}
    for seq in sorted(original):
        entry = s95.load_unit(paths[93], paths[89], p93, p89, d89, origin, jobs[seq]["spec"])[0]
        if entry["sequence"] != seq or len(entry["frames"]) != len(entry["ground_truth"]):
            raise ValueError(f"Unexpected Stage-170 source video {seq}")
        native = old.previous.archive_trace(paths[170], p170, d170, seq, "native")
        entries[seq] = (entry["frames"], np.asarray(entry["ground_truth"], float), native, checked_image, "recent_development")
    for entry in p188["clips"]:
        seq = entry["sequence"]
        annotation = inside(dataset, entry["annotation"])
        if c.file_hash(annotation) != entry["annotation_sha256"] or seq in entries:
            raise ValueError(f"Unexpected Stage-188 source video {seq}")
        gt = read_boxes(annotation)
        native = paired.prior.read_cache(paths[188], d188, "native", seq)
        entries[seq] = (entry["frames"], gt, native, old.checked_frame, "additional_development")
    groups = {old.risk.SOURCE_GROUP(seq) for seq in entries}
    confirmation = c.c66.sealed(old.ROSTER, old.confirmation.SCHEMA)
    held = {r["source_group_proxy"] for r in confirmation["sequences"]}
    if len(original) != 28 or len(entries) != 38 or len(groups) != 38 or groups & held:
        raise ValueError("Need 38 distinct development groups outside the confirmation set")
    return entries, dataset, d170["runtime"]


def cohort_summary(rows):
    report = []
    for scope in ("all", "exclude_stage191", "recent_development", "additional_development"):
        for window in ("onset", "stable"):
            items = [r for r in rows if r["window"] == window and (scope == "all" or r["cohort"] == scope
                     or (scope == "exclude_stage191" and r["group"] not in PILOT))]
            valid = [r for r in items if r["valid"]]
            delta = sorted(r["delta128_or_shorter"] for r in valid)
            report.append(dict(scope=scope, window=window, windows=len(items), valid=len(valid),
                               groups=len({r["group"] for r in valid}), grid_good=sum(r["grid_good"] for r in items),
                               hann_good=sum(r["hann_good"] for r in items), readout_miss=sum(r["readout_miss"] for r in items),
                               changed=sum(r["changed"] for r in items), protected_rescues=sum(r["protected_rescue"] for r in valid),
                               event_harms=sum(r["event_harm"] for r in valid), added_failure_windows=sum(r["added_failure"] > 0 for r in valid),
                               tail_improvements=sum(x > 1e-9 for x in delta), tail_declines=sum(x < -1e-9 for x in delta),
                               mean_delta=float(np.mean(delta)) if delta else None,
                               without_best_group=float(np.mean(delta[:-1])) if len(delta) > 1 else None))
    return report


def evaluate(root, plan, entries, dataset, tracker, values, progress):
    tables = dict(anchors=[], paired=[], branches=[])
    artifacts, calls = {}, 0
    exact = not plan["runtime_differences"]
    for case in plan["cases"]:
        key, seq, frame, end = (case[k] for k in ("id", "sequence", "frame", "end"))
        files, gt, native, reader, _ = entries[seq]

        def read(f):
            return reader(dataset, files[f])

        template = read(0)
        with ExitStack() as stack:
            observer = old.crop_stage.center.CenterObserver(tracker, stack)
            data = observer.observe(template, read(frame), gt[0], native["parent"][frame - 1], frame)
        data["search_factor"] = tracker.rt.tracker.params.search_factor
        anchors, measure = paired.select_boxes(data, gt[frame])
        baseline = native["boxes"][frame - 1:end]
        if exact:
            np.testing.assert_array_equal(anchors[0], baseline[0])
            np.testing.assert_equal(data["layer"], native["layer"][frame - 1])
        progress.advance()
        known = [baseline] if exact else []

        def compute(boxes):
            with ExitStack() as stack:
                runner = online.RoutedObserver(tracker, stack)
                return old.source.continue_from(runner, read, template, gt[0], frame, end, boxes, progress)

        def collect():
            saved, _, required = old.crop_stage.continue_paths(anchors, known, LONG, compute, progress)
            return dict(saved, calls=np.asarray(required))

        saved, fresh = old.cached(root, plan, "tail", key, artifacts, collect)
        if not fresh:
            old.crop_stage.continue_paths(anchors, known, LONG, compute, progress, saved)
        calls += int(saved["calls"])
        label = dict(case, grid_good=measure["best_grid_iou"] >= GOOD, hann_good=measure["selected_iou"] >= GOOD,
                     readout_miss=measure["readout_miss"], changed=not np.array_equal(*anchors))
        tables["anchors"].append(dict(case, native_layer=int(data["layer"]), **measure))
        pair, branches, _, _ = paired.summarize(label, saved["paths"][saved["inverse"]], gt[frame:end + 1],
                                                values[seq][:frame - 1], baseline)
        tables["paired"].append(pair)
        tables["branches"].extend(branches)
    tables["summary"] = cohort_summary(tables["paired"])
    return tables, artifacts, calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (*(f"stage{i}-root" for i in SOURCES), "checkpoint", "output-dir"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.set_defaults(mode="stage192_development_grid_rescue")
    args = parser.parse_args()
    paths = {i: getattr(args, f"stage{i}_root").expanduser().resolve() for i in SOURCES}
    root = args.output_dir.expanduser().resolve()
    if any(root == p or p in root.parents or root in p.parents for p in paths.values()):
        parser.error("Use a separate Stage-192 output directory")
    with writer_lock(root):
        out = root / "stage192_decision.json"
        if out.exists():
            done = c.c66.sealed(out, SCHEMA)
            io.verify({str(root / k): v for k, v in (done["table_sha256"] | done["artifact_sha256"]).items()})
            return
        entries, dataset, expected = load_sources(paths)
        if root == dataset or dataset in root.parents or root in dataset.parents:
            parser.error("Use an output directory outside the source dataset")
        cases, support, values = [], [], {}
        for seq, (files, gt, trace, _, cohort) in sorted(entries.items()):
            if len(files) != len(gt):
                raise ValueError(f"Source frame count changed at {seq}")
            old.previous.audit(trace, len(gt) - 1, gt[0], "native", 0.)
            values[seq] = online.tracking_iou(trace["boxes"], gt[1:])
            rows, coverage = window_roster(seq, old.risk.SOURCE_GROUP(seq), cohort, values[seq])
            cases.extend(rows)
            support.append(coverage)

        import torch

        torch.set_num_threads(1)
        runtime = old.base.s103.s76.device_runtime(dict(runtime_identity=expected), torch, args.gpu)
        differences = old.base.s109.check_runtime(expected, runtime)
        plan = io.seal(dict(schema=SCHEMA, source_stage170=SOURCE170, source_stage188=paired.prior.SOURCE188,
                            source_stage191=SOURCE191, runtime=runtime, runtime_differences=differences,
                            cases=cases, support=support, horizon=LONG,
                            code_sha256={Path(f).name: c.file_hash(Path(f)) for f in (__file__, paired.__file__, probe.__file__)}))
        c.publish_identical(root / "manifest.json", plan)
        (root / "tail").mkdir(exist_ok=True)
        progress = OverallProgress("Stage-192", len(cases) * (1 + 2 * LONG) + 1, unit="work")
        progress.report("Loading frozen SGLA")
        args.parameter = runtime["tracker"]["parameter"]
        tracker, _ = old.base.s103.s76.startup.checked_runtime(args, dict(fingerprint=plan["fingerprint"],
                                                                 runtime_identity=runtime, runtime=runtime["tracker"]), root)
        progress.report("Testing grid writeback on the frozen development roster")
        tables, artifacts, calls = evaluate(root, plan, entries, dataset, tracker, values, progress)
        if progress.done != progress.total - 1:
            raise ValueError("Incomplete Stage-192 progress accounting")
        tracker.assert_frozen()
        hashes = old.gain.prior.publish(root, dict(support=support, **tables), stage="stage192")
        c.publish_identical(out, io.seal(dict(schema=SCHEMA, status="COMPLETE_STAGE192_DEVELOPMENT_GRID_RESCUE",
             plan_fingerprint=plan["fingerprint"], runtime_differences=differences, source_groups=len(entries),
             windows=len(cases), anchor_calls=len(cases), required_tail_calls=calls, confirmation_groups_opened=0,
             training_performed=False, table_sha256=hashes, artifact_sha256=artifacts)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
