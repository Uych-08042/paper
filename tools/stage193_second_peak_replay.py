"""Replay a separated Hann runner-up on the frozen Stage-192 development windows."""

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
import stage192_development_grid_rescue as prior
from stage27_uav_collect import writer_lock
from stage28_response_common import peak_candidates


paired, probe, old, c, io, online = prior.paired, prior.probe, prior.old, prior.c, prior.io, prior.online
SCHEMA = "stage193_second_peak_replay_v1"
SOURCE192 = "0d39f60a743eb0d683f9b5a14693286574e6b55db82d7f5e128cefbe15905c41"
SOURCES = (*prior.SOURCES, 192)
ARMS = ("hann", "second_peak", "grid_best_gt")
LONG, GOOD = prior.LONG, prior.GOOD


def second_peak(data, first_cell):
    """Use the original Stage-28 spatial peak pool. Inputs are prediction-only."""
    scores = data["raw"] * data["window"]
    grid = probe.dense_boxes(data["size"], data["offset"], data["parent"], data["factor"],
                            data["search_size"], data["image_hw"])
    grid[first_cell] = data["boxes"]
    peaks = peak_candidates(scores, grid)
    np.testing.assert_equal(peaks[0], first_cell)
    cell = peaks[1] if len(peaks) > 1 else first_cell
    flat = scores.ravel()
    return grid[cell], dict(has_second=len(peaks) > 1, second_cell=cell, second_score=float(flat[cell]),
                           second_to_first_score=float(flat[cell] / flat[first_cell]) if len(peaks) > 1 else None)


def summaries(rows):
    report = []
    for scope in ("all", "exclude_stage191", "recent_development", "additional_development"):
        for window in ("onset", "stable"):
            sample = [r for r in rows if r["window"] == window and (scope == "all" or r["cohort"] == scope
                      or (scope == "exclude_stage191" and r["group"] not in prior.PILOT))]
            valid = [r for r in sample if r["valid"]]
            delta = sorted(r["delta128_or_shorter"] for r in valid)
            report.append(dict(scope=scope, window=window, windows=len(sample), valid=len(valid),
                               second_available=sum(r["has_second"] for r in sample), changed=sum(r["changed"] for r in sample),
                               hann_good=sum(r["hann_good"] for r in sample), second_good=sum(r["second_good"] for r in sample),
                               grid_good=sum(r["grid_good"] for r in sample),
                               second_rescues=sum(r["protected_rescue"] for r in valid),
                               grid_rescues=sum(r["grid_rescue"] for r in valid),
                               shared_rescues=sum(r["protected_rescue"] and r["grid_rescue"] for r in valid),
                               second_only_rescues=sum(r["protected_rescue"] and not r["grid_rescue"] for r in valid),
                               event_harms=sum(r["event_harm"] for r in valid),
                               added_failure_windows=sum(r["added_failure"] > 0 for r in valid),
                               tail_improvements=sum(x > 1e-9 for x in delta), tail_declines=sum(x < -1e-9 for x in delta),
                               mean_delta=float(np.mean(delta)) if delta else None,
                               without_best_group=float(np.mean(delta[:-1])) if len(delta) > 1 else None,
                               mean_vs_grid=float(np.mean([r["delta128_or_shorter"] - r["grid_delta128"] for r in valid])) if delta else None))
    return report


def reference_paths(source, decision, case, anchors):
    key = case["id"]
    data = paired.prior.read_cache(source, decision, "tail", key)
    paths = data["paths"][data["inverse"]]
    if paths.shape != (2, LONG + 1, 4):
        raise ValueError(f"Unexpected Stage-192 reference paths at {key}")
    np.testing.assert_array_equal(paths[:, 0], anchors[[0, 2]])
    return list(paths)


def evaluate(root, plan, entries, dataset, tracker, source, decision, source_anchors, source_pairs, progress):
    tables = dict(anchors=[], paired=[], branches=[])
    artifacts, calls = {}, 0
    same_hardware = not plan["runtime_differences"]
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
        control, measure = paired.select_boxes(data, gt[frame])
        box, secondary = second_peak(data, measure["selected_cell"])
        anchors = np.stack((control[0], box, control[1]))
        source_anchor = source_anchors[key]
        copied = all(str(measure[k]) == source_anchor[k] for k in ("selected_cell", "best_grid_cell", "selected_iou", "best_grid_iou"))
        if same_hardware and (not copied or int(data["layer"]) != int(source_anchor["native_layer"])):
            raise ValueError(f"Stage-192 anchor did not replay at {key}")
        progress.advance()
        known = reference_paths(source, decision, case, anchors) if same_hardware else []

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
        paths = saved["paths"][saved["inverse"]]
        prefix = online.tracking_iou(native["boxes"][:frame - 1], gt[1:frame])
        target = gt[frame:end + 1]
        baseline = native["boxes"][frame - 1:end]
        summary = old.source.tail_summary(paths, target, prefix, 0, [1, 2])
        # Keep Stage-192 and Stage-193 metrics directly comparable.
        published = source_pairs[key]
        if same_hardware and any(str(v) != published[k] for k, v in summary[1].items()):
            raise ValueError(f"Stage-192 oracle reference metrics changed at {key}")
        second_iou = float(online.tracking_iou(box, gt[frame]))
        label = dict(case, has_second=secondary["has_second"], changed=not np.array_equal(anchors[0], anchors[1]),
                     hann_good=measure["selected_iou"] >= GOOD, second_good=second_iou >= GOOD,
                     grid_good=measure["best_grid_iou"] >= GOOD)
        pair, branches, _, _ = paired.summarize(label, paths[:2], target, prefix, baseline)
        pair.update(grid_rescue=summary[1]["protected_rescue"], grid_delta128=summary[1]["delta128_or_shorter"],
                    second_also_grid_best=bool(np.array_equal(anchors[1], anchors[2])))
        tables["anchors"].append(dict(case, native_layer=int(data["layer"]), stage192_anchor_match=copied,
                                      second_iou=second_iou, **secondary, **measure))
        tables["paired"].append(pair)
        for record, arm in zip(branches, ARMS[:2], strict=True):
            tables["branches"].append(dict(record, arm=arm))
        metrics, _ = paired.prior.local_metrics(online.tracking_iou(paths[2], target), prefix,
                                                online.tracking_iou(paths[0], target), frame)
        metrics["tail_delta_hann"] = metrics.pop("tail_delta_native")
        metrics["higher_than_hann_successors"] = metrics.pop("higher_than_native_successors")
        tables["branches"].append(dict(label, arm=ARMS[2], **metrics))
    tables["summary"] = summaries(tables["paired"])
    return tables, artifacts, calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (*(f"stage{i}-root" for i in SOURCES), "checkpoint", "output-dir"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.set_defaults(mode="stage193_second_peak_replay")
    args = parser.parse_args()
    paths = {i: getattr(args, f"stage{i}_root").expanduser().resolve() for i in SOURCES}
    source = paths[192]
    root = args.output_dir.expanduser().resolve()
    if any(root == p or p in root.parents or root in p.parents for p in paths.values()):
        parser.error("Use a separate Stage-193 output directory")
    decision = c.c66.sealed(source / "stage192_decision.json", prior.SCHEMA)
    archived = c.c66.sealed(source / "manifest.json", prior.SCHEMA)
    if decision["fingerprint"] != SOURCE192 or decision["plan_fingerprint"] != archived["fingerprint"]:
        raise ValueError("Need the completed Stage-192 result")
    anchors, pairs = ({r["id"]: r for r in old.risk.read_verified(source, decision, 192, name)}
                      for name in ("anchors", "paired"))
    cases = archived["cases"]
    if not (len(cases) == 52 and sum(r["window"] == "onset" for r in cases) == 16
            and set(anchors) == set(pairs) == {r["id"] for r in cases} and archived["horizon"] == LONG):
        raise ValueError("Need the frozen 16 onset and 36 stable windows")
    with writer_lock(root):
        out = root / "stage193_decision.json"
        if out.exists():
            done = c.c66.sealed(out, SCHEMA)
            if done["source_stage192"] != SOURCE192:
                raise ValueError("Existing output is bound to another Stage-192 result")
            io.verify({str(root / k): v for k, v in (done["table_sha256"] | done["artifact_sha256"]).items()})
            return
        entries, dataset, _ = prior.load_sources(paths)
        if root == dataset or dataset in root.parents or root in dataset.parents:
            parser.error("Use an output directory outside the source dataset")

        import torch

        torch.set_num_threads(1)
        runtime = old.base.s103.s76.device_runtime(dict(runtime_identity=archived["runtime"]), torch, args.gpu)
        differences = old.base.s109.check_runtime(archived["runtime"], runtime)
        plan = io.seal(dict(schema=SCHEMA, source_stage192=SOURCE192, cases=cases, arms=list(ARMS),
                            runtime=runtime, runtime_differences=differences, horizon=LONG,
                            code_sha256={Path(f).name: c.file_hash(Path(f)) for f in (__file__, prior.__file__, peak_candidates.__code__.co_filename)}))
        c.publish_identical(root / "manifest.json", plan)
        (root / "tail").mkdir(exist_ok=True)
        progress = OverallProgress("Stage-193", len(cases) * (1 + len(ARMS) * LONG) + 1, unit="work")
        progress.report("Loading frozen SGLA")
        args.parameter = runtime["tracker"]["parameter"]
        tracker, _ = old.base.s103.s76.startup.checked_runtime(args, dict(fingerprint=plan["fingerprint"],
                                                                 runtime_identity=runtime, runtime=runtime["tracker"]), root)
        progress.report("Replaying matched Hann and second-peak writebacks")
        tables, artifacts, calls = evaluate(root, plan, entries, dataset, tracker, source, decision, anchors, pairs, progress)
        if progress.done != progress.total - 1:
            raise ValueError("Incomplete Stage-193 progress accounting")
        tracker.assert_frozen()
        hashes = old.gain.prior.publish(root, tables, stage="stage193")
        c.publish_identical(out, io.seal(dict(schema=SCHEMA, status="COMPLETE_STAGE193_SECOND_PEAK_REPLAY",
             source_stage192=SOURCE192, plan_fingerprint=plan["fingerprint"], runtime_differences=differences,
             windows=len(cases), anchor_calls=len(cases), required_tail_calls=calls, confirmation_groups_opened=0,
             training_performed=False, table_sha256=hashes, artifact_sha256=artifacts)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
