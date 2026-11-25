"""Measure the closed-loop effect of committing the best available CENTER box once."""

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
import stage190_successor_readout as probe
from stage27_uav_collect import writer_lock


prior, old, c, io, online = probe.prior, probe.old, probe.c, probe.io, probe.online
SCHEMA = "stage191_grid_writeback_v1"
SOURCE190 = "6567d6ed4f07724d05ab21094ab7fd777b72fda3c6c582b1ab0c018baaeea9b9"
ARMS = ("hann", "grid_best_gt")
LONG = old.LONG


def archive_paths(root, decision, key, trace, frame, inputs):
    tail = prior.read_cache(root, decision, "tail", key)
    source = prior.read_cache(root, decision, "probe", key)
    paths = tail["paths"][tail["inverse"]]
    np.testing.assert_array_equal(paths[:, 0], source["boxes"])
    names = list(source["names"])
    if key == probe.PRIMARY:
        # Stage-188 starts one frame earlier on this sequence.
        mapping = {name: paths[names.index(name), 1:] for name, _, _ in inputs if name != "native"}
        mapping["native"] = trace["boxes"][frame - 1:frame + LONG]
    else:
        mapping = {name: paths[names.index(name)] for name, _, _ in inputs}
    for name, _, ref in inputs:
        np.testing.assert_array_equal(mapping[name][0], ref)
    return mapping


def select_boxes(data, target):
    summary = probe.measure(data, target)
    grid = probe.dense_boxes(data["size"], data["offset"], data["parent"], data["factor"],
                            data["search_size"], data["image_hw"])
    hann, best = summary["selected_cell"], summary["best_grid_cell"]
    grid[hann] = data["boxes"]
    boxes = np.stack((grid[hann], grid[best]))
    np.testing.assert_allclose(online.tracking_iou(boxes, target),
                               [summary["selected_iou"], summary["best_grid_iou"]], rtol=0., atol=1e-12)
    return boxes, summary


def summarize(identity, paths, target, prefix, archive):
    values = online.tracking_iou(paths, target[None])
    if (values < 0.).any():
        raise ValueError(f"Invalid continuation at {identity}")
    pair = old.source.tail_summary(paths, target, prefix, 0, [1])[0]
    flags = old.source.event.event_labels(prefix, values)
    pair.update(identity, prevented_terminal=flags["prevented_terminal"], rescued_recovery=flags["rescued_recovery"],
                lost_recovery=flags["lost_recovery"], anchor_gain=float(values[1, 0] - values[0, 0]),
                control_archive_overlap_frames=len(archive),
                control_archive_max_abs_error=float(np.max(np.abs(paths[0, :len(archive)] - archive))))
    frame = identity["frame"]
    branches, events, frames = [], [], []
    for arm, series in zip(ARMS, values, strict=True):
        detail, observed = prior.local_metrics(series, prefix, values[0], frame)
        detail["tail_delta_hann"] = detail.pop("tail_delta_native")
        detail["higher_than_hann_successors"] = detail.pop("higher_than_native_successors")
        label = dict(identity, arm=arm)
        branches.append(dict(label, **detail))
        events.extend(dict(label, onset=e["start"] + 1, confirmation=e["confirmation"] + 1, end=e["end"] + 1,
                           recovery=None if e["recovery"] is None else e["recovery"] + 1,
                           censor="window_end" if e["censor"] == "video_end" else e["censor"]) for e in observed)
        frames.extend(dict(label, observed_frame=frame + j, iou=float(iou), delta_hann=float(iou - values[0, j]))
                      for j, iou in enumerate(series))
    return pair, branches, events, frames


def evaluate(target, plan, source_root, source_receipt, manifest, source_rows, tracker, dataset, jobs, exact, progress):
    tables = dict(anchors=[], paired=[], branches=[], events=[], frames=[])
    clips = {r["sequence"]: r for r in manifest["clips"]}
    artifacts = {}
    tail_calls = 0
    for job in jobs:
        key, seq, frame, gt, inputs = (job[k] for k in ("key", "sequence", "frame", "gt", "inputs"))
        end = frame + LONG
        clip = clips[seq]
        if end >= len(gt):
            raise ValueError("Need a complete 128-frame continuation")
        trace = prior.read_cache(source_root, source_receipt, "native", seq)
        archive = archive_paths(source_root, source_receipt, key, trace, frame, inputs)

        def read(f):
            return old.checked_frame(dataset, clip["frames"][f])

        template, image = read(0), read(frame)
        anchors = []
        for branch, parent, _ in inputs:
            row = source_rows[seq, frame, branch]
            np.testing.assert_array_equal(parent, [float(row[f"parent_{field}"]) for field in ("x", "y", "w", "h")])
            with ExitStack() as stack:
                observer = probe.FixedRoute(tracker, stack)
                observer.fixed_layer = int(row["fixed_layer"])
                data = observer.observe(template, image, gt[0], parent, frame)
            if data["layer"] != observer.fixed_layer or data["head_calls"] != 1 or data["crop_calls"] != 1:
                raise ValueError("Expected one search at the Stage-190 fixed layer")
            data["search_factor"] = tracker.rt.tracker.params.search_factor
            boxes, info = select_boxes(data, gt[frame])
            stage190_match = all(str(info[k]) == row[k] for k in ("selected_cell", "best_grid_cell", "selected_iou", "best_grid_iou"))
            if exact and not stage190_match:
                raise ValueError(f"Stage-190 grid or output changed for {key}/{branch}")
            anchors.append(boxes)
            tables["anchors"].append(dict(sequence=seq, frame=frame, parent_branch=branch, layer=observer.fixed_layer,
                                           stage190_match=stage190_match, **info))
            progress.advance()
        anchors = np.asarray(anchors).reshape(-1, 4)

        def compute(initial):
            with ExitStack() as stack:
                observer = online.RoutedObserver(tracker, stack)
                return old.source.continue_from(observer, read, template, gt[0], frame, end, initial, progress)

        def collect():
            data, _, required = old.crop_stage.continue_paths(anchors, [], LONG, compute, progress)
            return dict(data, calls=np.asarray(required))

        if exact:
            for i, (branch, _, _) in enumerate(inputs):
                np.testing.assert_allclose(anchors[2 * i], archive[branch][0], atol=probe.BOX_ATOL, rtol=0.)
        cached, fresh = old.cached(target, plan, "tail", key, artifacts, collect)
        if not fresh:
            old.crop_stage.continue_paths(anchors, [], LONG, compute, progress, cached)
        tail_calls += int(cached["calls"])
        paths = cached["paths"][cached["inverse"]].reshape(len(inputs), len(ARMS), LONG + 1, 4)
        target_boxes = gt[frame:end + 1]
        native_prefix = online.tracking_iou(trace["boxes"][:frame - 1], gt[1:frame])
        for i, (branch, parent, _) in enumerate(inputs):
            baseline = archive[branch]
            prefix = native_prefix.copy()
            if key == probe.PRIMARY:
                prefix[-1] = online.tracking_iou(parent, gt[frame - 1])
            label = dict(sequence=seq, frame=frame, parent_branch=branch)
            paired, branches, events, frames = summarize(label, paths[i], target_boxes, prefix, baseline)
            # Compare the common observed suffix, including the archived anchor.
            if exact and paired["control_archive_max_abs_error"] > probe.BOX_ATOL:
                raise ValueError(f"Archived Hann continuation changed for {key}/{branch}")
            for k, val in zip(("paired", "branches", "events", "frames"), ([paired], branches, events, frames), strict=True):
                tables[k].extend(val)
    return tables, artifacts, tail_calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = (188, 189, 190)
    for key in (*(f"stage{i}-root" for i in stages), "dataset-root", "checkpoint", "output-dir"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.set_defaults(mode="stage191_grid_writeback")
    args = parser.parse_args()
    roots = {i: getattr(args, f"stage{i}_root").expanduser().resolve() for i in stages}
    dataset, target = (p.expanduser().resolve() for p in (args.dataset_root, args.output_dir))
    if any(target == p or p in target.parents or target in p.parents for p in (*roots.values(), dataset)):
        parser.error("Use a separate Stage-191 output directory")
    schemas = (old.SCHEMA, prior.SCHEMA, probe.SCHEMA)
    decisions = {i: c.c66.sealed(roots[i] / f"stage{i}_decision.json", schema) for i, schema in zip(stages, schemas, strict=True)}
    for i, fingerprint in zip(stages, (prior.SOURCE188, probe.SOURCE189, SOURCE190), strict=True):
        if decisions[i]["fingerprint"] != fingerprint:
            raise ValueError(f"Unexpected Stage-{i} result")
    m188, m190 = (c.c66.sealed(roots[i] / "manifest.json", schema) for i, schema in ((188, old.SCHEMA), (190, probe.SCHEMA)))
    if (m188["fingerprint"] != decisions[188]["plan_fingerprint"] or m190["fingerprint"] != decisions[190]["plan_fingerprint"]
            or m190["runtime"] != m188["runtime"]):
        raise ValueError("Need the original same-runtime Stage-188 and Stage-190 archives")
    rows = old.risk.read_verified(roots[190], decisions[190], 190, "arms")
    source_rows = {(r["sequence"], int(r["frame"]), r["branch"]): r for r in rows}
    with writer_lock(target):
        out = target / "stage191_decision.json"
        if out.exists():
            done = c.c66.sealed(out, SCHEMA)
            if done["source_stage190"] != SOURCE190:
                raise ValueError("Existing output is bound to another Stage-190 result")
            io.verify({str(target / k): v for k, v in (done["table_sha256"] | done["artifact_sha256"]).items()})
            return
        jobs = probe.jobs(roots[188], dataset, m188, decisions[188], roots[189], decisions[189])
        if len(rows) != 6 or set(source_rows) != {(j["sequence"], j["frame"], n) for j in jobs for n, _, _ in j["inputs"]}:
            raise ValueError("Unexpected Stage-190 branches")

        import torch

        torch.set_num_threads(1)
        runtime = old.base.s103.s76.device_runtime(dict(runtime_identity=m190["runtime"]), torch, args.gpu)
        differences = old.base.s109.check_runtime(m190["runtime"], runtime)
        plan = io.seal(dict(schema=SCHEMA, source_stage190=SOURCE190, runtime=runtime, runtime_differences=differences,
                            horizon=LONG, code_sha256={Path(f).name: c.file_hash(Path(f)) for f in (__file__, probe.__file__)}))
        c.publish_identical(target / "manifest.json", plan)
        (target / "tail").mkdir(exist_ok=True)
        args.parameter = runtime["tracker"]["parameter"]
        progress = OverallProgress("Stage-191", len(rows) * (1 + len(ARMS) * LONG) + 1, unit="work")
        progress.report("Loading frozen SGLA")
        tracker, _ = old.base.s103.s76.startup.checked_runtime(args, dict(fingerprint=plan["fingerprint"],
                                                                 runtime_identity=runtime, runtime=runtime["tracker"]), target)
        progress.report("Continuing matched Hann and grid-best boxes")
        tables, artifacts, calls = evaluate(target, plan, roots[188], decisions[188], m188, source_rows,
                                            tracker, dataset, jobs, not differences, progress)
        if progress.done != progress.total - 1:
            raise ValueError("Incomplete continuation accounting")
        tracker.assert_frozen()
        hashes = old.gain.prior.publish(target, tables, stage="stage191")
        c.publish_identical(out, io.seal(dict(schema=SCHEMA, status="COMPLETE_STAGE191_GRID_WRITEBACK_DIAGNOSTIC",
            source_stage190=SOURCE190, plan_fingerprint=plan["fingerprint"], runtime_differences=differences,
            pairs=len(rows), anchor_calls=len(rows), required_tail_calls=calls, training_performed=False,
            table_sha256=hashes, artifact_sha256=artifacts)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
