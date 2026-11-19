"""Replay two frozen Hann runner-up gates and native SGLA on complete videos."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import stage135_peak_gate as gate
from stage27_uav_common import checked_image, failure_events, success_auc, tracking_iou
from stage27_uav_summary import write_table


prior, c, io = gate.prior, gate.c, gate.io
base = prior.base
SCHEMA = "stage136_same_crop_peak_closed_loop_v1"
STATUS = "COMPLETE_PEAK_GATE_CLOSED_LOOP_DEVELOPMENT"
SOURCE135 = "c496707edcedc327ce3b78fe7552c3d80b1c6a42f82cd33530093e5d4a857e18"
ARMS = (gate.NATIVE, gate.GATES[0], gate.PRIMARY)
DESIGN = dict(gates=gate.DESIGN, low_iou=.20, long_failure_frames=32,
              bootstrap_seed=23136, bootstrap_samples=20000)
TABLES = ("policy_summary", "paired", "video_summary", "failures", "switches")
EVENT_COLUMNS = ("sequence", "cohort", "arm", "onset", "confirmation", "end", "recovery", "duration", "censor")
SWITCH_COLUMNS = ("sequence", "cohort", "arm", "frame", "first_peak_iou_own_crop", "runner_iou",
                  "relative_score_gap", "pair_iou", "history_advantage")
METRICS = ("auc", "low_iou_fraction", "failure_events", "longest_failure_fraction", "long_events",
           "unrecovered_events", "switched_fraction")


def decide(row, ap, arm, previous, earlier):
    response = row["raw"] * np.asarray(ap["window"], np.float32)
    first = int(np.argmax(response))
    native = np.asarray(row["boxes"], float)
    cells = np.array([first, first], int)
    boxes = np.stack((native, native))
    if arm != gate.NATIVE:
        grid = prior.dense_boxes(row["size"], row["offset"], previous, row["factor"], ap["search_size"], row["image_hw"])
        grid[first] = native
        peaks = prior.peak_candidates(response, grid)
        if len(peaks) > 1:
            cells[1] = peaks[1]
            boxes[1] = grid[cells[1]]
    evidence = gate.online_evidence(row["raw"], ap["window"], boxes, cells, previous, earlier)
    switched = arm != gate.NATIVE and gate.gate(evidence)[arm]
    return dict(cells=cells, candidates=boxes, switched=switched, boxes=boxes[int(switched)].copy(), **evidence)


def collect(observer, frames, initial_box, ap, arm, image_reader=checked_image, observe=base.e63.actor.observe):
    if arm not in ARMS:
        raise ValueError(f"Unknown Stage-136 arm: {arm}")
    observer.initialize(image_reader(ap["root"], frames[0]), initial_box)
    previous, earlier = np.asarray(initial_box, float).copy(), None
    rows = []
    for frame, item in enumerate(frames[1:], 1):
        observed = observe(observer, image_reader(ap["root"], item), previous, frame, ap)
        choice = decide(observed, ap, arm, previous, earlier)
        calls = {k: observed[k] for k in ("head_calls", "selector_calls", "block_calls", "layer")}
        rows.append(dict(choice, frame_id=frame, parent=previous.copy(), **calls))
        previous, earlier = choice["boxes"], previous
        if frame % 500 == 0:
            print(f"  [{arm}] {frame}/{len(frames)-1} frames", flush=True)
    return {key: np.asarray([r[key] for r in rows]) for key in rows[0]}


def audit_trace(data, frames, initial_box, arm):
    np.testing.assert_array_equal(data["frame_id"], np.arange(1, frames + 1))
    boxes, candidates = data["boxes"], data["candidates"]
    if (boxes.shape != (frames, 4) or candidates.shape != (frames, 2, 4)
            or not np.isfinite(candidates).all() or (candidates[..., 2:] <= 0).any()):
        raise ValueError("Invalid or incomplete Stage-136 boxes")
    np.testing.assert_array_equal(data["parent"], np.vstack((initial_box, boxes[:-1])))
    selected = np.array([arm != gate.NATIVE and gate.gate(dict(
        distinct=bool(data["distinct"][i]), relative_score_gap=float(data["relative_score_gap"][i]),
        pair_iou=float(data["pair_iou"][i]), history_advantage=float(data["history_advantage"][i])))[arm]
        for i in range(frames)])
    np.testing.assert_array_equal(data["switched"], selected)
    np.testing.assert_array_equal(boxes, candidates[np.arange(frames), selected.astype(int)])
    base.e63.source_common.prior.audit_calls(data, (frames,))


def first_run(mask):
    ids = np.flatnonzero(mask)
    if not ids.size:
        return None, 0
    runs = np.split(ids, np.where(np.diff(ids) != 1)[0] + 1)
    return int(ids[0] + 1), max(map(len, runs))


def report_video(tables, entry, traces, cohort):
    native = traces[gate.NATIVE]
    truth = np.asarray(entry["ground_truth"], float)[1:]
    native_events = failure_events(tracking_iou(native["boxes"], truth))
    for arm, trace in traces.items():
        selected_iou = tracking_iou(trace["boxes"], truth)
        events = failure_events(selected_iou)
        lengths = [r["end"] - r["start"] + 1 for r in events]
        first, consecutive = first_run(trace["switched"])
        prefix = len(truth) if first is None else first - 1
        np.testing.assert_array_equal(trace["boxes"][:prefix], native["boxes"][:prefix])
        diverged = np.flatnonzero(np.any(trace["boxes"] != native["boxes"], axis=1))
        if (int(diverged[0]) + 1 if diverged.size else None) != first:
            raise ValueError("First divergence differs from first change of candidate")
        ident = dict(sequence=entry["sequence"], cohort=cohort, arm=arm)
        switches = int(trace["switched"].sum())
        frames = len(truth)
        tables["video_summary"].append(dict(
            ident, frames=frames, physical_calls=int(trace["head_calls"].sum()),
            auc=success_auc(selected_iou), low_iou_fraction=float((selected_iou < DESIGN["low_iou"]).mean()),
            failure_events=len(events), longest_failure_fraction=max(lengths, default=0) / frames,
            long_events=sum(n >= DESIGN["long_failure_frames"] for n in lengths),
            unrecovered_events=sum(r["recovery"] is None for r in events), native_stable=not native_events,
            new_failure_on_native_stable=not native_events and bool(events), switched_frames=switches,
            switched_fraction=switches / frames, first_switch_frame=first, longest_consecutive_switches=consecutive))
        for event, length in zip(events, lengths):
            recovery = event["recovery"]
            tables["failures"].append(dict(ident, onset=event["start"] + 1, confirmation=event["confirmation"] + 1,
                                           end=event["end"] + 1, recovery=None if recovery is None else recovery + 1,
                                           duration=length, censor=event["censor"]))
        for i in np.flatnonzero(trace["switched"]):
            first_iou = float(tracking_iou(trace["candidates"][i, 0], truth[i]))
            tables["switches"].append(dict(ident, frame=int(i + 1), first_peak_iou_own_crop=first_iou,
                                          runner_iou=float(selected_iou[i]),
                                          relative_score_gap=float(trace["relative_score_gap"][i]),
                                          pair_iou=float(trace["pair_iou"][i]),
                                          history_advantage=float(trace["history_advantage"][i])))


def summarize(rows):
    lookup = {(r["sequence"], r["arm"]): r for r in rows}
    videos = sorted({r["sequence"] for r in rows})
    summary, paired = [], []
    for arm in ARMS:
        data = [lookup[v, arm] for v in videos]
        summary.append(dict(arm=arm, videos=len(videos), physical_calls=sum(r["physical_calls"] for r in data),
                            native_stable_videos=sum(r["native_stable"] for r in data),
                            new_failure_on_native_stable=sum(r["new_failure_on_native_stable"] for r in data),
                            worst_auc_delta_on_native_stable=min(
                                (r["auc"] - lookup[r["sequence"], gate.NATIVE]["auc"] for r in data if r["native_stable"]),
                                default=None),
                            switched_frames=sum(r["switched_frames"] for r in data),
                            **{m: float(np.mean([r[m] for r in data])) for m in METRICS}))
    for arm, ref in ((ARMS[1], ARMS[0]), (ARMS[2], ARMS[0]), (ARMS[2], ARMS[1])):
        for metric in ("auc", "low_iou_fraction", "failure_events", "longest_failure_fraction", "unrecovered_events"):
            values = [lookup[v, arm][metric] - lookup[v, ref][metric] for v in videos]
            paired.append(base.s103.contrast(values, arm + "_minus_" + ref, metric, DESIGN,
                                             family=metric == "auc" and ref == gate.NATIVE))
    return summary, paired


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    report = c.c66.sealed(root / "stage136_decision.json", SCHEMA)
    if (plan["design"] != DESIGN or report["plan_fingerprint"] != plan["fingerprint"]
            or report["source_stage135"] != SOURCE135 or report["status"] != STATUS
            or report["new_search_calls"] != plan["budget"] or report["new_full_trajectories"] != len(ARMS) * len(plan["jobs"])
            or len(report["artifact_sha256"]) != 2 * report["new_full_trajectories"]
            or set(report["table_sha256"]) != {f"stage136_{k}.csv" for k in TABLES}):
        raise ValueError("Incomplete Stage-136 closed-loop run")
    io.verify({str(root / name): digest for name, digest in (report["table_sha256"] | report["artifact_sha256"]).items()})


def run(root, r121, p121, d121, plan, load_entry, observer, ap):
    tables, artifacts = {k: [] for k in TABLES}, {}
    evaluation = root / "evaluation"
    evaluation.mkdir(exist_ok=True)
    for job in plan["jobs"]:
        entry = load_entry(job["spec"])
        seq, frames = entry["sequence"], job["frames"]
        if seq != job["spec"]["unit"]["sequence"] or len(entry["frames"]) != frames + 1:
            raise ValueError("Stage-121 native video or frame count changed")
        traces = {}
        with base.e63.source_experiment.installed(observer, [], job["spec"]["unit"]) as head:
            if head != p121["native_head_sha256"]:
                raise ValueError("Native SGLA head changed")
            for arm in ARMS:
                path = evaluation / f"{seq}__{arm}.npz"
                meta = dict(plan_fingerprint=plan["fingerprint"], sequence=seq, arm=arm)
                trace = c.read_cache(path, meta)
                if trace is None:
                    print(f"[Stage-136] running {seq} {arm}, {frames} frames", flush=True)
                    trace = collect(observer, entry["frames"], entry["initial_box"], ap, arm)
                    audit_trace(trace, frames, entry["initial_box"], arm)
                    c.save(path, trace, meta)
                else:
                    audit_trace(trace, frames, entry["initial_box"], arm)
                for item in (path, path.with_suffix(".receipt.json")):
                    artifacts[item.relative_to(root).as_posix()] = c.file_hash(item)
                traces[arm] = trace
            if not plan["runtime_differences"]:
                key = "evaluation/" + base.job_key(job)
                io.verify({str(r121 / n): d121["artifact_sha256"][n] for n in (key + ".npz", key + ".receipt.json")})
                source = prior.source.prior.prior.load_trace(r121, p121, job)
                np.testing.assert_allclose(traces[gate.NATIVE]["boxes"], source["boxes"],
                                           atol=ap["design"]["box_atol"], rtol=0)
        report_video(tables, entry, traces, job["spec"]["cohort"])
    tables["policy_summary"], tables["paired"] = summarize(tables["video_summary"])
    calls = sum(r["physical_calls"] for r in tables["video_summary"])
    if calls != plan["budget"]:
        raise ValueError("Stage-136 search budget changed")
    hashes = {}
    empty_columns = {"failures": EVENT_COLUMNS, "switches": SWITCH_COLUMNS}
    for name, rows in tables.items():
        path = root / f"stage136_{name}.csv"
        write_table(path, rows, fields=empty_columns.get(name))
        hashes[path.name] = c.file_hash(path)
    c.publish_identical(root / "stage136_decision.json", io.seal(dict(
        schema=SCHEMA, status=STATUS, role="consumed_development_full_closed_loop",
        plan_fingerprint=plan["fingerprint"], source_stage135=SOURCE135, source_stage121=prior.source.SOURCE121,
        runtime=plan["runtime"], runtime_differences=plan["runtime_differences"], videos=len(plan["jobs"]),
        new_full_trajectories=len(tables["video_summary"]), new_search_calls=calls,
        confirmation_inference_calls=0, training_updates=0, table_sha256=hashes, artifact_sha256=artifacts)))
    check_results(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76", "stage89", "stage93", "stage121", "stage135")
    for name in names:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage136", parameter="deit_distilled")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not args.checkpoint or any(not getattr(args, k) for k in names):
        parser.error("Supply the frozen source directories and original checkpoint")
    r76, r89, r93, r121, r135 = paths = [getattr(args, k).expanduser().resolve() for k in names]
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate Stage-136 directory")
    if not (r135 / "stage135_decision.json").is_file():
        parser.error("Need the completed Stage-135 server audit")
    s121 = prior.source.s121
    p121 = c.c66.sealed(r121 / "manifest.json", s121.SCHEMA)
    d121 = c.c66.sealed(r121 / "analysis/stage121_decision.json", s121.SCHEMA)
    d135 = c.c66.sealed(r135 / "stage135_decision.json", gate.SCHEMA)
    if (d135["fingerprint"] != SOURCE135 or d135["design"] != gate.DESIGN or d135["status"] != gate.STATUS
            or d121["fingerprint"] != prior.source.SOURCE121 or d121["plan_fingerprint"] != p121["fingerprint"]
            or d135["source_stage121"] != d121["fingerprint"]):
        raise ValueError("Need the archived Stage-135 design and Stage-121 development source")
    jobs = sorted((j for j in p121["jobs"] if j["arm"] == gate.NATIVE), key=base.job_key)
    frames = sum(j["frames"] for j in jobs)
    if len(jobs) != 28 or len({j["spec"]["unit"]["sequence"] for j in jobs}) != 28 or frames != 29047:
        raise ValueError("Need the 28 complete Stage-121 development videos")
    with prior.writer_lock(root):
        if (root / "stage136_decision.json").is_file():
            check_results(root)
            return
        s95 = base.s103.prior.s95
        p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
        old = origin["original"]

        def load_entry(spec):
            return s95.load_unit(r93, r89, p93, p89, d89, origin, spec)[0]

        import torch
        torch.set_num_threads(1)
        runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
        differences = base.s109.check_runtime(p121["runtime"], runtime)
        manifest = root / "manifest.json"
        if manifest.is_file() and c.c66.sealed(manifest, SCHEMA)["runtime"] != runtime:
            raise ValueError("GPU/runtime changed on resume. Set STAGE136_ROOT to a new directory.")
        plan = io.seal(dict(schema=SCHEMA, design=DESIGN, jobs=jobs, arms=list(ARMS),
                            source_stage135=SOURCE135, source_stage121=prior.source.SOURCE121,
                            source_roster_fingerprints=[p93["fingerprint"], p89["fingerprint"], old["fingerprint"]],
                            runtime=runtime, runtime_differences=differences, budget=frames * len(ARMS),
                            code_sha256={Path(m.__file__).name: c.file_hash(Path(m.__file__)) for m in (gate, prior)}
                            | {Path(__file__).name: c.file_hash(Path(__file__))}))
        c.publish_identical(manifest, plan)
        frozen = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = base.s103.s76.startup.checked_runtime(args, frozen, root)
        observer = base.e63.Capture(native)
        try:
            run(root, r121, p121, d121, plan, load_entry, observer, base.e63.c.actor_plan(old))
        finally:
            observer.handle.remove()
    print(f"Stage-136: {STATUS}", flush=True)


if __name__ == "__main__":
    main()
