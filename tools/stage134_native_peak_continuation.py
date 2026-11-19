"""Replace one box from a shared SGLA search and replay the native continuation."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import stage133_native_failure_audit as audit
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, failure_events, tracking_iou
from stage28_response_common import dense_boxes, peak_candidates
from stage46_shift_audit import publish_table


base, source, c, io = audit.base, audit.source, audit.c, audit.io
read_table = source.prior.source.read_table
SCHEMA = "stage134_native_peak_continuation_v1"
STATUS = "COMPLETE_NATIVE_PEAK_CONTINUATION_DIAGNOSTIC"
SOURCE133 = "b748935be8765678a92156c9000ee79fd0e308bc951ed29a19790a5b7f23e814"
DESIGN = dict(successor_frames=32, good_iou=.5, low_iou=.2, good_run=3)
ARMS = ("native", "hann_runner_up", "peak_pool_oracle", "dense_oracle")
TABLES = ("summary", "windows", "scope", "timeline")


def healthy_frame(ious, horizon):
    """Choose the complete healthy window whose anchor is closest to video midpoint."""
    if len(ious) <= horizon:
        return None
    healthy = np.lib.stride_tricks.sliding_window_view(ious >= DESIGN["good_iou"], horizon + 1).all(axis=1)
    candidates = np.flatnonzero(healthy) + 1
    if not candidates.size:
        return None
    return int(min(candidates, key=lambda f: (abs(2 * f - len(ious) - 1), f)))


def choices(observation, truth, ap):
    raw = observation["raw"]
    weighted = raw * np.asarray(ap["window"], np.float32)
    grid = dense_boxes(observation["size"], observation["offset"], observation["parent"],
                       observation["factor"], ap["search_size"], observation["image_hw"])
    native = int(np.argmax(weighted))
    grid[native] = observation["boxes"]
    hann_peaks = peak_candidates(weighted, grid)
    raw_peaks = peak_candidates(raw, grid)
    peak_pool = np.array(sorted(set(hann_peaks + raw_peaks)), int)
    iou = tracking_iou(grid, truth)

    def best(cells):
        cell = int(cells[np.argmax(iou[cells])])
        return native if iou[native] == iou[cell] else cell

    cells = np.array([native, hann_peaks[1] if len(hann_peaks) > 1 else native,
                      best(peak_pool), best(np.arange(len(grid)))])
    return dict(cells=cells, boxes=grid[cells], iou=iou[cells], peak_pool=peak_pool,
                hann_peaks=np.asarray(hann_peaks), raw_peaks=np.asarray(raw_peaks))


def freeze_cases(job, entry, trace, events):
    sequence = entry["sequence"]
    frames = len(entry["frames"]) - 1
    truth = np.asarray(entry["ground_truth"], float)
    if (sequence != job["spec"]["unit"]["sequence"] or frames != job["frames"]
            or truth.shape != (frames + 1, 4)):
        raise ValueError(f"Frozen native entry differs: {sequence}")
    np.testing.assert_array_equal(trace["frame_id"], np.arange(1, frames + 1))
    iou = tracking_iou(trace["boxes"], truth[1:])
    onsets = [int(r["onset_frame"]) for r in events]
    if onsets != [e["start"] + 1 for e in failure_events(iou)]:
        raise ValueError(f"Stage-133 onset inventory changed: {sequence}")
    horizon = DESIGN["successor_frames"]
    stable = healthy_frame(iou, horizon)
    anchors = [(f, "onset") for f in onsets]
    if stable is not None:
        anchors.append((stable, "healthy"))
    cases = []
    for frame, kind in anchors:
        if frames - frame < horizon:
            raise ValueError(f"Need {horizon} successor frames for {sequence} frame {frame}")
        cases.append(dict(case_id=f"{sequence}__{kind}__f{frame:06d}", sequence=sequence, kind=kind,
                          frame=frame, successors=horizon, source_anchor_iou=float(iou[frame - 1])))
    return cases


def collect(observer, case, entry, parent, ap, image_reader=checked_image, observe=base.e63.actor.observe):
    first = case["frame"]
    frame_ids = np.arange(first, first + case["successors"] + 1)
    frames = entry["frames"]
    truth = np.asarray(entry["ground_truth"])[first]
    observer.initialize(image_reader(ap["root"], frames[0]), entry["initial_box"])
    anchor = observe(observer, image_reader(ap["root"], frames[first]), parent, first, ap)
    selected = choices(anchor, truth, ap)
    fields = ("boxes", "parent", "crop", "crop_sha256", "head_calls", "selector_calls", "block_calls", "layer")
    paths = [[dict(anchor, boxes=box.copy())] for box in selected["boxes"]]
    for frame in frame_ids[1:]:
        image = image_reader(ap["root"], frames[frame])
        for path in paths:
            # Capture restores frame_id and this branch's previous box before every forward.
            path.append(observe(observer, image, path[-1]["boxes"].copy(), int(frame), ap))
    data = {k: np.asarray([[r[k] for r in path] for path in paths]) for k in fields}
    return dict(data, frame_id=frame_ids, cells=selected["cells"], peak_pool=selected["peak_pool"],
                hann_peaks=selected["hann_peaks"], raw_peaks=selected["raw_peaks"],
                anchor_raw=anchor["raw"], anchor_size=anchor["size"], anchor_offset=anchor["offset"],
                anchor_factor=np.asarray(anchor["factor"]), anchor_hw=anchor["image_hw"])


def verify_case(data, case, entry, trace, ap, same_runtime):
    first = case["frame"]
    count = case["successors"] + 1
    i = first - 1
    shape = (len(ARMS), count)
    np.testing.assert_array_equal(data["frame_id"], np.arange(first, first + count))
    if (data["boxes"].shape != (*shape, 4) or not np.isfinite(data["boxes"]).all()
            or (data["boxes"][..., 2:] <= 0).any()):
        raise ValueError("Incomplete or invalid native branch")
    parent = trace["candidate_parents"][i, 0]
    for key in ("parent", "crop", "crop_sha256"):
        source_key = "candidate_parents" if key == "parent" else key
        for v in data[key][:, 0]:
            np.testing.assert_array_equal(v, trace[source_key][i, 0], err_msg=key)
    np.testing.assert_array_equal(data["anchor_hw"], trace["image_hw"][i])
    np.testing.assert_array_equal(data["parent"][:, 1:], data["boxes"][:, :-1])
    base.e63.source_common.prior.audit_calls(data, shape)
    observation = dict(raw=data["anchor_raw"], size=data["anchor_size"], offset=data["anchor_offset"],
                       parent=parent, factor=float(data["anchor_factor"]), image_hw=data["anchor_hw"],
                       boxes=data["boxes"][0, 0])
    selected = choices(observation, entry["ground_truth"][first], ap)
    for name in ("cells", "peak_pool", "hann_peaks", "raw_peaks"):
        np.testing.assert_array_equal(data[name], selected[name], err_msg=name)
    np.testing.assert_array_equal(data["boxes"][:, 0], selected["boxes"])
    archived = trace["boxes"][i:i + count]
    source_error = float(np.abs(data["boxes"][0] - archived).max())
    if same_runtime:
        np.testing.assert_allclose(data["boxes"][0], archived, rtol=0, atol=ap["design"]["box_atol"])
        archived_cell = np.argmax(trace["raw"][i, 0] * np.asarray(ap["window"], np.float32))
        if int(data["cells"][0]) != int(archived_cell):
            raise ValueError("Same-device native anchor chose a different peak")
    return source_error


def describe(data, case, entry):
    truth = np.asarray(entry["ground_truth"], float)[data["frame_id"]]
    ious = tracking_iou(data["boxes"], truth[None])
    rows, timeline = [], []
    native_recovers = False
    native_tail_low = int((ious[0, 1:] < DESIGN["low_iou"]).sum())
    for k, arm in enumerate(ARMS):
        iou = ious[k]
        tail = iou[1:]
        length = DESIGN["good_run"]
        good = (np.flatnonzero(np.lib.stride_tricks.sliding_window_view(tail >= DESIGN["good_iou"], length).all(axis=1))
                if len(tail) >= length else np.empty(0, int))
        recovers = bool(good.size)
        if k == 0:
            native_recovers = recovers
        tail_low = int((tail < DESIGN["low_iou"]).sum())
        rows.append(dict(case, arm=arm, cell=int(data["cells"][k]), anchor_iou=float(iou[0]),
                         anchor_delta=float(iou[0] - ious[0, 0]),
                         active=not np.array_equal(data["boxes"][k, 0], data["boxes"][0, 0]),
                         tail_mean_iou=float(tail.mean()), tail_mean_delta=float((tail - ious[0, 1:]).mean()),
                         tail_low_frames=tail_low, tail_low_frame_delta=tail_low - native_tail_low,
                         new_tail_low=tail_low > 0 and native_tail_low == 0,
                         tail_good_frames=int((tail >= DESIGN["good_iou"]).sum()),
                         first_tail_good_run=int(data["frame_id"][good[0] + 1]) if recovers else None,
                         new_tail_good_run=recovers and not native_recovers,
                         lost_tail_good_run=native_recovers and not recovers))
        timeline.extend(dict(case_id=case["case_id"], sequence=case["sequence"], arm=arm,
                             frame=int(frame), offset=i, iou=float(iou[i]))
                        for i, frame in enumerate(data["frame_id"]))
    return rows, timeline


def summarize(windows):
    groups = defaultdict(list)
    for row in windows:
        groups[row["kind"], row["arm"]].append(row)
        if row["kind"] == "onset" and row["live_native_low_anchor"]:
            groups["onset_live_low", row["arm"]].append(row)
    result = []
    for (kind, arm), rows in sorted(groups.items()):
        videos = defaultdict(list)
        for row in rows:
            videos[row["sequence"]].append(row["tail_mean_delta"])
        values = [float(np.mean(window_deltas)) for window_deltas in videos.values()]
        result.append(dict(kind=kind, arm=arm, cases=len(rows), videos=len(videos),
                           active_cases=sum(r["active"] for r in rows),
                           matched_live_low_anchor_cases=sum(r["live_native_low_anchor"] for r in rows),
                           anchor_good_cases=sum(r["anchor_iou"] >= DESIGN["good_iou"] for r in rows),
                           tail_cases_with_good_run=sum(r["first_tail_good_run"] is not None for r in rows),
                           new_tail_good_runs=sum(r["new_tail_good_run"] for r in rows),
                           lost_tail_good_runs=sum(r["lost_tail_good_run"] for r in rows),
                           new_tail_low_cases=sum(r["new_tail_low"] for r in rows),
                           mean_tail_low_frame_delta=float(np.mean([r["tail_low_frame_delta"] for r in rows])),
                           mean_anchor_delta=float(np.mean([r["anchor_delta"] for r in rows])),
                           video_mean_tail_iou_delta=float(np.mean(values)),
                           tail_improved_cases=sum(r["tail_mean_delta"] > 0 for r in rows),
                           tail_degraded_cases=sum(r["tail_mean_delta"] < 0 for r in rows)))
    return result


def check_results(root):
    plan = c.c66.sealed(root / "manifest.json", SCHEMA)
    report = c.c66.sealed(root / "stage134_decision.json", SCHEMA)
    if (plan["design"] != DESIGN or report["plan_fingerprint"] != plan["fingerprint"]
            or report["status"] != STATUS or plan["source_stage133"] != SOURCE133
            or report["new_search_calls"] != plan["budget"]
            or report["cases"] != len(plan["cases"])
            or len(report["artifact_sha256"]) != 2 * len(plan["cases"])
            or set(report["table_sha256"]) != {f"stage134_{k}.csv" for k in TABLES}):
        raise ValueError("Incomplete Stage-134 diagnostic")
    bindings = report["table_sha256"] | report["artifact_sha256"]
    io.verify({str(root / k): checksum for k, checksum in bindings.items()})


def run(root, plan, contexts, observer, ap):
    windows, timeline, scope, artifacts = [], [], [], {}
    branch_dir = root / "branches"
    branch_dir.mkdir(exist_ok=True)
    for case in plan["cases"]:
        job, entry, trace = contexts[case["sequence"]]
        parent = trace["candidate_parents"][case["frame"] - 1, 0]
        path = branch_dir / f"{case['case_id']}.npz"
        key = dict(plan_fingerprint=plan["fingerprint"], case_id=case["case_id"])
        verifier = (case, entry, trace, ap, not plan["runtime_differences"])
        data = c.read_cache(path, key)
        if data is None:
            with base.e63.source_experiment.installed(observer, [], job["spec"]["unit"]) as head:
                if head != plan["native_head_sha256"]:
                    raise ValueError("Frozen native SGLA head differs")
                data = collect(observer, case, entry, parent, ap)
                error = verify_case(data, *verifier)
            c.save(path, data, key)
        else:
            error = verify_case(data, *verifier)
        for item in (path, path.with_suffix(".receipt.json")):
            artifacts[item.relative_to(root).as_posix()] = c.file_hash(item)
        rows, frames = describe(data, case, entry)
        live_low = rows[0]["anchor_iou"] < DESIGN["low_iou"]
        windows.extend(dict(r, live_native_low_anchor=live_low) for r in rows)
        timeline.extend(frames)
        scope.append(dict(case, live_native_low_anchor=live_low, live_native_anchor_iou=rows[0]["anchor_iou"],
                          anchor_peak_pool_size=len(data["peak_pool"]), native_source_max_box_error=error,
                          physical_search_calls=1 + int(data["head_calls"][:, 1:].sum()),
                          logical_search_calls=int(data["head_calls"].sum())))
        print(f"[Stage-134] {case['case_id']}: {case['successors']} successor frames", flush=True)
    calls = sum(r["physical_search_calls"] for r in scope)
    if calls != plan["budget"]:
        raise ValueError("Physical search budget differs")
    hashes = {}
    for name, rows in zip(TABLES, (summarize(windows), windows, scope, timeline)):
        path = root / f"stage134_{name}.csv"
        publish_table(path, rows)
        hashes[path.name] = c.file_hash(path)
    c.publish_identical(root / "stage134_decision.json", io.seal(dict(
        schema=SCHEMA, status=STATUS, plan_fingerprint=plan["fingerprint"],
        role="consumed_development_one_step_oracle_and_runner_up_diagnostic",
        source_stage133=SOURCE133, runtime=plan["runtime"], runtime_differences=plan["runtime_differences"],
        cases=len(scope), onset_cases=sum(r["kind"] == "onset" for r in scope),
        healthy_cases=sum(r["kind"] == "healthy" for r in scope),
        new_search_calls=calls, logical_search_calls=sum(r["logical_search_calls"] for r in scope),
        training_updates=0, new_full_trajectories=0, confirmation_inference_calls=0,
        table_sha256=hashes, artifact_sha256=artifacts)))
    check_results(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("stage76", "stage89", "stage93", "stage121", "stage133")
    for name in names:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage134", parameter="deit_distilled")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if not args.checkpoint or not all(getattr(args, k) for k in names):
        parser.error("Supply the original checkpoint and frozen source directories")
    r76, r89, r93, r121, r133 = paths = [getattr(args, k).expanduser().resolve() for k in names]
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate Stage-134 directory")
    if not (r133 / "stage133_decision.json").is_file():
        parser.error("Need the completed Stage-133 server audit")
    with writer_lock(root):
        if (root / "stage134_decision.json").is_file():
            check_results(root)
            return
        audit.check_results(r133)
        d133 = c.c66.sealed(r133 / "stage133_decision.json", audit.SCHEMA)
        p121 = c.c66.sealed(r121 / "manifest.json", source.s121.SCHEMA)
        d121 = c.c66.sealed(r121 / "analysis/stage121_decision.json", source.s121.SCHEMA)
        if (d133["fingerprint"] != SOURCE133 or d121["fingerprint"] != source.SOURCE121
                or d121["plan_fingerprint"] != p121["fingerprint"]):
            raise ValueError("Need the archived Stage-133 and Stage-121 results")
        io.verify({str(r121 / k): v for k, v in d133["native_artifact_sha256"].items()})
        events = defaultdict(list)
        for row in read_table(r133 / "stage133_events.csv"):
            events[row["sequence"]].append(row)
        jobs = sorted((j for j in p121["jobs"] if j["arm"] == "native"), key=base.job_key)
        if len(jobs) != audit.DESIGN["videos"]:
            raise ValueError("Need the complete development roster")
        s95 = base.s103.prior.s95
        p93, _, p89, report89, origin = s95.s94.load_run(r93, r76, r89)
        old = origin["original"]
        ap = base.e63.c.actor_plan(old)
        cases, contexts = [], {}
        for job in jobs:
            spec = job["spec"]
            entry = s95.load_unit(r93, r89, p93, p89, report89, origin, spec)[0]
            trace = source.prior.prior.load_trace(r121, p121, job)
            local = freeze_cases(job, entry, trace, events[entry["sequence"]])
            if local:
                cases.extend(local)
                contexts[entry["sequence"]] = job, entry, trace
        if sum(case["kind"] == "onset" for case in cases) != audit.DESIGN["events"]:
            raise ValueError("Stage-134 failure cohort changed")

        import torch
        torch.set_num_threads(1)
        runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
        diffs = base.s109.check_runtime(p121["runtime"], runtime)
        manifest = root / "manifest.json"
        if manifest.is_file() and c.c66.sealed(manifest, SCHEMA)["runtime"] != runtime:
            raise ValueError("GPU/runtime changed on resume. Set STAGE134_ROOT to a new directory.")
        budget = sum(1 + len(ARMS) * r["successors"] for r in cases)
        plan = io.seal(dict(schema=SCHEMA, design=DESIGN, source_stage133=SOURCE133,
                            source_stage121=source.SOURCE121, arms=list(ARMS), cases=cases, budget=budget,
                            code_sha256=c.file_hash(Path(__file__)), native_head_sha256=p121["native_head_sha256"],
                            runtime=runtime, runtime_differences=diffs))
        c.publish_identical(manifest, plan)
        frozen = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        native, _ = base.s103.s76.startup.checked_runtime(args, frozen, root)
        observer = base.e63.Capture(native)
        try:
            run(root, plan, contexts, observer, ap)
        finally:
            observer.handle.remove()


if __name__ == "__main__":
    main()
