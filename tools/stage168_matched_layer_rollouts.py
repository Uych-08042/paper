"""Replay single and repeated layer interventions from matched Stage-167 states."""

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from experiment_progress import OverallProgress
import stage167_online_layers as online


gate, gain, history, veto, c, io = online.gate, online.gain, online.history, online.veto, online.c, online.io
base, LAYERS, TRACKING = online.base, online.LAYERS, online.TRACKING
SCHEMA = "stage168_matched_layer_rollouts_v1"
SOURCE167 = "ce4232a7f7a3774866fd526c89ebf267d42c5b0078fcc3188e931d4f2a464cbb"
SOURCES = (*online.SOURCES, 167)
NATIVE, FIXED, PAIRWISE = online.ARMS
BRANCHES = {
    "native": (NATIVE, NATIVE),
    "original_once": (FIXED, NATIVE),
    "pairwise_once": (PAIRWISE, NATIVE),
    "original_continued": (FIXED, FIXED),
    "pairwise_continued": (PAIRWISE, PAIRWISE),
}
DESIGN = dict(lookback=32, successors=64, reference_grace=32)
TARGETS = (
    ("0304_original_loss", "uav0000304", FIXED, "onset"),
    ("0304_pairwise_loss", "uav0000304", PAIRWISE, "onset"),
    ("0080_loss", "uav0000080", FIXED, "onset"),
    ("0089_recovery", "uav0000089", FIXED, "recovery"),
    ("0176_loss", "uav0000176", FIXED, "onset"),
    ("0176_recovery", "uav0000176", FIXED, "recovery"),
)


def windows(decision, videos, failures):
    if (decision["status"] != online.STATUS or decision["source_stage166"] != online.SOURCE166
            or decision["videos"] != online.DESIGN["videos"] or decision["full_trajectories"] != 84):
        raise ValueError("Need the completed Stage-167 full video comparison")
    rows = []
    for name, group, method, event in TARGETS:
        selected = [r for r in failures if veto.source_group(r["sequence"]) == group and r["arm"] == method and r[event]]
        if len(selected) != 1:
            raise ValueError(f"Expected a unique archived event for {name}")
        seq = selected[0]["sequence"]
        video = [r for r in videos if r["sequence"] == seq and r["arm"] == NATIVE]
        if len(video) != 1:
            raise ValueError(f"Missing native result for {name}")
        target = int(selected[0][event])
        rows.append(dict(window=name, sequence=seq, kind=event, reference_arm=method, target_frame=target,
                         search_from=max(1, target - DESIGN["lookback"]), search_to=target, frames=int(video[0]["frames"])))
    return rows


def select_cases(windows, traces):
    cases, support, selected = [], [], {}

    def add(window, arm, frame, reason):
        seq = window["sequence"]
        key = (seq, arm, frame)
        if key not in selected:
            selected[key] = dict(case_id=f"{seq}__{arm}__{frame:06d}", sequence=seq, history=arm, frame=frame,
                                 end=min(frame + DESIGN["successors"], window["frames"]), windows=[], targets=[])
        case = selected[key]
        if window["window"] not in case["windows"]:
            case["windows"].append(window["window"])
            case["targets"].append(window["target_frame"])
        support.append(dict(**window, history=arm, role=reason, anchor_frame=frame, case_id=case["case_id"]))

    for window in windows:
        seq, first, last = window["sequence"], window["search_from"], window["search_to"]
        for arm in (FIXED, PAIRWISE):
            hits = np.flatnonzero(traces[seq, arm]["switched"][first - 1:last]) + first
            if not hits.size:
                support.append(dict(**window, history=arm, role="no_switch", anchor_frame=None, case_id=None))
                continue
            add(window, arm, int(hits[0]), "first")
            if len(hits) > 1:
                add(window, arm, int(hits[-1]), "last")
    for seq in sorted({r["sequence"] for r in windows}):
        hits = [(int(active[0]) + 1, arm) for arm in (FIXED, PAIRWISE)
                if (active := np.flatnonzero(traces[seq, arm]["switched"])).size]
        if hits:
            frame, arm = min(hits)
            reference = next(r for r in windows if r["sequence"] == seq)
            window = dict(reference, window="first_online_switch", kind="first_switch", reference_arm=arm,
                          target_frame=frame, search_from=1, search_to=frame)
            add(window, arm, frame, "first_in_video")
    for key in sorted(selected):
        cases.append(selected[key])
    return cases, support


def replay_branch(observer, images, template, initial_box, anchor, source, branch, model, progress):
    frame, stop = anchor["frame"], anchor["end"]
    start = max(0, frame - 1 - online.DESIGN["history"])
    at = frame - 1
    parent = source["parent"][at]
    first, following = BRANCHES[branch]
    observer.initialize(template, initial_box, first, model)
    observer.boxes.extend(source["boxes"][start:at])
    observer.peaks.extend(source["hann_peak"][start:at])
    observer.rt.tracker.frame_id = frame - 1
    rows = []
    for t in range(frame, stop + 1):
        observer.arm = first if t == frame else following
        observer.model = dict(model, threshold=-1e9) if t == frame and first != NATIVE else model
        row = observer.step(images[t], parent, t)
        parent = row["boxes"]
        rows.append(row)
        progress.advance()
    return {name: np.asarray([r[name] for r in rows]) for name in rows[0]}


def audit_branch(data, case, source, branch, threshold):
    frame, end = case["frame"], case["end"]
    parent = source["parent"][frame - 1]
    np.testing.assert_array_equal(data["frame_id"], np.arange(frame, end + 1))
    # The shared Stage-167 audit enforces a full-video warmup; here it has already
    # been replayed from the 16 cached frames before this window.
    count = end - frame + 1
    np.testing.assert_array_equal(data["parent"], np.vstack((parent, data["boxes"][:-1])))
    if (data["boxes"].shape != (count, 4) or not np.isfinite(data["boxes"]).all()
            or (data["boxes"][:, 2:] <= 0).any()):
        raise ValueError("Invalid matched continuation boxes")
    for name in ("layer", "native_layer", "proposal"):
        if not np.isin(data[name], LAYERS).all():
            raise ValueError(f"Invalid layer in {name}")
    if not np.isfinite(data["gate_value"]).all() or not np.isfinite(data["hann_peak"]).all():
        raise ValueError("Invalid matched continuation gate signal")
    thresholds = np.full(count, threshold)
    if BRANCHES[branch][0] != NATIVE:
        thresholds[0] = -1e9
    qualified = (data["proposal"] != data["native_layer"]) & (data["gate_value"] > thresholds + veto.EPSILON)
    switched = data["layer"] != data["native_layer"]
    np.testing.assert_array_equal(data["switched"], switched)
    np.testing.assert_array_equal(switched, qualified)
    if (BRANCHES[branch][1] == NATIVE and switched[1:].any()) or (branch == NATIVE and switched.any()):
        raise ValueError("An unplanned later switch occurred")
    if BRANCHES[branch][0] == FIXED:
        np.testing.assert_array_equal(data["layer"][switched], data["proposal"][switched])
    blocks = np.zeros((count, 12), int)
    blocks[:, :6] = 1
    blocks[np.arange(count), data["layer"] - 1] = 1
    np.testing.assert_array_equal(data["block_calls"], blocks)
    for name in ("crop_calls", "selector_calls", "head_calls"):
        np.testing.assert_array_equal(data[name], np.ones(count, int))


def verify_matched(arms, case, source, exact_runtime):
    for head, tail in (("original_once", "original_continued"), ("pairwise_once", "pairwise_continued")):
        for name in ("boxes", "layer", "proposal", "gate_value"):
            np.testing.assert_array_equal(arms[head][name][0], arms[tail][name][0])
    ids = [arms[arm]["native_layer"][0] for arm in BRANCHES]
    if len(set(ids)) != 1:
        raise ValueError("Starting native layer differs across matched rollouts")
    np.testing.assert_array_equal(arms["original_once"]["proposal"][0], arms["pairwise_once"]["proposal"][0])
    np.testing.assert_allclose(arms["original_once"]["gate_value"][0], arms["pairwise_once"]["gate_value"][0], atol=0, rtol=0)
    matching = "original_continued" if case["history"] == FIXED else "pairwise_continued"
    segment = slice(case["frame"] - 1, case["end"])
    difference = float(np.max(np.abs(arms[matching]["boxes"] - source["boxes"][segment])))
    layer_match = bool(np.array_equal(arms[matching]["layer"], source["layer"][segment]))
    if exact_runtime and (difference != 0. or not layer_match):
        raise ValueError(f"Continuous own-history replay differs at {case['case_id']}")
    return difference, layer_match


def report_case(case, truth, traces, replay_error, layers_match, threshold):
    frame = case["frame"]
    ious = {arm: online.tracking_iou(data["boxes"], truth) for arm, data in traces.items()}
    windows = "|".join(case["windows"])
    targets = "|".join(map(str, case["targets"]))
    base_row = dict(case_id=case["case_id"], sequence=case["sequence"], history=case["history"],
                    windows=windows, target_frames=targets, frame=frame, end=case["end"])
    rows, contrasts, recoveries = [], [], []
    for arm, data in traces.items():
        iou = ious[arm]
        valid = iou >= 0.
        ev = online.failure_events(iou)
        mean = float(iou[valid].mean()) if valid.any() else None
        low = float((iou[valid] < TRACKING["failure_iou"]).mean()) if valid.any() else None
        rows.append(dict(base_row, arm=arm, anchor_layer=int(data["layer"][0]), anchor_iou=float(iou[0]),
                         mean_iou=mean, low_iou_fraction=low, failure_events=len(ev),
                         first_failure_frame=frame + ev[0]["start"] if ev else None,
                         unrecovered_events=sum(e["recovery"] is None for e in ev),
                         switches=int(data["switched"].sum()), later_switches=int(data["switched"][1:].sum())))
        for name, target in zip(case["windows"], case["targets"], strict=True):
            if "recovery" not in name:
                continue
            length = TRACKING["recovery_run"]
            start = max(0, target - frame)
            stop = min(len(iou), target - frame + DESIGN["reference_grace"] + length)
            good = iou[start:stop] >= TRACKING["recovery_iou"]
            hits = np.flatnonzero(np.lib.stride_tricks.sliding_window_view(good, length).all(axis=1)) if len(good) >= length else []
            recoveries.append(dict(base_row, window=name, target=target, arm=arm, available_frames=len(good),
                                   reached_good_run=bool(len(hits)),
                                   first_good_frame=frame + start + int(hits[0]) if len(hits) else None))
    by = {r["arm"]: r for r in rows}
    native = by[NATIVE]
    for name in ("original", "pairwise"):
        one, repeat = by[f"{name}_once"], by[f"{name}_continued"]
        contrasts.append(dict(
            base_row, policy=name, replay_max_box_error=replay_error, replay_layers_equal=layers_match,
            original_gate_open=bool(traces["original_once"]["proposal"][0] != traces["original_once"]["native_layer"][0]
                                    and traces["original_once"]["gate_value"][0] > threshold + veto.EPSILON),
            anchor_changed=native["anchor_layer"] != one["anchor_layer"],
            single_minus_native_iou=None if one["mean_iou"] is None else one["mean_iou"] - native["mean_iou"],
            continued_minus_single_iou=None if one["mean_iou"] is None else repeat["mean_iou"] - one["mean_iou"],
            native_failure=bool(native["failure_events"]), single_failure=bool(one["failure_events"]),
            continued_failure=bool(repeat["failure_events"]),
            single_new_failure=not native["failure_events"] and bool(one["failure_events"]),
            continued_only_new_failure=not native["failure_events"] and not one["failure_events"]
            and bool(repeat["failure_events"]), continued_later_switches=repeat["later_switches"]))
    return rows, contrasts, recoveries


def rollout_location(root, plan, case, branch):
    return root / "rollouts" / f"{case['case_id']}__{branch}.npz", dict(
        plan_fingerprint=plan["fingerprint"], case_id=case["case_id"], branch=branch)


def publish(root, plan, cases, support, traces, entries, old, model_bank, native, progress):
    tables = dict(windows=support, branches=[], contrasts=[], recovery_windows=[])
    files = {}
    fold_map = {group: f["fold"] for f in plan["folds"] for group in f["held"]}
    with ExitStack() as stack:
        observer = online.RoutedObserver(native, stack)
        for case in cases:
            seq, start, end = case["sequence"], case["frame"], case["end"]
            entry = entries[seq]
            source = traces[seq, case["history"]]
            model = model_bank[fold_map[veto.source_group(seq)]]
            arms, images = {}, None
            for branch in BRANCHES:
                path, metadata = rollout_location(root, plan, case, branch)
                data = c.read_cache(path, metadata)
                if data is None:
                    if images is None:
                        template = online.checked_image(old["root"], entry["frames"][0])
                        images = {i: online.checked_image(old["root"], entry["frames"][i]) for i in range(start, end + 1)}
                    data = replay_branch(observer, images, template, entry["initial_box"], case, source, branch, model, progress)
                    audit_branch(data, case, source, branch, model["threshold"])
                    native.assert_frozen()
                    c.save(path, data, metadata)
                else:
                    audit_branch(data, case, source, branch, model["threshold"])
                arms[branch] = data
                for file in (path, path.with_suffix(".receipt.json")):
                    files[file.relative_to(root).as_posix()] = c.file_hash(file)
            error, layer_match = verify_matched(arms, case, source, exact_runtime=not plan["runtime_differences"])
            truth = np.asarray(entry["ground_truth"], float)[start:end + 1]
            for label, rows in zip(("branches", "contrasts", "recovery_windows"),
                                   report_case(case, truth, arms, error, layer_match, model["threshold"]), strict=True):
                tables[label].extend(rows)
    native.assert_frozen()
    hashes = gain.prior.publish(root, {k: v for k, v in tables.items() if v}, stage="stage168")
    return hashes, files, tables


def execute(args, root, paths):
    r76, r89, r93, r121, r159, r165, r166, r167 = paths
    p167 = c.c66.sealed(r167 / "manifest.json", online.SCHEMA)
    d167 = c.c66.sealed(r167 / "stage167_decision.json", online.SCHEMA)
    if (d167["fingerprint"] != SOURCE167 or d167["plan_fingerprint"] != p167["fingerprint"]
            or d167["runtime"] != p167["runtime"] or p167["source_stage159"] != veto.SOURCE159):
        raise ValueError("Need the exact Stage-167 trajectory bundle")
    names = ("stage167_video_summary.csv", "stage167_failures.csv")
    io.verify({str(r167 / k): d167["table_sha256"][k] for k in names})
    targets = windows(d167, *(gain.prior.prior.read_table(r167 / k) for k in names))
    p159, points, gain_models, rank_models, ranks = online.load_training(r159, r165, r166)
    _, d121, jobs = history.native_jobs(r121)
    if d121["fingerprint"] != p167["source_stage121"] or p159["folds"] != p167["folds"]:
        raise ValueError("Stage-167 source folds or Stage-121 lineage differ")
    folds = p167["folds"]
    fold_of = {g: f["fold"] for f in folds for g in f["held"]}
    thresholds = {r["fold"]: r["threshold"] for r in d167["calibration"]}
    s95 = base.s103.prior.s95
    p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
    old = origin["original"]
    traces, entries = {}, {}
    for seq in sorted({w["sequence"] for w in targets}):
        job = jobs[seq]
        entry = s95.load_unit(r93, r89, p93, p89, d89, origin, job["spec"])[0]
        if entry["sequence"] != seq or len(entry["frames"]) != len(entry["ground_truth"]) or len(entry["frames"]) != job["frames"] + 1:
            raise ValueError(f"Unexpected source entry for {seq}")
        entries[seq] = entry
        for arm in (FIXED, PAIRWISE):
            path, metadata = online.cache_location(r167, p167, seq, arm)
            io.verify({str(f): d167["artifact_sha256"][f.relative_to(r167).as_posix()]
                       for f in (path, path.with_suffix(".receipt.json"))})
            trace = c.read_cache(path, metadata)
            if trace is None:
                raise ValueError(f"Missing Stage-167 trajectory: {path}")
            online.audit(trace, job["frames"], entry["initial_box"], arm, thresholds[fold_of[veto.source_group(seq)]])
            traces[seq, arm] = trace
    cases, support = select_cases(targets, traces)
    if not cases:
        raise ValueError("No archived switch supports the matched continuation diagnostic")
    import torch

    torch.set_num_threads(1)
    runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
    differences = base.s109.check_runtime(p167["runtime"], runtime)
    code = (Path(__file__), Path(online.__file__), Path(gate.__file__))
    plan = io.seal(dict(schema=SCHEMA, source_stage167=SOURCE167, source_stage121=d121["fingerprint"],
                        design=DESIGN, runtime=runtime, runtime_differences=differences, folds=folds, cases=cases,
                        code_sha256={p.name: c.file_hash(p) for p in code}))
    c.publish_identical(root / "manifest.json", plan)
    (root / "rollouts").mkdir(exist_ok=True)
    searches = len(BRANCHES) * sum(case["end"] - case["frame"] + 1 for case in cases)
    reused = sum(case["end"] - case["frame"] + 1 for case in cases for arm in BRANCHES
                 if rollout_location(root, plan, case, arm)[0].with_suffix(".receipt.json").exists())
    work = searches + len(points) + sum(len(f["fit"]) + 1 for f in folds) + 1
    progress = OverallProgress("Stage-168", work, reused, args.progress_interval, "work")
    progress.report("Restoring source models")
    vectors, _ = history.prepare(points, p159, r159, r121, progress)
    models, calibration, _ = gate.fitted_gate(points, folds, vectors, gain_models, rank_models, ranks, progress)
    for row, recorded in zip(calibration, d167["calibration"], strict=True):
        for field in recorded:
            np.testing.assert_allclose(row[field], recorded[field], rtol=0, atol=1e-10,
                                       err_msg=f"Stage-167 calibration differs for fold {recorded['fold']}")
        models[recorded["fold"]]["threshold"] = recorded["threshold"]
    progress.report("Replaying matched continuations")
    args.parameter = runtime["tracker"]["parameter"]
    args.checkpoint = args.checkpoint.expanduser().resolve()
    snapshot = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
    native, _ = base.s103.s76.startup.checked_runtime(args, snapshot, root)
    hashes, artifacts, tables = publish(root, plan, cases, support, traces, entries, old, models, native, progress)
    contrasts = tables["contrasts"]
    counts = {name: sum(r[name] for r in contrasts if r["original_gate_open"] and r["anchor_changed"])
              for name in ("single_new_failure", "continued_only_new_failure")}
    details = dict(plan_fingerprint=plan["fingerprint"], runtime=runtime, runtime_differences=differences,
                   windows=len(targets), cases=len(cases), branches=len(tables["branches"]), searches=searches,
                   original_gate_replays=sum(r["original_gate_open"] for r in contrasts if r["policy"] == "original"),
                   max_replay_box_error=max(r["replay_max_box_error"] for r in contrasts),
                   artifact_sha256=artifacts, table_sha256=hashes, **counts)
    return details, progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for stage in SOURCES:
        parser.add_argument(f"--stage{stage}-root", type=Path)
    parser.add_argument("--cached-windows", type=Path, metavar="STAGE167_BUNDLE")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-interval", type=float, default=15.)
    parser.set_defaults(mode="stage168_matched_layer_rollouts")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    paths = [p.expanduser().resolve() for n in SOURCES if (p := getattr(args, f"stage{n}_root")) is not None]
    cached = args.cached_windows is not None
    if (cached and (paths or args.checkpoint is not None)
            or not cached and (len(paths) != len(SOURCES) or args.checkpoint is None) or args.progress_interval <= 0):
        parser.error("Supply either the Stage-167 bundle or all server roots and the original checkpoint")
    if any(root == p or root in p.parents or p in root.parents for p in paths):
        parser.error("Use a separate output directory")
    status = "COMPLETE_STAGE168_CACHED_EVENT_WINDOWS" if cached else "COMPLETE_STAGE168_MATCHED_CONTINUATIONS"
    with gain.prior.prior.writer_lock(root):
        destination = root / "stage168_decision.json"
        if destination.exists():
            saved = c.c66.sealed(destination, SCHEMA)
            if saved["status"] != status or saved["source_stage167"] != SOURCE167:
                raise ValueError("Use separate directories for the event windows and matched rollouts")
            io.verify({str(root / k): h for k, h in (saved["artifact_sha256"] | saved["table_sha256"]).items()})
            OverallProgress("Stage-168", saved["work"], saved["work"], args.progress_interval, "work").report("Complete")
            return
        if cached:
            d, b = gain.prior.archives.bundle(args.cached_windows, 167, SOURCE167)
            rows = windows(d, b["stage167_video_summary.csv"], b["stage167_failures.csv"])
            progress = OverallProgress("Stage-168", 1, interval=args.progress_interval, unit="work")
            progress.report("Auditing source events")
            hashes = gain.prior.publish(root, dict(event_windows=rows), stage="stage168")
            details = dict(windows=len(rows), cases=0, searches=0, artifact_sha256={}, table_sha256=hashes)
        else:
            details, progress = execute(args, root, paths)
        if progress.done != progress.total - 1:
            raise ValueError("Incomplete matched-continuation progress accounting")
        c.publish_identical(destination, io.seal(dict(schema=SCHEMA, status=status, source_stage167=SOURCE167,
            work=progress.total, new_full_trajectories=0, **details)))
        progress.phase = "Complete"
        progress.advance()
        progress.report()


if __name__ == "__main__":
    main()
