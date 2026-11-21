"""Run closed-loop native, fixed-proposal and pairwise-proposal SGLA trajectories."""

import argparse
from collections import deque
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiment_progress import OverallProgress
import stage167_layer_gate as gate
from stage27_uav_common import DESIGN as TRACKING, checked_image, failure_events, success_auc, tracking_iou


ranker, stage165 = gate.stage166, gate.stage165
gain, history, veto, c, io = stage165.gain, gate.history, gate.veto, ranker.c, ranker.io
trained, base, LAYERS = gate.trained, gate.trained.base, gate.LAYERS
SOURCE166 = "c85ff5d3f9b591261088f160c9af73b3bff4b76a0a8909947f1867925dc318e3"
SCHEMA = "stage167_online_layer_ranking_v1"
STATUS = "COMPLETE_STAGE167_ONLINE_LAYER_DEVELOPMENT"
ARMS = ("native", "shared_gate_original", "shared_gate_pairwise")
SOURCES = (76, 89, 93, 121, 159, 165, 166)
DESIGN = dict(videos=28, tracked_frames=29047, history=16, recovery_grace=32,
              bootstrap_seed=23167, bootstrap_samples=20000)
METRICS = ("auc", "low_iou_fraction", "longest_failure_fraction", "failure_events", "unrecovered_events")


class RoutedObserver:
    """Change only the MLP choice on a gated frame, inside the original tracker."""

    def __init__(self, native, stack):
        self.native, self.rt = native, native.rt
        stack.callback(self.rt.backbone.blocks[5].register_forward_hook(self.capture_prefix).remove)
        stack.callback(self.rt.backbone.MLP.register_forward_hook(self.route).remove)
        stack.callback(self.rt.net.register_forward_hook(self.capture_peak).remove)

    def initialize(self, image, box, arm, model):
        self.arm, self.model = arm, model
        self.boxes = deque(maxlen=DESIGN["history"])
        self.peaks = deque(maxlen=DESIGN["history"])
        self.native.initialize(image, box)

    def capture_prefix(self, module, args, tokens):
        if self.arm != "native" and len(self.boxes) >= trained.DESIGN["history"]:
            self.tokens = tokens

    def capture_peak(self, module, args, output):
        self.peak = float((self.rt.tracker.output_window * output["score_map"]).amax().item())

    def route(self, module, args, output):
        self.original = native = self.rt.selected_layer(output)
        self.proposal, self.value, self.switched, self.chosen = native, 0., False, native
        if self.arm == "native" or len(self.boxes) < trained.DESIGN["history"]:
            return None
        if self.tokens is None:
            raise ValueError("Missing L6 representation at the original selector")
        projection = trained.projected_prefix(self.rt, self.tokens)
        boxes, peaks = np.asarray(self.boxes), np.asarray(self.peaks)
        frame = len(boxes) + 1
        short = np.r_[projection, trained.history_features(boxes, peaks, frame)]
        long = np.r_[projection, history.features_from_history(boxes, peaks, frame, DESIGN["history"], True)]
        self.proposal, self.value, self.switched, self.chosen = gate.online_decision(
            self.model, native, short, long, self.arm == ARMS[2])
        if not self.switched:
            return None
        forced = self.rt.torch.zeros_like(output)
        forced[:, LAYERS.index(self.chosen)] = 1
        return forced

    def step(self, image, parent, frame):
        tracker = self.rt.tracker
        self.tokens = self.peak = self.original = None
        namespace = tracker.track.__func__.__globals__
        sample = namespace["sample_target"]
        crops = 0

        def sample_once(*args, **kwargs):
            nonlocal crops
            crops += 1
            return sample(*args, **kwargs)

        with patch.dict(namespace, sample_target=sample_once):
            out = self.native.step(image, parent)
        if (self.peak is None or not np.isfinite(self.peak) or tracker.frame_id != frame
                or out["layer"] != self.chosen or crops != 1):
            raise ValueError("Tracker route or Hann peak did not replay")
        self.boxes.append(out["box"])
        self.peaks.append(self.peak)
        return dict(frame_id=frame, parent=np.asarray(parent, float), boxes=out["box"], layer=out["layer"],
                    native_layer=self.original, proposal=self.proposal, gate_value=self.value,
                    switched=self.switched, hann_peak=self.peak, crop_calls=crops, head_calls=out["head_calls"],
                    selector_calls=out["selector_calls"], block_calls=out["block_calls"], tracker_ms=out["network_ms"])


def collect(observer, frames, initial_box, root, arm, model, progress, reader=checked_image):
    observer.initialize(reader(root, frames[0]), initial_box, arm, model)
    parent, rows = initial_box, []
    for frame, file in enumerate(frames[1:], 1):
        row = observer.step(reader(root, file), parent, frame)
        parent = row["boxes"]
        rows.append(row)
        progress.advance()
    return {name: np.asarray([r[name] for r in rows]) for name in rows[0]}


def audit(trace, frames, initial, arm, threshold):
    np.testing.assert_array_equal(trace["frame_id"], np.arange(1, frames + 1))
    boxes = trace["boxes"]
    np.testing.assert_array_equal(trace["parent"], np.vstack((initial, boxes[:-1])))
    if boxes.shape != (frames, 4) or not np.isfinite(boxes).all() or (boxes[:, 2:] <= 0.).any():
        raise ValueError("Invalid full trajectory")
    for field in ("layer", "native_layer", "proposal"):
        if not np.isin(trace[field], LAYERS).all():
            raise ValueError(f"Unexpected {field}")
    for field in ("gate_value", "hann_peak", "tracker_ms"):
        if trace[field].shape != (frames,) or not np.isfinite(trace[field]).all():
            raise ValueError(f"Invalid trace {field}")
    switched = trace["layer"] != trace["native_layer"]
    applied = (trace["proposal"] != trace["native_layer"]) & (trace["gate_value"] > threshold + veto.EPSILON)
    np.testing.assert_array_equal(trace["switched"], switched)
    np.testing.assert_array_equal(switched, applied)
    if switched[:trained.DESIGN["history"]].any() or (arm == "native" and switched.any()):
        raise ValueError("Unexpected switch before causal history was available")
    if arm == ARMS[1]:
        np.testing.assert_array_equal(trace["layer"][switched], trace["proposal"][switched])
    counts = np.zeros((frames, 12), int)
    counts[:, :6] = 1
    counts[np.arange(frames), trace["layer"] - 1] = 1
    np.testing.assert_array_equal(trace["block_calls"], counts)
    for name in ("crop_calls", "selector_calls", "head_calls"):
        np.testing.assert_array_equal(trace[name], np.ones(frames, int))


def audit_shared_state(control, other, same_gate):
    different = np.flatnonzero(control["layer"] != other["layer"])
    first = int(different[0]) if different.size else len(control["layer"])
    for key in ("parent", "native_layer"):
        np.testing.assert_array_equal(control[key][:first + 1], other[key][:first + 1])
    for key in ("boxes", "hann_peak"):
        np.testing.assert_array_equal(control[key][:first], other[key][:first])
    if same_gate:
        for key in ("proposal", "gate_value", "switched"):
            np.testing.assert_array_equal(control[key][:first + 1], other[key][:first + 1])


def misses_recovery(event, iou):
    recovery = event["recovery"]
    if recovery is None:
        return False
    length = TRACKING["recovery_run"]
    stop = min(len(iou), recovery + DESIGN["recovery_grace"] + length)
    valid = iou[event["start"]:stop] >= TRACKING["recovery_iou"]
    return len(valid) < length or not np.lib.stride_tricks.sliding_window_view(valid, length).all(axis=1).any()


def report_video(sequence, truth, traces):
    ious = {arm: tracking_iou(trace["boxes"], truth) for arm, trace in traces.items()}
    events = {arm: failure_events(values) for arm, values in ious.items()}
    rows, failures, routing = [], [], []
    for arm, trace in traces.items():
        u = ious[arm]
        ev = events[arm]
        valid = u >= 0
        timings = trace["tracker_ms"][min(10, len(u) - 1):]
        rows.append(dict(sequence=sequence, arm=arm, frames=len(u), auc=success_auc(u[valid]),
                         low_iou_fraction=float((u[valid] < TRACKING["failure_iou"]).mean()),
                         longest_failure_fraction=max((e["end"] - e["start"] + 1 for e in ev), default=0) / len(u),
                         failure_events=len(ev), unrecovered_events=sum(e["recovery"] is None for e in ev),
                         stable_native_new_failure=not events["native"] and bool(ev),
                         stable_fixed_new_failure=not events[ARMS[1]] and bool(ev),
                         missed_native_recoveries=sum(misses_recovery(e, u) for e in events["native"]),
                         missed_fixed_recoveries=sum(misses_recovery(e, u) for e in events[ARMS[1]]),
                         switches=int(trace["switched"].sum()), head_calls=int(trace["head_calls"].sum()),
                         block_calls=int(trace["block_calls"].sum()), median_tracker_ms=float(np.median(timings)),
                         p95_tracker_ms=float(np.quantile(timings, .95))))
        failures.extend(dict(sequence=sequence, arm=arm, onset=e["start"] + 1, end=e["end"] + 1,
                             recovery=None if e["recovery"] is None else e["recovery"] + 1, censor=e["censor"])
                        for e in ev)
        for k in LAYERS:
            route = trace["layer"] == k
            routing.append(dict(sequence=sequence, arm=arm, layer=k, frames=int(route.sum()),
                                switched=int((route & trace["switched"]).sum())))
    return rows, failures, routing


def summary(rows):
    videos = sorted({r["sequence"] for r in rows})
    index = {(r["sequence"], r["arm"]): r for r in rows}
    policies, paired = [], []
    for arm in ARMS:
        group = [index[v, arm] for v in videos]
        policies.append(dict(arm=arm, videos=len(group), **{m: float(np.mean([r[m] for r in group]))
                                                          for m in (*METRICS, "median_tracker_ms", "p95_tracker_ms")},
                             **{m: sum(r[m] for r in group) for m in (
                                 "switches", "head_calls", "block_calls", "stable_native_new_failure",
                                 "stable_fixed_new_failure", "missed_native_recoveries", "missed_fixed_recoveries")}))
    for test, control in ((ARMS[1], ARMS[0]), (ARMS[2], ARMS[0]), (ARMS[2], ARMS[1])):
        for cohort in ("all", "without_0176"):
            sequences = [v for v in videos if cohort == "all" or veto.source_group(v) != "uav0000176"]
            for metric in METRICS:
                values = [index[v, test][metric] - index[v, control][metric] for v in sequences]
                row = base.s103.contrast(values, f"{test}_minus_{control}", metric, DESIGN)
                paired.append(dict(arm=test, baseline=control, cohort=cohort, **row))
    candidate = policies[-1]
    quality = {(r["baseline"], r["cohort"]): r for r in paired if r["arm"] == ARMS[2] and r["metric"] == "auc"}
    mean_key = "mean_delta"
    checks = dict(
        auc_positive_vs_native=quality[ARMS[0], "all"][mean_key] > 0.,
        auc_positive_vs_fixed=quality[ARMS[1], "all"][mean_key] > 0.,
        auc_positive_without_0176=all(quality[arm, "without_0176"][mean_key] > 0. for arm in ARMS[:2]),
        no_new_failure_on_stable_videos=candidate["stable_native_new_failure"] == candidate["stable_fixed_new_failure"] == 0,
        retained_recoveries=candidate["missed_native_recoveries"] == candidate["missed_fixed_recoveries"] == 0,
        low_iou_not_higher=all(candidate["low_iou_fraction"] <= r["low_iou_fraction"] for r in policies[:2]),
        unrecovered_not_higher=all(candidate["unrecovered_events"] <= r["unrecovered_events"] for r in policies[:2]),
        any_switches=candidate["switches"] > 0)
    return policies, paired, checks


def cache_location(root, plan, sequence, arm):
    return root / f"evaluation/{sequence}__{arm}.npz", dict(
        plan_fingerprint=plan["fingerprint"], sequence=sequence, arm=arm)


def replay_native(trace, stage121, p121, d121, job):
    key = f"evaluation/{base.job_key(job)}"
    io.verify({str(stage121 / k): d121["artifact_sha256"][k] for k in (key + ".npz", key + ".receipt.json")})
    old = trained.source.source.prior.prior.load_trace(stage121, p121, job)
    np.testing.assert_array_equal(trace["boxes"], old["boxes"])
    np.testing.assert_array_equal(trace["layer"], old["layer"][:, 0])


def run(root, plan, jobs, entries, native, models, calibration, curve, stage121, p121, d121, image_root, progress):
    tables = {name: [] for name in ("video_summary", "failures", "routing")}
    artifacts = {}
    fold_of = {source: f["fold"] for f in plan["folds"] for source in f["held"]}
    with ExitStack() as stack:
        observer = RoutedObserver(native, stack)
        for sequence, job in sorted(jobs.items()):
            entry = entries[sequence]
            model = models[fold_of[veto.source_group(sequence)]]
            traces = {}
            for arm in ARMS:
                path, metadata = cache_location(root, plan, sequence, arm)
                data = c.read_cache(path, metadata)
                fresh = data is None
                if fresh:
                    data = collect(observer, entry["frames"], entry["initial_box"], image_root, arm, model, progress)
                audit(data, job["frames"], entry["initial_box"], arm, model["threshold"])
                if arm == "native" and not plan["runtime_differences"]:
                    replay_native(data, stage121, p121, d121, job)
                if fresh:
                    native.assert_frozen()
                    c.save(path, data, metadata)
                traces[arm] = data
                for file in (path, path.with_suffix(".receipt.json")):
                    artifacts[file.relative_to(root).as_posix()] = c.file_hash(file)
            audit_shared_state(traces[ARMS[0]], traces[ARMS[1]], same_gate=False)
            audit_shared_state(traces[ARMS[0]], traces[ARMS[2]], same_gate=False)
            audit_shared_state(traces[ARMS[1]], traces[ARMS[2]], same_gate=True)
            values = report_video(sequence, np.asarray(entry["ground_truth"], float)[1:], traces)
            for rows, items in zip(tables.values(), values, strict=True):
                rows.extend(items)
    policy, paired, checks = summary(tables["video_summary"])
    tables.update(policy_summary=policy, paired=paired, calibration=calibration, calibration_curve=curve)
    hashes = gain.prior.publish(root, tables, stage="stage167")
    return artifacts, hashes, policy, checks


def load_training(r159, r165, r166):
    plan, models, points = veto.load(r159)
    d165 = c.c66.sealed(r165 / "stage165_decision.json", stage165.SCHEMA)
    d166 = c.c66.sealed(r166 / "stage166_decision.json", ranker.SCHEMA)
    if (d165["fingerprint"] != ranker.SOURCE165 or d166["fingerprint"] != SOURCE166
            or d166["status"] != "COMPLETE_STAGE166_PAIRWISE_LAYER_RANKING" or not d166["ready_for_online_test"]
            or d166["source_stage165"] != d165["fingerprint"]):
        raise ValueError("Need the completed Stage-165 and passing Stage-166 runs")
    names = ("stage165_choices.csv", "stage165_layer_scores.csv")
    io.verify({str(r165 / k): d165["table_sha256"][k] for k in names})
    paths = {r166 / "stage166_models.json": d166["model_sha256"],
             r166 / "stage166_layer_scores.csv": d166["table_sha256"]["stage166_layer_scores.csv"]}
    io.verify({str(k): h for k, h in paths.items()})
    ranker.load_baselines(points, d165, *(gain.prior.prior.read_table(r165 / k) for k in names))
    score_rows = gain.prior.prior.read_table(r166 / "stage166_layer_scores.csv")
    index = {(r["case_id"], int(r["layer"])): r for r in score_rows}
    if len(index) != len(score_rows) or len(index) != len(points) * len(LAYERS):
        raise ValueError("Incomplete Stage-166 score roster")
    ranks = np.array([[float(index[p["case_id"], k][ranker.LEARNED]) for k in LAYERS] for p in points])
    ranked = c.c66.sealed(r166 / "stage166_models.json", ranker.SCHEMA)
    if ranked["source_stage165"] != ranker.SOURCE165:
        raise ValueError("Archived Stage-166 ranker has a different training source")
    return plan, points, models, ranked["models"], ranks


def start(args, root, paths):
    r76, r89, r93, r121, r159, r165, r166 = paths
    p159, points, gain_models, rank_models, ranks = load_training(r159, r165, r166)
    p121, d121, jobs = history.native_jobs(r121)
    if (len(jobs) != DESIGN["videos"] or sum(j["frames"] for j in jobs.values()) != DESIGN["tracked_frames"]
            or set(jobs) != {p["sequence"] for p in points} or p159["source_stage121"] != d121["fingerprint"]):
        raise ValueError("Need the frozen 28-video development roster")
    s95 = base.s103.prior.s95
    p93, _, p89, d89, origin = s95.s94.load_run(r93, r76, r89)
    old = origin["original"]
    entries = {seq: s95.load_unit(r93, r89, p93, p89, d89, origin, job["spec"])[0] for seq, job in jobs.items()}
    for seq, entry in entries.items():
        if entry["sequence"] != seq or len(entry["frames"]) != len(entry["ground_truth"]) or len(entry["frames"]) != jobs[seq]["frames"] + 1:
            raise ValueError(f"Video roster mismatch at {seq}")
    import torch

    torch.set_num_threads(1)
    runtime = base.s103.s76.device_runtime(old, torch, args.gpu)
    differences = base.s109.check_runtime(p121["runtime"], runtime)
    base.s109.check_runtime(p159["runtime"], runtime)
    code = (Path(__file__), Path(gate.__file__), Path(ranker.__file__))
    plan = io.seal(dict(schema=SCHEMA, source_stage121=d121["fingerprint"], source_stage159=veto.SOURCE159,
                        source_stage165=ranker.SOURCE165, source_stage166=SOURCE166, runtime=runtime,
                        runtime_differences=differences, design=DESIGN, folds=p159["folds"], arms=list(ARMS),
                        code_sha256={p.name: c.file_hash(p) for p in code}))
    c.publish_identical(root / "manifest.json", plan)
    (root / "evaluation").mkdir(exist_ok=True)
    reused = sum(job["frames"] for seq, job in jobs.items() for arm in ARMS
                 if cache_location(root, plan, seq, arm)[0].with_suffix(".receipt.json").exists())
    folds = p159["folds"]
    work = DESIGN["tracked_frames"] * len(ARMS) + len(points) + sum(len(f["fit"]) + 1 for f in folds) + 1
    progress = OverallProgress("Stage-167", work, reused, args.progress_interval, "work")
    progress.report("Reading frozen validation inputs")
    vectors, _ = history.prepare(points, p159, r159, r121, progress)
    progress.report("Calibrating on training source groups")
    models, calibration, curve = gate.fitted_gate(points, folds, vectors, gain_models, rank_models, ranks, progress)
    args.parameter = runtime["tracker"]["parameter"]
    snapshot = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
    native, _ = base.s103.s76.startup.checked_runtime(args, snapshot, root)
    progress.report("Tracking complete videos")
    artifacts, tables, policy, checks = run(root, plan, jobs, entries, native, models, calibration, curve,
                                            r121, p121, d121, old["root"], progress)
    if progress.done != work - 1:
        raise ValueError("Incomplete full-trajectory progress accounting")
    c.publish_identical(root / "stage167_decision.json", io.seal(dict(
        schema=SCHEMA, status=STATUS, role="consumed_development_full_trajectory_validation",
        plan_fingerprint=plan["fingerprint"], source_stage166=SOURCE166, runtime=runtime, runtime_differences=differences,
        videos=len(jobs), full_trajectories=len(jobs) * len(ARMS), work=work,
        head_calls=sum(r["head_calls"] for r in policy), block_calls=sum(r["block_calls"] for r in policy),
        calibration=calibration, artifact_sha256=artifacts, table_sha256=tables, checks=checks,
        passes_development_checks=all(checks.values()), confirmation_inference_calls=0)))
    progress.phase = "Complete"
    progress.advance()
    progress.report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for stage in SOURCES:
        parser.add_argument(f"--stage{stage}-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-interval", type=float, default=15.)
    parser.set_defaults(mode="stage167_online_layers")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    paths = [getattr(args, f"stage{n}_root").expanduser().resolve() for n in SOURCES]
    if args.progress_interval <= 0 or any(root == p or p in root.parents or root in p.parents for p in paths):
        parser.error("Use separate source and output directories and a positive progress interval")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    with gain.prior.prior.writer_lock(root):
        destination = root / "stage167_decision.json"
        if destination.exists():
            saved = c.c66.sealed(destination, SCHEMA)
            plan = c.c66.sealed(root / "manifest.json", SCHEMA)
            if saved["status"] != STATUS or saved["plan_fingerprint"] != plan["fingerprint"]:
                raise ValueError("Unexpected Stage-167 result")
            io.verify({str(root / k): h for k, h in (saved["artifact_sha256"] | saved["table_sha256"]).items()})
            OverallProgress("Stage-167", saved["work"], saved["work"], args.progress_interval, "work").report("Complete")
            return
        start(args, root, paths)


if __name__ == "__main__":
    main()
