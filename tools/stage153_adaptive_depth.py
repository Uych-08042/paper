"""Compare native SGLA with one optional serial late block on uncertain frames."""
from __future__ import annotations

import argparse
import hashlib
from collections import deque
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

import stage133_native_failure_audit as prior
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, failure_events, success_auc, tracking_iou
from stage27_uav_summary import write_table


base, source, c, io = prior.base, prior.source, prior.c, prior.io
SCHEMA = "stage153_serial_one_block_v1"
STATUS = "COMPLETE_ADAPTIVE_SERIAL_DEPTH_DEVELOPMENT"
ARMS = ("native", "adaptive_depth")
DESIGN = dict(videos=28, tracked_frames=29047, warmup=32, margin_history=64, margin_quantile=.25,
              low_iou=.2, bootstrap_seed=23153, bootstrap_samples=20000, timing_discard=10)
SOURCES = ("stage76", "stage89", "stage93", "stage121")
TABLES = ("policy_summary", "paired", "video_summary", "failures", "routing")
METRICS = ("auc", "low_iou_fraction", "longest_failure_fraction", "unrecovered_events", "failure_events")
EVENT_FIELDS = ("sequence", "arm", "onset", "end", "recovery", "duration", "censor")
WORK = dict(training_updates=0, confirmation_inference_calls=0)


class SelectorHistory:
    def __init__(self):
        self.state = deque(maxlen=DESIGN["margin_history"])

    def observe(self, margin):
        if not np.isfinite(margin) or not 0. <= margin <= 1.:
            raise ValueError("Invalid top-two SGLA selector score margin")
        threshold = float(np.quantile(self.state, DESIGN["margin_quantile"])) if len(self.state) >= DESIGN["warmup"] else -1.
        self.state.append(margin)
        return threshold, margin < threshold


class DepthObserver:
    """Extend the selected block's tokens before the original norm and CENTER head."""
    def __init__(self, native, stack):
        self.native = native
        self.rt = native.rt
        self.blocks = self.rt.backbone.blocks
        stack.callback(self.rt.backbone.MLP.register_forward_hook(self.capture_selector).remove)
        for layer in range(7, 13):
            stack.callback(self.blocks[layer-1].register_forward_hook(self.capture_late(layer)).remove)

    def initialize(self, image, box, arm):
        self.native.initialize(image, box)
        self.arm = arm
        self.history = SelectorHistory()

    def capture_selector(self, module, args, scores):
        values = scores.detach().cpu().numpy()
        if values.shape != (1, 6) or not np.isfinite(values).all():
            raise ValueError("Expected one native SGLA selector pass with six probabilities")
        ordered = np.sort(values[0])
        self.margin = float(ordered[-1] - ordered[-2])
        self.threshold, self.difficult = self.history.observe(self.margin)

    def capture_late(self, layer):
        def after_block(module, args, tokens):
            if self.selected:
                if self.added != layer:
                    raise ValueError("Native SGLA executed an unexpected second late block")
                return None
            self.selected = layer
            if layer < 12 and (self.force_extra or (self.arm == "adaptive_depth" and self.difficult)):
                self.added = layer+1
                return self.blocks[layer](tokens)
            return None
        return after_block

    def step(self, image, parent, frame, *, force_extra=False, capture_crop=False):
        native, tracker = self.native, self.rt.tracker
        native.blocks[:] = 0
        native.heads = native.selectors = 0
        self.selected = self.added = 0
        self.force_extra = force_extra
        self.margin = self.threshold = None
        self.difficult = False
        tracker.state = np.asarray(parent, float).tolist()
        tracker.frame_id = frame-1
        namespace = tracker.track.__func__.__globals__
        sample = namespace["sample_target"]
        crops = 0
        crop_sha256 = None

        def sample_once(*args, **kwargs):
            nonlocal crops, crop_sha256
            crops += 1
            result = sample(*args, **kwargs)
            if capture_crop:
                crop_sha256 = hashlib.sha256(np.ascontiguousarray(result[0]).tobytes()).hexdigest()
            return result

        native.synchronize()
        with patch.dict(namespace, sample_target=sample_once), self.rt.torch.no_grad():
            started = perf_counter()
            output = tracker.track(image)["target_bbox"]
            native.synchronize()
            elapsed = (perf_counter()-started)*1000.
        if tracker.frame_id != frame or self.margin is None or not self.selected:
            raise ValueError("Incomplete native tracker call")
        np.testing.assert_array_equal(tracker.state, output)
        row = dict(boxes=np.array(output, float), parent=np.array(parent, float),
                   frame_id=frame, margin=self.margin, threshold=self.threshold, difficult=self.difficult,
                   selected=self.selected, added=self.added, crop_calls=crops, head_calls=native.heads,
                   selector_calls=native.selectors, block_calls=native.blocks.copy(), tracker_ms=elapsed)
        if capture_crop:
            row["crop_sha256"] = crop_sha256
        return row


def collect(observer, entry, dataset_root, arm, reader=checked_image):
    frames = entry["frames"]
    initial_box = entry["initial_box"]
    observer.initialize(reader(dataset_root, frames[0]), initial_box, arm)
    parent, records = np.array(initial_box, float), []
    for frame, item in enumerate(frames[1:], 1):
        row = observer.step(reader(dataset_root, item), parent, frame)
        records.append(row)
        parent = row["boxes"]
        if frame % 500 == 0:
            print(f"  [{arm}] {frame}/{len(frames)-1} frames", flush=True)
    return {k:np.asarray([r[k] for r in records]) for k in records[0]}


def audit(data, frames, initial_box, arm):
    np.testing.assert_array_equal(data["frame_id"], np.arange(1, frames+1))
    boxes = data["boxes"]
    if boxes.shape != (frames, 4) or not np.isfinite(boxes).all() or (boxes[:, 2:] <= 0).any():
        raise ValueError("Invalid Stage-153 output trajectory")
    np.testing.assert_array_equal(data["parent"], np.vstack((initial_box, boxes[:-1])))
    if not np.isfinite(data["tracker_ms"]).all() or (data["tracker_ms"] <= 0.).any():
        raise ValueError("Invalid tracked frame duration")
    history = SelectorHistory()
    thresholds, difficult = zip(*(history.observe(margin) for margin in data["margin"]))
    np.testing.assert_array_equal(data["threshold"], thresholds)
    np.testing.assert_array_equal(data["difficult"], difficult)
    selected = data["selected"]
    if not np.isin(selected, np.arange(7, 13)).all():
        raise ValueError("Unexpected late block index")
    should_add = (arm == "adaptive_depth") & data["difficult"] & (selected < 12)
    np.testing.assert_array_equal(data["added"], np.where(should_add, selected+1, 0))
    expected = np.zeros((frames, 12), int)
    expected[:, :6] = 1
    expected[np.arange(frames), selected-1] = 1
    expected[should_add, data["added"][should_add]-1] = 1
    np.testing.assert_array_equal(data["block_calls"], expected)
    for k in ("crop_calls", "head_calls", "selector_calls"):
        np.testing.assert_array_equal(data[k], np.ones(frames, int))


def audit_prefix(native, adaptive):
    extensions = np.flatnonzero(adaptive["added"])
    first = int(extensions[0]) if extensions.size else len(native["boxes"])
    for key in ("parent", "selected", "margin", "threshold"):
        np.testing.assert_array_equal(native[key][:first+1], adaptive[key][:first+1])
    np.testing.assert_array_equal(native["boxes"][:first], adaptive["boxes"][:first])


def report_video(entry, traces):
    truth = np.asarray(entry["ground_truth"], float)[1:]
    ious = {arm:tracking_iou(t["boxes"], truth) for arm,t in traces.items()}
    failures = {arm:failure_events(u) for arm,u in ious.items()}
    videos, events, routing = [], [], []
    for arm, trace in traces.items():
        u, ev = ious[arm], failures[arm]
        valid = u >= 0.
        extended = trace["added"] > 0
        lengths = [e["end"]-e["start"]+1 for e in ev]
        times = trace["tracker_ms"][min(DESIGN["timing_discard"], len(u)-1):]
        prefix = dict(sequence=entry["sequence"], arm=arm)
        videos.append(dict(prefix, tracked_frames=len(u), labeled_frames=int(valid.sum()), auc=success_auc(u[valid]),
            low_iou_fraction=float((u[valid] < DESIGN["low_iou"]).mean()), failure_events=len(ev),
            unrecovered_events=sum(e["recovery"] is None for e in ev),
            longest_failure_fraction=max(lengths, default=0)/len(u), native_stable=not failures["native"],
            new_failure_on_native_stable=not failures["native"] and bool(ev),
            extended_frames=int(extended.sum()), difficult_frames=int(trace["difficult"].sum()),
            difficult_at_l12=int((trace["difficult"] & (trace["selected"] == 12)).sum()),
            block_calls=int(trace["block_calls"].sum()), head_calls=int(trace["head_calls"].sum()),
            median_tracker_ms=float(np.median(times)), p95_tracker_ms=float(np.quantile(times,.95))))
        for e,length in zip(ev,lengths):
            events.append(dict(prefix, onset=e["start"]+1, end=e["end"]+1,
                recovery=None if e["recovery"] is None else e["recovery"]+1, duration=length, censor=e["censor"]))
        for layer in range(7, 13):
            mask = trace["selected"] == layer
            routing.append(dict(prefix, selected_layer=layer, frames=int(mask.sum()),
                difficult=int((mask & trace["difficult"]).sum()), extended=int((mask & extended).sum())))
    return videos, events, routing


def summarize(rows):
    videos = sorted({r["sequence"] for r in rows})
    index = {(r["sequence"], r["arm"]):r for r in rows}
    policy = []
    for arm in ARMS:
        local = [index[v,arm] for v in videos]
        counts = {k:sum(r[k] for r in local) for k in (
            "head_calls", "block_calls", "extended_frames", "difficult_frames", "difficult_at_l12", "new_failure_on_native_stable")}
        policy.append(dict(arm=arm, videos=len(local), **counts, blocks_per_frame=counts["block_calls"]/counts["head_calls"],
            **{m:float(np.mean([r[m] for r in local])) for m in (*METRICS, "median_tracker_ms", "p95_tracker_ms")}))
    paired = [base.s103.contrast([index[v,ARMS[1]][m]-index[v,ARMS[0]][m] for v in videos],
                               "adaptive_depth_minus_native", m, DESIGN) for m in METRICS]
    return policy, paired


def check_results(root):
    plan = c.c66.sealed(root/"manifest.json", SCHEMA)
    report = c.c66.sealed(root/"stage153_decision.json", SCHEMA)
    if (report["status"] != STATUS or plan["design"] != DESIGN or report["plan_fingerprint"] != plan["fingerprint"]
            or plan["source_stage121"] != source.SOURCE121
            or report["new_search_calls"] != DESIGN["tracked_frames"] * len(ARMS)
            or report["new_block_calls"] != 7*report["new_search_calls"]+report["extra_blocks"]
            or report["new_full_trajectories"] != DESIGN["videos"] * len(ARMS)
            or len(report["artifact_sha256"]) != 2*report["new_full_trajectories"]
            or any(report[k] != value for k,value in WORK.items())
            or set(report["table_sha256"]) != {f"stage153_{k}.csv" for k in TABLES}):
        raise ValueError("Incomplete Stage-153 comparison")
    io.verify({str(root/k):v for k,v in (report["artifact_sha256"]|report["table_sha256"]).items()})


def run(root, r121, p121, d121, plan, observer, dataset_root, load_entry):
    tables = {k:[] for k in TABLES}
    artifacts = {}
    (root/"evaluation").mkdir(exist_ok=True)
    for job in plan["jobs"]:
        entry = load_entry(job["spec"])
        seq, n = job["spec"]["unit"]["sequence"], job["frames"]
        if seq != entry["sequence"] or len(entry["frames"]) != len(entry["ground_truth"]) or len(entry["frames"]) != n+1:
            raise ValueError("Stage-121 roster differs")
        traces = {}
        for arm in ARMS:
            path = root/"evaluation"/f"{seq}__{arm}.npz"
            meta = dict(plan_fingerprint=plan["fingerprint"], sequence=seq, arm=arm)
            data = c.read_cache(path, meta)
            fresh = data is None
            if fresh:
                print(f"[Stage-153] {seq} {arm}: {n} frames", flush=True)
                data = collect(observer, entry, dataset_root, arm)
            audit(data, n, entry["initial_box"], arm)
            if arm == "native" and not plan["runtime_differences"]:
                key = f"evaluation/{base.job_key(job)}"
                io.verify({str(r121/k):d121["artifact_sha256"][k] for k in (key+".npz", key+".receipt.json")})
                old = source.prior.prior.load_trace(r121, p121, job)
                np.testing.assert_array_equal(data["boxes"], old["boxes"])
                np.testing.assert_array_equal(data["selected"], old["layer"][:,0])
            if fresh:
                observer.native.assert_frozen()
                c.save(path, data, meta)
            traces[arm] = data
            for item in (path, path.with_suffix(".receipt.json")):
                artifacts[item.relative_to(root).as_posix()] = c.file_hash(item)
        audit_prefix(*(traces[k] for k in ARMS))
        for name, rows in zip(("video_summary", "failures", "routing"), report_video(entry, traces)):
            tables[name].extend(rows)
    tables["policy_summary"], tables["paired"] = summarize(tables["video_summary"])
    hashes = {}
    for name, rows in tables.items():
        path = root/f"stage153_{name}.csv"
        write_table(path, rows, fields=EVENT_FIELDS if name == "failures" else None)
        hashes[path.name] = c.file_hash(path)
    policies = tables["policy_summary"]
    c.publish_identical(root/"stage153_decision.json", io.seal(dict(
        schema=SCHEMA, status=STATUS, role="consumed_development_adaptive_depth_comparison",
        plan_fingerprint=plan["fingerprint"], source_stage121=plan["source_stage121"],
        runtime=plan["runtime"], runtime_differences=plan["runtime_differences"], videos=len(plan["jobs"]),
        new_full_trajectories=len(tables["video_summary"]), new_search_calls=sum(r["head_calls"] for r in policies),
        new_block_calls=sum(r["block_calls"] for r in policies), extra_blocks=policies[1]["extended_frames"],
        table_sha256=hashes, artifact_sha256=artifacts, **WORK)))
    check_results(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in SOURCES:
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-results", action="store_true")
    parser.set_defaults(mode="stage153_adaptive_depth")
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if args.check_results:
        check_results(root)
        return
    if args.checkpoint is None or any(getattr(args, name) is None for name in SOURCES):
        parser.error("Supply the original checkpoint and frozen source directories")
    paths = [getattr(args, name).expanduser().resolve() for name in SOURCES]
    for name,path in zip(SOURCES, paths):
        if not path.is_dir():
            parser.error(f"Missing --{name} source directory: {path}")
        if root == path or root in path.parents or path in root.parents:
            parser.error("Use a separate Stage-153 directory")
    r76,r89,r93,r121 = paths
    with writer_lock(root):
        if (root/"stage153_decision.json").exists():
            check_results(root)
            return
        p121 = c.c66.sealed(r121/"manifest.json", source.s121.SCHEMA)
        d121 = c.c66.sealed(r121/"analysis/stage121_decision.json", source.s121.SCHEMA)
        jobs = sorted((j for j in p121["jobs"] if j["arm"] == "native"), key=base.job_key)
        if (d121["fingerprint"] != source.SOURCE121 or d121["plan_fingerprint"] != p121["fingerprint"]
                or len({j["spec"]["unit"]["sequence"] for j in jobs}) != DESIGN["videos"]
                or len(jobs) != DESIGN["videos"] or sum(j["frames"] for j in jobs) != DESIGN["tracked_frames"]):
            raise ValueError("Expected the archived Stage-121 development roster")
        s95 = base.s103.prior.s95
        p93,_,p89,d89,origin = s95.s94.load_run(r93,r76,r89)
        old = origin["original"]

        def load_entry(spec):
            return s95.load_unit(r93,r89,p93,p89,d89,origin,spec)[0]

        import torch
        torch.set_num_threads(1)
        runtime = base.s103.s76.device_runtime(old,torch,args.gpu)
        differences = base.s109.check_runtime(p121["runtime"],runtime)
        manifest = root/"manifest.json"
        if manifest.is_file() and c.c66.sealed(manifest,SCHEMA)["runtime"] != runtime:
            raise ValueError("Runtime changed on resume; use a new STAGE153_ROOT")
        plan = io.seal(dict(schema=SCHEMA, design=DESIGN, arms=list(ARMS), jobs=jobs,
                            source_stage121=d121["fingerprint"], runtime=runtime, runtime_differences=differences,
                            code_sha256=c.file_hash(Path(__file__))))
        c.publish_identical(manifest, plan)
        frozen = dict(old, fingerprint=plan["fingerprint"], runtime_identity=runtime, runtime=runtime["tracker"])
        args.parameter = runtime["tracker"]["parameter"]
        args.checkpoint = args.checkpoint.expanduser().resolve()
        native,_ = base.s103.s76.startup.checked_runtime(args,frozen,root)
        with ExitStack() as stack:
            run(root,r121,p121,d121,plan,DepthObserver(native,stack),old["root"],load_entry)
    print(f"Stage-153: {STATUS}", flush=True)


if __name__ == "__main__":
    main()
