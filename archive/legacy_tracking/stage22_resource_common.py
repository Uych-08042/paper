"""Dependency-light contracts and helpers for single-GPU timing calibration."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint, frame_path, load_annotations


SCHEMA = "stage22_single_gpu_calibration_v1"
OPERATIONS = ("predict_only", "fast", "sgla") + tuple("l%d" % n for n in range(7, 13))
CONDITIONS = ("idle", "sustained", "switching")
RAW_FIELDS = (
    "run_fingerprint", "condition", "sequence", "frame_id", "repeat", "order",
    "operation", "crop_state_fingerprint", "start_ns", "end_ns", "latency_ms",
    "condition_elapsed_s", "phase", "nominal_duty", "timing_basis",
    "telemetry_sample_id", "telemetry_end_ns", "telemetry_age_ms", "telemetry_lookup_ms",
)
LOAD_FIELDS = ("start_ns", "end_ns", "elapsed_s", "phase", "nominal_duty",
               "kernels", "busy_ms", "maximum_kernel_ms")


def sealed(body):
    return dict(body, fingerprint=fingerprint(body))


def read_sealed(path, schema=None):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or (schema and data.get("schema") != schema)
            or fingerprint({k: v for k, v in data.items() if k != "fingerprint"})
            != data.get("fingerprint")):
        raise ValueError("Invalid fingerprint/schema: " + str(path))
    return data


def load_phase(condition, elapsed_s, config):
    if condition not in CONDITIONS or not math.isfinite(elapsed_s) or elapsed_s < 0:
        raise ValueError("Invalid load condition/time")
    if condition == "idle":
        return 0, 0.0
    if condition == "sustained":
        return 0, config["sustained_duty"]
    phase = int(elapsed_s / config["switch_hold_s"])
    return phase, config["switching_duty"] if phase % 2 else 0.0


def interleaved_order(seed, condition, sequence, frame_id, repeat):
    digest = fingerprint([seed, condition, sequence, frame_id, repeat])
    rng = np.random.default_rng(int(digest[:16], 16))
    return [OPERATIONS[i] for i in rng.permutation(len(OPERATIONS))]


def predict_box(history, capture_time_s, image_hw):
    """Constant center velocity from past accepted boxes; retain last box size."""
    if not history:
        raise ValueError("Prediction requires an initialized visual history")
    timestamp, box = history[-1]
    box = np.asarray(box, dtype=np.float64).copy()
    h, w = map(int, image_hw)
    if (box.shape != (4,) or not np.isfinite(box).all() or min(box[2:]) <= 0
            or min(h, w) <= 0 or not math.isfinite(capture_time_s)
            or not math.isfinite(timestamp) or capture_time_s < timestamp):
        raise ValueError("Invalid motion-predictor inputs")
    if len(history) >= 2:
        older_t, older_box = history[-2]
        if not math.isfinite(older_t) or older_t >= timestamp:
            raise ValueError("Accepted visual timestamps must strictly increase")
        older_box = np.asarray(older_box, dtype=np.float64)
        if older_box.shape != (4,) or not np.isfinite(older_box).all() or min(older_box[2:]) <= 0:
            raise ValueError("Invalid previous visual box")
        velocity = ((box[:2] + box[2:] / 2) - (older_box[:2] + older_box[2:] / 2))
        box[:2] += velocity * (capture_time_s - timestamp) / (timestamp - older_t)
    box[2:] = np.clip(box[2:], 1, [w, h])
    box[:2] = np.clip(box[:2], 0, np.asarray([w, h]) - box[2:])
    return box.tolist()


def run_operation(rt, image, operation, history, capture_time_s):
    """One complete path. Caller restores the shared crop outside timing."""
    if operation == "predict_only":
        return predict_box(history, capture_time_s, image.shape[:2])
    if operation == "sgla":
        return rt.tracker.track(image)["target_bbox"]
    if operation not in OPERATIONS:
        raise ValueError("Unknown operation: " + operation)
    search, factor = rt.prepare(image)
    tokens = rt.prefix(search)
    output = rt.head(tokens) if operation == "fast" else rt.candidate(tokens, int(operation[1:]))
    return rt.decode(output, factor, image)


def restore_call(rt, fn):
    state, frame = list(rt.tracker.state), rt.tracker.frame_id
    try:
        return fn()
    finally:
        rt.tracker.state, rt.tracker.frame_id = state, frame


def check_execution_counts(operation, counts):
    if operation not in OPERATIONS or not isinstance(counts, dict):
        raise ValueError("Unknown path/call-count record")
    allowed = {"head"} | {"l%d" % n for n in range(1, 13)}
    if not set(counts).issubset(allowed) or any(type(v) is not int or v < 0 for v in counts.values()):
        raise ValueError("Invalid module call counts")
    prefix = 0 if operation == "predict_only" else 1
    if any(counts.get("l%d" % n, 0) != prefix for n in range(1, 7)):
        raise ValueError("Path did not execute exactly one L1-L6 prefix: " + operation)
    candidates = [n for n in range(7, 13) if counts.get("l%d" % n, 0)]
    expected = 0 if operation in ("predict_only", "fast") else 1
    if (len(candidates) != expected or counts.get("head", 0) != prefix
            or any(counts.get("l%d" % n, 0) > 1 for n in range(7, 13))):
        raise ValueError("Path has an extra candidate block/head: " + operation)
    if operation.startswith("l") and candidates != [int(operation[1:])]:
        raise ValueError("Wrong independent candidate: " + operation)


def calibration_sources(manifest, bundle, frames, videos):
    if manifest["fingerprint"] != bundle["source_manifest_fingerprint"]:
        raise ValueError("Manifest differs from the frozen split bundle")
    names = bundle["pilot_videos"][:videos]
    if (len(names) != videos or not set(names).issubset(bundle["fit_videos"])
            or set(names) & set(bundle["tune_videos"])):
        raise ValueError("Calibration must use the frozen fit-pilot videos only")
    entries = {e["sequence"]: e for e in manifest["sequences"] if e["split"] == "train"}
    sources = []
    for name in names:
        entry = entries[name]
        directory, gt, valid = load_annotations(manifest["root"], manifest["dataset"], name)
        start, stop = entry["start"], entry["start"] + frames + 1
        if (stop > entry["stop"] or stop > len(gt) or not valid[start]
                or file_hash(directory / "groundtruth.txt") != entry["annotation_sha256"]
                or fingerprint(valid.tolist()) != entry["visible_sha256"]):
            raise ValueError("Changed/incomplete source: " + name)
        paths = [frame_path(directory, manifest["dataset"], i) for i in range(start, stop)]
        images = [dict(path=str(p.resolve()), sha256=file_hash(p)) for p in paths]
        # No later GT boxes or quality outcomes are exposed to the profiler.
        sources.append(dict(sequence=name, start=start, init_box=gt[start].tolist(), images=images))
    return sources


class CsvJournal:
    def __init__(self, path, fields):
        self.handle = Path(path).open("x", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=fields)
        self.writer.writeheader()
        self.count = 0

    def append(self, row):
        self.writer.writerow(row)
        self.count += 1

    def flush(self):
        self.handle.flush()

    def close(self):
        self.handle.close()


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if len(set(reader.fieldnames or [])) != len(reader.fieldnames or []):
            raise ValueError("Duplicate CSV columns")
        rows = list(reader)
        if any(None in r or None in r.values() for r in rows):
            raise ValueError("Malformed CSV: " + str(path))
        return rows


def summarize_samples(run, records):
    """Require the exact Cartesian measurement grid and causal telemetry links."""
    config = run["config"]
    sources = {s["sequence"]: s for s in run["sources"]}
    design = (5, 64, 3) if config["mode"] == "full" else (1, 8, 1)
    if (config["mode"] not in ("smoke", "full")
            or (len(sources), config["frames"], config["repeats"]) != design
            or config["videos"] != design[0] or len(sources) != len(run["sources"])
            or not run["conditions"] or len(set(run["conditions"])) != len(run["conditions"])
            or not set(run["conditions"]).issubset(CONDITIONS)):
        raise ValueError("Calibration differs from the fixed smoke/full sampling design")
    expected = {(c, s, f, r, op) for c in run["conditions"] for s in sources
                for f in range(sources[s]["start"] + 1, sources[s]["start"] + config["frames"] + 1)
                for r in range(config["repeats"]) for op in OPERATIONS}
    groups, states, seen = defaultdict(list), {}, set()
    for row in records:
        if set(row) != set(RAW_FIELDS):
            raise ValueError("Unexpected/missing latency columns")
        key = (row["condition"], row["sequence"], int(row["frame_id"]),
               int(row["repeat"]), row["operation"])
        if row["run_fingerprint"] != run["fingerprint"] or key not in expected or key in seen:
            raise ValueError("Duplicate, unexpected or mixed-run measurement")
        seen.add(key)
        condition, sequence, frame, repeat, op = key
        order = interleaved_order(config["seed"], condition, sequence, frame, repeat)
        order_index = int(row["order"])
        if not 0 <= order_index < len(order) or order[order_index] != op:
            raise ValueError("Measurement order differs from frozen interleaving")
        start, end = int(row["start_ns"]), int(row["end_ns"])
        latency = float(row["latency_ms"])
        if (start < 0 or end <= start or not math.isfinite(latency)
                or abs(latency - (end - start) / 1e6) > 1e-8):
            raise ValueError("Invalid monotonic latency")
        elapsed = float(row["condition_elapsed_s"])
        phase, duty = load_phase(condition, elapsed, config)
        nominal = float(row["nominal_duty"])
        if not math.isfinite(nominal) or int(row["phase"]) != phase or abs(nominal - duty) > 1e-12:
            raise ValueError("Load-demand phase mismatch")
        timing = "cpu_wall" if op == "predict_only" else "cuda_synchronized_host_wall"
        if row["timing_basis"] != timing:
            raise ValueError("Unexpected timing basis")
        if row["telemetry_sample_id"] != "":
            sample_end = int(row["telemetry_end_ns"])
            age = float(row["telemetry_age_ms"])
            if (sample_end < 0 or not math.isfinite(age) or sample_end > start
                    or abs(age - (start - sample_end) / 1e6) > 1e-8):
                raise ValueError("Future telemetry leaked into pre-action observation")
        elif row["telemetry_end_ns"] != "" or row["telemetry_age_ms"] != "":
            raise ValueError("Telemetry ID missing")
        lookup = float(row["telemetry_lookup_ms"])
        if not math.isfinite(lookup) or lookup < 0:
            raise ValueError("Invalid telemetry lookup timing")
        state_key = sequence, frame
        digest = row["crop_state_fingerprint"]
        if len(digest) != 64 or (state_key in states and states[state_key] != digest):
            raise ValueError("Measured paths do not share the same reference crop")
        states[state_key] = digest
        groups[condition, op].append(row)
    if seen != expected:
        raise ValueError("Incomplete measurement grid: %d of %d" % (len(seen), len(expected)))
    reports = []
    for (condition, operation), rows in sorted(groups.items()):
        times = np.asarray([float(r["latency_ms"]) for r in rows])
        reports.append(dict(condition=condition, operation=operation, samples=len(rows),
                            videos=len({r["sequence"] for r in rows}), mean_ms=float(times.mean()),
                            p50_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95)),
                            max_ms=float(times.max()),
                            deadline_miss_ratio=float(np.mean(times > config["deadline_ms"]))))
    return reports
