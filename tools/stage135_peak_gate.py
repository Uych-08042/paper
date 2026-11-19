"""Evaluate predeclared one-switch Hann peak gates on cached SGLA branches."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io as textio
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import stage134_native_peak_continuation as prior
from stage27_uav_common import tracking_iou


c, io = prior.c, prior.io
SCHEMA = "stage135_cached_peak_gate_v1"
SOURCE134 = "68ea03c1dd76446b9d58c0621ff7dc783ad684d95377ea3e04b6529cf4a3dfde"
STATUS = "COMPLETE_CACHED_PEAK_GATE_DIAGNOSTIC"
DESIGN = dict(maximum_relative_score_gap=.20, maximum_pair_iou=.20,
              minimum_history_iou_advantage=.10, maximum_velocity_box_sides=.50)
NATIVE, RUNNER = prior.ARMS[:2]
GATES = ("close_score", "close_conflict", "close_conflict_history")
PRIMARY = GATES[-1]


def frozen_windows(rows):
    pairs = defaultdict(dict)
    for row in rows:
        if row["arm"] in (NATIVE, RUNNER):
            pair = pairs[row["case_id"]]
            if row["arm"] in pair:
                raise ValueError(f"Duplicate cached arm: {row['case_id']}")
            pair[row["arm"]] = row
    if any(set(pair) != {NATIVE, RUNNER} for pair in pairs.values()):
        raise ValueError("Missing native or runner-up arm")
    roster = [pair[NATIVE] for pair in pairs.values()]
    kinds = [row["kind"] for row in roster]
    if len(roster) != 47 or kinds.count("onset") != 19 or kinds.count("healthy") != 28:
        raise ValueError("Stage-134 window roster changed")
    return pairs


def online_evidence(raw, window, boxes, cells, previous, earlier):
    weighted = (np.asarray(raw, np.float32) * np.asarray(window, np.float32)).ravel()
    if not np.isfinite(weighted).all() or int(weighted.argmax()) != int(cells[0]):
        raise ValueError("Stage-134 native Hann response differs")
    first, second = map(float, weighted[cells])
    if first <= 0 or second < 0:
        raise ValueError("Invalid peak scores")
    previous = np.asarray(previous, float)
    forecast = previous.copy()
    if earlier is not None:
        earlier = np.asarray(earlier, float)
        velocity = previous[:2] + previous[2:] / 2 - earlier[:2] - earlier[2:] / 2
        cap = DESIGN["maximum_velocity_box_sides"] * float(np.sqrt(previous[2:].prod()))
        length = float(np.linalg.norm(velocity))
        if length > cap:
            velocity *= cap / length
        forecast[:2] += velocity
    native_iou, runner_iou = map(float, tracking_iou(boxes, forecast))
    return dict(distinct=bool(cells[0] != cells[1] and not np.array_equal(boxes[0], boxes[1])),
                native_score=first, runner_score=second, relative_score_gap=(first - second) / first,
                pair_iou=float(tracking_iou(boxes[0], boxes[1])), native_history_iou=native_iou,
                runner_history_iou=runner_iou, history_advantage=runner_iou - native_iou)


def gate(evidence):
    score = evidence["distinct"] and evidence["relative_score_gap"] <= DESIGN["maximum_relative_score_gap"]
    conflict = score and evidence["pair_iou"] <= DESIGN["maximum_pair_iou"]
    history = conflict and evidence["history_advantage"] >= DESIGN["minimum_history_iou_advantage"]
    return dict(zip(GATES, (score, conflict, history)))


def evaluate(pairs, evidence=None):
    windows, decisions = [], []
    for case_id, pair in pairs.items():
        native, runner = pair[NATIVE], pair[RUNNER]
        active = runner["active"] == "True"
        rescued = native["kind"] == "onset" and runner["new_tail_good_run"] == "True"
        switches = {NATIVE: False, "forced_runner_up": active}
        if evidence is not None:
            features = evidence[case_id]
            if active != features["distinct"]:
                raise ValueError(f"Stage-134 runner-up activation differs: {case_id}")
            selected = gate(features)
            switches.update(selected)
            decisions.append(dict(case_id=case_id, sequence=native["sequence"], kind=native["kind"],
                                  frame=int(native["frame"]), **features, **selected,
                                  runner_new_recovery=rescued, runner_lost_recovery=runner["lost_tail_good_run"] == "True",
                                  runner_new_tail_low=runner["new_tail_low"] == "True",
                                  runner_tail_mean_delta=float(runner["tail_mean_delta"])))
        switches["hindsight_recovery_bound"] = rescued
        for policy, use_runner in switches.items():
            row = runner if use_runner else native
            windows.append(dict(sequence=row["sequence"], kind=row["kind"], arm=policy,
                                active=use_runner, anchor_iou=float(row["anchor_iou"]),
                                anchor_delta=float(row["anchor_delta"]), tail_mean_delta=float(row["tail_mean_delta"]),
                                tail_low_frame_delta=int(row["tail_low_frame_delta"]),
                                new_tail_good_run=row["new_tail_good_run"] == "True",
                                lost_tail_good_run=row["lost_tail_good_run"] == "True",
                                new_tail_low=row["new_tail_low"] == "True",
                                first_tail_good_run=row["first_tail_good_run"] or None,
                                live_native_low_anchor=native["live_native_low_anchor"] == "True"))
    # The source's original live-low subset is identical to its complete onset stratum.
    return [row for row in prior.summarize(windows) if row["kind"] != "onset_live_low"], decisions


def verify_source(report):
    if report["fingerprint"] != SOURCE134 or report["status"] != prior.STATUS or report["runtime_differences"]:
        raise ValueError("Need the completed Stage-134 L40 source")


def bound_from_bundle(path):
    sections = {m[1].decode(): m[2] for m in re.finditer(
        rb"^===== (stage134_\S+) =====\r?\n(.*?)(?=\r?\n===== |\Z)", path.read_bytes(), re.M | re.S)}
    report = json.loads(sections["stage134_decision.json"])
    verify_source(report)
    if io.seal({k: v for k, v in report.items() if k != "fingerprint"}) != report:
        raise ValueError("Stage-134 bundle decision fingerprint mismatch")
    name = "stage134_windows.csv"
    if hashlib.sha256(sections[name]).hexdigest() != report["table_sha256"][name]:
        raise ValueError("Stage-134 bundle window hash mismatch")
    return evaluate(frozen_windows(list(csv.DictReader(textio.StringIO(sections[name].decode())))))[0]


def load_evidence(r134, r121, p134, d134, pairs):
    import torch
    from lib.test.utils.hann import hann2d

    s121 = prior.source.s121
    p121 = c.c66.sealed(r121 / "manifest.json", s121.SCHEMA)
    d121 = c.c66.sealed(r121 / "analysis/stage121_decision.json", s121.SCHEMA)
    if (d121["fingerprint"] != prior.source.SOURCE121 or d121["plan_fingerprint"] != p121["fingerprint"]
            or p134["source_stage121"] != d121["fingerprint"] or p134["runtime"] != p121["runtime"]):
        raise ValueError("Mismatched Stage-121 native source")
    jobs = {j["spec"]["unit"]["sequence"]: j for j in p121["jobs"] if j["arm"] == NATIVE}
    windows, traces, weights = {}, {}, {}
    for case in p134["cases"]:
        case_id, sequence, frame = (case[k] for k in ("case_id", "sequence", "frame"))
        pair = pairs[case_id]
        if any((row["sequence"], int(row["frame"]), row["kind"]) != (sequence, frame, case["kind"])
               for row in pair.values()):
            raise ValueError(f"Stage-134 case differs: {case_id}")
        job = jobs[sequence]
        if sequence not in traces:
            key = prior.base.job_key(job)
            paths = [f"evaluation/{key}{suffix}" for suffix in (".npz", ".receipt.json")]
            io.verify({str(r121 / name): d121["artifact_sha256"][name] for name in paths})
            trace = prior.source.prior.prior.load_trace(r121, p121, job)
            traces[sequence] = (trace["frame_id"], trace["candidate_parents"][:, 0], trace["boxes"])
        frames, parents, boxes = traces[sequence]
        index = frame - 1
        if int(frames[index]) != frame:
            raise ValueError(f"Stage-121 frame differs: {case_id}")
        cache = c.read_cache(r134 / "branches" / f"{case_id}.npz",
                             dict(plan_fingerprint=d134["plan_fingerprint"], case_id=case_id))
        if cache is None:
            raise ValueError(f"Missing Stage-134 branch cache: {case_id}")
        np.testing.assert_array_equal(cache["parent"][0, 0], parents[index])
        np.testing.assert_array_equal(cache["boxes"][0, 0], boxes[index])
        if index:
            np.testing.assert_array_equal(parents[index], boxes[index - 1])
        cells = cache["cells"][:2]
        np.testing.assert_array_equal(cells, [int(pair[a]["cell"]) for a in (NATIVE, RUNNER)])
        raw = cache["anchor_raw"]
        if raw.shape not in weights:
            weights[raw.shape] = hann2d(torch.tensor(raw.shape), centered=True).reshape(raw.shape).numpy()
        windows[case_id] = online_evidence(raw, weights[raw.shape], cache["boxes"][:2, 0], cells,
                                          parents[index], parents[index - 1] if index else None)
    if set(windows) != set(pairs):
        raise ValueError("Stage-134 case and table inventories differ")
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage134", type=Path)
    parser.add_argument("--stage121", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bound-bundle", type=Path, help="Evaluate only the outcome-based bound from the archived text")
    args = parser.parse_args()
    if args.bound_bundle:
        print(json.dumps(bound_from_bundle(args.bound_bundle), indent=2))
        return
    if not all((args.stage134, args.stage121, args.output_dir)):
        parser.error("Supply --stage134, --stage121 and --output-dir")
    r134, r121, root = [p.expanduser().resolve() for p in (args.stage134, args.stage121, args.output_dir)]
    if any(root == p or root in p.parents or p in root.parents for p in (r134, r121)):
        parser.error("Use a separate Stage-135 directory")
    prior.check_results(r134)
    p134 = c.c66.sealed(r134 / "manifest.json", prior.SCHEMA)
    d134 = c.c66.sealed(r134 / "stage134_decision.json", prior.SCHEMA)
    verify_source(d134)
    pairs = frozen_windows(prior.read_table(r134 / "stage134_windows.csv"))
    evidence = load_evidence(r134, r121, p134, d134, pairs)
    summary, decisions = evaluate(pairs, evidence)
    onset = next(r for r in summary if r["kind"] == "onset" and r["arm"] == PRIMARY)
    healthy = next(r for r in summary if r["kind"] == "healthy" and r["arm"] == PRIMARY)
    with prior.writer_lock(root):
        hashes = {}
        for name, rows in (("summary", summary), ("decisions", decisions)):
            path = root / f"stage135_{name}.csv"
            prior.publish_table(path, rows)
            hashes[path.name] = c.file_hash(path)
        c.publish_identical(root / "stage135_decision.json", io.seal(dict(
            schema=SCHEMA, status=STATUS, design=DESIGN, primary_policy=PRIMARY,
            source_stage134=SOURCE134, source_stage121=p134["source_stage121"],
            role="consumed_development_cached_one_switch_gate",
            cases=len(pairs), new_search_calls=0, training_updates=0, new_full_trajectories=0,
            primary_onset=onset, primary_healthy=healthy, table_sha256=hashes)))
    print(f"Stage-135: {STATUS}", flush=True)


if __name__ == "__main__":
    main()
