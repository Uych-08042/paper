"""CPU-only integrity audit and deadline-pressure report for Stage-22 timing."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_csv, write_json
from stage22_resource_common import (
    CONDITIONS, OPERATIONS, SCHEMA, check_execution_counts, load_phase, read_csv, read_sealed, sealed,
    summarize_samples,
)


def audit_files(directory):
    directory = Path(directory).expanduser().resolve()
    run = read_sealed(directory / "run.json", SCHEMA)
    if (directory / "interrupted.json").exists():
        raise ValueError("Interrupted calibration; no complete result can be reported")
    completion = read_sealed(directory / "completion.json", SCHEMA + "_completion")
    expected_files = {"latency_raw.csv", "telemetry.csv", "conditions.json"}
    expected_files.update("load_%s.csv" % c for c in run["conditions"] if c != "idle")
    if (completion["run_fingerprint"] != run["fingerprint"]
            or set(completion["files"]) != expected_files):
        raise ValueError("Completion record has a different run or artifact list")
    for name, digest in completion["files"].items():
        if file_hash(directory / name) != digest:
            raise ValueError("Completed calibration artifact changed: " + name)
    records = read_csv(directory / "latency_raw.csv")
    if len(records) != completion["samples"]:
        raise ValueError("Completion sample count mismatch")
    reports = summarize_samples(run, records)
    conditions = json.loads((directory / "conditions.json").read_text(encoding="utf-8"))
    if [c["condition"] for c in conditions] != run["conditions"]:
        raise ValueError("Condition list/order mismatch")
    timelines = {c["condition"]: c for c in conditions}
    telemetry = read_csv(directory / "telemetry.csv")
    telemetry_by_id = {}
    for index, row in enumerate(telemetry):
        if (int(row["sample_id"]) != index or int(row["start_ns"]) > int(row["end_ns"])
                or row["gpu_uuid"] != run["hardware"]["uuid"]):
            raise ValueError("Telemetry identity or sample order mismatch")
        telemetry_by_id[index] = row
    for row in records:
        timeline = timelines[row["condition"]]
        start, end = int(row["start_ns"]), int(row["end_ns"])
        if (not timeline["origin_ns"] <= start < end <= timeline["end_ns"]
                or abs((start - timeline["origin_ns"]) / 1e9 - float(row["condition_elapsed_s"])) > 1e-8):
            raise ValueError("Measurement outside condition timeline")
        if row["telemetry_sample_id"] != "":
            sample = telemetry_by_id.get(int(row["telemetry_sample_id"]))
            if (sample is None or sample["error"] or sample["end_ns"] != row["telemetry_end_ns"]):
                raise ValueError("Telemetry observation does not link to a valid recorded sample")
    load_reports = []
    for condition, timeline in timelines.items():
        checks = timeline["execution_checks"]
        if set(checks) != {s["sequence"] for s in run["sources"]}:
            raise ValueError("Missing per-clip execution verification")
        if any(not set(OPERATIONS).issubset(c) for c in checks.values()):
            raise ValueError("Missing operation integrity checks")
        for clip in checks.values():
            for operation in OPERATIONS:
                check_execution_counts(operation, clip[operation])
            difference = clip.get("native_scripted_bbox_max_abs_diff")
            if difference is None or not math.isfinite(difference) or not 0 <= difference <= 1e-5:
                raise ValueError("Native/scripted SGLA equivalence check failed")
        if condition == "idle":
            if timeline["worker"] is not None:
                raise ValueError("Idle condition unexpectedly ran a load worker")
            continue
        worker = timeline["worker"]
        if (not worker or worker["status"] != "ready"
                or worker["gpu_uuid"] != run["hardware"]["uuid"]
                or worker["idle_max_kernel_ms"] > run["config"]["maximum_idle_kernel_ms"]):
            raise ValueError("Load worker GPU identity or idle kernel safety bound failed")
        cycles = read_csv(directory / ("load_%s.csv" % condition))
        if not cycles:
            raise ValueError("Load worker has no recorded cycles")
        previous_end = timeline["origin_ns"]
        for cycle in cycles:
            start, end = int(cycle["start_ns"]), int(cycle["end_ns"])
            elapsed = float(cycle["elapsed_s"])
            phase, duty = load_phase(condition, elapsed, run["config"])
            numeric = (elapsed, float(cycle["busy_ms"]), float(cycle["maximum_kernel_ms"]),
                       float(cycle["nominal_duty"]))
            if (not all(math.isfinite(v) for v in numeric) or end <= start or start < previous_end
                    or abs((start - timeline["origin_ns"]) / 1e9 - elapsed) > 1e-8
                    or int(cycle["phase"]) != phase or float(cycle["nominal_duty"]) != duty
                    or int(cycle["kernels"]) < 0 or float(cycle["busy_ms"]) < 0
                    or (duty == 0 and int(cycle["kernels"]) != 0)):
                raise ValueError("Invalid background workload timeline")
            previous_end = end
        if not any(int(c["kernels"]) > 0 for c in cycles):
            raise ValueError("Authorized load condition never issued a GPU kernel")
        load_reports.append(dict(condition=condition, cycles=len(cycles),
            kernels=sum(int(c["kernels"]) for c in cycles),
            busy_wall_ms=sum(float(c["busy_ms"]) for c in cycles),
            maximum_kernel_ms=max(float(c["maximum_kernel_ms"]) for c in cycles)))
    return run, completion, records, reports, telemetry, load_reports


def calibration_decision(run, completion, records, reports, telemetry, load_reports):
    sample_map = {int(s["sample_id"]): s for s in telemetry}
    observed = []
    for row in records:
        sample = sample_map.get(int(row["telemetry_sample_id"])) if row["telemetry_sample_id"] != "" else None
        if (sample and sample["utilization.gpu"] != "" and sample["memory.free"] != ""
                and float(row["telemetry_age_ms"]) <= 1000):
            observed.append(row)
    telemetry_fraction = len(observed) / len(records)
    phase_coverage = {}
    for op in OPERATIONS:
        rows = [r for r in records if r["condition"] == "switching" and r["operation"] == op]
        phase_coverage[op] = dict(low=sum(float(r["nominal_duty"]) == 0 for r in rows),
                                  high=sum(float(r["nominal_duty"]) > 0 for r in rows))
    switching_covered = all(min(c.values()) >= 10 for c in phase_coverage.values())
    full_design = (run["config"]["mode"] == "full" and set(run["conditions"]) == set(CONDITIONS))
    sgla = {r["condition"]: r for r in reports if r["operation"] == "sgla"}
    pressure = None
    if full_design and switching_covered:
        if sgla["idle"]["deadline_miss_ratio"] > .05:
            pressure = "IDLE_BASELINE_ALREADY_OVER_DEADLINE"
        elif any(r["deadline_miss_ratio"] > .05 for c, r in sgla.items() if c != "idle"):
            pressure = "LOAD_DEPENDENT_DEADLINE_PRESSURE_OBSERVED"
        else:
            pressure = "NO_MEANINGFUL_DEADLINE_PRESSURE_OBSERVED"
    status = "COMPLETE_CALIBRATION" if full_design and switching_covered else "PILOT_CALIBRATION_ONLY"
    ready = bool(full_design and switching_covered and telemetry_fraction >= .9
                 and pressure == "LOAD_DEPENDENT_DEADLINE_PRESSURE_OBSERVED")
    return sealed(dict(schema=SCHEMA + "_decision", status=status, run_fingerprint=run["fingerprint"],
        completion_fingerprint=completion["fingerprint"], deadline_ms=run["config"]["deadline_ms"],
        timely_telemetry_fraction=telemetry_fraction, minimum_telemetry_fraction=.9,
        maximum_pre_action_telemetry_age_ms=1000, switching_phase_counts=phase_coverage,
        switching_coverage_pass=switching_covered, minimum_samples_per_phase_and_operation=10,
        resource_pressure=pressure, environment_ready_for_next_design_step=ready,
        load_worker_summary=load_reports,
        next_step=("Review raw traces and freeze the streaming/PPO implementation contract."
                   if ready else "Do not start full PPO. Review pilot coverage, telemetry and deadline headroom first."),
        warning="Calibration only: no learned policy, streaming gain, speedup or tracker-only energy claim. "
                "Repeated frames are timing observations, not independent tracking videos. "
                "Nominal load duty is not measured utilization. Do not lower the deadline or add tracker sleeps."))


def summarize_telemetry(run, records, telemetry):
    reports = []
    for condition in run["conditions"]:
        measured = [r for r in records if r["condition"] == condition]
        begin = min(int(r["start_ns"]) for r in measured)
        end = max(int(r["end_ns"]) for r in measured)
        samples = [s for s in telemetry if begin <= int(s["end_ns"]) <= end and not s["error"]]
        def values(key):
            return [float(s[key]) for s in samples if s[key] != ""]
        def mean(key):
            data = values(key)
            return float(np.mean(data)) if data else ""
        utilization = values("utilization.gpu")
        lookup = [float(r["telemetry_lookup_ms"]) for r in measured]
        query_times = [(int(s["end_ns"]) - int(s["start_ns"])) / 1e6 for s in samples]
        reports.append(dict(condition=condition, valid_samples=len(samples),
            utilization_samples=len(utilization), mean_gpu_util_percent=mean("utilization.gpu"),
            p95_gpu_util_percent=float(np.percentile(utilization, 95)) if utilization else "",
            mean_free_memory_mib=mean("memory.free"), mean_sm_clock_mhz=mean("clocks.current.sm"),
            mean_whole_board_power_w=mean("power.draw"),
            mean_query_wall_ms=float(np.mean(query_times)) if query_times else "",
            mean_lookup_ms=float(np.mean(lookup)), p95_lookup_ms=float(np.percentile(lookup, 95)),
            scope="Whole GPU during profiling window: includes tracker, background and instrumentation; not worker-only"))
    return reports


def summarize_run(directory):
    directory = Path(directory).expanduser().resolve()
    values = audit_files(directory)
    run, completion, records, reports, telemetry, loads = values
    decision = calibration_decision(*values)
    write_csv(directory / "latency_summary.csv", reports)
    write_csv(directory / "telemetry_summary.csv", summarize_telemetry(run, records, telemetry))
    write_json(directory / "stage22_calibration_decision.json", decision)
    print("condition   operation       samples    mean_ms     p95_ms    late%")
    for row in reports:
        print("{condition:<11} {operation:<15} {samples:>7d} {mean_ms:>10.3f} {p95_ms:>10.3f} {late:>8.2f}".format(
            **row, late=100 * row["deadline_miss_ratio"]))
    print("Calibration status: " + decision["status"])
    print("Resource pressure: " + str(decision["resource_pressure"]))
    print("Reports: " + str(directory))
    print("No PPO policy has been trained or evaluated by this calibration.")
    return decision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    args = parser.parse_args(argv)
    if not args.input.strip():
        parser.error("--input must not be empty")
    summarize_run(args.input)


if __name__ == "__main__":
    main()
