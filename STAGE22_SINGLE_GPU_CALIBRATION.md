# Stage-22: single-GPU calibration for the new resource-aware PPO design

Implemented: real GPU-contention timing and CPU-only integrity reports.
Not implemented here: the streaming event environment, controller, PPO trainer,
or policy evaluation. This is the first hardware gate in
`SINGLE_GPU_PPO_EXPERIMENT.md`, not a continuation of Stage-21.

## Preconditions

- Run on your Linux tracking server with its existing `sglatrack-test` CUDA
  environment, original SGLATrack checkpoint and installed training videos.
- Obtain an **exclusive full GPU allocation** before running. The confirmation
  flag does not reserve a GPU. Do not interfere with another user's job.
- `nvidia-smi` must be available. No NVML Python package is required. MIG and
  MPS are outside this initial implementation; no device-wide settings change.
- The verified Stage-18 representation bundle supplies only frozen fit/tune
  identities and the original tracker lineage. Its learned memory/adapters and
  PPO checkpoints are **not loaded**. No tune/validation images are profiled.
- Stage-18/20 code and artifacts remain unchanged. Do not regenerate their
  contracts to make a mismatch disappear.

## 1. Local Tests

```bash
python -B tracking/test_stage22_device_calibration.py
python -B tracking/stage22_device_calibrate.py --help
```

These tests use synthetic fixtures, not hardware or tracking-quality evidence.
Help, mathematics, telemetry mocks and summary audits require no torch/CUDA.

## 2. Server Smoke

Run from the repository root. Define paths explicitly to avoid empty shell
variables resolving to the working directory or `/filename`.

```bash
export STAGE18_ROOT="$PWD/output/stage18/temporal_layer_policy_v1"
export STAGE18_REP_BUNDLE="$STAGE18_ROOT/frozen_representation_v1.json"
export INTERVENTION_MANIFEST="$PWD/output/stage8/l9_intervention_v1/intervention_manifest.json"
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
export STAGE22_ROOT="$PWD/output/stage22/single_gpu_resource_v1"

python tracking/stage22_device_calibrate.py \
  --bundle "${STAGE18_REP_BUNDLE:?Set frozen split bundle}" \
  --manifest "${INTERVENTION_MANIFEST:?Set manifest}" \
  --checkpoint "${CHECKPOINT:?Set original checkpoint}" \
  --parameter deit_distilled --gpu 0 \
  --mode smoke \
  --exclusive_gpu_confirmed --require_telemetry \
  --output_dir "$STAGE22_ROOT/smoke_v1"
```

`--gpu` is a CUDA-visible logical index, including under `CUDA_VISIBLE_DEVICES`.
The process is mapped to its physical UUID before telemetry/worker startup.
Known competing compute PIDs cause rejection; an unambiguous mapping is required.
This is not a general detector of jobs hidden by containers, permissions or MPS.

### Older Torch / Container Identity Check

If torch omits its UUID property or nvidia-smi cannot see the current PID,
identity resolution reads the selected current CUDA context's UUID (or PCI
bus ID) through the installed driver, then verifies it with nvidia-smi. It
does not assume logical index 0 equals physical index 0, nor map by GPU name or
memory size. No CUDA toolkit headers, sudo, torch upgrade or extra package is
needed. Explicit procfs `NSpid` aliases are considered when checking processes.
The same resolver is used inside the spawned workload process.

For a diagnostic without model weights, dataset access or a load worker:

```bash
python tracking/stage22_gpu_hardware.py --gpu 0
```

Expect `GPU_IDENTITY_RESOLVED`, a physical `uuid`, and the `mapping` method.
Review `other_compute_pids` before starting the smoke. If PID namespaces remain
unaligned, listed PIDs are not proof of another job; verify the exclusive
allocation on the host rather than killing PIDs or bypassing the safety check.
If identity still fails, the error now includes the CUDA driver query error,
torch UUID representation, PID aliases and `CUDA_VISIBLE_DEVICES` for diagnosis.
An identity-resolution failure occurs before `run.json` or a worker is created.

Smoke measures one frozen fit-pilot video, eight tracked frames and one
interleaved repeat per condition. It checks nine paths:

- CPU-only constant-velocity prediction from past visual boxes;
- Fast: prefix L1-L6, then one head;
- **original native SGLA**, including its selector and state update;
- six independent L7-L12 paths: prefix, exactly one candidate, one head.

Untimed module hooks verify actual prefix/candidate/head call counts. Native
and scripted SGLA decoded boxes are checked for agreement. Hooks are removed
before timing. No Fast head is run before the fixed-layer paths.

The three conditions are idle, sustained nominal 60% demand, and alternating
zero/85% nominal demand with one-second holds. A separate spawned process uses
one synchronized FP32 matmul at a time; nominal duty is **not GPU utilization**.
Default matrix side is 1024, with a 5 ms idle single-kernel safety bound. The
observed contended kernel time can be longer and is recorded. Startup/settling,
template initialization and warm-up are untimed; tracker latency is never
inflated by sleeps. The apparatus settling wait spans a full switching cycle
even during the short smoke.

The background worker stops on normal exit, errors, Ctrl-C, and detected parent
death. Abrupt machine/driver failure cannot guarantee Python cleanup. Never
delete a partial directory or claim its journals are a complete result.

If the idle worker safety check fails, review hardware timing only; a smaller
matrix may be declared in a **new** smoke directory. Preserve failed metadata.
Copy the same worker settings into the subsequent full calibration. Do not
change the 30 Hz deadline to obtain a favorable result.

## 3. Full Timing Calibration

Only after reviewing the smoke, run the fixed five fit-pilot videos, 64 tracked
frames each and three repeats in each condition. This yields **25,920 path
measurements**, not 25,920 independent tracking examples.

```bash
python tracking/stage22_device_calibrate.py \
  --bundle "${STAGE18_REP_BUNDLE:?}" \
  --manifest "${INTERVENTION_MANIFEST:?}" \
  --checkpoint "${CHECKPOINT:?}" \
  --parameter deit_distilled --gpu 0 \
  --mode full \
  --exclusive_gpu_confirmed --require_telemetry \
  --output_dir "$STAGE22_ROOT/calibration_v1"
```

Every output directory must be **new**. There is no overwrite/resume option:
wall-clock load realizations cannot be spliced into an interrupted calibration.
The full run is a calibration realization, not the later assessment trace.

Each path/repeat uses the same crop from a common native-SGLA trajectory.
Reference state advances once per frame outside timing, without later GT boxes.
Measurement order is reproducibly shuffled per frame/repeat/condition. Images
are loaded outside timing. Visual path timing includes crop, preprocessing,
network, decoding and its own-context CUDA completion. Prediction does not
synchronize or invoke CUDA. GPU polling queries and warm-up can still contend
with measured work; their timestamps are retained.

## 4. Inspect Reports

```bash
python tracking/stage22_device_summary.py \
  --input "$STAGE22_ROOT/calibration_v1"
```

Files:

- `run.json`: fingerprinted configuration, original checkpoint, frozen split,
  input-image hashes, model/source hashes, hardware and software identity.
- `latency_raw.csv`: operation, common crop ID, monotonic start/completion,
  load-demand phase and **pre-action** telemetry link. Telemetry lookup cost is
  separately timed, not claimed to be zero controller overhead.
- `telemetry.csv`: timestamped GPU utilization, memory, clocks, temperature and
  whole-board power when supported. Unavailable values are empty, not zero.
- `telemetry_summary.csv`: measured utilization, clocks, memory, board power,
  query time and lookup time by profiling condition, not worker-only load.
- `load_sustained.csv`, `load_switching.csv`: actual worker kernel activity.
- `conditions.json`: condition origins, worker identity and execution checks.
- `completion.json`: hashes of all raw artifacts, written only after clean exit.
- `latency_summary.csv`: sample counts, mean/P50/P95/max and fraction above
  **33.333 ms service time**. These are not yet camera-to-output deadlines.
- `stage22_calibration_decision.json`: completeness, telemetry/phase coverage
  and descriptive deadline-pressure classification. No AUC is selected here.

The summary rejects missing/duplicate samples, changed raw files, mixed run IDs,
future telemetry, changed shared crops, invalid timing and inconsistent load
phases. A smoke/subset or fewer than ten samples in each switching phase per
operation remains `PILOT_CALIBRATION_ONLY`.

For the full design, at least 90% of path observations must have readable
utilization/free-memory telemetry at most one second old to mark the environment
ready. Faster polling does not imply the driver refreshes metrics equally fast.
The descriptive timing classification is one of:

- `NO_MEANINGFUL_DEADLINE_PRESSURE_OBSERVED`: native SGLA has <=5% late service
  times in all conditions. The hardware motivation is not established here.
- `IDLE_BASELINE_ALREADY_OVER_DEADLINE`: even idle SGLA exceeds the 5% target;
  investigate the base runtime before calling it load-adaptive scheduling.
- `LOAD_DEPENDENT_DEADLINE_PRESSURE_OBSERVED`: idle SGLA meets that target and
  at least one loaded condition does not. This supports building the streaming
  environment, **not** a claim that PPO will help.

This controlled workload is not representative of all production applications.
Whole-board power includes the worker, cannot be attributed to tracker energy,
and is not integrated into a primary energy claim. No speedup is inferred from
candidate count. No learned gate/feature-transfer overhead has been measured yet.
Keep OTB/UAV123 unavailable for new policy selection.

Telemetry query behavior follows the read-only
[NVIDIA nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html).
Context/device identity queries follow the
[CUDA Driver API](https://docs.nvidia.com/cuda/archive/12.6.1/cuda-driver-api/group__CUDA__CTX.html).
