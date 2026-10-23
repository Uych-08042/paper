# Single-GPU task/resource-aware PPO: proposed experiment

Status: hardware calibration is implemented in Stage-22; see
`STAGE22_SINGLE_GPU_CALIBRATION.md`. The streaming environment and PPO trainer
remain DESIGN ONLY. No hardware result is established by this document.
The unfinished Stage-21 counterfactual prototype is a separate
idea, not an executable implementation of this protocol. Preserve Stage-18 and
Stage-20 sources, checkpoints, contracts and reported outcomes unchanged.

## Question and scope

On one server/PC GPU with variable competing load, does causal hardware
telemetry improve PPO's real-time tracking decisions beyond visual history
alone? Does learning a candidate layer add value beyond the original selector?

This is a single-GPU resource-contention experiment, not embedded-device,
multi-device, distributed-inference, or UAV flight validation. Resource-state
augmentation and PPO are not novelty claims by themselves.

## Fixed tracking representation and policy

Use the same original SGLATrack checkpoint for all methods. Freeze the original
backbone, head and selector weights. Do not include Stage-18's learned feature
memory/adapters, Stage-20 compression, or new representation training in this
first comparison. Initially use a fixed eight-camera-tick causal history window
with a small MLP controller, not a new learned recurrent tracking memory. This
keeps PPO likelihood replay simple and avoids another representation confound.

Use a hierarchical action:

1. Before any expensive current-frame backbone computation, a small CPU gate
   chooses `predict_only`, `fast`, or `refine`.
2. `predict_only` uses the common causal motion fallback, without image backbone
   or tracking-head execution. `fast` runs L1-L6 and one head.
3. `refine` runs L1-L6, then a small conditional policy selects exactly one
   independent L7-L12 block, then runs one head. The layer head may use current
   prefix features; the earlier gate must not use them. No preliminary Fast
   head is executed for deciding whether to refine.

The PPO action log probability is gate log probability plus the conditional
layer log probability only on `refine` actions. Predict/Fast carry no invented
layer action or layer likelihood. Use the identical sampling/argmax rules in
collection, likelihood replay and deployment. No layer-diversity reward, L9
preference, compulsory candidate spending or inherited P30 token constraint.

Inputs to the gate are cached visual confidence/ambiguity, box motion and scale,
time since the last accepted visual result, and optionally causal resource
observations. Device observations include GPU utilization, clocks, available
memory, completed-inference latency history, own in-flight status, and telemetry
age. Do not claim to observe other applications' GPU queue lengths via NVML.
Polling frequency is not necessarily the driver's metric refresh frequency.

The conditional layer head receives pooled current prefix features and the
same allowed history. Avoid the previous large controller's overhead. Profile
the actual gate, telemetry access, feature transfer and conditional head costs;
do not treat their costs as zero or add them artificially to static baselines.

## Streaming environment

For controlled replay, declare a 30-Hz camera clock, not an assumption that all
GOT-10k source videos were captured at 30 Hz. Metadata-backed real timestamps
can be studied under a separately frozen protocol. Do not compare these scores
directly to ordinary offline benchmark AUC.

At each capture timestamp t_i, score the available output against that frame's
GT. A result becomes available only after its actual/simulated completion time.
The action at t_i affects later outputs, not a retroactively improved result at
t_i. Use the same causal motion fallback and delayed-result handling for every
method, including always-SGLA. Define the prediction fallback from past accepted
visual boxes/timestamps only, with image-bound clipping and no GT resets after
initialization. The first implementation should use a common fixed constant-
velocity predictor, not different learned predictors for different methods.

Allow at most one non-preemptible visual job in flight. While busy, visual
actions are masked and only prediction is available; do not enqueue a backlog
of obsolete frames. At a new idle decision, process the current available frame,
not a future one. Tag every result with capture/completion timestamps and its
capture-time crop origin. A delayed result must never be decoded using the
current crop origin or scored as if it arrived on time. Its box is propagated
to the current time by the same causal fallback; reject out-of-order results.
Freeze these event semantics before any policy-outcome inspection.

Use GT only to compute training reward and evaluation quality, outside the
observation/runtime interfaces. Time-indexed IoU at the next camera tick is the
quality reward for a decision. Record a separate cost for visual jobs completing
more than 33.3 ms after capture. Optimize quality with PPO and an explicit
deadline-cost constraint (initial target: <=5% of submitted visual jobs late),
using a separate cost critic/dual update rather than searching quality/latency
reward weights on evaluation outcomes. A low late-job fraction alone is not a
success: always predicting can avoid submissions while losing the target.
Masks guarantee legality, not that all stochastic latency deadlines are met.

## Hardware load and calibration

Use only an exclusive GPU allocated to the experiment. Do not launch contention
workers on a GPU shared with other people's jobs. A controlled background worker
issues bounded short GPU kernels with a reproducible demand schedule; no global
clock/power changes or administrator privileges are required by this design.
Nominal worker duty cycle is not the same as measured GPU utilization.

First profile Predict, Fast, native SGLA and each candidate path using five fit
videos, 64 tracked frames each, three interleaved repetitions per operating
condition. Include crop/preprocessing, model and decode. Log device/model IDs,
software, warm-up, telemetry timestamps and concurrent workload. Initially
sample idle, sustained load and switching/burst load; select reproducible worker
settings using hardware timing only, not tracking-policy AUC. If all meaningful
conditions readily meet 30 Hz, report that the resource-scheduling motivation is
not established on this hardware. Do not add sleep or lower the deadline after
seeing policy outcomes to manufacture an advantage.

Collect independent calibration and assessment load realizations. Training may
use a trace-driven event simulator calibrated from measured service times, but
tracking boxes must come from actual frozen-model inference on each requested
crop, not an oracle accuracy table. Actual job decisions update the simulated
busy state and result age; do not force the same endogenous state sequence on
all policies. Only exogenous competing-demand schedules are shared.

Frozen policies must subsequently run real streaming replay on the single GPU
with unseen demand schedules. Simulation-only gains are not a hardware result.
Record whole-board power/energy only if supported. Under contention this includes
the background workload: it is not tracker-only energy, and simple subtraction
does not generally recover tracker energy. Energy is descriptive in this first
experiment, not a primary constrained quantity or an edge-energy claim.

## Comparisons

Train these three PPO variants with identical gate/layer architectures where
applicable, initialization, representation, optimizer settings and frame budget:

- V-PPO: visual/history observations; real-time hardware telemetry channels are
  constant. Keep timestamps and legal-action masks needed by every method.
- VR-PPO: visual/history plus measured causal hardware telemetry.
- VR-PPO-native: the same resource-aware gate, but `refine` uses the original
  SGLA selector instead of the learned layer policy. Retrain this gate; merely
  overriding a trained policy's layers is only an additional deployment ablation.

Also evaluate always-Fast, always-SGLA, fixed periodic visual updates, and a
causal latency-threshold scheduler. Fix the threshold/schedule using fit-only
timing/quality calibration, not final assessment outcomes. All share the same
streaming output and fallback rules. A hindsight deadline-aware oracle may be
reported separately but is never a deployable baseline.

Primary contrast: VR-PPO minus V-PPO under the predeclared switching-load test.
Secondary: VR-PPO minus VR-PPO-native, and versus each static/rule baseline.
Report original SGLA comparison, even if unfavorable. Do not claim matched cost
from a shared nominal constraint alone; show actual work, late jobs and usage.

## Data, budget and reporting

Reuse the frozen Stage-18 96 fit / 24 tune video identities, excluding source
validation videos, UAV123 and OTB. All are development evidence; the tune set
has already been used in earlier work and is not a new confirmatory test set.
Fit normalization, resource predictors and rule thresholds on fit data only.
Hold out complete load realizations, not individual frames of the same trace.

First run: five fit-pilot videos, 1,024 actual camera transitions per PPO variant,
one seed, checking optimization, event causality, delayed results and likelihood
replay. No efficacy decision comes from this smoke run.

After runtime review, fixed short development training: 32,768 camera transitions
per variant/seed; seeds 2070, 2071, 2072; rollout 1,024; minibatch 256; four PPO
epochs; LR 3e-4; gamma .97; GAE lambda .95; clip .2; entropy .01; grad clip .5.
The cost critic and dual learning rate must be implementation-tested and written
to the final contract before launch; no executable/full-run readiness is implied
by this proposed document. Use every declared seed's final checkpoint, not the
best assessment epoch or winning seed. No parallel architecture changes.

Primary metric: time-aligned success AUC over all annotated camera timestamps,
not only timestamps where a visual job completed. Also report offline AUC,
IoU<.2 failure fraction/longest run, visual-result age, deadline misses, actual
visual/candidate call fractions, latency P50/P95, controller overhead, workload
realizations, and wall-clock training time. Report actual energy only as above.

Average seeds within each video before video-clustered bootstrap. Predeclare
5,000 draws, analysis seed 2073, and report the three load conditions separately.
Do not count seeds or repeated workload runs as independent new videos.

Development support for device awareness requires a positive primary paired-CI
lower bound, a satisfied measured deadline target, and idle-case noninferiority
to always-SGLA (AUC margin 0.5 points). Layer-selection benefit is a separate
claim requiring the VR-PPO-native contrast. Ordinary offline AUC or a reward
increase alone is not a streaming resource-aware improvement. If these fail,
retain the result and do not change deadlines/workload/reward on the same
assessment outcomes. An untouched external benchmark remains necessary later.

## Research context

RL adaptive feature computation is not new: Learning Policies for Adaptive
Tracking with Deep Feature Cascades, ICCV 2017:
https://arxiv.org/abs/1708.02973

Joint latency/quality evaluation and dynamic scheduling have established roots
in Towards Streaming Perception, ECCV 2020. This protocol adapts the principle
to tracking; it does not claim a new streaming metric:
https://arxiv.org/abs/2005.10420

Constraints are not guaranteed merely by adding a penalty to PPO. For context,
see Constrained Policy Optimization, ICML 2017:
https://proceedings.mlr.press/v70/achiam17a.html
