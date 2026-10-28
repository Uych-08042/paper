# Stage-23B: Fixed-Budget Three-Seed Accuracy Experiment

## Why This Run

The reported Stage-23 smoke completed all three 1,024-live / 4,480-branch
transition quotas with likelihood replay errors around 2.4e-7. It passed
implementation checks. The latest fit IoUs (0.9230 / 0.9245 / 0.9228) are
not paired deployment AUCs and do not rank the three methods.

This run tests the accuracy/long-term-credit hypothesis. It does not reopen
Stage-22 device pressure or make a new speed claim. Existing Stage-23 smoke
code and outputs are not edited. Startup verifies the passed smoke's code,
bundle identity, matched counts, and saved checkpoint bytes without loading
those trained smoke policies as initial weights.

## Frozen Split Compatibility Correction

The initial Stage-23B script incorrectly required exactly 96 fit / 24 tune
videos. Stage-18 first filters training clips by the pilot's frame requirement
(64 by default), so the verified frozen split can have different counts.
The correction uses the existing bundle's exact identities/order, including
its representation rows, and checks disjointness and train-only membership.
It never resamples, drops or adds videos to achieve a nominal count. Training
steps, seeds, objectives and decision thresholds are unchanged.

The reported split-count error occurs before GPU runtime construction and
before publishing a Stage-23B contract/checkpoint. Sync both Stage-23B Python
entry points and their test, then rerun the same command/output directory for
that failed attempt. Do not regenerate the Stage-18 bundle, edit its JSON, or
rerun passed smoke. If a separate directory already contains a started formal
run under older code, do not overwrite its contract or bypass its hashes.

## Fixed Plan

- Arms: ordinary `ppo_gae`, immediate `ppo_cf_h1`, primary `ppo_cf_h20`.
- Seeds: 2074, 2075, 2076, paired across arms. Fresh policy initialization.
- Representation, backbone and head: the original frozen Stage-18 bundle.
- Data: exactly the fit/tune video IDs and order in the verified Stage-18
  representation bundle. Tune videos are development evaluation only. The
  original validation split, OTB and UAV123 are not loaded. The nominal 96/24
  counts apply only when all 120 original train clips meet Stage-18's 64-frame
  eligibility rule. Do not reconstruct, pad or randomly resample the split.
  Actual counts are printed at startup and recorded in the new run contract.
- Fit clip maximum: 128 tracked frames; tune clip maximum: 256. Original
  manifest stop points still apply; clips are not extended to meet the cap.
- Per seed/arm: 32,768 live transitions in 128 rollouts of 256.
- Same H20 branches, gamma 0.97, two shared-draw continuations, four anchors
  per rollout, credit normalization and 0.5 auxiliary weight as smoke.
- 143,360 branch transitions/seed/arm; 176,128 live+branch transitions.
  Nine runs total 1,585,152 tracking transitions, plus control/final evaluation
  and value-only prefixes. This is not a 294,912-frame training workload.
- Same Adam, GAE, clipping, entropy, value weight and four update epochs as
  smoke. Realized candidate FLOPs and wall times are not forced equal.
- Final checkpoint only. No early stopping, best-step selection, best-seed
  selection, intermediate tune evaluation, or outcome-driven budget extension.
- Deterministic argmax deployment, under the same causal P30 maximum budget.
  Do not switch to stochastic deployment after viewing the results.
- No diversity constraint: Fast-only or fixed-layer collapse must be reported,
  not hidden by forcing candidate usage or a minimum layer count.

Legitimate invalid-label rollouts need explicit handling for the larger fit
set. Missing labels never reset tracking. Invalid rewards are zero and losses
are masked. A rollout with no valid branch anchors gets zero auxiliary credit
in all arms; a rollout with no valid live labels skips optimization. Quotas,
valid-label counts and actual update counts are audited across paired arms.
This differs from smoke's fail-fast data-quality rule, not its valid-data
objective. A tensor test checks valid-data optimizer parity against smoke.

## Closed-Loop Evaluation

Every policy/control starts from GT once, then uses its own predicted boxes,
crops and memory. There are no evaluation branches, future labels, or GT
resets. Greedy policy evaluation executes one head per tracked frame.

Controls are original Fast, original SGLA, same-representation memory-Fast,
each static L7-L12 with the same causal P30 cap, and one reproducible
random-layer P30 control. All six static layers are specified in advance;
none is chosen as a new method from tune outcomes. The frozen controls need
only be collected once, not once per PPO seed. Their predictions do not
depend on the freshly initialized policy encoder/critic. Fixed controls
currently retain actor-forward overhead, so their timings are not optimized
deployment implementations.

The reported AUC averages 21 strict IoU-threshold success rates on valid
tracked frames only, excluding the initial GT frame. Each video's AUC has
equal macro weight. Invalid frames still advance the environment. This new
metric identifier is recorded, so do not subtract historical Stage-18 AUCs
that included the initial frame and counted invisible frames as failures.
Native SGLA's candidate layer is not inferred from a fabricated action index;
its summary labels layer usage as `native`.

Per-video JSONs preserve predicted boxes, action indices, IoUs, visibility,
candidate counts and timing. They are fingerprinted and bound to the final
checkpoint. An interrupted evaluation resumes completed videos after checking
their source-label/reward consistency. No partial-seed efficacy report is
produced.

## Analysis and Decision

The two required credit contrasts are H20 minus ordinary PPO and H20 minus
H1. First average the three paired seed differences within each video, then
bootstrap all videos in the frozen tune split (5,000 samples; analysis seed 2077). The primary
credit criterion requires both video-bootstrap 95% lower bounds above zero
and all three seed mean differences positive. Requiring both contrasts is
an intersection claim, not selecting the better of two tests.

The report also shows each seed's mean delta, seed standard deviation, and
a crossed paired seed/video bootstrap interval. Repeating N tune videos with
three seeds does not create 3N independent videos; seed-population inference
remains limited.

- `NO_CREDIT_EFFECT_ESTABLISHED`: the predeclared credit criterion failed.
- `CREDIT_EFFECT_ONLY_NOT_COMPETITIVE`: credit criterion passed, but the
  strategy did not clear Fast, memory-Fast and all six static-layer controls.
- `DEVELOPMENT_POLICY_GAIN_SUPPORTED`: both criteria passed. The six static
  comparisons use conservative Bonferroni-adjusted lower confidence bounds.

SGLA is always reported separately, including the accuracy gap. A successful
credit test does not mean SGLA was beaten. None of these development statuses
establishes external generalization, publication novelty, or faster inference.
The tune set has already been inspected in earlier work. A successful new
method would still require a new, untouched external benchmark.

## Server Commands

Sync only the four new files:

- `tracking/stage23_development.py`
- `tracking/stage23_development_summary.py`
- `tracking/test_stage23_development.py`
- this document

Keep all passed smoke and frozen Stage-18 files unchanged.

```bash
export STAGE18_ROOT="$PWD/output/stage18/temporal_layer_policy_v1"
export STAGE18_REP_BUNDLE="$STAGE18_ROOT/frozen_representation_v1.json"
export INTERVENTION_MANIFEST="$PWD/output/stage8/l9_intervention_v1/intervention_manifest.json"
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
export STAGE23_ROOT="$PWD/output/stage23/onpolicy_branch_credit_v1"
export STAGE23B_DIR="$STAGE23_ROOT/development_v1"

python tracking/test_stage23_branch_credit.py
python tracking/test_stage23_development.py

for SEED in 2074 2075 2076; do
  python tracking/stage23_development.py \
    --bundle "${STAGE18_REP_BUNDLE:?}" \
    --manifest "${INTERVENTION_MANIFEST:?}" \
    --checkpoint "${CHECKPOINT:?}" \
    --smoke_dir "$STAGE23_ROOT/smoke_v1" \
    --parameter deit_distilled --gpu 0 --seed "$SEED" \
    --output_dir "$STAGE23B_DIR" || break
done
```

The PyTorch tests must run, not skip, on the server. They include real CPU
torch optimizer parity, evaluation head counts and an interrupted/resumed
training comparison. The local NumPy-only test result cannot substitute for
the server's torch/CUDA checks.

For the first invocation it is reasonable to run only seed 2074 with the
same command, inspect execution and wall time, and then run seeds 2075/2076.
That is a scheduling decision, not permission to analyze only the best seed.
All seeds/arms belong to the same fixed root contract. Do not run simultaneous
trainers on this single GPU or the same output root.

## Progress, Interruptions and Results

```bash
python tracking/stage23_development_summary.py \
  --input "$STAGE23B_DIR" --status

# Only after all nine final policies and their frozen-tune evaluations complete:
python tracking/stage23_development_summary.py --input "$STAGE23B_DIR"
cat "$STAGE23B_DIR/analysis/stage23b_decision.json"
```

Atomic `last.pt` is saved after each rollout/update; repeat exactly the same
seed command after interruption. No smoke weights are resumed. A normal exit
or Ctrl+C removes the writer lock. After a hard kill, confirm no writer exists
before removing only the development directory's `.stage23_writer.lock`.
Progress CSVs can lag one checkpoint; the resumed checkpoint is authoritative.

Training loop time excludes checkpoint publication and final evaluation.
Plan runtime from the measured smoke wall times, then allow extra overhead
and video difficulty differences. Do not silently extend beyond 32,768 steps
because a promising seed appears late, or alter H/gamma/weights on these tune
outcomes. A failure means this fixed experiment did not establish the claim,
not that all possible PPO tracking methods are impossible.
