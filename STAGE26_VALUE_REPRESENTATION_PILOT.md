# Stage-26: Intervention-Value Representation A/B Pilot

## Scope

Stage-25 failed its frozen relational-state screen despite cross-repeat
hindsight headroom. Keep that decision and all old artifacts unchanged.
This experiment tests a NEW gradient intervention: does supervision of
future differences between candidate actions help when it reaches the
visual features that actually drive tracking, rather than only a downstream
observation encoder/value head?

This is not PPO, a new benchmark result, proof of novelty, or a speed test.
The first command runs a three-video engineering smoke. A passed smoke
permits the predeclared 32/16 A/B development pilot, not unrestricted training.

## Matched Intervention

Both arms initialize from the same original checkpoint, frozen Stage-18
tracking-memory/adapters bundle, and seed-specific new six-output value head.
Both update ONLY shared backbone L5-L6 and the same value head. All original
candidate L7-L12 blocks, tracking head, other backbone layers, and Stage-18
memory/adapters stay unchanged and in evaluation mode.

| Arm | Localization gradient to L5-L6 | Value-difference gradient to L5-L6 |
| --- | --- | --- |
| `stop_gradient` (A) | Yes | No; value head/observation encoder still trains |
| `value_aligned` (B) | Yes | Yes |

The candidate value head observes the current shared-prefix grid, template,
frozen predicted-history memory, prior confidence/action/motion and credit.
It never observes GT, future images, or any unexecuted candidate outputs.
Its convolutional/template encoder and 128-wide hidden layer have identical
parameter shapes/initialization in the paired arms. There are no new manual
relation features, layer-diversity targets, or assigned expert specialties.

Target supervision is the difference between six **H2-H20** returns, averaged
over two matched continuations, gamma 0.97, normalized by visible discounted
weights. The action frame is excluded. Pairwise Smooth-L1 compares all 15
candidate pairs after dividing target differences by the frozen scale 0.02.
GT returns are detached labels, never a differentiable route to future GT.
Total loss is the existing `2*GIoU + 5*L1 + center_focal` plus rank loss at
weight 1. No original selector-similarity loss is added.

P30 call times are canonical and fixed. When credit permits a call, deployment
chooses argmax over the six candidate utilities; otherwise it executes Fast.
Each frame computes one prefix, a small value head, at most one independent
candidate block and one tracking head. This is not sequential depth selection.
Static L7-L12 controls use these exact same call positions. No additional
candidate is executed at inference to decide which candidate to execute.

## Fresh Labels at Every Update

Updating L5-L6 changes both features and tracking dynamics. Stage-25 NPZ labels
are NOT reused as current-model labels. Its frozen failure is provenance only.

For each predeclared training anchor:

1. Start a new training episode at the clip's original initialization and
   replay the current model's own predicted crops/memory to that anchor.
   These episode boundaries are specified in advance, not failure-triggered
   GT corrections. Evaluation starts from GT once and never resets within a clip.
2. Keep the model weights fixed while branching over all six first layers.
   Each spends one candidate call. Both continuations have identical P30 call
   masks; repeat zero replays behavior and repeat one redraws later layers.
3. Check that the behavior's first layer plus repeat-zero continuation replays
   its entire subsequent H20 boxes, not only the first box. Restore actual state
   and Python/NumPy/torch/CUDA sampling state after branch simulation.
4. Recompute the ANCHOR image/crop and prefix with autograd enabled. Frozen
   candidate/head parameters must still transmit input gradients. Verify this
   differentiable replay against the actual pre-update features/output.
5. Compute localization and value losses, audit both separate prefix-gradient
   norms, then perform ONE AdamW step. No more updates reuse that target.
6. Save an atomic checkpoint containing this cursor, both optimizers' parameter
   groups, weights, history and RNG states. The next anchor replays its prefix
   afresh using the newly updated weights and recollects every return.

Memory weights are frozen, and prefix-history memory is a detached condition
for the anchor gradient. This is one-frame representation backpropagation with
future-outcome supervision, NOT differentiable H20 crop/trajectory BPTT.

Every branch file is bound to its exact pre-update prefix/value-head hashes.
Every history row links before/after model hashes and its branch-file hash.
Analysis reconstructs targets, continuations, equal costs and those version
chains without unpickling checkpoints. It cannot silently train a changed
model on old returns and call them fresh returns.

## Fixed Small Budget

- Choose 32 training and 16 checking videos from the original 95 fit IDs using
  split seed 2086. Keep the remaining 47 unused in this run. Source order/IDs
  are verified before selection; no outcome-based or short-clip resampling.
- These videos were previously consumed development data. The frozen warm-up
  already saw them. The 16 checking videos are withheld from NEW parameter
  updates, not an untouched confirmation or external validation set.
- Original tune/validation images and labels, OTB and UAV123 are not opened.
- Smoke: first three new training IDs, 64 tracked frames, one pass, seed 2087.
  Four anchors/video/arm, 12 attempts and 2,880 branch frames per arm.
- Full pilot: maximum 128 tracked frames; two complete passes; four anchors
  per video; paired seeds 2087/2088/2089. That is 256 attempts and 61,440 branch
  frames per arm/seed, or 368,640 training branch frames across six runs.
- Prefix replays, live H20 verification, differentiable anchor replays, final
  checking branches and all deployed controls are additional work. Time the
  smoke rather than estimating runtime only from branch counts.
- Each anchor needs a valid first label and >=80% H20 visibility. Other frames
  still advance tracking. Ineligible anchors consume their branch quota but
  skip optimization in BOTH arms. Complete pilot analysis requires >=80%
  eligible attempts and >=80% checking anchors, with coverage in every video.
- AdamW: L5-L6 lr 1e-5, value head lr 3e-4, weight decay 0.01, clip norm 1.
  No AMP. One update per fresh eligible anchor, no replay-buffer epochs.
- Final checkpoint only, never best step/seed/checking AUC. No extra passes
  after inspecting the checking results. There is no diversity requirement.

## Gates and Reports

`PASS_VALUE_REPRESENTATION_SMOKE` requires matched initial weights and valid
update/branch quotas; zero value gradient to A's prefix, nonzero to B's;
nonzero localization gradients in both arms; changed prefix/value weights;
full replay error <=1e-6; and unchanged frozen runtime/memory parameters.
No checking labels or learned-policy efficacy are reported by smoke.

After all three seeds and both arms complete, inspect the 16 development
checking videos. Report both kinds of outcomes, without treating them as
interchangeable:

1. **Support-state H2-H20 values.** Collect NEW final-model branches. Compare
   greedy selected value with all six constant choices on the SAME states.
   Define routing excess as selected minus uniform-layer expected value.
   Require B-minus-A routing-excess video-CI lower bound >0 and all three seed
   mean differences positive. Also require B-minus-every-fixed-layer adjusted
   lower bound >0 and mean >=0.001 normalized discounted IoU.
2. **True closed-loop AUC.** Execute each arm's greedy router, memory-Fast,
   all six fixed layers and random-layer control using that arm's OWN final
   tracking weights. Each policy follows its own boxes/crops/memory. The
   closed-loop guard requires B-greedy minus A-greedy lower95 >0 and B-greedy
   minus every B-fixed-layer adjusted lower95 >0. Original Fast/SGLA are
   separately reported with original unmodified weights; they are not silently
   evaluated using the new prefix. Their accuracy gaps remain visible.

Average paired seeds within video before bootstrapping 16 videos, 5,000
draws, seed 2090. Use Bonferroni lower bounds for each six-static family.
Repeated seeds do not create 48 independent videos. AUC is strict 21-threshold
success AUC on valid tracked frames, excluding the initial GT frame.

- `FAIL_VALUE_REPRESENTATION_PILOT`: support-state mechanism gate failed.
- `MECHANISM_ONLY_NO_CLOSED_LOOP_GAIN`: support gate passed, deployment guard failed.
- `PASS_VALUE_REPRESENTATION_PILOT`: both passed; permits designing one matched
  PPO-necessity smoke. It does NOT establish PPO usefulness, SGLA superiority,
  deployment speed, external generalization or publication novelty.

The 0.001 minimum is an engineering investment threshold, not a theorem.
An inconclusive small pilot stops this specific candidate at its fixed budget;
it does not falsify all joint-learning methods. Do not lower the gates after
seeing results or reopen excluded benchmarks for policy selection.

## Server: Start With Smoke

Sync ONLY these new files, keeping frozen dependencies unchanged:

```text
tracking/stage26_value_common.py
tracking/stage26_value_runtime.py
tracking/stage26_value_train.py
tracking/stage26_value_summary.py
tracking/test_stage26_value.py
STAGE26_VALUE_REPRESENTATION_PILOT.md
```

```bash
cd ~/workspace/paper
export STAGE18_REP_BUNDLE="$PWD/output/stage18/temporal_layer_policy_v1/frozen_representation_v1.json"
export INTERVENTION_MANIFEST="$PWD/output/stage8/l9_intervention_v1/intervention_manifest.json"
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
export STAGE25_ROOT="$PWD/output/stage25/relational_state_probe_v1"
export STAGE26_ROOT="$PWD/output/stage26/intervention_value_representation_v1"

python tracking/test_stage26_value.py

python tracking/stage26_value_train.py \
  --bundle "${STAGE18_REP_BUNDLE:?}" \
  --manifest "${INTERVENTION_MANIFEST:?}" \
  --stage25_root "${STAGE25_ROOT:?}" \
  --checkpoint "${CHECKPOINT:?}" \
  --parameter deit_distilled --gpu 0 \
  --mode smoke --seed 2087 \
  --output_dir "$STAGE26_ROOT/smoke_v1"

cat "$STAGE26_ROOT/smoke_v1/stage26_smoke_decision.json"
```

All real-PyTorch tests must run on the server, not skip. A gradient/runtime
failure means repair implementation before interpreting any quality result.
Do not edit a passed contract or bypass a source hash to continue.
If a repair changes source after a run has started, keep the failed run and
use a new smoke directory such as `smoke_v2`; do not overwrite its freeze.

## After Smoke Passes

The full pilot starts FRESH from the matched original initialization, not
from smoke's trained weights. Run one writer/job at a time:

```bash
for SEED in 2087 2088 2089; do
  python tracking/stage26_value_train.py \
    --bundle "${STAGE18_REP_BUNDLE:?}" \
    --manifest "${INTERVENTION_MANIFEST:?}" \
    --stage25_root "${STAGE25_ROOT:?}" \
    --checkpoint "${CHECKPOINT:?}" \
    --parameter deit_distilled --gpu 0 \
    --mode full --seed "$SEED" \
    --smoke_dir "$STAGE26_ROOT/smoke_v1" \
    --output_dir "$STAGE26_ROOT/pilot_v1" || break
done

python tracking/stage26_value_summary.py --input "$STAGE26_ROOT/pilot_v1" --status

# Only after all six arms and checking evaluations complete:
python tracking/stage26_value_summary.py --input "$STAGE26_ROOT/pilot_v1"
cat "$STAGE26_ROOT/pilot_v1/analysis/stage26_decision.json"
cat "$STAGE26_ROOT/pilot_v1/analysis/stage26_branch_paired.csv"
cat "$STAGE26_ROOT/pilot_v1/analysis/stage26_closed_loop_paired.csv"
```

Interruptions resume from the latest complete optimizer attempt by repeating
the SAME command and output directory. An unfinished attempt is recollected
at that checkpoint's model version. CSV can lag one atomic checkpoint and
is reconciled on resume. Completed checking files are reused after identity
checks; final analysis validates hashes, geometry, labels and call masks.
The inherited writer helper uses `.stage23_writer.lock` inside the NEW root.
After a hard kill, verify no writer remains before removing only that lock.
Never remove old experiment data or alter JSON to make a gate pass.
