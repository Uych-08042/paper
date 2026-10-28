# Stage-20: fixed dense student and rollout-state correction

Stage-19 did not establish a transferable closed-loop pruning criterion. Keep
that result and its cache unchanged. This experiment fixes the student first
and tests training interventions; it does not search masks using those outcomes.

## Frozen Design

- Physically halve the FFN hidden width in all 12 backbone blocks, retaining
  channels by checkpoint-only input/output magnitude product. For DeiT-Tiny
  this is 768 -> 384. Keep embedding width, token grids, prediction head, and
  the original selector rule. Freeze selector weights, not its input features.
- Use only the source manifest's 120 training videos: seeded 96 fit / 24
  development-check. The original 40 validation videos, UAV123, and OTB are
  excluded. These are reused development videos, not a new holdout benchmark.
- Four arms form a 2x2 control: `supervised_reference`, `kd_reference`,
  `supervised_student`, `kd_student`. Reference means the original teacher's
  predicted crop history; student means the current student's own history.
- Match initial student weights, frame order, frame exposures, optimizer,
  updates, epochs, and teacher-query count for all four arms within each seed.
  No GT boxes reset a training rollout after initialization. Crops and decoded
  history are detached, so this is rollout-state training, not BPTT or RL.
- GT loss is the existing Stage-18 tracking loss in actual crop coordinates.
  Both KD arms add the same spatial-distribution KL + 5 * box L1, weight 1.
  Teacher targets are detached and use the exact student's input tensor for
  that step. Apply KD only with visible GT, an in-crop GT center, and a decoded
  teacher box having continuous image-coordinate IoU >= 0.5 with GT. This
  training-only filter is identical across KD arms; report accepted counts.
- Train all non-selector parameters with AdamW, LR 1e-5, weight decay .01,
  four-frame accumulation, clip norm 1. Keep BN/dropout in eval mode in every
  arm. Full default: five fixed epochs, <=128 tracked frames/clip, all seeds
  2059/2060/2061. No epoch or seed selection on check-set AUC.
- Evaluate teacher, untrained compact student, and four trained students
  autonomously on the same clips. The teacher is absent from student inference.
  Use the same forward/crop/decode implementation, interleave methods, preload
  images outside timing, warm up every architecture, and discard five initial
  timing frames. Report dense parameter counts and measured latency separately.

Primary contrasts are KD-self minus KD-reference and KD-self minus supervised-self.
Also require the interaction `(KDself - Sself) - (KDref - Sref)` to be positive:
this distinguishes correction benefit from just exposing a student to its own
crops. Bootstrap videos after averaging all declared seeds, not seed-video pairs.
Per-seed results are retained. Full development feasibility additionally requires
teacher noninferiority (paired CI lower bound > -0.5 AUC points), >=20% fewer
parameters, and >=10% lower measured latency. These are prospective engineering
targets, not guaranteed outcomes. Smoke results do not test these claims.

## First Run: Smoke Only

Run in the server repository using the existing `sglatrack-test` environment.
No new dataset or intervention-cache collection is needed.

```bash
export INTERVENTION_MANIFEST="$PWD/output/stage8/l9_intervention_v1/intervention_manifest.json"
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
export STAGE20_ROOT="$PWD/output/stage20/fixed_student_correction_v1"
python tracking/test_stage20_compression.py
python tracking/stage20_student_train.py \
  --source_manifest "$INTERVENTION_MANIFEST" \
  --checkpoint "$CHECKPOINT" --parameter deit_distilled --gpu 0 \
  --mode smoke --seeds 2059 \
  --output_dir "$STAGE20_ROOT/smoke"
python tracking/stage20_compression_summary.py --input "$STAGE20_ROOT/smoke"
```

Smoke uses two fit and two check videos, 16 tracked frames each, one epoch,
one seed. Its videos are subsets of the full fit/check split. Inspect
`analysis/stage20_decision.json`, `stage20_contract.json`, all four `history.csv`
files and the six evaluation traces. PASS_MATCHED_TRAINING_SMOKE establishes
execution integrity only, not compression quality or a speed improvement.

## Full Development Run (After Smoke Review)

```bash
python tracking/stage20_student_train.py \
  --source_manifest "$INTERVENTION_MANIFEST" \
  --checkpoint "$CHECKPOINT" --parameter deit_distilled --gpu 0 \
  --mode full --epochs 5 --frames 128 --seeds 2059 2060 2061 \
  --output_dir "$STAGE20_ROOT/full"
python tracking/stage20_compression_summary.py --input "$STAGE20_ROOT/full"
```

Checkpoints are atomic `checkpoints/seed*/ARM/last.pt`, saved at epoch boundaries.
Ctrl-C and rerun the exact command to resume; an incomplete epoch is replayed.
Finished arms and intact evaluation traces are reused. Settings, code, source
images, annotations and initialization are fingerprint-bound. A changed contract
requires a new directory. Do not run two writers in the same directory. A hard
kill can leave `stage20_training.lock`: remove only that lock after confirming
the old process has stopped; do not delete checkpoints or recreate the contract.

Partial progress is in each arm's `history.csv`; the final summary deliberately
requires all declared seeds/arms. No check AUC is used to choose a checkpoint.
Inference checkpoints include compact model state; the contract stores exact
channel indices needed to reconstruct the dense compact architecture.

## Interpretation Boundary

Student-rollout distillation is not itself proof of a new method, and a positive
development comparison is not external validation. This experiment asks whether
state-conditioned teacher correction adds measurable value at fixed student
architecture. A failure must be kept as such: do not scan pruning ratios, KD
weights, epochs or seeds on this check split to rescue the claim. A positive
result would justify designing a specific compression-training contribution
and then evaluating one frozen method on an untouched external benchmark.
