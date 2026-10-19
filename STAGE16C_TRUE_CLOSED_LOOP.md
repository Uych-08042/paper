# Stage-16C: Frozen TAVRTrack True Closed Loop

Stage-16B passed the offline-Q feasibility screen.  It did **not** establish a
tracking gain: the P30 policy ranked supported candidate actions better than a
matched random candidate, but its support-state estimate was still below Fast.
Stage-16C is therefore the required true closed-loop decision experiment.

## What "fingerprint freeze" means

A fingerprint is a SHA-256 digest of a canonical experiment contract.  The
Stage-16C policy bundle also stores the SHA-256 and byte size of every model and
preprocessing artifact.  Freezing does not copy or retrain anything.  It binds:

- the passed Stage-16B decision and experiment contract;
- all three recurrent-CQL checkpoints;
- train-only token PCA and scalar normalization;
- the **training-calibrated** P30 advantage threshold;
- the Stage-16 behavior-cache and manifest lineage;
- the action set, model dimensions, budget rule, and relevant source code.

Any changed checkpoint, threshold, preprocessing file, source contract, or
manifest causes validation to stop.  This is how the experiment prevents a
post-validation threshold or budget change from silently becoming "the same"
policy.

## 1. Freeze the passed policy

```bash
export STAGE16C_BUNDLE="$STAGE16B_DIR/stage16c_policy_bundle.json"
export STAGE16C_ROOT="$PWD/output/stage16/tavrtrack_behavior_v1/closed_loop_p30_v1"
export STAGE16C_SANITY="$STAGE16C_ROOT/sanity_train"
export STAGE16C_VAL="$STAGE16C_ROOT/val"

python tracking/stage16c_freeze_policy.py \
  --stage16b_dir "$STAGE16B_DIR" \
  --freeze_manifest "$STAGE16_FREEZE" \
  --output "$STAGE16C_BUNDLE"

python tracking/stage16c_freeze_policy.py \
  --verify_bundle "$STAGE16C_BUNDLE"
```

The second command verifies every frozen artifact.  `--output ignored` is not
used in verification mode.

## 2. Training-clip runtime sanity check

Use a training clip only.  Do not inspect a validation trajectory before the
full run.

```bash
export FIRST_STAGE16_TRAIN=$(python -c \
  'import json,sys; d=json.load(open(sys.argv[1])); print(next(x["sequence"] for x in d["sequences"] if x["split"]=="train"))' \
  "$INTERVENTION_MANIFEST")

python tracking/stage16c_closed_loop_validate.py \
  --manifest "$INTERVENTION_MANIFEST" \
  --policy_bundle "$STAGE16C_BUNDLE" \
  --parameter deit_distilled \
  --checkpoint "$CHECKPOINT" \
  --gpu 0 \
  --split train \
  --sequence "$FIRST_STAGE16_TRAIN" \
  --output_dir "$STAGE16C_SANITY" \
  --latency_warmup_frames 20 \
  --verify_actions

python tracking/stage16c_closed_loop_summary.py \
  --input "$STAGE16C_SANITY" \
  --allow_incomplete \
  --bootstrap_samples 1000 \
  --seed 2050
```

The sanity decision must be `PILOT_ONLY`.  It checks execution, recurrent-state
continuity, exact-cost control, and output contracts; it is not an efficacy
result.

## 3. One full validation run

The four policies are:

- `fast`;
- `uniform_l9_p30`, an evenly spaced L9 control that spends the full P30 cap;
- `sgla`, the original selector and full-compute reference;
- `tavrtrack_p30`, the frozen recurrent ensemble with a per-clip P30 hard cap.

```bash
for SHARD in 0 1 2 3; do
  python tracking/stage16c_closed_loop_validate.py \
    --manifest "$INTERVENTION_MANIFEST" \
    --policy_bundle "$STAGE16C_BUNDLE" \
    --parameter deit_distilled \
    --checkpoint "$CHECKPOINT" \
    --gpu 0 \
    --split val \
    --num_shards 4 \
    --shard_id "$SHARD" \
    --output_dir "$STAGE16C_VAL" \
    --latency_warmup_frames 20 \
    --verify_actions
done

python tracking/stage16c_closed_loop_summary.py \
  --input "$STAGE16C_VAL" \
  --bootstrap_samples 5000 \
  --seed 2050
```

## Frozen decision rule

`PASS_CLOSED_LOOP_DEVELOPMENT` requires all of the following:

1. all 40 validation clips and all four policies are complete;
2. TAVRTrack never exceeds the P30 per-clip hard cap;
3. at least three candidate layers are active in the true rollout;
4. paired `TAVRTrack-P30 - Fast` 95% CI lower bound is above zero;
5. paired `TAVRTrack-P30 - Uniform-L9-P30` 95% CI lower bound is above zero.

The second efficacy contrast is deliberately conservative: Uniform-L9 spends
the full P30 cap, while TAVRTrack may spend less but can never spend more.
Without this contrast, a positive result would show only that extra computation
helps, not that the learned layer scheduler is valuable.  A pass authorizes one
untouched external evaluation.  A failure
stops this policy version; the P30 threshold and budget must not be tuned on
these 40 validation videos.  OTB and UAV123 remain unavailable for policy
selection.
