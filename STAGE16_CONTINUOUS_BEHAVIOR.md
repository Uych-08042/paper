# Stage-16A Continuous Multi-Action Behavior Data

## Purpose

Stage-14 and Stage-15 reject a one-step layer selector: hindsight action value
exists, but the best layer is transient and cannot be predicted reliably from
the old per-frame state. Stage-16 starts a different method family,
provisionally called TAVRTrack (Temporal Action-Value Routing Tracker).

TAVRTrack treats routing as a recurrent, budget-constrained sequential control
problem. SGLA supplies the frozen L1-L6 representation and the six independent
L7-L12 candidate actions; its collapsed MLP selector is not used as the new
policy.

Stage-16A does not train a policy. It creates and audits the continuous
transition data needed for one frozen conservative offline-RL screen.

## Frozen Collection Design

- Development source: the existing 120/40 Stage-8 train/val manifest only.
- Actions: `Fast, L7, L8, L9, L10, L11, L12` at every tracked frame.
- Every executed box becomes the state used to crop the next frame.
- Behavior candidate rates: `0, 10, 30, 50, 100%`.
- The zero-rate Fast trajectory is collected once; every nonzero rate has two
  independent deterministic randomization repeats, for nine episodes per clip.
- Candidate positions are sampled without replacement at exact cost. Candidate
  layers are balanced to differ by at most one occurrence per episode.
- State representation: 8x8 spatial L6 search tokens, template mean/max tokens,
  Fast score map, native-selector diagnostic response, causal box/action
  history, and remaining budget.
- Ground truth is read only after the action table and behavior action are
  fixed. It is stored only in reward/label fields.
- The cache stores the exact conditional behavior-action probabilities for a
  later conservative offline policy evaluation.

The 8x8 token grid preserves spatial structure while keeping the full run near
a practical storage scale. Do not enable `--save_full_tokens` for the primary
run; doing so defines a different state representation and run fingerprint.

## Environment

```bash
cd ~/workspace/paper
conda activate sglatrack-test

export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
export INTERVENTION_MANIFEST="$PWD/output/stage8/l9_intervention_v1/intervention_manifest.json"
export STAGE16_ROOT="$PWD/output/stage16/tavrtrack_behavior_v1"
export STAGE16_CACHE="$STAGE16_ROOT/cache"
export STAGE16_AUDIT="$STAGE16_ROOT/audit"

test -f "$CHECKPOINT"
test -f "$INTERVENTION_MANIFEST"

export FIRST_STAGE16_SEQ=$(python -c \
  'import json,sys; d=json.load(open(sys.argv[1])); print(d["sequences"][0]["sequence"])' \
  "$INTERVENTION_MANIFEST")
```

## 1. One-Clip Integrity Run

Use a new empty directory. This checks all seven rerun paths, deterministic
behavior schedules, next-state feedback, and label reconstruction.

```bash
python tracking/stage16_behavior_collect.py \
  --manifest "$INTERVENTION_MANIFEST" \
  --parameter deit_distilled \
  --checkpoint "$CHECKPOINT" \
  --gpu 0 \
  --sequence "$FIRST_STAGE16_SEQ" \
  --state_grid 8 \
  --behavior_rates 0 10 30 50 100 \
  --behavior_repeats 2 \
  --seed 2046 \
  --verify_actions \
  --output_dir "$STAGE16_ROOT/sanity_cache"

python tracking/stage16_behavior_audit.py \
  --input "$STAGE16_ROOT/sanity_cache" \
  --output_dir "$STAGE16_ROOT/sanity_audit" \
  --allow_incomplete
```

Expected status: `PILOT_PASS`. A one-clip result is an integrity check only.

## 2. Eight-Clip Runtime and Storage Pilot

The main cache is resumable, so these files can be retained for the full run.

```bash
python tracking/stage16_behavior_collect.py \
  --manifest "$INTERVENTION_MANIFEST" \
  --parameter deit_distilled \
  --checkpoint "$CHECKPOINT" \
  --gpu 0 \
  --split train \
  --max_sequences 8 \
  --state_grid 8 \
  --behavior_rates 0 10 30 50 100 \
  --behavior_repeats 2 \
  --seed 2046 \
  --verify_actions \
  --output_dir "$STAGE16_CACHE"

python tracking/stage16_behavior_audit.py \
  --input "$STAGE16_CACHE" \
  --output_dir "$STAGE16_ROOT/pilot_audit" \
  --allow_incomplete

find "$STAGE16_CACHE" -name '*__stage16_behavior.npz' | wc -l
du -sh "$STAGE16_CACHE"
```

Proceed only if the status is `PILOT_PASS`, no audit exception occurs, and the
measured storage/runtime is acceptable. Do not interpret pilot AUC differences.

## 3. Full Frozen Collection

The loop below is sequential on one GPU. Independent GPUs may run different
shards concurrently against the same cache because `run.json` publication is
locked and each shard owns disjoint files.

```bash
for SHARD in 0 1 2 3; do
  python tracking/stage16_behavior_collect.py \
    --manifest "$INTERVENTION_MANIFEST" \
    --parameter deit_distilled \
    --checkpoint "$CHECKPOINT" \
    --gpu 0 \
    --state_grid 8 \
    --behavior_rates 0 10 30 50 100 \
    --behavior_repeats 2 \
    --seed 2046 \
    --verify_actions \
    --num_shards 4 \
    --shard_id "$SHARD" \
    --output_dir "$STAGE16_CACHE"
done

find "$STAGE16_CACHE" -name '*__stage16_behavior.npz' | wc -l
```

Expected file count: `160`.

## 4. Pre-Training Audit

```bash
python tracking/stage16_behavior_audit.py \
  --input "$STAGE16_CACHE" \
  --output_dir "$STAGE16_AUDIT"

cat "$STAGE16_AUDIT/stage16_behavior_decision.json"
cat "$STAGE16_AUDIT/stage16_behavior_action_coverage.csv"
```

`PASS_BEHAVIOR_DATA` requires all 160 clips, both frozen splits, every action
with at least 100 transitions and 2% coverage per split, at least 80% valid
labels, and candidate actions changing the next-state box on at least half of
their executions in every clip.

Only `PASS_BEHAVIOR_DATA` permits Stage-16B. The next stage is one predeclared
recurrent conservative offline-Q experiment trained on the 120-video split and
evaluated once on the frozen 40-video split. OTB and UAV123 are not policy
development datasets and must not be used for Stage-16 selection.
