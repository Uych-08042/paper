# Stage-16B Recurrent Conservative Offline Q Screen

## What "freezing the fingerprint" means

The Stage-16 collection fingerprint in `run.json` identifies the manifest,
runtime, behavior schedules, state schema, and collection code. It does not by
itself hash every large NPZ file. The freeze step adds a second immutable
contract that records:

- the accepted `PASS_BEHAVIOR_DATA` decision and its SHA-256;
- the Stage-16 `run.json` and run fingerprint;
- the relative path, byte size, and SHA-256 of every one of the 160 caches;
- one fingerprint over the complete inventory.

Stage-16B refuses to train if any source file is missing, added, replaced,
truncated, or changed. `--make_read_only` is an additional operational guard;
the cryptographic verification remains the authoritative check. A new data
design must use a new cache and versioned output directory rather than editing
this frozen cache.

## 1. Freeze And Verify Stage-16A

```bash
cd ~/workspace/paper
conda activate sglatrack-test

export STAGE16_ROOT="$PWD/output/stage16/tavrtrack_behavior_v1"
export STAGE16_CACHE="$STAGE16_ROOT/cache"
export STAGE16_AUDIT="$STAGE16_ROOT/audit"
export STAGE16_FREEZE="$STAGE16_ROOT/stage16_frozen_cache.json"

python tracking/stage16_freeze_cache.py \
  --input "$STAGE16_CACHE" \
  --audit_decision "$STAGE16_AUDIT/stage16_behavior_decision.json" \
  --output "$STAGE16_FREEZE" \
  --make_read_only

python tracking/stage16_freeze_cache.py \
  --input "$STAGE16_CACHE" \
  --verify_manifest "$STAGE16_FREEZE"
```

The second command must print the same frozen-cache fingerprint. Hashing the
full cache is intentionally performed before training.

## 2. Frozen Stage-16B Experiment

Stage-16B is one predeclared experiment, not a hyperparameter search:

- train/tune split: 96/24 videos drawn only from the 120-video train split;
- final validation: the existing 40-video val split, evaluated once;
- state encoder: train-only 24-dimensional token PCA, spatial CNN, score-map
  CNN, causal scalar state, and a GRU;
- action values: `Fast, L7, L8, L9, L10, L11, L12`;
- learning: recurrent Double Q targets, discrete CQL penalty, behavior-cloning
  regularizer, and a slowly updated target network;
- reward: per-frame success score, matching the tracking success-AUC metric;
- ensemble seeds: 2047, 2048, and 2049;
- primary support-state diagnostic: 30% candidate budget.

```bash
export STAGE16B_DIR="$STAGE16_ROOT/recurrent_cql_v1"

python tracking/stage16_recurrent_cql.py \
  --input "$STAGE16_CACHE" \
  --freeze_manifest "$STAGE16_FREEZE" \
  --output_dir "$STAGE16B_DIR" \
  --device cuda \
  --seeds 2047 2048 2049 \
  --split_seed 2047 \
  --tune_fraction 0.2 \
  --token_pca_dims 24 \
  --token_pca_samples 100000 \
  --hidden_dim 128 \
  --spatial_channels 32 \
  --batch_size 4 \
  --epochs 40 \
  --patience 8 \
  --learning_rate 0.0003 \
  --weight_decay 0.0001 \
  --gamma 0.95 \
  --cql_alpha 0.1 \
  --cql_temperature 1.0 \
  --bc_weight 0.05 \
  --policy_temperature 1.0 \
  --target_tau 0.01 \
  --grad_clip 5 \
  --budgets 10 30 50 \
  --primary_budget 30 \
  --bootstrap_samples 5000
```

The preprocessing directory is resumable and is derived from the frozen
cache. The experiment contract prevents rerunning the same output directory
with changed hyperparameters. Once `stage16b_decision.json` exists, the script
reports the existing result instead of evaluating validation again.

## 3. Decision

```bash
cat "$STAGE16B_DIR/stage16b_value_summary.csv"
cat "$STAGE16B_DIR/stage16b_budget_summary.csv"
cat "$STAGE16B_DIR/stage16b_decision.json"
```

`PASS_OFFLINE_Q_SCREEN` requires all of the following on the untouched val
split:

- behavior-action discounted-return Pearson correlation at least 0.10;
- within-trajectory centered return correlation at least 0.02;
- the 30% policy's immediate support-state lift over a matched random
  candidate has a sequence-bootstrap 95% lower bound above zero;
- at least three candidate layers remain active.

The budget table evaluates actions only on held-out states represented in the
frozen behavior data. It is not a closed-loop AUC result. A pass permits one
true Stage-16C closed-loop development run. A failure stops this policy version
without tuning against the 40 validation videos. OTB and UAV123 remain excluded
from policy development.
