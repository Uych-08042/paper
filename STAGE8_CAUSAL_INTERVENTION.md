# Stage-8B: Single-L9 Causal Intervention

## Question

Stage-8A showed that sparse L9 inference improves Fast, but did not establish
whether the improvement comes from the timing rule itself. Stage-8B asks a
narrower causal question:

> Starting from exactly the same Fast tracker state, what is the immediate and
> future effect of replacing one Fast decision with one fixed-L9 decision?

Each clip follows one uninterrupted Fast baseline. Twelve deterministic,
outcome-independent anchors are spread across the usable interval. At every
anchor, the treatment branch executes L9 once and then returns to Fast. The
control is the corresponding slice of the uninterrupted Fast trajectory.

The predeclared confirmatory result is the validation-split cumulative AUC
effect at horizon 20. Horizon 1 measures the direct effect. Horizons 5, 10 and
40 and all temporal-bin analyses are secondary. The post-anchor metric removes
the intervention frame and tests whether the effect propagates into later Fast
predictions.

## 1. Environment

Run from the repository root on the CUDA server:

```bash
cd /home/u25600003160604/workspace/paper

export GOT_ROOT=/home/u25600003160604/workspace/datasets/GOT10K_subset/train
export INSTALLED_SEQUENCES="$PWD/output/stage4/got10k_installed_train.txt"
export PILOT_MANIFEST="$PWD/output/stage4/pilot_manifest.json"
export HOLDOUT50="$PWD/output/stage7/temporal_holdout_v1/holdout_manifest.json"
export HOLDOUT150="$PWD/output/stage7/temporal_holdout_confirm_v1/holdout_manifest.json"
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"

export STAGE8B_ROOT="$PWD/output/stage8/l9_intervention_v1"
export INTERVENTION_MANIFEST="$STAGE8B_ROOT/intervention_manifest.json"
export INTERVENTION_CACHE="$STAGE8B_ROOT/cache"
```

Confirm that all inputs exist:

```bash
for PATH_TO_CHECK in \
  "$INSTALLED_SEQUENCES" \
  "$PILOT_MANIFEST" \
  "$HOLDOUT50" \
  "$HOLDOUT150" \
  "$CHECKPOINT"; do
  test -f "$PATH_TO_CHECK" || echo "MISSING: $PATH_TO_CHECK"
done
```

## 2. Freeze a fresh development manifest

The manifest excludes every video used by the Stage-4 pilot and both Stage-7
cohorts. It creates disjoint 120-video train and 40-video validation splits.

```bash
python tracking/stage8_intervention_manifest.py \
  --dataset got10k_train \
  --root "$GOT_ROOT" \
  --sequence_file "$INSTALLED_SEQUENCES" \
  --exclude_manifest "$PILOT_MANIFEST" "$HOLDOUT50" "$HOLDOUT150" \
  --train_sequences 120 \
  --val_sequences 40 \
  --clip_frames 256 \
  --minimum_clip_frames 64 \
  --anchors 12 \
  --horizons 1 5 10 20 40 \
  --seed 2030 \
  --output "$INTERVENTION_MANIFEST"
```

Do not regenerate this manifest after looking at results. Record its printed
fingerprint. If fewer than 160 usable videos remain, keep 40 validation videos
and reduce only `--train_sequences`; do not shorten the 64-frame eligibility
rule.

## 3. One-clip sanity run

```bash
export FIRST_INTERVENTION_SEQ=$(python -c \
  'import json,sys; d=json.load(open(sys.argv[1])); print(next(x["sequence"] for x in d["sequences"] if x["split"]=="train"))' \
  "$INTERVENTION_MANIFEST")

python tracking/stage8_intervention_collect.py \
  --manifest "$INTERVENTION_MANIFEST" \
  --parameter deit_distilled \
  --checkpoint "$CHECKPOINT" \
  --gpu 0 \
  --split train \
  --sequence "$FIRST_INTERVENTION_SEQ" \
  --pool_grid 4 \
  --verify_l9 \
  --output_dir "$INTERVENTION_CACHE"

python tracking/stage8_intervention_summary.py \
  --input "$INTERVENTION_CACHE" \
  --bootstrap_samples 1000
```

The collector must report 12 anchors. The summary must audit one cache and the
cache report must show `control_probe_max_abs_diff` no larger than `1e-6`.

## 4. Small pilot

Use the same cache and exactly the same collection arguments. The run
fingerprint deliberately rejects changed horizons, pooling, checkpoint, code,
or verification settings.

```bash
python tracking/stage8_intervention_collect.py \
  --manifest "$INTERVENTION_MANIFEST" \
  --parameter deit_distilled \
  --checkpoint "$CHECKPOINT" \
  --gpu 0 \
  --split train \
  --max_sequences 8 \
  --pool_grid 4 \
  --verify_l9 \
  --output_dir "$INTERVENTION_CACHE"

python tracking/stage8_intervention_collect.py \
  --manifest "$INTERVENTION_MANIFEST" \
  --parameter deit_distilled \
  --checkpoint "$CHECKPOINT" \
  --gpu 0 \
  --split val \
  --max_sequences 8 \
  --pool_grid 4 \
  --verify_l9 \
  --output_dir "$INTERVENTION_CACHE"

python tracking/stage8_intervention_summary.py \
  --input "$INTERVENTION_CACHE" \
  --bootstrap_samples 2000
```

The pilot is an implementation and scale check, not a decision point for the
scientific hypothesis. Check that all arrays audit successfully, both splits
appear, effects are finite, and H1 differs from longer-horizon effects.

## 5. Full collection

On one GPU:

```bash
python tracking/stage8_intervention_collect.py \
  --manifest "$INTERVENTION_MANIFEST" \
  --parameter deit_distilled \
  --checkpoint "$CHECKPOINT" \
  --gpu 0 \
  --split all \
  --pool_grid 4 \
  --verify_l9 \
  --output_dir "$INTERVENTION_CACHE"
```

The collector is resumable: existing files are fully audited before they are
skipped. Do not use `--overwrite` when resuming. For multiple GPUs, launch one
process per GPU with a common output directory and matching
`--num_shards N --shard_id K` arguments.

Finalize only after every manifest clip is present:

```bash
python tracking/stage8_intervention_summary.py \
  --input "$INTERVENTION_CACHE" \
  --bootstrap_samples 5000 \
  --seed 2030 \
  --require_complete
```

## 6. Outputs

The cache contains one NPZ per sequence. Keep these files out of Git. The four
CSV reports are:

- `stage8_intervention_cache_summary.csv`: cache and replay audit per video.
- `stage8_intervention_anchor_summary.csv`: one row per anchor and horizon.
- `stage8_intervention_effect_summary.csv`: sequence-clustered effect estimates.
- `stage8_intervention_temporal_summary.csv`: exploratory anchor-time bins.

For review, retain the manifest, `run.json`, console output, and all four CSVs.

## 7. Decision rule

Read the `val` rows before the descriptive `all` rows.

- Primary support: H20 `auc_delta_macro > 0` and its clustered 95% interval is
  above zero.
- Persistent correction: the H20 post-anchor interval is also above zero.
- Immediate-only correction: H1 is positive but post-anchor H10/H20/H40 is not.
- No causal value: H1 and longer horizons are near zero or negative.
- Instability: train is positive but validation is not; do not build a learned
  router from that result.

Only after a replicated validation effect should Stage-8C train a gate to
predict intervention value from the saved pre-L9 features. Stage-7 cohorts
remain untouched evaluation sets for any later deployable policy.
