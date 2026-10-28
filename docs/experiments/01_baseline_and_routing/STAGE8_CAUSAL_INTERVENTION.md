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

## 8. Immediate-effect budget oracle

The completed 120/40 run rejected the predeclared persistent H20 hypothesis on
validation (`-0.0289` AUC, clustered 95% CI `[-0.4217, 0.2281]`) but replicated
an immediate H1 effect (`+1.4385`, CI `[0.8730, 2.1032]`). Therefore Stage-8C
first measures the upper bound for selecting immediate L9 actions:

```bash
python tracking/stage8_immediate_oracle.py \
  --input "$INTERVENTION_CACHE" \
  --budgets 0 5 10 20 30 50 100 \
  --bootstrap_samples 5000 \
  --seed 2031
```

`exact_topk` spends the complete budget even when the remaining actions are
harmful. `positive_cap` treats the budget as a maximum and abstains whenever
the ground-truth immediate IoU delta is non-positive. Both are ground-truth
upper bounds. Their matched-random baselines use exactly the selected action
ratio.

Proceed to a learned gate only if the validation Oracle has substantial
low-budget recovery and its lift over matched-random is positive. A weak
Oracle means the intervention itself is too diffuse for routing. A strong
Oracle with a weak learned model means the pre-action features are the
bottleneck. In either case, H20 must not be presented as a positive result.

## 9. Immediate-effect causal gate

The completed Oracle has substantial and replicated low-budget headroom: on
validation, 10% of anchors recover 42.81% of the positive immediate effect and
outperform matched random selection. Train the gate only for signed H1 IoU
gain. The 120 training videos select every alpha by video-grouped cross-
validation; the 40 validation videos stay outside PCA, normalization, fitting,
and hyperparameter selection.

`token_selector` is the predeclared primary model. The other variants are
feature and latency ablations. In particular, `fast_only` and `token_fast`
require the Fast head before deciding whether to execute L9.

```bash
export STAGE8_GATE_DIR="$PWD/output/stage8/l9_intervention_v1/immediate_gate_v1"

python tracking/stage8_immediate_gate.py \
  --input "$INTERVENTION_CACHE" \
  --output_dir "$STAGE8_GATE_DIR" \
  --variants \
    history_only selector_only token_only token_selector fast_only token_fast \
  --primary_variant token_selector \
  --folds 5 \
  --pca_dims 64 \
  --pca_oversample 12 \
  --pca_power_iterations 1 \
  --ridge_alphas 0.001 0.01 0.1 1 10 100 1000 \
  --budgets 0 5 10 20 30 50 100 \
  --bootstrap_samples 5000 \
  --seed 2032
```

The primary reports are:

- `stage8_immediate_gate_summary.csv`: grouped-CV and untouched-validation
  prediction diagnostics.
- `stage8_immediate_gate_budget.csv`: frozen-score gains, matched-random lift,
  Oracle recovery, and selected benefit/harm rates.
- `stage8_immediate_gate_alpha_selection.csv`: training-only model selection.
- `stage8_immediate_gate_val_predictions.csv`: auditable frozen predictions.

The exact top-k budgets use all validation scores but no validation labels, so
they test ranking under a known offline budget. They are not yet an online
scheduler. Continue to a held-out intervention collection and true closed-loop
gate only when the primary model has useful validation AUROC and its 10%-30%
matched-random lift is positive. If only a Fast-aware ablation succeeds, report
its extra head cost and benchmark it separately. If all learned variants fail
while the Oracle remains strong, conclude that the saved pre-L9 features, not
the intervention, are the current bottleneck.

## 10. Frozen Stage-8D result

The predeclared `token_selector` gate did not pass the continuation rule. Its
training OOF benefit AUROC was `0.5078`; untouched-validation AUROC and Pearson
correlation were only `0.5323` and `0.0538`. At 10% budget it recovered 20.28%
of the Oracle cap, but its IoU lift over matched random was only `0.000271`,
with a clustered 95% interval of `[-0.001083, 0.001806]`. The corresponding
intervals also crossed zero at 5%, 20%, 30%, and 50% budgets.

The ablations do not rescue the result. Selector features have exactly zero
measurable marginal value over the matching history/token variants. Fast-aware
features improve training OOF regression slightly but do not transfer: their
validation benefit AUROCs remain near chance. Therefore these models must not
be taken into a claimed learned closed-loop policy, and reinforcement learning
is not justified by this evidence.

The supported result is narrower but useful: a single L9 action has a robust
average immediate causal benefit and a strong ground-truth selection Oracle,
while the currently saved pre-L9 features cannot predict that benefit. A next
exploratory model may add genuine causal trajectory history (past velocity,
scale change, acceleration, and prediction jitter), because `history_only` in
Stage-8D contains state geometry but no multi-frame motion history. This new
hypothesis must be screened by grouped training OOF results first. Any positive
model then requires a fresh video holdout; the existing 40 validation videos
have already served their confirmatory purpose.

## 11. Stage-8E trajectory-history screen

Stage-8E reads only the 120 manifest training videos. It derives causal motion
features from Fast predictions strictly before each intervention anchor. Its
inner video folds choose Ridge regularization and its outer video folds produce
the reported scores. The existing 40-video validation split is not loaded.

```bash
export STAGE8_MOTION_DIR="$PWD/output/stage8/l9_intervention_v1/motion_gate_oof_v1"

python tracking/stage8_motion_gate.py \
  --input "$INTERVENTION_CACHE" \
  --output_dir "$STAGE8_MOTION_DIR" \
  --variants \
    history_only token_selector motion_only motion_selector \
    token_motion token_selector_motion \
  --primary_variant token_selector_motion \
  --outer_folds 5 \
  --inner_folds 4 \
  --pca_dims 64 \
  --pca_oversample 12 \
  --pca_power_iterations 1 \
  --ridge_alphas 0.01 0.1 1 10 100 1000 \
  --budgets 0 5 10 20 30 50 100 \
  --bootstrap_samples 5000 \
  --seed 2033
```

This CPU-only nested run fits 25 fold-specific PCAs and will be slower than
Stage-8D. The predeclared primary screen passes only if all of the following
hold in nested training OOF predictions:

- benefit AUROC is at least `0.55`;
- signed-delta Pearson correlation is at least `0.10`;
- at least one of the 10%, 20%, or 30% budget lift intervals is above zero.

A failure ends the current frame-level gate branch; do not inspect the old
40-video validation split again. A pass permits fitting a frozen model and
collecting a new video-disjoint intervention holdout before any closed-loop
claim. The primary outputs are `stage8_motion_gate_summary.csv` and
`stage8_motion_gate_budget.csv`.

## 12. Nested Stage-8E result

The predeclared `token_selector_motion` screen failed. Across 120 training
videos and 1,440 intervention anchors, its nested OOF benefit AUROC was
`0.5144`, signed-delta Pearson correlation was `0.0412`, and weighted MSE was
`0.0013814`, compared with `0.0013547` for the fold-specific constant
baseline. None of the 10%, 20%, or 30% matched-random lift intervals excluded
zero. Performance was also fold-dependent: per-fold AUROC ranged from
`0.4779` to `0.6002`, with only one of five outer folds reaching 0.60.

Motion history provides no reproducible marginal value in this representation.
`motion_selector` is numerically identical to `motion_only`, and
`token_selector_motion` is numerically identical to `token_motion`, confirming
that the saved selector features add no measurable information. The strongest
single baseline correlation came from `token_selector` (`0.1032`), but its
benefit AUROC remained `0.5043`, so it also fails the joint continuation rule.

At 50% budget, the primary ranking has an exploratory positive lift of
`0.002263` with a clustered 95% interval of `[0.000778, 0.003723]`. This does
not rescue the gate: it was outside the predeclared low-budget continuation
criterion, appears despite near-chance classification and correlation, and is
not consistent with the 5%-30% intervals. It may motivate a separate
non-learned high-coverage or burst-scheduling study, but it is not evidence for
a deployable frame-level router.

Accordingly, the current frame-level learned-gating branch stops here. The
existing 40 validation videos remain untouched by Stage-8E, no model is fitted
for deployment, and no reinforcement-learning experiment is justified from
these labels. Further experiments should test predeclared, label-free temporal
allocation rules or consolidate the fixed/periodic Pareto result, rather than
continue feature search on these intervention anchors.
