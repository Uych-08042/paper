# Stage 13: Long-Horizon Value-Driven Computation

## Purpose

SGLA is the existing high-compute reference method. Stage 13 does not claim
SGLA or fixed Layer-9 scheduling as a new method. It tests a new decision
principle: execute extra computation only when its causal effect has positive
future trajectory value.

For an anchor state `s_t`, Stage-8 supplies two audited counterfactual branches:

- control: Fast at `t`, then Fast;
- treatment: fixed L9 at `t`, then Fast.

The primary target is the normalized discounted return

```text
V_H(s_t, L9) = sum_k gamma^k [IoU_treatment(t+k) - IoU_control(t+k)]
                / sum_k gamma^k,
```

using valid annotations in an H-frame window. This makes delayed correction
and delayed drift part of the learning target. The model receives only causal
Fast/token/selector/trajectory-history features available before the action.

## Feasibility screen

Two critics use identical features, grouped folds, preprocessing and Ridge
families:

1. `immediate_h1`: predicts only the current-frame intervention effect.
2. `discounted_h20`: predicts the H20 discounted future effect.

All 160 Stage-8 videos are development data. Predictions are nested,
video-grouped OOF predictions. The long-value critic passes only if:

- OOF Pearson is at least 0.10;
- benefit AUROC is at least 0.55;
- at 10%, 20%, or 30% budget, lift over matched random has a positive 95% CI;
- at one of those budgets, long-target ranking beats immediate-target ranking
  with a positive paired 95% CI.

A failure stops controller and reinforcement-learning work for this state
representation. A pass permits a frozen online controller experiment, followed
by a genuinely fresh video-disjoint holdout. It is not itself a tracking result.

## Command

```bash
export STAGE13_VALUE_DIR="$PWD/output/stage13/long_horizon_value_screen_v1"

python tracking/stage13_long_horizon_value.py \
  --input "$INTERVENTION_CACHE" \
  --output_dir "$STAGE13_VALUE_DIR" \
  --variants \
    history_only token_selector motion_only token_motion token_selector_motion \
  --primary_variant token_selector_motion \
  --horizon 20 \
  --gamma 0.9 \
  --min_valid_fraction 0.8 \
  --outer_folds 5 \
  --inner_folds 4 \
  --pca_dims 64 \
  --pca_oversample 12 \
  --pca_power_iterations 1 \
  --ridge_alphas 0.01 0.1 1 10 100 1000 \
  --budgets 0 5 10 20 30 50 100 \
  --bootstrap_samples 5000 \
  --seed 2043
```

The decisive artifact is `stage13_value_decision.json`. Do not tune the
horizon, discount, feature set or screen thresholds after reading its result.
