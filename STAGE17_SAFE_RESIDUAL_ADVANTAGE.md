# Stage-17A: Safe Residual Advantage Screen

Stage-16C showed that accurate absolute-return prediction did not imply useful
closed-loop action advantages. Stage-17A changes the learning target rather
than retuning the failed P30 policy:

`A(s, Lk) = success(s, Lk) - success(s, Fast)`, for `k = 7,...,12`.

The model is a causal recurrent ensemble. It chooses a candidate layer only
when `ensemble_mean - z * ensemble_std > 0`. The requested budget is a hard
per-trajectory cap, not a quota, so the policy may spend less than P30.
Random-behavior target-rate, remaining-rate, and budget-progress fields are
excluded from the value network, and each budget diagnostic uses only the
matching randomized behavior trajectories.

## Data rule

This is a development-only grouped-OOF screen over the 120 Stage-16 training
videos. The script explicitly excludes the 40 consumed Stage-16 validation
videos and never opens their prepared files. Its output is not a closed-loop
tracking result.

## Run

```bash
export STAGE17_DIR="$PWD/output/stage17/safe_residual_advantage_oof_v1"

python tracking/stage17_residual_advantage.py \
  --input "$STAGE16_CACHE" \
  --freeze_manifest "$STAGE16_FREEZE" \
  --prepared_dir "$STAGE16B_DIR/prepared" \
  --output_dir "$STAGE17_DIR" \
  --device cuda \
  --outer_folds 5 \
  --seeds 2051 2052 \
  --epochs 25 \
  --patience 5 \
  --batch_size 4 \
  --budgets 10 30 50 \
  --primary_budget 30 \
  --uncertainty_z 1 \
  --minimum_advantage 0 \
  --bootstrap_samples 5000 \
  --seed 2051
```

The run is resumable with the exact same command. Completed fold/seed
checkpoints are verified against the experiment fingerprint before reuse.

## Frozen decision

The primary P30 screen passes only if all of the following hold:

- OOF residual Pearson and within-state Pearson are both at least `0.10`.
- The conservative policy uses at least `2%` of frames and never exceeds P30.
- Gain versus Fast has a sequence-clustered 95% CI lower bound above zero.
- Lift over an equal-cost uniform L9 schedule has a 95% CI lower bound above zero.
- At least three candidate layers remain active.

A pass permits freezing a final training-only policy specification. It does
not permit evaluation on the already consumed Stage-16 validation split,
UAV123, or OTB. A genuinely untouched dataset is required for the next
closed-loop claim.
