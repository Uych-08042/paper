# Stage-14 Multi-Action Long-Horizon Counterfactual Audit

## Question

Before training reinforcement learning, test whether choosing among L7-L12 is
both valuable and predictable. The experiment estimates the matched branch
value

`Q^Fast_H(s, a) = one action a at the anchor, then Fast for H-1 frames`

for `a in {Fast, L7, L8, L9, L10, L11, L12}`.

This is a development-only counterfactual audit. It is not an on-policy RL
return, a closed-loop learned-policy result, or external validation.

## Frozen design

- Reuse the audited Stage-8 intervention manifest: 120 train and 40 val clips.
- Use 12 outcome-independent anchors per clip.
- Use `H=20`, `gamma=0.9`, and at least 80% valid annotations.
- Decode L7-L12 from the same L1-L6 token state at each anchor.
- Execute one selected action, then use Fast for the remaining 19 frames.
- Fit six action values with nested 5/4-fold video-grouped OOF Ridge models.
- Primary state: pooled tokens, selector response, Fast response, and causal
  trajectory history.

The predeclared screen passes only when:

1. The long-horizon candidate oracle beats the native selector, and the oracle
   including Fast beats Fast, with both clustered 95% CI lower bounds above
   zero.
2. The primary OOF value model has Pearson correlation at least 0.10.
3. Its selected layer beats the native selector with a clustered 95% CI lower
   bound above zero.

## Server commands

```bash
cd ~/workspace/paper
conda activate sglatrack-test

export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
export INTERVENTION_MANIFEST="$PWD/output/stage8/l9_intervention_v1/intervention_manifest.json"
export STAGE14_ROOT="$PWD/output/stage14/multiaction_counterfactual_v1"
export STAGE14_CACHE="$STAGE14_ROOT/cache"
export STAGE14_AUDIT="$STAGE14_ROOT/audit"

# Keep the exact Stage-8 intervention manifest used by the completed 160-clip run.
test -f "$INTERVENTION_MANIFEST"

export FIRST_STAGE14_SEQ=$(python -c \
  'import json,sys; d=json.load(open(sys.argv[1])); print(d["sequences"][0]["sequence"])' \
  "$INTERVENTION_MANIFEST")
```

### One-clip integrity check

```bash
python tracking/stage14_multiaction_collect.py \
  --manifest "$INTERVENTION_MANIFEST" \
  --parameter deit_distilled \
  --checkpoint "$CHECKPOINT" \
  --gpu 0 \
  --sequence "$FIRST_STAGE14_SEQ" \
  --horizon 20 \
  --gamma 0.9 \
  --min_valid_fraction 0.8 \
  --verify_actions \
  --output_dir "$STAGE14_ROOT/sanity_cache"

python tracking/stage14_multiaction_audit.py \
  --input "$STAGE14_ROOT/sanity_cache" \
  --output_dir "$STAGE14_ROOT/sanity_audit" \
  --allow_incomplete \
  --audit_only
```

The integrity audit must report one audited clip, zero replay discrepancy, and
no exception. Do not interpret one-clip action effects.

### Full collection

The collector is resumable. The following loop runs four shards sequentially;
the same shard commands may instead be launched in separate jobs when each job
has an independent GPU.

```bash
for SHARD in 0 1 2 3; do
  python tracking/stage14_multiaction_collect.py \
    --manifest "$INTERVENTION_MANIFEST" \
    --parameter deit_distilled \
    --checkpoint "$CHECKPOINT" \
    --gpu 0 \
    --horizon 20 \
    --gamma 0.9 \
    --min_valid_fraction 0.8 \
    --verify_actions \
    --num_shards 4 \
    --shard_id "$SHARD" \
    --output_dir "$STAGE14_CACHE"
done

find "$STAGE14_CACHE" -name '*__stage14_multiaction.npz' | wc -l
```

The expected count is 160.

### Frozen audit and OOF screen

```bash
python tracking/stage14_multiaction_audit.py \
  --input "$STAGE14_CACHE" \
  --output_dir "$STAGE14_AUDIT" \
  --variants history_only selector_fast_motion token_selector_fast_motion \
  --primary_variant token_selector_fast_motion \
  --outer_folds 5 \
  --inner_folds 4 \
  --pca_dims 64 \
  --pca_oversample 12 \
  --pca_power_iterations 1 \
  --ridge_alphas 0.01 0.1 1 10 100 1000 \
  --budgets 0 5 10 20 30 50 100 \
  --bootstrap_samples 5000 \
  --seed 2044

cat "$STAGE14_AUDIT/stage14_decision.json"
cat "$STAGE14_AUDIT/stage14_selector_oracle_summary.csv"
cat "$STAGE14_AUDIT/stage14_model_summary.csv"
```

## Decision meanings

- `NO_MULTIACTION_HEADROOM`: changing L7-L12 does not reliably improve on the
  native selector under matched H20 branches. Stop layer-policy development.
- `HEADROOM_NOT_PREDICTABLE`: an oracle layer advantage exists, but the current
  causal state cannot recover it out of fold. Do not start RL with this state;
  redesign state or collect on-policy transitions first.
- `PASS_SCREEN`: a small offline contextual-bandit or fitted-Q pilot is
  justified. It still requires a true switched closed-loop holdout before any
  method claim.

Do not alter horizons, features, alphas, budgets, or the pass rule after reading
the results. Any new design must receive a new stage/output directory.
