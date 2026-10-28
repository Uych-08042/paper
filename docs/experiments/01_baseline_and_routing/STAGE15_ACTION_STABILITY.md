# Stage-15 disjoint-window action-value stability audit

## Question

Stage-14 found a large hindsight oracle gap across L7-L12, but the frozen
causal state could not predict the winning layer. Stage-15 asks one narrower
question before any reinforcement-learning data collection:

> Does a layer selected from the first half of an H20 counterfactual retain an
> advantage over the native selector in the disjoint second half?

This is still a ground-truth mechanism diagnostic. It is not a deployable
policy and does not establish an RL result.

## Frozen primary test

- Selection window: post-action frames H1-H10.
- Evaluation window: disjoint post-action frames H11-H20.
- Candidate actions: L7-L12.
- Reference: the checkpoint's native selector action at the same anchor.
- Unit of inference: sequence-macro paired difference.
- Uncertainty: sequence-clustered bootstrap with 5,000 samples.
- Confirmatory scope: the existing Stage-14 `val` split.
- Pass rule: the 95% CI lower bound for selected minus native value is above
  zero.

Odd-to-even transfer, reverse-window transfer, H1/H5/H10/H20 agreement, and
best-versus-second margins are supporting diagnostics only.

## Run

```bash
export STAGE15_DIR="$STAGE14_ROOT/action_stability_h20_v1"

python tracking/stage15_action_stability.py \
  --input "$STAGE14_CACHE" \
  --output_dir "$STAGE15_DIR" \
  --split_point 10 \
  --prefix_horizons 1 5 10 20 \
  --min_valid_fraction 0.8 \
  --bootstrap_samples 5000 \
  --seed 2045
```

## Decision

- `PASS_STABILITY_SCREEN`: collect randomized on-policy L7-L12 transitions
  with a richer recurrent state, then run one frozen offline-RL screen.
- `FAIL_STABILITY_SCREEN`: stop the L7-L12 RL route for this one-step
  intervention design. The Stage-14 hindsight oracle is not sufficiently
  persistent to justify policy training.

The Stage-15 development result must not be presented as closed-loop or
external validation.
