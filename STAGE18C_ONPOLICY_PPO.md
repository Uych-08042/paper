# Stage-18C: Frozen Representation and Live On-Policy PPO

This implements Phase B of `STAGE18_LONG_TERM_LAYER_POLICY.md`. It is not an
efficacy result. The completed representation run selected epoch 7 by tune
tracking loss, with `best.pt` SHA256:

`021afa8b68ee1347f62e7b381a2aa2549c0cf25b651e26deeb137053d1e9b968`.

Its same-process engineering schedule was 1.1092 times SGLA latency. Preserve
that failed resource target. A lower supervised loss does not establish higher
tracking AUC, useful routing, long-term credit, or improved speed.

## Sync and test

New runtime files, without modifying the passed pilot/warm-up code:

- `tracking/stage18_freeze_representation.py`
- `tracking/stage18_policy_runtime.py`
- `tracking/stage18_ppo_math.py`
- `tracking/stage18_onpolicy_ppo.py`

Also sync their four matching `test_stage18_*.py` files. Run on the server:

```bash
python -m unittest discover -s tracking -p 'test_stage18*.py'
```

The local Windows environment has no PyTorch: NumPy/filesystem tests run
there, but gradient, optimizer, and real tracker tests cannot be claimed from
that environment. The server test run should not skip PyTorch tests.

## Freeze the selected representation

Freezing means recording exact identities and subsequently rejecting changes;
it does not mean renaming a directory or setting filesystem permissions.
The freezer checks the full training decision, best epoch against the complete
history, checkpoint bytes, original passed pilot, source code, and exact
fit/tune video separation. It creates a separate JSON bundle without modifying
source checkpoints. An identical freeze is idempotent; an unequal existing
bundle is never overwritten.

Do not rerun representation training into the frozen directory: even rewriting
an otherwise similar decision changes its bytes and invalidates verification.
Verification deliberately fails instead of silently adopting a new checkpoint.

```bash
export STAGE18_ROOT="$PWD/output/stage18/temporal_layer_policy_v1"
export STAGE18_REP_BUNDLE="$STAGE18_ROOT/frozen_representation_v1.json"
export INTERVENTION_MANIFEST="$PWD/output/stage8/l9_intervention_v1/intervention_manifest.json"
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"

python tracking/stage18_freeze_representation.py \
  --representation_dir "$STAGE18_ROOT/representation_full_v1" \
  --pilot_dir "$STAGE18_ROOT/pilot_v1" \
  --output "$STAGE18_REP_BUNDLE"

python tracking/stage18_freeze_representation.py \
  --verify_bundle "$STAGE18_REP_BUNDLE"
```

The original tracker checkpoint and runtime identity must also match during
PPO startup. Empty required paths fail before launching the tracker.

## Next run: live PPO smoke only

```bash
python tracking/stage18_onpolicy_ppo.py \
  --bundle "${STAGE18_REP_BUNDLE:?Set representation bundle}" \
  --manifest "${INTERVENTION_MANIFEST:?Set manifest}" \
  --checkpoint "${CHECKPOINT:?Set original tracker checkpoint}" \
  --parameter deit_distilled \
  --gpu 0 \
  --mode smoke \
  --seed 2054 \
  --steps 1024 \
  --rollout_steps 256 \
  --output_dir "$STAGE18_ROOT/ppo_smoke_seed2054_v1"

cat "$STAGE18_ROOT/ppo_smoke_seed2054_v1/stage18_ppo_decision.json"
cat "$STAGE18_ROOT/ppo_smoke_seed2054_v1/stage18_ppo_history.csv"
```

This uses only the five original fit-pilot videos. It loads no tune sources
and no original validation sources. Supervision is queried for rewards after
the action and predicted-box state update. Actor inputs and memory writes
receive no GT. Smoke weights are not an extra pretraining stage for full PPO.

`PASS_ONPOLICY_SMOKE` requires actual actor weight changes, finite nonzero
optimization gradients, unchanged frozen tracking/memory weights, and replay
of sampled old log probabilities within 1e-4 before every update. The environment
enforces the P30 cap at every frame. Decision/history files also record peak
GPU allocation, rollout/update wall time, and likelihood replay error. This
status makes no AUC-gain or layer-specialization claim.

## Optimization semantics

Only the actor's spatial/observation encoder and actor/critic output heads
are trainable. Original tracker, six adapters, and memory read/write weights
remain frozen. Memory values still evolve according to actual executed actions.
Actor and critic updates use cached pre-action observations from the current
rollout; these are not previous experiments' cached counterfactual labels.

The seven actions are Fast and the six independent L7-L12 blocks. L12 means
one independent L12 block on the shared prefix, not sequential L7-L12 depth.
There is no privileged L9 action, required layer-diversity score, or forced
spending of the candidate budget. Initially Fast has probability 0.7 and each
candidate 0.05 before the legal-action mask. Candidate usage is learned, not
guaranteed to equal 30%.

Before each action, credit increases by 3 up to 20; candidates require and
consume 10 credit. Credit starts at zero. Every prefix of length T satisfies
`candidate_calls <= floor(3*T/10)`. Each executed frame uses one tracking head.
Value-only bootstrap reads an extra prefix but executes no tracking head or
action and does not change the live environment or consume sampling RNG.

Reward is executed IoU, gamma=0.97, GAE lambda=0.95. Invalid labels retain the
transition and memory/crop updates, contribute reward zero, and are masked out
of policy/value losses and advantage-normalization statistics. No failure
triggers a GT crop reset. An entirely invalid rollout advances and checkpoints
the environment without an optimizer update; it is reported as skipped rather
than repeatedly aborting at the same unlabeled segment.

True video terminals bootstrap zero. Artificial source-clip ends bootstrap
the value of the next real image, then stop the GAE trace before initialization
of another clip. This may read one image beyond the original fit clip from the
same development video; no extra future label is queried for the bootstrap.
Rollout-size boundaries also bootstrap, but do not reset live memory, crop,
or budget. Tune evaluation preserves/restores the complete active fit state.

Each fresh rollout is used for four PPO epochs, then discarded. Use clipping
0.2, Adam LR=3e-4/epsilon=1e-5, value weight 0.5 on raw MSE, entropy weight
0.01, gradient norm cap 0.5. There is no further factor of 0.5 on value MSE.
Sampling and PPO recomputation share the same compact FP32 observation path.

`last.pt` stores model, optimizer, source order, live episode state and RNG.
It also atomically stores the tune-selected weights with their step/score;
completion exports and verifies `best.pt` against that selection. This avoids
mixing a newly written best policy with an older resume checkpoint after an
interruption between file writes.
Reissuing an identical command resumes at the last completed rollout/update;
an interrupted update is replayed from the preceding checkpoint. A changed
code/configuration contract requires a new output directory. Do not run two
writers against one directory or resume using different runtime software.

## Full-run boundary

Review smoke integrity, wall time, GPU memory and the failed representation
resource target before launching the full development budget. The named new
external benchmark must have an acquisition/coverage audit without inspecting
policy outcomes; it is not yet established by these scripts. Do not use OTB,
UAV123, or the 40 consumed original validation videos for policy search.

The trainer already implements `--mode full`, with 96 fit / 24 tune identities,
250,000 transitions per seed, rollout size 2,048, and seeds 2054/2055/2056 in
separate output directories. Each starts directly from the same frozen
tracking representation, not from smoke weights. The untrained policy-specific
encoder and critic are initialized afresh by each seed; actor logits start at
the predeclared 0.7/0.05 probabilities. Do not select a winning seed.

Tune selection uses the exact deployment rule: masked argmax. Evaluate every
20,000 steps and at the final step; select by tune AUC subject to P30, breaking
ties by lower descriptive timing and then earlier step. Reports include the
original Fast/SGLA, memory-only Fast, all six single-layer P30 controls, and a
fixed-seed random-layer P30 control. These are development comparisons, not
external confirmation or controlled repeated latency benchmarks.

Completion of full PPO will still require paired closed-loop comparisons,
gamma=0 and no-memory ablations, and a warmed interleaved latency audit. Neither
a smoke pass nor `COMPLETE_ONPOLICY_TRAINING` substitutes for those results.
