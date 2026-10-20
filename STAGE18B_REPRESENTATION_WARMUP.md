# Stage-18B: Chronological Supervised Representation Warm-up

Stage-18A passed execution checks on five fit videos. Its 10-16 ms timing
did not include contemporaneous Fast/SGLA references. This next run trains
the new memory/adapters on real tracking losses and measures all three
execution paths in one process before committing to full training.

## First run: five-video supervised smoke

Sync the new `stage18_representation_train.py`, `stage18_tracking_loss.py`,
`test_stage18_representation.py`, and `test_stage18_tracking_loss.py` files.
The passed Stage-18A model, environment, and pilot sources remain unchanged.

```bash
python -m unittest discover -s tracking -p 'test_stage18*.py'

export STAGE18_ROOT="$PWD/output/stage18/temporal_layer_policy_v1"
export INTERVENTION_MANIFEST="$PWD/output/stage8/l9_intervention_v1/intervention_manifest.json"
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"

python tracking/stage18_representation_train.py \
  --manifest "${INTERVENTION_MANIFEST:?Set manifest}" \
  --pilot_dir "$STAGE18_ROOT/pilot_v1" \
  --checkpoint "${CHECKPOINT:?Set checkpoint}" \
  --parameter deit_distilled \
  --gpu 0 \
  --mode smoke \
  --epochs 1 \
  --frames 64 \
  --unroll 32 \
  --batch_clips 2 \
  --seed 2053 \
  --output_dir "$STAGE18_ROOT/representation_smoke_v1"

cat "$STAGE18_ROOT/representation_smoke_v1/stage18_representation_decision.json"
cat "$STAGE18_ROOT/representation_smoke_v1/stage18_representation_history.csv"
```

No tune or original validation labels are loaded in smoke mode. The exact
five fit identities and the larger fit/tune separation come from the passed
pilot contract. This run is not a policy-gain or generalization experiment.

## Training semantics

Use a fresh chronological predicted-crop rollout for each video/epoch. Fast
has probability 0.5; each independent candidate has probability 1/12. This
representation-training behavior intentionally has no P30 cap. The deployed
environment's original P30 credit logic is unchanged.

Only the new template-memory projection, read/write memory components,
action embedding, and six adapters are optimized. The old tracker remains
frozen and in eval mode; actor, critic, and policy-specific spatial/observation
encoder are not optimized in this phase. The optimizer is AdamW at 1e-4,
weight decay 0.01, with gradient norm clipped to 1.0.

Accumulate gradients from two videos per optimizer update. Within each
video, backpropagate every 32 frames and detach memory at those boundaries.
Average over valid frames per video, then over videos in the batch. Only
initialization uses GT for tracker state. Invalid GT does not stop the
trajectory, reset a crop, or write GT into memory.

Tracking supervision is `2*GIoU + 5*L1 + center_focal`. Targets are mapped
into the actual predicted search crop. Out-of-crop cases and saturated
probabilities receive explicit treatment in the loss helper; there is no
exception handler that silently replaces a failed loss with zero. The
original selector-similarity loss is omitted.

`PASS_SUPERVISED_SMOKE` requires finite optimization and nonzero observed
tracking-loss gradients in the memory read, recurrent writer, and every
candidate adapter. The first optimizer step need not give writer gradients:
the read projection starts at zero and must first learn a nonzero path.

## Same-process resource audit

After the smoke epoch, run five interleaved-order timing repeats on the same
fit clips for original Fast, original SGLA, and new memory plus a balanced
P30 engineering schedule. Each starts from its own GT initialization, then
uses its own predicted crops. The new actor is evaluated for its overhead,
but its decisions are not used for this engineering schedule.

Exclude image loading and the first 20 frames of each repeat; synchronize
CUDA at timing boundaries. Include crop, prefix, decision modules, selected
block, one head, decode, and memory update. Report per-video/repeat means
and p95 in `stage18_same_process_latency.csv`.

The decision separately records `resource_target_met` and
`memory_to_sgla_ratio`. A gradient pass does not establish speed. If new
execution remains substantially slower, inspect the resource result before
full training; the accuracy-and-efficiency target has not been met.

## Full representation training after smoke review

The trainer implements `--mode full`: 10 epochs by default, up to 128
tracked frames on every fit video, and a fixed chronological tune behavior
up to 256 frames per tune video. Use a different output directory from smoke.
Full training starts from the original zero-residual initialization rather
than treating the five-video smoke weights as an additional training stage.
It selects `best.pt` by tune tracking loss only, not tune policy AUC. Neither
the actor nor PPO has been trained at this point.

`last.pt` resumes completed epochs with optimizer state; an interrupted epoch
is replayed from its preceding checkpoint. The contract rejects changed
code/configuration in the same output directory. Freeze `best.pt` and its
lineage only after full representation training is complete and reviewed.
