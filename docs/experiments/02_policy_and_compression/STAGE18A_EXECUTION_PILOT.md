# Stage-18A Execution Pilot

This pilot exercises the new tracking memory and candidate adapters on real
predicted-crop trajectories. It does not train or select a layer policy.
The seven execution paths are Fast and independent L7-L12 candidates.

## Server command

Sync these new files to the server's `tracking/` directory:

- `stage18_temporal_model.py`
- `stage18_environment.py`
- `stage18_pilot.py`
- `test_stage18_temporal.py`

From the repository root in the existing `sglatrack-test` environment:

```bash
python tracking/test_stage18_temporal.py

export INTERVENTION_MANIFEST="$PWD/output/stage8/l9_intervention_v1/intervention_manifest.json"
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
export STAGE18_ROOT="$PWD/output/stage18/temporal_layer_policy_v1"

python tracking/stage18_pilot.py \
  --manifest "${INTERVENTION_MANIFEST:?Set INTERVENTION_MANIFEST}" \
  --checkpoint "${CHECKPOINT:?Set CHECKPOINT}" \
  --parameter deit_distilled \
  --gpu 0 \
  --videos 5 \
  --frames 64 \
  --seed 2053 \
  --output_dir "$STAGE18_ROOT/pilot_v1"

cat "$STAGE18_ROOT/pilot_v1/stage18_pilot_decision.json"
cat "$STAGE18_ROOT/pilot_v1/stage18_pilot_summary.csv"
```

All six tests should execute on the server with PyTorch. A CPU-only machine
without PyTorch can run the geometry and budget tests but skips the three
model/environment tests. Unit tests need no dataset or CUDA.

## Checks and result meaning

Eligible original training videos are split 80/20 by video with seed 2053.
Five fit videos are selected by a seeded hash order, with 64 tracked frames
per video. The contract records both fit and tune identities. The engineering schedule takes an available token
and cycles L7-L12 evenly. The actor is initialized but does not choose the
pilot schedule. No validation video or tracking efficacy score is inspected.

The runtime checks zero-residual identity against all seven original paths,
nonzero gradients through the frozen blocks into a disposable copy of the
memory/adapter model, live memory changes, exact branch snapshot/replay,
all-seven-action coverage, and one actual head call per tracked frame.
Extra head calls in the replay test are counted separately.

The gradient probe temporarily activates the read projection on the disposable
copy because its zero initialization otherwise blocks writer gradients on the
first backward pass. It uses a feature/response surrogate for plumbing only,
not the Phase-A supervised tracking loss. The measured rollout uses the
original untrained zero-residual modules.

Timing excludes image loading, skips the first 20 tracked frames, synchronizes
CUDA at boundaries, and includes memory, observation encoder/actor, crop,
prefix, selected block, head, decode, and memory write. These are pilot
untrained-module measurements; no speedup or trained-policy gain is inferred.

`PASS_EXECUTION_PILOT` establishes only that the tested runtime checks passed.
The next deliverable is the chronological supervised representation trainer.
Architecture, dataset manifest, checkpoint, and code hashes are bound in
`stage18_pilot_contract.json`. Use a new output directory after code changes;
do not delete or overwrite the Stage-16/Stage-17 caches.
