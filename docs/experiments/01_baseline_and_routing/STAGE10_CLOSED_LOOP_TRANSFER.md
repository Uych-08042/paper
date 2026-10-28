# Stage-10: Frozen Periodic-L9 Closed-Loop Transfer

## Purpose and claim boundary

Stage-9 rejected the confirmatory claim that Uniform-4 is superior to Burst-4
at equal cost. It nevertheless showed that sparse L9 actions improve H20 AUC
over Fast, while the post-action advantage disappears. Stage-10 therefore
tests whether the already frozen Stage-5 `periodic_l9_20` operating point
transfers from local interventions to complete closed-loop clip trajectories.

The Stage-9 clips have already been inspected. This is a descriptive
mechanism-to-trajectory transfer diagnostic, not an independent confirmatory
result. Do not add policies, tune the 20% rate, or select a result after seeing
these outcomes. The frozen comparison is:

- `fast`: no candidate block;
- `periodic_l9_20`: fixed L9 on 20% of tracked frames;
- `sgla`: one selector-chosen candidate block on every tracked frame.

## 1. Environment

Run from the CUDA server repository root:

```bash
cd /home/u25600003160604/workspace/paper

export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
export STAGE9_ROOT="$PWD/output/stage9/equal_cost_schedule_v1"
export STAGE9_MANIFEST="$STAGE9_ROOT/schedule_manifest.json"
export STAGE10_ROOT="$PWD/output/stage10/periodic_l9_20_transfer_v1"
export STAGE10_FAST="$STAGE10_ROOT/fast"
export STAGE10_P20="$STAGE10_ROOT/periodic_l9_20"
export STAGE10_SGLA="$STAGE10_ROOT/sgla"
```

## 2. One-clip sanity run

Use the single Stage-9 pilot clip. This checks execution and exact periodic
cost without consuming a new cohort.

```bash
export FIRST_STAGE9_SEQ=$(python -c \
  'import json,sys; d=json.load(open(sys.argv[1])); print(next(x["sequence"] for x in d["sequences"] if x["split"]=="train"))' \
  "$STAGE9_MANIFEST")

for POLICY in fast periodic_l9_20 sgla; do
  python tracking/stage10_closed_loop_transfer.py \
    sglatrack deit_distilled \
    --manifest "$STAGE9_MANIFEST" \
    --split train \
    --sequence "$FIRST_STAGE9_SEQ" \
    --policy "$POLICY" \
    --checkpoint "$CHECKPOINT" \
    --gpu 0 \
    --latency_warmup_frames 20 \
    --verify_sgla \
    --output_dir "$STAGE10_ROOT/sanity/$POLICY"
done

python tracking/stage10_closed_loop_summary.py \
  --fast "$STAGE10_ROOT/sanity/fast" \
  --periodic "$STAGE10_ROOT/sanity/periodic_l9_20" \
  --sgla "$STAGE10_ROOT/sanity/sgla" \
  --bootstrap_samples 1000 \
  --seed 2036 \
  --output_dir "$STAGE10_ROOT/sanity/comparison"
```

The periodic run should report a candidate-block ratio close to 20%. The
summary must audit one matched clip under all three policies.

## 3. Full Stage-9 validation split

Run the 28 confirmatory clips under each frozen policy. The collector is
resumable, so omit `--overwrite` after the sanity check.

```bash
for POLICY in fast periodic_l9_20 sgla; do
  python tracking/stage10_closed_loop_transfer.py \
    sglatrack deit_distilled \
    --manifest "$STAGE9_MANIFEST" \
    --split val \
    --policy "$POLICY" \
    --checkpoint "$CHECKPOINT" \
    --gpu 0 \
    --latency_warmup_frames 20 \
    --verify_sgla \
    --output_dir "$STAGE10_ROOT/$POLICY"
done
```

Multiple GPUs may share each policy directory only when every process uses
matching `--num_shards N --shard_id K` values.

## 4. Frozen analysis

```bash
python tracking/stage10_closed_loop_summary.py \
  --fast "$STAGE10_FAST" \
  --periodic "$STAGE10_P20" \
  --sgla "$STAGE10_SGLA" \
  --bootstrap_samples 5000 \
  --seed 2036 \
  --output_dir "$STAGE10_ROOT/comparison"

cat "$STAGE10_ROOT/comparison/stage10_closed_loop_summary.csv"
cat "$STAGE10_ROOT/comparison/stage10_closed_loop_paired.csv"
cat "$STAGE10_ROOT/comparison/stage10_closed_loop_analysis.json"
```

Report macro AUC, paired sequence-level confidence intervals, candidate blocks
per frame, SGLA-gain recovery, and measured model/decode latency. The primary
diagnostic contrast is `periodic_l9_20 - fast`; `periodic_l9_20 - sgla` reports
remaining accuracy headroom. Regardless of outcome, retain the explicit
post-Stage-9 descriptive label in the manuscript and use UAV123 Stage-5 as the
independent full-sequence closed-loop benchmark.
