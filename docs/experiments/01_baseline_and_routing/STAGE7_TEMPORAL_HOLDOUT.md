# Stage 7: Frozen Temporal-Layout Holdout

This experiment freezes the UAV123-selected `late_p10_exact` schedule and tests
it on GOT-10k clips that do not occur in the Stage-4 pilot manifest. It is a
scheduler-transfer diagnostic, not an official GOT-10k benchmark: the base
tracker may have been trained on GOT-10k train videos.

## 1. Create a disjoint manifest

```bash
cd /home/u25600003160604/workspace/paper

export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
export STAGE4_DIR="$PWD/output/stage4"
export PILOT_MANIFEST="$STAGE4_DIR/pilot_manifest.json"
export INSTALLED_SEQUENCES="$STAGE4_DIR/got10k_installed_train.txt"
export GOT_ROOT="/home/u25600003160604/workspace/datasets/GOT10K_subset/train"
export STAGE7_ROOT="$PWD/output/stage7/temporal_holdout_v1"
export HOLDOUT_MANIFEST="$STAGE7_ROOT/holdout_manifest.json"

python tracking/stage7_holdout_manifest.py \
  --dataset got10k_train \
  --root "$GOT_ROOT" \
  --sequence_file "$INSTALLED_SEQUENCES" \
  --exclude_manifest "$PILOT_MANIFEST" \
  --reserve_sequences 1 \
  --holdout_sequences 50 \
  --clip_frames 256 \
  --seed 2027 \
  --output "$HOLDOUT_MANIFEST"
```

The expected message reports `40` excluded videos and `selected overlap=0`.
The one reserve clip keeps the manifest compatible with the Stage-4 schema;
all evaluation commands below use only the 50 `val` clips.

## 2. One-clip sanity check

Read the first holdout name:

```bash
export FIRST_HOLDOUT=$(python -c 'import json,os; d=json.load(open(os.environ["HOLDOUT_MANIFEST"])); print(next(x["sequence"] for x in d["sequences"] if x["split"]=="val"))')
echo "$FIRST_HOLDOUT"
```

Run all four frozen policies in separate output directories:

```bash
for POLICY in fast uniform_p5 late_p10_exact sgla; do
  python tracking/stage7_temporal_holdout_validate.py \
    sglatrack deit_distilled \
    --manifest "$HOLDOUT_MANIFEST" \
    --split val \
    --sequence "$FIRST_HOLDOUT" \
    --policy "$POLICY" \
    --checkpoint "$CHECKPOINT" \
    --gpu 0 \
    --latency_warmup_frames 20 \
    --verify_sgla \
    --output_dir "$STAGE7_ROOT/sanity/$POLICY" \
    --overwrite
done

python tracking/stage6_switched_compare.py \
  --reference "$STAGE7_ROOT/sanity/late_p10_exact" \
  --controls \
    "$STAGE7_ROOT/sanity/fast" \
    "$STAGE7_ROOT/sanity/uniform_p5" \
    "$STAGE7_ROOT/sanity/sgla" \
  --skip_prefix_check \
  --output_dir "$STAGE7_ROOT/sanity/comparison"
```

The `blocks/f` values for `late_p10_exact` and `uniform_p5` must be identical.

## 3. Full 50-clip run

```bash
for POLICY in fast uniform_p5 late_p10_exact sgla; do
  python tracking/stage7_temporal_holdout_validate.py \
    sglatrack deit_distilled \
    --manifest "$HOLDOUT_MANIFEST" \
    --split val \
    --policy "$POLICY" \
    --checkpoint "$CHECKPOINT" \
    --gpu 0 \
    --latency_warmup_frames 20 \
    --output_dir "$STAGE7_ROOT/full/$POLICY"
done
```

The commands are resumable. Existing files are skipped and checked by the
run fingerprint. For multiple GPUs, give each process the same output directory
and use matching `--num_shards N --shard_id K` values.

## 4. Paired report

```bash
python tracking/stage6_switched_compare.py \
  --reference "$STAGE7_ROOT/full/late_p10_exact" \
  --controls \
    "$STAGE7_ROOT/full/fast" \
    "$STAGE7_ROOT/full/uniform_p5" \
    "$STAGE7_ROOT/full/sgla" \
  --skip_prefix_check \
  --bootstrap_samples 5000 \
  --seed 2027 \
  --output_dir "$STAGE7_ROOT/full/comparison"
```

The primary endpoint is `late_p10_exact - uniform_p5`, because those two runs
have the same candidate-block count for every clip. A positive paired 95% CI
supports transfer of the temporal placement effect. A CI crossing zero means
the UAV123 result remains dataset-specific evidence. Comparisons with Fast and
SGLA are secondary accuracy bounds, not equal-cost tests.
