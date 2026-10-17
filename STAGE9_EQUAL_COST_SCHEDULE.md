# Stage-9: Equal-Cost Temporal Schedule Intervention

## Question

Stage-8 established that one fixed-L9 action has a robust immediate causal
benefit, but neither token, selector, Fast-head, nor causal motion-history
features predict that benefit reliably. Stage-9 therefore freezes all routing
and asks a mechanism question:

> With exactly four L9 actions in the first 20 frames, does spreading the
> actions over time outperform concentrating them at the anchor?

Every branch starts from the same uninterrupted Fast state. The frozen policy
set is:

- `fast`: no L9 action;
- `burst4`: L9 at relative offsets `0,1,2,3`;
- `paired4`: L9 at offsets `0,1,10,11`;
- `uniform4`: L9 at offsets `0,5,10,15`.

The primary endpoint is confirmatory-split H20 AUC. `uniform4-burst4` is the
primary equal-cost layout contrast. `uniform4-fast` must also have a clustered
95% confidence-interval lower bound above zero for the predeclared result to
pass. H40, the common action-free tail at offsets 16-19, and the persistence
window at offsets 20-39 are secondary mechanism diagnostics.

## 1. Environment

Run from the repository root on the CUDA server:

```bash
cd /home/u25600003160604/workspace/paper

export GOT_ROOT=/home/u25600003160604/workspace/datasets/GOT10K_subset/train
export INSTALLED_SEQUENCES="$PWD/output/stage4/got10k_installed_train.txt"
export PILOT_MANIFEST="$PWD/output/stage4/pilot_manifest.json"
export HOLDOUT50="$PWD/output/stage7/temporal_holdout_v1/holdout_manifest.json"
export HOLDOUT150="$PWD/output/stage7/temporal_holdout_confirm_v1/holdout_manifest.json"
export INTERVENTION_MANIFEST="$PWD/output/stage8/l9_intervention_v1/intervention_manifest.json"
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"

export STAGE9_ROOT="$PWD/output/stage9/equal_cost_schedule_v1"
export STAGE9_MANIFEST="$STAGE9_ROOT/schedule_manifest.json"
export STAGE9_CACHE="$STAGE9_ROOT/cache"
```

Confirm every exclusion manifest and input exists:

```bash
for PATH_TO_CHECK in \
  "$INSTALLED_SEQUENCES" \
  "$PILOT_MANIFEST" \
  "$HOLDOUT50" \
  "$HOLDOUT150" \
  "$INTERVENTION_MANIFEST" \
  "$CHECKPOINT"; do
  test -f "$PATH_TO_CHECK" || echo "MISSING: $PATH_TO_CHECK"
done
```

## 2. Freeze the fresh manifest

The 10-video train split is an implementation pilot. The 40-video val split is
the untouched confirmatory cohort. All Stage-4, Stage-7, and Stage-8 videos are
excluded explicitly.

```bash
python tracking/stage9_schedule_manifest.py \
  --dataset got10k_train \
  --root "$GOT_ROOT" \
  --sequence_file "$INSTALLED_SEQUENCES" \
  --exclude_manifest \
    "$PILOT_MANIFEST" \
    "$HOLDOUT50" \
    "$HOLDOUT150" \
    "$INTERVENTION_MANIFEST" \
  --pilot_sequences 10 \
  --confirm_sequences 40 \
  --clip_frames 128 \
  --minimum_clip_frames 52 \
  --anchors 12 \
  --seed 2034 \
  --output "$STAGE9_MANIFEST"
```

Do not regenerate the manifest after viewing any Stage-9 result. Record the
printed fingerprint. The 52-frame minimum is exact for H40 plus 12 distinct
anchor positions and must not be reduced. If fewer than 50 eligible videos
remain, the error reports the largest valid pilot/confirmatory split. Preserve
40 confirmatory videos when possible; otherwise use one pilot video and place
every other fresh video in the confirmatory split.

## 3. One-clip sanity run

```bash
export FIRST_STAGE9_SEQ=$(python -c \
  'import json,sys; d=json.load(open(sys.argv[1])); print(next(x["sequence"] for x in d["sequences"] if x["split"]=="train"))' \
  "$STAGE9_MANIFEST")

python tracking/stage9_schedule_collect.py \
  --manifest "$STAGE9_MANIFEST" \
  --parameter deit_distilled \
  --checkpoint "$CHECKPOINT" \
  --gpu 0 \
  --split train \
  --sequence "$FIRST_STAGE9_SEQ" \
  --verify_l9 \
  --output_dir "$STAGE9_CACHE"

python tracking/stage9_schedule_summary.py \
  --input "$STAGE9_CACHE" \
  --bootstrap_samples 1000 \
  --seed 2035
```

The collector must report 12 anchors and treatment cost 4. The audit must show
`control_probe_max_abs_diff <= 1e-6`. An incomplete summary is diagnostic only
and must report `PENDING_INCOMPLETE`.

## 4. Pilot collection

Run all 10 train videos before touching the confirmatory split:

```bash
python tracking/stage9_schedule_collect.py \
  --manifest "$STAGE9_MANIFEST" \
  --parameter deit_distilled \
  --checkpoint "$CHECKPOINT" \
  --gpu 0 \
  --split train \
  --verify_l9 \
  --output_dir "$STAGE9_CACHE"

python tracking/stage9_schedule_summary.py \
  --input "$STAGE9_CACHE" \
  --bootstrap_samples 2000 \
  --seed 2035
```

Use this only to verify runtime, scale, equal costs, and output integrity. Do
not alter schedules, endpoints, or the decision rule from pilot effects.

## 5. Confirmatory collection

```bash
python tracking/stage9_schedule_collect.py \
  --manifest "$STAGE9_MANIFEST" \
  --parameter deit_distilled \
  --checkpoint "$CHECKPOINT" \
  --gpu 0 \
  --split val \
  --verify_l9 \
  --output_dir "$STAGE9_CACHE"
```

The collector is resumable. Existing caches are completely audited before
being skipped, so do not use `--overwrite` when resuming. Multiple GPUs may
share the output directory when every process uses matching
`--num_shards N --shard_id K` arguments.

Finalize only when every manifest clip is present:

```bash
python tracking/stage9_schedule_summary.py \
  --input "$STAGE9_CACHE" \
  --bootstrap_samples 5000 \
  --seed 2035 \
  --require_complete
```

Inspect the frozen decision and primary rows:

```bash
cat "$STAGE9_CACHE/stage9_schedule_decision.json"

grep '^val,20,' \
  "$STAGE9_CACHE/stage9_schedule_pairwise.csv"

cat "$STAGE9_CACHE/stage9_schedule_post_action.csv"
```

The primary result passes only when confirmatory H20 `uniform4-burst4` and
`uniform4-fast` both have clustered 95% confidence-interval lower bounds above
zero. Pooled rows and step-wise effects are descriptive and cannot replace
that decision.
