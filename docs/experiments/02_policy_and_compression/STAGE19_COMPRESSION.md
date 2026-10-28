# Stage-19: persistent structural-compression mechanism pilot

This is the first mechanism experiment for a proposed closed-loop compression
method. It does not implement the eventual joint structure/weight learner.
The question is whether compression damages autonomous tracking differently from
tracking on identical reference crops, and whether this difference helps choose
structures across videos.

## Frozen protocol

- Source: only `train` entries of the existing intervention manifest. Its 40 val
  videos, UAV123, and OTB are not used. Both new splits are development data.
- Default split: 12 calibration / 12 development-check physical videos, selected
  without reading outcome labels, seed 2057. Reuse original clips and annotation/
  visibility hashes. Source images must still be available.
- Four anchors per clip, H20. All trajectories start from the same SGLA state.
- Reference: original SGLA L1-L6 plus its selected L7-L12 candidate, not fixed L9.
- Twelve equal-size masks: remove a disjoint, seeded 25% hidden-channel group
  from one prefix FFN (two groups per layer, L1-L6). These are untrained probes.
  Each probe changes one FFN. It does not remove 25% of the entire tracker.
- Compile smaller `fc1` rows/bias and matching `fc2` columns. Preserve `fc2` bias.
  Compression stays active for every frame in the window, not just the anchor.
- Same-crop branch: before every frame restore the reference previous box; this
  also fixes the origin used by bounding-box decoding. Never feed this branch's
  predicted box into its next crop.
- Free branch: feed the compressed model's previous predicted box into its next
  crop. Both branches use the compressed prefix's original SGLA selector rule.
- Audit original reference replay, a physically rebuilt zero-deletion model,
  restoration after surgery, first-frame equality, shapes, IoU recomputation,
  initial state and all crop-state continuity. Check physical surgery against a
  zeroed hidden-channel reference before collecting the first anchor of each mask.
- Eligible labels: first anchor frame visible, >=80% valid H5 and H20 outcomes.
  Use identical eligible anchors for every criterion. Average anchors within each
  video, then videos equally; bootstrap paired video values, never individual
  overlapping frames or anchors as independent observations.

Positive `damage` means mean IoU fell relative to SGLA. Positive `feedback_damage`
means free-rollout damage exceeded same-crop damage. This is a controlled effect
of feeding back predictions; it is not proof of exponential instability. The
selector can also change as crop/prefix features change.

Select one individual mask using calibration mean damage under each of H1,
same-crop H5, and free H5. Compare their free H20 outcomes only on development-check
videos. `MECHANISM_SUPPORTED` requires the free-H5 choice to beat both alternatives
with paired 95% CI lower bounds above zero (at least six videos per split).
`MECHANISM_NOT_ESTABLISHED` is not a universal rejection of compression or RL.
Identical chosen masks yield zero contrast, not evidence of a new criterion.
This pilot does not combine several masks or claim that their effects add.

## Server commands

Run from the repository root in `sglatrack-test`. The existing intervention
manifest and original tracker checkpoint must still be available. Set paths
explicitly so a new login cannot silently expand unset variables to the cwd.
Repeating the manifest command audits and reuses an identical existing file
without writing it. A changed source/configuration or a damaged fingerprint is
rejected; keep the original file and arguments for resuming this frozen run.

```bash
export INTERVENTION_MANIFEST="$PWD/output/stage8/l9_intervention_v1/intervention_manifest.json"
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
export STAGE19_ROOT="$PWD/output/stage19/closed_loop_compression_mechanism_v1"
export STAGE19_MANIFEST="$STAGE19_ROOT/manifest.json"
export STAGE19_CACHE="$STAGE19_ROOT/cache"
test -f "$INTERVENTION_MANIFEST" && test -f "$CHECKPOINT"

python tracking/test_stage19_compression.py
python tracking/stage19_compression_manifest.py \
  --source_manifest "$INTERVENTION_MANIFEST" \
  --output "$STAGE19_MANIFEST" \
  --calibration_sequences 12 --check_sequences 12 --anchors 4 --horizon 20 --seed 2057

export FIRST_STAGE19_SEQ=$(python -c \
  'import json,sys; d=json.load(open(sys.argv[1])); print(next(e["sequence"] for e in d["sequences"] if e["split"]=="train"))' \
  "$STAGE19_MANIFEST")
python tracking/stage19_compression_collect.py \
  --manifest "$STAGE19_MANIFEST" --checkpoint "$CHECKPOINT" \
  --parameter deit_distilled --gpu 0 --sequence "$FIRST_STAGE19_SEQ" \
  --output_dir "$STAGE19_ROOT/sanity_cache"
python tracking/stage19_compression_summary.py \
  --input "$STAGE19_ROOT/sanity_cache" --output_dir "$STAGE19_ROOT/sanity_analysis" \
  --allow_incomplete --bootstrap_samples 1000 --seed 2058
```

After the sanity audit passes, collect the exact frozen full run:

```bash
python tracking/stage19_compression_collect.py \
  --manifest "$STAGE19_MANIFEST" --checkpoint "$CHECKPOINT" \
  --parameter deit_distilled --gpu 0 --output_dir "$STAGE19_CACHE"
python tracking/stage19_compression_summary.py \
  --input "$STAGE19_CACHE" --output_dir "$STAGE19_ROOT/analysis" \
  --bootstrap_samples 5000 --seed 2058
```

Repeating collection audits and skips completed files. It resumes at clip
boundaries; an interrupted clip is recomputed. `--num_shards` / `--shard_id` may
divide clips; use the same manifest, runtime, masks and cache directory. A different
checkpoint, code or compression configuration requires a new output directory.
No overwrite or automatic deletion of old caches is provided.

## What to inspect

`stage19_mask_summary.csv` reports H1/H5/H20 same-crop, free and feedback damage.
`stage19_sequence_summary.csv` supports video-level paired inspection.
`stage19_selection_comparison.csv` is written only after the frozen cohort is
complete. `stage19_decision.json` records the mechanism result and provenance.
Partial data cannot select a structure or produce a positive screen decision.

No latency claim is tested. Parameter reduction and mathematical FFN equivalence
do not establish end-to-end speedup. A successful pilot would justify designing
and comparing a joint compression algorithm; it is not itself a new trained
method, external validation, or a replacement for eventual edge-device profiling.
Old feature caches cannot replay a pruned prefix; collection needs source images.
