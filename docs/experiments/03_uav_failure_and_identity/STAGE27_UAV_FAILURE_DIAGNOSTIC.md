# Stage-27: UAV failure localization before method development

This is a descriptive development audit of the original SGLA checkpoint on
VisDrone2019-SOT **train**. It trains nothing and does not evaluate PPO, a new
student, or a new policy. Prior Stage-1 through Stage-26 files are unchanged.

## Data and selection

The supplied installation has 86 clips / 69,941 images with finite positive
xywh annotations. The manifest repeats the structural checks, then samples
24 distinct `uavNNNNNNN` source-prefix groups, seed 2091, one clip per group.
The prefix is a conservative grouping proxy, not verified physical identity.
Sampling never uses model predictions. Fewer groups cause an explicit error;
there is no outcome-dependent replacement. All selected videos run in full.

Every selected image is decoded on CPU and bound by SHA256 and dimensions.
Unselected images receive structural checks only. The manifest records all
selected GT rows and annotation hashes. Unknown visibility stays UNKNOWN:
positive box width/height is not an absence/occlusion label. No independent
generalization claim is available; pretraining overlap has not been ruled out.

## Tracking and measurements

Initialize from first-frame GT once. At every later frame use the actual prior
prediction to crop, the original selector to choose one independent L7-L12
candidate, and the original tracking head to update the trajectory. Verify
scripted execution against the unmodified network on the first tracked frame.
No learned memory, adapter, controller, GT reset, or shortened pilot clip is used.

Record the predicted parent/box, GT, image dimensions, integer search geometry,
resize factor, raw score-map maximum, candidate identity and official-style IoU.
The crop geometry preserves the legacy right/bottom `+1` padding convention.
Coverage divides GT intersection with **usable image pixels in the crop** by
GT rectangle area; padding supplies no target evidence. GT portion outside the
image is separately reported. Projected GT input dimensions do not imply that
all those pixels are visible or even lie inside the crop.

After each prediction, estimate background apparent motion on resized original
images (max side 640), excluding expanded GT and predicted boxes. Use Shi-Tomasi,
LK and RANSAC partial-affine estimation. At least 20 tracks, 12 inliers and 50%
inliers are required. Failure/resolution changes produce NaN/UNKNOWN, never zero
motion. The proxy is median transformed grid displacement / image diagonal.
GT residual motion and target Laplacian variance are offline review aids, not
flight telemetry or validated blur labels. No diagnostic is fed to the tracker.

## Analysis

Exclude initialization. Use strict 21-threshold success AUC and IoU<0.2 frame
ratios. A failure event requires three consecutive low-IoU frames, and recovery
requires three consecutive IoU>=0.5 frames; video ends censor active events.
Review windows cover ten preceding and thirty following confirmation frames.
Event causes remain `unreviewed`; low IoU does not establish target absence,
identity switch, or memory contamination.

Frozen descriptive strata:

- usable crop coverage: <0.1, [0.1,0.9), >=0.9;
- projected input equivalent target side: <16, [16,32), >=32 pixels;
- background motion: <0.005, [0.005,0.02), >=0.02 image diagonals, plus unknown;
- size x motion combinations restricted to >=0.9 coverage.

Average within each contributing video before bootstrap (5,000 draws, seed
2093); groups with one video have no CI. Report video/frame counts. Different
strata can contain different videos and are not paired treatment effects.
There is no hypothesis pass/fail or automatic method-selection threshold.
Instrumented collection is not a speed benchmark. GT-centered search and
same-crop compact comparisons are NOT part of this first run.

Reports: frame diagnostics, sequence summary, scenario summary, failure events,
and `stage27_diagnostic.json`. Review the events before choosing a mechanism
intervention. Do not use OTB/UAV123 for further policy search.

## Server commands

Copy these six new files from the local repository: this document and
`tracking/stage27_uav_{common,manifest,collect,summary}.py`, plus
`tracking/test_stage27_uav_diagnostic.py`. Keep frozen old dependencies unchanged.

```bash
conda activate sglatrack-test
cd "$HOME/workspace/paper"
export VISDRONE_SOT_ROOT="$HOME/workspace/datasets/VisDrone-SOT/VisDrone2019-SOT-train"
export STAGE27_ROOT="$PWD/output/stage27/uav_failure_diagnostic_v1"
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"

python tracking/test_stage27_uav_diagnostic.py

# CPU only, no GPU allocation: decode selected images and freeze full-video IDs.
python tracking/stage27_uav_manifest.py \
  --dataset_root "${VISDRONE_SOT_ROOT:?}" \
  --sequences 24 --seed 2091 \
  --output_dir "$STAGE27_ROOT"

export FIRST_STAGE27_SEQ=$(python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["sequences"][0]["sequence"])' \
  "$STAGE27_ROOT/manifest.json")

# First selected video, still its full length; original network verification included.
python tracking/stage27_uav_collect.py \
  --manifest "$STAGE27_ROOT/manifest.json" \
  --checkpoint "${CHECKPOINT:?}" \
  --parameter deit_distilled --gpu 0 \
  --sequence "${FIRST_STAGE27_SEQ:?}" \
  --output_dir "$STAGE27_ROOT/cache"

python tracking/stage27_uav_summary.py \
  --input "$STAGE27_ROOT/cache" --allow_incomplete

# After reviewing the first trace, collect the same frozen 24-video run.
python tracking/stage27_uav_collect.py \
  --manifest "$STAGE27_ROOT/manifest.json" \
  --checkpoint "${CHECKPOINT:?}" \
  --parameter deit_distilled --gpu 0 \
  --output_dir "$STAGE27_ROOT/cache"

python tracking/stage27_uav_summary.py --input "$STAGE27_ROOT/cache"
```

The first command after syncing is the test, then the **CPU manifest audit**.
Do not start all-video GPU collection until the image audit and first-video
trace pass. Identical manifests can be reused; changed data/code/configuration
require a new output path. Completed NPZs are audited and skipped on resume;
an interrupted video is recollected from its first frame. Ctrl+C normally
cleans `collect.lock`. After SIGKILL, verify no collector is active before
removing only that stale lock. No multi-writer collection is supported.
