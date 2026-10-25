# Stage-27 follow-up: read-only onset review

The frozen 24-video manifest and original SGLA traces remain unchanged. This
helper reads their audited NPZ files and emits human-review artifacts only.
There is no tracker forward pass, GPU allocation, model update, GT-centered
intervention, new selection of videos, or automatic cause assignment.

Review **all** persistent low-IoU events. Each contact sheet includes onset
offsets -10, -3, -1, 0, +1, +3, +10 and +30 when those frames exist. Do not
wrap or duplicate missing boundary frames. The recorded source filename is
printed beside the Stage-27 frame index to avoid one-frame indexing mistakes.

Green is GT, red is prediction and blue is the actual recorded integer search
region. Each panel contains the full frame and a separate padded search-region
view. The original crop pixels are reconstructed; Pillow resizing is for human
display, not an exact regeneration of the normalized model input tensor.

The additional event CSV records original GT dimensions, projected input size,
center error / GT equivalent side, width/height/area ratios, raw response peak,
native layer and background/residual motion before the onset. Projected size
does not restore missing raw-image detail. A low onset motion proxy does not
rule out motion immediately before onset, residual target motion, blur or a
bad global registration model. Confidence peaks are not calibrated probabilities.

`manual_review.csv` is a template. Review images before choosing visibility
and mechanism. Keep uncertainty explicit. For example, an IoU collapse alone
cannot establish a distractor switch. Multiple mechanisms can be described in
notes. Existing manual labels are preserved on reruns; they are not incorporated
into the automated descriptive counts. Source/pretraining overlap remains a
limitation, and hand-reviewed cases are not an untouched test set.

## Run on the server

Copy only the three new files in `stage27_event_review_source_v1.zip` into the
repository. Do NOT replace the frozen Stage-27 common/manifest/collector/summary.

```bash
conda activate sglatrack-test
cd "$HOME/workspace/paper"
unzip stage27_event_review_source_v1.zip
export STAGE27_ROOT="$PWD/output/stage27/uav_failure_diagnostic_v1"

python tracking/test_stage27_uav_review.py
python tracking/stage27_uav_review.py \
  --input "$STAGE27_ROOT/cache" \
  --output_dir "$STAGE27_ROOT/review_v1"

# Download this archive to inspect the gallery locally.
python -m zipfile -c "$STAGE27_ROOT/review_v1.zip" "$STAGE27_ROOT/review_v1"
```

Open `review_v1/index.html` after extracting the results archive. Images are
relative local files; no server, network or external JavaScript is required.
`review_summary.json` distinguishes frame-weighted and video-macro failure
fractions. `stage27_event_onset_diagnostic.csv` includes raw size and the prior
ten-frame motion context. `manual_review.csv` starts unreviewed. Rendered PNG
hashes are recorded in `review_outputs.json`.

If Pillow is unavailable, `--tables_only` creates tables without decoding or
rendering images. Use a separate output folder for this mode because its review
contract differs. It is not a replacement for visually reviewing the events.

Changed source traces, render mode or helper code require a new review output
directory. The original source artifacts are always read-only.
