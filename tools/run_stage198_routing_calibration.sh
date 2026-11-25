#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python -u -B tools/stage198_routing_calibration.py \
  --source-bundle "${STAGE197_BUNDLE:-$repo/docs/experiments/06_sgla_uav_tracking/results/stage197_results_20261125T102154Z_1143642.txt}" \
  --output-dir "${STAGE198_ROOT:-$repo/output/stage198/fit_mean_state_correspondence_v1}" "$@"
