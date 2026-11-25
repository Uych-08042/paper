#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python -u -B tools/stage201_score_baseline.py \
  --source-bundle "${STAGE200_BUNDLE:-$repo/docs/experiments/06_sgla_uav_tracking/results/stage200_results_20261125T140231Z_1329509.txt}" \
  --output-dir "${STAGE201_ROOT:-$repo/output/stage201/score_baseline_v1}" "$@"
