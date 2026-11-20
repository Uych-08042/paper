#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE160_ROOT:-$repo/output/stage160/calibrated_layer_veto_v1}"
python -u -B tools/stage160_layer_veto.py \
  --stage159-root "${STAGE159_ROOT:-$repo/output/stage159/layer_selector_supervision_v1}" \
  --progress-interval "${STAGE160_PROGRESS_SECONDS:-15}" --output-dir "$run"

bundle="$run/stage160_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json summary.csv fit.csv calibration.csv choices.csv; do
    printf '\n===== stage160_%s =====\n' "$file"
    cat -- "$run/stage160_$file"
  done
} > "$bundle"
printf '[Stage-160] result bundle: %s\n' "$bundle"
