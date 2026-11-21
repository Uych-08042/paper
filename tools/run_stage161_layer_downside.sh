#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE161_ROOT:-$repo/output/stage161/matched_layer_downside_v1}"
python -u -B tools/stage161_layer_downside.py \
  --stage159-root "${STAGE159_ROOT:-$repo/output/stage159/layer_selector_supervision_v1}" \
  --stage160-root "${STAGE160_ROOT:-$repo/output/stage160/calibrated_layer_veto_v1}" \
  --progress-interval "${STAGE161_PROGRESS_SECONDS:-15}" --output-dir "$run"

bundle="$run/stage161_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json summary.csv paired.csv folds.csv choices.csv; do
    printf '\n===== stage161_%s =====\n' "$file"
    cat -- "$run/stage161_$file"
  done
} > "$bundle"
printf '[Stage-161] result bundle: %s\n' "$bundle"
