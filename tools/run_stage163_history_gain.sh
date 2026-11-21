#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE163_ROOT:-$repo/output/stage163/history_gain_v1}"
python -u -B tools/stage163_history_gain.py \
  --stage121-root "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage159-root "${STAGE159_ROOT:-$repo/output/stage159/layer_selector_supervision_v1}" \
  --stage162-root "${STAGE162_ROOT:-$repo/output/stage162/history_ablation_v1}" \
  --progress-interval "${STAGE163_PROGRESS_SECONDS:-15}" --output-dir "$run"

bundle="$run/stage163_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json summary.csv paired.csv folds.csv focus.csv choices.csv; do
    printf '\n===== stage163_%s =====\n' "$file"
    cat -- "$run/stage163_$file"
  done
} > "$bundle"
printf '[Stage-163] result bundle: %s\n' "$bundle"
