#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE165_ROOT:-$repo/output/stage165/layer_reproposal_v1}"

python -u -B tools/stage165_layer_reproposal.py \
  --stage121-root "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage159-root "${STAGE159_ROOT:-$repo/output/stage159/layer_selector_supervision_v1}" \
  --stage164-root "${STAGE164_ROOT:-$repo/output/stage164/l6_local_match_gain_v1}" \
  --progress-interval "${STAGE165_PROGRESS_SECONDS:-15}" --output-dir "$run"

bundle="$run/stage165_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json summary.csv paired.csv folds.csv focus.csv choices.csv layer_scores.csv oracle_bounds.csv oracle_choices.csv; do
    printf '\n===== stage165_%s =====\n' "$file"
    cat -- "$run/stage165_$file"
  done
} > "$bundle"
printf '[Stage-165] result bundle: %s\n' "$bundle"
