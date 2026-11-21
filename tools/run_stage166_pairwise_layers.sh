#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE166_ROOT:-$repo/output/stage166/pairwise_layer_ranking_v1}"

python -u -B tools/stage166_pairwise_layers.py \
  --stage159-root "${STAGE159_ROOT:-$repo/output/stage159/layer_selector_supervision_v1}" \
  --stage165-root "${STAGE165_ROOT:-$repo/output/stage165/layer_reproposal_v1}" \
  --progress-interval "${STAGE166_PROGRESS_SECONDS:-15}" --output-dir "$run"

bundle="$run/stage166_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json summary.csv paired.csv folds.csv focus.csv choices.csv pair_quality.csv fit.csv layer_scores.csv; do
    printf '\n===== stage166_%s =====\n' "$file"
    cat -- "$run/stage166_$file"
  done
} > "$bundle"
printf '[Stage-166] result bundle: %s\n' "$bundle"
