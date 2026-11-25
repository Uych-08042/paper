#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE196_ROOT:-$repo/output/stage196/relative_loss_comparison_v1}"
python -u -B tools/stage196_relative_loss.py \
  --stage194-root "${STAGE194_ROOT:-$repo/output/stage194/module_ablation_v1}" \
  --stage195-root "${STAGE195_ROOT:-$repo/output/stage195/router_objective_audit_v1}" \
  --output-dir "$run" "$@"
if [[ ! -f "$run/stage196_decision.json" ]]; then
  exit 0
fi
bundle="$run/stage196_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json model_scope.csv policy_summary.csv paired.csv error_summary.csv \
      source_bridge.csv error_video.csv policy_video.csv training_history.csv frame_predictions.csv; do
    printf '\n===== stage196_%s =====\n' "$file"
    cat -- "$run/stage196_$file"
  done
} > "$bundle"
printf '[Stage-196] result bundle: %s\n' "$bundle"
