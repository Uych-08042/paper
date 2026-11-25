#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE195_ROOT:-$repo/output/stage195/router_objective_audit_v1}"
python -u -B tools/stage195_router_objective_audit.py \
  --input "${STAGE194_ROOT:-$repo/output/stage194/module_ablation_v1}" \
  --output-dir "$run" "$@"
if [[ ! -f "$run/stage195_decision.json" ]]; then
  exit 0
fi
bundle="$run/stage195_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json model_scope.csv error_summary.csv policy_summary.csv paired.csv \
      fixed_path_audit.csv error_video.csv policy_video.csv training_history.csv frame_predictions.csv; do
    printf '\n===== stage195_%s =====\n' "$file"
    cat -- "$run/stage195_$file"
  done
} > "$bundle"
printf '[Stage-195] result bundle: %s\n' "$bundle"
