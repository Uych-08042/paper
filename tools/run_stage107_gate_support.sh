#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
source106="${STAGE106_ROOT:-$repo/output/stage106/cross_fitted_commit_gate_v1}"
run="${STAGE107_ROOT:-$repo/output/stage107/fixed_gate_support_audit_v1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
case "${1:-run}" in
  run) python -B tools/stage107_gate_support_audit.py --input "$source106" --output_dir "$run" ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage107_gate_support_audit.py --check-results --output_dir "$run"
bundle="$run/stage107_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json model_scope.csv prediction_summary.csv prediction_macro.csv state_support.csv \
    exposure_units.csv exposure_features.csv focus.csv focus_features.csv focus_neighbors.csv focus_sources.csv; do
    printf '\n===== stage107_%s =====\n' "$file"
    cat -- "$run/analysis/stage107_$file"
  done
} > "$bundle"
printf 'Stage-107 result bundle: %s\n' "$bundle"
