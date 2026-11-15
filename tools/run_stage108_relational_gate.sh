#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
source106="${STAGE106_ROOT:-$repo/output/stage106/cross_fitted_commit_gate_v1}"
source107="${STAGE107_ROOT:-$repo/output/stage107/fixed_gate_support_audit_v1}"
run="${STAGE108_ROOT:-$repo/output/stage108/relational_mode_commit_gate_v1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
case "${1:-run}" in
  run) python -B tools/stage108_relational_commit_gate.py --stage106 "$source106" --stage107 "$source107" --output_dir "$run" ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage108_relational_commit_gate.py --check-results --output_dir "$run"
bundle="$run/stage108_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json model_scope.csv training_summary.csv training_macro.csv policy_summary.csv paired.csv \
    state_predictions.csv sample_decisions.csv focus.csv exposure_units.csv exposure_summary.csv; do
    printf '\n===== stage108_%s =====\n' "$file"
    cat -- "$run/analysis/stage108_$file"
  done
} > "$bundle"
printf 'Stage-108 result bundle: %s\n' "$bundle"
