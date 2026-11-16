#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
source118="${STAGE118_ROOT:-$repo/output/stage118/provider_isolated_commit_learning_v1}"
run="${STAGE119_ROOT:-$repo/output/stage119/fixed_commit_support_diagnostic_v1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
case "${1:-run}" in
  run) python -B tools/stage119_commit_support.py --stage118 "$source118" --output_dir "$run" ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage119_commit_support.py --check-results --output_dir "$run"
bundle="$run/stage119_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json scope.csv fit_quality.csv fit_support.csv generalization.csv support_summary.csv focus.csv sample_audit.csv; do
    printf '\n===== stage119_%s =====\n' "$file"
    cat -- "$run/analysis/stage119_$file"
  done
} > "$bundle"
printf 'Stage-119 result bundle: %s\n' "$bundle"
