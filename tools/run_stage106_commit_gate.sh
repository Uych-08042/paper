#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage76="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
stage89="${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}"
stage93="${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}"
stage103="${STAGE103_ROOT:-$repo/output/stage103/four_candidate_verification_v1}"
stage105="${STAGE105_ROOT:-$repo/output/stage105/matched_geometry_memory_commit_v1}"
run="${STAGE106_ROOT:-$repo/output/stage106/cross_fitted_commit_gate_v1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
case "${1:-run}" in
  run)
    python -B tools/stage106_commit_gate.py --stage76 "$stage76" --stage89 "$stage89" \
      --stage93 "$stage93" --stage103 "$stage103" --stage105 "$stage105" --output_dir "$run"
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage106_commit_gate.py --check-results --output_dir "$run"
bundle="$run/stage106_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json matched_summary.csv paired.csv fit_support.csv exposure_summary.csv samples.csv sample_decisions.csv exposure_units.csv; do
    printf '\n===== stage106_%s =====\n' "$file"
    cat -- "$run/analysis/stage106_$file"
  done
} > "$bundle"
printf 'Stage-106 result bundle: %s\n' "$bundle"
