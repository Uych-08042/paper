#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE126_ROOT:-$repo/output/stage126/commit_by_alternate_v1}"
case "${1:-run}" in
  run)
    python -B tools/stage126_factorial_continuation.py \
      --stage76 "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
      --stage89 "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
      --stage93 "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
      --stage112 "${STAGE112_ROOT:-$repo/output/stage112/nested_shared_scoring_budget_v1}" \
      --stage121 "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
      --stage124 "${STAGE124_ROOT:-$repo/output/stage124/candidate_independence_v1}" \
      --stage125 "${STAGE125_ROOT:-$repo/output/stage125/alternate_trigger_continuation_v1}" \
      --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
      --gpu "${STAGE126_GPU:-0}" --output-dir "$run"
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage126_factorial_continuation.py --check-results --output-dir "$run"
bundle="$run/stage126_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json coverage.csv cells.csv effects.csv paired.csv; do
    printf '\n===== stage126_%s =====\n' "$file"
    cat -- "$run/analysis/stage126_$file"
  done
} > "$bundle"
printf 'Stage-126 result bundle: %s\n' "$bundle"
