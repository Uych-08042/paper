#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage76="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
stage89="${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}"
stage93="${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}"
stage112="${STAGE112_ROOT:-$repo/output/stage112/nested_shared_scoring_budget_v1}"
stage113="${STAGE113_ROOT:-$repo/output/stage113/frozen_generated_shared_closed_loop_v1}"
run="${STAGE114_ROOT:-$repo/output/stage114/cached_generated_failure_chain_v1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
case "${1:-run}" in
  run)
    python -B tools/stage114_cached_failure_chain.py --stage76 "$stage76" --stage89 "$stage89" \
      --stage93 "$stage93" --stage112 "$stage112" --stage113 "$stage113" --output_dir "$run"
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage114_cached_failure_chain.py --check-results --output_dir "$run"
bundle="$run/stage114_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json policy_summary.csv seed_video.csv cases.csv failure_events.csv recovery.csv prefix.csv support_summary.csv scope.csv; do
    printf '\n===== stage114_%s =====\n' "$file"
    cat -- "$run/analysis/stage114_$file"
  done
} > "$bundle"
printf 'Stage-114 result bundle: %s\n' "$bundle"
