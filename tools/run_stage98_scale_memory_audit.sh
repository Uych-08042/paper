#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage76="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
stage89="${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}"
stage93="${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}"
stage97="${STAGE97_ROOT:-$repo/output/stage97/own_trusted_scale_continuation_v1}"
run="${STAGE98_ROOT:-$repo/output/stage98/cached_scale_memory_audit_v1}"
case "${1:-run}" in
  run)
    python -B tools/stage98_scale_memory_audit.py --stage76 "$stage76" --stage89 "$stage89" \
      --stage93 "$stage93" --stage97 "$stage97" --output_dir "$run"
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage98_scale_memory_audit.py --check-results --output_dir "$run"
bundle="$run/stage98_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json cases.csv branch_summary.csv; do
    printf '\n===== stage98_%s =====\n' "$file"
    cat -- "$run/stage98_$file"
  done
} > "$bundle"
printf 'Stage-98 result bundle: %s\n' "$bundle"
