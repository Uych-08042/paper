#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage76="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
stage89="${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}"
stage93="${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}"
stage103="${STAGE103_ROOT:-$repo/output/stage103/four_candidate_verification_v1}"
run="${STAGE104_ROOT:-$repo/output/stage104/cached_replacement_state_audit_v1}"
case "${1:-run}" in
  run)
    python -B tools/stage104_replacement_audit.py --stage76 "$stage76" --stage89 "$stage89" \
      --stage93 "$stage93" --stage103 "$stage103" --output_dir "$run"
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage104_replacement_audit.py --check-results --output_dir "$run"
bundle="$run/stage104_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json summary.csv unit_summary.csv cases.csv recovery.csv signal_summary.csv reference_summary.csv chain_excerpt.csv; do
    printf '\n===== stage104_%s =====\n' "$file"
    cat -- "$run/analysis/stage104_$file"
  done
} > "$bundle"
printf 'Stage-104 result bundle: %s\n' "$bundle"
