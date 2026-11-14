#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage76="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
stage89="${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}"
stage93="${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}"
stage100="${STAGE100_ROOT:-$repo/output/stage100/online_scale_episode_pulse_v1}"
stage101="${STAGE101_ROOT:-$repo/output/stage101/matched_scale_trigger_v1}"
run="${STAGE102_ROOT:-$repo/output/stage102/cached_state_chain_v1}"
case "${1:-run}" in
  run)
    python -B tools/stage102_state_chain_audit.py --stage76 "$stage76" --stage89 "$stage89" \
      --stage93 "$stage93" --stage100 "$stage100" --stage101 "$stage101" --output_dir "$run"
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage102_state_chain_audit.py --check-results --output_dir "$run"
bundle="$run/stage102_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json summary.csv unit_summary.csv cases.csv trigger_summary.csv recovery.csv branches.csv chain_excerpt.csv reference_summary.csv; do
    printf '\n===== stage102_%s =====\n' "$file"
    cat -- "$run/analysis/stage102_$file"
  done
} > "$bundle"
printf 'Stage-102 result bundle: %s\n' "$bundle"
