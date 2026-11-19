#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
run="${STAGE128_ROOT:-$repo/output/stage128/matched_padding_trim_v1}"
case "${1:-run}" in
  run)
    python -B tools/stage128_padding_trim.py \
      --stage76 "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
      --stage89 "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
      --stage93 "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
      --stage112 "${STAGE112_ROOT:-$repo/output/stage112/nested_shared_scoring_budget_v1}" \
      --stage121 "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
      --stage125 "${STAGE125_ROOT:-$repo/output/stage125/alternate_trigger_continuation_v1}" \
      --stage127 "${STAGE127_ROOT:-$repo/output/stage127/trigger_crop_coverage_v1}" \
      --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
      --gpu "${STAGE128_GPU:-0}" --output-dir "$run"
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage128_padding_trim.py --check-results --output-dir "$run"
cat -- "$run/stage128_summary.csv"
bundle="$run/stage128_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json summary.csv paired.csv points.csv crops.csv; do
    printf '\n===== stage128_%s =====\n' "$file"
    cat -- "$run/stage128_$file"
  done
} > "$bundle"
printf 'Stage-128 result bundle: %s\n' "$bundle"
