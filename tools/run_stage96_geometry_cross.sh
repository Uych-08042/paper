#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage76="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
stage89="${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}"
stage93="${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}"
stage95="${STAGE95_ROOT:-$repo/output/stage95/recovery_dense_response_v1}"
run="${STAGE96_ROOT:-$repo/output/stage96/search_center_size_cross_v1}"
checkpoint="${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
case "${1:-run}" in
  run)
    python -B tools/stage96_geometry_cross.py --stage76 "$stage76" --stage89 "$stage89" \
      --stage93 "$stage93" --stage95 "$stage95" --output_dir "$run" --checkpoint "$checkpoint" --gpu "${STAGE96_GPU:-0}"
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage96_geometry_cross.py --check-results --output_dir "$run"
bundle="$run/stage96_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json events.csv summary.csv factor_summary.csv; do
    printf '\n===== stage96_%s =====\n' "$file"
    cat -- "$run/analysis/stage96_$file"
  done
} > "$bundle"
printf 'Stage-96 result bundle: %s\n' "$bundle"
