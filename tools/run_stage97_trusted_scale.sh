#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage76="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
stage89="${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}"
stage93="${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}"
stage96="${STAGE96_ROOT:-$repo/output/stage96/search_center_size_cross_v1}"
run="${STAGE97_ROOT:-$repo/output/stage97/own_trusted_scale_continuation_v1}"
checkpoint="${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
case "${1:-run}" in
  run)
    python -B tools/stage97_trusted_scale.py --stage76 "$stage76" --stage89 "$stage89" \
      --stage93 "$stage93" --stage96 "$stage96" --output_dir "$run" --checkpoint "$checkpoint" --gpu "${STAGE97_GPU:-0}"
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage97_trusted_scale.py --check-results --output_dir "$run"
bundle="$run/stage97_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json windows.csv video_summary.csv summary.csv; do
    printf '\n===== stage97_%s =====\n' "$file"
    cat -- "$run/analysis/stage97_$file"
  done
} > "$bundle"
printf 'Stage-97 result bundle: %s\n' "$bundle"
