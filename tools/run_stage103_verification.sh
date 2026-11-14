#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage76="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
stage89="${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}"
stage93="${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}"
stage100="${STAGE100_ROOT:-$repo/output/stage100/online_scale_episode_pulse_v1}"
stage102="${STAGE102_ROOT:-$repo/output/stage102/cached_state_chain_v1}"
run="${STAGE103_ROOT:-$repo/output/stage103/four_candidate_verification_v1}"
checkpoint="${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
case "${1:-run}" in
  run)
    python -B tools/stage103_candidate_verification.py --stage76 "$stage76" --stage89 "$stage89" \
      --stage93 "$stage93" --stage100 "$stage100" --stage102 "$stage102" --output_dir "$run" \
      --checkpoint "$checkpoint" --gpu "${STAGE103_GPU:-0}"
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage103_candidate_verification.py --check-results --output_dir "$run"
bundle="$run/stage103_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json policy_summary.csv paired.csv video_summary.csv seed_video.csv scope.csv recovery.csv matched.csv matched_candidates.csv; do
    printf '\n===== stage103_%s =====\n' "$file"
    cat -- "$run/analysis/stage103_$file"
  done
} > "$bundle"
printf 'Stage-103 result bundle: %s\n' "$bundle"
