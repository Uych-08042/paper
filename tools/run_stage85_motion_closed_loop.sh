#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage76="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
stage84="${STAGE84_ROOT:-$repo/output/stage84/matched_motion_candidates_v1}"
run="${STAGE85_ROOT:-$repo/output/stage85/motion_candidate_closed_loop_v1}"
checkpoint="${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"

case "${1:-run}" in
  run)
    if [[ ! -f "$run/analysis/stage85_decision.json" ]]; then
      python -B tools/stage85_motion_closed_loop.py --stage76 "$stage76" --stage84 "$stage84" \
        --output_dir "$run" --checkpoint "$checkpoint" --gpu "${STAGE85_GPU:-0}"
    fi
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage85_motion_closed_loop.py --check-results --output_dir "$run"

results="$repo/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$results"
bundle="$results/stage85_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json policy_summary.csv paired.csv video_summary.csv seed_video_summary.csv scope.csv; do
    printf '\n===== stage85_%s =====\n' "$file"
    cat -- "$run/analysis/stage85_$file"
    printf '\n'
  done
} > "$bundle"
printf 'Stage-85 result bundle: %s\n' "$bundle"
