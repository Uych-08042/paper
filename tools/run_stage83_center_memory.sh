#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage76="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
stage77="${STAGE77_ROOT:-$repo/output/stage77/sgla_search_decode_diagnostic_v1}"
stage82="${STAGE82_ROOT:-$repo/output/stage82/same_history_scale_supply_v1}"
run="${STAGE83_ROOT:-$repo/output/stage83/conflict_center_memory_v1}"
checkpoint="${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
case "${1:-run}" in
  run)
    if [[ ! -f "$run/analysis/stage83_decision.json" ]]; then
      python -B tools/stage83_center_memory.py --stage76 "$stage76" --stage77 "$stage77" --stage82 "$stage82" \
        --output_dir "$run" --checkpoint "$checkpoint" --gpu "${STAGE83_GPU:-0}"
    fi
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage83_center_memory.py --check-results --output_dir "$run"

results="$repo/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$results"
bundle="$results/stage83_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json policy_summary.csv paired.csv video_summary.csv seed_video_summary.csv scope.csv memory_frames.csv; do
    printf '\n===== stage83_%s =====\n' "$file"
    cat -- "$run/analysis/stage83_$file"
    printf '\n'
  done
} > "$bundle"
printf 'Stage-83 result bundle: %s\n' "$bundle"
