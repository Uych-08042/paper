#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage76="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
run="${STAGE86_ROOT:-$repo/output/stage86/same_output_search_center_v1}"
checkpoint="${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"

case "${1:-run}" in
  run)
    if [[ ! -f "$run/analysis/stage86_decision.json" ]]; then
      python -B tools/stage86_search_center.py --stage76 "$stage76" \
        --output_dir "$run" --checkpoint "$checkpoint" --gpu "${STAGE86_GPU:-0}"
    fi
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage86_search_center.py --check-results --output_dir "$run"

results="$repo/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$results"
bundle="$results/stage86_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json paired.csv video_summary.csv windows.csv scope.csv; do
    printf '\n===== stage86_%s =====\n' "$file"
    cat -- "$run/analysis/stage86_$file"
    printf '\n'
  done
} > "$bundle"
printf 'Stage-86 result bundle: %s\n' "$bundle"
