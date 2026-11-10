#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage76="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
stage86="${STAGE86_ROOT:-$repo/output/stage86/same_output_search_center_v1}"
stage87="${STAGE87_ROOT:-$repo/output/stage87/search_geometry_action_cross_v1}"
run="${STAGE88_ROOT:-$repo/output/stage88/single_layout_write_support_v1}"
checkpoint="${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
args=(--stage76 "$stage76" --stage86 "$stage86" --stage87 "$stage87" --output_dir "$run")

case "${1:-run}" in
  run)
    if [[ ! -f "$run/analysis/stage88_decision.json" ]]; then
      python -B tools/stage88_single_layout.py "${args[@]}" --checkpoint "$checkpoint" --gpu "${STAGE88_GPU:-0}"
    fi
    ;;
  --plan-only)
    python -B tools/stage88_single_layout.py "${args[@]}" --plan-only
    exit 0
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--plan-only|--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage88_single_layout.py --check-results --output_dir "$run"

results="$repo/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$results"
bundle="$results/stage88_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json paired.csv signal_support.csv windows.csv; do
    printf '\n===== stage88_%s =====\n' "$file"
    cat -- "$run/analysis/stage88_$file"
    printf '\n'
  done
} > "$bundle"
printf 'Stage-88 result bundle: %s\n' "$bundle"
