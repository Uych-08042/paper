#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage76="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
stage86="${STAGE86_ROOT:-$repo/output/stage86/same_output_search_center_v1}"
run="${STAGE87_ROOT:-$repo/output/stage87/search_geometry_action_cross_v1}"
checkpoint="${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
args=(--stage76 "$stage76" --stage86 "$stage86" --output_dir "$run")

case "${1:-run}" in
  run)
    if [[ ! -f "$run/analysis/stage87_decision.json" ]]; then
      python -B tools/stage87_action_cross.py "${args[@]}" --checkpoint "$checkpoint" --gpu "${STAGE87_GPU:-0}"
    fi
    ;;
  --prepare-only)
    python -B tools/stage87_action_cross.py "${args[@]}" --prepare-only
    printf 'Stage-87 first-frame table: %s\n' "$run/analysis/stage87_first.csv"
    exit 0
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--prepare-only|--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage87_action_cross.py --check-results --output_dir "$run"

results="$repo/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$results"
bundle="$results/stage87_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json paired.csv branches.csv first.csv; do
    printf '\n===== stage87_%s =====\n' "$file"
    cat -- "$run/analysis/stage87_$file"
    printf '\n'
  done
} > "$bundle"
printf 'Stage-87 result bundle: %s\n' "$bundle"
