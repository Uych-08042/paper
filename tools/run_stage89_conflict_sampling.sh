#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage76="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
run="${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}"
checkpoint="${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
case "${1:-run}" in
  run)
    if [[ ! -f "$run/analysis/stage89_decision.json" ]]; then
      python -B tools/stage89_conflict_sampling.py --stage76 "$stage76" --output_dir "$run" \
        --checkpoint "$checkpoint" --gpu "${STAGE89_GPU:-0}"
    fi
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage89_conflict_sampling.py --check-results --output_dir "$run"

results="$repo/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$results"
bundle="$results/stage89_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json support.csv paired.csv scope.csv windows.csv; do
    printf '\n===== stage89_%s =====\n' "$file"
    cat -- "$run/analysis/stage89_$file"
    printf '\n'
  done
} > "$bundle"
printf 'Stage-89 result bundle: %s\n' "$bundle"
