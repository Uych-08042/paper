#!/usr/bin/env bash
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage76="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
stage77="${STAGE77_ROOT:-$repo/output/stage77/sgla_search_decode_diagnostic_v1}"
run="${STAGE78_ROOT:-$repo/output/stage78/sgla_scale_feedback_v1}"
checkpoint="${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
if [[ $# -eq 0 ]]; then
  if [[ ! -f "$run/analysis/stage78_decision.json" ]]; then
    python -B tools/stage78_scale_experiment.py --input "$stage76" --diagnostic "$stage77" \
      --output_dir "$run" --checkpoint "$checkpoint" --gpu "${STAGE78_GPU:-0}"
  fi
elif [[ $# -ne 1 || $1 != --pack-only ]]; then
  printf 'Usage: %s [--pack-only]\n' "$0" >&2
  exit 2
fi
python -B tools/stage78_scale_experiment.py --check-results --output_dir "$run"

results="$repo/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$results"
bundle="$results/stage78_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json windows.csv paired.csv timeline.csv; do
    printf '\n===== stage78_%s =====\n' "$file"
    cat -- "$run/analysis/stage78_$file"
    printf '\n'
  done
} > "$bundle"
printf 'Stage-78 result bundle: %s\n' "$bundle"
