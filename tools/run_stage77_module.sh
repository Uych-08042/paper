#!/usr/bin/env bash
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
source_root="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
run_root="${STAGE77_ROOT:-$repo/output/stage77/sgla_search_decode_diagnostic_v1}"
checkpoint="${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
python -B tools/stage77_module_diagnostic.py --input "$source_root" --output_dir "$run_root" \
  --checkpoint "$checkpoint" --gpu "${STAGE77_GPU:-0}"

results="$repo/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$results"
bundle="$results/stage77_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
files=(diagnostic.json summary.csv states.csv candidates.csv)
for file in "${files[@]}"; do test -f "$run_root/analysis/stage77_$file"; done
{
  for file in "${files[@]}"; do
    printf '\n===== stage77_%s =====\n' "$file"
    cat -- "$run_root/analysis/stage77_$file"
    printf '\n'
  done
} > "$bundle"
printf 'Stage-77 result bundle: %s\n' "$bundle"
