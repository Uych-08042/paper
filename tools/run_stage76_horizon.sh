#!/usr/bin/env bash
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
source_root="${STAGE74_ROOT:-$repo/output/stage74/direct_four_candidate_ppo_v1}"
run_root="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
checkpoint="${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
python -B tools/stage76_horizon_experiment.py --input "$source_root" --output_dir "$run_root" \
  --checkpoint "$checkpoint" --gpu "${STAGE76_GPU:-0}"

results="$repo/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$results"
bundle="$results/stage76_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
files=(decision.json policy_summary.csv paired.csv seed_video_summary.csv fit_support.csv scope.csv)
for file in "${files[@]}"; do test -f "$run_root/analysis/stage76_$file"; done
{
  for file in "${files[@]}"; do
    printf '\n===== stage76_%s =====\n' "$file"
    cat -- "$run_root/analysis/stage76_$file"
    printf '\n'
  done
} > "$bundle"
printf 'Stage-76 result bundle: %s\n' "$bundle"
