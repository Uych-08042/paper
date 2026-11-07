#!/usr/bin/env bash
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage74_root="${STAGE74_ROOT:-$repo/output/stage74/direct_four_candidate_ppo_v1}"
l20_root="${STAGE75_L20_ROOT:-$repo/output/stage75/four_choice_failure_replay_l20_v1}"
checkpoint="${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"

python -B tools/stage75_l20_replay.py --input "$stage74_root" --output_dir "$l20_root" \
  --checkpoint "$checkpoint" --gpu "${STAGE75_L20_GPU:-0}"

results="$repo/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$results"
bundle="$results/stage75_l20_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
files=(stage75_l20_diagnostic.json stage75_l20_device_bridge.csv stage75_l20_branches.csv
       stage75_l40_cached_anchors.csv stage75_l40_cached_same_state.csv stage75_l40_cached_divergence.csv)
for file in "${files[@]}"; do
  test -f "$l20_root/analysis/$file"
done
{
  for file in "${files[@]}"; do
    printf '\n===== %s =====\n' "$file"
    cat -- "$l20_root/analysis/$file"
    printf '\n'
  done
} > "$bundle"
printf 'Stage-75 L20 result bundle: %s\n' "$bundle"
