#!/usr/bin/env bash
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage74_root="${STAGE74_ROOT:-$repo/output/stage74/direct_four_candidate_ppo_v1}"
stage75_root="${STAGE75_ROOT:-$repo/output/stage75/four_choice_failure_replay_v1}"
checkpoint="${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"

python -B tools/stage75_ppo_failure_experiment.py --input "$stage74_root" --output_dir "$stage75_root" \
  --checkpoint "$checkpoint" --gpu "${STAGE75_GPU:-0}"

results="$repo/docs/experiments/06_sgla_uav_tracking/results"
mkdir -p "$results"
bundle="$results/stage75_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
files=(stage75_diagnostic.json stage75_anchors.csv stage75_same_state.csv stage75_divergence.csv stage75_branches.csv)
for file in "${files[@]}"; do
  test -f "$stage75_root/analysis/$file"
done
{
  for file in "${files[@]}"; do
    printf '\n===== %s =====\n' "$file"
    cat -- "$stage75_root/analysis/$file"
    printf '\n'
  done
} > "$bundle"
printf 'Stage-75 result bundle: %s\n' "$bundle"
