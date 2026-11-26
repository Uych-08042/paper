#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE207_ROOT:-$repo/output/stage207/paired_provider_state_training_v1}"
python -u -B tools/stage207_state_training.py \
  --stage194-root "${STAGE194_ROOT:-$repo/output/stage194/module_ablation_v1}" \
  --stage206-root "${STAGE206_ROOT:-$repo/output/stage206/native_routed_spatial_adaptation_v1}" \
  --stage206-bundle "${STAGE206_BUNDLE:-$repo/docs/experiments/06_sgla_uav_tracking/results/stage206_results_20261126T042709Z_1653346.txt}" \
  --stage76-root "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89-root "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93-root "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage121-root "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage170-root "${STAGE170_ROOT:-$repo/output/stage170/full_video_cooldown_v1}" \
  --stage188-root "${STAGE188_ROOT:-$repo/output/stage188/temporal_support_v1}" \
  --stage191-root "${STAGE191_ROOT:-$repo/output/stage191/grid_writeback_v1}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE207_GPU:-0}" --output-dir "$run" "$@"
files=(stage207_support_decision.json manifest.json stage207_source_replay.csv stage207_provider_scope.csv stage207_support.csv stage207_sampling.csv)
if [[ -f "$run/stage207_decision.json" ]]; then
  files+=(stage207_decision.json stage207_policy_summary.csv stage207_paired.csv stage207_scope_summary.csv \
    stage207_training_history.csv stage207_control_replay.csv stage207_readout_video.csv stage207_seed_video_summary.csv stage207_routing.csv)
elif [[ ! -f "$run/stage207_support_decision.json" ]]; then
  exit 0
fi
bundle="$run/stage207_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for name in "${files[@]}"; do
    printf '\n===== %s =====\n' "$name"
    cat -- "$run/$name"
  done
} > "$bundle"
printf '[Stage-207] result bundle (complete or support preflight): %s\n' "$bundle"
