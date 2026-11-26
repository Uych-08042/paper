#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
run="${STAGE208_ROOT:-$repo/output/stage208/paired_state_interaction_v1}"
python -u -B tools/stage208_state_interaction.py \
  --stage194-root "${STAGE194_ROOT:-$repo/output/stage194/module_ablation_v1}" \
  --stage207-root "${STAGE207_ROOT:-$repo/output/stage207/paired_provider_state_training_v1}" \
  --stage207-bundle "${STAGE207_BUNDLE:-$repo/docs/experiments/06_sgla_uav_tracking/results/stage207_results_20261126T070603Z_1751488.txt}" \
  --stage76-root "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89-root "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93-root "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage121-root "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage170-root "${STAGE170_ROOT:-$repo/output/stage170/full_video_cooldown_v1}" \
  --stage188-root "${STAGE188_ROOT:-$repo/output/stage188/temporal_support_v1}" \
  --stage191-root "${STAGE191_ROOT:-$repo/output/stage191/grid_writeback_v1}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE208_GPU:-0}" --workers "${STAGE208_WORKERS:-2}" --output-dir "$run" "$@"
bundle="$run/stage208_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
files=(stage208_decision.json manifest.json stage208_policy_summary.csv stage208_paired.csv stage208_factor_summary.csv
  stage208_scope_summary.csv stage208_control_replay.csv stage208_support.csv stage208_sampling.csv
  stage208_reference_audit.csv stage208_readout_video.csv stage208_parallel_smoke.csv stage208_evaluation_execution.csv
  stage208_training_history.csv stage208_seed_video_summary.csv stage208_routing.csv)
{
  for name in "${files[@]}"; do
    printf '\n===== %s =====\n' "$name"
    cat -- "$run/$name"
  done
} > "$bundle"
printf '[Stage-208] result bundle: %s\n' "$bundle"
