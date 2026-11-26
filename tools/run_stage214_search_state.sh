#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
run="${STAGE214_ROOT:-$repo/output/stage214/learned_search_state_v1}"
python -u -B tools/stage214_search_state.py \
  --stage213-bundle "${STAGE213_BUNDLE:-$repo/docs/experiments/06_sgla_uav_tracking/results/stage213_results_20261126T161130Z_2281744.txt}" \
  --stage212-bundle "${STAGE212_BUNDLE:-$repo/docs/experiments/06_sgla_uav_tracking/results/stage212_results_20261126T153146Z_2174614.txt}" \
  --stage208-bundle "${STAGE208_BUNDLE:-$repo/docs/experiments/06_sgla_uav_tracking/results/stage208_results_20261126T095430Z_1911332.txt}" \
  --stage212-root "${STAGE212_ROOT:-$repo/output/stage212/online_output_search_state_v1}" \
  --stage208-root "${STAGE208_ROOT:-$repo/output/stage208/paired_state_interaction_v1}" \
  --stage76-root "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89-root "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93-root "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage121-root "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage170-root "${STAGE170_ROOT:-$repo/output/stage170/full_video_cooldown_v1}" \
  --stage188-root "${STAGE188_ROOT:-$repo/output/stage188/temporal_support_v1}" \
  --stage191-root "${STAGE191_ROOT:-$repo/output/stage191/grid_writeback_v1}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE214_GPU:-0}" --workers "${STAGE214_WORKERS:-2}" --output-dir "$run" "$@"
bundle="$run/stage214_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
files=(stage214_decision.json manifest.json stage214_frozen_scope.csv stage214_fit_support.csv
  stage214_scope_summary.csv stage214_training_history.csv stage214_smoke.csv stage214_source_replay.csv
  stage214_policy_summary.csv stage214_paired.csv stage214_seed_video_summary.csv
  stage214_search_geometry.csv stage214_calls.csv stage214_execution.csv)
{
  for name in "${files[@]}"; do
    printf '\n===== %s =====\n' "$name"
    cat -- "$run/$name"
  done
} > "$bundle"
printf '[Stage-214] result bundle: %s\n' "$bundle"
