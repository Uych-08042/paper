#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
run="${STAGE213_ROOT:-$repo/output/stage213/fixed_nn_output_ceiling_v1}"
python -u -B tools/stage213_output_ceiling.py \
  --stage212-root "${STAGE212_ROOT:-$repo/output/stage212/online_output_search_state_v1}" \
  --stage212-bundle "${STAGE212_BUNDLE:-$repo/docs/experiments/06_sgla_uav_tracking/results/stage212_results_20261126T153146Z_2174614.txt}" \
  --stage76-root "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89-root "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93-root "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage121-root "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage170-root "${STAGE170_ROOT:-$repo/output/stage170/full_video_cooldown_v1}" \
  --stage188-root "${STAGE188_ROOT:-$repo/output/stage188/temporal_support_v1}" \
  --stage191-root "${STAGE191_ROOT:-$repo/output/stage191/grid_writeback_v1}" \
  --output-dir "$run" "$@"
bundle="$run/stage213_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
files=(stage213_decision.json manifest.json stage213_policy_summary.csv stage213_paired.csv
  stage213_seed_video_summary.csv stage213_complementarity.csv stage213_recovery_events.csv
  stage213_source_replay.csv stage213_source_influence.csv)
{
  for name in "${files[@]}"; do
    printf '\n===== %s =====\n' "$name"
    cat -- "$run/$name"
  done
} > "$bundle"
printf '[Stage-213] result bundle: %s\n' "$bundle"
