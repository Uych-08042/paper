#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
run="${STAGE211_ROOT:-$repo/output/stage211/recovery_center_size_v1}"
python -u -B tools/stage211_geometry_audit.py \
  --stage210-root "${STAGE210_ROOT:-$repo/output/stage210/all_source_recovery_audit_v1}" \
  --stage210-bundle "${STAGE210_BUNDLE:-$repo/docs/experiments/06_sgla_uav_tracking/results/stage210_results_20261126T121031Z_2077681.txt}" \
  --stage208-root "${STAGE208_ROOT:-$repo/output/stage208/paired_state_interaction_v1}" \
  --stage208-bundle "${STAGE208_BUNDLE:-$repo/docs/experiments/06_sgla_uav_tracking/results/stage208_results_20261126T095430Z_1911332.txt}" \
  --stage76-root "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89-root "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93-root "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage121-root "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage170-root "${STAGE170_ROOT:-$repo/output/stage170/full_video_cooldown_v1}" \
  --stage188-root "${STAGE188_ROOT:-$repo/output/stage188/temporal_support_v1}" \
  --stage191-root "${STAGE191_ROOT:-$repo/output/stage191/grid_writeback_v1}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE211_GPU:-0}" --workers "${STAGE211_WORKERS:-2}" --output-dir "$run" "$@"
bundle="$run/stage211_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
files=(stage211_decision.json manifest.json stage211_events.csv stage211_source_support.csv stage211_selection.csv
  stage211_cells.csv stage211_event_cells.csv stage211_summary.csv stage211_contrasts.csv stage211_endpoint_replay.csv
  stage211_model_scope.csv stage211_execution.csv)
{
  for name in "${files[@]}"; do
    printf '\n===== %s =====\n' "$name"
    cat -- "$run/$name"
  done
} > "$bundle"
printf '[Stage-209] result bundle: %s\n' "$bundle"
