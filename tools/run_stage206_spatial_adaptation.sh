#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE206_ROOT:-$repo/output/stage206/native_routed_spatial_adaptation_v1}"
python -u -B tools/stage206_spatial_adaptation.py \
  --stage194-root "${STAGE194_ROOT:-$repo/output/stage194/module_ablation_v1}" \
  --stage205-bundle "${STAGE205_BUNDLE:-$repo/docs/experiments/06_sgla_uav_tracking/results/stage205_results_20261126T013956Z_1605754.txt}" \
  --stage76-root "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89-root "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93-root "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage121-root "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage170-root "${STAGE170_ROOT:-$repo/output/stage170/full_video_cooldown_v1}" \
  --stage188-root "${STAGE188_ROOT:-$repo/output/stage188/temporal_support_v1}" \
  --stage191-root "${STAGE191_ROOT:-$repo/output/stage191/grid_writeback_v1}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE206_GPU:-0}" --output-dir "$run" "$@"
if [[ ! -f "$run/stage206_decision.json" ]]; then exit 0; fi
bundle="$run/stage206_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for name in stage206_decision.json manifest.json stage206_policy_summary.csv stage206_paired.csv \
      stage206_scope_summary.csv stage206_training_history.csv stage206_smoke_summary.csv stage206_reference_audit.csv \
      stage206_readout_video.csv stage206_seed_video_summary.csv stage206_routing.csv; do
    printf '\n===== %s =====\n' "$name"
    cat -- "$run/$name"
  done
} > "$bundle"
printf '[Stage-206] result bundle: %s\n' "$bundle"
