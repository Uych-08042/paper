#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE202_ROOT:-$repo/output/stage202/matched_regret_margin_v1}"
python -u -B tools/stage202_regret_routing.py \
  --stage194-root "${STAGE194_ROOT:-$repo/output/stage194/module_ablation_v1}" \
  --stage200-bundle "${STAGE200_BUNDLE:-$repo/docs/experiments/06_sgla_uav_tracking/results/stage200_results_20261125T140231Z_1329509.txt}" \
  --output-dir "$run" "$@"
if [[ ! -f "$run/stage202_decision.json" ]]; then exit 0; fi
bundle="$run/stage202_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for name in stage202_decision.json manifest.json stage202_model_scope.csv stage202_fit_summary.csv stage202_source_bridge.csv \
      stage202_pair_support.csv stage202_pair_schedule.csv stage202_policy_summary.csv stage202_paired.csv stage202_error_summary.csv \
      stage202_error_video.csv stage202_policy_video.csv stage202_training_history.csv stage202_frame_predictions.csv stage202_seed_summary.csv; do
    printf '\n===== %s =====\n' "$name"
    cat -- "$run/$name"
  done
} > "$bundle"
printf '[Stage-202] result bundle: %s\n' "$bundle"
