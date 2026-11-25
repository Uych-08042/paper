#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE194_ROOT:-$repo/output/stage194/module_ablation_v1}"

python -u -B tools/stage194_module_ablation.py \
  --stage76-root "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89-root "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93-root "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage121-root "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
  --stage170-root "${STAGE170_ROOT:-$repo/output/stage170/full_video_cooldown_v1}" \
  --stage188-root "${STAGE188_ROOT:-$repo/output/stage188/temporal_support_v1}" \
  --stage191-root "${STAGE191_ROOT:-$repo/output/stage191/grid_writeback_v1}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE194_GPU:-0}" --output-dir "$run" "$@"

# Prepare-only is deliberately cheap and does not produce a result bundle.
if [[ ! -f "$run/stage194_decision.json" ]]; then
  exit 0
fi
bundle="$run/stage194_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json policy_summary.csv paired.csv factor_summary.csv scope_summary.csv \
      smoke_summary.csv fit_support.csv fixed_layers.csv seed_video_summary.csv readout_summary.csv routing.csv sampling.csv; do
    printf '\n===== stage194_%s =====\n' "$file"
    cat -- "$run/stage194_$file"
  done
} > "$bundle"
printf '[Stage-194] result bundle: %s\n' "$bundle"
