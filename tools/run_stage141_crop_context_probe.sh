#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE141_ROOT:-$repo/output/stage141/crop_context_probe_v1}"
python -B tools/stage141_crop_context_probe.py \
  --stage76 "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
  --stage89 "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
  --stage93 "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
  --stage140 "${STAGE140_ROOT:-$repo/output/stage140/crop_center_size_cross_v1}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE141_GPU:-0}" --output-dir "$run"
python -B tools/stage141_crop_context_probe.py --check-results --output-dir "$run"
cat -- "$run/stage141_summary.csv"
bundle="$run/stage141_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json summary.csv frames.csv effects.csv scope.csv; do
    printf '\n===== stage141_%s =====\n' "$file"
    cat -- "$run/stage141_$file"
  done
} > "$bundle"
printf 'Stage-141 result bundle: %s\n' "$bundle"
