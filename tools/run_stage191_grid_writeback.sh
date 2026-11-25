#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE191_ROOT:-$repo/output/stage191/grid_writeback_v1}"

python -u -B tools/stage191_grid_writeback.py \
  --stage188-root "${STAGE188_ROOT:-$repo/output/stage188/temporal_support_v1}" \
  --stage189-root "${STAGE189_ROOT:-$repo/output/stage189/continuation_audit_v1}" \
  --stage190-root "${STAGE190_ROOT:-$repo/output/stage190/successor_readout_v1}" \
  --dataset-root "${VISDRONE_SOT_ROOT:-$HOME/workspace/datasets/VisDrone-SOT/VisDrone2019-SOT-train}" \
  --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
  --gpu "${STAGE191_GPU:-0}" --output-dir "$run"

bundle="$run/stage191_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json anchors.csv paired.csv branches.csv events.csv frames.csv; do
    printf '\n===== stage191_%s =====\n' "$file"
    cat -- "$run/stage191_$file"
  done
} > "$bundle"
printf '[Stage-191] result bundle: %s\n' "$bundle"
