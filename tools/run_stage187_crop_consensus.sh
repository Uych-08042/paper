#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE187_ROOT:-$repo/output/stage187/crop_consensus_v1}"

python -u -B tools/stage187_crop_consensus.py \
  --stage186-root "${STAGE186_ROOT:-$repo/output/stage186/crop_quantization_v1}" \
  --output-dir "$run"

bundle="$run/stage187_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json cases.csv candidates.csv support.csv choices.csv summary.csv paired.csv; do
    printf '\n===== stage187_%s =====\n' "$file"
    cat -- "$run/stage187_$file"
  done
} > "$bundle"
printf '[Stage-187] result bundle: %s\n' "$bundle"
