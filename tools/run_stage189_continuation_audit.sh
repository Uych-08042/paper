#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
run="${STAGE189_ROOT:-$repo/output/stage189/continuation_audit_v1}"

python -u -B tools/stage189_continuation_audit.py \
  --stage188-root "${STAGE188_ROOT:-$repo/output/stage188/temporal_support_v1}" \
  --dataset-root "${VISDRONE_SOT_ROOT:-$HOME/workspace/datasets/VisDrone-SOT/VisDrone2019-SOT-train}" \
  --output-dir "$run"

bundle="$run/stage189_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json native_events.csv branches.csv branch_events.csv frames.csv; do
    printf '\n===== stage189_%s =====\n' "$file"
    cat -- "$run/stage189_$file"
  done
} > "$bundle"
printf '[Stage-189] result bundle: %s\n' "$bundle"
