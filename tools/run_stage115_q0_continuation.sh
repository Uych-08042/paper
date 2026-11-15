#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage76="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
stage89="${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}"
stage93="${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}"
stage112="${STAGE112_ROOT:-$repo/output/stage112/nested_shared_scoring_budget_v1}"
stage113="${STAGE113_ROOT:-$repo/output/stage113/frozen_generated_shared_closed_loop_v1}"
stage114="${STAGE114_ROOT:-$repo/output/stage114/cached_generated_failure_chain_v1}"
run="${STAGE115_ROOT:-$repo/output/stage115/matched_one_q0_continuation_v1}"
checkpoint="${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
case "${1:-run}" in
  run)
    python -B tools/stage115_q0_continuation.py --stage76 "$stage76" --stage89 "$stage89" --stage93 "$stage93" \
      --stage112 "$stage112" --stage113 "$stage113" --stage114 "$stage114" --output_dir "$run" \
      --checkpoint "$checkpoint" --gpu "${STAGE115_GPU:-0}"
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage115_q0_continuation.py --check-results --output_dir "$run"
bundle="$run/stage115_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json sampling_scope.csv anchors.csv scope.csv branches.csv contrasts.csv summary.csv paired.csv bridge.csv; do
    printf '\n===== stage115_%s =====\n' "$file"
    cat -- "$run/analysis/stage115_$file"
  done
} > "$bundle"
printf 'Stage-115 result bundle: %s\n' "$bundle"
