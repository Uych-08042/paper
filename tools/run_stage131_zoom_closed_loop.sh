#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
run="${STAGE131_ROOT:-$repo/output/stage131/q3_zoom_closed_loop_v1}"
case "${1:-run}" in
  run)
    python -B tools/stage131_zoom_closed_loop.py \
      --stage76 "${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}" \
      --stage89 "${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}" \
      --stage93 "${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}" \
      --stage112 "${STAGE112_ROOT:-$repo/output/stage112/nested_shared_scoring_budget_v1}" \
      --stage121 "${STAGE121_ROOT:-$repo/output/stage121/alternate_search_state_v1}" \
      --stage130 "${STAGE130_ROOT:-$repo/output/stage130/preserve_q3_single_zoom_v1}" \
      --checkpoint "${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}" \
      --gpu "${STAGE131_GPU:-0}" --output-dir "$run"
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage131_zoom_closed_loop.py --check-results --output-dir "$run"
cat -- "$run/analysis/stage131_policy_summary.csv"
bundle="$run/stage131_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json policy_summary.csv paired.csv seed_video.csv scope.csv recovery.csv; do
    printf '\n===== stage131_%s =====\n' "$file"
    cat -- "$run/analysis/stage131_$file"
  done
} > "$bundle"
printf 'Stage-131 result bundle: %s\n' "$bundle"
