#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
stage76="${STAGE76_ROOT:-$repo/output/stage76/matched_four_candidate_horizon_v1}"
stage89="${STAGE89_ROOT:-$repo/output/stage89/conflict_sampled_write_support_v1}"
stage93="${STAGE93_ROOT:-$repo/output/stage93/appearance_closed_loop_v1}"
stage103="${STAGE103_ROOT:-$repo/output/stage103/four_candidate_verification_v1}"
stage106="${STAGE106_ROOT:-$repo/output/stage106/cross_fitted_commit_gate_v1}"
stage108="${STAGE108_ROOT:-$repo/output/stage108/relational_mode_commit_gate_v1}"
run="${STAGE109_ROOT:-$repo/output/stage109/gt_free_sampled_commit_branches_v1}"
checkpoint="${CHECKPOINT:-$repo/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
case "${1:-run}" in
  run)
    python -B tools/stage109_sampled_commit.py --stage76 "$stage76" --stage89 "$stage89" \
      --stage93 "$stage93" --stage103 "$stage103" --stage106 "$stage106" --stage108 "$stage108" \
      --output_dir "$run" --checkpoint "$checkpoint" --gpu "${STAGE109_GPU:-0}"
    ;;
  --pack-only) ;;
  *) printf 'Usage: %s [--pack-only]\n' "$0" >&2; exit 2 ;;
esac
python -B tools/stage109_sampled_commit.py --check-results --output_dir "$run"
bundle="$run/stage109_results_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
{
  for file in decision.json sampling_scope.csv scope.csv branches.csv samples.csv state_predictions.csv \
    sample_decisions.csv policy_summary.csv paired.csv; do
    printf '\n===== stage109_%s =====\n' "$file"
    cat -- "$run/analysis/stage109_$file"
  done
} > "$bundle"
printf 'Stage-109 result bundle: %s\n' "$bundle"
