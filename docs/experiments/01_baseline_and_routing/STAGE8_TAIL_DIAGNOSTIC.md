# Stage 8A: Closed-loop Tail and Temporal Diagnostic

Stage 7 established that sparse L9 evaluation improves on Fast in two disjoint
GOT-10k-train clip cohorts. It did not establish that placing the same budget
late is better than distributing it uniformly. This CPU-only diagnostic reuses
the frozen 50-clip and 150-clip outputs to locate gains, negative tails, and
failure persistence before collecting causal branch-intervention data.

This is post-hoc analysis. Keep the two cohorts separate for confirmatory
claims; the pooled rows are descriptive only. Do not use these holdout clips to
fit the next router or tune a threshold.

## 1. Set result directories

```bash
cd /home/u25600003160604/workspace/paper

export STAGE7_50="$PWD/output/stage7/temporal_holdout_v1"
export STAGE7_150="$PWD/output/stage7/temporal_holdout_confirm_v1"
export STAGE8_DIR="$PWD/output/stage8/tail_diagnostic_v1"
```

If the second experiment used another directory name, change `STAGE7_150` to
the directory that contains `full/fast`, `full/uniform_p5`,
`full/late_p10_exact`, and `full/sgla`.

## 2. Run the diagnostic

No GPU or checkpoint is required.

```bash
python tracking/stage8_tail_diagnostic.py \
  --cohort stage7_50 \
    "$STAGE7_50/full/fast" \
    "$STAGE7_50/full/uniform_p5" \
    "$STAGE7_50/full/late_p10_exact" \
    "$STAGE7_50/full/sgla" \
  --cohort stage7_150 \
    "$STAGE7_150/full/fast" \
    "$STAGE7_150/full/uniform_p5" \
    "$STAGE7_150/full/late_p10_exact" \
    "$STAGE7_150/full/sgla" \
  --output_dir "$STAGE8_DIR" \
  --bins 5 \
  --tail_count 10 \
  --bootstrap_samples 10000 \
  --seed 2029
```

The script checks all of the following before writing results:

- every policy contains the same sequence and frame IDs;
- `gt_valid` agrees across policies;
- the two cohorts do not reuse a sequence;
- `uniform_p5` and `late_p10_exact` spend exactly the same number of candidate
  blocks on every sequence, not merely on average.

## 3. Inspect the primary reports

```bash
cat "$STAGE8_DIR/stage8_delta_summary.csv"
cat "$STAGE8_DIR/stage8_temporal_bins.csv"
cat "$STAGE8_DIR/stage8_failure_summary.csv"
cat "$STAGE8_DIR/stage8_disagreement_summary.csv"
```

Useful targeted views:

```bash
python - <<'PY'
import csv
import os

root = os.environ["STAGE8_DIR"]

print("\nLate minus Uniform, sequence-level:")
rows = csv.DictReader(open(
    os.path.join(root, "stage8_delta_summary.csv"), encoding="utf-8"
))
for row in rows:
    if row["comparison"] == "late_minus_uniform":
        print(row)

print("\nLate minus Uniform, five temporal bins:")
rows = csv.DictReader(open(
    os.path.join(root, "stage8_temporal_bins.csv"), encoding="utf-8"
))
for row in rows:
    if row["comparison"] == "late_minus_uniform":
        print(row)
PY
```

The detailed files have these roles:

- `stage8_policy_summary.csv`: AUC, valid-frame mean IoU, and cost audit.
- `stage8_delta_summary.csv`: mean, median, 5/10% trimmed means, quantiles,
  paired bootstrap interval, and win/tie/loss counts.
- `stage8_sequence_deltas.csv`: one row per real holdout clip for later figures.
- `stage8_tail_sequences.csv`: the ten strongest positive and negative clips
  for each comparison.
- `stage8_temporal_bins.csv`: sequence-macro AUC deltas in five normalized
  temporal bins.
- `stage8_failure_summary.csv`: low-IoU frame ratios and continuous failure-run
  lengths at IoU thresholds 0.1 and 0.2.
- `stage8_disagreement_summary.csv`: how often two policies differ by at least
  0.1 IoU and when persistent disagreement first appears.

## 4. Decision rule for the next data collection

Read `stage7_50` and `stage7_150` separately first.

- If `late_minus_fast` stays positive but `late_minus_uniform` remains
  inconsistent, retain sparse L9 as the robust result and do not claim a
  universal late-placement advantage.
- If the same temporal bin is positive in both cohorts, oversample anchors from
  that normalized region in the new development set, while still sampling the
  other bins for control.
- If gains mainly shorten IoU failure runs, define the next intervention target
  over a future horizon rather than as same-frame IoU gain.
- If a small set of sequences dominates the mean while trimmed means collapse,
  use a tail-aware objective and report median/quantile effects alongside AUC.

The next GPU experiment should use new development sequences and branch each
Fast trajectory at sampled anchors: continue Fast or execute L9 once, then
measure future utility over horizons 1, 5, 10, 20, and 40. Stage-7 holdouts stay
evaluation-only.
