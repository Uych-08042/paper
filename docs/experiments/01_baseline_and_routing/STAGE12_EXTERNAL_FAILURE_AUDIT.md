# Stage-12 External Failure Audit

Stage-12 is a terminal, descriptive analysis of the frozen Stage-11 OTB
`FAIL_TRANSFER` result. It does not evaluate a new schedule and must not be
used to choose an OTB rate, phase, layer, threshold, or controller.

## Question

The audit asks why periodic L9 at 30% cost failed to recover the externally
confirmed SGLA gain:

- Does P30 improve the same sequence objects that benefit from SGLA?
- Does P30 reproduce SGLA's reduction in low-IoU failure frames?
- Is the missing gain concentrated in a normalized part of the trajectory?
- Is the conclusion explained by fixed sequence difficulty or length groups?

All uncertainty intervals use cluster bootstrap over the 98 OTB physical
videos. The two target objects in `Jogging` and `Skating2` remain in the same
cluster.

## Run

```bash
export STAGE11_DIR="$PWD/output/stage11/otb2015_external_v1"

python tracking/stage12_external_failure_audit.py \
  --input "$STAGE11_DIR" \
  --bootstrap_samples 5000 \
  --seed 2042
```

The default output directory is:

```text
$STAGE11_DIR/analysis/stage12_failure_audit
```

It contains sequence-level mechanism metrics, fixed subgroup summaries,
five-bin temporal contrasts, low-IoU failure summaries, and a machine-readable
mechanism JSON.

## Interpretation Boundary

Stage-12 can motivate the design requirements of a future adaptive method, but
it cannot validate that method. Any new controller must be developed without
OTB outcomes and receive final confirmation on a new untouched external
benchmark.
