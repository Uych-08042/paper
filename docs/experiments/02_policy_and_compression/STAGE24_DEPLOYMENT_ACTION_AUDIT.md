# Stage-24: Read-Only PPO Deployment Action Audit

## Purpose

Stage-23B completed all nine final policies on the frozen 95-fit / 24-tune
split and returned `NO_CREDIT_EFFECT_ESTABLISHED`. H20 did not clear the
predeclared ordinary-PPO and immediate-credit comparisons. Preserve that
decision. This post-hoc descriptive audit is not a new accuracy experiment,
another PPO training round, or permission to select new hyperparameters on
the same tune outcomes.

The reported summary numbers suggest that H1 may execute exactly the static
L9-P30 schedule and that H20 seed2074 may be memory-Fast. Equal AUC and cost
do not prove action or trajectory equivalence. This audit reads the stored
actions and boxes to check these suggestions directly.

## No New GPU Work or Frozen Edits

Only sync the following new files:

- `tracking/stage24_deployment_audit.py`
- `tracking/test_stage24_deployment_audit.py`
- this document

Do not edit or replace any Stage-18/23 source, checkpoint, contract, evaluation
JSON, or original decision. The script requires NumPy and the existing
dependency-light helpers, not PyTorch/CUDA or the original image dataset.
It hashes the nine saved `.pt` files as opaque bytes; it never unpickles them,
loads a model, generates predictions, or reruns tracking.

The completed Stage-23B directory, its original `analysis/stage23b_decision.json`,
all referenced per-video predictions, checkpoint files and histories must be
present. The original source-code hashes, run fingerprints, checkpoint hashes,
evaluation references, training quotas and shared frame/visibility coverage
are verified using the frozen Stage-23B loader. Missing or changed artifacts
are errors, not reasons to bypass provenance checks.

Source-file hashes are compared before and after analysis. Reports go to a
separate directory; an output inside or above the frozen input is rejected.
An active source writer lock is also rejected. Wait for the training process
to exit. After a hard kill, confirm no writer is running before handling a
stale source lock; never delete checkpoints to make the audit run.

## Comparisons and Definitions

Every final policy is compared on the same frames to memory-Fast, each static
L7-L12 P30 control, the random-layer control, original Fast and original SGLA.
Only the memory/static/random controls share the policy's frozen tracking
representation. Original Fast/SGLA are output references, not candidates for
an action-equivalence classification. SGLA's stored `-1` sentinel does not
identify its selected candidate layer and must not be relabeled as L7.

- Exact integer action equality is checked on all tracked frames, including
  invisible frames because those still advance state and budget.
- Candidate-call timing equality is separate from candidate-layer equality
  at jointly called frames. One layer with nonstatic call timing is not a
  fixed-schedule collapse.
- Box equality is reported both as exact stored floats and within a fixed
  absolute 1e-6 coordinate tolerance. Valid-frame IoU tolerance is 1e-12.
  Tolerances are recorded, not tuned after looking at outcomes.
- Trajectory equivalence requires identical actions, boxes within tolerance
  and matching valid-frame IoUs on these cached videos. It does not imply
  functional equivalence on all unseen videos or bitwise equality of weights.
- Candidate-layer switches count changes between successive non-Fast
  candidate events within one video. A Fast/L9/Fast/L9 schedule has zero
  candidate-layer switches. A video reset is never counted as a switch.
- Multiple layers across different videos are distinguished from multiple
  layers within a video. All raw call counts are retained, with no threshold
  chosen to force or suppress a diversity finding.
- Legal candidate opportunities are reconstructed with the original causal
  token-credit object. They depend on each policy's prior spending and must
  not be treated as independent, common counterfactual opportunities.
- Executed-layer frequency entropy is descriptive, not the stochastic
  actor's entropy. Actor probabilities/logits were not cached, so calibrated
  confidence, training-policy entropy and causal action value are unavailable.

Per-seed-pair reports compare actual trajectories and the already recorded
whole temporal-policy-module weight digests. Different weights can have
identical greedy argmax behavior. Whole-module digests include the critic and
observation encoder, so they are not actor-only comparisons. File-byte hashes
include optimizer/RNG/history state and are not policy-weight equality tests.

## Descriptive Classifications

- `MEMORY_FAST_EQUIVALENT`: no candidate calls, actions and outputs match
  memory-Fast on all audited videos.
- `STATIC_L7_P30_EQUIVALENT` through `STATIC_L12_P30_EQUIVALENT`: identical
  actions and outputs to that pre-existing static control.
- `SINGLE_LAYER_NONSTATIC_SCHEDULE`: one executed candidate layer but no
  full static-control action/output match; this is not evidence of useful
  learned timing.
- `CROSS_VIDEO_SINGLE_LAYER`: more than one layer globally but no video
  executes more than one candidate layer.
- `WITHIN_VIDEO_MULTI_LAYER`: at least one audited video switches candidate
  layer; this alone does not establish useful adaptation or improved AUC.
- `FAST_ONLY_OUTPUT_MISMATCH` / `STATIC_ACTIONS_OUTPUT_MISMATCH`: identical
  memory/static actions but output differences beyond the recorded tolerances.
  Check numerical/path provenance before interpreting these as a new method.

The audit status is `COMPLETE_DESCRIPTIVE_ACTION_AUDIT`, not a method PASS.
Same-action output discrepancies in control or seed-pair comparisons produce
`REQUIRES_OUTPUT_CONSISTENCY_REVIEW`. Neither status changes the original
Stage-23B efficacy decision. Equivalence is assessed over all frozen tune
videos; no better seed, subset, schedule, phase or layer is selected.

## Run on the Server

```bash
export STAGE23_ROOT="$PWD/output/stage23/onpolicy_branch_credit_v1"
export STAGE23B_DIR="$STAGE23_ROOT/development_v1"
export STAGE24_DIR="$PWD/output/stage24/ppo_deployment_action_audit_v1"

python tracking/test_stage24_deployment_audit.py

python tracking/stage24_deployment_audit.py \
  --input "${STAGE23B_DIR:?}" \
  --output_dir "${STAGE24_DIR:?}"

cat "$STAGE24_DIR/stage24_policy_actions.csv"
cat "$STAGE24_DIR/stage24_seed_agreement.csv"
```

No background GPU job is needed. Runtime is primarily byte hashing and JSON
parsing; no original images or GPU operations are read/executed. An identical
rerun is accepted. If audit code/inputs differ from an existing audit contract,
use a new audit output directory rather than editing its fingerprint. The
audit output uses the existing generic writer helper's `.stage23_writer.lock`
to prevent concurrent CSV publication; this lock is not in the source run.

## Reports

- `stage24_policy_actions.csv`: nine seed/method summaries, classification,
  candidate usage, actual layer counts, within-video switching, legal
  opportunities and L9 action agreement.
- `stage24_video_actions.csv`: the same descriptive action statistics for
  each policy/video, without outcome-based subgroup selection.
- `stage24_control_agreement.csv`: per-policy exact-action/timing agreement,
  box/IoU differences and output consistency against all frozen controls.
- `stage24_video_control_agreement.csv`: video-level comparison details.
- `stage24_seed_agreement.csv`: paired-seed behavior versus reported module
  and checkpoint identity.
- `stage24_audit_contract.json`: source hashes, code, tolerances and runtime.
- `stage24_action_audit.json`: descriptive findings, original decision identity
  and explicit no-training/no-new-prediction flags.

After this audit, archive the fixed Stage-23B result as reported. A genuinely
new PPO method would need a separate, mechanism-level hypothesis and protocol;
this audit cannot justify extending the budget or selecting H/gamma, reward
weights, deployment sampling rules or layers on the same tune outcomes.
