# Stage-23: Same-State Branch Credit for Accuracy-Oriented PPO

## Question and Scope

The human-selected objective is accuracy and long-term decisions with PPO,
not device-pressure scheduling. Stage-22 found no meaningful 30-Hz deadline
pressure on the L20. Preserve that result; do not lower the deadline, insert
tracker sleeps, or increase background load to manufacture a motivation.

Stage-18's ordinary PPO did not beat SGLA reliably. The new hypothesis is that
same-state alternative-action outcomes can improve policy credit assignment
relative to sampled-trajectory GAE, and that a delayed outcome window adds
useful information beyond immediate reward. Neither proposition is assumed
true. This implementation is a bounded, real optimizer smoke, not another
oracle-only ranking screen and not yet a method-superiority experiment.

## Matched Arms

| Arm | PPO sampled-trajectory objective | Additional action credit |
| --- | --- | --- |
| `ppo_gae` | GAE PPO | None; branch data collected and auxiliary evaluated with zero weight |
| `ppo_cf_h1` | Same GAE PPO | Immediate same-state action reward |
| `ppo_cf_h20` (primary) | Same GAE PPO | Unbootstrapped discounted H20 action return |

All three start from identical seed-specific actor, critic and observation
encoder parameters. They share the frozen Stage-18 tracking representation,
backbone, head, seven actions and causal P30 budget. No old PPO policy is
loaded. Actions remain Fast or exactly one independent block L7-L12, not a
depth/early-exit chain. L9 is neither a preferred action nor a privileged
training target. Do not require layer diversity to manufacture a positive
result; a fixed-layer collapse would instead limit the learned-policy claim.

Use the original five frozen fit-pilot videos only. No tune video is loaded,
no original validation source is loaded, and no checkpoint is selected by
tracking outcomes. OTB and UAV123 remain unavailable for method selection.

## Branch Definition

At four causally spaced eligible states in each 256-frame live rollout:

1. Cache the actual before-state, old action probabilities and after-state.
2. Recreate the before-state for each of the seven first actions. All seven
   must be legal with positive old-policy support at this anchor.
3. Execute the first action, then follow the unchanged old policy for another
   19 frames. Do this twice, using common uniform draws across the seven
   actions within each repeat. These draws never advance live sampling RNG.
4. Query GT only after actions, for IoU reward. Crops, memory writes, boxes
   and each branch's token budget evolve from that branch's predictions.
   Do not reset any branch to GT, borrow compute credit, or use future GT as
   an observation. Branches do not leave the source clip.
5. Verify that the chosen first-action branch reproduces the real transition.
   Restore the live after-state, actor transient fields and all global RNG
   states even if a branch raises. Snapshot equality checks cover boxes,
   crop-driving state, memory, frame ID, previous action/confidence and budget.

The branch targets are finite-window estimates under the current old policy:

```text
Q_H(s,a) = average_repeat sum(t=0..H-1) gamma**t * IoU(branch_t)
A_H(s,a) = Q_H(s,a) - sum_b pi_old(b|s) * Q_H(s,b)
```

`gamma=0.97`. No critic bootstrap enters these branch targets. The two-repeat
Monte Carlo estimate is noisy; the H20 window is not an infinite-horizon Q
or a stability guarantee. All-action advantages are normalized by their
old-probability-weighted RMS over valid anchors. An anchor must have a valid
initial label and at least 80% valid labels in H20. Invalid labels get zero
reward; they never cause a tracking reset. The same validity rule applies to
all arms. Eligibility/spacing itself never uses GT visibility or reward.

The auxiliary is an old-probability-weighted clipped all-action surrogate,
with fixed weight 0.5 in both experimental arms. Ordinary PPO uses weight 0
but runs the same branch and auxiliary-forward workload. This is a training
auxiliary; it is not a claim of a new policy-gradient theorem. Standard PPO
retains its GAE critic bootstrap at artificial clip and rollout boundaries.

## Frozen Smoke Contract

- Seed: 2074; fresh policy initialization for each arm.
- 1,024 live transitions/arm; four rollouts of 256 transitions.
- Four anchors/rollout, seven first actions, two continuations, H20.
- 1,120 branch transitions/rollout, 4,480/arm, 5,504 live+branch/arm.
- Total over three arms: 16,512 tracking transitions, not 3,072.
- All arms simulate the entire H20, including the H1 and ordinary-PPO arms.
- Adam lr 0.0003, epsilon 0.00001; GAE lambda 0.95; four full-rollout update
  epochs; clip 0.2; value weight 0.5; entropy weight 0.01; gradient cap 0.5.
- Exactly 16 optimizer steps/arm. All arms evaluate the same auxiliary actor
  gradient diagnostic before the joint backward pass.
- P30 is a maximum available candidate budget, not a forced candidate rate.
  Every trajectory prefix satisfies calls <= floor(0.3 * frames).
- Value-only bootstrap prefix calls, initial template processing, CPU image
  reads, checkpointing and gradient diagnostics are additional work beyond
  the reported tracking-transition count. Wall time and peak GPU allocation
  are recorded; do not call this training-cost-neutral or faster inference.
- Different trained arms will visit different states. Match quotas and
  initialization, not outcomes or trajectories after policies diverge.
- Matching transition/update/branch quotas does not make realized candidate
  FLOPs or wall time identical. Candidate usage is policy-dependent under
  the same P30 cap. Report both live and branch candidate-block counts and
  value-only bootstrap prefix calls alongside wall time; do not label these
  runs exactly equal-FLOP experiments.

`PASS_BRANCH_PPO_SMOKE` requires three completed matched arms, unchanged
frozen weights, changed actors, nonzero branch credit/actor gradient, finite
updates, old-likelihood replay <= 1e-4, correct chosen-action replay, and
exact branch/optimizer counts. A low fit IoU is not a smoke failure; a high
fit IoU is not evidence of generalization. No AUC efficacy gate exists here.

## Run on the GPU Server

Sync the three new `tracking/stage23_*.py` files, the matching
`tracking/test_stage23_branch_credit.py`, and this document. The existing
`stage21_counterfactual_math.py` and `stage21_counterfactual_runtime.py`
must also be present. Stage-23 reuses their tested math and coupled sampling
environment, not the unfinished Stage-21 collector.

```bash
export STAGE18_ROOT="$PWD/output/stage18/temporal_layer_policy_v1"
export STAGE18_REP_BUNDLE="$STAGE18_ROOT/frozen_representation_v1.json"
export INTERVENTION_MANIFEST="$PWD/output/stage8/l9_intervention_v1/intervention_manifest.json"
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
export STAGE23_ROOT="$PWD/output/stage23/onpolicy_branch_credit_v1"

python tracking/test_stage23_branch_credit.py
python -m unittest discover -s tracking -p 'test_stage18*.py'

python tracking/stage23_branch_ppo.py \
  --bundle "${STAGE18_REP_BUNDLE:?Set frozen representation bundle}" \
  --manifest "${INTERVENTION_MANIFEST:?Set development manifest}" \
  --checkpoint "${CHECKPOINT:?Set original checkpoint}" \
  --parameter deit_distilled --gpu 0 --seed 2074 \
  --output_dir "$STAGE23_ROOT/smoke_v1"

python tracking/stage23_branch_status.py --input "$STAGE23_ROOT/smoke_v1"
cat "$STAGE23_ROOT/smoke_v1/stage23_decision.json"
```

The server test run must execute, not skip, the PyTorch tests. Local Windows
NumPy-only tests do not establish real optimizer, tracker or CUDA correctness.
The historical Stage-18 frozen representation's timing target failed
(memory schedule/SGLA = 1.1092); no speed claim is restored by this smoke.

## Interruptions and Resume

Each arm atomically saves `last.pt` after every completed rollout/update.
Ctrl+C loses at most the in-flight rollout. Resume with exactly the same
command, output directory, seed, code and frozen input artifacts. Completed
arms are verified and skipped; an incomplete arm restores model, optimizer,
source queue, active episode, boxes, memory, budget and RNG. Reports are
reconciled from the authoritative checkpoint. There is no outcome-selected
`best.pt`.

The root writer lock prevents two simultaneous trainers. A normal exit or
Ctrl+C releases it. After a hard process kill, first confirm that no trainer
is running; only then remove that output directory's `.stage23_writer.lock`
and resume. Do not remove a live writer's lock. Do not delete model caches or
edit frozen contracts to bypass a fingerprint error. The CPU status command
reads atomic CSVs, which can lag the checkpoint by one publication if killed.

## After Smoke, Not Before

Review integrity and actual wall time before a larger preregistered run. A
reasonable next development comparison is all three matched arms on the
96 frozen fit videos with three paired seeds, a fixed step budget and a
fixed evaluation schedule on the 24 tune videos. The primary contrast must
be H20 minus ordinary PPO; H20 minus H1 tests the long-horizon explanation.
Do not report the best seed. Include same-representation static controls,
memory-Fast, original Fast and SGLA in the eventual closed-loop evaluation.
Independent videos, not branch frames, are the statistical units. The old
tune split is already development data, not a new external confirmation.

If H20 fails to beat immediate credit and ordinary PPO under matched training
quotas (with realized costs audited), reject this credit-assignment explanation instead of changing H,
gamma, seed, continuation count or thresholds on the same outcomes. Any new
efficacy or generalization claim still needs an untouched external benchmark.

## Prior Art and Claim Boundary

PPO clipping, expected-action gradients and counterfactual baselines are not
new. This experiment tests their tracking-specific integration with exact
same-state crop/memory/budget branches and old-policy continuation. It does
not establish that integration is publishably novel or successful.

- PPO: https://arxiv.org/abs/1707.06347
- Expected Policy Gradients (JMLR 2020): https://www.jmlr.org/papers/v21/18-012.html
- COMA: https://arxiv.org/abs/1705.08926 (counterfactual-baseline prior art;
  this single-tracker experiment is not a multi-agent COMA implementation).
