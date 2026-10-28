# Stage-18: Long-Term Layer Policy With Persistent Target Memory

Design date: 2026-10-20.
Status: design specification; no new model has been trained or evaluated.
Working identifier: `stage18_temporal_layer_policy_v1`.

## Research question

Can a layer choice improve future tracking by updating a target memory that
subsequent frames actually use, while staying within a fixed compute budget?

The proposed contribution is the coupling of candidate-layer execution,
action-conditioned target-memory updates, and policy optimization on the
resulting tracking trajectories. PPO, recurrent memory, residual adapters,
and dynamic routing are established building blocks, not individual novelty
claims. This is a testable method hypothesis; literature review and successful
ablations are needed before claiming novelty or effectiveness.

## Evidence and implementation boundary

- Stage-14 found hindsight multi-action headroom, but did not learn a useful
  policy from the frozen states.
- Stage-16B return prediction passed an offline screen; Stage-16C lost to
  both SGLA and an equal-cost fixed-layer schedule in real tracking.
- Stage-17 learned immediate residual correlations, but lost to equal-cost
  L9 at all three tested budgets. Its P30 layer choice also lost to L9 at the
  very same selected states.
- These results motivate changing the transition mechanism and trainable
  representation. They do not prove that every layer policy is unlearnable.

`tracking/stage4_runtime.py` executes a shared L1-L6 prefix and, optionally,
one independent block from L7-L12 on that prefix. Selecting L12 does not
execute L7-L11. Stage-18 preserves this action meaning to avoid conflating
layer identity with sequential depth or cost.

## Proposed model

The initial template, crop construction, and original SGLA weights initialize
the tracker. Add a target-memory module and one residual adapter per candidate
block. The memory changes the features consumed by the tracking head, as well
as the observations used by the actor.

For tracked frame t:

1. Crop using the executed prediction from frame t-1 and compute prefix
   tokens `F_t = Prefix6(template, crop_t)`.
2. Encode the initial-template relation and spatial search tokens using all
   channels. Read the previous target memory `M_(t-1)` through a small
   cross-attention residual, giving `F'_t`.
3. Before executing any candidate or current-frame tracking head, the actor
   observes pooled prefix features, memory, previous executed confidence,
   previous action, predicted box motion, and current compute credit.
4. Choose `a_t` from `{Fast, L7, L8, L9, L10, L11, L12}`. An unavailable
   compute credit masks all six candidate logits before sampling or argmax.
5. Fast uses `F'_t`. A candidate k uses
   `Y_t = B_k(F'_t) + Adapter_k(B_k(F'_t))`, with the block evaluated once.
   Apply the original normalization and shared tracking head once.
6. Decode the selected output to obtain the executed box `b_t`. Pool target
   features from `Y_t` inside the predicted box, not the ground-truth box.
7. Update memory using the selected target features, action embedding, and
   executed confidence. Write `b_t` to the next crop state.

For Fast, use `Y_t = F'_t` in steps 6 and 7. Initial-template tokens remain
an appearance anchor; memory stores adaptations inferred from predictions.

Initial module sizes:

| Component | Specification |
| --- | --- |
| Search representation | 8x8 pooled spatial tokens, original embedding channels |
| Target memory | 4 slots x 64 channels |
| Memory read | Single attention read, 64-channel projection, residual output |
| Memory update | Shared gated recurrent update, conditioned on action embedding |
| Candidate adapters | Six token-wise residual bottlenecks, rank 16 |
| Actor and critic | Shared 128-channel observation encoder; separate output heads |
| Actor outputs | Seven logits |
| Current tracking heads per deployed frame | Exactly one |

Residual output projections start at zero. A diagnostic bypass must recover
the original seven execution paths before learning. Memory and adapters are
new trainable tracking components, not features cached from old frozen runs.

There is no inference-time oracle, all-candidate scoring, ground-truth
feedback, or extra Fast-head pass before a candidate decision.

## Long-term objective and budget

Use an actor-critic policy on its own executed trajectories:

`maximize E_pi[sum_t gamma^t * IoU(b_t, gt_t)]`, with `gamma = 0.97`.

Ground truth provides training rewards and supervised losses only. It is
absent from actor observations, memory updates, crop updates, and deployment.
Per-frame IoU is deliberately accumulated over future states; do not replace
it with a same-state candidate-minus-Fast immediate label.

Primary compute budget: 30% candidate-block calls. Use integer token credit:

`credit = min(20, credit + 3)` before each tracked-frame decision;
`candidate is legal iff credit >= 10`; candidate execution subtracts 10.

Start every episode with zero credit, including after initialization. This
allows a two-call burst capacity, does not require knowing video length,
and guarantees `candidate_calls(T) <= floor(3*T/10)`. Credit is a causal
observation. Fast remains legal and unused credit need not be spent.

The block budget is a resource proxy, not a latency measurement. End-to-end
latency includes memory read/write, actor inference, crop/preprocessing,
candidate execution, one head, and decode. Report image loading separately.
Do not use a path-mixture estimate as the new method's measured speed.

## Training and data

### Data roles

Reuse the original 120 Stage-16 training-video identities as development
data. Split by physical video into 96 fit / 24 tune with seed 2053. Every
clip, rollout, and branch from one video stays in the same group.

The 40 previously consumed Stage-16 validation videos, UAV123, OTB, and all
earlier inspected holdouts cannot supply a new confirmatory claim. Stage-18
does not use them for checkpoint or design selection.

Before training, audit actual installed frame/annotation coverage. Training
clips contain 64-128 tracked frames after one GT initialization; videos too
short for 64 tracked frames are excluded and recorded before fitting.
Partition the remaining eligible videos with the same 80/20 video rule and
record actual counts if the nominal 96/24 split is not feasible. No silent
shortening of the temporal design is permitted. Development evaluation uses
one predeclared contiguous clip per tune video, up to 256 tracked frames,
with at least 64. Full external videos supply the eventual long-video claim.

Old Stage-16/Stage-14 NPZ files are diagnostic records only. New trainable
memory/adapters change both features and transitions, so old action labels
and old next states cannot train the new policy as if they were current.

### Phase A: tracking representation warm-up

Freeze original prefix, candidate blocks, normalization, and box head. Train
the new memory and adapters with `2 * GIoU + 5 * L1 + 1 * center_focal`,
matching the existing tracking-loss weights. Do not include the original
selector-similarity loss: the original selector is not the Stage-18 actor.
Record box coordinate conventions, valid-label masks, and loss weights in
the runtime training contract before launch. Tracking failures do not trigger
GT crop resets or removal of difficult transitions.

Use predicted crops after the first frame and real memory updates. Sample
Fast with probability 0.5 and each candidate with probability 1/12 during
this representation-only phase. Candidate identities have equal probability;
no layer is assigned a desired appearance or motion specialization.

Initial budget: 10 epochs, two clips per batch, truncated backpropagation
through 32 consecutive frames, optimizer AdamW at 1e-4. Detach crop
coordinates at frame transitions; preserve gradients through recurrent
memory within each 32-frame chunk. Select the representation checkpoint
using predeclared tune tracking loss under the same fixed behavior seed.

Frozen blocks and head must remain differentiable with respect to their
inputs. Setting their parameters to `requires_grad=False` is compatible with
this; wrapping memory-conditioned block/head execution in `no_grad()` is not.
Freeze normalization running statistics during training.

### Phase B: on-policy temporal decision learning

Freeze the Phase-A tracking components, including memory update weights.
Train the actor and critic using PPO and newly executed tracking episodes.
The memory state continues to evolve according to the actual selected
outputs. The input memory can be stored for PPO updates because its weights
remain fixed during this phase. Do not reuse stale policies' rollouts as
current on-policy data or substitute another policy's future trajectory.

Predeclared starting settings:

| Setting | Value |
| --- | --- |
| Independent training seeds | 2054, 2055, 2056 |
| Maximum environment transitions | 250,000 per seed |
| Discount | 0.97 |
| GAE lambda | 0.95 |
| PPO clipping | 0.2 |
| Optimizer learning rate | 3e-4 |
| Rollout transitions per update | 2,048, reset memory only on actual episode reset |
| Minibatch size | 256 |
| Update epochs per rollout | 4 |
| Entropy coefficient | 0.01 |
| Value coefficient | 0.5 |
| Gradient norm limit | 0.5 |
| Primary candidate budget | 30%, integer token rule above |

Initialize actor probabilities to Fast=0.7, each candidate=0.05 before the
credit mask. Deployment uses masked argmax with a fixed tie rule. Evaluate
that exact deployment rule during checkpoint selection. Bootstrap artificial
rollout truncations from the critic; true video terminals have zero bootstrap.

Evaluate tune AUC every 20,000 environment steps and at the final step.
Choose each seed's checkpoint by tune AUC subject to the exact budget.
Tie-break by lower measured model latency, then earlier checkpoint. No seed
is selected as the official winner: report all three and their per-video mean.
Do not deploy a three-model ensemble without including its measured overhead.

The update counts are a fixed development compute envelope. Performance
failure does not authorize silently increasing it. Pilot wall-time and peak
memory measurements are required before launching all seeds and ablations.

## Controls and claims

All proposed-architecture controls use the same Phase-A tracking weights,
same predicted-crop protocol, and same credit rule. Only routing varies,
except for the explicitly retrained no-memory ablation.

| Control | Purpose |
| --- | --- |
| Original Fast | Original low-compute reference |
| Original complete SGLA | Main existing-method reference |
| New architecture, always Fast | Does target memory itself improve tracking? |
| New architecture, uniform L7 ... uniform L12 | Six periodic single-layer controls at P30 |
| Best static control | Tune-selected best among those six and new always-Fast; freeze before external testing |
| New architecture, uniform-time random layer | Same P30 credit rule; candidates uniformly sampled from six layers |
| Full architecture, gamma=0 actor | Same capacity and PPO compute, immediate-only optimization |
| No-memory long-horizon policy | Retrain the new modules/policy with memory disabled; same data and compute |
| Full proposed long-horizon policy | Main proposed system |

The static baseline executes its fixed layer at the first available credit
and otherwise Fast. The randomized control uses the same times and chooses
a layer with a fixed recorded random seed. No method evaluates all six
candidates in the deployed path.

Action diversity is descriptive. Do not require a minimum number of active
layers for efficacy. If one layer is sufficient, the policy may use one
layer; then the experiment must establish a timing or memory contribution.

The best static layer is selected only on development tune data. Report all
six external results for transparency, without promoting the external winner
to a newly selected baseline or policy.

## Execution order and decisions

1. **Stage-18A, implementation pilot:** use five fit videos chosen by a fixed
   hash order. Verify all seven bypass paths, coordinate transforms, temporal
   causality, memory reset/clone, finite memory/adapter gradients, exact
   credit reconstruction, and one-head inference. Measure wall time, peak
   GPU memory, and untrained module overhead. This stage makes no efficacy
   decision from five videos.
2. **Stage-18B, development training:** run the two phases and the required
   ablations within the fixed compute envelope. Evaluate real closed-loop
   tune trajectories. A correlation or offline oracle cannot advance a model.
3. **Stage-18C, frozen development decision:** compare paired per-video AUCs
   after averaging the three seeds for each video. Advance this version only
   if the full policy beats original SGLA and the tune-selected best static
   control with paired video-bootstrap 95% lower bounds above zero. Report
   gamma=0 and no-memory contrasts to decide whether the stated mechanism
   is supported. If the main contrasts fail, stop this version before an
   external run, preserving partial findings.
4. **Stage-18D, external confirmation:** freeze weights, all controls, seeds,
   crop/memory rules, metrics, and profiling code before a single complete
   evaluation on a new benchmark. Use the exact external protocol below.

The tune videos have already served development in earlier experiments.
Even a passed development decision is not independent confirmation.

## External protocol

Proposed primary benchmark: the NfS 30-FPS protocol. Confirm acquisition,
annotation coverage, physical-video overlap, and evaluation access before
starting the full development training budget. Do not inspect tracker
outcomes on it during implementation. If unavailable, amend the named
benchmark before training; never choose a replacement from policy outcomes.

Run complete videos, initialize from first-frame GT once, and use no GT
resets. Record annotation validity separately from policy state transitions.
All methods receive identical frame and annotation conventions. Check for
cross-benchmark duplicate physical videos before declaring independence.

Primary accuracy contrasts are full policy minus original SGLA and full
policy minus the frozen best static control. Use 5,000 paired physical-video
bootstrap draws, analysis seed 2057. Average seeds per video first; frames,
seeds, overlapping clips, and multiple targets from a physical video are not
independent bootstrap units.

For the intended accuracy-and-efficiency result, require both accuracy
contrasts' 95% lower bounds above zero and a paired measured mean end-to-end
latency ratio (full / original SGLA) no greater than 1.0. Also report the
latency ratio CI, p50/p95, peak GPU memory, and candidate usage. Benchmark
latency in five warmed, interleaved method-order repeats on one device with
no concurrent GPU workloads. Training-set profiling fixes the timing scope;
external timing does not select a new policy.

Mechanism claims require full minus gamma=0 and full minus no-memory
contrasts to have 95% lower bounds above zero. These are secondary tests;
apply Holm correction to the two positive-effect tests, separately from the
two jointly required primary accuracy contrasts. Report failed claims too.

Interpret outcomes separately:

- Better than original SGLA, but not best static: representation improvement
  may be useful; adaptive routing is unestablished.
- Better than best static, but not gamma=0: adaptive routing may help;
  long-term credit is unestablished.
- Better accuracy with higher latency: an accuracy result, not a speed result.
- Main accuracy contrasts fail: reject this policy version within the stated
  compute envelope; do not search another rate/layer/threshold on external data.

## Artifacts and immediate implementation deliverable

This document is the method/experiment design, not a completed data freeze.
The Stage-18A memory-aware model, live environment, pilot, and gradient tests
are now implemented in new `tracking/stage18_*.py` modules. See
`STAGE18A_EXECUTION_PILOT.md` for the executable pilot command. Supervised
warm-up and PPO training are subsequent implementation deliverables. Existing
Stage-16/Stage-17 source and frozen artifacts are not edited.

Before full training, publish a runtime contract containing actual eligible
video IDs, split, supervised loss definitions/weights, checkpoint hash,
module dimensions, code hashes, optimizer settings, resource audit, and the
named untouched external manifest. New trajectory files must bind both
tracking-model and actor versions. Never overwrite earlier frozen caches.

## Primary references and novelty boundary

- [SGLATrack, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Xue_Similarity-Guided_Layer-Adaptive_Vision_Transformer_for_UAV_Tracking_CVPR_2025_paper.pdf): source layer-adaptive tracking framework.
- [SkipNet, ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Xin_Wang_SkipNet_Learning_Dynamic_ECCV_2018_paper.html): dynamic routing with supervised learning and RL already exists.
- [ADNet, CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Yun_Action-Decision_Networks_for_CVPR_2017_paper.html): RL-based visual tracking already exists.
- [TCTrack, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Cao_TCTrack_Temporal_Contexts_for_Aerial_Tracking_CVPR_2022_paper.html): temporal feature and similarity-map modeling already exists.
- [PPO](https://arxiv.org/abs/1707.06347): established policy-gradient optimizer used here as an implementation choice.
- [LGTrack, 2026 preprint](https://arxiv.org/abs/2602.13636): nearby layer-adaptive feature-enhancement work that must be differentiated in a full related-work review.
- [NfS, ICCV 2017](https://arxiv.org/abs/1703.05884): proposed independent benchmark, subject to the explicit data-access audit.

These references motivate and delimit the proposal. None establishes that
the proposed coupling is new or that it will improve this tracker.
