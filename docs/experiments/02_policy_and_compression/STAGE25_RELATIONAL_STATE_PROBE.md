# Stage-25: Equal-Cost Relational State Feasibility

## Question and Scope

Stage-24 established that all three H1 policies were exactly equivalent to
static L9-P30 on the 24 development videos. H20 produced memory-Fast, L9 with
a slightly changed schedule, or static L9-P30. Ordinary PPO changed layers,
but did not beat static L9. The original negative Stage-23B decision stands.

This experiment asks a different, limited question: can an explicit
target-versus-distractor encoding make **which layer** to execute learnable,
without mixing that question with **when** to execute a candidate?

This is a representation feasibility probe, not a new PPO training run, a
finished tracking method, or a novelty claim. Passing permits designing one
new matched PPO smoke. Failure stops this particular representation design.
Do not keep searching its settings on the existing tune videos.

Only new Stage-25 files are added. Do not modify frozen Stage-18/23/24 code,
contracts, checkpoints, or results to make this experiment run.

## Frozen Collection

- Use exactly the verified Stage-18 fit IDs and order, currently 95 videos.
  Neither the 24 tune videos nor the original 40 validation videos are opened
  for images or annotations. OTB/UAV123 are not used.
- Freeze the existing Stage-18 tracking representation, backbone, and head.
  No policy checkpoints from the best-looking PPO seed are reused.
- Each fit video starts from GT once, then follows its own predicted boxes
  and memory for at most 128 tracked frames. No GT crop correction/resets.
- Canonical P30 call positions are fixed before seeing images or labels:
  the call count at tracked frame t is floor(3t/10). Candidate layers are a
  reproducible shuffled balanced multiset of L7-L12 (collection seed 2080).
- Pick eight evenly spaced eligible call positions per video, before reading
  labels. Each must have 20 future tracked frames available, including itself.
- At each anchor, force each of the six independent first layers L7-L12.
  Every branch spends exactly one candidate call at that first frame.
  There is no privileged L9 fallback or forced diversity objective.
- Execute two matched H20 continuations. Repeat zero uses the future behavior
  actions; repeat one independently redraws later layers uniformly. Their
  call masks are identical. All six first actions receive the same future
  action sequence within a repeat. Every branch has exactly equal cost.
- Each branch crops from its own predictions and writes its own memory.
  The chosen first layer with repeat-zero continuation must replay the
  actual subsequent H20 boxes, not only the first box.
- Restore the actual after-state, actor transient state, and Python/NumPy/
  torch/CUDA RNG state even on branch exceptions. Verify frozen model weights.
- Save and audit one atomic NPZ per video, with contract/code hashes. Restart
  the same command to skip validated completed videos. An interrupted video
  restarts from its beginning; no partially written video is considered done.

For 95 videos the plan has 760 anchors and 182,400 branch frame executions,
plus at most 12,160 live tracked frames. The two continuation repeats are
Monte Carlo repeats, not additional independent videos.

## Explicit Causal Relations

Both probe arms get the same compact pre-action 8x8 shared-prefix grid,
template mean, frozen target memory, causal history, and previous predicted
box expressed in the current crop. This ROI is **not current ground truth**.

The augmented arm also receives a deterministic 280-value encoding of those
same inputs. It introduces an explicit representation/inductive bias, not
new information unavailable to the control:

1. Search-token cosine similarity to the template.
2. Similarity to a target prototype pooled inside the prior predicted ROI.
3. Maximum similarity to three spatially separated outside-ROI peaks.
4. Target-minus-distractor similarity, plus peak, margin, and entropy summaries.

Outside-ROI peaks are only distractor hypotheses. A drifted prior ROI can be
wrong. This design does not silently treat a predicted target as ground truth.
No candidate block, extra detection head, future image, action outcome, or
label is evaluated to construct the state. All cached relation values must
reconstruct from the saved raw pre-action fields.

## Frozen Probe

- Target: visibility-normalized H20 discounted IoU (gamma 0.97), centered
  across the six actions within each state. Common video difficulty therefore
  cannot alone produce good action-value prediction.
- Eligibility: anchor frame visible and at least 80% of H20 frames visible.
  Invalid rewards are zero. Need eligible anchors in every frozen fit video
  and at least 80% of all planned anchors for formal screening.
- Five video-disjoint folds, split seed 2081. Only the new probe is OOF:
  the frozen tracking representation previously trained on these fit videos.
  This is development evidence, not a fully unseen-video generalization test.
- Same convolutional/template encoders, 128-wide hidden layer, and six-output
  centered regression head for both arms. The control zeros the 280 relation
  inputs; parameter shapes and initial weights are matched across arms.
- Fit-only normalization in each fold; clip normalized inputs to [-10,10].
  AdamW lr 0.0003, weight decay 0.001, gradient clip 1, 40 epochs, batch 64.
  Video-balanced loss. No test-label early stopping or hyperparameter search.
- Seeds 2082/2083/2084. Average their scores before choosing an action. No
  best-seed selection, and no claim that these repeats create more videos.
- Report raw versus relational probes, a training-fold-selected fixed layer,
  uniform random-layer expectation, and **all six** static layers.
- Average anchor effects within video, then bootstrap videos 5,000 times
  (seed 2085). The six static comparisons use Bonferroni lower bounds.
- A supporting hindsight diagnostic selects using one continuation repeat
  and scores on the other, symmetrically. This still uses GT and the same
  initial intervention frame. It is NOT the disjoint temporal-window audit
  from Stage-15 and cannot overturn that failed stability result.

`PASS_RELATIONAL_STATE_SCREEN` requires positive lower bounds for all:

- Cross-repeat hindsight minus training-fold fixed layer.
- Relational minus raw state.
- Relational minus every one of L7-L12, with the static-family adjustment.
- The last two conditions must also pass on **H2-H20**, excluding the action
  frame, using the same H20-trained predictions. Immediate-only benefit must
  not masquerade as useful long-horizon state information.

Otherwise the result is `FAIL_EQUAL_COST_HEADROOM_SCREEN` or
`FAIL_RELATIONAL_STATE_SCREEN`. Do not impose a minimum number of selected
layers or force budget spending to manufacture adaptivity.

These predictions score isolated branch choices under matched randomized
continuations. They are not closed-loop policy AUC, deployed PPO returns,
speed measurements, or external validation. A pass alone does not justify a
full PPO training campaign.

## Server Commands

Sync the three Stage-25 Python modules, their test, and this document. First:

```text
tracking/stage25_relation_common.py
tracking/stage25_relation_collect.py
tracking/stage25_relation_screen.py
tracking/test_stage25_relation.py
STAGE25_RELATIONAL_STATE_PROBE.md
```

```bash
export STAGE18_ROOT="$PWD/output/stage18/temporal_layer_policy_v1"
export STAGE18_REP_BUNDLE="$STAGE18_ROOT/frozen_representation_v1.json"
export INTERVENTION_MANIFEST="$PWD/output/stage8/l9_intervention_v1/intervention_manifest.json"
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"
export STAGE24_DIR="$PWD/output/stage24/ppo_deployment_action_audit_v1"
export STAGE25_ROOT="$PWD/output/stage25/relational_state_probe_v1"
export STAGE25_CACHE="$STAGE25_ROOT/cache"

python tracking/test_stage25_relation.py

export FIRST_STAGE25_SEQ=$(python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["fit_videos"][0])' \
  "${STAGE18_REP_BUNDLE:?}")

python tracking/stage25_relation_collect.py \
  --bundle "${STAGE18_REP_BUNDLE:?}" \
  --manifest "${INTERVENTION_MANIFEST:?}" \
  --stage24_dir "${STAGE24_DIR:?}" \
  --checkpoint "${CHECKPOINT:?}" \
  --parameter deit_distilled --gpu 0 \
  --sequence "${FIRST_STAGE25_SEQ:?}" \
  --output_dir "${STAGE25_CACHE:?}"

python tracking/stage25_relation_screen.py \
  --input "${STAGE25_CACHE:?}" \
  --output_dir "$STAGE25_ROOT/sanity_audit" --audit_only
```

The tests must run their real-PyTorch checks on the server, not skip them.
Expect `PILOT_INTEGRITY_ONLY` after one video, with zero replay error (or at
most 1e-6), exact cost and relation reconstruction checks passing. No quality
decision is made from the pilot. If integrity fails, stop before full collection.

After the pilot passes, repeat the collection command **without `--sequence`**
and keep the same cache directory. This collects all frozen fit videos,
skipping the validated pilot. Then:

```bash
python tracking/stage25_relation_screen.py \
  --input "${STAGE25_CACHE:?}" \
  --output_dir "$STAGE25_ROOT/full_data_audit" --audit_only

# Only after the complete data audit passes:
python tracking/stage25_relation_screen.py \
  --input "${STAGE25_CACHE:?}" \
  --output_dir "$STAGE25_ROOT/screen_v1" --device cuda

cat "$STAGE25_ROOT/screen_v1/stage25_decision.json"
cat "$STAGE25_ROOT/screen_v1/stage25_paired.csv"
```

Use one writer/GPU job at a time. The screen resumes completed fold/arm/seed
artifacts after hash checks; an interrupted unfinished fit restarts its 40
epochs. Resume with identical code/arguments, not an altered epoch/seed list.
After SIGKILL, verify no process is writing before removing only the new
directory's `.stage23_writer.lock` (the shared writer helper's lock name).
Never remove old experiment caches or edit frozen JSON to bypass a check.
