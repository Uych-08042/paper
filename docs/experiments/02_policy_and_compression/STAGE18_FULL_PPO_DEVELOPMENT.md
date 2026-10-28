# Stage-18 Full PPO Development Run

The user-reported 1,024-step smoke passed. This run now executes the already
declared full development budget, not another feasibility screen or a new
method variant. No previously passed training/runtime source is modified.

## Workflow amendment

`STAGE18_FULL_DEVELOPMENT_AMENDMENT_V1.json` records the prospective revision:
external dataset acquisition is no longer a prerequisite for development
training. NfS is unavailable locally and LaSOT download/storage is impractical.
No external benchmark has yet been finalized. Choose and freeze that protocol
before inspecting any of its tracker outcomes, and audit physical-video
overlap and coverage before external evaluation.

This revises only experiment order. It does not change the model, data split,
P30 rule, training settings, seeds, checkpoint selection, controls, ablations
or claim thresholds. Old protocol documents remain historical records; their
external-data-before-training requirement is superseded by this amendment.
The recorded predecessor hashes came from the user's server logs. The server
must still verify the actual frozen representation before launch.

The earlier resource target remains failed: the representation engineering
schedule was 1.1092 times original SGLA latency. Neither smoke throughput nor
candidate-call frequency constitutes a learned-policy speed result.

## Fixed experiment

- Fit/tune: exactly the frozen 96/24 video identities, excluding the original
  40 validation videos, UAV123 and OTB from selection.
- Initialize each seed from epoch-7 frozen tracking representation, never
  from the five-video smoke policy. The policy encoder/critic are seeded anew.
- Actions: Fast and six independent candidate blocks L7-L12. No L9 preference,
  required number of active layers or forced candidate spending.
- Seeds: 2054, 2055 and 2056. Run sequentially; do not select a winning seed.
- Budget: 250,000 newly executed transitions per seed; rollout size 2,048,
  minibatches 256, gamma 0.97. All other original PPO settings are unchanged.
- Select within each seed by exact deployment masked-argmax tune AUC at
  20,000-step intervals and at 250,000; tie by lower descriptive latency,
  then earlier step. There are 13 planned evaluations per seed.

## Server command

Sync this document and the amendment JSON only. The already passed Python
sources must remain unchanged so the smoke/representation lineage is retained.
Run the following from the repository root. The child Bash shell stops if a
seed or verification fails; it does not change interactive-shell options.

```bash
bash <<'BASH'
set -euo pipefail

export STAGE18_ROOT="$PWD/output/stage18/temporal_layer_policy_v1"
export STAGE18_FULL_ROOT="$STAGE18_ROOT/ppo_full_v1"
export STAGE18_REP_BUNDLE="$STAGE18_ROOT/frozen_representation_v1.json"
export INTERVENTION_MANIFEST="$PWD/output/stage8/l9_intervention_v1/intervention_manifest.json"
export CHECKPOINT="$PWD/output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar"

python tracking/stage18_freeze_representation.py \
  --verify_bundle "$STAGE18_REP_BUNDLE"

mkdir -p "$STAGE18_FULL_ROOT"
AMENDMENT=STAGE18_FULL_DEVELOPMENT_AMENDMENT_V1.json
RECORD="$STAGE18_FULL_ROOT/protocol_amendment_v1.json"
if [ -e "$RECORD" ]; then
  cmp "$AMENDMENT" "$RECORD"
else
  cp -n "$AMENDMENT" "$RECORD"
fi

for SEED in 2054 2055 2056; do
  python -u tracking/stage18_onpolicy_ppo.py \
    --bundle "$STAGE18_REP_BUNDLE" \
    --manifest "$INTERVENTION_MANIFEST" \
    --checkpoint "$CHECKPOINT" \
    --parameter deit_distilled \
    --gpu 0 \
    --mode full \
    --seed "$SEED" \
    --steps 250000 \
    --rollout_steps 2048 \
    --minibatch_size 256 \
    --gamma 0.97 \
    --output_dir "$STAGE18_FULL_ROOT/seed${SEED}" \
    2>&1 | tee -a "$STAGE18_FULL_ROOT/seed${SEED}.log"
  python -c 'import json,sys; d=json.load(open(sys.argv[1])); ok=d["status"]=="COMPLETE_ONPOLICY_TRAINING" and d["steps"]==250000 and d["frozen_representation_unchanged"] and d["actor_changed"]; print(d["status"]); sys.exit(0 if ok else 1)' \
    "$STAGE18_FULL_ROOT/seed${SEED}/stage18_ppo_decision.json"
done
BASH
```

Keep the shell/session alive while running. Reissue the identical command to
resume at the last completed rollout/update. Earlier completed seeds reload
their existing completed run before the next seed resumes. Never run two copies
against the same output root. Keep logs outside individual seed directories:
the trainer rejects unidentified nonempty output directories.

At startup each seed first computes original Fast/SGLA, memory-only Fast, all
six P30 single-layer controls and fixed-seed random-layer controls on tune.
This may take time before the first `[PPO full]` line. Do not interpret those
static control computations as an all-candidate deployed actor.

## Outputs to inspect

After all three finish:

```bash
export STAGE18_FULL_ROOT="$PWD/output/stage18/temporal_layer_policy_v1/ppo_full_v1"
for SEED in 2054 2055 2056; do
  printf '\n===== seed %s =====\n' "$SEED"
  cat "$STAGE18_FULL_ROOT/seed${SEED}/stage18_ppo_decision.json"
  tail -n 5 "$STAGE18_FULL_ROOT/seed${SEED}/stage18_ppo_history.csv"
done
```

Preserve each seed's `best.pt`, `last.pt`, `stage18_ppo_contract.json`,
`stage18_controls.csv`, `stage18_tune_policy.csv` and history/decision files.
`best.pt` is exported on completion from the atomic selected state in `last.pt`.
Do not substitute the final step for a different tune-selected best checkpoint.

`COMPLETE_ONPOLICY_TRAINING` means optimization/integrity completed, not that
the method beat a baseline. Track training loss and action usage descriptively;
the smoke's `maximum_gradient_norm` is the pre-clipping norm returned by
`clip_grad_norm_`, not the applied gradient norm. A high early value alone does
not authorize changing the declared loss weights.

## Interpretation after completion

Use all three selected policies; average their AUCs per video first, then use
5,000 paired physical-video bootstrap draws (analysis seed 2057). The primary
development contrasts remain full policy minus original SGLA and full policy
minus the best tune-selected static control among memory-Fast and L7-L12.
Static controls must be compared at the same P30 cap; also report actual
utilization because a learned policy may spend less. Do not pick a baseline
from external outcomes or require layer diversity for a pass.

All checkpoint selection and these comparisons reuse tune data: any confidence
intervals here are exploratory development diagnostics, not untouched-test
evidence or selection-adjusted confirmatory inference. Before claiming the
long-term mechanism, the predeclared gamma=0 and retrained no-memory ablations
are still required. Before an accuracy-and-efficiency claim, retain the original
separate end-to-end interleaved timing requirement. External confirmation remains
a later stage and is not silently replaced by development results.
