"""Matched three-arm PPO branch-credit smoke. No tune/validation labels loaded."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np

from stage4_common import file_hash, fingerprint, load_manifest, write_csv, write_json
from stage4_runtime import Runtime, add_runtime_arguments
from stage18_freeze_representation import verify_representation_bundle
from stage18_onpolicy_ppo import action_distribution, ppo_sources
from stage18_policy_runtime import configure_policy, load_tracking_representation, policy_forward
from stage18_ppo_math import gae, normalize_advantages, ppo_objective
from stage18_representation_train import atomic_torch_save, load_checkpoint
from stage18_temporal_model import build_temporal_modules
from stage21_counterfactual_math import expected_clipped_loss
from stage23_branch_credit import ARMS, ANCHORS, GAMMA, HORIZON, REPEATS, BranchCollector, branch_credit


STEPS, ROLLOUT = 1024, 256


@contextmanager
def single_writer(output):
    """Fail closed after SIGKILL; never guess that a foreign PID is stale."""
    output.mkdir(parents=True, exist_ok=True)
    path = output / ".stage23_writer.lock"
    try:
        handle = path.open("x", encoding="ascii")
    except FileExistsError as error:
        raise RuntimeError("Stage-23 lock exists. Confirm no writer is running before removing " + str(path)) from error
    try:
        with handle:
            handle.write(str(os.getpid()) + "\n")
        yield
    finally:
        path.unlink()


def update_policy(torch, modules, optimizer, data, arm, seed, update):
    advantages, returns = gae(data["rewards"], data["values"], data["next_values"],
                              data["terminated"], data["truncated"], GAMMA, .95)
    if not data["valid"].any():
        raise RuntimeError("Smoke rollout has no valid live rewards")
    advantages = normalize_advantages(advantages, data["valid"])
    anchors = data["counterfactual"]
    anchor_valid = np.asarray([a["valid"] for a in anchors], bool)
    if not anchor_valid.any():
        raise RuntimeError("No valid branch anchors; do not silently run an auxiliary-free arm")
    probabilities = np.stack([a["old_probabilities"] for a in anchors])
    credit = branch_credit(np.stack([a["rewards"] for a in anchors]), probabilities, anchor_valid, arm)
    device = next(modules.parameters()).device
    tensor = lambda x, dtype: torch.as_tensor(x, dtype=dtype, device=device)
    anchor_indices = np.asarray([a["index"] for a in anchors])
    anchor_obs = {k: v[anchor_indices].to(device) for k, v in data["observation"].items()}
    old_p = tensor(probabilities, torch.float32)
    cf_adv = tensor(credit["advantage"], torch.float32)
    cf_valid = tensor(anchor_valid, torch.bool)
    observation = {k: v.to(device) for k, v in data["observation"].items()}
    action = tensor(data["actions"], torch.long)
    old_logp = tensor(data["old_log_prob"], torch.float32)
    with torch.no_grad():
        logits, _ = policy_forward(torch, modules, observation)
        replay = torch.distributions.Categorical(logits=logits).log_prob(action)
        replay_error = float((replay - old_logp).abs().max())
        anchor_logits, _ = policy_forward(torch, modules, anchor_obs)
        probability_error = float((torch.softmax(anchor_logits, -1) - old_p).abs().max())
    if max(replay_error, probability_error) > 1e-4:
        raise RuntimeError("Live/branch old-policy likelihoods do not replay")
    metrics, maximum_norm = [], 0.
    for epoch in range(4):
        order = np.random.default_rng(seed + 1000 * update + epoch).permutation(len(advantages))
        # One full-rollout minibatch in this fixed smoke, matched across arms.
        obs = {k: v[order] for k, v in observation.items()}
        logits, value = policy_forward(torch, modules, obs)
        loss, result = ppo_objective(
            torch, logits, value, action[order], old_logp[order],
            tensor(returns[order], torch.float32), tensor(advantages[order], torch.float32),
            tensor(data["valid"][order], torch.bool))
        anchor_logits, _ = policy_forward(torch, modules, anchor_obs)
        auxiliary = expected_clipped_loss(torch, anchor_logits, old_p, cf_adv, cf_valid)
        auxiliary_gradients = torch.autograd.grad(auxiliary, tuple(modules.actor.parameters()), retain_graph=True)
        auxiliary_norm = float(torch.sqrt(sum(g.detach().square().sum() for g in auxiliary_gradients)))
        loss = loss + credit["weight"] * auxiliary
        if not torch.isfinite(loss) or not np.isfinite(auxiliary_norm):
            raise RuntimeError("Nonfinite branch PPO loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in modules.parameters() if p.requires_grad], .5)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite branch PPO gradients")
        maximum_norm = max(maximum_norm, float(norm))
        optimizer.step()
        metrics.append(dict(**{k: float(v) for k, v in result.items()},
                            auxiliary_loss=float(auxiliary.detach()), auxiliary_actor_gradient_norm=auxiliary_norm))
    result = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
    result.update(likelihood_replay_error=replay_error, probability_replay_error=probability_error,
                  chosen_replay_error=max(a["chosen_replay_error"] for a in anchors),
                  branch_transitions=sum(a["simulation_transitions"] for a in anchors),
                  branch_candidate_blocks=sum(a["candidate_calls"] for a in anchors),
                  live_candidate_blocks=int(np.sum(data["actions"] > 0)),
                  bootstrap_prefix_calls=data["bootstrap_prefix_calls"],
                  valid_anchors=int(anchor_valid.sum()), anchor_count=len(anchors),
                  credit_scale=credit["scale"], auxiliary_weight=credit["weight"],
                  maximum_gradient_norm=maximum_norm, optimizer_steps=len(metrics),
                  candidate_ratio=float(np.mean(data["actions"] > 0)),
                  action_distribution=action_distribution(data["actions"]),
                  mean_valid_iou=float(data["rewards"][data["valid"]].mean()))
    return result


def check_runtime(identity, frozen):
    # Moving to another GPU is allowed; moving to another tracker is not.
    fields = ("parameter", "checkpoint_sha256", "config", "lens_z", "lens_x", "embed_dim")
    if any(identity.get(k) != frozen.get(k) for k in fields):
        raise ValueError("Original tracker model/config differs from the representation bundle")


def verify_frozen(torch, modules, frozen):
    if any(not torch.equal(frozen[n], p.detach().cpu()) for n, p in modules.named_parameters() if n in frozen):
        raise RuntimeError("Frozen tracking representation changed")


def model_digest(modules):
    digest = hashlib.sha256()
    for name, value in sorted(modules.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("ascii"))
        digest.update(str((array.shape, array.dtype)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def train_arm(rt, bundle, repdir, sources, arm, seed, output, contract):
    torch = rt.torch
    import torch.nn as nn
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    config = bundle["model"]
    modules = build_temporal_modules(torch, nn, config["embed_dim"], config["lens_z"],
                                     config["search_side"]).to(next(rt.net.parameters()).device)
    representation = load_checkpoint(torch, repdir / "best.pt")
    if (representation["contract_fingerprint"] != bundle["representation_contract_fingerprint"]
            or representation["epoch"] != bundle["best_epoch"]):
        raise ValueError("Frozen representation checkpoint metadata mismatch")
    load_tracking_representation(modules, representation["model"])
    parameters = configure_policy(modules)
    modules.eval()
    frozen = {n: p.detach().cpu().clone() for n, p in modules.named_parameters() if not p.requires_grad}
    initial_actor = {n: p.detach().cpu().clone() for n, p in modules.actor.named_parameters()}
    initial_digest = model_digest(modules)
    optimizer = torch.optim.Adam(parameters, lr=3e-4, eps=1e-5)
    collector = BranchCollector(rt, modules, sources, seed)
    output.mkdir(parents=True, exist_ok=True)
    last = output / "last.pt"
    history, steps = [], 0
    if last.exists():
        saved = load_checkpoint(torch, last)
        if (saved["contract_fingerprint"] != contract["fingerprint"] or saved["arm"] != arm
                or saved["initial_model_sha256"] != initial_digest):
            raise ValueError("Checkpoint belongs to a different experiment or arm")
        modules.load_state_dict(saved["model"])
        verify_frozen(torch, modules, frozen)
        optimizer.load_state_dict(saved["optimizer"])
        steps, history = saved["steps"], saved["history"]
        if (steps % ROLLOUT or not 0 <= steps <= STEPS or len(history) != steps // ROLLOUT
                or collector.total_steps != 0):
            raise ValueError("Invalid atomic checkpoint progress")
        collector.restore(saved["collector"])
        if collector.total_steps != steps:
            raise ValueError("Collector/checkpoint progress mismatch")
        torch.set_rng_state(saved["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
    while steps < STEPS:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        data = collector.collect(ROLLOUT)
        metrics = update_policy(torch, modules, optimizer, data, arm, seed, steps // ROLLOUT + 1)
        verify_frozen(torch, modules, frozen)
        torch.cuda.synchronize()
        steps += ROLLOUT
        metrics.update(seconds=time.perf_counter() - start,
                       peak_allocated_mb=torch.cuda.max_memory_allocated() / 1024**2)
        history.append(dict(arm=arm, steps=steps, update=steps // ROLLOUT, **metrics))
        state = dict(contract_fingerprint=contract["fingerprint"], arm=arm, steps=steps,
                     initial_model_sha256=initial_digest,
                     model=modules.state_dict(), optimizer=optimizer.state_dict(), history=history,
                     collector=collector.state_dict(), torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(),
                     python_rng=random.getstate())
        atomic_torch_save(torch, state, last)
        # The checkpoint is the authority if an interruption precedes CSV publication.
        write_csv(output / "stage23_history.csv", history)
        print("[{}] {}/{} live, {} branch, IoU={:.4f}, credit_std={:.5f}, replay={:.2g}".format(
            arm, steps, STEPS, metrics["branch_transitions"], metrics["mean_valid_iou"],
            metrics["credit_scale"], metrics["likelihood_replay_error"]), flush=True)
    write_csv(output / "stage23_history.csv", history)
    actor_changed = any(not torch.equal(initial_actor[n], p.detach().cpu())
                        for n, p in modules.actor.named_parameters())
    row = dict(arm=arm, live_transitions=steps,
               initial_model_sha256=initial_digest,
               branch_transitions=sum(r["branch_transitions"] for r in history),
               branch_candidate_blocks=sum(r["branch_candidate_blocks"] for r in history),
               live_candidate_blocks=sum(r["live_candidate_blocks"] for r in history),
               bootstrap_prefix_calls=sum(r["bootstrap_prefix_calls"] for r in history),
               optimizer_steps=sum(r["optimizer_steps"] for r in history),
               valid_anchors=sum(r["valid_anchors"] for r in history),
               actor_changed=actor_changed, frozen_representation_unchanged=True,
               maximum_replay_error=max(max(r["likelihood_replay_error"], r["probability_replay_error"],
                                            r["chosen_replay_error"]) for r in history),
               auxiliary_credit_nonzero=any(r["credit_scale"] > 1e-8 for r in history),
               auxiliary_actor_gradient_nonzero=any(r["auxiliary_actor_gradient_norm"] > 1e-8 for r in history),
               seconds=sum(r["seconds"] for r in history), checkpoint=str(last),
               checkpoint_sha256=file_hash(last))
    write_json(output / "stage23_arm_summary.json", row)
    return row


def publish_contract(output, contract):
    path = output / "stage23_contract.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != contract:
            raise ValueError("Contract changed; use a new output directory, not overwrite")
    else:
        if any(p.name != ".stage23_writer.lock" for p in output.iterdir()):
            raise ValueError("Unidentified output files; use an empty output directory")
        write_json(path, contract)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    for name in ("bundle", "manifest", "output_dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, default=2074)
    args = parser.parse_args()
    if any(not (getattr(args, n) or "").strip() for n in ("bundle", "manifest", "checkpoint", "output_dir")):
        parser.error("Set nonempty bundle, manifest, original checkpoint and output paths explicitly")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    _, bundle, repdir = verify_representation_bundle(args.bundle)
    manifest = load_manifest(args.manifest)
    if manifest["fingerprint"] != bundle["source_manifest_fingerprint"]:
        raise ValueError("Manifest differs from frozen bundle")
    names = bundle["pilot_videos"]
    if (len(names) != 5 or len(set(names)) != 5 or not set(names) <= set(bundle["fit_videos"])
            or set(names) & set(bundle["tune_videos"])):
        raise ValueError("Smoke requires five distinct frozen fit-pilot videos and no tune videos")
    sources = ppo_sources(manifest, names, 128)
    output = Path(args.output_dir).expanduser().resolve()
    with single_writer(output):
        rt = Runtime(args.parameter, args.gpu, args.checkpoint)
        check_runtime(rt.identity, bundle["runtime"])
        contract = dict(schema="stage23_branch_credit_smoke_v1", bundle_fingerprint=bundle["fingerprint"],
                        manifest_fingerprint=manifest["fingerprint"], runtime=rt.identity,
                        seed=args.seed, arms=list(ARMS), primary="ppo_cf_h20", fit=list(names),
                        tune=[], validation=[], live_steps_per_arm=STEPS, rollout_steps=ROLLOUT,
                        branch_horizon=HORIZON, repeats=REPEATS, anchors_per_rollout=ANCHORS,
                        gamma=GAMMA, gae_lambda=.95, auxiliary_weight=.5,
                        branch_bootstrap=False, branch_valid_fraction=.8,
                        learning_rate=3e-4, adam_epsilon=1e-5, clip=.2, grad_clip=.5,
                        update_epochs=4, minibatch_size=ROLLOUT, value_weight=.5, entropy_weight=.01,
                        budget="causal P30 token credit, initial zero, +3/frame, -10/candidate, cap20",
                        role="execution_and_training_integrity_only_not_efficacy",
                        code_sha256={n: file_hash(Path(__file__).parent / n) for n in (
                            "stage23_branch_ppo.py", "stage23_branch_credit.py", "stage21_counterfactual_math.py",
                            "stage21_counterfactual_runtime.py", "stage18_policy_runtime.py", "stage18_ppo_math.py",
                            "stage18_onpolicy_ppo.py", "stage18_representation_train.py", "stage18_environment.py",
                            "stage18_temporal_model.py", "stage18_freeze_representation.py", "stage4_common.py",
                            "stage4_runtime.py", "validation_metrics.py")})
        contract["fingerprint"] = fingerprint(contract)
        publish_contract(output, contract)
        rows = [train_arm(rt, bundle, repdir, sources, arm, args.seed, output / arm, contract) for arm in ARMS]
        expected_branch = STEPS // ROLLOUT * ANCHORS * REPEATS * 7 * HORIZON
        passed = len({r["initial_model_sha256"] for r in rows}) == 1 and all(
                     r["actor_changed"] and r["frozen_representation_unchanged"]
                     and r["maximum_replay_error"] <= 1e-4
                     and r["branch_transitions"] == expected_branch and r["optimizer_steps"] == 16
                     and r["valid_anchors"] > 0 and r["auxiliary_credit_nonzero"]
                     and r["auxiliary_actor_gradient_nonzero"] for r in rows)
        decision = dict(schema="stage23_branch_credit_smoke_v1", contract_fingerprint=contract["fingerprint"],
                        status="PASS_BRANCH_PPO_SMOKE" if passed else "FAIL_BRANCH_PPO_SMOKE",
                        validation_labels_loaded=False, tune_labels_loaded=False, arms=rows,
                        training_transitions_per_arm=STEPS + expected_branch,
                        warning="Integrity only. No policy accuracy, generalization or speed claim.",
                        next_step="Review replay and matched costs before preregistering a fixed three-seed fit/tune comparison.")
        decision["fingerprint"] = fingerprint(decision)
        write_csv(output / "stage23_arm_summary.csv", rows)
        write_json(output / "stage23_decision.json", decision)
        print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
