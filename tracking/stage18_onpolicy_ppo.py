"""Live Stage-18 PPO; the representation is frozen and every rollout is fresh."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from stage4_common import (file_hash, fingerprint, frame_path, load_annotations,
                           load_manifest, tracking_iou, write_csv, write_json)
from stage4_runtime import Runtime, add_runtime_arguments
from stage18_environment import TemporalEnvironment
from stage18_temporal_model import build_temporal_modules
from stage18_policy_runtime import (PolicyEnvironment, configure_policy, policy_forward,
                                    load_tracking_representation)
from stage18_ppo_math import gae, normalize_advantages, ppo_objective
from stage18_freeze_representation import verify_representation_bundle
from stage18_representation_train import load_sources, load_checkpoint, atomic_torch_save


ACTION_NAMES = ("fast", "l7", "l8", "l9", "l10", "l11", "l12")


def action_distribution(actions):
    return "|".join("{}:{}".format(name, int(np.sum(np.asarray(actions) == index)))
                    for index, name in enumerate(ACTION_NAMES))


def ppo_sources(manifest, names, maximum):
    sources = load_sources(manifest, names, maximum)
    entries = {e["sequence"]: e for e in manifest["sequences"]}
    for source in sources:
        directory, gt, _ = load_annotations(manifest["root"], manifest["dataset"], source["sequence"])
        end = entries[source["sequence"]]["start"] + len(source["paths"])
        source["terminal"] = end == len(gt)
        source["bootstrap_path"] = None if source["terminal"] else frame_path(directory, manifest["dataset"], end)
        if source["bootstrap_path"] is not None and not source["bootstrap_path"].is_file():
            raise ValueError("Missing next image for clip-truncation bootstrap: " + source["sequence"])
    return sources


class Collector:
    def __init__(self, runtime, modules, sources, seed):
        self.rt, self.modules, self.sources = runtime, modules, sources
        self.env = PolicyEnvironment(runtime, modules)
        self.rng = np.random.default_rng(seed)
        self.queue = []
        self.active = None
        self.index = 0

    def _reset(self):
        if not self.queue:
            self.queue = self.rng.permutation(len(self.sources)).tolist()
        self.active = int(self.queue.pop())
        source = self.sources[self.active]
        self.env.reset(self.rt.read(source["paths"][0]), source["gt"][0])
        self.index = 1

    def state_dict(self):
        return dict(active=self.active, index=self.index, queue=list(self.queue),
                    rng_state=self.rng.bit_generator.state,
                    environment=None if self.active is None else self.env.snapshot())

    def restore(self, state):
        self.active, self.index, self.queue = state["active"], state["index"], list(state["queue"])
        self.rng.bit_generator.state = state["rng_state"]
        if self.active is not None:
            source = self.sources[self.active]
            self.env.reset(self.rt.read(source["paths"][0]), source["gt"][0])
            snapshot = dict(state["environment"])
            if snapshot["memory"] is not None:
                snapshot["memory"] = snapshot["memory"].to(next(self.modules.parameters()).device)
            self.env.restore(snapshot)

    def collect(self, count):
        if count < 1:
            raise ValueError("A rollout must contain at least one transition")
        torch = self.rt.torch
        observations, actions, logp, values, rewards, valid, terminal, truncated = [], [], [], [], [], [], [], []
        bootstraps = {}
        for step in range(count):
            if self.active is None:
                self._reset()
            source = self.sources[self.active]
            result = self.env.step(self.rt.read(source["paths"][self.index]), stochastic=True)
            is_valid = bool(source["valid"][self.index])
            overlap = float(tracking_iou(result["box"], source["gt"][self.index], is_valid))
            if is_valid and (not np.isfinite(overlap) or overlap < 0):
                raise RuntimeError("Invalid live tracking prediction")
            observations.append({k: v.cpu() for k, v in result["observation"].items()})
            actions.append(result["action"])
            logp.append(float(result["log_probability"].item()))
            values.append(float(result["value"].item()))
            rewards.append(overlap if is_valid else 0.)
            valid.append(is_valid)
            self.index += 1
            ended = self.index == len(source["paths"])
            terminal.append(ended and source["terminal"])
            truncated.append(ended and not source["terminal"])
            if ended:
                bootstraps[step] = (0. if source["terminal"] else
                                    self.env.peek_value(self.rt.read(source["bootstrap_path"])))
                self.active = None
        next_values = np.r_[values[1:], 0.]
        if self.active is not None:
            source = self.sources[self.active]
            bootstraps[count - 1] = self.env.peek_value(self.rt.read(source["paths"][self.index]))
            truncated[-1] = True
        for step, bootstrap in bootstraps.items():
            next_values[step] = bootstrap
        data = dict(observation={k: torch.cat([o[k] for o in observations]) for k in observations[0]},
                    actions=np.asarray(actions), old_log_prob=np.asarray(logp),
                    values=np.asarray(values), rewards=np.asarray(rewards),
                    valid=np.asarray(valid), terminated=np.asarray(terminal),
                    truncated=np.asarray(truncated), next_values=next_values)
        return data


def update_policy(torch, modules, optimizer, data, args, update):
    advantages, returns = gae(data["rewards"], data["values"], data["next_values"],
                              data["terminated"], data["truncated"], args.gamma, .95)
    valid_labels = bool(data["valid"].any())
    advantages = (normalize_advantages(advantages, data["valid"])
                  if valid_labels else np.zeros_like(advantages))
    device = next(modules.parameters()).device
    tensor = lambda value, dtype: torch.as_tensor(value, dtype=dtype, device=device)
    # Keep observations on CPU until a minibatch is used. They are FP32 so
    # recomputed old likelihoods are not altered by lossy cache quantization.
    def batch(indices):
        obs = {k: v[indices].to(device) for k, v in data["observation"].items()}
        return obs, tensor(data["actions"][indices], torch.long), \
            tensor(data["old_log_prob"][indices], torch.float32)
    count = len(advantages)
    error = 0.
    with torch.no_grad():
        for begin in range(0, count, args.minibatch_size):
            indices = np.arange(begin, min(count, begin + args.minibatch_size))
            obs, action, old = batch(indices)
            logits, _ = policy_forward(torch, modules, obs)
            replay = torch.distributions.Categorical(logits=logits).log_prob(action)
            error = max(error, float((replay - old).abs().max()))
    if error > 1e-4:
        raise RuntimeError("Old-policy likelihoods do not replay: {}".format(error))
    metrics = []
    gradient_norm = 0.
    for epoch in range(4 if valid_labels else 0):
        order = np.random.default_rng(args.seed + 1000 * update + epoch).permutation(count)
        for begin in range(0, count, args.minibatch_size):
            indices = order[begin:begin + args.minibatch_size]
            if not data["valid"][indices].any():
                continue
            obs, action, old = batch(indices)
            logits, value = policy_forward(torch, modules, obs)
            loss, result = ppo_objective(
                torch, logits, value, action, old, tensor(returns[indices], torch.float32),
                tensor(advantages[indices], torch.float32), tensor(data["valid"][indices], torch.bool))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in modules.parameters() if p.requires_grad], .5)
            if not torch.isfinite(norm):
                raise RuntimeError("Non-finite PPO gradients")
            gradient_norm = max(gradient_norm, float(norm))
            optimizer.step()
            metrics.append({k: float(v) for k, v in result.items()})
    result = ({k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
              if metrics else dict(policy_loss=0., value_loss=0., entropy=0.,
                                   approx_kl=0., clip_fraction=0.))
    result.update(likelihood_replay_error=error, maximum_gradient_norm=gradient_norm,
                  skipped_update=not valid_labels,
                  candidate_ratio=float(np.mean(data["actions"] > 0)),
                  action_distribution=action_distribution(data["actions"]),
                  mean_valid_iou=(float(data["rewards"][data["valid"]].mean()) if valid_labels else None),
                  valid_transitions=int(data["valid"].sum()))
    return result


def success_auc(overlaps):
    values = np.r_[1., np.asarray(overlaps, dtype=np.float64)]
    return 100 * float(np.mean([np.mean(values > i * .05) for i in range(21)]))


def evaluate(rt, modules, sources, method="policy"):
    rows = []
    with rt.torch.no_grad():
        for source in sources:
            env = PolicyEnvironment(rt, modules) if method == "policy" else TemporalEnvironment(rt, modules)
            env.reset(rt.read(source["paths"][0]), source["gt"][0])
            overlaps, elapsed, actions = [], [], []
            rng = np.random.default_rng(int(fingerprint(source["sequence"])[:16], 16))
            for index, path in enumerate(source["paths"][1:], start=1):
                image = rt.read(path)
                rt.torch.cuda.synchronize()
                started = time.perf_counter()
                if method in ("fast", "sgla"):
                    box = rt.run_image(image, method)
                    rt.tracker.state, rt.tracker.frame_id = list(box), index
                    action = int(method == "sgla")
                elif method == "policy":
                    result = env.step(image)
                    box, action = result["box"], result["action"]
                else:
                    action = 0
                    if method != "memory_fast" and min(20, env.budget.credit + 3) >= 10:
                        action = int(rng.integers(1, 7)) if method == "random" else int(method[1:]) - 6
                    result = env.step(image, action)
                    box = result["box"]
                rt.torch.cuda.synchronize()
                if index > 20:
                    elapsed.append((time.perf_counter() - started) * 1000)
                overlaps.append(float(tracking_iou(box, source["gt"][index], source["valid"][index])))
                actions.append(action)
            rows.append(dict(sequence=source["sequence"], method=method, auc=success_auc(overlaps),
                             frames=len(actions), candidate_calls=sum(a > 0 for a in actions),
                             action_distribution=("native_selector:{}".format(len(actions))
                                                  if method == "sgla" else action_distribution(actions)),
                             latency_ms=float(np.mean(elapsed))))
    return rows


def evaluate_preserving_collector(collector, modules, sources, method="policy"):
    state = collector.state_dict()
    try:
        return evaluate(collector.rt, modules, sources, method)
    finally:
        collector.restore(state)


def verify_selection(torch, selected, contract_fingerprint, step, frozen, score=None, model=None):
    if (not isinstance(selected, dict) or selected.get("contract_fingerprint") != contract_fingerprint
            or selected.get("steps") != step or not isinstance(selected.get("model"), dict)
            or (score is not None and selected.get("tune_auc") != score)):
        raise ValueError("Selected policy checkpoint metadata differs from selection state")
    state = selected["model"]
    if any(name not in state or not torch.equal(value, state[name].detach().cpu())
           for name, value in frozen.items()):
        raise ValueError("Selected policy checkpoint changed frozen representation")
    if model is not None and (state.keys() != model.keys() or any(
            not torch.equal(value.detach().cpu(), state[name].detach().cpu())
            for name, value in model.items())):
        raise ValueError("Selected policy weights differ from the atomic training checkpoint")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--seed", type=int, default=2054)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--rollout_steps", type=int)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=.97)
    args = parser.parse_args()
    if any(not getattr(args, n).strip() for n in ("bundle", "manifest", "output_dir")):
        parser.error("A required path is empty")
    args.steps = args.steps if args.steps is not None else (1024 if args.mode == "smoke" else 250000)
    args.rollout_steps = args.rollout_steps if args.rollout_steps is not None else (256 if args.mode == "smoke" else 2048)
    if (args.steps < 1 or args.rollout_steps < 2 or args.minibatch_size < 1
            or not 0 <= args.gamma < 1 or (args.mode == "smoke" and args.steps > 4096)):
        parser.error("Invalid PPO configuration; smoke is capped at 4096 transitions")
    _, bundle, repdir = verify_representation_bundle(args.bundle)
    manifest = load_manifest(args.manifest)
    if manifest["fingerprint"] != bundle["source_manifest_fingerprint"]:
        raise ValueError("Manifest differs from the frozen representation")
    names = bundle["pilot_videos"] if args.mode == "smoke" else bundle["fit_videos"]
    fit = ppo_sources(manifest, names, 128)
    tune = [] if args.mode == "smoke" else load_sources(manifest, bundle["tune_videos"], 256)
    rt = Runtime(args.parameter, args.gpu, args.checkpoint)
    if rt.identity != bundle["runtime"]:
        raise ValueError("Original tracker runtime differs from representation freeze")
    torch = rt.torch
    import torch.nn as nn
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config = bundle["model"]
    modules = build_temporal_modules(torch, nn, config["embed_dim"], config["lens_z"],
                                     config["search_side"]).to(next(rt.net.parameters()).device)
    representation = load_checkpoint(torch, repdir / "best.pt")
    if (representation["contract_fingerprint"] != bundle["representation_contract_fingerprint"]
            or representation["epoch"] != bundle["best_epoch"]):
        raise ValueError("Frozen checkpoint metadata mismatch")
    load_tracking_representation(modules, representation["model"])
    parameters = configure_policy(modules)
    modules.eval()
    frozen = {n: p.detach().cpu().clone() for n, p in modules.named_parameters() if not p.requires_grad}
    initial_actor = modules.actor.weight.detach().cpu().clone()
    optimizer = torch.optim.Adam(parameters, lr=3e-4, eps=1e-5)
    collector = Collector(rt, modules, fit, args.seed)
    contract = dict(schema="stage18_onpolicy_ppo_v1", bundle_fingerprint=bundle["fingerprint"],
                    mode=args.mode, seed=args.seed, steps=args.steps, rollout_steps=args.rollout_steps,
                    minibatch_size=args.minibatch_size, gamma=args.gamma, gae_lambda=.95,
                    learning_rate=3e-4, adam_epsilon=1e-5, clip_epsilon=.2,
                    policy_initialization="fresh seed-specific policy encoder/critic; actor 0.7/0.05",
                    value_coefficient=.5, entropy_coefficient=.01, update_epochs=4, grad_clip=.5,
                    invalid_label_rule="zero reward, masked losses, no state reset",
                    clip_end_rule="bootstrap next image at artificial clip end; stop GAE trace before reset",
                    fit=[s["sequence"] for s in fit], tune=[s["sequence"] for s in tune],
                    code_sha256={n: file_hash(Path(__file__).parent / n) for n in
                                 ("stage18_onpolicy_ppo.py", "stage18_policy_runtime.py", "stage18_ppo_math.py",
                                  "stage18_freeze_representation.py")})
    contract["fingerprint"] = fingerprint(contract)
    output = Path(args.output_dir).expanduser().resolve()
    path = output / "stage18_ppo_contract.json"
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise ValueError("PPO contract changed; use a new output directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Unidentified PPO output directory")
        write_json(path, contract)
    steps, updates, history, evaluations = 0, 0, [], []
    best_score, best_latency, best_step = -float("inf"), float("inf"), None
    best_selection = None
    last_path, best_path = output / "last.pt", output / "best.pt"
    if last_path.exists():
        saved = load_checkpoint(torch, last_path)
        if saved["contract_fingerprint"] != contract["fingerprint"]:
            raise ValueError("PPO checkpoint contract mismatch")
        modules.load_state_dict(saved["model"])
        if any(not torch.equal(frozen[name], p.detach().cpu())
               for name, p in modules.named_parameters() if name in frozen):
            raise ValueError("Resumed checkpoint changed frozen representation weights")
        optimizer.load_state_dict(saved["optimizer"])
        steps, updates, history = saved["steps"], saved["updates"], saved["history"]
        evaluations = saved["evaluations"]
        best_score, best_latency, best_step = saved["best_score"], saved["best_latency"], saved["best_step"]
        best_selection = saved["best_selection"]
        if tune and best_step is not None:
            verify_selection(torch, best_selection, contract["fingerprint"], best_step, frozen, best_score)
        collector.restore(saved["collector"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    if tune and not (output / "stage18_controls.csv").exists():
        controls = []
        for method in ("fast", "sgla", "memory_fast", "l7", "l8", "l9", "l10", "l11", "l12", "random"):
            controls.extend(evaluate_preserving_collector(collector, modules, tune, method))
        write_csv(output / "stage18_controls.csv", controls)
    next_eval = (steps // 20000 + 1) * 20000
    while steps < args.steps:
        # Split at evaluation boundaries so selection occurs exactly at the
        # declared 20k intervals, not at whichever update happens to cross one.
        count = min(args.rollout_steps, args.steps - steps,
                    next_eval - steps if tune else args.rollout_steps)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        data = collector.collect(count)
        updates += 1
        metrics = update_policy(torch, modules, optimizer, data, args, updates)
        torch.cuda.synchronize()
        duration = time.perf_counter() - started
        metrics.update(collect_and_update_seconds=duration, transitions_per_second=count / duration,
                       peak_allocated_mb=torch.cuda.max_memory_allocated() / 1024**2)
        steps += count
        history.append(dict(steps=steps, update=updates, **metrics))
        if tune and (steps >= next_eval or steps == args.steps):
            rows = evaluate_preserving_collector(collector, modules, tune)
            score, latency = float(np.mean([r["auc"] for r in rows])), float(np.mean([r["latency_ms"] for r in rows]))
            evaluations.extend([dict(steps=steps, **row) for row in rows])
            feasible = all(r["candidate_calls"] <= 3 * r["frames"] // 10 for r in rows)
            if not feasible:
                raise RuntimeError("Deployment argmax exceeded P30")
            if score > best_score or (score == best_score and latency < best_latency):
                best_score, best_latency, best_step = score, latency, steps
                best_selection = dict(contract_fingerprint=contract["fingerprint"], steps=steps,
                                      model={n: v.detach().cpu().clone() for n, v in modules.state_dict().items()},
                                      tune_auc=score)
            next_eval = (steps // 20000 + 1) * 20000
            write_csv(output / "stage18_tune_policy.csv", evaluations)
        if not tune:
            best_step = steps  # Smoke never selects weights using tune outcomes.
        checkpoint = dict(contract_fingerprint=contract["fingerprint"], model=modules.state_dict(),
                          optimizer=optimizer.state_dict(), steps=steps, updates=updates, history=history,
                          evaluations=evaluations, best_score=best_score, best_latency=best_latency,
                          best_step=best_step, best_selection=best_selection,
                          collector=collector.state_dict(), torch_rng=torch.get_rng_state(),
                          cuda_rng=torch.cuda.get_rng_state_all())
        atomic_torch_save(torch, checkpoint, last_path)
        write_csv(output / "stage18_ppo_history.csv", history)
        iou_text = "n/a" if metrics["mean_valid_iou"] is None else "{:.4f}".format(metrics["mean_valid_iou"])
        print("[PPO {}] {}/{} steps, IoU={}, candidate={:.2%}, replay={:.2g}".format(
            args.mode, steps, args.steps, iou_text, metrics["candidate_ratio"],
            metrics["likelihood_replay_error"]), flush=True)
    unchanged = all(torch.equal(frozen[n], p.detach().cpu()) for n, p in modules.named_parameters() if n in frozen)
    actor_changed = not torch.equal(initial_actor, modules.actor.weight.detach().cpu())
    optimized = [r for r in history if not r["skipped_update"]]
    passed = unchanged and actor_changed and bool(optimized) and all(r["maximum_gradient_norm"] > 0 for r in optimized)
    selected_path = best_path if tune else last_path
    if tune:
        verify_selection(torch, best_selection, contract["fingerprint"], best_step, frozen, best_score)
        if not best_path.exists():
            atomic_torch_save(torch, best_selection, best_path)
    selected = load_checkpoint(torch, selected_path)
    verify_selection(torch, selected, contract["fingerprint"], best_step, frozen,
                     best_score if tune else None, best_selection["model"] if tune else modules.state_dict())
    representation_decision = json.loads((repdir / "stage18_representation_decision.json").read_text())
    decision = dict(schema="stage18_onpolicy_ppo_v1", contract_fingerprint=contract["fingerprint"],
                    status=("PASS_ONPOLICY_SMOKE" if args.mode == "smoke" else "COMPLETE_ONPOLICY_TRAINING")
                    if passed else "FAIL_ONPOLICY_INTEGRITY", steps=steps, updates=updates,
                    optimized_rollouts=len(optimized), skipped_rollouts=len(history) - len(optimized),
                    frozen_representation_unchanged=unchanged, actor_changed=actor_changed,
                    best_step=best_step, best_tune_auc=best_score if tune else None,
                    maximum_likelihood_replay_error=max(r["likelihood_replay_error"] for r in history),
                    collect_and_update_seconds=sum(r["collect_and_update_seconds"] for r in history),
                    peak_allocated_mb=max(r["peak_allocated_mb"] for r in history),
                    representation_resource_audit=representation_decision["resource_audit"],
                    resource_target_met=None,
                    resource_role="No policy speed claim tested here; see separate representation audit",
                    checkpoint=str(selected_path), checkpoint_sha256=file_hash(selected_path),
                    warning="Development only. Smoke is an optimizer/runtime check, not a policy-gain result.")
    decision["fingerprint"] = fingerprint(decision)
    write_json(output / "stage18_ppo_decision.json", decision)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
