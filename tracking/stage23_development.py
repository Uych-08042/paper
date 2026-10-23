"""Fixed-budget, three-seed Stage-23B development experiment after passed smoke."""

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np

from stage4_common import file_hash, fingerprint, load_manifest, tracking_iou, write_csv, write_json
from stage4_runtime import Runtime, add_runtime_arguments
from stage18_freeze_representation import verify_representation_bundle
from stage18_onpolicy_ppo import action_distribution, ppo_sources
from stage18_policy_runtime import configure_policy, load_tracking_representation, policy_forward
from stage18_ppo_math import gae, normalize_advantages, ppo_objective
from stage18_representation_train import atomic_torch_save, load_checkpoint, load_sources
from stage18_temporal_model import build_temporal_modules
from stage21_counterfactual_math import expected_clipped_loss
from stage21_counterfactual_runtime import CoupledEnvironment
from stage23_branch_credit import ARMS, ANCHORS, GAMMA, HORIZON, REPEATS, BranchCollector, branch_credit
from stage23_branch_ppo import check_runtime, model_digest, single_writer, verify_frozen


SEEDS = (2074, 2075, 2076)
STEPS, ROLLOUT = 32768, 256
CONTROLS = ("fast", "sgla", "memory_fast", "l7", "l8", "l9", "l10", "l11", "l12", "random")
SCHEMA = "stage23b_fixed_budget_development_v1"
METRIC = "valid_tracked_strict_21_threshold_auc_excludes_initial_v1"


def sealed(value):
    return dict(value, fingerprint=fingerprint(value))


def read_sealed(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("fingerprint") != fingerprint(
            {k: v for k, v in value.items() if k != "fingerprint"}):
        raise ValueError("Invalid report fingerprint: " + str(path))
    return value


def verify_smoke(directory, bundle_fingerprint):
    directory = Path(directory).expanduser().resolve()
    contract = read_sealed(directory / "stage23_contract.json")
    decision = read_sealed(directory / "stage23_decision.json")
    if (contract.get("schema") != "stage23_branch_credit_smoke_v1"
            or contract.get("bundle_fingerprint") != bundle_fingerprint
            or decision.get("status") != "PASS_BRANCH_PPO_SMOKE"
            or decision.get("contract_fingerprint") != contract["fingerprint"]):
        raise ValueError("A matching, completed Stage-23 smoke is required")
    code = contract.get("code_sha256", {})
    required = {"stage23_branch_credit.py", "stage23_branch_ppo.py", "stage21_counterfactual_runtime.py",
                "stage21_counterfactual_math.py", "stage18_environment.py", "stage18_policy_runtime.py"}
    if not required <= set(code):
        raise ValueError("Smoke evidence is missing critical code hashes")
    for name, digest in code.items():
        if Path(name).name != name or "\\" in name or file_hash(Path(__file__).parent / name) != digest:
            raise ValueError("Passed smoke code changed: " + name)
    rows = decision["arms"]
    if len(rows) != 3 or {r["arm"] for r in rows} != set(ARMS):
        raise ValueError("Smoke is missing a matched arm")
    if len({r["initial_model_sha256"] for r in rows}) != 1:
        raise ValueError("Smoke initializations were not matched")
    for row in rows:
        if (row["live_transitions"] != 1024 or row["branch_transitions"] != 4480
                or row["optimizer_steps"] != 16 or not row["actor_changed"]
                or not row["frozen_representation_unchanged"] or row["maximum_replay_error"] > 1e-4
                or file_hash(directory / row["arm"] / "last.pt") != row["checkpoint_sha256"]):
            raise ValueError("Smoke artifacts/counts do not match their passed report")
    return dict(contract_fingerprint=contract["fingerprint"], decision_fingerprint=decision["fingerprint"],
                rough_training_seconds=sum(r["seconds"] for r in rows) * (STEPS / 1024) * len(SEEDS))


def valid_auc(overlaps, valid):
    x, valid = np.asarray(overlaps, float), np.asarray(valid)
    if (x.ndim != 1 or x.shape != valid.shape or valid.dtype != np.bool_ or not valid.any()
            or not np.isfinite(x).all() or np.any((x[valid] < 0) | (x[valid] > 1))):
        raise ValueError("Invalid tracked-frame overlap/visibility vectors")
    return float(100 * (x[valid, None] > np.arange(21)[None] * .05).mean())


def evaluate_video(rt, modules, source, method):
    """One chronological deployment trajectory; no branches or label-driven resets."""
    if method not in CONTROLS + ("policy",):
        raise ValueError("Unknown deployment method")
    env = CoupledEnvironment(rt, modules)
    env.reset(rt.read(source["paths"][0]), source["gt"][0])
    rng = np.random.default_rng(int(fingerprint(source["sequence"])[:16], 16))
    boxes, overlaps, actions, elapsed = [], [], [], []
    cuda = rt.torch.cuda.is_available()
    with rt.torch.no_grad():
        for index, path in enumerate(source["paths"][1:], 1):
            image = rt.read(path)
            if cuda:
                rt.torch.cuda.synchronize()
            start = time.perf_counter()
            if method in ("fast", "sgla"):
                box = rt.run_image(image, method)
                rt.tracker.state, rt.tracker.frame_id = list(box), index
                action = 0 if method == "fast" else -1
            else:
                action = None
                if method != "policy":
                    action = 0
                    if method != "memory_fast" and min(20, env.budget.credit + 3) >= 10:
                        action = int(rng.integers(1, 7)) if method == "random" else int(method[1:]) - 6
                result = env.step(image, action=action)  # uniform=None is deterministic argmax.
                box, action = result["box"], result["action"]
            if cuda:
                rt.torch.cuda.synchronize()
            if index > 20:
                elapsed.append((time.perf_counter() - start) * 1000)
            boxes.append(np.asarray(box, float).tolist())
            overlaps.append(float(tracking_iou(box, source["gt"][index], bool(source["valid"][index]))))
            actions.append(action)
    valid = source["valid"][1:].astype(bool)
    candidate = np.ones(len(actions), bool) if method == "sgla" else np.asarray(actions) > 0
    if method not in ("fast", "sgla") and np.any(candidate.cumsum() > 3 * np.arange(1, len(actions) + 1) // 10):
        raise RuntimeError("Deployment exceeded a causal P30 prefix budget")
    return dict(sequence=source["sequence"], method=method, metric=METRIC, frames=len(actions),
                valid_frames=int(valid.sum()), auc=valid_auc(overlaps, valid), boxes=boxes,
                overlaps=overlaps, valid=valid.tolist(), actions=actions,
                candidate_blocks=int(candidate.sum()), mean_crop_model_decode_ms=float(np.mean(elapsed)))


def cached_evaluation(rt, modules, sources, method, directory, contract_id, policy_id):
    directory.mkdir(parents=True, exist_ok=True)
    references = []
    for source in sources:
        path = directory / (source["sequence"] + ".json")
        identity = dict(contract_fingerprint=contract_id, policy_sha256=policy_id,
                        sequence=source["sequence"], method=method)
        if path.exists():
            record = read_sealed(path)
            if any(record.get(k) != v for k, v in identity.items()):
                raise ValueError("Cached evaluation belongs to another policy: " + str(path))
            # Audit cached labels/rewards against the current frozen source without rerunning the model.
            recomputed = [float(tracking_iou(box, gt, bool(v))) for box, gt, v in zip(
                record["boxes"], source["gt"][1:], source["valid"][1:])]
            if (len(record["boxes"]) != len(source["paths"]) - 1
                    or record["valid"] != source["valid"][1:].tolist()
                    or not np.allclose(recomputed, record["overlaps"], rtol=0, atol=1e-12)):
                raise ValueError("Cached evaluation no longer matches its source")
        else:
            record = sealed(dict(evaluate_video(rt, modules, source, method), **{
                k: v for k, v in identity.items() if k not in ("sequence", "method")}))
            write_json(path, record)
        references.append(dict(sequence=source["sequence"], file=str(path), sha256=file_hash(path)))
    return references


def build_model(rt, bundle, repdir, seed):
    torch = rt.torch
    import torch.nn as nn
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    config = bundle["model"]
    modules = build_temporal_modules(torch, nn, config["embed_dim"], config["lens_z"],
                                     config["search_side"]).to(next(rt.net.parameters()).device)
    checkpoint = load_checkpoint(torch, repdir / "best.pt")
    if (checkpoint["epoch"] != bundle["best_epoch"]
            or checkpoint["contract_fingerprint"] != bundle["representation_contract_fingerprint"]):
        raise ValueError("Frozen representation checkpoint metadata changed")
    load_tracking_representation(modules, checkpoint["model"])
    configure_policy(modules)
    return modules.eval()


def update_policy(torch, modules, optimizer, data, arm, seed, update):
    """Same smoke objectives; zero valid credit means zero auxiliary, not fabricated labels."""
    advantage, returns = gae(data["rewards"], data["values"], data["next_values"],
                             data["terminated"], data["truncated"], GAMMA, .95)
    valid_live = bool(data["valid"].any())
    advantage = normalize_advantages(advantage, data["valid"]) if valid_live else np.zeros_like(advantage)
    anchors = data["counterfactual"]
    cf_valid = np.array([a["valid"] for a in anchors], bool)
    old_p = np.stack([a["old_probabilities"] for a in anchors])
    credit = branch_credit(np.stack([a["rewards"] for a in anchors]), old_p, cf_valid, arm)
    device = next(modules.parameters()).device
    tensor = lambda a, dtype: torch.as_tensor(a, dtype=dtype, device=device)
    obs = {k: v.to(device) for k, v in data["observation"].items()}
    anchor_index = np.array([a["index"] for a in anchors])
    anchor_obs = {k: v[anchor_index] for k, v in obs.items()}
    action, old_logp = tensor(data["actions"], torch.long), tensor(data["old_log_prob"], torch.float32)
    p, a, valid = tensor(old_p, torch.float32), tensor(credit["advantage"], torch.float32), tensor(cf_valid, torch.bool)
    with torch.no_grad():
        logits, _ = policy_forward(torch, modules, obs)
        replay_error = float((torch.distributions.Categorical(logits=logits).log_prob(action) - old_logp).abs().max())
        logits, _ = policy_forward(torch, modules, anchor_obs)
        probability_error = float((torch.softmax(logits, -1) - p).abs().max())
    if max(replay_error, probability_error) > 1e-4:
        raise RuntimeError("Development old-policy likelihoods do not replay")
    rows, maximum_norm = [], 0.
    for epoch in range(4 if valid_live else 0):
        order = np.random.default_rng(seed + 1000 * update + epoch).permutation(len(advantage))
        logits, value = policy_forward(torch, modules, {k: v[order] for k, v in obs.items()})
        loss, metrics = ppo_objective(torch, logits, value, action[order], old_logp[order],
                                      tensor(returns[order], torch.float32), tensor(advantage[order], torch.float32),
                                      tensor(data["valid"][order], torch.bool))
        logits, _ = policy_forward(torch, modules, anchor_obs)
        auxiliary = expected_clipped_loss(torch, logits, p, a, valid)
        gradients = torch.autograd.grad(auxiliary, tuple(modules.actor.parameters()), retain_graph=True)
        cf_norm = float(torch.sqrt(sum(g.detach().square().sum() for g in gradients)))
        loss = loss + credit["weight"] * auxiliary
        if not torch.isfinite(loss) or not np.isfinite(cf_norm):
            raise RuntimeError("Nonfinite PPO branch loss/gradient")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in modules.parameters() if p.requires_grad], .5)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite PPO gradient")
        maximum_norm = max(maximum_norm, float(norm))
        optimizer.step()
        rows.append(dict(**{k: float(v) for k, v in metrics.items()}, auxiliary_loss=float(auxiliary.detach()),
                         auxiliary_actor_gradient_norm=cf_norm))
    result = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else dict(
        policy_loss=0., value_loss=0., entropy=0., approx_kl=0., clip_fraction=0.,
        auxiliary_loss=0., auxiliary_actor_gradient_norm=0.)
    result.update(optimizer_steps=len(rows), valid_transitions=int(data["valid"].sum()),
                  valid_anchors=int(cf_valid.sum()), anchor_count=len(anchors), credit_scale=credit["scale"],
                  likelihood_replay_error=replay_error, probability_replay_error=probability_error,
                  chosen_replay_error=max(a["chosen_replay_error"] for a in anchors), maximum_gradient_norm=maximum_norm,
                  branch_transitions=sum(a["simulation_transitions"] for a in anchors),
                  branch_candidate_blocks=sum(a["candidate_calls"] for a in anchors),
                  bootstrap_prefix_calls=data["bootstrap_prefix_calls"], live_candidate_blocks=int((data["actions"] > 0).sum()),
                  candidate_ratio=float((data["actions"] > 0).mean()), action_distribution=action_distribution(data["actions"]),
                  mean_valid_iou=float(data["rewards"][data["valid"]].mean()) if valid_live else None)
    return result


def train_arm(rt, bundle, repdir, fit, tune, arm, seed, output, contract):
    torch = rt.torch
    modules = build_model(rt, bundle, repdir, seed)
    frozen = {n: p.detach().cpu().clone() for n, p in modules.named_parameters() if not p.requires_grad}
    initial = model_digest(modules)
    initial_actor = {n: p.detach().cpu().clone() for n, p in modules.actor.named_parameters()}
    optimizer = torch.optim.Adam([p for p in modules.parameters() if p.requires_grad], lr=3e-4, eps=1e-5)
    collector = BranchCollector(rt, modules, fit, seed)
    output.mkdir(parents=True, exist_ok=True)
    last = output / "last.pt"
    steps, history = 0, []
    if last.exists():
        saved = load_checkpoint(torch, last)
        if any(saved.get(k) != v for k, v in dict(contract_fingerprint=contract["fingerprint"],
                                                 arm=arm, seed=seed, initial_model_sha256=initial).items()):
            raise ValueError("Development checkpoint identity mismatch")
        modules.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        steps, history = saved["steps"], saved["history"]
        if steps % ROLLOUT or not 0 <= steps <= STEPS or len(history) != steps // ROLLOUT:
            raise ValueError("Invalid atomic development progress")
        collector.restore(saved["collector"])
        if collector.total_steps != steps:
            raise ValueError("Collector progress differs from checkpoint")
        torch.set_rng_state(saved["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
    verify_frozen(torch, modules, frozen)
    while steps < STEPS:
        torch.cuda.synchronize()
        start = time.perf_counter()
        data = collector.collect(ROLLOUT)
        metrics = update_policy(torch, modules, optimizer, data, arm, seed, steps // ROLLOUT + 1)
        verify_frozen(torch, modules, frozen)
        torch.cuda.synchronize()
        steps += ROLLOUT
        history.append(dict(seed=seed, arm=arm, steps=steps, seconds=time.perf_counter() - start, **metrics))
        atomic_torch_save(torch, dict(contract_fingerprint=contract["fingerprint"], arm=arm, seed=seed,
            initial_model_sha256=initial, steps=steps, history=history, model=modules.state_dict(),
            optimizer=optimizer.state_dict(), collector=collector.state_dict(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(), python_rng=random.getstate()), last)
        write_csv(output / "history.csv", history)
        print("[seed{} {}] {}/{} live; candidate={:.1%}; valid anchors={}; replay={:.2g}".format(
            seed, arm, steps, STEPS, metrics["candidate_ratio"], metrics["valid_anchors"],
            metrics["likelihood_replay_error"]), flush=True)
    write_csv(output / "history.csv", history)
    policy_id = file_hash(last)
    evaluations = cached_evaluation(rt, modules, tune, "policy", output / "evaluation", contract["fingerprint"], policy_id)
    summary = sealed(dict(schema=SCHEMA, contract_fingerprint=contract["fingerprint"], arm=arm, seed=seed,
        steps=steps, initial_model_sha256=initial, final_model_sha256=model_digest(modules), checkpoint_sha256=policy_id,
        actor_changed=any(not torch.equal(initial_actor[n], p.detach().cpu()) for n, p in modules.actor.named_parameters()),
        frozen_representation_unchanged=True, evaluations=evaluations, history_sha256=file_hash(output / "history.csv")))
    write_json(output / "complete.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    for name in ("bundle", "manifest", "smoke_dir", "output_dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    args = parser.parse_args()
    if any(not (getattr(args, n) or "").strip() for n in ("bundle", "manifest", "smoke_dir", "output_dir", "checkpoint")):
        parser.error("Explicit nonempty frozen-input, smoke and output paths are required")
    _, bundle, repdir = verify_representation_bundle(args.bundle)
    smoke = verify_smoke(args.smoke_dir, bundle["fingerprint"])
    manifest = load_manifest(args.manifest)
    if manifest["fingerprint"] != bundle["source_manifest_fingerprint"]:
        raise ValueError("Source manifest differs from frozen bundle")
    fit_names, tune_names = bundle["fit_videos"], bundle["tune_videos"]
    if (len(fit_names) != 96 or len(set(fit_names)) != 96 or len(tune_names) != 24
            or len(set(tune_names)) != 24 or set(fit_names) & set(tune_names)):
        raise ValueError("Expected the frozen 96-fit / 24-tune video split")
    fit, tune = ppo_sources(manifest, fit_names, 128), load_sources(manifest, tune_names, 256)
    output = Path(args.output_dir).expanduser().resolve()
    with single_writer(output):
        rt = Runtime(args.parameter, args.gpu, args.checkpoint)
        check_runtime(rt.identity, bundle["runtime"])
        code = read_sealed(Path(args.smoke_dir).expanduser().resolve() / "stage23_contract.json")["code_sha256"]
        contract = sealed(dict(schema=SCHEMA, bundle_fingerprint=bundle["fingerprint"], manifest_fingerprint=manifest["fingerprint"],
            smoke=smoke, runtime=rt.identity, seeds=list(SEEDS), arms=list(ARMS), primary="ppo_cf_h20",
            fit=list(fit_names), tune=list(tune_names), validation=[], live_steps=STEPS, rollout_steps=ROLLOUT,
            anchors=ANCHORS, branch_horizon=HORIZON, repeats=REPEATS, gamma=GAMMA, auxiliary_weight=.5,
            fit_clip_maximum=128, tune_clip_maximum=256, checkpoint_rule="final_only_no_early_stopping",
            deployment="deterministic_argmax_under_causal_P30", metric=METRIC, controls=list(CONTROLS),
            bootstrap_samples=5000, analysis_seed=2077,
            inference="average paired seeds within video before video bootstrap; seed-video bootstrap also reported",
            credit_rule="both H20-GAE and H20-H1 video-CI lows >0 and all three seed mean deltas >0",
            invalid_rule="zero invalid rewards and masked loss; no valid anchors gives zero auxiliary; no valid live labels skips update",
            optimizer=dict(lr=.0003, eps=.00001, epochs=4, clip=.2, grad_clip=.5, gae_lambda=.95,
                           value_weight=.5, entropy_weight=.01),
            source_code=code, code_sha256={n: file_hash(Path(__file__).parent / n) for n in (
                "stage23_development.py", "stage23_development_summary.py")}))
        path = output / "stage23b_contract.json"
        if path.exists():
            if read_sealed(path) != contract:
                raise ValueError("Development contract changed; use a new output directory")
        else:
            if any(p.name != ".stage23_writer.lock" for p in output.iterdir()):
                raise ValueError("Unidentified development outputs")
            write_json(path, contract)
        print("Frozen plan: 3 seeds x 3 arms x {} live frames; no best-checkpoint selection.".format(STEPS), flush=True)
        modules = build_model(rt, bundle, repdir, SEEDS[0])
        control_refs = {}
        for method in CONTROLS:
            control_refs[method] = cached_evaluation(rt, modules, tune, method, output / "controls" / method,
                                                    contract["fingerprint"], bundle["fingerprint"])
        write_json(output / "controls_complete.json", sealed(dict(contract_fingerprint=contract["fingerprint"],
                                                                  evaluations=control_refs)))
        del modules
        offset = SEEDS.index(args.seed)
        for arm in ARMS[offset:] + ARMS[:offset]:
            train_arm(rt, bundle, repdir, fit, tune, arm, args.seed, output / ("seed" + str(args.seed)) / arm, contract)
        print("Seed {} complete. Run all three fixed seeds before making a method comparison.".format(args.seed), flush=True)


if __name__ == "__main__":
    main()
