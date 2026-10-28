"""Matched stop-gradient/value-aligned L5-L6 training; no PPO in Stage-26."""

import argparse
from pathlib import Path
import random
import time

import numpy as np

from stage4_common import file_hash, tracking_iou, write_cache, write_csv, write_json
from stage4_runtime import Runtime, add_runtime_arguments
from stage18_representation_train import atomic_torch_save, load_checkpoint, load_sources
from stage23_branch_ppo import check_runtime, model_digest, single_writer
from stage23_development import build_model, cached_evaluation, read_sealed, sealed, valid_auc
from stage25_relation_common import keyed_seed
from stage26_value_common import (ARMS, CONTROLS, HEAD_LR, PREFIX_LR, SCHEMA, SEEDS, WEIGHT_DECAY,
    example_schedule, make_contract, planned_jobs, source_lineage, verify_contract)
from stage26_value_runtime import (build_value_head, collect_example, current_losses, trainable_prefix, value_environment)


def cpu_state(module):
    return {n: p.detach().cpu().clone() for n, p in module.state_dict().items()}


def prepare_system(rt, bundle, repdir, seed, original_prefix):
    prefix = trainable_prefix(rt)
    prefix.load_state_dict(original_prefix)
    modules = build_model(rt, bundle, repdir, seed).eval().requires_grad_(False)
    head = build_value_head(rt.torch, rt.identity["embed_dim"]).to(next(prefix.parameters()).device).eval()
    optimizer = rt.torch.optim.AdamW([
        dict(params=list(prefix.parameters()), lr=PREFIX_LR), dict(params=list(head.parameters()), lr=HEAD_LR)
    ], weight_decay=WEIGHT_DECAY)
    return prefix, modules, head, optimizer


def job_list(contract, sources, seed):
    if [s["sequence"] for s in sources] != contract["training"] or any(
            len(s["paths"]) - 1 != contract["clip_steps"][s["sequence"]] for s in sources):
        raise ValueError("Loaded fit source bounds/identities differ from frozen plan")
    return planned_jobs(contract, seed)


def frozen_parameters(rt):
    return {n: p.detach().cpu().clone() for n, p in rt.net.state_dict().items()
            if not n.startswith(("backbone.blocks.4.", "backbone.blocks.5."))}


def verify_frozen_runtime(torch, rt, frozen, modules, memory_id):
    state = rt.net.state_dict()
    if any(not torch.equal(state[n].detach().cpu(), value) for n, value in frozen.items()):
        raise RuntimeError("Runtime parameters/buffers outside L5-L6 changed")
    if model_digest(modules) != memory_id:
        raise RuntimeError("Frozen Stage-18 tracking memory/adapters changed")


def train_arm(rt, bundle, repdir, sources, arm, seed, directory, contract, original_prefix):
    torch = rt.torch
    prefix, modules, head, optimizer = prepare_system(rt, bundle, repdir, seed, original_prefix)
    initial_prefix, initial_head, memory_id = model_digest(prefix), model_digest(head), model_digest(modules)
    frozen = frozen_parameters(rt)
    directory.mkdir(parents=True, exist_ok=True)
    jobs, by_name = job_list(contract, sources, seed), {s["sequence"]: s for s in sources}
    last, cursor, history = directory / "last.pt", 0, []
    identity = dict(contract_fingerprint=contract["fingerprint"], seed=seed, arm=arm,
                    initial_prefix_sha256=initial_prefix, initial_head_sha256=initial_head)
    if last.exists():
        saved = load_checkpoint(torch, last)
        if saved.get("identity") != identity:
            raise ValueError("Checkpoint belongs to another arm/seed/contract")
        cursor, history = saved["cursor"], saved["history"]
        if type(cursor) is not int or not 0 <= cursor <= len(jobs) or len(history) != cursor:
            raise ValueError("Invalid atomic training cursor")
        prefix.load_state_dict(saved["prefix"])
        head.load_state_dict(saved["head"])
        optimizer.load_state_dict(saved["optimizer"])
        for i, row in enumerate(history):
            if file_hash(directory / "samples" / ("{:05d}.npz".format(i))) != row["sample_sha256"]:
                raise ValueError("Committed current-version branch sample changed")
        torch.set_rng_state(saved["torch_rng"].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([v.cpu() for v in saved["cuda_rng"]])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        if history and (model_digest(prefix) != history[-1]["prefix_after_sha256"] or
                        model_digest(head) != history[-1]["head_after_sha256"]):
            raise ValueError("Checkpoint differs from its committed update history")
    for index in range(cursor, len(jobs)):
        name, round_index, anchor, actions = jobs[index]
        started = time.perf_counter()
        arrays, metadata, context = collect_example(rt, modules, head, prefix, by_name[name], actions,
                                                   anchor, seed, round_index)
        metadata.update(identity=identity, attempt=index)
        loss, metrics = current_losses(rt, modules, head, prefix, context, arrays, metadata, arm)
        if metadata["eligible"]:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(list(prefix.parameters()) + list(head.parameters()), 1.)
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite representation/value gradient")
            optimizer.step()
            optimizer_steps = 1
        else:
            optimizer_steps, norm = 0, 0.
        verify_frozen_runtime(torch, rt, frozen, modules, memory_id)
        sample = directory / "samples" / ("{:05d}.npz".format(index))
        write_cache(sample, arrays, metadata)
        row = dict(attempt=index, sequence=name, round=round_index, anchor=anchor,
            eligible=metadata["eligible"], optimizer_steps=optimizer_steps,
            branch_frames=metadata["branch_frames"], prefix_frames=metadata["prefix_frames"],
            replay_frames=metadata["replay_frames"], prefix_before_sha256=metadata["teacher_prefix_sha256"],
            head_before_sha256=metadata["teacher_head_sha256"], prefix_after_sha256=model_digest(prefix),
            head_after_sha256=model_digest(head), sample_sha256=file_hash(sample),
            full_h20_replay_error=float(np.abs(arrays["live_boxes"] - arrays["replay_boxes"]).max()),
            maximum_gradient_norm=float(norm), seconds=time.perf_counter() - started, **metrics)
        history.append(row)
        atomic_torch_save(torch, dict(identity=identity, cursor=index + 1, history=history,
            prefix=cpu_state(prefix), head=cpu_state(head), optimizer=optimizer.state_dict(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            numpy_rng=np.random.get_state(), python_rng=random.getstate()), last)
        write_csv(directory / "history.csv", history)
        print("[seed{} {}] {}/{}; value_grad={:.3g}; track_grad={:.3g}; replay={:.2g}".format(
            seed, arm, index + 1, len(jobs), metrics["rank_prefix_gradient"], metrics["tracking_prefix_gradient"],
            row["full_h20_replay_error"]), flush=True)
    write_csv(directory / "history.csv", history)
    verify_frozen_runtime(torch, rt, frozen, modules, memory_id)
    result = sealed(dict(schema=SCHEMA, **identity, attempts=len(jobs),
        optimizer_steps=sum(r["optimizer_steps"] for r in history),
        branch_frames=sum(r["branch_frames"] for r in history),
        maximum_rank_prefix_gradient=max(r["rank_prefix_gradient"] for r in history),
        maximum_tracking_prefix_gradient=max(r["tracking_prefix_gradient"] for r in history),
        maximum_replay_error=max(max(r["full_h20_replay_error"], r["differentiable_replay_error"]) for r in history),
        prefix_changed=model_digest(prefix) != initial_prefix, head_changed=model_digest(head) != initial_head,
        frozen_runtime_and_memory_unchanged=True, frozen_memory_sha256=memory_id,
        final_prefix_sha256=model_digest(prefix), final_head_sha256=model_digest(head),
        checkpoint_sha256=file_hash(last), history_sha256=file_hash(directory / "history.csv")))
    write_json(directory / "training_complete.json", result)
    return prefix, modules, head, result


def evaluate(rt, modules, head, source, method):
    env = value_environment(rt, modules, head)
    env.reset(rt.read(source["paths"][0]), source["gt"][0])
    actions, boxes, overlaps = [], [], []
    generator = np.random.default_rng(keyed_seed(source["sequence"], "stage26_check_random"))
    for index, path in enumerate(source["paths"][1:], 1):
        action = None
        if method != "value":
            action = 0
            if method != "memory_fast" and min(20, env.budget.credit + 3) >= 10:
                action = int(generator.integers(1, 7)) if method == "random" else int(method[1:]) - 6
        result = env.step(rt.read(path), action=action)
        actions.append(result["action"])
        boxes.append(np.asarray(result["box"]).tolist())
        overlaps.append(float(tracking_iou(result["box"], source["gt"][index], bool(source["valid"][index]))))
    valid = source["valid"][1:].astype(bool)
    costs = (np.asarray(actions) > 0).cumsum()
    expected = 3 * np.arange(1, len(actions) + 1) // 10
    if not np.array_equal(costs, np.zeros_like(costs) if method == "memory_fast" else expected):
        raise RuntimeError("Checking trajectories did not use identical canonical call positions")
    return dict(sequence=source["sequence"], method=method, actions=actions, boxes=boxes,
                overlaps=overlaps, valid=valid.tolist(), frames=len(actions), candidate_blocks=int(costs[-1]),
                auc=valid_auc(overlaps, valid))


def check_branches(rt, modules, head, prefix, source, seed):
    actions, anchors = example_schedule(len(source["paths"]) - 1, seed, source["sequence"], 0)
    qs, scores, eligibility, replay = [], [], [], []
    for anchor in anchors:
        arrays, metadata, context = collect_example(rt, modules, head, prefix, source, actions, int(anchor), seed, 0)
        qs.append(arrays["target"])
        scores.append(context["actual"]["candidate_values"].cpu().numpy()[0])
        eligibility.append(metadata["eligible"])
        replay.append(float(np.abs(arrays["live_boxes"] - arrays["replay_boxes"]).max()))
    return dict(sequence=source["sequence"], anchors=anchors.tolist(), post_values=np.asarray(qs).tolist(),
                predicted_values=np.asarray(scores).tolist(), eligible=eligibility, maximum_replay_error=max(replay))


def checking_evaluations(rt, modules, head, prefix, sources, seed, directory, contract, policy_id):
    references = []
    for source in sources:
        for method in CONTROLS + ("branches",):
            path = directory / "checking" / method / (source["sequence"] + ".json")
            identity = dict(contract_fingerprint=contract["fingerprint"], policy_sha256=policy_id,
                            sequence=source["sequence"], method=method)
            if path.exists():
                value = read_sealed(path)
                if any(value.get(k) != v for k, v in identity.items()):
                    raise ValueError("Cached checking result belongs to another final model")
            else:
                record = (check_branches(rt, modules, head, prefix, source, seed) if method == "branches" else
                          evaluate(rt, modules, head, source, method))
                value = sealed(dict(record, **{k: v for k, v in identity.items() if k not in record}))
                write_json(path, value)
            references.append(dict(sequence=source["sequence"], method=method,
                                   file=path.relative_to(directory).as_posix(), sha256=file_hash(path)))
    return references


def verify_smoke(directory, bundle_id, groups):
    path = Path(directory).expanduser().resolve()
    smoke = verify_contract(path)
    decision = read_sealed(path / "stage26_smoke_decision.json")
    if (smoke["mode"] != "smoke" or smoke["bundle_fingerprint"] != bundle_id or smoke["groups"] != groups
            or decision["status"] != "PASS_VALUE_REPRESENTATION_SMOKE"
            or decision["contract_fingerprint"] != smoke["fingerprint"]):
        raise ValueError("Require the matching passed A/B gradient/replay smoke")
    from stage26_value_summary import summarize_smoke
    if summarize_smoke(path, smoke) != decision:
        raise ValueError("Passed smoke decision no longer reproduces from actual training artifacts")
    for row in decision["arms"]:
        if file_hash(path / ("seed" + str(SEEDS[0])) / row["arm"] / "last.pt") != row["checkpoint_sha256"]:
            raise ValueError("Passed smoke checkpoint changed")
    return decision["fingerprint"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    for name in ("bundle", "manifest", "stage25_root", "output_dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--smoke_dir")
    parser.add_argument("--seed", choices=SEEDS, type=int, default=SEEDS[0])
    args = parser.parse_args()
    if any(not (getattr(args, n) or "").strip() for n in ("bundle", "manifest", "stage25_root", "output_dir", "checkpoint")):
        parser.error("Explicit nonempty frozen inputs/output/checkpoint required")
    if args.mode == "smoke" and args.seed != SEEDS[0]:
        parser.error("Smoke uses the first predeclared seed only")
    if args.mode == "full" and not (args.smoke_dir or "").strip():
        parser.error("--smoke_dir is required before the full 32/16 pilot")
    bundle, repdir, manifest, groups, source_id = source_lineage(args.bundle, args.manifest, args.stage25_root)
    smoke_id = verify_smoke(args.smoke_dir, bundle["fingerprint"], groups) if args.mode == "full" else None
    output = Path(args.output_dir).expanduser().resolve()
    protected = [repdir, Path(args.stage25_root).expanduser().resolve()]
    if args.smoke_dir:
        protected.append(Path(args.smoke_dir).expanduser().resolve())
    for original in protected:
        if output == original or output.is_relative_to(original) or original.is_relative_to(output):
            parser.error("Use a new separate output root")
    with single_writer(output):
        rt = Runtime(args.parameter, args.gpu, args.checkpoint)
        check_runtime(rt.identity, bundle["runtime"])
        rt.torch.backends.cudnn.benchmark = False
        rt.torch.backends.cudnn.deterministic = True
        contract = make_contract(bundle, manifest, groups, source_id, rt.identity, args.mode)
        contract = sealed({**{k: v for k, v in contract.items() if k != "fingerprint"}, "passed_smoke_fingerprint": smoke_id})
        path = output / "stage26_contract.json"
        if path.exists():
            if read_sealed(path) != contract:
                raise ValueError("Stage-26 plan/code changed; do not overwrite the existing run")
        elif any(p.name != ".stage23_writer.lock" for p in output.iterdir()):
            raise ValueError("Unidentified outputs; use an empty new root")
        else:
            write_json(path, contract)
        original_prefix = cpu_state(trainable_prefix(rt))
        sources = load_sources(manifest, contract["training"], contract["maximum_frames"])
        checks = load_sources(manifest, contract["checking"], 128) if args.mode == "full" else []
        print("Stage-26 {}: {} train / {} check videos; {} fresh updates per arm; no PPO.".format(
            args.mode, len(sources), len(checks), contract["expected_attempts"]), flush=True)
        if checks:
            dummy = build_model(rt, bundle, repdir, args.seed).requires_grad_(False)
            references = {method: cached_evaluation(rt, dummy, checks, method, output / "original_controls" / method,
                          contract["fingerprint"], bundle["fingerprint"]) for method in ("fast", "sgla")}
            write_json(output / "original_controls.json", sealed(dict(contract_fingerprint=contract["fingerprint"], references=references)))
        arms = ARMS if SEEDS.index(args.seed) % 2 == 0 else ARMS[::-1]
        for arm in arms:
            directory = output / ("seed" + str(args.seed)) / arm
            prefix, modules, head, training = train_arm(rt, bundle, repdir, sources, arm, args.seed, directory, contract, original_prefix)
            refs = checking_evaluations(rt, modules, head, prefix, checks, args.seed, directory, contract,
                                        training["checkpoint_sha256"]) if checks else []
            write_json(directory / "complete.json", sealed(dict(schema=SCHEMA, contract_fingerprint=contract["fingerprint"],
                seed=args.seed, arm=arm, training=training, evaluations=refs)))
        if args.mode == "smoke":
            from stage26_value_summary import summarize_smoke
            result = summarize_smoke(output, contract)
            write_json(output / "stage26_smoke_decision.json", result)
            print("Decision: " + result["status"])
        else:
            print("Seed complete. Finish all three paired seeds before inspecting efficacy.")


if __name__ == "__main__":
    main()
