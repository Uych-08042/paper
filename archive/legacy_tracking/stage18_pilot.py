"""Stage-18A execution pilot only: no learned-policy efficacy decision."""

import argparse
import copy
import time
from pathlib import Path

import numpy as np

from stage4_common import (file_hash, fingerprint, frame_path, load_annotations,
                           load_manifest, write_csv, write_json)
from stage4_runtime import Runtime, add_runtime_arguments
from stage18_environment import TemporalEnvironment
from stage18_temporal_model import build_temporal_modules


def select_fit_pilot(manifest, videos, frames, seed):
    eligible = sorted([e for e in manifest["sequences"]
                       if e["split"] == "train" and e["stop"] - e["start"] - 1 >= frames],
                      key=lambda e: e["sequence"])
    order = np.random.default_rng(seed).permutation(len(eligible))
    tune_count = max(1, int(round(0.2 * len(eligible))))
    tune = [eligible[i] for i in order[:tune_count]]
    fit = [eligible[i] for i in order[tune_count:]]
    fit.sort(key=lambda e: fingerprint(dict(seed=seed, sequence=e["sequence"])))
    if len(fit) < videos:
        raise ValueError("Only {} eligible fit videos after 80/20 split; need {}".format(len(fit), videos))
    return fit[:videos], fit, tune


def path_identity(rt, modules, image):
    torch = rt.torch
    with torch.no_grad():
        search, _ = rt.prepare(image)
        tokens = rt.prefix(search)
        conditioned = modules.read_memory(tokens, modules.initial_memory(tokens))
        maximum = 0.0
        for action in range(7):
            original = tokens if not action else rt.backbone.blocks[action + 5](tokens)
            features = conditioned if not action else rt.backbone.blocks[action + 5](conditioned)
            adapted = modules.adapt(features, action)
            reference, actual = rt.head(original), rt.head(adapted)
            for key in reference:
                if torch.is_tensor(reference[key]):
                    torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=1e-6)
                    maximum = max(maximum, float((actual[key] - reference[key]).abs().max()))
        return maximum


def gradient_probe(rt, modules, images, initial):
    # Perturb only a disposable copy: zero output initialization otherwise
    # blocks gradients to the memory writer on the very first backward pass.
    probe = copy.deepcopy(modules).train()
    with rt.torch.no_grad():
        probe.read_out.weight.fill_(0.01)
    env = TemporalEnvironment(rt, probe)
    env.reset(images[0], initial)
    for index, image in enumerate(images[1:5]):
        result = env.step(image, action=1 if index == 3 else 0)
    head_gradient = rt.torch.autograd.grad(
        result["output"]["score_map"].sum(), result["features"], retain_graph=True)[0]
    if not rt.torch.isfinite(head_gradient).all() or head_gradient.abs().sum() <= 0:
        raise RuntimeError("Frozen prediction head does not transmit gradients")
    loss = result["output"]["score_map"].square().mean() + result["features"].square().mean()
    loss.backward()
    if not rt.torch.isfinite(loss):
        raise RuntimeError("Non-finite gradient probe loss")
    norms = {}
    for group in ("read_out", "update", "adapters.0"):
        gradients = [p.grad for name, p in probe.named_parameters()
                     if name.startswith(group) and p.grad is not None]
        if not gradients or any(not rt.torch.isfinite(g).all() for g in gradients):
            raise RuntimeError("Missing or non-finite gradients: " + group)
        norms[group] = sum(float(g.abs().sum()) for g in gradients)
        if norms[group] <= 0:
            raise RuntimeError("Zero gradient through frozen execution path: " + group)
    return norms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--videos", type=int, default=5)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2053)
    args = parser.parse_args()
    if not args.manifest.strip() or not args.output_dir.strip():
        parser.error("manifest/output_dir is empty; check the shell variables")
    if args.videos < 1 or args.frames < 64:
        parser.error("videos must be positive and frames must be at least 64")
    manifest = load_manifest(args.manifest)
    entries, fit, tune = select_fit_pilot(manifest, args.videos, args.frames, args.seed)
    # Audit source coverage before allocating the GPU runtime.
    sources = []
    for entry in entries:
        directory, gt, valid = load_annotations(manifest["root"], manifest["dataset"], entry["sequence"])
        if file_hash(directory / "groundtruth.txt") != entry["annotation_sha256"]:
            raise ValueError("Annotation changed: " + entry["sequence"])
        if fingerprint(valid.tolist()) != entry["visible_sha256"] or not valid[entry["start"]]:
            raise ValueError("Visibility or initialization changed: " + entry["sequence"])
        paths = [frame_path(directory, manifest["dataset"], i)
                 for i in range(entry["start"], entry["start"] + args.frames + 1)]
        if any(not p.is_file() for p in paths) or entry["start"] + args.frames >= len(gt):
            raise ValueError("Incomplete source clip: " + entry["sequence"])
        sources.append((entry, paths, gt[entry["start"]].copy()))

    rt = Runtime(args.parameter, args.gpu, args.checkpoint)
    torch = rt.torch
    import torch.nn as nn
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    modules = build_temporal_modules(torch, nn, rt.identity["embed_dim"], rt.lens_z,
                                     rt.search_side).to(next(rt.net.parameters()).device).eval()
    contract = dict(schema="stage18_execution_pilot_v1", manifest=manifest["fingerprint"],
                    runtime=rt.identity, model=modules.config, seed=args.seed,
                    frames=args.frames, videos=[e["sequence"] for e in entries],
                    fit_videos=[e["sequence"] for e in fit],
                    tune_videos=[e["sequence"] for e in tune],
                    schedule="first-available-credit balanced candidate cycle",
                    code_sha256={name: file_hash(Path(__file__).parent / name) for name in
                                 ("stage18_pilot.py", "stage18_environment.py",
                                  "stage18_temporal_model.py", "stage4_runtime.py",
                                  "stage1_validate.py")})
    contract["fingerprint"] = fingerprint(contract)
    output = Path(args.output_dir).expanduser().resolve()
    contract_path = output / "stage18_pilot_contract.json"
    import json
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise ValueError("Pilot output contract changed; use a new output directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Unidentified output files; use an empty pilot output directory")
        write_json(contract_path, contract)
    rows = []
    for entry, paths, initial in sources:
        # Only five images are needed for the disposable gradient test.
        probe_images = [rt.read(path) for path in paths[:5]]
        rt.initialize(probe_images[0], initial)
        identity_diff = path_identity(rt, modules, probe_images[1])
        gradients = gradient_probe(rt, modules, probe_images, initial)
        env = TemporalEnvironment(rt, modules)
        env.reset(probe_images[0], initial)
        latencies, actions, credits = [], [], []
        candidate_index = 0
        head_calls = [0]
        def count_head(_module, _inputs, _output):
            head_calls[0] += 1
        hook = rt.net.box_head.register_forward_hook(count_head)
        torch.cuda.reset_peak_memory_stats()
        memory_changed = False
        try:
            with torch.no_grad():
                for index, path in enumerate(paths[1:]):
                    image = rt.read(path)
                    available = min(20, env.budget.credit + 3)
                    action = 0
                    if available >= 10:
                        action = 1 + candidate_index % 6
                        candidate_index += 1
                    before = None if env.memory is None else env.memory.clone()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    result = env.step(image, action)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - started) * 1000
                    if index >= 20:
                        latencies.append(elapsed)
                    if not np.isfinite(result["box"]).all() or not torch.isfinite(env.memory).all():
                        raise RuntimeError("Non-finite live state")
                    if before is not None and not torch.equal(before, env.memory):
                        memory_changed = True
                    actions.append(action)
                    credits.append(result["credit_after"])
                    # A restored branch must replay exactly, including memory and credit.
                    if index == 10:
                        snapshot = env.snapshot()
                        first = env.step(image, 0)
                        first_memory = env.memory.clone()
                        env.restore(snapshot)
                        second = env.step(image, 0)
                        np.testing.assert_array_equal(first["box"], second["box"])
                        torch.testing.assert_close(first_memory, env.memory, rtol=0, atol=0)
                        env.restore(snapshot)
        finally:
            hook.remove()
        if head_calls[0] != args.frames + 2 or not memory_changed:
            raise RuntimeError("One-head execution or live memory update check failed")
        if set(actions) != set(range(7)):
            raise RuntimeError("Pilot did not execute all seven paths")
        row = dict(sequence=entry["sequence"], frames=args.frames,
                   candidate_calls=sum(a > 0 for a in actions),
                   candidate_ratio=sum(a > 0 for a in actions) / args.frames,
                   identity_max_abs_diff=identity_diff, memory_changed=memory_changed,
                   main_head_calls=head_calls[0] - 2, branch_replay_pass=True,
                   mean_model_ms=float(np.mean(latencies)), p95_model_ms=float(np.quantile(latencies, .95)),
                   peak_allocated_mb=torch.cuda.max_memory_allocated() / 1024**2,
                   gradient_read=gradients["read_out"], gradient_write=gradients["update"],
                   gradient_adapter_l7=gradients["adapters.0"])
        rows.append(row)
        write_json(output / (entry["sequence"] + "__pilot.json"),
                   dict(contract_fingerprint=contract["fingerprint"], report=row,
                        actions=actions, credit_after=credits))
        print("[pilot] {}: {} frames, {} candidate calls, {:.3f} ms".format(
            entry["sequence"], args.frames, row["candidate_calls"], row["mean_model_ms"]), flush=True)
    write_csv(output / "stage18_pilot_summary.csv", rows)
    write_json(output / "stage18_pilot_decision.json", dict(
        schema="stage18_execution_pilot_v1", status="PASS_EXECUTION_PILOT",
        contract_fingerprint=contract["fingerprint"], videos=len(rows),
        role="untrained_module_integrity_only",
        warning="No trained policy, AUC gain, or learned-policy speed claim is established.",
        next_step="Implement supervised chronological representation warm-up on fit videos."))
    print("PASS_EXECUTION_PILOT; no efficacy decision. Reports: {}".format(output))


if __name__ == "__main__":
    main()
