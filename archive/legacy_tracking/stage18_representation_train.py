"""Chronological Stage-18 representation warm-up and same-process timing."""

import argparse
import contextlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from stage4_common import (file_hash, fingerprint, frame_path, load_annotations,
                           load_manifest, write_csv, write_json)
from stage4_runtime import Runtime, add_runtime_arguments
from stage18_environment import TemporalEnvironment, executed_crop_roi
from stage18_temporal_model import build_temporal_modules
from stage18_tracking_loss import tracking_loss


REPRESENTATION_PREFIXES = ("template.", "slots", "query.", "key.", "value.",
                           "read_out.", "adapters.", "target.", "action_embedding.",
                           "update.", "write_gate.")


def behavior_actions(seed, sequence, epoch, length):
    digest = fingerprint(dict(seed=int(seed), sequence=sequence, epoch=int(epoch)))
    rng = np.random.default_rng(int(digest[:16], 16))
    return rng.choice(7, size=length, p=[.5] + [1 / 12] * 6)


def load_pilot_contract(directory):
    directory = Path(directory).expanduser().resolve()
    contract = json.loads((directory / "stage18_pilot_contract.json").read_text())
    decision = json.loads((directory / "stage18_pilot_decision.json").read_text())
    body = {k: v for k, v in contract.items() if k != "fingerprint"}
    if (fingerprint(body) != contract.get("fingerprint")
            or decision.get("status") != "PASS_EXECUTION_PILOT"
            or decision.get("contract_fingerprint") != contract["fingerprint"]):
        raise ValueError("Require a valid passed Stage-18A pilot contract")
    for name, digest in contract["code_sha256"].items():
        if file_hash(Path(__file__).parent / name) != digest:
            raise ValueError("Pilot-bound execution code changed: " + name)
    return contract


class RepresentationCredit:
    """Phase-A balanced behavior is not subject to the deployment P30 cap."""
    def __init__(self):
        self.credit, self.steps, self.calls = 0, 0, 0

    def advance(self):
        self.credit = 20

    def spend(self, action):
        if action not in range(7):
            raise ValueError("Unknown action")
        self.steps += 1
        self.calls += int(action > 0)
        self.credit -= 10 * int(action > 0)


class CapturedRuntime:
    def __init__(self, runtime):
        self.runtime = runtime
        self.factor = None

    def __getattr__(self, name):
        return getattr(self.runtime, name)

    def prepare(self, image):
        search, self.factor = self.runtime.prepare(image)
        return search, self.factor


class RepresentationEnvironment(TemporalEnvironment):
    def __init__(self, runtime, modules):
        super().__init__(CapturedRuntime(runtime), modules)

    def reset(self, image, initial_box):
        super().reset(image, initial_box)
        self.budget = RepresentationCredit()

    def step(self, image, action):
        previous = list(self.rt.tracker.state)
        result = super().step(image, action)
        result["previous_box"] = previous
        result["resize_factor"] = self.rt.factor
        return result


def configure_representation(modules):
    for name, parameter in modules.named_parameters():
        parameter.requires_grad_(name.startswith(REPRESENTATION_PREFIXES))
    return [p for p in modules.parameters() if p.requires_grad]


def load_sources(manifest, names, maximum):
    entries = {e["sequence"]: e for e in manifest["sequences"] if e["split"] == "train"}
    sources = []
    for name in names:
        if name not in entries:
            raise ValueError("Pilot split escaped the original train videos: " + name)
        entry = entries[name]
        directory, gt, valid = load_annotations(manifest["root"], manifest["dataset"], name)
        if (file_hash(directory / "groundtruth.txt") != entry["annotation_sha256"]
                or fingerprint(valid.tolist()) != entry["visible_sha256"]):
            raise ValueError("Source annotation changed: " + name)
        start = entry["start"]
        stop = min(entry["stop"], start + maximum + 1)
        paths = [frame_path(directory, manifest["dataset"], i) for i in range(start, stop)]
        if stop - start - 1 < 64 or stop > len(gt) or not valid[start] or any(not p.is_file() for p in paths):
            raise ValueError("Incomplete source clip: " + name)
        if not valid[start + 1:stop].any():
            raise ValueError("No valid tracked-frame supervision: " + name)
        sources.append(dict(sequence=name, paths=paths, gt=gt[start:stop].copy(),
                            valid=valid[start:stop].copy()))
    return sources


def clip_loss(rt, modules, source, actions, unroll, backward_scale=None):
    torch = rt.torch
    env = RepresentationEnvironment(rt, modules)
    env.reset(rt.read(source["paths"][0]), source["gt"][0])
    valid_count = int(source["valid"][1:].sum())
    chunk = []
    totals = dict(loss=0., valid_frames=valid_count, frames=len(actions), regression_frames=0)
    totals["candidate_frames"] = int(np.sum(np.asarray(actions) > 0))
    manager = contextlib.nullcontext if backward_scale is not None else torch.no_grad
    with manager():
        for index, (path, action) in enumerate(zip(source["paths"][1:], actions)):
            result = env.step(rt.read(path), int(action))
            valid = bool(source["valid"][index + 1])
            target = np.zeros(4, dtype=np.float32)
            if valid:
                target = executed_crop_roi(source["gt"][index + 1], result["previous_box"],
                                           result["resize_factor"], rt.tracker.params.search_size)
            loss, stats = tracking_loss(torch, result["output"],
                                        result["features"].new_tensor(target)[None], valid=valid)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite tracking loss: " + source["sequence"])
            totals["loss"] += float(loss.detach())
            totals["regression_frames"] += int(stats["regression_count"])
            if backward_scale is not None:
                chunk.append(loss)
            if (index + 1) % unroll == 0 or index + 1 == len(actions):
                if backward_scale is not None:
                    (torch.stack(chunk).sum() * backward_scale / max(1, valid_count)).backward()
                    chunk.clear()
                env.memory = env.memory.detach()
    totals["mean_loss"] = totals["loss"] / max(1, valid_count)
    return totals


def atomic_torch_save(torch, value, path):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), suffix=".tmp", delete=False) as f:
        temporary = f.name
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.isfile(temporary):
            os.unlink(temporary)


def load_checkpoint(torch, path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def profile_references(rt, modules, sources, repeats=5):
    """Interleave separately initialized trajectories; include all model overhead."""
    torch = rt.torch
    modules.eval()
    rows = []
    with torch.no_grad():
        for source in sources:
            images = [rt.read(p) for p in source["paths"][:65]]
            for repeat in range(repeats):
                names = ["fast", "sgla", "memory_balanced_p30"]
                order = np.random.default_rng(2053 + repeat).permutation(3)
                for method in [names[i] for i in order]:
                    env = TemporalEnvironment(rt, modules)
                    env.reset(images[0], source["gt"][0])
                    elapsed = []
                    candidate = 0
                    for index, image in enumerate(images[1:]):
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        if method == "memory_balanced_p30":
                            action = 0
                            if min(20, env.budget.credit + 3) >= 10:
                                action = 1 + candidate % 6
                                candidate += 1
                            env.step(image, action)
                        else:
                            box = rt.run_image(image, method)
                            rt.tracker.state = list(box)
                            rt.tracker.frame_id += 1
                        torch.cuda.synchronize()
                        if index >= 20:
                            elapsed.append((time.perf_counter() - started) * 1000)
                    rows.append(dict(sequence=source["sequence"], repeat=repeat, method=method,
                                     frames=len(elapsed), mean_ms=float(np.mean(elapsed)),
                                     p95_ms=float(np.quantile(elapsed, .95))))
    means = {name: float(np.mean([r["mean_ms"] for r in rows if r["method"] == name]))
             for name in ("fast", "sgla", "memory_balanced_p30")}
    ratio = means["memory_balanced_p30"] / means["sgla"]
    return rows, dict(means_ms=means, memory_to_sgla_ratio=ratio,
                      resource_target_met=ratio <= 1.0,
                      role="same_process_engineering_schedule_not_learned_policy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pilot_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--seed", type=int, default=2053)
    parser.add_argument("--unroll", type=int, default=32)
    parser.add_argument("--batch_clips", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    args = parser.parse_args()
    if any(not getattr(args, k).strip() for k in ("manifest", "pilot_dir", "output_dir")):
        parser.error("A required path is empty; check shell variables")
    args.epochs = args.epochs if args.epochs is not None else (1 if args.mode == "smoke" else 10)
    args.frames = args.frames if args.frames is not None else (64 if args.mode == "smoke" else 128)
    if (args.epochs < 1 or not 64 <= args.frames <= 128 or args.unroll < 2
            or args.batch_clips < 1 or args.learning_rate <= 0):
        parser.error("Invalid training configuration")
    pilot = load_pilot_contract(args.pilot_dir)
    manifest = load_manifest(args.manifest)
    if manifest["fingerprint"] != pilot["manifest"] or args.seed != pilot["seed"]:
        raise ValueError("Manifest/split seed differs from passed pilot")
    fit_names = pilot["videos"] if args.mode == "smoke" else pilot["fit_videos"]
    fit = load_sources(manifest, fit_names, args.frames)
    tune = [] if args.mode == "smoke" else load_sources(manifest, pilot["tune_videos"], 256)
    rt = Runtime(args.parameter, args.gpu, args.checkpoint)
    if rt.identity != pilot["runtime"]:
        raise ValueError("Runtime or original checkpoint differs from pilot")
    torch = rt.torch
    import torch.nn as nn
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    modules = build_temporal_modules(torch, nn, rt.identity["embed_dim"], rt.lens_z,
                                     rt.search_side).to(next(rt.net.parameters()).device)
    parameters = configure_representation(modules)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    local = Path(__file__).parent
    contract = dict(schema="stage18_representation_v1", mode=args.mode,
                    pilot_fingerprint=pilot["fingerprint"], epochs=args.epochs,
                    seed=args.seed, maximum_frames=args.frames, unroll=args.unroll,
                    batch_clips=args.batch_clips, learning_rate=args.learning_rate,
                    weight_decay=.01, grad_clip=1.0, tracking_loss_weights=[2, 5, 1],
                    fit=[dict(sequence=s["sequence"], frames=len(s["paths"]) - 1) for s in fit],
                    tune=[dict(sequence=s["sequence"], frames=len(s["paths"]) - 1) for s in tune],
                    behavior="Fast=0.5; each candidate=1/12; unconstrained warm-up",
                    code_sha256={n: file_hash(local / n) for n in
                                 ("stage18_representation_train.py", "stage18_tracking_loss.py")})
    contract["code_sha256"]["lib/utils/heapmap_utils.py"] = file_hash(
        local.parent / "lib/utils/heapmap_utils.py")
    contract["code_sha256"]["lib/utils/box_ops.py"] = file_hash(local.parent / "lib/utils/box_ops.py")
    contract["code_sha256"]["lib/utils/focal_loss.py"] = file_hash(local.parent / "lib/utils/focal_loss.py")
    contract["fingerprint"] = fingerprint(contract)
    output = Path(args.output_dir).expanduser().resolve()
    path = output / "stage18_representation_contract.json"
    if path.is_file():
        if json.loads(path.read_text()) != contract:
            raise ValueError("Training contract changed; use a new output directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Unidentified training output; use an empty output directory")
        write_json(path, contract)
    last_path, best_path = output / "last.pt", output / "best.pt"
    start, best, history = 1, float("inf"), []
    gradient_max = dict(read_out=0., update=0.)
    gradient_max.update({"adapters.{}".format(i): 0. for i in range(6)})
    if last_path.exists():
        checkpoint = load_checkpoint(torch, last_path)
        if checkpoint["contract_fingerprint"] != contract["fingerprint"]:
            raise ValueError("Checkpoint contract mismatch")
        modules.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, best, history = checkpoint["epoch"] + 1, checkpoint["best"], checkpoint["history"]
        gradient_max = checkpoint["gradient_max"]
    for epoch in range(start, args.epochs + 1):
        modules.train()
        torch.cuda.reset_peak_memory_stats()
        order = np.random.default_rng(args.seed + epoch).permutation(len(fit))
        rows = []
        for begin in range(0, len(order), args.batch_clips):
            batch = [fit[i] for i in order[begin:begin + args.batch_clips]]
            optimizer.zero_grad(set_to_none=True)
            for source in batch:
                actions = behavior_actions(args.seed, source["sequence"], epoch, len(source["paths"]) - 1)
                stats = clip_loss(rt, modules, source, actions, args.unroll, 1 / len(batch))
                rows.append(stats)
            for name, p in modules.named_parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        raise RuntimeError("Non-finite parameter gradient: " + name)
                    for group in gradient_max:
                        if name.startswith(group):
                            gradient_max[group] = max(gradient_max[group], float(p.grad.abs().sum()))
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("Non-finite representation gradient norm")
            optimizer.step()
            print("[epoch {}] {}/{} fit clips; loss={:.4f}".format(
                epoch, len(rows), len(fit), rows[-1]["mean_loss"]), flush=True)
        modules.eval()
        tune_loss = None
        if tune:
            evaluated = [clip_loss(rt, modules, s, behavior_actions(
                args.seed, s["sequence"], 0, len(s["paths"]) - 1), args.unroll) for s in tune]
            tune_loss = float(np.mean([r["mean_loss"] for r in evaluated]))
        score = tune_loss if tune else float(np.mean([r["mean_loss"] for r in rows]))
        improved = score < best
        best = min(best, score)
        history.append(dict(epoch=epoch, fit_loss=float(np.mean([r["mean_loss"] for r in rows])),
                            tune_loss=tune_loss, valid_fit_frames=sum(r["valid_frames"] for r in rows),
                            fit_candidate_ratio=sum(r["candidate_frames"] for r in rows) /
                            sum(r["frames"] for r in rows),
                            peak_allocated_mb=torch.cuda.max_memory_allocated() / 1024**2))
        checkpoint = dict(contract_fingerprint=contract["fingerprint"], epoch=epoch,
                          model=modules.state_dict(), optimizer=optimizer.state_dict(),
                          best=best, history=history, gradient_max=gradient_max)
        atomic_torch_save(torch, checkpoint, last_path)
        if improved:
            atomic_torch_save(torch, checkpoint, best_path)
        write_csv(output / "stage18_representation_history.csv", history)
    # Recover cumulative gradients even when an already complete run is resumed.
    checkpoint = load_checkpoint(torch, last_path)
    gradient_max = checkpoint["gradient_max"]
    training_pass = all(v > 0 for v in gradient_max.values())
    best_checkpoint = load_checkpoint(torch, best_path)
    modules.load_state_dict(best_checkpoint["model"])
    profile_rows, resource = profile_references(rt, modules, fit[:5])
    write_csv(output / "stage18_same_process_latency.csv", profile_rows)
    status = ("PASS_SUPERVISED_SMOKE" if args.mode == "smoke" else "COMPLETE_REPRESENTATION_WARMUP")
    if not training_pass:
        status = "FAIL_REPRESENTATION_GRADIENTS"
    decision = dict(schema="stage18_representation_v1", status=status,
                    contract_fingerprint=contract["fingerprint"], gradient_max=gradient_max,
                    training_pass=training_pass, resource_audit=resource,
                    best_epoch=best_checkpoint["epoch"], best_checkpoint=str(best_path),
                    best_checkpoint_sha256=file_hash(best_path),
                    next_step=("Review same-process timing before full training" if args.mode == "smoke"
                               else "Freeze tune-selected representation before on-policy PPO"),
                    warning="Representation training only; no learned layer-policy gain is established.")
    write_json(output / "stage18_representation_decision.json", decision)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
