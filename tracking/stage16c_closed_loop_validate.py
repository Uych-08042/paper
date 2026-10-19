"""Run the frozen TAVRTrack-P30 policy in a true continuous closed loop.

This is the single Stage-16C development evaluation authorized by a passed
Stage-16B screen.  The recurrent ensemble, P30 budget, and train-calibrated
advantage threshold are immutable.  Fast, same-cap uniform L9, and native
SGLA are collected as paired references on exactly the same manifest clips.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

from stage4_common import (
    IOU_PROTOCOL,
    file_hash,
    fingerprint,
    frame_path,
    load_annotations,
    load_manifest,
    tracking_iou,
    write_json,
)
from stage4_runtime import Runtime, add_runtime_arguments
from stage16_behavior_common import ACTION_NAMES, CANDIDATE_LAYERS
from stage16_recurrent_cql import SCALAR_NAMES, build_q_model
from stage16c_freeze_policy import (
    POLICY_NAME,
    artifact_path,
    load_policy_bundle,
)


SCHEMA = "stage16c_closed_loop_v1"
POLICIES = ("fast", "uniform_l9_p30", "sgla", POLICY_NAME)
CSV_FIELDS = (
    "dataset",
    "split",
    "sequence",
    "policy",
    "frame_id",
    "decision_index",
    "gt_valid",
    "iou",
    "candidate_blocks",
    "action",
    "action_index",
    "selected_layer",
    "native_selected_layer",
    "candidate_budget_total",
    "candidate_used_before",
    "candidate_used_after",
    "candidate_remaining_after",
    "candidate_advantage_threshold",
    "predicted_advantage",
    "q_fast",
    "q_l7",
    "q_l8",
    "q_l9",
    "q_l10",
    "q_l11",
    "q_l12",
    "model_decode_latency_ms",
    "gt_x",
    "gt_y",
    "gt_w",
    "gt_h",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "manifest_fingerprint",
    "policy_bundle_fingerprint",
)


def candidate_budget(num_steps, budget_percent):
    return int(np.floor(int(num_steps) * float(budget_percent) / 100.0 + 0.5))


def uniform_exact_schedule(num_steps, count):
    num_steps, count = int(num_steps), int(count)
    if num_steps < 1 or count < 0 or count > num_steps:
        raise ValueError("Invalid exact uniform schedule")
    previous = np.floor(
        np.arange(num_steps, dtype=np.float64) * count / num_steps + 1e-12
    ).astype(np.int64)
    completed = np.floor(
        (np.arange(num_steps, dtype=np.float64) + 1.0) * count / num_steps
        + 1e-12
    ).astype(np.int64)
    active = completed > previous
    if int(active.sum()) != count:
        raise RuntimeError("Exact uniform schedule did not spend its budget")
    return active


def single_step_scalar_features(
    selector_prob,
    native_selected_layer,
    previous_state,
    previous_fast_box,
    fast_box,
    previous_action,
    image_hw,
    candidate_used_before,
    candidate_remaining_before,
    num_steps,
    step_index,
    resize_factor,
):
    """Build the exact Stage-16B scalar feature vector for one online state."""
    selector_prob = np.asarray(selector_prob, dtype=np.float32).reshape(6)
    native_selected_layer = int(native_selected_layer)
    if not 7 <= native_selected_layer <= 12:
        raise ValueError("Native selected layer must be L7-L12")
    selected = np.eye(6, dtype=np.float32)[native_selected_layer - 7]
    image_hw = np.asarray(image_hw, dtype=np.float64).reshape(2)
    scale = np.asarray(
        [image_hw[1], image_hw[0], image_hw[1], image_hw[0]],
        dtype=np.float64,
    )

    def normalized(box):
        return (np.asarray(box, dtype=np.float64).reshape(4) / scale).astype(
            np.float32
        )

    previous_state = normalized(previous_state)
    previous_fast = normalized(previous_fast_box)
    fast = normalized(fast_box)
    previous_action = int(previous_action)
    if not -1 <= previous_action < len(ACTION_NAMES):
        raise ValueError("Invalid previous action")
    previous_one_hot = np.eye(8, dtype=np.float32)[previous_action + 1]
    total = int(candidate_used_before) + int(candidate_remaining_before)
    remaining_frames = int(num_steps) - int(step_index)
    if remaining_frames < 1 or total < 0:
        raise ValueError("Invalid online budget state")
    remaining_rate = float(candidate_remaining_before) / remaining_frames
    target_rate = float(total) / int(num_steps)
    budget_progress = (
        float(candidate_used_before) / total if total > 0 else 0.0
    )
    time_fraction = float(step_index) / max(1, int(num_steps) - 1)
    values = np.concatenate(
        (
            selector_prob,
            selected,
            previous_state,
            previous_fast,
            fast,
            previous_state - previous_fast,
            previous_one_hot,
            np.asarray(
                [
                    remaining_rate,
                    target_rate,
                    budget_progress,
                    time_fraction,
                    np.log(float(resize_factor)),
                ],
                dtype=np.float32,
            ),
        )
    ).astype(np.float32)
    if values.shape != (len(SCALAR_NAMES),) or not np.isfinite(values).all():
        raise RuntimeError("Online Stage-16B scalar feature contract changed")
    return values


def _spatial_state(rt, tokens, grid):
    template = tokens[:, : rt.lens_z]
    search = tokens[:, rt.lens_z :].transpose(1, 2).reshape(
        tokens.shape[0], tokens.shape[2], rt.search_side, rt.search_side
    )
    search = rt.torch.nn.functional.adaptive_avg_pool2d(search, (grid, grid))
    search = search.permute(0, 2, 3, 1)
    template_summary = rt.torch.stack(
        (template.mean(1), template.max(1).values), dim=1
    )
    return template_summary, search


class FrozenTavrPolicy:
    def __init__(self, runtime, bundle):
        self.rt = runtime
        self.torch = runtime.torch
        self.bundle = bundle
        self.device = next(runtime.net.parameters()).device
        self.budget_percent = float(bundle["primary_budget_percent"])
        self.threshold = float(bundle["candidate_advantage_threshold"])
        self.state_grid = int(bundle["state_grid"])

        with np.load(str(artifact_path(bundle, "token_pca")), allow_pickle=False) as archive:
            pca_mean = np.asarray(archive["mean"], dtype=np.float32)
            pca_components = np.asarray(archive["components"], dtype=np.float32)
        with np.load(
            str(artifact_path(bundle, "scalar_normalization")), allow_pickle=False
        ) as archive:
            scalar_mean = np.asarray(archive["mean"], dtype=np.float32)
            scalar_scale = np.asarray(archive["scale"], dtype=np.float32)
            names = [str(value) for value in archive["names"].tolist()]
        expected_runtime = bundle["runtime_feature_identity"]
        observed_runtime = {
            key: runtime.identity[key] for key in ("lens_z", "lens_x", "embed_dim")
        }
        if (
            names != list(SCALAR_NAMES)
            or pca_mean.shape != (runtime.identity["embed_dim"],)
            or observed_runtime != expected_runtime
        ):
            raise ValueError("Frozen preprocessing is incompatible with this runtime")
        if self.state_grid > runtime.search_side:
            raise ValueError("Frozen state grid exceeds runtime search grid")
        self.pca_mean = self.torch.as_tensor(pca_mean, device=self.device)
        self.pca_components = self.torch.as_tensor(
            pca_components, device=self.device
        )
        self.scalar_mean = self.torch.as_tensor(scalar_mean, device=self.device)
        self.scalar_scale = self.torch.as_tensor(scalar_scale, device=self.device)

        self.models = []
        for seed in bundle["seeds"]:
            checkpoint_path = artifact_path(bundle, "checkpoint_seed_{}".format(seed))
            checkpoint = self.torch.load(str(checkpoint_path), map_location=self.device)
            if (
                checkpoint.get("schema") != "stage16b_recurrent_cql_v1"
                or int(checkpoint.get("seed")) != int(seed)
                or checkpoint.get("model_config") != bundle["model_config"]
                or not np.allclose(
                    np.asarray(checkpoint["scalar_mean"]), scalar_mean, rtol=0, atol=0
                )
                or not np.allclose(
                    np.asarray(checkpoint["scalar_scale"]), scalar_scale, rtol=0, atol=0
                )
            ):
                raise ValueError("Checkpoint contract mismatch: {}".format(checkpoint_path))
            model = build_q_model(
                self.torch, self.torch.nn, bundle["model_config"]
            ).to(self.device)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.eval().requires_grad_(False)
            self.models.append(model)
        self.hidden = None
        self.num_steps = None
        self.budget_total = None
        self.used = None

    def reset(self, num_steps):
        self.num_steps = int(num_steps)
        self.budget_total = candidate_budget(num_steps, self.budget_percent)
        self.used = 0
        self.hidden = [None for _ in self.models]

    def _project(self, values):
        # Stage-16A stored raw token tensors as float16, and Stage-16B stored
        # their PCA projections as float16. Reproduce both quantization steps.
        quantized = values.to(dtype=self.torch.float16).float()
        projected = (quantized - self.pca_mean) @ self.pca_components.t()
        return projected.to(dtype=self.torch.float16).float()

    def _step_model(self, model, batch, hidden):
        grid = batch["search_grid"].reshape(
            -1, *batch["search_grid"].shape[2:]
        ).permute(0, 3, 1, 2)
        spatial = model.spatial(grid).flatten(1)
        template = model.template(batch["template"].reshape(1, -1))
        score = model.score(
            batch["score_map"].reshape(-1, *batch["score_map"].shape[2:])
        ).flatten(1)
        scalar = model.scalar(batch["scalar"].reshape(1, -1))
        fused = model.fusion(
            self.torch.cat((spatial, template, score, scalar), dim=1)
        )
        recurrent, hidden = model.gru(fused.reshape(1, 1, -1), hidden)
        return model.q_head(recurrent[:, 0]), hidden

    def decide(
        self,
        template_summary,
        search_grid,
        score_map,
        selector_prob,
        native_selected_layer,
        previous_state,
        previous_fast_box,
        fast_box,
        previous_action,
        image_hw,
        resize_factor,
        step_index,
    ):
        if self.num_steps is None:
            raise RuntimeError("Frozen policy must be reset before use")
        remaining = self.budget_total - self.used
        scalar = single_step_scalar_features(
            selector_prob,
            native_selected_layer,
            previous_state,
            previous_fast_box,
            fast_box,
            previous_action,
            image_hw,
            self.used,
            remaining,
            self.num_steps,
            step_index,
            resize_factor,
        )
        batch = {
            "search_grid": self._project(search_grid).reshape(
                1, 1, self.state_grid, self.state_grid, -1
            ),
            "template": self._project(template_summary).reshape(1, 1, 2, -1),
            "score_map": score_map.to(dtype=self.torch.float16).float().reshape(
                1, 1, *score_map.shape[1:]
            ),
            "scalar": (
                (self.torch.as_tensor(scalar, device=self.device) - self.scalar_mean)
                / self.scalar_scale
            ).reshape(1, 1, -1),
        }
        q_values = []
        updated_hidden = []
        for model, hidden in zip(self.models, self.hidden):
            q, hidden = self._step_model(model, batch, hidden)
            q_values.append(q)
            updated_hidden.append(hidden)
        self.hidden = updated_hidden
        q = self.torch.stack(q_values, dim=0).mean(dim=0)[0]
        candidate_action = int(self.torch.argmax(q[1:]).item()) + 1
        advantage = float((q[candidate_action] - q[0]).item())
        action = (
            candidate_action
            if remaining > 0 and advantage >= self.threshold
            else 0
        )
        used_before = self.used
        if action > 0:
            self.used += 1
        return {
            "action": action,
            "candidate_action": candidate_action,
            "advantage": advantage,
            "q": q.detach().float().cpu().numpy(),
            "used_before": used_before,
            "used_after": self.used,
            "remaining_after": self.budget_total - self.used,
        }


def _safe_name(value):
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def output_path(output_dir, entry, policy):
    return Path(output_dir) / entry["split"] / "{}__{}__stage16c_closed_loop.csv".format(
        _safe_name(entry["sequence"]), policy
    )


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def publish_closed_loop_run(output, run):
    """Publish one shard-safe contract and reject unidentified CSV outputs."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "run.lock"
    deadline = time.monotonic() + 30.0
    while True:
        try:
            lock.open("x").close()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RuntimeError("run.lock persists; check for a running Stage-16C shard")
            time.sleep(0.1)
    try:
        path = output / "run.json"
        if path.is_file():
            observed = json.loads(path.read_text(encoding="utf-8"))
            if observed != run:
                raise ValueError(
                    "Stage-16C configuration changed; use a new output_dir"
                )
        else:
            unidentified = next(output.rglob("*__stage16c_closed_loop.csv"), None)
            if unidentified is not None:
                raise ValueError(
                    "Stage-16C CSV exists without run.json: {}".format(unidentified)
                )
            write_json(path, run)
    finally:
        lock.unlink()


def make_run(manifest_path, manifest, entries, policies, runtime, bundle, args):
    run = {
        "schema": SCHEMA,
        "analysis_role": "single_true_closed_loop_development_evaluation",
        "manifest_path": str(Path(manifest_path).expanduser().resolve()),
        "manifest_fingerprint": manifest["fingerprint"],
        "dataset": manifest["dataset"],
        "split": args.split,
        "entries": entries,
        "policies": list(policies),
        "policy_bundle_path": str(Path(args.policy_bundle).expanduser().resolve()),
        "policy_bundle_fingerprint": bundle["fingerprint"],
        "primary_policy": POLICY_NAME,
        "matched_budget_baseline": "uniform_l9_p30",
        "budget_percent": float(bundle["primary_budget_percent"]),
        "candidate_advantage_threshold": float(
            bundle["candidate_advantage_threshold"]
        ),
        "threshold_source": bundle["threshold_source"],
        "runtime": runtime.identity,
        "latency_warmup_frames": int(args.latency_warmup_frames),
        "iou_protocol": IOU_PROTOCOL,
        "claim_boundary": (
            "Stage-16C is a one-shot development closed-loop test on the already "
            "designated Stage-16 validation videos; it is not external validation."
        ),
        "code_sha256": {
            "stage16c_closed_loop_validate.py": file_hash(Path(__file__).resolve()),
            "stage16c_freeze_policy.py": file_hash(
                Path(__file__).resolve().parent / "stage16c_freeze_policy.py"
            ),
            "stage16_recurrent_cql.py": file_hash(
                Path(__file__).resolve().parent / "stage16_recurrent_cql.py"
            ),
            "stage4_runtime.py": file_hash(
                Path(__file__).resolve().parent / "stage4_runtime.py"
            ),
        },
    }
    run["fingerprint"] = fingerprint(run)
    return run


def _initial_row(entry, manifest, policy, bundle, ground_truth, valid):
    frame_id = entry["start"]
    box = np.asarray(ground_truth[frame_id], dtype=np.float64)
    valid_gt = bool(valid[frame_id])
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "dataset": manifest["dataset"],
            "split": entry["split"],
            "sequence": entry["sequence"],
            "policy": policy,
            "frame_id": frame_id,
            "decision_index": 0,
            "gt_valid": int(valid_gt),
            "iou": float(tracking_iou(box, box, valid_gt)),
            "candidate_blocks": 0,
            "action": "init",
            "action_index": -1,
            "selected_layer": "",
            "native_selected_layer": "",
            "candidate_budget_total": 0,
            "candidate_used_before": 0,
            "candidate_used_after": 0,
            "candidate_remaining_after": 0,
            "gt_x": box[0],
            "gt_y": box[1],
            "gt_w": box[2],
            "gt_h": box[3],
            "bbox_x": box[0],
            "bbox_y": box[1],
            "bbox_w": box[2],
            "bbox_h": box[3],
            "manifest_fingerprint": manifest["fingerprint"],
            "policy_bundle_fingerprint": bundle["fingerprint"],
        }
    )
    return row


def _tracked_row(
    entry,
    manifest,
    policy,
    bundle,
    frame_id,
    step,
    ground_truth,
    valid,
    box,
    action,
    native_layer,
    budget_total,
    used_before,
    used_after,
    threshold,
    q,
    advantage,
    latency_ms,
):
    gt = np.asarray(ground_truth[frame_id], dtype=np.float64)
    valid_gt = bool(valid[frame_id])
    row = {field: "" for field in CSV_FIELDS}
    selected_layer = action + 6 if action > 0 else ""
    row.update(
        {
            "dataset": manifest["dataset"],
            "split": entry["split"],
            "sequence": entry["sequence"],
            "policy": policy,
            "frame_id": frame_id,
            "decision_index": step + 1,
            "gt_valid": int(valid_gt),
            "iou": float(tracking_iou(box, gt, valid_gt)),
            "candidate_blocks": int(action > 0),
            "action": ACTION_NAMES[action],
            "action_index": action,
            "selected_layer": selected_layer,
            "native_selected_layer": "" if native_layer is None else native_layer,
            "candidate_budget_total": budget_total,
            "candidate_used_before": used_before,
            "candidate_used_after": used_after,
            "candidate_remaining_after": budget_total - used_after,
            "candidate_advantage_threshold": threshold,
            "predicted_advantage": advantage,
            "model_decode_latency_ms": latency_ms,
            "gt_x": gt[0],
            "gt_y": gt[1],
            "gt_w": gt[2],
            "gt_h": gt[3],
            "bbox_x": box[0],
            "bbox_y": box[1],
            "bbox_w": box[2],
            "bbox_h": box[3],
            "manifest_fingerprint": manifest["fingerprint"],
            "policy_bundle_fingerprint": bundle["fingerprint"],
        }
    )
    if q is not None:
        for name, value in zip(ACTION_NAMES, q):
            row["q_{}".format(name)] = float(value)
    return row


def run_policy_clip(
    runtime,
    controller,
    directory,
    manifest,
    entry,
    ground_truth,
    valid,
    policy,
    bundle,
    output_dir,
    latency_warmup_frames,
    verify_actions,
    overwrite,
):
    target = output_path(output_dir, entry, policy)
    if target.is_file() and not overwrite:
        print("[skip] {}/{} -> {}".format(entry["sequence"], policy, target))
        return False
    start, stop = entry["start"], entry["stop"]
    steps = stop - start - 1
    initial = np.asarray(ground_truth[start], dtype=np.float64)
    runtime.initialize(runtime.read(frame_path(directory, manifest["dataset"], start)), initial)
    rows = [_initial_row(entry, manifest, policy, bundle, ground_truth, valid)]
    budget_total = candidate_budget(steps, bundle["primary_budget_percent"])
    uniform_schedule = uniform_exact_schedule(steps, budget_total)
    if policy == POLICY_NAME:
        controller.reset(steps)
    used = 0
    previous_fast = initial.copy()
    previous_action = -1

    with runtime.torch.no_grad():
        for step, frame_id in enumerate(range(start + 1, stop)):
            image = runtime.read(frame_path(directory, manifest["dataset"], frame_id))
            previous_state = np.asarray(runtime.tracker.state, dtype=np.float64).copy()
            search, resize_factor = runtime.prepare(image)
            runtime.torch.cuda.synchronize()
            started = time.perf_counter()
            tokens = runtime.prefix(search)
            fast_output = runtime.head(tokens)
            fast_box = np.asarray(
                runtime.decode(fast_output, resize_factor, image), dtype=np.float64
            )
            action = 0
            native_layer = None
            q = None
            advantage = ""
            threshold = ""
            used_before = used

            if policy == "sgla":
                probabilities = runtime.selector(tokens)
                native_layer = runtime.selected_layer(probabilities)
                action = native_layer - 6
            elif policy == "uniform_l9_p30" and uniform_schedule[step]:
                action = 3
            elif policy == POLICY_NAME:
                probabilities = runtime.selector(tokens)
                native_layer = runtime.selected_layer(probabilities)
                template_summary, search_grid = _spatial_state(
                    runtime, tokens, controller.state_grid
                )
                decision = controller.decide(
                    template_summary,
                    search_grid,
                    fast_output["score_map"],
                    probabilities.detach().float().cpu().numpy()[0],
                    native_layer,
                    previous_state,
                    previous_fast,
                    fast_box,
                    previous_action,
                    image.shape[:2],
                    resize_factor,
                    step,
                )
                action = decision["action"]
                used_before = decision["used_before"]
                used = decision["used_after"]
                q = decision["q"]
                advantage = decision["advantage"]
                threshold = controller.threshold

            if action > 0:
                candidate_output = runtime.candidate(tokens, action + 6)
                box = np.asarray(
                    runtime.decode(candidate_output, resize_factor, image),
                    dtype=np.float64,
                )
                if policy != POLICY_NAME:
                    used += 1
            else:
                candidate_output = None
                box = fast_box
            runtime.torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - started) * 1000.0

            if verify_actions and step == 0:
                runtime.tracker.state = previous_state.tolist()
                integrated_fast = np.asarray(runtime.run_image(image, "fast"))
                if not np.allclose(integrated_fast, fast_box, rtol=0, atol=1e-6):
                    raise RuntimeError("Stage-16C Fast path verification failed")
                if policy == "sgla" and candidate_output is not None:
                    runtime.verify(search, candidate_output)

            rows.append(
                _tracked_row(
                    entry,
                    manifest,
                    policy,
                    bundle,
                    frame_id,
                    step,
                    ground_truth,
                    valid,
                    box,
                    action,
                    native_layer,
                    budget_total if policy in ("uniform_l9_p30", POLICY_NAME) else steps if policy == "sgla" else 0,
                    used_before,
                    used,
                    threshold,
                    q,
                    advantage,
                    "" if step < latency_warmup_frames else latency_ms,
                )
            )
            runtime.tracker.state = box.tolist()
            runtime.tracker.frame_id = step + 1
            previous_fast = fast_box.copy()
            previous_action = action

    if policy in ("uniform_l9_p30", POLICY_NAME) and used > budget_total:
        raise RuntimeError("Stage-16C policy exceeded the frozen candidate budget")
    if policy == "uniform_l9_p30" and used != budget_total:
        raise RuntimeError("Matched-cost uniform control did not spend its budget")
    _write_rows(target, rows)
    print(
        "[done] {}/{}: {} decisions, {}/{} candidates -> {}".format(
            entry["sequence"], policy, steps, used, budget_total, target
        ),
        flush=True,
    )
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--policy_bundle", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--latency_warmup_frames", type=int, default=20)
    parser.add_argument("--verify_actions", action="store_true")
    parser.add_argument("--skip_artifact_hashes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Require 0 <= shard_id < num_shards")
    if args.latency_warmup_frames < 0:
        raise ValueError("latency_warmup_frames must be non-negative")
    policies = tuple(dict.fromkeys(args.policies))
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    _, bundle = load_policy_bundle(
        args.policy_bundle, verify_artifacts=not args.skip_artifact_hashes
    )
    if manifest["fingerprint"] != bundle["source_manifest_fingerprint"]:
        raise ValueError("Stage-16C manifest is not the one used by Stage-16B")
    all_entries = [
        entry
        for entry in manifest["sequences"]
        if entry["split"] == args.split
        and (args.sequence is None or entry["sequence"] == args.sequence)
    ]
    if args.max_sequences is not None:
        if args.max_sequences < 1:
            raise ValueError("max_sequences must be positive")
        all_entries = all_entries[: args.max_sequences]
    if not all_entries:
        raise ValueError("No manifest clips match the requested Stage-16C subset")
    entries = all_entries[args.shard_id :: args.num_shards]
    if not entries:
        print("This shard has no clips.")
        return

    runtime = Runtime(args.parameter, args.gpu, args.checkpoint)
    controller = FrozenTavrPolicy(runtime, bundle)
    output_dir = Path(args.output_dir).expanduser().resolve()
    run = make_run(
        manifest_path, manifest, all_entries, policies, runtime, bundle, args
    )
    publish_closed_loop_run(output_dir, run)
    print("Manifest: {}".format(manifest_path))
    print("Policy bundle: {}".format(Path(args.policy_bundle).expanduser().resolve()))
    print("Closed-loop output: {}".format(output_dir))
    print("Run fingerprint: {}".format(run["fingerprint"]))
    print(
        "Frozen TAVRTrack policy: P{:.0f}, threshold {:.12g}, {}-seed ensemble".format(
            bundle["primary_budget_percent"],
            bundle["candidate_advantage_threshold"],
            len(bundle["seeds"]),
        )
    )
    print("Policies: {}".format(", ".join(policies)))
    print("Shard {}/{}: {} clip(s)".format(args.shard_id, args.num_shards, len(entries)))

    completed = 0
    for entry in entries:
        directory, ground_truth, valid = load_annotations(
            manifest["root"], manifest["dataset"], entry["sequence"]
        )
        if (
            file_hash(directory / "groundtruth.txt") != entry["annotation_sha256"]
            or fingerprint(valid.tolist()) != entry["visible_sha256"]
        ):
            raise ValueError("Annotations changed: {}".format(entry["sequence"]))
        for policy in policies:
            completed += int(
                run_policy_clip(
                    runtime,
                    controller,
                    directory,
                    manifest,
                    entry,
                    ground_truth,
                    valid,
                    policy,
                    bundle,
                    output_dir,
                    args.latency_warmup_frames,
                    args.verify_actions,
                    args.overwrite,
                )
            )
    print("Newly completed Stage-16C policy-clips: {}".format(completed))
    print(
        'Next: python tracking/stage16c_closed_loop_summary.py --input "{}"'.format(
            output_dir
        )
    )


if __name__ == "__main__":
    main()
