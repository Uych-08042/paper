"""Train and audit the Stage-16B recurrent conservative offline Q model.

This is a development-only feasibility screen. It trains on the frozen
Stage-16 train split, uses a video-disjoint tune subset for early stopping,
and touches the frozen validation split once after all model seeds are fixed.
Passing this script permits a true closed-loop Stage-16C run; it is not itself
a tracking benchmark result.
"""

import argparse
import copy
import json
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint, write_csv, write_json
from stage8_intervention_common import OVERLAP_THRESHOLDS
from stage16_behavior_common import ACTION_NAMES, behavior_cache_path
from stage16_freeze_cache import verify_frozen_cache


SCHEMA = "stage16b_recurrent_cql_v1"
PREPARED_SCHEMA = "stage16b_prepared_v1"
SCALAR_NAMES = (
    *("selector_p{}".format(layer) for layer in range(7, 13)),
    *("selector_layer{}".format(layer) for layer in range(7, 13)),
    *("previous_state_{}".format(name) for name in ("x", "y", "w", "h")),
    *("previous_fast_{}".format(name) for name in ("x", "y", "w", "h")),
    *("fast_box_{}".format(name) for name in ("x", "y", "w", "h")),
    *("state_minus_fast_{}".format(name) for name in ("x", "y", "w", "h")),
    *("previous_action_{}".format(index) for index in range(8)),
    "remaining_candidate_rate",
    "target_candidate_rate",
    "budget_progress",
    "time_fraction",
    "log_resize_factor",
)


def _safe_pearson(left, right):
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    good = np.isfinite(left) & np.isfinite(right)
    left, right = left[good], right[good]
    if left.size < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def discounted_returns(reward, valid, gamma):
    reward = np.asarray(reward, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if reward.shape != valid.shape:
        raise ValueError("Reward and validity masks must have identical shapes")
    output = np.zeros_like(reward, dtype=np.float64)
    running = np.zeros(reward.shape[:-1], dtype=np.float64)
    for index in range(reward.shape[-1] - 1, -1, -1):
        running = np.where(valid[..., index], reward[..., index], 0.0) + gamma * running
        output[..., index] = running
    return output


def success_reward(iou):
    values = np.asarray(iou, dtype=np.float64)
    return np.mean(values[..., None] > OVERLAP_THRESHOLDS, axis=-1).astype(np.float32)


def grouped_fit_tune(entries, tune_fraction, seed):
    train = [entry for entry in entries if entry["split"] == "train"]
    if len(train) < 3 or not 0.0 < tune_fraction < 0.5:
        raise ValueError("Need at least three training videos and tune_fraction in (0,0.5)")
    ordered = sorted(train, key=lambda entry: entry["sequence"])
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(ordered))
    tune_count = max(1, min(len(ordered) - 1, int(round(len(ordered) * tune_fraction))))
    tune_index = set(order[:tune_count].tolist())
    fit = [entry for index, entry in enumerate(ordered) if index not in tune_index]
    tune = [entry for index, entry in enumerate(ordered) if index in tune_index]
    return fit, tune


def _sample_token_vectors(directory, run, entries, maximum, seed):
    rng = np.random.default_rng(int(seed))
    per_file = max(64, int(math.ceil(maximum / max(1, len(entries)))))
    chunks = []
    for entry in entries:
        path = behavior_cache_path(directory, entry)
        with np.load(str(path), allow_pickle=False) as archive:
            grid = archive["search_token_grid"].reshape(-1, run["runtime"]["embed_dim"])
            template = archive["template_summary"].reshape(-1, run["runtime"]["embed_dim"])
            grid_count = min(len(grid), per_file)
            template_count = min(len(template), max(16, per_file // 16))
            grid_index = rng.choice(len(grid), size=grid_count, replace=False)
            template_index = rng.choice(len(template), size=template_count, replace=False)
            chunks.append(np.asarray(grid[grid_index], dtype=np.float32))
            chunks.append(np.asarray(template[template_index], dtype=np.float32))
    values = np.concatenate(chunks, axis=0)
    if len(values) > maximum:
        values = values[rng.choice(len(values), size=maximum, replace=False)]
    return values


def fit_token_pca(directory, run, fit_entries, dimensions, maximum_samples, seed):
    try:
        from sklearn.decomposition import PCA
    except ImportError as error:
        raise RuntimeError("Stage-16B requires scikit-learn for train-only token PCA") from error
    samples = _sample_token_vectors(directory, run, fit_entries, maximum_samples, seed)
    maximum_dimensions = min(samples.shape[0] - 1, samples.shape[1])
    if not 1 <= dimensions <= maximum_dimensions:
        raise ValueError("token_pca_dims must be in [1,{}]".format(maximum_dimensions))
    pca = PCA(
        n_components=int(dimensions),
        svd_solver="randomized",
        iterated_power=2,
        random_state=int(seed),
    )
    pca.fit(samples)
    return {
        "mean": np.asarray(pca.mean_, dtype=np.float32),
        "components": np.asarray(pca.components_, dtype=np.float32),
        "explained_variance_ratio": np.asarray(
            pca.explained_variance_ratio_, dtype=np.float32
        ),
        "sample_count": int(len(samples)),
    }


def project_tokens(values, pca, chunk_size=65536):
    values = np.asarray(values)
    original_shape = values.shape[:-1]
    flat = values.reshape(-1, values.shape[-1])
    output = np.empty((len(flat), len(pca["components"])), dtype=np.float32)
    for start in range(0, len(flat), chunk_size):
        stop = min(len(flat), start + chunk_size)
        centered = np.asarray(flat[start:stop], dtype=np.float32) - pca["mean"]
        output[start:stop] = centered @ pca["components"].T
    return output.reshape(original_shape + (output.shape[-1],))


def _normalized_boxes(boxes, image_hw):
    boxes = np.asarray(boxes, dtype=np.float64)
    image_hw = np.asarray(image_hw, dtype=np.float64)
    scale = np.stack(
        (image_hw[:, 1], image_hw[:, 0], image_hw[:, 1], image_hw[:, 0]), axis=1
    )
    return (boxes / scale[None]).astype(np.float32)


def scalar_features(data):
    actions = np.asarray(data["previous_action"], dtype=np.int64)
    episodes, steps = actions.shape
    previous_one_hot = np.eye(8, dtype=np.float32)[actions + 1]
    selected = np.asarray(data["selected_layer"], dtype=np.int64) - 7
    selected_one_hot = np.eye(6, dtype=np.float32)[selected]
    previous_state = _normalized_boxes(data["previous_state"], data["image_hw"])
    previous_fast = _normalized_boxes(data["previous_fast_box"], data["image_hw"])
    fast_box = _normalized_boxes(data["fast_box"], data["image_hw"])
    total = data["candidate_used_before"] + data["candidate_remaining_before"]
    target_rate = total / float(steps)
    budget_progress = np.divide(
        data["candidate_used_before"],
        np.maximum(total, 1),
        out=np.zeros_like(total, dtype=np.float64),
        where=total > 0,
    )
    time_fraction = np.broadcast_to(
        np.arange(steps, dtype=np.float64) / max(1, steps - 1), (episodes, steps)
    )
    features = np.concatenate(
        (
            np.asarray(data["selector_prob"], dtype=np.float32),
            selected_one_hot,
            previous_state,
            previous_fast,
            fast_box,
            previous_state - previous_fast,
            previous_one_hot,
            np.asarray(data["remaining_candidate_rate"], dtype=np.float32)[..., None],
            np.asarray(target_rate, dtype=np.float32)[..., None],
            np.asarray(budget_progress, dtype=np.float32)[..., None],
            np.asarray(time_fraction, dtype=np.float32)[..., None],
            np.log(np.asarray(data["resize_factor"], dtype=np.float32))[..., None],
        ),
        axis=2,
    )
    if features.shape != (episodes, steps, len(SCALAR_NAMES)):
        raise RuntimeError("Stage-16B scalar feature contract changed")
    return features


def _atomic_savez(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=str(path.parent), suffix=".tmp", delete=False
    ) as handle:
        temporary = handle.name
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, str(path))


def _prepared_path(output, entry):
    return Path(output) / entry["split"] / "{}__stage16b.npz".format(entry["sequence"])


def _save_pca(path, pca):
    _atomic_savez(
        path,
        mean=pca["mean"],
        components=pca["components"],
        explained_variance_ratio=pca["explained_variance_ratio"],
        sample_count=np.asarray(pca["sample_count"], dtype=np.int64),
    )


def _load_pca(path):
    with np.load(str(path), allow_pickle=False) as archive:
        return {
            "mean": archive["mean"],
            "components": archive["components"],
            "explained_variance_ratio": archive["explained_variance_ratio"],
            "sample_count": int(archive["sample_count"].item()),
        }


def prepare_dataset(directory, run, frozen, output, fit_entries, tune_entries, args,
                    entries_to_prepare=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pca_path = output / "stage16b_token_pca.npz"
    metadata_path = output / "stage16b_prepared_run.json"
    config = {
        "schema": PREPARED_SCHEMA,
        "source_run_fingerprint": run["fingerprint"],
        "source_freeze_fingerprint": frozen["fingerprint"],
        "token_pca_dims": int(args.token_pca_dims),
        "token_pca_samples": int(args.token_pca_samples),
        "split_seed": int(args.split_seed),
        "tune_fraction": float(args.tune_fraction),
        "fit_sequences": [entry["sequence"] for entry in fit_entries],
        "tune_sequences": [entry["sequence"] for entry in tune_entries],
        "scalar_names": list(SCALAR_NAMES),
    }
    config["fingerprint"] = fingerprint(config)

    if metadata_path.is_file():
        observed = json.loads(metadata_path.read_text(encoding="utf-8"))
        if observed != config or not pca_path.is_file():
            raise ValueError(
                "Prepared Stage-16B directory belongs to a different frozen configuration"
            )
        pca = _load_pca(pca_path)
    else:
        existing = list(output.rglob("*__stage16b.npz"))
        if existing:
            raise ValueError("Unidentified prepared files exist without prepared metadata")
        print("Fitting train-only token PCA...", flush=True)
        pca = fit_token_pca(
            directory,
            run,
            fit_entries,
            args.token_pca_dims,
            args.token_pca_samples,
            args.split_seed,
        )
        _save_pca(pca_path, pca)
        write_json(metadata_path, config)

    entries_to_prepare = list(
        run["manifest"]["sequences"]
        if entries_to_prepare is None
        else entries_to_prepare
    )
    for index, entry in enumerate(entries_to_prepare, start=1):
        target = _prepared_path(output, entry)
        if target.is_file():
            with np.load(str(target), allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata.get("prepared_fingerprint") != config["fingerprint"]:
                raise ValueError("Prepared cache fingerprint mismatch: {}".format(target))
            continue
        source = behavior_cache_path(directory, entry)
        with np.load(str(source), allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files if key != "metadata_json"}
        valid = np.broadcast_to(data["valid_gt"][None], data["behavior_action"].shape)
        arrays = {
            "search_grid": project_tokens(data["search_token_grid"], pca).astype(np.float16),
            "template": project_tokens(data["template_summary"], pca).astype(np.float16),
            "score_map": np.asarray(data["fast_score_map"], dtype=np.float16),
            "scalar": scalar_features(data),
            "action": np.asarray(data["behavior_action"], dtype=np.int64),
            "action_propensity": np.asarray(data["action_propensity"], dtype=np.float32),
            "behavior_probability": np.asarray(data["behavior_probability"], dtype=np.float32),
            "reward": success_reward(data["executed_iou"]),
            "action_reward": success_reward(data["action_iou"]),
            "valid": np.asarray(valid, dtype=bool),
            "done": np.broadcast_to(data["done"][None], valid.shape).copy(),
            "remaining_count": np.asarray(data["candidate_remaining_before"], dtype=np.int64),
            "native_action": np.asarray(data["selected_layer"] - 6, dtype=np.int64),
        }
        metadata = {
            "prepared_fingerprint": config["fingerprint"],
            "source_run_fingerprint": run["fingerprint"],
            "entry": entry,
        }
        arrays["metadata_json"] = np.asarray(json.dumps(metadata, allow_nan=False))
        _atomic_savez(target, **arrays)
        if index % 10 == 0 or index == len(entries_to_prepare):
            print("[prepare] {}/{} requested clips".format(index, len(entries_to_prepare)), flush=True)

    scalar_sum = np.zeros(len(SCALAR_NAMES), dtype=np.float64)
    scalar_square = np.zeros(len(SCALAR_NAMES), dtype=np.float64)
    scalar_count = 0
    fit_names = {entry["sequence"] for entry in fit_entries}
    for entry in run["manifest"]["sequences"]:
        if entry["sequence"] not in fit_names:
            continue
        with np.load(str(_prepared_path(output, entry)), allow_pickle=False) as archive:
            values = np.asarray(archive["scalar"], dtype=np.float64).reshape(-1, len(SCALAR_NAMES))
        scalar_sum += values.sum(axis=0)
        scalar_square += np.square(values).sum(axis=0)
        scalar_count += len(values)
    mean = scalar_sum / scalar_count
    variance = np.maximum(scalar_square / scalar_count - np.square(mean), 1e-8)
    scale = np.sqrt(variance)
    normalization_path = output / "stage16b_scalar_normalization.npz"
    _atomic_savez(
        normalization_path,
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        names=np.asarray(SCALAR_NAMES),
    )
    return config, pca, mean.astype(np.float32), scale.astype(np.float32)


class PreparedClipDataset:
    def __init__(self, prepared_dir, entries):
        self.prepared_dir = Path(prepared_dir)
        self.entries = list(entries)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        path = _prepared_path(self.prepared_dir, entry)
        with np.load(str(path), allow_pickle=False) as archive:
            values = {
                key: archive[key]
                for key in (
                    "search_grid",
                    "template",
                    "score_map",
                    "scalar",
                    "action",
                    "action_propensity",
                    "behavior_probability",
                    "reward",
                    "action_reward",
                    "valid",
                    "done",
                    "remaining_count",
                    "native_action",
                )
            }
        values["sequence"] = entry["sequence"]
        values["split"] = entry["split"]
        return values


def collate_clips(items, torch, scalar_mean, scalar_scale):
    if not items:
        raise ValueError("Cannot collate an empty Stage-16B batch")
    episodes = items[0]["action"].shape[0]
    maximum = max(item["action"].shape[1] for item in items)
    flattened = len(items) * episodes

    def padded(name, tail_shape, dtype):
        return np.zeros((flattened, maximum) + tuple(tail_shape), dtype=dtype)

    example = items[0]
    arrays = {
        "search_grid": padded("search_grid", example["search_grid"].shape[2:], np.float32),
        "template": padded("template", example["template"].shape[2:], np.float32),
        "score_map": padded("score_map", example["score_map"].shape[2:], np.float32),
        "scalar": padded("scalar", (len(SCALAR_NAMES),), np.float32),
        "action": padded("action", (), np.int64),
        "reward": padded("reward", (), np.float32),
        "valid": padded("valid", (), bool),
        "done": padded("done", (), bool),
        "remaining_count": padded("remaining_count", (), np.int64),
    }
    mask = np.zeros((flattened, maximum), dtype=bool)
    sequences = []
    lengths = []
    for clip_index, item in enumerate(items):
        steps = item["action"].shape[1]
        for episode in range(episodes):
            row = clip_index * episodes + episode
            for key in arrays:
                arrays[key][row, :steps] = item[key][episode]
            mask[row, :steps] = True
            sequences.append(item["sequence"])
            lengths.append(steps)
    arrays["scalar"] = (arrays["scalar"] - scalar_mean) / scalar_scale
    batch = {key: torch.from_numpy(value) for key, value in arrays.items()}
    batch["mask"] = torch.from_numpy(mask)
    batch["lengths"] = torch.as_tensor(lengths, dtype=torch.long)
    batch["sequences"] = sequences
    return batch


def build_q_model(torch, nn, config):
    class RecurrentQ(nn.Module):
        def __init__(self):
            super().__init__()
            token_dim = config["token_dim"]
            channels = config["spatial_channels"]
            self.spatial = nn.Sequential(
                nn.Conv2d(token_dim, channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(channels, channels, 3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((2, 2)),
            )
            self.template = nn.Sequential(
                nn.Linear(2 * token_dim, 48), nn.GELU(), nn.LayerNorm(48)
            )
            self.score = nn.Sequential(
                nn.Conv2d(1, 8, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(8, 16, 3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((2, 2)),
            )
            self.scalar = nn.Sequential(
                nn.Linear(config["scalar_dim"], 64), nn.GELU(), nn.LayerNorm(64)
            )
            fusion_dim = 4 * channels + 48 + 64 + 64
            self.fusion = nn.Sequential(
                nn.Linear(fusion_dim, config["hidden_dim"]),
                nn.GELU(),
                nn.LayerNorm(config["hidden_dim"]),
            )
            self.gru = nn.GRU(
                config["hidden_dim"], config["hidden_dim"], batch_first=True
            )
            self.q_head = nn.Sequential(
                nn.LayerNorm(config["hidden_dim"]),
                nn.Linear(config["hidden_dim"], config["hidden_dim"]),
                nn.GELU(),
                nn.Linear(config["hidden_dim"], len(ACTION_NAMES)),
            )

        def forward(self, batch):
            batch_size, steps = batch["search_grid"].shape[:2]
            grid = batch["search_grid"].reshape(
                -1, *batch["search_grid"].shape[2:]
            ).permute(0, 3, 1, 2)
            spatial = self.spatial(grid).flatten(1)
            template = self.template(batch["template"].reshape(batch_size * steps, -1))
            score = self.score(
                batch["score_map"].reshape(-1, *batch["score_map"].shape[2:])
            ).flatten(1)
            scalar = self.scalar(batch["scalar"].reshape(batch_size * steps, -1))
            fused = self.fusion(torch.cat((spatial, template, score, scalar), dim=1))
            recurrent, _ = self.gru(fused.reshape(batch_size, steps, -1))
            return self.q_head(recurrent)

    return RecurrentQ()


def _move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
        for key, value in batch.items()
        if key != "sequences"
    }


def recurrent_cql_loss(torch, functional, q, target_q, batch, args):
    action = batch["action"]
    mask = batch["mask"] & batch["valid"]
    q_taken = q.gather(2, action.unsqueeze(2)).squeeze(2)
    with torch.no_grad():
        online_next = q[:, 1:].detach().clone()
        no_budget = batch["remaining_count"][:, 1:] <= 0
        online_next[:, :, 1:] = online_next[:, :, 1:].masked_fill(
            no_budget.unsqueeze(2), -1e9
        )
        next_action = online_next.argmax(dim=2)
        target_next = target_q[:, 1:].gather(2, next_action.unsqueeze(2)).squeeze(2)
        bootstrap = torch.zeros_like(batch["reward"])
        bootstrap[:, :-1] = target_next
        td_target = batch["reward"] + args.gamma * (~batch["done"]).float() * bootstrap
    denominator = mask.float().sum().clamp_min(1.0)
    td = functional.smooth_l1_loss(q_taken, td_target, reduction="none")
    td_loss = (td * mask.float()).sum() / denominator
    conservative = (
        args.cql_temperature
        * torch.logsumexp(q / args.cql_temperature, dim=2)
        - q_taken
    )
    cql_loss = (conservative * mask.float()).sum() / denominator
    behavior_clone = functional.cross_entropy(
        (q / args.policy_temperature).reshape(-1, len(ACTION_NAMES)),
        action.reshape(-1),
        reduction="none",
    ).reshape_as(action)
    bc_loss = (behavior_clone * mask.float()).sum() / denominator
    total = td_loss + args.cql_alpha * cql_loss + args.bc_weight * bc_loss
    return total, td_loss, cql_loss, bc_loss, q_taken, td_target, mask


def _loader(torch, dataset, batch_size, shuffle, seed, scalar_mean, scalar_scale):
    from torch.utils.data import DataLoader

    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        collate_fn=lambda items: collate_clips(
            items, torch, scalar_mean, scalar_scale
        ),
    )


def _set_seed(torch, seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def evaluate_model(torch, functional, model, target, loader, device, args):
    model.eval()
    target.eval()
    losses = []
    predicted, observed, centered_predicted, centered_observed = [], [], [], []
    initial_predicted, initial_observed = [], []
    with torch.no_grad():
        for raw in loader:
            batch = _move_batch(raw, device)
            q = model(batch)
            target_q = target(batch)
            total, td, cql, bc, q_taken, _, mask = recurrent_cql_loss(
                torch, functional, q, target_q, batch, args
            )
            losses.append((float(total.item()), float(td.item()), float(cql.item()), float(bc.item())))
            returns = discounted_returns(
                batch["reward"].cpu().numpy(),
                batch["valid"].cpu().numpy(),
                args.gamma,
            )
            q_numpy = q_taken.cpu().numpy()
            mask_numpy = mask.cpu().numpy()
            predicted.append(q_numpy[mask_numpy])
            observed.append(returns[mask_numpy])
            for row, length in enumerate(batch["lengths"].cpu().numpy()):
                good = mask_numpy[row, :length]
                if good.any():
                    left = q_numpy[row, :length][good]
                    right = returns[row, :length][good]
                    centered_predicted.append(left - left.mean())
                    centered_observed.append(right - right.mean())
                    initial_predicted.append(left[0])
                    initial_observed.append(right[0])
    values = np.asarray(losses, dtype=np.float64)
    return {
        "loss": float(values[:, 0].mean()),
        "td_loss": float(values[:, 1].mean()),
        "cql_loss": float(values[:, 2].mean()),
        "bc_loss": float(values[:, 3].mean()),
        "return_pearson": _safe_pearson(np.concatenate(predicted), np.concatenate(observed)),
        "centered_return_pearson": _safe_pearson(
            np.concatenate(centered_predicted), np.concatenate(centered_observed)
        ),
        "initial_return_pearson": _safe_pearson(initial_predicted, initial_observed),
    }


def train_one_seed(torch, nn, functional, prepared_dir, fit_entries, tune_entries,
                   scalar_mean, scalar_scale, model_config, args, seed, output):
    _set_seed(torch, seed)
    device = torch.device(args.device)
    model = build_q_model(torch, nn, model_config).to(device)
    target = copy.deepcopy(model).to(device).eval()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    fit_dataset = PreparedClipDataset(prepared_dir, fit_entries)
    tune_dataset = PreparedClipDataset(prepared_dir, tune_entries)
    tune_loader = _loader(
        torch, tune_dataset, args.batch_size, False, seed, scalar_mean, scalar_scale
    )
    best_state = None
    best_epoch = 0
    best_score = -float("inf")
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loader = _loader(
            torch,
            fit_dataset,
            args.batch_size,
            True,
            seed + epoch,
            scalar_mean,
            scalar_scale,
        )
        train_values = []
        for raw in loader:
            batch = _move_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            q = model(batch)
            with torch.no_grad():
                target_q = target(batch)
            losses = recurrent_cql_loss(torch, functional, q, target_q, batch, args)
            losses[0].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            with torch.no_grad():
                for target_parameter, parameter in zip(target.parameters(), model.parameters()):
                    target_parameter.mul_(1.0 - args.target_tau).add_(
                        parameter, alpha=args.target_tau
                    )
            train_values.append([float(value.item()) for value in losses[:4]])
        tune = evaluate_model(torch, functional, model, target, tune_loader, device, args)
        train_mean = np.asarray(train_values, dtype=np.float64).mean(axis=0)
        row = {
            "seed": int(seed),
            "epoch": epoch,
            "train_loss": train_mean[0],
            "train_td_loss": train_mean[1],
            "train_cql_loss": train_mean[2],
            "train_bc_loss": train_mean[3],
            **{"tune_{}".format(key): value for key, value in tune.items()},
        }
        history.append(row)
        score = tune["return_pearson"]
        if score > best_score + 1e-4:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        print(
            "[seed {} epoch {}/{}] train={:.5f} tune={:.5f} pearson={:.4f} centered={:.4f}".format(
                seed,
                epoch,
                args.epochs,
                train_mean[0],
                tune["loss"],
                tune["return_pearson"],
                tune["centered_return_pearson"],
            ),
            flush=True,
        )
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Stage-16B training did not produce a checkpoint")
    model.load_state_dict(best_state)
    target.load_state_dict(best_state)
    checkpoint = {
        "schema": SCHEMA,
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_tune_return_pearson": float(best_score),
        "model_config": model_config,
        "state_dict": best_state,
        "scalar_mean": scalar_mean,
        "scalar_scale": scalar_scale,
    }
    checkpoint_path = Path(output) / "stage16b_seed_{}.pt".format(seed)
    torch.save(checkpoint, str(checkpoint_path))
    return model, history, checkpoint_path, best_epoch, best_score


def predict_records(torch, models, dataset, scalar_mean, scalar_scale, args):
    loader = _loader(
        torch, dataset, args.batch_size, False, args.split_seed, scalar_mean, scalar_scale
    )
    device = torch.device(args.device)
    records = []
    offset = 0
    for raw in loader:
        batch = _move_batch(raw, device)
        with torch.no_grad():
            q = torch.stack([model(batch) for model in models], dim=0).mean(dim=0)
        q = q.cpu().numpy()
        clip_count = len(raw["sequences"]) // len(args.behavior_ids)
        for clip in range(clip_count):
            entry_item = dataset[offset + clip]
            steps = entry_item["action"].shape[1]
            start = clip * len(args.behavior_ids)
            stop = start + len(args.behavior_ids)
            records.append(
                {
                    **entry_item,
                    "q": q[start:stop, :steps],
                }
            )
        offset += clip_count
    return records


def value_diagnostics(records, gamma):
    predicted, observed, centered_left, centered_right = [], [], [], []
    distribution = np.zeros(len(ACTION_NAMES), dtype=np.int64)
    for record in records:
        returns = discounted_returns(record["reward"], record["valid"], gamma)
        chosen = np.take_along_axis(record["q"], record["action"][..., None], axis=2)[..., 0]
        good = record["valid"]
        predicted.append(chosen[good])
        observed.append(returns[good])
        for episode in range(len(chosen)):
            mask = good[episode]
            if mask.any():
                left = chosen[episode, mask]
                right = returns[episode, mask]
                centered_left.append(left - left.mean())
                centered_right.append(right - right.mean())
        distribution += np.bincount(
            np.argmax(record["q"], axis=2).reshape(-1), minlength=len(ACTION_NAMES)
        )
    predicted = np.concatenate(predicted)
    observed = np.concatenate(observed)
    advantage = np.concatenate(
        [record["q"][..., 1:].max(axis=2).reshape(-1) - record["q"][..., 0].reshape(-1)
         for record in records]
    )
    return {
        "num_sequences": len(records),
        "num_transitions": int(sum(record["action"].size for record in records)),
        "return_pearson": _safe_pearson(predicted, observed),
        "centered_return_pearson": _safe_pearson(
            np.concatenate(centered_left), np.concatenate(centered_right)
        ),
        "q_taken_mean": float(predicted.mean()),
        "q_taken_std": float(predicted.std()),
        "candidate_advantage_std": float(advantage.std()),
        "greedy_distribution": "|".join(
            "{}:{}".format(name, distribution[index])
            for index, name in enumerate(ACTION_NAMES)
        ),
    }


def _behavior_indices(args, budget):
    prefix = "p{}_".format(("{:g}".format(float(budget))).replace(".", "p"))
    return [index for index, name in enumerate(args.behavior_ids) if name.startswith(prefix)]


def calibrate_threshold(records, args, budget):
    indices = _behavior_indices(args, budget)
    if not indices:
        raise ValueError("No behavior episodes match budget {}".format(budget))
    scores = []
    for record in records:
        q = record["q"][indices]
        good = record["valid"][indices]
        available = record["remaining_count"][indices] > 0
        advantage = q[..., 1:].max(axis=2) - q[..., 0]
        scores.append(advantage[good & available])
    scores = np.concatenate(scores)
    if not len(scores):
        raise ValueError("No valid candidate-supported states for budget {}".format(budget))
    return float(np.quantile(scores, 1.0 - float(budget) / 100.0))


def _cluster_bootstrap(values_by_sequence, samples, seed):
    values = np.asarray(values_by_sequence, dtype=np.float64)
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(values), size=(int(samples), len(values)))
    means = values[draws].mean(axis=1)
    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
        float(np.mean(means > 0.0)),
    )


def budget_diagnostic(records, args, budget, threshold, seed):
    indices = _behavior_indices(args, budget)
    sequence_lift = []
    sequence_fast = []
    action_counts = np.zeros(len(ACTION_NAMES), dtype=np.int64)
    selected = 0
    total = 0
    positive = 0
    harmful = 0
    for record in records:
        q = record["q"][indices]
        reward = record["action_reward"][indices]
        valid = record["valid"][indices]
        available = record["remaining_count"][indices] > 0
        candidate_layer = np.argmax(q[..., 1:], axis=2) + 1
        advantage = q[..., 1:].max(axis=2) - q[..., 0]
        choose_candidate = (advantage >= threshold) & available & valid
        policy_action = np.where(choose_candidate, candidate_layer, 0)
        chosen = np.take_along_axis(reward, policy_action[..., None], axis=2)[..., 0]
        fast = reward[..., 0]
        random_candidate = reward[..., 1:].mean(axis=2)
        random_matched = np.where(choose_candidate, random_candidate, fast)
        sequence_lift.append(float(np.mean((chosen - random_matched)[valid])))
        sequence_fast.append(float(np.mean((chosen - fast)[valid])))
        action_counts += np.bincount(policy_action[valid].reshape(-1), minlength=len(ACTION_NAMES))
        selected += int(choose_candidate.sum())
        total += int(valid.sum())
        selected_delta = (chosen - fast)[choose_candidate]
        positive += int(np.sum(selected_delta > 0.0))
        harmful += int(np.sum(selected_delta < 0.0))
    lift_ci = _cluster_bootstrap(sequence_lift, args.bootstrap_samples, seed)
    fast_ci = _cluster_bootstrap(sequence_fast, args.bootstrap_samples, seed + 1)
    candidate_counts = action_counts[1:]
    active_layers = int(np.sum(candidate_counts >= max(1, 0.05 * candidate_counts.sum())))
    return {
        "budget_percent": float(budget),
        "threshold": float(threshold),
        "num_sequences": len(records),
        "candidate_ratio": selected / total,
        "gain_vs_fast": float(np.mean(sequence_fast)),
        "gain_vs_fast_ci_low": fast_ci[0],
        "gain_vs_fast_ci_high": fast_ci[1],
        "gain_vs_fast_probability_positive": fast_ci[2],
        "lift_over_random_matched": float(np.mean(sequence_lift)),
        "lift_ci_low": lift_ci[0],
        "lift_ci_high": lift_ci[1],
        "lift_probability_positive": lift_ci[2],
        "selected_positive_ratio": positive / selected if selected else float("nan"),
        "selected_harmful_ratio": harmful / selected if selected else float("nan"),
        "active_candidate_layers": active_layers,
        "action_distribution": "|".join(
            "{}:{}".format(name, action_counts[index])
            for index, name in enumerate(ACTION_NAMES)
        ),
    }


def publish_experiment_contract(output, run, frozen, prepared_config, args):
    contract = {
        "schema": SCHEMA,
        "analysis_role": "predeclared_stage16b_training_contract",
        "source_run_fingerprint": run["fingerprint"],
        "source_freeze_fingerprint": frozen["fingerprint"],
        "prepared_fingerprint": prepared_config["fingerprint"],
        "seeds": list(args.seeds),
        "split_seed": args.split_seed,
        "tune_fraction": args.tune_fraction,
        "token_pca_dims": args.token_pca_dims,
        "token_pca_samples": args.token_pca_samples,
        "hidden_dim": args.hidden_dim,
        "spatial_channels": args.spatial_channels,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gamma": args.gamma,
        "cql_alpha": args.cql_alpha,
        "cql_temperature": args.cql_temperature,
        "bc_weight": args.bc_weight,
        "policy_temperature": args.policy_temperature,
        "target_tau": args.target_tau,
        "grad_clip": args.grad_clip,
        "budgets": list(args.budgets),
        "primary_budget": args.primary_budget,
        "bootstrap_samples": args.bootstrap_samples,
        "behavior_ids": list(args.behavior_ids),
        "screen": {
            "minimum_val_return_pearson": 0.10,
            "minimum_val_centered_return_pearson": 0.02,
            "primary_random_lift_ci_low_must_exceed_zero": True,
            "minimum_active_candidate_layers": 3,
        },
        "code_sha256": {
            "stage16_recurrent_cql.py": file_hash(Path(__file__).resolve()),
            "stage16_freeze_cache.py": file_hash(
                Path(__file__).resolve().parent / "stage16_freeze_cache.py"
            ),
        },
    }
    contract["fingerprint"] = fingerprint(contract)
    path = Path(output) / "stage16b_experiment.json"
    if path.is_file():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != contract:
            raise ValueError(
                "Stage-16B output belongs to a different experiment contract; use a new output_dir"
            )
    else:
        unexpected = [item for item in Path(output).iterdir() if item.name != "prepared"]
        if unexpected:
            raise ValueError("Unidentified Stage-16B output files exist; use a new output_dir")
        write_json(path, contract)
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--freeze_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2047, 2048, 2049])
    parser.add_argument("--split_seed", type=int, default=2047)
    parser.add_argument("--tune_fraction", type=float, default=0.2)
    parser.add_argument("--token_pca_dims", type=int, default=24)
    parser.add_argument("--token_pca_samples", type=int, default=100000)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--spatial_channels", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4, help="clips per batch; each clip has all episodes")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--cql_alpha", type=float, default=0.1)
    parser.add_argument("--cql_temperature", type=float, default=1.0)
    parser.add_argument("--bc_weight", type=float, default=0.05)
    parser.add_argument("--policy_temperature", type=float, default=1.0)
    parser.add_argument("--target_tau", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--budgets", nargs="+", type=float, default=[10.0, 30.0, 50.0])
    parser.add_argument("--primary_budget", type=float, default=30.0)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--skip_cache_hash_verification", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as error:
        raise RuntimeError("Stage-16B requires PyTorch") from error
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.primary_budget not in args.budgets:
        raise ValueError("primary_budget must be included in budgets")
    if not 0.0 < args.gamma < 1.0 or args.epochs < 1 or args.patience < 1:
        raise ValueError("Invalid discount or training schedule")

    directory, run, frozen = verify_frozen_cache(
        args.freeze_manifest,
        input_dir=args.input,
        verify_hashes=not args.skip_cache_hash_verification,
    )
    args.behavior_ids = [item["behavior_id"] for item in run["behaviors"]]
    for budget in args.budgets:
        if not _behavior_indices(args, budget):
            raise ValueError("Budget {} has no matching frozen behavior episodes".format(budget))
    fit_entries, tune_entries = grouped_fit_tune(
        run["manifest"]["sequences"], args.tune_fraction, args.split_seed
    )
    val_entries = [entry for entry in run["manifest"]["sequences"] if entry["split"] == "val"]
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepared_dir = output / "prepared"
    prepared, _, scalar_mean, scalar_scale = prepare_dataset(
        directory,
        run,
        frozen,
        prepared_dir,
        fit_entries,
        tune_entries,
        args,
        entries_to_prepare=fit_entries + tune_entries,
    )
    contract = publish_experiment_contract(output, run, frozen, prepared, args)
    completed_path = output / "stage16b_decision.json"
    if completed_path.is_file():
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        digest = completed.get("fingerprint")
        body = {key: value for key, value in completed.items() if key != "fingerprint"}
        if (
            completed.get("experiment_fingerprint") != contract["fingerprint"]
            or fingerprint(body) != digest
        ):
            raise ValueError("Existing Stage-16B decision does not match the frozen contract")
        print("Stage-16B already complete: {}".format(completed["status"]))
        print("Decision JSON: {}".format(completed_path))
        return

    model_config = {
        "token_dim": int(args.token_pca_dims),
        "scalar_dim": len(SCALAR_NAMES),
        "hidden_dim": int(args.hidden_dim),
        "spatial_channels": int(args.spatial_channels),
        "num_actions": len(ACTION_NAMES),
    }
    models = []
    history = []
    seed_rows = []
    checkpoint_paths = []
    for seed in args.seeds:
        model, rows, checkpoint_path, best_epoch, best_score = train_one_seed(
            torch,
            nn,
            functional,
            prepared_dir,
            fit_entries,
            tune_entries,
            scalar_mean,
            scalar_scale,
            model_config,
            args,
            seed,
            output,
        )
        model.eval()
        models.append(model)
        history.extend(rows)
        checkpoint_paths.append(checkpoint_path)
        seed_rows.append(
            {
                "seed": seed,
                "best_epoch": best_epoch,
                "best_tune_return_pearson": best_score,
                "checkpoint": str(checkpoint_path),
            }
        )

    fit_records = predict_records(
        torch,
        models,
        PreparedClipDataset(prepared_dir, fit_entries + tune_entries),
        scalar_mean,
        scalar_scale,
        args,
    )
    print("Models frozen; evaluating the untouched Stage-16 validation split once...", flush=True)
    prepare_dataset(
        directory,
        run,
        frozen,
        prepared_dir,
        fit_entries,
        tune_entries,
        args,
        entries_to_prepare=val_entries,
    )
    val_records = predict_records(
        torch,
        models,
        PreparedClipDataset(prepared_dir, val_entries),
        scalar_mean,
        scalar_scale,
        args,
    )
    train_value = value_diagnostics(fit_records, args.gamma)
    val_value = value_diagnostics(val_records, args.gamma)
    value_rows = [
        {"split": "train", **train_value},
        {"split": "val", **val_value},
    ]

    budget_rows = []
    for index, budget in enumerate(args.budgets):
        threshold = calibrate_threshold(fit_records, args, budget)
        train_row = budget_diagnostic(
            fit_records, args, budget, threshold, args.split_seed + 100 * index
        )
        val_row = budget_diagnostic(
            val_records, args, budget, threshold, args.split_seed + 100 * index + 50
        )
        budget_rows.extend(
            [
                {"split": "train", **train_row},
                {"split": "val", **val_row},
            ]
        )

    primary = next(
        row for row in budget_rows
        if row["split"] == "val" and row["budget_percent"] == args.primary_budget
    )
    screen = {
        "minimum_val_return_pearson": 0.10,
        "observed_val_return_pearson": val_value["return_pearson"],
        "return_prediction_pass": val_value["return_pearson"] >= 0.10,
        "minimum_val_centered_return_pearson": 0.02,
        "observed_val_centered_return_pearson": val_value["centered_return_pearson"],
        "centered_prediction_pass": val_value["centered_return_pearson"] >= 0.02,
        "primary_budget": args.primary_budget,
        "primary_lift_over_random": primary["lift_over_random_matched"],
        "primary_lift_ci_low": primary["lift_ci_low"],
        "random_lift_pass": primary["lift_ci_low"] > 0.0,
        "minimum_active_candidate_layers": 3,
        "observed_active_candidate_layers": primary["active_candidate_layers"],
        "noncollapse_pass": primary["active_candidate_layers"] >= 3,
    }
    passed = all(
        screen[key]
        for key in (
            "return_prediction_pass",
            "centered_prediction_pass",
            "random_lift_pass",
            "noncollapse_pass",
        )
    )
    decision = {
        "schema": SCHEMA,
        "status": "PASS_OFFLINE_Q_SCREEN" if passed else "FAIL_OFFLINE_Q_SCREEN",
        "analysis_role": "development_only_recurrent_conservative_offline_q_screen",
        "source_run_fingerprint": run["fingerprint"],
        "source_freeze_fingerprint": frozen["fingerprint"],
        "prepared_fingerprint": prepared["fingerprint"],
        "experiment_fingerprint": contract["fingerprint"],
        "fit_sequences": len(fit_entries),
        "tune_sequences": len(tune_entries),
        "val_sequences": len(val_entries),
        "seeds": list(args.seeds),
        "model_config": model_config,
        "training": {
            "gamma": args.gamma,
            "cql_alpha": args.cql_alpha,
            "bc_weight": args.bc_weight,
            "epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
        },
        "screen": screen,
        "next_step": (
            "Freeze the ensemble and run one true Stage-16C closed-loop development evaluation."
            if passed
            else "Stop this Stage-16B policy. Do not tune on the validation split; redesign the state, objective, or behavior data under a new versioned experiment."
        ),
        "warning": (
            "The budget table is a held-out support-state counterfactual diagnostic, not a "
            "closed-loop policy result. OTB and UAV123 remain unavailable for policy search."
        ),
        "artifacts": {
            "checkpoints": [str(path) for path in checkpoint_paths],
            "token_pca": str(prepared_dir / "stage16b_token_pca.npz"),
            "scalar_normalization": str(prepared_dir / "stage16b_scalar_normalization.npz"),
        },
        "code_sha256": {
            "stage16_recurrent_cql.py": file_hash(Path(__file__).resolve()),
            "stage16_freeze_cache.py": file_hash(
                Path(__file__).resolve().parent / "stage16_freeze_cache.py"
            ),
        },
    }
    decision["fingerprint"] = fingerprint(decision)
    write_csv(output / "stage16b_training_history.csv", history)
    write_csv(output / "stage16b_seed_summary.csv", seed_rows)
    write_csv(output / "stage16b_value_summary.csv", value_rows)
    write_csv(output / "stage16b_budget_summary.csv", budget_rows)
    write_json(output / "stage16b_decision.json", decision)

    print("\nFrozen Stage-16B value diagnostics:")
    print("split  sequences  transitions  pearson  centered  advantage_std")
    for row in value_rows:
        print(
            "{split:<6s} {num_sequences:>9d} {num_transitions:>12d} "
            "{return_pearson:>8.4f} {centered_return_pearson:>9.4f} "
            "{candidate_advantage_std:>13.6f}".format(**row)
        )
    print("\nHeld-out support-state budget diagnostic:")
    print("budget  actual  gain_fast  lift_random  lift_95%            layers")
    for row in budget_rows:
        if row["split"] != "val":
            continue
        print(
            "{budget_percent:>6.1f}% {actual:>6.2f}% {gain_vs_fast:>10.6f} "
            "{lift_over_random_matched:>11.6f} [{lift_ci_low:.6f},{lift_ci_high:.6f}] "
            "{active_candidate_layers:>6d}".format(
                actual=100.0 * row["candidate_ratio"], **row
            )
        )
    print("\nPredeclared Stage-16B decision: {}".format(decision["status"]))
    print("Decision JSON: {}".format(output / "stage16b_decision.json"))
    print("Warning: {}".format(decision["warning"]))


if __name__ == "__main__":
    main()
