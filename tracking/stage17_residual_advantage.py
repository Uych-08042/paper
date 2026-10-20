"""Stage-17A train-only OOF screen for a safe residual-advantage policy.

Stage-16B learned absolute returns well, but its action advantages did not
survive true closed-loop deployment.  This experiment therefore predicts the
dense, one-frame counterfactual residual

    A(s, Lk) = success(s, Lk) - success(s, Fast)

directly for L7-L12.  The screen is intentionally restricted to the frozen
Stage-16 training videos.  It never opens a prepared validation clip and does
not produce a deployable checkpoint.  A candidate action is accepted only
when an ensemble lower confidence bound is positive; each requested budget is
a hard cap rather than a quota.
"""

import argparse
import json
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint, write_csv, write_json
from stage16_behavior_common import ACTION_NAMES, CANDIDATE_LAYERS
from stage16_freeze_cache import verify_frozen_cache
from stage16_recurrent_cql import PREPARED_SCHEMA, PreparedClipDataset, SCALAR_NAMES


SCHEMA = "stage17_safe_residual_advantage_v1"
OOF_SCHEMA = "stage17_oof_prediction_v1"
EXCLUDED_BUDGET_SCALARS = {
    "remaining_candidate_rate",
    "target_candidate_rate",
    "budget_progress",
}
RESIDUAL_SCALAR_NAMES = tuple(
    name for name in SCALAR_NAMES if name not in EXCLUDED_BUDGET_SCALARS
)
RESIDUAL_SCALAR_INDEX = np.asarray(
    [index for index, name in enumerate(SCALAR_NAMES) if name in RESIDUAL_SCALAR_NAMES],
    dtype=np.int64,
)


def _safe_pearson(left, right):
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    good = np.isfinite(left) & np.isfinite(right)
    left, right = left[good], right[good]
    if left.size < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _safe_auroc(labels, scores):
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    good = np.isfinite(scores)
    labels, scores = labels[good], scores[good]
    if labels.size < 2 or np.unique(labels).size < 2:
        return 0.5
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as error:
        raise RuntimeError("Stage-17A requires scikit-learn") from error
    return float(roc_auc_score(labels, scores))


def grouped_outer_folds(entries, folds, seed):
    entries = sorted(entries, key=lambda entry: entry["sequence"])
    if folds < 2 or len(entries) < folds:
        raise ValueError("Need at least one video per outer fold")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(entries))
    chunks = np.array_split(order, int(folds))
    output = []
    for fold, test_index in enumerate(chunks):
        test_set = set(int(index) for index in test_index.tolist())
        test = [entry for index, entry in enumerate(entries) if index in test_set]
        train = [entry for index, entry in enumerate(entries) if index not in test_set]
        output.append((fold, train, test))
    return output


def grouped_fit_tune(entries, tune_fraction, seed):
    entries = sorted(entries, key=lambda entry: entry["sequence"])
    if len(entries) < 3 or not 0.0 < tune_fraction < 0.5:
        raise ValueError("Need at least three videos and tune_fraction in (0,0.5)")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(entries))
    tune_count = max(1, min(len(entries) - 1, int(round(len(entries) * tune_fraction))))
    tune_index = set(int(index) for index in order[:tune_count].tolist())
    fit = [entry for index, entry in enumerate(entries) if index not in tune_index]
    tune = [entry for index, entry in enumerate(entries) if index in tune_index]
    return fit, tune


def _prepared_path(directory, entry):
    return (
        Path(directory)
        / entry["split"]
        / "{}__stage16b.npz".format(entry["sequence"])
    )


def _load_fingerprinted_json(path, schema):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("Required contract missing: {}".format(path))
    value = json.loads(path.read_text(encoding="utf-8"))
    digest = value.get("fingerprint")
    body = {key: item for key, item in value.items() if key != "fingerprint"}
    if value.get("schema") != schema or digest is None or fingerprint(body) != digest:
        raise ValueError("Invalid contract or fingerprint: {}".format(path))
    return path, value


def audit_prepared_training_data(prepared_dir, run, frozen):
    prepared_dir = Path(prepared_dir).expanduser().resolve()
    contract_path, contract = _load_fingerprinted_json(
        prepared_dir / "stage16b_prepared_run.json", PREPARED_SCHEMA
    )
    if (
        contract.get("source_run_fingerprint") != run["fingerprint"]
        or contract.get("source_freeze_fingerprint") != frozen["fingerprint"]
        or contract.get("scalar_names") != list(SCALAR_NAMES)
    ):
        raise ValueError("Prepared Stage-16B features do not match the frozen cache")
    pca_path = prepared_dir / "stage16b_token_pca.npz"
    if not pca_path.is_file():
        raise FileNotFoundError("Prepared token PCA missing: {}".format(pca_path))
    with np.load(str(pca_path), allow_pickle=False) as archive:
        components = archive["components"]
        mean = archive["mean"]
    if components.ndim != 2 or mean.shape != (components.shape[1],):
        raise ValueError("Invalid Stage-16B token PCA")

    train_entries = [
        entry for entry in run["manifest"]["sequences"] if entry["split"] == "train"
    ]
    validation_entries = [
        entry for entry in run["manifest"]["sequences"] if entry["split"] == "val"
    ]
    if not train_entries or not validation_entries:
        raise ValueError("Stage-17A requires the original train/validation separation")
    for entry in train_entries:
        path = _prepared_path(prepared_dir, entry)
        if not path.is_file():
            raise FileNotFoundError("Prepared training clip missing: {}".format(path))
        with np.load(str(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata.get("prepared_fingerprint") != contract["fingerprint"]:
                raise ValueError("Prepared clip fingerprint mismatch: {}".format(path))
            if archive["search_grid"].shape[-1] != components.shape[0]:
                raise ValueError("Prepared token dimension mismatch: {}".format(path))
    return {
        "directory": prepared_dir,
        "contract_path": contract_path,
        "contract": contract,
        "pca_path": pca_path,
        "token_dim": int(components.shape[0]),
        "train_entries": train_entries,
        "validation_entries": validation_entries,
    }


def scalar_normalization(prepared_dir, entries):
    total = np.zeros(len(RESIDUAL_SCALAR_NAMES), dtype=np.float64)
    square = np.zeros(len(RESIDUAL_SCALAR_NAMES), dtype=np.float64)
    count = 0
    for entry in entries:
        with np.load(str(_prepared_path(prepared_dir, entry)), allow_pickle=False) as archive:
            values = np.asarray(archive["scalar"], dtype=np.float64)[
                ..., RESIDUAL_SCALAR_INDEX
            ].reshape(
                -1, len(RESIDUAL_SCALAR_NAMES)
            )
        total += values.sum(axis=0)
        square += np.square(values).sum(axis=0)
        count += len(values)
    if count == 0:
        raise ValueError("Cannot normalize an empty Stage-17A fold")
    mean = total / count
    variance = np.maximum(square / count - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def collate_advantage_clips(items, torch, scalar_mean, scalar_scale):
    if not items:
        raise ValueError("Cannot collate an empty Stage-17A batch")
    episodes = items[0]["action_reward"].shape[0]
    if any(item["action_reward"].shape[0] != episodes for item in items):
        raise ValueError("All Stage-17A clips must contain the same behavior episodes")
    maximum = max(item["action_reward"].shape[1] for item in items)
    flattened = len(items) * episodes
    example = items[0]

    def padded(tail, dtype):
        return np.zeros((flattened, maximum) + tuple(tail), dtype=dtype)

    arrays = {
        "search_grid": padded(example["search_grid"].shape[2:], np.float32),
        "template": padded(example["template"].shape[2:], np.float32),
        "score_map": padded(example["score_map"].shape[2:], np.float32),
        "scalar": padded((len(RESIDUAL_SCALAR_NAMES),), np.float32),
        "target": padded((len(CANDIDATE_LAYERS),), np.float32),
        "valid": padded((), bool),
    }
    mask = np.zeros((flattened, maximum), dtype=bool)
    lengths = []
    sequences = []
    for clip_index, item in enumerate(items):
        steps = item["action_reward"].shape[1]
        target = np.asarray(item["action_reward"], dtype=np.float32)
        target = target[..., 1:] - target[..., :1]
        for episode in range(episodes):
            row = clip_index * episodes + episode
            arrays["search_grid"][row, :steps] = item["search_grid"][episode]
            arrays["template"][row, :steps] = item["template"][episode]
            arrays["score_map"][row, :steps] = item["score_map"][episode]
            arrays["scalar"][row, :steps] = np.take(
                item["scalar"][episode], RESIDUAL_SCALAR_INDEX, axis=-1
            )
            arrays["target"][row, :steps] = target[episode]
            arrays["valid"][row, :steps] = item["valid"][episode]
            mask[row, :steps] = True
            lengths.append(steps)
            sequences.append(item["sequence"])
    arrays["scalar"] = (arrays["scalar"] - scalar_mean) / scalar_scale
    batch = {key: torch.from_numpy(value) for key, value in arrays.items()}
    batch["mask"] = torch.from_numpy(mask)
    batch["lengths"] = torch.as_tensor(lengths, dtype=torch.long)
    batch["sequences"] = sequences
    batch["episodes"] = episodes
    batch["clip_count"] = len(items)
    return batch


def build_advantage_model(torch, nn, config):
    class RecurrentAdvantage(nn.Module):
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
            self.advantage_head = nn.Sequential(
                nn.LayerNorm(config["hidden_dim"]),
                nn.Linear(config["hidden_dim"], config["hidden_dim"]),
                nn.GELU(),
                nn.Linear(config["hidden_dim"], len(CANDIDATE_LAYERS)),
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
            return self.advantage_head(recurrent)

    return RecurrentAdvantage()


def residual_loss(torch, prediction, batch, material_threshold, material_weight):
    target = batch["target"]
    valid = (batch["mask"] & batch["valid"]).unsqueeze(2)
    weight = 1.0 + float(material_weight) * (
        target.abs() >= float(material_threshold)
    ).float()
    denominator = (weight * valid.float()).sum().clamp_min(1.0)
    squared = torch.square(prediction - target)
    weighted = (squared * weight * valid.float()).sum() / denominator
    expanded_valid = valid.expand_as(squared)
    plain = (squared * expanded_valid.float()).sum() / expanded_valid.float().sum().clamp_min(1.0)
    return weighted, plain, valid


def _move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
        for key, value in batch.items()
        if key not in ("sequences", "episodes", "clip_count")
    }


def _set_seed(torch, seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _loader(torch, dataset, batch_size, shuffle, seed, mean, scale):
    from torch.utils.data import DataLoader

    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        collate_fn=lambda items: collate_advantage_clips(
            items, torch, mean, scale
        ),
    )


def evaluate_model(torch, model, loader, device, args):
    model.eval()
    weighted_losses = []
    plain_losses = []
    predicted = []
    observed = []
    with torch.no_grad():
        for raw in loader:
            batch = _move_batch(raw, device)
            output = model(batch)
            weighted, plain, valid = residual_loss(
                torch,
                output,
                batch,
                args.material_threshold,
                args.material_weight,
            )
            mask = valid.expand_as(output)
            weighted_losses.append(float(weighted.item()))
            plain_losses.append(float(plain.item()))
            predicted.append(output[mask].cpu().numpy())
            observed.append(batch["target"][mask].cpu().numpy())
    return {
        "weighted_mse": float(np.mean(weighted_losses)),
        "mse": float(np.mean(plain_losses)),
        "pearson": _safe_pearson(np.concatenate(predicted), np.concatenate(observed)),
    }


def _load_checkpoint(torch, path, experiment_fingerprint, fold, seed, model_config, device):
    try:
        checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(path), map_location=device)
    if (
        checkpoint.get("schema") != SCHEMA
        or checkpoint.get("experiment_fingerprint") != experiment_fingerprint
        or checkpoint.get("fold") != int(fold)
        or checkpoint.get("seed") != int(seed)
        or checkpoint.get("model_config") != model_config
    ):
        raise ValueError("Stage-17A checkpoint contract mismatch: {}".format(path))
    return checkpoint


def train_fold_seed(
    torch,
    nn,
    prepared_dir,
    fit_entries,
    tune_entries,
    scalar_mean,
    scalar_scale,
    model_config,
    args,
    fold,
    seed,
    experiment_fingerprint,
    output,
):
    device = torch.device(args.device)
    checkpoint_path = (
        Path(output) / "checkpoints" / "fold{}_seed{}.pt".format(fold, seed)
    )
    if checkpoint_path.is_file():
        checkpoint = _load_checkpoint(
            torch,
            checkpoint_path,
            experiment_fingerprint,
            fold,
            seed,
            model_config,
            device,
        )
        model = build_advantage_model(torch, nn, model_config).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        print(
            "[fold {}/{} seed {}] resumed epoch {}".format(
                fold + 1, args.outer_folds, seed, checkpoint["best_epoch"]
            ),
            flush=True,
        )
        return model, checkpoint

    _set_seed(torch, seed + 1000 * fold)
    model = build_advantage_model(torch, nn, model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    fit_dataset = PreparedClipDataset(prepared_dir, fit_entries)
    tune_dataset = PreparedClipDataset(prepared_dir, tune_entries)
    tune_loader = _loader(
        torch,
        tune_dataset,
        args.batch_size,
        False,
        seed,
        scalar_mean,
        scalar_scale,
    )
    best_state = None
    best_epoch = 0
    best_score = float("inf")
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
        train_losses = []
        for raw in loader:
            batch = _move_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            weighted, plain, _ = residual_loss(
                torch,
                prediction,
                batch,
                args.material_threshold,
                args.material_weight,
            )
            weighted.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append((float(weighted.item()), float(plain.item())))
        tune = evaluate_model(torch, model, tune_loader, device, args)
        train_mean = np.asarray(train_losses, dtype=np.float64).mean(axis=0)
        history.append(
            {
                "fold": fold,
                "seed": seed,
                "epoch": epoch,
                "train_weighted_mse": train_mean[0],
                "train_mse": train_mean[1],
                "tune_weighted_mse": tune["weighted_mse"],
                "tune_mse": tune["mse"],
                "tune_pearson": tune["pearson"],
            }
        )
        print(
            "[fold {}/{} seed {} epoch {}/{}] train={:.7f} tune={:.7f} r={:.4f}".format(
                fold + 1,
                args.outer_folds,
                seed,
                epoch,
                args.epochs,
                train_mean[0],
                tune["weighted_mse"],
                tune["pearson"],
            ),
            flush=True,
        )
        if tune["weighted_mse"] < best_score - 1e-7:
            best_score = tune["weighted_mse"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Stage-17A training did not produce a checkpoint")
    checkpoint = {
        "schema": SCHEMA,
        "experiment_fingerprint": experiment_fingerprint,
        "fold": int(fold),
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_tune_weighted_mse": float(best_score),
        "model_config": model_config,
        "scalar_mean": scalar_mean,
        "scalar_scale": scalar_scale,
        "fit_sequences": [entry["sequence"] for entry in fit_entries],
        "tune_sequences": [entry["sequence"] for entry in tune_entries],
        "state_dict": best_state,
        "history": history,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(checkpoint_path))
    model.load_state_dict(best_state)
    model.eval()
    return model, checkpoint


def predict_ensemble(
    torch, models, dataset, scalar_mean, scalar_scale, args, fold
):
    loader = _loader(
        torch,
        dataset,
        args.batch_size,
        False,
        args.seed + fold,
        scalar_mean,
        scalar_scale,
    )
    device = torch.device(args.device)
    records = []
    offset = 0
    for raw in loader:
        batch = _move_batch(raw, device)
        with torch.no_grad():
            predictions = torch.stack([model(batch) for model in models], dim=0)
        predictions = predictions.cpu().numpy()
        episodes = int(raw["episodes"])
        clip_count = int(raw["clip_count"])
        for clip in range(clip_count):
            item = dataset[offset + clip]
            steps = item["action_reward"].shape[1]
            start = clip * episodes
            stop = start + episodes
            values = predictions[:, start:stop, :steps]
            action_reward = np.asarray(item["action_reward"], dtype=np.float32)
            target = action_reward[..., 1:] - action_reward[..., :1]
            records.append(
                {
                    "sequence": item["sequence"],
                    "fold": int(fold),
                    "mean": values.mean(axis=0),
                    "std": values.std(axis=0),
                    "target": target,
                    "valid": np.asarray(item["valid"], dtype=bool),
                    "behavior_ids": tuple(args.behavior_ids),
                }
            )
        offset += clip_count
    if offset != len(dataset):
        raise RuntimeError("Stage-17A prediction ordering changed")
    return records


def _atomic_savez(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=str(path.parent), suffix=".tmp", delete=False
    ) as handle:
        temporary = handle.name
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, str(path))


def save_oof_records(records, output, experiment_fingerprint):
    for record in records:
        path = Path(output) / "oof" / "{}__stage17_oof.npz".format(
            record["sequence"]
        )
        metadata = {
            "schema": OOF_SCHEMA,
            "experiment_fingerprint": experiment_fingerprint,
            "sequence": record["sequence"],
            "fold": record["fold"],
        }
        _atomic_savez(
            path,
            metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)),
            predicted_mean=np.asarray(record["mean"], dtype=np.float32),
            predicted_std=np.asarray(record["std"], dtype=np.float32),
            target=np.asarray(record["target"], dtype=np.float32),
            valid=np.asarray(record["valid"], dtype=bool),
        )


def oof_diagnostics(records, material_threshold):
    predicted = []
    observed = []
    centered_predicted = []
    centered_observed = []
    best_hits = 0
    states = 0
    regret = []
    gate_scores = []
    any_positive = []
    any_material = []
    uncertainty = []
    for record in records:
        valid = record["valid"]
        left = record["mean"][valid]
        right = record["target"][valid]
        std = record["std"][valid]
        predicted.append(left.reshape(-1))
        observed.append(right.reshape(-1))
        centered_predicted.append((left - left.mean(axis=1, keepdims=True)).reshape(-1))
        centered_observed.append((right - right.mean(axis=1, keepdims=True)).reshape(-1))
        chosen = np.argmax(left, axis=1)
        oracle = np.argmax(right, axis=1)
        best_hits += int(np.sum(chosen == oracle))
        states += len(chosen)
        regret.append(right.max(axis=1) - right[np.arange(len(right)), chosen])
        gate_scores.append(left.max(axis=1))
        any_positive.append(right.max(axis=1) > 0.0)
        any_material.append(right.max(axis=1) >= float(material_threshold))
        uncertainty.append(std.mean(axis=1))
    predicted = np.concatenate(predicted)
    observed = np.concatenate(observed)
    gate_scores = np.concatenate(gate_scores)
    return {
        "num_sequences": len(records),
        "num_valid_states": int(states),
        "num_action_labels": int(len(observed)),
        "advantage_pearson": _safe_pearson(predicted, observed),
        "within_state_pearson": _safe_pearson(
            np.concatenate(centered_predicted), np.concatenate(centered_observed)
        ),
        "any_positive_auroc": _safe_auroc(
            np.concatenate(any_positive), gate_scores
        ),
        "any_material_positive_auroc": _safe_auroc(
            np.concatenate(any_material), gate_scores
        ),
        "best_layer_hit_rate": best_hits / states,
        "mean_candidate_regret": float(np.concatenate(regret).mean()),
        "mean_ensemble_std": float(np.concatenate(uncertainty).mean()),
    }


def rounded_budget(num_steps, budget_percent):
    return int(np.floor(int(num_steps) * float(budget_percent) / 100.0 + 0.5))


def safe_cap_actions(scores, budget_percent, minimum_advantage=0.0):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    selected = np.zeros(len(scores), dtype=bool)
    cap = rounded_budget(len(scores), budget_percent)
    eligible = np.flatnonzero(np.isfinite(scores) & (scores > float(minimum_advantage)))
    selected[eligible[:cap]] = True
    return selected


def _uniform_positions(num_steps, count):
    num_steps, count = int(num_steps), int(count)
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if count > num_steps:
        raise ValueError("Uniform schedule count exceeds trajectory length")
    positions = np.floor((np.arange(count) + 0.5) * num_steps / count).astype(np.int64)
    if len(np.unique(positions)) != count:
        raise RuntimeError("Uniform position construction produced duplicates")
    return positions


def _cluster_bootstrap(values, samples, seed):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("Cluster bootstrap requires one value per sequence")
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(values), size=(int(samples), len(values)))
    means = values[draws].mean(axis=1)
    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
        float(np.mean(means > 0.0)),
    )


def _behavior_indices(behavior_ids, budget_percent):
    token = ("{:g}".format(float(budget_percent))).replace(".", "p")
    prefix = "p{}_".format(token)
    indices = [
        index for index, behavior_id in enumerate(behavior_ids)
        if behavior_id.startswith(prefix)
    ]
    if not indices:
        raise ValueError(
            "No frozen behavior episode matches budget {}".format(budget_percent)
        )
    return indices


def budget_diagnostic(
    records,
    budget_percent,
    uncertainty_z,
    minimum_advantage,
    bootstrap_samples,
    seed,
):
    sequence_gain = []
    sequence_random_lift = []
    sequence_uniform_l9_lift = []
    sequence_selected_l9_lift = []
    action_counts = np.zeros(len(CANDIDATE_LAYERS), dtype=np.int64)
    selected_total = 0
    decision_total = 0
    cap_total = 0
    selected_valid = 0
    selected_positive = 0
    selected_harmful = 0
    l9_index = CANDIDATE_LAYERS.index(9)

    for record in records:
        per_episode_gain = []
        per_episode_random = []
        per_episode_uniform = []
        per_episode_selected_l9 = []
        episode_indices = _behavior_indices(
            record["behavior_ids"], budget_percent
        )
        for episode in episode_indices:
            mean = record["mean"][episode]
            std = record["std"][episode]
            target = record["target"][episode]
            valid = record["valid"][episode]
            lower = mean - float(uncertainty_z) * std
            candidate = np.argmax(lower, axis=1)
            score = lower[np.arange(len(lower)), candidate]
            selected = safe_cap_actions(score, budget_percent, minimum_advantage)
            count = int(selected.sum())
            cap = rounded_budget(len(score), budget_percent)
            chosen = np.zeros(len(score), dtype=np.float64)
            chosen[selected] = target[np.flatnonzero(selected), candidate[selected]]

            uniform = np.zeros(len(score), dtype=np.float64)
            positions = _uniform_positions(len(score), count)
            uniform[positions] = target[positions, l9_index]

            selected_l9 = np.zeros(len(score), dtype=np.float64)
            selected_l9[selected] = target[selected, l9_index]

            valid_count = max(1, int(valid.sum()))
            gain = float(chosen[valid].sum() / valid_count)
            uniform_gain = float(uniform[valid].sum() / valid_count)
            selected_l9_gain = float(selected_l9[valid].sum() / valid_count)
            random_gain = float(
                (count / max(1, len(score))) * target[valid].mean()
            )
            per_episode_gain.append(gain)
            per_episode_random.append(gain - random_gain)
            per_episode_uniform.append(gain - uniform_gain)
            per_episode_selected_l9.append(gain - selected_l9_gain)

            action_counts += np.bincount(
                candidate[selected], minlength=len(CANDIDATE_LAYERS)
            )
            selected_total += count
            decision_total += len(score)
            cap_total += cap
            valid_selected = selected & valid
            selected_values = target[
                np.flatnonzero(valid_selected), candidate[valid_selected]
            ]
            selected_valid += len(selected_values)
            selected_positive += int(np.sum(selected_values > 0.0))
            selected_harmful += int(np.sum(selected_values < 0.0))
        sequence_gain.append(float(np.mean(per_episode_gain)))
        sequence_random_lift.append(float(np.mean(per_episode_random)))
        sequence_uniform_l9_lift.append(float(np.mean(per_episode_uniform)))
        sequence_selected_l9_lift.append(float(np.mean(per_episode_selected_l9)))

    gain_ci = _cluster_bootstrap(sequence_gain, bootstrap_samples, seed)
    random_ci = _cluster_bootstrap(sequence_random_lift, bootstrap_samples, seed + 1)
    uniform_ci = _cluster_bootstrap(
        sequence_uniform_l9_lift, bootstrap_samples, seed + 2
    )
    selected_l9_ci = _cluster_bootstrap(
        sequence_selected_l9_lift, bootstrap_samples, seed + 3
    )
    active_threshold = max(1, int(math.ceil(0.05 * max(1, selected_total))))
    return {
        "budget_percent": float(budget_percent),
        "uncertainty_z": float(uncertainty_z),
        "minimum_advantage": float(minimum_advantage),
        "num_sequences": len(records),
        "candidate_blocks": int(selected_total),
        "candidate_cap_blocks": int(cap_total),
        "candidate_ratio": selected_total / decision_total,
        "cap_ratio": cap_total / decision_total,
        "budget_utilization": selected_total / cap_total if cap_total else 0.0,
        "gain_vs_fast": float(np.mean(sequence_gain)),
        "gain_vs_fast_ci_low": gain_ci[0],
        "gain_vs_fast_ci_high": gain_ci[1],
        "gain_vs_fast_probability_positive": gain_ci[2],
        "lift_vs_random_time_and_layer": float(np.mean(sequence_random_lift)),
        "random_lift_ci_low": random_ci[0],
        "random_lift_ci_high": random_ci[1],
        "random_lift_probability_positive": random_ci[2],
        "lift_vs_uniform_l9": float(np.mean(sequence_uniform_l9_lift)),
        "uniform_l9_lift_ci_low": uniform_ci[0],
        "uniform_l9_lift_ci_high": uniform_ci[1],
        "uniform_l9_lift_probability_positive": uniform_ci[2],
        "layer_lift_vs_l9_at_selected_states": float(
            np.mean(sequence_selected_l9_lift)
        ),
        "selected_l9_lift_ci_low": selected_l9_ci[0],
        "selected_l9_lift_ci_high": selected_l9_ci[1],
        "selected_l9_lift_probability_positive": selected_l9_ci[2],
        "selected_positive_ratio": (
            selected_positive / selected_valid if selected_valid else float("nan")
        ),
        "selected_harmful_ratio": (
            selected_harmful / selected_valid if selected_valid else float("nan")
        ),
        "active_candidate_layers": int(np.sum(action_counts >= active_threshold)),
        "action_distribution": "|".join(
            "l{}:{}".format(layer, action_counts[index])
            for index, layer in enumerate(CANDIDATE_LAYERS)
        ),
    }


def build_experiment_contract(prepared, run, frozen, folds, args):
    contract = {
        "schema": SCHEMA,
        "analysis_role": "train_only_group_oof_residual_advantage_screen",
        "source_run_fingerprint": run["fingerprint"],
        "source_freeze_fingerprint": frozen["fingerprint"],
        "prepared_fingerprint": prepared["contract"]["fingerprint"],
        "prepared_pca_sha256": file_hash(prepared["pca_path"]),
        "label_scope": "stage16_train_only",
        "validation_labels_permitted": False,
        "train_sequences": [
            entry["sequence"] for entry in prepared["train_entries"]
        ],
        "excluded_validation_sequences": [
            entry["sequence"] for entry in prepared["validation_entries"]
        ],
        "outer_folds": [
            {
                "fold": fold,
                "train_sequences": [entry["sequence"] for entry in train],
                "test_sequences": [entry["sequence"] for entry in test],
            }
            for fold, train, test in folds
        ],
        "seeds": list(args.seeds),
        "seed": int(args.seed),
        "tune_fraction": float(args.tune_fraction),
        "model": {
            "token_dim": prepared["token_dim"],
            "scalar_dim": len(RESIDUAL_SCALAR_NAMES),
            "scalar_names": list(RESIDUAL_SCALAR_NAMES),
            "hidden_dim": args.hidden_dim,
            "spatial_channels": args.spatial_channels,
            "num_actions": len(CANDIDATE_LAYERS),
        },
        "training": {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "material_threshold": args.material_threshold,
            "material_weight": args.material_weight,
            "grad_clip": args.grad_clip,
        },
        "policy": {
            "uncertainty_z": args.uncertainty_z,
            "minimum_advantage": args.minimum_advantage,
            "budgets": list(args.budgets),
            "primary_budget": args.primary_budget,
            "budget_semantics": "chronological_hard_cap_without_forced_spending",
        },
        "behavior_ids": [item["behavior_id"] for item in run["behaviors"]],
        "screen": {
            "minimum_advantage_pearson": 0.10,
            "minimum_within_state_pearson": 0.10,
            "minimum_candidate_ratio": 0.02,
            "gain_vs_fast_ci_low_must_exceed_zero": True,
            "lift_vs_uniform_l9_ci_low_must_exceed_zero": True,
            "minimum_active_candidate_layers": 3,
        },
        "bootstrap_samples": args.bootstrap_samples,
        "code_sha256": {
            "stage17_residual_advantage.py": file_hash(Path(__file__).resolve())
        },
    }
    contract["fingerprint"] = fingerprint(contract)
    return contract


def publish_contract(output, contract):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "stage17_experiment.json"
    if path.is_file():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != contract:
            raise ValueError(
                "Stage-17A output belongs to a different experiment; use a new output_dir"
            )
    else:
        unexpected = list(output.iterdir())
        if unexpected:
            raise ValueError("Unidentified Stage-17A output files; use a new output_dir")
        write_json(path, contract)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Frozen Stage-16 behavior cache")
    parser.add_argument("--freeze_manifest", required=True)
    parser.add_argument("--prepared_dir", required=True, help="Stage-16B prepared directory")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--outer_folds", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2051, 2052])
    parser.add_argument("--seed", type=int, default=2051)
    parser.add_argument("--tune_fraction", type=float, default=0.2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--spatial_channels", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--material_threshold", type=float, default=0.01)
    parser.add_argument("--material_weight", type=float, default=3.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--uncertainty_z", type=float, default=1.0)
    parser.add_argument("--minimum_advantage", type=float, default=0.0)
    parser.add_argument("--budgets", nargs="+", type=float, default=[10.0, 30.0, 50.0])
    parser.add_argument("--primary_budget", type=float, default=30.0)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--skip_cache_hash_verification", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
    except ImportError as error:
        raise RuntimeError("Stage-17A requires PyTorch") from error
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if len(set(args.seeds)) < 2:
        raise ValueError("Stage-17A requires at least two ensemble seeds")
    if len(set(args.budgets)) != len(args.budgets):
        raise ValueError("Stage-17A budgets must be unique")
    if args.primary_budget not in args.budgets:
        raise ValueError("primary_budget must be included in budgets")
    if (
        args.outer_folds < 2
        or args.epochs < 1
        or args.patience < 1
        or args.bootstrap_samples < 1
        or not 0.0 < args.tune_fraction < 0.5
        or args.material_threshold < 0.0
        or args.material_weight < 0.0
        or args.uncertainty_z < 0.0
        or args.minimum_advantage < 0.0
        or args.hidden_dim < 1
        or args.spatial_channels < 1
        or args.batch_size < 1
        or args.learning_rate <= 0.0
        or args.weight_decay < 0.0
        or args.grad_clip <= 0.0
        or any(not 0.0 <= budget <= 100.0 for budget in args.budgets)
    ):
        raise ValueError("Invalid Stage-17A configuration")

    directory, run, frozen = verify_frozen_cache(
        args.freeze_manifest,
        input_dir=args.input,
        verify_hashes=not args.skip_cache_hash_verification,
    )
    prepared = audit_prepared_training_data(args.prepared_dir, run, frozen)
    args.behavior_ids = [item["behavior_id"] for item in run["behaviors"]]
    for budget in args.budgets:
        _behavior_indices(args.behavior_ids, budget)
    folds = grouped_outer_folds(
        prepared["train_entries"], args.outer_folds, args.seed
    )
    contract = build_experiment_contract(prepared, run, frozen, folds, args)
    output = Path(args.output_dir).expanduser().resolve()
    publish_contract(output, contract)
    decision_path = output / "stage17_decision.json"
    if decision_path.is_file():
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        digest = decision.get("fingerprint")
        body = {key: value for key, value in decision.items() if key != "fingerprint"}
        if (
            decision.get("experiment_fingerprint") != contract["fingerprint"]
            or digest is None
            or fingerprint(body) != digest
        ):
            raise ValueError("Existing Stage-17A decision does not match the contract")
        print("Stage-17A already complete: {}".format(decision["status"]))
        print("Decision JSON: {}".format(decision_path))
        return

    print(
        "Frozen train-only Stage-17A screen: {} train videos; {} validation videos excluded.".format(
            len(prepared["train_entries"]), len(prepared["validation_entries"])
        )
    )
    print(
        "Target: six dense candidate-minus-Fast immediate advantages; {}-fold grouped OOF.".format(
            args.outer_folds
        )
    )

    model_config = dict(contract["model"])
    all_records = []
    training_rows = []
    checkpoint_paths = []
    for fold, outer_train, outer_test in folds:
        fit_entries, tune_entries = grouped_fit_tune(
            outer_train, args.tune_fraction, args.seed + 100 * fold
        )
        scalar_mean, scalar_scale = scalar_normalization(
            prepared["directory"], outer_train
        )
        models = []
        for seed in args.seeds:
            model, checkpoint = train_fold_seed(
                torch,
                nn,
                prepared["directory"],
                fit_entries,
                tune_entries,
                scalar_mean,
                scalar_scale,
                model_config,
                args,
                fold,
                seed,
                contract["fingerprint"],
                output,
            )
            models.append(model)
            checkpoint_path = (
                output / "checkpoints" / "fold{}_seed{}.pt".format(fold, seed)
            )
            checkpoint_paths.append(checkpoint_path)
            training_rows.extend(checkpoint["history"])
        test_dataset = PreparedClipDataset(prepared["directory"], outer_test)
        fold_records = predict_ensemble(
            torch,
            models,
            test_dataset,
            scalar_mean,
            scalar_scale,
            args,
            fold,
        )
        all_records.extend(fold_records)
        print(
            "[outer {}/{}] fit={} tune={} test={} complete".format(
                fold + 1,
                args.outer_folds,
                len(fit_entries),
                len(tune_entries),
                len(outer_test),
            ),
            flush=True,
        )

    expected_names = {entry["sequence"] for entry in prepared["train_entries"]}
    observed_names = [record["sequence"] for record in all_records]
    if len(observed_names) != len(set(observed_names)) or set(observed_names) != expected_names:
        raise RuntimeError("Stage-17A OOF predictions are missing, duplicated, or contaminated")
    all_records.sort(key=lambda record: record["sequence"])
    save_oof_records(all_records, output, contract["fingerprint"])

    diagnostics = oof_diagnostics(all_records, args.material_threshold)
    diagnostic_rows = [{"scope": "train_group_oof", **diagnostics}]
    budget_rows = []
    for index, budget in enumerate(args.budgets):
        budget_rows.append(
            budget_diagnostic(
                all_records,
                budget,
                args.uncertainty_z,
                args.minimum_advantage,
                args.bootstrap_samples,
                args.seed + 1000 + 10 * index,
            )
        )
    primary = next(
        row
        for row in budget_rows
        if abs(row["budget_percent"] - args.primary_budget) <= 1e-12
    )
    screen = {
        "minimum_advantage_pearson": 0.10,
        "observed_advantage_pearson": diagnostics["advantage_pearson"],
        "advantage_prediction_pass": diagnostics["advantage_pearson"] >= 0.10,
        "minimum_within_state_pearson": 0.10,
        "observed_within_state_pearson": diagnostics["within_state_pearson"],
        "within_state_prediction_pass": diagnostics["within_state_pearson"] >= 0.10,
        "primary_budget": args.primary_budget,
        "minimum_candidate_ratio": 0.02,
        "observed_candidate_ratio": primary["candidate_ratio"],
        "nontrivial_usage_pass": primary["candidate_ratio"] >= 0.02,
        "hard_cap_pass": primary["candidate_blocks"] <= primary["candidate_cap_blocks"],
        "primary_gain_vs_fast": primary["gain_vs_fast"],
        "primary_gain_vs_fast_ci_low": primary["gain_vs_fast_ci_low"],
        "gain_vs_fast_pass": primary["gain_vs_fast_ci_low"] > 0.0,
        "primary_lift_vs_uniform_l9": primary["lift_vs_uniform_l9"],
        "primary_lift_vs_uniform_l9_ci_low": primary["uniform_l9_lift_ci_low"],
        "uniform_l9_lift_pass": primary["uniform_l9_lift_ci_low"] > 0.0,
        "minimum_active_candidate_layers": 3,
        "observed_active_candidate_layers": primary["active_candidate_layers"],
        "noncollapse_pass": primary["active_candidate_layers"] >= 3,
    }
    passed = all(
        screen[key]
        for key in (
            "advantage_prediction_pass",
            "within_state_prediction_pass",
            "nontrivial_usage_pass",
            "hard_cap_pass",
            "gain_vs_fast_pass",
            "uniform_l9_lift_pass",
            "noncollapse_pass",
        )
    )
    decision = {
        "schema": SCHEMA,
        "status": (
            "PASS_TRAIN_OOF_RESIDUAL_SCREEN"
            if passed
            else "FAIL_TRAIN_OOF_RESIDUAL_SCREEN"
        ),
        "analysis_role": "development_only_train_group_oof_screen",
        "experiment_fingerprint": contract["fingerprint"],
        "source_run_fingerprint": run["fingerprint"],
        "source_freeze_fingerprint": frozen["fingerprint"],
        "prepared_fingerprint": prepared["contract"]["fingerprint"],
        "train_sequences": len(prepared["train_entries"]),
        "excluded_validation_sequences": len(prepared["validation_entries"]),
        "validation_labels_loaded": False,
        "screen": screen,
        "next_step": (
            "Freeze the Stage-17 policy specification and acquire a genuinely untouched "
            "evaluation dataset before any closed-loop claim. Do not evaluate or tune on "
            "the consumed Stage-16 validation split."
            if passed
            else "Stop this residual-advantage state representation; it did not clear the "
            "train-only grouped-OOF gate against same-cost Uniform-L9."
        ),
        "warning": (
            "This is a behavior-support immediate counterfactual diagnostic, not a "
            "closed-loop policy result. OTB, UAV123, and the Stage-16 validation split "
            "remain unavailable for policy selection."
        ),
        "artifacts": {
            "checkpoints": [
                {"path": str(path), "sha256": file_hash(path)}
                for path in checkpoint_paths
            ],
            "oof_directory": str(output / "oof"),
        },
        "code_sha256": {
            "stage17_residual_advantage.py": file_hash(Path(__file__).resolve())
        },
    }
    decision["fingerprint"] = fingerprint(decision)

    write_csv(output / "stage17_training_history.csv", training_rows)
    write_csv(output / "stage17_oof_value_summary.csv", diagnostic_rows)
    write_csv(output / "stage17_oof_budget_summary.csv", budget_rows)
    write_json(decision_path, decision)

    print("\nTrain-only grouped-OOF residual diagnostics:")
    print(
        "advantage_pearson={:.4f}  within_state={:.4f}  AUROC+={:.4f}  hit={:.2%}".format(
            diagnostics["advantage_pearson"],
            diagnostics["within_state_pearson"],
            diagnostics["any_positive_auroc"],
            diagnostics["best_layer_hit_rate"],
        )
    )
    print("\nSafe residual policy support-state diagnostic:")
    print("budget  actual  utilization  gain_fast  lift_uniform_l9  paired_95%           layers")
    for row in budget_rows:
        print(
            "{:>6.1f}% {:>6.2f}% {:>10.2f}% {:>10.6f} {:>16.6f} [{:.6f},{:.6f}] {:>7d}".format(
                row["budget_percent"],
                100.0 * row["candidate_ratio"],
                100.0 * row["budget_utilization"],
                row["gain_vs_fast"],
                row["lift_vs_uniform_l9"],
                row["uniform_l9_lift_ci_low"],
                row["uniform_l9_lift_ci_high"],
                row["active_candidate_layers"],
            )
        )
    print("\nPredeclared Stage-17A decision: {}".format(decision["status"]))
    print("Decision JSON: {}".format(decision_path))
    print(
        "Validation labels loaded: no ({} sequences explicitly excluded).".format(
            len(prepared["validation_entries"])
        )
    )


if __name__ == "__main__":
    main()
