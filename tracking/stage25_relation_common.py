"""Causal relational state and equal-cost layer probes; no tracker imports."""

from pathlib import Path

import numpy as np

from stage4_common import fingerprint
from stage18_environment import TokenCredit
from stage23_development import read_sealed, sealed


SCHEMA = "stage25_relational_state_probe_v1"
HORIZON, REPEATS, ANCHORS = 20, 2, 8
GAMMA, COLLECTION_SEED = .97, 2080
FOLDS, FOLD_SEED = 5, 2081
MODEL_SEEDS = (2082, 2083, 2084)
BOOTSTRAP_SEED, BOOTSTRAPS = 2085, 5000
RELATION_DIM = 280
CODE_FILES = ("stage25_relation_common.py", "stage25_relation_collect.py", "stage25_relation_screen.py")


def keyed_seed(*values):
    return int(fingerprint(list(values))[:16], 16)


def schedule(steps, seed):
    """Canonical P30 call positions, independently shuffled balanced layers."""
    if type(steps) is not int or steps < HORIZON:
        raise ValueError("Need at least H20 tracked frames")
    positions = np.flatnonzero(np.diff(np.r_[0, 3 * np.arange(1, steps + 1) // 10]))
    rng = np.random.default_rng(seed)
    layers = np.tile(np.arange(1, 7), (len(positions) + 5) // 6)[:len(positions)]
    rng.shuffle(layers)
    actions = np.zeros(steps, np.int64)
    actions[positions] = layers
    return actions


def anchor_indices(actions):
    actions = np.asarray(actions)
    available = np.flatnonzero((actions > 0) & (np.arange(len(actions)) + HORIZON <= len(actions)))
    if len(available) < ANCHORS:
        raise ValueError("Clip cannot support eight label-independent H20 anchors")
    return available[np.linspace(0, len(available) - 1, ANCHORS).astype(int)]


def continuation(actions, anchor, seed):
    """Repeat zero replays behavior; repeat one independently redraws later layers."""
    first = np.asarray(actions[anchor:anchor + HORIZON], np.int64).copy()
    if first.shape != (HORIZON,) or not first[0]:
        raise ValueError("Anchor must be a candidate slot with a complete horizon")
    second = first.copy()
    positions = np.flatnonzero(second[1:] > 0) + 1
    second[positions] = np.random.default_rng(seed).integers(1, 7, len(positions))
    return np.stack((first, second))


def raw_dimensions(embed_dim):
    return 65 * embed_dim + 256 + 14 + 4


def unpack_raw(raw, embed_dim):
    raw = np.asarray(raw, np.float32)
    if raw.shape != (raw_dimensions(embed_dim),) or not np.isfinite(raw).all():
        raise ValueError("Invalid compact pre-action state")
    return raw[:64 * embed_dim].reshape(embed_dim, 8, 8), raw[64 * embed_dim:65 * embed_dim], raw[-4:]


def relation_features(grid, template, roi):
    """Explicit relations derived from the SAME grid/template supplied to control.

    ROI is the previous predicted box in the current crop, never current GT.
    The three outside-ROI peaks are hypotheses, not annotated distractors.
    """
    grid, template, roi = (np.asarray(x, np.float64) for x in (grid, template, roi))
    if (grid.ndim != 3 or grid.shape[1:] != (8, 8) or template.shape != (grid.shape[0],)
            or roi.shape != (4,) or not all(np.isfinite(x).all() for x in (grid, template, roi))
            or np.any(roi[2:] <= 0)):
        raise ValueError("Expected finite Cx8x8 grid, template and positive xywh ROI")
    tokens = grid.reshape(grid.shape[0], -1).T
    unit = lambda x: x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)
    yy, xx = np.meshgrid((np.arange(8) + .5) / 8, (np.arange(8) + .5) / 8, indexing="ij")
    prior = ((xx >= roi[0]) & (xx <= roi[0] + roi[2]) &
             (yy >= roi[1]) & (yy <= roi[1] + roi[3])).ravel()
    if not prior.any():
        center = roi[:2] + roi[2:] / 2
        prior[np.argmin((xx.ravel() - center[0]) ** 2 + (yy.ravel() - center[1]) ** 2)] = True
    target = tokens[prior].mean(0)
    template_map = unit(tokens) @ unit(template)
    target_map = unit(tokens) @ unit(target)
    excluded = prior.copy()
    distractors = []
    for _ in range(3):
        scores = np.where(excluded, -np.inf, template_map)
        if not np.isfinite(scores).any():
            # A crop-filling prior has no outside peak: retain a finite null relation.
            distractors.append(np.zeros_like(target))
            continue
        position = int(scores.argmax())
        distractors.append(tokens[position])
        y, x = divmod(position, 8)
        excluded |= ((np.abs(xx * 8 - .5 - x) <= 1) & (np.abs(yy * 8 - .5 - y) <= 1)).ravel()
    distractor_map = (unit(tokens) @ unit(np.stack(distractors)).T).max(1)
    maps = (template_map, target_map, distractor_map, target_map - distractor_map)
    stats = []
    for values in maps:
        ordered = np.sort(values)
        p = np.exp(10 * (values - values.max()))
        p /= p.sum()
        stats.extend((values.mean(), values.std(), values.max(), ordered[-1] - ordered[-2],
                      -np.sum(p * np.log(np.maximum(p, 1e-15))) / np.log(64)))
    stats.extend([float(unit(target) @ unit(template))] +
                 [float(unit(target) @ unit(d)) for d in distractors])
    result = np.r_[np.concatenate(maps), stats].astype(np.float32)
    if result.shape != (RELATION_DIM,) or not np.isfinite(result).all():
        raise RuntimeError("Invalid relational encoding")
    return result


def discounted_values(rewards, valid, post_anchor=False):
    rewards, valid = np.asarray(rewards, float), np.asarray(valid)
    if (rewards.ndim != 4 or rewards.shape[1:] != (REPEATS, 6, HORIZON)
            or valid.shape != (len(rewards), HORIZON) or valid.dtype != np.bool_
            or not np.isfinite(rewards).all() or np.any((rewards < 0) | (rewards > 1))
            or np.any(rewards * ~valid[:, None, None, :] != 0)):
        raise ValueError("Invalid H20 rewards/visibility")
    weights = GAMMA ** np.arange(HORIZON)
    if post_anchor:
        weights[0] = 0
    denominator = valid @ weights
    q = (rewards * weights).sum(-1) / np.maximum(denominator[:, None, None], 1e-12)
    eligible = valid[:, 0] & (valid.mean(1) >= .8)
    return q, eligible


def validate_cache(data, contract, sequence):
    dim = contract["embed_dim"]
    steps = len(data["live_actions"])
    if steps != contract["clip_steps"][sequence] or not 64 <= steps <= 128:
        raise ValueError("Cache clip bounds differ from frozen manifest")
    shapes = dict(anchor=(ANCHORS,), raw=(ANCHORS, raw_dimensions(dim)),
                  relation=(ANCHORS, RELATION_DIM), rewards=(ANCHORS, REPEATS, 6, HORIZON),
                  valid=(ANCHORS, HORIZON), live_actions=(steps,), live_boxes=(steps, 4),
                  live_valid=(steps,), replay_error=(ANCHORS,), replay_boxes=(ANCHORS, HORIZON, 4),
                  branch_cost=(ANCHORS, REPEATS, 6))
    if set(data) != set(shapes):
        raise ValueError("Unexpected or missing cache fields")
    for name, shape in shapes.items():
        if data[name].shape != shape or not np.isfinite(data[name]).all():
            raise ValueError("Bad cache field: " + name)
    if data["valid"].dtype != np.bool_ or data["live_valid"].dtype != np.bool_:
        raise ValueError("Visibility must be boolean")
    for name in ("anchor", "live_actions", "branch_cost"):
        if data[name].dtype.kind not in "iu":
            raise ValueError("Integer cache field required: " + name)
    expected = schedule(steps, keyed_seed(COLLECTION_SEED, sequence, "behavior"))
    if not np.array_equal(data["live_actions"], expected) or not np.array_equal(data["anchor"], anchor_indices(expected)):
        raise ValueError("Behavior schedule or label-independent anchors changed")
    budget = TokenCredit()
    for action in expected:
        budget.advance()
        budget.spend(int(action))
    if (np.any(data["live_boxes"][:, 2:] <= 0) or np.any(data["replay_error"] < 0)
            or data["replay_error"].max() > 1e-6):
        raise ValueError("Invalid boxes or chosen-action replay")
    if not np.allclose(data["rewards"][:, 0, :, 0], data["rewards"][:, 1, :, 0], rtol=0, atol=1e-12):
        raise ValueError("Continuation repeats disagree on the identical first intervention frame")
    for i, anchor in enumerate(data["anchor"]):
        grid, template, roi = unpack_raw(data["raw"][i], dim)
        if not np.allclose(data["relation"][i], relation_features(grid, template, roi), rtol=0, atol=1e-6):
            raise ValueError("Relation fields do not reconstruct from pre-action state")
        if not np.array_equal(data["valid"][i], data["live_valid"][anchor:anchor + HORIZON]):
            raise ValueError("Branch labels do not match live frame visibility")
        if not np.all(data["branch_cost"][i] == np.count_nonzero(expected[anchor:anchor + HORIZON])):
            raise ValueError("First-layer branches did not have exactly equal costs")
        replay_error = float(np.abs(data["replay_boxes"][i] - data["live_boxes"][anchor:anchor + HORIZON]).max())
        if replay_error > 1e-6 or not np.isclose(data["replay_error"][i], replay_error, rtol=0, atol=1e-12):
            raise ValueError("Chosen first layer/behavior continuation does not replay live H20 boxes")
        if not np.isclose(data["raw"][i, -4 - 14 + 5], (3 * (int(anchor) + 1) % 10 + 10) / 20):
            # Canonical slots arrive with credit 10, 11 or 12, before spending.
            raise ValueError("Pre-action credit differs from canonical schedule")
        previous = int(expected[anchor - 1]) if anchor else -1
        if not np.array_equal(data["raw"][i, -12:-4], np.eye(8)[previous + 1]):
            raise ValueError("History action differs from the actual previous executed action")
    discounted_values(data["rewards"], data["valid"])


def cache_path(directory, sequence):
    return Path(directory) / "fit" / (sequence + "__stage25_relation.npz")


def load_cache(path, contract, sequence):
    with np.load(path, allow_pickle=False) as archive:
        import json
        meta = json.loads(str(archive["metadata_json"].item()))
        data = {k: archive[k] for k in archive.files if k != "metadata_json"}
    if meta != dict(contract_fingerprint=contract["fingerprint"], sequence=sequence):
        raise ValueError("Cache lineage mismatch")
    validate_cache(data, contract, sequence)
    return data


def load_contract(directory):
    contract = read_sealed(Path(directory) / "stage25_contract.json")
    fit = contract.get("fit", [])
    excluded = contract.get("excluded_tune", []) + contract.get("excluded_validation", [])
    if (contract.get("schema") != SCHEMA or not fit or len(set(fit)) != len(fit)
            or any(not isinstance(n, str) or not n or n in (".", "..") or "/" in n or "\\" in n for n in fit)
            or set(fit) & set(excluded) or contract.get("horizon") != HORIZON
            or contract.get("repeats") != REPEATS or contract.get("anchors") != ANCHORS
            or contract.get("gamma") != GAMMA or contract.get("seed") != COLLECTION_SEED
            or set(contract.get("clip_steps", {})) != set(fit)):
        raise ValueError("Invalid fit-only Stage-25 contract")
    return contract


def build_probe(torch, embed_dim):
    """Same parameter shapes for raw and relational arms; raw zeros relation input."""
    nn = torch.nn

    class Probe(nn.Module):
        def __init__(self):
            super().__init__()
            self.spatial = nn.Conv2d(embed_dim, 16, 1)
            self.template = nn.Linear(embed_dim, 64)
            self.output = nn.Sequential(nn.Linear(1024 + 64 + 256 + 14 + 4 + RELATION_DIM, 128),
                                        nn.GELU(), nn.LayerNorm(128), nn.Linear(128, 6))

        def forward(self, raw, relation):
            grid = raw[:, :64 * embed_dim].reshape(-1, embed_dim, 8, 8)
            template = raw[:, 64 * embed_dim:65 * embed_dim]
            tail = raw[:, 65 * embed_dim:]
            prediction = self.output(torch.cat((self.spatial(grid).flatten(1),
                                               self.template(template), tail, relation), 1))
            return prediction - prediction.mean(1, keepdim=True)

    return Probe()
