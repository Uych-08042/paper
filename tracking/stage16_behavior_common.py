"""Dependency-light contracts for Stage-16 continuous behavior trajectories."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from stage4_common import IOU_PROTOCOL, file_hash, fingerprint, load_manifest, tracking_iou


SCHEMA = "stage16_continuous_behavior_v1"
ACTION_NAMES = ("fast", "l7", "l8", "l9", "l10", "l11", "l12")
CANDIDATE_LAYERS = tuple(range(7, 13))
DEFAULT_BEHAVIOR_RATES = (0.0, 10.0, 30.0, 50.0, 100.0)


def behavior_specs(rates=DEFAULT_BEHAVIOR_RATES, repeats=2):
    """Return the frozen trajectory list; Fast is collected once."""
    values = tuple(sorted(set(float(value) for value in rates)))
    repeats = int(repeats)
    if (
        not values
        or not np.isfinite(values).all()
        or values[0] != 0.0
        or values[-1] > 100.0
    ):
        raise ValueError("Behavior rates must be unique percentages in [0,100] and include 0")
    if values[0] < 0.0 or repeats < 1:
        raise ValueError("Behavior rates must be non-negative and repeats must be positive")
    output = []
    for rate in values:
        count = 1 if rate == 0.0 else repeats
        token = ("{:g}".format(rate)).replace(".", "p")
        for repeat in range(count):
            output.append(
                {
                    "behavior_id": "p{}_r{}".format(token, repeat),
                    "candidate_rate_percent": rate,
                    "repeat": repeat,
                }
            )
    return output


def behavior_seed(seed, manifest_fingerprint, sequence, behavior_id):
    payload = "{}|{}|{}|{}".format(
        int(seed), manifest_fingerprint, sequence, behavior_id
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFFFFFFFFFF


def make_behavior_schedule(num_steps, candidate_rate_percent, seed):
    """Create an exact-cost randomized schedule with balanced candidate layers.

    Candidate positions are sampled without replacement. Candidate labels are a
    shuffled balanced multiset, so every layer differs in count by at most one.
    The stored propensity is the exact conditional probability given the
    actions already revealed in the without-replacement schedule.
    """
    num_steps = int(num_steps)
    rate = float(candidate_rate_percent)
    if num_steps < 1 or not 0.0 <= rate <= 100.0:
        raise ValueError("Invalid trajectory length or candidate rate")
    candidate_count = int(np.floor(num_steps * rate / 100.0 + 0.5))
    rng = np.random.default_rng(int(seed))
    actions = np.zeros(num_steps, dtype=np.int64)
    if candidate_count:
        positions = rng.choice(num_steps, size=candidate_count, replace=False)
        layer_order = np.arange(1, len(ACTION_NAMES), dtype=np.int64)
        rng.shuffle(layer_order)
        labels = np.resize(layer_order, candidate_count)
        rng.shuffle(labels)
        actions[positions] = labels

    counts = np.bincount(actions, minlength=len(ACTION_NAMES)).astype(np.int64)
    one_hot = np.eye(len(ACTION_NAMES), dtype=np.int64)[actions]
    used_by_action_before = np.vstack(
        (
            np.zeros((1, len(ACTION_NAMES)), dtype=np.int64),
            np.cumsum(one_hot[:-1], axis=0, dtype=np.int64),
        )
    )
    remaining_by_action = counts[None, :] - used_by_action_before
    frames_remaining = num_steps - np.arange(num_steps, dtype=np.int64)
    action_probability = remaining_by_action / frames_remaining[:, None]
    propensity = action_probability[np.arange(num_steps), actions]
    candidate = actions > 0
    used_before = np.r_[0, np.cumsum(candidate[:-1], dtype=np.int64)]
    remaining_before = candidate_count - used_before
    remaining_rate = remaining_before.astype(np.float64) / frames_remaining
    return {
        "action": actions,
        "action_cost": candidate.astype(np.int64),
        "action_propensity": propensity,
        "behavior_probability": action_probability,
        "candidate_used_before": used_before.astype(np.int64),
        "candidate_remaining_before": remaining_before.astype(np.int64),
        "remaining_candidate_rate": remaining_rate,
        "action_counts": counts,
    }


def behavior_cache_path(output_dir, entry):
    return (
        Path(output_dir)
        / entry["split"]
        / "{}__stage16_behavior.npz".format(entry["sequence"])
    )


def make_run(
    manifest,
    runtime_identity,
    state_grid,
    behavior_rates,
    behavior_repeats,
    seed,
    verify_actions,
    save_full_tokens=False,
):
    specs = behavior_specs(behavior_rates, behavior_repeats)
    state_grid = int(state_grid)
    if state_grid < 1:
        raise ValueError("state_grid must be positive")
    directory = Path(__file__).resolve().parent
    files = (
        "stage16_behavior_common.py",
        "stage16_behavior_collect.py",
        "stage4_common.py",
        "stage4_collect.py",
        "stage4_runtime.py",
        "validation_metrics.py",
        "stage1_validate.py",
    )
    state_fields = [
        "template_summary",
        "search_token_grid",
        "fast_score_map",
        "selector_prob",
        "selected_layer",
        "previous_state",
        "previous_fast_box",
        "previous_action",
        "candidate_used_before",
        "candidate_remaining_before",
        "remaining_candidate_rate",
        "image_hw",
        "resize_factor",
    ]
    if save_full_tokens:
        state_fields.append("full_tokens")
    run = {
        "schema": SCHEMA,
        "analysis_role": "development_only_randomized_continuous_behavior_data",
        "manifest": manifest,
        "runtime": runtime_identity,
        "state_grid": state_grid,
        "behaviors": specs,
        "behavior_seed": int(seed),
        "verify_actions": bool(verify_actions),
        "save_full_tokens": bool(save_full_tokens),
        "actions": list(ACTION_NAMES),
        "candidate_layers": list(CANDIDATE_LAYERS),
        "transition_semantics": "executed_action_box_is_written_to_next_tracker_state",
        "randomization": "exact_cost_positions_without_replacement_balanced_candidate_layers",
        "iou_protocol": IOU_PROTOCOL,
        "state_fields": state_fields,
        "behavior_fields": [
            "behavior_action",
            "action_cost",
            "action_propensity",
            "behavior_probability",
        ],
        "label_fields": [
            "gt",
            "valid_gt",
            "action_iou",
            "executed_iou",
            "reward_gain_vs_fast",
        ],
        "causal_contract": (
            "All state fields and behavior actions are computed before reading the current "
            "ground-truth label; ground truth is used only for label fields."
        ),
        "code_sha256": {name: file_hash(directory / name) for name in files},
    }
    run["fingerprint"] = fingerprint(run)
    return run


def load_behavior_run(directory):
    directory = Path(directory).expanduser().resolve()
    path = directory / "run.json"
    if not path.is_file():
        raise FileNotFoundError("Stage-16 run metadata missing: {}".format(path))
    run = json.loads(path.read_text(encoding="utf-8"))
    digest = run.get("fingerprint")
    body = {key: value for key, value in run.items() if key != "fingerprint"}
    if (
        run.get("schema") != SCHEMA
        or run.get("iou_protocol") != IOU_PROTOCOL
        or run.get("actions") != list(ACTION_NAMES)
        or run.get("candidate_layers") != list(CANDIDATE_LAYERS)
        or run.get("behaviors")
        != behavior_specs(
            [item["candidate_rate_percent"] for item in run.get("behaviors", [])],
            max((item["repeat"] for item in run.get("behaviors", [])), default=-1) + 1,
        )
        or fingerprint(body) != digest
    ):
        raise ValueError("Invalid Stage-16 run metadata or fingerprint")
    load_manifest(run["manifest"])
    return directory, run


def _expected_shapes(entry, run):
    steps = entry["stop"] - entry["start"] - 1
    episodes = len(run["behaviors"])
    dim = int(run["runtime"]["embed_dim"])
    side = int(run["runtime"]["lens_x"] ** 0.5)
    if side ** 2 != run["runtime"]["lens_x"]:
        raise ValueError("Expected a square search token grid")
    grid = int(run["state_grid"])
    shapes = {
        "frame_id": (steps,),
        "gt": (steps, 4),
        "valid_gt": (steps,),
        "image_hw": (steps, 2),
        "done": (steps,),
        "behavior_action": (episodes, steps),
        "action_cost": (episodes, steps),
        "action_propensity": (episodes, steps),
        "behavior_probability": (episodes, steps, len(ACTION_NAMES)),
        "candidate_used_before": (episodes, steps),
        "candidate_remaining_before": (episodes, steps),
        "remaining_candidate_rate": (episodes, steps),
        "template_summary": (episodes, steps, 2, dim),
        "search_token_grid": (episodes, steps, grid, grid, dim),
        "fast_score_map": (episodes, steps, 1, side, side),
        "selector_prob": (episodes, steps, len(CANDIDATE_LAYERS)),
        "selected_layer": (episodes, steps),
        "previous_state": (episodes, steps, 4),
        "previous_fast_box": (episodes, steps, 4),
        "previous_action": (episodes, steps),
        "fast_box": (episodes, steps, 4),
        "action_boxes": (episodes, steps, len(ACTION_NAMES), 4),
        "executed_box": (episodes, steps, 4),
        "action_iou": (episodes, steps, len(ACTION_NAMES)),
        "executed_iou": (episodes, steps),
        "reward_gain_vs_fast": (episodes, steps),
        "resize_factor": (episodes, steps),
    }
    if run["save_full_tokens"]:
        shapes["full_tokens"] = (
            episodes,
            steps,
            run["runtime"]["lens_z"] + run["runtime"]["lens_x"],
            dim,
        )
    return shapes


def _require_allclose(actual, expected, message, path, atol=1e-9):
    if not np.allclose(actual, expected, rtol=0, atol=atol, equal_nan=True):
        difference = np.abs(
            np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
        )
        finite = difference[np.isfinite(difference)]
        maximum = float(finite.max()) if finite.size else float("nan")
        raise ValueError("{} in {} (max abs diff {})".format(message, path, maximum))


def load_and_audit_cache(path, entry, run):
    """Validate schedule randomization, causal state continuity and reward labels."""
    path = Path(path)
    with np.load(str(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        data = {key: archive[key] for key in archive.files if key != "metadata_json"}
    if (
        metadata.get("run_fingerprint") != run["fingerprint"]
        or metadata.get("entry") != entry
    ):
        raise ValueError("Cache metadata mismatch: {}".format(path))

    shapes = _expected_shapes(entry, run)
    if set(data) != set(shapes):
        raise ValueError("Unexpected/missing Stage-16 fields: {}".format(path))
    for key, shape in shapes.items():
        if data[key].shape != shape:
            raise ValueError("Bad {} shape in {}: {} != {}".format(key, path, data[key].shape, shape))
        if key != "gt" and not np.isfinite(data[key]).all():
            raise ValueError("Non-finite {} in {}".format(key, path))

    for key in (
        "frame_id",
        "image_hw",
        "behavior_action",
        "action_cost",
        "candidate_used_before",
        "candidate_remaining_before",
        "selected_layer",
        "previous_action",
    ):
        if not np.issubdtype(data[key].dtype, np.integer):
            raise ValueError("{} must contain integers".format(key))
    for key in ("valid_gt", "done"):
        if data[key].dtype != np.bool_:
            raise ValueError("{} must be a boolean mask".format(key))

    steps = entry["stop"] - entry["start"] - 1
    expected_frames = np.arange(entry["start"] + 1, entry["stop"])
    if not np.array_equal(data["frame_id"], expected_frames):
        raise ValueError("Wrong decision frame IDs in {}".format(path))
    if not np.array_equal(data["done"], np.arange(steps) == steps - 1):
        raise ValueError("Terminal flags changed in {}".format(path))
    if np.any(data["image_hw"] <= 0) or np.any(data["resize_factor"] <= 0):
        raise ValueError("Invalid image dimensions or resize factor in {}".format(path))

    initial = np.asarray(metadata.get("init_box"), dtype=np.float64)
    if initial.shape != (4,) or not np.isfinite(initial).all() or np.any(initial[2:] <= 0):
        raise ValueError("Invalid Stage-16 initialization in {}".format(path))

    for episode, spec in enumerate(run["behaviors"]):
        seed = behavior_seed(
            run["behavior_seed"],
            run["manifest"]["fingerprint"],
            entry["sequence"],
            spec["behavior_id"],
        )
        expected = make_behavior_schedule(steps, spec["candidate_rate_percent"], seed)
        for key, field in (
            ("action", "behavior_action"),
            ("action_cost", "action_cost"),
            ("action_propensity", "action_propensity"),
            ("behavior_probability", "behavior_probability"),
            ("candidate_used_before", "candidate_used_before"),
            ("candidate_remaining_before", "candidate_remaining_before"),
            ("remaining_candidate_rate", "remaining_candidate_rate"),
        ):
            actual = data[field][episode]
            target = expected[key]
            if np.issubdtype(actual.dtype, np.integer):
                if not np.array_equal(actual, target):
                    raise ValueError("Behavior schedule changed for {} in {}".format(spec["behavior_id"], path))
            else:
                _require_allclose(actual, target, "Behavior schedule changed", path, 1e-12)

        actions = data["behavior_action"][episode]
        previous = np.concatenate(
            (initial[None], data["executed_box"][episode, :-1]), axis=0
        )
        previous_fast = np.concatenate(
            (initial[None], data["fast_box"][episode, :-1]), axis=0
        )
        _require_allclose(data["previous_state"][episode], previous, "Executed action was not written to the next state", path)
        _require_allclose(data["previous_fast_box"][episode], previous_fast, "Fast history is discontinuous", path)
        if not np.array_equal(data["previous_action"][episode], np.r_[-1, actions[:-1]]):
            raise ValueError("Previous-action history is non-causal in {}".format(path))

    actions = data["behavior_action"]
    if np.any((actions < 0) | (actions >= len(ACTION_NAMES))):
        raise ValueError("Invalid behavior action in {}".format(path))
    episode_index = np.arange(actions.shape[0])[:, None]
    step_index = np.arange(actions.shape[1])[None, :]
    executed = data["action_boxes"][episode_index, step_index, actions]
    _require_allclose(data["executed_box"], executed, "Executed boxes disagree with actions", path)
    _require_allclose(data["fast_box"], data["action_boxes"][:, :, 0], "Fast boxes disagree with action table", path)

    probabilities = data["selector_prob"]
    selected = data["selected_layer"]
    if (
        np.any((probabilities < 0.0) | (probabilities > 1.0))
        or np.any((selected < 7) | (selected > 12))
        or not np.allclose(
            probabilities[episode_index, step_index, selected - 7],
            probabilities.max(axis=2),
            rtol=0,
            atol=1e-7,
        )
    ):
        raise ValueError("Native selector outputs are inconsistent in {}".format(path))

    if np.any(data["action_boxes"][..., 2:] <= 0):
        raise ValueError("Invalid action box dimensions in {}".format(path))
    expected_iou = tracking_iou(
        data["action_boxes"],
        data["gt"][None, :, None, :],
        data["valid_gt"][None, :, None],
    )
    _require_allclose(data["action_iou"], expected_iou, "Action IoU labels changed", path, 1e-10)
    executed_iou = expected_iou[episode_index, step_index, actions]
    _require_allclose(data["executed_iou"], executed_iou, "Executed rewards changed", path, 1e-10)
    _require_allclose(
        data["reward_gain_vs_fast"],
        executed_iou - expected_iou[:, :, 0],
        "Immediate Fast-relative rewards changed",
        path,
        1e-10,
    )

    nonfast = actions > 0
    effect = np.max(np.abs(data["executed_box"] - data["fast_box"]), axis=2) > 1e-7
    effect_ratio = float(effect[nonfast].mean()) if nonfast.any() else 0.0
    fast_episode = next(
        index
        for index, spec in enumerate(run["behaviors"])
        if spec["candidate_rate_percent"] == 0.0
    )
    other = np.arange(len(run["behaviors"])) != fast_episode
    divergence = np.max(
        np.abs(data["previous_state"][other] - data["previous_state"][fast_episode]),
        axis=2,
    ) > 1e-7
    action_counts = Counter(actions.reshape(-1).tolist())
    valid = data["valid_gt"]
    report = {
        "split": entry["split"],
        "sequence": entry["sequence"],
        "episodes": len(run["behaviors"]),
        "steps_per_episode": steps,
        "transitions": int(actions.size),
        "valid_label_ratio": float(valid.mean()),
        "candidate_action_ratio": float(nonfast.mean()),
        "candidate_action_effect_ratio": effect_ratio,
        "state_divergence_vs_fast_ratio": float(divergence.mean()),
        "action_distribution": "|".join(
            "{}:{}".format(name, action_counts.get(index, 0))
            for index, name in enumerate(ACTION_NAMES)
        ),
    }
    return report, data, metadata
