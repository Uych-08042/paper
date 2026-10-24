"""Frozen small-budget A/B intervention-value representation experiment."""

from pathlib import Path, PurePosixPath

import numpy as np

from stage4_common import file_hash, load_manifest
from stage18_freeze_representation import verify_representation_bundle
from stage23_development import frozen_split, read_sealed, sealed
from stage25_relation_common import continuation, keyed_seed, load_contract, schedule
from stage25_relation_screen import verify_code


SCHEMA = "stage26_intervention_value_representation_v1"
ARMS = ("stop_gradient", "value_aligned")
SEEDS, SPLIT_SEED = (2087, 2088, 2089), 2086
HORIZON, REPEATS, ANCHORS, ROUNDS = 20, 2, 4, 2
GAMMA, VALUE_SCALE, RANK_WEIGHT = .97, .02, 1.
PREFIX_LR, HEAD_LR, WEIGHT_DECAY = 1e-5, 3e-4, .01
BOOTSTRAPS, ANALYSIS_SEED, MINIMUM_GAIN = 5000, 2090, .001
CONTROLS = ("value", "memory_fast", "l7", "l8", "l9", "l10", "l11", "l12", "random")
FILES = ("stage26_value_common.py", "stage26_value_runtime.py", "stage26_value_train.py", "stage26_value_summary.py")
LIB_FILES = ("lib/models/sglatrack/vit.py", "lib/models/sglatrack/sglatrack.py",
             "lib/models/sglatrack/deit.py", "lib/models/sglatrack/base_backbone.py", "lib/models/layers/head.py",
             "lib/utils/box_ops.py", "lib/utils/focal_loss.py",
             "lib/utils/heapmap_utils.py")


def split_videos(names):
    if (not isinstance(names, list) or len(names) < 48 or len(set(names)) != len(names)
            or any(not isinstance(n, str) or not n or n in (".", "..") or "/" in n or "\\" in n for n in names)):
        raise ValueError("Require at least 48 distinct frozen fit video identities")
    order = np.random.default_rng(SPLIT_SEED).permutation(len(names))
    ordered = [names[int(i)] for i in order]
    return ordered[:32], ordered[32:48], ordered[48:]


def anchors_for(actions):
    actions = np.asarray(actions)
    candidates = np.flatnonzero((actions > 0) & (np.arange(len(actions)) + HORIZON <= len(actions)))
    if len(candidates) < ANCHORS:
        raise ValueError("Too few complete H20 candidate slots")
    return candidates[np.linspace(0, len(candidates) - 1, ANCHORS).astype(int)]


def example_schedule(steps, seed, name, round_index):
    actions = schedule(steps, keyed_seed(seed, name, int(round_index), "stage26_behavior"))
    return actions, anchors_for(actions)


def branch_actions(actions, anchor, seed, name, round_index):
    return continuation(actions, anchor, keyed_seed(seed, name, int(round_index), int(anchor), "stage26_later_layers"))


def planned_jobs(contract, seed):
    jobs = []
    for round_index in range(contract["rounds"]):
        order = np.random.default_rng(keyed_seed(seed, round_index, "stage26_video_order")).permutation(contract["training"])
        for name in order:
            actions, anchors = example_schedule(contract["clip_steps"][str(name)], seed, str(name), round_index)
            jobs.extend((str(name), round_index, int(anchor), actions) for anchor in anchors)
    if len(jobs) != contract["expected_attempts"]:
        raise ValueError("Unexpected training quota")
    return jobs


def post_values(rewards, valid):
    rewards, valid = np.asarray(rewards, float), np.asarray(valid)
    if (rewards.shape != (REPEATS, 6, HORIZON) or valid.shape != (HORIZON,)
            or valid.dtype != np.bool_ or not np.isfinite(rewards).all()
            or np.any((rewards < 0) | (rewards > 1)) or np.any(rewards[..., ~valid] != 0)
            or not np.allclose(rewards[0, :, 0], rewards[1, :, 0], rtol=0, atol=1e-12)):
        raise ValueError("Invalid paired H20 intervention rewards")
    weight = GAMMA ** np.arange(HORIZON)
    weight[0] = 0
    denominator = float(weight @ valid)
    q = (rewards * weight).sum(-1) / max(denominator, 1e-12)
    return q, bool(valid[0] and valid.mean() >= .8)


def pairwise_value_loss(torch, prediction, values, eligible=True):
    """Targets are constant finite-horizon labels, never differentiable returns."""
    if prediction.shape != (1, 6):
        raise ValueError("Expected six normalized candidate utility predictions")
    target = torch.as_tensor(values, dtype=prediction.dtype, device=prediction.device).detach()
    if target.shape != (6,) or not torch.isfinite(target).all():
        raise ValueError("Expected six finite post-action values")
    if not eligible:
        return prediction.sum() * 0
    left, right = torch.triu_indices(6, 6, offset=1, device=prediction.device)
    predicted_difference = prediction[0, left] - prediction[0, right]
    target_difference = (target[left] - target[right]) / VALUE_SCALE
    return torch.nn.functional.smooth_l1_loss(predicted_difference, target_difference)


def validate_sample(data, metadata):
    """Check quotas, full chosen H20 replay and the exact target of this version."""
    expected = dict(rewards=(2, 6, 20), valid=(20,), costs=(2, 6),
                    continuation=(2, 20), replay_boxes=(20, 4), live_boxes=(20, 4), target=(6,))
    if set(data) != set(expected):
        raise ValueError("Unexpected training-branch sample fields")
    for field, shape in expected.items():
        if np.shape(data[field]) != shape or not np.isfinite(data[field]).all():
            raise ValueError("Malformed training-branch field: " + field)
    actions = data["continuation"]
    if (actions.dtype.kind not in "iu" or not np.isin(actions, range(7)).all()
            or np.any(actions[:, 0] == 0) or not np.array_equal(actions[0] > 0, actions[1] > 0)
            or data["costs"].dtype.kind not in "iu"
            or not np.all(data["costs"] == np.count_nonzero(actions[0]))):
        raise ValueError("Candidate branches did not use equal exact cost")
    q, eligible = post_values(data["rewards"], data["valid"])
    if (not np.allclose(data["target"], q.mean(0), rtol=0, atol=1e-12)
            or eligible != metadata["eligible"]):
        raise ValueError("Cached target differs from current-version post-action branches")
    if (np.any(data["live_boxes"][:, 2:] <= 0) or np.any(data["replay_boxes"][:, 2:] <= 0)
            or not np.allclose(data["replay_boxes"], data["live_boxes"], rtol=0, atol=1e-6)):
        raise ValueError("Chosen first layer does not replay the full subsequent live H20")
    for field in ("teacher_prefix_sha256", "teacher_head_sha256"):
        if not isinstance(metadata.get(field), str) or len(metadata[field]) != 64:
            raise ValueError("Missing current teacher version identity")


def source_lineage(bundle_path, manifest_path, stage25_root):
    _, bundle, repdir = verify_representation_bundle(bundle_path)
    manifest = load_manifest(manifest_path)
    if manifest["fingerprint"] != bundle["source_manifest_fingerprint"]:
        raise ValueError("Frozen manifest differs from the original representation")
    fit, tune = frozen_split(bundle, manifest)
    root = Path(stage25_root).expanduser().resolve()
    source = load_contract(root / "cache")
    verify_code(source)
    screen = read_sealed(root / "screen_v1/stage25_screen_contract.json")
    decision = read_sealed(root / "screen_v1/stage25_decision.json")
    if (source["fit"] != fit or source["bundle_fingerprint"] != bundle["fingerprint"]
            or screen["collection_fingerprint"] != source["fingerprint"]
            or decision["screen_fingerprint"] != screen["fingerprint"]
            or decision["status"] != "FAIL_RELATIONAL_STATE_SCREEN"
            or decision["tune_labels_loaded"] or decision["validation_labels_loaded"]):
        raise ValueError("Require the reported frozen negative Stage-25 screen and matching source IDs")
    for relative, digest in screen["cache_sha256"].items():
        name = PurePosixPath(relative)
        if name.is_absolute() or ".." in name.parts or ":" in relative or "\\" in relative:
            raise ValueError("Unsafe frozen source cache reference")
        if file_hash(root / "cache" / relative) != digest:
            raise ValueError("Stage-25 source cache changed: " + relative)
    training, check, unused = split_videos(fit)
    return bundle, repdir, manifest, dict(train=training, check=check, unused_fit=unused,
        excluded_tune=tune, excluded_validation=[e["sequence"] for e in manifest["sequences"] if e["split"] == "val"]), decision["fingerprint"]


def make_contract(bundle, manifest, groups, source_id, runtime, mode):
    if mode not in ("smoke", "full"):
        raise ValueError("Unknown Stage-26 mode")
    names = groups["train"][:3] if mode == "smoke" else groups["train"]
    maximum = 64 if mode == "smoke" else 128
    rounds = 1 if mode == "smoke" else ROUNDS
    return sealed(dict(schema=SCHEMA, mode=mode, bundle_fingerprint=bundle["fingerprint"],
        manifest_fingerprint=manifest["fingerprint"], stage25_decision_fingerprint=source_id,
        groups=groups, training=names, checking=[] if mode == "smoke" else groups["check"],
        seeds=[SEEDS[0]] if mode == "smoke" else list(SEEDS), arms=list(ARMS), rounds=rounds,
        maximum_frames=maximum, anchors=ANCHORS, horizon=HORIZON, repeats=REPEATS, gamma=GAMMA,
        clip_steps={e["sequence"]: min(128 if e["sequence"] in groups["check"] else maximum,
                    e["stop"] - e["start"] - 1) for e in manifest["sequences"]
                    if e["sequence"] in names + ([] if mode == "smoke" else groups["check"])},
        expected_attempts=len(names) * ANCHORS * rounds,
        branch_frames_per_attempt=REPEATS * 6 * HORIZON,
        runtime=runtime, split_seed=SPLIT_SEED, rank_weight=RANK_WEIGHT, value_scale=VALUE_SCALE,
        optimizer=dict(prefix_lr=PREFIX_LR, head_lr=HEAD_LR, weight_decay=WEIGHT_DECAY, grad_clip=1.),
        trainable_prefix_layers=[5, 6], fixed_tracking_memory=True,
        freshness="one fresh independent prefix/H20 episode per anchor; update once only after all six current-version labels",
        no_stale_label_reuse=True, ppo_trained=False, checkpoint_rule="final_only",
        metric="valid_tracked_strict_21_threshold_auc_excludes_initial_v1",
        inference="greedy six-layer choice at canonical P30 calls; no current candidate trial outputs in state",
        minimum_gain=MINIMUM_GAIN, bootstrap_samples=BOOTSTRAPS, analysis_seed=ANALYSIS_SEED,
        gate="H2-H20 B-minus-A routing-excess lower95>0 and all three positive; B-minus-each-static family lower95>0 and mean>=.001; separate true closed-loop AUC guard",
        warning="All 48 videos were previously used development data; frozen initialization already saw them. No external claim.",
        library_code_sha256={n: file_hash(Path(__file__).parent.parent / n) for n in LIB_FILES},
        code_sha256={n: file_hash(Path(__file__).parent / n) for n in FILES + (
            "stage4_runtime.py", "stage4_common.py", "validation_metrics.py", "stage18_environment.py",
            "stage18_tracking_loss.py", "stage18_temporal_model.py", "stage18_policy_runtime.py",
            "stage18_representation_train.py", "stage21_counterfactual_runtime.py", "stage23_development.py",
            "stage23_branch_ppo.py", "stage23_development_summary.py", "stage18_freeze_representation.py",
            "stage25_relation_collect.py", "stage25_relation_common.py", "stage25_relation_screen.py")}))


def verify_contract(directory):
    value = read_sealed(Path(directory) / "stage26_contract.json")
    if value.get("schema") != SCHEMA or value.get("arms") != list(ARMS) or value.get("ppo_trained") is not False:
        raise ValueError("Not a frozen Stage-26 A/B experiment")
    mode, groups = value.get("mode"), value.get("groups", {})
    if mode not in ("smoke", "full") or len(groups.get("train", [])) != 32 or len(groups.get("check", [])) != 16:
        raise ValueError("Invalid frozen 32/16 development split")
    roles = [groups.get(field, []) for field in ("train", "check", "unused_fit", "excluded_tune", "excluded_validation")]
    names = [n for role in roles for n in role]
    if len(names) != len(set(names)) or any(not isinstance(n, str) or not n or n in (".", "..") or "/" in n or "\\" in n for n in names):
        raise ValueError("Duplicate, unsafe or overlapping development video roles")
    if (value["training"] != (groups["train"][:3] if mode == "smoke" else groups["train"])
            or value["checking"] != ([] if mode == "smoke" else groups["check"])
            or value["seeds"] != ([SEEDS[0]] if mode == "smoke" else list(SEEDS))
            or value["rounds"] != (1 if mode == "smoke" else ROUNDS)
            or value["anchors"] != ANCHORS or value["horizon"] != HORIZON or value["repeats"] != REPEATS
            or value["gamma"] != GAMMA or value["minimum_gain"] != MINIMUM_GAIN
            or value["expected_attempts"] != len(value["training"]) * ANCHORS * value["rounds"]
            or set(value["clip_steps"]) != set(value["training"] + value["checking"])):
        raise ValueError("Frozen training/budget/decision rules changed")
    if any(type(v) is not int or not 64 <= v <= (64 if mode == "smoke" else 128) for v in value["clip_steps"].values()):
        raise ValueError("Frozen clip bounds escaped the predeclared frame budget")
    for name, digest in value["code_sha256"].items():
        if Path(name).name != name or "\\" in name or file_hash(Path(__file__).parent / name) != digest:
            raise ValueError("Frozen Stage-26 dependency changed: " + name)
    if set(value.get("library_code_sha256", {})) != set(LIB_FILES):
        raise ValueError("Missing tracking library code freeze")
    for name, digest in value["library_code_sha256"].items():
        if file_hash(Path(__file__).parent.parent / name) != digest:
            raise ValueError("Frozen tracking library changed: " + name)
    return value
