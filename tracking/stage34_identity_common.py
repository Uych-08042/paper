"""Frozen matched spatial-identity experiment; original artifacts stay immutable."""

import copy
import json
import math
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint
from stage27_uav_common import (IMAGE_SUFFIXES, checked_image, crop_geometry, frame_number, inside,
    nonempty_path, publish_identical, read_boxes, read_rgb, scene_geometry, seal, source_group)
from stage32_identity_attribution import load_sources, verify_unchanged


SCHEMA = "stage34_spatial_identity_representation_v1"
ARMS = ("readout_standard", "readout_contrast", "representation_standard", "representation_contrast")
METHODS = ("sgla", "fast_initial") + ARMS
DESIGN = dict(seed=2103, seeds=[2103, 2104, 2105], checking_videos=6,
    fit_samples_per_video=64, fit_hard_quota=32, minimum_hard_samples=32, minimum_hard_videos=4,
    minimum_coverage=.9, updates=512, smoke_updates=8, batch_size=2, save_every=128,
    identity_dim=32, temperature=.1, contrast_margin=.2, contrast_weight=.25, hard_negatives=8,
    learning_rate=.0001, weight_decay=.0001, gradient_clip=1.,
    bootstrap_samples=5000, bootstrap_seed=2106, minimum_auc_gain=.3)


def code_hashes():
    names = ("stage34_identity_common.py", "stage34_identity_runtime.py", "stage34_identity_experiment.py", "stage34_identity_summary.py")
    return {name: file_hash(Path(__file__).with_name(name)) for name in names}


def load_sealed(path, schema=SCHEMA):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != schema or data != seal({k: v for k, v in data.items() if k != "fingerprint"}):
        raise ValueError("Schema/fingerprint mismatch: " + str(path))
    return data


def load_plan(directory):
    data = load_sealed(Path(directory) / "manifest.json")
    if data.get("role") != "matched_single_path_identity_plan" or data.get("design") != DESIGN or data.get("code_sha256") != code_hashes():
        raise ValueError("Stage-34 design/code changed; preserve this run and create a new plan")
    for name, digest in data["source_file_sha256"].items():
        if file_hash(name) != digest:
            raise ValueError("Frozen source changed: " + name)
    for name, digest in data["dependency_sha256"].items():
        if file_hash(Path(__file__).parent / name) != digest:
            raise ValueError("Frozen dependency changed: " + name)
    for name, digest in data["original_source_code_sha256"].items():
        if file_hash(Path(__file__).parent / name) != digest:
            raise ValueError("Frozen original SGLA source changed: " + name)
    groups = [e["source_group_proxy"] for e in data["checking"]]
    if (len(groups) != DESIGN["checking_videos"] or len(set(groups)) != len(groups)
            or set(groups) & set(data["excluded_checking_groups"])):
        raise ValueError("Fresh checking split changed or overlaps consumed source groups")
    if len(data["fit"]) != 8 or any(e["split"] != "fit" for e in data["fit"]) or set(groups) & {e["source_group_proxy"] for e in data["fit"]}:
        raise ValueError("Fit/checking group isolation changed")
    return data


def supervision_masks(target, side):
    target = np.asarray(target, np.float32)
    grid = (np.arange(side, dtype=np.float32) + np.float32(.5)) / np.float32(side)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    center = target[:2] + target[2:] / 2
    positive = (xx >= target[0]) & (xx < target[0] + target[2]) & (yy >= target[1]) & (yy < target[1] + target[3])
    # Also supervise exactly the rounded center cell used by the CENTER loss.
    # This supplies tiny targets and prevents conflicting neighboring labels.
    pixel = np.rint(center * side).astype(int).clip(0, side - 1)
    positive[pixel[1], pixel[0]] = True
    expanded = (np.abs(xx - center[0]) <= .75 * target[2]) & (np.abs(yy - center[1]) <= .75 * target[3])
    padded = np.pad(positive, 1)
    neighbors = np.logical_or.reduce([padded[i:i + side, j:j + side] for i in range(3) for j in range(3)])
    return positive, ~expanded & ~neighbors


def fixed_samples(entry, data, factor, size, rng, side=16):
    eligible = {"easy": [], "hard": []}
    for row, frame_id in enumerate(data["frame_id"]):
        geometry = scene_geometry(data["previous"][row], data["gt"][row], data["image_hw"][row], factor, size)
        if geometry["coverage"] < DESIGN["minimum_coverage"] or not geometry["center_in_usable"]:
            continue
        cohort = "hard" if data["iou"][row] < .5 else "easy"
        crop = geometry["crop"]
        roi = np.r_[(data["gt"][row, :2] - crop[:2]) / crop[2], data["gt"][row, 2:] / crop[2]]
        if not supervision_masks(roi, side)[1].any():
            continue
        eligible[cohort].append(dict(sequence=entry["sequence"], frame_id=int(frame_id), cohort=cohort,
            previous=data["previous"][row].tolist(), factor=float(data["factor"][row]), target_roi=roi.tolist()))
    chosen = []
    for cohort, quota in (("hard", DESIGN["fit_hard_quota"]), ("easy", DESIGN["fit_samples_per_video"] - DESIGN["fit_hard_quota"])):
        rows = eligible[cohort]
        chosen.extend(rows[int(i)] for i in rng.permutation(len(rows))[:quota])
    if len(chosen) < DESIGN["fit_samples_per_video"]:
        ids = {r["frame_id"] for r in chosen}
        rest = [r for rows in eligible.values() for r in rows if r["frame_id"] not in ids]
        chosen.extend(rest[int(i)] for i in rng.permutation(len(rest))[:DESIGN["fit_samples_per_video"] - len(chosen)])
    return sorted(chosen, key=lambda r: r["frame_id"])


def sample_weights(samples):
    groups = {}
    for i, sample in enumerate(samples):
        groups.setdefault(sample["cohort"], {}).setdefault(sample["sequence"], []).append(i)
    if set(groups) != {"easy", "hard"}:
        raise ValueError("Both easy and hard fit supervision are required")
    weights = np.zeros(len(samples))
    for videos in groups.values():
        for ids in videos.values():
            weights[ids] = .5 / len(videos) / len(ids)
    return weights / weights.sum()


def fresh_entries(root, excluded, count, reader=read_rgb, expected_sequences=86, expected_frames=69941):
    """Inventory names/images only; never open excluded annotation contents."""
    directories = sorted(p for p in (root / "sequences").iterdir() if p.is_dir())
    inventory, total = {}, 0
    for directory in directories:
        frames = sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=frame_number)
        ids = [frame_number(p) for p in frames]
        if len(frames) < 2 or any(b != a + 1 for a, b in zip(ids, ids[1:])):
            raise ValueError("Invalid frame inventory: " + directory.name)
        total += len(frames)
        group = source_group(directory.name)
        if group not in excluded:
            inventory.setdefault(group, []).append((directory, frames))
    if len(directories) != expected_sequences or total != expected_frames:
        raise ValueError("VisDrone inventory differs from frozen 86-video / 69941-frame source")
    print("Fresh checking source groups available: {}; need {}".format(len(inventory), count), flush=True)
    if len(inventory) < count:
        raise ValueError("Insufficient untouched source groups; do not shrink or reuse old checking videos")
    rng = np.random.default_rng(DESIGN["seed"])
    selected = []
    groups = sorted(inventory)
    for index in rng.permutation(len(groups))[:count]:
        choices = inventory[groups[int(index)]]
        directory, paths = choices[int(rng.integers(len(choices)))]
        annotation = root / "annotations" / (directory.name + ".txt")
        gt = read_boxes(annotation)  # Preparation audit only, never method selection.
        if len(gt) != len(paths):
            raise ValueError("Fresh checking annotation coverage mismatch")
        frames = []
        for path in paths:
            digest = file_hash(path)
            image = reader(path)
            if image.ndim != 3 or image.shape[2] != 3 or file_hash(path) != digest:
                raise ValueError("Fresh image audit failed")
            frames.append(dict(file=path.relative_to(root).as_posix(), sha256=digest, height=int(image.shape[0]), width=int(image.shape[1])))
        selected.append(dict(sequence=directory.name, source_group_proxy=source_group(directory.name), split="checking",
            frames=frames, annotation=annotation.relative_to(root).as_posix(), annotation_sha256=file_hash(annotation), initial_box=gt[0].tolist()))
        print("[fresh image audit] {}: {} tracked frames".format(directory.name, len(frames) - 1), flush=True)
    return sorted(selected, key=lambda e: e["sequence"])


def prepare_plan(source, stage33, output, reader=read_rgb, expected_sequences=86, expected_frames=69941):
    source, stage33, output = Path(source).resolve(), Path(stage33).resolve(), Path(output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Use a separate Stage-34 output tree")
    if (output / "manifest.json").exists():
        existing = load_plan(output)
        if existing["source_stage31"] != str(source) or existing["source_stage33"] != str(stage33):
            raise ValueError("Existing plan belongs to another source")
        print("[reuse, audited] manifest " + existing["fingerprint"])
        return existing
    contract, _, records, bindings, _ = load_sources(source, "fit")
    prior_plan = load_sealed(stage33 / "plan.json", "stage33_pooled_descriptor_oof_v1")
    prior = load_sealed(stage33 / "stage33_decision.json", "stage33_pooled_descriptor_oof_v1")
    from stage33_descriptor_probe import dependency_hashes as probe_hashes
    expected_artifacts = set()
    for fold in range(8):
        for seed in (2100, 2101):
            for arm in ("scalar_balanced", "descriptor_fixed", "descriptor_learned"):
                stem = "fold{:02d}_seed{}_{}".format(fold, seed, arm)
                expected_artifacts.update(("checkpoints/" + stem + ".npz", "checkpoints/" + stem + ".json", "oof_predictions/" + stem + ".npz"))
    if (prior.get("mode") != "full" or prior.get("plan_fingerprint") != prior_plan["fingerprint"]
            or prior.get("role") != "development_only_pooled_descriptor_feasibility"
            or prior_plan.get("role") != "consumed_development_fit_only_fixed_budget_lovo"
            or prior_plan.get("code_sha256") != probe_hashes()
            or prior_plan.get("source_contract_fingerprint") != contract["fingerprint"]
            or prior.get("checking_labels_loaded") is not False or set(prior.get("artifacts_sha256", {})) != expected_artifacts):
        raise ValueError("Need the matched complete Stage-33 probe, with checking excluded")
    for relative, digest in prior["artifacts_sha256"].items():
        path = inside(stage33, relative)
        if file_hash(path) != digest:
            raise ValueError("Stage-33 artifact changed: " + relative)
        bindings[str(path)] = digest
    for filename in ("plan.json", "stage33_decision.json"):
        bindings[str(stage33 / filename)] = file_hash(stage33 / filename)
    rng = np.random.default_rng(DESIGN["seed"])
    samples, fit = [], []
    for entry, _, data, _ in records:
        rows = fixed_samples(entry, data, contract["search_factor"], contract["search_size"], rng,
                             math.isqrt(contract["runtime"].get("lens_x", 256)))
        if not rows:
            raise ValueError("No usable fit samples: " + entry["sequence"])
        samples.extend(rows)
        fit.append(copy.deepcopy(entry))
        print("[fit samples] {}: {} easy / {} hard".format(entry["sequence"], sum(r["cohort"] == "easy" for r in rows), sum(r["cohort"] == "hard" for r in rows)), flush=True)
    hard = [r for r in samples if r["cohort"] == "hard"]
    if len(hard) < DESIGN["minimum_hard_samples"] or len({r["sequence"] for r in hard}) < DESIGN["minimum_hard_videos"]:
        raise ValueError("Insufficient geometry-eligible hard supervision; preserve audit, do not relax the frozen rule")
    sample_weights(samples)
    excluded = set(contract["excluded_groups"]) | {e["source_group_proxy"] for e in contract["sequences"]}
    root = Path(contract["root"])
    checking = fresh_entries(root, excluded, DESIGN["checking_videos"], reader, expected_sequences, expected_frames)
    # Fit pixel verification needs only selected fit frames and each initial frame.
    for entry in fit:
        indices = {0} | {r["frame_id"] for r in samples if r["sequence"] == entry["sequence"]}
        for index in sorted(indices):
            checked_image(root, entry["frames"][index], reader)
    verify_unchanged(bindings)
    dependencies = ("stage4_runtime.py", "stage18_tracking_loss.py", "stage20_student_train.py",
                    "stage31_identity_common.py", "stage31_identity_engine.py", "stage32_identity_attribution.py")
    plan = seal(dict(schema=SCHEMA, role="matched_single_path_identity_plan", design=DESIGN,
        source_stage31=str(source), source_stage33=str(stage33), source_contract_fingerprint=contract["fingerprint"],
        source_file_sha256=bindings, dependency_sha256={name: file_hash(Path(__file__).with_name(name)) for name in dependencies},
        runtime=contract["runtime"], original_source_code_sha256=contract["source_code_sha256"],
        root=str(root), fit=fit, checking=checking, samples=samples, excluded_checking_groups=sorted(excluded),
        search_factor=contract["search_factor"], search_size=contract["search_size"], code_sha256=code_hashes(),
        checking_annotation_use="Preparation geometry audit; training never opens checking annotations/images",
        training_state="Fixed original SGLA crop parents; samples may use fit GT; no on-policy claim",
        novelty="Matched mechanism pilot, not an established novel tracking/compression method"))
    publish_identical(output / "manifest.json", plan)
    print("Manifest: {} (8 fit / 6 fresh checking groups)".format(output / "manifest.json"), flush=True)
    return plan
