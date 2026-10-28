"""Contracts and audits for paired persistent structural-compression rollouts."""

import argparse
import json
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint, load_manifest, tracking_iou
from stage8_intervention_common import select_anchor_offsets
from stage19_ffn_compression import make_masks


SCHEMA = "stage19_closed_loop_compression_mechanism_v1"


def nonempty_path(value):
    if not str(value).strip():
        raise argparse.ArgumentTypeError("Empty path; check that the shell variable is exported")
    return value


def validate_manifest(manifest):
    manifest = load_manifest(manifest)
    design = manifest.get("compression_design", {})
    if (manifest.get("purpose") != SCHEMA or design.get("calibration_horizon") != 5
            or design.get("primary_horizon") != design.get("horizon")
            or design.get("horizon", 0) < 5 or design.get("anchors", 0) < 1):
        raise ValueError("Expected a frozen Stage-19 compression manifest")
    excluded = set(manifest["excluded_source_val_sequences"])
    if any(entry["sequence"] in excluded for entry in manifest["sequences"]):
        raise ValueError("Source validation videos must not enter Stage-19")
    for entry in manifest["sequences"]:
        select_anchor_offsets(entry["stop"] - entry["start"] - 1, design["horizon"], design["anchors"])
    return manifest


def make_run(manifest, identity, widths, fraction, per_layer, seed):
    manifest = validate_manifest(manifest)
    if len(widths) != 6:
        raise ValueError("Stage-19 compresses exactly the six prefix FFNs")
    directory = Path(__file__).resolve().parent
    files = ("stage19_compression_common.py", "stage19_ffn_compression.py",
             "stage19_compression_collect.py", "stage4_runtime.py", "stage1_validate.py",
             "stage4_common.py", "validation_metrics.py")
    run = {"schema": SCHEMA, "manifest": manifest, "runtime": identity,
           "masks": make_masks(widths, fraction, per_layer, seed),
           "fraction": float(fraction), "per_layer": int(per_layer), "seed": int(seed),
           "design": manifest["compression_design"],
           "reference": "native_sgla_rule_l1_l6_plus_one_candidate",
           "compression": "physical_fc1_rows_and_fc2_columns_persistent_for_entire_window",
           "speed_claim_tested": False,
           "code_sha256": {name: file_hash(directory / name) for name in files}}
    run["fingerprint"] = fingerprint(run)
    return run


def cache_path(directory, entry):
    return Path(directory) / entry["split"] / (entry["sequence"] + "__stage19_compression.npz")


def _same(actual, expected, label, atol=1e-8):
    if np.shape(actual) != np.shape(expected) or not np.allclose(
            actual, expected, rtol=0, atol=atol, equal_nan=True):
        raise ValueError("Stage-19 audit failed: " + label)


def load_and_audit_cache(path, entry, run):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        data = {key: archive[key] for key in archive.files if key != "metadata_json"}
    if metadata.get("entry") != entry or metadata.get("run_fingerprint") != run["fingerprint"]:
        raise ValueError("Stage-19 cache identity mismatch: {}".format(path))
    a, h, m = run["design"]["anchors"], run["design"]["horizon"], len(run["masks"])
    n = entry["stop"] - entry["start"] - 1
    shapes = {"baseline_boxes": (n, 4), "baseline_layers": (n,), "baseline_gt": (n, 4),
              "baseline_valid": (n,), "baseline_iou": (n,), "anchor_offset": (a,),
              "reference_states": (a, h, 4), "reference_boxes": (a, h, 4),
              "gt": (a, h, 4), "valid": (a, h), "reference_iou": (a, h),
              "sync_boxes": (a, m, h, 4), "free_boxes": (a, m, h, 4),
              "sync_states": (a, m, h, 4), "free_states": (a, m, h, 4),
              "sync_layers": (a, m, h), "free_layers": (a, m, h),
              "sync_iou": (a, m, h), "free_iou": (a, m, h),
              "reference_probe_boxes": (h, 4), "post_restore_probe_boxes": (h, 4), "identity_boxes": (h, 4),
              "removed_parameters": (m,), "surgery_error": (m,)}
    if set(data) != set(shapes):
        raise ValueError("Unexpected/missing Stage-19 cache fields")
    for name, shape in shapes.items():
        if data[name].shape != shape:
            raise ValueError("Wrong {} shape in {}".format(name, path))
        if name not in ("baseline_gt", "gt") and not np.isfinite(data[name]).all():
            raise ValueError("Nonfinite {}".format(name))
    for name in ("valid", "baseline_valid"):
        if data[name].dtype != np.bool_:
            raise ValueError("Visibility must be boolean")
    for name in ("sync_layers", "free_layers", "baseline_layers"):
        if not np.issubdtype(data[name].dtype, np.integer) or np.any((data[name] < 7) | (data[name] > 12)):
            raise ValueError("Invalid candidate layer trace")
    offsets = select_anchor_offsets(n, h, a)
    if not np.array_equal(offsets, data["anchor_offset"]):
        raise ValueError("Anchors changed")
    initial = np.asarray(metadata["init_box"], dtype=np.float64)
    if initial.shape != (4,) or not np.isfinite(initial).all() or np.any(initial[2:] <= 0):
        raise ValueError("Invalid initial state")
    baseline_states = np.concatenate((initial[None], data["baseline_boxes"][:-1]))
    for name, source in (("reference_states", baseline_states), ("reference_boxes", data["baseline_boxes"]),
                         ("gt", data["baseline_gt"]), ("valid", data["baseline_valid"])):
        _same(data[name], np.stack([source[o - 1:o - 1 + h] for o in offsets]), name)
    expected_sync = np.broadcast_to(data["reference_states"][:, None], (a, m, h, 4))
    _same(data["sync_states"], expected_sync, "teacher crops and decoding origin")
    _same(data["free_states"][:, :, 0], expected_sync[:, :, 0], "shared branch initial state")
    _same(data["free_states"][:, :, 1:], data["free_boxes"][:, :, :-1], "student crop feedback continuity")
    _same(data["sync_boxes"][:, :, 0], data["free_boxes"][:, :, 0], "identical first-step outputs", 1e-6)
    _same(data["sync_layers"][:, :, 0], data["free_layers"][:, :, 0], "identical first-step choices")
    for name in ("reference_probe_boxes", "post_restore_probe_boxes", "identity_boxes"):
        _same(data[name], data["reference_boxes"][0], name, 1e-6)
    for name in ("baseline_boxes", "reference_boxes", "sync_boxes", "free_boxes"):
        if np.any(data[name][..., 2:] <= 0):
            raise ValueError("Invalid box size")
    _same(data["baseline_iou"], tracking_iou(data["baseline_boxes"], data["baseline_gt"], data["baseline_valid"]), "baseline labels")
    _same(data["reference_iou"], tracking_iou(data["reference_boxes"], data["gt"], data["valid"]), "reference labels")
    for kind in ("sync", "free"):
        _same(data[kind + "_iou"], tracking_iou(data[kind + "_boxes"], data["gt"][:, None],
                                               data["valid"][:, None]), kind + " labels")
    embed = int(run["runtime"]["embed_dim"])
    expected_removed = np.asarray([len(spec["drop"]) * (2 * embed + 1) for spec in run["masks"]])
    _same(data["removed_parameters"], expected_removed, "physical parameter reduction", 0)
    if len(set(expected_removed.tolist())) != 1 or np.any(data["surgery_error"] < 0):
        raise ValueError("Unequal masks or invalid surgery error")
    return data


def load_run(directory):
    directory = Path(directory).expanduser().resolve()
    path = directory / "run.json"
    if not path.is_file():
        raise FileNotFoundError("Stage-19 run.json missing: {}".format(path))
    run = json.loads(path.read_text(encoding="utf-8"))
    if run.get("schema") != SCHEMA or run.get("fingerprint") != fingerprint(
            {key: value for key, value in run.items() if key != "fingerprint"}):
        raise ValueError("Stage-19 run fingerprint mismatch")
    validate_manifest(run["manifest"])
    widths = [run["masks"][i * run["per_layer"]]["width"] for i in range(6)]
    if run["masks"] != make_masks(widths, run["fraction"], run["per_layer"], run["seed"]):
        raise ValueError("Frozen compression masks changed")
    return directory, run
