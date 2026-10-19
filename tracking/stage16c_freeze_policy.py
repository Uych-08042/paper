"""Freeze the passed Stage-16B ensemble into one deployable policy bundle.

The bundle binds the three checkpoints, train-only preprocessing, the frozen
P30 threshold, and the Stage-16 behavior-cache lineage.  It deliberately reads
the threshold from the *training* budget row; validation results cannot change
the deployed Stage-16C policy.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint, write_json
from stage16_behavior_common import ACTION_NAMES, CANDIDATE_LAYERS, load_behavior_run
from stage16_freeze_cache import load_freeze_manifest
from stage16_recurrent_cql import PREPARED_SCHEMA, SCALAR_NAMES


SCHEMA = "stage16c_frozen_policy_v1"
STAGE16B_SCHEMA = "stage16b_recurrent_cql_v1"
STAGE16B_PASS = "PASS_OFFLINE_Q_SCREEN"
POLICY_NAME = "tavrtrack_p30"


def _load_fingerprinted_json(path, schema=None):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("Required JSON missing: {}".format(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    digest = data.get("fingerprint")
    body = {key: value for key, value in data.items() if key != "fingerprint"}
    if digest is None or fingerprint(body) != digest:
        raise ValueError("Fingerprint mismatch: {}".format(path))
    if schema is not None and data.get("schema") != schema:
        raise ValueError("Unexpected schema in {}: {}".format(path, data.get("schema")))
    return path, data


def _resolve_artifact(path, stage16b_dir, fallback):
    candidate = Path(path).expanduser() if path else Path()
    if path and candidate.is_file():
        return candidate.resolve()
    replacement = (Path(stage16b_dir) / fallback).resolve()
    if replacement.is_file():
        return replacement
    raise FileNotFoundError(
        "Stage-16B artifact missing at {} and {}".format(path, replacement)
    )


def _primary_train_row(path, primary_budget):
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [
        row
        for row in rows
        if row.get("split") == "train"
        and abs(float(row["budget_percent"]) - float(primary_budget)) <= 1e-12
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one training budget row for {}, got {}".format(
                primary_budget, len(matches)
            )
        )
    return matches[0]


def _artifact_record(role, path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("Policy artifact missing: {}".format(path))
    return {
        "role": role,
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": file_hash(path),
    }


def build_policy_bundle(stage16b_dir, freeze_manifest_path):
    stage16b_dir = Path(stage16b_dir).expanduser().resolve()
    decision_path, decision = _load_fingerprinted_json(
        stage16b_dir / "stage16b_decision.json", STAGE16B_SCHEMA
    )
    experiment_path, experiment = _load_fingerprinted_json(
        stage16b_dir / "stage16b_experiment.json", STAGE16B_SCHEMA
    )
    if decision.get("status") != STAGE16B_PASS:
        raise ValueError(
            "Stage-16C requires {}, got {}".format(
                STAGE16B_PASS, decision.get("status")
            )
        )
    if decision.get("experiment_fingerprint") != experiment["fingerprint"]:
        raise ValueError("Stage-16B decision and experiment contract disagree")

    _, frozen = load_freeze_manifest(freeze_manifest_path)
    if frozen["fingerprint"] != decision.get("source_freeze_fingerprint"):
        raise ValueError("Behavior-cache freeze fingerprint differs from Stage-16B")
    if frozen["source_run_fingerprint"] != decision.get("source_run_fingerprint"):
        raise ValueError("Behavior run fingerprint differs from Stage-16B")
    behavior_directory, behavior_run = load_behavior_run(frozen["source_cache"])
    if (
        behavior_run["fingerprint"] != frozen["source_run_fingerprint"]
        or behavior_run["manifest"]["fingerprint"]
        != frozen["source_manifest_fingerprint"]
        or file_hash(behavior_directory / "run.json") != frozen["run_json_sha256"]
    ):
        raise ValueError("Frozen Stage-16 behavior run metadata changed")

    recorded_code = decision.get("code_sha256", {}).get(
        "stage16_recurrent_cql.py"
    )
    current_code_path = Path(__file__).resolve().parent / "stage16_recurrent_cql.py"
    if recorded_code != file_hash(current_code_path):
        raise ValueError(
            "stage16_recurrent_cql.py changed after the passed Stage-16B run"
        )

    primary_budget = float(decision["screen"]["primary_budget"])
    if abs(primary_budget - 30.0) > 1e-12:
        raise ValueError(
            "The predeclared Stage-16C policy is P30, got budget {}".format(
                primary_budget
            )
        )
    budget_path = stage16b_dir / "stage16b_budget_summary.csv"
    train_row = _primary_train_row(budget_path, primary_budget)
    threshold = float(train_row["threshold"])
    if not np.isfinite(threshold):
        raise ValueError("Frozen candidate-advantage threshold is not finite")

    artifacts = decision.get("artifacts", {})
    checkpoint_paths = [
        _resolve_artifact(
            path,
            stage16b_dir,
            "stage16b_seed_{}.pt".format(seed),
        )
        for path, seed in zip(artifacts.get("checkpoints", []), decision["seeds"])
    ]
    if len(checkpoint_paths) != len(decision["seeds"]) or not checkpoint_paths:
        raise ValueError("Stage-16B checkpoint count does not match the frozen seeds")
    pca_path = _resolve_artifact(
        artifacts.get("token_pca"),
        stage16b_dir,
        "prepared/stage16b_token_pca.npz",
    )
    normalization_path = _resolve_artifact(
        artifacts.get("scalar_normalization"),
        stage16b_dir,
        "prepared/stage16b_scalar_normalization.npz",
    )
    prepared_path = stage16b_dir / "prepared" / "stage16b_prepared_run.json"
    if not prepared_path.is_file():
        raise FileNotFoundError("Prepared Stage-16B contract missing: {}".format(prepared_path))
    _, prepared = _load_fingerprinted_json(prepared_path, PREPARED_SCHEMA)
    if (
        prepared["fingerprint"] != decision.get("prepared_fingerprint")
        or prepared["fingerprint"] != experiment.get("prepared_fingerprint")
        or prepared.get("source_run_fingerprint")
        != decision.get("source_run_fingerprint")
        or prepared.get("source_freeze_fingerprint")
        != decision.get("source_freeze_fingerprint")
        or experiment.get("source_run_fingerprint")
        != decision.get("source_run_fingerprint")
        or experiment.get("source_freeze_fingerprint")
        != decision.get("source_freeze_fingerprint")
        or experiment.get("seeds") != decision.get("seeds")
        or abs(float(experiment.get("primary_budget", -1.0)) - primary_budget)
        > 1e-12
    ):
        raise ValueError("Stage-16B prepared, experiment, and decision contracts disagree")

    with np.load(str(pca_path), allow_pickle=False) as archive:
        mean = archive["mean"]
        components = archive["components"]
    with np.load(str(normalization_path), allow_pickle=False) as archive:
        scalar_names = [str(value) for value in archive["names"].tolist()]
        scalar_mean = archive["mean"]
        scalar_scale = archive["scale"]
    model_config = dict(decision["model_config"])
    if (
        components.ndim != 2
        or mean.shape != (components.shape[1],)
        or components.shape[0] != int(model_config["token_dim"])
    ):
        raise ValueError("Token PCA dimensions disagree with the frozen model")
    if (
        scalar_names != list(SCALAR_NAMES)
        or scalar_mean.shape != (len(SCALAR_NAMES),)
        or scalar_scale.shape != scalar_mean.shape
        or np.any(scalar_scale <= 0.0)
        or int(model_config["scalar_dim"]) != len(SCALAR_NAMES)
    ):
        raise ValueError("Scalar normalization disagrees with Stage-16B features")

    records = [
        _artifact_record("stage16b_decision", decision_path),
        _artifact_record("stage16b_experiment", experiment_path),
        _artifact_record("stage16b_budget_summary", budget_path),
        _artifact_record("stage16b_prepared_contract", prepared_path),
        _artifact_record("token_pca", pca_path),
        _artifact_record("scalar_normalization", normalization_path),
    ]
    records.extend(
        _artifact_record("checkpoint_seed_{}".format(seed), path)
        for seed, path in zip(decision["seeds"], checkpoint_paths)
    )

    bundle = {
        "schema": SCHEMA,
        "analysis_role": "immutable_deployment_contract_for_one_stage16c_run",
        "policy_name": POLICY_NAME,
        "source_stage16b_directory": str(stage16b_dir),
        "source_decision_fingerprint": decision["fingerprint"],
        "source_experiment_fingerprint": experiment["fingerprint"],
        "source_run_fingerprint": decision["source_run_fingerprint"],
        "source_freeze_fingerprint": frozen["fingerprint"],
        "source_manifest_fingerprint": frozen["source_manifest_fingerprint"],
        "prepared_fingerprint": decision["prepared_fingerprint"],
        "seeds": list(decision["seeds"]),
        "model_config": model_config,
        "state_grid": int(behavior_run["state_grid"]),
        "runtime_feature_identity": {
            key: behavior_run["runtime"][key]
            for key in ("lens_z", "lens_x", "embed_dim")
        },
        "actions": list(ACTION_NAMES),
        "candidate_layers": list(CANDIDATE_LAYERS),
        "scalar_names": list(SCALAR_NAMES),
        "primary_budget_percent": primary_budget,
        "candidate_advantage_threshold": threshold,
        "threshold_source": "stage16b_training_support_quantile_only",
        "budget_rounding": "floor(num_decisions * rate + 0.5)",
        "budget_semantics": "per_clip_hard_upper_bound_without_forced_spending",
        "validation_policy": (
            "The threshold, ensemble, action set, and P30 budget are frozen. "
            "Stage-16C may be run once on the manifest validation split without retuning."
        ),
        "artifacts": records,
        "code_sha256": {
            "stage16_recurrent_cql.py": recorded_code,
            "stage16c_freeze_policy.py": file_hash(Path(__file__).resolve()),
        },
    }
    bundle["fingerprint"] = fingerprint(bundle)
    return bundle


def load_policy_bundle(path, verify_artifacts=True):
    path, bundle = _load_fingerprinted_json(path, SCHEMA)
    if (
        bundle.get("policy_name") != POLICY_NAME
        or bundle.get("actions") != list(ACTION_NAMES)
        or bundle.get("candidate_layers") != list(CANDIDATE_LAYERS)
        or bundle.get("scalar_names") != list(SCALAR_NAMES)
        or abs(float(bundle.get("primary_budget_percent", -1.0)) - 30.0) > 1e-12
        or int(bundle.get("state_grid", 0)) < 1
        or set(bundle.get("runtime_feature_identity", {}))
        != {"lens_z", "lens_x", "embed_dim"}
    ):
        raise ValueError("Invalid or altered Stage-16C policy contract")
    roles = [item["role"] for item in bundle.get("artifacts", [])]
    expected_roles = {
        "stage16b_decision",
        "stage16b_experiment",
        "stage16b_budget_summary",
        "stage16b_prepared_contract",
        "token_pca",
        "scalar_normalization",
        *("checkpoint_seed_{}".format(seed) for seed in bundle.get("seeds", [])),
    }
    if len(roles) != len(set(roles)) or set(roles) != expected_roles:
        raise ValueError("Stage-16C artifact roles are missing, extra, or duplicated")
    source_dir = Path(__file__).resolve().parent
    for name in ("stage16_recurrent_cql.py", "stage16c_freeze_policy.py"):
        expected = bundle.get("code_sha256", {}).get(name)
        observed = file_hash(source_dir / name)
        if expected != observed:
            raise ValueError(
                "Frozen policy source changed for {}: expected {}, observed {}".format(
                    name, expected, observed
                )
            )
    if verify_artifacts:
        for item in bundle["artifacts"]:
            artifact = Path(item["path"]).expanduser().resolve()
            if not artifact.is_file():
                raise FileNotFoundError("Frozen policy artifact missing: {}".format(artifact))
            if artifact.stat().st_size != item["size_bytes"]:
                raise ValueError("Frozen policy artifact size changed: {}".format(artifact))
            if file_hash(artifact) != item["sha256"]:
                raise ValueError("Frozen policy artifact content changed: {}".format(artifact))
    return path, bundle


def artifact_path(bundle, role):
    matches = [item for item in bundle["artifacts"] if item["role"] == role]
    if len(matches) != 1:
        raise ValueError("Expected one frozen artifact for role {}".format(role))
    return Path(matches[0]["path"]).expanduser().resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage16b_dir")
    parser.add_argument("--freeze_manifest")
    parser.add_argument("--output")
    parser.add_argument("--verify_bundle", default=None)
    parser.add_argument("--skip_artifact_hashes", action="store_true")
    args = parser.parse_args()

    if args.verify_bundle:
        _, bundle = load_policy_bundle(
            args.verify_bundle, verify_artifacts=not args.skip_artifact_hashes
        )
        print("Frozen Stage-16C policy verified: {}".format(bundle["fingerprint"]))
        return
    if not args.stage16b_dir or not args.freeze_manifest or not args.output:
        parser.error(
            "creation requires --stage16b_dir, --freeze_manifest, and --output"
        )
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Policy bundle exists; do not overwrite a frozen policy")
    bundle = build_policy_bundle(args.stage16b_dir, args.freeze_manifest)
    write_json(output, bundle)
    load_policy_bundle(output)
    print("Policy bundle: {}".format(output))
    print("Policy fingerprint: {}".format(bundle["fingerprint"]))
    print(
        "Frozen policy: {} | budget {:.1f}% | threshold {:.12g} | seeds {}".format(
            bundle["policy_name"],
            bundle["primary_budget_percent"],
            bundle["candidate_advantage_threshold"],
            "/".join(str(seed) for seed in bundle["seeds"]),
        )
    )


if __name__ == "__main__":
    main()
