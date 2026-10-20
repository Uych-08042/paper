"""Freeze a completed full Stage-18 warm-up for subsequent on-policy PPO.

Verification hashes checkpoint bytes without importing torch or unpickling them.
The PPO loader must also check the checkpoint's internal epoch and contract ID.
"""

import argparse
import csv
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath

from stage4_common import file_hash, fingerprint
from stage18_representation_train import load_pilot_contract


SCHEMA = "stage18_frozen_representation_v1"
REPRESENTATION_SCHEMA = "stage18_representation_v1"
PILOT_SCHEMA = "stage18_execution_pilot_v1"
REPOSITORY = Path(__file__).resolve().parent.parent
REPRESENTATION_CODE = {
    "tracking/stage18_representation_train.py",
    "tracking/stage18_tracking_loss.py",
    "lib/utils/heapmap_utils.py",
    "lib/utils/box_ops.py",
    "lib/utils/focal_loss.py",
}
PILOT_CODE = {
    "tracking/stage18_pilot.py",
    "tracking/stage18_environment.py",
    "tracking/stage18_temporal_model.py",
    "tracking/stage4_runtime.py",
    "tracking/stage1_validate.py",
}


def _read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: {}".format(path))
    return value


def _check_fingerprint(value, schema):
    body = {key: item for key, item in value.items() if key != "fingerprint"}
    if value.get("schema") != schema or fingerprint(body) != value.get("fingerprint"):
        raise ValueError("Invalid {} schema or canonical fingerprint".format(schema))


def _code_path(name):
    if not isinstance(name, str) or not name:
        raise ValueError("Invalid contract code path")
    relative = PurePosixPath(name.replace("\\", "/"))
    if (relative.is_absolute() or PureWindowsPath(name).drive
            or ".." in relative.parts or ":" in name):
        raise ValueError("Unsafe contract code path: " + name)
    if len(relative.parts) == 1:
        relative = PurePosixPath("tracking") / relative
    if not relative.parts or relative.parts[0] not in ("tracking", "lib"):
        raise ValueError("Unknown contract code path: " + name)
    path = (REPOSITORY / str(relative)).resolve()
    if not path.is_relative_to(REPOSITORY):
        raise ValueError("Contract code escaped the repository: " + name)
    return relative.as_posix(), path


def _validate_code(contract, required):
    recorded = contract.get("code_sha256")
    if not isinstance(recorded, dict):
        raise ValueError("Missing contract code_sha256")
    hashes = {}
    for name, digest in recorded.items():
        relative, path = _code_path(name)
        if relative in hashes:
            raise ValueError("Duplicate contract code path: " + relative)
        if file_hash(path) != digest:
            raise ValueError("Contract-bound source code changed: " + name)
        hashes[relative] = digest
    if not required.issubset(hashes):
        raise ValueError("Missing required contract code hashes: {}".format(
            sorted(required - hashes.keys())))
    return hashes


def _video_ids(value, label):
    if (not isinstance(value, list) or not value
            or any(not isinstance(name, str) or not name.strip()
                   or name in (".", "..") or "/" in name or "\\" in name
                   for name in value)
            or len(set(value)) != len(value)):
        raise ValueError("Invalid or duplicate {} video IDs".format(label))
    return value


def _split_rows(contract, field):
    rows = contract.get(field)
    if (not isinstance(rows, list) or not rows
            or any(not isinstance(row, dict)
                   or type(row.get("frames")) is not int or row["frames"] <= 0
                   for row in rows)):
        raise ValueError("Invalid representation {} rows".format(field))
    return _video_ids([row.get("sequence") for row in rows], field)


def _best_history(path, epochs):
    if type(epochs) is not int or epochs < 1:
        raise ValueError("Invalid representation epoch count")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if (len(set(columns)) != len(columns)
                or not {"epoch", "fit_loss", "tune_loss"}.issubset(columns)):
            raise ValueError("History requires epoch, fit_loss and tune_loss columns")
        rows = list(reader)
    if len(rows) != epochs:
        raise ValueError("History does not contain every completed training epoch")
    losses = []
    for expected_epoch, row in enumerate(rows, 1):
        try:
            epoch = int(row["epoch"])
            fit, tune = float(row["fit_loss"]), float(row["tune_loss"])
        except (ValueError, TypeError) as error:
            raise ValueError("Invalid epoch or loss in representation history") from error
        if (epoch != expected_epoch or None in row or None in row.values()
                or not math.isfinite(fit) or not math.isfinite(tune)):
            raise ValueError("History must have consecutive epochs and finite losses")
        losses.append((epoch, tune))
    # The trainer saves only strict improvements, so equal losses keep the first.
    return min(losses, key=lambda item: item[1])


def _artifact(role, path):
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError("Required frozen artifact missing: {}".format(path))
    return dict(role=role, path=str(path), size_bytes=path.stat().st_size,
                sha256=file_hash(path))


def _build_bundle(representation_dir, pilot_dir):
    repdir = Path(representation_dir).expanduser().resolve()
    pilotdir = Path(pilot_dir).expanduser().resolve()
    paths = {
        "best_checkpoint": repdir / "best.pt",
        "representation_contract": repdir / "stage18_representation_contract.json",
        "representation_decision": repdir / "stage18_representation_decision.json",
        "representation_history": repdir / "stage18_representation_history.csv",
        "pilot_contract": pilotdir / "stage18_pilot_contract.json",
        "pilot_decision": pilotdir / "stage18_pilot_decision.json",
    }
    artifacts = [_artifact(role, path) for role, path in paths.items()]
    contract = _read_json(paths["representation_contract"])
    _check_fingerprint(contract, REPRESENTATION_SCHEMA)
    decision = _read_json(paths["representation_decision"])
    if (contract.get("mode") != "full"
            or decision.get("schema") != REPRESENTATION_SCHEMA
            or decision.get("status") != "COMPLETE_REPRESENTATION_WARMUP"
            or decision.get("training_pass") is not True
            or decision.get("contract_fingerprint") != contract["fingerprint"]):
        raise ValueError("Require a completed, passed full representation warm-up")

    pilot = load_pilot_contract(pilotdir)
    _check_fingerprint(pilot, PILOT_SCHEMA)
    if (pilot != _read_json(paths["pilot_contract"])
            or contract.get("pilot_fingerprint") != pilot["fingerprint"]
            or contract.get("seed") != pilot.get("seed")):
        raise ValueError("Representation lineage differs from the original passed pilot")
    code = _validate_code(pilot, PILOT_CODE)
    for name, digest in _validate_code(contract, REPRESENTATION_CODE).items():
        if name in code and code[name] != digest:
            raise ValueError("Pilot and representation code hashes disagree: " + name)
        code[name] = digest

    fit = _video_ids(pilot.get("fit_videos"), "pilot fit")
    tune = _video_ids(pilot.get("tune_videos"), "pilot tune")
    videos = _video_ids(pilot.get("videos"), "pilot")
    if set(fit) & set(tune) or not set(videos).issubset(fit):
        raise ValueError("Pilot fit/tune overlap or pilot videos escaped fit split")
    if _split_rows(contract, "fit") != fit or _split_rows(contract, "tune") != tune:
        raise ValueError("Representation fit/tune rows differ from the exact pilot split")
    if (not isinstance(pilot.get("manifest"), str) or not pilot["manifest"]
            or not isinstance(pilot.get("runtime"), dict) or not pilot["runtime"]
            or not isinstance(pilot.get("model"), dict) or not pilot["model"]):
        raise ValueError("Pilot requires its original manifest, runtime and model")

    best_epoch, tune_loss = _best_history(paths["representation_history"], contract.get("epochs"))
    checkpoint_path = decision.get("best_checkpoint")
    if (type(decision.get("best_epoch")) is not int or decision["best_epoch"] != best_epoch
            or not isinstance(checkpoint_path, str) or not checkpoint_path
            or Path(checkpoint_path).expanduser().resolve() != paths["best_checkpoint"].resolve()
            or decision.get("best_checkpoint_sha256") != artifacts[0]["sha256"]):
        raise ValueError("Decision checkpoint/hash or best_epoch differs from tune-selected best.pt")
    if artifacts != [_artifact(role, path) for role, path in paths.items()]:
        raise ValueError("Representation or original pilot artifacts changed during verification")

    bundle = dict(
        schema=SCHEMA, representation_dir=str(repdir), pilot_dir=str(pilotdir),
        representation_contract_fingerprint=contract["fingerprint"],
        pilot_contract_fingerprint=pilot["fingerprint"],
        source_manifest_fingerprint=pilot["manifest"],
        model=pilot["model"], runtime=pilot["runtime"],
        fit_videos=fit, tune_videos=tune, pilot_videos=videos,
        fit=contract["fit"], tune=contract["tune"],
        best_epoch=best_epoch, best_tune_loss=tune_loss,
        artifacts=artifacts, code_sha256=code,
    )
    bundle["fingerprint"] = fingerprint(bundle)
    return bundle


def freeze_representation(representation_dir, output_path, *, pilot_dir):
    """Validate and write a freeze; return its bundle dict.

    ``pilot_dir`` is required and must name the original passed execution pilot.
    An existing identical bundle is accepted without writing it; an unequal one
    is never overwritten. Source artifacts and permissions are never modified.
    """
    bundle = _build_bundle(representation_dir, pilot_dir)
    output = Path(output_path).expanduser().resolve()
    content = json.dumps(bundle, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = output.open("x", encoding="utf-8")
    except FileExistsError:
        existing = _read_json(output)
        _check_fingerprint(existing, SCHEMA)
        if existing != bundle or existing["fingerprint"] != bundle["fingerprint"]:
            raise FileExistsError("Unequal frozen representation bundle already exists: {}".format(output))
    else:
        with handle:
            handle.write(content)
    return bundle


def verify_representation_bundle(bundle_path):
    """Return ``(resolved bundle Path, verified bundle dict, resolved repdir Path)``.

    Revalidate all six artifacts, source code, passed pilot, exact splits and
    tune selection, including every saved top-level lineage field.
    """
    path = Path(bundle_path).expanduser().resolve()
    bundle = _read_json(path)
    _check_fingerprint(bundle, SCHEMA)
    for field in ("representation_dir", "pilot_dir"):
        value = bundle.get(field)
        if (not isinstance(value, str) or not value
                or not Path(value).is_absolute() or str(Path(value).resolve()) != value):
            raise ValueError("Freeze requires a resolved absolute " + field)
    expected = _build_bundle(bundle["representation_dir"], bundle["pilot_dir"])
    if bundle != expected or bundle["fingerprint"] != expected["fingerprint"]:
        raise ValueError("Frozen representation artifacts or freeze lineage changed")
    return path, bundle, Path(bundle["representation_dir"]).resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation_dir")
    parser.add_argument("--pilot_dir")
    parser.add_argument("--output")
    parser.add_argument("--verify_bundle")
    args = parser.parse_args(argv)
    if args.verify_bundle is not None:
        if not args.verify_bundle.strip():
            parser.error("--verify_bundle must not be empty")
        if any(value is not None for value in (args.representation_dir, args.pilot_dir, args.output)):
            parser.error("--verify_bundle cannot be combined with creation arguments")
        path, bundle, _ = verify_representation_bundle(args.verify_bundle)
    else:
        if any(not value or not value.strip() for value in
               (args.representation_dir, args.pilot_dir, args.output)):
            parser.error("creation requires --representation_dir, --pilot_dir and --output")
        bundle = freeze_representation(args.representation_dir, args.output, pilot_dir=args.pilot_dir)
        path = Path(args.output).expanduser().resolve()
    print("Frozen representation: {}".format(path))
    print("Fingerprint: {}".format(bundle["fingerprint"]))


if __name__ == "__main__":
    main()
