"""Freeze or verify a complete Stage-16 behavior cache.

The freeze manifest is a cryptographic inventory, not a new copy of the data.
It binds every cache file, run.json, and the PASS_BEHAVIOR_DATA decision to one
fingerprint that later training stages can require.
"""

import argparse
import json
import stat
from pathlib import Path

from stage4_common import file_hash, fingerprint, write_json
from stage16_behavior_common import behavior_cache_path, load_behavior_run


SCHEMA = "stage16_frozen_cache_v1"
AUDIT_SCHEMA = "stage16_behavior_audit_v1"
AUDIT_STATUS = "PASS_BEHAVIOR_DATA"


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_audit_decision(path, run):
    path = Path(path).expanduser().resolve()
    decision = _load_json(path)
    digest = decision.get("fingerprint")
    body = {key: value for key, value in decision.items() if key != "fingerprint"}
    if (
        decision.get("schema") != AUDIT_SCHEMA
        or decision.get("status") != AUDIT_STATUS
        or decision.get("source_run_fingerprint") != run["fingerprint"]
        or decision.get("missing_clips") != 0
        or not decision.get("checks", {}).get("complete")
        or fingerprint(body) != digest
    ):
        raise ValueError("Stage-16 audit decision is not a valid complete pass: {}".format(path))
    return path, decision


def _expected_files(directory, run):
    return [
        behavior_cache_path(directory, entry).resolve()
        for entry in run["manifest"]["sequences"]
    ]


def _relative_path(path, directory):
    try:
        return path.relative_to(directory).as_posix()
    except ValueError as error:
        raise ValueError("Frozen cache file escaped the cache root: {}".format(path)) from error


def build_freeze_manifest(input_dir, audit_decision_path):
    directory, run = load_behavior_run(input_dir)
    audit_path, audit = _validate_audit_decision(audit_decision_path, run)
    expected = _expected_files(directory, run)
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot freeze an incomplete Stage-16 cache; first missing file: {}".format(
                missing[0]
            )
        )

    files = []
    total_bytes = 0
    for index, path in enumerate(expected, start=1):
        size = path.stat().st_size
        files.append(
            {
                "path": _relative_path(path, directory),
                "size_bytes": int(size),
                "sha256": file_hash(path),
            }
        )
        total_bytes += size
        if index % 20 == 0 or index == len(expected):
            print("[hash] {}/{} cache files".format(index, len(expected)), flush=True)

    manifest = {
        "schema": SCHEMA,
        "analysis_role": "immutable_input_contract_for_stage16b",
        "source_cache": str(directory),
        "source_run_fingerprint": run["fingerprint"],
        "source_manifest_fingerprint": run["manifest"]["fingerprint"],
        "audit_decision_path": str(audit_path),
        "audit_decision_fingerprint": audit["fingerprint"],
        "audit_decision_sha256": file_hash(audit_path),
        "run_json_sha256": file_hash(directory / "run.json"),
        "num_files": len(files),
        "total_bytes": int(total_bytes),
        "files": files,
    }
    manifest["fingerprint"] = fingerprint(manifest)
    return manifest


def load_freeze_manifest(path):
    path = Path(path).expanduser().resolve()
    data = _load_json(path)
    digest = data.get("fingerprint")
    body = {key: value for key, value in data.items() if key != "fingerprint"}
    if data.get("schema") != SCHEMA or fingerprint(body) != digest:
        raise ValueError("Invalid Stage-16 freeze manifest or fingerprint: {}".format(path))
    return path, data


def verify_frozen_cache(manifest_path, input_dir=None, verify_hashes=True):
    manifest_path, frozen = load_freeze_manifest(manifest_path)
    directory = Path(input_dir or frozen["source_cache"]).expanduser().resolve()
    directory, run = load_behavior_run(directory)
    if run["fingerprint"] != frozen["source_run_fingerprint"]:
        raise ValueError("Stage-16 run fingerprint differs from the frozen contract")
    if file_hash(directory / "run.json") != frozen["run_json_sha256"]:
        raise ValueError("Frozen run.json content changed")
    audit_path = Path(frozen["audit_decision_path"]).expanduser().resolve()
    audit_path, audit = _validate_audit_decision(audit_path, run)
    if (
        audit["fingerprint"] != frozen["audit_decision_fingerprint"]
        or file_hash(audit_path) != frozen["audit_decision_sha256"]
    ):
        raise ValueError("Frozen PASS_BEHAVIOR_DATA decision changed")

    expected_relative = [
        _relative_path(path, directory) for path in _expected_files(directory, run)
    ]
    observed_relative = [item["path"] for item in frozen["files"]]
    if observed_relative != expected_relative or frozen["num_files"] != len(expected_relative):
        raise ValueError("Frozen cache inventory no longer matches the Stage-16 manifest")
    discovered = sorted(
        _relative_path(path.resolve(), directory)
        for path in directory.rglob("*__stage16_behavior.npz")
    )
    if discovered != sorted(expected_relative):
        raise ValueError("Stage-16 cache contains added or missing behavior files")

    total_bytes = 0
    for index, item in enumerate(frozen["files"], start=1):
        path = (directory / item["path"]).resolve()
        if _relative_path(path, directory) != item["path"] or not path.is_file():
            raise FileNotFoundError("Frozen cache file missing or unsafe: {}".format(path))
        size = path.stat().st_size
        if size != item["size_bytes"]:
            raise ValueError("Frozen cache file size changed: {}".format(path))
        if verify_hashes and file_hash(path) != item["sha256"]:
            raise ValueError("Frozen cache file content changed: {}".format(path))
        total_bytes += size
        if verify_hashes and (index % 20 == 0 or index == len(frozen["files"])):
            print("[verify] {}/{} cache files".format(index, len(frozen["files"])), flush=True)
    if total_bytes != frozen["total_bytes"]:
        raise ValueError("Frozen cache total size changed")
    return directory, run, frozen


def make_cache_read_only(directory, frozen):
    paths = [directory / "run.json"] + [directory / item["path"] for item in frozen["files"]]
    for path in paths:
        mode = path.stat().st_mode
        path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    return len(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Stage-16 behavior cache directory")
    parser.add_argument("--audit_decision", help="PASS_BEHAVIOR_DATA decision JSON")
    parser.add_argument("--output", help="Freeze manifest to create")
    parser.add_argument("--verify_manifest", help="Existing freeze manifest to verify")
    parser.add_argument("--skip_file_hashes", action="store_true")
    parser.add_argument("--make_read_only", action="store_true")
    args = parser.parse_args()

    if args.verify_manifest:
        directory, _, frozen = verify_frozen_cache(
            args.verify_manifest,
            input_dir=args.input,
            verify_hashes=not args.skip_file_hashes,
        )
        if args.make_read_only:
            count = make_cache_read_only(directory, frozen)
            print("Marked {} source files read-only.".format(count))
        print("Frozen cache verified: {}".format(frozen["fingerprint"]))
        return

    if not args.input or not args.audit_decision or not args.output:
        parser.error("creation requires --input, --audit_decision, and --output")
    output = Path(args.output).expanduser().resolve()
    frozen = build_freeze_manifest(args.input, args.audit_decision)
    write_json(output, frozen)
    directory, _, verified = verify_frozen_cache(
        output, input_dir=args.input, verify_hashes=False
    )
    if args.make_read_only:
        count = make_cache_read_only(directory, verified)
        print("Marked {} source files read-only.".format(count))
    print("Freeze manifest: {}".format(output))
    print("Frozen cache fingerprint: {}".format(frozen["fingerprint"]))
    print("Files: {}; bytes: {}".format(frozen["num_files"], frozen["total_bytes"]))


if __name__ == "__main__":
    main()
