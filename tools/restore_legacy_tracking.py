"""Restore archived experiment scripts at their original paths without overwriting changes."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            value.update(block)
    return value.hexdigest()


TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".ipynb", ".yaml", ".yml", ".ini", ".toml", ".sh", ".ps1", ".rst", ".cfg"}


def text_lf_digest(path):
    """Maintenance comparison only: preserve all bytes except CRLF -> LF.

    This does not replace any experiment's strict file_hash or fingerprint.
    Binary weights/caches remain byte-exact even if their bytes decode as UTF-8.
    """
    path = Path(path)
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name != ".gitignore":
        return None
    data = path.read_bytes()
    if b"\0" in data:
        return None
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def match_file(path,expected_sha256,expected_lf=None):
    path = Path(path)
    if not path.is_file():
        return None
    if digest(path) == expected_sha256:
        return "exact"
    if expected_lf is not None and text_lf_digest(path) == expected_lf:
        return "line_endings_only"
    return None


def inside(root,relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Manifest path must be repository-relative")
    target = (root / relative).resolve()
    if target == root or root not in target.parents or ".git" in target.relative_to(root).parts:
        raise ValueError("Manifest path escapes repository")
    return target


def restore(root,apply=False):
    root = Path(root).resolve()
    plan = json.loads((root / "docs/maintenance/CLEANUP_MANIFEST.json").read_text(encoding="utf-8"))
    if plan.get("schema") != "workspace_cleanup_v1":
        raise ValueError("Unknown cleanup manifest")
    pending,already,line_endings = [],0,0
    # Validate the complete batch before copying any file.
    for item in plan["moved_files"]:
        if not item["destination"].startswith("archive/legacy_tracking/"):
            continue
        source = inside(root,item["destination"])
        target = inside(root,item["source"])
        if target.parent != root / "tracking" or target.suffix != ".py":
            raise ValueError("Legacy restoration is restricted to tracking/*.py")
        source_match = match_file(source,item["sha256"],item.get("sha256_lf"))
        if source_match is None:
            raise ValueError("Archived source changed: " + item["destination"])
        line_endings += source_match == "line_endings_only"
        if target.exists():
            target_match = match_file(target,item["sha256"],item.get("sha256_lf"))
            if target_match is None:
                raise ValueError("Refusing to overwrite changed file: " + item["source"])
            line_endings += target_match == "line_endings_only"
            already += 1
        else:
            pending.append((source,target))
    if apply:
        for source,target in pending:
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
    return dict(archived_scripts=len(pending)+already,already_present=already,
        restored=len(pending) if apply else 0,would_restore=len(pending),applied=apply,
        line_ending_only_checks=line_endings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply",action="store_true",help="Copy all archived scripts back; default only checks and previews")
    args = parser.parse_args()
    result = restore(Path(__file__).resolve().parents[1],args.apply)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
