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
    pending,already = [],0
    # Validate the complete batch before copying any file.
    for item in plan["moved_files"]:
        if not item["destination"].startswith("archive/legacy_tracking/"):
            continue
        source = inside(root,item["destination"])
        target = inside(root,item["source"])
        if target.parent != root / "tracking" or target.suffix != ".py":
            raise ValueError("Legacy restoration is restricted to tracking/*.py")
        if not source.is_file() or digest(source) != item["sha256"]:
            raise ValueError("Archived source changed: " + item["destination"])
        if target.exists():
            if not target.is_file() or digest(target) != item["sha256"]:
                raise ValueError("Refusing to overwrite changed file: " + item["source"])
            already += 1
        else:
            pending.append((source,target))
    if apply:
        for source,target in pending:
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
    return dict(archived_scripts=len(pending)+already,already_present=already,
        restored=len(pending) if apply else 0,would_restore=len(pending),applied=apply)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply",action="store_true",help="Copy all archived scripts back; default only checks and previews")
    args = parser.parse_args()
    result = restore(Path(__file__).resolve().parents[1],args.apply)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
