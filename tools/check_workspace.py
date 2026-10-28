"""Verify preserved bytes, archive hashes and the active frozen-source dependency closure."""

import ast
import json
import re
from pathlib import Path

from restore_legacy_tracking import inside,match_file


def check(root):
    root = Path(root).resolve()
    plan = json.loads((root / "docs/maintenance/CLEANUP_MANIFEST.json").read_text(encoding="utf-8"))
    if plan.get("schema") != "workspace_cleanup_v1":
        raise ValueError("Unknown cleanup manifest")
    errors,restored,regenerated,line_endings = [],0,0,[]
    def verify(path,sha,lf,label,error):
        result = match_file(path,sha,lf)
        if result is None:
            errors.append(error)
        elif result == "line_endings_only":
            line_endings.append(label)
    for item in plan["preserved_files"]:
        path = inside(root,item["path"])
        verify(path,item["sha256"],item.get("sha256_lf"),item["path"],"Preserved file changed: " + item["path"])
    archived = {}
    for item in plan["moved_files"]:
        path = inside(root,item["destination"])
        verify(path,item["sha256"],item.get("sha256_lf"),item["destination"],"Moved file changed: " + item["destination"])
        if item["destination"].startswith("archive/legacy_tracking/"):
            archived[Path(item["source"]).stem] = item
            original = inside(root,item["source"])
            if original.exists():
                verify(original,item["sha256"],item.get("sha256_lf"),item["source"],"Restored historical source differs: " + item["source"])
                restored += 1
        elif inside(root,item["source"]).exists():
            errors.append("Old document path unexpectedly exists: " + item["source"])
    for item in plan["edited_files"]:
        path = inside(root,item["path"])
        verify(path,item["after_sha256"],item.get("after_sha256_lf"),item["path"],"Document/ignore edit differs: " + item["path"])
    for item in plan["deleted_files"]:
        regenerated += int(inside(root,item["path"]).exists())
    for name in plan["retained_tracking_modules"]:
        path = root / "tracking" / (name+".py")
        if not path.is_file():
            errors.append("Missing retained module: " + name)
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        referenced = set()
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                referenced.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node,ast.ImportFrom) and node.module:
                referenced.add(node.module.split(".")[0])
            elif isinstance(node,ast.Constant) and isinstance(node.value,str):
                referenced.update(re.findall("[A-Za-z_][A-Za-z0-9_]*",node.value))
        for missing in referenced.intersection(archived):
            if not (root / "tracking" / (missing+".py")).is_file():
                errors.append("Active import/hash filename points into archive: {} -> {}".format(name,missing))
    return dict(status="PASS" if not errors else "FAIL",preserved_files=len(plan["preserved_files"]),
        archived_moves=len(plan["moved_files"]),retained_tracking_modules=len(plan["retained_tracking_modules"]),
        restored_legacy_scripts=restored,regenerated_disposable_caches=regenerated,
        line_ending_only_differences=len(line_endings),line_ending_examples=line_endings[:8],
        hash_policy="Maintenance: UTF-8 text allows CRLF/LF only; binaries and experiment hashes remain strict",errors=errors)


def main():
    result = check(Path(__file__).resolve().parents[1])
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
