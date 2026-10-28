"""Check restoration preflight, idempotence and refusal to overwrite local work."""

import json
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from restore_legacy_tracking import digest,match_file,restore,text_lf_digest
from check_workspace import check

HERE = Path(__file__).resolve().parent


class RestoreTests(unittest.TestCase):
    def test_preview_restore_idempotence_and_tamper(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root = Path(tmp).resolve()
            self.assertEqual(root.parent,HERE)
            archive = root / "archive/legacy_tracking"
            archive.mkdir(parents=True)
            records = []
            for name in ("stage1_demo.py","test_stage1_demo.py"):
                path = archive / name
                path.write_text("value = 1\n",encoding="utf-8")
                records.append(dict(source="tracking/"+name,destination="archive/legacy_tracking/"+name,sha256=digest(path)))
            manifest = root / "docs/maintenance/CLEANUP_MANIFEST.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps(dict(schema="workspace_cleanup_v1",moved_files=records)),encoding="utf-8")
            self.assertEqual(restore(root)["would_restore"],2)
            self.assertFalse((root/"tracking").exists())
            self.assertEqual(restore(root,True)["restored"],2)
            self.assertEqual(restore(root,True)["restored"],0)
            changed = root / records[1]["source"]
            changed.write_text("local work\n",encoding="utf-8")
            (root / records[0]["source"]).unlink()
            with self.assertRaisesRegex(ValueError,"Refusing to overwrite"):
                restore(root,True)
            self.assertFalse((root / records[0]["source"]).exists())
            self.assertEqual(changed.read_text(encoding="utf-8"),"local work\n")

    def test_manifest_cannot_escape_workspace(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root = Path(tmp).resolve()
            self.assertEqual(root.parent,HERE)
            manifest = root / "docs/maintenance/CLEANUP_MANIFEST.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps(dict(schema="workspace_cleanup_v1",moved_files=[dict(
                source="../elsewhere.py",destination="archive/legacy_tracking/../../../../elsewhere.py",sha256="bad")])),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"repository-relative"):
                restore(root,True)

    def test_lf_crlf_restoration_keeps_checkout_bytes_and_rejects_real_changes(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root = Path(tmp).resolve()
            self.assertEqual(root.parent,HERE)
            source = root / "archive/legacy_tracking/stage_old.py"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"value = 1\r\n")
            row = dict(source="tracking/stage_old.py",destination="archive/legacy_tracking/stage_old.py",
                sha256=digest(source),sha256_lf=text_lf_digest(source))
            manifest = root / "docs/maintenance/CLEANUP_MANIFEST.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps(dict(schema="workspace_cleanup_v1",moved_files=[row])),encoding="utf-8")
            source.write_bytes(b"value = 1\n")
            self.assertEqual(restore(root)["line_ending_only_checks"],1)
            self.assertEqual(restore(root,True)["restored"],1)
            target = root / row["source"]
            self.assertEqual(target.read_bytes(),b"value = 1\n")
            self.assertEqual(restore(root,True)["restored"],0)
            target.write_bytes(b"value = 2\n")
            with self.assertRaisesRegex(ValueError,"Refusing to overwrite"):
                restore(root,True)
            self.assertEqual(target.read_bytes(),b"value = 2\n")

    def test_only_newline_representation_is_normalized_and_binary_stays_strict(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root = Path(tmp).resolve()
            self.assertEqual(root.parent,HERE)
            source = root / "code.py"
            source.write_bytes(b"value = 1\r\n")
            raw,lf = digest(source),text_lf_digest(source)
            source.write_bytes(b"value = 1\n")
            self.assertEqual(match_file(source,raw,lf),"line_endings_only")
            for changed in (b"value = 2\n",b"value = 1 \n",b"\xef\xbb\xbfvalue = 1\n",b"value = 1\r"):
                source.write_bytes(changed)
                self.assertIsNone(match_file(source,raw,lf))
            binary = root / "model.pt"
            binary.write_bytes(b"model\r\n")
            raw = digest(binary)
            binary.write_bytes(b"model\n")
            self.assertIsNone(text_lf_digest(binary))
            self.assertIsNone(match_file(binary,raw,hashlib.sha256(b"model\n").hexdigest()))

    @unittest.skipUnless(shutil.which("git"),"Git needed to reproduce checkout EOL conversion")
    def test_real_git_lf_checkout_passes_and_still_rejects_code_and_weight_changes(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            base = Path(tmp).resolve()
            self.assertEqual(base.parent,HERE)
            repo,checkout = base / "source",base / "linux_checkout"
            (repo / "tracking").mkdir(parents=True)
            (repo / "tools").mkdir()
            for name in ("check_workspace.py","restore_legacy_tracking.py"):
                shutil.copy2(HERE / name,repo / "tools" / name)
            code = repo / "tracking/stage44_demo.py"
            code.write_bytes(b"value = 1\r\n")
            model = repo / "model.pt"
            model.write_bytes(b"\0weights\r\n")
            notebook = repo / "tracking/analysis_results.ipynb"
            notebook.write_bytes(b'{\r\n "cells": []\r\n}\r\n')
            archived = repo / "archive/legacy_tracking/stage_old.py"
            archived.parent.mkdir(parents=True)
            archived.write_bytes(b"old = 1\r\n")
            (repo / ".gitattributes").write_bytes(b"*.py text\n*.ipynb text\n*.pt -text\n")
            manifest = repo / "docs/maintenance/CLEANUP_MANIFEST.json"
            manifest.parent.mkdir(parents=True)
            plan = dict(schema="workspace_cleanup_v1",preserved_files=[
                dict(path="tracking/stage44_demo.py",sha256=digest(code),sha256_lf=text_lf_digest(code)),
                dict(path="model.pt",sha256=digest(model)),
                dict(path="tracking/analysis_results.ipynb",sha256=digest(notebook),sha256_lf=text_lf_digest(notebook))],moved_files=[dict(
                source="tracking/stage_old.py",destination="archive/legacy_tracking/stage_old.py",
                sha256=digest(archived),sha256_lf=text_lf_digest(archived))],edited_files=[],deleted_files=[],
                retained_tracking_modules=["stage44_demo"])
            manifest.write_text(json.dumps(plan),encoding="utf-8")
            def git(*args):
                return subprocess.run(["git","-C",str(repo),*args],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            git("init","--quiet")
            git("-c","core.autocrlf=true","add","-A")
            checkout.mkdir()
            git("-c","core.autocrlf=false","-c","core.eol=lf","checkout-index","--all","--prefix="+checkout.as_posix()+"/")
            checked_code = checkout / "tracking/stage44_demo.py"
            self.assertEqual(checked_code.read_bytes(),b"value = 1\n")
            self.assertNotEqual(digest(checked_code),plan["preserved_files"][0]["sha256"])
            env = dict(os.environ,PYTHONDONTWRITEBYTECODE="1")
            result = subprocess.run([sys.executable,"-B",str(checkout / "tools/check_workspace.py")],env=env,
                check=True,stdout=subprocess.PIPE,text=True,encoding="utf-8")
            audit = json.loads(result.stdout)
            self.assertEqual(audit["status"],"PASS")
            self.assertEqual(audit["line_ending_only_differences"],3)
            self.assertEqual(restore(checkout,True)["restored"],1)
            checked_code.write_bytes(b"value = 2\n")
            self.assertEqual(check(checkout)["status"],"FAIL")
            checked_code.write_bytes(b"value = 1\n")
            (checkout / "model.pt").write_bytes(b"\0different weights\n")
            self.assertEqual(check(checkout)["status"],"FAIL")


if __name__ == "__main__":
    unittest.main()
