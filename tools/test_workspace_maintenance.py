"""Check restoration preflight, idempotence and refusal to overwrite local work."""

import json
import tempfile
import unittest
from pathlib import Path

from restore_legacy_tracking import digest,restore

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


if __name__ == "__main__":
    unittest.main()
