"""Filesystem-only Stage-18 freeze lineage and tampering tests."""

import contextlib
import csv
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import stage18_freeze_representation as freeze
import stage18_representation_train as train
from stage4_common import file_hash, fingerprint


class MemoryFiles:
    """Virtualize fixture files only; repository code still uses real readers."""

    def __init__(self, root):
        self.root, self.files, self.directories = root, {}, {root}
        self.modified, self.clock = {}, 0
        self.original = {name: getattr(Path, name) for name in
                         ("open", "stat", "exists", "is_file", "mkdir", "unlink")}
        self.patches = []
        for name in self.original:
            handler = getattr(self, name)
            replacement = lambda path, *a, _handler=handler, **kw: _handler(path, *a, **kw)
            active = patch.object(Path, name, new=replacement)
            active.start()
            self.patches.append(active)

    def close(self):
        for active in reversed(self.patches):
            active.stop()

    def contains(self, path):
        return path == self.root or self.root in path.parents

    def save(self, path, data):
        self.clock += 1
        self.files[path] = data
        self.modified[path] = self.clock

    def open(self, path, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        if not self.contains(path):
            return self.original["open"](path, mode, buffering, encoding, errors, newline)
        encoding = "utf-8" if encoding in (None, "locale") else encoding
        if "x" in mode and path in self.files:
            raise FileExistsError(str(path))
        writing = any(flag in mode for flag in ("w", "x", "a", "+"))
        if not writing and path not in self.files:
            raise FileNotFoundError(str(path))
        if "x" in mode or "w" in mode:
            self.save(path, b"")
        data = self.files.get(path, b"")
        binary = "b" in mode
        owner = self
        stream_type = io.BytesIO if binary else io.StringIO
        class SavedBuffer(stream_type):
            def close(buffer):
                if writing and not buffer.closed:
                    value = buffer.getvalue()
                    owner.save(path, value if binary else value.encode(encoding or "utf-8"))
                super().close()
        stream = SavedBuffer(data if binary else data.decode(encoding or "utf-8"))
        if "a" in mode:
            stream.seek(0, 2)
        return stream

    def stat(self, path, **kwargs):
        if not self.contains(path):
            return self.original["stat"](path, **kwargs)
        if path not in self.files:
            raise FileNotFoundError(str(path))
        return SimpleNamespace(st_size=len(self.files[path]), st_mtime_ns=self.modified[path],
                               st_mode=0o100644)

    def exists(self, path, **kwargs):
        if self.contains(path):
            return path in self.files or path in self.directories
        return self.original["exists"](path, **kwargs)

    def is_file(self, path, **kwargs):
        if self.contains(path):
            return path in self.files
        return self.original["is_file"](path, **kwargs)

    def mkdir(self, path, *args, **kwargs):
        if not self.contains(path):
            return self.original["mkdir"](path, *args, **kwargs)
        self.directories.add(path)

    def unlink(self, path, **kwargs):
        if not self.contains(path):
            return self.original["unlink"](path, **kwargs)
        del self.files[path]


class RepresentationFreezeTests(unittest.TestCase):
    def setUp(self):
        self.root = (freeze.REPOSITORY / "__stage18_freeze_test_files__").resolve()
        self.files = MemoryFiles(self.root)
        self.addCleanup(self.files.close)
        self.repdir = self.root / "representation"
        self.pilotdir = self.root / "original-pilot"
        self.output = self.root / "frozen.json"
        self.repdir.mkdir()
        self.pilotdir.mkdir()
        self.pilot = dict(
            schema="stage18_execution_pilot_v1", manifest=fingerprint({"source": "train"}),
            runtime=dict(embed_dim=192, lens_z=64, lens_x=256, checkpoint_sha256="original"),
            model=dict(embed_dim=192, lens_z=64, search_side=16, memory_slots=4, memory_dim=64),
            seed=2053, videos=["fit-a"], fit_videos=["fit-a", "fit-b"],
            tune_videos=["tune-a"],
            code_sha256={name: file_hash(freeze.REPOSITORY / "tracking" / name) for name in (
                "stage18_pilot.py", "stage18_environment.py", "stage18_temporal_model.py",
                "stage4_runtime.py", "stage1_validate.py")},
        )
        self._save_pilot(rebind=False)
        self.contract = dict(
            schema="stage18_representation_v1", mode="full", seed=2053, epochs=3,
            pilot_fingerprint=self.pilot["fingerprint"],
            fit=[dict(sequence=name, frames=128) for name in self.pilot["fit_videos"]],
            tune=[dict(sequence="tune-a", frames=256)],
            code_sha256={
                name: file_hash(freeze.REPOSITORY / ("tracking/" + name if "/" not in name else name))
                for name in ("stage18_representation_train.py", "stage18_tracking_loss.py",
                             "lib/utils/heapmap_utils.py", "lib/utils/box_ops.py", "lib/utils/focal_loss.py")
            },
        )
        # Arbitrary bytes deliberately cannot be loaded as a torch checkpoint.
        (self.repdir / "best.pt").write_bytes(b"checkpoint-a")
        self.decision = dict(
            schema="stage18_representation_v1", status="COMPLETE_REPRESENTATION_WARMUP",
            training_pass=True, best_epoch=2, best_checkpoint=str(self.repdir / "best.pt"),
            best_checkpoint_sha256=file_hash(self.repdir / "best.pt"),
        )
        self._save_representation()
        self.history = [dict(epoch=epoch, fit_loss=1 / epoch, tune_loss=loss)
                        for epoch, loss in enumerate([5., 2., 3.], 1)]
        self._save_history()

    @staticmethod
    def _write(path, data):
        path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @staticmethod
    def _seal(data):
        data["fingerprint"] = fingerprint({key: value for key, value in data.items()
                                           if key != "fingerprint"})

    def _save_pilot(self, rebind=True):
        self._seal(self.pilot)
        self._write(self.pilotdir / "stage18_pilot_contract.json", self.pilot)
        self._write(self.pilotdir / "stage18_pilot_decision.json", dict(
            schema="stage18_execution_pilot_v1", status="PASS_EXECUTION_PILOT",
            contract_fingerprint=self.pilot["fingerprint"]))
        if rebind:
            self.contract["pilot_fingerprint"] = self.pilot["fingerprint"]
            self._save_representation()

    def _save_representation(self):
        self._seal(self.contract)
        self.decision["contract_fingerprint"] = self.contract["fingerprint"]
        self._write(self.repdir / "stage18_representation_contract.json", self.contract)
        self._write(self.repdir / "stage18_representation_decision.json", self.decision)

    def _save_history(self):
        with (self.repdir / "stage18_representation_history.csv").open(
                "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "fit_loss", "tune_loss"])
            writer.writeheader()
            writer.writerows(self.history)

    def _freeze(self):
        return freeze.freeze_representation(self.repdir, self.output, pilot_dir=self.pilotdir)

    def test_round_trip_exposes_ppo_integration_fields_and_original_artifacts(self):
        with patch.object(freeze, "load_pilot_contract", wraps=train.load_pilot_contract) as loader:
            bundle = self._freeze()
            resolved, verified, repdir = freeze.verify_representation_bundle(self.output)
        self.assertEqual(loader.call_count, 2)
        loader.assert_called_with(self.pilotdir)
        self.assertEqual((resolved, verified, repdir), (self.output, bundle, self.repdir))
        self.assertEqual(bundle["schema"], "stage18_frozen_representation_v1")
        self.assertEqual(bundle["pilot_dir"], str(self.pilotdir))
        self.assertEqual(bundle["representation_contract_fingerprint"], self.contract["fingerprint"])
        self.assertEqual(bundle["pilot_contract_fingerprint"], self.pilot["fingerprint"])
        self.assertEqual(bundle["source_manifest_fingerprint"], self.pilot["manifest"])
        for key in ("model", "runtime", "fit_videos", "tune_videos"):
            self.assertEqual(bundle[key], self.pilot[key])
        self.assertEqual(bundle["pilot_videos"], self.pilot["videos"])
        self.assertEqual(bundle["fit"], self.contract["fit"])
        self.assertEqual(bundle["tune"], self.contract["tune"])
        self.assertEqual(bundle["best_epoch"], 2)
        self.assertEqual(len(bundle["artifacts"]), 6)
        for record in bundle["artifacts"]:
            path = Path(record["path"])
            self.assertTrue(path.is_absolute())
            self.assertEqual(record["size_bytes"], path.stat().st_size)
            self.assertEqual(record["sha256"], file_hash(path))
        self.assertEqual(set(bundle["code_sha256"]), freeze.PILOT_CODE | freeze.REPRESENTATION_CODE)
        self.assertTrue(all(name.startswith(("tracking/", "lib/")) for name in bundle["code_sha256"]))
        self._seal(verified)
        self.assertEqual(verified["fingerprint"], bundle["fingerprint"])

    def test_idempotent_freeze_does_not_write_or_chmod_existing_files(self):
        bundle = self._freeze()
        paths = [Path(item["path"]) for item in bundle["artifacts"]] + [self.output]
        before = [(path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_mode) for path in paths]
        with patch.object(Path, "chmod", side_effect=AssertionError("chmod forbidden")):
            self.assertEqual(self._freeze(), bundle)
            freeze.verify_representation_bundle(self.output)
        self.assertEqual(before, [(path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_mode)
                                  for path in paths])

    def test_unequal_existing_bundle_is_never_overwritten(self):
        self._freeze()
        self.history[0]["fit_loss"] = 9.
        self._save_history()
        original = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            self._freeze()
        self.assertEqual(self.output.read_bytes(), original)

    def test_creation_requires_explicit_original_pilot_path(self):
        with self.assertRaises(TypeError):
            freeze.freeze_representation(self.repdir, self.output)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            freeze.main(["--representation_dir", str(self.repdir), "--output", str(self.output)])
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.output.exists())

    def test_smoke_and_unpassed_or_mismatched_decisions_are_rejected(self):
        for field, value in (("status", "PASS_SUPERVISED_SMOKE"), ("training_pass", False),
                             ("training_pass", 1), ("contract_fingerprint", "wrong"),
                             ("schema", "wrong")):
            with self.subTest(field=field, value=value):
                changed = dict(self.decision, **{field: value})
                self._write(self.repdir / "stage18_representation_decision.json", changed)
                with self.assertRaises(ValueError):
                    self._freeze()
        self.contract["mode"] = "smoke"
        self._save_representation()
        with self.assertRaisesRegex(ValueError, "full"):
            self._freeze()
        self.assertFalse(self.output.exists())

    def test_noncanonical_representation_contract_is_rejected(self):
        self.contract["epochs"] = 4
        self._write(self.repdir / "stage18_representation_contract.json", self.contract)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self._freeze()

    def test_original_pilot_pass_and_fingerprint_are_revalidated(self):
        decision_path = self.pilotdir / "stage18_pilot_decision.json"
        original = json.loads(decision_path.read_text(encoding="utf-8"))
        for field, value in (("status", "FAIL"), ("contract_fingerprint", "wrong")):
            with self.subTest(field=field):
                self._write(decision_path, dict(original, **{field: value}))
                with self.assertRaisesRegex(ValueError, "passed Stage-18A"):
                    self._freeze()
        self._write(decision_path, original)
        self.pilot["seed"] += 1
        self._write(self.pilotdir / "stage18_pilot_contract.json", self.pilot)
        with self.assertRaisesRegex(ValueError, "passed Stage-18A"):
            self._freeze()

    def test_different_original_pilot_cannot_be_substituted(self):
        self.pilot["runtime"]["checkpoint_sha256"] = "another original checkpoint"
        self._save_pilot(rebind=False)
        with self.assertRaisesRegex(ValueError, "lineage"):
            self._freeze()

    def test_representation_rows_must_match_complete_ordered_pilot_split(self):
        original = self.contract["fit"]
        for rows in (original[:1], list(reversed(original)), original + original[:1],
                     [dict(sequence="tune-a", frames=128)]):
            with self.subTest(rows=rows):
                self.contract["fit"] = rows
                self._save_representation()
                with self.assertRaises(ValueError):
                    self._freeze()

    def test_overlap_duplicates_and_pilot_outside_fit_are_rejected(self):
        for key, names in (("tune_videos", ["fit-a"]),
                           ("fit_videos", ["fit-a", "fit-a"]),
                           ("videos", ["tune-a"])):
            with self.subTest(key=key):
                original = self.pilot[key]
                self.pilot[key] = names
                self._save_pilot()
                with self.assertRaises(ValueError):
                    self._freeze()
                self.pilot[key] = original

    def test_repo_relative_tracking_and_lib_code_paths_are_supported(self):
        self.contract["code_sha256"] = {
            "tracking/" + name if "/" not in name else name: digest
            for name, digest in self.contract["code_sha256"].items()}
        self._save_representation()
        bundle = self._freeze()
        self.assertIn("tracking/stage18_representation_train.py", bundle["code_sha256"])
        self.assertIn("lib/utils/heapmap_utils.py", bundle["code_sha256"])
        freeze.verify_representation_bundle(self.output)

    def test_missing_wrong_and_unsafe_code_hashes_are_rejected(self):
        original = self.contract["code_sha256"]
        cases = [dict(original), dict(original), dict(original), dict(original)]
        del cases[0]["stage18_tracking_loss.py"]
        cases[1]["lib/utils/box_ops.py"] = "0" * 64
        cases[2]["../stage4_common.py"] = "0" * 64
        cases[3]["tracking/stage18_tracking_loss.py"] = original["stage18_tracking_loss.py"]
        for hashes in cases:
            with self.subTest(hashes=hashes):
                self.contract["code_sha256"] = hashes
                self._save_representation()
                with self.assertRaises(ValueError):
                    self._freeze()

    def test_tune_selection_uses_first_minimum_and_rejects_other_best_epoch(self):
        self.history[2]["tune_loss"] = 2.
        self._save_history()
        self.assertEqual(self._freeze()["best_epoch"], 2)
        self.decision["best_epoch"] = 3
        self._save_representation()
        with self.assertRaisesRegex(ValueError, "best_epoch"):
            freeze.verify_representation_bundle(self.output)

    def test_incomplete_nonfinite_and_out_of_order_histories_are_rejected(self):
        original = [dict(row) for row in self.history]
        cases = [[], original[:2], original + original[:1],
                 [original[0], original[0], original[2]], list(reversed(original))]
        for field, value in (("tune_loss", "nan"), ("tune_loss", "inf"),
                             ("fit_loss", "nan"), ("tune_loss", ""), ("epoch", "bad")):
            rows = [dict(row) for row in original]
            rows[1][field] = value
            cases.append(rows)
        for rows in cases:
            with self.subTest(rows=rows):
                self.history = rows
                self._save_history()
                with self.assertRaisesRegex(ValueError, "[Hh]istory"):
                    self._freeze()

    def test_decision_must_point_to_best_pt_with_its_hash(self):
        original = dict(self.decision)
        for field, value in (("best_epoch", 1), ("best_epoch", True),
                             ("best_checkpoint_sha256", "0" * 64),
                             ("best_checkpoint", str(self.repdir / "last.pt"))):
            with self.subTest(field=field, value=value):
                self.decision = dict(original, **{field: value})
                self._save_representation()
                with self.assertRaisesRegex(ValueError, "best.pt"):
                    self._freeze()

    def test_each_frozen_artifact_is_hashed_even_if_semantics_are_unchanged(self):
        bundle = self._freeze()
        for item in bundle["artifacts"]:
            with self.subTest(role=item["role"]):
                path = Path(item["path"])
                original = path.read_bytes()
                if item["role"] == "best_checkpoint":
                    path.write_bytes(b"checkpoint-b")  # Same byte count, different hash.
                else:
                    path.write_bytes(original + b"\n")
                try:
                    with self.assertRaises(ValueError):
                        freeze.verify_representation_bundle(self.output)
                finally:
                    path.write_bytes(original)

    def test_missing_artifact_is_rejected(self):
        self._freeze()
        (self.repdir / "best.pt").unlink()
        with self.assertRaises(FileNotFoundError):
            freeze.verify_representation_bundle(self.output)

    def test_changed_history_and_self_consistent_replacement_checkpoint_are_rejected(self):
        self._freeze()
        self.history[2]["tune_loss"] = 1.
        self._save_history()
        (self.repdir / "best.pt").write_bytes(b"replacement checkpoint")
        self.decision.update(best_epoch=3, best_checkpoint_sha256=file_hash(self.repdir / "best.pt"))
        self._save_representation()
        with self.assertRaisesRegex(ValueError, "freeze lineage changed"):
            freeze.verify_representation_bundle(self.output)

    def test_forged_bundle_metadata_inventory_and_canonical_fingerprint_are_rejected(self):
        original = self._freeze()
        cases = []
        for key, value in (("model", {"memory_slots": 99}), ("runtime", {}),
                           ("fit_videos", ["other"]), ("tune_videos", ["other"]),
                           ("pilot_videos", ["other"]), ("source_manifest_fingerprint", "wrong"),
                           ("pilot_contract_fingerprint", "wrong"),
                           ("representation_contract_fingerprint", "wrong"),
                           ("code_sha256", {}), ("artifacts", original["artifacts"][:-1]),
                           ("pilot_dir", "relative-pilot")):
            cases.append(dict(original, **{key: value}))
        cases.append(dict(original, artifacts=original["artifacts"] + original["artifacts"][:1]))
        for changed in cases:
            with self.subTest(changed=changed):
                self._seal(changed)
                self._write(self.output, changed)
                with self.assertRaises(ValueError):
                    freeze.verify_representation_bundle(self.output)
        self._write(self.output, dict(original, best_epoch=1))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            freeze.verify_representation_bundle(self.output)

    def test_source_code_and_original_pilot_code_are_revalidated(self):
        self._freeze()
        def changed_hash(path):
            return "0" * 64 if Path(path).name == "stage18_tracking_loss.py" else file_hash(path)
        with patch.object(freeze, "file_hash", side_effect=changed_hash):
            with self.assertRaisesRegex(ValueError, "source code changed"):
                freeze.verify_representation_bundle(self.output)
        with patch.object(train, "file_hash", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "Pilot-bound execution code changed"):
                freeze.verify_representation_bundle(self.output)

    def test_cli_creation_and_verification_work_with_torch_imports_blocked(self):
        script = """
import importlib.abc
import sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Filesystem freeze must not import torch')
sys.meta_path.insert(0, NoTorch())
import stage18_freeze_representation as freeze
assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
freeze.main(['--help'])
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=str(Path(__file__).resolve().parent), text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--pilot_dir", result.stdout)
        with patch.dict(sys.modules, {"torch": None}), contextlib.redirect_stdout(io.StringIO()):
            freeze.main(["--representation_dir", str(self.repdir), "--pilot_dir", str(self.pilotdir),
                         "--output", str(self.output)])
            freeze.main(["--verify_bundle", str(self.output)])
        freeze.verify_representation_bundle(self.output)


if __name__ == "__main__":
    unittest.main()
