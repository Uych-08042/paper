"""Reproduce float32/float64 feature replay drift and preserve frozen Stage-31 data."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_cache, write_json
from stage27_uav_common import checked_image, seal
from stage31_identity_common import DESIGN, FEATURES, METHODS, IdentityState, code_hashes, ground_truth, select_candidate
import stage31_identity_engine as engine
import stage31_identity_fit as fit_module
from stage31_precision_compat import precision_replay_auditor, replay_view
from test_stage27_uav_diagnostic import fake_image
from test_stage31_identity import FakeObserver


HERE = Path(__file__).resolve().parent


class Float32Observer(FakeObserver):
    def observe(self, image, features):
        packet = super().observe(image, features)
        packet["scores"] = packet["scores"].astype(np.float32)
        packet["raw_score_peak"] = float(np.float32(packet["raw_score_peak"]))
        return packet


def contract_fixture(root):
    dataset = root / "dataset"
    (dataset / "annotations").mkdir(parents=True)
    entries = []
    for video in range(14):
        name = "uav{:07d}_00000_s".format(video + 1)
        directory = dataset / "sequences" / name
        directory.mkdir(parents=True)
        frames = []
        for i in range(25):
            path = directory / ("img{:07d}.jpg".format(i + 1))
            path.write_bytes(bytes([i + 1]) * 12)
            frames.append(dict(file=path.relative_to(dataset).as_posix(), sha256=file_hash(path), height=64, width=96))
        annotation = dataset / "annotations" / (name + ".txt")
        annotation.write_text("\n".join(["40,20,10,10"] * len(frames)))
        entries.append(dict(sequence=name, source_group_proxy=name.split("_")[0], split="fit" if video < 8 else "checking",
            frames=frames, annotation=annotation.relative_to(dataset).as_posix(), annotation_sha256=file_hash(annotation),
            initial_box=[40, 20, 10, 10]))
    evidence = root / "synthetic_evidence" / "analysis" / "stage30_response_diagnostic.json"
    write_json(evidence, {"role": "synthetic_precision_test_fixture_only"})
    contract = seal(dict(schema="stage31_continuous_identity_pilot_v1", role="frozen_fresh_group_identity_pilot",
        design=DESIGN, root=str(dataset), sequences=entries, excluded_groups=[],
        runtime=dict(embed_dim=2), search_factor=4., search_size=256, code_sha256=code_hashes(),
        source_code_sha256={"stage4_runtime.py": file_hash(HERE / "stage4_runtime.py")},
        evidence_input=str(evidence.parent.parent), evidence_sha256=file_hash(evidence)))
    directory = root / "stage31"
    write_json(directory / "manifest.json", contract)
    reader = lambda directory, frame: checked_image(directory, frame, reader=fake_image)
    return directory, contract, reader


class PrecisionTests(unittest.TestCase):
    def test_reproduces_only_relative_response_drift(self):
        state = IdentityState([40, 20, 10, 10], [1, 0])
        boxes, desc = np.array([[40., 20, 10, 10], [20., 20, 10, 10]]), np.eye(2)
        scores = np.array([.5675574, .04159], np.float32)
        live = state.features(boxes, desc, scores)
        replay = state.features(boxes, desc, scores.astype(np.float64))
        differences = np.max(np.abs(live - replay), axis=0)
        self.assertGreater(differences[7], 1e-9)
        np.testing.assert_array_equal(np.delete(differences, 7), 0)

    def test_exact_dtype_restoration_not_rounding_or_tolerance_relaxation(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            _, contract, reader = contract_fixture(Path(temporary))
            entry = contract["sequences"][0]
            data = engine.run_video(Float32Observer(contract["runtime"]), contract, entry, "sgla", fit_collect=True, image_reader=reader)
            with self.assertRaises(AssertionError):
                engine.audit_video(data, contract, entry, "sgla", fit_collect=True)
            before = {k: v.tobytes() for k, v in data.items()}
            old_engine, old_fit = engine.audit_video, fit_module.audit_video
            with precision_replay_auditor() as stats:
                engine.audit_video(data, contract, entry, "sgla", fit_collect=True)
                fit_module.audit_video(data, contract, entry, "sgla", fit_collect=True)
                self.assertGreater(stats["max_float64_relative_response_delta"], 1e-9)
                self.assertEqual(stats["max_float32_relative_response_delta"], 0)
            self.assertIs(engine.audit_video, old_engine)
            self.assertIs(fit_module.audit_video, old_fit)
            self.assertEqual(before, {k: v.tobytes() for k, v in data.items()})
            wrong = {k: v.copy() for k, v in data.items()}
            wrong["features"][0, 0, 5] += 1e-6
            with precision_replay_auditor(), self.assertRaises(AssertionError):
                engine.audit_video(wrong, contract, entry, "sgla", fit_collect=True)
            wrong = {k: v.copy() for k, v in data.items()}
            wrong["scores"][0, 0] += 1e-12
            with self.assertRaisesRegex(ValueError, "losslessly"):
                replay_view(wrong)

    def test_near_tie_policy_choice_uses_original_online_precision(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            _, contract, reader = contract_fixture(Path(temporary))
            entry = contract["sequences"][0]
            scores = np.array([.9, .8], np.float32)
            r32 = float(scores[1] / scores[0])
            r64 = float(scores.astype(float)[1] / scores.astype(float)[0])
            weights = np.zeros(len(FEATURES))
            weights[0], weights[7] = 1 - (r32 + r64) / 2, 1
            model = dict(weights=weights, feature_scale=np.ones(len(FEATURES)))
            data = engine.run_video(Float32Observer(contract["runtime"]), contract, entry, "identity", model, image_reader=reader)
            state = IdentityState(entry["initial_box"], data["anchor"])
            replay64 = state.features(data["candidate_boxes"][0, :2], data["descriptors"][0, :2], data["scores"][0, :2])
            self.assertNotEqual(select_candidate("identity", replay64, model), int(data["chosen"][0]))
            with precision_replay_auditor():
                engine.audit_video(data, contract, entry, "identity", model)


class FrozenWorkflowTests(unittest.TestCase):
    def test_training_and_summary_keep_existing_contract_and_cache_bytes(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            directory, contract, reader = contract_fixture(Path(temporary))
            run = seal(dict(schema=contract["schema"], role="native_fit_candidate_features", contract_fingerprint=contract["fingerprint"]))
            write_json(directory / "fit_run.json", run)
            with precision_replay_auditor():
                for entry in contract["sequences"]:
                    if entry["split"] == "fit":
                        data = engine.run_video(Float32Observer(contract["runtime"]), contract, entry, "sgla", fit_collect=True, image_reader=reader)
                        engine.audit_video(data, contract, entry, "sgla", fit_collect=True)
                        write_cache(directory / "fit_cache" / (entry["sequence"] + ".npz"), data, engine.cache_metadata(run, entry, "sgla"))
            before_manifest = file_hash(directory / "manifest.json")
            before = {p.name: file_hash(p) for p in (directory / "fit_cache").glob("*.npz")}
            cmd = [sys.executable, "-B", str(HERE / "stage31_precision_compat.py"), "train", "--input", str(directory)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(before_manifest, file_hash(directory / "manifest.json"))
            self.assertEqual(before, {p.name: file_hash(p) for p in (directory / "fit_cache").glob("*.npz")})
            record = json.loads((directory / "stage31_precision_compat.json").read_text())
            self.assertEqual(record["original_feature_atol"], 1e-9)
            self.assertEqual(record["contract_fingerprint"], contract["fingerprint"])
            model = fit_module.load_model(directory, contract)
            checking_run = seal(dict(schema=contract["schema"], role="frozen_full_video_identity_checking",
                contract_fingerprint=contract["fingerprint"], model_fingerprint=model["fingerprint"], methods=list(METHODS)))
            write_json(directory / "checking_run.json", checking_run)
            with precision_replay_auditor():
                for entry in contract["sequences"]:
                    if entry["split"] == "checking":
                        for method in METHODS:
                            data = engine.run_video(Float32Observer(contract["runtime"]), contract, entry, method, model, image_reader=reader)
                            engine.audit_video(data, contract, entry, method, model)
                            write_cache(directory / "checking" / (entry["sequence"] + "__" + method + ".npz"), data,
                                        engine.cache_metadata(checking_run, entry, method))
            result = subprocess.run([sys.executable, "-B", str(HERE / "stage31_precision_compat.py"), "summary", "--input", str(directory)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((directory / "analysis" / "stage31_decision.json").read_text())
            self.assertEqual(report["checking_videos"], 6)
            audit = json.loads((directory / "precision_compat" / "summary_audit.json").read_text())
            self.assertEqual(audit["audited_arrays"], 30)
            self.assertEqual(audit["max_float32_relative_response_delta"], 0)
            self.assertEqual(before_manifest, file_hash(directory / "manifest.json"))


if __name__ == "__main__":
    unittest.main()
