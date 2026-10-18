"""CPU-only tests for the frozen Stage-11 external validation."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from stage11_external_summary import build_contrasts, primary_decision
from stage11_external_validate import (
    FROZEN_POLICIES,
    audit_otb_sequences,
    expected_candidate_blocks,
)


class ExternalValidationContractTests(unittest.TestCase):
    def test_frozen_periodic_costs(self):
        self.assertEqual(expected_candidate_blocks(99, "fast"), 0)
        self.assertEqual(expected_candidate_blocks(99, "periodic_l9_20"), 19)
        self.assertEqual(expected_candidate_blocks(99, "periodic_l9_30"), 29)
        self.assertEqual(expected_candidate_blocks(99, "sgla"), 99)
        with self.assertRaisesRegex(ValueError, "not frozen"):
            expected_candidate_blocks(99, "periodic_l9_10")

    def test_dataset_audit_fingerprints_complete_sequences(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequences = []
            for sequence_index in range(2):
                directory = root / "sequence{}".format(sequence_index)
                directory.mkdir()
                frames = []
                for frame_index in range(3):
                    path = directory / "{:04d}.jpg".format(frame_index + 1)
                    path.write_bytes(b"frame")
                    frames.append(str(path))
                sequences.append(
                    SimpleNamespace(
                        name="sequence{}".format(sequence_index),
                        frames=frames,
                        ground_truth_rect=np.ones((3, 4), dtype=np.float64),
                    )
                )
            report = audit_otb_sequences(sequences, root, expected_count=2)
            self.assertEqual(report["sequence_objects"], 2)
            self.assertEqual(report["physical_videos"], 2)
            self.assertEqual(report["tracked_frames"], 4)
            self.assertEqual(len(report["fingerprint"]), 64)

    def test_dataset_audit_rejects_missing_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing.jpg"
            sequence = SimpleNamespace(
                name="missing",
                frames=[str(missing), str(missing)],
                ground_truth_rect=np.ones((2, 4), dtype=np.float64),
            )
            with self.assertRaisesRegex(FileNotFoundError, "incomplete"):
                audit_otb_sequences([sequence], root, expected_count=1)


class ExternalDecisionTests(unittest.TestCase):
    @staticmethod
    def _policies(p30_latency=4.0, sgla_latency=4.5):
        return [
            {
                "policy": policy,
                "candidate_blocks_per_frame": {
                    "fast": 0.0,
                    "periodic_l9_20": 0.2,
                    "periodic_l9_30": 0.3,
                    "sgla": 1.0,
                }[policy],
                "mean_latency_ms": {
                    "fast": 3.8,
                    "periodic_l9_20": 3.9,
                    "periodic_l9_30": p30_latency,
                    "sgla": sgla_latency,
                }[policy],
            }
            for policy in FROZEN_POLICIES
        ]

    @staticmethod
    def _contrasts(headroom=1, superiority=1, noninferiority=1):
        return [
            {
                "contrast": "sgla_minus_fast",
                "passes_predeclared_threshold": headroom,
            },
            {
                "contrast": "p30_minus_fast",
                "passes_predeclared_threshold": superiority,
            },
            {
                "contrast": "p30_minus_sgla",
                "passes_predeclared_threshold": noninferiority,
            },
        ]

    def test_pass_requires_accuracy_cost_and_latency(self):
        decision = primary_decision(self._policies(), self._contrasts())
        self.assertEqual(decision["status"], "PASS")
        slow = primary_decision(
            self._policies(p30_latency=4.6), self._contrasts()
        )
        self.assertEqual(slow["status"], "FAIL_TRANSFER")

    def test_no_confirmed_headroom_is_not_schedule_search(self):
        decision = primary_decision(
            self._policies(),
            self._contrasts(headroom=0, superiority=0, noninferiority=1),
        )
        self.assertEqual(decision["status"], "NO_CONFIRMED_HEADROOM")
        self.assertIn("Do not tune", decision["interpretation"])

    def test_paired_ci_clusters_shared_physical_videos(self):
        names = ["a_1", "a_2", "b"]
        sequence_auc = {
            "fast": {"a_1": 50.0, "a_2": 50.0, "b": 50.0},
            "periodic_l9_20": {"a_1": 50.5, "a_2": 50.5, "b": 50.5},
            "periodic_l9_30": {"a_1": 51.0, "a_2": 51.0, "b": 51.0},
            "sgla": {"a_1": 51.1, "a_2": 51.1, "b": 51.1},
        }
        clusters = {"a_1": "a", "a_2": "a", "b": "b"}
        rows = build_contrasts(
            sequence_auc, names, clusters, 100, 7
        )
        self.assertTrue(
            all(row["num_physical_video_clusters"] == 2 for row in rows)
        )
        p30_fast = next(
            row for row in rows if row["contrast"] == "p30_minus_fast"
        )
        self.assertAlmostEqual(p30_fast["ci_low"], 1.0)


if __name__ == "__main__":
    unittest.main()
