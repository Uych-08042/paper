"""CPU-only tests for the Stage-12 external failure audit."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import stage12_external_failure_audit as stage12
from stage12_external_failure_audit import (
    _average_ranks,
    _correlation,
    failure_statistics,
    normalized_time_aucs,
)


def _rows(overlaps):
    return [
        {
            "frame_id": index,
            "gt_valid": 1,
            "iou": float(overlap),
        }
        for index, overlap in enumerate([1.0] + list(overlaps))
    ]


class Stage12ExternalFailureAuditTests(unittest.TestCase):
    def test_failure_statistics_counts_entries_recoveries_and_longest_run(self):
        rows = _rows([0.8, 0.1, 0.0, 0.6, 0.1, 0.7])
        result = failure_statistics(rows, threshold=0.2)
        self.assertAlmostEqual(result["failure_frame_ratio"], 0.5)
        self.assertEqual(result["failure_entries"], 2)
        self.assertEqual(result["failure_recoveries"], 2)
        self.assertEqual(result["longest_failure_run"], 2)

    def test_normalized_time_aucs_cover_each_tracked_frame_once(self):
        rows = _rows([1.0] * 5 + [0.0] * 5)
        aucs = normalized_time_aucs(rows, bins=5)
        self.assertEqual(len(aucs), 5)
        self.assertGreater(aucs[0], 90.0)
        self.assertLess(aucs[-1], 5.0)

    def test_average_ranks_use_midrank_for_ties(self):
        np.testing.assert_array_equal(
            _average_ranks([2.0, 1.0, 2.0, 4.0]),
            np.asarray([1.5, 0.0, 1.5, 3.0]),
        )

    def test_rank_correlation_detects_monotonic_alignment(self):
        self.assertAlmostEqual(
            _correlation([1.0, 2.0, 3.0], [10.0, 20.0, 40.0], rank=True),
            1.0,
        )

    def test_main_writes_complete_descriptive_audit(self):
        names = ["sequence_{}".format(index) for index in range(6)]
        run = {
            "fingerprint": "run-fingerprint",
            "dataset_fingerprint": "dataset-fingerprint",
            "expected_sequences": {name: 11 for name in names},
            "sequence_clusters": {
                name: "video_{}".format(index // 2)
                for index, name in enumerate(names)
            },
        }
        policy_offsets = {
            "fast": [0.00] * 6,
            "periodic_l9_20": [0.00, 0.01, 0.00, 0.02, 0.01, -0.01],
            "periodic_l9_30": [0.01, 0.00, 0.03, -0.01, 0.05, 0.00],
            "sgla": [0.04, -0.02, 0.08, 0.01, 0.10, -0.01],
        }
        grouped = {}
        for sequence_index, name in enumerate(names):
            for policy in stage12.FROZEN_POLICIES:
                rows = []
                for frame_id in range(11):
                    overlap = 1.0 if frame_id == 0 else np.clip(
                        0.25
                        + 0.08 * sequence_index
                        + 0.02 * frame_id
                        + policy_offsets[policy][sequence_index],
                        0.0,
                        1.0,
                    )
                    rows.append(
                        {
                            "frame_id": frame_id,
                            "gt_valid": 1,
                            "iou": float(overlap),
                        }
                    )
                grouped[(stage12.DATASET_NAME, name, policy)] = rows

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "audit"
            argv = [
                "stage12_external_failure_audit.py",
                "--input",
                str(root),
                "--output_dir",
                str(output),
                "--bootstrap_samples",
                "20",
                "--seed",
                "7",
            ]
            with patch.object(sys, "argv", argv), patch.object(
                stage12, "load_contract", return_value=(root, run)
            ), patch.object(
                stage12,
                "load_failed_decision",
                return_value=(root / "decision.json", {"status": "FAIL_TRANSFER"}),
            ), patch.object(
                stage12, "load_closed_loop", return_value=(root, [], [])
            ), patch.object(stage12, "audit_rows", return_value=grouped):
                stage12.main()

            expected = {
                "stage12_sequence_mechanism.csv",
                "stage12_subgroup_summary.csv",
                "stage12_temporal_summary.csv",
                "stage12_failure_summary.csv",
                "stage12_mechanism.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            mechanism = json.loads(
                (output / "stage12_mechanism.json").read_text(encoding="utf-8")
            )
            self.assertEqual(mechanism["stage11_status"], "FAIL_TRANSFER")
            self.assertEqual(mechanism["num_sequence_objects"], 6)
            self.assertEqual(mechanism["num_physical_video_clusters"], 3)
            self.assertFalse(mechanism["policy_search_permitted"])


if __name__ == "__main__":
    unittest.main()
