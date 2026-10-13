"""CPU-only tests for the Stage-6 temporal stability audit."""

import unittest

from stage6_temporal_stability import (
    bootstrap_mean,
    build_temporal_sequences,
)


class TemporalStabilityTests(unittest.TestCase):
    def test_split_excludes_initialization_and_preserves_block_counts(self):
        rows = []
        for policy, overlaps, blocks in (
            ("fast", [1.0, 0.1, 0.2, 0.3, 0.4], [0, 0, 0, 0, 0]),
            ("sgla", [1.0, 0.9, 0.8, 0.7, 0.6], [1, 1, 1, 1, 1]),
        ):
            for frame_id, (iou, block) in enumerate(zip(overlaps, blocks)):
                rows.append(
                    {
                        "dataset": "test",
                        "sequence": "video",
                        "policy": policy,
                        "frame_id": frame_id,
                        "iou": iou,
                        "candidate_blocks": block,
                    }
                )

        sequence = build_temporal_sequences(
            rows, ("fast", "sgla"), 0.5
        )[0]

        self.assertEqual(sequence["early_frames"], 2)
        self.assertEqual(sequence["late_frames"], 2)
        self.assertEqual(
            sequence["records"]["sgla"]["early_candidate_blocks"], 2
        )
        self.assertEqual(
            sequence["records"]["sgla"]["late_candidate_blocks"], 2
        )
        self.assertGreater(
            sequence["records"]["sgla"]["early_auc"],
            sequence["records"]["fast"]["early_auc"],
        )

    def test_bootstrap_is_reproducible(self):
        import numpy as np

        first = bootstrap_mean(
            [1.0, -1.0, 2.0], 100, np.random.default_rng(2026)
        )
        second = bootstrap_mean(
            [1.0, -1.0, 2.0], 100, np.random.default_rng(2026)
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
