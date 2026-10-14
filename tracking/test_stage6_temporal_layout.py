"""CPU-only tests for equal-cost temporal placement helpers."""

import unittest

from stage6_temporal_layout import (
    active_policy,
    candidate_blocks,
    expected_candidate_blocks,
    split_frames,
)


class TemporalLayoutTests(unittest.TestCase):
    def test_split_frames_matches_stage6_boundary(self):
        self.assertEqual(split_frames(11, 0.5), (5, 6))
        self.assertEqual(split_frames(10, 0.5), (5, 5))

    def test_active_phase_policy(self):
        self.assertEqual(active_policy(5, 5, "early_p10"), "periodic_l9_10")
        self.assertEqual(active_policy(6, 5, "early_p10"), "fast")
        self.assertEqual(active_policy(5, 5, "late_p10"), "fast")
        self.assertEqual(active_policy(6, 5, "late_p10"), "periodic_l9_10")
        self.assertEqual(active_policy(1, 5, "uniform_p5"), "periodic_l9_05")

    def test_equal_cost_over_twenty_frames(self):
        for layout in ("early_p10", "late_p10", "uniform_p5"):
            self.assertEqual(expected_candidate_blocks(20, 10, layout), 1)

    def test_global_decision_index_is_preserved(self):
        self.assertEqual(candidate_blocks(10, 10, "early_p10"), 1)
        self.assertEqual(candidate_blocks(20, 10, "late_p10"), 1)
        self.assertEqual(candidate_blocks(20, 10, "uniform_p5"), 1)


if __name__ == "__main__":
    unittest.main()
