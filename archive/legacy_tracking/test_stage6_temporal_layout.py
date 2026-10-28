"""CPU-only tests for equal-cost temporal placement helpers."""

import unittest

from stage6_temporal_layout import (
    active_policy,
    candidate_blocks,
    default_early_fraction,
    expected_candidate_blocks,
    split_frames,
    split_frames_for_layout,
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
        self.assertEqual(active_policy(8, 7, "late_p20"), "periodic_l9_20")
        self.assertEqual(active_policy(11, 10, "late_p30"), "periodic_l9_30")
        self.assertEqual(active_policy(10, 9, "late_p50"), "periodic_l9_50")
        self.assertEqual(active_policy(1, 5, "uniform_p5"), "periodic_l9_05")

    def test_default_boundaries_hold_nominal_budget_at_five_percent(self):
        expected = {
            "late_p10": 0.5,
            "late_p20": 0.75,
            "late_p30": 5.0 / 6.0,
            "late_p50": 0.9,
        }
        for layout, fraction in expected.items():
            self.assertAlmostEqual(default_early_fraction(layout), fraction)
            early_frames, _ = split_frames(600, fraction)
            self.assertEqual(
                expected_candidate_blocks(600, early_frames, layout), 30
            )

    def test_equal_cost_over_twenty_frames(self):
        for layout in ("early_p10", "late_p10", "uniform_p5"):
            self.assertEqual(expected_candidate_blocks(20, 10, layout), 1)

    def test_global_decision_index_is_preserved(self):
        self.assertEqual(candidate_blocks(10, 10, "early_p10"), 1)
        self.assertEqual(candidate_blocks(20, 10, "late_p10"), 1)
        self.assertEqual(candidate_blocks(20, 10, "uniform_p5"), 1)

    def test_exact_late_layouts_match_uniform_p5_per_sequence(self):
        exact_layouts = (
            "late_p10_exact",
            "late_p20_exact",
            "late_p30_exact",
            "late_p50_exact",
        )
        for tracked_frames in (79, 89, 99, 119, 3084):
            uniform_early, _ = split_frames(tracked_frames, 0.5)
            target = expected_candidate_blocks(
                tracked_frames, uniform_early, "uniform_p5"
            )
            for layout in exact_layouts:
                early_frames, late_frames = split_frames_for_layout(
                    tracked_frames,
                    default_early_fraction(layout),
                    layout,
                )
                self.assertEqual(early_frames + late_frames, tracked_frames)
                self.assertEqual(
                    expected_candidate_blocks(
                        tracked_frames, early_frames, layout
                    ),
                    target,
                )


if __name__ == "__main__":
    unittest.main()
