"""CPU-only tests for the Stage-13 long-horizon value screen."""

import unittest

import numpy as np

from stage13_long_horizon_value import (
    compare_objective_rankings,
    discounted_value_target,
    ranked_long_value_rows,
)


class LongHorizonValueTests(unittest.TestCase):
    def test_discounted_target_uses_only_valid_frames_and_coverage(self):
        delta = np.asarray(
            [
                [1.0, 0.0, -1.0, 1.0],
                [2.0, 2.0, 100.0, 100.0],
                [3.0, 3.0, 3.0, 3.0],
            ]
        )
        valid = np.asarray(
            [
                [True, True, True, True],
                [True, True, False, False],
                [False, True, True, True],
            ]
        )
        target, coverage, eligible = discounted_value_target(
            delta, valid, horizon=4, gamma=0.5, min_valid_fraction=0.75
        )
        expected = (1.0 + 0.5 * 0.0 + 0.25 * -1.0 + 0.125 * 1.0) / 1.875
        self.assertAlmostEqual(target[0], expected)
        self.assertTrue(np.isnan(target[1]))
        self.assertTrue(np.isnan(target[2]))
        np.testing.assert_allclose(coverage, [1.0, 0.5, 0.75])
        np.testing.assert_array_equal(eligible, [True, False, False])

    def test_ranked_budget_detects_predictive_long_value_ordering(self):
        sequences = np.repeat(np.asarray(["s0", "s1", "s2", "s3"]), 4)
        target = np.tile(np.asarray([0.08, 0.04, -0.03, -0.05]), 4)
        data = {"sequence": sequences, "discounted_value": target}
        rows = ranked_long_value_rows(
            "discounted_h20",
            "motion_only",
            target,
            data,
            budgets=(25.0, 50.0),
            bootstrap_samples=200,
            seed=7,
        )
        self.assertAlmostEqual(rows[0]["mean_discounted_value_gain"], 0.02)
        self.assertGreater(rows[0]["lift_over_random"], 0.0)
        self.assertEqual(rows[0]["selected_beneficial_ratio"], 1.0)
        self.assertGreater(rows[1]["mean_discounted_value_gain"], rows[0]["mean_discounted_value_gain"])

    def test_long_objective_comparison_is_paired_by_sequence(self):
        sequences = np.repeat(np.asarray(["s0", "s1", "s2", "s3"]), 4)
        target = np.tile(np.asarray([0.08, 0.04, -0.03, -0.05]), 4)
        data = {"sequence": sequences, "discounted_value": target}
        immediate_scores = -target
        long_scores = target
        rows = compare_objective_rankings(
            "motion_only",
            immediate_scores,
            long_scores,
            data,
            budgets=(25.0,),
            bootstrap_samples=200,
            seed=11,
            primary=True,
        )
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["is_primary"])
        self.assertGreater(rows[0]["long_minus_immediate"], 0.0)
        self.assertGreater(rows[0]["ci_low"], 0.0)
        self.assertEqual(rows[0]["selection_jaccard"], 0.0)


if __name__ == "__main__":
    unittest.main()
