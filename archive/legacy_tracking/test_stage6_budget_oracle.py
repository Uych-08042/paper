"""CPU-only tests for the Stage-6 sequence-budget oracle."""

import unittest

import numpy as np

from stage6_budget_oracle import solve_multiple_choice


class BudgetOracleTests(unittest.TestCase):
    def test_allocates_limited_budget_to_highest_value_sequence(self):
        values = np.asarray([[10.0, 20.0], [10.0, 11.0]])
        costs = np.asarray([[0, 10], [0, 10]])

        solutions = solve_multiple_choice(values, costs, [0, 10, 20])

        np.testing.assert_array_equal(
            solutions[0]["policy_indices"], [0, 0]
        )
        np.testing.assert_array_equal(
            solutions[10]["policy_indices"], [1, 0]
        )
        np.testing.assert_array_equal(
            solutions[20]["policy_indices"], [1, 1]
        )
        self.assertAlmostEqual(solutions[10]["total_value"] / 2, 15.0)
        self.assertAlmostEqual(solutions[20]["total_value"] / 2, 15.5)

    def test_equal_accuracy_prefers_lower_total_cost(self):
        values = np.asarray([[10.0, 10.0], [20.0, 20.0]])
        costs = np.asarray([[0, 5], [0, 5]])

        solution = solve_multiple_choice(values, costs, [10])[10]

        self.assertEqual(solution["total_cost"], 0)
        np.testing.assert_array_equal(solution["policy_indices"], [0, 0])

    def test_rejects_negative_cost(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            solve_multiple_choice([[1.0]], [[-1]], [0])


if __name__ == "__main__":
    unittest.main()
