"""CPU-only tests for the Stage-8C immediate intervention oracle."""

import unittest

from stage8_immediate_oracle import budget_oracle, effect_distribution


def synthetic_rows():
    values = {
        "seq_a": ((0.20, 20.0), (0.10, 10.0), (-0.10, -10.0), (0.0, 0.0)),
        "seq_b": ((0.05, 5.0), (-0.20, -20.0), (0.02, 2.0), (0.0, 0.0)),
    }
    rows = []
    for sequence, effects in values.items():
        for anchor_index, (iou_delta, auc_delta) in enumerate(effects):
            rows.append(
                {
                    "split": "val",
                    "sequence": sequence,
                    "anchor_index": anchor_index,
                    "anchor_fraction": (anchor_index + 1) / 5.0,
                    "iou_delta": iou_delta,
                    "auc_delta": auc_delta,
                }
            )
    return rows


class ImmediateOracleTests(unittest.TestCase):
    def test_distribution_separates_benefit_and_harm(self):
        row = effect_distribution(synthetic_rows())[0]
        self.assertAlmostEqual(row["always_l9_mean_iou_gain"], 0.00875)
        self.assertAlmostEqual(row["positive_cap_mean_iou_gain"], 0.04625)
        self.assertEqual(row["positive_ratio"], 0.5)
        self.assertEqual(row["negative_ratio"], 0.25)

    def test_budget_oracle_and_matched_random_use_same_cost(self):
        rows = budget_oracle(synthetic_rows(), (25, 100), 100, 7)
        by_key = {
            (row["oracle_mode"], row["budget_percent"]): row for row in rows
        }
        exact_25 = by_key[("exact_topk", 25.0)]
        self.assertEqual(exact_25["num_selected"], 2)
        self.assertAlmostEqual(exact_25["mean_iou_gain"], 0.0375)
        self.assertAlmostEqual(
            exact_25["random_matched_mean_iou_gain"], 0.0021875
        )
        self.assertGreater(exact_25["lift_ci_low"], 0.0)

        exact_100 = by_key[("exact_topk", 100.0)]
        capped_100 = by_key[("positive_cap", 100.0)]
        self.assertAlmostEqual(exact_100["mean_iou_gain"], 0.00875)
        self.assertEqual(capped_100["num_selected"], 4)
        self.assertAlmostEqual(capped_100["mean_iou_gain"], 0.04625)
        self.assertEqual(capped_100["selected_negative_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
