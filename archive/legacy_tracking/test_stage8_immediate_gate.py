"""CPU-only tests for the Stage-8D immediate intervention gate."""

import unittest

import numpy as np

from stage8_immediate_gate import (
    build_variant_low_features,
    grouped_folds,
    ranked_budget_rows,
    uses_fast_output,
    uses_selector,
    uses_tokens,
)


def synthetic_feature_data(count=8):
    previous = np.tile(np.asarray([[10.0, 20.0, 30.0, 40.0]]), (count, 1))
    fast = previous + np.asarray([1.0, -1.0, 2.0, -2.0])
    probabilities = np.tile(
        np.asarray([[0.1, 0.1, 0.4, 0.15, 0.15, 0.1]]), (count, 1)
    )
    score = np.arange(count * 16, dtype=np.float32).reshape(count, 1, 4, 4)
    return {
        "previous_state": previous,
        "previous_fast_box": previous.copy(),
        "previous_action": np.zeros(count, dtype=np.int64),
        "fast_box": fast,
        "image_hw": np.tile(np.asarray([[100, 200]]), (count, 1)),
        "resize_factor": np.ones(count),
        "selector_prob": probabilities,
        "fast_score_map": score,
    }


class ImmediateGateTests(unittest.TestCase):
    def test_grouped_folds_never_split_a_video(self):
        sequences = np.asarray(
            ["a"] * 3 + ["b"] * 2 + ["c"] * 4 + ["d"] * 2 + ["e"] * 3
        )
        folds = grouped_folds(sequences, 3, 9)
        seen = np.zeros(len(sequences), dtype=np.int64)
        for train, test, names in folds:
            self.assertFalse(set(sequences[train]).intersection(sequences[test]))
            self.assertEqual(set(sequences[test]), set(names))
            seen[test] += 1
        np.testing.assert_array_equal(seen, np.ones(len(sequences), dtype=np.int64))

    def test_feature_variants_have_declared_runtime_requirements(self):
        data = synthetic_feature_data()
        dimensions = {}
        for variant in (
            "history_only",
            "selector_only",
            "token_only",
            "token_selector",
            "fast_only",
            "token_fast",
        ):
            features, names = build_variant_low_features(data, variant)
            self.assertEqual(features.shape[0], 8)
            self.assertEqual(features.shape[1], len(names))
            self.assertTrue(np.isfinite(features).all())
            dimensions[variant] = features.shape[1]
        self.assertGreater(dimensions["selector_only"], dimensions["history_only"])
        self.assertGreater(dimensions["fast_only"], dimensions["history_only"])
        self.assertTrue(uses_tokens("token_selector"))
        self.assertTrue(uses_selector("selector_only"))
        self.assertTrue(uses_fast_output("token_fast"))
        self.assertFalse(uses_fast_output("token_selector"))

    def test_ranked_budget_uses_scores_not_ground_truth(self):
        data = {
            "sequence": np.asarray(["a"] * 4 + ["b"] * 4),
            "delta_iou": np.asarray([0.4, 0.3, -0.2, 0.0, 0.2, -0.3, 0.1, 0.0]),
            "delta_auc": np.asarray([40.0, 30.0, -20.0, 0.0, 20.0, -30.0, 10.0, 0.0]),
        }
        scores = np.asarray([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        rows = ranked_budget_rows(
            "test", "history_only", scores, data, (25, 100), 200, 11, True
        )
        first = rows[0]
        self.assertEqual(first["num_selected"], 2)
        self.assertAlmostEqual(first["mean_iou_gain"], 0.0875)
        self.assertAlmostEqual(first["oracle_positive_cap_mean_iou_gain"], 0.0875)
        self.assertEqual(first["selected_negative_ratio"], 0.0)
        full = rows[1]
        self.assertAlmostEqual(full["mean_iou_gain"], 0.0625)
        self.assertAlmostEqual(full["lift_over_random_mean_iou"], 0.0)


if __name__ == "__main__":
    unittest.main()
