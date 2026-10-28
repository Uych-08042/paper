"""CPU-only tests for the Stage-6 causal sequence router."""

import copy
import unittest

import numpy as np

from stage6_causal_router import (
    FEATURE_NAMES,
    analyze_fraction,
    fit_oof_router,
    trajectory_features,
)


def _rows(scale=1.0, offset_x=0.0, offset_y=0.0):
    rows = []
    boxes = (
        (10.0, 20.0, 20.0, 10.0),
        (12.0, 21.0, 21.0, 10.5),
        (15.0, 23.0, 20.0, 11.0),
        (17.0, 22.0, 22.0, 10.0),
    )
    for frame_id, box in enumerate(boxes):
        rows.append(
            {
                "dataset": "test",
                "sequence": "video",
                "policy": "fast",
                "frame_id": frame_id,
                "iou": 0.5,
                "bbox_x": scale * box[0] + offset_x,
                "bbox_y": scale * box[1] + offset_y,
                "bbox_w": scale * box[2],
                "bbox_h": scale * box[3],
            }
        )
    return rows


class CausalRouterTests(unittest.TestCase):
    def test_features_are_translation_and_scale_invariant(self):
        first = trajectory_features(_rows(), 3)
        transformed = trajectory_features(
            _rows(scale=2.5, offset_x=100.0, offset_y=-30.0), 3
        )
        self.assertEqual(len(first), len(FEATURE_NAMES))
        np.testing.assert_allclose(first, transformed, atol=1e-12)

    def test_features_do_not_read_overlap_labels(self):
        rows = _rows()
        changed = copy.deepcopy(rows)
        for index, row in enumerate(changed):
            row["iou"] = -1000.0 + index
            row["gt_x"] = 1e9
        np.testing.assert_array_equal(
            trajectory_features(rows, 3), trajectory_features(changed, 3)
        )

    def test_oof_ridge_learns_heldout_linear_signal(self):
        x = np.linspace(-2.0, 2.0, 40)[:, None]
        features = np.concatenate([x, x ** 2], axis=1)
        utilities = np.concatenate(
            [np.zeros_like(x), 1.5 * x + 0.2, -0.5 * x - 0.1], axis=1
        )
        predictions, _, fold_ids, folds, alpha_rows = fit_oof_router(
            features,
            utilities,
            fast_index=0,
            folds=5,
            inner_folds=4,
            alphas=(0.001, 0.01, 0.1, 1.0),
            seed=2026,
        )
        self.assertEqual(set(fold_ids.tolist()), set(range(5)))
        self.assertEqual(len(folds), 5)
        self.assertEqual(sum(row["selected"] for row in alpha_rows), 5)
        correlation = np.corrcoef(
            predictions[:, 1:].reshape(-1), utilities[:, 1:].reshape(-1)
        )[0, 1]
        self.assertGreater(correlation, 0.98)
        np.testing.assert_array_equal(predictions[:, 0], 0.0)

    def test_end_to_end_analysis_respects_budget(self):
        sequences = []
        for index in range(18):
            feature = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
            feature[0] = index / 17.0
            gain = 4.0 * feature[0] - 2.0
            sequences.append(
                {
                    "dataset": "test",
                    "sequence": "video_{:02d}".format(index),
                    "early_frames": 10,
                    "late_frames": 10,
                    "features": feature,
                    "records": {
                        "fast": {
                            "early_auc": 50.0,
                            "late_auc": 50.0,
                            "late_candidate_blocks": 0,
                        },
                        "sgla": {
                            "early_auc": 50.0 + gain,
                            "late_auc": 50.0 + gain,
                            "late_candidate_blocks": 10,
                        },
                    },
                }
            )
        results = analyze_fraction(
            sequences=sequences,
            policies=("fast", "sgla"),
            fraction=0.5,
            budgets=(50.0,),
            folds=3,
            inner_folds=3,
            alphas=(0.01, 0.1, 1.0),
            bootstrap_samples=100,
            rng=np.random.default_rng(2026),
            seed=2026,
        )
        summary = results[0][0]
        self.assertLessEqual(summary["router_candidate_blocks"], 90)
        self.assertGreater(summary["router_auc"], summary["best_fixed_auc"])
        self.assertEqual(len(results[1]), 18)
        self.assertEqual(len(results[2]), 36)


if __name__ == "__main__":
    unittest.main()
