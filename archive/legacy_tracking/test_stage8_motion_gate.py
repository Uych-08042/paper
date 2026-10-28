"""CPU-only tests for the Stage-8E causal motion-history screen."""

import unittest

import numpy as np

from stage8_motion_gate import (
    MOTION_FEATURE_NAMES,
    build_variant_low_features,
    causal_motion_features,
    nested_group_oof,
    uses_motion,
)


def synthetic_data():
    sequences = np.repeat(np.asarray(["s{}".format(i) for i in range(8)]), 3)
    count = len(sequences)
    previous = np.column_stack(
        (
            np.linspace(10.0, 20.0, count),
            np.full(count, 12.0),
            np.full(count, 20.0),
            np.full(count, 16.0),
        )
    )
    selector = np.tile(
        np.asarray([[0.1, 0.1, 0.4, 0.15, 0.15, 0.1]]), (count, 1)
    )
    motion = np.zeros((count, len(MOTION_FEATURE_NAMES)), dtype=np.float32)
    motion[:, 1] = np.linspace(-1.0, 1.0, count)
    target = 0.02 * motion[:, 1] + 0.001 * (np.arange(count) % 3)
    return {
        "sequence": sequences,
        "token_features": np.zeros((count, 8), dtype=np.float32),
        "fast_score_map": np.ones((count, 1, 2, 2), dtype=np.float32),
        "selector_prob": selector,
        "selected_layer": np.full(count, 9, dtype=np.int64),
        "previous_state": previous,
        "previous_fast_box": previous.copy(),
        "previous_action": np.zeros(count, dtype=np.int64),
        "fast_box": previous.copy(),
        "image_hw": np.tile(np.asarray([[100, 160]]), (count, 1)),
        "resize_factor": np.ones(count),
        "frame_id": np.arange(count),
        "anchor_offset": np.tile(np.asarray([2, 4, 6]), 8),
        "motion_history": motion,
        "delta_iou": target.astype(np.float64),
        "delta_auc": (100.0 * target).astype(np.float64),
        "control_iou": np.full(count, 0.5),
        "treatment_iou": np.full(count, 0.5) + target,
    }


class MotionGateTests(unittest.TestCase):
    def test_motion_features_do_not_read_future_states(self):
        boxes = np.asarray(
            [[1.0 + index, 2.0, 10.0, 8.0] for index in range(20)]
        )
        anchors = np.asarray([2, 8, 16])
        initial = np.asarray([0.0, 2.0, 10.0, 8.0])
        original = causal_motion_features(boxes, anchors, initial)
        changed = boxes.copy()
        changed[2:] += np.asarray([100.0, 100.0, 5.0, 5.0])
        modified = causal_motion_features(changed, anchors, initial)
        np.testing.assert_allclose(original[0], modified[0])
        self.assertFalse(np.allclose(original[1], modified[1]))
        self.assertEqual(original.shape[1], len(MOTION_FEATURE_NAMES))

    def test_motion_variants_extend_only_the_low_features(self):
        data = synthetic_data()
        baseline, _ = build_variant_low_features(data, "history_only")
        motion, names = build_variant_low_features(data, "motion_only")
        self.assertTrue(uses_motion("token_selector_motion"))
        self.assertEqual(
            motion.shape[1], baseline.shape[1] + len(MOTION_FEATURE_NAMES)
        )
        self.assertEqual(motion.shape[1], len(names))

    def test_nested_oof_is_complete_for_motion_only_model(self):
        data = synthetic_data()
        result = nested_group_oof(
            data,
            ("motion_only",),
            outer_folds=4,
            inner_folds=3,
            alphas=(0.1, 1.0),
            pca_dimensions=2,
            pca_oversample=0,
            pca_power_iterations=0,
            seed=17,
            batch_size=16,
        )
        prediction = result["predictions"]["motion_only"]
        self.assertEqual(prediction.shape, data["delta_iou"].shape)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(len(result["outer_rows"]), 4)
        self.assertEqual(len(result["selected_alphas"]["motion_only"]), 4)


if __name__ == "__main__":
    unittest.main()
