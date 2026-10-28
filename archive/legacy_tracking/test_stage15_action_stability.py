"""CPU-only tests for the Stage-15 disjoint-window stability audit."""

import unittest
from types import SimpleNamespace

import numpy as np

from stage15_action_stability import (
    build_outcome_windows,
    discounted_window_values,
    horizon_agreement_rows,
    horizon_summary_rows,
    make_decision,
    transfer_summary_rows,
)


def synthetic_data(stable=True):
    sequences = np.repeat(
        ["video_{:02d}".format(index) for index in range(8)], 2
    )
    count = len(sequences)
    delta = np.zeros((count, 6, 20), dtype=np.float64)
    delta[:, 2, :] = 0.01  # Native L9.
    delta[:, 1, :10] = 0.04
    delta[:, 1, 10:] = 0.04 if stable else -0.02
    return {
        "sequence": sequences,
        "split": np.repeat(["train"] * 4 + ["val"] * 4, 2),
        "anchor_offset": np.tile([1, 2], 8),
        "selected_layer": np.full(count, 9, dtype=np.int64),
        "delta_iou": delta,
        "valid_gt": np.ones((count, 20), dtype=bool),
    }


class DiscountedWindowTests(unittest.TestCase):
    def test_window_masks_invalid_outcomes_and_preserves_action_gain(self):
        delta = np.zeros((2, 6, 4), dtype=np.float64)
        delta[:, 0] = 0.2
        valid = np.asarray(
            [[True, True, True, True], [True, False, False, False]],
            dtype=bool,
        )
        values, eligible, coverage = discounted_window_values(
            delta,
            valid,
            [0, 1, 2, 3],
            gamma=0.9,
            min_valid_fraction=0.5,
        )
        self.assertTrue(eligible[0])
        self.assertFalse(eligible[1])
        self.assertAlmostEqual(values[0, 0], 0.2)
        self.assertTrue(np.isnan(values[1]).all())
        np.testing.assert_allclose(coverage, [1.0, 0.25])


class StabilitySummaryTests(unittest.TestCase):
    def _run(self, stable):
        data = synthetic_data(stable)
        run = {
            "horizon": 20,
            "gamma": 0.9,
            "fingerprint": "source",
        }
        windows = build_outcome_windows(data, run, 10, 0.8)
        transfers = transfer_summary_rows(data, windows, 100, 15)
        horizons, values = horizon_summary_rows(
            data, run, (1, 5, 10, 20), 0.8, 100, 16
        )
        agreements = horizon_agreement_rows(data, values, 100, 17)
        return data, run, transfers, horizons, agreements

    def test_stable_early_action_transfers_to_disjoint_late_window(self):
        data, run, transfers, horizons, agreements = self._run(True)
        primary = next(
            row
            for row in transfers
            if row["scope"] == "val" and row["transfer"] == "early_to_late"
        )
        self.assertAlmostEqual(primary["selected_minus_native"], 0.03)
        self.assertGreater(primary["ci_low"], 0.0)
        self.assertEqual(primary["selected_action_distribution"], "l7:0|l8:8|l9:0|l10:0|l11:0|l12:0")
        self.assertEqual(len(horizons), 12)
        self.assertEqual(len(agreements), 18)

        args = SimpleNamespace(split_point=10)
        decision = make_decision(run, transfers, horizons, args, data["sequence"])
        self.assertEqual(decision["status"], "PASS_STABILITY_SCREEN")

    def test_transient_early_winner_fails_late_transfer(self):
        data, run, transfers, horizons, _ = self._run(False)
        primary = next(
            row
            for row in transfers
            if row["scope"] == "val" and row["transfer"] == "early_to_late"
        )
        self.assertAlmostEqual(primary["selected_minus_native"], -0.03)
        self.assertLess(primary["ci_high"], 0.0)
        args = SimpleNamespace(split_point=10)
        decision = make_decision(run, transfers, horizons, args, data["sequence"])
        self.assertEqual(decision["status"], "FAIL_STABILITY_SCREEN")


if __name__ == "__main__":
    unittest.main()
