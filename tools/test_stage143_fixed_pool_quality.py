"""Protect frozen candidate identity and source-separated scoring."""
import unittest

import numpy as np

import stage143_fixed_pool_quality as stage


def fixture(truth):
    raw = np.full((4, 4), .01, np.float32)
    raw[2, 2], raw[1, 1], raw[0, 3], raw[3, 0] = 1., .95, .7, .6
    size = np.full((2, 4, 4), .25, np.float32)
    parent = np.array([40., 40., 16., 16.])
    ap = dict(window=np.ones_like(raw), search_size=64, design={"box_atol": 1e-6})
    return raw, size, np.zeros_like(size), parent, 1., (128, 128), parent, np.asarray(truth), ap


class FixedPoolQuality(unittest.TestCase):
    def test_stage142_candidate_order_survives_both_policies(self):
        args = fixture([24., 24., 16., 16.])
        row = stage.extract(*args)
        stage.check_frame(row, {k: str(v) for k, v in stage.prior.compare(*args).items()})
        scores = np.ones(len(row["cells"]))
        decisions = stage.decide(row, scores)
        self.assertEqual(decisions["dense_oracle_iou"], 1.)
        self.assertEqual(decisions["peak_oracle_iou"], 0.)
        for pool in stage.POOLS:
            self.assertEqual(decisions[f"{pool}_hann_cell"], decisions[f"{pool}_quality_cell"])

    def test_held_annotations_do_not_change_model_or_action(self):
        original = stage.extract(*fixture([24., 24., 16., 16.]))
        different_label = stage.extract(*fixture([40., 40., 16., 16.]))
        bank = {"a": stage.moments([original]), "b": stage.moments([different_label])}
        first = stage.fit_peer_sources(bank, "b")
        bank["b"] = stage.moments([original])
        second = stage.fit_peer_sources(bank, "b")
        self.assertEqual(first, second)
        self.assertEqual(first["fit_sources"], ["a"])
        np.testing.assert_array_equal(original["x"], different_label["x"])
        a, b = [stage.decide(row, stage.score(first, row["x"])) for row in (original, different_label)]
        self.assertEqual(a["dense_quality_cell"], b["dense_quality_cell"])
        self.assertNotEqual(a["dense_oracle_cell"], b["dense_oracle_cell"])

    def test_severe_window_guard_counts_consecutive_misses(self):
        rows = []
        for j in range(32):
            row = dict(frame=j+1, sequence="uav0001_00000_s", source="uav0001", native_iou=.7, stable_window=0)
            row.update(peak_raw_iou=.7, dense_raw_iou=.7, peak_quality_iou=.1 if j in (2,4,6) else .7,
                       dense_quality_iou=.1 if 2 <= j <= 4 else .7)
            rows.append(row)
        stage.mark_stable(rows)
        report, = stage.stable_report(rows)
        self.assertEqual(report["dense_quality_low_frames"], 3)
        self.assertTrue(report["dense_quality_low_run"])
        self.assertFalse(report["peak_quality_low_run"])


if __name__ == "__main__":
    unittest.main()
