"""Check non-peak recovery, exact candidate budgets and per-video reporting."""
import unittest

import numpy as np

import stage142_dense_candidate_recall as stage


class DenseCandidateRecall(unittest.TestCase):
    def fixture(self, truth):
        raw = np.full((4, 4), .01, np.float32)
        raw[2, 2], raw[1, 1], raw[0, 3], raw[3, 0] = 1., .95, .7, .6
        size = np.full((2, 4, 4), .25, np.float32)
        parent = np.array([40., 40., 16., 16.])
        ap = dict(window=np.ones_like(raw), search_size=64, design={"box_atol": 1e-6})
        return stage.compare(raw, size, np.zeros_like(size), parent, 1., (128, 128), parent, np.asarray(truth), ap)

    def test_dense_adds_non_peak_inside_same_budget(self):
        result = self.fixture([24., 24., 16., 16.])
        self.assertEqual(result["native_iou"], 0.)
        self.assertEqual(result["matched_peak_best_iou"], 0.)
        self.assertEqual(result["matched_dense_best_iou"], 1.)
        self.assertTrue(result["dense_contains_best_cell"])
        self.assertFalse(result["peak_contains_best_cell"])
        for k in ("peak_cells", "dense_cells"):
            self.assertEqual(len(result[k].split("|")), result["matched_budget"])
            self.assertEqual(result[k].split("|")[0], "10")

    def test_report_counts_gains_and_losses_separately(self):
        won = dict(self.fixture([24., 24., 16., 16.]), sequence="win")
        lost = dict(won, sequence="loss", matched_peak_best_iou=1., matched_dense_best_iou=0.)
        kept = dict(self.fixture([40., 40., 16., 16.]), sequence="base")
        summary, videos = stage.summarize([won, lost, kept], [won, lost], [won])
        cohorts = {r["group"]: r for r in summary}
        focus = cohorts["native_low_grid_good"]
        self.assertEqual((focus["dense_added_good"], focus["dense_lost_good"], focus["frames"]), (1, 1, 2))
        self.assertEqual(focus["video_mean_recall_delta"], 0.)
        self.assertEqual(cohorts["native_all"]["native_good"], 1)
        self.assertEqual(cohorts["stage139_branch_replays"]["video_mean_recall_delta"], 1.)
        self.assertTrue(any(r["group"] == "native_onsets" and r["sequence"] == "win" for r in videos))


if __name__ == "__main__":
    unittest.main()
