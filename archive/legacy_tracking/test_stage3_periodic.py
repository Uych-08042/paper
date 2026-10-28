"""CPU-only tests for periodic L9 closed-loop validation helpers."""

import math
import unittest

from stage3_closed_loop_summary import summarize
from stage3_periodic import PERIODIC_L9_BUDGETS, periodic_l9_active


class PeriodicPolicyTests(unittest.TestCase):
    def test_each_prefix_uses_exact_floor_budget(self):
        for policy, budget in PERIODIC_L9_BUDGETS.items():
            used = 0
            for decision_index in range(1, 102):
                used += int(periodic_l9_active(policy, decision_index))
                self.assertEqual(
                    used,
                    int(math.floor(decision_index * budget / 100.0 + 1e-12)),
                )
            self.assertFalse(periodic_l9_active(policy, 0))

    def test_summary_reports_average_blocks_per_tracked_frame(self):
        rows = []
        for policy in ("fast", "periodic_l9_30"):
            for frame_id in range(11):
                rows.append({
                    "dataset": "test",
                    "sequence": "sequence",
                    "policy": policy,
                    "frame_id": frame_id,
                    "gt_valid": 1,
                    "iou": 1.0,
                    "candidate_blocks": (
                        0 if policy == "fast" or frame_id == 0
                        else int(periodic_l9_active(policy, frame_id))
                    ),
                    "model_decode_latency_ms": (
                        None if frame_id == 0 else 4.0
                    ),
                })
        summaries = {row["policy"]: row for row in summarize(rows)}
        self.assertEqual(summaries["fast"]["candidate_blocks"], 0.0)
        self.assertEqual(
            summaries["periodic_l9_30"]["candidate_blocks"], 0.3)


if __name__ == "__main__":
    unittest.main()

