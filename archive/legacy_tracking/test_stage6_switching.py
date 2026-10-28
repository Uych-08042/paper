"""CPU-only tests for Stage-6 switched closed-loop helpers."""

import unittest

from stage6_switch_common import active_policy, select_assignments
from stage6_switched_summary import analyze


class SwitchingTests(unittest.TestCase):
    def test_active_policy_switches_after_complete_prefix(self):
        self.assertEqual(active_policy(0, 5, "sgla"), "fast")
        self.assertEqual(active_policy(1, 5, "sgla"), "fast")
        self.assertEqual(active_policy(5, 5, "sgla"), "fast")
        self.assertEqual(active_policy(6, 5, "sgla"), "sgla")

    def test_assignment_filter_is_exact_and_typed(self):
        rows = [
            {
                "early_fraction": "0.5",
                "budget_percent": "10",
                "dataset": "test",
                "sequence": "video",
                "early_frames": "5",
                "late_frames": "5",
                "router_policy": "periodic_l9_30",
                "router_candidate_blocks": "2",
                "fold": "3",
            },
            {
                "early_fraction": "0.25",
                "budget_percent": "10",
                "dataset": "test",
                "sequence": "video",
                "early_frames": "2",
                "late_frames": "8",
                "router_policy": "fast",
                "router_candidate_blocks": "0",
                "fold": "3",
            },
        ]
        assignments = select_assignments(rows, 0.5, 10.0)
        self.assertEqual(list(assignments), [("test", "video")])
        assignment = assignments[("test", "video")]
        self.assertEqual(assignment["router_policy"], "periodic_l9_30")
        self.assertEqual(assignment["early_frames"], 5)
        self.assertEqual(assignment["router_candidate_blocks"], 2)

    def test_summary_pairs_sequences_by_key(self):
        switched = []
        baselines = []
        for sequence, switch_iou, fast_iou, sgla_iou in (
            ("a", 0.8, 0.2, 0.9),
            ("b", 0.6, 0.1, 0.7),
        ):
            for frame_id in range(3):
                common = {
                    "dataset": "test",
                    "sequence": sequence,
                    "frame_id": frame_id,
                    "gt_valid": 1,
                    "candidate_blocks": int(frame_id == 2),
                    "model_decode_latency_ms": 2.0,
                    "bbox_x": float(frame_id),
                    "bbox_y": 0.0,
                    "bbox_w": 10.0,
                    "bbox_h": 10.0,
                }
                switched.append(
                    dict(
                        common,
                        policy="router",
                        iou=(
                            1.0
                            if frame_id == 0
                            else fast_iou if frame_id == 1 else switch_iou
                        ),
                        early_frames="1",
                        assigned_policy="sgla",
                    )
                )
                for policy, value in (("fast", fast_iou), ("sgla", sgla_iou)):
                    baselines.append(
                        dict(
                            common,
                            policy=policy,
                            iou=1.0 if frame_id == 0 else value,
                        )
                    )
        methods, paired, comparisons = analyze(
            list(reversed(switched)),
            baselines,
            ("fast", "sgla"),
            bootstrap_samples=100,
            seed=2026,
        )
        router = next(row for row in methods if row["method"] == "router")
        self.assertGreater(router["gain_vs_fast"], 0.0)
        self.assertLess(router["gain_vs_sgla"], 0.0)
        fast_pair = next(row for row in paired if row["baseline"] == "fast")
        self.assertGreater(fast_pair["mean_delta"], 0.0)
        self.assertEqual(len(comparisons), 4)

    def test_summary_rejects_non_fast_prefix(self):
        switched = []
        baselines = []
        for frame_id in range(3):
            base = {
                "dataset": "test",
                "sequence": "video",
                "frame_id": frame_id,
                "gt_valid": 1,
                "candidate_blocks": 0,
                "model_decode_latency_ms": 2.0,
                "bbox_x": 0.0,
                "bbox_y": 0.0,
                "bbox_w": 10.0,
                "bbox_h": 10.0,
                "iou": 1.0,
            }
            baselines.extend(
                [dict(base, policy="fast"), dict(base, policy="sgla")]
            )
            switched.append(
                dict(
                    base,
                    policy="router",
                    early_frames="1",
                    assigned_policy="fast",
                    bbox_x=1.0 if frame_id == 1 else 0.0,
                )
            )
        with self.assertRaisesRegex(RuntimeError, "Fast prefix differs"):
            analyze(
                switched,
                baselines,
                ("fast", "sgla"),
                bootstrap_samples=10,
                seed=2026,
            )


if __name__ == "__main__":
    unittest.main()
