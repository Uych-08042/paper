"""CPU-only tests for Stage-6 matched control helpers."""

import unittest

from stage6_matched_assignments import (
    assignment_rows,
    policy_cost,
    random_budget_selection,
    uniform_selection,
)
from stage6_switched_compare import compare


def _assignments():
    return {
        ("test", "a"): {
            "dataset": "test",
            "sequence": "a",
            "early_frames": 5,
            "late_frames": 5,
            "router_policy": "fast",
            "router_candidate_blocks": 0,
            "fold": 0,
        },
        ("test", "b"): {
            "dataset": "test",
            "sequence": "b",
            "early_frames": 10,
            "late_frames": 10,
            "router_policy": "periodic_l9_30",
            "router_candidate_blocks": 3,
            "fold": 1,
        },
    }


class MatchedControlTests(unittest.TestCase):
    def test_policy_cost_uses_global_decision_index(self):
        self.assertEqual(policy_cost("fast", 5, 5), 0)
        self.assertEqual(policy_cost("periodic_l9_10", 5, 5), 1)
        self.assertEqual(policy_cost("periodic_l9_30", 5, 5), 2)
        self.assertEqual(policy_cost("sgla", 5, 5), 5)

    def test_uniform_rows_are_runner_compatible(self):
        assignments = _assignments()
        selected = uniform_selection(assignments, "periodic_l9_10")
        rows = assignment_rows(
            assignments,
            selected,
            early_fraction=0.5,
            budget_percent=10.0,
            control_name="uniform_late_p10",
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            {row["router_policy"] for row in rows},
            {"periodic_l9_10"},
        )
        self.assertEqual(
            sum(row["router_candidate_blocks"] for row in rows), 2
        )

    def test_random_budget_is_deterministic_and_feasible(self):
        assignments = _assignments()
        policies = ("fast", "periodic_l9_30", "sgla")
        first = random_budget_selection(assignments, policies, 4, seed=2026)
        second = random_budget_selection(assignments, policies, 4, seed=2026)
        self.assertEqual(first, second)
        cost = sum(
            policy_cost(
                first[key],
                assignments[key]["early_frames"],
                assignments[key]["late_frames"],
            )
            for key in assignments
        )
        self.assertLessEqual(cost, 4)

    def test_paired_comparison_checks_prefix_and_delta(self):
        reference_rows = []
        control_rows = []
        for sequence, reference_late, control_late in (
            ("a", 0.8, 0.5),
            ("b", 0.7, 0.6),
        ):
            for frame_id in range(3):
                common = {
                    "dataset": "test",
                    "sequence": sequence,
                    "frame_id": frame_id,
                    "early_frames": "1",
                    "candidate_blocks": int(frame_id == 2),
                    "bbox_x": float(frame_id),
                    "bbox_y": 0.0,
                    "bbox_w": 10.0,
                    "bbox_h": 10.0,
                }
                reference_rows.append(
                    dict(
                        common,
                        policy="router",
                        assigned_policy="periodic_l9_30",
                        iou=1.0 if frame_id == 0 else 0.4 if frame_id == 1 else reference_late,
                    )
                )
                control_rows.append(
                    dict(
                        common,
                        policy="uniform",
                        assigned_policy="periodic_l9_10",
                        iou=1.0 if frame_id == 0 else 0.4 if frame_id == 1 else control_late,
                    )
                )
        reference = {
            "method": "router",
            "files": ["a", "b"],
            "grouped": {
                ("test", sequence): [
                    row for row in reference_rows if row["sequence"] == sequence
                ]
                for sequence in ("a", "b")
            },
        }
        control = {
            "method": "uniform",
            "files": ["a", "b"],
            "grouped": {
                ("test", sequence): [
                    row for row in control_rows if row["sequence"] == sequence
                ]
                for sequence in ("a", "b")
            },
        }
        summaries, paired = compare(
            reference,
            [control],
            bootstrap_samples=100,
            seed=2026,
            tolerance=1e-8,
        )
        self.assertEqual(len(summaries), 2)
        self.assertEqual(paired[0]["wins"], 2)
        self.assertGreater(paired[0]["mean_delta"], 0.0)
        self.assertEqual(summaries[1]["prefix_max_bbox_abs_diff"], 0.0)

    def test_temporal_comparison_can_skip_prefix_equality(self):
        reference = {
            "method": "late",
            "files": ["a"],
            "grouped": {
                ("test", "a"): [
                    {
                        "dataset": "test",
                        "sequence": "a",
                        "policy": "late",
                        "assigned_policy": "late_p10",
                        "frame_id": frame_id,
                        "early_frames": "1",
                        "candidate_blocks": int(frame_id == 2),
                        "iou": value,
                        "bbox_x": value,
                        "bbox_y": 0.0,
                        "bbox_w": 10.0,
                        "bbox_h": 10.0,
                    }
                    for frame_id, value in enumerate((1.0, 0.4, 0.8))
                ]
            },
        }
        control = {
            "method": "early",
            "files": ["a"],
            "grouped": {
                ("test", "a"): [
                    dict(
                        row,
                        policy="early",
                        assigned_policy="early_p10",
                        bbox_x=row["bbox_x"] + (1.0 if row["frame_id"] == 1 else 0.0),
                    )
                    for row in reference["grouped"][("test", "a")]
                ]
            },
        }
        summaries, paired = compare(
            reference,
            [control],
            bootstrap_samples=20,
            seed=2026,
            tolerance=1e-8,
            check_prefix=False,
        )
        self.assertEqual(len(summaries), 2)
        self.assertEqual(len(paired), 1)
        self.assertEqual(summaries[1]["prefix_frames_compared"], "")


if __name__ == "__main__":
    unittest.main()
