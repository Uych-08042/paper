"""CPU-only tests for Stage-8 tail and temporal diagnostics."""

import unittest

from stage8_tail_diagnostic import (
    POLICIES,
    _disagreement_metrics,
    _failure_metrics,
    _trimmed_mean,
    analyze,
    build_cohort,
)


def make_rows(sequence, policy, ious, candidate_frames=()):
    rows = []
    for frame_id, iou in enumerate(ious):
        rows.append(
            {
                "dataset": "test",
                "sequence": sequence,
                "policy": policy,
                "frame_id": frame_id,
                "gt_valid": 1,
                "iou": float(iou),
                "candidate_blocks": int(frame_id in candidate_frames),
                "model_decode_latency_ms": 1.0,
            }
        )
    return rows


def make_run(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["sequence"]), []).append(row)
    return {
        "path": None,
        "files": [],
        "rows": rows,
        "grouped": grouped,
    }


def make_cohort(name="cohort", late_cost=(4,), uniform_cost=(2,)):
    values = {
        "fast": [1.0, 0.2, 0.2, 0.2, 0.2],
        "uniform_p5": [1.0, 0.2, 0.3, 0.2, 0.4],
        "late_p10_exact": [1.0, 0.2, 0.2, 0.3, 0.4],
        "sgla": [1.0, 0.5, 0.5, 0.5, 0.5],
    }
    costs = {
        "fast": (),
        "uniform_p5": uniform_cost,
        "late_p10_exact": late_cost,
        "sgla": (1, 2, 3, 4),
    }
    runs = {
        policy: make_run(make_rows("video", policy, values[policy], costs[policy]))
        for policy in POLICIES
    }
    return build_cohort(name, runs)


class Stage8DiagnosticTests(unittest.TestCase):
    def test_equal_cost_cohort_and_report_shapes(self):
        cohort = make_cohort()
        reports = analyze(
            [("cohort", cohort)],
            bins=2,
            tail_count=1,
            bootstrap_samples=20,
            seed=3,
            disagreement_threshold=0.05,
        )
        self.assertEqual(cohort[0]["costs"]["uniform_p5"], 1)
        self.assertEqual(cohort[0]["costs"]["late_p10_exact"], 1)
        late_fast = next(
            row
            for row in reports["delta_summary"]
            if row["comparison"] == "late_minus_fast"
        )
        self.assertGreater(late_fast["mean_delta"], 0.0)
        self.assertEqual(len(reports["temporal_bins"]), 4 * 2)
        self.assertEqual(len(reports["tail_sequences"]), 4 * 2)

    def test_rejects_per_sequence_cost_mismatch(self):
        with self.assertRaisesRegex(ValueError, "Equal-cost audit failed"):
            make_cohort(late_cost=(3, 4), uniform_cost=(2,))

    def test_rejects_overlapping_cohorts(self):
        first = make_cohort("first")
        second = make_cohort("second")
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            analyze(
                [("first", first), ("second", second)],
                bins=2,
                tail_count=1,
                bootstrap_samples=10,
                seed=1,
                disagreement_threshold=0.1,
            )

    def test_failure_run_breaks_on_recovery(self):
        rows = make_rows("video", "fast", [1.0, 0.0, 0.0, 0.8, 0.0])
        metrics = _failure_metrics(rows, 0.1)
        self.assertEqual(metrics["failure_frames"], 3)
        self.assertEqual(metrics["longest_failure_run"], 2)
        self.assertEqual(metrics["failure_episodes"], 2)
        self.assertEqual(metrics["recovered_episodes"], 1)
        self.assertEqual(metrics["terminal_failure"], 1)

    def test_disagreement_reports_first_and_sustained_locations(self):
        left = make_rows(
            "video", "late_p10_exact", [1.0, 0.5, 0.7, 0.7, 0.7, 0.5]
        )
        right = make_rows(
            "video", "uniform_p5", [1.0, 0.5, 0.5, 0.5, 0.5, 0.5]
        )
        metrics = _disagreement_metrics(left, right, 0.1)
        self.assertAlmostEqual(metrics["first_disagreement_fraction"], 0.4)
        self.assertAlmostEqual(metrics["first_sustained3_fraction"], 0.4)
        self.assertEqual(metrics["disagreement_frames"], 3)

    def test_invalid_frame_breaks_sustained_disagreement(self):
        left = make_rows(
            "video", "late_p10_exact", [1.0, 0.7, 0.7, 0.7, 0.7]
        )
        right = make_rows(
            "video", "uniform_p5", [1.0, 0.5, 0.5, 0.5, 0.5]
        )
        left[2]["gt_valid"] = 0
        right[2]["gt_valid"] = 0
        metrics = _disagreement_metrics(left, right, 0.1)
        self.assertTrue(
            metrics["first_sustained3_fraction"]
            != metrics["first_sustained3_fraction"]
        )

    def test_trimmed_mean_removes_both_tails(self):
        values = [-100.0] + [1.0] * 8 + [100.0]
        self.assertAlmostEqual(_trimmed_mean(values, 0.1), 1.0)


if __name__ == "__main__":
    unittest.main()
