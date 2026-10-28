"""CPU-only tests for the Stage-5 Pareto summary."""

import unittest

from stage5_pareto_summary import POLICY_ORDER, build_pareto


class ParetoTests(unittest.TestCase):
    def test_periodic_latency_is_measured_path_mixture(self):
        auc = {
            policy: {
                "auc": value,
                "candidate_blocks": blocks,
            }
            for policy, value, blocks in zip(
                POLICY_ORDER,
                (60.0, 60.2, 60.4, 60.6, 60.8, 61.0),
                (0.0, 0.1, 0.2, 0.3, 0.5, 1.0),
            )
        }
        rows = {
            row["policy"]: row
            for row in build_pareto(
                auc, {"fast": 4.0, "fixed_l9": 5.0, "sgla": 6.0}
            )
        }

        self.assertAlmostEqual(rows["periodic_l9_30"]["latency_ms"], 4.3)
        self.assertAlmostEqual(
            rows["periodic_l9_30"]["recovery_of_sgla_gain"], 0.6
        )
        self.assertAlmostEqual(rows["sgla"]["fps"], 1000.0 / 6.0)
        self.assertEqual(
            rows["periodic_l9_30"]["latency_source"],
            "measured_path_mixture",
        )

    def test_rejects_nonpositive_sgla_gain(self):
        auc = {
            policy: {"auc": 60.0, "candidate_blocks": 0.0}
            for policy in POLICY_ORDER
        }
        with self.assertRaisesRegex(ValueError, "SGLA AUC"):
            build_pareto(
                auc, {"fast": 4.0, "fixed_l9": 5.0, "sgla": 6.0}
            )


if __name__ == "__main__":
    unittest.main()
