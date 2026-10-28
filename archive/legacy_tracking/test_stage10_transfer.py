"""CPU-only tests for the Stage-10 closed-loop transfer diagnostic."""

import unittest
from types import SimpleNamespace

from stage4_common import fingerprint
from stage10_closed_loop_summary import (
    add_reference_metrics,
    audit_contracts,
    audit_run_coverage,
)
from stage10_closed_loop_transfer import (
    STAGE9_PURPOSE,
    assignment_for_policy,
    expected_candidate_blocks,
    resolve_policy,
    validate_manifest,
)


class TransferPolicyTests(unittest.TestCase):
    def setUp(self):
        self.sequence = SimpleNamespace(
            dataset="got10k_train",
            name="clip",
            frames=[None] * 100,
        )

    def test_frozen_policy_costs(self):
        self.assertEqual(expected_candidate_blocks(99, "fast"), 0)
        self.assertEqual(expected_candidate_blocks(99, "periodic_l9_20"), 19)
        self.assertEqual(expected_candidate_blocks(99, "sgla"), 99)
        assignment = assignment_for_policy(self.sequence, "periodic_l9_20")
        self.assertEqual(assignment["router_candidate_blocks"], 19)
        self.assertEqual(
            assignment["early_frames"] + assignment["late_frames"], 99
        )

    def test_fixed_resolver_ignores_reporting_boundary(self):
        self.assertEqual(resolve_policy(1, 49, "periodic_l9_20"), "periodic_l9_20")
        self.assertEqual(resolve_policy(99, 49, "periodic_l9_20"), "periodic_l9_20")
        with self.assertRaisesRegex(ValueError, "Unknown Stage-10 policy"):
            resolve_policy(1, 49, "periodic_l9_30")

    def test_requires_stage9_manifest_purpose(self):
        manifest = {"purpose": STAGE9_PURPOSE}
        self.assertIs(validate_manifest(manifest), manifest)
        with self.assertRaisesRegex(ValueError, "requires a frozen Stage-9"):
            validate_manifest({"purpose": "stage7_frozen_temporal_holdout"})


class TransferSummaryTests(unittest.TestCase):
    def _contract(self, policy):
        contract = {
            "schema": "stage10_closed_loop_transfer_v1",
            "manifest_fingerprint": "manifest",
            "manifest_purpose": STAGE9_PURPOSE,
            "dataset": "got10k_train",
            "split": "val",
            "sequence_filter": None,
            "expected_sequences": ["a", "b"],
            "policy": policy,
            "tracker_name": "sglatrack",
            "tracker_parameter": "deit_distilled",
            "checkpoint_sha256": "checkpoint",
            "latency_warmup_frames": 20,
            "iou_protocol": "protocol",
        }
        contract["fingerprint"] = fingerprint(contract)
        return contract

    def test_contracts_must_share_manifest_and_checkpoint(self):
        contracts = {
            policy: self._contract(policy)
            for policy in ("fast", "periodic_l9_20", "sgla")
        }
        self.assertEqual(audit_contracts(contracts), contracts["fast"])
        contracts["sgla"]["checkpoint_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
            audit_contracts(contracts)

    def test_coverage_requires_every_frozen_sequence(self):
        contract = self._contract("fast")
        run = {
            "path": "run",
            "method": "fast",
            "grouped": {
                ("got10k_train", "a"): [],
                ("got10k_train", "b"): [],
            },
        }
        audit_run_coverage(run, contract, "fast")
        run["grouped"].pop(("got10k_train", "b"))
        with self.assertRaisesRegex(ValueError, "expected 2, got 1"):
            audit_run_coverage(run, contract, "fast")

    def test_reference_metrics_compute_recovery(self):
        summaries = [
            {"method": "periodic_l9_20", "auc": 65.0},
            {"method": "fast", "auc": 64.0},
            {"method": "sgla", "auc": 66.0},
        ]
        empty_run = {"grouped": {("d", "s"): []}}
        runs = {
            "fast": empty_run,
            "periodic_l9_20": empty_run,
            "sgla": empty_run,
        }
        rows = add_reference_metrics(summaries, runs)
        periodic = next(row for row in rows if row["method"] == "periodic_l9_20")
        self.assertAlmostEqual(periodic["gain_vs_fast"], 1.0)
        self.assertAlmostEqual(periodic["gain_vs_sgla"], -1.0)
        self.assertAlmostEqual(periodic["recovery_of_sgla_gain"], 0.5)


if __name__ == "__main__":
    unittest.main()
