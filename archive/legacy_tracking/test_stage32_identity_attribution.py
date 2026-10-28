"""Attribution accounting, strict source binding and no-checking/no-image regression."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from stage4_common import file_hash, write_cache, write_json
from stage27_uav_common import seal, tracking_iou
from stage31_identity_common import FEATURES, METHODS
from stage31_identity_fit import load_model
import stage31_identity_engine as engine
from stage31_precision_compat import compatibility_record, precision_replay_auditor
from stage32_identity_attribution import (analyze_video, closed_loop_row, load_sources, pair_diagnostics,
    run_attribution, score_candidates, summarize_pairs, verify_unchanged)
from test_stage31_precision_compat import Float32Observer, contract_fixture


HERE = Path(__file__).resolve().parent


def diagnostic_arrays():
    """Native succeeds once; identity regresses, misses once, fixes once; one absent GT."""
    target, other = [40., 20., 10., 10.], [10., 20., 10., 10.]
    boxes = np.array([[target, other], [other, target], [other, target], [other, other]])
    features = np.zeros((4, 2, len(FEATURES)))
    features[..., 0] = [[.1, .9], [.9, .1], [.1, .9], [.9, .1]]
    chosen = np.array([1, 0, 1, 0])
    gt = np.tile(target, (4, 1))
    data = dict(frame_id=np.arange(1, 5), features=features, valid=np.ones((4, 2), bool),
        candidate_boxes=boxes, chosen=chosen, gt=gt, boxes=boxes[np.arange(4), chosen],
        iou=tracking_iou(boxes[np.arange(4), chosen], gt))
    model = dict(weights=[1.] + [0.] * 8, feature_scale=[1.] * 9,
                 static_weights=[1.] + [0.] * 8, static_feature_scale=[1.] * 9)
    entry = dict(sequence="uav0000001_00000_s", source_group_proxy="uav0000001", split="fit")
    return entry, data, model


class AccountingTests(unittest.TestCase):
    def test_partition_corrections_regressions_and_missed_opportunities(self):
        entry, data, model = diagnostic_arrays()
        candidate, choices, pairs, frames = analyze_video(entry, data, model, "identity")
        self.assertEqual(candidate["native_matched_frames"], 1)
        self.assertEqual(candidate["rank_recoverable_frames"], 2)
        self.assertEqual(candidate["no_gt_matching_candidate_frames"], 1)
        self.assertEqual(candidate["native_low_iou_frames"], 3)
        self.assertEqual(candidate["low_iou_recoverable_frames"], 2)
        self.assertEqual(candidate["low_iou_candidate_missing_frames"], 1)
        identity = next(r for r in choices if r["scorer"] == "identity")
        self.assertEqual(identity["corrected_native_low_frames"], 1)
        self.assertEqual(identity["regressed_native_matched_frames"], 1)
        self.assertEqual(identity["missed_recoverable_frames"], 1)
        self.assertEqual(identity["same_state_mean_iou_gain"], 0)
        self.assertEqual([r["identity_outcome"] for r in frames],
                         ["regress_native_matched", "keep_native", "correct_native_low", "keep_native"])
        for row in frames:
            if row["gt_retained"]:
                contribution = sum(row["identity_contribution_" + name] for name in FEATURES)
                self.assertAlmostEqual(contribution, row["identity_gt_best_minus_native_score"])
        # Replacing actor outputs does not change candidate/support-state counterfactuals.
        changed = dict(data, boxes=np.full_like(data["boxes"], 500), iou=np.zeros(4))
        _, choices2, _, _ = analyze_video(entry, changed, model, "identity")
        self.assertEqual(choices, choices2)

    def test_baseline_missing_top5_is_unknown_not_failure(self):
        entry, data, model = diagnostic_arrays()
        candidate, choices, pairs, frames = analyze_video(dict(entry, split="checking"), data, model, "sgla")
        self.assertFalse(candidate["candidate_sets_recorded"])
        self.assertIsNone(candidate["retained_gt_frames"])
        self.assertEqual((choices, pairs, frames), ([], [], []))

    def test_frame_then_video_balance_ignores_ambiguous_pairs_and_keeps_ties(self):
        entry, _, model = diagnostic_arrays()
        features = np.zeros((2, 4, 9))
        features[0, :, 0] = [.1, .9, .9, .9]  # Three losses in frame one.
        features[1, :, 0] = [.9, .1, .9, .9]  # One win; remaining ambiguous boxes excluded.
        overlap = np.array([[1., 0, 0, 0], [1., 0, .3, .3]])
        rows = pair_diagnostics(entry, features, np.ones((2, 4), bool), overlap,
                                score_candidates(features, model), model)
        appearance = next(r for r in rows if r["cohort"] == "all_pair_frames" and r["signal"] == "initial_cos")
        self.assertEqual(appearance["pairs"], 4)
        self.assertEqual(appearance["pair_accuracy"], .5)  # Not micro pair accuracy .25.
        ties = next(r for r in rows if r["cohort"] == "all_pair_frames" and r["signal"] == "trusted_cos")
        self.assertEqual((ties["pair_accuracy"], ties["strict_pair_accuracy"], ties["tie_ratio"]), (.5, 0., 1.))
        summary = summarize_pairs([dict(r, trajectory="sgla") for r in rows])
        absent = next(r for r in summary if r["cohort"] == "native_low_iou" and r["signal"] == "initial_cos")
        self.assertEqual(absent["supported_videos"], 0)
        self.assertIsNone(absent["macro_pair_accuracy"])
        self.assertIsNone(absent["descriptive_ci_low"])
        all_absent = summarize_pairs([], {("fit", "sgla")})
        self.assertTrue(all(r["macro_pair_accuracy"] is None for r in all_absent))

    def test_closed_loop_prefix_is_separate_from_same_state_changes(self):
        entry, data, _ = diagnostic_arrays()
        identity = dict(data)
        identity["chosen"] = np.array([0, 0, 1, 0])
        identity["boxes"] = data["candidate_boxes"][np.arange(4), identity["chosen"]]
        identity["iou"] = tracking_iou(identity["boxes"], data["gt"])
        baseline = dict(data, boxes=data["candidate_boxes"][:, 0], iou=np.array([1., 0, 0, 0]))
        row = closed_loop_row(entry, baseline, identity)
        self.assertEqual(row["first_alternative_frame"], 3)
        self.assertEqual(row["first_bbox_divergence_frame"], 3)
        self.assertEqual(row["frames_before_first_alternative"], 2)
        self.assertFalse(row["prefix_difference_above_1e_minus9"])


def frozen_fixture(temporary):
    directory, contract, reader = contract_fixture(temporary)
    run = seal(dict(schema=contract["schema"], role="native_fit_candidate_features", contract_fingerprint=contract["fingerprint"]))
    write_json(directory / "fit_run.json", run)
    write_json(directory / "stage31_precision_compat.json", compatibility_record(contract))
    with precision_replay_auditor():
        for entry in contract["sequences"]:
            if entry["split"] == "fit":
                data = engine.run_video(Float32Observer(contract["runtime"]), contract, entry, "sgla", fit_collect=True, image_reader=reader)
                engine.audit_video(data, contract, entry, "sgla", fit_collect=True)
                write_cache(directory / "fit_cache" / (entry["sequence"] + ".npz"), data, engine.cache_metadata(run, entry, "sgla"))
    result = subprocess.run([sys.executable, "-B", str(HERE / "stage31_precision_compat.py"), "train", "--input", str(directory)],
                            text=True, capture_output=True)
    if result.returncode:
        raise AssertionError(result.stderr)
    model = load_model(directory, contract)
    checking = seal(dict(schema=contract["schema"], role="frozen_full_video_identity_checking", contract_fingerprint=contract["fingerprint"],
                        model_fingerprint=model["fingerprint"], methods=list(METHODS)))
    write_json(directory / "checking_run.json", checking)
    with precision_replay_auditor():
        for entry in contract["sequences"]:
            if entry["split"] == "checking":
                for method in METHODS:
                    data = engine.run_video(Float32Observer(contract["runtime"]), contract, entry, method, model, image_reader=reader)
                    engine.audit_video(data, contract, entry, method, model)
                    write_cache(directory / "checking" / (entry["sequence"] + "__" + method + ".npz"), data,
                                engine.cache_metadata(checking, entry, method))
    result = subprocess.run([sys.executable, "-B", str(HERE / "stage31_precision_compat.py"), "summary", "--input", str(directory)],
                            text=True, capture_output=True)
    if result.returncode:
        raise AssertionError(result.stderr)
    return directory, contract


class IntegrationTests(unittest.TestCase):
    def test_frozen_workflow_no_images_no_checking_by_default_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            root = Path(temporary)
            directory, contract = frozen_fixture(root)
            before = {str(p): file_hash(p) for p in directory.rglob("*") if p.is_file()}
            # Image decoding is forbidden. Annotations remain for original cache audits.
            for p in (root / "dataset").rglob("*.jpg"):
                p.unlink()
            original_gt = engine.ground_truth
            def fit_labels_only(contract, entry):
                if entry["split"] != "fit":
                    raise AssertionError("Fit-only attribution opened checking labels")
                return original_gt(contract, entry)
            with mock.patch.object(engine, "ground_truth", side_effect=fit_labels_only):
                fit_report = run_attribution(directory, root / "fit_audit", "fit")
            self.assertFalse(fit_report["checking_labels_loaded"])
            self.assertEqual(fit_report["replay_audit"]["audited_arrays"], 8)
            self.assertEqual(fit_report["replay_audit"]["max_float32_relative_response_delta"], 0)
            both = run_attribution(directory, root / "full_audit", "both")
            self.assertEqual(both["replay_audit"]["audited_arrays"], 38)
            self.assertFalse(both["training_performed"])
            self.assertFalse(both["gpu_inference_performed"])
            self.assertEqual(before, {str(p): file_hash(p) for p in directory.rglob("*") if p.is_file()})
            # The actual CLI also works without images or a GPU. Same-source
            # rerun is deterministic and source files stay untouched.
            first = file_hash(root / "fit_audit" / "stage32_attribution.json")
            cli = subprocess.run([sys.executable, "-B", str(HERE / "stage32_identity_attribution.py"),
                "--input", str(directory), "--output_dir", str(root / "fit_audit")], text=True, capture_output=True)
            self.assertEqual(cli.returncode, 0, cli.stderr)
            self.assertEqual(first, file_hash(root / "fit_audit" / "stage32_attribution.json"))
            with (root / "full_audit" / "stage32_candidate_summary.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            baseline = next(r for r in rows if r["split"] == "checking" and r["trajectory"] == "sgla")
            self.assertEqual(baseline["candidate_sets_recorded"], "False")
            self.assertEqual(baseline["retained_gt_frames"], "")
            for destination in (directory, directory / "analysis_new", root):
                with self.assertRaisesRegex(ValueError, "separate Stage-32"):
                    run_attribution(directory, destination)
            # Changing even the bytes of a semantically valid source cache must be rejected.
            changed = next((directory / "checking").glob("*.npz"))
            changed.write_bytes(changed.read_bytes() + b"altered archive bytes")
            with self.assertRaisesRegex(ValueError, "differs from frozen"):
                load_sources(directory, "both")
            with self.assertRaisesRegex(ValueError, "changed during analysis"):
                verify_unchanged(before)
            # The fit-only path remains independent of consumed checking artifacts.
            run_attribution(directory, root / "fit_after_checking_change", "fit")


if __name__ == "__main__":
    unittest.main()
