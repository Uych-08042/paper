"""NumPy-only regression checks for the immutable post-hoc Stage-35 audit."""

import contextlib
import io
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

import numpy as np

from stage4_common import file_hash, write_cache, write_json
from stage27_uav_common import failure_events, read_boxes, seal, success_auc, tracking_iou
from stage27_uav_summary import write_table
from stage35_preserve_common import ARMS, BASELINES, DESIGN
from test_stage35_preservation import fixture
import stage35_supervised_audit as audit_module

HERE = Path(__file__).parent


def metric_rows():
    rows = []
    for video in range(8):
        for method in BASELINES + ARMS:
            for seed in (DESIGN["seeds"] if method in ARMS else ["baseline"]):
                # One large positive video; the other seven are negative.
                gain = (16. if video == 0 else -1.) + (1. if seed == DESIGN["seeds"][0] else -1.)
                auc = 60. + gain if method == "supervised" else 60.
                rows.append(dict(sequence=str(video), method=method, seed=seed, fold=video // 2,
                    auc=auc, low_iou_fraction=.1, failure_events=0, first_failure_frame=None, instrumented_ms=1.,
                    fast_reliable_auc=80. + gain if method == "supervised" else 80.,
                    sgla_reliable_auc=80. + gain if method == "supervised" else 80., sgla_failure_auc=None))
    return rows


def source_fixture(root):
    plan = fixture(root)
    output = root / "stage35"
    models = {}
    for seed in DESIGN["seeds"]:
        records = []
        for fold in range(DESIGN["folds"]):
            for arm in ARMS:
                path = output / ("training/seed{}/fold{}/{}/step{:06d}.pt".format(seed, fold, arm, DESIGN["updates"]))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes("fake checkpoint {} {} {}".format(seed, fold, arm).encode())
                models[str(path.resolve())] = file_hash(path)
                records.append(dict(fold=fold, arm=arm, checkpoint=str(path.resolve()), checkpoint_sha256=file_hash(path)))
        path = output / ("training/seed{}/stage35_training_decision.json".format(seed))
        write_json(path, seal(dict(schema=plan["schema"], status="COMPLETE_MATCHED_OOF_TRAINING",
            seed=seed, plan_fingerprint=plan["fingerprint"], arms=records)))
        models[str(path.resolve())] = file_hash(path)
    methods = list(BASELINES) + sorted(a + "_seed" + str(s) for a in ARMS for s in DESIGN["seeds"])
    run = seal(dict(schema=plan["schema"], plan_fingerprint=plan["fingerprint"], methods=methods, frozen_model_sha256=models))
    write_json(output / "oof/run.json", run)
    hashes, measurements = {}, defaultdict(list)
    for number, entry in enumerate(plan["fit"]):
        gt = read_boxes(Path(plan["root"]) / entry["annotation"])[1:]
        n = len(gt)
        fold = next(f["fold"] for f in plan["folds"] if entry["sequence"] in f["held_out"])
        for method in methods:
            arm, seed = method.rsplit("_seed", 1) if "_seed" in method else (method, "baseline")
            boxes = gt.copy()
            if arm == "fast_initial" and number % 2:
                boxes[n // 2:, 0] += 20
            elif arm == "sgla" and number % 3:
                boxes[n // 2:, 0] += 20
            elif arm == "supervised" and number == 0 and seed == str(DESIGN["seeds"][1]):
                boxes[n // 2:, 0] += 20  # One seed introduces a failure on a stable video.
            elif arm not in BASELINES + ("supervised",):
                boxes[:, 0] += 3
            iou = tracking_iou(boxes, gt)
            data = dict(frame_id=np.arange(1, n + 1), previous=np.vstack([entry["initial_box"], boxes[:-1]]),
                boxes=boxes, gt=gt, iou=iou, elapsed_ms=np.ones(n), main_head_calls=np.asarray(n))
            path = output / "oof" / (entry["sequence"] + "__" + method + ".npz")
            write_cache(path, data, dict(schema=plan["schema"], run_fingerprint=run["fingerprint"],
                sequence=entry["sequence"], fold=fold, method=method))
            hashes[path.name] = file_hash(path)
            measurements[(entry["sequence"], arm)].append((success_auc(iou), float(np.mean(iou < .2))))
    original = seal(dict(schema=plan["schema"], role="consumed_development_fit_only_group_oof_screen",
        status="NO_PRESERVE_CORRECT_GAIN_ESTABLISHED", plan_fingerprint=plan["fingerprint"],
        oof_run_fingerprint=run["fingerprint"], videos=8, seeds=DESIGN["seeds"], cache_sha256=hashes,
        original_checking_labels_loaded=False))
    write_json(output / "analysis/stage35_decision.json", original)
    policies = []
    for method in BASELINES + ARMS:
        per_video = [np.mean(measurements[(e["sequence"], method)], axis=0) for e in plan["fit"]]
        values = np.mean(per_video, axis=0)
        policies.append(dict(method=method, videos=8, auc=float(values[0]), low_iou_fraction=float(values[1]), instrumented_ms=1.))
    write_table(output / "analysis/stage35_policy_summary.csv", policies)
    return output


class AuditTests(unittest.TestCase):
    def test_video_pairing_seed_consistency_and_single_video_influence(self):
        with mock.patch.dict(audit_module.AUDIT, bootstrap_samples=128):
            rows = metric_rows()
            averaged, paired, influence, stability, per_seed = audit_module.analyze(rows)
            pair = next(r for r in paired if r["metric"] == "auc" and r["contrast"] == "supervised_minus_fast_initial")
            self.assertEqual(pair["videos"], 8)  # Not 16 seed-video pairs.
            self.assertAlmostEqual(pair["mean_delta"], 9 / 8)
            self.assertEqual((pair["wins"], pair["ties"], pair["losses"]), (1, 0, 7))
            impact = next(r for r in influence if r["sequence"] == "0" and r["reference"] == "fast_initial")
            self.assertEqual(impact["positive_gain_share"], 1.)
            self.assertEqual(impact["leave_one_video_out_mean_delta"], -1.)
            self.assertTrue(impact["mean_sign_changes_if_omitted"])
            means = [r["mean_auc_delta"] for r in per_seed if r["reference"] == "fast_initial"]
            np.testing.assert_allclose(means, [2.125, .125])
            unsupported = next(r for r in paired if r["metric"] == "sgla_failure_auc")
            self.assertEqual(unsupported["videos"], 0)
            self.assertIsNone(unsupported["mean_delta"])
            self.assertIsNone(unsupported["ci_low"])
            self.assertEqual(audit_module.analyze(list(reversed(rows)))[1], paired)
            reduced = [dict(r, low_iou_fraction=.05) if r["method"] == "supervised" else dict(r) for r in rows]
            errors = next(r for r in audit_module.analyze(reduced)[1] if r["metric"] == "low_iou_fraction" and r["contrast"] == "supervised_minus_fast_initial")
            self.assertEqual(errors["wins"], 8)  # Lower failure fraction is favorable.
            self.assertEqual(errors["favorable_direction"], "negative")
            with self.assertRaisesRegex(ValueError, "Missing/duplicate"):
                audit_module.analyze(rows + [rows[0]])

    def test_immutable_full_cache_audit_repeat_and_changed_source_rejection(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            output = source_fixture(root)
            old_files = {str(p): file_hash(p) for p in root.rglob("*") if p.is_file()}
            with mock.patch.dict(audit_module.AUDIT, bootstrap_samples=128):
                report = audit_module.audit(output)
                self.assertEqual(report["status"], "COMPLETE_POSTHOC_SUPERVISED_AUDIT")
                self.assertEqual(report["original_status"], "NO_PRESERVE_CORRECT_GAIN_ESTABLISHED")
                self.assertEqual(report["videos"], 8)
                self.assertFalse(report["training_performed"])
                self.assertFalse(report["new_inference_performed"])
                self.assertIsNone(report["adoption_gate"])
                self.assertEqual(report["summaries"]["fast_initial"]["introduced_failure_videos"], 1)
                # Repeat is deterministic, preserving old decisions, caches and model bytes.
                self.assertEqual(report, audit_module.audit(output))
                audit_module.verify_bindings(old_files)
                destination = output / "analysis/supervised_baseline_audit_v1"
                self.assertEqual(len(list(destination.glob("*.csv"))), 6)
                self.assertEqual(json.loads((destination / "stage35b_audit.json").read_text()), report)
                # Detect even an otherwise loadable archive with bytes appended.
                corrupt = next((output / "oof").glob("*.npz"))
                original_cache_bytes = corrupt.read_bytes()
                corrupt.write_bytes(original_cache_bytes + b"changed source")
                with self.assertRaisesRegex(ValueError, "Frozen Stage-35 artifact changed"):
                    audit_module.audit(output)
                # No replacement audit result is written after a source failure.
                self.assertEqual(json.loads((destination / "stage35b_audit.json").read_text()), report)
                corrupt.write_bytes(original_cache_bytes)
                second_seed_model = output / ("training/seed{}/fold3/correct/step{:06d}.pt".format(DESIGN["seeds"][1], DESIGN["updates"]))
                second_seed_model.write_bytes(second_seed_model.read_bytes() + b"changed model")
                with self.assertRaisesRegex(ValueError, "Frozen Stage-35 artifact changed"):
                    audit_module.audit(output)

    def test_incomplete_and_mixed_cache_are_rejected_before_analysis(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary, contextlib.redirect_stdout(io.StringIO()):
            output = source_fixture(Path(temporary))
            path = next((output / "oof").glob("*.npz"))
            original_bytes = path.read_bytes()
            path.unlink()
            with self.assertRaisesRegex(ValueError, "original 96"):
                audit_module.audit(output)
            path.write_bytes(original_bytes)
            # Metadata corruption with a matching cache SHA must still be rejected.
            with np.load(path, allow_pickle=False) as archive:
                data = {k: archive[k] for k in archive.files if k != "metadata_json"}
                meta = json.loads(str(archive["metadata_json"].item()))
            meta["fold"] = 99
            write_cache(path, data, meta)
            decision_path = output / "analysis/stage35_decision.json"
            decision = json.loads(decision_path.read_text())
            decision.pop("fingerprint")
            decision["cache_sha256"][path.name] = file_hash(path)
            write_json(decision_path, seal(decision))
            with self.assertRaisesRegex(ValueError, "Mixed source trace metadata"):
                audit_module.audit(output)


if __name__ == "__main__":
    unittest.main()
