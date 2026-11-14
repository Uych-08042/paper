"""Check harmful replacements, causal state references, full support and source seals."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage104_replacement_audit as stage
import test_stage103_candidate_verification as fixture


class ReplacementAudit(unittest.TestCase):
    setUp = fixture.Verification.setUp

    def synthetic(self, nonfinite_first=False, center_separation=100.):
        design = dict(stage.source103.DESIGN, center_separation=center_separation)
        anchor = np.array([1., 0.])
        history, memory, records = [self.entry["initial_box"].copy()], [], []
        for frame in range(1, 13):
            parents = stage.e63.source_common.prior.history_parents(history, self.ap["design"])
            boxes = history[-1]+np.array([[1., 0., 0., 0.], [31., 0., 0., 0.], [71., 0., 0., 0.], [5., 0., 0., 0.]])
            features = np.zeros((4, len(stage.e63.source_common.prior.FEATURES)))
            features[:, stage.e63.source_common.prior.FEATURES.index("hann_peak")] = [.91, .90, .89, .90]
            q = np.array([[.8, .6], [.99, np.sqrt(1-.99**2)], [.1, np.sqrt(.99)], [.7, np.sqrt(.51)]])
            if nonfinite_first and frame == 1:
                q[:] = np.nan
            row = dict(features=features, candidate_boxes=boxes, candidate_parents=parents, history_latest=history[-1], selected=2)
            decision = stage.source103.decide(row, q, anchor, memory, frame, "four_matched", design)
            box = boxes[decision["selected"]].copy()
            records.append(row | decision | dict(frame_id=frame, boxes=box, all_queries=q, image_hw=np.array([2000, 2000])))
            history = (history+[box])[-self.ap["design"]["history"]:]
        data = {k: np.asarray([r[k] for r in records]) for k in records[0]} | dict(anchor=anchor)
        # Attach labels AFTER generating all outputs and their histories.
        gt = np.vstack([self.entry["initial_box"], [data["candidate_boxes"][i, 0 if i % 2 else 1] for i in range(12)]])
        entry = dict(self.entry, ground_truth=gt)
        spec = dict(cohort="historical", unit=dict(sequence="test", seed=2171))
        return data, entry, spec, design

    def test_harm_rescue_wrong_memory_and_rejected_write_geometry_are_distinct(self):
        data, entry, spec, design = self.synthetic()
        rows, edges = stage.analyze_trace(data, entry, spec, "four_matched", self.old, design)
        self.assertTrue(any(r["good_lost"] for r in rows))
        self.assertTrue(any(r["good_rescued"] for r in rows))
        self.assertTrue(any(r["low_memory_write"] for r in rows))
        self.assertTrue(any(r["reads_low_appearance"] for r in rows))
        self.assertTrue(any(r["next_crop_changed"] for r in rows))
        self.assertTrue(edges)
        self.assertTrue(all(e["reference_frame"] < e["frame"] for e in edges))
        self.assertTrue(any(e["reference_quality"] == "low" for e in edges))
        self.assertTrue(all(e["reference_was_replacement"] for e in edges))
        changed = {k: v.copy() for k, v in data.items()}
        changed["candidate_parents"][3, 0, 0] += 1
        with self.assertRaises(AssertionError):
            stage.analyze_trace(changed, entry, spec, "four_matched", self.old, design)
        # The actual decision rejects writes with the original center-separation limit.
        rejected, labels, spec, design = self.synthetic(center_separation=.5)
        blocked, _ = stage.analyze_trace(rejected, labels, spec, "four_matched", self.old, design)
        self.assertFalse(any(r["memory_write"] for r in blocked))
        self.assertTrue(any(r["rejected_low_geometry_commit"] and r["next_crop_changed"] for r in blocked))

    def test_labels_do_not_change_reference_deletion_or_geometry_and_invalid_gaps(self):
        data, entry, spec, design = self.synthetic()
        rows, edges = stage.analyze_trace(data, entry, spec, "four_matched", self.old, design)
        wrong = dict(entry, ground_truth=np.tile([1000., 1000., 20., 20.], (13, 1)))
        other, dropped = stage.analyze_trace(data, wrong, spec, "four_matched", self.old, design)
        for a, b in zip(rows, other):
            for k in ("selected", "same_state_pair_selected", "appearance_refs", "memory_write", "next_crop_changed_mask", "gain_over_pair"):
                self.assertEqual(a[k], b[k])
        for a, b in zip(edges, dropped):
            for k in ("frame", "reference_frame", "actual_selected", "drop_selected", "drop_used_history", "drop_box_changed"):
                self.assertEqual(a[k], b[k])
        bad = dict(entry, ground_truth=entry["ground_truth"].copy())
        bad["ground_truth"][5] = [0., 0., 0., 0.]
        invalid, _ = stage.analyze_trace(data, bad, spec, "four_matched", self.old, design)
        self.assertFalse(invalid[4]["gt_valid"])
        self.assertIsNone(invalid[4]["same_state_iou_delta"])
        with self.assertRaises(ValueError):
            stage.replay_decision(data, 4, [(5, data["anchor"])], "four_matched", design)

    def test_nonfinite_write_is_not_an_effective_history_read(self):
        data, entry, spec, design = self.synthetic(nonfinite_first=True)
        self.assertTrue(data["memory_write"][0])
        rows, edges = stage.analyze_trace(data, entry, spec, "four_matched", self.old, design)
        self.assertFalse(any(e["reference_frame"] == 1 for e in edges))
        self.assertFalse(any("1" in r["appearance_refs"].split("|") for r in rows))

    def test_video_weights_and_all_recovery_losses_are_preserved(self):
        data, entry, spec, design = self.synthetic()
        rows, _ = stage.analyze_trace(data, entry, spec, "four_matched", self.old, design)
        sample = [dict(rows[0], cohort="historical", sequence="a", seed=1, good_lost=True)]*3
        sample += [dict(rows[0], cohort="historical", sequence="a", seed=2, good_lost=False),
                   dict(rows[0], cohort="expanded", sequence="b", seed=1, good_lost=False)]
        summary, _ = stage.aggregate(sample)
        self.assertAlmostEqual(next(r for r in summary if r["cohort"] == "all")["good_lost_fraction_video"], .25)
        pair = {k: v.copy() for k, v in data.items()}
        pair["boxes"] = entry["ground_truth"][1:].copy()
        pair["crop"] = data["candidate_parents"].copy()
        data["crop"] = data["candidate_parents"].copy()
        ref = dict(reference_arm="ppo", ppo_failure_frame=1, ppo_recovery_frame=5)
        recovery = [dict(ref, arm="pair", matched_good_frame=5), dict(ref, arm="four_matched", matched_good_frame=None)]
        cases, excerpt = stage.case_rows(dict(spec, truth=entry["ground_truth"]), "four_matched", rows, data, pair, recovery)
        self.assertTrue(any(r["role"] == "newly_lost_ppo_recovery" and r["frame"] == 5 for r in cases))
        self.assertTrue(excerpt)
        # Event indices from failure_events are zero-based 'start', never an 'onset' key.
        wrong = dict(spec, truth=np.tile([1000., 1000., 20., 20.], (13, 1)))
        cases, _ = stage.case_rows(wrong, "four_matched", rows, data, pair, [])
        self.assertTrue(any(r["role"] == "first_failure" and r["frame"] == 1 for r in cases))

    def test_complete_cached_pipeline_source_tables_resumption_and_tamper(self):
        verify = fixture.Verification()
        verify.old, verify.ap, verify.entry, verify.observer, verify.reader = self.old, self.ap, self.entry, self.observer, self.reader
        native, controls, _ = verify.baselines()
        traces = dict(ppo=controls["ppo"])
        for arm in stage.source103.MODES:
            traces[arm] = stage.source103.collect(self.observer, self.entry, self.old, fixture.choose, arm, stage.source103.DESIGN, image_reader=self.reader)
        unit = dict(sequence="test", seed=2171, origin="native", policy="ppo", fold=0)
        specs = [dict(unit=unit, cohort="historical"), dict(unit=dict(unit, seed=2172), cohort="historical"),
                 dict(unit=dict(unit, sequence="other"), cohort="expanded")]
        with tempfile.TemporaryDirectory() as directory:
            upstream, root = Path(directory)/"stage103", Path(directory)/"stage104"
            upstream.mkdir(); root.mkdir()
            plan103 = stage.io.seal(dict(schema=stage.source103.SCHEMA, units=specs,
                                        design=dict(stage.source103.DESIGN, bootstrap_samples=16)))
            seeds, recovery, scope, natives = [], [], [], set()
            for spec in specs:
                owner = spec["unit"]
                first = owner["sequence"] not in natives
                local = ({"native": native} if first else {}) | traces
                for arm, data in local.items():
                    identity = dict(owner, seed=0, policy="q0") if arm == "native" else owner
                    stage.c.save(upstream/"evaluation"/f"{stage.c.key(identity)}__{arm}.npz", data,
                                 dict(plan_fingerprint=plan103["fingerprint"], unit=identity, arm=arm))
                    scope.append(dict(cohort=spec["cohort"], sequence=owner["sequence"], seed=identity["seed"], arm=arm,
                        frames=len(data["frame_id"]),
                        first_output_difference_from_pair=stage.source103.first_commit_difference(data, traces["pair"]) if arm in stage.source103.MODES[1:] else None,
                        outside_pair_frames=int(np.count_nonzero((data["selected"] != data["ppo_selected"]) & (data["selected"] != data["legacy_challenger"]))) if arm in stage.source103.MODES else 0))
                a, b = stage.source_metrics(dict(spec, entry=self.entry), local, native, plan103["design"])
                seeds.extend(a); recovery.extend(b); natives.add(owner["sequence"])
            policies, pairs, videos = stage.source103.summarize(seeds, plan103["design"])
            (upstream/"analysis").mkdir()
            for name, values in (("policy_summary", policies), ("paired", pairs), ("video_summary", videos), ("seed_video", seeds), ("recovery", recovery), ("scope", scope)):
                stage.source103.s86.publish_table(upstream/f"analysis/stage103_{name}.csv", values)
            plan = stage.io.seal(dict(schema=stage.SCHEMA, expected=dict(units=3, unit_frames=36, recovery_rows=len(recovery)),
                source_runtime={}, source_sha256={}, code_sha256={}, design=stage.DESIGN))
            stage.c.publish_identical(root/"manifest.json", plan)
            with patch.object(self.observer, "capture", side_effect=AssertionError("New inference")), contextlib.redirect_stdout(io.StringIO()):
                tables = stage.run(upstream, plan103, lambda _: self.entry, lambda _: fixture.choose, self.old)
                stage.publish(root, plan, tables)
                stage.publish(root, plan, tables)
            self.assertEqual(len(tables["state_frames"]), 108)
            stage.check_results(root)
            path = root/"analysis/stage104_cases.csv"
            path.write_bytes(path.read_bytes()+b"tamper")
            with self.assertRaises(ValueError):
                stage.check_results(root)


if __name__ == "__main__":
    unittest.main()
