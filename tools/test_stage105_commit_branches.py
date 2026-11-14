"""Verify matched first outputs, independent commits, real continuations and budgets."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage105_commit_branches as stage
import test_stage103_candidate_verification as fixture
import test_stage104_replacement_audit as cached_fixture


class CommitBranches(unittest.TestCase):
    setUp = fixture.Verification.setUp

    def synthetic(self, separation=100.):
        parent = cached_fixture.ReplacementAudit()
        parent.entry, parent.ap = self.entry, self.ap
        trace, entry, spec, design = parent.synthetic(center_separation=separation)
        return trace, entry, spec, design

    def observation(self, trace, index):
        row = {k: trace[k][index].copy() for k in trace if k != "anchor"}
        row["crop"] = np.asarray([stage.crop_geometry(p, row["image_hw"], self.ap["search_factor"], self.ap["search_size"])[0] for p in row["candidate_parents"]])
        return row

    @staticmethod
    def choose_two(row, history):
        return dict(row, selected=2, history_latest=history[-1], boxes=row["candidate_boxes"][2].copy())

    def test_factorial_commits_share_published_output_and_only_intervene_once(self):
        trace, entry, _, design = self.synthetic()
        first, after = 7, {}
        observation = self.observation(trace, first-1)
        for treatment in stage.TREATMENTS:
            state = stage.initial_state(trace, entry, first, self.old, design)
            result = stage.advance(observation, state, trace["anchor"], first, first, treatment, "four_matched", self.old, self.choose_two, design)
            self.assertEqual(result["selected"], 1)
            self.assertEqual(result["pair_selected"], 0)
            self.assertEqual(result["geometry_selected"], 0 if stage.TREATMENTS[treatment][0] else 1)
            self.assertEqual(result["memory_selected"], 0 if stage.TREATMENTS[treatment][1] else 1)
            self.assertTrue(result["memory_committed"])
            np.testing.assert_array_equal(result["boxes"], observation["candidate_boxes"][1])
            np.testing.assert_array_equal(state["history"][-1], result["geometry_box"])
            np.testing.assert_array_equal(state["memory"][-1][1], observation["all_queries"][result["memory_selected"]])
            after[treatment] = state
            # Build a new observation on that branch's own successor parents.
            parents = stage.e63.source_common.prior.history_parents(state["history"], self.ap["design"])
            successor = dict(observation, frame_id=first+1, history_latest=state["history"][-1], candidate_parents=parents,
                crop=np.asarray([stage.crop_geometry(p, observation["image_hw"], self.ap["search_factor"], self.ap["search_size"])[0] for p in parents]))
            next_row = stage.advance(successor, state, trace["anchor"], first+1, first, treatment, "four_matched", self.old, self.choose_two, design)
            self.assertFalse(next_row["commit_intervention"])
            self.assertFalse(next_row["geometry_commit_changed"])
            self.assertFalse(next_row["memory_commit_changed"])
        self.assertFalse(np.array_equal(after["keep"]["history"][-2], after["geometry_pair"]["history"][-2]))

    def test_rejected_appearance_write_can_still_have_geometry_intervention(self):
        trace, entry, _, design = self.synthetic(separation=.5)
        first = 7
        for treatment in stage.TREATMENTS:
            state = stage.initial_state(trace, entry, first, self.old, design)
            result = stage.advance(self.observation(trace, first-1), state, trace["anchor"], first, first, treatment,
                                   "four_matched", self.old, self.choose_two, design)
            self.assertFalse(result["memory_write"])
            self.assertFalse(result["memory_committed"])
            self.assertFalse(result["memory_commit_changed"])
            self.assertEqual(result["geometry_commit_changed"], stage.TREATMENTS[treatment][0])
        with self.assertRaises(ValueError):
            state = stage.initial_state(trace, entry, first, self.old, design)
            state["memory"].append((first, trace["anchor"]))
            stage.advance(self.observation(trace, first-1), state, trace["anchor"], first, first, "keep",
                          "four_matched", self.old, self.choose_two, design)

    def test_anchors_cover_benefits_failures_and_censored_recoveries_without_lookahead_policy(self):
        trace, entry, _, _ = self.synthetic()
        cases = [dict(role="first_box_replacement", frame="2"),
                 dict(role="first_output_difference_from_full_pair", frame="2"),
                 dict(role="first_good_lost", frame="4"),
                 dict(role="new_failure_on_pair_stable", frame="8", last_replacement="6"),
                 dict(role="newly_lost_ppo_recovery", frame="11", last_replacement="9", last_good_lost="4", reference_event="8:11")]
        anchors = stage.select_anchors(trace, entry, cases, dict(horizon=4))
        by_frame = {r["frame"]: r for r in anchors}
        self.assertEqual({r["role"] for r in by_frame[2]["roles"]}, {"first_box_replacement", "first_output_difference_from_full_pair"})
        self.assertIn(6, by_frame); self.assertIn(7, by_frame); self.assertIn(10, by_frame)
        self.assertTrue(any(r["role"] == "first_good_rescued" for a in anchors for r in a["roles"]))
        self.assertLess(by_frame[4]["last"], 11)
        data = dict(frame_id=np.arange(4, 8), boxes=entry["ground_truth"][4:8])
        censored = stage.s103.source102.recovery_match(data, entry, 11, stage.s103.DESIGN)
        self.assertFalse(censored["complete_search_window"])
        self.assertFalse(censored["aligned_recovery"])

    def test_role_and_seed_video_weights_preserve_interaction_and_missing_future(self):
        anchors, branches = [], []
        for seq, seed, count, cohort in (("a", 1, 3, "historical"), ("a", 2, 1, "historical"), ("b", 1, 1, "expanded")):
            for frame in range(1, count+1):
                identity = dict(sequence=seq, seed=seed, source_arm="four_matched", frame=frame)
                anchors.append(dict(identity, roles=[dict(role="control"), dict(role="control")]))
                for treatment in stage.TREATMENTS:
                    value = 1. if seq == "a" and seed == 1 and treatment in ("geometry_pair", "both_pair") else 0.
                    branches.append(dict(identity, cohort=cohort, treatment=treatment,
                        **{m: value if m != "endpoint_iou" else None for m in stage.METRICS}))
        summary, paired = stage.aggregate(branches, anchors)
        row = next(r for r in paired if r["cohort"] == "all" and r["stratum"] == "all_unique_anchors" and r["contrast"] == "geometry_pair_minus_keep")
        self.assertAlmostEqual(row["future_mean_iou_delta"], .25)
        self.assertIsNone(row["endpoint_iou_delta"])
        self.assertEqual(row["endpoint_iou_supported_videos"], 0)
        interaction = next(r for r in paired if r["cohort"] == "all" and r["stratum"] == "all_unique_anchors" and r["contrast"] == "commit_interaction")
        self.assertEqual(interaction["future_mean_iou_delta"], 0.)
        self.assertEqual(next(r for r in summary if r["cohort"] == "all" and r["stratum"] == "control")["anchors"], 5)

    def test_real_continuations_gt_isolation_shared_first_audit_and_frozen_source(self):
        trace = stage.s103.collect(self.observer, self.entry, self.old, fixture.choose, "four_matched", stage.s103.DESIGN, image_reader=self.reader)
        first, last = 3, 9
        shared = stage.point(self.observer, trace, self.entry, first, self.old, fixture.choose, stage.s103.DESIGN, image_reader=self.reader)
        stage.audit_point(shared, trace, self.entry, first, "four_matched", self.old, fixture.choose, stage.s103.DESIGN)
        outputs = {}
        for treatment in stage.TREATMENTS:
            before = self.observer.calls
            data = stage.collect(self.observer, trace, shared, self.entry, first, last, treatment, "four_matched", self.old, fixture.choose, stage.s103.DESIGN, image_reader=self.reader)
            self.assertEqual(self.observer.calls-before, 4*(last-first))
            stage.audit(data, trace, shared, self.entry, first, last, treatment, "four_matched", self.old, fixture.choose, stage.s103.DESIGN)
            labels = dict(self.entry, ground_truth=np.full((13, 4), 999.))
            changed = stage.collect(self.observer, trace, shared, labels, first, last, treatment, "four_matched", self.old, fixture.choose, stage.s103.DESIGN, image_reader=self.reader)
            for key in stage.FIELDS:
                np.testing.assert_array_equal(data[key], changed[key])
            outputs[treatment] = data
        stage.compare_fields(outputs["keep"], {k: trace[k][first-1:last] for k in stage.s103.FIELDS}, stage.s103.FIELDS)
        bad = {k: v.copy() for k, v in outputs["geometry_pair"].items()}
        bad["geometry_box"][0, 0] += 1
        with self.assertRaises(AssertionError):
            stage.audit(bad, trace, shared, self.entry, first, last, "geometry_pair", "four_matched", self.old, fixture.choose, stage.s103.DESIGN)

    def test_effective_geometry_commit_changes_actual_successor_searches(self):
        original = stage.s103.observe
        capture = self.observer.capture
        design = dict(stage.s103.DESIGN, center_separation=100.)
        def moving_capture(image, parent, frame):
            result = capture(image, parent, frame)
            result["offset"][0] = .7
            _, _, factor = stage.crop_geometry(parent, image.shape[:2], 5., 100)
            dense = stage.s103.prior.s97.s95.dense_boxes(result["size"], result["offset"], parent, factor, 100, image.shape[:2])
            result["boxes"] = dense[int(result["raw"].argmax())]
            return result
        def observed(*args):
            row, _, grid = original(*args)
            x = np.array([.8, .99, .1, .7])
            return row, np.c_[x, np.sqrt(1-x*x)], grid
        with (patch.object(stage.s103, "observe", side_effect=observed),
              patch.object(self.observer, "capture", side_effect=moving_capture)):
            trace = stage.s103.collect(self.observer, self.entry, self.old, self.choose_two,
                                      "four_matched", design, image_reader=self.reader)
            positions = [i for i in range(11) if not np.array_equal(trace["candidate_boxes"][i, trace["selected"][i]],
                                                                   trace["candidate_boxes"][i, trace["pair_selected"][i]])]
            self.assertTrue(positions)
            first = positions[0]+1
            shared = stage.point(self.observer, trace, self.entry, first, self.old, self.choose_two, design, image_reader=self.reader)
            results = {}
            for treatment in stage.TREATMENTS:
                data = stage.collect(self.observer, trace, shared, self.entry, first, 12, treatment, "four_matched",
                                     self.old, self.choose_two, design, image_reader=self.reader)
                stage.audit(data, trace, shared, self.entry, first, 12, treatment, "four_matched", self.old, self.choose_two, design)
                results[treatment] = data
            self.assertTrue(results["geometry_pair"]["geometry_commit_changed"][0])
            self.assertTrue(results["memory_pair"]["memory_commit_changed"][0])
            np.testing.assert_array_equal(results["keep"]["boxes"][0], results["both_pair"]["boxes"][0])
            self.assertEqual(stage.first_difference(results["geometry_pair"], results["keep"], "candidate_parents"), first+1)
            stage.compare_fields(results["keep"], {k: trace[k][first-1:12] for k in stage.s103.FIELDS}, stage.s103.FIELDS)

    def test_mixed_cohort_pipeline_both_card_modes_real_call_budget_resume_and_tamper(self):
        traces = {mode: stage.s103.collect(self.observer, self.entry, self.old, fixture.choose, mode, stage.s103.DESIGN, image_reader=self.reader)
                  for mode in stage.s103.MODES[1:]}
        unit = dict(sequence="test", seed=2171, origin="native", policy="ppo", fold=0)
        specs = [dict(unit=unit, cohort="historical"), dict(unit=dict(unit, seed=2172), cohort="historical"),
                 dict(unit=dict(unit, sequence="other"), cohort="expanded")]
        jobs = [dict(spec=spec, source_arm=mode, anchors=[
            dict(frame=3, last=7, roles=[dict(role="neutral", target_frame=3, reference_arm="", reference_event=""),
                                       dict(role="recovery", target_frame=5, reference_arm="ppo", reference_event="1:5")]),
            dict(frame=12, last=12, roles=[dict(role="end", target_frame=12, reference_arm="", reference_event="")])])
                for spec in specs for mode in stage.s103.MODES[1:]]
        original_point, original_collect = stage.point, stage.collect
        for different_card in (False, True):
            with self.subTest(different_card=different_card), tempfile.TemporaryDirectory() as directory:
                upstream, root = Path(directory)/"stage103", Path(directory)/"stage105"
                upstream.mkdir(); root.mkdir()
                plan103 = stage.io.seal(dict(schema=stage.s103.SCHEMA, design=stage.s103.DESIGN))
                for spec in specs:
                    for mode, data in traces.items():
                        stage.c.save(upstream/"evaluation"/f"{stage.c.key(spec['unit'])}__{mode}.npz", data,
                            dict(plan_fingerprint=plan103["fingerprint"], unit=spec["unit"], arm=mode))
                plan = stage.io.seal(dict(schema=stage.SCHEMA, units=jobs, design=stage.DESIGN, runtime={},
                    runtime_differences=[dict(field="runtime.gpu_uuid", expected="old", actual="new")] if different_card else [],
                    native_head_sha256="native", source_sha256={}, code_sha256={}))
                stage.c.publish_identical(root/"manifest.json", plan)
                before = self.observer.calls
                with (patch.object(stage.e63.source_experiment, "installed", side_effect=lambda *_: contextlib.nullcontext("native")),
                      patch.object(stage, "point", side_effect=lambda *a: original_point(*a, image_reader=self.reader)),
                      patch.object(stage, "collect", side_effect=lambda *a: original_collect(*a, image_reader=self.reader)),
                      contextlib.redirect_stdout(io.StringIO())):
                    stage.run(root, upstream, plan103, plan, lambda _: self.entry, lambda _: fixture.choose, self.old, self.observer)
                    self.assertEqual(self.observer.calls-before, stage.budget(jobs)["new_search_calls"])
                    with patch.object(self.observer, "capture", side_effect=AssertionError("Repeated inference")):
                        stage.run(root, upstream, plan103, plan, lambda _: self.entry, lambda _: fixture.choose, self.old, self.observer)
                stage.check_results(root)
                report = stage.c.c66.sealed(root/"analysis/stage105_decision.json", stage.SCHEMA)
                self.assertEqual(report["budget"]["new_search_calls"], 432)
                self.assertEqual(report["budget"]["short_branches"], 48)
                self.assertEqual(report["maximum_keep_source_box_error_px"], 0.)
                next((root/"points").glob("*.npz")).write_bytes(b"tampered")
                with self.assertRaises(ValueError):
                    stage.check_results(root)


if __name__ == "__main__":
    unittest.main()
