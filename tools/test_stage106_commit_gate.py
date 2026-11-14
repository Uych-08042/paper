"""Verify online whitelists, group isolation, utility controls and CPU-only cache replay."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage106_commit_gate as stage
import test_stage103_candidate_verification as fixture


def choose(row, history):
    out = fixture.choose(row, history)
    return dict(out, probabilities=np.array([.7, .1, .1, .1]), value=.2)


def sample(group, seed=1, mode="four_matched", frame=1, value=1., target=.4):
    x = np.zeros(len(stage.FEATURES)); x[0] = value
    return dict(cohort="historical" if seed == 1 else "expanded", sequence=group+"_00000_s", group=group,
                seed=seed, source_arm=mode, frame=frame, last=frame+3, roles="unused_offline_role", x=x,
                eligible=True, reason="eligible", future=target, myopic=-target, four_selected=1, pair_selected=0)


class CommitGate(unittest.TestCase):
    setUp = fixture.Verification.setUp

    def test_online_features_past_only_finite_whitelist_and_noop(self):
        parent = fixture.Verification()
        row, queries, anchor, memory = parent.decision_fixture()
        row = dict(row, probabilities=np.array([.1, .2, .6, .1]), value=.4)
        history = [row["history_latest"].copy()]*3
        actual = stage.s103.decide(row, queries, anchor, list(memory), 7, "four_matched", stage.s103.DESIGN)
        pair = stage.s103.decide(row, queries, anchor, list(memory), 7, "pair", stage.s103.DESIGN)
        args = (row, queries, anchor, history, memory, 7, "four_matched", actual, pair, stage.s103.DESIGN)
        x, eligible, reason = stage.online_features(*args)
        self.assertEqual(x.shape, (49,)); self.assertTrue(np.isfinite(x).all())
        self.assertTrue(eligible); self.assertEqual(reason, "eligible")
        changed = dict(row, ground_truth=np.full((4, 4), 999.), role="benefit", future_iou=999., sequence="leak")
        np.testing.assert_array_equal(x, stage.online_features(changed, *args[1:])[0])
        no_op = stage.online_features(row, queries, anchor, history, memory, 7, "four_matched", actual, actual, stage.s103.DESIGN)
        self.assertFalse(no_op[1]); self.assertEqual(no_op[2], "no_commit_change")
        # Unselected unused query may be NaN; only actually required evidence is mandatory.
        partial = queries.copy(); partial[3] = np.nan
        self.assertTrue(stage.online_features(row, partial, *args[2:])[1])
        bad = queries.copy(); bad[actual["selected"]] = np.nan
        self.assertFalse(stage.online_features(row, bad, *args[2:])[1])
        bad_prob = dict(row, probabilities=np.full(4, np.nan))
        self.assertFalse(stage.online_features(bad_prob, *args[1:])[1])
        with self.assertRaises(ValueError):
            stage.online_features(row, queries, anchor, history, memory+[(7, anchor)], *args[5:])

    def test_source_group_exclusion_and_fit_only_scaling(self):
        rows = [sample("uav0000001", seed=1, value=-1., target=-.4),
                sample("uav0000002", seed=2, value=1., target=.4),
                sample("uav0000003", seed=1, value=900., target=.9),
                sample("uav0000003", seed=2, mode="four_pairwise", value=-800., target=-.9)]
        # Two clips of one source still share one exclusion group.
        rows[-1]["sequence"] = "uav0000003_12345_s"
        models = {h: stage.fit_gate(rows, "uav0000003", h) for h in stage.HEADS}
        for model in models.values():
            self.assertEqual(model["fit_groups"], ["uav0000001", "uav0000002"])
            self.assertEqual(model["fit_rows"], 2); self.assertEqual(model["center"][0], 0.)
            self.assertEqual(model["scale"][0], 1.); self.assertEqual(model["scale"][1], 1.)
        changed = copy.deepcopy(rows)
        for r in changed[2:]:
            r.update(future=-100., myopic=100., roles="other")
            r["x"][:] = 1e6
        for head in stage.HEADS:
            self.assertEqual(models[head], stage.fit_gate(changed, "uav0000003", head))
        # Opposite targets use exactly matched rows/weights but opposite learned decisions.
        plus = rows[1]["x"]
        self.assertTrue(stage.predict(models["future"], plus, True)["commit"])
        self.assertFalse(stage.predict(models["myopic"], plus, True)["commit"])
        self.assertFalse(stage.predict(models["future"], plus, False)["commit"])
        fallback = stage.fit_gate(rows[2:], "uav0000003", "future")
        self.assertFalse(fallback["supported"])
        self.assertFalse(stage.predict(fallback, plus, True)["commit"])
        # Missing future is not imputed to zero, and both heads use common support.
        missing = sample("uav0000004", value=100., target=1.)
        missing["future"] = np.nan
        for head in stage.HEADS:
            self.assertEqual(models[head], stage.fit_gate(rows+[missing], "uav0000003", head))

    def test_group_seed_context_weights_and_expected_rate_control(self):
        rows = [sample("uav0000001", frame=i, value=1., target=.5) for i in range(10)]
        rows += [sample("uav0000001", seed=2, value=-1., target=-.5),
                 sample("uav0000001", seed=2, mode="four_pairwise", value=-1., target=-.5),
                 sample("uav0000002", value=1., target=.5)]
        w = stage.balanced_weights(rows)
        self.assertAlmostEqual(w[:10].sum(), .25)
        self.assertAlmostEqual(w[10], .125); self.assertAlmostEqual(w[11], .125)
        self.assertAlmostEqual(w[12], .5); self.assertAlmostEqual(w.sum(), 1.)
        test = [sample("uav0000003", frame=1, value=1., target=.5),
                sample("uav0000003", frame=2, value=-1., target=-.5)]
        models = {("uav0000003", h): stage.fit_gate(rows, "uav0000003", h) for h in stage.HEADS}
        decisions = stage.matched_decisions(test, models)
        bank = {p: [r for r in decisions if r["policy"] == p] for p in stage.POLICIES}
        self.assertEqual([r["commit_probability"] for r in bank["future"]], [1., 0.])
        self.assertEqual([r["commit_probability"] for r in bank["rate_expected"]], [.5, .5])
        self.assertAlmostEqual(stage.mean(bank["future"], "utility"), .25)
        self.assertAlmostEqual(stage.mean(bank["rate_expected"], "utility"), 0.)
        self.assertAlmostEqual(stage.mean(bank["future_oracle"], "utility"), .25)
        self.assertAlmostEqual(stage.mean(bank["future"], "oracle_regret"), 0.)
        summary, paired = stage.summarize(decisions, stage.POLICIES, ("utility", "commit_probability"))
        self.assertAlmostEqual(next(r for r in paired if r["cohort"] == "all" and r["source_arm"] == "all" and r["contrast"] == "future_minus_myopic")["mean_delta"], .5)
        # Repeating anchors in one context does not change another group's macro weight.
        repeated = [dict(r, utility=1.) for r in bank["future"]]*10
        repeated += [dict(bank["future"][0], group="uav0000004", sequence="uav0000004_00000_s", utility=-1.)]
        self.assertAlmostEqual(np.mean(list(stage.macro(repeated, "utility").values())), 0.)

    def make_sources(self, root105, root103):
        old, p103 = self.old, dict(schema=stage.s103.SCHEMA, design=stage.s103.DESIGN, fingerprint="source103", units=[])
        jobs, entries = [], {}
        for index, seed in ((1, 1), (2, 2)):
            sequence = f"uav{index:07d}_00000_s"
            entry = dict(self.entry, sequence=sequence)
            unit = dict(sequence=sequence, seed=seed, fold=0, origin="fixture", policy="four_ppo")
            spec = dict(unit=unit, cohort="historical" if seed == 1 else "expanded")
            entries[sequence] = entry; p103["units"].append(spec)
            for mode in stage.s103.MODES[1:]:
                trace = stage.s103.collect(self.observer, entry, old, choose, mode, p103["design"], image_reader=self.reader)
                stage.c.save(root103/"evaluation"/(stage.c.key(unit)+"__"+mode+".npz"), trace,
                             dict(plan_fingerprint=p103["fingerprint"], unit=unit, arm=mode))
                case = dict(frame=3, last=9, roles=[dict(role="offline_fixture")])
                jobs.append(dict(spec=spec, source_arm=mode, anchors=[case]))
        p105 = dict(schema=stage.s105.SCHEMA, fingerprint="source105", units=jobs)
        for job in jobs:
            spec, mode = job["spec"], job["source_arm"]; unit = spec["unit"]; entry = entries[unit["sequence"]]
            trace = stage.s105.prior.required_trace(root103, p103, unit, mode)
            stem = f"{stage.c.key(unit)}__{mode}__f000003"
            meta = dict(plan_fingerprint=p105["fingerprint"], unit=unit, source_arm=mode, first=3, last=9)
            shared = stage.s105.point(self.observer, trace, entry, 3, old, choose, p103["design"], image_reader=self.reader)
            stage.c.save(root105/"points"/(stem+".npz"), shared, dict(meta, kind="point"))
            for treatment in ("keep", "both_pair"):
                data = stage.s105.collect(self.observer, trace, shared, entry, 3, 9, treatment, mode, old, choose, p103["design"], image_reader=self.reader)
                stage.c.save(root105/"branches"/(stem+"__"+treatment+".npz"), data, dict(meta, kind="branch", treatment=treatment))
        return p105, p103, lambda spec: entries[spec["unit"]["sequence"]]

    def test_cpu_pipeline_resume_receipts_and_frozen_trace_exposure(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            base = Path(temporary); root105, root103, root = base/"source105", base/"source103", base/"run"
            root.mkdir()
            p105, p103, load = self.make_sources(root105, root103)
            plan = stage.io.seal(dict(schema=stage.SCHEMA, design=stage.DESIGN, features=list(stage.FEATURES),
                groups=["uav0000001", "uav0000002"], anchors=4, exposure_contexts=4, exposure_states=48,
                source_sha256={}, code_sha256={}))
            stage.c.publish_identical(root/"manifest.json", plan)
            before = self.observer.calls
            with patch.object(stage.s103, "observe", side_effect=AssertionError("No GPU inference permitted")), \
                 patch.object(stage.s105, "checked_image", side_effect=AssertionError("No image reads permitted")):
                stage.run(root, root105, root103, p105, p103, plan, load, lambda unit: choose, self.old)
                self.assertEqual(self.observer.calls, before)
                report = stage.c.c66.sealed(root/"analysis/stage106_decision.json", stage.SCHEMA)
                self.assertEqual(report["exposure_states"], 48); self.assertEqual(report["new_gate_models"], 4)
                self.assertEqual(report["new_search_calls"], 0)
                # Full pipeline repeat uses the sealed model/cache and produces identical artifacts.
                stage.run(root, root105, root103, p105, p103, plan, load, lambda unit: choose, self.old)
            # GT can change diagnostic labels, never feature/decision/exposure-memory arrays.
            spec = p103["units"][0]; mode = "four_matched"
            trace = stage.s105.prior.required_trace(root103, p103, spec["unit"], mode)
            rows = [sample("uav0000002", value=1., target=.3)]
            models = {("uav0000001", h): stage.fit_gate(rows, "uav0000001", h) for h in stage.HEADS}
            original = stage.exposure(trace, load(spec), spec, mode, choose, self.old, p103["design"], models)
            altered = stage.exposure(trace, dict(load(spec), ground_truth=np.full((13, 4), 999.)), spec, mode, choose, self.old, p103["design"], models)
            for key in original:
                if key != "offline_current_commit_delta":
                    np.testing.assert_array_equal(original[key], altered[key])
            # A source/cache corruption must be detected, not silently resumed.
            path = root/"models/uav0000001__future.json"
            original_bytes = path.read_bytes()
            path.write_text(path.read_text(encoding="utf8").replace('"intercept": 0.0', '"intercept": 0.1'), encoding="utf8")
            with self.assertRaises(ValueError):
                stage.check_results(root)
            path.write_bytes(original_bytes)
            stage.check_results(root)
            exposure_file = next((root/"exposure").glob("*.npz"))
            with exposure_file.open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaises(ValueError):
                stage.check_results(root)


if __name__ == "__main__":
    unittest.main()
