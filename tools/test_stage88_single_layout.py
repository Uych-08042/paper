"""Test one-layout action selection and continuation with the CPU fixture."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import stage88_single_layout as stage
import test_stage75_ppo_failure as fixture


class Signals(unittest.TestCase):
    def test_challenger_uses_only_unique_fixed_candidates(self):
        boxes = np.array([[[0., 0., 10., 10.], [0., 0., 10., 10.], [12., 0., 10., 10.], [6., 0., 10., 10.]]])
        features = np.zeros((1, 4, len(stage.s76.branches.prior.FEATURES)))
        features[0, :, stage.s76.branches.prior.FEATURES.index("hann_peak")] = [.3, 1., .8, .8]
        parents = boxes.copy()
        parents[0, 2, 0] = -6
        fixed = dict(candidate_boxes=boxes, selected=[0], features=features,
                     history_latest=boxes[:, 0], candidate_parents=parents)
        action, signals = stage.challenger(fixed)
        self.assertEqual(action, 2)
        self.assertAlmostEqual(signals["hann"], .5)
        self.assertAlmostEqual(signals["velocity"], 0.)
        self.assertAlmostEqual(signals["consensus"], 0.)
        fixed["candidate_boxes"] = np.repeat(boxes[:, :1], 4, axis=1)
        self.assertEqual(stage.challenger(fixed), (None, {}))

    def test_rank_support_requires_both_signs_across_source_groups(self):
        rows = []
        for stratum in stage.s86.STRATA:
            for i in range(6):
                positive = i < (3 if stratum == "high" else 1)
                gain = .1 if positive else -.1
                rows.append(dict(stratum=stratum, sequence=f"uav{i:07d}_00000_s", case=f"{stratum}_{i}",
                                 challenger_action=1, anchor_iou=.1, ppo_return=not positive, challenger_return=positive,
                                 followup_gain=gain, ppo_low_iou=.2, challenger_low_iou=.2,
                                 **{name + "_margin": gain for name in stage.SIGNALS}))
        _, support = stage.summarize(rows, dict(stage.s86.DESIGN, **stage.DESIGN))
        high, low = [next(r for r in support if r["stratum"] == s and r["method"] == "hann")
                     for s in stage.s86.STRATA]
        self.assertEqual((high["rank_auc"], high["helpful_switches"], high["ppo_only_recoveries_lost"]), (1., 3, 0))
        self.assertIsNone(low["rank_auc"])


@unittest.skipUnless(fixture.torch is not None and fixture.cv2 is not None, "Need CPU Torch and OpenCV")
class Replay(unittest.TestCase):
    def test_resume_new_tail_and_reuse_natural_tail(self):
        from stage48_stability_runtime import digest
        from test_stage61_search import TorchTests

        fixture.torch.set_num_threads(1)
        s86, c = stage.s86, stage.c
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            original, native, old_observer, _ = TorchTests().fixture(root)
            old_observer.handle.remove()
            old_observer.pre_handle.remove()
            observer = stage.e63.Capture(native)
            self.addCleanup(observer.handle.remove)
            old = dict(original["plan"], actor_design=dict(stage.e63.source_common.DESIGN))
            entry = dict(old["entries"][0], sequence="uav0000126_07915_s")
            ap = stage.e74.e67.actor_plan(old)
            prefix = np.array([entry["initial_box"], np.asarray(entry["initial_box"]) + [24., 0., 0., 0.]])
            frames = np.array([2, 3, 4])
            unit = dict(sequence=entry["sequence"], origin="native", fold=0, policy="ppo", seed=2171)

            def choose(row, history, *_):
                return dict(row, selected=0, boxes=row["candidate_boxes"][0].copy())

            arms = [s86.collect(observer, entry, prefix, frames, mix, ap, choose) for mix in (0., .5)]
            inputs = [(dict(case=s, stratum=s, unit=unit, anchor=1), entry, frames, prefix, arms, stage.s87.cross(arms))
                      for s in s86.STRATA]
            with patch.object(stage.s87, "cases", return_value=iter(inputs)):
                planned = list(stage.planned_cases(root, {}, {}))
            self.assertTrue(all(item["ref"] == "new" for item in planned))
            report = dict(native_head_sha256=digest(native.rt.net.box_head))
            source, actor = dict(original=old, report=report), dict(models={}, prior={})
            output = root / "stage88"
            output.mkdir()
            plan = stage.io.seal(dict(schema=stage.SCHEMA, windows=2, new=list(s86.STRATA),
                design=dict(s86.DESIGN, horizon=3) | stage.DESIGN, source_stage87_fingerprint="fixture",
                source_sha256={}, code_sha256={}))
            c.publish_identical(output / "manifest.json", plan)
            with (patch.object(stage.e74, "choose", side_effect=choose), patch.object(stage.e74, "learned_pack"),
                  contextlib.redirect_stdout(io.StringIO())):
                stage.execute(output, planned, root, {}, plan, source, actor, observer)
                stage.check_results(output)
                decision = c.c66.sealed(output / "analysis/stage88_decision.json", stage.SCHEMA)
                self.assertEqual((decision["new_continuations"], decision["search_calls"]), (2, 24))
                with patch.object(observer, "capture", side_effect=AssertionError("Cached inference repeated")):
                    stage.execute(output, planned, root, {}, plan, source, actor, observer)
                    tail, path = stage.resolve_tail(dict(planned[0], ref="ff"), root, {}, output, plan, source, None, None)
                    self.assertIsNone(path)
                    np.testing.assert_array_equal(tail["boxes"], arms[0]["boxes"][1:])
                    cache87 = root / "stage87" / "collection" / "high__fm.npz"
                    c.save(cache87, tail, dict(plan_fingerprint="fixture87", branch="high__fm"))
                    reused, path = stage.resolve_tail(dict(planned[0], ref="fm", box=arms[0]["boxes"][0]),
                        cache87.parents[1], dict(fingerprint="fixture87"), output, plan, source, None, None)
                    self.assertIsNone(path)
                    np.testing.assert_array_equal(reused["boxes"], tail["boxes"])


if __name__ == "__main__":
    unittest.main()
