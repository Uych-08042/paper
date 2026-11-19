"""Check the gate's causal features and same-window branch accounting."""
import unittest
from pathlib import Path

import numpy as np

import stage135_peak_gate as stage


class PeakGate(unittest.TestCase):
    def test_history_can_favor_the_runner_up(self):
        raw = np.array([[1., .81], [.1, .1]], np.float32)
        boxes = np.array([[30., 10., 10., 10.], [15., 10., 10., 10.]])
        parent = [10., 10., 10., 10.]
        evidence = stage.online_evidence(raw, np.ones_like(raw), boxes, np.array([0, 1]),
                                         parent, [0., 10., 10., 10.])
        self.assertEqual(evidence["runner_history_iou"], 1.)
        self.assertEqual(evidence["native_history_iou"], 0.)
        self.assertTrue(all(stage.gate(evidence).values()))

    def test_fallback_and_nested_conditions(self):
        baseline = dict(distinct=True, relative_score_gap=.20, pair_iou=.20, history_advantage=.10)
        variants = ((dict(distinct=False), [False, False, False]),
                    (dict(relative_score_gap=.21), [False, False, False]),
                    (dict(pair_iou=.21), [True, False, False]),
                    (dict(history_advantage=.09), [True, True, False]))
        for override, expected in variants:
            with self.subTest(override=override):
                self.assertEqual(list(stage.gate(baseline | override).values()), expected)

    def test_gate_uses_only_the_selected_trajectory(self):
        native = dict(sequence="fixture", kind="onset", frame="2", anchor_iou="0", anchor_delta="0",
                      tail_mean_delta="0", tail_low_frame_delta="0", new_tail_good_run="False",
                      lost_tail_good_run="False", new_tail_low="False", first_tail_good_run="",
                      live_native_low_anchor="True")
        runner = dict(native, active="True", anchor_iou="0.6", anchor_delta="0.6", tail_mean_delta="0.4",
                      tail_low_frame_delta="-6", new_tail_good_run="True", first_tail_good_run="3")
        pairs = {"fixture": {stage.NATIVE: native, stage.RUNNER: runner}}
        evidence = {"fixture": dict(distinct=True, relative_score_gap=.10, pair_iou=.10, history_advantage=0.)}
        summary, decisions = stage.evaluate(pairs, evidence)
        by_arm = {r["arm"]: r for r in summary}
        self.assertEqual(by_arm["close_conflict"]["new_tail_good_runs"], 1)
        self.assertEqual(by_arm[stage.PRIMARY]["new_tail_good_runs"], 0)
        self.assertFalse(decisions[0][stage.PRIMARY])

    def test_archived_outcome_bound(self):
        folder = Path(__file__).resolve().parents[1] / "docs/experiments/06_sgla_uav_tracking/results"
        path = folder / "stage134_results_20261119T100654Z_391381.txt"
        summary = stage.bound_from_bundle(path)
        rows = {(r["kind"], r["arm"]): r for r in summary}
        bound = rows["onset", "hindsight_recovery_bound"]
        self.assertEqual((bound["new_tail_good_runs"], bound["lost_tail_good_runs"]), (3, 0))
        self.assertEqual(rows["healthy", "forced_runner_up"]["new_tail_low_cases"], 4)


if __name__ == "__main__":
    unittest.main()
