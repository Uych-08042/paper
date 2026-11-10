"""Verify read suppression, cached replay and recovery alignment without GPU."""
import unittest

import numpy as np

import stage94_recovery_audit as stage


def fixture():
    design = stage.s93.DESIGN
    good = np.array([40., 40., 20., 20.])
    bad = np.array([48., 48., 4., 4.])
    anchor = np.array([0., 1.])
    count = 24
    features = np.zeros((4, len(stage.s93.s76.branches.prior.FEATURES)))
    features[:, stage.s93.s76.branches.prior.FEATURES.index("hann_peak")] = .5
    memory, rows = [], []
    for frame in range(1, count + 1):
        before = frame <= 4
        first, second = (bad, good) if before else (good, bad)
        boxes = np.stack([first, second, first, second])
        query = np.array([[1., 0.], [1., 0.]]) if before else np.array([[0., 1.], [1., 0.]])
        row = dict(candidate_boxes=boxes, candidate_parents=np.tile(good, (4, 1)),
                   history_latest=good, features=features, selected=0)
        result = stage.s93.choose_write(row, query, anchor, memory, frame, "gated", design)
        rows.append(dict(row | result, boxes=boxes[result["selected"]], queries=query, frame_id=frame,
                         crop=np.ones((4, 4)), crop_sha256=["gated"] * 4))
    gated = {k: np.asarray([r[k] for r in rows]) for k in rows[0]} | dict(anchor=anchor)
    ppo = dict(frame_id=gated["frame_id"], boxes=np.stack([bad] * 4 + [good] * (count - 4)),
               candidate_boxes=gated["candidate_boxes"], selected=np.zeros(count, dtype=int),
               crop=np.zeros((count, 4, 4)), crop_sha256=np.full((count, 4), "ppo"))
    return ppo, gated, np.tile(good, (count, 1))


class Replay(unittest.TestCase):
    def test_recovery_and_single_read_ablation(self):
        ppo, gated, truth = fixture()
        spec = dict(cohort="historical", unit=dict(sequence="test", seed=2171))
        events, frames, drops = stage.audit_unit(spec, ppo, gated, truth, stage.s93.DESIGN)
        event, = events
        self.assertEqual((event["ppo_recovery_frame"], event["outcome"]), (5, "missed"))
        self.assertEqual(event["first_harmful_switch_in_window"], 5)
        self.assertEqual(event["first_template_low_to_good"], 5)
        self.assertEqual(event["first_drop_low_to_good"], 5)
        self.assertAlmostEqual(event["own_iou_gap"], event["same_state_choice_gap"] + event["state_associated_gap"])
        self.assertEqual(frames[4]["refs"], "4|1")
        self.assertEqual([r["dropped_write_frame"] for r in drops if r["frame"] == 5 and r["low_to_good"]], [4, 1])

    def test_replay_rejects_tampered_reference(self):
        _, gated, _ = fixture()
        gated["refs"][4, 0] = 5
        with self.assertRaises(AssertionError):
            stage.memory_replay(gated, 5, stage.s93.DESIGN)


if __name__ == "__main__":
    unittest.main()
