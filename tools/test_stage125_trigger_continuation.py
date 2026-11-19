"""Test matched starts, seed coverage and causal continuation with a fixture observer."""
import unittest
from unittest.mock import patch

import numpy as np

import stage125_trigger_continuation as stage
import test_stage113_frozen_closed_loop as fixture


class TriggerContinuation(unittest.TestCase):
    setUp = fixture.FrozenClosedLoop.setUp

    def test_same_selected_slot_can_produce_different_next_requests(self):
        boxes = np.array([[0, 0, 4, 4], [30, 0, 4, 4], [8, 0, 4, 4], [12, 0, 4, 4]])
        start = dict(selected=1, candidate_boxes=boxes, scores=np.array([0.46, 0.50, 0.3, 0.2]))
        other = dict(start, scores=np.array([0.36, 0.50, 0.3, 0.2]))
        self.assertEqual(stage.continuation_key(start), (1, True))
        self.assertEqual(stage.continuation_key(other), (1, False))

    def test_control_replays_exactly_and_changed_action_advances_its_own_state(self):
        entry = self.entry
        generator = self.package["generator"]
        control = stage.s121.collect(self.observer, entry["frames"], entry["initial_box"], self.old,
                                     generator, self.model, image_reader=self.reader)
        frame = 4
        history = list(stage.prior.history(control, entry["initial_box"], frame-1, self.ap["design"]["history"]))
        start = stage.snapshot(control, frame-1)
        actual = int(start["selected"])
        replacement = next(q for q in range(4) if not np.array_equal(start["boxes"], start["candidate_boxes"][q]))
        score = np.zeros(4)
        score[replacement] = 1.
        changed = stage.commit(start, score, replacement)
        for selected in (start, changed):
            self.observer.initialize(self.reader(self.ap["root"], entry["frames"][0]), entry["initial_box"])
            with patch.object(stage, "DESIGN", dict(stage.DESIGN, future_frames=4)):
                data = stage.rollout(self.observer, entry["frames"], history, selected, self.ap, generator,
                                     self.model, self.reader, lambda: None)
                stage.audit(data, entry["frames"], history, selected, self.ap, generator, self.model)
            self.assertEqual(int(selected["frame_id"]), frame)
            np.testing.assert_array_equal(data["candidate_parents"][0,0], selected["boxes"])
            if int(selected["selected"])==actual:
                for key in stage.FIELDS:
                    np.testing.assert_array_equal(data[key], control[key][frame:frame+4])

    def test_selection_censors_only_the_tail_and_keeps_model_seeds(self):
        jobs = {("test",2312):dict(frames=100)}
        rows = [dict(sequence="test", seed="2312", model_seed=str(s), frame=str(f), alternate_activation="True",
                     candidate_ious="0|0|0|.6", recorded_action="0", recorded_iou="0", fold="0",
                     new_good_added="True") for s in stage.SEEDS for f in (69,70)]
        chosen = stage.select_points(rows, jobs)
        self.assertEqual([p["full_window"] for p in chosen], [True,False])
        self.assertEqual([p["available_future"] for p in chosen], [31,30])
        self.assertEqual(set(chosen[0]["stage124_predictions"]), {"2312","2313"})

    def test_after_start_iou_and_sustained_good_are_measured_separately(self):
        truth = np.tile([10., 10., 8., 8.], (32, 1))
        start = {"boxes": np.array([50., 50., 8., 8.])}
        boxes = truth[1:].copy()
        boxes[:2] = start["boxes"]
        metrics = stage.outcome(start, {"boxes": boxes}, truth)
        self.assertEqual(metrics["current_iou"], 0.)
        self.assertAlmostEqual(metrics["future_mean_iou"], 29/31)
        self.assertEqual(metrics["first_sustained_good_offset"], 3)
        self.assertEqual(metrics["future_low_frames"], 2)


if __name__ == "__main__":
    unittest.main()
