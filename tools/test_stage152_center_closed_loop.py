"""Ensure causal gray acceptance and independent search-state feedback."""
import unittest
from unittest.mock import Mock, patch

import numpy as np

import stage152_center_closed_loop as stage


class CenterClosedLoop(unittest.TestCase):
    def test_online_menu_agrees_with_stage149_and_requires_recent_color(self):
        row = dict(frame=4, boxes=np.array([[0., 0., 16., 16.], [30., 30., 16., 16.]]),
                   cells=np.array([10, 11]), pools=dict(dense=np.array([0, 1])),
                   x=np.zeros((2, 9)), usable=np.array([0., 0., 64., 64.]))
        model = dict(mean=np.zeros(9), scale=np.ones(9), coef=np.zeros(9), intercept=0.)
        offline_cues = dict(template_ncc=np.array([.2, .8]), recent_hist=np.array([.5, .5]))
        history = {1:dict(frame=1, descriptor={"hist":np.array([.5])})}
        offline = stage.ablation.evidence((row, {}, offline_cues), model)
        with (
            patch.object(stage.appearance, "recent_rows", return_value=list(history.values())),
            patch.object(stage.appearance, "candidate_descriptor"),
            patch.object(stage.appearance, "similarities", side_effect=[
                {cue:v[i] for cue,v in offline_cues.items()} for i in range(2)]),
        ):
            a, b, margin = stage.gray_proposal(row, np.zeros(2), None, None, history)
        self.assertEqual((a,b), (offline["anchor"], int(offline["eligible"][0])))
        self.assertEqual(margin, offline["margins"]["template_ncc_only"][0])
        offline_cues["recent_hist"][1] = np.nan
        with (
            patch.object(stage.appearance, "recent_rows", return_value=list(history.values())),
            patch.object(stage.appearance, "candidate_descriptor"),
            patch.object(stage.appearance, "similarities", side_effect=[
                {cue:v[i] for cue,v in offline_cues.items()} for i in range(2)]),
        ):
            self.assertEqual(stage.gray_proposal(row, np.zeros(2), None, None, history), (0, 0, 0.))

    def test_writes_same_output_but_changes_next_crop_parent(self):
        frames, initial = [0, 1, 2], [24., 24., 16., 16.]
        image = np.zeros((64,64,3), np.uint8)
        model = dict(mean=np.zeros(9), scale=np.ones(9), coef=np.zeros(9), intercept=0.)
        inputs = dict(frame=1, cells=np.array([0, 1]), x=np.zeros((2, 9)))
        grid = np.array([[24., 24., 16., 16.], [20., 16., 32., 24.]])

        def observe(observer, pixels, parent, frame, ap):
            return dict(parent=parent.copy(), usable=np.array([0.,0.,64.,64.]),
                        crop_sha256=str(parent), layer=9, head_calls=1, selector_calls=1,
                        block_calls=np.array([1]*6+[0,0,1,0,0,0]))

        traces = {}
        with (
            patch.object(stage.prior, "decode_search", side_effect=lambda *_:(dict(inputs), grid, 0, 0)),
            patch.object(stage.appearance, "trusted", return_value=False),
            patch.object(stage, "gray_proposal", return_value=(0, 1, .04)),
        ):
            for arm in stage.ARMS:
                traces[arm] = stage.collect(Mock(), frames, initial, model, {"root":"fixture"}, arm,
                                            reader=lambda root,f:image, observe=observe)
                stage.audit(traces[arm], 2, initial, arm)
        stage.audit_paired_prefix(traces)
        full, center = (traces[k] for k in ("gray_full", "gray_center"))
        np.testing.assert_array_equal(full["boxes"], center["boxes"])
        np.testing.assert_array_equal(full["parent"][1], grid[1])
        np.testing.assert_array_equal(center["parent"][1], [28., 20., 16., 16.])
        self.assertTrue(traces["native"]["shadow_accepted"].all())
        self.assertFalse(traces["native"]["switched"].any())


if __name__ == "__main__":
    unittest.main()
