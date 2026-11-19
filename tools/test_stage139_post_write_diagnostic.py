"""Exercise relapse selection, matched-parent replay and failure decomposition."""
from unittest import TestCase, main
from unittest.mock import Mock

import numpy as np

import stage139_post_write_diagnostic as stage
from stage27_uav_common import crop_geometry, scene_geometry


class PostWriteDiagnostic(TestCase):
    def setUp(self):
        self.parent = np.array([40., 40., 16., 16.])
        self.truth = np.array([8., 8., 16., 16.])
        self.hw = (128, 128)
        self.ap = dict(root=None, window=np.ones((4, 4), np.float32), search_factor=4., search_size=64,
                       design={"box_atol": 1e-6})
        self.raw = np.full((4, 4), .01, np.float32)
        self.raw[2, 2] = .9
        self.size = np.full((2, 4, 4), .25, np.float32)
        self.shifts = np.zeros_like(self.size)

    def test_followup_index_first_relapse_after_confirmed_good_run(self):
        tail = [.1] * 3 + [.6] * 3 + [.3] * 3 + [.1] * 23
        self.assertEqual(stage.relapse_offset(tail), 10)
        case, start = stage.RELAPSE_CASE
        cases = [dict(case_id="target", sequence=case, frame=start, successors=32)] + [
            dict(case_id=f"control{i}", sequence=f"control{i}", frame=1, successors=32) for i in range(5)]
        table = [dict(case_id=r["case_id"], arm=arm, offset=j, iou=v)
                 for r in cases for arm in stage.ARMS for j, v in enumerate([0.] + tail)]
        result, _ = stage.replay_plan(cases, table)
        self.assertEqual(result[0]["offsets"], list(range(1, 9)) + [9, 10, 11])
        self.assertEqual(sum(len(r["offsets"]) for r in result) * len(stage.ARMS), 255)

    def test_crop_limited_and_covered_no_supply_and_rank_miss(self):
        for parent, truth, expected in (
                (self.parent, self.truth, "good_grid_not_selected"),
                (self.parent, np.array([16., 16., 35., 35.]), "covered_no_good_grid"),
                (self.parent, np.array([100., 100., 16., 16.]), "limited_crop_no_good_grid"),
                (self.parent, self.parent, "output_good")):
            with self.subTest(expected=expected):
                geo = scene_geometry(parent, truth, self.hw, 4., 64)
                row = stage.diagnose(self.raw, self.size, self.shifts, parent, 1., self.hw, parent, truth, geo, self.ap)
                self.assertEqual(row["mechanism"], expected)

    def test_replay_shares_identical_frozen_parents_and_scores_their_crop(self):
        arm_count = len(stage.ARMS)
        trace = dict(parent=np.tile(self.parent, (arm_count, 2, 1)), boxes=np.tile(self.parent, (arm_count, 3, 1)),
                     crop=np.tile(crop_geometry(self.parent, self.hw, 4., 64)[0], (arm_count, 2, 1)),
                     crop_sha256=np.full((arm_count, 2), "matching"), layer=np.full((arm_count, 2), 7))
        frames = [dict(height=self.hw[0], width=self.hw[1]) for _ in range(4)]
        covered_truth = np.array([24., 24., 16., 16.])
        entry = dict(frames=frames, initial_box=self.parent, ground_truth=np.tile(covered_truth, (4, 1)))
        case = dict(case_id="test", sequence="test", kind="onset", frame=1, successors=2, offsets=[1, 2], relapse_offsets=[])
        indexed = {(case["case_id"], arm): {1: 0., 2: 0.} for arm in stage.ARMS}

        def observe(_observer, _image, parent, _frame, _ap):
            crop, _, factor = crop_geometry(parent, self.hw, 4., 64)
            return dict(boxes=parent.copy(), parent=parent.copy(), crop=crop, crop_sha256="matching",
                        raw=self.raw, size=self.size, offset=self.shifts, image_hw=self.hw, factor=factor,
                        head_calls=1, selector_calls=1, block_calls=[1] * 7 + [0] * 5, layer=7)

        forward = Mock(side_effect=observe)
        data = stage.collect(Mock(), entry, case, trace, self.ap, image_reader=lambda _root, item: item, observe=forward)
        self.assertEqual(forward.call_count, 2)
        self.assertEqual(stage.verify_replay(data, case, trace, True, 1e-6), (0., 2))
        rows, coverage = stage.score_case(data, trace, entry, case, self.ap, indexed)
        self.assertEqual(len(rows), 2 * arm_count)
        self.assertEqual({r["mechanism"] for r in rows}, {"good_grid_not_selected"})
        self.assertEqual([r["low_adequately_covered"] for r in coverage], [2] * arm_count)


if __name__ == "__main__":
    main()
