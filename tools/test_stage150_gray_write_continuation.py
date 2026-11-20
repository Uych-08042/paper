"""Check that a grayscale write changes only the first parent of its own branch."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np

import stage150_gray_write_continuation as stage
from test_stage143_fixed_pool_quality import fixture


class GrayWriteContinuation(unittest.TestCase):
    def setUp(self):
        raw, self.size, self.offset, parent, _, self.hw, _, _, self.ap = fixture([40., 40., 16., 16.])
        self.raw = raw
        self.ap.update(root="fixture", search_factor=4.)
        self.case = dict(case_id="fixture", sequence="uav0000003_00000_s", source="uav0000003",
                         frame=1, onset=True, cells=[10, 5], margin=.03,
                         anchor_boxes=[parent.tolist(), (parent + [-16., -16., 0., 0.]).tolist()],
                         anchor_ious=[1., 0.], anchor_crop_sha256="shared", successors=3)
        count = len(stage.fixed.FEATURES)
        self.model = dict(mean=np.zeros(count), scale=np.ones(count), coef=np.r_[-1., np.zeros(count-1)], intercept=0.)

    def test_continuations_use_regression_and_their_own_write(self):
        reads = []

        def read_image(root, frame):
            reads.append(frame)
            return frame

        def observe(observer, image, parent, frame, ap):
            self.assertIs(observer, self)
            self.assertEqual(image, frame)
            crop, _, factor = stage.crop_geometry(parent, self.hw, ap["search_factor"], ap["search_size"])
            box = stage.fixed.dense_boxes(self.size, self.offset, parent, factor, ap["search_size"], self.hw)[self.raw.argmax()]
            return dict(raw=self.raw, size=self.size, offset=self.offset, parent=parent, boxes=box, crop=crop,
                        factor=factor, image_hw=self.hw, crop_sha256=str(crop), layer=9, head_calls=1,
                        selector_calls=1, block_calls=np.array([1]*6 + [0, 0, 1, 0, 0, 0]))

        data = stage.collect(self, self.case, list(range(5)), [40., 40., 16., 16.], self.model, self.ap,
                             read_image=read_image, observe=observe)
        self.assertEqual(reads, [0, 2, 3, 4])
        stage.audit_branch(data, self.case)
        self.assertTrue((data["cell"] != int(self.raw.argmax())).all())
        self.assertFalse(np.array_equal(data["boxes"][0, 0], data["boxes"][1, 0]))
        data["parent"][1, 1] = data["boxes"][0, 0]
        with self.assertRaises(AssertionError):
            stage.audit_branch(data, self.case)

    def initialize(self, image, box):
        self.assertEqual(image, 0)
        self.assertEqual(box, [40., 40., 16., 16.])

    def test_terminal_anchor_has_no_inference_or_successor_metric(self):
        case = dict(self.case, successors=0)
        data = stage.collect(None, case, [0, 1], None, self.model, self.ap, read_image=None, observe=None)
        stage.audit_branch(data, case)
        windows, timeline = stage.describe(data, case, None)
        self.assertEqual(len(timeline), 2)
        self.assertEqual([r["head_calls"] for r in timeline], [0, 0])
        self.assertIsNone(windows[0]["delta_successor_iou"])
        self.assertEqual(list(stage.summarize(windows))[0]["anchor_only_cases"], 1)

    def test_select_all_calibrated_switches_and_reject_a_duplicate(self):
        sequence = self.case["sequence"]
        cue = stage.CUE
        row = dict(sequence=sequence, source=self.case["source"], frame="1", threshold_mode="calibrated",
                   anchor_cell="10", dense_quality_cell="10", anchor_iou=".4", matched_cells="5", onset="False",
                   state="native", **{f"dense_{cue}_cell":"5", f"dense_{cue}_iou":".6", f"dense_{cue}_switch":"True",
                                      f"{cue}_threshold":".025", f"{cue}_margin":".03"})
        rows = [dict(row, frame=str(i + 1)) for i in range(stage.DESIGN["cases"])]
        selected = stage.switches(rows + [dict(row, threshold_mode="zero_threshold")])
        self.assertEqual(len(selected[sequence]), stage.DESIGN["cases"])
        with self.assertRaisesRegex(ValueError, "Unexpected Stage-149 switch"):
            stage.switches(rows + [row])

    def test_metric_does_not_bridge_invalid_labels(self):
        case = dict(self.case, successors=4)
        boxes = np.array([[20., 20., 16., 16.]]*4)
        data = dict(frame_id=np.arange(2, 6), boxes=np.stack([boxes, boxes]), cell=np.full((2, 4), 10),
                    crop_sha256=np.full((2, 4), "shared"), head_calls=np.ones((2, 4), int),
                    selector_calls=np.ones((2, 4), int), block_calls=np.ones((2, 4, 12), int))
        truth = np.vstack([boxes[:2], boxes])
        truth[3, 2] = 0.
        windows, _ = stage.describe(data, case, truth)
        self.assertEqual(windows[0]["valid_successors"], 3)
        self.assertIsNone(windows[0]["regression_first_good_run"])
        self.assertEqual(windows[0]["delta_successor_iou"], 0.)

    def test_terminal_cases_publish_and_verify_with_zero_searches(self):
        cases = [dict(self.case, case_id=f"fixture_{i}", successors=0) for i in range(stage.DESIGN["cases"])]
        plan = stage.io.seal(dict(schema=stage.SCHEMA, design=stage.DESIGN, cases=cases,
                                  source_stage149=stage.SOURCE149, search_budget=0,
                                  runtime={}, stage121_runtime_differences=[]))
        entry = dict(frames=[0, 1], initial_box=self.case["anchor_boxes"][0], ground_truth=[[40., 40., 16., 16.]]*2)
        observer = Mock(spec=stage.base.e63.Capture)
        with TemporaryDirectory() as temp, patch("builtins.print"):
            root = Path(temp)
            stage.c.publish_identical(root / "manifest.json", plan)
            stage.run(root, plan, {self.case["sequence"]:(entry, self.model)}, observer, self.ap)
            stage.check_results(root)
            observer.initialize.assert_not_called()
            self.assertEqual(len(list((root / "branches").glob("*.receipt.json"))), len(cases))


if __name__ == "__main__":
    unittest.main()
