"""Exercise frozen frame selection, geometry crossing and result publication."""
import csv
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

import stage140_crop_center_size_cross as stage
from stage27_uav_common import crop_geometry
from stage28_response_common import dense_boxes


class CropCenterSizeCross(unittest.TestCase):
    def test_real_source_points_and_primary_endpoint_values(self):
        p = Path(__file__).resolve().parents[1] / "docs/experiments/06_sgla_uav_tracking/results"
        payload = (p / "stage139_results_20261119T140133Z_454742.txt").read_text(encoding="utf-8")
        part = payload.split("===== stage139_replay.csv =====\n", 1)[1].split("\n===== ", 1)[0]
        rows = list(csv.DictReader(io.StringIO(part)))
        cases = {}
        for row in rows:
            offset = int(row["offset"])
            case = cases.setdefault(row["case_id"], dict(sequence=row["sequence"], case_id=row["case_id"],
                                                        frame=int(row["frame"]) - offset, offsets=set()))
            case["offsets"].add(offset)
        plan = {"cases": [dict(r, offsets=sorted(r["offsets"])) for r in cases.values()]}
        points = stage.freeze_points(plan)
        self.assertEqual(len(points), 11)
        self.assertEqual((points[0]["group"], points[0]["frame"]), ("primary", 243))
        r = {x["arm"]: x for x in rows if (x["sequence"], int(x["frame"])) == stage.PRIMARY}
        self.assertAlmostEqual(float(r["native"]["dense_best_iou"]), .5087, places=4)
        self.assertAlmostEqual(float(r["cross_layer_top1_oracle"]["dense_best_iou"]), .3205, places=4)

    def test_crossing_preserves_endpoints_and_controls_geometry(self):
        native = np.array([40., 40., 16., 16.])
        cross = np.array([36., 20., 30., 30.])
        boxes = stage.cross_parents(native, cross)
        np.testing.assert_array_equal(boxes[[0, 3]], [native, cross])
        centers = boxes[:, :2] + .5 * boxes[:, 2:]
        np.testing.assert_array_equal(centers[0], centers[2])
        np.testing.assert_array_equal(centers[1], centers[3])
        np.testing.assert_array_equal(boxes[0, 2:], boxes[1, 2:])
        np.testing.assert_array_equal(boxes[2, 2:], boxes[3, 2:])
        np.testing.assert_array_equal(stage.unique_mask(stage.cross_parents(native, native)), [True, False, False, False])
        rows = [dict.fromkeys(stage.METRICS, v) for v in (.5, .25, .75, 0.)]
        row = stage.effects(rows, {})[0]
        self.assertEqual((row["restore_native_center"], row["restore_native_size"], row["interaction"]), (.75, .25, -.50))

    def test_frozen_endpoints_through_published_replay(self):
        hw = (128, 128)
        native, cross = np.array([40., 40., 16., 16.]), np.array([36., 20., 30., 30.])
        ap = dict(root=None, window=np.ones((4, 4), np.float32), search_factor=4., search_size=64,
                  design={"box_atol": 1e-6})
        point = dict(point_id="point", case_id="case", sequence="fixture", frame=1, ordinal=0, offset=1,
                     cross_arm="cross_layer_top1_oracle", group="primary")
        entry = dict(sequence="fixture", frames=[None, None], ground_truth=[native, native], initial_box=native)
        plan = stage.io.seal(dict(schema=stage.SCHEMA, points=[point], runtime={}, runtime_differences=[]))

        def observe(_obs, _image, parent, _frame, _plan):
            crop, _, factor = crop_geometry(parent, hw, 4., 64)
            raw = np.full((4, 4), .05, np.float32)
            raw[2, 2] = .9
            size = np.full((2, 4, 4), .25, np.float32)
            offset = np.zeros_like(size)
            box = dense_boxes(size, offset, parent, factor, 64, hw)[10]
            return dict(boxes=box, parent=parent, crop=crop, crop_sha256=str(tuple(crop)), image_hw=hw,
                        raw=raw, size=size, offset=offset, factor=factor, block_calls=[1] * 7 + [0] * 5,
                        layer=7, head_calls=1, selector_calls=1)

        collect = stage.collect

        def execute(*args):
            return collect(*args, image_reader=lambda _root, item: item, observe=observe)

        observer = Mock()
        cached = execute(observer, point, entry, [native, native, native, cross, cross], ap)
        cached = {k: v[None] for k, v in cached.items()}
        cached["offsets"] = np.array([1])
        with TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            r139 = Path(directory) / "source"
            root.mkdir()
            (r139 / "replays").mkdir(parents=True)
            stage.c.publish_identical(root / "manifest.json", plan)
            stage.c.save(r139 / "replays/case.npz", cached, dict(plan_fingerprint="plan139", case_id="case"))
            with patch.object(stage, "collect", side_effect=execute) as replay:
                for _ in range(2):
                    stage.run(root, plan, r139, {"fingerprint": "plan139"}, observer, ap, lambda _: entry)
            replay.assert_called_once()
            stage.check_results(root)


if __name__ == "__main__":
    unittest.main()
