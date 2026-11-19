"""Validate exactly preserved pixels and same-geometry context comparisons."""
import csv
import hashlib
import io
from pathlib import Path
import unittest
from unittest.mock import Mock

import numpy as np

import stage141_crop_context_probe as stage
from stage27_uav_common import crop_geometry
from stage28_response_common import dense_boxes


class CropContextProbe(unittest.TestCase):
    def test_all_real_stage140_pairs_are_nested(self):
        folder = Path(__file__).resolve().parents[1] / "docs/experiments/06_sgla_uav_tracking/results"
        text = (folder / "stage140_results_20261119T142415Z_460351.txt").read_text(encoding="utf-8")
        csv_text = text.split("===== stage140_frames.csv =====\n", 1)[1].split("\n===== ", 1)[0]
        rows = list(csv.DictReader(io.StringIO(csv_text)))
        self.assertEqual(len(rows), 44)
        for i in range(0, len(rows), 4):
            crops = np.asarray([[float(r["crop_left"]), float(r["crop_top"]), float(r["crop_side"]), float(r["crop_side"])]
                                for r in rows[i:i+4]])
            pairs = stage.crop_pairs(crops)
            self.assertEqual([set(row) for row in pairs], [{0, 2}, {1, 3}])
            self.assertTrue((crops[pairs[:, 0], 2] < crops[pairs[:, 1], 2]).all())
            if i == 0:
                np.testing.assert_array_equal(pairs, [[0, 2], [1, 3]])

    def test_mean_and_edge_keep_tight_source_pixels(self):
        source = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        before = source.copy()
        support = [2, 3, 4, 2]
        mean = stage.fill_outer(source, support, "mean")
        edge = stage.fill_outer(source, support, "edge")
        for filled in (mean, edge):
            np.testing.assert_array_equal(filled[3:5, 2:6], source[3:5, 2:6])
        np.testing.assert_array_equal(mean[0, 0], stage.MEAN_RGB)
        np.testing.assert_array_equal(edge[0, 0], source[3, 2])
        np.testing.assert_array_equal(edge[7, 7], source[4, 5])
        np.testing.assert_array_equal(source, before)

    def test_replay_and_scoring_keep_large_geometry(self):
        hw = (128, 128)
        pixels = np.arange(hw[0] * hw[1] * 3, dtype=np.uint8).reshape(*hw, 3)
        native, cross = np.array([44., 44., 16., 16.]), np.array([52., 42., 24., 24.])
        parents = stage.source.cross_parents(native, cross)
        ap = dict(root=None, search_size=64, search_factor=4., window=np.ones((4, 4), np.float32), design={"box_atol": 1e-6})
        point = dict(point_id="fixture", sequence="fixture", frame=1, group="primary")
        entry = dict(frames=[pixels, pixels], initial_box=native)

        def observe(_observer, image, parent, _frame, _plan):
            crop, _, factor = crop_geometry(parent, hw, 4., 64)
            raw = np.full((4, 4), .01, np.float32)
            raw[2, 2] = .8
            size = np.full((2, 4, 4), .25, np.float32)
            offset = np.zeros_like(size)
            box = dense_boxes(size, offset, parent, factor, 64, hw)[10]
            return dict(parent=parent, boxes=box, crop=crop, factor=factor, image_hw=hw, raw=raw, size=size,
                        offset=offset, crop_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                        layer=7, block_calls=[1]*7+[0]*5, head_calls=1, selector_calls=1)

        frozen = stage.source.collect(Mock(), point, entry, parents, ap,
                                      image_reader=lambda _root, image: image, observe=observe)
        pairs = stage.crop_pairs(frozen["crop"])
        forward = Mock(side_effect=observe)
        data = stage.collect(Mock(), point, entry, frozen, pairs, ap,
                             image_reader=lambda _root, image: image, observe=forward)
        self.assertEqual(forward.call_count, 8)
        self.assertEqual(stage.verify(data, frozen, pairs, True, 1e-6), 0.)
        self.assertTrue((data["crop_sha256"][:, 1] != data["crop_sha256"][:, 2]).all())
        rows, effects = stage.score(data, point, native, ap)
        self.assertEqual(len(rows), 8)
        self.assertEqual(len(effects), 6)
        self.assertTrue(all(r["target_fully_kept"] for r in effects))


if __name__ == "__main__":
    unittest.main()
