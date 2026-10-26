"""NumPy-only factor isolation, legacy endpoint parity, timelines and immutability."""

import contextlib
import io
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from stage4_common import file_hash, write_cache, write_json
from stage27_uav_common import crop_geometry, seal, tracking_iou
from stage27_uav_summary import write_table
from stage28_response_common import clip_boxes, dense_boxes
from stage36_same_crop_common import METHODS, SCHEMA as SOURCE_SCHEMA, video_plan
from stage36_same_crop_summary import describe
from test_stage36_same_crop import response_example
import stage36_geometry_audit as module

HERE = Path(__file__).parent


def minimal_source(root):
    """Numeric Stage-36 fixture; only the previously tested provenance adapter is mocked."""
    names = ["uav0000016_00000_s", "uav0000085_00000_s", "uav0000089_00920_s", "uav0000090_00276_s"]
    n, side, gt = 15, 4, np.array([16., 16., 16., 16.])
    source = dict(runtime=dict(lens_x=16), search_factor=4., search_size=256, fit=[])
    videos, traces, raw_data = [], {}, {}
    source_file = root / "source_binding.txt"
    source_file.write_text("frozen original source")
    all_responses, all_hybrids, all_parents = [], [], []
    for case, name in enumerate(names):
        entry = dict(sequence=name, source_group_proxy=name.split("_")[0], initial_box=gt.tolist(),
            frames=[dict(height=64, width=64)] * (n + 1))
        source["fit"].append(entry)
        predicted = np.tile(gt, (n, 4, 1))
        if case == 0:
            predicted[4:, 2, 0] += 4
            predicted[4:, 3, 0] += 12
        elif case == 1:
            for evaluator, width in ((2, 24.), (3, 32.)):
                predicted[4:, evaluator, 2:] = width
                predicted[4:, evaluator, :2] = 24 - width / 2
        elif case == 3:
            predicted[4:, 1, 0] += 12
        traces[name] = {}
        for evaluator, method in enumerate(METHODS):
            boxes = predicted[:, evaluator]
            traces[name][method] = dict(frame_id=np.arange(1, n + 1), previous=np.vstack([gt, boxes[:-1]]),
                boxes=boxes.copy(), gt=np.tile(gt, (n, 1)), iou=tracking_iou(boxes, gt))
        video = video_plan(entry, traces[name])
        videos.append(video)
        ids = np.array([t["frame_id"] for t in video["frames"]])
        count = len(ids)
        data = dict(frame_id=ids, raw=np.full((count, 4, 4, side, side), .01, np.float32),
            size=np.empty((count, 4, 4, 2, side, side), np.float32), offset=np.empty((count, 4, 4, 2, side, side), np.float32),
            native_boxes=np.empty((count, 4, 4, 4)), previous=np.empty((count, 4, 4)), gt=np.tile(gt, (count, 1)),
            factor=np.empty((count, 4)), image_hw=np.full((count, 2), 64, np.int64),
            layer=np.tile([0, 9, 0, 0], (count, 4, 1)), crop_sha256=np.full((count, 4), "a" * 64),
            replay_box_error=np.zeros((count, 4)), decode_box_error=np.empty((count, 4, 4)),
            window=np.ones((4, 4), np.float32), model_head_calls=np.array([8 * count, 4 * count, 4 * count]))
        for row, frame in enumerate(ids):
            for parent, owner in enumerate(METHODS):
                previous = traces[name][owner]["previous"][frame - 1]
                factor = crop_geometry(previous, [64, 64], 4., 256)[2]
                data["previous"][row, parent], data["factor"][row, parent] = previous, factor
                for evaluator in range(4):
                    box = predicted[frame - 1, evaluator]
                    cell = 10 if case == 2 and evaluator >= 2 else 5
                    y, x = divmod(cell, side)
                    center = (box[:2] + box[2:] / 2 - previous[:2] - previous[2:] / 2 + .5 * 256 / factor) / (256 / factor)
                    off = center * side - [x, y]
                    data["raw"][row, parent, evaluator, y, x] = .9
                    data["size"][row, parent, evaluator] = (box[2:] / (256 / factor))[:, None, None]
                    data["offset"][row, parent, evaluator] = off[:, None, None]
                    data["native_boxes"][row, parent, evaluator] = box
                    dense = dense_boxes(data["size"][row, parent, evaluator], data["offset"][row, parent, evaluator], previous, factor, 256, [64, 64])
                    data["decode_box_error"][row, parent, evaluator] = np.max(np.abs(dense[cell] - box))
        raw_data[name] = data
        r, h, d = describe(video, data, source)
        all_responses.extend(r)
        all_hybrids.extend(h)
        all_parents.extend(d)
    plan = seal(dict(schema=SOURCE_SCHEMA, videos=videos, source_file_sha256={str(source_file): file_hash(source_file)}))
    run = seal(dict(schema=SOURCE_SCHEMA, plan_fingerprint=plan["fingerprint"], methods=list(METHODS), parents=list(METHODS), window=np.ones((4, 4)).tolist()))
    write_json(root / "plan.json", plan)
    write_json(root / "run.json", run)
    caches = {}
    for name, data in raw_data.items():
        path = root / "responses" / (name + "__stage36_same_crop.npz")
        write_cache(path, data, dict(schema=SOURCE_SCHEMA, run_fingerprint=run["fingerprint"], sequence=name))
        caches[path.name] = file_hash(path)
    tables = dict(stage36_responses=all_responses, stage36_hybrids=all_hybrids, stage36_parent_decomposition=all_parents)
    hashes = {}
    for label, rows in tables.items():
        path = root / "analysis" / (label + ".csv")
        write_table(path, rows)
        hashes[path.name] = file_hash(path)
    diagnostic = seal(dict(schema=SOURCE_SCHEMA, status="COMPLETE_SAME_CROP_DIAGNOSTIC", audited_cases=4, missing_cases=[],
        plan_fingerprint=plan["fingerprint"], run_fingerprint=run["fingerprint"], response_cache_sha256=caches, table_sha256=hashes, figures=[]))
    write_json(root / "analysis/stage36_diagnostic.json", diagnostic)
    return plan, source, traces


class GeometryTests(unittest.TestCase):
    def test_single_cell_matches_original_dense_float32_mapping(self):
        rng = np.random.default_rng(2115)
        for side in (4, 16):
            size = rng.uniform(.05, .7, (2, side, side)).astype(np.float32)
            offset = rng.normal(size=(2, side, side)).astype(np.float32)
            previous = np.array([10.125, 18.25, 24.5, 20.75])
            expected = dense_boxes(size, offset, previous, 1.97, 256, [120, 160])
            for cell in rng.choice(side * side, min(25, side * side), replace=False):
                _, actual = module.decode_cell(size, offset, int(cell), previous, 1.97, 256, [120, 160])
                np.testing.assert_array_equal(actual, expected[cell])

    def test_factor_isolation_identity_and_order_averaged_interactions(self):
        ranking = response_example()
        values, _, _ = module.factorial(ranking, 0, 0, 0, 2, 256)
        result = module.split_values(values)
        self.assertAlmostEqual(result["order_averaged_response_gain"], 1., places=12)
        self.assertEqual(result["order_averaged_offset_gain"], 0.)
        self.assertEqual(result["order_averaged_size_gain"], 0.)
        scale = response_example(geometry=True)
        values, _, _ = module.factorial(scale, 0, 0, 0, 2, 256)
        result = module.split_values(values)
        self.assertGreater(result["order_averaged_size_gain"], .5)
        self.assertEqual(result["order_averaged_offset_gain"], 0.)
        offset = response_example(geometry=True)
        offset["size"].fill(.25)
        offset["offset"][:, :, :2] = 2
        for parent in range(4):
            for evaluator in range(4):
                grid = dense_boxes(offset["size"][0, parent, evaluator], offset["offset"][0, parent, evaluator],
                    offset["previous"][0, parent], 4., 256, [64, 64])
                offset["native_boxes"][0, parent, evaluator] = grid[5]
        values, _, _ = module.factorial(offset, 0, 0, 0, 2, 256)
        self.assertAlmostEqual(module.split_values(values)["order_averaged_offset_gain"], 1., places=12)
        identity = response_example()
        for key in ("raw", "size", "offset", "native_boxes"):
            identity[key][:, :, 2] = identity[key][:, :, 0]
        identity["native_boxes"][..., 2] -= 1e-6
        values, _, _ = module.factorial(identity, 0, 0, 0, 2, 256)
        np.testing.assert_array_equal(values, np.full((2, 2, 2), values[0, 0, 0]))
        polynomial = np.empty((2, 2, 2))
        for r, o, z in itertools.product((0, 1), repeat=3):
            polynomial[r, o, z] = (r + 2 * o + 3 * z + 4 * r * o + 5 * o * z + 6 * r * z + 7 * r * o * z) / 28
        summary = module.split_values(polynomial)
        self.assertAlmostEqual(summary["order_averaged_response_gain"], (1 + 2 + 3 + 7 / 3) / 28)
        self.assertLessEqual(summary["max_accounting_residual"], 1e-12)

    def test_clipping_does_not_masquerade_as_offset_change(self):
        unclipped = np.array([-10., 10., 40., 20.])
        clipped = clip_boxes(unclipped, [64, 64])
        metrics = module.box_metrics(clipped, unclipped, clipped, np.array([0., 10., 20., 20.]))
        self.assertEqual(metrics["unclipped_center_error_gt_units"], 0.)
        self.assertEqual(metrics["center_error_gt_units"], .25)
        self.assertEqual(metrics["clip_max_abs_delta"], 10.)

    def test_completed_cache_audit_is_read_only_repeatable_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            info = minimal_source(root)
            originals = {str(p): file_hash(p) for p in root.rglob("*") if p.is_file()}
            with mock.patch.object(module, "load_inputs", return_value=info) as adapter:
                report = module.audit(root)
                self.assertEqual(adapter.call_count, 1)
                self.assertEqual(report["status"], "COMPLETE_CACHED_GEOMETRY_AUDIT")
                self.assertEqual(report["maximum_original_endpoint_error"], 0.)
                self.assertLessEqual(report["maximum_factorial_accounting_residual"], 1e-12)
                self.assertFalse(report["training_performed"])
                self.assertFalse(report["new_inference_performed"])
                self.assertGreater(report["timeline_rows"], 0)
                module.verify_bindings(originals)
                self.assertEqual(report, module.audit(root))
                output = root / "analysis/geometry_factor_audit_v1"
                self.assertEqual(len(list(output.glob("*.csv"))), 8)
                # A cache remains loadable after byte appending, but its frozen SHA must fail.
                path = next((root / "responses").glob("*.npz"))
                original_bytes = path.read_bytes()
                path.write_bytes(original_bytes + b"tampered")
                with self.assertRaisesRegex(ValueError, "Frozen Stage-35 artifact changed"):
                    module.audit(root)
                path.write_bytes(original_bytes)
                # New report tables are also immutable on reuse.
                table = output / "stage36b_factor_summary.csv"
                table.write_bytes(table.read_bytes() + b"tampered")
                with self.assertRaisesRegex(ValueError, "Frozen Stage-35 artifact changed"):
                    module.audit(root)
                self.assertEqual(json.loads((output / "stage36b_diagnostic.json").read_text()), report)


if __name__ == "__main__":
    unittest.main()
