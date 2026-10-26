"""NumPy-only tests for grid/offset separation, center closure and frozen events."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from stage4_common import file_hash
from test_stage36_geometry_audit import minimal_source
import stage36_geometry_audit as geometry
import stage36_onset_jump_audit as module

HERE = Path(__file__).parent


def state(grid=(1, 1), parent=(30., 30.), stride=10., offsets=None, gt=None,
          fp=(0., 0.), clipping=(0., 0.), native=(0., 0.)):
    if offsets is None:
        offsets = np.full((2, 4, 4), .25, np.float32)
    grid = np.array(grid, float)
    x, y = map(int, grid)
    selected = offsets[:, y, x].astype(float)
    fp, clipping, native = map(lambda x: np.array(x, float), (fp, clipping, native))
    center = np.asarray(parent, float) + stride * (grid + selected - 2)
    return dict(side=4, grid=grid, offset_map=offsets, parent_center=np.array(parent), stride=stride,
        native_center=center + fp + clipping + native, float32_delta=fp, clipping_delta=clipping,
        native_delta=native, gt=np.array([10., 10., 20., 20.] if gt is None else gt), image_hw=np.array([100, 100]))


class OnsetTests(unittest.TestCase):
    def test_grid_jump_and_fixed_grid_offset_change_are_separated(self):
        result, pieces = module.center_accounting(state(), state(grid=(2, 1)))
        by_term = {r["term"]: r for r in pieces}
        self.assertTrue(result["peak_changed"])
        self.assertAlmostEqual(by_term["response_grid"]["along_error_gt_units"], .5)
        self.assertEqual(by_term["offset_field"]["magnitude_gt_units"], 0.)
        self.assertEqual(by_term["offset_resampling"]["magnitude_gt_units"], 0.)
        offsets = np.full((2, 4, 4), .25, np.float32)
        offsets[0] = .75
        result, pieces = module.center_accounting(state(), state(offsets=offsets))
        by_term = {r["term"]: r for r in pieces}
        self.assertFalse(result["peak_changed"])
        self.assertAlmostEqual(by_term["offset_field"]["along_error_gt_units"], .25)
        self.assertEqual(by_term["response_grid"]["magnitude_gt_units"], 0.)

    def test_spatial_offset_resampling_is_not_mislabeled_as_temporal_field_change(self):
        offsets = np.zeros((2, 4, 4), np.float32)
        offsets[0] = np.arange(4, dtype=np.float32)[None] * .2
        result, pieces = module.center_accounting(state(offsets=offsets), state(grid=(2, 1), offsets=offsets))
        by_term = {r["term"]: r for r in pieces}
        self.assertTrue(result["peak_changed"])
        self.assertEqual(by_term["offset_field"]["magnitude_gt_units"], 0.)
        self.assertAlmostEqual(by_term["offset_resampling"]["along_error_gt_units"], .1, places=7)
        self.assertLessEqual(result["accounting_residual_px"], 1e-12)

    def test_crop_gt_clipping_numerical_terms_close_and_zero_error_has_no_projection(self):
        a = state(parent=(20., 20.), stride=8., fp=(1e-5, 0))
        offsets = np.full((2, 4, 4), .6, np.float32)
        b = state(grid=(2, 3), parent=(21., 22.), stride=10., offsets=offsets,
            gt=(10.5, 9.75, 20., 20.), fp=(-2e-5, 1e-5), clipping=(1., -2.), native=(1e-6, 2e-6))
        result, pieces = module.center_accounting(a, b)
        self.assertLessEqual(result["accounting_residual_px"], 1e-12)
        self.assertAlmostEqual(sum(r["along_error_gt_units"] for r in pieces), result["error_vector_change_gt_units"], places=12)
        result, pieces = module.center_accounting(state(), state(parent=(35., 30.), gt=(15., 10., 20., 20.)))
        self.assertEqual(result["error_vector_change_gt_units"], 0.)
        self.assertTrue(all(r["along_error_gt_units"] is None for r in pieces))
        self.assertIsNone(result["projection_residual_gt_units"])
        b["image_hw"] = np.array([200, 200])
        with self.assertRaisesRegex(ValueError, "resolution changed"):
            module.center_accounting(a, b)

    def test_cached_event_audit_candidate_retention_repeat_and_source_tamper(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            source = minimal_source(root)
            event = dict(sequence="uav0000016_00000_s", student="supervised_seed2111", onset=5)
            with mock.patch.object(geometry, "load_inputs", return_value=source):
                geometry.audit(root)
                before = {str(p): file_hash(p) for p in root.rglob("*") if p.is_file()}
                with mock.patch.dict(module.CONFIG, events=[event]):
                    report = module.audit(root)
                    self.assertEqual(report["status"], "COMPLETE_CACHED_ONSET_AUDIT")
                    self.assertEqual(report["source_videos"], 1)
                    self.assertEqual(report["distinct_frame_images"], 5)
                    self.assertEqual(report["cached_model_crop_frame_views"], 80)
                    self.assertEqual(len(report["events"]), 1)
                    self.assertLessEqual(report["maximum_accounting_residual_px"], 1e-9)
                    self.assertIsNone(report["adoption_gate"])
                    self.assertFalse(report["new_inference_performed"])
                    self.assertIn("onset_weighted_top5_gt_recall", report["events"][0])
                    module.verify_bindings(before)
                    self.assertEqual(report, module.audit(root))
                    output = root / "analysis/onset_jump_audit_v1"
                    self.assertEqual(len(list(output.glob("*.csv"))), 6)
                    saved = json.loads((output / "stage36c_diagnostic.json").read_text())
                    table = root / "analysis/geometry_factor_audit_v1/stage36b_geometry_focus.csv"
                    table.write_bytes(table.read_bytes() + b"changed source")
                    with self.assertRaisesRegex(ValueError, "Frozen Stage-35 artifact changed"):
                        module.audit(root)
                    self.assertEqual(json.loads((output / "stage36c_diagnostic.json").read_text()), saved)

    def test_missing_context_and_unplanned_onset_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            root = Path(temporary)
            plan, source, traces = minimal_source(root)
            video = plan["videos"][0]
            with np.load(root / "responses" / (video["sequence"] + "__stage36_same_crop.npz"), allow_pickle=False) as archive:
                data = {k: archive[k] for k in archive.files if k != "metadata_json"}
            event = dict(sequence=video["sequence"], student="supervised_seed2111", onset=5)
            bad = dict(data, frame_id=np.array(data["frame_id"]))
            bad["frame_id"][np.flatnonzero(bad["frame_id"] == 4)[0]] = 99
            with self.assertRaisesRegex(ValueError, "context is absent"):
                module.analyze_event(event, bad, source, video)
            with self.assertRaisesRegex(ValueError, "original planned first failure"):
                module.analyze_event(dict(event, onset=6), data, source, video)


if __name__ == "__main__":
    unittest.main()
