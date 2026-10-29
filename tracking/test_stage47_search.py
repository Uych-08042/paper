"""Matched-state, map/decode, capture-hook and frozen workflow checks for Stage-47."""
import contextlib
import copy
import csv
import inspect
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import stage46_motion_common as motion
import stage46_shift_audit as prior
import stage47_search_common as c
import stage47_search_collect as collect
import stage47_search_summary as summary
import test_stage46_shift_audit as fixtures

try:
    import torch
except ImportError:
    torch = None


def maps_for_box(box, parent, hw=(160, 180)):
    factor = c.crop_geometry(parent, hw, 4., 256)[2]
    side = 16
    raw = np.full((side, side), .01, np.float32)
    raw[8, 8] = .8
    size = np.ones((2, side, side), np.float32) * (np.asarray(box[2:]) * factor / 256).astype(np.float32)[:, None, None]
    offset = np.full((2, side, side), .5, np.float32)
    local = motion.center(box) - motion.center(parent) + 128 / factor
    offset[:, 8, 8] = local * factor / 256 * side - [8, 8]
    return raw, size, offset, factor


class TreatmentTests(unittest.TestCase):
    def test_shared_history_original_parent_bindings_and_relative_only(self):
        e = fixtures.entry(4)
        for history in c.HISTORIES:
            data = fixtures.trace(history, [[41 + i, 40, 20, 20] for i in range(4)], e["frames"])
            parents = c.proposals(data, 2, (160, 180))
            np.testing.assert_array_equal(parents[c.TREATMENTS.index(c.EXECUTED[history])], data["parents"][2])
            np.testing.assert_array_equal(parents[0], data["previous"][2])
            expected_relative = data["matrix"][2, :, :2] @ data["relative_velocity"][2]
            np.testing.assert_allclose(parents[3, :2] - data["previous"][2, :2], expected_relative, atol=1e-12)
            changed = copy.deepcopy(data)
            changed["boxes"][:] = [100, 100, 20, 20]
            np.testing.assert_array_equal(c.proposals(changed, 2, (160, 180)), parents)
            changed["motion_valid"][2] = False
            changed["matrix"][2] = motion.IDENTITY
            failed = c.proposals(changed, 2, (160, 180))
            np.testing.assert_array_equal(failed[0], failed[2])
            np.testing.assert_array_equal(failed[0], failed[3])
            np.testing.assert_array_equal(failed[1], failed[4])
        self.assertNotIn("gt", inspect.signature(collect.collect_case).parameters)
        self.assertNotIn("ground_truth", inspect.signature(collect.collect_case).parameters)

    def test_unclipped_mapping_and_shape_diagnostics(self):
        parent = np.array([40., 40., 20., 20.])
        box = np.array([40., 40., 20., 20.])
        raw, size, offset, factor = maps_for_box(box, parent)
        decoded = summary.unclipped_box(size, offset, int(np.argmax(raw)), parent, factor, 256)
        np.testing.assert_allclose(decoded, box, atol=1e-5)
        shifted = summary.unclipped_box(size, offset, int(np.argmax(raw)), parent + [.2, .1, 0, 0], factor, 256)
        np.testing.assert_allclose(shifted - decoded, [.2, .1, 0, 0], atol=1e-10)
        small = np.array([48., 48., 4., 4.])
        metrics = summary.shape_metrics(small, small, box)
        self.assertEqual(metrics["center_error_gt_units"], 0.)
        self.assertAlmostEqual(metrics["width_ratio"], .2)
        self.assertLess(metrics["iou"], .2)
        self.assertGreater(metrics["gt_size_at_pred_center_iou"], .99)
        thin = summary.shape_metrics([49, 0, 2, 100], [49, 0, 2, 100], [48, 0, 2, 100])
        self.assertLess(thin["center_error_gt_units"], .1)
        self.assertEqual(thin["dx_gt_width"], .5)


class FakeCapture:
    def __init__(self):
        self.calls, self.snapshots = 0, None
    def initialize(self, image, box):
        self.ordinal = 0
    def capture(self, image, parent, frame):
        history = c.HISTORIES[(self.ordinal // 5) % 4]
        treatment = c.TREATMENTS[self.ordinal % 5]
        self.ordinal += 1
        self.calls += 1
        saved = self.snapshots[history]
        box = saved["boxes"][frame - 1].copy() if treatment == c.EXECUTED[history] else np.array([40., 40., 20., 20.])
        raw, size, offset, factor = maps_for_box(box, parent)
        blocks = np.zeros(12, int)
        blocks[:6], blocks[8] = 1, 1
        crop = c.crop_geometry(parent, image.shape[:2], 4., 256)[0]
        return dict(boxes=box, layer=9, peak=float(raw.max()), raw=raw, size=size, offset=offset,
            factor=factor, block_calls=blocks, head_calls=1, selector_calls=1,
            crop_sha256=c.fingerprint(crop.tolist()), mask_sha256=c.fingerprint(crop.tolist()))
    def assert_frozen(self):
        pass


class WorkflowTests(unittest.TestCase):
    def test_source_smoke_partial_complete_resume_tamper_and_no_source_writes(self):
        # Only the Stage-27 source loader is mocked. Actual Stage-46/46B plans,
        # receipts, controller states, source hashes and Stage-47 caches are audited.
        config = copy.deepcopy(c.CONFIG)
        for event, frame in zip(config["events"], [15, 20, 15, 25]):
            event["frame"] = frame
        with tempfile.TemporaryDirectory(prefix="stage47-test-", dir=Path(__file__).parent) as folder:
            root = Path(folder)
            old, inputs = fixtures.FrozenWorkflowTests().fixture(root)
            inputs[1].update(manifest=dict(root=str(root)), runtime=dict(lens_x=256))
            with patch.object(motion, "load_source", return_value=inputs), patch.dict(c.CONFIG, config), contextlib.redirect_stdout(io.StringIO()):
                prior.audit(old)
                original = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
                output = root / "stage47"
                collect.prepare(old, output)
                plan, loaded = c.load_plan(output)
                self.assertEqual(plan["expected_head_calls"], 400)
                fake = FakeCapture()
                original_collect = collect.collect_case
                def capture_fixture(observer, event, snapshots, frames, dataset_root, initial_box, search_factor, search_size):
                    fake.snapshots = snapshots
                    return original_collect(fake, event, snapshots, frames, dataset_root, initial_box, search_factor, search_size,
                        image_reader=lambda *args: np.zeros((160, 180, 3), np.uint8))
                with patch.object(collect, "collect_case", side_effect=capture_fixture):
                    collect.execute(output, plan, loaded, fake, loaded["old_run"]["runtime"], np.ones((16, 16)), smoke=True)
                    self.assertEqual(fake.calls, 100)
                    self.assertIsNone(summary.summarize(output))
                    collect.execute(output, plan, loaded, fake, loaded["old_run"]["runtime"], np.ones((16, 16)))
                    self.assertEqual(fake.calls, 400)
                    collect.execute(output, plan, loaded, fake, loaded["old_run"]["runtime"], np.ones((16, 16)))
                    self.assertEqual(fake.calls, 400)
                report = summary.summarize(output)
                self.assertEqual(report["replay_head_calls"], 400)
                self.assertEqual(report["replay_block_calls"], 2800)
                self.assertFalse(report["new_closed_loop_policy_evaluated"])
                self.assertEqual(report, summary.summarize(output))
                with (output / "analysis/stage47_onset_summary.csv").open(encoding="utf-8", newline="") as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), 20)
                with (output / "analysis/stage47_shape_focus.csv").open(encoding="utf-8", newline="") as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), 25)
                with (output / "analysis/stage47_hybrids.csv").open(encoding="utf-8", newline="") as handle:
                    unchanged = [r for r in csv.DictReader(handle) if r["treatment"] == "no_shift"]
                self.assertEqual(len(unchanged), 80)
                self.assertTrue(all(float(r[key]) == 0. for r in unchanged for key in
                    ("mapping_only_gain", "changed_maps_gain", "full_gain", "interaction")))
                for name, payload in original.items():
                    self.assertEqual(Path(name).read_bytes(), payload)
                event = plan["events"][0]
                run = c.read_sealed(output / "run.json")
                path = c.path_for(output, event)
                data = c.read_cache(path, c.metadata(run, event))
                wrong = copy.deepcopy(data)
                wrong["parents"][0, 0, 0, 0] += 1
                with self.assertRaises(AssertionError):
                    c.audit_response(wrong, event, loaded, np.ones((16, 16)))
                wrong = copy.deepcopy(data)
                wrong["head_calls"][0, 0, 0] = 2
                with self.assertRaises(AssertionError):
                    c.audit_response(wrong, event, loaded, np.ones((16, 16)))
                with path.open("ab") as handle:
                    handle.write(b"changed")
                with self.assertRaisesRegex(ValueError, "response changed"):
                    summary.summarize(output)
                config["offsets"] = [-1, 0, 1]
                with patch.dict(c.CONFIG, config):
                    with self.assertRaisesRegex(ValueError, "source/code/configuration changed"):
                        c.load_plan(output)


@unittest.skipUnless(torch is not None, "Torch required for real dense output hooks")
class CaptureTests(unittest.TestCase):
    def test_real_hooks_one_forward_capture_and_sampling_wrapper_restored(self):
        from torch import nn
        from stage46_motion_runtime import NativeObserver
        class Head(nn.Module):
            def forward(self, x):
                return dict(score_map=torch.full((1, 1, 4, 4), .5),
                    size_map=torch.full((1, 2, 4, 4), .25), offset_map=torch.full((1, 2, 4, 4), .5))
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Module()
                self.backbone.blocks = nn.ModuleList([nn.Identity() for _ in range(12)])
                self.backbone.MLP, self.box_head = nn.Identity(), Head()
            def forward(self):
                x = torch.ones(1)
                for b in self.backbone.blocks[:6]:
                    x = b(x)
                self.backbone.MLP(x)
                return self.box_head(self.backbone.blocks[8](x))
        class Tracker:
            def __init__(self, net):
                self.net, self.frame_id = net, 0
            def track(self, image):
                sample_target(image, self.state, 4., output_sz=8)
                self.frame_id += 1
                self.net()
                return dict(target_bbox=self.state)
        def sampler(*args, **kwargs):
            return np.zeros((8, 8, 3), np.uint8), 1., np.zeros((8, 8), bool)
        net = Net().eval()
        rt = types.SimpleNamespace(net=net, tracker=Tracker(net), torch=torch, search_side=4)
        native = NativeObserver(rt)
        native.synchronize = lambda: None
        observer = collect.Capture(native)
        with patch.dict(globals(), sample_target=sampler):
            result = observer.capture(np.zeros((8, 8, 3)), np.array([1., 1., 4., 4.]), 8)
            self.assertIs(globals()["sample_target"], sampler)
            self.assertEqual(rt.tracker.frame_id, 8)
            self.assertEqual(result["block_calls"].sum(), 7)
            self.assertEqual(result["head_calls"], 1)
            np.testing.assert_array_equal(result["raw"], np.full((4, 4), .5))
            with patch.object(net, "forward", side_effect=RuntimeError("intentional")):
                with self.assertRaisesRegex(RuntimeError, "intentional"):
                    observer.capture(np.zeros((8, 8, 3)), np.array([1., 1., 4., 4.]), 9)
            self.assertIs(globals()["sample_target"], sampler)
        observer.assert_frozen()


if __name__ == "__main__":
    unittest.main()
