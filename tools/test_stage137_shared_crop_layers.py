"""Check shared-prefix accounting and the distinction between output and grid supply."""
import csv
import io
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

import stage137_shared_crop_layers as stage
from stage27_uav_common import crop_geometry


class Module:
    def __init__(self, function):
        self.function, self.hooks = function, []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))

    def __call__(self, value):
        output = self.function(value)
        for hook in self.hooks:
            hook(self, (value,), output)
        return output


class SharedCropLayers(unittest.TestCase):
    def setUp(self):
        self.parent = np.array([40., 40., 16., 16.])
        self.truth = np.array([8., 8., 16., 16.])
        self.image = np.zeros((128, 128, 3), np.uint8)
        self.ap = dict(window=np.ones((4, 4), np.float32), search_size=64, search_factor=4.,
                       design={"box_atol": 1e-6})
        self.case = dict(case_id="example", sequence="example", frame=47, kind="onset",
                         group="onset_no_good_native_grid", source_native_dense_iou=.25)

    def fixture(self, expose_alternative):
        block_inputs = []

        def candidate_block(layer):
            def forward(tokens):
                block_inputs.append(tokens)
                return layer
            return Module(forward)

        blocks = [Module(lambda tokens: tokens + 1) for _ in range(6)] + [candidate_block(k) for k in stage.LAYERS]
        selector = Module(lambda _tokens: np.arange(6., 0., -1.))

        def output(layer):
            raw = np.full((4, 4), .01, np.float32)
            raw[0, 0] = 1. if expose_alternative and layer == 8 else .01
            raw[2, 2] = .8
            size = np.full((2, 4, 4), .25 if layer == 8 else .50, np.float32)
            return dict(score_map=raw, size_map=size, offset_map=np.zeros_like(size))

        head = Module(output)
        source = SimpleNamespace(sample_target=lambda image, _parent: (image, 1., None))

        def prefix(_search):
            tokens = 0
            for block in blocks[:6]:
                tokens = block(tokens)
            return tokens

        def decode(out, factor, image):
            return stage.dense_boxes(out["size_map"], out["offset_map"], self.parent, factor, 64,
                                     image.shape[:2])[out["score_map"].argmax()]

        rt = SimpleNamespace(backbone=SimpleNamespace(blocks=blocks, MLP=selector), net=SimpleNamespace(box_head=head),
                             tracker=SimpleNamespace(), torch=SimpleNamespace(no_grad=nullcontext), s1=source,
                             prepare=lambda image: source.sample_target(image, self.parent)[:2], prefix=prefix,
                             selector=selector, selected_layer=lambda scores: int(np.argmax(scores)) + 7,
                             candidate=lambda tokens, layer: head(blocks[layer - 1](tokens)), decode=decode, search_side=4)
        data = stage.collect(rt, self.image, self.parent, 47)
        self.assertEqual(block_inputs, [6] * 6)
        self.assertTrue(all(not m.hooks for m in (*blocks, selector, head)))
        crop = crop_geometry(self.parent, self.image.shape[:2], 4., 64)[0]
        cached = dict(parent=data["parent"][None, None], crop_sha256=data["crop_sha256"].reshape(1, 1),
                      anchor_hw=data["image_hw"], crop=crop[None, None], boxes=data["boxes"][:1][None],
                      layer=np.array([[7]]), cells=np.array([10]),
                      **{f"anchor_{k}": data[k][0] for k in ("raw", "size", "offset")})
        return data, cached

    def test_alternative_standard_decode_supplies_box(self):
        data, cached = self.fixture(True)
        row, layers = stage.score_case(data, self.case, cached, self.truth, self.ap, True)
        self.assertEqual(len(layers), 6)
        self.assertEqual(row["native_dense_iou"], .25)
        self.assertEqual(row["best_other_top1_iou"], 1.)
        self.assertTrue(row["adds_good_beyond_native_grid"])
        self.assertTrue(row["selector_second_adds_good"])

    def test_other_grid_good_without_standard_output(self):
        data, cached = self.fixture(False)
        row, _ = stage.score_case(data, self.case, cached, self.truth, self.ap, True)
        self.assertFalse(row["adds_good_standard"])
        self.assertTrue(row["adds_good_only_in_other_grid"])
        data["counts"][7] = 2
        with self.assertRaises(AssertionError):
            stage.score_case(data, self.case, cached, self.truth, self.ap, True)

    def test_freeze_real_archived_groups(self):
        results = Path(__file__).resolve().parents[1] / "docs/experiments/06_sgla_uav_tracking/results"

        def table(file, name):
            data = (results / file).read_text(encoding="utf-8")
            payload = data.split(f"===== {name} =====\n", 1)[1].split("\n===== ", 1)[0]
            return list(csv.DictReader(io.StringIO(payload)))

        events = table("stage133_results_20261119T091759Z_379705.txt", "stage133_events.csv")
        scope = table("stage134_results_20261119T100654Z_391381.txt", "stage134_scope.csv")
        plan = {"cases": [dict(r, frame=int(r["frame"])) for r in scope]}
        cases = stage.freeze_cases(plan, events)
        primary = [r for r in cases if r["group"] == "onset_no_good_native_grid"]
        self.assertEqual(len(primary), 8)
        self.assertEqual(len({r["sequence"] for r in primary}), 8)


if __name__ == "__main__":
    unittest.main()
