"""Exercise selective serial routing through a small public-tracker fixture."""
import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace

import numpy as np

import stage153_adaptive_depth as stage


def sample_target(image, parent):
    return image


class Module:
    def __init__(self, forward):
        self.forward = forward
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda:self.hooks.remove(hook))

    def __call__(self, value):
        output = self.forward(value)
        for hook in self.hooks:
            replacement = hook(self, (value,), output)
            if replacement is not None:
                output = replacement
        return output


class Scores:
    def __init__(self, frame):
        self.layer = 12 if frame == 34 else 9
        margin = .05 if frame == 34 else .1 if frame == 33 else .4
        self.values = np.array([[.9, .9-margin, .2, .1, .09, .08]])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class NativeFixture:
    def __init__(self):
        self.blocks = np.zeros(12, int)
        self.heads = self.selectors = 0
        blocks = [Module(lambda value, i=i:value+i+1) for i in range(12)]
        self.backbone = SimpleNamespace(blocks=blocks, MLP=Module(lambda _:Scores(self.frame_id)))
        self.head = Module(lambda tokens:tokens)
        self.rt = SimpleNamespace(backbone=self.backbone, tracker=self, torch=SimpleNamespace(no_grad=nullcontext))
        for j, module in enumerate(blocks):
            module.register_forward_hook(lambda *args, j=j:self.count_block(j))
        self.backbone.MLP.register_forward_hook(lambda *_:self.count("selectors"))
        self.head.register_forward_hook(lambda *_:self.count("heads"))

    def count_block(self, index):
        self.blocks[index] += 1

    def count(self, field):
        setattr(self, field, getattr(self,field)+1)

    def initialize(self, image, box):
        self.frame_id = 0
        self.state = list(box)

    def synchronize(self):
        pass

    def track(self, image):
        self.frame_id += 1
        tokens = sample_target(image, self.state)
        for block in self.backbone.blocks[:6]:
            tokens = block(tokens)
        selected = self.backbone.MLP(tokens).layer
        decoded = self.head(self.backbone.blocks[selected-1](tokens))
        self.state = [decoded, 0, 10, 10]
        return {"target_bbox":self.state}


class AdaptiveDepth(unittest.TestCase):
    def test_hard_frame_gets_one_serial_block_with_same_head_and_crop(self):
        tracker = NativeFixture()
        entry = dict(frames=[0]*35, initial_box=[0,0,10,10], ground_truth=[[30,0,10,10]]*35,
                     sequence="fixture")
        with ExitStack() as stack:
            observer = stage.DepthObserver(tracker, stack)
            traces = {arm:stage.collect(observer, entry, None, arm, reader=lambda _,frame:frame) for arm in stage.ARMS}
            for arm, data in traces.items():
                stage.audit(data, 34, entry["initial_box"], arm)
            native, adaptive = (traces[k] for k in stage.ARMS)
            stage.audit_prefix(native, adaptive)
            self.assertEqual(adaptive["added"][-2:].tolist(), [10, 0])
            self.assertEqual(adaptive["difficult"][-2:].tolist(), [True, True])
            self.assertEqual(adaptive["boxes"][-2,0], native["boxes"][-2,0]+10)
            self.assertEqual(adaptive["block_calls"].sum(), 34*7+1)
            self.assertEqual(adaptive["head_calls"].sum(), 34)
            rows, _, routing = stage.report_video(entry, traces)
            self.assertEqual(len(routing), 12)
            self.assertAlmostEqual(stage.summarize(rows)[0][1]["blocks_per_frame"], 7+1/34)
        self.assertEqual(len(tracker.backbone.blocks[8].hooks), 1)

    def test_threshold_uses_only_previous_frames_and_has_strict_ties(self):
        state = stage.SelectorHistory()
        for _ in range(stage.DESIGN["warmup"]):
            self.assertEqual(state.observe(.4), (-1., False))
        self.assertEqual(state.observe(.4), (.4, False))
        self.assertEqual(state.observe(.3), (.4, True))
        for _ in range(stage.DESIGN["margin_history"]):
            state.observe(.2)
        self.assertEqual(state.observe(.2), (.2, False))


if __name__ == "__main__":
    unittest.main()
