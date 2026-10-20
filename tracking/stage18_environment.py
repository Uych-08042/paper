"""Causal live tracking environment; labels are intentionally outside this API."""

import copy
import numpy as np


class TokenCredit:
    def __init__(self):
        self.credit = 0
        self.steps = 0
        self.calls = 0

    def advance(self):
        self.credit = min(20, self.credit + 3)

    def spend(self, action):
        if action not in range(7):
            raise ValueError("Action must be Fast or L7-L12 (indices 0-6)")
        if action and self.credit < 10:
            raise ValueError("Candidate action exceeds available compute credit")
        if action:
            self.credit -= 10
            self.calls += 1
        self.steps += 1
        if self.calls > 3 * self.steps // 10:
            raise RuntimeError("Causal candidate budget exceeded")


def executed_crop_roi(box, previous_box, resize_factor, search_size):
    box = np.asarray(box, dtype=np.float64)
    previous = np.asarray(previous_box, dtype=np.float64)
    side = float(search_size) / float(resize_factor)
    if (box.shape != (4,) or previous.shape != (4,) or not np.isfinite(box).all()
            or not np.isfinite(previous).all() or not np.isfinite(side) or side <= 0
            or np.any(box[2:] <= 0)):
        raise ValueError("Invalid executed box or crop geometry")
    origin = previous[:2] + previous[2:] / 2 - side / 2
    return np.r_[(box[:2] - origin) / side, box[2:] / side].astype(np.float32)


class TemporalEnvironment:
    def __init__(self, runtime, modules):
        self.rt, self.modules = runtime, modules
        self.memory = None
        self.budget = TokenCredit()
        self.previous_action = -1
        self.previous_confidence = 0.0
        self.motion = np.zeros(4, dtype=np.float32)

    def reset(self, image, initial_box):
        self.rt.initialize(image, initial_box)
        self.memory = None
        self.budget = TokenCredit()
        self.previous_action = -1
        self.previous_confidence = 0.0
        self.motion = np.zeros(4, dtype=np.float32)

    def snapshot(self):
        return dict(memory=None if self.memory is None else self.memory.clone(),
                    box=list(self.rt.tracker.state), frame_id=self.rt.tracker.frame_id,
                    budget=copy.deepcopy(self.budget), previous_action=self.previous_action,
                    confidence=self.previous_confidence, motion=self.motion.copy())

    def restore(self, state):
        self.memory = None if state["memory"] is None else state["memory"].clone()
        self.rt.tracker.state = list(state["box"])
        self.rt.tracker.frame_id = state["frame_id"]
        self.budget = copy.deepcopy(state["budget"])
        self.previous_action, self.previous_confidence = state["previous_action"], state["confidence"]
        self.motion = state["motion"].copy()

    def step(self, image, action=None):
        rt, torch = self.rt, self.rt.torch
        previous = np.asarray(rt.tracker.state, dtype=np.float64).copy()
        self.budget.advance()
        credit_before = self.budget.credit
        # Prefix has no trainable memory input. Later frozen operations must
        # retain autograd so losses can reach the adapters and memory read.
        with torch.no_grad():
            search, factor = rt.prepare(image)
            tokens = rt.prefix(search)
        if self.memory is None:
            self.memory = self.modules.initial_memory(tokens)
        history = np.r_[self.motion, self.previous_confidence, credit_before / 20,
                        np.eye(8, dtype=np.float32)[self.previous_action + 1]]
        logits, value = self.modules.observe(tokens, self.memory, tokens.new_tensor(history)[None])
        legal_logits = logits.clone()
        if credit_before < 10:
            legal_logits[:, 1:] = -torch.inf
        if action is None:
            action = int(legal_logits.argmax(1).item())
        self.budget.spend(action)
        conditioned = self.modules.read_memory(tokens, self.memory)
        features = conditioned if action == 0 else rt.backbone.blocks[action + 5](conditioned)
        features = self.modules.adapt(features, action)
        output = rt.head(features)
        box = np.asarray(rt.decode(output, factor, image), dtype=np.float64)
        roi = executed_crop_roi(box, previous, factor, rt.tracker.params.search_size)
        confidence = output["score_map"].detach().amax().reshape(1)
        self.memory = self.modules.write_memory(
            self.memory, features, tokens.new_tensor(roi)[None],
            torch.tensor([action], device=tokens.device), confidence)
        rt.tracker.state = box.tolist()
        rt.tracker.frame_id += 1
        self.motion = ((box - previous) / np.array(
            [image.shape[1], image.shape[0], image.shape[1], image.shape[0]])).astype(np.float32)
        self.previous_action, self.previous_confidence = action, float(confidence.item())
        return dict(box=box, action=action, logits=legal_logits, value=value,
                    output=output, features=features, memory=self.memory,
                    credit_before=credit_before, credit_after=self.budget.credit)
