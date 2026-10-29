"""Pinned official YOLOFT detector; frozen native network and small external gates."""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from stage45_detection_common import DESIGN, NAMES, digest, verify_upstream


def tensor_digest(state):
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        h.update(key.encode())
        h.update(str((list(tensor.shape), str(tensor.dtype))).encode())
        h.update(tensor.detach().cpu().contiguous().reshape(-1).view(__import__("torch").uint8).numpy().tobytes())
    return h.hexdigest()


class Runtime:
    def __init__(self, plan, output, device="0"):
        root = Path(verify_upstream(plan["upstream"]))
        os.environ["YOLO_AUTOINSTALL"] = "false"
        os.environ["YOLO_CONFIG_DIR"] = str(Path(output).resolve() / "runtime_config")
        os.environ["MPLCONFIGDIR"] = str(Path(output).resolve() / "matplotlib")
        sys.path.insert(0, str(root))
        import torch
        import ultralytics
        if root not in Path(ultralytics.__file__).resolve().parents:
            raise ValueError("A different ultralytics was already imported")
        from ultralytics import YOLOFT
        from ultralytics.utils.ops import non_max_suppression
        self.torch, self.nms = torch, non_max_suppression
        self.device = torch.device("cpu" if str(device) == "cpu" else "cuda:" + str(device))
        torch.manual_seed(DESIGN["seed"])
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA runtime required for the server pilot")
            torch.cuda.set_device(self.device)
            torch.cuda.manual_seed_all(DESIGN["seed"])
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        self.net = YOLOFT(str(root / "config/xsvid/yoloft-l-temporal.yaml")).model.float()
        spec = importlib.util.spec_from_file_location("stage45_official_checkpoint", root / "tools/xsvid/checkpoint_utils.py")
        ck = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ck)
        ck.load_release(self.net, Path(plan["weights"]), detector_only=True)
        self.net.to(self.device).eval().requires_grad_(False)
        self.net.names = dict(enumerate(NAMES))
        self.modules = [m for m in self.net.model if type(m).__name__ == "MSTFv1"]
        if len(self.modules) != 3 or [m.i for m in self.modules] != [19, 22, 25]:
            raise ValueError("Unexpected official fusion structure")
        self.head_calls = 0
        self.net.model[-1].register_forward_hook(self._count_head)
        self.base_sha256 = tensor_digest(self.net.state_dict())
        self.hooks = self.gates = None
        self.arm = "native"
        self.criterion = None
        self.reset()
        self.identity = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                             device=str(self.device), upstream=plan["upstream_commit"], precision="FP32",
                             detector_sha256=self.base_sha256)
        if self.device.type == "cuda":
            prop = torch.cuda.get_device_properties(self.device)
            self.identity.update(gpu=prop.name, gpu_uuid=str(getattr(prop, "uuid", "unavailable")))

    def _count_head(self, module, args, output):
        self.head_calls += 1

    def configure(self, arm, state=None):
        from stage45_temporal_gate import ResidualGates, GateHooks
        if self.hooks is not None:
            self.hooks.close()
        self.hooks = self.gates = None
        self.arm = arm
        if arm != "native":
            if arm not in ("capacity", "signals"):
                raise ValueError("Unknown arm")
            self.torch.manual_seed(DESIGN["seed"])
            self.gates = ResidualGates(DESIGN["hidden"]).to(self.device)
            if state is not None:
                self.gates.load_state_dict(state, strict=True)
            self.hooks = GateHooks(self.modules, self.gates, arm, DESIGN)
        self.reset()

    def reset(self):
        self.history, self.boxes, self.age = [None, None, None], [], 0
        if self.hooks is not None:
            self.hooks.reset()

    def image(self, frame):
        import numpy as np
        from PIL import Image
        if digest(frame["path"]) != frame["sha256"]:
            raise ValueError("Frozen image changed: " + frame["path"])
        with Image.open(frame["path"]) as image:
            if image.size != (DESIGN["image_size"], DESIGN["image_size"]):
                raise ValueError("Canonical image dimensions changed")
            array = np.array(image.convert("RGB"), copy=True)
        return self.torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device).float() / 255

    def sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def forward(self, image, loss_batch=None):
        if self.hooks is not None:
            self.hooks.begin(self.boxes, self.age)
        calls = self.head_calls
        self.sync()
        start = time.perf_counter()
        predictions, history = self.net((image, self.history))
        if self.head_calls - calls != 1 or len(history) != 3:
            raise ValueError("Expected exactly one detector call and three own history tensors")
        decoded = predictions[0] if isinstance(predictions, tuple) else predictions
        if not self.torch.isfinite(decoded).all():
            raise FloatingPointError("Non-finite detector predictions")
        with self.torch.no_grad():
            boxes = self.nms(decoded.detach().clone(), conf_thres=DESIGN["confidence"],
                iou_thres=DESIGN["nms_iou"], nc=7, max_det=DESIGN["max_det"],
                multi_label=True, end2end=False)[0]
            boxes[:, :4].clamp_(0, DESIGN["image_size"])
            boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
            self.boxes = boxes.cpu().tolist()
        self.history = [h.detach() for h in history]  # own trajectory; truncated gradient between frames
        self.age += 1
        self.sync()
        elapsed = (time.perf_counter() - start) * 1000
        stats = self.hooks.stats if self.hooks is not None else []
        loss = self.loss(predictions, loss_batch) if loss_batch is not None else None
        return dict(boxes=self.boxes, ms=elapsed, head_calls=1, gains=stats), loss, decoded

    def loss(self, predictions, batch):
        from ultralytics.utils.loss import v8DetectionLoss
        from ultralytics.utils.tal import make_anchors
        if self.criterion is None:
            self.net.args = SimpleNamespace(cls=.5, dfl=1.5, use_trend_loss_weight=False)
            self.criterion = v8DetectionLoss(self.net)
        features = predictions[1] if isinstance(predictions, tuple) else predictions
        anchors, strides = make_anchors(features, self.net.model[-1].stride, .5)
        centers = anchors * strides
        ignored = self.torch.zeros(len(centers), device=self.device, dtype=self.torch.bool)
        for x0, y0, x1, y1 in batch["ignore_xyxy"]:
            ignored |= ((centers[:, 0] >= x0) & (centers[:, 0] <= x1) &
                        (centers[:, 1] >= y0) & (centers[:, 1] <= y1))
        def ignore_background(module, args, values):
            targets = args[1]
            valid = ~(ignored.unsqueeze(0) & (targets.sum(-1) == 0))
            channels = self.torch.ones(7, device=values.device, dtype=values.dtype)
            channels[4] = 0  # ignore slot is not a foreground class for adaptation
            return values * valid.unsqueeze(-1) * channels
        handle = self.criterion.bce.register_forward_hook(ignore_background)
        try:
            loss, _ = self.criterion(predictions, batch)
        finally:
            handle.remove()
        return loss

    def target(self, image, annotations):
        positives, ignores = [], []
        size = DESIGN["image_size"]
        for a in annotations:
            x, y, w, h = a["bbox"]
            if a.get("ignore", 0) or a.get("iscrowd", 0) or a["category_id"] == 4:
                ignores.append((x, y, x+w, y+h))
                continue
            x0, y0, x1, y1 = max(0, x), max(0, y), min(size, x+w), min(size, y+h)
            if x1 > x0 and y1 > y0:
                positives.append([a["category_id"], (x0+x1)/(2*size), (y0+y1)/(2*size),
                                  (x1-x0)/size, (y1-y0)/size])
        values = self.torch.tensor(positives, device=self.device, dtype=self.torch.float32).reshape(-1, 5)
        return dict(img=image, batch_idx=self.torch.zeros(len(values), device=self.device),
                    cls=values[:, :1], bboxes=values[:, 1:], ignore_xyxy=ignores)

    def assert_frozen(self):
        if tensor_digest(self.net.state_dict()) != self.base_sha256:
            raise ValueError("Frozen detector weights/BN buffers changed")
        if any(p.grad is not None for p in self.net.parameters()):
            raise ValueError("Unexpected detector parameter gradients")
