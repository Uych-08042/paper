"""Fixed dense compression and matched state-exposure/distillation contracts."""

import copy
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from stage4_common import fingerprint, load_manifest, tracking_iou
from stage19_ffn_compression import compress_ffn


SCHEMA = "stage20_fixed_student_correction_v1"
ARMS = ("supervised_reference", "kd_reference", "supervised_student", "kd_student")
METHODS = ("teacher", "compact_initial") + ARMS


def nonempty_path(value):
    import argparse
    if not str(value).strip():
        raise argparse.ArgumentTypeError("Empty path; export the shell variable first")
    return value


def design_split(source, mode="smoke", seed=2059, check_sequences=24):
    source = load_manifest(source)
    if mode not in ("smoke", "full"):
        raise ValueError("Unknown experiment mode")
    train = sorted((e for e in source["sequences"] if e["split"] == "train"),
                   key=lambda e: e["sequence"])
    count = check_sequences
    if count < 2 or len(train) - count < 2:
        raise ValueError("Need at least two fit and two development-check videos")
    order = np.random.default_rng(seed).permutation(len(train))
    check = [train[i] for i in order[:count]]
    fit = [train[i] for i in order[count:]]
    if mode == "smoke":
        fit = fit[:2]
        check = check[:2]
    return dict(fit=fit, check=check,
                excluded_source_val=sorted(e["sequence"] for e in source["sequences"]
                                           if e["split"] == "val"))


def channel_plan(weights):
    """Checkpoint-only magnitude product, fixed 50% width in every FFN."""
    specs = []
    for layer, (fc1, fc2) in enumerate(weights, start=1):
        fc1, fc2 = np.asarray(fc1), np.asarray(fc2)
        if fc1.ndim != 2:
            raise ValueError("Expected a matrix for each FC1 weight")
        width = fc1.shape[0]
        if (not width or fc2.shape != (fc1.shape[1], width)
                or width % 64 or not np.isfinite(fc1).all() or not np.isfinite(fc2).all()):
            raise ValueError("Require compatible finite dense FFNs with width divisible by 64")
        scores = np.abs(fc1.astype(np.float64)).mean(1) * np.abs(fc2.astype(np.float64)).mean(0)
        # Channel index breaks exact score ties reproducibly.
        keep = np.sort(np.lexsort((np.arange(width), -scores))[:width // 2])
        specs.append(dict(layer=layer, width=width,
                          drop=np.setdiff1d(np.arange(width), keep).tolist()))
    if len(specs) != 12:
        raise ValueError("This experiment fixes all twelve FFNs at half width")
    return specs


def compact_copy(torch, teacher, specs):
    candidate = copy.deepcopy(teacher)
    runtime = SimpleNamespace(torch=torch, backbone=candidate.backbone)
    audits = []
    with ExitStack() as stack:
        for spec in specs:
            audits.append(stack.enter_context(compress_ffn(runtime, spec, verify=True)))
        compact = copy.deepcopy(candidate)
    for name, parameter in compact.named_parameters():
        parameter.requires_grad_(not name.startswith("backbone.MLP."))
    compact.eval()
    return compact, audits


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.shape, str(array.dtype))).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def forward_model(rt, model, search):
    """Same L1-L6 + one independent candidate path for teacher and student."""
    bb = model.backbone
    z = bb.patch_embed(rt.tracker.z_dict1.tensors) + bb.pos_embed_z
    x = bb.patch_embed(search) + bb.pos_embed_x
    tokens = bb.pos_drop(rt.s1.combine_tokens(z, x, mode=bb.cat_mode))
    for block in bb.blocks[:6]:
        tokens = block(tokens)
    with rt.torch.no_grad():
        layer = rt.selected_layer(bb.MLP(tokens[:, :, 0].detach().clone()))
    tokens = bb.blocks[layer - 1](tokens)
    tokens = rt.s1.recover_tokens(tokens, rt.lens_z, rt.lens_x, mode=bb.cat_mode)
    return model.forward_head(bb.norm(tokens), None), layer


def crop_parent(arm, reference_previous, student_previous):
    if arm not in ARMS:
        raise ValueError("Unknown training arm")
    result = reference_previous if arm.endswith("_reference") else student_previous
    return np.asarray(result, dtype=np.float64).copy()


def xywh_iou(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if (left.shape != (4,) or right.shape != (4,) or not np.isfinite([left, right]).all()
            or np.any(left[2:] <= 0) or np.any(right[2:] <= 0)):
        return 0.0
    intersection = np.maximum(0, np.minimum(left[:2] + left[2:], right[:2] + right[2:])
                              - np.maximum(left[:2], right[:2]))
    area = float(np.prod(intersection))
    return area / max(1e-12, float(np.prod(left[2:]) + np.prod(right[2:]) - area))


def accept_teacher(target_roi, teacher_box, gt, valid):
    if not valid:
        return False
    roi = np.asarray(target_roi)
    center = roi[:2] + roi[2:] / 2
    return bool(np.isfinite(roi).all() and np.all((center >= 0) & (center < 1))
                and xywh_iou(teacher_box, gt) >= .5)


def distillation_loss(torch, student, teacher):
    """Spatial probability KL + normalized box L1; teacher always detached."""
    def logits(output):
        p = output["score_map"].float().flatten(1).clamp(1e-6, 1 - 1e-6)
        return torch.log(p) - torch.log1p(-p)
    log_student = torch.nn.functional.log_softmax(logits(student), dim=1)
    target = torch.nn.functional.softmax(logits(teacher).detach(), dim=1)
    spatial = torch.nn.functional.kl_div(log_student, target, reduction="batchmean")
    boxes = torch.nn.functional.l1_loss(student["pred_boxes"].float(),
                                        teacher["pred_boxes"].detach().float())
    return spatial + 5 * boxes


def auc(values):
    values = np.asarray(values, dtype=float)
    values = values[values >= 0]
    if not len(values):
        raise ValueError("No valid evaluation labels")
    return float(100 * (values[:, None] > np.arange(21)[None] * .05).mean())


def paired_interval(values, samples=5000, seed=2060):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or samples < 1:
        raise ValueError("Invalid paired bootstrap input")
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for start in range(0, samples, 256):
        size = min(256, samples - start)
        means[start:start + size] = values[rng.integers(len(values), size=(size, len(values)))].mean(1)
    return dict(mean_delta=float(values.mean()), ci_low=float(np.percentile(means, 2.5)),
                ci_high=float(np.percentile(means, 97.5)), wins=int((values > 1e-8).sum()),
                ties=int((np.abs(values) <= 1e-8).sum()), losses=int((values < -1e-8).sum()))


def factorial_effect(scores):
    return ((scores["kd_student"] - scores["supervised_student"])
            - (scores["kd_reference"] - scores["supervised_reference"]))


def load_contract(directory):
    directory = Path(directory).expanduser().resolve()
    data = json.loads((directory / "stage20_contract.json").read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA or data.get("fingerprint") != fingerprint(
            {key: value for key, value in data.items() if key != "fingerprint"}):
        raise ValueError("Stage-20 contract fingerprint mismatch")
    return directory, data


def audit_trace(data, initial, expected_frames):
    n = len(expected_frames)
    if set(data) != {"frame_id", "previous", "boxes", "gt", "valid", "iou", "layer", "latency_ms"}:
        raise ValueError("Unexpected trace fields")
    for key, shape in dict(frame_id=(n,), previous=(n, 4), boxes=(n, 4), gt=(n, 4),
                           valid=(n,), iou=(n,), layer=(n,), latency_ms=(n,)).items():
        if data[key].shape != shape:
            raise ValueError("Trace shape mismatch: " + key)
        if key != "gt" and not np.isfinite(data[key]).all():
            raise ValueError("Nonfinite trace: " + key)
    if (data["valid"].dtype != np.bool_ or not np.array_equal(data["frame_id"], expected_frames)
            or not np.issubdtype(data["layer"].dtype, np.integer)
            or np.any((data["layer"] < 7) | (data["layer"] > 12))
            or np.any(data["latency_ms"] <= 0) or np.any(data["boxes"][:, 2:] <= 0)):
        raise ValueError("Invalid trace metadata/geometry")
    expected = np.concatenate((np.asarray(initial)[None], data["boxes"][:-1]))
    if not np.allclose(data["previous"], expected, rtol=0, atol=1e-8):
        raise ValueError("Student must advance its own crop during evaluation")
    expected = tracking_iou(data["boxes"], data["gt"], data["valid"])
    if not np.allclose(data["iou"], expected, rtol=0, atol=1e-10):
        raise ValueError("Trace IoU label mismatch")
    return auc(data["iou"])
