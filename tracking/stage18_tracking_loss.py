"""Supervised Stage-18 loss; importing this module needs no tracker or torch.

Targets use the *executed search crop*, not full-image coordinates. For image
GT (x, y, w, h), use stage18_environment.executed_crop_roi with the previous
executed box, actual resize factor, and search size. Do not clip that result.
Label masks affect supervision only; callers must retain every transition.
"""


def _center_heatmap(torch, target, height, width):
    from lib.utils.heapmap_utils import CenterNetHeatMap

    scale = target.new_tensor([width, height])
    centers = target[:, :2] + target[:, 2:] / 2
    onscreen = ((centers >= 0) & (centers < 1)).all(dim=-1)
    # Match the actor's rounded centers, keeping an in-crop edge center on-grid.
    pixels = (centers * scale).round().clamp_min(0)
    pixels = torch.minimum(pixels, scale - 1)
    radius = CenterNetHeatMap.get_gaussian_radius(target[:, 2:] * scale, 0.7)
    radius = radius.clamp_min(0).floor()[:, None, None]
    dx = torch.arange(width, device=target.device, dtype=target.dtype)[None, None, :]
    dy = torch.arange(height, device=target.device, dtype=target.dtype)[None, :, None]
    dx = dx - pixels[:, 0, None, None]
    dy = dy - pixels[:, 1, None, None]
    sigma = (2 * radius + 1) / 6
    gaussian = torch.exp(-(dx.square() + dy.square()) / (2 * sigma.square()))
    support = (dx.abs() <= radius) & (dy.abs() <= radius) & onscreen[:, None, None]
    return torch.where(support, gaussian, 0).unsqueeze(1), onscreen.sum()


def tracking_loss(torch, output, target_xywh, valid=True):
    """Return ``(loss, stats)`` for one chronological frame or a batch.

    ``output`` is rt.head's dictionary: normalized cxcywh ``pred_boxes`` of
    shape [B, 1, 4], and probability ``score_map`` of shape [B, 1, H, W].
    ``target_xywh`` is a tensor [B, 4] in normalized executed-crop coordinates.
    ``valid`` is a Python bool, scalar bool tensor, or bool tensor [B]. False
    rows are ignored before geometry/loss arithmetic, even for NaN labels.
    True rows must have finite coordinates and strictly positive sizes.

    Intersecting GT boxes retain their full, unclipped coordinates for GIoU
    and L1. Disjoint/touching boxes receive no box regression. Offscreen GT
    centers receive all-zero heatmaps, including partially visible boxes.
    Heatmaps use the repository's CenterNet radius at the actual H/W scale.

    The scalar loss is the mean over valid rows of 2*GIoU + 5*L1 + focal;
    L1 averages the four corner coordinates per box. Missing box supervision
    contributes zero, while localization negatives still contribute focal.
    All-invalid input returns a zero connected to both head outputs.

    ``stats`` contains detached scalar tensors: loss_sum (weighted), giou_sum,
    l1_sum, focal_sum, iou_sum, supervised_count, regression_count, and
    center_count. Counts are int64; regression_count supports int(...).
    Sum these across frames, then divide by the appropriate summed count.
    No values are converted to Python for logging here.
    """
    boxes, scores = output["pred_boxes"], output["score_map"]
    if boxes.ndim != 3 or boxes.shape[1:] != (1, 4) or boxes.shape[0] == 0:
        raise ValueError("pred_boxes must have nonempty shape [B, 1, 4]")
    batch = boxes.shape[0]
    if (scores.ndim != 4 or scores.shape[:2] != (batch, 1)
            or min(scores.shape[2:]) <= 0):
        raise ValueError("score_map must have shape [B, 1, H, W] with H/W > 0")
    if not torch.is_tensor(target_xywh) or target_xywh.shape != (batch, 4):
        raise ValueError("target_xywh must be a tensor [B, 4]")
    if not boxes.is_floating_point() or not scores.is_floating_point():
        raise TypeError("Head outputs must be floating-point tensors")
    if boxes.device != scores.device:
        raise ValueError("Head outputs must share a device")
    mask = torch.as_tensor(valid, device=boxes.device)
    if mask.dtype != torch.bool:
        raise TypeError("valid must be a bool or boolean tensor")
    if mask.ndim == 0:
        mask = mask.expand(batch)
    elif mask.shape != (batch,):
        raise ValueError("valid must be scalar or have shape [B]")

    # Float32 protects probability logs and geometry under fp16/bfloat16 AMP.
    dtype = torch.float64 if torch.float64 in (boxes.dtype, scores.dtype) else torch.float32
    boxes, scores = boxes.to(dtype=dtype)[mask], scores.to(dtype=dtype)[mask]
    target = target_xywh.detach().to(device=boxes.device, dtype=dtype)[mask]
    count = mask.sum()
    zero = boxes.sum() * 0 + scores.sum() * 0
    giou_sum = l1_sum = focal_sum = iou_sum = zero
    regression_count = center_count = count.new_zeros(())

    if boxes.shape[0]:
        if not torch.isfinite(target).all() or (target[:, 2:] <= 0).any():
            raise ValueError("Valid GT must be finite xywh with positive width/height")
        if not torch.isfinite(boxes).all() or (boxes[..., 2:] < 0).any():
            raise ValueError("Valid predictions must be finite cxcywh with nonnegative sizes")
        if not torch.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("Valid score_map values must be finite probabilities in [0, 1]")

        target_corners = torch.cat((target[:, :2], target[:, :2] + target[:, 2:]), dim=-1)
        intersects = (torch.minimum(target_corners[:, 2:], target.new_ones(2))
                      > torch.maximum(target_corners[:, :2], target.new_zeros(2))).all(dim=-1)
        regression_count = intersects.sum()
        if intersects.any():
            from lib.utils.box_ops import box_cxcywh_to_xyxy, giou_loss

            pred_corners = box_cxcywh_to_xyxy(boxes[intersects, 0])
            regression_target = target_corners[intersects]
            giou_mean, iou = giou_loss(pred_corners, regression_target)
            giou_sum = giou_mean * regression_count
            l1_sum = torch.nn.functional.l1_loss(
                pred_corners, regression_target, reduction="none").mean(dim=-1).sum()
            iou_sum = iou.sum()

        from lib.utils.focal_loss import FocalLoss

        heatmap, center_count = _center_heatmap(torch, target, *scores.shape[-2:])
        epsilon = torch.finfo(scores.dtype).eps
        probabilities = scores.clamp(min=epsilon, max=1 - epsilon)
        # Undo the legacy positive-count reduction to accumulate per-frame sums.
        focal_sum = FocalLoss()(probabilities, heatmap) * center_count.clamp_min(1)

    loss_sum = 2 * giou_sum + 5 * l1_sum + focal_sum
    loss = loss_sum / count.clamp_min(1)
    stats = {name: value.detach() for name, value in dict(
        loss_sum=loss_sum, giou_sum=giou_sum, l1_sum=l1_sum,
        focal_sum=focal_sum, iou_sum=iou_sum, supervised_count=count,
        regression_count=regression_count, center_count=center_count).items()}
    return loss, stats
