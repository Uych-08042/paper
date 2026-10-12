"""NumPy-only metrics shared by experimental tracking entry points."""

import numpy as np


IOU_PROTOCOL = "sgla_int_prediction_inclusive_v1"


def tracking_iou(predictions, ground_truth, valid=True):
    """Broadcast xywh boxes; truncate predictions exactly as SGLA evaluation.

    GT coordinates stay floating point. Invalid rows return -1. Initialization
    is handled by the caller because the official evaluator forces IoU=1 there.
    """
    pred, gt = np.broadcast_arrays(
        np.asarray(predictions, dtype=np.float64),
        np.asarray(ground_truth, dtype=np.float64),
    )
    if pred.shape[-1] != 4:
        raise ValueError("Expected xywh boxes with last dimension 4")
    good = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(pred).all(axis=-1)
        & np.isfinite(gt).all(axis=-1)
        & (pred[..., 2:] >= 0).all(axis=-1)
        & (gt[..., 2:] > 0).all(axis=-1)
    )
    # np.trunc matches astype(int) for normal finite image coordinates, also
    # for negative x/y; it avoids integer overflow on malformed inputs.
    pred = np.trunc(np.where(np.isfinite(pred), pred, 0.0))
    gt = np.where(np.isfinite(gt), gt, 0.0)
    top_left = np.maximum(pred[..., :2], gt[..., :2])
    bottom_right = np.minimum(
        pred[..., :2] + pred[..., 2:] - 1.0,
        gt[..., :2] + gt[..., 2:] - 1.0,
    )
    intersection = np.prod(np.maximum(bottom_right - top_left + 1.0, 0), axis=-1)
    union = np.prod(pred[..., 2:], axis=-1) + np.prod(gt[..., 2:], axis=-1) - intersection
    good = good & (union > 0)
    return np.divide(intersection, union, out=np.full_like(union, -1.0), where=good)
