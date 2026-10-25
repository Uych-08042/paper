"""NumPy-only descriptor probe, balanced pairs and causal memory stress inputs."""

import numpy as np

from stage27_uav_common import tracking_iou
from stage31_identity_common import normalize


ARMS = ("scalar_balanced", "descriptor_fixed", "descriptor_learned")
DESIGN = dict(seeds=[2100, 2101], projection_dim=16, updates=600, smoke_updates=12,
    batch_size=128, learning_rate=.003, head_l2=.01, projection_l2=.0001,
    gradient_clip=5., adam_beta1=.9, adam_beta2=.999, adam_epsilon=1e-8,
    positive_iou=.5, negative_iou=.1, low_iou=.2, bootstrap_samples=5000, bootstrap_seed=2102,
    minimum_hard_videos=6, minimum_hard_pair_frames=100, minimum_hard_accuracy=.6,
    minimum_hard_accuracy_lift=.05, minimum_recovery_recall=.2, minimum_easy_retention=.99)


def prepare_video(entry, data):
    """No identity/GT label is an input to a scorer; keep outcome arrays separate."""
    d = normalize(data["descriptors"])
    valid = np.asarray(data["valid"], bool)
    overlap = np.where(valid, tracking_iou(data["candidate_boxes"], data["gt"][:, None, :]), -1.)
    scalar = np.array(data["features"], float, copy=True)
    stress = scalar.copy()
    swapped = np.zeros(len(scalar), bool)
    known_negative_history = np.zeros(len(scalar), bool)
    # Apply to every eligible frame, before any outcome-based subgrouping.
    # Use previous response-rank 2, not a GT-selected candidate. Only descriptor
    # memory changes; crop, velocity, trusted prototype and responses stay fixed.
    for i in range(1, len(scalar)):
        if not valid[i - 1, 1]:
            continue
        previous = d[i - 1, valid[i - 1]]
        others = np.delete(previous, 1, axis=0)
        recent = d[i] @ previous[1]
        other = (d[i] @ others.T).max(1)
        stress[i, :, 2], stress[i, :, 3], stress[i, :, 4] = recent, other, recent - other
        swapped[i] = True
        known_negative_history[i] = data["labels"][i - 1, 1] == 0  # Subgroup label only.
    return dict(sequence=entry["sequence"], source_group=entry["source_group_proxy"],
        frame_id=np.asarray(data["frame_id"]), descriptors=d, anchor=normalize(data["anchor"]),
        scalar=scalar, stress_scalar=stress, valid=valid, overlap=overlap,
        swapped=swapped, known_negative_history=known_negative_history)


def training_pairs(videos):
    """Equal available easy/hard cohorts, then video, frame, positive/negative pair.

    This differs deliberately from Stage-31's video/frame balance: an abundant
    already-correct cohort cannot consume almost all of the objective.
    """
    groups = {False: [], True: []}
    for video in videos:
        per_cohort = {False: [], True: []}
        for i, mask in enumerate(video["valid"]):
            pos = np.flatnonzero(mask & (video["overlap"][i] >= .5))
            neg = np.flatnonzero(mask & (video["overlap"][i] < .1))
            if len(pos) and len(neg):
                per_cohort[bool(video["overlap"][i, 0] < .5)].append((i, pos, neg))
        for hard, frames in per_cohort.items():
            if frames:
                groups[hard].append((video, frames))
    if not all(groups.values()):
        raise ValueError("Both easy and hard training cohorts need positive/negative pairs; no silent fallback")
    result = {key: [] for key in ("anchor", "positive", "negative", "sp", "sn", "weight")}
    for hard, records in groups.items():
        for video, frames in records:
            for i, pos, neg in frames:
                p, q = np.meshgrid(pos, neg, indexing="ij")
                p, q = p.ravel(), q.ravel()
                count = len(p)
                result["anchor"].append(np.broadcast_to(video["anchor"], (count, len(video["anchor"]))))
                result["positive"].append(video["descriptors"][i, p])
                result["negative"].append(video["descriptors"][i, q])
                result["sp"].append(video["scalar"][i, p])
                result["sn"].append(video["scalar"][i, q])
                result["weight"].append(np.full(count, .5 / len(records) / len(frames) / count))
    result = {key: np.concatenate(value) for key, value in result.items()}
    result["weight"] /= result["weight"].sum()
    result["scale"] = np.maximum(np.sqrt(np.sum(result["weight"][:, None] * (result["sp"] - result["sn"]) ** 2, axis=0)), .001)
    result["support"] = {"hard" if hard else "easy": dict(videos=len(records),
        pair_frames=sum(len(frames) for _, frames in records),
        pairs=sum(len(pos) * len(neg) for _, frames in records for _, pos, neg in frames))
        for hard, records in groups.items()}
    return result


def initial_parameters(arm, dim, seed, rank=16):
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    rng = np.random.default_rng(seed)
    return (dict(head=np.zeros(9)) if arm == "scalar_balanced" else
            dict(projection=rng.normal(0., 1., (dim, rank)), head=np.zeros(2 * rank + 2)))


def forward(parameters, arm, anchor, descriptor, scalar, scale):
    if arm == "scalar_balanced":
        phi = scalar / scale
        return phi @ parameters["head"], (phi,)
    a = np.tanh(anchor @ parameters["projection"])
    d = np.tanh(descriptor @ parameters["projection"])
    phi = np.concatenate([a * d, np.abs(a - d), scalar[:, 7:9] / scale[7:9]], axis=1)
    return phi @ parameters["head"], (phi, a, d, anchor, descriptor)


def backward(parameters, arm, cache, gradient):
    phi = cache[0]
    result = dict(head=phi.T @ gradient)
    if arm == "scalar_balanced":
        return result
    _, a, d, anchor, descriptor = cache
    rank = a.shape[1]
    product, absolute = parameters["head"][:rank], parameters["head"][rank:2 * rank]
    ga = (d * product + np.sign(a - d) * absolute) * gradient[:, None] * (1 - a * a)
    gd = (a * product - np.sign(a - d) * absolute) * gradient[:, None] * (1 - d * d)
    result["projection"] = anchor.T @ ga + descriptor.T @ gd
    return result


def pair_loss_gradient(parameters, arm, batch, scale):
    positive, pc = forward(parameters, arm, batch["anchor"], batch["positive"], batch["sp"], scale)
    negative, nc = forward(parameters, arm, batch["anchor"], batch["negative"], batch["sn"], scale)
    margin = positive - negative
    derivative = -1 / (1 + np.exp(np.clip(margin, -60, 60))) / len(margin)
    gp, gn = backward(parameters, arm, pc, derivative), backward(parameters, arm, nc, -derivative)
    gradient = {key: gp[key] + gn[key] for key in gp}
    loss = float(np.mean(np.logaddexp(0., -margin)))
    loss += .5 * DESIGN["head_l2"] * float(np.sum(parameters["head"] ** 2))
    gradient["head"] += DESIGN["head_l2"] * parameters["head"]
    if arm == "descriptor_learned":
        w = parameters["projection"]
        loss += .5 * DESIGN["projection_l2"] * float(np.mean(w * w))
        gradient["projection"] += DESIGN["projection_l2"] * w / w.size
    elif arm == "descriptor_fixed":
        gradient.pop("projection")
    return loss, gradient


def train_model(pairs, arm, seed, updates, rank=16):
    parameters = initial_parameters(arm, pairs["anchor"].shape[1], seed, rank)
    initial = {key: value.copy() for key, value in parameters.items()}
    # Equal sampling schedules across arms; separate stream from parameter initialization.
    rng = np.random.default_rng(seed + 10000)
    first, second, history = {}, {}, []
    maximum_data_projection_gradient = 0.
    for step in range(1, updates + 1):
        index = rng.choice(len(pairs["weight"]), DESIGN["batch_size"], p=pairs["weight"])
        batch = {key: pairs[key][index] for key in ("anchor", "positive", "negative", "sp", "sn")}
        loss, gradient = pair_loss_gradient(parameters, arm, batch, pairs["scale"])
        if arm == "descriptor_learned":
            data_gradient = gradient["projection"] - DESIGN["projection_l2"] * parameters["projection"] / parameters["projection"].size
            maximum_data_projection_gradient = max(maximum_data_projection_gradient, float(np.linalg.norm(data_gradient)))
        norm = float(np.sqrt(sum(float(np.sum(value * value)) for value in gradient.values())))
        if not np.isfinite(loss) or not np.isfinite(norm):
            raise ValueError("Nonfinite descriptor training loss/gradient")
        multiplier = min(1., DESIGN["gradient_clip"] / max(norm, 1e-12))
        for key, value in gradient.items():
            value = value * multiplier
            first[key] = .9 * first.get(key, np.zeros_like(value)) + .1 * value
            second[key] = .999 * second.get(key, np.zeros_like(value)) + .001 * value * value
            parameters[key] -= DESIGN["learning_rate"] * (first[key] / (1 - .9 ** step)) / (np.sqrt(second[key] / (1 - .999 ** step)) + 1e-8)
        if step == 1 or step % 100 == 0 or step == updates:
            history.append([step, loss, norm])
    if not all(np.isfinite(value).all() for value in parameters.values()):
        raise ValueError("Nonfinite learned parameters")
    changed = {key: not np.array_equal(initial[key], value) for key, value in parameters.items()}
    return parameters, np.asarray(history), dict(changed=changed, maximum_data_projection_gradient=maximum_data_projection_gradient)


def predict(video, parameters, arm, scale, stress=False):
    n, k, dim = video["descriptors"].shape
    scalar = video["stress_scalar"] if stress else video["scalar"]
    anchor = np.broadcast_to(video["anchor"], (n * k, dim))
    score, _ = forward(parameters, arm, anchor, video["descriptors"].reshape(-1, dim), scalar.reshape(-1, 9), scale)
    return np.where(video["valid"], score.reshape(n, k), -np.inf)


def evaluate_scores(video, scores):
    overlap, valid = video["overlap"], video["valid"]
    native = overlap[:, 0]
    best = overlap.max(1)
    choice = np.argmax(scores, axis=1)
    selected = overlap[np.arange(len(choice)), choice]
    masks = dict(all=np.ones(len(native), bool), native_matched=native >= .5,
        recoverable=(native < .5) & (best >= .5), hard=native < .5, low=native < .2,
        swapped=video["swapped"], known_negative_history=video["known_negative_history"])
    pair_accuracy = np.full(len(native), np.nan)
    for i, mask in enumerate(valid):
        pos = np.flatnonzero(mask & (overlap[i] >= .5))
        neg = np.flatnonzero(mask & (overlap[i] < .1))
        if len(pos) and len(neg):
            gap = (scores[i, pos, None] - scores[i, neg][None]).ravel()
            pair_accuracy[i] = np.mean((gap > 0).astype(float) + .5 * (gap == 0))
    rows = []
    for cohort, mask in masks.items():
        supported = mask & np.isfinite(pair_accuracy)
        rows.append(dict(cohort=cohort, frames=int(mask.sum()), pair_frames=int(supported.sum()),
            pair_accuracy=float(np.mean(pair_accuracy[supported])) if supported.any() else None,
            selected_match_ratio=float(np.mean(selected[mask] >= .5)) if mask.any() else None,
            native_match_ratio=float(np.mean(native[mask] >= .5)) if mask.any() else None,
            mean_immediate_iou_gain=float(np.mean(selected[mask] - native[mask])) if mask.any() else None,
            changed_frames=int(((choice > 0) & mask).sum()),
            corrected_low_frames=int(((native < .2) & (selected >= .5) & mask).sum()),
            regressed_matched_frames=int(((native >= .5) & (selected < .2) & mask).sum())))
    return rows, choice
