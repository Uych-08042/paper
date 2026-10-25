"""Prepare, matched train and autonomous fresh-group spatial-identity evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_cache, write_json
from stage4_runtime import Runtime
from stage20_compression_common import state_digest
from stage20_student_train import atomic_save, load_checkpoint
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, read_boxes, seal, tracking_iou, publish_identical
from stage28_response_collect import validate_runtime
from stage34_identity_common import ARMS, DESIGN, METHODS, SCHEMA, load_plan, load_sealed, nonempty_path, prepare_plan, sample_weights
from stage34_identity_runtime import (allowed_parameters, forward_student, gradient_norm, losses, make_student,
    restore_state, template_weights, trainable_state)


def seed_runtime(torch, seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def frozen_digest(net, arm):
    import hashlib
    trainable = set(allowed_parameters(net, arm))
    digest = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        if name not in trainable:
            array = value.detach().cpu().contiguous().numpy()
            digest.update(name.encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def restore_latest(torch, directory, metadata):
    completed = []
    for receipt_path in sorted(directory.glob("step*.json")):
        receipt = load_sealed(receipt_path)
        checkpoint = receipt_path.with_suffix(".pt")
        if receipt.get("metadata") != metadata or receipt.get("checkpoint_sha256") != file_hash(checkpoint):
            raise ValueError("Completed training checkpoint changed: " + str(checkpoint))
        completed.append(checkpoint)
    if not completed:
        return None
    value = load_checkpoint(torch, completed[-1])
    if value.get("metadata") != metadata or not 0 < value["step"] <= metadata["updates"]:
        raise ValueError("Checkpoint budget/source mismatch")
    print("[resume] {} at {}/{}".format(directory.name, value["step"], metadata["updates"]), flush=True)
    return value


def save_step(torch, directory, value):
    path = directory / ("step{:06d}.pt".format(value["step"]))
    receipt = path.with_suffix(".json")
    if receipt.exists():
        raise ValueError("Never overwrite a committed checkpoint")
    if path.exists():
        orphan = load_checkpoint(torch, path)
        if orphan.get("metadata") != value["metadata"]:
            raise ValueError("Unidentified orphan checkpoint")
    atomic_save(torch, value, path)
    write_json(receipt, seal(dict(schema=SCHEMA, role="completed_training_step", metadata=value["metadata"], checkpoint_sha256=file_hash(path))))
    return path


def train_arm(rt, plan, directory, arm, seed, smoke=False, image_reader=checked_image):
    torch = rt.torch
    seed_runtime(torch, seed)
    net = make_student(rt, arm)
    before = trainable_state(net, arm)
    frozen_before = frozen_digest(net, arm)
    optimizer = torch.optim.AdamW(list(allowed_parameters(net, arm).values()), lr=DESIGN["learning_rate"], weight_decay=DESIGN["weight_decay"])
    updates = DESIGN["smoke_updates"] if smoke else DESIGN["updates"]
    metadata = dict(schema=SCHEMA, plan_fingerprint=plan["fingerprint"], arm=arm, seed=seed,
        mode="smoke" if smoke else "full", updates=updates, runtime=rt.identity,
        teacher_state_sha256=state_digest(rt.net))
    metadata = json.loads(json.dumps(metadata))
    state = restore_latest(torch, directory, metadata)
    history, max_contrast_prefix = [], 0.
    start = 0
    if state is not None:
        restore_state(net, arm, state["parameters"])
        optimizer.load_state_dict(state["optimizer"])
        start, history, max_contrast_prefix = state["step"], state["history"], state["max_contrast_prefix"]
    samples = plan["samples"]
    draws = np.random.default_rng(seed).choice(len(samples), (updates, DESIGN["batch_size"]), p=sample_weights(samples))
    entries = {e["sequence"]: e for e in plan["fit"]}
    templates = {}
    weights = {}
    root = plan["root"]
    for step in range(start + 1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        total_loss, supervised_loss, contrast_loss = 0., 0., 0.
        for index in draws[step - 1]:
            sample = samples[int(index)]
            entry = entries[sample["sequence"]]
            name = entry["sequence"]
            if name not in templates:
                initial = image_reader(root, entry["frames"][0])
                with torch.no_grad():
                    rt.initialize(initial, entry["initial_box"])
                templates[name] = rt.tracker.z_dict1
                weights[name] = template_weights(rt, entry)
            rt.tracker.z_dict1 = templates[name]
            rt.tracker.state = sample["previous"].copy()
            image = image_reader(root, entry["frames"][sample["frame_id"]])
            search, factor = rt.prepare(image)
            np.testing.assert_allclose(factor, sample["factor"], atol=1e-12, rtol=0)
            target = torch.tensor([sample["target_roi"]], device=search.device, dtype=torch.float32)
            total, supervised, contrast, _, _ = losses(rt, net, search, weights[name], target, arm)
            if step == 1 or step % DESIGN["save_every"] == 0 or step == updates:
                prefix = [p for n, p in allowed_parameters(net, arm).items() if n.startswith("backbone.")]
                if arm.endswith("_contrast"):
                    max_contrast_prefix = max(max_contrast_prefix, gradient_norm(torch, contrast, prefix))
            (total / DESIGN["batch_size"]).backward()
            total_loss += float(total.detach()) / DESIGN["batch_size"]
            supervised_loss += float(supervised.detach()) / DESIGN["batch_size"]
            contrast_loss += float(contrast.detach()) / DESIGN["batch_size"]
        norm = torch.nn.utils.clip_grad_norm_(list(allowed_parameters(net, arm).values()), DESIGN["gradient_clip"])
        if not np.isfinite([total_loss, float(norm)]).all():
            raise ValueError("Nonfinite matched training loss/gradient")
        optimizer.step()
        if step == 1 or step % DESIGN["save_every"] == 0 or step == updates:
            history.append(dict(step=step, loss=total_loss, supervised=supervised_loss, contrast=contrast_loss, gradient_norm=float(norm)))
            print("[{} seed{}] {}/{} loss={:.4f}".format(arm, seed, step, updates, total_loss), flush=True)
        if step % DESIGN["save_every"] == 0 or step == updates:
            if frozen_digest(net, arm) != frozen_before:
                raise ValueError("A frozen student parameter or buffer changed")
            checkpoint = save_step(torch, directory, dict(metadata=metadata, step=step,
                parameters=trainable_state(net, arm), optimizer=optimizer.state_dict(), history=history,
                max_contrast_prefix=max_contrast_prefix))
    if state is not None and start == updates:
        checkpoint = directory / ("step{:06d}.pt".format(updates))
    after = trainable_state(net, arm)
    changed = {name: not torch.equal(before[name], after[name]) for name in before}
    if frozen_digest(net, arm) != frozen_before:
        raise ValueError("Frozen student state changed")
    return dict(arm=arm, seed=seed, updates=updates, checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=file_hash(checkpoint), prefix_changed=any(v for n, v in changed.items() if n.startswith("backbone.")),
        identity_changed=any(v for n, v in changed.items() if n.startswith("spatial_identity.")),
        applied_contrast_prefix_gradient=max_contrast_prefix, frozen_student_unchanged=True,
        teacher_parameters=sum(p.numel() for p in rt.net.parameters()), student_parameters=sum(p.numel() for p in net.parameters()),
        student_blocks=len(net.backbone.blocks), identity_gain=float(net.spatial_identity.gain.detach()), history=history)


def train_experiment(rt, plan, root, seed, smoke=False, image_reader=checked_image):
    if seed not in DESIGN["seeds"] or (smoke and seed != DESIGN["seeds"][0]):
        raise ValueError("Use only the fixed seed set; smoke uses its first seed")
    if not smoke:
        prior = load_sealed(root / "smoke" / "stage34_training_decision.json")
        if prior.get("status") != "PASS_SPATIAL_IDENTITY_SMOKE" or prior.get("plan_fingerprint") != plan["fingerprint"]:
            raise ValueError("Need the matched passing smoke before full training")
    directory = root / ("smoke" if smoke else "training/seed{}".format(seed))
    teacher_before = state_digest(rt.net)
    results = []
    with writer_lock(directory):
        for arm in ARMS:
            results.append(train_arm(rt, plan, directory / arm, arm, seed, smoke, image_reader))
        if state_digest(rt.net) != teacher_before:
            raise ValueError("Original SGLA teacher changed")
        load_plan(root)
        passed = all(r["identity_changed"] and r["frozen_student_unchanged"] and r["student_blocks"] == 6
            and r["student_parameters"] < r["teacher_parameters"]
            and r["prefix_changed"] == r["arm"].startswith("representation_") for r in results)
        passed = passed and next(r for r in results if r["arm"] == "representation_contrast")["applied_contrast_prefix_gradient"] > 0
        status = ("PASS_SPATIAL_IDENTITY_SMOKE" if passed else "FAIL_SPATIAL_IDENTITY_SMOKE") if smoke else (
                  "COMPLETE_MATCHED_TRAINING" if passed else "FAIL_TRAINING_INTEGRITY")
        report = seal(dict(schema=SCHEMA, role="matched_representation_training", status=status,
            plan_fingerprint=plan["fingerprint"], seed=seed, checking_labels_loaded=False,
            teacher_unchanged=True, teacher_state_sha256=teacher_before, arms=results,
            warning="Training integrity only; no learned-policy gain, speed or external claim"))
        write_json(directory / "stage34_training_decision.json", report)
    print("Stage-34 training: " + status, flush=True)
    return report


def evaluation_models(rt, plan, root):
    models = {}
    hashes = {}
    teacher_digest = state_digest(rt.net)
    for seed in DESIGN["seeds"]:
        decision_path = root / "training" / ("seed{}".format(seed)) / "stage34_training_decision.json"
        report = load_sealed(decision_path)
        if (report.get("status") != "COMPLETE_MATCHED_TRAINING" or report.get("plan_fingerprint") != plan["fingerprint"]
                or report.get("teacher_state_sha256") != teacher_digest):
            raise ValueError("Complete all three fixed seeds before opening fresh checking outcomes")
        hashes[str(decision_path)] = file_hash(decision_path)
        if {r["arm"] for r in report["arms"]} != set(ARMS):
            raise ValueError("Incomplete matched training arms")
        for record in report["arms"]:
            path = Path(record["checkpoint"])
            if file_hash(path) != record["checkpoint_sha256"]:
                raise ValueError("Frozen trained checkpoint changed")
            seed_runtime(rt.torch, seed)
            net = make_student(rt, record["arm"])
            state = load_checkpoint(rt.torch, path)
            if (state["step"] != DESIGN["updates"] or state["metadata"]["plan_fingerprint"] != plan["fingerprint"]
                    or state["metadata"]["arm"] != record["arm"] or state["metadata"]["seed"] != seed):
                raise ValueError("Wrong final checkpoint")
            restore_state(net, record["arm"], state["parameters"])
            net.requires_grad_(False)
            key = record["arm"] + "_seed" + str(seed)
            models[key] = net
            hashes[str(path)] = file_hash(path)
    return models, hashes


def collect_trace(rt, plan, entry, method, net=None, image_reader=checked_image):
    import time
    torch = rt.torch
    with torch.no_grad():
        initial = image_reader(plan["root"], entry["frames"][0])
        rt.initialize(initial, entry["initial_box"])
        weights = template_weights(rt, entry) if net is not None else None
        previous, boxes, times, parents = np.asarray(entry["initial_box"], float), [], [], []
        calls = [0]
        head = net.box_head if net is not None else rt.net.box_head
        hook = head.register_forward_hook(lambda *unused: calls.__setitem__(0, calls[0] + 1))
        try:
            for index, frame in enumerate(entry["frames"][1:], 1):
                image = image_reader(plan["root"], frame)
                rt.tracker.state = previous.tolist()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                search, factor = rt.prepare(image)
                if net is None:
                    tokens = rt.prefix(search)
                    output = rt.head(tokens) if method == "fast_initial" else rt.candidate(tokens, rt.selected_layer(rt.selector(tokens)))
                else:
                    output, _ = forward_student(rt, net, search, weights)
                box = np.asarray(rt.decode(output, factor, image), float)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
                parents.append(previous.copy())
                boxes.append(box.copy())
                previous = box
                rt.tracker.frame_id = index
                if index % 500 == 0:
                    print("[{}] {} {}/{}".format(method, entry["sequence"], index, len(entry["frames"]) - 1), flush=True)
        finally:
            hook.remove()
    # Future checking labels are first opened after the autonomous rollout.
    annotation = Path(plan["root"]) / entry["annotation"]
    if file_hash(annotation) != entry["annotation_sha256"]:
        raise ValueError("Checking annotations changed")
    gt = read_boxes(annotation)
    n = len(boxes)
    if len(gt) != n + 1 or calls[0] != n:
        raise ValueError("Checking coverage or one-head-per-frame invariant failed")
    return dict(frame_id=np.arange(1, n + 1), previous=np.asarray(parents), boxes=np.asarray(boxes), gt=gt[1:],
                iou=tracking_iou(np.asarray(boxes), gt[1:]), elapsed_ms=np.asarray(times), main_head_calls=np.asarray(calls[0]))


def audit_trace(data, entry):
    n = len(entry["frames"]) - 1
    shapes = dict(frame_id=(n,), previous=(n, 4), boxes=(n, 4), gt=(n, 4), iou=(n,), elapsed_ms=(n,), main_head_calls=())
    if set(data) != set(shapes) or any(data[k].shape != shape for k, shape in shapes.items()):
        raise ValueError("Checking trace shape mismatch")
    if not all(np.isfinite(v).all() for v in data.values()) or (data["boxes"][:, 2:] <= 0).any():
        raise ValueError("Nonfinite/invalid trace")
    np.testing.assert_array_equal(data["frame_id"], np.arange(1, n + 1))
    np.testing.assert_array_equal(data["previous"], np.vstack([entry["initial_box"], data["boxes"][:-1]]))
    np.testing.assert_allclose(data["iou"], tracking_iou(data["boxes"], data["gt"]), atol=1e-12, rtol=0)
    if int(data["main_head_calls"]) != n:
        raise ValueError("Head call count changed")


def evaluate(rt, plan, root, image_reader=checked_image):
    models, bindings = evaluation_models(rt, plan, root)  # No checking outcomes opened until all arms are frozen.
    methods = list(METHODS[:2]) + sorted(models)
    directory = root / "checking"
    run = seal(dict(schema=SCHEMA, role="fresh_group_autonomous_single_path_checking", plan_fingerprint=plan["fingerprint"],
        methods=methods, runtime=rt.identity, frozen_model_sha256=bindings,
        gt_use="First-frame initialization; future annotations opened after each full-video rollout",
        timing_role="Instrumented crop/model/decode wall time excluding image IO and labels; not deployment FPS"))
    with writer_lock(directory):
        publish_identical(directory / "run.json", run)
        for i, entry in enumerate(plan["checking"]):
            order = methods[i % len(methods):] + methods[:i % len(methods)]
            for method in order:
                path = directory / (entry["sequence"] + "__" + method + ".npz")
                metadata = dict(schema=SCHEMA, run_fingerprint=run["fingerprint"], sequence=entry["sequence"], method=method)
                if path.exists():
                    with np.load(path, allow_pickle=False) as a:
                        if json.loads(str(a["metadata_json"].item())) != metadata:
                            raise ValueError("Mixed checking cache")
                        data = {k: a[k] for k in a.files if k != "metadata_json"}
                else:
                    data = collect_trace(rt, plan, entry, method, models.get(method), image_reader)
                    audit_trace(data, entry)
                    write_cache(path, data, metadata)
                audit_trace(data, entry)
                print("[checking, audited] " + path.name, flush=True)
    print("Next: python tracking/stage34_identity_summary.py --input " + str(root), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True, help="Stage-31 root for prepare; Stage-34 root otherwise")
    parser.add_argument("--mode", choices=("prepare", "smoke", "train", "evaluate"), required=True)
    parser.add_argument("--stage33_dir", type=nonempty_path)
    parser.add_argument("--output_dir", type=nonempty_path)
    parser.add_argument("--checkpoint", type=nonempty_path)
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DESIGN["seeds"][0])
    args = parser.parse_args()
    if args.mode == "prepare":
        if not args.stage33_dir or not args.output_dir:
            parser.error("prepare needs --stage33_dir and --output_dir")
        root = Path(args.output_dir).expanduser().resolve()
        with writer_lock(root):
            prepare_plan(args.input, args.stage33_dir, root)
        return
    root = Path(args.input).expanduser().resolve()
    plan = load_plan(root)
    if not args.checkpoint or file_hash(Path(args.checkpoint).expanduser()) != plan["runtime"]["checkpoint_sha256"]:
        parser.error("Use --checkpoint with the frozen original SGLA weights")
    import torch
    seed_runtime(torch, DESIGN["seed"])  # Same base initialization across all seed processes.
    rt = Runtime(args.parameter, args.gpu, args.checkpoint)
    validate_runtime(rt, plan)
    if args.mode in ("smoke", "train"):
        report = train_experiment(rt, plan, root, args.seed, args.mode == "smoke")
        if report["status"].startswith("FAIL_"):
            raise SystemExit(2)
    else:
        evaluate(rt, plan, root)


if __name__ == "__main__":
    main()
