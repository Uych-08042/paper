"""Prepare, smoke, resume fixed training, and collect fit-only OOF rollouts."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint, write_cache, write_json
from stage4_runtime import Runtime
from stage20_compression_common import state_digest
from stage20_student_train import atomic_save, load_checkpoint
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, publish_identical, read_boxes, seal, tracking_iou
from stage28_response_collect import validate_runtime
from stage34_identity_experiment import audit_trace, frozen_digest, seed_runtime
from stage34_identity_runtime import gradient_norm
from stage35_preserve_common import (ARMS, BASELINES, DESIGN, SCHEMA, fold_samples,
    load_plan, load_sealed, nonempty_path, prepare)
from stage35_preserve_runtime import (forward_student, losses, make_student,
    parameters, restore_state, trainable_state)


def restore_latest(torch, directory, metadata):
    completed = []
    for path in sorted(directory.glob("step*.json")):
        receipt = load_sealed(path)
        checkpoint = path.with_suffix(".pt")
        if receipt.get("metadata") != metadata or receipt.get("checkpoint_sha256") != file_hash(checkpoint):
            raise ValueError("Completed checkpoint changed: " + str(checkpoint))
        if path.stem != "step{:06d}".format(receipt["step"]):
            raise ValueError("Checkpoint receipt step/name mismatch")
        completed.append((checkpoint, receipt["step"]))
    if not completed:
        return None
    path, step = completed[-1]
    value = load_checkpoint(torch, path)
    if value.get("metadata") != metadata or value["step"] != step or not 0 < step <= metadata["updates"]:
        raise ValueError("Checkpoint budget/source mismatch")
    print("[resume] {}: {}/{}".format(directory, step, metadata["updates"]), flush=True)
    return value


def save_step(torch, directory, value):
    path = directory / "step{:06d}.pt".format(value["step"])
    if path.with_suffix(".json").exists():
        raise ValueError("Never overwrite a committed checkpoint")
    if path.exists() and load_checkpoint(torch, path).get("metadata") != value["metadata"]:
        raise ValueError("Unidentified orphan checkpoint")
    atomic_save(torch, value, path)
    write_json(path.with_suffix(".json"), seal(dict(schema=SCHEMA, role="completed_training_step",
        step=value["step"], metadata=value["metadata"], checkpoint_sha256=file_hash(path))))
    return path


def train_arm(rt, plan, directory, fold, seed, arm, smoke=False, image_reader=checked_image):
    torch = rt.torch
    seed_runtime(torch, seed + 100 * fold)
    net = make_student(rt)
    initial_digest = state_digest(net)
    before = trainable_state(net)
    frozen = frozen_digest(net, "representation_standard")
    optimizer = torch.optim.AdamW(list(parameters(net).values()), lr=DESIGN["learning_rate"], weight_decay=DESIGN["weight_decay"])
    updates = DESIGN["smoke_updates"] if smoke else DESIGN["updates"]
    samples, weights = fold_samples(plan, fold)
    draws = np.random.default_rng(seed + 100 * fold).choice(len(samples), (updates, DESIGN["batch_size"]), p=weights)
    metadata = json.loads(json.dumps(dict(schema=SCHEMA, plan_fingerprint=plan["fingerprint"],
        fold=fold, seed=seed, arm=arm, updates=updates, mode="smoke" if smoke else "full",
        runtime=rt.identity, teacher_sha256=state_digest(rt.net), initial_student_sha256=initial_digest,
        fit_videos=plan["folds"][fold]["fit"], held_out_videos=plan["folds"][fold]["held_out"],
        draws_fingerprint=fingerprint(draws.tolist()))))
    state = restore_latest(torch, directory, metadata)
    start, history = (0, []) if state is None else (state["step"], state["history"])
    gates = dict(easy=0, hard=0) if state is None else state["gates"]
    maxima = dict(kd=0., correction=0.) if state is None else state["maximum_applied_gradient"]
    identity_error = 0. if state is None else state["initial_fast_max_error"]
    if state is not None:
        restore_state(net, state["parameters"])
        optimizer.load_state_dict(state["optimizer"])
    entries = {e["sequence"]: e for e in plan["fit"] if e["sequence"] in plan["folds"][fold]["fit"]}
    templates = {}
    for step in range(start + 1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        totals = np.zeros(4)
        for index in draws[step - 1]:
            sample = samples[int(index)]
            entry = entries[sample["sequence"]]
            name = entry["sequence"]
            if name not in templates:
                with torch.no_grad():
                    rt.initialize(image_reader(plan["root"], entry["frames"][0]), entry["initial_box"])
                templates[name] = rt.tracker.z_dict1
            rt.tracker.z_dict1 = templates[name]
            rt.tracker.state = sample["previous"].copy()
            image = image_reader(plan["root"], entry["frames"][sample["frame_id"]])
            search, factor = rt.prepare(image)
            np.testing.assert_allclose(factor, sample["factor"], atol=1e-12, rtol=0)
            if step == 1:
                with torch.no_grad():
                    a, b = rt.head(rt.prefix(search)), forward_student(rt, net, search)
                    error = max(float((a[k] - b[k]).abs().max()) for k in a if torch.is_tensor(a[k]))
                    identity_error = max(identity_error, error)
                    if error > 1e-6:
                        raise ValueError("Initial six-block student is not Fast-equivalent")
            target = torch.tensor([sample["target_roi"]], device=search.device, dtype=torch.float32)
            reliable = sample["cohort"] == "easy"
            values = losses(rt, net, search, target, arm, reliable)
            if step <= DESIGN["smoke_updates"] or step % DESIGN["save_every"] == 0 or step == updates:
                maxima["kd"] = max(maxima["kd"], gradient_norm(torch, values[2], parameters(net).values()))
                maxima["correction"] = max(maxima["correction"], gradient_norm(torch, values[3], parameters(net).values()))
            (values[0] / DESIGN["batch_size"]).backward()
            totals += np.array([float(v.detach()) for v in values]) / DESIGN["batch_size"]
            gates[sample["cohort"]] += 1
        norm = torch.nn.utils.clip_grad_norm_(list(parameters(net).values()), DESIGN["gradient_clip"])
        if not np.isfinite([*totals, float(norm)]).all():
            raise ValueError("Nonfinite training loss/gradient")
        optimizer.step()
        if step == 1 or step % DESIGN["save_every"] == 0 or step == updates:
            history.append(dict(step=step, loss=float(totals[0]), supervised=float(totals[1]),
                applied_kd=float(totals[2]), applied_correction=float(totals[3]), gradient_norm=float(norm)))
            print("[fold{} seed{} {}] {}/{} loss={:.4f}".format(fold, seed, arm, step, updates, totals[0]), flush=True)
        if step % DESIGN["save_every"] == 0 or step == updates:
            if frozen_digest(net, "representation_standard") != frozen:
                raise ValueError("Frozen student state changed")
            save_step(torch, directory, dict(metadata=metadata, step=step, parameters=trainable_state(net),
                optimizer=optimizer.state_dict(), history=history, gates=gates,
                maximum_applied_gradient=maxima, initial_fast_max_error=identity_error))
    path = directory / "step{:06d}.pt".format(updates)
    after = trainable_state(net)
    if frozen_digest(net, "representation_standard") != frozen:
        raise ValueError("Frozen student state changed")
    return dict(fold=fold, seed=seed, arm=arm, updates=updates,
        checkpoint=str(path.resolve()), checkpoint_sha256=file_hash(path),
        initial_student_sha256=initial_digest, draws_fingerprint=metadata["draws_fingerprint"],
        prefix_changed=any(not torch.equal(before[n], after[n]) for n in before),
        frozen_student_unchanged=True, student_blocks=len(net.backbone.blocks),
        no_identity_branch=not hasattr(net, "spatial_identity"),
        student_parameters=sum(p.numel() for p in net.parameters()),
        teacher_parameters=sum(p.numel() for p in rt.net.parameters()),
        maximum_applied_gradient=maxima, gates=gates, initial_fast_max_error=identity_error,
        history=history)


def train(rt, plan, root, seed, smoke=False, image_reader=checked_image):
    if seed not in DESIGN["seeds"] or (smoke and seed != DESIGN["seeds"][0]):
        raise ValueError("Use the two fixed seeds; smoke uses the first")
    if not smoke:
        report = load_sealed(root / "smoke/stage35_training_decision.json")
        if report.get("status") != "PASS_CORRECTIVE_DISTILLATION_SMOKE" or report.get("plan_fingerprint") != plan["fingerprint"]:
            raise ValueError("Need passing matched smoke before full training")
    directory = root / ("smoke" if smoke else "training/seed{}".format(seed))
    teacher = state_digest(rt.net)
    records = []
    with writer_lock(directory):
        for fold in ([0] if smoke else range(DESIGN["folds"])):
            for arm in ARMS:
                records.append(train_arm(rt, plan, directory / "fold{}".format(fold) / arm,
                    fold, seed, arm, smoke, image_reader))
        if state_digest(rt.net) != teacher:
            raise ValueError("Original SGLA teacher changed")
        passed = all(r["prefix_changed"] and r["frozen_student_unchanged"] and r["no_identity_branch"] and
            r["student_blocks"] == 6 and r["student_parameters"] < r["teacher_parameters"] and
            r["initial_fast_max_error"] <= 1e-6 for r in records)
        for fold in sorted({r["fold"] for r in records}):
            values = [r for r in records if r["fold"] == fold]
            passed &= len({r["initial_student_sha256"] for r in values}) == 1
            passed &= len({r["draws_fingerprint"] for r in values}) == 1
            for r in values:
                expected_kd = r["arm"] in ("preserve", "preserve_correct", "distill_all_correct")
                expected_c = r["arm"] in ("correct", "preserve_correct", "distill_all_correct")
                passed &= (r["maximum_applied_gradient"]["kd"] > 0) == expected_kd
                passed &= (r["maximum_applied_gradient"]["correction"] > 0) == expected_c
        load_plan(root)
        status = ("PASS_CORRECTIVE_DISTILLATION_SMOKE" if passed else "FAIL_SMOKE_INTEGRITY") if smoke else (
            "COMPLETE_MATCHED_OOF_TRAINING" if passed else "FAIL_TRAINING_INTEGRITY")
        report = seal(dict(schema=SCHEMA, role="matched_fit_only_training_integrity", status=status,
            plan_fingerprint=plan["fingerprint"], teacher_sha256=teacher, seed=seed, arms=records,
            original_checking_labels_loaded=False, held_out_videos_used_for_fitting=False,
            warning="Optimizer/integrity result only; no OOF gain or speed established"))
        publish_identical(directory / "stage35_training_decision.json", report)
    print("Stage-35 training: " + status, flush=True)
    return report


def evaluation_models(rt, plan, root):
    models, bindings = {}, {}
    teacher = state_digest(rt.net)
    for seed in DESIGN["seeds"]:
        path = root / "training/seed{}".format(seed) / "stage35_training_decision.json"
        report = load_sealed(path)
        expected = {(f, a) for f in range(DESIGN["folds"]) for a in ARMS}
        if (report.get("status") != "COMPLETE_MATCHED_OOF_TRAINING" or report.get("seed") != seed or
                report.get("plan_fingerprint") != plan["fingerprint"] or report.get("teacher_sha256") != teacher or
                {(r["fold"], r["arm"]) for r in report["arms"]} != expected or len(report["arms"]) != len(expected)):
            raise ValueError("Freeze both seeds / all 40 models before any OOF rollout")
        bindings[str(path)] = file_hash(path)
        for record in report["arms"]:
            checkpoint = Path(record["checkpoint"])
            if file_hash(checkpoint) != record["checkpoint_sha256"]:
                raise ValueError("Frozen checkpoint changed")
            state = load_checkpoint(rt.torch, checkpoint)
            meta = state["metadata"]
            fold, arm = record["fold"], record["arm"]
            if (state["step"] != DESIGN["updates"] or meta["plan_fingerprint"] != plan["fingerprint"] or
                    meta["mode"] != "full" or meta["fold"] != fold or meta["arm"] != arm or meta["seed"] != seed or
                    meta["fit_videos"] != plan["folds"][fold]["fit"] or
                    meta["held_out_videos"] != plan["folds"][fold]["held_out"] or meta["teacher_sha256"] != teacher):
                raise ValueError("Wrong OOF source/budget checkpoint")
            net = make_student(rt)
            if state_digest(net) != meta["initial_student_sha256"]:
                raise ValueError("OOF initialization differs from original Fast")
            restore_state(net, state["parameters"])
            net.requires_grad_(False)
            models[(fold, arm + "_seed" + str(seed))] = net
            bindings[str(checkpoint)] = file_hash(checkpoint)
    return models, bindings


def collect_trace(rt, plan, entry, method, net=None, image_reader=checked_image):
    torch = rt.torch
    with torch.no_grad():
        rt.initialize(image_reader(plan["root"], entry["frames"][0]), entry["initial_box"])
        previous = np.asarray(entry["initial_box"], float)
        boxes, parents, times, calls = [], [], [], [0]
        head = rt.net.box_head if net is None else net.box_head
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
                    output = forward_student(rt, net, search)
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
    # No future GT is used by inference, scoring, cropping, or state updates.
    annotation = Path(plan["root"]) / entry["annotation"]
    if file_hash(annotation) != entry["annotation_sha256"]:
        raise ValueError("OOF annotations changed")
    gt = read_boxes(annotation)
    n = len(boxes)
    if len(gt) != n + 1 or calls[0] != n:
        raise ValueError("Whole-video coverage / one-head-call invariant failed")
    data = dict(frame_id=np.arange(1, n + 1), previous=np.asarray(parents), boxes=np.asarray(boxes),
        gt=gt[1:], iou=tracking_iou(np.asarray(boxes), gt[1:]), elapsed_ms=np.asarray(times), main_head_calls=np.asarray(calls[0]))
    audit_trace(data, entry)
    return data


def evaluate(rt, plan, root, image_reader=checked_image):
    models, bindings = evaluation_models(rt, plan, root)
    methods = list(BASELINES) + sorted(a + "_seed" + str(s) for a in ARMS for s in DESIGN["seeds"])
    directory = root / "oof"
    run = seal(dict(schema=SCHEMA, role="consumed_development_group_oof_autonomous_rollouts",
        plan_fingerprint=plan["fingerprint"], methods=methods, runtime=rt.identity,
        frozen_model_sha256=bindings, gt_use="First-frame initialization only; future labels after full rollout",
        timing_role="Instrumented crop/model/decode wall time, excludes images and GT logs; no speed claim"))
    with writer_lock(directory):
        publish_identical(directory / "run.json", run)
        for i, entry in enumerate(plan["fit"]):
            fold = next(f["fold"] for f in plan["folds"] if entry["sequence"] in f["held_out"])
            order = methods[i % len(methods):] + methods[:i % len(methods)]
            for method in order:
                path = directory / (entry["sequence"] + "__" + method + ".npz")
                meta = dict(schema=SCHEMA, run_fingerprint=run["fingerprint"], sequence=entry["sequence"], fold=fold, method=method)
                if path.exists():
                    with np.load(path, allow_pickle=False) as a:
                        if json.loads(str(a["metadata_json"].item())) != meta:
                            raise ValueError("Mixed OOF cache")
                        data = {k: a[k] for k in a.files if k != "metadata_json"}
                else:
                    data = collect_trace(rt, plan, entry, method, models.get((fold, method)), image_reader)
                    write_cache(path, data, meta)
                audit_trace(data, entry)
                print("[OOF, audited] " + path.name, flush=True)
        for name, digest in bindings.items():
            if file_hash(name) != digest:
                raise ValueError("Frozen model changed during OOF evaluation")
        load_plan(root)
    print("Next: python tracking/stage35_preserve_summary.py --input " + str(root), flush=True)


def status(root, plan):
    for seed in DESIGN["seeds"]:
        for fold in range(DESIGN["folds"]):
            for arm in ARMS:
                directory = root / "training/seed{}".format(seed) / "fold{}".format(fold) / arm
                receipts = sorted(directory.glob("step*.json"))
                step = 0
                if receipts:
                    receipt = load_sealed(receipts[-1])
                    if receipt["metadata"]["plan_fingerprint"] != plan["fingerprint"] or file_hash(receipts[-1].with_suffix(".pt")) != receipt["checkpoint_sha256"]:
                        raise ValueError("Progress checkpoint fingerprint mismatch")
                    step = receipt["step"]
                print("seed{} fold{} {:<17s} committed={}/{}".format(seed, fold, arm, step, DESIGN["updates"]))
    print("OOF traces: {}/96; training progress is not a gain result".format(len(list((root / "oof").glob("*.npz")))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True, help="Stage-34 root for prepare; Stage-35 root otherwise")
    parser.add_argument("--mode", choices=("prepare", "smoke", "train", "evaluate", "status"), required=True)
    parser.add_argument("--output_dir", type=nonempty_path)
    parser.add_argument("--checkpoint", type=nonempty_path)
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DESIGN["seeds"][0])
    args = parser.parse_args()
    if args.mode == "prepare":
        if not args.output_dir:
            parser.error("prepare requires --output_dir")
        root = Path(args.output_dir).expanduser().resolve()
        with writer_lock(root):
            prepare(args.input, root)
        return
    root = Path(args.input).expanduser().resolve()
    plan = load_plan(root)
    if args.mode == "status":
        status(root, plan)
        return
    if not args.checkpoint or file_hash(Path(args.checkpoint).expanduser()) != plan["runtime"]["checkpoint_sha256"]:
        parser.error("Use the frozen original SGLA --checkpoint")
    import torch
    seed_runtime(torch, DESIGN["fold_seed"])
    rt = Runtime(args.parameter, args.gpu, args.checkpoint)
    validate_runtime(rt, plan)
    if args.mode in ("smoke", "train"):
        report = train(rt, plan, root, args.seed, args.mode == "smoke")
        if report["status"].startswith("FAIL_"):
            raise SystemExit(2)
    else:
        evaluate(rt, plan, root)


if __name__ == "__main__":
    main()
