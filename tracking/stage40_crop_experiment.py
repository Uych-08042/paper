"""Matched two-view training, resumable checkpoints and autonomous OOF checking."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from stage4_common import file_hash, fingerprint, write_cache
from stage20_compression_common import state_digest
from stage20_student_train import atomic_save, load_checkpoint
from stage18_tracking_loss import tracking_loss
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, nonempty_path, publish_identical, read_boxes, seal, tracking_iou
from stage28_response_collect import validate_runtime
from stage28_response_common import cpu_array
from stage34_identity_experiment import frozen_digest, seed_runtime
from stage35_preserve_runtime import forward_student, make_student, parameters, restore_state, trainable_state
from stage35_preserve_experiment import collect_trace
from stage37_crossed_common import fold_for
from stage39_perturb_experiment import run_branch
from stage40_crop_common import (ARMS, DESIGN, SCHEMA, SEEDS, arm_seed, audit_stress, baseline, load_plan,
    load_result, methods, prepare, prior, read_sealed, result_path, schedule, target_roi, verify_bindings)


def save_step(torch, directory, value):
    path = directory / "step{:06d}.pt".format(value["step"])
    if path.with_suffix(".json").exists():
        raise ValueError("Never overwrite a committed Stage-40 checkpoint")
    if path.exists() and load_checkpoint(torch, path).get("metadata") != value["metadata"]:
        raise ValueError("Mixed orphan checkpoint")
    atomic_save(torch, value, path)
    publish_identical(path.with_suffix(".json"), seal(dict(schema=SCHEMA, metadata=value["metadata"],
        step=value["step"], checkpoint_sha256=file_hash(path))))
    return path


def restore_latest(torch, directory, metadata):
    latest = None
    for path in sorted(directory.glob("step*.json")):
        receipt = read_sealed(path, SCHEMA)
        if (receipt.get("metadata") != metadata or path.stem != "step{:06d}".format(receipt["step"]) or
                not 0 < receipt["step"] <= metadata["updates"] or
                file_hash(path.with_suffix(".pt")) != receipt["checkpoint_sha256"]):
            raise ValueError("Completed Stage-40 checkpoint changed: " + str(path.with_suffix(".pt")))
        latest = receipt, path.with_suffix(".pt")
    if latest is None:
        return None
    receipt, path = latest
    value = load_checkpoint(torch, path)
    if value.get("metadata") != metadata or value.get("step") != receipt["step"]:
        raise ValueError("Wrong saved training step/source")
    print("[resume] {}: {}/{}".format(directory, value["step"], metadata["updates"]), flush=True)
    return value


def train_arm(rt, plan, directory, fold, seed, arm, smoke=False, image_reader=checked_image):
    torch = rt.torch
    seed_runtime(torch, seed + 100 * fold)
    net = make_student(rt).eval()
    initial, before = state_digest(net), trainable_state(net)
    frozen = frozen_digest(net, "representation_standard")
    optimizer = torch.optim.AdamW(list(parameters(net).values()), lr=DESIGN["learning_rate"], weight_decay=DESIGN["weight_decay"])
    updates = DESIGN["smoke_updates"] if smoke else DESIGN["updates"]
    samples, draws, perturbations = schedule(plan, fold, seed, updates)
    meta = json.loads(json.dumps(dict(schema=SCHEMA, plan_fingerprint=plan["fingerprint"], mode="smoke" if smoke else "full",
        fold=fold, seed=seed, arm=arm, updates=updates, runtime=rt.identity, teacher_sha256=state_digest(rt.net),
        initial_student_sha256=initial, draws_fingerprint=fingerprint(draws.tolist()),
        perturbations_fingerprint=fingerprint(perturbations.tolist()), fit_videos=plan["folds"][fold]["fit"],
        held_out_videos=plan["folds"][fold]["held_out"])))
    if arm not in ARMS or seed not in SEEDS or set(meta["fit_videos"]) & set(meta["held_out_videos"]):
        raise ValueError("Wrong matched training arm/OOF isolation")
    state = restore_latest(torch, directory, meta)
    start = 0 if state is None else state["step"]
    history, calls, max_gradient = ([], 0, 0.) if state is None else (state["history"], state["head_calls"], state["maximum_gradient"])
    initial_error = 0. if state is None else state["initial_fast_max_error"]
    if state is not None:
        restore_state(net, state["parameters"])
        optimizer.load_state_dict(state["optimizer"])
    entries = {e["sequence"]: e for e in plan["fit"] if e["sequence"] in meta["fit_videos"]}
    labels, templates = {}, {}
    for step in range(start + 1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        totals = np.zeros(3)
        for column, sample_index in enumerate(draws[step - 1]):
            sample = samples[int(sample_index)]
            entry = entries[sample["sequence"]]  # A held-out sequence cannot be indexed here.
            name = entry["sequence"]
            if name not in templates:
                with torch.no_grad():
                    rt.initialize(image_reader(plan["root"], entry["frames"][0]), entry["initial_box"])
                templates[name] = rt.tracker.z_dict1
                annotation = Path(plan["root"]) / entry["annotation"]
                verify_bindings({str(annotation): entry["annotation_sha256"]})
                labels[name] = read_boxes(annotation)
            rt.tracker.z_dict1 = templates[name]
            frame = int(sample["frame_id"])
            image = image_reader(plan["root"], entry["frames"][frame])
            label = prior.PERTURBATIONS[int(perturbations[step - 1, column]) + 1]
            losses = []
            for view in range(2):
                parent = prior.perturb(sample["previous"], label) if arm == "augmented" and view == 1 else np.asarray(sample["previous"], float)
                rt.tracker.state, rt.tracker.frame_id = parent.tolist(), frame - 1
                search, actual_factor = rt.prepare(image)
                roi, factor = target_roi(labels[name][frame], parent, image.shape[:2], plan)
                np.testing.assert_allclose(actual_factor, factor, atol=1e-12, rtol=0)
                if view == 0:
                    np.testing.assert_allclose(roi, sample["target_roi"], atol=1e-7, rtol=0)
                output = forward_student(rt, net, search)
                calls += 1
                if step == 1:
                    with torch.no_grad():
                        original = rt.head(rt.prefix(search))
                        error = max(float((output[k] - original[k]).abs().max()) for k in ("score_map", "size_map", "offset_map"))
                        initial_error = max(initial_error, error)
                        if error > 1e-6:
                            raise ValueError("New student is not initialized from original Fast")
                target = output["pred_boxes"].new_tensor(roi)[None]
                loss, _ = tracking_loss(torch, output, target, True)
                losses.append(loss)
            total = .5 * (losses[0] + losses[1])
            (total / DESIGN["batch_size"]).backward()
            totals += np.array([float(total.detach()), float(losses[0].detach()), float(losses[1].detach())]) / DESIGN["batch_size"]
        norm = torch.nn.utils.clip_grad_norm_(list(parameters(net).values()), DESIGN["gradient_clip"])
        if not np.isfinite([*totals, float(norm)]).all():
            raise ValueError("Nonfinite matched crop loss/gradient")
        max_gradient = max(max_gradient, float(norm))
        optimizer.step()
        if step == 1 or step % DESIGN["save_every"] == 0 or step == updates:
            history.append(dict(step=step, loss=float(totals[0]), clean_loss=float(totals[1]), second_loss=float(totals[2]), gradient_norm=float(norm)))
            print("[fold{} seed{} {}] {}/{} loss={:.4f}".format(fold, seed, arm, step, updates, totals[0]), flush=True)
        if step % DESIGN["save_every"] == 0 or step == updates:
            if frozen_digest(net, "representation_standard") != frozen:
                raise ValueError("Frozen student parameters changed")
            save_step(torch, directory, dict(metadata=meta, step=step, parameters=trainable_state(net), optimizer=optimizer.state_dict(),
                history=history, head_calls=calls, maximum_gradient=max_gradient, initial_fast_max_error=initial_error))
    path = directory / "step{:06d}.pt".format(updates)
    changed = any(not torch.equal(v, trainable_state(net)[k]) for k, v in before.items())
    if (not changed or max_gradient <= 0 or calls != updates * DESIGN["batch_size"] * 2 or
            frozen_digest(net, "representation_standard") != frozen):
        raise ValueError("Training gradient/budget/frozen-state integrity failed")
    return dict(fold=fold, seed=seed, arm=arm, checkpoint=str(path.resolve()), checkpoint_sha256=file_hash(path),
        receipt_sha256=file_hash(path.with_suffix(".json")), updates=updates, initial_student_sha256=initial,
        draws_fingerprint=meta["draws_fingerprint"], perturbations_fingerprint=meta["perturbations_fingerprint"],
        initial_fast_max_error=initial_error, head_calls=calls, maximum_gradient=max_gradient, history=history,
        prefix_changed=changed, frozen_student_unchanged=True)


def train(rt, plan, root, seed, smoke=False, image_reader=checked_image):
    validate_runtime(rt, plan)
    if rt.identity != plan["source_runtime"] or seed not in SEEDS or (smoke and seed != SEEDS[0]):
        raise ValueError("Use original runtime and frozen seeds; smoke uses first seed")
    if not smoke:
        smoke_gate(root, plan)
    directory = Path(root) / ("smoke/training" if smoke else "training/seed{}".format(seed))
    teacher, records = state_digest(rt.net), []
    with writer_lock(directory):
        for fold in ([0] if smoke else range(DESIGN["folds"])):
            for arm in ARMS:
                records.append(train_arm(rt, plan, directory / "fold{}".format(fold) / arm, fold, seed, arm, smoke, image_reader))
        for fold in {r["fold"] for r in records}:
            selected = [r for r in records if r["fold"] == fold]
            for key in ("initial_student_sha256", "draws_fingerprint", "perturbations_fingerprint", "head_calls"):
                if len({r[key] for r in selected}) != 1:
                    raise ValueError("Arms do not match initialization, samples or forward budget")
        if state_digest(rt.net) != teacher:
            raise ValueError("Original teacher changed")
        report = seal(dict(schema=SCHEMA, status="PASS_MATCHED_CROP_TRAINING_INTEGRITY", plan_fingerprint=plan["fingerprint"],
            mode="smoke" if smoke else "full", seed=seed, teacher_sha256=teacher, runtime=rt.identity, arms=records,
            held_out_videos_used_for_fitting=False, original_checking_labels_loaded=False))
        verify_bindings(plan["source_file_sha256"])
        publish_identical(directory / "training_decision.json", report)
    return report


def restore_model(rt, plan, record, mode):
    path = Path(record["checkpoint"])
    verify_bindings({str(path): record["checkpoint_sha256"], str(path.with_suffix(".json")): record["receipt_sha256"]})
    value = load_checkpoint(rt.torch, path)
    meta = value["metadata"]
    fold = record["fold"]
    samples, draws, perturbations = schedule(plan, fold, record["seed"], record["updates"])
    if (meta.get("plan_fingerprint") != plan["fingerprint"] or meta.get("mode") != mode or
            meta.get("fold") != fold or meta.get("seed") != record["seed"] or meta.get("arm") != record["arm"] or
            meta.get("runtime") != rt.identity or meta.get("teacher_sha256") != state_digest(rt.net) or
            meta.get("fit_videos") != plan["folds"][fold]["fit"] or meta.get("held_out_videos") != plan["folds"][fold]["held_out"] or
            meta.get("draws_fingerprint") != fingerprint(draws.tolist()) or
            meta.get("perturbations_fingerprint") != fingerprint(perturbations.tolist()) or value["step"] != record["updates"]):
        raise ValueError("Wrong fold/seed/budget training checkpoint")
    net = make_student(rt)
    if state_digest(net) != meta["initial_student_sha256"]:
        raise ValueError("Changed Fast initialization")
    restore_state(net, value["parameters"])
    return net.eval().requires_grad_(False)


def evaluation_models(rt, plan, root):
    models, bindings = {}, {}
    for seed in SEEDS:
        path = Path(root) / "training/seed{}".format(seed) / "training_decision.json"
        report = read_sealed(path, SCHEMA)
        expected = {(f, a) for f in range(DESIGN["folds"]) for a in ARMS}
        if (report.get("status") != "PASS_MATCHED_CROP_TRAINING_INTEGRITY" or report.get("mode") != "full" or
                report.get("plan_fingerprint") != plan["fingerprint"] or report.get("seed") != seed or
                report.get("runtime") != rt.identity or report.get("teacher_sha256") != state_digest(rt.net) or
                len(report["arms"]) != len(expected) or {(r["fold"], r["arm"]) for r in report["arms"]} != expected):
            raise ValueError("Freeze all 16 matched models before OOF evaluation")
        bindings[str(path)] = file_hash(path)
        for record in report["arms"]:
            if record["updates"] != DESIGN["updates"] or record["seed"] != seed:
                raise ValueError("Wrong fixed training budget/seed")
            models[(record["fold"], record["arm"] + "_seed" + str(seed))] = restore_model(rt, plan, record, "full")
            bindings[record["checkpoint"]] = record["checkpoint_sha256"]
            bindings[str(Path(record["checkpoint"]).with_suffix(".json"))] = record["receipt_sha256"]
    return models, bindings


def collect_stress(rt, plan, entry, method, anchor, net, horizon, image_reader=checked_image):
    reference = baseline(plan, entry, "sgla")
    initial = image_reader(plan["root"], entry["frames"][0])
    images = [image_reader(plan["root"], entry["frames"][i]) for i in range(anchor, anchor + horizon)]
    branches, calls = [], []
    for name in prior.PERTURBATIONS:
        branch, used, _ = run_branch(rt, plan, entry, anchor, images, initial,
            prior.perturb(reference["previous"][anchor - 1], name), method, net)
        branches.append(branch)
        calls.append(used)
    data = {k: np.stack([b[k] for b in branches]) for k in branches[0]}
    annotation = Path(plan["root"]) / entry["annotation"]
    verify_bindings({str(annotation): entry["annotation_sha256"]})
    gt = read_boxes(annotation)[anchor:anchor + horizon]
    data.update(gt=gt, iou=np.stack([tracking_iou(b, gt) for b in data["boxes"]]), head_calls=np.asarray(calls),
        image_hw=np.asarray([i.shape[:2] for i in images]),
        window=cpu_array(rt.tracker.output_window).reshape(rt.search_side, rt.search_side).astype(np.float32))
    audit_stress(data, plan, entry, anchor, horizon, reference)
    return data


def smoke_gate(root, plan):
    path = Path(root) / "smoke/stage40_smoke.json"
    report = read_sealed(path, SCHEMA)
    if report.get("status") != "PASS_MATCHED_CROP_SMOKE" or report.get("plan_fingerprint") != plan["fingerprint"]:
        raise ValueError("Need unchanged matched crop smoke")
    verify_bindings(report["file_sha256"])
    return report


def smoke(rt, plan, root, image_reader=checked_image):
    report = train(rt, plan, root, SEEDS[0], True, image_reader)
    directory = Path(root) / "smoke"
    run = seal(dict(schema=SCHEMA, mode="smoke", plan_fingerprint=plan["fingerprint"], runtime=rt.identity,
        teacher_sha256=state_digest(rt.net)))
    entry = next(e for e in plan["fit"] if e["sequence"] == plan["folds"][0]["held_out"][0])
    anchor = plan["tasks"][entry["sequence"]][0]
    bindings = {str(directory / "training/training_decision.json"): file_hash(directory / "training/training_decision.json")}
    with writer_lock(directory):
        publish_identical(directory / "run.json", run)
        for record in report["arms"]:
            net = restore_model(rt, plan, record, "smoke")
            method = record["arm"] + "_seed" + str(SEEDS[0])
            path = result_path(root, entry, method, anchor, "smoke")
            if not path.exists():
                data = collect_stress(rt, plan, entry, method, anchor, net, prior.DESIGN["smoke_horizon"], image_reader)
                write_cache(path, data, dict(schema=SCHEMA, run_fingerprint=run["fingerprint"], sequence=entry["sequence"], method=method, anchor=anchor))
            load_result(path, plan, run, entry, method, anchor, allow_orphan=True)
            for bound in (path, path.with_suffix(".json"), Path(record["checkpoint"]), Path(record["checkpoint"]).with_suffix(".json")):
                bindings[str(bound)] = file_hash(bound)
        bindings[str(directory / "run.json")] = file_hash(directory / "run.json")
        result = seal(dict(schema=SCHEMA, status="PASS_MATCHED_CROP_SMOKE", plan_fingerprint=plan["fingerprint"],
            file_sha256=bindings, warning="Matched optimizer/crop/gradient/continuation checks only; no policy gain"))
        publish_identical(directory / "stage40_smoke.json", result)
    print("Stage-40: PASS_MATCHED_CROP_SMOKE")


def evaluate(rt, plan, root, sequence=None, image_reader=checked_image):
    validate_runtime(rt, plan)
    smoke_gate(root, plan)
    if rt.identity != plan["source_runtime"]:
        raise ValueError("Evaluation runtime differs from source")
    models, bindings = evaluation_models(rt, plan, root)
    digests = {"{}:{}".format(*k): state_digest(n) for k, n in sorted(models.items())}
    teacher = state_digest(rt.net)
    run = seal(dict(schema=SCHEMA, mode="evaluation", plan_fingerprint=plan["fingerprint"], runtime=rt.identity,
        teacher_sha256=teacher, methods=methods(), model_file_sha256=bindings, model_state_sha256=digests,
        timing_role="No deployment speed claim; diagnostic training/inference instrumentation"))
    entries = [e for e in plan["fit"] if sequence is None or e["sequence"] == sequence]
    if not entries:
        raise ValueError("Sequence must be in the original eight")
    with writer_lock(Path(root) / "evaluation"):
        publish_identical(Path(root) / "evaluation/run.json", run)
        for entry in entries:
            fold = fold_for(plan, entry)
            for method in methods():
                net = models[(fold, method)]
                for anchor in [None] + plan["tasks"][entry["sequence"]]:
                    path = result_path(root, entry, method, anchor)
                    if not path.exists():
                        if path.with_suffix(".json").exists():
                            raise ValueError("Committed evaluation file missing")
                        data = collect_trace(rt, plan, entry, method, net, image_reader) if anchor is None else collect_stress(
                            rt, plan, entry, method, anchor, net, prior.DESIGN["horizon"], image_reader)
                        write_cache(path, data, dict(schema=SCHEMA, run_fingerprint=run["fingerprint"], sequence=entry["sequence"], method=method, anchor=anchor))
                    load_result(path, plan, run, entry, method, anchor, allow_orphan=True)
                    print("[audited] " + path.name, flush=True)
        if state_digest(rt.net) != teacher or any(state_digest(net) != digests["{}:{}".format(*k)] for k, net in models.items()):
            raise ValueError("Frozen evaluation model changed")
        verify_bindings(bindings)
        verify_bindings(plan["source_file_sha256"])


def status(root, plan):
    for seed in SEEDS:
        for fold in range(DESIGN["folds"]):
            for arm in ARMS:
                directory = Path(root) / "training/seed{}".format(seed) / "fold{}".format(fold) / arm
                step = 0
                for path in sorted(directory.glob("step*.json")):
                    receipt = read_sealed(path, SCHEMA)
                    verify_bindings({str(path.with_suffix(".pt")): receipt["checkpoint_sha256"]})
                    if receipt["metadata"]["plan_fingerprint"] != plan["fingerprint"]:
                        raise ValueError("Mixed progress metadata")
                    step = receipt["step"]
                print("seed{} fold{} {:<10s} {}/{} committed updates".format(seed, fold, arm, step, DESIGN["updates"]))
    path = Path(root) / "evaluation/run.json"
    if path.exists():
        run = read_sealed(path, SCHEMA)
        if run["plan_fingerprint"] != plan["fingerprint"]:
            raise ValueError("Mixed evaluation progress")
        verify_bindings(run["model_file_sha256"])
        totals = [0, 0]
        for entry in plan["fit"]:
            for method in methods():
                for anchor in [None] + plan["tasks"][entry["sequence"]]:
                    result = result_path(root, entry, method, anchor)
                    if result.with_suffix(".json").exists():
                        load_result(result, plan, run, entry, method, anchor)
                        totals[int(anchor is not None)] += 1
        print("Evaluation: {}/32 full traces; {}/96 stress packs".format(*totals))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=nonempty_path, required=True, help="Stage-39 root for prepare; Stage-40 otherwise")
    parser.add_argument("--mode", required=True, choices=("prepare", "smoke", "train", "evaluate", "status"))
    parser.add_argument("--output_dir", type=nonempty_path)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument("--checkpoint", type=nonempty_path)
    parser.add_argument("--parameter", default="deit_distilled")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sequence")
    args = parser.parse_args()
    if args.sequence is not None and args.mode != "evaluate":
        parser.error("--sequence is only for evaluate; training always uses the frozen fold's complete fit sample set")
    root = Path(args.input).expanduser().resolve()
    if not (root / "manifest.json").is_file():
        parser.error("Input manifest missing: {}. Reset STAGE39_ROOT / STAGE40_ROOT from the repository root.".format(root / "manifest.json"))
    if args.mode == "prepare":
        if not args.output_dir:
            parser.error("prepare requires --output_dir")
        with writer_lock(Path(args.output_dir).expanduser().resolve()):
            prepare(root, args.output_dir)
        return
    plan = load_plan(root)
    if args.mode == "status":
        status(root, plan)
        return
    checkpoint = args.checkpoint or os.environ.get("CHECKPOINT") or None
    if checkpoint and file_hash(Path(checkpoint).expanduser()) != plan["runtime"]["checkpoint_sha256"]:
        parser.error("Use the original SGLA checkpoint")
    import torch
    from stage4_runtime import Runtime
    seed_runtime(torch, plan["runtime_seed"])
    rt = Runtime(args.parameter, args.gpu, checkpoint)
    if args.mode == "smoke":
        smoke(rt, plan, root)
    elif args.mode == "train":
        train(rt, plan, root, args.seed)
    else:
        evaluate(rt, plan, root, args.sequence)


if __name__ == "__main__":
    main()
