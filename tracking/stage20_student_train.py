"""Matched 2x2 dense-student training, followed by autonomous development rollouts."""

import argparse
import copy
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from stage4_common import (file_hash, fingerprint, frame_path, load_annotations,
                           load_manifest, tracking_iou, write_cache, write_csv, write_json)
from stage4_runtime import Runtime, add_runtime_arguments
from stage18_environment import executed_crop_roi
from stage18_tracking_loss import tracking_loss
from stage20_compression_common import (ARMS, METHODS, SCHEMA, accept_teacher, audit_trace,
                                        channel_plan, compact_copy, crop_parent,
                                        distillation_loss, forward_model, nonempty_path,
                                        state_digest)


def atomic_save(torch, value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
        temporary = f.name
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.isfile(temporary):
            os.unlink(temporary)


def load_checkpoint(torch, path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_sources(manifest, entries, maximum):
    sources = []
    for entry in entries:
        directory, gt, valid = load_annotations(manifest["root"], manifest["dataset"], entry["sequence"])
        if (file_hash(directory / "groundtruth.txt") != entry["annotation_sha256"]
                or fingerprint(valid.tolist()) != entry["visible_sha256"]):
            raise ValueError("Annotations/visibility changed: " + entry["sequence"])
        gt = np.where(np.isfinite(gt), gt, np.nan)
        start, stop = entry["start"], min(entry["stop"], entry["start"] + maximum + 1)
        paths = [frame_path(directory, manifest["dataset"], i) for i in range(start, stop)]
        if (stop > len(gt) or len(paths) < 3 or not valid[start]
                or not valid[start + 1:stop].any() or any(not path.is_file() for path in paths)):
            raise ValueError("Missing frames or unusable clip: " + entry["sequence"])
        sources.append(dict(sequence=entry["sequence"], start=start, stop=stop, paths=paths,
                            gt=gt[start:stop].copy(), valid=valid[start:stop].copy()))
    return sources


def source_identity(source):
    return dict(sequence=source["sequence"], start=source["start"], stop=source["stop"],
                image_sha256=[file_hash(path) for path in source["paths"]],
                gt=[[float(v) if np.isfinite(v) else None for v in row] for row in source["gt"]],
                valid=source["valid"].tolist())


def decode_at(rt, output, factor, image, previous):
    rt.tracker.state = list(previous)
    with rt.torch.no_grad():
        return np.asarray(rt.decode(output, factor, image), dtype=np.float64)


def initialize(rt, source, images):
    with rt.torch.no_grad():
        rt.initialize(images[0], source["gt"][0])


def train_clip(rt, student, source, arm, optimizer, accumulation):
    torch = rt.torch
    images = [rt.read(path) for path in source["paths"]]
    initialize(rt, source, images)
    reference_previous = student_previous = source["gt"][0].copy()
    totals = dict(frames=0, valid_frames=0, updates=0, teacher_accepted=0,
                  supervised_loss_sum=0., kd_loss_sum=0., loss_sum=0., crop_shift_sum=0.)
    chunk, valid_chunk = [], 0
    student.eval()
    optimizer.zero_grad(set_to_none=True)
    for index, image in enumerate(images[1:], start=1):
        # All four arms share the same reference process and teacher-query count.
        with torch.no_grad():
            rt.tracker.state = reference_previous.tolist()
            reference_search, reference_factor = rt.prepare(image)
            reference_output, _ = forward_model(rt, rt.net, reference_search)
            reference_next = decode_at(rt, reference_output, reference_factor, image, reference_previous)
        previous = crop_parent(arm, reference_previous, student_previous)
        rt.tracker.state = previous.tolist()
        search, factor = rt.prepare(image)
        with torch.no_grad():
            teacher_output, _ = forward_model(rt, rt.net, search)
            teacher_box = decode_at(rt, teacher_output, factor, image, previous)
        output, _ = forward_model(rt, student, search)
        student_next = decode_at(rt, output, factor, image, previous)
        valid = bool(source["valid"][index])
        roi = np.zeros(4, dtype=np.float32)
        if valid:
            roi = executed_crop_roi(source["gt"][index], previous, factor, rt.tracker.params.search_size)
        target = output["pred_boxes"].new_tensor(roi)[None]
        supervised, _ = tracking_loss(torch, output, target, valid=valid)
        accepted = accept_teacher(roi, teacher_box, source["gt"][index], valid)
        kd = supervised * 0
        if arm.startswith("kd_") and accepted:
            kd = distillation_loss(torch, output, teacher_output)
        loss = supervised + kd
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss: {}/{}".format(arm, source["sequence"]))
        chunk.append(loss)
        valid_chunk += int(valid)
        totals["frames"] += 1
        totals["valid_frames"] += int(valid)
        totals["teacher_accepted"] += int(accepted)
        totals["supervised_loss_sum"] += float(supervised.detach())
        totals["kd_loss_sum"] += float(kd.detach())
        totals["loss_sum"] += float(loss.detach())
        totals["crop_shift_sum"] += float(np.abs(previous - reference_previous).mean())
        student_previous, reference_previous = student_next, reference_next
        if index % accumulation == 0 or index == len(images) - 1:
            if valid_chunk:
                (torch.stack(chunk).sum() / valid_chunk).backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad], 1.0)
                if not torch.isfinite(norm):
                    raise RuntimeError("Nonfinite student gradient")
                optimizer.step()
                totals["updates"] += 1
            optimizer.zero_grad(set_to_none=True)
            chunk, valid_chunk = [], 0
    return totals


def fit_arm(rt, initial, sources, arm, seed, directory, contract):
    torch = rt.torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = copy.deepcopy(initial)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=contract["training"]["learning_rate"], weight_decay=.01)
    path = directory / "last.pt"
    history = []
    if path.is_file():
        saved = load_checkpoint(torch, path)
        if (saved.get("contract_fingerprint") != contract["fingerprint"]
                or saved.get("arm") != arm or saved.get("seed") != seed):
            raise ValueError("Checkpoint belongs to another experiment: " + str(path))
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        history = saved["history"]
        if len(history) != saved["epoch"]:
            raise ValueError("Corrupt epoch history")
        if saved["initial_state_sha256"] != contract["initial_state_sha256"]:
            raise ValueError("Student initialization changed")
    for epoch in range(len(history) + 1, contract["training"]["epochs"] + 1):
        torch.manual_seed(seed + epoch)
        torch.cuda.manual_seed_all(seed + epoch)
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(sources))
        total, started = {}, time.perf_counter()
        for number, index in enumerate(order, start=1):
            result = train_clip(rt, model, sources[index], arm, optimizer,
                                contract["training"]["accumulation_frames"])
            for key, value in result.items():
                total[key] = total.get(key, 0) + value
            if number % 8 == 0 or number == len(order):
                print("[train] seed={} {} epoch={}/{} clip={}/{} loss={:.4f}".format(
                    seed, arm, epoch, contract["training"]["epochs"], number, len(order),
                    total["loss_sum"] / max(1, total["valid_frames"])), flush=True)
        row = dict(epoch=epoch, **total, seconds=time.perf_counter() - started)
        history.append(row)
        directory.mkdir(parents=True, exist_ok=True)
        atomic_save(torch, dict(schema=SCHEMA, contract_fingerprint=contract["fingerprint"],
                                arm=arm, seed=seed, epoch=epoch, model=model.state_dict(),
                                optimizer=optimizer.state_dict(), history=history,
                                initial_state_sha256=contract["initial_state_sha256"]), path)
        write_csv(directory / "history.csv", history)
    if state_digest(model) == contract["initial_state_sha256"]:
        raise RuntimeError("Training did not change the compact student")
    original_selector = initial.backbone.MLP.state_dict()
    if any(not torch.equal(value, original_selector[name])
           for name, value in model.backbone.MLP.state_dict().items()):
        raise RuntimeError("Frozen selector weights changed")
    return model, dict(seed=seed, method=arm, path=str(path.resolve()), sha256=file_hash(path),
                       epoch=len(history), frames=sum(r["frames"] for r in history),
                       valid_frames=sum(r["valid_frames"] for r in history),
                       updates=sum(r["updates"] for r in history),
                       teacher_accepted=sum(r["teacher_accepted"] for r in history),
                       kd_loss_sum=sum(r["kd_loss_sum"] for r in history),
                       final_state_sha256=state_digest(model))


def evaluate(rt, model, source, images, verify=False):
    torch = rt.torch
    initialize(rt, source, images)
    previous = source["gt"][0].copy()
    boxes, parents, layers, elapsed = [], [], [], []
    model.eval()
    with torch.no_grad():
        for index, image in enumerate(images[1:]):
            rt.tracker.state = previous.tolist()
            torch.cuda.synchronize()
            started = time.perf_counter()
            search, factor = rt.prepare(image)
            output, layer = forward_model(rt, model, search)
            box = decode_at(rt, output, factor, image, previous)
            torch.cuda.synchronize()
            elapsed.append((time.perf_counter() - started) * 1000)
            if verify and index == 0:
                rt.verify(search, output)
            parents.append(previous.copy())
            boxes.append(box.copy())
            layers.append(layer)
            previous = box
    data = dict(frame_id=np.arange(source["start"] + 1, source["stop"]),
                previous=np.asarray(parents), boxes=np.asarray(boxes), gt=source["gt"][1:],
                valid=source["valid"][1:], layer=np.asarray(layers, dtype=np.int64),
                latency_ms=np.asarray(elapsed))
    data["iou"] = tracking_iou(data["boxes"], data["gt"], data["valid"])
    audit_trace(data, source["gt"][0], data["frame_id"])
    return data


def evaluate_all(rt, models, sources, seed, output, contract, checkpoints):
    torch = rt.torch
    hashes = {item["method"]: item["sha256"] for item in checkpoints}
    hashes.update(teacher=contract["runtime"]["checkpoint_sha256"],
                  compact_initial=contract["initial_state_sha256"])
    # Untimed warm-up includes every architecture before interleaved runs.
    first = sources[0]
    image0, image1 = rt.read(first["paths"][0]), rt.read(first["paths"][1])
    with torch.no_grad():
        rt.initialize(image0, first["gt"][0])
        search, _ = rt.prepare(image1)
        for model in models.values():
            for _ in range(10):
                forward_model(rt, model, search)
    for index, source in enumerate(sources):
        images = [rt.read(path) for path in source["paths"]]
        order = np.random.default_rng(np.random.SeedSequence([seed, index, 20])).permutation(len(METHODS))
        for method in (METHODS[i] for i in order):
            path = output / "evaluation" / ("seed{}".format(seed)) / method / (source["sequence"] + ".npz")
            metadata = dict(contract_fingerprint=contract["fingerprint"], seed=seed, method=method,
                            sequence=source["sequence"], checkpoint_sha256=hashes[method])
            if path.is_file():
                with np.load(path, allow_pickle=False) as archive:
                    existing = json.loads(str(archive["metadata_json"].item()))
                    data = {k: archive[k] for k in archive.files if k != "metadata_json"}
                if existing != metadata:
                    raise ValueError("Evaluation identity mismatch: " + str(path))
                audit_trace(data, source["gt"][0], np.arange(source["start"] + 1, source["stop"]))
            else:
                data = evaluate(rt, models[method], source, images,
                                verify=method == "teacher" and index == 0)
                write_cache(path, data, metadata)
        print("[evaluate] seed={} {}/{} {}".format(seed, index + 1, len(sources),
                                                   source["sequence"]), flush=True)


def make_contract(rt, manifest, split, sources, initial, audits, plan, args):
    teacher_count = sum(p.numel() for p in rt.net.parameters())
    student_count = sum(p.numel() for p in initial.parameters())
    files = ("stage20_compression_common.py", "stage20_student_train.py", "stage20_compression_summary.py",
             "stage19_ffn_compression.py", "stage18_tracking_loss.py", "stage18_environment.py",
             "stage4_runtime.py", "stage1_validate.py", "stage4_common.py", "validation_metrics.py",
             "../lib/models/sglatrack/sglatrack.py", "../lib/models/sglatrack/vit.py",
             "../lib/models/sglatrack/base_backbone.py", "../lib/models/sglatrack/deit.py",
             "../lib/models/sglatrack/utils.py", "../lib/models/layers/patch_embed.py",
             "../lib/train/data/processing_utils.py", "../lib/test/tracker/data_utils.py",
             "../lib/test/tracker/sglatrack.py", "../lib/models/layers/head.py", "../lib/utils/box_ops.py",
             "../lib/utils/heapmap_utils.py", "../lib/utils/focal_loss.py")
    contract = dict(schema=SCHEMA, mode=args.mode, runtime=rt.identity,
                    source_manifest_fingerprint=manifest["fingerprint"], split_seed=args.split_seed,
                    excluded_source_val=split["excluded_source_val"],
                    sources={role: [source_identity(s) for s in items] for role, items in sources.items()},
                    architecture=dict(rule="all_12_ffns_half_hidden_width_checkpoint_magnitude_product",
                                      selector="original_rule_frozen_weights", plan=plan,
                                      teacher_parameters=teacher_count, student_parameters=student_count,
                                      parameter_reduction=1 - student_count / teacher_count,
                                      surgery_audits=audits),
                    initial_state_sha256=state_digest(initial), teacher_state_sha256=state_digest(rt.net),
                    arms=list(ARMS), seeds=args.seeds, primary="kd_student",
                    training=dict(epochs=args.epochs, learning_rate=args.learning_rate,
                                  accumulation_frames=args.accumulation_frames,
                                  maximum_tracked_frames=args.frames, optimizer="AdamW_wd0.01_clip1",
                                  checkpoint_selection="fixed_final_epoch_no_check_selection",
                                  batchnorm_dropout="eval_mode_all_arms", kd_weight=1.,
                                  kd_loss="spatial_logit_KL_plus_5_normalized_box_L1",
                                  kd_acceptance="visible_GT_center_in_executed_crop_and_teacher_image_IoU_ge0.5",
                                  teacher_queries="reference_and_same_executed_tensor_all_arms",
                                  state_gradient="decoded_crop_history_detached_no_BPTT"),
                    analysis=dict(primary_contrasts=["kd_student_minus_kd_reference",
                                                     "kd_student_minus_supervised_student"],
                                  required_interaction="(KDself-Sself)-(KDref-Sref)",
                                  paired_unit="video_averaging_all_declared_seeds_first",
                                  auc_protocol="sgla_int_prediction_inclusive_v1_without_initialization",
                                  bootstrap_samples=5000, seed=2060, latency_warmup_frames=5,
                                  teacher_auc_noninferiority_margin=.5,
                                  target_parameter_reduction=.20, target_latency_reduction=.10),
                    numerics=dict(precision="float32_no_AMP", deterministic_algorithms=True,
                                  allow_tf32=False, cublas_workspace=os.environ["CUBLAS_WORKSPACE_CONFIG"]),
                    role="development_feasibility_not_external_confirmation",
                    code_sha256={name: file_hash(Path(__file__).parent / name) for name in files})
    contract["fingerprint"] = fingerprint(contract)
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_manifest", required=True, type=nonempty_path)
    parser.add_argument("--output_dir", required=True, type=nonempty_path)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2059, 2060, 2061])
    parser.add_argument("--split_seed", type=int, default=2059)
    parser.add_argument("--check_sequences", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--accumulation_frames", type=int, default=4)
    add_runtime_arguments(parser)
    args = parser.parse_args()
    if args.checkpoint is not None and not args.checkpoint.strip():
        parser.error("Empty checkpoint path; export CHECKPOINT first")
    if args.mode == "smoke":
        args.epochs, args.frames, args.seeds = 1, 16, args.seeds[:1]
    if (args.epochs < 1 or args.frames < 8 or args.accumulation_frames < 1 or args.gpu < 0
            or not np.isfinite(args.learning_rate) or args.learning_rate <= 0
            or len(args.seeds) != len(set(args.seeds)) or min(args.seeds + [args.split_seed]) < 0):
        parser.error("Invalid training settings")
    from stage20_compression_common import design_split
    manifest = load_manifest(Path(args.source_manifest).expanduser().resolve())
    split = design_split(manifest, args.mode, args.split_seed, args.check_sequences)
    sources = {role: load_sources(manifest, split[role], args.frames) for role in ("fit", "check")}
    print("Stage-20 {}: {} fit / {} development-check videos; {} source-val videos excluded".format(
        args.mode, len(sources["fit"]), len(sources["check"]), len(split["excluded_source_val"])), flush=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    rt = Runtime(args.parameter, args.gpu, args.checkpoint)
    torch = rt.torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    plan = channel_plan([(b.mlp.fc1.weight.detach().cpu().numpy(),
                          b.mlp.fc2.weight.detach().cpu().numpy()) for b in rt.backbone.blocks])
    initial, audits = compact_copy(torch, rt.net, plan)
    contract = make_contract(rt, manifest, split, sources, initial, audits, plan, args)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "stage20_contract.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != contract:
            raise ValueError("Frozen contract differs; use a new output directory")
    elif any(output.iterdir()):
        raise ValueError("Unidentified files exist; use an empty output directory")
    else:
        write_json(path, contract)
    lock = output / "stage20_training.lock"
    try:
        descriptor = os.open(str(lock), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        raise RuntimeError("Another run owns stage20_training.lock; after a killed process, verify it stopped before removing this lock")
    os.close(descriptor)
    try:
        artifacts = []
        for seed in args.seeds:
            models = dict(teacher=rt.net, compact_initial=initial)
            for arm in ARMS:
                model, artifact = fit_arm(rt, initial, sources["fit"], arm, seed,
                                          output / "checkpoints" / ("seed{}".format(seed)) / arm, contract)
                models[arm] = model
                artifacts.append(artifact)
            if state_digest(rt.net) != contract["teacher_state_sha256"]:
                raise RuntimeError("Frozen original teacher changed")
            evaluate_all(rt, models, sources["check"], seed, output, contract,
                         [a for a in artifacts if a["seed"] == seed])
            write_json(output / "stage20_artifacts.json", dict(contract_fingerprint=contract["fingerprint"],
                                                               teacher_unchanged=True, checkpoints=artifacts))
            del models
        print("Training and autonomous rollouts complete. Next: python tracking/stage20_compression_summary.py --input \"{}\"".format(output))
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
