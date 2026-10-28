"""Adapt retained SGLA under the frozen six-block control's exact data schedule."""

import argparse
import os
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_cache
from stage18_tracking_loss import tracking_loss
from stage20_compression_common import state_digest
from stage20_student_train import atomic_save, load_checkpoint
from stage27_uav_collect import writer_lock
from stage27_uav_common import checked_image, nonempty_path, publish_identical, read_boxes, seal, tracking_iou
from stage28_response_collect import validate_runtime
from stage28_response_common import cpu_array, dense_boxes
from stage34_identity_experiment import seed_runtime
from stage35_preserve_runtime import make_student
from stage37_crossed_common import fold_for
from stage44_matched_common import (DESIGN, SCHEMA, SEEDS, audit_trace, checkpoint_meta, load_plan,
    load_trace, original_plan, prepare, read_sealed, reused_trace, source, trace_meta, trace_path, verify_bindings)
from stage44_matched_runtime import (forward_retained, frozen_digest, make_retained, parameters,
    restore_state, shared_prefix_identity, trainable_state)


def save_step(torch,directory,value):
    path = Path(directory) / "step{:06d}.pt".format(value["step"])
    if path.with_suffix(".json").exists():
        raise ValueError("Never overwrite a committed Stage-44 checkpoint")
    if path.exists() and load_checkpoint(torch,path).get("metadata") != value["metadata"]:
        raise ValueError("Mixed orphan checkpoint")
    atomic_save(torch,value,path)
    publish_identical(path.with_suffix(".json"),seal(dict(schema=SCHEMA,metadata=value["metadata"],step=value["step"],checkpoint_sha256=file_hash(path))))
    return path


def restore_latest(torch,directory,meta):
    latest = None
    for path in sorted(Path(directory).glob("step*.json")):
        receipt = read_sealed(path,SCHEMA)
        if (receipt.get("metadata") != meta or not 0 < receipt["step"] <= meta["updates"] or
                path.stem != "step{:06d}".format(receipt["step"]) or file_hash(path.with_suffix(".pt")) != receipt["checkpoint_sha256"]):
            raise ValueError("Completed Stage-44 checkpoint changed")
        latest = load_checkpoint(torch,path.with_suffix(".pt"))
        if latest.get("metadata") != meta or latest.get("step") != receipt["step"]:
            raise ValueError("Wrong Stage-44 optimizer checkpoint")
    return latest


def matched_record(plan,fold,seed):
    path = Path(plan["source_stage40"]) / "training/seed{}".format(seed) / "training_decision.json"
    report = read_sealed(path,source.SCHEMA)
    if report["plan_fingerprint"] != plan["source_model_plan_fingerprint"] or report.get("seed") != seed:
        raise ValueError("Wrong six-block training source")
    rows = [r for r in report["arms"] if r["fold"] == fold and r["arm"] == "clean"]
    if len(rows) != 1:
        raise ValueError("Missing matched clean training record")
    record = rows[0]
    verify_bindings({record["checkpoint"]:record["checkpoint_sha256"],
        str(Path(record["checkpoint"]).with_suffix(".json")):record["receipt_sha256"]})
    return record


def train_arm(rt,plan,directory,fold,seed,smoke=False,image_reader=checked_image):
    torch = rt.torch
    seed_runtime(torch,seed + 100 * fold)
    net = make_retained(rt)
    shared = shared_prefix_identity(rt,net,make_student(rt))
    initial, frozen = state_digest(net),frozen_digest(net)
    before = trainable_state(net)
    meta = checkpoint_meta(plan,rt,fold,seed,smoke,initial)
    updates = meta["updates"]
    samples,draws,_ = source.schedule(plan,fold,seed,updates)
    optimizer = torch.optim.AdamW(list(parameters(net).values()),lr=source.DESIGN["learning_rate"],weight_decay=source.DESIGN["weight_decay"])
    state = restore_latest(torch,directory,meta)
    start = state["step"] if state else 0
    history = state["history"] if state else []
    heads = [state["head_calls"] if state else 0]
    blocks = [state["block_calls"] if state else 0]
    parity = state["initial_native_max_error"] if state else 0.
    max_gradient = state["maximum_gradient"] if state else 0.
    distribution = state["layer_counts"] if state else [0] * 6
    if state:
        restore_state(net,state["parameters"])
        optimizer.load_state_dict(state["optimizer"])
    entries = {e["sequence"]:e for e in plan["fit"] if e["sequence"] in meta["fit_videos"]}
    templates,labels = {},{}
    hooks = [net.box_head.register_forward_hook(lambda *unused:heads.__setitem__(0,heads[0]+1))]
    hooks += [b.register_forward_hook(lambda *unused:blocks.__setitem__(0,blocks[0]+1)) for b in net.backbone.blocks]
    try:
        for step in range(start+1,updates+1):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for sample_index in draws[step-1]:
                sample = samples[int(sample_index)]
                entry = entries[sample["sequence"]]  # A held-out image cannot enter this mapping.
                name,frame = entry["sequence"],int(sample["frame_id"])
                if name not in templates:
                    with torch.no_grad():
                        rt.initialize(image_reader(plan["root"],entry["frames"][0]),entry["initial_box"])
                    templates[name] = rt.tracker.z_dict1
                    annotation = Path(plan["root"]) / entry["annotation"]
                    verify_bindings({str(annotation):entry["annotation_sha256"]})
                    labels[name] = read_boxes(annotation)
                rt.tracker.z_dict1 = templates[name]
                image = image_reader(plan["root"],entry["frames"][frame])
                views = []
                for _ in range(2):
                    parent = np.asarray(sample["previous"],float)
                    rt.tracker.state,rt.tracker.frame_id = parent.tolist(),frame-1
                    search,factor = rt.prepare(image)
                    roi,expected_factor = source.target_roi(labels[name][frame],parent,image.shape[:2],plan)
                    np.testing.assert_allclose(factor,expected_factor,atol=1e-12,rtol=0)
                    np.testing.assert_allclose(roi,sample["target_roi"],atol=1e-7,rtol=0)
                    h,b = heads[0],blocks[0]
                    output,layer = forward_retained(rt,net,search)
                    if heads[0]-h != 1 or blocks[0]-b != 7:
                        raise ValueError("Retained training must execute six prefix blocks plus one candidate and one head")
                    distribution[layer-7] += 1
                    if step == 1:
                        with torch.no_grad():
                            tokens = rt.prefix(search)
                            native_layer = rt.selected_layer(rt.selector(tokens))
                            original = rt.candidate(tokens,native_layer)
                            parity = max(parity,max(float((output[k]-original[k]).abs().max()) for k in ("score_map","size_map","offset_map")))
                            if layer != native_layer or parity > 1e-6:
                                raise ValueError("Retained initialization does not reproduce native SGLA")
                    target = output["pred_boxes"].new_tensor(roi)[None]
                    loss,_ = tracking_loss(torch,output,target,True)
                    views.append(loss)
                total = .5 * (views[0]+views[1])
                (total / source.DESIGN["batch_size"]).backward()
                losses.append(float(total.detach()))
            norm = torch.nn.utils.clip_grad_norm_(list(parameters(net).values()),source.DESIGN["gradient_clip"])
            if not np.isfinite([*losses,float(norm)]).all():
                raise ValueError("Nonfinite matched retained loss/gradient")
            max_gradient = max(max_gradient,float(norm))
            optimizer.step()
            if step == 1 or step % source.DESIGN["save_every"] == 0 or step == updates:
                history.append(dict(step=step,loss=float(np.mean(losses)),gradient_norm=float(norm)))
                print("[retained fold{} seed{}] {}/{} loss={:.4f}".format(fold,seed,step,updates,np.mean(losses)),flush=True)
            if step % source.DESIGN["save_every"] == 0 or step == updates:
                if frozen_digest(net) != frozen:
                    raise ValueError("Frozen candidate/selector/head/other prefix changed")
                save_step(torch,directory,dict(metadata=meta,step=step,parameters=trainable_state(net),optimizer=optimizer.state_dict(),
                    history=history,head_calls=heads[0],block_calls=blocks[0],layer_counts=distribution,
                    maximum_gradient=max_gradient,initial_native_max_error=parity))
    finally:
        for hook in hooks:
            hook.remove()
    expected = updates * source.DESIGN["batch_size"] * 2
    changed = any(not torch.equal(v,trainable_state(net)[k]) for k,v in before.items())
    if not changed or max_gradient <= 0 or heads[0] != expected or blocks[0] != 7*expected or frozen_digest(net) != frozen:
        raise ValueError("Matched retained training integrity failed")
    matched = matched_record(plan,fold,seed)
    if not smoke and (matched["updates"] != updates or matched["head_calls"] != heads[0] or matched["draws_fingerprint"] != meta["draws_fingerprint"]):
        raise ValueError("Retained and reused six-block training budgets/draws differ")
    if matched["initial_student_sha256"] != state_digest(make_student(rt)):
        raise ValueError("Reused six-block initialization differs from current runtime")
    path = Path(directory) / "step{:06d}.pt".format(updates)
    return dict(fold=fold,seed=seed,updates=updates,checkpoint=str(path.resolve()),checkpoint_sha256=file_hash(path),
        receipt_sha256=file_hash(path.with_suffix(".json")),initial_sha256=initial,shared_initial=shared,
        draws_fingerprint=meta["draws_fingerprint"],head_calls=heads[0],block_calls=blocks[0],layer_counts=distribution,
        frozen_sha256=frozen,maximum_gradient=max_gradient,initial_native_max_error=parity,
        matched_six_checkpoint=matched["checkpoint"],matched_six_sha256=matched["checkpoint_sha256"],history=history)


def restore_model(rt,plan,record,smoke=False):
    path = Path(record["checkpoint"])
    verify_bindings({str(path):record["checkpoint_sha256"],str(path.with_suffix(".json")):record["receipt_sha256"]})
    net = make_retained(rt)
    value = load_checkpoint(rt.torch,path)
    meta = checkpoint_meta(plan,rt,record["fold"],record["seed"],smoke,state_digest(net))
    receipt = read_sealed(path.with_suffix(".json"),SCHEMA)
    if (value.get("metadata") != meta or value.get("step") != meta["updates"] or record["updates"] != meta["updates"] or
            receipt.get("metadata") != meta or receipt.get("step") != meta["updates"] or
            record["draws_fingerprint"] != meta["draws_fingerprint"] or record["initial_sha256"] != state_digest(net)):
        raise ValueError("Wrong retained fold/seed/budget checkpoint")
    restore_state(net,value["parameters"])
    if frozen_digest(net) != record["frozen_sha256"]:
        raise ValueError("Frozen retained state changed")
    return net.eval().requires_grad_(False)


def training_report(root,plan,seed,smoke=False):
    directory = Path(root) / ("smoke/training" if smoke else "training/seed{}".format(seed))
    report = read_sealed(directory / "stage44_training.json",SCHEMA)
    expected = [0] if smoke else list(range(source.DESIGN["folds"]))
    if (report.get("status") != "PASS_MATCHED_RETAINED_TRAINING_INTEGRITY" or report.get("plan_fingerprint") != plan["fingerprint"] or
            report.get("seed") != seed or report.get("mode") != ("smoke" if smoke else "full") or
            sorted(r["fold"] for r in report["models"]) != expected or len(report["models"]) != len(expected) or
            report.get("runtime") != plan["source_runtime"]):
        raise ValueError("Need all matched retained models before evaluation")
    return report,directory / "stage44_training.json"


def train(rt,plan,root,seed,smoke=False,image_reader=checked_image):
    validate_runtime(rt,plan)
    if state_digest(rt.net) != plan["source_teacher_sha256"]:
        raise ValueError("Original pretrained SGLA state differs from frozen source")
    if seed not in SEEDS or (smoke and seed != SEEDS[0]):
        raise ValueError("Use the two fixed source seeds; smoke uses first seed")
    if not smoke:
        smoke_gate(root,plan)
    directory = Path(root) / ("smoke/training" if smoke else "training/seed{}".format(seed))
    teacher = state_digest(rt.net)
    with writer_lock(directory):
        records = [train_arm(rt,plan,directory / "fold{}".format(f),f,seed,smoke,image_reader)
            for f in ([0] if smoke else range(source.DESIGN["folds"]))]
        if state_digest(rt.net) != teacher:
            raise ValueError("Original SGLA model changed")
        verify_bindings(plan["source_file_sha256"])
        report = seal(dict(schema=SCHEMA,status="PASS_MATCHED_RETAINED_TRAINING_INTEGRITY",plan_fingerprint=plan["fingerprint"],
            mode="smoke" if smoke else "full",seed=seed,runtime=rt.identity,teacher_sha256=teacher,models=records,
            held_out_videos_used_for_fitting=False,original_checking_labels_loaded=False))
        publish_identical(directory / "stage44_training.json",report)
    return report


def collect_trace(rt,plan,entry,net,limit=None,image_reader=checked_image):
    n = min(len(entry["frames"])-1,limit) if limit else len(entry["frames"])-1
    data = dict(previous=np.empty((n,4)),boxes=np.empty((n,4)),layer=np.empty(n,np.int64),
        head_calls=np.empty(n,np.int64),block_calls=np.empty(n,np.int64),decode_error=np.empty(n))
    heads,blocks = [0],[0]
    hooks = [net.box_head.register_forward_hook(lambda *unused:heads.__setitem__(0,heads[0]+1))]
    hooks += [b.register_forward_hook(lambda *unused:blocks.__setitem__(0,blocks[0]+1)) for b in net.backbone.blocks]
    try:
        with rt.torch.no_grad():
            rt.initialize(image_reader(plan["root"],entry["frames"][0]),entry["initial_box"])
            previous = np.asarray(entry["initial_box"],float)
            for i in range(n):
                image = image_reader(plan["root"],entry["frames"][i+1])
                rt.tracker.state,rt.tracker.frame_id = previous.tolist(),i
                search,factor = rt.prepare(image)
                h,b = heads[0],blocks[0]
                output,layer = forward_retained(rt,net,search)
                box = np.asarray(rt.decode(output,factor,image),float)
                np.testing.assert_array_equal(rt.tracker.state,previous)
                data["previous"][i],data["boxes"][i],data["layer"][i] = previous,box,layer
                data["head_calls"][i],data["block_calls"][i] = heads[0]-h,blocks[0]-b
                side = rt.search_side
                dense = dense_boxes(cpu_array(output["size_map"]).reshape(2,side,side),cpu_array(output["offset_map"]).reshape(2,side,side),
                    previous,factor,plan["search_size"],image.shape[:2])
                cell = int(np.argmax(cpu_array(output["score_map"]).reshape(side,side)*np.asarray(plan["window"],np.float32)))
                data["decode_error"][i] = np.max(np.abs(dense[cell]-box))
                previous = box.copy()
                if (i+1) % 500 == 0:
                    print("[retained] {} {}/{}".format(entry["sequence"],i+1,n),flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    annotation = Path(plan["root"]) / entry["annotation"]
    verify_bindings({str(annotation):entry["annotation_sha256"]})
    data["gt"] = read_boxes(annotation)[1:n+1]
    data["iou"] = tracking_iou(data["boxes"],data["gt"])
    audit_trace(data,plan,entry,smoke=limit is not None)
    return data


def models_for_evaluation(rt,plan,root):
    from stage40_crop_experiment import evaluation_models
    # Revalidate all old checkpoints with THEIR original plan, never Stage-44's fingerprint.
    old_models,bindings = evaluation_models(rt,original_plan(plan),Path(plan["source_stage40"]))
    models = {}
    for seed in SEEDS:
        report,path = training_report(root,plan,seed)
        if report["teacher_sha256"] != state_digest(rt.net):
            raise ValueError("Retained training used a different original model")
        bindings[str(path)] = file_hash(path)
        for r in report["models"]:
            matched = matched_record(plan,r["fold"],seed)
            if (r["head_calls"] != matched["head_calls"] or r["block_calls"] != 7*r["head_calls"] or
                    r["matched_six_checkpoint"] != matched["checkpoint"] or r["matched_six_sha256"] != matched["checkpoint_sha256"]):
                raise ValueError("Reused six and retained controls are not matched")
            models[r["fold"],seed] = restore_model(rt,plan,r)
            for path,digest in ((Path(r["checkpoint"]),r["checkpoint_sha256"]),(Path(r["checkpoint"]).with_suffix(".json"),r["receipt_sha256"])):
                bindings[str(path)] = digest
    del old_models
    return models,bindings


def smoke_gate(root,plan):
    report = read_sealed(Path(root) / "smoke/stage44_smoke.json",SCHEMA)
    if report.get("status") != "PASS_MATCHED_STRUCTURE_SMOKE" or report.get("plan_fingerprint") != plan["fingerprint"]:
        raise ValueError("Need unchanged Stage-44 smoke before full training/evaluation")
    verify_bindings(report["file_sha256"])


def smoke(rt,plan,root,image_reader=checked_image):
    report = train(rt,plan,root,SEEDS[0],True,image_reader)
    record = report["models"][0]
    # Smoke evaluation uses a training-fold video, not an OOF selection video.
    entry = next(e for e in plan["fit"] if e["sequence"] == plan["folds"][0]["fit"][0])
    net = restore_model(rt,plan,record,True)
    data = collect_trace(rt,plan,entry,net,DESIGN["smoke_frames"],image_reader)
    initial = make_retained(rt).eval().requires_grad_(False)
    native = collect_trace(rt,plan,entry,initial,DESIGN["smoke_frames"],image_reader)
    reference = reused_trace(plan,entry,SEEDS[0],"retained_initial")
    n = len(native["boxes"])
    error = float(np.max(np.abs(native["boxes"]-reference["boxes"][:n])))
    if error > .001 or float(np.max(np.abs(native["iou"]-reference["iou"][:n]))) > 1e-12:
        raise ValueError("Retained initial full-trajectory replay differs")
    # Direct native output and shared-prefix initialization are also checked during training.
    run = seal(dict(schema=SCHEMA,plan_fingerprint=plan["fingerprint"],mode="smoke",runtime=rt.identity,
        model_states={"0:{}".format(SEEDS[0]):state_digest(net)},file_sha256={record["checkpoint"]:record["checkpoint_sha256"]}))
    directory = Path(root) / "smoke"
    with writer_lock(directory):
        publish_identical(directory / "run.json",run)
        path = trace_path(root,entry,SEEDS[0],True)
        if not path.exists():
            write_cache(path,data,trace_meta(run,entry,SEEDS[0]))
        load_trace(path,plan,run,entry,SEEDS[0],True,True)
        bindings = {str(p):file_hash(p) for p in (directory / "run.json",path,path.with_suffix(".json"),
            directory / "training/stage44_training.json",Path(record["checkpoint"]),Path(record["checkpoint"]).with_suffix(".json"))}
        result = seal(dict(schema=SCHEMA,status="PASS_MATCHED_STRUCTURE_SMOKE",plan_fingerprint=plan["fingerprint"],
            file_sha256=bindings,initial_native_box_error=error,trainable_parameters=record["shared_initial"]["trainable_parameters"],
            warning="Runtime, gradients, native selection and budget only; no compression-gap or method gain established"))
        publish_identical(directory / "stage44_smoke.json",result)
    print("Stage-44: " + result["status"])
    return result


def validate_run(run,plan):
    if (run.get("plan_fingerprint") != plan["fingerprint"] or run.get("runtime") != plan["source_runtime"] or
            run.get("mode") != "full" or run.get("teacher_sha256") != plan["source_teacher_sha256"]):
        raise ValueError("Mixed retained evaluation run")
    expected = {"{}:{}".format(f,s) for f in range(source.DESIGN["folds"]) for s in SEEDS}
    if set(run.get("model_states",{})) != expected:
        raise ValueError("Freeze all eight retained models before evaluation")
    verify_bindings(run["file_sha256"])


def evaluate(rt,plan,root,sequence=None,image_reader=checked_image):
    validate_runtime(rt,plan)
    smoke_gate(root,plan)
    models,bindings = models_for_evaluation(rt,plan,root)
    hashes = {"{}:{}".format(*k):state_digest(v) for k,v in sorted(models.items())}
    run = seal(dict(schema=SCHEMA,plan_fingerprint=plan["fingerprint"],mode="full",runtime=rt.identity,
        model_states=hashes,file_sha256=bindings,teacher_sha256=state_digest(rt.net)))
    validate_run(run,plan)
    selected = [e for e in plan["fit"] if sequence is None or e["sequence"] == sequence]
    if not selected:
        raise ValueError("Sequence must belong to the original eight videos")
    directory = Path(root) / "evaluation"
    with writer_lock(directory):
        if not (directory / "run.json").exists() and any(directory.rglob("*.npz")):
            raise ValueError("Unidentified retained traces")
        publish_identical(directory / "run.json",run)
        for entry in selected:
            for seed in SEEDS:
                path = trace_path(root,entry,seed)
                if not path.exists():
                    if path.with_suffix(".json").exists():
                        raise ValueError("Committed retained trace missing")
                    data = collect_trace(rt,plan,entry,models[fold_for(plan,entry),seed],image_reader=image_reader)
                    write_cache(path,data,trace_meta(run,entry,seed))
                load_trace(path,plan,run,entry,seed,allow_orphan=True)
                print("[complete, audited] " + path.name,flush=True)
        if hashes != {"{}:{}".format(*k):state_digest(v) for k,v in sorted(models.items())} or state_digest(rt.net) != run["teacher_sha256"]:
            raise ValueError("Frozen retained model changed during inference")
        verify_bindings(plan["source_file_sha256"])
    return run


def status(root,plan):
    for seed in SEEDS:
        for fold in range(source.DESIGN["folds"]):
            directory = Path(root) / "training/seed{}/fold{}".format(seed,fold)
            receipts = sorted(directory.glob("step*.json"))
            latest = 0
            for path in receipts:
                receipt = read_sealed(path,SCHEMA)
                verify_bindings({str(path.with_suffix(".pt")):receipt["checkpoint_sha256"]})
                if receipt["metadata"]["plan_fingerprint"] != plan["fingerprint"]:
                    raise ValueError("Mixed training status")
                latest = receipt["step"]
            print("seed{} fold{}: {}/{} committed updates".format(seed,fold,latest,source.DESIGN["updates"]))
    complete = 0
    if (Path(root) / "evaluation/run.json").exists():
        run = read_sealed(Path(root) / "evaluation/run.json",SCHEMA)
        validate_run(run,plan)
        for entry in plan["fit"]:
            for seed in SEEDS:
                path = trace_path(root,entry,seed)
                if path.with_suffix(".json").exists():
                    load_trace(path,plan,run,entry,seed)
                    complete += 1
    print("evaluation: {}/16 committed full videos".format(complete))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=nonempty_path,required=True,help="Stage-43 root for prepare; Stage-44 otherwise")
    parser.add_argument("--mode",choices=("prepare","smoke","train","evaluate","status"),required=True)
    parser.add_argument("--output_dir",type=nonempty_path)
    parser.add_argument("--checkpoint",type=nonempty_path)
    parser.add_argument("--parameter",default="deit_distilled")
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--seed",type=int,choices=SEEDS,default=SEEDS[0])
    parser.add_argument("--sequence")
    args = parser.parse_args()
    root = Path(args.input).expanduser().resolve()
    if not (root / "manifest.json").is_file():
        parser.error("--input must contain manifest.json; prepare reads the complete Stage-43 root, not analysis/")
    if args.mode == "prepare":
        if not args.output_dir:
            parser.error("prepare requires --output_dir")
        with writer_lock(Path(args.output_dir).expanduser().resolve()):
            prepare(root,args.output_dir)
        return
    plan = load_plan(root)
    if args.mode == "status":
        status(root,plan)
        return
    checkpoint = args.checkpoint or os.environ.get("CHECKPOINT") or None
    if checkpoint and file_hash(Path(checkpoint).expanduser()) != plan["runtime"]["checkpoint_sha256"]:
        parser.error("Use the original SGLA checkpoint")
    import torch
    from stage4_runtime import Runtime
    seed_runtime(torch,plan["runtime_seed"])
    rt = Runtime(args.parameter,args.gpu,checkpoint)
    if args.mode == "smoke":
        smoke(rt,plan,root)
    elif args.mode == "train":
        train(rt,plan,root,args.seed)
    else:
        evaluate(rt,plan,root,args.sequence)


if __name__ == "__main__":
    main()
